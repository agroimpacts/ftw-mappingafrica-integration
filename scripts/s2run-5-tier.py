#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# PRIMARY-ONLY DRY RUN — 5-tier cloud fallback (±30d window)
# Tiers: ≤5% → ≤10% → ≤20% → ≤30% → ≤40%
# Ready for terminal/screen runs with live tqdm progress.

from pathlib import Path
import logging, random, time
import pandas as pd
from tqdm import tqdm
from pystac_client import Client
from pystac_client.exceptions import APIError

# --- Headless plotting setup ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
CSV_IN = Path("/home/airg/gessuman/projects/ftw-mappingafrica-integration/data/mappingafrica-catalog-dated.csv")
OUTDIR = CSV_IN.parent / "dry_run_primary_5tier"
OUTDIR.mkdir(parents=True, exist_ok=True)

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

PRIMARY_PAD_DAYS   = 30       # ± days around Planet date
# Added tiers 4 & 5 per advisor (plan to cloud-mask later)
CLOUD_TIERS = [
    (1, 5.0,  "strict"),
    (2, 10.0, "low"),
    (3, 20.0, "moderate"),
    (4, 30.0, "loose"),
    (5, 40.0, "very_loose"),
]
S2_MAX_ITEMS       = 30
RATE_DELAY_SEC     = 0.25
RETRIES            = 4
BACKOFF_BASE       = 1.5
RANDOM_SEED        = 42
SAMPLE_N           = None     # e.g., 1000 for quick test; None = full dataset

random.seed(RANDOM_SEED)

# ------------- Logging -------------
logger = logging.getLogger("primary_5tier")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(h)

# ------------- STAC client -------------
client = Client.open(STAC_URL)

# ------------- tqdm setup for terminal -------------
tqdm.pandas(dynamic_ncols=True)

# ---------------- Functions ----------------
def query_s2_point(lon: float, lat: float, start_date: str, end_date: str,
                   cloud_max: float, max_items=S2_MAX_ITEMS) -> pd.DataFrame:
    """
    Query Sentinel-2 L2A items intersecting a POINT (lon,lat) in [start,end],
    with server-side filter on eo:cloud_cover <= cloud_max. Retries on APIError.
    Returns DataFrame columns: s2_id, s2_datetime, s2_cloud.
    """
    params = dict(
        collections=["sentinel-2-l2a"],
        intersects={"type": "Point", "coordinates": [lon, lat]},
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lte": cloud_max}},
        limit=max_items,
        max_items=max_items,
    )
    attempt = 0
    while True:
        try:
            search = client.search(**params)
            items = list(search.items())
            if not items:
                return pd.DataFrame(columns=["s2_id", "s2_datetime", "s2_cloud"])
            rows = [{
                "s2_id": it.id,
                "s2_datetime": pd.to_datetime(it.datetime, utc=True),
                "s2_cloud": it.properties.get("eo:cloud_cover", None),
            } for it in items]
            time.sleep(RATE_DELAY_SEC)
            return pd.DataFrame(rows)
        except APIError:
            attempt += 1
            if attempt >= RETRIES:
                raise
            sleep_s = BACKOFF_BASE ** attempt
            logger.info(f"STAC timeout, retry {attempt}/{RETRIES-1} in {sleep_s:.1f}s …")
            time.sleep(sleep_s)

def pick_best(items: pd.DataFrame, planet_date: pd.Timestamp):
    """Pick closest-in-time; then lower cloud; then earliest timestamp."""
    if items is None or items.empty:
        return None
    need = {"s2_cloud", "s2_datetime"}
    if not need.issubset(items.columns):
        return None
    items = items.dropna(subset=["s2_cloud", "s2_datetime"]).copy()
    if items.empty:
        return None
    items["s2_cloud"] = pd.to_numeric(items["s2_cloud"], errors="coerce")
    items["s2_datetime"] = pd.to_datetime(items["s2_datetime"], errors="coerce", utc=True)
    items = items.dropna(subset=["s2_cloud", "s2_datetime"])
    if items.empty:
        return None
    items["abs_days"] = (items["s2_datetime"] - planet_date).abs().dt.days
    return items.sort_values(["abs_days", "s2_cloud", "s2_datetime"]).iloc[0]

def save_hist(series, title, outpath, xlabel="Value", bins=20):
    """Save a histogram to disk (headless)."""
    if series is None or len(series) == 0:
        return
    plt.figure()
    series.plot(kind="hist", bins=bins)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(Path(outpath), dpi=300)
    plt.close()

# ------------- Load catalog -------------
df = pd.read_csv(CSV_IN)

# If CSV contains mixed datasets, keep only MappingAfrica
if "dataset" in df.columns:
    df = df[df["dataset"].astype(str).str.lower() == "mappingafrica"].copy()

# Required fields (no hard requirement for 'country')
df["planet_date"] = pd.to_datetime(df["planet_date"], errors="coerce", utc=True)
df = df.dropna(subset=["planet_date", "x", "y"]).copy()

# Country only for logs
df["country"] = df.get("country", "").astype(str).str.upper()
df["country"] = df["country"].where(df["country"].str.len() > 0, "UNK")

logger.info(f"Rows available (mappingafrica, valid coords & date): {len(df)}")
use_df = df.sample(SAMPLE_N, random_state=RANDOM_SEED) if SAMPLE_N else df
logger.info(f"Processing rows: {len(use_df)}")

# ------------- Outputs -------------
tier_rows = {1: [], 2: [], 3: [], 4: [], 5: []}
failed_rows = []

for _, row in tqdm(use_df.iterrows(), total=len(use_df), ncols=100,
                   desc="Processing chips", unit="row"):
    chip_id  = row.get("name")
    country  = row.get("country", "UNK")
    lon, lat = float(row["x"]), float(row["y"])
    pdate    = row["planet_date"]

    p_start = (pdate - pd.Timedelta(days=PRIMARY_PAD_DAYS)).strftime("%Y-%m-%d")
    p_end   = (pdate + pd.Timedelta(days=PRIMARY_PAD_DAYS)).strftime("%Y-%m-%d")

    accepted = None
    last_reason = None

    for tier_num, cloud_max, tier_label in CLOUD_TIERS:
        try:
            s2 = query_s2_point(lon, lat, p_start, p_end, cloud_max=cloud_max)
        except APIError:
            last_reason = f"STAC timeout after retries (tier {tier_num}, ≤{cloud_max:.0f}%)"
            continue

        best = pick_best(s2, pdate)
        if best is not None:
            accepted = {
                "selection_tier": tier_label,   # 'strict'|'low'|'moderate'|'loose'|'very_loose'
                "tier_num": tier_num,
                "chip_id": chip_id,
                "country": country,
                "planet_date": pdate,
                "s2_primary_id": best["s2_id"],
                "s2_primary_date": best["s2_datetime"],
                "s2_primary_cloud": float(best["s2_cloud"]),
                "abs_days": int(best["abs_days"]),
                "window_days": PRIMARY_PAD_DAYS,
                "cloud_max": cloud_max,
                "lon": lon, "lat": lat,
            }
            break
        else:
            last_reason = f"No S2 items with valid cloud/date at tier {tier_num} (≤{cloud_max:.0f}%)."

    if accepted is None:
        failed_rows.append({
            "chip_id": chip_id,
            "country": country,
            "planet_date": pdate,
            "reason": last_reason if last_reason else
                      f"No S2 items ≤5/10/20/30/40% in ±{PRIMARY_PAD_DAYS} days",
        })
    else:
        tier_rows[accepted["tier_num"]].append(accepted)

# ------------- Save CSVs -------------
dfs = {k: pd.DataFrame(v) for k, v in tier_rows.items()}
failed_df = pd.DataFrame(failed_rows)

paths = {
    1: OUTDIR / "primary_accepted_tier1_cloudle_5.csv",
    2: OUTDIR / "primary_accepted_tier2_cloudle_10.csv",
    3: OUTDIR / "primary_accepted_tier3_cloudle_20.csv",
    4: OUTDIR / "primary_accepted_tier4_cloudle_30.csv",
    5: OUTDIR / "primary_accepted_tier5_cloudle_40.csv",
    "fail": OUTDIR / "primary_failed_no_match.csv",
}

for tier in [1,2,3,4,5]:
    dfs[tier].to_csv(paths[tier], index=False)
failed_df.to_csv(paths["fail"], index=False)

logger.info(f"Accepted tier1 (≤5%):   {len(dfs[1])}")
logger.info(f"Accepted tier2 (≤10%):  {len(dfs[2])}")
logger.info(f"Accepted tier3 (≤20%):  {len(dfs[3])}")
logger.info(f"Accepted tier4 (≤30%):  {len(dfs[4])}")
logger.info(f"Accepted tier5 (≤40%):  {len(dfs[5])}")
logger.info(f"Failed (no match):      {len(failed_df)}")
for tier in [1,2,3,4,5]:
    logger.info(f"Saved → {paths[tier]}")
logger.info(f"Saved → {paths['fail']}")

# ------------- Save figures (headless) -------------
for tier in [1,2,3,4,5]:
    df_t = dfs[tier]
    if df_t.empty:
        continue
    save_hist(
        df_t["s2_primary_cloud"].dropna(),
        f"Tier {tier} cloud",
        OUTDIR / f"hist_tier{tier}_cloud.png",
        xlabel="Cloud cover (%)",
    )
    save_hist(
        df_t["abs_days"].dropna(),
        f"Tier {tier} |Δdays|",
        OUTDIR / f"hist_tier{tier}_absdays.png",
        xlabel="|Δdays| to Planet date",
    )

