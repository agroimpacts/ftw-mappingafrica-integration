#!/usr/bin/env python3
"""
windowA_rescue_2725_southernNA.py

Rescue Sentinel-2 Window A matches for the 2,725 problematic chips:
- 2,543 rows from windowA_s2_matches_scored.csv with status != "ok"
- 182 rows that never appeared in the scored CSV

Steps:
1) Load rescue CSV (2725 rows).
2) Merge with dated catalog to ensure we have name, x, y.
3) Derive country from chip_id prefix if missing (first 2 letters, e.g., 'NA').
4) Map country -> region using REGION_MAP (Namibia -> 'southern').
5) Use REGION_RULES (including 'southern') + extended logic to get target months.
6) Build seasonal windows enforcing WinA < WinB.
7) Query Sentinel-2 via Planetary Computer STAC.
8) Pick best candidate (score = seasonal alignment + time diff + cloud).
9) Update rescue CSV with new s2_id, s2_datetime, s2_cloud, abs_days, score, status.

Output:
  data/windowA/windowA_rescue_matches_2725.csv
"""

from pathlib import Path
import logging
import time
from datetime import datetime, timedelta
from calendar import monthrange

import numpy as np
import pandas as pd
from tqdm import tqdm

from pystac_client import Client
from pystac_client.exceptions import APIError
import planetary_computer as pc

# ---------------- CONFIG ----------------

BASE = Path("/home/airg/gessuman/projects/ftw-mappingafrica-integration")

RESCUE_CSV = BASE / "data" / "windowA_rescue_input_2725.csv"
DATED_CSV  = BASE / "data" / "mappingafrica-catalog-dated.csv"

OUTDIR = BASE / "data" / "windowA"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUTDIR / "windowA_rescue_matches_2725.csv"
LOGFILE = OUTDIR / "windowA_rescue_2725.log"

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_COLLECTION = "sentinel-2-l2a"

CLOUD_THRESHOLDS = [5, 10, 15, 20, 30, 40]
MAX_ITEMS_PER_QUERY = 100
MAX_RETRIES = 4
BACKOFF_BASE = 1.7

WINDOW_DAYS = 60.0  # +/- 30 days around seasonal month center


# ---------------- LOGGING ----------------

logger = logging.getLogger("windowA_rescue_2725")
logger.setLevel(logging.INFO)
logger.handlers.clear()
_fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
_sh = logging.StreamHandler()
_sh.setFormatter(_fmt)
logger.addHandler(_sh)
_fh = logging.FileHandler(LOGFILE)
_fh.setFormatter(_fmt)
logger.addHandler(_fh)


# ---------------- HELPERS ----------------

def ensure_naive(dt):
    """Return a timezone-naive Python datetime from a pandas Timestamp or datetime."""
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    if hasattr(dt, "tzinfo") and dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    return dt


# ---------------- REGIONS & RULES ----------------

REGION_MAP = {
    # Region 1: West Africa – Sudano-Sahel
    "sudano_sahel": [
        "GH", "NG", "ML", "NE", "SD", "TD",
        "BF", "MR", "SN",
    ],

    # Region 2: West Africa – Coastal / Forest
    "coastal_forest": [
        "CI", "TG", "BJ",
        "GM", "GN", "GW", "SL", "GA", "CM", "CG", "CD",
    ],

    # Region 3: East Africa – Equatorial Bimodal
    "bimodal": [
        "UG", "KE", "RW", "BI", "TZ",
        "CF", "SO",
    ],

    # Region 4: East Africa – Highlands / Ethiopia
    "ethiopia": [
        "ET", "ER",
    ],

    # Region 5: Southern Africa – Summer rainfall (incl. Namibia "NA")
    "southern": [
        "ZM", "ZW", "MZ", "ZA", "MW", "BW", "NA",
        "AO", "LS", "SZ",
    ],
}

REGION_RULES = {
    # Sudano-Sahel
    "sudano_sahel": [
        {"months": [8, 9], "target": [12, 1, 2], "label": "Aug–Sep→Dec–Feb"},
        {"months": [2, 3], "target": [8, 9, 10], "label": "Feb–Mar→Aug–Oct"},
    ],
    # Coastal / Forest
    "coastal_forest": [
        {"months": [3, 4], "target": [9, 10, 11], "label": "Mar–Apr→Sep–Nov"},
        {"months": [10, 11], "target": [2, 3], "label": "Oct–Nov→Feb–Mar"},
    ],
    # Equatorial Bimodal
    "bimodal": [
        {"months": [3, 4, 5], "target": [6, 7], "label": "Mar–May→Jun–Jul"},
        {"months": [9, 10, 11], "target": [1, 2], "label": "Sep–Nov→Jan–Feb"},
    ],
    # Ethiopia / Highlands
    "ethiopia": [
        {"months": [8, 9], "target": [12, 1, 2], "label": "Aug–Sep→Dec–Feb"},
        {"months": [1, 2, 3], "target": [8, 9], "label": "Jan–Mar→Aug–Sep"},
    ],
    # Southern (incl. Namibia NA)
    "southern": [
        {"months": [1, 2], "target": [7, 8], "label": "Jan–Feb→Jul–Aug"},
        {"months": [8],    "target": [1, 2], "label": "Aug→Jan–Feb"},
    ],
}

def which_region(country_code: str):
    cc = str(country_code).upper()
    for reg, codes in REGION_MAP.items():
        if cc in codes:
            return reg
    return None

def pick_target_months(region_name: str, planet_month: int):
    """
    Extended version:
    - Use exact rule month if available.
    - Otherwise attach this month to the nearest rule in circular month space.
    """
    rules = REGION_RULES.get(region_name)
    if not rules:
        return ([], "no_region_rule")

    m = int(planet_month)

    # 1) Exact match if possible
    for r in rules:
        if m in r["months"]:
            return (r["target"], r["label"])

    # 2) Nearest rule
    def month_distance(a: int, b: int) -> int:
        d = abs(a - b)
        return min(d, 12 - d)

    best_rule = None
    best_dist = 999
    for r in rules:
        d = min(month_distance(m, mm) for mm in r["months"])
        if d < best_dist:
            best_dist = d
            best_rule = r

    if best_rule is not None:
        return (best_rule["target"], best_rule["label"] + "_extended")

    return ([], "no_rule")

def seasonal_target_windows(planet_dt: datetime, target_months):
    """
    Build seasonal windows for S2 search, enforcing WinA < WinB.

    Logic:
      - If target month < planet month: same year.
      - Else: previous year.
    Then build a +/- WINDOW_DAYS/2 window around month center,
    and clamp so that end < planet_dt.
    """
    pm = planet_dt.month
    py = planet_dt.year
    half = WINDOW_DAYS / 2.0

    windows = []

    for tm in target_months:
        tm = int(tm)

        # Year choice to enforce A < B
        if tm < pm:
            year = py
        else:
            year = py - 1

        days_in_month = monthrange(year, tm)[1]
        month_start = datetime(year, tm, 1)
        month_end   = datetime(year, tm, days_in_month, 23, 59, 59)
        month_center = month_start + (month_end - month_start) / 2.0

        w_start = month_center - timedelta(days=half)
        w_end   = month_center + timedelta(days=half)

        # Enforce WinA < WinB
        if w_end >= planet_dt:
            w_end = planet_dt - timedelta(seconds=1)
            w_start = w_end - timedelta(days=WINDOW_DAYS)

        windows.append(
            {
                "start": ensure_naive(w_start),
                "end": ensure_naive(w_end),
                "center": ensure_naive(month_center),
            }
        )

    return windows


# ---------------- STAC & SCORING ----------------

def query_s2_items(client, lon, lat, start_dt, end_dt,
                   cloud_max, max_items=MAX_ITEMS_PER_QUERY):
    start_iso = start_dt.strftime("%Y-%m-%d")
    end_iso   = end_dt.strftime("%Y-%m-%d")

    params = dict(
        collections=[S2_COLLECTION],
        intersects={"type": "Point", "coordinates": [lon, lat]},
        datetime=f"{start_iso}/{end_iso}",
        query={"eo:cloud_cover": {"lte": cloud_max}},
        limit=max_items,
        max_items=max_items,
    )

    attempt = 0
    while True:
        try:
            search = client.search(**params)
            items = list(search.items())
            rows = []
            for it in items:
                dt = ensure_naive(pd.to_datetime(it.datetime))
                rows.append(
                    {
                        "s2_id": it.id,
                        "s2_datetime": dt,
                        "s2_cloud": float(it.properties.get("eo:cloud_cover", np.nan)),
                    }
                )
            return rows
        except APIError as e:
            attempt += 1
            if attempt >= MAX_RETRIES:
                logger.warning(
                    f"STAC error for lon={lon}, lat={lat}, {start_iso}..{end_iso}: {e}"
                )
                return []
            wait = BACKOFF_BASE ** attempt
            logger.info(
                f"STAC timeout, retry {attempt}/{MAX_RETRIES-1} in {wait:.1f}s … ({e})"
            )
            time.sleep(wait)

def score_candidate(row, target, planet_dt):
    s2_dt = ensure_naive(row["s2_datetime"])
    center = ensure_naive(target["center"])
    planet_dt = ensure_naive(planet_dt)

    days_center = abs((s2_dt - center).total_seconds()) / 86400.0
    days_planet = abs((s2_dt - planet_dt).total_seconds()) / 86400.0
    cloud = row["s2_cloud"] if np.isfinite(row["s2_cloud"]) else 100.0

    return days_center + 0.5 * days_planet + 0.05 * cloud

def pick_best_scored(candidates, planet_dt):
    if not candidates:
        return None
    best = None
    best_score = np.inf
    for c in candidates:
        row = c["row"]
        target = c["target"]
        sc = score_candidate(row, target, planet_dt)
        if sc < best_score:
            best_score = sc
            best = (row, target, sc)
    return best


# ---------------- MAIN ----------------

def main():
    logger.info("Loading rescue CSV (2725 rows)…")
    df = pd.read_csv(RESCUE_CSV)
    df["chip_id"] = df["chip_id"].astype(str)

    logger.info("Loading dated catalog for coordinates & name…")
    cat = pd.read_csv(DATED_CSV)
    cat["planet_date"] = pd.to_datetime(cat["planet_date"], errors="coerce")
    cat["chip_id"] = (
        cat["name"].astype(str)
        + "_"
        + cat["planet_date"].dt.strftime("%Y-%m")
    )

    cat_sub = cat[["chip_id", "name", "country", "x", "y", "planet_date"]].drop_duplicates("chip_id")

    # Drop potentially incomplete cols in rescue and refill from catalog
    df = df.drop(columns=["name", "country", "x", "y"], errors="ignore").merge(
        cat_sub, on="chip_id", how="left"
    )

    # Use planet_date from catalog if missing in rescue
    if "planet_date" in df.columns:
        df["planet_date"] = pd.to_datetime(df["planet_date"], errors="coerce")
        df["planet_date"] = df["planet_date"].fillna(df["planet_date_y"] if "planet_date_y" in df.columns else df["planet_date"])
    else:
        df["planet_date"] = df["planet_date_y"]

    df["planet_date"] = pd.to_datetime(df["planet_date"], errors="coerce")

    # Derive country from chip_id if missing
    df["country_from_id"] = df["chip_id"].str.extract(r"^([A-Za-z]{2})")
    if "country" not in df.columns:
        df["country"] = df["country_from_id"]
    else:
        df["country"] = df["country"].fillna(df["country_from_id"])

    df["x"] = df["x"].astype(float)
    df["y"] = df["y"].astype(float)

    logger.info(f"After merging, rows: {len(df)}")
    logger.info("Missing counts in key fields:")
    logger.info(df[["country", "x", "y", "planet_date"]].isna().sum().to_string())

    client = Client.open(STAC_URL)

    updated_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Rescuing 2725", unit="chip"):
        chip_id = str(row["chip_id"])
        planet_dt = ensure_naive(row["planet_date"])
        lon = float(row["x"])
        lat = float(row["y"])
        country = str(row["country"]).upper()

        base = {
            "chip_id": chip_id,
            "country": country,
            "planet_date": planet_dt,
            "s2_id": row.get("s2_id", None),
            "s2_datetime": row.get("s2_datetime", None),
            "s2_cloud": row.get("s2_cloud", None),
            "abs_days": row.get("abs_days", None),
            "score": row.get("score", None),
            "season_label": row.get("season_label", None),
            "tier_used": row.get("tier_used", None),
            "cloud_limit": row.get("cloud_limit", None),
            "status": row.get("status", None),
        }

        if pd.isna(planet_dt) or np.isnan(lon) or np.isnan(lat):
            base["status"] = "no_coords_or_date_rescue"
            updated_rows.append(base)
            continue

        region = which_region(country)
        if region is None:
            base["season_label"] = "no_region"
            base["status"] = "no_region_rule_rescue"
            updated_rows.append(base)
            continue

        target_months, season_label = pick_target_months(region, planet_dt.month)
        if not target_months:
            base["season_label"] = season_label
            base["status"] = "no_pair_for_month_rescue"
            updated_rows.append(base)
            continue

        targets = seasonal_target_windows(planet_dt, target_months)

        match_row = None
        match_score = None
        tier_used = None
        cloud_limit = None

        for t_idx, cloud_max in enumerate(CLOUD_THRESHOLDS, start=1):
            tier_label = f"tier{t_idx}_le{cloud_max}"
            tier_candidates = []

            for t in targets:
                rows_s2 = query_s2_items(
                    client, lon, lat, t["start"], t["end"], cloud_max
                )
                for r in rows_s2:
                    tier_candidates.append({"row": r, "target": t})

            best = pick_best_scored(tier_candidates, planet_dt)
            if best is not None:
                r, tg, sc = best
                match_row = r
                match_score = sc
                tier_used = tier_label
                cloud_limit = cloud_max
                break

        if match_row is None:
            base["season_label"] = season_label
            base["tier_used"] = None
            base["cloud_limit"] = None
            base["s2_id"] = None
            base["s2_datetime"] = None
            base["s2_cloud"] = None
            base["abs_days"] = None
            base["score"] = None
            base["status"] = "no_match_all_tiers_rescue"
        else:
            s2_dt = ensure_naive(match_row["s2_datetime"])
            abs_days = abs((s2_dt - planet_dt).total_seconds()) / 86400.0

            base["season_label"] = season_label
            base["tier_used"] = tier_used
            base["cloud_limit"] = cloud_limit
            base["s2_id"] = match_row["s2_id"]
            base["s2_datetime"] = s2_dt
            base["s2_cloud"] = float(match_row["s2_cloud"])
            base["abs_days"] = abs_days
            base["score"] = match_score
            base["status"] = "ok_rescue"

        updated_rows.append(base)
        time.sleep(0.15)

    out_df = pd.DataFrame(updated_rows)
    out_df.to_csv(OUT_CSV, index=False)
    logger.info(f"Wrote {len(out_df)} rows to {OUT_CSV}")
    logger.info("Rescue status breakdown:")
    logger.info(out_df["status"].value_counts().to_string())
    logger.info("Done.")


if __name__ == "__main__":
    main()
