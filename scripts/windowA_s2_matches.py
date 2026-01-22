#!/usr/bin/env python3
"""
windowA_s2_matches.py

Find Sentinel-2 matches for "window A" (seasonal pair) for each Planet chip
using seasonal pairing rules, multi-tier cloud thresholds and temporal proximity.

Outputs:
  /home/airg/gessuman/projects/ftw-mappingafrica-integration/data/windowA/windowA_s2_matches.csv
"""

from pathlib import Path
import logging
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm

from pystac_client import Client
from pystac_client.exceptions import APIError
import planetary_computer as pc


# ---------------- CONFIG ----------------

BASE = Path("/home/airg/gessuman/projects/ftw-mappingafrica-integration")

# CSV describing the Planet "window B" chip catalog we want to match for window A.
# Must have columns: window_b (filepath), country, planet_date, x, y
PLANET_CSV = BASE / "data" / "mappingafrica-catalog-dated.csv"

OUTDIR = BASE / "data" / "windowA"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUTDIR / "windowA_s2_matches.csv"
LOGFILE = OUTDIR / "windowA_matches.log"

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_COLLECTION = "sentinel-2-l2a"

# Cloud tiers (pct cover). We will try in this order:
CLOUD_THRESHOLDS = [5, 10, 15, 20, 30, 40]

MAX_ITEMS_PER_QUERY = 100  # safeguard
MAX_RETRIES = 4
BACKOFF_BASE = 1.7


# ---------------- LOGGING ----------------
logger = logging.getLogger("windowA_s2")
logger.setLevel(logging.INFO)
logger.handlers.clear()
fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
sh = logging.StreamHandler()
sh.setFormatter(fmt)
logger.addHandler(sh)
fh = logging.FileHandler(LOGFILE)
fh.setFormatter(fmt)
logger.addHandler(fh)


# ---------------- SEASONAL RULES ----------------
def in_month_range(month, start, end):
    """Inclusive, handles wrap (e.g., 11..2 covers Nov–Feb)."""
    if start <= end:
        return start <= month <= end
    return (month >= start) or (month <= end)

REGION_MAP = {
    "sudano_sahel":   ["GH", "NG", "ML", "NE", "SD", "TD"],
    "coastal_forest": ["CI", "TG", "BJ"],  # (south GH/NG split can be added later)
    "bimodal":        ["UG", "KE", "RW", "BI", "TZ"],
    "ethiopia":       ["ET"],
    "southern":       ["ZM", "ZW", "MZ", "ZA", "MW", "BW", "NA"],
}

def which_region(country_code):
    cc = (country_code or "").upper()
    for region_name, code_list in REGION_MAP.items():
        if cc in code_list:
            return region_name
    return None  # unknown

def pick_target_months(region_name, planet_month):
    """Return (target_months, label) for the opposite seasonal window."""
    m = planet_month

    if region_name == "sudano_sahel":
        if in_month_range(m, 8, 9):   # Aug–Sep -> Dec–Feb
            return ([12, 1, 2], "Dec–Feb")
        if in_month_range(m, 2, 3):   # Feb–Mar -> Aug–Oct
            return ([8, 9, 10], "Aug–Oct")
        return ([], "no_rule")

    if region_name == "coastal_forest":
        if in_month_range(m, 3, 4):   # Mar–Apr -> Sep–Nov
            return ([9, 10, 11], "Sep–Nov")
        if in_month_range(m, 10, 11): # Oct–Nov -> Feb–Mar
            return ([2, 3], "Feb–Mar")
        return ([], "no_rule")

    if region_name == "bimodal":
        if in_month_range(m, 3, 5):   # Mar–May -> Jun–Jul (choose this deterministically)
            return ([6, 7], "Jun–Jul")
        if in_month_range(m, 9, 11):  # Sep–Nov -> Jan–Feb
            return ([1, 2], "Jan–Feb")
        return ([], "no_rule")

    if region_name == "ethiopia":
        if in_month_range(m, 8, 9):   # Aug–Sep -> Dec–Feb
            return ([12, 1, 2], "Dec–Feb")
        if in_month_range(m, 1, 3):   # Jan–Mar -> Aug–Sep
            return ([8, 9], "Aug–Sep")
        return ([], "no_rule")

    if region_name == "southern":
        if in_month_range(m, 1, 2):   # Jan–Feb -> Jul–Aug
            return ([7, 8], "Jul–Aug")
        if m == 8:                     # Aug -> Jan–Feb
            return ([1, 2], "Jan–Feb")
        return ([], "no_rule")

    return ([], "no_region_rule")

def build_date_ranges(planet_date, target_months):
    """
    Assign target months to nearest plausible year relative to planet_date,
    merge into contiguous date ranges, return [(YYYY-MM-DD, YYYY-MM-DD), ...].
    """
    pm = planet_date.month
    py = planet_date.year

    assigned = []
    for tm in target_months:
        forward = (tm - pm) % 12
        backward = (pm - tm) % 12
        if forward <= backward:
            year = py if tm >= pm else py + 1
        else:
            year = py if tm <= pm else py - 1

        start = datetime(year, tm, 1)
        end = (datetime(year + (tm == 12), (tm % 12) + 1, 1) - timedelta(seconds=1))
        assigned.append((start, end))

    assigned.sort(key=lambda x: x[0])
    merged = []
    for st, ed in assigned:
        if not merged or st > merged[-1][1] + timedelta(days=1):
            merged.append([st, ed])
        else:
            if ed > merged[-1][1]:
                merged[-1][1] = ed

    return [(a[0].strftime("%Y-%m-%d"), a[1].strftime("%Y-%m-%d")) for a in merged]


# ---------------- STAC QUERY HELPERS ----------------
def query_s2_items(client, lon, lat, start_date, end_date, cloud_max, max_items=MAX_ITEMS_PER_QUERY):
    """Query S2 L2A items intersecting a POINT in a date range with cloud ≤ cloud_max."""
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
            return [{
                "s2_id": it.id,
                "s2_datetime": pd.to_datetime(it.datetime),
                "s2_cloud": it.properties.get("eo:cloud_cover", np.nan),
            } for it in items]
        except APIError as e:
            attempt += 1
            if attempt >= MAX_RETRIES:
                raise
            sleep_s = BACKOFF_BASE ** attempt
            logger.info(f"STAC timeout, retry {attempt}/{MAX_RETRIES-1} in {sleep_s:.1f}s … ({e})")
            time.sleep(sleep_s)

def pick_best_item(candidates, planet_dt):
    """Pick closest-in-time, then lowest cloud, then earliest timestamp."""
    if not candidates:
        return None
    df = pd.DataFrame(candidates).dropna(subset=["s2_datetime", "s2_cloud"])
    if df.empty:
        return None
    df["abs_days"] = (df["s2_datetime"] - planet_dt).abs().dt.total_seconds() / 86400.0
    df = df.sort_values(["abs_days", "s2_cloud", "s2_datetime"], ascending=[True, True, True])
    return df.iloc[0].to_dict()


# ---------------- MAIN ----------------
def main():
    logger.info("Loading Planet chip CSV…")
    df = pd.read_csv(PLANET_CSV)

    # Expect: window_b (filepath), country, planet_date, x, y
    need_cols = {"window_b", "country", "planet_date", "x", "y"}
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns {missing}")

    # Parse and clean
    df["planet_date"] = pd.to_datetime(df["planet_date"], errors="coerce", utc=True)
    df = df.dropna(subset=["planet_date", "x", "y", "country", "window_b"]).copy()

    # Derive chip_id from the window_b filename (without extension)
    df["chip_id"] = df["window_b"].apply(lambda p: Path(str(p)).stem)
    df["country"] = df["country"].astype(str).str.upper()
    df["x"] = df["x"].astype(float)
    df["y"] = df["y"].astype(float)

    logger.info(f"Total chips to process: {len(df)}")

    client = Client.open(STAC_URL)

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Finding windowA matches", unit="chip"):
        chip_id = str(row["chip_id"])
        country = str(row["country"]).upper()
        planet_dt = row["planet_date"]
        lon = float(row["x"])
        lat = float(row["y"])

        region = which_region(country)
        if region is None:
            results.append({
                "chip_id": chip_id,
                "country": country,
                "planet_date": planet_dt,
                "season_label": "no_region",
                "tier_used": None,
                "cloud_limit": None,
                "s2_id": None,
                "s2_datetime": None,
                "s2_cloud": None,
                "abs_days": None,
                "status": "no_region_rule",
            })
            continue

        target_months, season_label = pick_target_months(region, planet_dt.month)
        if not target_months:
            results.append({
                "chip_id": chip_id,
                "country": country,
                "planet_date": planet_dt,
                "season_label": season_label,
                "tier_used": None,
                "cloud_limit": None,
                "s2_id": None,
                "s2_datetime": None,
                "s2_cloud": None,
                "abs_days": None,
                "status": "no_pair_for_month",
            })
            continue

        ranges = build_date_ranges(
            planet_dt.to_pydatetime() if hasattr(planet_dt, "to_pydatetime") else planet_dt,
            target_months,
        )

        match_row = None
        tier_used = None
        cloud_limit = None

        for t_idx, cloud_max in enumerate(CLOUD_THRESHOLDS, start=1):
            tier_label = f"tier{t_idx}_le{cloud_max}"
            candidates = []
            for (start_date, end_date) in ranges:
                try:
                    candidates.extend(query_s2_items(client, lon, lat, start_date, end_date, cloud_max))
                except Exception as e:
                    logger.info(f"{chip_id}: STAC error in {tier_label} for {start_date}..{end_date}: {e}")
            best = pick_best_item(candidates, planet_dt)
            if best is not None:
                match_row = best
                tier_used = tier_label
                cloud_limit = cloud_max
                break  # found at this tier

        if match_row is None:
            results.append({
                "chip_id": chip_id,
                "country": country,
                "planet_date": planet_dt,
                "season_label": season_label,
                "tier_used": None,
                "cloud_limit": None,
                "s2_id": None,
                "s2_datetime": None,
                "s2_cloud": None,
                "abs_days": None,
                "status": "no_match_all_tiers",
            })
        else:
            results.append({
                "chip_id": chip_id,
                "country": country,
                "planet_date": planet_dt,
                "season_label": season_label,
                "tier_used": tier_used,
                "cloud_limit": cloud_limit,
                "s2_id": match_row["s2_id"],
                "s2_datetime": match_row["s2_datetime"],
                "s2_cloud": float(match_row["s2_cloud"]),
                "abs_days": float(match_row["abs_days"]),
                "status": "ok",
            })

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUT_CSV, index=False)
    logger.info(f"Saved matches → {OUT_CSV}")


if __name__ == "__main__":
    main()
