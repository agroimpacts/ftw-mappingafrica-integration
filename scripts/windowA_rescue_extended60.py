#!/usr/bin/env python3
"""
windowA_rescue_extended60.py

Rescue script for window A S2 matches for chips that previously failed
(status != 'ok') in windowA_s2_matches_scored.csv.

Logic:
- Use extended seasonal rules (nearest-season fallback).
- For each FAILED chip, for each seasonal target month:
    * Build a 60-day window centered on that month,
      BUT ensure the window ends BEFORE the Planet date (WinA < WinB).
    * Search Sentinel-2 L2A within that window with eo:cloud_cover <= 40.
- Combine all candidates across target months, score them, and pick the best.
- Update only the failed rows in the original CSV.

Output:
  windowA_s2_matches_scored_v2.csv
"""

from pathlib import Path
import logging
import time
from datetime import datetime, timedelta
from calendar import monthrange

import numpy as np
import pandas as pd
from pystac_client import Client
from pystac_client.exceptions import APIError
import planetary_computer as pc

# ---------------- CONFIG ----------------

BASE = Path("/home/airg/gessuman/projects/ftw-mappingafrica-integration")

WINDOWA_CSV = BASE / "data" / "windowA" / "windowA_s2_matches_scored.csv"
PLANET_CSV  = BASE / "data" / "mappingafrica-catalog-dated.csv"

OUT_CSV     = BASE / "data" / "windowA" / "windowA_s2_matches_scored_v2.csv"

STAC_URL    = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_COLLECTION = "sentinel-2-l2a"

CLOUD_MAX   = 40        # % cloud ceiling for rescue
WINDOW_DAYS = 60.0      # total width; roughly +/- 30 days around center
MAX_ITEMS   = 100
MAX_RETRIES = 4
BACKOFF_BASE = 1.7

# ---------------- REGION MAP & RULES (match main script) ----------------

REGION_MAP = {
    # Region 1: West Africa – Sudano-Sahel
    "sudano_sahel": [
        "GH", "NG", "ML", "NE", "SD", "TD",
        "BF", "MR", "SN",
    ],
    # Region 2: West Africa – Coastal/Forest Zone
    "coastal_forest": [
        "CI", "TG", "BJ",
        "GM", "GN", "GW", "SL", "GA", "CM", "CG", "CD",
    ],
    # Region 3: East Africa – Equatorial Bimodal
    "bimodal": [
        "UG", "KE", "RW", "BI", "TZ",
        "CF", "SO",
    ],
    # Region 4: East Africa – Unimodal Highlands / Ethiopia
    "ethiopia": [
        "ET", "ER",
    ],
    # Region 5: Southern Africa – Summer rainfall
    "southern": [
        "ZM", "ZW", "MZ", "ZA", "MW", "BW", "NA",
        "AO", "LS", "SZ",
    ],
}

REGION_RULES = {
    # ---------- Region 1: West Africa – Sudano-Sahel ----------
    #   If chip in Aug–Sep pair with Dec–Feb
    #   If chip in Feb–Mar pair with Aug–Oct
    "sudano_sahel": [
        {"months": [8, 9], "target": [12, 1, 2], "label": "Aug–Sep→Dec–Feb"},
        {"months": [2, 3], "target": [8, 9, 10], "label": "Feb–Mar→Aug–Oct"},
    ],

    # ---------- Region 2: West Africa – Coastal/Forest Zone ----------
    #   If chip in Mar–Apr pair with Sep–Nov
    #   If chip in Oct–Nov pair with Feb–Mar
    "coastal_forest": [
        {"months": [3, 4], "target": [9, 10, 11], "label": "Mar–Apr→Sep–Nov"},
        {"months": [10, 11], "target": [2, 3], "label": "Oct–Nov→Feb–Mar"},
    ],

    # ---------- Region 3: East Africa – Equatorial Bimodal ----------
    #   If chip in Mar–May pair with Jun–Jul
    #   If chip in Sep–Nov pair with Jan–Feb
    "bimodal": [
        {"months": [3, 4, 5], "target": [6, 7], "label": "Mar–May→Jun–Jul"},
        {"months": [9, 10, 11], "target": [1, 2], "label": "Sep–Nov→Jan–Feb"},
    ],

    # ---------- Region 4: East Africa – Unimodal Highlands / Ethiopia ----------
    #   If chip in Aug–Sep pair with Dec–Feb
    #   If chip in Jan–Mar pair with Aug–Sep
    "ethiopia": [
        {"months": [8, 9], "target": [12, 1, 2], "label": "Aug–Sep→Dec–Feb"},
        {"months": [1, 2, 3], "target": [8, 9], "label": "Jan–Mar→Aug–Sep"},
    ],

    # ---------- Region 5: Southern Africa – Summer rainfall ----------
    #   If chip in Jan–Feb pair with Jul–Aug
    #   If chip in Aug pair with Jan–Feb
    "southern": [
        {"months": [1, 2], "target": [7, 8], "label": "Jan–Feb→Jul–Aug"},
        {"months": [8], "target": [1, 2], "label": "Aug→Jan–Feb"},
    ],
}

# ---------------- LOGGING ----------------

logger = logging.getLogger("windowA_rescue_extended60")
logger.setLevel(logging.INFO)
logger.handlers.clear()
h = logging.StreamHandler()
h.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
logger.addHandler(h)

# ---------------- HELPERS ----------------

def ensure_naive(ts):
    """Convert pandas Timestamp or datetime to tz-naive Python datetime."""
    if isinstance(ts, pd.Timestamp):
        ts = ts.to_pydatetime()
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.replace(tzinfo=None)
    return ts

def which_region(country_code: str):
    cc = country_code.upper()
    for reg, codes in REGION_MAP.items():
        if cc in codes:
            return reg
    return None

def pick_target_months(region_name: str, planet_month: int):
    """
    Extended version:
    - First, use exact rule month matches.
    - If none, attach this month to the NEAREST seasonal rule in this region.
    """
    rules = REGION_RULES.get(region_name)
    if not rules:
        return ([], "no_region_rule")

    m = int(planet_month)

    # 1) Exact match
    for r in rules:
        if m in r["months"]:
            return (r["target"], r["label"])

    # 2) Nearest rule in circular month space
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

def seasonal_target_windows(planet_dt, target_months):
    """
    For each target month, build a 60-day window centered on that month,
    with the constraint that the window ends BEFORE planet_dt (WinA < WinB).

    Returns list of dicts:
      {"start": datetime, "end": datetime, "center": datetime}
    """
    pm = planet_dt.month
    py = planet_dt.year
    half = WINDOW_DAYS / 2.0

    windows = []

    for tm in target_months:
        tm = int(tm)

        # Decide year so that this seasonal month is before the Planet date.
        # Simple rule:
        #   if tm < pm: same year
        #   else:       previous year
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

        # enforce WinA < WinB
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

def query_s2_items(client, lon, lat, start_dt, end_dt,
                   cloud_max=CLOUD_MAX, max_items=MAX_ITEMS):
    """
    Query Sentinel-2 L2A items intersecting a point between start_dt and end_dt
    with eo:cloud_cover <= cloud_max.
    """
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
                    f"STAC error for lon={lon}, lat={lat} in {start_iso}..{end_iso}: {e}"
                )
                return []
            wait = BACKOFF_BASE ** attempt
            logger.info(
                f"STAC timeout, retry {attempt}/{MAX_RETRIES-1} in {wait:.1f}s … ({e})"
            )
            time.sleep(wait)

def score_candidate(row, target, planet_dt):
    """
    Score a candidate:
      score = |S2 - target_center| (days) + 0.5*|S2 - Planet| (days) + 0.05*cloud
    """
    s2_dt = ensure_naive(row["s2_datetime"])
    center = ensure_naive(target["center"])
    planet_dt = ensure_naive(planet_dt)

    days_center = abs((s2_dt - center).total_seconds()) / 86400.0
    days_planet = abs((s2_dt - planet_dt).total_seconds()) / 86400.0
    cloud = row["s2_cloud"] if np.isfinite(row["s2_cloud"]) else 100.0

    return days_center + 0.5 * days_planet + 0.05 * cloud

def pick_best_scored(candidates, planet_dt):
    """
    candidates: list of dicts {"row": row_dict, "target": target_dict}
    Returns best (row, target, score) or None.
    """
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
    logger.info("Loading windowA matches CSV…")
    df = pd.read_csv(WINDOWA_CSV, parse_dates=["planet_date", "s2_datetime"])

    logger.info("Loading Planet catalog for coordinates…")
    cat = pd.read_csv(PLANET_CSV)

    # Ensure catalog has chip ids in 'name'
    if "name" not in cat.columns:
        raise ValueError("Planet catalog must have a 'name' column for chip ids.")

    # Build a base chip id in both frames:
    # scored:  'BF0742031_2018-12' -> 'BF0742031'
    # catalog: 'BF0742031'         -> 'BF0742031'
    df["chip_base"] = df["chip_id"].astype(str).str.split("_").str[0]
    cat["chip_base"] = cat["name"].astype(str)

    # Check that coordinates exist in catalog
    if not {"x", "y"}.issubset(cat.columns):
        raise ValueError("Planet catalog must have columns 'x' and 'y'.")

    # Drop any x,y in df if present
    df = df.drop(columns=["x", "y"], errors="ignore")

    # Merge x,y from catalog onto scored results
    df = df.merge(
        cat[["chip_base", "x", "y"]],
        on="chip_base",
        how="left",
        validate="1:1"
    )

    logger.info(f"Total rows in windowA CSV: {len(df)}")
    to_rescue = df[df["status"] != "ok"].copy()
    logger.info(f"Rows with status != 'ok' (rescue set): {len(to_rescue)}")
    logger.info("Rescue status breakdown:")
    logger.info(to_rescue["status"].value_counts().to_string())

    client = Client.open(STAC_URL)

    updated_df = df.copy()
    rescued = 0
    still_failed = 0

    for idx, row in to_rescue.iterrows():
        chip_id = row["chip_id"]
        planet_dt = ensure_naive(row["planet_date"])
        lon = float(row["x"])
        lat = float(row["y"])
        country = str(row["country"]).upper()

        if np.isnan(lon) or np.isnan(lat):
            logger.warning(f"{chip_id}: missing coords, cannot rescue")
            updated_df.loc[idx, "status"] = "no_coords_rescue60"
            continue

        region = which_region(country)
        if region is None:
            logger.warning(f"{chip_id}: no region rule for country {country}")
            updated_df.loc[idx, "status"] = "no_region_rescue60"
            continue

        target_months, season_label = pick_target_months(region, planet_dt.month)
        if not target_months:
            logger.warning(f"{chip_id}: no seasonal target months after extension")
            updated_df.loc[idx, "status"] = "no_pair_rescue60"
            continue

        targets = seasonal_target_windows(planet_dt, target_months)

        tier_candidates = []
        for t in targets:
            rows_s2 = query_s2_items(
                client, lon, lat, t["start"], t["end"], cloud_max=CLOUD_MAX
            )
            for r in rows_s2:
                tier_candidates.append({"row": r, "target": t})

        best = pick_best_scored(tier_candidates, planet_dt)

        if best is None:
            updated_df.loc[idx, "status"] = "no_match_rescue60"
            updated_df.loc[idx, "s2_id"] = None
            updated_df.loc[idx, "s2_datetime"] = pd.NaT
            updated_df.loc[idx, "s2_cloud"] = np.nan
            updated_df.loc[idx, "abs_days"] = np.nan
            updated_df.loc[idx, "score"] = np.nan
            still_failed += 1
        else:
            r, tg, sc = best
            s2_dt = ensure_naive(r["s2_datetime"])
            abs_days = abs((s2_dt - planet_dt).total_seconds()) / 86400.0

            updated_df.loc[idx, "s2_id"] = r["s2_id"]
            updated_df.loc[idx, "s2_datetime"] = s2_dt
            updated_df.loc[idx, "s2_cloud"] = float(r["s2_cloud"])
            updated_df.loc[idx, "abs_days"] = abs_days
            updated_df.loc[idx, "score"] = sc
            updated_df.loc[idx, "season_label"] = season_label
            updated_df.loc[idx, "status"] = "ok_extended60"

            rescued += 1
            logger.info(
                f"{chip_id}: rescued with S2 {r['s2_id']} at {s2_dt.date()} "
                f"(Δ={abs_days:.1f} days, cloud={r['s2_cloud']:.1f}%)"
            )

        time.sleep(0.15)

    logger.info(f"Rescue finished. Rescued rows: {rescued}, still failed: {still_failed}")
    logger.info(f"Writing updated CSV → {OUT_CSV}")
    updated_df.to_csv(OUT_CSV, index=False)
    logger.info("Done.")

if __name__ == "__main__":
    main()
