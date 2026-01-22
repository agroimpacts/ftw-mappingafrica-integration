#!/usr/bin/env python3
"""
windowA_rescue_94_region_tweaks.py

Re-run seasonal matching for the 94 windowA chips that still failed
after the main rescue run (windowA_rescue_2725).

Key points:
- Input:  94-chip CSV with planet_date + chip_id
          data/windowA/windowA_rescue_94_region_tweaks.csv
- We merge x,y from mappingafrica-catalog-dated.csv using name = chip_id prefix.
- Country is ALWAYS derived from chip_id[:2].
- Region is mapped from country using REGION_MAP.
- Region-specific seasonal overrides for these 94 chips are applied.
- We enforce WinA < WinB via year logic in seasonal_target_windows().
- Cloud tiers: [5, 10, 15, 20, 30, 40].
- If seasonal tiers find no match, we do a final 90-day fallback search
  (planet_date - 90 days → planet_date, cloud <= 40%).
- Output: data/windowA/windowA_rescue_94_region_tweaks_matches.csv
"""

from pathlib import Path
import logging
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

from pystac_client import Client
from pystac_client.exceptions import APIError
import planetary_computer as pc

# ---------------- CONFIG ----------------

BASE = Path("/home/airg/gessuman/projects/ftw-mappingafrica-integration")

INPUT_94_CSV = BASE / "data" / "windowA" / "windowA_rescue_94_region_tweaks.csv"
DATED_CSV    = BASE / "data" / "mappingafrica-catalog-dated.csv"
OUT_CSV      = BASE / "data" / "windowA" / "windowA_rescue_94_region_tweaks_matches.csv"

STAC_URL      = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_COLLECTION = "sentinel-2-l2a"

# Cloud thresholds (%) for tiered search (seasonal windows)
CLOUD_THRESHOLDS = [5, 10, 15, 20, 30, 40]

MAX_ITEMS_PER_QUERY = 100
MAX_RETRIES         = 4
BACKOFF_BASE        = 1.7

# Fallback 90-day window (WinA must be before WinB)
FALLBACK_DAYS = 90
FALLBACK_CLOUD_MAX = 40

# ---------------- LOGGING ----------------

logger = logging.getLogger("windowA_rescue_94")
logger.setLevel(logging.INFO)
logger.handlers.clear()
fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
sh = logging.StreamHandler()
sh.setFormatter(fmt)
logger.addHandler(sh)

# ---------------- UTILITIES ----------------


def ensure_naive(dt):
    """Return timezone-naive datetime from pandas Timestamp or datetime."""
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    if hasattr(dt, "tzinfo") and dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    return dt


# ---------------- REGION MAP & RULES ----------------

REGION_MAP = {
    # Region 1: West Africa – Sudano-Sahel
    "sudano_sahel": [
        "GH", "NG", "ML", "NE", "SD", "TD",
        "BF", "MR", "SN",
    ],

    # Region 2: West Africa – Coastal/Forest
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

# Baseline windowB→windowA rules (original logic)
BASE_RULES = {
    "sudano_sahel": [
        {"months": [8, 9], "target": [12, 1, 2], "label": "Aug–Sep→Dec–Feb"},
        {"months": [2, 3], "target": [8, 9, 10], "label": "Feb–Mar→Aug–Oct"},
    ],
    "coastal_forest": [
        {"months": [3, 4], "target": [9, 10, 11], "label": "Mar–Apr→Sep–Nov"},
        {"months": [10, 11], "target": [2, 3], "label": "Oct–Nov→Feb–Mar"},
    ],
    "bimodal": [
        {"months": [3, 4, 5], "target": [6, 7], "label": "Mar–May→Jun–Jul"},
        {"months": [9, 10, 11], "target": [1, 2], "label": "Sep–Nov→Jan–Feb"},
    ],
    "ethiopia": [
        {"months": [8, 9], "target": [12, 1, 2], "label": "Aug–Sep→Dec–Feb"},
        {"months": [1, 2, 3], "target": [8, 9], "label": "Jan–Mar→Aug–Sep"},
    ],
    "southern": [
        {"months": [1, 2], "target": [7, 8], "label": "Jan–Feb→Jul–Aug"},
        {"months": [8], "target": [1, 2], "label": "Aug→Jan–Feb"},
    ],
}

# Extra overrides tuned specifically from the 94-chip diagnostics
# (region, planet_month) → new target months
REGION_OVERRIDES_94 = {
    "sudano_sahel": {
        2: {"target": [8, 9], "label": "Feb→Aug–Sep_94"},
        4: {"target": [8, 9], "label": "Apr→Aug–Sep_94"},
        12: {"target": [7, 8], "label": "Dec→Jul–Aug_prev_94"},
    },
    "southern": {
        6: {"target": [1, 2], "label": "Jun→Jan–Feb_next_94"},
        8: {"target": [1, 2], "label": "Aug→Jan–Feb_next_94"},
        9: {"target": [1, 2], "label": "Sep→Jan–Feb_next_94"},
    },
    "ethiopia": {
        2: {"target": [8, 9], "label": "Feb→Aug–Sep_prev_94"},
        3: {"target": [8, 9], "label": "Mar→Aug–Sep_prev_94"},
        4: {"target": [8, 9], "label": "Apr→Aug–Sep_prev_94"},
    },
    "coastal_forest": {
        3: {"target": [9, 10, 11], "label": "Mar→Sep–Nov_94"},
        8: {"target": [2, 3], "label": "Aug→Feb–Mar_next_94"},
    },
    "bimodal": {
        8: {"target": [3, 4, 5], "label": "Aug→Mar–May_prev_94"},
    },
}


def which_region(country_code):
    """
    Map a country code (2-letter) to region.
    Robust to NaNs, floats, or longer strings.
    """
    if isinstance(country_code, float) and np.isnan(country_code):
        return "other"

    cc = str(country_code).strip().upper()
    if len(cc) >= 2:
        cc = cc[:2]

    for reg, codes in REGION_MAP.items():
        if cc in codes:
            return reg
    return "other"


def pick_target_months_with_overrides(region_name: str, planet_month: int):
    """
    Given region + windowB month, return (target_months, label) for windowA.

    Priority:
      1. Region-specific overrides for the 94 chips (REGION_OVERRIDES_94)
      2. Baseline rules (BASE_RULES)
      3. Extended nearest rule (same behaviour as the main script)
    """
    m = int(planet_month)

    # 1. Region-specific override
    overrides = REGION_OVERRIDES_94.get(region_name, {})
    if m in overrides:
        o = overrides[m]
        return o["target"], o["label"]

    # 2. Exact match in baseline rules
    rules = BASE_RULES.get(region_name)
    if not rules:
        return [], "no_region_rule"

    for r in rules:
        if m in r["months"]:
            return r["target"], r["label"]

    # 3. Nearest-rule extension (like *_extended in main script)
    def month_distance(a, b):
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
        return best_rule["target"], best_rule["label"] + "_extended_94"

    return [], "no_rule_94"


def seasonal_target_windows(planet_dt: datetime, target_months):
    """
    Build seasonal target windows for WinA such that WinA < WinB (Planet date).

    For each target month 'tm', we assign a year so that the window's
    END date is strictly before planet_dt.
    """
    pm = planet_dt.month
    py = planet_dt.year

    windows = []
    for tm in target_months:
        # If target month >= planet month, use previous year; otherwise same year.
        if tm >= pm:
            year = py - 1
        else:
            year = py

        start = datetime(year, tm, 1)
        if tm == 12:
            end = datetime(year + 1, 1, 1) - timedelta(seconds=1)
        else:
            end = datetime(year, tm + 1, 1) - timedelta(seconds=1)

        # Only keep if the window ends strictly before the Planet date
        if end < planet_dt:
            center = start + (end - start) / 2
            windows.append(
                {
                    "start": ensure_naive(start),
                    "end": ensure_naive(end),
                    "center": ensure_naive(center),
                }
            )

    # merge overlapping windows if any
    windows.sort(key=lambda x: x["start"])
    merged = []
    for w in windows:
        if not merged:
            merged.append(w)
        else:
            last = merged[-1]
            if w["start"] <= last["end"] + timedelta(days=1):
                new_start = last["start"]
                new_end = max(last["end"], w["end"])
                new_center = new_start + (new_end - new_start) / 2
                merged[-1] = {
                    "start": new_start,
                    "end": new_end,
                    "center": ensure_naive(new_center),
                }
            else:
                merged.append(w)
    return merged


# ---------------- STAC & SCORING ----------------


def query_s2_items(client, lon, lat, start_dt, end_dt, cloud_max):
    """
    Query Sentinel-2 L2A items intersecting (lon,lat) for [start_dt,end_dt]
    with eo:cloud_cover <= cloud_max. Returns list of dict rows.
    """
    start_iso = start_dt.strftime("%Y-%m-%d")
    end_iso = end_dt.strftime("%Y-%m-%d")

    params = dict(
        collections=[S2_COLLECTION],
        intersects={"type": "Point", "coordinates": [lon, lat]},
        datetime=f"{start_iso}/{end_iso}",
        query={"eo:cloud_cover": {"lte": cloud_max}},
        limit=MAX_ITEMS_PER_QUERY,
        max_items=MAX_ITEMS_PER_QUERY,
    )

    attempt = 0
    while True:
        try:
            search = Client.open(STAC_URL).search(**params)
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
                raise
            sleep_s = BACKOFF_BASE ** attempt
            logger.info(
                f"STAC timeout, retry {attempt}/{MAX_RETRIES} in {sleep_s:.1f}s … ({e})"
            )
            time.sleep(sleep_s)


def score_candidate(row, target, planet_dt):
    """
    Score a candidate S2 row against a target seasonal window.
    Lower is better.
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
    logger.info(f"Loading 94-chip rescue CSV: {INPUT_94_CSV}")
    df = pd.read_csv(INPUT_94_CSV)

    # Ensure planet_date is datetime
    df["planet_date"] = pd.to_datetime(df["planet_date"], errors="coerce")

    # Derive 'name' from chip_id (prefix before first "_")
    df["name"] = df["chip_id"].astype(str).str.split("_").str[0]

    # Merge x,y from dated catalog
    logger.info(f"Loading dated catalog for coordinates: {DATED_CSV}")
    dated = pd.read_csv(DATED_CSV, usecols=["name", "x", "y"]).drop_duplicates("name")

    df = df.merge(dated, on="name", how="left")

    # Check coordinate completeness
    missing_x = df["x"].isna().sum()
    missing_y = df["y"].isna().sum()
    if missing_x > 0 or missing_y > 0:
        logger.warning(f"Missing x coords: {missing_x}, missing y coords: {missing_y}")
    else:
        logger.info("All 94 chips have x,y after merging.")

    client = Client.open(STAC_URL)

    results = []

    for _, row in tqdm(
        df.iterrows(), total=len(df), desc="Rescuing 94 chips", unit="chip"
    ):
        chip_id = str(row["chip_id"])
        planet_dt = ensure_naive(row["planet_date"])

        # derive country from chip_id prefix ALWAYS (robust)
        country = chip_id[:2].upper()
        region = which_region(country)

        lon = float(row["x"]) if not pd.isna(row["x"]) else np.nan
        lat = float(row["y"]) if not pd.isna(row["y"]) else np.nan

        base = {
            "chip_id": chip_id,
            "country": country,
            "planet_date": planet_dt,
        }

        if pd.isna(planet_dt) or np.isnan(lon) or np.isnan(lat):
            base.update(
                {
                    "season_label": "no_date_or_coords_94",
                    "tier_used": None,
                    "cloud_limit": None,
                    "s2_id": None,
                    "s2_datetime": None,
                    "s2_cloud": None,
                    "abs_days": None,
                    "score": None,
                    "status": "no_date_or_coords_94",
                }
            )
            results.append(base)
            continue

        target_months, season_label = pick_target_months_with_overrides(
            region, planet_dt.month
        )
        if not target_months:
            base.update(
                {
                    "season_label": season_label,
                    "tier_used": None,
                    "cloud_limit": None,
                    "s2_id": None,
                    "s2_datetime": None,
                    "s2_cloud": None,
                    "abs_days": None,
                    "score": None,
                    "status": "no_pair_for_month_94",
                }
            )
            results.append(base)
            continue

        targets = seasonal_target_windows(planet_dt, target_months)
        if not targets:
            base.update(
                {
                    "season_label": season_label + "_no_valid_WinA_window",
                    "tier_used": None,
                    "cloud_limit": None,
                    "s2_id": None,
                    "s2_datetime": None,
                    "s2_cloud": None,
                    "abs_days": None,
                    "score": None,
                    "status": "no_valid_WinA_window_94",
                }
            )
            results.append(base)
            continue

        match_row = None
        match_score = None
        tier_used = None
        cloud_limit = None

        # ---- 1) Seasonal tiered search ----
        for t_idx, cloud_max in enumerate(CLOUD_THRESHOLDS, start=1):
            tier_label = f"tier{t_idx}_le{cloud_max}"
            tier_candidates = []

            for t in targets:
                try:
                    rows_s2 = query_s2_items(
                        client, lon, lat, t["start"], t["end"], cloud_max
                    )
                    for r in rows_s2:
                        tier_candidates.append({"row": r, "target": t})
                except Exception as e:
                    logger.info(
                        f"{chip_id}: STAC error tier {tier_label} "
                        f"{t['start'].date()}..{t['end'].date()}: {e}"
                    )

            best = pick_best_scored(tier_candidates, planet_dt)
            if best is not None:
                r, tg, sc = best
                match_row = r
                match_score = sc
                tier_used = tier_label
                cloud_limit = cloud_max
                break  # stop at first successful tier

        # ---- 2) If still no match, do 90-day fallback around Planet date ----
        if match_row is None:
            start_fb = planet_dt - timedelta(days=FALLBACK_DAYS)
            end_fb   = planet_dt - timedelta(seconds=1)  # enforce WinA < WinB

            try:
                fb_rows = query_s2_items(
                    client, lon, lat, start_fb, end_fb, FALLBACK_CLOUD_MAX
                )
            except Exception as e:
                logger.info(
                    f"{chip_id}: fallback 90-day STAC error "
                    f"{start_fb.date()}..{end_fb.date()}: {e}"
                )
                fb_rows = []

            if fb_rows:
                fb_df = pd.DataFrame(fb_rows)
                fb_df["abs_days"] = (
                    fb_df["s2_datetime"] - planet_dt
                ).abs().dt.total_seconds() / 86400.0
                fb_df = fb_df.sort_values(["abs_days", "s2_cloud"])
                best_fb = fb_df.iloc[0]

                match_row = {
                    "s2_id": best_fb["s2_id"],
                    "s2_datetime": best_fb["s2_datetime"],
                    "s2_cloud": best_fb["s2_cloud"],
                }
                match_score = None
                tier_used = "fallback90"
                cloud_limit = FALLBACK_CLOUD_MAX

        # ---- Final write-out logic ----
        if match_row is None:
            base.update(
                {
                    "season_label": season_label,
                    "tier_used": None,
                    "cloud_limit": None,
                    "s2_id": None,
                    "s2_datetime": None,
                    "s2_cloud": None,
                    "abs_days": None,
                    "score": None,
                    "status": "no_match_all_tiers_94",
                }
            )
        else:
            s2_dt = ensure_naive(match_row["s2_datetime"])
            abs_days = abs((s2_dt - planet_dt).total_seconds()) / 86400.0
            base.update(
                {
                    "season_label": season_label,
                    "tier_used": tier_used,
                    "cloud_limit": cloud_limit,
                    "s2_id": match_row["s2_id"],
                    "s2_datetime": s2_dt,
                    "s2_cloud": float(match_row["s2_cloud"]),
                    "abs_days": abs_days,
                    "score": match_score,
                    "status": (
                        "ok_rescue_94_fallback90"
                        if tier_used == "fallback90"
                        else "ok_rescue_94"
                    ),
                }
            )

        results.append(base)
        time.sleep(0.15)  # be polite to Planetary Computer

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUT_CSV, index=False)
    logger.info(f"Wrote {len(out_df)} rows to {OUT_CSV}")
    logger.info("Status breakdown:")
    logger.info(out_df["status"].value_counts().to_string())
    logger.info("Done.")


if __name__ == "__main__":
    main()
