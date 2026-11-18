#!/usr/bin/env python3
"""
windowA_s2_matches_scored.py

Find Sentinel-2 matches for "window A" (seasonal pair) for each Planet chip,
using seasonal pairing rules, multi-tier cloud thresholds, and a scored pick.

Key changes:
  - WinA must be strictly earlier than WinB (Planet) → enforced in
    build_season_targets_earlier().
  - REGION_RULES expanded based on audit (GH-12, NG/ML/BF-4, TZ-8/12, ET-4, ZA/ZM/ZW-6, …).
  - pick_target_months() is STRICT: only applies explicit rules (no _extended fallback).
  - Assertion to guarantee WinA < WinB.

Output:
  /home/airg/gessuman/projects/ftw-mappingafrica-integration/data/windowA/windowA_s2_matches_scored.csv
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

# Planet catalog with window B metadata
# Must have: window_b, country, planet_date, x, y
PLANET_CSV = BASE / "data" / "qa" / "mappingafrica_catalog_badchips.csv"

OUTDIR = BASE / "data" / "windowA"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUTDIR / "bad_chips_windowA_s2_matches_scored.csv"
LOGFILE = OUTDIR / "bad_chips_windowA_matches_scored_adj.log"

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_COLLECTION = "sentinel-2-l2a"

# Cloud tiers (%)
CLOUD_THRESHOLDS = [5, 10, 15, 20, 30, 50]

MAX_ITEMS_PER_QUERY = 100
MAX_RETRIES = 4
BACKOFF_BASE = 1.7

# ---------------- LOGGING ----------------

logger = logging.getLogger("windowA_s2_scored")
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
# Region definitions
REGION_MAP = {
    # Region 1: West Africa – Sudano-Sahel
    "sudano_sahel": ["GH", "NG", "ML", "NE", "SD", "TD", "BF", "MR", "SN"],
    # Region 2: West Africa – Coastal/Forest
    "coastal_forest": ["CI", "TG", "BJ", "GM", "GN", "GW", "SL", "GA", "CM", "CG", "CD"],
    # Region 3: East Africa – Equatorial Bimodal
    "bimodal": ["UG", "KE", "RW", "BI", "TZ", "CF", "SO"],
    # Region 4: East Africa – Unimodal Highlands / Ethiopia
    "ethiopia": ["ET", "ER"],
    # Region 5: Southern Africa – Summer rainfall
    "southern": ["ZM", "ZW", "MZ", "ZA", "MW", "BW", "NA", "AO", "LS", "SZ"],
}

# Refined opposite-window rules (source months -> target months)
REGION_RULES = {
    # ---------- 1) Sudano–Sahel ----------
    # Adds coverage for Dec–Jan and April based on audit
    "sudano_sahel": [
        {"months": [8, 9],        "target": [12, 1, 2],     "label": "Aug–Sep→Dec–Feb"},
        {"months": [2, 3, 4],     "target": [8, 9, 10],     "label": "Feb–Apr→Aug–Oct"},
        {"months": [12, 1],       "target": [8, 9, 10],     "label": "Dec–Jan→Aug–Oct"},
    ],

    # ---------- 2) Coastal / Forest ----------
    # Expands early-year and late dry-season overlap
    "coastal_forest": [
        {"months": [2, 3, 4],     "target": [9, 10, 11],    "label": "Feb–Apr→Sep–Nov"},
        {"months": [8, 9, 10],    "target": [2, 3, 4],      "label": "Aug–Oct→Feb–Apr"},
    ],

    # ---------- 3) Bimodal East Africa ----------
    # Extends both ends to capture Feb and Dec signals
    "bimodal": [
        {"months": [2, 3, 4, 5],  "target": [7, 8],         "label": "Feb–May→Jul–Aug"},
        {"months": [8, 9, 10, 11, 12], "target": [1, 2, 3], "label": "Aug–Dec→Jan–Mar"},
    ],

    # ---------- 4) Ethiopia / Highlands ----------
    "ethiopia": [
        {"months": [8, 9],        "target": [12, 1, 2],     "label": "Aug–Sep→Dec–Feb"},
        {"months": [1, 2, 3, 4],  "target": [8, 9],         "label": "Jan–Apr→Aug–Sep"},
    ],

    # ---------- 5) Southern Africa ----------
    "southern": [
        {"months": [1, 2, 3],     "target": [7, 8, 9],      "label": "Jan–Mar→Jul–Sep"},
        {"months": [6, 7, 8, 9],  "target": [1, 2, 3],      "label": "Jun–Sep→Jan–Mar"},
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
    STRICT: only return target months if planet_month is explicitly covered
    by a rule in the given region. Otherwise return ([], "no_rule").
    """
    rules = REGION_RULES.get(region_name)
    if not rules:
        return ([], "no_region_rule")

    m = int(planet_month)
    for r in rules:
        if m in r["months"]:
            return (r["target"], r["label"])
    return ([], "no_rule")

# ---------------- WINDOW BUILDER (WinA must be earlier than WinB) ----------------

def build_season_targets_earlier(planet_dt: datetime, target_months):
    """
    Build seasonal target windows for S2 search such that each window is
    STRICTLY EARLIER than the Planet (win_B) datetime.

    Returns a list of dicts:
      {"start": dt, "end": dt, "center": dt}
    """
    pm = planet_dt.month
    py = planet_dt.year

    windows = []
    for tm in sorted(set(target_months)):
        # try same year
        year = py
        start = datetime(year, tm, 1)
        end = datetime(year + (1 if tm == 12 else 0), (1 if tm == 12 else tm + 1), 1) - timedelta(seconds=1)

        # if not earlier, shift to previous year
        if end >= planet_dt:
            year -= 1
            start = datetime(year, tm, 1)
            end = datetime(year + (1 if tm == 12 else 0), (1 if tm == 12 else tm + 1), 1) - timedelta(seconds=1)

        if end < planet_dt:
            center = start + (end - start) / 2
            windows.append({"start": ensure_naive(start), "end": ensure_naive(end), "center": ensure_naive(center)})

    # merge contiguous/overlapping windows
    windows.sort(key=lambda x: x["start"])
    merged = []
    for w in windows:
        if not merged:
            merged.append(w)
            continue
        last = merged[-1]
        if w["start"] <= last["end"] + timedelta(days=1):
            new_start = last["start"]
            new_end = max(last["end"], w["end"])
            new_center = new_start + (new_end - new_start) / 2
            merged[-1] = {"start": new_start, "end": new_end, "center": new_center}
        else:
            merged.append(w)
    return merged

# ---------------- STAC & SCORING ----------------

def query_s2_items(client, lon, lat, start_date, end_date, cloud_max, max_items=MAX_ITEMS_PER_QUERY):
    """
    Query Sentinel-2 L2A items intersecting a point for [start_date, end_date],
    with eo:cloud_cover <= cloud_max.
    Returns list of row dicts.
    """
    params = dict(
        collections=[S2_COLLECTION],
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
            logger.info(f"STAC timeout, retry {attempt}/{MAX_RETRIES-1} in {sleep_s:.1f}s … ({e})")
            time.sleep(sleep_s)

def score_candidate(row, target, planet_dt):
    """
    Lower is better. Emphasize seasonal alignment, then temporal proximity to WinB,
    with small cloud penalty.
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
    candidates: list of {"row": row_dict, "target": target_dict}
    Returns best (row, target, score) or None.
    """
    if not candidates:
        return None
    best = None
    best_score = np.inf
    for c in candidates:
        sc = score_candidate(c["row"], c["target"], planet_dt)
        if sc < best_score:
            best_score = sc
            best = (c["row"], c["target"], sc)
    return best

# ---------------- MAIN ----------------

def main():
    logger.info("Loading Planet chip CSV…")
    df = pd.read_csv(PLANET_CSV)

    # required columns
    req_cols = ["window_b", "country", "planet_date", "x", "y"]
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns {missing}")

    # chip_id from window_b filename
    df["chip_id"] = df["window_b"].astype(str).apply(lambda p: Path(p).stem)

    # parse date
    df["planet_date"] = pd.to_datetime(df["planet_date"], errors="coerce")
    df = df.dropna(subset=["planet_date", "x", "y", "country"]).copy()

    # ----- RESUME -----
    if OUT_CSV.exists():
        done = pd.read_csv(OUT_CSV)
        if "chip_id" in done.columns:
            processed = set(done["chip_id"].astype(str))
            before = len(df)
            df = df[~df["chip_id"].astype(str).isin(processed)].copy()
            logger.info(f"Resume: {before - len(df)} chips already in {OUT_CSV}, {len(df)} remaining.")
        else:
            logger.info(f"{OUT_CSV} exists but no 'chip_id' column; will overwrite.")

    if df.empty:
        logger.info("No chips left to process. Exiting.")
        return

    logger.info(f"Total chips to process this run: {len(df)}")
    client = Client.open(STAC_URL)

    new_results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scored windowA matches", unit="chip"):
        chip_id = str(row["chip_id"])
        country = str(row["country"]).upper()
        planet_dt = ensure_naive(row["planet_date"])
        lon = float(row["x"])
        lat = float(row["y"])

        region = which_region(country)
        if region is None:
            new_results.append({
                "chip_id": chip_id, "country": country, "planet_date": planet_dt,
                "season_label": "no_region", "tier_used": None, "cloud_limit": None,
                "s2_id": None, "s2_datetime": None, "s2_cloud": None,
                "abs_days": None, "score": None, "status": "no_region_rule",
            })
            continue

        target_months, season_label = pick_target_months(region, planet_dt.month)
        if not target_months:
            new_results.append({
                "chip_id": chip_id, "country": country, "planet_date": planet_dt,
                "season_label": season_label, "tier_used": None, "cloud_limit": None,
                "s2_id": None, "s2_datetime": None, "s2_cloud": None,
                "abs_days": None, "score": None, "status": "no_pair_for_month",
            })
            continue

        targets = build_season_targets_earlier(planet_dt, target_months)
        if not targets:
            new_results.append({
                "chip_id": chip_id, "country": country, "planet_date": planet_dt,
                "season_label": season_label, "tier_used": None, "cloud_limit": None,
                "s2_id": None, "s2_datetime": None, "s2_cloud": None,
                "abs_days": None, "score": None, "status": "no_window_before_winB",
            })
            continue

        match_row = None
        match_target = None
        match_score = None
        tier_used = None
        cloud_limit = None

        # multi-tier search
        for t_idx, cloud_max in enumerate(CLOUD_THRESHOLDS, start=1):
            tier_label = f"tier{t_idx}_le{cloud_max}"
            tier_candidates = []

            for t in targets:
                start_iso = t["start"].strftime("%Y-%m-%d")
                end_iso = t["end"].strftime("%Y-%m-%d")
                try:
                    rows = query_s2_items(client, lon, lat, start_iso, end_iso, cloud_max)
                    for r in rows:
                        tier_candidates.append({"row": r, "target": t})
                except Exception as e:
                    logger.info(f"{chip_id}: STAC error {tier_label} {start_iso}..{end_iso}: {e}")

            best = pick_best_scored(tier_candidates, planet_dt)
            if best is not None:
                r, tg, sc = best
                # FINAL SAFETY: enforce WinA < WinB
                if r["s2_datetime"] >= planet_dt:
                    continue
                match_row = r
                match_target = tg
                match_score = sc
                tier_used = tier_label
                cloud_limit = cloud_max
                break  # stop at first tier that yields a valid earlier match

        if match_row is None:
            new_results.append({
                "chip_id": chip_id, "country": country, "planet_date": planet_dt,
                "season_label": season_label, "tier_used": None, "cloud_limit": None,
                "s2_id": None, "s2_datetime": None, "s2_cloud": None,
                "abs_days": None, "score": None, "status": "no_match_all_tiers",
            })
        else:
            s2_dt = ensure_naive(match_row["s2_datetime"])
            # assertion (should always pass due to earlier checks)
            assert s2_dt < planet_dt, f"{chip_id}: WinA ({s2_dt}) is not earlier than WinB ({planet_dt})"
            abs_days = abs((s2_dt - planet_dt).total_seconds()) / 86400.0

            new_results.append({
                "chip_id": chip_id, "country": country, "planet_date": planet_dt,
                "season_label": season_label, "tier_used": tier_used, "cloud_limit": cloud_limit,
                "s2_id": match_row["s2_id"], "s2_datetime": s2_dt, "s2_cloud": float(match_row["s2_cloud"]),
                "abs_days": abs_days, "score": match_score, "status": "ok",
            })

    # ----- WRITE / APPEND OUTPUT -----
    new_df = pd.DataFrame(new_results)

    if OUT_CSV.exists() and "chip_id" in pd.read_csv(OUT_CSV, nrows=1).columns:
        new_df.to_csv(OUT_CSV, mode="a", header=False, index=False)
        logger.info(f"Appended {len(new_df)} rows to existing {OUT_CSV}")
    else:
        new_df.to_csv(OUT_CSV, index=False)
        logger.info(f"Wrote {len(new_df)} rows to {OUT_CSV}")

    logger.info("Done.")

if __name__ == "__main__":
    main()
