#!/usr/bin/env python3
"""
windowA_s2_matches_scored.py

Find Sentinel-2 matches for "window A" (seasonal pair) for each Planet chip,
using regional seasonal pairing rules, multi-tier cloud thresholds, and a scored pick.

Includes:
  - Strict seasonal rule selection (no nearest/_extended fallback)
  - WinA must be strictly earlier than WinB (Planet date)
  - CLI filters for smoke tests: countries, months, sample-per-country, max-total, out-suffix
  - Resume mode (append new results)
  - Post-run validator (asserts WinA < WinB and prints gap stats)

USAGE (smoke test examples):
  python windowA_s2_matches_scored_smoke_test.py \
    --countries GH,NG,TZ,ET,ZA \
    --months 2,3,4,8,9,12 \
    --sample-per-country 15 \
    --max-total 60 \
    --out-suffix _TEST

Full run later (no filters, overwrite default OUT_CSV):
  python windowA_s2_matches_scored.py --out-suffix ""
"""

from pathlib import Path
import argparse
import logging
import random
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
PLANET_CSV = BASE / "data" / "mappingafrica-catalog-dated.csv"

OUTDIR = BASE / "data" / "windowA"
OUTDIR.mkdir(parents=True, exist_ok=True)
DEFAULT_OUT = OUTDIR / "windowA_s2_matches_scored.csv"
LOGFILE = OUTDIR / "windowA_matches_scored.log"

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_COLLECTION = "sentinel-2-l2a"

# Cloud tiers (%)
CLOUD_THRESHOLDS = [5, 10, 15, 20, 30, 40]

MAX_ITEMS_PER_QUERY = 100
MAX_RETRIES = 4
BACKOFF_BASE = 1.7

# ---------------- CLI ----------------

parser = argparse.ArgumentParser(description="WindowA S2 matching with smoke-test filters")
parser.add_argument("--countries", type=str, default="",
                    help="Comma-separated ISO2 country codes to include (e.g., 'GH,NG,TZ,ET,ZA'). Empty=all.")
parser.add_argument("--months", type=str, default="",
                    help="Comma-separated Planet months to include (1-12). Empty=all.")
parser.add_argument("--sample-per-country", type=int, default=0,
                    help="Random sample size per country (0 = take all that pass filters).")
parser.add_argument("--max-total", type=int, default=0,
                    help="Hard cap on total rows processed after sampling (0 = unlimited).")
parser.add_argument("--out-suffix", type=str, default="_TEST",
                    help="Suffix to append to output CSV filename (default: _TEST). Use empty string to overwrite default file.")
args = parser.parse_args()

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
    "sudano_sahel": [
        {"months": [8, 9],            "target": [12, 1, 2],     "label": "Aug–Sep→Dec–Feb"},
        {"months": [2, 3, 4],         "target": [8, 9, 10],     "label": "Feb–Apr→Aug–Oct"},
        {"months": [11, 12, 1],           "target": [6, 7, 8],     "label": "Nov–Jan→Jun–Aug"},
    ],

    # ---------- 2) Coastal / Forest ----------
    "coastal_forest": [
        {"months": [2, 3, 4],         "target": [9, 10, 11],    "label": "Feb–Apr→Sep–Nov"},
        {"months": [8, 9, 10],        "target": [2, 3, 4],      "label": "Aug–Oct→Feb–Apr"},
    ],

    # ---------- 3) Bimodal East Africa ----------
    "bimodal": [
        # If WinB is in LR (Mar–Jun), pair to the preceding SR (Oct–Feb)
        {"months": [3, 4, 5, 6],
         "target": [10, 11, 12, 1, 2],
         "label": "LR(Mar–Jun)→SR(Oct–Feb)"},
    
        # If WinB is in SR (Sep–Dec), pair to the preceding LR (Mar–Jun)
        {"months": [9, 10, 11, 12],
         "target": [3, 4, 5, 6],
         "label": "SR(Sep–Dec)→LR(Mar–Jun)"},
    
        # If WinB is in late SR / pre-LR (Jan–Feb), pair to late LR/Post-LR of the previous cycle
        {"months": [1, 2],
         "target": [5, 6, 7],   # May–Jul captures late LR peak → early senescence
         "label": "Jan–Feb→Late-LR/Post-LR(May–Jul)"},
    
        # If WinB is in the long dry / post-LR (Jul–Aug), pair to the preceding SR core
        {"months": [7, 8],
         "target": [1, 2], # Jan–Feb high-contrast against Jul–Aug dryness
         "label": "Jul–Aug→(Nov–Jan)"},
    ]
,

    # ---------- 4) Ethiopia / Highlands ----------
    "ethiopia": [
        {"months": [8, 9],            "target": [12, 1, 2],     "label": "Aug–Sep→Dec–Feb"},
        {"months": [1, 2, 3, 4],      "target": [8, 9],         "label": "Jan–Apr→Aug–Sep"},
    ],

    # ---------- 5) Southern Africa ----------
    "southern": [
        {"months": [1, 2, 3],         "target": [7, 8, 9],      "label": "Jan–Mar→Jul–Sep"},
        {"months": [6, 7, 8, 9],      "target": [1, 2, 3],      "label": "Jun–Sep→Jan–Mar"},
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

from datetime import datetime as _dt

def _month_end(year: int, month: int) -> _dt:
    if month == 12:
        return _dt(year + 1, 1, 1) - timedelta(seconds=1)
    return _dt(year, month + 1, 1) - timedelta(seconds=1)

def build_season_targets_earlier(planet_dt: _dt, target_months):
    """
    Build seasonal target windows such that each window is STRICTLY EARLIER than WinB.

    Returns: list of {"start": dt, "end": dt, "center": dt}
    """
    py = planet_dt.year
    windows = []
    for tm in sorted(set(target_months)):
        # try same-year assignment
        start = _dt(py, tm, 1)
        end = _month_end(py, tm)
        # shift to previous year if not earlier than WinB
        if end >= planet_dt:
            start = _dt(py - 1, tm, 1)
            end = _month_end(py - 1, tm)
        if end < planet_dt:
            center = start + (end - start) / 2
            windows.append({"start": ensure_naive(start), "end": ensure_naive(end), "center": ensure_naive(center)})

    # merge overlapping/touching
    windows.sort(key=lambda x: x["start"])
    merged = []
    for w in windows:
        if not merged:
            merged.append(w); continue
        last = merged[-1]
        if w["start"] <= last["end"] + timedelta(days=1):
            s = last["start"]
            e = max(last["end"], w["end"])
            c = s + (e - s) / 2
            merged[-1] = {"start": s, "end": e, "center": c}
        else:
            merged.append(w)
    return merged

# ---------------- STAC & SCORING ----------------

def query_s2_items(client, lon, lat, start_date, end_date, cloud_max, max_items=MAX_ITEMS_PER_QUERY):
    """
    Query Sentinel-2 L2A items intersecting a point for [start_date, end_date],
    with eo:cloud_cover <= cloud_max. Returns list of dicts.
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
                rows.append({
                    "s2_id": it.id,
                    "s2_datetime": dt,
                    "s2_cloud": float(it.properties.get("eo:cloud_cover", np.nan)),
                })
            return rows
        except APIError as e:
            attempt += 1
            if attempt >= MAX_RETRIES:
                raise
            sleep_s = BACKOFF_BASE ** attempt
            logger.info(f"STAC timeout, retry {attempt}/{MAX_RETRIES-1} in {sleep_s:.1f}s … ({e})")
            time.sleep(sleep_s)

def score_candidate(row, target, planet_dt):
    """Lower is better: seasonal center alignment, then proximity to WinB, small cloud penalty."""
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
    best, best_score = None, np.inf
    for c in candidates:
        sc = score_candidate(c["row"], c["target"], planet_dt)
        if sc < best_score:
            best, best_score = (c["row"], c["target"], sc), sc
    return best

# ---------------- MAIN ----------------

def main():
    # Decide output file name (avoid overwriting during tests)
    OUT_CSV = DEFAULT_OUT if args.out_suffix == "" else OUTDIR / f"windowA_s2_matches_scored{args.out_suffix}.csv"

    logger.info("Loading Planet chip CSV…")
    df = pd.read_csv(PLANET_CSV)

    # required columns
    req = ["window_b", "country", "planet_date", "x", "y"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"CSV missing required columns {miss}")

    # chip_id from window_b filename
    df["chip_id"] = df["window_b"].astype(str).apply(lambda p: Path(p).stem)

    # parse date
    df["planet_date"] = pd.to_datetime(df["planet_date"], errors="coerce")
    df = df.dropna(subset=["planet_date", "x", "y", "country"]).copy()

    # --- optional smoke-test filters ---
    if args.countries:
        keep_countries = {c.strip().upper() for c in args.countries.split(",") if c.strip()}
        before = len(df)
        df = df[df["country"].str.upper().isin(keep_countries)].copy()
        logger.info(f"Country filter: kept {len(df)}/{before} rows ({sorted(keep_countries)})")

    if args.months:
        keep_months = {int(m.strip()) for m in args.months.split(",") if m.strip()}
        before = len(df)
        df = df[df["planet_date"].dt.month.isin(keep_months)].copy()
        logger.info(f"Month filter: kept {len(df)}/{before} rows (months={sorted(keep_months)})")

    if args.sample_per_country and args.sample_per_country > 0:
        before = len(df)
        sampled = []
        for c, g in df.groupby(df["country"].str.upper()):
            sampled.append(g if len(g) <= args.sample_per_country else g.sample(args.sample_per_country, random_state=42))
        df = pd.concat(sampled, ignore_index=True)
        logger.info(f"Per-country sample: kept {len(df)}/{before} rows (n={args.sample_per_country} each)")

    if args.max_total and args.max_total > 0 and len(df) > args.max_total:
        df = df.sample(args.max_total, random_state=42).reset_index(drop=True)
        logger.info(f"Total cap: trimmed to {len(df)} rows")

    # ----- RESUME -----
    if OUT_CSV.exists():
        try:
            done = pd.read_csv(OUT_CSV, usecols=["chip_id"])
            processed = set(done["chip_id"].astype(str))
            before = len(df)
            df = df[~df["chip_id"].astype(str).isin(processed)].copy()
            logger.info(f"Resume: {before - len(df)} chips already in {OUT_CSV}, {len(df)} remaining.")
        except Exception as e:
            logger.info(f"{OUT_CSV} exists but unreadable or missing 'chip_id'; will overwrite at end. ({e})")

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
            assert s2_dt < planet_dt, f"{chip_id}: WinA ({s2_dt}) is not earlier than WinB ({planet_dt})"
            abs_days = (planet_dt - s2_dt).total_seconds() / 86400.0

            new_results.append({
                "chip_id": chip_id, "country": country, "planet_date": planet_dt,
                "season_label": season_label, "tier_used": tier_used, "cloud_limit": cloud_limit,
                "s2_id": match_row["s2_id"], "s2_datetime": s2_dt, "s2_cloud": float(match_row["s2_cloud"]),
                "abs_days": abs_days, "score": match_score, "status": "ok",
            })

    # ----- WRITE / APPEND OUTPUT -----
    new_df = pd.DataFrame(new_results)

    if (args.out_suffix == "") and DEFAULT_OUT.exists():
        # Overwrite default file for full runs
        new_df.to_csv(DEFAULT_OUT, index=False)
        logger.info(f"Wrote {len(new_df)} rows to {DEFAULT_OUT}")
        out_path = DEFAULT_OUT
    else:
        # Append or write to test file
        if (OUTDIR / f"windowA_s2_matches_scored{args.out_suffix}.csv").exists():
            new_df.to_csv(OUTDIR / f"windowA_s2_matches_scored{args.out_suffix}.csv", mode="a", header=False, index=False)
            logger.info(f"Appended {len(new_df)} rows to {OUTDIR / f'windowA_s2_matches_scored{args.out_suffix}.csv'}")
        else:
            new_df.to_csv(OUTDIR / f"windowA_s2_matches_scored{args.out_suffix}.csv", index=False)
            logger.info(f"Wrote {len(new_df)} rows to {OUTDIR / f'windowA_s2_matches_scored{args.out_suffix}.csv'}")
        out_path = OUTDIR / f"windowA_s2_matches_scored{args.out_suffix}.csv"

    # ---- Post-run validation ----
    try:
        out_df = pd.read_csv(out_path, parse_dates=["planet_date","s2_datetime"])
        ok_rows = out_df[out_df["status"] == "ok"].copy()
        total_ok = len(ok_rows)
        bad = ok_rows[ok_rows["s2_datetime"] >= ok_rows["planet_date"]]
        logger.info(f"Validation: ok rows={total_ok}, violations (WinA>=WinB)={len(bad)}")
        if not bad.empty:
            logger.warning("Found rows with WinA>=WinB; first few:")
            logger.warning(bad.head(10).to_string())
        else:
            logger.info("All ok rows satisfy WinA < WinB ✅")

        if total_ok > 0:
            ok_rows["abs_days"] = (ok_rows["planet_date"] - ok_rows["s2_datetime"]).dt.days
            logger.info("Abs-day gap summary (WinB - WinA, days):")
            logger.info(ok_rows["abs_days"].describe().to_string())
    except Exception as e:
        logger.info(f"Validation skipped (could not load output): {e}")

    logger.info("Done.")

if __name__ == "__main__":
    main()
