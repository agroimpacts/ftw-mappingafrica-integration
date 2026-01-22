# #!/usr/bin/env python3
# """
# windowA_s2_matches_scored.py

# Find Sentinel-2 matches for "window A" (seasonal pair) for each Planet chip
# using seasonal pairing rules (incl. transition months), multi-tier cloud
# thresholds, and a simple scoring (temporal distance + cloud cover).

# - Accepts a Planet catalog CSV where the Planet chip path is in column `window_b`
#   (we derive chip_id from the filename stem). If your CSV already has `chip_id`,
#   that's fine too.
# - Requires columns: country, planet_date (ISO), x, y.
# - Outputs a CSV with chosen S2 item per chip (or status explaining why none).

# Author: you
# """

# from __future__ import annotations

# from pathlib import Path
# import logging
# import math
# import time
# from datetime import datetime, timedelta

# import numpy as np
# import pandas as pd
# from tqdm import tqdm

# from pystac_client import Client
# from pystac_client.exceptions import APIError
# import planetary_computer as pc


# # ---------------- CONFIG ----------------

# BASE = Path("/home/airg/gessuman/projects/ftw-mappingafrica-integration")

# # Planet catalog CSV. Must contain: country, planet_date, x, y, and either
# # `chip_id` or `window_b` (path to the Planet chip).
# PLANET_CSV = BASE / "data" / "mappingafrica-catalog-dated.csv"

# OUTDIR = BASE / "data" / "windowA"
# OUTDIR.mkdir(parents=True, exist_ok=True)
# OUT_CSV = OUTDIR / "windowA_s2_matches_scored.csv"
# LOGFILE = OUTDIR / "windowA_s2_matches_scored.log"

# STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
# S2_COLLECTION = "sentinel-2-l2a"

# # Cloud tiers (% max cover). We try in this order and stop on first tier with a match.
# CLOUD_THRESHOLDS = [5, 10, 15, 20, 30, 40]

# MAX_ITEMS_PER_QUERY = 100  # safeguard / page limit
# MAX_RETRIES = 4            # retry for STAC timeouts
# BACKOFF_BASE = 1.7

# # Scoring weights: lower is better
# #   score = w_days * abs_days + w_cloud * cloud_pct
# W_DAYS = 1.0
# W_CLOUD = 0.5


# # ---------------- LOGGING ----------------
# logger = logging.getLogger("windowA_s2_scored")
# logger.setLevel(logging.INFO)
# logger.handlers.clear()
# fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
# sh = logging.StreamHandler()
# sh.setFormatter(fmt)
# logger.addHandler(sh)
# fh = logging.FileHandler(LOGFILE)
# fh.setFormatter(fmt)
# logger.addHandler(fh)


# # ---------------- UTIL ----------------

# def to_naive_utc(dt) -> datetime:
#     """Return a timezone-naive UTC datetime."""
#     ts = pd.to_datetime(dt, utc=True)
#     # ts is tz-aware UTC; convert to python datetime and drop tzinfo
#     return ts.to_pydatetime().replace(tzinfo=None)


# def month_label(m: int) -> str:
#     return datetime(2001, m, 1).strftime("%b")  # Jan, Feb, ...


# # ---------------- SEASONAL RULES ----------------
# # Region map: which ISO2 country code goes to which region rule
# REGION_MAP = {
#     # 1) West Africa – Sudano-Sahel
#     "sudano_sahel": ["GH", "NG", "ML", "NE", "SD", "TD"],

#     # 2) West Africa – Coastal/Forest Zone
#     "coastal_forest": ["CI", "TG", "BJ"],

#     # 3) East Africa – Equatorial Bimodal
#     "bimodal": ["UG", "KE", "RW", "BI", "TZ"],

#     # 4) East Africa – Unimodal Highlands / Ethiopia
#     "ethiopia": ["ET"],

#     # 5) Southern Africa – Summer rainfall
#     "southern": ["ZM", "ZW", "MZ", "ZA", "MW", "BW", "NA"],
# }

# def which_region(country_code: str) -> str | None:
#     cc = str(country_code).upper()
#     for region_name, codes in REGION_MAP.items():
#         if cc in codes:
#             return region_name
#     return None

# def in_month_range(month: int, start: int, end: int) -> bool:
#     """Inclusive, with wrap support (e.g., Nov–Feb)."""
#     if start <= end:
#         return start <= month <= end
#     return month >= start or month <= end

# def pick_target_months_extended(region: str, m: int) -> tuple[list[int], str, str]:
#     """
#     Given region and Planet month m (1–12), return:
#       (target_months, label, rule_id)

#     We include transition months noted during QA:
#       - Sudano-Sahel: also Oct (targets Dec–Feb)
#       - Coastal/Forest: also May (targets Sep–Nov)
#       - Bimodal: also Feb (targets Jun–Jul)
#       - Ethiopia (highlands): also Oct (targets Dec–Feb)
#       - Southern: also Sep (targets Jan–Feb)
#     """
#     # Sudano-Sahel
#     if region == "sudano_sahel":
#         if m in (8, 9, 10):                 # Aug–Sep (+Oct)
#             return [12, 1, 2], "Dec–Feb", "sudano_aug-sep(+oct)->dec-feb"
#         if m in (2, 3):                      # Feb–Mar
#             return [8, 9, 10], "Aug–Oct", "sudano_feb-mar->aug-oct"
#         return [], "no_pair_for_month", "no_rule"

#     # Coastal/Forest
#     if region == "coastal_forest":
#         if m in (3, 4, 5):                   # Mar–Apr (+May)
#             return [9, 10, 11], "Sep–Nov", "coastal_mar-apr(+may)->sep-nov"
#         if m in (10, 11):                    # Oct–Nov
#             return [2, 3], "Feb–Mar", "coastal_oct-nov->feb-mar"
#         return [], "no_pair_for_month", "no_rule"

#     # Bimodal
#     if region == "bimodal":
#         if m in (3, 4, 5) or m == 2:         # Mar–May (+Feb)
#             return [6, 7], "Jun–Jul", "bimodal_feb-mar-may->jun-jul"
#         if m in (9, 10, 11):                 # Sep–Nov
#             return [1, 2], "Jan–Feb", "bimodal_sep-nov->jan-feb"
#         return [], "no_pair_for_month", "no_rule"

#     # Ethiopia (highlands)
#     if region == "ethiopia":
#         if m in (8, 9, 10):                  # Aug–Sep (+Oct)
#             return [12, 1, 2], "Dec–Feb", "ethiopia_aug-sep(+oct)->dec-feb"
#         if m in (1, 2, 3):                   # Jan–Mar
#             return [8, 9], "Aug–Sep", "ethiopia_jan-mar->aug-sep"
#         return [], "no_pair_for_month", "no_rule"

#     # Southern Africa
#     if region == "southern":
#         if m in (1, 2):                      # Jan–Feb
#             return [7, 8], "Jul–Aug", "southern_jan-feb->jul-aug"
#         if m in (8, 9):                       # Aug (+Sep)
#             return [1, 2], "Jan–Feb", "southern_aug(+sep)->jan-feb"
#         return [], "no_pair_for_month", "no_rule"

#     return [], "no_region_rule", "no_region_rule"

# def build_date_ranges(planet_dt: datetime, target_months: list[int]) -> list[tuple[str, str, dict]]:
#     """
#     For each target month (e.g., [12,1,2]) build a (start_iso, end_iso, meta)
#     where meta = {"center": center_datetime} (naive UTC).

#     We assign year by "closest direction" logic (future vs past) relative to planet_dt.
#     Ranges of consecutive months are merged.
#     """
#     pm, py = planet_dt.month, planet_dt.year

#     # Assign each target month a year
#     assigned: list[tuple[int, int]] = []  # (year, month)
#     for tm in target_months:
#         fwd = (tm - pm) % 12
#         back = (pm - tm) % 12
#         if fwd <= back:  # future
#             year = py if tm >= pm else py + 1
#         else:            # past
#             year = py if tm <= pm else py - 1
#         assigned.append((year, tm))

#     # Create month windows and a mid-month center
#     windows: list[tuple[datetime, datetime, datetime]] = []
#     for (y, m) in assigned:
#         start = datetime(y, m, 1)
#         if m == 12:
#             end_next = datetime(y + 1, 1, 1)
#         else:
#             end_next = datetime(y, m + 1, 1)
#         end = end_next - timedelta(seconds=1)
#         center = start + (end - start) / 2
#         windows.append((start, end, center))

#     # Merge consecutive touching windows
#     windows.sort(key=lambda t: t[0])
#     merged: list[tuple[datetime, datetime, list[datetime]]] = []
#     for st, ed, ctr in windows:
#         if not merged:
#             merged.append((st, ed, [ctr]))
#         else:
#             lst_st, lst_ed, ctrs = merged[-1]
#             if st <= lst_ed + timedelta(days=1):
#                 # extend
#                 if ed > lst_ed:
#                     merged[-1] = (lst_st, ed, ctrs + [ctr])
#                 else:
#                     merged[-1] = (lst_st, lst_ed, ctrs + [ctr])
#             else:
#                 merged.append((st, ed, [ctr]))

#     # Emit string ranges + center as the mean of centers in the merged block
#     out_ranges: list[tuple[str, str, dict]] = []
#     for st, ed, ctrs in merged:
#         center = ctrs[0]
#         if len(ctrs) > 1:
#             # average datetimes
#             secs = [c.timestamp() for c in ctrs]
#             center = datetime.utcfromtimestamp(sum(secs) / len(secs))
#         out_ranges.append((st.strftime("%Y-%m-%d"), ed.strftime("%Y-%m-%d"),
#                            {"center": center}))
#     return out_ranges


# # ---------------- STAC QUERY + SCORING ----------------

# def signed_asset(item, key: str) -> str:
#     return pc.sign(item.assets[key].href)

# def query_s2_items(client: Client, lon: float, lat: float,
#                    start_date: str, end_date: str,
#                    cloud_max: float,
#                    max_items: int = MAX_ITEMS_PER_QUERY) -> list[dict]:
#     """
#     Return list of {s2_id, s2_datetime (naive UTC), s2_cloud}.
#     """
#     params = dict(
#         collections=[S2_COLLECTION],
#         intersects={"type": "Point", "coordinates": [lon, lat]},
#         datetime=f"{start_date}/{end_date}",
#         query={"eo:cloud_cover": {"lte": cloud_max}},
#         limit=max_items,
#         max_items=max_items,
#     )
#     attempt = 0
#     while True:
#         try:
#             search = client.search(**params)
#             items = list(search.items())
#             out = []
#             for it in items:
#                 out.append({
#                     "s2_id": it.id,
#                     "s2_datetime": to_naive_utc(it.datetime),   # <-- normalize here
#                     "s2_cloud": it.properties.get("eo:cloud_cover", np.nan),
#                 })
#             return out
#         except APIError as e:
#             attempt += 1
#             if attempt >= MAX_RETRIES:
#                 raise
#             sleep_s = BACKOFF_BASE ** attempt
#             logger.info(f"STAC timeout, retry {attempt}/{MAX_RETRIES-1} in {sleep_s:.1f}s … ({e})")
#             time.sleep(sleep_s)

# def score_candidate(row: dict, target_meta: dict, planet_dt: datetime) -> float:
#     """
#     Lower is better. Combine absolute day distance to the target center
#     (NOT planet_dt) and cloud cover, both in naive UTC.
#     """
#     # all naive UTC now
#     center = to_naive_utc(target_meta["center"])
#     s2_time = to_naive_utc(row["s2_datetime"])
#     days = abs((s2_time - center).total_seconds()) / 86400.0
#     cloud = float(row["s2_cloud"]) if pd.notnull(row["s2_cloud"]) else 100.0
#     return W_DAYS * days + W_CLOUD * cloud

# def pick_best_scored(candidates: list[dict], target_meta: dict, planet_dt: datetime) -> dict | None:
#     if not candidates:
#         return None
#     best = None
#     best_score = math.inf
#     for c in candidates:
#         if pd.isna(c.get("s2_cloud")):
#             continue
#         s = score_candidate(c, target_meta, planet_dt)
#         if s < best_score:
#             best_score = s
#             best = dict(c)
#             best["__score"] = s
#     return best


# # ---------------- MAIN ----------------

# def main():
#     logger.info("Loading Planet chip CSV…")
#     df = pd.read_csv(PLANET_CSV)

#     # Standardize columns
#     have_chip_id = "chip_id" in df.columns
#     if not have_chip_id and "window_b" in df.columns:
#         # derive chip_id from filename stem
#         df["chip_id"] = df["window_b"].astype(str).apply(lambda p: Path(p).stem)
#         have_chip_id = True

#     required = ["chip_id", "country", "planet_date", "x", "y"]
#     missing = [c for c in required if c not in df.columns]
#     if missing:
#         raise ValueError(f"CSV missing required columns {missing}")

#     # Coerce types
#     df["planet_date"] = pd.to_datetime(df["planet_date"], errors="coerce", utc=True)
#     df = df.dropna(subset=["planet_date", "x", "y", "country"]).copy()

#     logger.info(f"Total chips to process: {len(df)}")

#     client = Client.open(STAC_URL)

#     # Results
#     rows_out = []

#     it = tqdm(df.itertuples(index=False), total=len(df), desc="Scored windowA matches", unit="chip")
#     for r in it:
#         chip_id = str(getattr(r, "chip_id"))
#         country = str(getattr(r, "country")).upper()
#         planet_dt = to_naive_utc(getattr(r, "planet_date"))
#         lon = float(getattr(r, "x"))
#         lat = float(getattr(r, "y"))

#         reg = which_region(country)
#         if reg is None:
#             rows_out.append({
#                 "chip_id": chip_id, "country": country, "planet_date": planet_dt,
#                 "season_label": "no_region_rule", "rule_id": "no_region_rule",
#                 "tier_used": None, "cloud_limit": None,
#                 "s2_id": None, "s2_datetime": None, "s2_cloud": None,
#                 "abs_days": None, "score": None, "status": "no_region_rule"
#             })
#             continue

#         target_months, season_label, rule_id = pick_target_months_extended(reg, planet_dt.month)
#         if not target_months:
#             rows_out.append({
#                 "chip_id": chip_id, "country": country, "planet_date": planet_dt,
#                 "season_label": season_label, "rule_id": rule_id,
#                 "tier_used": None, "cloud_limit": None,
#                 "s2_id": None, "s2_datetime": None, "s2_cloud": None,
#                 "abs_days": None, "score": None, "status": "no_pair_for_month"
#             })
#             continue

#         ranges = build_date_ranges(planet_dt, target_months)

#         best_overall = None
#         tier_used = None
#         cloud_limit = None

#         # Try cloud tiers in order; keep best within the first tier that returns any candidates
#         for t_idx, cloud_max in enumerate(CLOUD_THRESHOLDS, start=1):
#             candidates_this_tier = []
#             # Query each merged season range
#             for (start_iso, end_iso, meta) in ranges:
#                 try:
#                     cand = query_s2_items(client, lon, lat, start_iso, end_iso, cloud_max)
#                     # Attach the target metadata for scoring
#                     for c in cand:
#                         candidates_this_tier.append({"row": c, "target": meta})
#                 except Exception as e:
#                     logger.info(f"{chip_id}: STAC error @tier{t_idx} ({cloud_max}%) "
#                                 f"{start_iso}..{end_iso}: {e}")

#             if not candidates_this_tier:
#                 # no candidates at this tier—try next tier
#                 continue

#             # Score within this tier; pick best
#             scored_best = None
#             scored_best_score = math.inf
#             for c in candidates_this_tier:
#                 sc = score_candidate(c["row"], c["target"], planet_dt)
#                 if sc < scored_best_score:
#                     scored_best_score = sc
#                     scored_best = dict(c["row"])
#                     scored_best["__score"] = sc
#                     # add abs_days relative to target center for reporting
#                     center = to_naive_utc(c["target"]["center"])
#                     scored_best["__abs_days"] = abs(
#                         (to_naive_utc(c["row"]["s2_datetime"]) - center).total_seconds()
#                     ) / 86400.0

#             best_overall = scored_best
#             tier_used = f"tier{t_idx}_le{cloud_max}"
#             cloud_limit = cloud_max
#             break  # stop at the first tier that returns any candidates

#         if best_overall is None:
#             rows_out.append({
#                 "chip_id": chip_id, "country": country, "planet_date": planet_dt,
#                 "season_label": season_label, "rule_id": rule_id,
#                 "tier_used": None, "cloud_limit": None,
#                 "s2_id": None, "s2_datetime": None, "s2_cloud": None,
#                 "abs_days": None, "score": None, "status": "no_match_all_tiers"
#             })
#         else:
#             s2_time = to_naive_utc(best_overall["s2_datetime"])
#             rows_out.append({
#                 "chip_id": chip_id, "country": country, "planet_date": planet_dt,
#                 "season_label": season_label, "rule_id": rule_id,
#                 "tier_used": tier_used, "cloud_limit": cloud_limit,
#                 "s2_id": best_overall["s2_id"],
#                 "s2_datetime": s2_time,
#                 "s2_cloud": float(best_overall["s2_cloud"]),
#                 "abs_days": float(best_overall.get("__abs_days", np.nan)),
#                 "score": float(best_overall.get("__score", np.nan)),
#                 "status": "ok"
#             })

#     out_df = pd.DataFrame(rows_out)
#     out_df.to_csv(OUT_CSV, index=False)
#     logger.info(f"Saved matches → {OUT_CSV}")


# if __name__ == "__main__":
#     main()





# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------

# #!/usr/bin/env python3
# """
# windowA_s2_matches_scored.py

# Find Sentinel-2 matches for "window A" (seasonal pair) for each Planet chip,
# using seasonal pairing rules, multi-tier cloud thresholds, and a simple score
# (time difference + cloud cover).

# Supports RESUME mode:
# - If the output CSV already exists, load it and skip chip_ids that are
#   already processed, then APPEND only the new results.

# Output:
#   /home/airg/gessuman/projects/ftw-mappingafrica-integration/data/windowA/windowA_s2_matches_scored.csv
# """

# from pathlib import Path
# import logging
# import time
# from datetime import datetime, timedelta

# import numpy as np
# import pandas as pd
# from tqdm import tqdm

# from pystac_client import Client
# from pystac_client.exceptions import APIError
# import planetary_computer as pc


# # ---------------- CONFIG ----------------

# BASE = Path("/home/airg/gessuman/projects/ftw-mappingafrica-integration")

# # Planet catalog with window B metadata
# # Must have: window_b, country, planet_date, x, y
# PLANET_CSV = BASE / "data" / "mappingafrica-catalog-dated.csv"

# OUTDIR = BASE / "data" / "windowA"
# OUTDIR.mkdir(parents=True, exist_ok=True)
# OUT_CSV = OUTDIR / "windowA_s2_matches_scored.csv"
# LOGFILE = OUTDIR / "windowA_matches_scored.log"

# STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
# S2_COLLECTION = "sentinel-2-l2a"

# # Cloud tiers (%)
# CLOUD_THRESHOLDS = [5, 10, 15, 20, 30, 40]

# MAX_ITEMS_PER_QUERY = 100
# MAX_RETRIES = 4
# BACKOFF_BASE = 1.7


# # ---------------- LOGGING ----------------

# logger = logging.getLogger("windowA_s2_scored")
# logger.setLevel(logging.INFO)
# logger.handlers.clear()
# _fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
# _sh = logging.StreamHandler()
# _sh.setFormatter(_fmt)
# logger.addHandler(_sh)
# _fh = logging.FileHandler(LOGFILE)
# _fh.setFormatter(_fmt)
# logger.addHandler(_fh)


# # ---------------- HELPER: TZ-HANDLING ----------------

# def ensure_naive(dt):
#     """Return a timezone-naive Python datetime from a pandas Timestamp or datetime."""
#     if isinstance(dt, pd.Timestamp):
#         dt = dt.to_pydatetime()
#     if hasattr(dt, "tzinfo") and dt.tzinfo is not None:
#         dt = dt.replace(tzinfo=None)
#     return dt


# # ---------------- SEASONAL REGIONS & RULES ----------------

# def in_month_range(month, start, end):
#     """Inclusive month range, allowing wrap (e.g. 11–2)."""
#     if start <= end:
#         return start <= month <= end
#     else:
#         return (month >= start) or (month <= end)


# # Region definitions: which countries map to which seasonal rule-group
# REGION_MAP = {
#     # Region 1: West Africa – Sudano-Sahel
#     "sudano_sahel": [
#         "GH", "NG", "ML", "NE", "SD", "TD",
#         "BF", "MR", "SN",
#     ],

#     # Region 2: West Africa – Coastal/Forest Zone
#     "coastal_forest": [
#         "CI", "TG", "BJ",
#         "GM", "GN", "GW", "SL", "GA", "CM", "CG", "CD",
#     ],

#     # Region 3: East Africa – Equatorial Bimodal
#     "bimodal": [
#         "UG", "KE", "RW", "BI", "TZ",
#         "CF", "SO",
#     ],

#     # Region 4: East Africa – Unimodal Highlands / Ethiopia
#     "ethiopia": [
#         "ET", "ER",
#     ],

#     # Region 5: Southern Africa – Summer rainfall
#     "southern": [
#         "ZM", "ZW", "MZ", "ZA", "MW", "BW", "NA",
#         "AO", "LS", "SZ",
#     ],
# }

# For each region, encode the original pairing rules in terms of
# (source months) -> (target months).
# REGION_RULES = {
#     # ---------- Region 1: West Africa – Sudano-Sahel ----------
#     # Original:
#     #   If chip in Aug–Sep pair with Dec–Feb
#     #   If chip in Feb–Mar pair with Aug–Oct
#     "sudano_sahel": [
#         {"months": [8, 9], "target": [12, 1, 2], "label": "Aug–Sep→Dec–Feb"},
#         {"months": [2, 3], "target": [8, 9, 10], "label": "Feb–Mar→Aug–Oct"},
#     ],

#     # ---------- Region 2: West Africa – Coastal/Forest Zone ----------
#     # Original:
#     #   If chip in Mar–Apr pair with Sep–Nov
#     #   If chip in Oct–Nov pair with Feb–Mar
#     "coastal_forest": [
#         {"months": [3, 4], "target": [9, 10, 11], "label": "Mar–Apr→Sep–Nov"},
#         {"months": [10, 11], "target": [2, 3], "label": "Oct–Nov→Feb–Mar"},
#     ],

#     # ---------- Region 3: East Africa – Equatorial Bimodal ----------
#     # Original:
#     #   If chip in Mar–May pair with Jun–Jul
#     #   If chip in Sep–Nov pair with Jan–Feb
#     "bimodal": [
#         {"months": [3, 4, 5], "target": [6, 7], "label": "Mar–May→Jun–Jul"},
#         {"months": [9, 10, 11], "target": [1, 2], "label": "Sep–Nov→Jan–Feb"},
#     ],

#     # ---------- Region 4: East Africa – Unimodal Highlands / Ethiopia ----------
#     # Original:
#     #   If chip in Aug–Sep pair with Dec–Feb
#     #   If chip in Jan–Mar pair with Aug–Sep
#     "ethiopia": [
#         {"months": [8, 9], "target": [12, 1, 2], "label": "Aug–Sep→Dec–Feb"},
#         {"months": [1, 2, 3], "target": [8, 9], "label": "Jan–Mar→Aug–Sep"},
#     ],

#     # ---------- Region 5: Southern Africa – Summer rainfall ----------
#     # Original:
#     #   If chip in Jan–Feb pair with Jul–Aug
#     #   If chip in Aug pair with Jan–Feb
#     "southern": [
#         {"months": [1, 2], "target": [7, 8], "label": "Jan–Feb→Jul–Aug"},
#         {"months": [8], "target": [1, 2], "label": "Aug→Jan–Feb"},
#     ],
# }

# def which_region(country_code: str):
#     cc = country_code.upper()
#     for reg, codes in REGION_MAP.items():
#         if cc in codes:
#             return reg
#     return None


# def pick_target_months(region_name: str, planet_month: int):
#     """
#     Given the region and the Planet chip's month (1–12), return:
#         (target_months, label)

#     Behaviour:
#     - First, use the original rule windows exactly (e.g. Aug–Sep → Dec–Feb).
#     - If the month is not covered by any original rule, assign it to the
#       **nearest** rule for that region in circular month space and return that
#       rule's target months, with the label suffixed with "_extended".
#     """

#     rules = REGION_RULES.get(region_name)
#     if not rules:
#         # Region itself is unknown
#         return ([], "no_region_rule")

#     m = int(planet_month)

#     # ---------- 1) Exact match: original rules ----------
#     for r in rules:
#         if m in r["months"]:
#             return (r["target"], r["label"])

#     # ---------- 2) Fallback: nearest rule in month space ----------
#     def month_distance(a: int, b: int) -> int:
#         """Circular distance on months 1..12."""
#         d = abs(a - b)
#         return min(d, 12 - d)

#     best_rule = None
#     best_dist = 999

#     for r in rules:
#         # distance from this month to the closest month in this rule
#         d = min(month_distance(m, mm) for mm in r["months"])
#         if d < best_dist:
#             best_dist = d
#             best_rule = r

#     if best_rule is not None:
#         # We’re using a neighbouring season’s rule
#         return (best_rule["target"], best_rule["label"] + "_extended")

#     # Shouldn’t happen, but keep a safe fallback
#     return ([], "no_rule")


# def build_season_targets(planet_dt: datetime, target_months):
#     """
#     Build list of seasonal target windows for S2 search.

#     Returns a list of dicts:
#       { "start": datetime, "end": datetime, "center": datetime }
#     All datetimes are timezone-naive and in calendar years consistent
#     with the Planet chip date.
#     """
#     pm = planet_dt.month
#     py = planet_dt.year

#     windows = []
#     for tm in target_months:
#         # Decide year assignment based on "closest" direction
#         forward = (tm - pm) % 12
#         backward = (pm - tm) % 12
#         if forward <= backward:
#             year = py if tm >= pm else py + 1
#         else:
#             year = py if tm <= pm else py - 1

#         start = datetime(year, tm, 1)
#         if tm == 12:
#             end = datetime(year + 1, 1, 1) - timedelta(seconds=1)
#         else:
#             end = datetime(year, tm + 1, 1) - timedelta(seconds=1)

#         center = start + (end - start) / 2
#         windows.append((start, end, center))

#     # merge overlapping / touching windows
#     windows.sort(key=lambda x: x[0])
#     merged = []
#     for st, ed, ce in windows:
#         if not merged:
#             merged.append([st, ed, ce])
#         else:
#             mst, med, mce = merged[-1]
#             if st <= med + timedelta(days=1):
#                 # merge
#                 new_st = mst
#                 new_ed = max(med, ed)
#                 new_ce = new_st + (new_ed - new_st) / 2
#                 merged[-1] = [new_st, new_ed, new_ce]
#             else:
#                 merged.append([st, ed, ce])

#     targets = []
#     for st, ed, ce in merged:
#         targets.append(
#             {"start": ensure_naive(st), "end": ensure_naive(ed), "center": ensure_naive(ce)}
#         )
#     return targets


# # ---------------- STAC & SCORING ----------------

# def query_s2_items(client, lon, lat, start_date, end_date, cloud_max, max_items=MAX_ITEMS_PER_QUERY):
#     """
#     Query Sentinel-2 L2A items intersecting a point for [start_date, end_date],
#     with eo:cloud_cover <= cloud_max.
#     Returns list of row dicts.
#     """
#     params = dict(
#         collections=[S2_COLLECTION],
#         intersects={"type": "Point", "coordinates": [lon, lat]},
#         datetime=f"{start_date}/{end_date}",
#         query={"eo:cloud_cover": {"lte": cloud_max}},
#         limit=max_items,
#         max_items=max_items,
#     )

#     attempt = 0
#     while True:
#         try:
#             search = client.search(**params)
#             items = list(search.items())
#             rows = []
#             for it in items:
#                 dt = it.datetime
#                 dt = ensure_naive(pd.to_datetime(dt))
#                 rows.append(
#                     {
#                         "s2_id": it.id,
#                         "s2_datetime": dt,
#                         "s2_cloud": float(it.properties.get("eo:cloud_cover", np.nan)),
#                     }
#                 )
#             return rows
#         except APIError as e:
#             attempt += 1
#             if attempt >= MAX_RETRIES:
#                 raise
#             sleep_s = BACKOFF_BASE ** attempt
#             logger.info(
#                 f"STAC timeout, retry {attempt}/{MAX_RETRIES-1} in {sleep_s:.1f}s … ({e})"
#             )
#             time.sleep(sleep_s)


# def score_candidate(row, target, planet_dt):
#     """
#     Score a single candidate row against a given target seasonal window.
#     Lower is better.

#     Components:
#       - |S2 date − target center|  (primary seasonal alignment)
#       - |S2 date − Planet date|   (secondary)
#       - 0.05 * cloud cover        (penalty)
#     """
#     s2_dt = ensure_naive(row["s2_datetime"])
#     center = ensure_naive(target["center"])
#     planet_dt = ensure_naive(planet_dt)

#     days_center = abs((s2_dt - center).total_seconds()) / 86400.0
#     days_planet = abs((s2_dt - planet_dt).total_seconds()) / 86400.0
#     cloud = row["s2_cloud"] if np.isfinite(row["s2_cloud"]) else 100.0

#     score = days_center + 0.5 * days_planet + 0.05 * cloud
#     return score


# def pick_best_scored(candidates, planet_dt):
#     """
#     candidates: list of dicts {"row": row_dict, "target": target_dict}
#     Returns best (row, target, score) or None.
#     """
#     if not candidates:
#         return None

#     best = None
#     best_score = np.inf
#     for c in candidates:
#         row = c["row"]
#         target = c["target"]
#         sc = score_candidate(row, target, planet_dt)
#         if sc < best_score:
#             best_score = sc
#             best = (row, target, sc)
#     return best


# # ---------------- MAIN ----------------

# def main():
#     logger.info("Loading Planet chip CSV…")
#     df = pd.read_csv(PLANET_CSV)

#     # Basic checks
#     req_cols = ["window_b", "country", "planet_date", "x", "y"]
#     missing = [c for c in req_cols if c not in df.columns]
#     if missing:
#         raise ValueError(f"CSV missing required columns {missing}")

#     # Derive chip_id from window_b filename
#     df["chip_id"] = df["window_b"].astype(str).apply(lambda p: Path(p).stem)

#     # Parse date
#     df["planet_date"] = pd.to_datetime(df["planet_date"], errors="coerce")
#     df = df.dropna(subset=["planet_date", "x", "y", "country"]).copy()

#     # ----- RESUME LOGIC -----
#     if OUT_CSV.exists():
#         done = pd.read_csv(OUT_CSV)
#         if "chip_id" in done.columns:
#             processed = set(done["chip_id"].astype(str))
#             before = len(df)
#             df = df[~df["chip_id"].astype(str).isin(processed)].copy()
#             logger.info(
#                 f"Resume mode: {before - len(df)} chips already in {OUT_CSV}, "
#                 f"{len(df)} remaining to process."
#             )
#         else:
#             logger.info(
#                 f"{OUT_CSV} exists but has no 'chip_id' column; running from scratch "
#                 f"and overwriting at the end."
#             )
#     else:
#         logger.info(f"No existing {OUT_CSV}; running from scratch.")

#     if df.empty:
#         logger.info("No chips left to process. Exiting.")
#         return

#     logger.info(f"Total chips to process this run: {len(df)}")

#     client = Client.open(STAC_URL)

#     new_results = []

#     for _, row in tqdm(
#         df.iterrows(), total=len(df), desc="Scored windowA matches", unit="chip"
#     ):
#         chip_id = str(row["chip_id"])
#         country = str(row["country"]).upper()
#         planet_dt = ensure_naive(row["planet_date"])
#         lon = float(row["x"])
#         lat = float(row["y"])

#         region = which_region(country)
#         if region is None:
#             new_results.append(
#                 {
#                     "chip_id": chip_id,
#                     "country": country,
#                     "planet_date": planet_dt,
#                     "season_label": "no_region",
#                     "tier_used": None,
#                     "cloud_limit": None,
#                     "s2_id": None,
#                     "s2_datetime": None,
#                     "s2_cloud": None,
#                     "abs_days": None,
#                     "score": None,
#                     "status": "no_region_rule",
#                 }
#             )
#             continue

#         target_months, season_label = pick_target_months(region, planet_dt.month)
#         if not target_months:
#             new_results.append(
#                 {
#                     "chip_id": chip_id,
#                     "country": country,
#                     "planet_date": planet_dt,
#                     "season_label": season_label,
#                     "tier_used": None,
#                     "cloud_limit": None,
#                     "s2_id": None,
#                     "s2_datetime": None,
#                     "s2_cloud": None,
#                     "abs_days": None,
#                     "score": None,
#                     "status": "no_pair_for_month",
#                 }
#             )
#             continue

#         targets = build_season_targets(planet_dt, target_months)

#         match_row = None
#         match_target = None
#         match_score = None
#         tier_used = None
#         cloud_limit = None

#         # multi-tier search
#         for t_idx, cloud_max in enumerate(CLOUD_THRESHOLDS, start=1):
#             tier_label = f"tier{t_idx}_le{cloud_max}"
#             tier_candidates = []

#             for t in targets:
#                 start_iso = t["start"].strftime("%Y-%m-%d")
#                 end_iso = t["end"].strftime("%Y-%m-%d")
#                 try:
#                     rows = query_s2_items(
#                         client, lon, lat, start_iso, end_iso, cloud_max
#                     )
#                     for r in rows:
#                         tier_candidates.append({"row": r, "target": t})
#                 except Exception as e:
#                     logger.info(
#                         f"{chip_id}: STAC error tier {tier_label} "
#                         f"{start_iso}..{end_iso}: {e}"
#                     )

#             best = pick_best_scored(tier_candidates, planet_dt)
#             if best is not None:
#                 r, tg, sc = best
#                 match_row = r
#                 match_target = tg
#                 match_score = sc
#                 tier_used = tier_label
#                 cloud_limit = cloud_max
#                 break  # done with this chip

#         if match_row is None:
#             new_results.append(
#                 {
#                     "chip_id": chip_id,
#                     "country": country,
#                     "planet_date": planet_dt,
#                     "season_label": season_label,
#                     "tier_used": None,
#                     "cloud_limit": None,
#                     "s2_id": None,
#                     "s2_datetime": None,
#                     "s2_cloud": None,
#                     "abs_days": None,
#                     "score": None,
#                     "status": "no_match_all_tiers",
#                 }
#             )
#         else:
#             s2_dt = ensure_naive(match_row["s2_datetime"])
#             abs_days = abs((s2_dt - planet_dt).total_seconds()) / 86400.0

#             new_results.append(
#                 {
#                     "chip_id": chip_id,
#                     "country": country,
#                     "planet_date": planet_dt,
#                     "season_label": season_label,
#                     "tier_used": tier_used,
#                     "cloud_limit": cloud_limit,
#                     "s2_id": match_row["s2_id"],
#                     "s2_datetime": s2_dt,
#                     "s2_cloud": float(match_row["s2_cloud"]),
#                     "abs_days": abs_days,
#                     "score": match_score,
#                     "status": "ok",
#                 }
#             )

#     # ----- WRITE / APPEND OUTPUT -----
#     new_df = pd.DataFrame(new_results)

#     if OUT_CSV.exists() and "chip_id" in pd.read_csv(OUT_CSV, nrows=1).columns:
#         # append without header
#         new_df.to_csv(OUT_CSV, mode="a", header=False, index=False)
#         logger.info(f"Appended {len(new_df)} rows to existing {OUT_CSV}")
#     else:
#         new_df.to_csv(OUT_CSV, index=False)
#         logger.info(f"Wrote {len(new_df)} rows to {OUT_CSV}")

#     logger.info("Done.")


# if __name__ == "__main__":
#     main()


# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
#  WIN A < WIN B

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
