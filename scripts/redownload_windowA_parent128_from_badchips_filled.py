#!/usr/bin/env python3
"""
redownload_windowA_parent128_from_badchips_filled_NEWFOLDER.py

Re-download ONLY the Window A *bad chips* (QC-flagged) parent128 Sentinel-2 L2A (RGB+NIR),
using your uploaded filled CSV.

- Output size: 128x128
- Extent: 2x the Planet label-chip extent in each axis
- Planet label footprint sits in LOWER RIGHT quadrant

Input:
- /mnt/data/bad_chips_windowA_FILLED.csv  (uploaded)

Required columns (minimum):
- chip_id
- s2_id

Output folder:
- NEW folder (does NOT overwrite your existing windowA_parent128_nomask_rgbnir_bigger_extent)

Output naming (your required format):
<chip_prefix>_<planet_period>_<YYYY-MM-DD>_S2L2A_parent128_nomask.tif
Example:
RW0005623_2021-11_2021-06-26_S2L2A_parent128_nomask.tif
"""

from pathlib import Path
import time
import logging
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

import rasterio as rio
from rasterio.warp import reproject
from rasterio.enums import Resampling

from pystac_client import Client
import planetary_computer as pc


# ---------------- CONFIG ----------------
BASE = Path("/home/airg/gessuman/projects/ftw-mappingafrica-integration")

# <-- Uploaded CSV path (per your note)
CSV_IN = Path("/home/airg/gessuman/projects/ftw-mappingafrica-integration/data/windowA/bad_chips_windowA_FILLED.csv")

PLANET_IMG_DIR = Path("/home/airg/data/labels/cropland/mappingafrica-256/images")

# <-- NEW output folder (safe)
OUT_DIR = BASE / "data" / "s2_chips" / "windowA_parent128_nomask_rgbnir_bigger_extent_BADCHIPS_RERUN"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Do NOT overwrite outputs in this new folder either
SKIP_IF_EXISTS = True

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_COLLECTION = "sentinel-2-l2a"

# 4-band stack: Red, Green, Blue, NIR
BANDS = ["B04", "B03", "B02", "B08"]

MAX_ITEM_RETRIES   = 10
MAX_BAND_RETRIES   = 3
BACKOFF_BASE_ITEM  = 2.0
BACKOFF_BASE_BAND  = 1.6
RATE_SLEEP_SEC     = 0.25

rio_env = rio.Env(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF,.tiff,.TIFF",
    CPL_VSIL_CURL_NON_CACHED="/vsicurl/"
)

logger = logging.getLogger("windowA_parent128_badchips_newfolder")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(h)


# ---------------- HELPERS ----------------

def chip_prefix(chip_id: str) -> str:
    """Everything before first underscore (works for 'RW0005623_2021-11' and 'RW0005623')."""
    return chip_id.split("_", 1)[0] if "_" in chip_id else chip_id


def find_planet_chip(chip_id: str) -> Path | None:
    """
    Planet chip filenames typically start with the chip prefix (e.g., RW0005623...).
    So search by prefix, not necessarily the full chip_id.
    """
    pref = chip_prefix(chip_id)
    hits = list(PLANET_IMG_DIR.glob(f"{pref}*.tif"))
    if hits:
        return hits[0]
    for p in PLANET_IMG_DIR.glob("*.tif"):
        if pref in p.name:
            return p
    return None


def read_grid_for_parent128(p_path: Path) -> dict:
    """Parent128 grid where original Planet label footprint sits in lower-right quadrant."""
    with rio.open(p_path) as src:
        T = src.transform
        px = T.a
        py = T.e  # negative

        label_w = src.width * px
        label_h = src.height * abs(py)

        parent_w = 2 * label_w
        parent_h = 2 * label_h

        parent_px = parent_w / 128.0
        parent_py = -parent_h / 128.0

        left, bottom, right, top = src.bounds

        # parent bbox shifted LEFT and UP by 1 label width/height
        parent_left = left - label_w
        parent_top  = top + label_h

        parent_transform = rio.Affine(parent_px, 0.0, parent_left,
                                      0.0, parent_py, parent_top)

        return dict(crs=src.crs, transform=parent_transform, width=128, height=128)


def fetch_item_with_retry(client, s2_id: str):
    for attempt in range(1, MAX_ITEM_RETRIES + 1):
        try:
            search = client.search(collections=[S2_COLLECTION], ids=[s2_id])
            items = list(search.items())
            if not items:
                raise RuntimeError("no such STAC item")
            return items[0]
        except Exception as e:
            wait = BACKOFF_BASE_ITEM ** attempt
            logger.info(
                f"{s2_id}: STAC fetch failed (attempt {attempt}/{MAX_ITEM_RETRIES}): {e} — retry in {wait:.1f}s"
            )
            time.sleep(wait)
    return None


def signed_asset(item, key: str) -> str:
    return pc.sign(item.assets[key].href)


def reproj_band_once(src_url: str, tgt_meta: dict, resampling=Resampling.cubic) -> np.ndarray:
    with rio_env:
        with rio.open(src_url) as src:
            dst = np.zeros((tgt_meta["height"], tgt_meta["width"]), dtype=src.dtypes[0])
            reproject(
                source=rio.band(src, 1),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=tgt_meta["transform"],
                dst_crs=tgt_meta["crs"],
                resampling=resampling,
            )
            return dst


def safe_reproj_band(asset_url: str, tgt_meta: dict, bname: str, chip_id: str) -> np.ndarray | None:
    for attempt in range(1, MAX_BAND_RETRIES + 1):
        try:
            return reproj_band_once(asset_url, tgt_meta, resampling=Resampling.cubic)
        except Exception as e:
            wait = BACKOFF_BASE_BAND ** attempt
            logger.warning(f"{chip_id}: band {bname} fail {attempt}/{MAX_BAND_RETRIES}: {e} — retry in {wait:.1f}s")
            time.sleep(wait)
    return None


def save_stack(out_path: Path, stack: np.ndarray, tgt_meta: dict):
    profile = dict(
        driver="GTiff",
        height=tgt_meta["height"],
        width=tgt_meta["width"],
        count=stack.shape[0],
        crs=tgt_meta["crs"],
        transform=tgt_meta["transform"],
        dtype=stack.dtype.name,
        tiled=True,
        blockxsize=128,
        blockysize=128,
        compress="deflate",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rio.open(out_path, "w", **profile) as dst:
        for i in range(stack.shape[0]):
            dst.write(stack[i], i + 1)
        dst.set_band_description(1, "B04_red")
        dst.set_band_description(2, "B03_green")
        dst.set_band_description(3, "B02_blue")
        dst.set_band_description(4, "B08_nir")


def parse_s2_date_from_s2_id(s2_id: str) -> str:
    m = re.search(r"_(\d{8})T\d{6}_", s2_id)
    if not m:
        return "UNKNOWNDATE"
    yyyymmdd = m.group(1)
    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"


def get_planet_period(chip_id: str, row: pd.Series) -> str:
    """
    Priority:
    1) If chip_id contains "_" already: use the suffix after first underscore (e.g., 2021-11)
    2) Else, use a planet-period-like column if present
    3) Else UNKNOWN
    """
    if "_" in chip_id:
        return chip_id.split("_", 1)[1]
    for col in ("planet_period", "planet_month", "planet_date_month"):
        if col in row and pd.notna(row[col]):
            return str(row[col])
    return "UNKNOWN"


def make_out_name(chip_id: str, row: pd.Series) -> str:
    pref = chip_prefix(chip_id)
    period = get_planet_period(chip_id, row)
    s2_date = parse_s2_date_from_s2_id(str(row["s2_id"]))
    return f"{pref}_{period}_{s2_date}_S2L2A_parent128_nomask.tif"


# ---------------- MAIN ----------------

def main():
    if not CSV_IN.exists():
        raise FileNotFoundError(f"Input CSV not found: {CSV_IN}")

    logger.info(f"Loading CSV: {CSV_IN}")
    df = pd.read_csv(CSV_IN)

    need_cols = {"chip_id", "s2_id"}
    miss = need_cols - set(df.columns)
    if miss:
        raise RuntimeError(f"CSV missing columns: {miss}. Available: {list(df.columns)}")

    df["chip_id"] = df["chip_id"].astype(str)
    df["s2_id"] = df["s2_id"].astype(str)

    logger.info(f"Total bad chips to redownload into NEW folder: {len(df)}")
    logger.info(f"Output folder: {OUT_DIR}")

    client = Client.open(STAC_URL)

    accepted_rows = []
    failed_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Redownloading parent128 (bad chips)", unit="chip"):
        chip_id = str(row["chip_id"])
        sid = str(row["s2_id"])

        out_path = OUT_DIR / make_out_name(chip_id, row)

        if SKIP_IF_EXISTS and out_path.exists():
            accepted_rows.append({"chip_id": chip_id, "s2_id": sid, "out_path": str(out_path), "reason": "exists_skip"})
            continue

        p_path = find_planet_chip(chip_id)
        if p_path is None:
            failed_rows.append({"chip_id": chip_id, "s2_id": sid, "reason": "planet_chip_not_found"})
            continue

        tgt = read_grid_for_parent128(p_path)

        item = fetch_item_with_retry(client, sid)
        if item is None:
            failed_rows.append({"chip_id": chip_id, "s2_id": sid, "reason": "stac_fetch_failed"})
            continue

        band_arrays = []
        bad = False
        for b in BANDS:
            if b not in item.assets:
                failed_rows.append({"chip_id": chip_id, "s2_id": sid, "reason": f"missing_asset_{b}"})
                bad = True
                break

            arr = safe_reproj_band(signed_asset(item, b), tgt, bname=b, chip_id=chip_id)
            if arr is None:
                failed_rows.append({"chip_id": chip_id, "s2_id": sid, "reason": f"reproj_failed_{b}"})
                bad = True
                break

            band_arrays.append(arr)

        if bad:
            continue

        stack = np.stack(band_arrays, axis=0)

        try:
            save_stack(out_path, stack, tgt)
            accepted_rows.append({"chip_id": chip_id, "s2_id": sid, "out_path": str(out_path), "reason": "saved"})
        except Exception as e:
            failed_rows.append({"chip_id": chip_id, "s2_id": sid, "reason": f"save_failed:{e}"})

        time.sleep(RATE_SLEEP_SEC)

    acc_df = pd.DataFrame(accepted_rows)
    rej_df = pd.DataFrame(failed_rows)

    acc_csv = OUT_DIR / "accepted_windowA_parent128_badchips_redownload.csv"
    rej_csv = OUT_DIR / "failed_windowA_parent128_badchips_redownload.csv"
    acc_df.to_csv(acc_csv, index=False)
    rej_df.to_csv(rej_csv, index=False)

    logger.info(f"Done. Saved: {len(acc_df)}")
    logger.info(f"Failed: {len(rej_df)}")
    logger.info(f"Accepted list → {acc_csv}")
    logger.info(f"Failed list   → {rej_csv}")


if __name__ == "__main__":
    main()
