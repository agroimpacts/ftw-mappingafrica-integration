#!/usr/bin/env python3
"""
download_windowA_parent128_nomask_94.py

Downloads Window A Sentinel-2 L2A (RGB+NIR) into a parent128 grid:
- Output size: 128x128
- Extent: 2x the Planet label-chip extent in each axis
- Planet label footprint sits in LOWER RIGHT quadrant

Input:
- windowA_rescue_94_region_tweaks_matches.csv

Required columns:
- chip_id
- s2_id
- status  (we filter status == ok_rescue_94)

Output naming:
<chip_id>_<YYYY-MM>_<YYYY-MM-DD>_S2L2A_parent128_nomask.tif
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

CSV_IN = BASE / "data" / "windowA" / "windowA_rescue_94_region_tweaks_matches.csv"

PLANET_IMG_DIR = Path("/home/airg/data/labels/cropland/mappingafrica-256/images")

OUT_DIR = BASE / "data" / "s2_chips" / "windowA_parent128_nomask_rgbnir_bigger_extent_94"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_COLLECTION = "sentinel-2-l2a"

# 4-band stack: Red, Green, Blue, NIR
BANDS = ["B04", "B03", "B02", "B08"]

# status for this batch
ALLOWED_STATUSES = {"ok_rescue_94"}

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

logger = logging.getLogger("windowA_parent128_94")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(h)


def find_planet_chip(chip_id: str) -> Path | None:
    hits = list(PLANET_IMG_DIR.glob(f"{chip_id}*.tif"))
    if hits:
        return hits[0]
    for p in PLANET_IMG_DIR.glob("*.tif"):
        if chip_id in p.name:
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
            logger.info(f"{s2_id}: STAC fetch failed (attempt {attempt}/{MAX_ITEM_RETRIES}): {e} — retry in {wait:.1f}s")
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


def parse_planet_period_from_chip_id(chip_id: str) -> str:
    # chip_id like RW0005623_2021-11
    return chip_id.split("_", 1)[1] if "_" in chip_id else "UNKNOWN"


def parse_s2_date_from_s2_id(s2_id: str) -> str:
    m = re.search(r"_(\d{8})T\d{6}_", s2_id)
    if not m:
        return "UNKNOWNDATE"
    yyyymmdd = m.group(1)
    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"


def make_out_name(chip_id: str, s2_id: str) -> str:
    planet_period = parse_planet_period_from_chip_id(chip_id)
    s2_date = parse_s2_date_from_s2_id(s2_id)
    return f"{chip_id}_{planet_period}_{s2_date}_S2L2A_parent128_nomask.tif"


def main():
    if not CSV_IN.exists():
        raise FileNotFoundError(f"Input CSV not found: {CSV_IN}")

    logger.info(f"Loading CSV: {CSV_IN}")
    df = pd.read_csv(CSV_IN)

    need_cols = {"chip_id", "s2_id", "status"}
    miss = need_cols - set(df.columns)
    if miss:
        raise RuntimeError(f"CSV missing columns: {miss}")

    logger.info(f"CSV rows: {len(df)}")
    logger.info("Status counts:\n" + df["status"].value_counts(dropna=False).to_string())

    df = df[df["status"].astype(str).isin(ALLOWED_STATUSES)].copy()
    logger.info(f"Filtered status in {sorted(ALLOWED_STATUSES)} → {len(df)} rows")

    client = Client.open(STAC_URL)

    accepted_rows = []
    failed_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading windowA parent128 (94)", unit="chip"):
        chip_id = str(row["chip_id"])
        sid = str(row["s2_id"])

        out_path = OUT_DIR / make_out_name(chip_id, sid)
        if out_path.exists():
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

    acc_csv = OUT_DIR / "accepted_windowA_parent128_94.csv"
    rej_csv = OUT_DIR / "failed_windowA_parent128_94.csv"
    acc_df.to_csv(acc_csv, index=False)
    rej_df.to_csv(rej_csv, index=False)

    logger.info(f"Done. Saved: {len(acc_df)}")
    logger.info(f"Failed: {len(rej_df)}")
    logger.info(f"Accepted list → {acc_csv}")
    logger.info(f"Failed list   → {rej_csv}")


if __name__ == "__main__":
    main()
