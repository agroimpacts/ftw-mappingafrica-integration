#!/usr/bin/env python3
"""
download_windowA_parent128_nomask.py

Window A "parent128" re-download (same strategy as Window B parent128):
- Input: windowA_s2_matches_scored.csv (status == "ok")
- Uses s2_id column for S2 item id
- Uses Planet label chip grid to define the *target label footprint* (256x256)
- Builds a *parent* grid that is 2x the label extent (in X and Y),
  anchored so that the label footprint is in the bottom-right quadrant.
- Writes parent chips as 128x128 GeoTIFFs with B04,B03,B02,B08 (no mask)
- Bicubic resampling (Resampling.cubic)
- Output naming: <chip_id>_<YYYY-MM-DD>_S2L2A_parent128_nomask.tif
"""

from __future__ import annotations

from pathlib import Path
import time
import logging
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

WINDOWA_CSV = BASE / "data" / "windowA" / "windowA_s2_matches_scored.csv"

PLANET_IMG_DIR = Path("/home/airg/data/labels/cropland/mappingafrica-256/images")

OUT_DIR = BASE / "data" / "s2_chips" / "windowA_parent128_nomask_rgbnir_bigger_extent"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_COLLECTION = "sentinel-2-l2a"

# Band order for output files (matches your earlier downloads)
BANDS = ["B04", "B03", "B02", "B08"]

MAX_ITEM_RETRIES   = 4
MAX_BAND_RETRIES   = 3
BACKOFF_BASE_ITEM  = 1.7
BACKOFF_BASE_BAND  = 1.5
RATE_SLEEP_SEC     = 0.15  # gentle pause between chips

# Use GDAL env optimizations for vsicurl
rio_env = rio.Env(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF,.tiff,.TIFF",
    CPL_VSIL_CURL_NON_CACHED="/vsicurl/"
)

# ---------------- LOGGING ----------------
logger = logging.getLogger("windowA_parent128")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(h)


# ---------------- HELPERS ----------------
def find_planet_chip(chip_id: str) -> Path | None:
    """Find Planet label chip corresponding to chip_id."""
    hits = list(PLANET_IMG_DIR.glob(f"{chip_id}*.tif"))
    if hits:
        return hits[0]
    for p in PLANET_IMG_DIR.glob("*.tif"):
        if chip_id in p.name:
            return p
    return None


def read_planet_grid(p_path: Path) -> dict:
    """Read Planet chip CRS/transform/shape."""
    with rio.open(p_path) as src:
        return dict(
            crs=src.crs,
            transform=src.transform,
            width=src.width,
            height=src.height,
        )


def make_parent128_grid(label_meta: dict) -> dict:
    """
    Build a parent grid:
    - 2x the label footprint extent (in map units)
    - but 128x128 pixels
    - anchored so label footprint lies in bottom-right quadrant

    Label meta corresponds to the original label chip (typically 256x256).
    """
    T = label_meta["transform"]
    crs = label_meta["crs"]

    # Label pixel sizes (map units per pixel)
    px_w = T.a
    px_h = T.e  # usually negative for north-up

    # Label dimensions
    wL, hL = label_meta["width"], label_meta["height"]

    # Label extent sizes in map units
    extent_w = wL * px_w
    extent_h = hL * px_h  # negative

    # For a 2x extent parent, anchored so that label is bottom-right:
    # - keep the same top-left (origin) as the *parent* should start at label_top_left shifted up-left
    #
    # Easiest: derive parent origin by shifting label origin up-left by exactly 1 label extent.
    #
    # Label transform maps pixel (0,0) to top-left corner.
    # So parent top-left should be one label-width to the west, and one label-height to the north.
    parent_origin_x = T.c - extent_w
    parent_origin_y = T.f - extent_h  # subtracting negative = moving north

    # Parent pixel size: since parent covers 2x extent but is 128 pixels,
    # pixel size must be 4x label pixel size (because 2x extent / half pixels => 4x).
    parent_px_w = px_w * 4.0
    parent_px_h = px_h * 4.0

    parent_transform = rio.Affine(parent_px_w, T.b, parent_origin_x,
                                 T.d, parent_px_h, parent_origin_y)

    return dict(
        crs=crs,
        transform=parent_transform,
        width=128,
        height=128,
    )


def fetch_item_with_retry(client, s2_id: str):
    """Fetch a single S2 STAC item by ID, with retries."""
    for attempt in range(1, MAX_ITEM_RETRIES + 1):
        try:
            search = client.search(collections=[S2_COLLECTION], ids=[s2_id])
            items = list(search.items())
            if len(items) == 0:
                raise RuntimeError("no such STAC item")
            return items[0]
        except Exception as e:
            wait = BACKOFF_BASE_ITEM ** attempt
            logger.info(f"{s2_id}: STAC fetch failed (attempt {attempt}/{MAX_ITEM_RETRIES}): {e} — retry in {wait:.1f}s")
            time.sleep(wait)
    return None


def signed_asset(item, key: str) -> str:
    """Sign a Planetary Computer asset URL."""
    return pc.sign(item.assets[key].href)


def reproj_band_once(src_url: str, tgt_meta: dict, resampling=Resampling.cubic) -> np.ndarray:
    """Open remote band and reproject to target meta grid."""
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
    """Retry wrapper for band reproject."""
    for attempt in range(1, MAX_BAND_RETRIES + 1):
        try:
            return reproj_band_once(asset_url, tgt_meta, resampling=Resampling.cubic)
        except Exception as e:
            wait = BACKOFF_BASE_BAND ** attempt
            logger.warning(f"{chip_id}: band {bname} fail {attempt}/{MAX_BAND_RETRIES}: {e} — retry in {wait:.1f}s")
            time.sleep(wait)
    logger.error(f"{chip_id}: giving up on band {bname}")
    return None


def save_stack(out_path: Path, stack: np.ndarray, tgt_meta: dict):
    """Save (C,H,W) array as GeoTIFF."""
    profile = {
        "driver": "GTiff",
        "height": tgt_meta["height"],
        "width": tgt_meta["width"],
        "count": stack.shape[0],
        "crs": tgt_meta["crs"],
        "transform": tgt_meta["transform"],
        "dtype": stack.dtype.name,
        "tiled": True,
        "blockxsize": 128,
        "blockysize": 128,
        "compress": "deflate",
    }
    with rio.open(out_path, "w", **profile) as dst:
        for i in range(stack.shape[0]):
            dst.write(stack[i], i + 1)


# ---------------- MAIN ----------------
def main():
    logger.info("Loading windowA matches CSV...")
    df = pd.read_csv(WINDOWA_CSV)

    # Keep only successful matches
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower() == "ok"].copy()

    need_cols = {"chip_id", "s2_id"}
    miss = need_cols - set(df.columns)
    if miss:
        raise RuntimeError(f"windowA CSV missing columns: {miss}")

    # Optional date column (for filenames)
    date_col = None
    for cand in ["s2_datetime", "s2_primary_date", "s2_date", "datetime"]:
        if cand in df.columns:
            date_col = cand
            break
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    logger.info(f"Total chips to download (status=ok): {len(df)}")

    client = Client.open(STAC_URL)

    accepted_rows = []
    failed_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc="Downloading windowA parent128 S2 (no mask)", unit="chip"):

        chip_id = str(row["chip_id"])
        sid = str(row["s2_id"])

        # build output name
        if date_col and pd.notna(row[date_col]):
            s2_date_str = pd.to_datetime(row[date_col]).strftime("%Y-%m-%d")
            out_path = OUT_DIR / f"{chip_id}_{s2_date_str}_S2L2A_parent128_nomask.tif"
        else:
            out_path = OUT_DIR / f"{chip_id}_S2L2A_parent128_nomask.tif"

        # resume-friendly
        if out_path.exists():
            accepted_rows.append({"chip_id": chip_id, "s2_id": sid, "out_path": str(out_path), "reason": "exists_skip"})
            continue

        # planet chip used to define label grid
        p_path = find_planet_chip(chip_id)
        if p_path is None:
            failed_rows.append({"chip_id": chip_id, "s2_id": sid, "reason": "planet_chip_not_found"})
            continue

        label_meta = read_planet_grid(p_path)

        # build parent128 grid (2x extent, label is BR quadrant)
        tgt = make_parent128_grid(label_meta)

        # fetch S2 item
        item = fetch_item_with_retry(client, sid)
        if item is None:
            failed_rows.append({"chip_id": chip_id, "s2_id": sid, "reason": "stac_fetch_failed"})
            continue

        # bands
        band_arrays = []
        bad = False
        for b in BANDS:
            if b not in item.assets:
                failed_rows.append({"chip_id": chip_id, "s2_id": sid, "reason": f"missing_asset_{b}"})
                bad = True
                break
            url = signed_asset(item, b)
            arr = safe_reproj_band(url, tgt, bname=b, chip_id=chip_id)
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
            continue

        time.sleep(RATE_SLEEP_SEC)

    # audit CSVs
    acc_df = pd.DataFrame(accepted_rows)
    rej_df = pd.DataFrame(failed_rows)

    acc_csv = OUT_DIR / "accepted_windowA_parent128_nomask.csv"
    rej_csv = OUT_DIR / "failed_windowA_parent128_nomask.csv"
    acc_df.to_csv(acc_csv, index=False)
    rej_df.to_csv(rej_csv, index=False)

    logger.info(f"Done. Saved {len(acc_df)} windowA parent128 chips.")
    logger.info(f"Accepted list → {acc_csv}")
    logger.info(f"Failed list   → {rej_csv}")


if __name__ == "__main__":
    main()
