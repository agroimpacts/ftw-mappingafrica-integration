#!/usr/bin/env python3
"""
download_windowA_s2_nomask.py

- Download Sentinel-2 L2A "window A" chips aligned to Planet chips
- Uses matches from windowA_s2_matches_scored.csv (status == "ok")
- NO SCL MASKING
- Saves 4-band stacks (B04,B03,B02,B08) in Planet grid
- Bicubic resampling (Resampling.cubic)
- Output naming: <chip_id>_<YYYY-MM-DD>_S2L2A_nomask.tif
"""

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

# Window A match CSV from your big run
WINDOWA_CSV = BASE / "data" / "windowA" / "windowA_rescue_first_19.csv"

# Planet chips directory (same as before)
PLANET_IMG_DIR = Path("/home/airg/data/labels/cropland/mappingafrica-256/images")

# Output directory for window A S2 chips
OUT_DIR = BASE / "data" / "s2_chips" / "19_ok"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_COLLECTION = "sentinel-2-l2a"

# Band order: Red, Green, Blue, NIR
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
logger = logging.getLogger("windowA_19")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(h)

# ---------------- HELPERS ----------------

def find_planet_chip(chip_id: str) -> Path | None:
    """
    Find the Planet chip file on disk that corresponds to this chip_id.
    First tries prefix match <chip_id>*.tif, then slower substring scan.
    """
    hits = list(PLANET_IMG_DIR.glob(f"{chip_id}*.tif"))
    if hits:
        return hits[0]
    # slow fallback
    for p in PLANET_IMG_DIR.glob("*.tif"):
        if chip_id in p.name:
            return p
    return None

def read_planet_grid(p_path: Path) -> dict:
    """Read Planet chip just to grab CRS and grid transform/shape."""
    with rio.open(p_path) as src:
        return dict(
            crs=src.crs,
            transform=src.transform,
            width=src.width,
            height=src.height,
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
            logger.info(
                f"{s2_id}: STAC fetch failed (attempt {attempt}/{MAX_ITEM_RETRIES}): "
                f"{e} — retry in {wait:.1f}s"
            )
            time.sleep(wait)
    return None

def signed_asset(item, key: str) -> str:
    """Sign a Planetary Computer asset URL."""
    return pc.sign(item.assets[key].href)

def reproj_band_once(src_url: str, tgt_meta: dict, resampling=Resampling.cubic) -> np.ndarray:
    """
    Open a remote Sentinel-2 band and reproject it to the Planet chip grid.

    NOTE: Resampling.cubic = bicubic resampling.
    """
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
    """Retry wrapper: returns array or None if all attempts fail."""
    for attempt in range(1, MAX_BAND_RETRIES + 1):
        try:
            return reproj_band_once(asset_url, tgt_meta, resampling=Resampling.cubic)
        except Exception as e:
            wait = BACKOFF_BASE_BAND ** attempt
            logger.warning(
                f"{chip_id}: band {bname} read/reproj fail attempt "
                f"{attempt}/{MAX_BAND_RETRIES}: {e} — retry in {wait:.1f}s"
            )
            time.sleep(wait)
    logger.error(f"{chip_id}: giving up on band {bname} after {MAX_BAND_RETRIES} attempts")
    return None

def save_stack(out_path: Path, stack: np.ndarray, tgt_meta: dict):
    """Save a (C,H,W) array as a tiled GeoTIFF."""
    profile = {
        "driver": "GTiff",
        "height": tgt_meta["height"],
        "width": tgt_meta["width"],
        "count": stack.shape[0],
        "crs": tgt_meta["crs"],
        "transform": tgt_meta["transform"],
        "dtype": stack.dtype.name,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "deflate",
    }
    with rio.open(out_path, "w", **profile) as dst:
        for i in range(stack.shape[0]):
            dst.write(stack[i], i + 1)

# ---------------- MAIN ----------------

def main():
    logger.info("Loading windowA matches CSV...")
    df = pd.read_csv(WINDOWA_CSV, parse_dates=["planet_date", "s2_datetime"])

    # Only keep rows with successful matches
    df = df[df["status"] == "ok"].copy()

    need_cols = {"chip_id", "s2_id", "s2_datetime"}
    miss = need_cols - set(df.columns)
    if miss:
        raise RuntimeError(f"windowA CSV missing columns: {miss}")

    client = Client.open(STAC_URL)

    accepted_rows = []
    failed_rows = []

    logger.info(f"Total chips to download (status=ok): {len(df)}")

    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc="Downloading windowA S2 chips (no mask)", unit="chip"):

        chip_id = str(row["chip_id"])
        sid = str(row["s2_id"])
        s2_dt = row["s2_datetime"]

        if pd.isna(s2_dt):
            failed_rows.append(
                {"chip_id": chip_id, "s2_id": sid, "reason": "missing_s2_datetime"}
            )
            continue

        # format date for filename
        s2_date_str = pd.to_datetime(s2_dt).strftime("%Y-%m-%d")
        out_path = OUT_DIR / f"{chip_id}_{s2_date_str}_S2L2A_nomask.tif"

        # Skip if already exists (resume-friendly)
        if out_path.exists():
            accepted_rows.append(
                {"chip_id": chip_id, "s2_id": sid, "out_path": str(out_path), "reason": "exists_skip"}
            )
            continue

        # find planet chip / grid
        p_path = find_planet_chip(chip_id)
        if p_path is None:
            failed_rows.append(
                {"chip_id": chip_id, "s2_id": sid, "reason": "planet_chip_not_found"}
            )
            continue
        tgt = read_planet_grid(p_path)

        # fetch S2 item
        item = fetch_item_with_retry(client, sid)
        if item is None:
            failed_rows.append(
                {"chip_id": chip_id, "s2_id": sid, "reason": "stac_fetch_failed"}
            )
            continue

        # read & reproject bands
        band_arrays = []
        bad = False
        for b in BANDS:
            if b not in item.assets:
                failed_rows.append(
                    {"chip_id": chip_id, "s2_id": sid, "reason": f"missing_asset_{b}"}
                )
                bad = True
                break
            url = signed_asset(item, b)
            arr = safe_reproj_band(url, tgt, bname=b, chip_id=chip_id)
            if arr is None:
                failed_rows.append(
                    {"chip_id": chip_id, "s2_id": sid, "reason": f"reproj_failed_{b}"}
                )
                bad = True
                break
            band_arrays.append(arr)

        if bad:
            continue

        stack = np.stack(band_arrays, axis=0)

        try:
            save_stack(out_path, stack, tgt)
            accepted_rows.append(
                {"chip_id": chip_id, "s2_id": sid, "out_path": str(out_path), "reason": "saved"}
            )
        except Exception as e:
            failed_rows.append(
                {"chip_id": chip_id, "s2_id": sid, "reason": f"save_failed:{e}"}
            )
            continue

        time.sleep(RATE_SLEEP_SEC)

    # write audit CSVs
    acc_df = pd.DataFrame(accepted_rows)
    rej_df = pd.DataFrame(failed_rows)

    acc_csv = OUT_DIR / "accepted_windowA_nomask.csv"
    rej_csv = OUT_DIR / "failed_windowA_nomask.csv"
    acc_df.to_csv(acc_csv, index=False)
    rej_df.to_csv(rej_csv, index=False)

    logger.info(f"Done. Saved {len(acc_df)} windowA chips (status=ok)")
    logger.info(f"Accepted list → {acc_csv}")
    logger.info(f"Failed list   → {rej_csv}")

if __name__ == "__main__":
    main()
