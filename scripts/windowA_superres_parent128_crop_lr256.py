#!/usr/bin/env python3
"""
windowA_superres_parent128_crop_lr256.py

Goal
----
Super-resolve Window A parent128 Sentinel-2 chips using SEN2SR (x4) and then
crop the lower-right quadrant to recover the label footprint.

Input assumption
----------------
Each input GeoTIFF is a 4-band stack:
  Band order: [B04 (Red), B03 (Green), B02 (Blue), B08 (NIR)]
and is 128x128 pixels ("parent128"), whose LOWER-RIGHT 64x64 corresponds to the
label footprint.

Pipeline
--------
1) Read input (4,128,128), normalize reflectance to 0..1 (divide by 10_000 if needed)
2) SEN2SR (x4): 128x128 -> 512x512
3) Crop SR lower-right quadrant: (rows 256:512, cols 256:512) -> 256x256
4) Compute correct transform for the cropped chip
5) Write as Cloud Optimized GeoTIFF (COG)

Output naming
-------------
Keeps the same prefix, swaps suffix:
  *_S2L2A_parent128_nomask.tif
becomes:
  *_S2L2A_sr2p5m_nomask.tif

Example:
RW0005623_2021-11_2021-06-26_S2L2A_parent128_nomask.tif
-> RW0005623_2021-11_2021-06-26_S2L2A_sr2p5m_nomask.tif
"""

import argparse
from pathlib import Path
import logging
import time

import numpy as np
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from tqdm import tqdm

import torch
import mlstac


# ---------------- LOGGING ----------------
logger = logging.getLogger("windowA_sr")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(h)


# ---------------- MODEL ----------------
MODEL_URL = (
    "https://huggingface.co/tacofoundation/sen2sr/resolve/main/"
    "SEN2SRLite/NonReference_RGBN_x4/mlm.json"
)


def ensure_model(model_dir: Path) -> None:
    """Download SEN2SR model to model_dir if missing."""
    if model_dir.exists() and any(model_dir.rglob("mlm.json")):
        return
    model_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading SEN2SR model into: {model_dir}")
    mlstac.download(file=MODEL_URL, output_dir=str(model_dir))


def load_sen2sr(model_dir: Path, device: torch.device):
    """Load SEN2SR compiled model."""
    logger.info("Loading SEN2SR model...")
    model = mlstac.load(str(model_dir)).compiled_model(device=device)
    model.eval()
    model.to(device)
    return model


# ---------------- IO HELPERS ----------------
def list_tifs(root: Path):
    """List all tif files in root (non-recursive)."""
    return sorted(root.glob("*.tif"))


def normalize_reflectance(arr: np.ndarray) -> np.ndarray:
    """
    Normalize reflectance to 0..1.

    Heuristic:
    - If max > 50 -> assume 0..10000 scale, divide by 10000
    - Else assume already in 0..1-ish
    """
    arr = arr.astype("float32", copy=False)
    mx = float(np.nanmax(arr))
    if mx > 50.0:
        arr = arr / 10_000.0
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def build_out_name(in_name: str) -> str:
    """
    Keep prefix, swap suffix to sr2p5m.
    """
    if in_name.endswith("_S2L2A_parent128_nomask.tif"):
        return in_name.replace("_S2L2A_parent128_nomask.tif", "_S2L2A_sr2p5m_nomask.tif")
    if in_name.endswith("_S2L2A_nomask_parent128.tif"):
        # handle older wrong-format suffix if any remain
        return in_name.replace("_S2L2A_nomask_parent128.tif", "_S2L2A_sr2p5m_nomask.tif")
    # fallback
    stem = in_name[:-4] if in_name.lower().endswith(".tif") else in_name
    return f"{stem}_S2L2A_sr2p5m_nomask.tif"


def write_cog_uint16(
    out_path: Path,
    sr_crop_uint16: np.ndarray,
    crs,
    transform: Affine,
):
    """
    Write (4,256,256) uint16 as COG GeoTIFF.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    profile = dict(
        driver="COG",
        height=sr_crop_uint16.shape[1],
        width=sr_crop_uint16.shape[2],
        count=sr_crop_uint16.shape[0],
        dtype="uint16",
        crs=crs,
        transform=transform,
        compress="zstd",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )

    with rio.open(out_path, "w", **profile) as dst:
        dst.write(sr_crop_uint16[0], 1)  # B04 red
        dst.write(sr_crop_uint16[1], 2)  # B03 green
        dst.write(sr_crop_uint16[2], 3)  # B02 blue
        dst.write(sr_crop_uint16[3], 4)  # B08 nir

        dst.set_band_description(1, "B04_red")
        dst.set_band_description(2, "B03_green")
        dst.set_band_description(3, "B02_blue")
        dst.set_band_description(4, "B08_nir")


# ---------------- GEOMETRY / TRANSFORM ----------------
def transform_for_sr_full(parent_transform: Affine) -> Affine:
    """
    Parent128 covers the parent extent at 128x128.
    SR-full covers the same extent at 512x512 => pixel size /4.
    """
    # Scale pixel size by (128/512) = 0.25
    return parent_transform * Affine.scale(0.25, 0.25)


def transform_for_crop(sr_full_transform: Affine, row_off: int, col_off: int) -> Affine:
    """
    Compute the transform for a window crop from the SR-full grid.

    For a crop starting at (row_off, col_off), you shift the origin accordingly:
      x0' = x0 + col_off * a
      y0' = y0 + row_off * e
    where a = pixel width, e = pixel height (negative).
    """
    a = sr_full_transform.a
    e = sr_full_transform.e
    c = sr_full_transform.c + col_off * a
    f = sr_full_transform.f + row_off * e
    return Affine(a, sr_full_transform.b, c, sr_full_transform.d, e, f)


# ---------------- MAIN ----------------
def main(args):
    root = Path(args.root).expanduser()
    out_dir = Path(args.out).expanduser()
    model_dir = Path(args.model_dir).expanduser()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logger.info(f"Input folder : {root}")
    logger.info(f"Output folder: {out_dir}")
    logger.info(f"Device       : {device}")
    out_dir.mkdir(parents=True, exist_ok=True)

    ensure_model(model_dir)
    model = load_sen2sr(model_dir, device=device)

    tifs = list_tifs(root)
    if not tifs:
        raise RuntimeError(f"No .tif files found in {root}")

    logger.info(f"Found {len(tifs)} parent128 chips to process.")

    # Batch processing without DataLoader (simpler + fewer moving parts on clusters)
    batch = []
    meta = []

    accepted = []
    failed = []

    def flush_batch():
        nonlocal batch, meta
        if not batch:
            return

        x = torch.stack(batch, dim=0).to(device)  # (B,4,128,128)

        with torch.inference_mode():
            y = model(x).detach().cpu().numpy()  # (B,4,512,512) float

        # Convert SR output to uint16 in reflectance scale
        # SEN2SR output is typically 0..1, convert back to 0..10000-ish then to uint16
        y_u16 = np.clip(y * 10_000.0, 0, 65_535).astype(np.uint16)

        # Crop LR 256x256: rows 256:512, cols 256:512
        r0, r1 = 256, 512
        c0, c1 = 256, 512

        for i, info in enumerate(meta):
            try:
                out_path = info["out_path"]
                if out_path.exists() and not args.overwrite:
                    accepted.append({"in_path": str(info["in_path"]), "out_path": str(out_path), "reason": "exists_skip"})
                    continue

                sr_crop = y_u16[i, :, r0:r1, c0:c1]  # (4,256,256)

                # Build transforms
                sr_full_T = transform_for_sr_full(info["parent_transform"])
                crop_T = transform_for_crop(sr_full_T, row_off=r0, col_off=c0)

                write_cog_uint16(
                    out_path=out_path,
                    sr_crop_uint16=sr_crop,
                    crs=info["crs"],
                    transform=crop_T,
                )

                accepted.append({"in_path": str(info["in_path"]), "out_path": str(out_path), "reason": "saved"})

            except Exception as e:
                failed.append({"in_path": str(info["in_path"]), "reason": str(e)})
        batch = []
        meta = []

    for p in tqdm(tifs, desc="Super-resolve (WindowA parent128 -> SR512 -> crop LR256)", unit="chip"):
        try:
            out_name = build_out_name(p.name)
            out_path = out_dir / out_name

            with rio.open(p) as src:
                arr = src.read()  # (C,H,W)
                if arr.shape != (4, 128, 128):
                    raise ValueError(f"Expected (4,128,128) got {arr.shape}")

                # Normalize
                arr = normalize_reflectance(arr)

                # Convert to tensor float32
                ten = torch.from_numpy(arr).to(torch.float32)

                # store metadata to compute output georef
                info = dict(
                    in_path=p,
                    out_path=out_path,
                    crs=src.crs,
                    parent_transform=src.transform,  # transform for 128x128 parent extent
                )

            batch.append(ten)
            meta.append(info)

            if len(batch) >= args.batch_size:
                flush_batch()
                if args.sleep_sec > 0:
                    time.sleep(args.sleep_sec)

        except Exception as e:
            failed.append({"in_path": str(p), "reason": str(e)})

    # flush remaining
    flush_batch()

    # write audit CSVs
    acc_df = np.array(accepted, dtype=object)
    fail_df = np.array(failed, dtype=object)

    import pandas as pd
    pd.DataFrame(accepted).to_csv(out_dir / "accepted_windowA_sr_crop.csv", index=False)
    pd.DataFrame(failed).to_csv(out_dir / "failed_windowA_sr_crop.csv", index=False)

    logger.info(f"Done. Saved/Skipped: {len(accepted)}")
    logger.info(f"Failed: {len(failed)}")
    logger.info(f"Accepted CSV → {out_dir / 'accepted_windowA_sr_crop.csv'}")
    logger.info(f"Failed CSV   → {out_dir / 'failed_windowA_sr_crop.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Folder containing WindowA parent128 chips (128x128, 4-band B04,B03,B02,B08).",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output folder for cropped SR chips (256x256).",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="model/SEN2SR_RGBN",
        help="Where to download/load the SEN2SR model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='"auto", "cuda", or "cpu".',
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs if present.",
    )
    parser.add_argument(
        "--sleep_sec",
        type=float,
        default=0.0,
        help="Optional sleep between batches (politeness / throttling).",
    )
    args = parser.parse_args()
    main(args)
