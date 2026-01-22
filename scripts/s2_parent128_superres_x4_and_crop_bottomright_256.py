#!/usr/bin/env python3
"""
s2_parent128_superres_x4_and_crop_bottomright_256.py

Goal
----
Input : "parent" Sentinel-2 chips that are already 128x128 and contain the target label footprint
        in the *bottom-right quadrant*.
        Each chip is 4-band and stored in order: [B04,B03,B02,B08] = [R,G,B,NIR].

Process
-------
1) Read each 4-band chip (expects shape (4,128,128))
2) Reorder bands to SEN2SR expected order: [B02,B03,B04,B08] = [B,G,R,NIR]
3) Scale reflectance to 0–1 if needed
4) Run SEN2SR x4  : 128x128 -> 512x512 (same geographic footprint, 4x finer pixels)
5) Crop bottom-right quadrant from SR:
      (rows 256:512, cols 256:512) => 256x256
   This corresponds to the bottom-right quadrant of the original 128x128 footprint.
6) Reorder back to your desired output order: [R,G,B,NIR]
7) Save cropped SR GeoTIFF with correct CRS + transform for that cropped footprint.

Outputs
-------
- One GeoTIFF per input chip: 4 bands [R,G,B,NIR], 256x256
- Pixel size becomes 1/4 of the input chip pixel size (because SEN2SR is x4)

Notes
-----
- This script does NOT try to "reproject" the arrays into a different shape (that was causing your
  shape mismatch errors). Instead it computes the correct affine transform for the SR + crop window.
- If you want compression / tiling adjustments, edit the raster profile in `write_tif()`.

"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import rasterio as rio
from rasterio.transform import Affine
from tqdm import tqdm

import torch
import mlstac


# ---------------- LOGGING ----------------
logger = logging.getLogger("s2_sr_crop")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(h)


# ---------------- MODEL ----------------
def download_model(model_dir: Path) -> None:
    """
    Download SEN2SR RGBN x4 model into `model_dir` if not already present.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    # If already downloaded, mlstac will reuse local files.
    mlstac.download(
        file=(
            "https://huggingface.co/tacofoundation/sen2sr/resolve/main/"
            "SEN2SRLite/NonReference_RGBN_x4/mlm.json"
        ),
        output_dir=str(model_dir),
    )


def load_sen2sr(device: torch.device, model_dir: Path) -> torch.nn.Module:
    download_model(model_dir)
    logger.info("Loading SEN2SR model...")
    model = mlstac.load(str(model_dir)).compiled_model(device=device)
    model.eval()
    model.to(device)
    return model


# ---------------- IO HELPERS ----------------
def read_chip(path: Path) -> tuple[np.ndarray, dict]:
    """
    Read a 4-band chip and return:
      arr: np.ndarray (4, H, W)
      meta: dict with crs, transform, dtype, etc
    """
    with rio.open(path) as src:
        arr = src.read()  # (bands, H, W)
        meta = src.meta.copy()
        transform = src.transform
        crs = src.crs

    meta["transform"] = transform
    meta["crs"] = crs
    return arr, meta


def auto_scale_to_0_1(arr: np.ndarray) -> np.ndarray:
    """
    If values look like 0-10000 reflectance, divide by 10000.
    """
    arr = arr.astype(np.float32)
    max_val = float(np.nanmax(arr))
    if max_val > 50.0:
        arr = arr / 10000.0
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def reorder_rgbn_to_bgrn(arr_rgbn: np.ndarray) -> np.ndarray:
    """
    Your files: [R,G,B,NIR] = [B04,B03,B02,B08]
    Model wants: [B,G,R,NIR] = [B02,B03,B04,B08]
    """
    # RGBN -> BGRN
    return arr_rgbn[[2, 1, 0, 3], :, :]


def reorder_bgrn_to_rgbn(arr_bgrn: np.ndarray) -> np.ndarray:
    """
    Model output: [B,G,R,NIR]
    You want:     [R,G,B,NIR]
    """
    return arr_bgrn[[2, 1, 0, 3], :, :]


def compute_sr_transform(orig_transform: Affine) -> Affine:
    """
    SR is x4 in pixels over the same footprint:
      pixel width/height are divided by 4
    """
    # For north-up rasters, transform looks like:
    # Affine(a, b, c,
    #        d, e, f)
    # where a is pixel width, e is pixel height (negative typically).
    a = orig_transform.a / 4.0
    b = orig_transform.b / 4.0
    c = orig_transform.c
    d = orig_transform.d / 4.0
    e = orig_transform.e / 4.0
    f = orig_transform.f
    return Affine(a, b, c, d, e, f)


def compute_crop_transform(sr_transform: Affine, x_off: int, y_off: int) -> Affine:
    """
    Given SR transform and a pixel offset (x_off, y_off) in SR pixel coords,
    compute the transform for the cropped window.

    The top-left corner of the crop becomes the new origin.
    """
    # world(x, y) = sr_transform * (col, row)
    # For a crop starting at (x_off, y_off), new origin is world of that point.
    x0, y0 = sr_transform * (x_off, y_off)
    return Affine(sr_transform.a, sr_transform.b, x0,
                  sr_transform.d, sr_transform.e, y0)


def write_tif(out_path: Path, arr: np.ndarray, crs, transform: Affine) -> None:
    """
    Write 4-band uint16 GeoTIFF.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # arr is (4, H, W)
    profile = {
        "driver": "GTiff",
        "height": arr.shape[1],
        "width": arr.shape[2],
        "count": arr.shape[0],
        "dtype": "uint16",
        "crs": crs,
        "transform": transform,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "deflate",
        "predictor": 2,
    }

    with rio.open(out_path, "w", **profile) as dst:
        dst.write(arr.astype(np.uint16))


def build_out_name(in_path: Path) -> str:
    """
    Example:
      GH0000047_S2L2A_nomask_parent128.tif
        -> GH0000047_S2L2A_sr2p5m_crop256_rgbn.tif
    """
    name = in_path.name
    stem = name[:-4] if name.lower().endswith(".tif") else name

    # Try to keep it readable
    if "parent128" in stem:
        stem = stem.replace("parent128", "sr2p5m_crop256_rgbn")
    else:
        stem = stem + "_sr2p5m_crop256_rgbn"

    return stem + ".tif"


# ---------------- MAIN PIPELINE ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="/home/airg/gessuman/projects/ftw-mappingafrica-integration/data/s2_chips/primary_nomask_rgbnir_bigger_extent",
        help="Folder with parent128 chips (*.tif).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="/home/airg/gessuman/projects/ftw-mappingafrica-integration/data/s2_chips/primary_parent128_sr2p5m_crop256",
        help="Output folder for cropped SR chips (256x256).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='cuda|cpu|auto (auto picks cuda if available)',
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for model inference.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="model/SEN2SR_RGBN",
        help="Where to store/download SEN2SR model.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite outputs if they already exist.",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    # Model
    model = load_sen2sr(device=device, model_dir=Path(args.model_dir))

    # Inputs
    paths = sorted(root.glob("*.tif"))
    if not paths:
        raise RuntimeError(f"No tif files found in: {root}")
    logger.info(f"Found {len(paths)} parent128 chips in: {root}")
    logger.info(f"Writing outputs to: {out_dir}")

    # Crop window in SR space:
    # SR is 512x512. Bottom-right quadrant is rows 256:512, cols 256:512.
    crop_x0, crop_y0 = 256, 256
    crop_size = 256

    # Process in batches
    bs = max(1, int(args.batch_size))

    for i0 in tqdm(range(0, len(paths), bs), desc="SR + crop label", unit="batch"):
        batch_paths = paths[i0:i0 + bs]

        # Read + prep batch tensors
        batch_tensors = []
        metas = []

        skip_all = True
        out_paths = []

        for p in batch_paths:
            out_path = out_dir / build_out_name(p)
            out_paths.append(out_path)
            if out_path.exists() and not args.overwrite:
                metas.append(None)
                batch_tensors.append(None)
                continue

            skip_all = False

            arr, meta = read_chip(p)
            if arr.shape != (4, 128, 128):
                raise ValueError(f"Expected (4,128,128) but got {arr.shape} for {p}")

            # Your input order on disk: [R,G,B,NIR]
            arr = auto_scale_to_0_1(arr)
            arr = reorder_rgbn_to_bgrn(arr)  # -> [B,G,R,NIR] for model

            t = torch.from_numpy(arr).float()  # (4,128,128)
            t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

            batch_tensors.append(t)
            metas.append(meta)

        if skip_all:
            continue

        # Stack only the non-skipped items for inference
        keep_idx = [k for k, t in enumerate(batch_tensors) if t is not None]
        x = torch.stack([batch_tensors[k] for k in keep_idx], dim=0).to(device)  # (B,4,128,128)

        with torch.inference_mode():
            sr = model(x)  # (B,4,512,512) in [B,G,R,NIR] order
            sr = sr.detach().cpu().numpy()

        # Convert to uint16 reflectance-like (consistent with your earlier pipeline)
        sr_u16 = (sr * 10000.0).clip(0, 65535).astype(np.uint16)

        # Write outputs (cropped bottom-right 256x256), in your desired order [R,G,B,NIR]
        for j, k in enumerate(keep_idx):
            p = batch_paths[k]
            meta = metas[k]
            out_path = out_paths[k]

            try:
                crs = meta["crs"]
                orig_transform = meta["transform"]

                # Compute SR transform (same footprint as parent128, 4x finer pixels)
                sr_transform = compute_sr_transform(orig_transform)

                # Crop bottom-right quadrant (SR pixel coords)
                crop_bgrn = sr_u16[j, :, crop_y0:crop_y0 + crop_size, crop_x0:crop_x0 + crop_size]
                if crop_bgrn.shape != (4, 256, 256):
                    raise ValueError(f"Unexpected crop shape {crop_bgrn.shape} for {p}")

                # Convert back to [R,G,B,NIR]
                crop_rgbn = reorder_bgrn_to_rgbn(crop_bgrn)

                # Transform for the cropped window
                crop_transform = compute_crop_transform(sr_transform, crop_x0, crop_y0)

                write_tif(out_path, crop_rgbn, crs=crs, transform=crop_transform)

            except Exception as e:
                logger.error(f"ERROR processing {p}: {e}")


if __name__ == "__main__":
    main()
