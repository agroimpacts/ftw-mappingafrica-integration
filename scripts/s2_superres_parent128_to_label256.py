#!/usr/bin/env python3
"""
s2_superres_parent128_to_label256.py

Goal
----
Input  : Sentinel-2 "parent" chips (4-band RGBN) of size 128x128.
Process: Apply SEN2SR (RGBN x4) to get 512x512 (4x upsample).
Crop   : Extract bottom-right quadrant as a 256x256 chip (label footprint).
Output : 256x256 SR chip with correct CRS + transform (2.5m-equivalent grid).

Key idea (georeferencing)
-------------------------
- Original parent chip has transform T (pixel size ~10m in projected CRS, or degrees in EPSG:4326).
- Full SR chip pixel size is 1/4 of original. So transform becomes:
      T_sr_full = T * Affine.scale(1/4, 1/4)
- Bottom-right crop starts at (row0=256, col0=256) in the 512x512 SR grid.
  So transform becomes:
      T_sr_crop = T_sr_full * Affine.translation(col0, row0)

Notes
-----
- SEN2SR expects band order: [B02, B03, B04, B08] (Blue, Green, Red, NIR).
  If your files are [B04, B03, B02, B08], we reorder.
"""

import argparse
from pathlib import Path

import numpy as np
import mlstac
import torch
import rioxarray as rxr
import rasterio
from rasterio.transform import Affine
from tqdm import tqdm


# -----------------------------
# Model download/load
# -----------------------------
def download_model(model_dir: Path) -> None:
    """Download SEN2SR model files if not already present."""
    if not model_dir.exists():
        model_dir.mkdir(parents=True, exist_ok=True)
        mlstac.download(
            file=(
                "https://huggingface.co/tacofoundation/sen2sr/resolve/main/"
                "SEN2SRLite/NonReference_RGBN_x4/mlm.json"
            ),
            output_dir=str(model_dir),
        )


def load_model(device: torch.device):
    """Load compiled SEN2SR model to device."""
    model_dir = Path("model/SEN2SR_RGBN")
    download_model(model_dir)
    model = mlstac.load(str(model_dir)).compiled_model(device=device)
    model.eval()
    model.to(device)
    return model


# -----------------------------
# I/O helpers
# -----------------------------
def build_output_path(in_path: Path, out_root: Path) -> Path:
    """Create an output filename for the cropped SR label chip."""
    name = in_path.name
    if name.lower().endswith(".tif"):
        stem = name[:-4]
    else:
        stem = name
    return out_root / f"{stem}_sr2p5m_label256.tif"


def read_chip_as_tensor(path: Path, assume_order: str) -> torch.Tensor:
    """
    Read a 4-band GeoTIFF and return torch tensor (4, H, W) float32 in [0,1].

    assume_order:
      - "B04B03B02B08" means reorder to B02,B03,B04,B08 for SEN2SR
      - "B02B03B04B08" means already correct
    """
    da = rxr.open_rasterio(path)  # (band, y, x)
    arr = da.to_numpy().astype("float32")  # (4, H, W)

    if arr.shape[0] != 4 or arr.shape[1] != 128 or arr.shape[2] != 128:
        raise ValueError(f"Expected (4,128,128), got {arr.shape} for {path}")

    if assume_order.upper() == "B04B03B02B08":
        # [R,G,B,NIR] -> [B,G,R,NIR]
        arr = arr[[2, 1, 0, 3], :, :]
    elif assume_order.upper() != "B02B03B04B08":
        raise ValueError(f"Unknown band order: {assume_order}")

    # Scale: if looks like 0..10000, convert to 0..1
    if float(np.nanmax(arr)) > 50.0:
        arr = arr / 10_000.0

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    return torch.from_numpy(arr)  # (4,128,128)


def write_geotiff(
    out_path: Path,
    data_uint16: np.ndarray,  # (4,256,256)
    crs,
    transform,
    compress: str = "zstd",
) -> None:
    """Write a 4-band uint16 GeoTIFF with given CRS+transform."""
    if data_uint16.shape != (4, 256, 256):
        raise ValueError(f"Expected (4,256,256), got {data_uint16.shape}")

    profile = {
        "driver": "GTiff",
        "height": 256,
        "width": 256,
        "count": 4,
        "dtype": "uint16",
        "crs": crs,
        "transform": transform,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": compress,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data_uint16[0], 1)
        dst.write(data_uint16[1], 2)
        dst.write(data_uint16[2], 3)
        dst.write(data_uint16[3], 4)


# -----------------------------
# Main processing
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run SEN2SR on 128x128 parent chips -> 512x512, crop bottom-right 256x256 label SR tile."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="/home/airg/gessuman/projects/ftw-mappingafrica-integration/data/s2_chips/primary_nomask_rgbnir_bigger_extent",
        help="Folder containing 128x128 parent chips (*.tif).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="/home/airg/gessuman/projects/ftw-mappingafrica-integration/data/s2_chips/primary_sr2p5m_label256",
        help="Output folder for cropped SR label chips (256x256).",
    )
    parser.add_argument("--device", type=str, default="auto", help='cuda|cpu|auto')
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)  # safest on HPC FS
    parser.add_argument(
        "--band_order",
        type=str,
        default="B04B03B02B08",
        help="Input band order. Use B04B03B02B08 if your chips are [R,G,B,NIR].",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    out_root = Path(args.out).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    tifs = sorted(root.glob("*.tif"))
    if not tifs:
        raise RuntimeError(f"No .tif files found in {root}")

    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Using device: {device}")

    model = load_model(device)

    # Simple manual batching (so we can keep per-file georeferencing clean)
    for i in tqdm(range(0, len(tifs), args.batch_size), desc="SR + crop label"):
        batch_paths = tifs[i : i + args.batch_size]

        # ---- read tensors ----
        imgs = []
        metas = []
        for p in batch_paths:
            try:
                # read georef from rasterio (fast + reliable)
                with rasterio.open(p) as src:
                    crs = src.crs
                    transform = src.transform

                x = read_chip_as_tensor(p, args.band_order)  # (4,128,128)
                imgs.append(x)
                metas.append((p, crs, transform))
            except Exception as e:
                tqdm.write(f"ERROR reading {p}: {e}")

        if not imgs:
            continue

        batch = torch.stack(imgs, dim=0).to(device)  # (B,4,128,128)

        # ---- model inference (x4 -> 512x512) ----
        with torch.inference_mode():
            sr = model(batch).detach().cpu().numpy()  # (B,4,512,512), float

        # ---- crop bottom-right 256x256 and write ----
        # bottom-right in pixel coords: rows 256:512, cols 256:512
        row0, col0 = 256, 256

        for (p, crs, transform), sr_img in zip(metas, sr):
            try:
                sr_crop = sr_img[:, row0:row0 + 256, col0:col0 + 256]  # (4,256,256)

                # scale back to uint16 reflectance-like (match your earlier convention)
                sr_crop_u16 = (
                    np.clip(sr_crop * 10_000.0, 0, 65_535).astype(np.uint16)
                )

                # Build transform for SR full grid (pixel size /4)
                t_sr_full = transform * Affine.scale(1 / 4, 1 / 4)

                # Shift to bottom-right crop window
                t_sr_crop = t_sr_full * Affine.translation(col0, row0)

                out_path = build_output_path(p, out_root)
                write_geotiff(out_path, sr_crop_u16, crs=crs, transform=t_sr_crop)

            except Exception as e:
                tqdm.write(f"ERROR processing {p}: {e}")


if __name__ == "__main__":
    main()
