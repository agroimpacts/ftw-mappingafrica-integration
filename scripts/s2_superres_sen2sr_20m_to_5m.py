#!/usr/bin/env python3
"""
s2_superres_sen2sr_20m_to_5m.py

Pipeline:
---------
Input  : Sentinel-2 chips, 256x256 pixels at ~10 m
Step 1 : Downscale to 128x128 (same footprint, ~20 m effective)
Step 2 : SEN2SR RGBN x4 super-resolution
         128x128  →  512x512  (4x upsampling in pixel space)
Output : 512x512 chips at ~5 m, *same geographic footprint*
         as the original 256x256 @10 m S2 chip.

Notes:
------
- Expects 4-band S2 (RGBN) with shape (4, 256, 256).
- SEN2SR expects bands in order [B02 (Blue), B03 (Green), B04 (Red), B08 (NIR)].
- We auto-detect reflectance scaling:
    * If max > 50 → assume 0–10_000, divide by 10_000.
    * Else       → assume already 0–1, leave as-is.
- Output written as Cloud Optimized GeoTIFFs (COG) with CRS/extent preserved.
"""

import argparse
from pathlib import Path

import mlstac
import rioxarray as rio
import torch
import torch.nn.functional as F
from rasterio.enums import Resampling
from tqdm import tqdm


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
class S2ChipDataset(torch.utils.data.Dataset):
    """
    Dataset over 4-band, 256x256 Sentinel-2 chips in a flat directory.

    For each chip:
    - Read GeoTIFF (C, H, W) = (4, 256, 256)
    - (Optionally) reorder bands to B02,B03,B04,B08
    - Auto-scale to 0–1 reflectance if needed
    - Downscale to 128x128 (same footprint) using bilinear interpolation
    - Return tensor (4, 128, 128)
    """

    def __init__(self, root: str):
        self.root = Path(root).expanduser()
        self.images = sorted(self.root.glob("*.tif"))

        if len(self.images) == 0:
            raise RuntimeError(f"No .tif files found in {self.root}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        path = self.images[idx]

        # Read as (band, y, x)
        da = rio.open_rasterio(path)
        arr = da.to_numpy().astype("float32")  # (C, H, W)
        C, H, W = arr.shape

        if H != 256 or W != 256:
            raise ValueError(f"Expected 256x256 chips, got {arr.shape} for {path}")

        # ---------------------------------------------------------
        # 1. ENSURE CORRECT BAND ORDER (if needed)
        # ---------------------------------------------------------
        # SEN2SR expects [B02 (Blue), B03 (Green), B04 (Red), B08 (NIR)].
        #
        # If your chips are in [B04, B03, B02, B08] order, uncomment this:
        # arr = arr[[2, 1, 0, 3], :, :]
        #
        # If already B02,B03,B04,B08, do nothing.

        # ---------------------------------------------------------
        # 2. AUTO-DETECT SCALING (0–1 vs 0–10_000)
        # ---------------------------------------------------------
        max_val = float(arr.max())
        if max_val > 50.0:
            # Looks like raw reflectance (0–10_000)
            arr = arr / 10_000.0
        # else: assume already scaled 0–1

        # ---------------------------------------------------------
        # 3. DOWNSCALE 256x256 → 128x128 (same footprint)
        # ---------------------------------------------------------
        # Use bilinear interpolation on the tensor.
        img = torch.from_numpy(arr).unsqueeze(0)  # (1, C, H, W)
        img_ds = F.interpolate(
            img,
            size=(128, 128),
            mode="bilinear",
            align_corners=False,
        )
        tensor = img_ds.squeeze(0)  # (C, 128, 128)

        # Clean NaNs/Infs just in case
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

        return {"image": tensor, "path": str(path)}


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def download_model(model_dir: Path) -> None:
    """
    Download SEN2SR RGBN x4 model into `model_dir` if not already present.
    """
    if not model_dir.exists():
        model_dir.mkdir(parents=True, exist_ok=True)
        mlstac.download(
            file=(
                "https://huggingface.co/tacofoundation/sen2sr/resolve/main/"
                "SEN2SRLite/NonReference_RGBN_x4/mlm.json"
            ),
            output_dir=str(model_dir),
        )


def build_output_path(in_path: str, out_root: Path) -> Path:
    """
    Construct output filename for 512x512 5 m SR chip.

    Example
    -------
    GH0000006_2022-08_S2L2A_nomask.tif
      -> GH0000006_2022-08_S2L2A_sr5m_512.tif
    """
    in_path = Path(in_path)
    name = in_path.name

    if "S2L2A_nomask" in name:
        new_name = name.replace("S2L2A_nomask", "S2L2A_sr5m_512")
    else:
        stem = name[:-4] if name.lower().endswith(".tif") else name
        new_name = stem + "_sr5m_512.tif"

    return out_root / new_name


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main(args: argparse.Namespace) -> None:
    # -----------------------------
    # 1. Device & model
    # -----------------------------
    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Using device: {device}")

    model_dir = Path("model/SEN2SR_RGBN")
    download_model(model_dir)

    print("Loading SEN2SR model...")
    model = mlstac.load(str(model_dir)).compiled_model(device=device)
    model.eval()
    model.to(device)

    # -----------------------------
    # 2. Dataset & DataLoader
    # -----------------------------
    dataset = S2ChipDataset(args.root)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    out_root = Path(args.out).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(dataset)} input chips in {dataset.root}")
    print(f"Outputs will be written to: {out_root}")

    # -----------------------------
    # 3. Inference loop
    # -----------------------------
    printed_stats = False

    for batch in tqdm(dataloader, total=len(dataloader), desc="Super-resolving"):
        images = batch["image"].to(device)  # (B, C, 128, 128)
        B, C, H, W = images.shape

        if H != 128 or W != 128:
            raise ValueError(f"Expected 128x128 inputs, got {H}x{W}")

        # Debug stats once
        if (not printed_stats) and args.debug_stats:
            print(
                "DEBUG: input tensor stats min/mean/max:",
                images.min().item(),
                images.mean().item(),
                images.max().item(),
            )
            printed_stats = True

        # Run SEN2SR
        with torch.inference_mode():
            sr_images = model(images).cpu()  # (B, C, 512, 512)
            sr_images = (
                sr_images.mul(10_000.0)
                .clamp(0.0, 65_535.0)
                .to(torch.uint16)
                .numpy()
            )

        # -------------------------
        # 4. Write each chip (512x512 @ ~5 m)
        # -------------------------
        for path, sr in zip(batch["path"], sr_images):
            try:
                # Open original for CRS + footprint
                da = rio.open_rasterio(path)

                _, h_sr, w_sr = sr.shape  # expect (4, 512, 512)

                # Reproject original to 512x512 grid over same extent
                up = da.rio.reproject(
                    da.rio.crs,
                    shape=(h_sr, w_sr),
                    resample=Resampling.bilinear,
                )

                # Replace values with SR output (no flips)
                up.values = sr

                out_path = build_output_path(path, out_root)
                out_path.parent.mkdir(parents=True, exist_ok=True)

                up.rio.to_raster(
                    out_path,
                    driver="COG",
                    compress="zstd",
                    dtype="uint16",
                )

            except Exception as e:
                tqdm.write(f"ERROR processing {path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Downscale S2 chips (256x256 @10 m) to 128x128, run SEN2SR x4, "
            "and write 512x512 (~5 m) super-resolved chips."
        )
    )

    parser.add_argument(
        "--root",
        type=str,
        default=(
            "/home/airg/gessuman/projects/ftw-mappingafrica-integration/"
            "data/s2_chips/primary_nomask_rgbnir"
        ),
        help="Directory containing input 10 m S2 chips (*.tif).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=(
            "/home/airg/gessuman/projects/ftw-mappingafrica-integration/"
            "data/s2_chips/windowB_sr5m_512"
        ),
        help="Directory to write 512x512 ~5 m SR chips.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='"cuda", "cpu", or "auto" (default: auto chooses cuda if available).',
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of chips per batch.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader worker processes for parallel I/O.",
    )
    parser.add_argument(
        "--debug_stats",
        action="store_true",
        help="Print min/mean/max of the first input batch for sanity-check.",
    )

    args = parser.parse_args()
    main(args)
