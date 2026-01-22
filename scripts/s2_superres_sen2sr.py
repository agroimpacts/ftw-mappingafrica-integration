#----------------------------------------------
#----------------------------------------------
# WINDOW A
#-----------------------------------------------
#-----------------------------------------------

# #!/usr/bin/env python3
# """
# s2_superres_sen2sr.py

# Upsample Sentinel-2 WindowA chips from 10 m → 2.5 m using the SEN2SR RGBN x4 model.

# - Expects 4-band S2 chips (RGBN) with shape (4, 256, 256)
# - Splits each chip into 4 patches of size 128×128
# - Runs SEN2SR on the 4 patches (LR 128→ HR 512)
# - Stitches them back into a 1024×1024 SR chip
# - Keeps CRS + geotransform and writes COGs
# """

# import argparse
# from pathlib import Path

# import mlstac
# import rioxarray as rio
# import torch
# from rasterio.enums import Resampling
# from tqdm import tqdm


# class S2WindowADataset(torch.utils.data.Dataset):
#     """Dataset over S2 WindowA chips in a flat directory."""

#     def __init__(self, root: str):
#         self.root = Path(root).expanduser()
#         self.images = sorted(self.root.glob("*.tif"))

#         if len(self.images) == 0:
#             raise RuntimeError(f"No .tif files found in {self.root}")

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         path = self.images[idx]
#         da = rio.open_rasterio(path)  # (band, y, x)

#         arr = da.to_numpy()  # (C, H, W)
#         if arr.shape[1] != 256 or arr.shape[2] != 256:
#             raise ValueError(f"Expected 256x256 chips, got {arr.shape} for {path}")

#         tensor = torch.from_numpy(arr).to(torch.float32) / 10_000.0
#         tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

#         return {"image": tensor, "path": str(path)}


# def download_model(model_dir: Path):
#     """Download SEN2SR model if not present."""
#     if not model_dir.exists():
#         model_dir.mkdir(parents=True, exist_ok=True)
#         mlstac.download(
#             file=(
#                 "https://huggingface.co/tacofoundation/sen2sr/resolve/main/"
#                 "SEN2SRLite/NonReference_RGBN_x4/mlm.json"
#             ),
#             output_dir=str(model_dir),
#         )


# def build_output_path(in_path: str, out_root: Path) -> Path:
#     """Construct a clean, predictable output filename."""
#     in_path = Path(in_path)
#     name = in_path.name

#     if "S2L2A_nomask" in name:
#         new_name = name.replace("S2L2A_nomask", "S2L2A_sr2p5m")
#     else:
#         if name.lower().endswith(".tif"):
#             new_name = name[:-4] + "_sr2p5m.tif"
#         else:
#             new_name = name + "_sr2p5m.tif"

#     return out_root / new_name


# def main(args):
#     # -----------------------------
#     # 1. Device & model
#     # -----------------------------
#     device_str = args.device
#     if device_str == "auto":
#         device_str = "cuda" if torch.cuda.is_available() else "cpu"
#     device = torch.device(device_str)
#     print(f"Using device: {device}")

#     model_dir = Path("model/SEN2SR_RGBN")
#     download_model(model_dir)

#     print("Loading SEN2SR model...")
#     model = mlstac.load(str(model_dir)).compiled_model(device=device)
#     model.eval()
#     model.to(device)

#     # -----------------------------
#     # 2. Dataset & DataLoader
#     # -----------------------------
#     dataset = S2WindowADataset(args.root)
#     dataloader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=args.num_workers,
#         pin_memory=(device.type == "cuda"),
#     )

#     out_root = Path(args.out).expanduser()
#     out_root.mkdir(parents=True, exist_ok=True)

#     print(f"Found {len(dataset)} input chips.")
#     print(f"Outputs will be written to: {out_root}")

#     # -----------------------------
#     # 3. Inference loop
#     # -----------------------------
#     for batch in tqdm(dataloader, total=len(dataloader), desc="Super-resolving"):
#         images = batch["image"].to(device)  # (B, C, 256, 256)
#         B, C, H, W = images.shape

#         if H != 256 or W != 256:
#             raise ValueError(f"Expected 256x256 chips in batch, got {H}x{W}")

#         # Split each 256×256 chip into 2×2 patches of 128×128
#         # images: (B, C, 256, 256) →
#         # patches: (B * 4, C, 128, 128)
#         n_h = 2
#         n_w = 2
#         patches = (
#             images.view(B, C, n_h, 128, n_w, 128)
#             .permute(0, 2, 4, 1, 3, 5)
#             .reshape(B * n_h * n_w, C, 128, 128)
#         )

#         with torch.inference_mode():
#             sr_patches = model(patches).cpu()  # (B*4, C, 512, 512)
#             sr_patches = (
#                 sr_patches.mul(10_000.0)
#                 .clamp(0.0, 65_535.0)
#                 .to(torch.uint16)
#             )

#         # Reassemble patches back into full SR chips: (B, C, 1024, 1024)
#         sr_patches = sr_patches.view(B, n_h, n_w, C, 512, 512)  # (B,2,2,C,512,512)
#         sr_patches = sr_patches.permute(0, 3, 1, 4, 2, 5)       # (B,C,2,512,2,512)
#         sr_full = sr_patches.reshape(B, C, 512 * n_h, 512 * n_w)  # (B,C,1024,1024)
#         sr_full = sr_full.numpy()

#         # -------------------------
#         # 4. Write each chip
#         # -------------------------
#         for path, sr in zip(batch["path"], sr_full):
#             try:
#                 da = rio.open_rasterio(path)

#                 # sr is (C, H_sr, W_sr) = (4, 1024, 1024)
#                 _, h_sr, w_sr = sr.shape

#                 # Reproject original to super-res grid to keep CRS/transform
#                 up = da.rio.reproject(
#                     da.rio.crs,
#                     shape=(h_sr, w_sr),
#                     resample=Resampling.bilinear,
#                 )

#                 # Flip along y-axis to match SEN2SR output orientation
#                 up.values = sr[:, ::-1, :]

#                 out_path = build_output_path(path, out_root)
#                 out_path.parent.mkdir(parents=True, exist_ok=True)

#                 up.rio.to_raster(
#                     out_path,
#                     driver="COG",
#                     compress="zstd",
#                     dtype="uint16",
#                 )

#             except Exception as e:
#                 tqdm.write(f"ERROR processing {path}: {e}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Upsample Sentinel-2 WindowA chips from 10 m to 2.5 m using SEN2SR."
#     )
#     parser.add_argument(
#         "--root",
#         type=str,
#         default=(
#             "/home/airg/gessuman/projects/ftw-mappingafrica-integration/"
#             "data/s2_chips/windowA_nomask"
#         ),
#         help="Directory containing input 10 m S2 chips (*.tif).",
#     )
#     parser.add_argument(
#         "--out",
#         type=str,
#         default=(
#             "/home/airg/gessuman/projects/ftw-mappingafrica-integration/"
#             "data/s2_chips/windowA_sr"
#         ),
#         help="Directory to write 2.5 m super-resolved chips.",
#     )
#     parser.add_argument(
#         "--device",
#         type=str,
#         default="auto",
#         help='"cuda", "cpu", or "auto" (default: auto).',
#     )
#     parser.add_argument("--batch_size", type=int, default=8)
#     parser.add_argument("--num_workers", type=int, default=4)

#     args = parser.parse_args()
#     main(args)


#----------------------------------------------
#----------------------------------------------
# WINDOW B
#-----------------------------------------------
#-----------------------------------------------


#!/usr/bin/env python3
"""
s2_superres_sen2sr.py

Upsample Sentinel-2 chips from 10 m → 2.5 m using the SEN2SR RGBN x4 model.

Assumptions
-----------
- Each input chip is a 4-band (RGBN) GeoTIFF with shape (4, 256, 256)
- Pixel values are Sentinel-2 surface reflectance scaled by 1e4
- We want an output chip covering the same area at 4x finer resolution:
    (4, 1024, 1024) at ~2.5 m
- Model: SEN2SR Lite RGBN x4, loaded via mlstac

High-level steps
----------------
1. Load all *.tif files in an input directory as a dataset.
2. For each chip: normalize reflectance and split into 4 patches (128x128).
3. Run SEN2SR on all patches in batches on GPU/CPU.
4. Stitch patches back into a full 1024x1024 chip.
5. Write as a Cloud Optimized GeoTIFF (COG) with original CRS/transform.
"""

import argparse
from pathlib import Path

import mlstac
import rioxarray as rio
import torch
from rasterio.enums import Resampling
from tqdm import tqdm


class S2ChipDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for 4-band, 256x256 Sentinel-2 chips in a flat directory.

    Parameters
    ----------
    root : str
        Path to a directory containing *.tif chips.

    Behavior
    --------
    - On init: collects a sorted list of all .tif files in `root`.
    - __getitem__(idx): returns a dict with:
        - "image" : torch.Tensor of shape (4, 256, 256), float32, normalized by 1e4
        - "path"  : str path to the original file
    """

    def __init__(self, root: str):
        # Store root as a Path object and expand '~' if present.
        self.root = Path(root).expanduser()
        # List all .tif files in the directory (no recursion).
        self.images = sorted(self.root.glob("*.tif"))

        # Fail early if no images are found.
        if len(self.images) == 0:
            raise RuntimeError(f"No .tif files found in {self.root}")

    def __len__(self):
        # Number of samples in the dataset.
        return len(self.images)

    def __getitem__(self, idx):
        """
        Load chip `idx`, normalize, and return image + path.

        Returns
        -------
        dict
            {
                "image": torch.Tensor[4, 256, 256],  # normalized reflectance
                "path":  str                         # file path
            }
        """
        path = self.images[idx]

        # Open raster as xarray.DataArray with dims (band, y, x).
        da = rio.open_rasterio(path)

        # Convert to numpy array with shape (C, H, W).
        arr = da.to_numpy()

        # Sanity check: ensure chips are 256x256 spatially.
        if arr.shape[1] != 256 or arr.shape[2] != 256:
            raise ValueError(f"Expected 256x256 chips, got {arr.shape} for {path}")

        # Convert to float32 tensor and normalize by 10_000 to get ~0–1 range.
        tensor = torch.from_numpy(arr).to(torch.float32) / 10_000.0

        # Replace NaNs and +/- inf with 0 (safer for CNNs).
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

        return {"image": tensor, "path": str(path)}


def download_model(model_dir: Path) -> None:
    """
    Download the SEN2SR RGBN x4 model into `model_dir` if not present.

    Parameters
    ----------
    model_dir : Path
        Directory where the SEN2SR model definition + weights will be stored.
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
    Construct a clean output filename for the super-resolved chip.

    Logic
    -----
    - If the filename contains 'S2L2A_nomask', replace with 'S2L2A_sr2p5m'.
    - Else, append '_sr2p5m' before the .tif extension (or at the end).

    Parameters
    ----------
    in_path : str
        Path to the input chip.
    out_root : Path
        Root directory where outputs should be written.

    Returns
    -------
    Path
        Full path to the output file.
    """
    in_path = Path(in_path)
    name = in_path.name

    # Common patterns in your project.
    if "S2L2A_nomask" in name:
        new_name = name.replace("S2L2A_nomask", "S2L2A_sr2p5m")
    else:
        if name.lower().endswith(".tif"):
            new_name = name[:-4] + "_sr2p5m.tif"
        else:
            new_name = name + "_sr2p5m.tif"

    return out_root / new_name


def main(args: argparse.Namespace) -> None:
    """
    Main entry point: load model, dataset, run super-resolution, and write outputs.
    """
    # -----------------------------
    # 1. Device & model
    # -----------------------------
    device_str = args.device
    if device_str == "auto":
        # Auto-select GPU if available; otherwise use CPU.
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Using device: {device}")

    model_dir = Path("model/SEN2SR_RGBN")
    download_model(model_dir)

    print("Loading SEN2SR model...")
    # mlstac.load() reads the mlm.json and builds the model.
    # compiled_model(device=...) returns a torch.nn.Module optimized for given device.
    model = mlstac.load(str(model_dir)).compiled_model(device=device)
    model.eval()          # Put model in eval mode (no dropout, etc.).
    model.to(device)      # Move model weights to GPU/CPU.

    # -----------------------------
    # 2. Dataset & DataLoader
    # -----------------------------
    dataset = S2ChipDataset(args.root)

    # DataLoader handles batching + parallel I/O.
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,                     # Keep deterministic order.
        num_workers=args.num_workers,      # Number of worker processes for loading.
        pin_memory=(device.type == "cuda"),  # Helps transfers to GPU.
    )

    out_root = Path(args.out).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(dataset)} input chips.")
    print(f"Outputs will be written to: {out_root}")

    # -----------------------------
    # 3. Inference loop
    # -----------------------------
    # tqdm shows a progress bar over all batches.
    for batch in tqdm(dataloader, total=len(dataloader), desc="Super-resolving"):
        # images: (B, C, 256, 256)
        images = batch["image"].to(device)
        B, C, H, W = images.shape

        if H != 256 or W != 256:
            raise ValueError(f"Expected 256x256 chips in batch, got {H}x{W}")

        # We want to split each 256x256 chip into 4 smaller 128x128 patches:
        #   top-left, top-right, bottom-left, bottom-right.
        #
        # images: (B, C, 256, 256)
        #   -> reshape to (B, C, 2, 128, 2, 128)
        #   -> permute to (B, 2, 2, C, 128, 128) so that spatial tiles come first
        #   -> reshape to (B*4, C, 128, 128) as a big batch of patches for the model.
        n_h = 2  # number of tiles along height
        n_w = 2  # number of tiles along width
        patches = (
            images.view(B, C, n_h, 128, n_w, 128)   # (B, C, 2, 128, 2, 128)
            .permute(0, 2, 4, 1, 3, 5)              # (B, 2, 2, C, 128, 128)
            .reshape(B * n_h * n_w, C, 128, 128)    # (B*4, C, 128, 128)
        )

        # -------------------------
        # 3a. Run SEN2SR on patches
        # -------------------------
        with torch.inference_mode():  # Disable gradient tracking for efficiency.
            # Model output per patch: (C, 512, 512) at 2.5 m resolution.
            sr_patches = model(patches).cpu()

            # Convert back to reflectance scaled by 1e4 and cast to uint16.
            sr_patches = (
                sr_patches.mul(10_000.0)     # scale from ~0–1 back to ~0–10_000
                .clamp(0.0, 65_535.0)        # clip to uint16 range
                .to(torch.uint16)
            )

        # -------------------------
        # 3b. Reassemble patches into full SR chips
        # -------------------------
        # sr_patches currently: (B*4, C, 512, 512)
        # Reshape back into grid of tiles, then stitch into (B, C, 1024, 1024).
        sr_patches = sr_patches.view(B, n_h, n_w, C, 512, 512)   # (B,2,2,C,512,512)
        sr_patches = sr_patches.permute(0, 3, 1, 4, 2, 5)        # (B,C,2,512,2,512)
        sr_full = sr_patches.reshape(B, C, 512 * n_h, 512 * n_w) # (B,C,1024,1024)

        # Move to numpy for assignment into xarray object later.
        sr_full = sr_full.numpy()

        # -------------------------
        # 4. Write each chip to disk
        # -------------------------
        for path, sr in zip(batch["path"], sr_full):
            try:
                # Re-open the original image to get CRS, transform, and metadata.
                da = rio.open_rasterio(path)

                # sr has shape (C, H_sr, W_sr) = (4, 1024, 1024)
                _, h_sr, w_sr = sr.shape

                # Reproject original image to super-res grid shape so that:
                # - CRS stays the same
                # - transform is consistent with new resolution
                up = da.rio.reproject(
                    da.rio.crs,
                    shape=(h_sr, w_sr),
                    resample=Resampling.bilinear,
                )

                # Orientation fix:
                # The original FTW/SEN2SR script flips the y-axis ([::-1] along rows).
                # We do the same to match expected orientation.
                up.values = sr[:, ::-1, :]

                # Build output path and ensure directory exists.
                out_path = build_output_path(path, out_root)
                out_path.parent.mkdir(parents=True, exist_ok=True)

                # Write as Cloud Optimized GeoTIFF with zstd compression.
                up.rio.to_raster(
                    out_path,
                    driver="COG",
                    compress="zstd",
                    dtype="uint16",
                )

            except Exception as e:
                # Don't crash on one bad chip: log error and continue.
                tqdm.write(f"ERROR processing {path}: {e}")


if __name__ == "__main__":
    # -----------------------------
    # CLI argument parsing
    # -----------------------------
    parser = argparse.ArgumentParser(
        description=(
            "Upsample Sentinel-2 chips from 10 m to 2.5 m using the "
            "SEN2SR RGBN x4 model."
        )
    )

    # Default paths are set for your windowB dataset, but can be overridden.
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
            "data/s2_chips/primary_sr"
        ),
        help="Directory to write 2.5 m super-resolved chips.",
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
        help="Number of chips per batch (before tiling into 4 patches each).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader worker processes for parallel I/O.",
    )

    args = parser.parse_args()
    main(args)
