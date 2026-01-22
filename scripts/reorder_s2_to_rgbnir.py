#!/usr/bin/env python3
"""
Reorder Sentinel-2 chips from [B02,B03,B04,B08] (Blue,Green,Red,NIR)
to [B04,B03,B02,B08] (Red,Green,Blue,NIR) and save to a new folder.

Usage:
    python reorder_s2_to_rgbnir.py
"""

from pathlib import Path
import rasterio as rio
import numpy as np
from tqdm import tqdm

# === Configuration ===
IN_DIR  = Path("/home/airg/gessuman/projects/ftw-mappingafrica-integration/data/s2_chips/primary_nomask_1")
OUT_DIR = IN_DIR.parent / "primary_nomask_rgbnir"   # new folder beside 'primary'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# === Process all .tif files ===
tifs = sorted(list(IN_DIR.glob("*.tif")))
print(f"Found {len(tifs)} Sentinel-2 chips to reorder...")

for src_path in tqdm(tifs, desc="Reordering to RGBNIR", unit="file"):
    with rio.open(src_path) as src:
        # assume order = [B02, B03, B04, B08]
        if src.count < 4:
            print(f"{src_path.name}: has {src.count} bands, skipping.")
            continue

        b02 = src.read(1)   # Blue
        b03 = src.read(2)   # Green
        b04 = src.read(3)   # Red
        b08 = src.read(4)   # NIR

        # new order = [Red, Green, Blue, NIR]
        stack_rgbnir = np.stack([b04, b03, b02, b08], axis=0)

        profile = src.profile.copy()
        profile.update(count=4)

        dst_path = OUT_DIR / src_path.name
        with rio.open(dst_path, "w", **profile) as dst:
            for i in range(4):
                dst.write(stack_rgbnir[i], i+1)

print(f"\n Reordered chips saved to: {OUT_DIR}")
