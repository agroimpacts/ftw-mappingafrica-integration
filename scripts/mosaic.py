#!/usr/bin/env python3
"""
Memory-safe, Dask-parallel COG mosaic builder (no GDAL CLI required).

Drop-in replacement for the original gdal-based mosaic.py script.

Example:
    python mosaic.py --in-dir ./tiles --out-cog ./mosaic.tif
        --required-crs EPSG:4326 --cog-compress ZSTD --cog-blocksize 2048
        --force-tr 0.0001 0.0001 --workers 4 --threads-per-worker 2
"""

import argparse
import logging
import sys
from pathlib import Path

import dask
import rioxarray as rxr
import xarray as xr
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster
from tqdm import tqdm
from rioxarray.merge import merge_arrays


# -------------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
log = logging.getLogger("mosaic")


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def open_tile_lazy(path: Path, chunksize: int, required_crs: str = None):
    """Open raster lazily with rioxarray, assign CRS if missing."""
    da = rxr.open_rasterio(
        str(path),
        chunks={"x": chunksize, "y": chunksize},
        masked=True,
        cache=False,
    )
    if (da.rio.crs is None or da.rio.crs == "") and required_crs:
        da = da.rio.write_crs(required_crs, inplace=False)
    return da


def reproject_tile(da: xr.DataArray, target_crs: str, target_resolution=None,
                   resampling="nearest"):
    """Reproject raster to target CRS and optional forced resolution."""
    if da.rio.crs and str(da.rio.crs) == str(target_crs):
        return da
    if target_resolution:
        trx, try_ = target_resolution
        return da.rio.reproject(target_crs, resolution=(trx, try_), 
                                resampling=resampling)
    else:
        return da.rio.reproject(target_crs, resampling=resampling)


# -------------------------------------------------------------------------
# Core pipeline
# -------------------------------------------------------------------------
def build_and_write_mosaic(
    in_dir: Path,
    out_cog: Path,
    pattern: str = "*.tif",
    required_crs: str = None,
    force_tr=None,
    force_srs=None,
    src_nodata="255",
    cog_compress="ZSTD",
    cog_blocksize="2000",
    cog_predictor="2",
    cog_overviews="AUTO",
    cog_threads="ALL_CPUS",
    chunksize=2048,
    workers=1,
    threads_per_worker=1,
):
    # ---------------------------------------------------------------
    # Setup
    # ---------------------------------------------------------------
    tiles = sorted(in_dir.glob(pattern))
    if not tiles:
        raise FileNotFoundError(f"No files matching {pattern} in {in_dir}")
    log.info(f"Found {len(tiles)} tiles in {in_dir}")

    # Use first tile CRS unless forced
    first_da = rxr.open_rasterio(str(tiles[0]), masked=True)
    detected_first_crs = str(first_da.rio.crs) if first_da.rio.crs else None
    first_da.close()

    target_crs = force_srs or required_crs or detected_first_crs
    if not target_crs:
        raise RuntimeError("No CRS available. Use --required-crs or --force-srs.")
    log.info(f"Target CRS: {target_crs}")

    # ---------------------------------------------------------------
    # Dask setup
    # ---------------------------------------------------------------
    if workers > 1 or threads_per_worker > 1:
        cluster = LocalCluster(n_workers=workers, 
                               threads_per_worker=threads_per_worker)
        client = Client(cluster)
        log.info(f"Dask client started: {client.dashboard_link}")
    else:
        dask.config.set(scheduler="threads")
        client = None

    # ---------------------------------------------------------------
    # Open & reproject tiles (lazy)
    # ---------------------------------------------------------------
    arrays = []
    for p in tqdm(tiles, desc="Opening tiles", unit="file"):
        da = open_tile_lazy(p, chunksize, required_crs=required_crs)
        if str(da.rio.crs) != str(target_crs):
            da = reproject_tile(da, target_crs, force_tr)
        arrays.append(da)

    log.info(f"All tiles opened and reprojected lazily (Dask arrays).")

    # ---------------------------------------------------------------
    # Merge tiles (lazy)
    # ---------------------------------------------------------------
    log.info("Building mosaic (lazy merge)...")
    mosaic = merge_arrays(arrays, method="first")

    # Apply nodata if specified
    if src_nodata is not None:
        try:
            nodata_val = float(src_nodata)
        except ValueError:
            nodata_val = None
        if nodata_val is not None:
            mosaic = mosaic.rio.write_nodata(nodata_val, inplace=False)

    out_cog.parent.mkdir(parents=True, exist_ok=True)

   # ---------------------------------------------------------------
# Write to COG (triggers computation)
# ---------------------------------------------------------------
log.info("Writing COG...")

# Remove _FillValue to prevent xarray encoding conflict
if "_FillValue" in mosaic.attrs:
    del mosaic.attrs["_FillValue"]

# Ensure nodata is properly set
if src_nodata is not None:
    try:
        nodata_val = float(src_nodata)
    except ValueError:
        nodata_val = None
    if nodata_val is not None:
        mosaic = mosaic.rio.write_nodata(nodata_val, inplace=False)

with ProgressBar():
    mosaic.rio.to_raster(
        str(out_cog),
        driver="COG",
        compress=cog_compress,
        BLOCKSIZE=int(cog_blocksize),
        PREDICTOR=int(cog_predictor),
        BIGTIFF="IF_SAFER",
        OVERVIEWS=cog_overviews,
        RESAMPLING="NEAREST",
        NUM_THREADS=cog_threads,
    )

log.info(f"COG written successfully: {out_cog}")

if __name__ == "__main__":
    main(sys.argv[1:])
