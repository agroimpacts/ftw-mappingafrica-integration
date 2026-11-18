from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Tuple, Union
import subprocess

import os
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from rasterio.features import geometry_mask
from shapely.geometry.base import BaseGeometry

from .quad_index import QuadIndex


DST_CRS = CRS.from_epsg(4326)

DST_RES = 0.005 / 200
DST_W = 2048
DST_H = 2048

GDALBUILDVRT = Path(
    "/home/airg/rabedi/mambaforge/envs/geo/bin/gdalbuildvrt"
)


def snapped_extent_from_tile(
    tile_geometry: BaseGeometry,
) -> Tuple[float, float, float, float]:
    """
    Reproduce the production Mozambique tiler's exact destination extent.
    """

    minx, miny, maxx, maxy = tile_geometry.bounds

    minx = np.round(minx / DST_RES) * DST_RES
    maxy = np.round(maxy / DST_RES) * DST_RES

    maxx = minx + DST_W * DST_RES
    miny = maxy - DST_H * DST_RES

    return (
        float(minx),
        float(miny),
        float(maxx),
        float(maxy),
    )


def _build_vrt(
    quad_paths,
    vrt_path: Path,
) -> Path:
    """
    Build a temporary VRT using the same nodata behavior as the
    historical production tiler.

    gdalbuildvrt comes from a separate GDAL environment, so give that
    subprocess its matching GDAL/PROJ data paths without modifying the
    parent Python/Rasterio environment.
    """

    cmd = [
        str(GDALBUILDVRT),
        "-overwrite",
        "-srcnodata",
        "0",
        "-vrtnodata",
        "0",
        str(vrt_path),
        *[str(p) for p in quad_paths],
    ]

    env = os.environ.copy()

    env["GDAL_DATA"] = (
        "/home/airg/rabedi/mambaforge/envs/geo/share/gdal"
    )

    env["PROJ_LIB"] = (
        "/home/airg/rabedi/mambaforge/envs/geo/share/proj"
    )

    env["PROJ_DATA"] = (
        "/home/airg/rabedi/mambaforge/envs/geo/share/proj"
    )

    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    return vrt_path


def _warp_to_memory(
    source_path: Union[str, Path],
    extent: Tuple[float, float, float, float],
) -> Tuple[np.ndarray, Dict]:
    """
    Reproject/resample a source raster or VRT directly into a
    2048 x 2048 UInt16 NumPy array.

    No imagery TIFF is written.
    """

    left, bottom, right, top = extent

    transform = Affine(
        DST_RES,
        0.0,
        left,
        0.0,
        -DST_RES,
        top,
    )

    with rasterio.open(source_path) as src:

        if src.count != 4:
            raise ValueError(
                f"Expected 4 source bands, found {src.count} "
                f"in {source_path}"
            )

        image_data = np.zeros(
            (src.count, DST_H, DST_W),
            dtype=np.uint16,
        )

        for band_idx in range(1, src.count + 1):

            reproject(
                source=rasterio.band(src, band_idx),
                destination=image_data[band_idx - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=0,
                dst_transform=transform,
                dst_crs=DST_CRS,
                dst_nodata=0,
                resampling=Resampling.cubic,
            )

    profile = {
        "driver": "GTiff",
        "dtype": "uint16",
        "nodata": 0,
        "width": DST_W,
        "height": DST_H,
        "count": 4,
        "crs": DST_CRS,
        "transform": transform,
    }

    return image_data, profile


def validate_tile_coverage(
    image_data: np.ndarray,
) -> Dict:
    """
    Check whether a buffered imagery tile has complete raster coverage.

    A spatial pixel is considered missing when all imagery bands
    contain nodata (0).
    """

    if image_data.ndim != 3:
        raise ValueError(
            "Expected image_data with shape (bands, height, width)."
        )

    missing = np.all(image_data == 0, axis=0)

    total_pixels = int(missing.size)
    missing_pixels = int(missing.sum())
    valid_pixels = total_pixels - missing_pixels

    coverage_percent = (
        100.0 * valid_pixels / total_pixels
        if total_pixels > 0
        else 0.0
    )

    return {
        "total_pixels": total_pixels,
        "valid_pixels": valid_pixels,
        "missing_pixels": missing_pixels,
        "coverage_percent": coverage_percent,
        "complete": missing_pixels == 0,
    }


def _prepare_historical_vrts(quad_paths, tmpdir):
    """
    Create temporary four-band VRTs for supplementary historical quads.

    The historical GeoTIFFs have five bands. Only bands 1-4
    are passed to the existing four-band warping pipeline.
    """
    gdal_translate = Path(
        "/home/airg/rabedi/mambaforge/envs/geo/bin/gdal_translate"
    )

    geo_env = Path("/home/airg/rabedi/mambaforge/envs/geo")

    env = os.environ.copy()
    env["GDAL_DATA"] = str(geo_env / "share/gdal")
    env["PROJ_LIB"] = str(geo_env / "share/proj")
    env["PROJ_DATA"] = str(geo_env / "share/proj")
    env["GTIFF_SRS_SOURCE"] = "EPSG"

    outputs = []

    for index, source in enumerate(quad_paths):
        output = tmpdir / f"historical_{index:03d}_4band.vrt"

        subprocess.run(
            [
                str(gdal_translate),
                "-of", "VRT",
                "-a_srs", "EPSG:3857",
                "-b", "1",
                "-b", "2",
                "-b", "3",
                "-b", "4",
                str(source),
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )

        outputs.append(output)

    return outputs


def _warp_source_group(quad_paths, extent, tmpdir, name):
    """
    Preserve existing production mosaicking semantics.
    """
    if not quad_paths:
        return None, None

    if len(quad_paths) == 1:
        return _warp_to_memory(
            source_path=quad_paths[0],
            extent=extent,
        )

    vrt_path = tmpdir / f"{name}.vrt"

    _build_vrt(
        quad_paths=quad_paths,
        vrt_path=vrt_path,
    )

    return _warp_to_memory(
        source_path=vrt_path,
        extent=extent,
    )


def build_tile_in_memory(
    tile_geometry: BaseGeometry,
    year: int,
    target_country: str,
    quad_index: QuadIndex,
) -> Tuple[np.ndarray, Dict]:
    """
    Construct a four-band MappingAfrica imagery tile in memory.

    Preserve existing source mosaicking and use supplementary
    historical imagery only where original imagery is nodata.
    """
    selected = quad_index.select_with_neighbors(
        tile_geometry=tile_geometry,
        year=year,
        target_country=target_country,
    )

    if selected.empty:
        raise RuntimeError(
            f"No source quads found for year {year} "
            f"intersecting the requested {target_country} tile."
        )

    # The cleaned index identifies imagery behavior using source_role.
    # Historical fallback imagery fills nodata without overwriting
    # valid pixels from the original imagery group.
    supplementary = selected[
        selected["source_role"] == "historical_fallback"
    ]

    existing = selected[
        selected["source_role"] != "historical_fallback"
    ]

    existing_paths = [
        Path(p) for p in existing["quad_path"].tolist()
    ]

    supplementary_paths = [
        Path(p) for p in supplementary["quad_path"].tolist()
    ]

    all_paths = existing_paths + supplementary_paths

    missing = [
        str(path)
        for path in all_paths
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Selected source quad files do not exist:\\n"
            + "\\n".join(missing)
        )

    extent = snapped_extent_from_tile(tile_geometry)

    with TemporaryDirectory(prefix="ftw_quad_vrt_") as tmp:
        tmpdir = Path(tmp)

        original_image, original_profile = _warp_source_group(
            existing_paths,
            extent,
            tmpdir,
            "original",
        )

        historical_image = None
        historical_profile = None

        if supplementary_paths:
            historical_vrts = _prepare_historical_vrts(
                supplementary_paths,
                tmpdir,
            )

            historical_image, historical_profile = _warp_source_group(
                historical_vrts,
                extent,
                tmpdir,
                "historical",
            )

        # Historical-only coverage.
        if original_image is None:
            if historical_image is None:
                raise RuntimeError("No usable imagery sources.")

            return historical_image, historical_profile

        # Preserve original production imagery when there
        # are no supplementary historical sources.
        if historical_image is None:
            return original_image, original_profile

        if (
            original_profile["transform"]
            != historical_profile["transform"]
            or original_profile["crs"]
            != historical_profile["crs"]
        ):
            raise RuntimeError(
                "Original and historical tile grids do not match."
            )

        original_missing = np.all(
            original_image == 0,
            axis=0,
        )

        historical_valid = ~np.all(
            historical_image == 0,
            axis=0,
        )

        fallback_mask = original_missing & historical_valid

        composite = original_image.copy()

        composite[:, fallback_mask] = historical_image[
            :, fallback_mask
        ]

        return composite, original_profile
