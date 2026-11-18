#!/usr/bin/env python

from pathlib import Path
import re

import argparse
import geopandas as gpd
import pandas as pd
import rasterio
from shapely.geometry import box


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DEFAULT_SOURCE_ROOT = Path(
    "/home/airg/TChakraborty/work_ra/gee_nicfi"
)

DEFAULT_OUTPUT_DIR = Path(
    "/home/airg/gessuman/working/catalogs/source_imagery"
)

# Directories that are not valid operational source collections
EXCLUDED_COLLECTIONS = {
    ".ipynb_checkpoints",
    "nigeria_2024_dec_faulty_quads",
    "qa_view_nigeria_2024_dec",
    "qc_tiles_from_quads",
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def extract_year(collection_name):
    """Extract a four-digit year from a collection directory name."""

    match = re.search(r"(20\d{2})", collection_name)

    if match is None:
        return None

    return int(match.group(1))


def extract_source_country(collection_name):
    """Determine source country from collection directory name."""

    name = collection_name.lower()

    country_prefixes = {
        "ethiopia": "ethiopia",
        "ghana": "ghana",
        "kenya": "kenya",
        "malawi": "malawi",
        "mozambique": "mozambique",
        "nigeria": "nigeria",
        "tanz": "tanzania",
        "uganda": "uganda",
        "zimbabwe": "zimbabwe",
    }

    for prefix, country in country_prefixes.items():
        if name.startswith(prefix):
            return country

    return None


def extract_quad_id(path):
    """Extract GRID identifier from a TIFF filename."""

    match = re.search(r"(GRID\d+)", path.stem, re.IGNORECASE)

    if match:
        return match.group(1).upper()

    # Fallback for unexpected filename patterns
    return path.stem


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build a spatial index of Planet NICFI TIFFs."
    )

    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Directory containing Planet NICFI collections.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the generated index and error report.",
    )

    args = parser.parse_args()

    source_root = args.source_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    index_path = output_dir / "planet_nicfi_quad_index.parquet"
    error_path = output_dir / "planet_nicfi_quad_index_errors.csv"

    if not source_root.is_dir():
        parser.error(
            f"Source directory does not exist: {source_root}"
        )

    for path in (index_path, error_path):
        if path.exists():
            parser.error(
                f"Refusing to overwrite existing file: {path}"
            )

    records = []
    errors = []

    collections = sorted(
        p
        for p in source_root.iterdir()
        if p.is_dir()
        and p.name not in EXCLUDED_COLLECTIONS
    )

    print(f"Source root: {source_root}")
    print(f"Collections found: {len(collections)}")
    print()

    for collection_dir in collections:

        collection = collection_dir.name
        year = extract_year(collection)
        source_country = extract_source_country(collection)

        if year is None:
            print(
                f"Skipping {collection}: "
                "could not determine year."
            )
            continue

        if source_country is None:
            print(
                f"Skipping {collection}: "
                "could not determine source country."
            )
            continue

        tif_paths = sorted(
            list(collection_dir.glob("*.tif"))
            + list(collection_dir.glob("*.tiff"))
        )

        print(
            f"{collection}: "
            f"{len(tif_paths)} TIFFs"
        )

        for tif_path in tif_paths:

            try:
                with rasterio.open(tif_path) as src:

                    if src.crs is None:
                        raise ValueError(
                            "Raster has no CRS."
                        )

                    source_bounds = src.bounds
                    source_crs = src.crs

                    footprint = box(
                        source_bounds.left,
                        source_bounds.bottom,
                        source_bounds.right,
                        source_bounds.top,
                    )

                    footprint_gdf = gpd.GeoDataFrame(
                        {"geometry": [footprint]},
                        crs=source_crs,
                    )

                    footprint_4326 = (
                        footprint_gdf
                        .to_crs("EPSG:4326")
                        .geometry
                        .iloc[0]
                    )

                    records.append(
                        {
                            "quad_id": extract_quad_id(
                                tif_path
                            ),
                            "year": year,
                            "source_country": source_country,
                            "source_collection": collection,
                            "quad_path": str(
                                tif_path.resolve()
                            ),
                            "source_crs": str(
                                source_crs
                            ),
                            "width": src.width,
                            "height": src.height,
                            "geometry": footprint_4326,
                        }
                    )

            except Exception as exc:

                errors.append(
                    {
                        "quad_path": str(
                            tif_path.resolve()
                        ),
                        "source_collection": collection,
                        "year": year,
                        "source_country": source_country,
                        "error": str(exc),
                    }
                )

                print(
                    f"  ERROR: {tif_path.name}: "
                    f"{exc}"
                )

    # -----------------------------------------------------------------
    # Build operational GeoParquet
    # -----------------------------------------------------------------

    if not records:
        raise RuntimeError(
            "No valid source quads were found."
        )

    quad_index = gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs="EPSG:4326",
    )

    quad_index = quad_index.sort_values(
        [
            "year",
            "source_country",
            "source_collection",
            "quad_id",
            "quad_path",
        ]
    ).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    quad_index.to_parquet(
        index_path,
        index=False,
    )

    # -----------------------------------------------------------------
    # Error report
    # -----------------------------------------------------------------

    error_df = pd.DataFrame(errors)

    error_df.to_csv(
        error_path,
        index=False,
    )

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    print()
    print("=" * 70)
    print("QUAD INDEX COMPLETE")
    print("=" * 70)

    print(
        f"Valid quads:   {len(quad_index):,}"
    )

    print(
        f"Failed quads:  {len(error_df):,}"
    )

    print(
        f"Index CRS:     {quad_index.crs}"
    )

    print(
        f"Index written: {index_path}"
    )

    print(
        f"Errors written: {error_path}"
    )

    print()

    summary = (
        quad_index
        .groupby(
            [
                "source_country",
                "year",
                "source_collection",
            ]
        )
        .size()
        .reset_index(name="quad_count")
    )

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()