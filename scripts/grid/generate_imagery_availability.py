#!/usr/bin/env python3

"""Generate year-specific MappingAfrica imagery availability catalogs."""

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.ops import unary_union

from ftw_ma.quad_index import QuadIndex
from ftw_ma.paths import CLEANED_QUAD_INDEX

def generate_availability(
    tiles,
    quad_index,
    year,
    country,
):
    records = []

    total = len(tiles)

    for i, row in enumerate(tiles.itertuples(index=False), 1):
        geom = row.geometry

        selected = quad_index.select_with_neighbors(
            tile_geometry=geom,
            year=year,
            target_country=country,
        )

        # Exclude quads that only touch the tile boundary.
        if not selected.empty:
            overlap = selected.geometry.intersection(geom).area > 0
            selected = selected.loc[overlap].copy()

        quad_count = len(selected)

        if quad_count == 0:
            coverage_pct = 0.0
        else:
            footprint = unary_union(selected.geometry.tolist())
            intersection = geom.intersection(footprint)

            coverage_pct = (
                100.0 * intersection.area / geom.area
            )

        coverage_pct = float(
            np.clip(coverage_pct, 0.0, 100.0)
        )

        if coverage_pct <= 1e-6:
            status = "none"
        elif coverage_pct >= 100.0 - 1e-6:
            status = "full"
        else:
            status = "partial"

        records.append({
            "tile": int(row.tile),
            "year": int(year),
            "quad_count": int(quad_count),
            "imagery_coverage_pct": coverage_pct,
            "has_imagery": bool(quad_count > 0),
            "coverage_status": status,
            "geometry": geom,
        })

        if i % 5000 == 0 or i == total:
            print(f"Processed {i:,}/{total:,} tiles", flush=True)

    return gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs=tiles.crs,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tiles", required=True)
    parser.add_argument(
        "--quad-index",
        type=Path,
        default=CLEANED_QUAD_INDEX,
        help=(
            "Planet quad-index GeoParquet "
            f"(default: {CLEANED_QUAD_INDEX})"
        ),
    )
    parser.add_argument("--country", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    output = Path(args.output)

    # Refuse overwrites before expensive processing.
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite: {output}"
        )

    tiles = gpd.read_file(args.tiles)

    if tiles.crs is None:
        raise ValueError("Tile catalog has no CRS.")

    tiles = tiles.to_crs("EPSG:4326")

    if tiles["tile"].duplicated().any():
        raise ValueError("Duplicate tile IDs detected.")

    quad_index = QuadIndex(args.quad_index)

    availability = generate_availability(
        tiles=tiles,
        quad_index=quad_index,
        year=args.year,
        country=args.country,
    )

    output.parent.mkdir(parents=True, exist_ok=True)

    availability.to_file(
        output,
        driver="GeoJSON",
    )

    print("\n=== IMAGERY AVAILABILITY SUMMARY ===")
    print("Country:", args.country)
    print("Year:", args.year)
    print("Total tiles:", len(availability))

    print(
        availability["coverage_status"]
        .value_counts()
        .to_string()
    )

    print("Output:", output)


if __name__ == "__main__":
    main()