
#!/usr/bin/env python3
"""Generate canonical MappingAfrica 0.05-degree center tiles."""

import argparse
import math
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union


TILE_SIZE = 0.05
SELECTION_TOLERANCE_DEG = 0.000001
PIXEL_RESOLUTION = 0.000025
DEFAULT_BUFFER_PIXELS = 24


def generate_tiles(coverage):
    coverage_union = unary_union(coverage.geometry.tolist())

    minx, miny, maxx, maxy = coverage_union.bounds

    start_col = math.floor(minx / TILE_SIZE) - 1
    end_col = math.ceil(maxx / TILE_SIZE) + 1

    start_row = math.floor(miny / TILE_SIZE) - 1
    end_row = math.ceil(maxy / TILE_SIZE) + 1

    records = []

    for col in range(start_col, end_col):
        for row in range(start_row, end_row):
            x0 = round(col * TILE_SIZE, 10)
            y0 = round(row * TILE_SIZE, 10)

            geom = box(
                x0,
                y0,
                round(x0 + TILE_SIZE, 10),
                round(y0 + TILE_SIZE, 10),
            )

            if not geom.intersects(coverage_union):
                continue

            tile_id = int(
                (col + 10000) * 1000
                + (row + 10000)
            )

            records.append({
                "tile": tile_id,
                "tile_col": col,
                "tile_row": row,
                "type": "center",
                "geometry": geom,
            })

    return gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs="EPSG:4326",
    )

def generate_buffered_tiles(center_tiles, buffer_pixels=24):
    """Expand center-tile bounds by a specified number of pixels."""

    buffer_deg = buffer_pixels * PIXEL_RESOLUTION

    buffered = center_tiles.copy()

    buffered["geometry"] = buffered.geometry.apply(
        lambda geom: box(
            round(geom.bounds[0] - buffer_deg, 10),
            round(geom.bounds[1] - buffer_deg, 10),
            round(geom.bounds[2] + buffer_deg, 10),
            round(geom.bounds[3] + buffer_deg, 10),
        )
    )

    buffered["type"] = "buffered"

    return buffered


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--coverage-grid", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--buffered-output")
    parser.add_argument(
        "--buffer-pixels",
        type=int,
        default=DEFAULT_BUFFER_PIXELS,
    )

    args = parser.parse_args()

    if args.buffer_pixels < 0:
        parser.error("--buffer-pixels must be nonnegative")

    coverage = gpd.read_file(args.coverage_grid).to_crs(
        "EPSG:4326"
    )

    tiles = generate_tiles(coverage)

    output = Path(args.output)

    # Validate all output paths before writing files.
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite: {output}"
        )

    buffered_path = None

    if args.buffered_output:
        buffered_path = Path(args.buffered_output)

        if buffered_path.resolve() == output.resolve():
            raise ValueError(
                "Center and buffered output paths must differ"
            )

        if buffered_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite: {buffered_path}"
            )

    # Write center tiles.
    output.parent.mkdir(parents=True, exist_ok=True)
    tiles.to_file(output, driver="GeoJSON")

    # Write buffered tiles if requested.
    if buffered_path is not None:
        buffered = generate_buffered_tiles(
            tiles,
            buffer_pixels=args.buffer_pixels,
        )

        buffered_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        buffered.to_file(
            buffered_path,
            driver="GeoJSON",
        )

        print("Buffered tiles:", len(buffered))
        print("Buffer pixels:", args.buffer_pixels)
        print("Buffered output:", buffered_path)

    print("=== MAPPINGAFRICA TILE GENERATION ===")
    print("Coverage cells:", len(coverage))
    print("Center tiles:", len(tiles))
    print("Unique tile IDs:", tiles["tile"].nunique())
    print("CRS:", tiles.crs)
    print("Output:", output)

if __name__ == "__main__":
    main()
