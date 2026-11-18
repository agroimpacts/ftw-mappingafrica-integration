
#!/usr/bin/env python3
"""Generate or reuse a country-level 0.5-degree coverage grid."""

import argparse
import math
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box
from ftw_ma.paths import AFRICA_BOUNDARIES


CELL_SIZE = 0.5


def generate_cells(boundary, margin_cells=0):
    """Generate globally aligned cells with positive-area boundary overlap."""

    minx, miny, maxx, maxy = boundary.bounds

    start_col = math.floor(minx / CELL_SIZE) - margin_cells
    end_col = math.ceil(maxx / CELL_SIZE) + margin_cells

    start_row = math.floor(miny / CELL_SIZE) - margin_cells
    end_row = math.ceil(maxy / CELL_SIZE) + margin_cells

    cells = []

    for col in range(start_col, end_col):
        for row in range(start_row, end_row):
            x = col * CELL_SIZE
            y = row * CELL_SIZE

            geom = box(x, y, x + CELL_SIZE, y + CELL_SIZE)

            if geom.intersection(boundary).area > 0:
                cells.append((col, row, geom))

    # Expand the selected cells by a configurable grid-cell margin.
    selected = {(col, row) for col, row, _ in cells}

    if margin_cells > 0:
        expanded = set()

        for col, row in selected:
            for dx in range(-margin_cells, margin_cells + 1):
                for dy in range(-margin_cells, margin_cells + 1):
                    expanded.add((col + dx, row + dy))

        selected = expanded

    records = []

    for col, row in sorted(selected):
        x = col * CELL_SIZE
        y = row * CELL_SIZE

        records.append({
            "cell_col": col,
            "cell_row": row,
            "geometry": box(
                x, y,
                x + CELL_SIZE,
                y + CELL_SIZE,
            ),
        })

    return gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs="EPSG:4326",
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--country", required=True)
    parser.add_argument(
        "--boundary-file",
        type=Path,
        default=AFRICA_BOUNDARIES,
        help=(
            "Country boundary GeoPackage or GeoJSON "
            f"(default: {AFRICA_BOUNDARIES})"
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--margin-cells", type=int, default=0)
    parser.add_argument("--reference-grid")

    args = parser.parse_args()

    if args.margin_cells < 0:
        parser.error("--margin-cells must be nonnegative")

    output = Path(args.output)

    if args.reference_grid:
        grid = gpd.read_file(args.reference_grid)

        if grid.crs is None:
            raise ValueError("Reference grid has no CRS")

        grid = grid.to_crs("EPSG:4326")
        mode = "reference"

    else:
        countries = gpd.read_file(args.boundary_file)

        if countries.crs is None:
            raise ValueError("Boundary dataset has no CRS")

        countries = countries.to_crs("EPSG:4326")

        # Resolve common country names and ISO codes.
        aliases = {
            "tanzania": "TZA",
            "ivory coast": "CIV",
            "cote d'ivoire": "CIV",
            "eswatini": "SWZ",
            "swaziland": "SWZ",
            "dr congo": "COD",
            "drc": "COD",
            "republic of the congo": "COG",
        }

        country_input = args.country.strip()
        country_key = country_input.casefold()

        admin_match = (
            countries["ADMIN"].astype(str).str.strip().str.casefold()
            == country_key
        )

        iso_code = aliases.get(country_key, country_input.upper())

        iso_match = (
            countries["ISO_A3"].astype(str).str.upper()
            == iso_code
        )

        match = countries.loc[admin_match | iso_match]

        if len(match) != 1:
            raise ValueError(
                f"Expected one boundary for {args.country}; "
                f"found {len(match)}"
            )

        grid = generate_cells(
            match.geometry.iloc[0],
            margin_cells=args.margin_cells,
        )
        mode = "generate"

    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {output}"
        )

    grid.to_file(output, driver="GeoJSON")

    print("=== COVERAGE GRID GENERATED ===")
    print("Mode:", mode)
    print("Country:", args.country)
    print("Cells:", len(grid))
    print("CRS:", grid.crs)
    print("Output:", output)


if __name__ == "__main__":
    main()
