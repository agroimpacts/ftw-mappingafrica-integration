"""CPU-only regression tests for the MappingAfrica grid pipeline."""

import geopandas as gpd
import pytest
from shapely.geometry import box

from scripts.grid.generate_mappingafrica_tiles import (
    generate_tiles,
    generate_buffered_tiles,
)
from scripts.grid.generate_imagery_availability import (
    generate_availability,
)

from scripts.grid.generate_coverage_grid import generate_cells

def test_tile_generation_is_deterministic():
    """Repeated generation should produce identical tile IDs and geometry."""

    coverage = gpd.GeoDataFrame(
        {"cell_id": [1]},
        geometry=[box(30.0, -10.0, 30.5, -9.5)],
        crs="EPSG:4326",
    )

    first = generate_tiles(coverage)
    second = generate_tiles(coverage)

    # The generator includes tiles touching the coverage boundary.
    # A 0.5-degree cell therefore selects a 12 x 12 tile grid.
    assert len(first) == 144
    assert len(second) == 144

    first = first.sort_values("tile").reset_index(drop=True)
    second = second.sort_values("tile").reset_index(drop=True)

    assert first["tile"].tolist() == second["tile"].tolist()
    assert first.geometry.equals(second.geometry)
    assert first["tile"].is_unique


def test_buffered_tile_dimensions():
    """A 24-pixel buffer should produce 0.0512-degree tiles."""

    coverage = gpd.GeoDataFrame(
        {"cell_id": [1]},
        geometry=[box(30.0, -10.0, 30.5, -9.5)],
        crs="EPSG:4326",
    )

    centers = generate_tiles(coverage)
    buffered = generate_buffered_tiles(
        centers,
        buffer_pixels=24,
    )

    assert len(buffered) == len(centers)
    assert set(buffered["tile"]) == set(centers["tile"])

    for geom in buffered.geometry:
        minx, miny, maxx, maxy = geom.bounds

        assert maxx - minx == pytest.approx(0.0512)
        assert maxy - miny == pytest.approx(0.0512)

def test_coverage_grid_positive_area():
    """Only cells with positive-area overlap should be selected."""

    boundary = box(30.1, -9.9, 30.4, -9.6)

    cells = generate_cells(boundary, margin_cells=0)

    assert len(cells) == 1
    assert cells.iloc[0]["cell_col"] == 60
    assert cells.iloc[0]["cell_row"] == -20
    assert cells.geometry.iloc[0].equals(
        box(30.0, -10.0, 30.5, -9.5)
    )


def test_coverage_grid_excludes_boundary_only_contacts():
    """Touching a grid-cell boundary should not add another cell."""

    boundary = box(30.0, -10.0, 30.5, -9.5)

    cells = generate_cells(boundary, margin_cells=0)

    assert len(cells) == 1


def test_coverage_grid_margin():
    """A one-cell margin should expand the selection to 3 x 3."""

    boundary = box(30.1, -9.9, 30.4, -9.6)

    cells = generate_cells(boundary, margin_cells=1)

    assert len(cells) == 9

    actual = set(
        zip(cells["cell_col"], cells["cell_row"])
    )

    expected = {
        (col, row)
        for col in range(59, 62)
        for row in range(-21, -18)
    }

    assert actual == expected

class FakeQuadIndex:
    """Minimal spatial index for testing availability classifications."""

    def __init__(self, footprints):
        self.footprints = footprints

    def select_with_neighbors(
        self,
        tile_geometry,
        year,
        target_country,
    ):
        matching = [
            geom
            for geom in self.footprints
            if geom.intersects(tile_geometry)
        ]

        return gpd.GeoDataFrame(
            geometry=matching,
            crs="EPSG:4326",
        )


@pytest.mark.parametrize(
    "footprints, expected_status, expected_pct",
    [
        ([box(0, 0, 1, 1)], "full", 100.0),
        ([box(0, 0, 0.5, 1)], "partial", 50.0),
        ([], "none", 0.0),
    ],
)
def test_imagery_availability_classification(
    footprints,
    expected_status,
    expected_pct,
):
    """Availability should reflect positive-area quad-footprint coverage."""

    tiles = gpd.GeoDataFrame(
        {"tile": [12345]},
        geometry=[box(0, 0, 1, 1)],
        crs="EPSG:4326",
    )

    result = generate_availability(
        tiles=tiles,
        quad_index=FakeQuadIndex(footprints),
        year=2024,
        country="tanzania",
    )

    assert len(result) == 1
    assert result.iloc[0]["coverage_status"] == expected_status
    assert result.iloc[0]["imagery_coverage_pct"] == pytest.approx(
        expected_pct
    )

    assert result.iloc[0]["has_imagery"] == bool(footprints)