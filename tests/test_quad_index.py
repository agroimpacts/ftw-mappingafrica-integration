"""tests for cleaned Planet quad selection."""

import geopandas as gpd
import pytest
from shapely.geometry import box

from ftw_ma.quad_index import QuadIndex


TILE = box(30.0, -10.0, 31.0, -9.0)


def make_index(tmp_path, records):
    """Create a small synthetic GeoParquet index."""
    frame = gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs="EPSG:4326",
    )

    path = tmp_path / "quads.parquet"
    frame.to_parquet(path, index=False)

    return QuadIndex(path)


def record(
    quad_id,
    planet_quad_id,
    country,
    path,
    geometry,
    role="primary",
    year=2024,
):
    return {
        "quad_id": quad_id,
        "planet_quad_id": planet_quad_id,
        "year": year,
        "source_country": country,
        "source_collection": f"{country}_{year}",
        "source_role": role,
        "quad_path": path,
        "geometry": geometry,
    }


def test_target_country_priority(tmp_path):
    """Prefer Tanzania when two countries contain the same NICFI quad."""

    index = make_index(tmp_path, [
        record("A", "P1", "kenya", "/kenya/a.tif", TILE),
        record("B", "P1", "tanzania", "/tanzania/a.tif", TILE),
    ])

    result = index.select_with_neighbors(TILE, 2024, "tanzania")

    assert len(result) == 1
    assert result.iloc[0]["source_country"] == "tanzania"
    assert result.iloc[0]["quad_path"] == "/tanzania/a.tif"


def test_neighbor_country_fallback(tmp_path):
    """Include neighboring imagery when it has a distinct Planet quad ID."""

    index = make_index(tmp_path, [
        record("A", "P1", "tanzania", "/tanzania/a.tif", TILE),
        record("B", "P2", "kenya", "/kenya/b.tif", TILE),
    ])

    result = index.select_with_neighbors(TILE, 2024, "tanzania")

    assert len(result) == 2
    assert set(result["planet_quad_id"]) == {"P1", "P2"}


def test_preserve_historical_sources(tmp_path):
    """Retain distinct historical records with overlapping footprints."""

    index = make_index(tmp_path, [
        record(
            "A", "P1", "tanzania",
            "/tanzania/a.tif", TILE,
        ),
        record(
            "H1", None, "tanzania",
            "/historical/one.tif", TILE,
            role="original_historical",
        ),
        record(
            "H2", None, "tanzania",
            "/historical/two.tif", TILE,
            role="historical_fallback",
        ),
    ])

    result = index.select_with_neighbors(TILE, 2024, "tanzania")

    assert len(result) == 3
    assert set(result["quad_path"]) == {
        "/tanzania/a.tif",
        "/historical/one.tif",
        "/historical/two.tif",
    }


def test_duplicate_quad_paths_rejected(tmp_path):
    """Reject different records that resolve to the same source path."""

    index = make_index(tmp_path, [
        record("A", "P1", "tanzania", "/shared/a.tif", TILE),
        record(
            "H1", None, "tanzania",
            "/shared/a.tif", TILE,
            role="original_historical",
        ),
    ])

    with pytest.raises(ValueError, match="Duplicate quad_path"):
        index.select_with_neighbors(TILE, 2024, "tanzania")


def test_invalid_source_role_rejected(tmp_path):
    """Reject unrecognized source-role metadata."""

    index = make_index(tmp_path, [
        record(
            "A", "P1", "tanzania",
            "/tanzania/a.tif", TILE,
            role="unexpected_role",
        ),
    ])

    with pytest.raises(ValueError, match="Unexpected source_role"):
        index.select_with_neighbors(TILE, 2024, "tanzania")


def test_primary_duplicate_ids_rejected(tmp_path):
    """Primary-only selection must reject duplicate target-country IDs."""

    index = make_index(tmp_path, [
        record("A", "P1", "tanzania", "/tanzania/a.tif", TILE),
        record("A", "P2", "tanzania", "/tanzania/b.tif", TILE),
    ])

    with pytest.raises(ValueError, match="same quad_id"):
        index.select_primary(TILE, 2024, "tanzania")