import numpy as np
import pytest
from shapely.geometry import box

from ftw_ma.on_the_fly_imagery import (
    DST_H,
    DST_RES,
    DST_W,
    snapped_extent_from_tile,
    validate_tile_coverage,
)


def test_snapped_extent_dimensions():
    tile = box(30.001, -10.001, 30.051, -9.951)

    left, bottom, right, top = snapped_extent_from_tile(tile)

    assert (right - left) == pytest.approx(DST_W * DST_RES)
    assert (top - bottom) == pytest.approx(DST_H * DST_RES)

    assert left / DST_RES == pytest.approx(round(left / DST_RES))
    assert top / DST_RES == pytest.approx(round(top / DST_RES))


def test_complete_imagery_coverage():
    image = np.ones((4, 10, 10), dtype=np.uint16)

    result = validate_tile_coverage(image)

    assert result["coverage_percent"] == 100.0
    assert result["missing_pixels"] == 0
    assert result["complete"] is True


def test_partial_imagery_coverage():
    image = np.ones((4, 10, 10), dtype=np.uint16)
    image[:, :, :5] = 0

    result = validate_tile_coverage(image)

    assert result["coverage_percent"] == 50.0
    assert result["missing_pixels"] == 50
    assert result["complete"] is False


def test_missing_imagery_coverage():
    image = np.zeros((4, 10, 10), dtype=np.uint16)

    result = validate_tile_coverage(image)

    assert result["coverage_percent"] == 0.0
    assert result["missing_pixels"] == 100
    assert result["complete"] is False

def test_historical_fallback_preserves_original_pixels(
    tmp_path, monkeypatch
):
    """Historical imagery fills nodata without replacing valid pixels."""

    from pathlib import Path
    import geopandas as gpd
    from ftw_ma import on_the_fly_imagery as otf

    tile = box(30.0, -10.0, 30.05, -9.95)

    selected = gpd.GeoDataFrame(
        [
            {
                "source_role": "primary",
                "quad_path": str(tmp_path / "original.tif"),
                "geometry": tile,
            },
            {
                "source_role": "historical_fallback",
                "quad_path": str(tmp_path / "historical.tif"),
                "geometry": tile,
            },
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )

    class FakeQuadIndex:
        def select_with_neighbors(self, **kwargs):
            return selected

    for name in ("original.tif", "historical.tif"):
        (tmp_path / name).touch()

    original = np.zeros((4, 2, 3), dtype=np.uint16)
    historical = np.zeros((4, 2, 3), dtype=np.uint16)

    # Valid original pixel must remain unchanged.
    original[:, 0, 0] = 100

    # Historical imagery covers the original pixel and one missing pixel.
    historical[:, 0, 0] = 200
    historical[:, 0, 1] = 300

    # Last pixel remains missing in both sources.
    profile = {
        "transform": "synthetic_transform",
        "crs": "EPSG:4326",
    }

    def fake_warp_source_group(paths, extent, tmpdir, name):
        if name == "original":
            return original.copy(), profile.copy()
        if name == "historical":
            return historical.copy(), profile.copy()
        raise AssertionError(f"Unexpected group: {name}")

    monkeypatch.setattr(
        otf,
        "_warp_source_group",
        fake_warp_source_group,
    )

    monkeypatch.setattr(
        otf,
        "_prepare_historical_vrts",
        lambda paths, tmpdir: [
            Path(tmpdir) / "historical_000_4band.vrt"
        ],
    )

    result, result_profile = otf.build_tile_in_memory(
        tile_geometry=tile,
        year=2024,
        target_country="mozambique",
        quad_index=FakeQuadIndex(),
    )

    assert np.all(result[:, 0, 0] == 100)
    assert np.all(result[:, 0, 1] == 300)
    assert np.all(result[:, 0, 2] == 0)
    assert result_profile == profile


def test_otf_prediction_crop_preserves_alignment():
    """Verify OTF cropping preserves pixel values and geographic alignment."""

    from affine import Affine
    from ftw_ma.inference import _crop_otf_prediction

    # Encode pixel coordinates into the prediction values.
    rows, cols = np.indices((2048, 2048))
    prediction = (
        rows * 2048 + cols
    )[np.newaxis, ...].astype(np.int32)

    transform = Affine(
        0.000025, 0, 30.0,
        0, -0.000025, -10.0,
    )

    profile = {
        "transform": transform,
        "height": 2048,
        "width": 2048,
    }

    cropped, cropped_profile = _crop_otf_prediction(
        prediction,
        profile,
    )

    assert cropped.shape == (1, 2000, 2000)

    # Check that the retained pixels correspond to the exact
    # 24-pixel offset in the original prediction.
    assert cropped[0, 0, 0] == prediction[0, 24, 24]
    assert cropped[0, -1, -1] == prediction[0, 2023, 2023]

    expected_transform = transform * Affine.translation(24, 24)

    assert cropped_profile["transform"] == expected_transform
    assert cropped_profile["height"] == 2000
    assert cropped_profile["width"] == 2000

    # Original profile must remain unchanged.
    assert profile["transform"] == transform
    assert profile["height"] == 2048
    assert profile["width"] == 2048

    # The cropped raster's upper-left geographic coordinate
    # must match the original raster at pixel (24, 24).
    assert cropped_profile["transform"] * (0, 0) == pytest.approx(
        transform * (24, 24)
    )

    # Verify the lower-right geographic boundary as well.
    assert cropped_profile["transform"] * (2000, 2000) == pytest.approx(
        transform * (2024, 2024)
    )


def test_otf_prediction_crop_rejects_unexpected_dimensions():
    """Unexpected prediction dimensions must raise an error."""

    from affine import Affine
    from ftw_ma.inference import _crop_otf_prediction

    prediction = np.zeros((1, 2047, 2048), dtype=np.uint8)

    profile = {
        "transform": Affine.identity(),
        "height": 2047,
        "width": 2048,
    }

    with pytest.raises(
        ValueError,
        match="Expected on-the-fly prediction size",
    ):
        _crop_otf_prediction(prediction, profile)
