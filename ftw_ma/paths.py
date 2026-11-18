"""Shared repository paths for MappingAfrica reference datasets."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = REPO_ROOT / "data"

AFRICA_BOUNDARIES = (
    DATA_DIR / "boundaries" / "africa_countries_50m.gpkg"
)

CLEANED_QUAD_INDEX = (
    DATA_DIR
    / "quad_index"
    / "planet_combined_quad_index_cleaned.parquet"
)