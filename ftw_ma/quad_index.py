from pathlib import Path
from typing import Union

import geopandas as gpd
from shapely.geometry.base import BaseGeometry
import pandas as pd


class QuadIndex:
    """Spatial lookup and source selection for NICFI source quads."""

    def __init__(self, index_path: Union[str, Path]):
        """Load a NICFI quad index."""

        self.index_path = Path(index_path)

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"Quad index not found: {self.index_path}"
            )

        self.quads = gpd.read_parquet(self.index_path)

        if self.quads.crs is None:
            raise ValueError("Quad index has no CRS.")

        required_columns = {
            "quad_id",
            "year",
            "source_country",
            "source_collection",
            "quad_path",
            "geometry",
        }

        missing = required_columns - set(self.quads.columns)

        if missing:
            raise ValueError(
                f"Quad index is missing columns: {sorted(missing)}"
            )

    def query(
        self,
        tile_geometry: BaseGeometry,
        year: int,
    ) -> gpd.GeoDataFrame:
        """Return all physical quads intersecting a tile for a year."""

        year_quads = self.quads.loc[
            self.quads["year"] == int(year)
        ]

        if year_quads.empty:
            return year_quads.copy()

        candidate_positions = year_quads.sindex.query(
            tile_geometry,
            predicate="intersects",
        )

        matches = year_quads.iloc[
            candidate_positions
        ].copy()

        return matches.reset_index(drop=True)

    def select_primary(
        self,
        tile_geometry: BaseGeometry,
        year: int,
        target_country: str,
    ) -> gpd.GeoDataFrame:
        """Select target-country quads intersecting a tile.

        The target country's source imagery is preferred over copies of
        the same Planet grid cells from neighboring-country collections.

        Neighbor fallback is intentionally not performed here yet.
        """

        candidates = self.query(
            tile_geometry=tile_geometry,
            year=year,
        )

        if candidates.empty:
            return candidates

        target_country = target_country.lower()

        primary = candidates.loc[
            candidates["source_country"].str.lower()
            == target_country
        ].copy()

        if primary.empty:
            return primary

        # There should normally be only one physical source per
        # year + country + quad_id. Fail explicitly if that assumption
        # is violated rather than silently choosing one.
        duplicate_ids = primary.loc[
            primary["quad_id"].duplicated(keep=False),
            "quad_id",
        ].unique()

        if len(duplicate_ids) > 0:
            raise ValueError(
                "Multiple target-country source files found for "
                f"the same quad_id: {sorted(duplicate_ids)}"
            )

        # Use a deterministic spatial source order for VRT mosaicking:
        # south-to-north, then west-to-east.
        #
        # Historical production relied on spatial-join return ordering,
        # which was not explicitly defined. This deterministic rule has
        # been validated against representative historical Mozambique
        # multi-quad tiles and reproduced their imagery exactly.
        bounds = primary.geometry.bounds

        primary = primary.assign(
            _sort_south=bounds["miny"].to_numpy(),
            _sort_west=bounds["minx"].to_numpy(),
        )

        primary = (
            primary
            .sort_values(
                ["_sort_south", "_sort_west"],
                ascending=[True, True],
                kind="stable",
            )
            .drop(columns=["_sort_south", "_sort_west"])
        )

        return primary.reset_index(drop=True)

    def select_with_neighbors(
        self,
        tile_geometry: BaseGeometry,
        year: int,
        target_country: str,
    ) -> gpd.GeoDataFrame:
        """Select imagery using the cleaned Planet quad index.

        Deduplicate NICFI copies by original Planet grid ID,
        preferring the target country's source.

        Preserve all distinct historical imagery records.
        """
        candidates = self.query(
            tile_geometry=tile_geometry,
            year=year,
        )

        if candidates.empty:
            return candidates

        required = {
            "source_role",
            "planet_quad_id",
        }

        missing = required - set(candidates.columns)

        if missing:
            raise ValueError(
                "Cleaned quad index missing columns: "
                f"{sorted(missing)}"
            )

        candidates = candidates.copy()

        valid_roles = {
            "primary",
            "original_historical",
            "historical_fallback",
        }

        unknown = (
            set(candidates["source_role"].dropna())
            - valid_roles
        )

        if unknown or candidates["source_role"].isna().any():
            raise ValueError(
                "Unexpected source_role values: "
                f"{sorted(map(str, unknown))}"
            )

        target_country = target_country.lower()

        # --------------------------------------------------
        # NICFI sources
        # --------------------------------------------------

        nicfi = candidates.loc[
            candidates["source_role"] == "primary"
        ].copy()

        if not nicfi.empty:

            if nicfi["planet_quad_id"].isna().any():
                raise ValueError(
                    "NICFI records contain null planet_quad_id."
                )

            nicfi["_target_priority"] = (
                nicfi["source_country"]
                .astype(str)
                .str.lower()
                != target_country
            )

            nicfi = (
                nicfi
                .sort_values(
                    [
                        "_target_priority",
                        "source_country",
                        "planet_quad_id",
                        "quad_path",
                    ],
                    kind="stable",
                )
                .drop_duplicates(
                    subset=["planet_quad_id"],
                    keep="first",
                )
                .drop(columns="_target_priority")
            )

        # --------------------------------------------------
        # Historical sources
        # --------------------------------------------------

        historical = candidates.loc[
            candidates["source_role"].isin(
                [
                    "original_historical",
                    "historical_fallback",
                ]
            )
        ].copy()

        # Historical mosaics may share spatial footprints.
        # Preserve all distinct source records.
        selected = gpd.GeoDataFrame(
            pd.concat(
                [nicfi, historical],
                ignore_index=True,
            ),
            geometry="geometry",
            crs=candidates.crs,
        )

        if selected["quad_path"].duplicated().any():
            raise ValueError(
                "Duplicate quad_path values in selection."
            )

        # --------------------------------------------------
        # Preserve deterministic spatial ordering
        # --------------------------------------------------

        bounds = selected.geometry.bounds

        selected = selected.assign(
            _sort_south=bounds["miny"].to_numpy(),
            _sort_west=bounds["minx"].to_numpy(),
        )

        selected = (
            selected
            .sort_values(
                ["_sort_south", "_sort_west"],
                ascending=[True, True],
                kind="stable",
            )
            .drop(
                columns=[
                    "_sort_south",
                    "_sort_west",
                ]
            )
        )

        return selected.reset_index(drop=True)
