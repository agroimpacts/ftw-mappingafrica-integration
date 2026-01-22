# GLOBAL STATS FOR `primary_nomask_rgbnir` folder

#!/usr/bin/env python3
"""
Compute global band statistics (min, max, mean, std, p1, p2, p98, p99)
for one or more raster windows listed in a catalog CSV OR
for all TIFFs inside a directory.

Examples:

1) Folder-only mode (your case: MappingAfrica S2 window_b chips):
    python3 global_band_stats.py \
        --tiff-dir /home/airg/gessuman/projects/ftw-mappingafrica-integration/data/s2_chips/primary_nomask_rgbnir \
        --out-prefix /home/airg/gessuman/projects/ftw-mappingafrica-integration/data/global_band_stats/global_stats_ma_s2_window_b

2) MappingAfrica only, window_b from catalog:
    python3 global_band_stats.py \
        --catalog /home/airg/gessuman/projects/ftw-mappingafrica-integration/data/mappingafrica-catalog.csv \
        --windows window_b \
        --out-prefix /home/airg/gessuman/projects/ftw-mappingafrica-integration/data/global_band_stats_mappingafrica_window_b

3) FTW+MA combined, window_b only:
    python3 global_band_stats.py \
        --catalog /home/airg/gessuman/projects/ftw-mappingafrica-integration/data/ftw-mappingafrica-combined-catalog.csv \
        --windows window_b \
        --out-prefix /home/airg/gessuman/projects/ftw-mappingafrica-integration/data/global_band_stats_ftw_ma_combined_window_b
"""

from __future__ import annotations
import argparse
import json
import math
import os
import glob
from collections import defaultdict
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.io import DatasetReader
from tqdm import tqdm


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def resolve_path(path_str: str, root_dir: str) -> str:
    """If path_str is relative, prepend root_dir. Otherwise leave it as-is."""
    return path_str if os.path.isabs(path_str) else os.path.join(root_dir, path_str)


def welford_merge_update(
    sample_values: np.ndarray,
    running_state: Tuple[int, float, float],
) -> Tuple[int, float, float]:
    """Merge a new chunk of values into a (n, mean, M2) running state using Welford's method."""
    n_running, mean_running, M2_running = running_state
    if sample_values.size == 0:
        return running_state

    n_chunk: int = int(sample_values.size)
    mean_chunk: float = float(np.mean(sample_values))
    M2_chunk: float = float(np.sum((sample_values - mean_chunk) ** 2))

    delta: float = mean_chunk - mean_running
    n_total: int = n_running + n_chunk
    if n_total == 0:
        return running_state

    mean_total: float = mean_running + delta * (n_chunk / n_total)
    M2_total: float = M2_running + M2_chunk + (delta**2) * (n_running * n_chunk / n_total)

    return n_total, mean_total, M2_total


def iterate_valid_blocks(
    dataset: DatasetReader,
    band_indices: List[int],
) -> Iterator[Tuple[int, np.ndarray]]:
    """
    Yield (band_index, 1D finite pixel values) for each block window of the dataset,
    one band at a time. Nodata / NaN / Inf are removed.
    """
    read_kwargs = dict(masked=True, boundless=False)

    for _, window in dataset.block_windows(band_indices[0]):
        block_data = dataset.read(band_indices, window=window, **read_kwargs)  # shape (B, h, w)

        for local_band_idx, band_index in enumerate(band_indices):
            band_block = block_data[local_band_idx]

            if np.ma.isMaskedArray(band_block):
                vals = band_block.compressed()
            else:
                vals = band_block.ravel()

            vals = vals[np.isfinite(vals)]

            if vals.size:
                yield band_index, vals


def percentile_from_histogram(
    histogram_counts: np.ndarray,
    bin_edges: np.ndarray,
    percentile_q: float,
) -> float:
    """
    Estimate percentile_q (0-100) from histogram counts via interpolation.
    """
    total_count = np.sum(histogram_counts)
    if total_count == 0:
        return float("nan")

    target_rank = (percentile_q / 100.0) * total_count
    cumulative_counts = np.cumsum(histogram_counts)

    bin_index = int(np.searchsorted(cumulative_counts, target_rank, side="left"))
    bin_index = int(np.clip(bin_index, 0, len(histogram_counts) - 1))

    count_before = 0 if bin_index == 0 else cumulative_counts[bin_index - 1]
    count_in_bin = int(histogram_counts[bin_index])

    if count_in_bin == 0:
        return float(bin_edges[bin_index])

    fraction_within_bin = (target_rank - count_before) / count_in_bin

    left_edge = float(bin_edges[bin_index])
    right_edge = float(bin_edges[bin_index + 1])

    return float(left_edge + fraction_within_bin * (right_edge - left_edge))


def build_path_list_from_catalog(
    catalog_path: str,
    window_columns: Sequence[str],
    max_files: int,
) -> List[str]:
    """
    Read the catalog and pull all non-null TIFF paths from the requested window columns.
    Deduplicate while keeping order. Optionally truncate.
    """
    df = pd.read_csv(catalog_path)

    all_paths: List[str] = []
    for col in window_columns:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in {catalog_path}")
        col_paths = df[col].dropna().astype(str).tolist()
        all_paths.extend(col_paths)

    seen = set()
    unique_paths: List[str] = []
    for p in all_paths:
        if p not in seen:
            seen.add(p)
            unique_paths.append(p)

    if max_files and max_files > 0:
        unique_paths = unique_paths[:max_files]

    return unique_paths


def build_path_list_from_dir(
    tiff_dir: str,
    max_files: int,
) -> List[str]:
    """
    Get all .tif/.tiff files from a directory (absolute paths), optionally truncated.
    """
    patterns = [
        os.path.join(tiff_dir, "*.tif"),
        os.path.join(tiff_dir, "*.tiff"),
        os.path.join(tiff_dir, "*.TIF"),
        os.path.join(tiff_dir, "*.TIFF"),
    ]
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(pat))

    files = sorted(set(files))  # dedupe + sort

    if max_files and max_files > 0:
        files = files[:max_files]

    return files


def compute_global_band_stats(
    tiff_paths: Sequence[str],
    root_dir: str,
    histogram_bins: int,
) -> Tuple[pd.DataFrame, Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Compute global min/max, mean/std (via Welford), and build histograms for all bands
    across all provided TIFF paths. Returns:
      (stats_df, histograms_by_band, edges_by_band)
    """

    band_min: Dict[int, float] = defaultdict(lambda: np.inf)
    band_max: Dict[int, float] = defaultdict(lambda: -np.inf)
    welford_state: Dict[int, Tuple[int, float, float]] = defaultdict(lambda: (0, 0.0, 0.0))

    detected_band_count: int | None = None
    processed_file_count = 0
    missing_file_count = 0

    for rel_path in tqdm(tiff_paths, desc="Pass 1 (min/max + mean/std)", unit="file"):
        abs_path = resolve_path(rel_path, root_dir)

        if not os.path.exists(abs_path):
            missing_file_count += 1
            tqdm.write(f"[WARN] Missing file: {abs_path}")
            continue

        with rasterio.open(abs_path) as ds:
            band_indices = list(range(1, ds.count + 1))

            if detected_band_count is None:
                detected_band_count = ds.count
            elif ds.count != detected_band_count:
                raise RuntimeError(
                    f"Inconsistent band count: {abs_path} has {ds.count}, expected {detected_band_count}"
                )

            for band_index, vals in iterate_valid_blocks(ds, band_indices):
                band_min[band_index] = min(band_min[band_index], float(np.min(vals)))
                band_max[band_index] = max(band_max[band_index], float(np.max(vals)))

                welford_state[band_index] = welford_merge_update(
                    vals.astype(np.float64, copy=False),
                    welford_state[band_index],
                )

        processed_file_count += 1

    if processed_file_count == 0:
        raise RuntimeError("No files were actually processed. Check paths/root_dir.")

    print(
        f"\nPass 1 complete:"
        f"\n  processed files: {processed_file_count}"
        f"\n  missing files:   {missing_file_count}"
        f"\n  bands detected:  {detected_band_count}"
    )

    # ---------- PASS 2: histograms ----------
    histograms_by_band: Dict[int, np.ndarray] = {}
    edges_by_band: Dict[int, np.ndarray] = {}

    for b in range(1, detected_band_count + 1):
        vmin = band_min[b]
        vmax = band_max[b]

        if not np.isfinite(vmin) or not np.isfinite(vmax):
            vmin, vmax = 0.0, 1.0
        if math.isclose(vmin, vmax):
            vmax = vmin + 1e-9

        bin_edges = np.linspace(vmin, vmax, histogram_bins + 1, dtype=np.float64)
        histograms_by_band[b] = np.zeros(histogram_bins, dtype=np.int64)
        edges_by_band[b] = bin_edges

    for rel_path in tqdm(tiff_paths, desc="Pass 2 (global histograms)", unit="file"):
        abs_path = resolve_path(rel_path, root_dir)
        if not os.path.exists(abs_path):
            continue

        with rasterio.open(abs_path) as ds:
            band_indices = list(range(1, ds.count + 1))

            for band_index, vals in iterate_valid_blocks(ds, band_indices):
                bin_edges = edges_by_band[band_index]
                left_edge, right_edge = bin_edges[0], bin_edges[-1]

                vals = vals[(vals >= left_edge) & (vals <= right_edge)]
                if vals.size == 0:
                    continue

                block_hist, _ = np.histogram(vals, bins=bin_edges)
                histograms_by_band[band_index] += block_hist

    print("Pass 2 complete.")

    requested_percentiles = [1, 2, 98, 99]
    per_band_records: List[dict] = []

    for b in range(1, detected_band_count + 1):
        n_obs, mean_val, M2_val = welford_state[b]
        std_val = math.sqrt(M2_val / (n_obs - 1)) if n_obs > 1 else float("nan")

        pct_vals = {
            f"p{q}": percentile_from_histogram(histograms_by_band[b], edges_by_band[b], q)
            for q in requested_percentiles
        }

        per_band_records.append(
            {
                "band": b,
                "count": int(n_obs),
                "mean": mean_val,
                "std": std_val,
                "min": float(band_min[b]),
                "max": float(band_max[b]),
                **pct_vals,
            }
        )

    stats_df = (
        pd.DataFrame(per_band_records)
        .sort_values("band")
        .reset_index(drop=True)
    )

    return stats_df, histograms_by_band, edges_by_band


def save_stats(
    stats_df: pd.DataFrame,
    out_prefix: str,
) -> None:
    out_csv = f"{out_prefix}.csv"
    out_json = f"{out_prefix}.json"

    stats_df.to_csv(out_csv, index=False)
    with open(out_json, "w") as f:
        json.dump(stats_df.to_dict(orient="records"), f, indent=2)

    print(f"\nWrote: {out_csv}")
    print(f"Wrote: {out_json}")


# -------------------------------------------------
# CLI
# -------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute global band stats over TIFFs from a catalog OR a directory."
    )

    parser.add_argument(
        "--catalog",
        help="Path to the catalog CSV (if using catalog mode).",
    )

    parser.add_argument(
        "--windows",
        nargs="+",
        help="One or more column names to include from the catalog (e.g. window_a window_b).",
    )

    parser.add_argument(
        "--tiff-dir",
        help="Directory containing TIFF files. If provided, catalog/windows are ignored.",
    )

    parser.add_argument(
        "--root-dir",
        default="/home/airg/data/labels/cropland/",
        help="Root directory to prepend to relative TIFF paths (catalog mode only).",
    )

    parser.add_argument(
        "--out-prefix",
        required=True,
        help="Prefix for output files. We'll write <prefix>.csv and <prefix>.json",
    )

    parser.add_argument(
        "--bins",
        type=int,
        default=4096,
        help="Number of histogram bins for percentile estimation (4096 is good for 12-bit data).",
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Limit number of TIFFs for a quick test. 0 = use all.",
    )

    return parser.parse_args()


# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == "__main__":
    args = parse_args()

    # Decide where TIFF paths come from
    if args.tiff_dir:
        # Folder mode (your S2 window_b chips in a directory)
        tiff_paths = build_path_list_from_dir(
            tiff_dir=args.tiff_dir,
            max_files=args.max_files,
        )
        root_dir = ""  # paths are already absolute
    else:
        # Catalog mode (original behavior)
        if not args.catalog or not args.windows:
            raise SystemExit("You must provide either --tiff-dir OR (--catalog and --windows).")

        tiff_paths = build_path_list_from_catalog(
            catalog_path=args.catalog,
            window_columns=args.windows,
            max_files=args.max_files,
        )
        root_dir = args.root_dir

    # 2. Compute stats
    stats_df, _, _ = compute_global_band_stats(
        tiff_paths=tiff_paths,
        root_dir=root_dir,
        histogram_bins=args.bins,
    )

    # 3. Save
    save_stats(stats_df, args.out_prefix)
