#!/usr/bin/env python3
"""
Compute global band statistics (min, max, mean, std, p1, p2, p98, p99)
for raster TIFFs coming from:
  - one catalog CSV (one or more window_* columns), and/or
  - one or more directories of .tif/.tiff files.

Example: FTW window_a + window_b (from catalog) + S2 MappingAfrica window_b (from directory):

  python global_stats_ftw_ma.py \
    --catalog /home/airg/gessuman/projects/ftw-mappingafrica-integration/data/ftw-catalog.csv \
    --windows window_a window_b \
    --dir /home/airg/data/labels/cropland/mappingafrica-s2-256/window_b \
    --root-dir /home/airg/data/labels/cropland/ \
    --out-prefix /home/airg/gessuman/projects/ftw-mappingafrica-integration/data/global_band_stats/global_band_stats_ftw_winab_ma_s2_winb
"""

from __future__ import annotations
import argparse
import json
import math
import os
from collections import defaultdict
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.io import DatasetReader
from tqdm import tqdm


# ----------------- Basic helpers -----------------
def resolve_path(path_str: str, root_dir: str) -> str:
    """If path_str is relative, prepend root_dir, else return as-is."""
    return path_str if os.path.isabs(path_str) else os.path.join(root_dir, path_str)


def welford_merge_update(
    sample_values: np.ndarray,
    running_state: Tuple[int, float, float],
) -> Tuple[int, float, float]:
    """Merge a chunk of values into Welford (n, mean, M2) state."""
    n_running, mean_running, M2_running = running_state
    if sample_values.size == 0:
        return running_state

    n_chunk = int(sample_values.size)
    mean_chunk = float(np.mean(sample_values))
    M2_chunk = float(np.sum((sample_values - mean_chunk) ** 2))

    delta = mean_chunk - mean_running
    n_total = n_running + n_chunk
    if n_total == 0:
        return running_state

    mean_total = mean_running + delta * (n_chunk / n_total)
    M2_total = M2_running + M2_chunk + (delta**2) * (n_running * n_chunk / n_total)
    return n_total, mean_total, M2_total


def iterate_valid_blocks(
    dataset: DatasetReader,
    band_indices: List[int],
) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (band_index, 1D finite pixel values) for each internal block window."""
    read_kwargs = dict(masked=True, boundless=False)
    for _, window in dataset.block_windows(band_indices[0]):
        block = dataset.read(band_indices, window=window, **read_kwargs)  # (B,h,w)
        for i, band_index in enumerate(band_indices):
            arr = block[i]
            vals = arr.compressed() if np.ma.isMaskedArray(arr) else arr.ravel()
            vals = vals[np.isfinite(vals)]
            if vals.size:
                yield band_index, vals


def percentile_from_histogram(
    histogram_counts: np.ndarray,
    bin_edges: np.ndarray,
    percentile_q: float,
) -> float:
    """Estimate percentile_q (0–100) from histogram via linear interpolation."""
    total = np.sum(histogram_counts)
    if total == 0:
        return float("nan")

    target = (percentile_q / 100.0) * total
    c = np.cumsum(histogram_counts)

    idx = int(np.searchsorted(c, target, side="left"))
    idx = int(np.clip(idx, 0, len(histogram_counts) - 1))

    before = 0 if idx == 0 else c[idx - 1]
    in_bin = int(histogram_counts[idx])
    if in_bin == 0:
        return float(bin_edges[idx])

    frac = (target - before) / in_bin
    left = float(bin_edges[idx])
    right = float(bin_edges[idx + 1])
    return float(left + frac * (right - left))


# ----------------- Path building -----------------
def build_paths_from_catalog(
    catalog_path: str,
    window_columns: Sequence[str],
    max_files: int,
) -> List[str]:
    """Collect non-null paths from specified window_* columns in a catalog."""
    if catalog_path is None:
        return []

    df = pd.read_csv(catalog_path)
    all_paths: List[str] = []

    for col in window_columns:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in {catalog_path}")
        col_paths = df[col].dropna().astype(str).tolist()
        all_paths.extend(col_paths)

    # Deduplicate while preserving order
    seen = set()
    unique: List[str] = []
    for p in all_paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    if max_files and max_files > 0:
        unique = unique[:max_files]

    print(f"[catalog] Selected {len(unique)} unique paths from columns {window_columns}")
    return unique


def build_paths_from_dirs(
    dirs: Sequence[str],
    max_files: int,
    recursive: bool = True,
) -> List[str]:
    """Collect all .tif/.tiff paths from the given directories."""
    collected: List[str] = []
    exts = {".tif", ".tiff", ".TIF", ".TIFF"}

    for d in dirs:
        if not d:
            continue
        if not os.path.isdir(d):
            print(f"[WARN] Directory does not exist or is not a dir: {d}")
            continue

        if recursive:
            walker = os.walk(d)
        else:
            # non-recursive: just this dir
            def walker_once(base_dir):
                yield base_dir, [], os.listdir(base_dir)
            walker = walker_once(d)

        for root, _, files in walker:
            for fname in files:
                _, ext = os.path.splitext(fname)
                if ext in exts:
                    collected.append(os.path.join(root, fname))

    # Deduplicate while preserving order
    seen = set()
    unique: List[str] = []
    for p in collected:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    if max_files and max_files > 0:
        unique = unique[:max_files]

    print(f"[dirs] Selected {len(unique)} unique TIFFs from dirs: {dirs}")
    return unique


# ----------------- Global stats core -----------------
def compute_global_stats(
    tiff_paths: Sequence[str],
    root_dir: str,
    histogram_bins: int,
) -> pd.DataFrame:
    """Compute global min/max, mean/std, and percentiles over all TIFFs."""
    band_min: Dict[int, float] = defaultdict(lambda: np.inf)
    band_max: Dict[int, float] = defaultdict(lambda: -np.inf)
    welford_state: Dict[int, Tuple[int, float, float]] = defaultdict(lambda: (0, 0.0, 0.0))

    detected_bands: int | None = None
    processed = missing = 0

    # ---- Pass 1: min/max + mean/std ----
    for rel in tqdm(tiff_paths, desc="Pass 1 (min/max + mean/std)", unit="file"):
        tif = resolve_path(rel, root_dir)
        if not os.path.exists(tif):
            missing += 1
            tqdm.write(f"[WARN] Missing file: {tif}")
            continue

        with rasterio.open(tif) as ds:
            bands = list(range(1, ds.count + 1))
            if detected_bands is None:
                detected_bands = ds.count
            elif ds.count != detected_bands:
                raise RuntimeError(f"Inconsistent band count: {tif} has {ds.count}, expected {detected_bands}")

            for b, vals in iterate_valid_blocks(ds, bands):
                band_min[b] = min(band_min[b], float(np.min(vals)))
                band_max[b] = max(band_max[b], float(np.max(vals)))
                welford_state[b] = welford_merge_update(vals.astype(np.float64, copy=False), welford_state[b])

        processed += 1

    if processed == 0:
        raise RuntimeError("No files processed. Check catalog/dirs/root_dir.")
    print(f"\nPass 1 complete: processed={processed}, missing={missing}, bands={detected_bands}")

    # ---- Pass 2: histograms ----
    hist_by_band: Dict[int, np.ndarray] = {}
    edges_by_band: Dict[int, np.ndarray] = {}

    for b in range(1, detected_bands + 1):
        vmin, vmax = band_min[b], band_max[b]
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            vmin, vmax = 0.0, 1.0
        if math.isclose(vmin, vmax):
            vmax = vmin + 1e-9
        edges = np.linspace(vmin, vmax, histogram_bins + 1, dtype=np.float64)
        edges_by_band[b] = edges
        hist_by_band[b] = np.zeros(histogram_bins, dtype=np.int64)

    for rel in tqdm(tiff_paths, desc="Pass 2 (global histograms)", unit="file"):
        tif = resolve_path(rel, root_dir)
        if not os.path.exists(tif):
            continue
        with rasterio.open(tif) as ds:
            bands = list(range(1, ds.count + 1))
            for b, vals in iterate_valid_blocks(ds, bands):
                edges = edges_by_band[b]
                vals = vals[(vals >= edges[0]) & (vals <= edges[-1])]
                if vals.size == 0:
                    continue
                h, _ = np.histogram(vals, bins=edges)
                hist_by_band[b] += h

    print("Pass 2 complete.")

    # ---- Finalize per-band stats ----
    records = []
    qs = [1, 2, 98, 99]
    for b in range(1, detected_bands + 1):
        n, mean, M2 = welford_state[b]
        std = math.sqrt(M2 / (n - 1)) if n > 1 else float("nan")
        pct = {f"p{q}": percentile_from_histogram(hist_by_band[b], edges_by_band[b], q) for q in qs}
        records.append({
            "band": b,
            "count": int(n),
            "mean": mean,
            "std": std,
            "min": float(band_min[b]),
            "max": float(band_max[b]),
            **pct,
        })

    return pd.DataFrame(records).sort_values("band").reset_index(drop=True)


def save_stats(stats_df: pd.DataFrame, out_prefix: str) -> None:
    out_csv = f"{out_prefix}.csv"
    out_json = f"{out_prefix}.json"
    stats_df.to_csv(out_csv, index=False)
    with open(out_json, "w") as f:
        json.dump(stats_df.to_dict(orient="records"), f, indent=2)
    print(f"\nWrote: {out_csv}\nWrote: {out_json}")


# ----------------- CLI -----------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute global band stats for TIFFs from a catalog (window columns) and/or directories."
    )
    p.add_argument(
        "--catalog",
        help="Path to catalog CSV (optional).",
    )
    p.add_argument(
        "--windows",
        nargs="+",
        default=[],
        help="Window column names in catalog to use (e.g., window_a window_b).",
    )
    p.add_argument(
        "--dir",
        action="append",
        default=[],
        help="Directory with S2/other TIFFs (can be used multiple times). All .tif/.tiff inside will be used.",
    )
    p.add_argument(
        "--root-dir",
        default="/home/airg/data/labels/cropland/",
        help="Root dir for resolving relative paths (catalog). Absolute paths are left as-is.",
    )
    p.add_argument(
        "--out-prefix",
        required=True,
        help="Output prefix (we write <prefix>.csv and <prefix>.json).",
    )
    p.add_argument(
        "--bins",
        type=int,
        default=4096,
        help="Histogram bins for percentile estimation (4096 good for 12-bit).",
    )
    p.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Limit number of paths (after dedupe); 0 = all.",
    )
    return p.parse_args()


# ----------------- Main -----------------
if __name__ == "__main__":
    args = parse_args()

    # 1) From catalog windows (if provided)
    catalog_paths: List[str] = []
    if args.catalog and args.windows:
        catalog_paths = build_paths_from_catalog(
            catalog_path=args.catalog,
            window_columns=args.windows,
            max_files=args.max_files,  # limit applied here
        )

    # 2) From directories (if provided)
    dir_paths: List[str] = []
    if args.dir:
        # Note: dir max_files is applied *after* combining with catalog; so here we pass 0
        dir_paths = build_paths_from_dirs(
            dirs=args.dir,
            max_files=0,  # don't truncate yet
            recursive=True,
        )

    # 3) Combine and deduplicate both sources; final max_files cap
    combined = catalog_paths + dir_paths
    seen_all = set()
    unique_all: List[str] = []
    for p in combined:
        if p not in seen_all:
            seen_all.add(p)
            unique_all.append(p)

    if args.max_files and args.max_files > 0:
        unique_all = unique_all[:args.max_files]

    print(f"\n[total] Final number of unique TIFF paths: {len(unique_all)}")

    if not unique_all:
        raise RuntimeError("No TIFF paths found from catalog and/or directories.")

    # 4) Compute stats & save
    stats_df = compute_global_stats(
        tiff_paths=unique_all,
        root_dir=args.root_dir,
        histogram_bins=args.bins,
    )
    save_stats(stats_df, args.out_prefix)
