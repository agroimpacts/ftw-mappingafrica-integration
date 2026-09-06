#!/usr/bin/env python3
"""
global_stats_ftw_planet_combos.py

Compute global band statistics (min, max, mean, std, p1, p2, p98, p99)
for the FTW-Planet dataset organized as:

  /home/airg/data/labels/cropland/ftw-planet/<country>/pl_images/window_a/*.tif
  /home/airg/data/labels/cropland/ftw-planet/<country>/pl_images/window_b/*.tif

Outputs (each as CSV + JSON):
  - FTW Planet window_a only
  - FTW Planet window_b only
  - FTW Planet window_a + window_b
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
    """Kept for compatibility; for this script we pass absolute paths anyway."""
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


# ----------------- FTW Planet path discovery -----------------
def collect_ftw_planet_window_dirs(ftw_root: str) -> Tuple[List[str], List[str]]:
    """
    Find all window_a and window_b directories under:
      <ftw_root>/<country>/pl_images/window_a
      <ftw_root>/<country>/pl_images/window_b
    """
    ftw_root = forcing_slash(ftw_root)
    if not os.path.isdir(ftw_root):
        raise FileNotFoundError(f"FTW root not found or not a directory: {ftw_root}")

    win_a_dirs: List[str] = []
    win_b_dirs: List[str] = []

    # countries are typically immediate children, but we’ll search robustly
    for root, dirs, _ in os.walk(ftw_root):
        base = os.path.basename(root)
        if base == "window_a":
            win_a_dirs.append(root)
        elif base == "window_b":
            win_b_dirs.append(root)

    # keep only those that contain ".../pl_images/window_a|window_b"
    def is_pl_images_window(d: str) -> bool:
        parts = d.replace("\\", "/").split("/")
        return len(parts) >= 2 and parts[-2] == "pl_images" and parts[-1] in ("window_a", "window_b")

    win_a_dirs = [d for d in win_a_dirs if is_pl_images_window(d)]
    win_b_dirs = [d for d in win_b_dirs if is_pl_images_window(d)]

    win_a_dirs = sorted(set(win_a_dirs))
    win_b_dirs = sorted(set(win_b_dirs))

    return win_a_dirs, win_b_dirs


def forcing_slash(p: str) -> str:
    return p.rstrip("/")


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
        raise RuntimeError("No files processed. Check ftw_root and folder structure.")
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
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    stats_df.to_csv(out_csv, index=False)
    with open(out_json, "w") as f:
        json.dump(stats_df.to_dict(orient="records"), f, indent=2)
    print(f"\nWrote: {out_csv}\nWrote: {out_json}")


def run_one(name: str, paths: List[str], root_dir: str, bins: int, out_prefix: str, max_files: int):
    # dedupe + optional truncate
    seen = set()
    unique = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    if max_files and max_files > 0:
        unique = unique[:max_files]

    print(f"\n=== RUN: {name} ===")
    print(f"TIFF count: {len(unique)}")

    if not unique:
        print("[WARN] No TIFFs found for this run; skipping.")
        return

    stats_df = compute_global_stats(unique, root_dir=root_dir, histogram_bins=bins)
    save_stats(stats_df, out_prefix)


# ----------------- CLI -----------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute global band stats for FTW-Planet window_a/window_b TIFFs under <ftw_root>."
    )
    p.add_argument(
        "--ftw-root",
        type=str,
        default="/home/airg/data/labels/cropland/ftw-planet/",
        help="FTW-Planet root folder containing <country>/pl_images/window_a and window_b.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="/home/airg/gessuman/projects/ftw-mappingafrica-integration/data/global_band_stats",
        help="Output directory for stats files.",
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
        help="Limit number of TIFFs per run after dedupe; 0 = all.",
    )
    p.add_argument(
        "--run-set",
        type=str,
        default="all",
        choices=["a", "b", "ab", "all"],
        help="Which run(s): a=window_a only, b=window_b only, ab=combined, all=all three.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    ftw_root = forcing_slash(args.ftw_root)
    out_dir = forcing_slash(args.out_dir)

    win_a_dirs, win_b_dirs = collect_ftw_planet_window_dirs(ftw_root)
    print(f"[discover] window_a dirs: {len(win_a_dirs)}")
    print(f"[discover] window_b dirs: {len(win_b_dirs)}")

    # collect TIFFs
    win_a_paths = build_paths_from_dirs(win_a_dirs, max_files=0, recursive=True)
    win_b_paths = build_paths_from_dirs(win_b_dirs, max_files=0, recursive=True)

    # NOTE: these are absolute paths, so root_dir doesn't matter, but kept for API consistency
    root_dir = "/"

    runs = []
    if args.run_set in ("a", "all"):
        runs.append(("FTW_PLANET_window_a_only", win_a_paths, f"{out_dir}/global_band_stats_ftw_planet_wina"))
    if args.run_set in ("b", "all"):
        runs.append(("FTW_PLANET_window_b_only", win_b_paths, f"{out_dir}/global_band_stats_ftw_planet_winb"))
    if args.run_set in ("ab", "all"):
        runs.append(("FTW_PLANET_window_a_plus_b", win_a_paths + win_b_paths, f"{out_dir}/global_band_stats_ftw_planet_winab"))

    for name, paths, prefix in runs:
        run_one(
            name=name,
            paths=paths,
            root_dir=root_dir,
            bins=args.bins,
            out_prefix=prefix,
            max_files=args.max_files,
        )

    print("\nAll requested FTW-Planet runs complete.")


if __name__ == "__main__":
    main()
