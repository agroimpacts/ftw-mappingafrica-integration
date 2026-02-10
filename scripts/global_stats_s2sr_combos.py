#!/usr/bin/env python3
"""
global_stats_s2sr_combos.py

Compute global band statistics (min, max, mean, std, p1, p2, p98, p99)
For raster TIFFs coming from:
  - FTW catalog CSV window columns (window_a/window_b) and
  - Directories of .tif/.tiff files. 

1) MA S2SR WINB only
2) MA S2SR WINA + WINB
3) FTW WINA+B + MA S2SR WINB only
4) FTW WINA+B + MA S2SR WINA+B

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

    print(f"[catalog] Selected {len(unique)} unique paths from columns {list(window_columns)}")
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

    print(f"[dirs] Selected {len(unique)} unique TIFFs from dirs: {list(dirs)}")
    return unique


def dedupe_preserve_order(paths: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


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
                raise RuntimeError(
                    f"Inconsistent band count: {tif} has {ds.count}, expected {detected_bands}"
                )

            for b, vals in iterate_valid_blocks(ds, bands):
                band_min[b] = min(band_min[b], float(np.min(vals)))
                band_max[b] = max(band_max[b], float(np.max(vals)))
                welford_state[b] = welford_merge_update(
                    vals.astype(np.float64, copy=False), welford_state[b]
                )

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
        pct = {
            f"p{q}": percentile_from_histogram(hist_by_band[b], edges_by_band[b], q)
            for q in qs
        }
        records.append(
            dict(
                band=b,
                count=int(n),
                mean=mean,
                std=std,
                min=float(band_min[b]),
                max=float(band_max[b]),
                **pct,
            )
        )

    return pd.DataFrame(records).sort_values("band").reset_index(drop=True)


def save_stats(stats_df: pd.DataFrame, out_prefix: str) -> None:
    out_csv = f"{out_prefix}.csv"
    out_json = f"{out_prefix}.json"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    stats_df.to_csv(out_csv, index=False)
    with open(out_json, "w") as f:
        json.dump(stats_df.to_dict(orient="records"), f, indent=2)

    print(f"\nWrote: {out_csv}\nWrote: {out_json}")


# ----------------- Combo runner -----------------
def run_one(name: str, paths: List[str], root_dir: str, bins: int, out_prefix: str, max_files: int = 0):
    paths = dedupe_preserve_order(paths)
    if max_files and max_files > 0:
        paths = paths[:max_files]
    print(f"\n==============================")
    print(f"RUN: {name}")
    print(f"Unique TIFF paths: {len(paths)}")
    print(f"Out prefix: {out_prefix}")
    print(f"==============================\n")
    if not paths:
        raise RuntimeError(f"No TIFF paths found for run: {name}")
    stats_df = compute_global_stats(paths, root_dir=root_dir, histogram_bins=bins)
    save_stats(stats_df, out_prefix)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute global band stats for SR combos (MA + FTW).")

    # inputs
    p.add_argument("--ftw-catalog", default="/home/airg/gessuman/projects/ftw-mappingafrica-integration/data/ftw-catalog.csv")
    p.add_argument("--ftw-windows", nargs="+", default=["window_a", "window_b"])

    p.add_argument("--ma-s2sr-wina-dir", default="/home/airg/gessuman/projects/ftw-mappingafrica-integration/data/s2_chips/windowA_parent128_nomask_rgbnir_bigger_extent")
    p.add_argument("--ma-s2sr-winb-dir", default="/home/airg/gessuman/projects/ftw-mappingafrica-integration/data/s2_chips/primary_parent128_sr2p5m_crop256")

    p.add_argument("--root-dir", default="/home/airg/data/labels/cropland/")

    p.add_argument("--out-dir", default="/home/airg/gessuman/projects/ftw-mappingafrica-integration/data/global_band_stats")
    p.add_argument("--bins", type=int, default=4096)
    p.add_argument("--max-files", type=int, default=0)

    p.add_argument(
        "--run-set",
        choices=["1", "2", "3", "4", "all"],
        default="all",
        help=(
            "Which combo(s) to run:\n"
            "1=MA S2SR WINB only\n"
            "2=MA S2SR WINA+B\n"
            "3=FTW WINA+B + MA S2SR WINB only\n"
            "4=FTW WINA+B + MA S2SR WINA+B\n"
            "all=run 1..4"
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Build common source lists
    ma_wina_paths = build_paths_from_dirs([args.ma_s2sr_wina_dir], max_files=0, recursive=True)
    ma_winb_paths = build_paths_from_dirs([args.ma_s2sr_winb_dir], max_files=0, recursive=True)

    ftw_paths = build_paths_from_catalog(
        catalog_path=args.ftw_catalog,
        window_columns=args.ftw_windows,
        max_files=0,
    )

    out_dir = args.out_dir.rstrip("/")

    runs = []
    if args.run_set in ("1", "all"):
        runs.append((
            "MA_S2SR_WINB_only",
            ma_winb_paths,
            f"{out_dir}/global_band_stats_ma_s2sr_winb_only"
        ))
    if args.run_set in ("2", "all"):
        runs.append((
            "MA_S2SR_WINA_plus_WINB",
            ma_wina_paths + ma_winb_paths,
            f"{out_dir}/global_band_stats_ma_s2sr_wina_plus_winb"
        ))
    if args.run_set in ("3", "all"):
        runs.append((
            "FTW_WINAplusB_plus_MA_S2SR_WINB_only",
            ftw_paths + ma_winb_paths,
            f"{out_dir}/global_band_stats_ftw_winab_plus_ma_s2sr_winb_only"
        ))
    if args.run_set in ("4", "all"):
        runs.append((
            "FTW_WINAplusB_plus_MA_S2SR_WINAplusB",
            ftw_paths + ma_wina_paths + ma_winb_paths,
            f"{out_dir}/global_band_stats_ftw_winab_plus_ma_s2sr_wina_plus_winb"
        ))

    for name, paths, prefix in runs:
        run_one(
            name=name,
            paths=paths,
            root_dir=args.root_dir,
            bins=args.bins,
            out_prefix=prefix,
            max_files=args.max_files
        )

    print("\nAll requested runs complete.")


if __name__ == "__main__":
    main()
