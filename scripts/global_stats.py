#!/usr/bin/env python3
"""
Unified global stats calculator.
Supports:
 - CSV path columns (e.g., FTW window_a, window_b)
 - Raw directories (e.g., S2 MA windowA/windowB folders)

Usage example:

python3 global_stats.py \
    --csv /path/to/ftw-catalog.csv window_a window_b \
    --dir /path/to/s2_chips/windowA_nomask \
    --dir /path/to/mappingafrica-s2-256/window_b \
    --root /home/airg/data/labels/cropland \
    --out out_stats.csv
"""

from __future__ import annotations
import os, math, json, argparse
from collections import defaultdict
from typing import List, Dict, Tuple, Iterator
import numpy as np
import pandas as pd
import rasterio
from rasterio.io import DatasetReader
from tqdm import tqdm

# ---------------------------------------
# Path resolver
# ---------------------------------------
def resolve_path(p: str, root: str) -> str:
    return p if os.path.isabs(p) else os.path.join(root, p)

# ---------------------------------------
# Welford streaming update
# ---------------------------------------
def welford_update(arr: np.ndarray, state: Tuple[int, float, float]):
    n, mean, M2 = state
    if arr.size == 0:
        return state

    n2 = arr.size
    mean2 = float(arr.mean())
    M2_2 = float(((arr - mean2)**2).sum())

    delta = mean2 - mean
    n_total = n + n2
    mean_total = mean + delta * (n2 / n_total)
    M2_total = M2 + M2_2 + delta*delta * (n * n2 / n_total)

    return n_total, mean_total, M2_total

# ---------------------------------------
# read finite block values
# ---------------------------------------
def iter_valid_blocks(ds: DatasetReader, bands: List[int]) -> Iterator[Tuple[int, np.ndarray]]:
    read_kwargs = dict(masked=True, boundless=False)
    for _, window in ds.block_windows(bands[0]):
        block = ds.read(bands, window=window, **read_kwargs)
        for i, b in enumerate(bands):
            vals = block[i]
            vals = vals.compressed() if np.ma.isMaskedArray(vals) else vals.ravel()
            vals = vals[np.isfinite(vals)]
            if vals.size:
                yield b, vals

# ---------------------------------------
# percentile from histogram
# ---------------------------------------
def percentile_from_hist(hist, edges, q):
    total = hist.sum()
    if total == 0:
        return float("nan")

    target = q/100 * total
    cum = np.cumsum(hist)
    idx = int(np.searchsorted(cum, target, "left"))
    idx = int(np.clip(idx, 0, len(hist)-1))

    before = 0 if idx == 0 else cum[idx-1]
    count = hist[idx]

    if count == 0:
        return float(edges[idx])

    frac = (target - before) / count
    return float(edges[idx] + frac * (edges[idx+1]-edges[idx]))

# ---------------------------------------
# MAIN PROCESSOR
# ---------------------------------------
def compute_global_stats(paths: List[str],
                         root_dir: str,
                         out_csv: str,
                         out_json: str,
                         bins: int = 4096):

    # PASS 1 ---------------------------------------------------------
    band_min = defaultdict(lambda: np.inf)
    band_max = defaultdict(lambda: -np.inf)
    welford_state = defaultdict(lambda: (0, 0.0, 0.0))

    n_bands = None

    for p in tqdm(paths, desc="Pass 1 (min/max + mean/std)", unit="file"):
        tif = resolve_path(p, root_dir)
        if not os.path.exists(tif):
            tqdm.write(f"[WARN] Missing: {tif}")
            continue

        with rasterio.open(tif) as ds:
            bands = list(range(1, ds.count+1))
            if n_bands is None:
                n_bands = ds.count
            elif ds.count != n_bands:
                raise RuntimeError(f"Inconsistent band count: {tif}")

            for b, vals in iter_valid_blocks(ds, bands):
                band_min[b] = min(band_min[b], float(vals.min()))
                band_max[b] = max(band_max[b], float(vals.max()))
                welford_state[b] = welford_update(vals.astype(np.float64), welford_state[b])

    # PASS 2 ---------------------------------------------------------
    hist = {b: np.zeros(bins, np.int64) for b in range(1, n_bands+1)}
    edges_map = {}

    for b in range(1, n_bands+1):
        vmin, vmax = band_min[b], band_max[b]
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            vmin, vmax = 0, 1
        if math.isclose(vmin, vmax):
            vmax = vmin + 1e-9
        edges_map[b] = np.linspace(vmin, vmax, bins+1)

    for p in tqdm(paths, desc="Pass 2 (histograms)", unit="file"):
        tif = resolve_path(p, root_dir)
        if not os.path.exists(tif):
            continue
        with rasterio.open(tif) as ds:
            for b, vals in iter_valid_blocks(ds, list(range(1, ds.count+1))):
                vals = vals[(vals >= edges_map[b][0]) & (vals <= edges_map[b][-1])]
                if vals.size == 0:
                    continue
                h, _ = np.histogram(vals, bins=edges_map[b])
                hist[b] += h

    # FINAL OUTPUT ---------------------------------------------------
    records = []
    for b in range(1, n_bands+1):
        n, mean, M2 = welford_state[b]
        std = math.sqrt(M2/(n-1)) if n>1 else float("nan")
        rec = {
            "band": b,
            "count": int(n),
            "mean": mean,
            "std": std,
            "min": float(band_min[b]),
            "max": float(band_max[b]),
            "p1": percentile_from_hist(hist[b], edges_map[b], 1),
            "p2": percentile_from_hist(hist[b], edges_map[b], 2),
            "p98": percentile_from_hist(hist[b], edges_map[b], 98),
            "p99": percentile_from_hist(hist[b], edges_map[b], 99),
        }
        records.append(rec)

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    with open(out_json, "w") as f:
        json.dump(records, f, indent=2)

    print(f"\nWrote:\n - {out_csv}\n - {out_json}")
