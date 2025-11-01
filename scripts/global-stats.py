# solution is from chatGPT 5
import numpy as np
import pandas as pd
import rioxarray as rxr
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# -------------------
# CONFIG
# -------------------
data_dir = Path("/path/to/data")   # <-- set this
bin_count = 20000                  # number of bins for histogram
max_workers = 8                    # parallel workers

# -------------------
# Welford’s update
# -------------------
def update_stats(arr, count, mean, M2):
    for x in arr:
        count += 1
        delta = x - mean
        mean += delta / count
        delta2 = x - mean
        M2 += delta * delta2
    return count, mean, M2

# -------------------
# Worker for pass 1 (min/max per band)
# -------------------
def worker_minmax(path):
    img = rxr.open_rasterio(path)
    nodata = getattr(img.rio, "nodata", None)
    arr = img.values  # shape (bands, y, x)
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    arr = arr.reshape(arr.shape[0], -1)

    band_mins = np.nanmin(arr, axis=1)
    band_maxs = np.nanmax(arr, axis=1)
    return band_mins, band_maxs

# -------------------
# Worker for pass 2 (hist + stats per band)
# -------------------
def worker_hist_stats(path, bin_edges):
    img = rxr.open_rasterio(path)
    nodata = getattr(img.rio, "nodata", None)
    arr = img.values  # shape (bands, y, x)
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    arr = arr.reshape(arr.shape[0], -1)

    n_bands = arr.shape[0]
    hist_list = []
    counts, means, M2s = [], [], []

    for b in range(n_bands):
        vals = arr[b, :]
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            hist_list.append(np.zeros(len(bin_edges)-1, dtype=np.int64))
            counts.append(0)
            means.append(0.0)
            M2s.append(0.0)
            continue

        # histogram
        hist, _ = np.histogram(vals, bins=bin_edges)
        hist_list.append(hist)

        # welford stats
        c, m, M2 = 0, 0.0, 0.0
        c, m, M2 = update_stats(vals, c, m, M2)
        counts.append(c)
        means.append(m)
        M2s.append(M2)

    return hist_list, counts, means, M2s

# -------------------
# Percentile from histogram
# -------------------
def percentile_from_hist(hist, bin_edges, q):
    cdf = np.cumsum(hist)
    cdf = cdf / cdf[-1]
    return np.interp(q/100, cdf, bin_edges[1:])

# -------------------
# Main computation
# -------------------
def compute_global_stats(ftw_cat, max_workers=8, bin_count=20000):
    paths = [data_dir / row["window_b"] for _, row in ftw_cat.iterrows()]

    # -------- Pass 1: global min/max per band --------
    mins, maxs = [], []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker_minmax, p) for p in paths]
        for f in tqdm(as_completed(futures), total=len(futures), 
                      desc="Pass 1 (min/max)"):
            bmins, bmaxs = f.result()
            mins.append(bmins)
            maxs.append(bmaxs)

    mins = np.vstack(mins)
    maxs = np.vstack(maxs)
    gmins = np.nanmin(mins, axis=0)
    gmaxs = np.nanmax(maxs, axis=0)

    # -------- Pass 2: hist + stats per band --------
    bin_edges_list = [
        np.linspace(gmins[b], gmaxs[b], bin_count + 1)
        for b in range(len(gmins))
    ]

    # Initialize accumulators
    n_bands = len(gmins)
    global_hists = [np.zeros(bin_count, dtype=np.int64) for _ in range(n_bands)]
    global_counts = [0] * n_bands
    global_means = [0.0] * n_bands
    global_M2s = [0.0] * n_bands

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker_hist_stats, p, bin_edges_list[b])
                   for p in paths for b in [0]]  # pass same bin_edges_list each call
        # We actually need to broadcast all bin_edges per worker
        futures = [ex.submit(worker_hist_stats, p, bin_edges_list) 
                   for p in paths]

        for f in tqdm(as_completed(futures), total=len(futures), 
                      desc="Pass 2 (hist/stats)"):
            hist_list, counts, means, M2s = f.result()
            for b in range(n_bands):
                global_hists[b] += hist_list[b]
                # combine Welford accumulators
                # careful: recomputing across workers isn't trivial → 
                # recompute by merging
                # Instead, we just sum raw counts, means, M2s
                # Merging Welford requires careful formula, easier is to 
                # re-update:
                c, m, M2 = global_counts[b], global_means[b], global_M2s[b]
                # Expand each worker’s values into update
                # (less efficient but exact)
                if counts[b] > 0:
                    # recompute via update_stats on aggregated vals?
                    # Instead: combine using Chan’s formula
                    delta = means[b] - m
                    tot_count = c + counts[b]
                    if tot_count > 0:
                        new_mean = (c * m + counts[b] * means[b]) / tot_count
                        new_M2 = M2 + M2s[b] + delta**2 \
                            * c * counts[b] / tot_count
                        m, M2, c = new_mean, new_M2, tot_count
                global_counts[b], global_means[b], global_M2s[b] = c, m, M2

    # -------- Final aggregation --------
    results = []
    for b in range(n_bands):
        gmean = global_means[b]
        gvar = global_M2s[b] / (global_counts[b] - 1) \
            if global_counts[b] > 1 else np.nan
        gstd = np.sqrt(gvar)

        p1 = percentile_from_hist(global_hists[b], bin_edges_list[b], 1)
        p2 = percentile_from_hist(global_hists[b], bin_edges_list[b], 2)
        p98 = percentile_from_hist(global_hists[b], bin_edges_list[b], 98)
        p99 = percentile_from_hist(global_hists[b], bin_edges_list[b], 99)

        results.append({
            "band": f"band_{b+1}",
            "min": gmins[b],
            "max": gmaxs[b],
            "mean": gmean,
            "std": gstd,
            "p1": p1,
            "p2": p2,
            "p98": p98,
            "p99": p99,
            "count": global_counts[b]
        })

    return pd.DataFrame(results)

# -------------------
# Example usage
# -------------------
# assuming ftw_cat is your DataFrame with "window_b" column
global_stats_df = compute_global_stats(ftw_cat, max_workers=max_workers, 
                                       bin_count=bin_count)
print(global_stats_df)
