"""
Image normalization utilities for FTW Mapping Africa.
"""

import numpy as np


def normalize_image(
    img,
    strategy="min_max",
    procedure="lab",
    global_stats=None,
    clip_val=0,
    nodata=[0, 65535],
    epsilon=1e-6,
):
    """
    Normalize the input image pixels to a user-defined range based on the
    minimum and maximum statistics of each band and optional clip value.

    Args:
        img (np.ndarray): Stacked image bands with a dimension of (C, H, W).
        strategy (str): Strategy for normalization. Either 'min_max' or
            'z_value'.
        procedure (str): Procedure to calculate the statistics used in
            normalization. Options:
                - 'lab': local tile over all bands.
                - 'gab': global over all bands.
                - 'lpb': local tile per band.
                - 'gpb': global per band.
        global_stats (dict, optional): Dictionary containing the 'min',
            'max', 'mean', and 'std' arrays for each band. If not provided,
            these values will be calculated from the data.
        clip_val (float, optional): Defines how much of the distribution
            tails to be cut off. Default is 0.
        nodata (list, optional): Values reserved to show nodata.
        epsilon (float, optional): Small constant added to the denominator
            to prevent division by zero.

    Returns:
        np.ndarray: Normalized image stack of size (C, H, W).

    Raises:
        ValueError: If normalization strategy or statistics procedure is
            not recognized, or if global statistics are required but not
            provided.
    """
    # print(f"normalize_image called with:")
    # print(f"  strategy: {strategy}")
    # print(f"  procedure: {procedure}")
    # print(f"  global_stats: {global_stats}")
    # print(f"  input shape: {img.shape}")
    # print(f"  input range: [{img.min():.2f}, {img.max():.2f}]")
    # print(f"  nodata values: {nodata}")

    if strategy not in ["min_max", "z_value"]:
        raise ValueError(
            "Normalization strategy is not recognized."
        )

    if procedure not in ["gpb", "lpb", "gab", "lab"]:
        raise ValueError(
            "Statistics calculation strategy is not recognized."
        )

    if procedure in ["gpb", "gab"] and global_stats is None:
        raise ValueError(
            "Global statistics must be provided for global normalization."
        )

    # Create a mask for nodata values and replace them with nan for computation.
    # Also create a copy for normalization
    img_tmp = np.where((img < 0) | np.isin(img, nodata), np.nan, img.astype(np.float64))
    normal_img = img.astype(np.float32)
    
    # Create valid pixel mask for setting nodata pixels to 0 at the end
    valid_mask = ~np.isnan(img_tmp)
    valid_pixels = np.sum(valid_mask)
    total_pixels = img.size
    valid_pct = (valid_pixels / total_pixels) * 100
    print(f"  Valid pixels: {valid_pixels}/{total_pixels} ({valid_pct:.1f}%)")

    if strategy == "min_max":
        if clip_val > 0:
            lower_percentiles = np.nanpercentile(
                img_tmp, clip_val, axis=(1, 2)
            )
            upper_percentiles = np.nanpercentile(
                img_tmp, 100 - clip_val, axis=(1, 2)
            )
            for b in range(img.shape[0]):
                # Only clip valid pixels
                normal_img[b] = np.where(
                    valid_mask[b],
                    np.clip(img[b], lower_percentiles[b], upper_percentiles[b]),
                    img[b]
                )

        if procedure == "gpb":
            gpb_mins = np.array(global_stats['min'])
            gpb_maxs = np.array(global_stats['max'])
            diff = gpb_maxs - gpb_mins
            diff[diff == 0] = epsilon
            # Only normalize valid pixels
            for b in range(img.shape[0]):
                normal_img[b] = np.where(
                    valid_mask[b],
                    np.clip((img[b] - gpb_mins[b]) / diff[b], 0, 1),
                    0
                )

        elif procedure == "gab":
            gab_min = np.mean(global_stats['min'])
            gab_max = np.mean(global_stats['max'])
            if gab_max == gab_min:
                gab_max += epsilon
            # Only normalize valid pixels
            normal_img = np.where(
                valid_mask,
                np.clip((img - gab_min) / (gab_max - gab_min), 0, 1),
                0
            )

        elif procedure == "lab":
            lab_min = np.nanmin(img_tmp)
            lab_max = np.nanmax(img_tmp)
            if lab_max == lab_min:
                lab_max += epsilon
            # Only normalize valid pixels
            normal_img = np.where(
                valid_mask,
                np.clip((img - lab_min) / (lab_max - lab_min), 0, 1),
                0
            )

        else:  # procedure == "lpb"
            lpb_mins = np.nanmin(img_tmp, axis=(1, 2))
            lpb_maxs = np.nanmax(img_tmp, axis=(1, 2))
            diff = lpb_maxs - lpb_mins
            diff[diff == 0] = epsilon
            # Only normalize valid pixels
            for b in range(img.shape[0]):
                normal_img[b] = np.where(
                    valid_mask[b],
                    np.clip((img[b] - lpb_mins[b]) / diff[b], 0, 1),
                    0
                )

    elif strategy == "z_value":
        if procedure == "gpb":
            gpb_means = np.array(global_stats['mean'])
            gpb_stds = np.array(global_stats['std'])
            gpb_stds[gpb_stds == 0] = epsilon
            # Only normalize valid pixels
            for b in range(img.shape[0]):
                normal_img[b] = np.where(
                    valid_mask[b],
                    (img[b] - gpb_means[b]) / gpb_stds[b],
                    0
                )

        elif procedure == "gab":
            gab_mean = np.mean(global_stats['mean'])
            gab_std = np.sqrt(
                np.sum(
                    (global_stats['std'] ** 2) * img.shape[1] * img.shape[2]
                ) / (img.shape[1] * img.shape[2] * len(global_stats['std']))
            )
            if gab_std == 0:
                gab_std += epsilon
            # Only normalize valid pixels
            normal_img = np.where(
                valid_mask,
                (img - gab_mean) / gab_std,
                0
            )

        elif procedure == "lpb":
            img_means = np.nanmean(img_tmp, axis=(1, 2))
            img_stds = np.nanstd(img_tmp, axis=(1, 2))
            img_stds[img_stds == 0] = epsilon
            # Only normalize valid pixels
            for b in range(img.shape[0]):
                normal_img[b] = np.where(
                    valid_mask[b],
                    (img[b] - img_means[b]) / img_stds[b],
                    0
                )

        elif procedure == "lab":
            img_mean = np.nanmean(img_tmp)
            img_std = np.nanstd(img_tmp)
            if img_std == 0:
                img_std += epsilon
            # Only normalize valid pixels
            normal_img = np.where(
                valid_mask,
                (img - img_mean) / img_std,
                0
            )

    # Ensure nodata pixels are set to 0 in the final output
    normal_img = normal_img.astype(np.float32)
    
    # print(f"  Set {np.sum(~valid_mask)} nodata pixels to 0.0")
    # print(f"  output range: [{normal_img.min():.2f}, {normal_img.max():.2f}]")
    # print(f"Normalized image range: [{normal_img.min():.2f}, {normal_img.max():.2f}]")
    return normal_img