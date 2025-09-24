"""
Utility functions for image loading and normalization for FTW Mapping Africa.
"""

import numpy as np
import rasterio
import matplotlib.pyplot as plt
import torch

def load_image(
    path,
    apply_normalization=True,
    normal_strategy="min_max",
    stat_procedure="lab",
    global_stats=None,
    clip_val=0,
    nodata_val_ls=[0, 65535],
    epsilon=1e-6
):
    """
    Load a raster image from the given path and optionally apply
    normalization and nodata masking.

    Args:
        path (str or Path): Path to the raster file.
        apply_normalization (bool): Whether to normalize the image after
            loading.
        normal_strategy (str): Normalization strategy ('min_max' or
            'zscore').
        stat_procedure (str): Procedure for normalization statistics.
        global_stats (tuple, optional): Precomputed global stats for
            normalization.
        clip_val (float): Value to clip image data to after normalization.
        nodata_val_ls (list, optional): List of nodata values to mask out.
        epsilon (float, optional): Small constant added to the denominator
            to prevent division by zero.

    Returns:
        np.ndarray: Loaded and processed image array.
    """
    with rasterio.open(path) as src:
        img = src.read()

    img_nodata = src.nodata
    nodata_val_ls = list(set(nodata_val_ls + [img_nodata])) if nodata_val_ls \
        else [img_nodata]
    
    if apply_normalization:
        img = normalize_image(
            img,
            strategy=normal_strategy,
            procedure=stat_procedure,
            global_stats=global_stats,
            clip_val=clip_val,
            nodata=nodata_val_ls,
            epsilon=epsilon,
        )
    return img

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
    img_tmp = np.where((img < 0) | np.isin(img, nodata), np.nan, img)

    if strategy == "min_max":
        if clip_val > 0:
            lower_percentiles = np.nanpercentile(
                img_tmp, clip_val, axis=(1, 2)
            )
            upper_percentiles = np.nanpercentile(
                img_tmp, 100 - clip_val, axis=(1, 2)
            )
            for b in range(img.shape[0]):
                img[b] = np.clip(
                    img[b], lower_percentiles[b], upper_percentiles[b]
                )

        if procedure == "gpb":
            gpb_mins = np.array(global_stats['min'])
            gpb_maxs = np.array(global_stats['max'])
            diff = gpb_maxs - gpb_mins
            diff[diff == 0] = epsilon
            normal_img = (
                (img - gpb_mins[:, None, None]) / diff[:, None, None]
            )
            normal_img = np.clip(normal_img, 0, 1)

        elif procedure == "gab":
            gab_min = np.mean(global_stats['min'])
            gab_max = np.mean(global_stats['max'])
            if gab_max == gab_min:
                gab_max += epsilon
            normal_img = (img - gab_min) / (gab_max - gab_min)
            normal_img = np.clip(normal_img, 0, 1)

        elif procedure == "lab":
            lab_min = np.nanmin(img_tmp)
            lab_max = np.nanmax(img_tmp)
            if lab_max == lab_min:
                lab_max += epsilon
            normal_img = (img - lab_min) / (lab_max - lab_min)
            normal_img = np.clip(normal_img, 0, 1)

        else:  # procedure == "lpb"
            lpb_mins = np.nanmin(img_tmp, axis=(1, 2))
            lpb_maxs = np.nanmax(img_tmp, axis=(1, 2))
            diff = lpb_maxs - lpb_mins
            diff[diff == 0] = epsilon
            normal_img = (
                (img - lpb_mins[:, None, None]) / diff[:, None, None]
            )
            normal_img = np.clip(normal_img, 0, 1)

    elif strategy == "z_value":
        if procedure == "gpb":
            gpb_means = np.array(global_stats['mean'])
            gpb_stds = np.array(global_stats['std'])
            gpb_stds[gpb_stds == 0] = epsilon
            normal_img = (
                (img - gpb_means[:, None, None]) / gpb_stds[:, None, None]
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
            normal_img = (img - gab_mean) / gab_std

        elif procedure == "lpb":
            img_means = np.nanmean(img_tmp, axis=(1, 2))
            img_stds = np.nanstd(img_tmp, axis=(1, 2))
            img_stds[img_stds == 0] = epsilon
            normal_img = (
                (img - img_means[:, None, None]) / img_stds[:, None, None]
            )

        elif procedure == "lab":
            img_mean = np.nanmean(img_tmp)
            img_std = np.nanstd(img_tmp)
            if img_std == 0:
                img_std += epsilon
            normal_img = (img - img_mean) / img_std

    return normal_img

def show_augmented_variants(
    datamodule,
    dataset: str = "train",
    index: int = 0,
    n: int = 6,
    seed_start: int = 1234,
    figsize: tuple[int, int] | None = None,
    show: bool = True,
):
    """
    Display original image+mask and N augmented variants (image above, mask 
    below). The function uses datamodule.apply_train_aug(batch, seed=...) if 
    available to mimic how Lightning would apply augmentations.

    Args:
        datamodule: a FTWMapAfricaDataModule instance.
        dataset: one of "train", "validate", "test" to choose which dataset to 
            sample.
        index: sample index to visualize.
        n: number of augmented variants to display.
        seed_start: starting seed; seeds used are seed_start + i for i in 
            range(n).
        figsize: matplotlib figure size; default (cols*3, 6).
        show: if True call plt.show() and close the figure to avoid double 
            display.
    Returns:
        matplotlib.figure.Figure
    """

    # pick dataset
    if dataset == "train":
        ds = datamodule.train_dataset
    elif dataset in ("validate", "val"):
        ds = datamodule.val_dataset
    elif dataset == "test":
        ds = datamodule.test_dataset
    else:
        raise ValueError(f"unknown dataset: {dataset!r}")

    sample = ds[index]
    batch = {"image": sample["image"].unsqueeze(0), 
             "mask": sample["mask"].unsqueeze(0)}

    def _prep_display(img: np.ndarray):
        vmin, vmax = float(img.min()), float(img.max())
        if vmax <= 1.0 and vmin >= 0.0:
            return img
        return (img - vmin) / (vmax - vmin + 1e-8)

    cols = n + 1
    figsize = figsize or (cols * 3, 6)
    fig, axs = plt.subplots(2, cols, figsize=figsize)

    # Original in column 0
    orig_img = sample["image"].permute(1, 2, 0).numpy()
    orig_mask = sample["mask"].squeeze().numpy()
    axs[0, 0].imshow(_prep_display(orig_img))
    axs[0, 0].axis("off")
    axs[0, 0].set_title("original")
    axs[1, 0].imshow(orig_mask, cmap="gray", vmin=0, 
                     vmax=max(1, orig_mask.max()))
    axs[1, 0].axis("off")
    axs[1, 0].set_title("mask")

    # Augmented variants
    # apply_fn = getattr(datamodule, "apply_train_aug", None)
    if dataset == "train":
        apply_fn = getattr(datamodule, "apply_train_aug", None)
    elif dataset in ("validate", "val"):
        apply_fn = getattr(datamodule, "apply_val_aug", None)
    elif dataset == "test":
        apply_fn = getattr(datamodule, "apply_test_aug", None)
    else:
        apply_fn = None

    for i in range(n):
        seed = seed_start + i
        if apply_fn is None:
            # fallback: try to use datamodule.train_aug directly if present
            aug_batch = {"image": batch["image"].clone(), 
                         "mask": batch["mask"].clone()}
            # only fall back to `train_aug` for the train dataset
            if dataset == "train" and getattr(datamodule, "train_aug", None) \
                is not None:
                # try to set deterministic seed for reproducibility
                import random, numpy as _np, torch as _torch

                random.seed(seed)
                _np.random.seed(seed)
                try:
                    _torch.manual_seed(seed)
                except Exception:
                    pass
                aug_batch = datamodule.train_aug(aug_batch)
        else:
            aug_batch = apply_fn(batch, seed=seed)

        aug_img = aug_batch["image"][0].permute(1, 2, 0).numpy()
        aug_mask = aug_batch["mask"][0].squeeze().numpy()

        axs[0, i + 1].imshow(_prep_display(aug_img))
        axs[0, i + 1].axis("off")
        axs[0, i + 1].set_title(f"seed={seed}")

        axs[1, i + 1].imshow(aug_mask, cmap="gray", vmin=0, 
                             vmax=max(1, aug_mask.max()))
        axs[1, i + 1].axis("off")
        axs[1, i + 1].set_title("mask")

    plt.tight_layout()
    if show:
        plt.show()
        plt.close(fig)
    return fig

def plot_image_band_histograms(
    datamodule=None,
    dataset: str = "train",
    index: int = 0,
    batch: dict | None = None,
    apply_aug: bool = False,
    bins: int = 50,
    figsize: tuple[int, int] | None = (10, 5),
    alpha: float = 0.5,
    title: str | None = None,
    show: bool = True,
):
    """
    Plot per-band histograms for an image. Use `batch` if provided (batched or 
    single-sample dict). Otherwise pull sample from datamodule.dataset[index].
    If apply_aug=True and datamodule provides apply_train_aug, that will be used
    to produce an augmented batch (useful for inspecting training 
    augmentations).

    Returns matplotlib.figure.Figure.
    """
    import matplotlib.pyplot as plt
    import numpy as _np
    import torch as _torch

    if batch is None:
        if datamodule is None:
            raise ValueError("Either datamodule or batch must be provided.")
        if dataset == "train":
            ds = datamodule.train_dataset
        elif dataset in ("validate", "val"):
            ds = datamodule.val_dataset
        elif dataset == "test":
            ds = datamodule.test_dataset
        else:
            raise ValueError(f"unknown dataset: {dataset!r}")

        sample = ds[index]
        batch = {"image": sample["image"].unsqueeze(0), 
                 "mask": sample.get("mask", None).unsqueeze(0) \
                    if sample.get("mask", None) is not None else None}

        if apply_aug:
            # pick augmentation function consistent with dataset (avoid using
            # train augmentations for val/test). If no aug function exists for
            # val/test use identity that returns cloned tensors.
            if dataset == "train":
                apply_fn = getattr(datamodule, "apply_train_aug", None)
            elif dataset in ("validate", "val"):
                apply_fn = getattr(datamodule, "apply_val_aug", None)
            elif dataset == "test":
                apply_fn = getattr(datamodule, "apply_test_aug", None)
            else:
                apply_fn = None

            if apply_fn is None and dataset != "train":
                def _identity_apply(batch, seed=None):
                    img = batch["image"].clone() \
                        if hasattr(batch["image"], "clone") else batch["image"]
                    m = batch.get("mask", None)
                    if m is not None and hasattr(m, "clone"):
                        m = m.clone()
                    return {"image": img, "mask": m}
                apply_fn = _identity_apply

            if apply_fn is not None:
                batch = apply_fn(batch, seed=None)

    # extract first image if batched
    img = batch["image"]
    if isinstance(img, _torch.Tensor):
        img = img.detach().cpu().numpy()
    img = _np.asarray(img)
    # possible shapes: (B,C,H,W) or (C,H,W) or (H,W,C)
    if img.ndim == 4:
        img = img[0]  # take first in batch
    if img.ndim == 3 and img.shape[0] <= 4:  # assume (C,H,W)
        img_np = img.transpose(1, 2, 0)
    else:
        img_np = img  # already (H,W,C)

    fig = plt.figure(figsize=figsize)
    for i in range(img_np.shape[2]):
        plt.hist(img_np[..., i].ravel(), bins=bins, alpha=alpha, 
                 label=f"Band {i}")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.legend()
    resolved_title = title or \
        f"Histogram of Image Bands ({dataset}, idx={index})"
    plt.title(resolved_title)
    if show:
        plt.show()
        plt.close(fig)
    return fig