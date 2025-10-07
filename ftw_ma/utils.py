"""
Utility functions for FTW Mapping Africa.
"""
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import torch

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