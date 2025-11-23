# FTWMapAfrica dataset class for loading and processing field boundary imagery 
# and labels.
import os
from pathlib import Path
import random
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional, Callable, Union, Dict, Any, Tuple, List
import pandas as pd
import numpy as np
import rasterio
import torch
from skimage.exposure import rescale_intensity
from torch import Tensor
from torchgeo.datasets import NonGeoDataset, RasterDataset
from .normalize import normalize_image

def load_image(
    filename,
    nodata_val_ls=None,
    apply_normalization=False,
    normal_strategy="min_max",
    stat_procedure="lab",
    global_stats=None,
    clip_val=0,
):
    """
    Load and preprocess an image.

    Args:
        filename (str): Filename of the image.
        nodata_val_ls (list, optional): List of no data value for each band.
        apply_normalization (bool, optional): Whether to apply normalization.
        normal_strategy (str, optional): Normalization strategy.
        stat_procedure (str, optional): Statistic procedure to be used for
            normalization.
        global_stats (tuple, list, optional): Global stats for normalization.
        clip_val (float, optional): Clipping value for normalization.

    Returns:
        np.ndarray: Loaded and preprocessed image as a numpy array.
    """
    with rasterio.open(filename, "r") as f:
        img = f.read()

    img = img.astype(np.float32)

    if apply_normalization:
        img = normalize_image(
            img,
            strategy=normal_strategy,
            procedure=stat_procedure,
            global_stats=global_stats,
            clip_val=clip_val,
            nodata=nodata_val_ls,
        )

    return img

# ... rest of the dataset.py code remains the same ...
class SimpleRasterDataset:
    """A simple dataset that reads raster files directly with rasterio."""
    
    def __init__(
        self,
        file_path: str,
        normalization_strategy: str = "min_max",
        normalization_stat_procedure: str = "lab",
        global_stats: Optional[Union[Dict[str, Any], Tuple, List]] = None,
        img_clip_val: float = 0,
        nodata: Optional[List] = None,
    ):
        """Initialize the SimpleRasterDataset."""
        self.file_path = file_path
        self.normalization_strategy = normalization_strategy
        self.normalization_stat_procedure = normalization_stat_procedure
        self.global_stats = global_stats
        self.img_clip_val = img_clip_val
        self.nodata = nodata or [65535]
        
        # Read the image and store it
        with rasterio.open(file_path) as src:
            self.image_data = src.read()  # Shape: (bands, height, width)
            self.profile = src.profile
            self.transform = src.transform
            self.bounds = src.bounds
            self.crs = src.crs
            self.height, self.width = src.height, src.width
            
        # print(f"Loaded image: {self.file_path}")
        # print(f"  Shape: {self.image_data.shape}")
        # print(f"  Data type: {self.image_data.dtype}")
        # # Print value range, breaking at 80 characters
        # print(
        #     f"  Value range: "
        #     f"[{self.image_data.min()}, {self.image_data.max()}]"
        # )
        # print(f"  CRS: {self.crs}")
        # print(f"  Bounds: {self.bounds}")
        
        # Apply normalization to the full image
        self.normalized_image = normalize_image(
            img=self.image_data.astype(np.float32),
            strategy=self.normalization_strategy,
            procedure=self.normalization_stat_procedure,
            global_stats=self.global_stats,
            clip_val=self.img_clip_val,
            nodata=self.nodata,
        )
        
        print(
            f"Normalized image range: "
            f"[{self.normalized_image.min():.2f}, "
            f"{self.normalized_image.max():.2f}]"
        )
    
    def get_full_image(self):
        """Get the full normalized image as a tensor.
        
        Returns:
            torch.Tensor: Full image as tensor (C, H, W)
        """
        return torch.from_numpy(self.normalized_image).float()

class FTWMapAfrica(NonGeoDataset):
    """
    Dataset class for loading and processing FTW Mapping Africa imagery
    and labels.
    """
    valid_splits = ["train", "validate", "test"]
    def __init__(
        self,
        catalog: str = "../data/ftw-catalog-small.csv",
        data_dir: str = None,
        split: str = "train",
        transforms: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        temporal_options: str = "windowB",
        num_samples: int = -1,
        normalization_strategy: str = "min_max",
        normalization_stat_procedure: str = "lab",
        global_stats: Optional[Union[Dict[str, Any], Tuple, List]] = None,
        img_clip_val: float = 0,
        nodata: list = None,
    ) -> None:

        """
        Initialize the FTWMapAfrica dataset.
        Args:
            catalog (str): Path to the label catalog CSV.
            data_dir (str): Directory containing imagery and labels 
                (hereafter referred to as masks).
            split (str): Which split to use ('train', 'validate', 'test').
            transforms (callable): Augmentation/transformation pipeline.
            temporal_options (str): Temporal stacking options.
            num_samples (int): Number of samples to use (-1 for all).
            normalization_strategy (str): Normalization strategy.
            normalization_stat_procedure (str): Procedure for normalization.
            global_stats (tuple, list): Precomputed global stats.
            img_clip_val (float): Value to clip image data.
            nodata (list): List of nodata values.
        """
        self.data = pd.read_csv(catalog).query(f"split == '{split}'")
        self.data_dir = data_dir
        self.temporal_options = temporal_options
        self.transforms = transforms
        self.num_samples = num_samples
        self.normalization_strategy = normalization_strategy
        self.normalization_stat_procedure = normalization_stat_procedure
        self.global_stats = global_stats
        self.img_clip_val = img_clip_val
        self.nodata = nodata
        self.filenames = []
        all_filenames = []

        base = Path(self.data_dir) if self.data_dir is not None else Path(".")

        def _to_path(value):
            # pandas may represent missing fields as NaN (a float)
            if pd.isna(value):
                return None
            return base / str(value)

        for _, row in self.data.iterrows():
            wa = _to_path(row.get("window_a"))
            wb = _to_path(row.get("window_b"))
            m = _to_path(row.get("mask"))
            # skip entries missing a mask (can't train without labels)
            if m is None:
                continue
            # only include window_b when temporal option requires it
            # wb_entry = wb if ("windowB" in self.temporal_options) else None
            wb_entry = wb if self.temporal_options in ("stacked", "windowB") \
                else None
            all_filenames.append({"window_a": wa, "window_b": wb_entry, 
                                  "mask": m})

        if self.num_samples == -1:  # select all samples
            self.filenames = all_filenames
        else:
            self.filenames = random.sample(
                all_filenames, min(self.num_samples, len(all_filenames))
            )
        
        print("Selecting : ", len(self.filenames), " samples")

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.filenames)
    
    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            dictionary containing "image" and "mask" PyTorch tensors
        """

        filenames = self.filenames[index]

        # Load image(s) - allows for multiple time points
        # Possible expansion to add in other layers, e.g., DEM, slope, etc. 
        images = []
        if self.temporal_options in ("stacked", "windowA"):
            window_a_img = load_image(
                filenames["window_a"], 
                nodata_val_ls=self.nodata, 
                apply_normalization=True,      
                normal_strategy=self.normalization_strategy,
                stat_procedure=self.normalization_stat_procedure,
                global_stats=self.global_stats,
                clip_val=self.img_clip_val
            )           
            images.append(window_a_img)

        if self.temporal_options in ("stacked", "windowB"):
            window_b_img = load_image(
                filenames["window_b"], 
                nodata_val_ls=self.nodata, 
                apply_normalization=True,      
                normal_strategy=self.normalization_strategy,
                stat_procedure=self.normalization_stat_procedure,
                global_stats=self.global_stats,
                clip_val=self.img_clip_val
            )
            images.append(window_b_img)

        # image = torch.cat(images, dim=0)
        image = np.concatenate(images, axis=0).astype(np.float32)
        image = torch.from_numpy(image).float()

        # Load mask
        with rasterio.open(filenames["mask"]) as f:
            mask = f.read(1)
        mask = torch.from_numpy(mask).long()

        sample = {"image": image, "mask": mask}

        # print(self.transforms)
        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample    

    def plot(self, sample: dict[str, Tensor], 
             suptitle: Optional[str] = None) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample return by :meth:`__getitem__`
            suptitle: optional suptitle to use for figure

        Returns:
            a matplotlib Figure with the rendered sample
        """
        img1 = sample["image"][0:3].numpy().transpose(1, 2, 0)
        # print("Image shape: ", img1.shape)

        if self.temporal_options == "stacked": 
            print("Plotting stacked images")
            img2 = sample["image"][4:7]
            img2 = img2.numpy().transpose(1, 2, 0)
            # print("Image shape: ", img2.shape)

        mask = sample["mask"].numpy().squeeze()
        num_panels = 3 if self.temporal_options in ("stacked", "rgb") else 2
        if "prediction" in sample:
            num_panels += 1
            predictions = sample["prediction"].numpy()

        fig, axs = plt.subplots(1, num_panels, figsize=(num_panels * 5, 8))
        axs = axs.flatten()
        axs[0].imshow(np.clip(img1, 0, 1))
        axs[0].axis("off")

        panel_id = 1
        if self.temporal_options in ("stacked", "rgb"):
            axs[panel_id].imshow(np.clip(img2, 0, 1))
            axs[panel_id].axis("off")
            axs[panel_id + 1].imshow(mask, vmin=0, vmax=2, cmap="gray")
            axs[panel_id + 1].axis("off")
            panel_id += 2
        else:
            axs[panel_id].imshow(mask, vmin=0, vmax=2, cmap="gray")
            axs[panel_id].axis("off")
            panel_id += 1

        if "prediction" in sample:
            axs[panel_id].imshow(predictions, vmin=0, vmax=2, cmap="gray")
            axs[panel_id].axis("off")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig