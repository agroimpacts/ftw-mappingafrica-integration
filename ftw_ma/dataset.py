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
# from torchgeo.datasets.utils import array_to_tensor
from torchgeo.datasets import NonGeoDataset, RasterDataset
from .utils import * 

# class SingleRasterDataset(RasterDataset):
#     """A torchgeo dataset that loads a single raster file with custom 
#     normalization."""

#     def __init__(
#         self, 
#         fn: str, 
#         transforms: Optional[Callable] = None,
#         normalization_strategy: str = "min_max",
#         normalization_stat_procedure: str = "lab", 
#         global_stats: Optional[Union[Dict[str, Any], Tuple, List]] = None,
#         img_clip_val: float = 0,
#         nodata: Optional[List] = None,
#         auto_detect_nodata: bool = True,
#     ):
#         """Initialize the SingleRasterDataset class."""
#         path = os.path.abspath(fn)
#         self.filename_regex = os.path.basename(path)
        
#         # Store normalization parameters
#         self.normalization_strategy = normalization_strategy
#         self.normalization_stat_procedure = normalization_stat_procedure
#         self.global_stats = global_stats
#         self.img_clip_val = img_clip_val
#         # self.auto_detect_nodata = auto_detect_nodata
        
#         # Get actual nodata from file if auto_detect is enabled
#         if auto_detect_nodata:
#             with rasterio.open(path) as src:
#                 file_nodata = src.nodata
#                 print(f"File nodata from metadata: {file_nodata}")
#                 if file_nodata is not None:
#                     self.nodata = [file_nodata]
#                 else:
#                     self.nodata = nodata or []  # Empty list if no nodata specified
#         else:
#             self.nodata = nodata or [65535]
        
#         print(f"Using nodata values: {self.nodata}")
        
#         super().__init__(paths=os.path.dirname(path), transforms=transforms)
        
#         # Debug: Check what the parent dataset thinks about this file
#         print(f"Dataset bounds: {self.bounds}")
#         print(f"Dataset CRS: {self.crs}")
#         print(f"Dataset files found: {len(self.files)}")
#         if self.files:
#             print(f"First few files: {list(self.files)[:3]}")

#     def __getitem__(self, query):
#         """Override to apply custom normalization using load_image approach."""
#         # print(f"SingleRasterDataset.__getitem__ called with query: {query}")
        
#         # Debug: Let's check what the query bounds are vs file bounds
#         if hasattr(query, '__len__') and len(query) >= 2:
#             query_x_slice = query[0] if hasattr(query[0], 'start') else None
#             query_y_slice = query[1] if hasattr(query[1], 'start') else None
#             if query_x_slice and query_y_slice:
#                 print(f"Query bounds: x=[{query_x_slice.start}, {query_x_slice.stop}], "
#                       f"y=[{query_y_slice.start}, {query_y_slice.stop}]")
#                 print(f"Dataset bounds: {self.bounds}")
#                 print(f"Dataset bounds type: {type(self.bounds)}")
                
#                 # Check if query is within dataset bounds - handle different bounds types
#                 try:
#                     if hasattr(self.bounds, 'minx'):
#                         # BoundingBox object
#                         dataset_minx, dataset_maxx = self.bounds.minx, self.bounds.maxx
#                         dataset_miny, dataset_maxy = self.bounds.miny, self.bounds.maxy
#                     elif len(self.bounds) == 4:
#                         # Tuple or list: (minx, miny, maxx, maxy)
#                         dataset_minx, dataset_miny, dataset_maxx, dataset_maxy = self.bounds
#                     else:
#                         print(f"Unknown bounds format: {self.bounds}")
#                         dataset_minx = dataset_miny = float('-inf')
#                         dataset_maxx = dataset_maxy = float('inf')
                    
#                     query_within_bounds = (
#                         query_x_slice.start >= dataset_minx and 
#                         query_x_slice.stop <= dataset_maxx and
#                         query_y_slice.start >= dataset_miny and 
#                         query_y_slice.stop <= dataset_maxy
#                     )
                    
#                     if query_within_bounds:
#                         print("Query is within dataset bounds ✓")
#                     else:
#                         print("Query is OUTSIDE dataset bounds ✗")
#                         print(f"  Query x: [{query_x_slice.start}, {query_x_slice.stop}] vs Dataset x: [{dataset_minx}, {dataset_maxx}]")
#                         print(f"  Query y: [{query_y_slice.start}, {query_y_slice.stop}] vs Dataset y: [{dataset_miny}, {dataset_maxy}]")
                        
#                 except Exception as e:
#                     print(f"Error checking bounds: {e}")
        
#         # Get the raw sample from parent (loads raster data without normalization)
#         print(f"Calling parent.__getitem__ with query: {query}")
#         sample = super().__getitem__(query)
#         print(f"Parent returned sample with keys: {sample.keys()}")
        
#         # Debug the actual data that came back
#         if 'image' in sample:
#             img = sample['image']
#             if hasattr(img, 'shape'):
#                 print(f"Sample image stats: shape={img.shape}, dtype={img.dtype}")
#                 if img.numel() > 0:
#                     print(f"  Data range: [{img.min():.2f}, {img.max():.2f}]")
#                     print(f"  Unique values: {torch.unique(img)[:10]}")
#                 else:
#                     print("  Image tensor is empty!")
        
#         # Extract image tensor - RasterDataset typically returns (C, H, W) tensors
#         image_tensor = sample["image"]
#         print(f"Image tensor type: {type(image_tensor)}")
#         print(f"Image tensor shape: {image_tensor.shape if hasattr(image_tensor, 'shape') else 'no shape'}")
        
#         # Convert tensor to numpy for our load_image normalization
#         if isinstance(image_tensor, torch.Tensor):
#             image_np = image_tensor.detach().cpu().numpy()
#         else:
#             image_np = np.asarray(image_tensor)
        
#         print(f"Raw image shape: {image_np.shape}")
#         print(f"Raw image range: [{image_np.min():.2f}, {image_np.max():.2f}]")
#         print(f"Raw image dtype: {image_np.dtype}")
#         print(f"Raw image unique values (first 10): {np.unique(image_np.flatten())[:10]}")
        
#         # Check for different types of nodata/invalid values
#         total_pixels = image_np.size
#         zero_pixels = np.sum(image_np == 0)
#         max_pixels = np.sum(image_np == 65535)
#         nan_pixels = np.sum(np.isnan(image_np))
#         # none_pixels = np.sum(image_np == None)  # This might not work as expected
        
#         print(f"Pixel analysis: total={total_pixels}, zeros={zero_pixels}, "
#               f"65535s={max_pixels}, nans={nan_pixels}")
        
#         # Check if image is mostly nodata
#         nodata_pixels = zero_pixels + max_pixels + nan_pixels
#         nodata_ratio = nodata_pixels / total_pixels
#         print(f"Nodata ratio: {nodata_ratio:.3f}")
        
#         if nodata_ratio > 0.9:  # More than 90% nodata
#             print(f"Warning: Image patch is mostly nodata ({nodata_ratio:.1%})")
        
#         # WORKAROUND: If we get all zeros but file has valid data, read directly
#         if (image_np.min() == 0 and image_np.max() == 0 and 
#             hasattr(query, '__len__') and len(query) >= 2):
            
#             print("🔧 WORKAROUND: Detected all zeros, trying direct file read...")
#             try:
#                 file_path = list(self.files)[0]  # Get the first (should be only) file
                
#                 with rasterio.open(file_path) as src:
#                     # Read the entire image directly
#                     direct_data = src.read()  # Shape: (bands, height, width)
#                     print(f"Direct read shape: {direct_data.shape}")
#                     print(f"Direct read range: [{direct_data.min()}, {direct_data.max()}]")
                    
#                     if direct_data.max() > 0:  # File has valid data
#                         # Convert to float and rearrange to (C, H, W)
#                         image_np = direct_data.astype(np.float32)
#                         print(f"Using direct file read instead of torchgeo query")
                        
#                         # Re-run the normalization with valid data
#                         normalized_image = normalize_image(
#                             img=image_np,
#                             strategy=self.normalization_strategy,
#                             procedure=self.normalization_stat_procedure,
#                             global_stats=self.global_stats,
#                             clip_val=self.img_clip_val,
#                             nodata=self.nodata,
#                         )
                        
#                         sample["image"] = torch.from_numpy(normalized_image).float()
#                         print(f"Workaround successful: range [{sample['image'].min():.2f}, {sample['image'].max():.2f}]")
#                         return sample
                        
#             except Exception as e:
#                 print(f"Workaround failed: {e}")
#                 # Fall through to use the zero data
        
#         # Normal path: apply normalization to whatever data we got
#         normalized_image = normalize_image(
#             img=image_np,
#             strategy=self.normalization_strategy,
#             procedure=self.normalization_stat_procedure,
#             global_stats=self.global_stats,
#             clip_val=self.img_clip_val,
#             nodata=self.nodata,
#         )
        
#         print(f"Normalized image shape: {normalized_image.shape}")
#         print(f"Normalized image range: [{normalized_image.min():.2f}, {normalized_image.max():.2f}]")
#         print(f"Normalized image dtype: {normalized_image.dtype}")
#         print(f"Normalized unique values (first 10): {np.unique(normalized_image.flatten())[:10]}")
        
#         # Convert back to tensor and update sample
#         sample["image"] = torch.from_numpy(normalized_image).float()
        
#         return sample

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
            
        print(f"Loaded image: {self.file_path}")
        print(f"  Shape: {self.image_data.shape}")
        print(f"  Data type: {self.image_data.dtype}")
        print(f"  Value range: [{self.image_data.min()}, {self.image_data.max()}]")
        print(f"  CRS: {self.crs}")
        print(f"  Bounds: {self.bounds}")
        
        # Apply normalization to the full image
        self.normalized_image = normalize_image(
            img=self.image_data.astype(np.float32),
            strategy=self.normalization_strategy,
            procedure=self.normalization_stat_procedure,
            global_stats=self.global_stats,
            clip_val=self.img_clip_val,
            nodata=self.nodata,
        )
        
        print(f"Normalized image range: [{self.normalized_image.min():.2f}, {self.normalized_image.max():.2f}]")
    
    def get_patch(self, bbox):
        """Extract a patch from the image given a bounding box.
        
        Args:
            bbox: (minx, miny, maxx, maxy) in the image's CRS
            
        Returns:
            torch.Tensor: Image patch as tensor (C, H, W)
        """
        # Convert geographic bounds to pixel coordinates
        from rasterio.windows import from_bounds
        window = from_bounds(*bbox, self.transform)
        
        # Convert window to integer pixel coordinates
        col_off = int(window.col_off)
        row_off = int(window.row_off)
        width = int(window.width)
        height = int(window.height)
        
        # Clamp to image bounds
        col_off = max(0, min(col_off, self.width))
        row_off = max(0, min(row_off, self.height))
        width = min(width, self.width - col_off)
        height = min(height, self.height - row_off)
        
        # Extract patch
        patch = self.normalized_image[:, row_off:row_off+height, 
                                      col_off:col_off+width]
        
        # Convert to tensor
        return torch.from_numpy(patch).float()
    
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
            wb_entry = wb if ("windowB" in self.temporal_options) else None
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