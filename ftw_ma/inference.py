import math
import os
import time
from typing import Literal, Optional, Union, Dict, Any, Tuple, List

import geopandas as gpd
import kornia.augmentation as K
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
import torch
import torch.nn.functional as F
from kornia.constants import Resample
from rasterio.enums import ColorInterp
from rasterio.transform import from_bounds
from rasterio.windows import from_bounds as window_from_bounds
from torch.utils.data import DataLoader
from torchgeo.datasets import stack_samples
from torchgeo.samplers import GridGeoSampler
from tqdm import tqdm
from pathlib import Path

from .dataset import SingleRasterDataset
from .trainers import CustomSemanticSegmentationTask

def run_batch_inference(
    input_data: Union[str, pd.DataFrame, gpd.GeoDataFrame],
    model: str,
    output_dir: str,
    normalization_strategy: str = "min_max",
    normalization_stat_procedure: str = "lab",
    global_stats: Optional[Union[Dict[str, Any], Tuple, List]] = None,
    img_clip_val: float = 0,
    nodata: Optional[List] = None,
    geometry_column: str = "geometry",
    path_column: str = "path",
    id_column: Optional[str] = None,
    crop_to_geometry: bool = True,
    **inference_kwargs
):
    """
    Run inference on multiple files, optionally cropping to geometries.
    
    Args:
        input_data: Single file path, DataFrame, or GeoDataFrame with file 
            paths
        model: Path to model checkpoint
        output_dir: Directory to save predictions
        geometry_column: Column name for geometries (GeoDataFrame only)
        path_column: Column name for file paths
        id_column: Column name for unique IDs (optional)
        crop_to_geometry: Whether to crop predictions to geometry bounds
        **inference_kwargs: Additional arguments for single-file inference
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle different input types
    if isinstance(input_data, str):
        # Single file - use existing inference
        output_path = output_dir / f"prediction_{Path(input_data).stem}.tif"
        return inference_run(
            input=input_data,
            model=model,
            out=str(output_path),
            normalization_strategy=normalization_strategy,
            normalization_stat_procedure=normalization_stat_procedure,
            global_stats=global_stats,
            img_clip_val=img_clip_val,
            nodata=nodata,
            **inference_kwargs
        )
    
    # DataFrame/GeoDataFrame processing
    if isinstance(input_data, gpd.GeoDataFrame):
        gdf = input_data.copy()
        has_geometry = True
    elif isinstance(input_data, pd.DataFrame):
        gdf = input_data.copy()
        has_geometry = False
        crop_to_geometry = False  # Can't crop without geometry
    else:
        raise ValueError("input_data must be str, DataFrame, or GeoDataFrame")
    
    # Validate required columns
    if path_column not in gdf.columns:
        raise ValueError(f"Column '{path_column}' not found in input data")
    
    if (has_geometry and crop_to_geometry and 
        geometry_column not in gdf.columns):
        raise ValueError(
            f"Column '{geometry_column}' not found in GeoDataFrame"
        )
    
    results = []
    
    for idx, row in tqdm(gdf.iterrows(), total=len(gdf), 
                         desc="Processing files"):
        try:
            input_path = row[path_column]
            
            # Generate output filename
            file_id = row.get(id_column, idx) if id_column else idx
            output_filename = (
                f"prediction_{file_id}_{Path(input_path).stem}.tif"
            )
            output_path = output_dir / output_filename
            
            if has_geometry and crop_to_geometry:
                # Run inference with geometry cropping
                success = run_inference_with_crop(
                    input_path=input_path,
                    geometry=row[geometry_column],
                    model=model,
                    output_path=output_path,
                    normalization_strategy=normalization_strategy,
                    normalization_stat_procedure=normalization_stat_procedure,
                    global_stats=global_stats,
                    img_clip_val=img_clip_val,
                    nodata=nodata,
                    **inference_kwargs
                )
            else:
                # Standard inference without cropping
                success = inference_run(
                    input=input_path,
                    model=model,
                    out=str(output_path),
                    normalization_strategy=normalization_strategy,
                    normalization_stat_procedure=normalization_stat_procedure,
                    global_stats=global_stats,
                    img_clip_val=img_clip_val,
                    nodata=nodata,
                    **inference_kwargs
                )
            
            results.append({
                'input_path': input_path,
                'output_path': str(output_path),
                'success': success,
                'id': file_id
            })
            
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            results.append({
                'input_path': input_path,
                'output_path': None,
                'success': False,
                'error': str(e),
                'id': file_id
            })
    
    # Save results summary
    results_df = pd.DataFrame(results)
    results_path = output_dir / "inference_results.csv"
    results_df.to_csv(results_path, index=False)
    
    print(f"Processed {len(results)} files")
    print(f"Success: {results_df['success'].sum()}/{len(results)}")
    print(f"Results saved to {results_path}")
    
    return results_df

def run_inference_with_crop(
    input_path: str,
    geometry,
    model: str,
    output_path: Path,
    buffer_pixels: int = 64,
    **inference_kwargs
):
    """
    Run inference on a file and crop the result to geometry bounds.
    
    Args:
        input_path: Path to input raster
        geometry: Shapely geometry to crop to
        model: Path to model checkpoint
        output_path: Path for output file
        buffer_pixels: Buffer around geometry in pixels
        **inference_kwargs: Arguments for inference_run
    """
    try:
        # Get geometry bounds and raster info
        with rasterio.open(input_path) as src:
            # Transform geometry to raster CRS if needed
            if hasattr(geometry, 'crs') and geometry.crs != src.crs:
                geometry = geometry.to_crs(src.crs)
            
            # Get bounding box with buffer
            bounds = geometry.bounds
            minx, miny, maxx, maxy = bounds
            
            # Convert to pixel coordinates
            left, top = ~src.transform * (minx, maxy)
            right, bottom = ~src.transform * (maxx, miny)
            
            # Add buffer and ensure within image bounds
            left = max(0, int(left) - buffer_pixels)
            top = max(0, int(top) - buffer_pixels)
            right = min(src.width, int(right) + buffer_pixels)
            bottom = min(src.height, int(bottom) + buffer_pixels)
            
            # Create window for cropping
            window = rasterio.windows.Window(
                left, top, right - left, bottom - top
            )
            
            # Create temporary cropped raster
            temp_path = output_path.parent / f"temp_{output_path.stem}.tif"
            
            # Read and write cropped data
            cropped_data = src.read(window=window)
            cropped_transform = rasterio.windows.transform(
                window, src.transform
            )
            
            profile = src.profile.copy()
            profile.update({
                'height': cropped_data.shape[1],
                'width': cropped_data.shape[2], 
                'transform': cropped_transform
            })
            
            with rasterio.open(temp_path, 'w', **profile) as dst:
                dst.write(cropped_data)
        
        # Run inference on cropped raster
        temp_pred_path = (
            output_path.parent / f"temp_pred_{output_path.stem}.tif"
        )
        
        success = inference_run(
            input=str(temp_path),
            model=model,
            out=str(temp_pred_path),
            **inference_kwargs
        )
        
        if success:
            # Mask prediction to exact geometry bounds
            with rasterio.open(temp_pred_path) as src:
                # Crop to geometry
                out_image, out_transform = rasterio.mask.mask(
                    src, [geometry], crop=True, filled=True, nodata=0
                )
                
                out_meta = src.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                    "nodata": 0
                })
                
                with rasterio.open(output_path, "w", **out_meta) as dst:
                    dst.write(out_image)
        
        # Clean up temporary files
        if temp_path.exists():
            temp_path.unlink()
        if temp_pred_path.exists():
            temp_pred_path.unlink()
            
        return success
        
    except Exception as e:
        print(f"Error in cropped inference for {input_path}: {e}")
        return False

def setup_inference(
    input,
    out,
    gpu,
    patch_size,
    padding,
    overwrite,
    mps_mode,
):
    """Setup inference parameters and validate inputs."""
    if not out:
        out = os.path.join(
            os.path.dirname(input), "inference." + os.path.basename(input)
        )
    if gpu is None:
        gpu = -1

    # IO related sanity checks
    assert os.path.exists(input), f"Input file {input} does not exist."
    assert input.endswith(".tif") or input.endswith(".vrt"), (
        "Input file must be a .tif or .vrt file."
    )
    assert overwrite or not os.path.exists(out), (
        f"Output file {out} already exists. Use -f to overwrite."
    )

    # Determine the device: GPU, MPS, or CPU
    if mps_mode:
        assert torch.backends.mps.is_available(), "MPS mode is not available."
        device = torch.device("mps")
    elif torch.cuda.is_available() and gpu >= 0:
        device = torch.device(f"cuda:{gpu}")
    else:
        print("Neither GPU nor MPS mode is enabled, defaulting to CPU.")
        device = torch.device("cpu")

    # Load the input raster
    with rasterio.open(input) as src:
        input_shape = src.shape
        input_height, input_width = input_shape[0], input_shape[1]
        print(f"Input image size: {input_height}x{input_width} pixels (HxW)")
        profile = src.profile
        transform = profile["transform"]

    # Determine the default patch size
    if patch_size is None:
        steps = [1024, 512, 256, 128]
        for step in steps:
            if step <= min(input_height, input_width):
                patch_size = step
                break
    print("Patch size:", patch_size)
    assert patch_size is not None, "Input image is too small"
    assert patch_size % 32 == 0, "Patch size must be a multiple of 32."
    assert patch_size <= min(input_height, input_width), (
        "Patch size must not be larger than the input image dimensions."
    )

    if padding is None:
        # 64 for patch sizes >= 1024, otherwise smaller paddings
        padding = math.ceil(min(1024, patch_size) / 16)
    print("Padding:", padding)

    stride = patch_size - padding * 2
    assert stride > 64, (
        "Patch size minus two times the padding must be greater than 64."
    )

    return device, transform, input_shape, patch_size, stride, padding

# Keep existing inference_run function with modifications for compatibility
def inference_run(
    input,
    model,
    out,
    resize_factor=1,
    gpu=None,
    patch_size=None,
    batch_size=1,
    num_workers=0,
    padding=None,
    overwrite=False,
    mps_mode=False,
    save_scores=False,
    # New normalization parameters
    normalization_strategy: str = "min_max",
    normalization_stat_procedure: str = "lab",
    global_stats: Optional[Union[Dict[str, Any], Tuple, List]] = None,
    img_clip_val: float = 0,
    nodata: Optional[List] = None,
):
    """Run inference with configurable normalization matching training."""
    
    try:
        device, transform, input_shape, patch_size, stride, padding = (
            setup_inference(
                input, out, gpu, patch_size, padding, overwrite, mps_mode
            )
        )

        assert os.path.exists(model), f"Model file {model} does not exist."
        assert model.endswith(".ckpt"), "Model file must be a .ckpt file."

        # Load task
        tic = time.time()
        task = CustomSemanticSegmentationTask.load_from_checkpoint(
            model, map_location="cpu"
        )
        task.freeze()
        model = task.model.eval().to(device)

        if mps_mode:
            up_sample = K.Resize(
                (
                    patch_size * resize_factor,
                    patch_size * resize_factor,
                )
            ).to("cpu")
            down_sample = (
                K.Resize(
                    (patch_size, patch_size), 
                    resample=Resample.NEAREST.name
                )
                .to(device)
                .to("cpu")
            )
        else:
            up_sample = K.Resize(
                (
                    patch_size * resize_factor,
                    patch_size * resize_factor,
                )
            ).to(device)
            down_sample = K.Resize(
                (patch_size, patch_size), resample=Resample.NEAREST.name
            ).to(device)

        # Use our custom dataset with normalization parameters
        dataset = SingleRasterDataset(
            fn=input,
            normalization_strategy=normalization_strategy,
            normalization_stat_procedure=normalization_stat_procedure,
            global_stats=global_stats,
            img_clip_val=img_clip_val,
            nodata=nodata,
        )
        
        sampler = GridGeoSampler(dataset, size=patch_size, stride=stride)
        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=stack_samples,
        )

        print(f"Using normalization: {normalization_strategy} with "
              f"{normalization_stat_procedure}")
        if global_stats:
            print(f"Global stats provided: {global_stats}")

        # Run inference
        if save_scores:
            output_mask = np.zeros([3, input_shape[0], input_shape[1]],
                                   dtype=np.float32)
        else:
            output_mask = np.zeros([1, input_shape[0], input_shape[1]], 
                                   dtype=np.uint8)
        dl_enumerator = tqdm(dataloader)

        for batch in dl_enumerator:
            images = batch["image"].to(device)
            images = up_sample(images)

            # WinB then WinA (B02_t2, B03_t2, B04_t2, B08_t2, B02_t1, 
            # B03_t1, B04_t1, B08_t1)
            num_bands = images.shape[1] // 2
            images = torch.cat(
                [images[:, num_bands:], images[:, :num_bands]], 
                dim=1
            )

            # torchgeo>=0.6 refers to the bounding box as "bounds" instead 
            # of "bbox"
            if "bounds" in batch and batch["bounds"] is not None:
                bboxes = batch["bounds"]
            else:
                bboxes = batch["bbox"]

            with torch.inference_mode():
                predictions = model(images)
                if save_scores:
                    # compute softmax to interpret logits as probabilities 
                    # [0, 1]
                    predictions = F.softmax(predictions, dim=1)
                    predictions = (
                        (down_sample(predictions.float())
                         .cpu().numpy().astype(np.float32))
                    )
                    # rescale probabilities from [0, 1] to [0, 255] & store 
                    # as uint8
                    predictions = (
                        (predictions * 255).clip(0, 255).astype(np.uint8)
                    )
                else:
                    predictions = predictions.argmax(axis=1).unsqueeze(0)
                    predictions = (down_sample(predictions.float())
                                   .int().cpu().numpy())

            for i in range(len(bboxes)):
                bb = bboxes[i]
                left, top = ~transform * (bb.minx, bb.maxy)
                right, bottom = ~transform * (bb.maxx, bb.miny)
                left, right, top, bottom = (
                    int(np.round(left)),
                    int(np.round(right)),
                    int(np.round(top)),
                    int(np.round(bottom)),
                )
                pleft = left + padding
                pright = right - padding
                ptop = top + padding
                pbottom = bottom - padding
                out_channels, destination_height, destination_width = (
                    output_mask[:, ptop:pbottom, pleft:pright].shape
                )
                if save_scores:
                    inp = predictions[i]
                else:
                    inp = predictions[:, i]
                inp = inp[
                    :,
                    padding : padding + destination_height,
                    padding : padding + destination_width,
                ]
                output_mask[:, ptop:pbottom, pleft:pright] = inp

        with rasterio.open(input) as src:
            profile = src.profile
            tags = src.tags()

        # Save predictions
        profile.update(
            {
                "driver": "GTiff",
                "count": out_channels,
                "dtype": "uint8",
                "compress": "lzw",
                "predictor": 2,
                "nodata": 0,
                "blockxsize": 512,
                "blockysize": 512,
                "tiled": True,
                "interleave": "pixel",
            }
        )

        with rasterio.open(out, "w", **profile) as dst:
            dst.update_tags(**tags)
            if save_scores:
                # write all logit channels
                dst.write(output_mask)
            else:
                # palette only for single-band labels
                dst.write_colormap(1, {1: (255, 0, 0), 2: (0, 255, 0)})
                dst.colorinterp = [ColorInterp.palette]
                dst.write(output_mask[0], 1)

        print(f"Finished inference and saved output to {out} "
              f"in {time.time() - tic:.2f}s")
        return True
        
    except Exception as e:
        print(f"Inference failed: {e}")
        return False
