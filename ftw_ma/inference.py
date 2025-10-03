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
import torchgeo
from torchgeo.datasets import stack_samples
from torchgeo.samplers import GridGeoSampler
from tqdm import tqdm
from pathlib import Path
from packaging.version import Version, parse

# from .dataset import SingleRasterDataset
from .dataset import SimpleRasterDataset
from .trainers import CustomSemanticSegmentationTask

# TORCHGEO_06 = Version("0.6.0")
# TORCHGEO_08 = Version("0.8.0.dev0")
# TORCHGEO_CURRENT = parse(torchgeo.__version__)

# def setup_inference(
#     input,
#     out,
#     gpu,
#     patch_size,
#     padding,
#     overwrite,
#     mps_mode,
# ):
#     """Setup inference parameters and validate inputs."""
#     if not out:
#         out = os.path.join(
#             os.path.dirname(input), "inference." + os.path.basename(input)
#         )
#     if gpu is None:
#         gpu = -1

#     # IO related sanity checks
#     assert os.path.exists(input), f"Input file {input} does not exist."
#     assert input.endswith(".tif") or input.endswith(".vrt"), (
#         "Input file must be a .tif or .vrt file."
#     )
#     assert overwrite or not os.path.exists(out), (
#         f"Output file {out} already exists. Use -f to overwrite."
#     )

#     # Determine the device: GPU, MPS, or CPU
#     if mps_mode:
#         assert torch.backends.mps.is_available(), "MPS mode is not available."
#         device = torch.device("mps")
#     elif torch.cuda.is_available() and gpu >= 0:
#         device = torch.device(f"cuda:{gpu}")
#     else:
#         print("Neither GPU nor MPS mode is enabled, defaulting to CPU.")
#         device = torch.device("cpu")

#     # Load the input raster
#     with rasterio.open(input) as src:
#         input_shape = src.shape
#         input_height, input_width = input_shape[0], input_shape[1]
#         print(f"Input image size: {input_height}x{input_width} pixels (HxW)")
#         profile = src.profile
#         transform = profile["transform"]

#     # Determine the default patch size
#     if patch_size is None:
#         steps = [1024, 512, 256, 128]
#         for step in steps:
#             if step <= min(input_height, input_width):
#                 patch_size = step
#                 break
#     print("Patch size:", patch_size)
#     assert patch_size is not None, "Input image is too small"
#     assert patch_size % 32 == 0, "Patch size must be a multiple of 32."
#     assert patch_size <= min(input_height, input_width), (
#         "Patch size must not be larger than the input image dimensions."
#     )

#     if padding is None:
#         # 64 for patch sizes >= 1024, otherwise smaller paddings
#         padding = math.ceil(min(1024, patch_size) / 16)
#     print("Padding:", padding)

#     stride = patch_size - padding * 2
#     assert stride > 64, (
#         "Patch size minus two times the padding must be greater than 64."
#     )

#     return device, transform, input_shape, patch_size, stride, padding



# # Keep existing inference_run function with modifications for compatibility
# def inference_run(
#     input,
#     model,
#     out,
#     resize_factor=1,
#     gpu=None,
#     patch_size=None,
#     batch_size=1,
#     num_workers=0,
#     padding=None,
#     overwrite=False,
#     mps_mode=False,
#     save_scores=False,
#     # New normalization parameters
#     normalization_strategy: str = "min_max",
#     normalization_stat_procedure: str = "lab",
#     global_stats: Optional[Union[Dict[str, Any], Tuple, List]] = None,
#     img_clip_val: float = 0,
#     nodata: Optional[List] = None,
# ):
#     """Run inference with configurable normalization matching training."""
    
#     device, transform, input_shape, patch_size, stride, padding = (
#         setup_inference(
#             input, out, gpu, patch_size, padding, overwrite, mps_mode
#         )
#     )

#     assert os.path.exists(model), f"Model file {model} does not exist."
#     assert model.endswith(".ckpt"), "Model file must be a .ckpt file."

#     # Load task
#     tic = time.time()
#     task = CustomSemanticSegmentationTask.load_from_checkpoint(
#         model, map_location="cpu"
#     )
#     task.freeze()
#     model = task.model.eval().to(device)
#     model_type = task.hparams["model"]

#     # Ensure resize factor produces integers
#     patch_size_resized = int(patch_size * resize_factor)
#     print(f"Patch size: {patch_size}, Resize factor: {resize_factor}, Resized: {patch_size_resized}")

#     if mps_mode:
#         # For MPS, keep everything on device and use native PyTorch functions
#         def up_sample_fn(x):
#             return F.interpolate(
#                 x.to(device), 
#                 size=(patch_size_resized, patch_size_resized), 
#                 mode='bilinear', 
#                 align_corners=False
#             )
        
#         def down_sample_fn(x):
#             return F.interpolate(
#                 x.to(device), 
#                 size=(patch_size, patch_size), 
#                 mode='nearest'
#             )
        
#         up_sample = up_sample_fn
#         down_sample = down_sample_fn
#     else:
#         up_sample = K.Resize(
#             (patch_size_resized, patch_size_resized)
#         ).to(device)
#         down_sample = K.Resize(
#             (patch_size, patch_size), resample=Resample.NEAREST.name
#         ).to(device)

#     # Use our custom dataset with normalization parameters
#     dataset = SingleRasterDataset(
#         fn=input,
#         normalization_strategy=normalization_strategy,
#         normalization_stat_procedure=normalization_stat_procedure,
#         global_stats=global_stats,
#         img_clip_val=img_clip_val,
#         nodata=nodata,
#     )
        
#     # Debug: Check a sample from dataset
#     try:
#         sample = dataset[dataset.bounds]
#         print(f"Dataset sample image shape: {sample['image'].shape}")
#         print(f"Dataset sample range: [{sample['image'].min():.2f}, {sample['image'].max():.2f}]")
#     except Exception as e:
#         print(f"Error getting dataset sample: {e}")
        
#     sampler = GridGeoSampler(dataset, size=patch_size, stride=stride)
#     dataloader = DataLoader(
#         dataset,
#         sampler=sampler,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         collate_fn=stack_samples,
#     )

#     print(f"Using normalization: {normalization_strategy} with {normalization_stat_procedure}")
#     if global_stats:
#         print(f"Global stats provided: {global_stats}")

#     # Run inference
#     input_height, input_width = input_shape[0], input_shape[1]
#     if save_scores:
#         out_channels = 3
#     else:
#         out_channels = 1
#     output_mask = np.zeros([out_channels, input_height, input_width], dtype=np.uint8)
#     dl_enumerator = tqdm(dataloader)

#     for batch in dl_enumerator:
#         images = batch["image"].to(device)

#         print(f"Images device before up_sample: {images.device}")
#         print(f"Batch image shape before up_sample: {images.shape}")
#         print(f"Batch image range before up_sample: [{images.min():.2f}, {images.max():.2f}]")
        
#         images = up_sample(images)
#         print(f"Batch image shape after up_sample: {images.shape}")
#         print(f"Images device after up_sample: {images.device}")

#         # Check if we have temporal data (8 bands) or single window (4 bands)
#         if images.shape[1] == 8:
#             # Temporal stacking: WinB then WinA reordering
#             num_bands = images.shape[1] // 2
#             images = torch.cat([images[:, num_bands:], images[:, :num_bands]], dim=1)
#             print(f"Applied temporal reordering for 8-band input")
#         elif images.shape[1] == 4:
#             # Single window data - no reordering needed
#             print(f"Using single window data (4 bands)")
#         else:
#             raise ValueError(f"Unexpected number of input bands: {images.shape[1]}. Expected 4 or 8.")
                
#         print(f"Batch image shape after band reordering: {images.shape}")
#         print(f"Model expects 4 input channels")

#         # Handle bboxes for different torchgeo versions
#         bboxes = []
#         if TORCHGEO_CURRENT >= TORCHGEO_08:
#             for slices in batch["bounds"]:
#                 minx = slices[0].start
#                 maxx = slices[0].stop
#                 miny = slices[1].start
#                 maxy = slices[1].stop
#                 bboxes.append((minx, miny, maxx, maxy))
#         elif TORCHGEO_CURRENT >= TORCHGEO_06:
#             for bbox in batch["bounds"]:
#                 bboxes.append((bbox.minx, bbox.miny, bbox.maxx, bbox.maxy))
#         else:
#             for bbox in batch["bbox"]:
#                 bboxes.append((bbox.minx, bbox.miny, bbox.maxx, bbox.maxy))
        
#         print(f"Images device before model: {images.device}")
#         print(f"Model device: {next(model.parameters()).device}")
        
#         with torch.inference_mode():
#             predictions = model(images)

#             print(f"Model output shape: {predictions.shape}")
#             print(f"Model output range: [{predictions.min():.2f}, {predictions.max():.2f}]")

#             if save_scores:
#                 predictions = F.softmax(predictions, dim=1)
#                 predictions = down_sample(predictions.float()).cpu().numpy().astype(np.float32)
#                 predictions = (predictions * 255).clip(0, 255).astype(np.uint8)
#             else:
#                 predictions = predictions.argmax(axis=1).unsqueeze(0)
#                 print(f"Predictions after argmax shape: {predictions.shape}")
#                 print(f"Unique prediction values: {torch.unique(predictions)}")
#                 predictions = down_sample(predictions.float()).int().cpu().numpy()
#                 print(f"Final predictions shape: {predictions.shape}")
#                 print(f"Final unique prediction values: {np.unique(predictions)}")

#         # Process predictions into output mask
#         for i in range(len(bboxes)):
#             minx, miny, maxx, maxy = bboxes[i]

#             left, bottom = ~transform * (minx, miny)
#             right, top = ~transform * (maxx, maxy)
#             left, right, top, bottom = (
#                 int(np.round(left)),
#                 int(np.round(right)),
#                 int(np.round(top)),
#                 int(np.round(bottom)),
#             )

#             # Determine per-side effective padding
#             effective_left_pad = 0 if left <= 0 else padding
#             effective_right_pad = 0 if right >= input_width else padding
#             effective_top_pad = 0 if top <= 0 else padding
#             effective_bottom_pad = 0 if bottom >= input_height else padding

#             # Interior coordinates after trimming padding
#             pleft = left + effective_left_pad
#             pright = right - effective_right_pad
#             ptop = top + effective_top_pad
#             pbottom = bottom - effective_bottom_pad

#             # Clamp to image bounds
#             dst_left = max(pleft, 0)
#             dst_top = max(ptop, 0)
#             dst_right = min(pright, input_width)
#             dst_bottom = min(pbottom, input_height)

#             # Source indices within prediction patch
#             src_left = effective_left_pad + (dst_left - pleft)
#             src_right = effective_left_pad + (dst_right - pleft)
#             src_top = effective_top_pad + (dst_top - ptop)
#             src_bottom = effective_top_pad + (dst_bottom - ptop)

#             _, h, w = predictions[i].shape
#             src_left = max(0, min(src_left, w))
#             src_right = max(0, min(src_right, w))
#             src_top = max(0, min(src_top, h))
#             src_bottom = max(0, min(src_bottom, h))
            
#             if src_right <= src_left or src_bottom <= src_top:
#                 continue

#             # Place prediction in output mask
#             output_mask[:, dst_top:dst_bottom, dst_left:dst_right] = \
#                 predictions[i][:, src_top:src_bottom, src_left:src_right]

#     # Save output
#     with rasterio.open(input) as src:
#         profile = src.profile
#         tags = src.tags()

#     profile.update({
#         "driver": "GTiff",
#         "count": out_channels,
#         "dtype": "uint8",
#         "compress": "lzw",
#         "predictor": 2,
#         "nodata": 0,
#         "blockxsize": 512,
#         "blockysize": 512,
#         "tiled": True,
#         "interleave": "pixel",
#     })

#     with rasterio.open(out, "w", **profile) as dst:
#         dst.update_tags(**tags)
#         if save_scores:
#             dst.write(output_mask)
#         else:
#             dst.write_colormap(1, {1: (255, 0, 0), 2: (0, 255, 0)})
#             dst.colorinterp = [ColorInterp.palette]
#             dst.write(output_mask[0], 1)

#     print(f"Finished inference and saved output to {out} in {time.time() - tic:.2f}s")
#     return True
        
#     # except Exception as e:
#     #     print(f"Inference failed: {e}")
#     #     return False

# Add this new function that separates model loading from inference
def load_model(model_path, device):
    """Load model once and return it for reuse."""
    from .trainers import CustomSemanticSegmentationTask
    
    print("Loading model...")
    task = CustomSemanticSegmentationTask.load_from_checkpoint(
        model_path, map_location="cpu"
    )
    task.freeze()
    model_net = task.model.eval().to(device)
    print(f"Model loaded and moved to {device}")
    return model_net

def inference_run_single(
    input_file,
    model_net,  # Pre-loaded model
    device,
    out,
    save_scores=False,
    normalization_strategy="min_max",
    normalization_stat_procedure="lab",
    global_stats=None,
    img_clip_val=0,
    nodata=None,
    overwrite=False,
    patch_size=None,  # Will auto-determine if None
    buffer_size=None, # Buffer around edges
    band_order=None,  # New parameter
):
    """Run inference on a single file with pre-loaded model."""
    
    try:
        # Validate inputs
        assert os.path.exists(input_file), f"Input file {input_file} does not exist."
        assert overwrite or not os.path.exists(out), f"Output file {out} already exists. Use -f to overwrite."
        
        # Load image with normalization
        print(f"Loading and normalizing image: {input_file}")
        dataset = SimpleRasterDataset(
            file_path=input_file,
            normalization_strategy=normalization_strategy,
            normalization_stat_procedure=normalization_stat_procedure,
            global_stats=global_stats,
            img_clip_val=img_clip_val,
            nodata=nodata,
        )
        
        # Get the full normalized image
        image_tensor = dataset.get_full_image()  # Shape: (C, H, W)
        print(f"Image tensor shape: {image_tensor.shape}")
        print(f"Image tensor range: [{image_tensor.min():.2f}, {image_tensor.max():.2f}]")
        
        channels, height, width = image_tensor.shape
        
        # Auto-determine patch size if not provided
        if patch_size is None:
            # Auto-determine optimal patch size based on image dimensions
            min_dim = min(height, width)
            max_dim = max(height, width)
            
            # Try different patch sizes and pick the one that gives good coverage
            possible_sizes = [256, 384, 512, 768, 1024]
            
            best_patch_size = 256  # Default fallback
            best_score = float('inf')
            
            print(f"Evaluating patch sizes for {height}x{width} image:")
            for size in possible_sizes:
                if size > min_dim:
                    continue  # Skip if patch is larger than image
                
                # Calculate how many patches we'd need
                buffer = max(32, size // 8)
                effective_h = height - 2 * buffer
                effective_w = width - 2 * buffer
                
                if effective_h <= 0 or effective_w <= 0:
                    continue
                
                n_patches_h = max(1, math.ceil(effective_h / size))
                n_patches_w = max(1, math.ceil(effective_w / size))
                total_patches = n_patches_h * n_patches_w
                
                # Score based on: fewer patches preferred, but not too few
                # Sweet spot is between 4-16 patches for good parallelization and memory usage
                if total_patches < 4:
                    score = 100 + (4 - total_patches) * 10  # Penalty for too few patches
                elif total_patches > 16:
                    score = total_patches * 2  # Penalty for too many patches
                else:
                    score = total_patches  # Good range
                
                print(f"  Patch size {size}: {n_patches_h}x{n_patches_w} = {total_patches} patches, score: {score}")
                
                if score < best_score:
                    best_score = score
                    best_patch_size = size
            
            patch_size = best_patch_size
            print(f"Selected optimal patch size: {patch_size}")
        
        # Auto-determine buffer size if not provided
        if buffer_size is None:
            # Scale buffer size with patch size
            buffer_size = max(32, patch_size // 8)  # At least 32px or 1/8 of patch size
        
        # Ensure patch size doesn't exceed image dimensions
        patch_size = min(patch_size, min(height, width))
        
        print(f"Using patch_size={patch_size}, buffer_size={buffer_size}")
        
        # Check if image is too large and needs patching
        if height > patch_size or width > patch_size:
            print(f"Image size ({height}x{width}) exceeds patch size ({patch_size}), using patch-based inference")
            output_data = _run_patch_inference_buffered(
                image_tensor, model_net, device, patch_size, buffer_size, save_scores, band_order
            )
        else:
            print(f"Image size ({height}x{width}) fits in single patch")
            output_data = _run_single_inference(
                image_tensor, model_net, device, save_scores, band_order
            )

        print(f"Final output shape: {output_data.shape}")
        print(f"Unique prediction values: {np.unique(output_data)}")
         
        # Save output
        print(f"Saving output to: {out}")
        profile = dataset.profile.copy()
        out_channels = output_data.shape[0] if len(output_data.shape) == 3 else 1
        profile.update({
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
        })
        
        with rasterio.open(out, "w", **profile) as dst:
            if save_scores:
                dst.write(output_data)
            else:
                dst.write_colormap(1, {1: (255, 0, 0), 2: (0, 255, 0)})
                dst.colorinterp = [ColorInterp.palette]
                if len(output_data.shape) == 3:
                    dst.write(output_data[0], 1)
                else:
                    dst.write(output_data, 1)
        
        print(f"Inference completed successfully!")
        return True
        
    except Exception as e:
        print(f"Inference failed: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return False

def _prepare_image_batch(image_tensor, device, band_order=None):
    """Prepare image tensor for model input with comprehensive debugging.
    Args:
        image_tensor: Input tensor
        device: Target device
        band_order: Optional band reordering specification
                   - None: no reordering
                   - "bgr_to_rgb": swap bands 0 and 2 (for 4-band) or [0,2] and 
                    [4,6] (for 8-band)
                   - [0,1,2,3]: explicit band order for 4-band data
                   - [0,1,2,3,4,5,6,7]: explicit band order for 8-band data
    """    
    print(f"\n=== DEBUGGING _prepare_image_batch ===")
    print(f"Input tensor shape: {image_tensor.shape}")
    print(f"Input tensor range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
    print(f"Band order parameter: {band_order}")
    
    # Add batch dimension and move to device
    image_batch = image_tensor.unsqueeze(0).to(device)  # Shape: (1, C, H, W)
    
    print(f"Before any reordering:")
    for i in range(image_batch.shape[1]):
        band_data = image_batch[0, i]
        print(f"  Band {i}: mean={band_data.mean():.3f}, std={band_data.std():.3f}, min={band_data.min():.3f}, max={band_data.max():.3f}")
    
    # Apply band reordering if specified
    if band_order is not None:
        image_batch = _reorder_bands(image_batch, band_order)
        print(f"After band reordering:")
        for i in range(image_batch.shape[1]):
            band_data = image_batch[0, i]
            print(f"  Band {i}: mean={band_data.mean():.3f}, std={band_data.std():.3f}, min={band_data.min():.3f}, max={band_data.max():.3f}")
    
    # Check band count and reorder for temporal data if needed
    if image_batch.shape[1] == 8:
        # Temporal stacking: WinB then WinA reordering
        num_bands = image_batch.shape[1] // 2
        print(f"Before temporal reordering - WinB (0-3), WinA (4-7):")
        for i in range(8):
            band_data = image_batch[0, i]
            window = "WinB" if i < 4 else "WinA"
            band_in_window = i % 4
            print(f"  Band {i} ({window}-{band_in_window}): mean={band_data.mean():.3f}, std={band_data.std():.3f}")
        
        image_batch = torch.cat([image_batch[:, num_bands:], image_batch[:, :num_bands]], dim=1)
        print("Applied temporal reordering for 8-band input (now WinA then WinB)")
        
        print(f"After temporal reordering - WinA (0-3), WinB (4-7):")
        for i in range(8):
            band_data = image_batch[0, i]
            window = "WinA" if i < 4 else "WinB"
            band_in_window = i % 4
            print(f"  Band {i} ({window}-{band_in_window}): mean={band_data.mean():.3f}, std={band_data.std():.3f}")
            
    elif image_batch.shape[1] == 4:
        print("Using single window data (4 bands)")
    else:
        raise ValueError(f"Unexpected number of input bands: {image_batch.shape[1]}. Expected 4 or 8.")
    
    print(f"Final tensor shape: {image_batch.shape}")
    print(f"Final tensor range: [{image_batch.min():.3f}, {image_batch.max():.3f}]")
    print(f"Final tensor device: {image_batch.device}")
    print("=== END _prepare_image_batch DEBUG ===\n")
    
    return image_batch

def _reorder_bands(image_batch, band_order):
    """Reorder bands in the image batch.
    
    Args:
        image_batch: Tensor of shape (1, C, H, W)
        band_order: Band reordering specification
    
    Returns:
        Reordered tensor
    """
    if band_order == "bgr_to_rgb":
        if image_batch.shape[1] == 4:
            # For 4-band BGR-NIR -> RGB-NIR: swap bands 0 and 2
            image_batch[:, [0, 2]] = image_batch[:, [2, 0]]
            print("Applied BGR->RGB conversion: swapped bands 0 and 2")
        elif image_batch.shape[1] == 8:
            # For 8-band temporal BGR-NIR: swap in both windows
            image_batch[:, [0, 2]] = image_batch[:, [2, 0]]  # First window
            image_batch[:, [4, 6]] = image_batch[:, [6, 4]]  # Second window
            print("Applied BGR->RGB conversion for both temporal windows")
        else:
            print(f"Warning: BGR->RGB conversion not implemented for {image_batch.shape[1]} bands")
    
    elif isinstance(band_order, (list, tuple)):
        # Explicit band reordering
        expected_bands = image_batch.shape[1]
        if len(band_order) != expected_bands:
            raise ValueError(f"Band order list length ({len(band_order)}) doesn't match number of bands ({expected_bands})")
        
        # Check if all indices are valid
        if not all(0 <= idx < expected_bands for idx in band_order):
            raise ValueError(f"Invalid band indices in {band_order} for {expected_bands} bands")
        
        # Reorder bands
        image_batch = image_batch[:, band_order]
        print(f"Applied custom band reordering: {band_order}")
    
    else:
        raise ValueError(f"Unknown band_order: {band_order}")
    
    return image_batch

def _run_single_inference(image_tensor, model_net, device, save_scores, band_order=None):
    """Run inference on a single image that fits in memory."""
    image_batch = _prepare_image_batch(image_tensor, device, band_order)
    
    print("Running single-patch inference...")
    with torch.inference_mode():
        predictions = model_net(image_batch)
        print(f"Model output shape: {predictions.shape}")
        print(f"Model output range: [{predictions.min():.2f}, {predictions.max():.2f}]")
        
        if save_scores:
            # Save probability scores
            predictions = F.softmax(predictions, dim=1)
            predictions = predictions.cpu().numpy().astype(np.float32)
            predictions = (predictions * 255).clip(0, 255).astype(np.uint8)
            output_data = predictions[0]  # Remove batch dimension: (C, H, W)
        else:
            # Save class predictions
            predictions = predictions.argmax(dim=1)  # Shape: (1, H, W)
            output_data = predictions[0].cpu().numpy().astype(np.uint8)  # Shape: (H, W)
            output_data = output_data[np.newaxis, ...]  # Add channel dimension: (1, H, W)
    
    return output_data

def _run_patch_inference_buffered(image_tensor, model_net, device, patch_size, 
                                  buffer_size, save_scores, band_order=None):
    """Run inference with extensive debugging."""
    channels, height, width = image_tensor.shape
    print(f"\n=== DEBUGGING PATCH INFERENCE ===")
    print(f"Input image shape: {image_tensor.shape}")
    print(f"Input image range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
    print(f"Device: {device}")
    print(f"Patch size: {patch_size}, Buffer size: {buffer_size}")
    
    # Test center crop BEFORE any processing
    print("\n--- Testing center crop BEFORE any processing ---")
    center_h_start = height // 4
    center_h_end = center_h_start + min(patch_size, height - center_h_start)
    center_w_start = width // 4  
    center_w_end = center_w_start + min(patch_size, width - center_w_start)
    
    center_crop = image_tensor[:, center_h_start:center_h_end, center_w_start:center_w_end]
    print(f"Center crop shape: {center_crop.shape}")
    print(f"Center crop range: [{center_crop.min():.3f}, {center_crop.max():.3f}]")
    
    # Test the center crop through the full pipeline
    print("\n--- Testing center crop through _prepare_image_batch ---")
    try:
        center_batch = _prepare_image_batch(center_crop, device, band_order)
        print(f"Prepared batch shape: {center_batch.shape}")
        print(f"Prepared batch range: [{center_batch.min():.3f}, {center_batch.max():.3f}]")
        
        print("\n--- Testing model inference on center crop ---")
        with torch.inference_mode():
            center_pred = model_net(center_batch)
            print(f"Model output shape: {center_pred.shape}")
            print(f"Model output range: [{center_pred.min():.3f}, {center_pred.max():.3f}]")
            
            # Check logits per class
            for c in range(center_pred.shape[1]):
                class_logits = center_pred[0, c]
                print(f"  Class {c} logits: min={class_logits.min():.2f}, max={class_logits.max():.2f}, mean={class_logits.mean():.2f}")
            
            # Check final predictions
            center_argmax = center_pred.argmax(dim=1)
            center_unique = torch.unique(center_argmax, return_counts=True)
            total_pixels = center_argmax.numel()
            print(f"Center crop predictions:")
            for val, count in zip(center_unique[0], center_unique[1]):
                percentage = (count.item() / total_pixels) * 100
                print(f"  Class {val.item()}: {count.item()} pixels ({percentage:.1f}%)")
                
            # If still getting all one class, there might be a fundamental issue
            if len(center_unique[0]) == 1:
                print("\n⚠️  STILL GETTING SINGLE CLASS PREDICTION!")
                print("Let's test with completely different inputs...")
                
                # Test 1: Random noise
                print("\n--- Test 1: Random noise input ---")
                random_input = torch.randn_like(center_batch) * 0.5
                with torch.inference_mode():
                    random_pred = model_net(random_input)
                    random_argmax = random_pred.argmax(dim=1)
                    random_unique = torch.unique(random_argmax, return_counts=True)
                    print(f"Random input predictions: Classes {random_unique[0].tolist()}, Counts {random_unique[1].tolist()}")
                
                # Test 2: All zeros
                print("\n--- Test 2: All zeros input ---")
                zero_input = torch.zeros_like(center_batch)
                with torch.inference_mode():
                    zero_pred = model_net(zero_input)
                    zero_argmax = zero_pred.argmax(dim=1)
                    zero_unique = torch.unique(zero_argmax, return_counts=True)
                    print(f"Zero input predictions: Classes {zero_unique[0].tolist()}, Counts {zero_unique[1].tolist()}")
                
                # Test 3: All ones
                print("\n--- Test 3: All ones input ---")
                ones_input = torch.ones_like(center_batch)
                with torch.inference_mode():
                    ones_pred = model_net(ones_input)
                    ones_argmax = ones_pred.argmax(dim=1)
                    ones_unique = torch.unique(ones_argmax, return_counts=True)
                    print(f"Ones input predictions: Classes {ones_unique[0].tolist()}, Counts {ones_unique[1].tolist()}")
                
                # Test 4: Different value ranges
                print("\n--- Test 4: Different value ranges ---")
                for scale in [-2, -1, 0.5, 1, 2, 5]:
                    test_input = torch.ones_like(center_batch) * scale
                    with torch.inference_mode():
                        test_pred = model_net(test_input)
                        test_argmax = test_pred.argmax(dim=1)
                        test_unique = torch.unique(test_argmax, return_counts=True)
                        print(f"  Scale {scale}: Classes {test_unique[0].tolist()}")
                        
    except Exception as e:
        print(f"ERROR in center crop test: {e}")
        import traceback
        print(traceback.format_exc())
        
    print("=== END PATCH INFERENCE DEBUG ===\n")

def _calculate_patch_grid(image_height, image_width, patch_size, buffer_size=32):
    """Calculate optimal patch grid with edge buffer and equal overlaps."""
    
    # Calculate the effective area after applying buffer
    effective_height = image_height - 2 * buffer_size
    effective_width = image_width - 2 * buffer_size
    
    # Calculate number of patches needed to cover the effective area
    n_patches_h = math.ceil(effective_height / patch_size)
    n_patches_w = math.ceil(effective_width / patch_size)
    
    # Ensure at least 1 patch in each dimension
    n_patches_h = max(1, n_patches_h)
    n_patches_w = max(1, n_patches_w)
    
    # Calculate the actual stride to distribute patches evenly
    if n_patches_h > 1:
        stride_h = (effective_height - patch_size) / (n_patches_h - 1)
    else:
        stride_h = 0
        
    if n_patches_w > 1:
        stride_w = (effective_width - patch_size) / (n_patches_w - 1)
    else:
        stride_w = 0
    
    # Calculate overlaps
    overlap_h = patch_size - stride_h if stride_h > 0 else 0
    overlap_w = patch_size - stride_w if stride_w > 0 else 0
    
    print(f"Patch grid calculation:")
    print(f"  Image size: {image_height}x{image_width}")
    print(f"  Buffer size: {buffer_size}")
    print(f"  Effective area: {effective_height}x{effective_width}")
    print(f"  Patches: {n_patches_h}x{n_patches_w}")
    print(f"  Patch size: {patch_size}x{patch_size}")
    print(f"  Stride: {stride_h:.1f}x{stride_w:.1f}")
    print(f"  Overlap: {overlap_h:.1f}x{overlap_w:.1f}")
    
    return n_patches_h, n_patches_w, stride_h, stride_w, overlap_h, overlap_w

def _create_buffered_image(image_tensor, buffer_size):
    """Add reflection padding buffer around the image."""
    if buffer_size <= 0:
        return image_tensor, 0, 0
    
    # Apply reflection padding
    buffered_image = F.pad(
        image_tensor, 
        (buffer_size, buffer_size, buffer_size, buffer_size), 
        mode='reflect'
    )
    
    print(f"Added buffer: {image_tensor.shape} -> {buffered_image.shape}")
    return buffered_image, buffer_size, buffer_size

def debug_normalization_step(image_array, strategy, procedure, global_stats, clip_val):
    """Debug the normalization process step by step."""
    print(f"\n=== DEBUGGING NORMALIZATION ===")
    print(f"Input shape: {image_array.shape}")
    print(f"Input dtype: {image_array.dtype}")
    print(f"Input range: [{image_array.min():.3f}, {image_array.max():.3f}]")
    print(f"Strategy: {strategy}, Procedure: {procedure}")
    print(f"Global stats: {global_stats}")
    print(f"Clip value: {clip_val}")
    
    # Check per-band statistics before normalization
    print("Per-band stats BEFORE normalization:")
    for i in range(image_array.shape[0]):
        band = image_array[i]
        print(f"  Band {i}: min={band.min():.3f}, max={band.max():.3f}, mean={band.mean():.3f}, std={band.std():.3f}")
    
    # Apply normalization and track changes
    normalized = normalize_image(
        image_array, 
        strategy=strategy, 
        procedure=procedure, 
        global_stats=global_stats, 
        img_clip_val=clip_val
    )
    
    print("Per-band stats AFTER normalization:")
    for i in range(normalized.shape[0]):
        band = normalized[i]
        print(f"  Band {i}: min={band.min():.3f}, max={band.max():.3f}, mean={band.mean():.3f}, std={band.std():.3f}")
    
    print(f"Output range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    print("=== END NORMALIZATION DEBUG ===\n")
    
    return normalized