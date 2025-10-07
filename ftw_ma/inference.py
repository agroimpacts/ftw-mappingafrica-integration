import math
import os
import time
from typing import (
    Literal, Optional, Union, Dict, Any, Tuple, List
)

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from rasterio.enums import ColorInterp
from rasterio.mask import mask
from rasterio.features import geometry_mask
from pathlib import Path
import geopandas as gpd
from shapely.geometry import mapping

from .dataset import SimpleRasterDataset
from .trainers import CustomSemanticSegmentationTask

def load_model(model_path, device):
    """Load model once and return it for reuse."""
    task = CustomSemanticSegmentationTask.load_from_checkpoint(
        model_path, map_location="cpu"
    )
    task.freeze()
    model_net = task.model.eval().to(device)
    print(f"Model loaded and moved to {device}")
    return model_net

def crop_to_polygon(image_array, raster_profile, geometry, nodata_value=0):
    """
    Crop image array to polygon boundary.
    
    Args:
        image_array: numpy array (C, H, W) or (H, W)
        raster_profile: rasterio profile dict
        geometry: shapely geometry object
        nodata_value: value to use for pixels outside polygon
        
    Returns:
        cropped_array: cropped image array
        cropped_profile: updated profile for cropped image
    """
    from rasterio.io import MemoryFile
    
    # print(f"DEBUG: Input image_array shape: {image_array.shape}")
    # print(f"DEBUG: Geometry type: {type(geometry)}")
    # print(f"DEBUG: Geometry bounds: {geometry.bounds}")
    # print(f"DEBUG: Raster profile CRS: {raster_profile.get('crs')}")
    # print(f"DEBUG: Raster transform: {raster_profile.get('transform')}")
    
    # Handle both (H, W) and (C, H, W) arrays
    if len(image_array.shape) == 2:
        # Single band case
        temp_array = image_array[np.newaxis, :, :]  # Add channel dimension
        single_band = True
    else:
        temp_array = image_array
        single_band = False
    
    # print(f"DEBUG: Temp array shape for processing: {temp_array.shape}")
    
    # Create temporary raster in memory
    temp_profile = raster_profile.copy()
    temp_profile.update({
        'count': temp_array.shape[0],
        'dtype': temp_array.dtype,
        'nodata': nodata_value
    })
    
    # print(f"DEBUG: Temp profile: {temp_profile}")
    
    with MemoryFile() as memfile:
        with memfile.open(**temp_profile) as dataset:
            dataset.write(temp_array)
            
            # print(f"DEBUG: Dataset bounds: {dataset.bounds}")
            # print(f"DEBUG: Dataset CRS: {dataset.crs}")
            
            # Check if geometry intersects with raster bounds
            from shapely.geometry import box
            raster_bbox = box(*dataset.bounds)
            
            if not geometry.intersects(raster_bbox):
                print("WARNING: Geometry does not intersect with raster bounds!")
                print(f"  Geometry bounds: {geometry.bounds}")
                print(f"  Raster bounds: {dataset.bounds}")
                return image_array, raster_profile
            
            # Crop to polygon
            try:
                # print(f"DEBUG: Attempting to mask with geometry...")
                cropped_data, cropped_transform = mask(
                    dataset, 
                    [mapping(geometry)], 
                    crop=True, 
                    nodata=nodata_value,
                    filled=True
                )
                
                # print(f"DEBUG: Cropped data shape: {cropped_data.shape}")
                # print(f"DEBUG: Cropped transform: {cropped_transform}")
                
                # Update profile for cropped image
                cropped_profile = temp_profile.copy()
                cropped_profile.update({
                    'height': cropped_data.shape[1],
                    'width': cropped_data.shape[2], 
                    'transform': cropped_transform
                })
                
                # Return to original shape format
                if single_band:
                    cropped_data = cropped_data[0]  # Remove channel dimension
                    cropped_profile['count'] = 1
                
                # print(f"DEBUG: Final cropped data shape: {cropped_data.shape}")
                # print(f"DEBUG: Cropping successful!")
                
                return cropped_data, cropped_profile
                
            except Exception as e:
                print(f"ERROR: Failed to crop to polygon: {e}")
                print(f"Full error: {repr(e)}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
                print("Returning original image without cropping")
                return image_array, raster_profile

def inference_run_single(
    input_file,
    model_net,
    device,
    out,
    save_scores=False,
    normalization_strategy="min_max",
    normalization_stat_procedure="lab",
    global_stats=None,
    img_clip_val=0,
    nodata=None,
    overwrite=False,
    band_order=None,
    crop_geometry=None,
):
    """Run inference on a single file with pre-loaded model."""
    
    try:
        assert os.path.exists(input_file), (
            f"Input file {input_file} does not exist."
        )
        assert overwrite or not os.path.exists(out), (
            f"Output file {out} already exists."
        )
        print(f"Loading and normalizing image: {input_file}")
        dataset = SimpleRasterDataset(
            file_path=input_file,
            normalization_strategy=normalization_strategy,
            normalization_stat_procedure=normalization_stat_procedure,
            global_stats=global_stats,
            img_clip_val=img_clip_val,
            nodata=nodata,
        )
        image_tensor = dataset.get_full_image()
        
        # Run inference
        output_data = _run_single_inference(image_tensor, model_net, device, 
                                            save_scores, band_order)
        
        # Crop to polygon if geometry provided
        output_profile = dataset.profile.copy()
        if crop_geometry is not None:
            geom = None
            geom_crs = None
            
            if hasattr(crop_geometry, 'iloc') and hasattr(crop_geometry, 'crs'):
                if len(crop_geometry) > 0:
                    geom = crop_geometry.iloc[0]
                    geom_crs = crop_geometry.crs
                    # print(f"DEBUG: Extracted geometry from GeoSeries")
                    # print(f"DEBUG: GeoSeries CRS: {geom_crs}")
                else:
                    print("ERROR: Empty GeoSeries provided")
                    
            elif hasattr(crop_geometry, 'bounds') \
                and hasattr(crop_geometry, 'geom_type'):
                geom = crop_geometry
                
            elif hasattr(crop_geometry, 'geometry'):
                geom = crop_geometry.geometry
                if hasattr(crop_geometry, 'crs'):
                    geom_crs = crop_geometry.crs
                # print(f"DEBUG: Extracted geometry from .geometry attribute")
            
            if geom is None:
                print(f"ERROR: Could not extract valid geometry "\
                      f"from crop_geometry")
                print(f"DEBUG: crop_geometry type: {type(crop_geometry)}")
                if hasattr(crop_geometry, 'shape'):
                    print(f"DEBUG: crop_geometry shape: {crop_geometry.shape}")
                print("Skipping cropping...")
            else:
                # print(f"DEBUG: Final geometry type: {geom.geom_type}")
                # print(f"DEBUG: Final geometry bounds: {geom.bounds}")
                
                dataset_crs = dataset.crs
                # print(f"DEBUG: Dataset CRS: {dataset_crs}")
                # print(f"DEBUG: Geometry CRS: {geom_crs}")
                
                if geom_crs is not None and geom_crs != dataset_crs:
                    print(f"Reprojecting geometry from {geom_crs} to {dataset_crs}")
                    import geopandas as gpd
                    temp_gdf = gpd.GeoDataFrame([1], geometry=[geom], crs=geom_crs)
                    temp_gdf = temp_gdf.to_crs(dataset_crs)
                    geom = temp_gdf.geometry.iloc[0]
                
                cropped_output, output_profile = crop_to_polygon(
                    output_data, dataset.profile, geom, nodata_value=0
                )
                
                if cropped_output.shape != output_data.shape:
                    output_data = cropped_output
                    print(f"✓ Cropped output shape: {output_data.shape}")
                else:
                    print("⚠ Warning: Output shape unchanged - cropping may not have worked")
         
        print(f"Saving Cloud Optimized GeoTIFF to: {out}")
        out_channels = (
            output_data.shape[0]
            if len(output_data.shape) == 3 else 1
        )
        
        # Configure COG profile
        output_profile.update({
            "driver": "GTiff",
            "count": out_channels,
            "dtype": "uint8",
            "compress": "deflate",  # Better compression for COGs
            "predictor": 2,
            "blockxsize": 512,
            "blockysize": 512,
            "tiled": True,
            "interleave": "pixel",
            "bigtiff": "yes",  # Support large files
            "SPARSE_OK": True,  # Optimize for sparse data
        })
        
        # Only set nodata for class predictions, not probability scores
        if not save_scores:
            output_profile["nodata"] = 0
        
        with rasterio.open(out, "w", **output_profile) as dst:
            if save_scores:
                dst.write(output_data)
                for i in range(out_channels):
                    dst.set_band_description(
                        i + 1, f"Class_{i}_probability"
                    )
                # Don't set nodata or colormap for probability scores
            else:
                dst.write_colormap(1, {1: (255, 0, 0), 2: (0, 255, 0)})
                dst.colorinterp = [ColorInterp.palette]
                if len(output_data.shape) == 3:
                    dst.write(output_data[0], 1)
                else:
                    dst.write(output_data, 1)
            
            # Build overviews for COGs
            print("Building overviews for COG...")
            overview_factors = [2, 4, 8, 16, 32, 64]
            dst.build_overviews(overview_factors, 
                                rasterio.enums.Resampling.nearest)
            
        # Additional COG optimization using rio-cogeo if available
        try:
            from rio_cogeo.cogeo import cog_translate
            from rio_cogeo.profiles import cog_profiles
            
            # Create temporary file for COG optimization
            temp_output = out.replace('.tif', '_temp.tif')
            
            # Move original to temp
            import shutil
            shutil.move(out, temp_output)
            
            # Optimize as COG
            cog_profile = cog_profiles.get("deflate")
            cog_profile.update({
                "BLOCKXSIZE": 512,
                "BLOCKYSIZE": 512,
                "OVERVIEW_RESAMPLING": "nearest",
            })
            
            with rasterio.open(temp_output) as src_dataset:
                cog_translate(
                    src_dataset,
                    out,
                    cog_profile,
                    in_memory=False,
                    quiet=True,
                )
            
            # Remove temp file
            os.remove(temp_output)
            print("✓ COG optimization completed using rio-cogeo")
            
        except ImportError:
            print("⚠ rio-cogeo not available, using basic COG settings")
        except Exception as e:
            print(f"⚠ COG optimization failed: {e}, using basic COG settings")
            # If COG optimization fails, rename temp back to original
            if os.path.exists(temp_output):
                shutil.move(temp_output, out)
        
        print(f"Cloud Optimized GeoTIFF created successfully!")
        return True
    except Exception as e:
        print(f"Inference failed: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return False

def _prepare_image_batch(
    image_tensor, device, band_order=None
):
    """Prepare image tensor for model input."""
    image_batch = image_tensor.unsqueeze(0).to(device)
    if band_order is not None:
        image_batch = _reorder_bands(image_batch, band_order)
    if image_batch.shape[1] == 8:
        num_bands = image_batch.shape[1] // 2
        image_batch = torch.cat(
            [image_batch[:, num_bands:], image_batch[:, :num_bands]], dim=1
        )
        print("Applied temporal reordering for 8-band input")
    elif image_batch.shape[1] == 4:
        print("Using single window data (4 bands)")
    else:
        raise ValueError(
            f"Unexpected number of input bands: {image_batch.shape[1]}. "
            f"Expected 4 or 8."
        )
    return image_batch

def _reorder_bands(image_batch, band_order):
    """Reorder bands in the image batch."""
    if band_order == "bgr_to_rgb":
        if image_batch.shape[1] == 4:
            image_batch[:, [0, 2]] = image_batch[:, [2, 0]]
            print("Applied BGR->RGB conversion: swapped bands 0 and 2")
        elif image_batch.shape[1] == 8:
            image_batch[:, [0, 2]] = image_batch[:, [2, 0]]
            image_batch[:, [4, 6]] = image_batch[:, [6, 4]]
            print("Applied BGR->RGB conversion for both temporal windows")
        else:
            print(
                f"Warning: BGR->RGB conversion not implemented for "
                f"{image_batch.shape[1]} bands"
            )
    elif isinstance(band_order, (list, tuple)):
        expected_bands = image_batch.shape[1]
        if len(band_order) != expected_bands:
            raise ValueError(
                f"Band order list length ({len(band_order)}) doesn't match "
                f"number of bands ({expected_bands})"
            )
        if not all(0 <= idx < expected_bands for idx in band_order):
            raise ValueError(
                f"Invalid band indices in {band_order} for "
                f"{expected_bands} bands"
            )
        image_batch = image_batch[:, band_order]
        print(f"Applied custom band reordering: {band_order}")
    else:
        raise ValueError(f"Unknown band_order: {band_order}")
    return image_batch

def _run_single_inference(
    image_tensor, model_net, device, save_scores, band_order=None
):
    """Run inference on a single image that fits in memory."""
    image_batch = _prepare_image_batch(
        image_tensor, device, band_order
    )
    print("Running inference...")
    with torch.inference_mode():
        predictions = model_net(image_batch)
        print(f"Model output shape: {predictions.shape}")
        print(
            f"Model output range: "
            f"[{predictions.min():.2f}, {predictions.max():.2f}]"
        )
        if save_scores:
            predictions = F.softmax(predictions, dim=1)
            predictions = predictions.cpu().numpy().astype(np.float32)
            predictions = (predictions * 255).clip(0, 255).astype(np.uint8)
            output_data = predictions[0]
        else:
            predictions = predictions.argmax(dim=1)
            print(f"Unique prediction values: {torch.unique(predictions)}")
            output_data = (
                predictions[0].cpu().numpy().astype(np.uint8)
            )
            output_data = output_data[np.newaxis, ...]
    return output_data
