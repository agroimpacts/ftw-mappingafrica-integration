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
from affine import Affine
from rasterio.enums import ColorInterp
from rasterio.mask import mask
from rasterio.features import geometry_mask
from pathlib import Path
import geopandas as gpd
from shapely.geometry import mapping

from .dataset import SimpleRasterDataset
from .trainers import CustomSemanticSegmentationTask
from .quad_index import QuadIndex
from .on_the_fly_imagery import (
    build_tile_in_memory,
    validate_tile_coverage,
)

def load_model(model_path, device):
    """Load model once and return it for reuse."""
    task = CustomSemanticSegmentationTask.load_from_checkpoint(
        model_path,
        map_location="cpu"
    )

    task.freeze()

    model_net = task.model.eval().to(device)

    print(f"Model loaded and moved to {device}")

    return model_net


def crop_to_polygon(
    image_array,
    raster_profile,
    geometry,
    nodata_value=0
):
    """
    Crop image array to polygon boundary.

    Args:
        image_array:
            numpy array (C, H, W) or (H, W)

        raster_profile:
            rasterio profile dict

        geometry:
            shapely geometry object

        nodata_value:
            value to use for pixels outside polygon

    Returns:
        cropped_array:
            cropped image array

        cropped_profile:
            updated profile for cropped image
    """

    from rasterio.io import MemoryFile

    # Handle both (H, W) and (C, H, W) arrays
    if len(image_array.shape) == 2:
        temp_array = image_array[np.newaxis, :, :]
        single_band = True
    else:
        temp_array = image_array
        single_band = False

    # Create temporary raster in memory
    temp_profile = raster_profile.copy()

    temp_profile.update({
        "count": temp_array.shape[0],
        "dtype": temp_array.dtype,
        "nodata": nodata_value
    })

    with MemoryFile() as memfile:

        with memfile.open(**temp_profile) as dataset:

            dataset.write(temp_array)

            # Check geometry against raster bounds
            from shapely.geometry import box

            raster_bbox = box(*dataset.bounds)

            if not geometry.intersects(raster_bbox):

                print(
                    "WARNING: Geometry does not intersect "
                    "with raster bounds!"
                )

                print(
                    f"  Geometry bounds: "
                    f"{geometry.bounds}"
                )

                print(
                    f"  Raster bounds: "
                    f"{dataset.bounds}"
                )

                return image_array, raster_profile

            try:

                cropped_data, cropped_transform = mask(
                    dataset,
                    [mapping(geometry)],
                    crop=True,
                    nodata=nodata_value,
                    filled=True
                )

                cropped_profile = temp_profile.copy()

                cropped_profile.update({
                    "height": cropped_data.shape[1],
                    "width": cropped_data.shape[2],
                    "transform": cropped_transform
                })

                if single_band:

                    cropped_data = cropped_data[0]

                    cropped_profile["count"] = 1

                return cropped_data, cropped_profile

            except Exception as e:

                print(
                    f"ERROR: Failed to crop "
                    f"to polygon: {e}"
                )

                print(
                    f"Full error: {repr(e)}"
                )

                import traceback

                print(
                    f"Traceback: "
                    f"{traceback.format_exc()}"
                )

                print(
                    "Returning original image "
                    "without cropping"
                )

                return image_array, raster_profile



def _crop_otf_prediction(output_data, raster_profile):
    """Remove the 24-pixel OTF buffer and update georeferencing."""

    buffer_pixels = 24
    inference_size = 2000
    expected_size = inference_size + 2 * buffer_pixels

    height, width = output_data.shape[-2:]

    if (height, width) != (expected_size, expected_size):
        raise ValueError(
            f"Expected on-the-fly prediction size "
            f"{expected_size}x{expected_size}, "
            f"got {height}x{width}"
        )

    row_start = col_start = buffer_pixels
    row_end = col_end = buffer_pixels + inference_size

    cropped = output_data[
        :,
        row_start:row_end,
        col_start:col_end,
    ]

    profile = raster_profile.copy()
    profile["transform"] = (
        raster_profile["transform"]
        * Affine.translation(col_start, row_start)
    )
    profile["height"] = inference_size
    profile["width"] = inference_size

    return cropped, profile


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
    image_data=None,
    image_profile=None,
    imagery_mode="pre_tiled",
    tile_geometry=None,
    year=None,
    target_country=None,
    quad_index=None,
):
    """
    Run inference on a single image with a pre-loaded model.

    Supported imagery modes
    -----------------------

    pre_tiled
        Read an existing raster using input_file.

    on_the_fly
        Construct a 2358 x 2358 buffered image directly
        from Planet/NICFI source quads in memory.

    An already-created in-memory image can also be supplied
    directly using image_data + image_profile.
    """

    try:

        assert overwrite or not os.path.exists(out), (
            f"Output file {out} already exists."
        )

        # =============================================================
        # INPUT MODE 1:
        # Build imagery directly from source quads
        # =============================================================

        if imagery_mode == "on_the_fly":

            if tile_geometry is None:

                raise ValueError(
                    "tile_geometry is required "
                    "for on-the-fly imagery."
                )

            if year is None:

                raise ValueError(
                    "year is required "
                    "for on-the-fly imagery."
                )

            if target_country is None:

                raise ValueError(
                    "target_country is required "
                    "for on-the-fly imagery."
                )

            if quad_index is None:

                raise ValueError(
                    "quad_index is required "
                    "for on-the-fly imagery."
                )

            if input_file is not None:

                raise ValueError(
                    "input_file must be None when "
                    "imagery_mode='on_the_fly'."
                )

            if image_data is not None:

                raise ValueError(
                    "image_data must not be supplied when "
                    "imagery_mode='on_the_fly'."
                )

            print(
                f"Building imagery on the fly: "
                f"{target_country}, {year}"
            )

            image_data, image_profile = build_tile_in_memory(
                tile_geometry=tile_geometry,
                year=year,
                target_country=target_country,
                quad_index=quad_index,
            )

            print(
                f"Built in-memory image: "
                f"shape={image_data.shape}, "
                f"dtype={image_data.dtype}"
            )

            coverage = validate_tile_coverage(
                image_data
            )

            print(
                f"Imagery coverage: "
                f"{coverage['coverage_percent']:.6f}% "
                f"({coverage['missing_pixels']:,} "
                f"missing pixels)"
            )

            if coverage["valid_pixels"] == 0:
                raise RuntimeError(
                    "On-the-fly imagery contains no valid pixels: "
                    f"{coverage['missing_pixels']:,} "
                    f"of {coverage['total_pixels']:,} "
                    "pixels are nodata. Skipping inference."
                )

            elif not coverage["complete"]:
                print(
                    "WARNING: Incomplete on-the-fly imagery: "
                    f"{coverage['missing_pixels']:,} "
                    f"of {coverage['total_pixels']:,} pixels are nodata "
                    f"({coverage['coverage_percent']:.6f}% coverage). "
                    "Proceeding with inference."
                )

            dataset = SimpleRasterDataset(
                image_data=image_data,
                profile=image_profile,
                normalization_strategy=normalization_strategy,
                normalization_stat_procedure=(
                    normalization_stat_procedure
                ),
                global_stats=global_stats,
                img_clip_val=img_clip_val,
                nodata=nodata,
            )

        # =============================================================
        # INPUT MODE 2:
        # Explicit in-memory image
        # =============================================================

        elif image_data is not None:

            if imagery_mode != "pre_tiled":

                raise ValueError(
                    f"image_data cannot be supplied with "
                    f"imagery_mode='{imagery_mode}'."
                )

            if input_file is not None:

                raise ValueError(
                    "Provide either input_file "
                    "or image_data, not both."
                )

            if image_profile is None:

                raise ValueError(
                    "image_profile is required "
                    "when image_data is provided."
                )

            print(
                "Loading and normalizing "
                "in-memory image"
            )

            dataset = SimpleRasterDataset(
                image_data=image_data,
                profile=image_profile,
                normalization_strategy=normalization_strategy,
                normalization_stat_procedure=(
                    normalization_stat_procedure
                ),
                global_stats=global_stats,
                img_clip_val=img_clip_val,
                nodata=nodata,
            )

        # =============================================================
        # INPUT MODE 3:
        # Existing pre-tiled raster
        # =============================================================

        elif imagery_mode == "pre_tiled":

            if input_file is None:

                raise ValueError(
                    "input_file is required when "
                    "imagery_mode='pre_tiled'."
                )

            assert os.path.exists(input_file), (
                f"Input file {input_file} "
                f"does not exist."
            )

            print(
                f"Loading and normalizing image: "
                f"{input_file}"
            )

            dataset = SimpleRasterDataset(
                file_path=input_file,
                normalization_strategy=normalization_strategy,
                normalization_stat_procedure=(
                    normalization_stat_procedure
                ),
                global_stats=global_stats,
                img_clip_val=img_clip_val,
                nodata=nodata,
            )

        else:

            raise ValueError(
                f"Unknown imagery_mode: "
                f"'{imagery_mode}'. "
                f"Expected 'pre_tiled' "
                f"or 'on_the_fly'."
            )

        # =============================================================
        # Get normalized tensor
        # =============================================================

        image_tensor = dataset.get_full_image()

        # =============================================================
        # Run model inference
        # =============================================================

        output_data = _run_single_inference(
            image_tensor,
            model_net,
            device,
            save_scores,
            band_order,
        )

        # =============================================================
        # Prepare output profile
        # =============================================================

        output_profile = dataset.profile.copy()

        # =============================================================
        # ON-THE-FLY CROP
        #
        # Mapping Africa imagery:
        #
        #   2048 pixels
        #   - 24 left
        #   - 24 right
        #   = 2000 inference pixels
        #
        # Crop using exact pixel indices rather than geographic
        # polygon masking. This avoids floating-point window
        # rounding differences.
        # =============================================================

        if imagery_mode == "on_the_fly":

            output_data, output_profile = _crop_otf_prediction(
                output_data,
                dataset.profile,
            )

            print(
                f"✓ Pixel-cropped on-the-fly "
                f"output shape: {output_data.shape}"
            )

        # =============================================================
        # EXISTING PRE-TILED POLYGON CROP
        #
        # Keep existing behavior unchanged for old workflow.
        # =============================================================

        elif crop_geometry is not None:

            geom = None
            geom_crs = None

            if (
                hasattr(crop_geometry, "iloc")
                and hasattr(crop_geometry, "crs")
            ):

                if len(crop_geometry) > 0:

                    geom = (
                        crop_geometry.iloc[0]
                    )

                    geom_crs = (
                        crop_geometry.crs
                    )

                else:

                    print(
                        "ERROR: Empty GeoSeries "
                        "provided"
                    )

            elif (
                hasattr(crop_geometry, "bounds")
                and hasattr(
                    crop_geometry,
                    "geom_type"
                )
            ):

                geom = crop_geometry

            elif hasattr(
                crop_geometry,
                "geometry"
            ):

                geom = crop_geometry.geometry

                if hasattr(
                    crop_geometry,
                    "crs"
                ):

                    geom_crs = (
                        crop_geometry.crs
                    )

            if geom is None:

                print(
                    "ERROR: Could not extract "
                    "valid geometry from "
                    "crop_geometry"
                )

                print(
                    f"DEBUG: crop_geometry type: "
                    f"{type(crop_geometry)}"
                )

                if hasattr(
                    crop_geometry,
                    "shape"
                ):

                    print(
                        f"DEBUG: crop_geometry shape: "
                        f"{crop_geometry.shape}"
                    )

                print(
                    "Skipping cropping..."
                )

            else:

                dataset_crs = dataset.crs

                if (
                    geom_crs is not None
                    and geom_crs != dataset_crs
                ):

                    print(
                        f"Reprojecting geometry "
                        f"from {geom_crs} "
                        f"to {dataset_crs}"
                    )

                    temp_gdf = (
                        gpd.GeoDataFrame(
                            [1],
                            geometry=[geom],
                            crs=geom_crs
                        )
                    )

                    temp_gdf = (
                        temp_gdf.to_crs(
                            dataset_crs
                        )
                    )

                    geom = (
                        temp_gdf
                        .geometry
                        .iloc[0]
                    )

                (
                    cropped_output,
                    output_profile
                ) = crop_to_polygon(
                    output_data,
                    dataset.profile,
                    geom,
                    nodata_value=0
                )

                if (
                    cropped_output.shape
                    != output_data.shape
                ):

                    output_data = (
                        cropped_output
                    )

                    print(
                        f"✓ Cropped output shape: "
                        f"{output_data.shape}"
                    )

                else:

                    print(
                        "⚠ Warning: Output shape "
                        "unchanged - cropping may "
                        "not have worked"
                    )

        # =============================================================
        # SAVE OUTPUT
        # =============================================================

        print(
            f"Saving Cloud Optimized "
            f"GeoTIFF to: {out}"
        )

        out_channels = (
            output_data.shape[0]
            if len(output_data.shape) == 3
            else 1
        )

        output_profile.update({
            "driver": "GTiff",
            "count": out_channels,
            "dtype": "uint8",
            "compress": "deflate",
            "predictor": 2,
            "blockxsize": 512,
            "blockysize": 512,
            "tiled": True,
            "interleave": "pixel",
            "bigtiff": "yes",
            "SPARSE_OK": True,
        })

        # Class prediction output
        if not save_scores:

            output_profile["nodata"] = 0

        # =============================================================
        # WRITE GEOTIFF
        # =============================================================

        with rasterio.open(
            out,
            "w",
            **output_profile
        ) as dst:

            if save_scores:

                dst.write(
                    output_data
                )

                for i in range(
                    out_channels
                ):

                    dst.set_band_description(
                        i + 1,
                        f"Class_{i}_probability"
                    )

            else:

                dst.write_colormap(
                    1,
                    {
                        1: (255, 0, 0),
                        2: (0, 255, 0)
                    }
                )

                dst.colorinterp = [
                    ColorInterp.palette
                ]

                if (
                    len(output_data.shape)
                    == 3
                ):

                    dst.write(
                        output_data[0],
                        1
                    )

                else:

                    dst.write(
                        output_data,
                        1
                    )

            # Build overviews
            print(
                "Building overviews "
                "for COG..."
            )

            overview_factors = [
                2,
                4,
                8,
                16,
                32,
                64
            ]

            dst.build_overviews(
                overview_factors,
                rasterio.enums
                .Resampling
                .nearest
            )

        # =============================================================
        # COG OPTIMIZATION
        # =============================================================

        try:

            from rio_cogeo.cogeo import (
                cog_translate
            )

            from rio_cogeo.profiles import (
                cog_profiles
            )

            temp_output = out.replace(
                ".tif",
                "_temp.tif"
            )

            import shutil

            shutil.move(
                out,
                temp_output
            )

            cog_profile = (
                cog_profiles.get(
                    "deflate"
                )
            )

            cog_profile.update({
                "BLOCKXSIZE": 512,
                "BLOCKYSIZE": 512,
                "OVERVIEW_RESAMPLING": (
                    "nearest"
                ),
            })

            with rasterio.open(
                temp_output
            ) as src_dataset:

                cog_translate(
                    src_dataset,
                    out,
                    cog_profile,
                    in_memory=False,
                    quiet=True,
                )

            os.remove(
                temp_output
            )

            print(
                "✓ COG optimization "
                "completed using rio-cogeo"
            )

        except ImportError:

            print(
                "⚠ rio-cogeo not available, "
                "using basic COG settings"
            )

        except Exception as e:

            print(
                f"⚠ COG optimization failed: "
                f"{e}, using basic "
                f"COG settings"
            )

            if os.path.exists(
                temp_output
            ):

                shutil.move(
                    temp_output,
                    out
                )

        print(
            "Cloud Optimized GeoTIFF "
            "created successfully!"
        )

        return True

    except Exception as e:

        print(
            f"Inference failed: {e}"
        )

        import traceback

        print(
            f"Full traceback: "
            f"{traceback.format_exc()}"
        )

        return False


def _prepare_image_batch(
    image_tensor,
    device,
    band_order=None
):
    """
    Prepare image tensor for model input.
    """

    image_batch = (
        image_tensor
        .unsqueeze(0)
        .to(device)
    )

    if band_order is not None:

        image_batch = _reorder_bands(
            image_batch,
            band_order
        )

    if image_batch.shape[1] == 8:

        num_bands = (
            image_batch.shape[1]
            // 2
        )

        image_batch = torch.cat(
            [
                image_batch[
                    :,
                    num_bands:
                ],
                image_batch[
                    :,
                    :num_bands
                ]
            ],
            dim=1
        )

        print(
            "Applied temporal reordering "
            "for 8-band input"
        )

    elif image_batch.shape[1] == 4:

        print(
            "Using single window data "
            "(4 bands)"
        )

    else:

        raise ValueError(
            f"Unexpected number of "
            f"input bands: "
            f"{image_batch.shape[1]}. "
            f"Expected 4 or 8."
        )

    return image_batch


def _reorder_bands(
    image_batch,
    band_order
):
    """
    Reorder bands in the image batch.
    """

    if band_order == "bgr_to_rgb":

        if image_batch.shape[1] == 4:

            image_batch[:, [0, 2]] = (
                image_batch[:, [2, 0]]
            )

            print(
                "Applied BGR->RGB conversion: "
                "swapped bands 0 and 2"
            )

        elif image_batch.shape[1] == 8:

            image_batch[:, [0, 2]] = (
                image_batch[:, [2, 0]]
            )

            image_batch[:, [4, 6]] = (
                image_batch[:, [6, 4]]
            )

            print(
                "Applied BGR->RGB conversion "
                "for both temporal windows"
            )

        else:

            print(
                f"Warning: BGR->RGB conversion "
                f"not implemented for "
                f"{image_batch.shape[1]} bands"
            )

    elif isinstance(
        band_order,
        (list, tuple)
    ):

        expected_bands = (
            image_batch.shape[1]
        )

        if (
            len(band_order)
            != expected_bands
        ):

            raise ValueError(
                f"Band order list length "
                f"({len(band_order)}) "
                f"doesn't match number "
                f"of bands "
                f"({expected_bands})"
            )

        if not all(
            0 <= idx < expected_bands
            for idx in band_order
        ):

            raise ValueError(
                f"Invalid band indices in "
                f"{band_order} for "
                f"{expected_bands} bands"
            )

        image_batch = (
            image_batch[
                :,
                band_order
            ]
        )

        print(
            f"Applied custom band "
            f"reordering: {band_order}"
        )

    else:

        raise ValueError(
            f"Unknown band_order: "
            f"{band_order}"
        )

    return image_batch


def _run_single_inference(
    image_tensor,
    model_net,
    device,
    save_scores,
    band_order=None
):
    """
    Run inference on a single image
    that fits in memory.
    """

    image_batch = _prepare_image_batch(
        image_tensor,
        device,
        band_order
    )

    print(
        "Running inference..."
    )

    with torch.inference_mode():

        predictions = model_net(
            image_batch
        )

        print(
            f"Model output shape: "
            f"{predictions.shape}"
        )

        print(
            f"Model output range: "
            f"[{predictions.min():.2f}, "
            f"{predictions.max():.2f}]"
        )

        if save_scores:

            predictions = F.softmax(
                predictions,
                dim=1
            )

            predictions = (
                predictions
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            predictions = (
                predictions * 255
            ).clip(
                0,
                255
            ).astype(
                np.uint8
            )

            output_data = (
                predictions[0]
            )

        else:

            predictions = (
                predictions.argmax(
                    dim=1
                )
            )

            print(
                f"Unique prediction values: "
                f"{torch.unique(predictions)}"
            )

            output_data = (
                predictions[0]
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

            output_data = (
                output_data[
                    np.newaxis,
                    ...
                ]
            )

    return output_data