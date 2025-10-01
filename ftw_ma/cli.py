# import datetime
# import enum
# import json
# import os
from pathlib import Path
from typing import Optional
import click

@click.group()
def ftw_ma():
    """Fields of The World (FTW) / Mapping Africa - Command Line Interface"""
    pass

@ftw_ma.group()
def model():
    """Training and testing FTW models."""
    pass

@model.command("fit", help="Fit the model")
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the config file",
)
@click.option(
    "--ckpt_path",
    "-m",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    show_default=True,
    help="Path to a checkpoint file to resume training from",
)
@click.argument(
    "cli_args", nargs=-1, type=click.UNPROCESSED
)  # Capture all remaining arguments
def model_fit(config, ckpt_path, cli_args):
    try:
        from .compiler import fit
    except Exception as exc:
        raise click.ClickException(f"Failed to import training code: {exc}") \
            from exc

    fit(config, ckpt_path, cli_args)

# test
@model.command("test", help="Test the model")
@click.option(
    "--config",
    "-cfg",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the config file",
)
@click.option(
    "--gpu",
    type=click.IntRange(min=-1),
    default=0,
    show_default=True,
    help=f"GPU to use, zero-based index. Set to -1 to use CPU."\
        "CPU is also always used if CUDA is not available.",
)
@click.option(
    "--model_path",
    "-m",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to model checkpoint",
)
@click.option(
    "--data_dir",
    "-d",
    type=click.Path(exists=False),
    default="/home/airg/data/labels/cropland",
    show_default=True,
    help="Path to the directory containing the data",
)
@click.option(
    "--catalog",
    "-cat",
    type=click.Path(exists=False),
    default="../data/ftw-catalog-small.csv",
    show_default=True,
    help="Path to the directory containing the data",
)
@click.option(
    "--split",
    "-spl",
    type=click.Choice(["validate", "test"]),
    default="test",
    show_default=True,
    help="Choose validate or test split",
)
@click.option(
    "--temporal_options",
    "-t",
    type=click.Choice(["stacked", "windowA", "windowB"]),
    default="windowB",
    show_default=True,
    help="Temporal option",
)
@click.option(
    "--iou_threshold",
    "-iou",
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.5,
    show_default=True,
    help="IoU threshold for matching predictions to ground truths",
)
@click.option(
    "--out",
    "-o",
    type=click.Path(exists=False),
    default="metrics.csv",
    show_default=True,
    help="Output file for metrics",
)

def model_test(
    config,    
    gpu,
    model_path,
    data_dir,
    catalog,
    split,
    temporal_options,
    iou_threshold,
    out,
):
    from .compiler import test

    test(
        config,    
        gpu,
        model_path,
        data_dir,
        catalog,
        split,
        temporal_options,
        iou_threshold,
        out,
    )

@model.command("predict", help="Run inference on new data")
    """
    CLI command to run inference imagery.
    This command supports running predictions on:
    - Single raster files (.tif, .vrt)
    - Batch processing from CSV/GeoJSON/ESRI Shapefile containing file paths
    - Directories of input files
    Parameters
    ----------
    input : str
        Input source. Can be a raster file (.tif/.vrt), a CSV/GeoJSON/Shapefile
        with file paths, or a directory.
    model : str
        Path to the trained model checkpoint (.ckpt).
    output : str, optional
        Output file path for single input, or directory for batch processing.
    path_column : str, default="path"
        Column name for file paths in DataFrame/GeoDataFrame inputs.
    geometry_column : str, default="geometry"
        Column name for geometries in GeoDataFrame inputs.
    id_column : str, optional
        Column name for unique IDs (optional).
    crop_to_geometry : bool, default=True
        Whether to crop predictions to geometry bounds (GeoDataFrame only).
    buffer_pixels : int, default=64
        Buffer size around geometry in pixels.
    Notes
    -----
    - For raster input, predictions are saved as a new raster file.
    - For DataFrame/GeoDataFrame input, predictions are saved in the specified
    output directory.
    - Additional keyword arguments (**kwargs) are passed to the inference
    function.
    """`
@click.option(
    "--input", "-i", required=True,
    help="Input: raster (.tif/.vrt), CSV/GeoJSON w/file paths, or \directory"
)
@click.option(
    "--model", "-m", required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to trained model checkpoint (.ckpt)"
)
@click.option(
    "--output", "-o", type=click.Path(),
    help="Output: file for single input, directory for batch processing"
)
@click.option(
    "--path_column", default="path", show_default=True,
    help="Column name for file paths (DataFrame/GeoDataFrame)"
)
@click.option(
    "--geometry_column", default="geometry", show_default=True,
    help="Column name for geometries (GeoDataFrame)"
)
@click.option(
    "--id_column", help="Column name for unique IDs (optional)"
)
@click.option(
    "--crop_to_geometry/--no-crop", default=True, show_default=True,
    help="Crop predictions to geometry bounds (GeoDataFrame only)"
)
@click.option(
    "--buffer_pixels", type=int, default=64, show_default=True,
    help="Buffer around geometry in pixels"
)
# ... existing options ...
def model_predict(
    input, model, output, path_column, geometry_column, id_column,
    crop_to_geometry, buffer_pixels, **kwargs
):
    """Run inference on raster files or batch process from DataFrame/
    GeoDataFrame."""
    from .inference import run_batch_inference
    import pandas as pd
    import geopandas as gpd

    # Determine input type
    input_path = Path(input)

    if input_path.suffix.lower() in ['.tif', '.vrt']:
        # Single raster file
        if not output:
            output = (
                input_path.parent / f"prediction_{input_path.stem}.tif"
            )

        run_batch_inference(
            input_data=str(input_path),
            model=model,
            output_dir=str(Path(output).parent),
            **kwargs
        )

    elif input_path.suffix.lower() in ['.csv', '.geojson', '.shp']:
        # DataFrame/GeoDataFrame
        if not output:
            output = input_path.parent / "predictions"

        # Load data
        if input_path.suffix.lower() == '.csv':
            data = pd.read_csv(input_path)
        else:
            data = gpd.read_file(input_path)

        run_batch_inference(
            input_data=data,
            model=model,
            output_dir=str(output),
            path_column=path_column,
            geometry_column=geometry_column,
            id_column=id_column,
            crop_to_geometry=crop_to_geometry,
            buffer_pixels=buffer_pixels,
            **kwargs
        )
        
    else:
        raise click.BadParameter(f"Unsupported input type: {input_path.suffix}")

if __name__ == "__main__":
    ftw_ma()
