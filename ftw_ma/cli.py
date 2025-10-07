# import datetime
# import enum
import json
# import os
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Optional
import click
import torch

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
        raise click.ClickException(
            f"Failed to import training code: {exc}"
        ) from exc

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
    help="GPU to use, zero-based index. Set to -1 to use CPU."
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

@model.command("predict", help="Run inference on CSV/GeoPackage catalog")
@click.option(
    "--catalog", "-c", required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="CSV catalog with file paths"
)
@click.option(
    "--model", "-m", required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to trained model checkpoint (.ckpt)"
)
@click.option(
    "--output", "-o", required=True,
    type=click.Path(),
    help="Output directory for predictions"
)
@click.option(
    "--data_dir", "-d", required=True,
    type=click.Path(exists=True),
    help="Base directory containing the image files"
)
@click.option(
    "--path_column", default="window_b", show_default=True,
    help="Column name for file paths in CSV (e.g., 'window_a', 'window_b')"
)
@click.option(
    "--id_column", default="name", show_default=True,
    help="Column name for unique IDs in CSV"
)
@click.option(
    "--split", type=str,
    help="Filter CSV to specific split (optional)"
)
@click.option(
    "--normalization_strategy",
    type=click.Choice(["min_max", "z_value"]),
    default="min_max", show_default=True,
    help="Normalization strategy (should match training)"
)
@click.option(
    "--normalization_stat_procedure",
    type=click.Choice(["lab", "lpb", "gab", "gpb"]),
    default="lab", show_default=True,
    help="Statistics procedure (should match training)"
)
@click.option(
    "--img_clip_val", type=float, default=0, show_default=True,
    help="Image clipping value (should match training)"
)
@click.option(
    "--global_stats", type=str,
    help="Global stats as JSON string, e.g. "
         "'{\"mean\": [0,0,0,0], \"std\": [3000,3000,3000,3000]}'"
)
@click.option(
    "--nodata", type=str, default="[null, 65535]", show_default=True,
    help="Nodata values as JSON list, e.g. '[null, 65535]'"
)
@click.option(
    "--band_order", type=str, default="", show_default=True,
    help="Band reordering: 'bgr_to_rgb' to convert BGR-NIR to RGB-NIR, "
         "or comma-separated indices like '0,1,2,3' for custom order, "
         "or leave empty for no reordering (preserve original order)"
)
@click.option(
    "--gpu", type=click.IntRange(min=-1), default=0, show_default=True,
    help="GPU device ID (-1 for CPU)"
)
@click.option(
    "--save_scores", is_flag=True,
    help="Save probability scores instead of class predictions"
)
@click.option(
    "--overwrite", "-f", is_flag=True,
    help="Overwrite existing output files"
)
@click.option(
    "--crop_to_geometry", is_flag=True,
    help="Crop output predictions to geometry boundaries "
         "(requires geometry column in catalog)"
)
@click.option(
    "--geometry_column", default="geometry", show_default=True,
    help="Name of geometry column in geodataframe catalog"
)
@click.option(
    "--mps_mode", is_flag=True,
    help="Use MPS (Apple Silicon) acceleration"
)
@click.option(
    "--num_workers", type=int, default=1, show_default=True,
    help="Number of parallel workers (use 1 for GPU, higher for CPU)"
)
@click.option(
    "--batch_process", is_flag=True,
    help="Enable batch processing for CPU inference"
)
@click.option(
    "--date_column", type=str,
    help="Column name for date values to include in output filename"
)
@click.option(
    "--date_format", type=str, default="%Y%m%d", show_default=True,
    help="Date format for output filename (strftime format)"
)
@click.option(
    "--create_plots", is_flag=True,
    help="Create visualization plots after inference"
)
@click.option(
    "--plot_output_dir", type=str,
    help="Custom output directory for plots (default: output_dir/plots)"
)
@click.option(
    "--plot_rgb_bands", type=str, default="0,1,2", show_default=True,
    help="RGB band indices for plots (comma-separated, 0-indexed)"
)
@click.option(
    "--plot_max_samples", type=int, default=20, show_default=True,
    help="Maximum number of samples to include in plots"
)
@click.option(
    "--plot_images_per_batch", type=int, default=5, show_default=True,
    help="Number of images per plot batch"
)
def model_predict(
    catalog, model, output, data_dir, path_column, id_column, split,
    normalization_strategy, normalization_stat_procedure, img_clip_val,
    global_stats, nodata, band_order, gpu,
    save_scores, overwrite, crop_to_geometry, geometry_column, mps_mode,
    num_workers, batch_process, date_column, date_format, create_plots,
    plot_output_dir, plot_rgb_bands, plot_max_samples, plot_images_per_batch
):
    """Run batch inference from CSV/GeoPackage catalog."""
    from .inference import load_model, inference_run_single

    # Parse JSON strings for complex parameters
    parsed_global_stats = None
    if global_stats:
        try:
            parsed_global_stats = json.loads(global_stats)
            print(f"Parsed global_stats: {parsed_global_stats}")
        except json.JSONDecodeError:
            raise click.BadParameter(
                f"Invalid JSON for global_stats: {global_stats}"
            )

    parsed_nodata = None
    if nodata:
        try:
            parsed_nodata = json.loads(nodata)
            # Convert JSON null to Python None
            parsed_nodata = [None if x is None else x for x in parsed_nodata]
            print(f"Parsed nodata: {parsed_nodata}")
        except json.JSONDecodeError:
            raise click.BadParameter(f"Invalid JSON for nodata: {nodata}")

    # Parse band_order parameter
    parsed_band_order = None
    if band_order:
        if band_order == "bgr_to_rgb":
            parsed_band_order = "bgr_to_rgb"
            print("Using BGR to RGB conversion")
        elif "," in band_order:
            try:
                parsed_band_order = [
                    int(x.strip()) for x in band_order.split(",")
                ]
                print(f"Using custom band order: {parsed_band_order}")
            except ValueError:
                raise click.BadParameter(
                    f"Invalid band order format: {band_order}. "
                    "Use comma-separated integers like '0,1,2,3'"
                )
        else:
            raise click.BadParameter(
                f"Invalid band_order: {band_order}. "
                "Use 'bgr_to_rgb' or comma-separated indices like '0,1,2,3', "
                "or leave empty for no reordering"
            )
    else:
        # Default: preserve original order (no reordering)
        parsed_band_order = None
        print("No band reordering: preserving original order")

    # Setup device with MPS support
    if mps_mode:
        assert torch.backends.mps.is_available(), "MPS mode is not available."
        device = torch.device("mps")
    elif torch.cuda.is_available() and gpu >= 0:
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    model_net = load_model(model, device)
    print("Model loaded successfully!")

    # Load catalog - handle both CSV and spatial formats
    catalog_path = Path(catalog)
    print(f"Loading catalog: {catalog}")

    try:
        # Try to load as GeoDataFrame first (handles .gpkg, .shp, .geojson,etc.)
        if catalog_path.suffix.lower() in ['.gpkg', '.shp', '.geojson']:
            df = gpd.read_file(catalog)
            is_geodataframe = True
            print(f"Loaded as GeoDataFrame with {len(df)} rows")
            if crop_to_geometry and geometry_column not in df.columns:
                raise click.BadParameter(
                    f"Geometry column '{geometry_column}' not found. "
                    f"Available columns: {list(df.columns)}"
                )
        else:
            # Fallback to regular CSV
            df = pd.read_csv(catalog)
            is_geodataframe = False
            print(f"Loaded as DataFrame with {len(df)} rows")
            if crop_to_geometry:
                raise click.BadParameter(
                    "Cannot crop to geometry with CSV catalog. "
                    "Use GeoPackage (.gpkg) or Shapefile (.shp) format."
                )
    except Exception as e:
        raise click.BadParameter(f"Failed to load catalog: {e}")

    # Filter by split if specified
    if split and 'split' in df.columns:
        df = df[df['split'] == split]
        print(f"Filtered to {len(df)} rows for split '{split}'")

    # Validate required columns
    if path_column not in df.columns:
        available_cols = list(df.columns)
        raise click.BadParameter(
            f"Column '{path_column}' not found in catalog. "
            f"Available columns: {available_cols}"
        )

    if id_column and id_column not in df.columns:
        print(f"Warning: ID column '{id_column}' not found, using row index")
        id_column = None

    # Validate date column
    if date_column and date_column not in df.columns:
        print(f"Warning: Date column '{date_column}' not found, ignoring date formatting")
        date_column = None

    # Create output directory
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Convert data_dir to Path object
    data_dir_path = Path(data_dir)

    # Validate parallelization settings
    if batch_process and device.type in ['cuda', 'mps']:
        print("Warning: Batch processing with GPU/MPS may cause memory issues.")
        print("Consider using num_workers=1 for GPU inference.")
    
    if num_workers > 1 and device.type in ['cuda', 'mps']:
        print("Warning: Multiple workers with GPU/MPS may cause conflicts.")
        print("Setting num_workers=1 for GPU/MPS inference.")
        num_workers = 1
    
    # Process files
    if num_workers > 1 and device.type == 'cpu':
        # Parallel processing for CPU
        from concurrent.futures import ProcessPoolExecutor
        results = _process_files_parallel(
            df, data_dir_path, output_dir, model, device, 
            parsed_global_stats, parsed_nodata, parsed_band_order,
            num_workers, path_column, id_column, save_scores,
            normalization_strategy, normalization_stat_procedure, 
            img_clip_val, overwrite, crop_to_geometry, geometry_column, 
            is_geodataframe, date_column, date_format
        )
    else:
        # Sequential processing (default for GPU/MPS)
        results = _process_files_sequential(
            df, data_dir_path, output_dir, model_net, device,
            parsed_global_stats, parsed_nodata, parsed_band_order,
            path_column, id_column, save_scores, normalization_strategy,
            normalization_stat_procedure, img_clip_val, overwrite,
            crop_to_geometry, geometry_column, is_geodataframe,
            date_column, date_format
        )

    # Save results summary (moved out of functions to avoid duplication)
    results_df = pd.DataFrame(results)
    results_file = output_dir / "inference_results.csv"
    results_df.to_csv(results_file, index=False)

    # Print summary
    total_files = len(results)
    successful = results_df['success'].sum()
    cropped_count = results_df.get(
        'cropped', pd.Series([False]*len(results_df))
    ).sum()
    print(f"\nBatch inference summary:")
    print(f"  Total files: {total_files}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {total_files - successful}")
    if crop_to_geometry:
        print(f"  Cropped to geometry: {cropped_count}")

    # Create plots if requested
    if create_plots:
        print(f"\nCreating visualization plots...")
        
        try:
            # Parse RGB bands
            rgb_bands = tuple(int(x.strip()) for x in plot_rgb_bands.split(","))
            if len(rgb_bands) != 3:
                raise ValueError("RGB bands must be exactly 3 values")
            
            # Determine plot output directory
            if plot_output_dir:
                plots_dir = Path(plot_output_dir)
            else:
                plots_dir = output_dir / "plots"
            
            # Extract experiment name from model checkpoint path
            experiment_name = _extract_experiment_name_from_model(Path(model))
            print(f"Extracted experiment name from model path: {experiment_name}")
            
            from .plotting import create_inference_plots
            
            create_inference_plots(
                catalog_path=catalog_path,
                data_dir=data_dir_path,
                inference_dir=output_dir,
                output_dir=plots_dir,
                rgb_bands=rgb_bands,
                images_per_plot=plot_images_per_batch,
                max_samples=plot_max_samples,
                experiment_name=experiment_name  # Add this parameter
            )
            
            print(f"✓ Plots saved to: {plots_dir}")
            
        except Exception as e:
            print(f"⚠ Failed to create plots: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")

def _extract_experiment_name_from_model(model_path: Path) -> str:
    """Extract experiment name from model checkpoint path."""
    path_parts = model_path.parts
    
    # Look for lightning_logs in the path
    if "lightning_logs" in path_parts:
        lightning_idx = path_parts.index("lightning_logs")
        if lightning_idx > 0:
            # Return the directory name before lightning_logs
            return path_parts[lightning_idx - 1]
    
    # Alternative patterns to look for
    # Look for common experiment directory patterns
    for i, part in enumerate(path_parts):
        # If we find version_X, the parent might be the experiment
        if part.startswith("version_") and i > 0:
            return path_parts[i - 1]
        # If we find checkpoints directory, go up two levels
        if part == "checkpoints" and i >= 2:
            return path_parts[i - 2]
    
    # If no recognizable pattern, use the grandparent directory of model file
    if len(path_parts) >= 3:
        return path_parts[-3]  # Grandparent directory
    elif len(path_parts) >= 2:
        return path_parts[-2]  # Parent directory
    
    # Fallback to model filename without extension
    return model_path.stem

def _process_files_sequential(
        df, data_dir_path, output_dir, model_net, device,
        parsed_global_stats, parsed_nodata, parsed_band_order,
        path_column, id_column, save_scores, 
        normalization_strategy, normalization_stat_procedure,
        img_clip_val, overwrite, crop_to_geometry, 
        geometry_column, is_geodataframe, date_column, 
        date_format
    ):
    """Process files sequentially (original implementation)."""
    from .inference import inference_run_single
    import geopandas as gpd
    import pandas as pd
    
    results = []
    
    print(f"\nStarting sequential inference on {len(df)} files...")
    if crop_to_geometry:
        print(f"Will crop outputs to geometries from '{geometry_column}' column")
    
    for idx, row in df.iterrows():
        try:
            # Get file path
            file_path = row[path_column]
            if pd.isna(file_path):
                print(f"Skipping row {idx}: missing file path")
                continue

            # Resolve full path
            full_path = data_dir_path / file_path
            if not full_path.exists():
                print(f"Warning: File not found: {full_path}")
                results.append({
                    'row_index': idx,
                    'file_path': str(file_path),
                    'full_path': str(full_path),
                    'success': False,
                    'error': 'File not found'
                })
                continue

            # Get crop geometry if requested
            crop_geom = None
            if crop_to_geometry and is_geodataframe:
                if geometry_column in row and pd.notna(row[geometry_column]):
                    # Get the geometry
                    geom = row[geometry_column]
                    
                    # Create a single-item GeoSeries to preserve CRS
                    crop_geom = gpd.GeoSeries([geom], crs=df.crs)
                else:
                    print(f"Warning: No valid geometry found for row {idx} "
                          f"in column '{geometry_column}'")
        
            # Generate output filename with optional date
            file_id = row[id_column] if id_column else f"row_{idx}"
            
            # Add date to filename if specified
            if date_column and pd.notna(row[date_column]):
                try:
                    # Handle different date formats
                    date_val = row[date_column]
                    if isinstance(date_val, str):
                        # Try to parse string dates
                        import datetime
                        parsed_date = pd.to_datetime(date_val)
                    else:
                        parsed_date = date_val
                    
                    date_str = parsed_date.strftime(date_format)
                    output_filename = f"{file_id}_{date_str}_prediction.tif"
                except Exception as e:
                    print(f"Warning: Could not format date for row {idx}: {e}")
                    output_filename = f"{file_id}_prediction.tif"
            else:
                output_filename = f"{file_id}_prediction.tif"
                
            output_file = output_dir / output_filename

            print(f"Processing {file_id}: {file_path}")
            if crop_geom is not None:
                # Access the bounds from the geometry, not the GeoSeries
                geom_bounds = crop_geom.iloc[0].bounds
                print(f"  Will crop to geometry bounds: {geom_bounds}")
            if date_column and pd.notna(row[date_column]):
                print(f"  Output: {output_filename}")

            # Run inference
            success = inference_run_single(
                input_file=str(full_path),
                model_net=model_net,
                device=device,
                out=str(output_file),
                save_scores=save_scores,
                normalization_strategy=normalization_strategy,
                normalization_stat_procedure=normalization_stat_procedure,
                global_stats=parsed_global_stats,
                img_clip_val=img_clip_val,
                nodata=parsed_nodata,
                overwrite=overwrite,
                band_order=parsed_band_order,
                crop_geometry=crop_geom,
            )

            results.append({
                'row_index': idx,
                'file_id': file_id,
                'file_path': str(file_path),
                'full_path': str(full_path),
                'output_path': str(output_file),
                'success': success,
                'cropped': crop_geom is not None
            })

            if success:
                crop_status = " (cropped)" if crop_geom is not None else ""
                print(f"  ✓ Success: {output_filename}{crop_status}")
            else:
                print(f"  ✗ Failed: {file_path}")

        except Exception as e:
            print(f"  ✗ Error processing row {idx}: {e}")
            results.append({
                'row_index': idx,
                'file_path': str(row.get(path_column, 'unknown')),
                'success': False,
                'error': str(e)
            })

    return results

def _process_files_parallel(
    df, data_dir_path, output_dir, model_path, device, 
    parsed_global_stats, parsed_nodata, parsed_band_order,
    num_workers, path_column, id_column, save_scores,
    normalization_strategy, normalization_stat_procedure,
    img_clip_val, overwrite, crop_to_geometry, 
    geometry_column, is_geodataframe, date_column, 
    date_format
):
    """Process files in parallel (CPU only)."""
    from concurrent.futures import ProcessPoolExecutor
    import functools
    
    print(f"\nStarting parallel inference on {len(df)} files with {num_workers} workers...")
    if crop_to_geometry:
        print(f"Will crop outputs to geometries from '{geometry_column}' column")
    
    # Split dataframe into chunks
    chunk_size = max(1, len(df) // num_workers)
    chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    
    # Process function for each chunk
    process_func = functools.partial(
        _process_chunk,
        data_dir_path=data_dir_path,
        output_dir=output_dir,
        model_path=model_path,
        device_type=device.type,
        parsed_global_stats=parsed_global_stats,
        parsed_nodata=parsed_nodata,
        parsed_band_order=parsed_band_order,
        path_column=path_column,
        id_column=id_column,
        save_scores=save_scores,
        normalization_strategy=normalization_strategy,
        normalization_stat_procedure=normalization_stat_procedure,
        img_clip_val=img_clip_val,
        overwrite=overwrite,
        crop_to_geometry=crop_to_geometry,
        geometry_column=geometry_column,
        is_geodataframe=is_geodataframe,
        date_column=date_column,
        date_format=date_format,
        df_crs=df.crs if is_geodataframe else None
    )
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        chunk_results = list(executor.map(process_func, chunks))
    
    # Flatten results
    results = []
    for chunk_result in chunk_results:
        results.extend(chunk_result)
    
    return results

def _process_chunk(
    chunk_df, data_dir_path, output_dir, model_path, device_type,
    parsed_global_stats, parsed_nodata, parsed_band_order,
    path_column, id_column, save_scores, normalization_strategy,
    normalization_stat_procedure, img_clip_val, overwrite,
    crop_to_geometry, geometry_column, is_geodataframe,
    date_column, date_format, df_crs
):
    """Process a chunk of the dataframe."""
    from .inference import load_model, inference_run_single
    
    # Load model in each process
    device = torch.device(device_type)
    model_net = load_model(model_path, device)
    
    results = []
    for idx, row in chunk_df.iterrows():
        try:
            # Get file path
            file_path = row[path_column]
            if pd.isna(file_path):
                print(f"Skipping row {idx}: missing file path")
                continue

            # Resolve full path
            full_path = data_dir_path / file_path
            if not full_path.exists():
                print(f"Warning: File not found: {full_path}")
                results.append({
                    'row_index': idx,
                    'file_path': str(file_path),
                    'full_path': str(full_path),
                    'success': False,
                    'error': 'File not found'
                })
                continue

            # Get crop geometry if requested
            crop_geom = None
            if crop_to_geometry and is_geodataframe:
                if geometry_column in row and pd.notna(row[geometry_column]):
                    # Get the geometry
                    geom = row[geometry_column]
                    
                    # Create a single-item GeoSeries to preserve CRS
                    crop_geom = gpd.GeoSeries([geom], crs=df_crs)
                else:
                    print(f"Warning: No valid geometry found for row {idx} "
                          f"in column '{geometry_column}'")
        
            # Generate output filename with optional date
            file_id = row[id_column] if id_column else f"row_{idx}"
            
            # Add date to filename if specified
            if date_column and pd.notna(row[date_column]):
                try:
                    # Handle different date formats
                    date_val = row[date_column]
                    if isinstance(date_val, str):
                        # Try to parse string dates
                        import datetime
                        parsed_date = pd.to_datetime(date_val)
                    else:
                        parsed_date = date_val
                    
                    date_str = parsed_date.strftime(date_format)
                    output_filename = f"{file_id}_{date_str}_prediction.tif"
                except Exception as e:
                    print(f"Warning: Could not format date for row {idx}: {e}")
                    output_filename = f"{file_id}_prediction.tif"
            else:
                output_filename = f"{file_id}_prediction.tif"
                
            output_file = output_dir / output_filename

            print(f"Processing {file_id}: {file_path}")
            if crop_geom is not None:
                # Access the bounds from the geometry, not the GeoSeries
                geom_bounds = crop_geom.iloc[0].bounds
                print(f"  Will crop to geometry bounds: {geom_bounds}")
            if date_column and pd.notna(row[date_column]):
                print(f"  Output: {output_filename}")

            # Run inference
            success = inference_run_single(
                input_file=str(full_path),
                model_net=model_net,
                device=device,
                out=str(output_file),
                save_scores=save_scores,
                normalization_strategy=normalization_strategy,
                normalization_stat_procedure=normalization_stat_procedure,
                global_stats=parsed_global_stats,
                img_clip_val=img_clip_val,
                nodata=parsed_nodata,
                overwrite=overwrite,
                band_order=parsed_band_order,
                crop_geometry=crop_geom,
            )

            results.append({
                'row_index': idx,
                'file_id': file_id,
                'file_path': str(file_path),
                'full_path': str(full_path),
                'output_path': str(output_file),
                'success': success,
                'cropped': crop_geom is not None
            })

            if success:
                crop_status = " (cropped)" if crop_geom is not None else ""
                print(f"  ✓ Success: {output_filename}{crop_status}")
            else:
                print(f"  ✗ Failed: {file_path}")

        except Exception as e:
            print(f"  ✗ Error processing row {idx}: {e}")
            results.append({
                'row_index': idx,
                'file_path': str(row.get(path_column, 'unknown')),
                'success': False,
                'error': str(e)
            })

    return results

if __name__ == "__main__":
    ftw_ma()
