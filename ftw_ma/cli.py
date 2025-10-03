# import datetime
# import enum
import json
# import os
import pandas as pd
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

@model.command("predict", help="Run inference on CSV catalog")
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
# Normalization options to match training
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
    help="Global stats as JSON string, e.g., "
         "'{\"mean\": [0,0,0,0], \"std\": [3000,3000,3000,3000]}'"
)
@click.option(
    "--nodata", type=str, default="[null, 65535]", show_default=True,
    help="Nodata values as JSON list, e.g., '[null, 65535]'"
)
# Band ordering option
@click.option(
    "--band_order", type=str,
    help="Band reordering: 'bgr_to_rgb' to convert BGR-NIR to RGB-NIR, "
         "or comma-separated indices like '0,1,2,3' for custom order, "
         "or leave empty for no reordering"
)
# Inference parameters
@click.option(
    "--gpu", type=click.IntRange(min=-1), default=0, show_default=True,
    help="GPU device ID (-1 for CPU)"
)
@click.option(
    "--patch_size", type=click.IntRange(min=64), 
    help="Patch size for inference (auto-detected if not specified)"
)
@click.option(
    "--batch_size", type=int, default=1, show_default=True,
    help="Batch size for inference"
)
@click.option(
    "--num_workers", type=int, default=0, show_default=True,
    help="Number of worker processes"
)
@click.option(
    "--padding", type=int,
    help="Padding size for patches (auto-calculated if not specified)"
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
    "--mps_mode", is_flag=True,
    help="Use MPS (Apple Silicon) acceleration"
)
def model_predict(
    catalog, model, output, data_dir, path_column, id_column, split,
    normalization_strategy, normalization_stat_procedure, img_clip_val, 
    global_stats, nodata, band_order, gpu, 
    patch_size, batch_size, #resize_factor, 
    num_workers, 
    padding, save_scores, overwrite, mps_mode
):
    """Run batch inference from CSV catalog."""
    from .inference import load_model, inference_run_single
    import torch

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
            # Parse comma-separated indices
            try:
                parsed_band_order = [int(x.strip()) for x in band_order.split(",")]
                print(f"Using custom band order: {parsed_band_order}")
            except ValueError:
                raise click.BadParameter(
                    f"Invalid band order format: {band_order}. "
                    "Use comma-separated integers like '0,1,2,3'"
                )
        else:
            raise click.BadParameter(
                f"Invalid band_order: {band_order}. "
                "Use 'bgr_to_rgb' or comma-separated indices like '0,1,2,3'"
            )

    # Setup device ONCE
    if mps_mode:
        assert torch.backends.mps.is_available(), "MPS mode is not available."
        device = torch.device("mps")
    elif torch.cuda.is_available() and gpu >= 0:
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Load model ONCE at the beginning
    print("🤖 Loading model (this will only happen once)...")
    model_net = load_model(model, device)
    print("✅ Model loaded successfully!")

    # Load and filter CSV
    print(f"Loading CSV catalog: {catalog}")
    df = pd.read_csv(catalog)
    print(f"Loaded CSV with {len(df)} rows")
    
    # Filter by split if specified
    if split and 'split' in df.columns:
        df = df[df['split'] == split]
        print(f"Filtered to {len(df)} rows for split '{split}'")
    
    # Validate required columns
    if path_column not in df.columns:
        available_cols = list(df.columns)
        raise click.BadParameter(
            f"Column '{path_column}' not found in CSV. "
            f"Available columns: {available_cols}"
        )
    
    if id_column and id_column not in df.columns:
        print(f"Warning: ID column '{id_column}' not found, using row index")
        id_column = None
    
    # Create output directory
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Process each file
    results = []
    data_dir_path = Path(data_dir)
    
    print(f"\n🚀 Starting batch inference on {len(df)} files...")
    
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
            
            # Debug: Check what's actually in the file before inference
            print(f"Debug: Checking file {full_path}")
            try:
                import rasterio
                with rasterio.open(full_path) as src:
                    # Read a small sample to check values
                    sample = src.read(1, window=rasterio.windows.Window(0, 0, 10, 10))
                    print(f"  File shape: {src.shape}")
                    print(f"  File dtype: {src.dtypes}")
                    print(f"  File nodata: {src.nodata}")
                    print(f"  Sample values: min={sample.min()}, max={sample.max()}")
                    print(f"  Sample dtype: {sample.dtype}")
            except Exception as e:
                print(f"  Error reading file directly: {e}")
            
            # Generate output filename
            file_id = row[id_column] if id_column else f"row_{idx}"
            output_filename = f"prediction_{file_id}.tif"
            output_file = output_dir / output_filename
            
            print(f"Processing {file_id}: {file_path}")
            
            # Run inference with pre-loaded model
            # Note: patch_size, batch_size, num_workers, padding are ignored in 
            # simplified version
            success = inference_run_single(
                input_file=str(full_path),
                model_net=model_net,  # Pre-loaded model
                device=device,
                out=str(output_file),
                save_scores=save_scores,
                normalization_strategy=normalization_strategy,
                normalization_stat_procedure=normalization_stat_procedure,
                global_stats=parsed_global_stats,
                img_clip_val=img_clip_val,
                nodata=parsed_nodata,
                overwrite=overwrite,
                patch_size=patch_size,   # Can be None for auto-determination
                buffer_size=padding,     # Use padding as buffer size
                band_order=parsed_band_order,  # Add band order parameter
            )
            
            results.append({
                'row_index': idx,
                'file_id': file_id,
                'file_path': str(file_path),
                'full_path': str(full_path),
                'output_path': str(output_file),
                'success': success
            })
            
            if success:
                print(f"  ✓ Success: {output_filename}")
            else:
                print(f"  ✗ Failed: {file_path}")
            
        except Exception as e:
            print(f"  ✗ Error processing row {idx}: {e}")
            import traceback
            print(f"  Full traceback: {traceback.format_exc()}")
            results.append({
                'row_index': idx,
                'file_path': str(row.get(path_column, 'unknown')),
                'success': False,
                'error': str(e)
            })
    
    # Save results summary
    results_df = pd.DataFrame(results)
    results_file = output_dir / "inference_results.csv"
    results_df.to_csv(results_file, index=False)
    
    # Print summary
    total_files = len(results)
    successful = results_df['success'].sum()
    print(f"\n📊 Batch inference summary:")
    print(f"  Total files: {total_files}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {total_files - successful}")
    print(f"  Results saved to: {results_file}")

if __name__ == "__main__":
    ftw_ma()
