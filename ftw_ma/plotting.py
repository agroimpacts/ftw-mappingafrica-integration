"""
Plotting utilities for visualizing inference results alongside original images.
Works with both chip-scale (image/mask/prediction) and tile-scale 
(image/prediction only) data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import rasterio
from typing import Tuple, Optional, Union
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_normalize_image(
    path: Path, bands: Tuple[int, int, int] = (2, 1, 0)
) -> Tuple[np.ndarray, dict]:
    """Load image and normalize for RGB display. Returns image and geospatial 
    info."""
    try:
        with rasterio.open(path) as src:
            # Get geospatial information
            transform = src.transform
            bounds = src.bounds
            crs = src.crs
            
            # Read specified bands (default B3,B2,B1 for RGB)
            img = src.read(
                [bands[0]+1, bands[1]+1, bands[2]+1]
            )  # rasterio is 1-indexed
            img = img.transpose(1, 2, 0)  # (H, W, C)

            # Normalize to 0-1 for display
            img = img.astype(np.float32)
            # Handle nodata and clip outliers
            img = np.where(img <= 0, 0, img)
            img = np.where(img >= 65535, 65535, img)

            # Percentile normalization for better visualization
            for i in range(3):
                band = img[:, :, i]
                valid_pixels = band[band > 0]
                if len(valid_pixels) > 0:
                    p2, p98 = np.percentile(valid_pixels, [2, 98])
                    img[:, :, i] = np.clip(
                        (band - p2) / (p98 - p2 + 1e-8), 0, 1
                    )

            geo_info = {
                'transform': transform,
                'bounds': bounds,
                'crs': crs,
                'shape': img.shape[:2]
            }

            return img, geo_info
    except Exception as e:
        print(
            f"Error loading image {path}: {e}"
        )
        return None, None

def load_inference_result(path: Path) -> Tuple[Optional[np.ndarray], dict]:
    """Load inference result and its geospatial info."""
    try:
        with rasterio.open(path) as src:
            pred = src.read(1)  # Assume single band
            
            geo_info = {
                'transform': src.transform,
                'bounds': src.bounds,
                'crs': src.crs,
                'shape': pred.shape
            }
            
            return pred, geo_info
    except Exception as e:
        print(
            f"Error loading inference result {path}: {e}"
        )
        return None, None

def load_ground_truth_mask(
    path: Union[Path, str], data_dir: Path
) -> Tuple[Optional[np.ndarray], dict]:
    """Load ground truth mask and its geospatial info."""
    try:
        if isinstance(path, str):
            full_path = data_dir / path
        else:
            full_path = path
        with rasterio.open(full_path) as src:
            mask = src.read(1)
            
            geo_info = {
                'transform': src.transform,
                'bounds': src.bounds,
                'crs': src.crs,
                'shape': mask.shape
            }
            
            return mask, geo_info
    except Exception as e:
        print(
            f"Error loading ground truth mask {full_path}: {e}"
        )
        return None, None

def get_image_path(row: pd.Series, data_dir: Path) -> Optional[Path]:
    """Get the correct image path based on dataset type."""
    # Try different possible column names for image paths
    path_columns = [
        'window_b', 'window_a', 'image_path', 'path', 'file_path', 
        'filename'
    ]
    
    for col in path_columns:
        if col in row and pd.notna(row[col]):
            img_path = data_dir / row[col]
            if img_path.exists():
                return img_path
    
    print(
        f"    Available columns: {list(row.index)}"
    )
    return None

def create_prediction_overlay(
    pred_data: np.ndarray, pred_geo: dict, img_geo: dict, 
    img_shape: Tuple[int, int]
) -> np.ndarray:
    """Create prediction overlay scaled to match the image spatial bounds."""
    try:
        import rasterio.warp
        from rasterio.windows import from_bounds
        
        # Create an empty array the same size as the image
        pred_overlay = np.zeros(img_shape, dtype=pred_data.dtype)
        
        # Calculate where the prediction fits within the image bounds
        img_bounds = img_geo['bounds']
        pred_bounds = pred_geo['bounds']
        
        print(
            f"    Image bounds: {img_bounds}"
        )
        print(
            f"    Prediction bounds: {pred_bounds}"
        )
        
        # Calculate pixel coordinates in the image where prediction should be placed
        img_transform = img_geo['transform']
        
        # Convert prediction bounds to image pixel coordinates
        # Upper left corner of prediction in image coordinates
        ul_col, ul_row = ~img_transform * (pred_bounds.left, pred_bounds.top)
        # Lower right corner of prediction in image coordinates  
        lr_col, lr_row = ~img_transform * (pred_bounds.right, pred_bounds.bottom)
        
        # Convert to integer pixel indices
        start_row = max(0, int(ul_row))
        end_row = min(img_shape[0], int(lr_row))
        start_col = max(0, int(ul_col))
        end_col = min(img_shape[1], int(lr_col))
        
        print(
            f"    Placing prediction at image pixels: rows "
            f"{start_row}:{end_row}, cols {start_col}:{end_col}"
        )
        
        # Calculate the size needed for the prediction in image space
        target_rows = end_row - start_row
        target_cols = end_col - start_col
        
        if target_rows <= 0 or target_cols <= 0:
            print(
                f"    Warning: No spatial overlap between image and prediction"
            )
            return pred_overlay
        
        # Resize prediction to fit the target area in the image
        from skimage.transform import resize
        resized_pred = resize(
            pred_data, 
            (target_rows, target_cols), 
            preserve_range=True, 
            anti_aliasing=False, 
            order=0  # Nearest neighbor for categorical data
        ).astype(pred_data.dtype)
        
        # Place the resized prediction in the overlay
        pred_overlay[start_row:end_row, start_col:end_col] = resized_pred
        
        print(
            f"    Created prediction overlay: {pred_overlay.shape}"
        )
        print(
            f"    Non-zero pixels in overlay: {np.count_nonzero(pred_overlay)}"
        )
        
        return pred_overlay
        
    except Exception as e:
        print(
            f"    Error creating prediction overlay: {e}"
        )
        print(
            f"    Returning empty overlay"
        )
        # Return empty overlay the same size as image
        return np.zeros(img_shape, dtype=pred_data.dtype)

def create_inference_plots(
    catalog_path: Path, data_dir: Path, inference_dir: Path, 
    output_dir: Path, rgb_bands: Tuple[int, int, int] = (0, 1, 2),
    images_per_plot: int = 5, max_samples: int = 20,
    experiment_name: Optional[str] = None
) -> None:
    """Main function to create inference plots from catalog."""
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine prefix for output files
    if experiment_name:
        file_prefix = f"{experiment_name}_"
        print(f"Using experiment name prefix: {experiment_name}")
    else:
        file_prefix = ""

    # Load catalog
    if not catalog_path.exists():
        print(
            f"Error: Catalog file not found: {catalog_path}"
        )
        return

    # Try loading as spatial format first, then as CSV
    try:
        if catalog_path.suffix.lower() in ['.gpkg', '.shp', '.geojson']:
            try:
                import geopandas as gpd
                df = gpd.read_file(catalog_path)
                print(
                    f"Loaded spatial catalog with {len(df)} samples"
                )
            except ImportError:
                print(
                    "Error: geopandas unavailable to read spatial formats"
                )
                return
        else:
            df = pd.read_csv(catalog_path)
            print(
                f"Loaded CSV catalog with {len(df)} samples"
            )
    except Exception as e:
        print(
            f"Error loading catalog: {e}"
        )
        return

    # Check if we have ground truth masks
    mask_columns = ['mask', 'mask_path', 'label', 'label_path']
    has_mask_column = any(col in df.columns for col in mask_columns)
    mask_column = None
    
    if has_mask_column:
        for col in mask_columns:
            if col in df.columns:
                mask_column = col
                break
        print(
            f"Found mask column: {mask_column}"
        )
    else:
        print(
            "No mask column found - will create prediction-only plots"
        )

    # Process each sample and collect successful ones
    successful_samples = []
    failed = 0

    for idx, row in df.iterrows():
        # Get sample name/ID - try multiple possible column names
        name_columns = ['name', 'id', 'file_id', 'sample_id']
        name = None
        for col in name_columns:
            if col in row and pd.notna(row[col]):
                name = str(row[col])
                break
        
        if name is None:
            name = f"sample_{idx}"

        print(
            f"\nProcessing {name} ({idx+1}/{len(df)})"
        )

        # Find inference result - try multiple possible naming patterns
        inference_patterns = [
            inference_dir / f'{name}.tif',
            inference_dir / f'{name}_prediction.tif',
            inference_dir / f'{name}_pred.tif',
        ]
        
        # If we have a date column, also try date-based patterns
        date_columns = ['date', 'acquisition_date', 'image_date']
        for date_col in date_columns:
            if date_col in row and pd.notna(row[date_col]):
                try:
                    date_val = row[date_col]
                    if isinstance(date_val, str):
                        parsed_date = pd.to_datetime(date_val)
                    else:
                        parsed_date = date_val
                    date_str = parsed_date.strftime("%Y%m%d")
                    inference_patterns.extend([
                        inference_dir / f'{name}_{date_str}_prediction.tif',
                        inference_dir / f'{name}_{date_str}.tif',
                    ])
                except:
                    pass  # Skip if date parsing fails

        inference_path = None
        for pattern in inference_patterns:
            if pattern.exists():
                inference_path = pattern
                break

        if inference_path is None:
            print(
                f"  Warning: No inference result found for {name}"
            )
            print(
                f"    Tried patterns: {[p.name for p in inference_patterns]}"
            )
            failed += 1
            continue

        # Load RGB image with geospatial info
        img_path = get_image_path(row, data_dir)
        if img_path is None:
            print(
                f"  Warning: Image not found for {name}"
            )
            failed += 1
            continue

        rgb_img, img_geo = load_and_normalize_image(img_path, rgb_bands)
        if rgb_img is None or img_geo is None:
            failed += 1
            continue

        # Load inference result with geospatial info
        pred_mask, pred_geo = load_inference_result(inference_path)
        if pred_mask is None or pred_geo is None:
            failed += 1
            continue

        # Load ground truth mask if available
        gt_mask = None
        gt_geo = None
        if has_mask_column and mask_column in row \
            and pd.notna(row[mask_column]):
            gt_mask, gt_geo = load_ground_truth_mask(row[mask_column], data_dir)

        # Handle dimension mismatches by creating spatial overlays
        if rgb_img.shape[:2] != pred_mask.shape:
            print(
                f"  Dimension mismatch - RGB: {rgb_img.shape[:2]}, "
                f"Pred: {pred_mask.shape}"
            )
            
            # Try to create spatial overlay if we have geospatial info
            if img_geo is not None and pred_geo is not None:
                print(
                    f"  Creating prediction overlay to match image spatial bounds..."
                )
                try:
                    pred_overlay = create_prediction_overlay(
                        pred_mask, pred_geo, img_geo, rgb_img.shape[:2]
                    )
                    pred_mask = pred_overlay
                    
                    # Also handle ground truth if available and has same issue
                    if gt_mask is not None and gt_mask.shape != rgb_img.shape[:2] \
                        and gt_geo is not None:
                        print(
                            f"  Also creating ground truth overlay..."
                        )
                        gt_overlay = create_prediction_overlay(
                            gt_mask, gt_geo, img_geo, rgb_img.shape[:2]
                        )
                        gt_mask = gt_overlay
                except Exception as e:
                    print(
                        f"  Warning: Could not create spatial overlay: {e}"
                    )
                    print(
                        f"  Skipping this sample due to spatial mismatch"
                    )
                    failed += 1
                    continue
            else:
                print(
                    f"  Warning: Missing geospatial info - cannot create spatial overlay"
                )
                print(
                    f"  Skipping this sample due to dimension mismatch"
                )
                failed += 1
                continue
        else:
            # If dimensions match, use original data
            print(
                f"  Dimensions match - RGB: {rgb_img.shape[:2]}, "
                f"Pred: {pred_mask.shape}"
            )

        # Final dimension check
        print(
            f"  Final dimensions - RGB: {rgb_img.shape[:2]}, Pred: {pred_mask.shape}"
        )
        if gt_mask is not None:
            print(
                f"  Final dimensions - GT: {gt_mask.shape}"
            )

        # Add to successful samples
        if gt_mask is not None:
            successful_samples.append((name, rgb_img, gt_mask, pred_mask))
        else:
            successful_samples.append((name, rgb_img, pred_mask))
            
        print(
            f"  ✓ Successfully loaded {name}"
        )

        # Stop if we have enough samples
        if len(successful_samples) >= max_samples:
            print(
                f"Reached maximum samples limit ({max_samples})"
            )
            break

    print(
        f"\nSummary:"
    )
    print(
        f"  Successfully processed: {len(successful_samples)}"
    )
    print(
        f"  Failed: {failed}"
    )

    # Create batch plots if we have any successful samples
    if successful_samples:
        n_samples = len(successful_samples)
        n_batches = (
            (n_samples + images_per_plot - 1) // images_per_plot
        )
        has_ground_truth = len(successful_samples[0]) == 4  # 4 items means has GT
        print(
            f"\nCreating {n_batches} separate plots with "
            f"{n_samples} total samples..."
        )
        create_batch_plots(
            successful_samples, output_dir, images_per_plot, has_ground_truth,
            file_prefix=file_prefix  # Add this parameter
        )
        print(f"✓ Plots created successfully!")
    else:
        print("No successful samples to plot!")

    print(f"Output directory: {output_dir}")

def create_batch_plots(
    sample_data: list, output_dir: Path, images_per_plot: int = 5, 
    has_ground_truth: bool = True, file_prefix: str = ""
) -> None:
    """Create and save separate plots for batches of images."""
    total_samples = len(sample_data)
    num_batches = (total_samples + images_per_plot - 1) // images_per_plot

    n_rows = 3 if has_ground_truth else 2
    row_labels = [
        'RGB Image', 'Ground Truth', 'Prediction'
    ] if has_ground_truth else ['RGB Image', 'Prediction']

    print(
        f"Creating {num_batches} separate plots with up to "
        f"{images_per_plot} images each ({n_rows} rows per plot)..."
    )

    # Create consistent colormap for 3-class data
    # 0 = transparent (background), 1 = red (field interior), 
    # 2 = blue (field edge)
    from matplotlib.colors import ListedColormap
    colors = ['none', 'red', 'blue']  # 'none' makes 0 values transparent
    consistent_cmap = ListedColormap(colors)

    for batch_idx in range(num_batches):
        # Get samples for this batch
        start_idx = batch_idx * images_per_plot
        end_idx = min(start_idx + images_per_plot, total_samples)
        batch_samples = sample_data[start_idx:end_idx]
        n_samples_in_batch = len(batch_samples)

        print(
            f"  Creating batch {batch_idx + 1}/{num_batches} with "
            f"{n_samples_in_batch} samples..."
        )

        # Create figure
        fig_width = n_samples_in_batch * 4
        fig_height = n_rows * 4

        fig, axes = plt.subplots(
            n_rows, n_samples_in_batch, figsize=(fig_width, fig_height)
        )

        # Handle single sample case
        if n_samples_in_batch == 1:
            axes = axes.reshape(n_rows, 1)

        for col, sample_data_item in enumerate(batch_samples):
            if has_ground_truth:
                name, rgb_img, gt_mask, pred_mask = sample_data_item
            else:
                name, rgb_img, pred_mask = sample_data_item

            # RGB Image (top row)
            axes[0, col].imshow(rgb_img)
            axes[0, col].set_title(
                f'{name}', fontsize=10, pad=5
            )
            axes[0, col].axis('off')
            if col == 0:  # Add row label only to first column
                axes[0, col].text(
                    -0.1, 0.5, 'RGB Image', rotation=90,
                    transform=axes[0, col].transAxes,
                    verticalalignment='center', fontsize=12,
                    fontweight='bold'
                )

            # Ground Truth (middle row, if available)
            if has_ground_truth:
                # Get data info for debugging
                gt_min, gt_max = gt_mask.min(), gt_mask.max()
                gt_unique = np.unique(gt_mask)
                gt_nonzero_pct = np.count_nonzero(gt_mask) / gt_mask.size * 100
                
                # Count pixels for each class
                class_counts = {
                    0: np.sum(gt_mask == 0),
                    1: np.sum(gt_mask == 1), 
                    2: np.sum(gt_mask == 2)
                }
                
                print(
                    f"    GT {name}: min={gt_min}, max={gt_max}, "\
                        f"unique={gt_unique}"
                )
                print(
                    f"    GT {name}: class counts - bg:{class_counts[0]}, "
                    f"interior:{class_counts[1]}, edge:{class_counts[2]}"
                )
                
                # Show RGB as background, then overlay the mask
                axes[1, col].imshow(rgb_img)
                
                # Create mask overlay with consistent colors
                im_gt = axes[1, col].imshow(
                    gt_mask, cmap=consistent_cmap, vmin=0, vmax=2, alpha=0.7
                )
                
                axes[1, col].set_title(
                    f'GT (bg:{class_counts[0]}, int:{class_counts[1]}, '
                    f'edge:{class_counts[2]})', 
                    fontsize=8
                )
                    
                axes[1, col].axis('off')
                if col == 0:  # Add row label only to first column
                    axes[1, col].text(
                        -0.1, 0.5, 'Ground Truth', rotation=90,
                        transform=axes[1, col].transAxes,
                        verticalalignment='center', fontsize=12,
                        fontweight='bold'
                    )

            # Prediction (bottom row)
            pred_row = 2 if has_ground_truth else 1
            
            # Get data info for debugging
            pred_min, pred_max = pred_mask.min(), pred_mask.max()
            pred_unique = np.unique(pred_mask)
            pred_nonzero_pct = np.count_nonzero(pred_mask) / pred_mask.size * 100
            
            # Count pixels for each class
            class_counts = {
                0: np.sum(pred_mask == 0),
                1: np.sum(pred_mask == 1), 
                2: np.sum(pred_mask == 2)
            }
            
            print(
                f"    Pred {name}: min={pred_min}, max={pred_max}, "\
                    f"unique={pred_unique}"
            )
            print(
                f"    Pred {name}: class counts - bg:{class_counts[0]}, "
                f"interior:{class_counts[1]}, edge:{class_counts[2]}"
            )
            
            # Show RGB as background, then overlay the prediction
            axes[pred_row, col].imshow(rgb_img)
            
            # Create prediction overlay with consistent colors
            im_pred = axes[pred_row, col].imshow(
                pred_mask, cmap=consistent_cmap, vmin=0, vmax=2, alpha=0.7
            )
            
            axes[pred_row, col].set_title(
                f'Pred (bg:{class_counts[0]}, int:{class_counts[1]}, '
                f'edge:{class_counts[2]})', 
                fontsize=8
            )
                
            axes[pred_row, col].axis('off')
            if col == 0:  # Add row label only to first column
                axes[pred_row, col].text(
                    -0.1, 0.5, 'Prediction', rotation=90,
                    transform=axes[pred_row, col].transAxes,
                    verticalalignment='center', fontsize=12,
                    fontweight='bold'
                )

        # Add a legend for the colormap (only on the first plot of each batch)
        if batch_idx == 0:
            # Create a custom legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='none', edgecolor='black', 
                      label='Background (0)'),
                Patch(facecolor='red', label='Field Interior (1)'),
                Patch(facecolor='blue', label='Field Edge (2)')
            ]
            fig.legend(
                legend_elements, 
                ['Background (0)', 'Field Interior (1)', 'Field Edge (2)'], 
                loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3, 
                fontsize=10
            )

        plt.tight_layout()

        # Save plot with batch number and experiment prefix
        mode_suffix = "with_gt" if has_ground_truth else "pred_only"
        now = datetime.now()
        minutes = now.hour * 60 + now.minute
        date_str = f"{now.strftime('%Y%j')}{minutes:04d}"
        output_file = output_dir / (
            f"{file_prefix}inf_{mode_suffix}_"
            f"{date_str}_b{batch_idx + 1:03d}.png"
        )
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {output_file}")

def main():
    """Main function for standalone script execution."""
    
    parser = argparse.ArgumentParser(
        description='Visualize inference results vs ground truth'
    )
    parser.add_argument(
        '--catalog', type=str,
        default='data/ftw-ma-combined-inference-tester.csv',
        help='Path to CSV/GeoPackage catalog file'
    )
    parser.add_argument(
        '--data-dir', type=str,
        default='/Users/LEstes/data/labels/cropland/',
        help='Base directory containing image data'
    )
    parser.add_argument(
        '--inference-dir', type=str,
        default='external/results/inference',
        help='Directory containing inference results'
    )
    parser.add_argument(
        '--output-dir', type=str,
        default='external/results/figures',
        help='Output directory for comparison plots'
    )
    parser.add_argument(
        '--rgb-bands', type=int, nargs=3, default=[0, 1, 2],
        help='RGB band indices (0-indexed, default: 0 1 2 for red,green,blue)'
    )
    parser.add_argument(
        '--images-per-plot', type=int, default=5,
        help='Number of images per plot (default: 5)'
    )
    parser.add_argument(
        '--max-samples', type=int, default=20,
        help='Maximum number of samples to display (default: 20)'
    )

    args = parser.parse_args()

    # Setup paths
    catalog_path = Path(args.catalog)
    data_dir = Path(args.data_dir)
    inference_dir = Path(args.inference_dir)
    output_dir = Path(args.output_dir)

    create_inference_plots(
        catalog_path, data_dir, inference_dir, output_dir,
        tuple(args.rgb_bands), args.images_per_plot, args.max_samples
    )

if __name__ == "__main__":
    main()