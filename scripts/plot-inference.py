#!/usr/bin/env python3
"""
Script to visualize inference results alongside original images.
Reads CSV catalog and matches with inference outputs to create comparison
plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import rasterio
import argparse
import os
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

def load_and_normalize_image(
    path: Path, bands: Tuple[int, int, int] = (2, 1, 0)
) -> np.ndarray:
    """Load image and normalize for RGB display."""
    try:
        with rasterio.open(path) as src:
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

            return img
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None

def load_inference_result(path: Path) -> Optional[np.ndarray]:
    """Load inference result (predicted mask)."""
    try:
        with rasterio.open(path) as src:
            pred = src.read(1)  # Assume single band
            return pred
    except Exception as e:
        print(f"Error loading inference result {path}: {e}")
        return None

def load_ground_truth_mask(
    path: Path, data_dir: Path
) -> Optional[np.ndarray]:
    """Load ground truth mask."""
    try:
        full_path = data_dir / path
        with rasterio.open(full_path) as src:
            mask = src.read(1)
            return mask
    except Exception as e:
        print(f"Error loading ground truth mask {full_path}: {e}")
        return None

def get_image_path(row: pd.Series, data_dir: Path) -> Optional[Path]:
    """Get the correct image path based on dataset type."""
    if row['dataset'] == 'ftw':
        # Use window_b for FTW dataset
        img_path = data_dir / row['window_b']
    else:
        # Use window_b for mapping africa (same column used)
        img_path = data_dir / row['window_b']

    return img_path if img_path.exists() else None

def create_batch_plots(
    sample_data: list, output_dir: Path, images_per_plot: int = 5
) -> None:
    """Create and save separate plots for batches of images."""
    total_samples = len(sample_data)
    num_batches = (total_samples + images_per_plot - 1) // images_per_plot

    print(
        f"Creating {num_batches} separate plots with up to "
        f"{images_per_plot} images each..."
    )

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

        # Create figure: 3 rows (RGB, GT, Pred) x n_samples columns
        fig_width = n_samples_in_batch * 3
        fig_height = 9  # 3 rows, 3 units high each

        fig, axes = plt.subplots(
            3, n_samples_in_batch, figsize=(fig_width, fig_height)
        )

        # Handle single sample case
        if n_samples_in_batch == 1:
            axes = axes.reshape(3, 1)

        for col, (name, rgb_img, gt_mask, pred_mask) in enumerate(
            batch_samples
        ):
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

            # Ground Truth (middle row)
            axes[1, col].imshow(
                gt_mask, cmap='gray', vmin=0,
                vmax=max(1, gt_mask.max())
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
            axes[2, col].imshow(
                pred_mask, cmap='gray', vmin=0,
                vmax=max(1, pred_mask.max())
            )
            axes[2, col].axis('off')
            if col == 0:  # Add row label only to first column
                axes[2, col].text(
                    -0.1, 0.5, 'Prediction', rotation=90,
                    transform=axes[2, col].transAxes,
                    verticalalignment='center', fontsize=12,
                    fontweight='bold'
                )

        plt.tight_layout()

        # Save plot with batch number
        output_file = output_dir / (
            f'inference_comparison_batch_{batch_idx + 1:03d}.png'
        )
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description='Visualize inference results vs ground truth'
    )
    parser.add_argument(
        '--catalog', type=str,
        default='data/ftw-ma-combined-inference-tester.csv',
        help='Path to CSV catalog file'
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

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load catalog
    if not catalog_path.exists():
        print(f"Error: Catalog file not found: {catalog_path}")
        return

    df = pd.read_csv(catalog_path)
    print(f"Loaded catalog with {len(df)} samples")

    # Process each sample and collect successful ones
    successful_samples = []
    failed = 0

    for idx, row in df.iterrows():
        name = row['name']
        print(f"\nProcessing {name} ({idx+1}/{len(df)})")

        # Find inference result
        inference_patterns = [
            inference_dir / f'{name}.tif',
            inference_dir / f'{name}_prediction.tif',
            inference_dir / f'{name}_pred.tif',
        ]

        inference_path = None
        for pattern in inference_patterns:
            if pattern.exists():
                inference_path = pattern
                break

        if inference_path is None:
            print(f"  Warning: No inference result found for {name}")
            failed += 1
            continue

        # Load RGB image
        img_path = get_image_path(row, data_dir)
        if img_path is None:
            print(f"  Warning: Image not found for {name}")
            failed += 1
            continue

        rgb_img = load_and_normalize_image(
            img_path, tuple(args.rgb_bands)
        )
        if rgb_img is None:
            failed += 1
            continue

        # Load ground truth mask
        gt_mask = load_ground_truth_mask(
            Path(row['mask']), data_dir
        )
        if gt_mask is None:
            failed += 1
            continue

        # Load inference result
        pred_mask = load_inference_result(inference_path)
        if pred_mask is None:
            failed += 1
            continue

        # Check dimensions match
        if (
            rgb_img.shape[:2] != gt_mask.shape or
            gt_mask.shape != pred_mask.shape
        ):
            print(f"  Warning: Dimension mismatch for {name}")
            print(
                f"    RGB: {rgb_img.shape[:2]}, "
                f"GT: {gt_mask.shape}, Pred: {pred_mask.shape}"
            )
            failed += 1
            continue

        # Add to successful samples
        successful_samples.append((name, rgb_img, gt_mask, pred_mask))
        print(f"  ✓ Successfully loaded {name}")

        # Stop if we have enough samples
        if len(successful_samples) >= args.max_samples:
            print(f"Reached maximum samples limit ({args.max_samples})")
            break

    print(f"\nSummary:")
    print(f"  Successfully processed: {len(successful_samples)}")
    print(f"  Failed: {failed}")

    # Create batch plots if we have any successful samples
    if successful_samples:
        n_samples = len(successful_samples)
        n_batches = (
            (n_samples + args.images_per_plot - 1) // args.images_per_plot
        )
        print(
            f"\nCreating {n_batches} separate plots with "
            f"{n_samples} total samples..."
        )
        create_batch_plots(
            successful_samples, output_dir, args.images_per_plot
        )
    else:
        print("No successful samples to plot!")

    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()