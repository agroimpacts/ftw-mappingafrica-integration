#!/usr/bin/env python3
"""
Batch test multiple models on catalog subsets by country.

Usage:
  python run_tests.py \
    --models ftwbaseline-exp1 ftwbaseline-exp2 \
    --catalog data/ftw-catalog2.csv \
    --split validate \
    --countries Kenya Uganda \
    --data_dir /scratch/data
"""

import argparse
import subprocess
from pathlib import Path
import pandas as pd
import tempfile
import re
import os


def expand_path(path_str):
    """Expand ~ and environment variables in path."""
    return Path(os.path.expanduser(os.path.expandvars(path_str)))


def find_latest_version(model_dir):
    """Find latest version number in lightning_logs."""
    versions_dir = model_dir / "lightning_logs"
    if not versions_dir.exists():
        return None
    version_dirs = [
        d for d in versions_dir.iterdir()
        if d.is_dir() and d.name.startswith("version_")
    ]
    if not version_dirs:
        return None
    nums = []
    for d in version_dirs:
        m = re.search(r"version_(\d+)", d.name)
        if m:
            nums.append(int(m.group(1)))
    return max(nums) if nums else None


def find_checkpoint(checkpoint_dir):
    """Find checkpoint file, preferring last.ckpt."""
    last_ckpt = checkpoint_dir / "last.ckpt"
    if last_ckpt.exists():
        return last_ckpt
    epoch_ckpts = list(checkpoint_dir.glob("epoch=*.ckpt"))
    if epoch_ckpts:
        epoch_ckpts.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        print(f"⚠️ last.ckpt not found, using: {epoch_ckpts[0].name}")
        return epoch_ckpts[0]
    return None


def run_test(model, catalog, split="validate", countries=None, data_dir=None):
    """Run ftw_ma test on catalog subsets by country."""
    home_dir = Path.home()
    cfg_path = Path(model)
    if cfg_path.suffix != ".yaml":
        cfg_path = Path("configs") / f"{model}.yaml"
    if not cfg_path.exists():
        print(f"❌ Config file not found: {cfg_path}")
        return
    config_file = cfg_path

    model_dir = home_dir / "working" / "models" / model
    version_num = find_latest_version(model_dir)
    if version_num is None:
        print(f"❌ No version dirs found for {model}.")
        return

    checkpoint_dir = (
        model_dir / "lightning_logs" / f"version_{version_num}" / "checkpoints"
    )
    checkpoint_file = find_checkpoint(checkpoint_dir)
    if checkpoint_file is None:
        print(f"❌ No checkpoints found for {model}.")
        return

    catalog_path = expand_path(catalog)
    if not catalog_path.exists():
        print(f"❌ Catalog file not found: {catalog_path}")
        return

    df = pd.read_csv(catalog_path)
    if "country" not in df.columns:
        print(f"❌ Catalog has no 'country' column: {catalog_path}")
        return

    output_dir = home_dir / "working" / "models" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog_base = catalog_path.stem
    combined_path = output_dir / f"{model}-{catalog_base}-{split}-combined.csv"

    # Determine countries to loop over
    if countries is None:
        country_list = sorted(df["country"].dropna().unique())
    elif len(countries) == 1 and countries[0].lower() == "all":
        country_list = ["all"]
    else:
        country_list = countries

    combined_df = pd.DataFrame()

    for country in country_list:
        subset_df = df if country == "all" else df[df["country"] == country]
        if subset_df.empty:
            print(f"⚠️ No records for {country}, skipping.")
            continue

        tmpfile = Path(tempfile.mkstemp(suffix=".csv")[1])
        subset_df.to_csv(tmpfile, index=False)

        print(f"🌍 Testing {model} on {country} ({len(subset_df)} rows)")

        cmd = [
            "ftw_ma", "model", "test",
            "-cfg", str(config_file),
            "-m", str(checkpoint_file),
            "-cat", str(tmpfile),
            "-spl", split
        ]
        if data_dir:
            cmd.extend(["-d", str(data_dir)])

        try:
            subprocess.run(cmd, check=True)
            new_data = pd.read_csv(tmpfile)  # Read results (or real output)
            combined_df = pd.concat([combined_df, new_data], ignore_index=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ {country} failed (exit {e.returncode})")
        finally:
            tmpfile.unlink(missing_ok=True)

    combined_df.to_csv(combined_path, index=False)
    print(f"🏁 All subsets done for {model}.")
    print(f"📁 Combined results: {combined_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch test models on catalog subsets.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--models", nargs="+", required=True,
        help="Model names (e.g., ftwbaseline-exp1) or full config paths"
    )
    parser.add_argument("--catalog", required=True, help="Path to catalog CSV")
    parser.add_argument(
        "--split", default="validate", help="Dataset split (default: validate)"
    )
    parser.add_argument(
        "--countries", nargs="+",
        help="Country list (e.g., Kenya Uganda) or 'all'"
    )
    parser.add_argument(
        "--data_dir", "-d", default=None,
        help="Optional data directory override for ftw_ma"
    )

    args = parser.parse_args()
    for model in args.models:
        run_test(model, args.catalog, args.split, args.countries, args.data_dir)


if __name__ == "__main__":
    main()
