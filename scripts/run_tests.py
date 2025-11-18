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
import numpy as np


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
        epoch_ckpts.sort(
            key=lambda x: x.stat().st_mtime, reverse=True
        )
        print(f"⚠️ last.ckpt not found, using: {epoch_ckpts[0].name}")
        return epoch_ckpts[0]
    return None


def run_test(model, catalog, split="validate", countries=None, data_dir=None):
    """Run ftw_ma test on catalog subsets by country."""
    home_dir = Path.home()
    cfg_path = Path(model)
    
    # If model contains a path separator, treat it as a path
    if "/" in model or cfg_path.suffix == ".yaml":
        # Ensure it has .yaml extension
        if cfg_path.suffix != ".yaml":
            cfg_path = cfg_path.with_suffix(".yaml")
        
        config_file = cfg_path
        model_name = cfg_path.stem  # For output filenames
        
        # For checkpoint lookup, strip "configs/" prefix and .yaml suffix
        checkpoint_subpath = cfg_path.with_suffix("").as_posix()
        if checkpoint_subpath.startswith("configs/"):
            checkpoint_subpath = checkpoint_subpath[8:]  # Remove "configs/"
    else:
        # model is just a name like "ftwbaseline-exp1"
        model_name = model
        checkpoint_subpath = model
        config_file = Path("configs") / f"{model}.yaml"
    
    if not config_file.exists():
        print(f"❌ Config file not found: {config_file}")
        return

    # Use checkpoint_subpath to preserve directory structure
    model_dir = home_dir / "working" / "models" / checkpoint_subpath
    version_num = find_latest_version(model_dir)
    if version_num is None:
        print(f"❌ No version dirs found for {checkpoint_subpath}.")
        print(f"   Expected checkpoint dir: {model_dir}")
        return

    checkpoint_dir = (
        model_dir / "lightning_logs" / f"version_{version_num}" / "checkpoints"
    )
    checkpoint_file = find_checkpoint(checkpoint_dir)
    if checkpoint_file is None:
        print(f"❌ No checkpoints found in {checkpoint_dir}")
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
    combined_path = output_dir / f"{model_name}-{catalog_base}-{split}-combined.csv"

    # Determine countries to loop over
    if countries is None:
        country_list = sorted(df["country"].dropna().unique())
    elif len(countries) == 1 and countries[0].lower() == "all":
        country_list = ["all"]
    else:
        country_list = countries

    summaries = []  # collect one-row summaries

    for country in country_list:
        subset_df = df if country == "all" else df[df["country"] == country]
        if subset_df.empty:
            print(f"⚠️ No records for {country}, skipping.")
            continue

        # prepare tmp input catalog for this subset
        # Use NamedTemporaryFile(delete=False) and close it so other
        # processes can open it reliably.
        tmp_in_f = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp_in = Path(tmp_in_f.name)
        tmp_in_f.close()
        subset_df.to_csv(tmp_in, index=False)

        # prepare tmp output path (do NOT create an empty file ahead
        # of time; let ftw_ma write it)
        tmp_out_f = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp_out = Path(tmp_out_f.name)
        tmp_out_f.close()
        # remove the empty file so ftw_ma can create/write it (some
        # tools refuse to overwrite an existing empty file)
        try:
            tmp_out.unlink(missing_ok=True)
        except Exception:
            pass

        print(f"🌍 Testing {model_name} on {country} ({len(subset_df)} rows)")

        cmd = [
            "ftw_ma", "model", "test",
            "-cfg", str(config_file),
            "-m", str(checkpoint_file),
            "-cat", str(tmp_in),
            "-spl", split,
            "-o", str(tmp_out),
        ]
        if data_dir:
            cmd.extend(["-d", str(data_dir)])

        try:
            # capture stdout/stderr to help debug empty outputs
            proc = subprocess.run(
                cmd, check=True, capture_output=True, text=True
            )
            if proc.stdout:
                print("ftw_ma stdout:", proc.stdout.strip())
            if proc.stderr:
                print("ftw_ma stderr:", proc.stderr.strip())

            if not tmp_out.exists():
                print(
                    f"⚠️ Expected output not produced for {country}: {tmp_out}"
                )
                # print captured output again for debugging
                print(
                    "ftw_ma may have exited successfully but didn't write the"
                    " output file."
                )
                continue

            out_df = pd.read_csv(tmp_out)
            if out_df.empty:
                print(f"⚠️ ftw_ma produced empty output for {country}")
                # dump stdout/stderr for troubleshooting
                print("ftw_ma stdout:", proc.stdout.strip())
                print("ftw_ma stderr:", proc.stderr.strip())
                continue

            out_df["model"] = model_name
            out_df["country"] = (country if country != "all" else "ALL")

            summaries.append(out_df)

        except subprocess.CalledProcessError as e:
            # print captured output to help debugging
            stdout = (
                e.stdout.decode() if isinstance(e.stdout, bytes) 
                else (e.stdout or "")
            )
            stderr = (
                e.stderr.decode() if isinstance(e.stderr, bytes) 
                else (e.stderr or "")
            )
            print(f"❌ {country} failed (exit {e.returncode})")
            if stdout:
                print("ftw_ma stdout:", stdout.strip())
            if stderr:
                print("ftw_ma stderr:", stderr.strip())
        finally:
            # cleanup temporary files (input + output)
            try:
                tmp_in.unlink(missing_ok=True)
            except Exception:
                pass
            try:
                tmp_out.unlink(missing_ok=True)
            except Exception:
                pass

    # build combined summary df and write one CSV with one row per country
    if summaries:
        combined_summary_df = pd.concat(summaries, ignore_index=True)
        combined_summary_df.to_csv(combined_path, index=False)
        print(f"✅ Combined summary written: {combined_path}")
    else:
        print("⚠️ No summaries produced; nothing written.")

    print(f"🏁 All subsets done for {model_name}.")
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
        "--split", default="validate",
        help="Dataset split (default: validate)"
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
