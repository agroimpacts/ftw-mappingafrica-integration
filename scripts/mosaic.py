#!/usr/bin/env python3
"""
Create a mosaic COG from a directory of tiles.

Usage:
    python scripts/mosaic.py --in-dir ./tiles --out-cog ./mosaic.tif
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
import argparse
import sys
import rasterio
from tqdm.auto import tqdm
import shutil
import os
import textwrap


def tile_has_crs(path: Path) -> bool:
    try:
        with rasterio.open(path) as ds:
            return ds.crs is not None
    except Exception:
        return False


def run_cmd(cmd, desc=None):
    if desc:
        print(desc)
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if proc.returncode != 0:
        msg = (
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
        raise RuntimeError(msg)
    return proc.stdout

def _ensure_gdal_binaries():
    """Ensure GDAL CLI tools are on PATH; try common locations if not found."""
    # look for core tools; gdal_edit may be named differently on some installs
    required_core = ("gdalbuildvrt", "gdal_translate")
    edit_candidates = ("gdal_edit.py", "gdal_edit", "gdal_edit.py3")

    missing_core = [c for c in required_core if shutil.which(c) is None]
    found_edit = next((c for c in edit_candidates if shutil.which(c)), None)
    if not missing_core and found_edit:
        # everything present
        # expose chosen edit command globally for later use
        global GDAL_EDIT_CMD
        GDAL_EDIT_CMD = found_edit
        return

    # try common locations (add any site-specific paths here)
    common_dirs = [
        "/opt/software/usr/bin",
        "/usr/local/bin",
        "/usr/bin",
        "/opt/local/bin",
        os.path.expanduser("~/.local/bin"),
    ]
    for d in common_dirs:
        if os.path.isdir(d):
            os.environ["PATH"] = f"{d}:{os.environ.get('PATH','')}"
            missing_core = [c for c in required_core if shutil.which(c) is None]
            found_edit = next((c for c in edit_candidates \
                               if shutil.which(c)), None)
            if not missing_core and found_edit:
                global GDAL_EDIT_CMD
                GDAL_EDIT_CMD = found_edit
                return

    # still missing -> build helpful message
    missing = []
    missing.extend([c for c in required_core if shutil.which(c) is None])
    if not found_edit:
        missing.append("gdal_edit (any variant)")

    hints = [
        f"Missing GDAL CLI tools: {', '.join(missing)}",
        "",
        "Install or enable GDAL:",
        "  macOS (Homebrew): brew install gdal",
        "  Conda (recommended): conda install -c conda-forge gdal",
        "  On clusters: load the module, e.g. module load gdal/3.x",
        "",
        "If GDAL is already installed in a non-standard location, set PATH or",
        "export GDAL_BIN_DIR=/path/to/gdal/bin and re-run; this script tries",
        "common locations including /opt/software/usr/bin.",
        "",
        "Example:",
        "  export PATH=/opt/software/usr/bin:$PATH",
        "  python scripts/mosaic.py --in-dir ./tiles --out-cog ./mosaic.tif",
    ]
    raise RuntimeError("\n".join(textwrap.wrap("\n".join(hints), width=9999)))


def main(argv):
    # ensure GDAL CLI available (this will try /opt/software/usr/bin etc)
    _ensure_gdal_binaries()

    p = argparse.ArgumentParser(
        description="Build mosaic COG from tiles (uses GDAL CLI tools)."
    )
    p.add_argument(
        "--in-dir", "-i", required=True, help="Input directory with tiles"
    )
    p.add_argument(
        "--out-cog", "-o", required=True, help="Output COG path"
    )
    p.add_argument(
        "--pattern", default="*.tif", 
        help="Glob pattern for tiles (default: *.tif)",
    )
    p.add_argument(
        "--required-crs", default="EPSG:4326",
        help="Assign CRS only if missing (no reprojection)",
    )
    p.add_argument(
        "--src-nodata", default="255",
        help="Source nodata value (string)",
    )
    p.add_argument(
        "--cog-compress", default="ZSTD",
        help="COG compression (ZSTD, LZW, DEFLATE)",
    )
    p.add_argument(
        "--cog-blocksize", default="2000",
        help="COG blocksize",
    )
    p.add_argument(
        "--cog-predictor", default="2",
        help="COG predictor",
    )
    p.add_argument(
        "--cog-overviews", default="AUTO",
        help="COG overviews option",
    )
    p.add_argument(
        "--cog-threads", default="ALL_CPUS",
        help="COG threads option",
    )
    p.add_argument(
        "--force-tr", nargs=2, type=float, metavar=("TRX", "TRY"),
        help="Force resolution (tr x tr) for the mosaic",
    )
    p.add_argument(
        "--force-tap", action="store_true",
        help="Force tap option when building VRT",
    )
    p.add_argument(
        "--force-srs", help="Force SRS (e.g. EPSG:4326)",
    )
    args = p.parse_args(argv)

    in_dir = Path(args.in_dir)
    out_cog = Path(args.out_cog)
    pattern = args.pattern
    required_crs = args.required_crs
    src_nodata = args.src_nodata

    tiles = sorted(in_dir.glob(pattern))
    if not tiles:
        print(f"No files matching {pattern} in {in_dir}", file=sys.stderr)
        return 2
    print(f"[i] Found {len(tiles)} tiles in {in_dir}")

    print("[i] Checking/setting CRS (only if missing)...")
    for tf in tqdm(tiles, desc="CRS check", unit="file"):
        if not tile_has_crs(tf):
            # assign CRS metadata only (no pixel transform)
            run_cmd(
                ["gdal_edit.py", "-a_srs", required_crs, str(tf)],
                desc=f"Assigning CRS {required_crs} to {tf}",
            )

    out_cog.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=str(out_cog.parent)) as tmpdir:
        tmpdir = Path(tmpdir)
        file_list = tmpdir / "files.txt"
        vrt_path = tmpdir / "mosaic.vrt"

        file_list.write_text("\n".join(str(p) for p in tiles))
        print(f"[i] Wrote file list: {file_list}")

        vrt_cmd = [
            "gdalbuildvrt",
            "-input_file_list",
            str(file_list),
            str(vrt_path),
            "-srcnodata",
            src_nodata,
            "-vrtnodata",
            src_nodata,
        ]
        if args.force_tr:
            vrt_cmd += [
                "-resolution",
                "user",
                "-tr",
                str(args.force_tr[0]),
                str(args.force_tr[1]),
            ]
        if args.force_tap:
            vrt_cmd += ["-tap"]
        if args.force_srs:
            vrt_cmd += ["-a_srs", args.force_srs]

        run_cmd(vrt_cmd, desc="[i] Building VRT...")

        cog_cmd = [
            "gdal_translate",
            "-of",
            "COG",
            "-r",
            "nearest",
            "-a_nodata",
            src_nodata,
            "-co",
            f"BLOCKSIZE={args.cog_blocksize}",
            "-co",
            f"COMPRESS={args.cog_compress}",
            "-co",
            f"PREDICTOR={args.cog_predictor}",
            "-co",
            "BIGTIFF=IF_SAFER",
            "-co",
            f"OVERVIEWS={args.cog_overviews}",
            "-co",
            "RESAMPLING=NEAREST",
            "-co",
            f"NUM_THREADS={args.cog_threads}",
            str(vrt_path),
            str(out_cog),
        ]

        with tqdm(total=1, desc="COG translate", unit="step") as pbar:
            run_cmd(cog_cmd, desc="[i] Translating VRT -> COG...")
            pbar.update(1)

    print(f"COG written: {out_cog}")
    return 0


if __name__ == "__main__":
    try:
        rc = main(sys.argv[1:])
        sys.exit(rc)
    except Exception as exc:
        print("ERROR:", exc, file=sys.stderr)
        sys.exit(1)