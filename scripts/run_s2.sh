#!/bin/bash
# run_s2.sh — safe launcher for Sentinel-2 downloads

python -u s2_download_masked.py \
  --csv /home/airg/gessuman/projects/ftw-mappingafrica-integration/data/dry_run_primary_5tier/merged/primary_merged_dedup.csv \
  --planet-dir /home/airg/data/labels/cropland/mappingafrica-256/images \
  --outdir /home/airg/gessuman/projects/ftw-mappingafrica-integration/data/s2_chips/primary \
  --mask-threshold 10 \
  --checkpoint-every 1000 \
  --retries 4
