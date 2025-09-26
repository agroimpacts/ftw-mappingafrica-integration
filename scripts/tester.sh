#!/usr/bin/env bash
set -euo pipefail

# Usage: ./tester.sh MOD [VERSION] [CTLG]
#   MOD      - required (model name, e.g., "mymodel")
#   VERSION  - optional (integer or "latest", default: latest)
#   CTLG     - optional (catalog .csv path, default: data/ftw-catalog2.csv)

if [ $# -lt 1 ]; then
    echo "❌ Missing required argument: MOD"
    echo "Usage: $0 MOD [VERSION] [CTLG]"
    exit 1
fi

MOD=$1
VERSION="latest"
CTLG="data/ftw-catalog2.csv"

# --- Parse arguments ---
if [ $# -ge 2 ]; then
    if [[ "$2" =~ ^[0-9]+$ || "$2" == "latest" ]]; then
        VERSION=$2
        if [ $# -ge 3 ]; then
            CTLG=$3
        fi
    else
        CTLG=$2
    fi
fi

# --- Resolve version ---
if [ "$VERSION" = "latest" ]; then
    VERSION=$(ls -d ~/working/models/$MOD/lightning_logs/version_* 2>/dev/null \
                | sed -E 's/.*version_([0-9]+)/\1/' \
                | sort -n \
                | tail -1 || true)
    if [ -z "$VERSION" ]; then
        echo "❌ No version directories found for $MOD."
        exit 1
    fi
fi

CFG="configs/$MOD.yaml"
CHKPT_DIR="~/working/models/$MOD/lightning_logs/version_$VERSION/checkpoints"
CHKPT="$CHKPT_DIR/last.ckpt"

# Expand ~ in paths
CHKPT_DIR=$(eval echo "$CHKPT_DIR")
CHKPT=$(eval echo "$CHKPT")
CTLG=$(eval echo "$CTLG")

# --- Validations ---
if [ ! -f "$CFG" ]; then
    echo "❌ Config file not found: $CFG"
    exit 1
fi

# Fallback if last.ckpt is missing
if [ ! -f "$CHKPT" ]; then
    echo "⚠️ last.ckpt not found, falling back to most recent checkpoint..."
    if ls "$CHKPT_DIR"/epoch=*.ckpt 1>/dev/null 2>&1; then
        CHKPT=$(ls -t "$CHKPT_DIR"/epoch=*.ckpt | head -1)
        echo "   Using checkpoint: $CHKPT"
    else
        echo "❌ No checkpoints found in $CHKPT_DIR"
        exit 1
    fi
fi

if [ ! -f "$CTLG" ]; then
    echo "❌ Catalog file not found: $CTLG"
    exit 1
fi

# Build OUT name as MOD-CTLGbase.csv
CTLG_BASE=$(basename "$CTLG" .csv)
OUT="~/working/models/results/${MOD}-${CTLG_BASE}.csv"
OUT=$(eval echo "$OUT")

# Ensure output directory exists
mkdir -p "$(dirname "$OUT")"

# --- Run ---
echo "✅ Running ftw_ma with:"
echo "   Config:    $CFG"
echo "   Checkpoint:$CHKPT"
echo "   Catalog:   $CTLG"
echo "   Output:    $OUT"

ftw_ma model test \
    -cfg "$CFG" \
    -m "$CHKPT" \
    -cat "$CTLG" \
    -spl validate \
    -o "$OUT"
