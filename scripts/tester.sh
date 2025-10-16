#!/usr/bin/env bash
set -euo pipefail

# Usage: ./tester.sh MOD [VERSION] [CTLG] [DATA_OVERRIDE]
#   MOD           - required (model name, e.g., "mymodel")
#   VERSION       - optional (integer or "latest", default: latest)
#   CTLG          - optional (catalog .csv path, default: data/ftw-catalog2.csv)
#   DATA_OVERRIDE - optional (path to override data.init_args.data_dir in config)

if [ $# -lt 1 ]; then
    echo "❌ Missing required argument: MOD"
    echo "Usage: $0 MOD [VERSION] [CTLG] [DATA_OVERRIDE]"
    exit 1
fi

MOD=$1
VERSION="latest"
CTLG="data/ftw-catalog2.csv"
DATA_OVERRIDE=""

# --- Parse arguments ---
if [ $# -ge 2 ]; then
    if [[ "$2" =~ ^[0-9]+$ || "$2" == "latest" ]]; then
        VERSION=$2
        if [ $# -ge 3 ]; then
            CTLG=$3
        fi
        if [ $# -ge 4 ]; then
            DATA_OVERRIDE=$4
        fi
    else
        CTLG=$2
        if [ $# -ge 3 ]; then
            DATA_OVERRIDE=$3
        fi
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

# --- Optional nested YAML override ---
TMP_CFG="$CFG"
if [ -n "$DATA_OVERRIDE" ]; then
    TMP_CFG=$(mktemp /tmp/${MOD}_cfg_XXXX.yaml)
    echo "📝 Creating temporary config with overridden data path: $TMP_CFG"

    python - <<PY
import yaml, sys
path = "$CFG"
out = "$TMP_CFG"
data_path = "$DATA_OVERRIDE"

with open(path) as f:
    cfg = yaml.safe_load(f) or {}

cfg.setdefault("data", {}).setdefault("init_args", {})["data_dir"] = data_path

with open(out, "w") as f:
    yaml.safe_dump(cfg, f, default_flow_style=False)
PY

    echo "🔍 Preview of modified config (first 10 lines):"
    head -n 10 "$TMP_CFG"
    echo "========================================"
fi

# --- Output path ---
CTLG_BASE=$(basename "$CTLG" .csv)
OUT="~/working/models/results/${MOD}-${CTLG_BASE}.csv"
OUT=$(eval echo "$OUT")

mkdir -p "$(dirname "$OUT")"

# --- Run ---
echo "✅ Running ftw_ma with:"
echo "   Config:    $TMP_CFG"
echo "   Checkpoint:$CHKPT"
echo "   Catalog:   $CTLG"
echo "   Output:    $OUT"

ftw_ma model test \
    -cfg "$TMP_CFG" \
    -m "$CHKPT" \
    -cat "$CTLG" \
    -spl validate \
    -o "$OUT"

# --- Cleanup ---
if [ "$TMP_CFG" != "$CFG" ]; then
    rm -f "$TMP_CFG"
fi

