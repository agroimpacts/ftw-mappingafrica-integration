#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --job-name=ftw-predict
#SBATCH --output=logs/predict/%x_%j.out
#SBATCH --error=logs/predict/%x_%j.err

# Unified runner: can be executed directly or submitted with sbatch.
# Usage examples:
#   Direct:
#     ./scripts/run-predictions.sh -d /data -c catalog.geojson \
#       -m /path/to/model.ckpt
#   Slurm:
#     sbatch scripts/run-predictions.sh -d /data -c catalog.geojson \
#       -m /path/to/model.ckpt
#
# Config file support: -f config.yml (or config.json).
# CLI args override config values.

set -euo pipefail

# ==================== Defaults ====================
MODEL=""
MODEL_CKPT=""
NORM=${NORM:-"z_value"}
NORM_PROC=${NORM_PROC:-"gpb"}
OUTPUT_BASE=${OUTPUT_BASE:-"~/working/models/outputs/predictions"}
PLOT_DIR=${PLOT_DIR:-"~/working/models/outputs/plots"}
GLOBAL_STATS=${GLOBAL_STATS:-"{\"mean\":[0,0,0,0], \
\"std\":[3000,3000,3000,3000]}"}
CREATE_PLOTS=${CREATE_PLOTS:-"yes"}
BAND_ORDER=${BAND_ORDER:-""}
CROP_TO_GEOMETRY=${CROP_TO_GEOMETRY:-"no"}
MPS_MODE=${MPS_MODE:-"no"}
LOG_FILE=${LOG_FILE:-"~/working/models/logs/predict_runs.log"}
DRY_RUN="no"
CONFIG_FILE=""

# Expand ~ helper
expand_path() {
    eval echo "$1"
}

# ==================== Parse Args ====================
while [[ $# -gt 0 ]]; do
    key="$1"
    case "$key" in
        -f|--config) CONFIG_FILE="$2"; shift 2 ;;
        -d|--data) DATA_DIR="$2"; shift 2 ;;
        -c|--catalog) CATALOG="$2"; shift 2 ;;
        -m|--model) MODEL_CKPT="$2"; shift 2 ;;
        -M|--model-name) MODEL="$2"; shift 2 ;;
        -p|--plot-dir) PLOT_DIR="$2"; shift 2 ;;
        -n|--norm) NORM="$2"; shift 2 ;;
        -np|--norm-proc) NORM_PROC="$2"; shift 2 ;;
        -g|--global-stats) GLOBAL_STATS="$2"; shift 2 ;;
        -bo|--band-order) BAND_ORDER="$2"; shift 2 ;;
        -crop|--crop) CROP_TO_GEOMETRY="$2"; shift 2 ;;
        -mps|--mps) MPS_MODE="$2"; shift 2 ;;
        -plot|--plot) CREATE_PLOTS="$2"; shift 2 ;;
        -dr|--dry-run) DRY_RUN="yes"; shift 1 ;;
        -h|--help)
            sed -n '1,240p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# ==================== Load config file (if provided) ====================
if [[ -n "${CONFIG_FILE:-}" ]]; then
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo "Config file not found: $CONFIG_FILE" >&2
        exit 1
    fi

    # Parse config with python: produce __CFG_<UPPER>=value shell vars
    mapfile -t __CFG_LINES < <(python3 - "$CONFIG_FILE" <<'PY'
import sys, json, os
p = sys.argv[1]
cfg = {}
try:
    import yaml
    cfg = yaml.safe_load(open(p))
except Exception:
    try:
        cfg = json.load(open(p))
    except Exception as e:
        print(f"ERROR_PARSING_CONFIG:{e}")
        sys.exit(2)
if not isinstance(cfg, dict):
    cfg = {}
out=[]
for k,v in cfg.items():
    name = k.upper()
    if v is None:
        sval = ""
    elif isinstance(v, bool):
        sval = "yes" if v else "no"
    else:
        sval = str(v)
    # escape single quotes
    sval = sval.replace("'", "'\"'\"'")
    out.append(f"__CFG_{name}='{sval}'")
print("\n".join(out))
PY
)
    # handle parse errors
    for l in "${__CFG_LINES[@]}"; do
        if [[ "$l" == ERROR_PARSING_CONFIG:* ]]; then
            echo "Failed to parse config: ${l#ERROR_PARSING_CONFIG:}" >&2
            exit 1
        fi
    done

    # evaluate lines to set __CFG_* variables
    for l in "${__CFG_LINES[@]}"; do
        eval "$l"
    done

    # helper to copy config values into script vars only when unset
    set_from_cfg_if_unset() {
        local varname="$1" cfgvar="__CFG_${2:-$1}"
        if [[ -n "${!cfgvar:-}" && -z "${!varname:-}" ]]; then
            eval "$varname=\"\${$cfgvar}\""
        fi
    }

    # map config keys -> script variables (only set if CLI did not set
    # them)
    set_from_cfg_if_unset DATA_DIR DATA_DIR
    set_from_cfg_if_unset CATALOG CATALOG
    set_from_cfg_if_unset MODEL_CKPT MODEL_CKPT
    set_from_cfg_if_unset MODEL MODEL
    set_from_cfg_if_unset OUTPUT_BASE OUTPUT_BASE
    set_from_cfg_if_unset PLOT_DIR PLOT_DIR
    set_from_cfg_if_unset GLOBAL_STATS GLOBAL_STATS
    set_from_cfg_if_unset NORM NORM
    set_from_cfg_if_unset NORM_PROC NORM_PROC
    set_from_cfg_if_unset BAND_ORDER BAND_ORDER
    set_from_cfg_if_unset CROP_TO_GEOMETRY CROP_TO_GEOMETRY
    set_from_cfg_if_unset MPS_MODE MPS_MODE
    set_from_cfg_if_unset CREATE_PLOTS CREATE_PLOTS
    set_from_cfg_if_unset LOG_FILE LOG_FILE
    set_from_cfg_if_unset DRY_RUN DRY_RUN
fi

# ==================== Validation ====================
if [[ -z "${DATA_DIR:-}" || -z "${CATALOG:-}" || -z "${MODEL_CKPT:-}"
]]; then
    echo "❌ Missing required arguments (either via CLI or config)."
    echo "Provide -d <data_dir> -c <catalog> -m <model_ckpt> or supply"
    echo "them in the config file."
    exit 1
fi

# ==================== Derive Model Name ====================
if [[ -z "${MODEL:-}" ]]; then
    if [[ "$MODEL_CKPT" == *"/lightning_logs/"* ]]; then
        MODEL_DIR="${MODEL_CKPT%%/lightning_logs/*}"
        MODEL=$(basename "$MODEL_DIR")
    else
        MODEL=$(
            basename "$(dirname "$(dirname "$(dirname \
                "$MODEL_CKPT")")")" 2>/dev/null \
            || echo "$(basename "$MODEL_CKPT")"
        )
    fi
fi

# Expand paths
OUTPUT_BASE=$(expand_path "$OUTPUT_BASE")
PLOT_DIR=$(expand_path "$PLOT_DIR")
LOG_FILE=$(expand_path "$LOG_FILE")
DATA_DIR=$(expand_path "$DATA_DIR")
CATALOG=$(expand_path "$CATALOG")
MODEL_CKPT=$(expand_path "$MODEL_CKPT")

# Output dir namespaced by model
OUTPUT_DIR="${OUTPUT_BASE%/}/${MODEL}"
mkdir -p "$OUTPUT_DIR" "$(dirname "$LOG_FILE")" "$PLOT_DIR"

# ==================== Build Command ====================
# Ensure JSON/global-stats is passed as a single argument (quoted)
CMD=(
    ftw_ma
    model
    predict
    -c "$CATALOG"
    -m "$MODEL_CKPT"
    -o "$OUTPUT_DIR"
    --path_column
    window_b
    --id_column
    name
    --normalization_strategy
    "$NORM"
    --normalization_stat_procedure
    "$NORM_PROC"
)

if [[ "$GLOBAL_STATS" != "none" && -n "$GLOBAL_STATS" ]]; then
    CMD+=(--global_stats "$GLOBAL_STATS")
fi

CMD+=(
    -d "$DATA_DIR"
    --overwrite
)

if [[ -n "$BAND_ORDER" ]]; then
    CMD+=(--band_order "$BAND_ORDER")
fi

if [[ "$CROP_TO_GEOMETRY" =~ ^(yes|true|1)$ ]]; then
    CMD+=(--crop_to_geometry)
fi

if [[ "$MPS_MODE" =~ ^(yes|true|1)$ ]]; then
    CMD+=(--mps_mode)
fi

CMD+=(--date_column date)

if [[ "$CREATE_PLOTS" =~ ^(yes|true|1)$ ]]; then
    # attach model namespace to plot output dir
    CMD+=(--create_plots)
    CMD+=(--plot_output_dir "${PLOT_DIR%/}/${MODEL}")
fi

# ==================== Logging (metadata + command only) ============
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
{
    echo "============================================================"
    echo "Run started: $TIMESTAMP"
    echo "Model: $MODEL"
    echo "Model checkpoint: $MODEL_CKPT"
    echo "Data directory: $DATA_DIR"
    echo "Catalog: $CATALOG"
    echo "Plot output: ${PLOT_DIR%/}/${MODEL}"
    echo "Create plots: $CREATE_PLOTS"
    echo "Normalization: $NORM ($NORM_PROC)"
    echo "Global stats: ${GLOBAL_STATS:-none}"
    echo "Band order: ${BAND_ORDER:-none}"
    echo "Crop to geometry: ${CROP_TO_GEOMETRY:-no}"
    echo "MPS mode: ${MPS_MODE:-no}"
    echo "Output directory: $OUTPUT_DIR"
    echo "Dry run: $DRY_RUN"
    echo "------------------------------------------------------------"
    echo "Command executed:"
    printf '%q ' "${CMD[@]}"
    echo
    echo "------------------------------------------------------------"
} >> "$LOG_FILE"

# ==================== Dry Run Mode ====================
if [[ "$DRY_RUN" == "yes" ]]; then
    echo "🧪 Dry run mode: command will not be executed."
    echo "------------------------------------------------------------"
    printf '%q ' "${CMD[@]}"
    echo
    echo "------------------------------------------------------------"
    {
        echo "Run finished: $(date +"%Y-%m-%d %H:%M:%S")"
        echo "Status: DRY-RUN (not executed)"
        echo "============================================================"
    } >> "$LOG_FILE"
    exit 0
fi

# ==================== Display Summary ====================
cat <<EOF
🚀 Running ftw_ma model predict...
Model: $MODEL
Model checkpoint: $MODEL_CKPT
Data directory: $DATA_DIR
Catalog: $CATALOG
Plot output: ${PLOT_DIR%/}/${MODEL}
Create plots: $CREATE_PLOTS
Normalization: $NORM ($NORM_PROC)
Global stats: ${GLOBAL_STATS:-none}
Band order: ${BAND_ORDER:-none}
Crop to geometry: ${CROP_TO_GEOMETRY:-no}
MPS mode: ${MPS_MODE:-no}
Output directory: $OUTPUT_DIR
Log file: $LOG_FILE
EOF

# ==================== Execute (live output to terminal) =============
# If running under sbatch, environment is already provisioned;
# otherwise user env is used.
"${CMD[@]}"
RC=$?

# ==================== Log result (status only) =======================
if [[ $RC -eq 0 ]]; then
    {
        echo "Run finished: $(date +"%Y-%m-%d %H:%M:%S")"
        echo "Status: SUCCESS (rc=0)"
        echo "============================================================"
    } >> "$LOG_FILE"
    echo "✅ Run completed successfully."
else
    {
        echo "Run finished: $(date +"%Y-%m-%d %H:%M:%S")"
        echo "Status: FAILURE (rc=$RC)"
        echo "============================================================"
    } >> "$LOG_FILE"
    echo "❌ Run failed (rc=$RC). See terminal output for details."
fi

echo "✅ Prediction complete. Log updated: $LOG_FILE"
exit $RC