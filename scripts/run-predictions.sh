#!/usr/bin/env bash
# Usage:
# ./run_predict.sh -d <data_dir> -c <catalog> -m <model_ckpt> \
#   [-M <model_name>] [-p <plot_output_dir>] [-n norm] [-np norm_proc] \
#   [-g '{"mean":[...],"std":[...]}'] [-bo <band_order>] [-crop yes/no] \
#   [-mps yes/no] [-plot yes/no] [--dry-run]
#
# Example:
# ./run_predict.sh \
#   -d ~/images/tiles \
#   -c data/mappingafrica-tile-catalog-small.geojson \
#   -m ~/working/models/fullcat-ftwbaseline-exp1/lightning_logs/version_1/\
#     checkpoints/last.ckpt \
#   -n z_value -np gpb \
#   -g '{"mean":[0,0,0,0], "std":[3000,3000,3000,3000]}' \
#   -bo bgr_to_rgb -crop yes -mps yes -plot yes

set -euo pipefail

# ==================== Defaults ====================
MODEL=""
NORM=${NORM:-"z_value"}
NORM_PROC=${NORM_PROC:-"gpb"}
OUTPUT_BASE=${OUTPUT_BASE:-"~/working/models/outputs/predictions"}
PLOT_DIR=${PLOT_DIR:-"~/working/models/outputs/plots"}
GLOBAL_STATS=${GLOBAL_STATS:-'{"mean":[0,0,0,0], "std":[3000,3000,3000,3000]}'}
CREATE_PLOTS=${CREATE_PLOTS:-"yes"}
BAND_ORDER=${BAND_ORDER:-""}
CROP_TO_GEOMETRY=${CROP_TO_GEOMETRY:-"no"}
MPS_MODE=${MPS_MODE:-"no"}
LOG_FILE=${LOG_FILE:-"~/working/models/logs/predict_runs.log"}
DRY_RUN="no"

# Expand ~ helper
expand_path() {
    eval echo "$1"
}

# ==================== Parse Args ====================
while [[ $# -gt 0 ]]; do
    case "$1" in
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
            sed -n '1,120p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# ==================== Validation ====================
if [[ -z "${DATA_DIR:-}" || -z "${CATALOG:-}" || -z "${MODEL_CKPT:-}" ]]; then
    echo "❌ Missing required arguments."
    echo "Usage: ./run_predict.sh -d <data_dir> -c <catalog> -m <model_ckpt>"
    exit 1
fi

# ==================== Derive Model Name ====================
if [[ -z "$MODEL" ]]; then
    if [[ "$MODEL_CKPT" == *"/lightning_logs/"* ]]; then
        MODEL_DIR="${MODEL_CKPT%%/lightning_logs/*}"
        MODEL=$(basename "$MODEL_DIR")
    else
        MODEL=$(basename "$(dirname "$(dirname \
            "$(dirname "$(dirname "$MODEL_CKPT")")")")")
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
OUTPUT_DIR="${OUTPUT_BASE}"
mkdir -p "$OUTPUT_DIR" "$(dirname "$LOG_FILE")" "$PLOT_DIR"

# ==================== Build Command ====================
CMD=(
    ftw_ma model predict
    -c "$CATALOG"
    -m "$MODEL_CKPT"
    -o "$OUTPUT_DIR"
    --path_column window_b
    --id_column name
    --normalization_strategy "$NORM"
    --normalization_stat_procedure "$NORM_PROC"
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
    CMD+=(--create_plots --plot_output_dir "$PLOT_DIR")
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
    echo "Plot output: $PLOT_DIR"
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
Plot output: $PLOT_DIR
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
