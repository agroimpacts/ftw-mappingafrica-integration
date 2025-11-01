#!/usr/bin/env bash
#SBATCH --account=benq-tgirails         # <- change to your account
#SBATCH --partition=gpu                # or cpu, as appropriate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1              # set to 0 or remove if not using GPU
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=ftw-predict
#SBATCH --output=logs/predict/%x_%j.out
#SBATCH --error=logs/predict/%x_%j.err

set -euo pipefail

# Defaults (override with flags)
MODEL_CKPT=""
MODEL=""
DATA_DIR=""
CATALOG=""
NORM=${NORM:-"z_value"}
NORM_PROC=${NORM_PROC:-"gpb"}
OUTPUT_BASE=${OUTPUT_BASE:-"$HOME/working/models/outputs/predictions"}
PLOT_DIR=${PLOT_DIR:-"$HOME/working/models/outputs/plots"}
GLOBAL_STATS=${GLOBAL_STATS:-'{"mean":[0,0,0,0], "std":[3000,3000,3000,3000]}'}
CREATE_PLOTS=${CREATE_PLOTS:-"yes"}
BAND_ORDER=${BAND_ORDER:-""}
CROP_TO_GEOMETRY=${CROP_TO_GEOMETRY:-"no"}
MPS_MODE=${MPS_MODE:-"no"}
LOG_DIR=${LOG_DIR:-"$HOME/working/models/logs"}
DRY_RUN=${DRY_RUN:-"no"}

# Expand ~ helper
expand_path() { eval echo "$1"; }

# Parse args (same flags as run-predictions.sh)
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
            sed -n '1,240p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# Basic validation
if [[ -z "${DATA_DIR:-}" || -z "${CATALOG:-}" || -z "${MODEL_CKPT:-}" ]]; then
    echo "❌ Missing required arguments."
    echo "Usage: sbatch scripts/run-predictions-sbatch.sh -d <data_dir> -c <catalog> -m <model_ckpt> [options]"
    exit 1
fi

# Derive MODEL name if not provided
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
LOG_DIR=$(expand_path "$LOG_DIR")
DATA_DIR=$(expand_path "$DATA_DIR")
CATALOG=$(expand_path "$CATALOG")
MODEL_CKPT=$(expand_path "$MODEL_CKPT")

# Create output dirs (namespaced by model)
OUTPUT_DIR="${OUTPUT_BASE}/${MODEL}"
mkdir -p "$OUTPUT_DIR" "$PLOT_DIR" "$LOG_DIR"

# Prepare log file
TIMESTAMP=$(date +"%Y-%m-%dT%H-%M-%S")
LOG_FILE="${LOG_DIR}/predict_${MODEL}_${TIMESTAMP}.log"

# Build command (ftw_ma CLI)
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

# Log command and metadata
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
    echo "Command to run:"
    printf '%q ' "${CMD[@]}"
    echo
    echo "------------------------------------------------------------"
} >> "$LOG_FILE"

# Dry-run
if [[ "$DRY_RUN" == "yes" ]]; then
    echo "🧪 Dry run mode: command will not be executed." | tee -a "$LOG_FILE"
    printf '%q ' "${CMD[@]}" | tee -a "$LOG_FILE"
    exit 0
fi

# Activate environment (adapt to your environment)
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
pyenv activate ftw-mapafrica || {
  echo "❌ Failed to activate pyenv environment 'ftw-mapafrica'"; exit 1
}

# Execute (stream output to both terminal and logfile)
echo "🚀 Executing prediction command..." | tee -a "$LOG_FILE"
"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
RC=${PIPESTATUS[0]:-0}

# Final logging
if [[ $RC -eq 0 ]]; then
    echo "Run finished: $(date) - SUCCESS (rc=0)" >> "$LOG_FILE"
    echo "✅ Run completed successfully."
else
    echo "Run finished: $(date) - FAILURE (rc=$RC)" >> "$LOG_FILE"
    echo "❌ Run failed (rc=$RC). See $LOG_FILE for details."
fi

exit $RC
