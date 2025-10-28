#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --job-name=ftw-predict
#SBATCH --output=logs/predict/%x_%j.out
#SBATCH --error=logs/predict/%x_%j.err

#!/usr/bin/env bash
# Prediction runner.
# Usage examples:
#  Direct: ./scripts/run-predict.sh -d /data -c catalog.geojson -m /path/to/model.ckpt
#  With config: ./scripts/run-predict.sh -f config.yml
#  Submit with sbatch (provide SBATCH flags on the sbatch command line):
#    sbatch --account=benq-tgirails --partition=gpu --time=04:00:00 scripts/run-predict.sh -f cfg.yml

set -euo pipefail

# Defaults
DATA_DIR=""
CATALOG=""
MODEL_CKPT=""
MODEL_NAME=""
OUTPUT_BASE="${OUTPUT_BASE:-$HOME/working/models/outputs/predictions}"
PLOT_DIR="${PLOT_DIR:-$HOME/working/models/outputs/plots}"
GLOBAL_STATS=""
NORMALIZATION="${NORMALIZATION:-z_value}"
NORMALIZATION_PROC="${NORMALIZATION_PROC:-gpb}"
CREATE_PLOTS="yes"
DRY_RUN="no"
CONFIG_FILE=""

usage() {
  cat <<EOF
Usage: $0 [options]
  -f <config.yml|config.json>  optional config file (keys: data_dir, catalog, model_ckpt, output_base, plot_dir, global_stats, normalization, normalization_proc, create_plots, model_name)
  -d <data_dir>                data directory (overrides config)
  -c <catalog>                 catalog file (overrides config)
  -m <model_ckpt>              model checkpoint path (overrides config)
  -o <output_base>             base output dir (overrides config)
  --plot-dir <dir>             plot output dir (overrides config)
  --no-plots                   disable plot creation
  --dry-run                    print command and exit
  -h                           show this help
EOF
  exit 1
}

# simple arg parse
while [[ $# -gt 0 ]]; do
  case "$1" in
    -f) CONFIG_FILE="$2"; shift 2 ;;
    -d) DATA_DIR="$2"; shift 2 ;;
    -c) CATALOG="$2"; shift 2 ;;
    -m) MODEL_CKPT="$2"; shift 2 ;;
    -o) OUTPUT_BASE="$2"; shift 2 ;;
    --plot-dir) PLOT_DIR="$2"; shift 2 ;;
    --no-plots) CREATE_PLOTS="no"; shift 1 ;;
    --dry-run) DRY_RUN="yes"; shift 1 ;;
    -h) usage ;;
    *) echo "Unknown arg: $1"; usage ;;
  esac
done

# load config if provided (only extract known keys)
if [[ -n "${CONFIG_FILE:-}" ]]; then
  if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Config file not found: $CONFIG_FILE" >&2
    exit 1
  fi
  # python reads config and prints bash assignments for keys we accept
  eval "$(python3 - "$CONFIG_FILE" <<'PY'
import sys, json
p=sys.argv[1]
cfg={}
try:
    import yaml
    cfg = yaml.safe_load(open(p))
except Exception:
    cfg = json.load(open(p))
if not isinstance(cfg, dict):
    cfg={}
keys = {
 'data_dir':'DATA_DIR',
 'catalog':'CATALOG',
 'model_ckpt':'MODEL_CKPT',
 'output_base':'OUTPUT_BASE',
 'plot_dir':'PLOT_DIR',
 'global_stats':'GLOBAL_STATS',
 'normalization':'NORMALIZATION',
 'normalization_proc':'NORMALIZATION_PROC',
 'create_plots':'CREATE_PLOTS',
 'model_name':'MODEL_NAME'
}
out=[]
for k,v in cfg.items():
    if k in keys and v is not None:
        # quote safely
        s = str(v).replace("'", "'\"'\"'")
        out.append(f"{keys[k]}='{s}'")
print("\n".join(out))
PY
  )"
fi

# Validate required inputs
if [[ -z "${DATA_DIR}" || -z "${CATALOG}" || -z "${MODEL_CKPT}" ]]; then
  echo "Missing required arguments. Provide -d, -c, -m or include them in config."
  usage
fi

# determine model name for namespacing
if [[ -z "${MODEL_NAME}" ]]; then
  MODEL_NAME=$(basename "$(dirname "$(dirname "$MODEL_CKPT")")" 2>/dev/null || echo "$(basename "$MODEL_CKPT")")
fi

# expand paths
DATA_DIR=$(eval echo "$DATA_DIR")
CATALOG=$(eval echo "$CATALOG")
MODEL_CKPT=$(eval echo "$MODEL_CKPT")
OUTPUT_BASE=$(eval echo "$OUTPUT_BASE")
PLOT_DIR=$(eval echo "$PLOT_DIR")
OUT_DIR="${OUTPUT_BASE%/}/${MODEL_NAME}"
PLOT_OUT_DIR="${PLOT_DIR%/}/${MODEL_NAME}"

mkdir -p "$OUT_DIR" "$PLOT_OUT_DIR"

# Build command
CMD=(ftw_ma model predict -c "$CATALOG" -m "$MODEL_CKPT" -o "$OUT_DIR" --path_column window_b --id_column name --normalization_strategy "$NORMALIZATION" --normalization_stat_procedure "$NORMALIZATION_PROC" -d "$DATA_DIR" --overwrite)

if [[ -n "${GLOBAL_STATS}" ]]; then
  CMD+=(--global_stats "$GLOBAL_STATS")
fi
if [[ "$CREATE_PLOTS" == "yes" ]]; then
  CMD+=(--create_plots --plot_output_dir "$PLOT_OUT_DIR")
fi

# show and run (or dry-run)
echo "Command:"
printf '%q ' "${CMD[@]}"
echo
echo "Output dir: $OUT_DIR"
echo "Plot dir:   $PLOT_OUT_DIR"

if [[ "$DRY_RUN" == "yes" ]]; then
  echo "Dry run; exiting."
  exit 0
fi

"${CMD[@]}"
RC=$?
exit $RC