#!/bin/bash
#SBATCH --account=benq-tgirails
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --job-name=ftw-train

# === Dynamic paths ===
PROJECT_ROOT="$HOME/projects/ftw-mappingafrica-integration"
LOG_DIR="${PROJECT_ROOT}/logs"

# Extract config name early to use in SLURM log filenames
config=$1
EXPR=$(basename "${config}" .yaml)

#SBATCH --output=%x_%j_${EXPR}.out
#SBATCH --error=%x_%j_${EXPR}.err

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT" || {
  echo "❌ Failed to change directory to project root: $PROJECT_ROOT"
  exit 1
}

# === Activate environment ===
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
pyenv activate ftw-mapafrica || {
  echo "❌ Failed to activate pyenv environment 'ftw-mapafrica'"
  exit 1
}

# === Arguments ===
data_path=$2     # optional override
ckpt_path=$3     # optional checkpoint resume
dryrun=$4        # optional --dryrun flag

if [ -z "$config" ]; then
  echo "❌ No config file provided."
  echo "Usage: sbatch scripts/run_ftw_ma.sh <config.yaml> [data_path] [ckpt_path] [--dryrun]"
  exit 1
fi

CONFIG_PATH="${PROJECT_ROOT}/configs/${config}"
if [ ! -f "$CONFIG_PATH" ]; then
  echo "❌ Config file not found: $CONFIG_PATH"
  exit 1
fi

TIMESTAMP=$(date +"%Y-%m-%dT%H-%M-%S")
LOG_FILE="${LOG_DIR}/${EXPR}_${TIMESTAMP}.log"

# === Info ===
echo "🚀 Launching FTW-MappingAfrica training: $EXPR"
echo "Using config: $CONFIG_PATH"
if [ -n "$data_path" ]; then
  echo "Overriding data.init_args.data_dir with: $data_path"
fi
if [ -n "$ckpt_path" ]; then
  echo "Resuming from checkpoint: $ckpt_path"
fi
echo "Logs will be saved to: $LOG_FILE"
echo "========================================"

# === Handle data override ===
if [ -n "$data_path" ]; then
  TMP_CONFIG=$(mktemp /tmp/ftw_config.XXXXXX.yaml)
  echo "📝 Creating temporary config: $TMP_CONFIG"
  python - <<PY
import yaml
path = "$CONFIG_PATH"
out = "$TMP_CONFIG"
data_path = "$data_path"
cfg = yaml.safe_load(open(path))
if cfg is None: cfg = {}
cfg.setdefault("data", {}).setdefault("init_args", {})["data_dir"] = data_path
yaml.safe_dump(cfg, open(out, "w"), default_flow_style=False)
PY
  RUN_CONFIG="$TMP_CONFIG"
else
  RUN_CONFIG="$CONFIG_PATH"
fi

echo "🔍 Preview of final config (first 10 lines):" | tee -a "$LOG_FILE"
head -n 10 "$RUN_CONFIG" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# === Dry run check ===
if [ "$dryrun" == "--dryrun" ]; then
  echo "🧪 Dry run mode: command will not be executed."
  exit 0
fi

# === Execute training ===
if [ -n "$ckpt_path" ]; then
  echo "Running: ftw_ma model fit -c $RUN_CONFIG -m $ckpt_path" | tee -a "$LOG_FILE"
  ftw_ma model fit -c "$RUN_CONFIG" -m "$ckpt_path" >> "$LOG_FILE" 2>&1
else
  echo "Running: ftw_ma model fit -c $RUN_CONFIG" | tee -a "$LOG_FILE"
  ftw_ma model fit -c "$RUN_CONFIG" >> "$LOG_FILE" 2>&1
fi

rc=$?

# === Cleanup ===
if [ -n "$TMP_CONFIG" ]; then
  rm -f "$TMP_CONFIG"
fi

echo "✅ Training completed at $(date) with exit code $rc" | tee -a "$LOG_FILE"
exit $rc

