#!/usr/bin/env bash
#SBATCH --job-name=ftw_catalog_test
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail

# ---------------------- CONFIGURATION -----------------------------------------
CATALOG="data/ftw-catalog2.csv"
SPLIT="validate"
MODELS_FILE="models_list.txt"
EXTRA_ARGS_FILE="extra_args.txt"

# Optional data directory override
DATA_DIR=""

# ---------------------- READ INPUT FILES --------------------------------------
if [[ ! -f "$MODELS_FILE" ]]; then
  echo "❌ Models file not found: $MODELS_FILE" >&2
  exit 1
fi
MODELS=$(awk 'NF{print $1}' "$MODELS_FILE" | tr '\n' ' ')

EXTRA_ARGS=()
if [[ -f "$EXTRA_ARGS_FILE" ]]; then
  while IFS= read -r line; do
    [[ -z "$line" || "$line" =~ ^# ]] && continue
    EXTRA_ARGS+=("$line")
  done < "$EXTRA_ARGS_FILE"
else
  echo "⚠️ No $EXTRA_ARGS_FILE found — running with defaults."
fi

# ---------------------- RUN CATALOG TEST --------------------------------------
echo "🚀 Running catalog test for models: $MODELS"
echo "Catalog: $CATALOG"
echo "Split:   $SPLIT"
echo "Extra args: ${EXTRA_ARGS[*]}"
[[ -n "$DATA_DIR" ]] && echo "Data dir override: $DATA_DIR"

python run_tests.py \
  --models $MODELS \
  --catalog "$CATALOG" \
  --split "$SPLIT" \
  --data_dir "$DATA_DIR" \
  "${EXTRA_ARGS[@]}"
