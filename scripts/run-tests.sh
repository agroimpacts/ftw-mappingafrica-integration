#!/usr/bin/env bash
#SBATCH --account=benq-tgirails
#SBATCH --job-name=ftw_test
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail

# ---------------------- ARGUMENTS --------------------------------------------
if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <catalog> <split> <models_file> " \
       "[countries_file] [data_dir]"
  exit 1
fi

CATALOG="$1"                          # relative path to project root
SPLIT="$2"
MODELS_FILE="$3"
COUNTRIES_FILE="${4:-scripts/countries.txt}"
DATA_DIR="${5:-}"                      # root folder for image chips

# ---------------------- VALIDATION -------------------------------------------
if [[ ! -f "$MODELS_FILE" ]]; then
  echo "❌ Models file not found: $MODELS_FILE" >&2
  exit 1
fi

if [[ ! -f "$CATALOG" ]]; then
  echo "❌ Catalog file not found: $CATALOG" >&2
  exit 1
fi

[[ -n "$DATA_DIR" ]] && echo "📂 Using data dir: $DATA_DIR"

# ---------------------- READ FILES -------------------------------------------
# Read models into array
readarray -t MODELS < "$MODELS_FILE"

# Read countries into array
readarray -t COUNTRIES < "$COUNTRIES_FILE"

# ---------------------- RUN TEST ---------------------------------------------
echo "🚀 Running catalog batch tester"
echo "  Catalog:      $CATALOG"
echo "  Split:        $SPLIT"
echo "  Models:       ${MODELS[*]}"
echo "  Countries:    ${COUNTRIES[*]}"
[[ -n "$DATA_DIR" ]] && echo "  Data dir:     $DATA_DIR"
echo "-----------------------------------------------------"

python scripts/run_tests.py \
  --catalog "$CATALOG" \
  --split "$SPLIT" \
  --models "${MODELS[@]}" \
  --countries "${COUNTRIES[@]}" \
  ${DATA_DIR:+--data_dir "$DATA_DIR"}

