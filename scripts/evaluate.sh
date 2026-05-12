#!/bin/bash
# Run inference + evaluation for a given fold.
# Usage:
#   bash scripts/evaluate.sh [FOLD] [MODEL_SUBDIR]
#   FOLD defaults to 2, MODEL_SUBDIR defaults to merged_hf
#
# Env overrides:
#   DATA_ROOT, VAL_CSV
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
P2VG_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$P2VG_ROOT"

export PYTHONPATH="$P2VG_ROOT/src:$P2VG_ROOT/M3D${PYTHONPATH:+:$PYTHONPATH}"

FOLD="${1:-2}"
MODEL_SUBDIR="${2:-merged_hf}"

DATA_ROOT="${DATA_ROOT:-$P2VG_ROOT/dataset_ttd_256}"
VAL_CSV="${VAL_CSV:-$DATA_ROOT/report/val.csv}"
MODEL_PATH="$P2VG_ROOT/outputs/fold${FOLD}/${MODEL_SUBDIR}"
OUTPUT_DIR="$P2VG_ROOT/outputs/fold${FOLD}/eval_results"

echo "Model     : $MODEL_PATH"
echo "Val CSV   : $VAL_CSV"
echo "Output    : $OUTPUT_DIR"

# Step 1: generate predictions
uv run python scripts/demo_csv.py \
    --model_name_or_path "$MODEL_PATH" \
    --data_root "$DATA_ROOT" \
    --amos_validation_cap_data_path "$VAL_CSV" \
    --output_dir "$OUTPUT_DIR" \
    --axt2_enable \
    --sagittal_modality fused

# Step 2: compute metrics
uv run python scripts/eval_captions.py \
    --input_csv "$OUTPUT_DIR/eval_caption.csv" \
    --output_csv "$OUTPUT_DIR/eval_scores.csv"
