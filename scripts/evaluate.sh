#!/bin/bash
# Run inference + evaluation for a given fold.
# Usage:
#   bash scripts/evaluate.sh [FOLD] [OUTPUT_SUFFIX] [MODEL_SUBDIR]
#   FOLD defaults to 2, OUTPUT_SUFFIX defaults to empty, MODEL_SUBDIR defaults to merged_hf
#
# Env overrides:
#   DATA_ROOT, VAL_CSV, MODEL_PATH, PYTHON_BIN
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
P2VG_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$P2VG_ROOT"

export PYTHONPATH="$P2VG_ROOT:$P2VG_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

FOLD="${1:-2}"
OUTPUT_SUFFIX="${2:-}"
MODEL_SUBDIR="${3:-merged_hf}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/storage/hoangnv/P2VG_outputs_dynamicfusion/fold2_1}"
RUN_DIR="$OUTPUT_ROOT/fold${FOLD}${OUTPUT_SUFFIX}_stage2"
DATA_ROOT="${DATA_ROOT:-/storage/hoangnv/dataset_ttd_256}"
VAL_CSV="${VAL_CSV:-/storage/hoangnv/dataset_ttd_256/report/test.csv}"
MODEL_PATH="${MODEL_PATH:-$RUN_DIR/$MODEL_SUBDIR}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR/eval}"
PYTHON_BIN="${PYTHON_BIN:-$(which python 2>/dev/null || echo python)}"

echo "Model     : $MODEL_PATH"
echo "Val CSV   : $VAL_CSV"
echo "Output    : $OUTPUT_DIR"

if [ ! -d "$MODEL_PATH" ]; then
    echo "[ERR] MODEL_PATH does not exist: $MODEL_PATH" >&2
    exit 1
fi

# Step 1: generate predictions
"$PYTHON_BIN" src/demo_csv.py \
    --model_name_or_path "$MODEL_PATH" \
    --max_length 768 \
    --data_root "$DATA_ROOT" \
    --amos_validation_cap_data_path "$VAL_CSV" \
    --output_dir "$OUTPUT_DIR" \
    --axt2_enable \
    --sagittal_modality fused

# Step 2: compute metrics
LLM_MODEL="${LLM_MODEL:-llama-3.3-70b-versatile}"
"$PYTHON_BIN" src/eval_caption_metrics.py \
    --input_csv "$OUTPUT_DIR/eval_caption.csv" \
    --output_csv "$OUTPUT_DIR/eval_scores.csv" \
    --llm_model "$LLM_MODEL"
