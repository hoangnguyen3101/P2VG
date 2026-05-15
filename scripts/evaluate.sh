#!/bin/bash
# Run inference + evaluation for a given fold.
# Usage:
#   bash scripts/evaluate.sh [FOLD] [MODEL_SUBDIR]
#   FOLD defaults to 2, MODEL_SUBDIR defaults to merged_hf
#
# Env overrides:
#   DATA_ROOT, VAL_CSV, MODEL_PATH
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
P2VG_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$P2VG_ROOT"

export PYTHONPATH="$P2VG_ROOT:$P2VG_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

FOLD="${1:-2}"
MODEL_SUBDIR="${2:-merged_hf}"

DATA_ROOT="${DATA_ROOT:-/storage/hoangnv/dataset_ttd_256/Volume}"
VAL_CSV="${VAL_CSV:-/storage/hoangnv/dataset_ttd_256/report/test.csv}"
MODEL_PATH="${MODEL_PATH:-/storage/hoangnv/P2VG_outputs_dynamicfusion/fold2_1/fold3fold3_dynamicfusion_udml_lora8_alpha32/merged_hf}"
OUTPUT_DIR="${OUTPUT_DIR:-/storage/hoangnv/P2VG_outputs_dynamicfusion/fold2_1/fold3fold3_dynamicfusion_udml_lora8_alpha32/eval}"

echo "Model     : $MODEL_PATH"
echo "Val CSV   : $VAL_CSV"
echo "Output    : $OUTPUT_DIR"

# Step 1: generate predictions
python src/demo_csv.py \
    --model_name_or_path "$MODEL_PATH" \
    --max_length 768 \
    --data_root "$DATA_ROOT" \
    --amos_validation_cap_data_path "$VAL_CSV" \
    --output_dir "$OUTPUT_DIR" \
    --axt2_enable \
    --sagittal_modality fused

# Step 2: compute metrics
LLM_MODEL="${LLM_MODEL:-llama-3.3-70b-versatile}"
python src/eval_caption_metrics.py \
    --input_csv "$OUTPUT_DIR/eval_caption.csv" \
    --output_csv "$OUTPUT_DIR/eval_scores.csv" \
    --llm_model "$LLM_MODEL"
