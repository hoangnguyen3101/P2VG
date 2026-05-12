#!/bin/bash
# Usage:
#   bash scripts/train.sh [FOLD] [OUTPUT_SUFFIX]
#   FOLD defaults to 2, OUTPUT_SUFFIX is optional (e.g. "_run2")
#
# Env overrides:
#   DATA_ROOT, WEIGHTS_DIR, DEEPSPEED_BIN, TRAIN_CSV, VAL_CSV
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
P2VG_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$P2VG_ROOT"

export PYTHONPATH="$P2VG_ROOT/src:$P2VG_ROOT/M3D${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

FOLD="${1:-2}"
OUTPUT_SUFFIX="${2:-}"

DATA_ROOT="${DATA_ROOT:-$P2VG_ROOT/dataset_ttd_256}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$P2VG_ROOT/weights}"
DEEPSPEED_BIN="${DEEPSPEED_BIN:-$(which deepspeed)}"
TRAIN_CSV="${TRAIN_CSV:-$DATA_ROOT/report/train.csv}"
VAL_CSV="${VAL_CSV:-$DATA_ROOT/report/val.csv}"
OUTPUT_DIR="$P2VG_ROOT/outputs/fold${FOLD}${OUTPUT_SUFFIX}"

if [ ! -f "$TRAIN_CSV" ] || [ ! -f "$VAL_CSV" ]; then
    echo "Missing dataset CSV files under DATA_ROOT=$DATA_ROOT" >&2
    exit 1
fi

if [ ! -f "$WEIGHTS_DIR/pretrained_ViT.bin" ]; then
    echo "Missing pretrained ViT: $WEIGHTS_DIR/pretrained_ViT.bin" >&2
    exit 1
fi

export WANDB_PROJECT="${WANDB_PROJECT:-P2VG_UDML}"
export WANDB_NAME="${WANDB_NAME:-fold${FOLD}${OUTPUT_SUFFIX}}"

echo "P2VG_ROOT  : $P2VG_ROOT"
echo "DATA_ROOT  : $DATA_ROOT"
echo "OUTPUT_DIR : $OUTPUT_DIR"

"$DEEPSPEED_BIN" scripts/train.py \
    --version v0 \
    --model_name_or_path "google/medgemma-1.5-4b-it" \
    --model_type gemma3 \
    --lora_enable True \
    --vision_tower vit3d \
    --axt2_enable True \
    --axial_only False \
    --freeze_vision_tower True \
    --pretrain_vision_model "$WEIGHTS_DIR/pretrained_ViT.bin" \
    --bf16 True \
    --data_root "$DATA_ROOT" \
    --amos_train_cap_data_path "$TRAIN_CSV" \
    --amos_validation_cap_data_path "$VAL_CSV" \
    --sagittal_modality fused \
    --udml_noise_enable True \
    --udml_noise_prob 0.5 \
    --udml_noise_min 2 \
    --udml_noise_max 12 \
    --udml_noise_std_scale 0.02 \
    --udml_var_loss_weight 0.1 \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --eval_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 5 \
    --load_best_model_at_end True \
    --metric_for_best_model "loss" \
    --greater_is_better False \
    --learning_rate 3e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_strategy "epoch" \
    --gradient_checkpointing True \
    --dataloader_pin_memory True \
    --dataloader_num_workers 4 \
    --report_to wandb \
    --deepspeed "$P2VG_ROOT/configs/ds_config_zero2.json"

# Merge LoRA into base model
LORA_BIN="$OUTPUT_DIR/model_with_lora.bin"
MERGED_DIR="$OUTPUT_DIR/merged_hf"

echo "Merging LoRA weights: $LORA_BIN -> $MERGED_DIR"
uv run python scripts/merge_lora.py \
    --model_name_or_path "google/medgemma-1.5-4b-it" \
    --model_type gemma3 \
    --axt2_enable True \
    --model_with_lora "$LORA_BIN" \
    --output_dir "$MERGED_DIR"

echo "Done. Merged model at: $MERGED_DIR"
