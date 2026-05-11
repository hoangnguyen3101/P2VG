#!/bin/bash

# Train P2VG on the normalized TTD dataset:
#   /storage/hoangnv/dataset_ttd_256
#
# Expected files:
#   report/train.csv
#   report/val.csv
#   Volume/sub-XXXX_fused.nii.gz
#   Volume/sub-XXXX_axt2.nii.gz
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
P2VG_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$P2VG_ROOT"
echo "Working directory: $(pwd)"

export PYTHONPATH=$PYTHONPATH:$P2VG_ROOT:$P2VG_ROOT/M3D

DATA_ROOT="/storage/hoangnv/dataset_ttd_256"
TRAIN_CSV="$DATA_ROOT/report/train.csv"
VAL_CSV="$DATA_ROOT/report/val.csv"
WEIGHTS_DIR="/home/hoangnv/AICD_HA/SPINE_BASE/SPINE/weights"
DEEPSPEED_BIN="/home/hoangnv/miniconda3/envs/p2vg/bin/deepspeed"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT="PKA_UMDL"
export WANDB_NAME="gemma3_TTD256_fused_axt2_ep5"

"$DEEPSPEED_BIN" src/custom_train.py \
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
    --output_dir /storage/hoangnv/PKA_UMDL/gemma3_TTD256_fused_axt2_ep5 \
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
    --sagittal_modality fused \
    --report_to wandb
