#!/bin/bash

# Script training cho mô hình SPINE với Gemma 3 4B + dynamic fusion (Sagittal Fused + Axial T2)
# Sử dụng DeepSpeed và các tham số theo format hoangnv/g.
# YÊU CẦU: conda env p2vg
#   conda activate p2vg
# Tự động xác định thư mục P2VG root (parent của src/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
P2VG_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$P2VG_ROOT"
echo "Working directory: $(pwd)"

# Thêm path để load module nội bộ
export PYTHONPATH=$PYTHONPATH:$P2VG_ROOT:$P2VG_ROOT/src

# Đường dẫn bộ lưu trữ cục bộ của người dùng
DATA_ROOT="/storage/hoangnv/dataset_ttd_256"
TRAIN_CSV="/storage/hoangnv/dataset_ttd_256/report/train.csv"
VAL_CSV="/storage/hoangnv/dataset_ttd_256/report/val.csv"
WEIGHTS_DIR="$P2VG_ROOT/weights"
OUTPUT_DIR="/storage/hoangnv/P2VG_outputs_dynamicfusion/fold2_1/fold3_fusion768_pretrainedprojection_v2"
PYTHON_BIN="/home/hoangnv/miniconda3/envs/p2vg/bin/python"
DEEPSPEED_BIN="/home/hoangnv/miniconda3/envs/p2vg/bin/deepspeed"

# Tối ưu hóa bộ nhớ CUDA
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT=P2VG_UDML
export WANDB_NAME=fold3_fusion768_pretrainedprojection_v2

# Sử dụng deepspeed launcher
# Previous LR: 5e-5
"$DEEPSPEED_BIN" src/custom_train.py \
    --version v0 \
    --model_name_or_path "google/medgemma-1.5-4b-it" \
    --model_type gemma3 \
    --lora_enable True \
    --lora_r 8 \
    --lora_alpha 32 \
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
    --udml_lm_aux_enable True \
    --udml_lm_aux_loss_weight 1.0 \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 10 \
    --model_max_length 768 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --eval_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 5 \
    --load_best_model_at_end True \
    --metric_for_best_model "loss" \
    --greater_is_better False \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.001 \
    --gradient_checkpointing True \
    --dataloader_pin_memory True \
    --dataloader_num_workers 4 \
    --report_to wandb

LORA_BIN="$OUTPUT_DIR/model_with_lora.bin"
MERGED_DIR="$OUTPUT_DIR/merged_hf"

if [ ! -f "$LORA_BIN" ]; then
    echo "Training finished but LoRA checkpoint was not found: $LORA_BIN" >&2
    exit 1
fi

echo "Merging LoRA weights: $LORA_BIN -> $MERGED_DIR"
"$PYTHON_BIN" src/merge_lora_weights_and_save_hf_model.py \
    --model_name_or_path "google/medgemma-1.5-4b-it" \
    --model_type gemma3 \
    --axt2_enable True \
    --pretrain_vision_model "$WEIGHTS_DIR/pretrained_ViT.bin" \
    --mm_projector_type spp \
    --model_with_lora "$LORA_BIN" \
    --output_dir "$MERGED_DIR"

echo "Done. Merged model at: $MERGED_DIR"
