#!/bin/bash

# Script training cho mô hình SPINE với Dual-Encoder (Sagittal Wavelet Fused + Axial T2)
# Sử dụng DeepSpeed và các tham số tối ưu từ cấu hình cũ của người dùng.

# Thêm path để load module LaMed và src
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/M3D

# Đường dẫn bộ lưu trữ cục bộ của người dùng
DATA_ROOT="/home/hoangnv/AICD_HA/SPINE_BASE/SPINE"
TRAIN_CSV="$DATA_ROOT/dataset/Report_Phenikaa/triplane_kfold/fold_1/train.csv"
VAL_CSV="$DATA_ROOT/dataset/Report_Phenikaa/triplane_kfold/fold_1/val.csv"
WEIGHTS_DIR="$DATA_ROOT/weights"

# Sử dụng deepspeed launcher nếu có sẵn trong môi trường
deepspeed custom_train.py \
    --version v0 \
    --model_name_or_path microsoft/Phi-3-mini-4k-instruct \
    --model_type phi3 \
    --lora_enable True \
    --vision_tower vit3d \
    --axt2_enable True \
    --pretrain_vision_model "$WEIGHTS_DIR/pretrained_ViT.bin" \
    --pretrain_mm_mlp_adapter "$WEIGHTS_DIR/mm_projector.bin" \
    --fp16 True \
    --data_root "$DATA_ROOT" \
    --amos_train_cap_data_path "$TRAIN_CSV" \
    --amos_validation_cap_data_path "$VAL_CSV" \
    --output_dir ./output_spine_dual_v1 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "steps" \
    --eval_accumulation_steps 1 \
    --eval_steps 0.1 \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --learning_rate 5e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.001 \
    --gradient_checkpointing True \
    --dataloader_pin_memory True \
    --dataloader_num_workers 32 \
    --report_to none
