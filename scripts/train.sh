#!/bin/bash
# Usage:
#   bash scripts/train.sh [FOLD] [OUTPUT_SUFFIX]
#   TRAIN_STAGE=stage1|stage2|both bash scripts/train.sh 3 _medgemma_adapter
#
# Env overrides:
#   DATA_ROOT, WEIGHTS_DIR, OUTPUT_ROOT, DEEPSPEED_BIN, PYTHON_BIN, TRAIN_CSV, VAL_CSV
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
P2VG_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$P2VG_ROOT"

err() { echo "[ERR] $*" >&2; }

export PYTHONPATH="$P2VG_ROOT:$P2VG_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-$USER}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

OUTPUT_SUFFIX="${1:-_spider_noaxial}"
TRAIN_STAGE="${TRAIN_STAGE:-both}"

DEFAULT_DATA_ROOT="/storage/hoangnv/dataset_PKA_Wavelate"
DEFAULT_SPLIT_ROOT="/storage/hoangnv/dataset_PKA_Wavelate"
DATA_ROOT="${DATA_ROOT:-$DEFAULT_DATA_ROOT}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$P2VG_ROOT/weights}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/storage/hoangnv/P2VG_outputs_dynamicfusion/dataset_PKA_Wavelate/v2}"
PYTHON_BIN="${PYTHON_BIN:-$(which python 2>/dev/null || echo '')}"
DEEPSPEED_BIN="${DEEPSPEED_BIN:-$(which deepspeed 2>/dev/null || echo '')}"
if [ -z "$PYTHON_BIN" ]; then
    err "python not found. Activate your conda env first, e.g.: conda activate p2vg"
    exit 1
fi
if [ -z "$DEEPSPEED_BIN" ]; then
    err "deepspeed not found. Activate your conda env first, e.g.: conda activate p2vg"
    exit 1
fi

TRAIN_CSV="${TRAIN_CSV:-$DEFAULT_SPLIT_ROOT/train.csv}"
VAL_CSV="${VAL_CSV:-$DEFAULT_SPLIT_ROOT/val.csv}"
if [ ! -f "$TRAIN_CSV" ] || [ ! -f "$VAL_CSV" ]; then
    err "Missing dataset CSV files under DATA_ROOT=$DATA_ROOT"
    exit 1
fi
if [ ! -f "$WEIGHTS_DIR/pretrained_ViT.bin" ]; then
    err "Missing pretrained ViT: $WEIGHTS_DIR/pretrained_ViT.bin"
    exit 1
fi

export WANDB_PROJECT="${WANDB_PROJECT:-P2VG_SPINED}"
USER_WANDB_NAME="${WANDB_NAME:-}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-32}"
UDML_NOISE_PROB="${UDML_NOISE_PROB:-0.2}"
UDML_NOISE_MAX="${UDML_NOISE_MAX:-6}"
AXT2_ENABLE="${AXT2_ENABLE:-False}"
AXIAL_ONLY="${AXIAL_ONLY:-False}"
SAGITTAL_MODALITY="${SAGITTAL_MODALITY:-fused}"
# UDML_NOISE_ENABLE và UDML_LM_AUX_ENABLE được set per-stage trong run_stage()

best_trainable_path() {
    local dir="$1"
    if [ -f "$dir/model_trainable_best.bin" ]; then
        echo "$dir/model_trainable_best.bin"
    elif [ -f "$dir/model_with_lora_best.bin" ]; then
        echo "$dir/model_with_lora_best.bin"
    elif [ -f "$dir/model_trainable.bin" ]; then
        echo "$dir/model_trainable.bin"
    elif [ -f "$dir/model_with_lora.bin" ]; then
        echo "$dir/model_with_lora.bin"
    fi
}

run_stage() {
    local stage="$1"
    local output_dir="$OUTPUT_ROOT/${OUTPUT_SUFFIX}_${stage}"
    local lora_enable num_epochs lr freeze_projection freeze_vision visual_ckpt

    if [ "$stage" = "stage1" ]; then
        lora_enable="${LORA_ENABLE:-False}"
        num_epochs="${STAGE1_EPOCHS:-${NUM_TRAIN_EPOCHS:-5}}"
        lr="${STAGE1_LEARNING_RATE:-${LEARNING_RATE:-1e-4}}"
        freeze_projection="${FREEZE_MEDGEMMA_PROJECTION:-True}"
        freeze_vision="${STAGE1_FREEZE_VISION_TOWER:-${FREEZE_VISION_TOWER:-False}}"
        visual_ckpt=""
        # Stage1: vision tower đang train, tắt noise để estimator không học trên moving target
        local udml_noise_enable="False"
        local udml_lm_aux_enable="${UDML_LM_AUX_ENABLE:-False}"
    elif [ "$stage" = "stage2" ]; then
        lora_enable="${LORA_ENABLE:-True}"
        num_epochs="${STAGE2_EPOCHS:-${NUM_TRAIN_EPOCHS:-15}}"
        lr="${STAGE2_LEARNING_RATE:-${LEARNING_RATE:-3e-5}}"
        freeze_projection="${FREEZE_MEDGEMMA_PROJECTION:-False}"
        freeze_vision="${STAGE2_FREEZE_VISION_TOWER:-${FREEZE_VISION_TOWER:-False}}"
        visual_ckpt="${VISUAL_ADAPTER_CHECKPOINT:-}"
        # Stage2: encoder đã ổn định, bật noise để train variance estimator
        local udml_noise_enable="${UDML_NOISE_ENABLE:-True}"
        local udml_lm_aux_enable="${UDML_LM_AUX_ENABLE:-True}"
        if [ -z "$visual_ckpt" ]; then
            visual_ckpt="$(best_trainable_path "$OUTPUT_ROOT/${OUTPUT_SUFFIX}_stage1")"
        fi
        if [ -z "$visual_ckpt" ]; then
            err "Stage2 requires a stage1 checkpoint. Run TRAIN_STAGE=stage1 first or set VISUAL_ADAPTER_CHECKPOINT."
            exit 1
        fi
    else
        err "Unknown stage: $stage"
        exit 1
    fi

    if [ -n "$USER_WANDB_NAME" ]; then
        export WANDB_NAME="$USER_WANDB_NAME"
    else
        export WANDB_NAME="${OUTPUT_SUFFIX}_${stage}"
    fi
    echo "P2VG_ROOT  : $P2VG_ROOT"
    echo "DATA_ROOT  : $DATA_ROOT"
    echo "OUTPUT_DIR : $output_dir"
    echo "PYTHON     : $PYTHON_BIN"
    echo "DEEPSPEED  : $DEEPSPEED_BIN"
    echo "STAGE      : $stage"
    echo "LORA       : enable=$lora_enable r=$LORA_R alpha=$LORA_ALPHA"
    echo "IMAGE      : sagittal_modality=$SAGITTAL_MODALITY axt2_enable=$AXT2_ENABLE axial_only=$AXIAL_ONLY"
    echo "MEDGEMMA   : freeze_projection=$freeze_projection freeze_vision_tower=$freeze_vision"
    echo "UDML_NOISE : enable=$udml_noise_enable prob=$UDML_NOISE_PROB max=$UDML_NOISE_MAX lm_aux=$udml_lm_aux_enable"
    if [ -n "$visual_ckpt" ]; then
        echo "VISUAL_CKPT: $visual_ckpt"
    fi

    local train_cmd=(
        "$DEEPSPEED_BIN" src/custom_train.py
        --version v0
        --model_name_or_path "google/medgemma-1.5-4b-it"
        --model_type gemma3
        --train_stage "$stage"
        --lora_enable "$lora_enable"
        --lora_r "$LORA_R"
        --lora_alpha "$LORA_ALPHA"
        --vision_tower vit3d
        --axt2_enable "$AXT2_ENABLE"
        --axial_only "$AXIAL_ONLY"
        --freeze_vision_tower "$freeze_vision"
        --pretrain_vision_model "$WEIGHTS_DIR/pretrained_ViT.bin"
        --medgemma_adapter_enable True
        --freeze_medgemma_projection "$freeze_projection"
        --bf16 False
        --mm_projector_type spp
        --data_root "$DATA_ROOT"
        --amos_train_cap_data_path "$TRAIN_CSV"
        --amos_validation_cap_data_path "$VAL_CSV"
        --sagittal_modality "$SAGITTAL_MODALITY"
        --udml_noise_enable "$udml_noise_enable"
        --udml_noise_prob "$UDML_NOISE_PROB"
        --udml_noise_min 2
        --udml_noise_max "$UDML_NOISE_MAX"
        --udml_noise_std_scale 0.02
        --udml_var_loss_weight 0.1
        --udml_lm_aux_enable "$udml_lm_aux_enable"
        --udml_lm_aux_loss_weight 1.0
        --output_dir "$output_dir"
        --num_train_epochs "$num_epochs"
        --model_max_length "${MODEL_MAX_LENGTH:-768}"
        --per_device_train_batch_size 1
        --per_device_eval_batch_size 1
        --gradient_accumulation_steps "${GRAD_ACCUM_STEPS:-8}"
        --eval_strategy epoch
        --save_strategy no
        --save_total_limit 1
        --load_best_model_at_end False
        --metric_for_best_model eval_loss
        --greater_is_better False
        --learning_rate "$lr"
        --weight_decay 0.01
        --warmup_ratio 0.03
        --lr_scheduler_type cosine
        --logging_strategy epoch
        --gradient_checkpointing True
        --dataloader_pin_memory True
        --dataloader_num_workers 4
        --report_to wandb
    )
    if [ "$stage" = "stage2" ]; then
        train_cmd+=(--visual_adapter_checkpoint "$visual_ckpt")
    fi
    train_cmd+=(--deepspeed "$P2VG_ROOT/configs/ds_config_zero2.json")

    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf 'DRY_RUN command:'
        printf ' %q' "${train_cmd[@]}"
        printf '\n'
        return
    fi

    "${train_cmd[@]}"

    if [ "$stage" = "stage1" ]; then
        echo "Stage1 done. Best visual checkpoint: $(best_trainable_path "$output_dir")"
        return
    fi

    local lora_bin
    lora_bin="$(best_trainable_path "$output_dir")"
    if [ -z "$lora_bin" ]; then
        err "No trainable checkpoint found in $output_dir"
        exit 1
    fi
    local merged_dir="$output_dir/merged_hf"
    echo "Merging LoRA weights: $lora_bin -> $merged_dir"
    "$PYTHON_BIN" src/merge_lora_weights_and_save_hf_model.py \
        --model_name_or_path "google/medgemma-1.5-4b-it" \
        --model_type gemma3 \
        --axt2_enable "$AXT2_ENABLE" \
        --axial_only "$AXIAL_ONLY" \
        --pretrain_vision_model "$WEIGHTS_DIR/pretrained_ViT.bin" \
        --medgemma_adapter_enable True \
        --mm_projector_type spp \
        --lora_r "$LORA_R" \
        --lora_alpha "$LORA_ALPHA" \
        --model_with_lora "$lora_bin" \
        --output_dir "$merged_dir"

    echo "Done. Merged model at: $merged_dir"
}

case "$TRAIN_STAGE" in
    stage1)
        run_stage stage1
        ;;
    stage2)
        run_stage stage2
        ;;
    both)
        run_stage stage1
        run_stage stage2
        ;;
    *)
        err "TRAIN_STAGE must be stage1, stage2, or both, got: $TRAIN_STAGE"
        exit 1
        ;;
esac
