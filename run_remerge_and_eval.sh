#!/bin/bash
set -e
echo "Re-merging model weights with vision adapter..."
/home/hoangnv/miniconda3/envs/p2vg/bin/python src/merge_lora_weights_and_save_hf_model.py \
  --model_name_or_path google/medgemma-1.5-4b-it \
  --model_type gemma3 \
  --axt2_enable True \
  --pretrain_vision_model weights/pretrained_ViT.bin \
  --medgemma_adapter_enable True \
  --mm_projector_type spp \
  --lora_r 8 --lora_alpha 32 \
  --model_with_lora /storage/hoangnv/P2VG_outputs_dynamicfusion/dataset_lumbar/fold2_s1unfreeze_e8_s2freeze_e4_ckpt_stage2/model_trainable_best.bin \
  --visual_adapter_checkpoint /storage/hoangnv/P2VG_outputs_dynamicfusion/dataset_lumbar/fold2_s1unfreeze_e8_s2freeze_e4_ckpt_stage1/model_trainable_best.bin \
  --output_dir /storage/hoangnv/P2VG_outputs_dynamicfusion/dataset_lumbar/fold2_s1unfreeze_e8_s2freeze_e4_ckpt_stage2/merged_hf

echo "Running evaluation..."
bash scripts/evaluate.sh 2 _s1unfreeze_e8_s2freeze_e4_ckpt
echo "Done!"
