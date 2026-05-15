#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
P2VG_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$P2VG_ROOT"
export PYTHONPATH=$PYTHONPATH:$P2VG_ROOT:$P2VG_ROOT/src
/home/hoangnv/miniconda3/envs/p2vg/bin/python -u src/demo_csv.py \
  --model_name_or_path /storage/hoangnv/P2VG_outputs_dynamicfusion/fold2_1/fold3_fusion768_pretrainedprojection_v2/merged_hf \
  --data_root /storage/hoangnv/dataset_ttd_256 \
  --amos_train_cap_data_path /storage/hoangnv/dataset_ttd_256/report/train.csv \
  --amos_validation_cap_data_path /storage/hoangnv/dataset_ttd_256/report/test.csv \
  --output_dir /storage/hoangnv/fold3_fusion768_pretrainedprojection_v2/eval \
  --axt2_enable \
  --sagittal_modality fused
