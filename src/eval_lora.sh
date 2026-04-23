#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
P2VG_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$P2VG_ROOT"
export PYTHONPATH=$PYTHONPATH:$P2VG_ROOT:$P2VG_ROOT/M3D
/home/hoangnv/miniconda3/envs/p2vg/bin/python -u src/demo_csv.py \
  --model_name_or_path /storage/hoangnv/triplane_kfold/gemma3_pka_fused_gated_fold3/merged_hf_54 \
  --data_root /home/hoangnv/AICD_HA/SPINE_BASE/P2VG/dataset_PKA \
  --amos_train_cap_data_path /home/hoangnv/AICD_HA/SPINE_BASE/P2VG/dataset_PKA/triplane_kfold/fold_3/train.csv \
  --amos_validation_cap_data_path /home/hoangnv/AICD_HA/SPINE_BASE/P2VG/dataset_PKA/triplane_kfold/fold_3/test.csv \
  --output_dir /storage/hoangnv/triplane_kfold/gemma3_pka_fused_gated_fold3/eval_results_54_test \
  --axt2_enable True \
  --axial_only False \
  --sagittal_modality fused \
  --do_sample False
