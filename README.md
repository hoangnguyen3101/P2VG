# P2VG — Spine MRI Captioning (Dual-Encoder + UDML Fusion)

Fine-tune MedGemma/Gemma3 để sinh báo cáo từ ảnh MRI cột sống 3D.
Mô hình dùng **hai visual encoder** (sagittal + axial T2), trộn đặc trưng qua module **fusion** rồi đưa vào LLM. Mặc định fusion là **UDML** (uncertainty- & dependency-aware dynamic fusion); ngoài ra có sẵn các baseline fusion để so sánh.

## Cấu Trúc

```
P2VG/
├── src/
│   ├── custom_train.py          # Entry point training (DeepSpeed)
│   ├── multi_dataset.py         # SpineCapDataset (NIfTI + CSV)
│   ├── demo_csv.py              # Inference theo CSV
│   ├── eval_caption_metrics.py  # Chấm điểm caption (LLM-based)
│   ├── merge_lora_weights_and_save_hf_model.py  # Merge LoRA → HF dir
│   ├── lamed_trainer.py         # Trainer tuỳ biến
│   └── model/
│       ├── lamed_arch.py        # Forward multimodal, build encoders + fusion
│       ├── lamed_gemma3.py      # LamedGemma3ForCausalLM
│       ├── udml_fusion.py       # UDMLFusion (mặc định)
│       ├── base_fusion.py       # Baseline: elementwise / concat / gate / bilinear
│       ├── multimodal_encoder/  # ViT3D
│       └── multimodal_projector/# Projector (spp)
├── scripts/
│   ├── train.sh                 # Train (stage1+stage2) + auto-merge LoRA
│   ├── evaluate.sh              # Inference + chấm điểm
│   └── normalize_monai_256.py   # Chuẩn hoá ảnh → 256, sinh CSV
├── configs/ds_config_zero2.json # DeepSpeed ZeRO-2
└── weights/pretrained_ViT.bin   # M3D pretrained ViT
```

## Dữ Liệu

```
<DATA_ROOT>/
├── report/{train,val,test}.csv      # cột: case_id, images_path, Clinician's Notes, split
└── Volume/
    ├── <sub>_sagt2.nii.gz           # sagittal T2
    ├── <sub>_sagt1.nii.gz           # sagittal T1
    └── <sub>_axt2.nii.gz            # axial T2
```

- `images_path` trong CSV nối với `DATA_ROOT` → trỏ tới thư mục `Volume` (ví dụ giá trị `Volume`). Dùng đường dẫn **tuyệt đối** hoặc **tương đối so với `DATA_ROOT`**, không phải so với thư mục CSV.
- Tên file ảnh: `<sub>_<modality>.nii.gz`. `modality` khớp với `SAGITTAL_MODALITY` (vd `sagt2`, `sagt1`) và `axt2` cho nhánh axial.

## Cài Đặt

```bash
conda activate p2vg          # cần torch, transformers, deepspeed, monai, nibabel...
# Base LLM: google/medgemma-1.5-4b-it (tải tự động, cần HF token)
# Pretrained ViT: weights/pretrained_ViT.bin
#   https://huggingface.co/GoodBaiBai88/M3D-CLIP/blob/main/pretrained_ViT.bin
```

## Training

```bash
bash scripts/train.sh [OUTPUT_SUFFIX]
# vd: bash scripts/train.sh medgemma_udml
```

Chạy stage1 (train vision/adapter) → stage2 (LoRA) → tự merge LoRA vào
`OUTPUT_ROOT/<OUTPUT_SUFFIX>_stage2/merged_hf/`.

### Biến môi trường chính

| Env | Mặc định | Mô tả |
|-----|----------|-------|
| `DATA_ROOT` | `/home/hoangnv/AICD_HA/dataset/Lumbar/dataset_lumbar_256` | Gốc dataset (chứa `Volume/`, `report/`) |
| `OUTPUT_ROOT` | `/storage/.../dataset_lumbar/3006` | Nơi lưu output |
| `TRAIN_STAGE` | `both` | `stage1` \| `stage2` \| `both` |
| `FUSION_TYPE` | `udml` | `udml` \| `elementwise` \| `concat` \| `gate` \| `bilinear` |
| `AXT2_ENABLE` | `True` | Bật encoder axial (dual-encoder). `False` = chỉ sagittal |
| `AXIAL_ONLY` | `False` | Chỉ dùng axial (bỏ qua sagittal) |
| `SAGITTAL_MODALITY` | `fused` | Hậu tố file sagittal: `sagt2`, `sagt1`, `fused`... |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU |

### Ví dụ cấu hình

```bash
# UDML dual-encoder (mặc định)
bash scripts/train.sh medgemma_udml

# Baseline fusion (gate / concat / bilinear / elementwise)
FUSION_TYPE=gate bash scripts/train.sh medgemma_gate

# Sagittal-T2 only (1 encoder, không fusion)
AXT2_ENABLE=False SAGITTAL_MODALITY=sagt2 bash scripts/train.sh medgemma_sagt2_only

# Axial only
AXIAL_ONLY=True bash scripts/train.sh medgemma_axial_only
```

> Khi `AXT2_ENABLE=False` hoặc `AXIAL_ONLY=True` chỉ một encoder chạy → fusion (UDML/baseline) và các loss UDML không có tác dụng.

### Flag UDML (chỉ áp dụng khi dual-encoder + `FUSION_TYPE=udml`)

| Flag | Mặc định | Mô tả |
|------|----------|-------|
| `UDML_NOISE_ENABLE` | stage2: `True` | Inject noise để supervise variance estimator |
| `UDML_NOISE_PROB` | `0.2` | Xác suất inject noise mỗi sample |
| `UDML_NOISE_MAX` | `6` | Mức variance tối đa |
| `udml_var_loss_weight` | `0.1` | Trọng số variance loss |
| `UDML_LM_AUX_ENABLE` | stage2: `True` | Bật unimodal LM aux loss (sag-only + ax-only) |

## Đánh Giá

```bash
export GROQ_API_KEY="..."     # cho metric chấm bằng LLM (llama-3.3-70b-versatile)
bash scripts/evaluate.sh [OUTPUT_SUFFIX] [MODEL_SUBDIR]
# vd: bash scripts/evaluate.sh medgemma_udml
```

Eval phải dùng **cùng cấu hình ảnh như lúc train** (giữ đúng `AXT2_ENABLE` / `AXIAL_ONLY` / `SAGITTAL_MODALITY`):

```bash
AXIAL_ONLY=True bash scripts/evaluate.sh medgemma_axial_only
```

- Bước 1: `demo_csv.py` sinh `eval/eval_caption.csv`.
- Bước 2: `eval_caption_metrics.py` chấm → `eval/eval_scores.csv`.

## Fusion (`src/model/base_fusion.py`)

Tất cả nhận `(feat_sag, feat_ax)` dạng `[B, N, H]` → trả `[B, N, H]`, dùng chung slot `model.udml_fusion` nên trọng số được lưu/merge tự động.

| `FUSION_TYPE` | Cơ chế |
|---------------|--------|
| `udml` | Trọng số động theo độ bất định + dependency của từng modality |
| `elementwise` | Cộng/nhân/mean theo phần tử (tuỳ chọn blend học được) |
| `concat` | Nối kênh `[B,N,2H]` → `Linear(2H→H)` |
| `gate` | Cổng `g=σ(W[sag;ax])`; `out = g·sag + (1−g)·ax` |
| `bilinear` | Low-rank bilinear (MFB): `(U·sag)⊙(V·ax) → Linear(rank→H)` |
