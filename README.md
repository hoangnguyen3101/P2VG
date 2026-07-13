# P2VG — Spine MRI Captioning (Dual-Encoder + UDML Fusion)

Fine-tune MedGemma/Gemma3 để sinh báo cáo từ ảnh MRI cột sống 3D.
Mô hình dùng **hai visual encoder** (sagittal + axial T2), trộn đặc trưng qua module **fusion** rồi đưa vào LLM. Fusion mặc định là **UDML** (uncertainty- & dependency-aware, có backbone gate học được); ngoài ra có các baseline fusion (`gate`, `concat`, `bilinear`, `elementwise`) để so sánh.

## Cấu Trúc

```
P2VG/
├── src/
│   ├── custom_train.py          # Entry point training (DeepSpeed)
│   ├── multi_dataset.py         # SpineCapDataset (NIfTI + CSV)
│   ├── demo_csv.py              # Inference theo CSV
│   ├── eval_caption_metrics.py  # Chấm điểm caption (NLG + CE-LLM)
│   ├── merge_lora_weights_and_save_hf_model.py  # Merge LoRA → HF dir
│   ├── lamed_trainer.py         # Trainer tuỳ biến (+ VISION_LR param group)
│   └── model/
│       ├── lamed_arch.py        # Forward multimodal, build encoders + fusion
│       ├── lamed_gemma3.py      # LamedGemma3ForCausalLM
│       ├── udml_fusion.py       # UDMLFusion (residual + gate backbone)
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
├── report/{train,val,test}.csv   # cột: case_id, image_path, Clinician's Notes, sub_id, split
└── Volume/
    ├── sub-<id>_<sag>.nii.gz      # sagittal: hậu tố = fused | sagt2 | sagt1 | t2 ... (tuỳ dataset)
    └── sub-<id>_axt2.nii.gz       # axial T2
```

- Tên file: `sub-<id>_<modality>.nii.gz`, với `<id>` lấy từ cột `sub_id` (`sub-0001` → `0001`). Hậu tố sagittal phải khớp `SAGITTAL_MODALITY`; nhánh axial luôn là `axt2`.
- Cột `image_path` (hoặc `images_path`) nối với `DATA_ROOT` để ra thư mục ảnh. Dùng **đường dẫn tuyệt đối**, hoặc **tương đối so với `DATA_ROOT`** (ví dụ giá trị `Volume`) — không phải tương đối với thư mục CSV.
- Loader raise lỗi cứng nếu thiếu file sagittal; thiếu `axt2` sẽ fallback zero-volume **trừ khi** path chứa `dataset_PKA` (khi đó cũng raise lỗi cứng).

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
# vd: SAVE_EACH_EPOCH=1 bash scripts/train.sh medgemma_udml_ttd
```

Chạy stage1 (train vision/adapter) → stage2 (LoRA) → tự merge LoRA vào
`OUTPUT_ROOT/<OUTPUT_SUFFIX>_stage2/merged_hf/`.

> ⚠️ **Auto-merge lấy checkpoint có `eval_loss` thấp nhất** (thường là epoch cuối). Nhưng ở bài này `eval_loss` thấp = model dễ collapse về báo cáo "bình thường". Xem mục **Chọn checkpoint** bên dưới — nên bật `SAVE_EACH_EPOCH=1` và chọn theo CE-F1.

### Biến môi trường chính

| Env | Mặc định | Mô tả |
|-----|----------|-------|
| `DATA_ROOT` | `/storage/hoangnv/dataset_ttd_256` | Gốc dataset (chứa `Volume/`, `report/`) |
| `OUTPUT_ROOT` | `/storage/.../dataset_ttd_256/0709` | Nơi lưu output |
| `TRAIN_STAGE` | `both` | `stage1` \| `stage2` \| `both` |
| `FUSION_TYPE` | `udml` | `udml` \| `gate` \| `concat` \| `bilinear` \| `elementwise` |
| `AXT2_ENABLE` | `True` | Bật encoder axial (dual-encoder). `False` = chỉ sagittal |
| `AXIAL_ONLY` | `False` | Chỉ dùng axial (bỏ qua sagittal) |
| `SAGITTAL_MODALITY` | `fused` | Hậu tố file sagittal: `fused`, `sagt2`, `sagt1`, `t2`... |
| `STAGE1_EPOCHS` | `10` | Số epoch stage1 |
| `STAGE2_EPOCHS` | `6` | Số epoch stage2 (đã hạ từ 15 để tránh collapse) |
| `STAGE1_LEARNING_RATE` | `1e-4` | LR stage1 |
| `STAGE2_LEARNING_RATE` | `3e-5` | LR stage2 (đừng dùng LR cao kiểu 5e-4 → collapse) |
| `VISION_LR` | *(off)* | Nếu set → vision encoder dùng LR riêng (param group tách) |
| `SAVE_EACH_EPOCH` | `0` | `1` → lưu `model_trainable_epoch{N}.bin` mỗi epoch |
| `WANDB_PROJECT` | `P2VG_Lumbar` | Tên project wandb |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU |

### Ví dụ cấu hình

```bash
# UDML dual-encoder (mặc định), lưu snapshot mỗi epoch để chọn checkpoint
SAVE_EACH_EPOCH=1 bash scripts/train.sh medgemma_udml_ttd

# Baseline fusion (gate / concat / bilinear / elementwise)
FUSION_TYPE=gate SAVE_EACH_EPOCH=1 bash scripts/train.sh medgemma_gate_ttd

# Sagittal only (1 encoder, không fusion)
AXT2_ENABLE=False bash scripts/train.sh medgemma_sag_only

# Axial only
AXIAL_ONLY=True bash scripts/train.sh medgemma_axial_only

# LR riêng cho vision encoder
VISION_LR=1e-5 bash scripts/train.sh medgemma_udml_ttd
```

> Khi `AXT2_ENABLE=False` hoặc `AXIAL_ONLY=True` chỉ một encoder chạy → fusion và các loss UDML không có tác dụng.

### Flag UDML (chỉ khi dual-encoder + `FUSION_TYPE=udml`)

| Flag | Mặc định | Mô tả |
|------|----------|-------|
| `UDML_NOISE_ENABLE` | stage2: `True` | Inject noise để supervise variance estimator (chỉ stage2) |
| `UDML_NOISE_PROB` | `0.2` | Xác suất inject noise mỗi sample |
| `UDML_NOISE_MAX` | `6` | Mức variance tối đa |
| `udml_var_loss_weight` | `0.1` | Trọng số variance loss |
| `UDML_LM_AUX_ENABLE` | stage2: `True` | Unimodal LM aux loss (nuôi dependency EMA) |

## Chọn Checkpoint (quan trọng)

`eval_loss` **không** phản ánh chất lượng sinh — loss thấp nhất thường là điểm model collapse về báo cáo "bình thường" (precision cao, recall thấp). **Chọn checkpoint theo `CE_LLM_Macro_F1`** (GREEN-style, tương quan cao với bác sĩ), không theo loss.

Quy trình:

```bash
RUN=$OUTPUT_ROOT/<SUFFIX>_stage2
# 1) Merge một epoch snapshot cụ thể (khớp fusion_type lúc train)
python src/merge_lora_weights_and_save_hf_model.py \
  --model_name_or_path google/medgemma-1.5-4b-it --model_type gemma3 \
  --axt2_enable True --axial_only False --fusion_type udml \
  --pretrain_vision_model weights/pretrained_ViT.bin --medgemma_adapter_enable True --mm_projector_type spp \
  --lora_r 8 --lora_alpha 32 \
  --model_with_lora $RUN/model_trainable_epoch4.bin --output_dir $RUN/merged_epoch4

# 2) Eval epoch đó
MODEL_PATH=$RUN/merged_epoch4 OUTPUT_DIR=$RUN/eval_epoch4 bash scripts/evaluate.sh <SUFFIX>
```

Merge + eval vài epoch (thường epoch ~4/6 là vùng ngọt), chọn `CE_LLM_Macro_F1` cao nhất, rồi `mv merged_epochN merged_hf`.

## Đánh Giá

```bash
export GROQ_API_KEY="..."     # cho metric chấm bằng LLM (llama-3.3-70b-versatile)
bash scripts/evaluate.sh [OUTPUT_SUFFIX] [MODEL_SUBDIR]
# vd: bash scripts/evaluate.sh medgemma_udml_ttd
```

Eval phải **cùng cấu hình ảnh như lúc train** (`AXT2_ENABLE` / `AXIAL_ONLY` / `SAGITTAL_MODALITY`). Các env override hữu ích: `MODEL_PATH`, `OUTPUT_DIR`, `DATA_ROOT`, `VAL_CSV`.

- Bước 1: `demo_csv.py` → `eval/eval_caption.csv` (predictions).
- Bước 2: `eval_caption_metrics.py` → `eval/eval_scores.csv` (BLEU/ROUGE/METEOR/BERTScore + CE-LLM). Có `--llm_resume` để tiếp tục khi bị rate-limit.

## Fusion (`src/model/base_fusion.py` + `udml_fusion.py`)

Mọi fusion nhận `(feat_sag, feat_ax)` dạng `[B, N, H]` → trả `[B, N, H]`, dùng chung slot `model.udml_fusion` nên trọng số được lưu/merge tự động. **Khi merge/eval phải truyền đúng `--fusion_type` như lúc train.**

| `FUSION_TYPE` | Cơ chế |
|---------------|--------|
| `udml` | Uncertainty+dependency weight → residual `(1+w)·feat` → **GateFusion backbone học được** (đúng shicaiwei123/UDML) |
| `gate` | Cổng `g=σ(W[sag;ax])`; `out = g·sag + (1−g)·ax` |
| `concat` | Nối kênh `[B,N,2H]` → `Linear(2H→H)` + GELU |
| `bilinear` | Low-rank bilinear (MFB): `(U·sag)⊙(V·ax) → Linear(rank→H)` |
| `elementwise` | Cộng/nhân/mean theo phần tử (tuỳ chọn blend học được) |
