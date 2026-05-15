# P2VG — Spine MRI Captioning with UDML Dynamic Fusion

Branch `dynamicfusion` fine-tunes MedGemma/Gemma3 cho bài toán sinh báo cáo MRI cột sống 3D.
Pipeline bao gồm hai nhánh visual (sagittal + axial T2) và module UDML fusion để ước lượng độ bất định của từng modality, từ đó tính trọng số trộn adaptive trước khi sinh báo cáo.

## Cấu Trúc Dự Án

```
P2VG/
├── src/p2vg/
│   ├── data/dataset.py          # SpineCapDataset (NIfTI + CSV)
│   ├── model/
│   │   ├── arch.py              # LamedMetaModel — forward multimodal
│   │   ├── gemma3.py            # LamedGemma3ForCausalLM
│   │   └── udml_fusion.py       # UDMLFusion — dynamic weight fusion
│   └── train/
│       ├── args.py              # TrainingArguments (Pydantic + HF)
│       ├── train.py             # LightningModule wrapper
│       ├── collator.py          # DataCollator
│       └── utils.py             # LoRA linear name finder
├── scripts/
│   ├── setup.sh                 # One-time server setup
│   ├── train.sh                 # Train + auto-merge LoRA
│   ├── evaluate.sh              # Inference + metric scoring
│   ├── train.py                 # DeepSpeed entry point
│   ├── merge_lora.py            # Merge LoRA → HF model dir
│   ├── demo_csv.py              # Batch inference on CSV
│   └── eval_captions.py         # Caption metric scoring
├── configs/
│   └── ds_config_zero2.json     # DeepSpeed ZeRO-2 config
├── dataset_ttd_256/
│   └── report/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── weights/
│   └── pretrained_ViT.bin       # M3D pretrained ViT weights
└── pyproject.toml
```

## Cài Đặt (Lần Đầu Trên Server Mới)

```bash
git clone <repo_url>
cd P2VG
git checkout dynamicfusion
bash scripts/setup.sh
```

`setup.sh` sẽ:
1. Cài `uv` nếu chưa có
2. `uv sync` — cài toàn bộ Python dependencies
3. Smoke test imports
4. Tự động fix đường dẫn tuyệt đối trong CSV dataset
5. Kiểm tra `weights/pretrained_ViT.bin`
6. Kiểm tra PM2

## Dữ Liệu

Tải dataset TTD 256 đã chuẩn bị:

```bash
pip install gdown
gdown --id 1JC48AF33eIlq-sTxG54NTJZfjQqE4tXs -O dataset_ttd_256.zip
unzip dataset_ttd_256.zip
```

Cấu trúc thư mục sau khi giải nén:

```
dataset_ttd_256/
├── report/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
└── Volume/
    ├── sub-0001_fused.nii.gz
    ├── sub-0001_axt2.nii.gz
    └── ...
```

CSV cần các cột: `case_id`, `image_path` (hoặc `images_path`), `Clinician's Notes`.

## Weights

Tải pretrained ViT từ Hugging Face:

```
https://huggingface.co/GoodBaiBai88/M3D-CLIP/blob/main/pretrained_ViT.bin
```

Đặt vào `weights/pretrained_ViT.bin`.

Base language model: `google/medgemma-1.5-4b-it` (tải tự động khi train lần đầu, cần HF token).

## Training

```bash
bash scripts/train.sh          # fold 2 (mặc định)
bash scripts/train.sh 3        # fold 3
```

Sau khi train xong, script tự động merge LoRA vào base model và lưu tại `outputs/fold{N}/merged_hf/`.

Chạy background với PM2:

```bash
pm2 start --name p2vg-train --no-autorestart -- bash scripts/train.sh 2
pm2 logs p2vg-train --lines 200
```

GPU mặc định: GPU 1 (`CUDA_VISIBLE_DEVICES=1`). Override bằng:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh
```

### Các Flag UDML Quan Trọng

| Flag | Mặc định | Mô tả |
|------|----------|--------|
| `--udml_noise_enable` | `True` | Thêm controlled noise để supervise variance estimator |
| `--udml_noise_prob` | `0.5` | Xác suất noise được inject mỗi sample |
| `--udml_noise_min/max` | `2/12` | Khoảng số slice bị corrupt |
| `--udml_var_loss_weight` | `0.1` | Trọng số variance auxiliary loss |
| `--udml_lm_aux_enable` | `True` | Bật unimodal LM aux loss (sag-only + ax-only) |
| `--udml_lm_aux_loss_weight` | `1.0` | Trọng số unimodal LM aux loss |

## Đánh Giá

```bash
bash scripts/evaluate.sh       # dùng outputs/fold2/merged_hf
bash scripts/evaluate.sh 3     # fold 3
```

Cần `GROQ_API_KEY` cho LLM-based metrics:

```bash
export GROQ_API_KEY="your_key_here"
bash scripts/evaluate.sh
```

Kết quả lưu tại `outputs/fold{N}/eval_scores.csv`.

## Kiến Trúc UDML Fusion

`UDMLFusion` (`src/p2vg/model/udml_fusion.py`) tính trọng số động:

1. **Variance estimator** — MLP 2 lớp output log-variance; chuyển sang std via `(x * 0.5).exp()`
2. **Trọng số uncertainty** — `w_sag = 2σ_ax² / (σ_sag² + σ_ax²)` (modality nào bất định hơn → trọng số nhỏ hơn)
3. **Dependency calculator** — Zero-ablation qua `shared_aux_head` để đo mức đóng góp của từng modality
4. **Trọng số cuối** — Chia theo dependency rồi renormalize về tổng = 2

Variance estimator được supervise bằng MSE loss so với mức noise được inject trong quá trình training (`--udml_noise_enable True`).
