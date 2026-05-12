# P2VG DynamicFusion

Branch `dynamicfusion` dùng để fine-tune mô hình captioning y khoa 3D cho MRI cột sống, dựa trên MedGemma/Gemma3. Branch này thêm nhánh hai góc nhìn cho ảnh sagittal và axial T2, đồng thời bổ sung module dynamic fusion theo hướng UDML để ước lượng độ bất định của từng modality và trộn visual tokens trước khi sinh báo cáo.

## Thành Phần Chính

- `src/custom_train.py`: entrypoint train cho MedGemma/Gemma3 + ViT3D + LoRA.
- `src/model/udml_fusion.py`: module fusion động giữa sagittal/axial với variance supervision.
- `src/model/lamed_arch.py`: logic forward/generation multimodal, có hỗ trợ nhánh axial.
- `src/multi_dataset.py`: dataset loader cho NIfTI volumes và CSV báo cáo.
- `src/merge_lora_weights_and_save_hf_model.py`: merge LoRA/trainable weights thành thư mục model kiểu Hugging Face.
- `src/demo_csv.py`: chạy caption generation trên một split CSV và ghi `eval_caption.csv`.
- `src/eval_caption_metrics.py`: tính metric caption từ CSV kết quả sinh ra.

Code LaMed/M3D gốc được giữ trong `M3D/` và vẫn được import trong quá trình train.

## Môi Trường

Tạo conda environment từ file đã đóng gói:

```bash
conda env create -f environment.yaml
conda activate p2vg
```

Nếu environment đã tồn tại:

```bash
conda env update -f environment.yaml --prune
conda activate p2vg
```

Thiết lập import path của repo trước khi chạy script:

```bash
export P2VG_ROOT="$(pwd)"
export PYTHONPATH="$P2VG_ROOT:$P2VG_ROOT/M3D:$PYTHONPATH"
```

Environment này dùng Python 3.10, PyTorch 2.6.0 với CUDA 12.4 wheels, DeepSpeed 0.18.8, Transformers 5.3.0, PEFT 0.18.1, MONAI, nibabel, cùng các package metric/utility cần cho branch này.

## Định Dạng Dữ Liệu

`SpineCapDataset` cần CSV có các cột sau:

- `case_id`: mã sample. Loader lấy subject number từ phần suffix sau dấu `_` cuối cùng.
- `images_path` hoặc `image_path`: đường dẫn tuyệt đối tới thư mục volume của subject, hoặc đường dẫn tương đối so với `--data_root`.
- `Clinician's Notes`: nội dung report/caption mục tiêu.

Với mỗi sample, loader sẽ tìm trong thư mục volume:

- Sagittal volume: `sub-{sub_num}_{sagittal_modality}.nii.gz` hoặc `{sub_num}_{sagittal_modality}.nii.gz`
- Axial T2 volume khi bật `--axt2_enable True`: `sub-{sub_num}_axt2.nii.gz`

Các giá trị thường dùng cho `--sagittal_modality` là `t1`, `t2`, `t1t2`, và `fused`. Volume được đọc bằng nibabel, normalize theo percentile, và đưa về dạng `[1, D, H, W]`.

Ví dụ cấu trúc TTD đã normalize:

```text
dataset_ttd_256/
  report/
    train.csv
    val.csv
  Volume/
    sub-0001_fused.nii.gz
    sub-0001_axt2.nii.gz
```

Tải dữ liệu TTD đã chuẩn bị bằng `gdown`:

```bash
pip install gdown
gdown --id 1JC48AF33eIlq-sTxG54NTJZfjQqE4tXs -O dataset_ttd_256.zip
unzip dataset_ttd_256.zip
```

## Weights Cần Có

Script train giả định đã có pretrained visual weights cục bộ:

```text
weights/
  pretrained_ViT.bin
```

Tải pretrained ViT từ Hugging Face:

```text
https://huggingface.co/GoodBaiBai88/M3D-CLIP/blob/main/pretrained_ViT.bin
```

Base language model thường dùng là `google/medgemma-1.5-4b-it`. Với MedGemma adapter mode, pretrained visual projector được tái sử dụng trực tiếp từ checkpoint MedGemma, nên path train hiện tại không cần `mm_projector.bin` riêng. Nếu model chưa có trong cache local, cần cấu hình quyền truy cập Hugging Face trước.

## Training

Launcher chính được giữ ở:

```bash
bash src/finetune_lora.sh
```

Lệnh DeepSpeed tối giản:

```bash
deepspeed src/custom_train.py \
  --version v0 \
  --model_name_or_path google/medgemma-1.5-4b-it \
  --model_type gemma3 \
  --lora_enable True \
  --vision_tower vit3d \
  --axt2_enable True \
  --axial_only False \
  --freeze_vision_tower True \
  --pretrain_vision_model /path/to/pretrained_ViT.bin \
  --bf16 True \
  --data_root /path/to/dataset \
  --amos_train_cap_data_path /path/to/train.csv \
  --amos_validation_cap_data_path /path/to/val.csv \
  --output_dir /path/to/output \
  --num_train_epochs 5 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --eval_strategy epoch \
  --save_strategy epoch \
  --learning_rate 3e-5 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --sagittal_modality fused \
  --report_to wandb
```

Một số flag DynamicFusion/UDML hữu ích:

- `--axt2_enable True`: bật nhánh axial T2.
- `--axial_only True`: chỉ dùng nhánh axial làm visual stream.
- `--sagittal_modality fused`: chọn suffix file sagittal.
- `--udml_noise_enable True`: thêm controlled noise để supervise variance.
- `--udml_var_loss_weight 0.1`: trọng số của variance auxiliary loss.
- `--udml_lm_aux_enable True`: bật LM auxiliary loss riêng cho sagittal-only và axial-only.
- `--udml_lm_aux_loss_weight 1.0`: trọng số cho unimodal LM auxiliary losses.

Trong lúc train LoRA, checkpoint sẽ lưu các tham số trainable dưới dạng `model_with_lora*.bin` trong `--output_dir`.

## Merge LoRA Checkpoint

Sau khi train, merge checkpoint trainable đã lưu thành một thư mục model kiểu Hugging Face:

```bash
python -u src/merge_lora_weights_and_save_hf_model.py \
  --model_name_or_path google/medgemma-1.5-4b-it \
  --model_type gemma3 \
  --model_with_lora /path/to/output/model_with_lora.bin \
  --output_dir /path/to/output/merged_hf \
  --vision_tower vit3d \
  --axt2_enable True \
  --axial_only False \
  --pretrain_vision_model /path/to/pretrained_ViT.bin
```

`demo_csv.py` sẽ load model từ thư mục merged này.

## Inference

Dùng script có sẵn hoặc chạy trực tiếp:

```bash
python -u src/demo_csv.py \
  --model_name_or_path /path/to/merged_hf \
  --data_root /path/to/dataset \
  --amos_validation_cap_data_path /path/to/test.csv \
  --output_dir /path/to/eval_output \
  --axt2_enable True \
  --axial_only False \
  --sagittal_modality fused \
  --do_sample False
```

File kết quả sinh ra là:

```text
/path/to/eval_output/eval_caption.csv
```

## Đánh Giá

Chạy đánh giá caption metric trên CSV đã sinh. Script cần API key của Groq/Grok; truyền bằng biến môi trường `GROQ_API_KEY` hoặc flag `--llm_api_key`.

```bash
export GROQ_API_KEY="your_api_key_here"

python src/eval_caption_metrics.py \
  --input_csv /path/to/eval_output/eval_caption.csv \
  --output_csv /path/to/eval_output/eval_scores.csv \
  --llm_model llama-3.3-70b-versatile
```

Xem `python src/eval_caption_metrics.py --help` để biết chính xác các tuỳ chọn metric mà script local hỗ trợ.

## Ghi Chú

- Script hiện có một số path theo máy hiện tại như `/storage/hoangnv` và `/home/hoangnv`; cần sửa lại để phù hợp hơn với máy của bạn
- Để import ổn định, nên chạy từ repo root hoặc set `PYTHONPATH` như hướng dẫn phía trên.
