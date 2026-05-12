# P2VG Project — Agent Coding Rules

## Package Layout
- Source lives in `src/p2vg/` (src layout).
- Internal imports use **relative imports**: `from .module import X`.
- Scripts in `scripts/` use **absolute imports**: `from p2vg.module import X`.

## Key Modules
- `src/p2vg/data/dataset.py` — `SpineCapDataset`, `SpineCapSegDataset`
- `src/p2vg/data/templates.py` — prompt templates
- `src/p2vg/model/udml_fusion.py` — UDML dynamic fusion (core of dynamicfusion branch)
- `src/p2vg/model/arch.py` — `LamedMetaModel`, `LamedMetaForCausalLM`
- `src/p2vg/model/gemma3.py` — `LamedGemma3ForCausalLM`
- `src/p2vg/train/train.py` — training entry point (called by DeepSpeed)
- `src/p2vg/eval/metrics.py` — CE and NLG metric functions

## UDML-Specific Rules
- `UDMLFusion` replaces gated sigmoid fusion; do NOT revert to `modal_fusion`.
- ViT weight loading uses `strict=True`; do NOT change to `strict=False`.
- `mm_projector` must be built BEFORE `udml_fusion` (MedGemma adapter depends on projector input dim).
- `encode_all_images` accepts `sag_noise_variance` and `ax_noise_variance`.
- Training tracks `model.udml_aux_loss`, `model.udml_sag_image_features`, `model.udml_ax_image_features`.

## Tools & Style
- Logging: **loguru** only — never `print()`.
- Package management: **uv** — never bare `pip`.
- Training: **PyTorch Lightning** for new training loops; existing code uses `LaMedTrainer` from `LaMed`.
- Linting: `uv run ruff check --fix .` and `uv run ruff format .`
- Type checking: `uv run pyright`

## Dataset
- Dataset root: `dataset_ttd_256/`
- CSVs: `report/train.csv`, `report/val.csv`, `report/test.csv`
- Column names: `case_id`, `image_path` (NOT `images_path`), `Clinician's Notes`
- Path in CSV is relative: `Volume` — resolved as `data_root / image_path`
