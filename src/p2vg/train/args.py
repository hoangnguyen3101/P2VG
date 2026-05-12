from dataclasses import dataclass, field
from typing import Optional

import transformers


@dataclass
class ModelArguments:
    version: Optional[str] = field(default="v0")
    model_name_or_path: Optional[str] = field(
        default="google/gemma-3-4b-it",
        metadata={"help": "Path to the LLM or MLLM."},
    )
    model_type: Optional[str] = field(default=None, metadata={"help": "gemma3"})

    freeze_backbone: bool = field(default=False)
    pretrain_mllm: Optional[str] = field(default=None)

    tune_mm_mlp_adapter: bool = field(
        default=False,
        metadata={"help": "Used in pretrain: tune mm_projector and embed_tokens"},
    )
    pretrain_mm_mlp_adapter: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained mm_projector and embed_tokens."},
    )

    # image
    image_channel: int = field(default=1)
    image_size: tuple = field(default=(32, 256, 256))
    patch_size: tuple = field(default=(4, 16, 16))

    # vision
    vision_tower: Optional[str] = field(default="vit3d")
    vision_select_layer: Optional[int] = field(default=-1)
    vision_select_feature: Optional[str] = field(default="patch")
    pretrain_vision_model: str = field(
        default=None, metadata={"help": "Path to pretrained model for ViT."}
    )
    freeze_vision_tower: bool = field(default=False)

    # axial
    axt2_enable: bool = field(default=False)
    axial_only: bool = field(default=False)

    # projector
    mm_projector_type: Optional[str] = field(default="spp", metadata={"help": "spp"})
    proj_layer_type: str = field(
        default="mlp",
        metadata={"help": "Type of layer in projector. options: [linear, mlp]."},
    )
    proj_layer_num: int = field(default=2, metadata={"help": "Number of layers in projector."})
    proj_pooling_type: str = field(
        default="spatial",
        metadata={"help": "Type of pooling in projector. options: [spatial, sequence]."},
    )
    proj_pooling_size: int = field(default=2, metadata={"help": "Size of pooling in projector."})

    # UDML
    udml_var_loss_weight: float = field(default=0.1)
    udml_lm_aux_enable: bool = field(
        default=False,
        metadata={"help": "Add sagittal-only and axial-only LM losses analogous to UDML unimodal CE losses."},
    )
    udml_lm_aux_loss_weight: float = field(default=1.0)


@dataclass
class DataArguments:
    data_root: str = field(
        default="./Data/data/", metadata={"help": "Root directory for all data."}
    )
    amos_train_cap_data_path: str = field(
        default="./Data/data/M3D_Cap_npy/M3D_Cap.json",
        metadata={"help": "Path to caption training data."},
    )
    amos_validation_cap_data_path: str = field(
        default="./Data/data/M3D_Cap_npy/M3D_Cap.json",
        metadata={"help": "Path to caption validation data."},
    )
    sagittal_modality: str = field(
        default="t1",
        metadata={"help": "Modality for sagittal images (t1, t2, fused)."},
    )
    udml_noise_enable: bool = field(
        default=False,
        metadata={"help": "Inject controlled Gaussian noise for UDML uncertainty supervision."},
    )
    udml_noise_prob: float = field(default=0.5)
    udml_noise_min: int = field(default=2)
    udml_noise_max: int = field(default=12)
    udml_noise_std_scale: float = field(default=0.02)
    udml_var_loss_weight: float = field(default=0.1)
    udml_lm_aux_enable: bool = field(
        default=False,
        metadata={"help": "Add sagittal-only and axial-only LM losses."},
    )
    udml_lm_aux_loss_weight: float = field(default=1.0)

    vqa_data_train_path: str = field(
        default="./Data/data/M3D-VQA/M3D_VQA_train.csv",
        metadata={"help": "Path to training VQA data."},
    )
    vqa_data_val_path: str = field(
        default="./Data/data/M3D-VQA/M3D_VQA_val.csv",
        metadata={"help": "Path to validation VQA data."},
    )
    vqa_data_test_path: str = field(
        default="./Data/data/M3D-VQA/M3D_VQA_test.csv",
        metadata={"help": "Path to testing VQA data."},
    )
    vqa_yn_data_train_path: str = field(
        default="./Data/data/M3D-VQA/M3D_VQA_yn_train.csv",
        metadata={"help": "Path to training VQA Yes or No data."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # lora
    lora_enable: bool = False
    lora_r: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    lora_weight_path: str = ""

    # training
    model_max_length: int = field(default=512)
    tune_mm_mlp_adapter: bool = field(default=False)
    resume_from_checkpoint: Optional[str] = field(default=None)
    remove_unused_columns: bool = field(default=False)
