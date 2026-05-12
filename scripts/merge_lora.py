"""Merge LoRA adapters into base model and save as HuggingFace checkpoint."""
import glob
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from loguru import logger
from transformers import AutoTokenizer

from p2vg.model.gemma3 import LamedGemma3ForCausalLM
from p2vg.train.utils import find_all_linear_names


@dataclass
class ModelArguments:
    version: Optional[str] = field(default="v0")
    model_name_or_path: Optional[str] = field(
        default="google/medgemma-1.5-4b-it",
        metadata={"help": "Path to the LLM or MLLM."},
    )
    model_type: Optional[str] = field(default=None, metadata={"help": "gemma3"})
    model_with_lora: Optional[str] = field(
        default=None,
        metadata={"help": "Path to model_with_lora.bin checkpoint from training."},
    )
    freeze_backbone: bool = field(default=False)
    pretrain_mllm: Optional[str] = field(default=None)
    tune_mm_mlp_adapter: bool = field(default=False)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)

    image_channel: int = field(default=1)
    image_size: tuple = field(default=(32, 256, 256))
    patch_size: tuple = field(default=(4, 16, 16))

    vision_tower: Optional[str] = field(default="vit3d")
    vision_select_layer: Optional[int] = field(default=-1)
    vision_select_feature: Optional[str] = field(default="patch")
    pretrain_vision_model: str = field(default=None)
    freeze_vision_tower: bool = field(default=False)

    axt2_enable: bool = field(default=True)
    axial_only: bool = field(default=False)

    mm_projector_type: Optional[str] = field(default="spp")
    proj_layer_type: str = field(default="mlp")
    proj_layer_num: int = field(default=2)
    proj_pooling_type: str = field(default="spatial")
    proj_pooling_size: int = field(default=2)

    udml_var_loss_weight: float = field(default=0.1)
    udml_lm_aux_enable: bool = field(default=False)
    udml_lm_aux_loss_weight: float = field(default=1.0)


@dataclass
class MergeArguments(transformers.TrainingArguments):
    lora_enable: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    cache_dir: Optional[str] = field(default=None)
    output_dir: str = field(default="./outputs/merged_hf")


def main():
    parser = transformers.HfArgumentParser((ModelArguments, MergeArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    logger.info("=" * 20 + " Tokenizer preparation " + "=" * 20)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        padding_side="right",
        trust_remote_code=True,
        local_files_only=True,
    )
    special_token = {"additional_special_tokens": ["<im_patch>", "<bx_start>", "<bx_end>"]}
    tokenizer.add_special_tokens(special_token)
    model_args.img_token_id = tokenizer.convert_tokens_to_ids("<im_patch>")
    model_args.vocab_size = len(tokenizer)

    logger.info("=" * 20 + " Model preparation " + "=" * 20)
    if "gemma3" in model_args.model_type:
        from transformers import AutoConfig, Gemma3TextConfig
        from safetensors.torch import load_file
        from p2vg.model.gemma3 import LamedGemma3Config

        raw_config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, cache_dir=training_args.cache_dir, local_files_only=True
        )
        if hasattr(raw_config, "text_config"):
            text_cfg = raw_config.text_config
            if isinstance(text_cfg, dict):
                text_cfg = Gemma3TextConfig(**text_cfg)
            config_dict = text_cfg.to_dict()
        else:
            config_dict = raw_config.to_dict()

        for k in ["model_type", "architectures", "transformers_version"]:
            config_dict.pop(k, None)

        lamed_config = LamedGemma3Config(**config_dict)
        model = LamedGemma3ForCausalLM(lamed_config)

        model_path = (
            model_args.model_name_or_path
            if os.path.isdir(model_args.model_name_or_path)
            else __import__("huggingface_hub").snapshot_download(
                model_args.model_name_or_path, cache_dir=training_args.cache_dir
            )
        )
        shard_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
        full_state_dict = {}
        for shard in shard_files:
            full_state_dict.update(load_file(shard, device="cpu"))

        remapped = {
            k[len("language_model."):]: v
            for k, v in full_state_dict.items()
            if k.startswith("language_model.")
        }
        model.load_state_dict(remapped, strict=False)
        model.tie_weights()
        del full_state_dict, remapped
    else:
        raise ValueError(f"Unknown Model Type {model_args.model_type}")

    model.config.medgemma_adapter = True

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args)

    model_args.num_new_tokens = 3
    model.initialize_vision_tokenizer(model_args, tokenizer)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    logger.info("Loading LoRA weights from: {}", model_args.model_with_lora)
    state_dict = torch.load(model_args.model_with_lora, map_location="cpu")
    first_key = list(state_dict.keys())[0]
    has_peft_prefix = first_key.startswith("base_model.model.")
    is_peft_model = hasattr(model, "base_model")

    if has_peft_prefix and not is_peft_model:
        state_dict = {k.replace("base_model.model.", ""): v for k, v in state_dict.items()}
    elif not has_peft_prefix and is_peft_model:
        state_dict = {"base_model.model." + k: v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Missing keys ({}): {}...", len(missing), missing[:5])
    if unexpected:
        logger.warning("Unexpected keys ({}): {}...", len(unexpected), unexpected[:5])

    logger.info("Merging LoRA weights...")
    model = model.merge_and_unload()
    model = model.to(torch.bfloat16)

    os.makedirs(training_args.output_dir, exist_ok=True)
    model.config.save_pretrained(training_args.output_dir)
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info("Merged model saved to: {}", training_args.output_dir)


if __name__ == "__main__":
    main()
