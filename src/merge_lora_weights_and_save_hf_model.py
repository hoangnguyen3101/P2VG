import os
import torch
from typing import Optional
import transformers
from transformers import AutoTokenizer
from dataclasses import dataclass, field
from src.model.lamed_gemma3 import LamedGemma3ForCausalLM


@dataclass
class ModelArguments:
    version: Optional[str] = field(default="v0")
    model_name_or_path: Optional[str] = field(
        default="google/medgemma-1.5-4b-it",
        metadata={
            "help": "Path to the LLM or MLLM."
        },
    )
    model_type: Optional[str] = field(default=None, metadata={"help": "gemma3"})

    model_with_lora: Optional[str] = field(
        default="/storage/hoangnv/triplane_kfold/gemma3_fold3/model_with_lora.bin",
        metadata={"help": "Path to the model_with_lora.bin checkpoint from training."},
    )

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
    vision_tower: Optional[str] = field(default="vit3d")  # None, "vit3d"
    vision_select_layer: Optional[int] = field(default=-1)
    vision_select_feature: Optional[str] = field(default="patch")
    pretrain_vision_model: str = field(
        default=None,
        metadata={"help": "Path to pretrained model for ViT."},
    )
    freeze_vision_tower: bool = field(default=False)

    # axial (dual-encoder)
    axt2_enable: bool = field(default=True, metadata={"help": "Enable axial T2 dual encoder."})
    axial_only: bool = field(default=False, metadata={"help": "Use axial encoder only (no sagittal)."})

    # projector
    mm_projector_type: Optional[str] = field(default="spp")
    proj_layer_type: str = field(
        default="mlp",
        metadata={"help": "Type of projector in Perceiver. options: [linear, mlp]."},
    )
    proj_layer_num: int = field(
        default=2, metadata={"help": "Number of projectors in Perceiver."}
    )
    proj_pooling_type: str = field(
        default="spatial",
        metadata={
            "help": "Type of pooling in Perceiver. options: [spatial, sequence].",
        },
    )
    proj_pooling_size: int = field(
        default=2, metadata={"help": "Size of pooling in Perceiver."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    lora_enable: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"

    cache_dir: Optional[str] = field(default=None)
    output_dir: str = "/storage/hoangnv/triplane_kfold/gemma3_fold3/merged_hf/"


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    # Process of elimination: LoRA only targets on LLM backbone
    ignore_keywords = [
        "vision_tower",
        "vision_tower_ax",
        "udml_fusion",
        "mm_projector",
        "embed_tokens",
        "lm_head",
    ]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in ignore_keywords):
            continue
        if isinstance(module, cls):
            lora_module_names.add(name)
    return list(lora_module_names)


def main():
    global local_rank
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    print("=" * 20 + " Tokenizer preparation " + "=" * 20)
    # Load tokenizer from the given path with specified configurations
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        padding_side="right",
        trust_remote_code=True,
        local_files_only=True,
    )

    # Define and add special tokens
    special_token = {
        "additional_special_tokens": ["<im_patch>", "<bx_start>", "<bx_end>"]
    }
    tokenizer.add_special_tokens(special_token)

    # Gemma3 has a dedicated <pad> token (id=0), separate from <eos> (id=1).
    # Keep the default pad_token — do NOT override.

    # Convert special tokens to token IDs and set related arguments
    model_args.img_token_id = tokenizer.convert_tokens_to_ids("<im_patch>")
    model_args.vocab_size = len(tokenizer)
    print("vocab_size: ", model_args.vocab_size)

    print("=" * 20 + " Model preparation " + "=" * 20)
    if "gemma3" in model_args.model_type:
        # Same manual weight loading as custom_train.py for MedGemma/Gemma3
        from transformers import AutoConfig, Gemma3TextConfig
        from src.model.lamed_gemma3 import LamedGemma3Config

        raw_config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, cache_dir=training_args.cache_dir, local_files_only=True
        )

        # Extract text_config from multimodal Gemma3Config
        if hasattr(raw_config, 'text_config'):
            text_cfg = raw_config.text_config
            if isinstance(text_cfg, dict):
                text_cfg = Gemma3TextConfig(**text_cfg)
            config_dict = text_cfg.to_dict()
        else:
            config_dict = raw_config.to_dict()

        for k in ['model_type', 'architectures', 'transformers_version']:
            config_dict.pop(k, None)

        lamed_config = LamedGemma3Config(**config_dict)
        model = LamedGemma3ForCausalLM(lamed_config)

        # Fix: Load and remap weights from multimodal checkpoint
        import glob
        from safetensors.torch import load_file
        
        if os.path.isdir(model_args.model_name_or_path):
            model_path = model_args.model_name_or_path
        else:
            from huggingface_hub import snapshot_download
            model_path = snapshot_download(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
            )
        print(f"Loading base weights from: {model_path}")

        shard_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
        full_state_dict = {}
        for shard in shard_files:
            full_state_dict.update(load_file(shard, device="cpu"))

        remapped = {}
        for k, v in full_state_dict.items():
            if k.startswith("language_model."):
                new_key = k[len("language_model."):]
                remapped[new_key] = v

        model.load_state_dict(remapped, strict=False)
        model.tie_weights()
        print("Weights tied: lm_head ↔ embed_tokens")
        del full_state_dict, remapped
    else:
        raise ValueError(f"Unknown Model Type {model_args.model_type}")

    # Enable MedGemma adapter mode
    model.config.medgemma_adapter = True

    # initialize vision modules on LLM (builds ViT + projector structure)
    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args)

    model_args.num_new_tokens = 3  # <im_patch>, <bx_start>, <bx_end>
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
        print("Adding LoRA adapters only on LLM.")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    print("=" * 20 + " Load weights with LoRA " + "=" * 20)
    print(f"Loading from: {model_args.model_with_lora}")
    state_dict = torch.load(model_args.model_with_lora, map_location="cpu")
    # Strip PEFT prefixes only if they don't match the current model structure
    # If the current model is a PeftModel, it expects base_model.model. prefix.
    # If the checkpoint already has it, we don't strip it.
    first_key = list(state_dict.keys())[0]
    has_peft_prefix = first_key.startswith("base_model.model.")
    
    # Check if the model we are loading into is a PeftModel
    is_peft_model = hasattr(model, "base_model")
    
    new_state_dict = {}
    if has_peft_prefix and not is_peft_model:
        # Strip if checkpoint has it but model doesn't (Inference case)
        for k, v in state_dict.items():
            new_state_dict[k.replace("base_model.model.", "")] = v
    elif not has_peft_prefix and is_peft_model:
        # Add if model has it but checkpoint doesn't
        for k, v in state_dict.items():
            new_state_dict["base_model.model." + k] = v
    else:
        # Both have it or both don't have it, load as is
        new_state_dict = state_dict
        
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"[WARNING] Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"[WARNING] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    print("=" * 20 + " Merge weights with LoRA " + "=" * 20)
    model = model.merge_and_unload()
    model = model.to(torch.bfloat16)  # Save in bfloat16 to match base model size (~8GB vs ~16GB in float32)

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    model.model.config.architectures = model.__class__.__name__
    model._name_or_path = training_args.output_dir

    print("=" * 20 + " Save pretrained " + "=" * 20)
    model.config.save_pretrained(training_args.output_dir)
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    print("=" * 20 + " Finish " + "=" * 20)
    print(f"Merged model saved to: {training_args.output_dir}")


if __name__ == "__main__":
    main()
