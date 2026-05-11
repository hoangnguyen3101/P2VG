import os
import logging
from typing import Optional, List, Dict
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer
from dataclasses import dataclass, field
from src.multi_dataset import(
    SpineCapDataset,
)
from src.model.lamed_gemma3 import LamedGemma3ForCausalLM
from LaMed.src.train.lamed_trainer import LaMedTrainer


local_rank = None
# os.environ["LOCAL_RANK"] = "-1"  

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


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
    vision_tower: Optional[str] = field(default="vit3d")  # None, "vit3d"
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
    proj_layer_num: int = field(
        default=2, metadata={"help": "Number of layers in projector."}
    )
    proj_pooling_type: str = field(
        default="spatial",
        metadata={
            "help": "Type of pooling in projector. options: [spatial, sequence]."
        },
    )
    proj_pooling_size: int = field(
        default=2, metadata={"help": "Size of pooling in projector."}
    )


@dataclass
class DataArguments:
    data_root: str = field(
        default="./Data/data/", metadata={"help": "Root directory for all data."}
    )

    # caption data
    cap_data_path: str = field(
        default="./Data/data/M3D_Cap_npy/M3D_Cap.json",
        metadata={"help": "Path to caption data."},
    )

    # caption data
    amos_train_cap_data_path: str = field(
        default="./Data/data/M3D_Cap_npy/M3D_Cap.json",
        metadata={"help": "Path to caption data."},
    )

    # caption data
    amos_validation_cap_data_path: str = field(
        default="./Data/data/M3D_Cap_npy/M3D_Cap.json",
        metadata={"help": "Path to amos caption data."},
    )

    sagittal_modality: str = field(
        default="t1",
        metadata={"help": "Modality for sagittal images (t1, t2, fused)."}
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
        metadata={"help": "Add sagittal-only and axial-only LM losses, analogous to UDML unimodal CE losses."},
    )
    udml_lm_aux_loss_weight: float = field(default=1.0)

    # VQA data
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
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"

    cache_dir: Optional[str] = field(default=None)
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(
        default=512,  # 512
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    ddp_backend: str = "nccl"
    ddp_find_unused_parameters: bool = True
    optim: str = field(default="adamw_torch")

    # This is set up to facilitate debugging, pls config these in bash file in training.
    bf16: bool = False
    output_dir: str = "./output_gemma3"
    num_train_epochs: float = 1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    eval_strategy: str = "steps"
    eval_accumulation_steps: int = 1
    eval_steps: float = 0.04
    save_strategy: str = "steps"
    save_steps: int = 2000
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "loss"
    greater_is_better: bool = False
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: float = 10  # 0.001
    gradient_checkpointing: bool = False  # train fast
    dataloader_pin_memory: bool = True  # fast
    dataloader_num_workers: int = 0
    report_to: str = "tensorboard"


def compute_metrics(eval_preds):
    labels_ids = eval_preds.label_ids
    pred_ids = eval_preds.predictions

    labels = labels_ids[:, 1:]
    preds = pred_ids[:, :-1]

    labels_flatten = labels.reshape(-1)
    preds_flatten = preds.reshape(-1)
    valid_indices = np.where(labels_flatten != -100)
    filtered_preds = preds_flatten[valid_indices]
    filtered_labels = labels_flatten[valid_indices]
    acc_score = sum(filtered_preds == filtered_labels) / len(filtered_labels)

    return {"accuracy": acc_score}


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(
                    f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}"
                )
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_projector_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {
        k: t
        for k, t in named_params
        if any(key_match in k for key_match in keys_to_match)
    }
    to_return = {
        k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()
    }
    return to_return


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save projector and embed_tokens in pretrain
        keys_to_match = ["mm_projector", "embed_tokens", "udml_fusion"]

        weight_to_save = get_mm_projector_state_maybe_zero_3(
            trainer.model.named_parameters(), keys_to_match
        )
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split("/")[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith("checkpoint-"):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(
                    weight_to_save,
                    os.path.join(mm_projector_folder, f"{current_folder}.bin"),
                )
            else:
                torch.save(
                    weight_to_save, os.path.join(output_dir, f"mm_projector.bin")
                )
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


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


@dataclass
class DataCollator:
    def __init__(self):
        pass

    def __call__(self, batch: list) -> dict:
        images, image_ax, input_ids, labels, attention_mask, sag_noise_variance, ax_noise_variance = tuple(
            [b[key] for b in batch]
            for key in (
                "image",
                "image_ax",
                "input_id",
                "label",
                "attention_mask",
                "sag_noise_variance",
                "ax_noise_variance",
            )
        )

        images = torch.cat([_.unsqueeze(0) for _ in images], dim=0)
        image_ax = torch.cat([_.unsqueeze(0) for _ in image_ax], dim=0)
        input_ids = torch.cat([_.unsqueeze(0) for _ in input_ids], dim=0)
        labels = torch.cat([_.unsqueeze(0) for _ in labels], dim=0)
        attention_mask = torch.cat([_.unsqueeze(0) for _ in attention_mask], dim=0)
        sag_noise_variance = torch.stack(sag_noise_variance, dim=0)
        ax_noise_variance = torch.stack(ax_noise_variance, dim=0)

        return_dict = dict(
            images=images,
            images_ax=image_ax,
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            sag_noise_variance=sag_noise_variance,
            ax_noise_variance=ax_noise_variance,
        )

        return return_dict


def main():
    global local_rank
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    data_args.axt2_enable = model_args.axt2_enable
    data_args.axial_only = model_args.axial_only
    model_args.udml_var_loss_weight = data_args.udml_var_loss_weight
    model_args.udml_lm_aux_enable = data_args.udml_lm_aux_enable
    model_args.udml_lm_aux_loss_weight = data_args.udml_lm_aux_loss_weight

    local_rank = training_args.local_rank

    rank0_print("=" * 20 + " Tokenizer preparation " + "=" * 20)
    # Load tokenizer from the given path with specified configurations
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        trust_remote_code=True,
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
    rank0_print("vocab_size: ", model_args.vocab_size)

    rank0_print("=" * 20 + " Model preparation " + "=" * 20)
    medgemma_proj_weights = {}  # Will be populated if MedGemma checkpoint contains projector weights
    if model_args.vision_tower is not None:
        if "gemma3" in model_args.model_type:
            # google/gemma-3-4b-it is multimodal (model_type="gemma3") with nested text_config.
            # Weights have "language_model.model." prefix, but our text-only model expects "model."
            from transformers import AutoConfig, Gemma3TextConfig
            from src.model.lamed_gemma3 import LamedGemma3Config
            from safetensors.torch import load_file
            import glob

            raw_config = AutoConfig.from_pretrained(
                model_args.model_name_or_path, cache_dir=training_args.cache_dir
            )

            # Extract text_config from multimodal Gemma3Config
            if hasattr(raw_config, 'text_config'):
                text_cfg = raw_config.text_config
                if isinstance(text_cfg, dict):
                    text_cfg = Gemma3TextConfig(**text_cfg)
                config_dict = text_cfg.to_dict()
            else:
                config_dict = raw_config.to_dict()

            # Remove keys not relevant to Gemma3TextConfig
            for k in ['model_type', 'architectures', 'transformers_version']:
                config_dict.pop(k, None)

            lamed_config = LamedGemma3Config(**config_dict)

            # Step 1: Create model from config (random weights)
            rank0_print("Creating LamedGemma3ForCausalLM from text config...")
            model = LamedGemma3ForCausalLM(lamed_config)

            # Step 2: Load and remap weights from multimodal checkpoint
            if os.path.isdir(model_args.model_name_or_path):
                model_path = model_args.model_name_or_path
            else:
                from huggingface_hub import snapshot_download
                model_path = snapshot_download(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                )
            rank0_print(f"Loading weights from: {model_path}")

            # Load all safetensor shards
            shard_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
            full_state_dict = {}
            for shard in shard_files:
                full_state_dict.update(load_file(shard, device="cpu"))

            # Remap: language_model.model.X -> model.X, language_model.lm_head.X -> lm_head.X
            remapped = {}
            for k, v in full_state_dict.items():
                if k.startswith("language_model."):
                    new_key = k[len("language_model."):]  # strip "language_model."
                    remapped[new_key] = v

            rank0_print(f"Remapped {len(remapped)} text backbone weights")
            missing, unexpected = model.load_state_dict(remapped, strict=False)
            rank0_print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
            if missing:
                rank0_print(f"Missing: {missing[:5]}...")

            # Preserve MedGemma projector weights for later injection
            medgemma_proj_weights = {}
            for k in ['multi_modal_projector.mm_input_projection_weight',
                      'multi_modal_projector.mm_soft_emb_norm.weight']:
                if k in full_state_dict:
                    medgemma_proj_weights[k] = full_state_dict[k]
            if medgemma_proj_weights:
                rank0_print(f"[MedGemma] Preserved {len(medgemma_proj_weights)} projector weights for reuse")
            del full_state_dict, remapped

            # Re-tie lm_head to embed_tokens (tie_word_embeddings=True)
            model.tie_weights()
            rank0_print("Weights tied: lm_head ↔ embed_tokens")
        else:
            raise ValueError(f"Unknown Model Type {model_args.model_type}")
    else:
        from transformers import Gemma3ForCausalLM
        model = Gemma3ForCausalLM.from_pretrained(
            model_args.model_name_or_path, cache_dir=training_args.cache_dir
        )

    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    model.enable_input_require_grads()
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # initialize vision modules on LLM
    if model_args.vision_tower is not None:
        # Enable MedGemma adapter mode if projector weights are available
        if medgemma_proj_weights:
            model.config.medgemma_adapter = True
        model.get_model().initialize_vision_modules(model_args=model_args)

        # Inject MedGemma pretrained projector weights
        if medgemma_proj_weights and hasattr(model.get_model().mm_projector, 'medgemma_projection'):
            proj = model.get_model().mm_projector
            w = medgemma_proj_weights['multi_modal_projector.mm_input_projection_weight']
            # MedGemma stores as [1152, 2560] (for matmul), nn.Linear expects [2560, 1152]
            proj.medgemma_projection.weight.data.copy_(w.T)
            rank0_print(f"[MedGemma] Loaded pretrained projection: {w.shape} → transposed to {w.T.shape}")
            if 'multi_modal_projector.mm_soft_emb_norm.weight' in medgemma_proj_weights:
                n = medgemma_proj_weights['multi_modal_projector.mm_soft_emb_norm.weight']
                proj.soft_emb_norm.weight.data.copy_(n)
                rank0_print(f"[MedGemma] Loaded pretrained RMSNorm: {n.shape}")
            del medgemma_proj_weights

    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = (
        model_args.tune_mm_mlp_adapter
    )
    if model_args.tune_mm_mlp_adapter:
        model.requires_grad_(False)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True

    model_args.num_new_tokens = 3  # <im_patch>, <bx_start>, <bx_end>
    model.initialize_vision_tokenizer(model_args, tokenizer)

    if model_args.pretrain_mllm:
        ckpt = torch.load(model_args.pretrain_mllm, map_location="cpu")
        model.load_state_dict(ckpt, strict=True)
        rank0_print("load pretrained MLLM weights.")

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
        rank0_print("Adding LoRA adapters only on LLM.")
        model = get_peft_model(model, lora_config)

        if training_args.lora_weight_path:
            ckpt = torch.load(training_args.lora_weight_path, map_location="cpu")
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            rank0_print(f"Loaded LoRA/start weights from {training_args.lora_weight_path}")
            rank0_print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

        # Enable training for non-LLM components
        trainable_keywords = [
            "mm_projector",
            "embed_tokens",
            "lm_head",
            "udml_fusion",
        ]
        # Optionally unfreeze vision tower
        if not model_args.freeze_vision_tower:
            trainable_keywords.append("vision_tower")

        for n, p in model.named_parameters():
            if any(x in n for x in trainable_keywords):
                p.requires_grad = True

        model.print_trainable_parameters()

    rank0_print("=" * 20 + " Dataset preparation " + "=" * 20)
    data_args.max_length = training_args.model_max_length
    data_args.proj_out_num = model.get_model().mm_projector.proj_out_num
    rank0_print("vision tokens output from projector: ", data_args.proj_out_num)

    train_dataset = SpineCapDataset(
        data_args,
        tokenizer,
        target_shape=(
            256,
            256,
            32,
        ),
        mode="train",
    )
    train_dataset = torch.utils.data.ConcatDataset(
        [
            train_dataset,
        ]
    )
    eval_dataset = SpineCapDataset(
        data_args,
        tokenizer,
        target_shape=(
            256,
            256,
            32,
        ),
        mode="validation",
    )
    eval_dataset = torch.utils.data.ConcatDataset(
        [
            eval_dataset,
        ]
    )
    data_collator = DataCollator()

    rank0_print("=" * 20 + " Training " + "=" * 20)
    trainer = LaMedTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    trainer.add_callback(MyCallback(trainer, tokenizer))
    # Thêm Early Stopping để tránh overfit
    trainer.add_callback(transformers.EarlyStoppingCallback(early_stopping_patience=2))
    
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_state()
    model.config.use_cache = True

    rank0_print("=" * 20 + " Save model " + "=" * 20)
    if training_args.lora_enable:
        state_dict_with_lora = model.state_dict()
        torch.save(
            state_dict_with_lora,
            os.path.join(training_args.output_dir, "model_with_lora.bin"),
        )
    else:
        safe_save_model_for_hf_trainer(
            trainer=trainer, output_dir=training_args.output_dir
        )


from transformers import TrainerCallback


class MyCallback(TrainerCallback):
    "A callback that prints a message at the beginning of training"

    def __init__(self, trainer, tokenizer):
        self._trainer = trainer
        self._tok = tokenizer

    def on_evaluate(self, args, state, control, **kwargs):
        print("evaluating..")
        vl = self._trainer.get_eval_dataloader()

        self._trainer.model.eval()
        gt = None
        with torch.no_grad():
            for batch in vl:
                gt = batch
                input_ids = batch["input_ids"]
                labels = batch["labels"]
                pad_id = self._tok.pad_token_id
                prompt_ids = []
                prompt_masks = []

                for sample_input_ids, sample_labels in zip(input_ids, labels):
                    supervised = torch.nonzero(sample_labels != -100, as_tuple=False)
                    prompt_len = supervised[0].item() if supervised.numel() > 0 else sample_input_ids.size(0)
                    ids = sample_input_ids[:prompt_len]
                    prompt_ids.append(ids)
                    prompt_masks.append(torch.ones_like(ids))

                max_prompt_len = max(ids.size(0) for ids in prompt_ids)
                padded_prompt_ids = input_ids.new_full((len(prompt_ids), max_prompt_len), pad_id)
                padded_prompt_masks = input_ids.new_zeros((len(prompt_ids), max_prompt_len))
                for i, (ids, mask) in enumerate(zip(prompt_ids, prompt_masks)):
                    padded_prompt_ids[i, :ids.size(0)] = ids
                    padded_prompt_masks[i, :mask.size(0)] = mask

                generation = self._trainer.model.generate(
                    batch["images"],
                    images_ax=batch.get("images_ax"),
                    inputs=padded_prompt_ids,
                    attention_mask=padded_prompt_masks,
                    max_new_tokens=256,
                    do_sample=False,
                )
                break

        generated_texts = self._tok.batch_decode(generation, skip_special_tokens=True)
        gt_texts = self._tok.batch_decode(gt["input_ids"], skip_special_tokens=True)
        print("*" * 100)
        for g, p in zip(gt_texts, generated_texts):
            print("-" * 100)
            print(f"GROUNDTRUTH:\t{g}")
            print(f"\n\nPREDICTION: \t{p}")

        self._trainer.model.train()

    def on_save(self, args, state, control, **kwargs):
        print(f"EPOCH {state.global_step}: SAVING.....\n")
        # Only save trainable parameters to reduce checkpoint size (~4GB vs ~16GB)
        trainable_keys = {n for n, p in self._trainer.model.named_parameters() if p.requires_grad}
        full_state = self._trainer.model.state_dict()
        trainable_state_dict = {k: v for k, v in full_state.items() if k in trainable_keys}
        save_path = os.path.join(args.output_dir, f"model_with_lora_{state.global_step}.bin")
        torch.save(trainable_state_dict, save_path)
        print(f"Saved trainable params ({len(trainable_state_dict)} keys) to {save_path}")


if __name__ == "__main__":
    main()
