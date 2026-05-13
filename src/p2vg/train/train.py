import glob
import os
from typing import Optional

import numpy as np
import torch
import transformers
from loguru import logger
from transformers import AutoTokenizer

from p2vg.data.dataset import SpineCapDataset
from p2vg.model.gemma3 import LamedGemma3ForCausalLM
from p2vg.train.args import DataArguments, ModelArguments, TrainingArguments
from p2vg.train.callbacks import EvalGenerationCallback
from p2vg.train.collator import DataCollator
from p2vg.train.utils import (
    compute_metrics,
    find_all_linear_names,
    preprocess_logits_for_metrics,
    safe_save_model_for_hf_trainer,
)
from LaMed.src.train.lamed_trainer import LaMedTrainer

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        logger.info(" ".join(str(a) for a in args))


def main():
    global local_rank
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    data_args.axt2_enable = model_args.axt2_enable
    data_args.axial_only = model_args.axial_only
    model_args.udml_var_loss_weight = data_args.udml_var_loss_weight
    model_args.udml_lm_aux_enable = data_args.udml_lm_aux_enable
    model_args.udml_lm_aux_loss_weight = data_args.udml_lm_aux_loss_weight

    local_rank = training_args.local_rank

    rank0_print("=" * 20 + " Tokenizer preparation " + "=" * 20)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        trust_remote_code=True,
        local_files_only=True,
    )

    special_token = {"additional_special_tokens": ["<im_patch>", "<bx_start>", "<bx_end>"]}
    tokenizer.add_special_tokens(special_token)

    model_args.img_token_id = tokenizer.convert_tokens_to_ids("<im_patch>")
    model_args.vocab_size = len(tokenizer)
    rank0_print("vocab_size: ", model_args.vocab_size)

    rank0_print("=" * 20 + " Model preparation " + "=" * 20)
    medgemma_proj_weights = {}
    if model_args.vision_tower is not None:
        if "gemma3" in model_args.model_type:
            from transformers import AutoConfig, Gemma3TextConfig
            from safetensors.torch import load_file

            raw_config = AutoConfig.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                local_files_only=True,
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

            from p2vg.model.gemma3 import LamedGemma3Config
            lamed_config = LamedGemma3Config(**config_dict)

            rank0_print("Creating LamedGemma3ForCausalLM from text config...")
            model = LamedGemma3ForCausalLM(lamed_config)

            if os.path.isdir(model_args.model_name_or_path):
                model_path = model_args.model_name_or_path
            else:
                from huggingface_hub import snapshot_download
                model_path = snapshot_download(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    local_files_only=True,
                )
            rank0_print(f"Loading weights from: {model_path}")

            shard_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
            full_state_dict = {}
            for shard in shard_files:
                full_state_dict.update(load_file(shard, device="cpu"))

            remapped = {}
            for k, v in full_state_dict.items():
                if k.startswith("language_model."):
                    new_key = k[len("language_model."):]
                    remapped[new_key] = v

            rank0_print(f"Remapped {len(remapped)} text backbone weights")
            missing, unexpected = model.load_state_dict(remapped, strict=False)
            rank0_print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
            if missing:
                rank0_print(f"Missing: {missing[:5]}...")

            for k in [
                "multi_modal_projector.mm_input_projection_weight",
                "multi_modal_projector.mm_soft_emb_norm.weight",
            ]:
                if k in full_state_dict:
                    medgemma_proj_weights[k] = full_state_dict[k]
            if medgemma_proj_weights:
                rank0_print(f"[MedGemma] Preserved {len(medgemma_proj_weights)} projector weights for reuse")
            del full_state_dict, remapped

            model.tie_weights()
            rank0_print("Weights tied: lm_head ↔ embed_tokens")
        else:
            raise ValueError(f"Unknown Model Type {model_args.model_type}")
    else:
        from transformers import Gemma3ForCausalLM
        model = Gemma3ForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            local_files_only=True,
        )

    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    model.enable_input_require_grads()
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if model_args.vision_tower is not None:
        if medgemma_proj_weights:
            model.config.medgemma_adapter = True
        model.get_model().initialize_vision_modules(model_args=model_args)

        if medgemma_proj_weights and hasattr(model.get_model().mm_projector, "medgemma_projection"):
            proj = model.get_model().mm_projector
            w = medgemma_proj_weights["multi_modal_projector.mm_input_projection_weight"]
            proj.medgemma_projection.weight.data.copy_(w.T)
            rank0_print(f"[MedGemma] Loaded pretrained projection: {w.shape} → transposed to {w.T.shape}")
            if "multi_modal_projector.mm_soft_emb_norm.weight" in medgemma_proj_weights:
                n = medgemma_proj_weights["multi_modal_projector.mm_soft_emb_norm.weight"]
                proj.soft_emb_norm.weight.data.copy_(n)
                rank0_print(f"[MedGemma] Loaded pretrained RMSNorm: {n.shape}")
            del medgemma_proj_weights

    model.config.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    if model_args.tune_mm_mlp_adapter:
        model.requires_grad_(False)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True

    model_args.num_new_tokens = 3
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

        trainable_keywords = [
            "mm_projector",
            "embed_tokens",
            "lm_head",
            "udml_fusion",
        ]
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

    train_dataset = torch.utils.data.ConcatDataset(
        [SpineCapDataset(data_args, tokenizer, target_shape=(256, 256, 32), mode="train")]
    )
    eval_dataset = torch.utils.data.ConcatDataset(
        [SpineCapDataset(data_args, tokenizer, target_shape=(256, 256, 32), mode="validation")]
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
    trainer.add_callback(EvalGenerationCallback(trainer, tokenizer))
    trainer.add_callback(transformers.EarlyStoppingCallback(early_stopping_patience=5))

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_state()
    model.config.use_cache = True

    rank0_print("=" * 20 + " Save model " + "=" * 20)
    if training_args.lora_enable:
        best_lora_path = os.path.join(training_args.output_dir, "model_with_lora.bin")
        if os.path.exists(best_lora_path):
            rank0_print(f"Best LoRA checkpoint already saved at: {best_lora_path}")
        else:
            trainable_keys = {n for n, p in model.named_parameters() if p.requires_grad}
            state_dict_with_lora = {
                k: v.detach().cpu()
                for k, v in model.state_dict().items()
                if k in trainable_keys
            }
            torch.save(state_dict_with_lora, best_lora_path)
            rank0_print(f"Saved final trainable LoRA params to: {best_lora_path}")
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    main()
