"""Run inference on a validation CSV and write predictions to eval_caption.csv."""
import csv
import os
import random
from dataclasses import dataclass, field

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from p2vg.data.dataset import SpineCapDataset
from p2vg.model.gemma3 import LamedGemma3ForCausalLM


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def parse_args(args=None):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="./outputs/merged_hf")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--data_root", type=str, default="./")
    parser.add_argument("--amos_train_cap_data_path", type=str, default="./Data/data")
    parser.add_argument("--amos_validation_cap_data_path", type=str, default="./test_split.csv")
    parser.add_argument("--output_dir", type=str, default="./")
    parser.add_argument("--proj_out_num", type=int, default=256)
    parser.add_argument("--axt2_enable", action="store_true", default=False)
    parser.add_argument("--axial_only", action="store_true", default=False)
    parser.add_argument("--sagittal_modality", type=str, default="fused", choices=["t1", "t2", "t1t2", "fused"])
    return parser.parse_args(args)


def main():
    seed_everything(42)
    args = parse_args()
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.max_length,
        padding_side="right",
        trust_remote_code=True,
        local_files_only=True,
    )

    from p2vg.model.gemma3 import LamedGemma3Config

    config = LamedGemma3Config.from_pretrained(args.model_name_or_path, local_files_only=True)
    model = LamedGemma3ForCausalLM(config)

    @dataclass
    class _VisionArgs:
        vision_tower: str = "vit3d"
        vision_select_layer: int = -1
        vision_select_feature: str = "patch"
        pretrain_vision_model: str = None
        pretrain_mm_mlp_adapter: str = None
        freeze_vision_tower: bool = False
        axt2_enable: bool = args.axt2_enable
        axial_only: bool = args.axial_only
        image_channel: int = getattr(config, "image_channel", 1)
        image_size: tuple = tuple(getattr(config, "image_size", (32, 256, 256)))
        patch_size: tuple = tuple(getattr(config, "patch_size", (4, 16, 16)))
        mm_projector_type: str = getattr(config, "mm_projector_type", "spp")
        proj_layer_type: str = getattr(config, "proj_layer_type", "mlp")
        proj_layer_num: int = getattr(config, "proj_layer_num", 2)
        proj_pooling_type: str = getattr(config, "proj_pooling_type", "spatial")
        proj_pooling_size: int = getattr(config, "proj_pooling_size", 2)
        img_token_id: int = tokenizer.convert_tokens_to_ids("<im_patch>")
        vocab_size: int = len(tokenizer)
        num_new_tokens: int = 3
        udml_var_loss_weight: float = 0.1
        udml_lm_aux_enable: bool = False
        udml_lm_aux_loss_weight: float = 1.0

    vision_args = _VisionArgs()
    model.get_model().initialize_vision_modules(model_args=vision_args)

    from safetensors.torch import load_file

    safetensors_path = os.path.join(args.model_name_or_path, "model.safetensors")
    bin_path = os.path.join(args.model_name_or_path, "merged_model.bin")
    if os.path.exists(safetensors_path):
        ckpt = load_file(safetensors_path)
    else:
        ckpt = torch.load(bin_path, map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model.tie_weights()
    logger.info("Loaded merged model from: {}", args.model_name_or_path)

    model = model.to(device=device)
    model.eval()

    test_dataset = SpineCapDataset(args, tokenizer, target_shape=(256, 256, 32), mode="validation")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=32,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "eval_caption.csv")

    with open(output_path, mode="w") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["Question", "Ground Truth", "pred"])
        with torch.no_grad():
            for sample in tqdm(test_dataloader):
                question = sample["question"]
                answer = sample["answer"]

                input_id = tokenizer(question, return_tensors="pt")["input_ids"].to(device=device)
                image = sample["image"].to(device=device)
                image_ax = sample["image_ax"].to(device=device) if "image_ax" in sample else None

                generation = model.generate(
                    image,
                    images_ax=image_ax,
                    inputs=input_id,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    top_p=args.top_p,
                    temperature=args.temperature,
                )
                generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)

                logger.info("ANSWER: {}", answer)
                logger.info("PREDICTION: {}", generated_texts)
                writer.writerow([question[0], answer[0], generated_texts[0]])


if __name__ == "__main__":
    main()
