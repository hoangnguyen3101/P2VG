import os
import sys
import torch
from transformers import AutoTokenizer, AutoConfig

# Add project root to sys.path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'M3D'))

print("Importing model classes...")
from src.model.lamed_gemma3 import LamedGemma3ForCausalLM, LamedGemma3Config

model_path = "/storage/hoangnv/triplane_kfold/gemma3_pka_fused_gated_fold3/merged_hf_54"

print("Loading config...")
config = LamedGemma3Config.from_pretrained(model_path, local_files_only=True)
print("Config loaded:", config.model_type)

print("Creating model skeleton...")
model = LamedGemma3ForCausalLM(config)
print("Model skeleton created.")

print("Loading weights...")
ckpt = torch.load(os.path.join(model_path, "merged_model.bin"), map_location="cpu") if os.path.exists(os.path.join(model_path, "merged_model.bin")) else None
if not ckpt:
    from safetensors.torch import load_file
    ckpt = load_file(os.path.join(model_path, "model.safetensors"))
model.load_state_dict(ckpt, strict=False)
print("Weights loaded.")
