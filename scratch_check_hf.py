from safetensors import safe_open
f = safe_open("/storage/hoangnv/triplane_kfold/gemma3_pka_dynamic_fused_gated_fold1/merged_hf/model.safetensors", framework="pt")
keys = f.keys()
has_vision = any("vision" in k for k in keys)
has_projector = any("projector" in k for k in keys)
has_fusion = any("fusion" in k for k in keys)
has_language = any("language" in k or "model.layers" in k for k in keys)

print(f"Total keys: {len(keys)}")
print(f"Has vision tower: {has_vision}")
print(f"Has projector: {has_projector}")
print(f"Has fusion module: {has_fusion}")
print(f"Has language model: {has_language}")
fusion_keys = [k for k in keys if "fusion" in k]
print(f"Fusion keys: {fusion_keys}")
