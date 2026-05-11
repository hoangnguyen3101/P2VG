from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_mm_projector
from .udml_fusion import UDMLFusion


class LamedMetaModel:
    def __init__(self, config):
        super(LamedMetaModel, self).__init__(config)

        self.config = config

        if hasattr(config, "vision_tower"):
            self.vision_tower = build_vision_tower(config)
            self.mm_projector = build_mm_projector(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        return vision_tower

    def initialize_vision_modules(self, model_args):
        self.config.image_channel = model_args.image_channel
        self.config.image_size = model_args.image_size
        self.config.patch_size = model_args.patch_size

        self.config.vision_tower = model_args.vision_tower
        self.config.vision_select_layer = model_args.vision_select_layer
        self.config.vision_select_feature = model_args.vision_select_feature

        self.config.mm_projector_type = model_args.mm_projector_type
        self.config.proj_layer_type = model_args.proj_layer_type
        self.config.proj_layer_num = model_args.proj_layer_num
        self.config.proj_pooling_type = model_args.proj_pooling_type
        self.config.proj_pooling_size = model_args.proj_pooling_size
        self.config.udml_var_loss_weight = getattr(model_args, "udml_var_loss_weight", 0.1)
        self.config.udml_lm_aux_enable = getattr(model_args, "udml_lm_aux_enable", False)
        self.config.udml_lm_aux_loss_weight = getattr(model_args, "udml_lm_aux_loss_weight", 1.0)

        if self.get_vision_tower() is None:
            self.vision_tower = build_vision_tower(self.config)
            # If you have a more robust vision encoder, try freezing the vision tower by requires_grad_(False)
            self.vision_tower.requires_grad_(not model_args.freeze_vision_tower)

        self.config.mm_hidden_size = self.vision_tower.hidden_size

        # mm_projector must exist before UDML fusion is built, because MedGemma
        # adapter mode fuses in the pretrained projection input space (1152-D).
        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_mm_projector(self.config)

        # Axial vision tower initialization
        self.axt2_enable = getattr(model_args, 'axt2_enable', False)
        self.axial_only = getattr(model_args, 'axial_only', False)
        self.config.axt2_enable = self.axt2_enable
        self.config.axial_only = self.axial_only
        if self.axt2_enable:
            if getattr(self, 'vision_tower_ax', None) is None:
                self.vision_tower_ax = build_vision_tower(self.config)
                self.vision_tower_ax.requires_grad_(not model_args.freeze_vision_tower)
                
            if getattr(self, 'udml_fusion', None) is None:
                udml_hidden_size = getattr(
                    self.mm_projector,
                    "medgemma_input_dim",
                    self.config.mm_hidden_size,
                ) if getattr(self.config, "medgemma_adapter", False) and getattr(self, "mm_projector", None) is not None else self.config.mm_hidden_size
                self.udml_fusion = UDMLFusion(
                    hidden_size=udml_hidden_size,
                    var_loss_weight=self.config.udml_var_loss_weight,
                )


        if model_args.pretrain_vision_model is not None:
            vision_model_weights = torch.load(model_args.pretrain_vision_model, map_location='cpu')
            self.vision_tower.vision_tower.load_state_dict(vision_model_weights, strict=True)
            print("[ViT Sagittal] Loaded pretrained weights with strict=True")

            if self.axt2_enable:
                self.vision_tower_ax.vision_tower.load_state_dict(vision_model_weights, strict=True)
                print("[ViT Axial] Loaded pretrained weights with strict=True")

        if model_args.pretrain_mm_mlp_adapter is not None and not getattr(self.config, 'medgemma_adapter', False):
            mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            proj_weights = get_w(mm_projector_weights, 'mm_projector')
            # Filter out shape-mismatched keys (e.g., Phi3 out_dim=3072 vs Gemma3 out_dim=2560)
            model_sd = self.mm_projector.state_dict()
            compatible = {}
            skipped = []
            for k, v in proj_weights.items():
                if k in model_sd and model_sd[k].shape == v.shape:
                    compatible[k] = v
                else:
                    skipped.append(k)
            if skipped:
                print(f"[Projector] Skipped {len(skipped)} shape-mismatched keys (random init): {skipped}")
            if compatible:
                self.mm_projector.load_state_dict(compatible, strict=False)
                print(f"[Projector] Loaded {len(compatible)} compatible weights")
        elif getattr(self.config, 'medgemma_adapter', False):
            print("[Projector] MedGemma adapter mode — pretrained weights will be injected by custom_train.py")


class LamedMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def encode_all_images(self, images_sag, images_ax=None, sag_noise_variance=None, ax_noise_variance=None):
        model = self.get_model()
        model.udml_aux_loss = None
        model.udml_sag_image_features = None
        model.udml_ax_image_features = None
        # Auto-cast inputs to match model dtype (e.g. float16 for inference)
        target_dtype = next(model.get_vision_tower().parameters()).dtype
        images_sag = images_sag.to(dtype=target_dtype)
        if images_ax is not None:
            images_ax = images_ax.to(dtype=target_dtype)
        if getattr(model, "axial_only", False):
            if images_ax is None:
                raise ValueError("axial_only=True but images_ax is None")
            final_feat = model.vision_tower_ax(images_ax)
            image_features = model.mm_projector(final_feat)
            return image_features

        # 1. Encode Sagittal (Fused T1+T2)
        feat_sag = model.get_vision_tower()(images_sag) # [B, 2048, 768]
        
        if model.axt2_enable and images_ax is not None:
            # 2. Encode Axial T2
            feat_ax = model.vision_tower_ax(images_ax) # [B, 2048, 768]
            
            if getattr(model.mm_projector, "medgemma_adapter", False):
                # Pool and adapt each view into MedGemma's native visual-projection input space.
                sag_med = model.mm_projector.encode_medgemma_inputs(feat_sag) # [B, 256, 1152]
                ax_med = model.mm_projector.encode_medgemma_inputs(feat_ax) # [B, 256, 1152]
                final_feat, udml_aux_loss = model.udml_fusion(
                    sag_med,
                    ax_med,
                    sag_variance=sag_noise_variance,
                    ax_variance=ax_noise_variance,
                )
                image_features = model.mm_projector.project_medgemma_inputs(final_feat) # [B, 256, hidden_size]
                if getattr(model.config, "udml_lm_aux_enable", False) and self.training:
                    model.udml_sag_image_features = model.mm_projector.project_medgemma_inputs(sag_med)
                    model.udml_ax_image_features = model.mm_projector.project_medgemma_inputs(ax_med)
                model.udml_aux_loss = udml_aux_loss
                return image_features

            # 3. UDML-style fusion in vision-token space for non-MedGemma projectors.
            final_feat, udml_aux_loss = model.udml_fusion(feat_sag, feat_ax, sag_noise_variance, ax_noise_variance)
            if getattr(model.config, "udml_lm_aux_enable", False) and self.training:
                model.udml_sag_image_features = model.mm_projector(feat_sag)
                model.udml_ax_image_features = model.mm_projector(feat_ax)
            model.udml_aux_loss = udml_aux_loss
        else:
            final_feat = feat_sag
            
        # 4. Project once
        image_features = model.mm_projector(final_feat) # [B, 256, hidden_size]
        return image_features

    def prepare_inputs_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, images_ax=None, sag_noise_variance=None, ax_noise_variance=None,
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        else:
            image_features = self.encode_all_images(
                images,
                images_ax,
                sag_noise_variance=sag_noise_variance,
                ax_noise_variance=ax_noise_variance,
            )
            inputs_embeds = self.get_model().embed_tokens(input_ids)
            inputs_embeds = torch.cat(
                (inputs_embeds[:, :1, :], image_features, inputs_embeds[:, (image_features.shape[1] + 1):, :]), dim=1)
        return None, position_ids, attention_mask, past_key_values, inputs_embeds, labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        num_new_tokens = model_args.num_new_tokens

        self.resize_token_embeddings(len(tokenizer))

        if num_new_tokens > 0:
            input_embeddings = self.get_input_embeddings().weight.data
            output_embeddings = self.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
            else:
                # we add 3 new tokens: <im_patch>, <bx_start>, <bx_end>
                # if new tokens need input, please train input_embeddings
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                # if new tokens need predict, please train output_embeddings
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = True

        if model_args.pretrain_mm_mlp_adapter:
            mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
            if 'model.embed_tokens.weight' in mm_projector_weights:
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']

                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings.copy_(embed_tokens_weight)
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    # Shape mismatch (e.g., Phi3 [32015,3072] vs Gemma3 [262148,2560])
                    # Skip loading — backbone embeddings are already correct
                    print(f"[Vision Tokenizer] Skipping embed_tokens from pretrained adapter "
                          f"(shape {embed_tokens_weight.shape} != {input_embeddings.shape}). "
                          f"Using backbone embeddings instead.")
