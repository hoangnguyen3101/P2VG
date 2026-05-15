from typing import Any, List, Optional, Tuple, Union, cast

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         Gemma3TextConfig, Gemma3TextModel, Gemma3ForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from .lamed_arch import LamedMetaModel, LamedMetaForCausalLM


class LamedGemma3Config(Gemma3TextConfig):
    model_type = "lamed_gemma3"


class LamedGemma3Model(LamedMetaModel, Gemma3TextModel):
    config_class = LamedGemma3Config
    def __init__(self, config: Gemma3TextConfig):
        super(LamedGemma3Model, self).__init__(config)


class LamedGemma3ForCausalLM(LamedMetaForCausalLM, Gemma3ForCausalLM):
    config_class = LamedGemma3Config

    def __init__(self, config):
        super(LamedGemma3ForCausalLM, self).__init__(config)
        # Replace the standard text model with our multimodal-aware version
        self.model = LamedGemma3Model(config)
        self.vocab_size = config.vocab_size
        # NOTE: Do NOT create separate lm_head here.
        # Gemma3 has tie_word_embeddings=True, so lm_head shares weights
        # with embed_tokens. The parent class already handles this.

    def get_model(self):
        return self.model

    def _inputs_embeds_with_image_features(self, input_ids, image_features):
        inputs_embeds = self.get_model().embed_tokens(input_ids)
        return torch.cat(
            (
                inputs_embeds[:, :1, :],
                image_features,
                inputs_embeds[:, (image_features.shape[1] + 1):, :],
            ),
            dim=1,
        )

    def _extract_loss(self, outputs):
        if hasattr(outputs, "loss"):
            return outputs.loss
        return outputs[0]

    def forward(
            self,
            images: Optional[torch.FloatTensor] = None,
            images_ax: Optional[torch.FloatTensor] = None,
            images_noisy: Optional[torch.FloatTensor] = None,
            images_ax_noisy: Optional[torch.FloatTensor] = None,
            sag_noise_variance: Optional[torch.FloatTensor] = None,
            ax_noise_variance: Optional[torch.FloatTensor] = None,
            input_ids: torch.LongTensor = None,
            labels: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,

            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        original_input_ids = input_ids
        original_attention_mask = attention_mask
        original_position_ids = position_ids
        original_labels = labels

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = cast(Any, self).prepare_inputs_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                images_ax,
                images_noisy,
                images_ax_noisy,
                sag_noise_variance,
                ax_noise_variance,
            )

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        udml_aux_loss = getattr(self.get_model(), "udml_aux_loss", None)
        if udml_aux_loss is not None and labels is not None:
            if hasattr(outputs, "loss") and outputs.loss is not None:
                outputs.loss = outputs.loss + udml_aux_loss
            elif isinstance(outputs, tuple) and len(outputs) > 0:
                outputs = (outputs[0] + udml_aux_loss,) + outputs[1:]

        model = self.get_model()
        aux_weight = getattr(model.config, "udml_lm_aux_loss_weight", 1.0)
        use_lm_aux = (
            getattr(model.config, "udml_lm_aux_enable", False)
            and self.training
            and original_labels is not None
            and original_input_ids is not None
            and getattr(model, "udml_sag_image_features", None) is not None
            and getattr(model, "udml_ax_image_features", None) is not None
        )
        if use_lm_aux and aux_weight > 0:
            sag_inputs_embeds = self._inputs_embeds_with_image_features(
                original_input_ids,
                model.udml_sag_image_features,
            )
            ax_inputs_embeds = self._inputs_embeds_with_image_features(
                original_input_ids,
                model.udml_ax_image_features,
            )
            sag_outputs = super().forward(
                input_ids=None,
                attention_mask=original_attention_mask,
                position_ids=original_position_ids,
                past_key_values=None,
                inputs_embeds=sag_inputs_embeds,
                labels=original_labels,
                use_cache=False,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            ax_outputs = super().forward(
                input_ids=None,
                attention_mask=original_attention_mask,
                position_ids=original_position_ids,
                past_key_values=None,
                inputs_embeds=ax_inputs_embeds,
                labels=original_labels,
                use_cache=False,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            lm_aux_loss = (self._extract_loss(sag_outputs) + self._extract_loss(ax_outputs)) * aux_weight
            if hasattr(outputs, "loss") and outputs.loss is not None:
                outputs.loss = outputs.loss + lm_aux_loss
            elif isinstance(outputs, tuple) and len(outputs) > 0:
                outputs = (outputs[0] + lm_aux_loss,) + outputs[1:]
        return outputs


    @torch.no_grad()
    def generate(
        self,
        images: Optional[torch.Tensor] = None,
        images_ax: Optional[torch.Tensor] = None,
        inputs: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor, Any]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = cast(Any, self).prepare_inputs_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                images_ax,
                None,
                None,
                None,
                None,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        output_ids = super().generate(
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        return output_ids


    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        images_ax = kwargs.pop("images_ax", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if images_ax is not None:
            inputs['images_ax'] = images_ax
        return inputs


AutoConfig.register("lamed_gemma3", LamedGemma3Config)
AutoModelForCausalLM.register(LamedGemma3Config, LamedGemma3ForCausalLM)
