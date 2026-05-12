import os

import torch
from transformers import TrainerCallback
from loguru import logger


class EvalGenerationCallback(TrainerCallback):
    def __init__(self, trainer, tokenizer):
        self._trainer = trainer
        self._tok = tokenizer

    def on_evaluate(self, args, state, control, **kwargs):
        logger.info("Running generation evaluation...")
        vl = self._trainer.get_eval_dataloader()

        device = next(self._trainer.model.parameters()).device
        self._trainer.model.eval()
        gt = None
        with torch.no_grad():
            for batch in vl:
                gt = batch
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
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
                    padded_prompt_ids[i, : ids.size(0)] = ids
                    padded_prompt_masks[i, : mask.size(0)] = mask

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
        logger.info("*" * 100)
        for g, p in zip(gt_texts, generated_texts):
            logger.info("-" * 100)
            logger.info("GROUNDTRUTH:\t{}", g)
            logger.info("PREDICTION:\t{}", p)

        self._trainer.model.train()

    def on_save(self, args, state, control, **kwargs):
        logger.info("EPOCH {}: SAVING.....", state.global_step)
        trainable_keys = {n for n, p in self._trainer.model.named_parameters() if p.requires_grad}
        full_state = self._trainer.model.state_dict()
        trainable_state_dict = {k: v for k, v in full_state.items() if k in trainable_keys}
        save_path = os.path.join(args.output_dir, f"model_with_lora_{state.global_step}.bin")
        torch.save(trainable_state_dict, save_path)
        logger.info("Saved trainable params ({} keys) to {}", len(trainable_state_dict), save_path)
