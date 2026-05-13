import os

import torch
from transformers import TrainerCallback
from loguru import logger


class EvalGenerationCallback(TrainerCallback):
    def __init__(self, trainer, tokenizer):
        self._trainer = trainer
        self._tok = tokenizer
        self._best_checkpoints = []
        self._max_best_checkpoints = 3

    def _save_trainable_params(self, args, state, filename):
        trainable_keys = {n for n, p in self._trainer.model.named_parameters() if p.requires_grad}
        full_state = self._trainer.model.state_dict()
        trainable_state_dict = {k: v.detach().cpu() for k, v in full_state.items() if k in trainable_keys}
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(args.output_dir, filename)
        tmp_path = f"{save_path}.tmp"
        torch.save(trainable_state_dict, tmp_path)
        os.replace(tmp_path, save_path)
        logger.info("Saved trainable params ({} keys) to {}", len(trainable_state_dict), save_path)
        return save_path

    def _update_best_link_and_manifest(self, args):
        self._best_checkpoints.sort(key=lambda item: item["eval_loss"])
        best = self._best_checkpoints[0]
        link_path = os.path.join(args.output_dir, "model_with_lora.bin")
        if os.path.lexists(link_path):
            os.remove(link_path)
        os.symlink(os.path.basename(best["path"]), link_path)

        manifest_path = os.path.join(args.output_dir, "best_checkpoints.txt")
        with open(manifest_path, "w", encoding="utf-8") as f:
            for rank, item in enumerate(self._best_checkpoints, start=1):
                f.write(
                    f"rank={rank}\tstep={item['step']}\teval_loss={item['eval_loss']:.8f}\tfile={os.path.basename(item['path'])}\n"
                )
        with open(os.path.join(args.output_dir, "best_eval_loss.txt"), "w", encoding="utf-8") as f:
            f.write(f"step={best['step']}\neval_loss={best['eval_loss']:.8f}\nfile={os.path.basename(best['path'])}\n")

    def on_evaluate(self, args, state, control, **kwargs):
        metrics = kwargs.get("metrics") or {}
        eval_loss = metrics.get("eval_loss")
        if eval_loss is not None:
            worst_top_loss = max((item["eval_loss"] for item in self._best_checkpoints), default=None)
            should_save = (
                len(self._best_checkpoints) < self._max_best_checkpoints
                or eval_loss < worst_top_loss
            )
            if should_save:
                filename = f"model_with_lora_step{state.global_step}_loss{eval_loss:.6f}.bin"
                logger.info(
                    "eval_loss {:.6f} entered top {}. Saving LoRA checkpoint at step {}.",
                    eval_loss,
                    self._max_best_checkpoints,
                    state.global_step,
                )
                save_path = self._save_trainable_params(args, state, filename)
                self._best_checkpoints.append(
                    {"eval_loss": eval_loss, "step": state.global_step, "path": save_path}
                )
                self._best_checkpoints.sort(key=lambda item: item["eval_loss"])

                while len(self._best_checkpoints) > self._max_best_checkpoints:
                    removed = self._best_checkpoints.pop()
                    if os.path.exists(removed["path"]):
                        os.remove(removed["path"])
                        logger.info(
                            "Removed checkpoint outside top {}: {}",
                            self._max_best_checkpoints,
                            removed["path"],
                        )
                self._update_best_link_and_manifest(args)
            else:
                logger.info(
                    "eval_loss {:.6f} did not enter top {}. Worst saved eval_loss is {:.6f}. Skipping checkpoint save.",
                    eval_loss,
                    self._max_best_checkpoints,
                    worst_top_loss,
                )

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
        logger.info("Skipping regular checkpoint save; top LoRA checkpoints are saved on eval_loss ranking.")
