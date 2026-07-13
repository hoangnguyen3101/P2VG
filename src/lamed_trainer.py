import os
from typing import Optional

import torch
from transformers import Trainer
from transformers.utils import WEIGHTS_NAME, logging


logger = logging.get_logger(__name__)
TRAINING_ARGS_NAME = "training_args.bin"


class LaMedTrainer(Trainer):
    def create_optimizer(self):
        """Optionally give the vision encoder its own (lower) learning rate.

        Enabled by env VISION_LR: params whose name contains "vision_tower"
        (covers vision_tower + vision_tower_ax) go into a separate param group
        with lr=VISION_LR; everything else keeps self.args.learning_rate. When
        VISION_LR is unset, fall back to the default HF optimizer.
        """
        vision_lr = os.environ.get("VISION_LR", "").strip()
        if not vision_lr or self.optimizer is not None:
            return super().create_optimizer()

        from transformers.trainer_pt_utils import get_parameter_names
        from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

        vision_lr = float(vision_lr)
        base_lr = self.args.learning_rate
        wd = self.args.weight_decay
        model = self.model

        decay = set(get_parameter_names(model, ALL_LAYERNORM_LAYERS))
        decay = {n for n in decay if "bias" not in n}

        def grp(is_vis, is_decay):
            return [
                p for n, p in model.named_parameters()
                if p.requires_grad
                and ("vision_tower" in n) == is_vis
                and (n in decay) == is_decay
            ]

        groups = [
            {"params": grp(True, True),  "weight_decay": wd,  "lr": vision_lr},
            {"params": grp(True, False), "weight_decay": 0.0, "lr": vision_lr},
            {"params": grp(False, True), "weight_decay": wd,  "lr": base_lr},
            {"params": grp(False, False),"weight_decay": 0.0, "lr": base_lr},
        ]
        groups = [g for g in groups if len(g["params"]) > 0]

        n_vis = sum(p.numel() for g in groups for p in g["params"] if g["lr"] == vision_lr)
        logger.info(f"[VISION_LR] vision params lr={vision_lr}, other lr={base_lr} "
                    f"(vision param count={n_vis})")

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        optimizer_kwargs.pop("lr", None)
        self.optimizer = optimizer_cls(groups, **optimizer_kwargs)
        return self.optimizer

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        if state_dict is None:
            state_dict = self.model.state_dict()

        logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
        torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))

        tokenizer = getattr(self, "tokenizer", None) or getattr(self, "processing_class", None)
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
