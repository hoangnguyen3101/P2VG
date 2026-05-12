from dataclasses import dataclass

import torch


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

        return dict(
            images=images,
            images_ax=image_ax,
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            sag_noise_variance=sag_noise_variance,
            ax_noise_variance=ax_noise_variance,
        )
