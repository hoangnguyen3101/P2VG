import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import nibabel as nib
from functools import partial
from loguru import logger

from .templates import Caption_templates, CapSeg_templates


class SpineCapDataset(Dataset):
    def __init__(self, args, tokenizer, target_shape=(256, 256, 32), mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode
        self.target_shape = target_shape
        self._missing_axt2_warned = set()

        self.image_tokens = "<im_patch>" * args.proj_out_num

        csv_path = args.amos_train_cap_data_path if mode == "train" else args.amos_validation_cap_data_path
        df = pd.read_csv(csv_path)

        self.case_ids = df["case_id"].values
        if "images_path" in df.columns:
            self.base_dirs = df["images_path"].values
        elif "image_path" in df.columns:
            self.base_dirs = df["image_path"].values
        else:
            raise KeyError("Caption CSV must contain either 'images_path' or 'image_path'.")
        self.captions = df["Clinician's Notes"].values

    def __nii_img_to_tensor(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        img = nib.load(path)
        img_data = img.get_fdata().astype(np.float32)

        p05 = np.percentile(img_data, 0.5)
        p995 = np.percentile(img_data, 99.5)
        img_data = np.clip(img_data, p05, p995)
        if p995 > p05:
            img_data = (img_data - p05) / (p995 - p05)

        if self.mode == "train":
            scale = random.uniform(0.9, 1.1)
            img_data = img_data * scale
            shift = random.uniform(-0.05, 0.05)
            img_data = img_data + shift
            img_data = np.clip(img_data, 0, 1)

        vol = torch.from_numpy(img_data).permute(2, 0, 1).unsqueeze(0)
        return vol

    def __is_preprocessed_pka(self, base_dir):
        abs_base_dir = os.path.abspath(base_dir)
        return any(
            token in abs_base_dir
            for token in ["dataset_PKA", "dataset_ttd", "dataset_ttd_256"]
        )

    def __resolve_base_dir(self, base_dir):
        base_dir = str(base_dir)
        if os.path.isabs(base_dir):
            return base_dir
        return os.path.join(self.data_root, base_dir)

    def __load_sagittal_tensor(self, base_dir, sub_num):
        sagittal_modality = getattr(self.args, "sagittal_modality", "t1")
        candidate_paths = [
            os.path.join(base_dir, f"sub-{sub_num}_{sagittal_modality}.nii.gz"),
            os.path.join(base_dir, f"{sub_num}_{sagittal_modality}.nii.gz"),
        ]
        for sagittal_path in candidate_paths:
            if os.path.exists(sagittal_path):
                return self.__nii_img_to_tensor(sagittal_path)

        raise FileNotFoundError(
            f"Missing sagittal file ({sagittal_modality}) for sub-{sub_num} in {base_dir}. "
            f"Tried: {candidate_paths}"
        )

    def __load_axt2_tensor(self, base_dir, sub_num, case_id):
        ax_path = os.path.join(base_dir, f"sub-{sub_num}_axt2.nii.gz")
        if os.path.exists(ax_path):
            return self.__nii_img_to_tensor(ax_path)

        if self.__is_preprocessed_pka(base_dir):
            raise FileNotFoundError(
                f"Missing preprocessed axt2 file for sub-{sub_num} in {base_dir}"
            )

        if case_id not in self._missing_axt2_warned:
            logger.warning("Missing axt2 for {}, using zero volume fallback.", case_id)
            self._missing_axt2_warned.add(case_id)

        dh, dw, dd = self.target_shape
        return torch.zeros((1, dd, dh, dw), dtype=torch.float32)

    def __sample_udml_variance(self):
        if self.mode != "train" or not getattr(self.args, "udml_noise_enable", False):
            return 1.0
        if random.random() > getattr(self.args, "udml_noise_prob", 0.5):
            return 1.0
        low = int(getattr(self.args, "udml_noise_min", 2))
        high = int(getattr(self.args, "udml_noise_max", 12))
        return float(random.randint(low, high))

    def __apply_udml_noise(self, image, variance):
        if variance <= 1.0:
            return image
        noise_std = float(variance) * float(getattr(self.args, "udml_noise_std_scale", 0.02))
        noisy = image + torch.randn_like(image) * noise_std
        return torch.clamp(noisy, 0.0, 1.0)

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        max_attempts = 10
        current_idx = idx
        for _ in range(max_attempts):
            try:
                case_id = self.case_ids[current_idx]
                sub_num = case_id.split("_")[-1]

                base_dir = self.__resolve_base_dir(self.base_dirs[current_idx])
                if getattr(self.args, "axt2_enable", False):
                    image_ax = self.__load_axt2_tensor(base_dir, sub_num, case_id)
                else:
                    image_ax = torch.zeros((1, 32, 256, 256), dtype=torch.float32)
                if getattr(self.args, "axial_only", False):
                    image_sag = image_ax
                else:
                    image_sag = self.__load_sagittal_tensor(base_dir, sub_num)

                sag_noise_variance = self.__sample_udml_variance()
                ax_noise_variance = self.__sample_udml_variance()
                image_sag = self.__apply_udml_noise(image_sag, sag_noise_variance)
                if getattr(self.args, "axt2_enable", False):
                    image_ax = self.__apply_udml_noise(image_ax, ax_noise_variance)

                answer = self.captions[current_idx]
                prompt_question = random.choice(Caption_templates)
                question = self.image_tokens + prompt_question

                text_tensor = self.tokenizer(
                    question + " " + answer,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                return {
                    "image": image_sag,
                    "image_ax": image_ax,
                    "input_id": input_id,
                    "label": label,
                    "attention_mask": attention_mask,
                    "sag_noise_variance": torch.tensor(sag_noise_variance, dtype=torch.float32),
                    "ax_noise_variance": torch.tensor(ax_noise_variance, dtype=torch.float32),
                    "question": question,
                    "answer": answer,
                    "question_type": "Caption",
                }

            except Exception as e:
                logger.error("Error in __getitem__ at index {}: {}", current_idx, e)
                current_idx = random.randint(0, len(self.case_ids) - 1)

        raise RuntimeError(f"Failed to load any valid sample after {max_attempts} attempts.")


class SpineCapSegDataset(Dataset):
    def __init__(self, args, tokenizer, target_shape=(256, 256, 6), mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<im_patch>" * args.proj_out_num

        df = pd.read_csv(
            args.refseg_data_train_path
            if mode == "train"
            else args.refseg_data_test_path
        )

        self.images_path = df["image_path"].values
        self.captions = df["Clinician's Notes"].values
        self.seg = df["mask_path"].values

        self.nii_to_tensor = partial(self.__nii_img_to_tensor, target_shape=target_shape)

    def __nii_img_to_tensor(self, path, target_shape, is_seg=False):
        img_data = nib.load(path).get_fdata()

        if is_seg:
            img_data = img_data.astype(np.int32)
            img_data = np.where(img_data == 250, 0, 1).astype(np.uint8)
        else:
            img_data = img_data.astype(np.float32)
            if self.mode == "train":
                img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))

        img_data = np.transpose(img_data, (1, 2, 0))
        tensor = torch.tensor(img_data)

        h, w, d = tensor.shape
        dh, dw, dd = target_shape
        h_start = max((h - dh) // 2, 0)
        h_end = min(h_start + dh, h)
        w_start = max((w - dw) // 2, 0)
        w_end = min(w_start + dw, w)
        d_start = max((d - dd) // 2, 0)
        d_end = min(d_start + dd, d)

        tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

        pad_h_before = (dh - tensor.size(0)) // 2
        pad_h_after = dh - tensor.size(0) - pad_h_before
        pad_w_before = (dw - tensor.size(1)) // 2
        pad_w_after = dw - tensor.size(1) - pad_w_before
        pad_d_before = (dd - tensor.size(2)) // 2
        pad_d_after = dd - tensor.size(2) - pad_d_before

        tensor = torch.nn.functional.pad(
            tensor,
            (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after),
            value=0,
        )

        tensor = tensor.permute(2, 0, 1)
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        return tensor[0]

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        try:
            image_path = os.path.join(self.data_root, self.images_path[idx])
            seg_path = os.path.join(self.data_root, self.seg[idx])

            image = self.nii_to_tensor(image_path, is_seg=False)
            mask = self.nii_to_tensor(seg_path, is_seg=True)

            prompt_question = random.choice(Caption_templates)
            seg_question = random.choice(CapSeg_templates)
            question = self.image_tokens + " " + prompt_question + seg_question

            answer = self.captions[idx]

            self.tokenizer.padding_side = "right"
            text_tensor = self.tokenizer(
                question + " " + answer,
                max_length=self.args.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            input_id = text_tensor["input_ids"][0]
            attention_mask = text_tensor["attention_mask"][0]

            valid_len = torch.sum(attention_mask)
            if valid_len < len(input_id):
                input_id[valid_len] = self.tokenizer.eos_token_id

            question_tensor = self.tokenizer(
                question,
                max_length=self.args.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            question_len = torch.sum(question_tensor["attention_mask"][0])

            label = input_id.clone()
            label[:question_len] = -100
            if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                label[label == self.tokenizer.pad_token_id] = -100
                if valid_len < len(label):
                    label[valid_len] = self.tokenizer.eos_token_id
            else:
                label[label == self.tokenizer.pad_token_id] = -100

            return {
                "image": image,
                "input_id": input_id,
                "label": label,
                "seg": mask,
                "attention_mask": attention_mask,
                "question": question,
                "answer": answer,
                "question_type": "refseg",
            }

        except Exception as e:
            logger.error("Error loading index {}: {}", idx, e)
            idx = random.randint(0, self.__len__() - 1)
