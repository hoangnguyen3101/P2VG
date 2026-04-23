import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
import json
import pandas as pd
import nibabel as nib
from functools import partial
import pywt

from .prompt_templates import Caption_templates, CapSeg_templates

def wavelet_fuse_3d(t1_vol, t2_vol):
    """
    Performs 3D image fusion using 2D Discrete Wavelet Transform (DWT) slice-by-slice.
    Fuses Sagittal T1 and Sagittal T2 into a single volume.
    """
    import pywt
    import torch.nn.functional as F
    
    # Ensure same shape for fusion
    if t1_vol.shape != t2_vol.shape:
        # Resize t2_vol to match t1_vol using torch (D, H, W order for convenience)
        t1_torch = torch.from_numpy(t1_vol).permute(2, 0, 1).unsqueeze(0).unsqueeze(0) # [1, 1, D, H1, W1]
        t2_torch = torch.from_numpy(t2_vol).permute(2, 0, 1).unsqueeze(0).unsqueeze(0) # [1, 1, D, H2, W2]
        
        target_d, target_h, target_w = t1_torch.shape[2:]
        t2_torch = F.interpolate(t2_torch, size=(target_d, target_h, target_w), mode='trilinear', align_corners=False)
        t2_vol = t2_torch.squeeze(0).squeeze(0).permute(1, 2, 0).numpy()

    fused_vol = np.zeros_like(t1_vol)
    
    for d in range(t1_vol.shape[2]):
        # Single level decomposition using Daubechies 1 (Haar)
        try:
            cA1, (cH1, cV1, cD1) = pywt.dwt2(t1_vol[:, :, d], 'db1')
            cA2, (cH2, cV2, cD2) = pywt.dwt2(t2_vol[:, :, d], 'db1')
            
            # Case where dwt2 might result in slightly different shapes if H, W are odd
            if cA1.shape != cA2.shape:
                import cv2
                cA2 = cv2.resize(cA2, (cA1.shape[1], cA1.shape[0]), interpolation=cv2.INTER_LINEAR)
                cH2 = cv2.resize(cH2, (cH1.shape[1], cH1.shape[0]), interpolation=cv2.INTER_LINEAR)
                cV2 = cv2.resize(cV2, (cV1.shape[1], cV1.shape[0]), interpolation=cv2.INTER_LINEAR)
                cD2 = cv2.resize(cD2, (cD1.shape[1], cD1.shape[0]), interpolation=cv2.INTER_LINEAR)

            # Fusion rules:
            # Approximation (low-freq): average
            cA_f = (cA1 + cA2) / 2
            
            # Details (high-freq): maximum absolute value (energy rule)
            cH_f = np.where(np.abs(cH1) >= np.abs(cH2), cH1, cH2)
            cV_f = np.where(np.abs(cV1) >= np.abs(cV2), cV1, cV2)
            cD_f = np.where(np.abs(cD1) >= np.abs(cD2), cD1, cD2)
            
            # Reconstruct
            fused_slice = pywt.idwt2((cA_f, (cH_f, cV_f, cD_f)), 'db1')
            
            # Handle potential 1-pixel difference after idwt2
            fs_h, fs_w = fused_slice.shape
            orig_h, orig_w = t1_vol.shape[0], t1_vol.shape[1]
            fused_vol[:orig_h, :orig_w, d] = fused_slice[:orig_h, :orig_w]
        except Exception as e:
            # Fallback to T1 if wavelet fails for a specific slice
            fused_vol[:, :, d] = t1_vol[:, :, d]
            
    return fused_vol

class SpineCapDataset(Dataset):
    def __init__(self, args, tokenizer, target_shape=(256, 256, 32), mode="train"):
        self.args = args
        self.data_root = args.data_root 
        self.tokenizer = tokenizer
        self.mode = mode
        self.target_shape = target_shape

        self.image_tokens = "<im_patch>" * args.proj_out_num

        csv_path = args.amos_train_cap_data_path if mode == "train" else args.amos_validation_cap_data_path
        df = pd.read_csv(csv_path)
        
        self.case_ids = df["case_id"].values
        self.base_dirs = df["images_path"].values 
        self.captions = df["Clinician's Notes"].values

    def __nii_img_to_tensor(self, path, target_shape):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
            
        img = nib.load(path)
        img_data = img.get_fdata().astype(np.float32)

        imin, imax = img_data.min(), img_data.max()
        if imax > imin:
            img_data = (img_data - imin) / (imax - imin)
            
        # img_data is (H, W, D) -> transpose to (D, H, W) for torch interpolate
        vol = torch.from_numpy(img_data).permute(2, 0, 1).unsqueeze(0).unsqueeze(0) # [1, 1, D, H, W]
        
        # Resize using trilinear interpolation
        # target_shape in args is (H, W, D), usually (256, 256, 32)
        # interpolate expects size=(D, H, W)
        dh, dw, dd = target_shape
        vol = torch.nn.functional.interpolate(
            vol, size=(dd, dh, dw), mode='trilinear', align_corners=False
        ) # [1, 1, dd, dh, dw]
        
        return vol.squeeze(0) # [1, dd, dh, dw]

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        max_attempts = 10
        current_idx = idx
        for _ in range(max_attempts):
            try:
                case_id = self.case_ids[current_idx]

                fused_path = self.base_dirs[current_idx]
                if not os.path.isabs(fused_path):
                    fused_path = os.path.join(self.data_root, fused_path)

                image_sag = self.__nii_img_to_tensor(fused_path, self.target_shape)

                if self.args.axt2_enable:
                    # Nếu cần thuận tiện kích hoạt ảnh axial T2 thì cấu trúc tên file phải thêm
                    # (hoặc điều chỉnh trực tiếp theo CSV/đường dẫn cụ thể của bạn).
                    # Ở workflow hiện tại tắt axt2, nên dùng zero tensor thay thế.
                    case_num = case_id.split('_')[-1]
                    ax_path = os.path.join(self.data_root, f"sub-{case_num}_axt2.nii.gz")
                    if os.path.exists(ax_path):
                        image_ax = self.__nii_img_to_tensor(ax_path, self.target_shape)
                    else:
                        image_ax = torch.zeros_like(image_sag)
                else:
                    image_ax = torch.zeros_like(image_sag)
                
                answer = self.captions[current_idx]
                prompt_question = random.choice(Caption_templates)

                # Gemma 3 chat template format
                user_content = f"{self.image_tokens}{prompt_question}"
                user_turn = f"<start_of_turn>user\n{user_content}<end_of_turn>\n"
                model_prefix = "<start_of_turn>model\n"
                model_turn = f"{model_prefix}{answer}<end_of_turn>"

                full_text = user_turn + model_turn

                text_tensor = self.tokenizer(
                    full_text,
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

                # Mask labels for user turn + model prefix (only train on model response)
                question_text = user_turn + model_prefix
                question_tensor = self.tokenizer(
                    question_text,
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
                    "question": user_turn + model_prefix,
                    "answer": answer,
                    "question_type": "Caption",
                }

            except Exception as e:
                print(f"Error in __getitem__ at index {current_idx}: {e}")
                current_idx = random.randint(0, len(self.case_ids) - 1)
        
        raise RuntimeError(f"Failed to load any valid sample after {max_attempts} attempts.")

class SpineCapSegDataset(Dataset):
    def __init__(self, args, tokenizer, target_shape=(256, 256, 6), mode="train"):
        self.args = args
        self.data_root = args.data_root
        #self.data_root = args.seg_data_path
        self.tokenizer = tokenizer
        self.mode = mode

        #self.data_list = pd.read_csv(args.seg_data_path, engine='python')

        self.image_tokens = "<im_patch>" * args.proj_out_num

        df = pd.read_csv(
            args.refseg_data_train_path 
            if mode == "train" 
            else args.refseg_data_test_path
        )

        # Convert series to numpy arrays to avoid pandas index issues
        self.images_path = df["image_path"].values
        self.captions = df["Clinician's Notes"].values
        self.seg = df["mask_path"].values

        self.nii_to_tensor = partial(self.__nii_img_to_tensor, target_shape=target_shape)

    def __nii_img_to_tensor(self, path, target_shape, is_seg=False):
        img_data = nib.load(path).get_fdata()

        if is_seg:
            # label_map = {50: 0, 100: 1, 150: 2, 200: 3, 250: 4}
            # # for k, v in label_map.items():
            # #     img_data[img_data == k] = v

            # # img_data = img_data.astype(np.int64)
            # img_data = np.vectorize(label_map.get)(img_data.astype(np.int32))
            # img_data[np.isnan(img_data)] = 250  # or your background class
            # img_data = img_data.astype(np.int64)
            img_data = img_data.astype(np.int32)
            img_data = np.where(img_data == 250, 0, 1).astype(np.uint8)
        else:
            img_data = img_data.astype(np.float32)
            if self.mode == "train":
                # Normalize only during training
                img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))
                    
        img_data = np.transpose(img_data, (1, 2, 0))    
        
        tensor = torch.tensor(img_data)

        # Pad/crop to match the target shape
        h, w, d = tensor.shape
        # Calculate cropping/padding values for height, width, and depth
        dh, dw, dd = target_shape
        h_start = max((h - dh) // 2, 0)
        h_end = min(h_start + dh, h)
        w_start = max((w - dw) // 2, 0)
        w_end = min(w_start + dw, w)
        d_start = max((d - dd) // 2, 0)
        d_end = min(d_start + dd, d)

        # Crop or pad the tensor
        tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

        pad_h_before = (dh - tensor.size(0)) // 2
        pad_h_after = dh - tensor.size(0) - pad_h_before

        pad_w_before = (dw - tensor.size(1)) // 2
        pad_w_after = dw - tensor.size(1) - pad_w_before

        pad_d_before = (dd - tensor.size(2)) // 2
        pad_d_after = dd - tensor.size(2) - pad_d_before

        tensor = torch.nn.functional.pad(
            tensor,
            (
                pad_d_before,
                pad_d_after,
                pad_w_before,
                pad_w_after,
                pad_h_before,
                pad_h_after,
            ),
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
            #print("Mask info", mask)

           
            prompt_question = random.choice(Caption_templates)
            seg_question = random.choice(CapSeg_templates)
            question = self.image_tokens +' ' +prompt_question + seg_question

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
                question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
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
            print(f"Error loading index {idx}: {e}")
            idx = random.randint(0, self.__len__() - 1)

# class MultiSegDataset(Dataset):
#     def __init__(self, args, tokenizer, mode='train'):
#         super(MultiSegDataset, self).__init__()
#         self.tokenizer = tokenizer

#         self.dataset_info = dataset_info

#         self.ds_list = []
#         # self.ds_list.append(RefSegDataset(args, tokenizer, mode=mode))
#         for dataset_code in self.dataset_info.keys():
#             self.ds_list.append(SpineCapSegDataset(args, tokenizer, tag=dataset_code, description=False, mode=mode))
#             self.ds_list.append(SpineCapSegDataset(args, tokenizer, tag=dataset_code, description=True, mode=mode))
#         self.dataset = ConcatDataset(self.ds_list)

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         return self.dataset[idx]          
                    
# class SpineCapSegDataset(Dataset):
#     def __init__(self, args, tokenizer,target_shape=(256, 256, 6), mode="train"):
#         self.args = args
#         self.data_root = args.data_root
#         self.seg_data_path=args.seg_data_path
#         self.tokenizer = tokenizer
#         self.mode = mode

#         self.image_tokens = "<im_patch>" * args.proj_out_num

#         df = pd.read_csv(
#                 args.refseg_data_train_path
#                 if mode == "train"
#                 else args.refseg_data_test_path
#             )
        
#         self.images_path = df["image_path"]
#         self.captions = df["Clinician's Notes"]
#         self.seg =df["mask_path"]

#         self.nii_to_tensor = partial(
#                 self.__nii_img_to_tensor, target_shape=target_shape
#             )

#     def __nii_img_to_tensor(self, path, target_shape, is_seg=False):
#         img_data = nib.load(path)
#         img_data = img_data.get_fdata()
        
#         if is_seg: 
#             img_data = img_data.astype(np.int8)
#             img_data = np.transpose(img_data, (1, 2, 0))
#         else:
#             img_data = img_data.astype(np.float32)

#             img_data = np.transpose(img_data, (1, 2, 0))
#             # img_data = img_data * 1000
#             # hu_min, hu_max = -1000, 200
#             #img_data = np.clip(img_data, hu_min, hu_max)

#             #img_data = (((img_data + 400) / 600)).astype(np.float32)
#             slices = []
#             # Use this part only for m3d
#             img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))

#             tensor = torch.tensor(img_data)

#             # Get the dimensions of the input tensor

#             # Extract dimensions
#             h, w, d = tensor.shape
#             # Calculate cropping/padding values for height, width, and depth
#             dh, dw, dd = target_shape
#             h_start = max((h - dh) // 2, 0)
#             h_end = min(h_start + dh, h)
#             w_start = max((w - dw) // 2, 0)
#             w_end = min(w_start + dw, w)
#             d_start = max((d - dd) // 2, 0)
#             d_end = min(d_start + dd, d)

#             # Crop or pad the tensor
#             tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

#             pad_h_before = (dh - tensor.size(0)) // 2
#             pad_h_after = dh - tensor.size(0) - pad_h_before

#             pad_w_before = (dw - tensor.size(1)) // 2
#             pad_w_after = dw - tensor.size(1) - pad_w_before

#             pad_d_before = (dd - tensor.size(2)) // 2
#             pad_d_after = dd - tensor.size(2) - pad_d_before

#             tensor = torch.nn.functional.pad(
#                 tensor,
#                 (
#                     pad_d_before,
#                     pad_d_after,
#                     pad_w_before,
#                     pad_w_after,
#                     pad_h_before,
#                     pad_h_after,
#                 ),
#                 value=0,
#             )

#             tensor = tensor.permute(2, 0, 1)
#             #print("tensor.shape")

#             tensor = tensor.unsqueeze(0).unsqueeze(0)
#             return tensor[0]


#     def __len__(self):
#         return len(self.data_list)

#     def __getitem__(self, idx):
#         max_attempts = 100
#         for _ in range(max_attempts):
#             try:
        
#                 image_path = os.path.join(self.data_root, self.images_path[idx])
#                 seg_path = os.path.join(self.seg_data_path, self.seg[idx])

#                 image = self.nii_to_tensor(image_path, is_seg=False)
#                 mask = self.nii_to_tensor(seg_path, is_seg=True)

#                 # image = self.nii_to_tensor(
#                 #         os.path.join(self.data_root, self.images_path[idx])
#                 #     )
#                 # mask = self.nii_to_tensor(
#                 #         os.path.join(self.data_root, self.images_path[idx])
#                 #     )
#                 answer = self.captions[idx]
#                 prompt_question = random.choice(Caption_templates)
#                 seg_question = random.choice(CapSeg_templates)

#                 question = self.image_tokens + prompt_question + seg_question
#                 self.tokenizer.padding_side = "right"

#                 text_tensor = self.tokenizer(
#                 question + " " + answer,
#                 max_length=self.args.max_length,
#                 truncation=True,
#                 padding="max_length",
#                 return_tensors="pt",)
                
#                 # image_path = os.path.join(self.args.data_root, data["Image"])

#                 # image_array = np.load(image_path)  # 1*32*256*256, normalized

#                 # seg_path = os.path.join(self.args.data_root, data["Mask"])
#                 # seg_array = np.load(seg_path)
#                 # seg_array = (seg_array == data["Mask_ID"]).astype(np.int8)

#                 # item = {
#                 #     "image": image,
#                 #     "seg": mask,
#                 # }


#                 # question = data["Question"]
#                 # question = self.image_tokens + ' ' + question

#                 # answer = data["Answer"]

#                 # self.tokenizer.padding_side = "right"
#                 # text_tensor = self.tokenizer(
#                 #     question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
#                 # )

#                 input_id = text_tensor["input_ids"][0]
#                 attention_mask = text_tensor["attention_mask"][0]

#                 valid_len = torch.sum(attention_mask)
#                 if valid_len < len(input_id):
#                     input_id[valid_len] = self.tokenizer.eos_token_id

#                 question_tensor = self.tokenizer(
#                     question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
#                 )
#                 question_len = torch.sum(question_tensor["attention_mask"][0])

#                 label = input_id.clone()
#                 label[:question_len] = -100
#                 if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
#                     label[label == self.tokenizer.pad_token_id] = -100
#                     if valid_len < len(label):
#                         label[valid_len] = self.tokenizer.eos_token_id
#                 else:
#                     label[label == self.tokenizer.pad_token_id] = -100

#                 ret = {
#                     'image': image,
#                     'input_id': input_id,
#                     'label': label,
#                     'seg': mask,
#                     'attention_mask': attention_mask,
#                     'question': question,
#                     'answer': answer,
#                     'question_type': "refseg",
#                 }

#                 return ret

#             except Exception as e:
#                 print(f"Error in __getitem__ at index {idx}: {e}")
#                 idx = random.randint(0, len(self.data_list) - 1)