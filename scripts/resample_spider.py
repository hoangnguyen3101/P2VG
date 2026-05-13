"""Resample SPIDER fused volumes to (256, 256, 32) for P2VG ViT3D input.

Uses the EXACT same preprocessing as SPINE GitHub's SpineCapDataset.__nii_img_to_tensor:
  1. Load NIfTI → float32
  2. Transpose axes (1, 2, 0)
  3. Normalize to [0, 1]
  4. Center-crop if larger than target
  5. Zero-pad if smaller than target
  6. Permute to (D, H, W)
  7. Save as NIfTI with shape (256, 256, 32) in (H, W, D) convention
"""
import nibabel as nib
import numpy as np
import torch
import os, glob

src_dir = '/home/hoangnv/AICD_HA/SPINE/dataset/SPIDER/output_fused'
dst_dir = '/home/hoangnv/AICD_HA/SPINE/dataset/SPIDER/output_fused_256'
os.makedirs(dst_dir, exist_ok=True)

# Target shape in (H, W, D) after transpose — matching SPINE GitHub exactly
TARGET_H, TARGET_W, TARGET_D = 256, 256, 32

files = sorted(glob.glob(os.path.join(src_dir, '[0-9]*_fused.nii.gz')))
print(f'Total files: {len(files)}')

for i, f in enumerate(files):
    basename = os.path.basename(f)
    dst_path = os.path.join(dst_dir, basename)

    img = nib.load(f)
    data = img.get_fdata().astype(np.float32)
    orig_shape = data.shape

    # Step 1: Transpose (1, 2, 0) — same as SPINE GitHub line 53
    data = np.transpose(data, (1, 2, 0))

    # Step 2: Normalize to [0, 1] — same as SPINE GitHub line 61
    dmin, dmax = data.min(), data.max()
    if dmax > dmin:
        data = (data - dmin) / (dmax - dmin)

    tensor = torch.tensor(data)

    # Step 3: Center-crop — same as SPINE GitHub lines 68-79
    h, w, d = tensor.shape
    dh, dw, dd = TARGET_H, TARGET_W, TARGET_D

    h_start = max((h - dh) // 2, 0)
    h_end = min(h_start + dh, h)
    w_start = max((w - dw) // 2, 0)
    w_end = min(w_start + dw, w)
    d_start = max((d - dd) // 2, 0)
    d_end = min(d_start + dd, d)

    tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

    # Step 4: Zero-pad if smaller — same as SPINE GitHub lines 81-101
    pad_h_before = (dh - tensor.size(0)) // 2
    pad_h_after = dh - tensor.size(0) - pad_h_before

    pad_w_before = (dw - tensor.size(1)) // 2
    pad_w_after = dw - tensor.size(1) - pad_w_before

    pad_d_before = (dd - tensor.size(2)) // 2
    pad_d_after = dd - tensor.size(2) - pad_d_before

    tensor = torch.nn.functional.pad(
        tensor,
        (
            pad_d_before, pad_d_after,
            pad_w_before, pad_w_after,
            pad_h_before, pad_h_after,
        ),
        value=0,
    )

    # Step 5: Permute to (D, H, W) — same as SPINE GitHub line 103
    tensor = tensor.permute(2, 0, 1)  # (D, H, W) = (32, 256, 256)

    # Step 6: Save back as NIfTI in (H, W, D) = (256, 256, 32) convention
    result = tensor.permute(1, 2, 0).numpy()  # back to (H, W, D) for NIfTI

    new_img = nib.Nifti1Image(result, np.eye(4))
    nib.save(new_img, dst_path)

    # Also create symlink with sub- prefix
    num = basename.replace('_fused.nii.gz', '')
    symlink_path = os.path.join(dst_dir, f'sub-{num}_fused.nii.gz')
    if os.path.exists(symlink_path):
        os.remove(symlink_path)
    os.symlink(basename, symlink_path)

    if (i + 1) % 20 == 0 or i == 0:
        print(f'[{i+1}/{len(files)}] {basename}: {orig_shape} -> transpose -> crop/pad -> (256, 256, 32)')

print('Done!')
