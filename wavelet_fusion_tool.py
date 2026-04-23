import os
import numpy as np
import nibabel as nib
import pywt
import torch
import torch.nn.functional as F
import argparse

def wavelet_fuse_3d(t1_vol, t2_vol):
    """
    Performs 3D image fusion using 2D Discrete Wavelet Transform (DWT) slice-by-slice.
    Based on the P2VG project logic.
    """
    # 1. Shape normalization: Resize T2 to match T1 if different
    if t1_vol.shape != t2_vol.shape:
        print(f"[*] Resizing T2 {t2_vol.shape} to match T1 {t1_vol.shape}...")
        t1_torch = torch.from_numpy(t1_vol).permute(2, 0, 1).unsqueeze(0).unsqueeze(0) # [1, 1, D, H, W]
        t2_torch = torch.from_numpy(t2_vol).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
        
        target_d, target_h, target_w = t1_torch.shape[2:]
        t2_torch = F.interpolate(t2_torch, size=(target_d, target_h, target_w), mode='trilinear', align_corners=False)
        t2_vol = t2_torch.squeeze(0).squeeze(0).permute(1, 2, 0).numpy()

    fused_vol = np.zeros_like(t1_vol)
    depth = t1_vol.shape[2]

    print(f"[*] Fusing {depth} slices using Wavelet (db1)...")
    for d in range(depth):
        # 2D DWT Level 1 (Haar/db1)
        cA1, (cH1, cV1, cD1) = pywt.dwt2(t1_vol[:, :, d], 'db1')
        cA2, (cH2, cV2, cD2) = pywt.dwt2(t2_vol[:, :, d], 'db1')

        # Fusion rules:
        # - Low-freq (cA): Average
        cA_f = (cA1 + cA2) / 2
        
        # - High-freq (cH, cV, cD): Max Absolute (energy rule)
        cH_f = np.where(np.abs(cH1) >= np.abs(cH2), cH1, cH2)
        cV_f = np.where(np.abs(cV1) >= np.abs(cV2), cV1, cV2)
        cD_f = np.where(np.abs(cD1) >= np.abs(cD2), cD1, cD2)

        # Reconstruct (Inverse DWT)
        fused_slice = pywt.idwt2((cA_f, (cH_f, cV_f, cD_f)), 'db1')
        
        # Handle 1-pixel mismatch due to odd dimensions
        target_h, target_w = t1_vol.shape[0], t1_vol.shape[1]
        if fused_slice.shape[0] > target_h: fused_slice = fused_slice[:target_h, :]
        if fused_slice.shape[1] > target_w: fused_slice = fused_slice[:, :target_w]
        
        fused_vol[:, :, d] = fused_slice

    return fused_vol

def main():
    parser = argparse.ArgumentParser(description="P2VG Wavelet 3D Fusion Tool (SAGT1 + SAGT2)")
    parser.add_argument("--t1", type=str, required=True, help="Path to SAG T1 nii.gz")
    parser.add_argument("--t2", type=str, required=True, help="Path to SAG T2 nii.gz")
    parser.add_argument("--output", type=str, default="fused_sagittal.nii.gz", help="Output filename")
    args = parser.parse_args()

    if not os.path.exists(args.t1) or not os.path.exists(args.t2):
        print("[!] Input files not found.")
        return

    print(f"--- Starting Fusion Process ---")
    # Load images
    t1_img = nib.load(args.t1)
    t1_data = t1_img.get_fdata().astype(np.float32)
    t2_data = nib.load(args.t2).get_fdata().astype(np.float32)

    # Intensity normalization (Min-Max)
    print("[*] Normalizing intensities...")
    for vol in [t1_data, t2_data]:
        v_min, v_max = vol.min(), vol.max()
        if v_max > v_min:
            vol[:] = (vol - v_min) / (v_max - v_min)

    # Perform fusion
    fused_data = wavelet_fuse_3d(t1_data, t2_data)

    # Save output
    print(f"[*] Saving result to: {args.output}")
    fused_img = nib.Nifti1Image(fused_data, t1_img.affine, t1_img.header)
    nib.save(fused_img, args.output)
    print("--- Done! ---")

if __name__ == "__main__":
    main()
