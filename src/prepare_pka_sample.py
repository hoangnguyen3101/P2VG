import argparse
import json
import os

import nibabel as nib
import numpy as np
from nibabel.orientations import aff2axcodes
from nibabel.processing import resample_to_output
from scipy.ndimage import zoom


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reorient one PKA sample to canonical orientation and resample to a target physical spacing."
    )
    parser.add_argument(
        "--input_dir",
        default="/home/hoangnv/AICD_HA/SPINE_BASE/SPINE/dataset/MRI_Spine_SPIDER_Format_pka_fused",
    )
    parser.add_argument(
        "--output_dir",
        default="/home/hoangnv/AICD_HA/SPINE_BASE/P2VG/dataset_PKA",
    )
    parser.add_argument("--subject_id", default="sub-0001")
    parser.add_argument(
        "--spacing",
        nargs=3,
        type=float,
        default=(1.0, 1.0, 4.0),
        metavar=("SX", "SY", "SZ"),
    )
    parser.add_argument(
        "--mode",
        choices=("canonical", "preserve_axis"),
        default="canonical",
        help="canonical: save in RAS world orientation. preserve_axis: keep array axis order unchanged.",
    )
    parser.add_argument(
        "--depth_size",
        type=int,
        default=0,
        help="If > 0, normalize the last axis to this many slices after spacing resample.",
    )
    parser.add_argument(
        "--xy_size",
        type=int,
        default=0,
        help="If > 0, resize the first two axes to this size after spacing resample.",
    )
    parser.add_argument(
        "--output_suffix",
        default="",
        help="Optional suffix added before .nii.gz in output file names.",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=("fused", "axt2"),
        help="List of input modality suffixes to preprocess, e.g. fused axt2 or t1 t2 axt2.",
    )
    return parser.parse_args()


def describe_image(img):
    return {
        "shape": list(img.shape[:3]),
        "spacing": [float(x) for x in img.header.get_zooms()[:3]],
        "axcodes": list(aff2axcodes(img.affine)),
    }


def reorient_and_resample(src_path, dst_path, voxel_sizes):
    img = nib.load(src_path)
    before = describe_image(img)

    canonical_img = nib.as_closest_canonical(img)
    canonical = describe_image(canonical_img)

    resampled = resample_to_output(
        canonical_img,
        voxel_sizes=tuple(voxel_sizes),
        order=1,
        mode="constant",
        cval=0.0,
    )
    data = resampled.get_fdata(dtype=np.float32)
    out_img = nib.Nifti1Image(data, resampled.affine, resampled.header)
    out_img.set_data_dtype(np.float32)
    nib.save(out_img, dst_path)

    after = describe_image(out_img)
    return {
        "source": src_path,
        "output": dst_path,
        "before": before,
        "canonical": canonical,
        "after": after,
    }


def _rescaled_affine(affine, old_zooms, new_zooms):
    new_affine = affine.copy()
    for axis in range(3):
        col = affine[:3, axis]
        norm = float(old_zooms[axis])
        if norm > 0:
            new_affine[:3, axis] = (col / norm) * float(new_zooms[axis])
    return new_affine


def resample_preserve_axis(src_path, dst_path, voxel_sizes, depth_size=0, xy_size=0):
    img = nib.load(src_path)
    before = describe_image(img)

    data = img.get_fdata(dtype=np.float32)
    old_zooms = np.asarray(img.header.get_zooms()[:3], dtype=np.float64)
    target_zooms = np.asarray(voxel_sizes, dtype=np.float64)
    zoom_factors = old_zooms / target_zooms

    spacing_resampled = zoom(data, zoom=tuple(float(z) for z in zoom_factors), order=1)
    spacing_affine = _rescaled_affine(img.affine, old_zooms, target_zooms)
    spacing_img = nib.Nifti1Image(spacing_resampled.astype(np.float32), spacing_affine)
    spacing_img.set_data_dtype(np.float32)
    after_spacing = describe_image(spacing_img)

    final_data = spacing_resampled
    final_zooms = target_zooms.copy()
    shape_before_normalize = np.asarray(final_data.shape[:3], dtype=np.float64)
    if depth_size and depth_size > 0 and final_data.shape[2] != depth_size:
        depth_zoom = float(depth_size) / float(final_data.shape[2])
        final_data = zoom(final_data, zoom=(1.0, 1.0, depth_zoom), order=1)
    if xy_size and xy_size > 0 and (
        final_data.shape[0] != xy_size or final_data.shape[1] != xy_size
    ):
        xy_zoom_x = float(xy_size) / float(final_data.shape[0])
        xy_zoom_y = float(xy_size) / float(final_data.shape[1])
        final_data = zoom(final_data, zoom=(xy_zoom_x, xy_zoom_y, 1.0), order=1)

    shape_after_normalize = np.asarray(final_data.shape[:3], dtype=np.float64)
    final_zooms = target_zooms * (shape_before_normalize / shape_after_normalize)

    final_affine = _rescaled_affine(img.affine, old_zooms, final_zooms)
    final_img = nib.Nifti1Image(final_data.astype(np.float32), final_affine)
    final_img.set_data_dtype(np.float32)
    nib.save(final_img, dst_path)

    after = describe_image(final_img)
    return {
        "source": src_path,
        "output": dst_path,
        "before": before,
        "after_spacing_resample": after_spacing,
        "after": after,
        "depth_normalized_to": int(depth_size) if depth_size else None,
        "xy_normalized_to": int(xy_size) if xy_size else None,
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    metadata = {
        "subject_id": args.subject_id,
        "mode": args.mode,
        "target_spacing_mm": [float(x) for x in args.spacing],
        "depth_size": int(args.depth_size) if args.depth_size else None,
        "xy_size": int(args.xy_size) if args.xy_size else None,
        "files": {},
    }

    for suffix in args.modalities:
        src_path = os.path.join(args.input_dir, f"{args.subject_id}_{suffix}.nii.gz")
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Missing input file: {src_path}")

        output_name = f"{args.subject_id}_{suffix}{args.output_suffix}.nii.gz"
        dst_path = os.path.join(args.output_dir, output_name)
        if args.mode == "canonical":
            metadata["files"][suffix] = reorient_and_resample(
                src_path=src_path,
                dst_path=dst_path,
                voxel_sizes=args.spacing,
            )
        else:
            metadata["files"][suffix] = resample_preserve_axis(
                src_path=src_path,
                dst_path=dst_path,
                voxel_sizes=args.spacing,
                depth_size=args.depth_size,
                xy_size=args.xy_size,
            )

    report_name = f"{args.subject_id}{args.output_suffix}_metadata.json"
    report_path = os.path.join(args.output_dir, report_name)
    with open(report_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(json.dumps(metadata, indent=2))
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
