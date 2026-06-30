#!/usr/bin/env python3
"""
Normalize NIfTI volumes to a fixed 256x256x32 grid using MONAI transforms.

Default input:
    /storage/hoangnv/dataset_ttd

Default output:
    /storage/hoangnv/dataset_ttd_256

Example:
    /home/hoangnv/miniconda3/envs/spine/bin/python TTD/normalize_monai_256.py

    /home/hoangnv/miniconda3/envs/spine/bin/python TTD/normalize_monai_256.py \
        --input_root /storage/hoangnv/dataset_ttd \
        --output_root /storage/hoangnv/dataset_ttd_256 \
        --intensity minmax \
        --overwrite
"""

import argparse
import csv
import json
import os
import shutil
import time
from collections import Counter

import nibabel as nib
import numpy as np
from monai.transforms import Compose, Lambda, NormalizeIntensity, Resize, ScaleIntensity, SqueezeDim


DEFAULT_INPUT_ROOT = "/storage/hoangnv/dataset_ttd"
DEFAULT_OUTPUT_ROOT = "/storage/hoangnv/dataset_ttd_256"
TARGET_SHAPE = (256, 256, 32)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Resize NIfTI volumes to 256x256x32 using MONAI."
    )
    parser.add_argument("--input_root", default=DEFAULT_INPUT_ROOT,
                        help="Dataset root containing a Volume directory.")
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT,
                        help="Output dataset root.")
    parser.add_argument("--input_volume_dir", default=None,
                        help="Override input Volume directory.")
    parser.add_argument("--output_volume_dir", default=None,
                        help="Override output Volume directory.")
    parser.add_argument("--target_shape", type=int, nargs=3, default=TARGET_SHAPE,
                        metavar=("H", "W", "D"),
                        help="Target spatial shape. Default: 256 256 32.")
    parser.add_argument("--pattern", default=".nii.gz",
                        help="Only process files ending with this suffix.")
    parser.add_argument("--intensity", choices=["none", "minmax", "zscore"],
                        default="none",
                        help="Optional intensity normalization after resizing.")
    parser.add_argument("--mode", default="trilinear",
                        help="MONAI Resize interpolation mode. Use nearest for labels.")
    parser.add_argument("--copy_report", action="store_true", default=True,
                        help="Copy report/grading CSV folders when present.")
    parser.add_argument("--no_copy_report", action="store_false",
                        dest="copy_report",
                        help="Do not copy report/grading CSV folders.")
    parser.add_argument("--filter_by_report", action="store_true",
                        help="Only process subjects listed in report split CSV files.")
    parser.add_argument("--report_splits", nargs="+", default=["train", "val", "test"],
                        help="Report CSV names used by --filter_by_report. Default: train val test.")
    parser.add_argument("--require_axial", action="store_true",
                        help="Skip every file for subjects missing the axial NIfTI.")
    parser.add_argument("--axial_suffix", default="_axt2.nii.gz",
                        help="Axial filename suffix checked by --require_axial.")
    parser.add_argument("--validation_suffixes", nargs="*", default=None,
                        help="Subject-level suffixes required by report validation.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing normalized NIfTI files.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional limit for smoke testing.")
    return parser.parse_args()


def build_transform(target_shape, mode, intensity):
    transforms = [
        Lambda(lambda x: x.astype(np.float32, copy=False)),
        Lambda(lambda x: x[None]),  # MONAI spatial transforms expect channel-first.
        Resize(spatial_size=tuple(target_shape), mode=mode, align_corners=False),
    ]

    if intensity == "minmax":
        transforms.append(ScaleIntensity(minv=0.0, maxv=1.0))
    elif intensity == "zscore":
        transforms.append(NormalizeIntensity(nonzero=False, channel_wise=False))

    transforms.extend([
        SqueezeDim(dim=0),
        Lambda(lambda x: np.asarray(x, dtype=np.float32)),
    ])
    return Compose(transforms)


def resize_affine(affine, old_shape, new_shape):
    """Update voxel sizes so the resized grid covers the same index-space FOV."""
    scale = np.array(old_shape, dtype=np.float64) / np.array(new_shape, dtype=np.float64)
    new_affine = affine.copy()
    new_affine[:3, :3] = affine[:3, :3] @ np.diag(scale)
    return new_affine


def normalize_one(path, output_path, transform, target_shape):
    nii = nib.load(path)
    data = nii.get_fdata(dtype=np.float32)
    old_shape = data.shape

    if len(old_shape) != 3:
        raise ValueError("Expected 3D NIfTI, got shape %s for %s" % (old_shape, path))

    out_data = transform(data)
    out_data = np.asarray(out_data, dtype=np.float32)

    if tuple(out_data.shape) != tuple(target_shape):
        raise RuntimeError(
            "Unexpected output shape %s for %s, expected %s"
            % (out_data.shape, path, tuple(target_shape))
        )

    header = nii.header.copy()
    header.set_data_dtype(np.float32)
    header.set_data_shape(tuple(target_shape))

    out_affine = resize_affine(nii.affine, old_shape, target_shape)
    out_nii = nib.Nifti1Image(out_data, out_affine, header)
    nib.save(out_nii, output_path)

    return {
        "input_shape": list(old_shape),
        "output_shape": list(out_data.shape),
        "input_zooms": [float(x) for x in nii.header.get_zooms()[:3]],
        "output_zooms": [float(x) for x in out_nii.header.get_zooms()[:3]],
        "min": float(np.nanmin(out_data)),
        "max": float(np.nanmax(out_data)),
    }


def copy_tree_if_exists(src_root, dst_root, dirname):
    src = os.path.join(src_root, dirname)
    dst = os.path.join(dst_root, dirname)
    if not os.path.isdir(src):
        return False
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return True


def subject_id_from_name(name):
    if not name.endswith(".nii.gz"):
        return None
    stem = name[:-len(".nii.gz")]
    if "_" not in stem:
        return None
    return stem.rsplit("_", 1)[0]


def row_subject_id(row):
    return row.get("sub_id", "").strip() or row.get("case_id", "").strip()


def load_report_subjects(input_root, split_names):
    report_dir = os.path.join(input_root, "report")
    subjects = set()
    files = []
    missing_files = []

    for split in split_names:
        name = split if split.endswith(".csv") else "%s.csv" % split
        path = os.path.join(report_dir, name)
        if not os.path.exists(path):
            missing_files.append(path)
            continue

        files.append(path)
        with open(path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                subject_id = row_subject_id(row)
                if subject_id:
                    subjects.add(subject_id)

    if not files:
        raise FileNotFoundError(
            "No report split CSVs found for --filter_by_report under %s" % report_dir
        )

    return subjects, files, missing_files


def filter_files(files, input_volume_dir, report_subjects=None,
                 require_axial=False, axial_suffix="_axt2.nii.gz"):
    kept = []
    skipped_not_in_report = []
    skipped_missing_axial = []
    missing_axial_subjects = set()

    for name in files:
        subject_id = subject_id_from_name(name)
        if not subject_id:
            kept.append(name)
            continue

        should_skip = False
        if report_subjects is not None and subject_id not in report_subjects:
            skipped_not_in_report.append(name)
            should_skip = True

        if require_axial:
            axial_path = os.path.join(input_volume_dir, "%s%s" % (subject_id, axial_suffix))
            if not os.path.exists(axial_path):
                skipped_missing_axial.append(name)
                missing_axial_subjects.add(subject_id)
                should_skip = True

        if should_skip:
            continue

        kept.append(name)

    return kept, {
        "skipped_not_in_report": skipped_not_in_report,
        "skipped_missing_axial": skipped_missing_axial,
        "missing_axial_subjects": sorted(missing_axial_subjects),
    }


def filter_rows_by_subject(rows, allowed_subjects):
    if allowed_subjects is None:
        return rows
    return [
        row for row in rows
        if not row_subject_id(row) or row_subject_id(row) in allowed_subjects
    ]


def copy_csv_tree_if_exists(src_root, dst_root, dirname, allowed_subjects=None):
    src = os.path.join(src_root, dirname)
    dst = os.path.join(dst_root, dirname)
    if not os.path.isdir(src):
        return False
    if allowed_subjects is None:
        return copy_tree_if_exists(src_root, dst_root, dirname)

    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst, exist_ok=True)

    for root, dirs, files in os.walk(src):
        rel_root = os.path.relpath(root, src)
        dst_root_dir = dst if rel_root == "." else os.path.join(dst, rel_root)
        os.makedirs(dst_root_dir, exist_ok=True)
        for dirname in dirs:
            os.makedirs(os.path.join(dst_root_dir, dirname), exist_ok=True)
        for name in files:
            src_path = os.path.join(root, name)
            dst_path = os.path.join(dst_root_dir, name)

            if not name.endswith(".csv"):
                shutil.copy2(src_path, dst_path)
                continue

            with open(src_path, newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                rows = filter_rows_by_subject(list(reader), allowed_subjects)
                fieldnames = reader.fieldnames or []

            with open(dst_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

    return True


def copy_report_with_output_paths(input_root, output_root, output_volume_dir,
                                  allowed_subjects=None):
    src_dir = os.path.join(input_root, "report")
    dst_dir = os.path.join(output_root, "report")
    if not os.path.isdir(src_dir):
        return False

    os.makedirs(dst_dir, exist_ok=True)
    
    # Use relative paths for better portability
    rel_volume_dir = os.path.relpath(output_volume_dir, dst_dir)

    for name in sorted(os.listdir(src_dir)):
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)

        if not name.endswith(".csv") or not os.path.isfile(src):
            if os.path.isdir(src):
                continue
            shutil.copy2(src, dst)
            continue

        with open(src, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = filter_rows_by_subject(list(reader), allowed_subjects)
            fieldnames = reader.fieldnames or []

        for row in rows:
            if "image_path" in row:
                row["image_path"] = rel_volume_dir
            if "images_path" in row:
                row["images_path"] = rel_volume_dir
            if "missing_axt2_path" in row:
                sub_id = row.get("sub_id", "").strip()
                if sub_id:
                    row["missing_axt2_path"] = os.path.join(
                        rel_volume_dir, "%s_axt2.nii.gz" % sub_id
                    )

        with open(dst, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return True


def validate_reports(output_root, output_volume_dir, required_suffixes=None):
    result = {"counts": {}, "missing": []}
    report_dir = os.path.join(output_root, "report")
    if not os.path.isdir(report_dir):
        return result
    required_suffixes = required_suffixes or ["_fused.nii.gz", "_axt2.nii.gz"]

    for split in ["train", "val", "test"]:
        path = os.path.join(report_dir, "%s.csv" % split)
        if not os.path.exists(path):
            continue

        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))

        result["counts"][split] = len(rows)
        for row in rows:
            subject_id = row.get("sub_id", "").strip() or row.get("case_id", "").strip()
            if not subject_id:
                continue
            missing_suffixes = [
                suffix for suffix in required_suffixes
                if not os.path.exists(os.path.join(output_volume_dir, "%s%s" % (subject_id, suffix)))
            ]
            if missing_suffixes:
                result["missing"].append({
                    "split": split,
                    "subject_id": subject_id,
                    "missing_suffixes": missing_suffixes,
                })

    return result


def main():
    args = parse_args()
    input_volume_dir = args.input_volume_dir or os.path.join(args.input_root, "Volume")
    output_volume_dir = args.output_volume_dir or os.path.join(args.output_root, "Volume")
    target_shape = tuple(args.target_shape)

    if not os.path.isdir(input_volume_dir):
        raise FileNotFoundError("Missing input volume dir: %s" % input_volume_dir)

    os.makedirs(output_volume_dir, exist_ok=True)
    transform = build_transform(target_shape, args.mode, args.intensity)

    all_files = [
        name for name in sorted(os.listdir(input_volume_dir))
        if name.endswith(args.pattern)
    ]

    report_subjects = None
    report_filter = {}
    if args.filter_by_report:
        report_subjects, report_files, missing_report_files = load_report_subjects(
            args.input_root, args.report_splits
        )
        report_filter = {
            "report_subject_count": len(report_subjects),
            "report_files": report_files,
            "missing_report_files": missing_report_files,
        }

    files, file_filter = filter_files(
        all_files,
        input_volume_dir,
        report_subjects=report_subjects,
        require_axial=args.require_axial,
        axial_suffix=args.axial_suffix,
    )
    if args.limit is not None:
        files = files[:args.limit]

    selected_subjects = sorted({
        subject_id_from_name(name) for name in files
        if subject_id_from_name(name)
    })
    validation_suffixes = args.validation_suffixes
    if validation_suffixes is None and args.require_axial:
        validation_suffixes = [args.axial_suffix]

    print("Input volume dir:  %s" % input_volume_dir)
    print("Output volume dir: %s" % output_volume_dir)
    print("Target shape:      %s" % (target_shape,))
    print("Intensity:         %s" % args.intensity)
    print("Input files:       %d" % len(all_files))
    if args.filter_by_report:
        print("Report subjects:   %d" % len(report_subjects))
        print("Skip not report:   %d" % len(file_filter["skipped_not_in_report"]))
    if args.require_axial:
        print("Skip no axial:     %d" % len(file_filter["skipped_missing_axial"]))
        print("No axial subjects: %d" % len(file_filter["missing_axial_subjects"]))
    print("Subjects selected: %d" % len(selected_subjects))
    print("Files to process:  %d" % len(files))

    counts = Counter()
    details = {}
    failures = []
    start_time = time.time()

    for index, name in enumerate(files, start=1):
        src = os.path.join(input_volume_dir, name)
        dst = os.path.join(output_volume_dir, name)

        if os.path.exists(dst) and not args.overwrite:
            counts["skipped_existing"] += 1
            print("[%d/%d] skip existing %s" % (index, len(files), name))
            continue

        try:
            info = normalize_one(src, dst, transform, target_shape)
            details[name] = info
            counts["normalized"] += 1
            print("[%d/%d] %s: %s -> %s" % (
                index, len(files), name, info["input_shape"], info["output_shape"]
            ))
        except Exception as exc:
            counts["failed"] += 1
            failures.append({"file": name, "error": str(exc)})
            print("[%d/%d] ERROR %s: %s" % (index, len(files), name, exc))

    copied_sidecars = {}
    if args.copy_report:
        copied_sidecars["report"] = copy_report_with_output_paths(
            args.input_root, args.output_root, output_volume_dir,
            allowed_subjects=selected_subjects if args.filter_by_report else None
        )
        copied_sidecars["grading"] = copy_tree_if_exists(
            args.input_root, args.output_root, "grading"
        )
        # Copy dataset_summary.json if present
        summary_src = os.path.join(args.input_root, "dataset_summary.json")
        summary_dst = os.path.join(args.output_root, "dataset_summary.json")
        if os.path.isfile(summary_src):
            shutil.copy2(summary_src, summary_dst)
            copied_sidecars["dataset_summary.json"] = True

    summary = {
        "input_root": args.input_root,
        "input_volume_dir": input_volume_dir,
        "output_root": args.output_root,
        "output_volume_dir": output_volume_dir,
        "target_shape": list(target_shape),
        "intensity": args.intensity,
        "mode": args.mode,
        "filter_by_report": args.filter_by_report,
        "require_axial": args.require_axial,
        "axial_suffix": args.axial_suffix,
        "validation_suffixes": validation_suffixes,
        "input_file_count": len(all_files),
        "selected_subject_count": len(selected_subjects),
        "selected_subjects": selected_subjects,
        "report_filter": report_filter,
        "file_filter": {
            "skipped_not_in_report_count": len(file_filter["skipped_not_in_report"]),
            "skipped_missing_axial_count": len(file_filter["skipped_missing_axial"]),
            "missing_axial_subjects": file_filter["missing_axial_subjects"],
            "skipped_not_in_report": file_filter["skipped_not_in_report"],
            "skipped_missing_axial": file_filter["skipped_missing_axial"],
        },
        "counts": dict(sorted(counts.items())),
        "failures": failures,
        "copied_sidecars": copied_sidecars,
        "report_validation": validate_reports(
            args.output_root, output_volume_dir,
            required_suffixes=validation_suffixes,
        ),
        "elapsed_seconds": round(time.time() - start_time, 3),
        "details": details,
    }

    os.makedirs(args.output_root, exist_ok=True)
    summary_path = os.path.join(args.output_root, "normalize_256_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nSummary: %s" % summary_path)
    print(json.dumps({
        key: summary[key]
        for key in [
            "output_root", "target_shape", "intensity", "counts",
            "filter_by_report", "require_axial", "input_file_count",
            "selected_subject_count", "file_filter", "copied_sidecars",
            "report_validation", "elapsed_seconds"
        ]
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
