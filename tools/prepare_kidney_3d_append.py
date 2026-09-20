#!/usr/bin/env python3
"""Prepare left/right 3D kidney masks for later appending to existing MCS files.

This module deliberately does not import Mimics or modify .mcs files.  It only
does the code-side part that is safe outside Mimics:

* match ``<case>.mcs`` with ``left/<case>.nii.gz`` and ``right/<case>.nii.gz``;
* copy the two prediction masks into a per-case staging tree;
* rename them to ``kidney_left_3d.nii.gz`` and ``kidney_right_3d.nii.gz``;
* validate basic NIfTI geometry and foreground information;
* write a JSON/CSV manifest describing the exact source and target paths.

The original MCS files and prediction directories are never changed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np


LEFT_MASK_NAME = "kidney_left_3d"
RIGHT_MASK_NAME = "kidney_right_3d"
SCHEMA_VERSION = "kidney_3d_append_plan.v1"


def _case_id_from_mask(path: Path) -> str:
    name = path.name
    lower = name.lower()
    if lower.endswith(".nii.gz"):
        return name[:-7]
    if lower.endswith(".nii"):
        return name[:-4]
    raise ValueError("Unsupported mask suffix: {}".format(path))


def _case_id_from_mcs(path: Path) -> str:
    return path.stem


def _index_files(root: Path, suffixes, kind: str):
    if not root.is_dir():
        raise FileNotFoundError("{} directory does not exist: {}".format(kind, root))

    indexed = {}
    duplicates = []
    for path in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_file():
            continue
        lower = path.name.lower()
        if not any(lower.endswith(suffix) for suffix in suffixes):
            continue
        case_id = (_case_id_from_mcs(path) if kind == "mcs"
                   else _case_id_from_mask(path)).lower()
        if case_id in indexed:
            duplicates.append((case_id, str(indexed[case_id]), str(path)))
            continue
        indexed[case_id] = path
    return indexed, duplicates


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _validate_mask(path: Path):
    """Return basic NIfTI metadata without changing the source file."""
    image = nib.load(str(path))
    data = np.asarray(image.dataobj)
    if data.ndim != 3:
        raise ValueError("expected a 3D mask, got shape {}".format(tuple(data.shape)))
    if not np.isfinite(image.affine).all():
        raise ValueError("NIfTI affine contains NaN or infinity")

    nonzero = int(np.count_nonzero(data))
    labels = np.unique(data)
    labels_preview = [int(value) for value in labels[:32]]
    return {
        "shape": [int(value) for value in data.shape],
        "affine": np.asarray(image.affine, dtype=float).tolist(),
        "foreground_voxels": nonzero,
        "unique_values_preview": labels_preview,
        "unique_value_count": int(labels.size),
        "dtype": str(data.dtype),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _geometry_warning(left_meta, right_meta):
    if left_meta["shape"] != right_meta["shape"]:
        return "left/right shape mismatch"
    left_affine = np.asarray(left_meta["affine"], dtype=float)
    right_affine = np.asarray(right_meta["affine"], dtype=float)
    if not np.allclose(left_affine, right_affine, rtol=0.0, atol=1e-4):
        return "left/right affine mismatch"
    return ""


def _write_json(path: Path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def build_plan(mcs_dir: Path, left_dir: Path, right_dir: Path, output_dir: Path,
               force: bool = False):
    """Build the code-only staging result and return the manifest payload."""
    mcs_files, mcs_duplicates = _index_files(mcs_dir, (".mcs",), "mcs")
    left_files, left_duplicates = _index_files(
        left_dir, (".nii.gz", ".nii"), "left mask"
    )
    right_files, right_duplicates = _index_files(
        right_dir, (".nii.gz", ".nii"), "right mask"
    )

    if not mcs_files:
        raise RuntimeError("No .mcs files found in: {}".format(mcs_dir))
    if not left_files:
        raise RuntimeError("No left prediction masks found in: {}".format(left_dir))
    if not right_files:
        raise RuntimeError("No right prediction masks found in: {}".format(right_dir))

    common_ids = sorted(set(mcs_files) & set(left_files) & set(right_files))
    if not common_ids:
        raise RuntimeError("No common case IDs exist across MCS, left, and right inputs.")

    if output_dir.exists() and not force:
        raise FileExistsError(
            "Output directory already exists; use --force to replace it: {}".format(output_dir)
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(
        prefix=output_dir.name + ".tmp-",
        dir=str(output_dir.parent),
    ))
    rows = []
    failed = []

    try:
        for case_key in common_ids:
            mcs_path = mcs_files[case_key]
            left_path = left_files[case_key]
            right_path = right_files[case_key]
            try:
                left_meta = _validate_mask(left_path)
                right_meta = _validate_mask(right_path)
                geometry_warning = _geometry_warning(left_meta, right_meta)

                case_root = temporary_dir / case_key
                case_root.mkdir(parents=True, exist_ok=True)
                staged_mcs = case_root / mcs_path.name
                shutil.copy2(str(mcs_path), str(staged_mcs))
                case_dir = case_root / "segmentations"
                case_dir.mkdir(parents=True, exist_ok=True)
                left_target = case_dir / (LEFT_MASK_NAME + ".nii.gz")
                right_target = case_dir / (RIGHT_MASK_NAME + ".nii.gz")
                shutil.copy2(str(left_path), str(left_target))
                shutil.copy2(str(right_path), str(right_target))

                rows.append({
                    "case_id": case_key,
                    "mcs_path": str(mcs_path.resolve()),
                    "staged_mcs_path": str(staged_mcs.relative_to(temporary_dir)),
                    "left_source_path": str(left_path.resolve()),
                    "right_source_path": str(right_path.resolve()),
                    "left_target_name": LEFT_MASK_NAME,
                    "right_target_name": RIGHT_MASK_NAME,
                    "left_target_path": str(left_target.relative_to(temporary_dir)),
                    "right_target_path": str(right_target.relative_to(temporary_dir)),
                    "geometry_warning": geometry_warning,
                    "left": left_meta,
                    "right": right_meta,
                })
            except Exception as exc:
                failed.append({"case_id": case_key, "error": str(exc)})

        if failed:
            raise RuntimeError(
                "Validation failed for {} common case(s); no staging result was published. {}"
                .format(len(failed), failed[:10])
            )

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "operation": "code_only_prepare_append_masks",
            "mimics_modified": False,
            "original_mcs_modified": False,
            "result_is_embedded_mcs": False,
            "note": (
                "Each case contains an unchanged copy of the source .mcs and two "
                "renamed external NIfTI masks under segmentations/. Mimics must "
                "later import these masks to create embedded Mask objects."
            ),
            "left_mask_name": LEFT_MASK_NAME,
            "right_mask_name": RIGHT_MASK_NAME,
            "mcs_dir": str(mcs_dir.resolve()),
            "left_dir": str(left_dir.resolve()),
            "right_dir": str(right_dir.resolve()),
            "output_dir": str(output_dir.resolve()),
            "counts": {
                "mcs": len(mcs_files),
                "left_masks": len(left_files),
                "right_masks": len(right_files),
                "common_cases": len(common_ids),
                "mcs_missing_left": len(set(mcs_files) - set(left_files)),
                "mcs_missing_right": len(set(mcs_files) - set(right_files)),
                "left_missing_mcs": len(set(left_files) - set(mcs_files)),
                "right_missing_mcs": len(set(right_files) - set(mcs_files)),
            },
            "duplicate_case_ids": {
                "mcs": mcs_duplicates,
                "left": left_duplicates,
                "right": right_duplicates,
            },
            "cases": rows,
        }

        _write_json(temporary_dir / "append_manifest.json", manifest)
        with (temporary_dir / "append_manifest.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "case_id", "mcs_path", "left_source_path", "right_source_path",
                "left_target_name", "right_target_name", "geometry_warning",
                "left_foreground_voxels", "right_foreground_voxels",
            ])
            for row in rows:
                writer.writerow([
                    row["case_id"], row["mcs_path"], row["left_source_path"],
                    row["right_source_path"], row["left_target_name"],
                    row["right_target_name"], row["geometry_warning"],
                    row["left"]["foreground_voxels"], row["right"]["foreground_voxels"],
                ])

        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(str(temporary_dir), str(output_dir))
        return manifest
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Prepare left/right kidney 3D masks without modifying MCS files."
    )
    parser.add_argument("--mcs-dir", default=r"R:\label_task_ct")
    parser.add_argument("--left-dir", default=r"R:\kidney_pred_3d\left")
    parser.add_argument("--right-dir", default=r"R:\kidney_pred_3d\right")
    parser.add_argument("--output-dir", default=r"R:\label_task_ct_kidney_3d_stage")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    manifest = build_plan(
        Path(args.mcs_dir),
        Path(args.left_dir),
        Path(args.right_dir),
        Path(args.output_dir),
        force=args.force,
    )
    counts = manifest["counts"]
    print("Code-only kidney 3D staging completed.")
    print("MCS files: {}".format(counts["mcs"]))
    print("Left masks: {}".format(counts["left_masks"]))
    print("Right masks: {}".format(counts["right_masks"]))
    print("Common cases prepared: {}".format(counts["common_cases"]))
    print("MCS missing left mask: {}".format(counts["mcs_missing_left"]))
    print("MCS missing right mask: {}".format(counts["mcs_missing_right"]))
    print("Output: {}".format(Path(args.output_dir).resolve()))
    print("Manifest: {}".format(Path(args.output_dir).resolve() / "append_manifest.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())