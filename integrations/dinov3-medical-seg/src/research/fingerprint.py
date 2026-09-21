"""Train-only data fingerprints and deterministic policy recommendations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import nibabel as nib
import numpy as np

from ..data.spatial import canonicalize, spacing_zyx, xyz_to_zyx


def _digest(payload: Mapping) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _bbox_extent(mask: np.ndarray) -> list[int]:
    coordinates = np.where(mask)
    if not coordinates[0].size:
        return [0, 0, 0]
    return [int(axis.max() - axis.min() + 1) for axis in coordinates]


def _normalized_bbox(mask: np.ndarray) -> list[list[float]]:
    coordinates = np.where(mask)
    if not coordinates[0].size:
        return [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    shape = np.maximum(np.asarray(mask.shape, dtype=float), 1.0)
    low = [float(axis.min() / shape[index]) for index, axis in enumerate(coordinates)]
    high = [float((axis.max() + 1) / shape[index]) for index, axis in enumerate(coordinates)]
    return [low, high]


def _patch_coverage(extent_zyx, shape_zyx, image_size: int, patch_size: int = 16) -> dict:
    """Express target extent in resized pixels and DINO patch units.

    This diagnostic is deliberately derived only from support labels.  It is
    a planning statistic, not an inference crop or label-dependent transform.
    """
    extent = np.asarray(extent_zyx, dtype=float)
    shape = np.maximum(np.asarray(shape_zyx, dtype=float), 1.0)
    inplane_pixels = extent[1:] / shape[1:] * float(image_size)
    inplane_patches = inplane_pixels / float(patch_size)
    return {
        "image_size": int(image_size),
        "patch_size": int(patch_size),
        "inplane_extent_pixels_yx": [float(value) for value in inplane_pixels],
        "inplane_extent_patches_yx": [float(value) for value in inplane_patches],
        "minimum_inplane_patches": float(np.min(inplane_patches)),
    }


def build_training_fingerprint(
    fold_root: Path,
    support_case_ids: Sequence[str],
    task_name: str,
    modality: str = "other",
) -> dict:
    """Describe only selected support images and labels, never evaluation data."""
    rows = []
    for case_id in support_case_ids:
        image_path = fold_root / "imagesTr" / (str(case_id) + ".nii.gz")
        label_path = fold_root / "labelsTr" / (str(case_id) + ".nii.gz")
        image = canonicalize(nib.load(str(image_path)))
        label = canonicalize(nib.load(str(label_path)))
        if image.shape != label.shape or not np.allclose(image.affine, label.affine, atol=1e-4, rtol=0.0):
            raise RuntimeError("Fingerprint input pair has mismatched physical grid: {}".format(case_id))
        image_data = image.get_fdata(dtype=np.float32)
        label_data = label.get_fdata(dtype=np.float32) > 0
        finite = image_data[np.isfinite(image_data)]
        if not finite.size:
            raise RuntimeError("Fingerprint image has no finite voxels: {}".format(case_id))
        rows.append(
            {
                "case_id": str(case_id),
                "shape_zyx": list(xyz_to_zyx(image_data).shape),
                "spacing_zyx": list(spacing_zyx(image)),
                "intensity_percentiles": [float(value) for value in np.percentile(finite, [1, 50, 99])],
                "foreground_voxels": int(np.count_nonzero(label_data)),
                "foreground_fraction": float(np.mean(label_data)),
                "foreground_extent_zyx": _bbox_extent(xyz_to_zyx(label_data)),
                "foreground_bbox_normalized_zyx": _normalized_bbox(xyz_to_zyx(label_data)),
            }
        )
    spacing = np.asarray([row["spacing_zyx"] for row in rows], dtype=float)
    shapes = np.asarray([row["shape_zyx"] for row in rows], dtype=float)
    foreground_fraction = np.asarray([row["foreground_fraction"] for row in rows], dtype=float)
    extents = np.asarray([row["foreground_extent_zyx"] for row in rows], dtype=float)
    boxes = np.asarray([row["foreground_bbox_normalized_zyx"] for row in rows], dtype=float)
    manifest_path = fold_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    records = manifest.get("records", {})
    median_spacing = [float(value) for value in np.median(spacing, axis=0)]
    median_shape = [float(value) for value in np.median(shapes, axis=0)]
    median_extent = [float(value) for value in np.median(extents, axis=0)]
    p90_extent = [float(value) for value in np.percentile(extents, 90, axis=0)]
    p95_extent = [float(value) for value in np.percentile(extents, 95, axis=0)]
    p90_extent_mm = [
        float(value)
        for value in np.asarray(p90_extent, dtype=float)
        * np.asarray(median_spacing, dtype=float)
    ]
    payload = {
        "schema_version": "dinov3_medical_training_fingerprint.v2",
        "task": task_name,
        "support_case_ids": [str(value) for value in support_case_ids],
        "modality": str(modality or "other").strip().lower(),
        "metadata_availability": {
            "scanner_or_site": False,
            "patient_ids": True,
            "source_orientation": any(
                bool(records.get(str(case_id), {}).get("source_orientation"))
                for case_id in support_case_ids
            ),
            "note": "The supplied TotalSegmentator NIfTI source does not expose scanner/site metadata; no external-site claim is permitted.",
        },
        "source_orientation_by_case": {
            str(case_id): records.get(str(case_id), {}).get("source_orientation")
            for case_id in support_case_ids
        },
        "cases": rows,
        "summary": {
            "median_spacing_zyx": median_spacing,
            "spacing_iqr_zyx": [float(value) for value in np.subtract(*np.percentile(spacing, [75, 25], axis=0))],
            "median_shape_zyx": median_shape,
            "median_foreground_fraction": float(np.median(foreground_fraction)),
            "median_foreground_extent_zyx": median_extent,
            "p90_foreground_extent_zyx": p90_extent,
            "p95_foreground_extent_zyx": p95_extent,
            "p90_foreground_extent_mm_zyx": p90_extent_mm,
            "support_bbox_union_normalized_zyx": [
                [float(value) for value in np.min(boxes[:, 0, :], axis=0)],
                [float(value) for value in np.max(boxes[:, 1, :], axis=0)],
            ],
            "target_patch_coverage": {
                str(image_size): _patch_coverage(median_extent, median_shape, image_size)
                for image_size in (224, 256, 320, 384)
            },
        },
    }
    payload["fingerprint_sha256"] = _digest(payload)
    return payload


def derive_policy(fingerprint: Mapping, *, gpu_memory_gb: float = 12.0) -> dict:
    """Derive a bounded config policy from training-set geometry and target scale."""
    summary = fingerprint["summary"]
    spacing = np.asarray(summary["median_spacing_zyx"], dtype=float)
    spacing_iqr = np.asarray(summary["spacing_iqr_zyx"], dtype=float)
    shape = np.asarray(summary["median_shape_zyx"], dtype=float)
    extent = np.asarray(summary["median_foreground_extent_zyx"], dtype=float)
    p90_extent = np.asarray(
        summary.get("p90_foreground_extent_zyx")
        or summary["median_foreground_extent_zyx"],
        dtype=float,
    )
    foreground_fraction = float(summary["median_foreground_fraction"])
    largest_spacing_axis = int(np.argmax(spacing))
    plane_axes = [
        axis for axis in range(3) if axis != largest_spacing_axis
    ]
    plane_scale_256 = min(
        256.0 / max(1.0, shape[axis])
        for axis in plane_axes
    )
    minimum_patches_256 = float(
        min(
            extent[axis] * plane_scale_256 / 16.0
            for axis in plane_axes
        )
    )
    anisotropy = float(
        np.max(spacing) / max(1e-6, np.min(spacing))
    )
    spacing_variability = float(np.max(spacing_iqr / np.maximum(spacing, 1e-6)))
    inplane_extent_fraction = float(
        min(
            extent[axis] / max(shape[axis], 1.0)
            for axis in plane_axes
        )
    )
    # Foreground fraction alone is not enough: a large organ may occupy under
    # 1% of a whole-body CT while still covering many ViT patches. Require
    # either direct patch under-coverage or two geometric/volume signals.
    small_target = bool(
        minimum_patches_256 < 6.0
        or (foreground_fraction < 0.01 and inplane_extent_fraction < 0.12)
    )
    tiny_target = bool(
        minimum_patches_256 < 3.0
        or (foreground_fraction < 0.002 and inplane_extent_fraction < 0.05)
    )
    # Legacy spacing-based window: keeps non-tiny targets (e.g. aorta) exactly
    # as previously confirmed. Face-plane floored at 160 so a mid-size organ
    # still spans enough ViT patches after resize.
    legacy_patch_size = [
        int(min(shape[0], max(32, min(96, round(80.0 / max(spacing[0], 0.5)))))),
        int(min(shape[1], max(160, min(320, round(256.0 / max(spacing[1], 0.5)))))),
        int(min(shape[2], max(160, min(320, round(256.0 / max(spacing[2], 0.5)))))),
    ]
    # Tiny needle-like targets (e.g. adrenal) are dominated by background in the
    # legacy window (~0.06% foreground), so the fine stage cannot learn. Drive
    # the window from the measured target bounding box instead: extent * margin,
    # clamped to a DINO-valid lower bound and to the volume shape. Data-driven,
    # so any future tiny target auto-adapts without a hand-tuned constant.
    patch_margin = 2.0
    # Lower bound keeps the window DINO-valid: Z >= 16 (one patch-stride deep)
    # and face-plane >= 24 voxels so the resize to img_size does not extreme-
    # upsample a near-empty crop. Prevents a sub-voxel target collapsing the box.
    patch_lower_bound = np.array([16.0, 24.0, 24.0], dtype=float)
    tiny_patch_size = [
        int(min(shape[axis], max(patch_lower_bound[axis], round(p90_extent[axis] * patch_margin))))
        for axis in range(3)
    ]
    patch_size = tiny_patch_size if tiny_target else legacy_patch_size
    # Patch model input still has to be a valid DINO patch grid after resize.
    image_size = 320 if small_target else 256
    if minimum_patches_256 < 2.0 and gpu_memory_gb >= 16:
        image_size = 384
    recommended_slice_axis = ("axial", "coronal", "sagittal")[largest_spacing_axis]
    policy = {
        "policy_version": "dinov3_medical_data_policy.v1",
        "input_size": [image_size, image_size],
        "slice_batch_size": 1 if image_size >= 320 or gpu_memory_gb <= 12 else 2,
        "use_2_5d": bool(anisotropy >= 2.5),
        "recommended_slice_axis": recommended_slice_axis,
        "target_spacing_xyz": None if spacing_variability < 0.15 else [float(spacing[2]), float(spacing[1]), float(spacing[0])],
        "patch": {
            "enabled": small_target,
            "size_zyx": patch_size,
            "foreground_probability": 0.75 if tiny_target else 0.6,
            "inference_sliding_window": small_target,
            "inference_overlap": 0.5,
        },
        "two_stage_candidate": tiny_target,
        "rationale": {
            "anisotropy_ratio_z_to_inplane": anisotropy,
            "spacing_relative_iqr": spacing_variability,
            "median_foreground_fraction": foreground_fraction,
            "minimum_inplane_target_patches_at_256": minimum_patches_256,
            "minimum_inplane_extent_fraction": inplane_extent_fraction,
            "p90_foreground_extent_zyx": [float(value) for value in p90_extent],
            "small_target": small_target,
            "tiny_target": tiny_target,
            "patch_window_source": "extent_times_margin" if tiny_target else "legacy_spacing",
            "patch_margin": float(patch_margin),
        },
    }
    policy["policy_sha256"] = _digest(policy)
    return policy
