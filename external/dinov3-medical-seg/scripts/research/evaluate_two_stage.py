#!/usr/bin/env python3
"""Evaluate a coarse-to-fine cascade without label-derived inference crops."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from nibabel.processing import resample_from_to

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.spatial import canonicalize, resample_image_to_spacing, spacing_zyx, xyz_to_zyx, zyx_to_xyz
from src.evaluation import binary_metrics
from src.inference import (
    expand_bbox,
    predict_array,
    predict_refinement_array,
    prediction_bbox_zyx,
    prediction_center_zyx,
)
from src.models.segmentor import DINOv33DSegmentor
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.device import get_device


def _case_stem(path: Path) -> str:
    return path.name[:-7] if path.name.endswith(".nii.gz") else path.stem


def _map_center(center_zyx, source_grid, destination_grid):
    if center_zyx is None:
        return None
    source_xyz = np.array([center_zyx[2], center_zyx[1], center_zyx[0], 1.0], dtype=float)
    world = source_grid.affine @ source_xyz
    target_xyz = np.linalg.inv(destination_grid.affine) @ world
    return int(round(target_xyz[2])), int(round(target_xyz[1])), int(round(target_xyz[0]))


def _world_center(mask_zyx, grid):
    coordinates = np.where(mask_zyx > 0)
    if not coordinates[0].size:
        return None
    xyz = np.array([np.mean(coordinates[2]), np.mean(coordinates[1]), np.mean(coordinates[0]), 1.0])
    return (grid.affine @ xyz)[:3]


def _roi_target_coverage(center_zyx, target_zyx, size_zyx):
    if center_zyx is None or not np.any(target_zyx):
        return 0.0
    if len(size_zyx) != 3 or any(int(value) <= 0 for value in size_zyx):
        raise ValueError("Fine-stage ROI coverage requires three positive patch size values")
    shape = np.asarray(target_zyx.shape, dtype=int)
    size = np.asarray(size_zyx, dtype=int)
    center = np.asarray(center_zyx, dtype=int)
    start = np.maximum(0, np.minimum(center - size // 2, np.maximum(shape - size, 0)))
    end = np.minimum(shape, start + size)
    region = tuple(slice(int(a), int(b)) for a, b in zip(start, end))
    return float(np.count_nonzero(target_zyx[region]) / max(1, np.count_nonzero(target_zyx)))


def _search_roi_size(coarse_prediction_zyx, fine_patch_size, shape_zyx):
    """Fine search-ROI size = coarse blob bbox + half-window margin, clamped.

    The coarse localizer predicts a large localization-ball blob whose bounding
    box already spans the target and its surroundings; padding by half the
    training window tolerates the residual coarse centre error. Falls back to
    one training window when the coarse prediction is empty.
    """
    patch = tuple(int(value) for value in fine_patch_size)
    if len(patch) != 3 or any(value <= 0 for value in patch):
        raise ValueError("Fine search ROI requires three positive patch size values")
    bbox = prediction_bbox_zyx(coarse_prediction_zyx)
    if bbox is None:
        return patch
    margin = tuple(max(1, value // 2) for value in patch)
    z0, z1, y0, y1, x0, x1 = expand_bbox(bbox, margin, tuple(int(v) for v in shape_zyx))
    return (max(patch[0], z1 - z0), max(patch[1], y1 - y0), max(patch[2], x1 - x0))


def _load_model(config_path, checkpoint_path, device):
    config = load_config(str(config_path), {})
    model = DINOv33DSegmentor(config).to(device)
    load_checkpoint(model, str(checkpoint_path), device=device)
    model.eval()
    return model, config


def main():
    parser = argparse.ArgumentParser(description="Evaluate a coarse-to-fine DINOv3 cascade")
    parser.add_argument("--coarse-config", required=True)
    parser.add_argument("--coarse-checkpoint", required=True)
    parser.add_argument("--fine-config", required=True)
    parser.add_argument("--fine-checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="Val", choices=["Tr", "Val"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    coarse_model, coarse_config = _load_model(args.coarse_config, args.coarse_checkpoint, device)
    fine_model, fine_config = _load_model(args.fine_config, args.fine_checkpoint, device)
    root = Path(args.data_root)
    labels = {_case_stem(path): path for path in (root / "labels{}".format(args.split)).glob("*.nii*")}
    rows = []
    for image_path in sorted((root / "images{}".format(args.split)).glob("*.nii*")):
        case_id = _case_stem(image_path)
        original = nib.load(str(image_path))
        canonical = canonicalize(original)
        label = canonicalize(nib.load(str(labels[case_id])))
        coarse_grid = resample_image_to_spacing(canonical, coarse_config.get("data", {}).get("target_spacing"))
        fine_grid = resample_image_to_spacing(canonical, fine_config.get("data", {}).get("target_spacing"))
        coarse_prediction = predict_array(coarse_model, coarse_grid, coarse_config, device)
        coarse_center = prediction_center_zyx(coarse_prediction)
        fine_center = _map_center(coarse_center, coarse_grid, fine_grid)
        coarse_target = resample_from_to(label, (coarse_grid.shape, coarse_grid.affine), order=0)
        coarse_target_zyx = xyz_to_zyx(coarse_target.get_fdata(dtype=np.float32)) > 0
        predicted_world = None
        if coarse_center is not None:
            predicted_world = coarse_grid.affine @ np.array(
                [coarse_center[2], coarse_center[1], coarse_center[0], 1.0], dtype=float)
        target_world = _world_center(coarse_target_zyx, coarse_grid)
        center_error_mm = (
            float(np.linalg.norm(predicted_world[:3] - target_world))
            if predicted_world is not None and target_world is not None else None
        )
        fine_target = resample_from_to(label, (fine_grid.shape, fine_grid.affine), order=0)
        fine_target_zyx = xyz_to_zyx(fine_target.get_fdata(dtype=np.float32)) > 0
        fine_patch_size = tuple(int(value) for value in fine_config.get("data", {}).get("patch", {}).get("size_zyx", []))
        # Search ROI = coarse predicted blob bbox + half-window margin, tiled by
        # the small training window at inference. The gate coverage is measured
        # over the SAME search ROI so the metric matches the fine mechanism; a
        # single small centred window would drop offset targets under coarse
        # centre error, but the tiled search ROI does not. coarse_grid and
        # fine_grid share the canonical native grid (both target_spacing None),
        # so the coarse bbox needs no cross-grid mapping.
        search_size_zyx = _search_roi_size(coarse_prediction, fine_patch_size, fine_target_zyx.shape)
        roi_coverage = _roi_target_coverage(fine_center, fine_target_zyx, search_size_zyx)
        fine_prediction = predict_refinement_array(
            fine_model, fine_grid, fine_config, device, fine_center, search_size_zyx=search_size_zyx)
        predicted_image = nib.Nifti1Image(zyx_to_xyz(fine_prediction), fine_grid.affine, header=fine_grid.header)
        if predicted_image.shape != canonical.shape or not np.allclose(predicted_image.affine, canonical.affine, atol=1e-4, rtol=0.0):
            predicted_image = resample_from_to(predicted_image, (canonical.shape, canonical.affine), order=0)
        if label.shape != canonical.shape or not np.allclose(label.affine, canonical.affine, atol=1e-4, rtol=0.0):
            label = resample_from_to(label, (canonical.shape, canonical.affine), order=0)
        metrics = binary_metrics(
            xyz_to_zyx(predicted_image.get_fdata(dtype=np.float32)) > 0,
            xyz_to_zyx(label.get_fdata(dtype=np.float32)) > 0,
            spacing_zyx(canonical),
            surface_tolerance_mm=float(fine_config.get("evaluation", {}).get("surface_tolerance_mm", 2.0)),
        )
        metrics.update({
            "case_id": case_id,
            "coarse_detected": coarse_center is not None,
            "coarse_center_error_mm": center_error_mm,
            "roi_target_coverage": roi_coverage,
            "roi_localization_success": bool(roi_coverage >= 0.95),
        })
        rows.append(metrics)
        print("{}: coarse={}, Dice={:.4f}".format(case_id, coarse_center is not None, metrics["dice"]))
    payload = {
        "schema_version": "dinov3_medical_two_stage_evaluation.v1",
        "n_cases": len(rows),
        "coarse_detection_rate": float(np.mean([row["coarse_detected"] for row in rows])),
        "roi_localization_success_rate": float(np.mean([row["roi_localization_success"] for row in rows])),
        "mean_roi_target_coverage": float(np.mean([row["roi_target_coverage"] for row in rows])),
        "mean_coarse_center_error_mm": (
            float(np.mean([row["coarse_center_error_mm"] for row in rows if row["coarse_center_error_mm"] is not None]))
            if any(row["coarse_center_error_mm"] is not None for row in rows) else None
        ),
        "mean_dice": float(np.mean([row["dice"] for row in rows])),
        "mean_hd95_mm": float(np.mean([row["hd95_mm"] for row in rows if row["hd95_mm"] is not None])) if any(row["hd95_mm"] is not None for row in rows) else None,
        "per_case": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
