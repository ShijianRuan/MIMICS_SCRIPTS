"""Evaluate a DINOv3 segmentation checkpoint with physical-space metrics."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from nibabel.processing import resample_from_to

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.spatial import (
    SPATIAL_CONVENTION,
    canonicalize,
    resample_image_to_spacing,
    restore_prediction_to_original,
    spacing_zyx,
    xyz_to_zyx,
    zyx_to_xyz,
)
from src.evaluation import binary_metrics
from src.inference import predict_array
from src.models.segmentor import DINOv33DSegmentor
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.device import get_device


def _case_stem(path: Path) -> str:
    return path.name[:-7] if path.name.endswith(".nii.gz") else path.stem


def evaluate_case(model, image_path: Path, label_path: Path, config, device, save_dir: Path | None = None) -> dict:
    original_image = nib.load(str(image_path))
    canonical_image = canonicalize(original_image)
    canonical_label = canonicalize(nib.load(str(label_path)))
    if (
        canonical_label.shape[:3] != canonical_image.shape[:3]
        or not np.allclose(canonical_label.affine, canonical_image.affine, atol=1e-4, rtol=0.0)
    ):
        canonical_label = resample_from_to(
            canonical_label,
            (canonical_image.shape, canonical_image.affine),
            order=0,
        )
    model_grid = resample_image_to_spacing(
        canonical_image,
        config.get("data", {}).get("target_spacing"),
    )
    prediction_zyx = predict_array(model, model_grid, config, device)
    prediction_grid = nib.Nifti1Image(zyx_to_xyz(prediction_zyx), model_grid.affine, header=model_grid.header)
    if (
        model_grid.shape[:3] != canonical_image.shape[:3]
        or not np.allclose(model_grid.affine, canonical_image.affine, atol=1e-4, rtol=0.0)
    ):
        prediction_grid = resample_from_to(
            prediction_grid,
            (canonical_image.shape, canonical_image.affine),
            order=0,
        )
    prediction_native_zyx = xyz_to_zyx(prediction_grid.get_fdata(dtype=np.float32)) > 0
    target_zyx = xyz_to_zyx(canonical_label.get_fdata(dtype=np.float32)) > 0
    tolerance_mm = float(config.get("evaluation", {}).get("surface_tolerance_mm", 2.0))
    result = binary_metrics(
        prediction_native_zyx,
        target_zyx,
        spacing_zyx(canonical_image),
        surface_tolerance_mm=tolerance_mm,
    )
    result.update(
        {
            "case_id": _case_stem(image_path),
            "image_path": str(image_path),
            "label_path": str(label_path),
            "spacing_zyx": list(spacing_zyx(canonical_image)),
            "native_shape_zyx": list(prediction_native_zyx.shape),
            "model_grid_shape_xyz": list(model_grid.shape[:3]),
        }
    )
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        restored = restore_prediction_to_original(
            prediction_zyx,
            model_grid=model_grid,
            original_image=original_image,
        )
        output = save_dir / "{}_prediction.nii.gz".format(_case_stem(image_path))
        nib.save(restored, str(output))
        result["prediction_path"] = str(output)
    return result


def evaluate_dataset(model, data_root: Path, split: str, config, device, save_dir: Path | None = None) -> dict:
    image_dir = data_root / "images{}".format(split)
    label_dir = data_root / "labels{}".format(split)
    labels = {_case_stem(path): path for path in label_dir.glob("*.nii*")}
    rows = []
    for image_path in sorted(image_dir.glob("*.nii*")):
        case_id = _case_stem(image_path)
        if case_id not in labels:
            raise RuntimeError("Missing label for {}".format(image_path))
        started = time.time()
        row = evaluate_case(model, image_path, labels[case_id], config, device, save_dir)
        row["elapsed_seconds"] = time.time() - started
        rows.append(row)
        print("{}: Dice={:.4f}, HD95={}".format(
            case_id,
            row["dice"],
            "NA" if row["hd95_mm"] is None else "{:.2f} mm".format(row["hd95_mm"]),
        ))
    if not rows:
        raise RuntimeError("No NIfTI cases found under {}".format(image_dir))
    def aggregate(key):
        values = [row[key] for row in rows if row[key] is not None]
        return None if not values else float(np.mean(values))
    empty_predictions = sum(1 for row in rows if int(row.get("pred_voxels", 0)) == 0)
    return {
        "spatial_convention": SPATIAL_CONVENTION,
        "n_cases": len(rows),
        "mean_dice": aggregate("dice"),
        "mean_hd95_mm": aggregate("hd95_mm"),
        "mean_assd_mm": aggregate("assd_mm"),
        "surface_tolerance_mm": float(config.get("evaluation", {}).get("surface_tolerance_mm", 2.0)),
        "mean_surface_dice": aggregate("surface_dice"),
        "mean_lesion_f1": aggregate("lesion_f1"),
        "mean_recall": aggregate("recall"),
        "mean_precision": aggregate("precision"),
        "empty_prediction_rate": float(empty_predictions / len(rows)),
        "mean_elapsed_seconds": aggregate("elapsed_seconds"),
        "per_case": rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate DINOv3 medical segmentation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="Val", choices=["Tr", "Val"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = load_config(args.config, {})
    device = torch.device(args.device) if args.device else get_device()
    model = DINOv33DSegmentor(config).to(device)
    checkpoint = load_checkpoint(model, args.checkpoint, device=device)
    model.eval()
    output_path = Path(args.output)
    summary = evaluate_dataset(
        model,
        Path(args.data_root),
        args.split,
        config,
        device,
        output_path.parent / "predictions" if args.save_predictions else None,
    )
    summary["checkpoint"] = str(args.checkpoint)
    summary["checkpoint_epoch"] = checkpoint.get("epoch")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("Summary: Dice={:.4f}, HD95={} over {} cases".format(
        summary["mean_dice"],
        "NA" if summary["mean_hd95_mm"] is None else "{:.2f} mm".format(summary["mean_hd95_mm"]),
        summary["n_cases"],
    ))
    print("Saved: {}".format(output_path))


if __name__ == "__main__":
    main()
