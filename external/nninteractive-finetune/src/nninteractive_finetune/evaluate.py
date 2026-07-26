"""Evaluation through the real nnInteractive inference session."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage

from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

from .data import load_manifest
from .runtime import write_json_atomic


def _load_unambiguous_nifti(path: str) -> nib.spatialimages.SpatialImage:
    image = nib.load(path)
    qform, qform_code = image.get_qform(coded=True)
    sform, sform_code = image.get_sform(coded=True)
    if (
        int(qform_code or 0) > 0
        and int(sform_code or 0) > 0
        and not np.allclose(qform, sform, rtol=1e-5, atol=1e-4)
    ):
        raise ValueError(
            "{} has conflicting valid qform and sform transforms.".format(path)
        )
    return image


def _canonical_pair(
    image_path: str, label_path: str, label_values: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    image_obj = nib.as_closest_canonical(_load_unambiguous_nifti(image_path))
    label_obj = nib.as_closest_canonical(_load_unambiguous_nifti(label_path))
    image = np.asarray(image_obj.dataobj)
    label = np.asarray(label_obj.dataobj)
    if image.ndim == 4 and image.shape[-1] == 1:
        image = image[..., 0]
    if label.ndim == 4 and label.shape[-1] == 1:
        label = label[..., 0]
    if image.shape != label.shape or not np.allclose(
        image_obj.affine, label_obj.affine, rtol=1e-5, atol=1e-4
    ):
        raise ValueError(
            "Evaluation image and label are not on the same canonical grid."
        )
    return (
        np.ascontiguousarray(image),
        np.ascontiguousarray(np.isin(label, label_values), dtype=bool),
    )


def _largest_error_point(mask: np.ndarray) -> tuple[int, int, int] | None:
    components, count = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    if count == 0:
        return None
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    component = components == int(np.argmax(sizes))
    distance = ndimage.distance_transform_edt(component)
    coordinate = np.unravel_index(int(np.argmax(distance)), distance.shape)
    return tuple(int(value) for value in coordinate)


def _dice(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = int(prediction.sum()) + int(target.sum())
    return (
        1.0
        if denominator == 0
        else 2.0 * int((prediction & target).sum()) / denominator
    )


def evaluate_model(
    model_dir: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
    label_values: list[int],
    fold: int | str = 0,
    checkpoint_name: str = "checkpoint_final.pth",
    clicks: int = 5,
    max_cases: int = 0,
    device: str = "auto",
) -> dict[str, Any]:
    if device == "auto":
        if torch.cuda.is_available():
            selected_device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            selected_device = torch.device("mps")
        else:
            selected_device = torch.device("cpu")
    else:
        selected_device = torch.device(device)
    cases = load_manifest(manifest_path)
    if max_cases > 0:
        cases = cases[:max_cases]
    session = nnInteractiveInferenceSession(
        device=selected_device,
        use_torch_compile=False,
        verbose=False,
        do_autozoom=True,
    )
    session.initialize_from_trained_model_folder(
        str(model_dir), use_fold=fold, checkpoint_name=checkpoint_name
    )
    results = []
    for row in cases:
        image, target = _canonical_pair(row["image"], row["label"], label_values)
        target_buffer = torch.zeros(image.shape, dtype=torch.uint8)
        started = time.time()
        session.set_image(image[None])
        session.set_target_buffer(target_buffer)
        trajectory = []
        for _ in range(int(clicks)):
            prediction = target_buffer.cpu().numpy().astype(bool)
            false_negative = target & ~prediction
            false_positive = prediction & ~target
            positive = _largest_error_point(false_negative)
            negative = _largest_error_point(false_positive)
            if positive is None and negative is None:
                trajectory.append(_dice(prediction, target))
                break
            if positive is not None:
                session.add_point_interaction(
                    positive, include_interaction=True, run_prediction=negative is None
                )
            if negative is not None:
                session.add_point_interaction(
                    negative, include_interaction=False, run_prediction=True
                )
            prediction = target_buffer.cpu().numpy().astype(bool)
            trajectory.append(_dice(prediction, target))
        results.append(
            {
                "case_id": row["case_id"],
                "dice_by_click": trajectory,
                "elapsed_seconds": time.time() - started,
            }
        )
    max_length = max((len(row["dice_by_click"]) for row in results), default=0)
    mean_trajectory = []
    for index in range(max_length):
        values = [
            row["dice_by_click"][min(index, len(row["dice_by_click"]) - 1)]
            for row in results
            if row["dice_by_click"]
        ]
        mean_trajectory.append(float(np.mean(values)) if values else 0.0)
    report = {
        "schema_version": "nninteractive_finetune_evaluation.v1",
        "model_dir": str(Path(model_dir).resolve()),
        "cases": results,
        "mean_dice_by_click": mean_trajectory,
        "trajectory_auc": float(np.mean(mean_trajectory)) if mean_trajectory else 0.0,
    }
    write_json_atomic(output_path, report)
    return report
