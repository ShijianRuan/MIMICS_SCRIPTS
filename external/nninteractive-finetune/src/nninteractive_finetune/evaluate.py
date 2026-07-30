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
from .prompts import InteractivePromptSampler
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


def _canonical_initial_mask(
    path: str,
    image_shape: tuple[int, ...],
    image_affine: np.ndarray,
) -> np.ndarray | None:
    if not str(path or "").strip():
        return None
    initial_obj = nib.as_closest_canonical(_load_unambiguous_nifti(path))
    initial = np.asarray(initial_obj.dataobj)
    if initial.ndim == 4 and initial.shape[-1] == 1:
        initial = initial[..., 0]
    if initial.shape != image_shape or not np.allclose(
        initial_obj.affine, image_affine, rtol=1e-5, atol=1e-4
    ):
        raise ValueError(
            "Evaluation Initial Mask is not on the canonical image grid: "
            "{}.".format(path)
        )
    result = np.ascontiguousarray(initial != 0, dtype=bool)
    return result if result.any() else None


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


def _largest_correction_event(
    false_negative: np.ndarray,
    false_positive: np.ndarray,
) -> tuple[tuple[int, int, int], bool] | None:
    """Return one deterministic correction for trajectory benchmarking."""
    candidates = []
    for mask, include in (
        (np.asarray(false_negative, dtype=bool), True),
        (np.asarray(false_positive, dtype=bool), False),
    ):
        components, count = ndimage.label(
            mask, structure=np.ones((3, 3, 3), dtype=np.uint8)
        )
        if count <= 0:
            continue
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        component_index = int(np.argmax(sizes))
        candidates.append(
            (int(sizes[component_index]), components == component_index, include)
        )
    if not candidates:
        return None
    _size, component, include = max(candidates, key=lambda item: item[0])
    point = _largest_error_point(component)
    return (point, include) if point is not None else None


def _dice(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = int(prediction.sum()) + int(target.sum())
    return (
        1.0
        if denominator == 0
        else 2.0 * int((prediction & target).sum()) / denominator
    )


def _trajectory_value(values: list[float], index: int) -> float:
    if not values:
        return 0.0
    return float(values[min(index, len(values) - 1)])


def _blend_trajectories(
    first: list[float],
    second: list[float],
    first_weight: float,
) -> list[float]:
    length = max(len(first), len(second))
    weight = max(0.0, min(1.0, float(first_weight)))
    return [
        weight * _trajectory_value(first, index)
        + (1.0 - weight) * _trajectory_value(second, index)
        for index in range(length)
    ]


def _run_click_trajectory(
    session: nnInteractiveInferenceSession,
    target: np.ndarray,
    clicks: int,
    initial_mask: np.ndarray | None = None,
) -> list[float]:
    target_buffer = torch.zeros(target.shape, dtype=torch.uint8)
    session.set_target_buffer(target_buffer)
    session.reset_interactions()
    if initial_mask is not None and np.any(initial_mask):
        session.add_initial_seg_interaction(
            np.asarray(initial_mask, dtype=np.uint8),
            run_prediction=False,
        )
    trajectory = []
    for _ in range(int(clicks)):
        prediction = target_buffer.cpu().numpy().astype(bool)
        event = _largest_correction_event(
            target & ~prediction,
            prediction & ~target,
        )
        if event is None:
            trajectory.append(_dice(prediction, target))
            break
        point, include = event
        session.add_point_interaction(
            point,
            include_interaction=include,
            run_prediction=True,
        )
        prediction = target_buffer.cpu().numpy().astype(bool)
        trajectory.append(_dice(prediction, target))
    return trajectory


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
    training_goal: str = "general",
    initial_mask_probability: float = 0.5,
    provided_initial_mask_probability: float = 0.7,
) -> dict[str, Any]:
    training_goal = str(training_goal or "general").strip().lower()
    if training_goal not in {"general", "start_empty", "refine_existing"}:
        raise ValueError(
            "training_goal must be general, start_empty, or refine_existing."
        )
    for name, value in (
        ("initial_mask_probability", initial_mask_probability),
        (
            "provided_initial_mask_probability",
            provided_initial_mask_probability,
        ),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError("{} must be in [0, 1].".format(name))
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
    for case_index, row in enumerate(cases):
        image, target = _canonical_pair(row["image"], row["label"], label_values)
        image_obj = nib.as_closest_canonical(
            _load_unambiguous_nifti(row["image"])
        )
        real_initial = _canonical_initial_mask(
            row.get("initial_mask") or "",
            tuple(int(value) for value in image.shape),
            np.asarray(image_obj.affine, dtype=np.float64),
        )
        if real_initial is not None and np.array_equal(real_initial, target):
            real_initial = None
        sampler = InteractivePromptSampler(
            seed=20260724 + case_index,
            initial_mask_probability=1.0,
            provided_initial_mask_probability=0.0,
        )
        synthetic_initial = (
            sampler.initial_prediction(
                torch.from_numpy(target[None].astype(np.int64)),
                force=True,
            )[0]
            .cpu()
            .numpy()
            .astype(bool)
        )
        started = time.time()
        session.set_image(image[None])
        empty_trajectory = (
            _run_click_trajectory(session, target, clicks)
            if training_goal in ("general", "start_empty")
            else []
        )
        synthetic_trajectory = (
            _run_click_trajectory(
                session, target, clicks, synthetic_initial
            )
            if training_goal in ("general", "refine_existing")
            else []
        )
        real_trajectory = (
            _run_click_trajectory(session, target, clicks, real_initial)
            if real_initial is not None
            and training_goal in ("general", "refine_existing")
            else []
        )
        existing_trajectory = (
            _blend_trajectories(
                real_trajectory,
                synthetic_trajectory,
                provided_initial_mask_probability,
            )
            if real_trajectory
            else synthetic_trajectory
        )
        if training_goal == "start_empty":
            trajectory = empty_trajectory
        elif training_goal == "refine_existing":
            trajectory = existing_trajectory
        else:
            trajectory = _blend_trajectories(
                existing_trajectory,
                empty_trajectory,
                initial_mask_probability,
            )
        results.append(
            {
                "case_id": row["case_id"],
                "dice_by_click": trajectory,
                "empty_mask_dice_by_click": empty_trajectory,
                "synthetic_initial_mask_dice_by_click": synthetic_trajectory,
                "real_initial_mask_dice_by_click": real_trajectory,
                "has_real_initial_mask": real_initial is not None,
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

    def mean_for(key: str) -> list[float]:
        length = max(
            (len(row.get(key) or []) for row in results),
            default=0,
        )
        return [
            float(
                np.mean(
                    [
                        _trajectory_value(row.get(key) or [], index)
                        for row in results
                        if row.get(key)
                    ]
                )
            )
            for index in range(length)
        ]

    report = {
        "schema_version": "nninteractive_finetune_evaluation.v1",
        "model_dir": str(Path(model_dir).resolve()),
        "training_goal": training_goal,
        "initial_mask_probability": float(initial_mask_probability),
        "provided_initial_mask_probability": float(
            provided_initial_mask_probability
        ),
        "real_initial_mask_cases": sum(
            bool(row.get("has_real_initial_mask")) for row in results
        ),
        "cases": results,
        "mean_dice_by_click": mean_trajectory,
        "mean_empty_mask_dice_by_click": mean_for(
            "empty_mask_dice_by_click"
        ),
        "mean_synthetic_initial_mask_dice_by_click": mean_for(
            "synthetic_initial_mask_dice_by_click"
        ),
        "mean_real_initial_mask_dice_by_click": mean_for(
            "real_initial_mask_dice_by_click"
        ),
        "trajectory_auc": float(np.mean(mean_trajectory)) if mean_trajectory else 0.0,
    }
    write_json_atomic(output_path, report)
    return report
