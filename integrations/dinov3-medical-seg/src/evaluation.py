"""Physical-space binary segmentation metrics shared by evaluation commands."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, label as connected_components
from scipy.spatial import cKDTree


def _surface(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    return mask if not mask.any() else mask ^ binary_erosion(mask, border_value=0)


def _crop_to_foreground_union(prediction: np.ndarray, target: np.ndarray):
    """Crop empty CT background without changing foreground distance metrics."""
    coordinates = np.where(np.logical_or(prediction, target))
    if not coordinates[0].size:
        return prediction, target
    starts = [max(0, int(axis.min()) - 1) for axis in coordinates]
    stops = [min(prediction.shape[index], int(axis.max()) + 2) for index, axis in enumerate(coordinates)]
    region = tuple(slice(start, stop) for start, stop in zip(starts, stops))
    return prediction[region], target[region]


def binary_metrics(prediction: np.ndarray, target: np.ndarray, spacing, *, surface_tolerance_mm: float = 2.0) -> dict:
    """Compute overlap and symmetric surface distances for a binary mask pair."""
    pred = np.asarray(prediction, dtype=bool)
    truth = np.asarray(target, dtype=bool)
    if pred.shape != truth.shape:
        raise RuntimeError("Prediction/label shapes differ: {} vs {}".format(pred.shape, truth.shape))
    intersection = int(np.logical_and(pred, truth).sum())
    pred_count = int(pred.sum())
    target_count = int(truth.sum())
    denom = pred_count + target_count
    dice = 1.0 if denom == 0 else (2.0 * intersection) / denom
    precision = 1.0 if pred_count == 0 and target_count == 0 else intersection / max(1, pred_count)
    recall = 1.0 if target_count == 0 and pred_count == 0 else intersection / max(1, target_count)
    tolerance_mm = float(surface_tolerance_mm)
    if tolerance_mm <= 0.0:
        raise ValueError("surface_tolerance_mm must be positive")
    result = {
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "pred_voxels": pred_count,
        "target_voxels": target_count,
        "hd95_mm": None,
        "assd_mm": None,
        "surface_dice": None,
        "surface_tolerance_mm": tolerance_mm,
        "lesion_f1": None,
    }
    if not pred_count or not target_count:
        return result
    pred, truth = _crop_to_foreground_union(pred, truth)
    pred_surface = _surface(pred)
    truth_surface = _surface(truth)
    scale = np.asarray(tuple(float(value) for value in spacing), dtype=np.float64)
    pred_points = np.argwhere(pred_surface).astype(np.float64) * scale
    truth_points = np.argwhere(truth_surface).astype(np.float64) * scale
    pred_to_truth = cKDTree(truth_points).query(pred_points, k=1)[0]
    truth_to_pred = cKDTree(pred_points).query(truth_points, k=1)[0]
    distances = np.concatenate([pred_to_truth, truth_to_pred])
    result["hd95_mm"] = float(np.percentile(distances, 95))
    result["assd_mm"] = float(np.mean(distances))
    matched = int(np.count_nonzero(pred_to_truth <= tolerance_mm)) + int(
        np.count_nonzero(truth_to_pred <= tolerance_mm)
    )
    result["surface_dice"] = float(matched / max(1, pred_surface.sum() + truth_surface.sum()))
    pred_components, pred_count_components = connected_components(pred)
    truth_components, truth_count_components = connected_components(truth)
    matched_truth = sum(
        bool(np.any(pred[truth_components == component]))
        for component in range(1, truth_count_components + 1)
    )
    matched_pred = sum(
        bool(np.any(truth[pred_components == component]))
        for component in range(1, pred_count_components + 1)
    )
    lesion_precision = matched_pred / max(1, pred_count_components)
    lesion_recall = matched_truth / max(1, truth_count_components)
    result["lesion_f1"] = float(
        2.0 * lesion_precision * lesion_recall / max(1e-8, lesion_precision + lesion_recall)
    )
    return result
