"""
3D segmentation metrics: Dice Similarity Coefficient, Hausdorff Distance.
"""

import torch
import torch.nn.functional as F
import numpy as np


def dice_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int,
               ignore_background: bool = True) -> dict:
    """Compute per-class and mean Dice score.

    Args:
        pred: (B, C, D, H, W) logits or (B, D, H, W) class indices
        target: (B, D, H, W) class indices

    Returns:
        dict with 'per_class': list of DSC per class, 'mean': mean DSC
    """
    if pred.dim() == 5:
        pred = pred.argmax(dim=1)

    smooth = 1e-6
    dsc_per_class = []

    start = 1 if ignore_background else 0
    for c in range(start, num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()

        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()

        dsc = (2.0 * intersection + smooth) / (union + smooth)
        dsc_per_class.append(dsc.item())

    return {
        "per_class": dsc_per_class,
        "mean": np.mean(dsc_per_class) if dsc_per_class else 0.0,
    }


def hausdorff_95(pred: torch.Tensor, target: torch.Tensor, num_classes: int,
                 spacing: tuple = (1.0, 1.0, 1.0)) -> dict:
    """Compute 95th percentile Hausdorff Distance.

    Uses scipy if available, otherwise returns placeholder.

    Args:
        pred: (B, D, H, W) class indices
        target: (B, D, H, W) class indices
        spacing: voxel spacing in mm

    Returns:
        dict with 'per_class' and 'mean' HD95
    """
    try:
        from scipy.ndimage import distance_transform_edt

        if pred.dim() == 5:
            pred = pred.argmax(dim=1)

        pred_np = pred.cpu().numpy().astype(np.int64)
        target_np = target.cpu().numpy().astype(np.int64)

        hd95_per_class = []
        start = 1  # skip background
        for c in range(start, num_classes):
            pred_c = (pred_np == c).astype(np.uint8)
            target_c = (target_np == c).astype(np.uint8)

            if pred_c.sum() == 0 or target_c.sum() == 0:
                hd95_per_class.append(float("nan"))
                continue

            # Distance transforms
            dt_pred = distance_transform_edt(1 - pred_c, sampling=spacing)
            dt_target = distance_transform_edt(1 - target_c, sampling=spacing)

            # 95th percentile of surface distances
            surface_pred = dt_target[pred_c > 0]
            surface_target = dt_pred[target_c > 0]

            hd95 = max(
                np.percentile(surface_pred, 95) if len(surface_pred) > 0 else 0,
                np.percentile(surface_target, 95) if len(surface_target) > 0 else 0,
            )
            hd95_per_class.append(hd95)

        # Filter NaN values
        valid = [h for h in hd95_per_class if not np.isnan(h)]
        return {
            "per_class": hd95_per_class,
            "mean": np.mean(valid) if valid else float("nan"),
        }

    except ImportError:
        return {"per_class": [], "mean": float("nan")}
