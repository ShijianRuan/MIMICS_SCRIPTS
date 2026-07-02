"""
Loss functions for 3D medical image segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """3D Dice loss for volumetric segmentation.

    Dice = 2 * |P ∩ G| / (|P| + |G|)
    """

    def __init__(self, smooth: float = 1e-6, reduction: str = "mean", class_weights=None):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.clone().detach())
        else:
            self.class_weights = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, D, H, W) logits
            target: (B, D, H, W) class indices

        Returns:
            scalar loss
        """
        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(target.long(), num_classes=num_classes)
        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()  # (B, C, D, H, W)

        pred_soft = F.softmax(pred, dim=1)

        # Flatten spatial dims
        pred_flat = pred_soft.reshape(pred.shape[0], num_classes, -1)
        target_flat = target_one_hot.reshape(target.shape[0], num_classes, -1)

        intersection = (pred_flat * target_flat).sum(dim=2)
        union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Exclude background class
        if num_classes > 1:
            dice = dice[:, 1:]
            # Apply class weights (foreground classes only)
            if self.class_weights is not None:
                dice = dice * self.class_weights[1:] if len(self.class_weights) > 1 else dice

        loss = 1.0 - dice
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class CrossEntropyLoss(nn.Module):
    """3D Cross-entropy loss."""

    def __init__(self, weight=None, ignore_index: int = -100):
        super().__init__()
        if weight is not None:
            self.register_buffer("weight", weight.clone().detach())
        else:
            self.weight = None
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, D, H, W) logits
            target: (B, D, H, W) class indices
        """
        return F.cross_entropy(pred, target.long(), weight=self.weight, ignore_index=self.ignore_index)


class DiceCELoss(nn.Module):
    """Combined Dice + Cross-Entropy loss (standard for medical segmentation)."""

    def __init__(
        self,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        smooth: float = 1e-6,
        class_weights=None,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice = DiceLoss(smooth=smooth, class_weights=class_weights)
        self.ce = CrossEntropyLoss(
            weight=class_weights if class_weights is not None else None
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """
        Returns:
            dict with 'loss', 'dice_loss', 'ce_loss'
        """
        d_loss = self.dice(pred, target)
        c_loss = self.ce(pred, target)
        total = self.dice_weight * d_loss + self.ce_weight * c_loss
        return {"loss": total, "dice_loss": d_loss.detach(), "ce_loss": c_loss.detach()}


def get_loss(config: dict) -> nn.Module:
    """Create loss function from config."""
    loss_cfg = config.get("loss", {})
    loss_type = loss_cfg.get("type", "dice_ce")

    # Class weights from config (e.g., [1.0, 5.0] for bg:class)
    cw = loss_cfg.get("class_weights", None)
    class_weights = torch.tensor(cw, dtype=torch.float32) if cw is not None else None

    if loss_type == "dice_ce":
        return DiceCELoss(
            dice_weight=loss_cfg.get("dice_weight", 0.5),
            ce_weight=loss_cfg.get("ce_weight", 0.5),
            class_weights=class_weights,
        )
    elif loss_type == "dice":
        return DiceLoss(class_weights=class_weights)
    elif loss_type == "ce":
        return CrossEntropyLoss(
            weight=class_weights if class_weights is not None else None
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
