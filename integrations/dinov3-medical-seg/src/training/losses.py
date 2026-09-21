"""
Loss functions for 3D medical image segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_target_and_mask(target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    valid = target >= 0
    safe_target = torch.where(valid, target, torch.zeros_like(target)).long()
    return safe_target, valid


def _masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weights = valid.to(dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


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
        safe_target, valid = _safe_target_and_mask(target)
        target_one_hot = F.one_hot(safe_target, num_classes=num_classes)
        target_one_hot = target_one_hot.movedim(-1, 1).float()

        pred_soft = F.softmax(pred, dim=1)
        valid_channels = valid.unsqueeze(1).to(dtype=pred_soft.dtype)
        pred_soft = pred_soft * valid_channels
        target_one_hot = target_one_hot * valid_channels

        # Flatten spatial dims
        pred_flat = pred_soft.reshape(pred.shape[0], num_classes, -1)
        target_flat = target_one_hot.reshape(target.shape[0], num_classes, -1)

        intersection = (pred_flat * target_flat).sum(dim=2)
        union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Exclude background class
        if num_classes > 1:
            dice = dice[:, 1:]
            # NOTE: class_weights are intentionally NOT applied to the Dice term.
            # Dice is already a region-overlap metric that is invariant to class
            # frequency; multiplying it by a large foreground weight (e.g. 20x)
            # turns the loss into 1 - w*dice, which goes negative and explodes in
            # scale as soon as dice > 1/w, destabilising small decoders. Class
            # balancing is handled by the cross-entropy term only (see DiceCELoss).

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


class ScalarLossAdapter(nn.Module):
    """Expose scalar-only losses through the trainer's structured loss API."""

    def __init__(self, loss: nn.Module, metric_name: str):
        super().__init__()
        self.loss = loss
        self.metric_name = metric_name

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        value = self.loss(pred, target)
        return {"loss": value, self.metric_name: value.detach()}


class FocalLoss(nn.Module):
    """Focal Loss for 3D segmentation — down-weights easy examples.

    FL(p_t) = -α_t (1 - p_t)^γ log(p_t)

    Critical for few-shot small-organ segmentation where > 99 % of voxels
    are background and the model would otherwise converge to "predict all
    background".

    Args:
        alpha: foreground weight (default 0.25 as in RetinaNet)
        gamma: focusing parameter (default 2.0; higher = harder examples)
        class_weights: tensor of per-class weights
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.clone().detach())
        else:
            self.class_weights = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = pred.shape[1]
        log_p = F.log_softmax(pred, dim=1)
        target, valid = _safe_target_and_mask(target)
        log_pt = log_p.gather(1, target.unsqueeze(1)).squeeze(1)
        ce = -log_pt
        if self.class_weights is not None:
            ce = ce * self.class_weights[target]

        # p_t must be the model probability, not weighted cross entropy.
        pt = torch.exp(log_pt)

        # Focal term
        focal_weight = (1 - pt) ** self.gamma

        # For binary segmentation alpha is the foreground weight. The previous
        # one-hot sum always equalled one, so changing alpha had no effect.
        if num_classes == 2:
            alpha_t = torch.where(
                target == 1,
                torch.as_tensor(self.alpha, dtype=pred.dtype, device=pred.device),
                torch.as_tensor(1.0 - self.alpha, dtype=pred.dtype, device=pred.device),
            )
        else:
            alpha_t = torch.ones_like(ce)

        loss = alpha_t * focal_weight * ce
        return _masked_mean(loss, valid)


class ZGradientConsistencyLoss(nn.Module):
    """Match foreground probability transitions to the target along depth.

    A plain total-variation penalty would also suppress legitimate organ
    boundaries. This loss instead compares the absolute Z-gradient of the
    predicted foreground probability with the binary target's Z-gradient:
    transitions are allowed where the label changes, while isolated slice
    islands and holes inside a constant-label run are penalized.
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = float(smooth)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape[2] < 2:
            return pred.sum() * 0.0
        safe_target, valid = _safe_target_and_mask(target)
        probability = F.softmax(pred, dim=1)[:, 1]
        predicted_gradient = torch.abs(probability[:, 1:] - probability[:, :-1])
        target_foreground = (safe_target == 1).to(dtype=probability.dtype)
        target_gradient = torch.abs(
            target_foreground[:, 1:] - target_foreground[:, :-1]
        )
        pair_valid = valid[:, 1:] & valid[:, :-1]
        error = F.smooth_l1_loss(
            predicted_gradient,
            target_gradient,
            reduction="none",
            beta=self.smooth,
        )
        return _masked_mean(error, pair_valid)


class TverskyLoss(nn.Module):
    """Tversky Loss — generalisation of Dice with asymmetry control.

    TL = (TP + smooth) / (TP + α*FP + β*FN + smooth)

    - α = β = 0.5 : standard Dice
    - α > β        : penalise FP more (precision-focused)
    - α < β        : penalise FN more (recall-focused)

    For small organs (adrenal, aorta) where missing the organ is worse
    than over-segmenting, use β > α (e.g. α=0.3, β=0.7).
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7,
                 smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = pred.shape[1]
        safe_target, valid = _safe_target_and_mask(target)
        target_one_hot = F.one_hot(safe_target, num_classes=num_classes)
        target_one_hot = target_one_hot.movedim(-1, 1).float()
        pred_soft = F.softmax(pred, dim=1)
        valid_channels = valid.unsqueeze(1).to(dtype=pred_soft.dtype)
        pred_soft = pred_soft * valid_channels
        target_one_hot = target_one_hot * valid_channels

        pred_flat = pred_soft.reshape(pred.shape[0], num_classes, -1)
        target_flat = target_one_hot.reshape(target.shape[0], num_classes, -1)

        tp = (pred_flat * target_flat).sum(dim=2)
        fp = (pred_flat * (1 - target_flat)).sum(dim=2)
        fn = ((1 - pred_flat) * target_flat).sum(dim=2)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        # Exclude background
        if num_classes > 1:
            tversky = tversky[:, 1:]

        return (1.0 - tversky).mean()


class DiceFocalLoss(nn.Module):
    """Dice + Focal with an optional target-guided Z continuity term."""

    def __init__(self, dice_weight: float = 0.5, focal_weight: float = 0.5,
                 focal_alpha: float = 0.25, focal_gamma: float = 2.0,
                 class_weights=None, z_consistency_weight: float = 0.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice = DiceLoss()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma,
                               class_weights=class_weights)
        self.z_consistency_weight = float(z_consistency_weight)
        if self.z_consistency_weight < 0.0:
            raise ValueError("z_consistency_weight must be non-negative")
        self.z_consistency = (
            ZGradientConsistencyLoss()
            if self.z_consistency_weight > 0.0
            else None
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        d_loss = self.dice(pred, target)
        f_loss = self.focal(pred, target)
        if self.z_consistency is not None:
            z_loss = self.z_consistency(pred, target)
        else:
            z_loss = d_loss * 0.0
        total = (
            self.dice_weight * d_loss
            + self.focal_weight * f_loss
            + self.z_consistency_weight * z_loss
        )
        return {
            "loss": total,
            "dice_loss": d_loss.detach(),
            "focal_loss": f_loss.detach(),
            "z_consistency_loss": z_loss.detach(),
        }


def _soft_boundary(values: torch.Tensor) -> torch.Tensor:
    """Differentiable one-voxel morphological boundary for a foreground map."""
    if values.dim() != 5:
        raise ValueError("Boundary loss expects (B, C, Z, Y, X) tensors")
    dilated = F.max_pool3d(values, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool3d(-values, kernel_size=3, stride=1, padding=1)
    return torch.clamp(dilated - eroded, 0.0, 1.0)


class DiceBoundaryLoss(nn.Module):
    """Dice-CE plus a foreground boundary Dice term for elongated structures."""

    def __init__(self, dice_weight=0.5, ce_weight=0.5, boundary_weight=0.2, class_weights=None):
        super().__init__()
        if not 0.0 <= float(boundary_weight) <= 1.0:
            raise ValueError("boundary_weight must be in [0, 1]")
        self.region = DiceCELoss(dice_weight=dice_weight, ce_weight=ce_weight, class_weights=class_weights)
        self.boundary_weight = float(boundary_weight)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        region = self.region(pred, target)
        _safe_target, valid = _safe_target_and_mask(target)
        valid = valid.unsqueeze(1).to(dtype=pred.dtype)
        foreground_probability = F.softmax(pred, dim=1)[:, 1:2] * valid
        foreground_target = (target == 1).unsqueeze(1).to(dtype=pred.dtype) * valid
        pred_boundary = _soft_boundary(foreground_probability)
        target_boundary = _soft_boundary(foreground_target)
        intersection = (pred_boundary * target_boundary).sum(dim=(1, 2, 3, 4))
        denominator = pred_boundary.sum(dim=(1, 2, 3, 4)) + target_boundary.sum(dim=(1, 2, 3, 4))
        boundary_loss = (1.0 - (2.0 * intersection + 1e-6) / (denominator + 1e-6)).mean()
        total = region["loss"] + self.boundary_weight * boundary_loss
        return {
            "loss": total,
            "dice_loss": region["dice_loss"],
            "ce_loss": region["ce_loss"],
            "boundary_loss": boundary_loss.detach(),
        }


def _soft_skeletonize(values: torch.Tensor, iterations: int) -> torch.Tensor:
    """Differentiable approximation of a 3D morphological skeleton."""
    image = values
    minimum = -F.max_pool3d(-image, kernel_size=3, stride=1, padding=1)
    opened = F.max_pool3d(minimum, kernel_size=3, stride=1, padding=1)
    skeleton = F.relu(image - opened)
    for _ in range(max(0, int(iterations) - 1)):
        image = -F.max_pool3d(-image, kernel_size=3, stride=1, padding=1)
        minimum = -F.max_pool3d(-image, kernel_size=3, stride=1, padding=1)
        opened = F.max_pool3d(minimum, kernel_size=3, stride=1, padding=1)
        delta = F.relu(image - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


class DiceFocalClDiceLoss(nn.Module):
    """Region loss plus a low-weight topology term for tubular anatomy."""

    def __init__(self, dice_weight=0.7, focal_weight=0.2, cldice_weight=0.1,
                 focal_alpha=0.75, focal_gamma=2.0, skeleton_iterations=5):
        super().__init__()
        weights = [float(dice_weight), float(focal_weight), float(cldice_weight)]
        if any(value < 0.0 for value in weights) or sum(weights) <= 0.0:
            raise ValueError("Dice/Focal/clDice weights must be non-negative with a positive sum")
        total = sum(weights)
        self.dice_weight, self.focal_weight, self.cldice_weight = [value / total for value in weights]
        self.dice = DiceLoss()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.skeleton_iterations = max(1, int(skeleton_iterations))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        d_loss = self.dice(pred, target)
        f_loss = self.focal(pred, target)
        _safe_target, valid = _safe_target_and_mask(target)
        valid = valid.unsqueeze(1).to(dtype=pred.dtype)
        probability = F.softmax(pred, dim=1)[:, 1:2] * valid
        truth = (target == 1).unsqueeze(1).to(dtype=pred.dtype) * valid
        # Topology is evaluated on a bounded half-resolution grid to avoid
        # turning skeletonization into the dominant memory/time cost.
        if min(probability.shape[-3:]) >= 4:
            probability = F.avg_pool3d(probability, kernel_size=2, stride=2)
            truth = F.max_pool3d(truth, kernel_size=2, stride=2)
        pred_skeleton = _soft_skeletonize(probability, self.skeleton_iterations)
        truth_skeleton = _soft_skeletonize(truth, self.skeleton_iterations)
        smooth = 1e-6
        tprec = (pred_skeleton * truth).sum() / (pred_skeleton.sum() + smooth)
        tsens = (truth_skeleton * probability).sum() / (truth_skeleton.sum() + smooth)
        cldice = (2.0 * tprec * tsens + smooth) / (tprec + tsens + smooth)
        cldice_loss = 1.0 - cldice
        total = self.dice_weight * d_loss + self.focal_weight * f_loss + self.cldice_weight * cldice_loss
        return {
            "loss": total,
            "dice_loss": d_loss.detach(),
            "focal_loss": f_loss.detach(),
            "cldice_loss": cldice_loss.detach(),
        }


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
        return ScalarLossAdapter(DiceLoss(class_weights=class_weights), "dice_loss")
    elif loss_type == "ce":
        return ScalarLossAdapter(
            CrossEntropyLoss(weight=class_weights if class_weights is not None else None),
            "ce_loss",
        )
    elif loss_type == "focal":
        return ScalarLossAdapter(
            FocalLoss(
                alpha=loss_cfg.get("focal_alpha", 0.25),
                gamma=loss_cfg.get("focal_gamma", 2.0),
                class_weights=class_weights,
            ),
            "focal_loss",
        )
    elif loss_type == "tversky":
        return ScalarLossAdapter(
            TverskyLoss(
                alpha=loss_cfg.get("tversky_alpha", 0.3),
                beta=loss_cfg.get("tversky_beta", 0.7),
                smooth=loss_cfg.get("smooth", 1e-6),
            ),
            "tversky_loss",
        )
    elif loss_type == "dice_focal":
        return DiceFocalLoss(
            dice_weight=loss_cfg.get("dice_weight", 0.5),
            focal_weight=loss_cfg.get("focal_weight", 0.5),
            focal_alpha=loss_cfg.get("focal_alpha", 0.25),
            focal_gamma=loss_cfg.get("focal_gamma", 2.0),
            class_weights=class_weights,
            z_consistency_weight=loss_cfg.get("z_consistency_weight", 0.0),
        )
    elif loss_type == "dice_boundary":
        return DiceBoundaryLoss(
            dice_weight=loss_cfg.get("dice_weight", 0.5),
            ce_weight=loss_cfg.get("ce_weight", 0.5),
            boundary_weight=loss_cfg.get("boundary_weight", 0.2),
            class_weights=class_weights,
        )
    elif loss_type == "dice_focal_cldice":
        return DiceFocalClDiceLoss(
            dice_weight=loss_cfg.get("dice_weight", 0.7),
            focal_weight=loss_cfg.get("focal_weight", 0.2),
            cldice_weight=loss_cfg.get("cldice_weight", 0.1),
            focal_alpha=loss_cfg.get("focal_alpha", 0.75),
            focal_gamma=loss_cfg.get("focal_gamma", 2.0),
            skeleton_iterations=loss_cfg.get("skeleton_iterations", 5),
        )
    else:
        raise ValueError(
            f"Unknown loss type: {loss_type}. "
            f"Choose from: dice_ce, dice, ce, focal, tversky, dice_focal, dice_boundary, dice_focal_cldice"
        )
