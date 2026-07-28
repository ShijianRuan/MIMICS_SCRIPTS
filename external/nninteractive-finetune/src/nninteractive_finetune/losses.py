"""Losses and trajectory metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def dice_ce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Unweighted foreground Dice plus cross entropy, averaged by sample."""
    cross_entropy = functional.cross_entropy(logits, target, reduction="none")
    cross_entropy = cross_entropy.flatten(1).mean(1)
    probability = torch.softmax(logits, dim=1)[:, 1]
    foreground = (target > 0).float()
    axes = tuple(range(1, probability.ndim))
    intersection = (probability * foreground).sum(dim=axes)
    denominator = probability.sum(dim=axes) + foreground.sum(dim=axes)
    dice_loss = 1.0 - (2.0 * intersection + 1e-5) / (denominator + 1e-5)
    result = cross_entropy + dice_loss
    if reduction == "none":
        return result
    if reduction != "mean":
        raise ValueError("Unsupported reduction: {}".format(reduction))
    return result.mean()


@torch.no_grad()
def binary_dice(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.bool()
    target = target.bool()
    axes = tuple(range(1, prediction.ndim))
    intersection = (prediction & target).sum(dim=axes).float()
    denominator = prediction.sum(dim=axes).float() + target.sum(dim=axes).float()
    return torch.where(
        denominator > 0,
        2.0 * intersection / denominator,
        torch.ones_like(denominator),
    )
