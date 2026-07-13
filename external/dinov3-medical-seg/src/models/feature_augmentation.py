"""Feature-space augmentation variants for controlled few-shot experiments."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


class WaveletDetailAttenuation(nn.Module):
    """Randomly attenuate in-plane Haar detail bands of ViT feature volumes.

    This acts after the frozen/PEFT encoder and before the decoder.  It avoids
    inventing label changes and is disabled during evaluation.  The operation
    is intentionally named as an experimental Haar proxy rather than claiming
    equivalence to DINO-AugSeg's complete WT-Aug method.
    """

    def __init__(self, probability: float = 0.0, maximum_attenuation: float = 0.35):
        super().__init__()
        if not 0.0 <= float(probability) <= 1.0:
            raise ValueError("feature_augmentation.probability must be in [0, 1]")
        if not 0.0 <= float(maximum_attenuation) < 1.0:
            raise ValueError("feature_augmentation.maximum_attenuation must be in [0, 1)")
        self.probability = float(probability)
        self.maximum_attenuation = float(maximum_attenuation)

    @staticmethod
    def _haar_attenuate(feature: torch.Tensor, attenuation: torch.Tensor) -> torch.Tensor:
        """Attenuate LH/HL/HH while exactly retaining the low-frequency band."""
        _, _, _, height, width = feature.shape
        padded = feature
        if height % 2:
            padded = torch.cat([padded, padded[..., -1:, :]], dim=-2)
        if width % 2:
            padded = torch.cat([padded, padded[..., :, -1:]], dim=-1)
        x00 = padded[..., 0::2, 0::2]
        x01 = padded[..., 0::2, 1::2]
        x10 = padded[..., 1::2, 0::2]
        x11 = padded[..., 1::2, 1::2]
        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (x00 - x01 + x10 - x11) * 0.5 * attenuation
        hl = (x00 + x01 - x10 - x11) * 0.5 * attenuation
        hh = (x00 - x01 - x10 + x11) * 0.5 * attenuation
        output = torch.empty_like(padded)
        output[..., 0::2, 0::2] = (ll + lh + hl + hh) * 0.5
        output[..., 0::2, 1::2] = (ll - lh + hl - hh) * 0.5
        output[..., 1::2, 0::2] = (ll + lh - hl - hh) * 0.5
        output[..., 1::2, 1::2] = (ll - lh - hl + hh) * 0.5
        return output[..., :height, :width]

    def forward(self, features: Iterable[torch.Tensor]) -> list[torch.Tensor]:
        values = list(features)
        if not self.training or self.probability <= 0.0 or not values:
            return values
        if float(torch.rand((), device=values[0].device)) >= self.probability:
            return values
        attenuation = 1.0 - torch.rand((), device=values[0].device) * self.maximum_attenuation
        return [self._haar_attenuate(feature, attenuation) for feature in values]


def build_feature_augmentation(config: dict | None) -> nn.Module:
    """Build a disabled identity or the explicit experimental augmentation."""
    cfg = dict(config or {})
    if not bool(cfg.get("enabled", False)):
        return nn.Identity()
    kind = str(cfg.get("type", "haar_detail_attenuation")).lower()
    if kind not in ("haar_detail_attenuation", "haar"):
        raise ValueError("Unknown feature_augmentation.type: {}".format(kind))
    return WaveletDetailAttenuation(
        probability=cfg.get("probability", 0.5),
        maximum_attenuation=cfg.get("maximum_attenuation", 0.35),
    )
