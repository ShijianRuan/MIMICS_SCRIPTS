"""Lightweight volume augmentation hooks.

The default Mimics few-shot integration keeps augmentation disabled. This class
exists so scripts/train.py can import the configured augmentation interface even
when no augmentation is requested.
"""

from __future__ import annotations

import random

import torch


class VolumeAugmentation:
    """Apply conservative tensor augmentations to an item dict."""

    def __init__(self, config: dict | None = None, seed: int = 42):
        self.config = config or {}
        self.rng = random.Random(seed)
        self.flip_probability = float(self.config.get("flip_probability", 0.0))

    def __call__(self, item: dict) -> dict:
        if self.flip_probability > 0.0 and self.rng.random() < self.flip_probability:
            axis = self.rng.choice([-1, -2])
            item = dict(item)
            item["image"] = torch.flip(item["image"], dims=(axis,))
            item["label"] = torch.flip(item["label"], dims=(axis,))
        return item
