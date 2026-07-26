"""Synthetic user-in-the-loop click generation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from scipy import ndimage

PREVIOUS_SEGMENTATION_CHANNEL = 0
POSITIVE_POINT_CHANNEL = 3
NEGATIVE_POINT_CHANNEL = 4
NUM_INTERACTION_CHANNELS = 7


def _choose_component(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray | None:
    components, count = ndimage.label(
        mask, structure=np.ones((3, 3, 3), dtype=np.uint8)
    )
    if count == 0:
        return None
    sizes = np.bincount(components.ravel())[1:].astype(np.float64)
    probabilities = sizes / sizes.sum()
    selected = int(rng.choice(np.arange(1, count + 1), p=probabilities))
    return components == selected


def sample_error_point(
    mask: np.ndarray,
    rng: np.random.Generator,
    center_bias: float = 8.0,
) -> tuple[int, int, int] | None:
    component = _choose_component(np.asarray(mask, dtype=bool), rng)
    if component is None:
        return None
    distance = ndimage.distance_transform_edt(component)
    coordinates = np.argwhere(component)
    weights = distance[component].astype(np.float64)
    if center_bias > 1:
        maximum = float(weights.max())
        if maximum > 0:
            weights = np.power(weights / maximum, center_bias)
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0:
        index = int(rng.integers(len(coordinates)))
    else:
        index = int(rng.choice(len(coordinates), p=weights / weights.sum()))
    return tuple(int(value) for value in coordinates[index])


def add_soft_point_(
    channel: torch.Tensor,
    coordinate: tuple[int, int, int],
    radius: int,
) -> None:
    """Place the same EDT-shaped spherical point used by nnInteractive."""
    radius = int(radius)
    point = _point_kernel(radius).to(device=channel.device, dtype=channel.dtype)
    starts = [max(0, coordinate[i] - radius) for i in range(3)]
    stops = [min(channel.shape[i], coordinate[i] + radius + 1) for i in range(3)]
    kernel_starts = [starts[i] - coordinate[i] + radius for i in range(3)]
    kernel_stops = [kernel_starts[i] + stops[i] - starts[i] for i in range(3)]
    slices = tuple(slice(starts[i], stops[i]) for i in range(3))
    kernel_slices = tuple(slice(kernel_starts[i], kernel_stops[i]) for i in range(3))
    torch.maximum(channel[slices], point[kernel_slices], out=channel[slices])


@lru_cache(maxsize=16)
def _point_kernel(radius: int) -> torch.Tensor:
    """Build the discrete spherical EDT kernel used by nnInteractive."""
    radius = int(radius)
    coordinates = np.ogrid[tuple(slice(-radius, radius + 1) for _ in range(3))]
    squared_distance = sum(axis.astype(np.float64) ** 2 for axis in coordinates)
    sphere = squared_distance <= float(radius * radius)
    distance = ndimage.distance_transform_edt(sphere)
    maximum = float(distance.max())
    if maximum > 0:
        distance /= maximum
    return torch.from_numpy(distance.astype(np.float32))


@dataclass
class InteractivePromptSampler:
    radius: int = 4
    center_bias: float = 8.0
    decay: float = 0.9
    seed: int = 20260724

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(int(self.seed))

    def new_interactions(self, target: torch.Tensor) -> torch.Tensor:
        shape = (target.shape[0], NUM_INTERACTION_CHANNELS, *target.shape[-3:])
        interactions = torch.zeros(shape, dtype=torch.float32, device=target.device)
        self.add_corrections(
            interactions, torch.zeros_like(target), target, initial=True
        )
        return interactions

    @torch.no_grad()
    def add_corrections(
        self,
        interactions: torch.Tensor,
        prediction: torch.Tensor,
        target: torch.Tensor,
        initial: bool = False,
        active: torch.Tensor | None = None,
    ) -> None:
        if not initial:
            interactions[:, 1:].mul_(float(self.decay))
        interactions[:, PREVIOUS_SEGMENTATION_CHANNEL].copy_(prediction.float())

        prediction_np = prediction.detach().cpu().numpy().astype(bool)
        target_np = target.detach().cpu().numpy().astype(bool)
        active_np = (
            np.ones(target.shape[0], dtype=bool)
            if active is None
            else active.detach().cpu().numpy().astype(bool)
        )
        for batch_index in range(target.shape[0]):
            if not active_np[batch_index]:
                continue
            false_negative = target_np[batch_index] & ~prediction_np[batch_index]
            false_positive = prediction_np[batch_index] & ~target_np[batch_index]
            positive = sample_error_point(false_negative, self.rng, self.center_bias)
            negative = sample_error_point(false_positive, self.rng, self.center_bias)
            if positive is not None:
                add_soft_point_(
                    interactions[batch_index, POSITIVE_POINT_CHANNEL],
                    positive,
                    self.radius,
                )
            if negative is not None:
                add_soft_point_(
                    interactions[batch_index, NEGATIVE_POINT_CHANNEL],
                    negative,
                    self.radius,
                )
