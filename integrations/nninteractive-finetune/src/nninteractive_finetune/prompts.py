"""User-in-the-loop click generation for empty or real Initial Masks."""

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
CORRECTION_POLICY_OFFICIAL_SINGLE = "official_single"
CORRECTION_POLICY_CLOPA_PAIRED = "clopa_paired"


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


def _sample_component_point(
    component: np.ndarray,
    rng: np.random.Generator,
    center_bias: float,
) -> tuple[int, int, int] | None:
    if not np.any(component):
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


def sample_error_point(
    mask: np.ndarray,
    rng: np.random.Generator,
    center_bias: float = 8.0,
) -> tuple[int, int, int] | None:
    component = _choose_component(np.asarray(mask, dtype=bool), rng)
    if component is None:
        return None
    return _sample_component_point(component, rng, center_bias)


def sample_correction_event(
    false_negative: np.ndarray,
    false_positive: np.ndarray,
    rng: np.random.Generator,
    center_bias: float = 8.0,
) -> tuple[tuple[int, int, int], bool] | None:
    """Select one signed error component using nnInteractive-style weighting."""
    candidates: list[tuple[np.ndarray, bool, int]] = []
    for mask, include in (
        (np.asarray(false_negative, dtype=bool), True),
        (np.asarray(false_positive, dtype=bool), False),
    ):
        components, count = ndimage.label(
            mask, structure=np.ones((3, 3, 3), dtype=np.uint8)
        )
        if count <= 0:
            continue
        sizes = np.bincount(components.ravel())[1:]
        for component_index, size in enumerate(sizes, start=1):
            if int(size) > 0:
                candidates.append(
                    (components == component_index, include, int(size))
                )
    if not candidates:
        return None
    weights = np.asarray([item[2] for item in candidates], dtype=np.float64)
    selected = int(rng.choice(len(candidates), p=weights / weights.sum()))
    component, include, _size = candidates[selected]
    point = _sample_component_point(component, rng, center_bias)
    if point is None:
        return None
    return point, include


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
    initial_mask_probability: float = 0.0
    correction_policy: str = CORRECTION_POLICY_CLOPA_PAIRED

    def __post_init__(self) -> None:
        if self.correction_policy not in {
            CORRECTION_POLICY_OFFICIAL_SINGLE,
            CORRECTION_POLICY_CLOPA_PAIRED,
        }:
            raise ValueError(
                "Unsupported correction policy: {}".format(
                    self.correction_policy
                )
            )
        if not 0.0 <= float(self.initial_mask_probability) <= 1.0:
            raise ValueError("initial_mask_probability must be in [0, 1].")
        self.rng = np.random.default_rng(int(self.seed))

    def sample_interaction_budgets(
        self,
        batch_size: int,
        minimum: int,
        maximum: int,
        short_probability: float = 0.7,
        weights: list[float] | None = None,
    ) -> torch.Tensor:
        """Draw per-sample sequential interaction lengths."""
        minimum = int(minimum)
        maximum = int(maximum)
        if maximum < minimum:
            raise ValueError("maximum interaction budget cannot be below minimum")
        choices = np.arange(minimum, maximum + 1, dtype=np.int64)
        if weights is not None:
            probabilities = np.asarray(weights, dtype=np.float64)
            if len(probabilities) != len(choices):
                raise ValueError(
                    "interaction budget weights must match the configured range"
                )
            if (
                not np.isfinite(probabilities).all()
                or np.any(probabilities < 0)
                or float(probabilities.sum()) <= 0
            ):
                raise ValueError(
                    "interaction budget weights must be finite, non-negative, "
                    "and contain a positive value"
                )
            probabilities /= probabilities.sum()
            values = self.rng.choice(
                choices, size=int(batch_size), p=probabilities
            )
        elif maximum <= minimum:
            values = np.full(int(batch_size), minimum, dtype=np.int64)
        else:
            short_max = min(maximum, max(minimum, 5))
            values = []
            for _ in range(int(batch_size)):
                upper = (
                    short_max
                    if self.rng.random() < float(short_probability)
                    else maximum
                )
                values.append(int(self.rng.integers(minimum, upper + 1)))
            values = np.asarray(values, dtype=np.int64)
        return torch.from_numpy(values)

    def _real_or_empty_initial_prediction(
        self,
        target: torch.Tensor,
        probability: float | None = None,
        provided: torch.Tensor | None = None,
        provided_available: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Use a supplied real Initial Mask or an empty Mask per sample."""
        target_np = target.detach().cpu().numpy().astype(bool)
        predictions = np.zeros_like(target_np, dtype=bool)
        provided_np = (
            provided.detach().cpu().numpy().astype(bool)
            if provided is not None
            else None
        )
        available_np = (
            provided_available.detach().cpu().numpy().astype(bool)
            if provided_available is not None
            else np.ones(target.shape[0], dtype=bool)
        )
        probability = (
            float(self.initial_mask_probability)
            if probability is None
            else float(probability)
        )
        for batch_index, foreground in enumerate(target_np):
            if (
                not foreground.any()
                or self.rng.random() >= probability
            ):
                continue
            if (
                provided_np is not None
                and available_np[batch_index]
                and provided_np[batch_index].any()
                and not np.array_equal(
                    provided_np[batch_index], foreground
                )
            ):
                predictions[batch_index] = provided_np[batch_index]
        return torch.from_numpy(predictions).to(
            device=target.device, dtype=target.dtype
        )

    def initial_prediction(
        self,
        target: torch.Tensor,
        *,
        force: bool = False,
        provided: torch.Tensor | None = None,
        provided_available: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the initial Mask used before the first correction."""
        return self._real_or_empty_initial_prediction(
            target,
            probability=1.0 if force else None,
            provided=provided,
            provided_available=provided_available,
        )

    def new_interactions(
        self,
        target: torch.Tensor,
        allow_initial_mask: bool = True,
        force_initial_mask: bool = False,
        initial_prediction: torch.Tensor | None = None,
        initial_mask_available: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shape = (target.shape[0], NUM_INTERACTION_CHANNELS, *target.shape[-3:])
        interactions = torch.zeros(shape, dtype=torch.float32, device=target.device)
        prediction = (
            self.initial_prediction(
                target,
                force=force_initial_mask,
                provided=initial_prediction,
                provided_available=initial_mask_available,
            )
            if allow_initial_mask
            else torch.zeros_like(target)
        )
        self.add_corrections(
            interactions, prediction, target, initial=True
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
            if self.correction_policy == CORRECTION_POLICY_CLOPA_PAIRED:
                events = [
                    (
                        sample_error_point(
                            false_negative, self.rng, self.center_bias
                        ),
                        True,
                    ),
                    (
                        sample_error_point(
                            false_positive, self.rng, self.center_bias
                        ),
                        False,
                    ),
                ]
            else:
                selected = sample_correction_event(
                    false_negative,
                    false_positive,
                    self.rng,
                    self.center_bias,
                )
                events = [] if selected is None else [selected]
            for point, include in events:
                if point is None:
                    continue
                channel = (
                    POSITIVE_POINT_CHANNEL
                    if include
                    else NEGATIVE_POINT_CHANNEL
                )
                add_soft_point_(
                    interactions[batch_index, channel],
                    point,
                    self.radius,
                )
