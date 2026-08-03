#!/usr/bin/env python3
"""Localized cubic B-spline free-form deformation for IGAC boundary pulls.

The implementation operates on a signed-distance field instead of converting a
Mask to a surface and rasterizing it again.  One compactly supported cubic
B-spline coefficient moves the grabbed boundary point exactly to the pointer;
the remaining displacement decays smoothly to zero in physical millimetres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class LocalFFDResult:
    slices_zyx: tuple[slice, slice, slice]
    values: Any
    drag_distance_mm: float
    applied_drag_distance_mm: float
    influence_radius_mm: float
    drag_was_clamped: bool
    applied_end_zyx: tuple[float, float, float]


def centered_cubic_bspline(torch: Any, coordinate: Any) -> Any:
    """Evaluate the centered cardinal cubic B-spline with support ``[-2, 2]``."""
    absolute = coordinate.abs()
    inner = (4.0 - 6.0 * absolute.square() + 3.0 * absolute.pow(3.0)) / 6.0
    outer = (2.0 - absolute).clamp_min(0.0).pow(3.0) / 6.0
    return torch.where(absolute < 1.0, inner, torch.where(absolute < 2.0, outer, 0.0))


def _weight(
    torch: Any,
    coordinates: Sequence[Any],
    center: Sequence[float],
    lattice_spacing_mm: float,
) -> Any:
    value = torch.ones_like(coordinates[0])
    for coordinate, origin in zip(coordinates, center):
        value = value * centered_cubic_bspline(
            torch,
            (coordinate - float(origin)) / float(lattice_spacing_mm),
        )
    # B3(0)^3 = (2/3)^3.  Normalizing makes the grabbed point follow the
    # pointer exactly while retaining a standard cubic B-spline falloff.
    return (value / ((2.0 / 3.0) ** 3)).clamp(0.0, 1.0)


def warp_signed_distance(
    *,
    torch: Any,
    functional: Any,
    base_phi: Any,
    spacing_zyx: Sequence[float],
    start_zyx: Sequence[float],
    end_zyx: Sequence[float],
    influence_radius_mm: float,
    maximum_drag_ratio: float = 0.65,
    inverse_iterations: int = 5,
) -> LocalFFDResult | None:
    """Warp a local signed-distance patch using a physical-space B-spline FFD.

    ``start_zyx`` and ``end_zyx`` are continuous voxel coordinates in the same
    local grid as ``base_phi``. The returned tensor contains only the affected
    patch. The support radius is fixed: a long pointer movement is clamped to a
    safe local displacement instead of silently turning into a global edit.
    """
    if int(base_phi.ndim) != 5:
        raise ValueError("FFD expects a [N, C, Z, Y, X] signed-distance tensor.")
    if len(spacing_zyx) != 3 or len(start_zyx) != 3 or len(end_zyx) != 3:
        raise ValueError("FFD requires three spacing, start, and end values.")
    spacing = tuple(float(value) for value in spacing_zyx)
    if any(value <= 0.0 or not math.isfinite(value) for value in spacing):
        raise ValueError("FFD spacing values must be finite and positive.")

    start_physical = tuple(float(value) * step for value, step in zip(start_zyx, spacing))
    end_physical = tuple(float(value) * step for value, step in zip(end_zyx, spacing))
    delta = tuple(end - start for start, end in zip(start_physical, end_physical))
    distance = math.sqrt(sum(value * value for value in delta))
    if not math.isfinite(distance) or distance < 1.0e-4:
        return None

    requested_radius = max(2.0, float(influence_radius_mm))
    ratio = min(0.8, max(0.2, float(maximum_drag_ratio)))
    maximum_distance = requested_radius * ratio
    applied_scale = min(1.0, maximum_distance / distance)
    applied_delta = tuple(value * applied_scale for value in delta)
    applied_end_physical = tuple(
        start + value for start, value in zip(start_physical, applied_delta)
    )
    applied_distance = distance * applied_scale
    effective_radius = requested_radius
    lattice_spacing = effective_radius / 2.0

    shape_zyx = tuple(int(value) for value in base_phi.shape[-3:])
    axes: list[tuple[int, int]] = []
    for start, end, step, size in zip(
        start_physical, applied_end_physical, spacing, shape_zyx
    ):
        lower = max(0, int(math.floor((min(start, end) - effective_radius) / step)) - 1)
        upper = min(size, int(math.ceil((max(start, end) + effective_radius) / step)) + 2)
        axes.append((lower, upper))
    if any(upper <= lower for lower, upper in axes):
        return None

    dtype = base_phi.dtype
    device = base_phi.device
    index_grids = torch.meshgrid(
        *[
            torch.arange(lower, upper, device=device, dtype=dtype)
            for lower, upper in axes
        ],
        indexing="ij",
    )
    target_physical = [
        grid * float(step) for grid, step in zip(index_grids, spacing)
    ]
    source_physical = [value.clone() for value in target_physical]
    for _unused in range(max(2, int(inverse_iterations))):
        weight = _weight(torch, source_physical, start_physical, lattice_spacing)
        source_physical = [
            target - weight * float(component)
            for target, component in zip(target_physical, applied_delta)
        ]

    source_indices = [
        coordinate / float(step)
        for coordinate, step in zip(source_physical, spacing)
    ]

    def normalized(coordinate: Any, size: int) -> Any:
        if size <= 1:
            return torch.zeros_like(coordinate)
        return 2.0 * coordinate / float(size - 1) - 1.0

    grid = torch.stack(
        (
            normalized(source_indices[2], shape_zyx[2]),
            normalized(source_indices[1], shape_zyx[1]),
            normalized(source_indices[0], shape_zyx[0]),
        ),
        dim=-1,
    ).unsqueeze(0)
    values = functional.grid_sample(
        base_phi,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return LocalFFDResult(
        slices_zyx=tuple(slice(lower, upper) for lower, upper in axes),  # type: ignore[arg-type]
        values=values,
        drag_distance_mm=float(distance),
        applied_drag_distance_mm=float(applied_distance),
        influence_radius_mm=float(effective_radius),
        drag_was_clamped=bool(applied_scale < 1.0 - 1.0e-6),
        applied_end_zyx=tuple(
            value / step for value, step in zip(applied_end_physical, spacing)
        ),
    )
