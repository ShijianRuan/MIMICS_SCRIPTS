#!/usr/bin/env python3
"""Spacing-aware PyTorch implementation of interactive 3D IGAC evolution.

The upstream IGAC application is an OpenCL/OpenGL desktop program.  This
module keeps the LGDF-style contour evolution and Add/Neutral/Barrier tools,
but operates directly on Mimics voxel buffers and treats every spatial
parameter in millimetres.  The external GUI is the only caller; no Mimics API
is imported here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from tools.local_bspline_ffd import warp_signed_distance


ORIENTATIONS = ("axial", "coronal", "sagittal")


def intensity_range(values: np.ndarray, maximum_samples: int = 2_000_000) -> tuple[float, float]:
    # Preserve memmap laziness: converting the complete volume to float32 here
    # doubled startup I/O and memory before the ROI was even known.
    array = np.asarray(values)
    stride = max(1, int(math.ceil(float(array.size) / float(maximum_samples))))
    sample = np.asarray(array.reshape(-1)[::stride], dtype=np.float32)
    finite = sample[np.isfinite(sample)]
    if finite.size == 0:
        raise RuntimeError("The active image contains no finite voxel values.")
    low, high = np.percentile(finite, (1.0, 99.0))
    if not math.isfinite(float(low)) or not math.isfinite(float(high)) or high <= low:
        low, high = float(finite.min()), float(finite.max())
    if high <= low:
        raise RuntimeError("The active image has no usable intensity variation.")
    return float(low), float(high)


def normalize_image(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    array = np.asarray(values, dtype=np.float32)
    low, high = intensity_range(array)
    normalized = np.clip((array - low) / (high - low), 0.0, 1.0)
    normalized[~np.isfinite(normalized)] = 0.0
    return normalized.astype(np.float32, copy=False), float(low), float(high)


def plane_from_xyz(array: np.ndarray, orientation: str, index: int) -> np.ndarray:
    """Return a display-ready plane whose first dimension is screen Y."""
    orientation = str(orientation).lower()
    if orientation == "axial":
        return np.asarray(array[:, :, int(index)]).T
    if orientation == "coronal":
        return np.asarray(array[:, int(index), :]).T
    if orientation == "sagittal":
        return np.asarray(array[int(index), :, :]).T
    raise ValueError("Unknown orientation: {}".format(orientation))


def display_to_xyz(orientation: str, index: int, u: float, v: float) -> tuple[float, float, float]:
    orientation = str(orientation).lower()
    if orientation == "axial":
        return float(u), float(v), float(index)
    if orientation == "coronal":
        return float(u), float(index), float(v)
    if orientation == "sagittal":
        return float(index), float(u), float(v)
    raise ValueError("Unknown orientation: {}".format(orientation))


def orientation_extent(shape_xyz: Iterable[int], orientation: str) -> tuple[int, int, int]:
    x, y, z = (int(value) for value in shape_xyz)
    orientation = str(orientation).lower()
    if orientation == "axial":
        return x, y, z
    if orientation == "coronal":
        return x, z, y
    if orientation == "sagittal":
        return y, z, x
    raise ValueError("Unknown orientation: {}".format(orientation))


def _mask_bounds(mask_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Return exclusive 3D foreground bounds without allocating Nx3 indices."""
    mask = np.asarray(mask_xyz, dtype=bool)
    minimum = []
    maximum = []
    for axis in range(3):
        reduced_axes = tuple(value for value in range(3) if value != axis)
        occupied = np.flatnonzero(np.any(mask, axis=reduced_axes))
        if occupied.size == 0:
            return None
        minimum.append(int(occupied[0]))
        maximum.append(int(occupied[-1]) + 1)
    return np.asarray(minimum, dtype=np.int64), np.asarray(maximum, dtype=np.int64)


def roi_slices(
    mask_xyz: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    margin_mm: float,
    max_voxels: int,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[slice, slice, slice]:
    shape = tuple(int(value) for value in mask_xyz.shape)
    bounds = bounds if bounds is not None else _mask_bounds(mask_xyz)
    if bounds is None:
        voxel_count = int(np.prod(shape))
        if voxel_count > int(max_voxels):
            raise RuntimeError(
                "The selected Mask is empty and the full image has {:,} voxels. "
                "Create a rough seed Mask first so IGAC can use a bounded 3D workspace."
                .format(voxel_count)
            )
        return tuple(slice(0, value) for value in shape)  # type: ignore[return-value]
    minimum, maximum = bounds
    result = []
    for axis in range(3):
        margin = max(2, int(math.ceil(float(margin_mm) / float(spacing_xyz[axis]))))
        result.append(
            slice(
                max(0, int(minimum[axis]) - margin),
                min(shape[axis], int(maximum[axis]) + margin),
            )
        )
    voxel_count = int(np.prod([item.stop - item.start for item in result]))
    if voxel_count > int(max_voxels):
        raise RuntimeError(
            "The IGAC workspace contains {:,} voxels, above the configured limit of {:,}. "
            "Use a tighter initial Mask or reduce workspace_margin_mm."
            .format(voxel_count, int(max_voxels))
        )
    return tuple(result)  # type: ignore[return-value]


@dataclass
class IGACParameters:
    range_mm: float = 3.0
    smooth: float = 50.0
    grow: float = 0.0
    max_displacement_mm: float = 10.0
    data_weight: float = 30.0
    regularization: float = 1.0
    time_step: float = 0.1
    variance_floor: float = 1.0e-4
    phi_limit: float = 20.0
    gaussian_radius_limit: int = 24

    @classmethod
    def from_mapping(cls, values: dict[str, Any] | None) -> "IGACParameters":
        source = values or {}
        return cls(
            range_mm=max(0.1, float(source.get("range_mm", 3.0))),
            smooth=max(0.0, float(source.get("smooth", 50.0))),
            grow=float(source.get("grow", 0.0)),
            max_displacement_mm=max(0.0, float(source.get("max_displacement_mm", 10.0))),
            data_weight=max(0.0, float(source.get("data_weight", 30.0))),
            regularization=max(0.0, float(source.get("regularization", 1.0))),
            time_step=max(1.0e-4, float(source.get("time_step", 0.1))),
            variance_floor=max(1.0e-8, float(source.get("variance_floor", 1.0e-4))),
            phi_limit=max(2.0, float(source.get("phi_limit", 20.0))),
            gaussian_radius_limit=max(2, int(source.get("gaussian_radius_limit", 24))),
        )


@dataclass
class IGACConvergenceMonitor:
    """Stop interactive evolution after the visible Mask becomes stable."""

    minimum_iterations: int = 2
    maximum_iterations: int = 24
    stable_cycles_required: int = 2
    changed_voxel_ratio: float = 2.0e-4
    changed_voxel_limit: int = 32
    maximum_seconds: float = 5.0
    stable_cycles: int = 0

    @classmethod
    def from_mapping(cls, values: dict[str, Any] | None) -> "IGACConvergenceMonitor":
        source = values or {}
        minimum_iterations = max(1, int(source.get("minimum_iterations", 2)))
        return cls(
            minimum_iterations=minimum_iterations,
            maximum_iterations=max(
                minimum_iterations,
                int(source.get("maximum_iterations", 24)),
            ),
            stable_cycles_required=max(1, int(source.get("stable_cycles", 2))),
            changed_voxel_ratio=max(0.0, float(source.get("changed_voxel_ratio", 2.0e-4))),
            changed_voxel_limit=max(1, int(source.get("changed_voxel_limit", 32))),
            maximum_seconds=max(0.1, float(source.get("maximum_seconds", 5.0))),
        )

    def reset(self) -> None:
        self.stable_cycles = 0

    def observe(
        self,
        *,
        iterations: int,
        elapsed_seconds: float,
        changed_voxel_ratio: float,
        changed_voxels: int | None = None,
        observed_voxels: int | None = None,
    ) -> str | None:
        stable = float(changed_voxel_ratio) <= self.changed_voxel_ratio
        if changed_voxels is not None and observed_voxels is not None:
            ratio_limit = int(
                math.ceil(max(1, int(observed_voxels)) * self.changed_voxel_ratio)
            )
            allowed = max(1, min(self.changed_voxel_limit, ratio_limit))
            stable = int(changed_voxels) <= allowed
        if stable:
            self.stable_cycles += 1
        else:
            self.stable_cycles = 0
        if (
            int(iterations) >= self.minimum_iterations
            and self.stable_cycles >= self.stable_cycles_required
        ):
            return "converged"
        if int(iterations) >= self.maximum_iterations:
            return "iteration_limit"
        if float(elapsed_seconds) >= self.maximum_seconds:
            return "time_limit"
        return None


class IGACEngine:
    """Own the 3D level set, guidance constraints, and display-plane mapping."""

    def __init__(
        self,
        image_xyz: np.ndarray,
        initial_mask_xyz: np.ndarray,
        spacing_xyz: tuple[float, float, float],
        *,
        device: str = "auto",
        workspace_margin_mm: float = 40.0,
        max_roi_voxels: int = 10_000_000,
        parameters: dict[str, Any] | None = None,
        display_window: tuple[float, float] | None = None,
    ) -> None:
        try:
            import torch
            import torch.nn.functional as functional
        except ImportError as exc:
            raise RuntimeError("IGAC requires PyTorch in nninteractive_env: {}".format(exc))
        try:
            from scipy.ndimage import distance_transform_edt
        except ImportError as exc:
            raise RuntimeError("IGAC requires scipy in nninteractive_env: {}".format(exc))

        self.torch = torch
        self.functional = functional
        self.distance_transform_edt = distance_transform_edt
        self.shape_xyz = tuple(int(value) for value in image_xyz.shape)
        if self.shape_xyz != tuple(int(value) for value in initial_mask_xyz.shape):
            raise RuntimeError("The Mimics image and selected Mask shapes differ.")
        self.spacing_xyz = tuple(float(value) for value in spacing_xyz)
        if len(self.spacing_xyz) != 3 or any(value <= 0 or not math.isfinite(value) for value in self.spacing_xyz):
            raise RuntimeError("IGAC requires three positive voxel spacing values.")
        self.parameters = IGACParameters.from_mapping(parameters)
        self.initial_mask_xyz = np.asarray(initial_mask_xyz, dtype=bool).copy()
        bounds = _mask_bounds(self.initial_mask_xyz)
        self.initial_bbox_xyz = None
        if bounds is not None:
            minimum, maximum = bounds
            self.initial_bbox_xyz = [
                [int(minimum[axis]), int(maximum[axis])] for axis in range(3)
            ]
        self.image_source_xyz = np.asarray(image_xyz)
        self.intensity_low, self.intensity_high = intensity_range(self.image_source_xyz)
        self.display_low, self.display_high = self._validated_display_window(display_window)
        self.initial_foreground_voxels = int(np.count_nonzero(self.initial_mask_xyz))
        self.roi = roi_slices(
            self.initial_mask_xyz,
            self.spacing_xyz,
            float(workspace_margin_mm),
            int(max_roi_voxels),
            bounds=bounds,
        )
        self.roi_shape_xyz = tuple(item.stop - item.start for item in self.roi)
        self.roi_offset_xyz = tuple(int(item.start) for item in self.roi)

        requested = str(device or "auto").lower()
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("IGAC was configured for CUDA, but CUDA is unavailable.")
        self.device = torch.device(requested)

        crop_image_xyz = np.asarray(self.image_source_xyz[self.roi], dtype=np.float32)
        crop_image_xyz = np.clip(
            (crop_image_xyz - self.intensity_low) / (self.intensity_high - self.intensity_low),
            0.0,
            1.0,
        )
        crop_image_xyz[~np.isfinite(crop_image_xyz)] = 0.0
        crop_mask_xyz = np.asarray(self.initial_mask_xyz[self.roi], dtype=bool)
        image_zyx = np.ascontiguousarray(np.transpose(crop_image_xyz, (2, 1, 0)))
        mask_zyx = np.ascontiguousarray(np.transpose(crop_mask_xyz, (2, 1, 0)))
        self.image = torch.from_numpy(image_zyx)[None, None].to(self.device)
        self.image_sq = self.image.square()
        self.spacing_zyx = tuple(reversed(self.spacing_xyz))
        self._initial_phi = self._signed_distance(mask_zyx)
        self.phi = self._initial_phi.clone()
        self.add_constraint = torch.zeros_like(self.phi, dtype=torch.bool)
        self.barrier_constraint = torch.zeros_like(self.phi, dtype=torch.bool)
        self.iteration = 0
        self.last_step_changed_voxels = 0
        self.last_step_changed_ratio = 0.0
        self.last_step_observed_voxels = self.roi_voxels
        self.evolution_slices_zyx: tuple[slice, slice, slice] | None = None
        self.last_edit_slices_zyx: tuple[slice, slice, slice] | None = None
        self._kernel_signature: tuple[float, tuple[float, float, float]] | None = None
        self._kernels: list[Any] = []
        self._refresh_gaussian_cache()

    @property
    def roi_voxels(self) -> int:
        return int(np.prod(self.roi_shape_xyz))

    @property
    def estimated_memory_mb(self) -> float:
        # Persistent fields plus temporary LGDF convolutions.  This deliberately
        # overestimates so a clinical workstation fails before CUDA OOM.
        return float(self.roi_voxels * 112) / (1024.0 * 1024.0)

    def metadata(self) -> dict[str, Any]:
        return {
            "device": str(self.device),
            "shape_xyz": list(self.shape_xyz),
            "spacing_xyz": list(self.spacing_xyz),
            "roi_shape_xyz": list(self.roi_shape_xyz),
            "roi_offset_xyz": list(self.roi_offset_xyz),
            "roi_voxels": self.roi_voxels,
            "estimated_memory_mb": round(self.estimated_memory_mb, 1),
            "intensity_low": self.intensity_low,
            "intensity_high": self.intensity_high,
            "display_low": self.display_low,
            "display_high": self.display_high,
            "initial_foreground_voxels": self.initial_foreground_voxels,
            "initial_bbox_xyz": self.initial_bbox_xyz,
        }

    def _validated_display_window(
        self, values: tuple[float, float] | None
    ) -> tuple[float, float]:
        if values is not None and len(values) == 2:
            low, high = float(values[0]), float(values[1])
            if math.isfinite(low) and math.isfinite(high) and high > low:
                return low, high
        return float(self.intensity_low), float(self.intensity_high)

    def set_display_window(self, low: float, high: float) -> None:
        self.display_low, self.display_high = self._validated_display_window((low, high))

    def foreground_voxels(self) -> int:
        with self.torch.inference_mode():
            return int(self.torch.count_nonzero(self.phi < 0).item())

    def _signed_distance(self, mask_zyx: np.ndarray) -> Any:
        if np.any(mask_zyx):
            outside = self.distance_transform_edt(~mask_zyx, sampling=self.spacing_zyx)
            inside = self.distance_transform_edt(mask_zyx, sampling=self.spacing_zyx)
            values = outside - inside
        else:
            values = np.full(mask_zyx.shape, self.parameters.phi_limit, dtype=np.float32)
        return self.torch.from_numpy(np.asarray(values, dtype=np.float32))[None, None].to(self.device)

    def _kernel1d(self, sigma_voxels: float) -> Any:
        sigma = max(0.05, float(sigma_voxels))
        radius = min(
            self.parameters.gaussian_radius_limit,
            max(1, int(math.ceil(3.0 * sigma))),
        )
        positions = self.torch.arange(-radius, radius + 1, device=self.device, dtype=self.image.dtype)
        kernel = self.torch.exp(-0.5 * (positions / sigma).square())
        return kernel / kernel.sum().clamp_min(1.0e-8)

    def _refresh_gaussian_cache(self) -> None:
        signature = (float(self.parameters.range_mm), self.spacing_zyx)
        if signature == self._kernel_signature:
            return
        self._kernel_signature = signature
        self._kernels = [
            self._kernel1d(self.parameters.range_mm / spacing)
            for spacing in self.spacing_zyx
        ]
        with self.torch.inference_mode():
            self.gaussian_image = self._gaussian(self.image)
            self.gaussian_image_sq = self._gaussian(self.image_sq)

    def set_parameters(self, values: dict[str, Any]) -> None:
        previous_range = self.parameters.range_mm
        merged = dict(self.parameters.__dict__)
        merged.update(values or {})
        self.parameters = IGACParameters.from_mapping(merged)
        if self.parameters.range_mm != previous_range:
            self._refresh_gaussian_cache()

    def _gaussian(self, value: Any) -> Any:
        output = value
        # Tensor dimensions are N,C,Z,Y,X.  Replicate padding prevents an
        # artificial zero-intensity boundary around the cropped workspace.
        for axis, kernel in enumerate(self._kernels):
            radius = int((kernel.numel() - 1) // 2)
            if axis == 0:
                padding = (0, 0, 0, 0, radius, radius)
                weight = kernel.view(1, 1, -1, 1, 1)
            elif axis == 1:
                padding = (0, 0, radius, radius, 0, 0)
                weight = kernel.view(1, 1, 1, -1, 1)
            else:
                padding = (radius, radius, 0, 0, 0, 0)
                weight = kernel.view(1, 1, 1, 1, -1)
            output = self.functional.conv3d(self.functional.pad(output, padding, mode="replicate"), weight)
        return output

    def _shifted(self, value: Any, axis: int) -> tuple[Any, Any]:
        if axis == 0:
            padded = self.functional.pad(value, (0, 0, 0, 0, 1, 1), mode="replicate")
            return padded[:, :, 2:, :, :], padded[:, :, :-2, :, :]
        if axis == 1:
            padded = self.functional.pad(value, (0, 0, 1, 1, 0, 0), mode="replicate")
            return padded[:, :, :, 2:, :], padded[:, :, :, :-2, :]
        padded = self.functional.pad(value, (1, 1, 0, 0, 0, 0), mode="replicate")
        return padded[:, :, :, :, 2:], padded[:, :, :, :, :-2]

    def _gradient(self, value: Any) -> list[Any]:
        result = []
        for axis, spacing in enumerate(self.spacing_zyx):
            plus, minus = self._shifted(value, axis)
            result.append((plus - minus) / (2.0 * float(spacing)))
        return result

    def _laplacian(self, value: Any) -> Any:
        result = self.torch.zeros_like(value)
        for axis, spacing in enumerate(self.spacing_zyx):
            plus, minus = self._shifted(value, axis)
            result = result + (plus - 2.0 * value + minus) / (float(spacing) ** 2)
        return result

    def _curvature(self, value: Any) -> Any:
        gradient = self._gradient(value)
        magnitude = self.torch.sqrt(sum(component.square() for component in gradient) + 1.0e-8)
        normals = [component / magnitude for component in gradient]
        divergence = self.torch.zeros_like(value)
        for axis, (normal, spacing) in enumerate(zip(normals, self.spacing_zyx)):
            plus, minus = self._shifted(normal, axis)
            divergence = divergence + (plus - minus) / (2.0 * float(spacing))
        return divergence

    def step(self, iterations: int = 1) -> int:
        torch = self.torch
        p = self.parameters
        self._refresh_gaussian_cache()
        with torch.inference_mode():
            full = tuple(
                slice(0, int(value)) for value in reversed(self.roi_shape_xyz)
            )
            active = self.evolution_slices_zyx or full
            tensor_slice = (slice(None), slice(None)) + active
            before = (self.phi[tensor_slice] < 0).clone()
            for _ in range(max(1, int(iterations))):
                if self.evolution_slices_zyx is None:
                    work = full
                    active_in_work = full
                    phi_field = self.phi
                    image_field = self.image
                    image_sq_field = self.image_sq
                    gaussian_image = self.gaussian_image
                    gaussian_image_sq = self.gaussian_image_sq
                else:
                    shape_zyx = tuple(int(value) for value in reversed(self.roi_shape_xyz))
                    work_values = []
                    relative_values = []
                    for item, kernel, size in zip(active, self._kernels, shape_zyx):
                        # The LGDF force contains two nested Gaussian passes
                        # (local statistics, then coefficient integration).
                        # Two kernel radii plus the derivative stencil preserve
                        # the same force inside the writable local block.
                        halo = 2 * int((kernel.numel() - 1) // 2) + 2
                        lower = max(0, int(item.start) - halo)
                        upper = min(size, int(item.stop) + halo)
                        work_values.append(slice(lower, upper))
                        relative_values.append(
                            slice(int(item.start) - lower, int(item.stop) - lower)
                        )
                    work = tuple(work_values)
                    active_in_work = tuple(relative_values)
                    work_slice = (slice(None), slice(None)) + work
                    phi_field = self.phi[work_slice]
                    image_field = self.image[work_slice]
                    image_sq_field = self.image_sq[work_slice]
                    gaussian_image = self._gaussian(image_field)
                    gaussian_image_sq = self._gaussian(image_sq_field)
                # phi < 0 is foreground.  This Heaviside therefore represents
                # foreground probability and has a positive Dirac magnitude.
                foreground = 0.5 * (1.0 - (2.0 / math.pi) * torch.atan(phi_field))
                k_fg = self._gaussian(foreground).clamp_min(1.0e-5)
                k_bg = (1.0 - k_fg).clamp_min(1.0e-5)
                image_fg = self._gaussian(foreground * image_field)
                image_sq_fg = self._gaussian(foreground * image_sq_field)
                mean_fg = image_fg / k_fg
                mean_bg = (gaussian_image - image_fg) / k_bg
                var_fg = (
                    image_sq_fg / k_fg - mean_fg.square()
                ).clamp_min(p.variance_floor)
                var_bg = (
                    (gaussian_image_sq - image_sq_fg) / k_bg - mean_bg.square()
                ).clamp_min(p.variance_floor)

                # The upstream kernel defines H for phi > 0 (background).  This
                # implementation uses H(-phi) for foreground readability, so
                # the two lambda terms and update sign are reversed together.
                lambda_fg = max(0.05, 1.0 + float(p.grow))
                lambda_bg = 1.0
                coefficient_a = lambda_fg / (2.0 * var_fg) - lambda_bg / (2.0 * var_bg)
                coefficient_b = -lambda_fg * mean_fg / var_fg + lambda_bg * mean_bg / var_bg
                coefficient_c = (
                    0.5 * lambda_fg * (torch.log(var_fg) + mean_fg.square() / var_fg)
                    - 0.5 * lambda_bg * (torch.log(var_bg) + mean_bg.square() / var_bg)
                )
                local_force = (
                    (lambda_fg - lambda_bg) * 0.9189385332
                    + image_sq_field * self._gaussian(coefficient_a)
                    + image_field * self._gaussian(coefficient_b)
                    + self._gaussian(coefficient_c)
                ).clamp(-10.0, 10.0)

                dirac = (1.0 / math.pi) / (1.0 + phi_field.square())
                curvature = self._curvature(phi_field)
                regularization = self._laplacian(phi_field) - curvature
                update = (
                    p.data_weight * dirac * local_force
                    + p.smooth * dirac * curvature
                    + p.regularization * regularization
                )
                update_slice = (slice(None), slice(None)) + active_in_work
                self.phi[tensor_slice].add_(p.time_step * update[update_slice])
                self.phi[tensor_slice].nan_to_num_(
                    nan=p.phi_limit,
                    posinf=p.phi_limit,
                    neginf=-p.phi_limit,
                )
                self.phi[tensor_slice].clamp_(-p.phi_limit, p.phi_limit)
                self._apply_constraints(active)
                self.iteration += 1
            changed = torch.count_nonzero(before != (self.phi[tensor_slice] < 0))
            self.last_step_changed_voxels = int(changed.item())
            self.last_step_observed_voxels = int(
                np.prod([int(item.stop) - int(item.start) for item in active])
            )
            self.last_step_changed_ratio = (
                float(self.last_step_changed_voxels)
                / float(max(1, self.last_step_observed_voxels))
            )
        return self.iteration

    def _apply_constraints(
        self, slices_zyx: tuple[slice, slice, slice] | None = None
    ) -> None:
        active = slices_zyx or tuple(
            slice(0, int(value)) for value in reversed(self.roi_shape_xyz)
        )
        tensor_slice = (slice(None), slice(None)) + active
        phi = self.phi[tensor_slice]
        initial_phi = self._initial_phi[tensor_slice]
        maximum = float(self.parameters.max_displacement_mm)
        if maximum > 0.0 and self.initial_foreground_voxels > 0:
            phi.masked_fill_(initial_phi > maximum, 6.0)
            phi.masked_fill_(initial_phi < -maximum, -6.0)
        phi.masked_fill_(self.add_constraint[tensor_slice], -6.0)
        phi.masked_fill_(self.barrier_constraint[tensor_slice], 6.0)

    @staticmethod
    def _merge_slices(
        first: tuple[slice, slice, slice] | None,
        second: tuple[slice, slice, slice] | None,
    ) -> tuple[slice, slice, slice] | None:
        if first is None:
            return second
        if second is None:
            return first
        return tuple(
            slice(min(int(a.start), int(b.start)), max(int(a.stop), int(b.stop)))
            for a, b in zip(first, second)
        )  # type: ignore[return-value]

    def begin_local_edit(self) -> None:
        self.last_edit_slices_zyx = None
        self.evolution_slices_zyx = None

    def set_local_evolution_region(
        self,
        slices_zyx: tuple[slice, slice, slice] | None,
        margin_mm: float = 3.0,
    ) -> bool:
        if slices_zyx is None:
            self.evolution_slices_zyx = None
            return False
        shape_zyx = tuple(int(value) for value in reversed(self.roi_shape_xyz))
        expanded = []
        for item, spacing, size in zip(slices_zyx, self.spacing_zyx, shape_zyx):
            margin = max(0, int(math.ceil(max(0.0, float(margin_mm)) / spacing)))
            expanded.append(
                slice(max(0, int(item.start) - margin), min(size, int(item.stop) + margin))
            )
        self.evolution_slices_zyx = tuple(expanded)  # type: ignore[assignment]
        return True

    def clear_evolution_region(self) -> None:
        self.evolution_slices_zyx = None

    def reset(self) -> None:
        with self.torch.inference_mode():
            self.phi.copy_(self._initial_phi)
            self.add_constraint.zero_()
            self.barrier_constraint.zero_()
        self.iteration = 0
        self.last_step_changed_voxels = 0
        self.last_step_changed_ratio = 0.0
        self.last_step_observed_voxels = self.roi_voxels
        self.evolution_slices_zyx = None
        self.last_edit_slices_zyx = None

    def capture_state(self) -> dict[str, Any]:
        """Capture one exact device-local undo point for an interactive action."""
        with self.torch.inference_mode():
            return {
                "phi": self.phi.clone(),
                "add_constraint": self.add_constraint.clone(),
                "barrier_constraint": self.barrier_constraint.clone(),
                "iteration": int(self.iteration),
                "evolution_slices_zyx": self.evolution_slices_zyx,
            }

    def restore_state(self, state: dict[str, Any]) -> None:
        required = ("phi", "add_constraint", "barrier_constraint", "iteration")
        if not isinstance(state, dict) or any(key not in state for key in required):
            raise ValueError("The IGAC undo state is incomplete.")
        if tuple(state["phi"].shape) != tuple(self.phi.shape):
            raise ValueError("The IGAC undo state does not match this workspace.")
        with self.torch.inference_mode():
            self.phi.copy_(state["phi"])
            self.add_constraint.copy_(state["add_constraint"])
            self.barrier_constraint.copy_(state["barrier_constraint"])
        self.iteration = max(0, int(state["iteration"]))
        self.last_step_changed_voxels = 0
        self.last_step_changed_ratio = 0.0
        self.last_step_observed_voxels = self.roi_voxels
        self.evolution_slices_zyx = state.get("evolution_slices_zyx")
        self.last_edit_slices_zyx = None

    def apply_brush(
        self,
        mode: str,
        orientation: str,
        slice_index: int,
        u: float,
        v: float,
        radius_mm: float,
    ) -> bool:
        return self.apply_brush_segment(
            mode,
            orientation,
            slice_index,
            u,
            v,
            u,
            v,
            radius_mm,
        )

    def apply_brush_segment(
        self,
        mode: str,
        orientation: str,
        slice_index: int,
        start_u: float,
        start_v: float,
        end_u: float,
        end_v: float,
        radius_mm: float,
    ) -> bool:
        start_xyz = display_to_xyz(orientation, slice_index, start_u, start_v)
        end_xyz = display_to_xyz(orientation, slice_index, end_u, end_v)
        local_start_xyz = tuple(start_xyz[axis] - self.roi_offset_xyz[axis] for axis in range(3))
        local_end_xyz = tuple(end_xyz[axis] - self.roi_offset_xyz[axis] for axis in range(3))
        start_zyx = tuple(reversed(local_start_xyz))
        end_zyx = tuple(reversed(local_end_xyz))
        radius = max(0.5, float(radius_mm))
        axes = []
        for start, end, spacing, size in zip(
            start_zyx,
            end_zyx,
            self.spacing_zyx,
            reversed(self.roi_shape_xyz),
        ):
            extent = int(math.ceil(radius / spacing))
            lower = max(0, int(math.floor(min(start, end))) - extent)
            upper = min(int(size), int(math.floor(max(start, end))) + extent + 1)
            axes.append((lower, upper, spacing))
        if any(upper <= lower for lower, upper, _spacing in axes):
            return False
        grids = self.torch.meshgrid(
            *[
                self.torch.arange(lower, upper, device=self.device, dtype=self.image.dtype)
                for lower, upper, _spacing in axes
            ],
            indexing="ij",
        )
        start_physical = self.torch.tensor(
            [float(value) * float(spacing) for value, spacing in zip(start_zyx, self.spacing_zyx)],
            device=self.device,
            dtype=self.image.dtype,
        )
        end_physical = self.torch.tensor(
            [float(value) * float(spacing) for value, spacing in zip(end_zyx, self.spacing_zyx)],
            device=self.device,
            dtype=self.image.dtype,
        )
        delta = end_physical - start_physical
        length_sq = delta.square().sum()
        physical_grids = [
            grid * float(axis[2]) for grid, axis in zip(grids, axes)
        ]
        projection = self.torch.zeros_like(grids[0])
        for grid, start, component in zip(physical_grids, start_physical, delta):
            projection = projection + (grid - start) * component
        fraction = (projection / length_sq.clamp_min(1.0e-8)).clamp(0.0, 1.0)
        distance_sq = self.torch.zeros_like(grids[0])
        for grid, start, component in zip(physical_grids, start_physical, delta):
            closest = start + fraction * component
            distance_sq = distance_sq + (grid - closest).square()
        capsule = distance_sq <= radius * radius
        signed_capsule = self.torch.sqrt(distance_sq.clamp_min(0.0)) - radius
        slices = tuple(slice(lower, upper) for lower, upper, _spacing in axes)
        tensor_slice = (0, 0) + slices
        mode = str(mode).lower()
        with self.torch.inference_mode():
            if mode == "add":
                self.add_constraint[tensor_slice] |= capsule
                self.barrier_constraint[tensor_slice] &= ~capsule
                self.phi[tensor_slice] = self.torch.where(
                    capsule,
                    self.torch.minimum(self.phi[tensor_slice], signed_capsule),
                    self.phi[tensor_slice],
                )
            elif mode == "barrier":
                self.barrier_constraint[tensor_slice] |= capsule
                self.add_constraint[tensor_slice] &= ~capsule
                self.phi[tensor_slice] = self.torch.where(
                    capsule, self.torch.maximum(self.phi[tensor_slice], self.torch.full_like(self.phi[tensor_slice], 6.0)), self.phi[tensor_slice]
                )
            elif mode == "neutral":
                self.add_constraint[tensor_slice] &= ~capsule
                self.barrier_constraint[tensor_slice] &= ~capsule
                self.phi[tensor_slice] = self.torch.where(
                    capsule, self.torch.full_like(self.phi[tensor_slice], 2.0), self.phi[tensor_slice]
                )
            else:
                raise ValueError("Unknown IGAC brush mode: {}".format(mode))
        self.last_edit_slices_zyx = self._merge_slices(
            self.last_edit_slices_zyx,
            slices,  # type: ignore[arg-type]
        )
        return True

    def _sample_phi_zyx(self, phi: Any, point_zyx: Iterable[float]) -> float:
        indices = []
        for value, size in zip(point_zyx, reversed(self.roi_shape_xyz)):
            indices.append(max(0, min(int(size) - 1, int(round(float(value))))))
        with self.torch.inference_mode():
            return float(phi[(0, 0) + tuple(indices)].item())

    def _boundary_pull_anchors(
        self,
        *,
        base_phi: Any,
        start_zyx: tuple[float, float, float],
        end_zyx: tuple[float, float, float],
        anchor_radius_mm: float,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float], float] | None:
        start_mm = np.asarray(
            [value * spacing for value, spacing in zip(start_zyx, self.spacing_zyx)],
            dtype=np.float64,
        )
        end_mm = np.asarray(
            [value * spacing for value, spacing in zip(end_zyx, self.spacing_zyx)],
            dtype=np.float64,
        )
        direction = end_mm - start_mm
        distance = float(np.linalg.norm(direction))
        if not math.isfinite(distance) or distance < 1.0e-4:
            return None
        direction /= distance
        probe_mm = max(min(self.spacing_zyx) * 1.25, min(2.0, distance * 0.25))
        plus_start = tuple(
            (start_mm[axis] + direction[axis] * probe_mm) / self.spacing_zyx[axis]
            for axis in range(3)
        )
        minus_start = tuple(
            (start_mm[axis] - direction[axis] * probe_mm) / self.spacing_zyx[axis]
            for axis in range(3)
        )
        plus_inside = self._sample_phi_zyx(base_phi, plus_start) < 0.0
        minus_inside = self._sample_phi_zyx(base_phi, minus_start) < 0.0
        if plus_inside == minus_inside:
            # A boundary can run almost tangent to the drag direction. The end
            # point still receives the FFD displacement, but pinning an
            # arbitrary inside/outside direction would damage the contour.
            return None
        inside_sign = 1.0 if plus_inside else -1.0
        offset_mm = max(min(self.spacing_zyx) * 0.75, float(anchor_radius_mm) * 0.75)
        inside_mm = end_mm + direction * inside_sign * offset_mm
        outside_mm = end_mm - direction * inside_sign * offset_mm
        inside_zyx = tuple(
            inside_mm[axis] / self.spacing_zyx[axis] for axis in range(3)
        )
        outside_zyx = tuple(
            outside_mm[axis] / self.spacing_zyx[axis] for axis in range(3)
        )
        return inside_zyx, outside_zyx, max(0.5, float(anchor_radius_mm))

    def _apply_constraint_sphere_zyx(
        self,
        mode: str,
        center_zyx: tuple[float, float, float],
        radius_mm: float,
    ) -> tuple[slice, slice, slice] | None:
        axes = []
        for center, spacing, size in zip(
            center_zyx,
            self.spacing_zyx,
            reversed(self.roi_shape_xyz),
        ):
            extent = max(1, int(math.ceil(float(radius_mm) / float(spacing))))
            lower = max(0, int(math.floor(center)) - extent)
            upper = min(int(size), int(math.floor(center)) + extent + 1)
            axes.append((lower, upper, float(spacing)))
        if any(upper <= lower for lower, upper, _spacing in axes):
            return None
        grids = self.torch.meshgrid(
            *[
                self.torch.arange(lower, upper, device=self.device, dtype=self.image.dtype)
                for lower, upper, _spacing in axes
            ],
            indexing="ij",
        )
        distance_sq = self.torch.zeros_like(grids[0])
        for grid, center, (_lower, _upper, spacing) in zip(grids, center_zyx, axes):
            distance_sq = distance_sq + ((grid - float(center)) * spacing).square()
        sphere = distance_sq <= float(radius_mm) ** 2
        slices_zyx = tuple(
            slice(lower, upper) for lower, upper, _spacing in axes
        )
        tensor_slice = (0, 0) + slices_zyx
        if str(mode) == "add":
            self.add_constraint[tensor_slice] |= sphere
            self.barrier_constraint[tensor_slice] &= ~sphere
        elif str(mode) == "barrier":
            self.barrier_constraint[tensor_slice] |= sphere
            self.add_constraint[tensor_slice] &= ~sphere
        else:
            raise ValueError("Unknown boundary pull anchor mode: {}".format(mode))
        return slices_zyx  # type: ignore[return-value]

    def apply_boundary_pull(
        self,
        *,
        base_state: dict[str, Any],
        orientation: str,
        slice_index: int,
        start_u: float,
        start_v: float,
        end_u: float,
        end_v: float,
        influence_radius_mm: float,
        maximum_drag_ratio: float = 0.65,
        anchor_radius_mm: float = 0.8,
        incremental_preview: bool = False,
        previous_slices_zyx: tuple[slice, slice, slice] | None = None,
    ) -> dict[str, Any] | None:
        """Move one captured 2D contour point with a local 3D B-spline FFD.

        Every preview is derived from ``base_state`` (the mouse-down snapshot),
        matching a boundary-point drag rather than accumulating brush strokes.
        The GUI uses incremental patch restoration so mouse movement does not
        copy the entire 3D workspace on every event.
        """
        if not isinstance(base_state, dict) or "phi" not in base_state:
            raise ValueError("Boundary pull requires the mouse-down IGAC state.")
        if incremental_preview:
            if previous_slices_zyx is not None:
                previous = (slice(None), slice(None)) + previous_slices_zyx
                with self.torch.inference_mode():
                    self.phi[previous].copy_(base_state["phi"][previous])
                    self.add_constraint[previous].copy_(base_state["add_constraint"][previous])
                    self.barrier_constraint[previous].copy_(
                        base_state["barrier_constraint"][previous]
                    )
            self.iteration = max(0, int(base_state.get("iteration", self.iteration)))
        else:
            self.restore_state(base_state)
        start_xyz = display_to_xyz(orientation, slice_index, start_u, start_v)
        end_xyz = display_to_xyz(orientation, slice_index, end_u, end_v)
        local_start_xyz = tuple(
            start_xyz[axis] - self.roi_offset_xyz[axis] for axis in range(3)
        )
        local_end_xyz = tuple(
            end_xyz[axis] - self.roi_offset_xyz[axis] for axis in range(3)
        )
        start_zyx = tuple(reversed(local_start_xyz))
        end_zyx = tuple(reversed(local_end_xyz))
        result = warp_signed_distance(
            torch=self.torch,
            functional=self.functional,
            base_phi=base_state["phi"],
            spacing_zyx=self.spacing_zyx,
            start_zyx=start_zyx,
            end_zyx=end_zyx,
            influence_radius_mm=float(influence_radius_mm),
            maximum_drag_ratio=float(maximum_drag_ratio),
        )
        if result is None:
            return None
        tensor_slice = (slice(None), slice(None)) + result.slices_zyx
        affected_slices = result.slices_zyx
        with self.torch.inference_mode():
            self.phi[tensor_slice].copy_(result.values)
            anchors = self._boundary_pull_anchors(
                base_phi=base_state["phi"],
                start_zyx=start_zyx,
                end_zyx=result.applied_end_zyx,
                anchor_radius_mm=float(anchor_radius_mm),
            )
            if anchors is not None:
                inside, outside, radius = anchors
                affected_slices = self._merge_slices(
                    affected_slices,
                    self._apply_constraint_sphere_zyx("add", inside, radius),
                )
                affected_slices = self._merge_slices(
                    affected_slices,
                    self._apply_constraint_sphere_zyx("barrier", outside, radius),
                )
            self._apply_constraints(affected_slices)
        self.last_step_changed_voxels = 0
        self.last_step_changed_ratio = 0.0
        self.last_edit_slices_zyx = affected_slices
        applied_local_xyz = tuple(reversed(result.applied_end_zyx))
        applied_xyz = tuple(
            applied_local_xyz[axis] + self.roi_offset_xyz[axis] for axis in range(3)
        )
        if str(orientation).lower() == "axial":
            applied_u, applied_v = applied_xyz[0], applied_xyz[1]
        elif str(orientation).lower() == "coronal":
            applied_u, applied_v = applied_xyz[0], applied_xyz[2]
        else:
            applied_u, applied_v = applied_xyz[1], applied_xyz[2]
        return {
            "drag_distance_mm": float(result.drag_distance_mm),
            "applied_drag_distance_mm": float(result.applied_drag_distance_mm),
            "influence_radius_mm": float(result.influence_radius_mm),
            "drag_was_clamped": bool(result.drag_was_clamped),
            "applied_end_u": float(applied_u),
            "applied_end_v": float(applied_v),
            "slices_zyx": affected_slices,
        }

    def nearest_boundary_display(
        self,
        *,
        orientation: str,
        slice_index: int,
        target_u: float,
        target_v: float,
        maximum_distance_mm: float,
    ) -> tuple[float, float] | None:
        """Find the current contour point nearest a display-space target."""
        orientation = str(orientation).lower()
        x_offset, y_offset, z_offset = self.roi_offset_xyz
        if orientation == "axial":
            fixed = int(slice_index) - z_offset
            if fixed < 0 or fixed >= self.roi_shape_xyz[2]:
                return None
            plane = self.phi[0, 0, fixed, :, :]
            local_u, local_v = float(target_u) - x_offset, float(target_v) - y_offset
            spacing_u, spacing_v = self.spacing_xyz[0], self.spacing_xyz[1]
            offset_u, offset_v = x_offset, y_offset
        elif orientation == "coronal":
            fixed = int(slice_index) - y_offset
            if fixed < 0 or fixed >= self.roi_shape_xyz[1]:
                return None
            plane = self.phi[0, 0, :, fixed, :]
            local_u, local_v = float(target_u) - x_offset, float(target_v) - z_offset
            spacing_u, spacing_v = self.spacing_xyz[0], self.spacing_xyz[2]
            offset_u, offset_v = x_offset, z_offset
        elif orientation == "sagittal":
            fixed = int(slice_index) - x_offset
            if fixed < 0 or fixed >= self.roi_shape_xyz[0]:
                return None
            plane = self.phi[0, 0, :, :, fixed]
            local_u, local_v = float(target_u) - y_offset, float(target_v) - z_offset
            spacing_u, spacing_v = self.spacing_xyz[1], self.spacing_xyz[2]
            offset_u, offset_v = y_offset, z_offset
        else:
            raise ValueError("Unknown orientation: {}".format(orientation))

        radius = max(min(spacing_u, spacing_v), float(maximum_distance_mm))
        column_extent = max(1, int(math.ceil(radius / spacing_u)))
        row_extent = max(1, int(math.ceil(radius / spacing_v)))
        left = max(0, int(math.floor(local_u)) - column_extent)
        right = min(int(plane.shape[1]), int(math.floor(local_u)) + column_extent + 1)
        top = max(0, int(math.floor(local_v)) - row_extent)
        bottom = min(int(plane.shape[0]), int(math.floor(local_v)) + row_extent + 1)
        if right <= left or bottom <= top:
            return None

        with self.torch.inference_mode():
            inside = plane < 0
            inner = inside.clone()
            inner[1:, :] &= inside[:-1, :]
            inner[:-1, :] &= inside[1:, :]
            inner[:, 1:] &= inside[:, :-1]
            inner[:, :-1] &= inside[:, 1:]
            boundary = (inside & ~inner)[top:bottom, left:right]
            candidates = self.torch.nonzero(boundary, as_tuple=False)
            if int(candidates.shape[0]) == 0:
                return None
            candidate_v = candidates[:, 0].to(self.image.dtype) + float(top) + 0.5
            candidate_u = candidates[:, 1].to(self.image.dtype) + float(left) + 0.5
            distance_sq = (
                ((candidate_u - local_u) * spacing_u).square()
                + ((candidate_v - local_v) * spacing_v).square()
            )
            index = int(self.torch.argmin(distance_sq).item())
            if float(distance_sq[index].item()) > radius * radius:
                return None
            return (
                float(candidate_u[index].item()) + offset_u,
                float(candidate_v[index].item()) + offset_v,
            )

    def apply_contour_attraction(
        self,
        *,
        orientation: str,
        slice_index: int,
        target_u: float,
        target_v: float,
        influence_radius_mm: float,
        maximum_drag_ratio: float = 0.65,
        anchor_radius_mm: float = 0.0,
        capture_distance_mm: float = 12.0,
    ) -> dict[str, Any] | None:
        """Pull the nearest current contour section towards a pointer target."""
        start = self.nearest_boundary_display(
            orientation=orientation,
            slice_index=slice_index,
            target_u=target_u,
            target_v=target_v,
            maximum_distance_mm=capture_distance_mm,
        )
        if start is None:
            return None
        start_xyz = display_to_xyz(orientation, slice_index, start[0], start[1])
        end_xyz = display_to_xyz(orientation, slice_index, target_u, target_v)
        local_start_xyz = tuple(
            start_xyz[axis] - self.roi_offset_xyz[axis] for axis in range(3)
        )
        local_end_xyz = tuple(
            end_xyz[axis] - self.roi_offset_xyz[axis] for axis in range(3)
        )
        start_zyx = tuple(reversed(local_start_xyz))
        end_zyx = tuple(reversed(local_end_xyz))
        result = warp_signed_distance(
            torch=self.torch,
            functional=self.functional,
            base_phi=self.phi,
            spacing_zyx=self.spacing_zyx,
            start_zyx=start_zyx,
            end_zyx=end_zyx,
            influence_radius_mm=float(influence_radius_mm),
            maximum_drag_ratio=float(maximum_drag_ratio),
        )
        if result is None:
            return None
        anchors = None
        if float(anchor_radius_mm) > 0.0:
            anchors = self._boundary_pull_anchors(
                base_phi=self.phi,
                start_zyx=start_zyx,
                end_zyx=result.applied_end_zyx,
                anchor_radius_mm=float(anchor_radius_mm),
            )
        tensor_slice = (slice(None), slice(None)) + result.slices_zyx
        affected_slices = result.slices_zyx
        with self.torch.inference_mode():
            self.phi[tensor_slice].copy_(result.values)
            if anchors is not None:
                inside, outside, radius = anchors
                affected_slices = self._merge_slices(
                    affected_slices,
                    self._apply_constraint_sphere_zyx("add", inside, radius),
                )
                affected_slices = self._merge_slices(
                    affected_slices,
                    self._apply_constraint_sphere_zyx("barrier", outside, radius),
                )
            self._apply_constraints(affected_slices)
        self.last_edit_slices_zyx = self._merge_slices(
            self.last_edit_slices_zyx, affected_slices
        )
        applied_local_xyz = tuple(reversed(result.applied_end_zyx))
        applied_xyz = tuple(
            applied_local_xyz[axis] + self.roi_offset_xyz[axis] for axis in range(3)
        )
        if str(orientation).lower() == "axial":
            applied_u, applied_v = applied_xyz[0], applied_xyz[1]
        elif str(orientation).lower() == "coronal":
            applied_u, applied_v = applied_xyz[0], applied_xyz[2]
        else:
            applied_u, applied_v = applied_xyz[1], applied_xyz[2]
        return {
            "start_u": float(start[0]),
            "start_v": float(start[1]),
            "target_u": float(target_u),
            "target_v": float(target_v),
            "applied_end_u": float(applied_u),
            "applied_end_v": float(applied_v),
            "drag_distance_mm": float(result.drag_distance_mm),
            "applied_drag_distance_mm": float(result.applied_drag_distance_mm),
            "drag_was_clamped": bool(result.drag_was_clamped),
            "slices_zyx": affected_slices,
        }

    def _roi_mask_xyz(self) -> np.ndarray:
        with self.torch.inference_mode():
            mask_zyx = (self.phi[0, 0] < 0).to("cpu").numpy()
        return np.transpose(mask_zyx, (2, 1, 0)).astype(bool, copy=False)

    def result_xyz(self) -> np.ndarray:
        result = self.initial_mask_xyz.copy()
        result[self.roi] = self._roi_mask_xyz()
        return result

    def _roi_display_plane(self, tensor: Any, orientation: str, index: int) -> np.ndarray | None:
        x_slice, y_slice, z_slice = self.roi
        with self.torch.inference_mode():
            values = tensor[0, 0]
            if orientation == "axial":
                if index < z_slice.start or index >= z_slice.stop:
                    return None
                plane = values[index - z_slice.start, :, :]
            elif orientation == "coronal":
                if index < y_slice.start or index >= y_slice.stop:
                    return None
                plane = values[:, index - y_slice.start, :]
            elif orientation == "sagittal":
                if index < x_slice.start or index >= x_slice.stop:
                    return None
                plane = values[:, :, index - x_slice.start]
            else:
                raise ValueError("Unknown orientation: {}".format(orientation))
            return plane.to("cpu").numpy()

    def _compose_roi_plane(
        self,
        base: np.ndarray,
        roi_display: np.ndarray | None,
        orientation: str,
        index: int,
    ) -> np.ndarray:
        output = np.asarray(base).copy()
        if roi_display is None:
            return output
        x_slice, y_slice, z_slice = self.roi
        if orientation == "axial":
            output[y_slice, x_slice] = roi_display
        elif orientation == "coronal":
            output[z_slice, x_slice] = roi_display
        elif orientation == "sagittal":
            output[z_slice, y_slice] = roi_display
        else:
            raise ValueError("Unknown orientation: {}".format(orientation))
        return output

    def frame(
        self,
        orientation: str,
        index: int,
        *,
        include_image: bool = True,
    ) -> dict[str, np.ndarray]:
        _width, _height, depth = orientation_extent(self.shape_xyz, orientation)
        safe_index = max(0, min(depth - 1, int(index)))
        base_mask = plane_from_xyz(self.initial_mask_xyz, orientation, safe_index)
        mask_plane = self._roi_display_plane(self.phi < 0, orientation, safe_index)
        mask = self._compose_roi_plane(base_mask, mask_plane, orientation, safe_index)
        empty = np.zeros_like(base_mask, dtype=bool)
        add_plane = self._roi_display_plane(self.add_constraint, orientation, safe_index)
        barrier_plane = self._roi_display_plane(self.barrier_constraint, orientation, safe_index)
        add = self._compose_roi_plane(empty, add_plane, orientation, safe_index)
        barrier = self._compose_roi_plane(empty, barrier_plane, orientation, safe_index)
        frame = {"mask": mask, "add": add, "barrier": barrier}
        if include_image:
            raw_plane = np.asarray(
                plane_from_xyz(self.image_source_xyz, orientation, safe_index),
                dtype=np.float32,
            )
            image = np.clip(
                (raw_plane - self.display_low) / (self.display_high - self.display_low),
                0.0,
                1.0,
            )
            image[~np.isfinite(image)] = 0.0
            frame["image"] = image
            frame["raw_image"] = raw_plane
        return frame


def load_raw_inputs(request: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    shape = tuple(int(value) for value in request["shape"])
    image_dtype = np.dtype(str(request["image_dtype"]))
    image_path = Path(str(request["image_path"]))
    mask_path = Path(str(request["mask_path"]))
    expected_image = int(np.prod(shape)) * image_dtype.itemsize
    expected_mask = int(np.prod(shape))
    if image_path.stat().st_size != expected_image:
        raise RuntimeError("The IGAC image buffer size does not match the Mimics image shape.")
    if mask_path.stat().st_size != expected_mask:
        raise RuntimeError("The IGAC Mask buffer size does not match the Mimics image shape.")
    image = np.memmap(image_path, dtype=image_dtype, mode="r", shape=shape, order="C")
    mask = np.memmap(mask_path, dtype=np.uint8, mode="r", shape=shape, order="C")
    return image, mask
