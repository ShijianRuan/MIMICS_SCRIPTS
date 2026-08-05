"""Strict NIfTI pairing and no-resampling patch preparation."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset

from .runtime import CancelledError, cancellation_requested, write_json_atomic

ProgressCallback = Callable[[dict[str, Any]], None]
MAX_CACHED_FOREGROUND_CENTERS = 250_000
INITIAL_MASK_QUALITY_THRESHOLDS = (0.4, 0.7)
NNINTERACTIVE_INPUT_CONTRACT = {
    "schema_version": "nninteractive_input_contract.v1",
    "spatial_orientation": "canonical_ras",
    "source_grid_policy": "mimics_dicom_compatible_minimal_resample",
    "intensity_space": "source_physical_values",
    "dicom_rescale": "apply_rescale_slope_and_intercept",
    "normalization": "nonzero_spatial_bbox_zscore",
    "normalization_channel": 0,
    "standard_deviation_correction": 1,
    "orientation_policy": "reindex_to_canonical_ras_without_interpolation",
    "spacing_policy": "preserve_source_spacing_no_spacing_resample",
    "session_preprocessing": (
        "nonzero_bbox_zscore_then_prompt_centered_crop_resize_to_model_plan_patch"
    ),
    "training_patch_policy": "fixed_voxel_patch_without_spacing_resample",
    "geometry_validation": "shape_and_affine_must_match_before_training",
}


def _absolute_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(cases, list) or not cases:
        raise ValueError("Dataset manifest must contain a non-empty cases list.")

    seen: set[str] = set()
    resolved: list[dict[str, Any]] = []
    for index, row in enumerate(cases):
        if not isinstance(row, dict):
            raise ValueError("Manifest case {} is not an object.".format(index))
        case_id = str(row.get("case_id") or "").strip()
        if not case_id or case_id in seen:
            raise ValueError("Manifest case_id values must be non-empty and unique.")
        seen.add(case_id)
        split = str(row.get("split") or "").lower()
        if split not in {"", "train", "val"}:
            raise ValueError("{} has unsupported split {!r}.".format(case_id, split))
        image = _absolute_path(str(row.get("image") or ""), source.parent)
        label = _absolute_path(str(row.get("label") or ""), source.parent)
        initial_mask_value = str(row.get("initial_mask") or "").strip()
        initial_mask = (
            _absolute_path(initial_mask_value, source.parent)
            if initial_mask_value
            else None
        )
        if not image.is_file():
            raise FileNotFoundError(
                "Image is missing for {}: {}".format(case_id, image)
            )
        if not label.is_file():
            raise FileNotFoundError(
                "Label is missing for {}: {}".format(case_id, label)
            )
        if initial_mask is not None and not initial_mask.is_file():
            raise FileNotFoundError(
                "Initial Mask is missing for {}: {}".format(
                    case_id, initial_mask
                )
            )
        resolved.append(
            {
                "case_id": case_id,
                "image": str(image),
                "label": str(label),
                "initial_mask": str(initial_mask) if initial_mask else "",
                "split": split,
                **{
                    key: str(
                        _absolute_path(str(row[key]), source.parent)
                    )
                    for key in (
                        "source_image",
                        "source_label",
                        "source_mcs",
                        "source_initial_mask",
                        "source_initial_mcs",
                    )
                    if str(row.get(key) or "").strip()
                },
                "initial_mask_source_type": str(
                    row.get("initial_mask_source_type")
                    or ("provided_mask" if initial_mask else "none")
                ).strip(),
                "initial_mask_source_model": str(
                    row.get("initial_mask_source_model") or ""
                ).strip(),
                "initial_mask_source_name": str(
                    row.get("initial_mask_source_name") or ""
                ).strip(),
            }
        )
    return resolved


def split_cases(
    cases: list[dict[str, Any]], validation_fraction: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    explicit = any(row["split"] for row in cases)
    if explicit:
        train = [row for row in cases if row["split"] != "val"]
        val = [row for row in cases if row["split"] == "val"]
    else:
        ordered = list(cases)
        random.Random(seed).shuffle(ordered)
        n_val = int(round(len(ordered) * validation_fraction))
        if validation_fraction > 0 and len(ordered) >= 5:
            n_val = max(1, n_val)
        n_val = min(n_val, max(0, len(ordered) - 1))
        val = ordered[:n_val]
        train = ordered[n_val:]
    if not train:
        raise ValueError("The dataset split contains no training cases.")
    return train, val


def _canonical_volume(path: str) -> tuple[np.ndarray, np.ndarray, tuple[float, ...]]:
    image = nib.load(path)
    qform, qform_code = image.get_qform(coded=True)
    sform, sform_code = image.get_sform(coded=True)
    if (
        int(qform_code or 0) > 0
        and int(sform_code or 0) > 0
        and not np.allclose(qform, sform, rtol=1e-5, atol=1e-4)
    ):
        raise ValueError(
            "{} has conflicting valid qform and sform transforms. "
            "Resolve the NIfTI geometry before fine-tuning.".format(path)
        )
    canonical = nib.as_closest_canonical(image)
    array = np.asarray(canonical.dataobj)
    if array.ndim == 4 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 3:
        raise ValueError(
            "{} must contain one 3D volume; got {}.".format(path, array.shape)
        )
    spacing = tuple(float(value) for value in canonical.header.get_zooms()[:3])
    return array, np.asarray(canonical.affine, dtype=np.float64), spacing


def normalize_like_nninteractive(image: np.ndarray) -> np.ndarray:
    """Match nnInteractive's full-image z-score using its nonzero bounding crop."""
    output = np.asarray(image, dtype=np.float32)
    if not np.isfinite(output).all():
        raise ValueError("Image contains NaN or infinite values.")
    nonzero = np.argwhere(output != 0)
    if nonzero.size == 0:
        raise ValueError("Image is entirely zero.")
    low = nonzero.min(axis=0)
    high = nonzero.max(axis=0) + 1
    crop = output[tuple(slice(int(a), int(b)) for a, b in zip(low, high))]
    mean = float(crop.mean(dtype=np.float64))
    std = float(crop.std(dtype=np.float64, ddof=1 if crop.size > 1 else 0))
    if not np.isfinite(std) or std < 1e-8:
        raise ValueError("Image has zero variance inside its nonzero region.")
    output = np.ascontiguousarray((output - mean) / std, dtype=np.float32)
    return output


def nninteractive_input_contract() -> dict[str, Any]:
    """Return the serialized training/deployment input contract."""
    return dict(NNINTERACTIVE_INPUT_CONTRACT)


def _binary_label(label: np.ndarray, values: Iterable[int]) -> np.ndarray:
    integer_values = [int(value) for value in values]
    result = np.isin(label, integer_values)
    return np.ascontiguousarray(result, dtype=np.uint8)


def initial_mask_quality_bin(dice: float) -> str:
    """Return a stable, human-readable Initial Mask quality stratum."""
    low, high = INITIAL_MASK_QUALITY_THRESHOLDS
    value = float(dice)
    if value < low:
        return "low"
    if value < high:
        return "medium"
    return "high"


def initial_mask_quality_metadata(
    initial_mask: np.ndarray,
    target: np.ndarray,
    *,
    source_type: str = "provided_mask",
    source_model: str = "",
    source_name: str = "",
) -> dict[str, Any]:
    """Describe a real draft relative to its final annotation."""
    initial = np.asarray(initial_mask, dtype=bool)
    final = np.asarray(target, dtype=bool)
    initial_voxels = int(initial.sum())
    target_voxels = int(final.sum())
    true_positive = int(np.count_nonzero(initial & final))
    denominator = initial_voxels + target_voxels
    dice = 1.0 if denominator == 0 else 2.0 * true_positive / denominator
    precision = true_positive / initial_voxels if initial_voxels else 0.0
    recall = true_positive / target_voxels if target_voxels else 1.0
    return {
        "initial_mask_source_type": str(source_type or "provided_mask"),
        "initial_mask_source_model": str(source_model or ""),
        "initial_mask_source_name": str(source_name or ""),
        "initial_mask_baseline_dice": float(dice),
        "initial_mask_precision": float(precision),
        "initial_mask_recall": float(recall),
        "initial_mask_volume_ratio": (
            float(initial_voxels / target_voxels) if target_voxels else None
        ),
        "initial_mask_quality_bin": initial_mask_quality_bin(dice),
    }


def _case_fingerprint(row: dict[str, Any], label_values: Iterable[int]) -> str:
    digest = hashlib.sha256()

    def first_existing(*values: Any) -> Path:
        candidates = [
            Path(str(value))
            for value in values
            if str(value or "").strip()
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        if candidates:
            return candidates[-1]
        raise FileNotFoundError("No source path is available for fingerprinting.")

    sources = (
        (
            "image",
            first_existing(row.get("source_image"), row.get("image")),
        ),
        (
            "label",
            first_existing(
                row.get("source_mcs"),
                row.get("source_label"),
                row.get("label"),
            ),
        ),
    )
    for key, value in sources:
        path = Path(value)
        stat = path.stat()
        digest.update(key.encode("ascii"))
        digest.update(path.name.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    initial_mask = str(row.get("initial_mask") or "").strip()
    if initial_mask:
        path = first_existing(
            row.get("source_initial_mcs"),
            row.get("source_initial_mask"),
            initial_mask,
        )
        stat = path.stat()
        digest.update(b"initial_mask")
        digest.update(path.name.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        digest.update(
            json.dumps(
                {
                    "source_type": row.get("initial_mask_source_type") or "",
                    "source_model": row.get("initial_mask_source_model") or "",
                    "source_name": row.get("initial_mask_source_name") or "",
                },
                sort_keys=True,
            ).encode("utf-8")
        )
    digest.update(json.dumps(list(label_values), sort_keys=True).encode("utf-8"))
    digest.update(
        b"canonical_ras+nninteractive_nonzero_bbox_zscore+initial_mask+quality+fg_centers.v6"
    )
    return digest.hexdigest()


def _safe_case_name(case_id: str) -> str:
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in case_id)
    return safe[:100] or hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:16]


def _save_npy_atomic(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(
        "{}.{}.{}.tmp".format(path.name, os.getpid(), uuid.uuid4().hex)
    )
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        last_error = None
        for attempt in range(20):
            try:
                os.replace(str(temporary), str(path))
                return
            except OSError as exc:
                last_error = exc
                time.sleep(min(0.25, 0.02 * (attempt + 1)))
        raise OSError(
            "Could not publish prepared array {}: {}".format(path, last_error)
        )
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass


def _foreground_centers(label: np.ndarray) -> np.ndarray:
    flat = np.flatnonzero(label)
    if flat.size > MAX_CACHED_FOREGROUND_CENTERS:
        selected = np.linspace(
            0, flat.size - 1, MAX_CACHED_FOREGROUND_CENTERS, dtype=np.int64
        )
        flat = flat[selected]
    coordinates = np.column_stack(np.unravel_index(flat, label.shape))
    return np.ascontiguousarray(coordinates, dtype=np.int32)


def prepare_cases(
    cases: list[dict[str, Any]],
    cache_dir: str | Path,
    label_values: Iterable[int],
    progress: ProgressCallback | None = None,
    cancel_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Materialize canonical arrays once so compressed NIfTI is not reread per step."""
    destination = Path(cache_dir)
    destination.mkdir(parents=True, exist_ok=True)
    prepared: list[dict[str, Any]] = []
    values = [int(value) for value in label_values]
    reused_cases = 0

    for index, row in enumerate(cases, start=1):
        if cancellation_requested(cancel_path):
            raise CancelledError("Cancelled while preparing training data.")
        case_dir = destination / _safe_case_name(row["case_id"])
        metadata_path = case_dir / "metadata.json"
        image_path = case_dir / "image.npy"
        label_path = case_dir / "label.npy"
        initial_mask_path = case_dir / "initial_mask.npy"
        foreground_path = case_dir / "foreground_centers.npy"
        fingerprint = _case_fingerprint(row, values)

        metadata: dict[str, Any] = {}
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except (OSError, ValueError):
            metadata = {}
        reusable = (
            metadata.get("fingerprint") == fingerprint
            and image_path.is_file()
            and label_path.is_file()
            and foreground_path.is_file()
            and (
                not row.get("initial_mask")
                or initial_mask_path.is_file()
                or bool(metadata.get("initial_mask_rejected_reason"))
            )
        )
        if not reusable:
            case_dir.mkdir(parents=True, exist_ok=True)
            image, image_affine, spacing = _canonical_volume(row["image"])
            label, label_affine, _ = _canonical_volume(row["label"])
            if image.shape != label.shape:
                raise ValueError(
                    "{} image/label shapes differ after canonical orientation: {} vs {}. "
                    "Export or resample the label to the image grid before fine-tuning.".format(
                        row["case_id"], image.shape, label.shape
                    )
                )
            affine_close = np.allclose(image_affine, label_affine, rtol=1e-5, atol=1e-4)
            if not affine_close:
                raise ValueError(
                    "{} image/label affines differ after canonical orientation. "
                    "The trainer will not hide this with an implicit label resample.".format(
                        row["case_id"]
                    )
                )
            image = normalize_like_nninteractive(image)
            label = _binary_label(label, values)
            if not label.any():
                raise ValueError(
                    "{} has no voxels matching label_values={}.".format(
                        row["case_id"], values
                    )
                )
            foreground_centers = _foreground_centers(label)
            _save_npy_atomic(image_path, image)
            _save_npy_atomic(label_path, label)
            has_initial_mask = bool(row.get("initial_mask"))
            initial_mask_rejected_reason = ""
            initial_quality: dict[str, Any] = {}
            if has_initial_mask:
                initial_mask, initial_affine, _ = _canonical_volume(
                    row["initial_mask"]
                )
                if initial_mask.shape != image.shape or not np.allclose(
                    initial_affine,
                    image_affine,
                    rtol=1e-5,
                    atol=1e-4,
                ):
                    raise ValueError(
                        "{} initial Mask is not on the image grid after "
                        "canonical orientation.".format(row["case_id"])
                    )
                initial_binary = np.ascontiguousarray(
                    initial_mask != 0, dtype=np.uint8
                )
                if not initial_binary.any():
                    has_initial_mask = False
                    initial_mask_rejected_reason = "empty"
                elif np.array_equal(initial_binary, label):
                    has_initial_mask = False
                    initial_mask_rejected_reason = "identical_to_target"
                else:
                    _save_npy_atomic(initial_mask_path, initial_binary)
                    initial_quality = initial_mask_quality_metadata(
                        initial_binary,
                        label,
                        source_type=str(
                            row.get("initial_mask_source_type")
                            or "provided_mask"
                        ),
                        source_model=str(
                            row.get("initial_mask_source_model") or ""
                        ),
                        source_name=str(
                            row.get("initial_mask_source_name") or ""
                        ),
                    )
            if not has_initial_mask:
                try:
                    initial_mask_path.unlink()
                except OSError:
                    pass
            _save_npy_atomic(foreground_path, foreground_centers)
            metadata = {
                "schema_version": "nninteractive_prepared_case.v1",
                "case_id": row["case_id"],
                "fingerprint": fingerprint,
                "shape": list(image.shape),
                "canonical_affine": image_affine.tolist(),
                "spacing": list(spacing),
                "foreground_voxels": int(label.sum()),
                "cached_foreground_centers": int(len(foreground_centers)),
                "source_image": row["image"],
                "source_label": row["label"],
                "source_initial_mask": row.get("initial_mask") or "",
                "has_initial_mask": has_initial_mask,
                "initial_mask_rejected_reason": (
                    initial_mask_rejected_reason
                ),
                **initial_quality,
                "affine_close": bool(affine_close),
            }
            write_json_atomic(metadata_path, metadata)
        else:
            reused_cases += 1

        prepared.append(
            {
                **row,
                "prepared_image": str(image_path),
                "prepared_label": str(label_path),
                "prepared_initial_mask": (
                    str(initial_mask_path)
                    if bool(metadata.get("has_initial_mask"))
                    else ""
                ),
                "prepared_foreground_centers": str(foreground_path),
                "metadata": metadata,
            }
        )
        if progress:
            progress(
                {
                    "phase": "preparing_data",
                    "completed_cases": index,
                    "total_cases": len(cases),
                    "case_id": row["case_id"],
                    "reused": reusable,
                    "prepared_cache_reused": reused_cases,
                    "prepared_cache_total": len(cases),
                    "phase": (
                        "reusing_prepared_cases"
                        if reusable
                        else "preparing_data"
                    ),
                }
            )
    return prepared


def remove_prepared_cache(path: str | Path) -> None:
    shutil.rmtree(str(path), ignore_errors=True)


def _extract_patch(
    array: np.ndarray, center: np.ndarray, patch_size: tuple[int, int, int]
) -> np.ndarray:
    starts = center.astype(np.int64) - np.asarray(patch_size, dtype=np.int64) // 2
    stops = starts + np.asarray(patch_size, dtype=np.int64)
    src_slices = []
    dst_slices = []
    for start, stop, size in zip(starts, stops, array.shape):
        src_start = max(0, int(start))
        src_stop = min(int(size), int(stop))
        dst_start = src_start - int(start)
        dst_stop = dst_start + (src_stop - src_start)
        src_slices.append(slice(src_start, src_stop))
        dst_slices.append(slice(dst_start, dst_stop))
    output = np.zeros(patch_size, dtype=array.dtype)
    output[tuple(dst_slices)] = array[tuple(src_slices)]
    return output


def _range(config: dict[str, Any], key: str, default: tuple[float, float]):
    values = config.get(key, default)
    return float(values[0]), float(values[1])


def _sample_spatial_matrix(
    config: dict[str, Any],
) -> np.ndarray | None:
    rotate = np.random.random() < float(
        config.get("rotation_probability", 0.2)
    )
    scale = np.random.random() < float(
        config.get("scaling_probability", 0.2)
    )
    if not rotate and not scale:
        return None
    affine = np.eye(3, dtype=np.float64)
    if rotate:
        low, high = _range(config, "rotation_degrees", (-30.0, 30.0))
        ax, ay, az = np.deg2rad(np.random.uniform(low, high, size=3))
        sx, cx = np.sin(ax), np.cos(ax)
        sy, cy = np.sin(ay), np.cos(ay)
        sz, cz = np.sin(az), np.cos(az)
        rotate_x = np.asarray(
            [[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64
        )
        rotate_y = np.asarray(
            [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64
        )
        rotate_z = np.asarray(
            [[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64
        )
        affine = rotate_z @ rotate_y @ rotate_x
    if scale:
        low, high = _range(config, "scaling_range", (0.7, 1.4))
        affine *= float(np.random.uniform(low, high))
    # batchgeneratorsv2 right-multiplies row-vector output coordinates by
    # this affine before sampling the input. scipy.ndimage uses column
    # vectors, hence the transpose. Do not invert it: in the official
    # transform a scale above one makes the sampled object smaller.
    return affine.T


def _sample_spatial_patch(
    array: np.ndarray,
    center: np.ndarray,
    patch_size: tuple[int, int, int],
    matrix: np.ndarray | None,
    *,
    order: int,
) -> np.ndarray:
    if matrix is None:
        return _extract_patch(array, center, patch_size)
    output_center = (np.asarray(patch_size, dtype=np.float64) - 1.0) / 2.0
    starts = (
        np.asarray(center, dtype=np.float64)
        - np.asarray(patch_size, dtype=np.int64) // 2
    )
    sampling_center = starts + output_center
    offset = sampling_center - matrix @ output_center
    return ndimage.affine_transform(
        np.asarray(array),
        matrix,
        offset=offset,
        output_shape=patch_size,
        order=int(order),
        mode="constant",
        cval=0.0,
        prefilter=bool(order > 1),
    )


def _simulate_low_resolution(
    image: np.ndarray, scale_range: tuple[float, float]
) -> np.ndarray:
    scale = float(np.random.uniform(*scale_range))
    source_shape = np.asarray(image.shape, dtype=np.int64)
    low_shape = np.maximum(1, np.rint(source_shape * scale).astype(np.int64))
    tensor = torch.from_numpy(np.ascontiguousarray(image))[None, None].float()
    low = torch.nn.functional.interpolate(
        tensor,
        size=tuple(int(value) for value in low_shape),
        mode="nearest-exact",
    )
    restored = torch.nn.functional.interpolate(
        low,
        size=tuple(int(value) for value in source_shape),
        mode="trilinear",
        align_corners=False,
    )
    return np.ascontiguousarray(restored[0, 0].numpy(), dtype=np.float32)


def _nnunet_gaussian_blur(
    image: np.ndarray, sigmas: tuple[float, float, float]
) -> np.ndarray:
    """Match batchgeneratorsv2's separable reflected Gaussian blur."""
    value = torch.from_numpy(np.ascontiguousarray(image))[None].float()
    for axis, sigma in enumerate(sigmas):
        sigma = float(sigma)
        requested = sigma * 6.0 + 0.5
        kernel_size = round(requested)
        if kernel_size % 2 == 0:
            kernel_size += 1 if requested - kernel_size >= 0 else -1
        kernel_size = max(1, int(kernel_size))
        half = (kernel_size - 1) * 0.5
        positions = torch.linspace(-half, half, steps=kernel_size)
        kernel = torch.exp(-0.5 * (positions / sigma).pow(2))
        kernel /= kernel.sum()
        pad = [0, 0, 0, 0, 0, 0]
        pad_index = {0: 4, 1: 2, 2: 0}[axis]
        pad[pad_index] = kernel_size // 2
        pad[pad_index + 1] = kernel_size // 2
        padded = torch.nn.functional.pad(value[None], pad, mode="reflect")
        shape = [1, 1, 1, 1, 1]
        shape[axis + 2] = kernel_size
        weight = kernel.reshape(shape)
        value = torch.nn.functional.conv3d(padded, weight)[0]
    return np.ascontiguousarray(value[0].numpy(), dtype=np.float32)


def _gamma_transform(
    image: np.ndarray, gamma_range: tuple[float, float], invert: bool
) -> np.ndarray:
    minimum = float(image.min())
    maximum = float(image.max())
    value_range = maximum - minimum
    if value_range < 1e-8:
        return image
    mean = float(image.mean(dtype=np.float64))
    std = float(image.std(dtype=np.float64, ddof=1))
    normalized = (image - minimum) / value_range
    if invert:
        normalized = 1.0 - normalized
    normalized = np.power(
        np.clip(normalized, 0.0, 1.0),
        float(np.random.uniform(*gamma_range)),
    )
    if invert:
        normalized = 1.0 - normalized
    transformed = normalized * value_range + minimum
    transformed_std = float(transformed.std(dtype=np.float64, ddof=1))
    if transformed_std > 1e-8:
        transformed = (transformed - float(transformed.mean())) / transformed_std
        transformed = transformed * std + mean
    return np.asarray(transformed, dtype=np.float32)


def _augment_intensity_like_nnunet(
    image: np.ndarray, config: dict[str, Any]
) -> np.ndarray:
    output = np.asarray(image, dtype=np.float32).copy()
    if np.random.random() < float(config.get("noise_probability", 0.1)):
        # batchgeneratorsv2 retains the historical ``noise_variance`` name,
        # but passes the sampled value directly to Normal(..., std=sigma).
        noise_sigma = np.random.uniform(
            *_range(config, "noise_variance_range", (0.0, 0.1))
        )
        if noise_sigma > 0:
            output += np.random.normal(
                0.0, noise_sigma, size=output.shape
            ).astype(np.float32)
    if (
        np.random.random() < float(config.get("blur_probability", 0.2))
        and np.random.random()
        < float(config.get("blur_channel_probability", 0.5))
    ):
        low, high = _range(config, "blur_sigma_range", (0.5, 1.0))
        output = _nnunet_gaussian_blur(
            output,
            tuple(float(value) for value in np.random.uniform(low, high, 3)),
        )
    if np.random.random() < float(config.get("brightness_probability", 0.15)):
        output *= float(
            np.random.uniform(
                *_range(
                    config, "brightness_multiplier_range", (0.75, 1.25)
                )
            )
        )
    if np.random.random() < float(config.get("contrast_probability", 0.15)):
        original_min = float(output.min())
        original_max = float(output.max())
        mean = float(output.mean(dtype=np.float64))
        factor = float(
            np.random.uniform(*_range(config, "contrast_range", (0.75, 1.25)))
        )
        output = np.clip(
            (output - mean) * factor + mean,
            original_min,
            original_max,
        )
    if (
        np.random.random()
        < float(config.get("low_resolution_probability", 0.25))
        and np.random.random()
        < float(config.get("low_resolution_channel_probability", 0.5))
    ):
        output = _simulate_low_resolution(
            output,
            _range(config, "low_resolution_scale_range", (0.5, 1.0)),
        )
    gamma_range = _range(config, "gamma_range", (0.7, 1.5))
    if np.random.random() < float(config.get("gamma_invert_probability", 0.1)):
        output = _gamma_transform(output, gamma_range, invert=True)
    if np.random.random() < float(config.get("gamma_probability", 0.3)):
        output = _gamma_transform(output, gamma_range, invert=False)
    return np.ascontiguousarray(output, dtype=np.float32)


def _augment_legacy(
    image: np.ndarray,
    label: np.ndarray,
    initial_mask: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flip_probability = float(config.get("flip_probability", 0.5))
    for axis in range(3):
        if np.random.random() < flip_probability:
            image = np.flip(image, axis=axis)
            label = np.flip(label, axis=axis)
            initial_mask = np.flip(initial_mask, axis=axis)
    scale_low, scale_high = config.get("intensity_scale_range", [1.0, 1.0])
    shift_low, shift_high = config.get("intensity_shift_range", [0.0, 0.0])
    noise_low, noise_high = config.get("noise_std_range", [0.0, 0.0])
    image = image.astype(np.float32, copy=True)
    image *= np.random.uniform(float(scale_low), float(scale_high))
    image += np.random.uniform(float(shift_low), float(shift_high))
    noise_std = np.random.uniform(float(noise_low), float(noise_high))
    if noise_std > 0:
        image += np.random.normal(0.0, noise_std, size=image.shape).astype(
            np.float32
        )
    return image, label, initial_mask


class InteractivePatchDataset(Dataset):
    """Random fixed-size patches from prepared image/label arrays."""

    def __init__(
        self,
        cases: list[dict[str, Any]],
        patch_size: Iterable[int],
        foreground_probability: float,
        virtual_length: int,
        augmentation: dict[str, Any] | None = None,
    ) -> None:
        self.cases = list(cases)
        self.patch_size = tuple(int(value) for value in patch_size)
        self.foreground_probability = float(foreground_probability)
        self.virtual_length = max(int(virtual_length), len(self.cases))
        self.augmentation = dict(augmentation or {})
        self._array_cache: dict[
            str, tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]
        ] = {}

    def __len__(self) -> int:
        return self.virtual_length

    def _case_arrays(
        self, row: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
        case_id = str(row["case_id"])
        cached = self._array_cache.get(case_id)
        if cached is not None:
            return cached
        image = np.load(row["prepared_image"], mmap_mode="r")
        label = np.load(row["prepared_label"], mmap_mode="r")
        initial_mask_path = str(row.get("prepared_initial_mask") or "")
        initial_mask = (
            np.load(initial_mask_path, mmap_mode="r")
            if initial_mask_path
            else None
        )
        foreground_path = row.get("prepared_foreground_centers")
        if foreground_path:
            foreground = np.load(foreground_path, mmap_mode="r")
        else:
            foreground = _foreground_centers(label)
        cached = (image, label, initial_mask, foreground)
        self._array_cache[case_id] = cached
        return cached

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.cases[index % len(self.cases)]
        image, label, initial_mask, foreground = self._case_arrays(row)
        if foreground.size and np.random.random() < self.foreground_probability:
            center = foreground[np.random.randint(len(foreground))]
        else:
            center = np.asarray(
                [np.random.randint(max(1, size)) for size in image.shape],
                dtype=np.int64,
            )
        augmentation_enabled = bool(self.augmentation.get("enabled", False))
        profile = str(self.augmentation.get("profile") or "legacy").lower()
        if augmentation_enabled and profile == "nninteractive_nnunet":
            matrix = _sample_spatial_matrix(self.augmentation)
            image_patch = _sample_spatial_patch(
                image, center, self.patch_size, matrix, order=1
            )
            label_patch = _sample_spatial_patch(
                label.astype(np.float32, copy=False),
                center,
                self.patch_size,
                matrix,
                order=1,
            ) > 0.5
            initial_mask_patch = (
                _sample_spatial_patch(
                    initial_mask.astype(np.float32, copy=False),
                    center,
                    self.patch_size,
                    matrix,
                    order=1,
                ) > 0.5
                if initial_mask is not None
                else np.zeros(self.patch_size, dtype=np.uint8)
            )
            image_patch = _augment_intensity_like_nnunet(
                image_patch, self.augmentation
            )
            mirror_probability = float(
                self.augmentation.get("flip_probability", 0.5)
            )
            for axis in self.augmentation.get("mirror_axes", [0, 1, 2]):
                if np.random.random() < mirror_probability:
                    image_patch = np.flip(image_patch, axis=int(axis))
                    label_patch = np.flip(label_patch, axis=int(axis))
                    initial_mask_patch = np.flip(
                        initial_mask_patch, axis=int(axis)
                    )
        else:
            image_patch = _extract_patch(image, center, self.patch_size)
            label_patch = _extract_patch(label, center, self.patch_size)
            initial_mask_patch = (
                _extract_patch(initial_mask, center, self.patch_size)
                if initial_mask is not None
                else np.zeros(self.patch_size, dtype=np.uint8)
            )
            if augmentation_enabled:
                image_patch, label_patch, initial_mask_patch = _augment_legacy(
                    image_patch,
                    label_patch,
                    initial_mask_patch,
                    self.augmentation,
                )
        metadata = row.get("metadata") or {}
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image_patch))[None],
            "target": torch.from_numpy(np.ascontiguousarray(label_patch)).long(),
            "initial_mask": torch.from_numpy(
                np.ascontiguousarray(initial_mask_patch)
            ).long(),
            "has_initial_mask": bool(initial_mask is not None),
            "case_id": row["case_id"],
            "initial_mask_source_type": str(
                metadata.get("initial_mask_source_type") or "none"
            ),
            "initial_mask_source_model": str(
                metadata.get("initial_mask_source_model") or ""
            ),
            "initial_mask_quality_bin": str(
                metadata.get("initial_mask_quality_bin") or "unknown"
            ),
            "initial_mask_baseline_dice": float(
                metadata.get("initial_mask_baseline_dice") or 0.0
            ),
        }
