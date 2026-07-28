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
from torch.utils.data import Dataset

from .runtime import CancelledError, cancellation_requested, write_json_atomic

ProgressCallback = Callable[[dict[str, Any]], None]
MAX_CACHED_FOREGROUND_CENTERS = 250_000
NNINTERACTIVE_INPUT_CONTRACT = {
    "schema_version": "nninteractive_input_contract.v1",
    "spatial_orientation": "canonical_ras",
    "source_grid_policy": "mimics_dicom_compatible_minimal_resample",
    "intensity_space": "source_physical_values",
    "dicom_rescale": "apply_rescale_slope_and_intercept",
    "normalization": "nonzero_spatial_bbox_zscore",
    "normalization_channel": 0,
    "standard_deviation_correction": 1,
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
        if not image.is_file():
            raise FileNotFoundError(
                "Image is missing for {}: {}".format(case_id, image)
            )
        if not label.is_file():
            raise FileNotFoundError(
                "Label is missing for {}: {}".format(case_id, label)
            )
        resolved.append(
            {
                "case_id": case_id,
                "image": str(image),
                "label": str(label),
                "split": split,
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
        # qform/sform conflict: prefer sform (NIfTI standard recommendation)
        # and re-encode so as_closest_canonical uses a consistent transform.
        image.set_qform(sform, code=sform_code)
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


def _case_fingerprint(row: dict[str, Any], label_values: Iterable[int]) -> str:
    digest = hashlib.sha256()
    for key in ("image", "label"):
        path = Path(row[key])
        stat = path.stat()
        digest.update(str(path).encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    digest.update(json.dumps(list(label_values), sort_keys=True).encode("utf-8"))
    digest.update(b"canonical_ras+nninteractive_nonzero_bbox_zscore+fg_centers.v2")
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

    for index, row in enumerate(cases, start=1):
        if cancellation_requested(cancel_path):
            raise CancelledError("Cancelled while preparing training data.")
        case_dir = destination / _safe_case_name(row["case_id"])
        metadata_path = case_dir / "metadata.json"
        image_path = case_dir / "image.npy"
        label_path = case_dir / "label.npy"
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
                "affine_close": bool(affine_close),
            }
            write_json_atomic(metadata_path, metadata)

        prepared.append(
            {
                **row,
                "prepared_image": str(image_path),
                "prepared_label": str(label_path),
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
        self._array_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return self.virtual_length

    def _case_arrays(
        self, row: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        case_id = str(row["case_id"])
        cached = self._array_cache.get(case_id)
        if cached is not None:
            return cached
        image = np.load(row["prepared_image"], mmap_mode="r")
        label = np.load(row["prepared_label"], mmap_mode="r")
        foreground_path = row.get("prepared_foreground_centers")
        if foreground_path:
            foreground = np.load(foreground_path, mmap_mode="r")
        else:
            foreground = _foreground_centers(label)
        cached = (image, label, foreground)
        self._array_cache[case_id] = cached
        return cached

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.cases[index % len(self.cases)]
        image, label, foreground = self._case_arrays(row)
        if foreground.size and np.random.random() < self.foreground_probability:
            center = foreground[np.random.randint(len(foreground))]
        else:
            center = np.asarray(
                [np.random.randint(max(1, size)) for size in image.shape],
                dtype=np.int64,
            )
        image_patch = _extract_patch(image, center, self.patch_size)
        label_patch = _extract_patch(label, center, self.patch_size)
        if bool(self.augmentation.get("enabled", False)):
            flip_probability = float(self.augmentation.get("flip_probability", 0.5))
            for axis in range(3):
                if np.random.random() < flip_probability:
                    image_patch = np.flip(image_patch, axis=axis)
                    label_patch = np.flip(label_patch, axis=axis)
            scale_low, scale_high = self.augmentation.get(
                "intensity_scale_range", [1.0, 1.0]
            )
            shift_low, shift_high = self.augmentation.get(
                "intensity_shift_range", [0.0, 0.0]
            )
            noise_low, noise_high = self.augmentation.get("noise_std_range", [0.0, 0.0])
            scale = np.random.uniform(float(scale_low), float(scale_high))
            shift = np.random.uniform(float(shift_low), float(shift_high))
            noise_std = np.random.uniform(float(noise_low), float(noise_high))
            image_patch = image_patch.astype(np.float32, copy=True)
            image_patch *= scale
            image_patch += shift
            if noise_std > 0:
                image_patch += np.random.normal(
                    0.0, noise_std, size=image_patch.shape
                ).astype(np.float32)
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image_patch))[None],
            "target": torch.from_numpy(np.ascontiguousarray(label_patch)).long(),
            "case_id": row["case_id"],
        }
