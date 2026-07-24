"""Native-grid slice preprocessing and frozen-feature cache datasets.

This path intentionally does not canonicalize or resample the NIfTI volume.
The source voxel array is normalized once per volume, resized slice by slice
for the 2D model, and restored to the original NIfTI grid after inference.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Callable, Iterable

import nibabel as nib
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .dataset_3d import _candidate_pairs, _case_id


DEFAULT_FEATURE_SLICE_SIZE = (256, 256)
DEFAULT_FEATURE_NORMALIZATION = "timeslice_casewise"


class FeatureCacheCancelled(RuntimeError):
    """Raised when an external cancellation marker is observed."""


def normalize_percentile_volume(volume: np.ndarray) -> np.ndarray:
    """Apply percentile min-max normalization for the PyTorch alternative."""
    values = np.asarray(volume, dtype=np.float32)
    if values.size == 0:
        return values
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros(values.shape, dtype=np.float32)
    low, high = np.percentile(values[finite], [0.5, 99.5])
    low, high = float(low), float(high)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros(values.shape, dtype=np.float32)
    result = np.clip((values - low) / (high - low), 0.0, 1.0)
    return np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0).astype(
        np.float32,
        copy=False,
    )


def normalize_timeslice_casewise_volume(volume: np.ndarray) -> np.ndarray:
    """Reproduce the model-specific casewise clip and z-score contract."""
    values = np.asarray(volume, dtype=np.float32)
    if values.size == 0:
        return values
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros(values.shape, dtype=np.float32)
    low, high = np.percentile(values[finite], [0.5, 99.5])
    low, high = float(low), float(high)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32,
            copy=False,
        )

    clipped = np.clip(values, low, high)
    # The reference computes statistics above the lower percentile from the
    # original case, then applies them to the clipped volume.
    statistics = values[finite & (values > low)]
    if statistics.size == 0:
        statistics = clipped[finite]
    mean = float(statistics.mean())
    std = float(statistics.std())
    if not np.isfinite(mean):
        mean = 0.0
    if not np.isfinite(std) or std <= np.finfo(np.float32).eps:
        std = 1.0
    result = (clipped - mean) / std
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32,
        copy=False,
    )


def normalize_feature_volume(volume: np.ndarray, method: str) -> np.ndarray:
    method = str(method or DEFAULT_FEATURE_NORMALIZATION).strip().lower()
    if method in ("timeslice_casewise", "casewise_clip_0.5_99.5_then_zscore"):
        return normalize_timeslice_casewise_volume(volume)
    if method in ("percentile_minmax", "minmax"):
        return normalize_percentile_volume(volume)
    raise ValueError("Unsupported frozen-feature normalization: {}".format(method))


def _resize_slice(
    values: np.ndarray,
    *,
    label: bool,
    slice_size: tuple[int, int],
) -> np.ndarray:
    interpolation = Image.Resampling.NEAREST if label else Image.Resampling.BILINEAR
    source = np.asarray(values, dtype=np.uint8 if label else np.float32)
    resized = Image.fromarray(source).resize(
        (slice_size[1], slice_size[0]),
        interpolation,
    )
    result = np.asarray(resized)
    if label:
        return (result > 0).astype(np.int64, copy=False)
    return result.astype(np.float32, copy=False)


def _load_native_zyx(path: Path, *, dtype=np.float32):
    image = nib.load(str(path))
    array_xyz = np.asanyarray(image.dataobj).astype(dtype, copy=False)
    if array_xyz.ndim != 3:
        raise RuntimeError("{} is not a 3D volume: {}".format(path, array_xyz.shape))
    return image, np.transpose(array_xyz, (2, 1, 0))


def prepare_native_case(
    image_path: Path,
    label_path: Path | None = None,
    *,
    slice_size=DEFAULT_FEATURE_SLICE_SIZE,
    normalization=DEFAULT_FEATURE_NORMALIZATION,
) -> dict:
    """Load one case without reorientation and create its model-grid slices."""
    slice_size = tuple(int(value) for value in slice_size)
    if len(slice_size) != 2 or any(value <= 0 or value % 16 for value in slice_size):
        raise ValueError("Frozen feature slice size must contain two positive multiples of 16")
    image_nii, image_zyx = _load_native_zyx(Path(image_path))
    native_shape_zyx = tuple(int(value) for value in image_zyx.shape)
    normalized = normalize_feature_volume(image_zyx, normalization)
    images = np.stack(
        [_resize_slice(slc, label=False, slice_size=slice_size) for slc in normalized],
        axis=0,
    )

    labels = None
    if label_path is not None:
        label_nii, label_zyx = _load_native_zyx(Path(label_path))
        if tuple(label_zyx.shape) != native_shape_zyx:
            raise RuntimeError(
                "Image/label native shape mismatch for {}: {} vs {}".format(
                    image_path,
                    native_shape_zyx,
                    tuple(label_zyx.shape),
                )
            )
        if not np.allclose(image_nii.affine, label_nii.affine, atol=1e-4, rtol=1e-5):
            raise RuntimeError(
                "Image/label affine mismatch for {}; export labels on the source image grid first".format(
                    image_path
                )
            )
        labels = np.stack(
            [
                _resize_slice(slc > 0, label=True, slice_size=slice_size)
                for slc in label_zyx
            ],
            axis=0,
        )

    return {
        "image_nii": image_nii,
        "images": images,
        "labels": labels,
        "native_shape_zyx": native_shape_zyx,
    }


def restore_native_prediction(
    prediction_model_zyx: np.ndarray,
    original_image: nib.spatialimages.SpatialImage,
) -> nib.Nifti1Image:
    """Resize model slices back and preserve the input NIfTI affine/header."""
    native_shape_xyz = tuple(int(value) for value in original_image.shape)
    native_shape_zyx = native_shape_xyz[::-1]
    prediction = np.asarray(prediction_model_zyx)
    if prediction.ndim != 3 or prediction.shape[0] != native_shape_zyx[0]:
        raise RuntimeError(
            "Prediction depth {} does not match native depth {}".format(
                prediction.shape if prediction.ndim == 3 else "invalid",
                native_shape_zyx[0],
            )
        )
    restored_zyx = np.stack(
        [
            np.asarray(
                Image.fromarray(np.asarray(slc, dtype=np.uint8)).resize(
                    (native_shape_zyx[2], native_shape_zyx[1]),
                    Image.Resampling.NEAREST,
                )
            )
            for slc in prediction
        ],
        axis=0,
    )
    restored_xyz = np.transpose(restored_zyx, (2, 1, 0)).astype(np.int16, copy=False)
    header = original_image.header.copy()
    header.set_data_dtype(np.int16)
    return nib.Nifti1Image(restored_xyz, original_image.affine, header)


class NativeSliceVolumeDataset(Dataset):
    """Volume-level source for deterministic feature-cache construction."""

    def __init__(
        self,
        data_root: str,
        split: str,
        slice_size=DEFAULT_FEATURE_SLICE_SIZE,
        normalization=DEFAULT_FEATURE_NORMALIZATION,
    ):
        root = Path(data_root)
        self.slice_size = tuple(int(value) for value in slice_size)
        self.normalization = str(normalization or DEFAULT_FEATURE_NORMALIZATION)
        if split in ("train", "tr"):
            pairs = _candidate_pairs(root / "imagesTr", root / "labelsTr")
        else:
            pairs = _candidate_pairs(root / "imagesVal", root / "labelsVal")
        if not pairs:
            raise RuntimeError(
                "No native-grid image/label pairs found for split '{}' under {}".format(
                    split,
                    root,
                )
            )
        self.samples = [
            {
                "case_id": _case_id(image, image=True),
                "image": image,
                "label": label,
            }
            for image, label in pairs
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        prepared = prepare_native_case(
            sample["image"],
            sample["label"],
            slice_size=self.slice_size,
            normalization=self.normalization,
        )
        return {
            "case_id": sample["case_id"],
            "image_path": str(sample["image"]),
            "label_path": str(sample["label"]),
            "images": prepared["images"],
            "labels": prepared["labels"],
            "native_shape_zyx": prepared["native_shape_zyx"],
        }


def _write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".{}.tmp".format(os.getpid()))
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def build_feature_cache(
    model,
    dataset: NativeSliceVolumeDataset,
    cache_dir: Path,
    device: torch.device,
    *,
    slice_batch_size: int,
    progress: Callable[[dict], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> Path:
    """Encode each source slice once and write memory-mapped cache arrays."""
    cache_dir = Path(cache_dir)
    temporary = cache_dir.with_name(cache_dir.name + ".building")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True, exist_ok=True)
    rows = []
    model.backbone.eval()
    encoder_device = (
        torch.device("cpu")
        if getattr(model, "encoder_backend", "pytorch") == "onnx"
        else device
    )
    total_slices = sum(int(nib.load(str(sample["image"])).shape[2]) for sample in dataset.samples)
    completed_slices = 0
    try:
        for case_index in range(len(dataset)):
            if cancelled and cancelled():
                raise FeatureCacheCancelled("Feature preparation cancelled")
            case = dataset[case_index]
            case_id = str(case["case_id"])
            case_dir = temporary / "{:04d}_{}".format(case_index, case_id)
            case_dir.mkdir(parents=True, exist_ok=True)
            images = np.asarray(case["images"], dtype=np.float32)
            labels = np.asarray(case["labels"], dtype=np.int64)
            depth = int(images.shape[0])
            if labels.shape != images.shape:
                raise RuntimeError(
                    "Prepared image/label mismatch for {}: {} vs {}".format(
                        case_id,
                        images.shape,
                        labels.shape,
                    )
                )
            raw_path = case_dir / "images.npy"
            label_path = case_dir / "labels.npy"
            np.save(str(raw_path), images[:, None, :, :], allow_pickle=False)
            np.save(str(label_path), labels, allow_pickle=False)

            embedding_path = case_dir / "embeddings.npy"
            embedding_map = None
            for start in range(0, depth, max(1, int(slice_batch_size))):
                if cancelled and cancelled():
                    raise FeatureCacheCancelled("Feature preparation cancelled")
                end = min(depth, start + max(1, int(slice_batch_size)))
                raw = torch.from_numpy(images[start:end, None]).to(
                    device=encoder_device,
                    dtype=torch.float32,
                )
                rgb = raw.repeat(1, 3, 1, 1)
                with torch.no_grad():
                    feature = model.backbone(rgb)[-1].detach().to(
                        device="cpu",
                        dtype=torch.float32,
                    ).numpy()
                if embedding_map is None:
                    embedding_map = np.lib.format.open_memmap(
                        str(embedding_path),
                        mode="w+",
                        dtype=np.float32,
                        shape=(depth, *feature.shape[1:]),
                    )
                embedding_map[start:end] = feature
                embedding_map.flush()
                completed_slices += end - start
                if progress is not None:
                    progress({
                        "phase": "feature_cache",
                        "case": case_index + 1,
                        "cases": len(dataset),
                        "case_id": case_id,
                        "slice": end,
                        "slices_in_case": depth,
                        "completed_slices": completed_slices,
                        "total_slices": total_slices,
                    })
            del embedding_map
            rows.append({
                "case_id": case_id,
                "directory": case_dir.name,
                "depth": depth,
                "native_shape_zyx": list(case["native_shape_zyx"]),
                "image_path": case["image_path"],
                "label_path": case["label_path"],
            })

        _write_manifest(temporary / "manifest.json", {
            "schema_version": "mimics_frozen_feature_cache.v1",
            "created_at_epoch": time.time(),
            "slice_size": list(dataset.slice_size),
            "normalization": dataset.normalization,
            "cases": rows,
        })
        shutil.rmtree(cache_dir, ignore_errors=True)
        os.replace(str(temporary), str(cache_dir))
        return cache_dir / "manifest.json"
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


class CachedFeatureSliceDataset(Dataset):
    """Flat slice index over one frozen-feature cache manifest."""

    def __init__(self, manifest_path: Path):
        self.manifest_path = Path(manifest_path)
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.cases = list(payload.get("cases") or [])
        if not self.cases:
            raise RuntimeError("Frozen feature cache contains no cases: {}".format(manifest_path))
        self.indices = []
        for case_index, case in enumerate(self.cases):
            for slice_index in range(int(case["depth"])):
                self.indices.append((case_index, slice_index))
        self._arrays = {}

    def __len__(self) -> int:
        return len(self.indices)

    def _case_arrays(self, case_index: int):
        if case_index not in self._arrays:
            directory = self.manifest_path.parent / self.cases[case_index]["directory"]
            self._arrays[case_index] = (
                np.load(str(directory / "embeddings.npy"), mmap_mode="r"),
                np.load(str(directory / "images.npy"), mmap_mode="r"),
                np.load(str(directory / "labels.npy"), mmap_mode="r"),
            )
        return self._arrays[case_index]

    def __getitem__(self, index: int) -> dict:
        case_index, slice_index = self.indices[index]
        embeddings, images, labels = self._case_arrays(case_index)
        return {
            "embedding": torch.from_numpy(np.array(embeddings[slice_index], copy=True)),
            "image": torch.from_numpy(np.array(images[slice_index], copy=True)),
            "label": torch.from_numpy(np.array(labels[slice_index], copy=True)).long(),
            "case_id": self.cases[case_index]["case_id"],
            "slice_index": slice_index,
        }

    def close(self) -> None:
        """Release memory maps before Windows cache deletion."""
        for arrays in self._arrays.values():
            for array in arrays:
                mmap = getattr(array, "_mmap", None)
                if mmap is not None:
                    try:
                        mmap.close()
                    except Exception:
                        pass
        self._arrays.clear()


def remove_feature_cache(paths: Iterable[Path]) -> list:
    remaining = []
    for path in paths:
        try:
            path = Path(path)
            shutil.rmtree(path, ignore_errors=False)
        except FileNotFoundError:
            continue
        except Exception:
            remaining.append(str(path))
    return remaining
