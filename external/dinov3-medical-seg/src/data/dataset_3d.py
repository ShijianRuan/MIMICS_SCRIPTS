"""NIfTI volume datasets for few-shot 3D segmentation.

The Mimics integration materializes one binary organ task at a time:

    imagesTr/<case>.nii.gz
    labelsTr/<case>.nii.gz
    imagesVal/<case>.nii.gz   (optional)
    labelsVal/<case>.nii.gz   (optional)

Labels are expected to be integer masks where non-zero voxels are foreground.
All model tensors use the explicit ``(C, Z, Y, X)`` convention.  NIfTI files
are first reoriented to canonical RAS, then converted from NIfTI ``(X, Y, Z)``
array order to model ``(Z, Y, X)`` order.  Training and inference share the
same helpers in :mod:`data.spatial`.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch.utils.data import Dataset, Subset

from .spatial import load_canonical_pair, resample_pair_to_spacing, spacing_zyx, xyz_to_zyx


def _unit_window(volume: np.ndarray, low: float, high: float) -> np.ndarray:
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("Invalid intensity window [{}, {}]".format(low, high))
    return np.clip((volume - low) / (high - low), 0.0, 1.0).astype(np.float32, copy=False)


def normalize_volume(
    data: np.ndarray,
    modality: str = "other",
    intensity: Mapping | None = None,
) -> np.ndarray:
    """Return a float32 volume normalized to [0, 1].

    The mapping never reads a segmentation label. CT uses a fixed window by
    default, while non-CT volumes use robust image-only percentiles. This keeps
    training and deployment numerically identical.
    """

    volume = np.asarray(data, dtype=np.float32)
    finite = np.isfinite(volume)
    if not np.any(finite):
        return np.zeros(volume.shape, dtype=np.float32)
    if not np.all(finite):
        replacement = float(np.nanmedian(volume[finite]))
        volume = np.where(finite, volume, replacement).astype(np.float32, copy=False)

    modality = str(modality or "other").lower()
    cfg = dict(intensity or {})
    if modality == "ct":
        window = cfg.get("window", [-1024.0, 1024.0])
        if not isinstance(window, (list, tuple)) or len(window) != 2:
            raise ValueError("CT intensity.window must contain [low, high]")
        lo, hi = float(window[0]), float(window[1])
    else:
        lo, hi = np.percentile(volume, [0.5, 99.5])
        lo, hi = float(lo), float(hi)
    if hi <= lo:
        lo, hi = float(volume.min()), float(volume.max())
    if hi > lo:
        volume = np.clip(volume, lo, hi)
        volume = (volume - lo) / (hi - lo)
    else:
        volume = np.zeros(volume.shape, dtype=np.float32)
    return volume.astype(np.float32, copy=False)


def build_input_channels(
    data_zyx: np.ndarray,
    modality: str = "other",
    intensity: Mapping | None = None,
    channel_policy: str = "repeat",
) -> np.ndarray:
    """Create label-free model input channels in ``(C, Z, Y, X)`` order.

    ``repeat`` and ``2_5d`` return one normalized channel.  The encoder builds
    RGB repeat/adjacent-slice inputs later. ``ct_windows`` creates exactly three
    fixed CT windows, so the DINOv3 input receives true three-channel data.
    """
    policy = str(channel_policy or "repeat").lower()
    cfg = dict(intensity or {})
    volume = np.asarray(data_zyx, dtype=np.float32)
    if policy in ("ct_windows", "multiwindow", "multi_window"):
        if str(modality or "").lower() != "ct":
            raise ValueError("ct_windows channel policy requires CT modality")
        windows = cfg.get("windows")
        if not isinstance(windows, (list, tuple)) or len(windows) != 3:
            raise ValueError("ct_windows requires exactly three fixed [low, high] windows")
        channels = [_unit_window(volume, float(pair[0]), float(pair[1])) for pair in windows]
        return np.stack(channels, axis=0).astype(np.float32, copy=False)
    if policy not in ("repeat", "single", "2_5d", "2.5d"):
        raise ValueError("Unknown channel policy: {}".format(channel_policy))
    return normalize_volume(volume, modality, cfg)[None, ...]


def prepare_model_input(
    data_zyx: np.ndarray,
    img_size: Iterable[int],
    *,
    modality: str = "other",
    intensity: Mapping | None = None,
    channel_policy: str = "repeat",
    slice_axis="axial",
) -> torch.Tensor:
    """Apply channel construction and resize the selected model slice plane."""
    size = tuple(int(value) for value in img_size)
    if len(size) != 2 or any(value <= 0 for value in size):
        raise ValueError("img_size must contain two positive values")
    channels = build_input_channels(
        data_zyx,
        modality=modality,
        intensity=intensity,
        channel_policy=channel_policy,
    )
    return _resize_slice_plane(torch.from_numpy(channels), size, slice_axis, mode="trilinear")


def _slice_axis_name(value) -> str:
    if isinstance(value, str):
        normalized = {"z": "axial", "y": "coronal", "x": "sagittal"}.get(
            value.strip().lower(), value.strip().lower()
        )
        if normalized in ("axial", "coronal", "sagittal"):
            return normalized
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        numeric = -1
    if numeric in {2: "axial", 1: "coronal", 0: "sagittal"}:
        return {2: "axial", 1: "coronal", 0: "sagittal"}[numeric]
    raise ValueError("slice_axis must be axial/coronal/sagittal or 2/1/0")


def _resize_slice_plane(tensor_czyx: torch.Tensor, size, slice_axis, *, mode: str) -> torch.Tensor:
    axis = _slice_axis_name(slice_axis)
    tensor = tensor_czyx.unsqueeze(0)
    if axis == "axial":
        oriented = tensor
    elif axis == "coronal":
        oriented = tensor.permute(0, 1, 3, 2, 4)
    else:
        oriented = tensor.permute(0, 1, 4, 2, 3)
    kwargs = {"align_corners": False} if mode != "nearest" else {}
    oriented = F.interpolate(
        oriented,
        size=(oriented.shape[2], *tuple(int(value) for value in size)),
        mode=mode,
        **kwargs,
    )
    if axis == "axial":
        restored = oriented
    elif axis == "coronal":
        restored = oriented.permute(0, 1, 3, 2, 4)
    else:
        restored = oriented.permute(0, 1, 3, 4, 2)
    return restored.squeeze(0)


def _candidate_pairs(image_dir: Path, label_dir: Path) -> list[tuple[Path, Path]]:
    if not image_dir.is_dir() or not label_dir.is_dir():
        return []
    label_by_case = {}
    for label in sorted(label_dir.glob("*.nii*")):
        case_id = _case_id(label, image=False)
        if case_id in label_by_case:
            raise RuntimeError(
                "Duplicate label case ID '{}' under {}: {} and {}".format(
                    case_id, label_dir, label_by_case[case_id].name, label.name
                )
            )
        label_by_case[case_id] = label
    image_by_case = {}
    for image in sorted(image_dir.glob("*.nii*")):
        case_id = _case_id(image, image=True)
        if case_id in image_by_case:
            raise RuntimeError(
                "Duplicate image case ID '{}' under {}: {} and {}".format(
                    case_id, image_dir, image_by_case[case_id].name, image.name
                )
            )
        image_by_case[case_id] = image
    missing_labels = sorted(set(image_by_case) - set(label_by_case))
    missing_images = sorted(set(label_by_case) - set(image_by_case))
    if missing_labels or missing_images:
        details = []
        if missing_labels:
            details.append("images without labels: {}".format(", ".join(missing_labels)))
        if missing_images:
            details.append("labels without images: {}".format(", ".join(missing_images)))
        raise RuntimeError("NIfTI image/label pairing mismatch under {} and {} ({})".format(
            image_dir, label_dir, "; ".join(details)
        ))
    return [(image_by_case[case_id], label_by_case[case_id]) for case_id in sorted(image_by_case)]


def _case_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def _case_id(path: Path, *, image: bool) -> str:
    """Normalize one supported NIfTI image/label pair identifier.

    MSD/nnU-Net commonly stores a single-modality image as ``case_0000`` and
    its label as ``case``. This project handles one image channel per task, so
    only the image-side ``_0000`` suffix is removed. Other modality suffixes
    are not silently merged: they would be an unsupported multi-modal input.
    """
    stem = _case_stem(path)
    return re.sub(r"_0000$", "", stem) if image else stem


class MedicalVolumeDataset(Dataset):
    """Load binary 3D segmentation volumes from a materialized dataset."""

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        img_size: Iterable[int] = (224, 224),
        modality: str = "other",
        target_spacing=None,
        intensity: Mapping | None = None,
        channel_policy: str = "repeat",
        patch: Mapping | None = None,
        roi: Mapping | None = None,
        target: Mapping | None = None,
        slice_axis="axial",
        augmentation=None,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = tuple(int(v) for v in img_size)
        self.modality = modality
        self.target_spacing = target_spacing
        self.intensity = dict(intensity or {})
        self.channel_policy = channel_policy
        self.patch = dict(patch or {})
        self.roi = dict(roi or {})
        self.target = dict(target or {})
        self.slice_axis = slice_axis
        self.use_patch = bool(self.patch.get("enabled", False)) and split in ("train", "tr")
        self.patch_size_zyx = tuple(int(value) for value in self.patch.get("size_zyx", []))
        self.foreground_probability = float(self.patch.get("foreground_probability", 0.67))
        self.patches_per_case = int(self.patch.get("patches_per_case_per_epoch", 1)) if self.use_patch else 1
        if self.patches_per_case < 1:
            raise ValueError("data.patch.patches_per_case_per_epoch must be >= 1")
        if self.use_patch and (len(self.patch_size_zyx) != 3 or any(value <= 0 for value in self.patch_size_zyx)):
            raise ValueError("data.patch.size_zyx must contain three positive ZYX values")
        self.augmentation = augmentation

        if split in ("train", "tr"):
            pairs = _candidate_pairs(self.data_root / "imagesTr", self.data_root / "labelsTr")
        else:
            pairs = _candidate_pairs(self.data_root / "imagesVal", self.data_root / "labelsVal")
            if not pairs:
                pairs = _candidate_pairs(self.data_root / "imagesTr", self.data_root / "labelsTr")
        if not pairs:
            raise RuntimeError(
                "No NIfTI image/label pairs found for split '{}' under {}".format(split, self.data_root)
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
        return len(self.samples) * self.patches_per_case

    @staticmethod
    def _normalized_roi_slices(shape, bounds):
        values = np.asarray(bounds, dtype=float)
        if values.shape != (2, 3) or np.any(values < 0.0) or np.any(values > 1.0) or np.any(values[1] <= values[0]):
            raise ValueError("data.roi.normalized_zyx must be [[z0,y0,x0],[z1,y1,x1]] within [0,1]")
        shape = np.asarray(shape, dtype=int)
        start = np.floor(values[0] * shape).astype(int)
        end = np.ceil(values[1] * shape).astype(int)
        start = np.maximum(0, np.minimum(start, shape - 1))
        end = np.maximum(start + 1, np.minimum(end, shape))
        return tuple(slice(int(a), int(b)) for a, b in zip(start, end))

    def _apply_roi(self, image_data, label_data):
        if not self.roi.get("enabled", False):
            return image_data, label_data
        slices = self._normalized_roi_slices(image_data.shape, self.roi.get("normalized_zyx"))
        return image_data[slices], label_data[slices]

    def _apply_target_transform(self, label_data, spacing):
        mode = str(self.target.get("mode", "segmentation"))
        if mode == "segmentation":
            return label_data
        if mode != "localization_ball":
            raise ValueError("Unknown data.target.mode: {}".format(mode))
        coordinates = np.where(label_data > 0)
        if not coordinates[0].size:
            return np.zeros_like(label_data)
        center = np.asarray([float(np.mean(axis)) for axis in coordinates])
        radius_mm = np.asarray(self.target.get("radius_mm_zyx", [24.0, 40.0, 40.0]), dtype=float)
        spacing = np.maximum(np.asarray(spacing, dtype=float), 1e-6)
        radius_voxels = np.maximum(radius_mm / spacing, 1.0)
        grids = np.ogrid[tuple(slice(0, int(length)) for length in label_data.shape)]
        distance = sum(((grid - center[axis]) / radius_voxels[axis]) ** 2 for axis, grid in enumerate(grids))
        return (distance <= 1.0).astype(np.int64)

    def _crop_patch(self, image_data: np.ndarray, label_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Sample a fixed-size support patch with a controlled foreground rate."""
        patch_size = np.asarray(self.patch_size_zyx, dtype=int)
        shape = np.asarray(label_data.shape, dtype=int)
        foreground_mask = label_data > 0
        foreground = np.flatnonzero(foreground_mask)
        sampling = dict(self.patch.get("sampling", {}))
        if sampling:
            erosion = ndimage.binary_erosion(foreground_mask, iterations=max(1, int(sampling.get("boundary_width", 2))))
            boundary = foreground_mask & ~erosion
            dilation = ndimage.binary_dilation(foreground_mask, iterations=max(1, int(sampling.get("near_negative_width", 8))))
            near_negative = dilation & ~foreground_mask
            pools = {
                "interior": np.flatnonzero(erosion if np.any(erosion) else foreground_mask),
                "boundary": np.flatnonzero(boundary),
                "near_negative": np.flatnonzero(near_negative),
                "random": np.arange(label_data.size),
            }
            weights = [(name, float(sampling.get(name, 0.0))) for name in pools]
            weights = [(name, value) for name, value in weights if value > 0.0 and pools[name].size]
            if weights:
                name = random.choices([name for name, _ in weights], weights=[value for _, value in weights], k=1)[0]
                chosen = int(random.choice(pools[name]))
                center = np.asarray(np.unravel_index(chosen, label_data.shape), dtype=int)
            else:
                center = np.asarray([random.randrange(int(length)) for length in shape], dtype=int)
        elif foreground.size and random.random() < self.foreground_probability:
            center = np.asarray(np.unravel_index(int(random.choice(foreground)), label_data.shape), dtype=int)
        else:
            center = np.asarray([random.randrange(int(length)) for length in shape], dtype=int)
        start = center - patch_size // 2
        end = start + patch_size
        pad_before = np.maximum(0, -start)
        pad_after = np.maximum(0, end - shape)
        if np.any(pad_before) or np.any(pad_after):
            pad = tuple((int(before), int(after)) for before, after in zip(pad_before, pad_after))
            image_data = np.pad(image_data, pad, mode="edge")
            label_data = np.pad(label_data, pad, mode="constant", constant_values=0)
            start = start + pad_before
        slices = tuple(slice(int(origin), int(origin + length)) for origin, length in zip(start, patch_size))
        return image_data[slices], label_data[slices]

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index % len(self.samples)]
        image, label, _, _, _ = load_canonical_pair(str(sample["image"]), str(sample["label"]))
        image, label = resample_pair_to_spacing(image, label, self.target_spacing)
        image_data = xyz_to_zyx(image.get_fdata(dtype=np.float32))
        label_data = (xyz_to_zyx(label.get_fdata(dtype=np.float32)) > 0).astype(np.int64)

        if image_data.shape != label_data.shape:
            raise RuntimeError(
                "Image/label shape mismatch for {}: {} vs {}".format(
                    sample["case_id"],
                    image_data.shape,
                    label_data.shape,
                )
            )
        image_data, label_data = self._apply_roi(image_data, label_data)
        label_data = self._apply_target_transform(label_data, spacing_zyx(image))
        if self.use_patch:
            image_data, label_data = self._crop_patch(image_data, label_data)

        image_tensor = prepare_model_input(
            image_data,
            self.img_size,
            modality=self.modality,
            intensity=self.intensity,
            channel_policy=self.channel_policy,
            slice_axis=self.slice_axis,
        )
        label_tensor = _resize_slice_plane(
            torch.from_numpy(label_data).unsqueeze(0).float(),
            self.img_size,
            self.slice_axis,
            mode="nearest",
        ).squeeze(0).long()

        item = {
            "image": image_tensor,
            "label": label_tensor,
            "case_id": sample["case_id"],
            "image_path": str(sample["image"]),
            "label_path": str(sample["label"]),
            "spacing_zyx": torch.tensor(spacing_zyx(image), dtype=torch.float32),
        }
        if self.augmentation is not None:
            item = self.augmentation(item)
        return item


class FewShotSubset(Subset):
    """Deterministic subset wrapper used by the upstream training script."""

    def __init__(self, dataset: Dataset, k: int, seed: int = 42):
        k = int(k)
        if k <= 0 or k >= len(dataset):
            indices = list(range(len(dataset)))
        else:
            rng = random.Random(seed)
            indices = list(range(len(dataset)))
            rng.shuffle(indices)
            indices = sorted(indices[:k])
        super().__init__(dataset, indices)


class CaseIdSubset(Subset):
    """Subset a dataset by an audited, explicit list of patient/case IDs."""

    def __init__(self, dataset: MedicalVolumeDataset, case_ids: Iterable[str]):
        requested = [str(case_id) for case_id in case_ids]
        index_by_case = {
            str(sample["case_id"]): index for index, sample in enumerate(dataset.samples)
        }
        missing = [case_id for case_id in requested if case_id not in index_by_case]
        if missing:
            raise RuntimeError("Support case IDs are absent from imagesTr: {}".format(", ".join(missing)))
        if len(set(requested)) != len(requested):
            raise RuntimeError("Support case IDs must be unique")
        super().__init__(dataset, [index_by_case[case_id] for case_id in requested])
        self.case_ids = requested
