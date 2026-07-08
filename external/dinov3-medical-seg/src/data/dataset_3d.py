"""NIfTI volume datasets for few-shot 3D segmentation.

The Mimics integration materializes one binary organ task at a time:

    imagesTr/<case>.nii.gz
    labelsTr/<case>.nii.gz
    imagesVal/<case>.nii.gz   (optional)
    labelsVal/<case>.nii.gz   (optional)

Labels are expected to be integer masks where non-zero voxels are foreground.
The loader keeps the same axis convention used by scripts/infer.py: the first
NIfTI array dimension is treated as the slice/depth dimension.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset


def normalize_volume(data: np.ndarray, modality: str = "other") -> np.ndarray:
    """Return a float32 volume normalized to [0, 1].

    CT volumes are clipped to a broad soft-tissue/body range before min-max
    scaling. Other modalities use robust percentiles to avoid a single outlier
    dominating the dynamic range.
    """

    volume = np.asarray(data, dtype=np.float32)
    finite = np.isfinite(volume)
    if not np.any(finite):
        return np.zeros(volume.shape, dtype=np.float32)
    if not np.all(finite):
        replacement = float(np.nanmedian(volume[finite]))
        volume = np.where(finite, volume, replacement).astype(np.float32, copy=False)

    modality = str(modality or "other").lower()
    if modality == "ct":
        lo, hi = -1024.0, 1024.0
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


def _candidate_pairs(image_dir: Path, label_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    if not image_dir.is_dir() or not label_dir.is_dir():
        return pairs
    label_by_stem = {}
    for label in sorted(label_dir.glob("*.nii*")):
        label_by_stem[_case_stem(label)] = label
    for image in sorted(image_dir.glob("*.nii*")):
        label = label_by_stem.get(_case_stem(image))
        if label is not None:
            pairs.append((image, label))
    return pairs


def _case_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


class MedicalVolumeDataset(Dataset):
    """Load binary 3D segmentation volumes from a materialized dataset."""

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        img_size: Iterable[int] = (224, 224),
        slice_axis: int = 2,
        modality: str = "other",
        target_spacing=None,
        augmentation=None,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = tuple(int(v) for v in img_size)
        self.slice_axis = int(slice_axis)
        self.modality = modality
        self.target_spacing = target_spacing
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
                "case_id": _case_stem(image),
                "image": image,
                "label": label,
            }
            for image, label in pairs
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        image = nib.as_closest_canonical(nib.load(str(sample["image"])))
        label = nib.as_closest_canonical(nib.load(str(sample["label"])))
        if not np.allclose(image.affine, label.affine, atol=1e-4, rtol=0.0):
            raise RuntimeError(
                "Image/label affine mismatch for {} after canonical orientation: {} vs {}".format(
                    sample["case_id"],
                    sample["image"],
                    sample["label"],
                )
            )
        image_data = normalize_volume(image.get_fdata(dtype=np.float32), self.modality)
        label_data = (label.get_fdata(dtype=np.float32) > 0).astype(np.int64)

        if image_data.shape != label_data.shape:
            raise RuntimeError(
                "Image/label shape mismatch for {}: {} vs {}".format(
                    sample["case_id"],
                    image_data.shape,
                    label_data.shape,
                )
            )

        image_tensor = torch.from_numpy(image_data).unsqueeze(0).unsqueeze(0)
        label_tensor = torch.from_numpy(label_data).unsqueeze(0).unsqueeze(0).float()
        image_tensor = F.interpolate(image_tensor, size=(image_data.shape[0], *self.img_size), mode="trilinear", align_corners=False)
        label_tensor = F.interpolate(label_tensor, size=(label_data.shape[0], *self.img_size), mode="nearest")
        image_tensor = image_tensor.squeeze(0)
        label_tensor = label_tensor.squeeze(0).squeeze(0).long()

        item = {
            "image": image_tensor,
            "label": label_tensor,
            "case_id": sample["case_id"],
            "image_path": str(sample["image"]),
            "label_path": str(sample["label"]),
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
