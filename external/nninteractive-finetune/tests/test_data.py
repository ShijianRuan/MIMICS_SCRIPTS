from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from nninteractive_finetune.data import (
    InteractivePatchDataset,
    nninteractive_input_contract,
    normalize_like_nninteractive,
    prepare_cases,
)


def _write(path: Path, array: np.ndarray, affine: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(array, affine), str(path))


def test_normalization_uses_nonzero_bounding_crop():
    image = np.zeros((6, 6, 6), dtype=np.float32)
    image[1:5, 2:4, 1:5] = np.arange(32, dtype=np.float32).reshape(4, 2, 4) + 1
    normalized = normalize_like_nninteractive(image)
    crop = normalized[1:5, 2:4, 1:5]
    assert abs(float(crop.mean())) < 1e-6
    assert abs(float(crop.std(ddof=1)) - 1.0) < 1e-6


def test_normalization_matches_official_session_math():
    rng = np.random.default_rng(42)
    image = rng.normal(-250.0, 430.0, size=(8, 9, 10)).astype(np.float32)
    image[:2, :3, :] = 0.0
    tensor = torch.from_numpy(image.copy()).float()
    nonzero = torch.nonzero(tensor != 0, as_tuple=False)
    low = nonzero.min(dim=0).values
    high = nonzero.max(dim=0).values + 1
    crop = tensor[
        tuple(slice(int(start), int(stop)) for start, stop in zip(low, high))
    ]
    official = ((tensor - crop.mean()) / crop.std()).numpy()
    np.testing.assert_allclose(
        normalize_like_nninteractive(image),
        official,
        rtol=2e-6,
        atol=2e-6,
    )
    assert (
        nninteractive_input_contract()["normalization"]
        == "nonzero_spatial_bbox_zscore"
    )


def test_prepare_reorients_pair_without_resampling(tmp_path):
    image_path = tmp_path / "image.nii.gz"
    label_path = tmp_path / "label.nii.gz"
    affine = np.diag([-1.0, 1.0, 1.0, 1.0])
    affine[0, 3] = 7.0
    image = np.arange(32**3, dtype=np.float32).reshape((32, 32, 32)) + 1
    label = np.zeros_like(image, dtype=np.uint8)
    label[5:9, 8:12, 10:14] = 1
    _write(image_path, image, affine)
    _write(label_path, label, affine)
    rows = [
        {
            "case_id": "case",
            "image": str(image_path),
            "label": str(label_path),
            "split": "train",
        }
    ]
    prepared = prepare_cases(rows, tmp_path / "cache", [1])
    cached_label = np.load(prepared[0]["prepared_label"])
    assert cached_label.sum() == label.sum()
    assert prepared[0]["metadata"]["affine_close"] is True


def test_prepare_rejects_affine_mismatch(tmp_path):
    image_path = tmp_path / "image.nii.gz"
    label_path = tmp_path / "label.nii.gz"
    image = np.ones((32, 32, 32), dtype=np.float32)
    label = np.zeros((32, 32, 32), dtype=np.uint8)
    label[4:8, 4:8, 4:8] = 1
    _write(image_path, image, np.eye(4))
    shifted = np.eye(4)
    shifted[0, 3] = 2.0
    _write(label_path, label, shifted)
    rows = [
        {
            "case_id": "case",
            "image": str(image_path),
            "label": str(label_path),
            "split": "train",
        }
    ]
    with pytest.raises(ValueError, match="affines differ"):
        prepare_cases(rows, tmp_path / "cache", [1])


def test_prepare_tolerates_conflicting_qform_and_sform(tmp_path):
    image_path = tmp_path / "image.nii.gz"
    label_path = tmp_path / "label.nii.gz"
    image = nib.Nifti1Image(
        np.linspace(0, 1, 32 * 32 * 32, dtype=np.float32).reshape(32, 32, 32),
        np.eye(4),
    )
    shifted = np.eye(4)
    shifted[1, 3] = 3.0
    image.set_qform(np.eye(4), code=1)
    image.set_sform(shifted, code=1)
    nib.save(image, str(image_path))
    label = np.zeros((32, 32, 32), dtype=np.uint8)
    label[4:8, 4:8, 4:8] = 1
    # Label shares the sform so that after qform is re-encoded to sform the
    # image/label affines agree and prepare_cases succeeds.
    _write(label_path, label, shifted)
    rows = [
        {
            "case_id": "case",
            "image": str(image_path),
            "label": str(label_path),
            "split": "train",
        }
    ]
    # A conflicting qform/sform is resolved by preferring the sform (NIfTI
    # standard recommendation) instead of raising. The case prepares normally.
    prepare_cases(rows, tmp_path / "cache", [1])


def test_augmentation_keeps_image_and_label_flips_aligned(tmp_path, monkeypatch):
    image = np.zeros((32, 32, 32), dtype=np.float32)
    label = np.zeros((32, 32, 32), dtype=np.uint8)
    image[2:6, 3:7, 4:8] = 5
    label[2:6, 3:7, 4:8] = 1
    image_path = tmp_path / "image.npy"
    label_path = tmp_path / "label.npy"
    np.save(image_path, image)
    np.save(label_path, label)
    dataset = InteractivePatchDataset(
        [
            {
                "case_id": "case",
                "prepared_image": str(image_path),
                "prepared_label": str(label_path),
            }
        ],
        (32, 32, 32),
        1.0,
        1,
        augmentation={
            "enabled": True,
            "flip_probability": 1.0,
            "intensity_scale_range": [1.0, 1.0],
            "intensity_shift_range": [0.0, 0.0],
            "noise_std_range": [0.0, 0.0],
        },
    )
    sample = dataset[0]
    assert np.array_equal(
        (sample["image"][0].numpy() > 0), sample["target"].numpy().astype(bool)
    )
