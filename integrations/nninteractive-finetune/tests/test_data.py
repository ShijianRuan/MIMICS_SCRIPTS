import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from nninteractive_finetune.data import (
    InteractivePatchDataset,
    _augment_intensity_like_nnunet,
    _nnunet_gaussian_blur,
    _sample_spatial_patch,
    load_manifest,
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
    image = (
        np.arange(32**3, dtype=np.float32).reshape((32, 32, 32)) + 1
    )
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


def test_prepare_normalizes_conflicting_qform_and_sform(tmp_path):
    """A conflicting qform/sform is resolved by preferring sform.

    Mimics-exported NIfTIs frequently carry a valid qform that disagrees
    with the (correct) sform.  Training must not fail on this; the sform
    geometry is used and the qform is cleared.
    """
    image_path = tmp_path / "image.nii.gz"
    label_path = tmp_path / "label.nii.gz"
    image = nib.Nifti1Image(
        np.linspace(1, 32, 32 * 32 * 32, dtype=np.float32).reshape(32, 32, 32),
        np.eye(4),
    )
    shifted = np.eye(4)
    shifted[1, 3] = 3.0
    image.set_qform(np.eye(4), code=1)
    image.set_sform(shifted, code=1)
    nib.save(image, str(image_path))
    label = np.zeros((32, 32, 32), dtype=np.uint8)
    label[4:8, 4:8, 4:8] = 1
    label_nii = nib.Nifti1Image(label, np.eye(4))
    label_nii.set_qform(np.eye(4), code=1)
    label_nii.set_sform(shifted, code=1)
    nib.save(label_nii, str(label_path))
    rows = [
        {
            "case_id": "case",
            "image": str(image_path),
            "label": str(label_path),
            "split": "train",
        }
    ]
    # Should not raise; sform wins.
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        prepared = prepare_cases(rows, tmp_path / "cache", [1])
    assert len(prepared) == 1


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


def test_prepare_and_patch_preserve_optional_initial_mask(tmp_path):
    image_path = tmp_path / "image.nii.gz"
    label_path = tmp_path / "label.nii.gz"
    initial_path = tmp_path / "initial.nii.gz"
    image = (
        np.arange(32**3, dtype=np.float32).reshape((32, 32, 32)) + 1
    )
    label = np.zeros_like(image, dtype=np.uint8)
    initial = np.zeros_like(label)
    label[6:26, 6:26, 6:26] = 1
    initial[8:24, 8:24, 8:24] = 1
    for path, array in (
        (image_path, image),
        (label_path, label),
        (initial_path, initial),
    ):
        _write(path, array, np.eye(4))
    prepared = prepare_cases(
        [
            {
                "case_id": "case",
                "image": str(image_path),
                "label": str(label_path),
                "initial_mask": str(initial_path),
                "initial_mask_source_type": "dinov3_prediction",
                "initial_mask_source_model": "liver_v2",
                "split": "train",
            }
        ],
        tmp_path / "cache",
        [1],
    )
    cached_initial = np.load(prepared[0]["prepared_initial_mask"])
    assert int(cached_initial.sum()) == int(initial.sum())
    metadata = prepared[0]["metadata"]
    assert metadata["initial_mask_quality_bin"] == "medium"
    assert metadata["initial_mask_precision"] == pytest.approx(1.0)
    assert metadata["initial_mask_recall"] == pytest.approx(16**3 / 20**3)
    assert metadata["initial_mask_source_type"] == "dinov3_prediction"
    assert metadata["initial_mask_source_model"] == "liver_v2"
    dataset = InteractivePatchDataset(
        prepared,
        (32, 32, 32),
        1.0,
        1,
        augmentation={"enabled": False},
    )
    sample = dataset[0]
    assert sample["has_initial_mask"] is True
    assert sample["initial_mask_quality_bin"] == "medium"
    assert sample["initial_mask_source_type"] == "dinov3_prediction"
    assert int(sample["initial_mask"].sum()) > 0
    assert torch.all(
        sample["initial_mask"].bool() <= sample["target"].bool()
    )


def test_nnunet_profile_keeps_spatially_transformed_masks_aligned(tmp_path):
    image = np.zeros((48, 48, 48), dtype=np.float32)
    label = np.zeros((48, 48, 48), dtype=np.uint8)
    image[14:34, 16:32, 12:36] = 1.0
    label[14:34, 16:32, 12:36] = 1
    image_path = tmp_path / "image.npy"
    label_path = tmp_path / "label.npy"
    initial_path = tmp_path / "initial.npy"
    np.save(image_path, image)
    np.save(label_path, label)
    np.save(initial_path, label)
    np.random.seed(11)
    sample = InteractivePatchDataset(
        [
            {
                "case_id": "case",
                "prepared_image": str(image_path),
                "prepared_label": str(label_path),
                "prepared_initial_mask": str(initial_path),
                "metadata": {
                    "has_initial_mask": True,
                    "initial_mask_quality_bin": "high",
                    "initial_mask_source_type": "test",
                },
            }
        ],
        (32, 32, 32),
        1.0,
        1,
        augmentation={
            "enabled": True,
            "profile": "nninteractive_nnunet",
            "rotation_probability": 1.0,
            "rotation_degrees": [20.0, 20.0],
            "scaling_probability": 1.0,
            "scaling_range": [0.9, 0.9],
            "noise_probability": 0.0,
            "blur_probability": 0.0,
            "brightness_probability": 0.0,
            "contrast_probability": 0.0,
            "low_resolution_probability": 0.0,
            "gamma_invert_probability": 0.0,
            "gamma_probability": 0.0,
            "flip_probability": 0.0,
            "mirror_axes": [0, 1, 2],
        },
    )[0]
    assert torch.equal(sample["target"], sample["initial_mask"])
    image_foreground = sample["image"][0].numpy() > 0.5
    target = sample["target"].numpy().astype(bool)
    overlap = 2 * np.count_nonzero(image_foreground & target)
    denominator = int(image_foreground.sum()) + int(target.sum())
    assert overlap / denominator > 0.9


def test_nnunet_noise_range_uses_official_sigma_semantics():
    np.random.seed(23)
    output = _augment_intensity_like_nnunet(
        np.zeros((48, 48, 48), dtype=np.float32),
        {
            "noise_probability": 1.0,
            "noise_variance_range": [0.1, 0.1],
            "blur_probability": 0.0,
            "brightness_probability": 0.0,
            "contrast_probability": 0.0,
            "low_resolution_probability": 0.0,
            "gamma_invert_probability": 0.0,
            "gamma_probability": 0.0,
        },
    )
    assert float(output.std()) == pytest.approx(0.1, abs=0.003)


def test_local_gaussian_blur_matches_public_batchgeneratorsv2_operator():
    gaussian_blur = pytest.importorskip(
        "batchgeneratorsv2.transforms.noise.gaussian_blur"
    )
    rng = np.random.default_rng(17)
    image = rng.normal(size=(18, 19, 20)).astype(np.float32)
    sigmas = (0.55, 0.8, 0.95)
    official = torch.from_numpy(image.copy())[None]
    for axis, sigma in enumerate(sigmas):
        official = gaussian_blur.blur_dimension(
            official,
            sigma,
            axis,
            force_use_fft=False,
            truncate=6,
        )
    local = _nnunet_gaussian_blur(image, sigmas)
    np.testing.assert_allclose(local, official[0].numpy(), rtol=1e-6, atol=1e-6)


def test_spatial_identity_matches_even_patch_crop_without_half_voxel_shift():
    array = np.arange(24**3, dtype=np.float32).reshape((24, 24, 24))
    center = np.asarray([12, 11, 10], dtype=np.int64)
    patch_size = (8, 10, 12)
    transformed = _sample_spatial_patch(
        array,
        center,
        patch_size,
        np.eye(3, dtype=np.float64),
        order=1,
    )
    expected = array[8:16, 6:16, 4:16]
    np.testing.assert_allclose(transformed, expected, rtol=0, atol=1e-5)


def test_spatial_scale_above_one_makes_object_smaller_like_nnunet():
    label = np.zeros((33, 33, 33), dtype=np.uint8)
    label[12:21, 12:21, 12:21] = 1
    center = np.asarray([16, 16, 16], dtype=np.int64)
    identity = _sample_spatial_patch(
        label, center, (17, 17, 17), np.eye(3), order=0
    )
    scaled = _sample_spatial_patch(
        label, center, (17, 17, 17), np.eye(3) * 1.4, order=0
    )
    assert 0 < int(scaled.sum()) < int(identity.sum())


def test_scipy_spatial_sampling_matches_public_batchgeneratorsv2_grid():
    spatial = pytest.importorskip(
        "batchgeneratorsv2.transforms.spatial.spatial"
    )
    rng = np.random.default_rng(31)
    image = rng.normal(size=(28, 30, 32)).astype(np.float32)
    patch_size = (14, 16, 18)
    center = np.asarray([14, 15, 16], dtype=np.int64)
    angles = np.deg2rad([7.0, -5.0, 9.0])
    official_affine = spatial.create_affine_matrix_3d(
        angles, [1.08, 1.08, 1.08]
    )

    grid = spatial._create_centered_identity_grid2(patch_size).float()
    grid = torch.matmul(grid, torch.from_numpy(official_affine).float())
    grid += torch.tensor(
        [float(c) - float(s) / 2.0 for c, s in zip(center, image.shape)]
    )
    grid = spatial._convert_my_grid_to_grid_sample_grid(grid, image.shape)
    official = torch.nn.functional.grid_sample(
        torch.from_numpy(image)[None, None],
        grid[None],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[0, 0].numpy()
    local = _sample_spatial_patch(
        image,
        center,
        patch_size,
        official_affine.T,
        order=1,
    )

    np.testing.assert_allclose(local, official, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    ("initial_kind", "expected_reason"),
    [
        ("empty", "empty"),
        ("target", "identical_to_target"),
    ],
)
def test_prepare_rejects_invalid_initial_mask_without_rejecting_case(
    tmp_path, initial_kind, expected_reason
):
    image_path = tmp_path / "image.nii.gz"
    label_path = tmp_path / "label.nii.gz"
    initial_path = tmp_path / "initial.nii.gz"
    image = (
        np.arange(32**3, dtype=np.float32).reshape((32, 32, 32)) + 1
    )
    label = np.zeros((32, 32, 32), dtype=np.uint8)
    label[6:26, 6:26, 6:26] = 1
    initial = (
        np.zeros_like(label)
        if initial_kind == "empty"
        else label.copy()
    )
    for path, array in (
        (image_path, image),
        (label_path, label),
        (initial_path, initial),
    ):
        _write(path, array, np.eye(4))
    rows = [
        {
            "case_id": "case",
            "image": str(image_path),
            "label": str(label_path),
            "initial_mask": str(initial_path),
            "split": "train",
        }
    ]
    prepared = prepare_cases(rows, tmp_path / "cache", [1])
    assert prepared[0]["prepared_initial_mask"] == ""
    assert prepared[0]["metadata"]["has_initial_mask"] is False
    assert (
        prepared[0]["metadata"]["initial_mask_rejected_reason"]
        == expected_reason
    )
    sample = InteractivePatchDataset(
        prepared,
        (32, 32, 32),
        1.0,
        1,
        augmentation={"enabled": False},
    )[0]
    assert sample["has_initial_mask"] is False
    assert int(sample["initial_mask"].sum()) == 0

    reused = prepare_cases(rows, tmp_path / "cache", [1])
    assert reused[0]["prepared_initial_mask"] == ""
    assert (
        reused[0]["metadata"]["initial_mask_rejected_reason"]
        == expected_reason
    )


def test_initial_mask_cache_uses_original_source_fingerprint(tmp_path):
    image_path = tmp_path / "image.nii.gz"
    label_path = tmp_path / "label.nii.gz"
    source_initial_path = tmp_path / "source_initial.nii.gz"
    staged_initial_path = tmp_path / "staged_initial.nii.gz"
    image = (
        np.arange(32**3, dtype=np.float32).reshape((32, 32, 32)) + 1
    )
    label = np.zeros((32, 32, 32), dtype=np.uint8)
    label[6:26, 6:26, 6:26] = 1
    initial = np.zeros_like(label)
    initial[8:24, 8:24, 8:24] = 1
    for path, array in (
        (image_path, image),
        (label_path, label),
        (source_initial_path, initial),
        (staged_initial_path, initial),
    ):
        _write(path, array, np.eye(4))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "case",
                        "image": str(image_path),
                        "label": str(label_path),
                        "initial_mask": str(staged_initial_path),
                        "source_initial_mask": str(source_initial_path),
                        "split": "train",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    rows = load_manifest(manifest_path)
    cache = tmp_path / "cache"
    first = prepare_cases(rows, cache, [1])
    metadata_path = Path(first[0]["prepared_image"]).parent / "metadata.json"
    first_mtime = metadata_path.stat().st_mtime_ns
    time.sleep(0.01)
    _write(staged_initial_path, initial, np.eye(4))
    second = prepare_cases(rows, cache, [1])
    assert second[0]["metadata"]["has_initial_mask"] is True
    assert metadata_path.stat().st_mtime_ns == first_mtime
