"""Regression tests for geometry, loss, and study-protocol invariants."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_3d import MedicalVolumeDataset, build_input_channels, prepare_model_input
from src.data.spatial import (
    canonicalize,
    load_canonical_pair,
    restore_prediction_to_original,
    xyz_to_zyx,
)
from src.evaluation import binary_metrics
from src.models.decoder_3d import (
    LearnableZSmooth,
    MLPProbeDecoder3D,
    SegFormer3DDecoder,
    TokenPyramid3DDecoder,
)
from src.models.encoder_3d import SliceWiseEncoder3D
from src.models.feature_augmentation import WaveletDetailAttenuation
from src.inference import predict_array, postprocess_foreground, prediction_bbox_zyx, expand_bbox
from src.research import protocol
from src.research.fingerprint import build_training_fingerprint, derive_policy
from src.research.protocol import (
    _image_intensity_percentiles,
    discover_records,
    materialize_fold,
    select_diverse_support,
)
from src.training.losses import get_loss
from src.utils.config import load_config


def _save(path, array, affine):
    nib.save(nib.Nifti1Image(np.asarray(array), affine), str(path))


def test_orientation_round_trip_preserves_original_grid(tmp_path):
    original_data = np.arange(3 * 4 * 5, dtype=np.int16).reshape(3, 4, 5)
    # LPS-like negative X/Y directions require both axis flips during canonicalization.
    original = nib.Nifti1Image(original_data, np.diag([-1.5, -2.0, 3.0, 1.0]))
    original_path = tmp_path / "original.nii.gz"
    nib.save(original, str(original_path))
    loaded_original = nib.load(str(original_path))
    canonical = canonicalize(loaded_original)
    prediction_zyx = xyz_to_zyx(canonical.get_fdata(dtype=np.float32)).astype(np.int16)
    restored = restore_prediction_to_original(
        prediction_zyx,
        model_grid=canonical,
        original_image=loaded_original,
    )
    assert np.array_equal(restored.get_fdata().astype(np.int16), original_data)
    assert np.allclose(restored.affine, loaded_original.affine)


def test_postprocess_foreground_threshold_and_largest_component():
    # A high-probability blob of 8 voxels and a smaller 1-voxel speck. At
    # threshold 0.5 both survive; keep_largest_component must drop the speck.
    prob = np.zeros((6, 6, 6), dtype=np.float32)
    prob[1:3, 1:3, 1:3] = 0.9  # 8-voxel component
    prob[5, 5, 5] = 0.8        # 1-voxel speck
    mask = postprocess_foreground(prob, threshold=0.5, keep_largest_component=False)
    assert int(mask.sum()) == 9
    largest = postprocess_foreground(prob, threshold=0.5, keep_largest_component=True)
    assert int(largest.sum()) == 8
    assert largest[5, 5, 5] == 0
    # A stricter threshold prunes the 0.8 speck even without component filtering.
    strict = postprocess_foreground(prob, threshold=0.85, keep_largest_component=False)
    assert int(strict.sum()) == 8


def test_pair_loader_rejects_mismatched_grid(tmp_path):
    image_path = tmp_path / "image.nii.gz"
    label_path = tmp_path / "label.nii.gz"
    _save(image_path, np.zeros((4, 5, 6), dtype=np.float32), np.eye(4))
    _save(label_path, np.zeros((5, 5, 6), dtype=np.uint8), np.eye(4))
    with pytest.raises(RuntimeError, match="grid mismatch"):
        load_canonical_pair(str(image_path), str(label_path))


def test_ct_windows_and_resize_are_label_free():
    volume = np.array([[[[-1024.0, 0.0, 1000.0]]]], dtype=np.float32).squeeze(0)
    channels = build_input_channels(
        volume,
        modality="ct",
        intensity={"windows": [[-1024, 1024], [-100, 300], [150, 1500]]},
        channel_policy="ct_windows",
    )
    assert channels.shape == (3, 1, 1, 3)
    assert np.all((channels >= 0.0) & (channels <= 1.0))
    tensor = prepare_model_input(volume, [8, 8], modality="ct", intensity={"window": [-1024, 1024]})
    assert tensor.shape == (1, 1, 8, 8)


def test_dataset_uses_zyx_and_nearest_labels(tmp_path):
    root = tmp_path / "dataset"
    (root / "imagesTr").mkdir(parents=True)
    (root / "labelsTr").mkdir()
    image = np.zeros((4, 5, 6), dtype=np.float32)
    image[1, 2, 3] = 100.0
    label = np.zeros_like(image, dtype=np.uint8)
    label[1, 2, 3] = 1
    _save(root / "imagesTr" / "case.nii.gz", image, np.eye(4))
    _save(root / "labelsTr" / "case.nii.gz", label, np.eye(4))
    item = MedicalVolumeDataset(str(root), img_size=(10, 12), modality="ct")[0]
    assert item["image"].shape == (1, 6, 10, 12)
    assert item["label"].shape == (6, 10, 12)
    assert set(item["label"].unique().tolist()) <= {0, 1}


def test_localization_ball_and_virtual_patch_sampling(tmp_path):
    root = tmp_path / "dataset"
    (root / "imagesTr").mkdir(parents=True)
    (root / "labelsTr").mkdir()
    image = np.zeros((20, 24, 28), dtype=np.float32)
    label = np.zeros_like(image, dtype=np.uint8)
    label[9:11, 11:13, 13:15] = 1
    _save(root / "imagesTr" / "case.nii.gz", image, np.eye(4))
    _save(root / "labelsTr" / "case.nii.gz", label, np.eye(4))
    dataset = MedicalVolumeDataset(
        str(root), split="train", img_size=(16, 16),
        target={"mode": "localization_ball", "radius_mm_zyx": [4, 5, 6]},
        patch={"enabled": True, "size_zyx": [16, 16, 16],
               "patches_per_case_per_epoch": 4,
               # This assertion verifies the localization target transform,
               # so sample a foreground-centered patch deterministically.
               # Random/near-negative cells are covered by separate sampling
               # tests and may legitimately contain no foreground.
               "sampling": {"interior": 1.0, "boundary": 0.0,
                            "near_negative": 0.0, "random": 0.0}},
    )
    assert len(dataset) == 4
    item = dataset[3]
    assert item["label"].sum().item() > label.sum()
    assert item["label"].shape == (16, 16, 16)


def test_normalized_roi_rejects_bad_bounds_and_crops():
    slices = MedicalVolumeDataset._normalized_roi_slices(
        (100, 200, 300), [[0.1, 0.2, 0.3], [0.5, 0.6, 0.7]]
    )
    assert [(s.start, s.stop) for s in slices] == [(10, 50), (40, 120), (90, 210)]
    with pytest.raises(ValueError, match="normalized_zyx"):
        MedicalVolumeDataset._normalized_roi_slices((10, 10, 10), [[0, 0, 0], [1.1, 1, 1]])


def test_dice_focal_cldice_has_finite_gradient():
    criterion = get_loss({"loss": {"type": "dice_focal_cldice", "skeleton_iterations": 2}})
    logits = torch.randn(1, 2, 8, 12, 12, requires_grad=True)
    target = torch.zeros(1, 8, 12, 12, dtype=torch.long)
    target[:, 1:7, 5:7, 5:7] = 1
    result = criterion(logits, target)
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_focused_plan_is_k5_only_and_small_matrix():
    import yaml
    modal_study = _load_modal_study_module()
    plan = yaml.safe_load((PROJECT_ROOT / "config/research/focused_5shot_study.yaml").read_text())
    assert plan["study"]["shot_counts"] == [5]
    # 2 aorta + 2 scapula + 1 scapula rescue candidate.
    assert len(plan["candidates"]) == 5
    screen = modal_study._phase_job_specs(plan, "screen")
    assert len(screen) == 5
    confirmation = modal_study._phase_job_specs(
        plan, "confirm", ["aorta_2p5d_reference", "aorta_2p5d_cldice"]
    )
    assert len(confirmation) == 2 * 3 * 1 * 2
    assert {item["support_count"] for item in confirmation} == {5}


def test_scapula_rescue_uses_whole_roi_inference_and_more_negatives():
    # The inference-gap diagnostic showed sliding-window @0.5 floods the thin
    # bone (P 0.05) while whole-ROI inference recovers Dice 0.34. The rescue
    # candidate must therefore deploy whole-ROI (no sliding window) and sample
    # more true/random negatives with a lower positive focal push.
    import yaml
    plan = yaml.safe_load((PROJECT_ROOT / "config/research/focused_5shot_study.yaml").read_text())
    rescue = next(c for c in plan["candidates"] if c["id"] == "scapula_roi_coronal3d_rescue")
    assert rescue["task"] == "scapula_left"
    patch = rescue["overrides"]["data"]["patch"]
    assert patch["inference_sliding_window"] is False
    sampling = patch["sampling"]
    # More negatives than the original 0.25/0.15 split.
    assert sampling["near_negative"] + sampling["random"] > 0.40
    assert rescue["overrides"]["model"]["slice_axis"] == "coronal"
    # Lower positive focal push than the original 0.30 weight / 0.65 alpha.
    assert rescue["overrides"]["loss"]["focal_weight"] < 0.30
    assert rescue["overrides"]["loss"]["focal_alpha"] < 0.65


def test_two_stage_roi_coverage_is_explicit_and_bounded():
    from scripts.research.evaluate_two_stage import _roi_target_coverage
    target = np.zeros((20, 20, 20), dtype=np.uint8)
    target[8:12, 8:12, 8:12] = 1
    assert _roi_target_coverage((10, 10, 10), target, (8, 8, 8)) == 1.0
    partial = _roi_target_coverage((5, 5, 5), target, (8, 8, 8))
    assert 0.0 < partial < 1.0
    assert _roi_target_coverage(None, target, (8, 8, 8)) == 0.0


def test_two_stage_localization_gate_stops_weak_coarse_model():
    from scripts.research.run_two_stage import _localization_gate

    passing = _localization_gate({
        "coarse_detection_rate": 1.0,
        "roi_localization_success_rate": 0.78,
        "mean_roi_target_coverage": 0.91,
    })
    failing = _localization_gate({
        "coarse_detection_rate": 1.0,
        "roi_localization_success_rate": 0.44,
        "mean_roi_target_coverage": 0.72,
    })

    assert passing["passed"] is True
    assert failing["passed"] is False
    assert any("roi_localization_success_rate" in item for item in failing["failures"])
    assert any("mean_roi_target_coverage" in item for item in failing["failures"])


def test_surface_metrics_ignore_empty_volume_padding():
    prediction = np.zeros((80, 90, 100), dtype=np.uint8)
    target = np.zeros_like(prediction)
    prediction[30:38, 40:48, 50:58] = 1
    target[31:39, 42:50, 50:58] = 1
    compact_prediction = prediction[20:50, 30:60, 40:70]
    compact_target = target[20:50, 30:60, 40:70]

    padded = binary_metrics(prediction, target, spacing=(2.0, 1.0, 1.0))
    compact = binary_metrics(compact_prediction, compact_target, spacing=(2.0, 1.0, 1.0))

    for key in ("dice", "hd95_mm", "assd_mm", "surface_dice", "lesion_f1"):
        assert padded[key] == pytest.approx(compact[key])


@pytest.mark.parametrize(
    ("slice_axis", "expected_shape"),
    [
        ("axial", (1, 5, 16, 32)),
        ("coronal", (1, 16, 7, 32)),
        ("sagittal", (1, 16, 32, 9)),
    ],
)
def test_prepare_model_input_resizes_selected_slice_plane(slice_axis, expected_shape):
    from src.data.dataset_3d import prepare_model_input

    volume = np.arange(5 * 7 * 9, dtype=np.float32).reshape(5, 7, 9)
    tensor = prepare_model_input(volume, (16, 32), slice_axis=slice_axis)

    assert tuple(tensor.shape) == expected_shape
    plane_shape = {
        "axial": tensor.shape[-2:],
        "coronal": (tensor.shape[1], tensor.shape[3]),
        "sagittal": (tensor.shape[1], tensor.shape[2]),
    }[slice_axis]
    assert all(int(value) % 16 == 0 for value in plane_shape)


def test_case_normalization_is_stable_across_full_and_patch_inputs():
    from src.data.dataset_3d import normalize_volume, prepare_model_input
    from src.inference import _case_normalized_input

    volume = np.linspace(-50.0, 250.0, 6 * 8 * 10, dtype=np.float32).reshape(
        6, 8, 10
    )
    config = {
        "data": {
            "modality": "mr",
            "intensity": {"percentiles": [0.5, 99.5]},
            "normalization_scope": "case_before_roi_or_patch",
        },
        "model": {"channel_policy": "repeat"},
    }
    normalized, ready = _case_normalized_input(volume, config)
    assert ready is True
    assert np.allclose(
        normalized,
        normalize_volume(
            volume,
            modality="mr",
            intensity={"percentiles": [0.5, 99.5]},
        ),
    )
    patch = normalized[2:5, 2:7, 3:8]
    prepared = prepare_model_input(
        patch,
        (16, 16),
        modality="mr",
        intensity={"percentiles": [0.5, 99.5]},
        pre_normalized=True,
    )
    assert float(prepared.min()) > 0.0
    assert float(prepared.max()) < 1.0

    legacy = prepare_model_input(
        patch,
        (16, 16),
        modality="mr",
        intensity={"percentiles": [0.5, 99.5]},
        pre_normalized=False,
    )
    assert float(legacy.min()) == pytest.approx(0.0)
    assert float(legacy.max()) == pytest.approx(1.0)


def test_dataset_pairs_msd_image_suffix_with_label_case_id(tmp_path):
    root = tmp_path / "dataset"
    (root / "imagesTr").mkdir(parents=True)
    (root / "labelsTr").mkdir()
    _save(root / "imagesTr" / "case_0000.nii.gz", np.zeros((4, 5, 6), dtype=np.float32), np.eye(4))
    _save(root / "labelsTr" / "case.nii.gz", np.zeros((4, 5, 6), dtype=np.uint8), np.eye(4))
    dataset = MedicalVolumeDataset(str(root), img_size=(8, 8), modality="ct")
    assert len(dataset) == 1
    assert dataset[0]["case_id"] == "case"


def test_dataset_rejects_unpaired_nifti_cases(tmp_path):
    root = tmp_path / "dataset"
    (root / "imagesTr").mkdir(parents=True)
    (root / "labelsTr").mkdir()
    _save(root / "imagesTr" / "case_0000.nii.gz", np.zeros((4, 5, 6), dtype=np.float32), np.eye(4))
    _save(root / "labelsTr" / "other.nii.gz", np.zeros((4, 5, 6), dtype=np.uint8), np.eye(4))
    with pytest.raises(RuntimeError, match="pairing mismatch"):
        MedicalVolumeDataset(str(root), img_size=(8, 8), modality="ct")


def test_foreground_patch_sampling_has_fixed_geometry(tmp_path):
    root = tmp_path / "dataset"
    (root / "imagesTr").mkdir(parents=True)
    (root / "labelsTr").mkdir()
    image = np.zeros((12, 16, 20), dtype=np.float32)
    label = np.zeros_like(image, dtype=np.uint8)
    label[8, 10, 15] = 1
    _save(root / "imagesTr" / "case.nii.gz", image, np.eye(4))
    _save(root / "labelsTr" / "case.nii.gz", label, np.eye(4))
    dataset = MedicalVolumeDataset(
        str(root),
        img_size=(16, 16),
        modality="ct",
        patch={"enabled": True, "size_zyx": [8, 10, 12], "foreground_probability": 1.0},
    )
    item = dataset[0]
    assert item["image"].shape == (1, 8, 16, 16)
    assert item["label"].shape == (8, 16, 16)
    assert int(item["label"].sum()) > 0


class _DummyBackbone(nn.Module):
    img_size = 8
    patch_size = 4
    out_indices = [0, 1]

    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(3, 4, 1, bias=False)

    def forward(self, x):
        value = self.projection(x)
        return [value, value * 2.0]


def test_encoder_retains_gradients_for_peft_path():
    backbone = _DummyBackbone()
    encoder = SliceWiseEncoder3D(backbone, channel_policy="repeat")
    features = encoder(torch.randn(1, 1, 3, 8, 8), slice_batch_size=2)
    sum(feature.sum() for feature in features).backward()
    assert backbone.projection.weight.grad is not None
    assert torch.count_nonzero(backbone.projection.weight.grad) > 0


def test_encoder_preserves_configured_resolution():
    backbone = _DummyBackbone()
    encoder = SliceWiseEncoder3D(backbone, channel_policy="repeat")
    features = encoder(torch.randn(1, 1, 2, 16, 24), slice_batch_size=2)
    assert features[0].shape[-2:] == (16, 24)


@pytest.mark.parametrize("loss_type", ["dice", "ce", "focal", "tversky", "dice_focal", "dice_ce", "dice_boundary"])
def test_all_configured_losses_follow_trainer_contract(loss_type):
    criterion = get_loss({"loss": {"type": loss_type}})
    output = criterion(torch.randn(1, 2, 2, 3, 4), torch.ones(1, 2, 3, 4, dtype=torch.long))
    assert isinstance(output, dict)
    assert torch.isfinite(output["loss"])


def test_focal_alpha_changes_foreground_weight():
    logits = torch.zeros(1, 2, 1, 2, 2)
    target = torch.ones(1, 1, 2, 2, dtype=torch.long)
    low = get_loss({"loss": {"type": "focal", "focal_alpha": 0.25}})(logits, target)["loss"]
    high = get_loss({"loss": {"type": "focal", "focal_alpha": 0.75}})(logits, target)["loss"]
    assert high > low


def test_3d_decoder_uses_group_norm_for_batch_one():
    decoder = SegFormer3DDecoder([8, 8, 8, 8], 2, proj_dim=8)
    assert not any(isinstance(module, nn.BatchNorm3d) for module in decoder.modules())
    assert any(isinstance(module, nn.GroupNorm) for module in decoder.modules())


def test_token_pyramid_decoder_creates_a_native_resolution_prediction():
    decoder = TokenPyramid3DDecoder([8, 8, 8, 8], 2, proj_dim=8)
    features = [torch.randn(1, 8, 3, 12, 16) for _ in range(4)]
    output = decoder(features, (1, 1, 3, 48, 64))
    assert output.shape == (1, 2, 3, 48, 64)


def test_feature_haar_augmentation_is_train_only_and_preserves_shapes():
    module = WaveletDetailAttenuation(probability=1.0, maximum_attenuation=0.5)
    features = [torch.randn(1, 4, 2, 5, 7)]
    module.eval()
    unchanged = module(features)[0]
    assert torch.equal(unchanged, features[0])
    torch.manual_seed(0)
    module.train()
    augmented = module(features)[0]
    assert augmented.shape == features[0].shape
    assert not torch.equal(augmented, features[0])


def _write_source_case(source_root, case_id, labels, shape=(16, 18, 20), affine=None):
    """Create a TotalSegmentator-style case: sXXXX/ct.nii.gz + segmentations/<label>.nii.gz."""
    if affine is None:
        affine = np.diag([1.5, 1.5, 2.0, 1.0])
    case_dir = source_root / case_id
    (case_dir / "segmentations").mkdir(parents=True)
    # Deterministic per-case CT so cache/no-cache comparisons are reproducible.
    rng = np.random.default_rng(int(case_id[1:]))
    ct = rng.integers(-1000, 1000, size=shape).astype(np.int16)
    _save(case_dir / "ct.nii.gz", ct, affine)
    for label_name, (z, y, x) in labels.items():
        mask = np.zeros(shape, dtype=np.uint8)
        mask[z:z + 2, y:y + 2, x:x + 2] = 1
        _save(case_dir / "segmentations" / (label_name + ".nii.gz"), mask, affine)


def test_image_intensity_percentiles_are_cached_and_lightweight(tmp_path):
    _image_intensity_percentiles.cache_clear()
    shape = (40, 50, 60)
    ct = np.random.default_rng(0).integers(-500, 500, size=shape).astype(np.int16)
    image_path = tmp_path / "ct.nii.gz"
    _save(image_path, ct, np.eye(4))
    resolved = str(image_path.resolve())

    first = _image_intensity_percentiles(resolved)
    second = _image_intensity_percentiles(resolved)

    # Only three ordered scalar percentiles are retained; no bulk array is cached.
    assert first == second
    assert len(first) == 3
    assert all(isinstance(value, float) for value in first)
    assert first[0] <= first[1] <= first[2]

    # The second call is served from cache rather than re-decompressing the gzip stream.
    info = _image_intensity_percentiles.cache_info()
    assert info.hits >= 1
    assert info.currsize == 1

    # Cached values equal a direct strided-subsample percentile computation.
    steps = tuple(max(1, int(np.ceil(length / 128.0))) for length in shape)
    sample = ct[::steps[0], ::steps[1], ::steps[2]].astype(np.float32)
    expected = tuple(float(value) for value in np.percentile(sample, [1, 50, 99]))
    assert first == pytest.approx(expected)


def test_discover_records_reuses_ct_percentiles_across_tasks(tmp_path):
    _image_intensity_percentiles.cache_clear()
    source = tmp_path / "source"
    source.mkdir()
    # Each case carries both a brain and a liver label on one shared CT file.
    for index in range(6):
        _write_source_case(source, "s{:04d}".format(index + 1), {"brain": (2, 3, 4), "liver": (6, 7, 8)})

    brain = discover_records(source, "brain")
    hits_after_first_task = _image_intensity_percentiles.cache_info().hits
    liver = discover_records(source, "liver")
    info = _image_intensity_percentiles.cache_info()

    assert len(brain) == len(liver) == 6
    # One cache entry per distinct CT, not per (task, CT) pair.
    assert info.currsize == 6
    # The second task rereads the same CTs, so every percentile lookup is a hit.
    assert info.hits - hits_after_first_task == 6
    brain_by_id = {record["case_id"]: record["image_intensity_percentiles"] for record in brain}
    liver_by_id = {record["case_id"]: record["image_intensity_percentiles"] for record in liver}
    assert brain_by_id == liver_by_id


def test_case_cache_produces_identical_materialized_content(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for index in range(6):
        _write_source_case(source, "s{:04d}".format(index + 1), {"liver": (5, 6, 7)})
    records = discover_records(source, "liver")

    plain = tmp_path / "plain"
    cached = tmp_path / "cached"
    cache_dir = tmp_path / "case_cache"
    manifest_plain = materialize_fold(records, "liver", plain, support_count=3, seed=1)
    manifest_cached = materialize_fold(records, "liver", cached, support_count=3, seed=1, case_cache=cache_dir)

    # Same selection and geometry regardless of the cache path.
    assert manifest_plain["fingerprint_sha256"] == manifest_cached["fingerprint_sha256"]
    for split in ("Tr", "Val"):
        for subdir in ("images", "labels"):
            reference_dir = plain / (subdir + split)
            for reference_file in sorted(reference_dir.glob("*.nii.gz")):
                relative = reference_file.relative_to(plain)
                cached_file = cached / relative
                assert cached_file.is_file()
                reference = nib.load(str(reference_file))
                candidate = nib.load(str(cached_file))
                assert np.array_equal(reference.get_fdata(), candidate.get_fdata())
                assert np.allclose(reference.affine, candidate.affine)
    # The cache actually retained reusable materialized image/label files.
    assert any((cache_dir / "images").glob("*.nii.gz"))
    assert any((cache_dir / "labels").glob("*.nii.gz"))


def test_case_cache_materializes_each_case_once_across_folds(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    for index in range(6):
        _write_source_case(source, "s{:04d}".format(index + 1), {"liver": (5, 6, 7)})
    records = discover_records(source, "liver")

    calls = []
    real_materialize_case = protocol.materialize_case

    def _counting(record, image_destination, label_destination):
        calls.append(record["case_id"])
        return real_materialize_case(record, image_destination, label_destination)

    monkeypatch.setattr(protocol, "materialize_case", _counting)

    cache_dir = tmp_path / "case_cache"
    # support_count=5 of 6 records means every fold touches all six cases.
    for fold in range(4):
        materialize_fold(records, "liver", tmp_path / "fold_{}".format(fold), support_count=5, seed=100 + fold, case_cache=cache_dir)

    # Every distinct case is materialized exactly once; later folds reuse the cache.
    assert sorted(set(calls)) == sorted(record["case_id"] for record in records)
    assert len(calls) == len(records)


def test_support_selection_does_not_depend_on_label_statistics():
    records = []
    for index in range(6):
        records.append(
            {
                "case_id": "s{:04d}".format(index),
                "image_features": [float(index), float(index * index)],
                "label_volume_ml": float(index),
            }
        )
    selected_a = [row["case_id"] for row in select_diverse_support(records, 3, seed=7)]
    for row in records:
        row["label_volume_ml"] *= 1000.0
    selected_b = [row["case_id"] for row in select_diverse_support(records, 3, seed=7)]
    assert selected_a == selected_b


def test_relative_base_config_is_resolved_from_child(tmp_path):
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text("model:\n  model_path: ./models/dinov3-vitb16\ntraining:\n  epochs: 2\n")
    child.write_text("_base_: base.yaml\ntraining:\n  epochs: 3\n")
    config = load_config(str(child), {})
    assert config["training"]["epochs"] == 3


def test_train_only_fingerprint_derives_a_bounded_patch_policy(tmp_path):
    root = tmp_path / "fold"
    (root / "imagesTr").mkdir(parents=True)
    (root / "labelsTr").mkdir()
    for case_id, location in (("s0001", (3, 8, 4)), ("s0002", (4, 9, 5))):
        image = np.zeros((16, 16, 8), dtype=np.float32)
        label = np.zeros_like(image, dtype=np.uint8)
        label[location[0], location[1], location[2]] = 1
        _save(root / "imagesTr" / (case_id + ".nii.gz"), image, np.eye(4))
        _save(root / "labelsTr" / (case_id + ".nii.gz"), label, np.eye(4))
    fingerprint = build_training_fingerprint(root, ["s0001", "s0002"], "adrenal_gland_right")
    policy = derive_policy(fingerprint)
    assert policy["patch"]["enabled"]
    assert policy["two_stage_candidate"]
    assert policy["input_size"] == [320, 320]
    coverage = fingerprint["summary"]["target_patch_coverage"]["256"]
    assert coverage["minimum_inplane_patches"] > 0.0


def _tiny_fingerprint(extent_zyx, shape_zyx=(229.0, 193.0, 194.0),
                      spacing_zyx=(1.5, 1.5, 1.5), foreground_fraction=9.2e-5,
                      min_patches_256=1.0):
    # A fingerprint summary shaped like the adrenal support set: a needle-like
    # target whose extent must drive the patch window.
    return {
        "summary": {
            "median_spacing_zyx": list(spacing_zyx),
            "spacing_iqr_zyx": [0.0, 0.0, 0.0],
            "median_shape_zyx": list(shape_zyx),
            "median_foreground_fraction": foreground_fraction,
            "median_foreground_extent_zyx": list(extent_zyx),
            "target_patch_coverage": {"256": {"minimum_inplane_patches": min_patches_256}},
        }
    }


def test_derive_policy_tiny_target_window_tracks_extent_times_margin():
    fingerprint = _tiny_fingerprint([18.0, 20.0, 13.0])
    policy = derive_policy(fingerprint)
    assert policy["rationale"]["tiny_target"] is True
    # Window ~ round(extent * 2.0), far smaller than the old hard 171 floor.
    assert policy["patch"]["size_zyx"] == [36, 40, 26]
    assert all(v < 100 for v in policy["patch"]["size_zyx"])


def test_derive_policy_tiny_target_respects_lower_bound():
    # A sub-voxel-thin target must not collapse the window below the DINO grid.
    fingerprint = _tiny_fingerprint([3.0, 4.0, 4.0])
    policy = derive_policy(fingerprint)
    z, y, x = policy["patch"]["size_zyx"]
    assert z >= 16 and y >= 24 and x >= 24


def test_derive_policy_tiny_target_clamped_to_shape():
    # extent * margin exceeding the volume must clamp to the median shape.
    fingerprint = _tiny_fingerprint([200.0, 300.0, 300.0], shape_zyx=(60.0, 80.0, 80.0))
    policy = derive_policy(fingerprint)
    z, y, x = policy["patch"]["size_zyx"]
    assert z <= 60 and y <= 80 and x <= 80


def test_derive_policy_non_tiny_target_keeps_legacy_window():
    # A small-but-not-tiny target (aorta-like: small_target True, tiny_target
    # False) must keep the legacy spacing formula so the confirmed aorta result
    # is never contaminated. min_patches in [3, 6) gives small=True, tiny=False.
    fingerprint = _tiny_fingerprint(
        [30.0, 60.0, 60.0], foreground_fraction=0.0046, min_patches_256=4.0,
    )
    policy = derive_policy(fingerprint)
    assert policy["rationale"]["small_target"] is True
    assert policy["rationale"]["tiny_target"] is False
    # Legacy formula: face-plane floored at 160, Z from 80/spacing.
    assert policy["patch"]["size_zyx"] == [53, 171, 171]


def test_prediction_bbox_zyx_returns_half_open_box():
    mask = np.zeros((20, 20, 20), dtype=np.int16)
    mask[5:9, 6:10, 7:12] = 1  # z in [5,9), y in [6,10), x in [7,12)
    bbox = prediction_bbox_zyx(mask)
    assert bbox == (5, 9, 6, 10, 7, 12)


def test_prediction_bbox_zyx_none_when_empty():
    assert prediction_bbox_zyx(np.zeros((8, 8, 8), dtype=np.int16)) is None


def test_expand_bbox_adds_margin_and_clamps_to_shape():
    # Box near both edges: low margin clamps to 0, high margin clamps to shape.
    bbox = (1, 5, 0, 4, 16, 20)
    expanded = expand_bbox(bbox, (3, 3, 3), (20, 20, 20))
    # z: 1-3 -> 0, 5+3 -> 8 ; y: 0-3 -> 0, 4+3 -> 7 ; x: 16-3 -> 13, 20+3 -> 20
    assert expanded == (0, 8, 0, 7, 13, 20)


def test_search_roi_size_from_coarse_bbox_plus_margin():
    from scripts.research.evaluate_two_stage import _search_roi_size
    coarse = np.zeros((60, 60, 60), dtype=np.int16)
    coarse[20:32, 18:34, 22:30] = 1  # bbox extent 12 x 16 x 8
    patch = (10, 10, 10)  # margin = patch // 2 = 5 per axis
    size = _search_roi_size(coarse, patch, coarse.shape)
    # bbox + 5 both sides: z 12+10=22, y 16+10=26, x 8+10=18, all >= patch.
    assert size == (22, 26, 18)


def test_search_roi_size_falls_back_to_window_when_coarse_empty():
    from scripts.research.evaluate_two_stage import _search_roi_size
    empty = np.zeros((40, 40, 40), dtype=np.int16)
    patch = (12, 20, 20)
    assert _search_roi_size(empty, patch, empty.shape) == patch


def test_refinement_search_roi_covers_offset_target():
    # A small window centered on an OFFSET coarse center misses a target that a
    # search-ROI tiling covers. This is the adrenal failure mode in miniature.
    from src.inference import predict_refinement_array

    data_xyz = np.zeros((24, 24, 24), dtype=np.float32)
    # Target foreground sits away from the volume centre.
    data_xyz[4:8, 4:8, 4:8] = 1.0
    grid = nib.Nifti1Image(data_xyz, np.eye(4))
    config = {
        "model": {"channel_policy": "repeat"},
        "data": {"img_size": [8, 8], "modality": "other",
                 "patch": {"size_zyx": [6, 6, 6], "inference_overlap": 0.5}},
        "inference": {},
    }
    # Coarse center is offset from the true target centroid (~5,5,5) to (9,9,9).
    center = (9, 9, 9)
    # Single window (search_size None) centered at (9,9,9) size 6 covers z,y,x
    # in [6,12) -> misses the [4,8) target substantially.
    single = predict_refinement_array(_DummyPredictionModel(), grid, config,
                                      torch.device("cpu"), center)
    # Search ROI large enough to reach the target from the offset center.
    searched = predict_refinement_array(_DummyPredictionModel(), grid, config,
                                        torch.device("cpu"), center,
                                        search_size_zyx=(16, 16, 16))
    target = np.zeros((24, 24, 24), dtype=bool)
    target[4:8, 4:8, 4:8] = True
    single_hit = int(np.logical_and(single > 0, target).sum())
    searched_hit = int(np.logical_and(searched > 0, target).sum())
    assert searched_hit > single_hit
    assert searched.shape == (24, 24, 24)


def test_derive_policy_tiny_window_lifts_foreground_fraction():
    # The new window must raise the achievable foreground fraction by orders of
    # magnitude versus the legacy window for the same ~928-voxel target.
    fg_voxels = 928
    tiny = derive_policy(_tiny_fingerprint([18.0, 20.0, 13.0]))
    new_vol = np.prod(tiny["patch"]["size_zyx"])
    legacy_vol = 53 * 171 * 171
    assert fg_voxels / new_vol > 20 * (fg_voxels / legacy_vol)


def test_surface_and_component_metrics_are_reported_in_physical_space():
    target = np.zeros((5, 7, 9), dtype=np.uint8)
    target[2, 3:5, 4:6] = 1
    result = binary_metrics(target, target, spacing=(2.0, 1.0, 1.0), surface_tolerance_mm=2.0)
    assert result["dice"] == 1.0
    assert result["surface_dice"] == 1.0
    assert result["lesion_f1"] == 1.0


class _DummyPredictionModel(nn.Module):
    patch_size = 4

    def forward(self, volume, spacing_zyx=None):
        foreground = volume[:, :1]
        return torch.cat([1.0 - foreground, foreground], dim=1)


def test_sliding_window_tta_and_multiscale_return_native_grid():
    image_xyz = np.zeros((8, 8, 4), dtype=np.float32)
    image_xyz[3:6, 3:6, 1:3] = 1.0
    grid = nib.Nifti1Image(image_xyz, np.eye(4))
    config = {
        "model": {"channel_policy": "repeat"},
        "data": {
            "img_size": [8, 8],
            "modality": "other",
            "patch": {
                "inference_sliding_window": True,
                "size_zyx": [2, 4, 4],
                "inference_overlap": 0.5,
            },
        },
        "inference": {"tta_axes": [[1]], "scales": [1.0, 1.5]},
    }
    prediction = predict_array(_DummyPredictionModel(), grid, config, torch.device("cpu"))
    assert prediction.shape == (4, 8, 8)
    assert set(np.unique(prediction)) <= {0, 1}


def test_roi_inference_restores_full_grid_and_keeps_largest_component():
    image_xyz = np.zeros((12, 12, 6), dtype=np.float32)
    image_xyz[6:10, 6:10, 2:5] = 10.0
    image_xyz[5:6, 5:6, 2:3] = 10.0
    grid = nib.Nifti1Image(image_xyz, np.eye(4))
    config = {
        "model": {"channel_policy": "repeat"},
        "data": {
            "img_size": [8, 8], "modality": "other", "patch": {},
            "roi": {"enabled": True, "normalized_zyx": [[0.0, 0.25, 0.25], [1.0, 1.0, 1.0]]},
        },
        "inference": {"keep_largest_component": True},
    }
    prediction = predict_array(_DummyPredictionModel(), grid, config, torch.device("cpu"))
    assert prediction.shape == (6, 12, 12)
    from scipy import ndimage
    _, count = ndimage.label(prediction)
    assert count <= 1


def test_study_candidates_are_explicit_and_confirmation_uses_three_seeds():
    import yaml

    plan = yaml.safe_load((PROJECT_ROOT / "config/research/multi_organ_study.yaml").read_text())
    study = plan["study"]
    assert len(study["confirmation_training_seeds"]) == 3
    assert all(candidate.get("changed_factor") for candidate in plan["candidates"])
    assert all(candidate["task"] in {"brain", "liver", "aorta", "adrenal_gland_right", "scapula_left"} for candidate in plan["candidates"])
    references = [candidate for candidate in plan["candidates"] if candidate.get("is_reference")]
    assert {candidate["task"] for candidate in references} == {
        "brain", "liver", "aorta", "adrenal_gland_right", "scapula_left"
    }


def _load_prepare_module():
    """Import the prepare script by path; scripts/research is not a package."""
    import importlib.util

    module_path = PROJECT_ROOT / "scripts" / "research" / "prepare_totalseg_benchmark.py"
    spec = importlib.util.spec_from_file_location("prepare_totalseg_benchmark", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_merges_manifest_across_per_task_invocations(tmp_path):
    prepare = _load_prepare_module()
    source = tmp_path / "source"
    source.mkdir()
    for index in range(6):
        _write_source_case(source, "s{:04d}".format(index + 1), {"liver": (5, 6, 7), "brain": (2, 3, 4)})
    output = tmp_path / "benchmark"

    # A per-task invocation must not erase another task already written to the manifest.
    prepare.build_benchmark(source=source, output=output, tasks=["liver"], support_pool_size=3, folds=2, seed=5)
    prepare.build_benchmark(source=source, output=output, tasks=["brain"], support_pool_size=3, folds=2, seed=5)

    manifest = json.loads((output / "benchmark_manifest.json").read_text())
    assert set(manifest["tasks"]) == {"liver", "brain"}
    assert len(manifest["tasks"]["liver"]) == 2
    assert len(manifest["tasks"]["brain"]) == 2


def test_prepare_is_resumable_and_reuses_case_cache(tmp_path, monkeypatch):
    prepare = _load_prepare_module()
    source = tmp_path / "source"
    source.mkdir()
    for index in range(6):
        _write_source_case(source, "s{:04d}".format(index + 1), {"liver": (5, 6, 7)})
    output = tmp_path / "benchmark"

    calls = []
    real_materialize_case = protocol.materialize_case

    def _counting(record, image_destination, label_destination):
        calls.append(record["case_id"])
        return real_materialize_case(record, image_destination, label_destination)

    monkeypatch.setattr(protocol, "materialize_case", _counting)

    # support_pool_size=5 of 6 records => every fold touches all six cases.
    prepare.build_benchmark(source=source, output=output, tasks=["liver"], support_pool_size=5, folds=4, seed=9)
    # A per-task case cache means each distinct case is materialized once, not once per fold.
    assert sorted(set(calls)) == sorted(record["case_id"] for record in discover_records(source, "liver"))
    assert len(calls) == 6

    # Re-running skips folds whose manifest already exists (no additional materialization).
    calls.clear()
    prepare.build_benchmark(source=source, output=output, tasks=["liver"], support_pool_size=5, folds=4, seed=9)
    assert calls == []

    manifest = json.loads((output / "benchmark_manifest.json").read_text())
    assert len(manifest["tasks"]["liver"]) == 4


def _load_modal_study_module():
    """Import the Modal entrypoint by path; it degrades gracefully without modal installed."""
    import importlib.util

    module_path = PROJECT_ROOT / "scripts" / "research" / "modal_study.py"
    spec = importlib.util.spec_from_file_location("modal_study", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_commits_after_each_task_not_once_at_end():
    modal_study = _load_modal_study_module()
    events = []
    tasks = ["brain", "liver", "aorta"]
    modal_study._prepare_tasks_with_checkpoint(
        tasks,
        run_task=lambda task: events.append(("run", task)),
        commit=lambda: events.append(("commit", None)),
    )
    # Each task's run is immediately followed by a commit, so a preempted worker
    # keeps every finished task rather than losing the whole batch.
    assert events == [
        ("run", "brain"), ("commit", None),
        ("run", "liver"), ("commit", None),
        ("run", "aorta"), ("commit", None),
    ]


def test_prepare_checkpoint_commits_partial_progress_on_failure():
    modal_study = _load_modal_study_module()
    committed = []

    def _run_task(task):
        if task == "aorta":
            raise RuntimeError("simulated preemption")

    with pytest.raises(RuntimeError, match="simulated preemption"):
        modal_study._prepare_tasks_with_checkpoint(
            ["brain", "liver", "aorta", "scapula_left"],
            run_task=_run_task,
            commit=lambda: committed.append(len(committed) + 1),
        )
    # brain and liver are committed before the failing task aborts the loop.
    assert committed == [1, 2]


def test_prepare_rebuilds_partial_fold_missing_manifest(tmp_path):
    prepare = _load_prepare_module()
    source = tmp_path / "source"
    source.mkdir()
    for index in range(6):
        _write_source_case(source, "s{:04d}".format(index + 1), {"liver": (5, 6, 7)})
    output = tmp_path / "benchmark"

    # Simulate a worker preempted mid-write: fold_00 has image dirs but no manifest.
    partial = output / "liver" / "fold_00"
    (partial / "imagesTr").mkdir(parents=True)
    (partial / "imagesTr" / "stale.nii.gz").write_bytes(b"partial")
    assert not (partial / "manifest.json").is_file()

    # Resume must rebuild the partial fold rather than raising FileExistsError.
    manifest = prepare.build_benchmark(
        source=source, output=output, tasks=["liver"], support_pool_size=5, folds=2, seed=9
    )
    assert (partial / "manifest.json").is_file()
    # The stale partial artifact is gone after the clean rebuild.
    assert not (partial / "imagesTr" / "stale.nii.gz").is_file()
    assert len(manifest["tasks"]["liver"]) == 2


def test_modal_image_pins_transformers_5x_for_dinov3vit():
    """The uploaded model is model_type=dinov3_vit; DINOv3ViTBackbone needs transformers>=5."""
    import re

    modal_study = _load_modal_study_module()
    packages = modal_study.IMAGE_PIP_PACKAGES
    transformers_pin = next((p for p in packages if p.startswith("transformers")), None)
    assert transformers_pin is not None, "modal image must pin transformers"
    # The lower bound must be a 5.x (or newer) release that exports DINOv3ViTBackbone.
    lower = re.search(r">=\s*(\d+)", transformers_pin)
    assert lower is not None, "transformers pin must declare a lower bound: {}".format(transformers_pin)
    assert int(lower.group(1)) >= 5, "transformers lower bound must be >=5, got {}".format(transformers_pin)
    # An upper bound that excludes 5.x (e.g. <5) is the exact bug that broke preflight.
    upper = re.search(r"<\s*(\d+)", transformers_pin)
    if upper is not None:
        assert int(upper.group(1)) >= 6, "transformers upper bound must not exclude 5.x: {}".format(transformers_pin)


def test_screen_job_specs_are_per_candidate_and_measurable():
    import yaml

    modal_study = _load_modal_study_module()
    plan = yaml.safe_load((PROJECT_ROOT / "config/research/multi_organ_study.yaml").read_text())
    specs = modal_study._phase_job_specs(plan, "screen")
    candidates = plan["candidates"]
    screen_seeds = plan["study"]["screening_training_seeds"]
    # One job per (candidate, screening seed); screen is fold_00, K=5 only.
    assert len(specs) == len(candidates) * len(screen_seeds)
    assert all(s["phase"] == "screen" for s in specs)
    assert all(s["fold"] == plan["study"]["screening_fold"] for s in specs)
    assert all(s["support_count"] == 5 for s in specs)
    assert {s["candidate_id"] for s in specs} == {c["id"] for c in candidates}
    # A measurement run slices the list; capping to 1 must yield exactly one job.
    assert len(specs[:1]) == 1


def test_confirm_job_specs_expand_folds_shots_seeds_for_selected():
    import yaml

    modal_study = _load_modal_study_module()
    plan = yaml.safe_load((PROJECT_ROOT / "config/research/multi_organ_study.yaml").read_text())
    study = plan["study"]
    # Simulate one winner id from a real task.
    confirmation_ids = ["brain_lora_r4"]
    specs = modal_study._phase_job_specs(plan, "confirm", confirmation_ids)
    expected = len(confirmation_ids) * len(study["confirmation_folds"]) * len(study["shot_counts"]) * len(study["confirmation_training_seeds"])
    assert len(specs) == expected
    assert all(s["phase"] == "confirm" for s in specs)
    assert {s["candidate_id"] for s in specs} == set(confirmation_ids)
    assert {s["fold"] for s in specs} == {int(f) for f in study["confirmation_folds"]}
    assert {s["support_count"] for s in specs} == {int(k) for k in study["shot_counts"]}


def test_confirm_job_specs_requires_ids():
    import yaml

    modal_study = _load_modal_study_module()
    plan = yaml.safe_load((PROJECT_ROOT / "config/research/multi_organ_study.yaml").read_text())
    import pytest as _pytest
    with _pytest.raises(ValueError):
        modal_study._phase_job_specs(plan, "confirm", None)


def test_confirmation_ids_excludes_weak_and_degenerate_but_keeps_passing():
    # A weak task (screen winner below guardrail) is reserved like a degenerate
    # one: its winner and reference are dropped from confirmation, yet a clearly
    # passing task still confirms. This is what lets aorta advance to folds
    # 1/2/3 without being blocked by scapula's weak screen.
    modal_study = _load_modal_study_module()
    plan = {
        "candidates": [
            {"id": "aorta_ref", "task": "aorta", "is_reference": True},
            {"id": "aorta_cldice", "task": "aorta"},
            {"id": "scap_ref", "task": "scapula_left", "is_reference": True},
            {"id": "scap_coronal", "task": "scapula_left"},
            {"id": "adrenal_base", "task": "adrenal_gland_right"},
        ],
    }
    selected_ids = ["aorta_cldice", "scap_coronal"]
    ids = modal_study._confirmation_ids(
        plan,
        selected_ids,
        weak_tasks=["scapula_left"],
        degenerate_tasks=["adrenal_gland_right"],
    )
    assert set(ids) == {"aorta_ref", "aorta_cldice"}
    assert ids == sorted(ids)


def test_confirmation_ids_without_exclusions_matches_select_union_references():
    modal_study = _load_modal_study_module()
    plan = {
        "candidates": [
            {"id": "aorta_ref", "task": "aorta", "is_reference": True},
            {"id": "aorta_cldice", "task": "aorta"},
        ],
    }
    ids = modal_study._confirmation_ids(plan, ["aorta_cldice"], weak_tasks=[], degenerate_tasks=[])
    assert set(ids) == {"aorta_ref", "aorta_cldice"}


def _load_run_ablations_module():
    import importlib.util

    module_path = PROJECT_ROOT / "scripts" / "research" / "run_ablations.py"
    spec = importlib.util.spec_from_file_location("run_ablations", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_epoch_budget_validation_rejects_non_positive_epochs():
    run_ablations = _load_run_ablations_module()
    # A zero/negative epoch budget silently skips the training loop and yields no
    # checkpoint; catch it up front with a clear error instead of a false failure.
    for bad in ({"screen_epochs": 0}, {"screen_epochs": -3}):
        with pytest.raises(ValueError, match="epochs"):
            run_ablations._validate_epoch_budget(bad, "screen")
    for bad in ({"confirmation_epochs": 0},):
        with pytest.raises(ValueError, match="epochs"):
            run_ablations._validate_epoch_budget(bad, "confirm")
    # Real study values pass unharmed.
    run_ablations._validate_epoch_budget({"screen_epochs": 25}, "screen")
    run_ablations._validate_epoch_budget({"confirmation_epochs": 80}, "confirm")
    # Absent key falls back to the base default and is allowed.
    run_ablations._validate_epoch_budget({}, "screen")


def test_regime_cells_produce_four_cells_with_correct_sampling_and_loss():
    from src.research.regime import regime_cells, REGIME_CELL_IDS
    fingerprint = {
        "summary": {
            "median_spacing_zyx": [3.0, 1.5, 1.5],
            "spacing_iqr_zyx": [0.1, 0.05, 0.05],
            "median_shape_zyx": [80.0, 256.0, 256.0],
            "median_foreground_fraction": 0.0085,
            "median_foreground_extent_zyx": [40.0, 120.0, 120.0],
            "target_patch_coverage": {"256": {"minimum_inplane_patches": 7.0}},
        }
    }
    cells = regime_cells(fingerprint)
    assert set(cells) == set(REGIME_CELL_IDS) == {"full_dice_ce", "full_dice_focal", "patch_dice_ce", "patch_dice_focal"}
    assert cells["full_dice_ce"]["data"]["patch"]["enabled"] is False
    assert cells["full_dice_focal"]["data"]["patch"]["enabled"] is False
    for cid in ("patch_dice_ce", "patch_dice_focal"):
        p = cells[cid]["data"]["patch"]
        assert p["enabled"] is True
        assert p["inference_sliding_window"] is True
        assert len(p["size_zyx"]) == 3 and all(isinstance(v, int) and v > 0 for v in p["size_zyx"])
        assert 0.0 < p["foreground_probability"] <= 1.0
    assert cells["full_dice_ce"]["loss"]["type"] == "dice_ce"
    assert cells["patch_dice_ce"]["loss"]["type"] == "dice_ce"
    assert cells["full_dice_focal"]["loss"]["type"] == "dice_focal"
    assert cells["patch_dice_focal"]["loss"]["type"] == "dice_focal"


def test_apply_regime_replaces_loss_and_sets_patch():
    from src.research.regime import apply_regime
    base = {
        "data": {"modality": "ct", "img_size": [256, 256],
                 "intensity": {"window": [-1024.0, 1024.0]}, "patch": {}},
        "loss": {"type": "dice_ce", "dice_weight": 0.5, "ce_weight": 0.5},
        "model": {"channel_policy": "repeat"},
    }
    regime = {
        "data": {"patch": {"enabled": True, "size_zyx": [32, 256, 256],
                           "foreground_probability": 0.67,
                           "inference_sliding_window": True, "inference_overlap": 0.5}},
        "loss": {"type": "dice_focal", "dice_weight": 0.7, "focal_weight": 0.3,
                 "focal_alpha": 0.75, "focal_gamma": 2.0},
    }
    out = apply_regime(base, regime)
    assert out["loss"] == regime["loss"]
    assert "ce_weight" not in out["loss"]
    assert out["data"]["patch"]["enabled"] is True
    assert out["data"]["patch"]["size_zyx"] == [32, 256, 256]
    assert out["data"]["intensity"]["window"] == [-1024.0, 1024.0]
    assert out["data"]["modality"] == "ct"
    assert base["loss"]["type"] == "dice_ce"
    assert base["data"]["patch"] == {}


def test_apply_regime_full_cell_disables_patch():
    from src.research.regime import apply_regime
    base = {"data": {"patch": {"enabled": True, "size_zyx": [8, 8, 8]}},
            "loss": {"type": "dice_focal"}}
    regime = {"data": {"patch": {"enabled": False}}, "loss": {"type": "dice_ce", "dice_weight": 0.5, "ce_weight": 0.5}}
    out = apply_regime(base, regime)
    assert out["data"]["patch"]["enabled"] is False
    assert out["loss"] == {"type": "dice_ce", "dice_weight": 0.5, "ce_weight": 0.5}


def test_regime_run_ids_cover_every_task_and_cell():
    import yaml
    run_ablations = _load_run_ablations_module()
    plan = yaml.safe_load((PROJECT_ROOT / "config/research/multi_organ_study.yaml").read_text())
    pairs = run_ablations._regime_run_ids(plan)
    tasks = {c["task"] for c in plan["candidates"]}
    from src.research.regime import REGIME_CELL_IDS
    assert len(pairs) == len(tasks) * len(REGIME_CELL_IDS)
    assert {t for t, _ in pairs} == tasks
    assert {cell for _, cell in pairs} == set(REGIME_CELL_IDS)


def test_regime_phase_dry_run_writes_regime_config_with_patch(tmp_path):
    import subprocess, sys as _sys, yaml
    prepare = _load_prepare_module()
    source = tmp_path / "source"; source.mkdir()
    for i in range(6):
        _write_source_case(source, "s{:04d}".format(i + 1), {"brain": (3, 8, 9)})
    bench = tmp_path / "bench"
    prepare.build_benchmark(source=source, output=bench, tasks=["brain"], support_pool_size=5, folds=1, seed=5)
    results = tmp_path / "results"
    plan = {
        "study": {"id": "t", "base_config": str(PROJECT_ROOT / "config/research/ct_fewshot_base.yaml"),
                  "benchmark_root": str(bench), "results_root": str(results),
                  "screening_fold": 0, "confirmation_folds": [0], "shot_counts": [5],
                  "screening_training_seeds": [1], "confirmation_training_seeds": [1],
                  "screen_epochs": 1, "confirmation_epochs": 1},
        "candidates": [{"id": "brain_base_frozen_segformer_256", "task": "brain", "is_reference": True,
                        "changed_factor": "x", "overrides": {}}],
    }
    plan_path = tmp_path / "plan.yaml"; plan_path.write_text(yaml.safe_dump(plan))
    subprocess.run([_sys.executable, "scripts/research/run_ablations.py", "--plan", str(plan_path),
                    "--phase", "regime", "--cell", "patch_dice_focal", "--dry-run"],
                   cwd=str(PROJECT_ROOT), check=True)
    cfg_path = results / "regime" / "brain" / "patch_dice_focal" / "fold_00" / "k5" / "seed_1" / "config.yaml"
    assert cfg_path.is_file()
    cfg = yaml.safe_load(cfg_path.read_text())
    assert cfg["data"]["patch"]["enabled"] is True
    assert cfg["data"]["patch"]["inference_sliding_window"] is True
    assert cfg["loss"]["type"] == "dice_focal"


def test_regime_override_for_returns_selected_cell():
    run_ablations = _load_run_ablations_module()
    fingerprint = {"summary": {"median_spacing_zyx":[3,1.5,1.5],"spacing_iqr_zyx":[0.1,0.05,0.05],
        "median_shape_zyx":[80,256,256],"median_foreground_fraction":0.0085,
        "median_foreground_extent_zyx":[40,120,120],
        "target_patch_coverage":{"256":{"minimum_inplane_patches":7.0}}}}
    selected = {"selected_by_task": {"brain": {"cell_id": "patch_dice_ce"}}}
    ov = run_ablations._regime_override_for(selected, "brain", fingerprint)
    assert ov["loss"]["type"] == "dice_ce"
    assert ov["data"]["patch"]["enabled"] is True
    # Fail closed: a task with no selected regime must raise, never silently
    # default to an arbitrary cell that could contaminate the formal screen.
    with pytest.raises((SystemExit, ValueError, KeyError)):
        run_ablations._regime_override_for(selected, "liver", fingerprint)


def _load_select_regime_module():
    import importlib.util
    p = PROJECT_ROOT / "scripts" / "research" / "select_regime_winners.py"
    spec = importlib.util.spec_from_file_location("select_regime_winners", p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def _load_select_screen_module():
    import importlib.util
    p = PROJECT_ROOT / "scripts" / "research" / "select_screening_winners.py"
    spec = importlib.util.spec_from_file_location("select_screening_winners", p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def _screen_result(root, candidate, task, dice, hd95, surface=0.2, seed=1, failed=False):
    d = root / "screen" / candidate / "fold_00" / "k5" / "seed_{}".format(seed)
    d.mkdir(parents=True)
    (d / "run_manifest.json").write_text(json.dumps({"candidate_id": candidate, "task": task}))
    (d / "evaluation.json").write_text(json.dumps({
        "mean_dice": dice, "mean_hd95_mm": hd95, "mean_surface_dice": surface,
        "mean_recall": 0.8, "empty_prediction_rate": 0.0,
    }))
    if failed:
        (d / "failed.json").write_text(json.dumps({"stage": "train", "returncode": 1}))


def test_screen_selector_requires_complete_clean_matrix(tmp_path):
    m = _load_select_screen_module()
    plan = {
        "study": {"screening_fold": 0, "screening_training_seeds": [1]},
        "candidates": [
            {"id": "brain_base", "task": "brain"},
            {"id": "brain_320", "task": "brain"},
        ],
    }
    _screen_result(tmp_path, "brain_base", "brain", 0.83, 9.0)
    with pytest.raises(SystemExit, match="(?i)incomplete"):
        m.select_screening(tmp_path, tmp_path / "out.json", plan)
    _screen_result(tmp_path, "brain_320", "brain", 0.85, 60.0, failed=True)
    with pytest.raises(SystemExit, match="failed.json"):
        m.select_screening(tmp_path, tmp_path / "out.json", plan)


def test_screen_selector_uses_tolerance_band_and_flags_weak(tmp_path):
    m = _load_select_screen_module()
    plan = {
        "study": {"screening_fold": 0, "screening_training_seeds": [1]},
        "screen_guardrails": {
            "minimum_dice_by_task": {"brain": 0.7, "scapula_left": 0.4},
            "maximum_hd95_mm_by_task": {"brain": 50.0, "scapula_left": 100.0},
        },
        "candidates": [
            {"id": "brain_base", "task": "brain"},
            {"id": "brain_320", "task": "brain"},
            {"id": "scapula_base", "task": "scapula_left"},
        ],
    }
    # 0.85 and 0.83 are within the registered 0.02 practical tie: HD95 wins.
    _screen_result(tmp_path, "brain_base", "brain", 0.83, 9.0)
    _screen_result(tmp_path, "brain_320", "brain", 0.85, 60.0)
    _screen_result(tmp_path, "scapula_base", "scapula_left", 0.33, 220.0, surface=0.15)
    payload = m.select_screening(tmp_path, tmp_path / "out.json", plan)
    assert payload["selected_by_task"]["brain"]["candidate_id"] == "brain_base"
    assert payload["selected_by_task"]["brain"]["weak"] is False
    assert payload["selected_by_task"]["scapula_left"]["weak"] is True
    assert payload["weak_tasks"] == ["scapula_left"]


def test_screen_selector_excludes_regime_degenerate_task(tmp_path):
    m = _load_select_screen_module()
    plan = {
        "study": {"screening_fold": 0, "screening_training_seeds": [1]},
        "candidates": [
            {"id": "brain_base", "task": "brain"},
            {"id": "adrenal_base", "task": "adrenal_gland_right"},
        ],
    }
    _screen_result(tmp_path, "brain_base", "brain", 0.8, 10.0)
    payload = m.select_screening(
        tmp_path, tmp_path / "out.json", plan,
        {"degenerate_tasks": ["adrenal_gland_right"]},
    )
    assert payload["selected_candidate_ids"] == ["brain_base"]
    assert payload["excluded_degenerate_tasks"] == ["adrenal_gland_right"]


def test_select_regime_picks_max_dice_and_flags_degenerate(tmp_path):
    m = _load_select_regime_module()
    root = tmp_path / "res"
    # Complete 4-cell matrix for brain (a clear winner) and liver (all degenerate).
    _regime_dir(root, "brain", "full_dice_ce", dice=0.0, hd95=455.0, recall=0.0, empty_rate=1.0)
    _regime_dir(root, "brain", "full_dice_focal", dice=0.72, hd95=9.0, recall=0.75, empty_rate=0.0)
    _regime_dir(root, "brain", "patch_dice_ce", dice=0.40, hd95=120.0, recall=0.5, empty_rate=0.0)
    _regime_dir(root, "brain", "patch_dice_focal", dice=0.55, hd95=60.0, recall=0.6, empty_rate=0.0)
    for c in ALL_CELLS:
        _regime_dir(root, "liver", c, dice=0.01, hd95=300.0, recall=0.01, empty_rate=0.5)
    out = tmp_path / "selected_regime.json"
    payload = m.select_regime(root, out, degenerate_threshold=0.05,
                              expected_tasks=["brain", "liver"], expected_cells=ALL_CELLS)
    assert payload["selected_by_task"]["brain"]["cell_id"] == "full_dice_focal"
    assert payload["selected_by_task"]["brain"]["degenerate"] is False
    assert payload["selected_by_task"]["liver"]["degenerate"] is True
    assert "liver" in payload["degenerate_tasks"]
    assert out.is_file()


def test_phase_job_specs_regime_covers_task_cell_grid():
    import yaml
    modal_study = _load_modal_study_module()
    plan = yaml.safe_load((PROJECT_ROOT / "config/research/multi_organ_study.yaml").read_text())
    specs = modal_study._phase_job_specs(plan, "regime")
    from src.research.regime import REGIME_CELL_IDS
    tasks = {c["task"] for c in plan["candidates"]}
    assert len(specs) == len(tasks) * len(REGIME_CELL_IDS)
    assert all(s["phase"] == "regime" for s in specs)
    assert {s["task"] for s in specs} == tasks
    assert {s["cell_id"] for s in specs} == set(REGIME_CELL_IDS)
    assert all(s["fold"] == plan["study"]["screening_fold"] for s in specs)


def test_epoch_budget_regime_phase_checks_screen_epochs():
    run_ablations = _load_run_ablations_module()
    # regime uses screen_epochs (Tier-0 runs on the screening fold), so a bad
    # screen_epochs must be rejected and confirmation_epochs must be irrelevant.
    with pytest.raises(ValueError, match="screen_epochs"):
        run_ablations._validate_epoch_budget({"screen_epochs": 0, "confirmation_epochs": 80}, "regime")
    # A valid screen_epochs passes even if confirmation_epochs is absent.
    run_ablations._validate_epoch_budget({"screen_epochs": 25}, "regime")
    # A bad confirmation_epochs must NOT block the regime phase.
    run_ablations._validate_epoch_budget({"screen_epochs": 25, "confirmation_epochs": 0}, "regime")


def test_screen_requires_regime_fails_closed(tmp_path):
    import subprocess, sys as _sys, yaml
    prepare = _load_prepare_module()
    source = tmp_path / "source"; source.mkdir()
    for i in range(6):
        _write_source_case(source, "s{:04d}".format(i + 1), {"brain": (3, 8, 9)})
    bench = tmp_path / "bench"
    prepare.build_benchmark(source=source, output=bench, tasks=["brain"], support_pool_size=5, folds=1, seed=5)
    plan = {
        "study": {"id": "t", "base_config": str(PROJECT_ROOT / "config/research/ct_fewshot_base.yaml"),
                  "benchmark_root": str(bench), "results_root": str(tmp_path / "results"),
                  "screening_fold": 0, "confirmation_folds": [0], "shot_counts": [5],
                  "screening_training_seeds": [1], "confirmation_training_seeds": [1],
                  "screen_epochs": 1, "confirmation_epochs": 1},
        "candidates": [{"id": "brain_base_frozen_segformer_256", "task": "brain", "is_reference": True,
                        "changed_factor": "x", "overrides": {}}],
    }
    plan_path = tmp_path / "plan.yaml"; plan_path.write_text(yaml.safe_dump(plan))
    # No --regime: must fail closed, not silently run on the raw baseline.
    r = subprocess.run([_sys.executable, "scripts/research/run_ablations.py", "--plan", str(plan_path),
                        "--phase", "screen", "--dry-run"], cwd=str(PROJECT_ROOT),
                       capture_output=True, text=True)
    assert r.returncode != 0
    assert "regime" in (r.stdout + r.stderr).lower()


def _regime_dir(root, task, cell, dice=None, hd95=None, recall=None, empty_rate=None, failed=False):
    d = root / "regime" / task / cell / "fold_00" / "k5" / "seed_1"
    d.mkdir(parents=True)
    (d / "run_manifest.json").write_text(json.dumps({"task": task, "cell_id": cell}))
    if failed:
        (d / "failed.json").write_text(json.dumps({"stage": "train", "returncode": 1}))
        return
    payload = {"mean_dice": dice, "mean_hd95_mm": hd95,
               "mean_recall": recall, "empty_prediction_rate": empty_rate,
               "mean_surface_dice": 0.5}
    (d / "evaluation.json").write_text(json.dumps(payload))


ALL_CELLS = ("full_dice_ce", "full_dice_focal", "patch_dice_ce", "patch_dice_focal")


def test_select_regime_rejects_incomplete_matrix(tmp_path):
    m = _load_select_regime_module()
    root = tmp_path / "res"
    # Only brain present, and only 2 of 4 cells -> incomplete.
    _regime_dir(root, "brain", "full_dice_ce", dice=0.0, hd95=455.0, recall=0.0, empty_rate=1.0)
    _regime_dir(root, "brain", "full_dice_focal", dice=0.83, hd95=9.4, recall=0.84, empty_rate=0.0)
    with pytest.raises((SystemExit, ValueError), match="(?i)incomplete|missing|expected"):
        m.select_regime(root, tmp_path / "out.json",
                        expected_tasks=["brain", "liver"], expected_cells=ALL_CELLS)


def test_select_regime_rejects_failed_or_nonfinite(tmp_path):
    m = _load_select_regime_module()
    root = tmp_path / "res"
    for c in ALL_CELLS:
        _regime_dir(root, "brain", c, dice=0.8, hd95=10.0, recall=0.8, empty_rate=0.0)
    # one cell failed
    import shutil
    shutil.rmtree(root / "regime" / "brain" / "patch_dice_focal")
    _regime_dir(root, "brain", "patch_dice_focal", failed=True)
    with pytest.raises((SystemExit, ValueError), match="(?i)fail|missing|incomplete"):
        m.select_regime(root, tmp_path / "out.json",
                        expected_tasks=["brain"], expected_cells=ALL_CELLS)


def test_select_regime_brain_prefers_lower_hd95_when_dice_close(tmp_path):
    m = _load_select_regime_module()
    root = tmp_path / "res"
    # two cells with near-equal Dice but very different HD95: the low-HD95 wins,
    # AND the near-tie must be flagged for more seeds.
    _regime_dir(root, "brain", "full_dice_ce", dice=0.0, hd95=455.0, recall=0.0, empty_rate=1.0)
    _regime_dir(root, "brain", "full_dice_focal", dice=0.835, hd95=9.4, recall=0.84, empty_rate=0.0)
    _regime_dir(root, "brain", "patch_dice_ce", dice=0.83, hd95=202.0, recall=0.82, empty_rate=0.0)
    _regime_dir(root, "brain", "patch_dice_focal", dice=0.67, hd95=320.0, recall=0.87, empty_rate=0.0)
    payload = m.select_regime(root, tmp_path / "out.json",
                              expected_tasks=["brain"], expected_cells=ALL_CELLS)
    sel = payload["selected_by_task"]["brain"]
    assert sel["cell_id"] == "full_dice_focal"          # lower HD95 breaks the near-tie
    assert sel["needs_more_seeds"] is True               # 0.835 vs 0.83 within 0.02


def test_select_regime_adrenal_uses_empty_rate_and_recall(tmp_path):
    m = _load_select_regime_module()
    root = tmp_path / "res"
    # adrenal: a higher-Dice cell that predicts empty on half the cases must lose
    # to a slightly lower-Dice cell that actually detects the organ.
    _regime_dir(root, "adrenal_gland_right", "full_dice_ce", dice=0.10, hd95=None, recall=0.05, empty_rate=0.9)
    _regime_dir(root, "adrenal_gland_right", "full_dice_focal", dice=0.32, hd95=15.0, recall=0.20, empty_rate=0.5)
    _regime_dir(root, "adrenal_gland_right", "patch_dice_ce", dice=0.30, hd95=12.0, recall=0.55, empty_rate=0.0)
    _regime_dir(root, "adrenal_gland_right", "patch_dice_focal", dice=0.31, hd95=11.0, recall=0.60, empty_rate=0.0)
    payload = m.select_regime(root, tmp_path / "out.json",
                              expected_tasks=["adrenal_gland_right"], expected_cells=ALL_CELLS)
    sel = payload["selected_by_task"]["adrenal_gland_right"]
    # winner must have low empty-rate and decent recall, not the empty-prone 0.32 cell
    assert sel["cell_id"] in ("patch_dice_ce", "patch_dice_focal")
    assert payload["selected_by_task"]["adrenal_gland_right"]["empty_prediction_rate"] == 0.0


def test_bounded_spawn_never_exceeds_concurrency_and_runs_all():
    modal_study = _load_modal_study_module()
    specs = list(range(10))
    launched = []
    # Simulated handles: each becomes "done" after being polled twice.
    class H:
        def __init__(self, spec): self.spec = spec; self.polls = 0
    inflight_peak = {"v": 0}
    state = {"handles": []}
    def launch(spec):
        launched.append(spec)
        h = H(spec)
        state["handles"].append(h)
        # peak concurrency = handles not yet done
        live = sum(1 for x in state["handles"] if x.polls < 2)
        inflight_peak["v"] = max(inflight_peak["v"], live)
        return h
    def poll(h):
        h.polls += 1
        return h.polls >= 2
    done = modal_study._bounded_spawn(specs, launch, poll, max_concurrency=4, poll_interval=0)
    assert sorted(launched) == specs           # every spec launched exactly once
    assert done == len(specs)
    assert inflight_peak["v"] <= 4             # concurrency bound respected


def test_bounded_spawn_rejects_bad_concurrency():
    modal_study = _load_modal_study_module()
    with pytest.raises((ValueError, AssertionError)):
        modal_study._bounded_spawn([1], lambda s: s, lambda h: True, max_concurrency=0, poll_interval=0)


def test_provenance_records_versions_and_no_false_git_dirty():
    run_ablations = _load_run_ablations_module()
    prov = run_ablations._provenance()
    # Package versions are pinned for reproducibility.
    assert "package_versions" in prov
    for pkg in ("torch", "transformers"):
        assert pkg in prov["package_versions"]
    # git_dirty must be a tri-state: True/False when git works, None when absent —
    # never a misleading False when git is simply unavailable (the container case).
    assert prov["git_dirty"] in (True, False, None)
    # When git_commit is None (no repo), git_dirty must NOT be False.
    if prov["git_commit"] is None:
        assert prov["git_dirty"] is None


def test_config_hash_is_stable_and_order_independent():
    run_ablations = _load_run_ablations_module()
    a = {"data": {"img_size": [256, 256]}, "loss": {"type": "dice_ce"}}
    b = {"loss": {"type": "dice_ce"}, "data": {"img_size": [256, 256]}}
    c = {"data": {"img_size": [320, 320]}, "loss": {"type": "dice_ce"}}
    ha, hb, hc = run_ablations._config_hash(a), run_ablations._config_hash(b), run_ablations._config_hash(c)
    assert ha == hb          # key order does not change the hash
    assert ha != hc          # a real config change does
    assert len(ha) == 64     # sha256 hex


def test_regime_skip_guard_reruns_on_config_change(tmp_path):
    import subprocess, sys as _sys, yaml
    prepare = _load_prepare_module()
    source = tmp_path / "source"; source.mkdir()
    for i in range(6):
        _write_source_case(source, "s{:04d}".format(i + 1), {"brain": (3, 8, 9)})
    bench = tmp_path / "bench"
    prepare.build_benchmark(source=source, output=bench, tasks=["brain"], support_pool_size=5, folds=1, seed=5)
    results = tmp_path / "results"
    run_root = results / "regime" / "brain" / "patch_dice_focal" / "fold_00" / "k5" / "seed_1"
    run_root.mkdir(parents=True)
    # Simulate a completed run from a DIFFERENT config (stale hash).
    (run_root / "evaluation.json").write_text(json.dumps({"mean_dice": 0.9}))
    (run_root / "run_manifest.json").write_text(json.dumps({"config_hash": "STALEHASH"}))
    plan = {
        "study": {"id": "t", "base_config": str(PROJECT_ROOT / "config/research/ct_fewshot_base.yaml"),
                  "benchmark_root": str(bench), "results_root": str(results),
                  "screening_fold": 0, "confirmation_folds": [0], "shot_counts": [5],
                  "screening_training_seeds": [1], "confirmation_training_seeds": [1],
                  "screen_epochs": 1, "confirmation_epochs": 1},
        "candidates": [{"id": "brain_base_frozen_segformer_256", "task": "brain", "is_reference": True,
                        "changed_factor": "x", "overrides": {}}],
    }
    plan_path = tmp_path / "plan.yaml"; plan_path.write_text(yaml.safe_dump(plan))
    r = subprocess.run([_sys.executable, "scripts/research/run_ablations.py", "--plan", str(plan_path),
                        "--phase", "regime", "--cell", "patch_dice_focal", "--dry-run"],
                       cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    # A stale hash must trigger a re-run (not a silent skip), refreshing the manifest.
    assert "Config changed" in (r.stdout + r.stderr)
    new_manifest = json.loads((run_root / "run_manifest.json").read_text())
    assert new_manifest["config_hash"] != "STALEHASH"
    assert len(new_manifest["config_hash"]) == 64


def test_select_regime_hd95_actually_breaks_near_ties(tmp_path):
    """The reviewer's A/B case: Dice 0.8001/HD95 300 vs Dice 0.8000/HD95 8.

    Lexicographic (dice,-hd95) wrongly picks A. A true tolerance-band rule treats
    the two as Dice-equivalent (within 0.02) and picks B for its far better HD95.
    """
    m = _load_select_regime_module()
    root = tmp_path / "res"
    _regime_dir(root, "brain", "full_dice_ce", dice=0.8001, hd95=300.0, recall=0.8, empty_rate=0.0)
    _regime_dir(root, "brain", "full_dice_focal", dice=0.8000, hd95=8.0, recall=0.8, empty_rate=0.0)
    _regime_dir(root, "brain", "patch_dice_ce", dice=0.4, hd95=120.0, recall=0.5, empty_rate=0.0)
    _regime_dir(root, "brain", "patch_dice_focal", dice=0.5, hd95=60.0, recall=0.6, empty_rate=0.0)
    payload = m.select_regime(root, tmp_path / "out.json",
                              expected_tasks=["brain"], expected_cells=ALL_CELLS)
    sel = payload["selected_by_task"]["brain"]
    assert sel["cell_id"] == "full_dice_focal"     # lower HD95 wins the Dice-tie band
    assert sel["needs_more_seeds"] is True          # two cells in the band => not Dice-separable


def test_select_regime_clear_dice_winner_not_flagged(tmp_path):
    """When the top Dice leads by more than the margin, no HD95 override, no flag."""
    m = _load_select_regime_module()
    root = tmp_path / "res"
    _regime_dir(root, "brain", "full_dice_ce", dice=0.0, hd95=455.0, recall=0.0, empty_rate=1.0)
    _regime_dir(root, "brain", "full_dice_focal", dice=0.835, hd95=9.4, recall=0.84, empty_rate=0.0)
    _regime_dir(root, "brain", "patch_dice_ce", dice=0.769, hd95=202.0, recall=0.82, empty_rate=0.0)
    _regime_dir(root, "brain", "patch_dice_focal", dice=0.675, hd95=320.0, recall=0.87, empty_rate=0.0)
    payload = m.select_regime(root, tmp_path / "out.json",
                              expected_tasks=["brain"], expected_cells=ALL_CELLS)
    sel = payload["selected_by_task"]["brain"]
    assert sel["cell_id"] == "full_dice_focal"      # leads by 0.066 > 0.02
    assert sel["needs_more_seeds"] is False          # sole member of the band


def test_poll_handle_semantics():
    modal_study = _load_modal_study_module()

    class NotReady:
        def get(self, timeout=0):
            raise TimeoutError()

    class Done:
        def get(self, timeout=0):
            return {"ok": True}

    class Failed:
        def get(self, timeout=0):
            raise RuntimeError("remote job crashed")

    # Still running -> not done (keep polling).
    assert modal_study._poll_handle(NotReady()) is False
    # Finished successfully -> done.
    assert modal_study._poll_handle(Done()) is True
    # Remote failure must NOT crash the driver: the cell wrote failed.json and the
    # selector's completeness gate will catch it. Treat as done (slot freed).
    assert modal_study._poll_handle(Failed()) is True


def test_screen_skips_degenerate_task_candidates_not_whole_run(tmp_path):
    """A degenerate task must be excluded from screening, but must NOT block the
    healthy organs (the old guard failed the whole run)."""
    import subprocess, sys as _sys, yaml
    prepare = _load_prepare_module()
    source = tmp_path / "source"; source.mkdir()
    for i in range(6):
        _write_source_case(source, "s{:04d}".format(i + 1), {"brain": (3, 8, 9), "liver": (5, 6, 7)})
    bench = tmp_path / "bench"
    prepare.build_benchmark(source=source, output=bench, tasks=["brain", "liver"], support_pool_size=5, folds=1, seed=5)
    results = tmp_path / "results"
    # selected_regime: brain healthy, liver degenerate.
    (results).mkdir(parents=True, exist_ok=True)
    regime = {
        "degenerate_tasks": ["liver"],
        "selected_by_task": {
            "brain": {"cell_id": "full_dice_focal"},
            "liver": {"cell_id": "full_dice_ce", "degenerate": True},
        },
    }
    regime_path = results / "selected_regime.json"; regime_path.write_text(json.dumps(regime))
    plan = {
        "study": {"id": "t", "base_config": str(PROJECT_ROOT / "config/research/ct_fewshot_base.yaml"),
                  "benchmark_root": str(bench), "results_root": str(results),
                  "screening_fold": 0, "confirmation_folds": [0], "shot_counts": [5],
                  "screening_training_seeds": [1], "confirmation_training_seeds": [1],
                  "screen_epochs": 1, "confirmation_epochs": 1},
        "candidates": [
            {"id": "brain_base_frozen_segformer_256", "task": "brain", "is_reference": True, "changed_factor": "x", "overrides": {}},
            {"id": "liver_base_frozen_segformer_256", "task": "liver", "is_reference": True, "changed_factor": "x", "overrides": {}},
        ],
    }
    plan_path = tmp_path / "plan.yaml"; plan_path.write_text(yaml.safe_dump(plan))
    r = subprocess.run([_sys.executable, "scripts/research/run_ablations.py", "--plan", str(plan_path),
                        "--phase", "screen", "--regime", str(regime_path), "--dry-run"],
                       cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    out = r.stdout + r.stderr
    # Must NOT crash: brain screens, liver is skipped with a clear note.
    assert r.returncode == 0, out
    assert "liver" in out.lower() and ("degenerate" in out.lower() or "skip" in out.lower())
    # brain's screen config was written; liver's was not.
    assert (results / "screen" / "brain_base_frozen_segformer_256").exists()
    assert not (results / "screen" / "liver_base_frozen_segformer_256").exists()


def test_screen_refuses_when_all_requested_tasks_degenerate(tmp_path):
    """If every candidate belongs to a degenerate task, there is nothing valid to
    screen and the run must fail closed rather than produce an empty result."""
    import subprocess, sys as _sys, yaml
    prepare = _load_prepare_module()
    source = tmp_path / "source"; source.mkdir()
    for i in range(6):
        _write_source_case(source, "s{:04d}".format(i + 1), {"liver": (5, 6, 7)})
    bench = tmp_path / "bench"
    prepare.build_benchmark(source=source, output=bench, tasks=["liver"], support_pool_size=5, folds=1, seed=5)
    results = tmp_path / "results"; results.mkdir(parents=True, exist_ok=True)
    regime = {"degenerate_tasks": ["liver"],
              "selected_by_task": {"liver": {"cell_id": "full_dice_ce", "degenerate": True}}}
    regime_path = results / "selected_regime.json"; regime_path.write_text(json.dumps(regime))
    plan = {
        "study": {"id": "t", "base_config": str(PROJECT_ROOT / "config/research/ct_fewshot_base.yaml"),
                  "benchmark_root": str(bench), "results_root": str(results),
                  "screening_fold": 0, "confirmation_folds": [0], "shot_counts": [5],
                  "screening_training_seeds": [1], "confirmation_training_seeds": [1],
                  "screen_epochs": 1, "confirmation_epochs": 1},
        "candidates": [{"id": "liver_base_frozen_segformer_256", "task": "liver", "is_reference": True, "changed_factor": "x", "overrides": {}}],
    }
    plan_path = tmp_path / "plan.yaml"; plan_path.write_text(yaml.safe_dump(plan))
    r = subprocess.run([_sys.executable, "scripts/research/run_ablations.py", "--plan", str(plan_path),
                        "--phase", "screen", "--regime", str(regime_path), "--dry-run"],
                       cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    assert r.returncode != 0
    assert "no " in (r.stdout + r.stderr).lower() or "degenerate" in (r.stdout + r.stderr).lower()


def test_memory_safe_slice_batch_forces_one_for_trainable_backbone():
    run_ablations = _load_run_ablations_module()
    # PEFT / full make the backbone trainable → full-volume activations OOM at
    # slice_batch_size=2 on a 24GB A10; force 1.
    for method in ("lora", "adapter", "full"):
        cfg = {"finetune": {"method": method}, "model": {"slice_batch_size": 2}}
        run_ablations._memory_safe_slice_batch(cfg)
        assert cfg["model"]["slice_batch_size"] == 1, method
    # Frozen backbone keeps whatever was set (no backbone grad to store).
    cfg = {"finetune": {"method": "frozen"}, "model": {"slice_batch_size": 2}}
    run_ablations._memory_safe_slice_batch(cfg)
    assert cfg["model"]["slice_batch_size"] == 2
    # Already-1 stays 1.
    cfg = {"finetune": {"method": "lora"}, "model": {"slice_batch_size": 1}}
    run_ablations._memory_safe_slice_batch(cfg)
    assert cfg["model"]["slice_batch_size"] == 1


def test_spatial_augmentation_handles_multichannel_images():
    """ct_windows / 2.5d produce C=3 images; spatial aug must not assume C=1."""
    from src.data.augmentation import VolumeAugmentation
    aug = VolumeAugmentation({"enabled": True, "spatial": {"rotation_deg": 7.0,
                              "scale_range": [0.95, 1.05], "translation_px": 4}}, seed=0)
    D, H, W = 5, 16, 16
    for C in (1, 3):
        item = {
            "image": torch.randn(C, D, H, W),
            "label": torch.zeros(D, H, W, dtype=torch.long),
            "case_id": "c", "image_path": "i", "label_path": "l",
            "spacing_zyx": torch.ones(3),
        }
        out = aug(item)
        assert out["image"].shape == (C, D, H, W), "C={} broke".format(C)
        assert out["label"].shape == (D, H, W)


def test_is_better_checkpoint_uses_train_loss_when_val_dice_tied():
    from src.training.trainer import is_better_checkpoint
    # Epoch 1 is always the initial best (a valid inference checkpoint must exist).
    assert is_better_checkpoint(0.0, 0.0, train_loss=0.70, best_train_loss=float("inf"),
                                min_delta=0.0, epoch=1, should_validate=True) is True
    # A genuine val-Dice improvement wins outright.
    assert is_better_checkpoint(0.30, 0.10, train_loss=0.50, best_train_loss=0.40,
                                min_delta=0.001, epoch=8, should_validate=True) is True
    # THE FIX: val Dice tied at 0 (needle target), but train loss dropped ->
    # this later checkpoint is better than the epoch-1 one.
    assert is_better_checkpoint(0.0, 0.0, train_loss=0.598, best_train_loss=0.705,
                                min_delta=0.001, epoch=10, should_validate=True) is True
    # Val tied and train loss did NOT improve -> not better.
    assert is_better_checkpoint(0.0, 0.0, train_loss=0.71, best_train_loss=0.70,
                                min_delta=0.001, epoch=12, should_validate=True) is False
    # Val Dice already positive; a train-loss drop must NOT override a val regression.
    assert is_better_checkpoint(0.05, 0.30, train_loss=0.10, best_train_loss=0.40,
                                min_delta=0.001, epoch=20, should_validate=True) is False


# ── LearnableZSmooth ─────────────────────────────────────────────────


class TestLearnableZSmooth:
    """Tests for the learnable z-axis Gaussian smoothing module."""

    def test_gaussian_initialisation(self):
        """Weights should be initialised as a normalised 1D Gaussian kernel."""
        smooth = LearnableZSmooth(num_channels=2, sigma=4.0, learnable=True)
        weight = smooth.conv.weight.data  # (2, 1, kZ, 1, 1)
        assert weight.shape[2] == 25  # int(4*3)*2 + 1
        # Each channel's kernel should sum to ≈ 1 (normalised).
        for c in range(2):
            kernel = weight[c, 0, :, 0, 0]
            assert abs(float(kernel.sum()) - 1.0) < 1e-6
            # Centre value should be the largest (Gaussian peak).
            assert torch.argmax(kernel) == weight.shape[2] // 2

    def test_learnable_flag_frozen(self):
        smooth = LearnableZSmooth(num_channels=1, sigma=3.0, learnable=False)
        assert not smooth.conv.weight.requires_grad

    def test_learnable_flag_trainable(self):
        smooth = LearnableZSmooth(num_channels=1, sigma=3.0, learnable=True)
        assert smooth.conv.weight.requires_grad

    def test_forward_preserves_shape(self):
        smooth = LearnableZSmooth(num_channels=3, sigma=4.0)
        x = torch.randn(1, 3, 32, 64, 64)
        out = smooth(x)
        assert out.shape == x.shape

    def test_forward_is_depth_only_convolution(self):
        """The smoothing should only mix along the depth axis (Z)."""
        smooth = LearnableZSmooth(num_channels=1, sigma=4.0, learnable=False)
        # Create a volume with a sharp step along Z: all zeros except slice 16 = 1
        x = torch.zeros(1, 1, 32, 8, 8)
        x[:, :, 16, :, :] = 1.0
        out = smooth(x)
        # The centre slice should still have the largest value after smoothing.
        assert out[0, 0, 16, 0, 0] > out[0, 0, 0, 0, 0]
        assert out[0, 0, 16, 0, 0] > out[0, 0, 31, 0, 0]
        # In-plane spatial dims should NOT be affected (groups=C independent along Z only).
        # All (H,W) positions in the same Z-slice should have identical values after smoothing
        # because the input was spatially uniform.
        for h in range(8):
            for w in range(8):
                assert abs(out[0, 0, 16, h, w] - out[0, 0, 16, 0, 0]) < 1e-6

    def test_gradient_flows_when_learnable(self):
        smooth = LearnableZSmooth(num_channels=1, sigma=4.0, learnable=True)
        x = torch.randn(1, 1, 16, 32, 32)
        out = smooth(x)
        loss = out.sum()
        loss.backward()
        assert smooth.conv.weight.grad is not None
        assert not torch.allclose(smooth.conv.weight.grad, torch.zeros_like(smooth.conv.weight.grad))

    def test_gradient_blocked_when_frozen(self):
        smooth = LearnableZSmooth(num_channels=1, sigma=4.0, learnable=False)
        x = torch.randn(1, 1, 16, 32, 32)
        out = smooth(x)
        # When the convolution weights are frozen, the output tensor has no
        # grad_fn and loss.backward() would fail.  Gradients still reach *x*
        # because the conv forward is a deterministic op, but the weight
        # gradient is never computed.
        assert smooth.conv.weight.grad is None  # never computed
        # Verify the output is still a valid float tensor.
        assert out.dtype == torch.float32


def test_mlp_probe_decoder_learnable_zsmooth_overrides_fixed():
    """When z_smooth_sigma > 0, MLPProbeDecoder3D should use LearnableZSmooth
    instead of the fixed non-parametric z-axis kernel."""
    import torch.nn.functional as F
    dec_learnable = MLPProbeDecoder3D([768], num_classes=2, z_smooth_sigma=4.0)
    dec_fixed = MLPProbeDecoder3D([768], num_classes=2, z_smooth_sigma=0.0)

    # Learnable version has the module; fixed version does not.
    assert dec_learnable._learnable_z_smooth is not None
    assert dec_fixed._learnable_z_smooth is None

    # Forward: both should produce same-shaped output.
    feat = torch.randn(1, 768, 8, 16, 16)
    shape = (1, 1, 16, 128, 128)
    out_learnable = dec_learnable([feat], shape)
    out_fixed = dec_fixed([feat], shape)
    assert out_learnable.shape == out_fixed.shape

    # With learnable=True and the kernel initialised as Gaussian, the learnable
    # output should be close (but not identical due to fixed vs parameterised)
    # to the fixed output on the same input.
    assert not torch.allclose(out_learnable, out_fixed)


def test_segformer_decoder_zsmooth_config():
    """SegFormer3DDecoder(z_smooth_sigma=0) should have z_smooth=None."""
    dec_no_smooth = SegFormer3DDecoder([768] * 4, num_classes=2, z_smooth_sigma=0.0)
    assert dec_no_smooth.z_smooth is None

    dec_smooth = SegFormer3DDecoder([768] * 4, num_classes=2, z_smooth_sigma=4.0)
    assert dec_smooth.z_smooth is not None
    assert isinstance(dec_smooth.z_smooth, LearnableZSmooth)

    # Forward both should work.
    feats = [torch.randn(1, 768, 8, 16, 16) for _ in range(4)]
    shape = (1, 1, 16, 128, 128)
    out_plain = dec_no_smooth(feats, shape)
    out_smooth = dec_smooth(feats, shape)
    assert out_plain.shape == out_smooth.shape
