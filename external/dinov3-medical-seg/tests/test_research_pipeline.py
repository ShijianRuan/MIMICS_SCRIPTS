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
from src.models.decoder_3d import SegFormer3DDecoder, TokenPyramid3DDecoder
from src.models.encoder_3d import SliceWiseEncoder3D
from src.models.feature_augmentation import WaveletDetailAttenuation
from src.inference import predict_array
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
    ov2 = run_ablations._regime_override_for(selected, "liver", fingerprint)
    assert ov2["data"]["patch"]["enabled"] is True
