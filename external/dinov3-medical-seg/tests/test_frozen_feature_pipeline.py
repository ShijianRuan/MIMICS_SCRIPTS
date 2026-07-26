from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
for candidate in (str(PROJECT_ROOT), str(REPO_ROOT / "tools")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from src.data.frozen_feature_slices import (  # noqa: E402
    normalize_percentile_volume,
    normalize_timeslice_casewise_volume,
    prepare_native_case,
    restore_native_prediction,
)
from src.models.decoder_2d import FrozenFeatureUNet2D  # noqa: E402
from src.models.backbone import ONNXDINOv3Backbone  # noqa: E402
from src.training.losses import DiceCELoss  # noqa: E402
from fewshot_pipeline import write_training_config  # noqa: E402
from fewshot_training_setup_ui import validate_options  # noqa: E402


def _recovered_pixel_unshuffle(values: torch.Tensor, factor: int) -> torch.Tensor:
    batch, channels, height, width = values.shape
    return (
        values.reshape(
            batch,
            channels,
            height // factor,
            factor,
            width // factor,
            factor,
        )
        .permute(0, 1, 3, 5, 2, 4)
        .reshape(batch, -1, height // factor, width // factor)
    )


def test_raw_skip_matches_recovered_reshape_order():
    values = torch.arange(2 * 1 * 32 * 48, dtype=torch.float32).reshape(2, 1, 32, 48)
    for factor in (2, 4, 8):
        expected = _recovered_pixel_unshuffle(values, factor)
        actual = FrozenFeatureUNet2D.raw_skip(values, factor)
        assert torch.equal(actual, expected)


def test_feature_decoder_forward_and_backward():
    torch.manual_seed(7)
    decoder = FrozenFeatureUNet2D([384], 2)
    embeddings = torch.randn(2, 384, 4, 4, requires_grad=True)
    raw = torch.rand(2, 1, 64, 64)
    logits = decoder.forward_slices(embeddings, raw)
    assert logits.shape == (2, 2, 64, 64)
    logits.mean().backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()


def test_feature_decoder_layer_contract():
    decoder = FrozenFeatureUNet2D([384], 2)
    assert decoder.bottleneck.conv1.weight.shape == (256, 384, 3, 3)
    assert decoder.up1.up.weight.shape == (256, 128, 2, 2)
    assert decoder.up1.body.conv1.weight.shape == (128, 192, 3, 3)
    assert decoder.up2.body.conv1.weight.shape == (64, 80, 3, 3)
    assert decoder.up3.body.conv1.weight.shape == (32, 36, 3, 3)
    assert decoder.up4.body.conv1.weight.shape == (16, 17, 3, 3)
    assert decoder.outc.weight.shape == (2, 16, 1, 1)
    assert decoder.bottleneck.norm1.affine
    assert not decoder.bottleneck.norm1.track_running_stats


def test_slice_losses_accept_two_dimensional_logits():
    logits = torch.randn(3, 2, 32, 32, requires_grad=True)
    labels = torch.randint(0, 2, (3, 32, 32))
    values = DiceCELoss()(logits, labels)
    values["loss"].backward()
    assert logits.grad is not None


def test_native_preprocess_and_restore_preserve_nifti_grid(tmp_path):
    image_xyz = np.zeros((11, 9, 3), dtype=np.float32)
    image_xyz[2:8, 3:7, 1] = 10.0
    label_xyz = np.zeros_like(image_xyz, dtype=np.uint8)
    label_xyz[3:7, 4:6, 1] = 1
    affine = np.array([
        [-1.2, 0.0, 0.0, 20.0],
        [0.0, 0.8, 0.0, -12.0],
        [0.0, 0.0, 2.5, 5.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    image_path = tmp_path / "image.nii.gz"
    label_path = tmp_path / "label.nii.gz"
    nib.save(nib.Nifti1Image(image_xyz, affine), image_path)
    nib.save(nib.Nifti1Image(label_xyz, affine), label_path)

    prepared = prepare_native_case(image_path, label_path)
    assert prepared["images"].shape == (3, 256, 256)
    assert prepared["labels"].shape == (3, 256, 256)
    restored = restore_native_prediction(prepared["labels"], prepared["image_nii"])
    assert restored.shape == image_xyz.shape
    assert np.allclose(restored.affine, affine, atol=1e-6)
    restored_values = np.asanyarray(restored.dataobj)
    assert np.count_nonzero(restored_values[:, :, 1]) > 0
    assert np.count_nonzero(restored_values[:, :, 0]) == 0


def test_timeslice_casewise_normalization_matches_instrumented_definition():
    values = np.arange(1000, dtype=np.float32).reshape(10, 10, 10)
    low, high = np.percentile(values, [0.5, 99.5])
    statistics = values[values > low]
    expected = (np.clip(values, low, high) - statistics.mean()) / statistics.std()
    actual = normalize_timeslice_casewise_volume(values)
    assert np.allclose(actual, expected.astype(np.float32))
    assert np.isclose(actual.min(), -1.7303283, atol=1e-6)
    assert np.isclose(actual.max(), 1.7129207, atol=1e-6)


def test_percentile_minmax_remains_available_for_native_pytorch():
    values = np.arange(1000, dtype=np.float32).reshape(10, 10, 10)
    expected_low, expected_high = np.percentile(values, [0.5, 99.5])
    expected = np.clip((values - expected_low) / (expected_high - expected_low), 0, 1)
    assert np.allclose(normalize_percentile_volume(values), expected.astype(np.float32))


def test_dynamic_onnx_contract_accepts_configured_512(monkeypatch, tmp_path):
    class ValueInfo:
        def __init__(self, name, shape):
            self.name = name
            self.shape = shape

    class Session:
        def __init__(self, _path, providers):
            self.providers = providers

        def get_inputs(self):
            return [ValueInfo("input", ["batch", 3, "height", "width"])]

        def get_outputs(self):
            return [ValueInfo("output", None)]

        def get_providers(self):
            return list(self.providers)

        def run(self, _outputs, feeds):
            values = feeds["input"]
            return [
                np.zeros(
                    (values.shape[0], 384, values.shape[2] // 16, values.shape[3] // 16),
                    dtype=np.float32,
                )
            ]

    runtime = SimpleNamespace(
        InferenceSession=Session,
        get_available_providers=lambda: ["CPUExecutionProvider"],
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime)
    path = tmp_path / "dynamic.onnx"
    path.write_bytes(b"dynamic-test-model")
    backbone = ONNXDINOv3Backbone(
        str(path),
        input_size=(512, 512),
        providers=["CPUExecutionProvider"],
    )
    output = backbone(torch.zeros(1, 3, 512, 512))[0]
    assert output.shape == (1, 384, 32, 32)


def test_fixed_onnx_contract_rejects_a_different_configured_size(monkeypatch, tmp_path):
    class ValueInfo:
        def __init__(self, name, shape):
            self.name = name
            self.shape = shape

    class Session:
        def __init__(self, _path, providers):
            self.providers = providers

        def get_inputs(self):
            return [ValueInfo("input", ["batch", 3, 256, 256])]

        def get_outputs(self):
            return [ValueInfo("output", None)]

        def get_providers(self):
            return list(self.providers)

    runtime = SimpleNamespace(
        InferenceSession=Session,
        get_available_providers=lambda: ["CPUExecutionProvider"],
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime)
    path = tmp_path / "fixed.onnx"
    path.write_bytes(b"fixed-test-model")
    try:
        ONNXDINOv3Backbone(
            str(path),
            input_size=(512, 512),
            providers=["CPUExecutionProvider"],
        )
    except RuntimeError as exc:
        assert "fixed input size" in str(exc)
    else:
        raise AssertionError("A fixed 256 encoder accepted a configured 512 input")


def _pipeline_args():
    return SimpleNamespace(
        batch_size=4,
        img_size="256,256",
        model_path=None,
        model_sha256="a" * 64,
        model_scale="vits16",
        finetune_method="frozen",
        decoder="feature_unet2d",
        lora_rank=8,
        lora_alpha=16,
        adapter_bottleneck=64,
        organ="brain",
        modality="mr",
        epochs=20,
        grad_accumulation=1,
        mixed_precision=False,
        lr=5e-4,
        weight_decay=1e-4,
        lr_scheduler="cosine",
        warmup_epochs=0,
        validation_interval=2,
        keep_last_checkpoints=2,
        sub_volume=False,
        sub_volume_size="32,256,256",
    )


def test_pipeline_writes_native_cached_slice_contract(tmp_path):
    config_path = tmp_path / "config.yaml"
    config = write_training_config(
        config_path,
        PROJECT_ROOT / "config" / "research" / "ct_fewshot_fast.yaml",
        tmp_path / "dataset",
        "test_run",
        _pipeline_args(),
        validation_enabled=True,
        strategy_overrides={
            "strategy": {"options": {"loss_type": "auto"}},
            "data": {"patch": {"enabled": True}},
            "loss": {"type": "dice_focal"},
        },
    )
    assert config["training"]["pipeline"] == "cached_slices"
    assert config["training"]["experiment_root"] == str(
        config_path.parent / "training_artifacts"
    )
    assert config["training"]["batch_size"] == 4
    assert config["training"]["grad_accumulation"] == 1
    assert config["training"]["scheduler"] == "cosine_epoch"
    assert config["training"]["cosine_min_lr_ratio"] == 0.1
    assert config["model"]["out_indices"] == [11]
    assert config["model"]["expected_sha256"] == "a" * 64
    assert config["model"]["slice_axis"] == "axial"
    assert config["model"]["input_normalization"] == "none"
    assert config["data"]["img_size"] == [256, 256]
    assert config["data"]["slice_normalization"] == "timeslice_casewise"
    assert config["data"]["target_spacing"] is None
    assert config["data"]["patch"] == {"enabled": False}
    assert config["loss"]["type"] == "ce"


def test_pytorch_feature_encoder_keeps_its_distinct_input_contract(tmp_path):
    args = _pipeline_args()
    args.encoder_backend = "pytorch"
    config = write_training_config(
        tmp_path / "config.yaml",
        PROJECT_ROOT / "config" / "research" / "ct_fewshot_fast.yaml",
        tmp_path / "dataset",
        "pytorch_feature_run",
        args,
        validation_enabled=True,
        strategy_overrides={
            "strategy": {"options": {"loss_type": "dice_focal"}},
            "loss": {"type": "dice_focal"},
        },
    )
    assert config["model"]["encoder_backend"] == "pytorch"
    assert config["model"]["input_normalization"] == "imagenet"
    assert config["data"]["slice_normalization"] == "percentile_minmax"
    assert config["data"]["native_grid"] is True
    assert config["data"]["target_spacing"] is None
    assert config["loss"] == {"type": "ce"}


def test_ui_validation_enforces_only_required_feature_constraints():
    values = validate_options({
        "strategy": "adaptive",
        "decoder": "feature_unet2d",
        "batch_size": 6,
        "epochs": 13,
        "grad_accumulation": 3,
        "lr": 0.0007,
        "weight_decay": 0.0002,
        "lr_scheduler": "cosine",
        "warmup_epochs": 5,
        "validation_interval": 3,
        "img_size": "256,256",
        "finetune_method": "lora",
        "model_scale": "vitb16",
        "sampling_mode": "patch",
        "channel_policy": "2_5d",
        "slice_axis": "coronal",
        "sub_volume": True,
        "sub_volume_size": "24,192,192",
        "mixed_precision": True,
    })
    assert values["epochs"] == 13
    assert values["lr"] == 0.0007
    assert values["batch_size"] == 6
    assert values["validation_interval"] == 3
    assert values["strategy"] == "full_volume"
    assert values["finetune_method"] == "frozen"
    assert values["model_scale"] == "vits16"
    assert values["img_size"] == "256,256"
    assert values["sampling_mode"] == "full"
    assert values["grad_accumulation"] == 1
    assert not values["mixed_precision"]


def test_ui_validation_locks_default_frozen_encoder_image_size():
    values = validate_options({
        "strategy": "full_volume",
        "decoder": "feature_unet2d",
        "batch_size": 4,
        "epochs": 2,
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "lr_scheduler": "cosine",
        "img_size": "512,512",
    })
    assert values["img_size"] == "256,256"

    custom = validate_options({
        "strategy": "full_volume",
        "decoder": "feature_unet2d",
        "batch_size": 4,
        "epochs": 2,
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "lr_scheduler": "cosine",
        "img_size": "512,512",
        "model_path": "/models/custom-vits16",
    })
    assert custom["img_size"] == "512,512"


def test_ui_validation_rejects_invalid_model_checksum():
    try:
        validate_options({
            "strategy": "full_volume",
            "decoder": "feature_unet2d",
            "batch_size": 4,
            "epochs": 2,
            "lr": 5e-4,
            "weight_decay": 1e-4,
            "lr_scheduler": "cosine",
            "img_size": "256,256",
            "model_sha256": "not-a-checksum",
        })
    except ValueError as exc:
        assert "SHA-256" in str(exc)
    else:
        raise AssertionError("Invalid model checksum was accepted")


def test_ui_validation_preserves_original_pytorch_strategy_options():
    values = validate_options({
        "strategy": "patch_focused",
        "decoder": "segformer3d",
        "batch_size": 1,
        "epochs": 20,
        "grad_accumulation": 2,
        "lr": 3e-4,
        "weight_decay": 0.01,
        "lr_scheduler": "cosine",
        "warmup_epochs": 3,
        "img_size": "224,224",
        "model_scale": "vitb16",
        "model_sha256": "a" * 64,
        "finetune_method": "lora",
        "sampling_mode": "patch",
        "channel_policy": "2_5d",
        "slice_axis": "coronal",
        "sub_volume": True,
        "sub_volume_size": "24,224,224",
        "mixed_precision": True,
    })
    assert values["strategy"] == "patch_focused"
    assert values["decoder"] == "segformer3d"
    assert values["model_scale"] == "vitb16"
    assert values["finetune_method"] == "lora"
    assert values["sampling_mode"] == "patch"
    assert values["channel_policy"] == "2_5d"
    assert values["slice_axis"] == "coronal"
    assert values["sub_volume"]
    assert values["mixed_precision"]
    assert values["model_sha256"] == ""
