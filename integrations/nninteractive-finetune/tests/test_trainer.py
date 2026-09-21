import json
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import torch

import nninteractive_finetune.trainer as trainer


class TinyInteractiveNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Module()
        self.encoder.stem = torch.nn.Sequential(torch.nn.Conv3d(8, 4, 3, padding=1))
        self.encoder.stages = torch.nn.ModuleList(
            [torch.nn.Sequential(torch.nn.Conv3d(4, 4, 3, padding=1))]
        )
        self.decoder = torch.nn.Module()
        self.decoder.stages = torch.nn.ModuleList(
            [torch.nn.Sequential(torch.nn.Conv3d(4, 4, 3, padding=1))]
        )
        self.decoder.seg_layers = torch.nn.ModuleList([torch.nn.Conv3d(4, 2, 1)])

    def forward(self, value):
        value = torch.relu(self.encoder.stem(value))
        value = torch.relu(self.encoder.stages[0](value))
        value = torch.relu(self.decoder.stages[0](value))
        return self.decoder.seg_layers[0](value)


class AlwaysBackgroundNetwork(torch.nn.Module):
    def forward(self, value):
        shape = (value.shape[0], 2, *value.shape[-3:])
        logits = torch.zeros(shape, dtype=value.dtype, device=value.device)
        logits[:, 0] = 1.0
        return logits


def _write_nifti(path: Path, array: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(array, np.eye(4)), str(path))


def test_validation_reports_empty_and_real_initial_auc_against_own_baseline():
    target = torch.zeros((1, 16, 16, 16), dtype=torch.long)
    target[:, 4:12, 4:12, 4:12] = 1
    initial = torch.zeros_like(target)
    initial[:, 5:11, 5:11, 5:11] = 1
    loader = [
        {
            "image": torch.ones((1, 1, 16, 16, 16)),
            "target": target,
            "initial_mask": initial,
            "has_initial_mask": torch.tensor([True]),
            "case_id": ["case"],
            "initial_mask_quality_bin": ["low"],
            "initial_mask_source_type": ["dinov3_prediction"],
        }
    ]
    metrics = trainer.validate(
        AlwaysBackgroundNetwork(),
        loader,
        torch.device("cpu"),
        {
            "point_radius": 2,
            "center_bias": 8.0,
            "interaction_decay": 0.9,
            "interaction_steps": 3,
            "validation_interaction_steps": [1, 3],
            "initial_mask_probability": 0.5,
            "validate_initial_masks": True,
            "correction_policy": "official_single",
        },
        batches=1,
        seed=7,
    )
    assert metrics["empty_mask_baseline_dice"] == 0.0
    assert metrics["empty_mask_trajectory_auc"] == 0.0
    assert metrics["empty_mask_auc_gain_vs_baseline"] == 0.0
    assert metrics["initial_mask_baseline_dice"] > 0.0
    assert metrics["initial_mask_trajectory_auc"] == 0.0
    assert metrics["initial_mask_auc_gain_vs_baseline"] < 0.0
    assert metrics["real_initial_mask_samples"] == 1
    assert metrics["initial_mask_quality_strata"]["low"]["case_count"] == 1
    assert (
        metrics["initial_mask_source_strata"]["dinov3_prediction"][
            "patch_sample_count"
        ]
        == 1
    )


def test_complete_training_flow_with_true_interaction_loop(tmp_path, monkeypatch):
    image = np.ones((64, 64, 64), dtype=np.float32)
    image += np.linspace(0, 1, image.size, dtype=np.float32).reshape(image.shape)
    label = np.zeros(image.shape, dtype=np.uint8)
    label[20:40, 22:42, 24:44] = 1
    image_path = tmp_path / "image.nii.gz"
    label_path = tmp_path / "label.nii.gz"
    _write_nifti(image_path, image)
    _write_nifti(label_path, label)
    manifest_path = tmp_path / "dataset.json"
    manifest_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "case",
                        "image": str(image_path),
                        "label": str(label_path),
                        "split": "train",
                    }
                ]
            }
        )
    )
    checkpoint_path = tmp_path / "base_checkpoint.pth"
    checkpoint_path.write_bytes(b"test checkpoint identity")

    monkeypatch.setattr(
        trainer,
        "load_model",
        lambda *_args, **_kwargs: SimpleNamespace(
            network=TinyInteractiveNetwork(),
            checkpoint_path=checkpoint_path,
        ),
    )

    def fake_export(_bundle, output_dir, summary, fold):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return {"training": summary, "fold": fold}

    monkeypatch.setattr(trainer, "export_model", fake_export)
    output_dir = tmp_path / "adapted_model"
    config = {
        "model": {
            "base_model_dir": str(tmp_path / "base"),
            "fold": "0",
            "checkpoint_name": "checkpoint_final.pth",
            "strategy": "full",
        },
        "data": {
            "manifest": str(manifest_path),
            "label_values": [1],
            "validation_fraction": 0.0,
            "patch_size": [64, 64, 64],
            "foreground_patch_probability": 1.0,
            "num_workers": 0,
            "prepared_cache_dir": str(tmp_path / "cache"),
            "keep_prepared_cache": True,
            "augmentation": {"enabled": False},
        },
        "prompts": {
            "mode": "clicks",
            "interaction_steps": 2,
            "point_radius": 4,
            "center_bias": 8.0,
            "interaction_decay": 0.9,
        },
        "training": {
            "output_dir": str(output_dir),
            "epochs": 1,
            "steps_per_epoch": 1,
            "batch_size": 1,
            "gradient_accumulation": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "mixed_precision": False,
            "seed": 7,
            "validation_batches": 1,
            "device": "cpu",
            "resume": True,
            "status_path": str(tmp_path / "status.json"),
            "cancel_path": str(tmp_path / "cancel.request"),
        },
    }
    result = trainer.train(config)
    assert result["training"]["epochs_completed"] == 1
    assert len(result["training"]["history"]) == 1
    status = json.loads((tmp_path / "status.json").read_text())
    # The CLI performs a fresh-network reload after train() releases its large
    # optimizer/network objects, then advances this to completed.
    assert status["status"] == "finalizing"
    assert status["phase"] == "awaiting_runtime_verification"
    assert not (
        output_dir.parent
        / "_nninteractive_finetune_work"
        / output_dir.name
        / "training.lock"
    ).exists()
