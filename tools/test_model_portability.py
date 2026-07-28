#!/usr/bin/env python3
"""Regression tests for cross-machine AI model relocation."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
for value in (str(ROOT), str(ROOT / "tools")):
    if value not in sys.path:
        sys.path.insert(0, value)

import ai_model_bundle
import fewshot_pipeline
import nninteractive_task_common as task_common


def _nninteractive_model(root: Path) -> Path:
    model = root / "tasks" / "brain" / "models" / "v1"
    for name in task_common.MODEL_METADATA_FILES:
        path = model / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    checkpoint = model / "fold_0" / "checkpoint_final.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"portable-nninteractive")
    (model / "finetune_manifest.json").write_text(
        json.dumps(
            {
                "input_contract": task_common.NNINTERACTIVE_INPUT_CONTRACT,
                "runtime_verification": {
                    "verified": True,
                    "expected_parameter_fingerprint": "portable-runtime-v1",
                    "loaded_parameter_fingerprint": "portable-runtime-v1",
                },
                "validated_prompt_types": ["point"],
            }
        ),
        encoding="utf-8",
    )
    return model


class ModelPortabilityTests(unittest.TestCase):
    def test_nninteractive_registry_survives_workspace_relocation(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source_workspace"
            model = _nninteractive_model(source)
            registry = {
                "tasks": [
                    {
                        "task_id": "brain",
                        "task_name": "Brain",
                        "models": [
                            {
                                "model_id": "v1",
                                "model_dir": str(model),
                            }
                        ],
                        "recommended_model_id": "v1",
                    }
                ]
            }
            task_common.save_registry(source, registry)
            saved = task_common.load_registry(source)
            record = saved["tasks"][0]["models"][0]
            self.assertNotIn("model_dir", record)
            self.assertEqual(
                "tasks/brain/models/v1", record.get("model_relpath")
            )

            relocated = root / "windows_copy"
            shutil.copytree(source, relocated)
            task = task_common.find_task(relocated, "brain")
            selected = task_common.selected_model(relocated, "brain")
            resolved = task_common.resolve_registered_model_dir(
                relocated, selected, task
            )
            self.assertEqual(
                (relocated / "tasks" / "brain" / "models" / "v1").resolve(),
                resolved,
            )
            self.assertTrue(task_common.audit_model_dir(resolved)["compatible"])

    def test_nninteractive_bundle_import_rebuilds_local_registry(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source"
            model = _nninteractive_model(source)
            task_common.save_registry(
                source,
                {
                    "tasks": [
                        {
                            "task_id": "brain",
                            "task_name": "Brain",
                            "mask_names": ["Brain"],
                            "models": [
                                {
                                    "model_id": "v1",
                                    "model_dir": str(model),
                                    "checkpoint_sha256": task_common.sha256_file(
                                        model
                                        / "fold_0"
                                        / "checkpoint_final.pth"
                                    ),
                                }
                            ],
                            "recommended_model_id": "v1",
                        }
                    ]
                },
            )
            bundle = root / "brain.zip"
            ai_model_bundle.main(
                [
                    "export-nninteractive",
                    "--workspace",
                    str(source),
                    "--task-id",
                    "brain",
                    "--output",
                    str(bundle),
                ]
            )
            target = root / "target"
            ai_model_bundle.main(
                [
                    "import-nninteractive",
                    "--workspace",
                    str(target),
                    "--bundle",
                    str(bundle),
                    "--set-current",
                ]
            )
            task = task_common.find_task(target, "brain")
            model_record = task_common.selected_model(target, "brain")
            self.assertEqual("v1", task.get("recommended_model_id"))
            self.assertTrue(
                task_common.resolve_registered_model_dir(
                    target, model_record, task
                ).is_dir()
            )

    def test_dinov3_relative_manifest_survives_model_directory_move(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source_model"
            source.mkdir()
            (source / "model.pth").write_bytes(b"PK\x03\x04portable-checkpoint")
            (source / "config.yaml").write_text(
                "model:\n  model_path: models/dinov3-vits16\n",
                encoding="utf-8",
            )
            manifest = {
                "schema_version": "mimics_fewshot_model.v2",
                "model_id": "v1",
                "organ": "liver",
                "checkpoint": "model.pth",
                "config": "config.yaml",
            }
            manifest_path = source / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            relocated = root / "relocated_model"
            shutil.copytree(source, relocated)
            checked = fewshot_pipeline.validate_model_manifest(
                json.loads((relocated / "manifest.json").read_text()),
                relocated / "manifest.json",
            )
            self.assertEqual(
                (relocated / "model.pth").resolve(),
                Path(checked["checkpoint"]),
            )
            self.assertEqual(
                (relocated / "config.yaml").resolve(),
                Path(checked["config"]),
            )

    def test_legacy_dinov3_latest_manifest_relocates_by_model_id(self):
        with tempfile.TemporaryDirectory() as raw:
            organ_dir = Path(raw) / "models" / "liver"
            model_dir = organ_dir / "train_v1"
            model_dir.mkdir(parents=True)
            (model_dir / "model.pth").write_bytes(b"PK\x03\x04checkpoint")
            (model_dir / "config.yaml").write_text("model: {}\n", encoding="utf-8")
            old = {
                "model_id": "train_v1",
                "organ": "liver",
                "checkpoint": "E:/old/fewshot_models/models/liver/train_v1/model.pth",
                "config": "E:/old/fewshot_models/models/liver/train_v1/config.yaml",
            }
            latest = organ_dir / "latest.json"
            latest.write_text(json.dumps(old), encoding="utf-8")
            checked = fewshot_pipeline.validate_model_manifest(old, latest)
            self.assertEqual(
                (model_dir / "model.pth").resolve(),
                Path(checked["checkpoint"]),
            )

    def test_dinov3_bundle_import_builds_target_latest_pointer(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source"
            source.mkdir()
            (source / "model.pth").write_bytes(b"PK\x03\x04portable-checkpoint")
            (source / "config.yaml").write_text("model: {}\n", encoding="utf-8")
            manifest = {
                "schema_version": "mimics_fewshot_model.v2",
                "model_id": "v1",
                "organ": "liver",
                "organ_slug": "liver",
                "checkpoint": "model.pth",
                "config": "config.yaml",
            }
            (source / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            bundle = root / "dino.zip"
            ai_model_bundle.main(
                [
                    "export-dinov3",
                    "--manifest",
                    str(source / "manifest.json"),
                    "--output",
                    str(bundle),
                ]
            )
            workspace = root / "target"
            with mock.patch.object(ai_model_bundle, "register_global_model"):
                ai_model_bundle.main(
                    [
                        "import-dinov3",
                        "--workspace",
                        str(workspace),
                        "--bundle",
                        str(bundle),
                        "--set-latest",
                    ]
                )
            latest_path = workspace / "models" / "liver" / "latest.json"
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
            checked = fewshot_pipeline.validate_model_manifest(
                latest, latest_path
            )
            self.assertTrue(Path(checked["checkpoint"]).is_file())

    def test_registered_dinov3_config_has_no_run_local_paths(self):
        with tempfile.TemporaryDirectory() as raw:
            dino_root = Path(raw) / "dinov3"
            model = dino_root / "models" / "dinov3-vits16"
            model.mkdir(parents=True)
            config = {
                "_base_": ["/old/config.yaml"],
                "model": {"model_path": str(model)},
                "data": {"data_root": "E:/training/data"},
                "runtime": {
                    "status_path": "E:/jobs/status.json",
                    "input_contract": {"schema_version": "test"},
                },
                "training": {"experiment_root": "E:/runs/train_1"},
            }
            portable = fewshot_pipeline.portable_inference_config(
                config, dino_root
            )
            self.assertNotIn("_base_", portable)
            self.assertNotIn("data_root", portable["data"])
            self.assertNotIn("status_path", portable["runtime"])
            self.assertNotIn("experiment_root", portable["training"])
            self.assertEqual(
                "models/dinov3-vits16", portable["model"]["model_path"]
            )


if __name__ == "__main__":
    unittest.main()
