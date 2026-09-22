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






if __name__ == "__main__":
    unittest.main()
