#!/usr/bin/env python3
"""Dependency-light tests for the nnInteractive task-model integration."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
for value in (str(ROOT), str(TOOLS)):
    if value not in sys.path:
        sys.path.insert(0, value)

import nninteractive_finetune_pipeline as pipeline
import nninteractive_task_common as common


def _model_dir(root: Path) -> Path:
    model = root / "model"
    for name in common.MODEL_METADATA_FILES:
        path = model / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    checkpoint = model / "fold_0" / "checkpoint_final.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"small-test-checkpoint")
    (model / "finetune_manifest.json").write_text(
        json.dumps(
            {
                "input_contract": common.NNINTERACTIVE_INPUT_CONTRACT,
                "runtime_verification": {
                    "verified": True,
                    "expected_parameter_fingerprint": "integration-runtime-v1",
                    "loaded_parameter_fingerprint": "integration-runtime-v1",
                },
                "validated_prompt_types": ["point"],
            }
        ),
        encoding="utf-8",
    )
    return model


class TaskModelIntegrationTests(unittest.TestCase):
    def test_training_pair_uses_the_same_minimal_mimics_grid(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            image_path = root / "source.nii.gz"
            label_path = root / "label.nii.gz"
            affine = np.eye(4)
            affine[0, 1] = 0.25
            image = np.arange(18 * 16 * 12, dtype=np.float32).reshape(
                (18, 16, 12)
            )
            label = np.zeros(image.shape, dtype=np.uint8)
            label[5:10, 4:9, 3:8] = 1
            nib.save(nib.Nifti1Image(image, affine), str(image_path))
            nib.save(nib.Nifti1Image(label, affine), str(label_path))

            prepared_image = pipeline._ensure_nifti_image(
                image_path, root / "prepared" / "image.nii.gz"
            )
            prepared_label = pipeline._ensure_binary_nifti_label(
                label_path,
                root / "prepared" / "label.nii.gz",
                prepared_image,
            )
            image_obj = nib.load(str(prepared_image))
            label_obj = nib.load(str(prepared_label))
            self.assertEqual(image_obj.shape, label_obj.shape)
            np.testing.assert_allclose(
                image_obj.affine, label_obj.affine, rtol=0.0, atol=1e-5
            )
            self.assertGreater(
                int(np.count_nonzero(np.asanyarray(label_obj.dataobj))), 0
            )

    def test_model_audit_can_skip_expensive_checksum(self):
        with tempfile.TemporaryDirectory() as value:
            model = _model_dir(Path(value))
            with mock.patch.object(
                common,
                "sha256_file",
                side_effect=AssertionError("checksum should not run"),
            ):
                audit = common.audit_model_dir(model, include_checksum=False)
        self.assertTrue(audit["compatible"])
        self.assertNotIn("checkpoint_sha256", audit)

    def test_unvalidated_first_model_is_not_selected_automatically(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            workspace = root / "workspace"
            model = _model_dir(root)
            request = {
                "task_id": "liver",
                "task_name": "Liver",
                "mask_names": ["Liver"],
                "model_id": "v1",
                "strategy": "clopa_in",
                "cases": [{"split": "train"}],
                "source_mode": "prepared",
            }
            registered, selected = pipeline._register_model(
                request,
                workspace,
                model,
                quality=None,
                validation_count=0,
            )
            task = common.find_task(workspace, "liver")
        self.assertEqual("unverified", registered["state"])
        self.assertFalse(selected)
        self.assertFalse(task.get("recommended_model_id"))

    def test_validated_improved_model_is_selected(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            workspace = root / "workspace"
            model = _model_dir(root)
            request = {
                "task_id": "liver",
                "task_name": "Liver",
                "mask_names": ["Liver"],
                "model_id": "v2",
                "strategy": "clopa_in",
                "cases": [
                    {"split": "train"},
                    {"split": "val"},
                    {"split": "val"},
                ],
                "source_mode": "prepared",
            }
            _registered, selected = pipeline._register_model(
                request,
                workspace,
                model,
                quality={
                    "qualifies": True,
                    "delta_auc": 0.04,
                    "severe_regressions": [],
                },
                validation_count=2,
            )
            task = common.find_task(workspace, "liver")
        self.assertTrue(selected)
        self.assertEqual("v2", task.get("recommended_model_id"))

    def test_stop_all_marker_is_understood_by_controller(self):
        with tempfile.TemporaryDirectory() as value:
            control = Path(value) / "control.json"
            control.write_text(
                json.dumps({"status": "stop_requested"}),
                encoding="utf-8",
            )
            self.assertEqual("stop", pipeline._control_action(control))

    def test_metric_history_survives_viewer_reopen(self):
        with tempfile.TemporaryDirectory() as value:
            status_path = Path(value) / "status.json"
            common.write_json_atomic(status_path, {"status": "training"})
            pipeline._copy_trainer_status(
                status_path,
                {
                    "status": "training",
                    "epoch": 1,
                    "epochs": 3,
                    "latest_epoch": {
                        "epoch": 1,
                        "train_loss": 0.8,
                        "validation": {"trajectory_auc": 0.5},
                    },
                },
            )
            pipeline._copy_trainer_status(
                status_path,
                {
                    "status": "training",
                    "epoch": 2,
                    "epochs": 3,
                    "latest_epoch": {
                        "epoch": 2,
                        "train_loss": 0.6,
                        "validation": {"trajectory_auc": 0.65},
                    },
                },
            )
            status = common.read_json(status_path, {})
            log_text = (status_path.parent / "job.log").read_text(
                encoding="utf-8"
            )
        self.assertEqual([1, 2], [row["epoch"] for row in status["metrics_history"]])
        self.assertIn("Epoch 2/3", log_text)
        self.assertIn("validation AUC 0.6500", log_text)


class DiagnoseJobTests(unittest.TestCase):
    """Phase D: aggregated failure diagnosis for Model Center."""

    def _failed_job(self, root: Path, error: str = "", **status_extra) -> Path:
        job = root / "job_001"
        job.mkdir()
        status = {
            "status": "failed",
            "phase": "failed",
            "error": error,
            "traceback": (
                "Traceback (most recent call last):\n"
                "  File \"outer.py\", line 10, in outer\n"
                "  File \"inner.py\", line 42, in inner\n"
                "RuntimeError: boom"
            ),
        }
        status.update(status_extra)
        common.write_json_atomic(job / "status.json", status)
        common.write_json_atomic(
            job / "request.json",
            {
                "task_id": "liver",
                "epochs": 10,
                "strategy": "clopa_in",
                "cases": [{"split": "train"}] * 3,
            },
        )
        (job / "job.log").write_text(
            "\n".join("line {}".format(i) for i in range(100))
            + "\nTraining failed.",
            encoding="utf-8",
        )
        (job / "trainer.log").write_text(
            "epoch 1 ok\nepoch 2 crash\n", encoding="utf-8"
        )
        common.write_json_atomic(
            job / "trainer_status.json", {"status": "crashed"}
        )
        return job

    def test_diagnosis_aggregates_error_stage_chain_and_log_tails(self):
        with tempfile.TemporaryDirectory() as value:
            job = self._failed_job(
                Path(value), error="RuntimeError: label export failed"
            )
            report = pipeline.diagnose_job(str(job))
        self.assertEqual("nninteractive_job_diagnosis.v1", report["schema_version"])
        self.assertEqual("failed", report["status"])
        self.assertEqual("RuntimeError: label export failed", report["error"])
        self.assertTrue(report["error_line"].strip().startswith('File "inner.py"'))
        self.assertEqual(["failed", "crashed"], report["stage_chain"])
        self.assertTrue(report["job_log_tail"].endswith("Training failed."))
        self.assertTrue(report["trainer_log_tail"].endswith("epoch 2 crash"))
        self.assertTrue(report["artifacts"]["job.log"])
        self.assertFalse(report["artifacts"]["control.json"])
        self.assertEqual(3, report["request_summary"]["case_count"])

    def test_diagnosis_tail_drops_partial_first_line(self):
        with tempfile.TemporaryDirectory() as value:
            job = self._failed_job(Path(value), error="RuntimeError: x")
            report = pipeline.diagnose_job(str(job))
        self.assertTrue(report["job_log_tail"].startswith("line "))
        self.assertNotIn("\r", report["job_log_tail"])

    def test_diagnosis_hint_maps_gpu_memory_errors(self):
        with tempfile.TemporaryDirectory() as value:
            job = self._failed_job(
                Path(value), error="RuntimeError: CUDA out of memory"
            )
            report = pipeline.diagnose_job(str(job))
        self.assertIn("GPU memory", report["hint"])

    def test_diagnosis_degrades_for_missing_directory(self):
        with tempfile.TemporaryDirectory() as value:
            report = pipeline.diagnose_job(
                str(Path(value) / "does_not_exist")
            )
        self.assertEqual("", report["error"])
        self.assertEqual([], report["stage_chain"])
        self.assertEqual("", report["job_log_tail"])
        self.assertIn("No error was recorded", report["hint"])

    def test_diagnosis_reports_stopped_jobs_as_not_failures(self):
        with tempfile.TemporaryDirectory() as value:
            job = self._failed_job(
                Path(value), error="", status="cancelled"
            )
            report = pipeline.diagnose_job(str(job))
        self.assertIn("stopped by user request", report["hint"])


if __name__ == "__main__":
    unittest.main()
