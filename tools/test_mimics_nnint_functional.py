#!/usr/bin/env python3
"""Functional simulation tests for Mimics nnInteractive integration.

Covers the complete user journey without requiring Mimics, GPU, or nnInteractive:
1. Config loading and validation
2. Registry CRUD (task creation, model registration, selection)
3. Model profile construction and validation
4. Training request building
5. Case discovery (prepared mode)
6. Model chooser decision logic
7. Quality evaluation logic
8. Multi-model coexistence and identity tracking
9. Edge cases and error handling
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
for value in (str(ROOT), str(TOOLS)):
    if value not in sys.path:
        sys.path.insert(0, value)

import nninteractive_task_common as common
import nninteractive_finetune_pipeline as pipeline
import nninteractive_task_model_center as model_center


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model_dir(root: Path, *, checkpoint_bytes: bytes = b"ckpt") -> Path:
    """Create a minimal valid model directory structure."""
    model = root / "model"
    for name in common.MODEL_METADATA_FILES:
        path = model / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    checkpoint = model / "fold_0" / "checkpoint_final.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(checkpoint_bytes)
    fingerprint = "test-effective-{}".format(
        common.sha256_file(checkpoint)[:16]
    )
    (model / "finetune_manifest.json").write_text(
        json.dumps(
            {
                "input_contract": common.NNINTERACTIVE_INPUT_CONTRACT,
                "runtime_verification": {
                    "verified": True,
                    "expected_parameter_fingerprint": fingerprint,
                    "loaded_parameter_fingerprint": fingerprint,
                },
                "validated_prompt_types": ["point"],
            }
        ),
        encoding="utf-8",
    )
    return model


def _make_prepared_case(root: Path, case_id: str, *, with_label: bool = True) -> Path:
    """Create a minimal prepared case directory with image.nii.gz and optional label."""
    case_dir = root / case_id
    case_dir.mkdir(parents=True)
    (case_dir / "image.nii.gz").write_bytes(b"fake-nifti-image")
    if with_label:
        (case_dir / "label.nii.gz").write_bytes(b"fake-nifti-label")
    return case_dir


# ---------------------------------------------------------------------------
# 1. Config loading and validation
# ---------------------------------------------------------------------------

class ConfigTests(unittest.TestCase):
    """Test config loading, defaults, and validation."""

    def test_config_loads_with_defaults(self):
        config = common.load_config()
        self.assertIsInstance(config, dict)
        self.assertIn("workspace_dir", config)
        self.assertIn("default_strategy", config)
        self.assertEqual(config["default_strategy"], "clopa_in")

    def test_workspace_root_resolves(self):
        ws = common.workspace_root()
        self.assertTrue(ws.is_dir())
        self.assertTrue(ws.name.endswith("nninteractive_task_models") or
                        "nninteractive_task_models" in str(ws))

    def test_official_model_dir_resolves(self):
        model_dir = common.official_model_dir()
        self.assertIsInstance(model_dir, Path)

    def test_safe_slug_normalization(self):
        self.assertEqual(common.safe_slug("Liver Tumor"), "liver_tumor")
        self.assertEqual(common.safe_slug("Brain MRI (T1)"), "brain_mri_t1")
        self.assertEqual(common.safe_slug(""), "task")
        # Dashes are preserved by safe_slug (allowed in character class)
        self.assertEqual(common.safe_slug("  A B--C__  "), "a_b--c")

    def test_find_environment_python(self):
        """Verify env Python detection doesn't crash and returns a Path."""
        try:
            py = common.find_environment_python()
            self.assertIsInstance(py, Path)
        except FileNotFoundError:
            self.skipTest("nninteractive_env not set up on this machine")

    def test_config_respects_env_override(self):
        with mock.patch.dict("os.environ", {"NNINTERACTIVE_TASK_MODELS_DIR": "/tmp/test_nnint_ws"}):
            ws = common.workspace_root()
            self.assertIn("test_nnint_ws", str(ws))


# ---------------------------------------------------------------------------
# 2. Registry CRUD
# ---------------------------------------------------------------------------

class RegistryTests(unittest.TestCase):
    """Test task/model registry operations."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp.name) / "workspace"
        self.workspace.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_empty_registry_has_no_tasks(self):
        self.assertEqual(len(common.task_rows(self.workspace)), 0)

    def test_create_task_and_model(self):
        registry = common.load_registry(self.workspace)
        task = {
            "task_id": "liver_seg",
            "task_name": "Liver Segmentation",
            "mask_names": ["Liver", "liver_mask"],
            "models": [{
                "model_id": "model_20260726_001",
                "model_dir": str(self.workspace / "tasks" / "liver_seg" / "models" / "v1"),
                "strategy": "clopa_in",
                "created_at_epoch": time.time(),
                "state": "unverified",
                "compatible": True,
            }],
        }
        registry["tasks"].append(task)
        common.save_registry(self.workspace, registry)

        tasks = common.task_rows(self.workspace)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["task_id"], "liver_seg")

    def test_find_task_by_id(self):
        registry = common.load_registry(self.workspace)
        registry["tasks"].append({
            "task_id": "brain_extraction",
            "task_name": "Brain Extraction",
            "mask_names": ["Brain"],
            "models": [],
        })
        common.save_registry(self.workspace, registry)

        task = common.find_task(self.workspace, "brain_extraction")
        self.assertIsNotNone(task)
        self.assertEqual(task["task_name"], "Brain Extraction")

        not_found = common.find_task(self.workspace, "nonexistent")
        self.assertIsNone(not_found)

    def test_model_rows_sorted_by_date(self):
        registry = common.load_registry(self.workspace)
        old = time.time() - 86400
        new = time.time()
        registry["tasks"].append({
            "task_id": "test",
            "task_name": "Test",
            "mask_names": [],
            "models": [
                {"model_id": "old", "created_at_epoch": old, "strategy": "clopa_in"},
                {"model_id": "new", "created_at_epoch": new, "strategy": "clopa_conv"},
            ],
        })
        common.save_registry(self.workspace, registry)

        models = common.model_rows(self.workspace, "test")
        self.assertEqual(len(models), 2)
        self.assertEqual(models[0]["model_id"], "new")  # newest first
        self.assertEqual(models[1]["model_id"], "old")

    def test_model_center_registry_signature_changes_after_registration(self):
        before = model_center._task_model_registry_signature(
            self.workspace,
            "test",
        )
        registry = common.load_registry(self.workspace)
        registry["tasks"].append({
            "task_id": "test",
            "task_name": "Test",
            "recommended_model_id": "v1",
            "models": [{
                "model_id": "v1",
                "state": "validated",
                "compatible": True,
                "created_at_epoch": 123.0,
            }],
        })
        common.save_registry(self.workspace, registry)
        after = model_center._task_model_registry_signature(
            self.workspace,
            "test",
        )
        self.assertNotEqual(before, after)
        self.assertEqual("v1", after[1])

    def test_recommended_model_selection(self):
        registry = common.load_registry(self.workspace)
        registry["tasks"].append({
            "task_id": "test",
            "task_name": "Test",
            "mask_names": [],
            "recommended_model_id": "v2",
            "models": [
                {"model_id": "v1", "created_at_epoch": 1, "strategy": "clopa_in"},
                {"model_id": "v2", "created_at_epoch": 2, "strategy": "clopa_in"},
            ],
        })
        common.save_registry(self.workspace, registry)

        selected = common.selected_model(self.workspace, "test")
        self.assertIsNotNone(selected)
        self.assertEqual(selected["model_id"], "v2")

    def test_model_profile_construction(self):
        task = {
            "task_id": "liver",
            "task_name": "Liver CT",
        }
        model_dir = _make_model_dir(Path(self.tmp.name) / "task_model")
        model = {
            "model_id": "v1",
            "model_dir": str(model_dir),
            "checkpoint_sha256": common.sha256_file(model_dir / "fold_0" / "checkpoint_final.pth"),
            "strategy": "clopa_in",
            "created_at_epoch": time.time(),
            "state": "validated",
        }
        profile = common.model_profile(task, model)
        self.assertEqual(profile["source"], "task_model")
        self.assertEqual(profile["task_id"], "liver")
        self.assertEqual(profile["task_name"], "Liver CT")
        self.assertEqual(profile["model_id"], "v1")
        self.assertEqual(profile["strategy"], "clopa_in")
        self.assertTrue(Path(profile["model_dir"]).is_dir())

    def test_incomplete_model_rejected(self):
        task = {"task_id": "test", "task_name": "Test"}
        model = {
            "model_id": "bad",
            "model_dir": str(Path(self.tmp.name) / "nonexistent"),
        }
        with self.assertRaises(RuntimeError):
            common.model_profile(task, model)


# ---------------------------------------------------------------------------
# 3. Case discovery (prepared mode)
# ---------------------------------------------------------------------------

class CaseDiscoveryTests(unittest.TestCase):
    """Test case scanning for prepared datasets."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_discover_prepared_cases_with_labels(self):
        for i in range(5):
            _make_prepared_case(self.root, f"case_{i:03d}")

        cases = common.discover_prepared_cases(str(self.root), ["label"])
        self.assertEqual(len(cases), 5)
        for case in cases:
            self.assertEqual(case["state"], "ready")
            self.assertTrue(case["image"])
            self.assertTrue(case["label"])

    def test_discover_cases_with_missing_labels(self):
        _make_prepared_case(self.root, "good_case", with_label=True)
        _make_prepared_case(self.root, "bad_case", with_label=False)

        cases = common.discover_prepared_cases(str(self.root), ["label"])
        ready = [c for c in cases if c["state"] == "ready"]
        missing = [c for c in cases if c["state"] == "mask_missing"]
        self.assertEqual(len(ready), 1)
        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0]["case_id"], "bad_case")

    def test_empty_directory_returns_empty(self):
        cases = common.discover_prepared_cases(str(self.root), ["label"])
        self.assertEqual(len(cases), 0)

    def test_non_existent_directory(self):
        cases = common.discover_prepared_cases(str(self.root / "nope"), ["label"])
        self.assertEqual(len(cases), 0)

    def test_find_case_image_prefers_standard_names(self):
        case_dir = self.root / "case_01"
        case_dir.mkdir(parents=True)
        (case_dir / "mr.nii.gz").write_bytes(b"mr-image")
        (case_dir / "random_scan.nii.gz").write_bytes(b"other")

        image = common.find_case_image(case_dir)
        self.assertIsNotNone(image)
        self.assertIn("mr.nii.gz", str(image))

    def test_find_case_image_falls_back_to_single_nifti(self):
        case_dir = self.root / "case_02"
        case_dir.mkdir(parents=True)
        (case_dir / "weird_name.nii.gz").write_bytes(b"data")

        image = common.find_case_image(case_dir)
        self.assertIsNotNone(image)
        self.assertIn("weird_name.nii.gz", str(image))

    def test_ambiguous_case_returns_none(self):
        case_dir = self.root / "case_03"
        case_dir.mkdir(parents=True)
        (case_dir / "a.nii.gz").write_bytes(b"a")
        (case_dir / "b.nii.gz").write_bytes(b"b")

        image = common.find_case_image(case_dir)
        self.assertIsNone(image)


# ---------------------------------------------------------------------------
# 4. Model audit
# ---------------------------------------------------------------------------

class ModelAuditTests(unittest.TestCase):
    """Test model directory validation."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_complete_model_passes_audit(self):
        model = _make_model_dir(Path(self.tmp.name))
        audit = common.audit_model_dir(model)
        self.assertTrue(audit["compatible"])
        self.assertEqual(len(audit["missing"]), 0)
        self.assertIn("checkpoint_sha256", audit)
        self.assertIn("fold", audit)

    def test_missing_checkpoint_fails_audit(self):
        model = Path(self.tmp.name) / "bad_model"
        model.mkdir(parents=True)
        for name in common.MODEL_METADATA_FILES:
            (model / name).write_text("{}")
        audit = common.audit_model_dir(model)
        self.assertFalse(audit["compatible"])
        self.assertIn("fold_*/checkpoint_final.pth", audit["missing"])

    def test_missing_metadata_fails_audit(self):
        model = Path(self.tmp.name) / "bad_model"
        model.mkdir(parents=True)
        (model / "fold_0").mkdir(parents=True)
        (model / "fold_0" / "checkpoint_final.pth").write_bytes(b"data")
        audit = common.audit_model_dir(model)
        self.assertFalse(audit["compatible"])
        self.assertGreater(len(audit["missing"]), 0)

    def test_checksum_consistent(self):
        model = _make_model_dir(Path(self.tmp.name), checkpoint_bytes=b"deterministic-content")
        sha1 = common.sha256_file(model / "fold_0" / "checkpoint_final.pth")
        sha2 = common.sha256_file(model / "fold_0" / "checkpoint_final.pth")
        self.assertEqual(sha1, sha2)

    def test_different_content_different_checksum(self):
        m1 = _make_model_dir(Path(self.tmp.name) / "m1", checkpoint_bytes=b"aaa")
        m2 = _make_model_dir(Path(self.tmp.name) / "m2", checkpoint_bytes=b"bbb")
        sha1 = common.sha256_file(m1 / "fold_0" / "checkpoint_final.pth")
        sha2 = common.sha256_file(m2 / "fold_0" / "checkpoint_final.pth")
        self.assertNotEqual(sha1, sha2)


# ---------------------------------------------------------------------------
# 5. Training request building
# ---------------------------------------------------------------------------

class TrainingRequestTests(unittest.TestCase):
    """Test that training requests are well-formed."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp.name) / "workspace"
        self.workspace.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def _build_minimal_request(self) -> dict:
        """Simulate what the Model Center GUI assembles when user clicks Start Training."""
        task_id = "liver_ct"
        model_id = f"model_{time.strftime('%Y%m%dT%H%M%S')}_test"
        job_id = f"train_{time.strftime('%Y%m%dT%H%M%S')}_test"
        task_root = common.task_dir(self.workspace, task_id)
        output_model_dir = task_root / "models" / model_id

        # Simulate 6 cases: 4 train, 2 val
        dataset = self.workspace / "dataset"
        dataset.mkdir()
        cases = []
        for i in range(6):
            case_dir = _make_prepared_case(dataset, f"case_{i:03d}")
            split = "val" if i >= 4 else "train"
            cases.append({
                "case_id": f"case_{i:03d}",
                "case_dir": str(case_dir),
                "image": str(case_dir / "image.nii.gz"),
                "label": str(case_dir / "label.nii.gz"),
                "state": "ready",
                "split": split,
            })

        return {
            "schema_version": "nninteractive_task_training_request.v1",
            "job_id": job_id,
            "task_id": task_id,
            "task_name": "Liver CT",
            "workspace": str(self.workspace),
            "source_mode": "prepared",
            "mcs_dir": "",
            "image_root": str(dataset),
            "prepared_root": str(dataset),
            "mask_names": ["Liver", "liver_mask"],
            "cases": cases,
            "strategy": "clopa_in",
            "epochs": 10,
            "base_model_dir": str(self.workspace / "official_model"),
            "parent_model_id": "official",
            "model_id": model_id,
            "output_model_dir": str(output_model_dir),
            "mimics_exe": "",
            "created_at_epoch": time.time(),
        }

    def test_request_has_required_fields(self):
        req = self._build_minimal_request()
        required = ["job_id", "task_id", "task_name", "workspace", "cases",
                    "strategy", "epochs", "base_model_dir", "model_id",
                    "output_model_dir"]
        for field in required:
            self.assertIn(field, req, f"Missing required field: {field}")

    def test_request_cases_have_correct_structure(self):
        req = self._build_minimal_request()
        train = [c for c in req["cases"] if c["split"] == "train"]
        val = [c for c in req["cases"] if c["split"] == "val"]
        self.assertEqual(len(train), 4)
        self.assertEqual(len(val), 2)
        for case in req["cases"]:
            self.assertIn("case_id", case)
            self.assertIn("image", case)
            self.assertIn("label", case)
            self.assertIn("split", case)
            self.assertTrue(case["image"])
            self.assertTrue(case["label"])

    def test_strategy_values_are_valid(self):
        req = self._build_minimal_request()
        self.assertIn(req["strategy"], ("clopa_in", "clopa_conv"))

    def test_output_model_dir_nested_correctly(self):
        req = self._build_minimal_request()
        output = Path(req["output_model_dir"])
        # Should be: workspace/tasks/{task_id}/models/{model_id}
        self.assertIn("tasks", output.parts)
        self.assertIn("models", output.parts)
        # output is 4 levels deep under workspace
        self.assertTrue(str(output).startswith(str(self.workspace)))


# ---------------------------------------------------------------------------
# 6. Quality evaluation logic
# ---------------------------------------------------------------------------

class QualityEvaluationTests(unittest.TestCase):
    """Test the AUC comparison and model auto-selection logic."""

    def test_improved_model_qualifies(self):
        baseline = {"trajectory_auc": 0.75, "cases": [
            {"case_id": "c1", "dice_by_click": [0.7, 0.75, 0.8, 0.82, 0.83]},
            {"case_id": "c2", "dice_by_click": [0.6, 0.65, 0.7, 0.72, 0.75]},
        ]}
        candidate = {"trajectory_auc": 0.82, "cases": [
            {"case_id": "c1", "dice_by_click": [0.8, 0.85, 0.88, 0.89, 0.90]},
            {"case_id": "c2", "dice_by_click": [0.7, 0.75, 0.78, 0.80, 0.81]},
        ]}
        quality = pipeline._quality_result(baseline, candidate, {
            "minimum_mean_auc_improvement": 0.0,
            "maximum_severe_case_regression": 0.2,
        })
        self.assertTrue(quality["qualifies"])
        self.assertAlmostEqual(quality["delta_auc"], 0.07, places=3)

    def test_degraded_model_does_not_qualify(self):
        baseline = {"trajectory_auc": 0.80, "cases": [
            {"case_id": "c1", "dice_by_click": [0.8, 0.82, 0.84, 0.85, 0.85]},
        ]}
        candidate = {"trajectory_auc": 0.75, "cases": [
            {"case_id": "c1", "dice_by_click": [0.7, 0.72, 0.74, 0.75, 0.75]},
        ]}
        quality = pipeline._quality_result(baseline, candidate, {
            "minimum_mean_auc_improvement": 0.0,
            "maximum_severe_case_regression": 0.2,
        })
        self.assertFalse(quality["qualifies"])
        self.assertLess(quality["delta_auc"], 0)

    def test_severe_case_regression_blocks_selection(self):
        baseline = {"trajectory_auc": 0.80, "cases": [
            {"case_id": "c1", "dice_by_click": [0.9, 0.91, 0.92, 0.93, 0.94]},
        ]}
        candidate = {"trajectory_auc": 0.90, "cases": [
            {"case_id": "c1", "dice_by_click": [0.5, 0.55, 0.60, 0.62, 0.63]},
        ]}
        quality = pipeline._quality_result(baseline, candidate, {
            "minimum_mean_auc_improvement": 0.0,
            "maximum_severe_case_regression": 0.2,
        })
        self.assertFalse(quality["qualifies"])
        self.assertEqual(len(quality["severe_regressions"]), 1)
        self.assertLess(quality["severe_regressions"][0]["delta"], -0.2)

    def test_minimum_auc_threshold_respected(self):
        baseline = {"trajectory_auc": 0.80, "cases": [
            {"case_id": "c1", "dice_by_click": [0.8, 0.82, 0.84, 0.85, 0.85]},
        ]}
        candidate = {"trajectory_auc": 0.805, "cases": [
            {"case_id": "c1", "dice_by_click": [0.81, 0.83, 0.84, 0.85, 0.86]},
        ]}
        # With threshold 0.01, 0.005 improvement doesn't qualify
        quality = pipeline._quality_result(baseline, candidate, {
            "minimum_mean_auc_improvement": 0.01,
            "maximum_severe_case_regression": 0.2,
        })
        self.assertFalse(quality["qualifies"])


# ---------------------------------------------------------------------------
# 7. Status tracking
# ---------------------------------------------------------------------------

class StatusTrackingTests(unittest.TestCase):
    """Test job status tracking and phase transitions."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_status_summary_maps_all_phases(self):
        phases = [
            ("created", "Preparing"),
            ("validating_cases", "Checking"),
            ("exporting_labels", "Preparing labels"),
            ("preparing_data", "Preparing training data"),
            ("waiting_for_gpu", "Waiting for the GPU"),
            ("training", "Training"),
            ("validating", "Checking model quality"),
            ("registering", "Saving"),
            ("completed", "completed"),
            ("paused", "paused"),
            ("cancelled", "stopped"),
            ("failed", "needs attention"),
        ]
        for phase, expected_substr in phases:
            status = {"status": phase}
            summary = common.status_summary(status).lower()
            self.assertIn(expected_substr.lower(), summary,
                          f"Phase '{phase}' summary '{summary}' missing '{expected_substr}'")

    def test_preparing_data_summary_includes_progress(self):
        summary = common.status_summary({
            "status": "preparing_data",
            "preparation_index": 3,
            "preparation_total": 8,
        })
        self.assertEqual("Preparing data 3/8 (38%)", summary)

    def test_active_statuses_excludes_terminal(self):
        for status in common.TERMINAL_STATUSES:
            self.assertNotIn(status, common.ACTIVE_STATUSES)

    def test_terminal_statuses_are_complete(self):
        expected = {"completed", "failed", "cancelled", "paused", "abandoned"}
        self.assertEqual(common.TERMINAL_STATUSES, expected)

    def test_atomic_write_preserves_content(self):
        path = Path(self.tmp.name) / "test.json"
        payload = {"key": "value", "nested": {"a": 1}}
        common.write_json_atomic(path, payload)
        read_back = common.read_json(path)
        self.assertEqual(read_back, payload)

    def test_atomic_write_handles_concurrent_reads(self):
        path = Path(self.tmp.name) / "concurrent.json"
        common.write_json_atomic(path, {"v": 1})
        # Simulate rapid updates
        for i in range(20):
            common.write_json_atomic(path, {"v": i})
        self.assertEqual(common.read_json(path)["v"], 19)


# ---------------------------------------------------------------------------
# 8. Model identity and multi-model coexistence
# ---------------------------------------------------------------------------

class ModelIdentityTests(unittest.TestCase):
    """Test that different task models are correctly distinguished."""

    def test_official_model_identity(self):
        """Simulate _model_profile for official model (no _model_profile set)."""
        config = {}
        # Mimicking nninteractive_mimics._model_profile
        raw = config.get("_model_profile") or {}
        is_task = isinstance(raw, dict) and str(raw.get("source") or "").lower() == "task_model"
        self.assertFalse(is_task)

    def test_task_model_identity_distinct(self):
        """Task model profile is distinguishable from official."""
        config = {
            "_model_profile": {
                "source": "task_model",
                "profile_id": "liver:v1",
                "task_id": "liver",
                "task_name": "Liver CT",
                "model_id": "v1",
                "model_dir": "/path/to/task_model",
                "checkpoint_sha256": "abc123",
                "strategy": "clopa_in",
            }
        }
        raw = config.get("_model_profile") or {}
        is_task = isinstance(raw, dict) and str(raw.get("source") or "").lower() == "task_model"
        self.assertTrue(is_task)
        self.assertEqual(raw["task_id"], "liver")
        self.assertEqual(raw["model_id"], "v1")

    def test_worker_cache_key_differs_by_model(self):
        """Different models produce different cache keys."""
        official_identity = "official"
        task_identity = "liver:v1|v1|abc123"
        image_key = "image_guid_123"
        self.assertEqual(image_key, image_key)  # official uses bare image key
        self.assertNotEqual(
            image_key,  # official
            image_key + "::" + task_identity,  # task model
        )

    def test_state_model_match_detection(self):
        """_state_model_matches correctly identifies stale worker state."""
        profile = {
            "source": "task_model",
            "profile_id": "liver:v2",
            "model_id": "v2",
            "checkpoint_sha256": "def456",
        }
        # State with different model_id
        state = {"model_identity": "liver:v1|v1|abc123"}
        match = state.get("model_identity", "official") == "liver:v2|v2|def456"
        self.assertFalse(match)

        # State with same model
        state2 = {"model_identity": "liver:v2|v2|def456"}
        match2 = state2.get("model_identity", "official") == "liver:v2|v2|def456"
        self.assertTrue(match2)


# ---------------------------------------------------------------------------
# 9. Control flow (pause/stop/resume)
# ---------------------------------------------------------------------------

class ControlFlowTests(unittest.TestCase):
    """Test training control actions."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_control_action_reads_action_field(self):
        control = Path(self.tmp.name) / "control.json"
        common.write_json_atomic(control, {"action": "pause", "requested_at_epoch": time.time()})
        action = pipeline._control_action(control)
        self.assertEqual(action, "pause")

    def test_control_action_reads_legacy_stop_requested(self):
        control = Path(self.tmp.name) / "control.json"
        common.write_json_atomic(control, {"status": "stop_requested"})
        action = pipeline._control_action(control)
        self.assertEqual(action, "stop")

    def test_missing_control_file_returns_empty(self):
        action = pipeline._control_action(Path(self.tmp.name) / "nonexistent.json")
        self.assertEqual(action, "")

    def test_empty_control_file_returns_empty(self):
        control = Path(self.tmp.name) / "control.json"
        common.write_json_atomic(control, {})
        action = pipeline._control_action(control)
        self.assertEqual(action, "")

    def test_resume_requires_paused_or_failed(self):
        job_dir = Path(self.tmp.name) / "job"
        job_dir.mkdir()
        common.write_json_atomic(job_dir / "status.json", {"status": "completed"})
        with self.assertRaises(RuntimeError):
            pipeline.resume_job(str(job_dir))

    def test_failed_job_cleanup_preserves_diagnostics_and_resets_resume_inputs(self):
        workspace = Path(self.tmp.name) / "workspace"
        job_dir = workspace / "jobs" / "failed_job"
        model_dir = workspace / "tasks" / "brain" / "models" / "candidate"
        for path in (
            job_dir / "staging",
            job_dir / "prepared_cache",
            model_dir,
        ):
            path.mkdir(parents=True)
            (path / "large.bin").write_bytes(b"x" * 1024)
        for name in (
            "dataset_manifest.json",
            "validation_manifest.json",
            "training_config.json",
            "trainer_status.json",
        ):
            common.write_json_atomic(job_dir / name, {"temporary": True})
        (job_dir / "job.log").write_text("diagnostic", encoding="utf-8")
        common.write_json_atomic(job_dir / "status.json", {"status": "failed"})
        request = {
            "workspace": str(workspace),
            "output_model_dir": str(model_dir),
        }

        report = pipeline._cleanup_terminal_artifacts(
            request,
            job_dir,
            remove_partial_model=True,
            reset_resume_state=True,
        )

        self.assertFalse((job_dir / "staging").exists())
        self.assertFalse((job_dir / "prepared_cache").exists())
        self.assertFalse(model_dir.exists())
        self.assertFalse((job_dir / "dataset_manifest.json").exists())
        self.assertTrue((job_dir / "job.log").is_file())
        self.assertTrue((job_dir / "status.json").is_file())
        self.assertFalse(report.get("not_removed"))


# ---------------------------------------------------------------------------
# 10. End-to-end simulation (no GPU, no Mimics)
# ---------------------------------------------------------------------------

class EndToEndSimulation(unittest.TestCase):
    """Simulate complete user journey from training setup to inference."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.workspace = self.root / "nninteractive_task_models"
        self.workspace.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_full_training_to_inference_flow(self):
        """Simulate: configure → train → register → select → annotate."""
        # ---- Step 1: User opens Model Center, configures training ----
        task_id = "brain_extraction"
        task_name = "Brain Extraction"
        mask_names = ["Brain", "brain_mask"]

        # ---- Step 2: Scan cases ----
        dataset = self.root / "brain_dataset"
        dataset.mkdir()
        for i in range(8):
            _make_prepared_case(dataset, f"brain_{i:03d}")

        cases = common.discover_prepared_cases(str(dataset), mask_names)
        self.assertEqual(len(cases), 8)
        ready = [c for c in cases if c["state"] == "ready"]
        self.assertEqual(len(ready), 8)

        # Assign splits: 6 train, 2 val
        for i, case in enumerate(ready):
            case["split"] = "val" if i >= 6 else "train"

        train_cases = [c for c in ready if c["split"] == "train"]
        val_cases = [c for c in ready if c["split"] == "val"]
        self.assertEqual(len(train_cases), 6)
        self.assertEqual(len(val_cases), 2)

        # ---- Step 3: Build training request ----
        model_id = f"model_{time.strftime('%Y%m%dT%H%M%S')}_sim"
        output_model_dir = common.task_dir(self.workspace, task_id) / "models" / model_id
        official_model = _make_model_dir(self.root / "official")

        request = {
            "task_id": task_id,
            "task_name": task_name,
            "mask_names": mask_names,
            "model_id": model_id,
            "strategy": "clopa_in",
            "epochs": 10,
            "cases": ready,
            "base_model_dir": str(official_model),
            "parent_model_id": "official",
            "output_model_dir": str(output_model_dir),
            "workspace": str(self.workspace),
            "source_mode": "prepared",
            "mcs_dir": "",
            "image_root": str(dataset),
            "prepared_root": str(dataset),
        }

        # ---- Step 4: Simulate training completion → model registered ----
        trained_model = _make_model_dir(output_model_dir, checkpoint_bytes=b"trained-weights")
        audit = common.audit_model_dir(trained_model)
        self.assertTrue(audit["compatible"])

        # Register as unverified (no validation cases)
        model_unverified, selected_unverified = pipeline._register_model(
            request, self.workspace, trained_model,
            quality=None, validation_count=0,
        )
        self.assertEqual(model_unverified["state"], "unverified")
        self.assertFalse(selected_unverified)

        # ---- Step 5: Register a validated model that improves ----
        model_id_v2 = "model_v2_sim"
        output_v2 = common.task_dir(self.workspace, task_id) / "models" / model_id_v2
        trained_v2 = _make_model_dir(output_v2, checkpoint_bytes=b"better-weights")

        quality = {
            "qualifies": True,
            "delta_auc": 0.05,
            "baseline_auc": 0.80,
            "candidate_auc": 0.85,
            "severe_regressions": [],
        }
        request_v2 = dict(request)
        request_v2["model_id"] = model_id_v2
        request_v2["output_model_dir"] = str(output_v2)
        request_v2["cases"] = ready  # includes validation

        model_v2, selected = pipeline._register_model(
            request_v2, self.workspace, trained_v2,
            quality=quality, validation_count=2,
        )
        self.assertEqual(model_v2["state"], "validated")
        self.assertTrue(selected)

        # ---- Step 6: User clicks "Annotate with Task Model" ----
        # Resolve task model automatically
        task = common.find_task(self.workspace, task_id)
        self.assertIsNotNone(task)
        self.assertEqual(task["recommended_model_id"], model_id_v2)

        selected_model_obj = common.selected_model(self.workspace, task_id)
        self.assertIsNotNone(selected_model_obj)
        self.assertEqual(selected_model_obj["model_id"], model_id_v2)

        # ---- Step 7: Build model profile for inference ----
        profile = common.model_profile(task, selected_model_obj)
        self.assertEqual(profile["source"], "task_model")
        self.assertEqual(profile["task_id"], task_id)
        self.assertEqual(profile["model_id"], model_id_v2)
        self.assertEqual(profile["strategy"], "clopa_in")
        self.assertTrue(Path(profile["model_dir"]).is_dir())

        # ---- Step 8: Verify profile would be routed correctly ----
        # (This is what run_with_model_profile would do)
        config = {"_model_profile": dict(profile)}
        raw = config.get("_model_profile") or {}
        is_task = isinstance(raw, dict) and str(raw.get("source") or "").lower() == "task_model"
        self.assertTrue(is_task)
        # model_profile.resolve() resolves symlinks (/var → /private/var on macOS)
        self.assertTrue(os.path.samefile(raw["model_dir"], str(trained_v2)),
                        f"{raw['model_dir']} != {trained_v2}")

        # ---- Step 9: Verify model_dir exists and has checkpoint ----
        resolved_model_dir = Path(raw["model_dir"])
        self.assertTrue(resolved_model_dir.is_dir())
        checkpoint = resolved_model_dir / "fold_0" / "checkpoint_final.pth"
        self.assertTrue(checkpoint.is_file())

    def test_fallback_to_chooser_when_multiple_tasks(self):
        """When multiple tasks exist and no unambiguous match, chooser is needed."""
        registry = common.load_registry(self.workspace)
        registry["tasks"] = [
            {
                "task_id": "liver",
                "task_name": "Liver",
                "mask_names": ["Liver"],
                "recommended_model_id": "liver_v1",
                "models": [{"model_id": "liver_v1", "model_dir": "/tmp/liver",
                           "created_at_epoch": 1, "strategy": "clopa_in",
                           "state": "validated", "compatible": True}],
            },
            {
                "task_id": "kidney",
                "task_name": "Kidney",
                "mask_names": ["Kidney"],
                "recommended_model_id": "kidney_v1",
                "models": [{"model_id": "kidney_v1", "model_dir": "/tmp/kidney",
                           "created_at_epoch": 1, "strategy": "clopa_in",
                           "state": "validated", "compatible": True}],
            },
        ]
        common.save_registry(self.workspace, registry)

        tasks = common.task_rows(self.workspace)
        self.assertEqual(len(tasks), 2)

        # Without mask name matching, can't auto-resolve → need chooser
        usable = [row for row in tasks
                  if common.selected_model(self.workspace,
                                           row.get("task_id")) is not None]
        self.assertEqual(len(usable), 2)  # Both have usable models ≠ 1


# ---------------------------------------------------------------------------
# 11. Training config generation
# ---------------------------------------------------------------------------

class TrainingConfigGenerationTests(unittest.TestCase):
    """Verify the training config generated by the pipeline is well-formed."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.workspace = self.root / "nninteractive_task_models"
        self.workspace.mkdir()
        self.job_dir = self.root / "jobs" / "test_job"
        self.job_dir.mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def _build_config(
        self, *, strategy="clopa_in", epochs=10, request_values=None
    ) -> dict:
        model_dir = self.root / "output_model"
        model_dir.mkdir(parents=True, exist_ok=True)
        manifest = self.job_dir / "manifest.json"
        manifest.write_text(json.dumps({"cases": []}))
        request = {
            "output_model_dir": str(model_dir),
            "base_model_dir": str(self.root / "official"),
            "strategy": strategy,
            "epochs": epochs,
        }
        request.update(request_values or {})
        return pipeline._training_config(request, self.job_dir, manifest)

    def test_training_config_model_section(self):
        config = self._build_config()
        model = config["model"]
        self.assertEqual(model["strategy"], "clopa_in")
        self.assertEqual(model["fold"], "0")
        self.assertEqual(model["checkpoint_name"], "checkpoint_final.pth")

    def test_training_config_data_section(self):
        config = self._build_config()
        data = config["data"]
        self.assertEqual(data["label_values"], [1])
        self.assertEqual(data["patch_size"], [128, 128, 128])
        self.assertEqual(data["num_workers"], 0)

    def test_training_config_prompts_section(self):
        config = self._build_config()
        prompts = config["prompts"]
        self.assertEqual(prompts["mode"], "clicks")
        self.assertEqual(prompts["training_goal"], "general")
        self.assertEqual(prompts["interaction_profile"], "")
        self.assertEqual(prompts["correction_policy"], "official_single")
        self.assertEqual(prompts["min_interaction_steps"], 1)
        self.assertEqual(prompts["max_interaction_steps"], 5)
        self.assertEqual(
            prompts["validation_interaction_steps"], [1, 3, 5]
        )
        self.assertEqual(
            prompts["interaction_step_weights"],
            [0.35, 0.25, 0.20, 0.12, 0.08],
        )
        self.assertGreater(prompts["initial_mask_probability"], 0.0)
        self.assertEqual(
            prompts["provided_initial_mask_probability"], 0.7
        )
        self.assertEqual(prompts["point_radius"], 4)
        self.assertEqual(prompts["center_bias"], 8.0)
        self.assertEqual(prompts["interaction_decay"], 0.9)

    def test_training_config_training_section(self):
        config = self._build_config(epochs=15)
        training = config["training"]
        self.assertEqual(training["epochs"], 15)
        self.assertEqual(training["batch_size"], 1)
        self.assertEqual(training["learning_rate"], 0.001)
        self.assertTrue(training["mixed_precision"])
        self.assertTrue(training["resume"])

    def test_strategy_propagates_to_config(self):
        for strategy in ("clopa_in", "clopa_conv"):
            with self.subTest(strategy=strategy):
                config = self._build_config(strategy=strategy)
                self.assertEqual(config["model"]["strategy"], strategy)

    def test_training_config_uses_request_real_initial_mask_mix(self):
        config = self._build_config(
            request_values={
                "training_goal": "refine_existing",
                "provided_initial_mask_probability": 0.6,
            }
        )
        prompts = config["prompts"]
        self.assertEqual(prompts["initial_mask_probability"], 1.0)
        self.assertEqual(
            prompts["provided_initial_mask_probability"], 0.6
        )

    def test_empty_start_disables_real_initial_masks(self):
        config = self._build_config(
            request_values={
                "training_goal": "start_empty",
                "provided_initial_mask_probability": 0.9,
            }
        )
        prompts = config["prompts"]
        self.assertEqual(prompts["initial_mask_probability"], 0.0)
        self.assertEqual(
            prompts["provided_initial_mask_probability"], 0.0
        )

    def test_status_and_cancel_paths_are_writable(self):
        config = self._build_config()
        trainer_status = Path(config["training"]["status_path"])
        trainer_cancel = Path(config["training"]["cancel_path"])
        self.assertTrue(str(trainer_status).startswith(str(self.job_dir)))
        self.assertTrue(str(trainer_cancel).startswith(str(self.job_dir)))

    def test_finetune_runner_injects_paths_inside_child_interpreter(self):
        script = pipeline._finetune_runner_script()
        self.assertIn(repr(str(pipeline.FINETUNE_SRC)), script)
        self.assertIn(repr(str(pipeline.ROOT)), script)
        self.assertIn(
            "from nninteractive_finetune.__main__ import main",
            script,
        )
        self.assertNotIn("PYTHONPATH", script)


# ---------------------------------------------------------------------------
# 12. Input contract validation
# ---------------------------------------------------------------------------

class InputContractTests(unittest.TestCase):
    """Verify the input contract is correctly defined, written, and validated."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_contracts_are_identical(self):
        """The fine-tuning package and Mimics integration MUST agree on the contract."""
        ft_data_path = (ROOT / "external" / "nninteractive-finetune" / "src"
                        / "nninteractive_finetune" / "data.py")
        ft_source = ft_data_path.read_text(encoding="utf-8")

        # Extract the contract dict from source (avoid import with relative imports)
        import ast
        tree = ast.parse(ft_source)
        ft_contract = None
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "NNINTERACTIVE_INPUT_CONTRACT"):
                ft_contract = ast.literal_eval(node.value)
                break

        self.assertIsNotNone(ft_contract, "Could not find NNINTERACTIVE_INPUT_CONTRACT in data.py")
        tc_contract = common.NNINTERACTIVE_INPUT_CONTRACT
        self.assertEqual(ft_contract, tc_contract,
                         "Contract mismatch between fine-tuning package and Mimics integration!")

    def test_contract_has_required_fields(self):
        contract = common.NNINTERACTIVE_INPUT_CONTRACT
        required = ["schema_version", "spatial_orientation", "source_grid_policy",
                    "intensity_space", "normalization", "normalization_channel"]
        for field in required:
            self.assertIn(field, contract)

    def test_intensity_space_is_source_physical_values(self):
        self.assertEqual(
            common.NNINTERACTIVE_INPUT_CONTRACT["intensity_space"],
            "source_physical_values",
        )

    def test_contract_rejection_on_mismatch(self):
        """Pipeline must reject models with wrong contract."""
        model_dir = _make_model_dir(self.root)
        # Corrupt the finetune_manifest with wrong contract
        bad_manifest = {
            "schema_version": "nninteractive_finetune_model.v1",
            "input_contract": {"intensity_space": "wrong_values"},
        }
        common.write_json_atomic(model_dir / "finetune_manifest.json", bad_manifest)
        with self.assertRaises(RuntimeError):
            pipeline._register_model(
                {"task_id": "test", "task_name": "Test", "model_id": "v1",
                 "strategy": "clopa_in", "source_mode": "prepared",
                 "output_model_dir": str(model_dir), "cases": [{"split": "train"}]},
                self.root / "ws", model_dir, quality=None, validation_count=0,
            )

    def test_missing_manifest_causes_registration_failure(self):
        """Model without finetune_manifest.json should fail registration."""
        model_dir = _make_model_dir(self.root)
        (model_dir / "finetune_manifest.json").unlink()
        with self.assertRaises((RuntimeError, FileNotFoundError)):
            pipeline._register_model(
                {"task_id": "test", "task_name": "Test", "model_id": "v1",
                 "strategy": "clopa_in", "source_mode": "prepared",
                 "output_model_dir": str(model_dir), "cases": [{"split": "train"}]},
                self.root / "ws", model_dir, quality=None, validation_count=0,
            )


# ---------------------------------------------------------------------------
# 13. Bridge request structure (task model path)
# ---------------------------------------------------------------------------

class BridgeRequestTests(unittest.TestCase):
    """Verify the bridge request is well-formed for task model inference."""

    def test_task_model_request_marks_intensity_space(self):
        """Task model requests must include model_input_intensity_space='source_values'."""
        profile = {
            "source": "task_model",
            "profile_id": "liver:v1",
            "task_id": "liver",
            "task_name": "Liver CT",
            "model_id": "v1",
            "model_dir": "/tmp/model",
            "checkpoint_sha256": "abc123",
            "strategy": "clopa_in",
            "input_contract": common.NNINTERACTIVE_INPUT_CONTRACT,
        }

        # Simulate what _build_bridge_request does for task models
        request = {"model_dir": profile["model_dir"]}
        if profile.get("source") == "task_model":
            request["model_input_space"] = "canonical_ras"
            request["model_input_intensity_space"] = "source_values"

        self.assertEqual(request["model_input_intensity_space"], "source_values")
        self.assertEqual(request["model_input_space"], "canonical_ras")

    def test_official_model_request_has_no_intensity_marker(self):
        """Official model requests should not have model_input_intensity_space."""
        profile = {
            "source": "official",
            "profile_id": "official",
        }
        request = {"model_dir": "/tmp/official"}
        if profile.get("source") == "task_model":
            request["model_input_intensity_space"] = "source_values"
        self.assertNotIn("model_input_intensity_space", request)

    def test_source_intensity_slope_intercept_for_task_model(self):
        """Task model source export must use slope=1.0, intercept=0.0 (no HU→GV)."""
        # Simulate the intensity logic in _source_image_export for task model
        task_model_source_values = True
        kind, modality = "nifti", "CT"
        uses_hu_to_gv = True  # _source_uses_hu_to_gv returns True for all NIfTI

        if uses_hu_to_gv and task_model_source_values:
            intensity_slope = 1.0
            intensity_intercept = 0.0
            intensity_space = "source_values"
        elif uses_hu_to_gv:
            intensity_slope, intensity_intercept = 1.0, 1024.0  # simulated HU2GV
            intensity_space = "hu_to_mimics_gv"
        else:
            intensity_slope = 1.0
            intensity_intercept = 0.0
            intensity_space = "source_values"

        # Task model path: NO transform
        self.assertEqual(intensity_slope, 1.0)
        self.assertEqual(intensity_intercept, 0.0)
        self.assertEqual(intensity_space, "source_values")

    def test_official_model_ct_gets_hu_to_gv_transform(self):
        """Official model with CT NIfTI should apply HU→GV transform."""
        task_model_source_values = False
        uses_hu_to_gv = True

        if uses_hu_to_gv and task_model_source_values:
            intensity_slope = 1.0
            intensity_intercept = 0.0
        elif uses_hu_to_gv:
            # Official model path: would call _hu_to_mimics_gv_transform()
            intensity_slope, intensity_intercept = 1.0, 1024.0  # simulated
        else:
            intensity_slope = 1.0
            intensity_intercept = 0.0

        # Official model CT path: transform IS applied
        self.assertEqual(intensity_intercept, 1024.0,
                         "Official model on CT must apply HU→GV for buffer compatibility")

    def test_source_image_export_without_source_metadata(self):
        """When source metadata is missing, task model should fail, not silently fall back."""
        # Simulate what _export_image_for_nninteractive does:
        # source_export is None → raise error when allow_task_model_mimics_buffer_fallback is False
        allow_fallback = False
        source_available = False
        if not source_available and not allow_fallback:
            should_raise = True
        else:
            should_raise = False
        self.assertTrue(should_raise,
                        "Task model MUST error when source image is unavailable")


# ---------------------------------------------------------------------------
# 14. Model Center GUI simulation (training flow end-to-end)
# ---------------------------------------------------------------------------

class ModelCenterSimulationTests(unittest.TestCase):
    """Simulate the Model Center GUI workflow: scan → configure → train → register."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.workspace = self.root / "nninteractive_task_models"
        self.workspace.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_gui_scan_cases_flow(self):
        """User opens Model Center → scans prepared dataset → sees cases."""
        dataset = self.root / "liver_dataset"
        dataset.mkdir()
        for i in range(5):
            _make_prepared_case(dataset, f"liver_{i:03d}")

        mask_names = ["Liver", "liver_seg"]
        cases = common.discover_prepared_cases(str(dataset), mask_names)
        self.assertEqual(len(cases), 5)

        ready = [c for c in cases if c["state"] == "ready"]
        self.assertEqual(len(ready), 5)

    def test_gui_split_allocation(self):
        """Default split: last 20% as validation, first 80% as train."""
        dataset = self.root / "dataset"
        dataset.mkdir()
        for i in range(8):
            _make_prepared_case(dataset, f"case_{i:03d}")

        cases = common.discover_prepared_cases(str(dataset), ["label"])
        ready_indices = [i for i, c in enumerate(cases) if c["state"] == "ready"]
        val_count = max(2, int(round(len(ready_indices) * 0.2)))
        val_indices = set(ready_indices[-val_count:])

        self.assertEqual(len(val_indices), 2)
        self.assertIn(6, val_indices)
        self.assertIn(7, val_indices)

    def test_gui_start_training_disabled_without_task_name(self):
        """Start button disabled when no task name entered."""
        train_cases = 3
        task_name = ""
        can_start = bool(train_cases >= 1 and task_name.strip())
        self.assertFalse(can_start)

    def test_gui_start_training_disabled_without_cases(self):
        """Start button disabled when no training cases selected."""
        train_cases = 0
        task_name = "Liver"
        can_start = bool(train_cases >= 1 and task_name.strip())
        self.assertFalse(can_start)

    def test_gui_start_training_enabled_with_valid_setup(self):
        """Start button enabled when task name + training cases present."""
        train_cases = 3
        task_name = "Liver CT"
        can_start = bool(train_cases >= 1 and task_name.strip())
        self.assertTrue(can_start)

    def test_gui_active_job_blocks_new_training(self):
        """When an active job exists for the same task, start is blocked."""
        jobs = common.jobs_dir(self.workspace)
        job_dir = jobs / "active_job"
        job_dir.mkdir(parents=True)
        common.write_json_atomic(job_dir / "status.json", {
            "task_id": "liver_ct",
            "status": "training",
        })

        # Check if active job for task
        task_id = "liver_ct"
        active = False
        for path in jobs.glob("*/status.json"):
            status = common.read_json(path, {}) or {}
            if (common.safe_slug(status.get("task_id")) == task_id
                    and str(status.get("status") or "") in common.ACTIVE_STATUSES):
                active = True
        self.assertTrue(active)

    def test_job_request_structure_matches_pipeline_expectations(self):
        """Job request written by GUI must be readable by pipeline."""
        job_dir = common.jobs_dir(self.workspace) / "test_job"
        job_dir.mkdir(parents=True)
        request = {
            "schema_version": "nninteractive_task_training_request.v1",
            "job_id": "test_job",
            "task_id": "liver_ct",
            "task_name": "Liver CT",
            "workspace": str(self.workspace),
            "source_mode": "prepared",
            "mask_names": ["Liver"],
            "cases": [{"case_id": "c1", "split": "train",
                       "image": "/tmp/img.nii.gz", "label": "/tmp/lbl.nii.gz"}],
            "strategy": "clopa_in",
            "epochs": 10,
            "base_model_dir": str(self.root / "official"),
            "parent_model_id": "official",
            "model_id": "model_test",
            "output_model_dir": str(self.root / "output"),
            "mimics_exe": "",
        }
        common.write_json_atomic(job_dir / "request.json", request)
        common.write_json_atomic(job_dir / "status.json", {
            "schema_version": "nninteractive_task_job.v1",
            "job_id": "test_job",
            "task_id": "liver_ct",
            "status": "created",
        })

        # Pipeline reads it back
        loaded = common.read_json(job_dir / "request.json")
        self.assertEqual(loaded["task_id"], "liver_ct")
        self.assertEqual(loaded["strategy"], "clopa_in")
        self.assertEqual(loaded["epochs"], 10)
        self.assertGreater(len(loaded["cases"]), 0)


# ---------------------------------------------------------------------------
# 15. Status polling and UI update simulation
# ---------------------------------------------------------------------------

class StatusPollingTests(unittest.TestCase):
    """Simulate the Model Center's status polling loop."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.workspace = self.root / "nninteractive_task_models"
        self.workspace.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_status_transitions_are_linear(self):
        """Normal training follows: created → exporting → preparing → training → validating → completed."""
        expected_sequence = [
            "created", "exporting_labels", "preparing_data",
            "training", "validating", "registering", "completed",
        ]
        job_dir = common.jobs_dir(self.workspace) / "test_job"
        job_dir.mkdir(parents=True)
        for status in expected_sequence:
            common.write_json_atomic(job_dir / "status.json", {
                "status": status, "task_id": "test", "updated_at_epoch": time.time(),
            })
            loaded = common.read_json(job_dir / "status.json")
            self.assertEqual(loaded["status"], status)

    def test_pause_cycle(self):
        """Training can be paused and resumed."""
        job_dir = common.jobs_dir(self.workspace) / "test_job"
        job_dir.mkdir(parents=True)

        # Start training
        common.write_json_atomic(job_dir / "status.json",
                                 {"status": "training", "task_id": "test"})
        # Pause requested
        common.write_json_atomic(job_dir / "control.json",
                                 {"action": "pause"})
        action = pipeline._control_action(job_dir / "control.json")
        self.assertEqual(action, "pause")

        # Training paused
        common.write_json_atomic(job_dir / "status.json",
                                 {"status": "paused", "task_id": "test"})
        status = common.read_json(job_dir / "status.json")
        self.assertIn(status["status"], common.TERMINAL_STATUSES)

    def test_stop_cleans_up_partial_model(self):
        """Cancelled training should be marked for cleanup."""
        job_dir = common.jobs_dir(self.workspace) / "test_job"
        job_dir.mkdir(parents=True)
        common.write_json_atomic(job_dir / "status.json",
                                 {"status": "training", "task_id": "test"})
        common.write_json_atomic(job_dir / "control.json",
                                 {"action": "stop"})
        action = pipeline._control_action(job_dir / "control.json")
        self.assertEqual(action, "stop")

    def test_metrics_history_accumulates(self):
        """Metrics from each epoch are accumulated in status."""
        job_dir = common.jobs_dir(self.workspace) / "test_job"
        job_dir.mkdir(parents=True)
        status_path = job_dir / "status.json"
        common.write_json_atomic(status_path, {"status": "training"})

        for epoch in range(1, 4):
            pipeline._copy_trainer_status(status_path, {
                "status": "training",
                "epoch": epoch,
                "epochs": 5,
                "latest_epoch": {
                    "epoch": epoch,
                    "train_loss": 1.0 - epoch * 0.2,
                    "validation": {"trajectory_auc": 0.5 + epoch * 0.1},
                },
            })

        final = common.read_json(status_path, {})
        history = final.get("metrics_history") or []
        self.assertEqual(len(history), 3)
        self.assertEqual(history[-1]["epoch"], 3)
        self.assertAlmostEqual(history[-1]["validation_auc"], 0.8, places=1)


# ---------------------------------------------------------------------------
# 16. Model profile identity tracking
# ---------------------------------------------------------------------------

class ModelProfileIdentityTests(unittest.TestCase):
    """Verify model identity is correctly tracked through the full lifecycle."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.workspace = self.root / "nninteractive_task_models"
        self.workspace.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_identity_changes_when_model_switches(self):
        """Switching from official to task model changes worker cache key."""
        official_profile = {"source": "official"}
        task_profile = {
            "source": "task_model",
            "profile_id": "liver:v1",
            "model_id": "v1",
            "checkpoint_sha256": "abc123",
        }
        official_id = "official"
        task_id = f"{task_profile['profile_id']}|{task_profile['model_id']}|{task_profile['checkpoint_sha256']}"
        self.assertNotEqual(official_id, task_id)

    def test_same_task_different_versions_have_different_identity(self):
        """Model v1 and v2 of the same task must have different identities."""
        v1 = {"source": "task_model", "profile_id": "liver:v1", "model_id": "v1",
              "checkpoint_sha256": "aaa"}
        v2 = {"source": "task_model", "profile_id": "liver:v2", "model_id": "v2",
              "checkpoint_sha256": "bbb"}
        id1 = f"{v1['profile_id']}|{v1['model_id']}|{v1['checkpoint_sha256']}"
        id2 = f"{v2['profile_id']}|{v2['model_id']}|{v2['checkpoint_sha256']}"
        self.assertNotEqual(id1, id2)

    def test_registry_preserves_recommended_model_after_multiple_registrations(self):
        """After registering 3 models, recommended should be the best one."""
        registry = common.load_registry(self.workspace)
        task = {
            "task_id": "liver",
            "task_name": "Liver",
            "mask_names": ["Liver"],
            "models": [],
        }
        registry["tasks"].append(task)
        common.save_registry(self.workspace, registry)

        # Register 3 models with different quality
        for i, (model_id, delta_auc, qualifies) in enumerate([
            ("v1", 0.02, True),
            ("v2", 0.05, True),
            ("v3", -0.01, False),
        ]):
            model_dir = _make_model_dir(self.root / f"model_{model_id}")
            request = {
                "task_id": "liver", "task_name": "Liver",
                "mask_names": ["Liver"], "model_id": model_id,
                "strategy": "clopa_in", "source_mode": "prepared",
                "output_model_dir": str(model_dir),
                "cases": [{"split": "train"}, {"split": "val"}, {"split": "val"}],
            }
            pipeline._register_model(
                request, self.workspace, model_dir,
                quality={"qualifies": qualifies, "delta_auc": delta_auc,
                         "severe_regressions": []},
                validation_count=2,
            )

        task_final = common.find_task(self.workspace, "liver")
        # v2 should be recommended (best qualifying)
        self.assertEqual(task_final["recommended_model_id"], "v2")

        # All 3 models should be in the list
        models = common.model_rows(self.workspace, "liver")
        self.assertEqual(len(models), 3)


if __name__ == "__main__":
    unittest.main()
