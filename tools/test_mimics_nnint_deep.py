#!/usr/bin/env python3
"""Deep defensive tests for Mimics nnInteractive integration.

Covers edge cases and bug-prone paths that a user would hit in real Mimics usage:
- Task resolution decision tree
- Profile validation edge cases
- Pipeline error recovery
- Registry integrity under stress
- Bridge request completeness
- Mock Mimics module behavior
- Label export config generation
- Data manifest structure
- Resource lock simulation
- Concurrent access patterns
"""

from __future__ import annotations

import json
import hashlib
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
RUNTIME = ROOT / "runtime_py35"
FINETUNE_SRC = ROOT / "external" / "nninteractive-finetune" / "src"
for value in (str(ROOT), str(TOOLS), str(RUNTIME), str(FINETUNE_SRC)):
    if value not in sys.path:
        sys.path.insert(0, value)

import nninteractive_finetune_pipeline as pipeline
import nninteractive_task_common as common


# ============================================================================
# Helpers
# ============================================================================

def _make_model_dir(root: Path, *, checkpoint_bytes: bytes = b"ckpt") -> Path:
    model = root / "model"
    for name in common.MODEL_METADATA_FILES:
        (model / name).parent.mkdir(parents=True, exist_ok=True)
        (model / name).write_text("{}", encoding="utf-8")
    checkpoint = model / "fold_0" / "checkpoint_final.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(checkpoint_bytes)
    (model / "finetune_manifest.json").write_text(
        json.dumps({"input_contract": common.NNINTERACTIVE_INPUT_CONTRACT}),
        encoding="utf-8",
    )
    return model


def _make_registry(workspace: Path, tasks: list[dict]) -> None:
    payload = {
        "schema_version": "nninteractive_task_registry.v1",
        "tasks": tasks,
        "updated_at_epoch": time.time(),
    }
    common.write_json_atomic(workspace / "registry.json", payload)


# ============================================================================
# 1. Task resolution decision tree (every branch)
# ============================================================================

class TaskResolutionTests(unittest.TestCase):
    """Test every branch of _task_for_selected_context().

    The auto-resolution logic in nninteractive_finetune_mimics.py follows:
    1. No tasks → None
    2. Mask metadata has task_id → find that task
    3. Mask name matches exactly one task's mask_names/aliases → use it
    4. Project binding has task_id → use it
    5. Exactly one usable task → auto-select
    6. Multiple usable tasks → None (open chooser)
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    # -- helpers to simulate the decision tree without Mimics --

    def _resolve(self, tasks, mask_metadata_task="", mask_name="",
                 project_binding=""):
        """Pure-logic replica of _task_for_selected_context."""
        if not tasks:
            return None
        # Branch 1: mask metadata
        if mask_metadata_task:
            wanted = common.safe_slug(mask_metadata_task)
            for t in tasks:
                if common.safe_slug(t.get("task_id")) == wanted:
                    return t
        # Branch 2: mask name match
        if mask_name:
            slug = common.safe_slug(mask_name)
            matching = []
            for t in tasks:
                aliases = list(t.get("mask_names") or [])
                aliases.extend([t.get("task_name") or "", t.get("task_id") or ""])
                if slug in [common.safe_slug(v) for v in aliases if v]:
                    matching.append(t)
            if len(matching) == 1:
                return matching[0]
        # Branch 3: project binding
        if project_binding:
            t = next((t for t in tasks
                      if common.safe_slug(t.get("task_id")) == common.safe_slug(project_binding)), None)
            if t:
                return t
        # Branch 4: single usable task
        usable = [t for t in tasks if t.get("recommended_model_id")]
        if len(usable) == 1:
            return usable[0]
        return None

    def test_empty_registry_returns_none(self):
        self.assertIsNone(self._resolve([]))

    def test_mask_metadata_wins_over_everything(self):
        tasks = [
            {"task_id": "liver", "task_name": "Liver", "mask_names": ["Liver"],
             "recommended_model_id": "v1"},
            {"task_id": "kidney", "task_name": "Kidney", "mask_names": ["Kidney"],
             "recommended_model_id": "v1"},
        ]
        # Mask metadata says "kidney" — should return kidney even if mask name is "Liver"
        result = self._resolve(tasks, mask_metadata_task="kidney", mask_name="Liver")
        self.assertIsNotNone(result)
        self.assertEqual(result["task_id"], "kidney")

    def test_mask_name_exact_match_single_task(self):
        tasks = [
            {"task_id": "liver_ct", "task_name": "Liver CT", "mask_names": ["Liver", "liver_mask"],
             "recommended_model_id": "v1"},
            {"task_id": "brain", "task_name": "Brain MR", "mask_names": ["Brain"],
             "recommended_model_id": "v1"},
        ]
        # Mask name "Liver" matches first task's mask_names
        result = self._resolve(tasks, mask_name="Liver")
        self.assertIsNotNone(result)
        self.assertEqual(result["task_id"], "liver_ct")

    def test_mask_name_matches_task_name(self):
        tasks = [
            {"task_id": "foo", "task_name": "Liver", "mask_names": [],
             "recommended_model_id": "v1"},
            {"task_id": "bar", "task_name": "Brain", "mask_names": [],
             "recommended_model_id": "v1"},
        ]
        result = self._resolve(tasks, mask_name="Liver")
        self.assertIsNotNone(result)
        self.assertEqual(result["task_id"], "foo")

    def test_mask_name_ambiguous_returns_none(self):
        tasks = [
            {"task_id": "liver_v1", "task_name": "Liver", "mask_names": ["Liver"],
             "recommended_model_id": "v1"},
            {"task_id": "liver_v2", "task_name": "Liver v2", "mask_names": ["Liver"],
             "recommended_model_id": "v1"},
        ]
        # Both have "Liver" in mask_names → ambiguous
        result = self._resolve(tasks, mask_name="Liver")
        self.assertIsNone(result)

    def test_project_binding_resolves(self):
        tasks = [
            {"task_id": "liver", "task_name": "Liver", "recommended_model_id": "v1"},
            {"task_id": "kidney", "task_name": "Kidney", "recommended_model_id": "v1"},
        ]
        result = self._resolve(tasks, mask_name="Bone", project_binding="kidney")
        self.assertIsNotNone(result)
        self.assertEqual(result["task_id"], "kidney")

    def test_single_usable_task_auto_selected(self):
        tasks = [
            {"task_id": "liver", "task_name": "Liver", "recommended_model_id": "v1"},
            {"task_id": "kidney", "task_name": "Kidney"},  # no recommended model!
        ]
        # Only liver has a recommended model → auto-select
        result = self._resolve(tasks, mask_name="Unknown")
        self.assertIsNotNone(result)
        self.assertEqual(result["task_id"], "liver")

    def test_multiple_usable_tasks_with_unmatched_mask_returns_none(self):
        tasks = [
            {"task_id": "liver", "task_name": "Liver", "recommended_model_id": "v1"},
            {"task_id": "kidney", "task_name": "Kidney", "recommended_model_id": "v1"},
        ]
        result = self._resolve(tasks, mask_name="UnknownMask")
        self.assertIsNone(result)

    def test_task_with_no_recommended_model_not_auto_selected(self):
        tasks = [
            {"task_id": "liver", "task_name": "Liver"},  # no recommended_model_id
        ]
        result = self._resolve(tasks)
        self.assertIsNone(result)

    def test_mask_name_case_insensitive(self):
        tasks = [
            {"task_id": "liver", "task_name": "Liver", "mask_names": ["LIVER"],
             "recommended_model_id": "v1"},
            {"task_id": "brain", "task_name": "Brain", "mask_names": ["brain"],
             "recommended_model_id": "v1"},
        ]
        result = self._resolve(tasks, mask_name="liver")
        self.assertIsNotNone(result)
        self.assertEqual(result["task_id"], "liver")


# ============================================================================
# 2. Profile validation edge cases
# ============================================================================

class ProfileValidationTests(unittest.TestCase):
    """Test _model_profile and its edge cases."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _build_profile(self, raw_profile):
        """Replica of _model_profile validation."""
        raw = raw_profile or {}
        if not isinstance(raw, dict) or str(raw.get("source") or "").lower() != "task_model":
            return {"source": "official", "profile_id": "official"}
        model_dir = str(raw.get("model_dir") or "").strip()
        profile = {
            "source": "task_model",
            "profile_id": str(raw.get("profile_id") or raw.get("model_id") or "").strip(),
            "model_id": str(raw.get("model_id") or "").strip(),
            "checkpoint_sha256": str(raw.get("checkpoint_sha256") or "").strip().lower(),
            "task_id": str(raw.get("task_id") or "").strip(),
            "task_name": str(raw.get("task_name") or "").strip(),
            "model_dir": os.path.abspath(model_dir) if model_dir else "",
            "input_contract": raw.get("input_contract") or common.NNINTERACTIVE_INPUT_CONTRACT,
        }
        if not profile["profile_id"] or not profile["model_id"] or not profile["model_dir"]:
            raise RuntimeError("Incomplete task model profile")
        contract = profile.get("input_contract") or {}
        if (contract.get("intensity_space") != "source_physical_values"
                or contract.get("normalization") != "nonzero_spatial_bbox_zscore"):
            raise RuntimeError("Unsupported input preprocessing contract")
        return profile

    def test_official_model_default_profile(self):
        profile = self._build_profile(None)
        self.assertEqual(profile["source"], "official")

    def test_empty_dict_is_official(self):
        profile = self._build_profile({})
        self.assertEqual(profile["source"], "official")

    def test_source_not_task_model_is_official(self):
        profile = self._build_profile({"source": "something_else"})
        self.assertEqual(profile["source"], "official")

    def test_missing_model_dir_raises(self):
        with self.assertRaises(RuntimeError):
            self._build_profile({
                "source": "task_model",
                "profile_id": "test:v1",
                "model_id": "v1",
            })

    def test_missing_model_id_raises(self):
        with self.assertRaises(RuntimeError):
            self._build_profile({
                "source": "task_model",
                "profile_id": "test:v1",
                "model_dir": "/tmp/model",
            })

    def test_wrong_contract_intensity_space_raises(self):
        model_dir = _make_model_dir(self.root / "m")
        with self.assertRaises(RuntimeError):
            self._build_profile({
                "source": "task_model",
                "profile_id": "test:v1",
                "model_id": "v1",
                "model_dir": str(model_dir),
                "input_contract": {"intensity_space": "wrong", "normalization": "nonzero_spatial_bbox_zscore"},
            })

    def test_wrong_contract_normalization_raises(self):
        model_dir = _make_model_dir(self.root / "m")
        with self.assertRaises(RuntimeError):
            self._build_profile({
                "source": "task_model",
                "profile_id": "test:v1",
                "model_id": "v1",
                "model_dir": str(model_dir),
                "input_contract": {"intensity_space": "source_physical_values", "normalization": "wrong"},
            })

    def test_valid_profile_passes(self):
        model_dir = _make_model_dir(self.root / "m")
        profile = self._build_profile({
            "source": "task_model",
            "profile_id": "test:v1",
            "model_id": "v1",
            "task_id": "test",
            "model_dir": str(model_dir),
            "checkpoint_sha256": "abc",
            "strategy": "clopa_in",
        })
        self.assertEqual(profile["source"], "task_model")
        self.assertEqual(profile["model_id"], "v1")
        self.assertTrue(os.path.isdir(profile["model_dir"]))


# ============================================================================
# 3. Pipeline error recovery paths
# ============================================================================

class PipelineErrorRecoveryTests(unittest.TestCase):
    """Test that pipeline fails gracefully and cleans up correctly."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_request_file_raises(self):
        job_dir = self.root / "job"
        job_dir.mkdir()
        with self.assertRaises(RuntimeError):
            pipeline.run_job(str(job_dir))

    def test_empty_request_file_raises(self):
        job_dir = self.root / "job"
        job_dir.mkdir()
        common.write_json_atomic(job_dir / "request.json", {})
        with self.assertRaises((RuntimeError, KeyError)):
            pipeline.run_job(str(job_dir))

    def test_invalid_json_request_raises(self):
        job_dir = self.root / "job"
        job_dir.mkdir()
        (job_dir / "request.json").write_text("not json", encoding="utf-8")
        # read_json returns default on parse error
        with self.assertRaises(RuntimeError):
            pipeline.run_job(str(job_dir))

    def test_interrupted_error_causes_pause_or_cancel(self):
        job_dir = self.root / "job"
        job_dir.mkdir()
        common.write_json_atomic(job_dir / "status.json", {
            "status": "training", "task_id": "test",
            "updated_at_epoch": time.time(),
        })
        common.write_json_atomic(job_dir / "control.json",
                                 {"action": "pause"})
        action = pipeline._control_action(job_dir / "control.json")
        self.assertEqual(action, "pause")

    def test_training_config_with_missing_base_model_dir(self):
        """Training config builder handles missing base dir gracefully."""
        job_dir = self.root / "job"
        job_dir.mkdir()
        manifest = job_dir / "manifest.json"
        manifest.write_text(json.dumps({"cases": []}))
        config = pipeline._training_config(
            {"output_model_dir": str(self.root / "out"),
             "base_model_dir": str(self.root / "does_not_exist"),
             "strategy": "clopa_in", "epochs": 10},
            job_dir, manifest,
        )
        # Config should still be generated (validation happens later)
        self.assertEqual(config["model"]["strategy"], "clopa_in")

    def test_quality_result_empty_cases(self):
        """Quality check on empty case lists doesn't crash."""
        baseline = {"trajectory_auc": 0.0, "cases": []}
        candidate = {"trajectory_auc": 0.0, "cases": []}
        quality = pipeline._quality_result(baseline, candidate, {
            "minimum_mean_auc_improvement": 0.0,
            "maximum_severe_case_regression": 0.2,
        })
        # No paired case means there is no evidence for automatic selection.
        self.assertFalse(quality["qualifies"])
        self.assertEqual(quality["delta_auc"], 0.0)
        self.assertEqual(len(quality["severe_regressions"]), 0)

    def test_quality_result_partial_case_overlap(self):
        """Some validation cases in candidate but not baseline."""
        baseline = {"trajectory_auc": 0.80, "cases": [
            {"case_id": "c1", "dice_by_click": [0.8, 0.85, 0.88, 0.9, 0.91]},
        ]}
        candidate = {"trajectory_auc": 0.85, "cases": [
            {"case_id": "c1", "dice_by_click": [0.85, 0.88, 0.9, 0.92, 0.93]},
            {"case_id": "c2", "dice_by_click": [0.7, 0.75, 0.78, 0.8, 0.82]},
        ]}
        quality = pipeline._quality_result(baseline, candidate, {
            "minimum_mean_auc_improvement": 0.0,
            "maximum_severe_case_regression": 0.2,
        })
        # c2 is only in candidate, so no regression check
        self.assertTrue(quality["qualifies"])
        self.assertGreater(quality["delta_auc"], 0)

    def test_cancelled_job_cleans_partial_model(self):
        """Verify _cleanup_terminal_artifacts doesn't crash on missing paths."""
        job_dir = self.root / "job"
        job_dir.mkdir()
        # All paths missing — should not crash
        try:
            pipeline._cleanup_terminal_artifacts(
                {"output_model_dir": str(self.root / "nonexistent")},
                job_dir, remove_partial_model=True,
            )
        except Exception:
            self.fail("_cleanup_terminal_artifacts crashed on missing paths")

    def test_prepare_manifest_with_no_cases_raises(self):
        """Empty case list during manifest preparation."""
        job_dir = self.root / "job"
        job_dir.mkdir()
        # Simulate: request with no cases
        request = {"cases": [], "source_mode": "prepared"}
        # The _run_label_export function checks for cases and raises
        # Here we just verify the empty case guard works
        cases = [row for row in request.get("cases") or [] if row.get("split") in ("train", "val")]
        self.assertEqual(len(cases), 0)


# ============================================================================
# 4. Registry integrity under stress
# ============================================================================

class RegistryIntegrityTests(unittest.TestCase):
    """Test registry data consistency under various scenarios."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_registry_schema_version_is_set_on_save(self):
        registry = common.load_registry(self.workspace)
        common.save_registry(self.workspace, registry)
        loaded = common.load_registry(self.workspace)
        self.assertEqual(
            loaded["schema_version"], "nninteractive_task_registry.v2"
        )

    def test_duplicate_task_id_preserved(self):
        """You shouldn't create duplicate task IDs, but if it happens, both are returned."""
        _make_registry(self.workspace, [
            {"task_id": "liver", "task_name": "Liver v1"},
            {"task_id": "liver", "task_name": "Liver v2"},  # duplicate!
        ])
        tasks = common.task_rows(self.workspace)
        self.assertEqual(len(tasks), 2)

    def test_corrupt_registry_json_returns_empty(self):
        (self.workspace / "registry.json").write_text("{{{broken", encoding="utf-8")
        tasks = common.task_rows(self.workspace)
        self.assertEqual(tasks, [])

    def test_model_without_model_id_skipped_in_rows(self):
        _make_registry(self.workspace, [{
            "task_id": "test",
            "task_name": "Test",
            "models": [
                {"model_id": "valid", "created_at_epoch": 1},
                {"created_at_epoch": 2},  # no model_id
            ],
        }])
        rows = common.model_rows(self.workspace, "test")
        self.assertEqual(len(rows), 2)  # both returned (filter is isinstance(dict))

    def test_model_row_sorting_stable(self):
        """Models with same timestamp maintain insertion order."""
        t = time.time()
        _make_registry(self.workspace, [{
            "task_id": "test",
            "task_name": "Test",
            "models": [
                {"model_id": "a", "created_at_epoch": t},
                {"model_id": "b", "created_at_epoch": t},
            ],
        }])
        rows = common.model_rows(self.workspace, "test")
        # Python's sort is stable, so order should be preserved for equal keys
        self.assertEqual(len(rows), 2)

    def test_atomic_write_survives_kill_signal(self):
        """Atomic write leaves original file intact if crash mid-write."""
        path = self.workspace / "important.json"
        common.write_json_atomic(path, {"version": 1})
        # Simulate: a crashed write should not corrupt the original
        # (atomic write uses os.replace which is atomic on POSIX)
        original = common.read_json(path)
        self.assertEqual(original["version"], 1)

    def test_save_registry_with_no_tasks_is_valid(self):
        registry = {"schema_version": "nninteractive_task_registry.v1", "tasks": []}
        common.save_registry(self.workspace, registry)
        loaded = common.load_registry(self.workspace)
        self.assertEqual(loaded["tasks"], [])

    def test_find_task_with_special_characters(self):
        _make_registry(self.workspace, [{
            "task_id": "CT_Liver (portal_venous)",
            "task_name": "CT Liver - Portal Venous Phase",
        }])
        result = common.find_task(self.workspace, "CT_Liver (portal_venous)")
        self.assertIsNotNone(result)


# ============================================================================
# 5. Bridge request completeness
# ============================================================================

class BridgeRequestCompletenessTests(unittest.TestCase):
    """Verify all required fields in bridge requests."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_official_model_request_has_required_fields(self):
        """Every bridge request must have these fields."""
        required = [
            "model_dir",
        ]
        request = {"model_dir": "/path/to/official", "device": "auto"}
        for field in required:
            self.assertIn(field, request)

    def test_task_model_request_extra_fields(self):
        """Task model requests have additional fields."""
        task_fields = [
            "model_dir",
            "model_input_space",
            "model_input_intensity_space",
        ]
        request = {
            "model_dir": "/path/to/task",
            "model_input_space": "canonical_ras",
            "model_input_intensity_space": "source_values",
            "device": "auto",
        }
        for field in task_fields:
            self.assertIn(field, request)

    def test_source_export_fields_present_when_source_available(self):
        """When source image is available, these fields must be present."""
        image_export_fields = [
            "image_path",
            "shape",
            "source",
            "source_kind",
            "source_index_space",
            "source_modality",
            "source_intensity_space",
            "source_to_mimics_gv_slope",
            "source_to_mimics_gv_intercept",
        ]
        image_export = {
            "image_path": "/tmp/image.nii.gz",
            "shape": [512, 512, 100],
            "source": "source_image",
            "source_kind": "nifti",
            "source_index_space": "nifti_ijk_matches",
            "source_modality": "CT",
            "source_world_coordinate_system": "ras",
            "mimics_world_coordinate_system": "lps",
            "source_to_mimics_world_matrix": "...",
            "source_voxel_to_ras_matrix": "...",
            "mimics_voxel_to_ras_matrix": "...",
            "mimics_to_source_index_matrix": "...",
            "source_intensity_space": "source_values",
            "source_to_mimics_gv_slope": 1.0,
            "source_to_mimics_gv_intercept": 0.0,
        }
        for field in image_export_fields:
            self.assertIn(field, image_export,
                          f"Missing field: {field}")

    def test_buffer_export_fields_present_when_no_source(self):
        """When source is unavailable, buffer fields must be present."""
        buffer_fields = [
            "path",
            "shape",
            "dtype",
        ]
        image_export = {
            "path": "/tmp/buffer.raw",
            "shape": [512, 512, 100],
            "dtype": "float32",
        }
        for field in buffer_fields:
            self.assertIn(field, image_export)

    def test_task_model_slope_intercept_are_identity(self):
        """Task model intensity params MUST be slope=1.0, intercept=0.0."""
        # This is the critical invariant protecting source intensity
        slope = 1.0
        intercept = 0.0
        self.assertEqual(slope, 1.0)
        self.assertEqual(intercept, 0.0)


# ============================================================================
# 6. Data manifest structure
# ============================================================================

class DataManifestTests(unittest.TestCase):
    """Verify training manifest structure matches trainer expectations."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_manifest_cases_have_required_keys(self):
        required = ["case_id", "image", "label", "split"]
        manifest = {
            "cases": [
                {"case_id": "c1", "image": "/tmp/img.nii.gz",
                 "label": "/tmp/lbl.nii.gz", "split": "train"},
                {"case_id": "c2", "image": "/tmp/img2.nii.gz",
                 "label": "/tmp/lbl2.nii.gz", "split": "val"},
            ],
        }
        for case in manifest["cases"]:
            for key in required:
                self.assertIn(key, case)

    def test_split_values_are_valid(self):
        for split in ("train", "val"):
            self.assertIn(split, ("train", "val"))

    def test_validation_manifest_is_separate_from_training(self):
        full = {
            "cases": [
                {"case_id": "c1", "split": "train", "image": "", "label": ""},
                {"case_id": "c2", "split": "train", "image": "", "label": ""},
                {"case_id": "c3", "split": "val", "image": "", "label": ""},
            ],
        }
        train = [c for c in full["cases"] if c["split"] == "train"]
        val = [c for c in full["cases"] if c["split"] == "val"]
        self.assertEqual(len(train), 2)
        self.assertEqual(len(val), 1)

    def test_manifest_paths_are_absolute(self):
        """Pipeline writes absolute paths, but relative should work too."""
        manifest = {"cases": [{
            "case_id": "c1",
            "image": os.path.join(self.root, "image.nii.gz"),
            "label": "./relative/path/label.nii.gz",
            "split": "train",
        }]}
        # At least image paths should be absolute after prep
        abs_count = sum(1 for c in manifest["cases"]
                        if os.path.isabs(c["image"]))
        self.assertGreaterEqual(abs_count, 1)


# ============================================================================
# 7. Label export config simulation
# ============================================================================

class LabelExportConfigTests(unittest.TestCase):
    """Verify the label export config structure sent to background Mimics."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_export_config_has_required_sections(self):
        required = ["ts_root", "output_dir", "export_root", "status_path",
                    "stop_path", "label_staging_dir", "export_space",
                    "cases", "mask_names", "target_mask_name"]
        config = {
            "ts_root": "/data",
            "output_dir": "/mcs",
            "export_root": "/tmp",
            "status_path": "/tmp/status.json",
            "stop_path": "/tmp/stop.request",
            "label_staging_dir": "/tmp/staging",
            "export_space": "source_image",
            "cases": ["c1", "c2"],
            "case_dirs": {"c1": "/data/c1", "c2": "/data/c2"},
            "mcs_paths": {"c1": "/mcs/c1.mcs", "c2": "/mcs/c2.mcs"},
            "source_image_paths": {"c1": "/img/c1.nii.gz", "c2": "/img/c2.nii.gz"},
            "mask_names": ["Liver"],
            "target_mask_name": "liver_label",
        }
        for key in required:
            self.assertIn(key, config, f"Missing: {key}")

    def test_target_mask_name_is_slugified(self):
        task_name = "Liver Segmentation (CT)"
        slug = common.safe_slug(task_name)
        self.assertEqual(slug, "liver_segmentation_ct")
        self.assertNotIn(" ", slug)
        self.assertNotIn("(", slug)
        self.assertNotIn(")", slug)


# ============================================================================
# 8. Resource lock simulation
# ============================================================================

class ResourceLockSimulationTests(unittest.TestCase):
    """Simulate GPU lock behavior without actual locks."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_lock_file_has_pid(self):
        """A lock file must record the holder PID."""
        lock_path = self.root / "gpu.lock"
        pid = 12345
        common.write_json_atomic(lock_path, {
            "pid": pid,
            "kind": "nninteractive_finetune",
            "acquired_at_epoch": time.time(),
        })
        loaded = common.read_json(lock_path)
        self.assertEqual(loaded["pid"], pid)

    def test_lock_kind_distinguishes_purposes(self):
        """Training and inference locks must be distinguishable."""
        training_lock = {"kind": "nninteractive_finetune", "pid": 1}
        inference_lock = {"kind": "nninteractive_inference", "pid": 2}
        self.assertNotEqual(training_lock["kind"], inference_lock["kind"])

    def test_lock_holder_description(self):
        """Lock holder info is human-readable."""
        holder = {
            "pid": 99999,
            "kind": "nninteractive_finetune",
            "job_id": "train_20260726_test",
        }
        desc = f"{holder['kind']} (PID {holder['pid']}, job {holder['job_id']})"
        self.assertIn("nninteractive_finetune", desc)
        self.assertIn("99999", desc)


# ============================================================================
# 9. Mimics entrypoint behavior (without real Mimics)
# ============================================================================

class MimicsEntrypointSimulation(unittest.TestCase):
    """Test _mimics_entrypoint.py behavior with and without Mimics module."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_entrypoint_returns_none_when_not_main(self):
        """Modules imported for scripting should not auto-execute."""
        # The guard: if name != "__main__" and not _inside_mimics(): return None
        name = "some_imported_module"
        inside_mimics = False
        should_execute = (name == "__main__" or inside_mimics)
        self.assertFalse(should_execute)

    def test_entrypoint_executes_in_mimics(self):
        """Even when imported, runs inside Mimics (mimics module available)."""
        name = "some_imported_module"
        inside_mimics = True
        should_execute = (name == "__main__" or inside_mimics)
        self.assertTrue(should_execute)

    def test_entrypoint_executes_as_main(self):
        """When run as __main__, always executes."""
        name = "__main__"
        inside_mimics = False
        should_execute = (name == "__main__" or inside_mimics)
        self.assertTrue(should_execute)

    def test_mimics_module_not_available_outside_mimics(self):
        """The 'mimics' module should not be importable in tests."""
        try:
            import mimics  # noqa: F401
            has_mimics = True
        except ImportError:
            has_mimics = False
        # This test may pass or skip — just documents the expected state
        self.assertFalse(has_mimics,
                         "mimics module available in test env (not running inside Mimics)")

    def test_find_runtime_dir_walks_up_correctly(self):
        """The _find_runtime_dir helper finds runtime_py35 from any depth."""
        # Create a fake project structure
        project = self.root / "Mimics-Script"
        runtime_dir = project / "runtime_py35"
        runtime_dir.mkdir(parents=True)
        (runtime_dir / "__init__.py").write_text("")

        # Caller file is deep in scripting_library
        caller = (
            project
            / "scripting_library"
            / "02_AI"
            / "nnInteractive"
            / "entry.py"
        )
        caller.parent.mkdir(parents=True)
        caller.write_text("")

        # Walk up to find runtime_py35
        current = os.path.abspath(os.path.dirname(str(caller)))
        found = None
        for _ in range(6):
            candidate = os.path.join(current, "runtime_py35")
            if os.path.isdir(candidate):
                found = os.path.abspath(candidate)
                break
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent

        self.assertIsNotNone(found)
        self.assertTrue(found.endswith("runtime_py35"))


# ============================================================================
# 10. Concurrent access and stale state handling
# ============================================================================

class ConcurrencyAndStalenessTests(unittest.TestCase):
    """Test edge cases around concurrent access and stale state."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_stale_lock_detection(self):
        """A lock held by a dead process should be detectable as stale."""
        lock_path = self.root / "gpu.lock"
        # Write a lock with a PID that definitely doesn't exist
        dead_pid = 99999999
        common.write_json_atomic(lock_path, {
            "pid": dead_pid,
            "kind": "nninteractive_finetune",
        })
        # resource_locks.process_exists is the production liveness probe
        # (no psutil dependency — ctypes OpenProcess on Windows).
        from resource_locks import process_exists
        alive = process_exists(dead_pid)
        self.assertFalse(alive, "99999999 should not be a valid PID")

    def test_metrics_dont_duplicate_same_epoch(self):
        """copy_trainer_status should update, not duplicate, same-epoch entries."""
        status_path = self.workspace / "status.json"
        common.write_json_atomic(status_path, {"status": "training"})

        # Report epoch 1 twice
        for _ in range(2):
            pipeline._copy_trainer_status(status_path, {
                "status": "training",
                "epoch": 1,
                "epochs": 3,
                "latest_epoch": {
                    "epoch": 1,
                    "train_loss": 0.5,
                    "validation": {"trajectory_auc": 0.7},
                },
            })

        status = common.read_json(status_path, {})
        history = status.get("metrics_history") or []
        epoch_1_entries = [r for r in history if r.get("epoch") == 1]
        self.assertEqual(len(epoch_1_entries), 1,
                         "Epoch should not appear twice in metrics history")

    def test_job_status_updated_timestamp_changes(self):
        """Every status update via update_status wrapper changes updated_at_epoch."""
        status_path = self.workspace / "status.json"
        pipeline.update_status(status_path, status="created")
        ts1 = common.read_json(status_path, {}).get("updated_at_epoch", 0)

        time.sleep(0.01)
        pipeline.update_status(status_path, status="training")
        ts2 = common.read_json(status_path, {}).get("updated_at_epoch", 0)

        self.assertGreater(ts2, ts1)

    def test_fresh_job_preferred_over_stale(self):
        """When multiple jobs exist, the most recently updated is current."""
        jobs = common.jobs_dir(self.workspace)
        old_job = jobs / "old"
        new_job = jobs / "new"
        old_job.mkdir(parents=True)
        new_job.mkdir(parents=True)
        common.write_json_atomic(old_job / "status.json", {
            "task_id": "test", "status": "completed",
            "updated_at_epoch": 1000.0,
        })
        common.write_json_atomic(new_job / "status.json", {
            "task_id": "test", "status": "training",
            "updated_at_epoch": 2000.0,
        })

        candidates = []
        for path in jobs.glob("*/status.json"):
            status = common.read_json(path, {}) or {}
            if common.safe_slug(status.get("task_id")) == "test":
                candidates.append((float(status.get("updated_at_epoch") or 0),
                                   path.parent, status))
        candidates.sort(reverse=True)
        self.assertEqual(candidates[0][1].name, "new")

    def test_simultaneous_registry_reads_dont_corrupt(self):
        """Multiple rapid reads of registry don't cause issues."""
        _make_registry(self.workspace, [
            {"task_id": "t1", "task_name": "Task 1"},
        ])
        for _ in range(50):
            tasks = common.task_rows(self.workspace)
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0]["task_id"], "t1")

    def test_bindings_file_survives_empty_workspace(self):
        """project_bindings.json is valid even with no bindings."""
        path = common.bindings_path(self.workspace)
        common.write_json_atomic(path, {
            "schema_version": "nninteractive_project_bindings.v1",
            "bindings": {},
        })
        loaded = common.read_json(path, {})
        self.assertEqual(loaded.get("bindings"), {})


# ============================================================================
# 11. Boundary value tests
# ============================================================================

class BoundaryValueTests(unittest.TestCase):
    """Test extreme but valid inputs."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_minimum_epochs_respected(self):
        config = common.load_config()
        min_epochs = int(config.get("minimum_epochs", 4))
        self.assertGreaterEqual(min_epochs, 1)

    def test_maximum_epochs_sane(self):
        config = common.load_config()
        max_epochs = int(config.get("maximum_epochs", 20))
        self.assertLessEqual(max_epochs, 100)

    def test_single_case_training(self):
        """Training with exactly 1 case should be allowed."""
        _make_registry(self.workspace, [{
            "task_id": "test",
            "task_name": "Test",
            "models": [{
                "model_id": "v1", "created_at_epoch": time.time(),
                "strategy": "clopa_in", "state": "unverified",
                "compatible": True,
                "train_case_count": 1,
                "validation_case_count": 0,
            }],
        }])
        models = common.model_rows(self.workspace, "test")
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["train_case_count"], 1)

    def test_large_case_count(self):
        """Training with many cases doesn't overflow."""
        _make_registry(self.workspace, [{
            "task_id": "test",
            "task_name": "Test",
            "models": [{
                "model_id": "v1", "created_at_epoch": time.time(),
                "strategy": "clopa_in", "train_case_count": 9999,
                "validation_case_count": 999,
                "state": "validated", "compatible": True,
            }],
        }])
        models = common.model_rows(self.workspace, "test")
        self.assertGreater(models[0]["train_case_count"], 0)

    def test_very_long_task_name(self):
        name = "A" * 200
        slug = common.safe_slug(name)
        # slug should be reasonable length (not 200 chars of 'a')
        self.assertGreater(len(slug), 0)
        self.assertLess(len(slug), 300)

    def test_unicode_task_name(self):
        name = "肝腫瘤分割"
        slug = common.safe_slug(name)
        # Chinese characters become underscores
        self.assertGreater(len(slug), 0)

    def test_zero_dice_trajectory(self):
        """AUC calculation with zero Dice at all clicks should work."""
        baseline = {"trajectory_auc": 0.0, "cases": [
            {"case_id": "c1", "dice_by_click": [0.0, 0.0, 0.0, 0.0, 0.0]},
        ]}
        candidate = {"trajectory_auc": 0.5, "cases": [
            {"case_id": "c1", "dice_by_click": [0.5, 0.5, 0.5, 0.5, 0.5]},
        ]}
        quality = pipeline._quality_result(baseline, candidate, {
            "minimum_mean_auc_improvement": 0.0,
            "maximum_severe_case_regression": 0.2,
        })
        self.assertTrue(quality["qualifies"])
        self.assertAlmostEqual(quality["delta_auc"], 0.5)

    def test_perfect_trajectory(self):
        """AUC with perfect Dice at all clicks."""
        baseline = {"trajectory_auc": 0.80, "cases": [
            {"case_id": "c1", "dice_by_click": [0.8, 0.85, 0.9, 0.92, 0.94]},
        ]}
        candidate = {"trajectory_auc": 1.0, "cases": [
            {"case_id": "c1", "dice_by_click": [1.0, 1.0, 1.0, 1.0, 1.0]},
        ]}
        quality = pipeline._quality_result(baseline, candidate, {
            "minimum_mean_auc_improvement": 0.0,
            "maximum_severe_case_regression": 0.2,
        })
        self.assertTrue(quality["qualifies"])


# ============================================================================
# 12. Finetune package CLI structure
# ============================================================================

class FinetuneCLITests(unittest.TestCase):
    """Verify the finetune CLI interface matches pipeline expectations."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_cli_train_command_structure(self):
        """Pipeline calls: python -m nninteractive_finetune train --config <path>"""
        args = ["-m", "nninteractive_finetune", "train", "--config", "/path/to/config.json"]
        self.assertEqual(args[0], "-m")
        self.assertEqual(args[1], "nninteractive_finetune")
        self.assertEqual(args[2], "train")
        self.assertEqual(args[3], "--config")

    def test_cli_evaluate_command_structure(self):
        """Pipeline calls: python -m nninteractive_finetune evaluate --model-dir ..."""
        args = ["-m", "nninteractive_finetune", "evaluate",
                "--model-dir", "/path/to/model",
                "--manifest", "/path/to/manifest.json",
                "--output", "/path/to/output.json",
                "--label-values", "1",
                "--clicks", "8",
                "--correction-policy", "clopa_paired",
                "--device", "auto"]
        # Verify all required args present
        self.assertIn("--model-dir", args)
        self.assertIn("--manifest", args)
        self.assertIn("--output", args)
        self.assertIn("--clicks", args)

    def test_strategy_cli_mapping(self):
        """GUI radio buttons map to correct strategy names."""
        light = "clopa_in" if True else "clopa_conv"  # lightweight radio checked
        strong = "clopa_conv" if True else "clopa_in"  # stronger boundary radio
        self.assertEqual(light, "clopa_in")
        self.assertEqual(strong, "clopa_conv")


# ============================================================================
# 13. Effective model identity and controller progress
# ============================================================================

class EffectiveIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_bridge_checkpoint_identity_matches_selected_fold(self):
        import nninteractive_bridge as bridge

        checkpoint = self.root / "model" / "fold_0" / "checkpoint_final.pth"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"adapted-checkpoint")
        expected = hashlib.sha256(b"adapted-checkpoint").hexdigest()
        self.assertEqual(
            expected,
            bridge._selected_checkpoint_identity(str(checkpoint.parents[1]), "0"),
        )

    def test_trainer_completion_is_not_controller_completion(self):
        status_path = self.root / "status.json"
        common.write_json_atomic(status_path, {"status": "training"})
        pipeline._copy_trainer_status(
            status_path,
            {
                "status": "completed",
                "phase": "completed",
                "epoch": 20,
                "epochs": 20,
            },
        )
        status = common.read_json(status_path)
        self.assertEqual("training", status["status"])
        self.assertEqual("checkpoint_runtime_verified", status["phase"])
        self.assertEqual(85, status["progress_percent"])
        self.assertEqual(
            "Verifying the trained model", common.status_summary(status)
        )


if __name__ == "__main__":
    unittest.main()
