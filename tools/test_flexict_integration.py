#!/usr/bin/env python3
"""Offline tests for the FlexiCT Mimics integration (Phase 4).

Covers flexict_common.py contracts (config, workspace, registry, dataset-id
band, request normalization) and flexict_pipeline.py behavior (request
normalization, worker environment, splits, preprocess cache, sweep, job
lifecycle with mocked stage workers, model registration). No GPU, no Mimics,
no nnU-Net execution — the stage workers are mocked.
"""

import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import flexict_common as fc
import flexict_pipeline as fp


def _make_case(root: Path, case_id: str, image_shape=(6, 8, 8), spacing=(1.0, 1.0, 1.0)):
    """Write a tiny synthetic CT + label pair in a case directory."""
    import nibabel as nib
    import numpy as np

    case_dir = root / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    affine = np.diag(list(spacing) + [1.0])
    image = np.zeros(image_shape, dtype=np.float32)
    image[1:-1, 1:-1, 1:-1] = 100.0
    nib.save(nib.Nifti1Image(image, affine), str(case_dir / "ct.nii.gz"))
    label = np.zeros(image_shape, dtype=np.uint8)
    label[2:-2, 2:-2, 2:-2] = 1
    # prepare_source_grid_cases matches labels by alias name (kidney_left)
    # inside the case dir, or <case>_<alias> beside it — never "label.nii.gz".
    nib.save(nib.Nifti1Image(label, affine), str(case_dir / "kidney_left.nii.gz"))
    return case_dir


def _training_request(dataset_root: str, workspace: str, cases=None, **overrides):
    values = {
        "operation": "train",
        "task_name": "Kidney",
        "label_name": "kidney_left",
        "cases": cases or ["case01", "case02", "case03", "case04"],
        "dataset_root": dataset_root,
        "label_source": "dataset_masks",
        "workspace": workspace,
        "dataset_id": 750,
        "configuration": "2d",
        "epochs": 150,
        "val_cases": 1,
        "mimics_exe": "",
    }
    values.update(overrides)
    return values


class TestNormalizeFlexictRequest(unittest.TestCase):
    def test_minimal_request_defaults(self):
        request = fp.normalize_flexict_request(_training_request("X:/d", "W:/w"))
        self.assertEqual(request["operation"], "train")
        self.assertEqual(request["configuration"], "2d")
        self.assertEqual(request["epochs"], 150)
        self.assertEqual(request["dataset_id"], 750)
        self.assertEqual(len(request["labels"]), 1)
        self.assertEqual(request["labels"][0]["name"], "kidney_left")
        self.assertEqual(request["labels"][0]["id"], 1)

    def test_accepts_all_configurations(self):
        for value in ("2d", "3d_fullres", "pair", "auto"):
            request = fp.normalize_flexict_request(
                _training_request("X:/d", "W:/w", configuration=value))
            self.assertEqual(request["configuration"], value)

    def test_rejects_unknown_configuration(self):
        with self.assertRaises(ValueError):
            fp.normalize_flexict_request(
                _training_request("X:/d", "W:/w", configuration="4d"))

    def test_label_spec_is_single_binary(self):
        request = fp.normalize_flexict_request(_training_request("X:/d", "W:/w"))
        spec = request["labels"][0]
        self.assertEqual(spec["source_mode"], "alternatives")
        self.assertEqual(spec["aliases"], ["kidney_left"])

    def test_infer_operation_uses_infer_normalization(self):
        request = fp.normalize_flexict_request({
            "operation": "infer", "workspace": "W:/w",
            "model_manifest": "W:/w/models/m1/manifest.json",
            "image_path": "X:/d/case/ct.nii.gz",
            "output_path": "X:/out/pred.nii.gz",
            "source_modality": "ct",
        })
        self.assertEqual(request["operation"], "infer")
        self.assertEqual(request["source_modality"], "CT")

    def test_bad_operation_rejected(self):
        with self.assertRaises(ValueError):
            fp.normalize_flexict_request({"operation": "explode"})


class TestConfigurationResolution(unittest.TestCase):
    def test_2d_and_3d_are_concrete(self):
        request = {"configuration": "2d"}
        self.assertEqual(fp._flexict_configurations(request), ["2d"])
        request = {"configuration": "3d_fullres"}
        self.assertEqual(fp._flexict_configurations(request), ["3d_fullres"])

    def test_pair_expands_to_both(self):
        self.assertEqual(
            fp._flexict_configurations({"configuration": "pair"}),
            ["2d", "3d_fullres"])

    def test_auto_small_gpu_picks_2d(self):
        with mock.patch("mimics_label_export.detect_gpu_memory_gb", return_value=12.0):
            self.assertEqual(
                fp._flexict_configurations({"configuration": "auto"}), ["2d"])

    def test_auto_big_gpu_picks_pair(self):
        with mock.patch("mimics_label_export.detect_gpu_memory_gb", return_value=24.0):
            self.assertEqual(
                fp._flexict_configurations({"configuration": "auto"}),
                ["2d", "3d_fullres"])


class TestWorkerEnvironment(unittest.TestCase):
    def _request(self):
        return fp.normalize_flexict_request(_training_request("X:/d", "W:/w"))

    def _roots(self):
        from pathlib import Path
        return {
            "raw": Path("W:/w/runtime/nnUNet_raw"),
            "preprocessed": Path("W:/w/runtime/nnUNet_preprocessed"),
            "results": Path("W:/w/runtime/nnUNet_results"),
        }

    def test_environment_carries_trainer_contract(self):
        env = fp.flexict_worker_environment(
            self._request(), self._roots(), "2d")
        repo = fc.flexict_repo_dir(fc.load_config())
        weights, _source = fc.resolve_pretrained_dir(fc.load_config())
        self.assertEqual(env["nnUNet_extTrainer"], str(repo / "trainers"))
        self.assertEqual(env["FLEXICT_EXT_DIR"], str(repo / "flexict"))
        self.assertEqual(env["FLEXICT2D_CKPT"],
                         str(weights / "flexict_2d" / "model.safetensors"))
        self.assertEqual(env["FLEXICT3D_CKPT"],
                         str(weights / "flexict_3d" / "model.safetensors"))
        self.assertEqual(env["NUM_EPOCHS"], "150")
        self.assertEqual(env["nnUNet_compile"], "0")
        self.assertNotIn("MIRROR_DISABLE_AXES", env)

    def test_mirror_axes_forwarded(self):
        request = fp.normalize_flexict_request(
            _training_request("X:/d", "W:/w", mirror_disable_axes="1"))
        env = fp.flexict_worker_environment(request, self._roots(), "2d")
        self.assertEqual(env["MIRROR_DISABLE_AXES"], "1")

    def test_gpu_id_forwarded(self):
        request = fp.normalize_flexict_request(
            _training_request("X:/d", "W:/w", gpu_id="1"))
        env = fp.flexict_worker_environment(request, self._roots(), "2d")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")

    def test_nnunet_roots_point_at_workspace(self):
        env = fp.flexict_worker_environment(
            self._request(), self._roots(), "2d")
        self.assertIn("nnUNet_raw", env)
        self.assertIn("nnUNet_results", env)


class TestTrainValSplit(unittest.TestCase):
    def _rows(self, count):
        return [{"case_id": "case{:02d}".format(i),
                 "image": "i.nii.gz", "label": "l.nii.gz"}
                for i in range(count)]

    def test_val_cases_count_respected(self):
        train, val = fp._split_train_val(self._rows(5), {"val_cases": 1})
        self.assertEqual(len(val), 1)
        self.assertEqual(len(train), 4)
        self.assertNotIn(val[0], train)

    def test_val_never_exceeds_half_minus(self):
        train, val = fp._split_train_val(self._rows(4), {"val_cases": 3})
        self.assertEqual(len(val), 3)
        self.assertEqual(len(train), 1)

    def test_two_case_minimum(self):
        train, val = fp._split_train_val(self._rows(2), {"val_cases": 1})
        self.assertEqual(len(val), 1)
        self.assertEqual(len(train), 1)

    def test_explicit_val_case_selection_wins(self):
        rows = self._rows(4)
        train, val = fp._split_train_val(
            rows, {"val_cases": 1, "val_case_ids": ["case02"]})
        self.assertEqual(val, ["case02"])
        self.assertEqual(train, ["case00", "case01", "case03"])

    def test_val_at_least_one(self):
        # val_cases<=0 falls back to the default (3) which is then clamped to
        # len-1 for a 3-case request → 1 train + 2 val.
        train, val = fp._split_train_val(self._rows(3), {"val_cases": 0})
        self.assertEqual(len(val), 2)
        self.assertEqual(len(train), 1)


class TestScanFlexictCases(unittest.TestCase):
    def test_scan_reports_missing_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "case01").mkdir()
            (root / "case01" / "ct.nii.gz").write_bytes(b"x")
            rows = fp.scan_flexict_cases(root, "kidney_left")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["case_id"], "case01")
            self.assertFalse(rows[0]["checked"])
            self.assertIn("label missing", rows[0]["status"])

    def test_scan_full_case_is_usable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_case(root, "case01")
            rows = fp.scan_flexict_cases(root, "kidney_left")
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0]["checked"])
            self.assertEqual(rows[0]["status"], "OK")
            self.assertTrue(Path(rows[0]["label"]).name.startswith("kidney_left"))

    def test_scan_rejects_missing_root(self):
        with self.assertRaises(ValueError):
            fp.scan_flexict_cases(Path(tempfile.gettempdir()) / "definitely_missing", "x")


class TestPreprocessCache(unittest.TestCase):
    def test_roundtrip_valid_after_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Dataset750_Kidney" / "2d").mkdir(parents=True)
            request = {"spacing": None, "patch_size": None, "batch_size": None}
            valid, identity = fp._flexict_preprocess_cache_valid(
                root, "Dataset750_Kidney", "fp1", request, "2d")
            self.assertFalse(valid)
            fp._record_flexict_preprocess(root, "Dataset750_Kidney", "fp1", "2d", identity)
            valid, identity2 = fp._flexict_preprocess_cache_valid(
                root, "Dataset750_Kidney", "fp1", request, "2d")
            self.assertTrue(valid)
            self.assertEqual(identity, identity2)

    def test_different_fingerprint_invalidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Dataset750_Kidney" / "2d").mkdir(parents=True)
            request = {"spacing": None, "patch_size": None, "batch_size": None}
            _, identity = fp._flexict_preprocess_cache_valid(
                root, "Dataset750_Kidney", "fp1", request, "2d")
            fp._record_flexict_preprocess(root, "Dataset750_Kidney", "fp1", "2d", identity)
            valid, _ = fp._flexict_preprocess_cache_valid(
                root, "Dataset750_Kidney", "fp2", request, "2d")
            self.assertFalse(valid)

    def test_pair_identities_are_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "Dataset750_Kidney"
            (dataset / "2d").mkdir(parents=True)
            request = {"spacing": None, "patch_size": None, "batch_size": None}
            _, identity = fp._flexict_preprocess_cache_valid(
                root, "Dataset750_Kidney", "fp1", request, "2d")
            fp._record_flexict_preprocess(root, "Dataset750_Kidney", "fp1", "2d", identity)
            # 3d not yet preprocessed
            valid, _ = fp._flexict_preprocess_cache_valid(
                root, "Dataset750_Kidney", "fp1", request, "3d_fullres")
            self.assertFalse(valid)


class TestSweepExpiredJobs(unittest.TestCase):
    def _make_job(self, tmp, name, status, completed_at=None):
        jobs = Path(tmp) / "jobs" / name
        jobs.mkdir(parents=True)
        from nnunet_common import write_json_atomic

        payload = {"status": status, "job_id": name}
        if completed_at is not None:
            payload["completed_at_epoch"] = completed_at
        write_json_atomic(jobs / "status.json", payload)
        (jobs / "payload.bin").write_bytes(b"x" * 100)
        return jobs

    def test_old_terminal_jobs_are_pruned_keeping_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._make_job(tmp, "train_old", "completed", completed_at=time.time() - 40 * 86400)
            result = fp._sweep_expired_jobs(tmp)
            self.assertEqual(result["removed_jobs"], ["train_old"])
            self.assertTrue((Path(tmp) / "jobs" / "train_old" / "status.json").is_file())
            self.assertFalse((Path(tmp) / "jobs" / "train_old" / "payload.bin").exists())

    def test_recent_jobs_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._make_job(tmp, "train_new", "completed", completed_at=time.time())
            result = fp._sweep_expired_jobs(tmp)
            self.assertEqual(result["removed_jobs"], [])
            self.assertTrue((Path(tmp) / "jobs" / "train_new" / "payload.bin").exists())

    def test_live_jobs_never_pruned(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._make_job(tmp, "train_live", "training")
            old = time.time() - 40 * 86400
            os.utime(str(Path(tmp) / "jobs" / "train_live" / "status.json"), (old, old))
            result = fp._sweep_expired_jobs(tmp)
            self.assertEqual(result["removed_jobs"], [])
            self.assertTrue((Path(tmp) / "jobs" / "train_live" / "payload.bin").exists())

    def test_zero_retention_disables(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._make_job(tmp, "train_old", "completed", completed_at=0.0)
            result = fp._sweep_expired_jobs(tmp, {"job_retention_days": 0})
            self.assertEqual(result["removed_jobs"], [])
            self.assertTrue((Path(tmp) / "jobs" / "train_old" / "payload.bin").exists())


class TestEntryRouting(unittest.TestCase):
    """Scripting shells + runtime module wiring (no Mimics import)."""

    def test_runtime_module_has_all_actions(self):
        import ast

        source = (ROOT / "runtime_py35" / "flexict_mimics.py").read_text(
            encoding="utf-8")
        tree = ast.parse(source)
        actions = set()
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.startswith("BUTTON_"):
                        actions.add(target.id)
        self.assertIn("BUTTON_TRAIN", actions)
        self.assertIn("BUTTON_PREDICT", actions)
        self.assertIn("BUTTON_STATUS", actions)
        self.assertIn("BUTTON_STOP", actions)
        self.assertIn("BUTTON_ACTIVE_LEARNING", actions)

    def test_shells_route_to_flexict_mimics(self):
        shells = {
            "01_Train_Model.py": "BUTTON_TRAIN",
            "02_Predict_Current_Case.py": "BUTTON_PREDICT",
            "03_Active_Learning_Review.py": "BUTTON_ACTIVE_LEARNING",
        }
        for name, action in shells.items():
            source = (ROOT / "scripting_library" / "02_AI" / "FlexiCT" / name).read_text(
                encoding="utf-8")
            self.assertIn('"flexict_mimics"', source, name)
            self.assertIn(action, source, name)

    def test_runtime_module_is_py35_clean(self):
        import ast

        source = (ROOT / "runtime_py35" / "flexict_mimics.py").read_text(
            encoding="utf-8")
        self.assertNotIn('f"', source)
        self.assertNotIn("f'", source)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertNotIn(
                    "pathlib", [alias.name for alias in node.names],
                    "runtime_py35 module must not import pathlib")
            if isinstance(node, ast.ImportFrom):
                self.assertNotEqual(node.module, "pathlib")


class TestJobCreation(unittest.TestCase):
    def test_create_job_launches_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            request = fp.normalize_flexict_request(
                _training_request("X:/missing", str(workspace)))
            launched = {}

            class FakeProcess:
                pid = 4321

            def fake_popen(command, **kwargs):
                launched["command"] = command
                return FakeProcess()

            with mock.patch("flexict_pipeline.subprocess.Popen", side_effect=fake_popen), \
                 mock.patch("flexict_pipeline.process_start_marker", return_value="m"):
                job = fp.create_flexict_job(dict(request, dataset_root="X:/d"))
            self.assertIn("flexict_pipeline.py", launched["command"][1])
            self.assertIn("run", launched["command"])
            status = json.loads(
                (Path(job["job_dir"]) / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["schema_version"], fp.SCHEMA_VERSION)
            self.assertEqual(status["kind"], "train")
            self.assertEqual(status["launcher_pid"], 4321)
            saved = json.loads(
                (Path(job["job_dir"]) / "request.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["configuration"], "2d")

    def test_list_flexict_jobs_reads_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(fp.list_flexict_jobs(tmp), [])
            jobs = Path(tmp) / "jobs" / "train_x"
            jobs.mkdir(parents=True)
            from nnunet_common import write_json_atomic

            write_json_atomic(jobs / "status.json", {
                "job_id": "train_x", "kind": "train", "status": "completed",
                "created_at_epoch": 1.0, "updated_at_epoch": time.time(),
            })
            rows = fp.list_flexict_jobs(tmp)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["job_id"], "train_x")


class TestModelRegistration(unittest.TestCase):
    def test_register_creates_usable_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            request = fp.normalize_flexict_request(
                _training_request("X:/d", tmp))
            paths = fc.workspace_paths(tmp)
            # Simulate a trainer output folder under runtime results.
            model_dir = (paths["results"] / "Dataset750_Kidney"
                         / "flexict2d_Trainer__nnUNetPlans__2d")
            (model_dir / "fold_0").mkdir(parents=True)
            (model_dir / "fold_0" / "checkpoint_best.pth").write_bytes(b"x")
            (model_dir / "fold_0" / "checkpoint_final.pth").write_bytes(b"x")
            (model_dir / "plans.json").write_text("{}", encoding="utf-8")
            (model_dir / "dataset.json").write_text("{}", encoding="utf-8")
            roots = {"raw": paths["raw"], "preprocessed": paths["preprocessed"],
                     "results": paths["results"]}
            manifest = fp._register_flexict_model(
                request, roots, "2d", "fp1", "pair1", {"case_count": 4})
            self.assertEqual(manifest["configuration"], "2d")
            self.assertEqual(manifest["trainer"], "flexict2d_Trainer")
            self.assertEqual(manifest["pair_id"], "pair1")
            self.assertEqual(manifest["label_name"], "kidney_left")
            usable, why = fc.model_usability(manifest)
            self.assertTrue(usable, why)
            models = fc.load_models(tmp)
            self.assertEqual(len(models), 1)
            self.assertEqual(models[0]["model_id"], manifest["model_id"])


class TestJobLifecycle(unittest.TestCase):
    """run_training with all heavyweight stages mocked (no GPU, no nnU-Net)."""

    def _run_job(self, tmp, cases_count=4, configuration="2d", overrides=None):
        dataset_root = Path(tmp) / "dataset"
        dataset_root.mkdir(parents=True, exist_ok=True)
        cases = []
        for index in range(cases_count):
            case_id = "case{:02d}".format(index)
            _make_case(dataset_root, case_id)
            cases.append(case_id)
        workspace = Path(tmp) / "workspace"
        request = _training_request(
            str(dataset_root), str(workspace), cases,
            configuration=configuration, **(overrides or {}))
        request = fp.normalize_flexict_request(request)
        job_dir = Path(tmp) / "jobs" / request["job_id"]
        job_dir.mkdir(parents=True)
        from nnunet_common import write_json_atomic

        write_json_atomic(job_dir / "request.json", request)
        write_json_atomic(job_dir / "control.json", {"action": "run"})
        write_json_atomic(
            job_dir / "status.json",
            {"schema_version": fp.SCHEMA_VERSION, "job_id": request["job_id"],
             "kind": "train", "status": "launching", "phase": "launching",
             "created_at_epoch": time.time(), "updated_at_epoch": time.time()},
        )
        # Mock the actual worker spawn: pretend the preprocess and train stages
        # produce the expected trainer output folders.
        from flexict_common import workspace_paths

        def fake_worker(stage, params, request, roots, configuration, *args, **kwargs):
            if stage == "train":
                model_dir = fp._flexict_model_dir(roots, request, configuration)
                (model_dir / "fold_0").mkdir(parents=True, exist_ok=True)
                (model_dir / "fold_0" / "checkpoint_best.pth").write_bytes(b"x")
                (model_dir / "fold_0" / "checkpoint_final.pth").write_bytes(b"x")
                (model_dir / "plans.json").write_text("{}", encoding="utf-8")
                (model_dir / "dataset.json").write_text("{}", encoding="utf-8")
            else:
                preprocessed = roots["preprocessed"] / fp._dataset_name(request)
                (preprocessed / params["configuration"]).mkdir(parents=True, exist_ok=True)
            return {"status": "ok", "result": {"trained": stage == "train"}}

        with mock.patch.object(fp, "_spawn_flexict_worker", side_effect=fake_worker), \
             mock.patch("nnunet_pipeline._acquire_local_gpu", return_value=None), \
             mock.patch("flexict_pipeline._acquire_local_gpu", return_value=None):
            exit_code = fp.run_training(job_dir)
        from nnunet_common import read_json

        status = read_json(job_dir / "status.json", {}) or {}
        return exit_code, status, job_dir

    def test_completed_lifecycle_2d(self):
        with tempfile.TemporaryDirectory() as tmp:
            exit_code, status, job_dir = self._run_job(tmp, configuration="2d")
            self.assertEqual(exit_code, 0)
            self.assertEqual(status["status"], "completed")
            self.assertEqual(status["phase"], "completed")
            self.assertEqual(len(status["models"]), 1)
            self.assertEqual(status["models"][0]["configuration"], "2d")
            self.assertEqual(status["models"][0]["trainer"], "flexict2d_Trainer")
            # dataset written under the 750-799 band
            raw = list((Path(tmp) / "workspace" / "runtime" / "nnUNet_raw").iterdir())
            self.assertEqual(len(raw), 1)
            self.assertTrue(raw[0].name.startswith("Dataset750_"))
            # single-fold splits written
            splits = json.loads((Path(tmp) / "workspace" / "runtime" /
                                 "nnUNet_preprocessed" / raw[0].name /
                                 "splits_final.json").read_text(encoding="utf-8"))
            self.assertEqual(len(splits), 1)
            self.assertEqual(len(splits[0]["val"]), 1)
            self.assertEqual(len(splits[0]["train"]), 3)
            # registry holds the model
            models = fc.load_models(str(Path(tmp) / "workspace"))
            self.assertEqual(len(models), 1)

    def test_completed_lifecycle_pair(self):
        with tempfile.TocabularyDirectory() if False else tempfile.TemporaryDirectory() as tmp:
            exit_code, status, job_dir = self._run_job(tmp, configuration="pair")
            self.assertEqual(exit_code, 0)
            self.assertEqual(status["status"], "completed")
            self.assertEqual(len(status["models"]), 2)
            configs = sorted(m["configuration"] for m in status["models"])
            self.assertEqual(configs, ["2d", "3d_fullres"])
            pair_ids = {m["pair_id"] for m in status["models"]}
            self.assertEqual(len(pair_ids), 1, "pair models share pair_id")
            # load_pair finds both ends
            model_2d, model_3d = fc.load_pair(str(Path(tmp) / "workspace"))
            self.assertIsNotNone(model_2d)
            self.assertIsNotNone(model_3d)
            self.assertEqual(model_2d["configuration"], "2d")
            lookup = {m["configuration"]: m for m in status["models"]}
            self.assertEqual(model_3d["model_id"], lookup["3d_fullres"]["model_id"])

    def test_cancelled_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = Path(tmp) / "dataset"
            dataset_root.mkdir(parents=True)
            cases = []
            for index in range(4):
                case_id = "case{:02d}".format(index)
                _make_case(dataset_root, case_id)
                cases.append(case_id)
            workspace = Path(tmp) / "workspace"
            request = fp.normalize_flexict_request(
                _training_request(str(dataset_root), str(workspace), cases))
            job_dir = Path(tmp) / "jobs" / request["job_id"]
            job_dir.mkdir(parents=True)
            from nnunet_common import write_json_atomic

            write_json_atomic(job_dir / "request.json", request)
            write_json_atomic(job_dir / "control.json", {"action": "cancel"})
            write_json_atomic(
                job_dir / "status.json",
                {"job_id": request["job_id"], "status": "launching",
                 "created_at_epoch": time.time(), "updated_at_epoch": time.time()},
            )
            exit_code = fp.run_training(job_dir)
            from nnunet_common import read_json

            status = read_json(job_dir / "status.json", {}) or {}
            self.assertEqual(exit_code, 0)
            self.assertEqual(status["status"], "cancelled")

    def test_failure_records_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = Path(tmp) / "dataset"
            dataset_root.mkdir(parents=True)
            cases = []
            for index in range(4):
                case_id = "case{:02d}".format(index)
                _make_case(dataset_root, case_id)
                cases.append(case_id)
            workspace = Path(tmp) / "workspace"
            request = fp.normalize_flexict_request(
                _training_request(str(dataset_root), str(workspace), cases))
            job_dir = Path(tmp) / "jobs" / request["job_id"]
            job_dir.mkdir(parents=True)
            from nnunet_common import write_json_atomic

            write_json_atomic(job_dir / "request.json", request)
            write_json_atomic(job_dir / "control.json", {"action": "run"})
            write_json_atomic(
                job_dir / "status.json",
                {"job_id": request["job_id"], "status": "launching",
                 "created_at_epoch": time.time(), "updated_at_epoch": time.time()},
            )
            with mock.patch.object(
                    fp, "_spawn_flexict_worker",
                    side_effect=RuntimeError("worker exploded")), \
                 mock.patch("flexict_pipeline._acquire_local_gpu", return_value=None):
                exit_code = fp.run_training(job_dir)
            from nnunet_common import read_json

            status = read_json(job_dir / "status.json", {}) or {}
            self.assertEqual(exit_code, 1)
            self.assertEqual(status["status"], "failed")
            self.assertIn("worker exploded", status["error"])
            self.assertTrue(status.get("traceback"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
