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

    def test_missing_dataset_id_suggested_from_flexict_band(self):
        """A request without dataset_id must NOT land in the nnU-Net band.

        The nnU-Net normalizer defaults a missing dataset_id to 701; FlexiCT
        owns 750-799, so an unset id must be suggested from that band instead
        (regression: programmatic callers without an id silently created
        Dataset701_* in the FlexiCT workspace).
        """
        with tempfile.TemporaryDirectory() as workspace:
            values = _training_request("X:/d", workspace)
            values.pop("dataset_id")
            request = fp.normalize_flexict_request(values)
            self.assertGreaterEqual(request["dataset_id"], 750)
            self.assertLessEqual(request["dataset_id"], 799)

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

    def test_windows_forces_single_process_da_train_stage_only(self):
        """Windows TRAIN workers must pin nnUNet_n_proc_DA=0.

        nnU-Net's spawn'd DA workers (12 by default, ~1GB commit each) crash
        with OpenBLAS allocation failures on RAM-tight Windows workstations
        and poison the main process's CUDA context; the fix (same as the
        MedDINOv3 experiments) is single-process data augmentation locally.
        Preprocess workers must NOT set it: nnunetv2 2.8.0's planner feeds
        the value to torch.set_num_threads and rejects 0. Non-Windows
        (remote Linux containers) never sets it.
        """
        import os as _os
        train_env = fp.flexict_worker_environment(
            self._request(), self._roots(), "2d", stage="train")
        preprocess_env = fp.flexict_worker_environment(
            self._request(), self._roots(), "2d", stage="preprocess")
        if _os.name == "nt":
            self.assertEqual(train_env.get("nnUNet_n_proc_DA"), "0")
        else:
            self.assertNotIn("nnUNet_n_proc_DA", train_env)
        self.assertNotIn("nnUNet_n_proc_DA", preprocess_env)

    def test_environment_caps_blas_threads_every_stage(self):
        """FlexiCT stages must cap BLAS threads (same A10 fix as nnunet).

        FlexiCT's preprocess runs the same spawn.Pool fingerprint extraction
        as the nnU-Net pipeline, and uncapped OpenBLAS threads are what killed
        those workers on RAM-tight workstations. flexict_worker_environment
        replaces _worker_environment wholesale for FlexiCT stages, so the caps
        must live here too.
        """
        for stage in ("preprocess", "train", "infer"):
            env = fp.flexict_worker_environment(
                self._request(), self._roots(), "2d", stage=stage)
            for blas_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                             "OPENBLAS_NUM_THREADS"):
                self.assertEqual(env.get(blas_var), "1",
                                 "{} stage must cap {}".format(stage,
                                                               blas_var))

    def test_infer_environment_caps_blas_threads(self):
        """FlexiCT inference workers replace _worker_environment too."""
        env = fp.flexict_infer_environment(
            self._request(), self._roots(), "2d")
        for blas_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                         "OPENBLAS_NUM_THREADS"):
            self.assertEqual(env.get(blas_var), "1",
                             "infer stage must cap {}".format(blas_var))

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


class TestLoadModelsManifestPath(unittest.TestCase):
    def test_registry_rows_derive_manifest_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            from nnunet_common import write_json_atomic

            workspace = Path(tmp) / "workspace"
            model_dir = workspace / "models" / "m1"
            (model_dir / "fold_0").mkdir(parents=True)
            (model_dir / "flexict_model_manifest.json").write_text("{}", encoding="utf-8")
            registry = workspace / "registry.json"
            registry.parent.mkdir(parents=True, exist_ok=True)
            write_json_atomic(registry, {
                "schema_version": fc.SCHEMA_VERSION,
                "recommended_model_id": "",
                "models": [{"model_id": "m1", "model_dir": str(model_dir),
                            "created_at_epoch": 1.0}],
            })
            models = fc.load_models(str(workspace))
            self.assertEqual(len(models), 1)
            self.assertEqual(
                models[0]["manifest_path"],
                str(model_dir / "flexict_model_manifest.json"))


class TestFlexictInferEnvironment(unittest.TestCase):
    def test_infer_environment_trainer_seam_only(self):
        """Inference needs the trainer seam but NOT the pretrained backbone
        checkpoints (the trained checkpoint already holds fine-tuned weights)."""
        with tempfile.TemporaryDirectory() as tmp:
            roots = {
                "raw": Path(tmp) / "raw",
                "preprocessed": Path(tmp) / "preprocessed",
                "results": Path(tmp) / "results",
            }
            request = {"gpu_id": "1"}
            environment = fp.flexict_infer_environment(request, roots, "2d")
            self.assertEqual(environment["nnUNet_raw"], str(roots["raw"]))
            self.assertEqual(environment["nnUNet_preprocessed"], str(roots["preprocessed"]))
            self.assertEqual(environment["nnUNet_results"], str(roots["results"]))
            self.assertTrue(environment["nnUNet_extTrainer"].endswith(
                os.path.join("flexict-finetune", "trainers")))
            self.assertEqual(environment["nnUNet_compile"], "0")
            self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "1")
            self.assertNotIn("FLEXICT2D_CKPT", environment)
            self.assertNotIn("FLEXICT3D_CKPT", environment)
            self.assertNotIn("FLEXICT_EXT_DIR", environment)
            self.assertNotIn("NUM_EPOCHS", environment)


class TestInferenceLifecycle(unittest.TestCase):
    """run_inference with the worker and GPU lock mocked (no GPU, no nnU-Net)."""

    def _setup_infer_job(self, tmp):
        import nibabel as nib
        import numpy as np

        dataset_root = Path(tmp) / "dataset"
        case_dir = _make_case(dataset_root, "case01")
        image_path = case_dir / "ct.nii.gz"
        workspace = Path(tmp) / "workspace"
        # A registered, usable FlexiCT model.
        model_dir = (workspace / "runtime" / "nnUNet_results"
                     / "Dataset750_Kidney" / "flexict2d_Trainer__nnUNetPlans__2d")
        (model_dir / "fold_0").mkdir(parents=True)
        (model_dir / "fold_0" / "checkpoint_best.pth").write_bytes(b"x")
        (model_dir / "plans.json").write_text("{}", encoding="utf-8")
        (model_dir / "dataset.json").write_text("{}", encoding="utf-8")
        manifest = {
            "model_id": "flexict_m1", "model_dir": str(model_dir),
            "configuration": "2d", "trainer": "flexict2d_Trainer",
            "label_name": "kidney_left", "task_id": "Kidney",
            "task_name": "Kidney", "dataset_id": 750,
        }
        manifest_path = model_dir / "flexict_model_manifest.json"
        from nnunet_common import write_json_atomic

        write_json_atomic(manifest_path, manifest)
        source = nib.load(str(image_path))
        request = {
            "operation": "infer",
            "job_id": "infer_test01",
            "workspace": str(workspace),
            "model_manifest": str(manifest_path),
            "image_path": str(image_path),
            "label_name": "kidney_left",
            "source_geometry_expected": {
                "source_shape": list(source.shape[:3]),
                "source_voxel_to_ras_matrix": source.affine.tolist(),
            },
        }
        request = fp.normalize_flexict_request(request)
        job_dir = Path(tmp) / "jobs" / request["job_id"]
        job_dir.mkdir(parents=True)
        write_json_atomic(job_dir / "request.json", request)
        write_json_atomic(job_dir / "control.json", {"action": "run"})
        write_json_atomic(
            job_dir / "status.json",
            {"schema_version": fp.SCHEMA_VERSION, "job_id": request["job_id"],
             "kind": "infer", "status": "launching",
             "created_at_epoch": time.time(), "updated_at_epoch": time.time()},
        )
        return job_dir, workspace, source

    def _fake_worker_factory(self, capture, shape_override=None):
        """Worker mock: writes a prediction NIfTI on the source grid."""
        import nibabel as nib
        import numpy as np

        def fake_worker(stage, params, *args, **kwargs):
            capture["stage"] = stage
            capture["params"] = dict(params)
            if stage != "infer":
                raise RuntimeError("unexpected stage: {}".format(stage))
            source = nib.load(str(params["input_path"]))
            shape = shape_override or source.shape[:3]
            affine = source.affine
            prediction = np.zeros(shape, dtype=np.uint8)
            prediction[2:-2, 2:-2, 2:-2] = 1
            output = Path(params["output_path"])
            output.parent.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(prediction, affine), str(output))
            return {"status": "ok", "result": {}}

        return fake_worker

    def test_completed_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            job_dir, workspace, source = self._setup_infer_job(tmp)
            capture = {}
            with mock.patch.object(
                    fp, "_spawn_worker",
                    side_effect=self._fake_worker_factory(capture)), \
                 mock.patch("flexict_pipeline._acquire_local_gpu",
                            return_value=None):
                exit_code = fp.run_inference(job_dir)
            from nnunet_common import read_json

            self.assertEqual(exit_code, 0)
            status = read_json(job_dir / "status.json", {}) or {}
            self.assertEqual(status["status"], "completed")
            self.assertEqual(status["phase"], "completed")
            self.assertEqual(status["label_name"], "kidney_left")
            self.assertEqual(status["model_id"], "flexict_m1")
            # locked recipe: best checkpoint, no TTA
            self.assertEqual(capture["stage"], "infer")
            self.assertEqual(capture["params"]["checkpoint_name"],
                             "checkpoint_best.pth")
            self.assertTrue(capture["params"]["disable_tta"])
            self.assertIn("flexict2d_Trainer", capture["params"]["model_folder"])
            self.assertEqual(
                capture["params"]["input_path"],
                str(job_dir / "input" / "source_image.nii.gz"))
            self.assertEqual(
                capture["params"]["output_path"],
                str(job_dir / "prediction.nii.gz"))
            # prediction written next to the job on the source grid
            prediction = job_dir / "prediction.nii.gz"
            self.assertTrue(prediction.is_file())
            import nibabel as nib
            import numpy as np

            result = nib.load(str(prediction))
            self.assertEqual(tuple(result.shape[:3]), tuple(source.shape[:3]))
            self.assertTrue(np.allclose(result.affine, source.affine, atol=1e-4))
            self.assertIn(str(job_dir / "input" / "source_image.nii.gz"),
                          status.get("inference_image_path") or "")

    def test_grid_mismatch_marks_failure(self):
        """A prediction on the wrong grid must never reach Mimics."""
        with tempfile.TemporaryDirectory() as tmp:
            job_dir, workspace, source = self._setup_infer_job(tmp)
            capture = {}
            bad_shape = (source.shape[0] + 2,) + tuple(source.shape[1:])
            with mock.patch.object(
                    fp, "_spawn_worker",
                    side_effect=self._fake_worker_factory(
                        capture, shape_override=bad_shape)), \
                 mock.patch("flexict_pipeline._acquire_local_gpu",
                            return_value=None):
                exit_code = fp.run_inference(job_dir)
            from nnunet_common import read_json

            self.assertEqual(exit_code, 1)
            status = read_json(job_dir / "status.json", {}) or {}
            self.assertEqual(status["status"], "failed")
            self.assertIn("grid", str(status.get("error") or "").lower())

    def test_stale_source_geometry_rejected_before_gpu(self):
        """If the on-disk image no longer matches the launch-time geometry,
        the job fails before any GPU work happens."""
        with tempfile.TemporaryDirectory() as tmp:
            job_dir, workspace, source = self._setup_infer_job(tmp)
            request = json.loads(
                (job_dir / "request.json").read_text(encoding="utf-8"))
            request["source_geometry_expected"] = {
                "source_shape": [3, 3, 3],
                "source_voxel_to_ras_matrix": source.affine.tolist(),
            }
            from nnunet_common import write_json_atomic

            write_json_atomic(job_dir / "request.json", request)
            with mock.patch.object(fp, "_spawn_worker") as never_called, \
                 mock.patch("flexict_pipeline._acquire_local_gpu",
                            return_value=None):
                exit_code = fp.run_inference(job_dir)
            self.assertFalse(never_called.called)
            self.assertEqual(exit_code, 1)
            from nnunet_common import read_json

            status = read_json(job_dir / "status.json", {}) or {}
            self.assertEqual(status["status"], "failed")
            self.assertIn("geometry", str(status.get("error") or "").lower())


class TestInferJobCreation(unittest.TestCase):
    def test_create_infer_job_fills_output_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            request = fp.normalize_flexict_request({
                "operation": "infer",
                "workspace": str(workspace),
                "model_manifest": "X:/missing/manifest.json",
                "image_path": "X:/missing/ct.nii.gz",
                "label_name": "kidney_left",
            })
            launched = {}

            class FakeProcess:
                pid = 8642

            def fake_popen(command, **kwargs):
                launched["command"] = command
                return FakeProcess()

            with mock.patch("flexict_pipeline.subprocess.Popen",
                            side_effect=fake_popen), \
                 mock.patch("flexict_pipeline.process_start_marker",
                            return_value="m"):
                job = fp.create_flexict_job(request)
            saved = json.loads(
                (Path(job["job_dir"]) / "request.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["output_path"],
                             str(Path(job["job_dir"]) / "prediction.nii.gz"))
            status = json.loads(
                (Path(job["job_dir"]) / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["kind"], "infer")


class TestResolveFlexictPair(unittest.TestCase):
    def test_explicit_manifests_win(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            model_dir = workspace / "models" / "m2"
            (model_dir / "fold_0").mkdir(parents=True)
            (model_dir / "fold_0" / "checkpoint_best.pth").write_bytes(b"x")
            (model_dir / "plans.json").write_text("{}", encoding="utf-8")
            (model_dir / "dataset.json").write_text("{}", encoding="utf-8")
            manifest_2d = {
                "model_id": "m2d", "model_dir": str(model_dir),
                "configuration": "2d",
            }
            manifest_3d = {
                "model_id": "m3d", "model_dir": str(model_dir),
                "configuration": "3d_fullres",
            }
            from nnunet_common import write_json_atomic

            path_2d = workspace / "m2d.json"
            path_3d = workspace / "m3d.json"
            write_json_atomic(path_2d, manifest_2d)
            write_json_atomic(path_3d, manifest_3d)
            pair = fp._resolve_flexict_pair(
                {"model_manifest_2d": str(path_2d),
                 "model_manifest_3d": str(path_3d)},
                workspace,
            )
            self.assertEqual(pair["2d"]["model_id"], "m2d")
            self.assertEqual(pair["3d_fullres"]["model_id"], "m3d")

    def test_one_manifest_only_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            path = workspace / "m2d.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            from nnunet_common import write_json_atomic

            write_json_atomic(path, {"configuration": "2d"})
            with self.assertRaises(ValueError):
                fp._resolve_flexict_pair(
                    {"model_manifest_2d": str(path)}, workspace)

    def test_registry_pair_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            for configuration in ("2d", "3d_fullres"):
                model_dir = (workspace / "models" / configuration)
                (model_dir / "fold_0").mkdir(parents=True)
                (model_dir / "fold_0" / "checkpoint_best.pth").write_bytes(b"x")
                (model_dir / "plans.json").write_text("{}", encoding="utf-8")
                (model_dir / "dataset.json").write_text("{}", encoding="utf-8")
            from nnunet_common import write_json_atomic

            write_json_atomic(workspace / "registry.json", {
                "schema_version": fc.SCHEMA_VERSION,
                "models": [
                    {"model_id": "p2d", "model_dir": str(
                        workspace / "models" / "2d"),
                     "configuration": "2d", "pair_id": "pairA",
                     "created_at_epoch": 1.0},
                    {"model_id": "p3d", "model_dir": str(
                        workspace / "models" / "3d_fullres"),
                     "configuration": "3d_fullres", "pair_id": "pairA",
                     "created_at_epoch": 2.0},
                ],
            })
            pair = fp._resolve_flexict_pair({}, workspace)
            self.assertEqual(pair["2d"]["model_id"], "p2d")
            self.assertEqual(pair["3d_fullres"]["model_id"], "p3d")

    def test_no_pair_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                fp._resolve_flexict_pair({}, Path(tmp) / "workspace")


class TestActiveLearningLifecycle(unittest.TestCase):
    """run_active_learning with workers + uncertainty mocked where heavy."""

    def _setup_al_job(self, tmp, cases_count=3):
        dataset_root = Path(tmp) / "pool"
        dataset_root.mkdir(parents=True, exist_ok=True)
        cases = []
        for index in range(cases_count):
            case_id = "case{:02d}".format(index)
            _make_case(dataset_root, case_id)
            cases.append(case_id)
        workspace = Path(tmp) / "workspace"
        # A registered pair.
        manifests = {}
        for configuration in ("2d", "3d_fullres"):
            model_dir = (
                workspace / "runtime" / "nnUNet_results"
                / "Dataset750_Kidney" / "flexict{}_Trainer__nnUNetPlans__{}".format(
                    "2d" if configuration == "2d" else "3d",
                    configuration))
            (model_dir / "fold_0").mkdir(parents=True)
            (model_dir / "fold_0" / "checkpoint_best.pth").write_bytes(b"x")
            (model_dir / "plans.json").write_text("{}", encoding="utf-8")
            (model_dir / "dataset.json").write_text("{}", encoding="utf-8")
            manifests[configuration] = model_dir / "flexict_model_manifest.json"
        from nnunet_common import write_json_atomic

        for configuration, manifest_path in manifests.items():
            write_json_atomic(manifest_path, {
                "model_id": "m_{}".format(configuration),
                "model_dir": str(manifest_path.parent),
                "configuration": configuration,
                "trainer": "flexict2d_Trainer" if configuration == "2d"
                else "flexict3d_Trainer",
                "label_name": "kidney_left",
                "task_id": "Kidney", "task_name": "Kidney",
            })
        request = fp.normalize_flexict_request({
            "operation": "active_learning",
            "workspace": str(workspace),
            "dataset_root": str(dataset_root),
            "label_name": "kidney_left",
            "model_manifest_2d": str(manifests["2d"]),
            "model_manifest_3d": str(manifests["3d_fullres"]),
        })
        job_dir = Path(tmp) / "jobs" / request["job_id"]
        job_dir.mkdir(parents=True)
        write_json_atomic(job_dir / "request.json", request)
        write_json_atomic(job_dir / "control.json", {"action": "run"})
        write_json_atomic(
            job_dir / "status.json",
            {"schema_version": fp.SCHEMA_VERSION, "job_id": request["job_id"],
             "kind": "active_learning", "status": "launching",
             "created_at_epoch": time.time(), "updated_at_epoch": time.time()},
        )
        return job_dir, workspace, cases

    def _fake_predict_worker(self, capture):
        import nibabel as nib
        import numpy as np

        def fake_worker(stage, params, *args, **kwargs):
            if stage != "infer":
                raise RuntimeError("unexpected stage {}".format(stage))
            capture.append(dict(params))
            source_dir = Path(params["input_path"])
            output = Path(params["output_path"])
            output.mkdir(parents=True, exist_ok=True)
            model_folder = Path(params["model_folder"])
            configuration = "2d" if "2d" in model_folder.name else "3d_fullres"
            for image_path in sorted(source_dir.glob("*.nii.gz")):
                source = nib.load(str(image_path))
                prediction = np.zeros(source.shape[:3], dtype=np.uint8)
                if configuration == "2d":
                    prediction[2:-2, 2:-2, 2:-2] = 1
                else:
                    prediction[1:-1, 1:-1, 1:-1] = 1  # disagrees with 2D
                nib.save(nib.Nifti1Image(prediction, source.affine),
                         str(output / image_path.name))
            return {"status": "ok", "result": {}}

        return fake_worker

    def test_completed_al_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            job_dir, workspace, cases = self._setup_al_job(tmp)
            capture = []
            with mock.patch.object(
                    fp, "_spawn_worker",
                    side_effect=self._fake_predict_worker(capture)), \
                 mock.patch("flexict_pipeline._acquire_local_gpu",
                            return_value=None):
                exit_code = fp.run_active_learning(job_dir)
            self.assertEqual(exit_code, 0)
            from nnunet_common import read_json

            status = read_json(job_dir / "status.json", {}) or {}
            self.assertEqual(status["status"], "completed")
            self.assertEqual(status["phase"], "completed")
            self.assertEqual(status["uncertainty_method"], "disagreement")
            # both pair ends ran with the locked recipe
            self.assertEqual(len(capture), 2)
            for params in capture:
                self.assertEqual(params["checkpoint_name"], "checkpoint_best.pth")
                self.assertTrue(params["disable_tta"])
            # ranking sorted descending by integrated score
            ranking = status.get("ranking") or []
            self.assertEqual(len(ranking), len(cases))
            scores = [row["integrated"] for row in ranking]
            self.assertEqual(scores, sorted(scores, reverse=True))
            self.assertEqual(status["top_case"], ranking[0]["case"])
            # uncertainty artifacts on disk
            uncertainty_dir = job_dir / "uncertainty"
            self.assertTrue((uncertainty_dir / "ranking.csv").is_file())
            bands = status.get("uncertainty_bands") or {}
            # a 2-model disagreement pair can only reach level 5 on the x10
            # scale, so the high threshold clamps to the max present level
            self.assertEqual(bands["high_threshold"], 5)
            self.assertEqual(bands["moderate_threshold"], 3)
            # both bands written for every case (2D/3D predictions disagree)
            self.assertEqual(bands["written"]["moderate"], len(cases))
            self.assertEqual(bands["written"]["high"], len(cases))
            # per-case source geometry recorded for the overlay contract
            geometries = read_json(
                job_dir / "input_geometries.json", {}) or {}
            for case_id in cases:
                self.assertIn(case_id, geometries.get("cases") or {})

    def test_al_fails_without_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = Path(tmp) / "pool"
            _make_case(dataset_root, "case00")
            request = fp.normalize_flexict_request({
                "operation": "active_learning",
                "workspace": str(Path(tmp) / "workspace"),
                "dataset_root": str(dataset_root),
                "label_name": "kidney_left",
            })
            job_dir = Path(tmp) / "jobs" / request["job_id"]
            job_dir.mkdir(parents=True)
            from nnunet_common import write_json_atomic

            write_json_atomic(job_dir / "request.json", request)
            write_json_atomic(job_dir / "control.json", {"action": "run"})
            write_json_atomic(
                job_dir / "status.json",
                {"schema_version": fp.SCHEMA_VERSION,
                 "job_id": request["job_id"],
                 "kind": "active_learning", "status": "launching",
                 "created_at_epoch": time.time(),
                 "updated_at_epoch": time.time()})
            exit_code = fp.run_active_learning(job_dir)
            from nnunet_common import read_json

            self.assertEqual(exit_code, 1)
            status = read_json(job_dir / "status.json", {}) or {}
            self.assertEqual(status["status"], "failed")
            self.assertIn("pair", str(status.get("error") or "").lower())


class TestMainRouting(unittest.TestCase):
    def test_operation_infer_routes_to_run_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmp) / "job"
            job_dir.mkdir()
            from nnunet_common import write_json_atomic

            write_json_atomic(job_dir / "request.json",
                              {"operation": "infer", "job_id": "x"})
            with mock.patch.object(fp, "run_inference",
                                   return_value=0) as infer_mock, \
                 mock.patch.object(fp, "run_training",
                                   return_value=0) as train_mock, \
                 mock.patch("sys.argv",
                            ["flexict_pipeline.py", "run",
                             "--job-dir", str(job_dir)]):
                self.assertEqual(fp.main(), 0)
            self.assertEqual(infer_mock.call_count, 1)
            self.assertEqual(train_mock.call_count, 0)
            self.assertEqual(infer_mock.call_args[0][0], job_dir)

    def test_operation_train_routes_to_run_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmp) / "job"
            job_dir.mkdir()
            from nnunet_common import write_json_atomic

            write_json_atomic(job_dir / "request.json",
                              {"operation": "train", "job_id": "x"})
            with mock.patch.object(fp, "run_inference",
                                   return_value=0) as infer_mock, \
                 mock.patch.object(fp, "run_training",
                                   return_value=0) as train_mock, \
                 mock.patch("sys.argv",
                            ["flexict_pipeline.py", "run",
                             "--job-dir", str(job_dir)]):
                self.assertEqual(fp.main(), 0)
            self.assertEqual(infer_mock.call_count, 0)
            self.assertEqual(train_mock.call_count, 1)

    def test_operation_active_learning_routes_to_run_active_learning(self):
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmp) / "job"
            job_dir.mkdir()
            from nnunet_common import write_json_atomic

            write_json_atomic(job_dir / "request.json",
                              {"operation": "active_learning", "job_id": "x"})
            with mock.patch.object(fp, "run_active_learning",
                                   return_value=0) as al_mock, \
                 mock.patch.object(fp, "run_inference",
                                   return_value=0) as infer_mock, \
                 mock.patch.object(fp, "run_training",
                                   return_value=0) as train_mock, \
                 mock.patch("sys.argv",
                            ["flexict_pipeline.py", "run",
                             "--job-dir", str(job_dir)]):
                self.assertEqual(fp.main(), 0)
            self.assertEqual(al_mock.call_count, 1)
            self.assertEqual(infer_mock.call_count, 0)
            self.assertEqual(train_mock.call_count, 0)


class TestActiveLearningReviewUI(unittest.TestCase):
    """Offline helpers of the review window (no Qt)."""

    def test_load_ranking_prefers_status_then_csv(self):
        from flexict_active_learning_ui import load_ranking
        from nnunet_common import write_json_atomic

        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmp) / "job"
            job_dir.mkdir()
            # no status, no csv -> empty
            self.assertEqual(load_ranking(job_dir), [])
            # csv fallback
            uncertainty = job_dir / "uncertainty"
            uncertainty.mkdir()
            rows = [
                "case,integrated,uncertain_vol,max\n",
                "caseB,9.0,100.0,1.0\n",
                "caseA,1.0,10.0,0.5\n",
            ]
            (uncertainty / "ranking.csv").write_text(
                "".join(rows), encoding="utf-8")
            ranking = load_ranking(job_dir)
            self.assertEqual(len(ranking), 2)
            self.assertEqual(ranking[0]["case"], "caseB")
            # status ranking wins
            write_json_atomic(job_dir / "status.json", {
                "ranking": [{"case": "caseC", "integrated": 5.0,
                             "uncertain_vol": 1.0, "max": 1.0}],
            })
            ranking = load_ranking(job_dir)
            self.assertEqual(len(ranking), 1)
            self.assertEqual(ranking[0]["case"], "caseC")

    def test_annotation_state_roundtrip(self):
        from flexict_active_learning_ui import (
            load_annotation_state,
            set_case_state,
        )

        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmp) / "job"
            job_dir.mkdir()
            state = load_annotation_state(job_dir)
            self.assertEqual(state["cases"], {})
            set_case_state(job_dir, "case01", "annotated")
            state = load_annotation_state(job_dir)
            self.assertEqual(
                state["cases"]["case01"]["state"], "annotated")
            set_case_state(job_dir, "case01", "skipped")
            state = load_annotation_state(job_dir)
            self.assertEqual(
                state["cases"]["case01"]["state"], "skipped")


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

    def test_prepared_case_missing_file_fails_before_training(self):
        """Remote label_source="prepared": an incomplete case must fail fast.

        The controller materializes /job/input/<case>/image.nii.gz and
        /job/labels/<case>/label.nii.gz before the container starts; if the
        container-side copy is missing (upload truncated, archive corrupt),
        training must refuse rather than silently skip the case.
        """
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            request = fp.normalize_flexict_request(
                _training_request(
                    "/job/input", str(workspace), ["case_1", "case_2"],
                    label_source="prepared",
                    prepared_cases=[
                        {
                            "case_id": "case_1",
                            "image": str(Path(tmp) / "input" / "case_1" / "image.nii.gz"),
                            "label": str(Path(tmp) / "labels" / "case_1" / "label.nii.gz"),
                            "fingerprint": "abc",
                        },
                        # case_2 image never materialized
                        {
                            "case_id": "case_2",
                            "image": str(Path(tmp) / "input" / "case_2" / "image.nii.gz"),
                            "label": str(Path(tmp) / "labels" / "case_2" / "label.nii.gz"),
                            "fingerprint": "def",
                        },
                    ],
                )
            )
            for top in ("input", "labels"):
                for case in ("case_1",):
                    directory = Path(tmp) / top / case
                    directory.mkdir(parents=True)
                    (directory / ("image.nii.gz" if top == "input" else "label.nii.gz")).write_bytes(b"x")
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
            # Any worker spawn proves the incomplete case slipped through.
            with mock.patch.object(
                fp, "_spawn_flexict_worker"
            ) as spawn, mock.patch(
                "flexict_pipeline._acquire_local_gpu", return_value=None
            ):
                exit_code = fp.run_training(job_dir)
            self.assertEqual(exit_code, 1)
            spawn.assert_not_called()
            from nnunet_common import read_json

            status = read_json(job_dir / "status.json", {}) or {}
            self.assertEqual(status["status"], "failed")
            self.assertIn("Prepared remote case is incomplete", status["error"])

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


class TestRemoteExecution(unittest.TestCase):
    """Remote (SSH/Docker) FlexiCT training and inference contracts.

    Mirrors the nnU-Net remote tests in test_nnunet_integration.py: the
    controller is exercised only through its pure preparation/registration
    functions; SSH, containers, and GPU work stay out of scope.
    """

    def _write_json(self, path, payload):
        from nnunet_common import write_json_atomic

        write_json_atomic(path, payload)
        return payload

    # -- job creation ------------------------------------------------------

    def _create_remote_job(self, tmp, **overrides):
        workspace = Path(tmp) / "workspace"
        values = _training_request("X:/d", str(workspace))
        values.update(execution_backend="remote", remote_profile_id="labgpu")
        values.update(overrides)
        launched = {}

        class FakeProcess:
            pid = 4321

        def fake_popen(command, **kwargs):
            launched["command"] = command
            return FakeProcess()

        with mock.patch("flexict_pipeline.subprocess.Popen",
                        side_effect=fake_popen), \
             mock.patch("flexict_pipeline.process_start_marker",
                        return_value="m"):
            job = fp.create_flexict_job(values)
        return job, launched

    def test_remote_training_job_launches_controller_with_spec(self):
        with tempfile.TemporaryDirectory() as tmp:
            job, launched = self._create_remote_job(tmp)
            command = launched["command"]
            self.assertIn("remote_training_controller.py", command[1])
            self.assertIn("run", command)
            self.assertIn("--spec", command)
            spec = json.loads(
                (Path(job["job_dir"]) / "remote_spec.json").read_text(
                    encoding="utf-8"))
            self.assertEqual(
                spec["schema_version"], "mimics_remote_flexict_spec.v1")
            self.assertEqual(spec["kind"], "flexict")
            self.assertEqual(spec["remote_profile_id"], "labgpu")
            status = json.loads(
                (Path(job["job_dir"]) / "status.json").read_text(
                    encoding="utf-8"))
            self.assertEqual(status["execution_backend"], "remote")
            self.assertEqual(status["remote_profile_id"], "labgpu")
            # The local pipeline must not be the launched command.
            self.assertNotIn("flexict_pipeline.py", command[1])

    def test_remote_infer_job_uses_flexict_infer_kind(self):
        with tempfile.TemporaryDirectory() as tmp:
            job, launched = self._create_remote_job(
                tmp,
                operation="infer",
                model_manifest="W:/m/flexict_model_manifest.json",
                image_path="X:/d/case01/ct.nii.gz",
                output_path="X:/out/pred.nii.gz",
                source_modality="ct",
            )
            spec = json.loads(
                (Path(job["job_dir"]) / "remote_spec.json").read_text(
                    encoding="utf-8"))
            self.assertEqual(spec["kind"], "flexict_infer")
            self.assertIn("remote_training_controller.py",
                          launched["command"][1])

    def test_remote_job_without_profile_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            values = _training_request("X:/d", str(workspace))
            values["execution_backend"] = "remote"
            with mock.patch("flexict_pipeline.subprocess.Popen") as popen:
                with self.assertRaisesRegex(RuntimeError, "server"):
                    fp.create_flexict_job(values)
            popen.assert_not_called()

    # -- worker environment -------------------------------------------------

    def test_worker_environment_resolves_remote_weights_override(self):
        roots = {
            "raw": Path("W:/raw"),
            "preprocessed": Path("W:/pre"),
            "results": Path("W:/res"),
        }
        request = fp.normalize_flexict_request(_training_request("X:/d", "W:/w"))
        request["flexict_pretrained_dir"] = "/models/flexict"
        environment = fp.flexict_worker_environment(request, roots, "2d")
        self.assertEqual(
            environment["FLEXICT2D_CKPT"],
            "/models/flexict/flexict_2d/model.safetensors")
        self.assertEqual(
            environment["FLEXICT3D_CKPT"],
            "/models/flexict/flexict_3d/model.safetensors")

    # -- controller preparation ----------------------------------------------

    def _controller(self):
        import remote_training_controller as controller

        return controller

    def _prepare_spec(self, tmp, request):
        job = Path(tmp) / "job"
        bundle = Path(tmp) / "bundle"
        job.mkdir(parents=True, exist_ok=True)
        bundle.mkdir(parents=True, exist_ok=True)
        request_path = job / "request.json"
        status_path = job / "status.json"
        self._write_json(request_path, request)
        self._write_json(
            status_path,
            {"status": "preparing_remote", "job_id": request.get("job_id", "j")},
        )
        self._write_json(job / "control.json", {"action": "run"})
        return {
            "job_dir": str(job),
            "status_path": str(status_path),
            "request_path": str(request_path),
        }, bundle

    def test_prepare_flexict_returns_remote_contract(self):
        controller = self._controller()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.nii.gz"
            label = root / "label.nii.gz"
            image.write_bytes(b"image")
            label.write_bytes(b"label")
            request = fp.normalize_flexict_request(
                _training_request("X:/d", str(root / "workspace")))
            spec, bundle = self._prepare_spec(tmp, request)
            rows = [
                {
                    "case_id": "case_1",
                    "image": image,
                    "label": label,
                    "fingerprint": "fingerprint",
                }
            ]
            with mock.patch(
                "tools.flexict_pipeline.prepare_source_grid_cases",
                return_value=rows,
            ):
                prepared = controller._prepare_flexict(spec, bundle)
            payload = json.loads(
                (bundle / "remote_request.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["kind"], "flexict")
            remote_request = payload["pipeline_request"]
            self.assertTrue(remote_request["gpu_lock_managed_externally"])
            self.assertEqual(
                remote_request["flexict_pretrained_dir"], "/models/flexict")
            self.assertEqual(remote_request["dataset_root"], "/job/input")
            self.assertEqual(remote_request["label_root"], "/job/labels")
            self.assertEqual(remote_request["workspace"], "/job/output")
            self.assertTrue(
                remote_request["runtime_roots"]["raw"].startswith(
                    "/remote-cache/flexict/"))
            self.assertEqual(
                remote_request["runtime_roots"]["results"],
                "/job/output/runtime/nnUNet_results")
            self.assertEqual(
                prepared["required_model_relative"], "flexict")
            self.assertEqual(prepared["train_count"], 1)
            self.assertTrue(
                prepared["remote_artifact_relative"].startswith(
                    "output/models/"))
            self.assertEqual(
                prepared["remote_status_relative"], "pipeline_job/status.json")
            self.assertEqual(
                prepared["remote_control_relative"], "pipeline_job/control.json")
            self.assertEqual(
                prepared["dataset_case_cache_keys"], {"case_1": "fingerprint"})
            self.assertIn("flexict", prepared["local_dataset_archive_cache"])
            # Staged per-case files use the container-side /job layout.
            self.assertTrue(
                (bundle / "input" / "case_1" / "image.nii.gz").is_file())
            self.assertTrue(
                (bundle / "labels" / "case_1" / "label.nii.gz").is_file())
            # The local materialization scratch folder is cleaned up.
            self.assertFalse((bundle / "prepared_local").exists())

    def test_prepare_flexict_rejects_non_training_request(self):
        controller = self._controller()
        with tempfile.TemporaryDirectory() as tmp:
            request = fp.normalize_flexict_request({
                "operation": "infer",
                "workspace": "W:/w",
                "model_manifest": "W:/m/manifest.json",
                "image_path": "X:/d/case/ct.nii.gz",
                "output_path": "X:/out/pred.nii.gz",
            })
            spec, bundle = self._prepare_spec(tmp, request)
            with self.assertRaisesRegex(RuntimeError, "non-training"):
                controller._prepare_flexict(spec, bundle)

    def _infer_fixture(self, tmp, expected_affine=None):
        import nibabel as nib
        import numpy as np

        root = Path(tmp)
        source = root / "image.nii.gz"
        affine = np.eye(4) if expected_affine is None else expected_affine
        nib.save(
            nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), affine),
            str(source),
        )
        model = root / "model"
        (model / "fold_0").mkdir(parents=True)
        (model / "fold_0" / "checkpoint_best.pth").write_bytes(b"weights")
        (model / "plans.json").write_text("{}", encoding="utf-8")
        (model / "dataset.json").write_text("{}", encoding="utf-8")
        manifest_path = model / "flexict_model_manifest.json"
        self._write_json(
            manifest_path,
            {"model_id": "flexict_m1", "model_dir": str(model),
             "configuration": "2d"},
        )
        request = {
            "operation": "infer",
            "job_id": "infer_remote",
            "workspace": str(root / "workspace"),
            "case_id": "case01",
            "image_path": str(source),
            "model_manifest": str(manifest_path),
            "source_modality": "CT",
            "source_geometry_expected": {
                "source_shape": [4, 4, 4],
                "source_voxel_to_ras_matrix": affine.tolist(),
            },
        }
        return request

    def test_prepare_flexict_infer_returns_remote_contract(self):
        controller = self._controller()
        with tempfile.TemporaryDirectory() as tmp:
            request = self._infer_fixture(tmp)
            spec, bundle = self._prepare_spec(tmp, request)
            prepared = controller._prepare_flexict_infer(spec, bundle)
            payload = json.loads(
                (bundle / "remote_request.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["kind"], "flexict_infer")
            remote_request = payload["pipeline_request"]
            self.assertEqual(
                remote_request["image_path"],
                "/job/input/case01/image.nii.gz")
            self.assertEqual(
                remote_request["model_manifest"],
                "/job/labels/model/model_dir/flexict_model_manifest.json")
            self.assertEqual(
                remote_request["model_dir_override"],
                "/job/labels/model/model_dir")
            self.assertEqual(
                remote_request["output_path"],
                "/job/output/prediction_bundle/prediction.nii.gz")
            self.assertTrue(
                remote_request["runtime_roots"]["raw"].startswith(
                    "/remote-cache/flexict/inference/"))
            self.assertEqual(
                prepared["remote_artifact_relative"], "output/prediction_bundle")
            self.assertIn(
                "model", prepared["dataset_case_cache_keys"])
            self.assertIn(
                "case01", prepared["dataset_case_cache_keys"])
            self.assertIn(
                "remote_inference_archives",
                prepared["local_dataset_archive_cache"])
            # No /models requirement: the trained model ships with the job.
            self.assertEqual(prepared["required_model_relative"], "")

    def test_prepare_flexict_infer_rejects_changed_source_geometry(self):
        import numpy as np

        controller = self._controller()
        with tempfile.TemporaryDirectory() as tmp:
            # Image on disk has the identity grid; the recorded expectation
            # claims a shifted grid, as if the Mimics project relinked a
            # different source.
            request = self._infer_fixture(tmp)
            wrong = np.eye(4)
            wrong[1, 3] = 9.0
            request["source_geometry_expected"] = {
                "source_shape": [4, 4, 4],
                "source_voxel_to_ras_matrix": wrong.tolist(),
            }
            spec, bundle = self._prepare_spec(tmp, request)
            with self.assertRaisesRegex(RuntimeError, "no longer matches"):
                controller._prepare_flexict_infer(spec, bundle)

    # -- controller registration ---------------------------------------------

    def test_register_flexict_registers_pair_with_remote_provenance(self):
        controller = self._controller()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            request = fp.normalize_flexict_request(
                _training_request("X:/d", str(workspace)))
            spec, _bundle = self._prepare_spec(tmp, request)
            artifact = root / "artifact" / "models" / "kidney" / "flexict_x"
            for configuration in ("2d", "3d_fullres"):
                model_dir = artifact / configuration
                model_dir.mkdir(parents=True)
                (model_dir / "checkpoint_best.pth").write_bytes(b"w")
                self._write_json(
                    model_dir / "flexict_model_manifest.json",
                    {
                        "model_id": "flexict_" + configuration,
                        "configuration": configuration,
                        "pair_id": "pair1",
                        "task_id": "kidney",
                    },
                )
            profile = {
                "profile_id": "labgpu",
                "name": "Lab GPU",
                "runtime_image": "mimics-ai-runtime:1.0",
                "gpu_device": "1",
            }
            remote_status = {
                "remote_gpu_device": "1",
                "dataset_fingerprint": "dsfp",
                "remote_runtime_image_id": "sha256:abc",
            }
            manifests = controller._register_flexict(
                spec, artifact, remote_status, profile
            )
            self.assertEqual(len(manifests), 2)
            models = fc.load_models(str(workspace))
            self.assertEqual(len(models), 2)
            by_config = {m["configuration"]: m for m in models}
            self.assertEqual(set(by_config), {"2d", "3d_fullres"})
            for manifest in manifests:
                self.assertEqual(manifest["execution_backend"], "remote")
                self.assertEqual(manifest["remote_profile_id"], "labgpu")
                self.assertEqual(manifest["remote_profile_name"], "Lab GPU")
                self.assertEqual(
                    manifest["remote_runtime_image"],
                    "mimics-ai-runtime:1.0")
                self.assertEqual(manifest["remote_gpu_device"], "1")
                self.assertEqual(
                    manifest["remote_dataset_fingerprint"], "dsfp")
                # model_dir points at the downloaded local artifact.
                self.assertTrue(
                    Path(manifest["model_dir"]).is_dir())
                self.assertEqual(
                    manifest["manifest_path"],
                    str(Path(manifest["model_dir"])
                        / "flexict_model_manifest.json"))

    def test_register_flexict_without_manifest_fails(self):
        controller = self._controller()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = fp.normalize_flexict_request(
                _training_request("X:/d", str(root / "workspace")))
            spec, _bundle = self._prepare_spec(tmp, request)
            empty = root / "artifact"
            empty.mkdir()
            with self.assertRaisesRegex(RuntimeError, "no model manifest"):
                controller._register_flexict(
                    spec, empty, {}, {"profile_id": "p", "name": "n",
                                      "runtime_image": "i"})

    # -- remote worker ---------------------------------------------------------

    def test_worker_run_flexict_writes_pipeline_job_and_dispatches(self):
        import remote_worker

        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmp) / "job"
            job_dir.mkdir()
            request = {
                "kind": "flexict",
                "pipeline_request": {
                    "job_id": "train_1",
                    "task_id": "kidney",
                    "task_name": "Kidney",
                    "operation": "train",
                },
            }
            commands = []

            def fake_run(command, cwd):
                commands.append((list(command), cwd))
                return 0

            with mock.patch.object(remote_worker, "_run", side_effect=fake_run), \
                 mock.patch.object(
                     remote_worker, "APP_ROOT", ROOT) as app_root_patch:
                self.assertTrue(
                    (app_root_patch / "tools" / "flexict_pipeline.py").is_file())
                exit_code = remote_worker.run_flexict(job_dir, request)
            self.assertEqual(exit_code, 0)
            pipeline_job = job_dir / "pipeline_job"
            saved = json.loads(
                (pipeline_job / "request.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["job_id"], "train_1")
            control = json.loads(
                (pipeline_job / "control.json").read_text(encoding="utf-8"))
            self.assertEqual(control["action"], "run")
            status = json.loads(
                (pipeline_job / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["schema_version"], "flexict_job.v1")
            self.assertEqual(status["status"], "created")
            self.assertEqual(len(commands), 1)
            command = commands[0][0]
            self.assertIn("flexict_pipeline.py", command[1])
            self.assertIn("run", command)
            self.assertIn("--job-dir", command)

    def test_worker_dispatch_covers_flexict_kinds(self):
        import inspect

        import remote_worker

        source = inspect.getsource(remote_worker.main)
        self.assertIn('kind in {"flexict", "flexict_infer"}', source)
        self.assertIn("run_flexict", source)

    def test_worker_preflight_reports_flexict_weights_and_import(self):
        import contextlib
        import io

        import remote_worker

        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp) / "models"
            for configuration in ("flexict_2d", "flexict_3d"):
                (models_dir / "flexict" / configuration).mkdir(parents=True)
                (models_dir / "flexict" / configuration
                 / "model.safetensors").write_bytes(b"weights")
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured), \
                 mock.patch.object(remote_worker, "APP_ROOT", ROOT):
                remote_worker.preflight(models_dir)
            result = json.loads(captured.getvalue())
            self.assertTrue(result["flexict_weights"])
            self.assertTrue(result["flexict_import"], result.get("flexict_error"))
            self.assertIsInstance(result["ok"], bool)


if __name__ == "__main__":
    unittest.main(verbosity=2)
