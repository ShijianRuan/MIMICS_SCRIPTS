#!/usr/bin/env python3
"""Cross-workflow transition test (Phase 3c of the product-quality program).

The existing suites test each operation in isolation with mocked stage
workers; nothing verified that the HANDOFFS between operations stay
consistent -- that a model registered by a (fake) training run is actually
usable by the real inference controller, that inference output satisfies
the grid contract the AL pipeline assumes, and that an active-learning run
produces overlay requests the Mimics apply path can consume.

This suite runs one continuous FlexiCT workflow on a synthetic kidney pool:

  train (fake worker completion state, real controller + registration)
    -> infer (REAL geometry validation path, fake predict worker)
    -> active learning (REAL uncertainty ranking on real mask pairs)
    -> overlay request file generation (the runtime_py35 apply contract)

Every step asserts the handoff surface between it and the previous step
(manifest fields, registry rows, grid geometry, band files, request JSON
shape), not just each step's own completion.

No GPU, no nnU-Net execution: the heavyweight stages complete via fake
workers, exactly like the lifecycle tests, but chained across operations.

Usage:
    python_env/python.exe tools/test_cross_workflow_transitions.py
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
for value in (str(ROOT), str(ROOT / "tools")):
    if value not in sys.path:
        sys.path.insert(0, value)

import flexict_pipeline as fp  # noqa: E402
from nnunet_common import read_json, write_json_atomic  # noqa: E402

RUNTIME_DIR = ROOT / "runtime_py35"


def _make_case(root: Path, case_id: str, shape=(6, 8, 8), spacing=(1.0, 1.0, 1.0)):
    """Tiny synthetic CT + kidney_left label pair (same fixture as the
    flexict integration suite)."""
    import nibabel as nib
    import numpy as np

    case_dir = root / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    affine = np.diag(list(spacing) + [1.0])
    image = np.zeros(shape, dtype=np.float32)
    image[1:-1, 1:-1, 1:-1] = 100.0
    nib.save(nib.Nifti1Image(image, affine), str(case_dir / "ct.nii.gz"))
    label = np.zeros(shape, dtype=np.uint8)
    label[2:-2, 2:-2, 2:-2] = 1
    nib.save(nib.Nifti1Image(label, affine), str(case_dir / "kidney_left.nii.gz"))
    return case_dir


def _write_job(job_dir: Path, request: dict, kind: str) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(job_dir / "request.json", request)
    write_json_atomic(job_dir / "control.json", {"action": "run"})
    write_json_atomic(
        job_dir / "status.json",
        {
            "schema_version": fp.SCHEMA_VERSION,
            "job_id": request["job_id"],
            "kind": kind,
            "status": "launching",
            "phase": "launching",
            "created_at_epoch": time.time(),
            "updated_at_epoch": time.time(),
        },
    )


class CrossWorkflowTransitionTests(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.dataset_root = self.tmp / "dataset"
        self.case_ids = []
        for index in range(4):
            case_id = "case{:02d}".format(index)
            _make_case(self.dataset_root, case_id)
            self.case_ids.append(case_id)
        self.workspace = self.tmp / "workspace"
        self.jobs = self.tmp / "jobs"
        # Isolate real lock acquisitions (dataset lock, GPU lock) to the
        # temp dir: per-run temp workspace paths hash to new lock names
        # and would leak one-byte guard anchors into the production
        # .mimics_runtime/locks directory forever.
        self._old_lock_dir = os.environ.get("MIMICS_RESOURCE_LOCK_DIR")
        os.environ["MIMICS_RESOURCE_LOCK_DIR"] = str(self.tmp / "locks")

    def tearDown(self):
        if self._old_lock_dir is None:
            os.environ.pop("MIMICS_RESOURCE_LOCK_DIR", None)
        else:
            os.environ["MIMICS_RESOURCE_LOCK_DIR"] = self._old_lock_dir
        self._tmp.cleanup()

    # -- step 1: train (fake worker) ---------------------------------

    def _fake_train_worker(self):
        """Fake preprocess+train workers writing a REAL-ish trainer output
        folder (checkpoints, plans, dataset json -- everything registration
        and later inference need on disk)."""

        def fake_worker(stage, params, request, roots, configuration, *a, **kw):
            if stage == "train":
                model_dir = fp._flexict_model_dir(roots, request, configuration)
                (model_dir / "fold_0").mkdir(parents=True, exist_ok=True)
                (model_dir / "fold_0" / "checkpoint_best.pth").write_bytes(b"x")
                (model_dir / "fold_0" / "checkpoint_final.pth").write_bytes(b"x")
                (model_dir / "plans.json").write_text("{}", encoding="utf-8")
                (model_dir / "dataset.json").write_text("{}", encoding="utf-8")
            else:
                preprocessed = roots["preprocessed"] / fp._dataset_name(request)
                (preprocessed / params["configuration"]).mkdir(
                    parents=True, exist_ok=True)
            return {"status": "ok", "result": {"trained": stage == "train"}}

        return fake_worker

    def _run_training(self, configuration="pair") -> dict:
        request = fp.normalize_flexict_request({
            "operation": "train",
            "task_name": "Kidney",
            "label_name": "kidney_left",
            "cases": self.case_ids,
            "dataset_root": str(self.dataset_root),
            "label_source": "dataset_masks",
            "workspace": str(self.workspace),
            "dataset_id": 750,
            "configuration": configuration,
            "epochs": 150,
            "val_cases": 1,
            "mimics_exe": "",
        })
        job_dir = self.jobs / request["job_id"]
        _write_job(job_dir, request, "train")
        with mock.patch.object(fp, "_spawn_flexict_worker",
                               side_effect=self._fake_train_worker()), \
             mock.patch("nnunet_pipeline._acquire_local_gpu", return_value=None), \
             mock.patch("flexict_pipeline._acquire_local_gpu", return_value=None):
            exit_code = fp.run_training(job_dir)
        status = read_json(job_dir / "status.json", {}) or {}
        self.assertEqual(0, exit_code, status.get("error"))
        self.assertEqual("completed", status["status"])
        return status

    # -- step 2: infer (real geometry validation, fake predict) ------

    def _run_inference(self, model_dir: Path, case_id: str) -> tuple[Path, dict]:
        output = self.tmp / "infer" / "{}_pred.nii.gz".format(case_id)
        request = fp.normalize_flexict_request({
            "operation": "infer",
            "workspace": str(self.workspace),
            "model_manifest": str(model_dir / "flexict_model_manifest.json"),
            "image_path": str(self.dataset_root / case_id / "ct.nii.gz"),
            "output_path": str(output),
            "source_modality": "ct",
        })
        job_dir = self.jobs / request["job_id"]
        _write_job(job_dir, request, "infer")

        import nibabel as nib
        import numpy as np

        def fake_predict(stage, params, *a, **kw):
            self.assertEqual("infer", stage)
            source = nib.load(params["input_path"])
            prediction = np.zeros(source.shape[:3], dtype=np.uint8)
            prediction[1:-1, 1:-1, 1:-1] = 1
            nib.save(nib.Nifti1Image(prediction, source.affine),
                     params["output_path"])
            return {"status": "ok", "result": {}}

        with mock.patch.object(fp, "_spawn_worker", side_effect=fake_predict), \
             mock.patch("flexict_pipeline._acquire_local_gpu", return_value=None):
            exit_code = fp.run_inference(job_dir)
        status = read_json(job_dir / "status.json", {}) or {}
        self.assertEqual(0, exit_code, status.get("error"))
        self.assertEqual("completed", status["status"])
        return output, status

    # -- step 3: active learning (real ranking) ----------------------

    def _fake_al_predict_worker(self, capture, manifests: dict):
        import nibabel as nib
        import numpy as np

        # Registered model dirs are timestamp-named and carry no 2d/3d hint;
        # resolve the configuration by matching the manifest's model_dir.
        folder_to_config = {
            str(Path(
                (read_json(Path(m), {}) or {}).get("model_dir") or "")): config
            for config, m in manifests.items()
        }

        def fake_worker(stage, params, *args, **kwargs):
            self.assertEqual("infer", stage)
            capture.append(dict(params))
            source_dir = Path(params["input_path"])
            output = Path(params["output_path"])
            output.mkdir(parents=True, exist_ok=True)
            configuration = folder_to_config.get(
                str(Path(params["model_folder"])), "3d_fullres")
            for image_path in sorted(source_dir.glob("*.nii.gz")):
                source = nib.load(str(image_path))
                prediction = np.zeros(source.shape[:3], dtype=np.uint8)
                if configuration == "2d":
                    prediction[2:-2, 2:-2, 2:-2] = 1
                else:
                    prediction[1:-1, 1:-1, 1:-1] = 1  # deliberate disagreement
                nib.save(nib.Nifti1Image(prediction, source.affine),
                         str(output / image_path.name))
            return {"status": "ok", "result": {}}

        return fake_worker

    def _run_active_learning(self, manifests: dict) -> tuple[Path, dict]:
        request = fp.normalize_flexict_request({
            "operation": "active_learning",
            "workspace": str(self.workspace),
            "dataset_root": str(self.dataset_root),
            "label_name": "kidney_left",
            "model_manifest_2d": str(manifests["2d"]),
            "model_manifest_3d": str(manifests["3d_fullres"]),
        })
        job_dir = self.jobs / request["job_id"]
        _write_job(job_dir, request, "active_learning")
        capture = []
        with mock.patch.object(
                fp, "_spawn_worker",
                side_effect=self._fake_al_predict_worker(capture, manifests)), \
             mock.patch("flexict_pipeline._acquire_local_gpu",
                        return_value=None):
            exit_code = fp.run_active_learning(job_dir)
        status = read_json(job_dir / "status.json", {}) or {}
        self.assertEqual(0, exit_code, status.get("error"))
        self.assertEqual("completed", status["status"])
        return job_dir, status

    # -- the full chain -----------------------------------------------

    def test_train_infer_al_overlay_handoffs(self):
        # === 1. train (pair) -> registered models with shared pair_id ===
        train_status = self._run_training(configuration="pair")
        models = train_status["models"]
        self.assertEqual(2, len(models))
        by_config = {m["configuration"]: m for m in models}
        pair_ids = {m["pair_id"] for m in models}
        self.assertEqual(1, len(pair_ids), "pair models must share pair_id")
        # handoff surface: manifest on disk, registry row, usable model
        import flexict_common as fc

        for configuration, model in by_config.items():
            manifest_path = Path(model["model_dir"]) / "flexict_model_manifest.json"
            self.assertTrue(
                manifest_path.is_file(),
                "registered {} model has no manifest".format(configuration))
            manifest = read_json(manifest_path, {}) or {}
            self.assertEqual(configuration, manifest["configuration"])
            self.assertEqual("kidney_left", manifest["label_name"])
            usable, why = fc.model_usability(manifest)
            self.assertTrue(usable, why)
        registry_rows = fc.load_models(str(self.workspace))
        self.assertEqual(2, len(registry_rows))
        # pair resolution finds both ends (the AL contract)
        model_2d, model_3d = fc.load_pair(str(self.workspace))
        self.assertIsNotNone(model_2d)
        self.assertIsNotNone(model_3d)
        self.assertEqual(
            by_config["2d"]["model_id"], model_2d["model_id"])
        self.assertEqual(
            by_config["3d_fullres"]["model_id"], model_3d["model_id"])

        # === 2. infer with the 2D model on a real source image =========
        # (materialize + geometry validation run for real; only the
        # network forward is faked)
        pred_path, infer_status = self._run_inference(
            Path(by_config["2d"]["model_dir"]), self.case_ids[0])
        self.assertTrue(pred_path.is_file())
        import nibabel as nib
        import numpy as np

        source = nib.load(
            str(self.dataset_root / self.case_ids[0] / "ct.nii.gz"))
        prediction = nib.load(str(pred_path))
        # handoff surface: prediction grid == source grid (the contract
        # the Mimics apply path and AL pool materialization both assume)
        self.assertEqual(tuple(source.shape[:3]), tuple(prediction.shape[:3]))
        self.assertTrue(
            np.allclose(source.affine, prediction.affine, atol=1e-4))
        self.assertEqual("kidney_left", infer_status["label_name"])
        self.assertGreater(int(np.asarray(prediction.dataobj).sum()), 0)

        # === 3. active learning with the registered pair ===============
        manifests = {
            config: Path(model["model_dir"]) / "flexict_model_manifest.json"
            for config, model in by_config.items()
        }
        al_job_dir, al_status = self._run_active_learning(manifests)
        # handoff surface: every case ranked, sorted, artifacts on disk
        ranking = al_status["ranking"]
        self.assertEqual(len(self.case_ids), len(ranking))
        scores = [row["integrated"] for row in ranking]
        self.assertEqual(scores, sorted(scores, reverse=True))
        uncertainty_dir = al_job_dir / "uncertainty"
        self.assertTrue((uncertainty_dir / "ranking.csv").is_file())
        bands = al_status["uncertainty_bands"] or {}
        self.assertEqual(len(self.case_ids), bands["written"]["moderate"])
        # per-case source geometry recorded (the overlay contract input)
        geometries = read_json(al_job_dir / "input_geometries.json", {}) or {}
        for case_id in self.case_ids:
            self.assertIn(case_id, geometries.get("cases") or {})

        # === 4. overlay request file the runtime apply path consumes ===
        # (the flexict_mimics._al_pending_requests contract: state=pending,
        # case_id, what, under <job>/apply_requests/*.json)
        top_case = al_status["top_case"]
        request_dir = al_job_dir / "apply_requests"
        request_dir.mkdir(parents=True, exist_ok=True)
        overlay_request = {
            "case_id": top_case,
            "what": "bands",
            "state": "pending",
            "requested_at_epoch": time.time(),
        }
        request_path = request_dir / "req_test.json"
        write_json_atomic(request_path, overlay_request)
        # the mask paths the runtime maps the request to must exist
        sys.path.insert(0, str(RUNTIME_DIR))
        try:
            import types

            if "mimics" not in sys.modules:
                # flexict_mimics imports the Mimics API at module load; the
                # apply contract under test (mask path resolution) does not
                # need it, so install a stub when running outside Mimics.
                sys.modules["mimics"] = types.ModuleType("mimics")
            import flexict_mimics
        finally:
            sys.path.remove(str(RUNTIME_DIR))
        mask_rows = flexict_mimics._al_mask_paths(
            str(al_job_dir), top_case, "bands")
        self.assertTrue(
            mask_rows, "no overlay band masks for case {}".format(top_case))
        for title, path in mask_rows:
            self.assertTrue(Path(path).is_file(), title)
        consensus_rows = flexict_mimics._al_mask_paths(
            str(al_job_dir), top_case, "consensus")
        self.assertEqual(1, len(consensus_rows))
        self.assertTrue(Path(consensus_rows[0][1]).is_file())
        # band masks are on the source grid (geometry the request carries)
        band_mask = nib.load(mask_rows[0][1])
        band_geometry = (geometries["cases"][top_case] or {})
        self.assertEqual(
            tuple(band_geometry.get("source_shape") or ()),
            tuple(band_mask.shape[:3]),
            "band mask must match the recorded source geometry",
        )

    def test_retrained_model_supersedes_in_al(self):
        """The workflow-switch seam: after retraining, AL must resolve the
        NEWEST pair, and the old pair must stop being returned by
        load_pair (registry freshness), so an annotator cycling
        train -> AL -> review -> retrain never ranks with stale models."""
        first = self._run_training(configuration="pair")
        # second training run on the same workspace registers a new pair
        second = self._run_training(configuration="pair")
        import flexict_common as fc

        model_2d, model_3d = fc.load_pair(str(self.workspace))
        new_ids = {m["model_id"] for m in second["models"]}
        old_ids = {m["model_id"] for m in first["models"]}
        self.assertIn(model_2d["model_id"], new_ids)
        self.assertIn(model_3d["model_id"], new_ids)
        self.assertFalse(new_ids & old_ids, "new run must produce new models")
        # registry grew, not replaced (history preserved for provenance)
        self.assertEqual(4, len(fc.load_models(str(self.workspace))))


if __name__ == "__main__":
    unittest.main(verbosity=2)
