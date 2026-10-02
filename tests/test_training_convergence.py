#!/usr/bin/env python3
"""Automated convergence smoke tests (Phase 3a of the product-quality program).

Every other test in the project mocks or fabricates the heavyweight training
stages, so "does training actually converge" had zero automated coverage.
This suite closes that gap with end-to-end runs of the REAL pipeline
controllers, REAL nnU-Net preprocessing, REAL trainer execution, and REAL
inference -- only the scale is shrunk (tiny synthetic kidney-shaped volumes,
2 epochs, one configuration).

What is verified per run:

1. the job completes (exit code 0, status "completed")
2. the training loss decreases (last-epoch train loss < first-epoch)
3. the registered model's checkpoints load with torch.load (not just bytes)
4. inference on a held-out case produces a non-empty mask with the expected
   geometry (the real geometry-check path, not a fabricated file)

Two families are covered:

* nnU-Net (MimicsNNUNetTrainer, 3d_fullres) -- the generic managed trainer
* FlexiCT (flexict2d_Trainer) -- the pretrained-backbone few-shot trainer
  (needs the real flexict_2d backbone weights shipped beside the repo)

Both runs use the GPU when available and fall back to CPU otherwise (slow but
functional); the runs are small enough that CPU completes in a few minutes.

Not part of the default gate: add to the ``full`` regression profile only
(see run_regression_matrix.py SUITES). Runtime is dominated by nnU-Net's
planning and dataloader warmup, roughly 3-8 minutes per family on GPU.

Usage:
    python_env/python.exe tests/test_training_convergence.py            # both
    python_env/python.exe tests/test_training_convergence.py Nnunet     # one
    python_env/python.exe tests/test_training_convergence.py Flexict
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for value in (str(ROOT), str(ROOT / "tools")):
    if value not in sys.path:
        sys.path.insert(0, value)

import flexict_pipeline as fp  # noqa: E402
import nnunet_pipeline as np_mod  # noqa: E402
from nnunet_common import read_json, write_json_atomic  # noqa: E402

# Epochs must stay small: the suite proves the *machinery* converges (loss
# goes down, checkpoints are real, inference works), not that a tiny
# synthetic kidney reaches production Dice.
EPOCHS = 2

# Train/val/infer case counts per family. More cases only cost wall-clock.
TRAIN_CASES = 4
VAL_CASES = 1
INFER_CASES = 1
TOTAL_CASES = TRAIN_CASES + VAL_CASES + INFER_CASES


def _gpu_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _make_kidney_case(case_dir: Path, case_index: int) -> None:
    """A tiny synthetic CT with an off-center ellipsoid "kidney" target.

    Geometry is chosen so nnU-Net's 2D planner produces reasonable patch
    sizes and every case differs slightly (per-case offset/size jitter) so
    the model has something to learn beyond a constant output.
    """
    import nibabel as nib
    import numpy as np

    case_dir.mkdir(parents=True, exist_ok=True)
    shape = (12, 40, 40)  # (z, y, x) -- small but not degenerate
    affine = np.diag([1.5, 1.0, 1.0, 1.0])
    image = np.full(shape, -400.0, dtype=np.float32)
    # soft-tissue blob as anatomy context
    yy, xx = np.ogrid[: shape[1], : shape[2]]
    body = ((yy - 20) ** 2 + (xx - 22) ** 2) < 15 ** 2
    image[:, body] = 0.0
    # the target: ellipsoid with per-case jitter
    label = np.zeros(shape, dtype=np.uint8)
    center_z = 6 + (case_index % 2)
    center_y = 18 + (case_index % 3)
    center_x = 20 + (case_index % 4)
    rz, ry, rx = 3, 4 + (case_index % 2), 4
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    target = (
        ((zz - center_z) / rz) ** 2
        + ((yy - center_y) / ry) ** 2
        + ((xx - center_x) / rx) ** 2
    ) < 1.0
    label[target] = 1
    image[target] = 80.0
    nib.save(nib.Nifti1Image(image, affine), str(case_dir / "ct.nii.gz"))
    nib.save(nib.Nifti1Image(label, affine), str(case_dir / "kidney_left.nii.gz"))


def _train_loss_values(log_path: Path) -> list[float]:
    """Extract per-epoch train_loss values from a trainer job.log.

    nnU-Net 2.8.0 prints "train_loss <value>" once per epoch (see
    nnUNetTrainer.on_epoch_end), prefixed by a timestamp because
    print_to_log_file stamps every line ("2026-09-25 04:50:00.123456:
    train_loss 1.2345"). Returns the values in epoch order.
    """
    if not log_path.is_file():
        return []
    text = log_path.read_text(errors="replace")
    return [
        float(value)
        for value in re.findall(
            r"^[0-9: .\-]*train_loss\s+([0-9.eE+-]+)", text, re.MULTILINE
        )
    ]


def _log_tail(log_path: Path, lines: int = 40) -> str:
    """Last ``lines`` of a job.log, for embedding in assertion messages."""
    try:
        text = log_path.read_text(errors="replace")
    except Exception:
        return "<job.log unavailable: {}>".format(log_path)
    return "\n".join(text.splitlines()[-lines:])


class ConvergenceSmokeBase(unittest.TestCase):
    """Shared helpers for the two family smoke tests."""

    maxDiff = None

    def _dataset_root(self, tmp: str) -> tuple[Path, list[str]]:
        dataset_root = Path(tmp) / "dataset"
        case_ids = []
        for index in range(TOTAL_CASES):
            case_id = "case{:02d}".format(index)
            _make_kidney_case(dataset_root / case_id, index)
            case_ids.append(case_id)
        return dataset_root, case_ids

    def _launch_job(self, tmp: str, request: dict) -> tuple[Path, Path]:
        job_dir = Path(tmp) / "jobs" / request["job_id"]
        job_dir.mkdir(parents=True)
        write_json_atomic(job_dir / "request.json", request)
        write_json_atomic(job_dir / "control.json", {"action": "run"})
        write_json_atomic(
            job_dir / "status.json",
            {
                "schema_version": np_mod.SCHEMA_VERSION,
                "job_id": request["job_id"],
                "kind": "train",
                "status": "launching",
                "phase": "launching",
                "created_at_epoch": time.time(),
                "updated_at_epoch": time.time(),
            },
        )
        return job_dir, job_dir / "job.log"

    def _wait_for_completion(self, job_dir: Path, timeout: float = 1800.0) -> dict:
        """Poll status.json until the job reaches a terminal state."""
        status_path = job_dir / "status.json"
        deadline = time.time() + timeout
        last = {}
        while time.time() < deadline:
            last = read_json(status_path, {}) or {}
            if str(last.get("status") or "").lower() in {
                "completed",
                "failed",
                "cancelled",
            }:
                return last
            time.sleep(5.0)
        return last

    def _assert_converged(
        self,
        exit_code: int,
        status: dict,
        log_path: Path,
        model_dir: Path,
    ) -> None:
        """The four convergence assertions shared by both families."""
        # 1. job completed. On failure, embed the job.log tail so the root
        # cause survives the TemporaryDirectory cleanup.
        if exit_code != 0:
            self.assertEqual(
                0,
                exit_code,
                "training controller exited non-zero; job.log tail:\n"
                + _log_tail(log_path),
            )
        self.assertEqual(
            "completed",
            str(status.get("status") or "").lower(),
            "training job did not complete: {}\njob.log tail:\n{}".format(
                str(status.get("error") or status.get("message") or "")[:500],
                _log_tail(log_path),
            ),
        )
        # 2. loss decreased
        losses = _train_loss_values(log_path)
        self.assertGreaterEqual(
            len(losses), EPOCHS, "job.log shows no per-epoch train_loss lines"
        )
        first = sum(losses[: max(1, len(losses) // 2)]) / max(1, len(losses) // 2)
        second_half = losses[max(1, len(losses) // 2):]
        last = sum(second_half) / max(1, len(second_half))
        self.assertLess(
            last,
            first,
            "train loss did not decrease (first-half mean {} vs last-half mean {})".format(
                first, last
            ),
        )
        # 3. checkpoints are real torch files (not just bytes on disk)
        import torch

        for name in ("checkpoint_final.pth",):
            checkpoint = model_dir / "fold_0" / name
            self.assertTrue(
                checkpoint.is_file(), "missing {} in {}".format(name, model_dir)
            )
            payload = torch.load(
                str(checkpoint), map_location="cpu", weights_only=False
            )
            self.assertIn("network_weights", payload)
        # 4. non-empty mask on a held-out case happens in the family tests
        # (they own the inference request shape).


class NnunetConvergenceSmoke(ConvergenceSmokeBase):
    """nnU-Net train -> register -> infer on synthetic kidneys (3d_fullres)."""

    def test_nnunet_train_infer_convergence(self):
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            dataset_root, case_ids = self._dataset_root(tmp)
            workspace = Path(tmp) / "workspace"
            request = {
                "operation": "train",
                "task_id": "kidney",
                "task_name": "Kidney",
                "dataset_id": 791,
                "configuration": "3d_fullres",
                "trainer": "MimicsNNUNetTrainer",
                "epochs": EPOCHS,
                "fold": "0",
                "modality": "CT",
                "cases": case_ids[: TRAIN_CASES + VAL_CASES],
                "dataset_root": str(dataset_root),
                "label_source": "dataset_masks",
                "labels": [
                    {
                        "name": "kidney_left",
                        "aliases": ["kidney_left"],
                        "id": 1,
                        "source_mode": "alternatives",
                    }
                ],
                "workspace": str(workspace),
                "use_cpu": not _gpu_available(),
                "mimics_exe": "",
            }
            request = np_mod.normalize_request(request)
            job_dir, log_path = self._launch_job(tmp, request)
            exit_code = np_mod.run_training(job_dir)
            status = read_json(job_dir / "status.json", {}) or {}

            model = status.get("model") or {}
            model_dir = Path(model.get("model_dir") or "")
            if not model_dir.is_dir():
                self.fail(
                    "no registered model after training; status: {}".format(
                        str(status)[:800]
                    )
                )
            self._assert_converged(exit_code, status, log_path, model_dir)

            # 4. real inference on a held-out case
            infer_case = dataset_root / case_ids[-1]
            output = Path(tmp) / "infer" / "pred.nii.gz"
            output.parent.mkdir(parents=True, exist_ok=True)
            self._run_nnunet_inference(
                model_dir, infer_case / "ct.nii.gz", output, status, tmp
            )
            mask = self._load_mask(output)
            self.assertGreater(
                int(mask.sum()), 0, "inference produced an empty mask"
            )
            self.assertEqual(mask.shape, (12, 40, 40))

    def _run_nnunet_inference(
        self, model_dir: Path, image: Path, output: Path, train_status: dict, tmp: str
    ) -> None:
        """Run inference through the real pipeline controller (run_inference)."""
        request = {
            "operation": "infer",
            "workspace": train_status.get("workspace") or str(Path(tmp) / "workspace"),
            "model_manifest": str(model_dir / "mimics_model_manifest.json"),
            "image_path": str(image),
            "output_path": str(output),
            "use_cpu": not _gpu_available(),
        }
        request = np_mod.normalize_request(request)
        job_dir = Path(tmp) / "jobs" / request["job_id"]
        job_dir.mkdir(parents=True)
        write_json_atomic(job_dir / "request.json", request)
        write_json_atomic(job_dir / "control.json", {"action": "run"})
        write_json_atomic(
            job_dir / "status.json",
            {
                "schema_version": np_mod.SCHEMA_VERSION,
                "job_id": request["job_id"],
                "kind": "infer",
                "status": "launching",
                "created_at_epoch": time.time(),
                "updated_at_epoch": time.time(),
            },
        )
        exit_code = np_mod.run_inference(job_dir)
        status = read_json(job_dir / "status.json", {}) or {}
        self.assertEqual(0, exit_code, "inference controller exited non-zero")
        self.assertEqual(
            "completed",
            str(status.get("status") or "").lower(),
            "inference did not complete: {}".format(
                str(status.get("error") or status.get("message") or "")[:500]
            ),
        )
        self.assertTrue(
            output.is_file(), "inference completed without writing {}".format(output)
        )

    def _load_mask(self, path: Path):
        import nibabel as nib
        import numpy as np

        return np.asarray(nib.load(str(path)).dataobj)


class FlexictConvergenceSmoke(ConvergenceSmokeBase):
    """FlexiCT 2D train -> register -> infer on synthetic kidneys.

    Requires the pretrained flexict_2d backbone weights (the real repo
    weights beside integrations/flexict-finetune). Skips with a clear
    message when the deployment has no weights -- the guidance window
    explains the same condition to annotators.
    """

    def test_flexict_2d_train_infer_convergence(self):
        import flexict_common

        config = flexict_common.load_config()
        try:
            weights, _source = flexict_common.resolve_pretrained_dir(config)
        except Exception:
            self.skipTest(
                "FlexiCT pretrained backbone weights are not installed; "
                "the convergence smoke needs them."
            )
        if not (weights / "flexict_2d" / "model.safetensors").is_file():
            self.skipTest(
                "FlexiCT pretrained backbone weights are not installed; "
                "the convergence smoke needs them."
            )

        with tempfile.TemporaryDirectory() as tmp:
            dataset_root, case_ids = self._dataset_root(tmp)
            workspace = Path(tmp) / "workspace"
            request = {
                "operation": "train",
                "task_name": "Kidney",
                "label_name": "kidney_left",
                "cases": case_ids[: TRAIN_CASES + VAL_CASES],
                "dataset_root": str(dataset_root),
                "label_source": "dataset_masks",
                "workspace": str(workspace),
                "dataset_id": 792,
                "configuration": "2d",
                "epochs": EPOCHS,
                "val_cases": VAL_CASES,
                "mimics_exe": "",
            }
            request = fp.normalize_flexict_request(request)
            job_dir, log_path = self._launch_job(tmp, request)
            exit_code = fp.run_training(job_dir)
            status = read_json(job_dir / "status.json", {}) or {}

            models = status.get("models") or []
            self.assertTrue(
                models, "no registered FlexiCT model; status: {}".format(str(status)[:800])
            )
            model_dir = Path(models[0].get("model_dir") or "")
            if not model_dir.is_dir():
                self.fail("registered model dir does not exist: {}".format(model_dir))
            self._assert_converged(exit_code, status, log_path, model_dir)

            # 4. real inference on a held-out case through the FlexiCT
            #    pipeline controller
            infer_case = dataset_root / case_ids[-1]
            output = Path(tmp) / "infer" / "pred.nii.gz"
            output.parent.mkdir(parents=True, exist_ok=True)
            infer_request = {
                "operation": "infer",
                "workspace": str(workspace),
                "model_manifest": str(model_dir / "flexict_model_manifest.json"),
                "image_path": str(infer_case / "ct.nii.gz"),
                "output_path": str(output),
                "use_cpu": not _gpu_available(),
                "source_modality": "ct",
            }
            infer_request = fp.normalize_flexict_request(infer_request)
            infer_dir = Path(tmp) / "jobs" / infer_request["job_id"]
            infer_dir.mkdir(parents=True)
            write_json_atomic(infer_dir / "request.json", infer_request)
            write_json_atomic(infer_dir / "control.json", {"action": "run"})
            write_json_atomic(
                infer_dir / "status.json",
                {
                    "schema_version": fp.SCHEMA_VERSION,
                    "job_id": infer_request["job_id"],
                    "kind": "infer",
                    "status": "launching",
                    "created_at_epoch": time.time(),
                    "updated_at_epoch": time.time(),
                },
            )
            exit_code = fp.run_inference(infer_dir)
            infer_status = read_json(infer_dir / "status.json", {}) or {}
            self.assertEqual(
                0,
                exit_code,
                "FlexiCT inference exited non-zero: {}\njob.log tail:\n{}".format(
                    str(infer_status.get("error") or "")[:500],
                    _log_tail(infer_dir / "job.log"),
                ),
            )
            self.assertEqual(
                "completed",
                str(infer_status.get("status") or "").lower(),
                "FlexiCT inference did not complete: {}\njob.log tail:\n{}".format(
                    str(infer_status.get("error") or infer_status.get("message") or "")[:500],
                    _log_tail(infer_dir / "job.log"),
                ),
            )
            self.assertTrue(
                output.is_file(), "FlexiCT inference wrote no output mask"
            )
            import nibabel as nib
            import numpy as np

            mask = np.asarray(nib.load(str(output)).dataobj)
            self.assertGreater(
                int(mask.sum()), 0, "FlexiCT inference produced an empty mask"
            )


def load_tests(loader, tests, pattern):
    """Allow selective family runs via argv: Nnunet | Flexict."""
    suite = unittest.TestSuite()
    if len(sys.argv) > 1 and sys.argv[1].lower().startswith("nnunet"):
        suite.addTests(loader.loadTestsFromTestCase(NnunetConvergenceSmoke))
    elif len(sys.argv) > 1 and sys.argv[1].lower().startswith("flexict"):
        suite.addTests(loader.loadTestsFromTestCase(FlexictConvergenceSmoke))
    else:
        suite.addTests(loader.loadTestsFromTestCase(NnunetConvergenceSmoke))
        suite.addTests(loader.loadTestsFromTestCase(FlexictConvergenceSmoke))
    return suite


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
