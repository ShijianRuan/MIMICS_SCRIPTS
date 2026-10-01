"""Offline tests for scripts/build_dataset.py (F29).

Run with the host python_env:
    python_env/python.exe integrations/flexict-finetune/tests/test_build_dataset.py

Covers the physical SI stratification axis (isotropic, anisotropic,
axis-permuted, flipped affines) and explicit rejection of degenerate
inputs (1 usable case, --n-train < 1) instead of a division by zero.
"""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = os.path.abspath(os.path.dirname(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402

import build_dataset as bd  # noqa: E402

HOST_PYTHON = sys.executable


def _save(path, arr, affine):
    nib.save(nib.Nifti1Image(arr, affine), str(path))


class TestZStartPhysicalAxis(unittest.TestCase):
    def _zstart(self, affine, target):
        img = nib.Nifti1Image(np.zeros((10, 10, 10), np.float32), affine)
        arr = np.zeros((10, 10, 10), bool)
        arr[target] = True
        with tempfile.TemporaryDirectory() as tmp:
            ct = Path(tmp) / "ct.nii.gz"
            lbl = Path(tmp) / "lbl.nii.gz"
            _save(ct, img.get_fdata(), affine)
            _save(lbl, arr.astype(np.uint8), affine)
            return bd._zstart(str(ct), str(lbl))

    def test_isotropic_ras_uses_physical_axis(self):
        # Old argmax(spacing) tie-broke to axis 0; physical SI is axis 2.
        self.assertEqual(
            self._zstart(np.diag([0.8, 0.8, 0.8, 1.0]), (1, 5, 8)), 0.8)

    def test_anisotropic_axial(self):
        self.assertEqual(
            self._zstart(np.diag([0.6, 0.6, 2.5, 1.0]), (1, 5, 8)), 0.8)

    def test_axis_permuted(self):
        # SI along voxel axis 0 with the thinnest spacing.
        affine = np.array([[0.0, 0.0, 0.6, 0.0],
                           [0.6, 0.0, 0.0, 0.0],
                           [0.0, 0.6, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]])
        self.assertEqual(self._zstart(affine, (8, 1, 5)), 0.8)

    def test_flipped_z(self):
        self.assertEqual(
            self._zstart(np.diag([0.8, 0.8, -0.8, 1.0]), (1, 5, 8)), 0.8)

    def test_empty_target_neutral(self):
        affine = np.diag([0.8, 0.8, 0.8, 1.0])
        with tempfile.TemporaryDirectory() as tmp:
            ct = Path(tmp) / "ct.nii.gz"
            lbl = Path(tmp) / "lbl.nii.gz"
            _save(ct, np.zeros((10, 10, 10), np.float32), affine)
            _save(lbl, np.zeros((10, 10, 10), np.uint8), affine)
            self.assertEqual(bd._zstart(str(ct), str(lbl)), 0.5)


class TestDegenerateInputs(unittest.TestCase):
    def _source(self, tmp, case_count):
        src = Path(tmp) / "src"
        for i in range(case_count):
            cdir = src / "case{:02d}".format(i)
            (cdir / "segmentations").mkdir(parents=True)
            _save(cdir / "ct.nii.gz",
                  np.zeros((6, 6, 6), np.float32), np.diag([1.0] * 4))
            _save(cdir / "segmentations" / "liver.nii.gz",
                  np.zeros((6, 6, 6), np.uint8), np.diag([1.0] * 4))
        return src

    def _run(self, src, out, n_train):
        return subprocess.run(
            [HOST_PYTHON, os.path.join(REPO, "scripts", "build_dataset.py"),
             "--source", str(src), "--label-name", "liver",
             "--dataset-id", "907", "--dataset-name", "LiverFS",
             "--n-train", str(n_train), "--out", str(out)],
            capture_output=True, text=True)

    def test_single_case_rejected(self):
        # 1 usable case used to reach n_train=0 → ZeroDivisionError; it must
        # now fail with a clear message and write nothing.
        with tempfile.TemporaryDirectory() as tmp:
            src = self._source(tmp, 1)
            out = Path(tmp) / "out"
            result = self._run(src, out, 4)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("at least 2", result.stderr)
            self.assertFalse(out.exists())

    def test_n_train_below_one_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = self._source(tmp, 4)
            out = Path(tmp) / "out"
            result = self._run(src, out, 0)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("n-train", result.stderr)
            self.assertFalse(out.exists())

    def test_two_case_minimum_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = self._source(tmp, 2)
            out = Path(tmp) / "out"
            result = self._run(src, out, 8)
            self.assertEqual(result.returncode, 0, result.stderr)
            dsjson = out / "Dataset907_LiverFS" / "dataset.json"
            self.assertTrue(dsjson.exists())
            import json
            manifest = json.loads(dsjson.read_text(encoding="utf-8"))["split"]
            self.assertEqual(len(manifest["train"]), 1)
            self.assertEqual(len(manifest["test"]), 1)


if __name__ == "__main__":
    unittest.main()
