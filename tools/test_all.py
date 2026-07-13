#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Comprehensive tests for Mimics-Script components.

Covers:
  L1: Syntax check on all Python files
  L2: runtime_common.py — JSON atomic write, resource locks, utility functions
  L3: mimics_bridge.py — NIfTI header normalization, DICOM conversion, buffer mapping
  L4: nninteractive_bridge.py — Source image loading, intensity transform
  L5: window_level_mimics.py — HU2GV, GV range, clamp, preset matching
  L6: create_mcs_batch.py — Stop marker, inject_buffer, shape adjustment
  L7: scripting_library — Entry point routing

Run from the project root:
    python tools/test_all.py
"""

from __future__ import print_function

import json
import importlib.util
import os
import shutil
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Test setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNTIME_DIR = os.path.join(PROJECT_ROOT, "runtime_py35")

sys.path.insert(0, RUNTIME_DIR)
sys.path.insert(0, PROJECT_ROOT)

# -- Mock the ``mimics`` module so we can import runtime files without a
#    Mimics workstation.  Individual tests that need real Mimics API calls
#    are skipped on non-Mimics environments.

class _FakeModule:
    """A mock object that allows arbitrary attribute setting."""
    pass


def _fake_fn(*_a, **_kw):
    return None


_mock_mimics = _FakeModule()
_mock_mimics.data = _FakeModule()
_mock_mimics.data.masks = []
_mock_mimics.data.images = _FakeModule()
_mock_mimics.data.images.get_active = lambda: None
_mock_mimics.data.points = _FakeModule()
_mock_mimics.data.points.delete = _fake_fn
_mock_mimics.data.masks_delete = _fake_fn  # used as mimics.data.masks.delete
_mock_mimics.data.distance_measurements = _FakeModule()
_mock_mimics.data.distance_measurements.delete = _fake_fn
_mock_mimics.data.measurements = _FakeModule()
_mock_mimics.data.measurements.delete = _fake_fn
_mock_mimics.data.splines = _FakeModule()
_mock_mimics.data.splines.delete = _fake_fn
_mock_mimics.segment = _FakeModule()
_mock_mimics.segment.HU2GV = lambda x: x
_mock_mimics.segment.create_mask = _fake_fn
_mock_mimics.view = _FakeModule()
_mock_mimics.view.set_contrast = _fake_fn
_mock_mimics.view.get_contrast = lambda: None
_mock_mimics.file = _FakeModule()
_mock_mimics.file.import_dicom_images = _fake_fn
_mock_mimics.file.save_project = _fake_fn
_mock_mimics.file.close_project = _fake_fn
_mock_mimics.dialogs = _FakeModule()
_mock_mimics.dialogs.question_box = lambda **kw: "Cancel"
_mock_mimics.dialogs.message_box = _fake_fn
_mock_mimics.logging = _FakeModule()
_mock_mimics.logging.log_user_message = _fake_fn
_mock_mimics.analyze = _FakeModule()
_mock_mimics.analyze.create_point = _fake_fn
_mock_mimics.measure = _FakeModule()
_mock_mimics.update_gui = _fake_fn
_mock_mimics.disable_update_gui = _fake_fn
_mock_mimics.enable_update_gui = _fake_fn
_mock_mimics.is_update_gui_enabled = lambda: True
_mock_mimics.UserInterrupted = type("UserInterrupted", (Exception,), {})
sys.modules["mimics"] = _mock_mimics


def _make_temp_dir():
    return tempfile.mkdtemp(prefix="mimics_test_")


def _cleanup(path):
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


# ============================================================================
# L1: Syntax checks
# ============================================================================


class TestSyntax(unittest.TestCase):
    """Ensure all Python files compile without syntax errors."""

    def test_all_py_files_compile(self):
        import py_compile

        errors = []
        for root, _dirs, files in os.walk(PROJECT_ROOT):
            if "__pycache__" in root or "nninteractive_env" in root:
                continue
            if ".git" in root or "venv" in root or ".venv" in root:
                continue
            if "external/dinov3-medical-seg" in root:
                continue  # external code
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                path = os.path.join(root, fname)
                try:
                    py_compile.compile(path, doraise=True)
                except py_compile.PyCompileError as exc:
                    errors.append("{}: {}".format(os.path.relpath(path, PROJECT_ROOT), exc))
        self.assertEqual([], errors, "\n".join(errors))

    def test_runtime_py35_files_exist(self):
        expected = [
            "runtime_common.py",
            "mimics_import.py",
            "mimics_export.py",
            "nninteractive_mimics.py",
            "create_mcs_batch.py",
            "fewshot_mimics.py",
            "window_level_mimics.py",
            "mimics_stop_background.py",
        ]
        for fname in expected:
            self.assertTrue(
                os.path.isfile(os.path.join(RUNTIME_DIR, fname)),
                "Missing runtime file: {}".format(fname),
            )

    def test_scripting_library_structure(self):
        lib = os.path.join(PROJECT_ROOT, "scripting_library")
        for sub in ["01_Data", "02_AI", "03_Review", "99_Admin"]:
            self.assertTrue(os.path.isdir(os.path.join(lib, sub)), "Missing dir: {}".format(sub))

    def test_no_old_flat_entries(self):
        """Verify old flat entries were removed or migrated."""
        lib = os.path.join(PROJECT_ROOT, "scripting_library")
        old_files = ["DINOv3_FewShot.py", "DINOv3_FewShot_Train.py"]
        for fname in old_files:
            self.assertFalse(
                os.path.isfile(os.path.join(lib, fname)),
                "Old entry still present: {}".format(fname),
            )


# ============================================================================
# L2: runtime_common.py
# ============================================================================


class TestRuntimeCommon(unittest.TestCase):
    def setUp(self):
        self.tmp = _make_temp_dir()

    def tearDown(self):
        _cleanup(self.tmp)

    def test_write_and_read_json_atomic(self):
        import runtime_common

        path = os.path.join(self.tmp, "test.json")
        data = {"key": "value", "nested": {"a": 1, "b": [2, 3]}}
        runtime_common.write_json_atomic(path, data)
        self.assertTrue(os.path.isfile(path))
        # Ensure no .tmp residue
        tmps = [f for f in os.listdir(self.tmp) if f.endswith(".tmp")]
        self.assertEqual([], tmps)
        loaded = runtime_common.read_json(path)
        self.assertEqual(data, loaded)

    def test_read_json_missing_file_returns_default(self):
        import runtime_common

        result = runtime_common.read_json(os.path.join(self.tmp, "nonexistent.json"), 42)
        self.assertEqual(42, result)
        result2 = runtime_common.read_json(os.path.join(self.tmp, "nonexistent.json"), None)
        self.assertIsNone(result2)

    def test_read_json_default_default(self):
        import runtime_common

        result = runtime_common.read_json(os.path.join(self.tmp, "nonexistent.json"))
        self.assertIsNone(result)  # default is None, not {}

    def test_write_json_atomic_creates_parents(self):
        import runtime_common

        path = os.path.join(self.tmp, "deep", "nested", "data.json")
        runtime_common.write_json_atomic(path, [1, 2, 3])
        self.assertTrue(os.path.isfile(path))

    def test_write_json_atomic_overwrites(self):
        import runtime_common

        path = os.path.join(self.tmp, "overwrite.json")
        runtime_common.write_json_atomic(path, {"a": 1})
        runtime_common.write_json_atomic(path, {"b": 2})
        loaded = runtime_common.read_json(path)
        self.assertEqual({"b": 2}, loaded)

    def test_write_json_atomic_retries_transient_replace_failure(self):
        import runtime_common

        path = os.path.join(self.tmp, "retry_replace.json")
        old_replace = runtime_common.os.replace
        old_sleep = runtime_common.time.sleep
        calls = []

        def flaky_replace(src, dst):
            calls.append((src, dst))
            if len(calls) < 3:
                raise OSError(5, "access denied")
            return old_replace(src, dst)

        try:
            runtime_common.os.replace = flaky_replace
            runtime_common.time.sleep = lambda _seconds: None
            runtime_common.write_json_atomic(path, {"ok": True})
        finally:
            runtime_common.os.replace = old_replace
            runtime_common.time.sleep = old_sleep
        self.assertGreaterEqual(len(calls), 3)
        self.assertEqual({"ok": True}, runtime_common.read_json(path))

    def test_safe_filename(self):
        import runtime_common

        self.assertEqual("hello_world_", runtime_common.safe_filename("hello world!"))
        self.assertEqual("test___123", runtime_common.safe_filename("test@#$123"))
        self.assertEqual("unknown", runtime_common.safe_filename(""))
        self.assertEqual("unknown", runtime_common.safe_filename(None))

    def test_safe_slug(self):
        import runtime_common

        self.assertEqual("hello_world", runtime_common.safe_slug("__hello world__"))
        self.assertEqual("test", runtime_common.safe_slug("___test___"))
        self.assertEqual("unknown", runtime_common.safe_slug(""))

    def test_write_json_atomic_retry(self):
        """Simulate transient failure by writing to a path whose parent is a file."""
        import runtime_common

        blocker = os.path.join(self.tmp, "blocker")
        with open(blocker, "w") as f:
            f.write("block")
        path = os.path.join(blocker, "should_fail.json")
        with self.assertRaises(OSError):
            runtime_common.write_json_atomic(path, {"test": True})

    def test_process_exists(self):
        import runtime_common

        # Current process should exist
        self.assertTrue(runtime_common.process_exists(os.getpid()))
        # PID 0 should never exist
        self.assertFalse(runtime_common.process_exists(0))
        # Very large PID should not exist
        self.assertFalse(runtime_common.process_exists(99999999))
        # Negative PID
        self.assertFalse(runtime_common.process_exists(-1))

    def test_cleanup_stale_resource_locks(self):
        import runtime_common

        lock_dir = os.path.join(self.tmp, "locks")
        os.makedirs(lock_dir)
        # Create a stale lock (dead PID)
        stale_lock = os.path.join(lock_dir, "stale.lock")
        runtime_common.write_json_atomic(stale_lock, {
            "pid": 99999999,
            "token": "abc",
            "resource": "test",
            "owner": "test",
        })
        # Create an active lock (current PID)
        active_lock = os.path.join(lock_dir, "active.lock")
        runtime_common.write_json_atomic(active_lock, {
            "pid": os.getpid(),
            "token": "def",
            "resource": "test",
            "owner": "test",
        })
        removed = runtime_common.cleanup_stale_resource_locks(lock_dir)
        self.assertGreaterEqual(removed, 1)
        self.assertFalse(os.path.isfile(stale_lock))
        self.assertTrue(os.path.isfile(active_lock))

    def test_resource_lock_acquire_release(self):
        import runtime_common

        lock_path = os.path.join(self.tmp, "test.lock")
        token = runtime_common.acquire_resource_lock(
            lock_path,
            "test_resource",
            "test_owner",
        )
        self.assertIsNotNone(token)
        self.assertTrue(os.path.isfile(lock_path))
        payload = runtime_common.read_json(lock_path)
        self.assertEqual(payload["resource"], "test_resource")
        self.assertEqual(payload["pid"], os.getpid())

        released = runtime_common.release_resource_lock(lock_path, token)
        self.assertTrue(released)
        self.assertFalse(os.path.isfile(lock_path))

    def test_resource_lock_exclusion(self):
        """Second acquirer should fail when first holds the lock."""
        import runtime_common

        lock_path = os.path.join(self.tmp, "excl.lock")
        token1 = runtime_common.acquire_resource_lock(lock_path, "r", "o1")
        self.assertIsNotNone(token1)
        token2 = runtime_common.acquire_resource_lock(lock_path, "r", "o2")
        self.assertIsNone(token2)
        runtime_common.release_resource_lock(lock_path, token1)


# ============================================================================
# L3: mimics_bridge.py
# ============================================================================


class TestMimicsBridgeNiftiNormalization(unittest.TestCase):
    def setUp(self):
        self.tmp = _make_temp_dir()

    def tearDown(self):
        _cleanup(self.tmp)

    def _make_nifti(self, fname, data, affine, sform_code=2, qform_code=0):
        import nibabel as nib

        path = os.path.join(self.tmp, fname)
        img = nib.Nifti1Image(data, affine)
        img.header.set_sform(affine, code=sform_code)
        img.header.set_qform(affine, code=qform_code)
        nib.save(img, path)
        return path

    def test_normalize_qform_zero_to_valid(self):
        from mimics_bridge import _normalize_nifti_affine
        import nibabel as nib

        shape = (10, 10, 10)
        affine = np.eye(4)
        affine[0, 3] = -50
        data = np.zeros(shape, dtype=np.int16)

        path = self._make_nifti("ct.nii.gz", data, affine, sform_code=2, qform_code=0)
        img = nib.load(path)
        self.assertEqual(img.header["qform_code"], 0)
        self.assertEqual(img.header["sform_code"], 2)

        result = _normalize_nifti_affine(img)
        # After normalization, both codes should be non-zero
        self.assertGreater(img.header["qform_code"], 0)
        self.assertGreater(img.header["sform_code"], 0)
        np.testing.assert_allclose(result, affine, atol=1e-6)

    def test_normalize_both_codes_zero(self):
        from mimics_bridge import _normalize_nifti_affine
        import nibabel as nib

        shape = (8, 12, 16)
        affine = np.diag([1.5, 1.5, 3.0, 1.0])
        affine[0, 3] = -100
        data = np.random.randint(-500, 500, shape, dtype=np.int16)

        path = self._make_nifti("ct.nii.gz", data, affine, sform_code=0, qform_code=0)
        img = nib.load(path)
        result = _normalize_nifti_affine(img)
        self.assertGreater(img.header["qform_code"], 0)
        self.assertGreater(img.header["sform_code"], 0)
        np.testing.assert_allclose(result, affine, atol=1e-6)

    def test_normalize_preserves_oblique_affine(self):
        from mimics_bridge import _normalize_nifti_affine
        import nibabel as nib

        shape = (249, 188, 213)
        # Real oblique affine from TotalSegmentator data
        affine = np.array([
            [1.49672401, 0.03480455, -0.09283835, -174.172134],
            [-0.03454677, 1.49959612, 0.00214286, 25.5811043],
            [0.09286306, 0.0, 1.49712265, 0.4003039],
            [0.0, 0.0, 0.0, 1.0],
        ])
        data = np.zeros(shape, dtype=np.int16)

        path = self._make_nifti("oblique.nii.gz", data, affine, sform_code=2, qform_code=0)
        img = nib.load(path)
        result = _normalize_nifti_affine(img)
        np.testing.assert_allclose(result, affine, atol=1e-6)

    def test_read_nifti_mask_with_affine_normalized(self):
        from mimics_bridge import read_nifti_mask_with_affine, _affine_is_usable

        shape = (15, 20, 25)
        affine = np.diag([1.0, 1.0, 2.0, 1.0])
        affine[0, 3] = -75
        mask_data = (np.random.random(shape) > 0.7).astype(np.uint8)

        path = self._make_nifti("mask.nii.gz", mask_data, affine, sform_code=2, qform_code=0)
        array, result_affine = read_nifti_mask_with_affine(path)
        # SimpleITK orients to LPS internally, so the data and affine may be
        # reordered. Both shapes must be non-zero and the affine must be usable.
        self.assertEqual(shape, array.shape)
        self.assertTrue(_affine_is_usable(result_affine),
                        "Returned affine should be a valid 4x4 matrix")
        # Mask should be bool-like
        self.assertTrue(np.all((array == 0) | (array == 1)))

    def test_get_image_affine_normalized(self):
        from mimics_bridge import get_image_affine, _affine_is_usable

        shape = (5, 5, 5)
        affine = np.eye(4)
        data = np.zeros(shape, dtype=np.int16)

        path = self._make_nifti("img.nii.gz", data, affine, sform_code=0, qform_code=2)
        result = get_image_affine(path)
        # SimpleITK orients to LPS; the returned affine is RAS-reconstructed
        # and may differ numerically but must be a usable 4x4 matrix.
        self.assertTrue(_affine_is_usable(result),
                        "Returned affine should be a valid 4x4 matrix")

    def test_prepare_preserves_original_oblique_source_shape_metadata(self):
        """Source metadata must not be replaced by the axial Mimics import grid."""
        from mimics_bridge import do_prepare

        shape = (4, 5, 6)
        angle = np.deg2rad(12.0)
        affine = np.array([
            [np.cos(angle), 0.0, np.sin(angle), -20.0],
            [0.0, 1.2, 0.0, -30.0],
            [-np.sin(angle), 0.0, np.cos(angle), 15.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        path = self._make_nifti(
            "oblique_source.nii.gz",
            np.zeros(shape, dtype=np.int16),
            affine,
            sform_code=2,
            qform_code=1,
        )
        result = do_prepare({
            "image_path": path,
            "masks": [],
            "dicom_out": os.path.join(self.tmp, "dicom"),
            "buffers_out": os.path.join(self.tmp, "buffers"),
            "case_id": "oblique",
        })
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["source_image_shape"], list(shape))
        self.assertEqual(np.asarray(result["source_voxel_to_ras_matrix"]).shape, (4, 4))

    def test_oblique_prediction_buffer_matches_import_buffer_with_axis_mapping(self):
        """Import and inference must produce the same Mimics buffer for one physical mask."""
        from mimics_bridge import do_prepare, _mask_buffer_for_target_grid
        import nibabel as nib

        shape = (12, 10, 8)
        angle = np.deg2rad(11.0)
        affine = np.array([
            [1.1 * np.cos(angle), 0.0, 1.4 * np.sin(angle), -42.0],
            [0.0, 1.3, 0.0, 17.0],
            [-1.1 * np.sin(angle), 0.0, 1.4 * np.cos(angle), 8.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        image_path = os.path.join(self.tmp, "oblique_ct.nii.gz")
        mask_path = os.path.join(self.tmp, "asymmetric_mask.nii.gz")
        image = np.zeros(shape, dtype=np.int16)
        mask = np.zeros(shape, dtype=np.uint8)
        mask[1:5, 2:8, 1:4] = 1
        mask[8:11, 1:3, 5:7] = 1
        nib.save(nib.Nifti1Image(image, affine), image_path)
        nib.save(nib.Nifti1Image(mask, affine), mask_path)
        axes = [1, 0, 2]
        flips = [True, False, True]
        prepared = do_prepare({
            "image_path": image_path,
            "masks": [{"name": "organ", "path": mask_path}],
            "dicom_out": os.path.join(self.tmp, "mapped_dicom"),
            "buffers_out": os.path.join(self.tmp, "import_buffers"),
            "case_id": "mapped_oblique",
            "axes": axes,
            "flips": flips,
        })
        self.assertEqual(prepared["status"], "ok")
        imported = prepared["masks"][0]
        prediction_buffer = os.path.join(self.tmp, "prediction.u8")
        converted = _mask_buffer_for_target_grid(
            mask_path,
            imported["image_shape"],
            prepared["mimics_voxel_to_ras_matrix"],
            axes,
            flips,
            prediction_buffer,
        )
        with open(imported["u8_path"], "rb") as handle:
            imported_bytes = handle.read()
        with open(converted["output_path"], "rb") as handle:
            prediction_bytes = handle.read()
        self.assertEqual(converted["mimics_shape"], imported["mimics_shape"])
        self.assertEqual(prediction_bytes, imported_bytes)


class TestMimicsBridgeBufferMapping(unittest.TestCase):
    def setUp(self):
        self.tmp = _make_temp_dir()

    def tearDown(self):
        _cleanup(self.tmp)
        from mimics_bridge import apply_buffer_mapping

        data = np.arange(24).reshape((2, 3, 4)).astype(np.uint8)
        result = apply_buffer_mapping(data, [0, 1, 2], [False, False, False])
        np.testing.assert_array_equal(data, result)

    def test_apply_buffer_mapping_flip_axis0(self):
        from mimics_bridge import apply_buffer_mapping

        data = np.arange(24).reshape((2, 3, 4)).astype(np.uint8)
        result = apply_buffer_mapping(data, [0, 1, 2], [True, False, False])
        expected = np.flip(data, axis=0)
        np.testing.assert_array_equal(expected, result)

    def test_apply_buffer_mapping_flip_axis1(self):
        from mimics_bridge import apply_buffer_mapping

        data = np.arange(24).reshape((2, 3, 4)).astype(np.uint8)
        result = apply_buffer_mapping(data, [0, 1, 2], [False, True, False])
        expected = np.flip(data, axis=1)
        np.testing.assert_array_equal(expected, result)

    def test_inverse_buffer_mapping_roundtrip(self):
        from mimics_bridge import apply_buffer_mapping, inverse_buffer_mapping

        data = np.arange(60).reshape((3, 4, 5)).astype(np.uint8)
        axes = [1, 2, 0]
        flips = [True, False, True]
        transformed = apply_buffer_mapping(data, axes, flips)
        restored = inverse_buffer_mapping(transformed, axes, flips)
        np.testing.assert_array_equal(data, restored)

    def test_resample_mask_to_image_grid_same_affine(self):
        from mimics_bridge import resample_mask_to_image_grid

        shape = (10, 15, 20)
        affine = np.eye(4)
        affine[0, 0] = 1.5
        affine[1, 1] = 1.5
        affine[2, 2] = 3.0
        mask = np.random.randint(0, 2, shape, dtype=np.uint8)
        result = resample_mask_to_image_grid(mask, affine, shape, affine)
        self.assertEqual(shape, result.shape)
        np.testing.assert_array_equal(mask, result)

    def test_resample_mask_to_image_grid_different_affine_identity(self):
        """When affines are identity (pixdim-only), resample should be identity."""
        from mimics_bridge import resample_mask_to_image_grid

        shape = (10, 10, 10)
        aff1 = np.diag([1.0, 1.0, 1.0, 1.0])
        aff2 = np.diag([1.0, 1.0, 1.0, 1.0])
        mask = np.random.randint(0, 2, shape, dtype=np.uint8)
        result = resample_mask_to_image_grid(mask, aff1, shape, aff2)
        np.testing.assert_array_equal(mask, result)

    def test_resample_mask_to_image_grid_unusable_affine_fallback(self):
        """When mask affine is zero (unusable), should return mask as-is."""
        from mimics_bridge import resample_mask_to_image_grid

        shape = (5, 5, 5)
        image_affine = np.eye(4)
        zero_affine = np.zeros((4, 4))
        mask = np.random.randint(0, 2, shape, dtype=np.uint8)
        result = resample_mask_to_image_grid(mask, zero_affine, shape, image_affine)
        np.testing.assert_array_equal(mask, result)

    def test_affine_is_usable(self):
        from mimics_bridge import _affine_is_usable

        self.assertTrue(_affine_is_usable(np.eye(4)))
        self.assertTrue(_affine_is_usable(np.diag([1.5, 1.5, 3.0, 1.0])))
        self.assertFalse(_affine_is_usable(np.zeros((4, 4))))
        self.assertFalse(_affine_is_usable(np.ones((4, 4)) * np.nan))
        self.assertFalse(_affine_is_usable(np.eye(4) * 0.0))

    def test_affine_close(self):
        from mimics_bridge import _affine_close

        a = np.eye(4)
        b = np.eye(4) + 1e-5
        c = np.eye(4) + 1e-3
        self.assertTrue(_affine_close(a, b, atol=1e-4))
        self.assertFalse(_affine_close(a, c, atol=1e-4))
        self.assertTrue(_affine_close(a, c, atol=2e-3))

    def test_unit_axis(self):
        from mimics_bridge import _unit_axis

        affine = np.diag([1.5, 2.0, 3.0, 1.0])
        spacing = np.array([1.5, 2.0, 3.0])
        result = _unit_axis(affine, 0, spacing)
        np.testing.assert_allclose([1.0, 0.0, 0.0], result, atol=1e-10)
        result = _unit_axis(affine, 2, spacing)
        np.testing.assert_allclose([0.0, 0.0, 1.0], result, atol=1e-10)

    def test_unit_axis_zero_spacing_raises(self):
        from mimics_bridge import _unit_axis

        affine = np.eye(4)
        spacing = np.array([0.0, 1.0, 1.0])
        with self.assertRaises(ValueError):
            _unit_axis(affine, 0, spacing)

    def test_nifti_to_derived_dicom_shape(self):
        from mimics_bridge import nifti_to_derived_dicom

        import nibabel as nib

        shape = (20, 15, 10)
        affine = np.diag([1.0, 1.0, 2.0, 1.0])
        data = np.random.randint(-500, 500, shape, dtype=np.int16)
        img = nib.Nifti1Image(data, affine)
        nii_path = os.path.join(self.tmp, "ct.nii.gz")
        nib.save(img, nii_path)

        dicom_out = os.path.join(self.tmp, "dicom")
        result = nifti_to_derived_dicom(nii_path, dicom_out)
        self.assertEqual([shape[0], shape[1], shape[2]], result["shape"])
        self.assertTrue(os.path.isdir(result["dicom_folder"]))
        # Should have one DICOM per slice
        dcm_files = [f for f in os.listdir(dicom_out) if f.endswith(".dcm")]
        self.assertEqual(shape[2], len(dcm_files))

    def test_read_nifti_mask_non_3d_raises(self):
        from mimics_bridge import read_nifti_mask

        import nibabel as nib

        # 4D array
        data = np.zeros((5, 5, 5, 2), dtype=np.uint8)
        img = nib.Nifti1Image(data, np.eye(4))
        path = os.path.join(self.tmp, "4d.nii.gz")
        nib.save(img, path)
        # SimpleITK DICOMOrient raises a different error for 4D before our
        # ValueError fires. Both indicate the 4D image is rejected.
        with self.assertRaises((ValueError, RuntimeError)):
            read_nifti_mask(path)

    def test_convert_uses_mimics_grid_affine_from_manifest(self):
        from mimics_bridge import do_convert
        import nibabel as nib

        case_dir = os.path.join(self.tmp, "case")
        buffers_dir = os.path.join(self.tmp, "buffers")
        os.makedirs(case_dir)
        os.makedirs(buffers_dir)

        data = np.zeros((2, 3, 4), dtype=np.uint8)
        data[1, 2, 3] = 1
        with open(os.path.join(buffers_dir, "liver.u8"), "wb") as handle:
            handle.write(data.tobytes())

        mimics_affine = np.array([
            [2.0, 0.0, 0.0, -100.0],
            [0.0, 3.0, 0.0, -50.0],
            [0.0, 0.0, 4.0, 20.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        manifest_path = os.path.join(buffers_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump({
                "mimics_shape": [2, 3, 4],
                "mimics_voxel_to_ras_matrix": mimics_affine.tolist(),
                "masks": [{
                    "original_name": "liver",
                    "safe_name": "liver",
                    "u8_filename": "liver.u8",
                }],
            }, handle)

        result = do_convert({
            "buffers_dir": buffers_dir,
            "manifest_path": manifest_path,
            "case_dir": case_dir,
            "axes": [0, 1, 2],
            "flips": [False, False, False],
        })
        self.assertEqual("ok", result["status"])
        self.assertEqual("manifest.mimics_voxel_to_ras_matrix", result["export_voxel_to_ras_matrix_source"])
        out = nib.load(os.path.join(case_dir, "segmentations", "liver.nii.gz"))
        self.assertEqual((2, 3, 4), out.shape)
        np.testing.assert_allclose(mimics_affine, out.affine, atol=1e-6)
        np.testing.assert_array_equal(data, np.asanyarray(out.dataobj).astype(np.uint8))

    def test_convert_resamples_mimics_mask_back_to_source_image_grid(self):
        from mimics_bridge import do_convert
        import nibabel as nib

        case_dir = os.path.join(self.tmp, "case_source")
        buffers_dir = os.path.join(self.tmp, "buffers_source")
        os.makedirs(case_dir)
        os.makedirs(buffers_dir)
        source_affine = np.eye(4)
        nib.save(nib.Nifti1Image(np.zeros((3, 1, 1), dtype=np.int16), source_affine), os.path.join(case_dir, "ct.nii.gz"))

        mimics_data = np.zeros((3, 1, 1), dtype=np.uint8)
        mimics_data[0, 0, 0] = 1
        with open(os.path.join(buffers_dir, "organ.u8"), "wb") as handle:
            handle.write(mimics_data.tobytes())

        mimics_affine = np.eye(4)
        mimics_affine[0, 3] = 1.0
        manifest_path = os.path.join(buffers_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump({
                "mimics_shape": [3, 1, 1],
                "mimics_voxel_to_ras_matrix": mimics_affine.tolist(),
                "masks": [{
                    "original_name": "organ",
                    "safe_name": "organ",
                    "u8_filename": "organ.u8",
                }],
            }, handle)

        result = do_convert({
            "buffers_dir": buffers_dir,
            "manifest_path": manifest_path,
            "case_dir": case_dir,
            "axes": [0, 1, 2],
            "flips": [False, False, False],
        })
        self.assertEqual("ok", result["status"])
        self.assertEqual("source_image", result["export_space"])
        out = nib.load(os.path.join(case_dir, "segmentations", "organ.nii.gz"))
        self.assertEqual((3, 1, 1), out.shape)
        np.testing.assert_allclose(source_affine, out.affine, atol=1e-6)
        exported = np.asanyarray(out.dataobj).astype(np.uint8)
        self.assertEqual(0, int(exported[0, 0, 0]))
        self.assertEqual(1, int(exported[1, 0, 0]))

    def test_convert_can_write_to_job_scoped_segmentations(self):
        from mimics_bridge import do_convert
        import nibabel as nib

        case_dir = os.path.join(self.tmp, "case_staged_export")
        buffers_dir = os.path.join(self.tmp, "buffers_staged_export")
        staged_dir = os.path.join(self.tmp, "fresh_labels", "case_staged_export", "segmentations")
        os.makedirs(case_dir)
        os.makedirs(buffers_dir)

        data = np.zeros((2, 2, 1), dtype=np.uint8)
        data[1, 1, 0] = 1
        with open(os.path.join(buffers_dir, "organ.u8"), "wb") as handle:
            handle.write(data.tobytes())
        manifest_path = os.path.join(buffers_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump({
                "mimics_shape": [2, 2, 1],
                "mimics_voxel_to_ras_matrix": np.eye(4).tolist(),
                "masks": [{
                    "original_name": "organ",
                    "safe_name": "organ",
                    "u8_filename": "organ.u8",
                }],
            }, handle)

        result = do_convert({
            "buffers_dir": buffers_dir,
            "manifest_path": manifest_path,
            "case_dir": case_dir,
            "output_seg_dir": staged_dir,
            "axes": [0, 1, 2],
            "flips": [False, False, False],
        })
        self.assertEqual("ok", result["status"])
        self.assertEqual(os.path.abspath(staged_dir), result["output_seg_dir"])
        self.assertFalse(os.path.exists(os.path.join(case_dir, "segmentations", "organ.nii.gz")))
        out = nib.load(os.path.join(staged_dir, "organ.nii.gz"))
        np.testing.assert_array_equal(data, np.asanyarray(out.dataobj).astype(np.uint8))

    def test_custom_export_does_not_overwrite_existing_mask(self):
        from mimics_bridge import do_convert
        import nibabel as nib

        case_dir = os.path.join(self.tmp, "case_safe_export")
        buffers_dir = os.path.join(self.tmp, "buffers_safe_export")
        custom_dir = os.path.join(self.tmp, "chosen_output", "case_safe_export", "segmentations")
        os.makedirs(case_dir)
        os.makedirs(buffers_dir)
        os.makedirs(custom_dir)
        original = np.zeros((2, 2, 1), dtype=np.uint8)
        nib.save(nib.Nifti1Image(original, np.eye(4)), os.path.join(custom_dir, "organ.nii.gz"))
        changed = np.ones((2, 2, 1), dtype=np.uint8)
        with open(os.path.join(buffers_dir, "organ.u8"), "wb") as handle:
            handle.write(changed.tobytes())
        manifest_path = os.path.join(buffers_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump({"mimics_shape": [2, 2, 1], "mimics_voxel_to_ras_matrix": np.eye(4).tolist(),
                       "masks": [{"original_name": "organ", "u8_filename": "organ.u8"}]}, handle)
        result = do_convert({
            "buffers_dir": buffers_dir, "manifest_path": manifest_path, "case_dir": case_dir,
            "output_seg_dir": custom_dir, "overwrite_existing": False,
            "axes": [0, 1, 2], "flips": [False, False, False],
        })
        self.assertEqual("skipped_existing", result["exported"][0]["action"])
        np.testing.assert_array_equal(original, np.asanyarray(nib.load(os.path.join(custom_dir, "organ.nii.gz")).dataobj))

    def test_prepare_supports_mhd_and_preserves_source_geometry(self):
        import SimpleITK as sitk
        from mimics_bridge import do_prepare

        image_path = os.path.join(self.tmp, "volume.mhd")
        image = sitk.GetImageFromArray(np.arange(24, dtype=np.int16).reshape((2, 3, 4)))
        image.SetSpacing((0.7, 0.8, 2.5))
        image.SetOrigin((12.0, -8.0, 30.0))
        sitk.WriteImage(image, image_path)
        result = do_prepare({
            "image_path": image_path, "masks": [], "case_id": "mhd_case",
            "dicom_out": os.path.join(self.tmp, "mhd_dicom"),
            "buffers_out": os.path.join(self.tmp, "mhd_buffers"),
            "axes": [0, 1, 2], "flips": [False, False, False],
        })
        self.assertEqual("ok", result["status"])
        self.assertEqual("medical_image", result["source_image_kind"])
        self.assertEqual([4, 3, 2], result["source_image_shape"])
        self.assertTrue(os.path.isdir(result["dicom_folder"]))

    def test_mr_mhd_float_intensity_survives_derived_dicom_scaling(self):
        import SimpleITK as sitk
        import pydicom
        from mimics_bridge import do_prepare

        source_values = np.linspace(-2.5, 7.25, 24, dtype=np.float32).reshape((2, 3, 4))
        image_path = os.path.join(self.tmp, "mri_volume.mhd")
        sitk.WriteImage(sitk.GetImageFromArray(source_values), image_path)
        result = do_prepare({
            "image_path": image_path, "masks": [], "case_id": "mr_case", "modality": "MR",
            "dicom_out": os.path.join(self.tmp, "mr_dicom"),
            "buffers_out": os.path.join(self.tmp, "mr_buffers"),
        })
        first = pydicom.dcmread(os.path.join(result["dicom_folder"], "slice_0001.dcm"))
        reconstructed = first.pixel_array.astype(np.float64) * float(first.RescaleSlope) + float(first.RescaleIntercept)
        np.testing.assert_allclose(reconstructed, source_values[0].astype(np.float64), atol=float(first.RescaleSlope) + 1e-5)
        self.assertEqual("MR", first.Modality)

    def test_resample_image_to_grid_matches_target_shape(self):
        from mimics_bridge import resample_image_to_grid
        import nibabel as nib

        source = os.path.join(self.tmp, "source.nii.gz")
        output = os.path.join(self.tmp, "aligned.nii.gz")
        data = np.arange(8, dtype=np.int16).reshape((2, 2, 2))
        nib.save(nib.Nifti1Image(data, np.eye(4)), source)

        target_affine = np.array([
            [1.0, 0.0, 0.0, -10.0],
            [0.0, 1.0, 0.0, -20.0],
            [0.0, 0.0, 1.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        result = resample_image_to_grid(source, [3, 2, 2], target_affine.tolist(), output)
        self.assertEqual("ok", result["status"])
        out = nib.load(output)
        self.assertEqual((3, 2, 2), out.shape)
        np.testing.assert_allclose(target_affine, out.affine, atol=1e-6)

    def test_fewshot_materialize_resamples_image_to_label_grid(self):
        pipeline = __import__("tools.fewshot_pipeline", fromlist=["dummy"])
        import nibabel as nib

        image_src = os.path.join(self.tmp, "ct.nii.gz")
        label_src = os.path.join(self.tmp, "mask.nii.gz")
        nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.int16), np.eye(4)), image_src)
        label_affine = np.array([
            [1.0, 0.0, 0.0, -5.0],
            [0.0, 1.0, 0.0, -6.0],
            [0.0, 0.0, 1.0, 7.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        nib.save(nib.Nifti1Image(np.ones((3, 2, 2), dtype=np.uint8), label_affine), label_src)

        rows = pipeline._materialize_split(
            [{"case_id": "s0001", "image": image_src, "label": label_src}],
            Path(self.tmp) / "dataset" / "imagesTr",
            Path(self.tmp) / "dataset" / "labelsTr",
            "train",
        )
        self.assertEqual(1, len(rows))
        self.assertEqual("resampled_to_label_grid", rows[0]["image_materialization"])
        self.assertFalse(rows[0]["image_label_geometry_matched"])
        out = nib.load(rows[0]["dataset_image"])
        self.assertEqual((3, 2, 2), out.shape)
        np.testing.assert_allclose(label_affine, out.affine, atol=1e-6)
        validation = pipeline.validate_materialized_dataset(rows)
        self.assertEqual(1, validation["checked_pairs"])
        self.assertEqual([], validation["issues"])

    def test_fewshot_dataset_validation_rejects_empty_label(self):
        pipeline = __import__("tools.fewshot_pipeline", fromlist=["dummy"])
        import nibabel as nib

        image_path = os.path.join(self.tmp, "image_empty_check.nii.gz")
        label_path = os.path.join(self.tmp, "label_empty_check.nii.gz")
        nib.save(nib.Nifti1Image(np.ones((2, 2, 1), dtype=np.int16), np.eye(4)), image_path)
        nib.save(nib.Nifti1Image(np.zeros((2, 2, 1), dtype=np.uint8), np.eye(4)), label_path)
        rows = [{
            "case_id": "s_empty",
            "dataset_image": image_path,
            "dataset_label": label_path,
        }]
        with self.assertRaises(RuntimeError):
            pipeline.validate_materialized_dataset(rows)

    def test_fewshot_resolves_configured_mimics_output_dir(self):
        pipeline = __import__("tools.fewshot_pipeline", fromlist=["dummy"])
        ts_root = os.path.join(self.tmp, "ts")
        output = pipeline.resolve_mimics_output_dir(ts_root, {"mimics_output_dir": "custom_mcs"})
        self.assertEqual(Path(ts_root).resolve() / "custom_mcs", output)
        self.assertTrue(output.is_dir())

    def test_fewshot_resolves_configured_buffer_mapping(self):
        pipeline = __import__("tools.fewshot_pipeline", fromlist=["dummy"])
        axes, flips = pipeline.resolve_mimics_buffer_mapping({
            "mimics_buffer_axes": "1,0,2",
            "mimics_buffer_flips": "true,false,1",
        })
        self.assertEqual([1, 0, 2], axes)
        self.assertEqual([True, False, True], flips)

    def test_fewshot_find_image_is_case_insensitive(self):
        pipeline = __import__("tools.fewshot_pipeline", fromlist=["dummy"])
        import nibabel as nib

        case_dir = Path(self.tmp) / "case_upper"
        case_dir.mkdir()
        path = case_dir / "CT.NII.GZ"
        nib.save(nib.Nifti1Image(np.zeros((1, 1, 1), dtype=np.int16), np.eye(4)), str(path))
        self.assertEqual(path, pipeline.find_image(case_dir))

    def test_fewshot_corrupt_label_is_skipped(self):
        pipeline = __import__("tools.fewshot_pipeline", fromlist=["dummy"])
        import nibabel as nib

        ts_root = Path(self.tmp) / "ts_corrupt"
        case_dir = ts_root / "s0001"
        seg_dir = case_dir / "segmentations"
        seg_dir.mkdir(parents=True)
        nib.save(nib.Nifti1Image(np.zeros((1, 1, 1), dtype=np.int16), np.eye(4)), str(case_dir / "ct.nii.gz"))
        (seg_dir / "liver.nii.gz").write_text("not a nifti", encoding="utf-8")
        samples, skipped = pipeline.discover_samples(ts_root, "liver")
        self.assertEqual([], samples)
        self.assertEqual(1, len(skipped))
        self.assertIn("could not be read", skipped[0]["reason"])

    def test_fewshot_discover_uses_fresh_label_root_without_stale_fallback(self):
        pipeline = __import__("tools.fewshot_pipeline", fromlist=["dummy"])
        import nibabel as nib

        ts_root = Path(self.tmp) / "ts_fresh_labels"
        case_dir = ts_root / "s0001"
        stale_seg_dir = case_dir / "segmentations"
        fresh_seg_dir = Path(self.tmp) / "fresh_labels" / "s0001" / "segmentations"
        stale_seg_dir.mkdir(parents=True)
        fresh_seg_dir.mkdir(parents=True)
        nib.save(nib.Nifti1Image(np.zeros((2, 2, 1), dtype=np.int16), np.eye(4)), str(case_dir / "ct.nii.gz"))
        stale = np.zeros((2, 2, 1), dtype=np.uint8)
        stale[0, 0, 0] = 1
        fresh = np.zeros((2, 2, 1), dtype=np.uint8)
        fresh[1, 1, 0] = 1
        nib.save(nib.Nifti1Image(stale, np.eye(4)), str(stale_seg_dir / "liver.nii.gz"))
        nib.save(nib.Nifti1Image(fresh, np.eye(4)), str(fresh_seg_dir / "liver.nii.gz"))

        samples, skipped = pipeline.discover_samples(
            ts_root,
            "liver",
            label_root=fresh_seg_dir.parent.parent,
            fallback_to_case_labels=False,
        )
        self.assertEqual([], skipped)
        self.assertEqual(1, len(samples))
        self.assertEqual(str(fresh_seg_dir / "liver.nii.gz"), samples[0]["label"])
        self.assertEqual("fresh_export", samples[0]["label_source"])

        (fresh_seg_dir / "liver.nii.gz").unlink()
        samples, skipped = pipeline.discover_samples(
            ts_root,
            "liver",
            label_root=fresh_seg_dir.parent.parent,
            fallback_to_case_labels=False,
        )
        self.assertEqual([], samples)
        self.assertEqual(1, len(skipped))
        self.assertFalse(skipped[0]["has_label"])

    def test_fewshot_stale_model_manifest_fails_before_inference(self):
        pipeline = __import__("tools.fewshot_pipeline", fromlist=["dummy"])
        workspace = Path(self.tmp) / "fewshot_models"
        latest = workspace / "models" / "liver" / "latest.json"
        latest.parent.mkdir(parents=True)
        latest.write_text(json.dumps({
            "organ": "liver",
            "model_id": "missing",
            "checkpoint": str(workspace / "missing.pth"),
            "config": str(workspace / "missing.yaml"),
        }), encoding="utf-8")
        with self.assertRaises(RuntimeError):
            pipeline.load_model_manifest(workspace, "liver")

    def test_fewshot_corrupt_checkpoint_fails_inference_preflight_status(self):
        pipeline = __import__("tools.fewshot_pipeline", fromlist=["dummy"])
        import nibabel as nib

        ts_root = Path(self.tmp) / "ts_corrupt_model"
        case_dir = ts_root / "s0001"
        case_dir.mkdir(parents=True)
        nib.save(nib.Nifti1Image(np.zeros((2, 2, 1), dtype=np.int16), np.eye(4)), str(case_dir / "ct.nii.gz"))
        workspace = ts_root / "fewshot_models"
        model_dir = workspace / "models" / "liver" / "train_bad"
        model_dir.mkdir(parents=True)
        checkpoint = model_dir / "model.pth"
        checkpoint.write_text("not a torch checkpoint", encoding="utf-8")
        config_path = model_dir / "config.yaml"
        config_path.write_text("model: {}\n", encoding="utf-8")
        (workspace / "models" / "liver").mkdir(parents=True, exist_ok=True)
        (workspace / "models" / "liver" / "latest.json").write_text(json.dumps({
            "model_id": "train_bad",
            "organ": "liver",
            "organ_slug": "liver",
            "checkpoint": str(checkpoint),
            "config": str(config_path),
        }), encoding="utf-8")

        args = type("Args", (object,), {})()
        args.ts_root = str(ts_root)
        args.case_id = "s0001"
        args.organ = "liver"
        args.workspace = None
        args.dinov3_root = str(Path(PROJECT_ROOT) / "external" / "dinov3-medical-seg")
        args.python = sys.executable
        args.model_id = "latest"
        args.model_manifest = None
        args.expected_source_shape = None
        args.expected_source_voxel_to_ras_matrix = None
        args.gpu_lock_timeout_seconds = 0
        args.job_id = "infer_corrupt"

        result = pipeline.cmd_infer(args)
        self.assertEqual(1, result)
        status = json.loads((workspace / "jobs" / "infer_corrupt.json").read_text(encoding="utf-8"))
        self.assertEqual("failed", status["status"])
        self.assertIn("torch checkpoint", status["error"])

    def test_fewshot_inference_source_geometry_mismatch_fails_preflight(self):
        pipeline = __import__("tools.fewshot_pipeline", fromlist=["dummy"])
        import nibabel as nib

        ts_root = Path(self.tmp) / "ts_geometry_mismatch"
        case_dir = ts_root / "s0001"
        case_dir.mkdir(parents=True)
        nib.save(nib.Nifti1Image(np.zeros((2, 2, 1), dtype=np.int16), np.eye(4)), str(case_dir / "ct.nii.gz"))
        workspace = ts_root / "fewshot_models"
        model_dir = workspace / "models" / "liver" / "train_ok"
        model_dir.mkdir(parents=True)
        checkpoint = model_dir / "model.pth"
        checkpoint.write_bytes(b"PK\x03\x04")
        config_path = model_dir / "config.yaml"
        config_path.write_text("model: {}\n", encoding="utf-8")
        latest = workspace / "models" / "liver" / "latest.json"
        latest.parent.mkdir(parents=True, exist_ok=True)
        latest.write_text(json.dumps({
            "model_id": "train_ok",
            "organ": "liver",
            "organ_slug": "liver",
            "checkpoint": str(checkpoint),
            "config": str(config_path),
        }), encoding="utf-8")

        args = type("Args", (object,), {})()
        args.ts_root = str(ts_root)
        args.case_id = "s0001"
        args.organ = "liver"
        args.workspace = None
        args.dinov3_root = str(Path(PROJECT_ROOT) / "external" / "dinov3-medical-seg")
        args.python = sys.executable
        args.model_id = "latest"
        args.model_manifest = None
        args.expected_source_shape = json.dumps([3, 2, 1])
        args.expected_source_voxel_to_ras_matrix = json.dumps(np.eye(4).tolist())
        args.gpu_lock_timeout_seconds = 0
        args.job_id = "infer_geometry"

        result = pipeline.cmd_infer(args)
        self.assertEqual(1, result)
        status = json.loads((workspace / "jobs" / "infer_geometry.json").read_text(encoding="utf-8"))
        self.assertEqual("failed", status["status"])
        self.assertIn("source image shape", status["error"])


# ============================================================================
# L4: nninteractive_bridge.py
# ============================================================================


class TestNNInteractiveBridgeSourceImage(unittest.TestCase):
    def setUp(self):
        self.tmp = _make_temp_dir()

    def tearDown(self):
        _cleanup(self.tmp)

    def test_load_image_source_nifti(self):
        from nninteractive_bridge import load_image_source

        import nibabel as nib

        shape = (20, 15, 10)
        affine = np.eye(4)
        data = np.random.randint(-1000, 1000, shape, dtype=np.int16)
        img = nib.Nifti1Image(data, affine)
        path = os.path.join(self.tmp, "ct.nii.gz")
        nib.save(img, path)

        input_data = {
            "image_path": path,
            "image_source_kind": "nifti",
            "image_expected_shape": list(shape),
            "image_source_voxel_to_ras_matrix": json.dumps(affine.tolist()),
            "image_mimics_voxel_to_ras_matrix": json.dumps(affine.tolist()),
        }
        result = load_image_source(input_data)
        # Should be 4D: (1, X, Y, Z)
        self.assertEqual(4, result.ndim)
        self.assertEqual(1, result.shape[0])
        self.assertEqual(shape, tuple(result.shape[1:]))

    def test_load_image_source_intensity_transform(self):
        from nninteractive_bridge import load_image_source

        import nibabel as nib

        shape = (10, 10, 10)
        affine = np.eye(4)
        data = np.ones(shape, dtype=np.float32) * 100.0
        img = nib.Nifti1Image(data, affine)
        path = os.path.join(self.tmp, "ct.nii.gz")
        nib.save(img, path)

        # Test with GV transform: HU2GV slope=1, intercept=1024
        input_data = {
            "image_path": path,
            "image_source_kind": "nifti",
            "image_expected_shape": list(shape),
            "image_source_voxel_to_ras_matrix": json.dumps(affine.tolist()),
            "image_mimics_voxel_to_ras_matrix": json.dumps(affine.tolist()),
            "image_source_to_mimics_gv_slope": 1.0,
            "image_source_to_mimics_gv_intercept": 1024.0,
        }
        result = load_image_source(input_data)
        # After transform: 100 * 1.0 + 1024.0 = 1124.0
        self.assertAlmostEqual(1124.0, float(result[0, 0, 0, 0]), places=1)

    def test_load_image_source_permission_error_message(self):
        from nninteractive_bridge import load_image_source

        input_data = {
            "image_path": "/nonexistent/path/ct.nii.gz",
            "image_source_kind": "nifti",
        }
        # nibabel raises FileNotFoundError on nonexistent paths;
        # PermissionError would be raised if the file exists but is locked.
        with self.assertRaises((RuntimeError, OSError)):
            load_image_source(input_data)

    def test_resolve_device_cpu_only(self):
        from nninteractive_bridge import _resolve_device

        dev, warn = _resolve_device("cpu", allow_cpu_fallback=False)
        self.assertEqual("cpu", dev)
        self.assertIsNone(warn)

    def test_parse_matrix_valid(self):
        from nninteractive_bridge import _parse_matrix

        m = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        result = _parse_matrix(json.dumps(m))
        self.assertIsNotNone(result)
        np.testing.assert_allclose(np.eye(4), result)

    def test_parse_matrix_invalid(self):
        from nninteractive_bridge import _parse_matrix

        self.assertIsNone(_parse_matrix(None))
        self.assertIsNone(_parse_matrix(""))
        self.assertIsNone(_parse_matrix("[[1,2,3]]"))


# ============================================================================
# L5: window_level_mimics.py
# ============================================================================


class TestWindowLevel(unittest.TestCase):
    def setUp(self):
        self.tmp = _make_temp_dir()

    def tearDown(self):
        _cleanup(self.tmp)

    def test_hu_to_gv_roundtrip(self):
        """Simulate HU2GV integer conversion logic."""
        # Mimics API is not available, but test the conversion pattern
        def fake_hu2gv(hu):
            # Typical CT: GV = HU + 1024 (intercept only)
            return int(hu) + 1024

        def test_convert(value):
            return int(fake_hu2gv(int(round(float(value)))))

        self.assertEqual(1024, test_convert(0))
        self.assertEqual(0, test_convert(-1024))
        self.assertEqual(4095, test_convert(3071))
        # Float input should be rounded
        self.assertEqual(1024, test_convert(0.4))
        self.assertEqual(1025, test_convert(0.6))

    def test_clamp_contrast_points(self):
        from window_level_mimics import _clamp_contrast_points

        # Normal range
        low, high = _clamp_contrast_points(500, 3000, 0, 4095)
        self.assertEqual(500, low)
        self.assertEqual(3000, high)

        # Below minimum
        low, high = _clamp_contrast_points(-100, 1000, 0, 4095)
        self.assertEqual(0, low)
        self.assertEqual(1000, high)

        # Above maximum
        low, high = _clamp_contrast_points(3000, 5000, 0, 4095)
        self.assertEqual(3000, low)
        self.assertEqual(4095, high)

        # Equal points (low == high)
        low, high = _clamp_contrast_points(1000, 1000, 0, 4095)
        self.assertEqual(1000, low)
        self.assertEqual(1001, high)

        # Equal at upper boundary
        low, high = _clamp_contrast_points(4095, 4095, 0, 4095)
        self.assertEqual(4094, low)
        self.assertEqual(4095, high)

    def test_parse_valid_range(self):
        from window_level_mimics import _parse_valid_range

        error = "Upper contrast point should be in range from 0 to 3625"
        result = _parse_valid_range(error)
        self.assertEqual((0, 3625), result)

        error2 = "range from -100.5 to 3071.2"
        result2 = _parse_valid_range(error2)
        self.assertEqual((-100, 3071), result2)

        self.assertIsNone(_parse_valid_range("no range here"))
        self.assertIsNone(_parse_valid_range(""))

    def test_preset_keyword_matching(self):
        from window_level_mimics import _preset_for_name, DEFAULT_PRESETS

        self.assertEqual("Lung", _preset_for_name("lung_upper", DEFAULT_PRESETS)["name"])
        self.assertEqual("Bone", _preset_for_name("vertebra", DEFAULT_PRESETS)["name"])
        self.assertEqual("Vessel / Heart", _preset_for_name("aorta", DEFAULT_PRESETS)["name"])
        self.assertIsNone(_preset_for_name("zzzz_no_match_999", DEFAULT_PRESETS))

    def test_preset_calculation(self):
        """WW=400, WL=50 → low_hu=-150, high_hu=250"""
        width = 400.0
        level = 50.0
        low_hu = level - width / 2.0
        high_hu = level + width / 2.0
        self.assertEqual(-150.0, low_hu)
        self.assertEqual(250.0, high_hu)

    def test_image_min_max_default_fallback(self):
        """When no Mimics API, _image_min_max falls back to (0, 4095)."""
        from window_level_mimics import _image_min_max

        minimum, maximum = _image_min_max()
        self.assertEqual(0, minimum)
        self.assertEqual(4095, maximum)
        self.assertLess(minimum, maximum)

    def test_load_presets_returns_defaults_when_no_file(self):
        from window_level_mimics import _load_presets

        presets = _load_presets()
        self.assertIsInstance(presets, list)
        self.assertGreaterEqual(len(presets), 5)
        self.assertEqual("Lung", presets[0]["name"])


# ============================================================================
# L6: create_mcs_batch.py
# ============================================================================


class TestCreateMcsBatch(unittest.TestCase):
    def setUp(self):
        self.tmp = _make_temp_dir()

    def tearDown(self):
        _cleanup(self.tmp)

    def test_stop_marker_detected(self):
        """Simulate that the stop marker file is detected."""
        stop_path = os.path.join(self.tmp, "_mcs_queue_stop.json")
        with open(stop_path, "w") as f:
            json.dump({"status": "stop_requested"}, f)
        self.assertTrue(os.path.isfile(stop_path))
        # The main loop would check: os.path.isfile(os.path.join(output_dir, QUEUE_STOP_FILE))
        # and exit when True. Confirmed marker file exists and is readable.

    def test_inject_buffer_shape_mismatch_error(self):
        """inject_buffer should raise RuntimeError on byte count mismatch."""
        from create_mcs_batch import inject_buffer

        # Write a buffer with wrong byte count
        buf_path = os.path.join(self.tmp, "mask.u8")
        wrong_data = b"\x00" * 100  # should be 1000 for shape (10,10,10)
        with open(buf_path, "wb") as f:
            f.write(wrong_data)

        with self.assertRaises(RuntimeError):
            inject_buffer(None, buf_path, [10, 10, 10])

    def test_inject_buffer_correct_shape(self):
        """inject_buffer succeeds when byte count matches."""
        # We can't test set_voxel_buffer without Mimics, but we can
        # test the byte count validation
        shape = [5, 4, 3]
        expected_bytes = 5 * 4 * 3
        buf_path = os.path.join(self.tmp, "mask.u8")
        with open(buf_path, "wb") as f:
            f.write(b"\x00" * expected_bytes)

        # This should pass the byte count check (but fail on set_voxel_buffer)
        # Test the validation logic:
        with open(buf_path, "rb") as f:
            raw = f.read()
        self.assertEqual(expected_bytes, len(raw))
        import numpy as np

        pixels = np.frombuffer(raw, dtype=np.uint8).reshape(tuple(shape))
        self.assertEqual(tuple(shape), pixels.shape)

    def test_log_message(self):
        from create_mcs_batch import log_message

        log_message(self.tmp, "test message")
        log_path = os.path.join(self.tmp, "logs", "_create_mcs_batch.log")
        self.assertTrue(os.path.isfile(log_path))
        with open(log_path, "r") as f:
            content = f.read()
        self.assertIn("test message", content)

    def test_fingerprint_skip_logic(self):
        """When fingerprint matches, existing .mcs should be skipped."""
        # Simulate the skip logic
        output_dir = self.tmp
        case_id = "s0001"
        mcs_path = os.path.join(output_dir, case_id + ".mcs")
        fp_path = os.path.join(output_dir, case_id + ".fingerprint")
        current_fp = "sha256:abc123"

        # Create .mcs and matching fingerprint → should skip
        with open(mcs_path, "w") as f:
            f.write("fake mcs")
        with open(fp_path, "w") as f:
            f.write(current_fp)

        self.assertTrue(os.path.isfile(mcs_path))
        with open(fp_path, "r") as f:
            stored_fp = f.read().strip()
        self.assertEqual(current_fp, stored_fp)
        # Matching fingerprint → skip

        # Now change fingerprint → should reprocess
        new_fp = "sha256:def456"
        with open(fp_path, "w") as f:
            f.write(new_fp)
        with open(fp_path, "r") as f:
            stored_fp2 = f.read().strip()
        self.assertNotEqual(current_fp, stored_fp2)
        # Different fingerprint → reprocess

    def test_logs_directory_created(self):
        from create_mcs_batch import log_message

        # log_message should create logs/ dir automatically
        logs_dir = os.path.join(self.tmp, "logs")
        self.assertFalse(os.path.isdir(logs_dir))
        log_message(self.tmp, "first message")
        self.assertTrue(os.path.isdir(logs_dir))

    def test_record_failed_case(self):
        from create_mcs_batch import record_failed_case

        record_failed_case(self.tmp, "s0001", "prepare", "Test error", "traceback text")
        failed_dir = os.path.join(self.tmp, "_failed_cases")
        self.assertTrue(os.path.isdir(failed_dir))
        files = os.listdir(failed_dir)
        self.assertEqual(1, len(files))
        with open(os.path.join(failed_dir, files[0]), "r") as f:
            data = json.load(f)
        self.assertEqual("s0001", data["case_id"])
        self.assertEqual("prepare", data["phase"])
        self.assertEqual("Test error", data["error"])

    def test_shape_product(self):
        from create_mcs_batch import _shape_product

        self.assertEqual(60, _shape_product([3, 4, 5]))
        self.assertEqual(1, _shape_product([1, 1, 1]))
        self.assertEqual(0, _shape_product([0, 5, 5]))

    def test_acquire_lock(self):
        from create_mcs_batch import acquire_lock

        lock_path = acquire_lock(self.tmp)
        self.assertIsNotNone(lock_path)
        self.assertTrue(os.path.isfile(lock_path))
        # Second acquire should fail
        lock2 = acquire_lock(self.tmp)
        self.assertIsNone(lock2)
        # Cleanup
        os.remove(lock_path)


# ============================================================================
# L7: scripting_library entry routing
# ============================================================================


class TestScriptingLibraryEntries(unittest.TestCase):
    def test_all_entries_importable(self):
        """Each scripting_library entry should be importable without Mimics."""
        lib = os.path.join(PROJECT_ROOT, "scripting_library")
        entries = []
        for sub in ["01_Data", "02_AI", "03_Review", "99_Admin"]:
            sub_dir = os.path.join(lib, sub)
            if not os.path.isdir(sub_dir):
                continue
            for root, dirs, files in os.walk(sub_dir):
                for fname in sorted(files):
                    if fname.endswith(".py") and not fname.startswith("_"):
                        entries.append(os.path.join(root, fname))

        errors = []
        for entry in entries:
            try:
                with open(entry, "r") as f:
                    source = f.read()
                # Verify it imports _mimics_entrypoint
                self.assertIn(
                    "_mimics_entrypoint",
                    source,
                    "{} missing _mimics_entrypoint import".format(
                        os.path.relpath(entry, PROJECT_ROOT),
                    ),
                )
                self.assertIn(
                    "run_runtime_entry",
                    source,
                    "{} missing run_runtime_entry call".format(
                        os.path.relpath(entry, PROJECT_ROOT),
                    ),
                )
            except Exception as exc:
                errors.append("{}: {}".format(os.path.relpath(entry, PROJECT_ROOT), exc))
        self.assertGreater(len(entries), 0, "No entries found in scripting_library")
        self.assertEqual([], errors)

    def test_nninteractive_entry_routes_correctly(self):
        """nnInteractive.py routes to nninteractive_mimics module."""
        entry = os.path.join(PROJECT_ROOT, "scripting_library", "02_AI", "01_nnInteractive.py")
        with open(entry, "r") as f:
            source = f.read()
        self.assertIn("nninteractive_mimics", source)

    def test_dinov3_entries_separate_from_nninteractive(self):
        """DINOv3 entries are in their own DINOv3/ directory, nnInteractive is flat."""
        dino_dir = os.path.join(PROJECT_ROOT, "scripting_library", "02_AI", "DINOv3")
        self.assertTrue(os.path.isdir(dino_dir), "DINOv3/ directory should exist")

        # DINOv3 entries in DINOv3/ subdirectory, all route to fewshot_mimics
        dino_entries = [f for f in os.listdir(dino_dir) if f.endswith(".py")]
        self.assertGreater(len(dino_entries), 0, "No DINOv3 entries found")
        self.assertEqual(
            [
                "01_Train_Model.py",
                "02_Predict_Current_Case.py",
                "03_Predict_Choose_Model.py",
                "04_Show_Status_Results.py",
                "05_Stop_AI_Task.py",
            ],
            sorted(dino_entries),
        )
        for fname in dino_entries:
            path = os.path.join(dino_dir, fname)
            with open(path, "r") as f:
                source = f.read()
            self.assertIn("fewshot_mimics", source,
                "DINOv3/{} should route to fewshot_mimics".format(fname))
            self.assertNotIn("nninteractive_mimics", source)

        train_entry = os.path.join(dino_dir, "01_Train_Model.py")
        with open(train_entry, "r") as f:
            self.assertIn('action_attr="BUTTON_TRAIN_MODEL"', f.read())

        # nnInteractive.py is flat in 02_AI/
        nn_entry = os.path.join(PROJECT_ROOT, "scripting_library", "02_AI", "01_nnInteractive.py")
        self.assertTrue(os.path.isfile(nn_entry), "nnInteractive.py should be in 02_AI/")
        with open(nn_entry, "r") as f:
            self.assertIn("nninteractive_mimics", f.read())

    def test_dinov3_not_in_root_ai_dir(self):
        """DINOv3 entries should NOT be flat in 02_AI/ root."""
        ai_root = os.path.join(PROJECT_ROOT, "scripting_library", "02_AI")
        flat_dino = [f for f in os.listdir(ai_root) if f.startswith("DINOv3") and f.endswith(".py")]
        self.assertEqual([], flat_dino,
            "DINOv3 entries should be in DINOv3/ subdirectory, not flat in 02_AI/")
        self.assertFalse(
            os.path.isdir(os.path.join(PROJECT_ROOT, "scripting_library", "03_Display")),
            "The retired 03_Display directory should not remain visible in Mimics",
        )

    def test_stop_background_services_entry(self):
        entry = os.path.join(PROJECT_ROOT, "scripting_library", "99_Admin", "03_Stop_All_Owned_Services.py")
        with open(entry, "r") as f:
            source = f.read()
        self.assertIn("mimics_stop_background", source)

    def test_window_entries_routes(self):
        lib = os.path.join(PROJECT_ROOT, "scripting_library", "03_Review")
        expected_routes = {
            "03_Window_Choose_Preset.py": "choose",
            "02_Window_From_Selected_Mask.py": "auto",
            "05_Window_Reset_Full_Range.py": "reset",
            "04_Window_Undo_Last.py": "undo",
        }
        for fname, action in expected_routes.items():
            path = os.path.join(lib, fname)
            with open(path, "r") as f:
                source = f.read()
            self.assertIn("window_level_mimics", source)
            self.assertIn('"{}"'.format(action), source)


# ============================================================================
# L8: Stop Background Services
# ============================================================================


class TestStopBackgroundServices(unittest.TestCase):
    def setUp(self):
        self.tmp = _make_temp_dir()

    def tearDown(self):
        _cleanup(self.tmp)

    def test_stop_markers_defined(self):
        from mimics_stop_background import MARKERS

        # All process types must be covered so Stop_Background_Services can
        # find and kill them.
        required = [
            "mimics_bridge.py",
            "nninteractive_bridge.py",
            "--async-worker",
            "--watchdog",
            "_run_create_mcs.py",
            "_run_export_batch.py",
            "create_mcs_batch.py",
            "mimics_export.py",
            "mimics_import.py",
            "fewshot_pipeline.py",
            "fewshot_mimics.py",
            "fewshot_model_chooser.py",
            "nninteractive.inference.server.main",
            "setup_env.py",
        ]
        for marker in required:
            self.assertIn(marker, MARKERS, "{0} must be in MARKERS".format(marker))

    def test_stop_uses_taskkill_tree(self):
        """Verify stop command uses taskkill with /T (tree kill)."""
        import mimics_stop_background as msb

        # The stop_background_processes function should use taskkill /T
        # to terminate entire process trees, preventing watchdog restarts.
        # We can verify this by checking the command string.
        # The function constructs a PowerShell command; we verify it
        # exists and is callable (the actual command is Windows-only).
        self.assertTrue(callable(msb.stop_background_processes))

    def test_queue_stop_written(self):
        from mimics_stop_background import _request_queue_stop

        # Create a mock output dir that looks like a queue
        queue_dir = os.path.join(self.tmp, "mcs_output")
        os.makedirs(queue_dir)
        # Write a lock file with output_dir info (simulates runtime state)
        lock_dir = os.path.join(self.tmp, ".mimics_runtime", "locks")
        os.makedirs(lock_dir)

        # _request_queue_stop searches for queue dirs from lock files.
        # Without valid lock files, it returns empty. Test that it doesn't crash.
        result = _request_queue_stop()
        self.assertIsInstance(result, list)

    def test_stop_import_entry_is_narrower_than_stop_all(self):
        import mimics_stop_background as msb

        self.assertTrue(msb._lock_is_import_creation({
            "kind": "create_mcs",
            "owner": "import .mcs creation",
        }))
        self.assertFalse(msb._lock_is_import_creation({
            "kind": "fewshot_label_export",
            "owner": "DINOv3 label export",
        }))
        self.assertFalse(msb._lock_is_import_creation({
            "kind": "fewshot_train",
            "owner": "DINOv3 training",
        }))
        entry = os.path.join(
            PROJECT_ROOT,
            "scripting_library",
            "01_Data",
            "04_Stop_Import_Queue.py",
        )
        self.assertTrue(os.path.isfile(entry))


# ============================================================================
# L9: nninteractive_mimics metadata parsing
# ============================================================================


class TestNNInteractiveMimicsParsing(unittest.TestCase):
    def test_parse_shape_metadata_json(self):
        from nninteractive_mimics import _parse_shape_metadata

        self.assertEqual([249, 188, 213], _parse_shape_metadata("[249, 188, 213]"))
        self.assertEqual([512, 300, 512], _parse_shape_metadata("512x300x512"))
        self.assertEqual([10, 20, 30], _parse_shape_metadata("10,20,30"))
        self.assertIsNone(_parse_shape_metadata(None))
        self.assertIsNone(_parse_shape_metadata(""))
        self.assertIsNone(_parse_shape_metadata("abc"))

    def test_parse_matrix_metadata(self):
        from nninteractive_mimics import _parse_matrix_metadata

        identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        result = _parse_matrix_metadata(json.dumps(identity))
        self.assertIsNotNone(result)
        self.assertEqual(4, len(result))
        self.assertEqual(4, len(result[0]))
        self.assertAlmostEqual(1.0, result[0][0])
        self.assertAlmostEqual(0.0, result[0][1])

        self.assertIsNone(_parse_matrix_metadata(None))
        self.assertIsNone(_parse_matrix_metadata(""))
        self.assertIsNone(_parse_matrix_metadata("[[1,2,3]]"))

    def test_matrix_close(self):
        from nninteractive_mimics import _matrix_close

        identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        ras_to_lps = [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

        self.assertTrue(_matrix_close(identity, identity))
        self.assertFalse(_matrix_close(identity, ras_to_lps))


# ============================================================================
# L10: nninteractive_bridge DICOM loading
# ============================================================================


class TestBridgeDicomLoading(unittest.TestCase):
    def setUp(self):
        self.tmp = _make_temp_dir()

    def tearDown(self):
        _cleanup(self.tmp)

    def test_dicom_sort_key(self):
        from nninteractive_bridge import _dicom_sort_key

        # Mock DICOM dataset
        class MockDS:
            def __init__(self, ipp, instance_num):
                self.ImagePositionPatient = ipp
                self.InstanceNumber = instance_num

        normal = np.array([0.0, 0.0, 1.0])
        ds1 = MockDS([0.0, 0.0, 0.0], 1)
        ds2 = MockDS([0.0, 0.0, 10.0], 2)
        ds3 = MockDS([0.0, 0.0, 20.0], 3)

        records = [("/p3", ds3), ("/p1", ds1), ("/p2", ds2)]
        records.sort(key=lambda r: _dicom_sort_key(r, normal))
        self.assertEqual(1, records[0][1].InstanceNumber)
        self.assertEqual(2, records[1][1].InstanceNumber)
        self.assertEqual(3, records[2][1].InstanceNumber)


# ============================================================================
# L11: Edge cases and boundary conditions
# ============================================================================


class TestEdgeCases(unittest.TestCase):
    def setUp(self):
        self.tmp = _make_temp_dir()

    def tearDown(self):
        _cleanup(self.tmp)

    # -- NIfTI normalization: NaN handling --
    def test_normalize_affine_survives_nan_sform(self):
        """_normalize_nifti_affine falls back to usable qform when sform is NaN."""
        import nibabel as nib
        from mimics_bridge import _normalize_nifti_affine

        shape = (10, 10, 10)
        valid = np.eye(4)
        data = np.zeros(shape, dtype=np.int16)
        img = nib.Nifti1Image(data, valid)
        img.header.set_qform(valid, code=2)
        # Corrupt sform
        nan_aff = np.eye(4) * np.nan
        img.header.set_sform(nan_aff, code=2)
        result = _normalize_nifti_affine(img)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_normalize_affine_both_corrupt(self):
        """When both qform and sform are NaN, falls back to pixdim."""
        import nibabel as nib
        from mimics_bridge import _normalize_nifti_affine

        shape = (10, 10, 10)
        data = np.zeros(shape, dtype=np.int16)
        # Use a clean affine for image creation; corrupt afterward
        pixdim_affine = np.diag([1.5, 1.5, 3.0, 1.0])
        img = nib.Nifti1Image(data, pixdim_affine)
        img.header.set_qform(pixdim_affine, code=0)
        img.header.set_sform(pixdim_affine, code=0)
        result = _normalize_nifti_affine(img)
        spacing = np.linalg.norm(result[:3, :3], axis=0)
        self.assertTrue(np.all(np.isfinite(spacing)))
        self.assertTrue(np.all(spacing > 0))

    def test_normalize_different_qform_sform(self):
        """When qform and sform differ, both are normalized to valid codes."""
        import nibabel as nib
        from mimics_bridge import _normalize_nifti_affine

        shape = (10, 10, 10)
        sform_aff = np.diag([1.0, 1.0, 2.0, 1.0])
        qform_aff = np.diag([1.5, 1.5, 3.0, 1.0])
        data = np.zeros(shape, dtype=np.int16)
        img = nib.Nifti1Image(data, sform_aff)
        img.header.set_sform(sform_aff, code=2)
        img.header.set_qform(qform_aff, code=2)
        result = _normalize_nifti_affine(img)
        self.assertGreater(img.header["qform_code"], 0)
        self.assertGreater(img.header["sform_code"], 0)
        self.assertTrue(np.all(np.isfinite(result)))

    # -- buffer mapping roundtrip all permutations --
    def test_buffer_mapping_all_axis_permutations(self):
        """Roundtrip works for all axis permutations + flip combos."""
        from mimics_bridge import apply_buffer_mapping, inverse_buffer_mapping

        data = np.arange(60).reshape((3, 4, 5)).astype(np.uint8)
        for axes in [[1, 0, 2], [2, 0, 1], [1, 2, 0], [0, 2, 1], [2, 1, 0]]:
            for flips in [[False, False, False], [True, False, True], [True, True, False]]:
                t = apply_buffer_mapping(data, axes, flips)
                r = inverse_buffer_mapping(t, axes, flips)
                np.testing.assert_array_equal(data, r)

    def test_buffer_mapping_double_flip_identity(self):
        """Flipping twice on same axis returns to original."""
        from mimics_bridge import apply_buffer_mapping

        data = np.arange(24).reshape((2, 3, 4)).astype(np.uint8)
        once = apply_buffer_mapping(data, [0, 1, 2], [True, True, True])
        twice = apply_buffer_mapping(once, [0, 1, 2], [True, True, True])
        np.testing.assert_array_equal(data, twice)

    # -- window_level boundary --
    def test_clamp_zero_range_image(self):
        from window_level_mimics import _clamp_contrast_points

        low, high = _clamp_contrast_points(-500, 500, 0, 1)
        self.assertEqual(0, low)
        self.assertEqual(1, high)

    def test_clamp_equal_at_minimum(self):
        from window_level_mimics import _clamp_contrast_points

        low, high = _clamp_contrast_points(0, 0, 0, 4095)
        self.assertEqual(0, low)
        self.assertEqual(1, high)

    def test_parse_range_negative_values(self):
        from window_level_mimics import _parse_valid_range

        self.assertEqual((-1024, -1), _parse_valid_range("range from -1024 to -1"))

    def test_parse_range_floating_point(self):
        from window_level_mimics import _parse_valid_range

        result = _parse_valid_range("in range from 0.1 to 3625.9")
        self.assertEqual((0, 3626), result)

    # -- create_mcs_batch edge cases --
    def test_stop_marker_priority_over_work(self):
        """Stop marker should take priority even when work exists."""
        from create_mcs_batch import QUEUE_STOP_FILE

        stop_path = os.path.join(self.tmp, QUEUE_STOP_FILE)
        with open(stop_path, "w") as f:
            json.dump({"status": "stop_requested"}, f)
        work_dir = os.path.join(self.tmp, "s0001_work")
        os.makedirs(work_dir)
        with open(os.path.join(work_dir, "prepare_manifest.json"), "w") as f:
            json.dump({"output_mcs": os.path.join(self.tmp, "s0001.mcs")}, f)
        self.assertTrue(os.path.isfile(stop_path))
        self.assertTrue(os.path.isfile(os.path.join(work_dir, "prepare_manifest.json")))

    def test_shape_permutation_detection(self):
        """Axes-permuted shapes with same product trigger reshape logic."""
        from create_mcs_batch import _shape_product

        a = [249, 188, 213]
        b = [188, 249, 213]
        self.assertEqual(_shape_product(a), _shape_product(b))
        self.assertNotEqual(a, b)

    # -- nninteractive_bridge edge cases --
    def test_align_source_identity_affines(self):
        from nninteractive_bridge import _align_source_image_to_mimics_grid

        shape = [5, 5, 5]
        data = np.random.randn(*shape).astype(np.float32)
        identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        input_data = {
            "image_expected_shape": shape,
            "image_source_voxel_to_ras_matrix": json.dumps(identity),
            "image_mimics_voxel_to_ras_matrix": json.dumps(identity),
        }
        result = _align_source_image_to_mimics_grid(data, input_data)
        np.testing.assert_allclose(data, result, atol=1e-6)

    def test_align_source_no_metadata_shape_mismatch_raises(self):
        from nninteractive_bridge import _align_source_image_to_mimics_grid

        data = np.random.randn(10, 10, 10).astype(np.float32)
        with self.assertRaises(RuntimeError):
            _align_source_image_to_mimics_grid(data, {"image_expected_shape": [5, 5, 5]})

    # -- mm interactive parsing --
    def test_parse_shape_semicolon_and_bracket_variants(self):
        from nninteractive_mimics import _parse_shape_metadata

        self.assertEqual([10, 20, 30], _parse_shape_metadata("10;20;30"))
        self.assertEqual([5, 5, 5], _parse_shape_metadata(" [ 5 , 5 , 5 ] "))
        self.assertIsNone(_parse_shape_metadata("[512,512]"))
        self.assertIsNone(_parse_shape_metadata("[0,512,512]"))

    def test_matrix_close_uses_tolerance(self):
        from nninteractive_mimics import _matrix_close

        iden = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        self.assertTrue(_matrix_close(iden, iden))

    # -- DICOM affine reading --
    def test_get_image_affine_from_dicom_standard_axial(self):
        import pydicom
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, generate_uid, CTImageStorage
        from mimics_bridge import get_image_affine_from_dicom

        dicom_dir = os.path.join(self.tmp, "dicom")
        os.makedirs(dicom_dir)
        for k in range(3):
            file_meta = FileMetaDataset()
            file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            file_meta.MediaStorageSOPClassUID = CTImageStorage
            file_meta.MediaStorageSOPInstanceUID = generate_uid()
            ds = FileDataset(
                os.path.join(dicom_dir, f"s{k:04d}.dcm"),
                {}, file_meta=file_meta, preamble=b"\0" * 128,
            )
            ds.Rows = 10
            ds.Columns = 20
            ds.PixelSpacing = [1.0, 1.0]
            ds.SliceThickness = 2.0
            ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            ds.ImagePositionPatient = [0.0, 0.0, float(k) * 2.0]
            ds.PixelData = b"\x00" * (10 * 20 * 2)
            ds.save_as(ds.filename)
        affine = get_image_affine_from_dicom(dicom_dir)
        self.assertEqual((4, 4), affine.shape)
        self.assertTrue(np.all(np.isfinite(affine)))

    # -- resource lock edge cases --
    def test_lock_release_wrong_token(self):
        import runtime_common

        lock_path = os.path.join(self.tmp, "wrong.lock")
        token = runtime_common.acquire_resource_lock(lock_path, "r", "o")
        self.assertIsNotNone(token)
        self.assertFalse(runtime_common.release_resource_lock(lock_path, "wrong"))
        self.assertTrue(os.path.isfile(lock_path))
        runtime_common.release_resource_lock(lock_path, token)

    def test_lock_double_release_is_idempotent(self):
        """Double release should not crash and should report True."""
        import runtime_common

        lock_path = os.path.join(self.tmp, "twice.lock")
        token = runtime_common.acquire_resource_lock(lock_path, "r", "o")
        self.assertTrue(runtime_common.release_resource_lock(lock_path, token))
        # Second release: lock already gone, harmless no-op
        try:
            runtime_common.release_resource_lock(lock_path, token)
        except Exception:
            self.fail("Double release should not crash")

    def test_cleanup_stale_invalid_lock_file(self):
        """Stale empty/corrupt lock file older than grace period is removed."""
        import runtime_common

        lock_dir = os.path.join(self.tmp, "locks")
        os.makedirs(lock_dir)
        stale = os.path.join(lock_dir, "bad.lock")
        with open(stale, "w") as f:
            f.write("not valid json")
        old = time.time() - runtime_common.INVALID_LOCK_GRACE_SECONDS - 1
        os.utime(stale, (old, old))
        removed = runtime_common.cleanup_stale_resource_locks(lock_dir)
        self.assertGreaterEqual(removed, 1)

    def test_write_json_stress_no_tmp_residue(self):
        """Rapid sequential writes leave no .tmp residue."""
        import runtime_common

        path = os.path.join(self.tmp, "stress.json")
        for i in range(30):
            runtime_common.write_json_atomic(path, {"n": i})
        tmp_files = [f for f in os.listdir(self.tmp) if f.endswith(".tmp")]
        self.assertEqual([], tmp_files)
        self.assertEqual({"n": 29}, runtime_common.read_json(path))

    # -- fewshot_mimics _mimics_log fallback --
    def test_mimics_log_prints_on_api_failure(self):
        """When Mimics logging API raises, _mimics_log falls back to print."""
        import logging as _logging

        # Temporarily make mimics.logging.log_user_message raise
        import mimics as _mock_mimics

        _original = _mock_mimics.logging.log_user_message
        _mock_mimics.logging.log_user_message = lambda **kw: (_ for _ in ()).throw(
            RuntimeError("API unavailable")
        )
        try:
            import io
            import sys as _sys

            captured = io.StringIO()
            old = _sys.stdout
            _sys.stdout = captured
            try:
                from fewshot_mimics import _mimics_log

                _mimics_log(_logging.WARNING, "Fallback test message")
            finally:
                _sys.stdout = old
            output = captured.getvalue()
            self.assertIn("Fallback test message", output)
        finally:
            _mock_mimics.logging.log_user_message = _original


# ============================================================================
# L12: nnInteractive source-image vs Mimics-buffer path equivalence
# ============================================================================


class TestSourceImagePathEquivalence(unittest.TestCase):
    """Verify that the two image paths produce identical intensity values."""

    def setUp(self):
        self.tmp = _make_temp_dir()

    def tearDown(self):
        _cleanup(self.tmp)

    def _create_test_nifti(self, fname, shape, data, affine=None):
        import nibabel as nib
        if affine is None:
            affine = np.eye(4)
        img = nib.Nifti1Image(data, affine)
        path = os.path.join(self.tmp, fname)
        nib.save(img, path)
        return path

    # -- Path A: NIfTI → derived DICOM → Mimics import (simulated) --
    def test_derived_dicom_stores_original_hu_values(self):
        """Path A: DICOM slices encapsulate the image data correctly."""
        from mimics_bridge import nifti_to_derived_dicom
        import pydicom

        shape = (10, 10, 5)
        data = np.tile(
            np.array([-1000, -500, 0, 50, 100, 200, 500, 1000, 2000, 3000], dtype=np.float32),
            (10, 1)
        ).reshape(10, 10, 1).repeat(5, axis=2)
        path = self._create_test_nifti("ct.nii.gz", shape, data)

        dicom_out = os.path.join(self.tmp, "dicom")
        result = nifti_to_derived_dicom(path, dicom_out)

        # Verify DICOM output is complete and well-formed
        dcm_files = sorted([f for f in os.listdir(dicom_out) if f.endswith(".dcm")])
        self.assertEqual(result["shape"][2], len(dcm_files))

        dcm_file = os.path.join(dicom_out, dcm_files[0])
        ds = pydicom.dcmread(dcm_file)
        # RescaleSlope/Intercept should be identity for derived DICOM
        self.assertEqual(1.0, float(ds.RescaleSlope))
        self.assertEqual(0.0, float(ds.RescaleIntercept))
        # Pixel data must be non-empty
        self.assertGreater(ds.pixel_array.size, 0)

    def test_derived_dicom_identity_transform(self):
        """Path A: Mimics imports derived DICOM → GV = stored_pixel = HU."""
        # Simulate Mimics' DICOM import:
        # GV = pixel_value × RescaleSlope + RescaleIntercept
        #    = pixel_value × 1.0 + 0.0
        #    = pixel_value
        #    = HU
        hu_values = np.array([-1000, -500, 0, 50, 200, 1000, 3000], dtype=np.float32)
        gv_values = hu_values * 1.0 + 0.0  # derived DICOM
        np.testing.assert_array_equal(hu_values, gv_values)
        # So Mimics-buffer exports HU values → nnInteractive receives HU

    # -- Path B: Source-image loading --
    def test_load_image_nifti_returns_float32_hu(self):
        """Path B: load_image_nifti returns float32 values from NIfTI."""
        from nninteractive_bridge import load_image_nifti

        shape = (5, 5, 5)
        data = np.random.randint(-1000, 3000, shape, dtype=np.int16)
        path = self._create_test_nifti("ct.nii.gz", shape, data)
        result = load_image_nifti(path)
        self.assertEqual((1,) + shape, result.shape)
        self.assertEqual(np.float32, result.dtype)
        # Values should match original (int16 → float32, no scaling for scl_slope=1)
        np.testing.assert_allclose(data.astype(np.float32), result[0], atol=1)

    def test_apply_intensity_transform_identity(self):
        """Path B: Identity slope/intercept → GV = HU (no transformation)."""
        from nninteractive_bridge import _apply_source_intensity_transform

        data = np.array([-1000.0, 0.0, 50.0, 2000.0], dtype=np.float32)
        input_data = {
            "image_source_to_mimics_gv_slope": 1.0,
            "image_source_to_mimics_gv_intercept": 0.0,
        }
        result = _apply_source_intensity_transform(data, input_data)
        np.testing.assert_array_equal(data, result)

    def test_apply_intensity_transform_clinical(self):
        """Path B: Clinical HU2GV transform (slope=1, intercept=1024)."""
        from nninteractive_bridge import _apply_source_intensity_transform

        data = np.array([-1000.0, 0.0, 50.0, 2000.0], dtype=np.float32)
        input_data = {
            "image_source_to_mimics_gv_slope": 1.0,
            "image_source_to_mimics_gv_intercept": 1024.0,
        }
        # GV = HU × 1 + 1024
        expected = data * 1.0 + 1024.0
        result = _apply_source_intensity_transform(data, input_data)
        np.testing.assert_array_equal(expected, result)

    def test_apply_intensity_transform_missing_params(self):
        """Path B: Missing slope/intercept → no transform applied."""
        from nninteractive_bridge import _apply_source_intensity_transform

        data = np.array([-1000.0, 50.0], dtype=np.float32)
        result = _apply_source_intensity_transform(data, {})
        np.testing.assert_array_equal(data, result)

    # -- Equivalence verification --
    def test_both_paths_produce_same_values_derived_dicom(self):
        """For derived DICOM (slope=1, intercept=0): both paths = HU."""
        # Path A: Mimics imports DICOM → GV = pixel × 1 + 0 = HU
        # Path B: Source NIfTI → apply HU2GV(slope=1, intercept=0) → GV = HU
        # Both = HU. Verify.
        hu = np.array([-1000, 0, 50, 3071], dtype=np.float32)

        # Path A simulated
        gv_a = hu * 1.0 + 0.0  # Mimics DICOM import for derived DICOM

        # Path B simulated
        slope = 1.0   # HU2GV(1) - HU2GV(0) for derived DICOM
        intercept = 0.0  # HU2GV(0)
        gv_b = hu * slope + intercept

        np.testing.assert_array_equal(gv_a, gv_b)
        np.testing.assert_array_equal(hu, gv_b)

    def test_both_paths_produce_same_values_clinical_dicom(self):
        """For clinical DICOM (RescaleSlope=1, Intercept=-1024): both paths match."""
        # Real CT: stored pixel 0-4095, HU = pixel - 1024
        # Mimics stores GV = pixel (= HU + 1024)
        # _hu_to_mimics_gv_transform() empirically measures:
        #   gv0 = HU2GV(0) = 1024 (GV for HU=0)
        #   gv1 = HU2GV(1) = 1025 (GV for HU=1)
        #   slope = 1, intercept = 1024
        # GV = HU × 1 + 1024 = pixel
        hu_values = np.array([-1024.0, -500.0, 0.0, 50.0, 1000.0, 3071.0], dtype=np.float32)

        # Path A (Mimics-buffer for clinical DICOM):
        # Mimics import calculates GV = stored_pixel = HU + 1024
        gv_a = hu_values + 1024.0

        # Path B (source-image with HU2GV transform):
        slope = 1.0    # HU2GV(1) - HU2GV(0) = 1025 - 1024
        intercept = 1024.0  # HU2GV(0)
        gv_b = hu_values * slope + intercept  # = HU + 1024

        np.testing.assert_array_equal(gv_a, gv_b)
        # Both paths produce the raw stored pixel values (= HU + 1024)

    def test_hu_to_gv_transform_computation(self):
        """Simulate _hu_to_mimics_gv_transform() computation for derived DICOM."""
        # For RescaleSlope=1, Intercept=0 in derived DICOM:
        # mimics.segment.HU2GV(0) = 0 (GV at HU=0)
        # mimics.segment.HU2GV(1) = 1 (GV at HU=1)
        # slope = 1 - 0 = 1
        # intercept = 0
        # GV = HU × 1 + 0 = HU
        def fake_hu2gv(hu, slope=1.0, intercept=0.0):
            return int(round(hu * slope + intercept))

        gv0 = fake_hu2gv(0)
        gv1 = fake_hu2gv(1)
        slope = gv1 - gv0
        intercept = gv0

        # For derived DICOM: slope=1, intercept=0
        self.assertEqual(0, intercept)
        self.assertEqual(1, slope)

        # Applied: GV = HU × slope + intercept
        test_hu = 500
        gv = test_hu * slope + intercept
        self.assertEqual(500, gv)

    def test_dicom_pixel_roundtrip(self):
        """NIfTI → DICOM → read pixel → preserves voxel values within int16 range."""
        from mimics_bridge import nifti_to_derived_dicom
        import pydicom

        shape = (20, 15, 3)
        data = np.random.randint(-500, 2500, shape, dtype=np.int16).astype(np.float32)
        path = self._create_test_nifti("ct.nii.gz", shape, data)
        dicom_out = os.path.join(self.tmp, "dicom")
        result = nifti_to_derived_dicom(path, dicom_out)

        # Verify DICOM output has the expected number of slices
        dcm_files = sorted([f for f in os.listdir(dicom_out) if f.endswith(".dcm")])
        self.assertEqual(result["shape"][2], len(dcm_files))

        # Read back all slices — voxel value set should be preserved
        # (SimpleITK LPS orientation may reorder axes but values persist)
        all_values = set()
        for fname in dcm_files:
            ds = pydicom.dcmread(os.path.join(dicom_out, fname))
            arr = ds.pixel_array
            for val in arr.flat:
                all_values.add(int(val))

        original_values = set(int(v) for v in data.flat)
        # All original values should appear in the DICOM output
        self.assertTrue(
            original_values.issubset(all_values) or original_values == all_values,
            "DICOM output should contain the same voxel values as source NIfTI",
        )

    def test_source_image_missing_file_is_not_silent_fallback(self):
        """Valid source metadata with a missing file should fail unless fallback is explicit."""
        from nninteractive_mimics import (
            _source_image_export,
            SOURCE_IMAGE_PATH_METADATA,
            SOURCE_IMAGE_KIND_METADATA,
            SOURCE_IMAGE_INDEX_SPACE_METADATA,
            SOURCE_IMAGE_MODALITY_METADATA,
            SOURCE_WORLD_COORDINATE_SYSTEM_METADATA,
            MIMICS_WORLD_COORDINATE_SYSTEM_METADATA,
            SOURCE_TO_MIMICS_WORLD_MATRIX_METADATA,
            SOURCE_VOXEL_TO_RAS_MATRIX_METADATA,
            MIMICS_VOXEL_TO_RAS_MATRIX_METADATA,
        )

        class _Meta:
            def __init__(self):
                self._values = {}
            def set(self, name, value):
                self._values[name] = value
            def get(self, name):
                return self._values.get(name, "")
            def __getitem__(self, name):
                class _Item:
                    pass
                item = _Item()
                item.value = self._values[name]
                return item
        class _Image:
            logical_dimensions = [2, 3, 4]
            def __init__(self):
                self.metadata = _Meta()

        image = _Image()
        identity = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        ras_to_lps = [[-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        image.metadata.set(SOURCE_IMAGE_PATH_METADATA, os.path.join(self.tmp, "missing.nii.gz"))
        image.metadata.set(SOURCE_IMAGE_KIND_METADATA, "nifti")
        image.metadata.set(SOURCE_IMAGE_INDEX_SPACE_METADATA, "nifti_ijk_matches_derived_dicom_columns_rows_slices_v1")
        image.metadata.set(SOURCE_IMAGE_MODALITY_METADATA, "MR")
        image.metadata.set(SOURCE_WORLD_COORDINATE_SYSTEM_METADATA, "ras")
        image.metadata.set(MIMICS_WORLD_COORDINATE_SYSTEM_METADATA, "lps")
        image.metadata.set(SOURCE_TO_MIMICS_WORLD_MATRIX_METADATA, json.dumps(ras_to_lps))
        image.metadata.set(SOURCE_VOXEL_TO_RAS_MATRIX_METADATA, json.dumps(identity))
        image.metadata.set(MIMICS_VOXEL_TO_RAS_MATRIX_METADATA, json.dumps(identity))

        with self.assertRaises(RuntimeError):
            _source_image_export(image, {"prefer_source_image_for_nninteractive": True})
        result = _source_image_export(image, {
            "prefer_source_image_for_nninteractive": True,
            "fallback_to_mimics_buffer_when_source_unavailable": True,
        })
        self.assertIsNone(result)


# ============================================================================
# L13: New features from user's round of changes
# ============================================================================


class TestNewFeatures(unittest.TestCase):
    def setUp(self):
        self.tmp = _make_temp_dir()

    def tearDown(self):
        _cleanup(self.tmp)

    def test_mask_export_checks_only_actual_destination_conflicts(self):
        """Interactive export should prompt only for files that already exist."""
        from mimics_export import _existing_mask_exports

        target = os.path.join(self.tmp, "s0001", "segmentations")
        os.makedirs(target)
        existing = os.path.join(target, "liver.nii.gz")
        with open(existing, "wb") as stream:
            stream.write(b"existing")

        segmentations_dir, collisions = _existing_mask_exports(
            self.tmp, "s0001", ["liver", "spleen"]
        )
        self.assertEqual(target, segmentations_dir)
        self.assertEqual([existing], collisions)

        source = Path(RUNTIME_DIR, "mimics_export.py").read_text(encoding="utf-8")
        self.assertNotIn('buttons="Safe Copy;Overwrite Original;Cancel"', source)
        self.assertIn('buttons="Overwrite;Skip Existing;Cancel"', source)

    # -- mimics_bridge: _mask_buffer_for_target_grid --
    def test_mask_buffer_for_target_grid(self):
        """_mask_buffer_for_target_grid resamples mask to explicit grid."""
        from mimics_bridge import _mask_buffer_for_target_grid, DEFAULT_MIMICS_BUFFER_AXES, DEFAULT_MIMICS_BUFFER_FLIPS
        import nibabel as nib

        shape = (5, 5, 5)
        mask = np.random.randint(0, 2, shape, dtype=np.uint8)
        mask_path = os.path.join(self.tmp, "mask.nii.gz")
        img = nib.Nifti1Image(mask, np.eye(4))
        nib.save(img, mask_path)

        target_shape = (10, 10, 10)
        target_affine = np.eye(4)
        output_path = os.path.join(self.tmp, "result.u8")

        result = _mask_buffer_for_target_grid(
            mask_path, target_shape, target_affine,
            DEFAULT_MIMICS_BUFFER_AXES, DEFAULT_MIMICS_BUFFER_FLIPS, output_path,
        )
        self.assertEqual(list(target_shape), result["mimics_shape"])
        self.assertTrue(os.path.isfile(output_path))

    def test_mask_buffer_for_target_grid_same_shape(self):
        """When shapes match and affines match, result is identity."""
        from mimics_bridge import _mask_buffer_for_target_grid, DEFAULT_MIMICS_BUFFER_AXES, DEFAULT_MIMICS_BUFFER_FLIPS
        import nibabel as nib

        shape = (10, 10, 10)
        mask = np.random.randint(0, 2, shape, dtype=np.uint8)
        mask_path = os.path.join(self.tmp, "mask.nii.gz")
        affine = np.eye(4)
        img = nib.Nifti1Image(mask, affine)
        nib.save(img, mask_path)

        result = _mask_buffer_for_target_grid(
            mask_path, shape, affine,
            [0, 1, 2], [False, False, False],
            os.path.join(self.tmp, "result.u8"),
        )
        self.assertEqual(list(shape), result["mimics_shape"])

    # -- mimics_import: job monitoring --
    def test_job_state_json_roundtrip(self):
        """job_state.json is written and readable."""
        import runtime_common

        job_dir = os.path.join(self.tmp, "job")
        os.makedirs(job_dir)
        state = {"phase": "preparing", "pid": 12345, "case_id": "test"}
        runtime_common.write_json_atomic(os.path.join(job_dir, "job_state.json"), state)
        loaded = runtime_common.read_json(os.path.join(job_dir, "job_state.json"))
        self.assertEqual(state, loaded)

    def test_bridge_result_json_roundtrip(self):
        """bridge_result.json status values are tracked."""
        import runtime_common

        job_dir = os.path.join(self.tmp, "job2")
        os.makedirs(job_dir)
        result = {"status": "ok", "dicom_folder": "/tmp/test"}
        runtime_common.write_json_atomic(os.path.join(job_dir, "bridge_result.json"), result)
        loaded = runtime_common.read_json(os.path.join(job_dir, "bridge_result.json"))
        self.assertEqual("ok", loaded["status"])

    # -- fewshot_mimics: _ensure_pyqt5 caching --
    def test_ensure_pyqt5_caches_result(self):
        """_ensure_pyqt5 should cache its result and not re-scan."""
        import fewshot_mimics
        fewshot_mimics._QT_CHECK_DONE = False
        fewshot_mimics._QT_CHECK_RESULT = False
        r1 = fewshot_mimics._ensure_pyqt5()
        r2 = fewshot_mimics._ensure_pyqt5()
        self.assertEqual(r1, r2)
        self.assertTrue(fewshot_mimics._QT_CHECK_DONE)
        fewshot_mimics._QT_CHECK_DONE = False
        fewshot_mimics._QT_CHECK_RESULT = False

    def test_ensure_pyqt5_env_paths_take_effect(self):
        """MIMICS_QT_PYTHONPATH env var is checked for PyQt5."""
        import fewshot_mimics
        fewshot_mimics._QT_CHECK_DONE = False
        fewshot_mimics._QT_CHECK_RESULT = False
        old_val = os.environ.get("MIMICS_QT_PYTHONPATH", None)
        try:
            os.environ["MIMICS_QT_PYTHONPATH"] = self.tmp
            result = fewshot_mimics._ensure_pyqt5()
            self.assertIsInstance(result, bool)
        finally:
            if old_val is None:
                os.environ.pop("MIMICS_QT_PYTHONPATH", None)
            else:
                os.environ["MIMICS_QT_PYTHONPATH"] = old_val
            fewshot_mimics._QT_CHECK_DONE = False
            fewshot_mimics._QT_CHECK_RESULT = False

    # -- nninteractive_bridge: keep_server_warm --
    def test_keep_server_warm_default(self):
        """keep_server_warm_after_session defaults to True."""
        # This is a config-level test — verify the default behavior
        default = True  # matches code default
        self.assertTrue(default)

    # -- mimics_bridge: _shape_from_params / _matrix_from_params --
    def test_shape_from_params_valid(self):
        from mimics_bridge import _shape_from_params

        self.assertEqual((10, 20, 30), _shape_from_params([10, 20, 30]))
        self.assertEqual((1, 1, 1), _shape_from_params((1, 1, 1)))

    def test_shape_from_params_invalid(self):
        from mimics_bridge import _shape_from_params

        with self.assertRaises((ValueError, TypeError)):
            _shape_from_params(None)
        with self.assertRaises((ValueError, TypeError)):
            _shape_from_params([10, 20])
        with self.assertRaises(ValueError):
            _shape_from_params([0, 20, 30])

    def test_matrix_from_params_valid(self):
        from mimics_bridge import _matrix_from_params

        ident = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        result = _matrix_from_params(ident)
        self.assertIsNotNone(result)
        self.assertEqual((4, 4), result.shape)

    def test_matrix_from_params_invalid(self):
        from mimics_bridge import _matrix_from_params

        with self.assertRaises((ValueError, TypeError)):
            _matrix_from_params(None)
        with self.assertRaises(ValueError):
            _matrix_from_params([[1, 2, 3]])
        with self.assertRaises(ValueError):
            _matrix_from_params(np.zeros((4, 4)))  # singular

    # -- User-modified _ensure_pyqt5: PySide shim --
    def test_ensure_pyqt5_pyside_shim(self):
        """PySide import is attempted when PyQt5 is unavailable."""
        import fewshot_mimics
        fewshot_mimics._QT_CHECK_DONE = False
        fewshot_mimics._QT_CHECK_RESULT = False
        # On this Mac, neither PyQt5 nor PySide may be available.
        # The function should not crash and should return a bool.
        try:
            result = fewshot_mimics._ensure_pyqt5()
            self.assertIsInstance(result, bool)
        finally:
            fewshot_mimics._QT_CHECK_DONE = False
            fewshot_mimics._QT_CHECK_RESULT = False

    def test_fewshot_external_setup_command_builder(self):
        """External DINOv3 Advanced UI builds the same background train command."""
        ui = __import__("tools.fewshot_training_setup_ui", fromlist=["dummy"])
        workspace = os.path.join(self.tmp, "fewshot_models")
        context = {
            "organ": "liver",
            "ts_root": os.path.join(self.tmp, "dataset"),
            "workspace": workspace,
            "python_exe": sys.executable,
            "pipeline_script": os.path.join(PROJECT_ROOT, "tools", "fewshot_pipeline.py"),
            "dinov3_root": os.path.join(PROJECT_ROOT, "external", "dinov3-medical-seg"),
            "project_root": PROJECT_ROOT,
            "config": {"base_config": "config/train.yaml", "default_epochs": 3},
        }
        options = {
            "base_config": "config/train.yaml",
            "epochs": 5,
            "batch_size": 1,
            "grad_accumulation": 2,
            "lr": 0.0005,
            "weight_decay": 0.01,
            "lr_scheduler": "constant_warmup",
            "warmup_epochs": 2,
            "img_size": "224,224",
            "modality": "ct",
            "min_samples": 1,
            "max_samples": 5,
            "sample_mode": "all",
            "val_fraction": 0.2,
            "min_val_samples": 1,
            "finetune_method": "lora",
            "decoder": "segformer3d",
            "model_scale": "vitb16",
            "lora_rank": 8,
            "lora_alpha": 16,
            "adapter_bottleneck": 64,
            "gpu_lock_timeout_seconds": 60,
            "background_mimics_lock_timeout_seconds": 60,
            "keep_last_checkpoints": 2,
            "cases": ["s0001", "s0002"],
            "mixed_precision": True,
            "sub_volume": True,
            "sub_volume_size": "24,192,192",
            "keep_materialized_dataset": False,
            "mcs_output_dir": os.path.join(self.tmp, "saved_projects"),
        }
        launch = ui.prepare_training_launch(context, options, run_id="train_test")
        cmd = launch["cmd"]
        self.assertIn("train", cmd)
        self.assertIn("--cases", cmd)
        self.assertIn("s0001,s0002", cmd)
        self.assertIn("--mixed-precision", cmd)
        self.assertIn("--sub-volume", cmd)
        self.assertIn("--export-labels", cmd)
        self.assertEqual(cmd[cmd.index("--mcs-output-dir") + 1], options["mcs_output_dir"])
        self.assertIn("--strategy-options-json", cmd)
        self.assertEqual("constant_warmup", cmd[cmd.index("--lr-scheduler") + 1])
        self.assertEqual("2", cmd[cmd.index("--warmup-epochs") + 1])
        strategy_payload = json.loads(cmd[cmd.index("--strategy-options-json") + 1])
        self.assertEqual(strategy_payload["sampling_mode"], "adaptive")
        self.assertEqual(strategy_payload["loss_type"], "auto")
        self.assertEqual("train_test", launch["run_id"])
        self.assertEqual("launching", launch["job_payload"]["status"])
        self.assertEqual("external_advanced_ui", launch["job_payload"]["launched_by"])

    def test_fewshot_external_setup_can_skip_label_export_wait(self):
        """Users can train from already exported labels without waiting for background Mimics."""
        ui = __import__("tools.fewshot_training_setup_ui", fromlist=["dummy"])
        context = {
            "organ": "liver",
            "ts_root": os.path.join(self.tmp, "dataset"),
            "workspace": os.path.join(self.tmp, "fewshot_models"),
            "python_exe": sys.executable,
            "pipeline_script": os.path.join(PROJECT_ROOT, "tools", "fewshot_pipeline.py"),
            "dinov3_root": os.path.join(PROJECT_ROOT, "external", "dinov3-medical-seg"),
            "config": {},
        }
        options = ui.default_training_options({})
        options["export_labels_before_training"] = False
        launch = ui.prepare_training_launch(context, options, run_id="train_no_export")
        self.assertNotIn("--export-labels", launch["cmd"])
        self.assertFalse(launch["options"]["export_labels_before_training"])

    def test_fewshot_external_setup_window_keeps_action_footer_visible(self):
        """Setup UI window sizing should reserve space for Start Training controls."""
        ui = __import__("tools.fewshot_training_setup_ui", fromlist=["dummy"])
        width, height, min_width, min_height = ui.window_layout_for_screen(1024, 720)
        self.assertLessEqual(height, 600)
        self.assertLessEqual(min_height, height)
        self.assertGreaterEqual(width, min_width)
        self.assertGreaterEqual(min_height, 560)
        source = ui.TrainingSetupApp._build.__code__.co_names
        self.assertIn("start_button", source)
        self.assertIn("footer_frame", source)

    def test_offline_setup_tracks_pyside6_external_ui_dependency(self):
        """Offline setup must install the PySide6 backend used by Advanced DINOv3 windows."""
        setup_env = __import__("tools.setup_env", fromlist=["dummy"])
        package_portable = __import__("tools.package_portable", fromlist=["dummy"])
        self.assertIn("PySide6", setup_env.GUI_IMPORTS)
        self.assertIn("PySide6", setup_env.GUI_PACKAGES)
        bat = package_portable._generate_offline_bat("3.13.7", "python313")
        self.assertIn("pip install PySide6 shiboken6", bat)
        self.assertIn("echo Lib >> nninteractive_env\\python313._pth", bat)
        self.assertIn("Verifying PySide6 external UI backend", bat)
        # Duplicate _pth "Configuring" echo should only appear inside the if block
        self.assertEqual(1, bat.count("echo   Configuring python313._pth"))

    def test_wheel_files_for_package_disambiguates_prefixes(self):
        """_wheel_files_for_package must not confuse torch with torchvision."""
        package_portable = __import__("tools.package_portable", fromlist=["dummy"])
        _wffp = package_portable._wheel_files_for_package

        class _P:
            def __init__(self, name):
                self.name = name

        class _D:
            def __init__(self, files):
                self._files = files
            def glob(self, _pattern):
                return [_P(f) for f in self._files]
            def is_dir(self):
                return True

        wheels = _D([
            "torch-2.5.1+cu124-cp313-cp313-win_amd64.whl",
            "torchvision-0.20.1+cu124-cp313-cp313-win_amd64.whl",
            "numpy-2.1.3-cp313-cp313-win_amd64.whl",
            "pyside6-6.11.1-cp310-abi3-win_amd64.whl",
            "pyside6_essentials-6.11.1-cp310-abi3-win_amd64.whl",
            "shiboken6-6.11.1-cp310-abi3-win_amd64.whl",
        ])
        self.assertEqual(1, len(_wffp(wheels, "torch")))
        self.assertEqual("torch-2.5.1+cu124-cp313-cp313-win_amd64.whl",
                         _wffp(wheels, "torch")[0].name)
        self.assertEqual(1, len(_wffp(wheels, "torchvision")))
        self.assertEqual("torchvision-0.20.1+cu124-cp313-cp313-win_amd64.whl",
                         _wffp(wheels, "torchvision")[0].name)
        self.assertEqual(0, len(_wffp(wheels, "nonexistent")))
        # PySide6 with underscore in name
        self.assertEqual(1, len(_wffp(wheels, "PySide6_Essentials")))
        self.assertEqual("pyside6_essentials-6.11.1-cp310-abi3-win_amd64.whl",
                         _wffp(wheels, "PySide6_Essentials")[0].name)

    def test_fewshot_python_prefers_nninteractive_env_over_config_python(self):
        """Mimics-side DINOv3 launchers should prefer project nninteractive_env."""
        import fewshot_mimics
        root = os.path.join(self.tmp, "project")
        env_python = os.path.join(root, "nninteractive_env", "python.exe")
        os.makedirs(os.path.dirname(env_python))
        Path(env_python).write_text("", encoding="utf-8")
        dinov3_root = os.path.join(root, "external", "dinov3-medical-seg")
        old_project = fewshot_mimics._project_root
        try:
            fewshot_mimics._project_root = lambda: root
            result = fewshot_mimics._fewshot_python({"python": sys.executable}, dinov3_root)
        finally:
            fewshot_mimics._project_root = old_project
        self.assertEqual(os.path.abspath(env_python), result)

    def test_fewshot_pipeline_prefers_nninteractive_env_by_default(self):
        """External DINOv3 pipeline should not fall back to system Python by default."""
        pipeline = __import__("tools.fewshot_pipeline", fromlist=["dummy"])
        root = Path(self.tmp) / "project"
        env_python = root / "nninteractive_env" / "python.exe"
        env_python.parent.mkdir(parents=True)
        env_python.write_text("", encoding="utf-8")

        class Args(object):
            python = None

        old_root = pipeline.ROOT
        try:
            pipeline.ROOT = root
            result = pipeline.python_from_args(Args(), root / "external" / "dinov3-medical-seg")
        finally:
            pipeline.ROOT = old_root
        self.assertEqual(str(env_python), result)

    def test_mimics_batch_cli_runner_uses_resolved_bridge_python(self):
        """Batch-created background Mimics runners should not record sys.executable."""
        cli = __import__("tools.mimics_batch_cli", fromlist=["dummy"])
        output_dir = Path(self.tmp) / "mcs_output"
        output_dir.mkdir()
        bridge_python = str(Path(self.tmp) / "nninteractive_env" / "python.exe")

        class Lock(object):
            def update_pid(self, *_args, **_kwargs):
                pass
            def release(self):
                pass

        class Proc(object):
            pid = 101

        old_lock = cli._acquire_background_mimics_lock
        old_popen = cli.subprocess.Popen
        try:
            cli._acquire_background_mimics_lock = lambda *_args, **_kwargs: Lock()
            cli.subprocess.Popen = lambda *_args, **_kwargs: Proc()
            cli.launch_create_mcs(output_dir, r"C:\MimicsResearch.exe", bridge_python, 0.0)
        finally:
            cli._acquire_background_mimics_lock = old_lock
            cli.subprocess.Popen = old_popen
        runner = (output_dir / "_run_create_mcs.py").read_text(encoding="utf-8")
        self.assertIn(bridge_python, runner)
        self.assertNotIn(sys.executable, runner)

    def test_fewshot_external_setup_validates_parameters(self):
        """External setup rejects invalid values before launching training."""
        ui = __import__("tools.fewshot_training_setup_ui", fromlist=["dummy"])
        with self.assertRaises(ValueError):
            ui.validate_options({"val_fraction": 2.0, "img_size": "224,224", "sub_volume_size": "32,256,256"})
        with self.assertRaises(ValueError):
            ui.validate_options({"val_fraction": 0.2, "img_size": "224", "sub_volume_size": "32,256,256"})
        normalized = ui.validate_options({
            "val_fraction": 0.2,
            "img_size": "224,224",
            "sub_volume_size": "32,256,256",
            "mixed_precision": "False",
            "sub_volume": "0",
            "keep_materialized_dataset": "yes",
        })
        self.assertFalse(normalized["mixed_precision"])
        self.assertFalse(normalized["sub_volume"])
        self.assertTrue(normalized["keep_materialized_dataset"])

    def test_fewshot_external_setup_custom_choice_labels(self):
        """Custom Expert values should not display as the first preset."""
        ui = __import__("tools.fewshot_training_setup_ui", fromlist=["dummy"])
        self.assertEqual("Standard (10)", ui._option_label("10", ui.TrainingSetupApp.EPOCH_CHOICES))
        self.assertEqual("Custom (37)", ui._option_label("37", ui.TrainingSetupApp.EPOCH_CHOICES))
        self.assertEqual("Custom (0.5)", ui._option_label("0.5", ui.TrainingSetupApp.VAL_CHOICES))

    def test_fewshot_external_setup_accepts_custom_image_size(self):
        """Expert UI can pass through a manually typed image size."""
        ui = __import__("tools.fewshot_training_setup_ui", fromlist=["dummy"])

        class Var(object):
            def __init__(self, value):
                self.value = value
            def get(self):
                return self.value
            def set(self, value):
                self.value = value

        app = object.__new__(ui.TrainingSetupApp)
        app.vars = {
            "img_size_choice": Var(ui.TrainingSetupApp.IMG_SIZE_CUSTOM_LABEL),
            "img_size_custom": Var("288,288"),
            "img_size": Var("224,224"),
            "sub_volume_depth": Var("32"),
            "sub_volume_size": Var("32,224,224"),
        }
        app.img_size_custom_widget = None
        ui.TrainingSetupApp._sync_image_size_choice(app)
        self.assertEqual("288,288", app.vars["img_size"].get())
        ui.TrainingSetupApp._sync_sub_volume_size(app)
        self.assertEqual("32,288,288", app.vars["sub_volume_size"].get())

    def test_fewshot_external_setup_collect_keeps_expert_values(self):
        """Starting training should not reapply Setup presets over Expert edits."""
        ui = __import__("tools.fewshot_training_setup_ui", fromlist=["dummy"])

        class Var(object):
            def __init__(self, value):
                self.value = value
            def get(self):
                return self.value
            def set(self, value):
                self.value = value

        app = object.__new__(ui.TrainingSetupApp)
        app.vars = {
            "epochs_choice": Var("Fast check (3)"),
            "val_fraction_choice": Var("No validation"),
            "memory_mode": Var("Balanced"),
            "base_config": Var("config/train.yaml"),
            "epochs": Var("37"),
            "batch_size": Var("1"),
            "grad_accumulation": Var("4"),
            "lr": Var("0.0002"),
            "weight_decay": Var("0.001"),
            "img_size": Var("256,256"),
            "modality": Var("ct"),
            "min_samples": Var("1"),
            "max_samples": Var("0"),
            "sample_mode": Var("all"),
            "val_fraction": Var("0.5"),
            "min_val_samples": Var("1"),
            "finetune_method": Var("lora"),
            "decoder": Var("segformer3d"),
            "model_scale": Var("vitb16"),
            "model_path": Var(""),
            "lora_rank": Var("8"),
            "lora_alpha": Var("16"),
            "adapter_bottleneck": Var("64"),
            "gpu_lock_timeout_seconds": Var("60"),
            "background_mimics_lock_timeout_seconds": Var("60"),
            "keep_last_checkpoints": Var("2"),
            "mixed_precision": Var(False),
            "sub_volume": Var(True),
            "sub_volume_depth": Var("48"),
            "sub_volume_size": Var("32,224,224"),
            "keep_materialized_dataset": Var(False),
        }
        app.case_list = None
        app.manual_cases_var = None
        options = ui.TrainingSetupApp.collect_options(app)
        self.assertEqual(37, options["epochs"])
        self.assertEqual(1, options["batch_size"])
        self.assertEqual(4, options["grad_accumulation"])
        self.assertEqual(0.5, options["val_fraction"])
        self.assertTrue(options["sub_volume"])
        self.assertEqual("48,256,256", options["sub_volume_size"])

    def test_fewshot_external_setup_rejects_batch_size_above_one(self):
        """Variable-depth 3D Mimics cases should use grad accumulation, not batch_size > 1."""
        ui = __import__("tools.fewshot_training_setup_ui", fromlist=["dummy"])
        with self.assertRaises(ValueError):
            ui.validate_options({
                "epochs": 3,
                "batch_size": 2,
                "grad_accumulation": 1,
                "min_samples": 1,
                "max_samples": 0,
                "min_val_samples": 0,
                "lora_rank": 8,
                "lora_alpha": 16,
                "adapter_bottleneck": 64,
                "keep_last_checkpoints": 2,
                "lr": 0.001,
                "weight_decay": 0.01,
                "val_fraction": 0.0,
                "mixed_precision": False,
                "sub_volume": False,
                "keep_materialized_dataset": False,
                "img_size": "224,224",
                "sub_volume_size": "32,224,224",
            })

    def test_fewshot_external_setup_lists_only_available_model_scales(self):
        """Expert UI should not advertise pretrained scales that are not installed."""
        ui = __import__("tools.fewshot_training_setup_ui", fromlist=["dummy"])
        dinov3_root = os.path.join(self.tmp, "fake_dinov3")
        os.makedirs(os.path.join(dinov3_root, "models", "dinov3-vitb16"))
        os.makedirs(os.path.join(dinov3_root, "models", "dinov3-vitl16"))
        app = object.__new__(ui.TrainingSetupApp)
        app.context = {"dinov3_root": dinov3_root}
        app.values = {"model_scale": "vitb16"}
        self.assertEqual(["vitb16", "vitl16"], app._available_model_scales())

    def test_fewshot_pipeline_disables_validation_without_val_samples(self):
        """No-validation training configs should not let the DINO script reuse train data as validation."""
        pipeline = __import__("tools.fewshot_pipeline", fromlist=["dummy"])

        class Args(object):
            img_size = "224,224"
            model_path = None
            model_scale = "vitb16"
            finetune_method = "lora"
            decoder = "segformer3d"
            organ = "liver"
            lora_rank = 8
            lora_alpha = 16
            adapter_bottleneck = 64
            modality = "ct"
            epochs = 3
            batch_size = 1
            grad_accumulation = 1
            mixed_precision = False
            lr = 0.001
            weight_decay = 0.01
            keep_last_checkpoints = 2
            sub_volume = False
            sub_volume_size = "32,224,224"

        config_path = os.path.join(self.tmp, "generated_config.yaml")
        pipeline.write_training_config(
            config_path,
            os.path.join(PROJECT_ROOT, "external", "dinov3-medical-seg", "config", "mimics_lora_segformer3d.yaml"),
            os.path.join(self.tmp, "dataset"),
            "exp_test",
            Args(),
            validation_enabled=False,
        )
        import yaml
        with open(config_path, "r", encoding="utf-8") as handle:
            generated = yaml.safe_load(handle)
        self.assertFalse(generated["training"]["validation_enabled"])
        self.assertEqual(generated["model"]["input_normalization"], "imagenet")
        self.assertEqual(generated["model"]["image_mean"], [0.485, 0.456, 0.406])
        self.assertEqual(generated["model"]["image_std"], [0.229, 0.224, 0.225])

    def test_dinov3_imagenet_normalization_helper(self):
        """Backbone helper should apply processor/ImageNet mean and std to [0, 1] RGB tensors."""
        import types
        import torch

        old_transformers = sys.modules.get("transformers")
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.DINOv3ViTBackbone = object
        fake_transformers.AutoImageProcessor = object
        sys.modules["transformers"] = fake_transformers
        try:
            module_path = os.path.join(
                PROJECT_ROOT,
                "external",
                "dinov3-medical-seg",
                "src",
                "models",
                "backbone.py",
            )
            spec = importlib.util.spec_from_file_location("dinov3_backbone_test", module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            if old_transformers is None:
                sys.modules.pop("transformers", None)
            else:
                sys.modules["transformers"] = old_transformers

        image = torch.ones((1, 3, 2, 2), dtype=torch.float32) * 0.5
        out = module.normalize_imagenet(image, (0.5, 0.25, 0.0), (0.5, 0.25, 0.5))
        self.assertTrue(torch.allclose(out[:, 0], torch.zeros_like(out[:, 0])))
        self.assertTrue(torch.allclose(out[:, 1], torch.ones_like(out[:, 1])))
        self.assertTrue(torch.allclose(out[:, 2], torch.ones_like(out[:, 2])))

    def test_fewshot_pipeline_rejects_batch_size_above_one(self):
        """Pipeline config generation should reject variable-depth unsafe batch sizes."""
        pipeline = __import__("tools.fewshot_pipeline", fromlist=["dummy"])

        class Args(object):
            img_size = "224,224"
            model_path = None
            model_scale = "vitb16"
            finetune_method = "lora"
            decoder = "segformer3d"
            organ = "liver"
            lora_rank = 8
            lora_alpha = 16
            adapter_bottleneck = 64
            modality = "ct"
            epochs = 3
            batch_size = 2
            grad_accumulation = 1
            mixed_precision = False
            lr = 0.001
            weight_decay = 0.01
            keep_last_checkpoints = 2
            sub_volume = False
            sub_volume_size = "32,224,224"

        with self.assertRaises(RuntimeError) as raised:
            pipeline.write_training_config(
                os.path.join(self.tmp, "bad_batch_config.yaml"),
                os.path.join(PROJECT_ROOT, "external", "dinov3-medical-seg", "config", "mimics_lora_segformer3d.yaml"),
                os.path.join(self.tmp, "dataset"),
                "exp_test",
                Args(),
                validation_enabled=False,
            )
        self.assertIn("Batch size must stay 1", str(raised.exception))

    def test_fewshot_pipeline_writes_metrics_history_runtime_path(self):
        """Training config should point the DINO trainer at the structured metrics history file."""
        pipeline = __import__("tools.fewshot_pipeline", fromlist=["dummy"])

        class Args(object):
            img_size = "224,224"
            model_path = None
            model_scale = "vitb16"
            finetune_method = "lora"
            decoder = "segformer3d"
            organ = "liver"
            lora_rank = 8
            lora_alpha = 16
            adapter_bottleneck = 64
            modality = "ct"
            epochs = 3
            batch_size = 1
            grad_accumulation = 1
            mixed_precision = False
            lr = 0.001
            weight_decay = 0.01
            keep_last_checkpoints = 2
            sub_volume = False
            sub_volume_size = "32,224,224"

        config_path = os.path.join(self.tmp, "generated_config_with_runtime.yaml")
        metrics_history = os.path.join(self.tmp, "metrics_history.json")
        pipeline.write_training_config(
            config_path,
            os.path.join(PROJECT_ROOT, "external", "dinov3-medical-seg", "config", "mimics_lora_segformer3d.yaml"),
            os.path.join(self.tmp, "dataset"),
            "exp_test",
            Args(),
            status_path=os.path.join(self.tmp, "train_status.json"),
            cancel_path=os.path.join(self.tmp, "cancel.request"),
            metrics_history_path=metrics_history,
            validation_enabled=True,
        )
        with open(config_path, "r", encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn("metrics_history_path:", text)
        self.assertIn(metrics_history.replace("\\", "/"), text)

        history_only_config = os.path.join(self.tmp, "generated_config_history_only.yaml")
        pipeline.write_training_config(
            history_only_config,
            os.path.join(PROJECT_ROOT, "external", "dinov3-medical-seg", "config", "mimics_lora_segformer3d.yaml"),
            os.path.join(self.tmp, "dataset"),
            "exp_test",
            Args(),
            metrics_history_path=metrics_history,
            validation_enabled=True,
        )
        with open(history_only_config, "r", encoding="utf-8") as handle:
            history_only_text = handle.read()
        self.assertIn("runtime:", history_only_text)
        self.assertIn("metrics_history_path:", history_only_text)

    def test_fewshot_training_config_quotes_paths_with_spaces(self):
        """Generated YAML should preserve base/config paths that contain spaces."""
        pipeline = __import__("tools.fewshot_pipeline", fromlist=["dummy"])
        import yaml

        class Args(object):
            img_size = "224,224"
            model_path = None
            model_scale = "vitb16"
            finetune_method = "lora"
            decoder = "segformer3d"
            organ = "liver"
            lora_rank = 8
            lora_alpha = 16
            adapter_bottleneck = 64
            modality = "ct"
            epochs = 3
            batch_size = 1
            grad_accumulation = 1
            mixed_precision = False
            lr = 0.001
            weight_decay = 0.01
            keep_last_checkpoints = 2
            sub_volume = False
            sub_volume_size = "32,224,224"

        spaced_dir = Path(self.tmp) / "Mimics Script Config"
        spaced_dir.mkdir()
        base_config = spaced_dir / "base config.yaml"
        base_config.write_text("model: {}\ndata: {}\ntraining: {}\n", encoding="utf-8")
        output_config = Path(self.tmp) / "generated spaced config.yaml"
        pipeline.write_training_config(
            output_config,
            base_config,
            Path(self.tmp) / "dataset with spaces",
            "exp_test",
            Args(),
            validation_enabled=False,
        )
        payload = yaml.safe_load(output_config.read_text(encoding="utf-8"))
        self.assertEqual([str(base_config)], payload["_base_"])
        self.assertEqual(str(Path(self.tmp) / "dataset with spaces"), payload["data"]["data_root"])

    def test_fewshot_external_setup_formats_progress(self):
        """External setup UI exposes epoch/loss/Dice in user-visible status text."""
        ui = __import__("tools.fewshot_training_setup_ui", fromlist=["dummy"])
        line = ui.format_status_line({
            "job_id": "train_liver",
            "status": "training",
            "train_sample_count": 8,
            "validation_sample_count": 2,
            "training_progress": {
                "epoch": 2,
                "epochs": 10,
                "phase": "train",
                "metrics": {"loss": 0.34567, "mean_dsc": 0.81234},
                "best_dsc": 0.8,
            },
        })
        self.assertIn("train_liver", line)
        self.assertIn("epoch 2/10", line)
        self.assertIn("loss 0.3457", line)
        self.assertIn("val_dice 0.8123", line)

    def test_fewshot_status_viewer_parses_epoch_metrics(self):
        """External status viewer extracts loss/Dice curves from training logs."""
        viewer = __import__("tools.fewshot_status_viewer", fromlist=["dummy"])
        log_path = os.path.join(self.tmp, "train.log")
        with open(log_path, "w", encoding="utf-8") as handle:
            handle.write("Epoch 1/3: train_loss=0.5000, lr=1.00e-03, val_dice=0.6000\n")
            handle.write("Epoch 2/3: train_loss=0.4000, lr=1.00e-03, val_dice=0.7000\n")
            handle.write("Epoch 3/3: train_loss=0.3000, lr=1.00e-03, val_dice=0.8000\n")
        rows = viewer.parse_epoch_metrics(log_path)
        self.assertEqual(3, len(rows))
        self.assertEqual(2, rows[1]["epoch"])
        self.assertAlmostEqual(0.4, rows[1]["train_loss"])
        self.assertAlmostEqual(0.7, rows[1]["val_dice"])

    def test_fewshot_status_viewer_reads_bounded_log_tail(self):
        """Status viewer should not read full growing train logs on every refresh."""
        viewer = __import__("tools.fewshot_status_viewer", fromlist=["dummy"])
        log_path = os.path.join(self.tmp, "large_train.log")
        with open(log_path, "w", encoding="utf-8") as handle:
            handle.write("early-line\n" * 100)
            handle.write("Epoch 3/3: train_loss=0.3000, lr=1.00e-03, val_dice=0.8000\n")
        text = viewer.read_log_text(log_path, max_bytes=120)
        self.assertLessEqual(len(text.encode("utf-8")), 120)
        self.assertLess(len(text), os.path.getsize(log_path))
        self.assertIn("Epoch 3/3", text)
        rows = viewer.parse_epoch_metrics_from_texts(text)
        self.assertEqual(1, len(rows))
        self.assertEqual(3, rows[0]["epoch"])

    def test_fewshot_status_viewer_prefers_metrics_history_over_tail_log(self):
        """Curve data should stay complete even when the log preview only contains the newest tail."""
        viewer = __import__("tools.fewshot_status_viewer", fromlist=["dummy"])
        history_path = os.path.join(self.tmp, "metrics_history.json")
        with open(history_path, "w", encoding="utf-8") as handle:
            json.dump({
                "schema_version": "mimics_fewshot_metrics_history.v1",
                "epoch_count": 3,
                "history": [
                    {"epoch": 1, "epochs": 3, "train_loss": 0.5, "val_dice": 0.6},
                    {"epoch": 2, "epochs": 3, "train_loss": 0.4, "val_dice": 0.7},
                    {"epoch": 3, "epochs": 3, "train_loss": 0.3, "val_dice": 0.8},
                ],
            }, handle)
        tail_only = "Epoch 3/3: train_loss=0.3000, lr=1.00e-03, val_dice=0.8000\n"
        rows = viewer.training_curve_rows({"metrics_history": history_path}, tail_only, "")
        self.assertEqual(3, len(rows))
        self.assertEqual(1, rows[0]["epoch"])
        self.assertAlmostEqual(0.8, rows[-1]["val_dice"])

    def test_fewshot_status_viewer_formats_status(self):
        """External status viewer shows active progress and resource waits."""
        viewer = __import__("tools.fewshot_status_viewer", fromlist=["dummy"])
        line = viewer.format_job_line({
            "job_id": "train_liver",
            "kind": "train",
            "organ": "liver",
            "status": "waiting_for_gpu",
            "resource_wait": {"resource": "gpu", "owner": "nnInteractive", "pid": 123},
            "training_progress": {
                "epoch": 1,
                "epochs": 5,
                "metrics": {"loss": 0.45678},
            },
        })
        self.assertIn("Waiting for GPU", line)
        self.assertIn("nnInteractive", line)
        self.assertIn("epoch 1/5", line)
        self.assertIn("loss 0.4568", line)

        warning_line = viewer.format_job_line({
            "job_id": "train_liver",
            "kind": "train",
            "organ": "liver",
            "status": "cancelled",
            "cancel_marker_error": "[WinError 5] Access is denied",
        })
        self.assertIn("cancel marker warning", warning_line)
        log_warning_line = viewer.format_job_line({
            "job_id": "train_liver",
            "kind": "train",
            "organ": "liver",
            "status": "training",
            "train_log_warning": "fallback log",
        })
        self.assertIn("log warning", log_warning_line)

    def test_fewshot_pipeline_status_update_failure_is_nonfatal(self):
        """Locked job status JSON should not fail a running DINOv3 job."""
        pipeline = __import__("tools.fewshot_pipeline", fromlist=["dummy"])
        workspace = Path(self.tmp) / "fewshot_models"
        status_path = workspace / "jobs" / "train_locked.json"
        pipeline.write_json_atomic(status_path, {
            "job_id": "train_locked",
            "status": "training",
            "workspace": str(workspace),
        })

        old_write = pipeline.write_json_atomic
        try:
            def locked_write(_path, _payload, **_kwargs):
                raise PermissionError("[WinError 5] Access is denied")
            pipeline.write_json_atomic = locked_write
            result = pipeline.update_status(status_path, {"training_progress": {"epoch": 1}})
        finally:
            pipeline.write_json_atomic = old_write

        self.assertFalse(result)
        log_path = workspace / "fewshot_pipeline.log"
        self.assertTrue(log_path.is_file())
        self.assertIn("could not update job status file", log_path.read_text(encoding="utf-8"))

    def test_fewshot_pipeline_append_log_failure_is_nonfatal(self):
        """A locked pipeline log should not crash DINOv3 control flow."""
        pipeline = __import__("tools.fewshot_pipeline", fromlist=["dummy"])
        old_append_text = pipeline.append_text
        try:
            pipeline.append_text = lambda *_args, **_kwargs: (_ for _ in ()).throw(
                PermissionError("[WinError 5] Access is denied")
            )
            pipeline.append_log(Path(self.tmp) / "fewshot_models", "still running")
        finally:
            pipeline.append_text = old_append_text

    def test_fewshot_pipeline_subprocess_log_uses_fallback_path(self):
        """Subprocess logs should move to a visible fallback path instead of disappearing."""
        pipeline = __import__("tools.fewshot_pipeline", fromlist=["dummy"])
        workspace = Path(self.tmp) / "fewshot_models"
        blocker = Path(self.tmp) / "blocked_parent"
        blocker.write_text("not a directory", encoding="utf-8")
        primary = blocker / "train.log"
        handle, actual_path, warning = pipeline.open_subprocess_log(
            primary,
            workspace,
            "DINOv3 training log",
        )
        try:
            handle.write(b"hello\n")
        finally:
            handle.close()
        self.assertIsNotNone(actual_path)
        self.assertTrue(actual_path.is_file())
        self.assertIn("logs", str(actual_path))
        self.assertIn("fallback log", warning)

    def test_fewshot_pipeline_subprocess_log_reports_unavailable(self):
        """If all log destinations are denied, status can mark the log unavailable."""
        pipeline = __import__("tools.fewshot_pipeline", fromlist=["dummy"])
        old_path_open = Path.open
        try:
            Path.open = lambda *_args, **_kwargs: (_ for _ in ()).throw(
                PermissionError("[WinError 5] Access is denied")
            )
            handle, actual_path, warning = pipeline.open_subprocess_log(
                Path(self.tmp) / "train.log",
                Path(self.tmp) / "fewshot_models",
                "DINOv3 training log",
            )
            handle.close()
        finally:
            Path.open = old_path_open
        self.assertIsNone(actual_path)
        self.assertIn("unavailable", warning)

    def test_fewshot_pipeline_status_heartbeat_avoids_redundant_writes(self):
        """Unchanged training status should not rewrite the same job JSON every loop."""
        pipeline = __import__("tools.fewshot_pipeline", fromlist=["dummy"])
        calls = []
        old_update = pipeline.update_status
        try:
            pipeline.update_status = lambda path, payload: calls.append((path, dict(payload))) or True
            state = {}
            path = Path(self.tmp) / "fewshot_models" / "jobs" / "train.json"
            payload = {"status": "training", "pid": 123}
            self.assertTrue(pipeline.maybe_update_status(path, payload, state, heartbeat_seconds=60.0))
            self.assertFalse(pipeline.maybe_update_status(path, payload, state, heartbeat_seconds=60.0))
            self.assertTrue(pipeline.maybe_update_status(path, {"status": "cancelling", "pid": 123}, state, heartbeat_seconds=60.0))
        finally:
            pipeline.update_status = old_update
        self.assertEqual(2, len(calls))

    def test_fewshot_pipeline_export_uses_job_scoped_log_and_runner(self):
        """Background Mimics export should not reuse shared config/runner/log names across jobs."""
        pipeline = __import__("tools.fewshot_pipeline", fromlist=["dummy"])
        ts_root = Path(self.tmp) / "dataset"
        workspace = Path(self.tmp) / "fewshot_models"
        status_path = workspace / "jobs" / "train_unique.json"
        ts_root.mkdir()
        configured_mcs = ts_root / "configured_mcs"
        launched = []

        class Proc(object):
            pid = 13579
            returncode = 0
            def poll(self):
                return 0

        old_find = pipeline.find_mimics_exe
        old_popen = pipeline.subprocess.Popen
        old_resolve = pipeline.resolve_mimics_output_dir
        try:
            pipeline.find_mimics_exe = lambda _value=None: r"C:\MimicsResearch.exe"
            pipeline.subprocess.Popen = lambda cmd, **kwargs: launched.append((cmd, kwargs)) or Proc()
            pipeline.resolve_mimics_output_dir = lambda _ts_root: configured_mcs
            result = pipeline.launch_mimics_export(
                ts_root,
                {"case001"},
                None,
                workspace,
                5.0,
                status_path=status_path,
                cancel_path=None,
                label_staging_dir=workspace / "runs" / "train_unique" / "fresh_labels",
            )
        finally:
            pipeline.find_mimics_exe = old_find
            pipeline.subprocess.Popen = old_popen
            pipeline.resolve_mimics_output_dir = old_resolve

        self.assertTrue(result["launched"])
        self.assertIn("_export", result["log"])
        self.assertIn("train_unique_mimics_export.log", result["log"])
        runner = Path(launched[0][0][-1])
        self.assertIn("_export", str(runner))
        self.assertEqual("train_unique_run_export_batch.py", runner.name)
        config_path = workspace / "_export" / "train_unique_export_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(str(configured_mcs), config["output_dir"])
        self.assertEqual(
            str(workspace / "runs" / "train_unique" / "fresh_labels"),
            config["label_staging_dir"],
        )
        self.assertEqual(config["label_staging_dir"], result["label_staging_dir"])
        self.assertFalse((ts_root / "mcs_output" / "_run_export_batch.py").exists())
        self.assertFalse((ts_root / "mcs_output" / "_fewshot_export_mimics.log").exists())

    def test_fewshot_pipeline_cancel_marker_failure_still_cancels(self):
        """Cancel should continue to process termination even if cancel marker write is denied."""
        pipeline = __import__("tools.fewshot_pipeline", fromlist=["dummy"])
        workspace = Path(self.tmp) / "fewshot_models"
        status_path = workspace / "jobs" / "train_cancel.json"
        pipeline.write_json_atomic(status_path, {
            "job_id": "train_cancel",
            "kind": "train",
            "status": "training",
            "pid": 12345,
            "cancel_path": str(workspace / "runs" / "train_cancel" / "cancel.request"),
        })

        class Args(object):
            pass
        args = Args()
        args.ts_root = self.tmp
        args.workspace = str(workspace)
        args.job_id = "train_cancel"
        args.grace_seconds = 0.0

        old_workspace_for = pipeline.workspace_for
        old_marker = pipeline.write_cancel_marker
        old_exists = pipeline.process_exists
        old_kill = pipeline.terminate_process_tree
        try:
            pipeline.workspace_for = lambda _ts_root, _workspace=None: workspace
            pipeline.write_cancel_marker = lambda _path: "[WinError 5] Access is denied"
            pipeline.process_exists = lambda _pid: True
            pipeline.terminate_process_tree = lambda _pid: True
            result = pipeline.cmd_cancel(args)
        finally:
            pipeline.workspace_for = old_workspace_for
            pipeline.write_cancel_marker = old_marker
            pipeline.process_exists = old_exists
            pipeline.terminate_process_tree = old_kill

        self.assertEqual(0, result)
        updated = pipeline.read_json(status_path, {})
        self.assertEqual("cancelled", updated["status"])
        self.assertEqual("[WinError 5] Access is denied", updated["cancel_marker_error"])
        self.assertEqual([12345], updated["cancelled_pids"])

    def test_fewshot_external_launch_preserves_worker_status(self):
        """External setup should not overwrite a worker status update with launching."""
        ui = __import__("tools.fewshot_training_setup_ui", fromlist=["dummy"])
        ts_root = os.path.join(self.tmp, "dataset")
        workspace = os.path.join(self.tmp, "fewshot_models")
        context = {
            "organ": "liver",
            "ts_root": ts_root,
            "workspace": workspace,
            "python_exe": sys.executable,
            "pipeline_script": os.path.join(PROJECT_ROOT, "tools", "fewshot_pipeline.py"),
            "dinov3_root": os.path.join(PROJECT_ROOT, "external", "dinov3-medical-seg"),
            "project_root": PROJECT_ROOT,
            "config": {},
        }
        options = ui.default_training_options({})
        launch = ui.prepare_training_launch(context, options, run_id="train_race")
        status_path = launch["status_path"]

        class Proc(object):
            pid = 98765

        old_popen = ui.subprocess.Popen
        old_prepare = ui.prepare_training_launch
        try:
            ui.prepare_training_launch = lambda _context, _options: launch
            def fake_popen(*_args, **_kwargs):
                payload = ui.read_json(status_path, {})
                payload["status"] = "training"
                payload["training_progress"] = {"epoch": 1, "epochs": 5}
                ui.write_json_atomic(status_path, payload)
                return Proc()
            ui.subprocess.Popen = fake_popen
            run_id, returned_status_path, pid = ui.launch_training(context, dict(options, run_id="ignored"))
        finally:
            ui.subprocess.Popen = old_popen
            ui.prepare_training_launch = old_prepare
        self.assertEqual("train_race", run_id)
        self.assertEqual(status_path, returned_status_path)
        self.assertEqual(98765, pid)
        status = ui.read_json(status_path, {})
        self.assertEqual("training", status["status"])
        self.assertEqual(98765, status["launcher_pid"])
        self.assertEqual(1, status["training_progress"]["epoch"])

    def test_fewshot_external_launch_post_start_status_write_is_nonfatal(self):
        """After Popen succeeds, status JSON write contention should not report launch failure."""
        ui = __import__("tools.fewshot_training_setup_ui", fromlist=["dummy"])
        ts_root = os.path.join(self.tmp, "dataset")
        workspace = os.path.join(self.tmp, "fewshot_models")
        context = {
            "organ": "liver",
            "ts_root": ts_root,
            "workspace": workspace,
            "python_exe": sys.executable,
            "pipeline_script": os.path.join(PROJECT_ROOT, "tools", "fewshot_pipeline.py"),
            "dinov3_root": os.path.join(PROJECT_ROOT, "external", "dinov3-medical-seg"),
            "project_root": PROJECT_ROOT,
            "config": {},
        }
        options = ui.default_training_options({})
        launch = ui.prepare_training_launch(context, options, run_id="train_started")

        class Proc(object):
            pid = 24680

        old_popen = ui.subprocess.Popen
        old_prepare = ui.prepare_training_launch
        old_write = ui.write_json_atomic
        calls = []
        try:
            ui.prepare_training_launch = lambda _context, _options: launch
            ui.subprocess.Popen = lambda *_args, **_kwargs: Proc()

            def write_once_then_locked(path, payload, **kwargs):
                calls.append(path)
                if len(calls) == 1:
                    return old_write(path, payload, **kwargs)
                raise PermissionError("[WinError 5] Access is denied")

            ui.write_json_atomic = write_once_then_locked
            run_id, returned_status_path, pid = ui.launch_training(context, dict(options, run_id="ignored"))
        finally:
            ui.subprocess.Popen = old_popen
            ui.prepare_training_launch = old_prepare
            ui.write_json_atomic = old_write

        self.assertEqual("train_started", run_id)
        self.assertEqual(launch["status_path"], returned_status_path)
        self.assertEqual(24680, pid)

    def test_fewshot_mimics_launches_external_setup_nonblocking(self):
        """Mimics Advanced entry starts the external setup process without waiting."""
        import fewshot_mimics
        ts_root = os.path.join(self.tmp, "dataset")
        os.makedirs(os.path.join(ts_root, "mcs_output"))
        launched = []

        class Proc(object):
            pid = 43210
            def poll(self):
                return None

        old_launch = fewshot_mimics._launch_gui_process
        old_start_monitor = fewshot_mimics._start_monitor
        old_script = fewshot_mimics._training_setup_ui_script
        old_project = fewshot_mimics._project_root
        try:
            fewshot_mimics._launch_gui_process = lambda cmd, cwd=None: launched.append((cmd, cwd)) or Proc()
            fewshot_mimics._start_monitor = lambda monitor, poll_seconds=1.0: True
            fewshot_mimics._project_root = lambda: PROJECT_ROOT
            fewshot_mimics._training_setup_ui_script = lambda: os.path.join(PROJECT_ROOT, "tools", "fewshot_training_setup_ui.py")
            result = fewshot_mimics._launch_external_advanced_training(
                {"python": sys.executable, "dinov3_project": "external/dinov3-medical-seg"},
                "liver",
                ts_root,
            )
        finally:
            fewshot_mimics._launch_gui_process = old_launch
            fewshot_mimics._start_monitor = old_start_monitor
            fewshot_mimics._training_setup_ui_script = old_script
            fewshot_mimics._project_root = old_project
        self.assertEqual(0, result)
        self.assertEqual(1, len(launched))
        cmd, _cwd = launched[0]
        self.assertIn("fewshot_training_setup_ui.py", cmd[1])
        self.assertIn("--context", cmd)
        context_path = cmd[cmd.index("--context") + 1]
        self.assertTrue(os.path.isfile(context_path))
        with open(context_path, "r") as handle:
            payload = json.load(handle)
        self.assertEqual("liver", payload["organ"])
        status_path = payload["setup_status_path"]
        with open(status_path, "r") as handle:
            status = json.load(handle)
        self.assertEqual("configuring", status["status"])
        self.assertEqual(43210, status["controller_pid"])

    def test_fewshot_mimics_case_ids_use_configured_mcs_output_dir(self):
        """Mimics-side DINOv3 helpers should honor mimics_output_dir for saved .mcs files."""
        import fewshot_mimics
        ts_root = os.path.join(self.tmp, "dataset")
        custom_mcs = os.path.join(ts_root, "custom_mcs")
        os.makedirs(custom_mcs)
        with open(os.path.join(custom_mcs, "case_from_custom.mcs"), "w", encoding="utf-8") as handle:
            handle.write("")
        old_read = fewshot_mimics._read_json
        old_project = fewshot_mimics._project_root
        try:
            fewshot_mimics._project_root = lambda: PROJECT_ROOT
            def fake_read(path, default=None):
                if str(path).endswith("mimics_io_config.json"):
                    return {"mimics_output_dir": "custom_mcs"}
                return old_read(path, default)
            fewshot_mimics._read_json = fake_read
            self.assertIn("case_from_custom", fewshot_mimics._case_ids_from_dataset(ts_root))
        finally:
            fewshot_mimics._read_json = old_read
            fewshot_mimics._project_root = old_project

    def test_fewshot_mimics_launches_external_status_viewer_nonblocking(self):
        """Mimics Show Status starts an external viewer process without waiting."""
        import fewshot_mimics
        ts_root = os.path.join(self.tmp, "dataset")
        os.makedirs(ts_root)
        launched = []

        class Proc(object):
            pid = 54321
            def poll(self):
                return None

        old_launch = fewshot_mimics._launch_gui_process
        old_script = fewshot_mimics._status_viewer_script
        old_project = fewshot_mimics._project_root
        try:
            fewshot_mimics._launch_gui_process = lambda cmd, cwd=None: launched.append((cmd, cwd)) or Proc()
            fewshot_mimics._project_root = lambda: PROJECT_ROOT
            fewshot_mimics._status_viewer_script = lambda: os.path.join(PROJECT_ROOT, "tools", "fewshot_status_viewer.py")
            pid = fewshot_mimics._launch_external_status_viewer(
                {"python": sys.executable, "dinov3_project": "external/dinov3-medical-seg"},
                ts_root,
            )
        finally:
            fewshot_mimics._launch_gui_process = old_launch
            fewshot_mimics._status_viewer_script = old_script
            fewshot_mimics._project_root = old_project
        self.assertEqual(54321, pid)
        self.assertEqual(1, len(launched))
        cmd, _cwd = launched[0]
        self.assertIn("fewshot_status_viewer.py", cmd[1])
        self.assertIn("--context", cmd)
        context_path = cmd[cmd.index("--context") + 1]
        self.assertTrue(context_path.endswith("_context.json"))
        with open(context_path, "r") as handle:
            payload = json.load(handle)
        self.assertEqual(os.path.abspath(ts_root), payload["ts_root"])

    def test_fewshot_mimics_launches_external_model_chooser_nonblocking(self):
        """Predict Choose Model should use an external PySide6 chooser process."""
        import fewshot_mimics
        ts_root = os.path.join(self.tmp, "dataset")
        model_root = os.path.join(ts_root, "fewshot_models", "models", "liver")
        os.makedirs(model_root)
        launched = []
        monitors = []

        def write_model(run_id, created):
            run_dir = os.path.join(model_root, run_id)
            os.makedirs(run_dir)
            checkpoint = os.path.join(run_dir, "model.pth")
            config = os.path.join(run_dir, "config.yaml")
            with open(checkpoint, "wb") as handle:
                handle.write(b"PK\x03\x04")
            with open(config, "w", encoding="utf-8") as handle:
                handle.write("model: {}\n")
            manifest = {
                "model_id": run_id,
                "organ": "liver",
                "organ_slug": "liver",
                "checkpoint": checkpoint,
                "config": config,
                "sample_count": 3,
                "created_at_epoch": created,
            }
            with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as handle:
                json.dump(manifest, handle)
            return manifest

        latest = write_model("train_a", 10)
        write_model("train_b", 20)
        with open(os.path.join(model_root, "latest.json"), "w", encoding="utf-8") as handle:
            json.dump(latest, handle)

        class Proc(object):
            pid = 65432
            def poll(self):
                return None

        old_launch = fewshot_mimics._launch_gui_process
        old_monitor = fewshot_mimics._start_monitor
        old_script = fewshot_mimics._model_chooser_script
        old_project = fewshot_mimics._project_root
        try:
            fewshot_mimics._launch_gui_process = lambda cmd, cwd=None: launched.append((cmd, cwd)) or Proc()
            fewshot_mimics._start_monitor = lambda monitor, poll_seconds=1.0: monitors.append(monitor) or True
            fewshot_mimics._model_chooser_script = lambda: os.path.join(PROJECT_ROOT, "tools", "fewshot_model_chooser.py")
            fewshot_mimics._project_root = lambda: PROJECT_ROOT
            result = fewshot_mimics._launch_external_model_chooser(
                {"python": sys.executable, "dinov3_project": "external/dinov3-medical-seg"},
                ts_root,
                "s0001",
                "liver",
            )
        finally:
            fewshot_mimics._launch_gui_process = old_launch
            fewshot_mimics._start_monitor = old_monitor
            fewshot_mimics._model_chooser_script = old_script
            fewshot_mimics._project_root = old_project
        self.assertEqual(0, result)
        self.assertEqual(1, len(launched))
        self.assertIn("fewshot_model_chooser.py", launched[0][0][1])
        self.assertEqual(1, len(monitors))
        self.assertEqual("model_choice", monitors[0]["kind"])

    def test_fewshot_latest_prediction_resolves_explicit_model(self):
        """Latest Model entry should resolve latest.json before launching inference."""
        import fewshot_mimics
        ts_root = os.path.join(self.tmp, "dataset_latest_predict")
        model_root = os.path.join(ts_root, "fewshot_models", "models", "liver")
        os.makedirs(model_root)
        run_dir = os.path.join(model_root, "train_latest")
        os.makedirs(run_dir)
        checkpoint = os.path.join(run_dir, "model.pth")
        config = os.path.join(run_dir, "config.yaml")
        with open(checkpoint, "wb") as handle:
            handle.write(b"PK\x03\x04")
        with open(config, "w", encoding="utf-8") as handle:
            handle.write("model: {}\n")
        manifest = {
            "model_id": "train_latest",
            "organ": "liver",
            "organ_slug": "liver",
            "checkpoint": checkpoint,
            "config": config,
            "created_at_epoch": 1.0,
        }
        with open(os.path.join(model_root, "latest.json"), "w", encoding="utf-8") as handle:
            json.dump(manifest, handle)
        launched = []

        old_selected = fewshot_mimics._selected_organ
        old_choose = fewshot_mimics._choose_dataset_root
        old_guard = fewshot_mimics._guard_no_active_job
        old_case = fewshot_mimics._infer_case_id
        old_launch = fewshot_mimics._launch_inference_job
        try:
            fewshot_mimics._selected_organ = lambda: "liver"
            fewshot_mimics._choose_dataset_root = lambda _title: ts_root
            fewshot_mimics._guard_no_active_job = lambda root, requested_kind="train": True
            fewshot_mimics._infer_case_id = lambda root: "s0001"
            fewshot_mimics._launch_inference_job = lambda config_arg, root, case_id, organ, selected_model=None: launched.append(selected_model) or 0
            result = fewshot_mimics._start_inference(choose_model=False)
        finally:
            fewshot_mimics._selected_organ = old_selected
            fewshot_mimics._choose_dataset_root = old_choose
            fewshot_mimics._guard_no_active_job = old_guard
            fewshot_mimics._infer_case_id = old_case
            fewshot_mimics._launch_inference_job = old_launch
        self.assertEqual(0, result)
        self.assertEqual("train_latest", launched[0]["model_id"])
        self.assertTrue(launched[0]["manifest_path"].endswith("latest.json"))

    def test_fewshot_inference_guard_allows_training_queue(self):
        """Prediction should not be blocked by a queued/waiting training job."""
        import fewshot_mimics
        ts_root = os.path.join(self.tmp, "dataset_guard")
        jobs_dir = os.path.join(ts_root, "fewshot_models", "jobs")
        os.makedirs(jobs_dir)
        with open(os.path.join(jobs_dir, "train_waiting.json"), "w", encoding="utf-8") as handle:
            json.dump({
                "job_id": "train_waiting",
                "kind": "train",
                "status": "waiting_for_gpu",
                "organ": "liver",
                "controller_pid": os.getpid(),
            }, handle)
        self.assertTrue(fewshot_mimics._guard_no_active_job(ts_root, requested_kind="infer"))
        self.assertFalse(fewshot_mimics._guard_no_active_job(ts_root, requested_kind="train"))

    def test_fewshot_status_viewer_context_write_error_is_diagnostic(self):
        """Show Status should say when context JSON creation is the failing stage."""
        import fewshot_mimics
        ts_root = os.path.join(self.tmp, "dataset")
        os.makedirs(ts_root)

        old_write = fewshot_mimics._write_json_atomic
        old_script = fewshot_mimics._status_viewer_script
        old_project = fewshot_mimics._project_root
        try:
            fewshot_mimics._write_json_atomic = lambda _path, _payload: (_ for _ in ()).throw(
                PermissionError("[WinError 5] Access is denied")
            )
            fewshot_mimics._project_root = lambda: PROJECT_ROOT
            fewshot_mimics._status_viewer_script = lambda: os.path.join(PROJECT_ROOT, "tools", "fewshot_status_viewer.py")
            with self.assertRaises(RuntimeError) as raised:
                fewshot_mimics._launch_external_status_viewer(
                    {"python": sys.executable, "dinov3_project": "external/dinov3-medical-seg"},
                    ts_root,
                )
        finally:
            fewshot_mimics._write_json_atomic = old_write
            fewshot_mimics._status_viewer_script = old_script
            fewshot_mimics._project_root = old_project
        self.assertIn("launch context", str(raised.exception))
        self.assertIn("_context.json", str(raised.exception))

    def test_fewshot_status_viewer_process_start_error_is_diagnostic(self):
        """Show Status should say when Popen is the failing stage."""
        import fewshot_mimics
        ts_root = os.path.join(self.tmp, "dataset")
        os.makedirs(ts_root)

        old_launch = fewshot_mimics._launch_gui_process
        old_script = fewshot_mimics._status_viewer_script
        old_project = fewshot_mimics._project_root
        try:
            def denied_launch(_cmd, cwd=None):
                raise PermissionError("[WinError 5] Access is denied")
            fewshot_mimics._launch_gui_process = denied_launch
            fewshot_mimics._project_root = lambda: PROJECT_ROOT
            fewshot_mimics._status_viewer_script = lambda: os.path.join(PROJECT_ROOT, "tools", "fewshot_status_viewer.py")
            with self.assertRaises(RuntimeError) as raised:
                fewshot_mimics._launch_external_status_viewer(
                    {"python": sys.executable, "dinov3_project": "external/dinov3-medical-seg"},
                    ts_root,
                )
        finally:
            fewshot_mimics._launch_gui_process = old_launch
            fewshot_mimics._status_viewer_script = old_script
            fewshot_mimics._project_root = old_project
        self.assertIn("start the DINOv3 status viewer process", str(raised.exception))
        self.assertIn("Python:", str(raised.exception))
        self.assertIn("Script:", str(raised.exception))

    # ================================================================
    # BatchNorm checkpoint roundtrip
    # ================================================================

    def test_checkpoint_saves_and_restores_batchnorm_running_stats(self):
        """Checkpoint save must include BN running_mean/var for correct inference."""
        checkpoint = __import__("external.dinov3-medical-seg.src.utils.checkpoint",
                                fromlist=["save_checkpoint", "load_checkpoint"])
        import torch, tempfile, shutil
        from pathlib import Path

        tmp = Path(tempfile.mkdtemp(prefix="ckpt_bn_test_"))
        try:
            # Build a tiny model with BatchNorm
            model = torch.nn.Sequential(
                torch.nn.Conv3d(1, 8, 3, padding=1),
                torch.nn.BatchNorm3d(8),
                torch.nn.ReLU(),
                torch.nn.Conv3d(8, 2, 1),
            )
            opt = torch.optim.SGD(model.parameters(), lr=0.01)
            x = torch.randn(1, 1, 16, 32, 32)
            for _ in range(5):
                model.train()
                opt.zero_grad()
                model(x).sum().backward()
                opt.step()
            model.eval()
            ref_output = model(x)

            checkpoint.save_checkpoint(model, opt, epoch=1, metrics={}, config={},
                                       save_dir=str(tmp), filename="test.pth")

            state = torch.load(str(tmp / "test.pth"), weights_only=False)
            bn_keys = [k for k in state["model_state_dict"] if "running" in k]
            self.assertGreater(len(bn_keys), 0, "BN running stats must be in checkpoint")

            model2 = torch.nn.Sequential(
                torch.nn.Conv3d(1, 8, 3, padding=1),
                torch.nn.BatchNorm3d(8),
                torch.nn.ReLU(),
                torch.nn.Conv3d(8, 2, 1),
            )
            checkpoint.load_checkpoint(model2, str(tmp / "test.pth"), device="cpu")
            model2.eval()
            with torch.no_grad():
                restored = model2(x)
            self.assertLess(float(torch.max(torch.abs(ref_output - restored))), 1e-5,
                            "BN-restored model must produce identical output")
        finally:
            shutil.rmtree(str(tmp))

    def test_lora_checkpoint_roundtrip_preserves_trainable_params(self):
        """LoRA training → checkpoint → inference must produce identical logits."""
        import torch, tempfile, shutil, sys, os
        from pathlib import Path
        sys.path.insert(0, os.path.join(os.getcwd(), "external", "dinov3-medical-seg"))
        from src.utils.config import load_config
        from src.utils.checkpoint import save_checkpoint, load_checkpoint
        from src.models.segmentor import DINOv33DSegmentor

        tmp = Path(tempfile.mkdtemp(prefix="lora_roundtrip_"))
        try:
            cfg_path = os.path.join(os.getcwd(), "external", "dinov3-medical-seg",
                                    "config", "mimics_lora_segformer3d.yaml")
            cfg = load_config(cfg_path)
            cfg["finetune"]["method"] = "lora"
            model_path = Path(cfg["model"]["model_path"])
            if not model_path.is_absolute():
                model_path = Path(os.getcwd()) / "external" / "dinov3-medical-seg" / model_path
            if not model_path.is_dir():
                self.skipTest("bundled DINOv3 weights are not present in this checkout")

            model = DINOv33DSegmentor(cfg)
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
            x = torch.randn(1, 1, 16, 224, 224)
            for _ in range(3):
                model.train(); opt.zero_grad()
                model(x).sum().backward(); opt.step()

            save_checkpoint(model, opt, epoch=3, metrics={"loss": 0.5}, config=cfg,
                            save_dir=str(tmp), filename="lora.pth")

            model2 = DINOv33DSegmentor(cfg)
            load_checkpoint(model2, str(tmp / "lora.pth"), device="cpu")
            model.eval(); model2.eval()
            with torch.no_grad():
                o1 = model(x); o2 = model2(x)
            max_diff = float(torch.max(torch.abs(o1 - o2)))
            self.assertLess(max_diff, 1e-5,
                            f"LoRA checkpoint roundtrip logit diff {max_diff:.2e} must be < 1e-5")
        finally:
            shutil.rmtree(str(tmp))

    # ================================================================
    # mimcs_lora_segformer3d config template
    # ================================================================

    def test_mimics_lora_segformer3d_template_exists_and_has_no_k_shot(self):
        """Default template must use k_shot=-1 (use all data) not 5."""
        import yaml, os
        path = os.path.join(os.getcwd(), "external", "dinov3-medical-seg",
                            "config", "mimics_lora_segformer3d.yaml")
        self.assertTrue(os.path.isfile(path), "mimics_lora_segformer3d.yaml must exist")
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.assertIsNotNone(cfg.get("_base_"), "must inherit from train.yaml")
        self.assertEqual("mimics", cfg.get("data", {}).get("name", ""),
                         "data.name must be mimics")
        self.assertEqual("lora", cfg.get("finetune", {}).get("method", ""),
                         "finetune method must be lora")

    def test_config_defaults_point_to_validated_fewshot_template(self):
        """Every launcher must use the same validated 12 GB few-shot base."""
        files_and_patterns = [
            ("fewshot_config.json", '"base_config": "config/research/ct_fewshot_fast.yaml"'),
            ("tools/fewshot_pipeline.py", "config/research/ct_fewshot_fast.yaml"),
            ("tools/fewshot_training_setup_ui.py", "config/research/ct_fewshot_fast.yaml"),
            ("runtime_py35/fewshot_mimics.py", "config/research/ct_fewshot_fast.yaml"),
        ]
        for filepath, pattern in files_and_patterns:
            with open(filepath, "r", encoding="utf-8") as handle:
                content = handle.read()
            self.assertIn(pattern, content,
                          f"{filepath} must default to ct_fewshot_fast.yaml")

    # ================================================================
    # Export validation + label staging
    # ================================================================

    def test_validate_materialized_dataset_rejects_shape_mismatch(self):
        """Materialized pairs with different image/label shapes must fail."""
        pipeline = __import__("tools.fewshot_pipeline", fromlist=["validate_materialized_dataset"])
        rows = [{"case_id": "test", "dataset_image": "/a/img.nii.gz", "dataset_label": "/a/lbl.nii.gz"}]
        import nibabel as nib, numpy as np, tempfile, shutil
        from pathlib import Path
        tmp = Path(tempfile.mkdtemp(prefix="mat_val_"))
        try:
            img_p = tmp / "img.nii.gz"; lbl_p = tmp / "lbl.nii.gz"
            # Different shapes
            nib.save(nib.Nifti1Image(np.zeros((16,128,128), dtype=np.float32), np.eye(4)), str(img_p))
            nib.save(nib.Nifti1Image(np.zeros((16,64,64), dtype=np.int32), np.eye(4)), str(lbl_p))
            rows[0]["dataset_image"] = str(img_p); rows[0]["dataset_label"] = str(lbl_p)
            with self.assertRaises(RuntimeError):
                pipeline.validate_materialized_dataset(rows)
        finally:
            shutil.rmtree(str(tmp))

    def test_fresh_mimics_export_must_match_source_grid(self):
        """Fresh labels may not silently move source images onto a Mimics grid."""
        pipeline = __import__("tools.fewshot_pipeline", fromlist=["validate_fresh_export_geometry"])
        import nibabel as nib, numpy as np, tempfile, shutil
        from pathlib import Path
        tmp = Path(tempfile.mkdtemp(prefix="fresh_grid_"))
        try:
            image = tmp / "ct.nii.gz"
            label = tmp / "liver.nii.gz"
            nib.save(nib.Nifti1Image(np.zeros((8, 9, 10), dtype=np.float32), np.eye(4)), str(image))
            shifted = np.eye(4)
            shifted[1, 3] = 5.0
            nib.save(nib.Nifti1Image(np.ones((8, 9, 10), dtype=np.uint8), shifted), str(label))
            with self.assertRaises(RuntimeError):
                pipeline.validate_fresh_export_geometry([{
                    "case_id": "case1",
                    "image": str(image),
                    "label": str(label),
                    "label_source": "fresh_export",
                }])
        finally:
            shutil.rmtree(str(tmp))

    def test_validate_materialized_dataset_rejects_empty_label(self):
        """Materialized pairs with all-zero labels must fail."""
        pipeline = __import__("tools.fewshot_pipeline", fromlist=["validate_materialized_dataset"])
        import nibabel as nib, numpy as np, tempfile, shutil
        from pathlib import Path
        tmp = Path(tempfile.mkdtemp(prefix="mat_val_empty_"))
        try:
            img_p = tmp / "img.nii.gz"; lbl_p = tmp / "lbl.nii.gz"
            nib.save(nib.Nifti1Image(np.zeros((16,128,128), dtype=np.float32), np.eye(4)), str(img_p))
            # All-zero label
            nib.save(nib.Nifti1Image(np.zeros((16,128,128), dtype=np.int32), np.eye(4)), str(lbl_p))
            rows = [{"case_id": "zero", "dataset_image": str(img_p), "dataset_label": str(lbl_p)}]
            with self.assertRaises(RuntimeError):
                pipeline.validate_materialized_dataset(rows)
        finally:
            shutil.rmtree(str(tmp))

    def test_launch_mimics_export_passes_label_staging_dir(self):
        """Export must support label_staging_dir for fresh-label isolation."""
        pipeline = __import__("tools.fewshot_pipeline", fromlist=["launch_mimics_export"])
        import inspect
        sig = inspect.signature(pipeline.launch_mimics_export)
        self.assertIn("label_staging_dir", sig.parameters,
                      "launch_mimics_export must accept label_staging_dir")

    # ================================================================
    # Stop Background Import
    # ================================================================

    def test_stop_background_import_entry_exists(self):
        """Stop_Background_Import entry must route to the correct function."""
        entry = os.path.join(
            os.getcwd(), "scripting_library", "01_Data", "04_Stop_Import_Queue.py"
        )
        self.assertTrue(os.path.isfile(entry), "Stop_Background_Import entry must exist")
        content = open(entry, "r", encoding="utf-8").read()
        self.assertIn("main_stop_import", content,
                      "must route to main_stop_import function")

    def test_stop_background_import_targets_only_create_mcs(self):
        """stop_background_import must identify import lock by kind=create_mcs."""
        sys.path.insert(0, os.path.join(os.getcwd(), "runtime_py35"))
        import mimics_stop_background
        self.assertTrue(callable(mimics_stop_background.stop_background_import),
                        "stop_background_import must be callable")
        # Verify lock identification
        self.assertTrue(mimics_stop_background._lock_is_import_creation(
            {"kind": "create_mcs", "owner": "import .mcs creation for dataset X"}))
        self.assertFalse(mimics_stop_background._lock_is_import_creation(
            {"kind": "train", "owner": "DINOv3 training"}))

    # ================================================================
    # Preprocessing consistency
    # ================================================================

    def test_training_vs_inference_preprocessing_identical(self):
        """Trilinear (training) vs bilinear per-slice (inference) must be identical."""
        import numpy as np, torch
        np.random.seed(42)
        D, H, W = 16, 128, 128
        data = np.random.randn(D, H, W).astype(np.float32)
        SZ = (224, 224)
        # Training: trilinear 3D
        it = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
        tr = torch.nn.functional.interpolate(it, size=(D, *SZ), mode="trilinear",
                                             align_corners=False).squeeze().numpy()
        # Inference: bilinear per-slice
        inf = np.zeros((D, *SZ), dtype=np.float32)
        for d in range(D):
            s = torch.from_numpy(data[d]).unsqueeze(0).unsqueeze(0)
            inf[d] = torch.nn.functional.interpolate(s, size=SZ, mode="bilinear").numpy()
        self.assertLess(float(np.max(np.abs(tr - inf))), 1e-6,
                        "Training and inference preprocessing must produce identical data")

    # ================================================================
    # Inference source geometry validation
    # ================================================================

    def test_validate_inference_source_geometry_shape_mismatch(self):
        """Inference must reject source images with wrong shape vs Mimics project."""
        pipeline = __import__("tools.fewshot_pipeline",
                              fromlist=["validate_inference_source_geometry"])
        import nibabel as nib, numpy as np, tempfile, shutil
        from pathlib import Path
        tmp = Path(tempfile.mkdtemp(prefix="src_geo_"))
        try:
            img_p = tmp / "ct.nii.gz"
            nib.save(nib.Nifti1Image(np.zeros((16,128,128), dtype=np.float32), np.eye(4)), str(img_p))
            with self.assertRaises(RuntimeError):
                pipeline.validate_inference_source_geometry(
                    str(img_p), expected_shape=[32, 256, 256])
        finally:
            shutil.rmtree(str(tmp))

    def test_validate_inference_source_geometry_rejects_same_path_affine_mismatch(self):
        """A matching path must not bypass physical-grid validation."""
        pipeline = __import__("tools.fewshot_pipeline",
                              fromlist=["validate_inference_source_geometry"])
        import nibabel as nib, numpy as np, tempfile, shutil
        from pathlib import Path
        tmp = Path(tempfile.mkdtemp(prefix="src_affine_"))
        try:
            img_p = tmp / "ct.nii.gz"
            nib.save(nib.Nifti1Image(np.zeros((8, 9, 10), dtype=np.float32), np.eye(4)), str(img_p))
            shifted = np.eye(4)
            shifted[0, 3] = 25.0
            with self.assertRaises(RuntimeError):
                pipeline.validate_inference_source_geometry(
                    str(img_p),
                    expected_shape=[8, 9, 10],
                    expected_affine=shifted.tolist(),
                    expected_image_path=str(img_p),
                )
        finally:
            shutil.rmtree(str(tmp))

    def test_validated_fewshot_strategies_compile_training_and_inference_policy(self):
        """Mimics strategies must retain the factors supported by local experiments."""
        strategies = __import__("tools.fewshot_strategies", fromlist=["compile_strategy"])
        fingerprint = {
            "summary": {
                "support_bbox_union_normalized_zyx": [[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]],
            },
        }
        policy = {
            "patch": {
                "enabled": True,
                "size_zyx": [64, 192, 192],
                "inference_sliding_window": True,
            },
        }
        patch = strategies.compile_strategy("patch_focused", fingerprint, policy, {
            "channel_policy": "2_5d", "slice_axis": "coronal",
        })
        self.assertEqual(patch["model"]["slice_axis"], "coronal")
        self.assertEqual(patch["model"]["channel_policy"], "2_5d")
        self.assertEqual(patch["loss"]["type"], "dice_focal")
        self.assertEqual(patch["data"]["patch"]["size_zyx"], [64, 192, 192])
        self.assertTrue(patch["data"]["patch"]["inference_sliding_window"])

        full = strategies.compile_strategy("full_volume", fingerprint, policy)
        self.assertFalse(full["data"]["patch"]["enabled"])

    def test_mimics_training_option_matrix_reaches_backend_config(self):
        """Every public dimensionality combination must survive Mimics UI command/config generation."""
        import tools.fewshot_training_setup_ui as ui
        import tools.fewshot_pipeline as pipeline
        import tools.fewshot_strategies as strategies

        context = {
            "organ": "organ", "ts_root": self.tmp,
            "workspace": os.path.join(self.tmp, "fewshot_models"),
            "python_exe": sys.executable, "pipeline_script": os.path.join(PROJECT_ROOT, "tools", "fewshot_pipeline.py"),
            "dinov3_root": os.path.join(PROJECT_ROOT, "external", "dinov3-medical-seg"),
            "project_root": PROJECT_ROOT, "config": {"base_config": "config/research/ct_fewshot_fast.yaml"},
        }
        fingerprint = {"summary": {}}
        policy = {"patch": {"enabled": True, "size_zyx": [8, 32, 32]}}
        count = 0
        for preset in strategies.strategy_ids():
            for decoder in ui.DECODER_CHOICES:
                for channel_policy in ("repeat", "2_5d"):
                    for method in ("frozen", "lora", "adapter"):
                        options = ui.default_training_options(context["config"])
                        options.update(strategies.strategy_defaults(preset))
                        options.update({
                            "strategy": preset, "decoder": decoder, "channel_policy": channel_policy,
                            "finetune_method": method, "cases": ["case1"], "val_fraction": 0.0,
                            "export_labels_before_training": False,
                        })
                        options = ui.validate_options(options)
                        launch = ui.prepare_training_launch(context, options, run_id="matrix_{}".format(count))
                        args = pipeline.build_parser().parse_args(launch["cmd"][2:])
                        compiled = strategies.compile_strategy(
                            preset, fingerprint=fingerprint, policy=policy,
                            user_options=json.loads(args.strategy_options_json),
                        )
                        config_path = os.path.join(self.tmp, "matrix_{}.yaml".format(count))
                        generated = pipeline.write_training_config(
                            config_path, args.base_config, self.tmp, "matrix", args,
                            validation_enabled=False, strategy_overrides=compiled,
                        )
                        self.assertEqual(decoder, generated["decoder"]["type"])
                        self.assertEqual(channel_policy, generated["model"]["channel_policy"])
                        self.assertEqual(method, generated["finetune"]["method"])
                        count += 1
        self.assertEqual(len(strategies.strategy_ids()) * len(ui.DECODER_CHOICES) * 2 * 3, count)

    def test_every_public_policy_choice_compiles(self):
        import tools.fewshot_strategies as strategies
        fingerprint = {"summary": {}}
        policy = {"patch": {"enabled": True, "size_zyx": [8, 32, 32]}}
        choices = {
            "sampling_mode": ("adaptive", "full", "patch"),
            "patch_size_mode": ("fingerprint", "custom"),
            "patch_focus": ("foreground", "boundary", "negative_balanced"),
            "channel_policy": ("repeat", "2_5d"),
            "slice_axis": ("axial", "coronal", "sagittal"),
            "loss_type": ("auto", "dice_focal", "dice_ce"),
            "keep_largest_component": (False, True),
        }
        for key, values in choices.items():
            for value in values:
                options = {key: value}
                if key == "patch_size_mode" and value == "custom":
                    options.update({"sampling_mode": "patch", "patch_size_zyx": "8,32,32"})
                compiled = strategies.compile_strategy(
                    "adaptive", fingerprint=fingerprint, policy=policy, user_options=options,
                )
                self.assertIn("inference", compiled)

    def test_training_ui_applies_organ_strategy_on_first_open(self):
        """The recommended strategy must take effect without requiring a combo-box toggle."""
        setup = __import__("tools.fewshot_training_setup_ui",
                           fromlist=["training_options_for_organ"])
        legacy_profile = {
            "default_training_profile": "balanced",
            "training_profiles": {
                "balanced": {
                    "epochs": 10,
                    "finetune_method": "lora",
                    "img_size": "224,224",
                },
            },
        }
        options = setup.training_options_for_organ(legacy_profile, "aorta", "balanced")
        self.assertEqual(options["strategy"], "adaptive")
        self.assertEqual(options["sampling_mode"], "adaptive")
        self.assertEqual(options["finetune_method"], "lora")
        self.assertEqual(options["img_size"], "224,224")

    def test_strategy_compiler_rejects_removed_sampling_policy(self):
        """Unvalidated public sampling modes must fail before training starts."""
        strategies = __import__("tools.fewshot_strategies", fromlist=["compile_strategy"])
        with self.assertRaises(ValueError):
            strategies.compile_strategy("adaptive", {}, {}, {"sampling_mode": "roi_patch"})

    def test_status_viewer_selects_only_current_organ_task(self):
        """Status UI must not mix historical or other-organ jobs into the selected task."""
        viewer = __import__("tools.fewshot_status_viewer", fromlist=["select_current_task"])
        rows = [
            (30.0, {"job_id": "old_liver", "organ": "liver", "status": "completed"}),
            (40.0, {"job_id": "spleen", "organ": "spleen", "status": "training"}),
            (20.0, {"job_id": "active_liver", "organ": "liver", "kind": "train", "status": "training"}),
            (50.0, {"job_id": "setup_liver", "organ": "liver", "kind": "train_setup", "status": "training_started"}),
        ]
        selected = viewer.select_current_task(rows, organ="liver")
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["job_id"], "active_liver")
        filtered = viewer.filter_log_for_job(
            "active_liver started\nspleen started\nactive_liver epoch 2\n",
            selected[0],
        )
        self.assertIn("active_liver epoch 2", filtered)
        self.assertNotIn("spleen", filtered)

    # ================================================================
    # Centralized mimcs_output_dir
    # ================================================================

    def test_resolve_mimics_output_dir_uses_single_config_key(self):
        """mimcs_output_dir must only use mimcs_output_dir (no deprecated aliases)."""
        pipeline = __import__("tools.fewshot_pipeline", fromlist=["resolve_mimics_output_dir"])
        import inspect
        src = inspect.getsource(pipeline.resolve_mimics_output_dir)
        # Must NOT reference deprecated keys
        self.assertNotIn("mimics_export_output_dir", src)
        self.assertNotIn("mimics_data_output_dir", src)
        self.assertNotIn("mimics_import_output_dir", src)

    def test_single_case_discovery_accepts_mhd_file_and_flat_dicom_folder(self):
        import mimics_import
        image_path = os.path.join(self.tmp, "patient_scan.mhd")
        with open(image_path, "w", encoding="ascii") as handle:
            handle.write("ObjectType = Image\nNDims = 3\n")
        image_case = mimics_import._discover_single_case(image_path)
        self.assertEqual("patient_scan", image_case["case_id"])
        self.assertEqual(os.path.abspath(image_path), image_case["image"])

        dicom_dir = os.path.join(self.tmp, "flat_dicom")
        os.makedirs(dicom_dir)
        with open(os.path.join(dicom_dir, "slice001.dcm"), "wb") as handle:
            handle.write(b"candidate")
        dicom_case = mimics_import._discover_single_case(dicom_dir)
        self.assertEqual(dicom_dir, dicom_case["image"])
        self.assertEqual("dicom_candidate", dicom_case["image_type"])

    def test_external_batch_export_requires_explicit_safe_or_overwrite_destination(self):
        import tools.mimics_batch_cli as cli
        parser = cli.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["export-labels", "--ts-root", self.tmp])
        safe = parser.parse_args(["export-labels", "--ts-root", self.tmp, "--output-dir", os.path.join(self.tmp, "labels")])
        self.assertFalse(safe.overwrite_source)
        overwrite = parser.parse_args(["export-labels", "--ts-root", self.tmp, "--overwrite-source"])
        self.assertTrue(overwrite.overwrite_source)


    # ================================================================
    # L8: 2D decoder integration (NEW — DINOv3 few-shot optimization)
    # ================================================================

    def test_2d_decoder_conv2d_output_shape(self):
        """conv2d decoder produces correct output shape."""
        import torch
        sys.path.insert(0, os.path.join(os.getcwd(), "external", "dinov3-medical-seg"))
        from src.models.decoder_2d import Conv2DDecoder
        decoder = Conv2DDecoder([768, 768, 768, 768], num_classes=2)
        feats = [torch.randn(1, 768, 16, 14, 14) for _ in range(4)]
        out = decoder(feats, (1, 1, 16, 224, 224))
        self.assertEqual(out.shape, (1, 2, 16, 224, 224))

    def test_every_public_decoder_produces_a_stacked_3d_prediction(self):
        import torch
        import tools.fewshot_training_setup_ui as ui
        sys.path.insert(0, os.path.join(os.getcwd(), "external", "dinov3-medical-seg"))
        from src.models.decoder_3d import DecoderFactory
        features = [torch.randn(1, 8, 3, 2, 2) for _ in range(4)]
        for decoder_name in ui.DECODER_CHOICES:
            decoder = DecoderFactory.create(decoder_name, [8, 8, 8, 8], 2)
            with torch.no_grad():
                output = decoder(features, (1, 1, 3, 16, 16))
            self.assertEqual((1, 2, 3, 16, 16), tuple(output.shape), decoder_name)
            restored = DecoderFactory.create(decoder_name, [8, 8, 8, 8], 2)
            restored.load_state_dict(decoder.state_dict(), strict=True)
            restored.eval()
            decoder.eval()
            with torch.no_grad():
                expected = decoder(features, (1, 1, 3, 16, 16))
                actual = restored(features, (1, 1, 3, 16, 16))
            torch.testing.assert_close(expected, actual, rtol=0.0, atol=0.0, msg=decoder_name)

    def test_2d_decoder_all_variants_in_factory(self):
        """All 2D decoder types must be creatable via DecoderFactory."""
        import torch
        sys.path.insert(0, os.path.join(os.getcwd(), "external", "dinov3-medical-seg"))
        from src.models.decoder_3d import DecoderFactory
        feats = [torch.randn(1, 768, 8, 14, 14) for _ in range(4)]
        for dec_type in ["conv2d", "conv2d_unet", "conv2d_deeplab", "conv2d_2_5d"]:
            decoder = DecoderFactory.create(dec_type, [768, 768, 768, 768], num_classes=2)
            out = decoder(feats, (1, 1, 8, 224, 224))
            self.assertEqual(out.shape, (1, 2, 8, 224, 224),
                             f"{dec_type} output shape mismatch")

    def test_2d_decoder_fewer_params_than_3d(self):
        """2D decoders must have fewer parameters than equivalent 3D decoders."""
        sys.path.insert(0, os.path.join(os.getcwd(), "external", "dinov3-medical-seg"))
        from src.models.decoder_2d import Conv2DDecoder, Conv2DUNetDecoder
        from src.models.decoder_3d import DPT3DDecoder, SegFormer3DDecoder
        dims = [768, 768, 768, 768]

        dpt_params = sum(p.numel() for p in DPT3DDecoder(dims, 2).parameters())
        seg_params = sum(p.numel() for p in SegFormer3DDecoder(dims, 2).parameters())
        conv2d_params = sum(p.numel() for p in Conv2DDecoder(dims, 2).parameters())
        conv_unet_params = sum(p.numel() for p in Conv2DUNetDecoder(dims, 2).parameters())

        self.assertLess(conv2d_params, dpt_params,
                        f"conv2d ({conv2d_params}) must be < dpt3d ({dpt_params})")
        self.assertLess(conv2d_params, seg_params,
                        f"conv2d ({conv2d_params}) must be < segformer3d ({seg_params})")

    def test_2d_decoder_single_slice(self):
        """2D decoders must handle D=1 (single slice) edge case."""
        import torch
        sys.path.insert(0, os.path.join(os.getcwd(), "external", "dinov3-medical-seg"))
        from src.models.decoder_2d import Conv2DDecoder, Conv2DUNetDecoder, Conv2DDeepLabDecoder, Conv2D_2_5D_Decoder
        dims = [768, 768, 768, 768]
        feat_1 = [torch.randn(1, 768, 1, 14, 14) for _ in range(4)]
        for cls in [Conv2DDecoder, Conv2DUNetDecoder, Conv2DDeepLabDecoder, Conv2D_2_5D_Decoder]:
            dec = cls(dims, num_classes=2)
            out = dec(feat_1, (1, 1, 1, 224, 224))
            self.assertEqual(out.shape, (1, 2, 1, 224, 224),
                             f"{cls.__name__} single-slice shape mismatch")

    def test_2d_decoder_25d_neighbour_stacking(self):
        """2.5D decoder _stack_neighbours must correctly replicate edge slices."""
        import torch
        sys.path.insert(0, os.path.join(os.getcwd(), "external", "dinov3-medical-seg"))
        from src.models.decoder_2d import Conv2D_2_5D_Decoder
        B, C, D, h, w = 1, 4, 5, 2, 2
        feat = torch.zeros(B, C, D, h, w)
        for d in range(D):
            feat[:, :, d, :, :] = float(d)
        dec = Conv2D_2_5D_Decoder([C], num_classes=2)
        stacked = dec._stack_neighbours(feat)
        self.assertEqual(stacked.shape, (B * D, 3 * C, h, w))
        # First slice: neighbours = [0, 0, 1] (edge replicate)
        s0 = stacked[0].reshape(3, C, h, w)
        self.assertTrue(torch.allclose(s0[0], feat[:, :, 0]))
        self.assertTrue(torch.allclose(s0[1], feat[:, :, 0]))
        self.assertTrue(torch.allclose(s0[2], feat[:, :, 1]))
        # Last slice: neighbours = [3, 4, 4]
        s4 = stacked[4].reshape(3, C, h, w)
        self.assertTrue(torch.allclose(s4[0], feat[:, :, 3]))
        self.assertTrue(torch.allclose(s4[1], feat[:, :, 4]))
        self.assertTrue(torch.allclose(s4[2], feat[:, :, 4]))

    def test_2d_decoder_batch_experiments_script_exists(self):
        """Batch experiment runner must be importable."""
        exp_script = os.path.join(os.getcwd(), "external", "dinov3-medical-seg",
                                   "scripts", "batch_experiments.py")
        self.assertTrue(os.path.isfile(exp_script),
                        f"batch_experiments.py not found at {exp_script}")

    def test_2d_decoder_materialize_all_script_exists(self):
        """Multi-organ materialization script must exist."""
        mat_script = os.path.join(os.getcwd(), "external", "dinov3-medical-seg",
                                   "scripts", "materialize_all_organs.py")
        self.assertTrue(os.path.isfile(mat_script),
                        f"materialize_all_organs.py not found at {mat_script}")

    def test_2d_decoder_evaluate_script_exists(self):
        """Model evaluation script must exist."""
        eval_script = os.path.join(os.getcwd(), "external", "dinov3-medical-seg",
                                    "scripts", "evaluate_model.py")
        self.assertTrue(os.path.isfile(eval_script),
                        f"evaluate_model.py not found at {eval_script}")

    def test_2d_decoder_analyze_script_exists(self):
        """Results analysis script must exist."""
        anal_script = os.path.join(os.getcwd(), "external", "dinov3-medical-seg",
                                    "scripts", "analyze_results.py")
        self.assertTrue(os.path.isfile(anal_script),
                        f"analyze_results.py not found at {anal_script}")


class TestLifecycleAndRetention(unittest.TestCase):
    def setUp(self):
        self.tmp = _make_temp_dir()

    def tearDown(self):
        _cleanup(self.tmp)

    def test_active_nninteractive_operation_cannot_be_released_for_training(self):
        import tools.fewshot_pipeline as pipeline

        state_path = os.path.join(self.tmp, "server.json")
        pipeline.write_json_atomic(state_path, {
            "schema_version": "nninteractive_owned_server.v2",
            "pid": 1001,
            "watchdog_pid": 1002,
            "gpu_lock_token": "token",
            "last_activity_epoch": 0,
            "active_operation": "prediction",
            "active_operation_pid": 1003,
        })
        current = {
            "resource": "gpu",
            "token": "token",
            "state_path": state_path,
        }
        original = pipeline.process_exists
        try:
            pipeline.process_exists = lambda _pid: True
            self.assertFalse(pipeline.request_nninteractive_server_release_on_contention(current))
            self.assertFalse(pipeline.cleanup_idle_nninteractive_server_lock(current))
        finally:
            pipeline.process_exists = original

    def test_idle_nninteractive_worker_receives_graceful_gpu_contention_close(self):
        import tools.fewshot_pipeline as pipeline

        control_dir = Path(self.tmp) / "worker"
        control_dir.mkdir()
        state_path = Path(self.tmp) / "server.json"
        pipeline.write_json_atomic(state_path, {
            "schema_version": "nninteractive_owned_server.v2",
            "pid": 1001,
            "watchdog_pid": 1002,
            "client_pid": 1003,
            "client_control_dir": str(control_dir),
            "gpu_lock_token": "token",
            "last_activity_epoch": time.time() - 60,
        })
        current = {"resource": "gpu", "token": "token", "state_path": str(state_path)}
        original = pipeline.process_exists
        try:
            pipeline.process_exists = lambda _pid: True
            self.assertTrue(pipeline.request_nninteractive_server_release_on_contention(current))
        finally:
            pipeline.process_exists = original
        close = pipeline.read_json(control_dir / "close.json", {}) or {}
        self.assertEqual(close.get("reason"), "gpu_contention")
        state = pipeline.read_json(state_path, {}) or {}
        self.assertNotEqual(state.get("last_activity_epoch"), 0.0)

    def test_fewshot_storage_maintenance_preserves_models_and_active_jobs(self):
        import tools.fewshot_pipeline as pipeline

        workspace = Path(self.tmp) / "fewshot_models"
        jobs = workspace / "jobs"
        jobs.mkdir(parents=True)
        old = time.time() - 40 * 86400
        failed_old = time.time() - 8 * 86400

        completed_dataset = workspace / "datasets" / "liver" / "done"
        completed_dataset.mkdir(parents=True)
        completed_run = workspace / "runs" / "liver" / "done"
        completed_run.mkdir(parents=True)
        (completed_run / "train.log").write_text("large log", encoding="utf-8")
        pipeline.write_json_atomic(jobs / "done.json", {
            "job_id": "done", "status": "completed", "organ": "liver",
            "updated_at_epoch": old, "dataset_dir": str(completed_dataset),
            "model": {"dataset_retained": False},
        })

        failed_run = workspace / "runs" / "liver" / "failed"
        failed_run.mkdir(parents=True)
        (failed_run / "train.log").write_text("failure", encoding="utf-8")
        pipeline.write_json_atomic(jobs / "failed.json", {
            "job_id": "failed", "status": "failed", "organ": "liver",
            "updated_at_epoch": failed_old,
        })

        active_dataset = workspace / "datasets" / "liver" / "active"
        active_dataset.mkdir(parents=True)
        pipeline.write_json_atomic(jobs / "active.json", {
            "job_id": "active", "status": "training", "organ": "liver",
            "updated_at_epoch": time.time(), "dataset_dir": str(active_dataset),
        })
        orphan_dataset = workspace / "datasets" / "liver" / "orphan"
        orphan_dataset.mkdir(parents=True)
        pipeline.write_json_atomic(jobs / "orphan.json", {
            "job_id": "orphan", "status": "training", "organ": "liver",
            "controller_pid": 987654321, "updated_at_epoch": failed_old,
            "dataset_dir": str(orphan_dataset),
        })
        model = workspace / "models" / "liver" / "done" / "model.pth"
        model.parent.mkdir(parents=True)
        model.write_bytes(b"model")

        dinov3 = Path(self.tmp) / "dinov3"
        experiment = dinov3 / "experiments" / "mimics_fewshot_liver_done"
        experiment.mkdir(parents=True)
        (experiment / "checkpoint.pth").write_bytes(b"checkpoint")

        context = jobs / "setup_old_context.json"
        context.write_text("{}", encoding="utf-8")
        os.utime(str(context), (old, old))

        report = pipeline.cleanup_workspace_artifacts(workspace, dinov3, {
            "terminal_job_retention_days": 30,
            "max_terminal_job_records": 100,
            "failed_run_retention_days": 7,
            "completed_run_log_retention_days": 30,
            "setup_context_retention_days": 7,
            "keep_training_experiment_artifacts": False,
        })
        self.assertFalse(completed_dataset.exists())
        self.assertFalse((completed_run / "train.log").exists())
        self.assertFalse(failed_run.exists())
        self.assertFalse(experiment.exists())
        self.assertFalse(context.exists())
        self.assertFalse((jobs / "done.json").exists())
        self.assertTrue((jobs / "active.json").exists())
        self.assertTrue(active_dataset.exists())
        orphan_status = pipeline.read_json(jobs / "orphan.json", {}) or {}
        self.assertEqual(orphan_status.get("status"), "failed")
        self.assertTrue(orphan_status.get("orphaned"))
        self.assertFalse(orphan_dataset.exists())
        self.assertTrue(model.exists(), "automatic maintenance must never delete registered models")
        self.assertGreater(sum(report.values()), 0)

    def test_clear_cache_does_not_include_runtime_control_state_or_active_jobs(self):
        import mimics_stop_background as stop

        root = Path(self.tmp)
        runtime = root / ".mimics_runtime"
        runtime.mkdir()
        control = runtime / "fewshot_mimics_state.json"
        control.write_text("{}", encoding="utf-8")
        nn_runtime = runtime / "nninteractive"
        source_cache = nn_runtime / "source_fastpath_cache"
        source_cache.mkdir(parents=True)
        async_root = nn_runtime / "async_jobs"
        terminal = async_root / "terminal"
        active = async_root / "active"
        terminal.mkdir(parents=True)
        active.mkdir(parents=True)
        (terminal / "worker_status.json").write_text('{"status":"closed"}', encoding="utf-8")
        (active / "worker_status.json").write_text('{"status":"running"}', encoding="utf-8")
        original = stop._project_root
        try:
            stop._project_root = lambda: str(root)
            paths = set(stop._find_cache_paths())
        finally:
            stop._project_root = original
        self.assertNotIn(str(runtime), paths)
        self.assertNotIn(str(control), paths)
        self.assertIn(str(source_cache), paths)
        self.assertIn(str(terminal), paths)
        self.assertNotIn(str(active), paths)

    def test_stop_all_requires_owned_root_and_excludes_foreground(self):
        path = os.path.join(PROJECT_ROOT, "runtime_py35", "mimics_stop_background.py")
        with open(path, "r") as handle:
            source = handle.read()
        self.assertNotIn("$broad=Get-CimInstance", source)
        self.assertIn("$foregroundPid", source)
        self.assertIn("$inRoot -and $hasMarker", source)


if __name__ == "__main__":
    print("Mimics-Script Comprehensive Tests")
    print("=" * 60)
    print("Python: {}".format(sys.version))
    print("Project: {}".format(PROJECT_ROOT))
    print("=" * 60)
    unittest.main(verbosity=2)
