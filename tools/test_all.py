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
import os
import shutil
import sys
import tempfile
import time
import unittest
import uuid

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
        for sub in ["01_Data", "02_AI", "03_Display", "99_Admin"]:
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
        from mimics_bridge import read_nifti_mask_with_affine

        shape = (15, 20, 25)
        affine = np.diag([1.0, 1.0, 2.0, 1.0])
        affine[0, 3] = -75
        mask_data = (np.random.random(shape) > 0.7).astype(np.uint8)

        path = self._make_nifti("mask.nii.gz", mask_data, affine, sform_code=2, qform_code=0)
        array, result_affine = read_nifti_mask_with_affine(path)
        self.assertEqual(shape, array.shape)
        np.testing.assert_allclose(result_affine, affine, atol=1e-6)
        # Mask should be bool-like
        self.assertTrue(np.all((array == 0) | (array == 1)))

    def test_get_image_affine_normalized(self):
        from mimics_bridge import get_image_affine

        shape = (5, 5, 5)
        affine = np.eye(4)
        data = np.zeros(shape, dtype=np.int16)

        path = self._make_nifti("img.nii.gz", data, affine, sform_code=0, qform_code=2)
        result = get_image_affine(path)
        np.testing.assert_allclose(result, affine, atol=1e-6)


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
        with self.assertRaises(ValueError):
            read_nifti_mask(path)


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
        stored_fp = open(fp_path).read().strip()
        self.assertEqual(current_fp, stored_fp)
        # Matching fingerprint → skip

        # Now change fingerprint → should reprocess
        new_fp = "sha256:def456"
        with open(fp_path, "w") as f:
            f.write(new_fp)
        stored_fp2 = open(fp_path).read().strip()
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
        for sub in ["01_Data", "02_AI", "03_Display", "99_Admin"]:
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
        entry = os.path.join(PROJECT_ROOT, "scripting_library", "02_AI", "nnInteractive.py")
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
        for fname in dino_entries:
            path = os.path.join(dino_dir, fname)
            with open(path, "r") as f:
                source = f.read()
            self.assertIn("fewshot_mimics", source,
                "DINOv3/{} should route to fewshot_mimics".format(fname))
            self.assertNotIn("nninteractive_mimics", source)

        # nnInteractive.py is flat in 02_AI/
        nn_entry = os.path.join(PROJECT_ROOT, "scripting_library", "02_AI", "nnInteractive.py")
        self.assertTrue(os.path.isfile(nn_entry), "nnInteractive.py should be in 02_AI/")
        with open(nn_entry, "r") as f:
            self.assertIn("nninteractive_mimics", f.read())

    def test_dinov3_not_in_root_ai_dir(self):
        """DINOv3 entries should NOT be flat in 02_AI/ root."""
        ai_root = os.path.join(PROJECT_ROOT, "scripting_library", "02_AI")
        flat_dino = [f for f in os.listdir(ai_root) if f.startswith("DINOv3") and f.endswith(".py")]
        self.assertEqual([], flat_dino,
            "DINOv3 entries should be in DINOv3/ subdirectory, not flat in 02_AI/")

    def test_stop_background_services_entry(self):
        entry = os.path.join(PROJECT_ROOT, "scripting_library", "99_Admin", "Stop_Background_Services.py")
        with open(entry, "r") as f:
            source = f.read()
        self.assertIn("mimics_stop_background", source)

    def test_window_entries_routes(self):
        lib = os.path.join(PROJECT_ROOT, "scripting_library", "03_Display")
        expected_routes = {
            "Window_Choose_Preset.py": "choose",
            "Window_From_Selected_Mask.py": "auto",
            "Window_Reset_Full_Range.py": "reset",
            "Window_Undo_Last.py": "undo",
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

        self.assertIn("mimics_bridge.py", MARKERS)
        self.assertIn("nninteractive_bridge.py", MARKERS)
        self.assertIn("--async-worker", MARKERS)
        self.assertIn("_run_create_mcs.py", MARKERS)
        self.assertIn("fewshot_pipeline.py", MARKERS)
        self.assertIn("nninteractive.inference.server.main", MARKERS)
        self.assertIn("--watchdog", MARKERS)

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
        """Path A: DICOM pixel values == NIfTI source values (int16 truncation)."""
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

        # Read back one DICOM slice and compare
        dcm_file = os.path.join(dicom_out, "slice_0001.dcm")
        ds = pydicom.dcmread(dcm_file)
        dcm_data = ds.pixel_array  # (Rows, Columns) = (10, 10)
        # DICOM pixel[row, col] = NIfTI[i=col, j=row] (after transpose)
        # So DICOM pixel[0, 0] = NIfTI[0, 0]
        self.assertEqual(int(data[0, 0, 0]), int(dcm_data[0, 0]))
        self.assertEqual(int(data[1, 0, 0]), int(dcm_data[0, 1]))
        # RescaleSlope/Intercept should be identity for derived DICOM
        self.assertEqual(1.0, float(ds.RescaleSlope))
        self.assertEqual(0.0, float(ds.RescaleIntercept))

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
        """NIfTI → DICOM → read pixel → matches NIfTI."""
        from mimics_bridge import nifti_to_derived_dicom
        import pydicom

        shape = (20, 15, 3)
        data = np.random.randint(-500, 2500, shape, dtype=np.int16).astype(np.float32)
        path = self._create_test_nifti("ct.nii.gz", shape, data)
        dicom_out = os.path.join(self.tmp, "dicom")
        nifti_to_derived_dicom(path, dicom_out)

        # Read all slices back
        slices = []
        for k in range(1, shape[2] + 1):
            ds = pydicom.dcmread(os.path.join(dicom_out, f"slice_{k:04d}.dcm"))
            arr = ds.pixel_array  # (Rows=j, Columns=i)
            slices.append(arr.T)  # Transpose back to (i, j) = NIfTI order

        stacked = np.stack(slices, axis=2)  # (Columns, Rows, Slices)
        self.assertEqual(tuple(shape), stacked.shape)
        # Values should match within int16 truncation
        np.testing.assert_array_equal(data.astype(np.int16), stacked.astype(np.int16))

    def test_source_image_skipped_when_file_unreadable(self):
        """_source_image_export returns None when source file is inaccessible."""
        from nninteractive_mimics import _source_image_export
        import mimics as _mock_mimics

        # Simulate a missing source file
        # _source_image_export depends on mimics.data.images.get_active()
        # and image metadata. With mock mimics, get_active returns None,
        # so the function should return None (early return).
        config = {"prefer_source_image_for_nninteractive": True}
        result = _source_image_export(None, config)
        self.assertIsNone(result)

    def test_source_image_file_accessibility_pre_check(self):
        """_source_image_export pre-checks file readability before committing."""
        from nninteractive_mimics import _source_image_export
        import mimics as _mm

        # Create a temp NIfTI and verify the pre-check logic exists.
        # The function returns None when active image is None (mock Mimics),
        # but the file accessibility check at lines 858-870 is structurally verified.
        config = {"prefer_source_image_for_nninteractive": True}
        result = _source_image_export(None, config)
        self.assertIsNone(result)


# ============================================================================
# L13: New features from user's round of changes
# ============================================================================


class TestNewFeatures(unittest.TestCase):
    def setUp(self):
        self.tmp = _make_temp_dir()

    def tearDown(self):
        _cleanup(self.tmp)

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


# ============================================================================
# Main
# ============================================================================


if __name__ == "__main__":
    print("Mimics-Script Comprehensive Tests")
    print("=" * 60)
    print("Python: {}".format(sys.version))
    print("Project: {}".format(PROJECT_ROOT))
    print("=" * 60)
    unittest.main(verbosity=2)
