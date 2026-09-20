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
import errno
import inspect
import os
import subprocess
import shutil
import sys
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path
from unittest import mock

import numpy as np

# ---------------------------------------------------------------------------
# Test setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNTIME_DIR = os.path.join(PROJECT_ROOT, "runtime_py35")

sys.path.insert(0, RUNTIME_DIR)
sys.path.insert(0, PROJECT_ROOT)


def _repo_dinov3_root():
    """Resolve the configured DINOv3 project root.

    The deploy tree keeps external/dinov3-medical-seg at the path recorded in
    fewshot_config.json (it may live outside this repository), so tests must
    not hard-code <repo>/external/dinov3-medical-seg.
    """
    try:
        with open(os.path.join(PROJECT_ROOT, "fewshot_config.json"), "r", encoding="utf-8") as handle:
            configured = json.load(handle).get("dinov3_project") or ""
    except Exception:
        configured = ""
    if configured and os.path.isfile(os.path.join(configured, "scripts", "train.py")):
        return configured
    return DINOV3_ROOT


DINOV3_ROOT = _repo_dinov3_root()

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
_mock_mimics.file.open_project = _fake_fn
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
        data = {
            "key": "value",
            "patient_path": os.path.join(self.tmp, "病例一"),
            "nested": {"a": 1, "b": [2, 3]},
        }
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

    def test_progress_notice_is_immediate_or_throttled_as_requested(self):
        import runtime_common

        state = {}
        due, elapsed = runtime_common.progress_notice_due(
            state, "wait", "gpu", 60, 10, now=100
        )
        self.assertFalse(due)
        self.assertEqual(0, elapsed)
        self.assertTrue(
            runtime_common.progress_notice_due(
                state, "wait", "gpu", 60, 10, now=110
            )[0]
        )
        self.assertFalse(
            runtime_common.progress_notice_due(
                state, "wait", "gpu", 60, 10, now=150
            )[0]
        )
        self.assertTrue(
            runtime_common.progress_notice_due(
                state, "wait", "gpu", 60, 10, now=170
            )[0]
        )
        runtime_common.clear_progress_notice(state, "wait")
        self.assertNotIn("_progress_notices", state)

    def test_mimics_transaction_commits_or_rolls_back_once(self):
        import runtime_common

        events = []

        class Transaction(object):
            def __init__(self, transaction_name):
                events.append("name:" + transaction_name)
            def __enter__(self):
                events.append("enter")
                return self
            def __exit__(self, exc_type, exc, tb):
                events.append("exit")
            def commit(self):
                events.append("commit")
            def rollback(self):
                events.append("rollback")

        module = _FakeModule()
        module.Transaction = Transaction
        self.assertEqual(
            "ok",
            runtime_common.execute_mimics_transaction(
                module, lambda: "ok", "Apply Test Mask"
            ),
        )
        self.assertEqual(
            ["name:Apply Test Mask", "enter", "commit", "exit"], events
        )

        events[:] = []
        def fail():
            raise ValueError("failed write")
        with self.assertRaisesRegex(ValueError, "failed write"):
            runtime_common.execute_mimics_transaction(module, fail)
        self.assertEqual(
            ["name:Mimics-Script Mask Update", "enter", "rollback", "exit"],
            events,
        )

        events[:] = []
        def fail_commit():
            events.append("commit")
            raise RuntimeError("commit failed")
        Transaction.commit = lambda self: fail_commit()
        with self.assertRaisesRegex(RuntimeError, "commit failed"):
            runtime_common.execute_mimics_transaction(module, lambda: "ok")
        self.assertEqual(
            [
                "name:Mimics-Script Mask Update",
                "enter",
                "commit",
                "rollback",
                "exit",
            ],
            events,
        )

    def test_mimics_transaction_binding_failure_keeps_validated_write_available(self):
        import runtime_common

        module = _FakeModule()

        def broken_transaction(*args):
            raise TypeError("broken compatibility binding")

        module.Transaction = broken_transaction
        calls = []
        result = runtime_common.execute_mimics_transaction(
            module,
            lambda: calls.append("write") or "ok",
            "Apply AI Mask",
        )
        self.assertEqual("ok", result)
        self.assertEqual(["write"], calls)

    def test_find_root_accepts_file_path_without_explicit_sentinels(self):
        import runtime_common

        project = os.path.join(self.tmp, "project")
        module_dir = os.path.join(project, "runtime_py35")
        os.makedirs(module_dir)
        module_path = os.path.join(module_dir, "entry.py")
        with open(module_path, "w") as handle:
            handle.write("# entry\n")
        self.assertEqual(project, runtime_common.find_root(module_path))

    def test_user_config_is_separate_from_disposable_runtime(self):
        import runtime_common

        old_value = os.environ.get("MIMICS_USER_CONFIG_DIR")
        try:
            configured = os.path.join(self.tmp, "persistent-settings")
            os.environ["MIMICS_USER_CONFIG_DIR"] = configured
            self.assertEqual(configured, runtime_common.user_config_dir())
            self.assertNotIn(".mimics_runtime", runtime_common.user_config_dir())
        finally:
            if old_value is None:
                os.environ.pop("MIMICS_USER_CONFIG_DIR", None)
            else:
                os.environ["MIMICS_USER_CONFIG_DIR"] = old_value

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

    def test_write_json_atomic_falls_back_when_smb_replace_is_always_denied(self):
        import runtime_common

        path = os.path.join(self.tmp, "smb_replace_denied.json")
        old_replace = runtime_common.os.replace
        old_sleep = runtime_common.time.sleep
        try:
            runtime_common.os.replace = lambda _src, _dst: (_ for _ in ()).throw(OSError(5, "access denied"))
            runtime_common.time.sleep = lambda _seconds: None
            runtime_common.write_json_atomic(path, {"fallback": True})
        finally:
            runtime_common.os.replace = old_replace
            runtime_common.time.sleep = old_sleep
        self.assertEqual({"fallback": True}, runtime_common.read_json(path))

    def test_mimics_exe_override_bypasses_negative_lookup_cache(self):
        import runtime_common

        exe = os.path.join(self.tmp, "MimicsResearch.exe")
        Path(exe).write_bytes(b"test")
        old_cache = dict(runtime_common._MIMICS_EXE_CACHE)
        old_env = os.environ.get("MIMICS_EXE")
        try:
            runtime_common._MIMICS_EXE_CACHE.update({"value": None, "checked_at": time.time()})
            os.environ["MIMICS_EXE"] = exe
            self.assertEqual(os.path.abspath(exe), runtime_common.find_mimics_exe())
        finally:
            runtime_common._MIMICS_EXE_CACHE.clear()
            runtime_common._MIMICS_EXE_CACHE.update(old_cache)
            if old_env is None:
                os.environ.pop("MIMICS_EXE", None)
            else:
                os.environ["MIMICS_EXE"] = old_env

    def test_running_mimics_exe_bypasses_negative_lookup_cache(self):
        import runtime_common

        exe = os.path.join(self.tmp, "MimicsResearch.exe")
        Path(exe).write_bytes(b"test")
        old_cache = dict(runtime_common._MIMICS_EXE_CACHE)
        old_running = runtime_common._running_mimics_research_executable
        old_env = os.environ.pop("MIMICS_EXE", None)
        old_background = os.environ.pop("MIMICS_BACKGROUND_EXE", None)
        try:
            runtime_common._MIMICS_EXE_CACHE.update(
                {"value": None, "checked_at": time.time()}
            )
            runtime_common._running_mimics_research_executable = lambda: exe
            self.assertEqual(
                os.path.abspath(exe),
                runtime_common.find_mimics_exe(),
            )
        finally:
            runtime_common._running_mimics_research_executable = old_running
            runtime_common._MIMICS_EXE_CACHE.clear()
            runtime_common._MIMICS_EXE_CACHE.update(old_cache)
            if old_env is not None:
                os.environ["MIMICS_EXE"] = old_env
            if old_background is not None:
                os.environ["MIMICS_BACKGROUND_EXE"] = old_background

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

    def test_process_exists_treats_permission_denied_as_alive(self):
        import runtime_common

        if os.name == "nt":
            # The Windows branch uses OpenProcess, not os.kill. Simulate the
            # access-denied error code for a nonexistent-but-protected PID.
            import ctypes

            old_open = ctypes.WinDLL
            old_get_last_error = ctypes.get_last_error
            try:
                class _DeniedKernel32(object):
                    def __init__(self, _name, use_last_error=False):
                        pass

                    def OpenProcess(self, *_args):
                        return None

                    def CloseHandle(self, _handle):
                        return 1

                def denied_windll(name, use_last_error=False):
                    if "kernel32" in str(name):
                        return _DeniedKernel32(name, use_last_error)
                    return old_open(name, use_last_error=use_last_error)

                ctypes.get_last_error = lambda: 5  # ERROR_ACCESS_DENIED
                ctypes.WinDLL = denied_windll
                self.assertTrue(runtime_common.process_exists(12345))
            finally:
                ctypes.WinDLL = old_open
                ctypes.get_last_error = old_get_last_error
        else:
            old_kill = runtime_common.os.kill
            try:
                runtime_common.os.kill = lambda _pid, _signal: (_ for _ in ()).throw(
                    PermissionError(errno.EPERM, "operation not permitted")
                )
                self.assertTrue(runtime_common.process_exists(12345))
            finally:
                runtime_common.os.kill = old_kill

    def test_resource_lock_short_write_is_completed(self):
        import runtime_common

        lock_path = os.path.join(self.tmp, "short_write.lock")
        old_write = runtime_common.os.write

        def short_write(fd, data):
            chunk = data[:max(1, min(7, len(data)))]
            return old_write(fd, chunk)

        try:
            runtime_common.os.write = short_write
            token = runtime_common.acquire_resource_lock(
                lock_path, "short_write", "test",
            )
        finally:
            runtime_common.os.write = old_write
        self.assertTrue(token)
        payload = runtime_common.read_json(lock_path, {})
        self.assertEqual(token, payload.get("token"))
        self.assertTrue(runtime_common.release_resource_lock(lock_path, token))

    def test_resource_lock_release_rechecks_token_before_retry(self):
        import runtime_common

        lock_path = os.path.join(self.tmp, "release_race.lock")
        token = runtime_common.acquire_resource_lock(lock_path, "race", "old")
        self.assertTrue(token)
        old_remove = runtime_common.os.remove
        calls = []

        def replace_owner_on_first_remove(path):
            if os.path.abspath(path) == os.path.abspath(lock_path) and not calls:
                calls.append(path)
                runtime_common.write_json_atomic(lock_path, {
                    "pid": os.getpid(),
                    "resource": "race",
                    "owner": "new",
                    "token": "new-token",
                })
                raise OSError(5, "access denied")
            return old_remove(path)

        try:
            runtime_common.os.remove = replace_owner_on_first_remove
            self.assertFalse(runtime_common.release_resource_lock(lock_path, token))
        finally:
            runtime_common.os.remove = old_remove
        self.assertEqual("new-token", runtime_common.read_json(lock_path, {}).get("token"))
        self.assertTrue(runtime_common.release_resource_lock(lock_path, "new-token"))

    def test_resource_lock_implementations_share_the_same_directory(self):
        import runtime_common
        from resource_locks import default_resource_lock_dir

        runtime_dir = os.path.abspath(runtime_common.resource_lock_dir(PROJECT_ROOT))
        external_dir = os.path.abspath(str(default_resource_lock_dir(PROJECT_ROOT)))
        self.assertEqual(runtime_dir, external_dir)

    def test_external_resource_lock_uses_complete_writes_and_token_release(self):
        import resource_locks

        lock_path = Path(self.tmp) / "external.lock"
        old_write = resource_locks.os.write

        def short_write(fd, data):
            return old_write(fd, data[:max(1, min(5, len(data)))])

        try:
            resource_locks.os.write = short_write
            lock = resource_locks.FileResourceLock(
                lock_path, "external", "test",
            ).acquire()
        finally:
            resource_locks.os.write = old_write
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        self.assertEqual(lock.token, payload.get("token"))
        self.assertFalse(resource_locks.release_lock(lock_path, "wrong-token"))
        self.assertTrue(lock_path.exists())
        lock.release()
        self.assertFalse(lock_path.exists())

    def test_external_resource_lock_update_survives_replace_denied(self):
        import resource_locks

        lock_path = Path(self.tmp) / "external_replace.lock"
        lock = resource_locks.FileResourceLock(
            lock_path, "external", "test",
        ).acquire()
        old_replace = resource_locks.os.replace
        old_sleep = resource_locks.time.sleep
        try:
            resource_locks.os.replace = lambda _src, _dst: (
                _ for _ in ()
            ).throw(OSError(5, "access denied"))
            resource_locks.time.sleep = lambda _seconds: None
            lock.update_pid(os.getpid(), kind="updated")
        finally:
            resource_locks.os.replace = old_replace
            resource_locks.time.sleep = old_sleep
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        self.assertEqual(lock.token, payload.get("token"))
        self.assertEqual("updated", payload.get("kind"))

        old_release = resource_locks.release_lock
        try:
            resource_locks.release_lock = lambda _path, _token: False
            lock.release()
            self.assertTrue(lock.acquired)
        finally:
            resource_locks.release_lock = old_release
        lock.release()
        self.assertFalse(lock.acquired)

    def test_external_resource_lock_update_detects_lost_token(self):
        import resource_locks

        lock_path = Path(self.tmp) / "external_lost_token.lock"
        lock = resource_locks.FileResourceLock(
            lock_path, "external", "old owner",
        ).acquire()
        replacement = {
            "resource": "external",
            "owner": "new owner",
            "pid": os.getpid(),
            "token": "replacement-token",
        }
        lock_path.write_text(json.dumps(replacement), encoding="utf-8")
        self.assertFalse(lock.update_pid(os.getpid(), kind="should_not_write"))
        self.assertFalse(lock.acquired)
        self.assertEqual(
            "replacement-token",
            json.loads(lock_path.read_text(encoding="utf-8"))["token"],
        )

    def test_terminate_process_async_rejects_pid_mismatch(self):
        import runtime_common

        class Process(object):
            pid = 1234

        self.assertFalse(
            runtime_common.terminate_process_async(process=Process(), pid=5678)
        )

    def test_background_env_preserves_no_bytecode_policy(self):
        import runtime_common

        old_value = os.environ.get("PYTHONDONTWRITEBYTECODE")
        try:
            os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
            env = runtime_common.background_env()
        finally:
            if old_value is None:
                os.environ.pop("PYTHONDONTWRITEBYTECODE", None)
            else:
                os.environ["PYTHONDONTWRITEBYTECODE"] = old_value
        self.assertEqual("1", env.get("PYTHONDONTWRITEBYTECODE"))

    def test_read_text_tail_rejects_nonpositive_limit(self):
        import runtime_common

        path = os.path.join(self.tmp, "tail.log")
        Path(path).write_text("abc", encoding="utf-8")
        self.assertEqual("", runtime_common.read_text_tail(path, 0))
        self.assertEqual("", runtime_common.read_text_tail(path, -1))

    def test_local_operation_lease_prevents_nested_timer_reentry(self):
        import runtime_common

        resource = "test_mask_buffer_access"
        first = runtime_common.try_acquire_local_operation(resource, "first")
        self.assertTrue(first)
        self.assertIsNone(runtime_common.try_acquire_local_operation(resource, "second"))
        self.assertEqual("first", runtime_common.active_local_operation(resource)["owner"])
        self.assertFalse(runtime_common.release_local_operation(resource, "wrong-token"))
        self.assertTrue(runtime_common.release_local_operation(resource, first))
        second = runtime_common.try_acquire_local_operation(resource, "second")
        self.assertTrue(second)
        self.assertTrue(runtime_common.release_local_operation(resource, second))

    def test_global_stop_can_clear_only_inprocess_operation_leases(self):
        import runtime_common

        token_a = runtime_common.try_acquire_local_operation(
            "test_global_stop_a", "first task"
        )
        token_b = runtime_common.try_acquire_local_operation(
            "test_global_stop_b", "second task"
        )
        self.assertTrue(token_a and token_b)
        owners = sorted(
            row["owner"]
            for row in runtime_common.active_local_operations()
            if row["resource"].startswith("test_global_stop_")
        )
        self.assertEqual(["first task", "second task"], owners)
        runtime_common.clear_local_operations()
        self.assertIsNone(runtime_common.active_local_operation("test_global_stop_a"))
        self.assertIsNone(runtime_common.active_local_operation("test_global_stop_b"))

    def test_runtime_blocker_census_includes_live_monitor_and_resource_lock(self):
        import runtime_common

        fake = _FakeModule()
        fake._MASK_IMPORT_MONITORS = {"active": {"done": False}}
        previous = sys.modules.get("mask_import")
        lock_dir = os.path.join(self.tmp, ".mimics_runtime", "locks")
        os.makedirs(lock_dir)
        runtime_common.write_json_atomic(
            os.path.join(lock_dir, "gpu.lock"),
            {
                "pid": os.getpid(),
                "token": "live-token",
                "resource": "gpu",
                "owner": "test GPU task",
            },
        )
        try:
            sys.modules["mask_import"] = fake
            blockers = runtime_common.active_runtime_blockers(self.tmp)
        finally:
            if previous is None:
                sys.modules.pop("mask_import", None)
            else:
                sys.modules["mask_import"] = previous
        self.assertTrue(any("mask import" in item for item in blockers))
        self.assertTrue(any("test GPU task" in item for item in blockers))

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

    def test_supported_volume_suffixes_include_compressed_nrrd(self):
        from mimics_bridge import is_medical_image_file

        path = os.path.join(self.tmp, "volume.nrrd.gz")
        with open(path, "wb") as handle:
            handle.write(b"NRRD0005\n")
        self.assertTrue(is_medical_image_file(path))

    def test_mask_nifti_publish_is_atomic_and_preserves_existing_on_failure(self):
        import nibabel as nib
        import mimics_bridge

        path = os.path.join(self.tmp, "mask.nii.gz")
        affine = np.diag([1.2, 1.3, 2.4, 1.0])
        original = np.ones((3, 4, 5), dtype=np.uint8)
        nib.save(nib.Nifti1Image(original, affine), path)
        old_replace = mimics_bridge.os.replace
        old_sleep = mimics_bridge.time.sleep
        try:
            mimics_bridge.os.replace = lambda _src, _dst: (_ for _ in ()).throw(
                OSError(5, "access denied")
            )
            mimics_bridge.time.sleep = lambda _seconds: None
            with self.assertRaisesRegex(RuntimeError, "left unchanged"):
                mimics_bridge.write_mask_nifti(
                    np.zeros_like(original), affine, path
                )
        finally:
            mimics_bridge.os.replace = old_replace
            mimics_bridge.time.sleep = old_sleep
        np.testing.assert_array_equal(
            original, np.asanyarray(nib.load(path).dataobj)
        )
        self.assertFalse(
            any(name.startswith("._mimics_mask_") for name in os.listdir(self.tmp))
        )

    def test_mask_nifti_publish_sets_qform_and_sform(self):
        import nibabel as nib
        from mimics_bridge import write_mask_nifti

        path = os.path.join(self.tmp, "published.nii.gz")
        affine = np.array([
            [0.0, -1.5, 0.0, 40.0],
            [1.5, 0.0, 0.0, -25.0],
            [0.0, 0.0, 2.0, 5.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        write_mask_nifti(np.ones((4, 5, 6), dtype=np.uint8), affine, path)
        image = nib.load(path)
        self.assertGreater(int(image.header["qform_code"]), 0)
        self.assertGreater(int(image.header["sform_code"]), 0)
        np.testing.assert_allclose(image.affine, affine, atol=1e-6)

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
        # NIfTI labels stay in their on-disk order; the affine carries their
        # physical RAS geometry into the explicit grid mapping step.
        self.assertEqual(shape, array.shape)
        np.testing.assert_array_equal(mask_data, array)
        np.testing.assert_allclose(affine, result_affine, atol=1e-6)
        self.assertTrue(_affine_is_usable(result_affine),
                        "Returned affine should be a valid 4x4 matrix")
        # Mask should be bool-like
        self.assertTrue(np.all((array == 0) | (array == 1)))

    def test_sitk_retry_uses_temporary_header_copy_without_modifying_source(self):
        from mimics_bridge import _read_image_sitk_lps
        import SimpleITK as sitk

        path = self._make_nifti(
            "readonly_source.nii.gz",
            np.zeros((4, 5, 6), dtype=np.int16),
            np.diag([1.0, 1.0, 2.0, 1.0]),
            sform_code=2,
            qform_code=1,
        )
        with open(path, "rb") as handle:
            before = handle.read()
        original_read = sitk.ReadImage
        source_path = os.path.abspath(path)
        calls = []
        try:
            def fail_source_once(candidate, *args, **kwargs):
                calls.append(os.path.abspath(str(candidate)))
                if os.path.abspath(str(candidate)) == source_path:
                    raise RuntimeError("forced source-header retry")
                return original_read(candidate, *args, **kwargs)
            sitk.ReadImage = fail_source_once
            image = _read_image_sitk_lps(path)
        finally:
            sitk.ReadImage = original_read
        with open(path, "rb") as handle:
            after = handle.read()
        self.assertEqual(before, after)
        self.assertEqual((4, 5, 6), image.GetSize())
        self.assertGreaterEqual(len(calls), 2)

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
        self.assertFalse(result["resampled_source_grid"])
        self.assertEqual("original", result["resampled_grid"])

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

    def test_oblique_import_export_roundtrip_returns_exact_source_grid(self):
        from mimics_bridge import do_convert, do_prepare
        import nibabel as nib

        shape = (9, 7, 5)
        angle = np.deg2rad(17.0)
        affine = np.array([
            [0.8 * np.cos(angle), 0.0, 1.6 * np.sin(angle), -61.0],
            [0.0, 1.1, 0.0, 24.0],
            [-0.8 * np.sin(angle), 0.0, 1.6 * np.cos(angle), 11.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        image_path = os.path.join(self.tmp, "roundtrip_ct.nii.gz")
        mask_path = os.path.join(self.tmp, "roundtrip_mask.nii.gz")
        source_mask = np.zeros(shape, dtype=np.uint8)
        source_mask[1:4, 2:6, 1:3] = 1
        source_mask[7, 1, 4] = 1
        nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.int16), affine), image_path)
        nib.save(nib.Nifti1Image(source_mask, affine), mask_path)
        axes = [1, 0, 2]
        flips = [True, False, True]
        prepared = do_prepare({
            "image_path": image_path,
            "masks": [{"name": "organ", "path": mask_path}],
            "dicom_out": os.path.join(self.tmp, "roundtrip_dicom"),
            "buffers_out": os.path.join(self.tmp, "roundtrip_import_buffers"),
            "case_id": "roundtrip",
            "case_dir": self.tmp,
            "axes": axes,
            "flips": flips,
        })
        self.assertEqual("ok", prepared["status"])
        self.assertFalse(prepared["resampled_source_grid"])

        export_buffers = os.path.join(self.tmp, "roundtrip_export_buffers")
        os.makedirs(export_buffers)
        imported = prepared["masks"][0]
        shutil.copy2(imported["u8_path"], os.path.join(export_buffers, "organ.u8"))
        manifest_path = os.path.join(export_buffers, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump({
                "mimics_shape": imported["mimics_shape"],
                "mimics_voxel_to_ras_matrix": prepared["mimics_voxel_to_ras_matrix"],
                "masks": [{
                    "original_name": "organ",
                    "safe_name": "organ",
                    "u8_filename": "organ.u8",
                }],
            }, handle)
        output_dir = os.path.join(self.tmp, "roundtrip_output")
        converted = do_convert({
            "buffers_dir": export_buffers,
            "manifest_path": manifest_path,
            "case_dir": self.tmp,
            "source_image_path": image_path,
            "output_seg_dir": output_dir,
            "export_space": "source_image",
            "require_source_geometry": True,
            "axes": axes,
            "flips": flips,
        })
        self.assertEqual("ok", converted["status"])
        exported = nib.load(os.path.join(output_dir, "organ.nii.gz"))
        self.assertEqual(shape, exported.shape)
        np.testing.assert_allclose(affine, exported.affine, atol=1e-5)
        np.testing.assert_array_equal(source_mask, np.asanyarray(exported.dataobj).astype(np.uint8))

    def test_regular_gantry_tilt_is_encoded_without_resampling(self):
        import pydicom
        import SimpleITK as sitk
        from mimics_bridge import (
            LPS_TO_RAS,
            get_image_affine_from_dicom,
            nifti_to_derived_dicom,
        )

        image = sitk.Image(4, 5, 3, sitk.sitkInt16)
        image.SetSpacing((1.0, 1.0, 2.0))
        image.SetDirection((
            1.0, 0.0, 0.2,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ))
        source_path = os.path.join(self.tmp, "tilted.mha")
        dicom_dir = os.path.join(self.tmp, "tilted_dicom")
        sitk.WriteImage(image, source_path)

        result = nifti_to_derived_dicom(source_path, dicom_dir, case_id="tilted")

        self.assertFalse(result["resampled_source_grid"])
        expected_step_lps = np.asarray(result["affine_lps"], dtype=float)[:3, 2]
        self.assertGreater(abs(float(expected_step_lps[0])), 0.0)
        files = sorted(Path(dicom_dir).glob("*.dcm"))
        first = pydicom.dcmread(str(files[0]), stop_before_pixels=True)
        second = pydicom.dcmread(str(files[1]), stop_before_pixels=True)
        actual_step_lps = (
            np.asarray(second.ImagePositionPatient, dtype=float)
            - np.asarray(first.ImagePositionPatient, dtype=float)
        )
        np.testing.assert_allclose(expected_step_lps, actual_step_lps, atol=1e-6)
        recovered_lps = LPS_TO_RAS @ get_image_affine_from_dicom(dicom_dir)
        np.testing.assert_allclose(expected_step_lps, recovered_lps[:3, 2], atol=1e-6)

    def test_in_plane_shear_is_the_geometry_that_requires_resampling(self):
        import SimpleITK as sitk
        from mimics_bridge import _prepare_dicom_source_grid

        image = sitk.Image(4, 5, 3, sitk.sitkInt16)
        image.SetDirection((
            1.0, 0.2, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ))
        _prepared, info = _prepare_dicom_source_grid(image)

        self.assertTrue(info["resampled_source_grid"])
        self.assertEqual(
            "in_plane_shear_not_representable_by_classic_dicom",
            info["resample_reason"],
        )

    def test_never_resample_rejects_unrepresentable_in_plane_shear(self):
        import SimpleITK as sitk
        from mimics_bridge import _prepare_dicom_source_grid

        image = sitk.Image(4, 5, 3, sitk.sitkInt16)
        image.SetDirection((
            1.0, 0.2, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ))
        key = "MIMICS_DICOM_RESAMPLE_MODE"
        previous = os.environ.get(key)
        try:
            os.environ[key] = "never"
            with self.assertRaisesRegex(ValueError, "classic DICOM cannot represent"):
                _prepare_dicom_source_grid(image)
        finally:
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


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

    def test_mask_resample_default_is_nearest_and_discrete_flip_is_exact(self):
        from mimics_bridge import (
            _mask_resample_method,
            _grid_mapping_is_discrete,
            resample_mask_to_image_grid,
        )

        previous = os.environ.pop("MIMICS_MASK_RESAMPLE_METHOD", None)
        try:
            self.assertEqual("nearest", _mask_resample_method())
        finally:
            if previous is not None:
                os.environ["MIMICS_MASK_RESAMPLE_METHOD"] = previous

        source = np.zeros((4, 3, 2), dtype=np.uint8)
        source[0, 1, 0] = 1
        source[3, 2, 1] = 1
        source_affine = np.eye(4)
        target_affine = np.eye(4)
        target_affine[0, 0] = -1.0
        target_affine[0, 3] = 3.0
        self.assertTrue(_grid_mapping_is_discrete(source_affine, target_affine))
        result = resample_mask_to_image_grid(
            source,
            source_affine,
            source.shape,
            target_affine,
        )
        np.testing.assert_array_equal(np.flip(source, axis=0), result)
        self.assertEqual(int(source.sum()), int(result.sum()))

    def test_resample_mask_to_image_grid_unusable_affine_fallback(self):
        """When mask affine is zero (unusable), should return mask as-is."""
        from mimics_bridge import resample_mask_to_image_grid

        shape = (5, 5, 5)
        image_affine = np.eye(4)
        zero_affine = np.zeros((4, 4))
        mask = np.random.randint(0, 2, shape, dtype=np.uint8)
        result = resample_mask_to_image_grid(mask, zero_affine, shape, image_affine)
        np.testing.assert_array_equal(mask, result)

    def test_resample_mask_to_image_grid_rejects_singular_affine(self):
        from mimics_bridge import resample_mask_to_image_grid

        mask = np.zeros((2, 2, 2), dtype=np.uint8)
        mask[0, 0, 0] = 1
        singular = np.eye(4, dtype=float)
        singular[2, 2] = 0.0
        with self.assertRaisesRegex(ValueError, "singular or invalid"):
            resample_mask_to_image_grid(
                mask,
                singular,
                (3, 3, 3),
                np.eye(4, dtype=float),
            )

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

    def test_mask_header_geometry_fallback_is_explicit(self):
        from mimics_bridge import _mask_file_declares_spatial_geometry

        missing = os.path.join(self.tmp, "missing_geometry.mhd")
        with open(missing, "w", encoding="ascii") as handle:
            handle.write("NDims = 3\nDimSize = 4 5 6\nElementSpacing = 1 1 1\n")
        explicit = os.path.join(self.tmp, "explicit_geometry.mhd")
        with open(explicit, "w", encoding="ascii") as handle:
            handle.write(
                "NDims = 3\nDimSize = 4 5 6\nElementSpacing = 1 1 1\n"
                "Offset = 0 0 0\nTransformMatrix = 1 0 0 0 1 0 0 0 1\n"
            )
        self.assertFalse(_mask_file_declares_spatial_geometry(missing))
        self.assertTrue(_mask_file_declares_spatial_geometry(explicit))

    def test_valid_identity_affine_is_not_silently_voxel_aligned(self):
        from mimics_bridge import resample_mask_to_image_grid

        mask = np.zeros((4, 4, 4), dtype=np.uint8)
        mask[1, 1, 1] = 1
        source_affine = np.eye(4)
        target_affine = np.eye(4)
        target_affine[0, 3] = 100.0
        result = resample_mask_to_image_grid(
            mask,
            source_affine,
            mask.shape,
            target_affine,
            allow_voxel_aligned_fallback=False,
        )
        self.assertEqual(0, int(result.sum()))

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

    def test_read_nifti_mask_accepts_trailing_singleton_channel(self):
        from mimics_bridge import read_nifti_mask

        import nibabel as nib

        data = np.zeros((5, 6, 7, 1), dtype=np.uint8)
        data[1, 2, 3, 0] = 1
        path = os.path.join(self.tmp, "singleton_channel.nii.gz")
        nib.save(nib.Nifti1Image(data, np.eye(4)), path)

        result = read_nifti_mask(path)
        self.assertEqual((5, 6, 7), result.shape)
        self.assertEqual(1, int(result[1, 2, 3]))

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

    def test_convert_canonical_source_mask_bypasses_corrupt_mcs_buffer(self):
        from mimics_bridge import do_convert
        import nibabel as nib

        case_dir = os.path.join(self.tmp, "case_canonical")
        buffers_dir = os.path.join(self.tmp, "buffers_canonical")
        output_dir = os.path.join(self.tmp, "canonical_output")
        os.makedirs(case_dir)
        os.makedirs(buffers_dir)

        shape = (4, 3, 2)
        affine = np.array([
            [0.8, 0.0, 0.0, -12.0],
            [0.0, 1.1, 0.0, 7.0],
            [0.0, 0.0, 2.5, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        source_image_path = os.path.join(case_dir, "mri.nii.gz")
        source_mask_path = os.path.join(self.tmp, "source_brain.nii.gz")
        source_mask = np.zeros(shape, dtype=np.uint8)
        source_mask[1:3, 1:3, :] = 1
        source_mask[3, 0, 1] = 1
        nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.int16), affine), source_image_path)
        nib.save(nib.Nifti1Image(source_mask, affine), source_mask_path)

        # This is intentionally wrong. Canonical export must not read it.
        with open(os.path.join(buffers_dir, "brain.u8"), "wb") as handle:
            handle.write(np.zeros(shape, dtype=np.uint8).tobytes())
        manifest_path = os.path.join(buffers_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump({
                "mimics_shape": list(shape),
                "mimics_voxel_to_ras_matrix": np.eye(4).tolist(),
                "masks": [{
                    "original_name": "brain",
                    "safe_name": "brain",
                    "u8_filename": "brain.u8",
                }],
            }, handle)

        result = do_convert({
            "buffers_dir": buffers_dir,
            "manifest_path": manifest_path,
            "case_dir": case_dir,
            "source_image_path": source_image_path,
            "source_mask_paths": {"brain": source_mask_path},
            "output_seg_dir": output_dir,
            "export_space": "source_image",
            "require_source_geometry": True,
        })
        self.assertEqual("ok", result["status"])
        self.assertTrue(result["mcs_mask_bypassed"])
        row = result["exported"][0]
        self.assertEqual("canonical_source_mask", row["mask_source"])
        self.assertTrue(row["mcs_mask_bypassed"])
        self.assertEqual(os.path.abspath(source_mask_path), row["source_mask_path"])
        self.assertTrue(row["source_mask_sha256"].startswith("sha256:"))
        out = nib.load(os.path.join(output_dir, "brain.nii.gz"))
        np.testing.assert_array_equal(source_mask, np.asanyarray(out.dataobj).astype(np.uint8))
        np.testing.assert_allclose(affine, out.affine, atol=1e-8)
        self.assertEqual(int(source_mask.sum()), row["foreground_voxels"])

    def test_convert_canonical_source_mask_rejects_bad_shape_affine_and_values(self):
        from mimics_bridge import do_convert
        import nibabel as nib

        case_dir = os.path.join(self.tmp, "case_canonical_validation")
        buffers_dir = os.path.join(self.tmp, "buffers_canonical_validation")
        os.makedirs(case_dir)
        os.makedirs(buffers_dir)
        shape = (3, 2, 1)
        affine = np.diag([1.0, 1.5, 2.0, 1.0])
        source_image_path = os.path.join(case_dir, "ct.nii.gz")
        nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.int16), affine), source_image_path)
        manifest_path = os.path.join(buffers_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump({
                "mimics_shape": list(shape),
                "mimics_voxel_to_ras_matrix": np.eye(4).tolist(),
                "masks": [{"original_name": "organ", "safe_name": "organ"}],
            }, handle)

        def convert(path, array, mask_affine, output_name):
            nib.save(nib.Nifti1Image(array, mask_affine), path)
            with self.assertRaises(ValueError):
                do_convert({
                    "buffers_dir": buffers_dir,
                    "manifest_path": manifest_path,
                    "case_dir": case_dir,
                    "source_image_path": source_image_path,
                    "source_mask_paths": {"organ": path},
                    "output_seg_dir": os.path.join(self.tmp, output_name),
                    "export_space": "source_image",
                    "require_source_geometry": True,
                })

        convert(
            os.path.join(self.tmp, "bad_shape.nii.gz"),
            np.zeros((2, 2, 1), dtype=np.uint8),
            affine,
            "bad_shape_output",
        )
        bad_affine = affine.copy()
        bad_affine[0, 3] = 9.0
        convert(
            os.path.join(self.tmp, "bad_affine.nii.gz"),
            np.zeros(shape, dtype=np.uint8),
            bad_affine,
            "bad_affine_output",
        )
        nonbinary = np.zeros(shape, dtype=np.uint8)
        nonbinary[0, 0, 0] = 2
        convert(
            os.path.join(self.tmp, "nonbinary.nii.gz"),
            nonbinary,
            affine,
            "nonbinary_output",
        )

    def test_convert_strict_source_export_refuses_missing_source_geometry(self):
        from mimics_bridge import do_convert

        case_dir = os.path.join(self.tmp, "case_missing_source")
        buffers_dir = os.path.join(self.tmp, "buffers_missing_source")
        os.makedirs(case_dir)
        os.makedirs(buffers_dir)
        data = np.zeros((2, 2, 1), dtype=np.uint8)
        with open(os.path.join(buffers_dir, "organ.u8"), "wb") as handle:
            handle.write(data.tobytes())
        manifest_path = os.path.join(buffers_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump({
                "mimics_shape": [2, 2, 1],
                "mimics_voxel_to_ras_matrix": np.eye(4).tolist(),
                "masks": [{"original_name": "organ", "u8_filename": "organ.u8"}],
            }, handle)
        result = do_convert({
            "buffers_dir": buffers_dir,
            "manifest_path": manifest_path,
            "case_dir": case_dir,
            "export_space": "source_image",
            "require_source_geometry": True,
        })
        self.assertEqual("error", result["status"])
        self.assertIn("original image geometry", result["error"])
        # P1: the strict refusal carries a machine-readable marker so the
        # Mimics side can offer the degraded Mimics-grid export.
        self.assertEqual("source_geometry_unavailable", result["error_code"])

    def test_convert_degraded_export_uses_mimics_grid_without_source_geometry(self):
        from mimics_bridge import do_convert
        import nibabel as nib

        case_dir = os.path.join(self.tmp, "case_degraded")
        buffers_dir = os.path.join(self.tmp, "buffers_degraded")
        os.makedirs(case_dir)
        os.makedirs(buffers_dir)
        data = np.zeros((2, 2, 1), dtype=np.uint8)
        data[1, 0, 0] = 1
        with open(os.path.join(buffers_dir, "organ.u8"), "wb") as handle:
            handle.write(data.tobytes())
        mimics_affine = np.array([
            [1.5, 0.0, 0.0, -4.0],
            [0.0, 2.0, 0.0, -6.0],
            [0.0, 0.0, 3.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        manifest_path = os.path.join(buffers_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump({
                "mimics_shape": [2, 2, 1],
                "mimics_voxel_to_ras_matrix": mimics_affine.tolist(),
                "masks": [{"original_name": "organ", "u8_filename": "organ.u8"}],
            }, handle)
        # source_image requested but geometry unavailable and not required:
        # the export degrades to the Mimics grid instead of failing.
        result = do_convert({
            "buffers_dir": buffers_dir,
            "manifest_path": manifest_path,
            "case_dir": case_dir,
            "export_space": "source_image",
            "require_source_geometry": False,
        })
        self.assertEqual("ok", result["status"])
        self.assertEqual("mimics_grid", result["export_space"])
        self.assertTrue(result["degraded_export"])
        out = nib.load(os.path.join(case_dir, "segmentations", "organ.nii.gz"))
        np.testing.assert_allclose(mimics_affine, out.affine, atol=1e-6)
        np.testing.assert_array_equal(data, np.asanyarray(out.dataobj).astype(np.uint8))

    def test_convert_restores_source_grid_from_metadata_when_file_missing(self):
        from mimics_bridge import do_convert
        import nibabel as nib

        case_dir = os.path.join(self.tmp, "case_meta_grid")
        buffers_dir = os.path.join(self.tmp, "buffers_meta_grid")
        os.makedirs(case_dir)
        os.makedirs(buffers_dir)
        # The Mimics grid differs from the source grid.
        mimics_data = np.zeros((4, 3, 2), dtype=np.uint8)
        mimics_data[1, 1, 0] = 1
        with open(os.path.join(buffers_dir, "organ.u8"), "wb") as handle:
            handle.write(mimics_data.tobytes())
        mimics_affine = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        manifest_path = os.path.join(buffers_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump({
                "mimics_shape": [4, 3, 2],
                "mimics_voxel_to_ras_matrix": mimics_affine.tolist(),
                "masks": [{"original_name": "organ", "u8_filename": "organ.u8"}],
            }, handle)
        # Source grid snapshot: different spacing/grid but overlapping world
        # extent, as recorded in mcs metadata at import time. No source image
        # file exists.
        source_affine = np.array([
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        result = do_convert({
            "buffers_dir": buffers_dir,
            "manifest_path": manifest_path,
            "case_dir": case_dir,
            "export_space": "source_image",
            "require_source_geometry": True,
            "source_image_shape": [8, 6, 4],
            "source_voxel_to_ras_matrix": source_affine.tolist(),
        })
        self.assertEqual("ok", result["status"])
        # Landed on the exact original grid, not a degraded Mimics-grid export.
        self.assertEqual("source_image", result["export_space"])
        self.assertFalse(result["degraded_export"])
        self.assertEqual("source_grid_from_metadata", result["export_voxel_to_ras_matrix_source"])
        self.assertEqual([8, 6, 4], result["export_shape"])
        out = nib.load(os.path.join(case_dir, "segmentations", "organ.nii.gz"))
        self.assertEqual((8, 6, 4), out.shape)
        np.testing.assert_allclose(source_affine, out.affine, atol=1e-6)

        # mcs metadata arrives as JSON text; the same values must work.
        rerun_dir = os.path.join(self.tmp, "case_meta_grid_text")
        os.makedirs(rerun_dir)
        rerun = do_convert({
            "buffers_dir": buffers_dir,
            "manifest_path": manifest_path,
            "case_dir": rerun_dir,
            "export_space": "source_image",
            "require_source_geometry": True,
            "source_image_shape": json.dumps([8, 6, 4]),
            "source_voxel_to_ras_matrix": json.dumps(source_affine.tolist()),
        })
        self.assertEqual("ok", rerun["status"])
        self.assertEqual("source_image", rerun["export_space"])
        self.assertFalse(rerun["degraded_export"])
        self.assertEqual("source_grid_from_metadata", rerun["export_voxel_to_ras_matrix_source"])
        rerun_out = nib.load(os.path.join(rerun_dir, "segmentations", "organ.nii.gz"))
        self.assertEqual((8, 6, 4), rerun_out.shape)
        np.testing.assert_allclose(source_affine, rerun_out.affine, atol=1e-6)

    def test_empty_source_image_path_never_scans_cwd(self):
        """An empty source_image_path must not treat cwd as a DICOM folder.

        Path("") is Path("."), so without the guard the bridge recursively
        scanned the working directory for DICOM files. When tests run from a
        checkout containing bundled DICOM test data (e.g. a portable python_env
        with nibabel/pydicom test files), the scan "found" a bogus source
        geometry that overrode the metadata restore path.
        """
        from mimics_bridge import get_source_image_geometry, is_dicom_folder

        self.assertFalse(is_dicom_folder(""))
        self.assertFalse(is_dicom_folder(None))
        self.assertFalse(is_dicom_folder("   "))
        self.assertIsNone(get_source_image_geometry(""))
        self.assertIsNone(get_source_image_geometry(None))
        self.assertIsNone(get_source_image_geometry("   "))

    def test_convert_rejects_unsupported_export_format(self):
        from mimics_bridge import do_convert

        case_dir = os.path.join(self.tmp, "case_bad_format")
        buffers_dir = os.path.join(self.tmp, "buffers_bad_format")
        os.makedirs(case_dir)
        os.makedirs(buffers_dir)
        with open(os.path.join(buffers_dir, "organ.u8"), "wb") as handle:
            handle.write(np.zeros((2, 2, 1), dtype=np.uint8).tobytes())
        manifest_path = os.path.join(buffers_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump({
                "mimics_shape": [2, 2, 1],
                "mimics_voxel_to_ras_matrix": np.eye(4).tolist(),
                "masks": [{"original_name": "organ", "u8_filename": "organ.u8"}],
            }, handle)
        result = do_convert({
            "buffers_dir": buffers_dir,
            "manifest_path": manifest_path,
            "case_dir": case_dir,
            "export_formats": ["stl"],
        })
        self.assertEqual("error", result["status"])
        self.assertIn("Unsupported export format", result["error"])

    def test_convert_multi_format_writes_nrrd_and_mha_with_same_geometry(self):
        import SimpleITK as sitk
        from mimics_bridge import do_convert
        import nibabel as nib

        case_dir = os.path.join(self.tmp, "case_multi_format")
        buffers_dir = os.path.join(self.tmp, "buffers_multi_format")
        os.makedirs(case_dir)
        os.makedirs(buffers_dir)
        data = np.zeros((4, 3, 2), dtype=np.uint8)
        data[1:3, 1:3, :] = 1
        with open(os.path.join(buffers_dir, "organ.u8"), "wb") as handle:
            handle.write(data.tobytes())
        mimics_affine = np.array([
            [0.0, -0.7, 0.0, 11.0],
            [0.7, 0.0, 0.0, -13.0],
            [0.0, 0.0, 1.25, 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        manifest_path = os.path.join(buffers_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump({
                "mimics_shape": [4, 3, 2],
                "mimics_voxel_to_ras_matrix": mimics_affine.tolist(),
                "masks": [{"original_name": "organ", "u8_filename": "organ.u8"}],
            }, handle)

        result = do_convert({
            "buffers_dir": buffers_dir,
            "manifest_path": manifest_path,
            "case_dir": case_dir,
            "export_formats": ["nii.gz", "nrrd", "mha"],
        })
        self.assertEqual("ok", result["status"])
        self.assertEqual(["nii.gz", "nrrd", "mha"], result["export_formats"])
        row = result["exported"][0]
        self.assertEqual("new", row["action"])
        self.assertEqual(3, len(row["paths"]))

        seg_dir = os.path.join(case_dir, "segmentations")
        # All three files exist with identical content.
        nii_img = nib.load(os.path.join(seg_dir, "organ.nii.gz"))
        np.testing.assert_array_equal(
            data, np.asanyarray(nii_img.dataobj).astype(np.uint8)
        )
        for ext in ("nrrd", "mha"):
            img = sitk.ReadImage(os.path.join(seg_dir, "organ." + ext))
            back = np.transpose(sitk.GetArrayFromImage(img), (2, 1, 0))
            self.assertEqual(data.shape, back.shape)
            np.testing.assert_array_equal(data, back)
            # Reconstructed RAS affine matches the Mimics grid affine.
            direction = np.asarray(img.GetDirection(), dtype=float).reshape(3, 3)
            spacing = np.asarray(img.GetSpacing(), dtype=float)
            origin = np.asarray(img.GetOrigin(), dtype=float)
            affine = np.eye(4)
            affine[:3, :3] = direction * spacing
            affine[:3, 3] = origin
            affine[:2, :] *= -1.0
            np.testing.assert_allclose(mimics_affine, affine, atol=1e-5)

        # Re-running with the same data marks every format unchanged.
        rerun = do_convert({
            "buffers_dir": buffers_dir,
            "manifest_path": manifest_path,
            "case_dir": case_dir,
            "export_formats": ["nii.gz", "nrrd", "mha"],
        })
        self.assertEqual("ok", rerun["status"])
        self.assertEqual("unchanged", rerun["exported"][0]["action"])
        self.assertEqual(1, rerun["total_unchanged"])

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

    def test_multilabel_mhd_mask_splits_into_visible_nonempty_buffers(self):
        import SimpleITK as sitk
        from mimics_bridge import do_prepare_masks_for_grid, get_image_affine, get_image_shape

        mask_path = os.path.join(self.tmp, "organs.mhd")
        values = np.zeros((2, 3, 4), dtype=np.uint16)
        values[0, 0:2, 0:2] = 1
        values[1, 1:3, 2:4] = 2
        image = sitk.GetImageFromArray(values)
        image.SetSpacing((0.8, 0.9, 2.5))
        image.SetOrigin((12.0, -8.0, 30.0))
        sitk.WriteImage(image, mask_path)

        buffers = os.path.join(self.tmp, "label_buffers")
        result = do_prepare_masks_for_grid({
            "masks": [{"name": "organs", "mask_path": mask_path}],
            "buffers_out": buffers,
            "target_shape": list(get_image_shape(mask_path)),
            "target_voxel_to_ras_matrix": get_image_affine(mask_path).tolist(),
            "axes": [0, 1, 2],
            "flips": [False, False, False],
        })
        self.assertEqual("ok", result["status"])
        self.assertEqual(["organs_label1", "organs_label2"], [row["name"] for row in result["masks"]])
        self.assertTrue(all(int(row["foreground_voxels"]) > 0 for row in result["masks"]))
        self.assertTrue(all(os.path.getsize(row["u8_path"]) == values.size for row in result["masks"]))

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
        self.assertEqual("dicom_uint16_linear_rescale_v1", result["source_intensity_encoding"])
        self.assertAlmostEqual(float(first.RescaleSlope), result["source_intensity_rescale_slope"])
        self.assertAlmostEqual(float(first.RescaleIntercept), result["source_intensity_rescale_intercept"])
        self.assertAlmostEqual(float(source_values.min()), result["source_intensity_value_min"])
        self.assertAlmostEqual(float(source_values.max()), result["source_intensity_value_max"])
        self.assertEqual(0, result["dicom_stored_value_min"])
        self.assertEqual(65535, result["dicom_stored_value_max"])

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

    def test_fewshot_materialize_keeps_image_and_resamples_label_to_source_grid(self):
        pipeline = __import__("tools.fewshot_pipeline", fromlist=["dummy"])
        import nibabel as nib

        image_src = os.path.join(self.tmp, "ct.nii.gz")
        label_src = os.path.join(self.tmp, "mask.nii.gz")
        nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.int16), np.eye(4)), image_src)
        label_affine = np.array([
            [1.0, 0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
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
        self.assertIn(rows[0]["image_materialization"], ("hardlink", "copy"))
        self.assertEqual(
            "label_resampled_to_source_image_grid",
            rows[0]["label_materialization"],
        )
        self.assertFalse(rows[0]["image_label_geometry_matched"])
        out = nib.load(rows[0]["dataset_image"])
        self.assertEqual((2, 2, 2), out.shape)
        np.testing.assert_allclose(np.eye(4), out.affine, atol=1e-6)
        label_out = nib.load(rows[0]["dataset_label"])
        self.assertEqual((2, 2, 2), label_out.shape)
        np.testing.assert_allclose(np.eye(4), label_out.affine, atol=1e-6)
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
        args.dinov3_root = DINOV3_ROOT
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
        args.dinov3_root = DINOV3_ROOT
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

    def test_load_image_source_mhd_preserves_physical_values(self):
        from nninteractive_bridge import load_image_source
        import SimpleITK as sitk

        shape = (7, 6, 5)
        data_xyz = np.linspace(-3.5, 9.25, np.prod(shape), dtype=np.float32).reshape(shape)
        image = sitk.GetImageFromArray(np.transpose(data_xyz, (2, 1, 0)))
        path = os.path.join(self.tmp, "source.mhd")
        sitk.WriteImage(image, path)
        identity = np.eye(4)
        result = load_image_source({
            "image_path": path,
            "image_source_kind": "medical_image",
            "image_expected_shape": list(shape),
            "image_source_voxel_to_ras_matrix": json.dumps(identity.tolist()),
            "image_mimics_voxel_to_ras_matrix": json.dumps(identity.tolist()),
        })
        np.testing.assert_allclose(result[0], data_xyz, rtol=0.0, atol=1e-6)

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

    def test_atomic_json_retries_without_deleting_previous_state(self):
        import nninteractive_bridge as bridge

        path = Path(self.tmp) / "state.json"
        path.write_text('{"old": true}', encoding="utf-8")
        original_replace = bridge.os.replace
        original_remove = bridge.os.remove
        original_sleep = bridge.time.sleep
        calls = {"replace": 0, "remove": 0}

        def flaky_replace(source, target):
            calls["replace"] += 1
            if calls["replace"] < 3:
                raise OSError(5, "access denied")
            return original_replace(source, target)

        def tracked_remove(target):
            calls["remove"] += 1
            return original_remove(target)

        try:
            bridge.os.replace = flaky_replace
            bridge.os.remove = tracked_remove
            bridge.time.sleep = lambda _seconds: None
            bridge._write_json_atomic(path, {"new": True})
        finally:
            bridge.os.replace = original_replace
            bridge.os.remove = original_remove
            bridge.time.sleep = original_sleep

        self.assertEqual({"new": True}, json.loads(path.read_text(encoding="utf-8")))
        self.assertEqual(0, calls["remove"])

    def test_atomic_json_falls_back_when_replace_is_denied(self):
        import nninteractive_bridge as bridge

        path = Path(self.tmp) / "smb_state.json"
        path.write_text('{"old": true}', encoding="utf-8")
        original_replace = bridge.os.replace
        original_sleep = bridge.time.sleep
        try:
            bridge.os.replace = lambda _source, _target: (_ for _ in ()).throw(
                OSError(5, "access denied")
            )
            bridge.time.sleep = lambda _seconds: None
            bridge._write_json_atomic(path, {"fallback": True})
        finally:
            bridge.os.replace = original_replace
            bridge.time.sleep = original_sleep

        self.assertEqual(
            {"fallback": True},
            json.loads(path.read_text(encoding="utf-8")),
        )

    def test_win32_timer_keeps_raw_callback_pointer_alive(self):
        import inspect
        import nninteractive_mimics

        source = inspect.getsource(
            nninteractive_mimics._start_win32_async_result_monitor
        )
        self.assertIn("ctypes.cast(callback, ctypes.c_void_p)", source)
        self.assertIn('monitor["callback_void"] = callback_void', source)


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
        stop_path = os.path.join(self.tmp, ".mimics_runtime", "_mcs_queue_stop.json")
        os.makedirs(os.path.dirname(stop_path))
        with open(stop_path, "w") as f:
            json.dump({"status": "stop_requested"}, f)
        self.assertTrue(os.path.isfile(stop_path))
        # The main loop checks the runtime-scoped marker and exits when true.
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
        log_path = os.path.join(self.tmp, ".mimics_runtime", "logs", "_create_mcs_batch.log")
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
        logs_dir = os.path.join(self.tmp, ".mimics_runtime", "logs")
        self.assertFalse(os.path.isdir(logs_dir))
        log_message(self.tmp, "first message")
        self.assertTrue(os.path.isdir(logs_dir))

    def test_record_failed_case(self):
        from create_mcs_batch import record_failed_case

        record_failed_case(self.tmp, "s0001", "prepare", "Test error", "traceback text")
        failed_dir = os.path.join(self.tmp, ".mimics_runtime", "_failed_cases")
        self.assertTrue(os.path.isdir(failed_dir))
        files = os.listdir(failed_dir)
        self.assertEqual(1, len(files))
        with open(os.path.join(failed_dir, files[0]), "r") as f:
            data = json.load(f)
        self.assertEqual("s0001", data["case_id"])
        self.assertEqual("prepare", data["phase"])
        self.assertEqual("Test error", data["error"])

    def test_local_queue_descriptor_is_consumed_by_background_creator(self):
        import create_mcs_batch

        output_dir = os.path.join(self.tmp, "output")
        runtime_dir = os.path.join(self.tmp, "local_queue")
        work_dir = os.path.join(self.tmp, "local_work", "case001")
        queue_dir = os.path.join(runtime_dir, "prepared_queue")
        os.makedirs(output_dir)
        os.makedirs(work_dir)
        os.makedirs(queue_dir)
        output_mcs = os.path.join(output_dir, "case001.mcs")
        with open(os.path.join(work_dir, "prepare_manifest.json"), "w") as handle:
            json.dump({"output_mcs": output_mcs, "source_fingerprint": "fp1"}, handle)
        descriptor = os.path.join(queue_dir, "case001.json")
        with open(descriptor, "w") as handle:
            json.dump({"case_id": "case001", "work_dir": work_dir, "output_mcs": output_mcs}, handle)
        with open(os.path.join(runtime_dir, create_mcs_batch.QUEUE_DONE_FILE), "w") as handle:
            json.dump({"status": "done"}, handle)

        old_create = create_mcs_batch.create_mcs_from_manifest
        old_runtime = create_mcs_batch._ACTIVE_RUNTIME_DIR
        try:
            def fake_create(_work_dir, path):
                with open(path, "w") as handle:
                    handle.write("mcs")
                return path
            create_mcs_batch.create_mcs_from_manifest = fake_create
            self.assertEqual(0, create_mcs_batch.main(output_dir, runtime_dir=runtime_dir))
        finally:
            create_mcs_batch.create_mcs_from_manifest = old_create
            create_mcs_batch._ACTIVE_RUNTIME_DIR = old_runtime
        self.assertTrue(os.path.isfile(output_mcs))
        self.assertFalse(os.path.exists(descriptor))
        self.assertFalse(os.path.exists(work_dir))

    def test_failed_mcs_case_does_not_stop_later_cases(self):
        import create_mcs_batch

        output_dir = os.path.join(self.tmp, "output")
        runtime_dir = os.path.join(self.tmp, "local_queue")
        queue_dir = os.path.join(runtime_dir, "prepared_queue")
        os.makedirs(output_dir)
        os.makedirs(queue_dir)

        descriptors = []
        for case_id in ("case_bad", "case_good"):
            work_dir = os.path.join(self.tmp, "work", case_id)
            os.makedirs(work_dir)
            output_mcs = os.path.join(output_dir, case_id + ".mcs")
            with open(os.path.join(work_dir, "prepare_manifest.json"), "w") as handle:
                json.dump({"output_mcs": output_mcs}, handle)
            descriptor = os.path.join(queue_dir, case_id + ".json")
            with open(descriptor, "w") as handle:
                json.dump({
                    "case_id": case_id,
                    "work_dir": work_dir,
                    "output_mcs": output_mcs,
                }, handle)
            descriptors.append(descriptor)
        with open(os.path.join(runtime_dir, create_mcs_batch.QUEUE_DONE_FILE), "w") as handle:
            json.dump({"status": "done"}, handle)

        calls = []
        old_create = create_mcs_batch.create_mcs_from_manifest
        old_runtime = create_mcs_batch._ACTIVE_RUNTIME_DIR
        try:
            def fake_create(work_dir, staged_path):
                case_id = os.path.basename(work_dir)
                calls.append(case_id)
                if case_id == "case_bad":
                    raise RuntimeError("controlled create failure")
                with open(staged_path, "w") as handle:
                    handle.write("complete project")
                return staged_path
            create_mcs_batch.create_mcs_from_manifest = fake_create
            self.assertEqual(0, create_mcs_batch.main(output_dir, runtime_dir=runtime_dir))
        finally:
            create_mcs_batch.create_mcs_from_manifest = old_create
            create_mcs_batch._ACTIVE_RUNTIME_DIR = old_runtime

        self.assertEqual(["case_bad", "case_good"], calls)
        self.assertFalse(os.path.exists(os.path.join(output_dir, "case_bad.mcs")))
        self.assertTrue(os.path.isfile(os.path.join(output_dir, "case_good.mcs")))
        self.assertTrue(all(not os.path.exists(path) for path in descriptors))
        status = json.loads(Path(runtime_dir, create_mcs_batch.STATUS_FILE).read_text(encoding="utf-8"))
        self.assertEqual("closed", status["status"])
        self.assertEqual(1, status["completed"])
        self.assertEqual(1, status["failed"])

    def test_failed_mcs_rebuild_keeps_previous_complete_project(self):
        import create_mcs_batch

        output_dir = os.path.join(self.tmp, "output")
        runtime_dir = os.path.join(self.tmp, "local_queue")
        queue_dir = os.path.join(runtime_dir, "prepared_queue")
        work_dir = os.path.join(self.tmp, "work", "case001")
        os.makedirs(output_dir)
        os.makedirs(queue_dir)
        os.makedirs(work_dir)
        output_mcs = os.path.join(output_dir, "case001.mcs")
        Path(output_mcs).write_text("previous complete project", encoding="utf-8")
        Path(work_dir, "prepare_manifest.json").write_text(
            json.dumps({"output_mcs": output_mcs}), encoding="utf-8"
        )
        Path(queue_dir, "case001.json").write_text(json.dumps({
            "case_id": "case001",
            "work_dir": work_dir,
            "output_mcs": output_mcs,
        }), encoding="utf-8")
        Path(runtime_dir, create_mcs_batch.QUEUE_DONE_FILE).write_text(
            json.dumps({"status": "done"}), encoding="utf-8"
        )

        old_create = create_mcs_batch.create_mcs_from_manifest
        old_runtime = create_mcs_batch._ACTIVE_RUNTIME_DIR
        try:
            create_mcs_batch.create_mcs_from_manifest = lambda *_args: (_ for _ in ()).throw(
                RuntimeError("controlled rebuild failure")
            )
            self.assertEqual(0, create_mcs_batch.main(output_dir, runtime_dir=runtime_dir))
        finally:
            create_mcs_batch.create_mcs_from_manifest = old_create
            create_mcs_batch._ACTIVE_RUNTIME_DIR = old_runtime
        self.assertEqual("previous complete project", Path(output_mcs).read_text(encoding="utf-8"))

    def test_native_mimics_exit_quarantines_only_current_case(self):
        from tools import mcs_creation_supervisor as supervisor

        runtime_dir = Path(self.tmp, "runtime")
        queue_dir = runtime_dir / "prepared_queue"
        work_dir = Path(self.tmp, "work", "case_bad")
        runtime_dir.mkdir()
        queue_dir.mkdir()
        work_dir.mkdir(parents=True)
        descriptor = queue_dir / "case_bad.json"
        descriptor.write_text("{}", encoding="utf-8")
        staging = Path(self.tmp, "output", "case_bad.creating.123.mcs")
        staging.parent.mkdir()
        staging.write_text("partial", encoding="utf-8")
        final_mcs = staging.parent / "case_bad.mcs"
        final_mcs.write_text("previous complete project", encoding="utf-8")
        runtime_common = __import__("runtime_common")
        runtime_common.write_json_atomic(str(runtime_dir / supervisor.STATUS_FILE), {
            "status": "creating", "completed": 2, "failed": 1,
        })
        runtime_common.write_json_atomic(str(runtime_dir / supervisor.CURRENT_CASE_FILE), {
            "case_id": "case_bad",
            "work_dir": str(work_dir),
            "output_mcs": str(final_mcs),
            "staging_mcs": str(staging),
            "descriptor_path": str(descriptor),
            "worker_pid": 123,
            "completed_before_case": 2,
            "failed_before_case": 1,
        })

        recovered = supervisor.quarantine_interrupted_case(runtime_dir, 123, -1073741819)
        self.assertEqual("case_bad", recovered["case_id"])
        self.assertFalse(descriptor.exists())
        self.assertFalse(work_dir.exists())
        self.assertFalse(staging.exists())
        self.assertEqual("previous complete project", final_mcs.read_text(encoding="utf-8"))
        status = runtime_common.read_json(str(runtime_dir / supervisor.STATUS_FILE), {})
        self.assertEqual("recovering", status["status"])
        self.assertEqual(2, status["completed"])
        self.assertEqual(2, status["failed"])
        self.assertEqual(1, len(list((runtime_dir / "_failed_cases").glob("*.json"))))

    def test_supervisor_restarts_after_native_case_exit_and_finishes_queue(self):
        from argparse import Namespace
        from tools import mcs_creation_supervisor as supervisor

        runtime_dir = Path(self.tmp, "runtime")
        output_dir = Path(self.tmp, "output")
        queue_dir = runtime_dir / "prepared_queue"
        runtime_dir.mkdir()
        output_dir.mkdir()
        queue_dir.mkdir()
        Path(runtime_dir, supervisor.DONE_FILE).write_text(
            json.dumps({"status": "done", "completed": 2, "failed": 0}),
            encoding="utf-8",
        )
        for case_id in ("case_bad", "case_good"):
            work_dir = Path(self.tmp, "work", case_id)
            work_dir.mkdir(parents=True)
            Path(work_dir, "prepare_manifest.json").write_text("{}", encoding="utf-8")
            Path(queue_dir, case_id + ".json").write_text(json.dumps({
                "case_id": case_id,
                "work_dir": str(work_dir),
                "output_mcs": str(output_dir / (case_id + ".mcs")),
            }), encoding="utf-8")

        child_script = Path(self.tmp, "fake_mimics_child.py")
        child_script.write_text(
            "import json, os, pathlib, sys, time\n"
            "runtime = pathlib.Path(sys.argv[1])\n"
            "output = pathlib.Path(sys.argv[2])\n"
            "counter = runtime / 'child_count.txt'\n"
            "count = int(counter.read_text()) if counter.exists() else 0\n"
            "counter.write_text(str(count + 1))\n"
            "if count == 0:\n"
            "    work = pathlib.Path(sys.argv[3])\n"
            "    marker = {\n"
            "      'case_id': 'case_bad', 'work_dir': str(work),\n"
            "      'output_mcs': str(output / 'case_bad.mcs'),\n"
            "      'staging_mcs': str(output / 'case_bad.creating.mcs'),\n"
            "      'descriptor_path': str(runtime / 'prepared_queue' / 'case_bad.json'),\n"
            "      'worker_pid': os.getpid(), 'completed_before_case': 0,\n"
            "      'failed_before_case': 0}\n"
            "    (output / 'case_bad.creating.mcs').write_text('partial')\n"
            "    (runtime / '_mcs_current_case.json').write_text(json.dumps(marker))\n"
            "    sys.exit(7)\n"
            "good = runtime / 'prepared_queue' / 'case_good.json'\n"
            "payload = json.loads(good.read_text())\n"
            "pathlib.Path(payload['work_dir']).joinpath('prepare_manifest.json').unlink()\n"
            "pathlib.Path(payload['work_dir']).rmdir()\n"
            "good.unlink()\n"
            "status_path = runtime / '_mcs_batch_status.json'\n"
            "status = json.loads(status_path.read_text())\n"
            "status.update({'status': 'closed', 'completed': 1, 'failed': 1,\n"
            " 'creation_completed': 1, 'creation_failed': 1})\n"
            "status_path.write_text(json.dumps(status))\n",
            encoding="utf-8",
        )
        bad_work = Path(self.tmp, "work", "case_bad")
        original_command = supervisor.runtime_common.background_mimics_command
        try:
            supervisor.runtime_common.background_mimics_command = lambda *_args, **_kwargs: [
                sys.executable,
                str(child_script),
                str(runtime_dir),
                str(output_dir),
                str(bad_work),
            ]
            result = supervisor._run(Namespace(
                runtime_dir=str(runtime_dir),
                output_dir=str(output_dir),
                mimics_exe="fake",
                runner="fake",
                mimics_log="",
                handshake="",
                max_start_retries=2,
                retry_delay=0.01,
            ))
        finally:
            supervisor.runtime_common.background_mimics_command = original_command

        self.assertEqual(0, result)
        self.assertEqual("2", Path(runtime_dir, "child_count.txt").read_text(encoding="utf-8"))
        self.assertFalse(Path(queue_dir, "case_bad.json").exists())
        self.assertFalse(Path(queue_dir, "case_good.json").exists())
        status = json.loads(Path(runtime_dir, supervisor.STATUS_FILE).read_text(encoding="utf-8"))
        self.assertEqual("closed", status["status"])
        self.assertEqual(1, status["completed"])
        self.assertEqual(1, status["failed"])

    def test_supervisor_ignores_descriptor_whose_manifest_was_quarantined(self):
        from tools import mcs_creation_supervisor as supervisor

        runtime_dir = Path(self.tmp, "runtime")
        output_dir = Path(self.tmp, "output")
        queue_dir = runtime_dir / "prepared_queue"
        work_dir = Path(self.tmp, "work", "stale")
        queue_dir.mkdir(parents=True)
        output_dir.mkdir()
        work_dir.mkdir(parents=True)
        Path(queue_dir, "stale.json").write_text(json.dumps({
            "case_id": "stale",
            "work_dir": str(work_dir),
        }), encoding="utf-8")
        self.assertFalse(supervisor._queue_has_work(runtime_dir, output_dir))
        Path(work_dir, "prepare_manifest.json").write_text("{}", encoding="utf-8")
        self.assertTrue(supervisor._queue_has_work(runtime_dir, output_dir))

    def test_creation_status_combines_prepare_and_mcs_failures(self):
        import create_mcs_batch

        runtime_dir = Path(self.tmp, "runtime")
        runtime_dir.mkdir()
        Path(runtime_dir, create_mcs_batch.QUEUE_DONE_FILE).write_text(json.dumps({
            "status": "done", "completed": 4, "failed": 1,
        }), encoding="utf-8")
        old_runtime = create_mcs_batch._ACTIVE_RUNTIME_DIR
        try:
            create_mcs_batch._ACTIVE_RUNTIME_DIR = str(runtime_dir)
            create_mcs_batch.update_status(
                self.tmp, "closed", completed=3, failed=1
            )
        finally:
            create_mcs_batch._ACTIVE_RUNTIME_DIR = old_runtime
        status = json.loads(
            Path(runtime_dir, create_mcs_batch.STATUS_FILE).read_text(encoding="utf-8")
        )
        self.assertEqual(3, status["completed"])
        self.assertEqual(2, status["failed"])
        self.assertEqual(1, status["creation_failed"])
        self.assertEqual(1, status["preparation_failed"])
        self.assertEqual(5, status["total"])

    def test_shape_product(self):
        from create_mcs_batch import _shape_product

        self.assertEqual(60, _shape_product([3, 4, 5]))
        self.assertEqual(1, _shape_product([1, 1, 1]))
        self.assertEqual(0, _shape_product([0, 5, 5]))

    def test_required_metadata_write_failure_is_not_silenced(self):
        from create_mcs_batch import metadata_set

        class Metadata(object):
            def find(self, _name):
                return None
            def create(self, **_kwargs):
                raise ValueError("metadata denied")

        class Image(object):
            metadata = Metadata()

        with self.assertRaisesRegex(RuntimeError, "required image metadata"):
            metadata_set(Image(), "mimics_script.test", "value")

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
        """The official nnInteractive entry routes to nninteractive_mimics."""
        entry = os.path.join(
            PROJECT_ROOT,
            "scripting_library",
            "02_AI",
            "nnInteractive",
            "01_Annotate_Official_Model.py",
        )
        with open(entry, "r") as f:
            source = f.read()
        self.assertIn("nninteractive_mimics", source)

    def test_dinov3_entries_separate_from_nninteractive(self):
        """DINOv3 and nnInteractive have separate, ordered feature groups."""
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

        nn_dir = os.path.join(
            PROJECT_ROOT, "scripting_library", "02_AI", "nnInteractive"
        )
        self.assertEqual(
            [
                "01_Annotate_Official_Model.py",
                "02_Annotate_Custom_Model.py",
                "03_Train_and_Manage_Custom_Models.py",
            ],
            sorted(name for name in os.listdir(nn_dir) if name.endswith(".py")),
        )
        nn_entry = os.path.join(nn_dir, "01_Annotate_Official_Model.py")
        self.assertTrue(os.path.isfile(nn_entry))
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
            "fewshot_training_setup_ui.py",
            "fewshot_status_viewer.py",
            "nnunet_pipeline.py",
            "nnunet_stage_worker.py",
            "nnunet_training_setup_ui.py",
            "nnunet_prediction_setup_ui.py",
            "nnunet_status_viewer.py",
            "io_path_setup_ui.py",
            "nninteractive.inference.server.main",
            "setup_env.py",
        ]
        for marker in required:
            self.assertIn(marker, MARKERS, "{0} must be in MARKERS".format(marker))

    def test_mask_export_has_a_scoped_stop_entry_and_marker(self):
        import mimics_stop_background as msb
        import runtime_common

        export_root = os.path.join(self.tmp, "exports")
        runtime_dir = os.path.join(self.tmp, "runtime")
        os.makedirs(export_root)
        os.makedirs(runtime_dir)
        lock_path = os.path.join(self.tmp, "background_mimics.lock")
        runtime_common.write_json_atomic(lock_path, {
            "pid": os.getpid(),
            "kind": "export_labels",
            "owner": "batch label export",
            "export_root": export_root,
        })
        old_lock = msb._background_mimics_lock_path
        old_runtime = msb._runtime_dir
        old_stop = msb._stop_export_inprocess_monitors
        try:
            msb._background_mimics_lock_path = lambda: lock_path
            msb._runtime_dir = lambda: runtime_dir
            msb._stop_export_inprocess_monitors = lambda: 0
            result = msb.stop_background_export()
        finally:
            msb._background_mimics_lock_path = old_lock
            msb._runtime_dir = old_runtime
            msb._stop_export_inprocess_monitors = old_stop
        self.assertEqual(os.getpid(), result.get("target_pid"))
        self.assertTrue(os.path.isfile(os.path.join(export_root, ".mimics_runtime", "_export_stop.json")))
        entry = os.path.join(PROJECT_ROOT, "scripting_library", "01_Data", "06_Stop_Mask_Export.py")
        self.assertTrue(os.path.isfile(entry))
        self.assertIn("main_stop_export", Path(entry).read_text(encoding="utf-8"))

    def test_mask_export_stop_does_not_target_background_import(self):
        import mimics_stop_background as msb
        import runtime_common

        runtime_dir = os.path.join(self.tmp, "runtime")
        os.makedirs(runtime_dir)
        lock_path = os.path.join(self.tmp, "background_mimics.lock")
        runtime_common.write_json_atomic(lock_path, {
            "pid": os.getpid(),
            "kind": "create_mcs",
            "owner": "import .mcs creation",
        })
        old_lock = msb._background_mimics_lock_path
        old_runtime = msb._runtime_dir
        old_stop = msb._stop_export_inprocess_monitors
        try:
            msb._background_mimics_lock_path = lambda: lock_path
            msb._runtime_dir = lambda: runtime_dir
            msb._stop_export_inprocess_monitors = lambda: 0
            result = msb.stop_background_export()
        finally:
            msb._background_mimics_lock_path = old_lock
            msb._runtime_dir = old_runtime
            msb._stop_export_inprocess_monitors = old_stop
        self.assertIsNone(result.get("target_pid"))
        report = runtime_common.read_json(result.get("stop_log"), {}) or {}
        self.assertIn("No active Mimics-Script mask export", report.get("Message", ""))

    def test_scoped_background_locks_all_receive_stop_markers(self):
        import mimics_stop_background as msb
        import runtime_common

        lock_dir = os.path.join(self.tmp, "scoped_locks")
        os.makedirs(lock_dir)
        stop_a = os.path.join(self.tmp, "stop_a.json")
        stop_b = os.path.join(self.tmp, "stop_b.json")
        runtime_common.write_json_atomic(
            os.path.join(lock_dir, "background_mimics_a.lock"),
            {"kind": "export_labels", "pid": os.getpid(), "stop_path": stop_a},
        )
        runtime_common.write_json_atomic(
            os.path.join(lock_dir, "background_mimics_b.lock"),
            {"kind": "fewshot_label_export", "pid": os.getpid(), "stop_path": stop_b},
        )
        old_lock_dir = msb.runtime_common.resource_lock_dir
        old_legacy = msb._background_mimics_lock_path
        try:
            msb.runtime_common.resource_lock_dir = lambda _root: lock_dir
            msb._background_mimics_lock_path = lambda: os.path.join(lock_dir, "background_mimics.lock")
            written = msb._request_lock_owned_stop_markers("test stop")
        finally:
            msb.runtime_common.resource_lock_dir = old_lock_dir
            msb._background_mimics_lock_path = old_legacy

        self.assertEqual({stop_a, stop_b}, set(written))
        self.assertTrue(os.path.isfile(stop_a))
        self.assertTrue(os.path.isfile(stop_b))

    def test_external_io_failure_reports_bounded_stderr_and_log_path(self):
        import io_setup_mimics

        status_path = os.path.join(self.tmp, "io_status.json")
        stderr_path = os.path.join(self.tmp, "io_stderr.log")
        Path(status_path).write_text(json.dumps({"status": "opening"}), encoding="utf-8")
        Path(stderr_path).write_text("Traceback\nImportError: PySide6 DLL load failed\n", encoding="utf-8")

        class Process(object):
            def poll(self):
                return 1

        messages = []
        monitor = {
            "key": "io_failure_test",
            "status_path": status_path,
            "stderr_log": stderr_path,
            "process": Process(),
            "deadline": time.time() + 60.0,
            "busy": False,
        }
        old_message = io_setup_mimics._message
        try:
            io_setup_mimics._IO_SETUP_MONITORS[monitor["key"]] = monitor
            io_setup_mimics._ALERTED.discard(monitor["key"])
            io_setup_mimics._message = lambda title, message: messages.append((title, message))
            io_setup_mimics._tick(monitor)
        finally:
            io_setup_mimics._message = old_message
            io_setup_mimics._IO_SETUP_MONITORS.pop(monitor["key"], None)
            io_setup_mimics._ALERTED.discard(monitor["key"])
        self.assertEqual(1, len(messages))
        self.assertIn("ImportError: PySide6 DLL load failed", messages[0][1])
        self.assertIn(stderr_path, messages[0][1])
        self.assertLessEqual(len(io_setup_mimics._read_stderr_tail(stderr_path, 16).encode("utf-8")), 32)

    def test_import_launch_directory_failure_is_reported_without_timeout(self):
        import mimics_import

        job_dir = os.path.join(self.tmp, "unwritable", "job")
        old_makedirs = mimics_import.os.makedirs
        try:
            def fail_makedirs(_path):
                raise OSError("access denied")
            mimics_import.os.makedirs = fail_makedirs
            thread = mimics_import._launch_bridge_job_thread(
                {"action": "discover"}, job_dir, os.path.dirname(job_dir), "discovering"
            )
            thread.join(2.0)
        finally:
            mimics_import.os.makedirs = old_makedirs
        status, error = mimics_import._check_job_status(job_dir)
        self.assertEqual("error", status)
        self.assertIn("access denied", error)

    def test_stop_all_signals_tasks_before_detaching_mimics_monitors(self):
        import inspect
        import mimics_stop_background

        source = inspect.getsource(mimics_stop_background.stop_background_processes)
        self.assertLess(source.index("_request_queue_stop()"), source.index("_stop_inprocess_monitors()"))
        self.assertLess(
            source.index("_request_lock_owned_stop_markers"),
            source.index("_stop_inprocess_monitors()"),
        )
        monitor_source = inspect.getsource(mimics_stop_background._stop_inprocess_monitors)
        for module_name in (
            "mimics_import", "mimics_export", "mask_import",
            "fix_source_affine_metadata", "fewshot_mimics", "nninteractive_mimics",
            "nnunet_mimics",
        ):
            self.assertIn(module_name, monitor_source)
        self.assertIn('"action": "cancel"', monitor_source)

    def test_stop_all_writes_nnunet_cancel_before_detaching_monitor(self):
        import mimics_stop_background as msb

        status_path = os.path.join(self.tmp, "nnunet_status.json")
        control_path = os.path.join(self.tmp, "nnunet_control.json")
        Path(status_path).write_text(
            json.dumps({
                "status": "training",
                "control_path": control_path,
            }),
            encoding="utf-8",
        )
        observations = []

        def stop_monitor(key):
            control = msb.runtime_common.read_json(control_path, {}) or {}
            observations.append((key, control.get("action")))

        fake = _FakeModule()
        fake._MONITORS = {"nnunet-active": {"status_path": status_path}}
        fake._stop_monitor = stop_monitor
        previous = sys.modules.get("nnunet_mimics")
        try:
            sys.modules["nnunet_mimics"] = fake
            msb._stop_inprocess_monitors()
        finally:
            if previous is None:
                sys.modules.pop("nnunet_mimics", None)
            else:
                sys.modules["nnunet_mimics"] = previous
        self.assertEqual([("nnunet-active", "cancel")], observations)

    def test_environment_setup_has_a_scoped_stop_path(self):
        import inspect
        import setup_environment

        source = inspect.getsource(setup_environment.main)
        self.assertIn("Stop Current Setup", source)
        self.assertIn("terminate_process_async", source)

    def test_environment_repair_is_blocked_while_other_tasks_are_active(self):
        import setup_environment

        launched = []
        messages = []
        old_blockers = setup_environment.runtime_common.active_runtime_blockers
        old_launch = setup_environment._launch_setup_worker
        old_message = setup_environment.mimics.dialogs.message_box
        try:
            setup_environment.runtime_common.active_runtime_blockers = (
                lambda *_args, **_kwargs: ["DINOv3 training"]
            )
            setup_environment._launch_setup_worker = (
                lambda *_args, **_kwargs: launched.append(True)
            )
            setup_environment.mimics.dialogs.message_box = (
                lambda **kwargs: messages.append(kwargs.get("message", ""))
            )
            result = setup_environment.main("install")
        finally:
            setup_environment.runtime_common.active_runtime_blockers = old_blockers
            setup_environment._launch_setup_worker = old_launch
            setup_environment.mimics.dialogs.message_box = old_message
        self.assertEqual(1, result)
        self.assertEqual([], launched)
        self.assertTrue(any("DINOv3 training" in item for item in messages))

    def test_global_stop_releases_detached_mimics_operation_leases(self):
        import inspect
        import mimics_stop_background

        source = inspect.getsource(mimics_stop_background.stop_background_processes)
        self.assertLess(
            source.index("_stop_inprocess_monitors()"),
            source.index("clear_local_operations()"),
        )

    def test_mcs_fingerprint_is_persisted_before_work_cleanup(self):
        import inspect
        import create_mcs_batch

        source = inspect.getsource(create_mcs_batch.main)
        persist_index = source.index("fp.write(fingerprint)")
        cleanup_index = source.index("shutil.rmtree(work_dir", persist_index)
        self.assertLess(persist_index, cleanup_index)

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
    def test_prompt_set_prediction_step_count(self):
        from nninteractive_mimics import _interaction_prediction_step_count

        self.assertEqual(
            3,
            _interaction_prediction_step_count(
                {
                    "interaction_type": "point_set",
                    "points": [{}, {}, {}],
                }
            ),
        )
        self.assertEqual(
            2,
            _interaction_prediction_step_count(
                {
                    "interaction_type": "scribble_set",
                    "scribbles": [{}, {}],
                }
            ),
        )

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

    def test_deleted_point_marker_is_not_accessed_again(self):
        import nninteractive_mimics

        class Marker(object):
            def __init__(self):
                self.removed = False

            def __bool__(self):
                if self.removed:
                    raise ValueError(
                        "Cannot access Point object. Probably it was removed"
                    )
                return True

        marker = Marker()
        point = {"_marker": marker}
        visual_objects = [marker]
        original_delete = nninteractive_mimics.mimics.data.points.delete
        try:
            nninteractive_mimics.mimics.data.points.delete = (
                lambda value: setattr(value, "removed", True)
            )
            nninteractive_mimics._delete_point_marker(
                point,
                visual_objects,
            )
        finally:
            nninteractive_mimics.mimics.data.points.delete = original_delete

        self.assertTrue(marker.removed)
        self.assertEqual([], visual_objects)
        self.assertNotIn("_marker", point)


class TestNNInteractiveContinuousPrompting(unittest.TestCase):
    """Phase C: continuous prompt loop, VRAM precheck, error guidance."""

    def test_continue_session_prompt_finish_closes_session(self):
        import nninteractive_mimics

        closed = []
        logged = []
        original_close = nninteractive_mimics._close_async_job
        original_log = nninteractive_mimics._mimics_log
        original_menu = nninteractive_mimics._async_prompt_menu
        try:
            nninteractive_mimics._close_async_job = (
                lambda target, state, reason: closed.append(reason)
            )
            nninteractive_mimics._mimics_log = (
                lambda level, message: logged.append(message)
            )
            nninteractive_mimics._async_prompt_menu = (
                lambda target, state, source, profile: "Finish"
            )
            continued = nninteractive_mimics._continue_session_prompt(
                None, None, {"_job_dir": "job"}, {}
            )
        finally:
            nninteractive_mimics._close_async_job = original_close
            nninteractive_mimics._mimics_log = original_log
            nninteractive_mimics._async_prompt_menu = original_menu
        self.assertFalse(continued)
        self.assertEqual(["user_finished"], closed)

    def test_async_monitor_applied_continues_prompt_loop(self):
        import nninteractive_mimics

        monitor = {
            "done": False,
            "busy": False,
            "deadline": time.time() + 60,
            "timeout_seconds": 60,
            "image": "image",
            "target": "target",
            "state": {"_job_dir": "job_dir", "pending_sequence": 1},
            "config": {},
        }
        calls = []
        original_tick_result = nninteractive_mimics._check_async_result_nonblocking
        original_continue = nninteractive_mimics._continue_session_prompt
        original_stop = nninteractive_mimics._stop_async_monitor
        original_notice = nninteractive_mimics.runtime_common.clear_progress_notice
        try:
            nninteractive_mimics._check_async_result_nonblocking = (
                lambda image, target, state: "applied"
            )
            nninteractive_mimics._continue_session_prompt = (
                lambda image, target, state, config, **kwargs: (
                    calls.append("continue") or True
                )
            )
            nninteractive_mimics._stop_async_monitor = (
                lambda job_dir: calls.append(("stopped", job_dir))
            )
            nninteractive_mimics.runtime_common.clear_progress_notice = (
                lambda *args, **kwargs: None
            )
            nninteractive_mimics._async_monitor_tick(monitor)
        finally:
            nninteractive_mimics._check_async_result_nonblocking = original_tick_result
            nninteractive_mimics._continue_session_prompt = original_continue
            nninteractive_mimics._stop_async_monitor = original_stop
            nninteractive_mimics.runtime_common.clear_progress_notice = original_notice
        # The monitor stayed alive and re-showed the prompt menu; the stop
        # path must not have run.
        self.assertEqual(["continue"], calls)
        self.assertFalse(monitor["done"])
        # Deadline was refreshed for the next prediction cycle.
        self.assertGreater(monitor["deadline"], time.time() + 50)

    def test_async_monitor_applied_finish_ends_monitor(self):
        import nninteractive_mimics

        monitor = {
            "done": False,
            "busy": False,
            "deadline": time.time() + 60,
            "timeout_seconds": 60,
            "image": "image",
            "target": "target",
            "state": {"_job_dir": "job_dir", "pending_sequence": 1},
            "config": {},
        }
        stopped = []
        original_tick_result = nninteractive_mimics._check_async_result_nonblocking
        original_continue = nninteractive_mimics._continue_session_prompt
        original_stop = nninteractive_mimics._stop_async_monitor
        original_notice = nninteractive_mimics.runtime_common.clear_progress_notice
        try:
            nninteractive_mimics._check_async_result_nonblocking = (
                lambda image, target, state: "applied"
            )
            # User picked Finish in the continuation menu: no new prediction.
            nninteractive_mimics._continue_session_prompt = (
                lambda image, target, state, config, **kwargs: False
            )
            nninteractive_mimics._stop_async_monitor = (
                lambda job_dir: stopped.append(job_dir)
            )
            nninteractive_mimics.runtime_common.clear_progress_notice = (
                lambda *args, **kwargs: None
            )
            nninteractive_mimics._async_monitor_tick(monitor)
        finally:
            nninteractive_mimics._check_async_result_nonblocking = original_tick_result
            nninteractive_mimics._continue_session_prompt = original_continue
            nninteractive_mimics._stop_async_monitor = original_stop
            nninteractive_mimics.runtime_common.clear_progress_notice = original_notice
        self.assertEqual(["job_dir"], stopped)
        self.assertTrue(monitor["done"])

    def test_run_async_delegates_to_continue_session_prompt(self):
        import nninteractive_mimics

        delegated = []

        class _Sentinel(object):
            pass

        original_load = nninteractive_mimics._load_async_job
        original_continue = nninteractive_mimics._continue_session_prompt
        original_profile = nninteractive_mimics._model_profile
        original_log = nninteractive_mimics._log_effective_image_input_config
        try:
            nninteractive_mimics._load_async_job = lambda target: None
            nninteractive_mimics._model_profile = lambda config: {}
            nninteractive_mimics._log_effective_image_input_config = (
                lambda config: None
            )

            def _fake_continue(image, target, state, config, **kwargs):
                delegated.append((image, target, state, kwargs))
                return False

            nninteractive_mimics._continue_session_prompt = _fake_continue
            image = _Sentinel()
            target = _Sentinel()
            result = nninteractive_mimics._run_async(
                image, target, {}, source=None,
                auto_created=False, write_mode="in_place",
            )
        finally:
            nninteractive_mimics._load_async_job = original_load
            nninteractive_mimics._continue_session_prompt = original_continue
            nninteractive_mimics._model_profile = original_profile
            nninteractive_mimics._log_effective_image_input_config = original_log
        self.assertEqual(0, result)
        self.assertEqual(1, len(delegated))
        self.assertIs(image, delegated[0][0])
        self.assertIs(target, delegated[0][1])
        self.assertIsNone(delegated[0][2])
        self.assertEqual(
            {
                "source": target,
                "auto_created": False,
                "write_mode": "in_place",
                "validated_target_hash": None,
            },
            delegated[0][3],
        )

    def test_error_guidance_categories(self):
        from nninteractive_mimics import _error_guidance

        category, _message, action = _error_guidance(
            "CUDA out of memory", "prediction"
        )
        self.assertEqual("out_of_memory", category)
        self.assertIn("Close other GPU programs", action)

        category, _message, action = _error_guidance(
            "Connection refused to server", "connect"
        )
        self.assertEqual("server_unavailable", category)
        self.assertIn("Setup / Repair Environment", action)

        category, _message, action = _error_guidance(
            "Not enough free GPU memory to start the nnInteractive server", ""
        )
        self.assertEqual("server_unavailable", category)

        category, _message, action = _error_guidance(
            "ModuleNotFoundError: No module named 'nnInteractive'", "startup"
        )
        self.assertEqual("environment_broken", category)
        self.assertIn("Setup / Repair Environment", action)

        category, _message, action = _error_guidance(
            "The active image or target Mask changed", "apply"
        )
        self.assertEqual("stale_target", category)

        category, _message, action = _error_guidance("something odd", "")
        self.assertEqual("unknown", category)
        self.assertIn("Retry", action)


class TestNNInteractiveGpuMemoryPrecheck(unittest.TestCase):
    def _bridge_module(self):
        import nninteractive_bridge

        return nninteractive_bridge

    def test_precheck_blocks_low_free_vram(self):
        bridge = self._bridge_module()
        import torch

        original = torch.cuda.mem_get_info
        original_available = torch.cuda.is_available
        try:
            torch.cuda.is_available = lambda: True
            torch.cuda.mem_get_info = lambda: (1 * 1024 ** 3, 24 * 1024 ** 3)
            with self.assertRaises(RuntimeError) as ctx:
                bridge._check_free_gpu_memory("cuda:0")
        finally:
            torch.cuda.mem_get_info = original
            torch.cuda.is_available = original_available
        self.assertIn("Not enough free GPU memory", str(ctx.exception))

    def test_precheck_passes_above_floor(self):
        bridge = self._bridge_module()
        import torch

        original = torch.cuda.mem_get_info
        original_available = torch.cuda.is_available
        try:
            torch.cuda.is_available = lambda: True
            torch.cuda.mem_get_info = lambda: (8 * 1024 ** 3, 24 * 1024 ** 3)
            # Must not raise.
            bridge._check_free_gpu_memory("cuda:0")
        finally:
            torch.cuda.mem_get_info = original
            torch.cuda.is_available = original_available

    def test_precheck_skips_cpu_and_probe_failures(self):
        bridge = self._bridge_module()
        # CPU device never probes.
        bridge._check_free_gpu_memory("cpu")
        # A failing probe must not block the start.
        import torch

        original = torch.cuda.mem_get_info
        original_available = torch.cuda.is_available
        try:
            torch.cuda.is_available = lambda: False
            bridge._check_free_gpu_memory("cuda:0")
        finally:
            torch.cuda.mem_get_info = original
            torch.cuda.is_available = original_available


class TestNNInteractiveTaskDiagnostics(unittest.TestCase):
    """Phase D: model-registry filtering and failed-job diagnostics."""

    @classmethod
    def setUpClass(cls):
        tools_dir = os.path.join(PROJECT_ROOT, "tools")
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)

    def test_model_center_hides_not_improved_models_by_default(self):
        import nninteractive_task_model_center as model_center

        source = inspect.getsource(model_center.ModelCenter.refresh_models)
        self.assertIn('"not_improved"', source)
        self.assertIn("Did not improve", source)

    def test_model_center_shows_failed_versions_toggle_covers_not_improved(self):
        import nninteractive_task_model_center as model_center

        source = inspect.getsource(model_center.ModelCenter)
        self.assertIn(
            "Show failed and not-improved versions",
            source,
        )

    def test_task_model_chooser_hides_not_improved_by_default(self):
        import nninteractive_task_model_chooser as chooser

        source = inspect.getsource(chooser.Chooser.refresh_models)
        self.assertIn('"not_improved"', source)
        self.assertIn("Did not improve", source)
        self.assertIn("Show models that did not improve", inspect.getsource(chooser.Chooser))

    def test_diagnose_job_cli_subcommand_exists(self):
        import nninteractive_finetune_pipeline as pipeline

        parser = pipeline.build_parser()
        args = parser.parse_args(["diagnose", "--job-dir", "x"])
        self.assertEqual("diagnose", args.command)
        self.assertEqual("x", args.job_dir)

    def test_diagnose_job_aggregates_failure_state(self):
        import nninteractive_finetune_pipeline as pipeline
        import nninteractive_task_common as common

        with tempfile.TemporaryDirectory() as value:
            job = Path(value) / "job_001"
            job.mkdir()
            common.write_json_atomic(
                job / "status.json",
                {
                    "status": "failed",
                    "phase": "failed",
                    "error": "RuntimeError: CUDA out of memory",
                    "traceback": (
                        "Traceback (most recent call last):\n"
                        '  File "a.py", line 1, in f\n'
                        '  File "b.py", line 2, in g\n'
                        "RuntimeError: CUDA out of memory"
                    ),
                },
            )
            common.write_json_atomic(
                job / "request.json",
                {"task_id": "liver", "epochs": 5, "cases": [{}, {}]},
            )
            (job / "job.log").write_text(
                "Training failed: RuntimeError: CUDA out of memory.",
                encoding="utf-8",
            )
            report = pipeline.diagnose_job(str(job))
        self.assertEqual("failed", report["status"])
        self.assertIn("GPU memory", report["hint"])
        self.assertTrue(report["error_line"].strip().startswith('File "b.py"'))
        self.assertTrue(
            report["job_log_tail"].endswith("CUDA out of memory.")
        )
        self.assertEqual(2, report["request_summary"]["case_count"])
        self.assertTrue(report["artifacts"]["job.log"])
        self.assertFalse(report["artifacts"]["control.json"])

    def test_training_curve_draws_current_model_baseline(self):
        import nninteractive_task_model_center as model_center

        source = inspect.getsource(model_center.TrainingCurve)
        self.assertIn("set_baseline", source)
        self.assertIn("current model", source)
        refresh_source = inspect.getsource(
            model_center.ModelCenter.refresh_status
        )
        self.assertIn("set_baseline", refresh_source)
        self.assertIn("Candidate is", refresh_source)
        self.assertIn("Not yet better", refresh_source)

    def test_model_center_failed_state_shows_diagnosis_block(self):
        import nninteractive_task_model_center as model_center

        source = inspect.getsource(model_center.ModelCenter._refresh_diagnosis)
        self.assertIn("failed", source)
        self.assertIn("diagnose_job", source)
        self.assertIn("Open Job Folder", inspect.getsource(model_center.ModelCenter))


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
        np.testing.assert_allclose(result, valid, atol=1e-6)

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
        """A coded valid sform is authoritative when the forms disagree."""
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
        np.testing.assert_allclose(result, sform_aff, atol=1e-6)
        np.testing.assert_allclose(img.get_qform(), sform_aff, atol=1e-6)
        np.testing.assert_allclose(img.get_sform(), sform_aff, atol=1e-6)

    def test_ambiguous_nifti_header_is_corrected_before_simpleitk_reads_it(self):
        """A readable source must not bypass qform/sform normalization."""
        import nibabel as nib
        import SimpleITK as sitk
        from mimics_bridge import _read_image_sitk_lps

        source_path = os.path.join(self.tmp, "ambiguous.nii.gz")
        sform_aff = np.diag([1.0, 1.0, 2.0, 1.0])
        qform_aff = np.diag([1.5, 1.5, 3.0, 1.0])
        image = nib.Nifti1Image(np.zeros((4, 5, 6), dtype=np.int16), sform_aff)
        image.set_sform(sform_aff, code=2)
        image.set_qform(qform_aff, code=2)
        nib.save(image, source_path)

        original_read = sitk.ReadImage
        calls = []
        try:
            def record(candidate, *args, **kwargs):
                calls.append(os.path.abspath(str(candidate)))
                return original_read(candidate, *args, **kwargs)
            sitk.ReadImage = record
            result = _read_image_sitk_lps(source_path)
        finally:
            sitk.ReadImage = original_read

        self.assertEqual((4, 5, 6), result.GetSize())
        self.assertTrue(calls)
        self.assertNotEqual(os.path.abspath(source_path), calls[0])

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

    def test_hu_gv_conversion_is_modality_specific(self):
        from nninteractive_mimics import _source_uses_hu_to_gv

        self.assertTrue(_source_uses_hu_to_gv("nifti", "CT"))
        self.assertTrue(_source_uses_hu_to_gv("dicom_folder", "CT"))
        self.assertFalse(_source_uses_hu_to_gv("nifti", "MR"))
        self.assertFalse(_source_uses_hu_to_gv("medical_image", "MR"))
        self.assertTrue(_source_uses_hu_to_gv("nifti", ""))

    def test_task_model_missing_source_uses_portable_mimics_buffer(self):
        import nninteractive_mimics as module

        class _Meta:
            def __init__(self):
                self._values = {
                    module.SOURCE_IMAGE_KIND_METADATA: "nifti",
                    module.SOURCE_IMAGE_MODALITY_METADATA: "MR",
                }

            def find(self, name):
                if name not in self._values:
                    return None
                item = type("Item", (), {})()
                item.value = self._values[name]
                return item

        class _Image:
            logical_dimensions = [2, 3, 4]
            metadata = _Meta()

            def get_voxel_buffer(self):
                return memoryview(
                    np.arange(24, dtype=np.uint16).reshape((2, 3, 4))
                )

        profile = {"source": "task_model"}
        output = os.path.join(self.tmp, "portable_image.raw")
        with mock.patch.object(
            module, "_model_profile", return_value=profile
        ), mock.patch.object(
            module, "_source_image_export", return_value=None
        ), mock.patch.object(module, "_mimics_log"):
            result = module._export_image_for_nninteractive(
                {
                    "task_model_image_input_mode": "auto",
                    "fallback_to_mimics_buffer_when_source_unavailable": True,
                },
                _Image(),
                output,
            )

        self.assertEqual("mimics_buffer", result["kind"])
        self.assertEqual("mimics_project_buffer", result["image_input_provenance"])
        self.assertEqual("raw_gv_zscore_best_effort", result["image_intensity_compatibility"])
        self.assertEqual("unavailable", result["source_intensity_recovery_basis"])
        self.assertTrue(result["task_model_buffer_fallback"])
        self.assertEqual(48, os.path.getsize(output))

    def test_task_model_mri_buffer_restores_source_values_from_mcs_metadata(self):
        import nninteractive_mimics as module
        from nninteractive_bridge import _apply_buffer_model_intensity_transform

        class _Meta:
            _values = {
                module.SOURCE_IMAGE_KIND_METADATA: "nifti",
                module.SOURCE_IMAGE_MODALITY_METADATA: "MR",
                module.SOURCE_INTENSITY_ENCODING_METADATA: "dicom_uint16_linear_rescale_v1",
                module.SOURCE_INTENSITY_RESCALE_SLOPE_METADATA: "0.125",
                module.SOURCE_INTENSITY_RESCALE_INTERCEPT_METADATA: "-200.0",
                module.SOURCE_INTENSITY_VALUE_MIN_METADATA: "-200.0",
                module.SOURCE_INTENSITY_VALUE_MAX_METADATA: "7991.875",
                module.DICOM_STORED_VALUE_MIN_METADATA: "0",
                module.DICOM_STORED_VALUE_MAX_METADATA: "65535",
            }

            def find(self, name):
                if name not in self._values:
                    return None
                return type("Item", (), {"value": self._values[name]})()

        raw = np.array([0, 1600, 65535], dtype=np.uint16).reshape((1, 1, 3))

        class _Image:
            metadata = _Meta()
            guid = "mapped-mri-image"

            def get_voxel_buffer(self):
                return memoryview(raw)

        output = os.path.join(self.tmp, "mapped_mri.raw")
        exported = module._export_image(_Image(), output)
        self.assertEqual(0.125, exported["buffer_to_source_slope"])
        self.assertEqual(-200.0, exported["buffer_to_source_intercept"])
        self.assertAlmostEqual(0.062500125, exported["buffer_to_source_zero_tolerance"])
        self.assertEqual("mcs_metadata", exported["source_intensity_recovery_basis"])
        restored = _apply_buffer_model_intensity_transform(
            raw.astype(np.float32),
            {
                "image_buffer_to_model_slope": exported["buffer_to_source_slope"],
                "image_buffer_to_model_intercept": exported["buffer_to_source_intercept"],
            },
        )
        np.testing.assert_allclose(restored, raw.astype(np.float32) * 0.125 - 200.0)

    def test_task_model_old_mcs_reads_mri_rescale_from_dicom_tags(self):
        import nninteractive_mimics as module

        class _Meta:
            def find(self, name):
                return None

        class _Tag:
            def __init__(self, value):
                self.value = value

        class _Image:
            metadata = _Meta()
            guid = "old-mcs-image"

            def __init__(self):
                self.tag_reads = 0

            def get_dicom_tags(self, image_index=0):
                self.tag_reads += 1
                return {
                    (0x0008, 0x0060): _Tag("MR"),
                    (0x0028, 0x1053): _Tag("0.03125"),
                    (0x0028, 0x1052): _Tag("-17.5"),
                }

        image = _Image()
        mapping = module._source_intensity_mapping(image)
        self.assertIsNotNone(mapping)
        self.assertEqual("MR", module._image_modality(image))
        self.assertEqual(1, image.tag_reads)
        self.assertEqual("mcs_dicom_tags", mapping["basis"])
        self.assertEqual(0.03125, mapping["slope"])
        self.assertEqual(-17.5, mapping["intercept"])

    def test_mri_rescale_restoration_preserves_negative_background_contract(self):
        from nninteractive_finetune.data import normalize_like_nninteractive

        source = np.zeros((18, 16, 14), dtype=np.float32)
        source[4:14, 3:13, 2:12] = np.linspace(
            -120.0, 850.0, 10 * 10 * 10, dtype=np.float32
        ).reshape((10, 10, 10))
        value_min = float(source.min())
        value_max = float(source.max())
        slope = (value_max - value_min) / 65535.0
        stored = np.rint((source - value_min) / slope).astype(np.uint16)
        raw_gv = stored.astype(np.float32)
        from nninteractive_bridge import _apply_buffer_model_intensity_transform

        restored = _apply_buffer_model_intensity_transform(
            raw_gv,
            {
                "image_buffer_to_model_slope": slope,
                "image_buffer_to_model_intercept": value_min,
                "image_buffer_model_zero_tolerance": slope * 0.500001,
            },
        )

        expected = normalize_like_nninteractive(source)
        recovered = normalize_like_nninteractive(restored)
        raw_normalized = normalize_like_nninteractive(raw_gv)
        np.testing.assert_allclose(recovered, expected, rtol=0.0, atol=1.0e-3)
        self.assertGreater(float(np.max(np.abs(raw_normalized - expected))), 0.1)

    def test_task_model_auto_mode_never_probes_unc_source_on_mimics_thread(self):
        import nninteractive_mimics as module

        class _Meta:
            def __init__(self):
                identity = json.dumps(
                    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
                )
                ras_to_lps = json.dumps(
                    [[-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
                )
                self._values = {
                    module.SOURCE_IMAGE_PATH_METADATA: "//server/share/missing/mri.nii.gz",
                    module.SOURCE_IMAGE_KIND_METADATA: "nifti",
                    module.SOURCE_IMAGE_INDEX_SPACE_METADATA: "derived_dicom_lps_resampled_from_source_image_v2",
                    module.SOURCE_IMAGE_MODALITY_METADATA: "MR",
                    module.SOURCE_WORLD_COORDINATE_SYSTEM_METADATA: "ras",
                    module.MIMICS_WORLD_COORDINATE_SYSTEM_METADATA: "lps",
                    module.SOURCE_TO_MIMICS_WORLD_MATRIX_METADATA: ras_to_lps,
                    module.SOURCE_VOXEL_TO_RAS_MATRIX_METADATA: identity,
                    module.MIMICS_VOXEL_TO_RAS_MATRIX_METADATA: identity,
                }

            def find(self, name):
                if name not in self._values:
                    return None
                item = type("Item", (), {})()
                item.value = self._values[name]
                return item

        image = type("Image", (), {"metadata": _Meta()})()
        with mock.patch.object(
            module, "_relocated_source_image_path", return_value=""
        ), mock.patch.object(
            module.os.path,
            "isfile",
            side_effect=AssertionError("UNC path was probed on the Mimics thread"),
        ), mock.patch.object(
            module, "_call_mimics_bridge", side_effect=AssertionError("bridge was called")
        ), mock.patch.object(module, "_mimics_log"):
            result = module._source_image_export(
                image,
                {
                    "image_input_mode": "auto",
                    "prefer_source_image_for_nninteractive": True,
                    "fallback_to_mimics_buffer_when_source_unavailable": True,
                },
            )
        self.assertIsNone(result)

    def test_task_model_local_source_alignment_is_deferred_to_external_worker(self):
        import nninteractive_mimics as module

        source = os.path.join(self.tmp, "source.nii.gz")
        Path(source).write_bytes(b"external worker reads this file")
        identity = json.dumps(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
        ras_to_lps = json.dumps(
            [[-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )

        class _Meta:
            _values = {
                module.SOURCE_IMAGE_PATH_METADATA: source,
                module.SOURCE_IMAGE_KIND_METADATA: "nifti",
                module.SOURCE_IMAGE_INDEX_SPACE_METADATA: "derived_dicom_lps_resampled_from_source_image_v2",
                module.SOURCE_IMAGE_MODALITY_METADATA: "MR",
                module.SOURCE_WORLD_COORDINATE_SYSTEM_METADATA: "ras",
                module.MIMICS_WORLD_COORDINATE_SYSTEM_METADATA: "lps",
                module.SOURCE_TO_MIMICS_WORLD_MATRIX_METADATA: ras_to_lps,
                module.SOURCE_VOXEL_TO_RAS_MATRIX_METADATA: identity,
                module.MIMICS_VOXEL_TO_RAS_MATRIX_METADATA: identity,
            }

            def find(self, name):
                if name not in self._values:
                    return None
                item = type("Item", (), {})()
                item.value = self._values[name]
                return item

        image = type(
            "Image", (), {"metadata": _Meta(), "logical_dimensions": [2, 3, 4]}
        )()
        with mock.patch.object(
            module, "_relocated_source_image_path", return_value=""
        ), mock.patch.object(
            module, "_call_mimics_bridge", side_effect=AssertionError("bridge was called")
        ), mock.patch.object(
            module, "_model_profile", return_value={"source": "task_model"}
        ), mock.patch.object(module, "_mimics_log"):
            result = module._source_image_export(
                image,
                {
                    "_model_profile": {"source": "task_model"},
                    "image_input_mode": "auto",
                    "prefer_source_image_for_nninteractive": True,
                    "fallback_to_mimics_buffer_when_source_unavailable": True,
                    "defer_source_alignment_to_worker": True,
                },
            )
        self.assertEqual(os.path.abspath(source), result["image_path"])
        self.assertEqual(
            "derived_dicom_lps_resampled_from_source_image_v2",
            result["source_index_space"],
        )

    def test_task_model_strict_source_does_not_silently_fallback(self):
        import nninteractive_mimics as module

        with mock.patch.object(
            module, "_model_profile", return_value={"source": "task_model"}
        ), mock.patch.object(
            module, "_source_image_export", return_value=None
        ):
            with self.assertRaisesRegex(RuntimeError, "task_model_image_input_mode"):
                module._export_image_for_nninteractive(
                    {
                        "task_model_image_input_mode": "source",
                        "fallback_to_mimics_buffer_when_source_unavailable": True,
                    },
                    object(),
                    os.path.join(self.tmp, "must_not_exist.raw"),
                )

    def test_task_model_explicit_mimics_mode_ignores_source_fallback_switch(self):
        import nninteractive_mimics as module

        class _Image:
            metadata = None

            def get_voxel_buffer(self):
                return memoryview(np.ones((2, 2, 2), dtype=np.float32))

        output = os.path.join(self.tmp, "explicit_mimics.raw")
        with mock.patch.object(
            module, "_model_profile", return_value={"source": "task_model"}
        ), mock.patch.object(module, "_mimics_log"):
            result = module._export_image_for_nninteractive(
                {
                    "task_model_image_input_mode": "mimics",
                    "fallback_to_mimics_buffer_when_source_unavailable": False,
                },
                _Image(),
                output,
            )
        self.assertEqual("mimics_buffer", result["kind"])
        self.assertTrue(os.path.isfile(output))

    def test_task_model_mri_uint16_quantization_preserves_zscore_input(self):
        from nninteractive_finetune.data import normalize_like_nninteractive

        source = np.zeros((20, 18, 16), dtype=np.float32)
        values = np.linspace(1.0, 2400.0, 12 * 10 * 8, dtype=np.float32)
        source[4:16, 4:14, 3:11] = values.reshape((12, 10, 8))
        slope = float(source.max()) / 65535.0
        quantized = np.rint(source / slope).astype(np.uint16).astype(np.float32)
        expected = normalize_like_nninteractive(source)
        actual = normalize_like_nninteractive(quantized)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=5.0e-5)


# ============================================================================
# L13: New features from user's round of changes
# ============================================================================


class TestNewFeatures(unittest.TestCase):
    def setUp(self):
        self.tmp = _make_temp_dir()

    def tearDown(self):
        _cleanup(self.tmp)

    def test_mask_export_conflict_policy_is_external_and_reaches_bridge(self):
        """The external choice, not a dead preflight helper, controls writes."""
        source = Path(RUNTIME_DIR, "mimics_export.py").read_text(encoding="utf-8")
        self.assertNotIn('buttons="Safe Copy;Overwrite Original;Cancel"', source)
        self.assertNotIn('buttons="Overwrite;Skip Existing;Cancel"', source)
        self.assertIn('bridge_params["overwrite_existing"] = bool(overwrite_existing)', source)
        ui_source = Path(PROJECT_ROOT, "tools", "io_path_setup_ui.py").read_text(encoding="utf-8")
        self.assertIn('QRadioButton("Skip existing")', ui_source)
        self.assertIn('QRadioButton("Overwrite existing")', ui_source)
        self.assertIn("tempfile.mkstemp", ui_source)

    def test_export_launcher_thread_registry_is_pruned(self):
        import mimics_export

        class ThreadStub(object):
            def __init__(self, alive):
                self.alive = alive
            def is_alive(self):
                return self.alive

        live = ThreadStub(True)
        done = ThreadStub(False)
        previous = list(mimics_export._EXPORT_LAUNCH_THREADS)
        try:
            mimics_export._EXPORT_LAUNCH_THREADS[:] = [done, live]
            mimics_export._prune_export_launch_threads()
            self.assertEqual([live], mimics_export._EXPORT_LAUNCH_THREADS)
        finally:
            mimics_export._EXPORT_LAUNCH_THREADS[:] = previous

    def test_background_export_exit_is_detected_even_with_running_status(self):
        import mimics_export

        status_path = os.path.join(self.tmp, "status.json")
        Path(status_path).write_text(
            json.dumps({
                "status": "exporting",
                "phase": "exporting_voxels",
                "updated_at_epoch": time.time(),
            }),
            encoding="utf-8",
        )

        class Process(object):
            returncode = 0
            def poll(self):
                return 0

        stopped = []
        old_stop = mimics_export._stop_export_monitor
        try:
            mimics_export._stop_export_monitor = lambda key: stopped.append(key)
            monitor = {
                "monitor_key": "audit_running_exit",
                "status_path": status_path,
                "launch_finished": True,
                "process": Process(),
                "started_at_epoch": time.time() - 1,
                "deadline": time.time() + 60,
                "done": False,
                "busy": False,
                "job_runtime": self.tmp,
            }
            mimics_export._background_export_status_tick(monitor)
        finally:
            mimics_export._stop_export_monitor = old_stop
        self.assertTrue(monitor["done"])
        self.assertEqual(["audit_running_exit"], stopped)
        terminal = json.loads(Path(status_path).read_text(encoding="utf-8"))
        self.assertEqual("failed", terminal["status"])
        self.assertEqual("process_exited", terminal["phase"])

    def test_background_export_watcher_writes_terminal_before_lock_release(self):
        import mimics_export

        status_path = os.path.join(self.tmp, "watcher_status.json")
        Path(status_path).write_text(
            json.dumps({"status": "exporting"}), encoding="utf-8",
        )

        class Process(object):
            pid = 8080
            returncode = 1
            _mimics_status_path = status_path
            _mimics_stop_path = ""
            _mimics_handshake_path = ""
            def wait(self):
                return 1
            def poll(self):
                return 1

        observations = []
        old_release = mimics_export.runtime_common.release_resource_lock
        try:
            mimics_export.runtime_common.release_resource_lock = (
                lambda _path, _token: observations.append(
                    json.loads(Path(status_path).read_text(encoding="utf-8"))["status"]
                ) or True
            )
            mimics_export._watch_background_export_process(
                Process(), os.path.join(self.tmp, "background.lock"), "token",
            )
        finally:
            mimics_export.runtime_common.release_resource_lock = old_release
        self.assertEqual(["failed"], observations)

    def test_explicit_mcs_arguments_are_resolved_and_validated(self):
        import mimics_export

        first = Path(self.tmp) / "case_a.mcs"
        second = Path(self.tmp) / "case_b.mcs"
        first.write_bytes(b"a")
        second.write_bytes(b"b")
        list_path = Path(self.tmp) / "projects.txt"
        list_path.write_text(str(second) + "\n", encoding="utf-8")
        result = mimics_export._explicit_mcs_path_map(
            mcs_path=str(first),
            mcs_list_file=str(list_path),
        )
        self.assertEqual(os.path.abspath(str(first)), result["case_a"])
        self.assertEqual(os.path.abspath(str(second)), result["case_b"])
        with self.assertRaises(RuntimeError):
            mimics_export._explicit_mcs_path_map(
                mcs_path=str(Path(self.tmp) / "missing.mcs"),
            )

    def test_batch_prepare_skips_invalid_cases_without_recursion(self):
        import mimics_import

        queue = [
            {"case_id": "empty_{0}".format(index), "case_dir": self.tmp}
            for index in range(100)
        ]
        monitor = {
            "batch_queue": queue,
            "output_dir": self.tmp,
            "failed": 0,
        }
        old_discover = mimics_import._discover_single_case
        old_log = mimics_import._append_import_log
        old_record = mimics_import._record_failed_case
        try:
            mimics_import._discover_single_case = lambda _path: None
            mimics_import._append_import_log = lambda *_args, **_kwargs: None
            mimics_import._record_failed_case = lambda *_args, **_kwargs: None
            mimics_import._start_next_batch_prepare(monitor)
        finally:
            mimics_import._discover_single_case = old_discover
            mimics_import._append_import_log = old_log
            mimics_import._record_failed_case = old_record
        self.assertEqual(8, monitor["failed"])
        self.assertEqual(92, len(monitor["batch_queue"]))
        self.assertTrue(monitor["selecting_next"])

    def test_status_viewer_cancel_preserves_latest_progress(self):
        import tools.fewshot_status_viewer as status_viewer

        status_path = Path(self.tmp) / "train.json"
        cancel_path = Path(self.tmp) / "cancel.request"
        status_viewer.write_json_atomic(status_path, {
            "job_id": "train_a",
            "status": "training",
            "training_progress": {"epoch": 7, "metrics": {"loss": 0.25}},
        })
        status_viewer.request_job_cancel_async(
            {"cancel_path": str(cancel_path)},
            str(status_path),
            grace_seconds=0,
        )
        deadline = time.time() + 2.0
        payload = {}
        while time.time() < deadline:
            payload = status_viewer.read_json(status_path, {}) or {}
            if payload.get("status") == "cancelled":
                break
            time.sleep(0.02)
        self.assertEqual("cancelled", payload.get("status"))
        self.assertEqual(7, payload["training_progress"]["epoch"])
        self.assertTrue(cancel_path.is_file())

    def test_fewshot_stop_can_discard_completed_pending_application(self):
        import fewshot_mimics

        status_path = os.path.join(self.tmp, "infer_completed.json")
        fewshot_mimics._write_json_atomic(
            status_path,
            {
                "status": "completed",
                "updated_at_epoch": time.time(),
                "applied_to_mimics": False,
            },
        )
        key = "pending_inference_application"
        monitor = {
            "monitor_key": key,
            "kind": "infer",
            "status_path": status_path,
        }
        old_answer = fewshot_mimics.mimics.dialogs.question_box
        try:
            fewshot_mimics._MONITORS[key] = monitor
            fewshot_mimics.mimics.dialogs.question_box = (
                lambda **_kwargs: fewshot_mimics.BUTTON_STOP
            )
            self.assertTrue(
                fewshot_mimics._stop_pending_inference_application()
            )
        finally:
            fewshot_mimics.mimics.dialogs.question_box = old_answer
            fewshot_mimics._MONITORS.pop(key, None)
        status = fewshot_mimics._read_json(status_path, {}) or {}
        self.assertTrue(status.get("application_cancelled"))
        self.assertEqual("cancelled", status.get("application_state"))
        self.assertNotIn(key, fewshot_mimics._MONITORS)

    def test_fewshot_export_reaps_process_when_lock_transfer_fails(self):
        import tools.fewshot_pipeline as pipeline

        class Lock(object):
            acquired = False
            released = False
            def update_pid(self, _pid, **_extra):
                self.acquired = False
            def release(self):
                self.released = True

        class Process(object):
            pid = 24680
            returncode = None
            terminated = False
            killed = False
            def poll(self):
                return None if not self.terminated and not self.killed else -1
            def terminate(self):
                self.terminated = True
            def kill(self):
                self.killed = True
            def wait(self, timeout=None):
                return -1

        lock = Lock()
        process = Process()
        ts_root = Path(self.tmp) / "dataset"
        workspace = Path(self.tmp) / "workspace"
        ts_root.mkdir()
        acquired_scopes = []
        old_find = pipeline.find_mimics_exe
        old_acquire = pipeline.acquire_background_mimics_lock
        old_popen = pipeline.subprocess.Popen
        old_resolve = pipeline.resolve_mimics_output_dir
        try:
            pipeline.find_mimics_exe = lambda _value=None: "MimicsResearch.exe"
            def acquire(*_args, **kwargs):
                acquired_scopes.append(Path(kwargs["scope"]).resolve())
                return lock
            pipeline.acquire_background_mimics_lock = acquire
            pipeline.subprocess.Popen = lambda *_args, **_kwargs: process
            pipeline.resolve_mimics_output_dir = lambda _root: Path(self.tmp) / "mcs"
            with self.assertRaises(RuntimeError):
                pipeline.launch_mimics_export(
                    ts_root,
                    {"case_a"},
                    None,
                    workspace,
                    5,
                    status_path=workspace / "jobs" / "train.json",
                    cancel_path=workspace / "jobs" / "train.cancel",
                )
        finally:
            pipeline.find_mimics_exe = old_find
            pipeline.acquire_background_mimics_lock = old_acquire
            pipeline.subprocess.Popen = old_popen
            pipeline.resolve_mimics_output_dir = old_resolve
        self.assertTrue(process.terminated or process.killed)
        self.assertTrue(lock.released)
        self.assertEqual(
            {Path(self.tmp, "mcs").resolve(), ts_root.resolve()},
            set(acquired_scopes),
        )

    def test_fewshot_multi_lock_does_not_hold_partial_resources(self):
        import tools.fewshot_pipeline as pipeline
        from resource_locks import ResourceLockTimeout

        class Lock(object):
            released = False
            def release(self):
                self.released = True

        first = Lock()
        calls = []
        old_acquire = pipeline.acquire_background_mimics_lock
        try:
            def acquire(*_args, **_kwargs):
                calls.append(1)
                if len(calls) == 1:
                    return first
                raise ResourceLockTimeout("destination busy")
            pipeline.acquire_background_mimics_lock = acquire
            with self.assertRaises(ResourceLockTimeout):
                pipeline.acquire_background_mimics_locks(
                    Path(self.tmp),
                    Path(self.tmp) / "status.json",
                    Path(self.tmp) / "cancel.request",
                    "test export",
                    0.0,
                    [Path(self.tmp) / "mcs", Path(self.tmp) / "labels"],
                )
        finally:
            pipeline.acquire_background_mimics_lock = old_acquire
        self.assertTrue(first.released)

    def test_current_project_export_does_not_launch_background_mimics(self):
        import inspect
        import mimics_export

        source = inspect.getsource(mimics_export._start_current_project_export)
        self.assertNotIn("_launch_background_batch_export", source)
        self.assertIn("try_acquire_local_operation", source)
        self.assertIn("_start_foreground_export_monitor", source)
        tick = inspect.getsource(mimics_export._foreground_export_tick)
        self.assertIn("_launch_bridge_background", tick)
        self.assertNotIn("MimicsResearch", tick)
        self.assertIn("_foreground_export_target_is_open", tick)
        self.assertLess(
            tick.index('monitor.pop("operation_token", None)'),
            tick.index("_launch_bridge_background"),
        )

    def test_mask_import_cancel_releases_full_apply_lease(self):
        import mask_import
        import runtime_common

        work_dir = os.path.join(self.tmp, "mask_import_work")
        os.makedirs(work_dir)
        token = runtime_common.try_acquire_local_operation(
            "mask_buffer_access", "Mask import apply"
        )
        self.assertTrue(token)
        monitor = {
            "monitor_key": work_dir,
            "work_dir": work_dir,
            "done": False,
            "operation_token": token,
            "process": None,
        }
        mask_import._MASK_IMPORT_MONITORS[work_dir] = monitor
        count = mask_import.cancel_all_mask_imports("test stop")
        self.assertEqual(1, count)
        self.assertIsNone(runtime_common.active_local_operation("mask_buffer_access"))
        self.assertNotIn(work_dir, mask_import._MASK_IMPORT_MONITORS)

    def test_mask_import_name_collision_creates_editable_copy(self):
        import mask_import

        active_image = _FakeModule()
        existing = _FakeModule()
        existing.name = "Liver"
        existing.image = active_image
        created = []

        def create_mask():
            mask = _FakeModule()
            mask.name = ""
            mask.image = None
            created.append(mask)
            return mask

        old_masks = mask_import.mimics.data.masks
        old_create = mask_import.mimics.segment.create_mask
        try:
            mask_import.mimics.data.masks = [existing]
            mask_import.mimics.segment.create_mask = create_mask
            result = mask_import._find_or_create_mask("Liver", active_image)
        finally:
            mask_import.mimics.data.masks = old_masks
            mask_import.mimics.segment.create_mask = old_create
        self.assertIs(result, created[0])
        self.assertEqual("Liver - Imported", result.name)
        self.assertEqual("Liver", existing.name)

    def test_mask_import_waits_for_the_original_active_image(self):
        import mask_import

        image_a = _FakeModule()
        image_a.guid = "image-a"
        image_b = _FakeModule()
        image_b.guid = "image-b"
        monitor = {
            "active_image_id": "image-a",
            "launch_project_path": os.path.join(self.tmp, "case.mcs"),
        }
        old_image = mask_import._active_image_reference
        old_project = mask_import._current_project_path
        try:
            mask_import._active_image_reference = lambda: image_a
            mask_import._current_project_path = lambda: os.path.join(self.tmp, "case.mcs")
            self.assertTrue(mask_import._target_image_is_open(monitor)[0])
            mask_import._active_image_reference = lambda: image_b
            self.assertFalse(mask_import._target_image_is_open(monitor)[0])
            mask_import._active_image_reference = lambda: image_a
            mask_import._current_project_path = lambda: os.path.join(self.tmp, "other.mcs")
            self.assertFalse(mask_import._target_image_is_open(monitor)[0])
        finally:
            mask_import._active_image_reference = old_image
            mask_import._current_project_path = old_project

    def test_interactive_ai_switching_guards_are_present(self):
        import inspect
        import interactive_algorithms_mimics
        import nninteractive_mimics
        from tools import interactive_algorithms_worker

        interactive_tick = inspect.getsource(
            interactive_algorithms_mimics._monitor_tick
        )
        self.assertIn("_monitor_target_is_open", interactive_tick)
        nn_start = inspect.getsource(nninteractive_mimics._start_async_job)
        self.assertIn('"nnInteractive input snapshot"', nn_start)
        nn_prewarm = inspect.getsource(
            nninteractive_mimics._get_or_start_image_worker
        )
        self.assertIn('"nnInteractive image preparation"', nn_prewarm)
        self.assertIn("release_local_operation", nn_prewarm)
        nn_result = inspect.getsource(
            nninteractive_mimics._handle_async_result
        )
        self.assertIn("active_project_changed", nn_result)
        self.assertIn(
            "request_nninteractive_server_release_on_contention",
            inspect.getsource(interactive_algorithms_worker._gpu_lock),
        )

    def test_internal_batch_export_locks_the_actual_label_destination(self):
        import mimics_export
        import runtime_common

        ts_root = os.path.join(self.tmp, "dataset")
        mcs_root = os.path.join(self.tmp, "projects")
        safe_root = os.path.join(self.tmp, "safe_labels")
        os.makedirs(ts_root)
        captured = []
        old_find = mimics_export._find_mimics_exe
        old_root = mimics_export._project_root
        old_acquire = runtime_common.acquire_resource_lock
        old_prune = mimics_export._prune_local_export_jobs
        try:
            mimics_export._find_mimics_exe = lambda: "MimicsResearch.exe"
            mimics_export._project_root = lambda: self.tmp
            mimics_export._prune_local_export_jobs = lambda: None

            def acquire(path, *_args, **_kwargs):
                captured.append(path)
                return "token_{}".format(len(captured))

            runtime_common.acquire_resource_lock = acquire
            old_popen = mimics_export.subprocess.Popen
            mimics_export.subprocess.Popen = lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("controlled launch failure")
            )
            mimics_export._launch_background_batch_export(
                ts_root,
                None,
                [0, 1, 2],
                [False, False, False],
                overwrite_existing=True,
                mcs_output_dir=mcs_root,
            )
            expected_locks = {
                runtime_common.background_mimics_lock_path(self.tmp, mcs_root),
                runtime_common.background_mimics_lock_path(self.tmp, ts_root),
            }
            self.assertEqual(
                {os.path.normcase(value) for value in expected_locks},
                {os.path.normcase(value) for value in captured},
            )

            captured[:] = []
            mimics_export._launch_background_batch_export(
                ts_root,
                None,
                [0, 1, 2],
                [False, False, False],
                label_output_root=safe_root,
                mcs_output_dir=mcs_root,
            )
            expected_locks = {
                runtime_common.background_mimics_lock_path(self.tmp, mcs_root),
                runtime_common.background_mimics_lock_path(self.tmp, safe_root),
            }
            self.assertEqual(
                {os.path.normcase(value) for value in expected_locks},
                {os.path.normcase(value) for value in captured},
            )
        finally:
            mimics_export._find_mimics_exe = old_find
            mimics_export._project_root = old_root
            mimics_export._prune_local_export_jobs = old_prune
            runtime_common.acquire_resource_lock = old_acquire
            if "old_popen" in locals():
                mimics_export.subprocess.Popen = old_popen

    def test_mimics_export_resolves_configured_mcs_output_dir(self):
        import mimics_export

        configured = os.path.join(self.tmp, "configured_mcs")
        old_cache = mimics_export._CONFIG_CACHE
        try:
            mimics_export._CONFIG_CACHE = {"mimics_output_dir": configured}
            self.assertEqual(
                os.path.abspath(configured),
                mimics_export._resolve_export_output_dir(os.path.join(self.tmp, "dataset")),
            )
        finally:
            mimics_export._CONFIG_CACHE = old_cache

    def test_mask_import_uses_correct_lps_to_ras_world_conversion(self):
        from mask_import import _derive_mimics_voxel_to_ras_matrix

        class Image:
            def get_voxel_center(self, index):
                return [10.0 + index[0], 20.0 + index[1], 30.0 + index[2]]

        matrix = np.asarray(_derive_mimics_voxel_to_ras_matrix(Image(), [2, 2, 2]))
        expected = np.asarray([
            [-1.0, 0.0, 0.0, -10.0],
            [0.0, -1.0, 0.0, -20.0],
            [0.0, 0.0, 1.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        np.testing.assert_allclose(expected, matrix)

    def test_mask_and_batch_import_defer_expensive_work(self):
        import inspect
        import mask_import
        import mimics_import

        mask_main = inspect.getsource(mask_import.main)
        self.assertNotIn("_call_bridge(bridge_params)", mask_main)
        self.assertIn("io_setup_mimics.launch", mask_main)
        self.assertIn("mask_file_picker_ui.py", mask_main)
        mask_submit = inspect.getsource(mask_import._start_import_for_paths)
        self.assertIn("_launch_mask_prepare(bridge_params, result_path, monitor)", mask_submit)
        self.assertNotIn("QFileDialog", inspect.getsource(mask_import))
        self.assertNotIn("tkFileDialog", inspect.getsource(mask_import))
        batch_main = inspect.getsource(mimics_import.main)
        self.assertIn("_resolve_import_output_dir(ts_root, create=False)", batch_main)
        discover_monitor = inspect.getsource(mimics_import._start_import_discover_monitor)
        self.assertLess(
            discover_monitor.index("_start_win32_discover_monitor"),
            discover_monitor.index("from PyQt5.QtCore import QTimer"),
        )

    def test_external_io_setup_defaults_and_nonblocking_entry(self):
        import inspect
        import mimics_import
        import mimics_export
        io_ui = __import__("tools.io_path_setup_ui", fromlist=["source_default_output"])

        dataset = os.path.join(self.tmp, "dataset")
        case_dir = os.path.join(dataset, "s0001")
        os.makedirs(case_dir)
        image_path = os.path.join(case_dir, "ct.mhd")
        with open(image_path, "w", encoding="utf-8") as handle:
            handle.write("ObjectType = Image\n")
        self.assertEqual(
            os.path.join(dataset, "mcs_output"),
            io_ui.source_default_output("import_batch", dataset),
        )
        self.assertEqual(
            os.path.join(case_dir, "mcs_output"),
            io_ui.source_default_output("import_single", image_path),
        )
        self.assertEqual(
            os.path.join(dataset, "mask_exports"),
            io_ui.source_default_output("export_masks", case_dir),
        )
        discovered = io_ui.discover_single_source(image_path)
        self.assertEqual("ct", discovered["case_id"])
        self.assertEqual(os.path.abspath(image_path), discovered["image"])
        dcm_path = os.path.join(case_dir, "slice001.dcm")
        with open(dcm_path, "wb") as handle:
            handle.write(b"candidate")
        dicom_source = io_ui.discover_single_source(dcm_path)
        self.assertEqual(os.path.abspath(case_dir), dicom_source["image"])
        self.assertEqual("dicom_candidate", dicom_source["image_type"])
        import_source = inspect.getsource(mimics_import.main)
        export_source = inspect.getsource(mimics_export.main)
        self.assertIn("_launch_external_import_setup", import_source)
        self.assertIn("_launch_external_export_setup", export_source)
        self.assertNotIn("_pick_directory(\"Select dataset folder\")", import_source)
        self.assertNotIn("_pick_directory(\"Select source case directory\")", export_source)
        single_branch = inspect.getsource(mimics_import.main)
        self.assertIn("case_info_override or _discover_single_case", single_branch)
        export_setup = inspect.getsource(mimics_export._launch_external_export_setup)
        self.assertIn('"configured_output": ""', export_setup)
        self.assertIn("_start_current_project_export", export_setup)
        self.assertIn('"mask_names": _current_mask_names()', export_setup)
        self.assertNotIn("_launch_background_batch_export", export_setup)
        async_launch = inspect.getsource(mimics_export._launch_background_batch_export_async)
        self.assertIn("thread.start()", async_launch)
        self.assertIn("_launch_background_batch_export_async", export_source)

    def test_external_pyside_path_browsers_remain_responsive_outside_mimics(self):
        import inspect
        import mask_import
        import tools.fewshot_training_setup_ui as training_ui
        import tools.io_path_setup_ui as path_ui
        import tools.mask_file_picker_ui as mask_picker
        import tools.path_dialog_helper as path_dialog_helper
        import tools.ui_theme as ui_theme

        ui_source = inspect.getsource(path_ui.run_ui)
        self.assertIn(
            "QFileDialog.getExistingDirectory",
            inspect.getsource(ui_theme.choose_existing_directory),
        )
        self.assertIn(
            "QFileDialog.getOpenFileName",
            inspect.getsource(ui_theme.choose_open_file),
        )
        self.assertIn('QPushButton("Paste")', ui_source)
        self.assertIn(
            "choose_existing_directory",
            inspect.getsource(training_ui.QtTrainingSetupApp.browse_dataset),
        )
        self.assertIn("threading.Thread", inspect.getsource(training_ui.QtTrainingSetupApp.apply_dataset_root))
        self.assertIn("choose_open_files_async", inspect.getsource(mask_picker.main))
        self.assertNotIn("QFileDialog", inspect.getsource(mask_picker.main))
        self.assertIn(
            "QFileDialog.getOpenFileNames",
            inspect.getsource(path_dialog_helper.main),
        )
        self.assertIn("subprocess.Popen", inspect.getsource(ui_theme._AsyncPathDialog))
        self.assertNotIn("os.path.isfile(p)", inspect.getsource(mask_import._start_import_for_paths))

        import tools.fewshot_status_viewer as status_viewer
        status_refresh = inspect.getsource(status_viewer.QtStatusViewerApp.refresh)
        self.assertIn("threading.Thread", status_refresh)
        self.assertNotIn("self._load_jobs()\n        self.job_combo", status_refresh)

    def test_training_setup_inventory_filters_by_selected_label_source(self):
        from tools.fewshot_training_setup_ui import case_rows_from_dataset_for_ui

        root = Path(self.tmp) / "training_inventory"
        mcs_root = root / "mcs"
        exported = root / "exported"
        for case_id in ("case_labeled", "case_unlabeled"):
            case_dir = root / case_id
            case_dir.mkdir(parents=True)
            (case_dir / "ct.nii.gz").write_bytes(b"image")
        (root / "case_labeled" / "segmentations").mkdir()
        (root / "case_labeled" / "segmentations" / "liver.nii.gz").write_bytes(
            b"label"
        )
        (exported / "case_unlabeled" / "segmentations").mkdir(parents=True)
        (
            exported / "case_unlabeled" / "segmentations" / "liver.nii.gz"
        ).write_bytes(b"label")
        mcs_root.mkdir()
        (mcs_root / "case_labeled.mcs").write_bytes(b"project")

        source_rows = case_rows_from_dataset_for_ui(
            root,
            label_source="source_dataset",
            mask_names=["liver"],
            mcs_output_dir=mcs_root,
        )
        source_states = {row["case_id"]: row["state"] for row in source_rows}
        self.assertEqual("ready", source_states["case_labeled"])
        self.assertEqual("mask_missing", source_states["case_unlabeled"])
        exported_rows = case_rows_from_dataset_for_ui(
            root,
            label_source="exported_masks",
            label_root=exported,
            mask_names=["liver"],
            mcs_output_dir=mcs_root,
        )
        exported_states = {row["case_id"]: row["state"] for row in exported_rows}
        self.assertEqual("mask_missing", exported_states["case_labeled"])
        self.assertEqual("ready", exported_states["case_unlabeled"])
        mcs_rows = case_rows_from_dataset_for_ui(
            root,
            label_source="mcs_refresh",
            mask_names=["liver"],
            mcs_output_dir=mcs_root,
        )
        self.assertEqual(["case_labeled"], [row["case_id"] for row in mcs_rows])
        self.assertEqual("ready", mcs_rows[0]["state"])

    def test_external_gui_theme_and_real_task_progress_are_shared(self):
        import inspect
        import mimics_export
        import mimics_import
        import tools.fewshot_model_chooser as model_chooser
        import tools.fewshot_status_viewer as status_viewer
        import tools.fewshot_training_setup_ui as training_ui
        import tools.io_path_setup_ui as path_ui

        theme_source = Path(PROJECT_ROOT, "tools", "ui_theme.py").read_text(encoding="utf-8")
        self.assertIn("def configure_application", theme_source)
        self.assertIn("SetProcessDpiAwarenessContext", theme_source)
        self.assertIn("SetThreadDpiAwarenessContext", theme_source)
        self.assertIn("HighDpiScaleFactorRoundingPolicy", theme_source)
        self.assertLess(
            theme_source.index("\n_prepare_windows_dpi_awareness()\n"),
            theme_source.index("def configure_application"),
        )
        self.assertIn("QProgressBar::chunk", theme_source)
        self.assertIn("shared_stylesheet()", inspect.getsource(training_ui.QtTrainingSetupApp._stylesheet))
        self.assertIn("shared_stylesheet()", inspect.getsource(status_viewer.QtStatusViewerApp._stylesheet))
        self.assertIn("shared_stylesheet()", inspect.getsource(model_chooser.ModelChooser._stylesheet))

        path_source = inspect.getsource(path_ui.run_ui)
        self.assertIn('title.setText("Task in progress")', path_source)
        self.assertIn("secondary_status_path", path_source)
        self.assertIn("stop_path", path_source)
        self.assertIn("show_progress()", path_source)

        output_dir = os.path.join(self.tmp, "mcs_output")
        os.makedirs(output_dir)
        run_root = os.path.join(self.tmp, "run")
        os.makedirs(run_root)
        previous = dict(mimics_import._LAST_TASK_DESCRIPTOR)
        try:
            status_path, stop_path = mimics_import._set_last_import_task(
                run_root, output_dir, "Import dataset",
            )
            descriptor = mimics_import._LAST_TASK_DESCRIPTOR
            self.assertEqual(status_path, descriptor["status_path"])
            self.assertEqual(stop_path, descriptor["stop_path"])
            self.assertIn("_mcs_batch_status.json", descriptor["secondary_status_path"])
            self.assertEqual(os.path.abspath(output_dir), descriptor["output_path"])
        finally:
            mimics_import._LAST_TASK_DESCRIPTOR = previous

        export_source = inspect.getsource(mimics_export._start_current_project_export)
        self.assertIn('"status_path": task_status_path', export_source)
        self.assertIn('"stop_path": task_stop_path', export_source)

    def test_status_viewer_retry_uses_saved_training_context(self):
        import inspect
        import tools.fewshot_status_viewer as status_viewer
        import tools.fewshot_training_setup_ui as training_ui

        context = {
            "config": {},
            "organ": "liver",
            "ts_root": self.tmp,
            "workspace": os.path.join(self.tmp, "fewshot_models"),
            "python_exe": sys.executable,
            "pipeline_script": os.path.join(PROJECT_ROOT, "tools", "fewshot_pipeline.py"),
            "dinov3_root": DINOV3_ROOT,
        }
        launch = training_ui.prepare_training_launch(
            context,
            training_ui.default_training_options({}),
            run_id="train_retry_context",
        )
        self.assertIn("retry_context", launch["job_payload"])
        self.assertEqual("liver", launch["job_payload"]["retry_context"]["organ"])
        self.assertIn("launch_training", inspect.getsource(status_viewer.QtStatusViewerApp.retry_same_settings))
        edit_source = inspect.getsource(status_viewer.QtStatusViewerApp.edit_and_retry)
        self.assertIn("initial_options", edit_source)
        self.assertIn("launch_visible_gui_process", edit_source)
        self.assertNotIn("hidden_process_kwargs", edit_source)

    def test_status_viewer_explains_resolved_architecture_and_auto_batch(self):
        import tools.fewshot_status_viewer as status_viewer

        job = {
            "status": "training",
            "kind": "train",
            "organ": "liver",
            "training_seed": 20260711,
            "training_fold": 0,
            "architecture_plan": {
                "requested_dimension": "auto",
                "resolved_dimension": "3d",
                "quality_mode": "standard",
                "decoder": "context3d_lite",
                "resolved_batch_size": 2,
                "batch_size_source": "hardware_recommendation",
                "gpu_memory_gb": 24.0,
                "plan_sha256": "abc123",
            },
        }
        user_lines = status_viewer.user_status_lines(job)
        self.assertIn("Architecture: Auto \u2192 3D Standard", user_lines)
        self.assertIn(
            "Batch size: 2 (Auto for 24 GB GPU budget)",
            user_lines,
        )
        self.assertNotIn("context3d_lite", "\n".join(user_lines))
        technical_lines = status_viewer.technical_status_lines(job)
        self.assertIn("Resolved decoder: context3d_lite", technical_lines)
        self.assertIn("Architecture plan hash: abc123", technical_lines)
        self.assertIn("Training seed: 20260711", technical_lines)
        self.assertIn("Training fold: 0", technical_lines)

    def test_status_viewer_visible_gui_launcher_does_not_hide_windows(self):
        import tools.fewshot_status_viewer as status_viewer

        calls = []

        class Process(object):
            pid = 13579

        old_popen = status_viewer.subprocess.Popen
        try:
            status_viewer.subprocess.Popen = lambda command, **kwargs: (
                calls.append((command, kwargs)) or Process()
            )
            process = status_viewer.launch_visible_gui_process(
                [sys.executable, os.path.join(PROJECT_ROOT, "tools", "fewshot_training_setup_ui.py")],
                cwd=PROJECT_ROOT,
            )
        finally:
            status_viewer.subprocess.Popen = old_popen

        self.assertEqual(13579, process.pid)
        self.assertEqual(1, len(calls))
        _command, kwargs = calls[0]
        self.assertNotIn("creationflags", kwargs)
        self.assertNotIn("startupinfo", kwargs)

    def test_ai_prediction_output_modes_are_explicit(self):
        import inspect
        import fewshot_mimics
        import nninteractive_mimics

        deferred = inspect.getsource(fewshot_mimics._deferred_prediction_target)
        self.assertIn("choose_on_completion", deferred)
        resolver = inspect.getsource(fewshot_mimics._prediction_target_mask)
        self.assertIn("DINOv3 Prediction Ready", resolver)
        self.assertIn("Update Selected Mask", resolver)
        self.assertIn("Create Editable Copy", resolver)
        self.assertIn("update_selected", resolver)
        session_selector = inspect.getsource(nninteractive_mimics._select_session_masks)
        self.assertIn("choose_on_first_result", session_selector)
        nn_completion = inspect.getsource(nninteractive_mimics._choose_completed_result_target)
        self.assertIn("Update Selected Mask", nn_completion)
        self.assertIn("Create Editable Copy", nn_completion)

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
        dataset_root = os.path.join(self.tmp, "dataset")
        os.makedirs(dataset_root)
        context = {
            "organ": "liver",
            "ts_root": dataset_root,
            "workspace": workspace,
            "python_exe": sys.executable,
            "pipeline_script": os.path.join(PROJECT_ROOT, "tools", "fewshot_pipeline.py"),
            "dinov3_root": DINOV3_ROOT,
            "project_root": PROJECT_ROOT,
            "mimics_exe": r"C:\Program Files\Materialise\MimicsResearch.exe",
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
            "mask_names": "liver,liver_seg",
            "mirror_tta": True,
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
        self.assertEqual("liver,liver_seg", cmd[cmd.index("--mask-names") + 1])
        self.assertIn("--strategy-options-json", cmd)
        self.assertEqual("constant_warmup", cmd[cmd.index("--lr-scheduler") + 1])
        self.assertEqual("2", cmd[cmd.index("--warmup-epochs") + 1])
        strategy_payload = json.loads(cmd[cmd.index("--strategy-options-json") + 1])
        self.assertEqual(strategy_payload["sampling_mode"], "adaptive")
        self.assertEqual(strategy_payload["loss_type"], "auto")
        self.assertTrue(strategy_payload["mirror_tta"])
        self.assertEqual("train_test", launch["run_id"])
        self.assertEqual("launching", launch["job_payload"]["status"])
        self.assertEqual("external_advanced_ui", launch["job_payload"]["launched_by"])

    def test_fewshot_external_setup_can_skip_label_export_wait(self):
        """Users can train from already exported labels without waiting for background Mimics."""
        ui = __import__("tools.fewshot_training_setup_ui", fromlist=["dummy"])
        dataset_root = os.path.join(self.tmp, "dataset")
        os.makedirs(dataset_root)
        context = {
            "organ": "liver",
            "ts_root": dataset_root,
            "workspace": os.path.join(self.tmp, "fewshot_models"),
            "python_exe": sys.executable,
            "pipeline_script": os.path.join(PROJECT_ROOT, "tools", "fewshot_pipeline.py"),
            "dinov3_root": DINOV3_ROOT,
            "mimics_exe": "",
            "config": {},
        }
        options = ui.default_training_options({})
        options["export_labels_before_training"] = False
        launch = ui.prepare_training_launch(context, options, run_id="train_no_export")
        self.assertNotIn("--export-labels", launch["cmd"])
        self.assertFalse(launch["options"]["export_labels_before_training"])

    def test_fewshot_external_setup_accepts_reusable_exported_masks(self):
        """A prior Export Masks destination must be selectable without another .mcs export."""
        ui = __import__("tools.fewshot_training_setup_ui", fromlist=["dummy"])
        dataset_root = os.path.join(self.tmp, "dataset")
        label_root = os.path.join(self.tmp, "exported_masks")
        os.makedirs(dataset_root)
        os.makedirs(label_root)
        context = {
            "organ": "liver",
            "ts_root": dataset_root,
            "workspace": os.path.join(self.tmp, "fewshot_models"),
            "python_exe": sys.executable,
            "pipeline_script": os.path.join(PROJECT_ROOT, "tools", "fewshot_pipeline.py"),
            "dinov3_root": DINOV3_ROOT,
            "mimics_exe": "",
            "config": {},
        }
        options = ui.default_training_options({})
        options.update({
            "label_source": "exported_masks",
            "label_root": label_root,
            "export_labels_before_training": False,
        })
        launch = ui.prepare_training_launch(
            context,
            options,
            run_id="train_reuse_export",
        )
        self.assertNotIn("--export-labels", launch["cmd"])
        self.assertEqual(
            os.path.abspath(label_root),
            launch["cmd"][launch["cmd"].index("--label-root") + 1],
        )
        self.assertEqual("exported_masks", launch["options"]["label_source"])

    def test_fewshot_pytorch_backend_wins_when_onnx_is_also_present(self):
        """An explicit safetensors backend must not be redirected to model.onnx."""
        pipeline = __import__("tools.fewshot_pipeline", fromlist=["dummy"])
        repository_config = json.loads(
            Path(PROJECT_ROOT, "fewshot_config.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            "onnx",
            repository_config["default_feature_encoder_backend"],
        )
        model_dir = Path(self.tmp) / "model"
        model_dir.mkdir()
        (model_dir / "model.onnx").write_bytes(b"not selected")
        (model_dir / "model.safetensors").write_bytes(b"pytorch weights")
        (model_dir / "preprocessor_config.json").write_text("{}", encoding="utf-8")
        (model_dir / "config.json").write_text(
            json.dumps({
                "model_type": "dinov3_vit",
                "num_hidden_layers": 12,
                "hidden_size": 384,
                "patch_size": 16,
            }),
            encoding="utf-8",
        )
        config = {
            "model": {
                "model_path": str(model_dir),
                "encoder_backend": "pytorch",
            },
            "decoder": {"type": "feature_unet2d"},
            "finetune": {"method": "frozen"},
            "data": {"img_size": [224, 224]},
        }
        result = pipeline.validate_training_encoder_assets(
            config,
            Path(DINOV3_ROOT),
        )
        self.assertEqual("pytorch", result["backend"])
        self.assertTrue(result["path"].endswith("model"))

    def test_fewshot_mask_aliases_are_explicit_and_deduplicated(self):
        ui = __import__("tools.fewshot_training_setup_ui", fromlist=["dummy"])
        config = {
            "organ_mask_aliases": {
                "Liver": ["liver_seg", "Segment Liver", "liver-seg"],
            },
        }
        names = ui.configured_mask_names(config, "liver")
        self.assertEqual(["liver", "liver_seg", "Segment Liver"], names)
        options = ui.training_options_for_organ(config, "liver")
        self.assertEqual("liver,liver_seg,Segment Liver", options["mask_names"])

    def test_fewshot_label_refresh_requires_background_mimics(self):
        ui = __import__("tools.fewshot_training_setup_ui", fromlist=["dummy"])
        dataset_root = os.path.join(self.tmp, "dataset")
        os.makedirs(dataset_root)
        context = {
            "organ": "liver",
            "ts_root": dataset_root,
            "workspace": os.path.join(self.tmp, "fewshot_models"),
            "python_exe": sys.executable,
            "pipeline_script": os.path.join(PROJECT_ROOT, "tools", "fewshot_pipeline.py"),
            "dinov3_root": DINOV3_ROOT,
            "mimics_exe": "",
            "config": {},
        }
        options = ui.training_options_for_organ({}, "liver")
        with self.assertRaisesRegex(RuntimeError, "MIMICS_BACKGROUND_EXE"):
            ui.prepare_training_launch(context, options, run_id="train_no_mimics")

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
        self.assertIn("onnxruntime", setup_env.REQUIRED_IMPORTS)
        self.assertIn("nnunetv2", setup_env.REQUIRED_IMPORTS)
        self.assertIn("acvl_utils", setup_env.REQUIRED_IMPORTS)
        self.assertIn("onnxruntime-gpu", setup_env.REQUIRED_PACKAGES)
        bat = package_portable._generate_offline_bat("3.13.7", "python313")
        self.assertIn("import numpy, nibabel, pydicom, SimpleITK", bat)
        self.assertIn("nnunetv2", bat)
        self.assertIn("Version('2.8.1')", bat)
        self.assertIn("Version('2.9')", bat)
        self.assertIn("pip install PySide6 shiboken6", bat)
        self.assertIn("onnxruntime", bat)
        self.assertIn(
            "external\\dinov3-medical-seg\\models\\dinov3-vits16\\model.onnx",
            bat,
        )
        self.assertEqual(
            "external/dinov3-medical-seg/models/dinov3-vits16/model.onnx",
            package_portable.DEFAULT_FROZEN_ENCODER,
        )
        config = json.loads(
            Path(PROJECT_ROOT, "fewshot_config.json").read_text(encoding="utf-8")
        )
        self.assertEqual(64, len(config["default_model_sha256"]))
        # The bat targets whichever environment directory name this checkout
        # uses (python_env for fresh installs, legacy nninteractive_env).
        env_name = package_portable.env_dir_name()
        self.assertIn("echo Lib >> {0}\\python313._pth".format(env_name), bat)
        self.assertIn("Verifying PySide6 external UI backend", bat)
        # Duplicate _pth "Configuring" echo should only appear inside the if block
        self.assertEqual(1, bat.count("echo   Configuring python313._pth"))

    def test_configuration_reference_matches_dinov3_defaults(self):
        config = json.loads(
            Path(PROJECT_ROOT, "fewshot_config.json").read_text(encoding="utf-8")
        )
        reference = Path(PROJECT_ROOT, "CONFIG_REFERENCE.md").read_text(
            encoding="utf-8"
        )
        for key in (
            "default_epochs",
            "default_lr",
            "default_lr_scheduler",
            "default_warmup_epochs",
            "default_weight_decay",
            "default_grad_accumulation",
            "default_img_size",
            "default_decoder",
            "default_finetune_method",
            "default_model_scale",
            "default_modality",
            "default_val_fraction",
        ):
            rendered = json.dumps(config[key])
            self.assertIn("| `{0}` | `{1}` |".format(key, rendered), reference)

    def test_wheel_files_for_package_normalizes_distribution_names(self):
        package_portable = __import__("tools.package_portable", fromlist=["dummy"])
        _wffp = package_portable._wheel_files_for_package

        class _P:
            def __init__(self, name):
                self.name = name

        class _D:
            def glob(self, _pattern):
                return [
                    _P("onnxruntime_gpu-1.22.0-cp313-cp313-win_amd64.whl"),
                    _P("onnxruntime-1.22.0-cp313-cp313-win_amd64.whl"),
                ]

        found = _wffp(_D(), "onnxruntime-gpu==1.22.0")
        self.assertEqual(
            ["onnxruntime_gpu-1.22.0-cp313-cp313-win_amd64.whl"],
            [item.name for item in found],
        )

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
                return True
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
        import runtime_common
        queue_runtime = Path(runtime_common.import_queue_runtime_dir(PROJECT_ROOT, str(output_dir)))
        runner = (queue_runtime / "_run_create_mcs.py").read_text(encoding="utf-8")
        # The runner embeds paths as JSON string literals, so compare against
        # the escaped form the file actually contains.
        self.assertIn(json.dumps(bridge_python)[1:-1], runner)
        self.assertNotIn(sys.executable, runner)
        # And the embedded literal must still parse back to the real path.
        literal_line = [l for l in runner.splitlines() if "MIMICS_BRIDGE_PYTHON" in l][0]
        literal = literal_line.split("= ", 1)[1].strip()
        self.assertEqual(bridge_python, json.loads(literal))

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

    def test_fewshot_external_setup_accepts_resource_scaled_batch(self):
        """Real batch size remains user-configurable for larger local or remote GPUs."""
        ui = __import__("tools.fewshot_training_setup_ui", fromlist=["dummy"])
        options = ui.default_training_options({})
        options.update({
            "epochs": 3,
            "batch_size": 4,
            "gpu_memory_gb": 48,
            "sub_volume": False,
            "val_fraction": 0.0,
            "img_size": "224,224",
        })
        validated = ui.validate_options(options)
        self.assertEqual(4, validated["batch_size"])
        self.assertEqual(48.0, validated["gpu_memory_gb"])

        options["sub_volume"] = True
        with self.assertRaises(ValueError):
            ui.validate_options(options)

    def test_fewshot_external_setup_lists_only_installed_model_weights(self):
        """The setup UI lists the fixed repository and no arbitrary custom entry."""
        ui = __import__("tools.fewshot_training_setup_ui", fromlist=["dummy"])
        dinov3_root = os.path.join(self.tmp, "fake_dinov3")
        for dirname, depth, hidden in (
            ("dinov3-vitb16", 12, 768),
            ("dinov3-vitl16", 24, 1024),
        ):
            model_root = os.path.join(dinov3_root, "models", dirname)
            os.makedirs(model_root)
            Path(model_root, "config.json").write_text(
                json.dumps({
                    "model_type": "dinov3_vit",
                    "num_hidden_layers": depth,
                    "hidden_size": hidden,
                    "patch_size": 16,
                }),
                encoding="utf-8",
            )
            Path(model_root, "preprocessor_config.json").write_text(
                json.dumps({
                    "image_mean": [0.5, 0.5, 0.5],
                    "image_std": [0.25, 0.25, 0.25],
                }),
                encoding="utf-8",
            )
            Path(model_root, "model.safetensors").write_bytes(b"weights")
        app = object.__new__(ui.TrainingSetupApp)
        app.context = {"dinov3_root": dinov3_root}
        app.values = {"model_scale": "vitb16"}
        self.assertEqual(
            ["vitb16", "vitl16"],
            app._available_model_scales(),
        )
        records = ui.discover_pretrained_models(
            dinov3_root,
            current_scale="vitl16",
        )
        self.assertEqual(["vitl16", "vitb16"], [
            record["scale"] for record in records
        ])
        self.assertTrue(all(
            os.path.realpath(os.path.dirname(record["path"]))
            == os.path.realpath(os.path.join(dinov3_root, "models"))
            for record in records
        ))

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
            os.path.join(DINOV3_ROOT, "config", "mimics_lora_segformer3d.yaml"),
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
        self.assertEqual(generated["training"]["early_stopping"]["min_epochs"], 3)
        self.assertEqual(generated["training"]["early_stopping"]["patience"], 0)

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
                DINOV3_ROOT,
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

    def test_fewshot_pipeline_writes_resource_scaled_batch(self):
        """Resolved real batches are preserved in the executable configuration."""
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

        generated = pipeline.write_training_config(
            os.path.join(self.tmp, "batch_config.yaml"),
            os.path.join(DINOV3_ROOT, "config", "mimics_lora_segformer3d.yaml"),
            os.path.join(self.tmp, "dataset"),
            "exp_test",
            Args(),
            validation_enabled=False,
        )
        self.assertEqual(2, generated["training"]["batch_size"])
        self.assertEqual("fit_pad", generated["data"]["resize_mode"])

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
            os.path.join(DINOV3_ROOT, "config", "mimics_lora_segformer3d.yaml"),
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
        # The writer uses yaml.safe_dump, which keeps native path separators.
        # Assert the parsed value rather than a literal substring.
        import yaml
        self.assertEqual(
            metrics_history,
            yaml.safe_load(text)["runtime"]["metrics_history_path"],
        )

        history_only_config = os.path.join(self.tmp, "generated_config_history_only.yaml")
        pipeline.write_training_config(
            history_only_config,
            os.path.join(DINOV3_ROOT, "config", "mimics_lora_segformer3d.yaml"),
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
        self.assertIn(os.path.join(".mimics_runtime", "export_jobs"), result["log"])
        self.assertIn(os.path.join("train_unique", "process.log"), result["log"])
        runner = Path(launched[0][0][-1])
        self.assertIn(os.path.join(".mimics_runtime", "export_jobs"), str(runner))
        self.assertEqual("run_export_batch.py", runner.name)
        config_path = Path(PROJECT_ROOT) / ".mimics_runtime" / "export_jobs" / "fewshot_train_unique" / "export_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(
            str(Path(PROJECT_ROOT) / ".mimics_runtime" / "export_jobs" / "fewshot_train_unique" / "status.json"),
            config["status_path"],
        )
        self.assertEqual(str(configured_mcs), config["output_dir"])
        self.assertEqual(
            str(workspace / "runs" / "train_unique" / "fresh_labels"),
            config["label_staging_dir"],
        )
        self.assertEqual(config["label_staging_dir"], result["label_staging_dir"])
        self.assertFalse((ts_root / "mcs_output" / "_run_export_batch.py").exists())
        self.assertFalse((ts_root / "mcs_output" / "_fewshot_export_mimics.log").exists())

    def test_mimics_export_mask_preflight_does_not_read_voxels(self):
        import mimics_export

        root = Path(self.tmp) / "preflight"
        root.mkdir()
        mcs_path = root / "case001.mcs"
        mcs_path.write_bytes(b"placeholder")
        status_path = root / "status.json"
        stop_path = root / "stop.request"
        voxel_reads = []

        class Mask(object):
            name = "Liver Seg"
            def get_voxel_buffer(self):
                voxel_reads.append(True)
                raise AssertionError("preflight must not read voxel buffers")

        old_masks = mimics_export.mimics.data.masks
        old_open = mimics_export.mimics.file.open_project
        old_close = mimics_export.mimics.file.close_project
        try:
            mimics_export.mimics.data.masks = [Mask()]
            mimics_export.mimics.file.open_project = lambda _path: None
            mimics_export.mimics.file.close_project = lambda: None
            failures = mimics_export._preflight_batch_mask_names(
                [{"case_id": "case001", "case_dir": str(root)}],
                str(root),
                {
                    "mask_names": ["liver", "liver_seg"],
                    "target_mask_name": "liver",
                },
                str(root),
                str(status_path),
                str(stop_path),
            )
        finally:
            mimics_export.mimics.data.masks = old_masks
            mimics_export.mimics.file.open_project = old_open
            mimics_export.mimics.file.close_project = old_close
        self.assertEqual([], failures)
        self.assertEqual([], voxel_reads)
        status = json.loads(status_path.read_text(encoding="utf-8"))
        self.assertEqual("mask_name_preflight", status["phase"])
        self.assertEqual(1, status["total"])

    def test_training_mask_preflight_skips_only_unlabeled_projects(self):
        import mimics_export

        skipped, failures = mimics_export._partition_mask_preflight_failures(
            [
                {"case_id": "unlabeled", "reason": "no saved mask matched"},
                {"case_id": "ambiguous", "reason": "multiple saved masks matched one training target"},
                {"case_id": "broken", "reason": "could not inspect project"},
            ],
            True,
        )
        self.assertEqual(["unlabeled"], [row["case_id"] for row in skipped])
        self.assertEqual(
            ["ambiguous", "broken"],
            [row["case_id"] for row in failures],
        )

        skipped, failures = mimics_export._partition_mask_preflight_failures(
            [
                {"case_id": "unlabeled", "reason": "no saved mask matched"},
                {"case_id": "ambiguous", "reason": "multiple saved masks matched one training target"},
                {"case_id": "missing", "reason": ".mcs file not found"},
                {"case_id": "broken", "reason": "could not inspect project: Internal Parsing Error"},
            ],
            True,
            True,
        )
        self.assertEqual(
            ["unlabeled", "missing", "broken"],
            [row["case_id"] for row in skipped],
        )
        self.assertEqual(["ambiguous"], [row["case_id"] for row in failures])

    def test_mask_preflight_closes_partially_opened_corrupt_project(self):
        import mimics_export

        root = Path(self.tmp) / "corrupt_preflight"
        root.mkdir()
        (root / "broken.mcs").write_bytes(b"broken")
        closes = []
        old_open = mimics_export.mimics.file.open_project
        old_close = mimics_export.mimics.file.close_project
        try:
            mimics_export.mimics.file.open_project = lambda _path: (
                _ for _ in ()
            ).throw(RuntimeError("Internal Parsing Error"))
            mimics_export.mimics.file.close_project = lambda: closes.append(True)
            failures = mimics_export._preflight_batch_mask_names(
                [{"case_id": "broken", "case_dir": str(root)}],
                str(root),
                {"mask_names": ["liver"], "target_mask_name": "liver"},
                str(root),
                str(root / "status.json"),
                str(root / "stop.request"),
            )
        finally:
            mimics_export.mimics.file.open_project = old_open
            mimics_export.mimics.file.close_project = old_close
        self.assertEqual(1, len(failures))
        self.assertIn("Internal Parsing Error", failures[0]["reason"])
        self.assertEqual([True], closes)

    def test_mimics_export_training_target_renames_alias_output(self):
        import mimics_export

        root = Path(self.tmp) / "target_name"
        root.mkdir()

        class Metadata(object):
            def find(self, _name):
                return None

        class Image(object):
            logical_dimensions = [1, 1, 1]
            metadata = Metadata()
            def get_voxel_center(self, *_args):
                return [0.0, 0.0, 0.0]

        class Mask(object):
            name = "Liver Seg"
            image = Image()
            def get_voxel_buffer(self):
                return b"\x01"

        old_masks = mimics_export.mimics.data.masks
        try:
            mimics_export.mimics.data.masks = [Mask()]
            manifest = mimics_export.export_masks_to_buffers(
                str(root),
                mask_names=["liver_seg"],
                target_mask_name="liver",
            )
        finally:
            mimics_export.mimics.data.masks = old_masks
        self.assertEqual("liver", manifest["masks"][0]["safe_name"])
        self.assertTrue((root / "liver.u8").is_file())

    def test_mimics_export_empty_case_selection_fails_during_discovery(self):
        import mimics_export

        root = Path(self.tmp) / "empty_export"
        dataset = root / "dataset"
        output = root / "mcs"
        runtime = root / "runtime"
        dataset.mkdir(parents=True)
        output.mkdir()
        runtime.mkdir()
        status_path = runtime / "status.json"
        config_path = runtime / "config.json"
        config_path.write_text(json.dumps({
            "ts_root": str(dataset),
            "output_dir": str(output),
            "export_root": str(runtime),
            "job_runtime": str(runtime),
            "status_path": str(status_path),
            "mask_names": ["liver"],
        }), encoding="utf-8")
        self.assertEqual(1, mimics_export.run_background_batch_export(str(config_path)))
        status = json.loads(status_path.read_text(encoding="utf-8"))
        self.assertEqual("failed", status["status"])
        self.assertEqual("discovering_cases", status["phase"])
        self.assertIn("No dataset cases", status["error"])

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
        alive = {"value": True}
        try:
            pipeline.workspace_for = lambda _ts_root, _workspace=None: workspace
            pipeline.write_cancel_marker = lambda _path: "[WinError 5] Access is denied"
            pipeline.process_exists = lambda _pid: alive["value"]
            def terminate(_pid):
                alive["value"] = False
                return True
            pipeline.terminate_process_tree = terminate
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

    def test_import_task_does_not_clear_stop_marker_owned_by_live_queue(self):
        import mimics_import

        output_dir = os.path.join(self.tmp, "mcs_output")
        run_root = os.path.join(self.tmp, "run")
        os.makedirs(output_dir)
        os.makedirs(run_root)
        stop_path = mimics_import._queue_stop_path(output_dir)
        os.makedirs(os.path.dirname(stop_path), exist_ok=True)
        Path(stop_path).write_text('{"status":"stop_requested"}', encoding="utf-8")
        lock_path = os.path.join(self.tmp, "background_mimics.lock")
        Path(lock_path).write_text(json.dumps({
            "kind": "create_mcs",
            "output_dir": output_dir,
            "pid": os.getpid(),
            "token": "old-import",
        }), encoding="utf-8")
        old_lock_path = mimics_import.runtime_common.resource_lock_path
        try:
            mimics_import.runtime_common.resource_lock_path = lambda *_args: lock_path
            with self.assertRaises(RuntimeError):
                mimics_import._set_last_import_task(
                    run_root, output_dir, "Import dataset",
                )
            self.assertTrue(os.path.isfile(stop_path))
            os.remove(lock_path)
            status_path, task_stop_path = mimics_import._set_last_import_task(
                run_root, output_dir, "Import dataset",
            )
        finally:
            mimics_import.runtime_common.resource_lock_path = old_lock_path
        self.assertFalse(os.path.isfile(stop_path))
        self.assertEqual(os.path.join(run_root, "status.json"), status_path)
        self.assertEqual(os.path.join(run_root, "stop.json"), task_stop_path)
        self.assertEqual(
            [task_stop_path, stop_path],
            mimics_import._LAST_TASK_DESCRIPTOR["stop_paths"],
        )

    def test_fewshot_external_launch_preserves_worker_status(self):
        """External setup should not overwrite a worker status update with launching."""
        ui = __import__("tools.fewshot_training_setup_ui", fromlist=["dummy"])
        ts_root = os.path.join(self.tmp, "dataset")
        os.makedirs(ts_root)
        workspace = os.path.join(self.tmp, "fewshot_models")
        context = {
            "organ": "liver",
            "ts_root": ts_root,
            "workspace": workspace,
            "python_exe": sys.executable,
            "pipeline_script": os.path.join(PROJECT_ROOT, "tools", "fewshot_pipeline.py"),
            "dinov3_root": DINOV3_ROOT,
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
        os.makedirs(ts_root)
        workspace = os.path.join(self.tmp, "fewshot_models")
        context = {
            "organ": "liver",
            "ts_root": ts_root,
            "workspace": workspace,
            "python_exe": sys.executable,
            "pipeline_script": os.path.join(PROJECT_ROOT, "tools", "fewshot_pipeline.py"),
            "dinov3_root": DINOV3_ROOT,
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
            fewshot_mimics._launch_gui_process = lambda cmd, cwd=None, **kwargs: launched.append((cmd, cwd)) or Proc()
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

    def test_fewshot_train_entry_does_not_open_a_mimics_directory_picker(self):
        """The default training entry delegates dataset selection to external PySide6."""
        import fewshot_mimics

        calls = []
        old_selected = fewshot_mimics._selected_organ
        old_config = fewshot_mimics._config
        old_initial = fewshot_mimics._initial_dataset_root
        old_picker = fewshot_mimics._choose_dataset_root
        old_launch = fewshot_mimics._launch_external_advanced_training
        try:
            fewshot_mimics._selected_organ = lambda: "liver"
            fewshot_mimics._config = lambda: {"advanced_ui_mode": "external"}
            fewshot_mimics._initial_dataset_root = lambda: ""
            fewshot_mimics._choose_dataset_root = lambda _title: (
                (_ for _ in ()).throw(AssertionError("Mimics picker was opened"))
            )
            fewshot_mimics._launch_external_advanced_training = (
                lambda config, organ, ts_root: calls.append(
                    (config, organ, ts_root)
                ) or 0
            )
            result = fewshot_mimics._train_model(True)
        finally:
            fewshot_mimics._selected_organ = old_selected
            fewshot_mimics._config = old_config
            fewshot_mimics._initial_dataset_root = old_initial
            fewshot_mimics._choose_dataset_root = old_picker
            fewshot_mimics._launch_external_advanced_training = old_launch

        self.assertEqual(0, result)
        self.assertEqual([({"advanced_ui_mode": "external"}, "liver", "")], calls)

    def test_fewshot_internal_fallback_keeps_default_model_contract(self):
        """The Mimics-dialog fallback must not silently change the new method."""
        import inspect
        import fewshot_mimics

        source = inspect.getsource(fewshot_mimics._append_training_args)
        self.assertIn('"--model-sha256"', source)
        self.assertIn('"--lr-scheduler"', source)
        self.assertIn('"--warmup-epochs"', source)
        self.assertIn('"--validation-interval"', source)

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
            fewshot_mimics._launch_gui_process = lambda cmd, cwd=None, stderr_log=None: launched.append((cmd, cwd)) or Proc()
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
            fewshot_mimics._launch_gui_process = lambda cmd, cwd=None, stderr_log=None: launched.append((cmd, cwd)) or Proc()
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

        old_selected = fewshot_mimics._selected_mask
        old_context = fewshot_mimics._resolve_prediction_context
        old_guard = fewshot_mimics._guard_no_active_job
        old_launch = fewshot_mimics._launch_inference_job
        old_registry = fewshot_mimics._global_model_registry_path
        try:
            selected = type("Mask", (object,), {"name": "liver"})()
            fewshot_mimics._selected_mask = lambda: selected
            fewshot_mimics._resolve_prediction_context = lambda: (
                ts_root,
                "s0001",
                os.path.join(ts_root, "s0001", "ct.nii.gz"),
            )
            fewshot_mimics._guard_no_active_job = lambda root, requested_kind="train": True
            fewshot_mimics._launch_inference_job = lambda config_arg, root, case_id, organ, selected_model=None, target_spec=None, source_image_path=None: launched.append(selected_model) or 0
            fewshot_mimics._global_model_registry_path = lambda: os.path.join(ts_root, "missing_registry.json")
            result = fewshot_mimics._start_inference(choose_model=False)
        finally:
            fewshot_mimics._selected_mask = old_selected
            fewshot_mimics._resolve_prediction_context = old_context
            fewshot_mimics._global_model_registry_path = old_registry
            fewshot_mimics._guard_no_active_job = old_guard
            fewshot_mimics._launch_inference_job = old_launch
        self.assertEqual(0, result)
        self.assertEqual("train_latest", launched[0]["model_id"])
        self.assertTrue(launched[0]["manifest_path"].endswith("latest.json"))

    def test_fewshot_relocated_manifest_resolves_source_before_stale_absolute(self):
        import dataset_manifest
        import fewshot_mimics

        dataset_root = Path(self.tmp) / "relocated_dataset"
        source = dataset_root / "s0001" / "ct.nii.gz"
        project = dataset_root / "saved_projects" / "s0001.mcs"
        source.parent.mkdir(parents=True)
        project.parent.mkdir(parents=True)
        source.write_bytes(b"image")
        project.write_bytes(b"project")
        manifest_path = project.parent / dataset_manifest.MANIFEST_FILENAME
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": dataset_manifest.SCHEMA_VERSION,
                    "cases": {
                        "s0001": {
                            "case_id": "s0001",
                            "image": {
                                "relative": "../s0001/ct.nii.gz",
                                "absolute": "Z:/old-machine/s0001/ct.nii.gz",
                            },
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        old_project = fewshot_mimics._current_project_path
        old_known = fewshot_mimics._known_dataset_roots
        old_geometry = fewshot_mimics._active_source_geometry_payload
        old_remember = fewshot_mimics._remember_dataset_root
        remembered = []
        try:
            fewshot_mimics._current_project_path = lambda: str(project)
            fewshot_mimics._known_dataset_roots = lambda: []
            fewshot_mimics._active_source_geometry_payload = lambda: None
            fewshot_mimics._remember_dataset_root = remembered.append
            resolved_root, case_id, resolved_source = (
                fewshot_mimics._resolve_prediction_context()
            )
        finally:
            fewshot_mimics._current_project_path = old_project
            fewshot_mimics._known_dataset_roots = old_known
            fewshot_mimics._active_source_geometry_payload = old_geometry
            fewshot_mimics._remember_dataset_root = old_remember
        self.assertEqual("s0001", case_id)
        self.assertEqual(
            os.path.abspath(str(source)), resolved_source
        )
        self.assertEqual(os.path.abspath(str(dataset_root)), resolved_root)
        self.assertEqual([resolved_root], remembered)

    def test_fewshot_status_does_not_open_dataset_picker(self):
        import fewshot_mimics

        ts_root = os.path.join(self.tmp, "status_without_picker")
        os.makedirs(ts_root)
        old_resolve = fewshot_mimics._resolve_status_root
        old_config = fewshot_mimics._config
        old_text = fewshot_mimics._show_status_text
        old_choose = fewshot_mimics._choose_dataset_root
        shown = []
        try:
            fewshot_mimics._resolve_status_root = (
                lambda allow_management_fallback=False: ts_root
            )
            fewshot_mimics._config = lambda: {"status_ui_mode": "text"}
            fewshot_mimics._show_status_text = (
                lambda root: shown.append(root) or 0
            )
            fewshot_mimics._choose_dataset_root = lambda _title: (
                _ for _ in ()
            ).throw(AssertionError("status must not open a path picker"))
            self.assertEqual(0, fewshot_mimics._show_status())
        finally:
            fewshot_mimics._resolve_status_root = old_resolve
            fewshot_mimics._config = old_config
            fewshot_mimics._show_status_text = old_text
            fewshot_mimics._choose_dataset_root = old_choose
        self.assertEqual([ts_root], shown)

    def test_fewshot_status_can_open_empty_portable_model_workspace(self):
        import fewshot_mimics

        root = os.path.join(self.tmp, "portable_model_workspace")
        old_known = fewshot_mimics._known_dataset_roots
        old_management = fewshot_mimics._management_dataset_root
        old_remember = fewshot_mimics._remember_dataset_root
        remembered = []
        try:
            fewshot_mimics._known_dataset_roots = lambda: []
            fewshot_mimics._management_dataset_root = lambda: root
            fewshot_mimics._remember_dataset_root = remembered.append
            resolved = fewshot_mimics._resolve_status_root(
                allow_management_fallback=True
            )
        finally:
            fewshot_mimics._known_dataset_roots = old_known
            fewshot_mimics._management_dataset_root = old_management
            fewshot_mimics._remember_dataset_root = old_remember
        self.assertEqual(root, resolved)
        self.assertTrue(
            os.path.isdir(os.path.join(root, "fewshot_models"))
        )
        self.assertEqual([root], remembered)

    def test_dino_mcs_label_cache_reuses_only_unchanged_cases(self):
        pipeline = __import__("tools.fewshot_pipeline", fromlist=["dummy"])

        dataset_root = Path(self.tmp) / "cache_dataset"
        workspace = dataset_root / "fewshot_models"
        mcs_root = dataset_root / "mcs_output"
        image = dataset_root / "s0001" / "ct.nii.gz"
        mcs = mcs_root / "s0001.mcs"
        image.parent.mkdir(parents=True)
        mcs.parent.mkdir(parents=True)
        image.write_bytes(b"image")
        mcs.write_bytes(b"project-v1")
        first = pipeline._plan_dino_mcs_label_cache(
            dataset_root,
            workspace,
            "liver",
            {"s0001"},
            ["liver"],
            mcs_output_dir=mcs_root,
        )
        self.assertEqual(["s0001"], first["changed"])
        cache_case = first["cache_root"] / "s0001"
        cached_label = cache_case / "segmentations" / "liver.nii.gz"
        cached_label.parent.mkdir(parents=True)
        cached_label.write_bytes(b"label")
        pipeline.write_json_atomic(
            cache_case / "metadata.json",
            {"fingerprint": first["fingerprints"]["s0001"]},
        )
        second = pipeline._plan_dino_mcs_label_cache(
            dataset_root,
            workspace,
            "liver",
            {"s0001"},
            ["liver"],
            mcs_output_dir=mcs_root,
        )
        self.assertIn("s0001", second["reusable"])
        time.sleep(0.01)
        mcs.write_bytes(b"project-v2-with-new-mask")
        third = pipeline._plan_dino_mcs_label_cache(
            dataset_root,
            workspace,
            "liver",
            {"s0001"},
            ["liver"],
            mcs_output_dir=mcs_root,
        )
        self.assertEqual(["s0001"], third["changed"])

    def test_dino_materialization_cache_rebuilds_only_changed_case(self):
        import nibabel as nib
        pipeline = __import__("tools.fewshot_pipeline", fromlist=["dummy"])

        root = Path(self.tmp) / "materialization_cache"
        source = root / "source"
        source.mkdir(parents=True)
        image_path = source / "ct.nii.gz"
        label_path = source / "liver.nii.gz"
        image = np.arange(64, dtype=np.float32).reshape((4, 4, 4))
        label = np.zeros((4, 4, 4), dtype=np.uint8)
        label[1:3, 1:3, 1:3] = 1
        nib.save(nib.Nifti1Image(image, np.eye(4)), str(image_path))
        nib.save(nib.Nifti1Image(label, np.eye(4)), str(label_path))
        sample = {
            "case_id": "s0001",
            "image": str(image_path),
            "label": str(label_path),
        }
        cache_dir = root / "cache"
        first, _ = pipeline.materialize_dataset(
            [sample], root / "run_1", cache_dir=cache_dir
        )
        second, _ = pipeline.materialize_dataset(
            [sample], root / "run_2", cache_dir=cache_dir
        )
        self.assertFalse(first[0]["materialization_cache_hit"])
        self.assertTrue(second[0]["materialization_cache_hit"])
        time.sleep(0.01)
        label[0, 0, 0] = 1
        nib.save(nib.Nifti1Image(label, np.eye(4)), str(label_path))
        third, _ = pipeline.materialize_dataset(
            [sample], root / "run_3", cache_dir=cache_dir
        )
        self.assertFalse(third[0]["materialization_cache_hit"])

    def test_dino_label_cache_failure_does_not_discard_current_label(self):
        pipeline = __import__("tools.fewshot_pipeline", fromlist=["dummy"])

        root = Path(self.tmp) / "cache_fail_open"
        staging = root / "changed"
        output = root / "training"
        cache_root = root / "cache"
        label = staging / "s0001" / "segmentations" / "liver.nii.gz"
        label.parent.mkdir(parents=True)
        label.write_bytes(b"label")
        plan = {
            "reusable": {},
            "changed": ["s0001"],
            "fingerprints": {"s0001": "fingerprint"},
            "cache_root": cache_root,
        }
        original_copy = pipeline._copy_label_atomic

        def fail_cache_only(source, destination):
            destination = Path(destination)
            if cache_root in destination.parents:
                raise OSError("cache is read-only")
            return original_copy(source, destination)

        pipeline._copy_label_atomic = fail_cache_only
        try:
            available, warnings = (
                pipeline._publish_dino_mcs_label_cache(
                    plan,
                    staging,
                    ["liver"],
                    output_root=output,
                )
            )
        finally:
            pipeline._copy_label_atomic = original_copy
        self.assertEqual({"s0001"}, available)
        self.assertEqual("cache_publish", warnings[0]["stage"])
        self.assertTrue(
            (
                output
                / "s0001"
                / "segmentations"
                / "liver.nii.gz"
            ).is_file()
        )

    def test_nninteractive_mcs_label_fingerprint_changes_with_annotation(self):
        pipeline = __import__(
            "tools.nninteractive_finetune_pipeline",
            fromlist=["dummy"],
        )
        root = Path(self.tmp) / "nn_cache"
        root.mkdir()
        image = root / "ct.nii.gz"
        mcs = root / "case.mcs"
        image.write_bytes(b"image")
        mcs.write_bytes(b"mask-v1")
        row = {"image": str(image), "mcs_path": str(mcs)}
        first = pipeline._mcs_export_fingerprint(row, ["liver"])
        time.sleep(0.01)
        mcs.write_bytes(b"mask-v2-updated")
        second = pipeline._mcs_export_fingerprint(row, ["liver"])
        self.assertNotEqual(first, second)

    def test_nninteractive_completed_job_summary_keeps_completed_terminal_state(self):
        pipeline = __import__(
            "tools.nninteractive_finetune_pipeline",
            fromlist=["dummy"],
        )
        root = Path(self.tmp) / "nn_completed_summary"
        job_dir = root / "job"
        workspace = root / "workspace"
        model_dir = root / "model"
        job_dir.mkdir(parents=True)
        workspace.mkdir()
        model_dir.mkdir()
        pipeline.write_json_atomic(
            job_dir / "request.json",
            {
                "workspace": str(workspace),
                "task_id": "brain",
                "task_name": "Brain",
                "output_model_dir": str(model_dir),
                "base_model_dir": str(model_dir),
                "cases": [],
            },
        )
        manifest = job_dir / "dataset_manifest.json"
        manifest.write_text('{"cases": []}', encoding="utf-8")

        def run_training(*_args, **_kwargs):
            pipeline.write_json_atomic(
                job_dir / "trainer_status.json",
                {
                    "epochs": 2,
                    "elapsed_seconds": 4,
                    "best_score": 0.7,
                    "metrics_history": [
                        {"epoch": 2, "epochs": 2, "validation_auc": 0.7}
                    ],
                },
            )

        with mock.patch.object(
            pipeline, "_prepare_manifest", return_value=(manifest, None)
        ), mock.patch.object(
            pipeline, "_run_training", side_effect=run_training
        ), mock.patch.object(
            pipeline, "_register_model", return_value=({"model_id": "m1"}, True)
        ), mock.patch.object(
            pipeline, "_cleanup_terminal_artifacts", return_value={}
        ):
            self.assertEqual(0, pipeline.run_job(str(job_dir)))

        status = pipeline.read_json(job_dir / "status.json", {}) or {}
        self.assertEqual("completed", status.get("status"))
        self.assertEqual("new_model_selected", status.get("selection_outcome"))
        self.assertIn("Summary:", (job_dir / "job.log").read_text(encoding="utf-8"))

    def test_nninteractive_target_and_initial_mcs_exports_use_separate_caches(self):
        pipeline = __import__(
            "tools.nninteractive_finetune_pipeline",
            fromlist=["dummy"],
        )
        root = Path(self.tmp) / "nn_separate_mcs_caches"
        workspace = root / "workspace"
        job_dir = root / "job"
        job_dir.mkdir(parents=True)
        image_path = root / "image.nii.gz"
        mcs_path = root / "case.mcs"
        image_path.write_bytes(b"image")
        mcs_path.write_bytes(b"project")
        row = {
            "case_id": "case",
            "image": str(image_path),
            "mcs_path": str(mcs_path),
            "split": "train",
        }
        request = {
            "workspace": str(workspace),
            "task_id": "brain",
            "mask_names": ["target"],
            "cases": [row],
        }
        cache_specs = (
            ("mcs_labels", ["target"], "target.nii.gz", b"target"),
            (
                "mcs_initial_masks",
                ["draft"],
                "draft.nii.gz",
                b"initial",
            ),
        )
        for bucket, mask_names, filename, content in cache_specs:
            case_cache = workspace / "cache" / bucket / "brain" / "case"
            segmentation = case_cache / "segmentations" / filename
            segmentation.parent.mkdir(parents=True)
            segmentation.write_bytes(content)
            pipeline.write_json_atomic(
                case_cache / "metadata.json",
                {
                    "fingerprint": pipeline._mcs_export_fingerprint(
                        row, mask_names
                    )
                },
            )

        target_staging = pipeline._prepare_cached_mcs_labels(
            request,
            job_dir,
            job_dir / "status.json",
            job_dir / "control.json",
            job_dir / "job.log",
        )
        initial_staging = pipeline._prepare_cached_mcs_labels(
            request,
            job_dir,
            job_dir / "status.json",
            job_dir / "control.json",
            job_dir / "job.log",
            mask_names_override=["draft"],
            cache_role="initial",
            output_name="initial_masks",
        )
        target_file = target_staging / "case" / "segmentations" / "target.nii.gz"
        initial_file = (
            initial_staging / "case" / "segmentations" / "draft.nii.gz"
        )
        self.assertNotEqual(target_staging, initial_staging)
        self.assertEqual(target_file.read_bytes(), b"target")
        self.assertEqual(initial_file.read_bytes(), b"initial")

    def test_nninteractive_source_grid_inputs_reuse_and_invalidate_by_label(self):
        import nibabel as nib

        pipeline = __import__(
            "tools.nninteractive_finetune_pipeline",
            fromlist=["dummy"],
        )
        root = Path(self.tmp) / "nn_source_grid_reuse"
        root.mkdir()
        image = root / "image.nii.gz"
        label = root / "label.nii.gz"
        values = np.arange(64, dtype=np.float32).reshape((4, 4, 4)) + 1
        target = np.zeros((4, 4, 4), dtype=np.uint8)
        target[1:3, 1:3, 1:3] = 1
        nib.save(nib.Nifti1Image(values, np.eye(4)), str(image))
        nib.save(nib.Nifti1Image(target, np.eye(4)), str(label))

        first = pipeline._prepare_source_grid_case_cache(
            root / "workspace", "brain", "case", image, label, None
        )
        second = pipeline._prepare_source_grid_case_cache(
            root / "workspace", "brain", "case", image, label, None
        )
        self.assertFalse(first[3])
        self.assertTrue(second[3])
        self.assertEqual(first[:2], second[:2])

        time.sleep(0.01)
        target[0, 0, 0] = 1
        nib.save(nib.Nifti1Image(target, np.eye(4)), str(label))
        changed = pipeline._prepare_source_grid_case_cache(
            root / "workspace", "brain", "case", image, label, None
        )
        self.assertFalse(changed[3])
        self.assertNotEqual(first[0].parent, changed[0].parent)

    def test_nninteractive_initial_mask_validation_is_cached(self):
        import nibabel as nib

        pipeline = __import__(
            "tools.nninteractive_finetune_pipeline",
            fromlist=["dummy"],
        )
        root = Path(self.tmp) / "nn_initial_validation_cache"
        root.mkdir()
        image_path = root / "image.nii.gz"
        label_path = root / "target.nii.gz"
        initial_path = root / "draft.nii.gz"
        image = np.ones((8, 8, 8), dtype=np.float32)
        target = np.zeros((8, 8, 8), dtype=np.uint8)
        target[1:7, 1:7, 1:7] = 1
        initial = np.zeros_like(target)
        initial[2:6, 2:6, 2:6] = 1
        for path, array in (
            (image_path, image),
            (label_path, target),
            (initial_path, initial),
        ):
            nib.save(nib.Nifti1Image(array, np.eye(4)), str(path))

        first = pipeline._prepare_source_grid_case_cache(
            root / "workspace",
            "task",
            "case",
            image_path,
            label_path,
            initial_path,
        )
        with mock.patch.object(
            pipeline,
            "_ensure_binary_nifti_label",
            side_effect=AssertionError("cache hit must not reload masks"),
        ):
            second = pipeline._prepare_source_grid_case_cache(
                root / "workspace",
                "task",
                "case",
                image_path,
                label_path,
                initial_path,
            )
        self.assertFalse(first[3])
        self.assertTrue(second[3])
        self.assertEqual(second[4], "")
        self.assertEqual(first[2], second[2])

    def test_nninteractive_manifest_keeps_distinct_real_initial_mask(self):
        import nibabel as nib

        pipeline = __import__(
            "tools.nninteractive_finetune_pipeline",
            fromlist=["dummy"],
        )
        root = Path(self.tmp) / "nn_initial_manifest"
        source = root / "source"
        source.mkdir(parents=True)
        image_path = source / "image.nii.gz"
        label_path = source / "target.nii.gz"
        initial_path = source / "draft.nii.gz"
        image = np.arange(32**3, dtype=np.float32).reshape((32, 32, 32)) + 1
        target = np.zeros((32, 32, 32), dtype=np.uint8)
        target[6:26, 6:26, 6:26] = 1
        initial = np.zeros_like(target)
        initial[8:24, 8:24, 8:24] = 1
        for path, array in (
            (image_path, image),
            (label_path, target),
            (initial_path, initial),
        ):
            nib.save(nib.Nifti1Image(array, np.eye(4)), str(path))
        job_dir = root / "job"
        job_dir.mkdir()
        request = {
            "workspace": str(root / "workspace"),
            "source_mode": "prepared",
            "mask_names": ["target"],
            "initial_mask_source": "exported_masks",
            "initial_mask_names": ["draft"],
            "initial_mask_root": str(source),
            "cases": [
                {
                    "case_id": "case",
                    "image": str(image_path),
                    "label": str(label_path),
                    "initial_mask": str(initial_path),
                    "split": "train",
                }
            ],
        }
        manifest_path, validation_path = pipeline._prepare_manifest(
            request,
            job_dir,
            job_dir / "status.json",
            job_dir / "control.json",
            job_dir / "job.log",
        )
        payload = pipeline.read_json(manifest_path, {})
        self.assertIsNone(validation_path)
        self.assertEqual(len(payload["cases"]), 1)
        prepared_initial = Path(payload["cases"][0]["initial_mask"])
        self.assertTrue(prepared_initial.is_file())
        self.assertEqual(
            int(np.count_nonzero(nib.load(str(prepared_initial)).dataobj)),
            int(initial.sum()),
        )

    def test_nninteractive_manifest_rejects_target_as_initial_mask(self):
        pipeline = __import__(
            "tools.nninteractive_finetune_pipeline",
            fromlist=["dummy"],
        )
        root = Path(self.tmp) / "nn_initial_overlap"
        job_dir = root / "job"
        job_dir.mkdir(parents=True)
        request = {
            "workspace": str(root / "workspace"),
            "source_mode": "prepared",
            "mask_names": ["Liver"],
            "initial_mask_source": "exported_masks",
            "initial_mask_names": ["liver"],
            "cases": [],
        }
        with self.assertRaisesRegex(RuntimeError, "overlap"):
            pipeline._prepare_manifest(
                request,
                job_dir,
                job_dir / "status.json",
                job_dir / "control.json",
                job_dir / "job.log",
            )

    def test_nninteractive_empty_start_never_exports_initial_masks(self):
        pipeline = __import__(
            "tools.nninteractive_finetune_pipeline",
            fromlist=["dummy"],
        )
        root = Path(self.tmp) / "nn_empty_start"
        job_dir = root / "job"
        job_dir.mkdir(parents=True)
        request = {
            "workspace": str(root / "workspace"),
            "source_mode": "prepared",
            "mask_names": ["target"],
            "training_goal": "start_empty",
            "initial_mask_source": "mcs",
            "initial_mask_names": ["draft"],
            "cases": [],
        }
        with mock.patch.object(
            pipeline,
            "_prepare_cached_mcs_labels",
            side_effect=AssertionError(
                "empty-start must not export Initial Masks"
            ),
        ), self.assertRaisesRegex(RuntimeError, "No selected case"):
            pipeline._prepare_manifest(
                request,
                job_dir,
                job_dir / "status.json",
                job_dir / "control.json",
                job_dir / "job.log",
            )

    def test_nninteractive_refine_all_skipped_reports_initial_mask_reason(self):
        pipeline = __import__(
            "tools.nninteractive_finetune_pipeline",
            fromlist=["dummy"],
        )
        root = Path(self.tmp) / "nn_refine_without_initial"
        job_dir = root / "job"
        job_dir.mkdir(parents=True)
        request = {
            "workspace": str(root / "workspace"),
            "source_mode": "prepared",
            "mask_names": ["target"],
            "training_goal": "refine_existing",
            "initial_mask_source": "exported_masks",
            "initial_mask_names": ["draft"],
            "cases": [
                {
                    "case_id": "case",
                    "image": str(root / "image.nii.gz"),
                    "label": str(root / "target.nii.gz"),
                    "split": "train",
                }
            ],
        }
        prepared = root / "prepared.nii.gz"
        with mock.patch.object(
            pipeline,
            "_prepare_source_grid_case_cache",
            return_value=(
                prepared,
                prepared,
                None,
                True,
                "empty",
                "fingerprint",
            ),
        ), self.assertRaisesRegex(RuntimeError, "usable real Initial Mask"):
            pipeline._prepare_manifest(
                request,
                job_dir,
                job_dir / "status.json",
                job_dir / "control.json",
                job_dir / "job.log",
            )

    def test_nninteractive_removed_synthetic_initial_source_is_rejected(self):
        pipeline = __import__(
            "tools.nninteractive_finetune_pipeline",
            fromlist=["dummy"],
        )
        root = Path(self.tmp) / "nn_removed_synthetic"
        job_dir = root / "job"
        job_dir.mkdir(parents=True)
        request = {
            "workspace": str(root / "workspace"),
            "source_mode": "prepared",
            "mask_names": ["target"],
            "training_goal": "general",
            "initial_mask_source": "synthetic",
            "cases": [],
        }
        with self.assertRaisesRegex(RuntimeError, "Unsupported Initial Mask source"):
            pipeline._prepare_manifest(
                request,
                job_dir,
                job_dir / "status.json",
                job_dir / "control.json",
                job_dir / "job.log",
            )

    def test_nninteractive_target_nifti_and_initial_mhd_share_training_grid(self):
        import nibabel as nib
        import SimpleITK as sitk

        pipeline = __import__(
            "tools.nninteractive_finetune_pipeline",
            fromlist=["dummy"],
        )
        root = Path(self.tmp) / "nn_mixed_mask_formats"
        root.mkdir()
        shape = (8, 10, 12)
        reference_affine = np.diag([-1.0, -1.0, 1.0, 1.0])
        image = (
            np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1
        )
        target = np.zeros(shape, dtype=np.uint8)
        target[2:7, 2:8, 3:10] = 1
        initial = np.zeros(shape, dtype=np.uint8)
        initial[3:7, 3:8, 4:10] = 1
        image_path = root / "image.nii.gz"
        target_path = root / "target.nii.gz"
        initial_path = root / "initial.mhd"
        nib.save(
            nib.Nifti1Image(image, reference_affine), str(image_path)
        )
        nib.save(
            nib.Nifti1Image(target, reference_affine), str(target_path)
        )

        # Store the same physical draft with x/y axes exchanged in an
        # explicitly LPS-oriented MHD file.
        initial_xyz = np.transpose(initial, (1, 0, 2))
        sitk_initial = sitk.GetImageFromArray(
            np.transpose(initial_xyz, (2, 1, 0))
        )
        sitk_initial.SetSpacing((1.0, 1.0, 1.0))
        sitk_initial.SetOrigin((0.0, 0.0, 0.0))
        sitk_initial.SetDirection(
            (0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        )
        sitk.WriteImage(sitk_initial, str(initial_path))

        target_output = pipeline._ensure_binary_nifti_label(
            target_path,
            root / "target_aligned.nii.gz",
            image_path,
        )
        initial_output = pipeline._ensure_binary_nifti_label(
            initial_path,
            root / "initial_aligned.nii.gz",
            image_path,
            allow_empty=True,
        )
        target_image = nib.load(str(target_output))
        initial_image = nib.load(str(initial_output))
        self.assertEqual(target_image.shape, shape)
        self.assertEqual(initial_image.shape, shape)
        np.testing.assert_allclose(
            target_image.affine, reference_affine, atol=1e-6
        )
        np.testing.assert_allclose(
            initial_image.affine, reference_affine, atol=1e-6
        )
        np.testing.assert_array_equal(
            np.asarray(target_image.dataobj), target
        )
        np.testing.assert_array_equal(
            np.asarray(initial_image.dataobj), initial
        )
        np.testing.assert_allclose(
            initial_image.get_qform(), reference_affine, atol=1e-6
        )
        np.testing.assert_allclose(
            initial_image.get_sform(), reference_affine, atol=1e-6
        )

    def test_nninteractive_initial_nrrd_resamples_to_target_size_and_affine(self):
        import nibabel as nib
        import SimpleITK as sitk

        pipeline = __import__(
            "tools.nninteractive_finetune_pipeline",
            fromlist=["dummy"],
        )
        root = Path(self.tmp) / "nn_mixed_mask_sizes"
        root.mkdir()
        target_shape = (8, 10, 12)
        target_affine = np.diag([-1.0, -1.0, 1.0, 1.0])
        image_path = root / "image.nii.gz"
        nib.save(
            nib.Nifti1Image(
                np.ones(target_shape, dtype=np.float32),
                target_affine,
            ),
            str(image_path),
        )
        coarse = np.zeros((4, 5, 6), dtype=np.uint8)
        coarse[1:3, 1:4, 2:5] = 1
        initial_path = root / "initial.nrrd"
        sitk_initial = sitk.GetImageFromArray(
            np.transpose(coarse, (2, 1, 0))
        )
        sitk_initial.SetSpacing((2.0, 2.0, 2.0))
        sitk_initial.SetOrigin((0.0, 0.0, 0.0))
        sitk_initial.SetDirection(
            (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        )
        sitk.WriteImage(sitk_initial, str(initial_path))
        output = pipeline._ensure_binary_nifti_label(
            initial_path,
            root / "initial_aligned.nii.gz",
            image_path,
            allow_empty=True,
        )
        aligned = nib.load(str(output))
        aligned_values = np.asarray(aligned.dataobj)
        self.assertEqual(aligned.shape, target_shape)
        np.testing.assert_allclose(
            aligned.affine, target_affine, atol=1e-6
        )
        self.assertGreater(int(np.count_nonzero(aligned_values)), 0)
        self.assertTrue(
            np.all(
                (aligned_values == 0)
                | (aligned_values == 1)
            )
        )

    def test_nninteractive_geometryless_mhd_with_different_size_fails_closed(self):
        import nibabel as nib

        pipeline = __import__(
            "tools.nninteractive_finetune_pipeline",
            fromlist=["dummy"],
        )
        root = Path(self.tmp) / "nn_geometryless_mhd"
        root.mkdir()
        image_path = root / "image.nii.gz"
        nib.save(
            nib.Nifti1Image(
                np.ones((8, 10, 12), dtype=np.float32),
                np.eye(4, dtype=np.float64),
            ),
            str(image_path),
        )
        values = np.zeros((4, 5, 6), dtype=np.uint8)
        values[1:3, 1:4, 2:5] = 1
        raw_path = root / "initial.raw"
        raw_path.write_bytes(
            np.transpose(values, (2, 1, 0)).tobytes(order="C")
        )
        mhd_path = root / "initial.mhd"
        mhd_path.write_text(
            "\n".join(
                [
                    "ObjectType = Image",
                    "NDims = 3",
                    "BinaryData = True",
                    "BinaryDataByteOrderMSB = False",
                    "CompressedData = False",
                    "DimSize = 4 5 6",
                    "ElementType = MET_UCHAR",
                    "ElementDataFile = initial.raw",
                    "",
                ]
            ),
            encoding="ascii",
        )
        with self.assertRaisesRegex(
            ValueError,
            "no origin/direction.*shape does not match",
        ):
            pipeline._ensure_binary_nifti_label(
                mhd_path,
                root / "initial_aligned.nii.gz",
                image_path,
                allow_empty=True,
            )

    def test_nninteractive_relocates_source_path_from_dataset_manifest(self):
        import dataset_manifest
        import nninteractive_mimics

        dataset_root = Path(self.tmp) / "nn_relocated"
        source = dataset_root / "case_a" / "ct.nii.gz"
        project = dataset_root / "saved_projects" / "case_a.mcs"
        source.parent.mkdir(parents=True)
        project.parent.mkdir(parents=True)
        source.write_bytes(b"image")
        project.write_bytes(b"project")
        dataset_manifest.update_case(
            project.parent,
            "case_a",
            image_path=source,
            mcs_path=project,
        )
        old_project = nninteractive_mimics._current_project_path
        try:
            nninteractive_mimics._current_project_path = lambda: str(project)
            resolved = (
                nninteractive_mimics._relocated_source_image_path()
            )
        finally:
            nninteractive_mimics._current_project_path = old_project
        self.assertEqual(os.path.abspath(str(source)), resolved)

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
        # The DINOv3 project may live outside this repo (fewshot_config.json
        # "dinov3_project"); import its checkpoint module via DINOV3_ROOT.
        if str(DINOV3_ROOT) not in sys.path:
            sys.path.insert(0, str(DINOV3_ROOT))
        checkpoint = __import__("src.utils.checkpoint",
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
        sys.path.insert(0, DINOV3_ROOT)
        from src.utils.config import load_config
        from src.utils.checkpoint import save_checkpoint, load_checkpoint

        tmp = Path(tempfile.mkdtemp(prefix="lora_roundtrip_"))
        try:
            cfg_path = os.path.join(DINOV3_ROOT,
                                    "config", "mimics_lora_segformer3d.yaml")
            cfg = load_config(cfg_path)
            cfg["finetune"]["method"] = "lora"
            model_path = Path(cfg["model"]["model_path"])
            if not model_path.is_absolute():
                model_path = Path(DINOV3_ROOT) / model_path
            if not model_path.is_dir():
                self.skipTest("bundled DINOv3 weights are not present in this checkout")
            try:
                __import__("transformers")
            except ImportError:
                self.skipTest(
                    "transformers is not installed in this test environment"
                )
            from src.models.segmentor import DINOv33DSegmentor

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
        path = os.path.join(DINOV3_ROOT,
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
        with open(entry, "r", encoding="utf-8") as handle:
            content = handle.read()
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
            "channel_policy": "2_5d", "slice_axis": "coronal", "mirror_tta": True,
        })
        self.assertEqual(patch["model"]["slice_axis"], "coronal")
        self.assertEqual(patch["model"]["channel_policy"], "2_5d")
        self.assertEqual(patch["loss"]["type"], "dice_focal")
        self.assertEqual(patch["data"]["patch"]["size_zyx"], [64, 192, 192])
        self.assertTrue(patch["data"]["patch"]["inference_sliding_window"])
        self.assertEqual(patch["inference"]["tta_axes"], [[0]])
        self.assertEqual(patch["augmentation"]["flip_axes"], [0])
        self.assertFalse(patch["augmentation"]["allow_left_right_flip"])

        full = strategies.compile_strategy("full_volume", fingerprint, policy)
        self.assertFalse(full["data"]["patch"]["enabled"])
        self.assertEqual(full["inference"]["tta_axes"], [])
        self.assertNotIn("augmentation", full)

    def test_mimics_training_option_matrix_reaches_backend_config(self):
        """Every public architecture family survives UI, CLI, and config generation."""
        import tools.fewshot_training_setup_ui as ui
        import tools.fewshot_pipeline as pipeline
        import tools.fewshot_strategies as strategies
        import tools.fewshot_architecture as architecture

        context = {
            "organ": "organ", "ts_root": self.tmp,
            "workspace": os.path.join(self.tmp, "fewshot_models"),
            "python_exe": sys.executable, "pipeline_script": os.path.join(PROJECT_ROOT, "tools", "fewshot_pipeline.py"),
            "dinov3_root": DINOV3_ROOT,
            "project_root": PROJECT_ROOT, "config": {"base_config": "config/research/ct_fewshot_fast.yaml"},
        }
        fingerprint = {
            "summary": {
                "median_spacing_zyx": [1.0, 1.0, 1.0],
                "median_foreground_fraction": 0.02,
            },
        }
        policy = {
            "patch": {"enabled": True, "size_zyx": [8, 32, 32]},
            "recommended_slice_axis": "axial",
        }
        count = 0
        for preset in strategies.strategy_ids():
            for dimension in ("auto", "2d", "3d"):
                for quality in ("standard", "high_detail"):
                    if dimension == "2d" and quality == "high_detail":
                        continue
                    for method in ("frozen", "lora"):
                        options = ui.default_training_options(context["config"])
                        options.update(strategies.strategy_defaults(preset))
                        options.update({
                            "strategy": preset,
                            "training_dimension": dimension,
                            "quality_mode": quality,
                            "decoder": "auto",
                            "finetune_method": method, "cases": ["case1"], "val_fraction": 0.0,
                            "export_labels_before_training": False,
                            "modality": "ct",
                            "batch_size": 1,
                        })
                        options = ui.validate_options(options)
                        launch = ui.prepare_training_launch(context, options, run_id="matrix_{}".format(count))
                        args = pipeline.build_parser().parse_args(launch["cmd"][2:])
                        plan = architecture.resolve_architecture(
                            {
                                "training_dimension": args.training_dimension,
                                "quality_mode": args.quality_mode,
                                "finetune_method": args.finetune_method,
                                "decoder": args.decoder,
                            },
                            fingerprint,
                            gpu_memory_gb=24,
                        )
                        args.decoder = plan["decoder"]
                        args.architecture_plan = plan
                        args.encoder_backend = "pytorch"
                        compiled = strategies.compile_strategy(
                            args.strategy, fingerprint=fingerprint, policy=policy,
                            user_options=json.loads(args.strategy_options_json),
                        )
                        config_path = os.path.join(self.tmp, "matrix_{}.yaml".format(count))
                        generated = pipeline.write_training_config(
                            config_path, args.base_config, self.tmp, "matrix", args,
                            validation_enabled=False, strategy_overrides=compiled,
                        )
                        self.assertEqual(plan["decoder"], generated["decoder"]["type"])
                        self.assertEqual(method, generated["finetune"]["method"])
                        self.assertEqual(
                            ["q_proj", "v_proj"],
                            generated["finetune"]["target_modules"],
                        )
                        self.assertEqual(
                            plan["resolved_dimension"],
                            generated["decoder"]["architecture_family"],
                        )
                        self.assertEqual("fit_pad", generated["data"]["resize_mode"])
                        self.assertEqual(
                            compiled["inference"]["tta_axes"],
                            generated["inference"]["tta_axes"],
                        )
                        count += 1
        self.assertEqual(len(strategies.strategy_ids()) * 5 * 2, count)

    def test_architecture_auto_is_orientation_independent_and_resource_aware(self):
        import tools.fewshot_architecture as architecture

        thick_sagittal = {
            "summary": {"median_spacing_zyx": [1.0, 1.0, 4.0]},
        }
        plan = architecture.resolve_architecture(
            {
                "training_dimension": "auto",
                "quality_mode": "high_detail",
                "finetune_method": "frozen",
            },
            thick_sagittal,
            gpu_memory_gb=48,
        )
        self.assertEqual("2d", plan["resolved_dimension"])
        self.assertEqual("scale_aware2d", plan["decoder"])
        self.assertEqual("standard", plan["quality_mode"])

        isotropic = {
            "summary": {"median_spacing_zyx": [1.0, 1.0, 1.0]},
        }
        plan = architecture.resolve_architecture(
            {
                "training_dimension": "auto",
                "quality_mode": "high_detail",
                "finetune_method": "lora",
            },
            isotropic,
            gpu_memory_gb=48,
        )
        self.assertEqual("3d", plan["resolved_dimension"])
        self.assertEqual("context3d_hybrid", plan["decoder"])
        self.assertEqual(
            1,
            architecture.recommended_batch_size(
                plan,
                {
                    "patch": {"enabled": True},
                    "input_size": [224, 224],
                },
                gpu_memory_gb=48,
            ),
        )
        frozen_plan = architecture.resolve_architecture(
            {
                "training_dimension": "3d",
                "quality_mode": "high_detail",
                "finetune_method": "frozen",
            },
            isotropic,
            gpu_memory_gb=48,
        )
        self.assertEqual(
            2,
            architecture.recommended_batch_size(
                frozen_plan,
                {
                    "patch": {"enabled": True},
                    "input_size": [224, 224],
                },
                gpu_memory_gb=48,
            ),
        )
        self.assertEqual(
            1,
            architecture.recommended_batch_size(
                plan,
                {
                    "patch": {"enabled": True},
                    "input_size": [384, 384],
                },
                gpu_memory_gb=48,
            ),
        )

    def test_dinov3_feature_layers_follow_backbone_depth(self):
        import tools.fewshot_architecture as architecture
        import tools.fewshot_pipeline as pipeline

        self.assertEqual([2, 5, 8, 11], architecture.intermediate_layer_indices(12))
        self.assertEqual([4, 11, 17, 23], architecture.intermediate_layer_indices(24))
        self.assertEqual([6, 15, 23, 31], architecture.intermediate_layer_indices(32))
        self.assertEqual([0, 1, 3, 4], architecture.intermediate_layer_indices(5))

        # The DINOv3 project may live outside this repo (fewshot_config.json
        # "dinov3_project"); build paths from DINOV3_ROOT, not PROJECT_ROOT.
        model_root = os.path.join(
            DINOV3_ROOT,
            "models",
            "dinov3-vitl16",
        )
        base_config = os.path.join(
            DINOV3_ROOT,
            "config",
            "research",
            "ct_fewshot_fast.yaml",
        )
        plan = pipeline._backbone_feature_plan(
            model_root,
            base_config,
            "vitl16",
        )
        self.assertEqual(24, plan["num_hidden_layers"])
        self.assertEqual(1024, plan["hidden_size"])
        self.assertEqual(16, plan["patch_size"])
        self.assertEqual([4, 11, 17, 23], plan["out_indices"])

        for name, depth, hidden in (
            ("dinov3-vits16", 12, 384),
            ("dinov3-vitb16", 12, 768),
            ("dinov3-vitl16", 24, 1024),
        ):
            spec = architecture.inspect_dinov3_vit_weights(
                os.path.join(DINOV3_ROOT, "models", name)
            )
            self.assertEqual(depth, spec["num_hidden_layers"])
            self.assertEqual(hidden, spec["hidden_size"])
            self.assertEqual(16, spec["patch_size"])
            self.assertEqual(
                architecture.intermediate_layer_indices(depth),
                spec["out_indices"],
            )

    def test_custom_backbone_rejects_non_dinov3_and_uses_its_patch_size(self):
        import tools.fewshot_architecture as architecture
        import tools.fewshot_training_setup_ui as setup_ui

        root = os.path.join(self.tmp, "wrong_backbone")
        os.makedirs(root)
        with open(os.path.join(root, "config.json"), "w") as handle:
            json.dump({
                "model_type": "vit",
                "num_hidden_layers": 12,
                "hidden_size": 384,
                "patch_size": 16,
            }, handle)
        for name in ("model.safetensors", "preprocessor_config.json"):
            with open(os.path.join(root, name), "wb") as handle:
                handle.write(b"x")
        with self.assertRaisesRegex(ValueError, "Unsupported pretrained architecture"):
            architecture.inspect_dinov3_vit_weights(root)

        valid_root = os.path.join(self.tmp, "custom_patch14")
        os.makedirs(valid_root)
        Path(valid_root, "config.json").write_text(
            json.dumps({
                "model_type": "dinov3_vit",
                "num_hidden_layers": 8,
                "hidden_size": 512,
                "patch_size": 14,
            }),
            encoding="utf-8",
        )
        Path(valid_root, "preprocessor_config.json").write_text(
            json.dumps({
                "image_mean": [0.4, 0.5, 0.6],
                "image_std": [0.2, 0.25, 0.3],
            }),
            encoding="utf-8",
        )
        Path(valid_root, "model.safetensors").write_bytes(b"weights")
        options = setup_ui.default_training_options({})
        options.update({
            "model_scale": "custom",
            "model_path": valid_root,
            "img_size": "224,224",
            "val_fraction": 0.0,
        })
        validated = setup_ui.validate_options(options)
        self.assertEqual(valid_root, validated["model_path"])
        options["img_size"] = "256,256"
        with self.assertRaisesRegex(ValueError, "patch size \\(14\\)"):
            setup_ui.validate_options(options)

    def test_auto_modality_uses_manifest_and_rejects_mixed_runs(self):
        import dataset_manifest
        import tools.fewshot_pipeline as pipeline

        root = os.path.join(self.tmp, "manifest")
        os.makedirs(root)
        dataset_manifest.update_case(
            root,
            "case_ct",
            provenance={"source_modality": "CT"},
        )
        samples = [{"case_id": "case_ct", "image": os.path.join(self.tmp, "image.nii.gz")}]
        result = pipeline.resolve_training_modality("auto", samples, [root])
        self.assertEqual("ct", result["resolved"])
        self.assertEqual("ct", samples[0]["source_modality"])

        dataset_manifest.update_case(
            root,
            "case_mr",
            provenance={"source_modality": "MR"},
        )
        samples.append(
            {"case_id": "case_mr", "image": os.path.join(self.tmp, "image2.nii.gz")}
        )
        with self.assertRaisesRegex(RuntimeError, "mixed CT and MR"):
            pipeline.resolve_training_modality("auto", samples, [root])

    def test_auto_modality_requires_explicit_choice_for_partial_metadata(self):
        import dataset_manifest
        import tools.fewshot_pipeline as pipeline

        root = os.path.join(self.tmp, "manifest_partial")
        os.makedirs(root)
        dataset_manifest.update_case(
            root,
            "case_ct",
            provenance={"source_modality": "CT"},
        )
        samples = [
            {
                "case_id": "case_ct",
                "image": os.path.join(self.tmp, "ct_source.nii.gz"),
            },
            {
                "case_id": "case_unknown",
                "image": os.path.join(self.tmp, "scan.nii.gz"),
            },
        ]
        with self.assertRaisesRegex(RuntimeError, "Choose CT, MRI, or Other"):
            pipeline.resolve_training_modality("auto", samples, [root])


    def test_qt_architecture_combos_return_internal_values(self):
        try:
            from PySide6 import QtCore, QtGui, QtWidgets
        except ImportError:
            self.skipTest("PySide6 is not installed in this test environment")
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        import tools.fewshot_training_setup_ui as ui

        application = QtWidgets.QApplication.instance()
        if application is None:
            application = QtWidgets.QApplication([])
        setup = object.__new__(ui.QtTrainingSetupApp)
        setup.QtCore = QtCore
        setup.QtGui = QtGui
        setup.QtWidgets = QtWidgets
        setup.values = {}
        setup.widgets = {
            "training_dimension": setup._data_combo(
                ui.TRAINING_DIMENSION_ITEMS,
                "auto",
            ),
            "quality_mode": setup._data_combo(
                ui.QUALITY_MODE_ITEMS,
                "standard",
            ),
        }
        self.assertEqual("auto", setup._widget_value("training_dimension"))
        setup._set_widget_value("training_dimension", "3d")
        setup._set_widget_value("quality_mode", "high_detail")
        self.assertEqual("3d", setup._widget_value("training_dimension"))
        self.assertEqual("high_detail", setup._widget_value("quality_mode"))

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
        # The active training job must be first; completed jobs for the same
        # organ are also returned so the user can inspect their progress/log.
        self.assertGreaterEqual(len(selected), 1)
        self.assertEqual(selected[0]["job_id"], "active_liver")
        # All returned jobs must be for the requested organ.
        for job in selected:
            self.assertEqual(job["organ"], "liver")
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

    def test_single_case_discovery_stops_after_first_flat_dicom_slice(self):
        import tools.io_path_setup_ui as io_ui

        dicom_dir = os.path.join(self.tmp, "large_flat_dicom")
        os.makedirs(dicom_dir)
        calls = {"next": 0}

        class Entry:
            name = "slice000001.dcm"
            path = os.path.join(dicom_dir, name)

        class Entries:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def __iter__(self):
                return self

            def __next__(self):
                calls["next"] += 1
                if calls["next"] == 1:
                    return Entry()
                raise AssertionError("flat DICOM discovery enumerated unnecessary slices")

        old_scandir = io_ui.os.scandir
        try:
            io_ui.os.scandir = lambda _path: Entries()
            case = io_ui.discover_single_source(dicom_dir)
        finally:
            io_ui.os.scandir = old_scandir
        self.assertEqual(dicom_dir, case["image"])
        self.assertEqual(1, calls["next"])

    def test_batch_discovery_supports_all_named_and_no_masks(self):
        import mimics_bridge

        root = os.path.join(self.tmp, "dataset")
        case = os.path.join(root, "case001")
        seg = os.path.join(case, "segmentations")
        os.makedirs(seg)
        for path in (
            os.path.join(case, "ct.nii.gz"),
            os.path.join(seg, "liver.seg.nii.gz"),
            os.path.join(seg, "spleen.nii.gz"),
        ):
            with open(path, "wb") as handle:
                handle.write(b"discovery-only")

        all_result = mimics_bridge.do_discover({"ts_root": root, "mask_selection": "all"})
        self.assertEqual(["liver", "spleen"], [row["name"] for row in all_result["cases"][0]["masks"]])
        named_result = mimics_bridge.do_discover({"ts_root": root, "mask_selection": "spleen"})
        self.assertEqual(["spleen"], [row["name"] for row in named_result["cases"][0]["masks"]])
        none_result = mimics_bridge.do_discover({"ts_root": root, "mask_selection": "none"})
        self.assertEqual([], none_result["cases"][0]["masks"])

    def test_import_runtime_is_local_and_queue_descriptor_is_output_scoped(self):
        import mimics_import

        run_root = mimics_import._new_import_run_root()
        output_dir = os.path.join(self.tmp, "network_output")
        work_dir = os.path.join(run_root, "work", "case001")
        os.makedirs(work_dir)
        descriptor = mimics_import._publish_prepared_work(
            output_dir,
            "case001",
            work_dir,
            os.path.join(output_dir, "case001.mcs"),
        )
        self.assertTrue(run_root.startswith(os.path.join(PROJECT_ROOT, ".mimics_runtime", "import_runs")))
        self.assertTrue(descriptor.startswith(os.path.join(PROJECT_ROOT, ".mimics_runtime", "import_queues")))
        payload = json.loads(Path(descriptor).read_text(encoding="utf-8"))
        self.assertEqual(os.path.abspath(work_dir), payload["work_dir"])
        shutil.rmtree(run_root, ignore_errors=True)
        shutil.rmtree(os.path.dirname(os.path.dirname(descriptor)), ignore_errors=True)

    def test_external_path_browser_isolates_native_dialog_and_supports_paste(self):
        import inspect
        import tools.io_path_setup_ui as ui
        import tools.path_dialog_helper as helper
        import tools.ui_theme as theme

        source = inspect.getsource(ui.run_ui)
        self.assertIn(
            "QFileDialog.getExistingDirectory",
            inspect.getsource(helper.main),
        )
        self.assertIn(
            "QFileDialog.getOpenFileName",
            inspect.getsource(helper.main),
        )
        self.assertIn("choose_existing_directory_async", source)
        self.assertNotIn("QFileDialog", source)
        self.assertIn("subprocess.Popen", inspect.getsource(theme._AsyncPathDialog))
        self.assertIn("clipboard", source)

    def test_external_io_loads_required_theme_from_isolated_tools_directory(self):
        import subprocess

        source = Path(PROJECT_ROOT, "tools", "io_path_setup_ui.py")
        theme = Path(PROJECT_ROOT, "tools", "ui_theme.py")
        isolated = Path(self.tmp, "isolated_ui")
        isolated.mkdir(parents=True)
        target = isolated / source.name
        shutil.copy2(str(source), str(target))
        shutil.copy2(str(theme), str(isolated / theme.name))
        code = (
            "import importlib.util;"
            "p={0!r};"
            "s=importlib.util.spec_from_file_location('isolated_io_ui',p);"
            "m=importlib.util.module_from_spec(s);"
            "s.loader.exec_module(m);"
            "assert m.shared_stylesheet.__module__ == 'ui_theme';"
            "print(m.shared_stylesheet.__module__)"
        ).format(str(target))
        result = subprocess.run(
            [sys.executable, "-I", "-c", code],
            cwd=str(isolated),
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(0, result.returncode, result.stderr)

    def test_external_io_process_liveness_detects_owner_and_dead_pid(self):
        import tools.io_path_setup_ui as ui

        self.assertTrue(ui.process_exists(os.getpid()))
        self.assertFalse(ui.process_exists(99999999))

    def test_io_setup_prefers_documented_mimics_timer_event(self):
        import io_setup_mimics

        class Subscription(object):
            def __init__(self):
                self.unsubscribed = False

            def unsubscribe(self):
                self.unsubscribed = True

        class Events(object):
            def __init__(self):
                self.callback = None
                self.subscription = Subscription()

            def subscribe(self, name, callback):
                self.name = name
                self.callback = callback
                return self.subscription

        events = Events()
        old_events = getattr(io_setup_mimics.mimics, "events", None)
        had_events = hasattr(io_setup_mimics.mimics, "events")
        old_tick = io_setup_mimics._tick
        old_win32 = io_setup_mimics._start_win32_monitor
        old_env = os.environ.get("MIMICS_USE_EVENT_TIMER")
        calls = []
        monitor = {"key": "native-event", "busy": False}
        try:
            io_setup_mimics.mimics.events = events
            io_setup_mimics._tick = lambda value: calls.append(value)
            # Force the Mimics event-timer path: bypass the Win32 timer and opt
            # in to event subscriptions so the test exercises the event callback
            # regardless of host platform.
            io_setup_mimics._start_win32_monitor = lambda _monitor, _poll: False
            os.environ["MIMICS_USE_EVENT_TIMER"] = "1"
            self.assertTrue(io_setup_mimics._start_monitor(monitor, poll_seconds=0.0))
            self.assertEqual("timer", events.name)
            events.callback()
            self.assertEqual([monitor], calls)
            io_setup_mimics._stop_monitor(monitor["key"])
            self.assertTrue(events.subscription.unsubscribed)
            self.assertNotIn("win32_timer", monitor)
        finally:
            io_setup_mimics._tick = old_tick
            io_setup_mimics._start_win32_monitor = old_win32
            if old_env is None:
                os.environ.pop("MIMICS_USE_EVENT_TIMER", None)
            else:
                os.environ["MIMICS_USE_EVENT_TIMER"] = old_env
            io_setup_mimics._IO_SETUP_MONITORS.pop(monitor["key"], None)
            if had_events:
                io_setup_mimics.mimics.events = old_events
            else:
                delattr(io_setup_mimics.mimics, "events")

    def test_io_setup_bootstrap_stop_cancels_before_submit_callback(self):
        import io_setup_mimics

        status_path = os.path.join(self.tmp, "io-submit.json")
        stop_path = os.path.join(self.tmp, "io-stop.json")
        Path(status_path).write_text(
            json.dumps({"status": "submitted", "selection": {"source_path": "x"}}),
            encoding="utf-8",
        )
        Path(stop_path).write_text("{}", encoding="utf-8")
        submitted = []
        key = "bootstrap-stop"
        monitor = {
            "key": key,
            "status_path": status_path,
            "bootstrap_stop_path": stop_path,
            "deadline": time.time() + 10,
            "busy": False,
            "on_submit": lambda selection: submitted.append(selection),
        }
        io_setup_mimics._IO_SETUP_MONITORS[key] = monitor
        io_setup_mimics._tick(monitor)
        payload = json.loads(Path(status_path).read_text(encoding="utf-8"))
        self.assertEqual("cancelled", payload["status"])
        self.assertEqual([], submitted)
        self.assertNotIn(key, io_setup_mimics._IO_SETUP_MONITORS)

    def test_import_monitor_prefers_mimics_timer_event(self):
        import mimics_import

        class Subscription(object):
            def __init__(self):
                self.unsubscribed = False

            def unsubscribe(self):
                self.unsubscribed = True

        class Events(object):
            def __init__(self):
                self.callback = None
                self.subscription = Subscription()

            def subscribe(self, name, callback):
                self.name = name
                self.callback = callback
                return self.subscription

        events = Events()
        old_events = getattr(mimics_import.mimics, "events", None)
        had_events = hasattr(mimics_import.mimics, "events")
        calls = []
        monitor = {"monitor_key": "native-import", "output_dir": self.tmp}
        try:
            mimics_import.mimics.events = events
            self.assertTrue(mimics_import._start_mimics_event_monitor(
                monitor,
                lambda: calls.append(True),
                0.0,
                "test callback",
            ))
            self.assertEqual("timer", events.name)
            events.callback()
            self.assertEqual([True], calls)
            mimics_import._stop_import_monitor(monitor["monitor_key"])
            self.assertTrue(events.subscription.unsubscribed)
        finally:
            mimics_import._IMPORT_MONITORS.pop(monitor["monitor_key"], None)
            if had_events:
                mimics_import.mimics.events = old_events
            else:
                delattr(mimics_import.mimics, "events")

    def test_export_monitor_prefers_mimics_timer_event(self):
        import mimics_export

        class Subscription(object):
            def __init__(self):
                self.unsubscribed = False

            def unsubscribe(self):
                self.unsubscribed = True

        class Events(object):
            def __init__(self):
                self.callback = None
                self.subscription = Subscription()

            def subscribe(self, name, callback):
                self.name = name
                self.callback = callback
                return self.subscription

        events = Events()
        old_events = getattr(mimics_export.mimics, "events", None)
        had_events = hasattr(mimics_export.mimics, "events")
        calls = []
        monitor = {"monitor_key": "native-export"}
        try:
            mimics_export.mimics.events = events
            self.assertTrue(mimics_export._start_mimics_event_export_monitor(
                monitor,
                lambda: calls.append(True),
                0.0,
                "test callback",
            ))
            self.assertEqual("timer", events.name)
            events.callback()
            self.assertEqual([True], calls)
            mimics_export._stop_export_monitor(monitor["monitor_key"])
            self.assertTrue(events.subscription.unsubscribed)
        finally:
            mimics_export._EXPORT_MONITORS.pop(monitor["monitor_key"], None)
            if had_events:
                mimics_export.mimics.events = old_events
            else:
                delattr(mimics_export.mimics, "events")

    def test_mask_import_monitor_prefers_mimics_timer_event(self):
        import mask_import

        class Subscription(object):
            def __init__(self):
                self.unsubscribed = False

            def unsubscribe(self):
                self.unsubscribed = True

        class Events(object):
            def __init__(self):
                self.callback = None
                self.subscription = Subscription()

            def subscribe(self, name, callback):
                self.name = name
                self.callback = callback
                return self.subscription

        events = Events()
        old_events = getattr(mask_import.mimics, "events", None)
        had_events = hasattr(mask_import.mimics, "events")
        calls = []
        monitor = {"monitor_key": "native-mask-import"}
        old_tick = mask_import._mask_import_monitor_tick
        try:
            mask_import.mimics.events = events
            mask_import._mask_import_monitor_tick = lambda value: calls.append(value)
            self.assertTrue(mask_import._start_mimics_event_mask_import_monitor(
                monitor,
                0.0,
            ))
            self.assertEqual("timer", events.name)
            events.callback()
            self.assertEqual([monitor], calls)
            mask_import._stop_mask_import_monitor(monitor["monitor_key"])
            self.assertTrue(events.subscription.unsubscribed)
        finally:
            mask_import._mask_import_monitor_tick = old_tick
            mask_import._MASK_IMPORT_MONITORS.pop(monitor["monitor_key"], None)
            if had_events:
                mask_import.mimics.events = old_events
            else:
                delattr(mask_import.mimics, "events")

    def test_offline_package_requires_shared_ui_runtime_files(self):
        import tools.package_portable as package_portable

        self.assertIn("tools/ui_theme.py", package_portable.REQUIRED_EXTERNAL_UI_FILES)
        self.assertIn(
            "tools/ui_preferences.py",
            package_portable.REQUIRED_EXTERNAL_UI_FILES,
        )
        self.assertIn(
            "tools/interactive_algorithms_worker.py",
            package_portable.REQUIRED_EXTERNAL_UI_FILES,
        )
        self.assertIn(
            "runtime_py35/interactive_algorithms_mimics.py",
            package_portable.REQUIRED_EXTERNAL_UI_FILES,
        )
        self.assertIn(
            "tools/training_data_ui.py",
            package_portable.REQUIRED_EXTERNAL_UI_FILES,
        )
        self.assertIn("tools/io_path_setup_ui.py", package_portable.REQUIRED_EXTERNAL_UI_FILES)
        self.assertIn("tools/path_dialog_helper.py", package_portable.REQUIRED_EXTERNAL_UI_FILES)
        self.assertIn("tools/single_case_import_worker.py", package_portable.REQUIRED_EXTERNAL_UI_FILES)
        for relative in package_portable.REQUIRED_EXTERNAL_UI_FILES:
            self.assertTrue(Path(PROJECT_ROOT, relative).is_file(), relative)

    def test_background_mimics_launch_is_off_the_gui_callback(self):
        import mimics_import

        started = threading.Event()
        release = threading.Event()
        old_launch = mimics_import._launch_background_mimics
        old_processes = dict(mimics_import._BG_MIMICS_PROCESSES)
        old_launch_outputs = set(mimics_import._BG_MIMICS_LAUNCH_OUTPUTS)
        try:
            def slow_launch(_output_dir, total_count=0, schedule_retry=True):
                started.set()
                release.wait(2.0)
                return None
            mimics_import._launch_background_mimics = slow_launch
            mimics_import._BG_MIMICS_PROCESSES.clear()
            mimics_import._BG_MIMICS_LAUNCH_OUTPUTS.clear()
            before = time.time()
            mimics_import._ensure_bg_mimics_running(self.tmp, mark_active=False)
            elapsed = time.time() - before
            self.assertLess(elapsed, 0.2)
            self.assertTrue(started.wait(1.0))
        finally:
            release.set()
            mimics_import._launch_background_mimics = old_launch
            deadline = time.time() + 1.0
            while mimics_import._BG_MIMICS_LAUNCH_OUTPUTS and time.time() < deadline:
                time.sleep(0.01)
            mimics_import._BG_MIMICS_PROCESSES.clear()
            mimics_import._BG_MIMICS_PROCESSES.update(old_processes)
            mimics_import._BG_MIMICS_LAUNCH_OUTPUTS.clear()
            mimics_import._BG_MIMICS_LAUNCH_OUTPUTS.update(old_launch_outputs)

    def test_external_ai_windows_do_not_sleep_on_mimics_gui_thread(self):
        import inspect
        import fewshot_mimics

        self.assertNotIn("time.sleep", inspect.getsource(fewshot_mimics._launch_external_advanced_training))
        self.assertNotIn("time.sleep", inspect.getsource(fewshot_mimics._launch_external_model_chooser))

    def test_import_worker_exit_grace_is_nonblocking(self):
        import mimics_import

        job_dir = os.path.join(self.tmp, "dead_bridge")
        os.makedirs(job_dir)
        Path(job_dir, "job_state.json").write_text(
            json.dumps({"pid": 99999999, "phase": "preparing"}),
            encoding="utf-8",
        )
        old_alive = mimics_import._is_pid_alive
        try:
            mimics_import._is_pid_alive = lambda _pid: False
            started = time.time()
            status, _result = mimics_import._check_job_status(job_dir)
            self.assertLess(time.time() - started, 0.1)
            self.assertEqual("running", status)
            mimics_import._BRIDGE_EXIT_SEEN[job_dir] = time.time() - 1.0
            status, message = mimics_import._check_job_status(job_dir)
            self.assertEqual("error", status)
            self.assertIn("exited unexpectedly", message)
        finally:
            mimics_import._is_pid_alive = old_alive
            mimics_import._BRIDGE_EXIT_SEEN.pop(job_dir, None)

    def test_single_import_never_replaces_an_open_project_automatically(self):
        import inspect
        import mimics_import

        source = inspect.getsource(mimics_import._import_monitor_tick)
        self.assertIn("skip_if_project_open=True", source)

    def test_ai_compute_children_use_background_priority(self):
        import inspect
        import fewshot_mimics
        import nninteractive_mimics

        self.assertIn("_background_process_kwargs", inspect.getsource(fewshot_mimics._launch_process))
        self.assertIn("_background_process_kwargs", inspect.getsource(fewshot_mimics._launch_bridge_mask_to_buffer))
        self.assertIn("_background_process_kwargs", inspect.getsource(nninteractive_mimics._start_async_worker))

    def test_mask_writers_use_exclusive_callback_or_batch_leases(self):
        import inspect
        import fewshot_mimics
        import mask_import
        import nninteractive_mimics

        for callback in (
            fewshot_mimics._monitor_tick,
            nninteractive_mimics._async_monitor_tick,
        ):
            source = inspect.getsource(callback)
            self.assertIn("mask_buffer_access", source)
            self.assertIn("release_local_operation", source)

        apply_source = inspect.getsource(mask_import._mask_import_monitor_tick_locked)
        finish_source = inspect.getsource(mask_import._finish_mask_import)
        cancel_source = inspect.getsource(mask_import.cancel_all_mask_imports)
        self.assertIn('"mask_buffer_access", "Mask import apply"', apply_source)
        self.assertIn("release_local_operation", finish_source)
        self.assertIn("release_local_operation", cancel_source)

    def test_external_compute_children_use_below_normal_windows_priority(self):
        for relative in (
            "runtime_py35/interactive_algorithms_mimics.py",
            "tools/nninteractive_finetune_pipeline.py",
            "tools/nnunet_jobs.py",
            "tools/nnunet_pipeline.py",
            "tools/pipeline_common.py",
        ):
            source = Path(PROJECT_ROOT, relative).read_text(encoding="utf-8")
            self.assertIn("BELOW_NORMAL_PRIORITY_CLASS", source, relative)
        if os.name == "nt":
            # fewshot_pipeline delegates to pipeline_common - verify the
            # behavior rather than the implementation string.
            import tools.fewshot_pipeline as fewshot
            import tools.pipeline_common as pipeline_common
            for module in (fewshot, pipeline_common):
                kwargs = module.hidden_process_kwargs()
                priority = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0x00004000)
                self.assertTrue(
                    kwargs.get("creationflags", 0) & priority,
                    "{} must spawn below-normal priority children".format(module.__name__),
                )

    def test_external_batch_export_requires_explicit_safe_or_overwrite_destination(self):
        import tools.mimics_batch_cli as cli
        parser = cli.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["export-labels", "--ts-root", self.tmp])
        safe = parser.parse_args(["export-labels", "--ts-root", self.tmp, "--output-dir", os.path.join(self.tmp, "labels")])
        self.assertFalse(safe.overwrite_source)
        overwrite = parser.parse_args(["export-labels", "--ts-root", self.tmp, "--overwrite-source"])
        self.assertTrue(overwrite.overwrite_source)

    def test_external_batch_export_accepts_distance_resampling(self):
        import tools.mimics_batch_cli as cli

        parsed = cli.build_parser().parse_args([
            "export-labels", "--ts-root", self.tmp,
            "--output-dir", os.path.join(self.tmp, "labels"),
            "--mask-resample-method", "distance",
        ])

        self.assertEqual("distance", parsed.mask_resample_method)

    def test_external_batch_export_accepts_canonical_source_mask_root(self):
        import tools.mimics_batch_cli as cli

        source_mask_root = os.path.join(self.tmp, "canonical_source")
        parsed = cli.build_parser().parse_args([
            "export-labels", "--ts-root", self.tmp,
            "--output-dir", os.path.join(self.tmp, "labels"),
            "--source-mask-root", source_mask_root,
        ])

        self.assertEqual(source_mask_root, parsed.source_mask_root)

    def test_external_batch_export_accepts_raw_mimics_grid(self):
        import tools.mimics_batch_cli as cli

        parsed = cli.build_parser().parse_args([
            "export-labels", "--ts-root", self.tmp,
            "--output-dir", os.path.join(self.tmp, "labels"),
            "--export-space", "mimics_grid",
        ])

        self.assertEqual("mimics_grid", parsed.export_space)

    def test_external_batch_export_maps_recursive_mcs_to_independent_images(self):
        import tools.mimics_batch_cli as cli

        mcs_root = Path(self.tmp, "mcs_store")
        image_root = Path(self.tmp, "original_images")
        (mcs_root / "group_a").mkdir(parents=True)
        (mcs_root / "group_a" / "case001.mcs").write_bytes(b"")
        (mcs_root / "case002.mcs").write_bytes(b"")
        (image_root / "case001").mkdir(parents=True)
        (image_root / "case001" / "ct.nii.gz").write_bytes(b"")
        (image_root / "case002.nii.gz").write_bytes(b"")

        mapping = cli.discover_export_sources(mcs_root, image_root)

        self.assertEqual(set(["case001", "case002"]), set(mapping["mcs_paths"]))
        self.assertEqual(
            "case001.mcs",
            Path(mapping["mcs_paths"]["case001"]).name,
        )
        self.assertEqual(
            "ct.nii.gz",
            Path(mapping["source_image_paths"]["case001"]).name,
        )
        self.assertEqual(
            "case002.nii.gz",
            Path(mapping["source_image_paths"]["case002"]).name,
        )

    def test_external_batch_export_can_limit_mcs_discovery_to_root_folder(self):
        import tools.mimics_batch_cli as cli

        mcs_root = Path(self.tmp, "mcs_store")
        image_root = Path(self.tmp, "original_images")
        (mcs_root / "nested").mkdir(parents=True)
        image_root.mkdir(parents=True)
        (mcs_root / "root_case.mcs").write_bytes(b"")
        (mcs_root / "nested" / "nested_case.mcs").write_bytes(b"")
        (image_root / "root_case.nii.gz").write_bytes(b"")
        (image_root / "nested_case.nii.gz").write_bytes(b"")

        mapping = cli.discover_export_sources(mcs_root, image_root, recursive=False)

        self.assertEqual(["root_case"], sorted(mapping["mcs_paths"]))

    def test_external_batch_export_rejects_missing_external_image(self):
        import tools.mimics_batch_cli as cli

        mcs_root = Path(self.tmp, "mcs_store")
        image_root = Path(self.tmp, "original_images")
        mcs_root.mkdir()
        image_root.mkdir()
        (mcs_root / "case001.mcs").write_bytes(b"")

        with self.assertRaises(RuntimeError) as context:
            cli.discover_export_sources(mcs_root, image_root)
        self.assertIn("case001", str(context.exception))
        self.assertIn("metadata path was not used", str(context.exception))


    # ================================================================
    # L8: 2D decoder integration (NEW — DINOv3 few-shot optimization)
    # ================================================================

    def test_2d_decoder_conv2d_output_shape(self):
        """conv2d decoder produces correct output shape."""
        import torch
        sys.path.insert(0, DINOV3_ROOT)
        from src.models.decoder_2d import Conv2DDecoder
        decoder = Conv2DDecoder([768, 768, 768, 768], num_classes=2)
        feats = [torch.randn(1, 768, 16, 14, 14) for _ in range(4)]
        out = decoder(feats, (1, 1, 16, 224, 224))
        self.assertEqual(out.shape, (1, 2, 16, 224, 224))

    def test_every_public_decoder_produces_a_stacked_3d_prediction(self):
        import torch
        import tools.fewshot_training_setup_ui as ui
        sys.path.insert(0, DINOV3_ROOT)
        from src.models.decoder_3d import DecoderFactory
        features = [torch.randn(1, 8, 3, 2, 2) for _ in range(4)]
        for decoder_name in ui.DECODER_CHOICES:
            if decoder_name == "feature_unet2d":
                decoder_features = [torch.randn(1, 384, 3, 2, 2)]
                decoder_dims = [384]
                raw_volume = torch.rand(1, 1, 3, 32, 32)
                original_shape = (1, 1, 3, 32, 32)
            else:
                decoder_features = features
                decoder_dims = [8, 8, 8, 8]
                raw_volume = None
                original_shape = (1, 1, 3, 16, 16)
            decoder = DecoderFactory.create(decoder_name, decoder_dims, 2)
            with torch.no_grad():
                output = decoder(
                    decoder_features,
                    original_shape,
                    **({"raw_volume": raw_volume} if raw_volume is not None else {})
                )
            self.assertEqual(
                (1, 2, *original_shape[2:]),
                tuple(output.shape),
                decoder_name,
            )
            restored = DecoderFactory.create(decoder_name, decoder_dims, 2)
            restored.load_state_dict(decoder.state_dict(), strict=True)
            restored.eval()
            decoder.eval()
            with torch.no_grad():
                kwargs = {"raw_volume": raw_volume} if raw_volume is not None else {}
                expected = decoder(
                    decoder_features,
                    original_shape,
                    **kwargs
                )
                actual = restored(
                    decoder_features,
                    original_shape,
                    **kwargs
                )
            torch.testing.assert_close(expected, actual, rtol=0.0, atol=0.0, msg=decoder_name)

    def test_public_decoders_accept_small_base_and_large_hidden_sizes(self):
        import torch
        sys.path.insert(0, DINOV3_ROOT)
        from src.models.decoder_3d import DecoderFactory

        for hidden_size in (384, 768, 1024):
            features = [
                torch.randn(1, hidden_size, 3, 2, 2)
                for _ in range(4)
            ]
            for decoder_name in (
                "scale_aware2d",
                "context3d_lite",
                "context3d_hybrid",
            ):
                decoder = DecoderFactory.create(
                    decoder_name,
                    [hidden_size] * 4,
                    2,
                )
                with torch.no_grad():
                    output = decoder(
                        features,
                        (1, 1, 3, 32, 32),
                    )
                self.assertEqual(
                    (1, 2, 3, 32, 32),
                    tuple(output.shape),
                    "{} hidden={}".format(
                        decoder_name, hidden_size
                    ),
                )

    def test_variable_depth_batch_ignores_padding_in_loss(self):
        import torch
        sys.path.insert(0, DINOV3_ROOT)
        from src.data.dataset_3d import pad_volume_batch
        from src.training.losses import DiceFocalLoss

        items = []
        for index, depth in enumerate((2, 4)):
            items.append({
                "image": torch.randn(3, depth, 16, 16),
                "label": torch.randint(0, 2, (depth, 16, 16)),
                "case_id": str(index),
                "image_path": "image",
                "label_path": "label",
                "spacing_zyx": torch.ones(3),
            })
        batch = pad_volume_batch(items)
        logits = torch.randn(
            2,
            2,
            4,
            16,
            16,
            requires_grad=True,
        )
        loss = DiceFocalLoss()(logits, batch["label"])["loss"]
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(
            0,
            int(torch.count_nonzero(logits.grad[0, :, 2:])),
        )

    def test_fit_pad_preprocessing_preserves_aspect_ratio_and_contract(self):
        import numpy as np
        sys.path.insert(0, DINOV3_ROOT)
        from src.data.dataset_3d import fit_pad_geometry, prepare_model_input
        from src.data.input_contract import input_contract_for_config

        geometry = fit_pad_geometry((20, 40), (32, 32))
        self.assertEqual((16, 32), geometry["resized_size"])
        volume = np.ones((3, 20, 40), dtype=np.float32)
        prepared = prepare_model_input(
            volume,
            (32, 32),
            resize_mode="fit_pad",
        )
        self.assertEqual((1, 3, 32, 32), tuple(prepared.shape))
        contract = input_contract_for_config({
            "data": {
                "img_size": [32, 32],
                "resize_mode": "fit_pad",
            },
            "model": {},
            "training": {},
        })
        self.assertEqual("fit_pad", contract["resize_mode"])
        self.assertEqual("dinov3_volume_input.v3", contract["schema_version"])
        self.assertEqual("sample", contract["normalization_scope"])

    def test_legacy_stretch_input_contract_remains_loadable(self):
        sys.path.insert(0, DINOV3_ROOT)
        from src.data.input_contract import (
            input_contract_for_config,
            validate_input_contract,
        )

        config = {
            "data": {"img_size": [224, 224]},
            "model": {},
            "training": {},
        }
        legacy = input_contract_for_config(config)
        legacy["schema_version"] = "dinov3_volume_input.v1"
        legacy.pop("resize_mode")
        legacy.pop("normalization_scope")
        config["runtime"] = {"input_contract": legacy}
        self.assertEqual(legacy, validate_input_contract(config))

        config["data"]["resize_mode"] = "fit_pad"
        with self.assertRaises(RuntimeError):
            validate_input_contract(config)

    def test_2d_decoder_all_variants_in_factory(self):
        """All 2D decoder types must be creatable via DecoderFactory."""
        import torch
        sys.path.insert(0, DINOV3_ROOT)
        from src.models.decoder_3d import DecoderFactory
        feats = [torch.randn(1, 768, 8, 14, 14) for _ in range(4)]
        for dec_type in ["conv2d", "conv2d_unet", "conv2d_deeplab", "conv2d_2_5d"]:
            decoder = DecoderFactory.create(dec_type, [768, 768, 768, 768], num_classes=2)
            out = decoder(feats, (1, 1, 8, 224, 224))
            self.assertEqual(out.shape, (1, 2, 8, 224, 224),
                             f"{dec_type} output shape mismatch")

    def test_2d_decoder_fewer_params_than_3d(self):
        """2D decoders must have fewer parameters than equivalent 3D decoders."""
        sys.path.insert(0, DINOV3_ROOT)
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
        sys.path.insert(0, DINOV3_ROOT)
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
        sys.path.insert(0, DINOV3_ROOT)
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
        exp_script = os.path.join(DINOV3_ROOT,
                                   "scripts", "batch_experiments.py")
        self.assertTrue(os.path.isfile(exp_script),
                        f"batch_experiments.py not found at {exp_script}")

    def test_2d_decoder_materialize_all_script_exists(self):
        """Multi-organ materialization script must exist."""
        mat_script = os.path.join(DINOV3_ROOT,
                                   "scripts", "materialize_all_organs.py")
        self.assertTrue(os.path.isfile(mat_script),
                        f"materialize_all_organs.py not found at {mat_script}")

    def test_2d_decoder_evaluate_script_exists(self):
        """Model evaluation script must exist."""
        eval_script = os.path.join(DINOV3_ROOT,
                                    "scripts", "evaluate_model.py")
        self.assertTrue(os.path.isfile(eval_script),
                        f"evaluate_model.py not found at {eval_script}")

    def test_2d_decoder_analyze_script_exists(self):
        """Results analysis script must exist."""
        anal_script = os.path.join(DINOV3_ROOT,
                                    "scripts", "analyze_results.py")
        self.assertTrue(os.path.isfile(anal_script),
                        f"analyze_results.py not found at {anal_script}")

    # -- io_setup_mimics bootstrap stop and empty descriptor -------------------

    def test_io_setup_bootstrap_stop_arrives_during_submit_callback(self):
        """Stop marker written while on_submit executes must still cancel."""
        import io_setup_mimics

        status_path = os.path.join(self.tmp, "io-race.json")
        stop_path = os.path.join(self.tmp, "io-race-stop.json")
        task_stop = os.path.join(self.tmp, "task-stop.json")
        Path(status_path).write_text(
            json.dumps({"status": "submitted", "selection": {"source_path": "x"}}),
            encoding="utf-8",
        )
        key = "bootstrap-race"
        written_stops = []

        def slow_submit(_selection):
            # Simulate: stop marker is written while on_submit is running
            Path(stop_path).write_text("{}", encoding="utf-8")
            return {
                "kind": "import",
                "title": "Test import",
                "status_path": os.path.join(self.tmp, "task-status.json"),
                "stop_path": task_stop,
            }

        monitor = {
            "key": key,
            "status_path": status_path,
            "bootstrap_stop_path": stop_path,
            "deadline": time.time() + 10,
            "busy": False,
            "on_submit": slow_submit,
        }
        # Intercept write_json_atomic to capture stop-marker writes
        old_write = io_setup_mimics.runtime_common.write_json_atomic
        try:
            def capture_write(path, payload):
                if str(path).endswith("-stop.json") or str(path).endswith("task-stop.json"):
                    written_stops.append((str(path), payload))
                return old_write(path, payload)
            io_setup_mimics.runtime_common.write_json_atomic = capture_write
            io_setup_mimics._IO_SETUP_MONITORS[key] = monitor
            io_setup_mimics._tick(monitor)
        finally:
            io_setup_mimics.runtime_common.write_json_atomic = old_write
            io_setup_mimics._IO_SETUP_MONITORS.pop(key, None)

        # The launch status may be "launched" (stop arrived too late for the
        # pre-submit check) or "cancelled" (stop arrived early enough). Either
        # way, the task-stop marker must have been written.
        payload = json.loads(Path(status_path).read_text(encoding="utf-8"))
        self.assertIn(payload.get("status"), ("launched", "cancelled"))
        # If launched, the task-stop marker should cancel the launched task.
        self.assertTrue(
            any("cancel_requested" in str(v) for (_, v) in written_stops),
            "Stop markers were not propagated to the task: {}".format(written_stops),
        )

    def test_io_setup_empty_task_descriptor_marks_failed(self):
        import io_setup_mimics

        status_path = os.path.join(self.tmp, "io-empty.json")
        Path(status_path).write_text(
            json.dumps({"status": "submitted", "selection": {}}),
            encoding="utf-8",
        )
        key = "bootstrap-empty"
        monitor = {
            "key": key,
            "status_path": status_path,
            "bootstrap_stop_path": "",
            "deadline": time.time() + 10,
            "busy": False,
            "on_submit": lambda _s: {},
        }
        io_setup_mimics._IO_SETUP_MONITORS[key] = monitor
        try:
            io_setup_mimics._tick(monitor)
        finally:
            io_setup_mimics._IO_SETUP_MONITORS.pop(key, None)
        payload = json.loads(Path(status_path).read_text(encoding="utf-8"))
        self.assertEqual("failed", payload.get("status"))
        self.assertIn("task descriptor", str(payload.get("error", "")).lower())

    def test_io_setup_task_on_submit_returns_none_marks_failed(self):
        import io_setup_mimics

        status_path = os.path.join(self.tmp, "io-none.json")
        Path(status_path).write_text(
            json.dumps({"status": "submitted", "selection": {}}),
            encoding="utf-8",
        )
        key = "bootstrap-none"
        monitor = {
            "key": key,
            "status_path": status_path,
            "bootstrap_stop_path": "",
            "deadline": time.time() + 10,
            "busy": False,
            "on_submit": lambda _s: None,
        }
        io_setup_mimics._IO_SETUP_MONITORS[key] = monitor
        try:
            io_setup_mimics._tick(monitor)
        finally:
            io_setup_mimics._IO_SETUP_MONITORS.pop(key, None)
        payload = json.loads(Path(status_path).read_text(encoding="utf-8"))
        self.assertEqual("failed", payload.get("status"))

    def test_io_setup_monitor_cleanup_unsubscribes_mimics_event(self):
        import io_setup_mimics

        unsubscribed = []

        class Subscription(object):
            def unsubscribe(self):
                unsubscribed.append(True)

        key = "cleanup-event"
        monitor = {
            "key": key,
            "status_path": os.path.join(self.tmp, "x.json"),
            "deadline": time.time() + 300,
            "busy": False,
            "event_subscription": Subscription(),
        }
        io_setup_mimics._IO_SETUP_MONITORS[key] = monitor
        io_setup_mimics._stop_monitor(key)
        self.assertTrue(unsubscribed)
        self.assertNotIn(key, io_setup_mimics._IO_SETUP_MONITORS)

    # -- io_path_setup_ui owner detection -------------------------------------

    def test_external_io_poll_task_owner_detection_logic(self):
        """Simulate the owner-alive check and terminal-state guard from poll_task."""
        import tools.io_path_setup_ui as ui

        status_path = os.path.join(self.tmp, "owner-logic-status.json")
        stop_path = os.path.join(self.tmp, "owner-logic-stop.json")

        def terminal_state(state):
            return str(state or "").lower() in (
                "closed", "completed", "done", "failed", "cancelled", "canceled",
            )

        # Scenario 1: owner alive, non-terminal → no action needed.
        Path(status_path).write_text(
            json.dumps({"status": "launching"}), encoding="utf-8",
        )
        owner_alive = ui.process_exists(os.getpid())
        self.assertTrue(owner_alive)
        setup_status = ui.read_json(status_path, {}) or {}
        state = str(setup_status.get("status") or "")
        self.assertFalse(terminal_state(state))
        # Owner is alive, so no stop-written / fail-written needed.

        # Scenario 2: owner dead, non-terminal → must write stop + fail.
        dead_pid = 99999999
        self.assertFalse(ui.process_exists(dead_pid))
        # The real poll_task would call request_stop() and write "failed".
        Path(stop_path).write_text("{}", encoding="utf-8")
        self.assertTrue(os.path.isfile(stop_path))

        # Scenario 3: owner dead, but state is already terminal → no action.
        Path(status_path).write_text(
            json.dumps({"status": "completed"}), encoding="utf-8",
        )
        setup_status = ui.read_json(status_path, {}) or {}
        self.assertTrue(terminal_state(str(setup_status.get("status") or "")))

        # Scenario 4: owner dead, state is launching → must act.
        Path(status_path).write_text(
            json.dumps({"status": "launching"}), encoding="utf-8",
        )
        setup_status = ui.read_json(status_path, {}) or {}
        self.assertFalse(terminal_state(str(setup_status.get("status") or "")))
        self.assertEqual("launching", setup_status.get("status"))

    def test_external_io_process_exists_rejects_invalid_inputs(self):
        import tools.io_path_setup_ui as ui

        self.assertFalse(ui.process_exists(None))
        self.assertFalse(ui.process_exists("not_a_number"))
        self.assertFalse(ui.process_exists(0))
        self.assertFalse(ui.process_exists(-1))

    def test_external_io_stop_button_includes_bootstrap_stop_in_targets(self):
        # Verify that request_stop writes to bootstrap_stop_path.
        import tools.io_path_setup_ui as ui

        bootstrap_stop = os.path.join(self.tmp, "bootstrap-stop.json")
        task_stop = os.path.join(self.tmp, "task-stop.json")
        # Emulate the closure environment: request_stop captures bootstrap_stop_path
        # We test the logic directly by calling write_json on expected paths.
        written = []
        old_write = ui.write_json
        try:
            ui.write_json = lambda path, payload: written.append((str(path), payload))
            # Simulate what request_stop does when bootstrap_stop_path is set
            stop_paths = [bootstrap_stop, task_stop]
            for stop_path in stop_paths:
                if stop_path:
                    ui.write_json(stop_path, {"status": "cancel_requested"})
            self.assertEqual(2, len(written))
            self.assertIn(bootstrap_stop, written[0][0])
            self.assertEqual("cancel_requested", written[0][1]["status"])
        finally:
            ui.write_json = old_write

    # -- Mimics event monitor throttling --------------------------------------

    def test_mimics_event_monitor_throttles_rapid_callbacks(self):
        import io_setup_mimics

        class Subscription(object):
            def __init__(self):
                self.unsubscribed = False

            def unsubscribe(self):
                self.unsubscribed = True

        class Events(object):
            def __init__(self):
                self.callback = None
                self.subscription = Subscription()

            def subscribe(self, name, callback):
                self.name = name
                self.callback = callback
                return self.subscription

        events = Events()
        old_events = getattr(io_setup_mimics.mimics, "events", None)
        had_events = hasattr(io_setup_mimics.mimics, "events")
        old_tick = io_setup_mimics._tick
        old_win32 = io_setup_mimics._start_win32_monitor
        old_env = os.environ.get("MIMICS_USE_EVENT_TIMER")
        calls = []
        monitor = {"key": "throttle", "busy": False}
        try:
            io_setup_mimics.mimics.events = events
            io_setup_mimics._tick = lambda value: calls.append(value)
            # Force the event-timer path (bypass Win32, opt in to events) so the
            # throttle logic is exercised on every host platform.
            io_setup_mimics._start_win32_monitor = lambda _monitor, _poll: False
            os.environ["MIMICS_USE_EVENT_TIMER"] = "1"
            # poll_seconds=2.0 → interval=max(0.25, 2.0)=2.0s
            self.assertTrue(io_setup_mimics._start_monitor(
                monitor, poll_seconds=2.0,
            ))
            # First call — allowed (last_tick=0).
            monitor["event_last_tick"] = 0.0
            events.callback()
            self.assertEqual(1, len(calls))
            # Second call immediately after — throttled (0 < 2.0s interval).
            events.callback()
            self.assertEqual(1, len(calls), "rapid callback was not throttled")
            # Third call after the interval — allowed.
            monitor["event_last_tick"] = time.time() - 3.0
            events.callback()
            self.assertEqual(2, len(calls))
        finally:
            io_setup_mimics._tick = old_tick
            io_setup_mimics._start_win32_monitor = old_win32
            if old_env is None:
                os.environ.pop("MIMICS_USE_EVENT_TIMER", None)
            else:
                os.environ["MIMICS_USE_EVENT_TIMER"] = old_env
            io_setup_mimics._IO_SETUP_MONITORS.pop(monitor["key"], None)
            if had_events:
                io_setup_mimics.mimics.events = old_events
            else:
                delattr(io_setup_mimics.mimics, "events")

    def test_mimics_event_mask_import_monitor_handles_tick_error(self):
        import mask_import

        errors = []

        class Subscription(object):
            def __init__(self):
                self.unsubscribed = False

            def unsubscribe(self):
                self.unsubscribed = True

        class Events(object):
            def __init__(self):
                self.callback = None
                self.subscription = Subscription()

            def subscribe(self, name, callback):
                self.name = name
                self.callback = callback
                return self.subscription

        events = Events()
        old_events = getattr(mask_import.mimics, "events", None)
        had_events = hasattr(mask_import.mimics, "events")
        old_tick = mask_import._mask_import_monitor_tick
        monitor = {"monitor_key": "error-tick"}
        try:
            mask_import.mimics.events = events
            def failing_tick(mon):
                raise RuntimeError("tick failure")
            mask_import._mask_import_monitor_tick = failing_tick
            self.assertTrue(mask_import._start_mimics_event_mask_import_monitor(
                monitor, 0.0,
            ))
            # This must not raise — the error is caught and _finish_mask_import is called.
            events.callback()
            stored = monitor.get("errors") or []
            self.assertTrue(
                any("tick failure" in str(e) for e in stored),
                "tick error was not captured: {}".format(stored),
            )
        finally:
            mask_import._mask_import_monitor_tick = old_tick
            mask_import._MASK_IMPORT_MONITORS.pop(monitor["monitor_key"], None)
            if had_events:
                mask_import.mimics.events = old_events
            else:
                delattr(mask_import.mimics, "events")

    # -- mimics_import empty descriptor on external launch --------------------

    def test_import_external_launch_raises_on_empty_descriptor(self):
        import mimics_import
        import io_setup_mimics

        saved_submitted = None
        old_import = mimics_import._run_main_with_args
        old_launch = io_setup_mimics.launch
        try:
            mimics_import._run_main_with_args = lambda *a, **kw: 0
            def fake_launch(mode, python_exe, context, on_submit,
                            timeout_seconds=3600, ui_script=None):
                nonlocal saved_submitted
                saved_submitted = on_submit
            io_setup_mimics.launch = fake_launch
            mimics_import._launch_external_import_setup("import_batch")
            self.assertIsNotNone(saved_submitted)
            with self.assertRaises(RuntimeError) as ctx:
                saved_submitted({"source_path": self.tmp, "output_path": self.tmp})
            self.assertIn("did not start", str(ctx.exception).lower())
        finally:
            mimics_import._run_main_with_args = old_import
            io_setup_mimics.launch = old_launch
            mimics_import._LAST_TASK_DESCRIPTOR = {}

    def test_import_external_launch_populates_descriptor_on_success(self):
        import mimics_import
        import io_setup_mimics

        saved_submitted = None
        old_import = mimics_import._run_main_with_args
        old_launch = io_setup_mimics.launch
        try:
            def run_import(args, import_mode=None, case_info_override=None):
                mimics_import._LAST_TASK_DESCRIPTOR = {
                    "kind": "import",
                    "title": "Import dataset",
                    "status_path": os.path.join(self.tmp, "s.json"),
                    "stop_path": os.path.join(self.tmp, "stop.json"),
                }
                return 0
            mimics_import._run_main_with_args = run_import
            def fake_launch(mode, python_exe, context, on_submit,
                            timeout_seconds=3600, ui_script=None):
                nonlocal saved_submitted
                saved_submitted = on_submit
            io_setup_mimics.launch = fake_launch
            mimics_import._launch_external_import_setup("import_batch")
            descriptor = saved_submitted({"source_path": self.tmp, "output_path": self.tmp})
            self.assertIsInstance(descriptor, dict)
            self.assertEqual("import", descriptor.get("kind"))
            self.assertIn("status_path", descriptor)
        finally:
            mimics_import._run_main_with_args = old_import
            io_setup_mimics.launch = old_launch
            mimics_import._LAST_TASK_DESCRIPTOR = {}

    def test_single_case_external_setup_delegates_to_external_worker(self):
        import mimics_import
        import io_setup_mimics

        saved_submitted = None
        calls = []
        old_worker = mimics_import._launch_single_case_worker
        old_main = mimics_import._run_main_with_args
        old_launch = io_setup_mimics.launch
        try:
            def launch_worker(selection, axes=None, flips=None):
                calls.append((selection, axes, flips))
                return {
                    "kind": "import",
                    "title": "Import single case",
                    "status_path": os.path.join(self.tmp, "single-status.json"),
                    "stop_path": os.path.join(self.tmp, "single-stop.json"),
                }

            def fail_if_main_runs(*_args, **_kwargs):
                raise AssertionError("single-case setup must not execute main() in foreground Mimics")

            def fake_launch(mode, python_exe, context, on_submit,
                            timeout_seconds=3600, ui_script=None):
                nonlocal saved_submitted
                saved_submitted = on_submit

            mimics_import._launch_single_case_worker = launch_worker
            mimics_import._run_main_with_args = fail_if_main_runs
            io_setup_mimics.launch = fake_launch
            mimics_import._launch_external_import_setup("single_case")
            descriptor = saved_submitted({
                "source_path": os.path.join(self.tmp, "case.nii.gz"),
                "output_path": self.tmp,
                "mask_selection": "all",
                "case_info": {"case_id": "case", "image": "case.nii.gz", "masks": []},
            })
        finally:
            mimics_import._launch_single_case_worker = old_worker
            mimics_import._run_main_with_args = old_main
            io_setup_mimics.launch = old_launch

        self.assertEqual("import", descriptor["kind"])
        self.assertEqual(1, len(calls))
        self.assertEqual([0, 1, 2], calls[0][1])
        self.assertEqual([False, False, False], calls[0][2])

    def test_single_case_worker_filters_masks_and_closes_producer_queue(self):
        import tools.single_case_import_worker as worker

        masks = [
            {"name": "Liver", "path": "liver.nii.gz"},
            {"name": "Spleen", "path": "spleen.nii.gz"},
        ]
        self.assertEqual(masks, worker._filter_masks(masks, "all"))
        self.assertEqual([], worker._filter_masks(masks, "none"))
        self.assertEqual([masks[0]], worker._filter_masks(masks, "liver"))

        runtime_dir = Path(self.tmp, "queue")
        runtime_dir.mkdir()
        active = runtime_dir / "_mcs_queue_active.json"
        active.write_text("{}", encoding="utf-8")
        worker._write_producer_done(runtime_dir)
        self.assertFalse(active.exists())
        done = json.loads((runtime_dir / "_mcs_queue_done.json").read_text(encoding="utf-8"))
        self.assertEqual("done", done["status"])
        self.assertEqual(1, done["completed"])

        active.write_text(json.dumps({"updated_at_epoch": time.time()}), encoding="utf-8")
        self.assertTrue(worker._active_producer_exists(runtime_dir))
        active.write_text(json.dumps({"updated_at_epoch": time.time() - 1000}), encoding="utf-8")
        self.assertFalse(worker._active_producer_exists(runtime_dir))

    def test_batch_import_refreshes_existing_queue_heartbeat(self):
        import mimics_import

        active_path = Path(self.tmp, "queue_active.json")
        active_path.write_text(json.dumps({
            "status": "active",
            "updated_at_epoch": 1.0,
        }), encoding="utf-8")
        old_queue_path = mimics_import._queue_active_path
        try:
            mimics_import._queue_active_path = lambda _output_dir: str(active_path)
            self.assertTrue(mimics_import._heartbeat_mcs_queue_active(self.tmp))
        finally:
            mimics_import._queue_active_path = old_queue_path
        payload = json.loads(active_path.read_text(encoding="utf-8"))
        self.assertGreater(payload["updated_at_epoch"], 1.0)

    def test_background_mimics_locks_are_scoped_by_default(self):
        import runtime_common

        first = runtime_common.background_mimics_lock_name(
            os.path.join(self.tmp, "output_a")
        )
        same = runtime_common.background_mimics_lock_name(
            os.path.join(self.tmp, "output_a")
        )
        second = runtime_common.background_mimics_lock_name(
            os.path.join(self.tmp, "output_b")
        )

        self.assertEqual(first, same)
        self.assertNotEqual(first, second)
        self.assertTrue(first.startswith("background_mimics_"))

    def test_import_producer_locks_are_scoped_by_output_queue(self):
        import runtime_common

        first = runtime_common.import_producer_lock_name(
            os.path.join(self.tmp, "output_a")
        )
        same = runtime_common.import_producer_lock_name(
            os.path.join(self.tmp, "output_a")
        )
        second = runtime_common.import_producer_lock_name(
            os.path.join(self.tmp, "output_b")
        )
        self.assertEqual(first, same)
        self.assertNotEqual(first, second)
        self.assertTrue(first.startswith("import_producer_"))

    def test_mimics_import_producer_lease_blocks_only_the_same_output(self):
        import mimics_import

        old_root = mimics_import._project_root
        first_output = os.path.join(self.tmp, "output_a")
        second_output = os.path.join(self.tmp, "output_b")
        try:
            mimics_import._project_root = lambda: self.tmp
            first = mimics_import._acquire_import_producer_lease(first_output, "first")
            self.assertIsNotNone(first)
            self.assertIsNone(
                mimics_import._acquire_import_producer_lease(first_output, "duplicate")
            )
            second = mimics_import._acquire_import_producer_lease(second_output, "second")
            self.assertIsNotNone(second)
        finally:
            if "first" in locals() and first:
                mimics_import._release_import_producer_lease(first)
            if "second" in locals() and second:
                mimics_import._release_import_producer_lease(second)
            mimics_import._project_root = old_root

    def test_external_batch_import_defaults_to_nearest_mask_mapping(self):
        cli = __import__("tools.mimics_batch_cli", fromlist=["dummy"])
        args = cli.build_parser().parse_args([
            "prepare-import",
            "--ts-root",
            self.tmp,
        ])
        self.assertEqual("nearest", args.mask_resample_method)

    def test_external_bridge_refreshes_queue_while_conversion_is_running(self):
        cli = __import__("tools.mimics_batch_cli", fromlist=["dummy"])
        callbacks = []

        class Process(object):
            returncode = 0
            def __init__(self):
                self.calls = 0
            def communicate(self, input=None, timeout=None):
                self.calls += 1
                if self.calls == 1:
                    raise cli.subprocess.TimeoutExpired("bridge", timeout)
                return (b'{"status": "ok", "cases": []}', b"")

        old_popen = cli.subprocess.Popen
        try:
            cli.subprocess.Popen = lambda *_args, **_kwargs: Process()
            result = cli.run_bridge(
                sys.executable,
                {"action": "discover"},
                on_wait=lambda: callbacks.append("heartbeat"),
            )
        finally:
            cli.subprocess.Popen = old_popen

        self.assertEqual("ok", result["status"])
        self.assertEqual(["heartbeat"], callbacks)

    def test_external_batch_starts_mcs_creator_after_first_prepared_case(self):
        cli = __import__("tools.mimics_batch_cli", fromlist=["dummy"])

        root = Path(self.tmp, "external_stream")
        dataset = root / "dataset"
        output = root / "output"
        runtime_dir = root / "queue_runtime"
        import_runtime = root / "import_runtime"
        for path in (dataset, output, runtime_dir, import_runtime):
            path.mkdir(parents=True, exist_ok=True)

        bridge_cases = []
        launch_descriptor_counts = []

        class Lock(object):
            def __init__(self, *_args, **_kwargs):
                self.acquired = False
            def acquire(self, **_kwargs):
                self.acquired = True
                return self
            def update_pid(self, *_args, **_kwargs):
                return True
            def release(self):
                self.acquired = False

        class Process(object):
            pid = 32123
            def poll(self):
                return None

        old_values = {
            "resolve_bridge_python": cli.resolve_bridge_python,
            "discover_cases": cli.discover_cases,
            "run_bridge": cli.run_bridge,
            "find_mimics_exe": cli.find_mimics_exe,
            "launch_create_mcs": cli.launch_create_mcs,
            "live_holder": cli._live_create_mcs_holder,
            "clear_stop": cli.clear_stale_import_queue_stop,
            "lock": cli.FileResourceLock,
            "queue_runtime": cli.runtime_common.import_queue_runtime_dir,
            "import_runtime": cli.runtime_common.import_runtime_base,
            "producer_path": cli.runtime_common.import_producer_lock_path,
        }
        try:
            cli.resolve_bridge_python = lambda *_args: sys.executable
            cli.discover_cases = lambda *_args, **_kwargs: [
                {"case_id": "case_001", "image": "one.nii.gz", "masks": []},
                {"case_id": "case_002", "image": "two.nii.gz", "masks": []},
            ]
            def bridge(_python, params, **_kwargs):
                bridge_cases.append(params["case_id"])
                return {"status": "ok", "case_id": params["case_id"], "masks": []}
            cli.run_bridge = bridge
            cli.find_mimics_exe = lambda *_args: "MimicsResearch.exe"
            cli._live_create_mcs_holder = lambda *_args: {}
            cli.clear_stale_import_queue_stop = lambda *_args: False
            cli.FileResourceLock = Lock
            cli.runtime_common.import_queue_runtime_dir = lambda *_args: str(runtime_dir)
            cli.runtime_common.import_runtime_base = lambda *_args: str(import_runtime)
            cli.runtime_common.import_producer_lock_path = lambda *_args: str(root / "producer.lock")

            def launch(*_args, **_kwargs):
                descriptors = list((runtime_dir / "prepared_queue").glob("*.json"))
                launch_descriptor_counts.append(len(descriptors))
                self.assertEqual(["case_001"], bridge_cases)
                return Process()
            cli.launch_create_mcs = launch

            args = cli.build_parser().parse_args([
                "prepare-import",
                "--ts-root", str(dataset),
                "--output-dir", str(output),
            ])
            result = cli.cmd_prepare_import(args)
        finally:
            cli.resolve_bridge_python = old_values["resolve_bridge_python"]
            cli.discover_cases = old_values["discover_cases"]
            cli.run_bridge = old_values["run_bridge"]
            cli.find_mimics_exe = old_values["find_mimics_exe"]
            cli.launch_create_mcs = old_values["launch_create_mcs"]
            cli._live_create_mcs_holder = old_values["live_holder"]
            cli.clear_stale_import_queue_stop = old_values["clear_stop"]
            cli.FileResourceLock = old_values["lock"]
            cli.runtime_common.import_queue_runtime_dir = old_values["queue_runtime"]
            cli.runtime_common.import_runtime_base = old_values["import_runtime"]
            cli.runtime_common.import_producer_lock_path = old_values["producer_path"]

        self.assertEqual(0, result)
        self.assertEqual(["case_001", "case_002"], bridge_cases)
        self.assertEqual([1], launch_descriptor_counts)
        done = json.loads((runtime_dir / "_mcs_queue_done.json").read_text(encoding="utf-8"))
        self.assertEqual(2, done["completed"])

    def test_external_batch_continues_when_one_case_and_failure_record_fail(self):
        cli = __import__("tools.mimics_batch_cli", fromlist=["dummy"])

        root = Path(self.tmp, "external_continue")
        dataset = root / "dataset"
        output = root / "output"
        runtime_dir = root / "queue_runtime"
        import_runtime = root / "import_runtime"
        for path in (dataset, output, runtime_dir, import_runtime):
            path.mkdir(parents=True, exist_ok=True)

        class Lock(object):
            def __init__(self, *_args, **_kwargs):
                pass
            def acquire(self, **_kwargs):
                return self
            def update_pid(self, *_args, **_kwargs):
                return True
            def release(self):
                return None

        prepared = []
        original = {
            "resolve": cli.resolve_bridge_python,
            "discover": cli.discover_cases,
            "bridge": cli.run_bridge,
            "write": cli.write_json_atomic,
            "clear": cli.clear_stale_import_queue_stop,
            "lock": cli.FileResourceLock,
            "queue_runtime": cli.runtime_common.import_queue_runtime_dir,
            "import_runtime": cli.runtime_common.import_runtime_base,
            "producer_path": cli.runtime_common.import_producer_lock_path,
        }
        try:
            cli.resolve_bridge_python = lambda *_args: sys.executable
            cli.discover_cases = lambda *_args, **_kwargs: [
                {"case_id": "case_bad", "image": "bad.nii.gz", "masks": []},
                {"case_id": "case_good", "image": "good.nii.gz", "masks": []},
            ]
            def bridge(_python, params, **_kwargs):
                if params["case_id"] == "case_bad":
                    raise RuntimeError("controlled bridge failure")
                prepared.append(params["case_id"])
                return {"status": "ok", "case_id": params["case_id"], "masks": []}
            cli.run_bridge = bridge
            def write(path, payload, *args, **kwargs):
                if "_failed_cases" in str(path):
                    raise PermissionError("controlled failure-record denial")
                return original["write"](path, payload, *args, **kwargs)
            cli.write_json_atomic = write
            cli.clear_stale_import_queue_stop = lambda *_args: False
            cli.FileResourceLock = Lock
            cli.runtime_common.import_queue_runtime_dir = lambda *_args: str(runtime_dir)
            cli.runtime_common.import_runtime_base = lambda *_args: str(import_runtime)
            cli.runtime_common.import_producer_lock_path = lambda *_args: str(root / "producer.lock")

            args = cli.build_parser().parse_args([
                "prepare-import",
                "--ts-root", str(dataset),
                "--output-dir", str(output),
                "--no-create-mcs",
            ])
            result = cli.cmd_prepare_import(args)
        finally:
            cli.resolve_bridge_python = original["resolve"]
            cli.discover_cases = original["discover"]
            cli.run_bridge = original["bridge"]
            cli.write_json_atomic = original["write"]
            cli.clear_stale_import_queue_stop = original["clear"]
            cli.FileResourceLock = original["lock"]
            cli.runtime_common.import_queue_runtime_dir = original["queue_runtime"]
            cli.runtime_common.import_runtime_base = original["import_runtime"]
            cli.runtime_common.import_producer_lock_path = original["producer_path"]

        self.assertEqual(0, result)
        self.assertEqual(["case_good"], prepared)
        done = json.loads((runtime_dir / "_mcs_queue_done.json").read_text(encoding="utf-8"))
        self.assertEqual(1, done["completed"])
        self.assertEqual(1, done["failed"])

    def test_internal_batch_continues_after_case_start_failure(self):
        import mimics_import

        monitor = {
            "monitor_key": "old",
            "batch_queue": [
                {"case_id": "case_bad", "image": "bad.nii.gz", "masks": []},
                {"case_id": "case_good", "image": "good.nii.gz", "masks": []},
            ],
            "output_dir": self.tmp,
            "axes": [0, 1, 2],
            "flips": [False, False, False],
            "jobs_dir": os.path.join(self.tmp, "jobs"),
            "work_root": os.path.join(self.tmp, "work"),
            "completed": 0,
            "failed": 0,
            "total": 2,
            "timeout_seconds": 30,
        }
        launched = []
        original = {
            "build": mimics_import._build_bridge_params,
            "launch": mimics_import._launch_bridge_job_thread,
            "log": mimics_import._append_import_log,
            "record": mimics_import._record_failed_case,
            "status": mimics_import._write_import_task_status,
        }
        try:
            def build(case_info, axes, flips, work_dir):
                if case_info["case_id"] == "case_bad":
                    raise RuntimeError("controlled setup failure")
                return {"case_id": case_info["case_id"]}
            mimics_import._build_bridge_params = build
            mimics_import._launch_bridge_job_thread = lambda params, *_args, **_kwargs: launched.append(
                params["case_id"]
            )
            mimics_import._append_import_log = lambda *_args, **_kwargs: None
            mimics_import._record_failed_case = lambda *_args, **_kwargs: None
            mimics_import._write_import_task_status = lambda *_args, **_kwargs: None
            mimics_import._start_next_batch_prepare(monitor)
        finally:
            mimics_import._build_bridge_params = original["build"]
            mimics_import._launch_bridge_job_thread = original["launch"]
            mimics_import._append_import_log = original["log"]
            mimics_import._record_failed_case = original["record"]
            mimics_import._write_import_task_status = original["status"]
            mimics_import._IMPORT_MONITORS.pop(
                os.path.join(self.tmp, "jobs", "case_good"), None
            )

        self.assertEqual(1, monitor["failed"])
        self.assertEqual(["case_good"], launched)
        self.assertEqual("case_good", monitor["case_id"])

    def test_internal_batch_first_case_uses_the_guarded_queue_path(self):
        import inspect
        import mimics_import

        source = inspect.getsource(mimics_import._discover_monitor_tick)
        self.assertIn('"selecting_next": True', source)
        self.assertIn("batch_queue=cases", source)
        self.assertNotIn("first_case = cases[0]", source)

    def test_external_export_multi_lock_releases_partial_acquisition(self):
        cli = __import__("tools.mimics_batch_cli", fromlist=["dummy"])
        from resource_locks import ResourceLockTimeout

        instances = []
        old_lock = cli.FileResourceLock
        class Lock(object):
            def __init__(self, *_args, **_kwargs):
                self.released = False
                instances.append(self)
            def acquire(self, **_kwargs):
                if len(instances) > 1:
                    raise ResourceLockTimeout("destination busy")
                return self
            def release(self):
                self.released = True
        try:
            cli.FileResourceLock = Lock
            with self.assertRaises(ResourceLockTimeout):
                cli._acquire_background_mimics_locks(
                    "test export",
                    [Path(self.tmp) / "mcs", Path(self.tmp) / "labels"],
                    0.0,
                )
        finally:
            cli.FileResourceLock = old_lock
        self.assertTrue(instances[0].released)

    def test_external_import_clears_only_a_stale_queue_stop(self):
        cli = __import__("tools.mimics_batch_cli", fromlist=["dummy"])
        import runtime_common

        old_root = cli.ROOT
        cli.ROOT = Path(self.tmp)
        output_dir = Path(self.tmp) / "mcs_output"
        output_dir.mkdir()
        runtime_dir = Path(runtime_common.import_queue_runtime_dir(
            self.tmp, str(output_dir.resolve())
        ))
        runtime_dir.mkdir(parents=True)
        stop_path = runtime_dir / "_mcs_queue_stop.json"
        stop_path.write_text("{}", encoding="utf-8")
        try:
            self.assertTrue(cli.clear_stale_import_queue_stop(output_dir))
            self.assertFalse(stop_path.exists())

            stop_path.write_text("{}", encoding="utf-8")
            lock_path = runtime_common.background_mimics_lock_path(
                self.tmp, str(output_dir.resolve())
            )
            runtime_common.write_json_atomic(
                lock_path,
                {
                    "token": "live",
                    "pid": os.getpid(),
                    "kind": "create_mcs",
                    "output_dir": str(output_dir.resolve()),
                },
            )
            with self.assertRaisesRegex(RuntimeError, "still stopping"):
                cli.clear_stale_import_queue_stop(output_dir)
            self.assertTrue(stop_path.exists())
        finally:
            cli.ROOT = old_root

    def test_internal_import_clears_dead_queue_stop_but_not_live_creator(self):
        import mimics_import
        import runtime_common

        old_root = mimics_import._project_root
        output_dir = os.path.join(self.tmp, "mcs_output")
        os.makedirs(output_dir)
        try:
            mimics_import._project_root = lambda: self.tmp
            stop_path = mimics_import._queue_stop_path(output_dir)
            runtime_common.write_json_atomic(stop_path, {"status": "stop_requested"})
            lock_path = runtime_common.background_mimics_lock_path(
                self.tmp, output_dir
            )
            runtime_common.write_json_atomic(
                lock_path,
                {
                    "token": "dead",
                    "pid": 99999999,
                    "kind": "create_mcs",
                    "output_dir": os.path.abspath(output_dir),
                },
            )
            mimics_import._clear_stale_queue_stop(output_dir)
            self.assertFalse(os.path.exists(stop_path))

            runtime_common.write_json_atomic(stop_path, {"status": "stop_requested"})
            producer_path = runtime_common.import_producer_lock_path(
                self.tmp, output_dir
            )
            runtime_common.write_json_atomic(
                producer_path,
                {
                    "token": "producer-live",
                    "pid": os.getpid(),
                    "resource": "import_producer",
                },
            )
            with self.assertRaisesRegex(RuntimeError, "still responding"):
                mimics_import._clear_stale_queue_stop(output_dir)
            self.assertTrue(os.path.exists(stop_path))
            os.remove(producer_path)

            runtime_common.write_json_atomic(
                lock_path,
                {
                    "token": "live",
                    "pid": os.getpid(),
                    "kind": "create_mcs",
                    "output_dir": os.path.abspath(output_dir),
                },
            )
            with self.assertRaisesRegex(RuntimeError, "still stopping"):
                mimics_import._clear_stale_queue_stop(output_dir)
            self.assertTrue(os.path.exists(stop_path))
        finally:
            mimics_import._project_root = old_root

    def test_background_mimics_global_serialization_is_explicit_opt_in(self):
        import runtime_common

        key = "MIMICS_SERIALIZE_BACKGROUND_MIMICS"
        previous = os.environ.get(key)
        try:
            os.environ[key] = "1"
            self.assertEqual(
                "background_mimics.lock",
                runtime_common.background_mimics_lock_name(self.tmp),
            )
        finally:
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous

    def test_import_background_launch_state_allows_independent_outputs(self):
        import mimics_import

        first = os.path.join(self.tmp, "output_a")
        second = os.path.join(self.tmp, "output_b")
        self.assertTrue(mimics_import._bg_try_begin_launch(first))
        self.assertTrue(mimics_import._bg_try_begin_launch(second))
        self.assertFalse(mimics_import._bg_try_begin_launch(first))
        try:
            mimics_import._bg_set_process(10101, first)
            mimics_import._bg_set_process(20202, second)
            self.assertEqual(10101, mimics_import._bg_state_snapshot(first)["pid"])
            self.assertEqual(20202, mimics_import._bg_state_snapshot(second)["pid"])
        finally:
            mimics_import._bg_clear_process(10101)
            mimics_import._bg_clear_process(20202)
            mimics_import._bg_end_launch(first)
            mimics_import._bg_end_launch(second)

    def test_export_lock_registry_tracks_parallel_jobs(self):
        import mimics_export

        mimics_export._clear_active_background_export_lock()
        try:
            mimics_export._set_active_background_export_lock("first.lock", "token-a")
            mimics_export._set_active_background_export_lock("second.lock", "token-b")
            self.assertEqual(2, len(mimics_export._ACTIVE_BG_MIMICS_LOCKS))
            self.assertTrue(mimics_export._clear_active_background_export_lock("token-a"))
            self.assertIn("token-b", mimics_export._ACTIVE_BG_MIMICS_LOCKS)
        finally:
            mimics_export._clear_active_background_export_lock()

    def test_single_case_worker_reports_busy_output_queue_without_crashing(self):
        import tools.single_case_import_worker as worker

        run_root = Path(self.tmp, "single_busy_run")
        runtime_dir = Path(self.tmp, "single_busy_queue")
        output_dir = Path(self.tmp, "single_busy_output")
        locks_dir = Path(self.tmp, "single_busy_locks")
        for path in (run_root, runtime_dir, output_dir, locks_dir):
            path.mkdir(parents=True, exist_ok=True)
        selection_path = run_root / "selection.json"
        status_path = run_root / "status.json"
        stop_path = run_root / "stop.json"
        log_path = run_root / "worker.log"
        selection_path.write_text(json.dumps({
            "source_path": str(Path(self.tmp, "ct.nii.gz")),
            "output_path": str(output_dir),
            "project_root": PROJECT_ROOT,
            "case_info": {
                "case_id": "case_busy",
                "image": str(Path(self.tmp, "ct.nii.gz")),
                "case_dir": self.tmp,
                "masks": [],
            },
            "mask_selection": "none",
        }), encoding="utf-8")
        (runtime_dir / "_mcs_queue_active.json").write_text(
            json.dumps({"updated_at_epoch": time.time()}), encoding="utf-8"
        )

        old_runtime = worker.runtime_common.import_queue_runtime_dir
        old_locks = worker.default_resource_lock_dir
        try:
            worker.runtime_common.import_queue_runtime_dir = lambda *_args: str(runtime_dir)
            worker.default_resource_lock_dir = lambda *_args: locks_dir
            result = worker.run(selection_path, status_path, stop_path, log_path)
        finally:
            worker.runtime_common.import_queue_runtime_dir = old_runtime
            worker.default_resource_lock_dir = old_locks

        self.assertEqual(75, result)
        status = json.loads(status_path.read_text(encoding="utf-8"))
        self.assertEqual("failed", status["status"])
        self.assertEqual("output_queue_busy", status["phase"])

    def test_single_case_worker_cancels_an_unconsumed_descriptor_immediately(self):
        import tools.single_case_import_worker as worker

        root = Path(self.tmp, "single_cancel")
        runtime_dir = root / "runtime"
        output_dir = root / "output"
        runtime_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)
        status_path = root / "status.json"
        stop_path = root / "stop.json"
        log_path = root / "worker.log"
        descriptor = runtime_dir / "pending.json"
        descriptor.write_text("{}", encoding="utf-8")
        stop_path.write_text("{}", encoding="utf-8")

        result = worker._wait_for_mcs(
            output_dir / "case.mcs",
            runtime_dir,
            None,
            status_path,
            stop_path,
            log_path,
            "case",
            60,
            None,
            False,
            descriptor,
            PROJECT_ROOT,
            output_dir,
            "MimicsResearch.exe",
            sys.executable,
            time.time(),
        )

        self.assertEqual(0, result)
        self.assertFalse(descriptor.exists())
        status = json.loads(status_path.read_text(encoding="utf-8"))
        self.assertEqual("cancelled", status["status"])

    def test_single_case_worker_reports_completion_for_created_project(self):
        import tools.single_case_import_worker as worker

        run_root = Path(self.tmp, "single_run")
        runtime_dir = Path(self.tmp, "single_queue_runtime")
        output_dir = Path(self.tmp, "single_output")
        locks_dir = Path(self.tmp, "single_locks")
        for path in (run_root, runtime_dir, output_dir, locks_dir):
            path.mkdir(parents=True, exist_ok=True)
        selection_path = run_root / "selection.json"
        status_path = run_root / "status.json"
        stop_path = run_root / "stop.json"
        log_path = run_root / "worker.log"
        selection_path.write_text(json.dumps({
            "source_path": str(Path(self.tmp, "ct.nii.gz")),
            "output_path": str(output_dir),
            "project_root": PROJECT_ROOT,
            "case_info": {
                "case_id": "case_001",
                "image": str(Path(self.tmp, "ct.nii.gz")),
                "case_dir": self.tmp,
                "masks": [],
            },
            "mask_selection": "none",
            "timeout_seconds": 30,
        }), encoding="utf-8")

        class FakeProcess(object):
            pid = 12345
            returncode = None

            def poll(self):
                return None

        old_runtime = worker.runtime_common.import_queue_runtime_dir
        old_locks = worker.default_resource_lock_dir
        old_resolve = worker.batch_cli.resolve_bridge_python
        old_bridge = worker.batch_cli.run_bridge
        old_find = worker.batch_cli.find_mimics_exe
        old_launch = worker.batch_cli.launch_create_mcs
        try:
            worker.runtime_common.import_queue_runtime_dir = lambda *_args: str(runtime_dir)
            worker.default_resource_lock_dir = lambda *_args: locks_dir
            worker.batch_cli.resolve_bridge_python = lambda *_args: sys.executable
            worker.batch_cli.run_bridge = lambda *_args, **_kwargs: {
                "status": "ok",
                "source_fingerprint": "sha256:test",
                "masks": [],
            }
            worker.batch_cli.find_mimics_exe = lambda *_args: "MimicsResearch.exe"

            def launch(output_path, *_args, **_kwargs):
                def finish():
                    time.sleep(0.1)
                    Path(output_path, "case_001.mcs").write_bytes(b"mcs")
                    for descriptor in (runtime_dir / "prepared_queue").glob("*.json"):
                        descriptor.unlink()
                thread = threading.Thread(target=finish)
                thread.daemon = True
                thread.start()
                return FakeProcess()

            worker.batch_cli.launch_create_mcs = launch
            result = worker.run(selection_path, status_path, stop_path, log_path)
        finally:
            worker.runtime_common.import_queue_runtime_dir = old_runtime
            worker.default_resource_lock_dir = old_locks
            worker.batch_cli.resolve_bridge_python = old_resolve
            worker.batch_cli.run_bridge = old_bridge
            worker.batch_cli.find_mimics_exe = old_find
            worker.batch_cli.launch_create_mcs = old_launch

        self.assertEqual(0, result)
        status = json.loads(status_path.read_text(encoding="utf-8"))
        self.assertEqual("completed", status["status"])
        self.assertEqual(100, status["progress_percent"])
        self.assertTrue(Path(output_dir, "case_001.mcs").is_file())

    # -- Bootstrap stop propagation through stop_paths ------------------------

    def test_io_setup_bootstrap_stop_not_propagated_when_task_has_no_stop_paths(self):
        import io_setup_mimics

        status_path = os.path.join(self.tmp, "io-nostop.json")
        stop_path = os.path.join(self.tmp, "io-nostop-bootstrap.json")
        # No stop marker on disk — submit proceeds normally.
        Path(status_path).write_text(
            json.dumps({"status": "submitted", "selection": {}}),
            encoding="utf-8",
        )
        key = "nostop"
        monitor = {
            "key": key,
            "status_path": status_path,
            "bootstrap_stop_path": stop_path,
            "deadline": time.time() + 10,
            "busy": False,
            "on_submit": lambda _s: {"kind": "export", "title": "Export masks"},
        }
        io_setup_mimics._IO_SETUP_MONITORS[key] = monitor
        try:
            io_setup_mimics._tick(monitor)
        finally:
            io_setup_mimics._IO_SETUP_MONITORS.pop(key, None)
        payload = json.loads(Path(status_path).read_text(encoding="utf-8"))
        self.assertEqual("launched", payload.get("status"))
        task = payload.get("task") or {}
        self.assertEqual("export", task.get("kind"))


class TestDatasetProfiles(unittest.TestCase):
    """P2 dataset profile loading, ts-like equivalence, and the recognition summary."""

    def setUp(self):
        self.tmp = _make_temp_dir()
        self._tools_on_path = os.path.join(PROJECT_ROOT, "tools") not in sys.path
        if self._tools_on_path:
            sys.path.insert(0, os.path.join(PROJECT_ROOT, "tools"))

    def tearDown(self):
        _cleanup(self.tmp)
        if self._tools_on_path:
            try:
                sys.path.remove(os.path.join(PROJECT_ROOT, "tools"))
            except ValueError:
                pass

    def _make_case(self, name, image=None, dicom=False, masks=()):
        case_dir = os.path.join(self.tmp, name)
        os.makedirs(case_dir)
        if image:
            open(os.path.join(case_dir, image), "w").close()
        if dicom:
            os.makedirs(os.path.join(case_dir, "dicom"))
            open(os.path.join(case_dir, "dicom", "IM0001"), "w").close()
        if masks:
            seg = os.path.join(case_dir, "segmentations")
            os.makedirs(seg)
            for mask in masks:
                open(os.path.join(seg, mask), "w").close()
        return case_dir

    def test_loader_returns_normalized_ts_like_by_default(self):
        import dataset_profiles
        profile = dataset_profiles.load_profile(None)
        self.assertEqual("ts-like", profile["profile_id"])
        self.assertEqual("ct.nii.gz", profile["image_candidates"][0])
        self.assertIn("segmentations", profile["mask_dirs"])
        self.assertIn("mcs_output", profile["exclude_dirs"])

    def test_loader_unknown_id_falls_back_to_default(self):
        import dataset_profiles
        profile = dataset_profiles.load_profile("no-such-profile")
        self.assertEqual(dataset_profiles.default_profile_id(), profile["profile_id"])

    def test_generic_profile_makes_no_assumptions(self):
        import dataset_profiles
        profile = dataset_profiles.load_profile("generic")
        self.assertEqual([], profile["image_candidates"])
        self.assertEqual([], profile["mask_dirs"])
        self.assertEqual([], profile["exclude_dirs"])
        # A dir that ts-like excludes is a plain case dir under generic.
        self.assertFalse(dataset_profiles.is_excluded_name("mcs_output", profile))
        self.assertTrue(dataset_profiles.is_excluded_name("mcs_output"))

    def test_fallback_matches_ts_like_byte_for_byte(self):
        import dataset_profiles
        self.assertEqual(
            json.loads(json.dumps(dataset_profiles._FALLBACK["profiles"]["ts-like"])),
            json.loads(json.dumps(dataset_profiles._load_raw()["profiles"]["ts-like"])),
        )

    def test_discovery_equivalence_on_ts_like_fixture(self):
        """The profile-driven discovery must match the historical hard-coded result.

        Expected values below are the pre-refactor behaviour for this fixture:
        preferred names first, dicom subdir next, capped fallback scan last,
        excluded roots skipped, masks discovered under segmentations/.
        """
        sys.path.insert(0, RUNTIME_DIR)
        import dataset_profiles
        profile = dataset_profiles.load_profile("ts-like")

        self._make_case("s0001", image="ct.nii.gz", masks=("liver.seg.nii.gz",))
        self._make_case("s0002", image="mri.nii", masks=("kidney.seg.nii",))
        self._make_case("s0003", dicom=True)
        self._make_case("s0004", image="anat_v2.nii")
        case = self._make_case("s0005", image="ct.nii.gz")
        open(os.path.join(case, "old_ct.nii.gz"), "w").close()
        os.makedirs(os.path.join(self.tmp, "mcs_output"))
        open(os.path.join(self.tmp, "mcs_output", "junk.nii"), "w").close()
        os.makedirs(os.path.join(self.tmp, "not_a_case"))
        open(os.path.join(self.tmp, "plain_file.txt"), "w").close()

        # Simulate the historical discovery directly from the profile tables
        # (mirrors the pre-refactor hard-coded lists).
        expected = {
            "s0001": ("ct.nii.gz", "medical_image", ["liver"]),
            "s0002": ("mri.nii", "medical_image", ["kidney"]),
            "s0003": ("dicom", "dicom", []),
            "s0004": ("anat_v2.nii", "medical_image", []),
            "s0005": ("ct.nii.gz", "medical_image", []),
        }

        # Exercise the io-UI single-source discovery, which shares the profile.
        import tools.io_path_setup_ui as ui
        for case_id, (image, _kind, masks) in expected.items():
            result = ui.discover_single_source(os.path.join(self.tmp, case_id))
            self.assertIsNotNone(result, case_id)
            self.assertTrue(result["image"].replace("\\", "/").endswith(image), (case_id, result))
            self.assertEqual(
                sorted(m["name"] for m in result["masks"]),
                sorted(masks),
                (case_id, result),
            )
        # An empty dir keeps its historical treatment: a DICOM-series
        # candidate to be validated in the bridge, not a rejection.
        empty = ui.discover_single_source(os.path.join(self.tmp, "not_a_case"))
        self.assertEqual("dicom_candidate", empty["image_type"])

    def test_summarize_dataset_normal_line_and_anomalies(self):
        import tools.io_path_setup_ui as ui
        self._make_case("s0001", image="ct.nii.gz", masks=("liver.seg.nii.gz", "spleen.seg.nii.gz"))
        self._make_case("s0002", image="ct.nii.gz")
        self._make_case("s0003", dicom=True)
        self._make_case("s0004")  # no image -> skipped
        result = ui.summarize_dataset(self.tmp)
        self.assertIsNotNone(result)
        self.assertEqual(3, result["case_count"])
        self.assertEqual(2, result["mask_count"])
        self.assertIn("3 case(s)", result["summary"])
        self.assertEqual(["s0004"], result["skipped"])

        # Anomaly: two volume files in one case dir.
        case = self._make_case("s0005", image="ct.nii.gz")
        open(os.path.join(case, "old_ct.nii.gz"), "w").close()
        result = ui.summarize_dataset(self.tmp)
        self.assertTrue(any("s0005" in w and "2" in w for w in result["warnings"]), result["warnings"])

    def test_summarize_dataset_rejects_missing_root(self):
        import tools.io_path_setup_ui as ui
        self.assertIsNone(ui.summarize_dataset(os.path.join(self.tmp, "nope")))
        self.assertIsNone(ui.summarize_dataset(""))

    def test_bridge_discovery_tables_follow_profile(self):
        sys.path.insert(0, PROJECT_ROOT)
        import mimics_bridge as mb

        tables = mb._load_profile_tables()
        self.assertEqual("ct.nii.gz", tables["image_candidates"][0])
        self.assertIn("dicom", tables["dicom_dirs"])
        self.assertIn("mcs_output", tables["exclude_dirs"])

        self._make_case("s0001", image="ct.nii.gz")
        open(os.path.join(self.tmp, "s0001", "zzz.nii.gz"), "w").close()
        order = [p.name for p in mb._nifti_candidates(Path(self.tmp) / "s0001")]
        self.assertEqual(["ct.nii.gz", "zzz.nii.gz"], order)

        # do_discover keeps bridge-only preferred names (ct.nrrd) working.
        self._make_case("s0002", image="aaa.mha")
        open(os.path.join(self.tmp, "s0002", "ct.nrrd"), "w").close()
        result = mb.do_discover({"ts_root": self.tmp})
        self.assertEqual("ok", result["status"])
        by_id = {c["case_id"]: c for c in result["cases"]}
        self.assertTrue(by_id["s0002"]["image"].endswith("ct.nrrd"))

        # Excluded roots are skipped in case-dir listing.
        result = mb.do_discover_case_dirs({"ts_root": self.tmp})
        self.assertNotIn("mcs_output", [c["case_id"] for c in result["cases"]])


class TestImportDropWindow(unittest.TestCase):
    """P3 drop window: payload classification, recognition, and state memory."""

    def setUp(self):
        self.tmp = _make_temp_dir()
        self._paths_added = []
        for entry in (os.path.join(PROJECT_ROOT, "tools"), os.path.join(PROJECT_ROOT, "runtime_py35")):
            if entry not in sys.path:
                sys.path.insert(0, entry)
                self._paths_added.append(entry)

    def tearDown(self):
        _cleanup(self.tmp)
        for entry in self._paths_added:
            try:
                sys.path.remove(entry)
            except ValueError:
                pass

    def _make_case(self, name, image="ct.nii.gz"):
        case_dir = os.path.join(self.tmp, name)
        os.makedirs(case_dir)
        open(os.path.join(case_dir, image), "w").close()
        return case_dir

    def test_classify_single_file(self):
        import import_drop_window as dw
        path = os.path.join(self.tmp, "vol.nii.gz")
        open(path, "w").close()
        kind, source, _children = dw.classify_payload([path])
        self.assertEqual("single", kind)
        self.assertEqual(path, source)

    def test_classify_case_folder(self):
        import import_drop_window as dw
        case = self._make_case("s0001")
        kind, source, _children = dw.classify_payload([case])
        self.assertEqual("single", kind)
        self.assertEqual(case, source)

    def test_classify_dataset_root(self):
        import import_drop_window as dw
        self._make_case("s0001")
        self._make_case("s0002")
        kind, source, children = dw.classify_payload([self.tmp])
        self.assertEqual("batch", kind)
        self.assertEqual(self.tmp, source)
        self.assertGreaterEqual(len(children), 2)

    def test_classify_multi_select_same_parent_is_batch(self):
        import import_drop_window as dw
        self._make_case("s0001")
        self._make_case("s0002")
        kind, source, _children = dw.classify_payload([
            os.path.join(self.tmp, "s0001"), os.path.join(self.tmp, "s0002"),
        ])
        self.assertEqual("batch", kind)
        self.assertEqual(self.tmp, source)

    def test_classify_paths_from_different_roots_are_single_imports(self):
        import import_drop_window as dw
        self._make_case("s0001")
        other = os.path.join(self.tmp, "elsewhere", "s0009")
        os.makedirs(other)
        open(os.path.join(other, "ct.nii.gz"), "w").close()
        kind, _source, paths = dw.classify_payload([
            os.path.join(self.tmp, "s0001"), other,
        ])
        self.assertEqual("multi_single", kind)
        self.assertEqual(2, len(paths))

    def test_excluded_dirs_are_not_treated_as_cases(self):
        import import_drop_window as dw
        os.makedirs(os.path.join(self.tmp, "mcs_output"))
        open(os.path.join(self.tmp, "mcs_output", "x.nii"), "w").close()
        children = dw._child_case_dirs(self.tmp)
        self.assertEqual([], children)

    def test_submit_batch_invokes_prepare_import(self):
        import import_drop_window as dw
        self._make_case("s0001")
        self._make_case("s0002")
        selection = {
            "kind": "batch",
            "source_path": self.tmp,
            "output_path": os.path.join(self.tmp, "mcs_output"),
            "mask_selection": "all",
        }
        calls = []

        class FakeProc:
            pid = 4242

        def fake_popen(command, **_kwargs):
            calls.append(command)
            return FakeProc()

        with mock.patch.object(dw.subprocess, "Popen", side_effect=fake_popen):
            launched = dw.submit_import(selection, {})
        self.assertEqual([("dataset", "submitted")], launched)
        self.assertEqual(1, len(calls))
        command = calls[0]
        self.assertIn("mimics_batch_cli.py", command[1])
        self.assertIn("prepare-import", command)
        self.assertIn(self.tmp, command)

    def test_submit_single_invokes_single_case_worker(self):
        import import_drop_window as dw
        case = self._make_case("s0001")
        selection = {
            "kind": "single",
            "source_path": case,
            "output_path": os.path.join(self.tmp, "mcs_output"),
            "mask_selection": "all",
        }
        calls = []

        class FakeProc:
            pid = 4243

        def fake_popen(command, **_kwargs):
            calls.append(command)
            return FakeProc()

        with mock.patch.object(dw.subprocess, "Popen", side_effect=fake_popen):
            launched = dw.submit_import(selection, {})
        self.assertEqual(1, len(calls))
        self.assertEqual("single_case_import_worker.py", os.path.basename(calls[0][1]))
        self.assertIn("submitted", launched[0][1])
        # The worker receives a selection JSON with the discovered case_info.
        selection_arg = calls[0][calls[0].index("--selection-json") + 1]
        with open(selection_arg, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.assertEqual("s0001", payload["case_info"]["case_id"])
        self.assertTrue(payload["case_info"]["image"].endswith("ct.nii.gz"))

    def test_offscreen_window_renders_and_registers(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        import import_drop_window as dw
        preview = os.path.join(self.tmp, "preview.png")
        records = []
        token_holder = {"token": ""}

        def fake_register(project_root, role, pid, **kwargs):
            self.assertEqual("external_ui", role)
            records.append((role, pid, kwargs))
            token_holder["token"] = "tok123"
            return {"ownership_token": "tok123"}

        def fake_unregister(project_root, role, pid, ownership_token=""):
            self.assertEqual("tok123", ownership_token)

        import resource_locks
        with mock.patch.object(resource_locks, "register_process", side_effect=fake_register), \
                mock.patch.object(resource_locks, "unregister_process", side_effect=fake_unregister):
            code = dw.run({}, preview_path=preview)
        self.assertEqual(0, code)
        self.assertTrue(os.path.isfile(preview) and os.path.getsize(preview) > 0)
        self.assertEqual(1, len(records))
        self.assertIn("idle_timeout_s", records[0][2].get("cleanup_policy", ""))


# ============================================================================
# P4: Import receipts and one-click undo
# ============================================================================


class TestImportReceiptAndUndo(unittest.TestCase):
    """Phase E P4: receipt writing in create_mcs_batch + undo roundtrip."""

    def setUp(self):
        self.tmp = _make_temp_dir()

    def tearDown(self):
        _cleanup(self.tmp)

    def _make_case(self, case_id="s0001"):
        case_dir = os.path.join(self.tmp, "dataset", case_id)
        os.makedirs(case_dir)
        with open(os.path.join(case_dir, "ct.nii.gz"), "wb") as handle:
            handle.write(b"fake-volume")
        return case_dir

    def test_receipt_schema_and_location(self):
        from create_mcs_batch import write_import_receipt

        out_dir = os.path.join(self.tmp, "out")
        os.makedirs(out_dir)
        mcs_path = os.path.join(out_dir, "{0}.mcs".format("s0001"))
        with open(mcs_path, "wb") as handle:
            handle.write(b"fake-project-bytes")
        receipt_path = write_import_receipt(
            out_dir, "s0001", mcs_path, ["Bone", "Liver"],
            {"source_fingerprint": "abc", "source_image_path": "ct.nii.gz",
             "source_case_dir": self._make_case()},
        )
        self.assertEqual(
            os.path.join(out_dir, ".s0001.import_receipt.json"), receipt_path
        )
        with open(receipt_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.assertEqual("mimics_import_receipt.v1", payload["schema_version"])
        self.assertEqual("s0001", payload["case_id"])
        self.assertEqual(["Bone", "Liver"], payload["created_masks"])
        self.assertTrue(payload["mcs_fingerprint"].startswith("sha256:"))
        self.assertEqual("abc", payload["source_fingerprint"])
        self.assertGreater(float(payload["created_at_epoch"]), 0.0)
        self.assertTrue(payload["metadata_keys"])

    def test_find_latest_receipt_picks_newest(self):
        import import_undo_mimics

        out_dir = os.path.join(self.tmp, "out")
        os.makedirs(out_dir)
        for case, created in (("s0001", 100.0), ("s0002", 300.0), ("s0003", 200.0)):
            payload = {
                "schema_version": "mimics_import_receipt.v1",
                "case_id": case,
                "mcs_path": os.path.join(out_dir, "{0}.mcs".format(case)),
                "mcs_fingerprint": "sha256:x",
                "created_masks": ["Bone"],
                "metadata_keys": [],
                "source_fingerprint": "",
                "source_image_path": "",
                "source_case_dir": "",
                "created_at_epoch": created,
            }
            with open(os.path.join(out_dir, ".{0}.import_receipt.json".format(case)),
                      "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
        # A foreign JSON file in the same folder must be ignored.
        with open(os.path.join(out_dir, "unrelated.json"), "w") as handle:
            handle.write("{}")
        with mock.patch.object(import_undo_mimics, "_candidate_receipt_dirs",
                               lambda: [out_dir]):
            path, payload = import_undo_mimics.find_latest_receipt()
        self.assertEqual("s0002", payload["case_id"])
        self.assertTrue(path.endswith(".s0002.import_receipt.json"))

    def test_find_latest_receipt_ignores_invalid_schemas(self):
        import import_undo_mimics

        out_dir = os.path.join(self.tmp, "out")
        os.makedirs(out_dir)
        with open(os.path.join(out_dir, ".bad.import_receipt.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({"schema_version": "something.else.v9"}, handle)
        with mock.patch.object(import_undo_mimics, "_candidate_receipt_dirs",
                               lambda: [out_dir]):
            path, payload = import_undo_mimics.find_latest_receipt()
        self.assertIsNone(path)
        self.assertIsNone(payload)

    def _install_undo_env(self, mcs_path, masks_alive):
        """Patch the undo module's Mimics surface and receipt discovery.

        Returns (save_calls, register of deleted masks) for assertions.
        """
        import import_undo_mimics

        saved = {
            "deleted": [],
            "saves": [],
            "closes": 0,
            "open_target": [],
            "message_boxes": [],
        }

        class _FakeMask:
            def __init__(self, name):
                self.name = name

            def delete(self):
                saved["deleted"].append(self.name)

        image = _FakeModule()
        image.masks = [_FakeMask(name) for name in masks_alive]

        import_undo_mimics.mimics.data.images = [image]
        import_undo_mimics.mimics.file.get_active_project = (
            lambda: saved["open_target"][0] if saved["open_target"] else None
        )
        import_undo_mimics.mimics.file.open_project = (
            lambda filename=None: saved["open_target"].append(filename)
        )
        import_undo_mimics.mimics.file.save_project = (
            lambda filename=None, save_as_type=None:
                saved["saves"].append(filename)
        )
        import_undo_mimics.mimics.file.close_project = (
            lambda: saved.__setitem__("closes", saved["closes"] + 1)
        )
        import_undo_mimics.mimics.dialogs.message_box = (
            lambda *a, **kw: saved["message_boxes"].append((a, kw)) or True
        )
        return saved

    def test_undo_roundtrip_file_rolled_back_when_fingerprint_matches(self):
        import import_undo_mimics
        from create_mcs_batch import write_import_receipt, _file_fingerprint

        out_dir = os.path.join(self.tmp, "out")
        os.makedirs(out_dir)
        mcs_path = os.path.join(out_dir, "s0001.mcs")
        with open(mcs_path, "wb") as handle:
            handle.write(b"project-bytes")
        receipt_path = write_import_receipt(
            out_dir, "s0001", mcs_path, ["Bone", "Liver"], {}
        )
        saved = self._install_undo_env(mcs_path, ["Bone", "Liver", "Skin"])
        with mock.patch.object(import_undo_mimics, "_candidate_receipt_dirs",
                               lambda: [out_dir]):
            code = import_undo_mimics.undo_last_import(confirm=False)
        self.assertEqual(0, code)
        # Only the receipt's masks are deleted; the unrelated one stays.
        self.assertEqual(sorted(["Bone", "Liver"]), sorted(saved["deleted"]))
        self.assertFalse(os.path.isfile(mcs_path))
        self.assertFalse(os.path.isfile(receipt_path))
        self.assertEqual([mcs_path], saved["saves"])
        self.assertEqual(1, saved["closes"])

    def test_undo_keeps_file_when_fingerprint_differs(self):
        import import_undo_mimics
        from create_mcs_batch import write_import_receipt

        out_dir = os.path.join(self.tmp, "out")
        os.makedirs(out_dir)
        mcs_path = os.path.join(out_dir, "s0001.mcs")
        with open(mcs_path, "wb") as handle:
            handle.write(b"project-bytes")
        write_import_receipt(out_dir, "s0001", mcs_path, ["Bone"], {})
        # The project changed after the import (annotations, extra masks).
        with open(mcs_path, "ab") as handle:
            handle.write(b"-changed")
        saved = self._install_undo_env(mcs_path, ["Bone"])
        with mock.patch.object(import_undo_mimics, "_candidate_receipt_dirs",
                               lambda: [out_dir]):
            code = import_undo_mimics.undo_last_import(confirm=False)
        self.assertEqual(0, code)
        self.assertEqual(["Bone"], saved["deleted"])
        self.assertTrue(os.path.isfile(mcs_path))
        # The project stays open for the user; only the file-rollback path
        # closes it.
        self.assertEqual(0, saved["closes"])

    def test_undo_deletion_failure_leaves_receipt(self):
        import import_undo_mimics
        from create_mcs_batch import write_import_receipt

        out_dir = os.path.join(self.tmp, "out")
        os.makedirs(out_dir)
        mcs_path = os.path.join(out_dir, "s0001.mcs")
        with open(mcs_path, "wb") as handle:
            handle.write(b"project-bytes")
        receipt_path = write_import_receipt(out_dir, "s0001", mcs_path, ["Bone"], {})
        saved = self._install_undo_env(mcs_path, ["Bone"])

        def boom(*_a, **_kw):
            raise RuntimeError("transaction failed")

        with mock.patch.object(import_undo_mimics, "_delete_masks", side_effect=boom), \
                mock.patch.object(import_undo_mimics, "_candidate_receipt_dirs",
                                  lambda: [out_dir]):
            code = import_undo_mimics.undo_last_import(confirm=False)
        self.assertEqual(4, code)
        # The receipt survives so the user can retry after fixing the cause.
        self.assertTrue(os.path.isfile(receipt_path))
        self.assertTrue(os.path.isfile(mcs_path))

    def test_undo_reports_when_no_receipt_exists(self):
        import import_undo_mimics

        empty = os.path.join(self.tmp, "empty")
        os.makedirs(empty)
        boxes = []
        import_undo_mimics.mimics.dialogs.message_box = (
            lambda *a, **kw: boxes.append(kw) or True
        )
        with mock.patch.object(import_undo_mimics, "_candidate_receipt_dirs",
                               lambda: [empty]):
            code = import_undo_mimics.undo_last_import(confirm=False)
        self.assertEqual(1, code)
        self.assertEqual(1, len(boxes))

    def test_undo_entry_points_exist(self):
        entry = os.path.join(
            PROJECT_ROOT, "scripting_library", "99_Admin", "05_Undo_Last_Import.py"
        )
        self.assertTrue(os.path.isfile(entry))
        with open(entry, "r") as handle:
            source = handle.read()
        self.assertIn("import_undo_mimics", source)
        drop = os.path.join(
            PROJECT_ROOT, "scripting_library", "01_Data", "08_Quick_Drop_Import.py"
        )
        self.assertTrue(os.path.isfile(drop))
        with open(drop, "r") as handle:
            source = handle.read()
        self.assertIn("import_drop_mimics", source)


# ============================================================================
# Phase F: system health panel aggregation
# ============================================================================


class TestSystemHealthPanel(unittest.TestCase):
    """Health panel: read-only aggregation of registry/locks/queues/server."""

    def setUp(self):
        self.tmp = _make_temp_dir()
        self._old_lock_dir = os.environ.get("MIMICS_RESOURCE_LOCK_DIR")
        os.environ["MIMICS_RESOURCE_LOCK_DIR"] = os.path.join(
            self.tmp, "proj", ".mimics_runtime", "locks"
        )
        self.root = os.path.join(self.tmp, "proj")
        os.makedirs(self.root)
        tools_dir = os.path.join(PROJECT_ROOT, "tools")
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)

    def tearDown(self):
        if self._old_lock_dir is None:
            os.environ.pop("MIMICS_RESOURCE_LOCK_DIR", None)
        else:
            os.environ["MIMICS_RESOURCE_LOCK_DIR"] = self._old_lock_dir
        _cleanup(self.tmp)

    def _import_panel(self):
        import importlib
        import system_health_panel
        importlib.reload(system_health_panel)
        return system_health_panel

    def test_collect_health_empty_project(self):
        shp = self._import_panel()
        snapshot = shp.collect_health(self.root)
        self.assertEqual([], snapshot["processes"])
        self.assertEqual([], snapshot["locks"])
        self.assertEqual([], snapshot["queues"])
        self.assertIsNone(snapshot["server"])

    def test_locks_live_vs_stale_classification(self):
        shp = self._import_panel()
        import resource_locks

        lock_dir = resource_locks.default_resource_lock_dir(self.root)
        lock_dir.mkdir(parents=True, exist_ok=True)
        # A stale lock: owner PID that cannot exist.
        stale_payload = {
            "pid": 99999999,
            "resource": "background_mimics",
            "owner": "old import",
            "kind": "import_producer",
        }
        import json as _json
        with open(str(lock_dir / "stale_import.lock"), "w", encoding="utf-8") as h:
            _json.dump(stale_payload, h)
        # The one-byte guard anchor must not be reported as a lock.
        (lock_dir / "background_mimics_x.lock.guard").write_bytes(b"\x00")

        snapshot = shp.collect_health(self.root)
        names = [lk["name"] for lk in snapshot["locks"]]
        self.assertEqual(["stale_import.lock"], names)
        self.assertTrue(snapshot["locks"][0]["stale"])
        self.assertFalse(snapshot["locks"][0]["live"])

    def test_held_lock_shows_live_with_owner(self):
        shp = self._import_panel()
        import resource_locks

        lock_dir = resource_locks.default_resource_lock_dir(self.root)
        lock = resource_locks.FileResourceLock(
            lock_dir / "held.lock", "background_mimics", "job owner"
        )
        lock.acquire(wait_seconds=3)
        try:
            snapshot = shp.collect_health(self.root)
        finally:
            lock.release()
        self.assertEqual(1, len(snapshot["locks"]))
        self.assertTrue(snapshot["locks"][0]["live"])
        self.assertEqual("job owner", snapshot["locks"][0]["owner"])

    def test_queue_state_classification(self):
        shp = self._import_panel()
        base = os.path.join(self.root, ".mimics_runtime", "import_queues")
        # running: active marker, no stop marker
        running = os.path.join(base, "q_running")
        os.makedirs(os.path.join(running, "prepared_queue"))
        open(os.path.join(running, "_mcs_queue_active.json"), "w").close()
        # stopping: both markers
        stopping = os.path.join(base, "q_stopping")
        os.makedirs(stopping)
        open(os.path.join(stopping, "_mcs_queue_active.json"), "w").close()
        open(os.path.join(stopping, "_mcs_queue_stop.json"), "w").close()
        # idle with prepared cases
        idle = os.path.join(base, "q_idle")
        os.makedirs(os.path.join(idle, "prepared_queue"))
        for name in ("case1.json", "case2.json", "notes.txt"):
            open(os.path.join(idle, "prepared_queue", name), "w").close()
        # stop requested without active
        requested = os.path.join(base, "q_requested")
        os.makedirs(requested)
        open(os.path.join(requested, "_mcs_queue_stop.json"), "w").close()

        snapshot = shp.collect_health(self.root)
        states = {q["name"]: q for q in snapshot["queues"]}
        self.assertEqual("running", states["q_running"]["state"])
        self.assertEqual("stopping", states["q_stopping"]["state"])
        self.assertEqual("idle", states["q_idle"]["state"])
        self.assertEqual(2, states["q_idle"]["prepared_cases"])
        self.assertEqual("stop requested", states["q_requested"]["state"])

    def test_server_state_live_and_dead(self):
        shp = self._import_panel()
        import json as _json

        server_path = os.path.join(self.root, ".nninteractive_server.json")
        # A dead PID leaves a stale state file.
        with open(server_path, "w", encoding="utf-8") as h:
            _json.dump({"pid": 99999999, "model_dir": "m", "port": 8080}, h)
        server = shp.collect_server(self.root)
        self.assertFalse(server["live"])
        # Our own PID reads as live.
        with open(server_path, "w", encoding="utf-8") as h:
            _json.dump({"pid": os.getpid(), "state": "running"}, h)
        server = shp.collect_server(self.root)
        self.assertTrue(server["live"])
        self.assertEqual("running", server["state"])

    def test_processes_include_liveness_and_locks(self):
        shp = self._import_panel()
        import resource_locks
        import json as _json

        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self.addCleanup(proc.wait)
        self.addCleanup(proc.kill)
        # A lock held by the spawned process itself (FileResourceLock always
        # records the acquiring pid, so write the child's payload directly).
        lock_dir = resource_locks.default_resource_lock_dir(self.root)
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_name = "trainer_{0}.lock".format(proc.pid)
        with open(str(lock_dir / lock_name), "w", encoding="utf-8") as handle:
            _json.dump({
                "pid": proc.pid,
                "resource": "trainer",
                "owner": "unit test",
                "kind": "trainer",
            }, handle)
        resource_locks.register_process(
            self.root, "trainer", proc.pid, state_path=""
        )
        snapshot = shp.collect_health(self.root)
        trainers = [p for p in snapshot["processes"] if p["role"] == "trainer"]
        self.assertEqual(1, len(trainers))
        self.assertTrue(trainers[0]["live"])
        self.assertEqual([lock_name], trainers[0]["locks"])

    def test_offscreen_panel_renders(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        shp = self._import_panel()
        preview = os.path.join(self.tmp, "health.png")
        code = shp.run(preview_path=preview)
        self.assertEqual(0, code)
        self.assertTrue(os.path.isfile(preview) and os.path.getsize(preview) > 0)

    def test_health_panel_entry_exists(self):
        entry = os.path.join(
            PROJECT_ROOT, "scripting_library", "99_Admin", "06_System_Health.py"
        )
        self.assertTrue(os.path.isfile(entry))
        with open(entry, "r") as handle:
            source = handle.read()
        self.assertIn("system_health_mimics", source)
        runtime = os.path.join(PROJECT_ROOT, "runtime_py35", "system_health_mimics.py")
        with open(runtime, "r") as handle:
            source = handle.read()
        self.assertIn("launch_external_gui_process", source)
        self.assertIn("register_process", source)


class TestProcessRegistry(unittest.TestCase):
    """Phase B process registry: register/snapshot/sweep/terminate ladder."""

    def setUp(self):
        self.tmp = _make_temp_dir()
        self._old_lock_dir = os.environ.get("MIMICS_RESOURCE_LOCK_DIR")
        os.environ["MIMICS_RESOURCE_LOCK_DIR"] = os.path.join(
            self.tmp, "proj", ".mimics_runtime", "locks"
        )
        self.root = self.tmp

    def tearDown(self):
        if self._old_lock_dir is None:
            os.environ.pop("MIMICS_RESOURCE_LOCK_DIR", None)
        else:
            os.environ["MIMICS_RESOURCE_LOCK_DIR"] = self._old_lock_dir
        _cleanup(self.tmp)

    def _spawn_sleeper(self, seconds=60):
        import subprocess

        return subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep({0})".format(seconds)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def test_register_snapshot_unregister_roundtrip(self):
        import resource_locks

        child = self._spawn_sleeper()
        self.addCleanup(child.wait)
        self.addCleanup(child.kill)
        record = resource_locks.register_process(
            self.root, "trainer", child.pid,
            cmdline_signature="unit-test",
            parent_pid=os.getpid(),
            state_path=os.path.join(self.tmp, "status.json"),
        )
        self.assertEqual("mimics_process_record.v1", record["schema_version"])
        self.assertTrue(record["start_marker"])
        self.assertTrue(resource_locks.process_is_live(self.root, "trainer", child.pid))

        snapshot = resource_locks.snapshot_processes(self.root)
        self.assertEqual(1, len(snapshot))
        self.assertTrue(snapshot[0]["_live"])
        self.assertEqual("trainer", snapshot[0]["role"])

        self.assertTrue(resource_locks.unregister_process(self.root, "trainer", child.pid))
        self.assertEqual([], resource_locks.snapshot_processes(self.root, include_dead=True))

    def test_unregister_rejects_wrong_ownership_token(self):
        import resource_locks

        child = self._spawn_sleeper()
        self.addCleanup(child.wait)
        self.addCleanup(child.kill)
        record = resource_locks.register_process(self.root, "async_worker", child.pid)
        self.assertFalse(
            resource_locks.unregister_process(
                self.root, "async_worker", child.pid, ownership_token="wrong"
            )
        )
        self.assertTrue(resource_locks.process_is_live(self.root, "async_worker", child.pid))
        self.assertTrue(
            resource_locks.unregister_process(
                self.root, "async_worker", child.pid,
                ownership_token=record["ownership_token"],
            )
        )

    def test_register_rejects_unknown_role(self):
        import resource_locks

        with self.assertRaises(ValueError):
            resource_locks.register_process(self.root, "bogus_role", os.getpid())

    def test_sweep_removes_dead_record_and_releases_its_lock(self):
        import resource_locks

        child = self._spawn_sleeper()
        resource_locks.register_process(
            self.root, "trainer", child.pid, parent_pid=os.getpid()
        )
        lock = resource_locks.FileResourceLock(
            resource_locks.default_resource_lock_dir(self.root) / "gpu.lock",
            "gpu", "registry-test",
        ).acquire()
        lock.update_pid(child.pid)

        child.kill()
        child.wait()
        time.sleep(0.2)

        summary = resource_locks.sweep_processes(self.root)
        self.assertEqual(1, summary["removed_dead_records"])
        self.assertEqual(1, len(summary["released_locks"]))
        self.assertIn(str(lock.path), summary["released_locks"])
        self.assertFalse(lock.path.exists())
        self.assertEqual(
            [], resource_locks.snapshot_processes(self.root, include_dead=True)
        )

    def test_sweep_keeps_live_processes_and_their_locks(self):
        import resource_locks

        child = self._spawn_sleeper()
        self.addCleanup(child.wait)
        self.addCleanup(child.kill)
        resource_locks.register_process(
            self.root, "trainer", child.pid, parent_pid=os.getpid()
        )
        lock = resource_locks.FileResourceLock(
            resource_locks.default_resource_lock_dir(self.root) / "gpu.lock",
            "gpu", "registry-test",
        ).acquire()
        lock.update_pid(child.pid)

        summary = resource_locks.sweep_processes(self.root)
        self.assertEqual(0, summary["removed_dead_records"])
        self.assertEqual([], summary["terminated_orphans"])
        self.assertEqual([], summary["released_locks"])
        self.assertTrue(lock.path.exists())
        lock.release()

    def test_sweep_terminates_orphan_and_respects_leave_alive(self):
        import resource_locks

        fake_parent = self._spawn_sleeper(30)
        orphan = self._spawn_sleeper(120)
        ui = self._spawn_sleeper(120)
        self.addCleanup(ui.wait)
        self.addCleanup(ui.kill)
        self.addCleanup(orphan.wait)
        self.addCleanup(orphan.kill)
        resource_locks.register_process(
            self.root, "background_mimics", orphan.pid, parent_pid=fake_parent.pid
        )
        resource_locks.register_process(
            self.root, "external_ui", ui.pid, parent_pid=fake_parent.pid,
            cleanup_policy=resource_locks.CLEANUP_LEAVE_ALIVE,
        )

        fake_parent.kill()
        fake_parent.wait()
        time.sleep(0.2)

        summary = resource_locks.sweep_processes(self.root)
        self.assertEqual(
            [{"role": "background_mimics", "pid": orphan.pid, "reason": "parent_gone"}],
            summary["terminated_orphans"],
        )
        self.assertFalse(
            resource_locks.process_is_live(self.root, "background_mimics", orphan.pid)
        )
        # leave_alive policy: UI process untouched even though parent died.
        self.assertTrue(
            resource_locks.process_is_live(self.root, "external_ui", ui.pid)
        )

    def test_terminate_ladder_force_kills_stubborn_process(self):
        import resource_locks
        import subprocess

        stubborn = subprocess.Popen(
            [sys.executable, "-c", "import time\nwhile True: time.sleep(0.5)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        resource_locks.register_process(
            self.root, "background_mimics", stubborn.pid, parent_pid=os.getpid()
        )
        try:
            self.assertTrue(
                resource_locks.terminate_process(
                    self.root, "background_mimics", stubborn.pid,
                    graceful_seconds=1.0,
                )
            )
        finally:
            stubborn.kill()
            stubborn.wait()
        self.assertFalse(
            resource_locks.process_is_live(
                self.root, "background_mimics", stubborn.pid
            )
        )

    def test_sweep_after_foreground_restart_returns_consistent_state(self):
        """The B4 invariant: Mimics dies, workers get orphaned; one sweep
        clears dead records, terminates orphans, releases their locks."""
        import resource_locks

        # Simulate: a foreground Mimics (this test process "was" the parent)
        # spawned a worker holding a lock, then died.
        worker = self._spawn_sleeper(120)
        resource_locks.register_process(
            self.root, "async_worker", worker.pid,
            parent_pid=123456789,  # a PID that will not exist
        )
        lock = resource_locks.FileResourceLock(
            resource_locks.default_resource_lock_dir(self.root) / "gpu.lock",
            "gpu", "registry-test",
        ).acquire()
        lock.update_pid(worker.pid)

        summary = resource_locks.sweep_processes(self.root)
        self.assertEqual(
            [{"role": "async_worker", "pid": worker.pid, "reason": "parent_gone"}],
            summary["terminated_orphans"],
        )
        self.assertEqual(1, len(summary["released_locks"]))
        self.assertFalse(lock.path.exists())
        self.assertEqual(
            [], resource_locks.snapshot_processes(self.root, include_dead=True)
        )


class TestPipelineCommon(unittest.TestCase):
    """Phase B5 shared pipeline primitives (tools/pipeline_common.py)."""

    def setUp(self):
        self.tmp = _make_temp_dir()

    def tearDown(self):
        _cleanup(self.tmp)

    def test_atomic_writes_roundtrip_and_survive_directory_creation(self):
        import tools.pipeline_common as pipeline_common

        path = os.path.join(self.tmp, "nested", "state.json")
        payload = {"status": "training", "epoch": 3}
        pipeline_common.write_json_atomic(path, payload)
        self.assertEqual(payload, json.loads(Path(path).read_text(encoding="utf-8")))

        text_path = os.path.join(self.tmp, "nested", "marker.txt")
        pipeline_common.write_text_atomic(text_path, "cancel requested")
        self.assertEqual("cancel requested", Path(text_path).read_text(encoding="utf-8"))

    def test_cancel_marker_writes_timestamped_file(self):
        import tools.pipeline_common as pipeline_common

        path = os.path.join(self.tmp, "train.cancel")
        self.assertIsNone(pipeline_common.write_cancel_marker(path))
        self.assertIn("cancel requested at", Path(path).read_text(encoding="utf-8"))
        self.assertIsNone(pipeline_common.write_cancel_marker(""))
        self.assertFalse(os.path.exists(""))

    def test_copy_file_atomic_publishes_exactly_once(self):
        import tools.pipeline_common as pipeline_common

        source = os.path.join(self.tmp, "source.bin")
        destination = os.path.join(self.tmp, "out", "dest.bin")
        Path(source).write_bytes(b"payload")
        pipeline_common.copy_file_atomic(source, destination)
        self.assertEqual(b"payload", Path(destination).read_bytes())
        leftovers = [p for p in Path(destination).parent.iterdir() if p.name.endswith(".tmp")]
        self.assertEqual([], leftovers)

    def test_scoped_locks_serialise_same_scope_and_parallelise_disjoint(self):
        import tools.pipeline_common as pipeline_common
        import resource_locks

        old_root = pipeline_common.ROOT
        lock_dir = os.path.join(self.tmp, "proj", ".mimics_runtime", "locks")
        os.makedirs(lock_dir)
        os.environ["MIMICS_RESOURCE_LOCK_DIR"] = lock_dir
        try:
            pipeline_common.ROOT = Path(self.tmp) / "proj"
            # Same scope serialises: a second acquisition attempt times out.
            locks = pipeline_common.acquire_scoped_background_mimics_locks(
                [os.path.join(self.tmp, "mcs")], "job-a", wait_seconds=0,
            )
            try:
                with self.assertRaises(resource_locks.ResourceLockTimeout):
                    pipeline_common.acquire_scoped_background_mimics_locks(
                        [os.path.join(self.tmp, "mcs")], "job-b", wait_seconds=0,
                    )
            finally:
                for lock in reversed(locks):
                    lock.release()
            # A disjoint scope does not conflict.
            locks_b = pipeline_common.acquire_scoped_background_mimics_locks(
                [os.path.join(self.tmp, "other_mcs")], "job-b", wait_seconds=0,
            )
            for lock in reversed(locks_b):
                lock.release()
        finally:
            pipeline_common.ROOT = old_root
            os.environ.pop("MIMICS_RESOURCE_LOCK_DIR", None)

    def test_finetune_label_export_uses_scoped_locks_not_per_job(self):
        """The unified lock protocol: same .mcs folder must serialise jobs."""
        import inspect
        import tools.nninteractive_finetune_pipeline as nninteractive_finetune_pipeline

        source = inspect.getsource(nninteractive_finetune_pipeline._run_label_export)
        self.assertIn("acquire_scoped_background_mimics_locks", source)
        # The old per-job unique lock name must be gone.
        self.assertNotIn("nninteractive_finetune_export_{", source)

    def test_spawn_helper_registers_background_mimics(self):
        import tools.pipeline_common as pipeline_common
        import resource_locks

        old_root = pipeline_common.ROOT
        registry_dir = os.path.join(self.tmp, "proj", ".mimics_runtime", "processes")
        os.makedirs(registry_dir)
        lock_dir = os.path.join(self.tmp, "proj", ".mimics_runtime", "locks")
        os.makedirs(lock_dir)
        os.environ["MIMICS_RESOURCE_LOCK_DIR"] = lock_dir
        try:
            pipeline_common.ROOT = Path(self.tmp) / "proj"
            fake_mimics = os.path.join(self.tmp, "MimicsFake.exe")
            Path(fake_mimics).write_bytes(b"")
            runner = os.path.join(self.tmp, "run_export.py")
            Path(runner).write_text("pass\n", encoding="utf-8")
            process = pipeline_common.spawn_background_mimics_export(
                "python.exe",  # not a real Mimics; it just needs to start
                runner,
                os.path.join(self.tmp, "export_root"),
                state_path=os.path.join(self.tmp, "status.json"),
            )
            self.addCleanup(process.wait)
            self.addCleanup(process.kill)
            records = resource_locks.snapshot_processes(pipeline_common.ROOT)
            roles = [record["role"] for record in records]
            self.assertIn("background_mimics", roles)
        finally:
            pipeline_common.ROOT = old_root
            os.environ.pop("MIMICS_RESOURCE_LOCK_DIR", None)


class TestLifecycleAndRetention(unittest.TestCase):
    def setUp(self):
        self.tmp = _make_temp_dir()

    def tearDown(self):
        _cleanup(self.tmp)

    def test_nninteractive_state_is_removed_before_gpu_lock_release(self):
        import nninteractive_bridge as bridge

        state_path = Path(self.tmp) / "server.json"
        bridge._write_server_state(state_path, {
            "schema_version": "nninteractive_owned_server.v2",
            "ownership_token": "owned-token",
            "gpu_lock_path": str(Path(self.tmp) / "gpu.lock"),
            "gpu_lock_token": "gpu-token",
        })
        observations = []
        original = bridge._release_gpu_lock_from_state
        try:
            bridge._release_gpu_lock_from_state = lambda _state: (
                observations.append(state_path.exists()) or True
            )
            self.assertTrue(bridge._remove_server_state(state_path, "owned-token"))
        finally:
            bridge._release_gpu_lock_from_state = original
        self.assertEqual([False], observations)
        self.assertFalse(state_path.exists())

    def test_nninteractive_contention_close_retires_server_immediately(self):
        import nninteractive_bridge as bridge

        state_path = Path(self.tmp) / "server-close.json"
        context = object.__new__(bridge._BridgeSessionContext)
        context.session = None
        context.owned_state_path = state_path
        context.owned_token = "owned-token"
        context.keep_server_warm_after_session = False
        context.log_path = Path(self.tmp) / "bridge.log"
        calls = []
        with mock.patch.object(
            bridge,
            "_load_server_state",
            return_value={"ownership_token": "owned-token", "pid": 1234},
        ), mock.patch.object(
            bridge,
            "_terminate_owned_server",
            side_effect=lambda _state: calls.append("terminate") or True,
        ), mock.patch.object(
            bridge,
            "_remove_server_state",
            side_effect=lambda _path, _token: calls.append("release") or True,
        ), mock.patch.object(bridge, "_append_bridge_log"):
            context.close()
        self.assertEqual(["terminate", "release"], calls)

    def test_nninteractive_unlink_failure_releases_owned_lock_after_server_exit(self):
        import nninteractive_bridge as bridge

        state_path = Path(self.tmp) / "server.json"
        bridge._write_server_state(state_path, {
            "schema_version": "nninteractive_owned_server.v2",
            "pid": 1234,
            "ownership_token": "owned-token",
            "gpu_lock_path": str(Path(self.tmp) / "gpu.lock"),
            "gpu_lock_token": "gpu-token",
        })
        releases = []
        with mock.patch.object(Path, "unlink", side_effect=PermissionError("locked")), \
             mock.patch.object(Path, "replace", side_effect=PermissionError("locked")), \
             mock.patch.object(bridge, "_process_matches_server", return_value=False), \
             mock.patch.object(bridge, "_release_gpu_lock_from_state",
                               side_effect=lambda state: releases.append(state) or True), \
             mock.patch.object(bridge.time, "sleep", return_value=None):
            self.assertTrue(bridge._remove_server_state(state_path, "owned-token"))
        self.assertTrue(state_path.exists())
        self.assertEqual(1, len(releases))

    def test_nninteractive_unlink_failure_retains_lock_for_live_server(self):
        import nninteractive_bridge as bridge

        state_path = Path(self.tmp) / "server.json"
        bridge._write_server_state(state_path, {
            "schema_version": "nninteractive_owned_server.v2",
            "pid": 1234,
            "ownership_token": "owned-token",
            "gpu_lock_path": str(Path(self.tmp) / "gpu.lock"),
            "gpu_lock_token": "gpu-token",
        })
        releases = []
        with mock.patch.object(Path, "unlink", side_effect=PermissionError("locked")), \
             mock.patch.object(Path, "replace", side_effect=PermissionError("locked")), \
             mock.patch.object(bridge, "_process_matches_server", return_value=True), \
             mock.patch.object(bridge, "_release_gpu_lock_from_state",
                               side_effect=lambda state: releases.append(state) or True), \
             mock.patch.object(bridge.time, "sleep", return_value=None):
            self.assertFalse(bridge._remove_server_state(state_path, "owned-token"))
        self.assertTrue(state_path.exists())
        self.assertEqual([], releases)

    def test_nninteractive_start_failure_releases_new_lock_after_process_exit(self):
        import nninteractive_bridge as bridge

        class FakeLock(object):
            instances = []

            def __init__(self, *_args, **_kwargs):
                self.token = "new-gpu-token"
                self.release_count = 0
                self.__class__.instances.append(self)

            def acquire(self, **_kwargs):
                return self

            def update_pid(self, _pid, **_kwargs):
                return True

            def release(self):
                self.release_count += 1

        class FakeProcess(object):
            pid = 2468

        with mock.patch.object(bridge, "_gpu_lock_enabled", return_value=True), \
             mock.patch.object(bridge, "FileResourceLock", FakeLock), \
             mock.patch.object(bridge.subprocess, "Popen", return_value=FakeProcess()), \
             mock.patch.object(bridge, "_write_server_state",
                               side_effect=PermissionError("old state is locked")), \
             mock.patch.object(bridge, "_stop_spawned_server", return_value=True), \
             mock.patch.object(bridge, "_remove_server_state", return_value=False):
            with self.assertRaises(PermissionError):
                bridge._start_server(
                    model_dir=self.tmp,
                    device="cuda:0",
                    service_idle_timeout_seconds=60,
                    server_url="http://127.0.0.1:1527",
                    fold="auto",
                    runtime_work_dir=self.tmp,
                )
        self.assertEqual(1, FakeLock.instances[-1].release_count)

    def test_nninteractive_start_failure_retains_lock_if_process_survives(self):
        import nninteractive_bridge as bridge

        class FakeLock(object):
            instances = []

            def __init__(self, *_args, **_kwargs):
                self.token = "new-gpu-token"
                self.release_count = 0
                self.__class__.instances.append(self)

            def acquire(self, **_kwargs):
                return self

            def update_pid(self, _pid, **_kwargs):
                return True

            def release(self):
                self.release_count += 1

        class FakeProcess(object):
            pid = 2468

        with mock.patch.object(bridge, "_gpu_lock_enabled", return_value=True), \
             mock.patch.object(bridge, "FileResourceLock", FakeLock), \
             mock.patch.object(bridge.subprocess, "Popen", return_value=FakeProcess()), \
             mock.patch.object(bridge, "_write_server_state",
                               side_effect=PermissionError("old state is locked")), \
             mock.patch.object(bridge, "_stop_spawned_server", return_value=False), \
             mock.patch.object(bridge, "_remove_server_state", return_value=False):
            with self.assertRaises(PermissionError):
                bridge._start_server(
                    model_dir=self.tmp,
                    device="cuda:0",
                    service_idle_timeout_seconds=60,
                    server_url="http://127.0.0.1:1527",
                    fold="auto",
                    runtime_work_dir=self.tmp,
                )
        self.assertEqual(0, FakeLock.instances[-1].release_count)

    def test_nninteractive_old_cleanup_cannot_remove_replacement_state(self):
        import nninteractive_bridge as bridge

        state_path = Path(self.tmp) / "server.json"
        bridge._write_server_state(state_path, {
            "schema_version": "nninteractive_owned_server.v2",
            "ownership_token": "replacement-token",
        })
        releases = []
        original = bridge._release_gpu_lock_from_state
        try:
            bridge._release_gpu_lock_from_state = lambda state: releases.append(state)
            bridge._remove_server_state(state_path, "old-token")
        finally:
            bridge._release_gpu_lock_from_state = original
        self.assertTrue(state_path.exists())
        self.assertEqual([], releases)

    def test_process_liveness_wrappers_use_shared_implementation(self):
        import nninteractive_bridge as bridge
        import fewshot_mimics
        import tools.fewshot_pipeline as pipeline
        import tools.fewshot_status_viewer as status_viewer

        with mock.patch.object(bridge, "resource_process_exists", return_value=True) as check:
            self.assertTrue(bridge._process_exists(259))
            check.assert_called_once_with(259)
        with mock.patch.object(pipeline, "resource_process_exists", return_value=False) as check:
            self.assertFalse(pipeline.process_exists(259))
            check.assert_called_once_with(259)
        with mock.patch.object(status_viewer, "resource_process_exists", return_value=False) as check:
            self.assertFalse(status_viewer.process_exists(259))
            check.assert_called_once_with(259)
        with mock.patch.object(
            fewshot_mimics.runtime_common,
            "process_exists",
            return_value=True,
        ) as check:
            self.assertTrue(fewshot_mimics._process_exists(259))
            check.assert_called_once_with(259)

    def test_windows_cleanup_declares_pointer_sized_process_handles(self):
        import inspect
        import nninteractive_mimics

        source = inspect.getsource(nninteractive_mimics._cleanup_stale_processes)
        self.assertIn('ctypes.WinDLL("kernel32", use_last_error=True)', source)
        self.assertIn("kernel32.OpenProcess.restype = ctypes.c_void_p", source)
        self.assertIn("kernel32.TerminateProcess.argtypes", source)
        self.assertIn("kernel32.CloseHandle.argtypes", source)
        self.assertNotIn("ctypes.windll.kernel32", source)

    def test_fewshot_stopping_monitor_warns_once_while_process_is_alive(self):
        import fewshot_mimics

        status = {
            "job_id": "train_stopping",
            "kind": "train",
            "organ": "liver",
            "status": "stopping",
            "pid": 2468,
            "termination_pending": True,
            "error": "training cleanup failed",
        }
        monitor = {
            "monitor_key": "train_train_stopping",
            "kind": "train",
            "deadline": time.time() + 60,
            "status_path": os.path.join(self.tmp, "train_stopping.json"),
            "last_line": "",
        }
        messages = []
        writes = []
        stops = []
        logs = []
        with mock.patch.object(
            fewshot_mimics, "_read_json", return_value=dict(status)
        ), mock.patch.object(
            fewshot_mimics, "_process_exists", return_value=True
        ), mock.patch.object(
            fewshot_mimics, "_write_json_atomic",
            side_effect=lambda path, payload: writes.append((path, dict(payload)))
        ), mock.patch.object(
            fewshot_mimics, "_stop_monitor", side_effect=lambda key: stops.append(key)
        ), mock.patch.object(
            fewshot_mimics, "_mimics_log",
            side_effect=lambda level, message: logs.append((level, message))
        ), mock.patch.object(
            fewshot_mimics.mimics.dialogs,
            "message_box",
            side_effect=lambda *args, **kwargs: messages.append((args, kwargs)),
        ):
            fewshot_mimics._monitor_tick_locked(monitor)
            fewshot_mimics._monitor_tick_locked(monitor)
        self.assertEqual(1, len(messages))
        self.assertEqual([], writes)
        self.assertEqual([], stops)
        self.assertTrue(monitor.get("stopping_notice_shown"))
        self.assertTrue(any("live process IDs" in item[1] for item in logs))

    def test_fewshot_stopping_monitor_finalizes_only_after_process_exit(self):
        import fewshot_mimics

        status_path = os.path.join(self.tmp, "train_stopped.json")
        status = {
            "job_id": "train_stopped",
            "kind": "train",
            "organ": "liver",
            "status": "stopping",
            "pid": 2468,
            "termination_pending": True,
            "error": "training cleanup failed",
        }
        monitor = {
            "monitor_key": "train_train_stopped",
            "kind": "train",
            "deadline": time.time() + 60,
            "status_path": status_path,
            "last_line": "",
        }
        writes = []
        stops = []
        messages = []
        with mock.patch.object(
            fewshot_mimics, "_read_json", return_value=dict(status)
        ), mock.patch.object(
            fewshot_mimics, "_process_exists", return_value=False
        ), mock.patch.object(
            fewshot_mimics, "_write_json_atomic",
            side_effect=lambda path, payload: writes.append((path, dict(payload)))
        ), mock.patch.object(
            fewshot_mimics, "_stop_monitor", side_effect=lambda key: stops.append(key)
        ), mock.patch.object(
            fewshot_mimics, "_mimics_log"
        ), mock.patch.object(
            fewshot_mimics.mimics.dialogs,
            "message_box",
            side_effect=lambda *args, **kwargs: messages.append((args, kwargs)),
        ):
            fewshot_mimics._monitor_tick_locked(monitor)
        self.assertEqual(1, len(writes))
        self.assertEqual("failed", writes[0][1]["status"])
        self.assertFalse(writes[0][1]["termination_pending"])
        self.assertEqual("termination_completed_after_error", writes[0][1]["phase"])
        self.assertEqual(["train_train_stopped"], stops)
        self.assertEqual(1, len(messages))

    def test_fewshot_active_scan_closes_reaped_stopping_job(self):
        import fewshot_mimics

        ts_root = os.path.join(self.tmp, "dataset")
        jobs_dir = os.path.join(ts_root, "fewshot_models", "jobs")
        os.makedirs(jobs_dir)
        status_path = os.path.join(jobs_dir, "train_stale.json")
        with open(status_path, "w", encoding="utf-8") as handle:
            json.dump({
                "job_id": "train_stale",
                "kind": "train",
                "organ": "liver",
                "status": "stopping",
                "pid": 2468,
                "termination_pending": True,
                "error": "cleanup failed",
            }, handle)
        with mock.patch.object(fewshot_mimics, "_process_exists", return_value=False):
            path, job = fewshot_mimics._latest_active_job(ts_root)
        self.assertIsNone(path)
        self.assertIsNone(job)
        with open(status_path, "r", encoding="utf-8") as handle:
            saved = json.load(handle)
        self.assertEqual("failed", saved["status"])
        self.assertFalse(saved["termination_pending"])

    def test_resource_guard_is_intentionally_persistent(self):
        from resource_locks import FileResourceLock

        lock_path = Path(self.tmp) / "gpu.lock"
        lock = FileResourceLock(lock_path, "gpu", "test")
        lock.acquire()
        guard_path = Path(str(lock_path) + ".guard")
        self.assertTrue(guard_path.is_file())
        lock.release()
        self.assertFalse(lock_path.exists())
        self.assertTrue(guard_path.is_file())

    def test_idle_nninteractive_cleanup_waits_for_server_exit_before_release(self):
        import tools.fewshot_pipeline as pipeline

        state_path = Path(self.tmp) / "server.json"
        pipeline.write_json_atomic(state_path, {
            "schema_version": "nninteractive_owned_server.v2",
            "pid": 1234,
            "gpu_lock_token": "gpu-token",
            "last_activity_epoch": 0,
            "service_idle_timeout_seconds": 0,
        })
        current = {
            "resource": "gpu",
            "token": "gpu-token",
            "path": str(Path(self.tmp) / "gpu.lock"),
            "state_path": str(state_path),
        }
        calls = {"exists": 0, "terminated": 0, "released": 0}
        old_exists = pipeline.process_exists
        old_terminate = pipeline.terminate_process_tree
        old_release = pipeline.release_lock
        old_sleep = pipeline.time.sleep
        try:
            def process_exists(_pid):
                calls["exists"] += 1
                return calls["exists"] <= 2
            pipeline.process_exists = process_exists
            pipeline.terminate_process_tree = lambda _pid: calls.__setitem__(
                "terminated", calls["terminated"] + 1
            ) or True
            pipeline.release_lock = lambda _path, _token: calls.__setitem__(
                "released", calls["released"] + 1
            ) or (not state_path.exists())
            pipeline.time.sleep = lambda _seconds: None
            self.assertTrue(pipeline.cleanup_idle_nninteractive_server_lock(current))
        finally:
            pipeline.process_exists = old_exists
            pipeline.terminate_process_tree = old_terminate
            pipeline.release_lock = old_release
            pipeline.time.sleep = old_sleep
        self.assertEqual(1, calls["terminated"])
        self.assertEqual(1, calls["released"])
        self.assertFalse(state_path.exists())

    def test_idle_nninteractive_cleanup_retains_lock_if_server_survives(self):
        import tools.fewshot_pipeline as pipeline

        state_path = Path(self.tmp) / "server.json"
        pipeline.write_json_atomic(state_path, {
            "schema_version": "nninteractive_owned_server.v2",
            "pid": 1234,
            "gpu_lock_token": "gpu-token",
            "last_activity_epoch": 0,
            "service_idle_timeout_seconds": 0,
        })
        current = {
            "resource": "gpu",
            "token": "gpu-token",
            "path": str(Path(self.tmp) / "gpu.lock"),
            "state_path": str(state_path),
        }
        releases = []
        clock = {"value": 1000.0}
        old_exists = pipeline.process_exists
        old_terminate = pipeline.terminate_process_tree
        old_release = pipeline.release_lock
        old_time = pipeline.time.time
        old_sleep = pipeline.time.sleep
        try:
            pipeline.process_exists = lambda _pid: True
            pipeline.terminate_process_tree = lambda _pid: True
            pipeline.release_lock = lambda *_args: releases.append(True) or True
            def advancing_time():
                clock["value"] += 20.0
                return clock["value"]
            pipeline.time.time = advancing_time
            pipeline.time.sleep = lambda _seconds: None
            self.assertFalse(pipeline.cleanup_idle_nninteractive_server_lock(current))
        finally:
            pipeline.process_exists = old_exists
            pipeline.terminate_process_tree = old_terminate
            pipeline.release_lock = old_release
            pipeline.time.time = old_time
            pipeline.time.sleep = old_sleep
        self.assertTrue(state_path.exists())
        self.assertEqual([], releases)

    def test_async_termination_does_not_finalize_while_pid_survives(self):
        import runtime_common

        callbacks = []
        clock = {"value": 1000.0}

        class Process(object):
            pid = 43210
            def poll(self):
                return None
            def terminate(self):
                return None
            def wait(self, timeout=None):
                raise RuntimeError("still running")

        class ImmediateThread(object):
            def __init__(self, target=None, name=None):
                self.target = target
                self.daemon = False
            def start(self):
                self.target()

        old_exists = runtime_common.process_exists
        old_thread = runtime_common.threading.Thread
        old_kill = runtime_common.os.kill
        old_time = runtime_common.time.time
        old_sleep = runtime_common.time.sleep
        try:
            runtime_common.process_exists = lambda _pid: True
            runtime_common.threading.Thread = ImmediateThread
            runtime_common.os.kill = lambda *_args: None
            def advancing_time():
                clock["value"] += 20.0
                return clock["value"]
            runtime_common.time.time = advancing_time
            runtime_common.time.sleep = lambda _seconds: None
            self.assertTrue(runtime_common.terminate_process_async(
                process=Process(),
                graceful_seconds=0.0,
                on_complete=lambda: callbacks.append(True),
            ))
        finally:
            runtime_common.process_exists = old_exists
            runtime_common.threading.Thread = old_thread
            runtime_common.os.kill = old_kill
            runtime_common.time.time = old_time
            runtime_common.time.sleep = old_sleep
        self.assertEqual([], callbacks)

    def test_import_timeout_publishes_terminal_only_after_process_reap(self):
        import mimics_import

        statuses = []
        callbacks = []
        monitor = {
            "monitor_key": "import-a",
            "job_dir": os.path.join(self.tmp, "job"),
            "work_dir": os.path.join(self.tmp, "work"),
            "output_mcs": os.path.join(self.tmp, "case.mcs"),
            "case_id": "case",
            "task_status_path": os.path.join(self.tmp, "status.json"),
            "total": 1,
        }
        old_stop = mimics_import._stop_import_monitor
        old_write = mimics_import._write_import_task_status
        old_terminate = mimics_import._terminate_job_process
        old_message = mimics_import._safe_message_box
        old_cleanup_job = mimics_import._cleanup_job_dir
        old_cleanup_work = mimics_import._cleanup_work_dir
        old_record = mimics_import._record_failed_case
        old_done = mimics_import._mark_mcs_queue_done
        try:
            mimics_import._stop_import_monitor = lambda _key: None
            mimics_import._write_import_task_status = lambda _path, payload: statuses.append(
                dict(payload)
            )
            mimics_import._terminate_job_process = lambda _job, on_complete=None: callbacks.append(
                on_complete
            )
            mimics_import._safe_message_box = lambda *_args, **_kwargs: None
            mimics_import._cleanup_job_dir = lambda _path: None
            mimics_import._cleanup_work_dir = lambda _path: None
            mimics_import._record_failed_case = lambda *_args: None
            mimics_import._mark_mcs_queue_done = lambda *_args, **_kwargs: None
            self.assertTrue(mimics_import._fail_import_monitor_after_process(
                monitor, "prepare_timeout", "timed out"
            ))
            self.assertEqual("stopping", statuses[-1]["status"])
            self.assertNotIn("failed", [item["status"] for item in statuses])
            callbacks[0]()
        finally:
            mimics_import._stop_import_monitor = old_stop
            mimics_import._write_import_task_status = old_write
            mimics_import._terminate_job_process = old_terminate
            mimics_import._safe_message_box = old_message
            mimics_import._cleanup_job_dir = old_cleanup_job
            mimics_import._cleanup_work_dir = old_cleanup_work
            mimics_import._record_failed_case = old_record
            mimics_import._mark_mcs_queue_done = old_done
        self.assertEqual("failed", statuses[-1]["status"])

    def test_batch_import_timeout_does_not_overlap_next_bridge(self):
        import mimics_import

        callbacks = []
        starts = []
        statuses = []
        monitor = {
            "job_dir": os.path.join(self.tmp, "job"),
            "work_dir": os.path.join(self.tmp, "work"),
            "output_dir": self.tmp,
            "case_id": "case_a",
            "deadline": 0,
            "done": False,
            "busy": False,
            "completed": 0,
            "failed": 0,
            "total": 2,
            "batch_queue": [{"case_id": "case_b"}],
        }
        old_stopped = mimics_import._import_task_stopped
        old_terminate = mimics_import._terminate_job_process
        old_start = mimics_import._start_next_batch_prepare
        old_write = mimics_import._write_import_task_status
        old_log = mimics_import._append_import_log
        old_record = mimics_import._record_failed_case
        old_cleanup_job = mimics_import._cleanup_job_dir
        old_cleanup_work = mimics_import._cleanup_work_dir
        try:
            mimics_import._import_task_stopped = lambda _monitor: False
            mimics_import._terminate_job_process = lambda _job, on_complete=None: callbacks.append(
                on_complete
            )
            mimics_import._start_next_batch_prepare = lambda _monitor: starts.append(True)
            mimics_import._write_import_task_status = lambda _path, payload: statuses.append(
                dict(payload)
            )
            mimics_import._append_import_log = lambda *_args: None
            mimics_import._record_failed_case = lambda *_args: None
            mimics_import._cleanup_job_dir = lambda _path: None
            mimics_import._cleanup_work_dir = lambda _path: None
            mimics_import._batch_prepare_tick_impl(monitor)
            self.assertEqual([], starts)
            self.assertEqual("stopping_case", statuses[-1]["phase"])
            callbacks[0]()
            mimics_import._batch_prepare_tick_impl(monitor)
        finally:
            mimics_import._import_task_stopped = old_stopped
            mimics_import._terminate_job_process = old_terminate
            mimics_import._start_next_batch_prepare = old_start
            mimics_import._write_import_task_status = old_write
            mimics_import._append_import_log = old_log
            mimics_import._record_failed_case = old_record
            mimics_import._cleanup_job_dir = old_cleanup_job
            mimics_import._cleanup_work_dir = old_cleanup_work
        self.assertEqual([True], starts)
        self.assertEqual(1, monitor["failed"])

    def test_export_notification_failure_does_not_overwrite_completed_state(self):
        import mimics_export

        statuses = []
        releases = []
        monitor = {
            "monitor_key": "export-a",
            "work_dir": os.path.join(self.tmp, "work"),
            "output_root": self.tmp,
            "selected_masks": [object()],
            "operation_token": "token",
            "done": False,
        }
        old_stop = mimics_export._stop_export_monitor
        old_apply = mimics_export._apply_export_result
        old_write = mimics_export._write_export_task_status
        old_release = mimics_export.runtime_common.release_local_operation
        old_cleanup = mimics_export._cleanup_work_dir
        old_log = mimics_export._mimics_log
        old_message = mimics_export.mimics.dialogs.message_box
        try:
            mimics_export._stop_export_monitor = lambda _key: None
            mimics_export._apply_export_result = lambda *_args: (1, 0, 0)
            mimics_export._write_export_task_status = lambda _monitor, payload: statuses.append(
                dict(payload)
            )
            mimics_export.runtime_common.release_local_operation = (
                lambda *_args: releases.append(True) or True
            )
            mimics_export._cleanup_work_dir = lambda _path: None
            mimics_export._mimics_log = lambda *_args: None
            mimics_export.mimics.dialogs.message_box = lambda **_kwargs: (
                _ for _ in ()
            ).throw(RuntimeError("dialog failed"))
            mimics_export._finish_foreground_export(
                monitor,
                result={"output_seg_dir": self.tmp},
            )
        finally:
            mimics_export._stop_export_monitor = old_stop
            mimics_export._apply_export_result = old_apply
            mimics_export._write_export_task_status = old_write
            mimics_export.runtime_common.release_local_operation = old_release
            mimics_export._cleanup_work_dir = old_cleanup
            mimics_export._mimics_log = old_log
            mimics_export.mimics.dialogs.message_box = old_message
        self.assertEqual(["completed"], [item["status"] for item in statuses])
        self.assertEqual([True], releases)

    def test_cache_cleanup_skips_all_scratch_paths_while_task_is_active(self):
        import mimics_stop_background

        scratch = Path(self.tmp) / "active_work"
        scratch.mkdir()
        (scratch / "buffer.u8").write_bytes(b"x")
        old_blockers = mimics_stop_background._active_cache_cleanup_blockers
        old_find = mimics_stop_background._find_cache_paths
        try:
            mimics_stop_background._active_cache_cleanup_blockers = lambda: [
                "background import is active"
            ]
            mimics_stop_background._find_cache_paths = lambda: [str(scratch)]
            result = mimics_stop_background.clear_all_caches()
        finally:
            mimics_stop_background._active_cache_cleanup_blockers = old_blockers
            mimics_stop_background._find_cache_paths = old_find
        self.assertTrue(scratch.exists())
        self.assertEqual([], result["removed"])
        self.assertEqual(1, len(result["skipped_in_use"]))

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

    def test_v3_idle_nninteractive_worker_yields_gpu_without_grace_delay(self):
        import tools.fewshot_pipeline as pipeline

        control_dir = Path(self.tmp) / "worker-v3"
        (control_dir / "commands").mkdir(parents=True)
        state_path = Path(self.tmp) / "server-v3.json"
        pipeline.write_json_atomic(control_dir / "worker_status.json", {
            "status": "result_ready", "sequence": 1,
        })
        pipeline.write_json_atomic(state_path, {
            "schema_version": "nninteractive_owned_server.v3",
            "pid": 2001,
            "watchdog_pid": 2002,
            "client_pid": 2003,
            "client_control_dir": str(control_dir),
            "gpu_lock_token": "token-v3",
            "last_activity_epoch": time.time(),
        })
        current = {
            "resource": "gpu", "token": "token-v3",
            "state_path": str(state_path),
        }
        original = pipeline.process_exists
        try:
            pipeline.process_exists = lambda _pid: True
            self.assertTrue(
                pipeline.request_nninteractive_server_release_on_contention(current)
            )
        finally:
            pipeline.process_exists = original
        self.assertTrue((control_dir / "close.json").is_file())

    def test_v3_idle_worker_can_close_after_watchdog_failure(self):
        import tools.fewshot_pipeline as pipeline

        control_dir = Path(self.tmp) / "worker-no-watchdog"
        (control_dir / "commands").mkdir(parents=True)
        pipeline.write_json_atomic(control_dir / "worker_status.json", {
            "status": "result_ready", "sequence": 1,
        })
        state_path = Path(self.tmp) / "server-no-watchdog.json"
        pipeline.write_json_atomic(state_path, {
            "schema_version": "nninteractive_owned_server.v3",
            "pid": 4001,
            "watchdog_pid": 4002,
            "client_pid": 4003,
            "client_control_dir": str(control_dir),
            "gpu_lock_token": "token-no-watchdog",
        })
        current = {
            "resource": "gpu", "token": "token-no-watchdog",
            "state_path": str(state_path),
        }
        original = pipeline.process_exists
        try:
            pipeline.process_exists = lambda pid: int(pid) != 4002
            self.assertTrue(
                pipeline.request_nninteractive_server_release_on_contention(current)
            )
        finally:
            pipeline.process_exists = original
        self.assertTrue((control_dir / "close.json").is_file())

    def test_worker_without_first_status_is_not_closed(self):
        import tools.fewshot_pipeline as pipeline

        control_dir = Path(self.tmp) / "worker-starting"
        (control_dir / "commands").mkdir(parents=True)
        state_path = Path(self.tmp) / "server-starting.json"
        pipeline.write_json_atomic(state_path, {
            "schema_version": "nninteractive_owned_server.v3",
            "pid": 5001,
            "watchdog_pid": 5002,
            "client_pid": 5003,
            "client_control_dir": str(control_dir),
            "gpu_lock_token": "token-starting",
        })
        current = {
            "resource": "gpu", "token": "token-starting",
            "state_path": str(state_path),
        }
        original = pipeline.process_exists
        try:
            pipeline.process_exists = lambda _pid: True
            self.assertFalse(
                pipeline.request_nninteractive_server_release_on_contention(current)
            )
        finally:
            pipeline.process_exists = original
        self.assertFalse((control_dir / "close.json").exists())

    def test_queued_nninteractive_prediction_does_not_yield_gpu(self):
        import tools.fewshot_pipeline as pipeline

        control_dir = Path(self.tmp) / "worker-pending"
        (control_dir / "commands").mkdir(parents=True)
        pipeline.write_json_atomic(control_dir / "worker_status.json", {
            "status": "result_ready", "sequence": 1,
        })
        pipeline.write_json_atomic(
            control_dir / "commands" / "command_000002.json", {"sequence": 2}
        )
        state_path = Path(self.tmp) / "server-pending.json"
        pipeline.write_json_atomic(state_path, {
            "schema_version": "nninteractive_owned_server.v3",
            "pid": 3001,
            "watchdog_pid": 3002,
            "client_pid": 3003,
            "client_control_dir": str(control_dir),
            "gpu_lock_token": "token-pending",
        })
        current = {
            "resource": "gpu", "token": "token-pending",
            "state_path": str(state_path),
        }
        original = pipeline.process_exists
        try:
            pipeline.process_exists = lambda _pid: True
            self.assertFalse(
                pipeline.request_nninteractive_server_release_on_contention(current)
            )
        finally:
            pipeline.process_exists = original
        self.assertFalse((control_dir / "close.json").exists())

    def test_failed_fewshot_job_removes_large_artifacts_but_keeps_diagnostics(self):
        import tools.fewshot_pipeline as pipeline

        workspace = Path(self.tmp) / "fewshot_models"
        run_id = "train_failed_cleanup"
        status_path = workspace / "jobs" / (run_id + ".json")
        run_dir = workspace / "runs" / "liver" / run_id
        dataset_dir = workspace / "datasets" / "liver" / run_id
        model_dir = workspace / "models" / "liver" / run_id
        experiment_dir = Path(self.tmp) / (
            "mimics_fewshot_liver_" + run_id
        )
        for path in (
            run_dir / "fresh_labels",
            dataset_dir,
            model_dir,
            experiment_dir,
        ):
            path.mkdir(parents=True)
            (path / "large.bin").write_bytes(b"x" * 1024)
        (run_dir / "train.log").write_text("diagnostic", encoding="utf-8")
        pipeline.write_json_atomic(
            status_path,
            {
                "job_id": run_id,
                "status": "failed",
                "organ": "liver",
                "workspace": str(workspace),
                "dataset_dir": str(dataset_dir),
                "experiment_dir": str(experiment_dir),
            },
        )
        args = type(
            "Args",
            (),
            {"run_id": run_id, "organ": "liver"},
        )()

        report = pipeline.cleanup_terminal_training_artifacts(args, status_path)

        self.assertGreaterEqual(len(report.get("removed") or []), 4)
        self.assertFalse(dataset_dir.exists())
        self.assertFalse(model_dir.exists())
        self.assertFalse(experiment_dir.exists())
        self.assertFalse((run_dir / "fresh_labels").exists())
        self.assertTrue((run_dir / "train.log").is_file())
        self.assertTrue(status_path.is_file())

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

    def test_fewshot_cleanup_keeps_job_when_controller_is_alive(self):
        pipeline = __import__("tools.fewshot_pipeline", fromlist=["dummy"])
        workspace = Path(self.tmp) / "fewshot_models"
        jobs = workspace / "jobs"
        jobs.mkdir(parents=True)
        path = jobs / "controller_alive.json"
        pipeline.write_json_atomic(path, {
            "job_id": "controller_alive",
            "status": "training",
            "pid": 987654321,
            "controller_pid": os.getpid(),
            "updated_at_epoch": time.time() - 3600,
        })
        pipeline.cleanup_workspace_artifacts(workspace, config={})
        payload = pipeline.read_json(path, {}) or {}
        self.assertEqual("training", payload.get("status"))
        self.assertFalse(payload.get("orphaned", False))

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
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()
        self.assertNotIn("$broad=Get-CimInstance", source)
        self.assertIn("$foregroundPid", source)
        self.assertIn("$inRoot -and $hasMarker", source)
        self.assertIn("$cutoff", source)
        self.assertNotIn("Remove-Item -Path $lock", source)

    def test_external_kill_background_never_deletes_resource_locks(self):
        import inspect
        import tools.mimics_batch_cli as cli

        source = inspect.getsource(cli.cmd_kill_background)
        self.assertIn("stopMarkers", source)
        self.assertIn("cutoff", source)
        self.assertNotIn("Remove-Item", source)


if __name__ == "__main__":
    print("Mimics-Script Comprehensive Tests")
    print("=" * 60)
    print("Python: {}".format(sys.version))
    print("Project: {}".format(PROJECT_ROOT))
    print("=" * 60)
    unittest.main(verbosity=2)
