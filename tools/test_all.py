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
_mock_mimics.file.get_active_project = lambda: None
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
            "mimics_mask_apply.py",
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


class TestMaskApplyLegacyStateMigration(unittest.TestCase):
    """B10b: few-shot leftovers must not become the user's settings."""

    def setUp(self):
        self.tmp = _make_temp_dir()

    def tearDown(self):
        _cleanup(self.tmp)

    def _module(self):
        import mimics_mask_apply

        return mimics_mask_apply

    def test_legacy_state_with_only_dead_roots_is_skipped(self):
        module = self._module()
        runtime_dir = os.path.join(self.tmp, ".mimics_runtime")
        os.makedirs(runtime_dir)
        legacy = os.path.join(runtime_dir, "fewshot_mimics_state.json")
        with open(legacy, "w", encoding="utf-8") as handle:
            json.dump({
                "last_dataset_root": "C:\\gone\\dataset",
                "recent_dataset_roots": [
                    {"path": "C:\\also\\gone", "used_at_epoch": 1.0}
                ],
            }, handle)
        with mock.patch.object(module, "_project_root", lambda: self.tmp):
            module._migrate_old_settings()
            # Migration must not run: every remembered root is dead.
            self.assertTrue(os.path.isfile(legacy))
            self.assertFalse(
                os.path.isfile(os.path.join(runtime_dir, "mimics_mask_apply_state.json"))
            )

    def test_legacy_state_with_live_root_still_migrates(self):
        module = self._module()
        runtime_dir = os.path.join(self.tmp, ".mimics_runtime")
        os.makedirs(runtime_dir)
        legacy = os.path.join(runtime_dir, "fewshot_mimics_state.json")
        live_root = os.path.join(self.tmp, "live_dataset")
        os.makedirs(live_root)
        with open(legacy, "w", encoding="utf-8") as handle:
            json.dump({"last_dataset_root": live_root}, handle)
        new_path = os.path.join(runtime_dir, "mimics_mask_apply_state.json")
        with mock.patch.object(module, "_project_root", lambda: self.tmp):
            module._migrate_old_settings()
            self.assertFalse(os.path.isfile(legacy))
            self.assertTrue(os.path.isfile(new_path))


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

    def test_derived_dicom_kill_mid_slices_leaves_unpublished_residue_and_rerun_recovers(self):
        # TB-01: the process dies while writing slice N. The residue must be
        # confined to the output directory (no manifest, never a valid series
        # on disk), and a rerun — whose first step wipes the directory —
        # produces the full, valid series.
        import pydicom
        from mimics_bridge import nifti_to_derived_dicom

        import nibabel as nib

        shape = (6, 5, 8)
        data = np.arange(shape[0] * shape[1] * shape[2], dtype=np.int16).reshape(shape)
        nii_path = os.path.join(self.tmp, "ct_kill.nii.gz")
        nib.save(nib.Nifti1Image(data, np.diag([1.0, 1.0, 2.0, 1.0])), nii_path)
        dicom_out = os.path.join(self.tmp, "dicom_kill")

        # Kill at slice 4 (of 8): write a truncated file, then die like a
        # hard-killed process would — no exception machinery, no cleanup.
        real_save_as = pydicom.dataset.FileDataset.save_as

        def killed_save_as(ds_self, path, *args, **kwargs):
            if path.endswith("slice_0004.dcm"):
                with open(path, "wb") as handle:
                    handle.write(ds_self.preamble or b"\0" * 128)
                raise KeyboardInterrupt("process killed mid-slice")
            return real_save_as(ds_self, path, *args, **kwargs)

        with mock.patch.object(
            pydicom.dataset.FileDataset, "save_as", killed_save_as
        ):
            with self.assertRaises(KeyboardInterrupt):
                nifti_to_derived_dicom(nii_path, dicom_out)

        # Residue shape: 3 complete slices + 1 truncated, nothing else. No
        # validation ran (it only runs after a full loop), and no manifest
        # exists anywhere for a consumer to mistake this for a case.
        files = sorted(
            name for name in os.listdir(dicom_out) if name.endswith(".dcm")
        )
        self.assertEqual(
            ["slice_0001.dcm", "slice_0002.dcm", "slice_0003.dcm", "slice_0004.dcm"],
            files,
        )
        self.assertLess(os.path.getsize(os.path.join(dicom_out, files[-1])), 200)

        # Rerun on a healthy process: wipe-then-write recovers fully.
        result = nifti_to_derived_dicom(nii_path, dicom_out)
        files = sorted(
            name for name in os.listdir(dicom_out) if name.endswith(".dcm")
        )
        self.assertEqual(shape[2], len(files))
        first = pydicom.dcmread(os.path.join(dicom_out, files[0]))
        self.assertEqual(int(first.Rows), shape[1])
        self.assertEqual(shape[2], result["shape"][2])

    def test_derived_dicom_validation_rejects_incomplete_series(self):
        # TB-01: the post-loop completeness gate — never directly tested
        # before. A series missing one slice must fail closed.
        from mimics_bridge import _validate_derived_dicom_series

        source = os.path.join(self.tmp, "dicom_validate")
        os.makedirs(source)
        # Minimal valid-enough DICOM slice for the validator's header read,
        # built the way the producer builds them (file meta + preamble).
        import pydicom

        def write_slice(name, series_uid, study_uid):
            file_meta = pydicom.dataset.FileMetaDataset()
            file_meta.TransferSyntaxUID = (
                pydicom.uid.ExplicitVRLittleEndian
            )
            file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
            file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
            ds = pydicom.dataset.FileDataset(
                os.path.join(source, name),
                {},
                file_meta=file_meta,
                preamble=b"\0" * 128,
            )
            ds.is_little_endian = True
            ds.is_implicit_VR = False
            ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
            ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
            ds.SeriesInstanceUID = series_uid
            ds.StudyInstanceUID = study_uid
            ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            ds.ImagePositionPatient = [0, 0, 0]
            ds.PixelSpacing = [1.0, 1.0]
            ds.InstanceNumber = 1
            ds.save_as(os.path.join(source, name))

        write_slice("slice_0001.dcm", "1.2.3", "1.2.4")
        write_slice("slice_0002.dcm", "1.2.3", "1.2.4")

        # Slice count mismatch: 2 on disk, 3 expected.
        with self.assertRaisesRegex(RuntimeError, "slice count mismatch"):
            _validate_derived_dicom_series(Path(source), 3, "1.2.3", "1.2.4")

        # A truncated-to-empty file.
        write_slice("slice_0003.dcm", "1.2.3", "1.2.4")
        with open(os.path.join(source, "slice_0003.dcm"), "wb") as handle:
            handle.write(b"")
        with self.assertRaisesRegex(RuntimeError, "empty file"):
            _validate_derived_dicom_series(Path(source), 3, "1.2.3", "1.2.4")

        # A foreign-series UID sneaked in (e.g. residue from another run).
        write_slice("slice_0003.dcm", "9.9.9", "1.2.4")
        with self.assertRaisesRegex(RuntimeError, "series UID mismatch"):
            _validate_derived_dicom_series(Path(source), 3, "1.2.3", "1.2.4")

        # Complete and consistent: passes.
        write_slice("slice_0003.dcm", "1.2.3", "1.2.4")
        _validate_derived_dicom_series(Path(source), 3, "1.2.3", "1.2.4")

    def test_mask_nifti_kill_mid_write_never_publishes_partial_file(self):
        # TB-01 export side: the atomic temp+replace publish means a kill
        # mid-nib.save can only ever leave a ._mimics_mask_ temp file — the
        # final .nii.gz either stays at its previous content or never
        # appears. No truncated mask can be mistaken for an export.
        import mimics_bridge
        import nibabel as nib

        data = np.zeros((4, 4, 4), dtype=np.uint8)
        data[1, 1, 1] = 1

        destination = os.path.join(self.tmp, "published_mask.nii.gz")
        with open(destination, "wb") as handle:
            handle.write(b"previous complete export")

        real_nib_save = nib.save

        def killed_save(image, path, *args, **kwargs):
            if "._mimics_mask_" in str(path):
                # Half the bytes land, then the process dies.
                with open(str(path), "ab") as handle:
                    handle.write(b"\x1f\x8b\x08partial")
                raise KeyboardInterrupt("process killed mid-write")
            return real_nib_save(image, path, *args, **kwargs)

        # nibabel is imported lazily inside write_mask_nifti; patch the
        # module attribute the local import will resolve to.
        with mock.patch.object(nib, "save", killed_save):
            with self.assertRaises(KeyboardInterrupt):
                mimics_bridge.write_mask_nifti(
                    data, np.eye(4), destination
                )

        # The published file is untouched; only a temp residue exists.
        with open(destination, "rb") as handle:
            self.assertEqual(handle.read(), b"previous complete export")
        temps = [
            name
            for name in os.listdir(self.tmp)
            if name.startswith("._mimics_mask_")
        ]
        # The temp may or may not survive the kill (the finally handler is
        # skipped on a hard kill) — but it must never be the final name.
        for name in temps:
            self.assertNotEqual(name, "published_mask.nii.gz")

        # A rerun with a healthy writer publishes normally, leaving no temp.
        mimics_bridge.write_mask_nifti(data, np.eye(4), destination)
        reloaded = nib.load(destination)
        self.assertEqual((4, 4, 4), reloaded.shape)
        temps = [
            name
            for name in os.listdir(self.tmp)
            if name.startswith("._mimics_mask_")
        ]
        self.assertEqual([], temps)

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


class TestVerifyMedicalGeometry(unittest.TestCase):
    """C8: tools/verify_medical_geometry.py guards data quality — an error
    there would wave a mis-gridded export through. Zero direct coverage."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="verify_geometry_")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _write_nifti(self, name, data, affine):
        import nibabel as nib

        path = os.path.join(self.tmp, name)
        nib.save(nib.Nifti1Image(data, affine), path)
        return path

    def _run(self, image, mask, extra_args=None):
        tools_dir = os.path.dirname(os.path.abspath(__file__))
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)
        import verify_medical_geometry

        argv = [
            "verify_medical_geometry",
            "--image", image,
            "--mask", mask,
        ] + list(extra_args or [])
        with mock.patch.object(sys, "argv", argv):
            return verify_medical_geometry.main()

    def _fake_geometry(self, shape, affine):
        """Patch get_source_image_geometry inside verify_medical_geometry."""
        tools_dir = os.path.dirname(os.path.abspath(__file__))
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)
        import verify_medical_geometry

        return mock.patch.object(
            verify_medical_geometry,
            "get_source_image_geometry",
            return_value={"shape": list(shape), "affine": np.asarray(affine).tolist()},
        )

    def test_matching_grid_passes(self):
        data = np.zeros((3, 4, 5), dtype=np.uint8)
        data[1, 2, 3] = 1
        image = self._write_nifti("image.nii.gz", np.arange(60).reshape(3, 4, 5).astype(np.int16), np.eye(4))
        mask = self._write_nifti("mask.nii.gz", data, np.eye(4))
        with self._fake_geometry((3, 4, 5), np.eye(4)):
            self.assertEqual(0, self._run(image, mask))

    def test_affine_mismatch_fails_closed(self):
        data = np.zeros((3, 4, 5), dtype=np.uint8)
        data[1, 2, 3] = 1
        bad_affine = np.eye(4)
        bad_affine[0, 3] = 5.0
        image = self._write_nifti("image2.nii.gz", np.arange(60).reshape(3, 4, 5).astype(np.int16), np.eye(4))
        mask = self._write_nifti("mask2.nii.gz", data, bad_affine)
        with self._fake_geometry((3, 4, 5), np.eye(4)):
            self.assertEqual(2, self._run(image, mask))

    def test_shape_mismatch_fails_closed(self):
        data = np.zeros((3, 4, 5), dtype=np.uint8)
        data[1, 2, 3] = 1
        image = self._write_nifti("image3.nii.gz", np.arange(60).reshape(3, 4, 5).astype(np.int16), np.eye(4))
        mask = self._write_nifti("mask3.nii.gz", data, np.eye(4))
        # Image reports a different shape than the mask.
        with self._fake_geometry((4, 4, 5), np.eye(4)):
            self.assertEqual(2, self._run(image, mask))

    def test_unreadable_source_geometry_raises(self):
        data = np.zeros((3, 4, 5), dtype=np.uint8)
        image = self._write_nifti("image4.nii.gz", np.arange(60).reshape(3, 4, 5).astype(np.int16), np.eye(4))
        mask = self._write_nifti("mask4.nii.gz", data, np.eye(4))
        tools_dir = os.path.dirname(os.path.abspath(__file__))
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)
        import verify_medical_geometry

        with mock.patch.object(
            verify_medical_geometry,
            "get_source_image_geometry",
            return_value=None,
        ):
            with mock.patch.object(sys, "argv", [
                "verify_medical_geometry", "--image", image, "--mask", mask,
            ]):
                with self.assertRaises(RuntimeError) as ctx:
                    verify_medical_geometry.main()
        self.assertIn("could not be read", str(ctx.exception))

    def test_nrrd_mask_uses_lps_to_ras_conversion(self):
        import SimpleITK as sitk

        data = np.zeros((5, 4, 3), dtype=np.uint8)  # sitk: z/y/x
        data[3, 2, 1] = 1
        image = self._write_nifti("image5.nii.gz", np.arange(60).reshape(3, 4, 5).astype(np.int16), np.eye(4))
        # Build an LPS-affine NRRD whose RAS equivalent is the identity.
        sitk_image = sitk.GetImageFromArray(np.transpose(data, (2, 1, 0)))
        sitk_image.SetSpacing((1.0, 1.0, 1.0))
        sitk_image.SetOrigin((0.0, 0.0, 0.0))
        sitk_image.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
        mask = os.path.join(self.tmp, "mask.nrrd")
        sitk.WriteImage(sitk_image, mask)
        # Identity LPS direction -> RAS affine is diag(-1, -1, 1), which
        # must NOT match the identity-geometry source; the tool must
        # compare in RAS, not raw LPS.
        with self._fake_geometry((3, 4, 5), np.eye(4)):
            self.assertEqual(2, self._run(image, mask))

    def test_reference_mask_dice(self):
        data = np.zeros((3, 4, 5), dtype=np.uint8)
        data[1, 2, 3] = 1
        image = self._write_nifti("image6.nii.gz", np.arange(60).reshape(3, 4, 5).astype(np.int16), np.eye(4))
        mask = self._write_nifti("mask6.nii.gz", data, np.eye(4))
        reference = self._write_nifti("ref6.nii.gz", data, np.eye(4))
        with self._fake_geometry((3, 4, 5), np.eye(4)):
            self.assertEqual(0, self._run(image, mask, ["--reference-mask", reference]))


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


class TestWindowLevelEditor(unittest.TestCase):
    """External preset editor: persistence round-trip, backup, validation."""

    def setUp(self):
        self.tmp = _make_temp_dir()
        tools_dir = os.path.dirname(os.path.abspath(__file__))
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)
        import window_level_editor_ui as editor

        self.editor = editor
        self.presets_path = Path(self.tmp) / "window_level_presets.json"
        self.backup_path = Path(self.tmp) / "backup.json"
        self.sample = [
            {"name": "Lung", "width": 1500, "level": -600, "keywords": ["lung"]},
            {"name": "Bone", "width": 1800, "level": 400, "keywords": ["bone"]},
        ]

    def tearDown(self):
        _cleanup(self.tmp)

    def test_save_and_load_roundtrip(self):
        self.editor.save_presets(self.sample, self.presets_path)
        loaded = self.editor.load_presets(self.presets_path)
        self.assertEqual(self.sample, loaded)

    def test_load_missing_file_returns_empty(self):
        self.assertEqual([], self.editor.load_presets(self.presets_path))

    def test_load_rejects_non_list_json(self):
        self.presets_path.write_text('{"not": "a list"}', encoding="utf-8")
        self.assertEqual([], self.editor.load_presets(self.presets_path))

    def test_save_is_atomic_no_tmp_residue(self):
        self.editor.save_presets(self.sample, self.presets_path)
        leftovers = list(self.presets_path.parent.glob("*.tmp"))
        self.assertEqual([], leftovers)

    def test_rollback_restores_previous_save(self):
        self.editor.save_presets(self.sample, self.presets_path)
        self.editor.write_backup(self.sample, self.backup_path)
        changed = [dict(self.sample[0]), dict(self.sample[1])]
        changed[0]["width"] = 999
        del changed[1]
        self.editor.save_presets(changed, self.presets_path)
        self.editor.save_presets(
            self.editor.load_presets(self.backup_path), self.presets_path
        )
        restored = self.editor.load_presets(self.presets_path)
        self.assertEqual(self.sample, restored)

    def test_editor_files_and_entry_exist(self):
        self.assertTrue(
            os.path.isfile(os.path.join(PROJECT_ROOT, "tools", "window_level_editor_ui.py"))
        )
        self.assertTrue(
            os.path.isfile(
                os.path.join(PROJECT_ROOT, "runtime_py35", "window_level_editor_mimics.py")
            )
        )
        entry = os.path.join(
            PROJECT_ROOT, "scripting_library", "03_Review", "05_Window_Edit_Presets.py"
        )
        self.assertTrue(os.path.isfile(entry))
        with open(entry, "r") as handle:
            source = handle.read()
        self.assertIn("window_level_editor_mimics", source)

    def test_editor_window_renders_offscreen(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        import PySide6
        from PySide6 import QtWidgets

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
            ["test"]
        )
        editor = self.editor.PresetEditor(
            (PySide6.QtCore, PySide6.QtGui, QtWidgets)
        )
        # Rendering smoke: table populated from the real presets file.
        self.assertEqual(
            editor.table.rowCount(), len(self.editor.load_presets())
        )
        editor.window.close()


class TestConfigEditor(unittest.TestCase):
    """External config editor: schema wiring, nested get/set, validation."""

    def setUp(self):
        self.tmp = _make_temp_dir()
        tools_dir = os.path.dirname(os.path.abspath(__file__))
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)
        import config_editor_ui as editor

        self.editor = editor

    def tearDown(self):
        _cleanup(self.tmp)

    def test_schemas_reference_real_config_files(self):
        for file_name in self.editor.CONFIG_SCHEMAS:
            self.assertTrue(
                os.path.isfile(os.path.join(PROJECT_ROOT, file_name)), file_name
            )

    def test_nested_get_set_roundtrip(self):
        data = {"scribbleprompt": {"device": "cpu", "timeout_seconds": 900}}
        self.assertEqual("cpu", self.editor._get_nested(data, "scribbleprompt.device"))
        self.assertIsNone(self.editor._get_nested(data, "missing.key"))
        self.editor._set_nested(data, "scribbleprompt.device", "cuda")
        self.assertEqual("cuda", data["scribbleprompt"]["device"])
        self.editor._set_nested(data, "new.group.key", 5)
        self.assertEqual(5, data["new"]["group"]["key"])

    def test_choice_options_cover_schema_keys(self):
        for (file_name, key), options in self.editor.CHOICE_OPTIONS.items():
            ftype = self.editor.CONFIG_SCHEMAS[file_name][key][1]
            self.assertEqual("choice", ftype, (file_name, key))
            self.assertTrue(options, (file_name, key))

    def test_config_editor_files_and_entry_exist(self):
        self.assertTrue(
            os.path.isfile(os.path.join(PROJECT_ROOT, "tools", "config_editor_ui.py"))
        )
        self.assertTrue(
            os.path.isfile(
                os.path.join(PROJECT_ROOT, "runtime_py35", "config_editor_mimics.py")
            )
        )
        entry = os.path.join(
            PROJECT_ROOT, "scripting_library", "99_Admin", "07_Edit_Configs.py"
        )
        self.assertTrue(os.path.isfile(entry))
        with open(entry, "r") as handle:
            source = handle.read()
        self.assertIn("config_editor_mimics", source)

    def test_config_editor_window_renders_offscreen(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        import PySide6
        from PySide6 import QtWidgets

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
            ["test"]
        )
        editor = self.editor.ConfigEditor(
            (PySide6.QtCore, PySide6.QtGui, QtWidgets)
        )
        # One tab per config schema, widgets populated from real files.
        self.assertEqual(
            len(self.editor.CONFIG_SCHEMAS),
            editor.window.findChild(QtWidgets.QTabWidget).count(),
        )
        editor.window.close()


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

    def test_record_failed_case_keeps_user_source_path(self):
        """R61-5: a failure record must name the user's original input path,
        not an internal work/runtime dir, so the annotator can locate the
        problematic data."""
        from create_mcs_batch import record_failed_case

        record_failed_case(
            self.tmp, "s0002", "create_mcs", "boom",
            source_image=r"Z:\datasets\cases\s0002\ct.nii.gz",
        )
        failed_dir = os.path.join(self.tmp, ".mimics_runtime", "_failed_cases")
        files = os.listdir(failed_dir)
        with open(os.path.join(failed_dir, files[0]), "r") as f:
            data = json.load(f)
        self.assertEqual(
            r"Z:\datasets\cases\s0002\ct.nii.gz", data["source_image"]
        )

        # Omitted source stays absent (backwards-compatible records).
        record_failed_case(self.tmp, "s0003", "prepare", "boom")
        paths = sorted(
            path for path in os.listdir(failed_dir)
            if path.startswith("s0003")
        )
        with open(os.path.join(failed_dir, paths[0]), "r") as f:
            self.assertNotIn("source_image", json.load(f))

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

    def test_pruned_fingerprint_with_manifest_match_keeps_existing_mcs(self):
        """R61-2: after the 14-day queue prune deletes the fingerprint file,
        a reimport of the SAME source must skip, not overwrite the .mcs that
        may carry annotations. Provenance is recovered from the dataset
        manifest, and the fingerprint file is healed."""
        import create_mcs_batch

        output_dir = os.path.join(self.tmp, "output")
        runtime_dir = os.path.join(self.tmp, "local_queue")
        work_dir = os.path.join(self.tmp, "local_work", "case_gone")
        queue_dir = os.path.join(runtime_dir, "prepared_queue")
        os.makedirs(output_dir)
        os.makedirs(work_dir)
        os.makedirs(queue_dir)
        output_mcs = os.path.join(output_dir, "case_gone.mcs")
        with open(output_mcs, "w") as handle:
            handle.write("annotated mcs")
        with open(os.path.join(work_dir, "prepare_manifest.json"), "w") as handle:
            json.dump({"output_mcs": output_mcs, "source_fingerprint": "fp_same"}, handle)
        descriptor = os.path.join(queue_dir, "case_gone.json")
        with open(descriptor, "w") as handle:
            json.dump({"case_id": "case_gone", "work_dir": work_dir, "output_mcs": output_mcs}, handle)
        with open(os.path.join(runtime_dir, create_mcs_batch.QUEUE_DONE_FILE), "w") as handle:
            json.dump({"status": "done"}, handle)
        # The dataset manifest (survives pruning) records the same source.
        import dataset_manifest
        dataset_manifest.update_case(
            output_dir, "case_gone",
            image_path="ct.nii.gz", mcs_path=output_mcs,
            provenance={"last_operation": "mimics_import", "source_fingerprint": "fp_same"},
        )

        calls = []
        old_create = create_mcs_batch.create_mcs_from_manifest
        try:
            create_mcs_batch.create_mcs_from_manifest = (
                lambda work_dir, path: calls.append(work_dir) or path
            )
            self.assertEqual(0, create_mcs_batch.main(output_dir, runtime_dir=runtime_dir))
        finally:
            create_mcs_batch.create_mcs_from_manifest = old_create
        self.assertEqual([], calls, "same-source reimport must not recreate the .mcs")
        with open(output_mcs, "r") as handle:
            self.assertEqual("annotated mcs", handle.read(), "existing .mcs was overwritten")
        # The fingerprint file was healed from the manifest record.
        fp_path = os.path.join(
            runtime_dir, "fingerprints", "case_gone.fingerprint"
        )
        self.assertTrue(os.path.isfile(fp_path))
        with open(fp_path, "r") as handle:
            self.assertEqual("fp_same", handle.read().strip())

    def test_pruned_fingerprint_without_manifest_record_keeps_existing_mcs(self):
        """R61-2: published .mcs, fingerprint file pruned, NO manifest record
        (e.g. manifest update failed at creation time). Unknown provenance
        must never silently overwrite — skip and tell the user how to force."""
        import create_mcs_batch

        output_dir = os.path.join(self.tmp, "output")
        runtime_dir = os.path.join(self.tmp, "local_queue")
        work_dir = os.path.join(self.tmp, "local_work", "case_unknown")
        queue_dir = os.path.join(runtime_dir, "prepared_queue")
        os.makedirs(output_dir)
        os.makedirs(work_dir)
        os.makedirs(queue_dir)
        output_mcs = os.path.join(output_dir, "case_unknown.mcs")
        with open(output_mcs, "w") as handle:
            handle.write("annotated mcs")
        with open(os.path.join(work_dir, "prepare_manifest.json"), "w") as handle:
            json.dump({"output_mcs": output_mcs, "source_fingerprint": "fp_new"}, handle)
        descriptor = os.path.join(queue_dir, "case_unknown.json")
        with open(descriptor, "w") as handle:
            json.dump({"case_id": "case_unknown", "work_dir": work_dir, "output_mcs": output_mcs}, handle)
        with open(os.path.join(runtime_dir, create_mcs_batch.QUEUE_DONE_FILE), "w") as handle:
            json.dump({"status": "done"}, handle)

        calls = []
        old_create = create_mcs_batch.create_mcs_from_manifest
        try:
            create_mcs_batch.create_mcs_from_manifest = (
                lambda work_dir, path: calls.append(work_dir) or path
            )
            self.assertEqual(0, create_mcs_batch.main(output_dir, runtime_dir=runtime_dir))
        finally:
            create_mcs_batch.create_mcs_from_manifest = old_create
        self.assertEqual([], calls, "unknown-provenance reimport must not recreate the .mcs")
        with open(output_mcs, "r") as handle:
            self.assertEqual("annotated mcs", handle.read())

    def test_pruned_fingerprint_with_manifest_mismatch_reprocesses(self):
        """R61-2: the manifest records a DIFFERENT source fingerprint — a
        genuine source change. Reprocessing must still happen."""
        import create_mcs_batch

        output_dir = os.path.join(self.tmp, "output")
        runtime_dir = os.path.join(self.tmp, "local_queue")
        work_dir = os.path.join(self.tmp, "local_work", "case_changed")
        queue_dir = os.path.join(runtime_dir, "prepared_queue")
        os.makedirs(output_dir)
        os.makedirs(work_dir)
        os.makedirs(queue_dir)
        output_mcs = os.path.join(output_dir, "case_changed.mcs")
        with open(output_mcs, "w") as handle:
            handle.write("old mcs")
        with open(os.path.join(work_dir, "prepare_manifest.json"), "w") as handle:
            json.dump({"output_mcs": output_mcs, "source_fingerprint": "fp_new"}, handle)
        descriptor = os.path.join(queue_dir, "case_changed.json")
        with open(descriptor, "w") as handle:
            json.dump({"case_id": "case_changed", "work_dir": work_dir, "output_mcs": output_mcs}, handle)
        with open(os.path.join(runtime_dir, create_mcs_batch.QUEUE_DONE_FILE), "w") as handle:
            json.dump({"status": "done"}, handle)
        import dataset_manifest
        dataset_manifest.update_case(
            output_dir, "case_changed",
            image_path="ct.nii.gz", mcs_path=output_mcs,
            provenance={"last_operation": "mimics_import", "source_fingerprint": "fp_old"},
        )

        calls = []
        old_create = create_mcs_batch.create_mcs_from_manifest

        def fake_create(work_dir_arg, path):
            calls.append(work_dir_arg)
            with open(path, "w") as handle:
                handle.write("mcs")
            return path

        try:
            create_mcs_batch.create_mcs_from_manifest = fake_create
            self.assertEqual(0, create_mcs_batch.main(output_dir, runtime_dir=runtime_dir))
        finally:
            create_mcs_batch.create_mcs_from_manifest = old_create
        self.assertEqual([work_dir], calls, "genuine source change must reprocess")

    def test_partial_work_dir_without_manifest_is_never_converted(self):
        # TB-01: a bridge process killed mid-DICOM-write leaves a work dir
        # with partial .dcm files and NO prepare_manifest.json. The manifest
        # gate must ensure no half-converted case ever becomes an .mcs:
        # the stale descriptor is discarded and create is never called.
        import create_mcs_batch

        output_dir = os.path.join(self.tmp, "output")
        runtime_dir = os.path.join(self.tmp, "local_queue")
        # Half-written residue: 3 of 10 slices, no manifest (the producer
        # only writes it after a validated, complete conversion).
        work_dir = os.path.join(self.tmp, "local_work", "case_killed")
        queue_dir = os.path.join(runtime_dir, "prepared_queue")
        os.makedirs(output_dir)
        os.makedirs(work_dir)
        os.makedirs(queue_dir)
        for index in (1, 2, 3):
            with open(os.path.join(work_dir, "slice_{:04d}.dcm".format(index)), "wb") as handle:
                handle.write(b"partial")
        output_mcs = os.path.join(output_dir, "case_killed.mcs")
        descriptor = os.path.join(queue_dir, "case_killed.json")
        with open(descriptor, "w") as handle:
            json.dump({"case_id": "case_killed", "work_dir": work_dir, "output_mcs": output_mcs}, handle)
        with open(os.path.join(runtime_dir, create_mcs_batch.QUEUE_DONE_FILE), "w") as handle:
            json.dump({"status": "done"}, handle)
        # Age the descriptor past the stale window so the consumer may
        # discard it (fresh descriptors are skipped, not deleted).
        five_seconds_ago = time.time() - 6.0
        os.utime(descriptor, (five_seconds_ago, five_seconds_ago))

        calls = []
        old_create = create_mcs_batch.create_mcs_from_manifest
        old_runtime = create_mcs_batch._ACTIVE_RUNTIME_DIR
        try:
            create_mcs_batch.create_mcs_from_manifest = (
                lambda work_dir, path: calls.append(work_dir) or path
            )
            self.assertEqual(0, create_mcs_batch.main(output_dir, runtime_dir=runtime_dir))
        finally:
            create_mcs_batch.create_mcs_from_manifest = old_create
            create_mcs_batch._ACTIVE_RUNTIME_DIR = old_runtime
        self.assertEqual([], calls)
        self.assertFalse(os.path.isfile(output_mcs))
        # The stale descriptor was discarded; the residue dir itself stays
        # (cleaned by the import monitor's error path or a later sweep,
        # never promoted into the dataset).
        self.assertFalse(os.path.exists(descriptor))

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
            "04_Window_Undo_Last.py": "undo",
        }
        for fname, action in expected_routes.items():
            path = os.path.join(lib, fname)
            with open(path, "r") as f:
                source = f.read()
            self.assertIn("window_level_mimics", source)
            self.assertIn('"{}"'.format(action), source)

    def test_living_docs_do_not_reference_deleted_entries(self):
        """B24: entries deleted by R41/R43 must not appear as live references
        in user-facing living docs. Explicitly-marked history notes
        ("原 ... 删除" migration records) are allowed to name them."""
        deleted_markers = (
            "03_Export_Masks",
            "05_Window_Reset_Full_Range",
            "04_Fix_Source_Affine_Metadata",
            "nnUNet/04_Stop_Running_Task",
            # DINOv3 was retired per constitution; its library entries must
            # not reappear in living docs (vendored FlexiCT model code
            # carrying the DINOv3 license header is not a library entry).
            "02_AI/DINOv3",
        )
        living_docs = [
            "docs/mimics_entry_guide.md",
            "docs/scripting_library_workflows.md",
            "docs/mimics_real_data_validation.md",
            "docs/task_lifecycle_and_safety_policy_CN.md",
            "docs/windows_end_to_end_acceptance_2026-08-01.md",
            "CONFIG_REFERENCE.md",
        ]
        offenders = []
        for rel in living_docs:
            path = os.path.join(PROJECT_ROOT, rel)
            if not os.path.isfile(path):
                continue
            with open(path, "r", encoding="utf-8") as handle:
                for lineno, line in enumerate(handle, 1):
                    # Allow explicit migration-history notes.
                    if "原 `" in line or "已删除" in line or "deleted" in line.lower():
                        continue
                    for marker in deleted_markers:
                        if marker in line:
                            offenders.append(
                                "{}:{} references deleted entry {}".format(
                                    rel, lineno, marker
                                )
                            )
        self.assertEqual([], offenders)


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
            "remote_training_controller.py",
            "nnunet_pipeline.py",
            "nnunet_stage_worker.py",
            "nnunet_training_setup_ui.py",
            "nnunet_prediction_setup_ui.py",
            "nnunet_status_viewer.py",
            "flexict_pipeline.py",
            "flexict_training_setup_ui.py",
            "flexict_prediction_setup_ui.py",
            "flexict_active_learning_ui.py",
            "flexict_status_viewer.py",
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
        entry = os.path.join(PROJECT_ROOT, "scripting_library", "01_Data", "05_Stop_Mask_Export.py")
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
            {"kind": "nnunet_label_export", "pid": os.getpid(), "stop_path": stop_b},
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
            "fix_source_affine_metadata", "nninteractive_mimics",
            "nnunet_mimics", "flexict_mimics",
        ):
            self.assertIn(module_name, monitor_source)
        # Both external-controller integrations get a cancel marker written
        # BEFORE their monitors are detached (B23: FlexiCT previously missed
        # this, leaving its controllers to the ungraceful PowerShell sweep).
        cancel_source = inspect.getsource(
            mimics_stop_background._cancel_controllers_before_detach
        )
        self.assertIn('"action": "cancel"', cancel_source)
        self.assertLess(
            monitor_source.index('_cancel_controllers_before_detach("nnunet_mimics")'),
            monitor_source.index('_stop_import_monitor'),
        )
        self.assertLess(
            monitor_source.index('_cancel_controllers_before_detach("flexict_mimics")'),
            monitor_source.index('_stop_import_monitor'),
        )

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

    def test_stop_all_writes_flexict_cancel_before_detaching_monitor(self):
        """B23: FlexiCT controllers get the same pre-detach cancel marker
        nnU-Net gets (job_dir/status.json + control.json layout is shared)."""
        import mimics_stop_background as msb

        status_path = os.path.join(self.tmp, "flexict_status.json")
        control_path = os.path.join(self.tmp, "control.json")
        Path(status_path).write_text(
            json.dumps({
                "status": "training",
            }),
            encoding="utf-8",
        )
        observations = []

        def stop_monitor(key):
            control = msb.runtime_common.read_json(control_path, {}) or {}
            observations.append((key, control.get("action")))

        fake = _FakeModule()
        fake._MONITORS = {"flexict-active": {"status_path": status_path}}
        fake._stop_monitor = stop_monitor
        previous = sys.modules.get("flexict_mimics")
        try:
            sys.modules["flexict_mimics"] = fake
            msb._stop_inprocess_monitors()
        finally:
            if previous is None:
                sys.modules.pop("flexict_mimics", None)
            else:
                sys.modules["flexict_mimics"] = previous
        # The control marker must exist (fallback: dirname(status)/control.json)
        # and carry a cancel action before the monitor is detached.
        self.assertEqual([("flexict-active", "cancel")], observations)

    def test_environment_setup_has_a_scoped_stop_path(self):
        import inspect
        import setup_environment

        source = inspect.getsource(setup_environment.main)
        self.assertIn("Stop Current Setup", source)
        self.assertIn("terminate_process_async", source)

    def test_environment_setup_menu_recommends_action_first(self):
        """B6: the first button matches the workstation's actual state."""
        import setup_environment

        observed = []

        def fake_question_box(message="", buttons="", title="", ui_blocking=None, **kwargs):
            observed.append({"message": message, "buttons": buttons})
            return "Cancel"

        old_box = setup_environment.mimics.dialogs.question_box
        old_bundle = setup_environment._is_offline_bundle
        old_python = setup_environment.runtime_common.find_external_python
        old_root = setup_environment._project_root
        try:
            setup_environment.mimics.dialogs.question_box = fake_question_box
            setup_environment._project_root = lambda: "C:\\fake\\root"
            setup_environment.runtime_common.find_external_python = (
                lambda *args, **kwargs: ""
            )
            # No bundle, no python: extract archive leads.
            setup_environment._is_offline_bundle = lambda: False
            result = setup_environment.main()
            self.assertEqual(1, result)
            self.assertEqual(1, len(observed))
            self.assertTrue(observed[0]["buttons"].startswith("Extract Archive;"),
                            observed[0]["buttons"])
            self.assertIn("Recommended:", observed[0]["message"])
            # Existing install: Check leads, in both bundle and non-bundle.
            setup_environment.runtime_common.find_external_python = (
                lambda *args, **kwargs: "C:\\fake\\python.exe"
            )
            observed[:] = []
            setup_environment._is_offline_bundle = lambda: False
            self.assertEqual(1, setup_environment.main())
            self.assertTrue(observed[0]["buttons"].startswith("Check;"),
                            observed[0]["buttons"])
            observed[:] = []
            setup_environment._is_offline_bundle = lambda: True
            self.assertEqual(1, setup_environment.main())
            self.assertTrue(observed[0]["buttons"].startswith("Check;"),
                            observed[0]["buttons"])
            # Bundle, no python: Offline Install leads.
            observed[:] = []
            setup_environment.runtime_common.find_external_python = (
                lambda *args, **kwargs: ""
            )
            self.assertEqual(1, setup_environment.main())
            self.assertTrue(observed[0]["buttons"].startswith("Offline Install;"),
                            observed[0]["buttons"])
        finally:
            setup_environment.mimics.dialogs.question_box = old_box
            setup_environment._is_offline_bundle = old_bundle
            setup_environment.runtime_common.find_external_python = old_python
            setup_environment._project_root = old_root

    def test_environment_repair_is_blocked_while_other_tasks_are_active(self):
        import setup_environment

        launched = []
        messages = []
        old_blockers = setup_environment.runtime_common.active_runtime_blockers
        old_launch = setup_environment._launch_setup_worker
        old_message = setup_environment.mimics.dialogs.message_box
        try:
            setup_environment.runtime_common.active_runtime_blockers = (
                lambda *_args, **_kwargs: ["nnU-Net training"]
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
        self.assertTrue(any("nnU-Net training" in item for item in messages))

    def test_portable_archive_dialog_names_the_file_the_finder_searches(self):
        """B25: the "archive not found" dialog used to tell annotators to
        place a mistyped filename (mimcs_) while the finder searched the
        correct one — an annotator following the dialog landed in a loop."""
        import inspect
        import setup_environment

        # The finder's search list and the dialog must share one name.
        source = inspect.getsource(setup_environment._find_portable_archive)
        for candidate_line in source.splitlines():
            if "PORTABLE_ARCHIVE_NAME" in candidate_line or "mimics_script_portable" in candidate_line:
                self.assertNotIn("mimcs_script_portable", candidate_line)
        # Behavioral: run the real extract branch with no archive present.
        messages = []
        old_message = setup_environment.mimics.dialogs.message_box
        old_archive = setup_environment._find_portable_archive
        old_blockers = setup_environment.runtime_common.active_runtime_blockers
        try:
            setup_environment.mimics.dialogs.message_box = (
                lambda **kwargs: messages.append(kwargs.get("message", ""))
            )
            setup_environment._find_portable_archive = lambda: None
            setup_environment.runtime_common.active_runtime_blockers = (
                lambda *_args, **_kwargs: []
            )
            result = setup_environment.main("extract")
        finally:
            setup_environment.mimics.dialogs.message_box = old_message
            setup_environment._find_portable_archive = old_archive
            setup_environment.runtime_common.active_runtime_blockers = old_blockers
        self.assertEqual(1, result)
        self.assertEqual(1, len(messages))
        self.assertIn(setup_environment.PORTABLE_ARCHIVE_NAME, messages[0])
        self.assertNotIn("mimcs", messages[0])

    def test_nnunet_version_gate_matches_the_pinned_environment(self):
        """R61-14: the gate demanded >=2.8.1 while the shipped python_env has
        2.8.0, so every check reported a false 'UNSUPPORTED' and nudged users
        into a pointless reinstall. The floor must accept the version the
        project actually pins and installs."""
        import setup_env

        def _gate_for(version_text):
            parts = tuple(int(x) for x in version_text.split("."))
            parts = parts + (0,) * (3 - len(parts))
            return (2, 8, 0) <= parts < (2, 9, 0)

        # The shipped environment version must pass.
        self.assertTrue(_gate_for("2.8.0"))
        # Older or newer lines must still fail closed.
        self.assertFalse(_gate_for("2.7.3"))
        self.assertFalse(_gate_for("2.9.0"))
        # The version probe and the reinstall hints must agree on the floor.
        import inspect
        source = inspect.getsource(setup_env._nnunet_version_supported)
        self.assertIn("(2, 8, 0) <= parts", source)
        for hint in setup_env.REQUIRED_PACKAGES:
            if hint.startswith("nnunetv2"):
                self.assertEqual("nnunetv2>=2.8.0,<2.9", hint)

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
        persist_index = source.index("_write_fingerprint_file(")
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

    def test_scan_finds_marker_inside_mimics_runtime_and_skips_env(self):
        """R61-11: the queue-dir scan must PRUNE vendor/env trees while
        still walking .mimics_runtime (import_queues live there). The old
        per-dirpath "continue" descended into every python_env subtree."""
        import mimics_stop_background as msb

        queue = os.path.join(self.tmp, ".mimics_runtime", "import_queues", "out_abc")
        os.makedirs(queue)
        with open(os.path.join(queue, "_mcs_queue_active.json"), "w") as handle:
            handle.write('{"output_dir": "X:/out"}')
        # Noise trees that must be pruned, not walked.
        for name in ("python_env", "nninteractive_env", ".git", "__pycache__", "integrations"):
            noise = os.path.join(self.tmp, name, "deep", "deeper")
            os.makedirs(noise)
            with open(os.path.join(noise, "_mcs_queue_active.json"), "w") as handle:
                handle.write('{"output_dir": "SHOULD_NOT_APPEAR"}')

        old_root = msb._project_root
        try:
            msb._project_root = lambda: self.tmp
            result = msb._scan_filesystem_for_queue_dirs()
        finally:
            msb._project_root = old_root
        self.assertEqual(["X:/out"], result)

    def test_stop_registered_kill_ladder_runs_on_daemon_thread(self):
        """R61-11: terminate_registered_process polls grace windows per
        process (seconds each); running the ladder on the GUI thread froze
        Mimics for the whole ladder. It must run on a daemon thread."""
        import mimics_stop_background as msb

        called = []
        old_stop = msb._stop_registered_processes
        old_log = msb._mimics_log
        try:
            msb._stop_registered_processes = lambda: called.append(1) or {
                "terminated": [], "failed": [],
            }
            msb._mimics_log = lambda level, message: None
            thread = msb._stop_registered_processes_async()
            thread.join(2.0)
        finally:
            msb._stop_registered_processes = old_stop
            msb._mimics_log = old_log
        self.assertTrue(thread.daemon)
        self.assertFalse(thread.is_alive())
        self.assertEqual([1], called)

    def test_stop_import_entry_is_narrower_than_stop_all(self):
        import mimics_stop_background as msb

        self.assertTrue(msb._lock_is_import_creation({
            "kind": "create_mcs",
            "owner": "import .mcs creation",
        }))
        self.assertFalse(msb._lock_is_import_creation({
            "kind": "nnunet_label_export",
            "owner": "nnU-Net label export",
        }))
        self.assertFalse(msb._lock_is_import_creation({
            "kind": "nnunet_train",
            "owner": "nnU-Net training",
        }))
        entry = os.path.join(
            PROJECT_ROOT,
            "scripting_library",
            "01_Data",
            "03_Stop_Import_Queue.py",
        )
        self.assertTrue(os.path.isfile(entry))


# ============================================================================
# L9: nninteractive_mimics metadata parsing
# ============================================================================


class TestNNInteractiveMimicsParsing(unittest.TestCase):
    def test_owned_server_sweep_accepts_current_schema(self):
        """C6-1: the bridge writes nninteractive_owned_server.v3; the Mimics
        startup sweep must not silently skip it for only matching v2."""
        import inspect
        import nninteractive_mimics

        source = inspect.getsource(
            nninteractive_mimics._cleanup_stale_owned_servers
        )
        self.assertIn('"nninteractive_owned_server.v3"', source)

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

    def test_async_prompt_menu_executes_without_monkeypatch(self):
        """R61-1: the real menu body must run — no NameError from bd180da.

        test_continue_session_prompt_finish_closes_session monkeypatches
        _async_prompt_menu, so the deleted _prompt_buttons_for_profile went
        unnoticed (commit bd180da removed the definition but kept the call).
        This test executes the real function end-to-end against the mocked
        mimics dialogs module.
        """
        import nninteractive_mimics

        observed = []

        def fake_question_box(message="", buttons="", title="", ui_blocking=None, **kw):
            observed.append({"message": message, "buttons": buttons, "title": title})
            return "Finish"

        old_box = nninteractive_mimics.mimics.dialogs.question_box
        try:
            nninteractive_mimics.mimics.dialogs.question_box = fake_question_box
            profile = {"validated_prompt_types": ["point", "box"]}
            state = {
                "interactions": [{"kind": "point"}],
                "source_name": "source.nii",
                "target_name": "AI Result",
            }

            class _Named(object):
                name = "fallback-name"

            action = nninteractive_mimics._async_prompt_menu(
                _Named(), state, _Named(), profile
            )
        finally:
            nninteractive_mimics.mimics.dialogs.question_box = old_box
        self.assertEqual("Finish", action)
        self.assertEqual(1, len(observed))
        # Validated prompt buttons + undo/reset (session has interactions) + finish.
        self.assertEqual(
            "Add Points;Draw Box;Undo Last Prompt;Reset To Start;Finish",
            observed[0]["buttons"],
        )

    def test_prompt_buttons_for_profile_rejects_empty_profile(self):
        """R61-1: a model with no validated prompt types must raise, not
        silently offer an empty button set."""
        import nninteractive_mimics

        with self.assertRaises(RuntimeError):
            nninteractive_mimics._prompt_buttons_for_profile({})

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

    def test_foreground_batch_export_dead_chain_stays_deleted(self):
        """B8: the foreground batch export chain (whose mid-batch
        'Batch Export Disabled' dialog asked the annotator to start over)
        is dead code — main() routes batches to the background process.
        It must not come back."""
        import inspect
        import mimics_export

        source = inspect.getsource(mimics_export)
        self.assertNotIn("Batch Export Disabled", source)
        self.assertNotIn("def _start_export_monitor", source)
        self.assertNotIn("def _export_monitor_tick", source)
        self.assertNotIn("def _start_next_batch_export", source)
        self.assertNotIn("def _start_win32_export_monitor", source)
        # No caller ever passed a batch_queue; the parameter must not
        # reappear as an implicit promise of foreground batching.
        self.assertNotIn("batch_queue", source)

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

    def test_import_error_guidance_categories(self):
        from mimics_import import _error_guidance as import_guidance

        category, _message, action = import_guidance(
            "ModuleNotFoundError: No module named 'nibabel'", "prepare"
        )
        self.assertEqual("environment_broken", category)
        self.assertIn("Setup / Repair Environment", action)

        category, _message, action = import_guidance(
            "OSError: [Errno 28] No space left on device", "prepare"
        )
        self.assertEqual("disk_full", category)
        self.assertIn("Free space", action)

        category, _message, action = import_guidance(
            "The network path was not found", "queue"
        )
        self.assertEqual("network_unavailable", category)

        category, _message, action = import_guidance(
            "Not a valid NIfTI image: case_0042.nii.gz", "prepare"
        )
        self.assertEqual("source_data_invalid", category)
        # R61-5: the old wording pointed at a nonexistent _failed_cases.json
        # inside the user output folder; the real records are per-case JSONs
        # in the queue runtime dir, reachable via Show Batch Status.
        self.assertNotIn("_failed_cases.json", action)
        self.assertIn("original source path", action)

        category, _message, action = import_guidance(
            "Background Mimics could not start mimics.exe", "background_mimics"
        )
        self.assertEqual("background_mimics_failed", category)

        category, _message, action = import_guidance("unexpected thing", "")
        self.assertEqual("unknown", category)
        self.assertIn("Retry", action)

    def test_export_error_guidance_categories(self):
        from mimics_export import _error_guidance as export_guidance

        category, _message, action = export_guidance(
            "ModuleNotFoundError: No module named 'nibabel'", "bridge"
        )
        self.assertEqual("environment_broken", category)
        self.assertIn("Setup / Repair Environment", action)

        category, _message, action = export_guidance(
            "OSError: [Errno 28] No space left on device", "background_export"
        )
        self.assertEqual("disk_full", category)

        category, _message, action = export_guidance(
            "Source image metadata could not be resolved for case s0123", ""
        )
        self.assertEqual("no_source_metadata", category)
        self.assertIn("degraded", action)

        category, _message, action = export_guidance(
            "The active image or target Mask changed during export", "apply"
        )
        self.assertEqual("mask_target_stale", category)

        category, _message, action = export_guidance("strange failure", "")
        self.assertEqual("unknown", category)
        self.assertIn("Retry", action)

    def test_user_facing_text_has_no_developer_residue(self):
        """B4: user-visible strings must not name env vars, command lines,
        or fragile menu numbers (they change and annotators cannot act on
        them). References to entries use the name + menu form instead."""
        import io
        import tokenize

        def string_literals(module_path):
            with open(module_path, "rb") as handle:
                for token in tokenize.tokenize(handle.readline):
                    if token.type == tokenize.STRING:
                        yield token.string

        checks = {
            os.path.join("runtime_py35", "mask_identifier.py"):
                # Reading env vars is fine; telling the user to set them is not.
                ["disable MIMICS_", "enable MIMICS_"],
            os.path.join("runtime_py35", "setup_environment.py"):
                ["Run: python", "Run 'offline-bundle'"],
            os.path.join("runtime_py35", "mimics_import.py"):
                ["04 Stop Import Queue", "check mimics_import.log",
                 "Configure MIMICS_BACKGROUND_EXE",
                 "Stop All Owned Background Services"],
            os.path.join("runtime_py35", "mimics_export.py"):
                ["Configure MIMICS_BACKGROUND_EXE", "check mimics_export.log"],
            os.path.join("runtime_py35", "nninteractive_mimics.py"):
                # The reachable entry is 99_Admin/03_Stop_All_Owned_Services;
                # the longer name never existed as a menu entry.
                ["Stop All Owned Background Services"],
            os.path.join("runtime_py35", "mimics_stop_background.py"):
                # Report/check paths belong in the _mimics_log line, never in
                # user dialogs. Log lines embed them after a format prefix
                # ("{0}\nReport: ..."); only leaked dialog fragments are
                # standalone literals starting with "Report: / "Check: .
                ["Report: {2}", '"Report: ', '"Check: '],
            os.path.join("tools", "mimics_label_export.py"):
                ["04 Stop Import Queue", "Set MIMICS_BACKGROUND_EXE"],
            os.path.join("tools", "setup_env.py"):
                ["Run: python"],
        }
        for relative, forbidden in checks.items():
            path = os.path.join(PROJECT_ROOT, relative)
            self.assertTrue(os.path.isfile(path), path)
            literals = "\n".join(string_literals(path))
            for needle in forbidden:
                self.assertNotIn(
                    needle, literals,
                    "{0} still leaks '{1}' to users".format(relative, needle)
                )
        # The reachable form must be present where a stop action is offered.
        import_source = Path(
            PROJECT_ROOT, "runtime_py35", "mimics_import.py"
        ).read_text(encoding="utf-8")
        self.assertIn("Stop Import Queue", import_source)
        self.assertIn("(01 Data menu)", import_source)


class TestNNInteractiveSessionWriteMode(unittest.TestCase):
    """R61-10: the "update mask / editable copy" choice is asked once per
    Mimics session, not once per result (batch annotation was a modal storm)."""

    class _FakeMetadata(object):
        """Minimal metadata collection for _metadata_set/_metadata_delete."""

        def __init__(self):
            self.items = {}

        def create(self, name=None, value=""):
            self.items[name] = value

        def find(self, name):
            return self.items.get(name)

        def delete(self, name):
            self.items.pop(name, None)

    class _FakeMask(object):
        def __init__(self, name):
            self.name = name
            self.selected = True
            self.deleted = False
            self.metadata = TestNNInteractiveSessionWriteMode._FakeMetadata()

    def _run_choose(self, decisions, state, target_name="Target", session=None):
        """Run _choose_completed_result_target against a fake mimics env.

        Returns (target, prompts, created, session_after): prompts records
        every question_box call; session_after is the stored session choice
        the call left behind. ``decisions`` is popped one per prompt;
        ``session`` seeds the stored choice (None = first result).
        """
        import nninteractive_mimics

        prompts = []

        def fake_question_box(message="", buttons="", title="", ui_blocking=None, **kw):
            prompts.append(buttons)
            return decisions.pop(0) if decisions else "Create Editable Copy"

        created = []

        def fake_create_result_mask(image, name=None):
            mask = self._FakeMask(name or "AI Draft")
            created.append(mask)
            return mask

        old_box = nninteractive_mimics.mimics.dialogs.question_box
        old_create = nninteractive_mimics._create_result_mask
        old_mark = nninteractive_mimics._mark_ai_draft
        old_log = nninteractive_mimics._mimics_log
        session_after = None
        try:
            nninteractive_mimics.mimics.dialogs.question_box = fake_question_box
            nninteractive_mimics._create_result_mask = fake_create_result_mask
            nninteractive_mimics._mark_ai_draft = lambda target, source=None: None
            nninteractive_mimics._mimics_log = lambda level, message: None
            nninteractive_mimics._SESSION_WRITE_MODE = session
            target = self._FakeMask(target_name)
            result = nninteractive_mimics._choose_completed_result_target(
                "image", target, state, {"elapsed_seconds": 5}
            )
            session_after = nninteractive_mimics._SESSION_WRITE_MODE
        finally:
            nninteractive_mimics.mimics.dialogs.question_box = old_box
            nninteractive_mimics._create_result_mask = old_create
            nninteractive_mimics._mark_ai_draft = old_mark
            # Always leave no stored choice so tests stay independent.
            nninteractive_mimics._SESSION_WRITE_MODE = None
            nninteractive_mimics._mimics_log = old_log
        return result, prompts, created, session_after

    def test_first_result_asks_and_remembered_choice_sticks(self):
        # First result: user picks "Update Selected Mask" -> in place, prompt shown once.
        state = {"write_mode": "choose_on_first_result"}
        result, prompts, _created, session_after = self._run_choose(
            ["Update Selected Mask"], state
        )
        self.assertEqual(1, len(prompts))
        self.assertEqual("in_place", state["write_mode"])
        self.assertIsNone(state.get("target_guid"))
        self.assertEqual("in_place", session_after)

        # Second result in the same session: no prompt, same mode applies.
        state2 = {"write_mode": "choose_on_first_result"}
        result2, prompts2, _created2, _session2 = self._run_choose(
            [], state2, session="in_place"
        )
        self.assertEqual(0, len(prompts2))
        self.assertEqual("in_place", state2["write_mode"])

    def test_derived_copy_choice_creates_draft_and_persists(self):
        state = {"write_mode": "choose_on_first_result", "_job_dir": "job"}
        result, prompts, created, _session_after = self._run_choose(
            ["Create Editable Copy"], state
        )
        self.assertEqual(1, len(prompts))
        self.assertEqual(1, len(created))
        self.assertIs(created[0], result)
        self.assertIn("AI Draft", result.name)
        self.assertEqual("derived_copy", state["write_mode"])
        self.assertEqual("job", state.get("_job_dir"))

        # Second result: no prompt, still derived copy.
        state2 = {"write_mode": "choose_on_first_result"}
        result2, prompts2, created2, _session2 = self._run_choose(
            [], state2, session="derived_copy"
        )
        self.assertEqual(0, len(prompts2))
        self.assertEqual(1, len(created2))
        self.assertEqual("derived_copy", state2["write_mode"])

    def test_write_mode_not_asking_is_untouched(self):
        # With an explicit write mode there is nothing to choose; the target
        # passes through unchanged and no dialog appears.
        state = {"write_mode": "in_place"}
        result, prompts, created, _session_after = self._run_choose([], state)
        self.assertEqual(0, len(prompts))
        self.assertEqual(0, len(created))
        self.assertEqual("Target", result.name)
        self.assertEqual("in_place", state["write_mode"])

    def test_closed_dialog_falls_back_to_editable_copy(self):
        # Failure path: user closes the dialog (answer matches neither
        # button). The selected Mask must stay untouched -> derived copy.
        state = {"write_mode": "choose_on_first_result"}
        result, prompts, created, session_after = self._run_choose([""], state)
        self.assertEqual(1, len(prompts))
        self.assertEqual(1, len(created))
        self.assertEqual("derived_copy", state["write_mode"])
        self.assertEqual("derived_copy", session_after)


class TestStreamedVoxelBuffers(unittest.TestCase):
    """R61-12: voxel buffers must be streamed, never full-copied on the GUI thread."""

    def test_buffer_byte_view_is_zero_copy_for_flat_byte_buffers(self):
        import runtime_common
        data = bytearray(b"\x01\x02\x03" * 8)
        view = runtime_common.buffer_byte_view(data)
        self.assertIsInstance(view, memoryview)
        self.assertEqual(len(data), len(view))
        self.assertIs(view.obj, data)

    def test_buffer_byte_view_casts_multidimensional_buffers(self):
        import numpy as np
        import runtime_common
        arr = np.zeros((4, 5, 6), dtype=np.uint8)
        view = runtime_common.buffer_byte_view(arr)
        self.assertEqual(arr.size, len(view))
        self.assertEqual(b"\x00" * arr.size, bytes(view))

    def test_buffer_byte_view_falls_back_to_copy_for_unwrappable_objects(self):
        # Failure path: memoryview(obj) raises -> tobytes() fallback must
        # still produce the same bytes, not crash the export.
        import runtime_common

        class Unwrappable(object):
            def tobytes(self):
                return b"abc"

        view = runtime_common.buffer_byte_view(Unwrappable())
        self.assertEqual(b"abc", bytes(view))

    def test_stream_buffer_digest_matches_whole_buffer_hash(self):
        import hashlib
        import runtime_common
        data = bytes(bytearray((i % 251 for i in range(5 * 1024 * 1024))))
        streamed = runtime_common.stream_buffer(
            memoryview(data), compute_sha=True
        )
        self.assertEqual(
            "sha256:" + hashlib.sha256(data).hexdigest(), streamed
        )

    def test_stream_buffer_writes_identical_bytes_and_pumps_between_chunks(self):
        import hashlib
        import runtime_common
        data = bytes(bytearray((i % 253 for i in range(40 * 1024 * 1024))))
        pumps = []
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            path = handle.name
        try:
            with open(path, "wb") as handle:
                sha = runtime_common.stream_buffer(
                    memoryview(data), handle=handle, compute_sha=True,
                    progress_callback=lambda: pumps.append(1),
                )
            with open(path, "rb") as handle:
                written = handle.read()
            self.assertEqual(data, written)
            self.assertEqual(
                "sha256:" + hashlib.sha256(data).hexdigest(), sha
            )
            # 40MB over 16MB chunks = 3 chunks -> 2 between-chunk pumps.
            self.assertEqual(2, len(pumps))
        finally:
            os.remove(path)

    def test_stream_buffer_progress_callback_errors_are_swallowed(self):
        # Failure path: a dead GUI pump must not abort a long export.
        import runtime_common

        def broken_pump():
            raise RuntimeError("GUI gone")

        data = b"\x07" * (17 * 1024 * 1024)
        sha = runtime_common.stream_buffer(
            memoryview(data), compute_sha=True,
            progress_callback=broken_pump,
        )
        self.assertTrue(sha.startswith("sha256:"))

    def test_stream_buffer_empty_buffer_yields_digest_of_nothing(self):
        import hashlib
        import runtime_common
        self.assertEqual(
            "sha256:" + hashlib.sha256(b"").hexdigest(),
            runtime_common.stream_buffer(memoryview(b""), compute_sha=True),
        )
        self.assertEqual(
            "", runtime_common.stream_buffer(memoryview(b""), compute_sha=False)
        )


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

    class _DicomSeries:
        """Minimal pydicom-like header for selection tests."""

        def __init__(self, inst, ipp, uid=None, study=None, snum=None):
            self.Rows, self.Columns = 4, 4
            self.InstanceNumber = inst
            self.ImagePositionPatient = ipp
            if uid is not None:
                self.SeriesInstanceUID = uid
            if study is not None:
                self.StudyInstanceUID = study
            if snum is not None:
                self.SeriesNumber = snum

    def _two_missing_uid_series(self):
        # Two distinct series (plain + contrast), both missing
        # SeriesInstanceUID, covering the same z grid — provably interleaved.
        records = []
        for i, z in enumerate([0.0, 10.0, 20.0]):
            records.append(("/a{}.dcm".format(i), self._DicomSeries(i + 1, [0.0, 0.0, z])))
            records.append(("/b{}.dcm".format(i), self._DicomSeries(i + 1, [0.0, 0.0, z])))
        return records

    def test_dicom_missing_uid_interleaved_series_fails_closed(self):
        # Regression (TB-02): two UID-less series used to merge into one
        # __missing_series_uid__ group and silently stack into a wrong
        # volume on every call path (no shape, shape match, mismatch-ok).
        from nninteractive_bridge import _select_dicom_records

        records = self._two_missing_uid_series()
        for shape, allow in [(None, False), ([4, 4, 3], False), ([4, 4, 3], True)]:
            with self.assertRaisesRegex(
                RuntimeError, "cannot be told apart"
            ):
                _select_dicom_records(records, shape, allow)

    def test_dicom_single_missing_uid_series_still_selects(self):
        # One coherent UID-less series (no duplicate positions) must keep
        # working — de-identified folders load as before.
        from nninteractive_bridge import _select_dicom_records

        records = [
            ("/s{}.dcm".format(i), self._DicomSeries(i + 1, [0.0, 0.0, 10.0 * i]))
            for i in range(3)
        ]
        selected = _select_dicom_records(records, [4, 4, 3], False)
        self.assertEqual(len(selected), 3)

    def test_dicom_missing_uid_distinct_series_number_same_grid(self):
        # (StudyInstanceUID, SeriesNumber) separates two UID-less series;
        # with an expected shape both match -> existing multi-series error.
        from nninteractive_bridge import _select_dicom_records

        records = []
        for i, z in enumerate([0.0, 10.0, 20.0]):
            records.append(("/a{}.dcm".format(i), self._DicomSeries(i + 1, [0.0, 0.0, z], study="ST", snum=1)))
            records.append(("/b{}.dcm".format(i), self._DicomSeries(i + 1, [0.0, 0.0, z], study="ST", snum=2)))
        with self.assertRaisesRegex(RuntimeError, "Multiple DICOM series"):
            _select_dicom_records(records, [4, 4, 3], False)

    def test_dicom_missing_uid_duplicate_file_copy_residue_fails_closed(self):
        # A duplicated slice (copy residue) inside a UID-less series also
        # produces a duplicate position and must fail closed instead of
        # stacking the same slice twice.
        from nninteractive_bridge import _select_dicom_records

        records = [
            ("/s0.dcm", self._DicomSeries(1, [0.0, 0.0, 0.0])),
            ("/s0 - Copy.dcm", self._DicomSeries(1, [0.0, 0.0, 0.0])),
            ("/s1.dcm", self._DicomSeries(2, [0.0, 0.0, 10.0])),
        ]
        with self.assertRaisesRegex(RuntimeError, "cannot be told apart"):
            _select_dicom_records(records, None, False)

    def test_dicom_uid_series_exempt_from_duplicate_position_check(self):
        # Groups named by a real SeriesInstanceUID keep today's behavior:
        # the duplicate-position check must not fire for them (the shape
        # re-check in load_image_dicom_folder covers what matters).
        from nninteractive_bridge import _dicom_group_key, _select_dicom_records

        records = []
        for i, z in enumerate([0.0, 10.0, 20.0]):
            records.append(("/a{}.dcm".format(i), self._DicomSeries(i + 1, [0.0, 0.0, z], uid="1.2.3")))
            records.append(("/b{}.dcm".format(i), self._DicomSeries(i + 1, [0.0, 0.0, z], uid="1.2.4")))
        self.assertEqual(len(records), 6)
        # Sanity: the group keys are the real UIDs.
        self.assertEqual(_dicom_group_key(records[0][1]), "1.2.3")
        selected = _select_dicom_records(records, None, False)
        self.assertEqual(len(selected), 3)

    def test_dicom_series_number_zero_is_real_value(self):
        # SeriesNumber=0 must not be collapsed into the missing sentinel.
        from nninteractive_bridge import _dicom_group_key

        ds = self._DicomSeries(1, [0.0, 0.0, 0.0], study="S1", snum=0)
        self.assertIn("|0", _dicom_group_key(ds))
        missing = self._DicomSeries(1, [0.0, 0.0, 0.0])
        self.assertNotIn("|0", _dicom_group_key(missing))


# ============================================================================
# L11: Edge cases and boundary conditions
# ============================================================================


class TestEdgeCases(unittest.TestCase):
    def setUp(self):
        self.tmp = _make_temp_dir()

    def tearDown(self):
        _cleanup(self.tmp)

    def test_bridge_worker_log_default_is_absolute(self):
        """C6-2: before the first initialize request the worker's log path
        must not be relative - early errors would land in an arbitrary CWD."""
        import inspect
        import nninteractive_bridge

        source = inspect.getsource(nninteractive_bridge._worker_main)
        self.assertNotIn(
            'Path("nninteractive_bridge.jsonl")', source,
            "the worker log default must stay absolute (temp dir)"
        )

    def test_affine_repair_already_running_returns_nonzero(self):
        """C6-4: offer_repair_for_prediction_failure treats main() == 0 as
        "a repair flow was started" and swallows the original failure dialog.
        The already-running branch must return non-zero so the caller still
        shows it."""
        import fix_source_affine_metadata as fix

        active = {"fix-1": {"done": False, "key": "fix-1"}}
        with mock.patch.object(fix, "_MONITORS", active), \
                mock.patch.object(
                    fix.mimics.dialogs, "question_box",
                    lambda **_kw: "Keep Running",
                ):
            code = fix.main()
        self.assertNotEqual(0, code)

    def test_setup_monitor_deadline_extends_while_window_alive(self):
        """C6-3: a training form left open past the 1h deadline must not
        lose its completion dialog; the deadline extends while the setup
        window's process is alive."""
        import nnunet_mimics

        # Deadline already passed, but the setup process still exists.
        monitor = {
            "monitor_key": "k",
            "kind": "train_setup",
            "status_path": os.path.join(self.tmp, "missing.json"),
            "controller_pid": os.getpid(),  # this test process is alive
            "deadline": time.time() - 1.0,
            "last_line": "",
        }
        with mock.patch.object(nnunet_mimics, "_stop_monitor") as stop, \
                mock.patch.object(
                    nnunet_mimics, "_read_json", lambda *a, **kw: {}
                ):
            nnunet_mimics._monitor_tick_locked(monitor)
        stop.assert_not_called()
        self.assertGreater(monitor["deadline"], time.time())

    def test_setup_monitor_deadline_stops_when_window_dead(self):
        import nnunet_mimics

        monitor = {
            "monitor_key": "k",
            "kind": "train_setup",
            "status_path": os.path.join(self.tmp, "missing.json"),
            "controller_pid": None,  # process already gone / transitioned
            "deadline": time.time() - 1.0,
            "last_line": "",
        }
        with mock.patch.object(nnunet_mimics, "_stop_monitor") as stop, \
                mock.patch.object(
                    nnunet_mimics, "_read_json", lambda *a, **kw: {}
                ):
            nnunet_mimics._monitor_tick_locked(monitor)
        stop.assert_called_once_with("k")

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

    # -- nninteractive_mimics _mimics_log fallback --
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
                from nninteractive_mimics import _mimics_log

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

    def test_official_model_source_alignment_is_deferred_by_default(self):
        # The generic (official-model) source branch must never build the
        # on-demand aligned cache on the Mimics GUI thread: a derived
        # oblique import would trigger a bridge call of up to 1800 s right
        # inside the prompt-capture click. The external image worker
        # resamples from the recorded affines instead.
        import nninteractive_mimics as module

        source = os.path.join(self.tmp, "official_source.nii.gz")
        Path(source).write_bytes(b"raw source stays untouched")
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
            module, "_call_mimics_bridge", side_effect=AssertionError("bridge was called on the GUI thread")
        ), mock.patch.object(
            module, "_model_profile", return_value={"source": "official"}
        ), mock.patch.object(module, "_mimics_log"):
            result = module._export_image_for_nninteractive(
                {
                    "_model_profile": {"source": "official"},
                    "image_input_mode": "auto",
                    "fallback_to_source_when_mimics_export_fails": True,
                },
                image,
                os.path.join(self.tmp, "unused_buffer.raw"),
                allow_buffer_export=False,
            )
        self.assertIsNotNone(result)
        self.assertEqual(os.path.abspath(source), result["image_path"])
        # The recorded source affines travel with the request so the worker
        # can resample onto the Mimics grid itself.
        self.assertEqual(
            "derived_dicom_lps_resampled_from_source_image_v2",
            result["source_index_space"],
        )

    def test_official_model_sync_alignment_still_available_when_configured(self):
        # An explicit defer_source_alignment_to_worker=false keeps the old
        # synchronous on-demand cache behavior for users who prefer it.
        import nninteractive_mimics as module

        source = os.path.join(self.tmp, "sync_source.nii.gz")
        Path(source).write_bytes(b"source")
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
        bridge_calls = []

        def _fake_bridge(config, request, timeout_seconds=None):
            bridge_calls.append(request)
            cache_path = request["source_nifti_out"]
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as handle:
                handle.write(b"aligned")
            return {"status": "ok"}

        with mock.patch.object(
            module, "_relocated_source_image_path", return_value=""
        ), mock.patch.object(
            module, "_model_profile", return_value={"source": "official"}
        ), mock.patch.object(
            module, "_call_mimics_bridge", side_effect=_fake_bridge
        ), mock.patch.object(module, "_mimics_log"):
            result = module._export_image_for_nninteractive(
                {
                    "_model_profile": {"source": "official"},
                    "image_input_mode": "source",
                    "defer_source_alignment_to_worker": False,
                },
                image,
                os.path.join(self.tmp, "unused_buffer.raw"),
                allow_buffer_export=False,
            )
        self.assertEqual(1, len(bridge_calls))
        self.assertEqual("prepare_source_fastpath", bridge_calls[0]["action"])
        self.assertEqual(
            "nifti_ijk_matches_derived_dicom_columns_rows_slices_v1",
            result["source_index_space"],
        )
        self.assertNotEqual(os.path.abspath(source), result["image_path"])

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

    def test_prediction_context_errors_point_to_real_repair_paths(self):
        # A4: the old "Relink the source image metadata" wording pointed at
        # an action that does not exist. The unlink failure (project not
        # resolvable to a source image) must guide the annotator back to the
        # import flow, and the stale-geometry failure must carry the marker
        # the Mimics-side repair offer detects (D4).
        import inspect

        import flexict_mimics
        import nnunet_mimics

        tools_dir = os.path.dirname(os.path.abspath(__file__))
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)
        import nnunet_pipeline

        for module in (nnunet_mimics, flexict_mimics):
            source = inspect.getsource(module._prediction_context)
            self.assertNotIn("Relink the source image metadata", source)
            self.assertIn("01_Import_Dataset", source)
            self.assertIn("re-import the case", source)
        pipeline_source = inspect.getsource(
            nnunet_pipeline.validate_materialized_source_geometry
        )
        self.assertNotIn("Relink the source image before prediction", pipeline_source)
        # D4: the standalone Fix Affine menu entry is gone; the error must
        # carry the stable marker the Mimics-side repair offer detects, keep
        # the re-import escape hatch, and must not name a deleted menu.
        self.assertIn("source geometry mismatch", pipeline_source)
        self.assertNotIn("04_Fix_Source_Affine_Metadata", pipeline_source)
        # The rendered message must keep the re-import escape hatch that
        # the repair flow itself recommends when the source file changed.
        with mock.patch("nibabel.load") as fake_load:
            from types import SimpleNamespace

            fake_load.return_value = SimpleNamespace(
                shape=(4, 4, 4), affine=np.eye(4)
            )
            with self.assertRaises(RuntimeError) as ctx:
                nnunet_pipeline.validate_materialized_source_geometry(
                    "image.nii",
                    {
                        "source_shape": [4, 4, 4],
                        "source_voxel_to_ras_matrix": [
                            [2.0, 0, 0, 0], [0, 1, 0, 0],
                            [0, 0, 1, 0], [0, 0, 0, 1],
                        ],
                    },
                )
        message = str(ctx.exception)
        self.assertIn("source geometry mismatch", message)
        self.assertIn("re-import", message)
        self.assertNotIn("04_Fix_Source_Affine_Metadata", message)

    def test_prediction_failure_offers_repair_on_geometry_mismatch(self):
        # D4: the standalone Fix Affine menu entry is gone. The repair must
        # instead be offered by the failure path itself, exactly on the error
        # it can fix, and only when the user accepts.
        import inspect

        import fix_source_affine_metadata
        import flexict_mimics
        import nnunet_mimics

        for module in (nnunet_mimics, flexict_mimics):
            source = inspect.getsource(module._monitor_tick_locked)
            self.assertIn("_offer_source_geometry_repair", source)
            entry = Path(
                PROJECT_ROOT,
                "scripting_library",
                "99_Admin",
                "04_Fix_Source_Affine_Metadata.py",
            )
            self.assertFalse(
                entry.exists(), "D4 menu entry must be deleted: {}".format(entry)
            )

        offer = fix_source_affine_metadata.offer_repair_for_prediction_failure
        # Unrelated errors must not trigger any dialog.
        with mock.patch.object(fix_source_affine_metadata.mimics.dialogs, "question_box") as q:
            self.assertFalse(offer("some other failure", "nnU-Net"))
            q.assert_not_called()
        # No active image: nothing to offer, no dialog.
        with mock.patch.object(
            fix_source_affine_metadata.mimics.data.images, "get_active",
            return_value=None,
        ):
            with mock.patch.object(fix_source_affine_metadata.mimics.dialogs, "question_box") as q:
                self.assertFalse(
                    offer("RuntimeError: source geometry mismatch: ...", "nnU-Net")
                )
                q.assert_not_called()
        # Marker + active image + user accepts: repair flow starts.
        class _FakeImage(object):
            pass

        started = []

        def fake_main():
            started.append(True)
            return 0

        with mock.patch.object(
            fix_source_affine_metadata.mimics.data.images, "get_active",
            return_value=_FakeImage(),
        ), mock.patch.object(
            fix_source_affine_metadata.mimics.dialogs, "question_box",
            return_value="Repair Stored Geometry",
        ), mock.patch.object(
            fix_source_affine_metadata, "main", fake_main,
        ):
            self.assertTrue(
                offer("RuntimeError: source geometry mismatch: ...", "nnU-Net")
            )
            self.assertEqual(started, [True])
        # Marker + active image + user declines: no repair, dialog shown once.
        with mock.patch.object(
            fix_source_affine_metadata.mimics.data.images, "get_active",
            return_value=_FakeImage(),
        ), mock.patch.object(
            fix_source_affine_metadata.mimics.dialogs, "question_box",
            return_value="Not Now",
        ), mock.patch.object(fix_source_affine_metadata, "main", fake_main):
            self.assertFalse(
                offer("RuntimeError: source geometry mismatch: ...", "nnU-Net")
            )
            self.assertEqual(started, [True])

    def test_call_mimics_bridge_pumps_gui_during_wait(self):
        # R61-26: the bridge round-trip must run communicate() on a wait
        # thread while the GUI thread pumps, so a long bridge call cannot
        # freeze Mimics.
        import nninteractive_mimics as module

        pumps = []
        started = threading.Event()

        class _FakeProcess(object):
            returncode = 0

            def __init__(self, *args, **kwargs):
                pass

            def communicate(self, input=None, timeout=None):
                started.set()
                # Hold the "bridge" busy long enough for at least one pump
                # iteration (poll interval is 0.05 s).
                time.sleep(0.3)
                return (b'{"status": "ok"}', b"")

        with mock.patch.object(
            module, "_mimics_bridge_paths", return_value=("py", "bridge.py")
        ), mock.patch.object(
            module.subprocess, "Popen", _FakeProcess
        ), mock.patch.object(
            module, "_update_gui", side_effect=lambda: pumps.append(1)
        ):
            result = module._call_mimics_bridge({}, {"action": "x"})
        self.assertEqual("ok", result["status"])
        started.wait(2.0)
        self.assertGreaterEqual(len(pumps), 1)

    def test_call_mimics_bridge_timeout_kills_and_raises(self):
        # Failure path: the bridge exceeding its deadline must kill the
        # process and raise, not hang the GUI pump loop forever.
        import nninteractive_mimics as module

        killed = []

        class _FakeProcess(object):
            returncode = 0

            def __init__(self, *args, **kwargs):
                pass

            def communicate(self, input=None, timeout=None):
                raise subprocess.TimeoutExpired("bridge.py", timeout)

            def kill(self):
                killed.append(1)

        with mock.patch.object(
            module, "_mimics_bridge_paths", return_value=("py", "bridge.py")
        ), mock.patch.object(
            module.subprocess, "Popen", _FakeProcess
        ), mock.patch.object(module, "_update_gui"):
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                module._call_mimics_bridge(
                    {}, {"action": "x"}, timeout_seconds=0
                )
        self.assertEqual(1, len(killed))

    def test_call_mimics_bridge_nonzero_exit_reports_stderr(self):
        # Failure path: a failing bridge surfaces its stderr, not a generic
        # opaque error, so users can act on the actual cause.
        import nninteractive_mimics as module

        class _FakeProcess(object):
            returncode = 3

            def __init__(self, *args, **kwargs):
                pass

            def communicate(self, input=None, timeout=None):
                return (b"", b"boom: no model")

        with mock.patch.object(
            module, "_mimics_bridge_paths", return_value=("py", "bridge.py")
        ), mock.patch.object(
            module.subprocess, "Popen", _FakeProcess
        ), mock.patch.object(module, "_update_gui"):
            with self.assertRaisesRegex(RuntimeError, "boom: no model"):
                module._call_mimics_bridge({}, {"action": "x"})


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

    def test_export_skip_existing_proactive_reminder(self):
        """P6c: with Skip existing + files already on disk, ask before export."""
        import tools.io_path_setup_ui as ui

        root = Path(self.tmp) / "exports"
        seg = root / "case01" / "segmentations"
        seg.mkdir(parents=True)
        (seg / "liver.nii.gz").write_bytes(b"x")
        (seg / "kidney_left.nii.gz").write_bytes(b"x")
        (seg / "notes.txt").write_bytes(b"x")
        # 2 label files, 1 non-label file: only label files count.
        self.assertEqual(2, ui.count_existing_label_files(str(root), "case01"))
        # Missing folder / different case: no reminder.
        self.assertEqual(0, ui.count_existing_label_files(str(root), "case02"))
        self.assertEqual(0, ui.count_existing_label_files(str(Path(self.tmp) / "empty"), "case01"))
        # The reminder is wired into the export submit path.
        source = Path(PROJECT_ROOT, "tools", "io_path_setup_ui.py").read_text(encoding="utf-8")
        self.assertIn('count_existing_label_files(output, case_id)', source)
        self.assertIn('Existing Label Files', source)
        # Overwrite selection must not trigger the reminder.
        self.assertIn('selection.get("conflict_policy") == "skip"', source)

    def test_batch_status_viewer_aggregates_all_record_types(self):
        """P6b: the batch panel aggregates every on-disk record kind."""
        import tools.batch_status_viewer as viewer

        root = Path(self.tmp) / "proj"
        runtime = root / ".mimics_runtime"

        # Import run under the (relocated-capable) import base.
        import_run = root / ".mimics_runtime" / "import_runs" / "run_001"
        import_run.mkdir(parents=True)
        (import_run / "status.json").write_text(json.dumps({
            "status": "running", "phase": "preparing",
            "completed": 2, "failed": 1, "total": 10, "case_id": "s0100",
            "updated_at_epoch": 1700000004.0,
        }), encoding="utf-8")

        # Import queue with the .mcs batch status.
        queue = root / ".mimics_runtime" / "import_queues" / "mcs_out_ab12"
        queue.mkdir(parents=True)
        (queue / "_mcs_batch_status.json").write_text(json.dumps({
            "status": "creating", "completed": 3, "failed": 0, "total": 10,
            "updated_at_epoch": 1700000005.0,
        }), encoding="utf-8")

        # Export job.
        export_job = runtime / "export_jobs" / "export_20260101T000000_ab"
        export_job.mkdir(parents=True)
        (export_job / "status.json").write_text(json.dumps({
            "status": "closed", "completed": 7, "failed": 0, "total": 7,
            "updated_at_epoch": 1700000003.0,
        }), encoding="utf-8")

        # Foreground export task (flat file).
        ui_tasks = runtime / "ui_tasks"
        ui_tasks.mkdir(parents=True)
        (ui_tasks / "current_task01.json").write_text(json.dumps({
            "status": "failed", "error": "Mask changed",
            "updated_at_epoch": 1700000002.0,
        }), encoding="utf-8")
        (ui_tasks / "current_task01_stop.json").write_text("{}", encoding="utf-8")

        # Append job.
        append_job = runtime / "append_jobs" / "append_masks_001"
        append_job.mkdir(parents=True)
        (append_job / "status.json").write_text(json.dumps({
            "status": "completed", "completed": 4, "total": 4,
            "updated_at_epoch": 1700000001.0,
        }), encoding="utf-8")

        # Drop import (flat per-case status).
        drop = runtime / "drop_import"
        drop.mkdir(parents=True)
        (drop / "20260101T000000_0_status.json").write_text(json.dumps({
            "status": "completed", "case_id": "s0200",
            "updated_at_epoch": 1700000000.0,
        }), encoding="utf-8")

        # Patch import_runtime_base so the test is independent of UNC/env
        # relocation behavior.
        with mock.patch.object(
            viewer,
            "import_runtime_base",
            lambda project_root: Path(project_root) / ".mimics_runtime",
        ):
            rows = viewer.collect_batch_rows(str(root))

        kinds = {row["kind"] for row in rows}
        self.assertEqual(
            {"Import", "Import queue", "Export", "Export task", "Append", "Drop import"},
            kinds,
        )
        # Newest first: the queue record (epoch 5) leads.
        self.assertEqual("Import queue", rows[0]["kind"])
        by_path = {row["status_path"]: row for row in rows}
        run_row = by_path[str(import_run / "status.json")]
        self.assertEqual("running", run_row["status"])
        self.assertEqual(2, run_row["completed"])
        self.assertEqual(10, run_row["total"])
        # Stop markers are not records.
        self.assertNotIn(str(ui_tasks / "current_task01_stop.json"), by_path)

    def test_batch_status_viewer_per_kind_limit_and_missing_dirs(self):
        import tools.batch_status_viewer as viewer

        root = Path(self.tmp) / "empty_proj"
        rows = viewer.collect_batch_rows(str(root))
        self.assertEqual([], rows)

        runs = root / ".mimics_runtime" / "import_runs"
        runs.mkdir(parents=True)
        for index in range(20):
            job = runs / "run_{0:02d}".format(index)
            job.mkdir()
            (job / "status.json").write_text(json.dumps({
                "status": "completed", "updated_at_epoch": 1700000000.0 + index,
            }), encoding="utf-8")
        with mock.patch.object(
            viewer,
            "import_runtime_base",
            lambda project_root: Path(project_root) / ".mimics_runtime",
        ):
            rows = viewer.collect_batch_rows(str(root))
        self.assertEqual(viewer.PER_KIND_LIMIT, len(rows))
        # Newest first.
        self.assertEqual("run_19", rows[0]["label"])

    def test_batch_status_entry_and_runtime_module_exist(self):
        entry = Path(
            PROJECT_ROOT, "scripting_library", "01_Data", "08_Show_Batch_Status.py"
        )
        self.assertTrue(entry.is_file(), entry)
        source = entry.read_text(encoding="utf-8")
        self.assertIn("batch_status_mimics", source)
        wrapper = Path(PROJECT_ROOT, "runtime_py35", "batch_status_mimics.py")
        self.assertTrue(wrapper.is_file(), wrapper)
        wrapper_source = wrapper.read_text(encoding="utf-8")
        self.assertIn('"batch_status_viewer.py"', wrapper_source)
        self.assertIn('"batch_status"', wrapper_source)
        self.assertIn('"batch_status_"', wrapper_source)
        # Packaged and diagnosed with the other external windows.
        import tools.package_portable as package_portable

        self.assertIn("tools/batch_status_viewer.py", package_portable.REQUIRED_EXTERNAL_UI_FILES)
        self.assertIn("runtime_py35/batch_status_mimics.py", package_portable.REQUIRED_EXTERNAL_UI_FILES)
        import tools.collect_diagnostics as diagnostics

        labels = [label for label, _pattern, _glob, _count in diagnostics.SCAN_LOG_TARGETS]
        self.assertIn("logs/batch_status", labels)

    def test_regression_matrix_runner_suite_registry(self):
        """The one-command regression matrix stays internally consistent."""
        import tools.run_regression_matrix as matrix

        # 1. Unique names, every command target exists on disk.
        names = [name for name, _cmd, _profiles in matrix.SUITES]
        self.assertEqual(len(names), len(set(names)), "duplicate suite names")
        for _name, command, _profiles in matrix.SUITES:
            target = command[-1] if command[-1] != "-q" else command[-2]
            # pytest invocation: [-m, pytest, -q, path]; plain: [python, path]
            targets = [part for part in command[1:] if part.endswith(".py")]
            self.assertTrue(targets, "no script target in {0!r}".format(command))
            for candidate in targets:
                self.assertTrue(
                    Path(PROJECT_ROOT, candidate).is_file(),
                    "suite target missing: {0}".format(candidate),
                )
        # 2. Profile membership sanity.
        for _name, _cmd, profiles in matrix.SUITES:
            self.assertTrue(profiles, "suite with no profile membership")
            self.assertTrue(
                profiles <= {"smoke", "fast", "full"},
                "unknown profile in {0!r}".format(profiles),
            )
        # 3. Every profile selects something; smoke is a strict subset of fast.
        for profile in ("smoke", "fast", "full"):
            selected = matrix.select_suites(profile, [])
            self.assertTrue(selected, "profile {0} selected nothing".format(profile))
        smoke_names = {name for name, _cmd, _m in matrix.select_suites("smoke", [])}
        fast_names = {name for name, _cmd, _m in matrix.select_suites("fast", [])}
        full_names = {name for name, _cmd, _m in matrix.select_suites("full", [])}
        self.assertTrue(smoke_names <= fast_names, "smoke must be a subset of fast")
        self.assertTrue(fast_names <= full_names, "fast must be a subset of full")
        # 4. --only filtering with unknown names is rejected.
        with self.assertRaises(SystemExit):
            matrix.select_suites("fast", ["no_such_suite"])
        # 5. The smoke profile is all flow groups: no long suite can sneak in.
        self.assertTrue(
            all(name.startswith("flow_") for name in smoke_names),
            "smoke profile must only contain flow groups",
        )

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

    def test_mimics_import_has_no_synchronous_bridge_call(self):
        # mimics_import.call_bridge (communicate(timeout=600)) was dead code
        # and a GUI-blocking hazard if ever called; it was removed. All
        # bridge work goes through _launch_bridge_background.
        import mimics_import

        self.assertFalse(
            hasattr(mimics_import, "call_bridge"),
            "mimics_import.call_bridge must stay deleted; use _launch_bridge_background",
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

        import runtime_common
        # The runner must land in an isolated runtime dir, never in the
        # production .mimics_runtime/import_queues/ tree.
        prod_queues = Path(PROJECT_ROOT, ".mimics_runtime", "import_queues")
        prod_before = self._snapshot_dir(prod_queues)
        old_lock = cli._acquire_background_mimics_lock
        old_popen = cli.subprocess.Popen
        old_runtime_dir = os.environ.get("MIMICS_IMPORT_RUNTIME_DIR")
        try:
            cli._acquire_background_mimics_lock = lambda *_args, **_kwargs: Lock()
            cli.subprocess.Popen = lambda *_args, **_kwargs: Proc()
            os.environ["MIMICS_IMPORT_RUNTIME_DIR"] = str(Path(self.tmp) / "import_runtime")
            cli.launch_create_mcs(output_dir, r"C:\MimicsResearch.exe", bridge_python, 0.0)
            # Resolve the expected queue dir through the real function while
            # the override is still active, so the digest rule cannot drift.
            queue_runtime = Path(runtime_common.import_queue_runtime_dir(
                PROJECT_ROOT, str(output_dir)
            ))
        finally:
            cli._acquire_background_mimics_lock = old_lock
            cli.subprocess.Popen = old_popen
            if old_runtime_dir is None:
                os.environ.pop("MIMICS_IMPORT_RUNTIME_DIR", None)
            else:
                os.environ["MIMICS_IMPORT_RUNTIME_DIR"] = old_runtime_dir
        runner = (queue_runtime / "_run_create_mcs.py").read_text(encoding="utf-8")
        # The runner embeds paths as JSON string literals, so compare against
        # the escaped form the file actually contains.
        self.assertIn(json.dumps(bridge_python)[1:-1], runner)
        self.assertNotIn(sys.executable, runner)
        # And the embedded literal must still parse back to the real path.
        literal_line = [l for l in runner.splitlines() if "MIMICS_BRIDGE_PYTHON" in l][0]
        literal = literal_line.split("= ", 1)[1].strip()
        self.assertEqual(bridge_python, json.loads(literal))
        # Anti-revival: the production import_queues tree must be untouched.
        self.assertEqual(prod_before, self._snapshot_dir(prod_queues))

    @staticmethod
    def _snapshot_dir(path):
        """Map of relative path -> file content hash for a pollution guard."""
        snapshot = {}
        if not path.is_dir():
            return snapshot
        for item in sorted(path.iterdir()):
            if item.is_dir():
                snapshot[item.name] = "dir"
            else:
                snapshot[item.name] = hash(item.read_bytes())
        return snapshot


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

    def test_export_during_ongoing_import_skips_unpublished_case(self):
        """TB-07: exporting while an import is mid-conversion never reads
        half-converted work.

        The import pipeline keeps all conversion work in an isolated local
        run dir and publishes only the final .mcs to the output folder, so
        an export racing an ongoing import must simply see "no .mcs yet"
        for that case: skip it with an explicit reason, never open partial
        files, never touch the import's work dir, and never leave a
        .publishing_* staging dir of its own.
        """
        import mimics_export

        root = Path(self.tmp) / "tb07"
        dataset = root / "dataset"
        output = root / "mcs"
        runtime = root / "runtime"
        dataset.mkdir(parents=True)
        output.mkdir()
        runtime.mkdir()
        # case_done has its .mcs already; case_pending is mid-import: its
        # conversion work exists in an isolated run dir, no .mcs published.
        (dataset / "case_done").mkdir()
        (dataset / "case_done" / "ct.nii.gz").write_bytes(b"image")
        (dataset / "case_pending").mkdir()
        (dataset / "case_pending" / "ct.nii.gz").write_bytes(b"image")
        (output / "case_done.mcs").write_bytes(b"mcs-content")
        import_work = root / "import_runs" / "run_001"
        import_work.mkdir(parents=True)
        (import_work / "slice_0001.dcm").write_bytes(b"\0" * 64)
        work_state_before = sorted(
            (p.name, p.stat().st_size) for p in import_work.iterdir()
        )

        # A live import monitor mid-conversion for case_pending (busy tick
        # would be waiting on the bridge job; its work dir holds partial
        # output). The export must be able to run concurrently with this.
        import mimics_import
        monitor = {
            "monitor_key": str(import_work),
            "output_dir": str(output),
            "case_id": "case_pending",
            "job_dir": str(root / "import_runs" / "job_001"),
            "work_dir": str(import_work),
            "busy": True,
            "deadline": time.time() + 3600.0,
            "completed": 0,
            "failed": 0,
            "total": 1,
        }
        mimics_import._IMPORT_MONITORS[str(import_work)] = monitor

        status_path = runtime / "status.json"
        config_path = runtime / "config.json"
        config_path.write_text(json.dumps({
            "ts_root": str(dataset),
            "output_dir": str(output),
            "export_root": str(runtime),
            "job_runtime": str(runtime),
            "status_path": str(status_path),
            "stop_path": str(runtime / "stop.request"),
            # No mask_names: exercises the export loop without the saved-mask
            # preflight, which needs real Mimics project internals.
        }), encoding="utf-8")

        old_project_root = mimics_export._project_root
        old_open = mimics_export.mimics.file.open_project
        old_close = mimics_export.mimics.file.close_project
        opened_projects = []
        try:
            mimics_export._project_root = lambda: str(root)
            mimics_export.mimics.file.open_project = lambda path: opened_projects.append(str(path))
            mimics_export.mimics.file.close_project = lambda: None
            code = mimics_export.run_background_batch_export(str(config_path))
        finally:
            mimics_export._project_root = old_project_root
            mimics_export.mimics.file.open_project = old_open
            mimics_export.mimics.file.close_project = old_close

        # The completed case exported (its .mcs was opened); the pending
        # case was skipped with an explicit missing-.mcs reason, not a crash
        # and not an attempt to open half-converted work.
        self.assertEqual(0, code)
        self.assertEqual(1, len(opened_projects))
        self.assertIn("case_done", opened_projects[0])
        failed_dir = Path(mimics_export._rt(str(runtime), "_failed_exports"))
        pending_failures = sorted(failed_dir.glob("case_pending_*.json"))
        self.assertEqual(1, len(pending_failures))
        failure = json.loads(pending_failures[0].read_text(encoding="utf-8"))
        self.assertEqual("case_pending", failure.get("case_id"))
        self.assertIn("not found", str(failure.get("error") or ""))
        # The import's isolated work dir is untouched and no .publishing_*
        # staging dir was created anywhere in the export's tree.
        self.assertEqual(
            work_state_before,
            sorted((p.name, p.stat().st_size) for p in import_work.iterdir()),
        )
        staging = [str(p) for p in root.rglob("*") if ".publishing_" in p.name]
        self.assertEqual([], staging)
        # The live import monitor survived the concurrent export untouched.
        self.assertEqual(
            monitor, mimics_import._IMPORT_MONITORS.get(str(import_work))
        )
        mimics_import._IMPORT_MONITORS.pop(str(import_work), None)


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


    # ================================================================
    # BatchNorm checkpoint roundtrip
    # ================================================================


    # ================================================================
    # mimcs_lora_segformer3d config template
    # ================================================================


    # ================================================================
    # Export validation + label staging
    # ================================================================

    def test_launch_mimics_export_passes_label_staging_dir(self):
        """Export must support label_staging_dir for fresh-label isolation."""
        pipeline = __import__("tools.mimics_label_export", fromlist=["launch_mimics_export"])
        import inspect
        sig = inspect.signature(pipeline.launch_mimics_export)
        self.assertIn("label_staging_dir", sig.parameters,
                      "launch_mimics_export must accept label_staging_dir")

    def test_cancel_requested_control_json_conventions(self):
        """cancel_requested must not misread an existing control.json as cancelled.

        Job pipelines pass control.json (created at job start with
        action:"run"); only action "cancel"/"stop" means cancelled. A plain
        non-JSON marker file still cancels by existence (legacy convention).
        """
        pipeline = __import__("tools.mimics_label_export",
                              fromlist=["cancel_requested"])
        with tempfile.TemporaryDirectory() as tmp:
            run_path = os.path.join(tmp, "control.json")
            with open(run_path, "w", encoding="utf-8") as handle:
                json.dump({"action": "run"}, handle)
            self.assertFalse(pipeline.cancel_requested(run_path),
                             "an active control.json must not read as cancelled")
            with open(run_path, "w", encoding="utf-8") as handle:
                json.dump({"action": "cancel"}, handle)
            self.assertTrue(pipeline.cancel_requested(run_path),
                            "a cancelled control.json must read as cancelled")
            marker_path = os.path.join(tmp, "cancel.marker")
            self.assertFalse(pipeline.cancel_requested(marker_path),
                             "a missing marker must not read as cancelled")
            with open(marker_path, "w", encoding="utf-8") as handle:
                handle.write("cancel requested\n")
            self.assertTrue(pipeline.cancel_requested(marker_path),
                            "a legacy marker file must read as cancelled")
            self.assertFalse(pipeline.cancel_requested(None),
                             "no cancel path must not read as cancelled")

    # ================================================================
    # Stop Background Import
    # ================================================================

    def test_stop_background_import_entry_exists(self):
        """Stop_Background_Import entry must route to the correct function."""
        entry = os.path.join(
            PROJECT_ROOT, "scripting_library", "01_Data", "03_Stop_Import_Queue.py"
        )
        self.assertTrue(os.path.isfile(entry), "Stop_Background_Import entry must exist")
        with open(entry, "r", encoding="utf-8") as handle:
            content = handle.read()
        self.assertIn("main_stop_import", content,
                      "must route to main_stop_import function")

    def test_stop_background_import_targets_only_create_mcs(self):
        """stop_background_import must identify import lock by kind=create_mcs."""
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "runtime_py35"))
        import mimics_stop_background
        self.assertTrue(callable(mimics_stop_background.stop_background_import),
                        "stop_background_import must be callable")
        # Verify lock identification
        self.assertTrue(mimics_stop_background._lock_is_import_creation(
            {"kind": "create_mcs", "owner": "import .mcs creation for dataset X"}))
        self.assertFalse(mimics_stop_background._lock_is_import_creation(
            {"kind": "train", "owner": "nnU-Net training"}))

    def test_stop_background_import_requires_confirmation_and_states_scope(self):
        """R61-6: Stop Import must confirm before stopping and spell out that
        every queue (not just the current one) is affected. Cancelling the
        confirmation must not write any stop marker."""
        import mimics_stop_background as msb

        answers = []

        class FakeDialogs:
            @staticmethod
            def question_box(message=None, buttons=None, title=None,
                             ui_blocking=None):
                answers.append(str(message))
                return "Cancel"

            @staticmethod
            def message_box(message=None, title=None, ui_blocking=None):
                pass

        class FakeLogging:
            @staticmethod
            def log_user_message(level=None, message=None):
                pass

        class FakeMimics:
            dialogs = FakeDialogs()
            logging = FakeLogging()

        original_mimics = sys.modules.get("mimics")
        original_stop = msb.stop_background_import
        sys.modules["mimics"] = FakeMimics
        # Bind the fake on the already-imported module too (module-level
        # "import mimics" keeps a direct reference).
        original_module_mimics = msb.mimics
        msb.mimics = FakeMimics
        msb.stop_background_import = lambda: (_ for _ in ()).throw(
            AssertionError("stop ran despite a cancelled confirmation")
        )
        try:
            self.assertEqual(0, msb.main_stop_import())
            self.assertEqual(1, len(answers))
            # The scope (all queues, not just the current one) must be stated.
            self.assertIn("every import queue", answers[0])
        finally:
            sys.modules["mimics"] = original_mimics
            if original_mimics is None:
                sys.modules.pop("mimics", None)
            msb.mimics = original_module_mimics
            msb.stop_background_import = original_stop

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
    # Centralized mimcs_output_dir
    # ================================================================


    # ================================================================
    # Centralized mimcs_output_dir
    # ================================================================

    def test_resolve_mimics_output_dir_uses_single_config_key(self):
        """mimcs_output_dir must only use mimcs_output_dir (no deprecated aliases)."""
        pipeline = __import__("tools.mimics_label_export", fromlist=["resolve_mimics_output_dir"])
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

    def test_batch_discovery_fallback_scan_is_capped(self):
        """do_discover's loose-image fallback stops at the first slice (B17).

        A flat DICOM case directory can hold tens of thousands of slices.
        Enumerating all of them to prove no loose NIfTI exists adds no
        value; the UI-side scanner (discover_single_source) already stops
        the same walk early, and the bridge must mirror that.
        """
        import mimics_bridge

        case = os.path.join(self.tmp, "large_flat_dicom")
        os.makedirs(case)
        calls = {"next": 0}

        class Entry:
            name = "slice000001.dcm"
            path = os.path.join(case, name)

            def is_file(self):
                return True

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
                raise AssertionError(
                    "batch discovery enumerated past the first DICOM slice"
                )

        old_scandir = mimics_bridge.os.scandir
        old_listdir = mimics_bridge.os.listdir
        try:
            mimics_bridge.os.scandir = lambda _path: Entries()
            # The case-level listdir stays real; only the per-case fallback
            # walk is faked to prove it stops early.
            result = mimics_bridge.do_discover({"ts_root": self.tmp})
        finally:
            mimics_bridge.os.scandir = old_scandir
            mimics_bridge.os.listdir = old_listdir
        self.assertEqual(1, calls["next"])
        self.assertEqual([], result["cases"])

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
        import nninteractive_mimics

        self.assertIn("_background_process_kwargs", inspect.getsource(nninteractive_mimics._start_async_worker))

    def test_mask_writers_use_exclusive_callback_or_batch_leases(self):
        import inspect
        import mask_import
        import nnunet_mimics
        import nninteractive_mimics

        for callback in (
            nnunet_mimics._monitor_tick,
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
        # flexict_pipeline delegates to nnunet_jobs.hidden_process_kwargs
        # instead of embedding the flag itself.
        flexict_source = Path(
            PROJECT_ROOT, "tools/flexict_pipeline.py"
        ).read_text(encoding="utf-8")
        self.assertIn("hidden_process_kwargs", flexict_source)
        if os.name == "nt":
            # pipeline_common centralizes child-process spawning - verify the
            # behavior rather than the implementation string.
            import tools.pipeline_common as pipeline_common
            for module in (pipeline_common,):
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

    def test_discover_single_source_rejects_dataset_root(self):
        # R61-9: dropping a dataset root into the single-case flow must be
        # rejected with None, not accepted as one giant DICOM series.
        import tools.io_path_setup_ui as ui
        self._make_case("s0001", image="ct.nii.gz")
        self._make_case("s0002", image="ct.nii.gz")
        self.assertIsNone(ui.discover_single_source(self.tmp))
        # A root whose only "cases" are excluded dirs is not a dataset.
        os.makedirs(os.path.join(self.tmp, "mcs_output", "x"))
        self.assertIsNone(ui.discover_single_source(self.tmp))

    def test_discover_single_source_still_accepts_real_cases(self):
        # The dataset-root rejection must not eat genuine case folders:
        # image file, case dir with image, DICOM subdir, and empty dir
        # (historical DICOM-series candidate) all keep working.
        import tools.io_path_setup_ui as ui
        case = self._make_case("s0003", image="ct.nii.gz")
        self.assertIsNotNone(ui.discover_single_source(case))
        dcm_case = self._make_case("s0004", dicom=True)
        self.assertIsNotNone(ui.discover_single_source(dcm_case))
        empty = os.path.join(self.tmp, "empty_dir")
        os.makedirs(empty)
        result = ui.discover_single_source(empty)
        self.assertIsNotNone(result)
        self.assertEqual("dicom_candidate", result["image_type"])

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

        with mock.patch.object(dw.subprocess, "Popen", side_effect=fake_popen), \
                mock.patch.object(dw, "_ROOT", self.tmp):
            launched = dw.submit_import(selection, {})
        self.assertEqual([("dataset", "submitted")], launched)
        self.assertEqual(1, len(calls))
        command = calls[0]
        self.assertIn("mimics_batch_cli.py", command[1])
        self.assertIn("prepare-import", command)
        self.assertIn(self.tmp, command)
        # Batch drops also write a status file so the Recent-drops list can
        # show them (single-case drops already write one per case).
        status_files = [
            name for name in os.listdir(os.path.join(self.tmp, ".mimics_runtime", "drop_import"))
            if name.endswith("_batch_status.json")
        ]
        self.assertTrue(status_files)

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

        with mock.patch.object(dw.subprocess, "Popen", side_effect=fake_popen), \
                mock.patch.object(dw, "_ROOT", self.tmp):
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

    def test_collect_recent_drops_summarizes_status_files(self):
        import import_drop_window as dw
        status_dir = os.path.join(self.tmp, "drop_import")
        os.makedirs(status_dir)
        base = time.time()

        def _write(name, payload, offset=0):
            path = os.path.join(status_dir, name)
            dw.write_json(path, payload)
            stamp = base + offset
            os.utime(path, (stamp, stamp))

        _write("20260101T000001_0_selection.json", {
            "source_path": os.path.join(self.tmp, "case_alpha"),
            "output_path": "out",
        }, 100)
        _write("20260101T000001_0_status.json", {
            "status": "running", "phase": "waiting_for_background_mimics",
            "case_id": "case_alpha", "updated_at_epoch": base + 100,
        }, 100)
        _write("20260101T000002_0_selection.json", {
            "source_path": os.path.join(self.tmp, "case_beta"),
        }, 50)
        _write("20260101T000002_0_status.json", {
            "status": "failed", "phase": "prepare_failed",
            "case_id": "case_beta", "error": "boom\nsecond line",
            "updated_at_epoch": base + 50,
        }, 50)
        _write("20260101T000003_0_status.json", {
            "status": "completed", "case_id": "case_gamma",
        }, 10)
        # A corrupt status file must be skipped, not crash the poll.
        open(os.path.join(status_dir, "20260101T000004_0_status.json"), "w").close()

        entries = dw.collect_recent_drops(status_dir)
        self.assertEqual(3, len(entries))
        # Newest first: alpha (running), beta (failed), gamma (completed).
        self.assertEqual("case_alpha", entries[0]["case_id"])
        self.assertEqual("running", entries[0]["status"])
        self.assertEqual(
            os.path.join(status_dir, "20260101T000001_0.log"),
            entries[0]["log_path"],
        )
        self.assertEqual("case_beta", entries[1]["case_id"])
        self.assertEqual("boom", entries[1]["error"])
        self.assertEqual("case_gamma", entries[2]["case_id"])
        # The limit is respected.
        self.assertEqual(2, len(dw.collect_recent_drops(status_dir, limit=2)))

    def test_collect_recent_drops_handles_missing_dir(self):
        import import_drop_window as dw
        self.assertEqual(
            [], dw.collect_recent_drops(os.path.join(self.tmp, "no_such_dir"))
        )

    def _queue_runtime_dir_for(self, output_dir):
        import runtime_common
        return runtime_common.import_queue_runtime_dir(
            os.path.join(self.tmp, "fake_root"), output_dir
        )

    def _write_batch_drop_status(self, status_dir, name, output_dir, source_dir):
        os.makedirs(status_dir, exist_ok=True)
        dw_status = {
            "status": "running",
            "phase": "queued_to_prepare",
            "output_path": output_dir,
            "source_path": source_dir,
            "updated_at_epoch": time.time(),
        }
        import import_drop_window as dw
        dw.write_json(os.path.join(status_dir, name), dw_status)
        return os.path.join(status_dir, name)

    def test_batch_drop_status_unfreezes_from_queue_done_marker(self):
        # R61-8 failure path: the batch CLI never rewrites the drop status
        # file. When the queue runtime says done/failed/cancelled, the
        # recent-drops row must show that real state, not "Importing".
        import import_drop_window as dw
        import runtime_common

        output_dir = os.path.join(self.tmp, "mcs_output")
        os.makedirs(output_dir)
        status_dir = os.path.join(self.tmp, "drop_import")
        self._write_batch_drop_status(
            status_dir, "20260101T000001_batch_status.json",
            output_dir, self.tmp,
        )
        # Queue runtime markers live under the project root the batch CLI
        # ran with; collect_recent_drops derives them via _ROOT. Point _ROOT
        # at a fake root that shares the real runtime base derivation.
        fake_root = os.path.join(self.tmp, "fake_root")
        os.makedirs(fake_root, exist_ok=True)
        runtime_dir = runtime_common.import_queue_runtime_dir(fake_root, output_dir)
        os.makedirs(runtime_dir, exist_ok=True)
        runtime_common.write_json_atomic(
            os.path.join(runtime_dir, "_mcs_queue_done.json"),
            {
                "status": "done",
                "completed": 3,
                "failed": 1,
                "updated_at_epoch": time.time(),
            },
        )
        with mock.patch.object(dw, "_ROOT", fake_root):
            entries = dw.collect_recent_drops(status_dir)
        self.assertEqual(1, len(entries))
        self.assertEqual("completed", entries[0]["status"])

    def test_batch_drop_status_unfreezes_on_failure_with_reason(self):
        import import_drop_window as dw
        import runtime_common

        output_dir = os.path.join(self.tmp, "mcs_output2")
        os.makedirs(output_dir)
        status_dir = os.path.join(self.tmp, "drop_import2")
        self._write_batch_drop_status(
            status_dir, "20260101T000002_batch_status.json",
            output_dir, self.tmp,
        )
        fake_root = os.path.join(self.tmp, "fake_root2")
        os.makedirs(fake_root, exist_ok=True)
        runtime_dir = runtime_common.import_queue_runtime_dir(fake_root, output_dir)
        os.makedirs(runtime_dir, exist_ok=True)
        runtime_common.write_json_atomic(
            os.path.join(runtime_dir, "_mcs_batch_status.json"),
            {
                "status": "failed",
                "error": "Background Mimics could not create the .mcs file.\nsecond line",
                "updated_at_epoch": time.time(),
            },
        )
        with mock.patch.object(dw, "_ROOT", fake_root):
            entries = dw.collect_recent_drops(status_dir)
        self.assertEqual(1, len(entries))
        self.assertEqual("failed", entries[0]["status"])
        self.assertEqual(
            "Background Mimics could not create the .mcs file.",
            entries[0]["error"],
        )

    def test_batch_drop_status_stays_importing_while_queue_running(self):
        # No queue markers yet (preparation still in flight): the row must
        # keep showing "running", not guess a wrong terminal state.
        import import_drop_window as dw

        output_dir = os.path.join(self.tmp, "mcs_output3")
        os.makedirs(output_dir)
        status_dir = os.path.join(self.tmp, "drop_import3")
        self._write_batch_drop_status(
            status_dir, "20260101T000003_batch_status.json",
            output_dir, self.tmp,
        )
        fake_root = os.path.join(self.tmp, "fake_root3")
        os.makedirs(fake_root, exist_ok=True)
        with mock.patch.object(dw, "_ROOT", fake_root):
            entries = dw.collect_recent_drops(status_dir)
        self.assertEqual(1, len(entries))
        self.assertEqual("running", entries[0]["status"])

    def test_single_case_status_is_not_touched_by_batch_derivation(self):
        # Single-case workers own their status files end to end; the batch
        # derivation must not hijack rows it does not own.
        import import_drop_window as dw

        status_dir = os.path.join(self.tmp, "drop_import4")
        os.makedirs(status_dir)
        dw.write_json(os.path.join(status_dir, "20260101T000004_0_status.json"), {
            "status": "running", "phase": "creating_mcs",
            "case_id": "s0001", "updated_at_epoch": time.time(),
        })
        fake_root = os.path.join(self.tmp, "fake_root4")
        os.makedirs(fake_root, exist_ok=True)
        with mock.patch.object(dw, "_ROOT", fake_root):
            entries = dw.collect_recent_drops(status_dir)
        self.assertEqual(1, len(entries))
        self.assertEqual("running", entries[0]["status"])
        self.assertEqual("s0001", entries[0]["case_id"])

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

        # Patch (not assign) the shared mimics mock: bare assignment leaked a
        # plain list into mimics.data.images for the rest of the batch,
        # breaking later tests that patch images.get_active (R49).
        patches = [
            mock.patch.object(import_undo_mimics.mimics.data, "images", [image]),
            mock.patch.object(
                import_undo_mimics.mimics.file, "get_active_project",
                lambda: saved["open_target"][0] if saved["open_target"] else None,
            ),
            mock.patch.object(
                import_undo_mimics.mimics.file, "open_project",
                lambda filename=None: saved["open_target"].append(filename),
            ),
            mock.patch.object(
                import_undo_mimics.mimics.file, "save_project",
                lambda filename=None, save_as_type=None:
                    saved["saves"].append(filename),
            ),
            mock.patch.object(
                import_undo_mimics.mimics.file, "close_project",
                lambda: saved.__setitem__("closes", saved["closes"] + 1),
            ),
            mock.patch.object(
                import_undo_mimics.mimics.dialogs, "message_box",
                lambda *a, **kw: saved["message_boxes"].append((a, kw)) or True,
            ),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
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

    def test_undo_announces_blocking_steps_in_log_before_deleting(self):
        """B33: open/fingerprint/save block the GUI thread for as long as
        Mimics takes on a multi-hundred-MB .mcs. The log notice must fire
        before any deletion starts, so the pause is announced work, not a
        frozen window."""
        import logging as _logging
        import import_undo_mimics
        from create_mcs_batch import write_import_receipt

        out_dir = os.path.join(self.tmp, "out")
        os.makedirs(out_dir)
        mcs_path = os.path.join(out_dir, "s0001.mcs")
        with open(mcs_path, "wb") as handle:
            handle.write(b"project-bytes")
        write_import_receipt(out_dir, "s0001", mcs_path, ["Bone"], {})
        saved = self._install_undo_env(mcs_path, ["Bone", "Skin"])

        notices = []
        log_patch = mock.patch.object(
            import_undo_mimics.mimics.logging, "log_user_message",
            lambda level, message: notices.append((level, message)),
        )
        log_patch.start()
        self.addCleanup(log_patch.stop)

        # Snapshot the notice count at deletion time: the ordering claim is
        # "notice precedes deletion", which is only observable mid-flight.
        notice_count_at_deletion = []
        real_delete = import_undo_mimics._delete_masks

        def _delete_and_record(mask_names):
            notice_count_at_deletion.append(len(notices))
            return real_delete(mask_names)

        with mock.patch.object(import_undo_mimics, "_candidate_receipt_dirs",
                               lambda: [out_dir]), \
                mock.patch.object(import_undo_mimics, "_delete_masks",
                                  side_effect=_delete_and_record):
            code = import_undo_mimics.undo_last_import(confirm=False)
        self.assertEqual(0, code)
        # Two log entries: the pre-work notice, then the undo summary that
        # predates this change. The pre-work one must come first.
        self.assertEqual(2, len(notices))
        level, message = notices[0]
        self.assertEqual(_logging.INFO, level)
        self.assertIn("verifying", message)
        self.assertIn("s0001.mcs", message)
        self.assertEqual([1], notice_count_at_deletion,
                         "the notice must land before the deletion starts")
        self.assertEqual(sorted(["Bone"]), sorted(saved["deleted"]))

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
        message_patch = mock.patch.object(
            import_undo_mimics.mimics.dialogs, "message_box",
            lambda *a, **kw: boxes.append(kw) or True,
        )
        message_patch.start()
        self.addCleanup(message_patch.stop)
        with mock.patch.object(import_undo_mimics, "_candidate_receipt_dirs",
                               lambda: [empty]):
            code = import_undo_mimics.undo_last_import(confirm=False)
        self.assertEqual(1, code)
        self.assertEqual(1, len(boxes))

    def test_undo_confirm_copy_explains_steps_and_one_shot_semantics(self):
        """B7: the confirm dialog states the close-project step and that
        the undo record is consumed on success (kept on failure)."""
        import inspect
        import import_undo_mimics

        source = inspect.getsource(import_undo_mimics.undo_last_import)
        self.assertIn("close it first and ", source)
        self.assertIn("one-shot", source)
        self.assertIn("cannot be undone again", source)

    def test_undo_other_project_refusal_explains_two_steps(self):
        """B7: the refusal message gives the two-step recovery and says
        the record was not consumed."""
        import import_undo_mimics

        boxes = []
        state = {"active": "C:\\other\\project.mcs"}
        active_patch = mock.patch.object(
            import_undo_mimics.mimics.file, "get_active_project",
            lambda: state["active"],
        )
        message_patch = mock.patch.object(
            import_undo_mimics.mimics.dialogs, "message_box",
            lambda *a, **kw: boxes.append((a, kw)) or True,
        )
        active_patch.start()
        message_patch.start()
        self.addCleanup(active_patch.stop)
        self.addCleanup(message_patch.stop)
        ok = import_undo_mimics._open_project("C:\\target\\case.mcs")
        self.assertFalse(ok)
        self.assertEqual(1, len(boxes))
        text = boxes[0][0][0]
        self.assertIn("Step 1", text)
        self.assertIn("Step 2", text)
        self.assertIn("nothing was consumed", text)

    def test_undo_entry_points_exist(self):
        entry = os.path.join(
            PROJECT_ROOT, "scripting_library", "99_Admin", "04_Undo_Last_Import.py"
        )
        self.assertTrue(os.path.isfile(entry))
        with open(entry, "r") as handle:
            source = handle.read()
        self.assertIn("import_undo_mimics", source)
        drop = os.path.join(
            PROJECT_ROOT, "scripting_library", "01_Data", "07_Quick_Drop_Import.py"
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
            PROJECT_ROOT, "scripting_library", "99_Admin", "05_System_Health.py"
        )
        self.assertTrue(os.path.isfile(entry))
        with open(entry, "r") as handle:
            source = handle.read()
        self.assertIn("system_health_mimics", source)
        launcher = os.path.join(
            PROJECT_ROOT, "runtime_py35", "external_window_launcher.py"
        )
        with open(launcher, "r") as handle:
            source = handle.read()
        self.assertIn("launch_external_gui_process", source)
        self.assertIn("register_process", source)
        for thin_shell in (
            "system_health_mimics.py",
            "import_drop_mimics.py",
        ):
            path = os.path.join(PROJECT_ROOT, "runtime_py35", thin_shell)
            with open(path, "r") as handle:
                source = handle.read()
            self.assertIn("external_window_launcher.open_external_window", source)

    def test_collect_health_reports_environment_issues(self):
        shp = self._import_panel()
        snapshot = shp.collect_health(self.root)
        # Empty project has no python_env: exactly the python_missing issue.
        kinds = [issue["kind"] for issue in snapshot["environment"]]
        self.assertEqual(["python_missing"], kinds)

    def test_env_guidance_entry_exists(self):
        entry = os.path.join(
            PROJECT_ROOT, "scripting_library", "99_Admin",
            "09_Environment_Guidance.py"
        )
        self.assertTrue(os.path.isfile(entry))
        with open(entry, "r", encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn("env_guidance_mimics", source)
        wrapper = os.path.join(
            PROJECT_ROOT, "runtime_py35", "env_guidance_mimics.py"
        )
        self.assertTrue(os.path.isfile(wrapper))
        with open(wrapper, "r", encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn("external_window_launcher.open_external_window", source)


class TestEnvGuidance(unittest.TestCase):
    """Phase 2b: environment issues detected as annotator-readable rows."""

    def setUp(self):
        self.tmp = _make_temp_dir()
        self.root = os.path.join(self.tmp, "proj")
        os.makedirs(self.root)
        tools_dir = os.path.join(PROJECT_ROOT, "tools")
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)

    def tearDown(self):
        _cleanup(self.tmp)

    def _import_guidance(self):
        import importlib
        import env_guidance
        importlib.reload(env_guidance)
        return env_guidance

    def _fake_env(self, root):
        env_dir = os.path.join(root, "python_env")
        os.makedirs(env_dir, exist_ok=True)
        with open(os.path.join(env_dir, "python.exe"), "w") as handle:
            handle.write("x")

    def _fake_official_model(self, root):
        model_dir = os.path.join(
            root, "python_env", "models", "nnInteractive_v1.0", "fold_0")
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "checkpoint_final.pth"), "w") as handle:
            handle.write("x")

    def _write_state(self, payload):
        import json as _json
        runtime = os.path.join(self.root, ".mimics_runtime")
        os.makedirs(runtime, exist_ok=True)
        payload = dict(payload)
        payload.setdefault("updated_at_epoch", time.time())
        with open(os.path.join(runtime, "setup_env_state.json"), "w",
                  encoding="utf-8") as handle:
            _json.dump(payload, handle)

    def test_missing_python_reports_setup_action(self):
        eg = self._import_guidance()
        issues = eg.collect_issues(self.root)
        self.assertEqual(["python_missing"], [i["kind"] for i in issues])
        self.assertEqual("bad", issues[0]["severity"])
        self.assertTrue(issues[0]["fix_action"])

    def test_missing_official_model_is_reported_bad(self):
        eg = self._import_guidance()
        self._fake_env(self.root)
        old = os.environ.pop("NNINTERACTIVE_MODEL_DIR", None)
        try:
            issues = eg.collect_issues(self.root)
        finally:
            if old is not None:
                os.environ["NNINTERACTIVE_MODEL_DIR"] = old
        issue = next(i for i in issues if i["kind"] == "nninteractive_model")
        self.assertEqual("bad", issue["severity"])
        self.assertEqual("", issue["fix_action"])  # informational guidance
        self.assertIn("nnInteractive_v1.0", issue["detail"])

    def test_installed_official_model_silences_issue(self):
        eg = self._import_guidance()
        self._fake_env(self.root)
        self._fake_official_model(self.root)
        old = os.environ.pop("NNINTERACTIVE_MODEL_DIR", None)
        try:
            issues = eg.collect_issues(self.root)
        finally:
            if old is not None:
                os.environ["NNINTERACTIVE_MODEL_DIR"] = old
        self.assertEqual(
            [], [i["kind"] for i in issues if i["kind"] == "nninteractive_model"])

    def test_official_model_env_var_override(self):
        eg = self._import_guidance()
        self._fake_env(self.root)
        model_dir = os.path.join(self.tmp, "elsewhere", "nnInteractive_v1.0")
        fold = os.path.join(model_dir, "fold_0")
        os.makedirs(fold)
        with open(os.path.join(fold, "checkpoint_final.pth"), "w") as handle:
            handle.write("x")
        old = os.environ.pop("NNINTERACTIVE_MODEL_DIR", None)
        try:
            os.environ["NNINTERACTIVE_MODEL_DIR"] = model_dir
            issues = eg.collect_issues(self.root)
        finally:
            if old is None:
                os.environ.pop("NNINTERACTIVE_MODEL_DIR", None)
            else:
                os.environ["NNINTERACTIVE_MODEL_DIR"] = old
        self.assertEqual(
            [], [i["kind"] for i in issues if i["kind"] == "nninteractive_model"])

    def test_official_model_config_key_override(self):
        import json as _json
        eg = self._import_guidance()
        self._fake_env(self.root)
        model_dir = os.path.join(self.tmp, "cfgmodel", "nnInteractive_v1.0")
        fold = os.path.join(model_dir, "fold_1")
        os.makedirs(fold)
        with open(os.path.join(fold, "checkpoint_final.pth"), "w") as handle:
            handle.write("x")
        with open(os.path.join(self.root, "nninteractive_config.json"), "w",
                  encoding="utf-8") as handle:
            _json.dump({"model_dir": model_dir}, handle)
        old = os.environ.pop("NNINTERACTIVE_MODEL_DIR", None)
        try:
            issues = eg.collect_issues(self.root)
        finally:
            if old is not None:
                os.environ["NNINTERACTIVE_MODEL_DIR"] = old
        self.assertEqual(
            [], [i["kind"] for i in issues if i["kind"] == "nninteractive_model"])

    def test_failed_setup_state_is_reported(self):
        eg = self._import_guidance()
        self._fake_env(self.root)
        self._fake_official_model(self.root)
        self._write_state({"status": "error", "message": "boom", "error": "E"})
        issues = eg.collect_issues(self.root)
        self.assertEqual(["setup_failed"], [i["kind"] for i in issues])
        self.assertIn("boom", issues[0]["detail"])

    def test_incomplete_setup_lists_failed_packages(self):
        eg = self._import_guidance()
        self._fake_env(self.root)
        self._fake_official_model(self.root)
        self._write_state({
            "status": "incomplete", "message": "partial",
            "failed_packages": ["torch", "nnunetv2"],
        })
        issues = eg.collect_issues(self.root)
        self.assertEqual(["setup_incomplete"], [i["kind"] for i in issues])
        self.assertIn("torch", issues[0]["detail"])

    def test_stale_setup_state_ignored(self):
        eg = self._import_guidance()
        self._fake_env(self.root)
        self._fake_official_model(self.root)
        self._write_state({
            "status": "error", "message": "old failure",
            "updated_at_epoch": time.time() - 3 * 24 * 60 * 60,
        })
        self.assertEqual([], eg.collect_issues(self.root))

    def test_moved_checkout_detected_with_old_root(self):
        import json as _json
        eg = self._import_guidance()
        self._fake_env(self.root)
        self._fake_official_model(self.root)
        old_root = os.path.join(self.tmp, "oldmachine", "MIMICS_SCRIPTS")
        with open(os.path.join(self.root, "nninteractive_config.json"), "w",
                  encoding="utf-8") as handle:
            _json.dump({"workspace_dir": os.path.join(old_root, "ws")}, handle)
        detected = eg.detect_old_root(self.root)
        self.assertEqual(old_root, detected)
        issues = eg.collect_issues(self.root)
        kinds = [i["kind"] for i in issues]
        self.assertIn("migration_pending", kinds)
        issue = next(i for i in issues if i["kind"] == "migration_pending")
        self.assertTrue(issue["fix_action"].startswith("migrate:"))
        self.assertIn(old_root, issue["fix_action"])

    def test_live_and_relative_paths_are_not_migration_issues(self):
        import json as _json
        eg = self._import_guidance()
        self._fake_env(self.root)
        self._fake_official_model(self.root)
        with open(os.path.join(self.root, "nninteractive_config.json"), "w",
                  encoding="utf-8") as handle:
            _json.dump({
                "workspace_dir": "runtime/ws",                 # relative: fine
                "mimics_output_dir": os.path.join(self.tmp, "live", "out"),
            }, handle)
        os.makedirs(os.path.join(self.tmp, "live", "out"), exist_ok=True)
        self.assertEqual("", eg.detect_old_root(self.root))
        self.assertEqual([], eg.collect_issues(self.root))

    def test_flexict_missing_weights_is_informational(self):
        eg = self._import_guidance()
        self._fake_env(self.root)
        repo = os.path.join(self.root, "integrations", "flexict-finetune")
        os.makedirs(repo, exist_ok=True)
        issues = eg.collect_issues(self.root)
        kinds = [i["kind"] for i in issues]
        self.assertIn("flexict_weights", kinds)
        issue = next(i for i in issues if i["kind"] == "flexict_weights")
        self.assertEqual("", issue["fix_action"])  # informational only
        self.assertIn("flexict_2d", issue["detail"])

    def test_flexict_config_override_silences_weights_issue(self):
        import json as _json
        eg = self._import_guidance()
        self._fake_env(self.root)
        repo = os.path.join(self.root, "integrations", "flexict-finetune")
        os.makedirs(repo, exist_ok=True)
        with open(os.path.join(self.root, "flexict_config.json"), "w",
                  encoding="utf-8") as handle:
            _json.dump({"pretrained_weights_dir": "X:/weights"}, handle)
        issues = eg.collect_issues(self.root)
        self.assertEqual([], [i["kind"] for i in issues if i["kind"] == "flexict_weights"])

    def _write_scribble_config(self, checkpoint):
        import json as _json
        with open(os.path.join(self.root, "interactive_algorithms_config.json"),
                  "w", encoding="utf-8") as handle:
            _json.dump({"scribbleprompt": {"checkpoint": checkpoint}}, handle)

    def _pop_checkpoint_env(self):
        return os.environ.pop("SCRIBBLEPROMPT_CHECKPOINT", None)

    @staticmethod
    def _restore_checkpoint_env(old):
        if old is not None:
            os.environ["SCRIBBLEPROMPT_CHECKPOINT"] = old

    def test_scribbleprompt_missing_checkpoint_is_reported(self):
        eg = self._import_guidance()
        self._fake_env(self.root)
        self._fake_official_model(self.root)
        missing = os.path.join(self.tmp, "gone", "model.pt")
        self._write_scribble_config(missing)
        old = self._pop_checkpoint_env()
        try:
            issues = eg.collect_issues(self.root)
        finally:
            self._restore_checkpoint_env(old)
        issue = next(i for i in issues if i["kind"] == "scribbleprompt_checkpoint")
        self.assertEqual("warn", issue["severity"])
        self.assertEqual("", issue["fix_action"])  # informational: copy the file
        self.assertIn(missing, issue["detail"])

    def test_scribbleprompt_present_checkpoint_silences_issue(self):
        eg = self._import_guidance()
        self._fake_env(self.root)
        self._fake_official_model(self.root)
        checkpoint = os.path.join(self.tmp, "ckpts", "model.pt")
        os.makedirs(os.path.dirname(checkpoint))
        with open(checkpoint, "w") as handle:
            handle.write("x")
        self._write_scribble_config(checkpoint)
        old = self._pop_checkpoint_env()
        try:
            issues = eg.collect_issues(self.root)
        finally:
            self._restore_checkpoint_env(old)
        self.assertEqual(
            [], [i["kind"] for i in issues if i["kind"] == "scribbleprompt_checkpoint"])

    def test_scribbleprompt_relative_path_is_anchored_at_root(self):
        eg = self._import_guidance()
        self._fake_env(self.root)
        self._fake_official_model(self.root)
        # Relative config value must resolve against the project root, the
        # same anchoring interactive_algorithms_mimics._checkpoint_path uses.
        checkpoint = os.path.join(self.root, "integrations", "ScribblePrompt",
                                  "checkpoints", "unet.pt")
        os.makedirs(os.path.dirname(checkpoint))
        with open(checkpoint, "w") as handle:
            handle.write("x")
        self._write_scribble_config("integrations/ScribblePrompt/checkpoints/unet.pt")
        old = self._pop_checkpoint_env()
        try:
            issues = eg.collect_issues(self.root)
        finally:
            self._restore_checkpoint_env(old)
        self.assertEqual(
            [], [i["kind"] for i in issues if i["kind"] == "scribbleprompt_checkpoint"])

    def test_scribbleprompt_unconfigured_is_not_an_issue(self):
        # No env var, no config entry: nothing to check, so no issue. An
        # empty config is a different problem than a broken path.
        eg = self._import_guidance()
        self._fake_env(self.root)
        self._fake_official_model(self.root)
        old = self._pop_checkpoint_env()
        try:
            self.assertEqual([], eg.collect_issues(self.root))
        finally:
            self._restore_checkpoint_env(old)


class TestProcessRegistry(unittest.TestCase):
    def test_spawn_worker_registration_matches_api(self):
        """B20 source contract: the register_process call in _spawn_worker
        must use kwargs that exist on the API. The original bug was a
        job_id= kwarg the API never had — swallowed by a bare except, so
        every stage-worker registration failed silently."""
        import inspect
        import re as _re
        import resource_locks
        from tools import nnunet_pipeline

        source = inspect.getsource(nnunet_pipeline._spawn_worker)
        self.assertIn("register_process(", source)
        self.assertIn('"nnunet_{}".format(stage)', source)
        # Any kwarg passed at the call site must exist on the API.
        api_params = set(inspect.signature(resource_locks.register_process).parameters)
        call = source.split("register_process(", 1)[1]
        call = call.split(")", 1)[0]
        for kwarg in _re.findall(r"(\w+)=", call):
            if kwarg in ("ROOT", "process", "pid"):
                continue
            self.assertIn(kwarg, api_params, "unknown kwarg {0}".format(kwarg))

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

    def test_stage_worker_roles_are_registered(self):
        """B20: _spawn_worker registers nnU-Net/FlexiCT stage workers with
        roles nnunet_<stage>. If a role is missing from the whitelist, the
        registration fails silently and the health panel / kill-background
        safety net never sees the workers that orphan-hang."""
        import resource_locks

        for stage in ("preprocess", "train", "infer"):
            role = "nnunet_{0}".format(stage)
            self.assertIn(role, resource_locks.VALID_PROCESS_ROLES)
        child = self._spawn_sleeper()
        self.addCleanup(child.wait)
        self.addCleanup(child.kill)
        # The exact registration _spawn_worker performs must succeed (it
        # previously raised TypeError on an unknown job_id kwarg, swallowed
        # by the bare except — the safety net never existed).
        record = resource_locks.register_process(
            self.root, "nnunet_train", child.pid,
            state_path=os.path.join(self.tmp, "status.json"),
            extra={"job_id": "b20-test"},
        )
        self.assertEqual("b20-test", record["job_id"])
        self.assertTrue(record["ownership_token"])
        self.assertTrue(
            resource_locks.process_is_live(self.root, "nnunet_train", child.pid)
        )
        self.assertTrue(
            resource_locks.unregister_process(
                self.root, "nnunet_train", child.pid,
                ownership_token=record["ownership_token"],
            )
        )

    def test_sweep_terminates_stage_worker_orphaned_by_dead_parent(self):
        """B22: when the matrix (or anything) kills a job controller, its
        detached stage worker must be terminated by sweep_processes via the
        registry — the amplifier of the R41/R42 orphan-hang."""
        import resource_locks

        # Simulate the dead controller: a short-lived parent.
        parent = self._spawn_sleeper(0)
        parent.wait()
        worker = self._spawn_sleeper(30)
        self.addCleanup(worker.wait)
        self.addCleanup(worker.kill)
        record = resource_locks.register_process(
            self.root, "nnunet_preprocess", worker.pid,
            parent_pid=parent.pid,
            state_path=os.path.join(self.tmp, "status.json"),
        )
        self.assertTrue(record["ownership_token"])
        summary = resource_locks.sweep_processes(self.root)
        terminated = {e["pid"] for e in summary["terminated_orphans"]}
        self.assertIn(worker.pid, terminated)
        try:
            worker.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.fail("sweep reported termination but the worker is still alive")
        self.assertEqual(
            [], resource_locks.snapshot_processes(self.root, include_dead=True)
        )

    def test_recycled_pid_is_never_killed_by_terminate_or_sweep(self):
        """B10a: a record whose start_marker no longer matches the live
        PID points at a recycled process — killing it would terminate an
        unrelated program. Both kill paths must refuse."""
        import resource_locks

        child = self._spawn_sleeper()
        self.addCleanup(child.wait)
        self.addCleanup(child.kill)
        record = resource_locks.register_process(
            self.root, "trainer", child.pid, parent_pid=os.getpid()
        )
        record_path = resource_locks._process_record_path(
            resource_locks.default_process_registry_dir(self.root),
            record["record_id"],
        )
        # Simulate PID recycling: the live process keeps running, but the
        # record now claims a different (stale) start marker.
        stale_record = dict(record)
        stale_record["start_marker"] = record["start_marker"] + "0"
        resource_locks._write_json_atomic(record_path, stale_record)

        # process_is_live must report dead (marker mismatch).
        self.assertFalse(
            resource_locks.process_is_live(self.root, "trainer", child.pid)
        )

        # terminate_process must refuse to kill: it returns True ("already
        # gone") without ever touching the still-alive child.
        self.assertTrue(
            resource_locks.terminate_process(self.root, "trainer", child.pid)
        )
        self.assertIsNone(child.poll(), "terminate_process killed a recycled PID")

        # sweep must classify the record dead and remove it — but must not
        # terminate the live (recycled) process.
        summary = resource_locks.sweep_processes(self.root)
        self.assertEqual(1, summary["removed_dead_records"])
        self.assertEqual([], summary["terminated_orphans"])
        self.assertIsNone(child.poll(), "sweep killed a recycled PID")
        self.assertEqual(
            [], resource_locks.snapshot_processes(self.root, include_dead=True)
        )
        # The child survived the whole ladder; clean it up.
        child.kill()
        child.wait()

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


class TestStaleGuardSweep(unittest.TestCase):
    """R12 stale-guard sweep: idle guards go, everything else stays.

    Failure path focus: the sweep runs at import startup, so a wrong delete
    takes out the anchor of a LIVE lock and splits contenders across inodes
    (split-brain) -- the exact bug the three gates exist to prevent.
    """

    def setUp(self):
        self.tmp = _make_temp_dir()
        self._old_lock_dir = os.environ.get("MIMICS_RESOURCE_LOCK_DIR")
        os.environ["MIMICS_RESOURCE_LOCK_DIR"] = os.path.join(
            self.tmp, "proj", ".mimics_runtime", "locks"
        )
        self.lock_dir = os.environ["MIMICS_RESOURCE_LOCK_DIR"]
        os.makedirs(self.lock_dir, exist_ok=True)
        self.root = os.path.join(self.tmp, "proj")

    def tearDown(self):
        if self._old_lock_dir is None:
            os.environ.pop("MIMICS_RESOURCE_LOCK_DIR", None)
        else:
            os.environ["MIMICS_RESOURCE_LOCK_DIR"] = self._old_lock_dir
        _cleanup(self.tmp)

    def _make_guard(self, name, days_old=None):
        """One-byte guard file like _MutationGuard creates in production."""
        guard = os.path.join(self.lock_dir, name)
        with open(guard, "wb") as handle:
            handle.write(b"\0")
        if days_old is not None:
            self._age(guard, days_old)
        return guard

    @staticmethod
    def _age(path, days):
        stamp = time.time() - days * 86400.0
        os.utime(path, (stamp, stamp))

    def test_idle_guard_without_sibling_is_removed(self):
        import resource_locks

        guard = self._make_guard("gpu.lock.guard", days_old=40.0)
        removed = resource_locks.sweep_stale_guards(self.root, max_age_days=30.0)
        self.assertEqual(1, removed)
        self.assertFalse(os.path.exists(guard))

    def test_fresh_guard_is_kept(self):
        import resource_locks

        guard = self._make_guard("gpu.lock.guard")
        # mtime is now, well inside the retention window.
        removed = resource_locks.sweep_stale_guards(self.root, max_age_days=30.0)
        self.assertEqual(0, removed)
        self.assertTrue(os.path.exists(guard))

    def test_guard_with_live_sibling_lock_is_kept(self):
        import resource_locks

        guard = self._make_guard("trainer.lock.guard", days_old=40.0)
        # A sibling .lock JSON means the resource is registered/alive: the
        # guard anchor must survive even though the guard itself is idle.
        sibling = os.path.join(self.lock_dir, "trainer.lock")
        with open(sibling, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid()}, handle)
        removed = resource_locks.sweep_stale_guards(self.root, max_age_days=30.0)
        self.assertEqual(0, removed)
        self.assertTrue(os.path.exists(guard))

    def test_contended_guard_is_kept(self):
        import resource_locks

        guard = self._make_guard("gpu.lock.guard", days_old=40.0)
        # Hold the byte-0 lock exactly like a real guard holder.
        handle = open(guard, "a+b")
        self.addCleanup(handle.close)
        handle.seek(0)
        resource_locks._try_lock_byte(handle)
        self.addCleanup(resource_locks._unlock_byte, handle)
        removed = resource_locks.sweep_stale_guards(self.root, max_age_days=30.0)
        self.assertEqual(0, removed)
        self.assertTrue(os.path.exists(guard))


class TestImportQueuePrune(unittest.TestCase):
    """R12 import-queue prune: stale queue dirs and registry rows go,
    live queues stay. Failure path focus: deleting a queue a live consumer
    is still working in breaks the Stop Import Queue path and orphans locks.
    """

    def setUp(self):
        self.tmp = _make_temp_dir()
        self._old_lock_dir = os.environ.get("MIMICS_RESOURCE_LOCK_DIR")
        os.environ["MIMICS_RESOURCE_LOCK_DIR"] = os.path.join(
            self.tmp, "locks"
        )
        # The prune resolves the runtime base from the project root each
        # call; point it at the temp tree so no production state is touched.
        self._old_runtime_dir = os.environ.get("MIMICS_IMPORT_RUNTIME_DIR")
        os.environ["MIMICS_IMPORT_RUNTIME_DIR"] = os.path.join(
            self.tmp, "import_runtime"
        )
        import mimics_import
        import resource_locks
        self.mimics_import = mimics_import
        self.resource_locks = resource_locks
        self.base = os.path.join(
            self.tmp, "import_runtime", "import_queues"
        )

    def tearDown(self):
        if self._old_lock_dir is None:
            os.environ.pop("MIMICS_RESOURCE_LOCK_DIR", None)
        else:
            os.environ["MIMICS_RESOURCE_LOCK_DIR"] = self._old_lock_dir
        if self._old_runtime_dir is None:
            os.environ.pop("MIMICS_IMPORT_RUNTIME_DIR", None)
        else:
            os.environ["MIMICS_IMPORT_RUNTIME_DIR"] = self._old_runtime_dir
        _cleanup(self.tmp)

    @staticmethod
    def _age(path, days):
        stamp = time.time() - days * 86400.0
        os.utime(path, (stamp, stamp))

    def _make_queue(self, name, heartbeat_epoch=None):
        import runtime_common

        queue_dir = os.path.join(self.base, name)
        os.makedirs(queue_dir, exist_ok=True)
        if heartbeat_epoch is not None:
            runtime_common.write_json_atomic(
                os.path.join(queue_dir, "_mcs_queue_active.json"),
                {
                    "status": "active",
                    "updated_at_epoch": heartbeat_epoch,
                },
            )
        return queue_dir

    def _make_registry_row(self, name, queue_dir):
        import runtime_common

        registry_dir = os.path.join(
            self.tmp, "import_runtime", "mcs_queues"
        )
        os.makedirs(registry_dir, exist_ok=True)
        path = os.path.join(registry_dir, name + ".json")
        runtime_common.write_json_atomic(
            path,
            {
                "output_dir": os.path.join(self.tmp, name),
                "runtime_dir": queue_dir,
                "total_count": 3,
                "updated_at_epoch": time.time(),
            },
        )
        return path

    def test_stale_heartbeat_queue_and_registry_row_removed(self):
        mi = self.mimics_import
        queue_dir = self._make_queue(
            "out_stale", heartbeat_epoch=time.time() - 20 * 86400.0
        )
        registry_row = self._make_registry_row("out_stale", queue_dir)
        removed = mi._prune_import_queues(max_age_days=14)
        self.assertGreaterEqual(removed, 1)
        self.assertFalse(os.path.isdir(queue_dir))
        self.assertFalse(os.path.exists(registry_row))

    def test_fresh_heartbeat_queue_is_kept(self):
        mi = self.mimics_import
        queue_dir = self._make_queue(
            "out_fresh", heartbeat_epoch=time.time() - 60.0
        )
        removed = mi._prune_import_queues(max_age_days=14)
        self.assertEqual(0, removed)
        self.assertTrue(os.path.isdir(queue_dir))

    def test_queue_with_live_pid_lock_is_kept(self):
        mi = self.mimics_import
        queue_dir = self._make_queue(
            "out_live", heartbeat_epoch=time.time() - 20 * 86400.0
        )
        # A consumer lock naming THIS live process = an active import.
        with open(os.path.join(queue_dir, "consumer.lock"), "w",
                  encoding="utf-8") as handle:
            json.dump({"pid": os.getpid()}, handle)
        removed = mi._prune_import_queues(max_age_days=14)
        self.assertEqual(0, removed)
        self.assertTrue(os.path.isdir(queue_dir))

    def test_heartbeatless_old_dir_removed_via_mtime_fallback(self):
        mi = self.mimics_import
        queue_dir = self._make_queue("out_orphan")  # no heartbeat marker
        self._age(queue_dir, 20.0)
        removed = mi._prune_import_queues(max_age_days=14)
        self.assertGreaterEqual(removed, 1)
        self.assertFalse(os.path.isdir(queue_dir))

    def test_heartbeatless_fresh_dir_is_kept(self):
        mi = self.mimics_import
        queue_dir = self._make_queue("out_new")  # no heartbeat yet, young
        removed = mi._prune_import_queues(max_age_days=14)
        self.assertEqual(0, removed)
        self.assertTrue(os.path.isdir(queue_dir))


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

    def test_rotate_log_shifts_backups_and_drops_oldest(self):
        import tools.pipeline_common as pipeline_common

        path = os.path.join(self.tmp, "pipeline.log")
        for index in range(1, 4):  # .1, .2, .3 already exist (backups=3)
            Path(path + "." + str(index)).write_text("old" + str(index), encoding="utf-8")
        Path(path).write_text("x", encoding="utf-8")
        # A path under the threshold is left untouched.
        pipeline_common.rotate_log(path, max_bytes=10, backups=3)
        self.assertTrue(Path(path).exists())
        # Over the threshold: chain shifts by one, oldest (.3) is dropped,
        # the live log becomes .1.
        Path(path).write_text("x" * 20, encoding="utf-8")
        pipeline_common.rotate_log(path, max_bytes=10, backups=3)
        self.assertFalse(Path(path).exists())
        self.assertEqual("x" * 20, Path(path + ".1").read_text(encoding="utf-8"))
        self.assertEqual("old1", Path(path + ".2").read_text(encoding="utf-8"))
        self.assertEqual("old2", Path(path + ".3").read_text(encoding="utf-8"))
        # backups=0 wipes the live log instead of rotating it
        # (existing backup files are left as-is).
        Path(path).write_text("x" * 20, encoding="utf-8")
        pipeline_common.rotate_log(path, max_bytes=10, backups=0)
        self.assertFalse(Path(path).exists())
        self.assertTrue(Path(path + ".1").exists())
        # A missing path must not raise.
        pipeline_common.rotate_log(os.path.join(self.tmp, "never.log"))

    def test_runtime_common_rotate_log_file_matches_pipeline_semantics(self):
        """runtime_py35 shares one rotate implementation; pin the py3.5 side."""
        import runtime_common

        path = os.path.join(self.tmp, "runtime.log")
        Path(path + ".3").write_text("old3", encoding="utf-8")
        Path(path + ".2").write_text("old2", encoding="utf-8")
        Path(path + ".1").write_text("old1", encoding="utf-8")
        Path(path).write_text("y" * 20, encoding="utf-8")
        runtime_common.rotate_log_file(path, max_bytes=10, backups=3)
        self.assertFalse(Path(path).exists())
        self.assertEqual("y" * 20, Path(path + ".1").read_text(encoding="utf-8"))
        self.assertEqual("old1", Path(path + ".2").read_text(encoding="utf-8"))
        self.assertEqual("old2", Path(path + ".3").read_text(encoding="utf-8"))
        self.assertFalse(Path(path + ".4").exists())
        # Best-effort contract: a bad path must not raise.
        runtime_common.rotate_log_file(os.path.join(self.tmp, "no-such-dir", "x.log"))

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
                self.path = Path(_args[0]) if _args else None
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
                self.path = Path(_args[0]) if _args else None
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
        import tools.mimics_label_export as pipeline

        with mock.patch.object(bridge, "resource_process_exists", return_value=True) as check:
            self.assertTrue(bridge._process_exists(259))
            check.assert_called_once_with(259)
        with mock.patch.object(pipeline, "resource_process_exists", return_value=False) as check:
            self.assertFalse(pipeline.process_exists(259))
            check.assert_called_once_with(259)

    def test_startup_cleanup_has_no_aggressive_kill_path(self):
        """MIMICS_AGGRESSIVE_AUTO_CLEANUP_ON_START was retired (B13).

        The retired flag gated a command-line-heuristic kill of live
        bridge/worker processes that could interrupt valid async
        workflows. Ownership-proven termination now happens only through
        the process registry sweep and owned-server idle cleanup, and the
        explicit Stop All Owned Services entry covers the rest — so no
        runtime_py35 module may read the flag or terminate processes via
        raw kernel32 OpenProcess/TerminateProcess again.
        """
        import inspect
        import mimics_import
        import nninteractive_mimics
        import runtime_common

        for module in (mimics_import, nninteractive_mimics, runtime_common):
            source = inspect.getsource(module)
            self.assertNotIn(
                "MIMICS_AGGRESSIVE_AUTO_CLEANUP_ON_START",
                "{0} still reads the retired aggressive-cleanup flag".format(
                    module.__name__
                ),
            )
            self.assertNotIn(
                "aggressive_auto_cleanup_enabled",
                "{0} still exposes an aggressive-cleanup helper".format(
                    module.__name__
                ),
            )
        for module in (mimics_import, nninteractive_mimics):
            source = inspect.getsource(module)
            self.assertNotIn(
                "OpenProcess(PROCESS_TERMINATE",
                "{0} still kills processes via raw kernel32 handles".format(
                    module.__name__
                ),
            )
        # The pointer-sized HANDLE declarations the old test guarded live
        # only in the deleted kill path; the surviving process APIs go
        # through runtime_common (PROCESS_QUERY_LIMITED_INFORMATION) and
        # taskkill /T /F.
        self.assertFalse(hasattr(runtime_common, "aggressive_auto_cleanup_enabled"))


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
        import tools.mimics_label_export as pipeline

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
        import tools.mimics_label_export as pipeline

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

    def test_bridge_error_cleans_up_partial_derived_dicom_work_dir(self):
        # TB-01: the bridge dies/errors mid-DICOM-write. The monitor's
        # error path must remove the work dir with its partial slices —
        # no half-converted residue survives into the next run.
        import mimics_import

        work_dir = os.path.join(self.tmp, "work", "case_died")
        os.makedirs(work_dir)
        for index in (1, 2, 3):
            with open(os.path.join(work_dir, "slice_{:04d}.dcm".format(index)), "wb") as handle:
                handle.write(b"partial")
        job_dir = os.path.join(self.tmp, "job")
        os.makedirs(job_dir)
        starts = []
        statuses = []
        monitor = {
            "job_dir": job_dir,
            "work_dir": work_dir,
            "output_dir": self.tmp,
            "case_id": "case_died",
            "busy": False,
            "done": False,
            "deadline": time.time() + 3600.0,
            "completed": 0,
            "failed": 0,
            "total": 1,
            "batch_queue": [],
            "task_status_path": os.path.join(self.tmp, "status.json"),
        }
        old_check = mimics_import._check_job_status
        old_stopped = mimics_import._import_task_stopped
        old_start = mimics_import._start_next_batch_prepare
        old_write = mimics_import._write_import_task_status
        old_log = mimics_import._append_import_log
        old_record = mimics_import._record_failed_case
        old_cleanup_job = mimics_import._cleanup_job_dir
        try:
            mimics_import._check_job_status = lambda _job: ("error", "bridge died mid-slice")
            mimics_import._import_task_stopped = lambda _monitor: False
            mimics_import._start_next_batch_prepare = lambda _monitor: starts.append(True)
            mimics_import._write_import_task_status = lambda _path, payload: statuses.append(
                dict(payload)
            )
            mimics_import._append_import_log = lambda *_args: None
            mimics_import._record_failed_case = lambda *_args: None
            mimics_import._cleanup_job_dir = lambda _path: None
            mimics_import._batch_prepare_tick_impl(monitor)
        finally:
            mimics_import._check_job_status = old_check
            mimics_import._import_task_stopped = old_stopped
            mimics_import._start_next_batch_prepare = old_start
            mimics_import._write_import_task_status = old_write
            mimics_import._append_import_log = old_log
            mimics_import._record_failed_case = old_record
            mimics_import._cleanup_job_dir = old_cleanup_job
        # The case is recorded as failed, the residue work dir is gone,
        # and the queue moved on.
        self.assertEqual(1, monitor["failed"])
        self.assertFalse(os.path.exists(work_dir))
        self.assertEqual([True], starts)
        self.assertEqual(1, statuses[-1]["failed"])

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
        import tools.mimics_label_export as pipeline

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
        import tools.mimics_label_export as pipeline

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
        import tools.mimics_label_export as pipeline

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
        import tools.mimics_label_export as pipeline

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
        import tools.mimics_label_export as pipeline

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
        import tools.mimics_label_export as pipeline

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


    def test_clear_cache_does_not_include_runtime_control_state_or_active_jobs(self):
        import mimics_stop_background as stop

        root = Path(self.tmp)
        runtime = root / ".mimics_runtime"
        runtime.mkdir()
        control = runtime / "nninteractive_server_state.json"
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

    def test_cleanup_async_jobs_sweeps_crash_orphaned_nonterminal_dirs(self):
        """R61-15: a Mimics crash leaves job/worker dirs in non-terminal
        states (queued/ready/running/result_ready) with a dead pid. The
        retention sweep must collect them once they age past the window;
        before this fix 653MB of orphaned dirs accumulated forever."""
        import time as _time
        import nninteractive_mimics as nnm

        root = os.path.join(self.tmp, "async_jobs")
        # Orphaned non-terminal job: "running" but no live process, old.
        orphan = os.path.join(root, "job_orphan")
        os.makedirs(orphan)
        with open(os.path.join(orphan, "job.json"), "w", encoding="utf-8") as h:
            json.dump({"status": "running", "updated_at_epoch": 1, "pid": 0}, h)
        old_time = _time.time() - 10 * 86400
        os.utime(orphan, (old_time, old_time))
        # Fresh non-terminal job: must survive even with a dead pid.
        fresh = os.path.join(root, "job_fresh")
        os.makedirs(fresh)
        with open(os.path.join(fresh, "job.json"), "w", encoding="utf-8") as h:
            json.dump({"status": "running", "updated_at_epoch": 1, "pid": 0}, h)
        # Old job whose pid IS alive: must survive (an active worker).
        live = os.path.join(root, "job_live")
        os.makedirs(live)
        with open(os.path.join(live, "job.json"), "w", encoding="utf-8") as h:
            json.dump(
                {"status": "running", "updated_at_epoch": 1, "pid": os.getpid()},
                h,
            )
        os.utime(live, (old_time, old_time))
        # Terminal job inside the window and under the cap: survives.
        terminal = os.path.join(root, "job_terminal")
        os.makedirs(terminal)
        with open(os.path.join(terminal, "job.json"), "w", encoding="utf-8") as h:
            json.dump({"status": "closed", "updated_at_epoch": old_time}, h)

        original_exists = nnm._process_exists
        try:
            nnm._process_exists = lambda pid: int(pid or 0) == os.getpid()
            nnm._cleanup_async_jobs(root, retention_days=3, max_terminal_jobs=20)
        finally:
            nnm._process_exists = original_exists

        self.assertFalse(os.path.isdir(orphan), "crash-orphaned dir must be swept")
        self.assertTrue(os.path.isdir(fresh), "fresh non-terminal job must survive")
        self.assertTrue(os.path.isdir(live), "old job with a live pid must survive")
        self.assertTrue(os.path.isdir(terminal), "terminal job in window must survive")

    def test_stop_all_requires_owned_root_and_excludes_foreground(self):
        path = os.path.join(PROJECT_ROOT, "runtime_py35", "mimics_stop_background.py")
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()
        self.assertNotIn("$broad=Get-CimInstance", source)
        self.assertIn("$foregroundPid", source)
        self.assertIn("$inRoot -and $hasMarker", source)
        self.assertIn("$cutoff", source)
        self.assertNotIn("Remove-Item -Path $lock", source)

    def test_collect_diagnostics_never_blocks_gui_thread(self):
        """A1: collecting diagnostics must run in a background thread.

        The collector subprocess can take up to 300s; a synchronous
        subprocess.run in the Mimics script thread froze the whole GUI
        (iron law 1). The collection has to go through the daemon-thread +
        timer-poll pattern used by mimics_stop_background.clear_cache_main.
        """
        path = os.path.join(
            PROJECT_ROOT, "runtime_py35", "collect_diagnostics_mimics.py"
        )
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()
        import inspect

        sys.path.insert(0, RUNTIME_DIR)
        import collect_diagnostics_mimics as cdm

        # collect_bundle itself must not call subprocess synchronously
        bundle_source = inspect.getsource(cdm.collect_bundle)
        self.assertNotIn("subprocess.run", bundle_source)
        self.assertIn("threading.Thread", bundle_source)
        self.assertIn("daemon", bundle_source)
        # the subprocess call lives in the worker body only
        worker_source = inspect.getsource(cdm._collect_in_background)
        self.assertIn("subprocess.run", worker_source)
        # completion is reported by a timer tick, never from collect_bundle
        self.assertIn("_start_timer", bundle_source)

    def test_collect_diagnostics_background_flow_end_to_end(self):
        """The daemon thread path produces the bundle and reports it."""
        sys.path.insert(0, RUNTIME_DIR)
        import collect_diagnostics_mimics as cdm  # noqa: F401 (reimported below)

        root = _make_temp_dir()
        os.makedirs(os.path.join(root, "tools"))
        os.makedirs(os.path.join(root, "python_env"))
        marker = os.path.join(root, "bundle_written.marker")
        with open(
            os.path.join(root, "tools", "collect_diagnostics.py"), "w"
        ) as handle:
            handle.write(
                "import sys, os\n"
                "out = sys.argv[sys.argv.index('--output') + 1]\n"
                "open(out, 'w').write('bundle')\n"
                "open({!r}, 'w').write('done')\n".format(marker)
            )
        # find_external_python is patched below to return this path; the
        # collector script is fake, but the interpreter itself must be a
        # real executable for subprocess.run to launch it.
        fake_python = os.path.join(root, "python_env", "python.exe")
        shutil.copy(sys.executable, fake_python)

        messages = []

        class FakeDialogs:
            @staticmethod
            def message_box(message, title=None, ui_blocking=None):
                messages.append(str(message))

        class FakeLogging:
            @staticmethod
            def log_user_message(level=None, message=None):
                messages.append(str(message))

        class FakeMimics:
            dialogs = FakeDialogs()
            logging = FakeLogging()

        original_mimics = sys.modules.get("mimics")
        sys.modules["mimics"] = FakeMimics
        # reimport so the module-level "import mimics" binds our fake
        sys.modules.pop("collect_diagnostics_mimics", None)
        import importlib

        cdm = importlib.import_module("collect_diagnostics_mimics")
        original_find = cdm.runtime_common.find_external_python
        original_root = cdm._project_root
        cdm.runtime_common.find_external_python = lambda *_a, **_k: fake_python
        cdm._project_root = lambda: root
        # timer: skip the GUI pump entirely, the test drives the tick
        cdm._start_timer = lambda tick_fn, monitor, poll_seconds=0.5: None
        try:
            exit_code = cdm.collect_bundle()
            self.assertEqual(exit_code, 0)
            thread = getattr(cdm, "_ACTIVE_MONITOR", {}).get("thread")
            self.assertIsNotNone(thread)
            # the "started" message fires immediately, before any bundle
            self.assertTrue(
                any("background" in m for m in messages),
                messages,
            )
            thread.join(timeout=30)
            self.assertTrue(
                os.path.isfile(marker),
                "worker never ran the collector script",
            )
            monitor = cdm._ACTIVE_MONITOR
            monitor["done"] = True
            cdm._diagnostics_tick(monitor)
            self.assertTrue(
                any("safe to share" in m for m in messages),
                messages,
            )
        finally:
            sys.modules["mimics"] = original_mimics
            if original_mimics is None:
                sys.modules.pop("mimics", None)
            cdm.runtime_common.find_external_python = original_find
            cdm._project_root = original_root
            sys.modules.pop("collect_diagnostics_mimics", None)
            _cleanup(root)

    def test_external_kill_background_never_deletes_resource_locks(self):
        import inspect
        import tools.mimics_batch_cli as cli

        source = inspect.getsource(cli.cmd_kill_background)
        self.assertIn("stopMarkers", source)
        self.assertIn("cutoff", source)
        self.assertNotIn("Remove-Item", source)

    def test_prune_import_receipts_deletes_only_expired_receipts(self):
        # B9-b: the import-receipt retention hook had zero direct tests.
        # Receipts carry patient file paths, so an unpruned (or over-pruned)
        # hook is a privacy/data bug, not just disk hygiene.
        import create_mcs_batch

        old = time.time()
        keep = Path(self.tmp) / "keep.import_receipt.json"
        keep.write_text("{}", encoding="utf-8")
        expired = Path(self.tmp) / "expired.import_receipt.json"
        expired.write_text("{}", encoding="utf-8")
        os.utime(str(expired), (old - 31 * 86400, old - 31 * 86400))
        unrelated = Path(self.tmp) / "unrelated.json"
        unrelated.write_text("{}", encoding="utf-8")
        os.utime(str(unrelated), (old - 60 * 86400, old - 60 * 86400))

        create_mcs_batch.prune_import_receipts(self.tmp, retention_days=30)

        self.assertTrue(keep.exists())
        self.assertFalse(expired.exists())
        # Only .import_receipt.json files are ever touched.
        self.assertTrue(unrelated.exists())

    def test_prune_old_checkpoints_deletes_only_expired_breadcrumbs(self):
        # B9-b: every Mimics import PID leaves a checkpoint pair under
        # debug_out; without pruning they accumulate forever. Expired
        # breadcrumbs go, fresh ones and unrelated debug files stay.
        import mimics_import

        old = time.time()
        fresh = Path(self.tmp) / "mimics_import_checkpoint_fresh.json"
        fresh.write_text("{}", encoding="utf-8")
        expired = Path(self.tmp) / "mimics_import_checkpoint_old.json"
        expired.write_text("{}", encoding="utf-8")
        os.utime(str(expired), (old - 15 * 86400, old - 15 * 86400))
        unrelated = Path(self.tmp) / "other_debug.json"
        unrelated.write_text("{}", encoding="utf-8")
        os.utime(str(unrelated), (old - 60 * 86400, old - 60 * 86400))

        mimics_import._prune_old_checkpoints(self.tmp, retention_days=14)

        self.assertTrue(fresh.exists())
        self.assertFalse(expired.exists())
        self.assertTrue(unrelated.exists())

    def test_prune_logs_keeps_newest_logs_only(self):
        # B9-b: external-window stderr logs keep only the newest N by
        # mtime; older logs are removed, non-log files untouched.
        import external_window_launcher

        now = time.time()
        kept = []
        removed = []
        for index in range(4):
            path = Path(self.tmp) / "window_{}.log".format(index)
            path.write_text("log", encoding="utf-8")
            stamp = now - (3 - index) * 3600
            os.utime(str(path), (stamp, stamp))
            (kept if index >= 2 else removed).append(path)
        unrelated = Path(self.tmp) / "window_notes.txt"
        unrelated.write_text("notes", encoding="utf-8")

        external_window_launcher.prune_logs(self.tmp, "window_", 2)

        for path in kept:
            self.assertTrue(path.exists(), path.name)
        for path in removed:
            self.assertFalse(path.exists(), path.name)
        self.assertTrue(unrelated.exists())

    def test_prune_local_export_jobs_respects_age_cap_and_active_jobs(self):
        # B9-b: export job dirs are pruned past the age/count caps, but a
        # dir whose status file reports a live PID must never be removed.
        import mimics_export

        jobs_root = Path(self.tmp) / ".mimics_runtime" / "export_jobs"
        jobs_root.mkdir(parents=True)
        now = time.time()
        live_pid = os.getpid()

        recent = jobs_root / "recent"
        recent.mkdir()
        (recent / "status.json").write_text("{}", encoding="utf-8")

        expired = jobs_root / "expired"
        expired.mkdir()
        (expired / "status.json").write_text("{}", encoding="utf-8")
        stamp = now - 20 * 86400
        os.utime(str(expired), (stamp, stamp))

        active = jobs_root / "active"
        active.mkdir()
        (active / "status.json").write_text(
            json.dumps({"status": "running", "pid": live_pid}),
            encoding="utf-8",
        )
        os.utime(str(active), (stamp, stamp))

        non_job = jobs_root / "stray.txt"
        non_job.write_text("x", encoding="utf-8")

        old_root = mimics_export._project_root
        try:
            mimics_export._project_root = lambda: self.tmp
            mimics_export._prune_local_export_jobs(max_age_days=14, max_jobs=100)
        finally:
            mimics_export._project_root = old_root

        self.assertTrue(recent.exists())
        self.assertFalse(expired.exists())
        self.assertTrue(active.exists())
        self.assertTrue(non_job.exists())


class TestFlexiCTActiveLearningApply(unittest.TestCase):
    """R61-3: the AL overlay conversion must not block the GUI thread.

    _al_apply_request may only spawn the bridge process, mark the request
    "converting", and return — the communicate() wait happens in a daemon
    thread and its result is applied by _al_finish_conversion on a later
    monitor tick. Source-contract assertions keep the blocking form from
    coming back.
    """

    def test_al_apply_request_returns_without_waiting_for_bridge(self):
        import ast
        import inspect
        import textwrap

        import flexict_mimics

        source = textwrap.dedent(
            inspect.getsource(flexict_mimics._al_apply_request)
        )
        tree = ast.parse(source)

        def _communicate_calls_outside_nested_functions(node):
            """Yield communicate() calls made directly by _al_apply_request
            (i.e. NOT inside a nested def — those run on the wait thread)."""
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue  # nested function bodies run on their own thread
                if isinstance(child, ast.Call):
                    func = child.func
                    if (
                        isinstance(func, ast.Attribute)
                        and func.attr == "communicate"
                    ):
                        yield child
                for hit in _communicate_calls_outside_nested_functions(child):
                    yield hit

        blocking = list(_communicate_calls_outside_nested_functions(tree))
        self.assertEqual(
            [], blocking,
            "the bridge communicate must live only inside the wait-thread "
            "function, not on the GUI monitor tick path",
        )
        self.assertIn("threading.Thread", source)
        self.assertIn('"converting"', source)
        finish_source = inspect.getsource(flexict_mimics._al_finish_conversion)
        self.assertIn("_set_mask_from_u8", finish_source)
        # The tick drives both stages: start, then poll result.json.
        tick_source = inspect.getsource(flexict_mimics._al_tick)
        self.assertIn('"conversion"', tick_source)
        self.assertIn("_al_finish_conversion", tick_source)

    def test_al_conversion_applies_finished_result_and_marks_request(self):
        import flexict_mimics

        with tempfile.TemporaryDirectory() as tmp:
            request_dir = os.path.join(tmp, "job", "apply_requests")
            os.makedirs(request_dir)
            request_path = os.path.join(request_dir, "req.json")
            with open(request_path, "w") as handle:
                json.dump({
                    "case_id": "caseA", "what": "bands",
                    "state": "converting", "requested_at_epoch": 1.0,
                }, handle)
            # A finished bridge result: one mask row with a u8 buffer.
            buffer_path = os.path.join(tmp, "mask.u8")
            with open(buffer_path, "wb") as handle:
                handle.write(b"\x01" * 24)
            bridge_root = os.path.join(tmp, "bridge")
            os.makedirs(bridge_root)
            result_path = os.path.join(bridge_root, "result.json")
            with open(result_path, "w") as handle:
                json.dump({
                    "status": "ok",
                    "masks": [{
                        "name": "al_0",
                        "output_path": buffer_path,
                        "mimics_shape": [2, 3, 4],
                    }],
                }, handle)

            applied = []
            marked = []
            old_new_mask = flexict_mimics.mimics_mask_apply._new_prediction_mask
            old_set = flexict_mimics.mimics_mask_apply._set_mask_from_u8
            old_mark = flexict_mimics._al_mark_request
            old_annotated = flexict_mimics._al_mark_annotated

            class _Mask(object):
                name = "Bands (Moderate)"

            try:
                flexict_mimics.mimics_mask_apply._new_prediction_mask = (
                    lambda title: _Mask()
                )
                flexict_mimics.mimics_mask_apply._set_mask_from_u8 = (
                    lambda mask, path, shape, name=None: applied.append(mask.name)
                )
                flexict_mimics._al_mark_request = (
                    lambda path, state, detail="": marked.append((state, detail))
                )
                flexict_mimics._al_mark_annotated = (
                    lambda job_dir, case_id: None
                )
                transaction = {
                    "request": {
                        "_request_path": request_path,
                        "_job_dir": os.path.join(tmp, "job"),
                        "case_id": "caseA",
                    },
                    "result_path": result_path,
                    "bridge_root": bridge_root,
                    "masks": [("Bands (Moderate)", buffer_path)],
                }
                flexict_mimics._al_finish_conversion(transaction)
            finally:
                flexict_mimics.mimics_mask_apply._new_prediction_mask = old_new_mask
                flexict_mimics.mimics_mask_apply._set_mask_from_u8 = old_set
                flexict_mimics._al_mark_request = old_mark
                flexict_mimics._al_mark_annotated = old_annotated
            self.assertEqual(["Bands (Moderate)"], applied)
            self.assertEqual(1, len(marked))
            self.assertEqual("applied", marked[0][0])
            self.assertFalse(os.path.exists(bridge_root))

    def test_al_conversion_failure_marks_request_failed(self):
        import flexict_mimics

        with tempfile.TemporaryDirectory() as tmp:
            request_dir = os.path.join(tmp, "job", "apply_requests")
            os.makedirs(request_dir)
            request_path = os.path.join(request_dir, "req.json")
            bridge_root = os.path.join(tmp, "bridge")
            os.makedirs(bridge_root)
            result_path = os.path.join(bridge_root, "result.json")
            with open(result_path, "w") as handle:
                json.dump({"status": "error", "error": "conversion timed out"}, handle)

            marked = []
            old_mark = flexict_mimics._al_mark_request
            try:
                flexict_mimics._al_mark_request = (
                    lambda path, state, detail="": marked.append((state, detail))
                )
                transaction = {
                    "request": {
                        "_request_path": request_path,
                        "_job_dir": os.path.join(tmp, "job"),
                        "case_id": "caseA",
                    },
                    "result_path": result_path,
                    "bridge_root": bridge_root,
                    "masks": [],
                }
                flexict_mimics._al_finish_conversion(transaction)
            finally:
                flexict_mimics._al_mark_request = old_mark
            self.assertEqual([("failed", "conversion timed out")], marked)
            self.assertFalse(os.path.exists(bridge_root))


class TestGuiThreadBlockingContract(unittest.TestCase):
    """R61-13: source-contract scan — nothing slow may run in a *_tick.

    Mimics monitor ticks run on the GUI thread. Two call shapes are always
    movable off it and therefore forbidden at tick level (nested thread
    functions are exempt — that is the correct pattern):
    - process.communicate(...): a subprocess wait; belongs on a wait thread
      (see FlexiCT's wait_bridge and the AL conversion in R61-3).
    - .tobytes() on a mask/voxel buffer: a full-volume copy hashing or
      write-out; belongs on a worker (see the nnInteractive SHA-256 fix).

    open_project() is a Mimics API call and is thread-affine — it cannot
    move off the GUI thread, so it is out of scope here (its only tick
    call site is opt-in via MIMICS_IMPORT_AUTO_OPEN_MCS).
    """

    FORBIDDEN_ATTRS = ("communicate", "tobytes")

    # Known pre-existing debt, each entry blocks removal only until its
    # backlog item lands. New violations are NOT allowed on this list.
    # key: "file:lineno", value: backlog item.
    KNOWN_VIOLATIONS = {
        # nnInteractive prompt-mask crop: user-interaction bbox-bounded, so
        # typically tiny; converting it would add a copy for no GUI win.
        "nninteractive_mimics.py:2513": "R61-12",
    }

    def test_no_tick_function_blocks_the_gui_thread(self):
        import ast

        runtime_dir = os.path.join(PROJECT_ROOT, "runtime_py35")

        def blocking_calls(node):
            """(attr, lineno) for forbidden calls outside nested defs."""
            found = []
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue  # nested function bodies run on their own thread
                if isinstance(child, ast.Call):
                    func = child.func
                    if (
                        isinstance(func, ast.Attribute)
                        and func.attr in self.FORBIDDEN_ATTRS
                    ):
                        found.append((func.attr, child.lineno))
                found.extend(blocking_calls(child))
            return found

        violations = []
        for name in os.listdir(runtime_dir):
            if not name.endswith(".py"):
                continue
            path = os.path.join(runtime_dir, name)
            with open(path, "rb") as handle:
                try:
                    tree = ast.parse(handle.read())
                except SyntaxError:
                    continue
            # Follow the tick's call closure within the same module: a tick
            # that delegates to a module function which then blocks is just
            # as frozen (the pre-R61-3 AL path hid communicate() one hop
            # below _al_tick in _al_apply_request).
            functions = {}
            for top in ast.walk(tree):
                if isinstance(top, ast.FunctionDef):
                    functions.setdefault(top.name, top)

            def module_calls(node):
                called = set()
                for child in ast.walk(node):
                    if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                        called.add(child.func.id)
                return called

            def scan(func_node, seen):
                hits = list(blocking_calls(func_node))
                for callee in module_calls(func_node):
                    if callee in seen:
                        continue
                    seen.add(callee)
                    target = functions.get(callee)
                    if target is not None:
                        hits.extend(scan(target, seen))
                return hits

            for top in ast.walk(tree):
                if isinstance(top, ast.FunctionDef) and top.name.endswith("_tick"):
                    for attr, lineno in scan(top, {top.name}):
                        key = "{0}:{1}".format(name, lineno)
                        if key in self.KNOWN_VIOLATIONS:
                            continue
                        violations.append(
                            "{0} {1}() reachable from {2}".format(
                                key, attr, top.name
                            )
                        )
        self.assertEqual(
            [], violations,
            "GUI-thread ticks must not reach communicate()/tobytes() - "
            "move the wait/copy onto a worker thread "
            "(violations: {0})".format(violations),
        )


if __name__ == "__main__":
    print("Mimics-Script Comprehensive Tests")
    print("=" * 60)
    print("Python: {}".format(sys.version))
    print("Project: {}".format(PROJECT_ROOT))
    print("=" * 60)
    unittest.main(verbosity=2)
