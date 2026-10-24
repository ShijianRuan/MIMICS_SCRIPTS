#!/usr/bin/env python3
"""Failure-path tests for the Mimics-side SLIP session manager.

runtime_py35/slip_mimics.py runs inside Mimics' embedded Python 3.5, so the
test imports it against the fake Mimics module from fake_mimics_flow_test.py
and drives the session state machine directly (job.json + results/ files),
never spawning the external worker.  Covered failure paths:

- image validation (dtype / modality / HU mapping),
- run() refusals while another task holds the mask buffer or gpu.lock,
- run() refusal when the external Python reports no CUDA (before any
  draft is created or buffer exported, SL06), plus the probe semantics:
  conclusive results are cached per process, inconclusive probes never
  block a start,
- run() resumes its own live session when its own worker holds gpu.lock
  (monitor-timer-loss recovery: prompt history kept, or pointed refusal
  when the session's result Mask is not on the active image),
- job state load/save (terminal jobs are not resumed),
- result handling: dead worker, worker errors, project switch, identity
  mismatch, manual mask change, empty prediction, unexpected statuses,
- command enqueueing (undo/reset never overwrite the retryable predict),
- closing: auto-created empty drafts are cleaned up, real work is kept,
- monitor tick: timeout surfaces a dialog, busy guard prevents re-entry,
- _start_job fails fast when the external Python environment is missing.

The GPU end-to-end bridge test (real model, real data) is a manual
acceptance item in docs/TEST_STRATEGY.md, not part of this suite.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
for value in (ROOT, TESTS, ROOT / "tools", ROOT / "runtime_py35"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from fake_mimics_flow_test import (  # noqa: E402
    FakeImage,
    FakeMask,
    FakeMimics,
    FakeVoxelBuffer,
    _u8_pattern_buffer,
    _voxel_count,
)


SHAPE = (2, 3, 4)


class SlipMimicsTests(unittest.TestCase):
    """Shared setUp: fresh fake Mimics, fresh runtime modules, temp dirs."""

    def setUp(self):
        self.original_mimics = sys.modules.get("mimics")
        self.fake = FakeMimics()
        sys.modules["mimics"] = self.fake
        for name in ("runtime_common", "nninteractive_mimics", "slip_mimics"):
            sys.modules.pop(name, None)
        self.runtime = importlib.import_module("slip_mimics")
        self.nnm = self.runtime.nnm
        self.runtime_common = self.runtime.runtime_common

        # Redirect every resource lock (gpu.lock) into a per-test directory so
        # tests never touch the developer machine's real lock state.
        self.old_lock_dir = os.environ.get("MIMICS_RESOURCE_LOCK_DIR")
        self.lock_dir = tempfile.mkdtemp(prefix="slip_test_locks_")
        os.environ["MIMICS_RESOURCE_LOCK_DIR"] = self.lock_dir

        self.tmp = tempfile.mkdtemp(prefix="slip_test_")
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        shutil.rmtree(self.lock_dir, ignore_errors=True)
        if self.old_lock_dir is None:
            os.environ.pop("MIMICS_RESOURCE_LOCK_DIR", None)
        else:
            os.environ["MIMICS_RESOURCE_LOCK_DIR"] = self.old_lock_dir
        for name in ("runtime_common", "nninteractive_mimics", "slip_mimics"):
            sys.modules.pop(name, None)
        if self.original_mimics is None:
            sys.modules.pop("mimics", None)
        else:
            sys.modules["mimics"] = self.original_mimics

    # -- helpers -----------------------------------------------------------

    def make_image(self, fmt="h"):
        buffer = FakeVoxelBuffer(SHAPE, fmt=fmt)
        image = FakeImage(buffer)
        self.fake.data.images = type(self.fake.data.images)([image])
        self.fake.data.images.set_active(image)
        return image

    def make_target(self, image, array=None, selected=False):
        mask = FakeMask("Target", image=image, array=array, selected=selected)
        self.fake.data.masks.append(mask)
        return mask

    def make_job_dir(self):
        job_dir = os.path.join(self.tmp, "job_under_test")
        for sub in ("inputs", "commands", "results"):
            os.makedirs(os.path.join(job_dir, sub))
        return job_dir

    def make_state(self, job_dir, image, target, **overrides):
        """A realistic mid-session state (one pending predict command)."""
        state = {
            "_job_dir": job_dir,
            "schema_version": "slip_async_job.v1",
            "job_id": "job_test",
            "status": "running",
            "worker_dir": job_dir,
            "image_guid": self.nnm._object_id(image),
            "image_name": str(getattr(image, "name", "")),
            "launch_project_path": self.nnm._current_project_path(),
            "target_guid": self.nnm._object_id(target),
            "target_name": str(getattr(target, "name", "")),
            "source_guid": self.nnm._object_id(target),
            "source_name": str(getattr(target, "name", "")),
            "write_mode": "in_place",
            "auto_created": False,
            "shape": list(SHAPE),
            "base_path": "",
            "base_pixel_count": 0,
            "expected_target_sha256": self.nnm._mask_sha256(target, list(SHAPE)),
            "interactions": [],
            "action_history_len": 0,
            "pending_sequence": 1,
            "applied_sequence": 0,
            "pid": None,
            "created_at_epoch": 0.0,
            "updated_at_epoch": 0.0,
        }
        state.update(overrides)
        return state

    def write_result(self, job_dir, sequence, payload):
        path = os.path.join(
            job_dir, "results", "result_{0:06d}.json".format(sequence))
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return path

    def write_output(self, array_or_buffer):
        """Persist a u8 mask as the worker would and return its path."""
        if isinstance(array_or_buffer, FakeVoxelBuffer):
            raw = array_or_buffer.tobytes()
        else:
            raw = bytes(array_or_buffer)
        path = os.path.join(self.tmp, "output.u8")
        with open(path, "wb") as handle:
            handle.write(raw)
        return path

    def answer(self, *choices):
        self.fake.dialogs.question_answers.extend(choices)

    def last_question(self):
        self.assertTrue(self.fake.dialogs.questions)
        return self.fake.dialogs.questions[-1]

    def assertNoDialogs(self):
        self.assertEqual(self.fake.dialogs.messages, [])
        self.assertEqual(self.fake.dialogs.questions, [])


# ---------------------------------------------------------------------------
# (a) 影像校验：dtype / 模态 / 强度映射
# ---------------------------------------------------------------------------

class SlipValidationTests(SlipMimicsTests):
    def test_int8_buffer_is_rejected(self):
        image = self.make_image(fmt="b")
        with self.assertRaisesRegex(RuntimeError, "不被 SLIP 支持"):
            self.runtime._validate_image(image)

    def test_mr_modality_is_accepted_without_hu_mapping(self):
        # SLIP 论文 Table 4 对所有模态共用同一强度管线：MR 不再被拒绝，
        # 也不要求 HU2GV（MR 无法恢复 HU 时按 GV 直通导出）。
        image = self.make_image()
        image.metadata.set(
            self.nnm.SOURCE_IMAGE_MODALITY_METADATA, "MR")
        self.assertIsNone(self.runtime._validate_image(image))

    def test_missing_hu_mapping_is_rejected_for_ct(self):
        image = self.make_image()
        original = self.nnm._hu_to_mimics_gv_transform
        self.nnm._hu_to_mimics_gv_transform = lambda: None
        try:
            with self.assertRaisesRegex(RuntimeError, "HU"):
                self.runtime._validate_image(image)
        finally:
            self.nnm._hu_to_mimics_gv_transform = original

    def test_valid_ct_image_passes(self):
        image = self.make_image()
        self.assertIsNone(self.runtime._validate_image(image))

    def test_ct_export_uses_hu_inverse(self):
        image = self.make_image()
        path = os.path.join(self.tmp, "ct_export.raw")
        result = self.runtime._export_image(image, path)
        slope, intercept = self.nnm._hu_to_mimics_gv_transform()
        self.assertEqual(
            result["to_hu_slope"], 1.0 / float(slope))
        self.assertEqual(
            result["to_hu_intercept"], -float(intercept) / float(slope))

    def test_mr_export_uses_source_intensity_mapping(self):
        image = self.make_image()
        image.metadata.set(
            self.nnm.SOURCE_IMAGE_MODALITY_METADATA, "MR")
        image.metadata.set(
            self.nnm.SOURCE_INTENSITY_RESCALE_SLOPE_METADATA, 2.5)
        image.metadata.set(
            self.nnm.SOURCE_INTENSITY_RESCALE_INTERCEPT_METADATA, -1024.0)
        path = os.path.join(self.tmp, "mr_export.raw")
        result = self.runtime._export_image(image, path)
        self.assertEqual(result["to_hu_slope"], 2.5)
        self.assertEqual(result["to_hu_intercept"], -1024.0)

    def test_mr_export_without_mapping_passes_gv_through(self):
        # 旧工程可能丢失 MR 强度映射：GV 直通（z-score 对 clamp 边界内的
        # 线性缩放不变），而不是拒绝会话。
        image = self.make_image()
        image.metadata.set(
            self.nnm.SOURCE_IMAGE_MODALITY_METADATA, "MR")
        path = os.path.join(self.tmp, "mr_passthrough.raw")
        result = self.runtime._export_image(image, path)
        self.assertEqual(result["to_hu_slope"], 1.0)
        self.assertEqual(result["to_hu_intercept"], 0.0)

    def test_other_modality_export_passes_gv_through(self):
        # PT 等其余模态：不做 HU 逆变换（HU 语义对它们不成立）。
        image = self.make_image()
        image.metadata.set(
            self.nnm.SOURCE_IMAGE_MODALITY_METADATA, "PT")
        path = os.path.join(self.tmp, "pt_export.raw")
        result = self.runtime._export_image(image, path)
        self.assertEqual(result["to_hu_slope"], 1.0)
        self.assertEqual(result["to_hu_intercept"], 0.0)


# ---------------------------------------------------------------------------
# (b) run() 拒绝路径：无影像 / 缓冲区占用 / gpu.lock 占用
# ---------------------------------------------------------------------------

class SlipRunRefusalTests(SlipMimicsTests):
    def test_main_without_active_image_shows_error(self):
        self.fake.data.images = type(self.fake.data.images)()
        self.assertEqual(self.runtime.main(), 2)
        self.assertEqual(len(self.fake.dialogs.messages), 1)
        self.assertIn("请先打开项目", self.fake.dialogs.messages[0]["message"])

    def test_run_refused_while_mask_buffer_is_busy(self):
        image = self.make_image()
        token = self.runtime_common.try_acquire_local_operation(
            "mask_buffer_access", "nnInteractive 导出")
        self.assertTrue(token)
        try:
            self.assertEqual(self.runtime.run(), 1)
            self.assertEqual(len(self.fake.dialogs.messages), 1)
            self.assertIn(
                "nnInteractive 导出",
                self.fake.dialogs.messages[0]["message"])
            # Nothing was created or asked before the refusal.
            self.assertNoDialogsExceptRefusal()
            self.assertEqual(list(self.fake.data.masks), [])
        finally:
            self.runtime_common.release_local_operation(
                "mask_buffer_access", token)

    def assertNoDialogsExceptRefusal(self):
        # The single refusal message box was recorded above; no question
        # boxes (write mode / prompts) may have appeared.
        self.assertEqual(self.fake.dialogs.questions, [])

    def patch_jobs_root(self):
        """把 job 扫描限制在本测试的临时目录内（hermeticity）。"""
        jobs_root = os.path.join(self.tmp, "slip_jobs")
        self.runtime._jobs_root = lambda root=None: jobs_root
        return jobs_root

    def acquire_gpu_lock_as_own_worker(self):
        # 模拟 SLIP worker 自己持有 gpu.lock：载荷里的 pid 即本进程。
        lock_path = self.runtime_common.resource_lock_path(
            str(self.tmp), "gpu.lock")
        token = self.runtime_common.acquire_resource_lock(
            lock_path, "gpu", "SLIP worker (pid {0})".format(os.getpid()))
        self.assertTrue(token)
        return lock_path, token

    def test_run_refused_while_gpu_lock_is_held(self):
        image = self.make_image()
        self.patch_jobs_root()
        lock_path = self.runtime_common.resource_lock_path(
            str(self.tmp), "gpu.lock")
        token = self.runtime_common.acquire_resource_lock(
            lock_path, "gpu", "nnInteractive session")
        self.assertTrue(token)
        try:
            self.assertEqual(self.runtime.run(), 1)
            self.assertEqual(len(self.fake.dialogs.messages), 1)
            message = self.fake.dialogs.messages[0]["message"]
            self.assertIn("GPU 当前正被", message)
            self.assertIn("nnInteractive session", message)
            self.assertEqual(self.fake.dialogs.questions, [])
            self.assertEqual(list(self.fake.data.masks), [])
        finally:
            self.runtime_common.release_resource_lock(lock_path, token)

    def test_run_resumes_own_session_when_own_worker_holds_gpu_lock(self):
        # 监控定时器失效后重新运行 SLIP 入口：自己的 worker 持有 gpu.lock
        # 时必须能取回会话——提示历史保留（撤销/重置入口仍在），完成时
        # 正常关闭，而不是被“GPU 被占用”拒之门外。
        image = self.make_image()
        target = self.make_target(image)
        jobs_root = self.patch_jobs_root()
        job_dir = os.path.join(jobs_root, "job_own")
        for sub in ("inputs", "commands", "results"):
            os.makedirs(os.path.join(job_dir, sub))
        state = self.make_state(
            job_dir, image, target,
            status="ready", pending_sequence=None, applied_sequence=1,
            pid=os.getpid(),
            interactions=[[1, 1, 1]], action_history_len=1)
        self.runtime._save_job(state)
        self.nnm._metadata_set(
            target, self.runtime.ASYNC_JOB_METADATA, job_dir)
        lock_path, token = self.acquire_gpu_lock_as_own_worker()
        try:
            self.answer(self.runtime.BUTTON_FINISH)
            self.assertEqual(self.runtime.run(), 0)
            menu = self.fake.dialogs.questions[0]
            buttons = menu["buttons"].split(";")
            self.assertIn(self.runtime.BUTTON_UNDO, buttons)
            self.assertIn(self.runtime.BUTTON_RESET, buttons)
            with open(os.path.join(job_dir, "close.json"),
                      encoding="utf-8") as handle:
                self.assertEqual(
                    json.load(handle)["reason"], "user_finished")
            # 会话被取回而不是另起炉灶。
            self.assertEqual(os.listdir(jobs_root), ["job_own"])
            self.assertEqual(
                self.nnm._metadata_get(
                    target, self.runtime.ASYNC_JOB_METADATA, ""), "")
        finally:
            self.runtime_common.release_resource_lock(lock_path, token)

    def test_run_refused_without_cuda_before_any_draft_or_export(self):
        # SL06：无 CUDA 的机器在创建草稿/导出缓冲之前明确拒绝，不进入
        # 必失败的模型加载。
        image = self.make_image()
        self.patch_jobs_root()
        self.runtime._external_cuda_available = lambda root: False
        self.assertEqual(self.runtime.run(), 1)
        self.assertEqual(len(self.fake.dialogs.messages), 1)
        message = self.fake.dialogs.messages[0]["message"]
        self.assertIn("没有可用的 CUDA GPU", message)
        self.assertNoDialogsExceptRefusal()
        self.assertEqual(list(self.fake.data.masks), [])
        self.assertFalse(os.path.isdir(
            os.path.join(self.tmp, "slip_jobs")))

    def test_run_refused_when_own_session_target_not_on_image(self):
        # 自己的 worker 活着但结果 Mask 不在当前影像上：给出指向该会话
        # 的提示（含 Mask 名），且不误关后台会话。
        image = self.make_image()
        other = self.make_target(image)
        jobs_root = self.patch_jobs_root()
        job_dir = os.path.join(jobs_root, "job_elsewhere")
        for sub in ("inputs", "commands", "results"):
            os.makedirs(os.path.join(job_dir, sub))
        state = self.make_state(
            job_dir, image, other,
            status="ready", pending_sequence=None, pid=os.getpid(),
            target_guid="bogus-guid", target_name="另一个序列的 AI 草稿")
        self.runtime._save_job(state)
        lock_path, token = self.acquire_gpu_lock_as_own_worker()
        try:
            self.assertEqual(self.runtime.run(), 1)
            self.assertEqual(len(self.fake.dialogs.messages), 1)
            message = self.fake.dialogs.messages[0]["message"]
            self.assertIn("进行中的会话", message)
            self.assertIn("另一个序列的 AI 草稿", message)
            self.assertEqual(self.fake.dialogs.questions, [])
            self.assertFalse(
                os.path.isfile(os.path.join(job_dir, "close.json")))
        finally:
            self.runtime_common.release_resource_lock(lock_path, token)


# ---------------------------------------------------------------------------
# (b2) CUDA 能力预检：确证结果缓存、探测失败不拦启动（SL06）
# ---------------------------------------------------------------------------

class SlipCudaProbeTests(SlipMimicsTests):
    """_external_cuda_available 的探测语义，不拉起真实子进程。"""

    def run_probe(self, returncode, stdout):
        calls = []

        class FakeProcess(object):
            def communicate(self, timeout=None):
                return (stdout, b"")

        FakeProcess.returncode = returncode
        original_popen = self.runtime.subprocess.Popen
        original_python = self.runtime_common.find_external_python
        self.runtime.subprocess.Popen = lambda command, **kw: (
            calls.append(command), FakeProcess())[1]
        self.runtime_common.find_external_python = lambda *a, **k: sys.executable
        try:
            result = self.runtime._external_cuda_available("fake_root")
        finally:
            self.runtime.subprocess.Popen = original_popen
            self.runtime_common.find_external_python = original_python
        return result, calls

    def test_probe_reports_no_cuda_and_caches_result(self):
        result, calls = self.run_probe(0, b"0\n")
        self.assertIs(result, False)
        self.assertEqual(len(calls), 1)
        self.assertIn("torch.cuda.is_available", calls[0][2])
        # 确证结果按进程缓存：第二次探测不再拉起子进程。
        result_again, calls_again = self.run_probe(0, b"1\n")
        self.assertIs(result_again, False)
        self.assertEqual(calls_again, [])

    def test_probe_reports_cuda_available(self):
        result, calls = self.run_probe(0, b"1\n")
        self.assertIs(result, True)
        self.assertEqual(len(calls), 1)

    def test_probe_failure_is_inconclusive_and_not_cached(self):
        # 探测本身失败（如 import torch 崩溃）返回 None：不确定时绝不
        # 拦启动，让 worker 报出真实错误；None 也不缓存。
        result, _ = self.run_probe(1, b"")
        self.assertIsNone(result)
        result_again, calls_again = self.run_probe(0, b"0\n")
        self.assertIs(result_again, False)
        self.assertEqual(len(calls_again), 1)


# ---------------------------------------------------------------------------
# (c) job 状态读写：终态不复活、下划线键不落盘
# ---------------------------------------------------------------------------

class SlipJobStateTests(SlipMimicsTests):
    def test_load_job_ignores_terminal_state(self):
        image = self.make_image()
        target = self.make_target(image)
        job_dir = self.make_job_dir()
        with open(os.path.join(job_dir, "job.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({"status": "closed"}, handle)
        self.nnm._metadata_set(
            target, self.runtime.ASYNC_JOB_METADATA, job_dir)
        self.assertIsNone(self.runtime._load_job(target))
        self.assertEqual(
            self.nnm._metadata_get(
                target, self.runtime.ASYNC_JOB_METADATA, ""), "")

    def test_load_job_ignores_closed_worker(self):
        image = self.make_image()
        target = self.make_target(image)
        job_dir = self.make_job_dir()
        with open(os.path.join(job_dir, "job.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({"status": "ready"}, handle)
        with open(os.path.join(job_dir, "close.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({"reason": "user_finished"}, handle)
        self.nnm._metadata_set(
            target, self.runtime.ASYNC_JOB_METADATA, job_dir)
        self.assertIsNone(self.runtime._load_job(target))
        self.assertEqual(
            self.nnm._metadata_get(
                target, self.runtime.ASYNC_JOB_METADATA, ""), "")

    def test_load_job_resumes_active_state(self):
        image = self.make_image()
        target = self.make_target(image)
        job_dir = self.make_job_dir()
        with open(os.path.join(job_dir, "job.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({"status": "ready", "pending_sequence": None}, handle)
        self.nnm._metadata_set(
            target, self.runtime.ASYNC_JOB_METADATA, job_dir)
        state = self.runtime._load_job(target)
        self.assertIsNotNone(state)
        self.assertEqual(state["_job_dir"], job_dir)
        self.assertEqual(state["status"], "ready")

    def test_save_job_strips_private_keys(self):
        image = self.make_image()
        target = self.make_target(image)
        job_dir = self.make_job_dir()
        state = self.make_state(job_dir, image, target)
        self.runtime._save_job(state)
        with open(os.path.join(job_dir, "job.json"), encoding="utf-8") as handle:
            saved = json.load(handle)
        self.assertNotIn("_job_dir", saved)
        self.assertEqual(saved["schema_version"], "slip_async_job.v1")
        self.assertEqual(saved["pending_sequence"], 1)


# ---------------------------------------------------------------------------
# (d) 结果处理：完整的失败路径矩阵
# ---------------------------------------------------------------------------

class SlipHandleResultTests(SlipMimicsTests):
    def setUp(self):
        super().setUp()
        self.image = self.make_image()
        self.target = self.make_target(self.image)
        self.job_dir = self.make_job_dir()

    def test_no_pending_command_returns_ready(self):
        state = self.make_state(
            self.job_dir, self.image, self.target, pending_sequence=None)
        self.assertEqual(
            self.runtime._handle_result(self.image, self.target, state), "ready")
        self.assertNoDialogs()

    def test_missing_result_with_live_worker_waits(self):
        state = self.make_state(self.job_dir, self.image, self.target)
        # pid None counts as "no live pid to disprove", no worker_status:
        # the job may still be starting, so the client keeps waiting.
        self.assertEqual(
            self.runtime._handle_result(self.image, self.target, state),
            "waiting")
        self.assertNoDialogs()

    def test_dead_worker_interactive_cancel_keeps_waiting(self):
        state = self.make_state(self.job_dir, self.image, self.target)
        with open(os.path.join(self.job_dir, "worker_status.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({
                "status": "failed",
                "stage": "load_model_and_image",
                "error": "boom"}, handle)
        self.answer(self.runtime.BUTTON_CANCEL)
        self.assertEqual(
            self.runtime._handle_result(self.image, self.target, state),
            "waiting")
        self.assertIn("停止了", self.last_question()["message"])

    def test_dead_worker_restart_closes_job(self):
        state = self.make_state(self.job_dir, self.image, self.target)
        with open(os.path.join(self.job_dir, "worker_status.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({
                "status": "failed",
                "stage": "load_model_and_image",
                "error": "boom"}, handle)
        self.answer(self.runtime.BUTTON_START_CURRENT)
        self.assertEqual(
            self.runtime._handle_result(self.image, self.target, state),
            "restart")
        self.assertTrue(
            os.path.isfile(os.path.join(self.job_dir, "close.json")))
        self.assertEqual(
            self.nnm._metadata_get(
                self.target, self.runtime.ASYNC_JOB_METADATA, ""), "")

    def test_dead_worker_non_interactive_raises(self):
        state = self.make_state(self.job_dir, self.image, self.target)
        with open(os.path.join(self.job_dir, "worker_status.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({
                "status": "failed",
                "stage": "load_model_and_image",
                "error": "boom"}, handle)
        with self.assertRaisesRegex(RuntimeError, "阶段"):
            self.runtime._handle_result(
                self.image, self.target, state, interactive=False)

    def test_half_written_result_waits(self):
        state = self.make_state(self.job_dir, self.image, self.target)
        self.write_result(self.job_dir, 1, {})
        self.assertEqual(
            self.runtime._handle_result(self.image, self.target, state),
            "waiting")
        self.assertNoDialogs()

    def test_nothing_to_undo_error_is_soft(self):
        state = self.make_state(self.job_dir, self.image, self.target)
        self.write_result(self.job_dir, 1, {
            "status": "error", "error": "Nothing to undo",
            "stage": "prediction"})
        self.assertEqual(
            self.runtime._handle_result(self.image, self.target, state),
            "ready")
        self.assertEqual(len(self.fake.dialogs.messages), 1)
        self.assertIn("撤销失败", self.fake.dialogs.messages[0]["message"])

    def test_error_retry_without_predict_is_soft(self):
        state = self.make_state(self.job_dir, self.image, self.target)
        self.write_result(self.job_dir, 1, {
            "status": "error", "error": "CUDA out of memory",
            "stage": "prediction"})
        self.answer(self.runtime.BUTTON_RETRY)
        self.assertEqual(
            self.runtime._handle_result(self.image, self.target, state),
            "ready")
        self.assertEqual(len(self.fake.dialogs.messages), 1)
        self.assertIn("没有可重试的提示",
                      self.fake.dialogs.messages[0]["message"])

    def test_error_retry_reenqueues_last_predict(self):
        state = self.make_state(
            self.job_dir, self.image, self.target,
            last_command={"action": "predict", "click": [0, 0, 0], "label": 1})
        # The original predict command file exists from the failed attempt,
        # so the retry must be numbered after it.
        with open(os.path.join(
                self.job_dir, "commands", "command_000001.json"), "w",
                encoding="utf-8") as handle:
            json.dump({"sequence": 1, "command_id": "cmd000001",
                       "action": "predict"}, handle)
        self.write_result(self.job_dir, 1, {
            "status": "error", "error": "CUDA out of memory",
            "stage": "prediction"})
        self.answer(self.runtime.BUTTON_RETRY)
        self.assertEqual(
            self.runtime._handle_result(self.image, self.target, state),
            "waiting")
        command_path = os.path.join(
            self.job_dir, "commands", "command_000002.json")
        with open(command_path, encoding="utf-8") as handle:
            command = json.load(handle)
        self.assertEqual(command["action"], "predict")
        self.assertEqual(command["label"], 1)

    def test_error_discard_ends_session(self):
        state = self.make_state(self.job_dir, self.image, self.target)
        self.write_result(self.job_dir, 1, {
            "status": "error", "error": "CUDA out of memory",
            "stage": "prediction"})
        self.answer(self.runtime.BUTTON_DISCARD_SESSION)
        self.assertEqual(
            self.runtime._handle_result(self.image, self.target, state),
            "ended")
        with open(os.path.join(self.job_dir, "close.json"),
                  encoding="utf-8") as handle:
            close = json.load(handle)
        self.assertEqual(close["reason"], "worker_error_discarded")

    def test_error_cancel_returns_ready(self):
        state = self.make_state(self.job_dir, self.image, self.target)
        self.write_result(self.job_dir, 1, {
            "status": "error", "error": "CUDA out of memory",
            "stage": "prediction"})
        self.answer(self.runtime.BUTTON_CANCEL)
        self.assertEqual(
            self.runtime._handle_result(self.image, self.target, state),
            "ready")

    def test_project_change_closes_session(self):
        state = self.make_state(
            self.job_dir, self.image, self.target,
            launch_project_path=os.path.join(self.tmp, "other_project.mcs"))
        output = self.write_output(_u8_pattern_buffer(SHAPE, lambda x, y, z: True))
        self.write_result(self.job_dir, 1, {
            "status": "refined", "output_path": output, "voxels": 24,
            "action": "predict", "action_history_len": 1})
        self.assertEqual(
            self.runtime._handle_result(self.image, self.target, state),
            "ended")
        with open(os.path.join(self.job_dir, "close.json"),
                  encoding="utf-8") as handle:
            close = json.load(handle)
        self.assertEqual(close["reason"], "project_changed")
        # The old project's mask is untouched in the new project context.
        self.assertEqual(self.target.number_of_pixels, 0)

    def test_identity_mismatch_raises(self):
        state = self.make_state(
            self.job_dir, self.image, self.target, image_guid="wrong")
        output = self.write_output(_u8_pattern_buffer(SHAPE, lambda x, y, z: True))
        self.write_result(self.job_dir, 1, {
            "status": "refined", "output_path": output, "voxels": 24,
            "action": "predict", "action_history_len": 1})
        with self.assertRaisesRegex(RuntimeError, "不匹配"):
            self.runtime._handle_result(self.image, self.target, state)

    def test_reset_result_restores_base_mask(self):
        pattern = _u8_pattern_buffer(
            SHAPE, lambda x, y, z: (x + y + z) % 2 == 0)
        base_path = self.write_output(pattern)
        state = self.make_state(
            self.job_dir, self.image, self.target,
            base_path=base_path,
            interactions=[{"click": [0, 0, 0], "label": 1}],
            action_history_len=1)
        self.write_result(self.job_dir, 1, {"status": "reset"})
        self.assertEqual(
            self.runtime._handle_result(self.image, self.target, state),
            "applied")
        self.assertEqual(
            self.target.number_of_pixels, pattern.tobytes().count(1))
        self.assertEqual(state["interactions"], [])
        self.assertEqual(state["action_history_len"], 0)
        self.assertIsNone(state["pending_sequence"])

    def test_reset_with_equal_volume_edit_shows_conflict_restart(self):
        # SL01：等体素数的边界移动不改变 number_of_pixels，只有内容哈希
        # 能发现它——reset 回执到达时不得把它静默覆盖回基线。
        before_edit = _u8_pattern_buffer(
            SHAPE, lambda x, y, z: (x + y + z) % 2 == 0)
        self.target.set_voxel_buffer(before_edit)
        base_path = os.path.join(self.tmp, "base.u8")
        with open(base_path, "wb") as handle:
            handle.write(before_edit.tobytes())
        state = self.make_state(
            self.job_dir, self.image, self.target,
            base_path=base_path,
            interactions=[{"click": [0, 0, 0], "label": 1}],
            action_history_len=1)
        # 会话外的人工编辑：体素数相同、位置互补。
        edited = _u8_pattern_buffer(
            SHAPE, lambda x, y, z: (x + y + z) % 2 == 1)
        self.target.set_voxel_buffer(edited)
        self.assertEqual(
            self.target.number_of_pixels, before_edit.tobytes().count(1))
        self.write_result(self.job_dir, 1, {"status": "reset"})
        self.answer(self.runtime.BUTTON_START_CURRENT)
        self.assertEqual(
            self.runtime._handle_result(self.image, self.target, state),
            "restart")
        # 人工编辑被保留，没有被 reset 覆盖回基线。
        self.assertEqual(
            self.target.get_voxel_buffer().tobytes(), edited.tobytes())
        with open(os.path.join(self.job_dir, "close.json"),
                  encoding="utf-8") as handle:
            close = json.load(handle)
        self.assertEqual(close["reason"], "manual_mask_change")

    def test_reset_with_equal_volume_edit_cancel_ends_session(self):
        before_edit = _u8_pattern_buffer(
            SHAPE, lambda x, y, z: (x + y + z) % 2 == 0)
        self.target.set_voxel_buffer(before_edit)
        state = self.make_state(
            self.job_dir, self.image, self.target,
            interactions=[{"click": [0, 0, 0], "label": 1}],
            action_history_len=1)
        edited = _u8_pattern_buffer(
            SHAPE, lambda x, y, z: (x + y + z) % 2 == 1)
        self.target.set_voxel_buffer(edited)
        self.write_result(self.job_dir, 1, {"status": "reset"})
        self.answer(self.runtime.BUTTON_CANCEL)
        self.assertEqual(
            self.runtime._handle_result(self.image, self.target, state),
            "ended")
        self.assertEqual(
            self.target.get_voxel_buffer().tobytes(), edited.tobytes())

    def test_unexpected_result_status_raises(self):
        state = self.make_state(self.job_dir, self.image, self.target)
        self.write_result(self.job_dir, 1, {"status": "weird"})
        with self.assertRaisesRegex(RuntimeError, "意外的结果状态"):
            self.runtime._handle_result(self.image, self.target, state)

    def test_manual_mask_change_restart_closes_job(self):
        state = self.make_state(
            self.job_dir, self.image, self.target,
            expected_target_sha256="stale")
        output = self.write_output(_u8_pattern_buffer(SHAPE, lambda x, y, z: True))
        self.write_result(self.job_dir, 1, {
            "status": "refined", "output_path": output, "voxels": 24,
            "action": "predict", "action_history_len": 1})
        self.answer(self.runtime.BUTTON_START_CURRENT)
        self.assertEqual(
            self.runtime._handle_result(self.image, self.target, state),
            "restart")
        with open(os.path.join(self.job_dir, "close.json"),
                  encoding="utf-8") as handle:
            close = json.load(handle)
        self.assertEqual(close["reason"], "manual_mask_change")

    def test_manual_mask_change_cancel_ends_session(self):
        state = self.make_state(
            self.job_dir, self.image, self.target,
            expected_target_sha256="stale")
        output = self.write_output(_u8_pattern_buffer(SHAPE, lambda x, y, z: True))
        self.write_result(self.job_dir, 1, {
            "status": "refined", "output_path": output, "voxels": 24,
            "action": "predict", "action_history_len": 1})
        self.answer(self.runtime.BUTTON_CANCEL)
        self.assertEqual(
            self.runtime._handle_result(self.image, self.target, state),
            "ended")

    def test_empty_prediction_kept_leaves_mask_unchanged(self):
        state = self.make_state(self.job_dir, self.image, self.target)
        pattern = _u8_pattern_buffer(SHAPE, lambda x, y, z: x == 0)
        self.target.set_voxel_buffer(pattern)
        state["expected_target_sha256"] = self.nnm._mask_sha256(
            self.target, list(SHAPE))
        output = self.write_output(bytes(_voxel_count(SHAPE)))
        self.write_result(self.job_dir, 1, {
            "status": "refined", "output_path": output, "voxels": 0,
            "action": "predict", "action_history_len": 1})
        self.answer(self.runtime.BUTTON_KEEP)
        self.assertEqual(
            self.runtime._handle_result(self.image, self.target, state),
            "ready")
        self.assertEqual(self.target.number_of_pixels, pattern.tobytes().count(1))
        # 被拒绝的点击仍留在 worker 撤销历史中：本地计数必须同步，
        # 否则菜单不会给出撤销入口（R88-5）。
        self.assertEqual(state["action_history_len"], 1)
        self.assertIn("撤销历史", self.last_question()["message"])

    def test_empty_prediction_applied_clears_mask(self):
        state = self.make_state(self.job_dir, self.image, self.target)
        pattern = _u8_pattern_buffer(SHAPE, lambda x, y, z: x == 0)
        self.target.set_voxel_buffer(pattern)
        state["expected_target_sha256"] = self.nnm._mask_sha256(
            self.target, list(SHAPE))
        output = self.write_output(bytes(_voxel_count(SHAPE)))
        self.write_result(self.job_dir, 1, {
            "status": "refined", "output_path": output, "voxels": 0,
            "action": "predict", "action_history_len": 1})
        self.answer(self.runtime.BUTTON_APPLY_EMPTY)
        self.assertEqual(
            self.runtime._handle_result(self.image, self.target, state),
            "applied")
        self.assertEqual(self.target.number_of_pixels, 0)

    def test_refined_result_is_applied_to_target(self):
        state = self.make_state(
            self.job_dir, self.image, self.target,
            interactions=[{"click": [0, 0, 0], "label": 1}])
        pattern = _u8_pattern_buffer(
            SHAPE, lambda x, y, z: (x + z) % 2 == 0)
        output = self.write_output(pattern)
        self.write_result(self.job_dir, 1, {
            "status": "refined", "output_path": output,
            "voxels": pattern.tobytes().count(1),
            "action": "predict", "action_history_len": 1,
            "interaction_seconds": 0.6})
        self.assertEqual(
            self.runtime._handle_result(self.image, self.target, state),
            "applied")
        self.assertEqual(
            self.target.number_of_pixels, pattern.tobytes().count(1))
        self.assertIsNone(state["pending_sequence"])
        self.assertEqual(state["applied_sequence"], 1)
        self.assertEqual(state["action_history_len"], 1)
        self.assertTrue(self.target.visible)
        self.assertTrue(self.target.selected)

    def test_undo_result_pops_last_interaction(self):
        state = self.make_state(
            self.job_dir, self.image, self.target,
            interactions=[
                {"click": [0, 0, 0], "label": 1},
                {"click": [1, 1, 1], "label": 0},
            ],
            action_history_len=2)
        pattern = _u8_pattern_buffer(SHAPE, lambda x, y, z: x == 1)
        output = self.write_output(pattern)
        self.write_result(self.job_dir, 1, {
            "status": "refined", "output_path": output,
            "voxels": pattern.tobytes().count(1),
            "action": "undo", "action_history_len": 1})
        self.assertEqual(
            self.runtime._handle_result(self.image, self.target, state),
            "applied")
        self.assertEqual(
            state["interactions"], [{"click": [0, 0, 0], "label": 1}])
        self.assertEqual(state["action_history_len"], 1)

    def test_undo_to_zero_with_nonempty_base_restores_base(self):
        # SL03：上游把"撤销第一个提示"实现为回到空模型状态（空输出），
        # 但本会话的基线是会话开始时的 Mask 内容——非空基线不能被一次
        # 撤销清空，结果必须逐体素等于基线而不是模型的空输出。
        base_pattern = _u8_pattern_buffer(SHAPE, lambda x, y, z: x == 0)
        base_path = os.path.join(self.tmp, "base.u8")
        with open(base_path, "wb") as handle:
            handle.write(base_pattern.tobytes())
        state = self.make_state(
            self.job_dir, self.image, self.target,
            base_path=base_path,
            interactions=[{"click": [0, 0, 0], "label": 1}],
            action_history_len=1)
        empty_output = self.write_output(bytes(_voxel_count(SHAPE)))
        self.write_result(self.job_dir, 1, {
            "status": "refined", "output_path": empty_output, "voxels": 0,
            "action": "undo", "action_history_len": 0})
        self.assertEqual(
            self.runtime._handle_result(self.image, self.target, state),
            "applied")
        self.assertEqual(
            self.target.get_voxel_buffer().tobytes(), base_pattern.tobytes())
        self.assertEqual(state["interactions"], [])
        self.assertEqual(state["action_history_len"], 0)
        self.assertIsNone(state["pending_sequence"])
        self.assertEqual(
            state["expected_target_sha256"],
            self.nnm._mask_sha256(self.target, list(SHAPE)))

    def test_undo_to_zero_with_empty_base_empties_mask(self):
        # 空基线的同一流程才应为空（与上游语义一致，不引入额外对话框）。
        base_path = os.path.join(self.tmp, "base.u8")
        with open(base_path, "wb") as handle:
            handle.write(bytes(_voxel_count(SHAPE)))
        state = self.make_state(
            self.job_dir, self.image, self.target,
            base_path=base_path,
            interactions=[{"click": [0, 0, 0], "label": 1}],
            action_history_len=1)
        empty_output = self.write_output(bytes(_voxel_count(SHAPE)))
        self.write_result(self.job_dir, 1, {
            "status": "refined", "output_path": empty_output, "voxels": 0,
            "action": "undo", "action_history_len": 0})
        self.assertEqual(
            self.runtime._handle_result(self.image, self.target, state),
            "applied")
        self.assertEqual(self.target.number_of_pixels, 0)
        self.assertEqual(state["interactions"], [])
        self.assertEqual(state["action_history_len"], 0)

    def test_missing_output_file_raises(self):
        state = self.make_state(self.job_dir, self.image, self.target)
        self.write_result(self.job_dir, 1, {
            "status": "refined",
            "output_path": os.path.join(self.tmp, "missing.u8"),
            "voxels": 5, "action": "predict", "action_history_len": 1})
        with self.assertRaisesRegex(RuntimeError, "结果文件缺失"):
            self.runtime._handle_result(self.image, self.target, state)


# ---------------------------------------------------------------------------
# (e) 命令入队：predict 可重试、undo/reset 不覆盖
# ---------------------------------------------------------------------------

class SlipEnqueueCommandTests(SlipMimicsTests):
    def test_predict_then_undo_command_sequence(self):
        image = self.make_image()
        target = self.make_target(image)
        job_dir = self.make_job_dir()
        state = self.make_state(
            job_dir, image, target, pending_sequence=None, status="ready")

        predict_payload = {
            "action": "predict", "click": [1, 2, 3], "label": 1}
        sequence = self.runtime._enqueue_command(state, predict_payload)
        self.assertEqual(sequence, 1)
        with open(os.path.join(job_dir, "commands", "command_000001.json"),
                  encoding="utf-8") as handle:
            command = json.load(handle)
        self.assertEqual(command["sequence"], 1)
        self.assertEqual(command["command_id"], "cmd000001")
        self.assertEqual(command["action"], "predict")
        self.assertEqual(command["click"], [1, 2, 3])
        self.assertEqual(state["last_command"], predict_payload)
        self.assertEqual(state["pending_sequence"], 1)
        self.assertEqual(state["status"], "running")

        # undo does not overwrite the retryable predict command.
        self.runtime._enqueue_command(state, {"action": "undo"})
        self.assertEqual(state["last_command"], predict_payload)
        self.assertEqual(state["pending_sequence"], 2)
        with open(os.path.join(job_dir, "commands", "command_000002.json"),
                  encoding="utf-8") as handle:
            command = json.load(handle)
        self.assertEqual(command["action"], "undo")


# ---------------------------------------------------------------------------
# (f) 会话关闭：自动创建的空草稿清理，真实成果保留
# ---------------------------------------------------------------------------

class SlipCloseJobTests(SlipMimicsTests):
    def test_auto_created_empty_draft_deleted_on_user_finish(self):
        image = self.make_image()
        target = self.make_target(image)
        job_dir = self.make_job_dir()
        state = self.make_state(
            job_dir, image, target, auto_created=True, interactions=[])
        self.nnm._metadata_set(
            target, self.runtime.ASYNC_JOB_METADATA, job_dir)
        self.runtime._close_job(target, state, "user_finished")
        # _delete_mimics_object tries mimics.data.points.delete first; the
        # fake collections never raise, so the delete request is recorded
        # there (a real Mimics session removes the mask from the project).
        self.assertIn(target, self.fake.data.points.deleted)

    def test_auto_created_draft_with_interactions_is_kept(self):
        image = self.make_image()
        target = self.make_target(image)
        job_dir = self.make_job_dir()
        state = self.make_state(
            job_dir, image, target, auto_created=True,
            interactions=[{"click": [0, 0, 0], "label": 1}])
        self.runtime._close_job(target, state, "user_finished")
        self.assertIn(target, self.fake.data.masks)
        self.assertNotIn(target, self.fake.data.points.deleted)

    def test_manual_target_is_never_deleted(self):
        image = self.make_image()
        target = self.make_target(image)
        job_dir = self.make_job_dir()
        state = self.make_state(
            job_dir, image, target, auto_created=False, interactions=[])
        self.runtime._close_job(target, state, "user_finished")
        self.assertIn(target, self.fake.data.masks)
        self.assertNotIn(target, self.fake.data.points.deleted)

    def test_close_writes_close_json_and_clears_metadata(self):
        image = self.make_image()
        target = self.make_target(image)
        job_dir = self.make_job_dir()
        state = self.make_state(job_dir, image, target)
        self.nnm._metadata_set(
            target, self.runtime.ASYNC_JOB_METADATA, job_dir)
        self.runtime._close_job(target, state, "worker_stopped")
        with open(os.path.join(job_dir, "close.json"),
                  encoding="utf-8") as handle:
            close = json.load(handle)
        self.assertEqual(close["reason"], "worker_stopped")
        self.assertEqual(state["status"], "closing")
        self.assertEqual(
            self.nnm._metadata_get(
                target, self.runtime.ASYNC_JOB_METADATA, ""), "")


# ---------------------------------------------------------------------------
# (g) 监控定时器：超时弹窗与 busy 防重入
# ---------------------------------------------------------------------------

class SlipMonitorTickTests(SlipMimicsTests):
    def make_monitor(self, job_dir, **overrides):
        state = self.make_state(job_dir, self.image, self.target)
        monitor = {
            "timer": None,
            "image": self.image,
            "target": self.target,
            "state": state,
            "done": False,
            "busy": False,
            "deadline": 0.0,
            "timeout_seconds": 600.0,
        }
        monitor.update(overrides)
        self.runtime._ASYNC_MONITORS[job_dir] = monitor
        return monitor

    def setUp(self):
        super().setUp()
        self.image = self.make_image()
        self.target = self.make_target(self.image)
        self.job_dir = self.make_job_dir()

    def test_timeout_marks_done_and_shows_failure(self):
        monitor = self.make_monitor(self.job_dir)
        self.runtime._monitor_tick(monitor)
        self.assertTrue(monitor["done"])
        self.assertEqual(len(self.fake.dialogs.messages), 1)
        message = self.fake.dialogs.messages[0]["message"]
        self.assertIn("SLIP 后台预测失败", message)
        self.assertIn("600", message)
        self.assertNotIn(self.job_dir, self.runtime._ASYNC_MONITORS)

    def test_busy_monitor_is_skipped(self):
        monitor = self.make_monitor(self.job_dir, busy=True)
        self.runtime._monitor_tick(monitor)
        self.assertFalse(monitor["done"])
        self.assertTrue(monitor["busy"])
        self.assertNoDialogs()
        self.assertIn(self.job_dir, self.runtime._ASYNC_MONITORS)


# ---------------------------------------------------------------------------
# (h) _start_job 快速失败：缺少外部 Python 环境
# ---------------------------------------------------------------------------

class SlipStartJobFailureTests(SlipMimicsTests):
    def test_missing_external_python_fails_before_any_directory(self):
        image = self.make_image()
        target = self.make_target(image)
        original = self.runtime_common.find_external_python
        self.runtime_common.find_external_python = lambda *a, **k: ""
        try:
            with self.assertRaisesRegex(RuntimeError, "未找到外部 Python"):
                self.runtime._start_job(image, target)
        finally:
            self.runtime_common.find_external_python = original
        # The environment check happens before the job directory is created.
        self.assertFalse(
            os.path.isdir(os.path.join(self.tmp, "async_jobs")))
        self.assertEqual(
            self.nnm._metadata_get(
                target, self.runtime.ASYNC_JOB_METADATA, ""), "")

    def test_missing_deps_dir_fails_before_any_directory(self):
        """A half-copied deployment (checkpoint without deps) fails fast.

        slip_models/deps carries the vendored dependencies the worker needs
        at import time; without it the worker would die after the job
        directory, input snapshot, and GPU lock are already in play. The
        preflight must reject before any of that state exists (SL05).
        """
        image = self.make_image()
        target = self.make_target(image)
        fake_root = os.path.join(self.tmp, "half_deployed_root")
        os.makedirs(os.path.join(fake_root, "slip_models"))
        with open(os.path.join(fake_root, "slip_bridge.py"), "w") as handle:
            handle.write("# stub\n")
        with open(
                os.path.join(fake_root, "slip_models", "SLIP_ckpt.pth"),
                "w") as handle:
            handle.write("stub")
        original_python = self.runtime_common.find_external_python
        original_root = self.runtime._project_root
        self.runtime_common.find_external_python = lambda *a, **k: "python.exe"
        self.runtime._project_root = lambda: fake_root
        try:
            with self.assertRaisesRegex(RuntimeError, "未找到 SLIP 依赖目录"):
                self.runtime._start_job(image, target)
        finally:
            self.runtime_common.find_external_python = original_python
            self.runtime._project_root = original_root
        # The deps check happens before the job directory is created.
        self.assertFalse(os.path.isdir(os.path.join(
            fake_root, ".mimics_runtime", "slip", "async_jobs")))
        self.assertEqual(
            self.nnm._metadata_get(
                target, self.runtime.ASYNC_JOB_METADATA, ""), "")


# ---------------------------------------------------------------------------
# (i) 目标 Mask 选择：自动草稿 / 多选 / 写入方式 / 会话冲突
# ---------------------------------------------------------------------------

class SlipSelectTargetTests(SlipMimicsTests):
    def test_no_selection_auto_creates_draft(self):
        image = self.make_image()
        session = self.runtime._select_session_target(image)
        self.assertTrue(session["auto_created"])
        self.assertEqual(session["write_mode"], "new_empty")
        self.assertIs(session["source"], session["target"])
        self.assertEqual(session["target"].name, "SLIP Result")
        self.assertTrue(self.nnm._is_ai_draft(session["target"]))
        self.assertNoDialogs()

    def test_multiple_selections_are_rejected(self):
        image = self.make_image()
        self.make_target(image, selected=True)
        self.make_target(image, selected=True)
        with self.assertRaisesRegex(RuntimeError, "只选择一个"):
            self.runtime._select_session_target(image)

    def test_manual_mask_update_in_place(self):
        image = self.make_image()
        pattern = _u8_pattern_buffer(SHAPE, lambda x, y, z: True)
        source = self.make_target(image, array=pattern)
        source.selected = True
        self.answer("更新所选 Mask")
        session = self.runtime._select_session_target(image)
        self.assertEqual(session["write_mode"], "in_place")
        self.assertIs(session["source"], source)
        self.assertIs(session["target"], source)
        self.assertFalse(session["auto_created"])

    def test_manual_mask_creates_derived_draft(self):
        image = self.make_image()
        pattern = _u8_pattern_buffer(SHAPE, lambda x, y, z: True)
        source = self.make_target(image, array=pattern)
        source.selected = True
        self.answer("创建可编辑副本")
        session = self.runtime._select_session_target(image)
        self.assertEqual(session["write_mode"], "derived_copy")
        self.assertTrue(session["auto_created"])
        self.assertIs(session["source"], source)
        self.assertIsNot(session["target"], source)
        self.assertEqual(session["target"].name, "Target - AI Draft")
        self.assertTrue(self.nnm._is_ai_draft(session["target"]))
        self.assertFalse(source.selected)
        self.assertTrue(session["target"].selected)

    def test_empty_source_is_used_in_place_without_asking(self):
        image = self.make_image()
        source = self.make_target(image)  # empty
        source.selected = True
        session = self.runtime._select_session_target(image)
        self.assertEqual(session["write_mode"], "in_place")
        self.assertIs(session["target"], source)
        self.assertNoDialogs()

    def test_ai_draft_source_is_used_in_place_without_asking(self):
        image = self.make_image()
        pattern = _u8_pattern_buffer(SHAPE, lambda x, y, z: True)
        source = self.make_target(image, array=pattern)
        self.nnm._mark_ai_draft(source)
        source.selected = True
        session = self.runtime._select_session_target(image)
        self.assertEqual(session["write_mode"], "in_place")
        self.assertIs(session["target"], source)
        self.assertNoDialogs()

    def test_mask_from_another_image_is_rejected(self):
        image = self.make_image()
        other = self.make_image()
        source = self.make_target(other, selected=True)
        # Real Mimics may list a selected mask whose image binding points at
        # another series; the binding check must refuse it.  Simulate that by
        # letting _masks_for_image hand out the foreign mask.
        original = self.nnm._masks_for_image
        self.nnm._masks_for_image = lambda active: [source]
        try:
            with self.assertRaisesRegex(RuntimeError, "另一个影像序列"):
                self.runtime._select_session_target(image)
        finally:
            self.nnm._masks_for_image = original

    def test_nninteractive_session_conflict_is_rejected(self):
        image = self.make_image()
        source = self.make_target(image)
        self.nnm._metadata_set(
            source, self.nnm.ASYNC_JOB_METADATA, "some/nninteractive/job")
        source.selected = True
        with self.assertRaisesRegex(RuntimeError, "nnInteractive"):
            self.runtime._select_session_target(image)

    def test_existing_slip_session_resumes_in_place(self):
        image = self.make_image()
        source = self.make_target(image)
        self.nnm._metadata_set(
            source, self.runtime.ASYNC_JOB_METADATA, "some/slip/job")
        source.selected = True
        session = self.runtime._select_session_target(image)
        self.assertEqual(session["write_mode"], "in_place")
        self.assertIs(session["target"], source)
        self.assertFalse(session["auto_created"])
        self.assertNoDialogs()


if __name__ == "__main__":
    unittest.main(verbosity=2)
