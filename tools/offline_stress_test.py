#!/usr/bin/env python3
"""Offline stress tests for Mimics-Script background orchestration.

These tests intentionally do not import Mimics. They exercise the parts that
can be validated on a non-Mimics workstation: resource locks, background job
status files, DINOv3 few-shot orchestration, log rotation, checkpoint cleanup,
and coordinate/mapping helpers.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import traceback
import types
import io
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resource_locks import FileResourceLock, ResourceLockCancelled, release_lock
import tools.fewshot_pipeline as fewshot


class StressFailure(RuntimeError):
    pass


class StressRunner:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.results: list[tuple[str, str, str]] = []

    def run(self, name: str, func):
        started = time.time()
        try:
            detail = func()
        except Exception:
            self.results.append((name, "FAIL", traceback.format_exc()))
            return
        elapsed = time.time() - started
        self.results.append((name, "PASS", "{} ({:.2f}s)".format(detail or "ok", elapsed)))

    def report(self) -> int:
        print("\nOffline stress test summary")
        print("=" * 72)
        failed = 0
        for name, status, detail in self.results:
            print("[{}] {}".format(status, name))
            if detail:
                if status == "FAIL":
                    failed += 1
                    print(detail.rstrip())
                else:
                    print("  " + detail)
        print("=" * 72)
        print("{} passed, {} failed".format(len(self.results) - failed, failed))
        return 1 if failed else 0


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def write_jsonl(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def assert_true(condition, message: str):
    if not condition:
        raise StressFailure(message)


def assert_equal(left, right, message: str):
    if left != right:
        raise StressFailure("{}: {!r} != {!r}".format(message, left, right))


def test_resource_lock_contention(tmp: Path) -> str:
    lock_path = tmp / "locks" / "gpu.lock"
    timeline_path = tmp / "lock_timeline.jsonl"
    overlaps: list[str] = []
    guard = threading.Lock()
    active = {"count": 0}

    def worker(index: int):
        lock = FileResourceLock(lock_path, "gpu", "stress-worker-{}".format(index))
        lock.acquire(wait_seconds=10, poll_seconds=0.05)
        try:
            with guard:
                active["count"] += 1
                if active["count"] != 1:
                    overlaps.append("overlap at worker {}".format(index))
                start = time.time()
            write_jsonl(timeline_path, {"worker": index, "event": "start", "time": start})
            time.sleep(0.04)
            write_jsonl(timeline_path, {"worker": index, "event": "end", "time": time.time()})
            with guard:
                active["count"] -= 1
        finally:
            lock.release()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)
    assert_true(not any(thread.is_alive() for thread in threads), "lock workers did not finish")
    assert_true(not overlaps, "lock allowed concurrent owners: {}".format(overlaps))
    assert_true(not lock_path.exists(), "lock file was not released")

    stale = {
        "schema_version": "mimics_script_resource_lock.v1",
        "resource": "gpu",
        "owner": "dead-owner",
        "pid": 99999999,
        "token": "dead-token",
        "created_at_epoch": time.time(),
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(stale), encoding="utf-8")
    lock = FileResourceLock(lock_path, "gpu", "stale-reclaimer")
    lock.acquire(wait_seconds=1, poll_seconds=0.05)
    token = lock.token
    assert_true(not release_lock(lock_path, "wrong-token"), "wrong token released a live lock")
    lock.release()
    assert_true(not lock_path.exists(), "stale lock replacement was not released")
    return "16 concurrent lock attempts serialized; stale lock reclaimed; token guard held"


def test_lock_wait_cancel(tmp: Path) -> str:
    lock_path = tmp / "locks" / "gpu.lock"
    holder = FileResourceLock(lock_path, "gpu", "stress-holder")
    holder.acquire(wait_seconds=1, poll_seconds=0.05)
    cancel = {"value": False}
    result = {"cancelled": False, "elapsed": None}

    def waiter():
        started = time.time()
        try:
            FileResourceLock(lock_path, "gpu", "stress-waiter").acquire(
                wait_seconds=10,
                poll_seconds=1.0,
                should_cancel=lambda: cancel["value"],
            )
        except ResourceLockCancelled:
            result["cancelled"] = True
            result["elapsed"] = time.time() - started

    thread = threading.Thread(target=waiter)
    thread.start()
    time.sleep(0.15)
    cancel["value"] = True
    thread.join(timeout=3)
    holder.release()
    assert_true(not thread.is_alive(), "cancelled lock waiter did not return")
    assert_true(result["cancelled"], "lock waiter was not cancelled")
    assert_true(float(result["elapsed"] or 99) < 2.0, "lock cancellation was too slow")
    return "waiting lock cancelled in {:.2f}s".format(float(result["elapsed"] or 0))


def test_fewshot_gpu_queue_and_cancel(tmp: Path) -> str:
    original_gpu_path = fewshot.GPU_LOCK_PATH
    original_gpu_enabled = fewshot.gpu_lock_enabled
    try:
        fewshot.GPU_LOCK_PATH = tmp / "locks" / "gpu.lock"
        fewshot.gpu_lock_enabled = lambda: True
        workspace = tmp / "workspace"
        status = workspace / "jobs" / "wait.json"
        cancel_path = workspace / "runs" / "cancel.request"
        status.parent.mkdir(parents=True, exist_ok=True)
        cancel_path.parent.mkdir(parents=True, exist_ok=True)
        fewshot.write_json_atomic(status, {"status": "launching"})

        holder = FileResourceLock(fewshot.GPU_LOCK_PATH, "gpu", "active nnInteractive inference")
        holder.acquire(wait_seconds=1, poll_seconds=0.05)

        acquired = {"lock": None}

        def waiter():
            acquired["lock"] = fewshot.acquire_gpu_lock_for_job(
                workspace,
                status,
                cancel_path,
                "queued DINOv3 training",
                10,
            )

        thread = threading.Thread(target=waiter)
        thread.start()
        deadline = time.time() + 4
        while time.time() < deadline:
            payload = read_json(status, {}) or {}
            if payload.get("status") == "waiting_for_gpu" and payload.get("resource_wait"):
                break
            time.sleep(0.05)
        else:
            raise StressFailure("waiting_for_gpu status was not written")
        holder.release()
        thread.join(timeout=6)
        assert_true(not thread.is_alive(), "GPU waiter did not acquire after release")
        assert_true(acquired["lock"] is not None, "GPU waiter returned no lock")
        acquired["lock"].release()
        payload = read_json(status, {}) or {}
        assert_true(payload.get("resource_wait") is None, "resource_wait was not cleared")

        holder = FileResourceLock(fewshot.GPU_LOCK_PATH, "gpu", "active DINOv3 training")
        holder.acquire(wait_seconds=1, poll_seconds=0.05)
        status2 = workspace / "jobs" / "cancel_wait.json"
        cancel2 = workspace / "runs" / "cancel2.request"
        fewshot.write_json_atomic(status2, {"status": "launching"})
        cancelled = {"value": False}

        def cancelled_waiter():
            try:
                fewshot.acquire_gpu_lock_for_job(workspace, status2, cancel2, "queued inference", 10)
            except ResourceLockCancelled:
                cancelled["value"] = True

        thread = threading.Thread(target=cancelled_waiter)
        thread.start()
        time.sleep(0.2)
        cancel2.write_text("cancel", encoding="utf-8")
        thread.join(timeout=4)
        holder.release()
        assert_true(not thread.is_alive(), "pipeline GPU waiter did not respond to cancel")
        assert_true(cancelled["value"], "pipeline GPU waiter did not raise ResourceLockCancelled")
        return "few-shot GPU wait status, acquire-after-release, and cancel path passed"
    finally:
        fewshot.GPU_LOCK_PATH = original_gpu_path
        fewshot.gpu_lock_enabled = original_gpu_enabled


def make_fake_dinov3_root(tmp: Path) -> Path:
    root = tmp / "fake_dinov3"
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "config").mkdir(parents=True, exist_ok=True)
    for name in ("mimics_lora_segformer3d.yaml", "synthstrip_lora_segformer3d.yaml"):
        (root / "config" / name).write_text("training:\n  epochs: 1\n", encoding="utf-8")
    train_script = r'''
import json
import os
import re
import sys
import time
from pathlib import Path


def scalar(config_text, key):
    match = re.search(r"^" + re.escape(key) + r":\s*\"?([^\"\n]+)\"?", config_text, re.M)
    return match.group(1).strip() if match else ""


args = sys.argv[1:]
config_path = Path(args[args.index("--config") + 1])
config_text = config_path.read_text(encoding="utf-8")
exp_name = scalar(config_text, "exp_name")
status_match = re.search(r"^\s*status_path:\s*\"?([^\"\n]+)\"?", config_text, re.M)
status_path = Path(status_match.group(1).strip()) if status_match else None
timeline = os.environ.get("FAKE_TRAIN_TIMELINE")
if timeline:
    with open(timeline, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "start", "pid": os.getpid(), "time": time.time(), "config": str(config_path)}) + "\n")
if status_path:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps({
        "status": "training",
        "epoch": 1,
        "epochs": 1,
        "phase": "train",
        "best_dsc": 0.0,
        "metrics": {"loss": 0.5},
    }), encoding="utf-8")
time.sleep(float(os.environ.get("FAKE_TRAIN_SLEEP", "0.4")))
ckpt = Path.cwd() / "experiments" / exp_name / "checkpoints" / "epoch_0001.pth"
ckpt.parent.mkdir(parents=True, exist_ok=True)
ckpt.write_text("fake checkpoint\n", encoding="utf-8")
if status_path:
    status_path.write_text(json.dumps({
        "status": "completed",
        "epoch": 1,
        "epochs": 1,
        "phase": "epoch_complete",
        "best_dsc": 0.9,
        "metrics": {"loss": 0.1, "mean_dsc": 0.9},
    }), encoding="utf-8")
if timeline:
    with open(timeline, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "end", "pid": os.getpid(), "time": time.time(), "config": str(config_path)}) + "\n")
'''
    (root / "scripts" / "train.py").write_text(train_script.lstrip(), encoding="utf-8")
    return root


def make_ts_root(tmp: Path) -> Path:
    import nibabel as nib
    import numpy as np

    ts_root = tmp / "dataset"
    for idx in range(1, 4):
        case = ts_root / "s{:04d}".format(idx)
        (case / "segmentations").mkdir(parents=True, exist_ok=True)
        image = np.full((4, 4, 2), idx, dtype=np.int16)
        label = np.zeros((4, 4, 2), dtype=np.uint8)
        label[1:3, 1:3, :] = 1
        nib.save(nib.Nifti1Image(image, np.eye(4)), str(case / "ct.nii.gz"))
        nib.save(nib.Nifti1Image(label, np.eye(4)), str(case / "segmentations" / "liver.nii.gz"))
    return ts_root


def train_args(ts_root: Path, workspace: Path, dinov3_root: Path, run_id: str):
    return SimpleNamespace(
        ts_root=str(ts_root),
        organ="liver",
        workspace=str(workspace),
        dinov3_root=str(dinov3_root),
        python=sys.executable,
        base_config=None,
        cases=None,
        sample_mode="all",
        min_samples=1,
        max_samples=0,
        epochs=1,
        batch_size=1,
        grad_accumulation=1,
        lr=0.001,
        weight_decay=0.01,
        img_size="64,64",
        modality="ct",
        val_fraction=0.34,
        val_cases=None,
        min_val_samples=1,
        finetune_method="lora",
        decoder="segformer3d",
        model_scale="vitb16",
        model_path=None,
        lora_rank=8,
        lora_alpha=16,
        adapter_bottleneck=64,
        mixed_precision=False,
        sub_volume=False,
        sub_volume_size="16,64,64",
        export_labels=False,
        mimics_exe=None,
        export_timeout_seconds=30,
        background_mimics_lock_timeout_seconds=1,
        gpu_lock_timeout_seconds=20,
        keep_last_checkpoints=1,
        keep_materialized_dataset=False,
        run_id=run_id,
    )


def test_fewshot_fake_training_queue(tmp: Path) -> str:
    original_gpu_path = fewshot.GPU_LOCK_PATH
    original_gpu_enabled = fewshot.gpu_lock_enabled
    original_registry = fewshot.global_registry_path
    try:
        fewshot.GPU_LOCK_PATH = tmp / "locks" / "gpu.lock"
        fewshot.gpu_lock_enabled = lambda: True
        fewshot.global_registry_path = lambda: tmp / "global_models.json"
        ts_root = make_ts_root(tmp)
        workspace = tmp / "fewshot_workspace"
        dinov3_root = make_fake_dinov3_root(tmp)
        timeline = tmp / "fake_train_timeline.jsonl"
        os.environ["FAKE_TRAIN_TIMELINE"] = str(timeline)
        os.environ["FAKE_TRAIN_SLEEP"] = "0.6"

        returns: dict[str, int] = {}

        def run_job(run_id: str):
            returns[run_id] = fewshot.cmd_train(train_args(ts_root, workspace, dinov3_root, run_id))

        t1 = threading.Thread(target=run_job, args=("stress_a",))
        t2 = threading.Thread(target=run_job, args=("stress_b",))
        t1.start()
        time.sleep(0.05)
        t2.start()
        t1.join(timeout=20)
        t2.join(timeout=20)
        assert_true(not t1.is_alive() and not t2.is_alive(), "fake DINO training jobs did not finish")
        assert_equal(returns.get("stress_a"), 0, "first fake training return code")
        assert_equal(returns.get("stress_b"), 0, "second fake training return code")

        rows = [json.loads(line) for line in timeline.read_text(encoding="utf-8").splitlines() if line.strip()]
        starts = [row for row in rows if row["event"] == "start"]
        ends = [row for row in rows if row["event"] == "end"]
        assert_equal(len(starts), 2, "fake train start count")
        assert_equal(len(ends), 2, "fake train end count")
        intervals = []
        for start in starts:
            matching_end = next(row for row in ends if row["pid"] == start["pid"])
            intervals.append((float(start["time"]), float(matching_end["time"])))
        intervals.sort()
        assert_true(intervals[0][1] <= intervals[1][0] + 0.02, "GPU-locked fake training processes overlapped")

        for run_id in ("stress_a", "stress_b"):
            status = read_json(workspace / "jobs" / (run_id + ".json"), {}) or {}
            assert_equal(status.get("status"), "completed", "{} status".format(run_id))
            manifest = status.get("model") or {}
            assert_true(Path(manifest.get("checkpoint", "")).is_file(), "{} checkpoint missing".format(run_id))
            assert_true(not Path(manifest.get("dataset_dir", "")).exists(), "{} materialized dataset was not cleaned".format(run_id))
        assert_true((tmp / "global_models.json").is_file(), "global model index was not written")
        return "two fake training jobs serialized on GPU lock; models registered; materialized datasets cleaned"
    finally:
        fewshot.GPU_LOCK_PATH = original_gpu_path
        fewshot.gpu_lock_enabled = original_gpu_enabled
        fewshot.global_registry_path = original_registry
        os.environ.pop("FAKE_TRAIN_TIMELINE", None)
        os.environ.pop("FAKE_TRAIN_SLEEP", None)


def test_log_rotation(tmp: Path) -> str:
    original_bytes = fewshot.LOG_ROTATE_BYTES
    original_backups = fewshot.LOG_ROTATE_BACKUPS
    try:
        fewshot.LOG_ROTATE_BYTES = 512
        fewshot.LOG_ROTATE_BACKUPS = 2
        workspace = tmp / "workspace"
        with contextlib.redirect_stdout(io.StringIO()):
            for index in range(80):
                fewshot.append_log(workspace, "stress log line {:03d} {}".format(index, "x" * 80))
        log_path = workspace / "fewshot_pipeline.log"
        backups = sorted(workspace.glob("fewshot_pipeline.log.*"))
        assert_true(log_path.is_file(), "active log file missing")
        assert_true(backups, "rotated log backups missing")
        assert_true(log_path.stat().st_size < 2048, "active log grew unexpectedly large")
        assert_true(len(backups) <= 2, "too many log backups were retained")
        return "log rotation retained {} backup file(s)".format(len(backups))
    finally:
        fewshot.LOG_ROTATE_BYTES = original_bytes
        fewshot.LOG_ROTATE_BACKUPS = original_backups


def load_checkpoint_module_with_stub():
    module_path = ROOT / "external" / "dinov3-medical-seg" / "src" / "utils" / "checkpoint.py"
    previous = sys.modules.get("torch")
    torch_stub = types.SimpleNamespace()
    torch_stub.nn = types.SimpleNamespace(Module=object)
    torch_stub.optim = types.SimpleNamespace(Optimizer=object)
    torch_stub.device = object
    torch_stub.save = lambda *args, **kwargs: None
    torch_stub.load = lambda *args, **kwargs: {}
    sys.modules["torch"] = torch_stub
    try:
        spec = importlib.util.spec_from_file_location("checkpoint_stress_stub", str(module_path))
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = previous


def test_checkpoint_cleanup(tmp: Path) -> str:
    checkpoint = load_checkpoint_module_with_stub()
    ckpt_dir = tmp / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, 6):
        path = ckpt_dir / "epoch_{:04d}.pth".format(epoch)
        path.write_text(str(epoch), encoding="utf-8")
        os.utime(path, (time.time() + epoch, time.time() + epoch))
    (ckpt_dir / "best_model.pth").write_text("best", encoding="utf-8")
    checkpoint._cleanup_old_epoch_checkpoints(str(ckpt_dir), 2)
    names = sorted(path.name for path in ckpt_dir.iterdir())
    assert_equal(names, ["best_model.pth", "epoch_0004.pth", "epoch_0005.pth"], "checkpoint retention with best model")

    shutil.rmtree(str(ckpt_dir))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, 4):
        path = ckpt_dir / "epoch_{:04d}.pth".format(epoch)
        path.write_text(str(epoch), encoding="utf-8")
        os.utime(path, (time.time() + epoch, time.time() + epoch))
    checkpoint._cleanup_old_epoch_checkpoints(str(ckpt_dir), 0)
    names = sorted(path.name for path in ckpt_dir.iterdir())
    assert_equal(names, ["epoch_0003.pth"], "checkpoint retention without best model")
    return "checkpoint cleanup keeps best plus latest N, and at least latest without best"


def test_nninteractive_mapping_and_resampling(tmp: Path) -> str:
    try:
        import numpy as np
        import nninteractive_bridge as bridge
    except Exception as exc:
        return "skipped: nnInteractive bridge dependencies unavailable ({})".format(exc)

    import itertools

    shape = [3, 4, 5]
    array = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
    for axes in itertools.permutations([0, 1, 2]):
        for flips in itertools.product([False, True], repeat=3):
            mapping = {
                "platform_to_mimics_axes": list(axes),
                "platform_to_mimics_flips": list(flips),
            }
            platform = bridge.mimics_to_platform(array, mapping)
            roundtrip = bridge.platform_to_mimics(platform, mapping)
            assert_true(np.array_equal(roundtrip, array), "mapping roundtrip failed: {} {}".format(axes, flips))
            assert_equal(
                list(platform.shape),
                bridge.mimics_shape_to_platform_shape(shape, mapping),
                "mapped shape mismatch",
            )

    data = np.arange(3 * 4 * 2, dtype=np.float32).reshape((3, 4, 2))
    source = np.eye(4, dtype=np.float64)
    mimics = np.array(
        [
            [-1.0, 0.0, 0.0, 2.0],
            [0.0, -1.0, 0.0, 3.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    try:
        aligned = bridge._resample_image_to_mimics_grid(data, source, mimics, [3, 4, 2])
    except RuntimeError as exc:
        return "mapping passed; resampling skipped ({})".format(exc)
    expected = data[::-1, ::-1, :]
    assert_true(np.array_equal(aligned, expected), "affine mirror resampling did not match expected Mimics grid")
    transformed = bridge._apply_source_intensity_transform(
        data,
        {
            "image_source_to_mimics_gv_slope": 1.0,
            "image_source_to_mimics_gv_intercept": 1024.0,
        },
    )
    assert_true(np.array_equal(transformed, data + 1024.0), "source HU-to-GV transform was not applied")

    import mimics_bridge
    mask = np.zeros((3, 4, 2), dtype=np.uint8)
    mask[2, 3, 0] = 1
    target_grid = mimics_bridge.resample_mask_to_image_grid(mask, source, (3, 4, 2), mimics)
    assert_equal(int(target_grid[0, 0, 0]), 1, "mask affine mirror did not land on target Mimics grid")
    assert_equal(int(target_grid.sum()), 1, "mask resampling changed foreground voxel count")
    return "all axis permutations/flips roundtrip; image and mask affine mirror resampling match expected Mimics grid"


def test_nninteractive_incremental_replay(tmp: Path) -> str:
    try:
        import numpy as np
        import nninteractive_bridge as bridge
    except Exception as exc:
        return "skipped: nnInteractive bridge dependencies unavailable ({})".format(exc)

    class FakeSession:
        def __init__(self, target):
            self.target = target
            self.reset_count = 0
            self.point_calls = []

        def reset_interactions(self):
            self.reset_count += 1

        def add_point_interaction(self, point, include_interaction=True, run_prediction=True):
            self.point_calls.append((tuple(point), bool(include_interaction), bool(run_prediction)))
            if include_interaction:
                self.target[tuple(point)] = 1
            else:
                self.target[tuple(point)] = 0

        def close(self):
            pass

    context = bridge._BridgeSessionContext.__new__(bridge._BridgeSessionContext)
    context.model_dir = str(tmp / "model")
    context.requested_device = "cpu"
    context.device = "cpu"
    context.device_warning = None
    context.server_url = "fake://server"
    context.first_call = False
    context.log_path = tmp / "bridge.jsonl"
    context.buffer_mapping = {
        "platform_to_mimics_axes": [0, 1, 2],
        "platform_to_mimics_flips": [False, False, False],
    }
    context.mimics_shape = [3, 3, 3]
    context.platform_shape = [3, 3, 3]
    context.initial_platform = None
    context.image_load_seconds = 0.0
    context.server_ready_seconds = 0.0
    context.set_image_seconds = 0.0
    context.set_target_seconds = 0.0
    context.incremental_interaction_replay = True
    context._applied_initial_key = None
    context._applied_interaction_fingerprints = []
    context.target = np.zeros((3, 3, 3), dtype=np.uint8)
    context.session = FakeSession(context.target)

    first = {
        "interaction_type": "point_set",
        "coordinates": "mimics",
        "points": [{"point": [0, 0, 0], "include_interaction": True}],
    }
    second = {
        "interaction_type": "point_set",
        "coordinates": "mimics",
        "points": [{"point": [1, 1, 1], "include_interaction": True}],
    }
    result1 = context.predict([first], str(tmp / "prediction1.u8"))
    result2 = context.predict([first, second], str(tmp / "prediction2.u8"))
    assert_true(not result1.get("incremental_replay"), "first prompt should use full replay")
    assert_true(result2.get("incremental_replay"), "second prompt did not use incremental replay")
    assert_equal(result2.get("skipped_replay_interactions"), 1, "skipped replay interaction count")
    assert_equal(result2.get("replayed_interactions"), 1, "new replay interaction count")
    assert_equal(context.session.reset_count, 1, "incremental replay should not reset on second prompt")
    assert_equal(len(context.session.point_calls), 2, "old point was sent again during incremental replay")
    assert_equal(int(np.count_nonzero(context.target)), 2, "target should contain both incremental points")
    return "append-only prompt update skipped replaying the previous prompt"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the temporary stress-test directory for inspection.",
    )
    parser.add_argument(
        "--only",
        choices=[
            "locks",
            "fewshot",
            "logs",
            "checkpoints",
            "mapping",
            "all",
        ],
        default="all",
        help="Run a subset of offline stress tests.",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    temp = tempfile.TemporaryDirectory(prefix="mimics_script_offline_stress_")
    tmp = Path(temp.name)
    runner = StressRunner(tmp)
    tests = []
    if args.only in ("locks", "all"):
        tests.extend([
            ("resource lock contention", lambda: test_resource_lock_contention(tmp / "resource_lock_contention")),
            ("lock wait cancellation", lambda: test_lock_wait_cancel(tmp / "lock_wait_cancel")),
        ])
    if args.only in ("fewshot", "all"):
        tests.extend([
            ("few-shot GPU queue and cancel", lambda: test_fewshot_gpu_queue_and_cancel(tmp / "fewshot_gpu")),
            ("few-shot fake training queue", lambda: test_fewshot_fake_training_queue(tmp / "fewshot_train")),
        ])
    if args.only in ("logs", "all"):
        tests.append(("few-shot log rotation", lambda: test_log_rotation(tmp / "logs")))
    if args.only in ("checkpoints", "all"):
        tests.append(("DINOv3 checkpoint cleanup", lambda: test_checkpoint_cleanup(tmp / "checkpoints")))
    if args.only in ("mapping", "all"):
        tests.append(("nnInteractive mapping and source-grid resampling", lambda: test_nninteractive_mapping_and_resampling(tmp / "mapping")))
        tests.append(("nnInteractive incremental prompt replay", lambda: test_nninteractive_incremental_replay(tmp / "incremental_replay")))

    for name, func in tests:
        runner.run(name, func)
    code = runner.report()
    if args.keep_temp:
        print("Temporary directory kept: {}".format(tmp))
    else:
        temp.cleanup()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
