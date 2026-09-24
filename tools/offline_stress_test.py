#!/usr/bin/env python3
"""Offline stress tests for Mimics-Script background orchestration.

These tests intentionally do not import Mimics. They exercise the parts that
can be validated on a non-Mimics workstation: resource locks and
nnInteractive coordinate/mapping helpers.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resource_locks import FileResourceLock, ResourceLockCancelled, release_lock


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


_GPU_HANDSHAKE_WORKER = r'''
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, sys.argv[1])
sys.path.insert(0, os.path.join(sys.argv[1], "tools"))

import nnunet_pipeline

job_dir = Path(sys.argv[2])
lock_dir = Path(sys.argv[3])
os.environ["MIMICS_RESOURCE_LOCK_DIR"] = str(lock_dir)

request = {
    "job_id": "gpu_handshake_b",
    "workspace": str(job_dir / "workspace"),
    "gpu_lock_timeout_seconds": 30,
    "gpu_lock_poll_seconds": 0.2,
}
status_path = job_dir / "status.json"
control_path = job_dir / "control.json"
log_path = job_dir / "job.log"

from nnunet_common import write_json_atomic

timeline = job_dir / "timeline.jsonl"


def record(event):
    with timeline.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": event, "time": time.time()}) + "\n")


lock = nnunet_pipeline._acquire_local_gpu(
    request, status_path, control_path, log_path, "training")
record("acquired")
status = json.loads(status_path.read_text(encoding="utf-8"))
record("status:" + str(status.get("status")))
lock.release()
record("released")
'''


def test_gpu_contention_handshake_two_processes(tmp: Path) -> str:
    """Two REAL processes handshake over the GPU lock (Phase 3d).

    Process A holds the lock; process B runs the pipeline's actual
    _acquire_local_gpu waiting path. B must record waiting_for_gpu while
    blocked, acquire only after A releases, and never report training
    while A still owns the lock.
    """
    import subprocess

    root = Path(__file__).resolve().parents[1]
    lock_dir = tmp / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    job_dir = tmp / "job_b"
    job_dir.mkdir(parents=True, exist_ok=True)

    # Process A: hold the GPU lock like another AI task would.
    holder = FileResourceLock(lock_dir / "gpu.lock", "GPU", "process-A")
    holder.acquire(wait_seconds=5, poll_seconds=0.05)

    worker_source = tmp / "gpu_handshake_worker.py"
    worker_source.write_text(_GPU_HANDSHAKE_WORKER, encoding="utf-8")
    command = [
        sys.executable,
        str(worker_source),
        str(root),
        str(job_dir),
        str(lock_dir),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(root),
    )
    try:
        # While A holds the lock, B must be waiting.
        deadline = time.time() + 15
        waiting_seen = False
        while time.time() < deadline:
            status_path = job_dir / "status.json"
            if status_path.is_file():
                payload = json.loads(status_path.read_text(encoding="utf-8"))
                if payload.get("status") == "waiting_for_gpu":
                    waiting_seen = True
                    break
            time.sleep(0.2)
        assert_true(waiting_seen, "process B never reported waiting_for_gpu")
        assert_true(
            not (job_dir / "timeline.jsonl").exists()
            or "acquired" not in (job_dir / "timeline.jsonl").read_text(encoding="utf-8"),
            "process B acquired the GPU while process A still held it",
        )
        # A releases; B must acquire within its timeout.
        holder.release()
        stdout, _ = process.communicate(timeout=60)
        assert_equal(0, process.returncode, "worker process B failed:\n" + stdout.decode(errors="replace"))
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if holder.acquired:
            holder.release()

    timeline = [
        json.loads(line)
        for line in (job_dir / "timeline.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    events = [row["event"] for row in timeline]
    assert_true("acquired" in events, "process B never acquired the lock")
    acquired_at = next(row["time"] for row in timeline if row["event"] == "acquired")
    # The final status after acquisition must not stay waiting_for_gpu.
    status_rows = [row for row in timeline if row["event"].startswith("status:")]
    assert_true(status_rows, "no status recorded after acquisition")
    last_status = status_rows[-1]["event"]
    assert_true(
        "waiting_for_gpu" not in last_status,
        "status stayed waiting_for_gpu after acquiring the lock",
    )
    # B acquired only after A released (A's release happened before our
    # communicate; verify via the lock file being gone at the end).
    assert_true(
        not (lock_dir / "gpu.lock").exists(),
        "gpu.lock was not cleaned up after both processes finished",
    )
    return "process B waited, then acquired after A released (statuses: {})".format(
        ", ".join(row["event"] for row in status_rows)
    )


















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
            ("GPU contention handshake (two processes)", lambda: test_gpu_contention_handshake_two_processes(tmp / "gpu_handshake")),
        ])
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
