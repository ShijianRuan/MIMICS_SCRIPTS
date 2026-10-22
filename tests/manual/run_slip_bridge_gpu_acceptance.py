#!/usr/bin/env python3
"""SLIP bridge GPU manual acceptance test (L5, docs/TEST_STRATEGY.md).

Not part of the automated regression matrix — it needs a real GPU, the
slip_models checkpoints and the read-only Totalsegmentator dataset on Z:.
Runs against the DISTRIBUTION layout (integrations/slip + slip_models),
exercising both modes of slip_bridge.py:

  1. single-shot stdin/stdout predict
  2. --async-worker file-queue mode: predict / undo / reset / error paths /
     close, plus GPU-lock mutual exclusion while the worker is running.

16 checks; prints ALL_OK and exits 0 on success. All outputs go to
.mimics_runtime/slip_gpu_acceptance/ — the dataset stays read-only.

Usage (python_env):
  python_env/python.exe tests/manual/run_slip_bridge_gpu_acceptance.py
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import SimpleITK as sitk

PROJECT = Path(__file__).resolve().parents[2]   # E:/MIMICS_SCRIPTS
PY = PROJECT / "python_env" / "python.exe"
BRIDGE = PROJECT / "slip_bridge.py"
CASE = r"Z:/ImageAnalysisData/1-CT/Segmentation/data/Totalsegmentator_dataset_v201/s0000"
OUT = PROJECT / ".mimics_runtime" / "slip_gpu_acceptance"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT))
from resource_locks import (FileResourceLock, ResourceLockTimeout,
                            default_resource_lock_dir)

failures = []


def check(name, ok, detail=""):
    print(("[PASS] " if ok else "[FAIL] ") + name + ("  " + detail if detail else ""))
    if not ok:
        failures.append(name)


# ---- export the case as a raw int16 buffer (HU values, no rescale needed) ----
img = sitk.ReadImage(CASE + "/ct.nii.gz")
vol = sitk.GetArrayFromImage(img)  # (Z, Y, X), int16 HU
assert vol.dtype == np.int16, vol.dtype
buffer_path = OUT / "ct_s0000_int16.bin"
buffer_path.write_bytes(vol.tobytes())
shape = list(vol.shape)
print("exported buffer:", buffer_path.name, shape, vol.dtype,
      "{0:.1f} MB".format(buffer_path.stat().st_size / 1e6))

gt = sitk.GetArrayFromImage(
    sitk.ReadImage(CASE + "/segmentations/femur_left.nii.gz")).astype(bool)
CLICK = [97, 67, 57]  # femur_left centroid (from the earlier smoke run)


def dice(pred):
    return 2.0 * (pred & gt).sum() / max(1, pred.sum() + gt.sum())


IMAGE = {
    "buffer_path": str(buffer_path),
    "shape": shape,
    "dtype": "int16",
    "to_hu_slope": 1.0,
    "to_hu_intercept": 0.0,
}

ENV = dict(os.environ, PYTHONUTF8="1")


def read_mask(path):
    data = np.fromfile(path, dtype=np.uint8)
    assert data.size == int(np.prod(shape)), (data.size, shape)
    return data.reshape(shape).astype(bool)


# ============================ 1. single-shot mode ============================
print("\n--- single-shot mode ---")
t0 = time.time()
request = {
    "device": "cuda",
    "image": IMAGE,
    "click": CLICK,
    "label": 1,
    "output_path": str(OUT / "single_shot_mask.u8"),
}
proc = subprocess.run(
    [str(PY), str(BRIDGE)],
    input=json.dumps(request).encode("utf-8"),
    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    cwd=str(PROJECT), env=ENV, timeout=600,
)
print("single-shot exit=%d elapsed=%.1fs" % (proc.returncode, time.time() - t0))
try:
    result = json.loads(proc.stdout.decode("utf-8", "replace"))
except ValueError:
    print("STDOUT:", proc.stdout.decode("utf-8", "replace")[:2000])
    print("STDERR:", proc.stderr.decode("utf-8", "replace")[:2000])
    raise
check("single-shot exit 0", proc.returncode == 0, "rc=%d" % proc.returncode)
check("single-shot status refined", result.get("status") == "refined",
      str({k: result.get(k) for k in ("status", "voxels", "interaction_seconds",
                                       "action_history_len", "error")}))
mask1 = read_mask(OUT / "single_shot_mask.u8")
d1 = dice(mask1)
print("single-shot voxels=%d dice=%.4f" % (mask1.sum(), d1))
check("single-shot dice in expected band (smoke run: 0.214)",
      0.10 <= d1 <= 0.45, "dice=%.4f" % d1)

# ============================ 2. async-worker mode ============================
print("\n--- async-worker mode ---")
job_dir = OUT / "async_job"
for sub in ("commands", "results"):
    (job_dir / sub).mkdir(parents=True, exist_ok=True)

initialize = {
    "device": "cuda",
    "image": IMAGE,
    "parent_pid": os.getpid(),
    "idle_timeout_seconds": 300,
    "poll_seconds": 0.2,
}
(job_dir / "initialize.json").write_text(
    json.dumps(initialize), encoding="utf-8")

worker_log = (job_dir / "worker_stdout.log").open("wb")
t0 = time.time()
worker = subprocess.Popen(
    [str(PY), str(BRIDGE), "--async-worker", str(job_dir)],
    stdout=worker_log, stderr=subprocess.STDOUT,
    cwd=str(PROJECT), env=ENV,
)


def wait_for_status(target, timeout_s, what):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            status = json.loads(
                (job_dir / "worker_status.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            status = {}
        if status.get("status") == target:
            return status
        if status.get("status") == "failed":
            raise RuntimeError("worker failed: " + json.dumps(status, indent=2))
        if worker.poll() is not None:
            raise RuntimeError(
                "worker exited rc=%s while waiting for %s: %s"
                % (worker.returncode, what, json.dumps(status)))
        time.sleep(0.5)
    raise RuntimeError("timeout waiting for status %s (%s)" % (target, what))


def send_command(sequence, payload):
    payload = dict(payload, sequence=sequence,
                   command_id="cmd%03d" % sequence)
    (job_dir / "commands" / ("command_%06d.json" % sequence)).write_text(
        json.dumps(payload), encoding="utf-8")


def wait_for_result(sequence, timeout_s):
    path = job_dir / "results" / ("result_%06d.json" % sequence)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.is_file():
            time.sleep(0.3)  # let the atomic replace settle
            return json.loads(path.read_text(encoding="utf-8"))
        if worker.poll() is not None:
            raise RuntimeError("worker exited rc=%s waiting for result %d"
                               % (worker.returncode, sequence))
        time.sleep(0.3)
    raise RuntimeError("timeout waiting for result %d" % sequence)


try:
    status = wait_for_status("ready", 300, "model load + process_image")
    print("worker ready in %.1fs: patches=%s model_load=%ss process_image=%ss"
          % (time.time() - t0, status.get("patches"),
             status.get("model_load_seconds"),
             status.get("process_image_seconds")))
    check("worker ready with patches cached",
          int(status.get("patches") or 0) > 0, "patches=%s" % status.get("patches"))

    # GPU lock mutual exclusion: while the SLIP worker is alive the shared
    # gpu.lock must refuse another holder.
    lock = FileResourceLock(
        default_resource_lock_dir(PROJECT) / "gpu.lock", "gpu", "bridge test")
    try:
        lock.acquire(wait_seconds=1.0, poll_seconds=0.3)
        lock.release()
        check("gpu.lock mutual exclusion (worker holds lock)", False,
              "test acquired the lock while worker was running")
    except ResourceLockTimeout:
        check("gpu.lock mutual exclusion (worker holds lock)", True)

    # --- predict (same click as single-shot) ---
    send_command(1, {"action": "predict", "click": CLICK, "label": 1})
    r1 = wait_for_result(1, 120)
    check("cmd1 predict refined", r1.get("status") == "refined", str(r1)[:200])
    m1 = read_mask(job_dir / "results" / "prediction_000001.u8")
    d = dice(m1)
    print("cmd1 voxels=%d dice=%.4f interaction=%ss"
          % (m1.sum(), d, r1.get("interaction_seconds")))
    check("cmd1 dice consistent with single-shot",
          0.10 <= d <= 0.45, "dice=%.4f" % d)
    check("cmd1 action_history_len == 1",
          r1.get("action_history_len") == 1)

    # --- undo -> back to state after nothing (empty mask), history 0 ---
    send_command(2, {"action": "undo"})
    r2 = wait_for_result(2, 120)
    check("cmd2 undo refined", r2.get("status") == "refined", str(r2)[:200])
    check("cmd2 action_history_len == 0", r2.get("action_history_len") == 0,
          str(r2.get("action_history_len")))

    # --- reset (no-op here but must succeed and keep patches) ---
    send_command(3, {"action": "reset"})
    r3 = wait_for_result(3, 60)
    check("cmd3 reset ok", r3.get("status") == "reset"
          and r3.get("patches_cached", 0) > 0, str(r3)[:200])

    # --- failure path: undo with empty history must return an error result,
    #     not crash the worker ---
    send_command(4, {"action": "undo"})
    r4 = wait_for_result(4, 60)
    check("cmd4 undo-on-empty returns error result",
          r4.get("status") == "error" and "Nothing to undo" in str(r4.get("error")),
          str(r4.get("error"))[:120])

    # --- failure path: unknown action ---
    send_command(5, {"action": "frobnicate"})
    r5 = wait_for_result(5, 60)
    check("cmd5 unknown action returns error result",
          r5.get("status") == "error", str(r5.get("error"))[:120])

    # --- worker still usable after error results ---
    send_command(6, {"action": "predict", "click": CLICK, "label": 1})
    r6 = wait_for_result(6, 120)
    check("cmd6 predict after errors still refined",
          r6.get("status") == "refined", str(r6)[:200])

    # --- graceful close ---
    (job_dir / "close.json").write_text(
        json.dumps({"reason": "test complete"}), encoding="utf-8")
    rc = worker.wait(timeout=60)
    check("worker exits 0 on close.json", rc == 0, "rc=%s" % rc)
    final = json.loads(
        (job_dir / "worker_status.json").read_text(encoding="utf-8"))
    check("final worker_status closed", final.get("status") == "closed",
          str(final.get("status")))
finally:
    if worker.poll() is None:
        (job_dir / "close.json").write_text('{"reason": "test abort"}',
                                            encoding="utf-8")
        try:
            worker.wait(timeout=30)
        except subprocess.TimeoutExpired:
            worker.kill()
    worker_log.close()

# --- gpu.lock must be free again after the worker exited ---
lock = FileResourceLock(
    default_resource_lock_dir(PROJECT) / "gpu.lock", "gpu", "bridge test")
try:
    lock.acquire(wait_seconds=5.0, poll_seconds=0.5)
    lock.release()
    check("gpu.lock released after worker exit", True)
except ResourceLockTimeout:
    check("gpu.lock released after worker exit", False)

print("\n%s" % ("ALL_OK" if not failures else "FAILED: %s" % failures))
sys.exit(1 if failures else 0)
