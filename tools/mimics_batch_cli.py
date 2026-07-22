#!/usr/bin/env python3
"""External batch utilities for Mimics-Script.

This script runs outside the foreground Mimics GUI.

Commands:
  prepare-import  Convert dataset cases to prepare manifests and optionally
                  launch background Mimics to create .mcs files.
  export-labels   Launch background Mimics to export labels from saved .mcs.
  kill-background Stop integration-created bridge, background Mimics, and
                  nnInteractive service processes.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resource_locks import (
    FileResourceLock,
    ResourceLockTimeout,
    default_resource_lock_dir,
)

BRIDGE = ROOT / "mimics_bridge.py"
RUNTIME = ROOT / "runtime_py35"
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))
import runtime_common
RESOURCE_LOCK_DIR = default_resource_lock_dir(ROOT)


def project_python_candidates():
    return [
        ROOT / "nninteractive_env" / "python.exe",
        ROOT / "nninteractive_env" / "Scripts" / "python.exe",
        ROOT / "nninteractive_env" / "python" / "python.exe",
        ROOT / "nninteractive_env" / "bin" / "python3",
        ROOT / "nninteractive_env" / "bin" / "python",
    ]


def resolve_bridge_python(explicit=None):
    candidates = []
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = ROOT / path
        candidates.append(path)
    candidates.extend(project_python_candidates())
    env_value = os.environ.get("MIMICS_BRIDGE_PYTHON") or os.environ.get("MIMICS_FEWSHOT_PYTHON")
    if env_value:
        path = Path(env_value)
        if not path.is_absolute():
            path = ROOT / path
        candidates.append(path)
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError(
        "The nninteractive_env Python was not found. Run setup_offline.bat or pass --python explicitly."
    )


def write_json_atomic(path, payload, retries=20, max_sleep=0.25):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    last_error = None
    for attempt in range(max(1, int(retries))):
        tmp = path.with_name(path.name + "." + str(os.getpid()) + "." + uuid.uuid4().hex + ".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as handle:
                handle.write(text)
                try:
                    handle.flush()
                    os.fsync(handle.fileno())
                except Exception:
                    pass
            os.replace(str(tmp), str(path))
            return
        except OSError as exc:
            last_error = exc
            try:
                if tmp.is_file():
                    tmp.unlink()
            except Exception:
                pass
            time.sleep(min(float(max_sleep), 0.05 * (attempt + 1)))
    # SMB servers can allow create/write but deny rename/replace. Control JSON
    # readers tolerate a short incomplete interval, so direct overwrite is a
    # better final fallback than aborting a long import.
    try:
        with path.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except Exception:
                pass
        return
    except OSError as exc:
        last_error = exc
    if last_error is not None:
        raise last_error


def run_bridge(python_exe, params, env_overrides=None, on_wait=None):
    env = os.environ.copy()
    if env_overrides:
        for key, value in env_overrides.items():
            if value is None:
                continue
            env[str(key)] = str(value)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "1")
    proc = subprocess.Popen(
        [python_exe, str(BRIDGE)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    payload = json.dumps(params).encode("utf-8")
    try:
        stdout, stderr = proc.communicate(input=payload, timeout=15.0)
    except subprocess.TimeoutExpired:
        while True:
            if on_wait is not None:
                try:
                    on_wait()
                except Exception:
                    pass
            try:
                stdout, stderr = proc.communicate(timeout=15.0)
                break
            except subprocess.TimeoutExpired:
                continue
    if proc.returncode != 0:
        raise RuntimeError(stderr.decode("utf-8", "replace")[:2000])
    try:
        result = json.loads(stdout.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError("bridge returned invalid JSON: {}".format(exc))
    if result.get("status") != "ok":
        raise RuntimeError(result.get("error", "bridge returned non-ok status"))
    return result


def find_mimics_exe(explicit=None):
    """Find MimicsResearch.exe, with optional explicit override."""
    if explicit and Path(explicit).is_file():
        return explicit
    # Delegate to runtime_common which has the full search logic
    return runtime_common.find_mimics_exe()


def discover_cases(ts_root, cases, python_exe, mask_selection="all", env_overrides=None):
    result = run_bridge(
        python_exe,
        {
            "action": "discover",
            "ts_root": str(ts_root),
            "cases_filter": sorted(cases) if cases else None,
            "mask_selection": mask_selection,
        },
        env_overrides=env_overrides,
    )
    if result.get("mask_mode") == "named" and int(result.get("mask_count", 0) or 0) == 0:
        raise RuntimeError("No segmentation files matched --masks={0}".format(mask_selection))
    if result.get("mask_mode") == "named":
        print(
            "Mask filter matched {} file(s); {} case(s) have no matching mask.".format(
                int(result.get("mask_count", 0) or 0),
                int(result.get("cases_without_selected_masks", 0) or 0),
            )
        )
    return list(result.get("cases", []))


def _acquire_background_mimics_lock(owner, scope, wait_seconds=0.0):
    lock_path = Path(runtime_common.background_mimics_lock_path(str(ROOT), str(scope)))
    lock = FileResourceLock(lock_path, "background_mimics", owner)
    lock.acquire(wait_seconds=float(wait_seconds), poll_seconds=5.0)
    return lock


def _acquire_background_mimics_locks(owner, scopes, wait_seconds=0.0):
    """Acquire source/destination locks together without hold-and-wait."""
    paths = sorted(set(
        Path(runtime_common.background_mimics_lock_path(str(ROOT), str(scope)))
        for scope in scopes
    ))
    deadline = time.time() + max(0.0, float(wait_seconds))
    while True:
        locks = []
        try:
            for path in paths:
                lock = FileResourceLock(path, "background_mimics", owner)
                lock.acquire(wait_seconds=0.0, poll_seconds=5.0)
                locks.append(lock)
            return locks
        except ResourceLockTimeout:
            for lock in reversed(locks):
                lock.release()
            if time.time() >= deadline:
                raise
            time.sleep(min(1.0, max(0.05, deadline - time.time())))
        except Exception:
            for lock in reversed(locks):
                lock.release()
            raise


def clear_stale_import_queue_stop(output_dir):
    """Remove an old queue stop only when no live creator owns this output."""
    output_dir = Path(output_dir).resolve()
    runtime_dir = Path(runtime_common.import_queue_runtime_dir(str(ROOT), str(output_dir)))
    stop_path = runtime_dir / "_mcs_queue_stop.json"
    if not stop_path.is_file():
        return False
    lock_path = Path(runtime_common.background_mimics_lock_path(str(ROOT), str(output_dir)))
    holder = runtime_common.read_json(str(lock_path), {}) or {}
    holder_output = str(holder.get("output_dir") or "")
    same_output = bool(
        holder_output
        and os.path.normcase(os.path.abspath(holder_output))
        == os.path.normcase(os.path.abspath(str(output_dir)))
    )
    if (
        str(holder.get("kind") or "").lower() == "create_mcs"
        and same_output
        and runtime_common.process_matches(
            holder.get("pid"), holder.get("process_start_marker")
        )
    ):
        raise RuntimeError(
            "The previous import queue is still stopping for this output folder."
        )
    stop_path.unlink()
    return True


def _live_create_mcs_holder(output_dir):
    """Return the live creator lock for this output queue, if one exists."""
    output_dir = Path(output_dir).resolve()
    lock_path = Path(runtime_common.background_mimics_lock_path(str(ROOT), str(output_dir)))
    holder = runtime_common.read_json(str(lock_path), {}) or {}
    holder_output = str(holder.get("output_dir") or "")
    if (
        str(holder.get("kind") or "").lower() == "create_mcs"
        and holder_output
        and os.path.normcase(os.path.abspath(holder_output))
        == os.path.normcase(os.path.abspath(str(output_dir)))
        and runtime_common.process_matches(
            holder.get("pid"), holder.get("process_start_marker")
        )
    ):
        return holder
    return {}


def _update_import_queue_active(runtime_dir, total, completed=0, failed=0):
    """Publish/refresh external preparation progress for the queue consumer."""
    active_path = Path(runtime_dir) / "_mcs_queue_active.json"
    payload = runtime_common.read_json(str(active_path), {}) or {}
    payload.update({
        "status": "active",
        "total": int(total or 0),
        "completed": int(completed or 0),
        "failed": int(failed or 0),
        "pid": os.getpid(),
        "updated_at_epoch": time.time(),
    })
    marker = runtime_common.process_start_marker(os.getpid())
    if marker:
        payload["process_start_marker"] = marker
    write_json_atomic(active_path, payload)


def launch_create_mcs(
    output_dir,
    mimics_exe,
    bridge_python,
    lock_timeout_seconds=0.0,
    dicom_resample_mode=None,
    mask_resample_method=None,
):
    runtime_dir = Path(runtime_common.import_queue_runtime_dir(str(ROOT), str(output_dir)))
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runner = runtime_dir / "_run_create_mcs.py"
    handshake = runtime_dir / "_background_import_runner_started.json"
    runner_lines = [
        "# Auto-generated runner for background Mimics .mcs creation",
        "import sys, os, json, time",
        "open({0}, 'w').write(json.dumps({{'pid': os.getpid(), 'started_at_epoch': time.time()}}))".format(json.dumps(str(handshake))),
        "sys.path.insert(0, {})".format(json.dumps(str(RUNTIME))),
        "os.environ['MIMICS_BRIDGE_PYTHON'] = {}".format(json.dumps(str(bridge_python))),
        "os.environ['MIMICS_BRIDGE_SCRIPT'] = {}".format(json.dumps(str(ROOT / "mimics_bridge.py"))),
    ]
    if dicom_resample_mode:
        runner_lines.append(
            "os.environ['MIMICS_DICOM_RESAMPLE_MODE'] = {}".format(json.dumps(str(dicom_resample_mode)))
        )
    if mask_resample_method:
        runner_lines.append(
            "os.environ['MIMICS_MASK_RESAMPLE_METHOD'] = {}".format(json.dumps(str(mask_resample_method)))
        )
    runner_lines.extend([
        "import create_mcs_batch",
        "create_mcs_batch.main({}, runtime_dir={})".format(
            json.dumps(str(output_dir)), json.dumps(str(runtime_dir))
        ),
        "",
    ])
    runner.write_text("\n".join(runner_lines), encoding="utf-8")
    lock = _acquire_background_mimics_lock(
        "batch .mcs creation", output_dir, lock_timeout_seconds
    )
    log = open(str(runtime_dir / "_background_mimics.log"), "ab")
    mimics_log = runtime_dir / "_background_mimics_application.log"
    supervisor = ROOT / "tools" / "mcs_creation_supervisor.py"
    if not supervisor.is_file():
        lock.release()
        log.close()
        raise RuntimeError("MCS creation supervisor was not found: {}".format(supervisor))
    try:
        proc = subprocess.Popen(
            runtime_common.mcs_creation_supervisor_command(
                bridge_python,
                str(supervisor),
                mimics_exe,
                str(runner),
                str(runtime_dir),
                str(output_dir),
                handshake_path=str(handshake),
                mimics_log_path=str(mimics_log),
            ),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        if not lock.update_pid(
            proc.pid,
            kind="create_mcs",
            output_dir=str(output_dir.resolve()),
        ):
            raise RuntimeError(
                "Background Mimics started, but the background-Mimics lock "
                "could not be transferred to PID {}.".format(proc.pid)
            )
        return proc
    except Exception:
        poll = getattr(proc, "poll", None) if "proc" in locals() else None
        process_running = False
        if "proc" in locals():
            try:
                process_running = poll is None or poll() is None
            except Exception:
                process_running = True
        if process_running:
            try:
                proc.terminate()
                if hasattr(proc, "wait"):
                    proc.wait(timeout=10)
            except Exception:
                try:
                    proc.kill()
                    if hasattr(proc, "wait"):
                        proc.wait(timeout=10)
                except Exception:
                    pass
        process_stopped = not process_running
        if process_running:
            try:
                process_stopped = proc.poll() is not None
            except Exception:
                process_stopped = not runtime_common.process_exists(
                    getattr(proc, "pid", 0)
                )
        if process_stopped:
            lock.release()
        else:
            try:
                lock.update_pid(
                    proc.pid,
                    kind="create_mcs",
                    output_dir=str(output_dir.resolve()),
                    termination_pending=True,
                )
            except Exception:
                pass
        raise
    finally:
        log.close()


def launch_export_labels(ts_root, cases, mimics_exe, axes, flips, lock_timeout_seconds=0.0,
                         mcs_dir=None, label_output_root=None, overwrite_source=False,
                         mask_names=None):
    output_dir = Path(mcs_dir).resolve() if mcs_dir else ts_root / "mcs_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    job_id = "export_{}_{}".format(time.strftime("%Y%m%dT%H%M%S"), uuid.uuid4().hex[:8])
    job_dir = Path(ROOT) / ".mimics_runtime" / "export_jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    config = job_dir / "export_config.json"
    runner = job_dir / "run_export_batch.py"
    status_path = job_dir / "status.json"
    stop_path = job_dir / "_export_stop.json"
    write_json_atomic(
        config,
        {
            "ts_root": str(ts_root),
            "cases": sorted(cases) if cases else None,
            "axes": axes,
            "flips": flips,
            "output_dir": str(output_dir),
            "export_root": str(job_dir),
            "display_output_root": str(Path(label_output_root).resolve()) if label_output_root else str(ts_root),
            "label_output_root": str(Path(label_output_root).resolve()) if label_output_root else "",
            "overwrite_existing": bool(overwrite_source),
            "mask_names": list(mask_names or []),
            "status_path": str(status_path),
            "stop_path": str(stop_path),
            "job_runtime": str(job_dir),
        },
    )
    runner.write_text(
        "\n".join([
            "# Auto-generated runner for background Mimics batch export",
            "import sys, os",
            "sys.path.insert(0, r'{}')".format(str(RUNTIME)),
            "import mimics_export",
            "mimics_export.run_background_batch_export(r'{}')".format(str(config)),
            "",
        ]),
        encoding="utf-8",
    )
    export_scope = Path(label_output_root).resolve() if label_output_root else ts_root.resolve()
    locks = _acquire_background_mimics_locks(
        "batch label export",
        [output_dir, export_scope],
        lock_timeout_seconds,
    )
    log = open(str(job_dir / "process.log"), "ab")
    mimics_log = job_dir / "mimics_application.log"
    try:
        proc = subprocess.Popen(
            runtime_common.background_mimics_command(
                mimics_exe, str(runner), mimics_log_path=str(mimics_log)
            ),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        for lock in locks:
            if not lock.update_pid(
                proc.pid,
                kind="export_labels",
                ts_root=str(ts_root.resolve()),
                mcs_source=str(output_dir.resolve()),
                export_root=str(Path(label_output_root).resolve()) if label_output_root else str(ts_root),
                stop_path=str(stop_path),
            ):
                raise RuntimeError(
                    "Background Mimics started, but a source/destination lock "
                    "could not be transferred to PID {}.".format(proc.pid)
                )
        return proc
    except Exception:
        poll = getattr(proc, "poll", None) if "proc" in locals() else None
        process_running = False
        if "proc" in locals():
            try:
                process_running = poll is None or poll() is None
            except Exception:
                process_running = True
        if process_running:
            try:
                proc.terminate()
                if hasattr(proc, "wait"):
                    proc.wait(timeout=10)
            except Exception:
                try:
                    proc.kill()
                    if hasattr(proc, "wait"):
                        proc.wait(timeout=10)
                except Exception:
                    pass
        process_stopped = not process_running
        if process_running:
            try:
                process_stopped = proc.poll() is not None
            except Exception:
                process_stopped = not runtime_common.process_exists(
                    getattr(proc, "pid", 0)
                )
        if process_stopped:
            for lock in locks:
                lock.release()
        else:
            for lock in locks:
                try:
                    lock.update_pid(
                        proc.pid,
                        kind="export_labels",
                        ts_root=str(ts_root.resolve()),
                        mcs_source=str(output_dir.resolve()),
                        export_root=(
                            str(Path(label_output_root).resolve())
                            if label_output_root else str(ts_root)
                        ),
                        stop_path=str(stop_path),
                        termination_pending=True,
                    )
                except Exception:
                    pass
        raise
    finally:
        log.close()


def cmd_prepare_import(args):
    ts_root = Path(args.ts_root).resolve()
    bridge_python = resolve_bridge_python(args.python)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else ts_root / "mcs_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_filter = set(args.cases.split(",")) if args.cases else None
    bridge_env = {
        "MIMICS_DICOM_RESAMPLE_MODE": args.dicom_resample_mode,
        "MIMICS_MASK_RESAMPLE_METHOD": args.mask_resample_method,
    }
    runtime_dir = Path(runtime_common.import_queue_runtime_dir(str(ROOT), str(output_dir)))
    run_root = Path(runtime_common.import_runtime_base(str(ROOT))) / "import_runs" / (
        time.strftime("%Y%m%dT%H%M%S") + "_cli_" + uuid.uuid4().hex[:10]
    )
    queue_dir = runtime_dir / "prepared_queue"
    producer_lock = FileResourceLock(
        Path(runtime_common.import_producer_lock_path(str(ROOT), str(output_dir))),
        "import_producer",
        "external batch preparation for {}".format(output_dir),
    )
    try:
        producer_lock.acquire(wait_seconds=0.0)
        if not producer_lock.update_pid(
            os.getpid(),
            kind="prepare_import",
            output_dir=str(output_dir),
        ):
            producer_lock.release()
            raise RuntimeError("Import producer lock ownership changed during startup.")
        clear_stale_import_queue_stop(output_dir)
    except ResourceLockTimeout as exc:
        print(
            "Another import is preparing data for this output folder: {}".format(exc),
            file=sys.stderr,
        )
        return 75
    except Exception:
        producer_lock.release()
        raise

    completed = 0
    failed = 0
    cancelled = False
    queue_started = False
    creator_proc = None
    creator_joined = False
    creator_unavailable = False
    creator_warning = ""
    mimics_exe = None

    def _ensure_creator(wait_seconds=0.0, final_attempt=False):
        """Start or join the queue consumer without stopping preparation."""
        nonlocal creator_proc, creator_joined, creator_unavailable, creator_warning, mimics_exe
        if args.no_create_mcs:
            return True
        holder = _live_create_mcs_holder(output_dir)
        if holder:
            if not creator_joined:
                print(
                    "Using the background Mimics process already consuming this queue, PID={}.".format(
                        holder.get("pid", "?")
                    )
                )
            creator_joined = True
            return True
        if creator_proc is not None:
            try:
                if creator_proc.poll() is None:
                    return True
            except Exception:
                if runtime_common.process_exists(getattr(creator_proc, "pid", 0)):
                    return True
            creator_proc = None
        creator_joined = False
        if creator_unavailable:
            return False
        if mimics_exe is None:
            mimics_exe = find_mimics_exe(args.mimics_exe)
            if not mimics_exe:
                creator_unavailable = True
                creator_warning = (
                    "A background Mimics executable was not found. Prepared cases remain queued; "
                    "run .mcs creation later."
                )
                print(creator_warning, file=sys.stderr)
                return False
        try:
            creator_proc = launch_create_mcs(
                output_dir,
                mimics_exe,
                bridge_python,
                wait_seconds,
                dicom_resample_mode=args.dicom_resample_mode,
                mask_resample_method=args.mask_resample_method,
            )
            print(
                "Background Mimics started for streaming .mcs creation, PID={}.".format(
                    creator_proc.pid
                )
            )
            creator_warning = ""
            return True
        except ResourceLockTimeout as exc:
            creator_warning = "Background Mimics is busy: {}".format(exc)
            if final_attempt:
                print(creator_warning, file=sys.stderr)
            return False
        except Exception as exc:
            creator_warning = "Background Mimics could not start: {}".format(exc)
            if final_attempt:
                print(creator_warning, file=sys.stderr)
            return False

    try:
        cases = discover_cases(ts_root, cases_filter, bridge_python, args.masks, env_overrides=bridge_env)
        print("Discovered {} case(s).".format(len(cases)))
        _update_import_queue_active(runtime_dir, len(cases))
        queue_started = True
        for index, case in enumerate(cases, 1):
            if (runtime_dir / "_mcs_queue_stop.json").is_file():
                cancelled = True
                print("Import stop requested; no additional cases will be prepared.")
                break
            case_id = str(
                case.get("case_id") if isinstance(case, dict) else ""
            ) or "case_{:05d}".format(index)
            work_dir = run_root / "work" / case_id
            descriptor_committed = False
            try:
                if not isinstance(case, dict):
                    raise RuntimeError("Dataset discovery returned a non-object case entry.")
                image_path = str(case.get("image") or "")
                if not image_path:
                    raise RuntimeError("Dataset discovery returned a case without image data.")
                print("[{}/{}] Preparing {}".format(index, len(cases), case_id))
                params = {
                    "action": "prepare",
                    "image_path": image_path,
                    "masks": case.get("masks", []),
                    "dicom_out": str(work_dir / "derived_dicom"),
                    "buffers_out": str(work_dir / "buffers"),
                    "axes": args.axes,
                    "flips": args.flips,
                    "case_id": case_id,
                }
                result = run_bridge(
                    bridge_python,
                    params,
                    env_overrides=bridge_env,
                    on_wait=lambda: _update_import_queue_active(
                        runtime_dir, len(cases), completed, failed
                    ),
                )
                if (runtime_dir / "_mcs_queue_stop.json").is_file():
                    cancelled = True
                    shutil.rmtree(str(work_dir), ignore_errors=True)
                    print("  stop requested; prepared data was not queued")
                    break
                result["output_mcs"] = str(output_dir / (case_id + ".mcs"))
                write_json_atomic(work_dir / "prepare_manifest.json", result)
                write_json_atomic(
                    queue_dir / (case_id + "_" + uuid.uuid4().hex + ".json"),
                    {
                        "case_id": case_id,
                        "work_dir": str(work_dir.resolve()),
                        "output_mcs": result["output_mcs"],
                        "created_at_epoch": time.time(),
                    },
                )
                descriptor_committed = True
                completed += 1
                _update_import_queue_active(
                    runtime_dir, len(cases), completed, failed
                )
                print(
                    "  prepared and queued; .mcs creation can run in parallel ({}/{} prepared)".format(
                        completed, len(cases)
                    )
                )
                _ensure_creator(wait_seconds=0.0)
            except Exception as exc:
                if descriptor_committed:
                    print(
                        "  warning: the case is queued, but progress telemetry failed: {}".format(
                            exc
                        ),
                        file=sys.stderr,
                    )
                    _ensure_creator(wait_seconds=0.0)
                    continue
                failed += 1
                fail_dir = runtime_dir / "_failed_cases"
                try:
                    write_json_atomic(
                        fail_dir / (case_id + "_prepare.json"),
                        {"case_id": case_id, "phase": "prepare", "error": str(exc)},
                    )
                except Exception as record_exc:
                    print(
                        "  warning: could not persist failure details for {}: {}".format(
                            case_id, record_exc
                        ),
                        file=sys.stderr,
                    )
                shutil.rmtree(str(work_dir), ignore_errors=True)
                print("  failed; continuing with the next case: {}".format(exc))
                try:
                    _update_import_queue_active(
                        runtime_dir, len(cases), completed, failed
                    )
                except Exception as telemetry_exc:
                    print(
                        "  warning: could not refresh queue progress: {}".format(
                            telemetry_exc
                        ),
                        file=sys.stderr,
                    )
    finally:
        if queue_started:
            active = runtime_dir / "_mcs_queue_active.json"
            try:
                if active.exists():
                    active.unlink()
            except OSError as exc:
                print("Warning: could not remove the active queue marker: {}".format(exc), file=sys.stderr)
            try:
                write_json_atomic(
                    runtime_dir / "_mcs_queue_done.json",
                    {
                        "status": "cancelled" if cancelled else "done",
                        "completed": completed,
                        "failed": failed,
                        "updated_at_epoch": time.time(),
                    },
                )
            except Exception as exc:
                print("Warning: could not write the queue completion marker: {}".format(exc), file=sys.stderr)
        producer_lock.release()
    if cancelled:
        return 130
    if args.no_create_mcs:
        return 0
    if completed <= 0:
        print("No case was prepared successfully; no .mcs creation was started.", file=sys.stderr)
        return 1
    if not _ensure_creator(
        wait_seconds=args.background_mimics_lock_timeout_seconds,
        final_attempt=True,
    ):
        if creator_unavailable:
            return 2
        if creator_warning.startswith("Background Mimics is busy:"):
            return 75
        return 1
    if creator_joined:
        print("All prepared cases are queued for the existing background Mimics process.")
    else:
        print("Preparation finished; background Mimics continues consuming the queue.")
    return 0


def cmd_export_labels(args):
    mimics_exe = find_mimics_exe(args.mimics_exe)
    if not mimics_exe:
        print("A background Mimics executable was not found.", file=sys.stderr)
        return 2
    cases = set(args.cases.split(",")) if args.cases else None
    try:
        proc = launch_export_labels(
            Path(args.ts_root).resolve(),
            cases,
            mimics_exe,
            args.axes,
            args.flips,
            args.background_mimics_lock_timeout_seconds,
            mcs_dir=args.mcs_dir,
            label_output_root=args.output_dir,
            overwrite_source=args.overwrite_source,
            mask_names=[
                value.strip() for value in str(args.masks or "all").split(",")
                if value.strip() and value.strip().lower() != "all"
            ],
        )
    except ResourceLockTimeout as exc:
        print("Background Mimics is busy: {}".format(exc), file=sys.stderr)
        return 75
    print("Background Mimics started for label export, PID={}".format(proc.pid))
    return 0


def _runtime_owned_roots():
    result = []
    lock_paths = list(RESOURCE_LOCK_DIR.glob("background_mimics*.lock"))
    lock_paths.extend(RESOURCE_LOCK_DIR.glob("import_producer*.lock"))
    for lock_path in lock_paths:
        try:
            payload = json.loads(lock_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        details = payload.get("details") or {}
        output_dir = payload.get("output_dir") or details.get("output_dir")
        if output_dir:
            result.append(Path(output_dir))
        ts_root = payload.get("ts_root") or details.get("ts_root")
        if ts_root:
            ts_root_path = Path(ts_root)
            result.append(ts_root_path)
            result.append(ts_root_path / "mcs_output")
    registry = Path(runtime_common.import_runtime_base(str(ROOT))) / "mcs_queues"
    if registry.is_dir():
        for path in registry.glob("*.json"):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            output_dir = row.get("output_dir")
            if output_dir:
                result.append(Path(output_dir))
    unique = []
    seen = set()
    for path in result:
        try:
            normalized = str(path.resolve())
        except Exception:
            normalized = str(path)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique


def cmd_kill_background(args):
    if os.name != "nt":
        print("Process cleanup is implemented for Windows Mimics workstations.")
        return 0
    stopped_queues = []
    stop_payload = {
        "status": "stop_requested",
        "requested_at_epoch": time.time(),
        "reason": "tools/mimics_batch_cli.py kill-background",
    }
    for queue_dir in _runtime_owned_roots():
        queue_path = Path(queue_dir)
        if not queue_path.is_dir():
            continue
        try:
            runtime_dir = Path(runtime_common.import_queue_runtime_dir(str(ROOT), str(queue_path)))
            runtime_dir.mkdir(parents=True, exist_ok=True)
            write_json_atomic(runtime_dir / "_mcs_queue_stop.json", stop_payload)
            active = runtime_dir / "_mcs_queue_active.json"
            if active.is_file():
                active.unlink()
            stopped_queues.append(str(queue_path))
        except Exception:
            pass
    stop_markers = []
    lock_paths = list(RESOURCE_LOCK_DIR.glob("background_mimics*.lock"))
    lock_paths.extend(RESOURCE_LOCK_DIR.glob("import_producer*.lock"))
    lock_paths.append(RESOURCE_LOCK_DIR / "gpu.lock")
    for lock_path in lock_paths:
        try:
            lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
        except Exception:
            lock_payload = {}
        details = lock_payload.get("details") or {}
        for key in ("stop_path", "cancel_path"):
            marker = lock_payload.get(key) or details.get(key)
            if not marker:
                continue
            try:
                write_json_atomic(Path(marker), stop_payload)
                stop_markers.append(str(marker))
            except Exception:
                pass
    markers = [
        "mimics_bridge.py",
        "mimics_batch_cli.py",
        "nninteractive_bridge.py",
        "--async-worker",
        "_run_create_mcs.py",
        "_run_export_batch.py",
        "fewshot_pipeline.py",
        "nninteractive.inference.server.main",
        "--watchdog",
    ]
    ps_markers = "@(" + ",".join("'{}'".format(m.replace("'", "''")) for m in markers) + ")"
    owned_roots = [
        str(ROOT),
        str(ROOT / "nninteractive_env"),
        str(ROOT / "external"),
        str(ROOT / "tools"),
        str(ROOT / "runtime_py35"),
    ]
    owned_roots.extend(_runtime_owned_roots())
    ps_roots = "@(" + ",".join("'{}'".format(str(r).replace("'", "''")) for r in owned_roots) + ")"
    ps_queues = "@(" + ",".join("'{}'".format(str(r).replace("'", "''")) for r in stopped_queues) + ")"
    ps_stop_markers = "@(" + ",".join(
        "'{}'".format(str(r).replace("'", "''")) for r in stop_markers
    ) + ")"
    stop_log = ROOT / ".mimics_runtime" / "stop_background_last.json"
    stop_log.parent.mkdir(parents=True, exist_ok=True)
    cutoff_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    command = (
        "$markers={};"
        "$roots={};"
        "$queues={};"
        "$stopMarkers={};"
        "$out='{}';"
        "$foregroundPid={};"
        "$cutoff=[DateTime]::Parse('{}').ToUniversalTime();"
        "$matched=Get-CimInstance Win32_Process | Where-Object {{"
        "$cmd=$_.CommandLine; "
        "$cmd -and $_.ProcessId -ne $PID -and $_.ProcessId -ne $foregroundPid -and "
        "(-not $_.CreationDate -or $_.CreationDate.ToUniversalTime() -le $cutoff) -and "
        "($roots | Where-Object {{ $cmd -like ('*' + $_ + '*') }}) -and "
        "($markers | Where-Object {{ $cmd -like ('*' + $_ + '*') }})"
        "}};"
        "$records=@($matched | Select-Object ProcessId,Name,CommandLine);"
        "$killed=@();"
        "$matched | ForEach-Object {{"
        "  $procId=$_.ProcessId;"
        "  Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue;"
        "  $killed += [PSCustomObject]@{{ProcessId=$procId;Name=$_.Name;ExitCode=0;CommandLine=$_.CommandLine}};"
        "}};"
        "Start-Sleep -Milliseconds 500;"
        "$report=[PSCustomObject]@{{RequestedAt=(Get-Date).ToString('s');QueueStopDirs=$queues;StopMarkers=$stopMarkers;OwnedRoots=$roots;Matched=$records;Killed=$killed}};"
        "$report | ConvertTo-Json -Depth 5 -Compress | Set-Content -Path $out -Encoding UTF8"
    ).format(
        ps_markers,
        ps_roots,
        ps_queues,
        ps_stop_markers,
        str(stop_log).replace("'", "''"),
        int(os.getpid()),
        cutoff_utc,
    )
    subprocess.Popen(["powershell", "-NoProfile", "-Command", command], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("Stop request submitted for integration background processes. Queue stop markers: {}. Details: {}".format(len(stopped_queues), stop_log))
    return 0


def parse_axes(value):
    result = [int(part.strip()) for part in value.split(",")]
    if sorted(result) != [0, 1, 2]:
        raise argparse.ArgumentTypeError("axes must be a permutation like 0,1,2")
    return result


def parse_flips(value):
    parts = [part.strip().lower() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("flips must have three values")
    return [part in ("1", "true", "yes") for part in parts]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("prepare-import", help="Prepare dataset import and optionally create .mcs in background Mimics")
    p.add_argument("--ts-root", required=True)
    p.add_argument("--output-dir")
    p.add_argument("--cases")
    p.add_argument("--masks", default="all", help="all, none, or comma-separated mask names")
    p.add_argument("--python")
    p.add_argument("--mimics-exe")
    p.add_argument("--axes", type=parse_axes, default=[0, 1, 2])
    p.add_argument("--flips", type=parse_flips, default=[False, False, False])
    p.add_argument(
        "--dicom-resample-mode",
        choices=["auto", "axial", "never"],
        default="auto",
        help="Derived DICOM grid policy for source medical images.",
    )
    p.add_argument(
        "--mask-resample-method",
        choices=["distance", "nearest"],
        default="nearest",
        help="Mask resampling method when mapping to image grids.",
    )
    p.add_argument("--no-create-mcs", action="store_true")
    p.add_argument("--background-mimics-lock-timeout-seconds", type=float, default=0.0)
    p.set_defaults(func=cmd_prepare_import)

    p = sub.add_parser("export-labels", help="Export labels from saved .mcs files in background Mimics")
    p.add_argument("--ts-root", required=True)
    p.add_argument("--cases")
    p.add_argument("--mimics-exe")
    p.add_argument("--mcs-dir", help="Folder containing the saved .mcs projects")
    p.add_argument("--masks", default="all", help="all or comma-separated mask names")
    destination = p.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output-dir", help="Safe export root; writes <root>/<case>/segmentations without overwriting")
    destination.add_argument("--overwrite-source", action="store_true", help="Explicitly overwrite <case>/segmentations")
    p.add_argument("--axes", type=parse_axes, default=[0, 1, 2])
    p.add_argument("--flips", type=parse_flips, default=[False, False, False])
    p.add_argument("--background-mimics-lock-timeout-seconds", type=float, default=0.0)
    p.set_defaults(func=cmd_export_labels)

    p = sub.add_parser("kill-background", help="Stop integration-created background processes")
    p.set_defaults(func=cmd_kill_background)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
