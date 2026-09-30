#!/usr/bin/env python3
"""External batch utilities for Mimics-Script.

This script runs outside the foreground Mimics GUI.

Commands:
  prepare-import  Convert dataset cases to prepare manifests and optionally
                  launch background Mimics to create .mcs files.
  export-labels   Launch background Mimics to export labels from saved .mcs.
    append-masks    Add arbitrary named NIfTI Masks to existing .mcs files and
                                    save new projects without overwriting source.
  kill-background Stop integration-created bridge, background Mimics, and
                  nnInteractive service processes.

For export-labels, --image-root is independent from --mcs-dir. Case IDs come
from .mcs filenames; source images are resolved from either
<image-root>/<case-id>/ct.nii.gz or <image-root>/<case-id>.nii.gz. The resolved
source path is passed explicitly to the bridge instead of using stale project
metadata.
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


TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
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
import pipeline_common
RESOURCE_LOCK_DIR = default_resource_lock_dir(ROOT)


def project_python_candidates():
    return [
        ROOT / "python_env" / "python.exe",
        ROOT / "python_env" / "Scripts" / "python.exe",
        ROOT / "python_env" / "python" / "python.exe",
        ROOT / "python_env" / "bin" / "python3",
        ROOT / "python_env" / "bin" / "python",
        ROOT / "nninteractive_env" / "python.exe",
        ROOT / "nninteractive_env" / "Scripts" / "python.exe",
        ROOT / "nninteractive_env" / "python" / "python.exe",
        ROOT / "nninteractive_env" / "bin" / "python3",
        ROOT / "nninteractive_env" / "bin" / "python",
    ]


def resolve_bridge_python(explicit=None):
    # Explicit arg wins, then canonical discovery (which honors
    # MIMICS_BRIDGE_PYTHON / NNINTERACTIVE_PYTHON + standard layouts).
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = ROOT / path
        if path.is_file():
            return str(path)
    found = runtime_common.find_external_python(str(ROOT))
    if found:
        return found
    candidates = list(project_python_candidates())
    env_value = os.environ.get("MIMICS_AI_PYTHON") or os.environ.get("MIMICS_FEWSHOT_PYTHON")
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
    pipeline_common.write_json_atomic(path, payload, retries=retries, max_sleep=max_sleep)


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
    started = time.time()
    last_notice = [0.0]

    def on_wait(holder):
        now = time.time()
        if now - last_notice[0] < 30.0:
            return
        print(
            "Waiting for background Mimics held by {} (pid {}); {:.0f}s "
            "elapsed. Press Ctrl+C to cancel this wait.".format(
                str((holder or {}).get("owner") or "another task"),
                str((holder or {}).get("pid") or "unknown"),
                now - started,
            ),
            flush=True,
        )
        last_notice[0] = now

    lock.acquire(
        wait_seconds=float(wait_seconds),
        poll_seconds=5.0,
        on_wait=on_wait,
    )
    return lock


def _acquire_background_mimics_locks(owner, scopes, wait_seconds=0.0):
    """Acquire source/destination locks together without hold-and-wait."""
    paths = sorted(set(
        Path(runtime_common.background_mimics_lock_path(str(ROOT), str(scope)))
        for scope in scopes
    ))
    deadline = time.time() + max(0.0, float(wait_seconds))
    started = time.time()
    last_notice = 0.0
    while True:
        locks = []
        blocked_holder = {}
        try:
            for path in paths:
                lock = FileResourceLock(path, "background_mimics", owner)
                try:
                    lock.acquire(wait_seconds=0.0, poll_seconds=5.0)
                except ResourceLockTimeout:
                    read_holder = getattr(lock, "read", None)
                    blocked_holder = read_holder() if callable(read_holder) else {}
                    raise
                locks.append(lock)
            return locks
        except ResourceLockTimeout:
            for lock in reversed(locks):
                lock.release()
            if time.time() >= deadline:
                raise
            now = time.time()
            if now - last_notice >= 30.0:
                print(
                    "Waiting for background Mimics held by {} (pid {}); "
                    "{:.0f}s elapsed. Press Ctrl+C to cancel this wait.".format(
                        str(blocked_holder.get("owner") or "another task"),
                        str(blocked_holder.get("pid") or "unknown"),
                        now - started,
                    ),
                    flush=True,
                )
                last_notice = now
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
                         mask_names=None, case_dirs=None, mcs_paths=None,
                         source_image_paths=None, mask_resample_method="nearest",
                         export_space="source_image", source_mask_root=None):
    output_dir = Path(mcs_dir).resolve() if mcs_dir else ts_root / "mcs_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    job_id = "export_{}_{}".format(time.strftime("%Y%m%dT%H%M%S"), uuid.uuid4().hex[:8])
    job_dir = Path(ROOT) / ".mimics_runtime" / "export_jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    config = job_dir / "export_config.json"
    runner = job_dir / "run_export_batch.py"
    status_path = job_dir / "status.json"
    stop_path = job_dir / "_export_stop.json"
    source_mask_root = Path(source_mask_root).resolve() if source_mask_root else None
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
            "case_dirs": dict(case_dirs or {}),
            "mcs_paths": dict(mcs_paths or {}),
            "source_image_paths": dict(source_image_paths or {}),
            "mask_resample_method": str(mask_resample_method or "nearest"),
            "export_space": str(export_space or "source_image"),
            "status_path": str(status_path),
            "stop_path": str(stop_path),
            "job_runtime": str(job_dir),
            "source_mask_root": str(Path(source_mask_root).resolve()) if source_mask_root else "",
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


_SOURCE_IMAGE_SUFFIXES = (
    ".nii.gz", ".nii", ".mha", ".mhd", ".nrrd", ".nrrd.gz",
)
_SOURCE_IMAGE_PREFERRED_NAMES = (
    "ct.nii.gz", "mr.nii.gz", "mri.nii.gz", "image.nii.gz",
    "ct.nii", "mr.nii", "mri.nii", "image.nii",
    "ct.mha", "mr.mha", "mri.mha", "image.mha",
    "ct.mhd", "mr.mhd", "mri.mhd", "image.mhd",
    "ct.nrrd", "mr.nrrd", "mri.nrrd", "image.nrrd",
)


def _is_source_image_file(path):
    return str(path).lower().endswith(_SOURCE_IMAGE_SUFFIXES)


def _find_source_image_in_case_dir(case_dir):
    """Return one unambiguous source image or DICOM folder in a case dir."""
    case_dir = Path(case_dir)
    if not case_dir.is_dir():
        return None
    files = [
        path for path in case_dir.iterdir()
        if path.is_file() and _is_source_image_file(path)
        and not any(token in path.name.lower() for token in ("label", "mask", "seg"))
    ]
    by_name = {path.name.lower(): path for path in files}
    for name in _SOURCE_IMAGE_PREFERRED_NAMES:
        if name in by_name:
            return by_name[name].resolve()
    if len(files) == 1:
        return files[0].resolve()
    dicom_dir = case_dir / "dicom"
    if dicom_dir.is_dir():
        return dicom_dir.resolve()
    # The bridge performs authoritative DICOM header validation later.
    try:
        if any(path.is_file() and path.suffix.lower() == ".dcm" for path in case_dir.iterdir()):
            return case_dir.resolve()
    except OSError:
        pass
    return None


def _source_case_id_from_image(path):
    name = Path(path).name.lower()
    for suffix in _SOURCE_IMAGE_SUFFIXES:
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return Path(path).stem.lower()


def _resolve_source_for_case(image_root, case_id, total_cases=1):
    """Resolve an image path without consulting the .mcs metadata."""
    root = Path(image_root).expanduser()
    if root.is_file():
        return root.resolve(), root.parent.resolve()
    if not root.is_dir():
        return None, None

    direct_case_dir = root / str(case_id)
    if direct_case_dir.is_dir():
        image = _find_source_image_in_case_dir(direct_case_dir)
        if image is not None:
            return image, direct_case_dir.resolve()

    exact_files = []
    case_text = str(case_id).lower()
    try:
        for path in root.iterdir():
            if path.is_file() and _is_source_image_file(path):
                if _source_case_id_from_image(path) == case_text:
                    exact_files.append(path.resolve())
    except OSError:
        pass
    if len(exact_files) == 1:
        return exact_files[0], exact_files[0].parent.resolve()
    if len(exact_files) > 1:
        raise RuntimeError(
            "More than one source image matches case '{}': {}".format(
                case_id, ", ".join(str(path) for path in exact_files)
            )
        )

    # Allow grouped image roots such as image_root/group_a/case001/ct.nii.gz.
    matching_dirs = []
    matching_files = []
    for current, _dirnames, filenames in os.walk(str(root)):
        current_path = Path(current)
        if current_path.name.lower() == case_text:
            matching_dirs.append(current_path)
        for filename in filenames:
            path = current_path / filename
            if _is_source_image_file(path) and _source_case_id_from_image(path) == case_text:
                matching_files.append(path)
    resolved = []
    for case_dir in matching_dirs:
        image = _find_source_image_in_case_dir(case_dir)
        if image is not None:
            resolved.append((image, case_dir.resolve()))
    resolved.extend((path.resolve(), path.parent.resolve()) for path in matching_files)
    unique = {}
    for image, case_dir in resolved:
        unique[str(image).lower()] = (image, case_dir)
    if len(unique) == 1:
        return list(unique.values())[0]
    if len(unique) > 1:
        raise RuntimeError(
            "More than one source image matches case '{}': {}".format(
                case_id, ", ".join(sorted(unique))
            )
        )

    if int(total_cases or 0) == 1:
        image = _find_source_image_in_case_dir(root)
        if image is not None:
            return image, root.resolve()
    return None, None


def discover_export_sources(mcs_dir, image_root, cases_filter=None, recursive=True):
    """Build explicit case-to-MCS/source-image mappings before Mimics starts."""
    mcs_root = Path(mcs_dir).expanduser()
    if mcs_root.is_file():
        mcs_files = [mcs_root] if mcs_root.suffix.lower() == ".mcs" else []
    elif mcs_root.is_dir():
        mcs_files = []
        if recursive:
            file_groups = (
                (current, filenames)
                for current, _dirnames, filenames in os.walk(str(mcs_root))
            )
        else:
            file_groups = ((str(mcs_root), os.listdir(str(mcs_root))),)
        for current, filenames in file_groups:
            for filename in filenames:
                if filename.lower().endswith(".mcs"):
                    mcs_files.append(Path(current) / filename)
        mcs_files.sort(key=lambda path: str(path).lower())
    else:
        raise RuntimeError("The .mcs directory was not found: {}".format(mcs_root))
    if not mcs_files:
        raise RuntimeError("No .mcs files were found under: {}".format(mcs_root))

    requested = set(
        str(value).strip() for value in (cases_filter or set()) if str(value).strip()
    )
    rows = {}
    for mcs_path in mcs_files:
        case_id = mcs_path.name[:-4]
        if requested and case_id not in requested:
            continue
        if case_id in rows:
            raise RuntimeError(
                "More than one .mcs file has the same case name '{}': {} and {}".format(
                    case_id, rows[case_id]["mcs_path"], mcs_path
                )
            )
        rows[case_id] = {"mcs_path": str(mcs_path.resolve())}
    if requested:
        missing = sorted(requested.difference(rows))
        if missing:
            raise RuntimeError(
                "Requested .mcs case(s) were not found: {}".format(", ".join(missing))
            )
    if not rows:
        raise RuntimeError("No selected .mcs files remain after applying --cases.")

    case_dirs = {}
    mcs_paths = {}
    source_image_paths = {}
    missing_images = []
    total = len(rows)
    for case_id in sorted(rows):
        image, case_dir = _resolve_source_for_case(image_root, case_id, total_cases=total)
        if image is None:
            missing_images.append(case_id)
            continue
        mcs_paths[case_id] = rows[case_id]["mcs_path"]
        source_image_paths[case_id] = str(image)
        case_dirs[case_id] = str(case_dir or Path(image).parent)
    if missing_images:
        raise RuntimeError(
            "Could not resolve an original image for case(s) {} under '{}'. "
            "Expected <image-root>/<case>/ct.nii.gz or <image-root>/<case>.nii.gz; "
            "the .mcs metadata path was not used.".format(
                ", ".join(missing_images), image_root
            )
        )
    return {
        "case_dirs": case_dirs,
        "mcs_paths": mcs_paths,
        "source_image_paths": source_image_paths,
    }


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
    image_root_value = args.image_root or args.ts_root
    if not image_root_value:
        print("export-labels requires --image-root (or the legacy --ts-root).", file=sys.stderr)
        return 2
    image_root = Path(image_root_value).resolve()
    mcs_dir = Path(args.mcs_dir).resolve() if args.mcs_dir else image_root / "mcs_output"
    source_mask_root = Path(args.source_mask_root).resolve() if args.source_mask_root else None
    if source_mask_root is not None and not source_mask_root.is_dir():
        print(
            "The canonical source-mask root was not found: {}".format(source_mask_root),
            file=sys.stderr,
        )
        return 2
    requested_cases = set(args.cases.split(",")) if args.cases else None
    try:
        mapping = discover_export_sources(
            mcs_dir,
            image_root,
            requested_cases,
            recursive=not args.mcs_root_only,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    cases = set(mapping["mcs_paths"])
    try:
        proc = launch_export_labels(
            image_root,
            cases,
            mimics_exe,
            args.axes,
            args.flips,
            args.background_mimics_lock_timeout_seconds,
            mcs_dir=mcs_dir,
            label_output_root=args.output_dir,
            overwrite_source=args.overwrite_source,
            mask_names=[
                value.strip() for value in str(args.masks or "all").split(",")
                if value.strip() and value.strip().lower() != "all"
            ],
            case_dirs=mapping["case_dirs"],
            mcs_paths=mapping["mcs_paths"],
            source_image_paths=mapping["source_image_paths"],
            mask_resample_method=args.mask_resample_method,
            export_space=args.export_space,
            source_mask_root=source_mask_root,
        )
    except ResourceLockTimeout as exc:
        print("Background Mimics is busy: {}".format(exc), file=sys.stderr)
        return 75
    print("Background Mimics started for label export, PID={}".format(proc.pid))
    return 0


def _index_append_files(root, suffixes, kind, nested_mask_name=None):
    root = Path(root).expanduser()
    if not root.is_dir():
        raise RuntimeError("The {} directory was not found: {}".format(kind, root))
    indexed = {}
    duplicates = []
    direct_paths = [
        path for path in sorted(root.iterdir(), key=lambda item: item.name.lower())
        if path.is_file()
        and any(path.name.lower().endswith(suffix) for suffix in suffixes)
    ]
    candidates = [(path, None) for path in direct_paths]
    if not candidates and nested_mask_name:
        expected_names = [
            str(nested_mask_name).strip() + suffix for suffix in suffixes
        ]
        for case_dir in sorted(root.iterdir(), key=lambda item: item.name.lower()):
            if not case_dir.is_dir():
                continue
            segmentation_dir = case_dir / "segmentations"
            for expected_name in expected_names:
                path = segmentation_dir / expected_name
                if path.is_file():
                    candidates.append((path, case_dir.name))
    for path, nested_case_id in candidates:
        if not path.is_file():
            continue
        lower = path.name.lower()
        if not any(lower.endswith(suffix) for suffix in suffixes):
            continue
        if kind == "MCS":
            case_id = path.stem
        elif nested_case_id:
            case_id = nested_case_id
        elif lower.endswith(".nii.gz"):
            case_id = path.name[:-7]
        else:
            case_id = path.name[:-4]
        key = case_id.lower()
        if key in indexed:
            duplicates.append((case_id, str(indexed[key]), str(path)))
        else:
            indexed[key] = path
    if duplicates:
        raise RuntimeError(
            "Duplicate case IDs were found in {}: {}".format(kind, duplicates[:5])
        )
    return indexed


APPEND_JOB_RETENTION_DAYS = 14
# Status values that mean a job dir may still be in use by a live process.
ACTIVE_JOB_STATUSES = {"", "running", "queued", "starting", "active"}


def prune_append_jobs(retention_days=APPEND_JOB_RETENTION_DAYS):
    """Delete append-mask job dirs past the retention window (best effort).

    Only terminal jobs (no live background Mimics holding their locks) are
    pruned; jobs younger than the window are left alone.
    """
    base = Path(ROOT) / ".mimics_runtime" / "append_jobs"
    if not base.is_dir():
        return
    cutoff = time.time() - retention_days * 86400
    try:
        entries = list(base.iterdir())
    except OSError:
        return
    for entry in entries:
        try:
            if not entry.is_dir() or entry.stat().st_mtime >= cutoff:
                continue
            status = runtime_common.read_json(
                str(entry / "status.json"), {}
            ) or {}
            if str(status.get("status") or "") in ACTIVE_JOB_STATUSES:
                continue
            shutil.rmtree(entry, ignore_errors=True)
        except OSError:
            continue


def launch_append_masks(plan, mimics_exe, force=False,
                        lock_timeout_seconds=0.0):
    """Launch a background Mimics process that embeds named Masks per case."""
    output_dir = Path(plan["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prune_append_jobs()
    job_id = "append_masks_{}_{}".format(
        time.strftime("%Y%m%dT%H%M%S"), uuid.uuid4().hex[:8]
    )
    job_dir = Path(ROOT) / ".mimics_runtime" / "append_jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    config = job_dir / "append_config.json"
    runner = job_dir / "run_append.py"
    status_path = job_dir / "status.json"
    stop_path = job_dir / "stop.json"
    write_json_atomic(
        config,
        {
            "schema_version": "append_masks_job.v1",
            "job_dir": str(job_dir),
            "mcs_dir": plan["mcs_dir"],
            "output_dir": str(output_dir),
            "in_place": bool(plan.get("in_place", False)),
            "mask_specs": plan["mask_specs"],
            "cases": plan["cases"],
            "force": bool(force),
            "status_path": str(status_path),
            "stop_path": str(stop_path),
        },
    )
    runner.write_text(
        "\n".join([
            "# Auto-generated named-Mask MCS append runner",
            "import sys",
            "sys.path.insert(0, r'{}')".format(str(RUNTIME)),
            "import append_masks_batch",
            "_result = append_masks_batch.main(r'{}')".format(str(config)),
            "if _result:\n    raise SystemExit(_result)",
            "",
        ]),
        encoding="utf-8",
    )
    lock_scopes = [Path(plan["mcs_dir"]), output_dir]
    lock_scopes.extend(
        Path(item["directory"])
        for item in plan.get("mask_specs", [])
    )
    locks = _acquire_background_mimics_locks(
        "append named Masks",
        lock_scopes,
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
                kind="append_masks",
                mcs_source=str(Path(plan["mcs_dir"]).resolve()),
                output_dir=str(output_dir),
                status_path=str(status_path),
                stop_path=str(stop_path),
            ):
                raise RuntimeError(
                    "Background Mimics started, but the append lock could not "
                    "be transferred to PID {}.".format(proc.pid)
                )
        proc._mimics_append_job_dir = str(job_dir)
        proc._mimics_append_status_path = str(status_path)
        return proc
    except Exception:
        for lock in locks:
            try:
                lock.release()
            except Exception:
                pass
        raise
    finally:
        log.close()


def _parse_append_mask_specs(args):
    if not args.mask:
        raise RuntimeError("--mask is required; use NAME=PATH and repeat --mask for each Mask.")
    specs = []
    raw_values = args.mask if isinstance(args.mask, (list, tuple)) else [args.mask]
    for raw in raw_values:
        value = raw.strip()
        if not value or "=" not in value:
            raise RuntimeError(
                "Invalid --mask value '{}'; expected NAME=PATH.".format(value)
            )
        name, path = value.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise RuntimeError(
                "Invalid --mask value '{}'; expected NAME=PATH.".format(value)
            )
        specs.append((name, path))
    names = [name for name, _path in specs]
    if len(set(names)) != len(names):
        raise RuntimeError("Duplicate Mask names were supplied: {}".format(", ".join(names)))
    return specs


def discover_append_mask_cases(mcs_dir, mask_specs, output_dir,
                               requested_cases=None, in_place=False):
    """Build source/output mappings for arbitrary named Mask files.

    Each mask specification is NAME=DIR.  Files are matched by case ID from
    their .nii/.nii.gz basename, and only the all-input intersection is queued.
    """
    mcs_root = Path(mcs_dir).expanduser().resolve()
    destination_root = Path(output_dir).expanduser().resolve()
    if mcs_root == destination_root and not in_place:
        raise RuntimeError(
            "Refusing to append in place. --output-dir must differ from --mcs-dir: {}"
            .format(mcs_root)
        )
    mcs = _index_append_files(mcs_root, (".mcs",), "MCS")
    indexed_masks = []
    for name, directory in mask_specs:
        indexed_masks.append((
            name,
            _index_append_files(
                directory,
                (".nii.gz", ".nii"),
                name + " masks",
                nested_mask_name=name,
            ),
        ))
    common = set(mcs)
    for _name, indexed in indexed_masks:
        common &= set(indexed)
    common = sorted(common)
    if requested_cases:
        requested = set(
            str(value).strip().lower()
            for value in requested_cases
            if str(value).strip()
        )
        missing = sorted(requested - set(mcs))
        if missing:
            raise RuntimeError(
                "Requested MCS case(s) were not found: {}".format(", ".join(missing))
            )
        common = [case_id for case_id in common if case_id in requested]
    if not common:
        raise RuntimeError("No cases have an MCS plus every requested Mask.")
    rows = []
    for key in common:
        case_id = mcs[key].stem
        rows.append({
            "case_id": case_id,
            "mcs_path": str(mcs[key]),
            "masks": [
                {"name": name, "mask_path": str(indexed[key])}
                for name, indexed in indexed_masks
            ],
            "output_mcs_path": str(
                mcs[key] if in_place else destination_root / (case_id + ".mcs")
            ),
        })
    counts = {
        "mcs": len(mcs),
        "common_cases": len(rows),
    }
    for name, indexed in indexed_masks:
        counts[name + "_masks"] = len(indexed)
        counts[name + "_missing_mcs"] = len(set(indexed) - set(mcs))
        counts["mcs_missing_" + name] = len(set(mcs) - set(indexed))
    return {
        "mcs_dir": str(mcs_root),
        "output_dir": str(destination_root),
        "in_place": bool(in_place),
        "mask_specs": [
            {"name": name, "directory": str(Path(directory).expanduser().resolve())}
            for name, directory in mask_specs
        ],
        "counts": counts,
        "cases": rows,
    }


def cmd_append_masks(args):
    mimics_exe = find_mimics_exe(args.mimics_exe)
    if not mimics_exe:
        print("A background Mimics executable was not found.", file=sys.stderr)
        return 2
    try:
        mask_specs = _parse_append_mask_specs(args)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    requested = args.cases.split(",") if args.cases else None
    try:
        plan = discover_append_mask_cases(
            args.mcs_dir,
            mask_specs,
            args.mcs_dir if args.in_place else args.output_dir,
            requested_cases=requested,
            in_place=args.in_place,
        )
        proc = launch_append_masks(
            plan,
            mimics_exe,
            force=args.force,
            lock_timeout_seconds=args.background_mimics_lock_timeout_seconds,
        )
    except (RuntimeError, ResourceLockTimeout) as exc:
        print(str(exc), file=sys.stderr)
        return 75 if isinstance(exc, ResourceLockTimeout) else 2
    counts = plan["counts"]
    print(
        "Background Mimics started for named-Mask MCS append, PID={}. "
        "{} case(s) queued.".format(proc.pid, counts["common_cases"])
    )
    if plan.get("in_place"):
        print("In-place MCS update: source files will be atomically replaced after successful save.")
    else:
        print("Output directory: {}".format(plan["output_dir"]))
    print("Job directory: {}".format(getattr(proc, "_mimics_append_job_dir", "")))
    print("Status: {}".format(getattr(proc, "_mimics_append_status_path", "")))
    return 0


def _predict_cases(image_root, cases_filter):
    """Resolve (case_id, image_path) pairs for batch prediction.

    Accepts a dataset-style root (<root>/<case>/ containing one source
    image or DICOM folder - the same layout prepare-import builds from)
    or a flat folder of image files. Ambiguous case dirs are skipped and
    reported, never guessed: predicting on the wrong image is worse than
    not predicting.
    """
    root = Path(image_root).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError("Image root not found: {}".format(root))
    if root.is_file():
        raise RuntimeError(
            "--image-root must be a directory of cases, not a single file."
        )
    requested = (
        set(v.strip() for v in cases_filter.split(",") if v.strip())
        if cases_filter else None
    )
    resolved, skipped = [], []
    for entry in sorted(root.iterdir()):
        case_id = entry.name
        if requested is not None and case_id not in requested:
            continue
        if entry.is_dir():
            image = _find_source_image_in_case_dir(entry)
        elif entry.is_file() and _is_source_image_file(entry):
            image = entry.resolve()
        else:
            continue
        if image is None:
            skipped.append(case_id)
            continue
        resolved.append((case_id, image))
    if requested:
        missing = sorted(requested - {case_id for case_id, _ in resolved})
        if missing:
            raise RuntimeError(
                "Requested case(s) not found: {}".format(", ".join(missing))
            )
    if not resolved:
        raise RuntimeError(
            "No predictable case was found under: {}".format(root)
        )
    return resolved, skipped


def cmd_predict(args):
    """Run an nnU-Net model over a directory of cases in background jobs."""
    from nnunet_common import load_models, model_usability
    from nnunet_jobs import create_job

    workspace = Path(args.workspace or Path.home() / ".mimics_script" / "nnunet").expanduser().resolve()
    models = load_models(workspace, include_missing=True)
    matches = [
        model for model in models
        if str(model.get("model_id") or "") == args.model_id
        or str(model.get("task_id") or "") == args.model_id
    ]
    usable = [model for model in matches if model_usability(model)[0]]
    if not usable:
        for model in matches:
            ok, reason = model_usability(model)
            if not ok:
                print("Model {} is not usable: {}".format(
                    model.get("model_id"), reason), file=sys.stderr)
        print(
            "No usable model matches '{}' in workspace {}.".format(
                args.model_id, workspace),
            file=sys.stderr,
        )
        return 2

    try:
        cases, skipped = _predict_cases(args.image_root, args.cases)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    for case_id in skipped:
        print("Skipped {}: no unambiguous source image.".format(case_id),
              file=sys.stderr)

    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    model = usable[0]
    manifest_path = str(model["manifest_path"])
    started = []
    for case_id, image_path in cases:
        request = {
            "operation": "infer",
            "workspace": str(workspace),
            "job_id": "predict_{}_{}".format(
                time.strftime("%Y%m%dT%H%M%S"), case_id),
            "task_id": str(model.get("task_id") or ""),
            "task_name": str(model.get("task_name") or model.get("task_id") or ""),
            "model_manifest": manifest_path,
            "image_path": str(image_path),
            "output_path": str(output_root / "{}.nii.gz".format(case_id)),
            "disable_tta": bool(args.disable_tta),
            "use_cpu": bool(args.use_cpu),
            "case_id": case_id,
            "source_modality": str(args.source_modality or ""),
        }
        try:
            job = create_job(request)
        except Exception as exc:
            print("Could not start job for {}: {}".format(case_id, exc),
                  file=sys.stderr)
            continue
        started.append((case_id, job))
    print(
        "Started {} prediction job(s) with model {}.".format(
            len(started), model.get("model_id"))
    )
    print("Output directory: {}".format(output_root))
    for case_id, job in started:
        print("  {}: {}".format(case_id, job["status_path"]))
    return 0 if started else 1


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
        "nninteractive.inference.server.main",
        "interactive_algorithms_worker.py",
        "--watchdog",
    ]
    ps_markers = "@(" + ",".join("'{}'".format(m.replace("'", "''")) for m in markers) + ")"
    owned_roots = [
        str(ROOT),
        str(ROOT / "python_env"),
        str(ROOT / "nninteractive_env"),
        str(ROOT / "integrations"),
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
    p.add_argument("--ts-root", help="Legacy alias for --image-root")
    p.add_argument(
        "--image-root",
        help="Root containing original images; this overrides paths stored in .mcs metadata",
    )
    p.add_argument("--cases")
    p.add_argument("--mimics-exe")
    p.add_argument("--mcs-dir", help="Folder containing the saved .mcs projects")
    p.add_argument(
        "--mcs-root-only",
        action="store_true",
        help="When --mcs-dir is a folder, use only .mcs files directly in that folder (do not recurse)",
    )
    p.add_argument("--masks", default="all", help="all or comma-separated mask names")
    p.add_argument(
        "--source-mask-root",
        help=(
            "Explicit canonical source-label root. For each case, selected masks are read from "
            "<root>/<case>/segmentations/<mask>.nii[.gz] instead of the voxel data stored in .mcs. "
            "Missing or geometrically mismatched source masks fail the case."
        ),
    )
    destination = p.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output-dir", help="Safe export root; writes <root>/<case>/segmentations without overwriting")
    destination.add_argument("--overwrite-source", action="store_true", help="Explicitly overwrite <case>/segmentations")
    p.add_argument("--axes", type=parse_axes, default=[0, 1, 2])
    p.add_argument("--flips", type=parse_flips, default=[False, False, False])
    p.add_argument(
        "--mask-resample-method",
        choices=["nearest", "distance"],
        default="nearest",
        help=(
            "Binary Mask resampling when a Mimics grid differs from the source grid. "
            "Use distance to reduce stair-step artifacts after oblique regridding."
        ),
    )
    p.add_argument(
        "--export-space",
        choices=["source_image", "mimics_grid"],
        default="source_image",
        help=(
            "Output grid. source_image matches the original image; mimics_grid "
            "writes the raw Mask on the grid stored in the .mcs project."
        ),
    )
    p.add_argument("--background-mimics-lock-timeout-seconds", type=float, default=0.0)
    p.set_defaults(func=cmd_export_labels)

    p = sub.add_parser(
        "append-masks",
        help="Open existing MCS files and add arbitrary named Masks",
    )
    p.add_argument("--mcs-dir", required=True)
    p.add_argument(
        "--mask",
        action="append",
        required=True,
        help="Mask specification NAME=DIR; repeat --mask for multiple named NIfTI directories",
    )
    destination = p.add_mutually_exclusive_group(required=True)
    destination.add_argument(
        "--output-dir",
        help="Separate output folder for newly saved MCS files; source MCS files are never overwritten",
    )
    destination.add_argument(
        "--in-place",
        action="store_true",
        help="Append to the existing MCS files; save beside each source and atomically replace it after success",
    )
    p.add_argument("--cases", help="Comma-separated case IDs; default is the intersection of all requested Masks")
    p.add_argument("--mimics-exe")
    p.add_argument("--force", action="store_true", help="Replace existing output MCS files")
    p.add_argument("--background-mimics-lock-timeout-seconds", type=float, default=0.0)
    p.set_defaults(func=cmd_append_masks)

    p = sub.add_parser(
        "predict",
        help="Run a registered nnU-Net model over a directory of cases",
    )
    p.add_argument("--model-id", required=True,
                   help="Model ID or task ID from the nnU-Net model library")
    p.add_argument("--image-root", required=True,
                   help="Dataset-style root (<root>/<case>/) or flat folder of images")
    p.add_argument("--output-dir", required=True,
                   help="Folder for <case>.nii.gz predictions; never overwrites sources")
    p.add_argument("--workspace", default=None,
                   help="nnU-Net workspace (default: ~/.mimics_script/nnunet)")
    p.add_argument("--cases", help="Comma-separated case IDs (default: all)")
    p.add_argument("--disable-tta", action="store_true", default=True,
                   help="Disable test-time augmentation (default: disabled)")
    p.add_argument("--use-cpu", action="store_true",
                   help="Force CPU inference (default: GPU when available)")
    p.add_argument("--source-modality", default="",
                   help="Source modality hint (CT/MR) passed to the model")
    p.set_defaults(func=cmd_predict)

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
