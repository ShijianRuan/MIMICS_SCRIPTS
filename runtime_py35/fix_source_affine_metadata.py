# -*- coding: utf-8 -*-
"""One-shot repair for stale source_voxel_to_ras_matrix image metadata.

Problem
-------
``source_voxel_to_ras_matrix`` is, by definition, the voxel-to-RAS affine of
the source NIfTI on disk (the file named by ``source_image_path`` metadata).
Inference geometry validation compares this stored value against
``nibabel.load(source_path).affine`` and rejects the run when they differ.

Older versions of mimics_bridge.py (before 2026-07-13) stored SimpleITK's
LPS-direction / DICOMOrient("LPS") affine into this field instead of the true
nibabel RAS affine. For cases prepared with that old bridge, the stored value
is wrong by a full axis flip (max abs diff ~ image width in mm), so inference
fails the geometry check even though the on-disk NIfTI is fine.

This entry rewrites the field to the true nibabel RAS affine. It is safe
because the field's defined value IS the nibabel RAS affine, and because mask
alignment during inference does not consume this field (it uses the live
Mimics grid measured at apply time).

Implementation note: Mimics' embedded Python (3.5.2) has no numpy/nibabel,
so the nibabel read is delegated to the external bridge subprocess (which runs
the nninteractive_env Python). This module only reads/writes Mimics metadata
and orchestrates that subprocess.

Run inside Mimics with the target project open and the source image active.
"""

from __future__ import print_function

import json
import os
import subprocess
import sys
import threading
import time

import mimics
import runtime_common

SOURCE_IMAGE_PATH_METADATA = "mimics_script.source_image_path"
SOURCE_IMAGE_SHAPE_METADATA = "mimics_script.source_image_shape"
SOURCE_VOXEL_TO_RAS_MATRIX_METADATA = "mimics_script.source_voxel_to_ras_matrix"

TITLE = "Fix Source Affine Metadata"
_MONITORS = {}


def _project_root():
    return runtime_common.find_root(
        os.path.dirname(os.path.abspath(__file__)),
        ("nninteractive_config.json", "fewshot_config.json", "mimics_bridge.py", ".git"),
    )


def _find_external_python():
    root = _project_root()
    for rel in (
        "nninteractive_env/python.exe",
        "nninteractive_env/Scripts/python.exe",
        "nninteractive_env/python/python.exe",
        "python/python.exe",
    ):
        path = os.path.join(root, rel)
        if os.path.isfile(path):
            return path
    return ""


def _bridge_script():
    return os.path.join(_project_root(), "mimics_bridge.py")


def _metadata_get(image, name, default=""):
    try:
        item = image.metadata.find(name)
    except Exception:
        return default
    if item is None:
        return default
    try:
        return getattr(item, "value", default)
    except Exception:
        return default


def _metadata_set(image, name, value):
    text = "" if value is None else str(value)
    try:
        item = image.metadata.find(name)
    except Exception:
        item = None
    if item is None:
        image.metadata.create(name=name, value=text)
    else:
        item.value = text


def _parse_matrix(text):
    if not text:
        return None
    try:
        m = json.loads(text)
    except Exception:
        return None
    try:
        if len(m) != 4 or any(len(row) != 4 for row in m):
            return None
        flat = [float(v) for row in m for v in row]
    except Exception:
        return None
    if not all(_is_finite(v) for v in flat):
        return None
    return [flat[0:4], flat[4:8], flat[8:12], flat[12:16]]


def _is_finite(value):
    try:
        return value == value and value not in (float("inf"), float("-inf"))
    except Exception:
        return False


def _matrices_equal(left, right, tol=1e-4):
    if left is None or right is None:
        return left is right
    if len(left) != 4 or len(right) != 4:
        return False
    for i in range(4):
        if len(left[i]) != 4 or len(right[i]) != 4:
            return False
        for j in range(4):
            if abs(float(left[i][j]) - float(right[i][j])) > tol:
                return False
    return True


def _max_abs_diff(left, right):
    best = 0.0
    for i in range(4):
        for j in range(4):
            best = max(best, abs(float(left[i][j]) - float(right[i][j])))
    return best


def _parse_shape(text):
    if not text:
        return None
    try:
        shape = json.loads(text)
    except Exception:
        return None
    try:
        return [int(v) for v in shape]
    except Exception:
        return None


def _read_nifti_ras_affine(image_path, monitor=None):
    """Call the bridge subprocess to read the nibabel RAS affine + shape."""
    python_exe = _find_external_python()
    script = _bridge_script()
    if not python_exe:
        raise RuntimeError(
            "Could not find the external Python interpreter (nninteractive_env) "
            "needed to read the NIfTI affine."
        )
    if not os.path.isfile(script):
        raise RuntimeError("Could not find mimics_bridge.py at: {0}".format(script))
    params = {"action": "read_nifti_ras_affine", "image_path": image_path}
    process = subprocess.Popen(
        [python_exe, script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **runtime_common.hidden_process_kwargs()
    )
    if monitor is not None:
        monitor["process"] = process
        if monitor.get("done"):
            runtime_common.terminate_process_async(process=process, graceful_seconds=0.0)
            raise RuntimeError("Affine repair was cancelled during bridge startup.")
    try:
        stdout, stderr = process.communicate(
            input=json.dumps(params).encode("utf-8"),
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise RuntimeError("Source-affine bridge timed out after 600 seconds.")
    if process.returncode != 0:
        raise RuntimeError(
            "Bridge exited with code {0}: {1}".format(
                process.returncode, stderr.decode("utf-8", "replace")[:1000]
            )
        )
    try:
        result = json.loads(stdout.decode("utf-8", "replace"))
    except Exception as exc:
        raise RuntimeError("Bridge returned invalid JSON: {0}".format(exc))
    if result.get("status") != "ok":
        raise RuntimeError(result.get("error", "bridge returned an error"))
    return result


def _stop_monitor(key):
    monitor = _MONITORS.pop(key, None)
    if not monitor:
        return
    timer = monitor.get("timer")
    if timer is not None:
        try:
            timer.stop()
        except Exception:
            pass
    win32 = monitor.get("win32_timer")
    if win32:
        try:
            win32[0].KillTimer(None, win32[1])
        except Exception:
            pass


def _finish_repair(monitor):
    image = monitor["image"]
    stored = monitor["stored"]
    stored_shape = monitor.get("stored_shape")
    info = monitor.get("result") or {}
    true_ras = info.get("voxel_to_ras_matrix")
    true_shape = [int(v) for v in (info.get("shape") or [])]
    if not true_ras or len(true_shape) != 3:
        raise RuntimeError("Bridge returned incomplete affine geometry.")
    if _matrices_equal(stored, true_ras):
        mimics.dialogs.message_box(
            title=TITLE,
            message="Metadata already matches the source image affine. Nothing was changed.",
            ui_blocking=False,
        )
        return
    if stored_shape and list(stored_shape) != list(true_shape):
        raise RuntimeError(
            "Stored source image shape {0} does not match the on-disk shape {1}. "
            "The source file changed; re-import this case instead.".format(stored_shape, true_shape)
        )
    _metadata_set(image, SOURCE_VOXEL_TO_RAS_MATRIX_METADATA, json.dumps(true_ras))
    written = _parse_matrix(_metadata_get(image, SOURCE_VOXEL_TO_RAS_MATRIX_METADATA, ""))
    if not _matrices_equal(written, true_ras):
        raise RuntimeError("Metadata write did not round-trip correctly; the project was not saved.")
    mimics.file.save_project()
    diff = _max_abs_diff(stored, true_ras)
    mimics.dialogs.message_box(
        title=TITLE,
        message=(
            "Source affine metadata was repaired and the project was saved.\n\n"
            "Maximum matrix correction: {0:.6g}"
        ).format(diff),
        ui_blocking=False,
    )


def _monitor_tick(monitor):
    if monitor.get("done") or monitor.get("busy"):
        return
    monitor["busy"] = True
    try:
        if time.time() > monitor.get("deadline", 0):
            monitor["done"] = True
            _stop_monitor(monitor["key"])
            runtime_common.terminate_process_async(process=monitor.get("process"), graceful_seconds=2.0)
            mimics.dialogs.message_box(
                title=TITLE,
                message="Affine inspection timed out and its external process was stopped. No metadata was changed.",
                ui_blocking=False,
            )
            return
        if not monitor.get("worker_done"):
            return
        monitor["done"] = True
        _stop_monitor(monitor["key"])
        if monitor.get("error"):
            raise RuntimeError(monitor.get("error"))
        _finish_repair(monitor)
    except Exception as exc:
        monitor["done"] = True
        _stop_monitor(monitor.get("key"))
        mimics.dialogs.message_box(
            title=TITLE,
            message="Source affine metadata was not changed.\n\n{0}".format(exc),
            ui_blocking=False,
        )
    finally:
        monitor["busy"] = False


def _start_monitor(monitor, poll_seconds=0.5):
    key = monitor["key"]
    _stop_monitor(key)
    if os.name == "nt":
        try:
            import ctypes
            user32 = ctypes.windll.user32
            TIMERPROC = ctypes.WINFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_uint)

            def _timer_proc(hwnd, message, timer_id, tick_count):
                _monitor_tick(monitor)

            callback = TIMERPROC(_timer_proc)
            user32.SetTimer.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, TIMERPROC]
            user32.SetTimer.restype = ctypes.c_size_t
            user32.KillTimer.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            timer_id = user32.SetTimer(None, 0, max(250, int(poll_seconds * 1000)), callback)
            if timer_id:
                monitor["callback"] = callback
                monitor["win32_timer"] = (user32, timer_id)
                _MONITORS[key] = monitor
                return True
        except Exception:
            pass
    try:
        from PyQt5.QtCore import QTimer
        timer = QTimer()
        timer.timeout.connect(lambda: _monitor_tick(monitor))
        timer.start(max(250, int(poll_seconds * 1000)))
        monitor["timer"] = timer
        _MONITORS[key] = monitor
        return True
    except Exception:
        return False


def main():
    active = [item for item in _MONITORS.values() if item and not item.get("done")]
    if active:
        answer = mimics.dialogs.question_box(
            title=TITLE,
            message="Source affine inspection is already running.",
            buttons="Keep Running;Stop Inspection",
            ui_blocking=True,
        )
        if answer == "Stop Inspection":
            for monitor in active:
                monitor["done"] = True
                _stop_monitor(monitor.get("key"))
                runtime_common.terminate_process_async(process=monitor.get("process"), graceful_seconds=2.0)
        return 0

    image = mimics.data.images.get_active()
    if image is None:
        mimics.dialogs.message_box(
            title=TITLE,
            message="No active image.\n\nOpen the target project and activate the source image first.",
        )
        return 1

    source_path = str(_metadata_get(image, SOURCE_IMAGE_PATH_METADATA, "") or "").strip()
    if not source_path or not os.path.isfile(source_path):
        mimics.dialogs.message_box(
            title=TITLE,
            message="No usable source_image_path metadata, or the file no longer exists:\n{0!r}".format(source_path),
        )
        return 1

    stored_shape = _parse_shape(_metadata_get(image, SOURCE_IMAGE_SHAPE_METADATA, ""))
    stored = _parse_matrix(_metadata_get(image, SOURCE_VOXEL_TO_RAS_MATRIX_METADATA, ""))
    if stored is None:
        mimics.dialogs.message_box(
            title=TITLE,
            message="No existing source_voxel_to_ras_matrix metadata on this image; nothing to repair.",
        )
        return 1

    monitor = {
        "key": "source_affine_" + str(int(time.time() * 1000)),
        "image": image,
        "source_path": source_path,
        "stored": stored,
        "stored_shape": stored_shape,
        "result": None,
        "error": "",
        "worker_done": False,
        "done": False,
        "deadline": time.time() + 120.0,
    }
    if not _start_monitor(monitor):
        mimics.dialogs.message_box(
            title=TITLE,
            message="This Mimics session cannot monitor the external affine inspection. No metadata was changed.",
            ui_blocking=False,
        )
        return 1

    def _worker():
        try:
            monitor["result"] = _read_nifti_ras_affine(source_path, monitor=monitor)
        except Exception as exc:
            monitor["error"] = str(exc)
        finally:
            monitor["worker_done"] = True

    thread = threading.Thread(target=_worker, name="SourceAffineInspector")
    thread.daemon = True
    monitor["thread"] = thread
    thread.start()
    try:
        mimics.logging.log_user_message(
            level=mimics.logging.Level.INFO,
            message="Source affine inspection started in external Python. Mimics remains available.",
        )
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
