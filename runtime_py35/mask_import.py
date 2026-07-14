# -*- coding: utf-8 -*-
"""Import mask / segmentation files into the currently open Mimics project.

Runs inside the Mimics Python process.  Calls mimics_bridge.py via subprocess
to resample masks into the active image grid, then injects .u8 buffers as
Mimics Masks.

Supports:
  - Binary masks  (0/1)  -> one Mimics Mask
  - Multi-label masks (0/1/2/...) -> one Mimics Mask per non-zero label
  - Formats: .nii.gz, .nii, .mha, .mhd, .nrrd, .nrrd.gz
"""

from __future__ import print_function

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback

import mimics

import runtime_common


TITLE = "Import Masks"

_MASK_IMPORT_MONITORS = {}

_write_json_atomic = runtime_common.write_json_atomic
_read_json = runtime_common.read_json
_find_root = runtime_common.find_root
_hidden_process_kwargs = runtime_common.hidden_process_kwargs
_background_process_kwargs = runtime_common.background_process_kwargs

_MASK_SUFFIXES = (".nii", ".nii.gz", ".mha", ".mhd", ".nrrd", ".nrrd.gz",
                  ".seg.nii", ".seg.nii.gz")

# Metadata keys written by mimics_import during .mcs creation
_MIMICS_VOXEL_TO_RAS_MATRIX_METADATA = "mimics_script.mimics_voxel_to_ras_matrix"
_SOURCE_VOXEL_TO_RAS_MATRIX_METADATA = "mimics_script.source_voxel_to_ras_matrix"


def _project_root():
    return _find_root(
        os.path.dirname(os.path.abspath(__file__)),
        ("fewshot_config.json", "nninteractive_config.json", ".git"),
    )


def _python_exe():
    """Find the nninteractive_env Python executable."""
    root = _project_root()
    candidates = [
        os.path.join(root, "nninteractive_env", "python.exe"),
        os.path.join(root, "nninteractive_env", "Scripts", "python.exe"),
        os.path.join(root, "nninteractive_env", "bin", "python3"),
        os.path.join(root, "nninteractive_env", "bin", "python"),
    ]
    env_val = os.environ.get("MIMICS_BRIDGE_PYTHON", "")
    if env_val and os.path.isfile(env_val):
        candidates.insert(0, env_val)
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise RuntimeError(
        "The nninteractive_env Python was not found. "
        "Run Setup Environment or setup_offline.bat first."
    )


def _bridge_script():
    """Find mimics_bridge.py."""
    root = _project_root()
    candidates = [
        os.path.join(root, "mimics_bridge.py"),
        os.path.join(root, "nninteractive_env", "Scripts", "mimics_bridge.py"),
    ]
    env_val = os.environ.get("MIMICS_BRIDGE_SCRIPT", "")
    if env_val and os.path.isfile(env_val):
        candidates.insert(0, env_val)
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise RuntimeError("mimics_bridge.py was not found.")


def _background_env():
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    return env


def _mask_name_from_path(path):
    """Derive a mask name from a file path, stripping common suffixes."""
    name = os.path.basename(path)
    lower = name.lower()
    if lower.endswith(".gz"):
        name = name[:-3]
        lower = name.lower()
    for suffix in (".nii", ".mha", ".mhd", ".nrrd", ".seg"):
        if lower.endswith(suffix):
            name = name[:-len(suffix)]
            break
    return name or "mask"


def _pick_mask_files(title):
    """Pick one or more mask/segmentation files (multi-select)."""
    file_filter = "Masks (*.nii *.nii.gz *.mha *.mhd *.nrrd *.nrrd.gz);;All files (*.*)"
    try:
        from PyQt5.QtWidgets import QFileDialog
        paths, _selected = QFileDialog.getOpenFileNames(None, title, "", file_filter)
        return [str(p) for p in paths if p]
    except Exception:
        pass
    try:
        import tkinter as tk
        from tkinter import filedialog as tkFileDialog
    except ImportError:
        return []
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    paths = tkFileDialog.askopenfilenames(
        parent=root,
        title=title,
        filetypes=[("Masks", "*.nii *.nii.gz *.mha *.mhd *.nrrd *.nrrd.gz"),
                    ("All files", "*.*")],
    )
    root.destroy()
    return list(paths) if paths else []


def _active_image_info():
    """Get the active Mimics image shape and voxel-to-RAS matrix."""
    image = None
    try:
        image = mimics.data.images.get_active()
    except Exception:
        pass
    # Fallback: if get_active() returned None, try the first image
    if image is None:
        try:
            if len(mimics.data.images) > 0:
                image = mimics.data.images[0]
                try:
                    mimics.data.images.set_active(image)
                except Exception:
                    pass
        except Exception:
            pass
    if image is None:
        return None, None, None
    # Get shape using the same approach as fewshot_mimics.py
    shape = _active_image_shape(image)
    if shape is None:
        return None, None, None
    # Derive voxel-to-RAS matrix from the image
    matrix = _derive_mimics_voxel_to_ras_matrix(image, shape)
    # Fallback: try reading from image metadata (written during import)
    if matrix is None:
        matrix = _matrix_from_metadata(image)
    return image, shape, matrix


def _active_image_shape(image):
    """Get image shape using the same approach as fewshot_mimics.py."""
    try:
        dims = getattr(image, "logical_dimensions", None)
        if dims is not None:
            shape = [int(dims[0]), int(dims[1]), int(dims[2])]
            if all(value > 0 for value in shape):
                return shape
    except Exception:
        pass
    try:
        view = image.get_voxel_buffer()
        shape = [int(value) for value in view.shape]
        if all(value > 0 for value in shape):
            return shape
    except Exception:
        pass
    return None


def _metadata_get(image, name, default=None):
    """Read a metadata value from a Mimics image object."""
    try:
        obj = image.metadata
        return obj[name].value
    except Exception:
        return default


def _parse_matrix_metadata(value):
    """Parse a JSON-encoded 4x4 matrix from metadata."""
    if not value:
        return None
    try:
        parsed = json.loads(str(value))
        if isinstance(parsed, list) and len(parsed) == 4:
            return parsed
    except Exception:
        pass
    return None


def _matrix_from_metadata(image):
    """Try to read the voxel-to-RAS matrix from image metadata.

    This is a fallback when get_voxel_center() is not available.
    The metadata is written by mimics_import during .mcs creation.
    """
    for key in (_MIMICS_VOXEL_TO_RAS_MATRIX_METADATA,
                _SOURCE_VOXEL_TO_RAS_MATRIX_METADATA):
        raw = _metadata_get(image, key, "")
        matrix = _parse_matrix_metadata(raw)
        if matrix is not None:
            return matrix
    return None


def _point_values(point):
    """Extract (x, y, z) from a Mimics point object."""
    if point is None:
        raise RuntimeError("Mimics returned an empty voxel center.")
    for names in (("x", "y", "z"), ("X", "Y", "Z")):
        try:
            return [float(getattr(point, names[0])),
                    float(getattr(point, names[1])),
                    float(getattr(point, names[2]))]
        except Exception:
            pass
    return [float(point[0]), float(point[1]), float(point[2])]


def _voxel_center(image, index):
    """Get the world-space center of a voxel using Mimics API."""
    getter = getattr(image, "get_voxel_center", None)
    if not callable(getter):
        raise RuntimeError("Mimics image does not expose get_voxel_center().")
    values = [int(value) for value in index]
    try:
        return _point_values(getter(values))
    except TypeError:
        pass
    try:
        return _point_values(getter(tuple(values)))
    except TypeError:
        return _point_values(getter(values[0], values[1], values[2]))


def _derive_mimics_voxel_to_ras_matrix(image, shape):
    """Derive a 4x4 voxel-to-RAS matrix from a Mimics image object.

    Uses the same approach as fewshot_mimics.py: measure voxel centers
    along each axis to build the affine, converting Mimics LPS to RAS.
    """
    matrix = [[0.0] * 4 for _ in range(4)]
    matrix[3] = [0.0, 0.0, 0.0, 1.0]

    try:
        # Mimics uses LPS internally; get_voxel_center returns LPS.
        # Convert LPS to RAS by negating the first two world axes.
        origin_lps = _voxel_center(image, [0, 0, 0])
        origin = [-origin_lps[0], -origin_lps[1], origin_lps[2]]

        for axis in range(3):
            step = [0, 0, 0]
            step[axis] = 1
            step_lps = _voxel_center(image, step)
            step_ras = [-step_lps[0], -step_lps[1], step_lps[2]]
            for row in range(3):
                matrix[row][axis] = step_ras[row] - origin[row]
            matrix[axis][3] = origin[axis]

        return matrix
    except Exception:
        return None


def _call_bridge(params, monitor=None):
    """Call mimics_bridge.py and return the parsed JSON result."""
    if monitor is not None and monitor.get("done"):
        raise RuntimeError("Mask import was cancelled before bridge startup.")
    python_exe = _python_exe()
    bridge = _bridge_script()
    process = subprocess.Popen(
        [python_exe, bridge],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_background_env(),
        **_hidden_process_kwargs()
    )
    if monitor is not None:
        monitor["process"] = process
        if monitor.get("done"):
            runtime_common.terminate_process_async(process=process, graceful_seconds=0.0)
            raise RuntimeError("Mask import was cancelled during bridge startup.")
    stdout, stderr = process.communicate(
        input=json.dumps(params).encode("utf-8")
    )
    if process.returncode != 0:
        raise RuntimeError(
            "Bridge failed (exit {0}): {1}".format(
                process.returncode,
                stderr.decode("utf-8", "replace")[:2000],
            )
        )
    try:
        result = json.loads(stdout.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError("Bridge returned invalid JSON: {0}".format(exc))
    if result.get("status") != "ok":
        raise RuntimeError(
            "Bridge error: {0}".format(result.get("error", "unknown"))
        )
    return result


def _inject_buffer(mask, buffer_path, mimics_shape):
    """Inject a .u8 buffer into a Mimics mask (binary)."""
    with open(buffer_path, "rb") as f:
        raw = f.read()
    expected = 1
    for dim in mimics_shape:
        expected *= int(dim)
    if len(raw) != expected:
        raise RuntimeError(
            "Buffer size mismatch: {0} != {1}".format(len(raw), expected)
        )
    try:
        import numpy as np
        pixels = np.frombuffer(raw, dtype=np.uint8).reshape(
            tuple(mimics_shape)
        ).astype(np.bool_)
        mask.set_voxel_buffer(pixels)
    except ImportError:
        view = memoryview(bytearray(raw)).cast("?", shape=list(mimics_shape))
        mask.set_voxel_buffer(view)


def _find_or_create_mask(name, active_image):
    """Find an existing mask by name, or create a new one."""
    for mask in mimics.data.masks:
        if str(getattr(mask, "name", "")) != name:
            continue
        try:
            if getattr(mask, "image", None) not in (None, active_image):
                continue
        except Exception:
            pass
        return mask
    mask = mimics.segment.create_mask()
    mask.name = name
    try:
        mask.image = active_image
    except Exception:
        pass
    return mask


def _update_gui():
    try:
        mimics.update_gui()
    except Exception:
        pass


def _safe_message(message, title=TITLE):
    try:
        mimics.dialogs.message_box(message, title=title, ui_blocking=False)
    except TypeError:
        try:
            mimics.dialogs.message_box(message, title=title)
        except Exception:
            print("{0}: {1}".format(title, message))
    except Exception:
        print("{0}: {1}".format(title, message))


def _stop_mask_import_monitor(key):
    monitor = _MASK_IMPORT_MONITORS.pop(key, None)
    if not monitor:
        return
    timer = monitor.get("timer")
    if timer is not None:
        try:
            timer.stop()
        except Exception:
            pass
    win32_timer = monitor.get("win32_timer")
    if win32_timer:
        try:
            user32, timer_id = win32_timer
            user32.KillTimer(None, timer_id)
        except Exception:
            pass


def _finish_mask_import(monitor):
    key = monitor.get("monitor_key")
    created_names = monitor.get("created_names", [])
    errors = monitor.get("errors", [])
    _stop_mask_import_monitor(key)
    _cleanup_work_dir(monitor.get("work_dir"))
    if created_names:
        message = "Imported {0} mask(s):\n\n{1}".format(
            len(created_names), "\n".join("- " + name for name in created_names)
        )
        if errors:
            message += "\n\nWarnings:\n" + "\n".join("- " + error for error in errors)
        _safe_message(message)
    else:
        _safe_message(
            "No visible masks were imported.\n\n{0}".format(
                "\n".join(errors) if errors else "The selected files produced no foreground labels."
            )
        )


def _mask_import_monitor_tick(monitor):
    if monitor.get("busy"):
        return
    monitor["busy"] = True
    try:
        _mask_import_monitor_tick_locked(monitor)
    finally:
        monitor["busy"] = False


def _mask_import_monitor_tick_locked(monitor):
    if time.time() > monitor.get("deadline", 0):
        runtime_common.terminate_process_async(
            process=monitor.get("process"),
            graceful_seconds=2.0,
        )
        monitor.setdefault("errors", []).append("Background mask preparation timed out.")
        _finish_mask_import(monitor)
        return

    if monitor.get("pending") is None:
        result = _read_json(monitor.get("result_path"), None)
        if result is None or result.get("status") == "running":
            return
        if result.get("status") != "ok":
            monitor.setdefault("errors", []).append(
                "Mask preparation failed: {0}".format(result.get("error", "unknown error"))
            )
            _finish_mask_import(monitor)
            return
        monitor["pending"] = list(result.get("masks", []))
        if not monitor["pending"]:
            monitor.setdefault("errors", []).append("The selected files produced no labels.")
            _finish_mask_import(monitor)
            return

    pending = monitor.get("pending") or []
    if not pending:
        _finish_mask_import(monitor)
        return

    # Apply one mask per GUI timer tick. Reading/resampling stays external and
    # Mimics can repaint between unavoidable set_voxel_buffer API calls.
    item = pending.pop(0)
    monitor["pending"] = pending
    name = str(item.get("name", "mask") or "mask")
    foreground = int(item.get("foreground_voxels", 0) or 0)
    if foreground <= 0:
        monitor.setdefault("errors", []).append(
            "Mask '{0}' was empty after spatial alignment and was not created.".format(name)
        )
    else:
        path = item.get("u8_path", "")
        shape = item.get("mimics_shape") or monitor.get("image_shape")
        try:
            _update_gui()
            mask = _find_or_create_mask(name, monitor.get("active_image"))
            _inject_buffer(mask, path, shape)
            mask.visible = True
            mask.selected = True
            monitor.setdefault("created_names", []).append(name)
        except Exception as exc:
            monitor.setdefault("errors", []).append("Mask '{0}': {1}".format(name, exc))
        _update_gui()

    if not pending:
        _finish_mask_import(monitor)


def _start_win32_mask_import_monitor(monitor, poll_seconds):
    if os.name != "nt":
        return False
    try:
        import ctypes
        user32 = ctypes.windll.user32
        TIMERPROC = ctypes.WINFUNCTYPE(
            None, ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_uint
        )

        def _timer_proc(hwnd, message, timer_id, tick_count):
            try:
                _mask_import_monitor_tick(monitor)
            except Exception as exc:
                monitor.setdefault("errors", []).append(
                    "Mask apply monitor failed: {0}".format(exc)
                )
                _finish_mask_import(monitor)

        callback = TIMERPROC(_timer_proc)
        user32.SetTimer.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, TIMERPROC]
        user32.SetTimer.restype = ctypes.c_size_t
        user32.KillTimer.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        timer_id = user32.SetTimer(None, 0, max(100, int(poll_seconds * 1000)), callback)
        if not timer_id:
            return False
        monitor["callback"] = callback
        monitor["win32_timer"] = (user32, timer_id)
        _MASK_IMPORT_MONITORS[monitor["monitor_key"]] = monitor
        return True
    except Exception:
        return False


def _start_mask_import_monitor(monitor, poll_seconds=0.25):
    # On Windows, use the native message-loop timer first. Importing a second
    # Qt binding inside Mimics can itself create a noticeable frozen interval.
    if _start_win32_mask_import_monitor(monitor, poll_seconds):
        return True
    try:
        from PyQt5.QtCore import QTimer
        from PyQt5.QtWidgets import QApplication
        if QApplication.instance() is None:
            return False
        timer = QTimer()
        monitor["timer"] = timer
        _MASK_IMPORT_MONITORS[monitor["monitor_key"]] = monitor
        def _tick():
            try:
                _mask_import_monitor_tick(monitor)
            except Exception as exc:
                monitor.setdefault("errors", []).append(
                    "Mask apply monitor failed: {0}".format(exc)
                )
                _finish_mask_import(monitor)
        timer.timeout.connect(_tick)
        monitor["callback"] = _tick
        timer.start(max(100, int(poll_seconds * 1000)))
        return True
    except Exception:
        return False


def _launch_mask_prepare(bridge_params, result_path, monitor):
    _write_json_atomic(result_path, {"status": "running"})

    def _run():
        try:
            result = _call_bridge(bridge_params, monitor=monitor)
        except Exception as exc:
            result = {"status": "error", "error": str(exc), "traceback": traceback.format_exc()}
        try:
            _write_json_atomic(result_path, result)
        except Exception:
            pass

    thread = threading.Thread(target=_run)
    thread.daemon = True
    thread.start()
    return thread


def main():
    """Entry point: import mask files into the current Mimics project."""
    active_monitors = [item for item in _MASK_IMPORT_MONITORS.values() if item and not item.get("done")]
    if active_monitors:
        answer = mimics.dialogs.question_box(
            message=(
                "A Mask import is already preparing or applying labels.\n\n"
                "Keep it running, or stop it before starting a different import."
            ),
            buttons="Keep Running;Stop Current Import",
            title=TITLE,
            ui_blocking=True,
        )
        if answer == "Stop Current Import":
            for active in active_monitors:
                active["done"] = True
                _stop_mask_import_monitor(active.get("monitor_key"))
                runtime_common.terminate_process_async(
                    process=active.get("process"),
                    graceful_seconds=2.0,
                    on_complete=lambda path=active.get("work_dir"): _cleanup_work_dir(path),
                )
            _safe_message("Mask import stop requested. No additional Masks will be applied.")
        return 0
    # 1. Check that an image is open
    active_image, image_shape, voxel_to_ras = _active_image_info()
    if active_image is None:
        # Provide a more specific diagnostic
        image_count = 0
        try:
            image_count = len(mimics.data.images)
        except Exception:
            pass
        if image_count > 0:
            msg = (
                "Mimics has {0} image(s) open but none is active, and the "
                "image shape could not be determined.\n\n"
                "Try clicking on the image in Mimics to activate it, then "
                "run Import Masks again.".format(image_count)
            )
        else:
            msg = (
                "No image is open in Mimics.\n\n"
                "Open a project first, then use Import Masks to add "
                "segmentations."
            )
        mimics.dialogs.message_box(msg, title=TITLE, ui_blocking=True)
        return 1

    # 2. Pick mask files
    mask_paths = _pick_mask_files(
        "Select mask / segmentation files to import"
    )
    if not mask_paths:
        return 0

    # Validate files exist
    valid_paths = []
    for p in mask_paths:
        if os.path.isfile(p):
            valid_paths.append(p)
        else:
            mimics.dialogs.message_box(
                "File not found: {0}".format(p),
                title=TITLE,
                ui_blocking=True,
            )
    if not valid_paths:
        return 1

    # 3. Build bridge params and call resample
    work_dir = tempfile.mkdtemp(prefix="mask_import_")
    buffers_dir = os.path.join(work_dir, "buffers")
    os.makedirs(buffers_dir, exist_ok=True)

    masks_param = []
    for p in valid_paths:
        masks_param.append({
            "name": _mask_name_from_path(p),
            "mask_path": os.path.abspath(p),
        })

    # Use the bridge's prepare_masks_for_grid action to resample masks
    # into the actual Mimics image grid.
    bridge_params = {
        "action": "prepare_masks_for_grid",
        "masks": masks_param,
        "target_shape": image_shape,
        "target_voxel_to_ras_matrix": voxel_to_ras or [],
        "buffers_out": buffers_dir,
        "axes": [0, 1, 2],
        "flips": [False, False, False],
    }

    result_path = os.path.join(work_dir, "bridge_result.json")
    monitor = {
        "monitor_key": work_dir,
        "result_path": result_path,
        "work_dir": work_dir,
        "active_image": active_image,
        "image_shape": image_shape,
        "pending": None,
        "created_names": [],
        "errors": [],
        "deadline": time.time() + 1800,
    }
    if not _start_mask_import_monitor(monitor):
        _cleanup_work_dir(work_dir)
        _safe_message("This Mimics session cannot monitor background mask preparation.")
        return 1
    _update_gui()
    _launch_mask_prepare(bridge_params, result_path, monitor)
    try:
        mimics.logging.log_user_message(
            mimics.logging.Level.INFO,
            "Mask preparation started in external Python. Mimics remains available.",
        )
    except Exception:
        print("Mask preparation started in external Python. Mimics remains available.")
    return 0


def _cleanup_work_dir(work_dir):
    """Remove temporary work directory."""
    if not work_dir or not os.path.isdir(work_dir):
        return
    try:
        import shutil
        shutil.rmtree(work_dir, ignore_errors=True)
    except Exception:
        pass


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:
        traceback.print_exc()
        try:
            mimics.dialogs.message_box(
                title=TITLE,
                message="Error: {0}".format(exc),
            )
        except Exception:
            pass
        raise
