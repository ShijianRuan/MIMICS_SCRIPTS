# -*- coding: utf-8 -*-
"""Background Mimics script to create .mcs files.

Runs inside Mimics Python 3.5.2 in background mode (MimicsResearch.exe -b).
Reads prepare manifests from work directories and creates .mcs files one by one.

Usage (from Mimics command line):
    MimicsResearch.exe -b -run_script create_mcs_batch.py <output_dir>

Or called programmatically:
    create_mcs_batch.main(output_dir)
"""

from __future__ import print_function

import json
import os
import shutil
import subprocess
import sys
import time
import traceback

import mimics

import runtime_common


QUEUE_ACTIVE_FILE = "_mcs_queue_active.json"
QUEUE_DONE_FILE = "_mcs_queue_done.json"
QUEUE_STOP_FILE = "_mcs_queue_stop.json"
STATUS_FILE = "_mcs_batch_status.json"
LOCK_FILE = "_mcs_batch.lock"
LOG_FILE = os.path.join("logs", "_create_mcs_batch.log")
LOG_ROTATE_BYTES = 5 * 1024 * 1024
LOG_BACKUPS = 3
SOURCE_IMAGE_PATH_METADATA = "mimics_script.source_image_path"
SOURCE_IMAGE_KIND_METADATA = "mimics_script.source_image_kind"
SOURCE_IMAGE_SHAPE_METADATA = "mimics_script.source_image_shape"
SOURCE_IMAGE_INDEX_SPACE_METADATA = "mimics_script.source_image_index_space"
SOURCE_IMAGE_MODALITY_METADATA = "mimics_script.source_image_modality"
SOURCE_WORLD_COORDINATE_SYSTEM_METADATA = "mimics_script.source_world_coordinate_system"
MIMICS_WORLD_COORDINATE_SYSTEM_METADATA = "mimics_script.mimics_world_coordinate_system"
SOURCE_TO_MIMICS_WORLD_MATRIX_METADATA = "mimics_script.source_to_mimics_world_matrix"
SOURCE_VOXEL_TO_RAS_MATRIX_METADATA = "mimics_script.source_voxel_to_ras_matrix"
MIMICS_VOXEL_TO_RAS_MATRIX_METADATA = "mimics_script.mimics_voxel_to_ras_matrix"
MIMICS_TO_SOURCE_INDEX_MATRIX_METADATA = "mimics_script.mimics_to_source_index_matrix"
SOURCE_CASE_DIR_METADATA = "mimics_script.source_case_dir"
write_json_atomic = runtime_common.write_json_atomic
safe_case_filename = runtime_common.safe_filename


def rotate_log(path, max_bytes=LOG_ROTATE_BYTES, backups=LOG_BACKUPS):
    try:
        if not os.path.isfile(path) or os.path.getsize(path) < max_bytes:
            return
        backups = int(backups)
        if backups <= 0:
            os.remove(path)
            return
        oldest = "{0}.{1}".format(path, backups)
        if os.path.isfile(oldest):
            os.remove(oldest)
        for index in range(backups - 1, 0, -1):
            src = "{0}.{1}".format(path, index)
            dst = "{0}.{1}".format(path, index + 1)
            if os.path.isfile(src):
                os.rename(src, dst)
        os.rename(path, path + ".1")
    except Exception:
        pass


def log_message(output_dir, message):
    text = "[{0}] {1}".format(time.strftime("%Y-%m-%d %H:%M:%S"), message)
    print(text)
    try:
        log_path = os.path.join(output_dir, LOG_FILE)
        log_dir = os.path.dirname(log_path)
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        rotate_log(log_path)
        with open(log_path, "a") as handle:
            handle.write(text + "\n")
    except Exception:
        pass


def record_failed_case(output_dir, case_id, phase, error, traceback_text=None):
    try:
        failed_dir = os.path.join(output_dir or os.getcwd(), "_failed_cases")
        if not os.path.isdir(failed_dir):
            os.makedirs(failed_dir)
        payload = {
            "case_id": str(case_id or "unknown"),
            "phase": str(phase or "unknown"),
            "error": str(error or ""),
            "failed_at_epoch": time.time(),
        }
        if traceback_text:
            payload["traceback"] = traceback_text
        filename = "{0}_{1}.json".format(
            safe_case_filename(case_id),
            safe_case_filename(phase),
        )
        write_json_atomic(os.path.join(failed_dir, filename), payload)
    except Exception:
        pass


def update_status(output_dir, status, **details):
    payload = {
        "status": status,
        "pid": os.getpid(),
        "updated_at_epoch": time.time(),
    }
    payload.update(details)
    try:
        write_json_atomic(os.path.join(output_dir, STATUS_FILE), payload)
    except Exception:
        pass


def acquire_lock(output_dir):
    lock_path = os.path.join(output_dir, LOCK_FILE)
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode("ascii"))
        os.close(fd)
        return lock_path
    except OSError:
        try:
            with open(lock_path, "r") as handle:
                pid = int((handle.read() or "0").strip() or "0")
            if pid and process_exists(pid):
                return None
        except Exception:
            pass
        try:
            os.remove(lock_path)
        except Exception:
            return None
        return acquire_lock(output_dir)


def process_exists(pid):
    if not pid:
        return False
    try:
        if os.name == "nt":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
            if not handle:
                return False
            exit_code = ctypes.c_ulong(0)
            kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            kernel32.CloseHandle(handle)
            return int(exit_code.value) == STILL_ACTIVE
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def metadata_set(obj, name, value):
    text = "" if value is None else str(value)
    try:
        item = obj.metadata.find(name)
    except Exception:
        item = None
    try:
        if item is None:
            obj.metadata.create(name=name, value=text)
        else:
            item.value = text
    except Exception:
        pass


def _matrix_close(left, right, tol=1e-4):
    try:
        if not left or not right:
            return False
        if len(left) != 4 or len(right) != 4:
            return False
        for row_index in range(4):
            if len(left[row_index]) != 4 or len(right[row_index]) != 4:
                return False
            for col_index in range(4):
                if abs(float(left[row_index][col_index]) - float(right[row_index][col_index])) > tol:
                    return False
        return True
    except Exception:
        return False


def _lps_point_to_ras(point):
    return [-float(point[0]), -float(point[1]), float(point[2])]


def _point_values(point):
    if point is None:
        raise RuntimeError("Mimics returned an empty voxel center.")
    for names in (("x", "y", "z"), ("X", "Y", "Z")):
        try:
            return [float(getattr(point, names[0])), float(getattr(point, names[1])), float(getattr(point, names[2]))]
        except Exception:
            pass
    return [float(point[0]), float(point[1]), float(point[2])]


def _voxel_center(image, index):
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
        pass
    return _point_values(getter(values[0], values[1], values[2]))


def _derive_mimics_voxel_to_ras_matrix(image, image_shape, fallback):
    """Derive the actual open Mimics image voxel grid in RAS coordinates."""
    if not image_shape:
        return None
    try:
        origin = _lps_point_to_ras(_voxel_center(image, [0, 0, 0]))
        matrix = [[0.0, 0.0, 0.0, 0.0] for _ in range(4)]
        for axis in range(3):
            if int(image_shape[axis]) > 1:
                index = [0, 0, 0]
                index[axis] = 1
                point = _lps_point_to_ras(_voxel_center(image, index))
                vector = [point[i] - origin[i] for i in range(3)]
            elif fallback and len(fallback) == 4 and len(fallback[axis]) == 4:
                vector = [float(fallback[row][axis]) for row in range(3)]
            else:
                vector = [0.0, 0.0, 0.0]
                vector[axis] = 1.0
            for row in range(3):
                matrix[row][axis] = vector[row]
        for row in range(3):
            matrix[row][3] = origin[row]
        matrix[3] = [0.0, 0.0, 0.0, 1.0]
        return matrix
    except Exception:
        return None


def _bridge_python():
    return os.environ.get("MIMICS_BRIDGE_PYTHON", "")


def _bridge_script():
    return os.environ.get("MIMICS_BRIDGE_SCRIPT", "")


def _resample_masks_to_actual_grid(result, active_image_shape, actual_mimics_voxel_to_ras, work_dir):
    expected_matrix = result.get("mimics_voxel_to_ras_matrix") or []
    masks = result.get("masks") or []
    if not active_image_shape or not actual_mimics_voxel_to_ras or not masks:
        return result
    expected_shapes = []
    for row in masks:
        if row.get("mimics_shape"):
            expected_shapes.append([int(value) for value in row.get("mimics_shape")])
    shape_matches = all(shape == active_image_shape for shape in expected_shapes) if expected_shapes else False
    matrix_matches = _matrix_close(expected_matrix, actual_mimics_voxel_to_ras)
    if shape_matches and matrix_matches:
        return result

    python_exe = _bridge_python()
    bridge_script = _bridge_script()
    if not python_exe or not bridge_script or not os.path.isfile(bridge_script):
        raise RuntimeError(
            "Imported Mimics image grid differs from prepared mask grid, but the external bridge "
            "is not configured. Refusing to inject reshaped masks because that would corrupt orientation."
        )

    actual_buffers_dir = os.path.join(work_dir, "buffers_mimics_actual")
    params = {
        "action": "prepare_masks_for_grid",
        "masks": [{"name": row.get("name"), "mask_path": row.get("mask_path")} for row in masks],
        "target_shape": active_image_shape,
        "target_voxel_to_ras_matrix": actual_mimics_voxel_to_ras,
        "source_voxel_to_ras_matrix": result.get("source_voxel_to_ras_matrix") or [],
        "buffers_out": actual_buffers_dir,
        "axes": [0, 1, 2],
        "flips": [False, False, False],
    }
    process = subprocess.Popen(
        [python_exe, bridge_script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **runtime_common.background_process_kwargs()
    )
    stdout, stderr = process.communicate(input=json.dumps(params).encode("utf-8"))
    if process.returncode != 0:
        raise RuntimeError(
            "External mask resampling failed with exit code {0}: {1}".format(
                process.returncode,
                stderr.decode("utf-8", "replace")[:1000],
            )
        )
    try:
        bridge_result = json.loads(stdout.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError("External mask resampling returned invalid JSON: {0}".format(exc))
    if bridge_result.get("status") != "ok":
        raise RuntimeError("External mask resampling failed: {0}".format(bridge_result.get("error", "unknown error")))

    updated = dict(result)
    updated["masks"] = bridge_result.get("masks") or []
    updated["mimics_voxel_to_ras_matrix"] = actual_mimics_voxel_to_ras
    if bridge_result.get("mimics_to_source_index_matrix"):
        updated["mimics_to_source_index_matrix"] = bridge_result.get("mimics_to_source_index_matrix")
    updated["actual_mimics_grid_resampled"] = True
    return updated


def inject_buffer(mask, buffer_path, mimics_shape):
    """Inject a .u8 buffer into a Mimics mask."""
    with open(buffer_path, "rb") as f:
        raw = f.read()
    expected = 1
    for dim in mimics_shape:
        expected *= int(dim)
    if len(raw) != expected:
        raise RuntimeError(
            "buffer byte count mismatch: {} != {}  path={}".format(len(raw), expected, buffer_path)
        )
    try:
        import numpy as np
        pixels = np.frombuffer(raw, dtype=np.uint8).reshape(tuple(mimics_shape)).astype(np.bool_)
        mask.set_voxel_buffer(pixels)
        return "numpy"
    except ImportError:
        view = memoryview(bytearray(raw)).cast("?", shape=list(mimics_shape))
        mask.set_voxel_buffer(view)
        return "memoryview"


def _shape_product(shape):
    result = 1
    for dim in shape:
        result *= int(dim)
    return result


def _active_image_shape(image):
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


def create_mcs_from_manifest(work_dir, output_mcs):
    """Create a .mcs file from a prepare manifest.

    Steps:
        1. Import DICOM into Mimics
        2. Create masks and inject .u8 buffers
        3. Save .mcs
        4. Close project
    """
    manifest_path = os.path.join(work_dir, "prepare_manifest.json")
    if not os.path.isfile(manifest_path):
        raise RuntimeError("manifest not found: {}".format(manifest_path))

    with open(manifest_path, "r") as f:
        result = json.load(f)

    dicom_folder = result["dicom_folder"]

    # Import DICOM
    mimics.file.import_dicom_images(source_folder=dicom_folder)
    if len(mimics.data.images) == 0:
        raise RuntimeError("DICOM import produced no images")

    image = mimics.data.images[0]
    mimics.data.images.set_active(image)
    active_image_shape = _active_image_shape(image)
    actual_mimics_voxel_to_ras = _derive_mimics_voxel_to_ras_matrix(
        image,
        active_image_shape,
        result.get("mimics_voxel_to_ras_matrix") or [],
    )
    result = _resample_masks_to_actual_grid(
        result,
        active_image_shape,
        actual_mimics_voxel_to_ras,
        work_dir,
    )
    if result.get("actual_mimics_grid_resampled"):
        print("  actual Mimics image grid differs from prepared grid; masks were resampled to the open Mimics grid.")
    mask_results = result["masks"]
    metadata_set(image, SOURCE_IMAGE_PATH_METADATA, result.get("source_image_path", ""))
    metadata_set(image, SOURCE_IMAGE_KIND_METADATA, result.get("source_image_kind", ""))
    metadata_set(image, SOURCE_IMAGE_SHAPE_METADATA, json.dumps(result.get("source_image_shape", [])))
    metadata_set(image, SOURCE_IMAGE_INDEX_SPACE_METADATA, result.get("source_image_index_space", ""))
    metadata_set(image, SOURCE_IMAGE_MODALITY_METADATA, result.get("source_image_modality", ""))
    metadata_set(image, SOURCE_WORLD_COORDINATE_SYSTEM_METADATA, result.get("source_world_coordinate_system", ""))
    metadata_set(image, MIMICS_WORLD_COORDINATE_SYSTEM_METADATA, result.get("mimics_world_coordinate_system", ""))
    metadata_set(
        image,
        SOURCE_TO_MIMICS_WORLD_MATRIX_METADATA,
        json.dumps(result.get("source_to_mimics_world_matrix", [])),
    )
    metadata_set(
        image,
        SOURCE_VOXEL_TO_RAS_MATRIX_METADATA,
        json.dumps(result.get("source_voxel_to_ras_matrix", [])),
    )
    metadata_set(
        image,
        MIMICS_VOXEL_TO_RAS_MATRIX_METADATA,
        json.dumps(result.get("mimics_voxel_to_ras_matrix", [])),
    )
    metadata_set(
        image,
        MIMICS_TO_SOURCE_INDEX_MATRIX_METADATA,
        json.dumps(result.get("mimics_to_source_index_matrix", [])),
    )
    metadata_set(image, SOURCE_CASE_DIR_METADATA, result.get("source_case_dir", ""))

    # Disable GUI updates during mask creation to prevent progressive
    # display and crashes from rapid UI refreshes.
    gui_was_enabled = True
    try:
        mimics.disable_update_gui()
    except Exception:
        gui_was_enabled = False

    try:
        # Create masks and inject buffers.
        # Note: mask.image is read-only; masks are automatically linked
        # to the active image at creation time.  Do NOT set mask.image.
        # Set visible=False BEFORE injecting buffer data so Mimics never
        # shows the mask even briefly during set_voxel_buffer.
        for mr in mask_results:
            name = mr["name"]
            u8_path = mr["u8_path"]
            buf_shape = mr["mimics_shape"]
            if active_image_shape and [int(value) for value in buf_shape] != active_image_shape:
                raise RuntimeError(
                    "mask buffer shape does not match imported Mimics image for {0}: buffer={1}, image={2}. "
                    "The buffer was not reshaped because blind reshape corrupts orientation.".format(
                        name,
                        buf_shape,
                        active_image_shape,
                    )
                )

            mask = mimics.segment.create_mask()
            mask.name = name
            # Hide immediately, before any data is injected
            try:
                mask.visible = False
            except Exception:
                pass

            method = inject_buffer(mask, u8_path, buf_shape)
            print("    -> {}".format(name))
    finally:
        if gui_was_enabled:
            try:
                mimics.enable_update_gui()
            except Exception:
                pass

    # Save .mcs
    mcs_path = os.path.abspath(output_mcs)
    mcs_dir = os.path.dirname(mcs_path)
    if mcs_dir and not os.path.isdir(mcs_dir):
        os.makedirs(mcs_dir)
    mimics.file.save_project(filename=mcs_path, save_as_type="Mimics Project Files")
    print("  saved: {}".format(mcs_path))

    # Close project to free memory
    try:
        mimics.file.close_project()
    except Exception:
        pass

    # Clean up work dir (DICOM + buffers, manifest no longer needed)
    try:
        shutil.rmtree(work_dir, ignore_errors=True)
    except Exception:
        pass

    return mcs_path


def main(output_dir=None):
    """Entry point. Find all prepare manifests and create .mcs files.

    Stays alive in a loop, periodically scanning for new manifests.
    This avoids frequent Mimics restarts when manifests are produced
    one-by-one by the conversion process.  Exits after two consecutive
    scans find no new work (meaning conversion is likely done).
    """
    if output_dir is None:
        # Read from sys.argv
        if len(sys.argv) > 1:
            output_dir = sys.argv[1]
        else:
            print("Usage: create_mcs_batch.py <output_dir>")
            return 1

    if not os.path.isdir(output_dir):
        print("Output dir not found: {}".format(output_dir))
        return 1

    lock_path = acquire_lock(output_dir)
    if not lock_path:
        log_message(output_dir, "Another background Mimics batch process is already running; exiting.")
        return 0

    active_path = os.path.join(output_dir, QUEUE_ACTIVE_FILE)
    done_path = os.path.join(output_dir, QUEUE_DONE_FILE)
    idle_timeout = 1800
    poll_seconds = 10
    last_activity = time.time()
    consecutive_empty = 0
    total_completed = 0
    total_failed = 0

    try:
        log_message(output_dir, "Background Mimics batch process started.")
        update_status(output_dir, "running", completed=0, failed=0)

        while True:
            # Find all work directories with prepare manifests.
            work_dirs = []
            for item in sorted(os.listdir(output_dir)):
                if not item.endswith("_work"):
                    continue
                work_dir = os.path.join(output_dir, item)
                manifest = os.path.join(work_dir, "prepare_manifest.json")
                failed_marker = os.path.join(work_dir, "create_failed.json")
                if os.path.isfile(manifest):
                    case_id = item.replace("_work", "")
                    try:
                        with open(manifest, "r") as handle:
                            manifest_data = json.load(handle)
                    except Exception:
                        manifest_data = {}
                    mcs_path = manifest_data.get("output_mcs") or os.path.join(output_dir, case_id + ".mcs")
                    fingerprint_path = os.path.join(output_dir, case_id + ".fingerprint")
                    current_fp = manifest_data.get("source_fingerprint", "")
                    if os.path.isfile(mcs_path):
                        stored_fp = ""
                        if os.path.isfile(fingerprint_path):
                            try:
                                with open(fingerprint_path, "r") as fp_handle:
                                    stored_fp = fp_handle.read().strip()
                            except Exception:
                                pass
                        if current_fp and stored_fp == current_fp:
                            # Source unchanged — skip
                            continue
                        # Fingerprint changed or missing — reprocess
                        if current_fp:
                            log_message(output_dir, "Source fingerprint changed for {0}; reprocessing.".format(case_id))
                    if os.path.isfile(failed_marker) and not current_fp:
                        continue
                    work_dirs.append((case_id, work_dir, mcs_path))

            total = len(work_dirs)
            if total == 0:
                consecutive_empty += 1
                active = os.path.isfile(active_path)
                done = os.path.isfile(done_path)
                stop = os.path.isfile(os.path.join(output_dir, QUEUE_STOP_FILE))
                if stop:
                    log_message(output_dir, "Stop marker found; exiting.")
                    break
                if done and not active:
                    log_message(output_dir, "No pending manifests and producer marked the queue done; exiting.")
                    break
                if not active and time.time() - last_activity >= idle_timeout:
                    log_message(output_dir, "No active producer and idle timeout reached; exiting.")
                    break
                log_message(output_dir, "No pending manifests; waiting for new work.")
                update_status(
                    output_dir,
                    "idle",
                    completed=total_completed,
                    failed=total_failed,
                    consecutive_empty=consecutive_empty,
                )
                time.sleep(poll_seconds)
                continue

            consecutive_empty = 0
            log_message(output_dir, "Creating {} .mcs file(s).".format(total))

            # Re-check stop marker before starting work
            if os.path.isfile(os.path.join(output_dir, QUEUE_STOP_FILE)):
                log_message(output_dir, "Stop marker found; exiting.")
                break

            for i, (case_id, work_dir, mcs_path) in enumerate(work_dirs):
                log_message(output_dir, "[{}/{}] Creating: {}".format(i + 1, total, case_id))
                update_status(
                    output_dir,
                    "creating",
                    case_id=case_id,
                    completed=total_completed,
                    failed=total_failed,
                )
                try:
                    create_mcs_from_manifest(work_dir, mcs_path)
                    total_completed += 1
                    last_activity = time.time()
                    log_message(output_dir, "Created: {}".format(mcs_path))
                    # Persist source fingerprint for incremental rebuild detection
                    manifest_data = {}
                    try:
                        with open(os.path.join(work_dir, "prepare_manifest.json"), "r") as mf:
                            manifest_data = json.load(mf)
                    except Exception:
                        pass
                    fingerprint = manifest_data.get("source_fingerprint", "")
                    if fingerprint:
                        fingerprint_path = os.path.join(output_dir, case_id + ".fingerprint")
                        try:
                            with open(fingerprint_path, "w") as fp:
                                fp.write(fingerprint)
                        except Exception:
                            pass
                except Exception as e:
                    total_failed += 1
                    last_activity = time.time()
                    log_message(output_dir, "Create failed for {}: {}".format(case_id, e))
                    traceback_text = traceback.format_exc()
                    record_failed_case(output_dir, case_id, "create_mcs", e, traceback_text)
                    try:
                        write_json_atomic(
                            os.path.join(work_dir, "create_failed.json"),
                            {
                                "case_id": case_id,
                                "error": str(e),
                                "traceback": traceback_text,
                                "failed_at_epoch": time.time(),
                            },
                        )
                    except Exception:
                        pass
                    print(traceback_text)
                    try:
                        mimics.file.close_project()
                    except Exception:
                        pass
                    try:
                        shutil.rmtree(work_dir, ignore_errors=True)
                    except Exception:
                        pass

        log_message(
            output_dir,
            "Background creation finished: {} succeeded, {} failed.".format(
                total_completed,
                total_failed,
            ),
        )
        update_status(output_dir, "closed", completed=total_completed, failed=total_failed)
        return 0
    finally:
        try:
            os.remove(lock_path)
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as error:
        traceback.print_exc()
        raise
