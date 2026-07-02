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
import sys
import time
import traceback

import mimics

import runtime_common


QUEUE_ACTIVE_FILE = "_mcs_queue_active.json"
QUEUE_DONE_FILE = "_mcs_queue_done.json"
STATUS_FILE = "_mcs_batch_status.json"
LOCK_FILE = "_mcs_batch.lock"
LOG_FILE = "_create_mcs_batch.log"
LOG_ROTATE_BYTES = 5 * 1024 * 1024
LOG_BACKUPS = 3
write_json_atomic = runtime_common.write_json_atomic
safe_case_filename = runtime_common.safe_filename


def rotate_log(path, max_bytes=LOG_ROTATE_BYTES, backups=LOG_BACKUPS):
    try:
        if not os.path.isfile(path) or os.path.getsize(path) < max_bytes:
            return
        for index in range(int(backups), 0, -1):
            src = "{0}.{1}".format(path, index)
            dst = "{0}.{1}".format(path, index + 1)
            if os.path.isfile(dst):
                os.remove(dst)
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
    mask_results = result["masks"]

    # Import DICOM
    mimics.file.import_dicom_images(source_folder=dicom_folder)
    if len(mimics.data.images) == 0:
        raise RuntimeError("DICOM import produced no images")

    image = mimics.data.images[0]
    mimics.data.images.set_active(image)

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
                    if os.path.isfile(mcs_path) or os.path.isfile(failed_marker):
                        continue
                    work_dirs.append((case_id, work_dir, mcs_path))

            total = len(work_dirs)
            if total == 0:
                consecutive_empty += 1
                active = os.path.isfile(active_path)
                done = os.path.isfile(done_path)
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
