# -*- coding: utf-8 -*-
"""create_mcs_batch.py — Background Mimics script to create .mcs files.

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
        # Note: mask.image is read-only — masks are automatically linked
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
    print("  已保存: {}".format(mcs_path))

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

    consecutive_empty = 0
    total_completed = 0
    total_failed = 0

    while True:
        # Find all work directories with prepare manifests
        work_dirs = []
        for item in sorted(os.listdir(output_dir)):
            if not item.endswith("_work"):
                continue
            work_dir = os.path.join(output_dir, item)
            manifest = os.path.join(work_dir, "prepare_manifest.json")
            if os.path.isfile(manifest):
                case_id = item.replace("_work", "")
                mcs_path = os.path.join(output_dir, case_id + ".mcs")
                # Skip if .mcs already exists
                if os.path.isfile(mcs_path):
                    continue
                work_dirs.append((case_id, work_dir, mcs_path))

        total = len(work_dirs)
        if total == 0:
            consecutive_empty += 1
            if consecutive_empty >= 2:
                print("没有新的待处理数据，退出。")
                break
            print("暂无新数据，等待 15 秒后重试...".format(consecutive_empty))
            time.sleep(15)
            continue

        consecutive_empty = 0
        print("正在创建 {} 个 .mcs 文件...".format(total))

        for i, (case_id, work_dir, mcs_path) in enumerate(work_dirs):
            print("\n[{}/{}] 正在创建: {}".format(i + 1, total, case_id))
            try:
                create_mcs_from_manifest(work_dir, mcs_path)
                total_completed += 1
            except Exception as e:
                print("  创建失败: {}".format(e))
                traceback.print_exc()
                total_failed += 1
                # Close project even on failure
                try:
                    mimics.file.close_project()
                except Exception:
                    pass

    print("\n后台创建完成: 成功 {} 个，失败 {} 个".format(total_completed, total_failed))
    return 0


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as error:
        traceback.print_exc()
        raise
