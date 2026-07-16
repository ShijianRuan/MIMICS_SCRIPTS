#!/usr/bin/env python3
"""External multi-file Mask picker; never imports a GUI toolkit into Mimics."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".{0}.{1}.tmp".format(os.getpid(), uuid.uuid4().hex))
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    last_error = None
    for attempt in range(8):
        try:
            os.replace(str(temp), str(target))
            return
        except OSError as exc:
            last_error = exc
            time.sleep(0.04 * (attempt + 1))
    try:
        target.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
        temp.unlink(missing_ok=True)
    except Exception:
        raise last_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", required=True)
    args = parser.parse_args()
    context = read_json(args.context)
    status_path = context["status_path"]

    try:
        from PySide6 import QtWidgets
    except Exception as exc:
        write_json(status_path, {
            "status": "failed",
            "error": "PySide6 is required for the external Mask selector: {0}".format(exc),
        })
        return 1

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Mimics Script")
    paths, _selected_filter = QtWidgets.QFileDialog.getOpenFileNames(
        None,
        "Select Masks to Import",
        str(Path.home()),
        "Segmentation files (*.nii *.nii.gz *.mha *.mhd *.nrrd *.seg.nii *.seg.nii.gz);;All files (*)",
    )
    if not paths:
        write_json(status_path, {"status": "cancelled", "updated_at_epoch": time.time()})
        return 0
    paths = [os.path.abspath(str(path)) for path in paths if path]
    write_json(status_path, {
        "status": "submitted",
        "selection": {"mask_paths": paths},
        "updated_at_epoch": time.time(),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
