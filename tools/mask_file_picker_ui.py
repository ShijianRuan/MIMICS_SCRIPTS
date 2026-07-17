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

# The embeddable Python (nninteractive_env) uses python313._pth, which fully
# replaces sys.path and does NOT include the script's own directory. Add both
# the script directory (so `import ui_theme` works) and the project root (so
# `import tools.ui_theme` works) before the imports below.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from ui_theme import configure_application
except ImportError:
    from tools.ui_theme import configure_application

def read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else default
    except Exception:
        return default


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
    context_path = Path(args.context)
    fallback_status_path = context_path.with_name(
        context_path.name.replace("_context.json", ".json")
    )
    context = read_json(context_path, None)
    if not context:
        error = "Mask selector context is missing or invalid: {0}".format(context_path)
        try:
            write_json(fallback_status_path, {
                "status": "failed",
                "error": error,
                "updated_at_epoch": time.time(),
            })
        except Exception:
            pass
        print(error, file=sys.stderr)
        return 1
    status_path = context.get("status_path") or str(fallback_status_path)

    try:
        from PySide6 import QtWidgets
    except Exception as exc:
        write_json(status_path, {
            "status": "failed",
            "error": "PySide6 is required for the external Mask selector: {0}".format(exc),
        })
        return 1

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    configure_application(app, "Mimics Mask Import")
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
