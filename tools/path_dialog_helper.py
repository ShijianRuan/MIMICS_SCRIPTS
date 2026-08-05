#!/usr/bin/env python3
"""Isolated native path dialog used by external Mimics-Script windows."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path


HERE = Path(__file__).resolve().parent
for candidate in (HERE, HERE.parent):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from ui_theme import configure_application  # noqa: E402


def read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def write_json_atomic(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        target.name + ".{}.{}.tmp".format(os.getpid(), uuid.uuid4().hex)
    )
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    last_error = None
    for attempt in range(8):
        try:
            os.replace(str(temporary), str(target))
            return
        except OSError as exc:
            last_error = exc
            time.sleep(0.04 * (attempt + 1))
    try:
        target.write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.unlink(missing_ok=True)
    except Exception:
        raise last_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request = read_json(args.request)
    status_path = str(request.get("status_path") or "")
    if not request or not status_path:
        raise RuntimeError("The path dialog request is missing or invalid.")
    try:
        from PySide6 import QtWidgets

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        configure_application(app, "Mimics-Script Path Selection")
        mode = str(request.get("mode") or "directory")
        title = str(request.get("title") or "Select Path")
        initial = str(request.get("initial") or Path.home())
        file_filter = str(request.get("file_filter") or "All files (*)")
        if mode == "directory":
            selection = QtWidgets.QFileDialog.getExistingDirectory(
                None,
                title,
                initial,
                QtWidgets.QFileDialog.ShowDirsOnly,
            )
        elif mode == "open_file":
            selection, _selected_filter = QtWidgets.QFileDialog.getOpenFileName(
                None, title, initial, file_filter
            )
        elif mode == "open_files":
            selection, _selected_filter = QtWidgets.QFileDialog.getOpenFileNames(
                None, title, initial, file_filter
            )
        elif mode == "save_file":
            selection, _selected_filter = QtWidgets.QFileDialog.getSaveFileName(
                None, title, initial, file_filter
            )
        else:
            raise ValueError("Unsupported path dialog mode: {}".format(mode))
        if selection:
            write_json_atomic(
                status_path,
                {
                    "status": "submitted",
                    "selection": selection,
                    "updated_at_epoch": time.time(),
                },
            )
        else:
            write_json_atomic(
                status_path,
                {"status": "cancelled", "updated_at_epoch": time.time()},
            )
        return 0
    except Exception as exc:
        write_json_atomic(
            status_path,
            {
                "status": "failed",
                "error": "{}: {}".format(type(exc).__name__, exc),
                "updated_at_epoch": time.time(),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
