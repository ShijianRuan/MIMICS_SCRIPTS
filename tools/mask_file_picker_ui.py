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

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _candidate in (_HERE, _ROOT):
    if _candidate and _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from ui_theme import (
    choose_open_files_async,
    configure_application,
    stylesheet,
)

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
        from PySide6 import QtCore, QtWidgets
    except Exception as exc:
        write_json(status_path, {
            "status": "failed",
            "error": "PySide6 is required for the external Mask selector: {0}".format(exc),
        })
        return 1

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    configure_application(app, "Mimics Mask Import")
    app.setStyleSheet(stylesheet())
    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle("导入 Mask")
    dialog.resize(720, 480)
    dialog.setMinimumSize(620, 420)
    root = QtWidgets.QVBoxLayout(dialog)
    root.setContentsMargins(22, 18, 22, 18)
    root.setSpacing(12)
    title = QtWidgets.QLabel("导入 Mask")
    title.setObjectName("title")
    subtitle = QtWidgets.QLabel(
        "选择一个或多个分割文件。文件发现和转换在 Mimics GUI 之外运行。"
    )
    subtitle.setObjectName("subtitle")
    subtitle.setWordWrap(True)
    root.addWidget(title)
    root.addWidget(subtitle)
    files = QtWidgets.QListWidget()
    files.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
    files.setAlternatingRowColors(True)
    root.addWidget(files, 1)
    tools = QtWidgets.QHBoxLayout()
    add_files = QtWidgets.QPushButton("添加文件...")
    paste = QtWidgets.QPushButton("粘贴路径")
    remove = QtWidgets.QPushButton("移除所选")
    clear = QtWidgets.QPushButton("清空")
    tools.addWidget(add_files)
    tools.addWidget(paste)
    tools.addStretch(1)
    tools.addWidget(remove)
    tools.addWidget(clear)
    root.addLayout(tools)
    status = QtWidgets.QLabel(
        "可以粘贴多行路径，避免在慢速网络盘上逐个浏览。"
    )
    status.setObjectName("hint")
    status.setWordWrap(True)
    root.addWidget(status)
    actions = QtWidgets.QHBoxLayout()
    actions.addStretch(1)
    cancel = QtWidgets.QPushButton("取消")
    submit = QtWidgets.QPushButton("导入 Mask")
    submit.setObjectName("primary")
    submit.setEnabled(False)
    actions.addWidget(cancel)
    actions.addWidget(submit)
    root.addLayout(actions)
    state = {"submitted": False}

    def current_paths():
        return [str(files.item(index).data(32) or "") for index in range(files.count())]

    def add_paths(values):
        existing = {os.path.normcase(path) for path in current_paths() if path}
        added = 0
        for value in values or []:
            path = os.path.abspath(os.path.expandvars(os.path.expanduser(str(value).strip().strip('"'))))
            if not path or os.path.normcase(path) in existing:
                continue
            item = QtWidgets.QListWidgetItem(os.path.basename(path) or path)
            item.setToolTip(path)
            item.setData(32, path)
            files.addItem(item)
            existing.add(os.path.normcase(path))
            added += 1
        submit.setEnabled(files.count() > 0)
        status.setText("{} file(s) selected.".format(files.count()))
        return added

    def browse():
        choose_open_files_async(
            QtCore,
            QtWidgets,
            dialog,
            "Select Masks to Import",
            str(context.get("initial_path") or Path.home()),
            "Segmentation files (*.nii *.nii.gz *.mha *.mhd *.nrrd *.seg.nii *.seg.nii.gz);;All files (*)",
            add_paths,
            button=add_files,
            error_callback=lambda message: status.setText(message),
        )

    def paste_paths():
        raw = QtWidgets.QApplication.clipboard().text()
        values = [line.strip() for line in raw.replace("\r", "\n").split("\n") if line.strip()]
        if not values:
            status.setText("剪贴板中没有路径。")
            return
        add_paths(values)

    def remove_selected():
        for item in files.selectedItems():
            files.takeItem(files.row(item))
        submit.setEnabled(files.count() > 0)
        status.setText("{} file(s) selected.".format(files.count()))

    def finish():
        paths = [path for path in current_paths() if path]
        if not paths:
            status.setText("请至少选择一个分割文件。")
            return
        state["submitted"] = True
        write_json(status_path, {
            "status": "submitted",
            "selection": {"mask_paths": paths},
            "updated_at_epoch": time.time(),
        })
        dialog.accept()

    def cancelled():
        if not state["submitted"]:
            write_json(status_path, {"status": "cancelled", "updated_at_epoch": time.time()})

    def clear_paths():
        files.clear()
        submit.setEnabled(False)
        status.setText("未选择任何文件。")

    add_files.clicked.connect(browse)
    paste.clicked.connect(paste_paths)
    remove.clicked.connect(remove_selected)
    clear.clicked.connect(clear_paths)
    submit.clicked.connect(finish)
    cancel.clicked.connect(dialog.reject)
    dialog.rejected.connect(cancelled)
    dialog.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
