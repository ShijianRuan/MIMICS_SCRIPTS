#!/usr/bin/env python3
"""Batch status viewer: one window for every import/export task on disk.

Aggregates the six on-disk record types an annotator can leave behind
(import runs, background .mcs queues, mask-export jobs, foreground export
tasks, mask-append jobs, drop imports) into a single live table with the
log/folder one click away, and a per-row Stop button that writes the same
stop marker the matching Stop menu entries used to write.

Launch from Mimics: scripting_library/01_Data/04_Task_Status.py.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "tools", ROOT / "runtime_py35"):
    value = str(candidate)
    if value not in sys.path:
        sys.path.insert(0, value)

from ui_theme import PALETTE, configure_application, stylesheet  # noqa: E402
from viewer_refresh import BackgroundRefresh  # noqa: E402


TERMINAL_STATES = {
    "completed", "closed", "failed", "cancelled", "canceled",
    "abandoned", "completed_with_errors", "finished",
}

# How many recent records to show per kind (matching diagnostics defaults).
PER_KIND_LIMIT = 12


def read_json(path, default=None):
    try:
        import json

        with open(str(path), "r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else default
    except Exception:
        return default


def import_runtime_base(project_root):
    """Import records may relocate off a UNC project root; resolve the base.

    Mirrors runtime_common.import_runtime_base without requiring the
    runtime_py35 package when it is unavailable.
    """
    try:
        import runtime_common

        return Path(runtime_common.import_runtime_base(str(project_root)))
    except Exception:
        return Path(project_root) / ".mimics_runtime"


def _row(kind, label, payload, status_path, job_dir=""):
    payload = payload or {}
    try:
        updated = float(payload.get("updated_at_epoch") or 0.0)
    except (TypeError, ValueError):
        updated = 0.0
    if not updated:
        try:
            updated = Path(status_path).stat().st_mtime
        except OSError:
            updated = 0.0
    try:
        completed = int(payload.get("completed", 0) or 0)
    except (TypeError, ValueError):
        completed = 0
    try:
        failed = int(payload.get("failed", 0) or 0)
    except (TypeError, ValueError):
        failed = 0
    total = payload.get("total") or payload.get("total_count") or 0
    try:
        total = int(total or 0)
    except (TypeError, ValueError):
        total = 0
    return {
        "kind": kind,
        "label": label,
        "status": str(payload.get("status") or "unknown"),
        "phase": str(payload.get("phase") or ""),
        "completed": completed,
        "failed": failed,
        "total": total,
        "case_id": str(payload.get("case_id") or ""),
        "error": str(payload.get("error") or ""),
        "updated_at_epoch": updated,
        "status_path": str(status_path),
        "job_dir": str(job_dir or (Path(status_path).parent if status_path else "")),
    }


def _scan_status_dir(rows, kind, base, status_name="status.json"):
    """Collect <base>/<job>/status_name records, newest PER_KIND_LIMIT kept."""
    base = Path(base)
    if not base.is_dir():
        return
    candidates = []
    try:
        for child in base.iterdir():
            if not child.is_dir():
                continue
            status_path = child / status_name
            if status_path.is_file():
                candidates.append(status_path)
    except OSError:
        return
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    for status_path in candidates[:PER_KIND_LIMIT]:
        rows.append(
            _row(
                kind,
                status_path.parent.name,
                read_json(status_path, {}),
                status_path,
                status_path.parent,
            )
        )


def _scan_ui_tasks(rows, kind, base):
    """Foreground export tasks: .mimics_runtime/ui_tasks/<task_id>.json."""
    base = Path(base)
    if not base.is_dir():
        return
    candidates = []
    try:
        for child in base.iterdir():
            if not child.name.endswith(".json") or child.name.endswith("_stop.json"):
                continue
            candidates.append(child)
    except OSError:
        return
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    for status_path in candidates[:PER_KIND_LIMIT]:
        rows.append(
            _row(
                kind,
                status_path.stem,
                read_json(status_path, {}),
                status_path,
                status_path.parent,
            )
        )


def _scan_drop_imports(rows, kind, base):
    """Drop imports: flat per-case/per-batch *_status.json files."""
    base = Path(base)
    if not base.is_dir():
        return
    candidates = []
    try:
        for child in base.iterdir():
            if child.name.endswith("_status.json"):
                candidates.append(child)
    except OSError:
        return
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    for status_path in candidates[:PER_KIND_LIMIT]:
        rows.append(
            _row(
                kind,
                status_path.name[: -len("_status.json")],
                read_json(status_path, {}),
                status_path,
                base,
            )
        )


def _scan_import_queues(rows, kind, base):
    """Background .mcs queues: <queue>/_mcs_batch_status.json."""
    _scan_status_dir(rows, kind, base, status_name="_mcs_batch_status.json")


def collect_batch_rows(project_root, per_kind_limit=None):
    """Aggregate all on-disk batch records, newest-first overall."""
    global PER_KIND_LIMIT
    original = PER_KIND_LIMIT
    if per_kind_limit is not None:
        PER_KIND_LIMIT = int(per_kind_limit)
    try:
        root = Path(project_root)
        runtime = root / ".mimics_runtime"
        rows = []
        _scan_status_dir(rows, "Import", import_runtime_base(root) / "import_runs")
        _scan_import_queues(rows, "Import queue", import_runtime_base(root) / "import_queues")
        _scan_status_dir(rows, "Export", runtime / "export_jobs")
        _scan_ui_tasks(rows, "Export task", runtime / "ui_tasks")
        _scan_status_dir(rows, "Append", runtime / "append_jobs")
        _scan_drop_imports(rows, "Drop import", runtime / "drop_import")
        rows.sort(key=lambda row: row["updated_at_epoch"], reverse=True)
        return rows
    finally:
        PER_KIND_LIMIT = original


def _format_time(value):
    try:
        return time.strftime("%m-%d %H:%M:%S", time.localtime(float(value)))
    except Exception:
        return "-"


class BatchStatusWindow:
    COLUMNS = ["更新时间", "类型", "名称", "状态", "进度", "当前例", "错误"]

    KIND_LABELS = {
        "Import": "导入",
        "Import queue": "导入队列",
        "Export": "导出",
        "Export task": "导出任务",
        "Append": "追加掩膜",
        "Drop import": "拖拽导入",
    }

    def __init__(self, project_root, qt_modules):
        self.QtCore, self.QtGui, self.QtWidgets = qt_modules
        self.project_root = str(project_root)
        QtWidgets = self.QtWidgets
        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle("批量任务状态")
        self.window.resize(1080, 640)
        self.window.setMinimumSize(860, 480)
        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(22, 18, 22, 16)
        root.setSpacing(10)
        title = QtWidgets.QLabel("导入 / 导出批量任务状态")
        title.setObjectName("title")
        root.addWidget(title)
        hint = QtWidgets.QLabel("AI 训练/推理任务请在 02_AI 各家族的状态窗口查看（nnInteractive 03 / nnU-Net 03 / FlexiCT 04）。")
        hint.setObjectName("hint")
        root.addWidget(hint)
        header = QtWidgets.QHBoxLayout()
        self.live_label = QtWidgets.QLabel("● Live")
        self.live_label.setObjectName("liveLabel")
        header.addWidget(self.live_label)
        header.addStretch(1)
        refresh = QtWidgets.QPushButton("刷新")
        refresh.clicked.connect(self.refresh)
        header.addWidget(refresh)
        root.addLayout(header)

        self.table = QtWidgets.QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        header_view = self.table.horizontalHeader()
        header_view.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header_view.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        header_view.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        header_view.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
        header_view.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeToContents)
        header_view.setSectionResizeMode(5, QtWidgets.QHeaderView.ResizeToContents)
        header_view.setSectionResizeMode(6, QtWidgets.QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        root.addWidget(self.table, 1)

        detail = QtWidgets.QLabel("")
        detail.setObjectName("hint")
        detail.setWordWrap(True)
        self.detail = detail
        root.addWidget(detail)

        actions = QtWidgets.QHBoxLayout()
        self.open_folder = QtWidgets.QPushButton("打开任务文件夹")
        self.open_folder.clicked.connect(self._open_folder)
        self.open_log = QtWidgets.QPushButton("打开状态文件位置")
        self.open_log.clicked.connect(self._open_status_location)
        self.stop = QtWidgets.QPushButton("停止")
        self.stop.setObjectName("dangerButton")
        self.stop.clicked.connect(self._stop_selected)
        actions.addWidget(self.open_folder)
        actions.addWidget(self.open_log)
        actions.addWidget(self.stop)
        actions.addStretch(1)
        close = QtWidgets.QPushButton("关闭")
        close.clicked.connect(self.window.close)
        actions.addWidget(close)
        root.addLayout(actions)

        self.window.setCentralWidget(central)
        self._rows = []
        # Periodic refresh collects off the GUI thread: the per-kind scans
        # stat every candidate job directory, which stalls on network paths.
        self.refresher = BackgroundRefresh(
            self.QtCore,
            parent=self.window,
            interval_ms=2000,
            collect=lambda: collect_batch_rows(self.project_root),
            apply=self.refresh,
        )
        self.refresher.refresh_now()

    def refresh(self, rows=None):
        if rows is None:
            # Manual/periodic path: collect synchronously (tests, first paint).
            rows = collect_batch_rows(self.project_root)
        QtWidgets = self.QtWidgets
        selected_status = ""
        for item in self.table.selectedItems():
            selected_status = item.data(_user_role(self.QtCore))
            break
        self._rows = rows
        self.table.setRowCount(len(self._rows))
        for row_index, row in enumerate(self._rows):
            progress = ""
            if row["total"]:
                progress = "{0} / {1}".format(row["completed"] + row["failed"], row["total"])
            values = [
                _format_time(row["updated_at_epoch"]),
                self.KIND_LABELS.get(row["kind"], row["kind"]),
                row["label"],
                _display_status(row),
                progress,
                row["case_id"],
                row["error"][:120],
            ]
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(value))
                item.setData(_user_role(self.QtCore), row["status_path"])
                if column == 3:
                    _style_status_item(self.QtGui, item, row["status"])
                item.setToolTip(
                    "{0}\n{1}".format(row["status_path"], row["error"])
                    if row["error"]
                    else row["status_path"]
                )
                self.table.setItem(row_index, column, item)
        self.live_label.setText("● Live · {0}".format(time.strftime("%H:%M:%S")))
        # Restore the selection that was active before the rebuild.
        if selected_status:
            for row_index, row in enumerate(self._rows):
                if row["status_path"] == selected_status:
                    self.table.selectRow(row_index)
                    break
        self._selection_changed()

    def _selection_changed(self):
        row = self._selected_row()
        if not row:
            self.detail.setText("选择一行即可查看其状态文件和文件夹。")
            self.open_folder.setEnabled(False)
            self.open_log.setEnabled(False)
            self.stop.setEnabled(False)
            return
        detail = "状态文件：{0}".format(row["status_path"])
        if row["phase"]:
            detail += "  ·  阶段：{0}".format(row["phase"])
        self.detail.setText(detail)
        self.open_folder.setEnabled(bool(row["job_dir"]))
        self.open_log.setEnabled(bool(row["status_path"]))
        self.stop.setEnabled(
            bool(_stop_marker(row)) and row["status"] not in TERMINAL_STATES
        )

    def _stop_selected(self):
        row = self._selected_row()
        if not row or row["status"] in TERMINAL_STATES:
            return
        QtWidgets = self.QtWidgets
        stoppable = _stop_marker(row)
        if not stoppable:
            return
        answer = QtWidgets.QMessageBox.question(
            self.window,
            "停止任务",
            "停止 {0} “{1}” 吗？\n\n"
            "停止请求写入后，任务会在当前病例完成后退出；"
            "已完成的病例不受影响。".format(self.KIND_LABELS.get(row["kind"], row["kind"]), row["label"]),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return
        ok, message = request_stop(row)
        if ok:
            self.detail.setText(message)
            self.refresher.refresh_now()
        else:
            QtWidgets.QMessageBox.warning(self.window, "停止失败", message)

    def _selected_row(self):
        indexes = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not indexes:
            return None
        row_index = indexes[0].row()
        if 0 <= row_index < len(self._rows):
            return self._rows[row_index]
        return None

    def _open_folder(self):
        row = self._selected_row()
        if row and row["job_dir"]:
            _open_in_explorer(row["job_dir"])

    def _open_status_location(self):
        row = self._selected_row()
        if row and row["status_path"]:
            _open_in_explorer(row["status_path"])

    def show(self):
        self.window.show()
        self.window.raise_()
        self.window.activateWindow()


def _user_role(QtCore):
    try:
        return QtCore.Qt.ItemDataRole.UserRole
    except AttributeError:
        return QtCore.Qt.UserRole


def _display_status(row):
    status = row["status"]
    if status in TERMINAL_STATES:
        return status.replace("_", " ").title()
    return status.replace("_", " ").title()


# Per-kind stop semantics. Every task kind polls a stop marker next to its
# status file and exits at the next poll; the kind only decides the marker's
# name. Import queue additionally drops its active marker so new cases cannot
# join a queue being stopped (mirrors mimics_stop_background).
def _stop_marker(row):
    """Return the stop-marker path for a row, or '' when not stoppable."""
    kind = row["kind"]
    status_path = row.get("status_path") or ""
    job_dir = row.get("job_dir") or ""
    if not status_path:
        return ""
    if kind == "Import":
        return str(Path(job_dir) / "stop.json")
    if kind == "Import queue":
        return str(Path(job_dir) / "_mcs_queue_stop.json")
    if kind == "Export":
        return str(Path(job_dir) / "_export_stop.json")
    if kind == "Export task":
        stem = Path(status_path).stem
        return str(Path(status_path).parent / "{0}_stop.json".format(stem))
    if kind == "Append":
        return str(Path(job_dir) / "stop.json")
    if kind == "Drop import":
        stem = Path(status_path).name
        if stem.endswith("_status.json"):
            stem = stem[: -len("_status.json")]
        return str(Path(status_path).parent / "{0}_stop.json".format(stem))
    return ""


def request_stop(row, reason="user"):
    """Write the kind-specific stop marker; return (ok, message)."""
    marker = _stop_marker(row)
    if not marker:
        return False, "This task kind cannot be stopped from here."
    payload = {
        "status": "stop_requested",
        "requested_at_epoch": time.time(),
        "reason": reason,
    }
    try:
        Path(marker).parent.mkdir(parents=True, exist_ok=True)
        tmp = marker + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp, marker)
    except OSError as exc:
        return False, "Failed to write the stop marker: {0}".format(exc)
    if row["kind"] == "Import queue":
        # A stopped queue must not pick up new cases: drop the active marker
        # the queue scanner keys on (same as _request_queue_stop).
        try:
            os.remove(str(Path(row["job_dir"]) / "_mcs_queue_active.json"))
        except OSError:
            pass
    return True, "Stop requested for {0} {1}.".format(row["kind"], row["label"])


def _style_status_item(QtGui, item, status):
    status = str(status or "").lower()
    color = PALETTE["success"]
    if status in ("failed", "abandoned"):
        color = PALETTE["danger"]
    elif status in ("cancelled", "canceled"):
        color = PALETTE["warning"]
    elif status not in TERMINAL_STATES:
        color = PALETTE["info"]  # still running
    item.setForeground(QtGui.QColor(color))


def _open_in_explorer(target):
    target = str(target)
    if not os.path.exists(target):
        return
    try:
        if os.name == "nt":
            if os.path.isfile(target):
                subprocess.Popen(["explorer", "/select,", os.path.normpath(target)])
            else:
                os.startfile(target)  # noqa
        elif sys.platform == "darwin":
            subprocess.Popen(["open", target])
        else:
            subprocess.Popen(["xdg-open", target])
    except Exception:
        pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=str(ROOT),
        help="Project root whose .mimics_runtime should be shown",
    )
    args = parser.parse_args(argv)
    from PySide6 import QtCore, QtGui, QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    configure_application(app, "批量任务状态")
    app.setStyleSheet(stylesheet())
    window = BatchStatusWindow(args.root, (QtCore, QtGui, QtWidgets))
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
