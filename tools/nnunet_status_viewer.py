#!/usr/bin/env python3
"""External live status and model viewer for managed nnU-Net tasks."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "tools"):
    value = str(candidate)
    if value not in sys.path:
        sys.path.insert(0, value)

from nnunet_common import load_models, model_usability, read_json  # noqa: E402
from nnunet_jobs import abandon_job, create_job, list_jobs, stop_job  # noqa: E402
from ui_theme import configure_application, stylesheet  # noqa: E402


def _format_time(value):
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(value)))
    except Exception:
        return "-"


class StatusWindow:
    def __init__(self, context, qt_modules):
        self.context = context
        self.QtCore, self.QtGui, self.QtWidgets = qt_modules
        QtCore, QtWidgets = self.QtCore, self.QtWidgets
        self.workspace = str(
            context.get("workspace")
            or Path.home() / ".mimics_script" / "nnunet"
        )
        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle("nnU-Net Status")
        self.window.resize(980, 740)
        self.window.setMinimumSize(800, 620)
        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(22, 18, 22, 18)
        root.setSpacing(12)
        title = QtWidgets.QLabel("nnU-Net Status")
        title.setObjectName("title")
        root.addWidget(title)
        header = QtWidgets.QHBoxLayout()
        header.addWidget(QtWidgets.QLabel("Task"))
        self.task_combo = QtWidgets.QComboBox()
        self.task_combo.currentIndexChanged.connect(self.refresh)
        header.addWidget(self.task_combo, 1)
        self.live_label = QtWidgets.QLabel("● Live")
        self.live_label.setObjectName("liveLabel")
        header.addWidget(self.live_label)
        refresh = QtWidgets.QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        header.addWidget(refresh)
        root.addLayout(header)

        summary = QtWidgets.QFrame()
        summary.setObjectName("surface")
        summary_layout = QtWidgets.QGridLayout(summary)
        summary_layout.setContentsMargins(16, 14, 16, 14)
        self.state_label = QtWidgets.QLabel("No task selected")
        self.state_label.setStyleSheet("font-size: 13pt; font-weight: 650;")
        self.detail_label = QtWidgets.QLabel("")
        self.detail_label.setObjectName("subtitle")
        self.detail_label.setWordWrap(True)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("Overall %p%")
        summary_layout.addWidget(self.state_label, 0, 0)
        summary_layout.addWidget(self.detail_label, 1, 0)
        summary_layout.addWidget(self.progress, 2, 0)
        root.addWidget(summary)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        left = QtWidgets.QFrame()
        left.setObjectName("surface")
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(14, 12, 14, 12)
        heading = QtWidgets.QLabel("Recent runs")
        heading.setObjectName("section")
        left_layout.addWidget(heading)
        self.jobs = QtWidgets.QListWidget()
        self.jobs.currentRowChanged.connect(self._show_selected_job)
        left_layout.addWidget(self.jobs, 1)
        splitter.addWidget(left)

        right = QtWidgets.QTabWidget()
        log_tab = QtWidgets.QWidget()
        log_layout = QtWidgets.QVBoxLayout(log_tab)
        log_layout.setContentsMargins(10, 10, 10, 10)
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        log_layout.addWidget(self.log_view)
        right.addTab(log_tab, "Log")
        models_tab = QtWidgets.QWidget()
        models_layout = QtWidgets.QVBoxLayout(models_tab)
        models_layout.setContentsMargins(10, 10, 10, 10)
        self.models = QtWidgets.QTableWidget(0, 6)
        self.models.setHorizontalHeaderLabels(
            ["Model", "Configuration", "Fold", "Output labels", "Created", "Ready"]
        )
        self.models.horizontalHeader().setStretchLastSection(True)
        self.models.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.Stretch
        )
        self.models.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.models.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        models_layout.addWidget(self.models)
        right.addTab(models_tab, "Models")
        curve_tab = QtWidgets.QWidget()
        curve_layout = QtWidgets.QVBoxLayout(curve_tab)
        curve_layout.setContentsMargins(10, 10, 10, 10)
        self.curve = QtWidgets.QLabel(
            "The nnU-Net training curve appears here after the trainer writes progress.png."
        )
        self.curve.setObjectName("hint")
        self.curve.setAlignment(QtCore.Qt.AlignCenter)
        self.curve.setScaledContents(False)
        curve_layout.addWidget(self.curve, 1)
        right.addTab(curve_tab, "Training curve")
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, 1)

        actions = QtWidgets.QHBoxLayout()
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.clicked.connect(self._stop)
        actions.addWidget(self.stop_button)
        self.retry_button = QtWidgets.QPushButton("Retry Same Settings")
        self.retry_button.clicked.connect(self._retry)
        actions.addWidget(self.retry_button)
        self.open_button = QtWidgets.QPushButton("Open Job Folder")
        self.open_button.clicked.connect(self._open_folder)
        actions.addWidget(self.open_button)
        actions.addStretch(1)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.window.close)
        actions.addWidget(close)
        root.addLayout(actions)
        self.window.setCentralWidget(central)
        self._job_rows = []
        self._model_rows = []
        self._populate_tasks()
        self.timer = QtCore.QTimer(self.window)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(2000)

    def _populate_tasks(self):
        wanted = str(self.context.get("task_id") or "")
        tasks = {}
        for row in list_jobs(self.workspace):
            task_id = str(row.get("task_id") or "Unassigned")
            tasks[task_id] = str(row.get("task_name") or task_id)
        for row in load_models(self.workspace):
            task_id = str(row.get("task_id") or "Unassigned")
            tasks[task_id] = str(row.get("task_name") or task_id)
        self.task_combo.blockSignals(True)
        self.task_combo.clear()
        selected = 0
        for index, task_id in enumerate(sorted(tasks)):
            self.task_combo.addItem(tasks[task_id], task_id)
            if task_id == wanted:
                selected = index
        self.task_combo.setCurrentIndex(selected)
        self.task_combo.blockSignals(False)
        self.refresh()

    def _selected_task(self):
        return str(self.task_combo.currentData() or "")

    def refresh(self):
        task_id = self._selected_task()
        if not task_id:
            self.state_label.setText("No nnU-Net task found")
            self.detail_label.setText("Start training to create the first managed task.")
            self.progress.setValue(0)
            return
        current_status_path = ""
        if 0 <= self.jobs.currentRow() < len(self._job_rows):
            current_status_path = str(self._job_rows[self.jobs.currentRow()].get("status_path") or "")
        rows = [row for row in list_jobs(self.workspace) if str(row.get("task_id") or "") == task_id]
        self._job_rows = rows
        self.jobs.blockSignals(True)
        self.jobs.clear()
        selected = 0
        for index, row in enumerate(rows):
            label = "{} · {} · {}".format(
                _format_time(row.get("created_at_epoch")),
                str(row.get("kind") or "task").title(),
                str(row.get("status") or "unknown").replace("_", " ").title(),
            )
            self.jobs.addItem(label)
            if str(row.get("status_path") or "") == current_status_path:
                selected = index
        self.jobs.setCurrentRow(selected if rows else -1)
        self.jobs.blockSignals(False)
        self._model_rows = [row for row in load_models(self.workspace) if str(row.get("task_id") or "") == task_id]
        self.models.setRowCount(len(self._model_rows))
        for row_index, model in enumerate(self._model_rows):
            usable, reason = model_usability(model)
            values = [
                model.get("model_id", ""),
                model.get("configuration", ""),
                ", ".join(str(value) for value in model.get("folds") or []),
                ", ".join(
                    str(label.get("name") or "")
                    for label in model.get("labels") or []
                ),
                _format_time(model.get("created_at_epoch")),
                "Yes" if usable else "No",
            ]
            for column, value in enumerate(values):
                item = self.QtWidgets.QTableWidgetItem(str(value))
                if column == 3:
                    item.setToolTip(str(value))
                if not usable:
                    item.setToolTip(reason)
                self.models.setItem(row_index, column, item)
        self._show_selected_job()
        self.live_label.setText("● Live · {}".format(time.strftime("%H:%M:%S")))

    def _show_selected_job(self):
        index = self.jobs.currentRow()
        status = self._job_rows[index] if 0 <= index < len(self._job_rows) else {}
        state = str(status.get("status") or "unknown")
        self.state_label.setText(state.replace("_", " ").title())
        self.detail_label.setText(
            str(status.get("message") or status.get("error") or status.get("phase") or "")
        )
        if state in {"uploading", "downloading"}:
            speed = float(status.get("transfer_bytes_per_second") or 0.0)
            eta = status.get("transfer_eta_seconds")
            detail = "{} {}%".format(
                "Uploading" if state == "uploading" else "Downloading",
                int(status.get("transfer_percent") or 0),
            )
            if speed > 0:
                detail += " at {:.1f} MiB/s".format(speed / float(1024 ** 2))
            if eta is not None:
                detail += ", about {} min remaining".format(
                    max(1, int(round(float(eta) / 60.0)))
                )
            self.detail_label.setText(detail)
        self.progress.setValue(max(0, min(100, int(status.get("progress_percent") or 0))))
        abandonable = bool(
            str(status.get("execution_backend") or "") == "remote"
            and status.get("remote_state_unknown")
            and state not in {
                "completed", "failed", "cancelled", "abandoned", "unknown"
            }
        )
        self.stop_button.setText("Abandon Locally" if abandonable else "Stop")
        self.stop_button.setEnabled(
            abandonable
            or state not in {
                "completed", "failed", "cancelled", "abandoned", "unknown"
            }
        )
        # A successful fold is already a registered model. Re-running the same
        # request would target the same nnU-Net results directory and could
        # replace its checkpoints, so one-click retry is intentionally limited
        # to interrupted or failed work.
        self.retry_button.setEnabled(state in {"failed", "cancelled"})
        log_path = Path(str(status.get("log_path") or ""))
        scrollbar = self.log_view.verticalScrollBar()
        follow_tail = scrollbar.value() >= scrollbar.maximum() - 4
        text = ""
        if log_path.is_file():
            try:
                with log_path.open("rb") as handle:
                    handle.seek(max(0, log_path.stat().st_size - 512 * 1024))
                    text = handle.read().decode("utf-8", "replace")
            except OSError as exc:
                text = "Could not read the log: {}".format(exc)
        self.log_view.setPlainText(text)
        if follow_tail:
            scrollbar.setValue(scrollbar.maximum())
        self._show_curve(status)

    def _show_curve(self, status):
        model = status.get("model") or {}
        model_dir = Path(str(model.get("model_dir") or ""))
        candidates = [Path(str(status.get("training_curve_path") or ""))]
        candidates.append(model_dir / "progress.png")
        candidates.extend(model_dir.glob("fold_*/progress.png") if model_dir.is_dir() else [])
        curve_path = next((path for path in candidates if path.is_file()), None)
        if curve_path is None:
            self.curve.setPixmap(self.QtGui.QPixmap())
            self.curve.setText(
                "Training curve is not available yet. Live epoch details remain in the Log tab."
            )
            return
        pixmap = self.QtGui.QPixmap(str(curve_path))
        if pixmap.isNull():
            self.curve.setText("The training curve could not be decoded.")
            return
        size = self.curve.size()
        self.curve.setText("")
        self.curve.setPixmap(
            pixmap.scaled(
                max(200, size.width() - 20),
                max(180, size.height() - 20),
                self.QtCore.Qt.KeepAspectRatio,
                self.QtCore.Qt.SmoothTransformation,
            )
        )

    def _stop(self):
        index = self.jobs.currentRow()
        if 0 <= index < len(self._job_rows):
            status = self._job_rows[index]
            abandonable = bool(
                str(status.get("execution_backend") or "") == "remote"
                and status.get("remote_state_unknown")
                and str(status.get("status") or "")
                not in {"completed", "failed", "cancelled", "abandoned"}
            )
            if abandonable:
                answer = self.QtWidgets.QMessageBox.warning(
                    self.window,
                    "Abandon Remote Task Locally",
                    "Stop waiting on this workstation?\n\nThe server cannot "
                    "confirm whether the container stopped. It may still use "
                    "GPU or disk resources. An administrator must inspect the "
                    "recorded container name.",
                    self.QtWidgets.QMessageBox.Yes
                    | self.QtWidgets.QMessageBox.No,
                    self.QtWidgets.QMessageBox.No,
                )
                if answer != self.QtWidgets.QMessageBox.Yes:
                    return
                abandon_job(status["status_path"])
            else:
                stop_job(status["status_path"])
            self.refresh()

    def _retry(self):
        index = self.jobs.currentRow()
        if not 0 <= index < len(self._job_rows):
            return
        status = self._job_rows[index]
        request = read_json(status.get("request_path"), {}) or {}
        request.pop("job_id", None)
        try:
            create_job(request)
            self.refresh()
        except Exception as exc:
            self.QtWidgets.QMessageBox.critical(
                self.window, "Retry Failed", str(exc)
            )

    def _open_folder(self):
        index = self.jobs.currentRow()
        if not 0 <= index < len(self._job_rows):
            return
        folder = Path(str(self._job_rows[index].get("job_dir") or ""))
        if not folder.is_dir():
            return
        if os.name == "nt":
            os.startfile(str(folder))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(folder)])
        else:
            subprocess.Popen(["xdg-open", str(folder)])

    def show(self):
        self.window.show()
        self.window.raise_()
        self.window.activateWindow()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", required=True)
    args = parser.parse_args()
    context = read_json(Path(args.context).expanduser().resolve(), {}) or {}
    from PySide6 import QtCore, QtGui, QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    configure_application(app, "nnU-Net Status")
    app.setStyleSheet(stylesheet())
    window = StatusWindow(context, (QtCore, QtGui, QtWidgets))
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
