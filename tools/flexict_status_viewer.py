#!/usr/bin/env python3
"""External live status and model viewer for managed FlexiCT tasks."""

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

from flexict_common import (  # noqa: E402
    load_models,
    load_pair,
    model_usability,
    recommended_model,
)
from flexict_pipeline import list_flexict_jobs  # noqa: E402
from nnunet_common import read_json  # noqa: E402
from nnunet_jobs import abandon_job, stop_job  # noqa: E402
from ui_theme import configure_application, stylesheet  # noqa: E402
from viewer_refresh import BackgroundRefresh  # noqa: E402


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
        self.workspace = str(context.get("workspace") or "")
        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle("FlexiCT Status")
        self.window.resize(980, 740)
        self.window.setMinimumSize(800, 620)
        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(22, 18, 22, 18)
        root.setSpacing(12)
        title = QtWidgets.QLabel("FlexiCT Status")
        title.setObjectName("title")
        root.addWidget(title)
        header = QtWidgets.QHBoxLayout()
        self.live_label = QtWidgets.QLabel("● Live")
        self.live_label.setObjectName("liveLabel")
        header.addWidget(self.live_label)
        header.addStretch(1)
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
        self.models = QtWidgets.QTableWidget(0, 5)
        self.models.setHorizontalHeaderLabels(
            ["Model", "Target", "Configuration", "Created", "Ready"]
        )
        self.models.horizontalHeader().setStretchLastSection(True)
        self.models.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.Stretch
        )
        self.models.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.models.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        models_layout.addWidget(self.models)
        pair_note = QtWidgets.QLabel("")
        pair_note.setObjectName("hint")
        pair_note.setWordWrap(True)
        self.pair_note = pair_note
        models_layout.addWidget(pair_note)
        right.addTab(models_tab, "Models")
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, 1)

        actions = QtWidgets.QHBoxLayout()
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.clicked.connect(self._stop)
        actions.addWidget(self.stop_button)
        self.reattach_button = QtWidgets.QPushButton("Re-attach")
        self.reattach_button.clicked.connect(self._reattach)
        self.reattach_button.setToolTip(
            "Resume monitoring a remote task whose local controller stopped "
            "(e.g. after a workstation restart)."
        )
        actions.addWidget(self.reattach_button)
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
        # Periodic refresh collects off the GUI thread (see viewer_refresh).
        self.refresher = BackgroundRefresh(
            self.QtCore,
            parent=self.window,
            interval_ms=2000,
            collect=lambda: self._collect(),
            apply=self.refresh,
        )
        self.refresher.refresh_now()

    def _collect(self):
        """Snapshot everything the UI needs (runs on a worker thread)."""
        workspace = self.workspace or None
        return {
            "jobs": list_flexict_jobs(workspace),
            "models": load_models(workspace),
            "pair": load_pair(workspace),
            "recommended": recommended_model(workspace),
        }

    def refresh(self, snapshot=None):
        if snapshot is None:
            # Manual/periodic path: collect synchronously (tests, first paint).
            snapshot = self._collect()
        rows = snapshot["jobs"]
        self._job_rows = rows
        current_status_path = ""
        if 0 <= self.jobs.currentRow() < len(self._job_rows):
            current_status_path = str(
                self._job_rows[self.jobs.currentRow()].get("status_path") or ""
            )
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

        workspace = self.workspace or None
        self._model_rows = snapshot["models"]
        self.models.setRowCount(len(self._model_rows))
        for row_index, model in enumerate(self._model_rows):
            usable, reason = model_usability(model)
            values = [
                model.get("model_id", ""),
                model.get("label_name", ""),
                model.get("configuration", ""),
                _format_time(model.get("created_at_epoch")),
                "Yes" if usable else "No",
            ]
            for column, value in enumerate(values):
                item = self.QtWidgets.QTableWidgetItem(str(value))
                if not usable:
                    item.setToolTip(reason)
                self.models.setItem(row_index, column, item)
        model_2d, model_3d = snapshot["pair"]
        if model_2d is not None and model_3d is not None:
            self.pair_note.setText(
                "Active-learning pair ready: {} (2D) + {} (3D).".format(
                    model_2d.get("model_id"), model_3d.get("model_id")
                )
            )
        else:
            recommended = snapshot["recommended"]
            if recommended is not None:
                self.pair_note.setText(
                    "Recommended model: {} ({}). Train a 2D+3D pair to enable "
                    "active learning.".format(
                        recommended.get("model_id"),
                        recommended.get("configuration"),
                    )
                )
            else:
                self.pair_note.setText(
                    "No usable model yet. Train one (or a pair) to enable "
                    "prediction and active learning."
                )
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
        self.reattach_button.setEnabled(
            str(status.get("execution_backend") or "") == "remote"
            and bool(str(status.get("remote_job_dir") or "").strip())
            and state not in {
                "completed", "failed", "cancelled", "abandoned", "unknown"
            }
        )
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
                # Same contract as the nnU-Net viewer: a remote task whose
                # state is unknown cannot be confirmed stopped, so plain
                # Stop must not claim the server GPU is free. The escape
                # hatch asks once, then abandons monitoring locally only.
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

    def _reattach(self):
        index = self.jobs.currentRow()
        if not 0 <= index < len(self._job_rows):
            return
        from nnunet_jobs import reattach_job

        try:
            reattach_job(self._job_rows[index]["status_path"])
            self.refresh()
        except Exception as exc:
            self.QtWidgets.QMessageBox.critical(
                self.window, "Re-attach Failed", str(exc)
            )

    def _retry(self):
        index = self.jobs.currentRow()
        if not 0 <= index < len(self._job_rows):
            return
        status = self._job_rows[index]
        request = read_json(status.get("request_path"), {}) or {}
        request.pop("job_id", None)
        try:
            from flexict_pipeline import create_flexict_job

            create_flexict_job(request)
            self.refresh()
        except Exception as exc:
            self.QtWidgets.QMessageBox.critical(self.window, "Retry Failed", str(exc))

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
    configure_application(app, "FlexiCT Status")
    app.setStyleSheet(stylesheet())
    window = StatusWindow(context, (QtCore, QtGui, QtWidgets))
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
