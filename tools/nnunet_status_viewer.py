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
from nnunet_jobs import (  # noqa: E402
    abandon_job,
    create_job,
    list_jobs,
    reattach_job,
    stop_job,
)
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
        self.workspace = str(
            context.get("workspace")
            or Path.home() / ".mimics_script" / "nnunet"
        )
        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle("nnU-Net 状态")
        self.window.resize(980, 740)
        self.window.setMinimumSize(800, 620)
        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(22, 18, 22, 18)
        root.setSpacing(12)
        title = QtWidgets.QLabel("nnU-Net 状态")
        title.setObjectName("title")
        root.addWidget(title)
        header = QtWidgets.QHBoxLayout()
        header.addWidget(QtWidgets.QLabel("任务"))
        self.task_combo = QtWidgets.QComboBox()
        self.task_combo.currentIndexChanged.connect(self.refresh)
        header.addWidget(self.task_combo, 1)
        self.live_label = QtWidgets.QLabel("● Live")
        self.live_label.setObjectName("liveLabel")
        header.addWidget(self.live_label)
        refresh = QtWidgets.QPushButton("刷新")
        refresh.clicked.connect(self.refresh)
        header.addWidget(refresh)
        root.addLayout(header)

        summary = QtWidgets.QFrame()
        summary.setObjectName("surface")
        summary_layout = QtWidgets.QGridLayout(summary)
        summary_layout.setContentsMargins(16, 14, 16, 14)
        self.state_label = QtWidgets.QLabel("未选择任务")
        self.state_label.setStyleSheet("font-size: 13pt; font-weight: 650;")
        self.detail_label = QtWidgets.QLabel("")
        self.detail_label.setObjectName("subtitle")
        self.detail_label.setWordWrap(True)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("总进度 %p%")
        summary_layout.addWidget(self.state_label, 0, 0)
        summary_layout.addWidget(self.detail_label, 1, 0)
        summary_layout.addWidget(self.progress, 2, 0)
        root.addWidget(summary)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        left = QtWidgets.QFrame()
        left.setObjectName("surface")
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(14, 12, 14, 12)
        heading = QtWidgets.QLabel("最近运行")
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
        right.addTab(log_tab, "日志")
        models_tab = QtWidgets.QWidget()
        models_layout = QtWidgets.QVBoxLayout(models_tab)
        models_layout.setContentsMargins(10, 10, 10, 10)
        self.models = QtWidgets.QTableWidget(0, 6)
        self.models.setHorizontalHeaderLabels(
            ['模型', '配置', 'Fold', '输出标签', '创建时间', '可用']
        )
        self.models.horizontalHeader().setStretchLastSection(True)
        self.models.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.Stretch
        )
        self.models.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.models.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        models_layout.addWidget(self.models)
        right.addTab(models_tab, "模型")
        curve_tab = QtWidgets.QWidget()
        curve_layout = QtWidgets.QVBoxLayout(curve_tab)
        curve_layout.setContentsMargins(10, 10, 10, 10)
        self.curve = QtWidgets.QLabel(
            "Trainer 写入 progress.png 后，nnU-Net 训练曲线会显示在这里。"
        )
        self.curve.setObjectName("hint")
        self.curve.setAlignment(QtCore.Qt.AlignCenter)
        self.curve.setScaledContents(False)
        curve_layout.addWidget(self.curve, 1)
        right.addTab(curve_tab, "训练曲线")
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, 1)

        actions = QtWidgets.QHBoxLayout()
        self.stop_button = QtWidgets.QPushButton("停止")
        self.stop_button.clicked.connect(self._stop)
        actions.addWidget(self.stop_button)
        self.reattach_button = QtWidgets.QPushButton("重新挂接")
        self.reattach_button.clicked.connect(self._reattach)
        self.reattach_button.setToolTip(
            "恢复监控本地控制器已停止的远程任务（例如工作站重启后）。"
        )
        actions.addWidget(self.reattach_button)
        self.retry_button = QtWidgets.QPushButton("以相同设置重试")
        self.retry_button.clicked.connect(self._retry)
        actions.addWidget(self.retry_button)
        self.open_button = QtWidgets.QPushButton("打开作业文件夹")
        self.open_button.clicked.connect(self._open_folder)
        actions.addWidget(self.open_button)
        actions.addStretch(1)
        close = QtWidgets.QPushButton("关闭")
        close.clicked.connect(self.window.close)
        actions.addWidget(close)
        root.addLayout(actions)
        self.window.setCentralWidget(central)
        self._job_rows = []
        self._model_rows = []
        self._populate_tasks()
        # Periodic refresh collects off the GUI thread: list_jobs +
        # load_models + log tails can stall on a network workspace.
        self.refresher = BackgroundRefresh(
            self.QtCore,
            parent=self.window,
            interval_ms=2000,
            collect=lambda: self._collect(),
            apply=self.refresh,
        )
        self.refresher.refresh_now()

    def _populate_tasks(self):
        wanted = str(self.context.get("task_id") or "")
        tasks = {}
        for row in list_jobs(self.workspace):
            task_id = str(row.get("task_id") or "Unassigned")
            tasks[task_id] = str(row.get("task_name") or task_id)
        for row in load_models(self.workspace, include_missing=True):
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

    def _selected_task(self):
        return str(self.task_combo.currentData() or "")

    def _collect(self):
        """Snapshot everything the UI needs (runs on a worker thread)."""
        return {
            "jobs": list_jobs(self.workspace),
            "models": load_models(self.workspace, include_missing=True),
        }

    def refresh(self, snapshot=None):
        if snapshot is None:
            # Manual/periodic path: collect synchronously (tests, first paint).
            snapshot = self._collect()
        task_id = self._selected_task()
        if not task_id:
            self.state_label.setText("未找到 nnU-Net 任务")
            self.detail_label.setText("开始训练即可创建第一个托管任务。")
            self.progress.setValue(0)
            return
        current_status_path = ""
        if 0 <= self.jobs.currentRow() < len(self._job_rows):
            current_status_path = str(self._job_rows[self.jobs.currentRow()].get("status_path") or "")
        rows = [row for row in snapshot["jobs"] if str(row.get("task_id") or "") == task_id]
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
        self._model_rows = [
            row
            for row in snapshot["models"]
            if str(row.get("task_id") or "") == task_id
        ]
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
        self.stop_button.setText("Abandon Locally" if abandonable else "停止")
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
                "训练曲线暂不可用。实时 epoch 详情仍见“日志”页。"
            )
            return
        pixmap = self.QtGui.QPixmap(str(curve_path))
        if pixmap.isNull():
            self.curve.setText("训练曲线无法解码。")
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
                    "在本机放弃远程任务",
                    "不再在本机等待该任务？\n\n服务器无法确认容器是否已停止，它可能仍在占用 GPU 或磁盘资源。需要管理员根据记录的容器名进行检查。",
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
        status = self._job_rows[index]
        try:
            reattach_job(status["status_path"])
            self.refresh()
        except Exception as exc:
            self.QtWidgets.QMessageBox.critical(
                self.window, "重新挂接失败", str(exc)
            )

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
                self.window, "重试失败", str(exc)
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
    configure_application(app, "nnU-Net 状态")
    app.setStyleSheet(stylesheet())
    window = StatusWindow(context, (QtCore, QtGui, QtWidgets))
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
