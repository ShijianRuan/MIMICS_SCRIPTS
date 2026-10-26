#!/usr/bin/env python3
"""External live status and model viewer for managed nnU-Net tasks."""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
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
    restore_replaced_output,
    reveal_in_file_manager,
    stop_job,
)
from ui_theme import (  # noqa: E402
    clamp_window_to_screen,
    configure_application,
    stylesheet,
)

# P3-1: 作业状态词对标注者显示为中文（此前是 "Uploading" 这类英文标题体）。
# "最近运行"列表行同样用这两个映射（此前是 "Training · Completed" 英文）。
_KIND_LABELS = {
    "train": "训练",
    "infer": "预测",
}
_STATUS_LABELS = {
    "running": "运行中",
    "launching": "启动中",
    "preparing_data": "准备数据中",
    "preparing_remote": "准备远程任务中",
    "connecting_remote": "连接远程服务器中",
    "starting_remote": "启动远程任务中",
    "monitoring_remote": "远程任务监控中",
    "reconnecting_remote": "重连远程服务器中",
    "remote_control_unavailable": "远程连接异常",
    "waiting_for_remote_gpu": "等待远程 GPU",
    "finalizing_remote": "远程收尾中",
    "remote_unreachable": "远程服务器不可达",
    "orphaned_remote": "远程任务失联",
    "reattaching": "重新接管远程任务中",
    "remote_termination_pending": "等待远程任务终止",
    "stopping": "停止中",
    "cancelling": "取消中",
    "pausing": "暂停中",
    "paused": "已暂停",
    "preprocessing": "预处理中",
    "training": "训练中",
    "uploading": "上传中",
    "downloading": "下载中",
    "exporting_labels": "导出标注中",
    "computing_uncertainty": "计算不确定度中",
    "registering": "配准中",
    "batch_predicting": "批量预测中",
    "predicting": "预测中",
    "finalizing": "收尾中",
    "waiting_for_dataset": "等待数据集",
    "waiting_for_gpu": "等待 GPU",
    "completed": "已完成",
    "completed_with_errors": "完成（有失败）",
    "failed": "失败",
    "cancelled": "已取消",
    "abandoned": "已放弃",
    "unknown": "未知",
}
from viewer_refresh import BackgroundRefresh, read_log_tail  # noqa: E402


def _format_time(value):
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(value)))
    except Exception:
        return "-"


def _validation_dice(model):
    """Mean foreground Dice from the fold's validation summary, if present.

    Local training always writes fold_*/validation/summary.json; remote
    downloads carry it too. Missing/unreadable returns "" so the table
    cell simply reads "-" and no error path interrupts the status view.
    """
    model_dir = Path(str(model.get("model_dir") or ""))
    if not model_dir.is_dir():
        return ""
    scores = []
    for fold in model.get("folds") or ["0"]:
        summary_path = (
            model_dir / ("fold_" + str(fold)) / "validation" / "summary.json"
        )
        try:
            summary = read_json(summary_path, {}) or {}
            scores.append(float(summary["foreground_mean"]["Dice"]))
        except Exception:
            continue
    return "{:.3f}".format(sum(scores) / len(scores)) if scores else ""


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
        # F37: 150% DPI screens can be smaller than the layout minimum.
        clamp_window_to_screen(self.window, (800, 620))
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
        self.live_label = QtWidgets.QLabel("● 实时")
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
        self.models = QtWidgets.QTableWidget(0, 8)
        self.models.setHorizontalHeaderLabels(
            ['模型', '配置', 'Fold', '输出标签', '验证 Dice', '创建时间', '来源', '可用']
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
        # S02: the recovery entry for a replaced previous result — the
        # annotator must not have to read JSON or guess file names to get
        # the old version back.
        self.previous_button = QtWidgets.QPushButton("旧结果…")
        self.previous_button.clicked.connect(self._previous_output)
        self.previous_button.setToolTip(
            "打开或恢复本次预测替换掉的旧结果。"
        )
        actions.addWidget(self.previous_button)
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
        self._log_tail_cache = {}
        self._populate_tasks()
        # Periodic refresh collects off the GUI thread: list_jobs +
        # load_models + log tails can stall on a network workspace.
        self.refresher = BackgroundRefresh(
            self.QtCore,
            parent=self.window,
            interval_ms=2000,
            collect=lambda: self._collect(),
            apply=self.refresh,
            # F34: a failed scan keeps the last frame — mark it so the
            # timestamp doesn't read as fresh.
            on_error=lambda: self.live_label.setText("● 上次刷新失败"),
        )
        self.refresher.refresh_now()
        # U02: restoring an old output runs voxel checks + checksums + a
        # file swap; on a slow workspace that stalls the GUI thread. The
        # swap runs on a worker, the result is polled back on the GUI thread.
        self._restore_results = queue.Queue()
        self._restore_busy = False
        self._restore_timer = self.QtCore.QTimer(self.window)
        self._restore_timer.setInterval(100)
        self._restore_timer.timeout.connect(self._poll_restore_result)
        # A closed window must stop collecting: the hidden heartbeat kept
        # polling forever, and timers firing while the window is torn down
        # corrupt the heap.
        self.window.closeEvent = self._close_event

    def _close_event(self, event):
        if self._restore_busy:
            self._close_after_restore = True
            self.window.setWindowTitle('正在完成结果恢复，完成后自动关闭')
            event.ignore()
            return
        self.refresher.stop()
        self._restore_timer.stop()
        # Hand the C++ widget tree to Qt: leaving it for Python GC to
        # destroy mid-collect corrupted the heap (0xC0000374) and killed
        # whole gui_smoke runs.
        self.window.deleteLater()

    def _populate_tasks(self, snapshot=None):
        """Rebuild the task combo, keeping the user's selection.

        Called once at construction and again on every refresh: a task
        started after the window opened must appear in the combo without
        reopening the window."""
        current = self._selected_task() or str(self.context.get("task_id") or "")
        tasks = {}
        for row in (snapshot or {}).get("jobs") or list_jobs(self.workspace):
            task_id = str(row.get("task_id") or "Unassigned")
            tasks[task_id] = str(row.get("task_name") or task_id)
        for row in (snapshot or {}).get("models") or load_models(
            self.workspace, include_missing=True
        ):
            task_id = str(row.get("task_id") or "Unassigned")
            tasks[task_id] = str(row.get("task_name") or task_id)
        self.task_combo.blockSignals(True)
        self.task_combo.clear()
        selected = 0
        for index, task_id in enumerate(sorted(tasks)):
            self.task_combo.addItem(tasks[task_id], task_id)
            if task_id == current:
                selected = index
        self.task_combo.setCurrentIndex(selected)
        self.task_combo.blockSignals(False)

    def _selected_task(self):
        return str(self.task_combo.currentData() or "")

    def _collect(self):
        """Snapshot everything the UI needs (runs on a worker thread)."""
        models = load_models(self.workspace, include_missing=True)
        # _validation_dice reads summary.json per fold per model; the log
        # tails read up to 512 KiB per job. Do both here, on the worker
        # thread, so the GUI thread never touches the workspace disk.
        for model in models:
            model["validation_dice"] = _validation_dice(model)
        jobs = list_jobs(self.workspace)
        for job in jobs:
            job["_log_tail"] = read_log_tail(job, self._log_tail_cache)
            # S02: the replaced-version timestamp for the detail line. Stat
            # here, on the worker thread — the GUI thread never touches the
            # workspace disk (铁律 1).
            replaced_path = str(job.get("replaced_previous_output") or "")
            if replaced_path:
                try:
                    job["_replaced_mtime_epoch"] = os.path.getmtime(replaced_path)
                except OSError:
                    pass
        return {
            "jobs": jobs,
            "models": models,
        }

    def refresh(self, snapshot=None):
        if snapshot is None:
            # Manual/periodic path: collect synchronously (tests, first paint).
            snapshot = self._collect()
        self._populate_tasks(snapshot)
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
                _KIND_LABELS.get(
                    str(row.get("kind") or ""),
                    str(row.get("kind") or "任务")),
                _STATUS_LABELS.get(
                    str(row.get("status") or ""),
                    str(row.get("status") or "未知").replace("_", " ")),
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
            # R62-9: surface where the model came from — a remote-trained
            # download is indistinguishable from a local one otherwise.
            if str(model.get("execution_backend") or "") == "remote":
                source = "远程（{0}）".format(
                    model.get("remote_profile_name")
                    or model.get("remote_profile_id")
                    or "remote",
                )
            else:
                source = "本地"
            values = [
                model.get("model_id", ""),
                model.get("configuration", ""),
                ", ".join(str(value) for value in model.get("folds") or []),
                ", ".join(
                    str(label.get("name") or "")
                    for label in model.get("labels") or []
                ),
                str(model.get("validation_dice") or "") or "-",
                _format_time(model.get("created_at_epoch")),
                source,
                "可用" if usable else "不可用",
            ]
            for column, value in enumerate(values):
                item = self.QtWidgets.QTableWidgetItem(str(value))
                if column == 3:
                    item.setToolTip(str(value))
                if not usable:
                    item.setToolTip(reason)
                self.models.setItem(row_index, column, item)
        self._show_selected_job()
        self.live_label.setText("● 实时 · {}".format(time.strftime("%H:%M:%S")))

    def _show_selected_job(self):
        index = self.jobs.currentRow()
        status = self._job_rows[index] if 0 <= index < len(self._job_rows) else {}
        state = str(status.get("status") or "unknown")
        self.state_label.setText(
            _STATUS_LABELS.get(state, state.replace("_", " ").title())
        )
        self.detail_label.setText(
            str(status.get("message") or status.get("error") or status.get("phase") or "")
        )
        if state in {"uploading", "downloading"}:
            speed = float(status.get("transfer_bytes_per_second") or 0.0)
            eta = status.get("transfer_eta_seconds")
            detail = "{} {}%".format(
                "上传中" if state == "uploading" else "下载中",
                int(status.get("transfer_percent") or 0),
            )
            if speed > 0:
                detail += "，速度 {:.1f} MiB/s".format(speed / float(1024 ** 2))
            if eta is not None:
                detail += "，约剩 {} 分钟".format(
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
        self.stop_button.setText("在本机放弃远程任务" if abandonable else "停止")
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
        # S02: surface what was replaced and how to get it back — the
        # replaced pointer only means recovery while the annotator can see it.
        replaced = str(status.get("replaced_previous_output") or "")
        self.previous_button.setEnabled(bool(replaced))
        if replaced:
            replaced_time = _format_time(status.get("_replaced_mtime_epoch"))
            self.detail_label.setText(
                "{}\n本次预测替换了旧结果（{}）。旧结果已保留，可用「旧结果…」打开或恢复。".format(
                    self.detail_label.text(), replaced_time
                )
            )
        scrollbar = self.log_view.verticalScrollBar()
        follow_tail = scrollbar.value() >= scrollbar.maximum() - 4
        # The tail itself is read on the collect worker thread (see
        # _collect) — the GUI thread only paints it.
        self.log_view.setPlainText(
            str(status.get("_log_tail") or ""))
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
            # Post-action refresh goes through the background
            # refresher: _collect() stats every job directory and can
            # stall the GUI thread on a network workspace (铁律 1).
            self.refresher.request()

    def _reattach(self):
        index = self.jobs.currentRow()
        if not 0 <= index < len(self._job_rows):
            return
        status = self._job_rows[index]
        try:
            reattach_job(status["status_path"])
            # Post-action refresh goes through the background
            # refresher: _collect() stats every job directory and can
            # stall the GUI thread on a network workspace (铁律 1).
            self.refresher.request()
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
            # Post-action refresh goes through the background
            # refresher: _collect() stats every job directory and can
            # stall the GUI thread on a network workspace (铁律 1).
            self.refresher.request()
        except Exception as exc:
            self.QtWidgets.QMessageBox.critical(
                self.window, "重试失败", str(exc)
            )

    def _previous_output(self):
        index = self.jobs.currentRow()
        if not 0 <= index < len(self._job_rows):
            return
        status = self._job_rows[index]
        replaced = str(status.get("replaced_previous_output") or "")
        if not replaced:
            return
        QtWidgets = self.QtWidgets
        box = QtWidgets.QMessageBox(self.window)
        box.setWindowTitle("旧结果")
        box.setText(
            "本次预测替换了一个旧结果：\n{}\n（旧版本时间：{}）\n\n"
            "旧结果文件已保留。可以直接打开它所在位置查看，"
            "或把它恢复为当前结果（当前版本会同样保留，不会丢失）。".format(
                replaced, _format_time(status.get("_replaced_mtime_epoch"))
            )
        )
        open_button = box.addButton("打开位置", QtWidgets.QMessageBox.ActionRole)
        restore_button = box.addButton("恢复旧版本", QtWidgets.QMessageBox.AcceptRole)
        box.addButton("取消", QtWidgets.QMessageBox.RejectRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked is open_button:
            reveal_in_file_manager(replaced)
        elif clicked is restore_button:
            self._restore_previous(status)

    def _restore_previous(self, status):
        if self._restore_busy:
            self.QtWidgets.QMessageBox.information(
                self.window, "正在恢复", "上一个恢复操作仍在进行中，请稍候。"
            )
            return
        self._restore_busy = True

        def worker():
            try:
                restore_replaced_output(status["status_path"])
                self._restore_results.put(None)
            except Exception as exc:
                self._restore_results.put(str(exc))

        # A restore transaction must outlive its window and finish even
        # when the application receives a normal quit request.
        thread = threading.Thread(target=worker)
        thread.daemon = False
        thread.start()
        self._restore_timer.start()

    def _poll_restore_result(self):
        try:
            error = self._restore_results.get_nowait()
        except queue.Empty:
            return
        self._restore_timer.stop()
        self._restore_busy = False
        close_after_restore = getattr(self, "_close_after_restore", False)
        self._close_after_restore = False
        if close_after_restore and error is None:
            self.window.close()
            return
        if error is None:
            self.QtWidgets.QMessageBox.information(
                self.window, "已恢复旧结果", "旧结果已恢复为当前结果。"
            )
        else:
            self.QtWidgets.QMessageBox.critical(
                self.window, "恢复失败", error
            )
        # Post-action refresh goes through the background refresher (铁律 1).
        self.refresher.request()

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
