#!/usr/bin/env python3
"""External batch prediction window: run a registered nnU-Net model over a
directory of cases and write one NIfTI per case.

GUI twin of ``mimics_batch_cli.py predict``: the same case resolution
(ambiguous case folders are skipped, never guessed) and the same per-case
inference jobs. Predictions land on disk only; nothing is applied back to
Mimics, so progress and stopping live in the nnU-Net status window.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import traceback
from pathlib import Path
from queue import Empty, Queue


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "tools"):
    value = str(candidate)
    if value not in sys.path:
        sys.path.insert(0, value)

from mimics_batch_cli import _predict_cases  # noqa: E402
from nnunet_common import load_models, model_usability, read_json  # noqa: E402
from nnunet_jobs import create_job  # noqa: E402
from ui_theme import (  # noqa: E402
    choose_existing_directory_async,
    configure_application,
    stylesheet,
)


class BatchPredictionWindow:
    def __init__(self, context, qt_modules):
        self.context = context
        self.QtCore, self.QtGui, self.QtWidgets = qt_modules
        QtWidgets = self.QtWidgets
        self.window = QtWidgets.QDialog()
        self.window.setWindowTitle("nnU-Net 批量预测")
        self.window.resize(760, 600)
        self.window.setMinimumSize(680, 540)
        root = QtWidgets.QVBoxLayout(self.window)
        root.setContentsMargins(22, 18, 22, 18)
        root.setSpacing(12)
        title = QtWidgets.QLabel("nnU-Net 批量预测")
        title.setObjectName("title")
        subtitle = QtWidgets.QLabel(
            "选择一个已训练模型和数据集文件夹，对每个病例推理并把结果写入输出文件夹。"
            "预测只写盘、不改当前 Mimics 工程；进度与停止在「nnU-Net Show Status & Models」窗口。"
        )
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(subtitle)

        folders = QtWidgets.QFrame()
        folders.setObjectName("surface")
        folder_layout = QtWidgets.QGridLayout(folders)
        folder_layout.setContentsMargins(16, 14, 16, 14)
        self.image_root_edit = QtWidgets.QLineEdit()
        self.image_root_edit.setPlaceholderText("数据集文件夹（每例一个子目录或图像文件）")
        folder_layout.addWidget(QtWidgets.QLabel("病例文件夹"), 0, 0)
        folder_layout.addWidget(self.image_root_edit, 0, 1)
        self._add_browse(folder_layout, self.image_root_edit, 0, "选择病例文件夹")
        self.output_edit = QtWidgets.QLineEdit()
        self.output_edit.setPlaceholderText("预测结果输出文件夹（每例一个 .nii.gz）")
        folder_layout.addWidget(QtWidgets.QLabel("输出文件夹"), 1, 0)
        folder_layout.addWidget(self.output_edit, 1, 1)
        self._add_browse(folder_layout, self.output_edit, 1, "选择输出文件夹")
        root.addWidget(folders)

        self.models = []
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ['任务', '模型', '配置', '标签', '后端']
        )
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        root.addWidget(self.table, 1)
        options = QtWidgets.QHBoxLayout()
        self.tta_check = QtWidgets.QCheckBox("启用镜像 TTA")
        self.tta_check.setChecked(False)
        self.tta_check.setToolTip("可能提高精度，但会成倍增加每例推理时间。")
        options.addWidget(self.tta_check)
        options.addStretch(1)
        options.addWidget(QtWidgets.QLabel("图像模态"))
        self.modality_combo = QtWidgets.QComboBox()
        # R62-5: the batch window bypassed the modality guardrail the
        # single-case window enforces — a CT model silently ran over MR
        # folders (and vice versa). The declared modality feeds
        # source_modality, which the per-job validator hard-rejects on
        # mismatch.
        self.modality_combo.addItems(["自动（按模型）", "CT", "MR"])
        self.modality_combo.setToolTip(
            "声明数据集的真实模态。与模型训练模态不符时任务会被拒绝，"
            "而不是静默产出错误结果。"
        )
        options.addWidget(self.modality_combo)
        self.cpu_check = QtWidgets.QCheckBox("无 GPU 时用 CPU")
        self.cpu_check.setChecked(False)
        self.cpu_check.setToolTip(
            "本机没有可用 GPU 时允许用 CPU 推理（明显更慢，适合少量病例）。"
        )
        options.addWidget(self.cpu_check)
        root.addLayout(options)
        # R72-C1 (=R62-5): large models + large batches can outrun the
        # annotation workstation. The remote backend reuses the remote
        # training channel (same container, cache, and transfer machinery);
        # each case is still one tracked job in the status window.
        self.remote_selector = None
        try:
            from remote_compute_ui import RemoteComputeSelector

            self.remote_selector = RemoteComputeSelector(
                self.window, (self.QtCore, self.QtGui, self.QtWidgets)
            )
            root.addWidget(self.remote_selector.group)
        except Exception:
            pass
        self.status_label = QtWidgets.QLabel("正在后台加载兼容模型...")
        self.status_label.setObjectName("hint")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)
        actions = QtWidgets.QHBoxLayout()
        actions.addStretch(1)
        cancel = QtWidgets.QPushButton("关闭")
        cancel.clicked.connect(self.window.close)
        actions.addWidget(cancel)
        self.start = QtWidgets.QPushButton("开始批量预测")
        self.start.setObjectName("primary")
        self.start.clicked.connect(self._submit)
        self.start.setEnabled(False)
        actions.addWidget(self.start)
        root.addLayout(actions)

        self._loading_models = True
        self._model_deadline = time.time() + 120.0
        self._model_results = Queue()
        self._dispatch_results = Queue()
        self._dispatch_deadline = 0.0
        self._timer = self.QtCore.QTimer(self.window)
        self._timer.timeout.connect(self._poll_background_results)
        self._timer.start(80)
        self._start_model_loading()

    def _add_browse(self, layout, edit, row, title):
        QtWidgets = self.QtWidgets
        button = QtWidgets.QPushButton("浏览...")

        def browse():
            choose_existing_directory_async(
                self.QtCore,
                self.QtWidgets,
                self.window,
                title,
                edit.text().strip(),
                lambda value: edit.setText(str(value)) if value else None,
                button=button,
            )

        button.clicked.connect(browse)
        layout.addWidget(button, row, 2)

    def _start_model_loading(self):
        workspace = self.context.get("workspace") or Path.home() / ".mimics_script" / "nnunet"

        def load():
            try:
                all_models = load_models(workspace)
                # R62-7: unusable models stay visible with their reason
                # instead of silently vanishing from the list.
                rows = []
                for row in all_models:
                    usable, reason = model_usability(row)
                    if not usable:
                        row = dict(row)
                        row["_unusable_reason"] = reason
                    rows.append(row)
                self._model_results.put((rows, ""))
            except Exception as exc:
                self._model_results.put(([], str(exc)))

        worker = threading.Thread(target=load, name="nnunet-model-discovery")
        worker.daemon = True
        worker.start()

    def _apply_models(self, models, error):
        self._loading_models = False
        self.models = list(models or [])
        self.table.setUpdatesEnabled(False)
        try:
            self.table.setRowCount(len(self.models))
            first_usable = -1
            unusable_count = 0
            for index, model in enumerate(self.models):
                unusable_reason = str(model.get("_unusable_reason") or "")
                values = [
                    model.get("task_name") or model.get("task_id"),
                    model.get("model_id"),
                    model.get("configuration"),
                    ", ".join(
                        str(row.get("name")) for row in model.get("labels") or []
                    ),
                    unusable_reason or model.get("execution_backend") or "local",
                ]
                for column, value in enumerate(values):
                    item = self.QtWidgets.QTableWidgetItem(str(value or ""))
                    if column == 3:
                        item.setToolTip(str(value or ""))
                    if unusable_reason:
                        # R62-7: registered-but-broken models stay
                        # visible, greyed out with the reason in the last
                        # column — not silently dropped.
                        item.setToolTip(unusable_reason)
                        item.setFlags(
                            item.flags() & ~self.QtCore.Qt.ItemIsEnabled
                        )
                    self.table.setItem(index, column, item)
                if unusable_reason:
                    unusable_count += 1
                elif first_usable < 0:
                    first_usable = index
            if first_usable >= 0:
                self.table.selectRow(first_usable)
        finally:
            self.table.setUpdatesEnabled(True)
        self.start.setEnabled(any(
            not str(model.get("_unusable_reason") or "")
            for model in self.models
        ))
        if error:
            self.status_label.setStyleSheet("color: #b42318; font-weight: 600;")
            self.status_label.setText("Could not load nnU-Net models: {}".format(error))
        elif unusable_count:
            if any(
                not str(model.get("_unusable_reason") or "")
                for model in self.models
            ):
                self.status_label.setText(
                    "{0} 个已注册模型不可用（列表最后列注明原因）。先在「状态与模型」窗口修复或删除。".format(
                        unusable_count
                    )
                )
            else:
                self.status_label.setText(
                    "已注册的 {0} 个模型全部不可用（列表最后列注明原因）。".format(
                        unusable_count
                    )
                )
        elif self.models:
            self.status_label.setText(
                "同一个模型会预测其全部输出标签。内容不明确的病例子目录会被跳过并在提交后列出。"
            )
        else:
            self.status_label.setText(
                "未找到可用的 nnU-Net 模型。请先训练或导入一个模型。"
            )

    def _poll_background_results(self):
        if self._loading_models:
            try:
                models, error = self._model_results.get_nowait()
            except Empty:
                pass
            else:
                self._apply_models(models, error)
            if self._loading_models and time.time() >= self._model_deadline:
                self._loading_models = False
                self.status_label.setStyleSheet(
                    "color: #b42318; font-weight: 600;"
                )
                self.status_label.setText(
                    "模型发现超时。模型库所在磁盘可能离线或过慢。请关闭本窗口、重新连接磁盘后重试。"
                )
        if self._dispatch_deadline and time.time() >= self._dispatch_deadline:
            self._dispatch_deadline = 0.0
            self.start.setEnabled(bool(self.models))
            self.status_label.setStyleSheet("color: #b42318; font-weight: 600;")
            # R62-8: the timeout only means the scan was slow — some jobs
            # may already be dispatched and will keep running. Retrying
            # blindly duplicates them; check the status window first.
            self.status_label.setText(
                "病例文件夹扫描超时。扫描仍在后台进行，已启动的预测任务"
                "不会被取消。请先在「nnU-Net Show Status & Models」窗口"
                "确认哪些病例已启动；仅对未启动的病例重试，避免同一病例"
                "跑两个任务。"
            )
        try:
            summary = self._dispatch_results.get_nowait()
        except Empty:
            return
        self._dispatch_deadline = 0.0
        self.start.setEnabled(bool(self.models))
        self.status_label.setStyleSheet("")
        self.status_label.setText(summary)

    def _submit(self):
        try:
            row = self.table.currentRow()
            if not 0 <= row < len(self.models):
                raise RuntimeError("Select one model.")
            model = self.models[row]
            unusable_reason = str(model.get("_unusable_reason") or "")
            if unusable_reason:
                raise RuntimeError(
                    "所选模型不可用：{0}".format(unusable_reason)
                )
            image_root = self.image_root_edit.text().strip()
            output_root = self.output_edit.text().strip()
            if not image_root:
                raise RuntimeError("请先选择病例文件夹。")
            if not output_root:
                raise RuntimeError("请先选择输出文件夹。")
            workspace = str(self.context.get("workspace") or Path.home() / ".mimics_script" / "nnunet")
            model_manifest = str(model["manifest_path"])
            disable_tta = not self.tta_check.isChecked()
            modality = str(self.modality_combo.currentText() or "").strip()
            source_modality = "" if "自动" in modality else modality
            use_cpu = self.cpu_check.isChecked()
            execution_backend, remote_profile_id = "local", ""
            if self.remote_selector is not None:
                execution_backend, remote_profile_id = (
                    self.remote_selector.selection()
                )
            self.start.setEnabled(False)
            self.status_label.setStyleSheet("")
            self.status_label.setText("正在后台扫描病例并启动预测任务...")
            self._dispatch_deadline = time.time() + 600.0

            def dispatch():
                try:
                    cases, skipped = _predict_cases(image_root, None)
                    output = Path(output_root).expanduser().resolve()
                    output.mkdir(parents=True, exist_ok=True)
                    started, failed = [], []
                    for case_id, image_path in cases:
                        request = {
                            "operation": "infer",
                            "workspace": workspace,
                            "job_id": "predict_{}_{}".format(
                                time.strftime("%Y%m%dT%H%M%S"), case_id
                            ),
                            "task_id": str(model.get("task_id") or ""),
                            "task_name": str(
                                model.get("task_name") or model.get("task_id") or ""
                            ),
                            "dataset_id": int(model.get("dataset_id") or 0),
                            "model_manifest": model_manifest,
                            "image_path": str(image_path),
                            "output_path": str(output / "{}.nii.gz".format(case_id)),
                            "disable_tta": disable_tta,
                            "case_id": case_id,
                            "source_modality": source_modality,
                        }
                        if execution_backend == "remote":
                            request["execution_backend"] = "remote"
                            request["remote_profile_id"] = remote_profile_id
                        else:
                            # local CPU/GPU flags only apply locally; the
                            # remote channel always uses the server GPU.
                            request["use_cpu"] = use_cpu
                        try:
                            create_job(request)
                        except Exception as exc:
                            failed.append("{}: {}".format(case_id, exc))
                            continue
                        started.append(case_id)
                    lines = [
                        "已启动 {0} 个预测任务（模型 {1}，{2}）。输出文件夹：{3}".format(
                            len(started),
                            model.get("model_id"),
                            "远程" if execution_backend == "remote" else "本机",
                            output,
                        ),
                        "进度与停止：「nnU-Net Show Status & Models」窗口。",
                    ]
                    if skipped:
                        lines.append(
                            "跳过 {0} 个病例（找不到唯一的源图像）：{1}".format(
                                len(skipped), ", ".join(skipped)
                            )
                        )
                    if failed:
                        lines.append(
                            "{0} 个病例启动失败：{1}".format(
                                len(failed), "; ".join(failed)
                            )
                        )
                    self._dispatch_results.put("\n".join(lines))
                except Exception as exc:
                    self._dispatch_results.put(str(exc))

            worker = threading.Thread(target=dispatch, name="nnunet-batch-dispatch")
            worker.daemon = True
            worker.start()
        except Exception as exc:
            self.start.setEnabled(bool(self.models))
            self.status_label.setStyleSheet("color: #b42318; font-weight: 600;")
            self.status_label.setText(str(exc))

    def show(self):
        self.window.show()
        self.window.raise_()
        self.window.activateWindow()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", required=True)
    args = parser.parse_args()
    context = read_json(Path(args.context).expanduser().resolve(), {}) or {}
    try:
        from PySide6 import QtCore, QtGui, QtWidgets

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        configure_application(app, "nnU-Net 批量预测")
        app.setStyleSheet(stylesheet())
        window = BatchPredictionWindow(context, (QtCore, QtGui, QtWidgets))
        window.show()
        return int(app.exec())
    except Exception as exc:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
