#!/usr/bin/env python3
"""External, non-modal FlexiCT few-shot training configuration window.

The training recipe (optimizer, learning rates, batch size, fp32, TTA) is
locked to the validated FlexiCT few-shot configuration — only data selection,
configuration (2D/3D/pair), epochs, and validation cases are exposed.
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

from flexict_common import (  # noqa: E402
    CONFIGURATIONS,
    load_config,
    load_models,
    model_usability,
    recommended_model,
    workspace_root,
)
from flexict_pipeline import create_flexict_job  # noqa: E402
from nnunet_common import read_json, update_status, write_json_atomic  # noqa: E402
from ui_theme import (  # noqa: E402
    choose_existing_directory_async,
    configure_application,
    stylesheet,
)


class TrainingSetupWindow:
    def __init__(self, context: dict, context_path: Path, qt_modules):
        self.context = context
        self.context_path = context_path
        self.QtCore, self.QtGui, self.QtWidgets = qt_modules
        QtCore, QtWidgets = self.QtCore, self.QtWidgets
        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle("FlexiCT 训练")
        self.window.resize(880, 720)
        self.window.setMinimumSize(760, 620)
        self.window.closeEvent = self._close_event
        self.submitted = False
        self._submission_pending = False
        self._submission_results = Queue()
        self._dataset_id_results = Queue()
        self._submission_generation = 0
        self._submission_deadline = 0.0
        self._case_rows: list[dict] = []

        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(22, 18, 22, 18)
        root.setSpacing(12)
        title = QtWidgets.QLabel("FlexiCT 训练")
        title.setObjectName("title")
        subtitle = QtWidgets.QLabel(
            "基于 FlexiCT ViT 主干、只需少量标注病例的小样本训练（2~10 例即可）。期间 Mimics 仍可正常使用。"
        )
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(subtitle)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._build_data_tab(), "数据")
        self.tabs.addTab(self._build_training_tab(), "训练")
        self.tabs.addTab(self._build_compute_tab(), "计算")
        self.tabs.addTab(self._build_models_tab(), "已有模型")
        root.addWidget(self.tabs, 1)

        self.status_label = QtWidgets.QLabel(
            "标签会先映射回原始图像网格，再进行 nnU-Net 预处理。"
        )
        self.status_label.setObjectName("hint")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)
        actions = QtWidgets.QHBoxLayout()
        actions.addStretch(1)
        cancel = QtWidgets.QPushButton("取消")
        cancel.clicked.connect(self.window.close)
        actions.addWidget(cancel)
        self.start_button = QtWidgets.QPushButton("开始训练")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(self._submit)
        actions.addWidget(self.start_button)
        root.addLayout(actions)
        self.window.setCentralWidget(central)
        self._load_context()
        self._submission_timer = self.QtCore.QTimer(self.window)
        self._submission_timer.timeout.connect(self._poll_submission)
        self._submission_timer.start(80)
        self._refresh_cases()

    def _surface(self, title):
        QtWidgets = self.QtWidgets
        group = QtWidgets.QGroupBox(title)
        layout = QtWidgets.QGridLayout(group)
        layout.setContentsMargins(16, 18, 16, 14)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(10)
        layout.setColumnMinimumWidth(0, 180)
        layout.setColumnStretch(1, 1)
        return group, layout

    def _path_row(self, edit, title, directory=True):
        QtWidgets = self.QtWidgets
        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(edit, 1)
        button = QtWidgets.QPushButton("浏览...")

        def browse():
            current = edit.text().strip()
            choose_existing_directory_async(
                self.QtCore,
                QtWidgets,
                self.window,
                title,
                current,
                lambda value: edit.setText(str(value)) if value else None,
                button=button,
            )

        button.clicked.connect(browse)
        layout.addWidget(button)
        return container

    def _build_data_tab(self):
        QtWidgets = self.QtWidgets
        tab = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(tab)
        outer.setContentsMargins(8, 10, 8, 8)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        body = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(body)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(12)

        project, form = self._surface("Task and images")
        self.label_edit = QtWidgets.QLineEdit()
        self.label_edit.setPlaceholderText("例如 kidney_left")
        self.label_edit.setToolTip(
            "一个 FlexiCT 模型只训练一个二值目标（器官或结构）。"
        )
        form.addWidget(QtWidgets.QLabel("目标 / 标签名称 *"), 0, 0)
        form.addWidget(self.label_edit, 0, 1)
        self.modality_combo = QtWidgets.QComboBox()
        for label, value in (("CT", "CT"), ("MRI", "MRI"), ("Other", "Other")):
            self.modality_combo.addItem(label, value)
        form.addWidget(QtWidgets.QLabel("成像模态"), 1, 0)
        form.addWidget(self.modality_combo, 1, 1)
        self.dataset_edit = QtWidgets.QLineEdit()
        self.dataset_edit.setPlaceholderText(
            "每个病例一个子文件夹的目录"
        )
        form.addWidget(QtWidgets.QLabel("原始图像数据集 *"), 2, 0)
        form.addWidget(
            self._path_row(self.dataset_edit, "Select original image dataset"), 2, 1
        )
        self.dataset_edit.textChanged.connect(self._refresh_cases)
        layout.addWidget(project)

        cases, cases_layout = self._surface("Cases and validation")
        hint = QtWidgets.QLabel(
            "每个病例文件夹需要原始图像和一个以目标命名的 Mask 文件（例如 kidney_left.nii.gz）。其余用于训练；验证病例按层位均匀抽取。"
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        cases_layout.addWidget(hint, 0, 0, 1, 2)
        self.case_table = QtWidgets.QTableWidget(0, 4)
        self.case_table.setHorizontalHeaderLabels(
            ['病例', '用于训练', '用于验证', '状态']
        )
        self.case_table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.Stretch
        )
        self.case_table.horizontalHeader().setSectionResizeMode(
            3, QtWidgets.QHeaderView.Stretch
        )
        self.case_table.setMinimumHeight(180)
        self.case_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.case_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.case_table.setAlternatingRowColors(True)
        cases_layout.addWidget(self.case_table, 1, 0, 1, 2)
        self.rescan_button = QtWidgets.QPushButton("重新扫描病例")
        self.rescan_button.clicked.connect(self._rescan_cases_async)
        cases_layout.addWidget(self.rescan_button, 2, 0)
        self.case_summary = QtWidgets.QLabel("")
        self.case_summary.setObjectName("hint")
        self.case_summary.setWordWrap(True)
        cases_layout.addWidget(self.case_summary, 2, 1)
        layout.addWidget(cases)

        storage, storage_form = self._surface("Storage")
        self.workspace_edit = QtWidgets.QLineEdit()
        storage_form.addWidget(QtWidgets.QLabel("模型库"), 0, 0)
        storage_form.addWidget(
            self._path_row(self.workspace_edit, "Select FlexiCT model library"), 0, 1
        )
        storage_hint = QtWidgets.QLabel(
            "包含托管作业、预处理缓存和模型注册表。"
        )
        storage_hint.setObjectName("hint")
        storage_hint.setWordWrap(True)
        storage_form.addWidget(storage_hint, 1, 0, 1, 2)
        layout.addWidget(storage)
        layout.addStretch(1)
        scroll.setWidget(body)
        outer.addWidget(scroll)
        return tab

    def _build_training_tab(self):
        QtWidgets = self.QtWidgets
        tab = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(tab)
        outer.setContentsMargins(8, 10, 8, 8)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        body = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(body)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(12)

        training, form = self._surface("训练")
        self.configuration_combo = QtWidgets.QComboBox()
        for label, value in (
            ("Automatic (3D pair on large GPUs, 2D otherwise)", "auto"),
            ("2D slices", "2d"),
            ("3D full resolution", "3d_fullres"),
            ("2D + 3D pair (for active learning)", "pair"),
        ):
            self.configuration_combo.addItem(label, value)
        self.configuration_combo.setToolTip(
            "已验证的小样本配方。显存 16GB+ 的 GPU 上自动选择成对模型，否则选择 2D。"
        )
        form.addWidget(QtWidgets.QLabel("配置"), 0, 0)
        form.addWidget(self.configuration_combo, 0, 1)
        self.epochs_spin = QtWidgets.QSpinBox()
        self.epochs_spin.setRange(10, 2000)
        self.epochs_spin.setValue(150)
        self.epochs_spin.setToolTip(
            "已验证的小样本默认值：150 epoch，使用 FlexiCT 优化器调度。"
        )
        form.addWidget(QtWidgets.QLabel("训练轮数 (epoch)"), 1, 0)
        form.addWidget(self.epochs_spin, 1, 1)
        self.val_spin = QtWidgets.QSpinBox()
        self.val_spin.setRange(1, 10)
        self.val_spin.setValue(1)
        self.val_spin.setToolTip(
            "从所选病例中留出多少例用于验证（用于选择最佳 checkpoint）。"
        )
        form.addWidget(QtWidgets.QLabel("验证病例数"), 2, 0)
        form.addWidget(self.val_spin, 2, 1)
        self.mirror_combo = QtWidgets.QComboBox()
        self.mirror_combo.addItem("双侧（默认）", "")
        self.mirror_combo.addItem("单侧器官，关闭 X 轴镜像", "1")
        self.mirror_combo.setToolTip(
            "关闭左右镜像增广——单侧器官（如单个肾脏）需要此设置。"
        )
        form.addWidget(QtWidgets.QLabel("镜像"), 3, 0)
        form.addWidget(self.mirror_combo, 3, 1)
        self.workers_spin = QtWidgets.QSpinBox()
        self.workers_spin.setRange(1, 64)
        self.workers_spin.setValue(4)
        form.addWidget(QtWidgets.QLabel("预处理进程数"), 4, 0)
        form.addWidget(self.workers_spin, 4, 1)
        layout.addWidget(training)

        recipe, recipe_form = self._surface("Locked recipe (read-only)")
        recipe_text = QtWidgets.QLabel(
            "FlexiCT ViT 主干（fp32，RoPE 安全）· AdamW 双参数组（backbone 3e-5 / decoder 3e-4，betas 0.9/0.98，wd 5e-2）· poly LR · gradclip 12 · 无 deep supervision · 前景过采样 0.33 · batch size 取自 nnU-Net plans · 按验证 Dice 选最佳 checkpoint · 推理不用 TTA。\n\n以上为已验证的小样本配方，Mimics 中不可修改。"
        )
        recipe_text.setObjectName("hint")
        recipe_text.setWordWrap(True)
        recipe_form.addWidget(recipe_text, 0, 0, 1, 2)
        layout.addWidget(recipe)
        layout.addStretch(1)
        scroll.setWidget(body)
        outer.addWidget(scroll)
        return tab

    def _build_models_tab(self):
        QtWidgets = self.QtWidgets
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(12, 14, 12, 12)
        self.models_table = QtWidgets.QTableWidget(0, 5)
        self.models_table.setHorizontalHeaderLabels(
            ['模型', '目标', "配置", '创建时间', '可用']
        )
        self.models_table.horizontalHeader().setStretchLastSection(True)
        self.models_table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.Stretch
        )
        self.models_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.models_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.models_table)
        hint = QtWidgets.QLabel(
            "已注册的 FlexiCT 模型。最近一个可用模型会预选用于预测和主动学习。"
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch(1)
        return tab

    def _build_compute_tab(self):
        QtWidgets = self.QtWidgets
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(12, 14, 12, 12)
        self.remote_selector = None
        try:
            from remote_compute_ui import RemoteComputeSelector

            self.remote_selector = RemoteComputeSelector(
                self.window, (self.QtCore, self.QtGui, self.QtWidgets)
            )
            layout.addWidget(self.remote_selector.group)
        except Exception:
            layout.addWidget(
                QtWidgets.QLabel("本工作站不可用远程计算。")
            )
        note = QtWidgets.QLabel(
            "本机与远程使用相同的请求、原始网格准备、配方、模型清单和状态格式。远程容器用后即弃；已验证的数据和预处理缓存仍可复用。"
        )
        note.setObjectName("hint")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch(1)
        return tab

    def _load_context(self):
        config = load_config()
        default_workspace = workspace_root(config)
        # Paths from the last successful submission are the default; the
        # context from the open Mimics project always wins when present.
        remembered = read_json(
            Path.home() / ".mimics_script" / "flexict_settings.json", {}
        ) or {}
        workspace = str(
            self.context.get("workspace")
            or remembered.get("workspace")
            or default_workspace
        )
        self.workspace_edit.setText(workspace)
        self.dataset_edit.setText(
            str(self.context.get("dataset_root") or remembered.get("dataset_root") or "")
        )
        selected = [
            str(value) for value in self.context.get("selected_mask_names") or []
        ]
        if len(selected) == 1:
            self.label_edit.setText(selected[0])
        elif remembered.get("label_name"):
            self.label_edit.setText(str(remembered["label_name"]))
        self._refresh_models()
        # An empty case table on a pre-filled dataset path reads as "the path
        # is wrong" - it usually just means nobody pressed Rescan yet. Run
        # the existing background scan once when both path and label are
        # already known (C6-6).
        if (
            self.dataset_edit.text().strip()
            and self.label_edit.text().strip()
        ):
            self._rescan_cases_async()

    def _refresh_models(self):
        QtWidgets = self.QtWidgets
        workspace = self.workspace_edit.text().strip()
        self.models_table.setRowCount(0)
        for model in load_models(workspace):
            usable, _why = model_usability(model)
            row = self.models_table.rowCount()
            self.models_table.insertRow(row)
            values = [
                model.get("model_id", ""),
                model.get("label_name", ""),
                model.get("configuration", ""),
                time.strftime(
                    "%Y-%m-%d %H:%M", time.localtime(model.get("created_at_epoch") or 0)
                ),
                "Yes" if usable else "No",
            ]
            for column, value in enumerate(values):
                self.models_table.setItem(row, column, QtWidgets.QTableWidgetItem(str(value)))

    def _refresh_cases(self):
        QtWidgets = self.QtWidgets
        self.case_table.setRowCount(0)
        rows = [row for row in self._case_rows if row.get("checked")]
        for row in rows:
            index = self.case_table.rowCount()
            self.case_table.insertRow(index)
            self.case_table.setItem(
                index, 0, QtWidgets.QTableWidgetItem(row["case_id"])
            )
            use_check = QtWidgets.QCheckBox()
            use_check.setChecked(bool(row.get("use_train", True)))
            use_check.toggled.connect(
                lambda checked, r=row: r.update({"use_train": checked})
            )
            self.case_table.setCellWidget(index, 1, use_check)
            val_check = QtWidgets.QCheckBox()
            val_check.setChecked(bool(row.get("use_val", False)))
            val_check.toggled.connect(
                lambda checked, r=row: r.update({"use_val": checked})
            )
            self.case_table.setCellWidget(index, 2, val_check)
            self.case_table.setItem(
                index, 3, QtWidgets.QTableWidgetItem(row.get("status", ""))
            )
        train_count = sum(1 for row in rows if row.get("use_train", True))
        val_count = sum(1 for row in rows if row.get("use_val", False))
        self.case_summary.setText(
            "{} case(s) selected · {} training · {} validation".format(
                len(rows), train_count, val_count
            )
        )

    def _scan_cases(self, dataset_root: str, label_name: str) -> list[dict]:
        from flexict_pipeline import scan_flexict_cases

        return scan_flexict_cases(dataset_root, label_name)

    def _rescan_cases_async(self):
        if getattr(self, "_scan_pending", False):
            return
        self._scan_pending = True
        self.rescan_button.setEnabled(False)
        dataset_root = self.dataset_edit.text().strip()
        label_name = self.label_edit.text().strip()

        def scan():
            try:
                result = self._scan_cases(dataset_root, label_name)
                self._dataset_id_results.put(("scan", result, ""))
            except Exception as exc:
                self._dataset_id_results.put(("scan", None, str(exc)))

        worker = threading.Thread(target=scan, name="flexict-case-scan")
        worker.daemon = True
        worker.start()

    def _poll_submission(self):
        try:
            kind, payload, error = self._dataset_id_results.get_nowait()
        except Empty:
            pass
        else:
            self._scan_pending = False
            self.rescan_button.setEnabled(True)
            if kind == "scan":
                if error:
                    self.status_label.setStyleSheet("color: #b42318; font-weight: 600;")
                    self.status_label.setText(error)
                else:
                    self._case_rows = payload or []
                    self._refresh_cases()
                    self.status_label.setStyleSheet("")
                    self.status_label.setText(
                        "Found {} case(s). Verify the training/validation split, then start training.".format(
                            len(self._case_rows)
                        )
                    )
        if (
            self._submission_pending
            and self._submission_deadline
            and time.time() >= self._submission_deadline
        ):
            self._submission_generation += 1
            self._submission_pending = False
            self._submission_deadline = 0.0
            self.start_button.setEnabled(True)
            self.status_label.setStyleSheet("color: #b42318; font-weight: 600;")
            self.status_label.setText(
                "路径检查超时。所选磁盘可能离线或响应过慢；请粘贴更具体的路径或重试。"
            )
        try:
            generation, request, error = self._submission_results.get_nowait()
        except Empty:
            return
        if generation != self._submission_generation:
            return
        self._submission_pending = False
        self._submission_deadline = 0.0
        if error:
            self.start_button.setEnabled(True)
            self.status_label.setStyleSheet("color: #b42318; font-weight: 600;")
            self.status_label.setText(error)
            return
        try:
            self.status_label.setText("正在提交 FlexiCT 训练...")
            write_json_atomic(
                Path.home() / ".mimics_script" / "flexict_settings.json",
                {
                    "schema_version": "mimics_flexict_settings.v1",
                    "workspace": request["workspace"],
                    "dataset_root": request["dataset_root"],
                    "label_name": request["label_name"],
                    "updated_at_epoch": time.time(),
                },
            )
            job = create_flexict_job(request)
            self.submitted = True
            setup_status = self.context.get("setup_status_path")
            if setup_status:
                update_status(
                    setup_status,
                    status="training_started",
                    phase="training_started",
                    training_job_id=job["job_id"],
                    training_status_path=job["status_path"],
                    message="FlexiCT training started outside Mimics.",
                    completed_at_epoch=time.time(),
                )
            self.window.close()
        except Exception as exc:
            self.start_button.setEnabled(True)
            self.status_label.setStyleSheet("color: #b42318; font-weight: 600;")
            self.status_label.setText(str(exc))

    def _request(self):
        rows = [row for row in self._case_rows if row.get("checked")]
        train_cases = [
            row["case_id"] for row in rows if row.get("use_train", True)
        ]
        val_cases = [
            row["case_id"] for row in rows if row.get("use_val", False)
        ]
        # Cases neither selected are dropped entirely.
        cases = [row["case_id"] for row in rows if row.get("use_train") or row.get("use_val")]
        if not val_cases:
            val_count = int(self.val_spin.value())
        else:
            val_count = len(val_cases)
        request = {
            "operation": "train",
            "task_name": self.label_edit.text().strip(),
            "task_id": self.label_edit.text().strip(),
            "label_name": self.label_edit.text().strip(),
            "workspace": self.workspace_edit.text().strip(),
            "dataset_root": self.dataset_edit.text().strip(),
            "label_source": "dataset_masks",
            "modality": str(self.modality_combo.currentData()),
            "configuration": str(self.configuration_combo.currentData()),
            "epochs": int(self.epochs_spin.value()),
            "mirror_disable_axes": str(self.mirror_combo.currentData()),
            "val_cases": val_count,
            "preprocess_workers": int(self.workers_spin.value()),
            "cases": cases,
            "val_case_ids": val_cases,
            "train_case_ids": train_cases,
            "mimics_exe": self.context.get("mimics_exe") or "",
        }
        if self.remote_selector is not None:
            backend, profile_id = self.remote_selector.selection()
            request["execution_backend"] = backend
            request["remote_profile_id"] = profile_id
        return request

    def _submit(self):
        if self._submission_pending:
            return
        try:
            request = self._request()
            if not request["label_name"]:
                raise ValueError("Target / label name is required.")
            self._submission_generation += 1
            generation = self._submission_generation
            self._submission_pending = True
            self._submission_deadline = time.time() + 60.0
            self.start_button.setEnabled(False)
            self.status_label.setStyleSheet("")
            self.status_label.setText(
                "正在后台检查所选数据位置..."
            )

            def validate():
                try:
                    if not Path(request["dataset_root"]).is_dir():
                        raise ValueError("Original image dataset does not exist.")
                    if len(request["cases"]) < 2:
                        # The copy says "at least two"; the pipeline's train/
                        # val split needs both, or training runs empty.
                        raise ValueError(
                            "Select at least two cases (rescan the dataset first)."
                        )
                    self._submission_results.put((generation, request, ""))
                except Exception as exc:
                    self._submission_results.put((generation, None, str(exc)))

            worker = threading.Thread(target=validate, name="flexict-path-check")
            worker.daemon = True
            worker.start()
        except Exception as exc:
            self._submission_pending = False
            self.start_button.setEnabled(True)
            self.status_label.setObjectName("errorLabel")
            self.status_label.setStyleSheet("color: #b42318; font-weight: 600;")
            self.status_label.setText(str(exc))

    def _close_event(self, event):
        self._submission_generation += 1
        self._submission_pending = False
        if not self.submitted:
            setup_status = self.context.get("setup_status_path")
            if setup_status:
                update_status(
                    setup_status,
                    status="cancelled",
                    phase="cancelled",
                    message="FlexiCT training setup was closed.",
                    completed_at_epoch=time.time(),
                )
        event.accept()

    def show(self):
        self.window.show()
        self.window.raise_()
        self.window.activateWindow()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", required=True)
    args = parser.parse_args()
    context_path = Path(args.context).expanduser().resolve()
    context = read_json(context_path, {}) or {}
    try:
        from PySide6 import QtCore, QtGui, QtWidgets

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        configure_application(app, "FlexiCT 训练")
        app.setStyleSheet(stylesheet())
        window = TrainingSetupWindow(context, context_path, (QtCore, QtGui, QtWidgets))
        window.show()
        return int(app.exec())
    except Exception as exc:
        status_path = context.get("setup_status_path")
        if status_path:
            update_status(
                status_path,
                status="failed",
                phase="ui_failed",
                error="{}: {}".format(type(exc).__name__, exc),
                traceback=traceback.format_exc(),
                completed_at_epoch=time.time(),
            )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
