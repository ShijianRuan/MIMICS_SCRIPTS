#!/usr/bin/env python3
"""External, non-modal FlexiCT few-shot training configuration window.

The training recipe (optimizer, learning rates, fp32, TTA) is locked to the
validated FlexiCT few-shot configuration — only data selection, configuration
(2D/3D/pair), epochs, validation cases, and the batch size override are
exposed. The batch size override exists because the nnU-Net planner sizes
batches for the training-time card, not for a 16GB remote GPU: without it an
OOM on the remote card leaves the annotator no self-rescue (R-P1-8).
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
from nnunet_common import (  # noqa: E402
    read_json,
    safe_identifier,
    update_status,
    write_json_atomic,
)
from ui_theme import (  # noqa: E402
    choose_existing_directory_async,
    clamp_window_to_screen,
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
        # F37: clamp the 620px minimum to the available screen height so
        # small high-DPI screens can still show the whole action bar.
        clamp_window_to_screen(self.window, (760, 620))
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
            "使用已有 CT 图像和标注训练任务模型。少样本效果取决于目标与数据，训练结果需要复核。"
        )
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(subtitle)
        hint = QtWidgets.QLabel(
            "如何选方法：训练可重复的分割任务 → nnU-Net；"
            "少量 CT 标注尝试任务训练 → FlexiCT；边点边标、交互修正 → nnInteractive。"
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        root.addWidget(hint)

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

        project, form = self._surface("任务与图像")
        self.label_edit = QtWidgets.QLineEdit()
        self.label_edit.setPlaceholderText("例如 kidney_left，或多个标签用逗号分隔")
        self.label_edit.setToolTip(
            "一个 FlexiCT 模型默认训练一个二值目标（器官或结构）。"
            "用逗号分隔多个标签（最多 4 个）可训练一个多分类模型。"
        )
        form.addWidget(QtWidgets.QLabel("目标 / 标签名称 *"), 0, 0)
        form.addWidget(self.label_edit, 0, 1)
        # Mask files are matched by safe_identifier(label_name): a Chinese
        # name folds to "item" and never matches any file, so the case scan
        # comes back with every case "label missing" (and the table renders
        # empty while the summary still counts the cases). Catch it here.
        self.label_name_hint = QtWidgets.QLabel("")
        self.label_name_hint.setStyleSheet(
            "color: #b42318; font-weight: 600;")
        self.label_name_hint.hide()
        form.addWidget(self.label_name_hint, 1, 0, 1, 2)

        def _check_label_names(_text):
            bad = [
                name for name in self._label_names()
                if safe_identifier(name) == "item"
            ]
            if bad:
                self.label_name_hint.setText(
                    "标签名仅支持英文字母、数字、下划线、中划线和点（要与数据集"
                    "里 Mask 文件的文件名一致；中文名永远匹配不到文件）。"
                )
                self.label_name_hint.show()
            else:
                self.label_name_hint.hide()

        self.label_edit.textChanged.connect(_check_label_names)
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
            self._path_row(self.dataset_edit, "选择原始图像数据集"), 2, 1
        )
        self.dataset_edit.textChanged.connect(self._refresh_cases)
        layout.addWidget(project)

        cases, cases_layout = self._surface("病例与验证")
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

        storage, storage_form = self._surface("存储")
        self.workspace_edit = QtWidgets.QLineEdit()
        storage_form.addWidget(QtWidgets.QLabel("模型库"), 0, 0)
        storage_form.addWidget(
            self._path_row(self.workspace_edit, "选择 FlexiCT 模型库"), 0, 1
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
            ("自动（大显存 GPU 用 3D 组合，否则 2D）", "auto"),
            ("仅 2D（逐层切片）", "2d"),
            ("仅 3D（全分辨率）", "3d_fullres"),
            ("2D + 3D 成对模型（主动学习用）", "pair"),
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
        self.auto_batch = QtWidgets.QCheckBox("由 nnU-Net 规划器自动决定 batch size")
        self.auto_batch.setChecked(True)
        self.auto_batch.setToolTip(
            "自动规划按训练卡显存取较大的 batch；远程小显存卡上首次前向即显存不足时，"
            "取消勾选并手动调小 batch size 重试。"
        )
        form.addWidget(self.auto_batch, 4, 0, 1, 2)
        self.batch_spin = QtWidgets.QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(6)
        self.batch_spin.setEnabled(False)
        self.auto_batch.toggled.connect(
            lambda checked: self.batch_spin.setEnabled(not checked)
        )
        self.batch_spin.setToolTip(
            "16GB 显存（如 A4000）上 2D 配置建议从 6 起步；显存更大可适当调大。"
        )
        form.addWidget(QtWidgets.QLabel("Batch size"), 5, 0)
        form.addWidget(self.batch_spin, 5, 1)
        self.workers_spin = QtWidgets.QSpinBox()
        self.workers_spin.setRange(1, 64)
        self.workers_spin.setValue(4)
        form.addWidget(QtWidgets.QLabel("预处理进程数"), 6, 0)
        form.addWidget(self.workers_spin, 6, 1)
        self.overlap_check = QtWidgets.QCheckBox(
            "允许排在后面的标签覆盖重叠体素"
        )
        self.overlap_check.setToolTip(
            "关闭更安全。多分类标签无法在同一体素上保留两个类别，"
            "默认遇到重叠直接停止训练。"
        )
        form.addWidget(self.overlap_check, 7, 0, 1, 2)
        self.continue_check = QtWidgets.QCheckBox("继续上次中断的训练")
        self.continue_check.setToolTip(
            "从同名任务中断处继续（沿用其 Dataset 和 checkpoint_latest）。"
            "须与上次使用相同的病例与标签，否则 checkpoint 无效。\n"
            "远程训练同样支持：从服务器上保留的同名任务 checkpoint 继续，"
            "不重新上传数据。"
        )
        form.addWidget(self.continue_check, 8, 0, 1, 2)
        layout.addWidget(training)

        recipe, recipe_form = self._surface("锁定配置（只读）")
        recipe_text = QtWidgets.QLabel(
            "FlexiCT ViT 主干（fp32，RoPE 安全）· AdamW 双参数组（backbone 3e-5 / decoder 3e-4，betas 0.9/0.98，wd 5e-2）· poly LR · gradclip 12 · 无 deep supervision · 前景过采样 0.33 · batch size 默认取自 nnU-Net plans（可手动调小）· 按验证 Dice 选最佳 checkpoint · 推理不用 TTA。\n\n以上为已验证的小样本配方，除 batch size 外在 Mimics 中不可修改。"
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
        self._refresh_models_async()
        # An empty case table on a pre-filled dataset path reads as "the path
        # is wrong" - it usually just means nobody pressed Rescan yet. Run
        # the existing background scan once when both path and label are
        # already known (C6-6).
        if (
            self.dataset_edit.text().strip()
            and self.label_edit.text().strip()
        ):
            self._rescan_cases_async()

    def _refresh_models_async(self):
        """load_models stats every model directory — run it off the GUI
        thread (铁律 1) and apply via _poll_submission (kind "models")."""
        if getattr(self, "_models_pending", False):
            return
        self._models_pending = True
        workspace = self.workspace_edit.text().strip()

        def scan():
            try:
                self._dataset_id_results.put(
                    ("models", load_models(workspace), ""))
            except Exception as exc:
                self._dataset_id_results.put(("models", None, str(exc)))

        worker = threading.Thread(target=scan, name="flexict-model-scan")
        worker.daemon = True
        worker.start()

    def _apply_models(self, models):
        QtWidgets = self.QtWidgets
        self.models_table.setRowCount(0)
        for model in models:
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
                "可用" if usable else "不可用",
            ]
            for column, value in enumerate(values):
                self.models_table.setItem(row, column, QtWidgets.QTableWidgetItem(str(value)))

    # Status strings come from scan_flexict_cases; translate the ones a
    # user can act on (the scan itself stays log-friendly English).
    _STATUS_LABELS = {
        "OK": "可用",
        "image missing": "缺少图像",
        "label missing": "缺少标签",
    }

    def _status_label(self, raw: str) -> str:
        # "label missing: item" -> "缺少标签：item"; unknown words pass
        # through so a new pipeline status is never silently blanked.
        parts = [part.strip() for part in str(raw or "").split(";")]
        rendered = []
        for part in parts:
            if part in self._STATUS_LABELS:
                rendered.append(self._STATUS_LABELS[part])
            elif part.startswith("label missing:"):
                rendered.append("缺少标签：" + part[len("label missing:"):].strip())
            else:
                rendered.append(part)
        return "；".join(rendered)

    def _merge_case_selection(self, fresh: list[dict]) -> list[dict]:
        """A rescan must not silently drop the user's train/val checkboxes.

        The scan pipeline returns use_train/use_val defaults; without this
        merge, rescanning after unchecking cases or marking validators reset
        every split to the default."""
        previous = {
            str(row.get("case_id")): row
            for row in getattr(self, "_case_rows", [])
            if row.get("checked")
        }
        if not previous:
            return fresh
        merged = []
        for row in fresh:
            old = previous.get(str(row.get("case_id")))
            if old is not None and row.get("checked"):
                row["use_train"] = bool(old.get("use_train", True))
                row["use_val"] = bool(old.get("use_val", False))
            merged.append(row)
        return merged

    def _refresh_cases(self):
        QtWidgets = self.QtWidgets
        QtCore = self.QtCore
        self.case_table.setRowCount(0)
        # Show every scanned case, not just the usable ones: hiding the
        # unusable rows makes "已找到 12 个病例" sit next to an empty table,
        # and the user has no way to see WHY a case is not usable.
        rows = list(self._case_rows)
        usable = 0
        for row in rows:
            index = self.case_table.rowCount()
            self.case_table.insertRow(index)
            case_item = QtWidgets.QTableWidgetItem(row["case_id"])
            if not row.get("checked"):
                # Unusable: grayed out, no train/val checkboxes (its split
                # would be ignored by the pipeline anyway).
                case_item.setFlags(QtCore.Qt.NoItemFlags)
                self.case_table.setItem(index, 0, case_item)
                status_item = QtWidgets.QTableWidgetItem(
                    self._status_label(row.get("status", ""))
                )
                status_item.setFlags(QtCore.Qt.NoItemFlags)
                self.case_table.setItem(index, 3, status_item)
                continue
            usable += 1
            self.case_table.setItem(index, 0, case_item)
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
            status_item = QtWidgets.QTableWidgetItem(
                self._status_label(row.get("status", ""))
            )
            self.case_table.setItem(index, 3, status_item)
        selected = [row for row in rows if row.get("checked")]
        train_count = sum(1 for row in selected if row.get("use_train", True))
        val_count = sum(1 for row in selected if row.get("use_val", False))
        summary = "共 {0} 例 · 可用 {1} · 已选 {2} · 训练 {3} · 验证 {4}".format(
            len(rows), usable, len(selected), train_count, val_count
        )
        if selected and train_count == 0:
            # Every checked case is validation-only: training would run on an
            # empty split. Surface it here instead of at submit time.
            summary += " —— 训练集为空，请至少保留一个训练病例"
        if rows and usable == 0:
            summary += " —— 没有可用病例：请检查标签名是否与 Mask 文件名一致"
        self.case_summary.setText(summary)

    def _label_names(self) -> list[str]:
        return [
            name.strip() for name in
            self.label_edit.text().replace("，", ",").split(",")
            if name.strip()
        ]

    def _scan_cases(self, dataset_root: str, label_names: list[str]) -> list[dict]:
        from flexict_pipeline import scan_flexict_cases

        return scan_flexict_cases(
            dataset_root,
            label_names[0] if label_names else "",
            label_names=label_names,
        )

    def _rescan_cases_async(self):
        if getattr(self, "_scan_pending", False):
            return
        self._scan_pending = True
        self.rescan_button.setEnabled(False)
        dataset_root = self.dataset_edit.text().strip()
        label_names = self._label_names()

        def scan():
            try:
                result = self._scan_cases(dataset_root, label_names)
                self._dataset_id_results.put(("scan", result, ""))
            except Exception as exc:
                self._dataset_id_results.put(("scan", None, str(exc)))

        worker = threading.Thread(target=scan, name="flexict-case-scan")
        worker.daemon = True
        worker.start()

    def _set_status(self, text, error=False):
        """One place for status coloring: an error set objectName("errorLabel")
        once and it was never reset, so every later neutral message ("正在提交
        ...") stayed red and bold forever."""
        self.status_label.setObjectName("errorLabel" if error else "hint")
        self.status_label.setStyleSheet(
            "color: #b42318; font-weight: 600;" if error else ""
        )
        self.status_label.setText(text)

    def _poll_submission(self):
        try:
            kind, payload, error = self._dataset_id_results.get_nowait()
        except Empty:
            pass
        else:
            self._scan_pending = False
            self.rescan_button.setEnabled(True)
            if kind == "models":
                self._models_pending = False
                self._apply_models(payload or [])
            elif kind == "scan":
                if error:
                    self._set_status(error, error=True)
                else:
                    self._case_rows = self._merge_case_selection(payload or [])
                    self._refresh_cases()
                    self._set_status(
                        "已找到 {0} 个病例。请确认训练/验证划分后开始训练。".format(
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
            self._set_status(
                "路径检查超时。所选磁盘可能离线或响应过慢；请粘贴更具体的路径或重试。",
                error=True,
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
            self._set_status(error, error=True)
            return
        try:
            self._set_status("正在提交 FlexiCT 训练...")
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
            self._set_status(str(exc), error=True)

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
        label_names = self._label_names()
        request = {
            "operation": "train",
            "task_name": self.label_edit.text().strip(),
            "task_id": self.label_edit.text().strip(),
            "label_name": label_names[0] if label_names else "",
            "labels": [
                {"name": name, "aliases": [name], "id": index + 1}
                for index, name in enumerate(label_names)
            ],
            "workspace": self.workspace_edit.text().strip(),
            "dataset_root": self.dataset_edit.text().strip(),
            "label_source": "dataset_masks",
            "modality": str(self.modality_combo.currentData()),
            "configuration": str(self.configuration_combo.currentData()),
            "epochs": int(self.epochs_spin.value()),
            "batch_size": None if self.auto_batch.isChecked()
            else int(self.batch_spin.value()),
            "mirror_disable_axes": str(self.mirror_combo.currentData()),
            "val_cases": val_count,
            "preprocess_workers": int(self.workers_spin.value()),
            "overlap_policy": (
                "replace" if self.overlap_check.isChecked() else "fail"
            ),
            "continue_training": bool(self.continue_check.isChecked()),
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

    def _duplicate_model_warning(self, request) -> str:
        """G7: retraining the same task_id piles up near-duplicate models;
        surface existing ones before the GPU time is spent (not a block)."""
        try:
            existing = load_models(request["workspace"])
        except Exception:
            return ""
        duplicates = [
            model for model in existing
            if str(model.get("task_id") or "") == request["task_id"]
        ]
        if not duplicates:
            return ""
        return (
            "该任务名下已有 {} 个模型（{}）。继续训练会新增一个模型而不是更新"
            "已有的；如需改进旧模型，请先在模型管理器中确认是否还需要它。".format(
                len(duplicates),
                ", ".join(
                    str(model.get("model_id") or "") for model in duplicates[:3]
                ),
            )
        )

    def _submit(self):
        if self._submission_pending:
            return
        try:
            request = self._request()
            if not request["label_name"]:
                raise ValueError("请填写目标 Mask 名称。")
            bad_labels = [
                name for name in self._label_names()
                if safe_identifier(name) == "item"
            ]
            if bad_labels:
                raise ValueError(
                    "标签名仅支持英文字母、数字、下划线、中划线和点"
                    "（要与数据集里 Mask 文件的文件名一致）：{0}".format(
                        "、".join(bad_labels)
                    )
                )
            if len(request.get("labels") or []) > 4:
                raise ValueError(
                    "FlexiCT 少样本训练每个模型最多支持 4 个标签；更多标签请使用 nnU-Net。"
                )
            # R72-C2: remote continue-training is allowed — the checkpoints
            # live on the remote server under the persistent data cache. A
            # caching-off server (checkpoints not retained) is rejected by
            # the remote controller before any data upload.
            warning = self._duplicate_model_warning(request)
            if warning:
                answer = self.QtWidgets.QMessageBox.question(
                    self.window, "重复任务", warning,
                    self.QtWidgets.QMessageBox.Yes | self.QtWidgets.QMessageBox.No,
                    self.QtWidgets.QMessageBox.No,
                )
                if answer != self.QtWidgets.QMessageBox.Yes:
                    return
            self._submission_generation += 1
            generation = self._submission_generation
            self._submission_pending = True
            self._submission_deadline = time.time() + 60.0
            self.start_button.setEnabled(False)
            self._set_status("正在后台检查所选数据位置...")

            def validate():
                try:
                    if not Path(request["dataset_root"]).is_dir():
                        raise ValueError("原始图像数据集不存在。")
                    if len(request["cases"]) < 2:
                        # The copy says "at least two"; the pipeline's train/
                        # val split needs both, or training runs empty.
                        raise ValueError(
                            "请至少选择两个病例（请先重新扫描数据集）。"
                        )
                    if request.get("train_case_ids") is not None and \
                            not request["train_case_ids"]:
                        # Every checked case is validation-only: the pipeline
                        # would fail on an empty train split only after
                        # spending a path-check round-trip (UI30).
                        raise ValueError(
                            "训练集为空：请至少保留一个训练病例。"
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
            self._set_status(str(exc), error=True)

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
