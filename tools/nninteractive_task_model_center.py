#!/usr/bin/env python3
"""Unified PySide6 setup, progress, and model center for nnInteractive."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for candidate in (str(ROOT), str(ROOT / "tools")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from nninteractive_task_common import (  # noqa: E402
    ACTIVE_STATUSES,
    TERMINAL_STATUSES,
    audit_model_dir,
    discover_mcs_cases,
    discover_prepared_cases,
    find_environment_python,
    find_prepared_label,
    find_task,
    jobs_dir,
    load_config,
    load_registry,
    model_is_usable,
    model_rows,
    official_model_dir,
    read_json,
    resolve_registered_model_dir,
    safe_slug,
    save_registry,
    selected_model,
    status_summary,
    strategy_display_name,
    task_dir,
    task_rows,
    workspace_root,
    write_json_atomic,
)
from training_data_ui import (  # noqa: E402
    LABEL_SOURCE_CHOICES,
    label_source_hint,
    normalized_source_mode,
)
from ui_theme import (  # noqa: E402
    choose_existing_directory_async,
    choose_open_file_async,
    choose_save_file_async,
    configure_application,
)
from ui_preferences import load_preferences, save_preferences  # noqa: E402
try:
    from remote_compute_ui import RemoteComputeSelector  # noqa: E402
except Exception:
    # Keep the existing local model center usable in incomplete/older bundles.
    RemoteComputeSelector = None


TITLE = "nnInteractive Custom Models"


def _launch_process(
    command: list[str], output_path: Path | None = None
) -> subprocess.Popen[Any]:
    launch = list(command)
    if os.name == "nt":
        python_path = Path(launch[0])
        pythonw = python_path.with_name("pythonw.exe")
        if pythonw.is_file():
            launch[0] = str(pythonw)
    env = dict(os.environ)
    existing = [value for value in env.get("PYTHONPATH", "").split(os.pathsep) if value]
    for value in (str(ROOT), str(ROOT / "tools"), str(ROOT / "integrations" / "nninteractive-finetune" / "src")):
        if value not in existing:
            existing.insert(0, value)
    env["PYTHONPATH"] = os.pathsep.join(existing)
    output_handle = None
    try:
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_handle = output_path.open(
                "a", encoding="utf-8", errors="replace"
            )
        process = subprocess.Popen(
            launch,
            cwd=str(ROOT),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=output_handle or subprocess.DEVNULL,
            stderr=subprocess.STDOUT if output_handle else subprocess.DEVNULL,
        )
    finally:
        if output_handle is not None:
            output_handle.close()
    # Register with the process registry so the health panel can see this
    # controller (best-effort; never blocks the launch).
    try:
        from resource_locks import register_process

        register_process(
            ROOT, "training_controller", process.pid,
            parent_pid=os.getpid(),
            state_path=str(output_path) if output_path is not None else "",
        )
    except Exception:
        pass
    return process


def _format_time(value: object) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(float(value)))
    except Exception:
        return "-"


def _format_delta(value: object) -> str:
    try:
        number = float(value)
    except Exception:
        return "-"
    return "{:+.1f}%".format(number * 100.0)


def _task_model_registry_signature(workspace: Path, task_id: str) -> tuple[Any, ...]:
    """Return a small signature that changes when one task's models change."""
    payload = load_registry(workspace)
    wanted = safe_slug(task_id)
    task = next(
        (
            row
            for row in payload.get("tasks") or []
            if isinstance(row, dict)
            and safe_slug(row.get("task_id")) == wanted
        ),
        {},
    )
    models = tuple(
        sorted(
            (
                str(row.get("model_id") or ""),
                str(row.get("state") or ""),
                bool(row.get("compatible", True)),
                float(row.get("created_at_epoch") or 0),
            )
            for row in task.get("models") or []
            if isinstance(row, dict)
        )
    )
    return (
        wanted,
        str(task.get("recommended_model_id") or ""),
        models,
    )


class TrainingCurve:
    def __init__(self, QtCore, QtGui, QtWidgets):
        class Widget(QtWidgets.QWidget):
            def __init__(inner_self):
                super().__init__()
                inner_self.points = []
                inner_self.baseline = None
                inner_self.setMinimumHeight(210)

            def set_points(inner_self, points):
                inner_self.points = list(points or [])
                inner_self.update()

            def set_baseline(inner_self, value):
                inner_self.baseline = (
                    float(value) if value is not None else None
                )
                inner_self.update()

            def paintEvent(inner_self, _event):
                painter = QtGui.QPainter(inner_self)
                painter.setRenderHint(QtGui.QPainter.Antialiasing)
                rect = inner_self.rect().adjusted(46, 18, -18, -34)
                painter.fillRect(inner_self.rect(), QtGui.QColor("#ffffff"))
                painter.setPen(QtGui.QPen(QtGui.QColor("#d9e0e8"), 1))
                painter.drawRect(rect)
                painter.setPen(QtGui.QColor("#667085"))
                painter.drawText(8, 18, "Metric")
                painter.drawText(rect.right() - 36, rect.bottom() + 25, "Epoch")
                points = inner_self.points
                if not points:
                    painter.drawText(rect, QtCore.Qt.AlignCenter, "Training metrics will appear here.")
                    return
                max_epoch = max(1, max(int(row.get("epoch") or 0) for row in points))

                def path_for(key, color):
                    values = [
                        (int(row.get("epoch") or 0), row.get(key))
                        for row in points
                        if row.get(key) is not None
                    ]
                    if not values:
                        return
                    path = QtGui.QPainterPath()
                    for index, (epoch, value) in enumerate(values):
                        x = rect.left() + (max(0, epoch - 1) / max(1, max_epoch - 1)) * rect.width()
                        y = rect.bottom() - max(0.0, min(1.0, float(value))) * rect.height()
                        if index == 0:
                            path.moveTo(x, y)
                        else:
                            path.lineTo(x, y)
                    painter.setPen(QtGui.QPen(QtGui.QColor(color), 2))
                    painter.drawPath(path)

                path_for("validation_auc", "#0f766e")
                if inner_self.baseline is not None:
                    # Dashed reference: the current model's AUC that the
                    # candidate must beat to be selected automatically.
                    value = max(0.0, min(1.0, float(inner_self.baseline)))
                    y = rect.bottom() - value * rect.height()
                    pen = QtGui.QPen(QtGui.QColor("#dc2626"), 1)
                    pen.setStyle(QtCore.Qt.DashLine)
                    painter.setPen(pen)
                    painter.drawLine(
                        QtCore.QLineF(rect.left(), y, rect.right(), y)
                    )
                    painter.setPen(QtGui.QColor("#dc2626"))
                    painter.drawText(
                        rect.left() + 4,
                        y - 4,
                        "current model {:.4f}".format(float(inner_self.baseline)),
                    )
                normalized = []
                losses = [float(row["train_loss"]) for row in points if row.get("train_loss") is not None]
                loss_max = max(losses) if losses else 1.0
                for row in points:
                    item = dict(row)
                    if item.get("train_loss") is not None:
                        item["normalized_loss"] = float(item["train_loss"]) / max(loss_max, 1e-8)
                    normalized.append(item)
                original = inner_self.points
                inner_self.points = normalized
                points = normalized
                path_for("normalized_loss", "#2563eb")
                inner_self.points = original
                painter.setPen(QtGui.QColor("#2563eb"))
                painter.drawText(rect.left(), rect.bottom() + 25, "Loss")
                painter.setPen(QtGui.QColor("#0f766e"))
                painter.drawText(rect.left() + 48, rect.bottom() + 25, "Validation AUC")

        self.Widget = Widget


class ModelCenter:
    def __init__(self, window, context, qt):
        self.window = window
        self.context = context
        self.QtCore, self.QtGui, self.QtWidgets = qt
        self.config = load_config()
        self.workspace = Path(
            context.get("workspace") or workspace_root(self.config)
        ).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.pipeline = ROOT / "tools" / "nninteractive_finetune_pipeline.py"
        self.python = find_environment_python()
        self.case_rows: list[dict[str, Any]] = []
        self.current_job_dir: Path | None = None
        self.current_status: dict[str, Any] = {}
        self.metrics: list[dict[str, Any]] = []
        self.last_metric_epoch = 0
        self.log_offset = 0
        self.scan_generation = 0
        self.scan_deadline = 0.0
        self.scan_progress = None
        self.pending_scan_result: tuple[
            int, list[dict[str, Any]], tuple[Any, ...], str
        ] | None = None
        self.last_scan_signature: tuple[Any, ...] | None = None
        self.mask_manually_edited = False
        self.last_terminal_signature = ""
        self.last_model_registry_signature = None
        self.case_filter_edit = None
        self.show_unavailable_cases = None
        self.choose_specific_cases = None
        self.case_selection_widget = None
        self.data_source_hint = None
        self.mcs_path_widget = None
        self.exported_path_widget = None
        self.initial_mask_name_widget = None
        self.initial_mask_path_widget = None
        self.initial_mask_hint = None
        self.model_io_process = None
        self.model_io_action = ""
        self.remote_selector = None
        self._case_population_generation = 0
        self._case_population_index = 0
        self._case_population_timer = self.QtCore.QTimer(self.window)
        self._case_population_timer.timeout.connect(self._populate_case_chunk)
        self._case_filter_timer = self.QtCore.QTimer(self.window)
        self._case_filter_timer.setSingleShot(True)
        self._case_filter_timer.timeout.connect(self._apply_case_filters)
        self._build()
        self._load_context_defaults()
        self.refresh_tasks()
        self.refresh_current_job()
        self.timer = self.QtCore.QTimer(self.window)
        self.timer.timeout.connect(self.refresh_status)
        self.timer.start(max(500, int(float(self.config.get("status_poll_seconds", 1.0)) * 1000)))

    def _build(self):
        QtCore, QtGui, QtWidgets = self.QtCore, self.QtGui, self.QtWidgets
        self.window.setWindowTitle(TITLE)
        self.window.resize(1040, 800)
        self.window.setMinimumSize(880, 620)
        root = QtWidgets.QVBoxLayout(self.window)
        root.setContentsMargins(24, 20, 24, 18)
        root.setSpacing(14)

        header = QtWidgets.QHBoxLayout()
        title_box = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel(TITLE)
        title.setObjectName("title")
        subtitle = QtWidgets.QLabel(
            "Train, monitor, and select reusable models for one annotation target."
        )
        subtitle.setObjectName("subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box, 1)
        header.addWidget(QtWidgets.QLabel("Annotation target"))
        self.task_combo = QtWidgets.QComboBox()
        self.task_combo.setEditable(True)
        self.task_combo.setMinimumWidth(240)
        self.task_combo.setToolTip(
            "Select an existing target or type a name for a new target."
        )
        self.task_combo.lineEdit().setPlaceholderText(
            "Select existing or type a new task"
        )
        self.task_combo.currentIndexChanged.connect(
            self._task_selection_changed
        )
        self.task_combo.lineEdit().editingFinished.connect(
            self._task_editing_finished
        )
        header.addWidget(self.task_combo)
        self.header_state = QtWidgets.QLabel("Ready")
        self.header_state.setObjectName("liveLabel")
        self.header_state.setMinimumWidth(110)
        header.addWidget(self.header_state)
        root.addLayout(header)

        self.tabs = QtWidgets.QTabWidget()
        self.setup_tab = self._build_setup_tab()
        self.progress_tab = self._build_progress_tab()
        self.models_tab = self._build_models_tab()
        self.tabs.addTab(self.setup_tab, "Training Setup")
        self.tabs.addTab(self.progress_tab, "Training Progress")
        self.tabs.addTab(self.models_tab, "Model Versions")
        self.tabs.currentChanged.connect(self.on_tab_changed)
        root.addWidget(self.tabs, 1)

    def _surface(self):
        frame = self.QtWidgets.QFrame()
        frame.setObjectName("surface")
        return frame

    def _path_row(self, label, initial=""):
        QtWidgets = self.QtWidgets
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(10)
        caption = QtWidgets.QLabel(label)
        caption.setMinimumWidth(172)
        edit = QtWidgets.QLineEdit(str(initial or ""))
        edit.setClearButtonEnabled(True)
        browse = QtWidgets.QPushButton("Browse...")
        browse.setIcon(self.window.style().standardIcon(QtWidgets.QStyle.SP_DirOpenIcon))
        browse.setToolTip("Choose folder")
        browse.setMinimumWidth(104)

        def choose():
            choose_existing_directory_async(
                self.QtCore,
                self.QtWidgets,
                self.window,
                "Choose {}".format(label),
                edit.text() or str(Path.home()),
                lambda selected: edit.setText(str(selected)) if selected else None,
                button=browse,
            )

        browse.clicked.connect(choose)
        row.addWidget(caption)
        row.addWidget(edit, 1)
        row.addWidget(browse)
        return row, edit

    def _build_setup_tab(self):
        QtCore, QtWidgets = self.QtCore, self.QtWidgets
        page = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(10)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        body = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(body)
        layout.setContentsMargins(4, 8, 4, 8)
        layout.setSpacing(12)

        if RemoteComputeSelector is not None:
            self.remote_selector = RemoteComputeSelector(
                self.window,
                (self.QtCore, self.QtGui, self.QtWidgets),
            )
            layout.addWidget(self.remote_selector.group)

        data = self._surface()
        data_layout = QtWidgets.QVBoxLayout(data)
        data_layout.setContentsMargins(16, 14, 16, 14)
        heading = QtWidgets.QLabel("Training data")
        heading.setObjectName("section")
        data_layout.addWidget(heading)
        source_row = QtWidgets.QHBoxLayout()
        source_label = QtWidgets.QLabel("Label source")
        source_label.setMinimumWidth(172)
        self.source_combo = QtWidgets.QComboBox()
        for label, value in LABEL_SOURCE_CHOICES:
            self.source_combo.addItem(label, value)
        self.source_combo.currentIndexChanged.connect(self._update_source_visibility)
        source_row.addWidget(source_label)
        source_row.addWidget(self.source_combo, 1)
        data_layout.addLayout(source_row)

        self.data_source_hint = QtWidgets.QLabel()
        self.data_source_hint.setObjectName("hint")
        self.data_source_hint.setWordWrap(True)
        data_layout.addWidget(self.data_source_hint)

        row, self.image_root_edit = self._path_row("Original image dataset *")
        self.image_root_edit.setPlaceholderText(
            "Folder containing one subfolder per case"
        )
        data_layout.addLayout(row)

        self.mcs_path_widget = QtWidgets.QWidget()
        mcs_layout = QtWidgets.QVBoxLayout(self.mcs_path_widget)
        mcs_layout.setContentsMargins(0, 0, 0, 0)
        row, self.mcs_edit = self._path_row("Saved .mcs folder *")
        mcs_layout.addLayout(row)
        data_layout.addWidget(self.mcs_path_widget)

        self.exported_path_widget = QtWidgets.QWidget()
        prepared_layout = QtWidgets.QVBoxLayout(self.exported_path_widget)
        prepared_layout.setContentsMargins(0, 0, 0, 0)
        row, self.prepared_label_edit = self._path_row(
            "Exported masks folder *"
        )
        prepared_layout.addLayout(row)
        data_layout.addWidget(self.exported_path_widget)
        self.prepared_edit = self.image_root_edit

        mask_row = QtWidgets.QHBoxLayout()
        mask_label = QtWidgets.QLabel("Target Mask name(s) *")
        mask_label.setMinimumWidth(172)
        self.mask_edit = QtWidgets.QLineEdit()
        self.mask_edit.setPlaceholderText("e.g. Liver, liver_seg")
        self.mask_edit.setToolTip(
            "Names used to find the same target Mask across saved Mimics projects."
        )
        self.mask_edit.textEdited.connect(self._mark_mask_manually_edited)
        mask_row.addWidget(mask_label)
        mask_row.addWidget(self.mask_edit, 1)
        self.scan_button = QtWidgets.QPushButton("Scan Data")
        self.scan_button.clicked.connect(self.scan_cases)
        mask_row.addWidget(self.scan_button)
        data_layout.addLayout(mask_row)
        self.mask_source_hint = QtWidgets.QLabel()
        self.mask_source_hint.setObjectName("hint")
        self.mask_source_hint.setWordWrap(True)
        self.mask_source_hint.setText(
            "Select a Mask in Mimics before opening this window, or enter a "
            "Task name to use it as the first Mask name."
        )
        data_layout.addWidget(self.mask_source_hint)

        initial_source_row = QtWidgets.QHBoxLayout()
        initial_source_label = QtWidgets.QLabel("Initial Mask source")
        initial_source_label.setMinimumWidth(172)
        self.initial_mask_source_combo = QtWidgets.QComboBox()
        self.initial_mask_source_combo.addItem(
            "No Initial Mask (start empty)", "none"
        )
        self.initial_mask_source_combo.addItem(
            "Another Mask in saved .mcs projects", "mcs"
        )
        self.initial_mask_source_combo.addItem(
            "Previously exported initial masks", "exported_masks"
        )
        self.initial_mask_source_combo.currentIndexChanged.connect(
            self._update_initial_mask_visibility
        )
        initial_source_row.addWidget(initial_source_label)
        initial_source_row.addWidget(self.initial_mask_source_combo, 1)
        data_layout.addLayout(initial_source_row)

        self.initial_mask_name_widget = QtWidgets.QWidget()
        initial_name_layout = QtWidgets.QHBoxLayout(
            self.initial_mask_name_widget
        )
        initial_name_layout.setContentsMargins(0, 0, 0, 0)
        initial_name_label = QtWidgets.QLabel("Initial Mask name(s) *")
        initial_name_label.setMinimumWidth(172)
        self.initial_mask_edit = QtWidgets.QLineEdit()
        self.initial_mask_edit.setPlaceholderText(
            "e.g. Liver_AI_Draft, liver_initial"
        )
        self.initial_mask_edit.setToolTip(
            "Names used to find the draft or partial Mask for the same case. "
            "Do not enter the final target Mask name."
        )
        initial_name_layout.addWidget(initial_name_label)
        initial_name_layout.addWidget(self.initial_mask_edit, 1)
        data_layout.addWidget(self.initial_mask_name_widget)

        self.initial_mask_path_widget = QtWidgets.QWidget()
        initial_path_layout = QtWidgets.QVBoxLayout(
            self.initial_mask_path_widget
        )
        initial_path_layout.setContentsMargins(0, 0, 0, 0)
        row, self.initial_mask_root_edit = self._path_row(
            "Initial masks folder *"
        )
        initial_path_layout.addLayout(row)
        data_layout.addWidget(self.initial_mask_path_widget)

        self.initial_mask_hint = QtWidgets.QLabel()
        self.initial_mask_hint.setObjectName("hint")
        self.initial_mask_hint.setWordWrap(True)
        data_layout.addWidget(self.initial_mask_hint)

        self.scan_hint = QtWidgets.QLabel(
            "Required paths are marked with *. Scan to find usable cases."
        )
        self.scan_hint.setObjectName("hint")
        data_layout.addWidget(self.scan_hint)
        self.scan_progress = QtWidgets.QProgressBar()
        self.scan_progress.setRange(0, 100)
        self.scan_progress.setValue(0)
        self.scan_progress.setFormat("Dataset has not been scanned")
        data_layout.addWidget(self.scan_progress)

        self.choose_specific_cases = QtWidgets.QCheckBox(
            "Choose specific cases and validation assignments"
        )
        self.choose_specific_cases.setChecked(False)
        self.choose_specific_cases.toggled.connect(
            self._update_case_selection_visibility
        )
        data_layout.addWidget(self.choose_specific_cases)

        self.case_selection_widget = QtWidgets.QWidget()
        case_selection_layout = QtWidgets.QVBoxLayout(self.case_selection_widget)
        case_selection_layout.setContentsMargins(0, 0, 0, 0)
        filter_row = QtWidgets.QHBoxLayout()
        self.case_filter_edit = QtWidgets.QLineEdit()
        self.case_filter_edit.setPlaceholderText("Filter cases")
        self.case_filter_edit.textChanged.connect(self._schedule_case_filters)
        filter_row.addWidget(self.case_filter_edit, 1)
        self.show_unavailable_cases = QtWidgets.QCheckBox("Show unavailable")
        self.show_unavailable_cases.toggled.connect(self._apply_case_filters)
        filter_row.addWidget(self.show_unavailable_cases)
        case_selection_layout.addLayout(filter_row)

        self.case_table = QtWidgets.QTableWidget(0, 4)
        self.case_table.setHorizontalHeaderLabels(["Use", "Case", "Split", "Status"])
        self.case_table.verticalHeader().setVisible(False)
        self.case_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.case_table.setAlternatingRowColors(True)
        self.case_table.setMinimumHeight(190)
        self.case_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.case_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.case_table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.case_table.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
        case_selection_layout.addWidget(self.case_table)
        data_layout.addWidget(self.case_selection_widget)
        layout.addWidget(data)

        adaptation = self._surface()
        adaptation_layout = QtWidgets.QVBoxLayout(adaptation)
        adaptation_layout.setContentsMargins(16, 14, 16, 14)
        heading = QtWidgets.QLabel("Training plan")
        heading.setObjectName("section")
        adaptation_layout.addWidget(heading)
        self.light_radio = QtWidgets.QRadioButton(
            "CLoPA-IN  ·  Lightweight intensity-domain adaptation for CT and MR"
        )
        self.strong_radio = QtWidgets.QRadioButton(
            "CLoPA-CN  ·  Instance normalization plus selected convolution adaptation"
        )
        self.light_radio.setChecked(True)
        adaptation_layout.addWidget(self.light_radio)
        adaptation_layout.addWidget(self.strong_radio)

        training_layout = QtWidgets.QGridLayout()
        training_layout.setContentsMargins(22, 4, 0, 0)
        training_layout.setHorizontalSpacing(14)
        training_layout.setVerticalSpacing(6)
        training_layout.addWidget(QtWidgets.QLabel("Maximum epochs"), 0, 0)
        self.epochs = QtWidgets.QSpinBox()
        self.epochs.setRange(
            int(self.config.get("minimum_epochs", 4)),
            int(self.config.get("maximum_epochs", 20)),
        )
        self.epochs.setValue(int(self.config.get("default_epochs", 10)))
        training_layout.addWidget(self.epochs, 1, 0)
        training_layout.addWidget(
            QtWidgets.QLabel("Training goal"), 0, 1
        )
        self.training_goal = QtWidgets.QComboBox()
        self.training_goal.addItem(
            "General adaptation (recommended)", "general"
        )
        self.training_goal.addItem(
            "Start from an empty Mask", "start_empty"
        )
        self.training_goal.addItem(
            "Refine an existing Mask", "refine_existing"
        )
        self.training_goal.currentIndexChanged.connect(
            self._update_initial_mask_visibility
        )
        self.training_goal.setToolTip(
            "General adaptation trains both new annotations and corrections. "
            "Each correction step can add one foreground and one background "
            "point before prediction; trajectories use one to eight steps and "
            "stop early when no error remains."
        )
        training_layout.addWidget(self.training_goal, 1, 1)
        training_layout.addWidget(QtWidgets.QLabel("Starting model"), 0, 2)
        self.start_model = QtWidgets.QComboBox()
        self.start_model.addItem("Official nnInteractive", "official")
        self.start_model_user_selected = False
        self.start_model.activated.connect(self._mark_start_model_selected)
        training_layout.addWidget(self.start_model, 1, 2)
        training_layout.setColumnStretch(1, 1)
        training_layout.setColumnStretch(2, 1)
        adaptation_layout.addLayout(training_layout)
        mirror_row = QtWidgets.QHBoxLayout()
        mirror_row.setContentsMargins(22, 4, 0, 0)
        mirror_label = QtWidgets.QLabel("Anatomy mirroring")
        mirror_label.setMinimumWidth(172)
        self.mirror_policy = QtWidgets.QComboBox()
        self.mirror_policy.addItem(
            "Automatic (protect left/right targets)", "auto"
        )
        self.mirror_policy.addItem(
            "Preserve left/right orientation", "preserve_lr"
        )
        self.mirror_policy.addItem("Allow all spatial axes", "all_axes")
        self.mirror_policy.setToolTip(
            "Automatic disables left-right mirroring when the task or Target "
            "Mask name contains left/right, L/R, or 左/右. Other spatial "
            "mirroring remains enabled."
        )
        mirror_row.addWidget(mirror_label)
        mirror_row.addWidget(self.mirror_policy, 1)
        adaptation_layout.addLayout(mirror_row)
        layout.addWidget(adaptation)
        layout.addStretch(1)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        footer = QtWidgets.QHBoxLayout()
        self.setup_summary = QtWidgets.QLabel("Scan cases to prepare training.")
        self.setup_summary.setObjectName("hint")
        self.setup_summary.setWordWrap(True)
        footer.addWidget(self.setup_summary, 1)
        self.start_button = QtWidgets.QPushButton("Start Training")
        self.start_button.setObjectName("primary")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_training)
        footer.addWidget(self.start_button)
        outer.addLayout(footer)
        self._update_source_visibility()
        self._update_initial_mask_visibility()
        self._update_case_selection_visibility()
        return page

    def _build_progress_tab(self):
        QtWidgets = self.QtWidgets
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(4, 10, 4, 4)
        layout.setSpacing(12)

        status_surface = self._surface()
        status_layout = QtWidgets.QVBoxLayout(status_surface)
        status_layout.setContentsMargins(16, 14, 16, 14)
        status_head = QtWidgets.QHBoxLayout()
        self.progress_title = QtWidgets.QLabel("No training is running")
        self.progress_title.setObjectName("section")
        self.elapsed_label = QtWidgets.QLabel("")
        self.elapsed_label.setObjectName("hint")
        status_head.addWidget(self.progress_title, 1)
        status_head.addWidget(self.elapsed_label)
        status_layout.addLayout(status_head)
        self.progress_detail = QtWidgets.QLabel("Start training from Training Setup.")
        self.progress_detail.setObjectName("hint")
        status_layout.addWidget(self.progress_detail)
        self.diagnosis_title = QtWidgets.QLabel("Failure diagnosis")
        self.diagnosis_title.setObjectName("section")
        self.diagnosis_title.setVisible(False)
        status_layout.addWidget(self.diagnosis_title)
        self.diagnosis_detail = QtWidgets.QLabel("")
        self.diagnosis_detail.setObjectName("hint")
        self.diagnosis_detail.setWordWrap(True)
        self.diagnosis_detail.setVisible(False)
        status_layout.addWidget(self.diagnosis_detail)
        self.diagnosis_log = QtWidgets.QPlainTextEdit()
        self.diagnosis_log.setReadOnly(True)
        self.diagnosis_log.setVisible(False)
        self.diagnosis_log.setStyleSheet(
            "QPlainTextEdit { font-family: Consolas, monospace; font-size: 12px; }"
        )
        status_layout.addWidget(self.diagnosis_log)
        diagnosis_actions = QtWidgets.QHBoxLayout()
        self.open_job_folder_button = QtWidgets.QPushButton("Open Job Folder")
        self.open_job_folder_button.clicked.connect(self.open_job_folder)
        self.open_job_folder_button.setVisible(False)
        diagnosis_actions.addStretch(1)
        diagnosis_actions.addWidget(self.open_job_folder_button)
        self.diagnosis_actions_layout = diagnosis_actions
        status_layout.addLayout(diagnosis_actions)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        status_layout.addWidget(self.progress_bar)
        metrics = QtWidgets.QHBoxLayout()
        self.loss_label = QtWidgets.QLabel("Loss  -")
        self.auc_label = QtWidgets.QLabel("Validation AUC  -")
        self.best_label = QtWidgets.QLabel("Best AUC  -")
        metrics.addWidget(self.loss_label)
        metrics.addWidget(self.auc_label)
        metrics.addWidget(self.best_label)
        metrics.addStretch(1)
        status_layout.addLayout(metrics)
        self.auc_detail_label = QtWidgets.QLabel("")
        self.auc_detail_label.setObjectName("hint")
        self.auc_detail_label.setWordWrap(True)
        status_layout.addWidget(self.auc_detail_label)
        self.comparison_label = QtWidgets.QLabel("")
        self.comparison_label.setWordWrap(True)
        status_layout.addWidget(self.comparison_label)
        layout.addWidget(status_surface)

        lower = QtWidgets.QSplitter(self.QtCore.Qt.Horizontal)
        curve_surface = self._surface()
        curve_layout = QtWidgets.QVBoxLayout(curve_surface)
        curve_layout.setContentsMargins(12, 12, 12, 12)
        curve_layout.addWidget(QtWidgets.QLabel("Training curve"))
        curve_class = TrainingCurve(self.QtCore, self.QtGui, self.QtWidgets).Widget
        self.curve = curve_class()
        curve_layout.addWidget(self.curve, 1)
        lower.addWidget(curve_surface)
        log_surface = self._surface()
        log_layout = QtWidgets.QVBoxLayout(log_surface)
        log_layout.setContentsMargins(12, 12, 12, 12)
        log_head = QtWidgets.QHBoxLayout()
        log_head.addWidget(QtWidgets.QLabel("Recent activity"))
        self.new_log_button = QtWidgets.QPushButton("New log entries")
        self.new_log_button.setVisible(False)
        self.new_log_button.clicked.connect(self.follow_latest_log)
        log_head.addStretch(1)
        log_head.addWidget(self.new_log_button)
        log_layout.addLayout(log_head)
        self.log_text = QtWidgets.QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setLineWrapMode(QtWidgets.QPlainTextEdit.WidgetWidth)
        log_layout.addWidget(self.log_text, 1)
        lower.addWidget(log_surface)
        lower.setSizes([560, 390])
        layout.addWidget(lower, 1)

        actions = QtWidgets.QHBoxLayout()
        self.pause_button = QtWidgets.QPushButton("Pause and Release GPU")
        self.pause_button.clicked.connect(lambda: self.request_action("pause"))
        self.resume_button = QtWidgets.QPushButton("Resume Training")
        self.resume_button.clicked.connect(self.resume_training)
        self.stop_button = QtWidgets.QPushButton("Stop Training")
        self.stop_button.setObjectName("dangerButton")
        self.stop_button.clicked.connect(self.request_primary_stop)
        self.open_log_button = QtWidgets.QPushButton("Open Log")
        self.open_log_button.clicked.connect(self.open_log)
        hide = QtWidgets.QPushButton("Hide Window")
        hide.clicked.connect(self.window.hide)
        actions.addWidget(self.pause_button)
        actions.addWidget(self.resume_button)
        actions.addWidget(self.stop_button)
        actions.addStretch(1)
        actions.addWidget(self.open_log_button)
        actions.addWidget(hide)
        layout.addLayout(actions)
        self._update_progress_actions({})
        return page

    def _build_models_tab(self):
        QtWidgets = self.QtWidgets
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(4, 10, 4, 4)
        current = self._surface()
        current_layout = QtWidgets.QVBoxLayout(current)
        current_layout.setContentsMargins(16, 14, 16, 14)
        self.current_model_title = QtWidgets.QLabel("No active annotation model")
        self.current_model_title.setObjectName("section")
        self.current_model_detail = QtWidgets.QLabel(
            "Train and validate a model before target-specific annotation."
        )
        self.current_model_detail.setObjectName("hint")
        self.current_model_detail.setWordWrap(True)
        current_layout.addWidget(self.current_model_title)
        current_layout.addWidget(self.current_model_detail)
        layout.addWidget(current)

        table_head = QtWidgets.QHBoxLayout()
        table_head.addWidget(QtWidgets.QLabel("Model history"))
        table_head.addStretch(1)
        self.import_model_button = QtWidgets.QPushButton("Import Model Package")
        self.import_model_button.setToolTip(
            "Install a portable custom-model .zip created on another machine."
        )
        self.import_model_button.clicked.connect(self.import_model_package)
        table_head.addWidget(self.import_model_button)
        self.export_model_button = QtWidgets.QPushButton("Export Active Model")
        self.export_model_button.setToolTip(
            "Create a self-contained model package for another workstation."
        )
        self.export_model_button.clicked.connect(self.export_current_model)
        table_head.addWidget(self.export_model_button)
        self.show_failed = QtWidgets.QCheckBox("Show failed and not-improved versions")
        self.show_failed.toggled.connect(self.refresh_models)
        table_head.addWidget(self.show_failed)
        layout.addLayout(table_head)
        self.model_table = QtWidgets.QTableWidget(0, 6)
        self.model_table.setHorizontalHeaderLabels(
            [
                "Date",
                "Model version",
                "Adaptation",
                "Data",
                "Validation",
                "Annotation model",
            ]
        )
        self.model_table.verticalHeader().setVisible(False)
        self.model_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.model_table.setAlternatingRowColors(True)
        self.model_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.model_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.model_table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.model_table.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
        self.model_table.horizontalHeader().setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeToContents)
        self.model_table.horizontalHeader().setSectionResizeMode(
            5, QtWidgets.QHeaderView.Fixed
        )
        self.model_table.setColumnWidth(5, 136)
        layout.addWidget(self.model_table, 1)
        self.model_io_status = QtWidgets.QLabel("")
        self.model_io_status.setObjectName("hint")
        self.model_io_status.setWordWrap(True)
        layout.addWidget(self.model_io_status)
        return page

    def _load_context_defaults(self):
        remembered = load_preferences("nninteractive_training")
        initial_name = str(self.context.get("mask_name") or "")
        self.task_combo.setEditText(initial_name)
        self.mask_edit.setText(initial_name)
        self.mcs_edit.setText(str(self.context.get("mcs_dir") or remembered.get("mcs_dir") or ""))
        self.image_root_edit.setText(str(self.context.get("image_root") or remembered.get("image_root") or ""))
        self.prepared_edit.setText(str(self.context.get("image_root") or ""))
        self.prepared_label_edit.setText(
            str(self.context.get("label_root") or remembered.get("label_root") or "")
        )

    def refresh_tasks(self):
        current_text = self.task_combo.currentText().strip()
        current_index = self.task_combo.currentIndex()
        current_id = ""
        if (
            current_index >= 0
            and current_text == self.task_combo.itemText(current_index)
        ):
            current_id = str(self.task_combo.currentData() or "")
        self.task_combo.blockSignals(True)
        self.task_combo.clear()
        for task in task_rows(self.workspace):
            self.task_combo.addItem(
                str(task.get("task_name") or task.get("task_id")),
                str(task.get("task_id") or ""),
            )
        if current_id or current_text:
            index = self.task_combo.findData(current_id)
            if index < 0 and current_text:
                wanted = safe_slug(current_text)
                for candidate in range(self.task_combo.count()):
                    if safe_slug(self.task_combo.itemData(candidate)) == wanted:
                        index = candidate
                        break
            if index >= 0:
                self.task_combo.setCurrentIndex(index)
            elif current_text:
                self.task_combo.setEditText(current_text)
        self.task_combo.blockSignals(False)
        self.on_task_changed(self.task_combo.currentText())

    def _task_selection_changed(self, _index):
        if self.task_combo.currentIndex() < 0:
            return
        self.mask_manually_edited = False
        self.on_task_changed(self.task_combo.currentText())

    def _task_editing_finished(self):
        self.on_task_changed(self.task_combo.currentText())

    def _mark_mask_manually_edited(self, _value):
        self.mask_manually_edited = True

    def current_task_id(self):
        text = self.task_combo.currentText().strip()
        index = self.task_combo.currentIndex()
        if index >= 0 and text == self.task_combo.itemText(index):
            data = self.task_combo.itemData(index)
        else:
            data = ""
        return safe_slug(data or text)

    def on_task_changed(self, value):
        task = find_task(self.workspace, self.current_task_id())
        if task:
            names = task.get("mask_names") or []
            if names:
                self.mask_edit.setText(", ".join(str(name) for name in names))
                self.mask_manually_edited = False
            self.mask_source_hint.setText(
                "Loaded the saved Mask name aliases for this annotation task. "
                "Edit only when the same target uses another name in some cases."
            )
        elif str(value or "").strip() and not self.mask_manually_edited:
            self.mask_edit.setText(str(value).strip())
            selected_mask = str(self.context.get("mask_name") or "").strip()
            if selected_mask and safe_slug(selected_mask) == safe_slug(value):
                self.mask_source_hint.setText(
                    "Filled from the Mask selected in Mimics: {}. Edit only to "
                    "add alternative names used by other cases.".format(
                        selected_mask
                    )
                )
            else:
                self.mask_source_hint.setText(
                    "For a new task, the Task name is used as the first Mask "
                    "name. Add comma-separated aliases only when needed."
                )
        self.start_model_user_selected = False
        self._refresh_start_models()
        self.refresh_current_job()
        self.refresh_models()
        self._update_setup_summary()

    def _mark_start_model_selected(self, _index):
        self.start_model_user_selected = True

    def _refresh_start_models(self):
        current_value = self.start_model.currentData() if hasattr(self, "start_model") else "official"
        if not hasattr(self, "start_model"):
            return
        self.start_model.clear()
        self.start_model.addItem("Official nnInteractive", "official")
        model = selected_model(self.workspace, self.current_task_id())
        if model and audit_model_dir(
            resolve_registered_model_dir(
                self.workspace,
                model,
                find_task(self.workspace, self.current_task_id()),
            ),
            include_checksum=False,
        ).get("compatible"):
            self.start_model.addItem("Current custom model", "current")
            use_current = (
                current_value == "current"
                or not self.start_model_user_selected
            )
            self.start_model.setCurrentIndex(1 if use_current else 0)

    def _update_source_visibility(self):
        source = normalized_source_mode(self.source_combo.currentData())
        initial_source = (
            str(self.initial_mask_source_combo.currentData() or "none")
            if hasattr(self, "initial_mask_source_combo")
            else "none"
        )
        self.mcs_path_widget.setVisible(
            source == "mcs_refresh" or initial_source == "mcs"
        )
        self.exported_path_widget.setVisible(source == "exported_masks")
        if self.data_source_hint is not None:
            self.data_source_hint.setText(label_source_hint(source))
        self._update_initial_mask_visibility()
        if self.case_rows:
            self.scan_hint.setText(
                "Data source changed. Scan Data again before starting training."
            )

    def _update_initial_mask_visibility(self):
        if not hasattr(self, "initial_mask_source_combo"):
            return
        goal = (
            str(self.training_goal.currentData() or "general")
            if hasattr(self, "training_goal")
            else "general"
        )
        enabled = goal != "start_empty"
        if not enabled:
            empty_index = self.initial_mask_source_combo.findData("none")
            if empty_index >= 0 and self.initial_mask_source_combo.currentIndex() != empty_index:
                self.initial_mask_source_combo.blockSignals(True)
                self.initial_mask_source_combo.setCurrentIndex(empty_index)
                self.initial_mask_source_combo.blockSignals(False)
        source = str(
            self.initial_mask_source_combo.currentData() or "none"
        )
        self.initial_mask_source_combo.setEnabled(enabled)
        show_names = enabled and source in ("mcs", "exported_masks")
        if self.initial_mask_name_widget is not None:
            self.initial_mask_name_widget.setVisible(show_names)
        if self.initial_mask_path_widget is not None:
            self.initial_mask_path_widget.setVisible(
                enabled and source == "exported_masks"
            )
        if self.mcs_path_widget is not None:
            label_source = normalized_source_mode(
                self.source_combo.currentData()
            )
            self.mcs_path_widget.setVisible(
                label_source == "mcs_refresh"
                or (enabled and source == "mcs")
            )
        if self.initial_mask_hint is not None:
            if not enabled:
                text = (
                    "Start from an empty Mask does not use initial Masks. "
                    "No initial Mask will be exported."
                )
            elif source == "mcs":
                text = (
                    "A matching draft Mask is read from each saved project. "
                    "Only real drafts are used. General training starts empty "
                    "for cases without one; refine-existing training skips them."
                )
            elif source == "exported_masks":
                text = (
                    "A matching NIfTI draft is read for each case. Only real "
                    "drafts are used; no synthetic Initial Mask is generated."
                )
            else:
                text = (
                    "Training starts from an empty Mask. Choose another source "
                    "only when real AI drafts or partial annotations are available."
                )
            self.initial_mask_hint.setText(text)
        if self.case_rows:
            self.scan_hint.setText(
                "Initial Mask settings changed. Scan Data again before training."
            )

    def _update_case_selection_visibility(self):
        visible = bool(
            self.choose_specific_cases and self.choose_specific_cases.isChecked()
        )
        if self.case_selection_widget is not None:
            self.case_selection_widget.setVisible(visible)
        if visible:
            self._populate_cases(preserve_assignments=True)
        elif hasattr(self, "case_table"):
            self._case_population_generation += 1
            self._case_population_timer.stop()
            self.case_table.setRowCount(0)
        self._update_setup_summary()

    def _schedule_case_filters(self):
        if hasattr(self, "_case_filter_timer"):
            self._case_filter_timer.start(180)

    def _apply_case_filters(self):
        if not hasattr(self, "case_table"):
            return
        if self.choose_specific_cases is not None and not self.choose_specific_cases.isChecked():
            return
        query = (
            str(self.case_filter_edit.text()).strip().lower()
            if self.case_filter_edit is not None
            else ""
        )
        show_unavailable = bool(
            self.show_unavailable_cases
            and self.show_unavailable_cases.isChecked()
        )
        for index, row in enumerate(self.case_rows):
            ready = row.get("state") == "ready"
            case_id = str(row.get("case_id") or "")
            hidden = (not ready and not show_unavailable) or (
                bool(query) and query not in case_id.lower()
            )
            self.case_table.setRowHidden(index, hidden)

    def _current_data_signature(self):
        def normalized_path(widget):
            value = str(widget.text()).strip() if widget is not None else ""
            return os.path.normcase(os.path.abspath(value)) if value else ""

        return (
            normalized_source_mode(self.source_combo.currentData()),
            normalized_path(self.image_root_edit),
            normalized_path(self.mcs_edit),
            normalized_path(self.prepared_label_edit),
            tuple(
                sorted(
                    safe_slug(value)
                    for value in self.mask_edit.text().split(",")
                    if value.strip()
                )
            ),
            (
                "none"
                if str(self.training_goal.currentData() or "general")
                == "start_empty"
                else str(
                    self.initial_mask_source_combo.currentData()
                    or "none"
                )
            ),
            normalized_path(self.initial_mask_root_edit),
            tuple(
                sorted(
                    safe_slug(value)
                    for value in self.initial_mask_edit.text().split(",")
                    if value.strip()
                )
            ),
        )

    def _set_scan_busy(self):
        if self.scan_progress is not None:
            self.scan_progress.setRange(0, 0)
            self.scan_progress.setFormat("Scanning dataset...")
        self.scan_button.setText("Scanning...")
        self.scan_button.setEnabled(False)

    def _finish_scan_progress(self, text, success):
        self.scan_deadline = 0.0
        if self.scan_progress is not None:
            self.scan_progress.setRange(0, 100)
            self.scan_progress.setValue(100 if success else 0)
            self.scan_progress.setFormat(str(text))
        self.scan_button.setText("Scan Data")
        self.scan_button.setEnabled(True)

    def scan_cases(self):
        mask_names = [
            value.strip() for value in self.mask_edit.text().split(",") if value.strip()
        ]
        if not mask_names:
            self.scan_hint.setText("Enter the target Mask name before scanning.")
            self._finish_scan_progress("Target Mask name is required", False)
            return
        self.scan_generation += 1
        generation = self.scan_generation
        self.scan_deadline = time.time() + float(
            self.config.get("case_scan_timeout_seconds", 120)
        )
        source_mode = self.source_combo.currentData()
        mcs_dir = self.mcs_edit.text().strip()
        image_root = self.image_root_edit.text().strip()
        prepared_labels = self.prepared_label_edit.text().strip()
        goal = str(self.training_goal.currentData() or "general")
        initial_source = (
            "none"
            if goal == "start_empty"
            else str(
                self.initial_mask_source_combo.currentData() or "none"
            )
        )
        initial_mask_names = [
            value.strip()
            for value in self.initial_mask_edit.text().split(",")
            if value.strip()
        ]
        initial_mask_root = self.initial_mask_root_edit.text().strip()
        if (
            initial_source in ("mcs", "exported_masks")
            and {
                safe_slug(value) for value in mask_names
            }
            & {safe_slug(value) for value in initial_mask_names}
        ):
            self.scan_hint.setText(
                "The Initial Mask must be different from the final Target Mask."
            )
            self._finish_scan_progress(
                "Initial and final Mask names overlap", False
            )
            return
        if not image_root:
            self.scan_hint.setText("Choose the original image dataset.")
            self._finish_scan_progress("Dataset path is required", False)
            return
        if source_mode == "mcs_refresh" and not mcs_dir:
            self.scan_hint.setText("Choose the folder containing saved .mcs projects.")
            self._finish_scan_progress("Saved .mcs folder is required", False)
            return
        if source_mode == "exported_masks" and not prepared_labels:
            self.scan_hint.setText("Choose the previously exported masks folder.")
            self._finish_scan_progress("Exported masks folder is required", False)
            return
        if initial_source in ("mcs", "exported_masks") and not initial_mask_names:
            self.scan_hint.setText(
                "Enter the Initial Mask name before scanning."
            )
            self._finish_scan_progress("Initial Mask name is required", False)
            return
        if goal == "refine_existing" and initial_source == "none":
            self.scan_hint.setText(
                "Refine an existing Mask requires saved .mcs drafts or "
                "previously exported Initial Masks."
            )
            self._finish_scan_progress("Initial Mask source is required", False)
            return
        if initial_source == "mcs" and not mcs_dir:
            self.scan_hint.setText(
                "Choose the folder containing saved .mcs projects."
            )
            self._finish_scan_progress("Saved .mcs folder is required", False)
            return
        if initial_source == "exported_masks" and not initial_mask_root:
            self.scan_hint.setText(
                "Choose the previously exported initial masks folder."
            )
            self._finish_scan_progress(
                "Initial masks folder is required", False
            )
            return
        self.scan_hint.setText("Scanning cases in the background...")
        self._set_scan_busy()
        scan_signature = self._current_data_signature()

        def worker():
            try:
                if not os.path.isdir(image_root):
                    raise RuntimeError(
                        "Original image dataset does not exist or is unavailable: "
                        "{}".format(image_root)
                    )
                if source_mode == "mcs_refresh":
                    if not os.path.isdir(mcs_dir):
                        raise RuntimeError(
                            "Saved .mcs folder does not exist or is unavailable: "
                            "{}".format(mcs_dir)
                        )
                    rows = discover_mcs_cases(mcs_dir, image_root)
                else:
                    if source_mode == "exported_masks" and not os.path.isdir(
                        prepared_labels
                    ):
                        raise RuntimeError(
                            "Exported masks folder does not exist or is "
                            "unavailable: {}".format(prepared_labels)
                        )
                    rows = discover_prepared_cases(
                        image_root,
                        mask_names,
                        label_root=(
                            prepared_labels
                            if source_mode == "exported_masks"
                            else None
                        ),
                    )
                if initial_source == "mcs":
                    if not os.path.isdir(mcs_dir):
                        raise RuntimeError(
                            "Saved .mcs folder does not exist or is unavailable: "
                            "{}".format(mcs_dir)
                        )
                    mcs_rows = {
                        str(row.get("case_id")): row
                        for row in discover_mcs_cases(mcs_dir, image_root)
                    }
                    for row in rows:
                        match = mcs_rows.get(str(row.get("case_id"))) or {}
                        if match.get("mcs_path"):
                            row["initial_mcs_path"] = str(
                                match["mcs_path"]
                            )
                            if not row.get("mcs_path"):
                                row["mcs_path"] = str(match["mcs_path"])
                elif initial_source == "exported_masks":
                    if not os.path.isdir(initial_mask_root):
                        raise RuntimeError(
                            "Initial masks folder does not exist or is "
                            "unavailable: {}".format(initial_mask_root)
                        )
                    for row in rows:
                        initial_mask = find_prepared_label(
                            Path(initial_mask_root)
                            / str(row.get("case_id")),
                            initial_mask_names,
                        )
                        if initial_mask is not None:
                            row["initial_mask"] = str(initial_mask)
                error = ""
            except Exception as exc:
                rows = []
                error = str(exc)

            self.pending_scan_result = (generation, rows, scan_signature, error)

        thread = threading.Thread(target=worker, name="nnInteractiveCaseScan")
        thread.daemon = True
        thread.start()

    def _apply_pending_scan_result(self):
        result = self.pending_scan_result
        if result is None:
            if (
                self.scan_deadline
                and time.time() >= self.scan_deadline
                and not self.scan_button.isEnabled()
            ):
                self.scan_generation += 1
                self.scan_deadline = 0.0
                self.last_scan_signature = None
                self.scan_hint.setText(
                    "Case scan timed out. The selected location may be offline "
                    "or responding too slowly; choose another path or retry."
                )
                self._finish_scan_progress("Scan timed out", False)
            return
        self.pending_scan_result = None
        generation, rows, scan_signature, error = result
        if generation != self.scan_generation:
            return
        self.scan_deadline = 0.0
        self.case_rows = rows
        self.last_scan_signature = scan_signature
        self._populate_cases()
        if error:
            self.scan_hint.setText("Case scan failed: {}".format(error))
            self._finish_scan_progress("Scan failed", False)
        elif rows:
            ready = sum(row.get("state") == "ready" for row in rows)
            unavailable = len(rows) - ready
            initial_source = str(scan_signature[5] or "none")
            if initial_source == "exported_masks":
                initial_count = sum(
                    bool(row.get("initial_mask"))
                    for row in rows
                    if row.get("state") == "ready"
                )
                initial_note = (
                    " {} of {} usable case(s) have a matching real Initial "
                    "Mask; other cases start empty in General mode."
                ).format(initial_count, ready)
            elif initial_source == "mcs":
                initial_note = (
                    " Initial Masks in saved projects are verified when "
                    "training starts. Missing drafts start empty in General "
                    "mode and are skipped in Refine Existing mode."
                )
            else:
                initial_note = " Training starts from an empty Mask."
            if scan_signature[0] == "mcs_refresh":
                self.scan_hint.setText(
                    "{} saved project candidate(s) found; {} case(s) are missing "
                    "a source image. Target Masks are verified in the background "
                    "when training starts, and projects without a match are "
                    "skipped automatically.{}".format(
                        ready, unavailable, initial_note
                    )
                )
            else:
                self.scan_hint.setText(
                    "{} usable case(s) found; {} unavailable case(s) hidden. "
                    "Only cases with both an image and matching Mask will be "
                    "used.{}".format(ready, unavailable, initial_note)
                )
            self._finish_scan_progress(
                "Scan complete: {} usable, {} unavailable".format(
                    ready, unavailable
                ),
                True,
            )
        else:
            self.scan_hint.setText("No cases were found in the selected location.")
            self._finish_scan_progress("Scan complete: no usable cases", True)

    def _populate_cases(self, preserve_assignments=False):
        QtWidgets = self.QtWidgets
        ready_indices = [
            index for index, row in enumerate(self.case_rows) if row.get("state") == "ready"
        ]
        val_count = 0
        if len(ready_indices) >= 2:
            val_count = 1
        if len(ready_indices) >= 6:
            val_count = max(2, int(round(len(ready_indices) * 0.2)))
        val_indices = set(ready_indices[-val_count:]) if val_count else set()
        for index, row in enumerate(self.case_rows):
            ready = row.get("state") == "ready"
            if not preserve_assignments or "_selected" not in row:
                row["_selected"] = ready
                row["_split"] = (
                    "validation"
                    if index in val_indices
                    else ("train" if ready else "exclude")
                )
        if self.choose_specific_cases is None or not self.choose_specific_cases.isChecked():
            self._case_population_generation += 1
            self._case_population_timer.stop()
            self.case_table.setRowCount(0)
            self._update_setup_summary()
            return
        self._case_population_generation += 1
        self._case_population_index = 0
        self.case_table.setRowCount(len(self.case_rows))
        self.case_table.setEnabled(False)
        self.setup_summary.setText("Loading case choices without blocking this window...")
        self._case_population_timer.start(0)

    def _populate_case_chunk(self):
        QtWidgets = self.QtWidgets
        start = self._case_population_index
        stop = min(len(self.case_rows), start + 40)
        for index in range(start, stop):
            row = self.case_rows[index]
            ready = row.get("state") == "ready"
            check = QtWidgets.QCheckBox()
            check.setChecked(bool(row.get("_selected", ready)))
            check.setEnabled(ready)
            check.stateChanged.connect(
                lambda state, case_index=index: self._case_choice_changed(
                    case_index, selected=bool(state)
                )
            )
            wrapper = QtWidgets.QWidget()
            wrapper_layout = QtWidgets.QHBoxLayout(wrapper)
            wrapper_layout.setContentsMargins(8, 0, 0, 0)
            wrapper_layout.addWidget(check)
            wrapper_layout.addStretch(1)
            self.case_table.setCellWidget(index, 0, wrapper)
            self.case_table.setItem(index, 1, QtWidgets.QTableWidgetItem(str(row.get("case_id"))))
            split = QtWidgets.QComboBox()
            split.addItems(["Train", "Validation", "Exclude"])
            split.setCurrentText(str(row.get("_split") or "exclude").title())
            split.setEnabled(ready)
            split.currentTextChanged.connect(
                lambda value, case_index=index: self._case_choice_changed(
                    case_index, split=str(value).lower()
                )
            )
            self.case_table.setCellWidget(index, 2, split)
            label = {
                "ready": row.get("detail") or "Ready",
                "image_missing": "Image missing",
                "mask_missing": "Mask missing",
            }.get(row.get("state"), str(row.get("state") or "Needs attention"))
            item = QtWidgets.QTableWidgetItem(label)
            if not ready:
                item.setForeground(self.QtGui.QColor("#b42318"))
            self.case_table.setItem(index, 3, item)
        self._case_population_index = stop
        if stop < len(self.case_rows):
            return
        self._case_population_timer.stop()
        self.case_table.setEnabled(True)
        self._apply_case_filters()
        self._update_setup_summary()

    def _case_choice_changed(self, index, selected=None, split=None):
        if not 0 <= int(index) < len(self.case_rows):
            return
        row = self.case_rows[int(index)]
        if selected is not None:
            row["_selected"] = bool(selected)
        if split is not None:
            row["_split"] = str(split).lower()
        self._update_setup_summary()

    def _selected_cases(self):
        result = []
        choose_specific = bool(
            self.choose_specific_cases
            and self.choose_specific_cases.isChecked()
        )
        for index, row in enumerate(self.case_rows):
            if (
                row.get("state") != "ready"
                or (choose_specific and not bool(row.get("_selected", False)))
            ):
                continue
            split_value = str(row.get("_split") or "train").lower()
            if split_value == "exclude":
                continue
            item = dict(row)
            item["split"] = "val" if split_value == "validation" else "train"
            result.append(item)
        return result

    def _update_setup_summary(self):
        if not self.task_combo.currentText().strip():
            self.setup_summary.setText("Choose or enter an annotation task.")
            self.start_button.setEnabled(False)
            return
        selected = self._selected_cases()
        train = sum(row.get("split") == "train" for row in selected)
        val = sum(row.get("split") == "val" for row in selected)
        strategy = "CLoPA-IN" if self.light_radio.isChecked() else "CLoPA-CN"
        if train < 1:
            self.setup_summary.setText("Select at least one training case.")
            self.start_button.setEnabled(False)
            return
        minimum_val = int(self.config.get("minimum_validation_cases_for_auto_selection", 2))
        suffix = ""
        if val < minimum_val:
            suffix = " · New model will not be selected automatically"
        self.setup_summary.setText(
            "{} training / {} validation / {} / up to {} epochs{}{}".format(
                train,
                val,
                strategy,
                self.epochs.value(),
                suffix,
                (
                    ""
                    if self.choose_specific_cases.isChecked()
                    else " / all matching Masks"
                ),
            )
        )
        self.start_button.setEnabled(not self._active_job_for_task())

    def _active_job_for_task(self):
        task_id = self.current_task_id()
        for path in jobs_dir(self.workspace).glob("*/status.json"):
            status = read_json(path, {}) or {}
            if safe_slug(status.get("task_id")) == task_id and str(status.get("status") or "") in ACTIVE_STATUSES:
                return path.parent
        return None

    def start_training(self):
        if self.last_scan_signature != self._current_data_signature():
            self.setup_summary.setText(
                "The data source, path, or Target Mask names changed. Scan Data "
                "again before starting training."
            )
            return
        selected = self._selected_cases()
        train_count = sum(row.get("split") == "train" for row in selected)
        if train_count < 1:
            return
        task_name = self.task_combo.currentText().strip()
        if not task_name:
            self.setup_summary.setText("Choose or enter an annotation task.")
            return
        task_id = safe_slug(task_name)
        mask_names = [
            value.strip() for value in self.mask_edit.text().split(",") if value.strip()
        ]
        model_id = "model_{}_{}".format(
            time.strftime("%Y%m%dT%H%M%S"), uuid.uuid4().hex[:8]
        )
        job_id = "train_{}_{}".format(
            time.strftime("%Y%m%dT%H%M%S"), uuid.uuid4().hex[:8]
        )
        task_root = task_dir(self.workspace, task_id)
        output_model_dir = task_root / "models" / model_id
        start_kind = self.start_model.currentData()
        parent = selected_model(self.workspace, task_id) if start_kind == "current" else None
        parent_task = find_task(self.workspace, task_id)
        execution_backend, remote_profile_id = (
            self.remote_selector.selection()
            if self.remote_selector is not None
            else ("local", "")
        )
        if execution_backend == "remote" and not remote_profile_id:
            self.setup_summary.setText(
                "Choose a saved remote server or use This workstation."
            )
            return
        base_model = (
            resolve_registered_model_dir(self.workspace, parent, parent_task)
            if parent
            else official_model_dir(self.config)
        )
        base_audit = (
            audit_model_dir(base_model, include_checksum=False)
            if parent or execution_backend == "local"
            else {"compatible": True, "remote_official_model": True}
        )
        if not base_audit.get("compatible"):
            self.setup_summary.setText(
                "The starting model is incomplete: {}".format(
                    ", ".join(base_audit.get("missing") or [])
                )
            )
            return
        job_dir = jobs_dir(self.workspace) / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        request = {
            "schema_version": "nninteractive_task_training_request.v1",
            "job_id": job_id,
            "task_id": task_id,
            "task_name": task_name,
            "workspace": str(self.workspace),
            "source_mode": (
                "mcs"
                if self.source_combo.currentData() == "mcs_refresh"
                else "prepared"
            ),
            "label_source": str(self.source_combo.currentData()),
            "mcs_dir": self.mcs_edit.text().strip(),
            "image_root": (
                self.image_root_edit.text().strip()
            ),
            "prepared_root": self.image_root_edit.text().strip(),
            "label_root": (
                self.prepared_label_edit.text().strip()
                if self.source_combo.currentData() == "exported_masks"
                else ""
            ),
            "mask_names": mask_names,
            "initial_mask_source": (
                "none"
                if str(self.training_goal.currentData() or "general")
                == "start_empty"
                else str(
                    self.initial_mask_source_combo.currentData()
                    or "none"
                )
            ),
            "initial_mask_names": [
                value.strip()
                for value in self.initial_mask_edit.text().split(",")
                if value.strip()
            ],
            "initial_mask_root": self.initial_mask_root_edit.text().strip(),
            "cases": selected,
            "strategy": "clopa_in" if self.light_radio.isChecked() else "clopa_conv",
            "training_goal": str(
                self.training_goal.currentData() or "general"
            ),
            "epochs": int(self.epochs.value()),
            "mirror_policy": str(self.mirror_policy.currentData() or "auto"),
            "base_model_dir": str(base_model.resolve()),
            "parent_model_id": str(parent.get("model_id") if parent else "official"),
            "model_id": model_id,
            "output_model_dir": str(output_model_dir.resolve()),
            "mimics_exe": str(self.context.get("mimics_exe") or ""),
            "execution_backend": execution_backend,
            "remote_profile_id": remote_profile_id,
            "created_at_epoch": time.time(),
        }
        try:
            save_preferences("nninteractive_training", {
                "image_root": request.get("image_root") or "",
                "mcs_dir": request.get("mcs_dir") or "",
                "label_root": request.get("label_root") or "",
                "initial_mask_root": request.get("initial_mask_root") or "",
            })
        except Exception:
            pass
        write_json_atomic(job_dir / "request.json", request)
        write_json_atomic(
            job_dir / "status.json",
            {
                "schema_version": "nninteractive_task_job.v1",
                "job_id": job_id,
                "task_id": task_id,
                "task_name": task_name,
                "status": "created",
                "phase": "created",
                "request_path": str(job_dir / "request.json"),
                "control_path": str(job_dir / "control.json"),
                "log_path": str(job_dir / "job.log"),
                "created_at_epoch": time.time(),
                "updated_at_epoch": time.time(),
                "execution_backend": execution_backend,
                "remote_profile_id": remote_profile_id,
            },
        )
        if execution_backend == "remote":
            spec_path = job_dir / "remote_launch.json"
            write_json_atomic(
                spec_path,
                {
                    "schema_version": "mimics_remote_launch.v1",
                    "kind": "nninteractive",
                    "job_id": job_id,
                    "job_dir": str(job_dir),
                    "status_path": str(job_dir / "status.json"),
                    "remote_profile_id": remote_profile_id,
                    "created_at_epoch": time.time(),
                },
            )
            process = _launch_process(
                [
                    str(self.python),
                    str(ROOT / "tools" / "remote_training_controller.py"),
                    "run",
                    "--spec",
                    str(spec_path),
                ],
                output_path=job_dir / "job.log",
            )
            status = read_json(job_dir / "status.json", {}) or {}
            status.update(
                {
                    "controller_pid": process.pid,
                    "launcher_pid": process.pid,
                    "remote_launch_spec": str(spec_path),
                    "updated_at_epoch": time.time(),
                }
            )
            write_json_atomic(job_dir / "status.json", status)
        else:
            _launch_process(
                [
                    str(self.python),
                    str(self.pipeline),
                    "run",
                    "--job-dir",
                    str(job_dir),
                ]
            )
        self.current_job_dir = job_dir
        self.metrics = []
        self.last_metric_epoch = 0
        self.log_offset = 0
        self.log_text.clear()
        self.tabs.setCurrentWidget(self.progress_tab)
        self.refresh_status()

    def refresh_current_job(self):
        task_id = self.current_task_id()
        candidates = []
        for path in jobs_dir(self.workspace).glob("*/status.json"):
            status = read_json(path, {}) or {}
            if safe_slug(status.get("task_id")) == task_id:
                candidates.append(
                    (float(status.get("updated_at_epoch") or 0), path.parent, status)
                )
        candidates.sort(reverse=True)
        if candidates:
            self.current_job_dir = candidates[0][1]
            self.current_status = candidates[0][2]
        else:
            self.current_job_dir = None
            self.current_status = {}
        self.log_offset = 0
        if hasattr(self, "log_text"):
            self.log_text.clear()
        self.refresh_status()

    def refresh_status(self):
        self._poll_model_io()
        self._apply_pending_scan_result()
        registry_signature = _task_model_registry_signature(
            self.workspace,
            self.current_task_id(),
        )
        if registry_signature != self.last_model_registry_signature:
            self.last_model_registry_signature = registry_signature
            self.refresh_models()
        if self.current_job_dir:
            status = read_json(self.current_job_dir / "status.json", {}) or {}
            self.current_status = status
        else:
            status = {}
        state = str(status.get("status") or "")
        self.header_state.setText(
            {
                "training": "Training",
                "exporting_labels": "Training",
                "preparing_data": "Preparing",
                "preparing_remote": "Preparing",
                "connecting_remote": "Connecting",
                "uploading": "Uploading",
                "starting_remote": "Starting",
                "waiting_for_remote_gpu": "Waiting",
                "reconnecting_remote": "Reconnecting",
                "remote_control_unavailable": "Connection issue",
                "finalizing_remote": "Finishing",
                "downloading": "Downloading",
                "waiting_for_gpu": "Waiting",
                "validating": "Validating",
                "registering": "Finishing",
                "paused": "Paused",
                "completed": "Completed",
                "cancelled": "Stopped",
                "abandoned": "Abandoned locally",
                "failed": "Needs attention",
            }.get(state, "Ready")
        )
        self.progress_title.setText(status_summary(status) if status else "No training is running")
        detail = str(
            status.get("error")
            or status.get("phase")
            or "Start training from Training Setup."
        )
        phase = str(status.get("phase") or "").strip().lower()
        if state == "training":
            if phase.startswith("validat"):
                detail = (
                    "Validating the current training epoch. The remaining "
                    "model comparison and registration stages are not yet complete."
                )
            elif phase.startswith("final") or "verif" in phase:
                detail = (
                    "Saving and verifying the trained checkpoint. Mimics remains "
                    "available for annotation."
                )
            else:
                detail = (
                    "Learning from the selected cases in the background. "
                    "Mimics remains available for annotation."
                )
        elif state == "exporting_labels":
            progress = status.get("label_export_progress") or {}
            case_id = str(progress.get("case_id") or status.get("current_case") or "")
            detail = (
                "Reading the saved target Mask{} and converting it to the source image grid."
            ).format(" for " + case_id if case_id else "")
        elif state == "preparing_data":
            if phase == "reusing_prepared_cases":
                detail = (
                    "Reusing verified prepared arrays on the remote server; "
                    "no image conversion is running."
                )
            elif phase == "reusing_local_prepared_data":
                detail = (
                    "Reusing verified local source-grid inputs before remote "
                    "transfer."
                )
            else:
                detail = "Checking image-label geometry and preparing the selected cases."
        elif state == "preparing_remote":
            if phase == "reusing_local_prepared_data":
                detail = (
                    "Reusing verified local source-grid inputs before remote "
                    "transfer."
                )
            else:
                detail = (
                    "Preparing source-grid images and Target Masks locally before "
                    "the remote transfer."
                )
        elif state == "connecting_remote":
            detail = "Checking the saved SSH server, runtime image, model, and GPU."
        elif state == "uploading":
            if phase == "verifying_remote_dataset_cache":
                detail = "Checking remote data cache ({}/{} cases); unchanged archives are not uploaded.".format(
                    status.get("dataset_case_completed") or 0,
                    status.get("dataset_case_total") or 0,
                )
            elif phase == "remote_dataset_cache_hit":
                detail = (
                    "The unchanged training data is already verified on the "
                    "server. Only this job's settings are being sent."
                )
            else:
                detail = "Uploading training data in the background."
        elif state == "starting_remote":
            if phase == "extracting_remote_dataset":
                detail = "Making verified remote data available to the container ({}/{} cases).".format(
                    status.get("remote_dataset_extract_index") or 0,
                    status.get("remote_dataset_extract_total") or 0,
                )
            else:
                detail = (
                    "Starting an isolated container on GPU {}."
                ).format(status.get("remote_gpu_device") or "automatic")
        elif state == "waiting_for_remote_gpu":
            detail = (
                "Waiting for GPU {} without blocking Mimics."
            ).format(status.get("remote_gpu_device") or "automatic")
        elif state == "reconnecting_remote":
            detail = (
                "The SSH connection was interrupted. Training remains on the "
                "server while the controller reconnects."
            )
        elif state == "remote_control_unavailable":
            detail = (
                "The server is reachable but Docker status is temporarily "
                "unavailable. The task remains active and is not assumed stopped."
            )
        elif state == "finalizing_remote":
            detail = (
                "Remote training finished. Verifying and packaging the model "
                "before local registration."
            )
        elif state == "downloading":
            detail = "Downloading and verifying the trained model."
        elif state == "waiting_for_gpu":
            detail = "Waiting without blocking Mimics. Pause or stop this task at any time."
        elif state == "validating":
            detail = (
                "Comparing the new model with the active annotation model on "
                "the validation cases."
            )
        elif state == "registering":
            detail = (
                "Verifying the loaded checkpoint identity and publishing this "
                "model version. Training is not complete until this finishes."
            )
        elif state == "completed":
            outcome = str(status.get("selection_outcome") or "")
            quality = status.get("quality") or {}
            if outcome == "new_model_selected":
                detail = (
                    "Validation passed. The new model is now used for this task "
                    "({:+.1f}% AUC)."
                ).format(float(quality.get("delta_auc") or 0.0) * 100.0)
            elif outcome == "current_model_retained":
                detail = (
                    "Training completed, but the active annotation model remains selected "
                    "because the candidate did not pass the quality check."
                )
            else:
                detail = (
                    "Training completed. This version was saved without automatic "
                    "selection because no sufficient validation set was provided."
                )
        elif state == "paused":
            detail = "Training is paused and its GPU memory has been released."
        elif state == "cancelled":
            detail = "Training was stopped. The incomplete model was removed."
        elif state == "abandoned":
            detail = str(
                status.get("remote_abandon_warning")
                or "Local monitoring ended, but the remote container state is unknown."
            )
        if state in {"preparing_remote", "connecting_remote", "uploading"}:
            archive_reused = status.get(
                "local_dataset_archive_cache_reused"
            )
            archive_total = status.get("local_dataset_archive_cache_total")
            if archive_reused is not None and archive_total is not None:
                detail += " Local packaging reused {}/{} case archives.".format(
                    archive_reused, archive_total
                )
        self.progress_detail.setText(detail)
        self._refresh_diagnosis(status, state)
        value = int(status.get("progress_percent") or 0)
        if state == "completed":
            value = 100
        elif state in ACTIVE_STATUSES:
            value = min(99, value)
        self.progress_bar.setValue(max(0, min(100, value)))
        progress_stage = {
            "validating": "Comparing models",
            "registering": "Registering model",
            "completed": "Completed",
            "failed": "Needs attention",
            "cancelled": "Stopped",
            "abandoned": "Abandoned locally",
        }.get(state, "")
        if state == "training" and (
            "verif" in phase or phase.startswith("final")
        ):
            progress_stage = "Checkpoint ready; final checks"
        self.progress_bar.setFormat(
            "%p% · {}".format(progress_stage)
            if progress_stage
            else "%p%"
        )
        latest = status.get("latest_epoch") or {}
        displayed_loss = status.get("loss")
        if displayed_loss is None:
            displayed_loss = latest.get("train_loss")
        self.loss_label.setText(
            "Loss  {:.4f}".format(float(displayed_loss))
            if displayed_loss is not None
            else "Loss  -"
        )
        validation = latest.get("validation") or {}
        quality = status.get("quality") or {}
        final_validation = bool(
            state == "completed" and quality.get("mode_comparisons")
        )
        auc = (
            quality.get("candidate_auc")
            if final_validation
            else validation.get("trajectory_auc")
        )
        self.auc_label.setText(
            "{}  {:.4f}".format(
                "Final validation AUC" if final_validation else "Validation AUC",
                float(auc),
            )
            if auc is not None
            else "Validation AUC  -"
        )
        auc_details = []
        if final_validation:
            metric_rows = []
            for title, mode in (
                ("Empty start", "empty_mask"),
                ("Existing Mask", "real_initial_mask"),
            ):
                comparison = (quality.get("mode_comparisons") or {}).get(mode) or {}
                metric_rows.append(
                    (
                        title,
                        comparison.get("candidate_auc"),
                        comparison.get("starting_mask_dice"),
                    )
                )
        else:
            metric_rows = [
                (
                    "Empty start",
                    validation.get("empty_mask_trajectory_auc"),
                    validation.get("empty_mask_baseline_dice"),
                ),
                (
                    "Existing Mask",
                    validation.get("initial_mask_trajectory_auc"),
                    validation.get("initial_mask_baseline_dice"),
                ),
            ]
        for title, mode_auc, mode_baseline in metric_rows:
            if mode_auc is None:
                continue
            detail = "{} AUC {:.4f}".format(title, float(mode_auc))
            if mode_baseline is not None:
                detail += " (baseline {:.4f}, {:+.4f})".format(
                    float(mode_baseline),
                    float(mode_auc) - float(mode_baseline),
                )
            auc_details.append(detail)
        self.auc_detail_label.setText("  ·  ".join(auc_details))
        baseline_auc = None
        baseline_model = selected_model(self.workspace, self.current_task_id())
        if baseline_model:
            # The current model's own validation AUC is the reference the
            # candidate must beat for automatic selection.
            baseline_quality = baseline_model.get("quality") or {}
            if baseline_quality.get("candidate_auc") is not None:
                baseline_auc = float(baseline_quality["candidate_auc"])
        self.curve.set_baseline(baseline_auc)
        if auc is not None and baseline_auc is not None:
            delta = float(auc) - baseline_auc
            if delta >= 0:
                self.comparison_label.setText(
                    "Candidate is {:+.1f}% AUC vs the current model so far.".format(
                        delta * 100.0
                    )
                )
            else:
                self.comparison_label.setText(
                    "Not yet better than the current model ({:.1f}% AUC so far).".format(
                        delta * 100.0
                    )
                )
        elif auc is not None and state in ACTIVE_STATUSES:
            self.comparison_label.setText(
                "No current model to compare against yet; this run will set the first reference."
            )
        else:
            self.comparison_label.setText("")
        self.best_label.setText(
            "Best AUC  {:.4f}".format(float(status["best_score"]))
            if status.get("best_score") not in (None, float("-inf"))
            else "Best AUC  -"
        )
        history = status.get("metrics_history") or []
        if history:
            self.metrics = list(history)
            self.last_metric_epoch = max(
                [int(row.get("epoch") or 0) for row in self.metrics] or [0]
            )
            self.curve.set_points(self.metrics)
        elif latest and int(latest.get("epoch") or 0) > self.last_metric_epoch:
            self.last_metric_epoch = int(latest.get("epoch") or 0)
            self.metrics.append(
                {
                    "epoch": self.last_metric_epoch,
                    "train_loss": latest.get("train_loss"),
                    "validation_auc": validation.get("trajectory_auc"),
                }
            )
            self.curve.set_points(self.metrics)
        created = float(status.get("created_at_epoch") or 0)
        if created:
            self.elapsed_label.setText(
                "Elapsed {}".format(self._duration(time.time() - created))
            )
        else:
            self.elapsed_label.setText("")
        self._append_log_incremental()
        self._update_progress_actions(status)
        terminal_signature = (
            "{}:{}".format(self.current_job_dir, state)
            if state in ("completed", "failed", "cancelled", "abandoned")
            else ""
        )
        if terminal_signature and terminal_signature != self.last_terminal_signature:
            self.last_terminal_signature = terminal_signature
            self.refresh_tasks()
            self.refresh_models()

    def _duration(self, seconds):
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return "{:02d}:{:02d}:{:02d}".format(hours, minutes, secs)

    def _append_log_incremental(self):
        if not self.current_job_dir:
            return
        path = self.current_job_dir / "job.log"
        if not path.is_file():
            return
        bar = self.log_text.verticalScrollBar()
        following = bar.value() >= max(0, bar.maximum() - 4)
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(self.log_offset)
                text = handle.read()
                self.log_offset = handle.tell()
        except Exception:
            return
        if not text:
            return
        self.log_text.moveCursor(self.QtGui.QTextCursor.End)
        self.log_text.insertPlainText(text)
        if following:
            bar.setValue(bar.maximum())
        else:
            self.new_log_button.setVisible(True)

    def follow_latest_log(self):
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
        self.new_log_button.setVisible(False)

    def _update_progress_actions(self, status):
        state = str(status.get("status") or "")
        active = state in ACTIVE_STATUSES
        remote = str(status.get("execution_backend") or "") == "remote"
        abandonable = bool(
            remote
            and status.get("remote_state_unknown")
            and state not in TERMINAL_STATUSES
        )
        self.pause_button.setEnabled(
            active and not remote and state not in ("pausing", "stopping")
        )
        self.resume_button.setEnabled(
            state in ("paused", "failed") and not remote
        )
        self.stop_button.setText(
            "Abandon Locally" if abandonable else "Stop Training"
        )
        self.stop_button.setEnabled(abandonable or active or state == "paused")
        self.open_log_button.setEnabled(bool(self.current_job_dir))

    def request_primary_stop(self):
        if not self.current_job_dir:
            return
        status = read_json(self.current_job_dir / "status.json", {}) or {}
        abandonable = bool(
            str(status.get("execution_backend") or "") == "remote"
            and status.get("remote_state_unknown")
            and str(status.get("status") or "") not in TERMINAL_STATUSES
        )
        if not abandonable:
            self.request_action("stop")
            return
        answer = self.QtWidgets.QMessageBox.warning(
            self.window,
            "Abandon Remote Task Locally",
            "Stop waiting on this workstation?\n\nThe server cannot confirm "
            "whether the container stopped. It may still use GPU or disk "
            "resources. An administrator must inspect the recorded container "
            "name.",
            self.QtWidgets.QMessageBox.Yes | self.QtWidgets.QMessageBox.No,
            self.QtWidgets.QMessageBox.No,
        )
        if answer != self.QtWidgets.QMessageBox.Yes:
            return
        _launch_process(
            [
                str(self.python),
                str(ROOT / "tools" / "remote_training_controller.py"),
                "abandon",
                "--status",
                str(self.current_job_dir / "status.json"),
            ]
        )
        self.progress_detail.setText(
            "Ending local monitoring without claiming the remote GPU is free."
        )

    def request_action(self, action):
        if not self.current_job_dir:
            return
        write_json_atomic(
            self.current_job_dir / "control.json",
            {"action": action, "requested_at_epoch": time.time()},
        )
        self.refresh_status()

    def resume_training(self):
        if not self.current_job_dir:
            return
        for path in (
            self.current_job_dir / "control.json",
            self.current_job_dir / "trainer_cancel.request",
        ):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        _launch_process(
            [
                str(self.python),
                str(self.pipeline),
                "resume",
                "--job-dir",
                str(self.current_job_dir),
            ]
        )
        self.refresh_status()

    def open_log(self):
        if not self.current_job_dir:
            return
        path = self.current_job_dir / "job.log"
        try:
            if os.name == "nt":
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception:
            pass

    def open_job_folder(self):
        if not self.current_job_dir:
            return
        try:
            if os.name == "nt":
                os.startfile(str(self.current_job_dir))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(self.current_job_dir)])
            else:
                subprocess.Popen(["xdg-open", str(self.current_job_dir)])
        except Exception:
            pass

    def _refresh_diagnosis(self, status, state):
        """Show the aggregated failure diagnosis when a job needs attention."""
        from nninteractive_finetune_pipeline import diagnose_job

        show = state == "failed" and bool(self.current_job_dir)
        if not show:
            self.diagnosis_title.setVisible(False)
            self.diagnosis_detail.setVisible(False)
            self.diagnosis_log.setVisible(False)
            self.open_job_folder_button.setVisible(False)
            return
        try:
            diagnosis = diagnose_job(str(self.current_job_dir))
        except Exception as exc:
            self.diagnosis_detail.setText(
                "The failure diagnosis could not be read: {}".format(exc)
            )
            self.diagnosis_title.setVisible(True)
            self.diagnosis_detail.setVisible(True)
            self.diagnosis_log.setVisible(False)
            self.open_job_folder_button.setVisible(True)
            return
        lines = [str(diagnosis.get("hint") or "")]
        error = str(diagnosis.get("error") or "")
        if error:
            lines.append("Error: {}".format(error))
        error_line = str(diagnosis.get("error_line") or "")
        if error_line:
            lines.append("At {}".format(error_line))
        chain = [
            str(stage) for stage in diagnosis.get("stage_chain") or [] if stage
        ]
        if chain:
            lines.append("Stages: {}".format(" -> ".join(chain)))
        cleanup = diagnosis.get("artifact_cleanup") or {}
        removed = cleanup.get("removed") or []
        if removed:
            lines.append(
                "Rebuildable data was cleaned up ({} items); logs and status were kept.".format(
                    len(removed)
                )
            )
        self.diagnosis_detail.setText("\n".join(line for line in lines if line))
        tail = str(diagnosis.get("job_log_tail") or "")
        trainer_tail = str(diagnosis.get("trainer_log_tail") or "")
        blocks = []
        if trainer_tail:
            blocks.append("--- trainer.log (last lines) ---\n{}".format(trainer_tail))
        if tail:
            blocks.append("--- job.log (last lines) ---\n{}".format(tail))
        self.diagnosis_log.setPlainText("\n\n".join(blocks))
        self.diagnosis_log.setMaximumHeight(140)
        self.diagnosis_title.setVisible(True)
        self.diagnosis_detail.setVisible(True)
        self.diagnosis_log.setVisible(True)
        self.open_job_folder_button.setVisible(True)

    def refresh_models(self):
        task_id = self.current_task_id()
        self.last_model_registry_signature = _task_model_registry_signature(
            self.workspace,
            task_id,
        )
        task = find_task(self.workspace, task_id) or {}
        recommended = str(task.get("recommended_model_id") or "")
        current = selected_model(self.workspace, task_id)
        if current:
            quality = current.get("quality") or {}
            self.current_model_title.setText(
                "Active annotation model · {}".format(
                    _format_time(current.get("created_at_epoch"))
                )
            )
            self.current_model_detail.setText(
                "{} · {} training + {} validation · validation {}".format(
                    strategy_display_name(current.get("strategy")),
                    current.get("train_case_count", 0),
                    current.get("validation_case_count", 0),
                    _format_delta(quality.get("delta_auc")),
                )
            )
        else:
            self.current_model_title.setText("No active annotation model")
            self.current_model_detail.setText(
                "Train and validate a model, or select an unverified version explicitly."
            )
        self.export_model_button.setEnabled(
            current is not None and self.model_io_process is None
        )
        self.import_model_button.setEnabled(self.model_io_process is None)
        rows = model_rows(self.workspace, task_id)
        if not self.show_failed.isChecked():
            rows = [
                row
                for row in rows
                if row.get("state")
                not in ("failed", "corrupt", "incompatible", "not_improved")
            ]
        for row_index in range(self.model_table.rowCount()):
            action_cell = self.model_table.cellWidget(row_index, 5)
            if action_cell is not None:
                action_cell.hide()
                self.model_table.removeCellWidget(row_index, 5)
                action_cell.deleteLater()
        self.model_table.clearContents()
        self.model_table.setRowCount(0)
        self.model_table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            quality = row.get("quality") or {}
            self.model_table.setItem(
                index, 0, self.QtWidgets.QTableWidgetItem(_format_time(row.get("created_at_epoch")))
            )
            self.model_table.setItem(
                index,
                1,
                self.QtWidgets.QTableWidgetItem(
                    str(row.get("display_name") or row.get("model_id") or "")
                ),
            )
            self.model_table.setItem(
                index,
                2,
                self.QtWidgets.QTableWidgetItem(
                    strategy_display_name(row.get("strategy"))
                ),
            )
            self.model_table.setItem(
                index,
                3,
                self.QtWidgets.QTableWidgetItem(
                    "{} + {}".format(
                        row.get("train_case_count", 0),
                        row.get("validation_case_count", 0),
                    )
                ),
            )
            validation = _format_delta(quality.get("delta_auc"))
            if row.get("state") == "not_improved":
                validation = "Did not improve"
            elif row.get("state") == "unverified":
                validation = "Not verified"
            self.model_table.setItem(
                index, 4, self.QtWidgets.QTableWidgetItem(validation)
            )
            button = self.QtWidgets.QPushButton(
                "Active"
                if row.get("model_id") == recommended
                else "Set Active"
            )
            button.setEnabled(
                model_is_usable(row)
                and row.get("model_id") != recommended
            )
            button.setMinimumWidth(96)
            button.clicked.connect(
                lambda _checked=False, model_id=row.get("model_id"): self.use_model(model_id)
            )
            button_cell = self.QtWidgets.QWidget()
            button_cell.setObjectName("modelActionCell")
            button_cell.setStyleSheet(
                "QWidget#modelActionCell { background: transparent; }"
            )
            button_layout = self.QtWidgets.QHBoxLayout(button_cell)
            button_layout.setContentsMargins(6, 3, 6, 3)
            button_layout.addWidget(button)
            self.model_table.setCellWidget(index, 5, button_cell)
            self.model_table.setRowHeight(index, 42)

    def _start_model_io(self, command, action):
        if self.model_io_process is not None:
            return
        kwargs = {
            "cwd": str(ROOT),
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
        }
        if os.name == "nt":
            kwargs["creationflags"] = getattr(
                subprocess, "CREATE_NO_WINDOW", 0x08000000
            )
        self.model_io_process = subprocess.Popen(command, **kwargs)
        self.model_io_action = action
        self.model_io_status.setText(
            "{} is running outside Mimics. Annotation can continue.".format(action)
        )
        self.import_model_button.setEnabled(False)
        self.export_model_button.setEnabled(False)

    def _poll_model_io(self):
        process = self.model_io_process
        if process is None or process.poll() is None:
            return
        try:
            output = process.communicate()[0].strip()
        except Exception as exc:
            output = str(exc)
        action = self.model_io_action
        returncode = process.returncode
        self.model_io_process = None
        self.model_io_action = ""
        if returncode == 0:
            self.model_io_status.setText(
                "{} completed. {}".format(action, output.splitlines()[-1] if output else "")
            )
            self.refresh_tasks()
            self.refresh_models()
        else:
            self.model_io_status.setText(
                "{} failed. {}".format(action, output or "See the process log.")
            )
            self.import_model_button.setEnabled(True)
            self.export_model_button.setEnabled(
                selected_model(self.workspace, self.current_task_id()) is not None
            )

    def import_model_package(self):
        choose_open_file_async(
            self.QtCore,
            self.QtWidgets,
            self.window,
            "Import nnInteractive Model Package",
            str(Path.home()),
            "AI model packages (*.zip)",
            self._import_model_package_path,
            button=self.import_model_button,
        )

    def _import_model_package_path(self, path):
        if not path:
            return
        self._start_model_io(
            [
                str(self.python),
                str(ROOT / "tools" / "ai_model_bundle.py"),
                "import-nninteractive",
                "--workspace",
                str(self.workspace),
                "--bundle",
                str(path),
                "--set-current",
            ],
            "Model import",
        )

    def export_current_model(self):
        task_id = self.current_task_id()
        current = selected_model(self.workspace, task_id)
        if not current:
            self.model_io_status.setText(
                "Select an active annotation model before exporting a package."
            )
            return
        suggested = Path.home() / "{}_{}.zip".format(
            safe_slug(task_id), safe_slug(current.get("model_id"))
        )
        choose_save_file_async(
            self.QtCore,
            self.QtWidgets,
            self.window,
            "Export nnInteractive Model Package",
            str(suggested),
            "AI model packages (*.zip)",
            lambda path: self._export_current_model_path(path, task_id, current),
            button=self.export_model_button,
        )

    def _export_current_model_path(self, path, task_id, current):
        if not path:
            return
        self._start_model_io(
            [
                str(self.python),
                str(ROOT / "tools" / "ai_model_bundle.py"),
                "export-nninteractive",
                "--workspace",
                str(self.workspace),
                "--task-id",
                str(task_id),
                "--model-id",
                str(current.get("model_id")),
                "--output",
                str(path),
            ],
            "Model export",
        )

    def use_model(self, model_id):
        registry = load_registry(self.workspace)
        task_id = self.current_task_id()
        changed = False
        for task in registry.get("tasks") or []:
            if safe_slug(task.get("task_id")) == task_id:
                if any(row.get("model_id") == model_id for row in task.get("models") or []):
                    task["recommended_model_id"] = str(model_id)
                    task["updated_at_epoch"] = time.time()
                    changed = True
                    write_json_atomic(task_dir(self.workspace, task_id) / "task.json", task)
                break
        if changed:
            save_registry(self.workspace, registry)
            project_path = str(self.context.get("project_path") or "").strip()
            if project_path:
                bindings_path = self.workspace / "project_bindings.json"
                payload = read_json(bindings_path, {}) or {}
                bindings = payload.get("bindings") or {}
                binding_key = os.path.normcase(
                    os.path.abspath(project_path)
                ).lower()
                bindings[binding_key] = {
                    "project_path": os.path.abspath(project_path),
                    "task_id": str(task_id),
                    "model_id": str(model_id),
                    "updated_at_epoch": time.time(),
                }
                payload["schema_version"] = "nninteractive_project_bindings.v2"
                payload["bindings"] = bindings
                write_json_atomic(bindings_path, payload)
            self.refresh_models()
            self._refresh_start_models()

    def on_tab_changed(self, _index):
        if self.tabs.currentWidget() == self.models_tab:
            self.refresh_models()
        elif self.tabs.currentWidget() == self.progress_tab:
            self.refresh_status()


def run_ui(context_path: str) -> int:
    from PySide6 import QtCore, QtGui, QtWidgets

    context = read_json(context_path, {}) or {}
    status_path = context.get("ui_status_path")
    try:
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        configure_application(app, TITLE)
        window = QtWidgets.QWidget()
        ModelCenter(window, context, (QtCore, QtGui, QtWidgets))
        window.show()
        window.raise_()
        window.activateWindow()
        if status_path:
            write_json_atomic(
                status_path,
                {"status": "ready", "pid": os.getpid(), "updated_at_epoch": time.time()},
            )
        result = int(app.exec())
        if status_path:
            write_json_atomic(
                status_path,
                {"status": "closed", "pid": os.getpid(), "updated_at_epoch": time.time()},
            )
        return result
    except Exception as exc:
        if status_path:
            write_json_atomic(
                status_path,
                {
                    "status": "failed",
                    "error": "{}: {}".format(type(exc).__name__, exc),
                    "pid": os.getpid(),
                    "updated_at_epoch": time.time(),
                },
            )
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", required=True)
    args = parser.parse_args(argv)
    return run_ui(args.context)


if __name__ == "__main__":
    raise SystemExit(main())
