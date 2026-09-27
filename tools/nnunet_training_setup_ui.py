#!/usr/bin/env python3
"""External, non-modal nnU-Net training configuration window."""

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

from nnunet_common import (  # noqa: E402
    load_label_sets,
    normalize_request,
    read_json,
    suggest_dataset_id,
    update_status,
    write_json_atomic,
)
from nnunet_jobs import create_job  # noqa: E402
from remote_compute_ui import RemoteComputeSelector  # noqa: E402
from ui_theme import (  # noqa: E402
    choose_existing_directory_async,
    choose_open_file_async,
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
        self.window.setWindowTitle("nnU-Net Training")
        self.window.resize(920, 760)
        self.window.setMinimumSize(780, 650)
        self.window.closeEvent = self._close_event
        self.submitted = False
        self._submission_pending = False
        self._submission_results = Queue()
        self._dataset_id_results = Queue()
        self._submission_generation = 0
        self._submission_deadline = 0.0

        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(22, 18, 22, 18)
        root.setSpacing(12)
        title = QtWidgets.QLabel("nnU-Net Training")
        title.setObjectName("title")
        subtitle = QtWidgets.QLabel(
            "Prepare source-grid labels, let nnU-Net plan the dataset, and train "
            "locally or on a saved remote GPU server. Mimics remains usable."
        )
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(subtitle)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._build_data_tab(), "Data")
        self.tabs.addTab(self._build_model_tab(), "Model and Training")
        self.tabs.addTab(self._build_compute_tab(), "Compute")
        root.addWidget(self.tabs, 1)

        self.status_label = QtWidgets.QLabel(
            "Labels are mapped to the original image grid before nnU-Net preprocessing."
        )
        self.status_label.setObjectName("hint")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)
        actions = QtWidgets.QHBoxLayout()
        actions.addStretch(1)
        cancel = QtWidgets.QPushButton("Cancel")
        cancel.clicked.connect(self.window.close)
        actions.addWidget(cancel)
        self.start_button = QtWidgets.QPushButton("Start Training")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(self._submit)
        actions.addWidget(self.start_button)
        root.addLayout(actions)
        self.window.setCentralWidget(central)
        self._load_context()
        self._submission_timer = self.QtCore.QTimer(self.window)
        self._submission_timer.timeout.connect(self._poll_submission)
        self._submission_timer.start(80)
        self._refresh_source()
        self._refresh_configuration()
        self._refresh_trainer()
        self._refresh_fold()
        self._refresh_continue_training()

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

    def _hint(self, text):
        """One-line explanation label under a form field (A5 minimal plan)."""
        QtWidgets = self.QtWidgets
        label = QtWidgets.QLabel(text)
        label.setObjectName("hint")
        label.setWordWrap(True)
        return label

    def _advanced_toggle(self, group):
        """Collapse-button for an advanced settings group (D2). Hidden by
        default; every field inside keeps its default and one-line hint."""
        QtCore, QtWidgets = self.QtCore, self.QtWidgets
        toggle = QtWidgets.QToolButton()
        toggle.setText("Advanced training settings (Trainer, Fold, GPUs, epochs)")
        toggle.setCheckable(True)
        toggle.setChecked(False)
        toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        toggle.setArrowType(QtCore.Qt.RightArrow)

        def refresh(checked):
            group.setVisible(checked)
            toggle.setArrowType(
                QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow
            )

        toggle.toggled.connect(refresh)
        group.setVisible(False)
        return toggle

    def _path_row(self, edit, title, directory=True, file_filter="All files (*)"):
        QtWidgets = self.QtWidgets
        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(edit, 1)
        button = QtWidgets.QPushButton("Browse...")

        def browse():
            current = edit.text().strip()
            if directory:
                choose_existing_directory_async(
                    self.QtCore,
                    QtWidgets,
                    self.window,
                    title,
                    current,
                    lambda value: edit.setText(str(value)) if value else None,
                    button=button,
                )
            else:
                choose_open_file_async(
                    self.QtCore,
                    QtWidgets,
                    self.window,
                    title,
                    current,
                    file_filter,
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
        self.task_edit = QtWidgets.QLineEdit()
        self.task_edit.setPlaceholderText("Example: abdomen_multi_organ_ct")
        self.task_edit.setToolTip(
            "Names the complete segmentation task/model, not one organ or Mask."
        )
        form.addWidget(QtWidgets.QLabel("Model task name *"), 0, 0)
        form.addWidget(self.task_edit, 0, 1)
        self.dataset_id = QtWidgets.QSpinBox()
        self.dataset_id.setRange(1, 999)
        self.dataset_id.setValue(701)
        self.dataset_id.setToolTip(
            "nnU-Net dataset number. Use a different number for unrelated tasks."
        )
        form.addWidget(QtWidgets.QLabel("Dataset ID *"), 1, 0)
        form.addWidget(self.dataset_id, 1, 1)
        self.dataset_id_hint = self._hint(
            "A number identifying this task. A free number is suggested "
            "automatically; only change it when training a second, unrelated task."
        )
        form.addWidget(self.dataset_id_hint, 2, 0, 1, 2)
        self.modality_combo = QtWidgets.QComboBox()
        for label, value in (("CT", "CT"), ("MRI", "MRI"), ("Other", "Other")):
            self.modality_combo.addItem(label, value)
        form.addWidget(QtWidgets.QLabel("Image modality"), 3, 0)
        form.addWidget(self.modality_combo, 3, 1)
        self.dataset_edit = QtWidgets.QLineEdit()
        self.dataset_edit.setPlaceholderText("Folder containing one subfolder per case")
        form.addWidget(QtWidgets.QLabel("Original image dataset *"), 4, 0)
        form.addWidget(
            self._path_row(self.dataset_edit, "Select original image dataset"), 4, 1
        )
        layout.addWidget(project)

        labels, label_layout = self._surface("Labels")
        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.addItem("Masks in each dataset case", "dataset_masks")
        self.source_combo.addItem("Previously exported Masks", "exported_masks")
        self.source_combo.addItem("Refresh from saved .mcs projects", "mcs_refresh")
        self.source_combo.currentIndexChanged.connect(self._refresh_source)
        label_layout.addWidget(QtWidgets.QLabel("Label source"), 0, 0)
        label_layout.addWidget(self.source_combo, 0, 1)
        self.source_hint = QtWidgets.QLabel()
        self.source_hint.setObjectName("hint")
        self.source_hint.setWordWrap(True)
        label_layout.addWidget(self.source_hint, 1, 0, 1, 2)
        self.label_root_label = QtWidgets.QLabel("Exported Mask folder *")
        self.label_root_edit = QtWidgets.QLineEdit()
        label_layout.addWidget(self.label_root_label, 2, 0)
        self.label_root_widget = self._path_row(
            self.label_root_edit, "Select exported Mask folder"
        )
        label_layout.addWidget(self.label_root_widget, 2, 1)
        self.mcs_label = QtWidgets.QLabel("Saved .mcs folder *")
        self.mcs_edit = QtWidgets.QLineEdit()
        label_layout.addWidget(self.mcs_label, 3, 0)
        self.mcs_widget = self._path_row(self.mcs_edit, "Select saved .mcs folder")
        label_layout.addWidget(self.mcs_widget, 3, 1)

        self.label_table = QtWidgets.QTableWidget(0, 4)
        self.label_table.setHorizontalHeaderLabels(
            ["Output label name", "ID", "Accepted Mask names", "Source rule"]
        )
        self.label_table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.Stretch
        )
        self.label_table.horizontalHeader().setSectionResizeMode(
            2, QtWidgets.QHeaderView.Stretch
        )
        self.label_table.horizontalHeader().setSectionResizeMode(
            3, QtWidgets.QHeaderView.ResizeToContents
        )
        self.label_table.setMinimumHeight(170)
        self.label_table.setAlternatingRowColors(True)
        label_layout.addWidget(self.label_table, 4, 0, 1, 2)
        row_actions = QtWidgets.QHBoxLayout()
        import_button = QtWidgets.QPushButton("Import Label Set...")
        import_button.setToolTip(
            "Load one multi-class task from ModelMap.toml or an nnU-Net dataset.json file."
        )
        import_button.clicked.connect(self._import_label_set)
        add_button = QtWidgets.QPushButton("Add Label")
        add_button.clicked.connect(lambda: self._add_label_row("", ""))
        remove_button = QtWidgets.QPushButton("Remove Selected")
        remove_button.clicked.connect(self._remove_label_rows)
        row_actions.addWidget(import_button)
        row_actions.addWidget(add_button)
        row_actions.addWidget(remove_button)
        row_actions.addStretch(1)
        label_layout.addLayout(row_actions, 5, 0, 1, 2)
        self.missing_policy_combo = QtWidgets.QComboBox()
        self.missing_policy_combo.addItem(
            "Require every configured label (safer)", "require_all"
        )
        self.missing_policy_combo.addItem(
            "Treat a missing Mask as background", "background"
        )
        self.missing_policy_combo.setToolTip(
            "Use the second option only when a missing file means the structure is truly absent or fully annotated as background."
        )
        label_layout.addWidget(QtWidgets.QLabel("Incomplete cases"), 6, 0)
        label_layout.addWidget(self.missing_policy_combo, 6, 1)
        self.overlap_check = QtWidgets.QCheckBox(
            "Allow later labels to replace overlapping voxels"
        )
        self.overlap_check.setToolTip(
            "Off is safer. nnU-Net multiclass labels cannot preserve two classes at one voxel."
        )
        label_layout.addWidget(self.overlap_check, 7, 0, 1, 2)
        label_hint = QtWidgets.QLabel(
            "One model may predict many output labels. Each row defines one model output class; "
            "Use Match one for alternate names of the same Mask; use Union all when one coarse "
            "output class combines several source structures."
        )
        label_hint.setObjectName("hint")
        label_hint.setWordWrap(True)
        label_layout.addWidget(label_hint, 8, 0, 1, 2)
        layout.addWidget(labels)

        library, library_form = self._surface("Storage")
        self.workspace_edit = QtWidgets.QLineEdit()
        library_form.addWidget(QtWidgets.QLabel("Model library"), 0, 0)
        library_form.addWidget(
            self._path_row(self.workspace_edit, "Select nnU-Net model library"), 0, 1
        )
        library_hint = QtWidgets.QLabel(
            "Contains managed jobs, source-grid cache, nnU-Net preprocessing, and portable models."
        )
        library_hint.setObjectName("hint")
        library_hint.setWordWrap(True)
        library_form.addWidget(library_hint, 1, 0, 1, 2)
        layout.addWidget(library)
        layout.addStretch(1)
        scroll.setWidget(body)
        outer.addWidget(scroll)
        return tab

    def _build_model_tab(self):
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

        planning, form = self._surface("Planning")
        self.configuration_combo = QtWidgets.QComboBox()
        self.configuration_combo.addItem("3D full resolution", "3d_fullres")
        self.configuration_combo.addItem("2D slices", "2d")
        self.configuration_combo.addItem("3D low resolution", "3d_lowres")
        self.configuration_combo.currentIndexChanged.connect(self._refresh_configuration)
        form.addWidget(QtWidgets.QLabel("Configuration"), 0, 0)
        form.addWidget(self.configuration_combo, 0, 1)

        self.auto_spacing = QtWidgets.QCheckBox("Let nnU-Net determine spacing")
        self.auto_spacing.setChecked(True)
        self.auto_spacing.toggled.connect(self._refresh_configuration)
        form.addWidget(self.auto_spacing, 1, 0, 1, 2)
        spacing_widget = QtWidgets.QWidget()
        spacing_row = QtWidgets.QHBoxLayout(spacing_widget)
        spacing_row.setContentsMargins(0, 0, 0, 0)
        self.spacing_spins = []
        for name in ("X", "Y", "Z"):
            spacing_row.addWidget(QtWidgets.QLabel(name))
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(0.05, 20.0)
            spin.setDecimals(3)
            spin.setValue(1.0)
            spin.setSuffix(" mm")
            self.spacing_spins.append(spin)
            spacing_row.addWidget(spin)
        form.addWidget(QtWidgets.QLabel("Target spacing"), 2, 0)
        form.addWidget(spacing_widget, 2, 1)

        self.auto_patch = QtWidgets.QCheckBox("Let nnU-Net determine patch size")
        self.auto_patch.setChecked(True)
        self.auto_patch.toggled.connect(self._refresh_configuration)
        form.addWidget(self.auto_patch, 3, 0, 1, 2)
        patch_widget = QtWidgets.QWidget()
        patch_row = QtWidgets.QHBoxLayout(patch_widget)
        patch_row.setContentsMargins(0, 0, 0, 0)
        self.patch_labels = []
        self.patch_spins = []
        for name, value in (("Axis 1", 128), ("Axis 2", 128), ("Axis 3", 96)):
            label = QtWidgets.QLabel(name)
            spin = QtWidgets.QSpinBox()
            spin.setRange(16, 1024)
            spin.setSingleStep(8)
            spin.setValue(value)
            self.patch_labels.append(label)
            self.patch_spins.append(spin)
            patch_row.addWidget(label)
            patch_row.addWidget(spin)
        form.addWidget(QtWidgets.QLabel("Patch size"), 4, 0)
        form.addWidget(patch_widget, 4, 1)
        patch_widget.setToolTip(
            "Patch dimensions use the nnU-Net array order recorded in plans.json, not patient RAS/LPS axis names."
        )

        self.auto_batch = QtWidgets.QCheckBox("Let nnU-Net determine batch size")
        self.auto_batch.setChecked(True)
        self.auto_batch.toggled.connect(self._refresh_configuration)
        form.addWidget(self.auto_batch, 5, 0, 1, 2)
        self.batch_spin = QtWidgets.QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(2)
        form.addWidget(QtWidgets.QLabel("Batch size"), 6, 0)
        form.addWidget(self.batch_spin, 6, 1)
        planning_hint = QtWidgets.QLabel(
            "Automatic planning is the recommended default. Manual spacing or patch changes "
            "are recorded with the model and reused during inference."
        )
        planning_hint.setObjectName("hint")
        planning_hint.setWordWrap(True)
        form.addWidget(planning_hint, 7, 0, 1, 2)
        layout.addWidget(planning)

        training, train_form = self._surface("Training")
        self.training_group = training
        self.trainer_combo = QtWidgets.QComboBox()
        self.trainer_combo.setEditable(True)
        self.trainer_combo.addItem("Configurable standard trainer", "MimicsNNUNetTrainer")
        self.trainer_combo.addItem(
            "Configurable trainer without mirroring",
            "MimicsNNUNetTrainerNoMirroring",
        )
        self.trainer_combo.addItem("Official standard trainer", "nnUNetTrainer")
        self.trainer_combo.addItem("Official trainer without mirroring", "nnUNetTrainerNoMirroring")
        self.trainer_combo.currentIndexChanged.connect(self._refresh_trainer)
        self.trainer_combo.lineEdit().editingFinished.connect(self._refresh_trainer)
        train_form.addWidget(QtWidgets.QLabel("Trainer"), 0, 0)
        train_form.addWidget(self.trainer_combo, 0, 1)
        self.trainer_hint = QtWidgets.QLabel()
        self.trainer_hint.setObjectName("hint")
        self.trainer_hint.setWordWrap(True)
        train_form.addWidget(self.trainer_hint, 1, 0, 1, 2)
        self.epochs_spin = QtWidgets.QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(1000)
        train_form.addWidget(QtWidgets.QLabel("Epochs"), 2, 0)
        train_form.addWidget(self.epochs_spin, 2, 1)
        train_form.addWidget(self._hint(
            "More epochs mean longer training. 1000 is the nnU-Net standard."
        ), 3, 0, 1, 2)
        self.fold_combo = QtWidgets.QComboBox()
        for value in ("0", "1", "2", "3", "4", "all"):
            self.fold_combo.addItem("Fold {}".format(value), value)
        self.fold_combo.currentIndexChanged.connect(self._refresh_fold)
        train_form.addWidget(QtWidgets.QLabel("Fold"), 4, 0)
        train_form.addWidget(self.fold_combo, 4, 1)
        self.fold_hint = self._hint(
            "One of five data splits is held out to measure quality. Keep "
            "Fold 0 unless you need cross-validation; 'all' trains five "
            "models one after another."
        )
        train_form.addWidget(self.fold_hint, 5, 0, 1, 2)
        self.validation_spin = QtWidgets.QDoubleSpinBox()
        self.validation_spin.setRange(0.0, 0.8)
        self.validation_spin.setSingleStep(0.05)
        self.validation_spin.setValue(0.2)
        train_form.addWidget(QtWidgets.QLabel("Validation fraction"), 6, 0)
        train_form.addWidget(self.validation_spin, 6, 1)
        train_form.addWidget(self._hint(
            "Share of cases kept out for quality measurement. 0.2 is "
            "standard. Not used when Fold is 'all'."
        ), 7, 0, 1, 2)
        self.workers_spin = QtWidgets.QSpinBox()
        self.workers_spin.setRange(1, 64)
        self.workers_spin.setValue(4)
        train_form.addWidget(QtWidgets.QLabel("Preprocessing workers"), 8, 0)
        train_form.addWidget(self.workers_spin, 8, 1)
        train_form.addWidget(self._hint(
            "CPU workers for preparing the training data. 4 works for most machines."
        ), 9, 0, 1, 2)
        self.num_gpus_spin = QtWidgets.QSpinBox()
        self.num_gpus_spin.setRange(1, 16)
        self.num_gpus_spin.setValue(1)
        train_form.addWidget(QtWidgets.QLabel("GPU count"), 10, 0)
        train_form.addWidget(self.num_gpus_spin, 10, 1)
        self.gpu_count_hint = self._hint(
            "How many GPUs to use in parallel. If you list specific devices "
            "below, their number must match this count."
        )
        train_form.addWidget(self.gpu_count_hint, 11, 0, 1, 2)
        self.gpu_devices_edit = QtWidgets.QLineEdit()
        self.gpu_devices_edit.setPlaceholderText("Optional, for example 0 or 0,1")
        self.gpu_devices_edit.setToolTip(
            "Restricts local execution to these GPU IDs. Remote GPU selection is configured on the Compute tab."
        )
        train_form.addWidget(QtWidgets.QLabel("Local GPU devices"), 12, 0)
        train_form.addWidget(self.gpu_devices_edit, 12, 1)
        self.gpu_devices_hint = self._hint(
            "Leave empty to let training pick the GPUs automatically. "
            "Example: 0 or 0,1. The number of listed devices must match "
            "the GPU count above."
        )
        train_form.addWidget(self.gpu_devices_hint, 13, 0, 1, 2)
        self.pretrained_edit = QtWidgets.QLineEdit()
        train_form.addWidget(QtWidgets.QLabel("Pretrained checkpoint"), 14, 0)
        self.pretrained_widget = self._path_row(
            self.pretrained_edit,
            "Select nnU-Net pretrained checkpoint",
            directory=False,
            file_filter="PyTorch checkpoint (*.pth);;All files (*)",
        )
        train_form.addWidget(self.pretrained_widget, 14, 1)
        self.continue_check = QtWidgets.QCheckBox("Continue an interrupted matching fold")
        self.continue_check.toggled.connect(self._refresh_continue_training)
        train_form.addWidget(self.continue_check, 15, 0, 1, 2)
        self.tta_check = QtWidgets.QCheckBox("Use mirroring TTA for later inference")
        self.tta_check.setChecked(False)
        train_form.addWidget(self.tta_check, 16, 0, 1, 2)
        # D2: the whole Training group is advanced — every field has a safe
        # default and a one-line hint; annotators should not face them unless
        # they look for them.
        self.advanced_toggle = self._advanced_toggle(training)
        layout.addWidget(self.advanced_toggle)
        layout.addWidget(training)
        layout.addStretch(1)
        scroll.setWidget(body)
        outer.addWidget(scroll)
        return tab

    def _build_compute_tab(self):
        QtWidgets = self.QtWidgets
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(12, 14, 12, 12)
        self.remote_selector = RemoteComputeSelector(
            self.window, (self.QtCore, self.QtGui, self.QtWidgets)
        )
        layout.addWidget(self.remote_selector.group)
        note = QtWidgets.QLabel(
            "Local and remote runs use the same request, source-grid preparation, "
            "nnU-Net planning settings, model manifest, and status format. Remote "
            "containers are disposable; verified data and preprocessing caches remain reusable."
        )
        note.setObjectName("hint")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch(1)
        return tab

    def _load_context(self):
        default_workspace = Path.home() / ".mimics_script" / "nnunet"
        # Paths from the last successful submission are the default; the
        # context from the open Mimics project always wins when present.
        remembered = read_json(
            Path.home() / ".mimics_script" / "nnunet_settings.json", {}
        ) or {}
        self.workspace_edit.setText(
            str(
                self.context.get("workspace")
                or remembered.get("workspace")
                or default_workspace
            )
        )
        self.dataset_id.setValue(701)
        workspace = self.workspace_edit.text().strip()

        def suggest():
            try:
                self._dataset_id_results.put(
                    (suggest_dataset_id(workspace, preferred=701), "")
                )
            except Exception as exc:
                self._dataset_id_results.put((None, str(exc)))

        worker = threading.Thread(target=suggest, name="nnunet-dataset-id-check")
        worker.daemon = True
        worker.start()
        self.dataset_edit.setText(
            str(self.context.get("dataset_root") or remembered.get("dataset_root") or "")
        )
        self.mcs_edit.setText(
            str(self.context.get("mcs_dir") or remembered.get("mcs_dir") or "")
        )
        self.label_root_edit.setText(
            str(self.context.get("label_root") or remembered.get("label_root") or "")
        )
        selected = [str(value) for value in self.context.get("selected_mask_names") or []]
        if len(selected) == 1:
            self.task_edit.setText(selected[0])
        if selected:
            for index, name in enumerate(selected, start=1):
                self._add_label_row(name, name, index)
        else:
            self._add_label_row("target", "target", 1)

    def _add_label_row(self, name, aliases, label_id=None, source_mode="alternatives"):
        QtWidgets = self.QtWidgets
        row = self.label_table.rowCount()
        self.label_table.insertRow(row)
        self.label_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(name)))
        spin = QtWidgets.QSpinBox()
        spin.setRange(1, 255)
        spin.setValue(int(label_id or row + 1))
        self.label_table.setCellWidget(row, 1, spin)
        self.label_table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(aliases)))
        mode = QtWidgets.QComboBox()
        mode.addItem("Match one", "alternatives")
        mode.addItem("Union all", "union")
        index = mode.findData(str(source_mode or "alternatives"))
        mode.setCurrentIndex(max(0, index))
        mode.setToolTip(
            "Match one treats names as alternatives. Union all combines every matching source Mask into this output class."
        )
        self.label_table.setCellWidget(row, 3, mode)

    def _remove_label_rows(self):
        selected = sorted({index.row() for index in self.label_table.selectedIndexes()}, reverse=True)
        for row in selected:
            self.label_table.removeRow(row)

    def _import_label_set(self):
        QtWidgets = self.QtWidgets
        default_map = (
            ROOT
            / "integrations"
            / "nnunet_segmentation_workflow"
            / "ModelMap.toml"
        )
        choose_open_file_async(
            self.QtCore,
            QtWidgets,
            self.window,
            "Select nnU-Net label map",
            str(default_map if default_map.is_file() else ""),
            "Label maps (*.toml *.json);;All files (*)",
            self._apply_label_set,
        )

    def _apply_label_set(self, selected):
        if not selected:
            return
        QtWidgets = self.QtWidgets
        try:
            label_sets = load_label_sets(selected)
            names = sorted(label_sets)
            chosen = names[0]
            if len(names) > 1:
                chosen, accepted = QtWidgets.QInputDialog.getItem(
                    self.window,
                    "Choose Label Set",
                    "Multi-class task:",
                    names,
                    0,
                    False,
                )
                if not accepted:
                    return
            rows = label_sets[str(chosen)]
            self.label_table.setRowCount(0)
            for row in rows:
                self._add_label_row(
                    row["name"],
                    "; ".join(row["aliases"]),
                    row["id"],
                    row.get("source_mode", "alternatives"),
                )
            self.task_edit.setText(str(chosen))
            self.status_label.setStyleSheet("")
            self.status_label.setText(
                "Loaded multi-class task '{}' with {} output labels.".format(
                    chosen, len(rows)
                )
            )
        except Exception as exc:
            self.status_label.setStyleSheet("color: #b42318; font-weight: 600;")
            self.status_label.setText(str(exc))

    def _refresh_source(self):
        source = str(self.source_combo.currentData() or "dataset_masks")
        exported = source == "exported_masks"
        mcs = source == "mcs_refresh"
        self.label_root_label.setVisible(exported)
        self.label_root_widget.setVisible(exported)
        self.mcs_label.setVisible(mcs)
        self.mcs_widget.setVisible(mcs)
        hints = {
            "dataset_masks": "Reads one case folder at a time and matches files below its segmentations folder.",
            "exported_masks": "Uses source-grid Masks from a previous export. No Mimics export process is started.",
            "mcs_refresh": "Exports only the configured Mask names from saved projects, then validates them against each original image.",
        }
        self.source_hint.setText(hints[source])

    def _refresh_configuration(self):
        configuration = str(self.configuration_combo.currentData() or "3d_fullres")
        for spin in self.spacing_spins:
            spin.setEnabled(not self.auto_spacing.isChecked())
        for spin in self.patch_spins:
            spin.setEnabled(not self.auto_patch.isChecked())
        self.batch_spin.setEnabled(not self.auto_batch.isChecked())
        is_2d = configuration == "2d"
        if is_2d:
            self.auto_spacing.setChecked(True)
        self.auto_spacing.setEnabled(not is_2d)
        self.patch_labels[2].setVisible(not is_2d)
        self.patch_spins[2].setVisible(not is_2d)

    def _refresh_fold(self):
        train_all = str(self.fold_combo.currentData() or "0") == "all"
        self.validation_spin.setEnabled(not train_all)
        if train_all:
            self.validation_spin.setValue(0.0)
        self.validation_spin.setMinimum(0.0 if train_all else 0.05)
        if not train_all and self.validation_spin.value() <= 0:
            self.validation_spin.setValue(0.2)

    def _refresh_continue_training(self):
        continuing = self.continue_check.isChecked()
        self.pretrained_widget.setEnabled(not continuing)

    def _trainer_value(self):
        data = self.trainer_combo.currentData()
        text = self.trainer_combo.currentText().strip()
        if data and text in {
            "Configurable standard trainer",
            "Configurable trainer without mirroring",
            "Official standard trainer",
            "Official trainer without mirroring",
        }:
            return str(data)
        return text

    def _refresh_trainer(self):
        trainer = self._trainer_value()
        configurable = trainer in {
            "MimicsNNUNetTrainer",
            "MimicsNNUNetTrainerNoMirroring",
        }
        self.epochs_spin.setEnabled(configurable)
        if configurable:
            self.trainer_hint.setText(
                "Uses standard nnU-Net behavior with the epoch count below. The bundled trainer is portable with Mimics-Script."
            )
        else:
            self.trainer_hint.setText(
                "This trainer controls its own training length. Epochs is disabled so the interface does not promise an ignored value."
            )

    def _labels(self):
        rows = []
        for row in range(self.label_table.rowCount()):
            name_item = self.label_table.item(row, 0)
            aliases_item = self.label_table.item(row, 2)
            spin = self.label_table.cellWidget(row, 1)
            source_mode = self.label_table.cellWidget(row, 3)
            rows.append(
                {
                    "name": name_item.text().strip() if name_item else "",
                    "id": int(spin.value()),
                    "aliases": aliases_item.text().strip() if aliases_item else "",
                    "source_mode": str(source_mode.currentData()),
                }
            )
        return rows

    def _request(self):
        backend, profile_id = self.remote_selector.selection()
        trainer = self._trainer_value()
        request = {
            "operation": "train",
            "task_name": self.task_edit.text().strip(),
            "task_id": self.task_edit.text().strip(),
            "workspace": self.workspace_edit.text().strip(),
            "dataset_root": self.dataset_edit.text().strip(),
            "dataset_id": int(self.dataset_id.value()),
            "modality": str(self.modality_combo.currentData()),
            "label_source": str(self.source_combo.currentData()),
            "label_root": self.label_root_edit.text().strip(),
            "mcs_dir": self.mcs_edit.text().strip(),
            "labels": self._labels(),
            "missing_label_policy": str(self.missing_policy_combo.currentData()),
            "overlap_policy": "replace" if self.overlap_check.isChecked() else "fail",
            "configuration": str(self.configuration_combo.currentData()),
            "spacing": None if self.auto_spacing.isChecked() else [spin.value() for spin in self.spacing_spins],
            "patch_size": None if self.auto_patch.isChecked() else [
                spin.value() for spin in self.patch_spins[: 2 if self.configuration_combo.currentData() == "2d" else 3]
            ],
            "batch_size": None if self.auto_batch.isChecked() else int(self.batch_spin.value()),
            "trainer": trainer,
            "epochs": int(self.epochs_spin.value()) if trainer in {
                "MimicsNNUNetTrainer",
                "MimicsNNUNetTrainerNoMirroring",
            } else 1000,
            "fold": str(self.fold_combo.currentData()),
            "validation_fraction": float(self.validation_spin.value()),
            "preprocess_workers": int(self.workers_spin.value()),
            "num_gpus": int(self.num_gpus_spin.value()),
            "gpu_id": self.gpu_devices_edit.text().strip(),
            "pretrained_weights": "" if self.continue_check.isChecked() else self.pretrained_edit.text().strip(),
            "continue_training": bool(self.continue_check.isChecked()),
            "disable_tta": not bool(self.tta_check.isChecked()),
            "execution_backend": backend,
            "remote_profile_id": profile_id,
            "mimics_exe": self.context.get("mimics_exe") or "",
        }
        return normalize_request(request)

    def _submit(self):
        if self._submission_pending:
            return
        try:
            request = self._request()
            self._submission_generation += 1
            generation = self._submission_generation
            self._submission_pending = True
            self._submission_deadline = time.time() + 60.0
            self.start_button.setEnabled(False)
            self.status_label.setStyleSheet("")
            self.status_label.setText(
                "Checking the selected data locations in the background..."
            )

            def validate():
                try:
                    if not Path(request["dataset_root"]).is_dir():
                        raise ValueError("Original image dataset does not exist.")
                    if request["label_source"] == "exported_masks" and not Path(
                        request["label_root"]
                    ).is_dir():
                        raise ValueError("Exported Mask folder does not exist.")
                    if request["label_source"] == "mcs_refresh" and not Path(
                        request["mcs_dir"]
                    ).is_dir():
                        raise ValueError("Saved .mcs folder does not exist.")
                    self._submission_results.put((generation, request, ""))
                except Exception as exc:
                    self._submission_results.put((generation, None, str(exc)))

            worker = threading.Thread(target=validate, name="nnunet-path-check")
            worker.daemon = True
            worker.start()
        except Exception as exc:
            self._submission_pending = False
            self.start_button.setEnabled(True)
            self.status_label.setObjectName("errorLabel")
            self.status_label.setStyleSheet("color: #b42318; font-weight: 600;")
            self.status_label.setText(str(exc))

    def _poll_submission(self):
        try:
            dataset_id, dataset_error = self._dataset_id_results.get_nowait()
        except Empty:
            pass
        else:
            if dataset_id is not None and int(self.dataset_id.value()) == 701:
                self.dataset_id.setValue(int(dataset_id))
            elif dataset_error:
                self.status_label.setText(
                    "Could not inspect existing Dataset IDs; using 701. {}".format(
                        dataset_error
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
                "Path checking timed out. The selected drive may be offline or "
                "responding too slowly; paste a more specific path or retry."
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
            self.status_label.setText("Submitting nnU-Net training...")
            write_json_atomic(
                Path.home() / ".mimics_script" / "nnunet_settings.json",
                {
                    "schema_version": "mimics_nnunet_settings.v1",
                    "workspace": request["workspace"],
                    "dataset_root": request["dataset_root"],
                    "mcs_dir": request.get("mcs_dir") or "",
                    "label_root": request.get("label_root") or "",
                    "last_task_id": request["task_id"],
                    "last_task_name": request["task_name"],
                    "updated_at_epoch": time.time(),
                },
            )
            job = create_job(request)
            self.submitted = True
            setup_status = self.context.get("setup_status_path")
            if setup_status:
                update_status(
                    setup_status,
                    status="training_started",
                    phase="training_started",
                    training_job_id=job["job_id"],
                    training_status_path=job["status_path"],
                    message="nnU-Net training started outside Mimics.",
                    completed_at_epoch=time.time(),
                )
            self.window.close()
        except Exception as exc:
            self.start_button.setEnabled(True)
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
                    message="nnU-Net training setup was closed.",
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
        configure_application(app, "nnU-Net Training")
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
