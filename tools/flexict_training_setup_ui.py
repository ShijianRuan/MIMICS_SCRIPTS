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
        self.window.setWindowTitle("FlexiCT Training")
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
        title = QtWidgets.QLabel("FlexiCT Training")
        title.setObjectName("title")
        subtitle = QtWidgets.QLabel(
            "Few-shot training on a handful of annotated cases with the FlexiCT "
            "ViT backbone. Two to ten cases are enough. Mimics remains usable."
        )
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(subtitle)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._build_data_tab(), "Data")
        self.tabs.addTab(self._build_training_tab(), "Training")
        self.tabs.addTab(self._build_models_tab(), "Existing models")
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
        button = QtWidgets.QPushButton("Browse...")

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
        self.label_edit.setPlaceholderText("Example: kidney_left")
        self.label_edit.setToolTip(
            "One FlexiCT model trains one binary target (organ or structure)."
        )
        form.addWidget(QtWidgets.QLabel("Target / label name *"), 0, 0)
        form.addWidget(self.label_edit, 0, 1)
        self.modality_combo = QtWidgets.QComboBox()
        for label, value in (("CT", "CT"), ("MRI", "MRI"), ("Other", "Other")):
            self.modality_combo.addItem(label, value)
        form.addWidget(QtWidgets.QLabel("Image modality"), 1, 0)
        form.addWidget(self.modality_combo, 1, 1)
        self.dataset_edit = QtWidgets.QLineEdit()
        self.dataset_edit.setPlaceholderText(
            "Folder containing one subfolder per case"
        )
        form.addWidget(QtWidgets.QLabel("Original image dataset *"), 2, 0)
        form.addWidget(
            self._path_row(self.dataset_edit, "Select original image dataset"), 2, 1
        )
        self.dataset_edit.textChanged.connect(self._refresh_cases)
        layout.addWidget(project)

        cases, cases_layout = self._surface("Cases and validation")
        hint = QtWidgets.QLabel(
            "Every case folder needs the original image and one Mask file named "
            "after the target (for example kidney_left.nii.gz). Training uses "
            "the rest; validation cases are drawn evenly across slice positions."
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        cases_layout.addWidget(hint, 0, 0, 1, 2)
        self.case_table = QtWidgets.QTableWidget(0, 4)
        self.case_table.setHorizontalHeaderLabels(
            ["Case", "Use for training", "Use for validation", "Status"]
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
        self.rescan_button = QtWidgets.QPushButton("Rescan cases")
        self.rescan_button.clicked.connect(self._rescan_cases_async)
        cases_layout.addWidget(self.rescan_button, 2, 0)
        self.case_summary = QtWidgets.QLabel("")
        self.case_summary.setObjectName("hint")
        self.case_summary.setWordWrap(True)
        cases_layout.addWidget(self.case_summary, 2, 1)
        layout.addWidget(cases)

        storage, storage_form = self._surface("Storage")
        self.workspace_edit = QtWidgets.QLineEdit()
        storage_form.addWidget(QtWidgets.QLabel("Model library"), 0, 0)
        storage_form.addWidget(
            self._path_row(self.workspace_edit, "Select FlexiCT model library"), 0, 1
        )
        storage_hint = QtWidgets.QLabel(
            "Contains managed jobs, preprocessing caches, and the model registry."
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

        training, form = self._surface("Training")
        self.configuration_combo = QtWidgets.QComboBox()
        for label, value in (
            ("Automatic (3D pair on large GPUs, 2D otherwise)", "auto"),
            ("2D slices", "2d"),
            ("3D full resolution", "3d_fullres"),
            ("2D + 3D pair (for active learning)", "pair"),
        ):
            self.configuration_combo.addItem(label, value)
        self.configuration_combo.setToolTip(
            "Few-shot validated recipes. Auto picks the pair on GPUs with 16GB+ "
            "VRAM, otherwise 2D."
        )
        form.addWidget(QtWidgets.QLabel("Configuration"), 0, 0)
        form.addWidget(self.configuration_combo, 0, 1)
        self.epochs_spin = QtWidgets.QSpinBox()
        self.epochs_spin.setRange(10, 2000)
        self.epochs_spin.setValue(150)
        self.epochs_spin.setToolTip(
            "Validated few-shot default: 150 epochs with the FlexiCT optimizer "
            "schedule."
        )
        form.addWidget(QtWidgets.QLabel("Epochs"), 1, 0)
        form.addWidget(self.epochs_spin, 1, 1)
        self.val_spin = QtWidgets.QSpinBox()
        self.val_spin.setRange(1, 10)
        self.val_spin.setValue(1)
        self.val_spin.setToolTip(
            "How many of the selected cases are held out for validation "
            "(best-checkpoint selection)."
        )
        form.addWidget(QtWidgets.QLabel("Validation cases"), 2, 0)
        form.addWidget(self.val_spin, 2, 1)
        self.mirror_combo = QtWidgets.QComboBox()
        self.mirror_combo.addItem("Both sides (default)", "")
        self.mirror_combo.addItem("Single-sided organ, disable X mirroring", "1")
        self.mirror_combo.setToolTip(
            "Disables left-right mirroring augmentation — needed for single "
            "organs like one kidney."
        )
        form.addWidget(QtWidgets.QLabel("Mirroring"), 3, 0)
        form.addWidget(self.mirror_combo, 3, 1)
        self.workers_spin = QtWidgets.QSpinBox()
        self.workers_spin.setRange(1, 64)
        self.workers_spin.setValue(4)
        form.addWidget(QtWidgets.QLabel("Preprocessing workers"), 4, 0)
        form.addWidget(self.workers_spin, 4, 1)
        layout.addWidget(training)

        recipe, recipe_form = self._surface("Locked recipe (read-only)")
        recipe_text = QtWidgets.QLabel(
            "FlexiCT ViT backbone (fp32, RoPE-safe) · AdamW dual groups "
            "(backbone 3e-5 / decoder 3e-4, betas 0.9/0.98, wd 5e-2) · poly LR "
            "· gradclip 12 · no deep supervision · foreground oversample 0.33 · "
            "batch size from nnU-Net plans · best checkpoint by validation Dice · "
            "inference without TTA.\n\nThese settings are the validated few-shot "
            "recipe and are not configurable from Mimics."
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
            ["Model", "Target", "Configuration", "Created", "Ready"]
        )
        self.models_table.horizontalHeader().setStretchLastSection(True)
        self.models_table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.Stretch
        )
        self.models_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.models_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.models_table)
        hint = QtWidgets.QLabel(
            "Registered FlexiCT models. The most recent usable model is "
            "pre-selected for prediction and active learning."
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch(1)
        return tab

    def _load_context(self):
        config = load_config()
        default_workspace = workspace_root(config)
        workspace = str(self.context.get("workspace") or default_workspace)
        self.workspace_edit.setText(workspace)
        self.dataset_edit.setText(str(self.context.get("dataset_root") or ""))
        selected = [
            str(value) for value in self.context.get("selected_mask_names") or []
        ]
        if len(selected) == 1:
            self.label_edit.setText(selected[0])
        self._refresh_models()

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
            self.status_label.setText("Submitting FlexiCT training...")
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
                "Checking the selected data locations in the background..."
            )

            def validate():
                try:
                    if not Path(request["dataset_root"]).is_dir():
                        raise ValueError("Original image dataset does not exist.")
                    if not request["cases"]:
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
        configure_application(app, "FlexiCT Training")
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
