#!/usr/bin/env python3
"""Small non-modal task-model chooser used only when automatic matching is ambiguous."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for candidate in (str(ROOT), str(ROOT / "tools")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from nninteractive_task_common import (  # noqa: E402
    audit_model_dir,
    find_task,
    model_profile,
    resolve_registered_model_dir,
    model_rows,
    read_json,
    selected_model,
    task_rows,
    write_json_atomic,
)
from ui_theme import configure_application  # noqa: E402


TITLE = "Choose nnInteractive Custom Model"


class Chooser:
    def __init__(self, window, context, qt):
        self.window = window
        self.context = context
        self.QtCore, self.QtGui, self.QtWidgets = qt
        self.terminal_written = False
        self.workspace = Path(context["workspace"]).resolve()
        self.status_path = Path(context["status_path"]).resolve()
        self.window.setWindowTitle(TITLE)
        self.window.resize(620, 300)
        self.window.setMinimumSize(560, 280)
        layout = self.QtWidgets.QVBoxLayout(self.window)
        layout.setContentsMargins(24, 20, 24, 18)
        layout.setSpacing(12)
        title = self.QtWidgets.QLabel(TITLE)
        title.setObjectName("title")
        layout.addWidget(title)
        subtitle = self.QtWidgets.QLabel(
            "Choose the annotation target and custom model for the selected "
            "Mask. This choice can be remembered for the current project."
        )
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        form = self.QtWidgets.QFormLayout()
        form.setLabelAlignment(self.QtCore.Qt.AlignRight)
        self.task_combo = self.QtWidgets.QComboBox()
        for task in task_rows(self.workspace):
            self.task_combo.addItem(
                str(task.get("task_name") or task.get("task_id")),
                str(task.get("task_id") or ""),
            )
        requested = str(context.get("task_id") or "")
        if requested:
            index = self.task_combo.findData(requested)
            if index >= 0:
                self.task_combo.setCurrentIndex(index)
        self.task_combo.currentIndexChanged.connect(self.refresh_models)
        form.addRow("Task", self.task_combo)
        self.model_combo = self.QtWidgets.QComboBox()
        self.model_combo.currentIndexChanged.connect(self.refresh_summary)
        form.addRow("Model version", self.model_combo)
        layout.addLayout(form)
        self.summary = self.QtWidgets.QLabel("")
        self.summary.setObjectName("hint")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.remember = self.QtWidgets.QCheckBox("Use this task for the current project")
        self.remember.setChecked(bool(context.get("project_path")))
        self.remember.setEnabled(bool(context.get("project_path")))
        layout.addWidget(self.remember)
        footer = self.QtWidgets.QHBoxLayout()
        footer.addStretch(1)
        cancel = self.QtWidgets.QPushButton("Cancel")
        cancel.clicked.connect(self.cancel)
        use = self.QtWidgets.QPushButton("Use Model")
        use.setObjectName("primary")
        use.clicked.connect(self.accept)
        footer.addWidget(cancel)
        footer.addWidget(use)
        layout.addLayout(footer)
        self.use_button = use
        self.refresh_models()

    def refresh_models(self):
        task_id = str(self.task_combo.currentData() or "")
        task = find_task(self.workspace, task_id) or {}
        recommended = str(task.get("recommended_model_id") or "")
        self.model_combo.clear()
        for model in model_rows(self.workspace, task_id):
            if not model.get("compatible", True):
                continue
            label = time.strftime(
                "%Y-%m-%d %H:%M",
                time.localtime(float(model.get("created_at_epoch") or 0)),
            )
            if model.get("model_id") == recommended:
                label += "  ·  Current"
            elif model.get("state") == "unverified":
                label += "  ·  Not verified"
            self.model_combo.addItem(label, str(model.get("model_id") or ""))
        index = self.model_combo.findData(recommended)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)
        self.refresh_summary()

    def refresh_summary(self):
        task_id = str(self.task_combo.currentData() or "")
        model_id = str(self.model_combo.currentData() or "")
        model = next(
            (row for row in model_rows(self.workspace, task_id) if row.get("model_id") == model_id),
            None,
        )
        if not model:
            self.summary.setText("This task has no usable model.")
            self.use_button.setEnabled(False)
            return
        quality = model.get("quality") or {}
        delta = quality.get("delta_auc")
        validation = (
            "{:+.1f}% validation AUC".format(float(delta) * 100.0)
            if delta is not None
            else "No automatic validation comparison"
        )
        strategy = (
            "Lightweight adaptation"
            if model.get("strategy") == "clopa_in"
            else "Stronger boundary adaptation"
        )
        self.summary.setText(
            "{} · {} training + {} validation · {}".format(
                strategy,
                model.get("train_case_count", 0),
                model.get("validation_case_count", 0),
                validation,
            )
        )
        audit = audit_model_dir(
            resolve_registered_model_dir(self.workspace, model, task),
            include_checksum=False,
        )
        self.use_button.setEnabled(bool(audit.get("compatible")))
        if not audit.get("compatible"):
            self.summary.setText(
                "This model is incomplete: {}".format(", ".join(audit.get("missing") or []))
            )

    def accept(self):
        task_id = str(self.task_combo.currentData() or "")
        model_id = str(self.model_combo.currentData() or "")
        task = find_task(self.workspace, task_id)
        model = next(
            (row for row in model_rows(self.workspace, task_id) if row.get("model_id") == model_id),
            None,
        )
        if not task or not model:
            return
        try:
            # The checkpoint is hashed once when it enters the registry.
            # Re-reading a large checkpoint here would freeze the chooser.
            profile = model_profile(
                task,
                model,
                workspace=self.workspace,
                verify_checksum=False,
            )
        except Exception as exc:
            self.summary.setText(str(exc))
            return
        write_json_atomic(
            self.status_path,
            {
                "schema_version": "nninteractive_task_model_choice.v1",
                "status": "selected",
                "profile": profile,
                "remember_project": bool(self.remember.isChecked()),
                "project_path": str(self.context.get("project_path") or ""),
                "updated_at_epoch": time.time(),
            },
        )
        self.terminal_written = True
        self.window.close()

    def write_cancelled(self):
        if self.terminal_written:
            return
        write_json_atomic(
            self.status_path,
            {
                "schema_version": "nninteractive_task_model_choice.v1",
                "status": "cancelled",
                "updated_at_epoch": time.time(),
            },
        )
        self.terminal_written = True

    def cancel(self):
        self.write_cancelled()
        self.window.close()


def run_ui(context_path):
    from PySide6 import QtCore, QtGui, QtWidgets

    context = read_json(context_path, {}) or {}
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    configure_application(app, TITLE)

    class ChooserWindow(QtWidgets.QWidget):
        controller = None

        def closeEvent(self, event):
            if self.controller is not None:
                self.controller.write_cancelled()
            super().closeEvent(event)

    window = ChooserWindow()
    controller = Chooser(window, context, (QtCore, QtGui, QtWidgets))
    window.controller = controller
    window.show()
    window.raise_()
    window.activateWindow()
    return int(app.exec())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", required=True)
    args = parser.parse_args(argv)
    return run_ui(args.context)


if __name__ == "__main__":
    raise SystemExit(main())
