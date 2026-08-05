#!/usr/bin/env python3
"""External model selection for nnU-Net prediction of the active Mimics case."""

from __future__ import annotations

import argparse
import os
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
    load_models,
    model_usability,
    read_json,
    update_status,
    write_json_atomic,
)
from nnunet_jobs import create_job  # noqa: E402
from remote_compute_ui import RemoteComputeSelector  # noqa: E402
from ui_theme import configure_application, stylesheet  # noqa: E402


def _matches_context(model, context):
    target = str(context.get("selected_mask_name") or "").strip().lower()
    if not target:
        return False
    for row in model.get("labels") or []:
        values = [row.get("name")] + list(row.get("aliases") or [])
        if target in {str(value or "").strip().lower() for value in values}:
            return True
    return False


def _rank_models(models, context):
    return sorted(
        models,
        key=lambda row: (
            0 if _matches_context(row, context) else 1,
            -float(row.get("created_at_epoch") or 0),
        ),
    )


class PredictionWindow:
    def __init__(self, context, qt_modules):
        self.context = context
        self.QtCore, self.QtGui, self.QtWidgets = qt_modules
        QtWidgets = self.QtWidgets
        self.window = QtWidgets.QDialog()
        self.window.setWindowTitle("nnU-Net Prediction")
        self.window.resize(760, 570)
        self.window.setMinimumSize(680, 520)
        self.window.closeEvent = self._close_event
        self.submitted = False
        self._loading_models = True
        self._model_deadline = time.time() + 120.0
        self._model_results = Queue()
        self._submission_results = Queue()
        self._submission_generation = 0
        self._submission_deadline = 0.0
        root = QtWidgets.QVBoxLayout(self.window)
        root.setContentsMargins(22, 18, 22, 18)
        root.setSpacing(12)
        title = QtWidgets.QLabel("nnU-Net Prediction")
        title.setObjectName("title")
        subtitle = QtWidgets.QLabel(
            "Choose a compatible trained model. Inference runs outside Mimics; "
            "the result is applied only after its physical grid is verified."
        )
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(subtitle)
        case = QtWidgets.QFrame()
        case.setObjectName("surface")
        case_layout = QtWidgets.QGridLayout(case)
        case_layout.setContentsMargins(16, 14, 16, 14)
        case_layout.addWidget(QtWidgets.QLabel("Current case"), 0, 0)
        case_layout.addWidget(QtWidgets.QLabel(str(context.get("case_id") or "-")), 0, 1)
        case_layout.addWidget(QtWidgets.QLabel("Selected Mask hint"), 1, 0)
        case_layout.addWidget(QtWidgets.QLabel(str(context.get("selected_mask_name") or "-")), 1, 1)
        root.addWidget(case)

        self.models = []
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Task", "Model", "Configuration", "Labels", "Backend"]
        )
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        root.addWidget(self.table, 1)
        self.tta_check = QtWidgets.QCheckBox("Use mirroring test-time augmentation")
        self.tta_check.setChecked(False)
        self.tta_check.setToolTip("May improve accuracy but increases inference time.")
        root.addWidget(self.tta_check)
        self.remote_selector = RemoteComputeSelector(
            self.window, (self.QtCore, self.QtGui, self.QtWidgets)
        )
        root.addWidget(self.remote_selector.group)
        self.status_label = QtWidgets.QLabel(
            "Loading compatible models in the background..."
        )
        self.status_label.setObjectName("hint")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)
        actions = QtWidgets.QHBoxLayout()
        actions.addStretch(1)
        cancel = QtWidgets.QPushButton("Cancel")
        cancel.clicked.connect(self.window.close)
        actions.addWidget(cancel)
        self.start = QtWidgets.QPushButton("Start Prediction")
        self.start.setObjectName("primary")
        self.start.clicked.connect(self._submit)
        self.start.setEnabled(False)
        actions.addWidget(self.start)
        root.addLayout(actions)
        self._timer = self.QtCore.QTimer(self.window)
        self._timer.timeout.connect(self._poll_background_results)
        self._timer.start(80)
        self._start_model_loading()

    def _start_model_loading(self):
        workspace = self.context.get("workspace") or Path.home() / ".mimics_script" / "nnunet"

        def load():
            try:
                all_models = load_models(workspace)
                usable = [row for row in all_models if model_usability(row)[0]]
                self._model_results.put((_rank_models(usable, self.context), ""))
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
            for index, model in enumerate(self.models):
                values = [
                    model.get("task_name") or model.get("task_id"),
                    model.get("model_id"),
                    model.get("configuration"),
                    ", ".join(
                        str(row.get("name")) for row in model.get("labels") or []
                    ),
                    model.get("execution_backend") or "local",
                ]
                for column, value in enumerate(values):
                    item = self.QtWidgets.QTableWidgetItem(str(value or ""))
                    if column == 3:
                        item.setToolTip(str(value or ""))
                    self.table.setItem(index, column, item)
            if self.models:
                self.table.selectRow(0)
        finally:
            self.table.setUpdatesEnabled(True)
        self.start.setEnabled(bool(self.models))
        if error:
            self.status_label.setStyleSheet("color: #b42318; font-weight: 600;")
            self.status_label.setText("Could not load nnU-Net models: {}".format(error))
        elif self.models:
            self.status_label.setText(
                "A selected Mask only ranks models containing that label. The chosen model predicts all of its output labels; its plans control normalization, spacing, and patching."
            )
        else:
            self.status_label.setText(
                "No usable nnU-Net model was found. Open Status and Models to inspect training results."
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
                    "Model discovery timed out. The model library may be on an "
                    "offline or slow drive. Close this window, reconnect the drive, "
                    "and retry."
                )
        if (
            self._submission_deadline
            and time.time() >= self._submission_deadline
        ):
            self._submission_generation += 1
            self._submission_deadline = 0.0
            self.start.setEnabled(bool(self.models))
            self.status_label.setStyleSheet("color: #b42318; font-weight: 600;")
            self.status_label.setText(
                "Image path checking timed out. The source drive may be offline; "
                "relink the source image or reconnect the drive before retrying."
            )
        try:
            generation, request, error = self._submission_results.get_nowait()
        except Empty:
            return
        if generation != self._submission_generation:
            return
        self._submission_deadline = 0.0
        if error:
            self.start.setEnabled(bool(self.models))
            self.status_label.setStyleSheet("color: #b42318; font-weight: 600;")
            self.status_label.setText(error)
            return
        self._create_prediction_job(request)

    def _submit(self):
        try:
            row = self.table.currentRow()
            if not 0 <= row < len(self.models):
                raise RuntimeError("Select one model.")
            model = self.models[row]
            image_path = Path(
                os.path.abspath(
                    os.path.expandvars(
                        os.path.expanduser(
                            str(self.context.get("source_image_path") or "")
                        )
                    )
                )
            )
            workspace = str(self.context.get("workspace") or Path.home() / ".mimics_script" / "nnunet")
            backend, profile_id = self.remote_selector.selection()
            request = {
                "operation": "infer",
                "workspace": workspace,
                "task_id": str(model.get("task_id") or ""),
                "task_name": str(model.get("task_name") or model.get("task_id") or ""),
                "dataset_id": int(model.get("dataset_id") or 0),
                "image_path": str(image_path),
                "model_manifest": str(model["manifest_path"]),
                "output_path": "",
                "disable_tta": not self.tta_check.isChecked(),
                "execution_backend": backend,
                "remote_profile_id": profile_id,
                "case_id": str(self.context.get("case_id") or image_path.stem),
                "ts_root": self.context.get("ts_root") or "",
                "prediction_target": self.context.get("prediction_target") or {},
                "matching_masks": self.context.get("matching_masks") or [],
                "source_geometry_expected": self.context.get("source_geometry_expected") or {},
                "source_modality": str(
                    (self.context.get("source_geometry_expected") or {}).get(
                        "source_modality"
                    )
                    or ""
                ),
                "target_grid": self.context.get("target_grid") or {},
                "launch_project_path": self.context.get("launch_project_path") or "",
            }
            self._submission_generation += 1
            generation = self._submission_generation
            self._submission_deadline = time.time() + 60.0
            self.start.setEnabled(False)
            self.status_label.setStyleSheet("")
            self.status_label.setText(
                "Checking the current image location in the background..."
            )

            def validate():
                try:
                    if not image_path.exists():
                        raise RuntimeError(
                            "The current source image is unavailable: {}".format(
                                image_path
                            )
                        )
                    self._submission_results.put((generation, request, ""))
                except Exception as exc:
                    self._submission_results.put((generation, None, str(exc)))

            worker = threading.Thread(target=validate, name="nnunet-image-check")
            worker.daemon = True
            worker.start()
        except Exception as exc:
            self.start.setEnabled(bool(self.models))
            self.status_label.setStyleSheet("color: #b42318; font-weight: 600;")
            self.status_label.setText(str(exc))

    def _create_prediction_job(self, request):
        try:
            workspace = str(request.get("workspace") or "")
            model = next(
                (
                    row
                    for row in self.models
                    if str(row.get("manifest_path") or "")
                    == str(request.get("model_manifest") or "")
                ),
                {},
            )
            provisional = create_job(request)
            settings_path = Path.home() / ".mimics_script" / "nnunet_settings.json"
            settings = read_json(settings_path, {}) or {}
            settings.update(
                {
                    "schema_version": "mimics_nnunet_settings.v1",
                    "workspace": workspace,
                    "last_task_id": request["task_id"],
                    "last_task_name": request["task_name"],
                    "updated_at_epoch": time.time(),
                }
            )
            write_json_atomic(settings_path, settings)
            self.submitted = True
            setup_status = self.context.get("setup_status_path")
            if setup_status:
                update_status(
                    setup_status,
                    status="prediction_started",
                    phase="prediction_started",
                    prediction_job_id=provisional["job_id"],
                    prediction_status_path=provisional["status_path"],
                    model=model,
                    completed_at_epoch=time.time(),
                )
            self.window.accept()
        except Exception as exc:
            self.start.setEnabled(bool(self.models))
            self.status_label.setStyleSheet("color: #b42318; font-weight: 600;")
            self.status_label.setText(str(exc))

    def _close_event(self, event):
        self._submission_generation += 1
        self._submission_deadline = 0.0
        if not self.submitted and self.context.get("setup_status_path"):
            update_status(
                self.context["setup_status_path"],
                status="cancelled",
                phase="cancelled",
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
    context = read_json(Path(args.context).expanduser().resolve(), {}) or {}
    try:
        from PySide6 import QtCore, QtGui, QtWidgets

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        configure_application(app, "nnU-Net Prediction")
        app.setStyleSheet(stylesheet())
        window = PredictionWindow(context, (QtCore, QtGui, QtWidgets))
        window.show()
        return int(app.exec())
    except Exception as exc:
        if context.get("setup_status_path"):
            update_status(
                context["setup_status_path"],
                status="failed",
                phase="ui_failed",
                error="{}: {}".format(type(exc).__name__, exc),
                traceback=traceback.format_exc(),
            )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
