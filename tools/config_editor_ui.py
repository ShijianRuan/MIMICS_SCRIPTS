#!/usr/bin/env python3
"""External editor for the user-facing JSON configuration files.

Annotators previously had to hand-edit configs like
``nninteractive_config.json`` or ``flexict_config.json`` in a text editor.
This window exposes the commonly tuned keys of each config with type
validation, per-key help text, atomic saves, and a one-step rollback
backup. Keys not declared in the schema below are preserved untouched.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui_theme import configure_application, stylesheet  # noqa: E402

BACKUP_DIR = ROOT / ".mimics_runtime" / "config_backups"

# field: (label, type, help). type: bool | int | float | str | choice(list)
# Only annotator-tunable keys are declared; everything else rides along
# unedited.
CONFIG_SCHEMAS: dict[str, dict[str, tuple]] = {
    "mimics_io_config.json": {
        "mimics_output_dir": (
            "Default .mcs output folder", "str",
            "Empty = <dataset>/mcs_output. Absolute or relative to the "
            "project root. Imports still ask when the folder is busy.",
        ),
        "mimics_background_exe": (
            "Background Mimics executable", "str",
            "Path to materialise.exe used for .mcs creation. Empty = "
            "auto-detect from the running Mimics.",
        ),
    },
    "nninteractive_config.json": {
        "device": (
            "Inference device", "choice",
            "auto = pick the first free GPU; cpu forces CPU. Use cpu when "
            "the GPU is needed by a training job.",
        ),
        "auto_start_server": (
            "Auto-start the model server", "bool",
            "Start the nnInteractive server automatically when an "
            "annotation session begins.",
        ),
        "existing_mask_result_mode": (
            "Existing Mask result mode", "choice",
            "ask = always ask what to do with the selected Mask; replace = "
            "update it in place; copy = always create an editable copy.",
        ),
        "server_idle_timeout_seconds": (
            "Server idle timeout (s)", "int",
            "The model server shuts down after this many idle seconds.",
        ),
        "keep_server_warm_after_session": (
            "Keep server warm after session", "bool",
            "Keep the server alive between annotation sessions so the next "
            "session starts faster.",
        ),
        "minimum_free_gpu_memory_gb": (
            "Minimum free GPU memory (GB)", "int",
            "Annotation sessions refuse to start when the GPU has less "
            "free memory than this.",
        ),
    },
    "nninteractive_finetune_config.json": {
        "default_epochs": (
            "Default training epochs", "int",
            "Epochs for a fine-tuning run (allowed range comes from "
            "minimum_epochs/maximum_epochs).",
        ),
        "default_validation_fraction": (
            "Validation fraction", "float",
            "Share of cases held out for validation during training.",
        ),
        "job_retention_days": (
            "Job retention (days)", "int",
            "Finished fine-tuning jobs older than this are swept.",
        ),
        "keep_failed_training_artifacts": (
            "Keep failed training artifacts", "bool",
            "Keep the training work folder after a failed run for "
            "diagnosis instead of cleaning it up.",
        ),
    },
    "flexict_config.json": {
        "default_configuration": (
            "Default configuration", "choice",
            "2d, 3d_fullres, pair (2D+3D for active learning), or auto "
            "(2d on small GPUs, pair otherwise).",
        ),
        "default_epochs": (
            "Default training epochs", "int",
            "Epochs for FlexiCT fine-tuning. 150 is the validated recipe.",
        ),
        "default_val_cases": (
            "Validation cases", "int",
            "Cases held out for validation when a split is needed.",
        ),
        "job_retention_days": (
            "Job retention (days)", "int",
            "Finished FlexiCT jobs older than this are swept.",
        ),
        "pretrained_weights_dir": (
            "Pretrained weights folder", "str",
            "Optional override for the FlexiCT backbone weights folder. "
            "Empty = use the bundled integrations/flexict-finetune/weights.",
        ),
    },
    "interactive_algorithms_config.json": {
        "scribbleprompt.timeout_seconds": (
            "ScribblePrompt timeout (s)", "int",
            "Give up waiting for a ScribblePrompt prediction after this "
            "many seconds.",
        ),
        "scribbleprompt.device": (
            "ScribblePrompt device", "str",
            "cpu or cuda. CPU is slower but never competes with training.",
        ),
    },
}

CHOICE_OPTIONS = {
    ("nninteractive_config.json", "device"): ["auto", "cpu"],
    ("nninteractive_config.json", "existing_mask_result_mode"): [
        "ask", "replace", "copy",
    ],
    ("flexict_config.json", "default_configuration"): [
        "auto", "2d", "3d_fullres", "pair",
    ],
}


def _get_nested(data: dict, dotted: str):
    node = data
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _set_nested(data: dict, dotted: str, value) -> None:
    parts = dotted.split(".")
    node = data
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


class ConfigEditor:
    def __init__(self, qt_modules):
        self.QtCore, self.QtGui, self.QtWidgets = qt_modules
        QtWidgets = self.QtWidgets
        self.values: dict[str, dict] = {}
        self.widgets: dict[str, dict[str, object]] = {}

        self.window = QtWidgets.QWidget()
        self.window.setWindowTitle("Configuration Editor")
        self.window.resize(760, 640)
        root = QtWidgets.QVBoxLayout(self.window)
        root.setContentsMargins(22, 18, 22, 18)
        root.setSpacing(12)

        title = QtWidgets.QLabel("Configuration Editor")
        title.setObjectName("title")
        root.addWidget(title)
        hint = QtWidgets.QLabel(
            "Only commonly tuned keys are shown; everything else in each "
            "file is preserved. Changes take effect the next time the "
            "related window or service starts."
        )
        hint.setObjectName("subtitle")
        hint.setWordWrap(True)
        root.addWidget(hint)

        tabs = QtWidgets.QTabWidget()
        root.addWidget(tabs, 1)
        for file_name, schema in CONFIG_SCHEMAS.items():
            path = ROOT / file_name
            data = self._load(path)
            self.values[file_name] = data
            tab = QtWidgets.QWidget()
            form = QtWidgets.QFormLayout(tab)
            form.setContentsMargins(16, 14, 16, 14)
            form.setSpacing(10)
            self.widgets[file_name] = {}
            for key, (label, ftype, help_text) in schema.items():
                widget = self._make_widget(
                    file_name, key, ftype, _get_nested(data, key)
                )
                form.addRow(label, widget)
                if help_text:
                    note = QtWidgets.QLabel(help_text)
                    note.setObjectName("hint")
                    note.setWordWrap(True)
                    form.addRow("", note)
            tabs.addTab(tab, file_name.replace("_config.json", "").replace(".json", ""))

        buttons = QtWidgets.QHBoxLayout()
        self.rollback = QtWidgets.QPushButton("Roll back last save")
        self.rollback.clicked.connect(self._rollback)
        self.rollback.setEnabled((BACKUP_DIR / "latest.json").is_file())
        buttons.addStretch(1)
        buttons.addWidget(self.rollback)
        save = QtWidgets.QPushButton("Save")
        save.setObjectName("primary")
        save.clicked.connect(self._save)
        buttons.addWidget(save)
        root.addLayout(buttons)

        self.status = QtWidgets.QLabel("")
        self.status.setObjectName("hint")
        root.addWidget(self.status)

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _load(path: Path) -> dict:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return {}

    def _make_widget(self, file_name, key, ftype, current):
        QtWidgets = self.QtWidgets
        options = CHOICE_OPTIONS.get((file_name, key))
        if options:
            widget = QtWidgets.QComboBox()
            widget.addItems(options)
            if current in options:
                widget.setCurrentText(str(current))
        elif ftype == "bool":
            widget = QtWidgets.QCheckBox()
            widget.setChecked(bool(current))
        else:
            widget = QtWidgets.QLineEdit(str(current if current is not None else ""))
        self.widgets[file_name][key] = (widget, ftype)
        return widget

    def _collect(self, file_name):
        """Return {key: typed_value}; None value marks a validation error."""
        result = {}
        for key, (widget, ftype) in self.widgets[file_name].items():
            if isinstance(widget, self.QtWidgets.QComboBox):
                result[key] = widget.currentText()
            elif isinstance(widget, self.QtWidgets.QCheckBox):
                result[key] = bool(widget.isChecked())
            else:
                text = widget.text().strip()
                if ftype == "int":
                    try:
                        result[key] = int(text)
                    except ValueError:
                        return key, None
                elif ftype == "float":
                    try:
                        result[key] = float(text)
                    except ValueError:
                        return key, None
                else:
                    result[key] = text
        return result

    def _save(self):
        updates = {}
        for file_name in self.widgets:
            collected = self._collect(file_name)
            if isinstance(collected, tuple):
                bad_key, _ = collected
                self.status.setText(
                    "{0}: '{1}' must be a number.".format(file_name, bad_key)
                )
                return
            updates[file_name] = collected
        try:
            BACKUP_DIR.mkdir(parents=True, exist_ok=True)
            backup = {}
            for file_name in self.widgets:
                backup[file_name] = self._load(ROOT / file_name)
            (BACKUP_DIR / "latest.json").write_text(
                json.dumps(backup, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            for file_name, keyed in updates.items():
                data = self.values[file_name]
                for key, value in keyed.items():
                    _set_nested(data, key, value)
                path = ROOT / file_name
                tmp = path.with_suffix(".json.tmp")
                tmp.write_text(
                    json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                tmp.replace(path)
        except OSError as exc:
            self.status.setText("Could not save: {0}".format(exc))
            return
        self.rollback.setEnabled(True)
        self.status.setText(
            "Saved {0} config file(s). Restart related windows/services to "
            "apply.".format(len(updates))
        )

    def _rollback(self):
        backup_path = BACKUP_DIR / "latest.json"
        if not backup_path.is_file():
            return
        try:
            backup = json.loads(backup_path.read_text(encoding="utf-8"))
            for file_name, data in backup.items():
                path = ROOT / file_name
                tmp = path.with_suffix(".json.tmp")
                tmp.write_text(
                    json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                tmp.replace(path)
        except (OSError, ValueError) as exc:
            self.status.setText("Roll back failed: {0}".format(exc))
            return
        # Reload the whole window state from disk.
        self.values = {
            name: self._load(ROOT / name) for name in self.widgets
        }
        self._rebuild_widgets()
        self.status.setText("Rolled back to the previous saved values.")

    def _rebuild_widgets(self):
        QtWidgets = self.QtWidgets
        # Refresh every widget in place from self.values.
        for file_name, keyed in self.widgets.items():
            data = self.values[file_name]
            schema = CONFIG_SCHEMAS[file_name]
            for key, (widget, ftype) in keyed.items():
                current = _get_nested(data, key)
                if isinstance(widget, QtWidgets.QComboBox):
                    if str(current) in [widget.itemText(i) for i in range(widget.count())]:
                        widget.setCurrentText(str(current))
                elif isinstance(widget, QtWidgets.QCheckBox):
                    widget.setChecked(bool(current))
                else:
                    widget.setText(str(current if current is not None else ""))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    from PySide6 import QtCore, QtGui, QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    configure_application(app, "Configuration Editor")
    app.setStyleSheet(stylesheet())
    editor = ConfigEditor((QtCore, QtGui, QtWidgets))
    editor.window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
