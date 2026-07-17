#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""External PySide6 model chooser for DINOv3 few-shot inference."""

from __future__ import print_function

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

try:
    from ui_theme import configure_application, stylesheet as shared_stylesheet
except ImportError:
    from tools.ui_theme import configure_application, stylesheet as shared_stylesheet


TITLE = "Select DINOv3 Model"


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json_atomic(path, payload, retries=20, max_sleep=0.25):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    last_error = None
    for attempt in range(max(1, int(retries))):
        tmp = path.with_name(path.name + "." + str(os.getpid()) + "." + uuid.uuid4().hex + ".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as handle:
                handle.write(text)
                try:
                    handle.flush()
                    os.fsync(handle.fileno())
                except Exception:
                    pass
            os.replace(str(tmp), str(path))
            return
        except OSError as exc:
            last_error = exc
            try:
                if tmp.is_file():
                    tmp.unlink()
            except Exception:
                pass
            time.sleep(min(float(max_sleep), 0.05 * (attempt + 1)))
    if last_error is not None:
        raise last_error


def update_status(context, payload):
    status_path = context.get("status_path")
    if not status_path:
        return
    current = read_json(status_path, {}) or {}
    current.update(payload)
    current["updated_at_epoch"] = time.time()
    write_json_atomic(status_path, current)


def format_time(epoch):
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(float(epoch)))
    except Exception:
        return "unknown time"


def best_dice_text(candidate):
    best = (candidate.get("manifest") or {}).get("best_dsc")
    if best is None:
        return "n/a"
    try:
        return "{0:.4f}".format(float(best))
    except Exception:
        return str(best)


class ModelChooser(object):
    def __init__(self, window, context, qt):
        self.window = window
        self.context = context
        self.QtCore, self.QtGui, self.QtWidgets = qt
        self.candidates = list(context.get("candidates") or [])
        self.selected = False
        self.list_widget = None
        self.detail = None
        self._build()

    def _build(self):
        QtWidgets = self.QtWidgets
        self.window.setWindowTitle(TITLE)
        self.window.resize(920, 560)
        self.window.setMinimumSize(760, 460)
        self.window.setStyleSheet(self._stylesheet())

        central = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(10)

        title = QtWidgets.QLabel("Select DINOv3 Model")
        title.setObjectName("titleLabel")
        subtitle = QtWidgets.QLabel(
            "Organ: {0}    Case: {1}".format(
                self.context.get("organ", "?"),
                self.context.get("case_id", "?"),
            )
        )
        subtitle.setObjectName("subtitleLabel")
        outer.addWidget(title)
        outer.addWidget(subtitle)

        splitter = QtWidgets.QSplitter(self.QtCore.Qt.Horizontal)
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setWordWrap(True)
        self.list_widget.setHorizontalScrollBarPolicy(self.QtCore.Qt.ScrollBarAlwaysOff)
        self.list_widget.currentRowChanged.connect(self._show_current)
        for candidate in self.candidates:
            self.list_widget.addItem(self._row_label(candidate))
        splitter.addWidget(self.list_widget)

        self.detail = QtWidgets.QTextEdit()
        self.detail.setReadOnly(True)
        splitter.addWidget(self.detail)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        outer.addWidget(splitter, 1)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        cancel = QtWidgets.QPushButton("Cancel")
        cancel.clicked.connect(self.cancel)
        buttons.addWidget(cancel)
        use = QtWidgets.QPushButton("Use Selected Model")
        use.setObjectName("primaryButton")
        use.clicked.connect(self.accept)
        buttons.addWidget(use)
        outer.addLayout(buttons)
        self.window.setCentralWidget(central)
        if self.candidates:
            self.list_widget.setCurrentRow(0)

    def _stylesheet(self):
        return shared_stylesheet()

    def _row_label(self, candidate):
        return "{scope}  |  {samples} samples  |  Dice {dice}\n{created}".format(
            scope=candidate.get("scope", "?"),
            samples=candidate.get("sample_count", "?"),
            dice=best_dice_text(candidate),
            created=format_time(candidate.get("created_at_epoch", 0.0)),
        )

    def _show_current(self, row):
        if row < 0 or row >= len(self.candidates):
            self.detail.setPlainText("")
            return
        candidate = self.candidates[row]
        manifest = candidate.get("manifest") or {}
        lines = [
            "Model: {0}".format(candidate.get("model_id", "?")),
            "Available in: {0}".format(candidate.get("scope", "?")),
            "Trained for: {0}".format(candidate.get("organ", "?")),
            "Training samples: {0}".format(candidate.get("sample_count", "?")),
            "Best validation Dice: {0}".format(best_dice_text(candidate)),
            "Trained: {0}".format(format_time(candidate.get("created_at_epoch", 0.0))),
            "",
            "Model file:",
            str(manifest.get("checkpoint", "")),
            "",
            "Training configuration:",
            str(manifest.get("config", "")),
        ]
        self.detail.setPlainText("\n".join(lines))

    def accept(self):
        row = self.list_widget.currentRow()
        if row < 0 or row >= len(self.candidates):
            return
        candidate = self.candidates[row]
        self.selected = True
        update_status(self.context, {
            "status": "selected",
            "selected_model": candidate,
            "model_id": candidate.get("model_id", ""),
        })
        self.window.close()

    def cancel(self):
        self.selected = True
        update_status(self.context, {"status": "cancelled"})
        self.window.close()

    def closeEvent(self, event):
        if not self.selected:
            update_status(self.context, {"status": "cancelled"})
        event.accept()


def run(context):
    from PySide6 import QtCore, QtGui, QtWidgets

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])
    configure_application(app, TITLE)

    class CloseAwareMainWindow(QtWidgets.QMainWindow):
        def closeEvent(self, event):
            controller = getattr(self, "controller", None)
            if controller is not None:
                controller.closeEvent(event)
            else:
                event.accept()

    window = CloseAwareMainWindow()
    controller = ModelChooser(window, context, (QtCore, QtGui, QtWidgets))
    window.controller = controller
    update_status(context, {"status": "selecting_model", "controller_pid": os.getpid()})
    window.show()
    return app.exec()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", required=True)
    args = parser.parse_args(argv)
    context = read_json(args.context, {}) or {}
    try:
        return run(context)
    except Exception as exc:
        update_status(context, {"status": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
