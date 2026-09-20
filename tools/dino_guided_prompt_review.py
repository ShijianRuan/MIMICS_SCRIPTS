#!/usr/bin/env python3
"""Non-modal review controls for DINO-guided nnInteractive points."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for candidate in (str(HERE), str(ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from ui_theme import configure_application, stylesheet as shared_stylesheet


TITLE = "Review Suggested Points"


def read_json(path, default=None):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else default
    except Exception:
        return default


def write_json_atomic(path, payload, retries=20):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    last_error = None
    for attempt in range(max(1, int(retries))):
        temporary = path.with_name(
            path.name + ".{}.{}.tmp".format(os.getpid(), uuid.uuid4().hex)
        )
        try:
            temporary.write_text(serialized, encoding="utf-8")
            os.replace(str(temporary), str(path))
            return
        except OSError as exc:
            last_error = exc
            try:
                temporary.unlink()
            except OSError:
                pass
            time.sleep(min(0.25, 0.03 * (attempt + 1)))
    if last_error is not None:
        raise last_error


class ReviewWindow(object):
    def __init__(self, window, context, qt):
        self.window = window
        self.context = context
        self.status_path = Path(context["status_path"])
        self.QtCore, self.QtGui, self.QtWidgets = qt
        self.finished = False
        self.request_id = 0
        self.request_sent_epoch = 0.0
        self.foreground_count = 0
        self.background_count = 0
        self.automatic_foreground_count = 0
        self.automatic_background_count = 0
        self.state = ""
        self._build()
        self.timer = self.QtCore.QTimer(window)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(350)
        self.refresh()

    def _build(self):
        QtWidgets = self.QtWidgets
        self.window.setWindowTitle(TITLE)
        self.window.resize(600, 390)
        self.window.setMinimumSize(540, 350)
        self.window.setStyleSheet(shared_stylesheet())
        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(12)

        title = QtWidgets.QLabel("Review DINOv3 suggestions")
        title.setObjectName("titleLabel")
        subtitle = QtWidgets.QLabel(
            "Move, add, or delete the temporary points in Mimics, then run the selected nnInteractive model."
        )
        subtitle.setObjectName("subtitleLabel")
        subtitle.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(subtitle)

        summary = QtWidgets.QFrame()
        summary.setObjectName("card")
        grid = QtWidgets.QGridLayout(summary)
        grid.setContentsMargins(16, 14, 16, 14)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        self.fg_count = QtWidgets.QLabel("0 total")
        self.fg_count.setObjectName("valueLabel")
        self.bg_count = QtWidgets.QLabel("0 total")
        self.bg_count.setObjectName("valueLabel")
        grid.addWidget(QtWidgets.QLabel("Foreground points"), 0, 0)
        grid.addWidget(self.fg_count, 0, 1)
        self.remove_fg = QtWidgets.QPushButton("−")
        self.remove_fg.setToolTip("Remove the last foreground suggestion")
        self.remove_fg.setFixedWidth(38)
        self.remove_fg.clicked.connect(lambda: self.request("remove_foreground"))
        self.add_fg = QtWidgets.QPushButton("+")
        self.add_fg.setToolTip("Add one foreground point in Mimics")
        self.add_fg.setFixedWidth(38)
        self.add_fg.clicked.connect(lambda: self.request("add_foreground"))
        grid.addWidget(self.remove_fg, 0, 2)
        grid.addWidget(self.add_fg, 0, 3)
        grid.addWidget(QtWidgets.QLabel("Background points"), 1, 0)
        grid.addWidget(self.bg_count, 1, 1)
        self.remove_bg = QtWidgets.QPushButton("−")
        self.remove_bg.setToolTip("Remove the last background suggestion")
        self.remove_bg.setFixedWidth(38)
        self.remove_bg.clicked.connect(lambda: self.request("remove_background"))
        self.add_bg = QtWidgets.QPushButton("+")
        self.add_bg.setToolTip("Add one background point in Mimics")
        self.add_bg.setFixedWidth(38)
        self.add_bg.clicked.connect(lambda: self.request("add_background"))
        grid.addWidget(self.remove_bg, 1, 2)
        grid.addWidget(self.add_bg, 1, 3)
        grid.setColumnStretch(0, 1)
        root.addWidget(summary)

        model_row = QtWidgets.QWidget()
        model_layout = QtWidgets.QHBoxLayout(model_row)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.setSpacing(10)
        model_layout.addWidget(QtWidgets.QLabel("nnInteractive model"))
        self.model_combo = QtWidgets.QComboBox()
        self.model_options = list(self.context.get("model_options") or [])
        for option in self.model_options:
            self.model_combo.addItem(
                str(option.get("label") or "nnInteractive"),
                str(option.get("key") or "official"),
            )
        default_key = str(self.context.get("default_model_key") or "official")
        for index in range(self.model_combo.count()):
            if str(self.model_combo.itemData(index)) == default_key:
                self.model_combo.setCurrentIndex(index)
                break
        self.model_combo.currentIndexChanged.connect(self.select_model)
        model_layout.addWidget(self.model_combo, 1)
        root.addWidget(model_row)

        note = QtWidgets.QLabel(
            "Green points include anatomy; red points exclude it. The automatic proposal uses at most three of each, but you may add or delete as many review points as needed. All temporary points are removed after inference or cancellation."
        )
        note.setWordWrap(True)
        note.setObjectName("hintLabel")
        root.addWidget(note)
        self.status_label = QtWidgets.QLabel("Waiting for Mimics...")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)
        root.addStretch(1)

        actions = QtWidgets.QHBoxLayout()
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel)
        actions.addWidget(self.cancel_button)
        actions.addStretch(1)
        self.run_button = QtWidgets.QPushButton("Run nnInteractive")
        self.run_button.setObjectName("primaryButton")
        self.run_button.clicked.connect(self.confirm)
        actions.addWidget(self.run_button)
        root.addLayout(actions)
        self.window.setCentralWidget(central)

    def update_status(self, payload):
        current = read_json(self.status_path, {}) or {}
        current.update(payload)
        current["updated_at_epoch"] = time.time()
        write_json_atomic(self.status_path, current)

    def request(self, command):
        self.request_id += 1
        self.request_sent_epoch = time.time()
        self.status_label.setText("Waiting for Mimics to apply the point change...")
        self.update_status(
            {
                "request": {
                    "id": self.request_id,
                    "command": str(command),
                }
            }
        )

    def select_model(self):
        key = self.model_combo.currentData()
        if key:
            self.update_status({"selected_model_key": str(key)})

    def refresh(self):
        status = read_json(self.status_path, {}) or {}
        state = str(status.get("status") or "")
        self.state = state
        self.foreground_count = int(status.get("foreground_count", 0) or 0)
        self.background_count = int(status.get("background_count", 0) or 0)
        self.automatic_foreground_count = int(
            status.get("automatic_foreground_count", 0) or 0
        )
        self.automatic_background_count = int(
            status.get("automatic_background_count", 0) or 0
        )
        self.fg_count.setText(
            "{} total ({} suggested)".format(
                self.foreground_count, self.automatic_foreground_count
            )
        )
        self.bg_count.setText(
            "{} total ({} suggested)".format(
                self.background_count, self.automatic_background_count
            )
        )
        # A point add/remove request is considered handled once Mimics writes
        # back the matching handled_request_id, clears request_pending, or the
        # 5s safety timeout elapses. The timeout guards against a lost status
        # write (read-modify-write race with the Mimics thread) leaving the UI
        # permanently stuck on "Waiting for Mimics...".
        handled_id = status.get("handled_request_id")
        acknowledged = (
            not bool(status.get("request_pending"))
            or (handled_id is not None and int(handled_id or 0) == self.request_id)
        )
        timed_out = (
            self.request_sent_epoch > 0.0
            and time.time() - self.request_sent_epoch > 5.0
        )
        if acknowledged or timed_out:
            self.request_sent_epoch = 0.0
        busy = bool(status.get("request_pending")) and not acknowledged and not timed_out
        editable = state == "reviewing" and not busy
        self.add_fg.setEnabled(editable)
        self.add_bg.setEnabled(editable)
        self.remove_fg.setEnabled(editable and self.foreground_count > 0)
        self.remove_bg.setEnabled(editable and self.background_count > 0)
        self.run_button.setEnabled(editable and self.foreground_count > 0)
        self.cancel_button.setEnabled(state in ("reviewing", "confirmed"))
        self.model_combo.setEnabled(editable)
        message = str(status.get("message") or "Review points in Mimics.")
        self.status_label.setText(message)
        if state in ("submitted", "cancelled", "failed"):
            self.finished = True
            self.window.close()

    def confirm(self):
        if self.foreground_count < 1:
            return
        self.run_button.setEnabled(False)
        self.update_status({"status": "confirmed", "message": "Starting nnInteractive..."})

    def cancel(self):
        if self.state == "submitting":
            return
        self.finished = True
        self.update_status({"status": "cancelled", "message": "Cancelled."})
        self.window.close()

    def close_event(self, event):
        if self.state == "submitting":
            event.ignore()
            return
        if not self.finished:
            self.cancel()
        event.accept()


def run(context):
    from PySide6 import QtCore, QtGui, QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    configure_application(app, TITLE)

    class CloseAwareWindow(QtWidgets.QMainWindow):
        def closeEvent(self, event):
            controller = getattr(self, "controller", None)
            if controller is None:
                event.accept()
            else:
                controller.close_event(event)

    window = CloseAwareWindow()
    controller = ReviewWindow(window, context, (QtCore, QtGui, QtWidgets))
    window.controller = controller
    window.show()
    window.raise_()
    window.activateWindow()
    return app.exec()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", required=True)
    args = parser.parse_args(argv)
    context = read_json(args.context, {}) or {}
    try:
        return run(context)
    except Exception as exc:
        if context.get("status_path"):
            status = read_json(context["status_path"], {}) or {}
            status.update({"status": "failed", "error": str(exc)})
            write_json_atomic(context["status_path"], status)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
