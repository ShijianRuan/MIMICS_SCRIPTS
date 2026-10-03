#!/usr/bin/env python3
"""Shared visual theme for external Mimics-Script PySide6 windows."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path


def _prepare_windows_dpi_awareness():
    """Enable native per-monitor rendering before QApplication creates a window."""
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    os.environ.setdefault("QT_SCALE_FACTOR_ROUNDING_POLICY", "PassThrough")
    if os.name != "nt":
        return
    try:
        import ctypes

        user32 = ctypes.windll.user32
        per_monitor_v2 = ctypes.c_void_p(-4)
        try:
            setter = user32.SetProcessDpiAwarenessContext
            setter.argtypes = [ctypes.c_void_p]
            setter.restype = ctypes.c_bool
            setter(per_monitor_v2)
        except Exception:
            try:
                shcore = ctypes.windll.shcore
                shcore.SetProcessDpiAwareness.argtypes = [ctypes.c_int]
                shcore.SetProcessDpiAwareness.restype = ctypes.c_long
                shcore.SetProcessDpiAwareness(2)
            except Exception:
                try:
                    user32.SetProcessDPIAware()
                except Exception:
                    pass
        # A Python executable manifest or Windows compatibility override can
        # lock the process context. The GUI main thread can still request PMv2.
        try:
            thread_setter = user32.SetThreadDpiAwarenessContext
            thread_setter.argtypes = [ctypes.c_void_p]
            thread_setter.restype = ctypes.c_void_p
            thread_setter(per_monitor_v2)
        except Exception:
            pass
    except Exception:
        pass


def _prepare_qt_high_dpi_policy():
    try:
        from PySide6 import QtCore, QtGui

        if QtCore.QCoreApplication.instance() is not None:
            return
        attribute = getattr(QtCore.Qt.ApplicationAttribute, "AA_UseHighDpiPixmaps", None)
        if attribute is not None:
            QtCore.QCoreApplication.setAttribute(attribute, True)
        policy_group = getattr(QtCore.Qt, "HighDpiScaleFactorRoundingPolicy", None)
        policy = getattr(policy_group, "PassThrough", None) if policy_group is not None else None
        if policy is not None:
            QtGui.QGuiApplication.setHighDpiScaleFactorRoundingPolicy(policy)
    except Exception:
        pass


_prepare_windows_dpi_awareness()
_prepare_qt_high_dpi_policy()


def _dialog_start_path(value, expect_directory=True):
    """Return a starting location without touching a slow or offline volume."""
    text = os.path.expandvars(os.path.expanduser(str(value or "").strip()))
    return text or str(Path.home())


def choose_existing_directory(QtWidgets, parent, title, initial=""):
    """Open the platform's familiar folder picker from an external UI process."""
    return str(
        QtWidgets.QFileDialog.getExistingDirectory(
            parent,
            str(title),
            _dialog_start_path(initial, expect_directory=True),
            QtWidgets.QFileDialog.ShowDirsOnly,
        )
        or ""
    )


def choose_open_file(
    QtWidgets,
    parent,
    title,
    initial="",
    file_filter="All files (*)",
):
    value, _selected_filter = QtWidgets.QFileDialog.getOpenFileName(
        parent,
        str(title),
        _dialog_start_path(initial, expect_directory=False),
        str(file_filter),
    )
    return str(value or "")


def choose_open_files(
    QtWidgets,
    parent,
    title,
    initial="",
    file_filter="All files (*)",
):
    values, _selected_filter = QtWidgets.QFileDialog.getOpenFileNames(
        parent,
        str(title),
        _dialog_start_path(initial, expect_directory=False),
        str(file_filter),
    )
    return [str(value) for value in values or []]


def choose_save_file(
    QtWidgets,
    parent,
    title,
    initial="",
    file_filter="All files (*)",
):
    value, _selected_filter = QtWidgets.QFileDialog.getSaveFileName(
        parent,
        str(title),
        str(initial or Path.home()),
        str(file_filter),
    )
    return str(value or "")


class _AsyncPathDialog:
    """Run a native file dialog in an isolated GUI process.

    Windows Shell can spend a long time enumerating disconnected drives or
    large SMB folders.  A native QFileDialog must run on its GUI thread, so a
    Python worker thread cannot make that dialog responsive.  Isolating it in
    a short-lived process keeps the calling setup window and Mimics responsive
    even while Explorer is waiting on the filesystem.
    """

    def __init__(
        self,
        QtCore,
        QtWidgets,
        parent,
        mode,
        title,
        initial,
        file_filter,
        callback,
        button=None,
        error_callback=None,
        timeout_seconds=3600,
    ):
        self.QtCore = QtCore
        self.QtWidgets = QtWidgets
        self.parent = parent
        self.callback = callback
        self.error_callback = error_callback
        self.button = button
        self.finished = False
        self.deadline = time.time() + max(60.0, float(timeout_seconds))
        runtime = Path(tempfile.gettempdir()) / "mimics_script_path_dialogs"
        runtime.mkdir(parents=True, exist_ok=True)
        token = "path_{}_{}".format(os.getpid(), uuid.uuid4().hex)
        self.request_path = runtime / (token + "_request.json")
        self.status_path = runtime / (token + "_status.json")
        self.stderr_path = runtime / (token + "_stderr.log")
        payload = {
            "mode": str(mode),
            "title": str(title),
            "initial": _dialog_start_path(initial, expect_directory=mode == "directory"),
            "file_filter": str(file_filter or "All files (*)"),
            "status_path": str(self.status_path),
        }
        self.request_path.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        helper = Path(__file__).with_name("path_dialog_helper.py")
        command = [sys.executable, str(helper), "--request", str(self.request_path)]
        if os.name == "nt" and Path(command[0]).name.lower() == "python.exe":
            pythonw = Path(command[0]).with_name("pythonw.exe")
            if pythonw.is_file():
                command[0] = str(pythonw)
        try:
            self.stderr_handle = self.stderr_path.open(
                "w", encoding="utf-8", errors="replace"
            )
            try:
                self.process = subprocess.Popen(
                    command,
                    cwd=str(helper.parent.parent),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=self.stderr_handle,
                    env=dict(os.environ),
                )
            finally:
                self.stderr_handle.close()
        except Exception:
            for path in (self.request_path, self.status_path, self.stderr_path):
                try:
                    path.unlink()
                except OSError:
                    pass
            raise
        if self.button is not None:
            self.button.setEnabled(False)
            self.button.setText("Opening...")
        self.timer = QtCore.QTimer(parent)
        self.timer.timeout.connect(self._poll)
        self.timer.start(100)
        try:
            parent.destroyed.connect(lambda *_args: self.cancel(cleanup=True))
        except Exception:
            pass

    def _read_status(self):
        try:
            value = json.loads(self.status_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else None
        except Exception:
            return None

    def _diagnostic(self):
        try:
            return self.stderr_path.read_text(
                encoding="utf-8", errors="replace"
            )[-2000:].strip()
        except Exception:
            return ""

    def _poll(self):
        if self.finished:
            return
        status = self._read_status()
        if status and str(status.get("status") or "") in {
            "submitted",
            "cancelled",
            "failed",
        }:
            self._complete(status)
            return
        returncode = self.process.poll()
        if returncode is not None:
            self._complete(
                {
                    "status": "failed",
                    "error": (
                        "The path window closed before returning a selection "
                        "(exit code {}). {}"
                    ).format(returncode, self._diagnostic()),
                }
            )
            return
        if time.time() >= self.deadline:
            self.cancel()
            self._complete(
                {
                    "status": "failed",
                    "error": "The path window timed out. The selected drive may be offline.",
                }
            )

    def _complete(self, status):
        if self.finished:
            return
        self.finished = True
        self.timer.stop()
        if self.button is not None:
            original = str(self.button.property("pathDialogText") or "Browse...")
            self.button.setText(original)
            self.button.setEnabled(True)
        state = str(status.get("status") or "failed")
        if state == "submitted":
            self.callback(status.get("selection"))
        elif state == "failed":
            message = str(status.get("error") or "路径选择窗口失败。")
            if self.error_callback is not None:
                self.error_callback(message)
            else:
                self.QtWidgets.QMessageBox.warning(
                    self.parent, "路径选择失败", message
                )
        controllers = getattr(self.parent, "_mimics_path_dialogs", [])
        if self in controllers:
            controllers.remove(self)
        self._cleanup()

    def _cleanup(self):
        for path in (self.request_path, self.status_path):
            try:
                path.unlink()
            except OSError:
                pass
        if not self._diagnostic():
            try:
                self.stderr_path.unlink()
            except OSError:
                pass

    def cancel(self, cleanup=False):
        if self.finished:
            return
        try:
            if self.process.poll() is None:
                self.process.terminate()
        except Exception:
            pass
        if cleanup:
            self.finished = True
            try:
                self.timer.stop()
            except Exception:
                pass
            self._cleanup()


def choose_path_async(
    QtCore,
    QtWidgets,
    parent,
    mode,
    title,
    initial,
    callback,
    file_filter="All files (*)",
    button=None,
    error_callback=None,
):
    """Open a native path dialog without blocking the calling Qt window."""
    if button is not None:
        button.setProperty("pathDialogText", button.text())
    try:
        controller = _AsyncPathDialog(
            QtCore,
            QtWidgets,
            parent,
            mode,
            title,
            initial,
            file_filter,
            callback,
            button=button,
            error_callback=error_callback,
        )
    except Exception as exc:
        if button is not None:
            button.setText(str(button.property("pathDialogText") or "Browse..."))
            button.setEnabled(True)
        message = "无法打开路径选择窗口：{}".format(exc)
        if error_callback is not None:
            error_callback(message)
        else:
            QtWidgets.QMessageBox.warning(parent, "路径选择失败", message)
        return None
    controllers = getattr(parent, "_mimics_path_dialogs", None)
    if controllers is None:
        controllers = []
        setattr(parent, "_mimics_path_dialogs", controllers)
    controllers.append(controller)
    return controller


def choose_existing_directory_async(
    QtCore, QtWidgets, parent, title, initial, callback, button=None, error_callback=None
):
    return choose_path_async(
        QtCore,
        QtWidgets,
        parent,
        "directory",
        title,
        initial,
        callback,
        button=button,
        error_callback=error_callback,
    )


def choose_open_file_async(
    QtCore,
    QtWidgets,
    parent,
    title,
    initial,
    file_filter,
    callback,
    button=None,
    error_callback=None,
):
    return choose_path_async(
        QtCore,
        QtWidgets,
        parent,
        "open_file",
        title,
        initial,
        callback,
        file_filter=file_filter,
        button=button,
        error_callback=error_callback,
    )


def choose_open_files_async(
    QtCore,
    QtWidgets,
    parent,
    title,
    initial,
    file_filter,
    callback,
    button=None,
    error_callback=None,
):
    return choose_path_async(
        QtCore,
        QtWidgets,
        parent,
        "open_files",
        title,
        initial,
        callback,
        file_filter=file_filter,
        button=button,
        error_callback=error_callback,
    )


def choose_save_file_async(
    QtCore,
    QtWidgets,
    parent,
    title,
    initial,
    file_filter,
    callback,
    button=None,
    error_callback=None,
):
    return choose_path_async(
        QtCore,
        QtWidgets,
        parent,
        "save_file",
        title,
        initial,
        callback,
        file_filter=file_filter,
        button=button,
        error_callback=error_callback,
    )


# Semantic status palette. Windows that color individual rows/labels must
# draw from here (enforced by TestUiThemePalette in test_all.py) instead of
# inventing near-identical hexes, which is how the windows drifted apart.
PALETTE = {
    "text": "#182230",
    "text_muted": "#667085",
    "border": "#d9e0e8",
    "border_input": "#c9d2dc",
    "success": "#067647",  # finished / ok
    "warning": "#8a5700",  # cancelled / degraded
    "danger": "#b42318",   # failed / error
    "info": "#175cd3",     # still running / neutral emphasis
    "accent": "#2563eb",   # primary buttons, focus, loss curves
    "teal": "#0f766e",     # progress, previews, AUC curves
}


def stylesheet(extra=""):
    base = """
    QMainWindow, QDialog, QWidget {
        background: #f5f7fa;
        color: #182230;
        font-family: "Segoe UI", "Microsoft YaHei UI", "Microsoft YaHei", "Helvetica Neue", "Arial";
        font-size: 10pt;
    }
    QLabel {
        background: transparent;
    }
    QLabel#titleLabel, QLabel#title {
        color: #101828;
        font-size: 18pt;
        font-weight: 650;
    }
    QLabel#subtitleLabel, QLabel#subtitle {
        color: #667085;
    }
    QLabel#warningLabel {
        color: #8a5700;
        background: #fff8e7;
        border: 1px solid #f1d28a;
        border-radius: 6px;
        padding: 7px 9px;
    }
    QLabel#section {
        color: #344054;
        font-size: 9pt;
        font-weight: 650;
    }
    QLabel#hint {
        color: #667085;
        font-size: 9pt;
    }
    QLabel#preview {
        color: #0f766e;
        font-size: 9pt;
        font-weight: 600;
    }
    QLabel#liveLabel {
        color: #067647;
        font-size: 9pt;
        font-weight: 600;
    }
    QLabel#errorLabel {
        color: #b42318;
        font-size: 9pt;
        font-weight: 600;
    }
    QFrame#panel, QFrame#surface {
        background: #ffffff;
        border: 1px solid #d9e0e8;
        border-radius: 7px;
    }
    QGroupBox {
        background: #ffffff;
        border: 1px solid #d9e0e8;
        border-radius: 7px;
        margin-top: 12px;
        padding-top: 14px;
    }
    QGroupBox QWidget {
        background: transparent;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 6px;
        color: #344054;
        font-weight: 650;
    }
    QTabWidget::pane {
        border: 1px solid #d9e0e8;
        background: #ffffff;
        border-radius: 7px;
    }
    QTabBar::tab {
        padding: 8px 18px;
        background: #eef2f6;
        color: #475467;
        border: 1px solid #d9e0e8;
        border-bottom: none;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
    }
    QTabBar::tab:selected {
        background: #ffffff;
        color: #101828;
        font-weight: 650;
    }
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QListWidget, QTableWidget,
    QTextEdit, QPlainTextEdit {
        background: #ffffff;
        border: 1px solid #c9d2dc;
        border-radius: 6px;
        padding: 5px 7px;
        min-height: 22px;
        selection-background-color: #dbeafe;
        selection-color: #101828;
    }
    QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus,
    QListWidget:focus, QTableWidget:focus, QTextEdit:focus, QPlainTextEdit:focus {
        border-color: #2563eb;
    }
    QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {
        color: #98a2b3;
        background: #f2f4f7;
        border-color: #e1e6ec;
    }
    QComboBox QAbstractItemView {
        background: #ffffff;
        border: 1px solid #c9d2dc;
        selection-background-color: #dbeafe;
        selection-color: #101828;
    }
    QListWidget::item {
        padding: 6px;
    }
    QListWidget::item:selected {
        background: #dbeafe;
        color: #101828;
    }
    QTableWidget {
        gridline-color: #e6ebf0;
        alternate-background-color: #f8fafc;
    }
    QTableWidget::item {
        padding: 5px 7px;
    }
    QTableWidget::item:selected {
        background: #dbeafe;
        color: #101828;
    }
    QHeaderView::section {
        background: #eef2f6;
        color: #475467;
        border: none;
        border-right: 1px solid #d9e0e8;
        border-bottom: 1px solid #d9e0e8;
        padding: 7px 8px;
        font-weight: 650;
    }
    QTableCornerButton::section {
        background: #eef2f6;
        border: none;
        border-right: 1px solid #d9e0e8;
        border-bottom: 1px solid #d9e0e8;
    }
    QScrollArea, QScrollArea > QWidget > QWidget {
        background: transparent;
        border: none;
    }
    QScrollBar:vertical {
        background: transparent;
        width: 12px;
        margin: 2px;
    }
    QScrollBar::handle:vertical {
        background: #b8c2ce;
        border-radius: 4px;
        min-height: 32px;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0;
    }
    QPushButton {
        min-height: 30px;
        background: #ffffff;
        border: 1px solid #aeb9c6;
        border-radius: 6px;
        padding: 2px 13px;
        color: #253244;
    }
    QPushButton:hover {
        background: #f0f4f8;
        border-color: #8795a6;
    }
    QPushButton:pressed {
        background: #e5ebf2;
    }
    QPushButton:disabled {
        color: #98a2b3;
        background: #f2f4f7;
        border-color: #d9e0e8;
    }
    QPushButton#primaryButton, QPushButton#primary {
        background: #2563eb;
        color: #ffffff;
        border-color: #2563eb;
        font-weight: 650;
    }
    QPushButton#primaryButton:hover, QPushButton#primary:hover {
        background: #1d4ed8;
        border-color: #1d4ed8;
    }
    /* F15: the ID selector outranks the generic QPushButton:disabled rule,
    so a disabled primary stayed saturated blue and looked clickable. */
    QPushButton#primaryButton:disabled, QPushButton#primary:disabled {
        color: #f2f4f7;
        background: #93b0e8;
        border-color: #93b0e8;
    }
    QPushButton#dangerButton {
        color: #b42318;
        border-color: #f0b3ad;
        background: #fff7f6;
    }
    QPushButton#dangerButton:hover {
        background: #feeceb;
    }
    QProgressBar {
        min-height: 18px;
        border: 1px solid #c9d2dc;
        border-radius: 5px;
        background: #eef2f6;
        text-align: center;
        color: #344054;
    }
    QProgressBar::chunk {
        border-radius: 4px;
        background: #0f766e;
    }
    QCheckBox, QRadioButton {
        spacing: 7px;
        color: #344054;
    }
    QCheckBox::indicator, QRadioButton::indicator {
        width: 16px;
        height: 16px;
    }
    QSplitter::handle {
        background: #e4e9ef;
    }
    """
    return base + "\n" + str(extra or "")


def _application_icon():
    """A simple programmatic icon so our windows stop showing the generic
    Python logo. Drawn, not loaded: no binary resource to ship or misplace."""
    try:
        from PySide6 import QtCore, QtGui

        icon = QtGui.QIcon()
        for size in (16, 24, 32, 48):
            pixmap = QtGui.QPixmap(size, size)
            pixmap.fill(QtCore.Qt.transparent)
            painter = QtGui.QPainter(pixmap)
            try:
                painter.setRenderHint(QtGui.QPainter.Antialiasing)
                margin = 1 + size // 16
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(QtGui.QColor(PALETTE["teal"]))
                painter.drawRoundedRect(
                    QtCore.QRect(margin, margin, size - 2 * margin, size - 2 * margin),
                    size // 5,
                    size // 5,
                )
                painter.setBrush(QtGui.QColor("#ffffff"))
                # A stylized "M" mark: two vertical bars and the connecting V.
                bar = max(2, size // 8)
                top = size // 3
                bottom = size - 2 * margin - max(2, size // 6)
                painter.drawRect(margin + size // 4, top, bar, bottom - top)
                painter.drawRect(size - margin - size // 4 - bar, top, bar, bottom - top)
                painter.drawPolygon(
                    QtGui.QPolygon(
                        [
                            QtCore.QPoint(size // 2 - bar, top + bar),
                            QtCore.QPoint(size // 2 + bar, top + bar),
                            QtCore.QPoint(size // 2, top + (bottom - top) // 2),
                        ]
                    )
                )
            finally:
                # A QPainter left active on its pixmap aborts the process at
                # destruction time if anything above raised - never skip end().
                painter.end()
            icon.addPixmap(pixmap)
        return icon
    except Exception:
        return None


def configure_application(app, name="Mimics Script"):
    app.setApplicationName(str(name))
    try:
        app.setOrganizationName("Mimics Script")
    except Exception:
        pass
    icon = _application_icon()
    if icon is not None:
        try:
            app.setWindowIcon(icon)
        except Exception:
            pass
    try:
        if os.name == "nt":
            from PySide6 import QtWidgets

            styles = {str(value).lower(): str(value) for value in QtWidgets.QStyleFactory.keys()}
            app.setStyle(styles.get("windowsvista", styles.get("windows", "Fusion")))
        else:
            app.setStyle("Fusion")
    except Exception:
        try:
            app.setStyle("Fusion")
        except Exception:
            pass
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "MimicsScript.ExternalTools"
            )
        except Exception:
            pass
    app.setStyleSheet(stylesheet())
