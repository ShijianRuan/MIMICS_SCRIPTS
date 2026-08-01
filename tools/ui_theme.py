#!/usr/bin/env python3
"""Shared visual theme for external Mimics-Script PySide6 windows."""

from __future__ import annotations

import os
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


def configure_application(app, name="Mimics Script"):
    app.setApplicationName(str(name))
    try:
        app.setOrganizationName("Mimics Script")
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
