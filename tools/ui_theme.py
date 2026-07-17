#!/usr/bin/env python3
"""Shared visual theme for external Mimics-Script PySide6 windows."""

from __future__ import annotations

import os


def stylesheet(extra=""):
    base = """
    QMainWindow, QDialog, QWidget {
        background: #f5f7fa;
        color: #182230;
        font-family: "Segoe UI", "Microsoft YaHei UI", "Microsoft YaHei", "Helvetica Neue", sans-serif;
        font-size: 10pt;
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
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QListWidget, QTextEdit {
        background: #ffffff;
        border: 1px solid #c9d2dc;
        border-radius: 6px;
        padding: 5px 7px;
        selection-background-color: #dbeafe;
        selection-color: #101828;
    }
    QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus,
    QListWidget:focus, QTextEdit:focus {
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
