#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""环境检查/设置进行中的实时进度窗口（外部 PySide6，由 Mimics 端启动）。

读取 .mimics_runtime/setup_env_state.json，把 tools/setup_env.py 每步写入的
进度实时显示出来：当前步骤、已用时，以及"环境在网络盘上"的提示。状态到达
终态（ok / error / incomplete / extracted）后短暂显示结果并自动关闭，Mimics
端随后会弹出最终结果对话框。

单独运行时（开发自检）：python tools/setup_progress_window.py
"""

from __future__ import print_function

import json
import sys
import time
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

ROOT = Path(__file__).resolve().parents[1]
STATE_FILE = ROOT / ".mimics_runtime" / "setup_env_state.json"
LOG_FILE = ROOT / ".mimics_runtime" / "setup_env.log"

TERMINAL = ("ok", "error", "incomplete", "extracted")

ACTION_TITLES = {
    "check": "环境检查进行中",
    "install": "环境修复进行中",
    "offline-install": "离线安装进行中",
    "extract": "解压安装包进行中",
    "setup-from-scratch": "环境设置进行中",
}


class ProgressWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self._baseline_epoch = None
        self._last_message = ""
        self._terminal_seen_at = None
        self._build_ui()
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(500)

    def _build_ui(self):
        self.setWindowTitle("环境设置进度")
        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, True)
        self.setMinimumWidth(430)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)

        self.title_label = QtWidgets.QLabel("环境检查/设置进行中")
        font = self.title_label.font()
        font.setPointSize(font.pointSize() + 2)
        font.setBold(True)
        self.title_label.setFont(font)
        layout.addWidget(self.title_label)

        self.step_label = QtWidgets.QLabel("正在启动…")
        self.step_label.setWordWrap(True)
        layout.addWidget(self.step_label)

        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 0)  # 不确定进度（跑马灯）
        self.bar.setTextVisible(False)
        layout.addWidget(self.bar)

        self.elapsed_label = QtWidgets.QLabel("已用时 00:00")
        self.elapsed_label.setAlignment(QtCore.Qt.AlignRight)
        layout.addWidget(self.elapsed_label)

        self.network_warning = QtWidgets.QLabel(
            "检测到环境在网络盘上：检查和 AI 运行会非常慢。\n"
            "建议先把整个 MIMICS_SCRIPTS 文件夹复制到本地盘（如 D:\\）再继续。")
        self.network_warning.setWordWrap(True)
        palette = self.network_warning.palette()
        palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor("#8a5700"))
        self.network_warning.setPalette(palette)
        self.network_warning.hide()
        layout.addWidget(self.network_warning)

        self.footer = QtWidgets.QLabel("完整日志：{0}".format(LOG_FILE))
        self.footer.setStyleSheet("color: #667085; font-size: 11px;")
        self.footer.setWordWrap(True)
        layout.addWidget(self.footer)

    def _read_state(self):
        try:
            with open(str(STATE_FILE), "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    @staticmethod
    def _fmt_elapsed(seconds):
        minutes, secs = divmod(max(0, int(seconds)), 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return "{0}:{1:02d}:{2:02d}".format(hours, minutes, secs)
        return "{0:02d}:{1:02d}".format(minutes, secs)

    def _tick(self):
        state = self._read_state()
        status = state.get("status", "")
        message = str(state.get("message") or "")

        if self._baseline_epoch is None:
            baseline = state.get("started_at_epoch") or state.get("updated_at_epoch")
            if not baseline:
                return  # 状态文件还没写入，继续等
            self._baseline_epoch = float(baseline)
            action = state.get("action")
            self.title_label.setText(ACTION_TITLES.get(action, "环境检查/设置进行中"))
            if state.get("network_drive"):
                self.network_warning.show()

        if message and message != self._last_message:
            self._last_message = message
            self.step_label.setText(message)

        self.elapsed_label.setText(
            "已用时 {0}".format(self._fmt_elapsed(time.time() - self._baseline_epoch)))

        if status in TERMINAL:
            if message:
                self.step_label.setText(message)
            self.bar.setRange(0, 1)
            self.bar.setValue(1)
            self.elapsed_label.setText("检查完成")
            if self._terminal_seen_at is None:
                self._terminal_seen_at = time.time()
            if time.time() - self._terminal_seen_at >= 2.0:
                self.close()
                return


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = ProgressWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
