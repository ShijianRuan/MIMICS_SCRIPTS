#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""环境检查/设置进行中的实时进度窗口（外部 PySide6，由 Mimics 端启动）。

读取 .mimics_runtime/setup_env_state.json，把 tools/setup_env.py 每步写入的
进度实时显示出来：当前步骤、已用时，以及"环境在网络盘上"的提示。状态到达
终态（ok / error / incomplete / extracted）后短暂显示结果并自动关闭，Mimics
端随后会弹出最终结果对话框。

状态文件可能残留上一次运行的终态：worker 在窗口启动前就已先行，且每个流程
都先写非终态，所以窗口第一个读到的终态必是陈旧的，不能当作本次结果展示或
触发关闭（R91-8 / N02）。状态读取由后台线程完成、Qt 线程只应用快照，网络盘
上的慢读取不会卡住计时器。

单独运行时（开发自检）：python tools/setup_progress_window.py
"""

import json
import sys
import threading
import time
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

ROOT = Path(__file__).resolve().parents[1]
STATE_FILE = ROOT / ".mimics_runtime" / "setup_env_state.json"
LOG_FILE = ROOT / ".mimics_runtime" / "setup_env.log"

TERMINAL = ("ok", "error", "incomplete", "extracted")

# 后台读状态文件的节奏；窗口自身 500ms 计时器只应用最新快照。
READ_INTERVAL = 0.25


class ProgressWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        # 本次运行的识别与显示状态（见 _begin_run / _apply_state）。
        self._active = False            # 已见到本次运行的非终态
        self._displaying_terminal = False
        self._run_started = None        # 状态里的 started_at_epoch（可无）
        self._baseline_epoch = None
        self._last_message = ""
        self._terminal_seen_at = None
        self._latest_state = None       # 读线程发布的最近一次快照
        self._stop_reader = threading.Event()
        self._build_ui()
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(500)
        self._reader = threading.Thread(target=self._read_loop)
        self._reader.daemon = True
        self._reader.start()

    def closeEvent(self, event):
        self._stop_reader.set()
        super().closeEvent(event)

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

    def _read_loop(self):
        """后台线程持续读状态文件，把最新快照发布给 Qt 线程。

        状态文件可能位于网络盘，读取可能很慢；放在这里读，窗口计时器
        的 _tick 只应用快照，永不被慢读取卡住（R91-8 / N02）。
        """
        while not self._stop_reader.is_set():
            self._latest_state = self._read_state()
            self._stop_reader.wait(READ_INTERVAL)

    def _tick(self, state=None):
        """应用最近一次状态快照（QTimer 定时触发；测试可显式传入状态）。"""
        if state is None:
            state = self._latest_state
        if state is None:
            return  # 还没读到任何状态，继续等
        self._apply_state(state)

    def _begin_run(self, state, run_started):
        """开始（或切换到）新的一次运行，全部运行期显示回到初始态。

        上次运行留在状态文件里的终态——100% 进度、旧耗时、2 秒关闭
        倒计时——不许被带进本次运行（R91-8 / N02）。"""
        self._active = True
        self._displaying_terminal = False
        self._run_started = run_started
        self._baseline_epoch = float(
            run_started or state.get("updated_at_epoch") or time.time())
        self._last_message = ""
        self._terminal_seen_at = None
        self.step_label.setText("正在启动…")
        self.bar.setRange(0, 0)  # 不确定进度（跑马灯）
        self.elapsed_label.setText("已用时 00:00")
        if state.get("network_drive"):
            self.network_warning.show()

    def _apply_state(self, state):
        status = state.get("status", "")
        message = str(state.get("message") or "")
        terminal = status in TERMINAL

        if status and not terminal:
            run_started = state.get("started_at_epoch")
            # 新一次运行的判定：从未见过运行、刚从终态回到运行态（新
            # 的 install 流程），或 started_at 变了（新的 check 流程）。
            # 同一流程内的常规写入（含 install 每步不带 started_at 的
            # 更新）不触发重置。
            if (not self._active or self._displaying_terminal
                    or run_started != self._run_started):
                self._begin_run(state, run_started)
        if not self._active:
            # 窗口先读到的是上次运行留下的终态：worker 启动后总会先写
            # 非终态，所以它不属于本次运行——显示"正在启动…"继续等，
            # 不拿旧结果当本次完成、不提前关闭。
            return
        self._displaying_terminal = terminal

        if message and message != self._last_message:
            self._last_message = message
            self.step_label.setText(message)

        if not terminal:
            self.elapsed_label.setText("已用时 {0}".format(
                self._fmt_elapsed(time.time() - self._baseline_epoch)))
            return

        if message:
            self.step_label.setText(message)
        self.bar.setRange(0, 1)
        self.bar.setValue(1)
        self.elapsed_label.setText("检查完成")
        if self._terminal_seen_at is None:
            self._terminal_seen_at = time.time()
        if time.time() - self._terminal_seen_at >= 2.0:
            self.close()


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = ProgressWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
