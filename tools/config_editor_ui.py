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
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
# ui_theme lives in tools/ next to this script (P0: inserting only ROOT
# made `import ui_theme` fail and the window die at startup when Mimics
# launched it — external windows must not depend on cwd).
for _candidate in (str(_HERE), str(ROOT)):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from ui_theme import clamp_window_to_screen, configure_application, scrollable_widget, stylesheet  # noqa: E402

BACKUP_DIR = ROOT / ".mimics_runtime" / "config_backups"

# field: (label, type, help). type: bool | int | float | str | choice(list)
# Only annotator-tunable keys are declared; everything else rides along
# unedited.
CONFIG_SCHEMAS: dict[str, dict[str, tuple]] = {
    "mimics_io_config.json": {
        "mimics_output_dir": (
            "默认 .mcs 输出文件夹", "str",
            "留空 = <数据集>/mcs_output。可用绝对或相对项目根目录的路径；"
            "文件夹被占用时导入仍会询问。",
        ),
        "mimics_background_exe": (
            "后台 Mimics 可执行文件", "str",
            "生成 .mcs 用的 materialise.exe 路径。留空 = 自动检测"
            "正在运行的 Mimics。",
        ),
    },
    "nninteractive_config.json": {
        "device": (
            "推理设备", "choice",
            "auto = 选第一个空闲 GPU；cpu = 强制用 CPU。训练占用 GPU 时"
            "建议选 cpu。",
        ),
        "auto_start_server": (
            "自动启动模型服务", "bool",
            "标注会话开始时自动启动 nnInteractive 服务。",
        ),
        "existing_mask_result_mode": (
            "已有 Mask 的结果模式", "choice",
            "ask = 每次询问如何处理所选 Mask；replace = 原地更新；"
            "copy = 总是创建可编辑副本。",
        ),
        "server_idle_timeout_seconds": (
            "服务空闲超时（秒）", "int",
            "模型服务空闲超过该秒数后自动退出。",
        ),
        "keep_server_warm_after_session": (
            "会话结束后保留模型服务", "bool",
            "标注会话之间保留模型服务，下次会话启动更快。",
        ),
        "minimum_free_gpu_memory_gb": (
            "最低空闲显存（GB）", "int",
            "GPU 空闲显存低于该值时，标注会话拒绝启动。",
        ),
    },
    "nninteractive_finetune_config.json": {
        "default_epochs": (
            "默认训练轮数", "int",
            "微调训练的轮数（允许范围由 minimum_epochs/maximum_epochs"
            "决定）。",
        ),
        "default_validation_fraction": (
            "验证集比例", "float",
            "训练时留作验证的病例占比。",
        ),
        "job_retention_days": (
            "任务保留天数", "int",
            "超过该天数的已完成微调任务会被清理。",
        ),
        "keep_failed_training_artifacts": (
            "保留失败训练产物", "bool",
            "微调失败后保留训练工作目录以便排查，而不是直接清理。",
        ),
    },
    "flexict_config.json": {
        "default_configuration": (
            "默认配置", "choice",
            "2d、3d_fullres、pair（主动学习用 2D+3D）或 auto"
            "（小显存用 2d，否则用 pair）。",
        ),
        "default_epochs": (
            "默认训练轮数", "int",
            "FlexiCT 微调轮数。150 为已验证配方。",
        ),
        "default_val_cases": (
            "验证病例数", "int",
            "需要划分时留作验证的病例数。",
        ),
        "job_retention_days": (
            "任务保留天数", "int",
            "超过该天数的已完成 FlexiCT 任务会被清理。",
        ),
        "pretrained_weights_dir": (
            "预训练权重文件夹", "str",
            "可选：覆盖 FlexiCT 主干权重文件夹。留空 = 用内置的"
            "integrations/flexict-finetune/weights。",
        ),
    },
    "interactive_algorithms_config.json": {
        "scribbleprompt.timeout_seconds": (
            "ScribblePrompt 超时（秒）", "int",
            "等待 ScribblePrompt 预测超过该秒数后放弃。",
        ),
        "scribbleprompt.device": (
            "ScribblePrompt 设备", "choice",
            "cpu 或 cuda。CPU 较慢但不与训练抢 GPU。",
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
    ("interactive_algorithms_config.json", "scribbleprompt.device"): [
        "cpu", "cuda",
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
        self.window.setWindowTitle("配置编辑器")
        self.window.resize(760, 640)
        root = QtWidgets.QVBoxLayout(self.window)
        root.setContentsMargins(22, 18, 22, 18)
        root.setSpacing(12)

        title = QtWidgets.QLabel("配置编辑器")
        title.setObjectName("title")
        root.addWidget(title)
        hint = QtWidgets.QLabel(
            "仅显示常用的配置项；每个文件中的其余内容原样保留。改动在相关窗口或服务下次启动时生效。"
        )
        hint.setObjectName("subtitle")
        hint.setWordWrap(True)
        root.addWidget(hint)

        tabs = QtWidgets.QTabWidget()
        root.addWidget(tabs, 1)
        self._load_errors: dict[str, str] = {}
        for file_name, schema in CONFIG_SCHEMAS.items():
            path = ROOT / file_name
            data, load_error = self._load_status(path)
            self.values[file_name] = data
            tab = QtWidgets.QWidget()
            form = QtWidgets.QFormLayout(tab)
            form.setContentsMargins(16, 14, 16, 14)
            form.setSpacing(10)
            if load_error:
                self._load_errors[file_name] = load_error
                warn = QtWidgets.QLabel(
                    "⚠ {0} —— 本页字段留空显示；修复该文件后重开本窗口。"
                    "在修复之前，保存将跳过该文件，不会覆盖它。".format(load_error)
                )
                warn.setObjectName("errorLabel")
                warn.setWordWrap(True)
                form.addRow(warn)
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
            titles = {
                "mimics_io_config.json": "数据导入导出",
                "nninteractive_config.json": "交互标注",
                "nninteractive_finetune_config.json": "交互模型训练",
                "flexict_config.json": "少样本训练",
                "interactive_algorithms_config.json": "单片修正",
            }
            index = tabs.addTab(scrollable_widget(tab), titles.get(file_name, file_name))
            tabs.setTabToolTip(index, file_name)

        buttons = QtWidgets.QHBoxLayout()
        self.rollback = QtWidgets.QPushButton("回滚上次保存")
        self.rollback.clicked.connect(self._rollback)
        self.rollback.setEnabled((BACKUP_DIR / "latest.json").is_file())
        buttons.addStretch(1)
        buttons.addWidget(self.rollback)
        save = QtWidgets.QPushButton("保存")
        save.setObjectName("primary")
        save.clicked.connect(self._save)
        buttons.addWidget(save)
        root.addLayout(buttons)

        self.status = QtWidgets.QLabel("")
        self.status.setObjectName("hint")
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        clamp_window_to_screen(self.window, (640, 440))

    # -- helpers ----------------------------------------------------------

    def _set_status(self, text, error=False):
        """One place for status coloring: an error must not look like a
        normal hint — a failed save read as success means silent loss."""
        self.status.setObjectName("errorLabel" if error else "hint")
        self.status.setStyleSheet(
            "color: #b42318; font-weight: 600;" if error else ""
        )
        self.status.setText(text)

    @staticmethod
    def _load(path: Path) -> dict:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return {}

    @staticmethod
    def _load_status(path: Path) -> tuple[dict, str]:
        """Load a config file, distinguishing "missing" from "broken".

        A broken file (unparseable or non-dict) must never be treated as an
        empty config: saving on top of that would silently rewrite the whole
        file down to the schema keys, destroying every undeclared key. The
        error string is non-empty exactly when the file exists but cannot be
        loaded.
        """
        if not path.is_file():
            return {}, ""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {}, "无法读取 {0}：{1}".format(path.name, exc)
        if not isinstance(data, dict):
            return {}, "无法读取 {0}：内容不是配置对象".format(path.name)
        return data, ""

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
            if file_name in self._load_errors:
                # Broken on load: writing the (empty) parsed values back
                # would rewrite the whole file down to the schema keys and
                # destroy every undeclared key. Skip it, tell the user.
                continue
            collected = self._collect(file_name)
            if isinstance(collected, tuple):
                bad_key, _ = collected
                self._set_status(
                    "{0}：'{1}' 必须是数字。".format(file_name, bad_key),
                    error=True,
                )
                return
            updates[file_name] = collected
        if not updates:
            self.status.setText(
                "没有可保存的文件：{0}。请先修复上述文件。".format(
                    "、".join(self._load_errors)
                )
            )
            return
        try:
            BACKUP_DIR.mkdir(parents=True, exist_ok=True)
            backup = {}
            for file_name in self.widgets:
                if file_name in self._load_errors:
                    # Broken on load: _load() would record {} for it, and a
                    # later rollback would overwrite the (still hand-
                    # recoverable) raw file with an empty object. The
                    # rollback skips files absent from the backup.
                    continue
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
            self._set_status("无法保存：{0}".format(exc), error=True)
            return
        self.rollback.setEnabled(True)
        self.status.setText(
            "已保存 {0} 个配置文件。相关窗口或服务重启后生效。{1}".format(
                len(updates),
                "已跳过无法读取的：{0}。".format(
                    "、".join(self._load_errors)
                ) if self._load_errors else "",
            )
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
            self._set_status("回滚失败：{0}".format(exc), error=True)
            return
        # Reload the whole window state from disk.
        self.values = {}
        self._load_errors = {}
        for name in self.widgets:
            data, load_error = self._load_status(ROOT / name)
            self.values[name] = data
            if load_error:
                self._load_errors[name] = load_error
        self._rebuild_widgets()
        self.status.setText("已回滚到上次保存的值。")

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
    configure_application(app, "配置编辑器")
    app.setStyleSheet(stylesheet())
    editor = ConfigEditor((QtCore, QtGui, QtWidgets))
    editor.window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
