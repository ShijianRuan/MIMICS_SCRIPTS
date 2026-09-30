#!/usr/bin/env python3
"""External preset editor for CT window/level values.

The Mimics-side entries (03_Review window presets) only let an annotator
pick one of the shipped presets. This window adds what the file previously
hard-coded: hand-entered width/level, custom presets, keyword editing, and
persistence to ``window_level_presets.json`` — without touching Mimics.

Editing happens on a copy; Save writes atomically and keeps a one-step
backup so a bad edit can be rolled back.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui_theme import configure_application, stylesheet  # noqa: E402

PRESETS_PATH = ROOT / "window_level_presets.json"
BACKUP_PATH = ROOT / ".mimics_runtime" / "window_level_presets_backup.json"


def load_presets(path: Path = PRESETS_PATH) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
    except Exception:
        pass
    return []


def save_presets(presets: list[dict], path: Path = PRESETS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(presets, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    tmp.replace(path)


def write_backup(presets: list[dict], path: Path = BACKUP_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(presets, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


class PresetEditor:
    def __init__(self, qt_modules):
        self.QtCore, self.QtGui, self.QtWidgets = qt_modules
        QtWidgets = self.QtWidgets
        self._original = load_presets()
        self.presets = [dict(row) for row in self._original]

        self.window = QtWidgets.QWidget()
        self.window.setWindowTitle("窗宽/窗位预设")
        self.window.resize(720, 520)
        root = QtWidgets.QVBoxLayout(self.window)
        root.setContentsMargins(22, 18, 22, 18)
        root.setSpacing(12)

        title = QtWidgets.QLabel("窗宽/窗位预设")
        title.setObjectName("title")
        root.addWidget(title)
        hint = QtWidgets.QLabel(
            "宽度和窗位为 HU 值。关键词（逗号分隔）自动匹配 Mimics 中的 Mask 名称；所选 Mask 名称包含任一关键词时应用该预设。"
        )
        hint.setObjectName("subtitle")
        hint.setWordWrap(True)
        root.addWidget(hint)

        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(['名称', '窗宽 (WW)', '窗位 (WL)'])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        root.addWidget(self.table, 1)

        keywords_label = QtWidgets.QLabel("所选预设的关键词：")
        keywords_label.setObjectName("section")
        root.addWidget(keywords_label)
        self.keywords = QtWidgets.QLineEdit()
        self.keywords.setPlaceholderText("liver, hepatic, bile")
        self.keywords.editingFinished.connect(self._store_keywords)
        root.addWidget(self.keywords)

        self.table.currentCellChanged.connect(self._show_keywords)
        self.table.itemChanged.connect(self._cell_changed)

        buttons = QtWidgets.QHBoxLayout()
        add = QtWidgets.QPushButton("添加预设")
        add.clicked.connect(self._add_row)
        remove = QtWidgets.QPushButton("移除所选")
        remove.clicked.connect(self._remove_row)
        self.rollback = QtWidgets.QPushButton("回滚上次保存")
        self.rollback.clicked.connect(self._do_rollback)
        self.rollback.setEnabled(BACKUP_PATH.is_file())
        buttons.addWidget(add)
        buttons.addWidget(remove)
        buttons.addStretch(1)
        buttons.addWidget(self.rollback)
        save = QtWidgets.QPushButton("保存")
        save.setObjectName("primary")
        save.clicked.connect(self._do_save)
        buttons.addWidget(save)
        root.addLayout(buttons)

        self.status = QtWidgets.QLabel("")
        self.status.setObjectName("hint")
        root.addWidget(self.status)
        self._reload()

    # -- table <-> model ------------------------------------------------

    def _reload(self):
        QtWidgets = self.QtWidgets
        self.table.itemChanged.disconnect(self._cell_changed)
        self.table.setRowCount(len(self.presets))
        for row, preset in enumerate(self.presets):
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(preset.get("name", ""))))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(preset.get("width", ""))))
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(preset.get("level", ""))))
        self.table.itemChanged.connect(self._cell_changed)
        if self.presets:
            self.table.selectRow(0)
            self._show_keywords(0, 0)
        else:
            self.keywords.clear()

    def _cell_changed(self, item):
        row, column = item.row(), item.column()
        if row >= len(self.presets):
            return
        text = item.text().strip()
        if column == 0:
            self.presets[row]["name"] = text
        else:
            key = "width" if column == 1 else "level"
            try:
                self.presets[row][key] = int(float(text))
            except ValueError:
                self.status.setText(
                    '"{0}" is not a number; enter integer HU values.'.format(text)
                )

    def _show_keywords(self, row, _column):
        if 0 <= row < len(self.presets):
            self.keywords.setText(
                ", ".join(str(k) for k in self.presets[row].get("keywords", []))
            )

    def _store_keywords(self):
        row = self.table.currentRow()
        if not (0 <= row < len(self.presets)):
            return
        parts = [
            p.strip() for p in self.keywords.text().split(",") if p.strip()
        ]
        self.presets[row]["keywords"] = parts

    # -- actions ----------------------------------------------------------

    def _add_row(self):
        self.presets.append(
            {"name": "New preset", "width": 400, "level": 50, "keywords": []}
        )
        self._reload()
        self.table.selectRow(len(self.presets) - 1)
        self.status.setText("已添加预设；编辑该行后点击保存。")

    def _remove_row(self):
        row = self.table.currentRow()
        if not (0 <= row < len(self.presets)):
            return
        del self.presets[row]
        self._reload()
        self.status.setText("已移除预设；保存后生效（可用“回滚”撤销）。")

    def _do_save(self):
        # Validate before touching the file.
        for index, preset in enumerate(self.presets):
            if not str(preset.get("name", "")).strip():
                self.status.setText("Row {0}: name is empty.".format(index + 1))
                return
            for key in ("width", "level"):
                value = preset.get(key)
                if not isinstance(value, int):
                    self.status.setText(
                        "Row {0}: {1} must be an integer HU value.".format(
                            index + 1, key
                        )
                    )
                    return
        self._store_keywords()
        try:
            write_backup(load_presets())
            save_presets(self.presets)
        except OSError as exc:
            self.status.setText("Could not write presets: {0}".format(exc))
            return
        self._original = [dict(row) for row in self.presets]
        self.rollback.setEnabled(True)
        self.status.setText(
            "Saved {0} presets at {1}.".format(
                len(self.presets), time.strftime("%H:%M:%S")
            )
        )

    def _do_rollback(self):
        if not BACKUP_PATH.is_file():
            return
        try:
            backup = load_presets(BACKUP_PATH)
            save_presets(backup)
        except OSError as exc:
            self.status.setText("Roll back failed: {0}".format(exc))
            return
        self.presets = [dict(row) for row in backup]
        self._reload()
        self.status.setText("已回滚到上次保存的预设。")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    from PySide6 import QtCore, QtGui, QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    configure_application(app, "窗宽/窗位预设")
    app.setStyleSheet(stylesheet())
    editor = PresetEditor((QtCore, QtGui, QtWidgets))
    editor.window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
