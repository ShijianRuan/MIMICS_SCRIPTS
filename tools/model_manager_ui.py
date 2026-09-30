#!/usr/bin/env python3
"""External one-stop manager for AI models of all three families.

Annotators receive model packages (zip bundles produced by
``ai_model_bundle.py``) from a developer or a colleague and import them
here — no folders to copy, no JSON to edit. The window lists every
registered nnInteractive task model, managed nnU-Net model, and FlexiCT
model in one table (family, target, created, source, usability), lets the
user switch the recommended/current model per family, and removes broken
registry rows. Rollback = import the previous bundle again or re-select
an earlier model; the table keeps every imported version visible.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
import zipfile
from pathlib import Path
from queue import Empty, Queue

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "tools"):
    value = str(candidate)
    if value not in sys.path:
        sys.path.insert(0, value)

import ai_model_bundle  # noqa: E402
import flexict_common  # noqa: E402
import nninteractive_task_common as task_common  # noqa: E402
import nnunet_common  # noqa: E402
from nnunet_common import read_json  # noqa: E402
from ui_theme import configure_application, stylesheet  # noqa: E402

FAMILIES = ("nninteractive", "nnunet", "flexict")
_IMPORT_COMMANDS = {
    "nninteractive_task": "import-nninteractive",
    "nnunet": "import-nnunet",
    "flexict": "import-flexict",
}


def bundle_model_family(bundle_path: Path) -> str:
    """Read model_family from a bundle's bundle.json without extracting."""
    try:
        with zipfile.ZipFile(bundle_path, "r") as archive:
            payload = json.loads(
                archive.read("bundle.json").decode("utf-8")
            )
    except (zipfile.BadZipFile, KeyError, ValueError):
        return ""
    return str(payload.get("model_family") or "")


def default_workspaces() -> dict[str, str]:
    return {
        "nninteractive": str(task_common.workspace_root()),
        "nnunet": str(Path.home() / ".mimics_script" / "nnunet"),
        "flexict": str(flexict_common.workspace_paths()["root"]),
    }


def collect_rows(workspaces: dict[str, str]) -> list[dict]:
    """One flat row list across all three families for the model table."""
    rows: list[dict] = []
    try:
        workspace = Path(workspaces.get("nninteractive") or "")
        for task in task_common.task_rows(workspace):
            task_id = str(task.get("task_id") or "")
            for model in task.get("models") or []:
                if not isinstance(model, dict):
                    continue
                usable = task_common.model_is_usable(model)
                resolved = task_common.resolve_registered_model_dir(
                    workspace, model, task
                )
                compatible = False
                if resolved:
                    compatible = bool(
                        task_common.audit_model_dir(resolved).get("compatible")
                    )
                current = (
                    str(task.get("recommended_model_id") or "")
                    == str(model.get("model_id") or "")
                )
                rows.append(
                    {
                        "family": "nninteractive",
                        "workspace": str(workspace),
                        "task_id": task_id,
                        "model_id": str(model.get("model_id") or ""),
                        "target": str(
                            task.get("task_name") or task_id
                        ),
                        "configuration": "",
                        "created_at_epoch": float(
                            model.get("created_at_epoch") or 0
                        ),
                        "imported": bool(model.get("imported_from_bundle")),
                        "usable": bool(usable and compatible),
                        "current": current,
                    }
                )
    except Exception:
        pass
    try:
        workspace = Path(workspaces.get("nnunet") or "")
        for model in nnunet_common.load_models(workspace, include_missing=True):
            usable, _ = nnunet_common.model_usability(model)
            rows.append(
                {
                    "family": "nnunet",
                    "workspace": str(workspace),
                    "task_id": str(model.get("task_id") or ""),
                    "model_id": str(model.get("model_id") or ""),
                    "target": str(
                        model.get("task_name") or model.get("task_id") or ""
                    ),
                    "configuration": str(model.get("configuration") or ""),
                    "created_at_epoch": float(
                        model.get("created_at_epoch") or 0
                    ),
                    "imported": bool(model.get("imported_from_bundle")),
                    "usable": bool(usable),
                    "current": False,
                }
            )
    except Exception:
        pass
    try:
        workspace = Path(workspaces.get("flexict") or "")
        registry: dict = {}
        try:
            registry = json.loads(
                flexict_common.workspace_paths(workspace)["registry"]
                .read_text(encoding="utf-8")
            )
        except Exception:
            registry = {}
        recommended_id = str(registry.get("recommended_model_id") or "")
        for model in flexict_common.load_models(workspace):
            usable, _ = flexict_common.model_usability(model)
            rows.append(
                {
                    "family": "flexict",
                    "workspace": str(workspace),
                    "task_id": str(model.get("task_id") or ""),
                    "model_id": str(model.get("model_id") or ""),
                    "target": str(
                        model.get("label_name")
                        or model.get("task_name")
                        or model.get("task_id")
                        or ""
                    ),
                    "configuration": "2D"
                    if str(model.get("configuration") or "") == "2d"
                    else "3D"
                    if str(model.get("configuration") or "") == "3d_fullres"
                    else str(model.get("configuration") or ""),
                    "created_at_epoch": float(
                        model.get("created_at_epoch") or 0
                    ),
                    "imported": bool(model.get("imported_from_bundle")),
                    "usable": bool(usable),
                    "current": (
                        str(model.get("model_id") or "") == recommended_id
                    ),
                }
            )
    except Exception:
        pass
    rows.sort(
        key=lambda row: (
            FAMILIES.index(row["family"])
            if row["family"] in FAMILIES
            else len(FAMILIES),
            -row["created_at_epoch"],
        )
    )
    return rows


def import_bundle(bundle_path: Path, workspaces: dict[str, str],
                  set_current: bool = True) -> list[str]:
    """Import a bundle into the matching family workspace.

    Returns the printed destination path(s). Raises RuntimeError with an
    annotator-friendly message on any failure (ai_model_bundle already
    validates family, files, and checksums before anything is written).
    """
    family = bundle_model_family(bundle_path)
    command = _IMPORT_COMMANDS.get(family)
    if not command:
        known = ", ".join(sorted(_IMPORT_COMMANDS))
        raise RuntimeError(
            "该文件不是可识别的模型包（应为以下类型之一：{0}）。"
            "请让发送方使用 ai_model_bundle 重新导出。".format(known)
        )
    argv = [
        command,
        "--workspace",
        workspaces[family if family != "nninteractive_task" else "nninteractive"],
        "--bundle",
        str(bundle_path),
    ]
    if set_current:
        argv.append("--set-current")
    try:
        ai_model_bundle.main(argv)
    except Exception as exc:
        raise RuntimeError(str(exc))
    return [str(bundle_path)]


def set_recommended_model(row: dict) -> None:
    """Make the row's model the recommended/current model of its family."""
    family = row["family"]
    if family == "nninteractive":
        workspace = Path(row["workspace"])
        task = task_common.find_task(workspace, row["task_id"])
        if not task:
            raise RuntimeError("Task was not found: {0}".format(row["task_id"]))
        task["recommended_model_id"] = row["model_id"]
        task["updated_at_epoch"] = time.time()
        from nnunet_common import write_json_atomic

        task_common.task_dir(workspace, row["task_id"])
        write_json_atomic(
            task_common.task_dir(workspace, row["task_id"]) / "task.json",
            task,
        )
        registry = task_common.load_registry(workspace)
        for entry in registry.get("tasks") or []:
            if task_common.safe_slug(entry.get("task_id")) == task_common.safe_slug(
                row["task_id"]
            ):
                entry["recommended_model_id"] = row["model_id"]
        task_common.save_registry(workspace, registry)
    elif family == "nnunet":
        raise RuntimeError(
            "nnU-Net 在每次预测时选择模型，请使用预测窗口选择模型。"
        )
    else:  # flexict
        flexict_common.save_registry(
            Path(row["workspace"]),
            None,
            flexict_common.load_models(Path(row["workspace"])),
            recommended_model_id=row["model_id"],
        )


def remove_model_row(row: dict) -> None:
    """Drop a broken registry row (folder already gone) without deleting
    any files. Only rows whose model folder is missing can be removed."""
    family = row["family"]
    if family == "nninteractive":
        workspace = Path(row["workspace"])
        registry = task_common.load_registry(workspace)
        for entry in registry.get("tasks") or []:
            if task_common.safe_slug(entry.get("task_id")) != task_common.safe_slug(
                row["task_id"]
            ):
                continue
            entry["models"] = [
                model
                for model in entry.get("models") or []
                if str(model.get("model_id") or "") != row["model_id"]
            ]
            if str(entry.get("recommended_model_id") or "") == row["model_id"]:
                entry["recommended_model_id"] = ""
        task_common.save_registry(workspace, registry)
    elif family == "nnunet":
        workspace = Path(row["workspace"])
        for registry_path in nnunet_common.model_registry_paths(workspace):
            payload = read_json(registry_path, {}) or {}
            models = [
                dict(model)
                for model in payload.get("models") or []
                if isinstance(model, dict)
                and str(model.get("model_id") or "") != row["model_id"]
            ]
            if len(models) != len(payload.get("models") or []):
                nnunet_common.write_json_atomic(registry_path, {
                    "schema_version": payload.get("schema_version")
                    or "mimics_nnunet_model_registry.v1",
                    "models": models,
                    "updated_at_epoch": time.time(),
                })
    else:  # flexict
        workspace = Path(row["workspace"])
        flexict_common.save_registry(
            workspace,
            None,
            [
                model
                for model in flexict_common.load_models(workspace)
                if str(model.get("model_id") or "") != row["model_id"]
            ],
        )


class ModelManagerWindow:
    def __init__(self, context, qt_modules):
        self.context = context
        self.QtCore, self.QtGui, self.QtWidgets = qt_modules
        QtWidgets = self.QtWidgets
        self.workspaces = dict(context.get("workspaces") or {}) or default_workspaces()
        self.rows: list[dict] = []
        self._loading = True
        self._results: Queue = Queue()

        self.window = QtWidgets.QDialog()
        self.window.setWindowTitle("AI 模型管理器")
        self.window.resize(980, 600)
        self.window.setMinimumSize(820, 480)
        root = QtWidgets.QVBoxLayout(self.window)
        root.setContentsMargins(22, 18, 22, 18)
        root.setSpacing(12)
        title = QtWidgets.QLabel("AI 模型管理器")
        title.setObjectName("title")
        root.addWidget(title)
        subtitle = QtWidgets.QLabel(
            "导入收到的模型包 (zip)，在一处查看所有系列的全部模型，并切换默认使用的模型。不会覆盖任何内容：旧模型未移除前，同名导入会被拒绝。"
        )
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        root.addWidget(subtitle)

        self.table = QtWidgets.QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ['系列', '目标', '模型', '配置', '创建时间',
             '来源', '状态']
        )
        self.table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.Stretch
        )
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._update_action_buttons)
        root.addWidget(self.table, 1)

        self.status_label = QtWidgets.QLabel("正在加载模型...")
        self.status_label.setObjectName("hint")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        actions = QtWidgets.QHBoxLayout()
        self.import_button = QtWidgets.QPushButton("导入模型包...")
        self.import_button.setObjectName("primary")
        self.import_button.clicked.connect(self._import_clicked)
        actions.addWidget(self.import_button)
        self.use_button = QtWidgets.QPushButton("使用此模型")
        self.use_button.clicked.connect(self._use_clicked)
        actions.addWidget(self.use_button)
        self.remove_button = QtWidgets.QPushButton("移除失效条目")
        self.remove_button.clicked.connect(self._remove_clicked)
        actions.addWidget(self.remove_button)
        actions.addStretch(1)
        refresh = QtWidgets.QPushButton("刷新")
        refresh.clicked.connect(self._reload)
        actions.addWidget(refresh)
        close = QtWidgets.QPushButton("关闭")
        close.clicked.connect(self.window.close)
        actions.addWidget(close)
        root.addLayout(actions)

        self._timer = self.QtCore.QTimer(self.window)
        self._timer.timeout.connect(self._poll)
        self._timer.start(120)
        self._reload()

    # -- data ---------------------------------------------------------------

    def _reload(self):
        self._loading = True
        self.status_label.setStyleSheet("")
        self.status_label.setText("正在加载模型...")
        workspaces = dict(self.workspaces)

        def load():
            try:
                self._results.put((collect_rows(workspaces), ""))
            except Exception as exc:
                self._results.put(([], str(exc)))

        worker = threading.Thread(target=load, name="model-manager-discovery")
        worker.daemon = True
        worker.start()

    def _poll(self):
        if not self._loading:
            return
        try:
            rows, error = self._results.get_nowait()
        except Empty:
            return
        self._loading = False
        self.rows = rows
        self._fill_table()
        if error:
            self.status_label.setStyleSheet("color: #b42318; font-weight: 600;")
            self.status_label.setText("无法加载模型：{0}".format(error))
        elif not rows:
            self.status_label.setText(
                "尚无已注册模型。导入一个模型包即可开始。"
            )
        else:
            self.status_label.setText(
                "共 {0} 个模型。高亮行为对应系列/任务的当前默认模型。".format(len(rows))
            )

    def _fill_table(self):
        QtWidgets = self.QtWidgets
        self.table.setUpdatesEnabled(False)
        try:
            self.table.setRowCount(len(self.rows))
            for index, row in enumerate(self.rows):
                created = time.strftime(
                    "%Y-%m-%d %H:%M",
                    time.localtime(row.get("created_at_epoch") or 0),
                )
                status = "current" if row["current"] else (
                    "usable" if row["usable"] else "broken"
                )
                values = [
                    row["family"],
                    row["target"] or row["task_id"],
                    row["model_id"],
                    row["configuration"],
                    created,
                    "imported" if row["imported"] else "trained here",
                    status,
                ]
                for column, value in enumerate(values):
                    item = QtWidgets.QTableWidgetItem(str(value))
                    if row["current"]:
                        item.setFont(self.QtGui.QFont(item.font().family(), weight=75))
                    elif not row["usable"]:
                        item.setForeground(
                            self.QtGui.QColor("#b42318")
                        )
                    self.table.setItem(index, column, item)
        finally:
            self.table.setUpdatesEnabled(True)
        self._update_action_buttons()

    # -- actions ------------------------------------------------------------

    def _update_action_buttons(self):
        row = self._selected_row()
        self.use_button.setEnabled(bool(row and row["usable"]))
        self.remove_button.setEnabled(bool(row and not row["usable"]))

    def _selected_row(self):
        row = self.table.currentRow()
        if 0 <= row < len(self.rows):
            return self.rows[row]
        return None

    def _import_clicked(self):
        QtWidgets = self.QtWidgets
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.window,
            "选择模型包",
            str(Path.home()),
            "模型包 (*.zip)",
        )
        if not path:
            return
        try:
            import_bundle(Path(path), self.workspaces, set_current=True)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self.window, "导入失败", str(exc)
            )
            return
        QtWidgets.QMessageBox.information(
            self.window,
            "导入完成",
            "模型包已导入并注册，现在是其任务的默认模型。",
        )
        self._reload()

    def _use_clicked(self):
        row = self._selected_row()
        if not row:
            return
        if not row["usable"]:
            self._warn(
                "该模型已损坏（文件缺失）。请先修复或重新导入后再使用。"
            )
            return
        try:
            set_recommended_model(row)
        except Exception as exc:
            self._warn(str(exc))
            return
        self._reload()

    def _remove_clicked(self):
        row = self._selected_row()
        if not row:
            return
        if row["usable"]:
            self._warn(
                "此处只能移除模型文件夹已丢失的条目，该模型的文件仍然存在。"
            )
            return
        try:
            remove_model_row(row)
        except Exception as exc:
            self._warn(str(exc))
            return
        self._reload()

    def _warn(self, text):
        self.QtWidgets.QMessageBox.warning(self.window, "AI 模型管理器", text)

    def show(self):
        self.window.show()
        self.window.raise_()
        self.window.activateWindow()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context")
    args = parser.parse_args()
    context = {}
    if args.context:
        context = read_json(Path(args.context).expanduser().resolve(), {}) or {}
    try:
        from PySide6 import QtCore, QtGui, QtWidgets

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        configure_application(app, "AI 模型管理器")
        app.setStyleSheet(stylesheet())
        window = ModelManagerWindow(context, (QtCore, QtGui, QtWidgets))
        window.show()
        return int(app.exec())
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
