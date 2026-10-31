#!/usr/bin/env python3
"""External one-stop manager for AI models of all three families.

Annotators receive model packages (zip bundles produced by
``ai_model_bundle.py``) from a developer or a colleague and import them
here — no folders to copy, no JSON to edit. External nnU-Net v2 training
results (a results folder or an official .zip) get their own import
entry: the model is validated, published, and immediately selectable in
the nnU-Net prediction window. The window lists every registered
nnInteractive task model, managed nnU-Net model, and FlexiCT model in
one table (family, target, created, source, usability), lets the user
switch the recommended/current model per family, and removes broken
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
from ui_theme import (  # noqa: E402
    choose_existing_directory_async,
    choose_open_file_async,
    configure_application,
    clamp_window_to_screen,
    stylesheet,
)

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


def _model_source(model: dict) -> str:
    """Human-readable provenance: where this model was trained/imported."""
    if str(model.get("origin") or "") == "external_pretrained_unverified":
        return "外部预训练（未本地验证）"
    if model.get("imported_from_bundle"):
        return "导入"
    if str(model.get("execution_backend") or "") == "remote":
        name = str(model.get("remote_profile_name") or "").strip()
        return "远程: " + (name or str(model.get("remote_profile_id") or "?"))
    return ""


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
                        "source": _model_source(model),
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
                    "source": _model_source(model),
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
                        " + ".join(model.get("label_names") or [])
                        or model.get("label_name")
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
                    "source": _model_source(model),
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
    # nnU-Net has no --set-current by design (it picks the model per
    # prediction, see set_recommended_model); only nninteractive/flexict do.
    if set_current and family != "nnunet":
        argv.append("--set-current")
    try:
        ai_model_bundle.main(argv)
    except SystemExit as exc:  # argparse exits with code 2 on bad flags
        raise RuntimeError(
            "模型包导入命令执行失败（参数错误）。请确认模型包为标准导出格式。"
        ) from exc
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
            raise RuntimeError("未找到该目标：{0}".format(row["task_id"]))
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
        remaining = [
            model
            for model in flexict_common.load_models(workspace)
            if str(model.get("model_id") or "") != row["model_id"]
        ]
        # If the removed model was the recommended one, clear the pointer so
        # the registry does not recommend a model that no longer exists.
        was_recommended = flexict_common.load_registry_recommended_id(
            workspace
        ) == row["model_id"]
        flexict_common.save_registry(
            workspace,
            None,
            remaining,
            recommended_model_id="" if was_recommended else None,
        )


class ModelManagerWindow:
    def __init__(self, context, qt_modules):
        self.context = context
        self.QtCore, self.QtGui, self.QtWidgets = qt_modules
        QtWidgets = self.QtWidgets
        self.workspaces = dict(context.get("workspaces") or {}) or default_workspaces()
        self.rows: list[dict] = []
        self._loading = True
        self._importing = False
        self._external_busy = False
        self._external_source: Path | None = None
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
            "导入收到的模型包 (zip) 或外部 nnU-Net 训练结果（结果文件夹 / 官方 .zip），在一处查看所有系列的全部模型，并切换默认使用的模型。不会覆盖任何内容：旧模型未移除前，同名导入会被拒绝。"
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
        # F15: fixed readable widths for the identity/time columns so the
        # stretch column cannot truncate them.
        self.table.setColumnWidth(2, 170)   # 模型 (model id)
        self.table.setColumnWidth(4, 150)   # 创建时间
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._update_action_buttons)
        root.addWidget(self.table, 1)

        self.selection_detail = QtWidgets.QPlainTextEdit()
        self.selection_detail.setReadOnly(True)
        self.selection_detail.setMinimumHeight(56)
        self.selection_detail.setMaximumHeight(86)
        self.selection_detail.setAccessibleName("所选模型完整信息")
        self.selection_detail.hide()
        root.addWidget(self.selection_detail)

        self.status_label = QtWidgets.QLabel("正在加载模型...")
        self.status_label.setObjectName("hint")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        actions = QtWidgets.QHBoxLayout()
        self.import_button = QtWidgets.QPushButton("导入模型包...")
        self.import_button.setObjectName("primary")
        self.import_button.clicked.connect(self._import_clicked)
        actions.addWidget(self.import_button)
        self.external_import_button = QtWidgets.QPushButton("导入 nnU-Net 训练结果...")
        self.external_import_button.clicked.connect(self._import_external_clicked)
        actions.addWidget(self.external_import_button)
        self.use_button = QtWidgets.QPushButton("设为默认模型")
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
        clamp_window_to_screen(self.window, (820, 480))

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
                self._results.put(("load", collect_rows(workspaces), ""))
            except Exception as exc:
                self._results.put(("load", [], str(exc)))

        worker = threading.Thread(target=load, name="model-manager-discovery")
        worker.daemon = True
        worker.start()

    def _poll(self):
        if self._external_busy:
            try:
                kind, payload, error = self._results.get_nowait()
            except Empty:
                return
            if kind not in ("external_scan", "external_import"):
                # Bundle-import/load results: keep them for their branch.
                self._results.put((kind, payload, error))
                return
            self._external_busy = False
            self._set_import_buttons_enabled(True)
            if error:
                self._warn("{0}：{1}".format(
                    "扫描失败" if kind == "external_scan" else "导入失败", error
                ))
            elif kind == "external_scan":
                self._external_scan_done(payload)
            else:
                self._external_import_done(payload)
            return
        if self._importing:
            try:
                kind, payload, error = self._results.get_nowait()
            except Empty:
                return
            if kind != "import":
                # A concurrent _reload result; keep it for the load branch.
                self._results.put((kind, payload, error))
                return
            self._importing = False
            self._set_import_buttons_enabled(True)
            if error:
                # 弹窗而非状态栏：_reload 会立刻改写状态栏文案，行内
                # 错误提示一闪而过，标注者根本看不到（P1-2）。
                self.QtWidgets.QMessageBox.warning(
                    self.window,
                    "AI 模型管理器",
                    "导入失败：{0}".format(error),
                )
            else:
                self.QtWidgets.QMessageBox.information(
                    self.window,
                    "导入完成",
                    "模型包已导入并注册，现在是其任务的默认模型。",
                )
            self._reload()
            return
        if not self._loading:
            return
        try:
            kind, rows, error = self._results.get_nowait()
        except Empty:
            return
        if kind != "load":
            # Import results are handled above; a stray entry stays queued.
            self._results.put((kind, rows, error))
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
                status = "当前默认" if row["current"] else (
                    "可用" if row["usable"] else "已损坏"
                )
                values = [
                    row["family"],
                    row["target"] or row["task_id"],
                    row["model_id"],
                    row["configuration"],
                    created,
                    row.get("source") or (
                        "导入" if row["imported"] else "本机训练"
                    ),
                    status,
                ]
                for column, value in enumerate(values):
                    item = QtWidgets.QTableWidgetItem(str(value))
                    # 长内容（模型 ID、任务名等）会被列宽省略成 "..."，
                    # 悬停即见完整内容，用户无需拖宽列（F15）。
                    item.setToolTip(str(value))
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
        self.selection_detail.setVisible(bool(row))
        if row:
            usage = (
                "在 nnU-Net 预测窗口选择此模型。"
                if row["family"] == "nnunet"
                else "设为默认仅影响后续模型选择，不切换正在标注的会话。"
            )
            self.selection_detail.setPlainText(
                "目标：{0}\n模型：{1}\n{2}".format(
                    row.get("target") or row.get("task_id") or "未声明",
                    row.get("model_id") or "未声明", usage,
                )
            )
        self.use_button.setVisible(bool(row and row["family"] != "nnunet"))
        self.remove_button.setVisible(bool(row and not row["usable"]))
        # nnU-Net picks its model in the prediction window at predict time;
        # "use this model" would only raise for those rows.
        self.use_button.setEnabled(
            bool(row and row["usable"] and row["family"] != "nnunet")
        )
        self.use_button.setToolTip(
            "nnU-Net 在每次预测时选择模型，请使用预测窗口选择。"
            if row and row["family"] == "nnunet"
            else ""
        )
        self.remove_button.setEnabled(bool(row and not row["usable"]))

    def _selected_row(self):
        row = self.table.currentRow()
        if 0 <= row < len(self.rows):
            return self.rows[row]
        return None

    def _import_clicked(self):
        choose_open_file_async(
            self.QtCore,
            self.QtWidgets,
            self.window,
            "选择模型包",
            str(Path.home()),
            "模型包 (*.zip)",
            self._import_bundle_from,
        )

    def _import_bundle_from(self, path):
        if not path:
            return
        # A bundle is a multi-GB zip: extracting + checksumming on the GUI
        # thread would freeze the whole window for tens of seconds. Same
        # thread+queue pattern as _reload; the buttons stay disabled until
        # the import finishes so double-clicks cannot start two imports.
        self._set_import_buttons_enabled(False)
        self.status_label.setStyleSheet("")
        self.status_label.setText("正在导入模型包（较大的包可能需要几十秒）...")

        def worker():
            try:
                import_bundle(Path(path), self.workspaces, set_current=True)
                self._results.put(("import", None, ""))
            except Exception as exc:
                self._results.put(("import", None, str(exc)))

        thread = threading.Thread(target=worker, name="model-manager-import")
        thread.daemon = True
        thread.start()
        self._importing = True

    def _set_import_buttons_enabled(self, state: bool):
        self.import_button.setEnabled(state)
        self.external_import_button.setEnabled(state)

    def _import_external_clicked(self):
        QtWidgets = self.QtWidgets
        box = QtWidgets.QMessageBox(self.window)
        box.setWindowTitle("导入 nnU-Net 训练结果")
        box.setText("请选择来源类型：nnU-Net 训练结果文件夹或官方导出的 .zip 包。")
        folder_button = box.addButton(
            "训练结果文件夹...", QtWidgets.QMessageBox.AcceptRole
        )
        zip_button = box.addButton(
            "官方 .zip 包...", QtWidgets.QMessageBox.AcceptRole
        )
        box.addButton("取消", QtWidgets.QMessageBox.RejectRole)
        box.exec_()
        clicked = box.clickedButton()
        if clicked is folder_button:
            choose_existing_directory_async(
                self.QtCore,
                self.QtWidgets,
                self.window,
                "选择 nnU-Net 训练结果文件夹",
                str(Path.home()),
                self._external_scan_from,
            )
        elif clicked is zip_button:
            choose_open_file_async(
                self.QtCore,
                self.QtWidgets,
                self.window,
                "选择 nnU-Net 官方导出包",
                str(Path.home()),
                "nnU-Net 导出包 (*.zip)",
                self._external_scan_from,
            )

    def _external_scan_from(self, path):
        if not path:
            return
        # 扫描要解压 zip 并逐文件校验（可能数 GB），放后台线程；两个导入
        # 按钮同时禁用，防止扫描期间再发起另一个导入。nnunet_external_model
        # 顶层 import torch，延迟到工作线程加载，窗口启动不被拖慢。
        self._external_source = Path(path)
        self._set_import_buttons_enabled(False)
        self.status_label.setStyleSheet("")
        self.status_label.setText(
            "正在扫描 nnU-Net 训练结果（较大的包可能需要几十秒）..."
        )

        def worker():
            import nnunet_external_model as external

            try:
                report = external.scan_external_model(self._external_source)
                self._results.put(("external_scan", report, ""))
            except Exception as exc:
                self._results.put(("external_scan", None, str(exc)))

        thread = threading.Thread(
            target=worker, name="model-manager-external-scan"
        )
        thread.daemon = True
        thread.start()
        self._external_busy = True

    def _external_scan_done(self, report):
        """Scan finished: pick configuration/fold, then import in background."""
        QtWidgets = self.QtWidgets
        importable = [str(name) for name in report.get("importable") or []]
        if not importable:
            details = []
            for row in report.get("candidates") or []:
                blockers = "；".join(row.get("blockers") or [])
                if blockers:
                    details.append("{0}：{1}".format(
                        row.get("configuration") or row.get("config_dir"),
                        blockers,
                    ))
            for row in report.get("rejected_candidates") or []:
                details.append("{0}：缺少 {1}".format(
                    row.get("config_dir"),
                    "、".join(row.get("missing") or []),
                ))
            self._warn(
                "该训练结果当前无法导入：\n{0}".format(
                    "\n".join(details)
                    if details
                    else str(report.get("hint") or "原因未知。")
                )
            )
            return
        candidates = {
            str(row.get("configuration")): row
            for row in report.get("candidates") or []
        }
        configuration = None
        if len(importable) > 1:
            choice, ok = QtWidgets.QInputDialog.getItem(
                self.window,
                "选择配置",
                "该训练结果包含多个可用配置，请选择要导入的配置：\n"
                "（常用：3d_fullres＝三维全分辨率，效果通常最好；"
                "2d＝逐层二维切片，速度更快）",
                importable,
                0,
                False,
            )
            if not ok:
                return
            configuration = str(choice)
        row = candidates.get(
            configuration if configuration is not None else importable[0]
        )
        use_folds = None
        folds = [str(name) for name in (row or {}).get("usable_folds") or []]
        if (row or {}).get("requires_fold_selection") and folds:
            default = "fold_all" if "fold_all" in folds else folds[0]
            choice, ok = QtWidgets.QInputDialog.getItem(
                self.window,
                "选择 fold",
                "该训练结果同时包含 fold_all 与数字 fold，请选择导入时使用的 fold：\n"
                "（fold_all＝用全部数据训练的最终模型，推荐；"
                "数字 fold＝只用部分数据训练，适合快速验证）",
                folds,
                folds.index(default),
                False,
            )
            if not ok:
                return
            use_folds = [str(choice)]
        fallback = str((row or {}).get("fallback_checkpoint") or "")
        if fallback:
            # 中断训练的结果只有最优权重可用：导入前说清楚，别让标注者
            # 在预测结果异常时才发现权重不是最终权重。
            answer = QtWidgets.QMessageBox.question(
                self.window,
                "没有最终权重",
                "该训练结果没有 checkpoint_final.pth（训练未跑满或中断导出），\n"
                "导入后将使用验证集最优权重 {0} 进行推理。\n\n继续导入吗？".format(
                    fallback
                ),
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if answer != QtWidgets.QMessageBox.Yes:
                return
        self._external_import_from(
            use_folds=use_folds, configuration=configuration
        )

    def _external_import_from(self, use_folds=None, configuration=None):
        source = self._external_source
        if source is None:
            return
        self._set_import_buttons_enabled(False)
        self.status_label.setStyleSheet("")
        self.status_label.setText(
            "正在导入并校验 nnU-Net 模型（复制与校验可能需要一两分钟）..."
        )
        workspace = Path(
            self.workspaces.get("nnunet") or default_workspaces()["nnunet"]
        )

        def worker():
            import nnunet_external_model as external

            try:
                result = external.import_external_model(
                    source,
                    workspace,
                    use_folds=use_folds,
                    configuration=configuration,
                )
                self._results.put(("external_import", result, ""))
            except Exception as exc:
                self._results.put(("external_import", None, str(exc)))

        thread = threading.Thread(
            target=worker, name="model-manager-external-import"
        )
        thread.daemon = True
        thread.start()
        self._external_busy = True

    def _external_import_done(self, result):
        manifest = result.get("manifest") or {}
        labels = "、".join(
            str(item.get("name"))
            for item in manifest.get("labels") or []
            if isinstance(item, dict)
        ) or "未声明"
        channels = "、".join(
            str(value)
            for value in (manifest.get("channel_names") or {}).values()
        ) or "未声明"
        source = manifest.get("source") or {}
        source_path = str(source.get("path") or "")
        source_text = (
            "官方 ZIP：{0}".format(source_path)
            if str(source.get("kind")) == "zip"
            else "文件夹：{0}".format(source_path)
        )
        total = sum(
            int(entry.get("bytes") or 0)
            for entry in manifest.get("files") or []
            if isinstance(entry, dict)
        )
        size_text = (
            "{0:.2f} GB".format(total / 1024.0 ** 3)
            if total >= 1024 ** 3
            else "{0:.1f} MB".format(total / 1024.0 ** 2)
        )
        usable, reason = nnunet_common.model_usability(manifest)
        status_text = "可正常使用" if usable else "不可用：{0}".format(reason)
        recipe = manifest.get("recipe") or {}
        checkpoint_name = str(recipe.get("checkpoint_name") or "")
        checkpoint_text = (
            "权重：{0}（训练未跑满，使用验证集最优权重）\n".format(checkpoint_name)
            if checkpoint_name and checkpoint_name != "checkpoint_final.pth"
            else ""
        )
        reused = bool(result.get("reused"))
        self.QtWidgets.QMessageBox.information(
            self.window,
            "该模型此前已导入" if reused else "导入完成",
            "模型名：{0}\n分割目标：{1}\n要求的输入：{2}\n来源：{3}\n"
            "大小：{4}\n兼容状态：{5}\n{6}{7}\n\n"
            "外部预训练／未做本地效果验证；"
            "请在 nnU-Net 预测窗口选择此模型使用。".format(
                manifest.get("display_name")
                or manifest.get("task_name")
                or manifest.get("task_id")
                or "未声明",
                labels,
                channels,
                source_text,
                size_text,
                status_text,
                checkpoint_text,
                "相同内容的模型此前已导入，未生成重复版本。"
                if reused
                else "模型已发布到本机模型库。",
            ),
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
