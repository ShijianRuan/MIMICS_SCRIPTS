#!/usr/bin/env python3
"""Shared PySide6 controls for optional remote training servers."""

from __future__ import annotations

import os
import threading
import time
from queue import Empty, Queue
from pathlib import Path
from typing import Any

try:
    from remote_compute import (
        DEFAULT_GPU_DEVICE,
        DEFAULT_IMAGE,
        DEFAULT_REMOTE_ROOT,
        HostKeyChangedError,
        RemoteComputeError,
        UnknownHostKeyError,
        delete_profile,
        load_profiles,
        normalize_profile,
        save_profile,
        store_password,
        test_connection,
    )
    from ui_theme import choose_open_file_async, clamp_window_to_screen, scrollable_widget
except ImportError:
    from tools.remote_compute import (
        DEFAULT_GPU_DEVICE,
        DEFAULT_IMAGE,
        DEFAULT_REMOTE_ROOT,
        HostKeyChangedError,
        RemoteComputeError,
        UnknownHostKeyError,
        delete_profile,
        load_profiles,
        normalize_profile,
        save_profile,
        store_password,
        test_connection,
    )
    from tools.ui_theme import choose_open_file_async, clamp_window_to_screen, scrollable_widget


def _format_bytes(value: object) -> str:
    try:
        size = float(value)
    except Exception:
        return "-"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return "{:.1f} {}".format(size, unit)
        size /= 1024.0
    return "-"


class ServerProfilesDialog:
    def __init__(self, parent, qt_modules, selected_profile_id: str = ""):
        self.QtCore, self.QtGui, self.QtWidgets = qt_modules
        QtCore, QtWidgets = self.QtCore, self.QtWidgets
        self.dialog = QtWidgets.QDialog(parent)
        self.dialog.setWindowTitle("远程训练服务器")
        self.dialog.setModal(True)
        self.dialog.resize(740, 820)
        self.dialog.setMinimumSize(660, 740)
        # F37: 150% DPI screens can be smaller than the layout minimum.
        clamp_window_to_screen(self.dialog, (660, 740))
        self._results: Queue[tuple[str, Any]] = Queue()
        self._testing = False

        root = QtWidgets.QVBoxLayout(self.dialog)
        root.setContentsMargins(22, 20, 22, 18)
        root.setSpacing(14)

        title = QtWidgets.QLabel("远程训练服务器")
        title.setObjectName("title")
        subtitle = QtWidgets.QLabel(
            "保存的密码使用 Windows 凭据管理器，绝不写入项目配置或日志。"
        )
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(subtitle)

        profile_row = QtWidgets.QHBoxLayout()
        profile_row.addWidget(QtWidgets.QLabel("服务器配置"))
        self.profile_combo = QtWidgets.QComboBox()
        self.profile_combo.setMinimumWidth(280)
        self.profile_combo.currentIndexChanged.connect(self._load_selected)
        profile_row.addWidget(self.profile_combo, 1)
        self.new_button = QtWidgets.QPushButton("新建")
        self.new_button.clicked.connect(self._new_profile)
        profile_row.addWidget(self.new_button)
        self.delete_button = QtWidgets.QPushButton("删除")
        self.delete_button.clicked.connect(self._delete_profile)
        profile_row.addWidget(self.delete_button)
        root.addLayout(profile_row)

        form_surface = QtWidgets.QFrame()
        form_surface.setObjectName("surface")
        form = QtWidgets.QGridLayout(form_surface)
        form.setContentsMargins(18, 16, 18, 16)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(11)
        form.setColumnMinimumWidth(0, 142)
        form.setColumnStretch(1, 1)

        self.name_edit = QtWidgets.QLineEdit()
        self.host_edit = QtWidgets.QLineEdit()
        self.port_spin = QtWidgets.QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(22)
        self.username_edit = QtWidgets.QLineEdit()
        self.auth_combo = QtWidgets.QComboBox()
        self.auth_combo.addItem("密码", "password")
        self.auth_combo.addItem("SSH 私钥", "key")
        self.auth_combo.currentIndexChanged.connect(self._refresh_auth)
        self.password_edit = QtWidgets.QLineEdit()
        self.password_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        self.password_edit.setPlaceholderText(
            "再次输入可替换已保存的密码"
        )
        self.remember_check = QtWidgets.QCheckBox(
            "在本机安全地记住密码"
        )
        self.remember_check.setChecked(True)
        self.key_edit = QtWidgets.QLineEdit()
        self.key_edit.setPlaceholderText("私钥路径")
        self.key_button = QtWidgets.QPushButton("浏览...")
        self.key_button.clicked.connect(self._browse_key)
        key_row = QtWidgets.QHBoxLayout()
        key_row.setContentsMargins(0, 0, 0, 0)
        key_row.addWidget(self.key_edit, 1)
        key_row.addWidget(self.key_button)
        self.key_widget = QtWidgets.QWidget()
        self.key_widget.setLayout(key_row)
        self.remote_root_edit = QtWidgets.QLineEdit(DEFAULT_REMOTE_ROOT)
        self.remote_root_edit.setToolTip(
            "相对路径在 SSH 用户的 home 目录内解析。"
        )
        self.image_edit = QtWidgets.QLineEdit(DEFAULT_IMAGE)
        self.gpu_combo = QtWidgets.QComboBox()
        self.gpu_combo.setEditable(True)
        self.gpu_combo.addItem("自动（默认 GPU）", DEFAULT_GPU_DEVICE)
        self.gpu_combo.setToolTip(
            "为每个训练作业选择一块 GPU。自动模式保持单一共享队列；填数字编号或 GPU UUID 会为该设备建立独立队列。"
        )
        self.cache_check = QtWidgets.QCheckBox(
            "复用未变化的已上传训练数据"
        )
        self.cache_check.setChecked(True)
        self.cache_check.setToolTip(
            "按内容指纹缓存原始网格的图像和标签归档。数据变化后作为新缓存条目上传。"
        )
        self.runtime_combo = QtWidgets.QComboBox()
        self.runtime_combo.addItem("Docker（默认）", "docker")
        self.runtime_combo.addItem("nerdctl", "nerdctl")
        self.runtime_combo.setToolTip(
            "服务器上的容器运行时。默认 Docker；仅当服务器用 containerd 而无 Docker 时选择 nerdctl。"
        )
        self.namespace_edit = QtWidgets.QLineEdit()
        self.namespace_edit.setPlaceholderText("可选")
        self.namespace_edit.setToolTip(
            "作业容器的 nerdctl namespace（例如管理员配置的 containerd namespace）。Docker 忽略此项。"
        )
        self.code_verify_combo = QtWidgets.QComboBox()
        for label, value in (
            ("发现不一致时警告（默认）", "warn"),
            ("发现不一致时拒绝", "strict"),
            ("跳过校验", "off"),
        ):
            self.code_verify_combo.addItem(label, value)
        self.code_verify_combo.setToolTip(
            "发往服务器的代码与本机副本必须匹配的严格程度。校验能发现过期或被篡改的运行时镜像。"
        )
        self.weights_verify_combo = QtWidgets.QComboBox()
        for label, value in (
            ("不匹配时拒绝（默认）", "strict"),
            ("不匹配时警告", "warn"),
            ("跳过校验", "off"),
        ):
            self.weights_verify_combo.addItem(label, value)
        self.weights_verify_combo.setToolTip(
            "训练权重下载后与记录的校验和核对的严格程度。"
        )
        self.gpu_busy_combo = QtWidgets.QComboBox()
        for label, value in (
            ("忙时等待（默认）", "block"),
            ("忙时警告并继续", "warn"),
            ("跳过检查", "off"),
        ):
            self.gpu_busy_combo.addItem(label, value)
        self.gpu_busy_combo.setToolTip(
            "共享服务器上所选 GPU 看起来已被占用时的处理方式。“阻止”拒绝开始训练；“警告”照常训练并记录备注。"
        )
        self.retention_spin = QtWidgets.QSpinBox()
        self.retention_spin.setRange(1, 3650)
        self.retention_spin.setValue(30)
        self.retention_spin.setSuffix(" 天")
        self.retention_spin.setToolTip(
            "已完成的作业和缓存训练数据在服务器上保留多久后自动清理。"
        )

        rows = [
            ("配置名称", self.name_edit),
            ("主机或 IP", self.host_edit),
            ("SSH 端口", self.port_spin),
            ("用户名", self.username_edit),
            ("认证方式", self.auth_combo),
            ("密码", self.password_edit),
            ("", self.remember_check),
            ("私钥", self.key_widget),
            ("远程工作目录", self.remote_root_edit),
            ("运行时镜像", self.image_edit),
            ("GPU 设备", self.gpu_combo),
            ("数据传输", self.cache_check),
            ("容器运行时", self.runtime_combo),
            ("nerdctl namespace", self.namespace_edit),
            ("代码校验", self.code_verify_combo),
            ("权重校验", self.weights_verify_combo),
            ("GPU 忙碌策略", self.gpu_busy_combo),
            ("服务器清理期限", self.retention_spin),
        ]
        self.password_label = None
        self.key_label = None
        advanced = QtWidgets.QGroupBox("运行环境与维护")
        advanced_form = QtWidgets.QGridLayout(advanced)
        advanced_form.setContentsMargins(14, 16, 14, 12)
        advanced_form.setHorizontalSpacing(12)
        advanced_form.setVerticalSpacing(10)
        advanced_form.setColumnStretch(1, 1)
        advanced_widgets = (
            self.remote_root_edit, self.image_edit, self.runtime_combo,
            self.namespace_edit, self.code_verify_combo, self.weights_verify_combo,
            self.gpu_busy_combo, self.retention_spin,
        )
        basic_row = advanced_row = 0
        for index, (label, widget) in enumerate(rows):
            label_widget = QtWidgets.QLabel(label)
            if widget is self.password_edit:
                self.password_label = label_widget
            elif widget is self.key_widget:
                self.key_label = label_widget
            elif not label:
                label_widget.setVisible(False)
            if widget in advanced_widgets:
                advanced_form.addWidget(label_widget, advanced_row, 0)
                advanced_form.addWidget(widget, advanced_row, 1)
                advanced_row += 1
            else:
                form.addWidget(label_widget, basic_row, 0)
                form.addWidget(widget, basic_row, 1)
                basic_row += 1
        self.advanced_toggle = QtWidgets.QToolButton()
        self.advanced_toggle.setText("高级运行设置")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.advanced_toggle.setArrowType(QtCore.Qt.RightArrow)
        advanced.hide()

        def toggle_advanced(checked):
            advanced.setVisible(checked)
            self.advanced_toggle.setArrowType(QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow)

        self.advanced_toggle.toggled.connect(toggle_advanced)
        form.addWidget(self.advanced_toggle, basic_row, 0, 1, 2)
        form.addWidget(advanced, basic_row + 1, 0, 1, 2)
        root.addWidget(scrollable_widget(form_surface), 1)

        self.status_label = QtWidgets.QLabel(
            "保存配置后，测试 SSH、Docker、GPU、存储和镜像环境。"
        )
        self.status_label.setObjectName("hint")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        actions = QtWidgets.QHBoxLayout()
        self.test_button = QtWidgets.QPushButton("测试连接")
        self.test_button.clicked.connect(self._test_connection)
        actions.addWidget(self.test_button)
        # R72-B2: visibility into what is already on the server. Identical
        # data skips the upload automatically (content-addressed cache);
        # this dialog shows what that cache holds, so a handing-over
        # annotator can confirm their data is already there.
        self.cache_button = QtWidgets.QPushButton("查看已缓存数据集...")
        self.cache_button.clicked.connect(self._show_cached_datasets)
        actions.addWidget(self.cache_button)
        actions.addStretch(1)
        close_button = QtWidgets.QPushButton("关闭")
        close_button.clicked.connect(self.dialog.accept)
        actions.addWidget(close_button)
        self.save_button = QtWidgets.QPushButton("保存")
        self.save_button.setObjectName("primary")
        self.save_button.clicked.connect(self._save)
        actions.addWidget(self.save_button)
        root.addLayout(actions)

        self.timer = QtCore.QTimer(self.dialog)
        self.timer.timeout.connect(self._poll_result)
        self.timer.start(100)
        self._reload_profiles(selected_profile_id)
        self._refresh_auth()

    def _set_status(self, text, error=False):
        """One place for status coloring: a failed save or connection
        test must not read like a normal hint."""
        self.status_label.setObjectName("errorLabel" if error else "hint")
        self.status_label.setStyleSheet(
            "color: #b42318; font-weight: 600;" if error else ""
        )
        self.status_label.setText(text)

    def exec(self) -> int:
        return int(self.dialog.exec())

    def _reload_profiles(self, selected_profile_id: str = "") -> None:
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        self.profile_combo.addItem("新建服务器", "")
        selected_index = 0
        for profile in load_profiles():
            self.profile_combo.addItem(profile["name"], profile["profile_id"])
            if profile["profile_id"] == selected_profile_id:
                selected_index = self.profile_combo.count() - 1
        self.profile_combo.setCurrentIndex(selected_index)
        self.profile_combo.blockSignals(False)
        self._load_selected()

    def _load_selected(self) -> None:
        profile_id = str(self.profile_combo.currentData() or "")
        profile = next(
            (
                row
                for row in load_profiles()
                if row.get("profile_id") == profile_id
            ),
            None,
        )
        if profile is None:
            self._clear_form()
            return
        self.name_edit.setText(profile["name"])
        self.host_edit.setText(profile["host"])
        self.port_spin.setValue(int(profile["port"]))
        self.username_edit.setText(profile["username"])
        index = self.auth_combo.findData(profile["auth_method"])
        self.auth_combo.setCurrentIndex(max(0, index))
        self.key_edit.setText(profile.get("key_path") or "")
        self.remote_root_edit.setText(profile.get("remote_root") or DEFAULT_REMOTE_ROOT)
        self.image_edit.setText(profile.get("runtime_image") or DEFAULT_IMAGE)
        self._set_gpu_device(profile.get("gpu_device") or DEFAULT_GPU_DEVICE)
        self.cache_check.setChecked(bool(profile.get("cache_training_data", True)))
        self._set_combo(self.runtime_combo, profile.get("container_runtime"), "docker")
        self.namespace_edit.setText(profile.get("container_namespace") or "")
        self._set_combo(
            self.code_verify_combo, profile.get("remote_code_verify"), "warn"
        )
        self._set_combo(
            self.weights_verify_combo,
            profile.get("remote_weights_verify"),
            "strict",
        )
        self._set_combo(
            self.gpu_busy_combo, profile.get("gpu_busy_policy"), "block"
        )
        try:
            retention = int(profile.get("remote_cache_retention_days") or 30)
        except Exception:
            retention = 30
        self.retention_spin.setValue(min(3650, max(1, retention)))
        self.password_edit.clear()
        self.status_label.setText(
            "配置已载入。修改服务器设置后请重新测试连接。"
        )
        self._refresh_auth()

    def _clear_form(self) -> None:
        self.name_edit.clear()
        self.host_edit.clear()
        self.port_spin.setValue(22)
        self.username_edit.clear()
        self.auth_combo.setCurrentIndex(0)
        self.password_edit.clear()
        self.key_edit.clear()
        self.remote_root_edit.setText(DEFAULT_REMOTE_ROOT)
        self.image_edit.setText(DEFAULT_IMAGE)
        self._set_gpu_device(DEFAULT_GPU_DEVICE)
        self.cache_check.setChecked(True)
        self._set_combo(self.runtime_combo, "docker", "docker")
        self.namespace_edit.clear()
        self._set_combo(self.code_verify_combo, "warn", "warn")
        self._set_combo(self.weights_verify_combo, "strict", "strict")
        self._set_combo(self.gpu_busy_combo, "block", "block")
        self.retention_spin.setValue(30)
        self.status_label.setText("请填写远程服务器连接信息。")
        self._refresh_auth()

    def _new_profile(self) -> None:
        self.profile_combo.setCurrentIndex(0)
        self._clear_form()
        self.name_edit.setFocus()

    def _delete_profile(self) -> None:
        profile_id = str(self.profile_combo.currentData() or "")
        if not profile_id:
            return
        answer = self.QtWidgets.QMessageBox.question(
            self.dialog,
            "删除服务器配置",
            "删除该服务器配置及其保存的密码？",
        )
        if answer != self.QtWidgets.QMessageBox.Yes:
            return
        delete_profile(profile_id)
        self._reload_profiles()

    def _browse_key(self) -> None:
        choose_open_file_async(
            self.QtCore,
            self.QtWidgets,
            self.dialog,
            "选择 SSH 私钥文件",
            self.key_edit.text() or str(Path.home() / ".ssh"),
            "SSH 私钥文件 (*)",
            lambda path: self.key_edit.setText(str(path)) if path else None,
            button=self.key_button,
        )

    def _refresh_auth(self) -> None:
        password_mode = self.auth_combo.currentData() == "password"
        self.password_edit.setVisible(password_mode)
        self.remember_check.setVisible(password_mode)
        if self.password_label is not None:
            self.password_label.setVisible(password_mode)
        self.key_widget.setVisible(not password_mode)
        if self.key_label is not None:
            self.key_label.setVisible(not password_mode)

    def _gpu_device(self) -> str:
        index = self.gpu_combo.currentIndex()
        data = self.gpu_combo.currentData()
        if (
            index >= 0
            and data is not None
            and self.gpu_combo.currentText() == self.gpu_combo.itemText(index)
        ):
            return str(data)
        text = self.gpu_combo.currentText().strip()
        match = text.split(" ", 1)[0]
        return match or DEFAULT_GPU_DEVICE

    @staticmethod
    def _set_combo(combo, value, default):
        index = combo.findData(str(value or default))
        combo.setCurrentIndex(max(0, index))

    def _set_gpu_device(self, value: object) -> None:
        wanted = str(value or DEFAULT_GPU_DEVICE)
        index = self.gpu_combo.findData(wanted)
        if index >= 0:
            self.gpu_combo.setCurrentIndex(index)
        else:
            self.gpu_combo.setEditText(wanted)

    def _set_detected_gpus(self, rows: list[dict[str, Any]]) -> None:
        selected = self._gpu_device()
        self.gpu_combo.blockSignals(True)
        self.gpu_combo.clear()
        self.gpu_combo.addItem(
            "自动（默认 GPU）", DEFAULT_GPU_DEVICE
        )
        for row in rows:
            index = str(row.get("index") or "")
            if not index:
                continue
            memory_gb = float(row.get("memory_mb") or 0) / 1024.0
            utilization = row.get("utilization_percent")
            load = (
                " · 已用 {}%".format(utilization)
                if utilization is not None
                else ""
            )
            self.gpu_combo.addItem(
                "GPU {} · {} · {:.1f} GB{}".format(
                    index, row.get("name") or "NVIDIA GPU", memory_gb, load
                ),
                index,
            )
        self.gpu_combo.blockSignals(False)
        self._set_gpu_device(selected)

    def _profile_values(self) -> dict[str, Any]:
        current_id = str(self.profile_combo.currentData() or "")
        return normalize_profile(
            {
                "profile_id": current_id,
                "name": self.name_edit.text(),
                "host": self.host_edit.text(),
                "port": self.port_spin.value(),
                "username": self.username_edit.text(),
                "auth_method": self.auth_combo.currentData(),
                "key_path": self.key_edit.text(),
                "remote_root": self.remote_root_edit.text(),
                "runtime_image": self.image_edit.text(),
                "gpu_device": self._gpu_device(),
                "cache_training_data": self.cache_check.isChecked(),
                "container_runtime": self.runtime_combo.currentData(),
                "container_namespace": self.namespace_edit.text(),
                "remote_code_verify": self.code_verify_combo.currentData(),
                "remote_weights_verify": self.weights_verify_combo.currentData(),
                "gpu_busy_policy": self.gpu_busy_combo.currentData(),
                "remote_cache_retention_days": self.retention_spin.value(),
            }
        )

    def _save(self, *, show_success: bool = True) -> dict[str, Any] | None:
        try:
            profile = save_profile(self._profile_values())
            if profile["auth_method"] == "password":
                password = self.password_edit.text()
                if password:
                    store_password(
                        profile["profile_id"],
                        profile["username"],
                        password,
                        remember=self.remember_check.isChecked(),
                    )
            self._reload_profiles(profile["profile_id"])
            if show_success:
                self.status_label.setText(
                    "服务器配置已保存。默认仍使用本机训练。"
                )
            return profile
        except Exception as exc:
            self._set_status("无法保存服务器配置：{}".format(exc), error=True)
            return None

    def _test_connection(self, trust_unknown: bool = False) -> None:
        if self._testing:
            return
        profile = self._save(show_success=False)
        if profile is None:
            return
        if profile["auth_method"] == "password" and not self.password_edit.text():
            # A previously stored password may still be valid; the worker will
            # report a clear error when none is available.
            pass
        self._testing = True
        self.test_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.status_label.setText("正在测试 SSH、Docker、GPU、存储和运行时镜像...")

        def worker() -> None:
            try:
                result = test_connection(
                    profile, trust_unknown=trust_unknown
                )
                self._results.put(("success", result))
            except UnknownHostKeyError as exc:
                self._results.put(("unknown_host", exc))
            except Exception as exc:
                self._results.put(("error", exc))

        thread = threading.Thread(target=worker, name="remote-server-test")
        thread.daemon = True
        thread.start()

    def _poll_result(self) -> None:
        try:
            kind, result = self._results.get_nowait()
        except Empty:
            return
        if kind == "cache":
            self._show_cache_rows(result)
            return
        if kind == "cache_error":
            self._show_cache_rows([], error=result)
            return
        self._testing = False
        self.test_button.setEnabled(True)
        self.save_button.setEnabled(True)
        if kind == "unknown_host":
            answer = self.QtWidgets.QMessageBox.question(
                self.dialog,
                "信任 SSH 服务器",
                (
                    "该服务器尚未使用过。\n\n主机：{}:{}\n指纹：{}\n\n信任该主机密钥？"
                ).format(result.host, result.port, result.fingerprint),
            )
            if answer == self.QtWidgets.QMessageBox.Yes:
                self._test_connection(trust_unknown=True)
            else:
                self.status_label.setText(
                    "连接已停止：SSH 主机密钥未受信任。"
                )
            return
        if kind == "error":
            prefix = (
                "主机身份校验失败"
                if isinstance(result, HostKeyChangedError)
                else "连接失败"
            )
            self._set_status("{}：{}".format(prefix, result), error=True)
            return
        gpu_rows = result.get("gpus") or []
        self._set_detected_gpus(gpu_rows)
        gpus = ", ".join(
            "GPU {} {} ({}% used)".format(
                row.get("index"),
                row.get("name"),
                row.get("utilization_percent") or 0,
            )
            for row in gpu_rows
        ) or "服务器未报告 GPU"
        message = (
            "已连接 · 工作目录 {} · {} · 选用 {} · 剩余 {} · "
            "镜像就绪 · {}".format(
                result.get("remote_root") or "?",
                gpus,
                result.get("gpu_device") or DEFAULT_GPU_DEVICE,
                _format_bytes(result.get("free_bytes")),
                result.get("fingerprint"),
            )
        )
        warning = str(result.get("warning") or "").strip()
        if warning:
            message = "⚠ {} — {}".format(warning, message)
        self.status_label.setText(message)

    def _show_cached_datasets(self) -> None:
        """List cached remote datasets (R72-B2) in a read-only dialog."""
        QtWidgets = self.QtWidgets
        try:
            profile = self._profile_values()
        except Exception as exc:  # noqa: BLE001 - surfaced in the dialog
            self._set_status(
                "无法读取服务器配置：{}".format(exc), error=True)
            return
        self.cache_button.setEnabled(False)
        self.status_label.setText("正在读取服务器上的数据集缓存...")

        def worker() -> None:
            try:
                from remote_training_controller import (
                    list_remote_dataset_cache,
                )
            except ImportError:
                from tools.remote_training_controller import (
                    list_remote_dataset_cache,
                )
            try:
                self._results.put(
                    ("cache", list_remote_dataset_cache(profile))
                )
            except Exception as exc:  # noqa: BLE001 - surfaced in the dialog
                self._results.put(("cache_error", exc))

        thread = threading.Thread(
            target=worker, name="remote-dataset-cache-list"
        )
        thread.daemon = True
        thread.start()

    def _show_cache_rows(self, rows, error=None) -> None:
        QtWidgets = self.QtWidgets
        self.cache_button.setEnabled(True)
        if error is not None:
            self._set_status(
                "读取数据集缓存失败：{}".format(error), error=True)
            return
        self.status_label.setText(
            "服务器上已缓存 {} 个数据集（相同数据再次提交会自动跳过上传）。"
            "列表见弹窗。".format(len(rows))
        )
        dialog = QtWidgets.QDialog(self.dialog)
        dialog.setWindowTitle("已缓存的数据集")
        dialog.resize(720, 420)
        layout = QtWidgets.QVBoxLayout(dialog)
        hint = QtWidgets.QLabel(
            "内容相同的数据再次提交会自动命中并跳过上传（按内容指纹比对，"
            "不同数据不会被误用）。缓存 30 天未使用后自动清理。"
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        table = QtWidgets.QTableWidget(0, 4)
        table.setHorizontalHeaderLabels(
            ["内容指纹", "大小", "最近使用", "缓存"]
        )
        table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        for row in rows:
            index = table.rowCount()
            table.insertRow(index)
            mtime = time.strftime(
                "%Y-%m-%d %H:%M", time.localtime(float(row.get("mtime_epoch") or 0))
            )
            for column, value in enumerate(
                (
                    str(row.get("fingerprint") or ""),
                    _format_bytes(row.get("size_bytes")),
                    mtime,
                    "本账号" if row.get("bucket") == "own" else "共享",
                )
            ):
                table.setItem(index, column, QtWidgets.QTableWidgetItem(value))
        layout.addWidget(table, 1)
        close = QtWidgets.QPushButton("关闭")
        close.clicked.connect(dialog.accept)
        layout.addWidget(close, 0, self.QtCore.Qt.AlignRight)
        dialog.exec()


class RemoteComputeSelector:
    """Small additive selector; it defaults to local on every window open."""

    def __init__(self, parent, qt_modules):
        self.parent = parent
        self.QtCore, self.QtGui, self.QtWidgets = qt_modules
        QtWidgets = self.QtWidgets
        self.group = QtWidgets.QGroupBox("计算")
        layout = QtWidgets.QVBoxLayout(self.group)
        layout.setContentsMargins(14, 12, 14, 12)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("训练运行于"))
        self.combo = QtWidgets.QComboBox()
        self.combo.currentIndexChanged.connect(self._refresh_hint)
        row.addWidget(self.combo, 1)
        self.manage_button = QtWidgets.QPushButton("管理服务器...")
        self.manage_button.clicked.connect(self._manage)
        row.addWidget(self.manage_button)
        layout.addLayout(row)
        self.hint = QtWidgets.QLabel()
        self.hint.setObjectName("hint")
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)
        self.refresh()

    def refresh(self, selected_profile_id: str = "") -> None:
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItem("本机", ("local", ""))
        selected_index = 0
        for profile in load_profiles():
            self.combo.addItem(
                "远程 · {}".format(profile["name"]),
                ("remote", profile["profile_id"]),
            )
            if profile["profile_id"] == selected_profile_id:
                selected_index = self.combo.count() - 1
        self.combo.setCurrentIndex(selected_index)
        self.combo.blockSignals(False)
        self._refresh_hint()

    def _manage(self) -> None:
        _backend, profile_id = self.selection()
        dialog = ServerProfilesDialog(
            self.parent,
            (self.QtCore, self.QtGui, self.QtWidgets),
            profile_id,
        )
        dialog.exec()
        self.refresh(profile_id)

    def _refresh_hint(self) -> None:
        backend, profile_id = self.selection()
        if backend == "local":
            self.hint.setText(
                "使用现有本机训练流程。不使用任何远程包或连接。"
            )
            return
        profile = next(
            (
                row
                for row in load_profiles()
                if row.get("profile_id") == profile_id
            ),
            None,
        )
        if profile:
            self.hint.setText(
                "{}@{}:{} · {} · GPU {}。数据传输与训练均在 Mimics 之外进行。".format(
                    profile["username"],
                    profile["host"],
                    profile["port"],
                    profile["runtime_image"],
                    profile.get("gpu_device") or DEFAULT_GPU_DEVICE,
                )
            )
        else:
            self.hint.setText("所选远程服务器配置不可用。")

    def selection(self) -> tuple[str, str]:
        value = self.combo.currentData()
        if isinstance(value, tuple) and len(value) == 2:
            return str(value[0]), str(value[1])
        if isinstance(value, list) and len(value) == 2:
            return str(value[0]), str(value[1])
        return "local", ""
