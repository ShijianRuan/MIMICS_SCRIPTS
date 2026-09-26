#!/usr/bin/env python3
"""Shared PySide6 controls for optional remote training servers."""

from __future__ import annotations

import os
import threading
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
    from ui_theme import choose_open_file_async
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
    from tools.ui_theme import choose_open_file_async


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
        self.dialog.setWindowTitle("Remote Training Servers")
        self.dialog.setModal(True)
        self.dialog.resize(740, 820)
        self.dialog.setMinimumSize(660, 740)
        self._results: Queue[tuple[str, Any]] = Queue()
        self._testing = False

        root = QtWidgets.QVBoxLayout(self.dialog)
        root.setContentsMargins(22, 20, 22, 18)
        root.setSpacing(14)

        title = QtWidgets.QLabel("Remote Training Servers")
        title.setObjectName("title")
        subtitle = QtWidgets.QLabel(
            "Saved passwords use Windows Credential Manager and are never "
            "written to project configuration or logs."
        )
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(subtitle)

        profile_row = QtWidgets.QHBoxLayout()
        profile_row.addWidget(QtWidgets.QLabel("Server profile"))
        self.profile_combo = QtWidgets.QComboBox()
        self.profile_combo.setMinimumWidth(280)
        self.profile_combo.currentIndexChanged.connect(self._load_selected)
        profile_row.addWidget(self.profile_combo, 1)
        self.new_button = QtWidgets.QPushButton("New")
        self.new_button.clicked.connect(self._new_profile)
        profile_row.addWidget(self.new_button)
        self.delete_button = QtWidgets.QPushButton("Delete")
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
        self.auth_combo.addItem("Password", "password")
        self.auth_combo.addItem("SSH private key", "key")
        self.auth_combo.currentIndexChanged.connect(self._refresh_auth)
        self.password_edit = QtWidgets.QLineEdit()
        self.password_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        self.password_edit.setPlaceholderText(
            "Enter again to replace the stored password"
        )
        self.remember_check = QtWidgets.QCheckBox(
            "Remember securely on this computer"
        )
        self.remember_check.setChecked(True)
        self.key_edit = QtWidgets.QLineEdit()
        self.key_edit.setPlaceholderText("Private key path")
        self.key_button = QtWidgets.QPushButton("Browse...")
        self.key_button.clicked.connect(self._browse_key)
        key_row = QtWidgets.QHBoxLayout()
        key_row.setContentsMargins(0, 0, 0, 0)
        key_row.addWidget(self.key_edit, 1)
        key_row.addWidget(self.key_button)
        self.key_widget = QtWidgets.QWidget()
        self.key_widget.setLayout(key_row)
        self.remote_root_edit = QtWidgets.QLineEdit(DEFAULT_REMOTE_ROOT)
        self.remote_root_edit.setToolTip(
            "Relative paths are resolved inside the SSH user's home directory."
        )
        self.image_edit = QtWidgets.QLineEdit(DEFAULT_IMAGE)
        self.gpu_combo = QtWidgets.QComboBox()
        self.gpu_combo.setEditable(True)
        self.gpu_combo.addItem("Automatic (default GPU)", DEFAULT_GPU_DEVICE)
        self.gpu_combo.setToolTip(
            "Choose one GPU for each training job. Automatic keeps the "
            "single shared queue; a numeric index or GPU UUID creates a "
            "separate queue for that device."
        )
        self.cache_check = QtWidgets.QCheckBox(
            "Reuse unchanged uploaded training data"
        )
        self.cache_check.setChecked(True)
        self.cache_check.setToolTip(
            "Caches source-grid image and label archives by content fingerprint. "
            "Changed data is uploaded as a new cache entry."
        )
        self.runtime_combo = QtWidgets.QComboBox()
        self.runtime_combo.addItem("Docker (default)", "docker")
        self.runtime_combo.addItem("nerdctl", "nerdctl")
        self.runtime_combo.setToolTip(
            "Container runtime on the server. Docker is the default; choose "
            "nerdctl only when the server runs containerd without Docker."
        )
        self.namespace_edit = QtWidgets.QLineEdit()
        self.namespace_edit.setPlaceholderText("optional")
        self.namespace_edit.setToolTip(
            "nerdctl namespace for job containers (for example the containerd "
            "namespace the admin configured). Ignored by Docker."
        )
        self.code_verify_combo = QtWidgets.QComboBox()
        for label, value in (
            ("Warn on drift (default)", "warn"),
            ("Refuse on drift", "strict"),
            ("Skip verification", "off"),
        ):
            self.code_verify_combo.addItem(label, value)
        self.code_verify_combo.setToolTip(
            "How strictly the code shipped to the server must match this "
            "workstation's copy. Verification catches a stale or tampered "
            "runtime image."
        )
        self.weights_verify_combo = QtWidgets.QComboBox()
        for label, value in (
            ("Refuse on mismatch (default)", "strict"),
            ("Warn on mismatch", "warn"),
            ("Skip verification", "off"),
        ):
            self.weights_verify_combo.addItem(label, value)
        self.weights_verify_combo.setToolTip(
            "How strictly trained weights are checked against their "
            "recorded checksum after download."
        )
        self.retention_spin = QtWidgets.QSpinBox()
        self.retention_spin.setRange(1, 3650)
        self.retention_spin.setValue(30)
        self.retention_spin.setSuffix(" days")
        self.retention_spin.setToolTip(
            "How long finished jobs and cached training data stay on the "
            "server before automatic cleanup."
        )

        rows = [
            ("Profile name", self.name_edit),
            ("Host or IP", self.host_edit),
            ("SSH port", self.port_spin),
            ("Username", self.username_edit),
            ("Authentication", self.auth_combo),
            ("Password", self.password_edit),
            ("", self.remember_check),
            ("Private key", self.key_widget),
            ("Remote work folder", self.remote_root_edit),
            ("Runtime image", self.image_edit),
            ("GPU device", self.gpu_combo),
            ("Data transfer", self.cache_check),
            ("Container runtime", self.runtime_combo),
            ("nerdctl namespace", self.namespace_edit),
            ("Code verification", self.code_verify_combo),
            ("Weights verification", self.weights_verify_combo),
            ("Server cleanup after", self.retention_spin),
        ]
        self.password_label = None
        self.key_label = None
        for index, (label, widget) in enumerate(rows):
            label_widget = QtWidgets.QLabel(label)
            if widget is self.password_edit:
                self.password_label = label_widget
            elif widget is self.key_widget:
                self.key_label = label_widget
            elif not label:
                label_widget.setVisible(False)
            form.addWidget(label_widget, index, 0)
            form.addWidget(widget, index, 1)
        root.addWidget(form_surface)

        self.status_label = QtWidgets.QLabel(
            "Save the profile, then test the SSH, Docker, GPU, storage, and image setup."
        )
        self.status_label.setObjectName("hint")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        actions = QtWidgets.QHBoxLayout()
        self.test_button = QtWidgets.QPushButton("Test Connection")
        self.test_button.clicked.connect(self._test_connection)
        actions.addWidget(self.test_button)
        actions.addStretch(1)
        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(self.dialog.accept)
        actions.addWidget(close_button)
        self.save_button = QtWidgets.QPushButton("Save")
        self.save_button.setObjectName("primary")
        self.save_button.clicked.connect(self._save)
        actions.addWidget(self.save_button)
        root.addLayout(actions)

        self.timer = QtCore.QTimer(self.dialog)
        self.timer.timeout.connect(self._poll_result)
        self.timer.start(100)
        self._reload_profiles(selected_profile_id)
        self._refresh_auth()

    def exec(self) -> int:
        return int(self.dialog.exec())

    def _reload_profiles(self, selected_profile_id: str = "") -> None:
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        self.profile_combo.addItem("New server", "")
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
        try:
            retention = int(profile.get("remote_cache_retention_days") or 30)
        except Exception:
            retention = 30
        self.retention_spin.setValue(min(3650, max(1, retention)))
        self.password_edit.clear()
        self.status_label.setText(
            "Profile loaded. Test the connection after changing server settings."
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
        self.retention_spin.setValue(30)
        self.status_label.setText("Enter the remote server connection details.")
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
            "Delete Server Profile",
            "Delete this server profile and its stored password?",
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
            "Choose SSH Private Key",
            self.key_edit.text() or str(Path.home() / ".ssh"),
            "SSH private keys (*)",
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
            "Automatic (default GPU)", DEFAULT_GPU_DEVICE
        )
        for row in rows:
            index = str(row.get("index") or "")
            if not index:
                continue
            memory_gb = float(row.get("memory_mb") or 0) / 1024.0
            self.gpu_combo.addItem(
                "GPU {} · {} · {:.1f} GB".format(
                    index, row.get("name") or "NVIDIA GPU", memory_gb
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
                    "Server profile saved. Local training remains the default."
                )
            return profile
        except Exception as exc:
            self.status_label.setText("Could not save server: {}".format(exc))
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
        self.status_label.setText("Testing SSH, Docker, GPU, storage, and runtime image...")

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
        self._testing = False
        self.test_button.setEnabled(True)
        self.save_button.setEnabled(True)
        if kind == "unknown_host":
            answer = self.QtWidgets.QMessageBox.question(
                self.dialog,
                "Trust SSH Server",
                (
                    "This server has not been used before.\n\n"
                    "Host: {}:{}\n"
                    "Fingerprint: {}\n\n"
                    "Trust this host key?"
                ).format(result.host, result.port, result.fingerprint),
            )
            if answer == self.QtWidgets.QMessageBox.Yes:
                self._test_connection(trust_unknown=True)
            else:
                self.status_label.setText(
                    "Connection stopped because the SSH host key was not trusted."
                )
            return
        if kind == "error":
            prefix = (
                "Host identity check failed"
                if isinstance(result, HostKeyChangedError)
                else "Connection failed"
            )
            self.status_label.setText("{}: {}".format(prefix, result))
            return
        gpu_rows = result.get("gpus") or []
        self._set_detected_gpus(gpu_rows)
        gpus = ", ".join(
            "GPU {} {}".format(row.get("index"), row.get("name"))
            for row in gpu_rows
        ) or "No GPU reported"
        message = (
            "Connected · work folder {} · {} · selected {} · {} free · "
            "image ready · {}".format(
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


class RemoteComputeSelector:
    """Small additive selector; it defaults to local on every window open."""

    def __init__(self, parent, qt_modules):
        self.parent = parent
        self.QtCore, self.QtGui, self.QtWidgets = qt_modules
        QtWidgets = self.QtWidgets
        self.group = QtWidgets.QGroupBox("Compute")
        layout = QtWidgets.QVBoxLayout(self.group)
        layout.setContentsMargins(14, 12, 14, 12)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Run training on"))
        self.combo = QtWidgets.QComboBox()
        self.combo.currentIndexChanged.connect(self._refresh_hint)
        row.addWidget(self.combo, 1)
        self.manage_button = QtWidgets.QPushButton("Manage Servers...")
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
        self.combo.addItem("This workstation", ("local", ""))
        selected_index = 0
        for profile in load_profiles():
            self.combo.addItem(
                "Remote · {}".format(profile["name"]),
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
                "Uses the existing local training path. No remote packages or "
                "connections are used."
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
                "{}@{}:{} · {} · GPU {}. Data transfer and training run outside Mimics.".format(
                    profile["username"],
                    profile["host"],
                    profile["port"],
                    profile["runtime_image"],
                    profile.get("gpu_device") or DEFAULT_GPU_DEVICE,
                )
            )
        else:
            self.hint.setText("The selected remote server profile is unavailable.")

    def selection(self) -> tuple[str, str]:
        value = self.combo.currentData()
        if isinstance(value, tuple) and len(value) == 2:
            return str(value[0]), str(value[1])
        if isinstance(value, list) and len(value) == 2:
            return str(value[0]), str(value[1])
        return "local", ""
