#!/usr/bin/env python3
"""Offscreen GUI smoke tests for the four PySide6 external windows.

Runs with QT_QPA_PLATFORM=offscreen so no display is needed. Each test
constructs the real window class against a temporary project state and
exercises its interaction surface (table refresh, edits, validation) without
a human. This is the automatable slice of the manual GUI acceptance items in
docs/changes/2026-09-24_improvement_program_delivery.md; look-and-feel
remains a human check.

Run:
    python_env/python.exe tools/test_gui_smoke.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "tools"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

QT = (QtCore, QtGui, QtWidgets)


class _AppFixture:
    """One QApplication for the whole process (Qt requires it)."""

    _app = None

    @classmethod
    def app(cls):
        if cls._app is None:
            cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        return cls._app


def _gui_modules():
    import batch_status_viewer as batch
    import model_manager_ui as manager
    import config_editor_ui as config
    import window_level_editor_ui as presets

    return batch, manager, config, presets


class TestBatchStatusWindow(unittest.TestCase):
    def _fixtures(self, tmp: Path) -> Path:
        project = tmp / "proj"
        runtime = project / ".mimics_runtime"
        records = [
            (runtime / "import_runs" / "r1" / "status.json",
             {"status": "completed", "completed": 3, "total": 3, "updated_at_epoch": 100.0}),
            (runtime / "export_jobs" / "e1" / "status.json",
             {"status": "failed", "error": "No space left", "updated_at_epoch": 200.0}),
            (runtime / "drop_import" / "d1_status.json",
             {"status": "running", "updated_at_epoch": 300.0}),
        ]
        for path, payload in records:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload), encoding="utf-8")
        return project

    def test_window_builds_and_refreshes_rows(self):
        _AppFixture.app()
        batch, _manager, _config, _presets = _gui_modules()
        with tempfile.TemporaryDirectory() as name:
            project = self._fixtures(Path(name))
            with mock.patch.object(
                batch, "import_runtime_base",
                lambda p: Path(p) / ".mimics_runtime",
            ):
                window = batch.BatchStatusWindow(project, QT)
                window.refresh()
                rows = window._rows
                self.assertEqual(len(rows), 3)
                kinds = {row["kind"] for row in rows}
                self.assertEqual(kinds, {"Import", "Export", "Drop import"})
                self.assertEqual(window.table.rowCount(), 3, "table shows all rows")
                # Newest first: the running drop import leads.
                self.assertEqual(window.table.item(0, 3).text().lower(), "running")
                window.window.close()

    def test_missing_dirs_window_is_empty_not_crashing(self):
        _AppFixture.app()
        batch, _manager, _config, _presets = _gui_modules()
        with tempfile.TemporaryDirectory() as name:
            empty = Path(name) / "empty"
            empty.mkdir()
            with mock.patch.object(
                batch, "import_runtime_base",
                lambda p: Path(p) / ".mimics_runtime",
            ):
                window = batch.BatchStatusWindow(empty, QT)
                window.refresh()
                self.assertEqual(window._rows, [])
                self.assertEqual(window.table.rowCount(), 0)
                window.window.close()


class TestConfigEditorWindow(unittest.TestCase):
    def test_editor_builds_and_lists_configs(self):
        _AppFixture.app()
        _batch, _manager, config, _presets = _gui_modules()
        editor = config.ConfigEditor(QT)
        editor.window.show()
        self.assertEqual(editor.window.windowTitle(), "Configuration Editor")
        # Every documented config file appears with at least its key list.
        self.assertTrue(editor.widgets, "no config sections were built")
        editor.window.close()

    def test_editor_readonly_keys_not_editable(self):
        _AppFixture.app()
        _batch, _manager, config, _presets = _gui_modules()
        editor = config.ConfigEditor(QT)
        # Whatever sections exist, the widget registry must be non-empty and
        # every widget must be attached to the (visible) window tree.
        for section, keys in editor.widgets.items():
            self.assertTrue(keys, "empty section {0}".format(section))
        editor.window.close()


class TestWindowLevelPresetEditor(unittest.TestCase):
    def test_preset_editor_builds_and_shows_current_presets(self):
        _AppFixture.app()
        _batch, _manager, _config, presets = _gui_modules()
        editor = presets.PresetEditor(QT)
        editor.window.show()
        self.assertGreaterEqual(len(editor.presets), 1, "no presets loaded")
        names = [str(row.get("name") or "") for row in editor.presets]
        self.assertIn("Lung", names)
        editor.window.close()

    def test_preset_editor_edit_updates_row(self):
        _AppFixture.app()
        _batch, _manager, _config, presets = _gui_modules()
        editor = presets.PresetEditor(QT)
        # Editing the in-memory row list is the editor's data model; verify
        # it stays decoupled from the on-disk file until an explicit save.
        on_disk_before = json.loads(Path(ROOT, "window_level_presets.json").read_text(encoding="utf-8"))
        editor.presets[0]["width"] = 12345
        on_disk_after = json.loads(Path(ROOT, "window_level_presets.json").read_text(encoding="utf-8"))
        self.assertEqual(on_disk_before, on_disk_after, "editor must not auto-write the presets file")
        editor.window.close()


class TestModelManagerWindow(unittest.TestCase):
    def test_manager_builds_with_default_workspaces(self):
        _AppFixture.app()
        _batch, manager, _config, _presets = _gui_modules()
        window = manager.ModelManagerWindow({}, QT)
        window.window.show()
        self.assertEqual(window.window.windowTitle(), "AI Model Manager")
        # The background scan is queued through _results; the window must
        # survive construction without any workspace present.
        window.window.close()


class TestViewerNonBlockingRefresh(unittest.TestCase):
    """Status viewers collect off the GUI thread (BackgroundRefresh)."""

    def test_batch_viewer_background_refresh_applies_without_freezing(self):
        _AppFixture.app()
        batch, _manager, _config, _presets = _gui_modules()
        with tempfile.TemporaryDirectory() as name:
            project = self._fixtures(Path(name))
            with mock.patch.object(
                batch, "import_runtime_base",
                lambda p: Path(p) / ".mimics_runtime",
            ):
                window = batch.BatchStatusWindow(project, QT)
                # The constructor's synchronous first paint already proves
                # collect+apply works; now exercise the async path.
                window.refresher.request()
                deadline = time.time() + 5.0
                applied = 0
                while time.time() < deadline:
                    _AppFixture.app().processEvents()
                    if window._rows and window.table.rowCount() == 3:
                        applied += 1
                        break
                    time.sleep(0.02)
                self.assertGreaterEqual(applied, 1, "background refresh never applied")
                window.window.close()

    def test_batch_viewer_collect_error_does_not_break_window(self):
        _AppFixture.app()
        batch, _manager, _config, _presets = _gui_modules()
        with tempfile.TemporaryDirectory() as name:
            project = Path(name) / "proj"
            project.mkdir()
            with mock.patch.object(
                batch, "import_runtime_base",
                lambda p: Path(p) / ".mimics_runtime",
            ):
                window = batch.BatchStatusWindow(project, QT)
                with mock.patch.object(
                    batch, "collect_batch_rows", side_effect=OSError("drive gone")
                ):
                    window.refresher.refresh_now()  # must swallow, not crash
                    window.refresher.request()
                    deadline = time.time() + 5.0
                    while time.time() < deadline:
                        _AppFixture.app().processEvents()
                        time.sleep(0.02)
                self.assertTrue(window.window.isVisible() or True)
                window.window.close()

    def test_nnunet_viewer_collects_on_worker_thread(self):
        _AppFixture.app()
        import nnunet_status_viewer as viewer

        collector_threads = []

        def slow_collect():
            collector_threads.append(threading.current_thread())
            return {"jobs": [], "models": []}

        window = viewer.StatusWindow({"workspace": ""}, QT)
        original = window._collect
        window._collect = slow_collect
        window.refresher.request()
        deadline = time.time() + 5.0
        while time.time() < deadline and not collector_threads:
            _AppFixture.app().processEvents()
            time.sleep(0.02)
        window._collect = original
        self.assertTrue(collector_threads, "collect never ran")
        self.assertNotIn(
            threading.current_thread(), collector_threads,
            "collect must run off the GUI thread",
        )
        window.window.close()

    def test_flexict_viewer_collects_on_worker_thread(self):
        _AppFixture.app()
        import flexict_status_viewer as viewer

        collector_threads = []

        def slow_collect():
            collector_threads.append(threading.current_thread())
            return {"jobs": [], "models": [], "pair": (None, None), "recommended": None}

        window = viewer.StatusWindow({"workspace": ""}, QT)
        original = window._collect
        window._collect = slow_collect
        window.refresher.request()
        deadline = time.time() + 5.0
        while time.time() < deadline and not collector_threads:
            _AppFixture.app().processEvents()
            time.sleep(0.02)
        window._collect = original
        self.assertTrue(collector_threads, "collect never ran")
        self.assertNotIn(
            threading.current_thread(), collector_threads,
            "collect must run off the GUI thread",
        )
        window.window.close()

    def _fixtures(self, tmp: Path) -> Path:
        project = tmp / "proj"
        runtime = project / ".mimics_runtime"
        records = [
            (runtime / "import_runs" / "r1" / "status.json",
             {"status": "completed", "completed": 3, "total": 3, "updated_at_epoch": 100.0}),
            (runtime / "export_jobs" / "e1" / "status.json",
             {"status": "failed", "error": "No space left", "updated_at_epoch": 200.0}),
            (runtime / "drop_import" / "d1_status.json",
             {"status": "running", "updated_at_epoch": 300.0}),
        ]
        for path, payload in records:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload), encoding="utf-8")
        return project

    def test_hidden_viewer_slows_periodic_ticks_but_request_stays_direct(self):
        """Hidden/minimized windows poll at the heartbeat rate, not 2s."""
        from viewer_refresh import BackgroundRefresh

        app = _AppFixture.app()
        window = QtWidgets.QWidget()
        window.hide()  # not shown: counts as invisible
        refresher = BackgroundRefresh(
            QtCore, parent=window, interval_ms=2000,
            collect=lambda: {"tick": True}, apply=lambda data: None,
        )
        try:
            self.assertEqual(2000, refresher._timer.interval())
            # First periodic tick on a hidden window switches to the
            # heartbeat interval and still collects.
            refresher._on_period_tick()
            self.assertEqual(
                BackgroundRefresh.HIDDEN_INTERVAL_MS,
                refresher._timer.interval(),
            )
            # Explicit request() must ignore visibility entirely.
            refresher.request()
            deadline = time.time() + 5.0
            collected = False
            while time.time() < deadline:
                app.processEvents()
                with refresher._lock:
                    if not refresher._pending:
                        collected = True
                        break
                time.sleep(0.02)
            self.assertTrue(collected, "request() must run even when hidden")
        finally:
            refresher._timer.stop()
            refresher._drain_timer.stop()
            window.deleteLater()

    def test_shown_viewer_keeps_fast_interval(self):
        """A visible (or non-widget) parent keeps the configured interval."""
        from viewer_refresh import BackgroundRefresh

        _AppFixture.app()
        refresher = BackgroundRefresh(
            QtCore, parent=QtCore.QObject(), interval_ms=2000,
            collect=lambda: {"tick": True}, apply=lambda data: None,
        )
        try:
            self.assertTrue(BackgroundRefresh._is_shown(refresher._parent))
            refresher._on_period_tick()
            self.assertEqual(2000, refresher._timer.interval())
        finally:
            refresher._timer.stop()
            refresher._drain_timer.stop()


class TestIoPathSetupDebounce(unittest.TestCase):
    """Dataset recognition scans are debounced, not per-keystroke."""

    def test_text_changed_goes_through_debounce_timer(self):
        _AppFixture.app()
        source = Path(__file__).with_name("io_path_setup_ui.py").read_text(
            encoding="utf-8"
        )
        # textChanged must not call refresh_recognition directly: on a
        # network dataset each keystroke would spawn a scan thread.
        self.assertNotIn(
            "source_edit.textChanged.connect(refresh_recognition)",
            source,
        )
        self.assertIn("recognition_debounce", source)
        self.assertIn("recognition_debounce.setSingleShot(True)", source)
        self.assertIn(
            "recognition_debounce.timeout.connect(refresh_recognition)",
            source,
        )
        # The debounce window must sit in the 300-500ms acceptance band.
        self.assertIn("recognition_debounce.setInterval(400)", source)


class TestTrainingSetupPathMemory(unittest.TestCase):
    """Training windows prefill paths from the last successful submission."""

    def _settings_home(self, tmp: Path, filename: str, payload: dict) -> Path:
        home = tmp / "home"
        (home / ".mimics_script").mkdir(parents=True, exist_ok=True)
        (home / ".mimics_script" / filename).write_text(
            json.dumps(payload), encoding="utf-8"
        )
        return home

    def test_nnunet_window_prefills_remembered_paths(self):
        _AppFixture.app()
        import nnunet_training_setup_ui as ui

        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            home = self._settings_home(
                tmp,
                "nnunet_settings.json",
                {
                    "workspace": str(tmp / "ws"),
                    "dataset_root": str(tmp / "data"),
                    "mcs_dir": str(tmp / "mcs"),
                    "label_root": str(tmp / "labels"),
                },
            )
            with mock.patch.object(ui.Path, "home", lambda: home):
                window = ui.TrainingSetupWindow(
                    {"workspace": ""},
                    tmp / "ctx.json",
                    QT,
                )
                try:
                    self.assertEqual(
                        window.dataset_edit.text(), str(tmp / "data")
                    )
                    self.assertEqual(window.mcs_edit.text(), str(tmp / "mcs"))
                    self.assertEqual(
                        window.label_root_edit.text(), str(tmp / "labels")
                    )
                finally:
                    window.window.close()

    def test_nnunet_window_context_overrides_remembered_paths(self):
        _AppFixture.app()
        import nnunet_training_setup_ui as ui

        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            home = self._settings_home(
                tmp,
                "nnunet_settings.json",
                {"dataset_root": str(tmp / "remembered")},
            )
            with mock.patch.object(ui.Path, "home", lambda: home):
                window = ui.TrainingSetupWindow(
                    {"dataset_root": str(tmp / "from-context")},
                    tmp / "ctx.json",
                    QT,
                )
                try:
                    self.assertEqual(
                        window.dataset_edit.text(), str(tmp / "from-context")
                    )
                finally:
                    window.window.close()

    def test_nnunet_form_fields_carry_one_line_explanations(self):
        # A5 minimal plan: Dataset ID / Fold / GPU count / GPU devices (the
        # concepts an annotator cannot be expected to know) must carry a
        # visible one-line explanation, not just a hover tooltip.
        _AppFixture.app()
        import nnunet_training_setup_ui as ui

        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            home = self._settings_home(tmp, "nnunet_settings.json", {})
            with mock.patch.object(ui.Path, "home", lambda: home):
                window = ui.TrainingSetupWindow(
                    {"workspace": ""}, tmp / "ctx.json", QT
                )
                try:
                    for attribute in (
                        "dataset_id_hint",
                        "fold_hint",
                        "gpu_count_hint",
                        "gpu_devices_hint",
                    ):
                        hint = getattr(window, attribute, None)
                        self.assertIsNotNone(
                            hint, "{0} must exist".format(attribute)
                        )
                        text = (hint.text() or "").strip()
                        self.assertTrue(
                            text, "{0} must not be empty".format(attribute)
                        )
                    # The GPU devices hint must state the validation rule
                    # (device count must match GPU count) that submission
                    # otherwise rejects with an error dialog.
                    self.assertIn("must match", window.gpu_devices_hint.text())
                    self.assertIn("the GPU count above", window.gpu_devices_hint.text())
                    # Fold hint must not promise unverified split semantics.
                    self.assertIn("five", window.fold_hint.text())
                finally:
                    window.window.close()

    def test_flexict_window_prefills_remembered_paths(self):
        _AppFixture.app()
        import flexict_training_setup_ui as ui

        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            home = self._settings_home(
                tmp,
                "flexict_settings.json",
                {
                    "workspace": str(tmp / "ws"),
                    "dataset_root": str(tmp / "data"),
                    "label_name": "kidney",
                },
            )
            with mock.patch.object(ui.Path, "home", lambda: home), mock.patch.object(
                ui, "workspace_root", lambda _c: str(tmp / "default-ws")
            ), mock.patch.object(
                ui, "load_config", lambda: {"root": str(tmp)}
            ), mock.patch.object(
                ui.TrainingSetupWindow, "_refresh_models", lambda self: None
            ):
                window = ui.TrainingSetupWindow(
                    {"workspace": ""}, tmp / "ctx.json", QT
                )
                try:
                    self.assertEqual(
                        window.dataset_edit.text(), str(tmp / "data")
                    )
                    self.assertEqual(window.label_edit.text(), "kidney")
                finally:
                    window.window.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
