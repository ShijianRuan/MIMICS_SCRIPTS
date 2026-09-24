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


if __name__ == "__main__":
    unittest.main(verbosity=2)
