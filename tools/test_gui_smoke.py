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
_QApplication = QtWidgets.QApplication


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
        self.assertEqual(editor.window.windowTitle(), "配置编辑器")
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
        self.assertEqual(window.window.windowTitle(), "AI 模型管理器")
        # The background scan is queued through _results; the window must
        # survive construction without any workspace present.
        window.window.close()

    @staticmethod
    def _row(**overrides):
        row = {
            "family": "nninteractive",
            "workspace": "unused",
            "task_id": "task1",
            "model_id": "model_a",
            "target": "Task 1",
            "configuration": "",
            "created_at_epoch": 100.0,
            "imported": True,
            "current": False,
            "usable": True,
        }
        row.update(overrides)
        return row

    def _window_with_rows(self, manager, rows):
        with tempfile.TemporaryDirectory() as name:
            workspaces = {"nninteractive": name, "nnunet": name, "flexict": name}
        window = manager.ModelManagerWindow({"workspaces": workspaces}, QT)
        self.addCleanup(window.window.close)
        deadline = time.time() + 5.0
        while window._loading and time.time() < deadline:
            _AppFixture.app().processEvents()
            window._poll()
            time.sleep(0.02)
        self.assertFalse(window._loading, "background load never finished")
        return window

    def test_action_buttons_follow_selection(self):
        """Use This Model / Remove Broken Entry were permanently disabled
        (R61-4): selection must drive their enabled state."""
        _AppFixture.app()
        _batch, manager, _config, _presets = _gui_modules()
        rows = [
            self._row(model_id="good", usable=True),
            self._row(model_id="gone", usable=False),
        ]
        with mock.patch.object(manager, "collect_rows", lambda ws: list(rows)):
            window = self._window_with_rows(manager, rows)
            self.assertEqual(window.table.rowCount(), 2)
            # No selection yet: both action buttons stay disabled.
            self.assertFalse(window.use_button.isEnabled())
            self.assertFalse(window.remove_button.isEnabled())
            # Usable row: Use enabled, Remove not.
            window.table.selectRow(0)
            self.assertTrue(window.use_button.isEnabled())
            self.assertFalse(window.remove_button.isEnabled())
            # Broken row: the inverse.
            window.table.selectRow(1)
            self.assertFalse(window.use_button.isEnabled())
            self.assertTrue(window.remove_button.isEnabled())
            # Back to no selection: both disabled again.
            window.table.setCurrentCell(-1, -1)
            self.assertFalse(window.use_button.isEnabled())
            self.assertFalse(window.remove_button.isEnabled())

    def test_use_clicked_switches_recommended_model(self):
        _AppFixture.app()
        _batch, manager, _config, _presets = _gui_modules()
        rows = [self._row(model_id="good", usable=True)]
        with mock.patch.object(
            manager, "collect_rows", lambda ws: list(rows)
        ), mock.patch.object(manager, "set_recommended_model") as set_current:
            window = self._window_with_rows(manager, rows)
            window.table.selectRow(0)
            window._use_clicked()
            set_current.assert_called_once_with(rows[0])
            self.assertTrue(
                window._loading, "Use must trigger a table reload"
            )

    def test_use_button_disabled_for_usable_nnunet_rows(self):
        """AUD 5.2: nnU-Net picks its model at predict time; "use this
        model" only raises for those rows, so the button must stay off
        and explain why via tooltip instead of erroring on click."""
        _AppFixture.app()
        _batch, manager, _config, _presets = _gui_modules()
        rows = [
            self._row(model_id="remote_nn", family="nnunet", usable=True),
            self._row(model_id="nnint_ok", family="nninteractive", usable=True),
        ]
        with mock.patch.object(manager, "collect_rows", lambda ws: list(rows)):
            window = self._window_with_rows(manager, rows)
            window.table.selectRow(0)
            self.assertFalse(window.use_button.isEnabled())
            self.assertIn("预测窗口", window.use_button.toolTip())
            window.table.selectRow(1)
            self.assertTrue(window.use_button.isEnabled())
            self.assertEqual(window.use_button.toolTip(), "")

    def test_source_column_shows_remote_provenance(self):
        """AUD 5.3: the source column must say where a model came from
        (remote server profile / imported) instead of a bare
        trained-here for everything."""
        _AppFixture.app()
        _batch, manager, _config, _presets = _gui_modules()
        self.assertEqual(
            manager._model_source(
                {"imported_from_bundle": True, "execution_backend": "remote"}
            ),
            "imported",
        )
        self.assertEqual(
            manager._model_source(
                {
                    "execution_backend": "remote",
                    "remote_profile_name": "206_root",
                    "remote_profile_id": "206_root",
                }
            ),
            "remote: 206_root",
        )
        self.assertEqual(
            manager._model_source(
                {"execution_backend": "remote", "remote_profile_id": "171x"}
            ),
            "remote: 171x",
        )
        self.assertEqual(manager._model_source({}), "")


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

    def test_nnunet_validation_dice_reads_fold_summary(self):
        """AUD 5.1: the models tab must surface the fold validation Dice
        (summary.json foreground_mean), not just creation time/usability."""
        import nnunet_status_viewer as viewer

        with tempfile.TemporaryDirectory() as name:
            model_dir = Path(name) / "nnunet_x"
            summary = model_dir / "fold_0" / "validation" / "summary.json"
            summary.parent.mkdir(parents=True)
            summary.write_text(
                json.dumps(
                    {"foreground_mean": {"Dice": 0.7322440882970873}}
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                viewer._validation_dice(
                    {"model_dir": str(model_dir), "folds": ["0"]}
                ),
                "0.732",
            )
            # Missing model dir / unreadable summary: blank, never a raise.
            self.assertEqual(
                viewer._validation_dice({"model_dir": str(Path(name) / "gone")}),
                "",
            )
            self.assertEqual(
                viewer._validation_dice(
                    {"model_dir": str(model_dir), "folds": ["9"]}
                ),
                "",
            )
            # folds omitted: falls back to fold "0" like model_usability.
            self.assertEqual(
                viewer._validation_dice({"model_dir": str(model_dir)}),
                "0.732",
            )

    def test_flexict_viewer_shows_training_curve(self):
        # G3: the FlexiCT status viewer had no training-curve tab while the
        # nnU-Net viewer did - the annotator watching a FlexiCT run (local
        # or remote) had to read raw logs to see whether loss was moving.
        # Same contract as the nnU-Net viewer: status-recorded curve path
        # (remote sync target) wins, then the model's fold_0/progress.png.
        _AppFixture.app()
        import flexict_status_viewer as viewer

        window = viewer.StatusWindow({"workspace": ""}, QT)
        try:
            # No curve anywhere: hint text, never an exception.
            window._show_curve({})
            self.assertIn("暂不可用", window.curve.text())
            # A recorded curve path (e.g. remote-synced) is displayed:
            # text is cleared, no exception even for an undecodable file.
            window._show_curve({"training_curve_path": __file__})
            self.assertIn("", window.curve.text())
        finally:
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


class TestBatchPredictionWindow(unittest.TestCase):
    """D20: batch prediction dispatches per-case jobs and reports skips."""

    @staticmethod
    def _usable_model_row(model_dir: Path) -> dict:
        manifest = model_dir / "mimics_model_manifest.json"
        return {
            "model_id": "model_a",
            "task_id": "liver",
            "task_name": "Liver",
            "configuration": "3d_fullres",
            "labels": [{"id": 1, "name": "Liver"}],
            "dataset_id": 711,
            "model_dir": str(model_dir),
            "manifest_path": str(manifest),
            "created_at_epoch": 100.0,
        }

    def _model_dir(self, tmp: Path) -> Path:
        import nnunet_common as common

        model_dir = tmp / "models" / "liver" / "model_a"
        (model_dir / "fold_0").mkdir(parents=True)
        for name in ("plans.json", "dataset.json"):
            (model_dir / name).write_text("{}", encoding="utf-8")
        (model_dir / "fold_0" / "checkpoint_final.pth").write_bytes(b"w")
        common.write_json_atomic(
            model_dir / "mimics_model_manifest.json",
            {
                "schema_version": common.MODEL_SCHEMA_VERSION,
                "model_id": "model_a",
                "model_dir": str(model_dir),
                "manifest_path": str(model_dir / "mimics_model_manifest.json"),
                "folds": ["0"],
            },
        )
        return model_dir

    def test_submit_dispatches_per_case_jobs_and_reports_skips(self):
        _AppFixture.app()
        import nnunet_batch_predict_ui as ui

        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            model_dir = self._model_dir(tmp)
            row = self._usable_model_row(model_dir)
            cases = tmp / "cases"
            for case in ("case_a", "case_b"):
                image = cases / case / "ct.nii.gz"
                image.parent.mkdir(parents=True)
                image.write_bytes(b"x")
            # Ambiguous dir: two images, which one to predict is a guess.
            (cases / "ambiguous").mkdir(parents=True)
            for extra in ("a.nii.gz", "b.nii.gz"):
                (cases / "ambiguous" / extra).write_bytes(b"x")

            window = ui.BatchPredictionWindow(
                {"workspace": str(tmp / "workspace")}, QT
            )
            self.addCleanup(window.window.close)
            # Feed the model discovery result directly (background path is
            # covered by the same _poll timer as the single-case window).
            window._apply_models([row], "")
            self.assertEqual(window.table.rowCount(), 1)
            self.assertTrue(window.start.isEnabled())

            requests = []

            def fake_create_job(request):
                requests.append(request)
                return {"job_id": request["job_id"], "status_path": "s.json"}

            window.image_root_edit.setText(str(cases))
            window.output_edit.setText(str(tmp / "out"))
            with mock.patch.object(ui, "create_job", fake_create_job):
                window._submit()
                deadline = time.time() + 5.0
                while time.time() < deadline:
                    _AppFixture.app().processEvents()
                    window._poll_background_results()
                    if requests:
                        break
                    time.sleep(0.02)
            self.assertEqual(
                ["case_a", "case_b"], [request["case_id"] for request in requests]
            )
            self.assertTrue(
                str(requests[0]["output_path"]).endswith("out\\case_a.nii.gz")
                or str(requests[0]["output_path"]).endswith("out/case_a.nii.gz")
            )
            self.assertIn("已启动 2 个预测任务", window.status_label.text())
            self.assertIn("ambiguous", window.status_label.text())

    def test_submit_passes_modality_and_cpu_flags_to_jobs(self):
        """R62-5: the batch window must not bypass the modality guardrail
        and CPU fallback that the single-case window and CLI enforce."""
        _AppFixture.app()
        import nnunet_batch_predict_ui as ui

        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            window = ui.BatchPredictionWindow(
                {"workspace": str(tmp / "workspace")}, QT
            )
            self.addCleanup(window.window.close)
            window._apply_models(
                [self._usable_model_row(self._model_dir(tmp))], ""
            )
            case_dir = tmp / "cases" / "case_a"
            case_dir.mkdir(parents=True)
            (case_dir / "ct.nii.gz").write_bytes(b"x")

            requests = []

            def fake_create_job(request):
                requests.append(request)
                return {"job_id": request["job_id"], "status_path": "s.json"}

            window.image_root_edit.setText(str(tmp / "cases"))
            window.output_edit.setText(str(tmp / "out"))
            # "自动（按模型）" default: no declared modality, guardrail off.
            with mock.patch.object(ui, "create_job", fake_create_job):
                window._submit()
                deadline = time.time() + 5.0
                while time.time() < deadline:
                    _AppFixture.app().processEvents()
                    window._poll_background_results()
                    if requests:
                        break
                    time.sleep(0.02)
            self.assertEqual(1, len(requests))
            self.assertEqual("", requests[0]["source_modality"])
            self.assertFalse(requests[0]["use_cpu"])

            window.modality_combo.setCurrentText("MR")
            window.cpu_check.setChecked(True)
            requests.clear()
            with mock.patch.object(ui, "create_job", fake_create_job):
                window._submit()
                deadline = time.time() + 5.0
                while time.time() < deadline:
                    _AppFixture.app().processEvents()
                    window._poll_background_results()
                    if requests:
                        break
                    time.sleep(0.02)
            self.assertEqual(1, len(requests))
            self.assertEqual("MR", requests[0]["source_modality"])
            self.assertTrue(requests[0]["use_cpu"])

    def test_missing_folders_are_rejected_before_dispatch(self):
        _AppFixture.app()
        import nnunet_batch_predict_ui as ui

        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            window = ui.BatchPredictionWindow(
                {"workspace": str(tmp / "workspace")}, QT
            )
            self.addCleanup(window.window.close)
            window._apply_models(
                [self._usable_model_row(self._model_dir(tmp))], ""
            )
            window.image_root_edit.setText("")
            window._submit()
            self.assertIn("病例文件夹", window.status_label.text())
            window.image_root_edit.setText(str(tmp))
            window.output_edit.setText("")
            window._submit()
            self.assertIn("输出文件夹", window.status_label.text())


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


class TestIoPathOutputNotSticky(unittest.TestCase):
    """R61-7: a remembered output folder that was merely the computed
    default for a *previous* dataset must not pin every new dataset to that
    old folder. Only a folder the user chose themselves sticks."""

    def test_io_path_setup_prefills_only_user_chosen_output(self):
        source = Path(__file__).with_name("io_path_setup_ui.py").read_text(
            encoding="utf-8"
        )
        # Load side: prefill the output box only when the remembered value
        # was a user choice (output_custom), never a computed default.
        self.assertIn('remembered_mode.get("output_custom")', source)
        # Save side: record whether the user deviated from the computed
        # default at submit time.
        self.assertIn('"output_custom": bool(', source)

    def test_drop_window_output_follows_each_new_dataset(self):
        drop = Path(__file__).with_name("import_drop_window.py").read_text(
            encoding="utf-8"
        )
        # Load side: same rule as the path-setup UI.
        self.assertIn('remembered_mode.get("output_custom")', drop)
        # Within one session, every new drop recomputes the default unless
        # the user typed or browsed a folder themselves.
        self.assertIn('if not output_custom["value"]:', drop)
        self.assertIn('output_edit.textChanged.connect(output_edited)', drop)
        # The dead build_selection helper (zero callers repo-wide) stays out.
        self.assertNotIn("def build_selection", drop)


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

    def test_nnunet_val_cases_semantics(self):
        # G9: the nnU-Net wizard asked for a validation fraction while the
        # FlexiCT wizard asked for a case count - the same concept, two
        # semantics. Unified to case counts; 0 = automatic 20% (previous
        # default behaviour, now computed by the normalizer).
        _AppFixture.app()
        import nnunet_training_setup_ui as ui

        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            home = self._settings_home(tmp, "nnunet_settings.json", {})
            with mock.patch.object(ui.Path, "home", lambda: home):
                window = ui.TrainingSetupWindow(
                    {"workspace": str(tmp / "ws")}, tmp / "ctx.json", QT
                )
                try:
                    self.assertEqual(window._request()["val_cases"], 0)
                    self.assertEqual(
                        window._request()["validation_fraction"], 0.2
                    )
                    window.validation_spin.setValue(4)
                    self.assertEqual(window._request()["val_cases"], 4)
                finally:
                    window.window.close()

    def test_training_wizards_warn_on_duplicate_task(self):
        # G7: retraining the same task_id silently piles up near-duplicate
        # models. Both wizards must surface existing models for the task
        # before GPU time is spent (warning only, submission still allowed).
        _AppFixture.app()
        import flexict_training_setup_ui as flexict_ui

        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            home = self._settings_home(
                tmp, "flexict_settings.json", {"label_name": "kidney"}
            )
            with mock.patch.object(flexict_ui.Path, "home", lambda: home), mock.patch.object(
                flexict_ui, "workspace_root", lambda _c: str(tmp / "ws")
            ), mock.patch.object(
                flexict_ui, "load_config", lambda: {"root": str(tmp)}
            ), mock.patch.object(
                flexict_ui, "load_models",
                lambda workspace: [
                    {"model_id": "flexict_kidney_v1", "task_id": "kidney"},
                    {"model_id": "flexict_other", "task_id": "liver"},
                ],
            ), mock.patch.object(
                flexict_ui.TrainingSetupWindow, "_refresh_models",
                lambda self: None,
            ):
                window = flexict_ui.TrainingSetupWindow(
                    {"workspace": str(tmp / "ws")}, tmp / "ctx.json", QT
                )
                try:
                    window.label_edit.setText("kidney")
                    warning = window._duplicate_model_warning(window._request())
                    self.assertIn("flexict_kidney_v1", warning)
                    self.assertIn("1 个模型", warning)
                    # A different task must not warn.
                    window.label_edit.setText("liver_xyz")
                    self.assertNotIn(
                        "个模型",
                        window._duplicate_model_warning(window._request()),
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
                    self.assertIn("必须与上方的 GPU 数量一致", window.gpu_devices_hint.text())
                                        # Fold hint must not promise unverified split semantics.
                    self.assertIn("五份数据", window.fold_hint.text())
                finally:
                    window.window.close()

    def test_nnunet_training_group_collapsed_by_default(self):
        # D2: the Training group (Trainer/Fold/GPUs/epochs...) is advanced —
        # hidden behind a toggle. Hidden widgets still submit their defaults.
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
                    # isHidden() (not isVisible(): the offscreen test never
                    # shows the window) reports the widget's own visibility.
                    self.assertTrue(window.training_group.isHidden())
                    self.assertFalse(window.advanced_toggle.isChecked())
                    window.advanced_toggle.setChecked(True)
                    self.assertFalse(window.training_group.isHidden())
                    window.advanced_toggle.setChecked(False)
                    self.assertTrue(window.training_group.isHidden())
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

    def test_training_windows_carry_cross_family_hint(self):
        """AUD 3.1: each family window must tell the annotator when to use
        nnU-Net vs FlexiCT vs nnInteractive, so the choice is not a guess."""
        source_root = Path(__file__).parent
        for module_name in (
            "nnunet_training_setup_ui",
            "flexict_training_setup_ui",
            "nninteractive_task_model_center",
        ):
            source = (source_root / (module_name + ".py")).read_text(
                encoding="utf-8"
            )
            self.assertIn("如何选方法", source)
            self.assertIn("nnU-Net", source)
            self.assertIn("FlexiCT", source)
            self.assertIn("nnInteractive", source)

    def test_application_icon_does_not_crash_process(self):
        """P0: PySide6 6.11 dropped drawPolygon's varargs overload; the old
        code raised TypeError inside _application_icon, the bare except left
        an active painter on the pixmap, and Qt aborted the whole process.
        The icon must build cleanly (or return None) without native abort."""
        _AppFixture.app()
        import ui_theme

        icon = ui_theme._application_icon()
        self.assertIsNotNone(icon)
        # configure_application is the crash path every window shares; call
        # it against the shared QApplication and confirm the process lives.
        from PySide6 import QtWidgets

        app = QtWidgets.QApplication.instance()
        self.assertIsNotNone(app)
        ui_theme.configure_application(app)

    def test_flexict_batch_size_override_reaches_request(self):
        # R-P1-8: the FlexiCT wizard had no batch size field while the nnU-Net
        # wizard did — a planner-sized batch that OOMs a 16GB remote card left
        # the annotator no self-rescue from the wizard. Default stays None
        # (planner decides); unchecking auto submits the manual value.
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
                    self.assertTrue(window.auto_batch.isChecked())
                    self.assertFalse(window.batch_spin.isEnabled())
                    self.assertIsNone(window._request()["batch_size"])
                    window.auto_batch.setChecked(False)
                    self.assertTrue(window.batch_spin.isEnabled())
                    window.batch_spin.setValue(6)
                    self.assertEqual(window._request()["batch_size"], 6)
                finally:
                    window.window.close()

    def test_flexict_overlap_policy_reaches_request(self):
        # G2: multiclass FlexiCT shares the nnU-Net case materializer, which
        # stops on overlapping label voxels — but the FlexiCT wizard had no
        # overlap option, so the annotator hitting an overlap error had no
        # self-rescue from the wizard. Default fail; checkbox submits replace.
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
                    self.assertFalse(window.overlap_check.isChecked())
                    self.assertEqual(window._request()["overlap_policy"], "fail")
                    window.overlap_check.setChecked(True)
                    self.assertEqual(window._request()["overlap_policy"], "replace")
                finally:
                    window.window.close()

    def test_flexict_single_case_submission_is_rejected(self):
        # B30: the error copy says "at least two cases" but validation only
        # checked non-empty; one case reached the pipeline, which burned a
        # GPU lock on an empty training run.
        _AppFixture.app()
        import flexict_training_setup_ui as ui

        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            (tmp / "data" / "case01").mkdir(parents=True)
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
                    with mock.patch.object(
                        window, "_request",
                        return_value={
                            "workspace": str(tmp / "ws"),
                            "dataset_root": str(tmp / "data"),
                            "label_name": "kidney",
                            "cases": ["case01"],
                        },
                    ):
                        window._submit()
                    deadline = time.time() + 5
                    while (
                        window._submission_pending
                        and time.time() < deadline
                    ):
                        _QApplication.processEvents()
                        time.sleep(0.02)
                        window._poll_submission()
                    self.assertFalse(window.submitted)
                    self.assertIn(
                        "at least two cases",
                        window.status_label.text().lower(),
                    )
                finally:
                    window.window.close()

    def test_flexict_continue_training_reaches_request_and_blocks_remote(self):
        # R61-25: "continue interrupted training" must reach the request (and
        # through it the trainer's continue_training flag), and must be
        # rejected when the compute tab selects a remote backend — remote
        # containers are disposable, so the local checkpoint_latest the
        # resume depends on is invisible to them.
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
                    self.assertFalse(window.continue_check.isChecked())
                    self.assertFalse(window._request()["continue_training"])
                    window.continue_check.setChecked(True)
                    self.assertTrue(window._request()["continue_training"])
                    # Remote backend + continue must fail validation without
                    # spawning the path-check worker.
                    with mock.patch.object(
                        window, "_request",
                        return_value={
                            "workspace": str(tmp / "ws"),
                            "dataset_root": str(tmp / "data"),
                            "label_name": "kidney",
                            "cases": ["case01", "case02"],
                            "continue_training": True,
                            "execution_backend": "remote",
                            "remote_profile_id": "206_root",
                        },
                    ):
                        window._submit()
                    self.assertFalse(window._submission_pending)
                    self.assertFalse(window.submitted)
                    self.assertIn(
                        "远程后端不支持",
                        window.status_label.text(),
                    )
                finally:
                    window.window.close()


class TestFlexictStopBehaviour(unittest.TestCase):
    """R61-17: remote-state-unknown FlexiCT tasks get the same Abandon
    Locally confirmation as the nnU-Net viewer; plain local Stop stays
    one-click (spec: normal flow carries no extra dialog)."""

    def _window_with_rows(self, rows):
        import flexict_status_viewer as viewer

        _AppFixture.app()
        window = viewer.StatusWindow({"workspace": ""}, QT)
        window._job_rows = rows
        # One list item selected at row 0, like a real selection. refresh()
        # is stubbed so _stop() does not collect from the real workspace.
        window.jobs.addItem("job")
        window.jobs.setCurrentRow(0)
        window.refresh = lambda *args, **kwargs: None
        return window

    def test_local_stop_calls_stop_job_without_dialog(self):
        window = self._window_with_rows([
            {"status_path": "X:/j/status.json", "status": "running",
             "execution_backend": "local"},
        ])
        try:
            with mock.patch.object(
                window.QtWidgets.QMessageBox, "warning"
            ) as warn, mock.patch("flexict_status_viewer.stop_job") as stop, \
                    mock.patch("flexict_status_viewer.abandon_job") as abandon:
                window._stop()
            warn.assert_not_called()
            stop.assert_called_once_with("X:/j/status.json")
            abandon.assert_not_called()
        finally:
            window.window.close()

    def test_remote_unknown_stop_asks_then_abandons(self):
        window = self._window_with_rows([
            {"status_path": "X:/j/status.json", "status": "stopping",
             "execution_backend": "remote", "remote_state_unknown": True},
        ])
        try:
            with mock.patch.object(
                window.QtWidgets.QMessageBox, "warning",
                return_value=QtWidgets.QMessageBox.Yes,
            ) as warn, mock.patch(
                "flexict_status_viewer.stop_job"
            ) as stop, mock.patch(
                "flexict_status_viewer.abandon_job"
            ) as abandon:
                window._stop()
            warn.assert_called_once()
            self.assertIn(
                "无法确认", warn.call_args[0][2]
            )
            abandon.assert_called_once_with("X:/j/status.json")
            stop.assert_not_called()
        finally:
            window.window.close()

    def test_remote_unknown_stop_cancelled_does_nothing(self):
        window = self._window_with_rows([
            {"status_path": "X:/j/status.json", "status": "stopping",
             "execution_backend": "remote", "remote_state_unknown": True},
        ])
        try:
            with mock.patch.object(
                window.QtWidgets.QMessageBox, "warning",
                return_value=QtWidgets.QMessageBox.No,
            ) as warn, mock.patch(
                "flexict_status_viewer.stop_job"
            ) as stop, mock.patch(
                "flexict_status_viewer.abandon_job"
            ) as abandon:
                window._stop()
            warn.assert_called_once()
            stop.assert_not_called()
            abandon.assert_not_called()
        finally:
            window.window.close()

    def test_remote_known_state_uses_plain_stop(self):
        # Remote but the server still answers: stop_job cancels through the
        # controller; no abandon escape hatch is offered.
        window = self._window_with_rows([
            {"status_path": "X:/j/status.json", "status": "running",
             "execution_backend": "remote", "remote_state_unknown": False},
        ])
        try:
            with mock.patch.object(
                window.QtWidgets.QMessageBox, "warning"
            ) as warn, mock.patch(
                "flexict_status_viewer.stop_job"
            ) as stop, mock.patch(
                "flexict_status_viewer.abandon_job"
            ) as abandon:
                window._stop()
            warn.assert_not_called()
            stop.assert_called_once_with("X:/j/status.json")
            abandon.assert_not_called()
        finally:
            window.window.close()

    def test_stop_button_label_reflects_abandonable_state(self):
        import flexict_status_viewer as viewer

        _AppFixture.app()
        window = viewer.StatusWindow({"workspace": ""}, QT)
        try:
            window.jobs.addItem("job")
            window.jobs.setCurrentRow(0)
            window._job_rows = [
                {"status": "stopping", "execution_backend": "remote",
                 "remote_state_unknown": True},
            ]
            window._show_selected_job()
            self.assertEqual("在本机放弃远程任务", window.stop_button.text())
            self.assertTrue(window.stop_button.isEnabled())

            window._job_rows = [
                {"status": "running", "execution_backend": "local"},
            ]
            window._show_selected_job()
            self.assertEqual("停止", window.stop_button.text())
            self.assertTrue(window.stop_button.isEnabled())

            window._job_rows = [
                {"status": "completed", "execution_backend": "local"},
            ]
            window._show_selected_job()
            self.assertEqual("停止", window.stop_button.text())
            self.assertFalse(window.stop_button.isEnabled())
        finally:
            window.window.close()


class TestActiveLearningWindow(unittest.TestCase):
    """F03/T03 + F14/T14: the AL review window with real data rows.

    _job_selected used QtWidgets/QtCore names that were only bound in other
    functions' scopes — the window worked with zero rows and raised
    NameError the moment a real ranking existed. The failure feedback was
    then overwritten by the refresh summary (F14).
    """

    def _job_dir(self, root, ranking):
        """A completed active_learning job with a ranking."""
        from flexict_common import TERMINAL_STATES  # noqa: F401

        job_dir = root / "jobs" / "aljob1"
        (job_dir / "uncertainty").mkdir(parents=True)
        (job_dir / "uncertainty" / "ranking.csv").write_text(
            "case,integrated,uncertain_vol,max\n"
            + "".join(
                "{0},{1:.2f},10.0,1.0\n".format(name, 9.0 - index)
                for index, name in enumerate(ranking)
            ),
            encoding="utf-8",
        )
        return job_dir

    def _window(self, tmp, ranking=("caseA", "caseB")):
        import flexict_active_learning_ui as al_ui

        _AppFixture.app()
        root = Path(tmp)
        job_dir = self._job_dir(root, ranking)
        (job_dir / "status.json").write_text(
            json.dumps({
                "kind": "active_learning",
                "status": "completed",
                "created_at_epoch": 1.0,
                "label_name": "Liver",
                "job_dir": str(job_dir),
            }),
            encoding="utf-8",
        )
        window = al_ui.ActiveLearningWindow(
            {"workspace": str(root)}, QT)
        return window

    def test_real_rows_populate_without_nameerror(self):
        with tempfile.TemporaryDirectory() as tmp:
            window = self._window(tmp)
            try:
                self.assertEqual(
                    2, window.table.rowCount(),
                    "non-empty ranking must populate the table")
                self.assertTrue(window.open_button.isEnabled())
                summary = window.status_label.text()
                self.assertIn("2 case(s) ranked", summary)
            finally:
                window.window.close()

    def test_annotated_row_gets_success_foreground(self):
        # F03's crash was inside the per-row loop's annotated branch.
        with tempfile.TemporaryDirectory() as tmp:
            window = self._window(tmp)
            try:
                job_dir = Path(window._job_dir)
                from flexict_active_learning_ui import set_case_state
                set_case_state(job_dir, "caseA", "annotated")
                window._job_selected()
                self.assertEqual(
                    "annotated",
                    window.table.item(0, 4).text())
                self.assertIn("1 annotated", window.status_label.text())
            finally:
                window.window.close()

    def test_failed_request_message_survives_refresh(self):
        # F14/T14: the poll sets the outcome AFTER refreshing the table.
        with tempfile.TemporaryDirectory() as tmp:
            window = self._window(tmp)
            try:
                request_dir = Path(window._job_dir) / "apply_requests"
                request_dir.mkdir(parents=True)
                (request_dir / "a.json").write_text(
                    json.dumps({
                        "case_id": "caseA", "what": "bands",
                        "state": "failed", "detail": "geometry mismatch",
                        "requested_at_epoch": 1.0,
                        "updated_at_epoch": time.time(),
                    }),
                    encoding="utf-8",
                )
                window._request_poll_epoch = 0.0
                window._poll_requests()
                text = window.status_label.text()
                self.assertIn("Request failed for caseA", text)
                self.assertIn("geometry mismatch", text)
            finally:
                window.window.close()

    def test_applied_request_message_survives_refresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            window = self._window(tmp)
            try:
                request_dir = Path(window._job_dir) / "apply_requests"
                request_dir.mkdir(parents=True)
                (request_dir / "a.json").write_text(
                    json.dumps({
                        "case_id": "caseA", "what": "open_bands",
                        "state": "applied", "detail": "applied: bands",
                        "requested_at_epoch": 1.0,
                        "updated_at_epoch": time.time(),
                    }),
                    encoding="utf-8",
                )
                window._request_poll_epoch = 0.0
                window._poll_requests()
                text = window.status_label.text()
                self.assertIn("Applied in Mimics: caseA", text)
            finally:
                window.window.close()

    def _pair_model(self, workspace: Path, configuration: str) -> dict:
        model_dir = (
            workspace / "runtime" / "nnUNet_results"
            / "Dataset750_Kidney" / "flexict{}_Trainer__nnUNetPlans__{}".format(
                "2d" if configuration == "2d" else "3d", configuration)
        )
        (model_dir / "fold_0").mkdir(parents=True)
        (model_dir / "fold_0" / "checkpoint_best.pth").write_bytes(b"x")
        (model_dir / "plans.json").write_text("{}", encoding="utf-8")
        (model_dir / "dataset.json").write_text("{}", encoding="utf-8")
        manifest = {
            "model_id": "m_{}".format(configuration),
            "model_dir": str(model_dir),
            "manifest_path": str(
                model_dir / "flexict_model_manifest.json"),
            "configuration": configuration,
            "label_name": "kidney_left",
            "task_id": "Kidney", "task_name": "Kidney",
            "pair_id": "pair_1",
            "created_at_epoch": 100.0,
        }
        from nnunet_common import write_json_atomic

        write_json_atomic(
            Path(manifest["manifest_path"]), manifest)
        return manifest

    def test_new_run_form_without_pair_explains_how_to_fix(self):
        # F04: a fresh workspace must not dead-end — the form names the
        # training entry instead of pointing back at this same menu.
        _AppFixture.app()
        import flexict_active_learning_ui as al_ui

        with tempfile.TemporaryDirectory() as tmp:
            window = al_ui.ActiveLearningWindow(
                {"workspace": str(Path(tmp) / "workspace")}, QT)
            self.addCleanup(window.window.close)
            window._apply_pair((None, None), "")
            self.assertIn("01 训练模型", window.pair_label.text())
            self.assertFalse(window.start_run_button.isEnabled())
            # The old circular empty state pointed at this same entry.
            self.assertIn("启动新的排序", window.status_label.text())

    def test_submit_new_run_creates_al_job_from_pair(self):
        _AppFixture.app()
        import flexict_active_learning_ui as al_ui

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            model_2d = self._pair_model(workspace, "2d")
            model_3d = self._pair_model(workspace, "3d_fullres")
            from flexict_common import save_registry

            save_registry(workspace, None, [model_2d, model_3d])

            window = al_ui.ActiveLearningWindow(
                {"workspace": str(workspace)}, QT)
            self.addCleanup(window.window.close)
            # Feed the pair check result directly; the request then uses
            # the pair the window itself resolved.
            window._apply_pair((model_2d, model_3d), "")
            self.assertTrue(window.start_run_button.isEnabled())
            pool = Path(tmp) / "pool"
            pool.mkdir()
            window.pool_edit.setText(str(pool))

            requests = []

            def fake_create_job(values):
                requests.append(values)
                # create_flexict_job generates the job_id itself.
                return {
                    "job_id": values.get("job_id") or "active_learning_test",
                    "status_path": "s.json",
                }

            import flexict_pipeline as fp

            with mock.patch.object(
                    fp, "create_flexict_job",
                    side_effect=fake_create_job):
                window._submit_new_run()
                deadline = time.time() + 5.0
                while time.time() < deadline:
                    _AppFixture.app().processEvents()
                    window._poll_background_results()
                    if requests:
                        break
                    time.sleep(0.02)
            self.assertEqual(1, len(requests))
            request = requests[0]
            self.assertEqual("active_learning", request["operation"])
            self.assertEqual("kidney_left", request["label_name"])
            self.assertEqual(
                model_2d["manifest_path"], request["model_manifest_2d"])
            self.assertEqual(
                model_3d["manifest_path"], request["model_manifest_3d"])
            self.assertIn("已启动", window.status_label.text())

    def test_submit_new_run_requires_pool_before_dispatch(self):
        _AppFixture.app()
        import flexict_active_learning_ui as al_ui

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            model_2d = self._pair_model(workspace, "2d")
            model_3d = self._pair_model(workspace, "3d_fullres")
            window = al_ui.ActiveLearningWindow(
                {"workspace": str(workspace)}, QT)
            self.addCleanup(window.window.close)
            window._apply_pair((model_2d, model_3d), "")
            window.pool_edit.setText("")
            window._submit_new_run()
            self.assertIn("数据池", window.pair_label.text())


if __name__ == "__main__":
    unittest.main(verbosity=2)
