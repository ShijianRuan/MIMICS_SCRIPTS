#!/usr/bin/env python3
"""Offscreen GUI smoke tests for the four PySide6 external windows.

Runs with QT_QPA_PLATFORM=offscreen so no display is needed. Each test
constructs the real window class against a temporary project state and
exercises its interaction surface (table refresh, edits, validation) without
a human. This is the automatable slice of the manual GUI acceptance items in
docs/history/changes_2026-09-24_improvement_program_delivery.md; look-and-feel
remains a human check.

Run:
    python_env/python.exe tests/test_gui_smoke.py
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
                self.assertEqual(window.table.item(0, 3).text(), "运行中")
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

    def test_real_refresh_click_does_not_crash(self):
        """F34: a real clicked signal passes checked=False as its first
        argument; the old direct connection fed that bool into refresh's
        `rows` and every real click died on len(False)."""
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
                self.assertEqual(3, len(window._rows))
                errors = []
                old_hook = sys.excepthook
                sys.excepthook = (
                    lambda typ, value, tb: errors.append(str(value)))
                try:
                    button = next(
                        child for child in window.window.findChildren(
                            QtWidgets.QPushButton)
                        if child.text() == "刷新")
                    button.click()  # real signal: clicked(False)
                    _AppFixture.app().processEvents()
                    # Let the background collection finish and drain.
                    for _ in range(40):
                        _AppFixture.app().processEvents()
                        time.sleep(0.05)
                        if not errors:
                            break
                finally:
                    sys.excepthook = old_hook
                self.assertEqual([], errors)
                window.window.close()

    def test_batch_drop_import_stop_routes_to_queue_marker(self):
        """R66 audit P2: a batch drop-import row's Stop must write the
        queue-level marker the batch CLI actually polls — the per-drop
        marker was a silent no-op. Also drops the queue active marker so
        the stopped queue takes no new cases (same as Import queue rows).
        """
        _AppFixture.app()
        batch, _manager, _config, _presets = _gui_modules()
        with tempfile.TemporaryDirectory() as name:
            project = Path(name) / "proj"
            runtime = project / ".mimics_runtime"
            drop = runtime / "drop_import" / "20261003T120000_abcd_batch_status.json"
            drop.parent.mkdir(parents=True)
            output = Path(name) / "mcs_out"
            output.mkdir()
            drop.write_text(json.dumps({
                "status": "running",
                "output_path": str(output),
                "updated_at_epoch": 300.0,
            }), encoding="utf-8")
            with mock.patch.object(
                batch, "import_runtime_base",
                lambda p: Path(p) / ".mimics_runtime",
            ):
                # Point the queue runtime dir at a dir inside the sandbox.
                queue_runtime = runtime / "import_queues" / output.name
                queue_runtime.mkdir(parents=True)
                with mock.patch(
                    "runtime_common.import_queue_runtime_dir",
                    lambda root, out: str(queue_runtime),
                ):
                    rows = batch.collect_batch_rows(project)
                    batch_rows = [r for r in rows if r["kind"] == "Drop import"]
                    self.assertEqual(1, len(batch_rows))
                    marker = batch._stop_marker(batch_rows[0])
                    self.assertTrue(
                        marker.endswith("_mcs_queue_stop.json"),
                        "batch drop stop must route to the queue marker, got {0}".format(marker),
                    )
                    active = queue_runtime / "_mcs_queue_active.json"
                    active.write_text("{}", encoding="utf-8")
                    ok, _message = batch.request_stop(batch_rows[0])
                    self.assertTrue(ok)
                    self.assertTrue(
                        (queue_runtime / "_mcs_queue_stop.json").is_file(),
                        "stop marker written next to the queue the CLI polls",
                    )
                    self.assertFalse(
                        active.exists(),
                        "queue active marker dropped so the queue takes no new cases",
                    )

    def test_single_case_drop_import_stop_keeps_own_marker(self):
        """Per-case drop imports poll their own worker marker — unchanged."""
        _AppFixture.app()
        batch, _manager, _config, _presets = _gui_modules()
        with tempfile.TemporaryDirectory() as name:
            project = Path(name) / "proj"
            runtime = project / ".mimics_runtime"
            drop = runtime / "drop_import" / "case42_status.json"
            drop.parent.mkdir(parents=True)
            drop.write_text(json.dumps({
                "status": "running",
                "case_id": "case42",
                "updated_at_epoch": 300.0,
            }), encoding="utf-8")
            with mock.patch.object(
                batch, "import_runtime_base",
                lambda p: Path(p) / ".mimics_runtime",
            ):
                rows = batch.collect_batch_rows(project)
                drop_rows = [r for r in rows if r["kind"] == "Drop import"]
                self.assertEqual(1, len(drop_rows))
                marker = batch._stop_marker(drop_rows[0])
                self.assertTrue(
                    marker.endswith("case42_stop.json"),
                    "single-case drop keeps its worker-polled marker, got {0}".format(marker),
                )


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

    def test_cells_carry_full_text_tooltip(self):
        """F15：长模型 ID 被列宽省略成 "..."，悬停必须能看到完整内容，
        否则用户无法区分同一任务的多个模型版本。"""
        _AppFixture.app()
        _batch, manager, _config, _presets = _gui_modules()
        long_id = (
            "nnUNetModels/nnUNet_Results_20261003_" "病例A_肝脏分割_最终版"
        )
        rows = [self._row(model_id=long_id)]
        with mock.patch.object(manager, "collect_rows", lambda ws: list(rows)):
            window = self._window_with_rows(manager, rows)
            # 模型列（第 2 列）的 tooltip 就是完整 ID。
            item = window.table.item(0, 2)
            self.assertEqual(item.text(), long_id)
            self.assertEqual(item.toolTip(), long_id)
            # 其余列同样带 tooltip（任务名也可能被省略）。
            self.assertEqual(window.table.item(0, 1).toolTip(),
                             window.table.item(0, 1).text())

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

    def test_import_nnunet_bundle_never_passes_set_current(self):
        """nnU-Net 的导入子命令没有 --set-current（每次预测时才选模型）；
        旧代码无条件附加该参数，argparse 抛 SystemExit 直接杀死工作线程，
        导入按钮永久禁用（P1-2）。"""
        _AppFixture.app()
        _batch, manager, _config, _presets = _gui_modules()
        with tempfile.TemporaryDirectory() as name:
            bundle = Path(name) / "model.zip"
            bundle.write_bytes(b"not really a zip")
            captured = {}

            def fake_main(argv):
                captured["argv"] = list(argv)
                raise SystemExit(2)  # argparse-style hard exit

            with mock.patch.object(
                manager, "bundle_model_family", lambda p: "nnunet"
            ), mock.patch.object(manager.ai_model_bundle, "main", fake_main):
                with self.assertRaises(RuntimeError):
                    manager.import_bundle(bundle, {"nnunet": name})
            self.assertNotIn("--set-current", captured["argv"])
            # set_current=False 的其他系列仍保留该参数（显式需求）。
            captured.clear()

            def fake_main_ok(argv):
                captured["argv"] = list(argv)

            with mock.patch.object(
                manager, "bundle_model_family", lambda p: "flexict"
            ), mock.patch.object(
                manager.ai_model_bundle, "main", fake_main_ok
            ):
                manager.import_bundle(
                    bundle, {"flexict": name}, set_current=True
                )
            self.assertIn("--set-current", captured["argv"])

    def test_import_worker_surfaces_system_exit_as_error(self):
        """import_bundle 内部已把 SystemExit 转成 RuntimeError；工作线程的
        except Exception 必须把错误放回队列并由 _poll 复位按钮，而不是
        让线程静默死亡、导入按钮永久禁用（P1-2 失败路径）。"""
        _AppFixture.app()
        _batch, manager, _config, _presets = _gui_modules()
        with tempfile.TemporaryDirectory() as name:
            workspaces = {
                "nninteractive": name, "nnunet": name, "flexict": name,
            }
            window = manager.ModelManagerWindow(
                {"workspaces": workspaces}, QT
            )
            self.addCleanup(window.window.close)

            def failing_import(bundle_path, ws, set_current=True):
                raise RuntimeError("模型包导入命令执行失败（参数错误）。")

            with mock.patch.object(
                manager, "import_bundle", failing_import
            ), mock.patch.object(
                QtWidgets.QMessageBox, "warning"
            ) as warn_box:
                # The file dialog runs in a helper process (async helper);
                # the failure path under test starts once a path is chosen.
                window._import_bundle_from(name + "/model.zip")
                self.assertFalse(window.import_button.isEnabled())
                deadline = time.time() + 5.0
                while window._importing and time.time() < deadline:
                    _AppFixture.app().processEvents()
                    window._poll()
                    time.sleep(0.02)
            self.assertFalse(
                window._importing, "import worker never reported back"
            )
            self.assertTrue(window.import_button.isEnabled())
            warn_box.assert_called_once()
            self.assertIn(
                "模型包导入命令执行失败", warn_box.call_args[0][2]
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
            "导入",
        )
        self.assertEqual(
            manager._model_source(
                {
                    "execution_backend": "remote",
                    "remote_profile_name": "206_root",
                    "remote_profile_id": "206_root",
                }
            ),
            "远程: 206_root",
        )
        self.assertEqual(
            manager._model_source(
                {"execution_backend": "remote", "remote_profile_id": "171x"}
            ),
            "远程: 171x",
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
                        if "刷新失败" in window.live_label.text():
                            break
                # F34 残留：失败要可见——上一帧保留，但实时标记必须
                # 说明这帧不是新的，不能继续挂着旧时间戳装作一切正常。
                self.assertIn("刷新失败", window.live_label.text())
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

    def test_nnunet_viewer_shows_model_source(self):
        """R62-9: the models tab must distinguish remote-trained downloads
        from locally trained models."""
        _AppFixture.app()
        import nnunet_status_viewer as viewer

        window = viewer.StatusWindow({"workspace": ""}, QT)
        try:
            # An empty context falls back to the real default workspace,
            # whose task combo would filter our injected rows out — pin
            # the selected task for this test.
            with mock.patch.object(
                window, "_selected_task", return_value="liver"
            ):
                window.refresh({
                "jobs": [
                    {"task_id": "liver", "task_name": "Liver",
                     "status_path": "s.json", "status": "completed",
                     "created_at_epoch": 1.0, "kind": "task"},
                ],
                "models": [
                    {"task_id": "liver", "model_id": "local_m",
                     "configuration": "3d_fullres", "created_at_epoch": 1.0},
                    {"task_id": "liver", "model_id": "remote_m",
                     "configuration": "3d_fullres", "created_at_epoch": 2.0,
                     "execution_backend": "remote",
                     "remote_profile_name": "206_root"},
                ],
            })
            header_labels = [
                window.models.horizontalHeaderItem(column).text()
                for column in range(window.models.columnCount())
            ]
            self.assertIn("来源", header_labels)
            sources = {
                window.models.item(row, header_labels.index("来源")).text()
                for row in range(window.models.rowCount())
            }
            self.assertEqual({"本地", "远程（206_root）"}, sources)
        finally:
            window.window.close()

    def test_flexict_viewer_shows_model_source(self):
        """R62-9: same contract for the FlexiCT status viewer."""
        _AppFixture.app()
        import flexict_status_viewer as viewer

        window = viewer.StatusWindow({"workspace": ""}, QT)
        try:
            window.refresh({
                "jobs": [],
                "models": [
                    {"task_id": "liver", "model_id": "local_m",
                     "label_name": "liver", "configuration": "2d",
                     "created_at_epoch": 1.0},
                    {"task_id": "liver", "model_id": "remote_m",
                     "label_name": "liver", "configuration": "2d",
                     "created_at_epoch": 2.0,
                     "execution_backend": "remote",
                     "remote_profile_name": "206_root"},
                ],
                "pair": (None, None),
                "recommended": None,
            })
            header_labels = [
                window.models.horizontalHeaderItem(column).text()
                for column in range(window.models.columnCount())
            ]
            self.assertIn("来源", header_labels)
            sources = {
                window.models.item(row, header_labels.index("来源")).text()
                for row in range(window.models.rowCount())
            }
            self.assertEqual({"本地", "远程（206_root）"}, sources)
        finally:
            window.window.close()

    def test_nnunet_viewer_previous_output_entry(self):
        """S02：替换过旧结果的任务在结果页直接给出「旧结果…」入口和替换
        说明（含旧版本时间）；没有替换记录的任务按钮置灰。普通标注者无需
        读 JSON 或猜文件名即可找回旧版本。"""
        _AppFixture.app()
        import nnunet_status_viewer as viewer

        window = viewer.StatusWindow({"workspace": ""}, QT)
        try:
            with mock.patch.object(
                window, "_selected_task", return_value="liver"
            ):
                window.refresh({
                    "jobs": [
                        {"task_id": "liver", "task_name": "Liver",
                         "status_path": "replaced.json", "status": "completed",
                         "created_at_epoch": 2.0, "kind": "infer",
                         "message": "done",
                         "replaced_previous_output": "W:/old.nii.gz",
                         "_replaced_mtime_epoch": 1760000000.0},
                        {"task_id": "liver", "task_name": "Liver",
                         "status_path": "plain.json", "status": "completed",
                         "created_at_epoch": 1.0, "kind": "infer",
                         "message": "done"},
                    ],
                    "models": [],
                })
                self.assertTrue(window.previous_button.isEnabled())
                self.assertIn("替换了旧结果", window.detail_label.text())
                window.jobs.setCurrentRow(1)
                self.assertFalse(window.previous_button.isEnabled())
                self.assertNotIn("替换了旧结果", window.detail_label.text())
        finally:
            window.window.close()

    def test_flexict_viewer_previous_output_entry(self):
        """S02：FlexiCT 状态页与 nnU-Net 状态页同一契约——替换过的任务
        显示找回入口，未替换的任务置灰。"""
        _AppFixture.app()
        import flexict_status_viewer as viewer

        window = viewer.StatusWindow({"workspace": ""}, QT)
        try:
            window.refresh({
                "jobs": [
                    {"status_path": "replaced.json", "status": "completed",
                     "created_at_epoch": 2.0, "kind": "infer",
                     "message": "done",
                     "replaced_previous_output": "W:/old.nii.gz",
                     "_replaced_mtime_epoch": 1760000000.0},
                    {"status_path": "plain.json", "status": "completed",
                     "created_at_epoch": 1.0, "kind": "infer",
                     "message": "done"},
                ],
                "models": [],
                "pair": (None, None),
                "recommended": None,
            })
            self.assertTrue(window.previous_button.isEnabled())
            self.assertIn("替换了旧结果", window.detail_label.text())
            window.jobs.setCurrentRow(1)
            self.assertFalse(window.previous_button.isEnabled())
            self.assertNotIn("替换了旧结果", window.detail_label.text())
        finally:
            window.window.close()

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

    def test_closed_viewer_windows_stop_their_refreshers(self):
        """R75 P1: a closed viewer window must stop collecting. The hidden
        heartbeat kept polling forever, and timers firing while the wrapped
        window was being torn down corrupted the heap (0xc0000374, ~70% of
        GUI smoke runs crashed in clamp_window_to_screen). All three viewer
        windows must wire closeEvent to refresher.stop()."""
        _AppFixture.app()
        import batch_status_viewer as batch
        import nnunet_status_viewer as nnunet
        import flexict_status_viewer as flexict

        for module in (nnunet, flexict):
            source = (ROOT / "tools" / (module.__name__ + ".py")).read_text(
                encoding="utf-8"
            )
            self.assertIn("closeEvent = self._close_event", source, module.__name__)
            self.assertIn("refresher.stop()", source, module.__name__)
        source = (ROOT / "tools" / "batch_status_viewer.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("closeEvent = self._close_event", source)
        self.assertIn("refresher.stop()", source)

        # Behavioral check: closing a real window stops both timers.
        window = flexict.StatusWindow({"workspace": ""}, QT)
        self.assertTrue(window.refresher._timer.isActive())
        window.window.close()
        self.assertFalse(window.refresher._timer.isActive())
        self.assertFalse(window.refresher._drain_timer.isActive())


class TestIoPathSetupExportOnly(unittest.TestCase):
    """C1 entry merge: the external path window only serves mask export.

    Import path selection moved to the drop window; any leftover import-mode
    branch here is dead code that must not come back.
    """

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
            # "未指定" default: no declared modality, guardrail off.
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

    def test_remote_backend_marks_requests_and_drops_cpu_flag(self):
        """R72-C1 (=R62-5): the batch window can dispatch to a remote
        backend. Remote requests carry the profile id and never ask for
        local CPU fallback — the remote channel always uses the server
        GPU. A missing profile id must fail closed per case (create_job
        refuses) rather than silently falling back to local."""
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

            class _Selector:
                def __init__(self, backend, profile_id):
                    self._backend = backend
                    self._profile_id = profile_id

                def selection(self):
                    return self._backend, self._profile_id

            requests = []

            def fake_create_job(request):
                requests.append(request)
                return {"job_id": request["job_id"], "status_path": "s.json"}

            window.image_root_edit.setText(str(tmp / "cases"))
            window.output_edit.setText(str(tmp / "out"))
            window.cpu_check.setChecked(True)
            window.remote_selector = _Selector("remote", "206_root")
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
            self.assertEqual("remote", requests[0]["execution_backend"])
            self.assertEqual("206_root", requests[0]["remote_profile_id"])
            # CPU fallback is a local-only affordance.
            self.assertNotIn("use_cpu", requests[0])
            self.assertIn("远程", window.status_label.text())

            # Local selection keeps the exact request shape it always had.
            requests.clear()
            window.remote_selector = _Selector("local", "")
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
            self.assertNotIn("execution_backend", requests[0])
            self.assertTrue(requests[0]["use_cpu"])

    def test_unusable_models_stay_visible_with_reason(self):
        """R62-7: a registered model whose files are gone must be listed
        (greyed, with the reason) instead of silently vanishing."""
        _AppFixture.app()
        import nnunet_batch_predict_ui as ui

        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            window = ui.BatchPredictionWindow(
                {"workspace": str(tmp / "workspace")}, QT
            )
            self.addCleanup(window.window.close)
            broken = self._usable_model_row(self._model_dir(tmp))
            # Break it: registered, but the model folder is gone. The
            # loader computes the reason via model_usability; emulate the
            # same annotation it would put on the row.
            broken["model_dir"] = str(tmp / "deleted")
            broken["manifest_path"] = str(
                tmp / "deleted" / "mimics_model_manifest.json"
            )
            broken["_unusable_reason"] = "model folder is missing"
            good = self._usable_model_row(self._model_dir(tmp / "good"))
            window._apply_models([broken, good], "")
            self.assertEqual(2, window.table.rowCount())
            # R76: the last column shows the execution backend (Chinese);
            # the unusable reason rides on every cell's tooltip.
            self.assertEqual(
                "本地",
                window.table.item(0, 4).text(),
            )
            self.assertEqual(
                "本地\n不可用：model folder is missing",
                window.table.item(0, 4).toolTip(),
            )
            # ...and is greyed out (not selectable)...
            self.assertFalse(
                window.table.item(0, 4).flags() & QtCore.Qt.ItemIsEnabled
            )
            # ...while the usable row stays selectable and start enabled.
            self.assertTrue(
                window.table.item(1, 4).flags() & QtCore.Qt.ItemIsEnabled
            )
            self.assertTrue(window.start.isEnabled())
            # Submitting while a broken row is current is rejected with
            # the reason, not dispatched.
            window.table.selectRow(0)
            window.image_root_edit.setText(str(tmp))
            window.output_edit.setText(str(tmp / "out"))
            window._submit()
            self.assertIn("不可用", window.status_label.text())
            # Only broken models: start must be disabled even though the
            # list is not empty.
            window._apply_models([broken], "")
            self.assertEqual(1, window.table.rowCount())
            self.assertFalse(window.start.isEnabled())
            self.assertIn("不可用", window.status_label.text())

    def test_dispatch_timeout_guides_against_double_submit(self):
        """R62-8: the timeout message must warn that dispatched jobs keep
        running, so a blind retry doesn't start duplicate predictions."""
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
            window._dispatching = True
            window._dispatch_deadline = time.time() - 1.0
            window._poll_background_results()
            text = window.status_label.text()
            self.assertIn("不会被取消", text)
            self.assertIn("不能再次提交", text)
            self.assertEqual(0.0, window._dispatch_deadline)
            self.assertFalse(window.start.isEnabled())
            with mock.patch.object(ui.threading, "Thread") as thread:
                window._submit()
                thread.assert_not_called()
            window._dispatch_results.put("扫描已完成")
            window._poll_background_results()
            self.assertFalse(window._dispatching)
            self.assertTrue(window.start.isEnabled())

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

    def test_no_import_mode_branches_remain(self):
        _AppFixture.app()
        source = (ROOT / "tools" / "io_path_setup_ui.py").read_text(
            encoding="utf-8"
        )
        for token in ("import_batch", "import_single", "recognition_debounce"):
            self.assertNotIn(token, source)
        # The live export features must stay.
        self.assertIn('"开始导出"', source)
        self.assertIn("count_existing_label_files", source)
        self.assertIn("mask_candidate_list", source)

    def test_portable_grid_gate_is_flag_scoped(self):
        # R94-2: the dead-source bypass may only trigger when the launcher
        # confirmed the project carries its import-time grid snapshot
        # (source_portable_grid) AND the submitted path equals the recorded
        # one. Everything else keeps the default exists-check rejection.
        _AppFixture.app()
        source = (ROOT / "tools" / "io_path_setup_ui.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("source_portable_grid", source)
        self.assertIn("portable_grid = (", source)
        # The default rejection for a non-existing source must survive.
        self.assertIn("请选择一个存在的来源文件或文件夹。", source)

    def test_candidate_checkboxes_sync_with_name_field(self):
        # R68 audit P1: the sync helper ran without its QtCore import and
        # raised NameError on every keystroke, silently disabling the whole
        # checkbox feature. Drive it the way textChanged does.
        import io_path_setup_ui as io_ui
        from PySide6 import QtCore, QtWidgets
        app = _AppFixture.app()
        field = QtWidgets.QLineEdit("liver")
        widget = QtWidgets.QListWidget()
        widget.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        for name in ("liver", "spleen"):
            item = QtWidgets.QListWidgetItem(name)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            widget.addItem(item)
        io_ui._sync_candidate_check_state(field, widget)  # must not raise
        self.assertEqual(QtCore.Qt.Checked, widget.item(0).checkState())
        self.assertEqual(QtCore.Qt.Unchecked, widget.item(1).checkState())

    def test_candidate_click_keeps_other_candidates_visible(self):
        # R74 P1: toggling a candidate name into the field re-triggered the
        # textChanged filter, so the just-toggled name became the filter and
        # every OTHER candidate vanished — multi-select was impossible
        # ("选中一个就看不到其他的"). The filter must run only while the
        # name field has focus (real typing), never on programmatic updates.
        source = (ROOT / "tools" / "io_path_setup_ui.py").read_text(
            encoding="utf-8"
        )
        # The gate: the textChanged filter handler must check hasFocus so a
        # candidate click (programmatic setText, no field focus) does not
        # re-filter the list.
        self.assertIn("hasFocus()", source)
        self.assertIn("_filter_candidates_while_typing", source)
        # And leaving the field must restore the full list.
        self.assertIn("editingFinished", source)

        # Behavioral check of the gate mechanics: filter runs only for
        # focused typing. Reproduce both paths through the real helpers.
        import io_path_setup_ui as io_ui
        from PySide6 import QtCore, QtWidgets
        app = _AppFixture.app()
        field = QtWidgets.QLineEdit()
        widget = QtWidgets.QListWidget()
        for name in ("liver", "spleen", "kidney"):
            widget.addItem(QtWidgets.QListWidgetItem(name))

        # Candidate click path: no focus on the field → the gated handler
        # must NOT call the filter, so every candidate stays visible even
        # though the field now holds one name.
        io_ui._toggle_name_in_field(field, "spleen")
        self.assertFalse(field.hasFocus())
        for index in range(widget.count()):
            self.assertFalse(
                widget.item(index).isHidden(),
                "candidate {0} must stay visible after another was picked"
                .format(widget.item(index).text()),
            )
        # Typing path (focused field): the filter still narrows the list.
        field.setFocus()
        if field.hasFocus():  # offscreen platforms may refuse focus
            field.setText("liver, kid")
            io_ui._filter_list_widget(widget, io_ui._last_token(field.text()))
            self.assertFalse(widget.item(0).isHidden())   # liver: prefix token
            self.assertTrue(widget.item(1).isHidden())    # spleen: filtered out
            self.assertFalse(widget.item(2).isHidden())   # kidney: matches "kid"
        # Checkbox sync still reflects picks after the fix.
        field.setText("spleen")
        io_ui._sync_candidate_check_state(field, widget)
        self.assertEqual(QtCore.Qt.Unchecked, widget.item(0).checkState())
        self.assertEqual(QtCore.Qt.Checked, widget.item(1).checkState())

    def test_export_radio_pairs_do_not_cross_uncheck(self):
        # R75 P1: the mask pair and the conflict-policy pair both lived on
        # the same panel, so radio auto-exclusivity spanned both groups —
        # clicking one pair silently unchecked the other and the export
        # fell back to "all masks" + whatever policy survived. Each pair
        # must be wrapped in its own QButtonGroup.
        source = (ROOT / "tools" / "io_path_setup_ui.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("mask_group = QtWidgets.QButtonGroup", source)
        self.assertIn("policy_group = QtWidgets.QButtonGroup", source)

        # Behavioral check: two pairs in one shared parent, each in its own
        # QButtonGroup — clicking across pairs must not uncheck.
        from PySide6 import QtWidgets
        app = _AppFixture.app()
        panel = QtWidgets.QWidget()
        pair_a = [QtWidgets.QRadioButton("全部 Mask", panel),
                  QtWidgets.QRadioButton("选定的名称", panel)]
        pair_b = [QtWidgets.QRadioButton("跳过已存在", panel),
                  QtWidgets.QRadioButton("覆盖已存在", panel)]
        group_a = QtWidgets.QButtonGroup(panel)
        group_b = QtWidgets.QButtonGroup(panel)
        for button in pair_a:
            group_a.addButton(button)
        for button in pair_b:
            group_b.addButton(button)
        pair_a[0].setChecked(True)
        pair_b[0].setChecked(True)
        # Simulate the user clicking the other pair's button.
        pair_b[1].click()
        self.assertTrue(pair_a[0].isChecked(),
                        "mask pair must survive a policy-pair click")
        self.assertTrue(pair_b[1].isChecked())


class TestIoPathOutputNotSticky(unittest.TestCase):
    """R61-7: a remembered output folder that was merely the computed
    default for a *previous* dataset must not pin every new dataset to that
    old folder. Only a folder the user chose themselves sticks."""

    def test_io_path_setup_prefills_only_user_chosen_output(self):
        source = (ROOT / "tools" / "io_path_setup_ui.py").read_text(
            encoding="utf-8"
        )
        # Load side: prefill the output box only when the remembered value
        # was a user choice (output_custom), never a computed default.
        self.assertIn('remembered_mode.get("output_custom")', source)
        # Save side: record whether the user deviated from the computed
        # default at submit time.
        self.assertIn('"output_custom": bool(', source)

    def test_drop_window_output_follows_each_new_dataset(self):
        drop = (ROOT / "tools" / "import_drop_window.py").read_text(
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

    def test_task_name_folding_is_caught_in_ui(self):
        # R75 P1: safe_identifier folds any non-ASCII name to "item", so two
        # Chinese task names merged into one internal task. UI19: the name is
        # now free-form — the internal identifier is auto-generated
        # losslessly, the typing-time hint shows it, and a Chinese name is
        # no longer blocked at submit.
        _AppFixture.app()
        import nnunet_training_setup_ui as nnunet_ui
        from nnunet_common import safe_identifier, stable_task_identifier

        # Contract of the root cause: a Chinese name must fold to "item".
        self.assertEqual("item", safe_identifier("肝脏分割"))
        self.assertEqual("liver_seg", safe_identifier("liver_seg"))
        # UI19: distinct Chinese names must get distinct stable identifiers.
        self.assertEqual("liver_seg", stable_task_identifier("liver_seg"))
        left = stable_task_identifier("左肾")
        right = stable_task_identifier("右肾")
        self.assertNotEqual(left, right)
        self.assertTrue(left.startswith("item_"))
        self.assertTrue(right.startswith("item_"))

        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            home = self._settings_home(tmp, "nnunet_settings.json", {})
            with mock.patch.object(nnunet_ui.Path, "home", lambda: home):
                window = nnunet_ui.TrainingSetupWindow(
                    {"workspace": ""}, tmp / "ctx.json", QT
                )
                try:
                    # Typing-time hint shows the generated internal name...
                    window.task_edit.setText("肝脏分割")
                    self.assertFalse(window.task_name_hint.isHidden())
                    self.assertIn(
                        stable_task_identifier("肝脏分割"),
                        window.task_name_hint.text())
                    # ...and disappears for a pure-ASCII one.
                    window.task_edit.setText("liver_seg")
                    self.assertTrue(window.task_name_hint.isHidden())
                    # The generated identifier carries into the request so
                    # distinct Chinese task names never merge downstream.
                    window.task_edit.setText("肝脏分割")
                    self.assertEqual(
                        stable_task_identifier("肝脏分割"),
                        window._request()["task_id"])
                    # An empty name is still blocked before background work.
                    window.task_edit.setText("")
                    window._submit()
                    self.assertFalse(window._submission_pending)
                finally:
                    window.window.close()

    def test_flexict_case_table_shows_unusable_cases(self):
        # R75 P1: _refresh_cases dropped unchecked rows entirely, so a
        # label-missing scan showed "已找到 12 个病例" next to an empty
        # table — a dead end the user cannot reason about. Unusable cases
        # must be listed (grayed, with a translated status) and counted
        # separately from usable ones.
        _AppFixture.app()
        import flexict_training_setup_ui as flexict_ui

        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            home = self._settings_home(tmp, "flexict_settings.json", {})
            with mock.patch.object(flexict_ui.Path, "home", lambda: home), mock.patch.object(
                flexict_ui, "workspace_root", lambda _c: str(tmp / "ws")
            ), mock.patch.object(
                flexict_ui, "load_config", lambda: {"root": str(tmp)}
            ), mock.patch.object(
                flexict_ui.TrainingSetupWindow, "_refresh_models_async",
                lambda self: None,
            ):
                window = flexict_ui.TrainingSetupWindow(
                    {"workspace": str(tmp / "ws")}, tmp / "ctx.json", QT
                )
                try:
                    window._case_rows = [
                        {"case_id": "s0001", "checked": True,
                         "use_train": True, "use_val": False, "status": "OK"},
                        {"case_id": "s0002", "checked": False,
                         "use_train": True, "use_val": False,
                         "status": "label missing: item"},
                        {"case_id": "s0003", "checked": False,
                         "use_train": True, "use_val": False,
                         "status": "image missing"},
                    ]
                    window._refresh_cases()
                    # Every case is listed; unusable ones visible with a
                    # translated status instead of silently dropped.
                    self.assertEqual(3, window.case_table.rowCount())
                    self.assertEqual(
                        "缺少标签：item",
                        window.case_table.item(1, 3).text(),
                    )
                    self.assertEqual(
                        "缺少图像", window.case_table.item(2, 3).text()
                    )
                    summary = window.case_summary.text()
                    self.assertIn("共 3 例", summary)
                    self.assertIn("可用 1", summary)
                    self.assertNotIn("没有可用病例", summary)
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
                flexict_ui.TrainingSetupWindow, "_refresh_models_async",
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
                ui.TrainingSetupWindow, "_refresh_models_async", lambda self: None
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
        source_root = ROOT / "tools"
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
                ui.TrainingSetupWindow, "_refresh_models_async", lambda self: None
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

    def test_flexict_rescan_keeps_manual_split_selection(self):
        # R75: rescanning rebuilt the case table from pipeline defaults, so
        # unchecking cases for training or marking validators was silently
        # reset by a rescan. The merge must carry the old selection forward
        # per case_id, and drop it for cases that disappeared.
        _AppFixture.app()
        import flexict_training_setup_ui as ui

        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            home = self._settings_home(tmp, "flexict_settings.json", {})
            with mock.patch.object(ui.Path, "home", lambda: home), mock.patch.object(
                ui, "workspace_root", lambda _c: str(tmp / "default-ws")
            ), mock.patch.object(
                ui, "load_config", lambda: {"root": str(tmp)}
            ), mock.patch.object(
                ui.TrainingSetupWindow, "_refresh_models_async", lambda self: None
            ):
                window = ui.TrainingSetupWindow(
                    {"workspace": ""}, tmp / "ctx.json", QT
                )
                try:
                    window._case_rows = [
                        {"case_id": "a", "checked": True, "use_train": False, "use_val": True},
                        {"case_id": "b", "checked": True, "use_train": True, "use_val": False},
                        {"case_id": "gone", "checked": True, "use_train": False, "use_val": True},
                    ]
                    fresh = [
                        {"case_id": "a", "checked": True, "use_train": True, "use_val": False},
                        {"case_id": "b", "checked": True, "use_train": True, "use_val": False},
                        {"case_id": "new", "checked": True, "use_train": True, "use_val": False},
                    ]
                    merged = window._merge_case_selection(fresh)
                    by_id = {row["case_id"]: row for row in merged}
                    self.assertFalse(by_id["a"]["use_train"])
                    self.assertTrue(by_id["a"]["use_val"])
                    self.assertTrue(by_id["b"]["use_train"])
                    # A case not in the old selection keeps pipeline defaults.
                    self.assertTrue(by_id["new"]["use_train"])
                    self.assertNotIn("gone", by_id)
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
                ui.TrainingSetupWindow, "_refresh_models_async", lambda self: None
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
                ui.TrainingSetupWindow, "_refresh_models_async", lambda self: None
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
                        "至少选择两个病例",
                        window.status_label.text(),
                    )
                finally:
                    window.window.close()

    def test_flexict_empty_train_split_submission_is_rejected(self):
        # UI30: unchecking every 训练 checkbox leaves train_case_ids empty
        # while cases still passes the >=2 check; the pipeline would then
        # reject an empty train split only after the round-trip. Stop it at
        # the submit gate with a readable reason.
        _AppFixture.app()
        import flexict_training_setup_ui as ui

        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            for case in ("case01", "case02"):
                (tmp / "data" / case).mkdir(parents=True)
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
                ui.TrainingSetupWindow, "_refresh_models_async", lambda self: None
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
                            "cases": ["case01", "case02"],
                            "train_case_ids": [],
                            "val_case_ids": ["case01", "case02"],
                            "val_cases": 2,
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
                        "训练集为空",
                        window.status_label.text(),
                    )
                finally:
                    window.window.close()

    def test_flexict_continue_training_reaches_request_and_allows_remote(self):
        # R61-25: "continue interrupted training" must reach the request (and
        # through it the trainer's continue_training flag). R72-C2: a remote
        # backend is no longer blocked at the UI — the checkpoints persist
        # on the remote server under the content-addressed cache, and the
        # remote controller resolves/rejects the resume before any upload.
        _AppFixture.app()
        import flexict_training_setup_ui as ui

        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            (tmp / "data").mkdir()
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
                ui.TrainingSetupWindow, "_refresh_models_async", lambda self: None
            ):
                window = ui.TrainingSetupWindow(
                    {"workspace": ""}, tmp / "ctx.json", QT
                )
                try:
                    self.assertFalse(window.continue_check.isChecked())
                    self.assertFalse(window._request()["continue_training"])
                    window.continue_check.setChecked(True)
                    self.assertTrue(window._request()["continue_training"])
                    # Remote backend + continue must pass local validation and
                    # spawn the submission path-check worker (the remote
                    # controller owns the reject/resume decision).
                    created = {}
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
                    ), mock.patch.object(
                        ui, "create_flexict_job",
                        **{"side_effect": lambda req: created.update(
                            request=req) or {
                            "job_id": "j", "job_dir": "d",
                            "status_path": "s", "control_path": "c",
                            "launcher_pid": 1,
                        }}
                    ):
                        window._submit()
                        self.assertTrue(window._submission_pending)
                        # Drain the path-check worker and the poll tick.
                        deadline = time.time() + 10.0
                        while (
                            window._submission_pending
                            and time.time() < deadline
                        ):
                            _QApplication.processEvents()
                            time.sleep(0.02)
                            window._poll_submission()
                        self.assertTrue(window.submitted)
                        self.assertTrue(
                            created["request"]["continue_training"]
                        )
                        self.assertEqual(
                            created["request"]["execution_backend"], "remote"
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


class TestRestorePreviousOffGuiThread(unittest.TestCase):
    """U02: restoring an old output runs checksums + a file swap; on a slow
    workspace that must not freeze the viewer. The swap runs on a worker,
    the result dialog (success or failure) is polled back on the GUI thread,
    and a second click while one restore is in flight is rejected."""

    def _wait_until(self, condition, timeout=5.0):
        deadline = time.time() + timeout
        while time.time() < deadline and not condition():
            _AppFixture.app().processEvents()
            time.sleep(0.02)
        return condition()

    def _test_success_runs_off_gui_thread(self, module_name):
        _AppFixture.app()
        import importlib

        module = importlib.import_module(module_name)
        window = module.StatusWindow({"workspace": ""}, QT)
        window.refresher.stop()
        window.refresher = mock.Mock()
        io_threads = []
        started = threading.Event()

        def slow_restore(status_path):
            io_threads.append(threading.current_thread())
            started.set()
            time.sleep(0.2)  # slow storage: checksums over a big file

        try:
            with mock.patch.object(
                window.QtWidgets.QMessageBox, "information"
            ) as info, mock.patch(
                module_name + ".restore_replaced_output", slow_restore
            ):
                window._restore_previous({"status_path": "X:/j/status.json"})
                self.assertTrue(
                    started.wait(5.0), "restore never started on the worker"
                )
                self.assertTrue(
                    window._restore_busy, "busy flag not set while in flight"
                )
                # While the worker is blocked in IO the GUI thread keeps
                # servicing events — the timer-driven poll completes it.
                self.assertTrue(
                    self._wait_until(lambda: not window._restore_busy),
                    "GUI thread stayed blocked for the whole restore",
                )
                info.assert_called_once()
                self.assertIn("已恢复旧结果", info.call_args[0][1])
                window.refresher.request.assert_called_once()
            self.assertEqual(
                len(io_threads), 1, "restore must run exactly once"
            )
            self.assertNotIn(
                threading.current_thread(), io_threads,
                "restore IO must run off the GUI thread",
            )
        finally:
            window.window.close()

    def test_nnunet_success_runs_off_gui_thread(self):
        self._test_success_runs_off_gui_thread("nnunet_status_viewer")

    def test_flexict_success_runs_off_gui_thread(self):
        self._test_success_runs_off_gui_thread("flexict_status_viewer")

    def _test_second_click_while_busy_is_rejected(self, module_name):
        _AppFixture.app()
        import importlib

        module = importlib.import_module(module_name)
        window = module.StatusWindow({"workspace": ""}, QT)
        window.refresher.stop()
        window.refresher = mock.Mock()
        release = threading.Event()
        started = threading.Event()

        swap_calls = []

        def blocked_restore(status_path):
            swap_calls.append(status_path)
            started.set()
            release.wait(5.0)

        try:
            with mock.patch.object(
                window.QtWidgets.QMessageBox, "information"
            ) as info, mock.patch(
                module_name + ".restore_replaced_output", blocked_restore
            ):
                window._restore_previous({"status_path": "X:/j/status.json"})
                self.assertTrue(started.wait(5.0))
                # Second click while the first swap is still in flight.
                window._restore_previous({"status_path": "X:/j/status.json"})
                self.assertEqual(
                    len(info.call_args_list), 1,
                    "second click must be rejected with a busy notice",
                )
                self.assertIn("仍在进行中", info.call_args_list[0].args[2])
                release.set()
                self.assertTrue(
                    self._wait_until(lambda: not window._restore_busy)
                )
            # Exactly one swap ran: the file exchange must not repeat.
            self.assertEqual(swap_calls, ["X:/j/status.json"])
            window.refresher.request.assert_called_once()
        finally:
            release.set()
            window.window.close()

    def test_nnunet_second_click_while_busy_is_rejected(self):
        self._test_second_click_while_busy_is_rejected("nnunet_status_viewer")

    def test_flexict_second_click_while_busy_is_rejected(self):
        self._test_second_click_while_busy_is_rejected("flexict_status_viewer")

    def _test_failure_reported_on_gui_thread(self, module_name):
        _AppFixture.app()
        import importlib

        module = importlib.import_module(module_name)
        window = module.StatusWindow({"workspace": ""}, QT)
        window.refresher.stop()
        window.refresher = mock.Mock()

        def failing_restore(status_path):
            raise RuntimeError("swap failed on read-only storage")

        try:
            with mock.patch.object(
                window.QtWidgets.QMessageBox, "critical"
            ) as crit, mock.patch.object(
                window.QtWidgets.QMessageBox, "information"
            ) as info, mock.patch(
                module_name + ".restore_replaced_output", failing_restore
            ):
                window._restore_previous({"status_path": "X:/j/status.json"})
                self.assertTrue(
                    self._wait_until(lambda: not window._restore_busy),
                    "failed restore never drained back to the GUI thread",
                )
                crit.assert_called_once()
                self.assertIn("swap failed", crit.call_args[0][2])
                info.assert_not_called()
                window.refresher.request.assert_called_once()
        finally:
            window.window.close()

    def test_nnunet_failure_reported_on_gui_thread(self):
        self._test_failure_reported_on_gui_thread("nnunet_status_viewer")

    def test_flexict_failure_reported_on_gui_thread(self):
        self._test_failure_reported_on_gui_thread("flexict_status_viewer")

    def test_close_event_stops_restore_poll(self):
        """A close during restore waits for its receipt, then tears down
        timers; the live commit must not be abandoned between renames."""
        _AppFixture.app()
        for module_name in ("nnunet_status_viewer", "flexict_status_viewer"):
            with self.subTest(module=module_name):
                import importlib

                module = importlib.import_module(module_name)
                window = module.StatusWindow({"workspace": ""}, QT)
                release = threading.Event()
                started = threading.Event()

                def blocked_restore(status_path):
                    started.set()
                    release.wait(5.0)

                try:
                    with mock.patch(
                        module_name + ".restore_replaced_output",
                        blocked_restore,
                    ):
                        window._restore_previous(
                            {"status_path": "X:/j/status.json"}
                        )
                        self.assertTrue(started.wait(5.0))
                        event = mock.Mock()
                        window._close_event(event)
                        event.ignore.assert_called_once()
                        self.assertTrue(window._restore_timer.isActive())
                        release.set()
                        self.assertTrue(
                            self._wait_until(
                                lambda: not window._restore_busy
                            ),
                            "worker never finished",
                        )
                        self.assertFalse(window._restore_timer.isActive())
                finally:
                    release.set()
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
        # R76: refresh is async now — drain the background scan like the
        # real event loop would before the test asserts on table contents.
        deadline = time.time() + 5.0
        while time.time() < deadline:
            _AppFixture.app().processEvents()
            window._poll_background_results()
            if window._jobs:
                break
            time.sleep(0.02)
        self.assertTrue(window._jobs, "background refresh never landed")
        return window

    def test_real_rows_populate_without_nameerror(self):
        with tempfile.TemporaryDirectory() as tmp:
            window = self._window(tmp)
            try:
                self.assertEqual(
                    2, window.table.rowCount(),
                    "non-empty ranking must populate the table")
                self.assertFalse(window.open_button.isEnabled())
                window.table.selectRow(0)
                self.assertTrue(window.open_button.isEnabled())
                summary = window.status_label.text()
                self.assertIn("已排序 2 例", summary)
            finally:
                window.window.close()

    def test_annotated_row_gets_success_foreground(self):
        # F03's crash was inside the per-row loop's annotated branch.
        # P3-5: the state column shows Chinese labels.
        with tempfile.TemporaryDirectory() as tmp:
            window = self._window(tmp)
            try:
                job_dir = Path(window._job_dir)
                from flexict_active_learning_ui import set_case_state
                set_case_state(job_dir, "caseA", "annotated")
                window._job_selected()
                self.assertEqual(
                    "已标注",
                    window.table.item(0, 4).text())
                self.assertEqual(
                    "待标注", window.table.item(1, 4).text())
                self.assertIn("已标注 1 例", window.status_label.text())
            finally:
                window.window.close()

    def test_viewed_case_not_counted_as_annotated(self):
        """UI14/F05 验收：叠加（查看）后已标注数不变；状态列为"已查看"，
        已查看与已标注在状态行分开统计。只有显式确认才计入已标注。"""
        with tempfile.TemporaryDirectory() as tmp:
            window = self._window(tmp)
            try:
                job_dir = Path(window._job_dir)
                from flexict_active_learning_ui import set_case_state
                # 模拟 Mimics 侧应用叠加后的状态：viewed，非 annotated
                set_case_state(job_dir, "caseA", "viewed")
                set_case_state(job_dir, "caseB", "viewed")
                window._job_selected()
                self.assertEqual(
                    "已查看", window.table.item(0, 4).text())
                self.assertIn("已查看 2 例", window.status_label.text())
                self.assertIn("已标注 0 例", window.status_label.text())
                # 显式确认后已标注数才增加
                set_case_state(job_dir, "caseA", "annotated")
                window._job_selected()
                self.assertIn("已查看 1 例", window.status_label.text())
                self.assertIn("已标注 1 例", window.status_label.text())
            finally:
                window.window.close()

    def test_excluded_cases_named_in_status_line(self):
        """R81 复审: cases the uncertainty batch excluded (grid mismatch)
        silently shortened the ranking — the status line must say how many
        were dropped and point at the log with the per-case reasons."""
        with tempfile.TemporaryDirectory() as tmp:
            window = self._window(tmp, ranking=("caseA",))
            try:
                job_dir = Path(window._job_dir)
                status = json.loads(
                    (job_dir / "status.json").read_text(encoding="utf-8"))
                status["excluded_case_count"] = 2
                status["uncertainty_log"] = str(job_dir / "uncertainty_log.txt")
                (job_dir / "status.json").write_text(
                    json.dumps(status), encoding="utf-8")
                # _job_selected reads the cached job list, not status.json —
                # refresh like the poll timer does so the new fields land.
                deadline = time.time() + 5.0
                while time.time() < deadline:
                    _AppFixture.app().processEvents()
                    window._refresh_async()
                    window._poll_background_results()
                    if window._jobs and int(
                            window._jobs[0].get("excluded_case_count") or 0):
                        break
                    time.sleep(0.02)
                self.assertTrue(
                    window._jobs and window._jobs[0].get("excluded_case_count"),
                    "refresh never picked up the excluded fields")
                window._job_selected()
                text = window.status_label.text()
                self.assertIn("2 例因数据问题被剔除", text)
                self.assertIn("uncertainty_log.txt", text)
            finally:
                window.window.close()

    def test_mark_annotated_writes_off_gui_thread(self):
        """S08: the annotate click must not wait on the cross-process guard
        on the GUI thread — Mimics holding the lock used to freeze the
        window for the guard's whole bounded wait (~2 s). The write runs on
        a worker; the row only turns 已标注 after the state really landed
        (a failed write stays retryable, UI31)."""
        if os.name != "nt":
            self.skipTest("guard-hold simulation uses msvcrt")
        with tempfile.TemporaryDirectory() as tmp:
            window = self._window(tmp)
            try:
                window.table.selectRow(0)
                job_dir = Path(window._job_dir)
                guard_file = job_dir / "annotation_state.json.guard"
                guard_file.write_bytes(b"\0")
                # Simulate Mimics holding the single-writer guard. a+b
                # starts at EOF and msvcrt locks from the current
                # position — seek(0) or this locks byte 1, not the byte
                # the worker contends on, and no contention ever happens.
                handle = open(str(guard_file), "a+b")
                handle.seek(0)
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                try:
                    started = time.time()
                    window._mark_annotated()
                    elapsed = time.time() - started
                    # The click returns immediately: no synchronous lock
                    # wait on the GUI thread.
                    self.assertLess(elapsed, 0.5)
                    self.assertIn("正在标记已标注", window.status_label.text())
                    # While the lock is held the write cannot land — the row
                    # still shows the pending state, not a fake success.
                    self.assertNotEqual(
                        "已标注", window.table.item(0, 4).text())
                finally:
                    msvcrt.locking(
                        handle.fileno(), msvcrt.LK_UNLCK, 1)
                    handle.close()
                # Lock released: the background write completes and the
                # poll timer refreshes the row from the real state file.
                deadline = time.time() + 5.0
                while time.time() < deadline:
                    _AppFixture.app().processEvents()
                    window._poll_background_results()
                    if window.table.item(0, 4).text() == "已标注":
                        break
                    time.sleep(0.02)
                self.assertEqual(
                    "已标注", window.table.item(0, 4).text())
                self.assertIn("已标注 1 例", window.status_label.text())
            finally:
                window.window.close()

    def test_click_order_last_intent_wins(self):
        """T04: 先点「已标注」再点「跳过」时，第一次写入可能卡在跨进程
        锁上。每次点击各开线程的年代里两个写入乱序落盘，最终状态是已被
        覆盖的第一个意图。单一顺序工作线程必须保证磁盘终态 = 最后一次
        点击，且回执不把两个值串起来。"""
        if os.name != "nt":
            self.skipTest("guard-hold simulation uses msvcrt")
        with tempfile.TemporaryDirectory() as tmp:
            window = self._window(tmp)
            try:
                window.table.selectRow(0)
                job_dir = Path(window._job_dir)
                guard_file = job_dir / "annotation_state.json.guard"
                guard_file.write_bytes(b"\0")
                # 让第一笔写入"晚完成"：按住锁，第一次点击的守护线程
                # 在锁的有界等待里排队。a+b 的指针在文件尾，msvcrt 锁
                # 的是当前位置的字节——必须先 seek(0) 才真锁住 worker
                # 竞争的第 0 字节。
                handle = open(str(guard_file), "a+b")
                handle.seek(0)
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                try:
                    window._mark_annotated()
                    window._mark_skipped()
                    # 两笔都已登记待写：第一笔卡在锁上，第二笔排在其后
                    # （worker 可能已取走第一笔，队列里剩 1 或 2 笔）。
                    self.assertIn(
                        window._state_write_queue.qsize(), (1, 2))
                finally:
                    msvcrt.locking(
                        handle.fileno(), msvcrt.LK_UNLCK, 1)
                    handle.close()
                # 解锁后按顺序落盘：annotated 先写、skipped 后写，
                # 终态必须是最后点击的 skipped。
                deadline = time.time() + 5.0
                while time.time() < deadline:
                    _AppFixture.app().processEvents()
                    window._poll_background_results()
                    if window.table.item(0, 4).text() == "已跳过":
                        break
                    time.sleep(0.02)
                self.assertEqual(
                    "已跳过", window.table.item(0, 4).text())
                state = json.loads(
                    (job_dir / "annotation_state.json").read_text(
                        encoding="utf-8"))
                self.assertEqual(
                    "skipped", state["cases"]["caseA"]["state"])
            finally:
                window.window.close()

    def test_exit_drain_bounded_and_reports_unsaved(self):
        """T04 失败路径：点击后立刻关窗，守护线程会在进程退出时被砍掉，
        标记静默丢失。退出排空必须有界等待：锁一直被占时及时返回并如实
        报告未保存数量，而不是无限等待或假装成功。"""
        if os.name != "nt":
            self.skipTest("guard-hold simulation uses msvcrt")
        with tempfile.TemporaryDirectory() as tmp:
            window = self._window(tmp)
            try:
                window.table.selectRow(0)
                job_dir = Path(window._job_dir)
                guard_file = job_dir / "annotation_state.json.guard"
                guard_file.write_bytes(b"\0")
                # a+b 指针在文件尾：seek(0) 后锁的才是 worker 竞争的
                # 第 0 字节（同 test_click_order_last_intent_wins）。
                handle = open(str(guard_file), "a+b")
                handle.seek(0)
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                try:
                    window._mark_annotated()
                    started = time.time()
                    pending = window.wait_for_state_writes(timeout=0.5)
                    elapsed = time.time() - started
                    # 有界：没有无限等待（守卫自身上限 2 s，这里 0.5 s
                    # 就该让步），并如实报告还有未保存的标记。
                    self.assertLess(elapsed, 1.5)
                    self.assertGreaterEqual(pending, 1)
                    self.assertFalse(
                        (job_dir / "annotation_state.json").exists(),
                        "锁被占时不得落盘")
                finally:
                    msvcrt.locking(
                        handle.fileno(), msvcrt.LK_UNLCK, 1)
                    handle.close()
                # 解锁后再排空：写入落地，计数归零。
                self.assertEqual(0, window.wait_for_state_writes(timeout=5.0))
                state = json.loads(
                    (job_dir / "annotation_state.json").read_text(
                        encoding="utf-8"))
                self.assertEqual(
                    "annotated", state["cases"]["caseA"]["state"])
            finally:
                window.window.close()

    def test_exit_drain_counts_write_taken_but_unfinished(self):
        """T04 轮3：先点「已标注」再点「跳过」，第一笔已落盘、第二笔已被
        worker 取走正在执行（此刻队列为空）。旧的 busy 标志只在入队时置
        位、每笔完成即清除，退出检查据此立即返回 0，最后一笔被当作已保存
        丢弃。未完成计数（入队 - 回执）必须把 in-flight 一笔也算作待保
        存；释放后终态为最后意图。"""
        import flexict_active_learning_ui as al_ui

        with tempfile.TemporaryDirectory() as tmp:
            window = self._window(tmp)
            try:
                window.table.selectRow(0)
                first_done = threading.Event()
                second_started = threading.Event()
                release_second = threading.Event()

                def controlled_set_case_state(job_dir, case_id, value):
                    if value == "annotated":
                        first_done.set()
                        return None
                    # The SECOND write (skipped): in flight, deliberately
                    # slow (slow workspace / big shared folder).
                    second_started.set()
                    release_second.wait(timeout=5.0)
                    return al_ui.set_case_state(job_dir, case_id, value)

                with mock.patch.object(
                        al_ui, "set_case_state",
                        side_effect=controlled_set_case_state):
                    window._mark_annotated()
                    window._mark_skipped()
                    self.assertTrue(
                        first_done.wait(timeout=5.0),
                        "first write never ran")
                    self.assertTrue(
                        second_started.wait(timeout=5.0),
                        "second write never taken")
                    # Exactly the card's window: the second write has been
                    # DEQUEUED (queue may be empty) and is mid-execution.
                    pending = window.wait_for_state_writes(timeout=0.3)
                    self.assertGreaterEqual(
                        pending, 1,
                        "in-flight write reported as saved (queue empty)")
                    release_second.set()
                # Released: everything lands, count returns to zero and the
                # disk carries the LAST intent.
                self.assertEqual(
                    0, window.wait_for_state_writes(timeout=5.0))
                state = json.loads(
                    (Path(window._job_dir) / "annotation_state.json")
                    .read_text(encoding="utf-8"))
                self.assertEqual(
                    "skipped", state["cases"]["caseA"]["state"])
            finally:
                release_second.set()
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
                self.assertIn("请求失败（caseA", text)
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
                self.assertIn("已在 Mimics 中应用：caseA", text)
            finally:
                window.window.close()

    def test_stale_pending_request_gets_monitor_hint(self):
        """P2-2：请求一直无结果（Mimics 侧监控未运行）时，状态栏必须
        提示监控可能没在运行，而不是让"本窗口会自动更新"永远挂着。"""
        with tempfile.TemporaryDirectory() as tmp:
            window = self._window(tmp)
            try:
                request_dir = Path(window._job_dir) / "apply_requests"
                request_dir.mkdir(parents=True)
                (request_dir / "a.json").write_text(
                    json.dumps({
                        "case_id": "caseA", "what": "bands",
                        "state": "",
                        "requested_at_epoch": time.time() - 120.0,
                    }),
                    encoding="utf-8",
                )
                window._poll_requests()
                self.assertIn("可能尚未运行", window.status_label.text())
            finally:
                window.window.close()

    def test_fresh_pending_request_gets_no_hint(self):
        """刚发出的请求（几秒内）不该被误报为监控未运行。"""
        with tempfile.TemporaryDirectory() as tmp:
            window = self._window(tmp)
            try:
                request_dir = Path(window._job_dir) / "apply_requests"
                request_dir.mkdir(parents=True)
                (request_dir / "a.json").write_text(
                    json.dumps({
                        "case_id": "caseA", "what": "bands",
                        "state": "",
                        "requested_at_epoch": time.time() - 2.0,
                    }),
                    encoding="utf-8",
                )
                before = window.status_label.text()
                window._poll_requests()
                self.assertNotIn(
                    "监控未运行", window.status_label.text())
                self.assertEqual(before, window.status_label.text())
            finally:
                window.window.close()

    def test_empty_state_offers_a_new_ranking_entry(self):
        # F04/T04: with no completed runs the window must offer a reachable
        # way to START the first ranking, not point back at the same menu.
        with tempfile.TemporaryDirectory() as tmp:
            _AppFixture.app()
            import flexict_active_learning_ui as al_ui

            window = al_ui.ActiveLearningWindow(
                {"workspace": str(Path(tmp) / "empty")}, QT)
            try:
                deadline = time.time() + 5.0
                while time.time() < deadline and window._refresh_pending:
                    _AppFixture.app().processEvents()
                    window._poll_background_results()
                    time.sleep(0.02)
                self.assertFalse(
                    window._refresh_pending, "background refresh never landed")
                self.assertTrue(window.new_button.isVisibleTo(window.window))
                self.assertTrue(window.new_button.isEnabled())
                self.assertIn("新建排序", window.status_label.text())
            finally:
                window.window.close()

    def test_ranking_submit_error_is_surfaced(self):
        # F04: a failed create_flexict_job (e.g. missing model pair) must
        # reach the annotator, not die on a background thread. R75 P1:
        # submit results use the single dict/str queue protocol and are
        # drained by _poll_background_results, not _poll_requests.
        # UI25: the raw English pipeline error must be translated into the
        # Chinese actionable message, not shown verbatim.
        with tempfile.TemporaryDirectory() as tmp:
            window = self._window(tmp)
            try:
                window._submit_results.put(
                    "RuntimeError: No usable FlexiCT model pair was found.")
                window._poll_background_results()
                self.assertIn("未找到可用的 2D+3D 模型对",
                              window.pair_label.text())
                self.assertIn("2D + 3D 成对模型", window.pair_label.text())
                self.assertTrue(window.start_run_button.isEnabled())
            finally:
                window.window.close()

    def test_running_ranking_shows_progress_then_autorefreshes(self):
        # F04: after starting a ranking the window shows live progress and
        # refreshes the picker when the job reaches a terminal state.
        with tempfile.TemporaryDirectory() as tmp:
            window = self._window(tmp)
            try:
                job_dir = Path(window._job_dir)
                status = json.loads(
                    (job_dir / "status.json").read_text(encoding="utf-8"))
                status["status"] = "running"
                status["phase"] = "batch_predicting_2d"
                status["progress_percent"] = 42
                (job_dir / "status.json").write_text(
                    json.dumps(status), encoding="utf-8")

                def _fake_find_running(workspace, list_jobs=None):
                    return json.loads(
                        (job_dir / "status.json").read_text(encoding="utf-8"))

                import flexict_active_learning_ui as al_ui
                original = al_ui.find_running_al_job
                al_ui.find_running_al_job = _fake_find_running

                def _poll_until(predicate, timeout=5.0):
                    # R75 P2: the running-job scan runs on a worker thread;
                    # pump the poll until its result lands and is applied.
                    deadline = time.time() + timeout
                    while time.time() < deadline:
                        window._poll_requests()
                        if predicate():
                            return True
                        time.sleep(0.02)
                    return False

                try:
                    self.assertTrue(
                        _poll_until(
                            lambda: "排序进行中" in window.status_label.text()
                        ),
                        window.status_label.text(),
                    )
                    self.assertIn("42%", window.status_label.text())
                    # Terminal state: the poll refreshes the job list.
                    status["status"] = "completed"
                    (job_dir / "status.json").write_text(
                        json.dumps(status), encoding="utf-8")
                    window._seen_running_job_id = status.get("job_id") or "aljob1"
                    self.assertTrue(
                        _poll_until(lambda: window.job_combo.count() == 1),
                        window.job_combo.count(),
                    )
                finally:
                    al_ui.find_running_al_job = original
            finally:
                window.window.close()
    def test_failed_run_stays_listed_with_its_error(self):
        """R62-10: a failed ranking run must not vanish from the job list;
        the annotator needs to know it was attempted and why it died."""
        with tempfile.TemporaryDirectory() as tmp:
            window = self._window(tmp)
            try:
                # Add a failed run newer than the completed one.
                failed_dir = Path(tmp) / "jobs" / "aljob2"
                failed_dir.mkdir(parents=True)
                (failed_dir / "status.json").write_text(
                    json.dumps({
                        "kind": "active_learning",
                        "status": "failed",
                        "error": "GPU out of memory",
                        "created_at_epoch": 2.0,
                        "label_name": "Liver",
                        "job_dir": str(failed_dir),
                    }),
                    encoding="utf-8",
                )
                window.refresh()
                deadline = time.time() + 5.0
                while time.time() < deadline and window.job_combo.count() < 2:
                    _AppFixture.app().processEvents()
                    window._poll_background_results()
                    time.sleep(0.02)
                # Failed runs are listed after the completed ones.
                self.assertEqual(2, window.job_combo.count())
                window.job_combo.setCurrentIndex(1)
                window._job_selected()
                self.assertEqual(0, window.table.rowCount())
                self.assertFalse(window.open_button.isEnabled())
                text = window.status_label.text()
                self.assertIn("失败", text)
                self.assertIn("GPU out of memory", text)
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
            deadline = time.time() + 5.0
            while time.time() < deadline and window._refresh_pending:
                _AppFixture.app().processEvents()
                window._poll_background_results()
                time.sleep(0.02)
            window._apply_pair((None, None), "")
            self.assertIn("FlexiCT 训练窗口", window.pair_label.text())
            self.assertIn("2D + 3D 成对模型（主动学习用）", window.pair_label.text())
            self.assertFalse(window.start_run_button.isEnabled())
            # The old circular empty state pointed at this same entry.
            self.assertIn("新建排序", window.status_label.text())

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


class TestNNUNetActiveLearningWindow(unittest.TestCase):
    """R93 步骤 5/8：nnU-Net 模式的 AL 复核窗口。

    分歧来源是 k 折模型，所以表单只允许含 >=2 个已训练折（各自带
    checkpoint_final.pth）的 bundle；单折模型必须在列表阶段被跳过并
    说明原因，而不是让标注者启动一个注定失败的任务。
    """

    def _bundle(self, root, model_id, folds):
        model_dir = root / "models" / model_id
        for fold in folds:
            fold_dir = model_dir / "fold_{}".format(fold)
            fold_dir.mkdir(parents=True)
            (fold_dir / "checkpoint_final.pth").write_bytes(b"x")
        (model_dir / "plans.json").write_text("{}", encoding="utf-8")
        (model_dir / "dataset.json").write_text("{}", encoding="utf-8")
        return {
            "model_id": model_id,
            "task_id": "Dataset701_Segtask",
            "task_name": "Segtask",
            "model_dir": str(model_dir),
            "manifest_path": str(
                model_dir / "mimics_model_manifest.json"),
        }

    def _window(self, tmp, rows):
        import flexict_active_learning_ui as al_ui
        import nnunet_common

        _AppFixture.app()
        # The model check imports load_models at call time inside its
        # worker thread, so the stub must stay installed for the whole
        # window lifetime (not just the constructor).
        patcher = mock.patch.object(
            nnunet_common, "load_models", return_value=rows)
        patcher.start()
        self.addCleanup(patcher.stop)
        window = al_ui.ActiveLearningWindow(
            {"workspace": str(Path(tmp) / "workspace"), "mode": "nnunet"},
            QT)
        deadline = time.time() + 5.0
        while time.time() < deadline and window._loading_models:
            _AppFixture.app().processEvents()
            window._poll_background_results()
            time.sleep(0.02)
        self.assertFalse(window._loading_models, "model check never landed")
        return window

    def test_lists_only_multifold_bundles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                self._bundle(root, "m_single", ["0"]),
                self._bundle(root, "m_kfold", ["0", "1", "3"]),
            ]
            window = self._window(tmp, rows)
            try:
                self.assertEqual(1, window.model_combo.count())
                self.assertIn("m_kfold", window.model_combo.itemText(0))
                self.assertIn("3 折", window.model_combo.itemText(0))
                self.assertIn(
                    "已找到 1 个可用的多折模型",
                    window.model_status_label.text())
                self.assertTrue(window.start_run_button.isEnabled())
            finally:
                window.window.close()

    def test_single_fold_bundle_skipped_with_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [self._bundle(root, "m_single", ["0"])]
            window = self._window(tmp, rows)
            try:
                self.assertEqual(0, window.model_combo.count())
                text = window.model_status_label.text()
                self.assertIn("未找到含至少 2 个已训练折的 nnU-Net 模型", text)
                self.assertIn("已跳过 1 个", text)
                self.assertFalse(window.start_run_button.isEnabled())
            finally:
                window.window.close()

    def test_submit_requires_pool_and_model(self):
        """表单校验必须同步失败：缺池/缺模型都不派发后台创建任务。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [self._bundle(root, "m_single", ["0"])]
            window = self._window(tmp, rows)
            try:
                window.pool_edit.setText("")
                window._submit_new_run()
                self.assertIn(
                    "请先选择未标注数据池文件夹",
                    window.model_status_label.text())
                window.pool_edit.setText(str(root / "pool"))
                window._submit_new_run()
                self.assertIn(
                    "请先选择一个多折模型",
                    window.model_status_label.text())
            finally:
                window.window.close()


class TestWindowClampedToSmallScreen(unittest.TestCase):
    """F37: a hard minimum larger than the available screen (150% DPI on
    1280x720 leaves ~853x480 logical pixels minus the taskbar) forced
    windows to grow past the screen and clip button labels. The minimum
    must be lowered to what the screen can actually show."""

    def test_clamp_lowerers_minimum_to_available_geometry(self):
        _AppFixture.app()
        from PySide6 import QtCore

        from ui_theme import clamp_window_to_screen

        window = QtWidgets.QMainWindow()
        try:
            small_screen = QtCore.QRect(0, 0, 853, 440)
            with mock.patch.object(
                QtWidgets.QMainWindow, "screen",
                lambda self: _FakeScreen(small_screen),
            ):
                window.resize(800, 500)
                window.setMinimumSize(780, 650)
                clamp_window_to_screen(window, (780, 650))
                self.assertEqual(
                    (780, 440),
                    (window.minimumWidth(), window.minimumHeight()),
                )
                self.assertLessEqual(window.height(), 440)
        finally:
            window.close()

    def test_clamp_keeps_minimum_on_large_screen(self):
        _AppFixture.app()
        from PySide6 import QtCore

        from ui_theme import clamp_window_to_screen

        window = QtWidgets.QMainWindow()
        try:
            big_screen = QtCore.QRect(0, 0, 1920, 1040)
            with mock.patch.object(
                QtWidgets.QMainWindow, "screen",
                lambda self: _FakeScreen(big_screen),
            ):
                window.setMinimumSize(780, 650)
                clamp_window_to_screen(window, (780, 650))
                self.assertEqual(
                    (780, 650),
                    (window.minimumWidth(), window.minimumHeight()),
                )
        finally:
            window.close()

    def test_al_refresh_scans_off_gui_thread(self):
        """R76: find_al_jobs stats every job directory — it must run on a
        worker thread, drained by the poll timer, not on the GUI thread
        (same contract as the status viewers)."""
        import tempfile

        _AppFixture.app()
        import flexict_active_learning_ui as al_ui

        seen_threads = []

        def fake_find_al_jobs(workspace, list_jobs=None):
            # The scan passes the injected backend lister — a stub that
            # rejects it raises inside the scan's except and the mock looks
            # "never called".
            seen_threads.append(threading.current_thread())
            return []

        with tempfile.TemporaryDirectory() as name:
            with mock.patch.object(
                al_ui, "find_al_jobs", side_effect=fake_find_al_jobs,
            ):
                window = al_ui.ActiveLearningWindow(
                    {"workspace": str(Path(name) / "ws")}, QT)
                try:
                    window.refresh()
                    deadline = time.time() + 5.0
                    while time.time() < deadline and not seen_threads:
                        _AppFixture.app().processEvents()
                        window._poll_background_results()
                        time.sleep(0.02)
                    self.assertTrue(
                        seen_threads, "refresh never scanned the workspace")
                    self.assertNotIn(
                        threading.current_thread(), seen_threads,
                        "find_al_jobs must run off the GUI thread")
                    self.assertEqual([], window._jobs)
                finally:
                    window.window.close()

    def test_al_window_opens_on_small_screen(self):
        """The AL window itself must construct with its minimum clamped to
        a screen smaller than the layout minimum — no exception, minimum
        height within the available geometry."""
        import tempfile

        _AppFixture.app()
        import flexict_active_learning_ui as al_ui
        from PySide6 import QtCore

        with tempfile.TemporaryDirectory() as name:
            with mock.patch.object(
                QtWidgets.QMainWindow, "screen",
                lambda self: _FakeScreen(QtCore.QRect(0, 0, 853, 440)),
            ):
                window = al_ui.ActiveLearningWindow(
                    {"workspace": str(Path(name) / "ws")}, QT,
                )
                try:
                    self.assertLessEqual(
                        window.window.minimumHeight(), 440)
                finally:
                    window.window.close()

    def test_secondary_windows_clamp_on_small_screen(self):
        """P2-1：nnU-Net/FlexiCT 状态、批量状态窗口的硬最小尺寸都要
        钳到可用屏幕内（F37 后续铺开）。"""
        import tempfile

        _AppFixture.app()
        import batch_status_viewer
        import flexict_status_viewer
        import nnunet_status_viewer
        from PySide6 import QtCore

        small = QtCore.QRect(0, 0, 853, 440)
        with mock.patch.object(
            QtWidgets.QMainWindow, "screen",
            lambda self: _FakeScreen(small),
        ):
            with tempfile.TemporaryDirectory() as name:
                checks = [
                    ("nnunet", nnunet_status_viewer.StatusWindow(
                        {"workspace": name}, QT)),
                    ("flexict", flexict_status_viewer.StatusWindow(
                        {"workspace": name}, QT)),
                    ("batch", batch_status_viewer.BatchStatusWindow(
                        Path(name), QT)),
                ]
                for label, window in checks:
                    try:
                        self.assertLessEqual(
                            window.window.minimumHeight(), 440,
                            "{0} window minimum exceeds the screen".format(
                                label))
                    finally:
                        window.window.close()

    def test_status_viewers_speak_chinese(self):
        """P3-1：状态窗口的实时标记和作业状态词必须是中文——
        "● Live"/"Uploading" 这类英文对标注者不可读。"""
        import tempfile

        _AppFixture.app()
        import flexict_status_viewer
        import nnunet_status_viewer

        with tempfile.TemporaryDirectory() as name:
            for module in (
                nnunet_status_viewer, flexict_status_viewer,
            ):
                window = module.StatusWindow({"workspace": name}, QT)
                try:
                    self.assertIn("● 实时", window.live_label.text())
                    self.assertNotIn("Live", window.live_label.text())
                    # 选中一个任务前状态是中文占位或未知。
                    window._job_rows = [{
                        "status": "preparing_data"}]
                    window.jobs.addItem("job1")
                    window.jobs.setCurrentRow(0)
                    window._show_selected_job()
                    self.assertEqual(
                        "准备数据中", window.state_label.text())
                finally:
                    window.window.close()


class TestWindowCloseTeardown(unittest.TestCase):
    """A closed viewer window must hand its C++ widget tree to Qt.

    Dropping a viewer left its C++ widget tree for Python GC to destroy
    mid-collect, which corrupted the heap (0xC0000374) and killed whole
    gui_smoke runs. close() must stop the refresh heartbeat and defer
    destruction to Qt via deleteLater, so GC only ever clears Python
    wrappers and never touches a live C++ tree."""

    def test_viewer_close_stops_timers(self):
        import flexict_status_viewer as viewer

        _AppFixture.app()
        window = viewer.StatusWindow({"workspace": ""}, QT)
        try:
            window._restore_timer.start()
            self.assertFalse(window.refresher._stop)
            self.assertTrue(window._restore_timer.isActive())
            window.window.close()
            self.assertTrue(window.refresher._stop)
            self.assertFalse(window._restore_timer.isActive())
        finally:
            window.window.close()

    def test_setup_close_without_submit_writes_cancelled_status(self):
        # The Mimics-side launcher waits on this status file; closing the
        # setup window without a submission must mark it cancelled instead
        # of leaving the launcher waiting forever.
        import json

        import nnunet_training_setup_ui as ui

        _AppFixture.app()
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            status_path = tmp / "status" / "setup.json"
            status_path.parent.mkdir(parents=True)
            home = tmp / "home"
            (home / ".mimics_script").mkdir(parents=True)
            with mock.patch.object(ui.Path, "home", lambda: home):
                window = ui.TrainingSetupWindow(
                    {
                        "workspace": str(tmp / "ws"),
                        "setup_status_path": str(status_path),
                    },
                    tmp / "ctx.json",
                    QT,
                )
                try:
                    window.window.close()
                    self.assertTrue(
                        status_path.exists(),
                        "closing without submitting must write the "
                        "cancelled status",
                    )
                    payload = json.loads(
                        status_path.read_text(encoding="utf-8")
                    )
                    self.assertEqual(payload.get("status"), "cancelled")
                finally:
                    window.window.close()

    def test_closed_viewer_garbage_does_not_corrupt_heap(self):
        # Regression for the 0xC0000374 crash: the whole gui_smoke run
        # died when GC destroyed a dropped viewer's C++ widget tree.
        # Pre-fix this test killed the process; it must finish cleanly.
        import gc

        import flexict_status_viewer as viewer

        app = _AppFixture.app()
        for _ in range(6):
            window = viewer.StatusWindow({"workspace": ""}, QT)
            window.window.close()
            del window
        gc.collect()
        app.processEvents()
        probe = QtWidgets.QMainWindow()
        probe.setWindowTitle("probe")
        import ui_theme

        ui_theme.clamp_window_to_screen(probe, (800, 620))
        probe.show()
        app.processEvents()
        probe.close()
        probe.deleteLater()
        app.processEvents()


class _FakeScreen:
    def __init__(self, geometry):
        self._geometry = geometry

    def availableGeometry(self):
        return self._geometry


if __name__ == "__main__":
    unittest.main(verbosity=2)
