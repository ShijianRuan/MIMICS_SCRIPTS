"""UI regressions verified against real Qt controls and front-door entry dispatch.

Synthetic state only. No Mimics API, model inference, training, SSH or repairs.
Windows font/DPI and native Mimics dialogs still require workstation acceptance.
"""
import ast
import contextlib
import importlib
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'runtime_py35'), str(ROOT / 'tests'), str(ROOT)]
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6 import QtCore, QtGui, QtWidgets
import ui_theme

QT = (QtCore, QtGui, QtWidgets)
APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
ui_theme.configure_application(APP)


class SmallScreen:
    def availableGeometry(self):
        return QtCore.QRect(0, 0, 853, 440)


def wait_until(predicate):
    end = time.monotonic() + 3
    while time.monotonic() < end:
        APP.processEvents()
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError('UI/background result did not settle')


class TestUiWorkflow(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.home = mock.patch.object(Path, 'home', return_value=self.root)
        self.home.start()
        self.addCleanup(self.home.stop)
        self.screen = mock.patch.object(QtWidgets.QWidget, 'screen', return_value=SmallScreen())
        self.screen.start()
        self.addCleanup(self.screen.stop)
        ui_theme.configure_application(APP)

    def keep(self, instance):
        w = instance if isinstance(instance, QtWidgets.QWidget) else getattr(instance, 'window', getattr(instance, 'dialog', None))
        def close():
            for timer in w.findChildren(QtCore.QTimer):
                timer.stop()
            w.close()
            APP.processEvents()
        self.addCleanup(close)
        w.show()
        APP.processEvents()
        return instance

    def test_label_choices_fit_actual_combo_and_warning_has_own_row(self):
        import nnunet_training_setup_ui as ui
        w = self.keep(ui.TrainingSetupWindow({'workspace': str(self.root / 'nn')}, self.root / 'context.json', QT))
        w.tabs.setCurrentIndex(0)
        w.window.resize(853, 440)
        APP.processEvents()
        combo = w.label_table.cellWidget(0, 3)
        for index in range(combo.count()):
            combo.setCurrentIndex(index)
            opt = QtWidgets.QStyleOptionComboBox()
            combo.initStyleOption(opt)
            rect = combo.style().subControlRect(QtWidgets.QStyle.CC_ComboBox, opt, QtWidgets.QStyle.SC_ComboBoxEditField, combo)
            self.assertGreaterEqual(rect.width(), combo.fontMetrics().horizontalAdvance(combo.currentText()))
        self.assertGreaterEqual(w.label_table.rowHeight(0), combo.sizeHint().height())
        w.task_edit.setText('中文任务')
        self.assertFalse(w.dataset_id.isVisible())
        task_layout = w.task_name_hint.parentWidget().layout()
        row, col, rs, cs = task_layout.getItemPosition(task_layout.indexOf(w.task_name_hint))
        self.assertEqual((row, col), (1, 0))
        self.assertNotEqual(w.task_name_hint.parentWidget(), w.dataset_id.parentWidget())
        w.tabs.setCurrentIndex(1)
        w.advanced_toggle.click()
        self.assertTrue(w.dataset_id.isVisible())
        self.assertGreater(w.dataset_id.value(), 0)

    def test_remote_scroll_preserves_form_and_footer_and_advanced_values(self):
        import remote_compute_ui as ui
        with mock.patch.object(ui, 'load_profiles', return_value=[]):
            w = self.keep(ui.ServerProfilesDialog(None, QT))
        w.dialog.resize(853, 440)
        APP.processEvents()
        self.assertFalse(w.runtime_combo.isVisible())
        self.assertTrue(w.advanced_toggle.isVisible())
        self.assertNotEqual(w.advanced_toggle.toolButtonStyle(), QtCore.Qt.ToolButtonIconOnly)
        runtime = w.runtime_combo.currentData()
        w.advanced_toggle.click()
        self.assertTrue(w.runtime_combo.isVisible())
        self.assertEqual(w.runtime_combo.currentData(), runtime)
        scroll = w.dialog.findChild(QtWidgets.QScrollArea)
        scroll.ensureWidgetVisible(w.retention_spin)
        APP.processEvents()
        self.assertGreater(scroll.verticalScrollBar().maximum(), 0)
        self.assertFalse(w.retention_spin.visibleRegion().isEmpty())
        self.assertFalse(w.save_button.visibleRegion().isEmpty())
        for edit in (w.host_edit, w.username_edit):
            self.assertGreaterEqual(edit.height(), edit.minimumSizeHint().height())

    def test_active_learning_single_creation_form_and_selection_gating(self):
        import flexict_active_learning_ui as ui
        with mock.patch.object(ui, 'find_al_jobs', return_value=[]):
            w = self.keep(ui.ActiveLearningWindow({'workspace': str(self.root / 'al')}, QT))
            self.assertFalse(w.open_button.isEnabled())
            self.assertFalse(w._new_run_group.isVisible())
            with mock.patch.object(QtWidgets.QDialog, 'exec', side_effect=AssertionError('duplicate modal')):
                w.new_button.click()
                self.assertTrue(w._new_run_group.isVisible())
                w.window.resize(853, 440)
                APP.processEvents()
                self.assertFalse(w.new_button.visibleRegion().isEmpty())
                w.pool_edit.setText('keep this pool')
                w.new_button.click()
                self.assertFalse(w._new_run_group.isVisible())
                w.new_button.click()
                self.assertEqual(w.pool_edit.text(), 'keep this pool')
            w._rows = [{'case': 'caseA'}]
            w.table.setRowCount(1)
            w.table.setItem(0, 0, QtWidgets.QTableWidgetItem('1'))
            w._set_actions_enabled(True)
            self.assertTrue(w.export_button.isEnabled())
            self.assertFalse(w.open_button.isEnabled())
            w.table.selectRow(0)
            self.assertTrue(w.open_button.isEnabled())
            w.table.clearSelection()
            w.table.setCurrentCell(-1, -1)
            self.assertFalse(w.open_button.isEnabled())

    def test_model_manager_shows_complete_id_and_only_applicable_actions(self):
        import model_manager_ui as ui
        long_id = 'wholebody_' + 'long_model_version_' * 8
        rows = [dict(family=f, workspace=str(self.root), task_id='task', model_id=long_id + f,
                     target='多器官', configuration='', created_at_epoch=1, imported=True,
                     current=False, usable=usable) for f, usable in [('nnunet', True), ('flexict', True), ('flexict', False)]]
        with mock.patch.object(ui, 'collect_rows', return_value=rows):
            w = self.keep(ui.ModelManagerWindow({'workspaces': {'nnunet': str(self.root)}}, QT))
            wait_until(lambda: (w._poll() or not w._loading))
        w.table.selectRow(0)
        self.assertIn(long_id, w.selection_detail.toPlainText())
        self.assertFalse(w.use_button.isVisible())
        w.table.selectRow(1)
        self.assertTrue(w.use_button.isEnabled())
        self.assertEqual(w.use_button.text(), '设为默认模型')
        self.assertIn('不切换', w.selection_detail.toPlainText())
        w.table.selectRow(2)
        self.assertTrue(w.remove_button.isVisible())
        self.assertFalse(w.use_button.isEnabled())

    def test_environment_repairs_target_their_own_feedback(self):
        import env_guidance as ui
        issues = [dict(title=t, detail=t, fix_action=a) for t, a in [('第一项', 'install'), ('第二项', 'migrate:old')]]
        def modal(d):
            boxes = d.findChildren(QtWidgets.QGroupBox)
            for box, expected in zip(boxes, ['done:install', 'done:migrate:old']):
                box.findChild(QtWidgets.QPushButton).click()
                labels = [label.text() for label in box.findChildren(QtWidgets.QLabel)]
                self.assertIn(expected, labels)
            return QtWidgets.QDialog.Accepted
        with mock.patch.object(ui, 'collect_issues', return_value=issues), mock.patch.object(ui, 'run_fix', side_effect=lambda r, a: 'done:' + a), mock.patch.object(QtWidgets.QDialog, 'exec', modal):
            self.assertEqual(ui.show_dialog(self.root), 0)

    def test_environment_keyboard_recheck_really_runs_detection_again(self):
        import env_guidance as ui
        rounds = []
        def modal(d):
            rounds.append(d)
            button = next(b for b in d.findChildren(QtWidgets.QPushButton) if b.text() == ('重新检查' if len(rounds) == 1 else '关闭'))
            self.assertFalse(button.underMouse())
            button.click()
            return d.result()
        with mock.patch.object(ui, 'collect_issues', return_value=[]) as collect, mock.patch.object(QtWidgets.QDialog, 'exec', modal):
            self.assertEqual(ui.show_dialog(self.root), 0)
            self.assertEqual(collect.call_count, 2)
            self.assertEqual(len(rounds), 2)

    def test_prediction_tables_preserve_long_names_and_failure_reason(self):
        import nnunet_prediction_setup_ui as single
        import nnunet_batch_predict_ui as batch
        import flexict_prediction_setup_ui as flex
        long_id = 'multi_organ_model_' * 10
        row = dict(task_id='wholebody', task_name='全身多器官', model_id=long_id,
                   configuration='3d_fullres', created_at_epoch=1,
                   labels=[{'name': 'Liver'}], label_names=['Liver'], execution_backend='local')
        for ui, ctor in [(single, single.PredictionWindow), (batch, batch.BatchPredictionWindow), (flex, flex.PredictionWindow)]:
            with self.subTest(window=ui.__name__):
                w = self.keep(ctor({'workspace': str(self.root / ui.__name__)}, QT))
                w._loading_models = False
                with mock.patch.object(ui, 'model_usability', return_value=(True, '')):
                    w._apply_models([row], '')
                self.assertEqual(w.table.item(0, 1).toolTip(), long_id)

    def test_flexict_unusable_models_never_claim_recommended_or_allow_predict(self):
        import flexict_prediction_setup_ui as ui
        w = self.keep(ui.PredictionWindow({'workspace': str(self.root / 'flex')}, QT))
        w._loading_models = False
        row = dict(task_name='肝脏', model_id='moved_model', label_names=['Liver'], configuration='2d')
        with mock.patch.object(ui, 'model_usability', return_value=(False, '模型路径已移动')):
            w._apply_models([row], '')
        self.assertFalse(w.start.isEnabled())
        self.assertIn('都不可用', w.status_label.text())
        self.assertIn('模型路径已移动', w.table.item(0, 1).toolTip())

    def test_export_small_screen_has_scroll_and_visible_primary_action(self):
        import io_path_setup_ui as ui
        seen = []
        def loop():
            w = next(w for w in APP.topLevelWidgets() if w.isVisible() and w.windowTitle() == '导出 Mask')
            self.keep(w)
            w.resize(853, 440)
            APP.processEvents()
            scrolls = w.findChildren(QtWidgets.QScrollArea)
            self.assertTrue(any(s.verticalScrollBar().maximum() > 0 for s in scrolls))
            start = next(b for b in w.findChildren(QtWidgets.QPushButton) if b.text() == '开始导出')
            self.assertFalse(start.visibleRegion().isEmpty())
            for s in scrolls:
                s.verticalScrollBar().setValue(s.verticalScrollBar().maximum())
            APP.processEvents()
            self.assertFalse(start.visibleRegion().isEmpty())
            seen.append(w)
            return 0
        with mock.patch.object(QtWidgets.QApplication, 'exec', side_effect=loop):
            ui.run_ui({'state_path': str(self.root / 'export.json'), 'status_path': str(self.root / 'export-status.json')})
        self.assertEqual(len(seen), 1)

    def test_disabled_stop_uses_disabled_visual_style(self):
        b = self.keep(QtWidgets.QPushButton('停止训练'))
        b.setObjectName('dangerButton')
        b.setStyleSheet(ui_theme.stylesheet())
        b.setEnabled(False)
        APP.processEvents()
        c = b.grab().toImage().pixelColor(8, 8)
        self.assertEqual(c.name(), '#f2f4f7')

    def _drop_check(self, kind, chosen):
        import import_drop_window as ui
        source = self.root / 'synthetic-source'
        source.mkdir()
        names = ['Liver', 'Spleen'] if chosen != 'all' else ['all']
        case = dict(case_id='synthetic', masks=[dict(name=n) for n in names])
        launched = []
        warnings = []
        def loop():
            w = next(w for w in APP.topLevelWidgets() if w.windowTitle() == '导入数据' and w.isVisible())
            self.keep(w)
            APP.clipboard().setText(str(source))
            next(b for b in w.findChildren(QtWidgets.QPushButton) if b.text() == '粘贴路径').click()
            named = next(b for b in w.findChildren(QtWidgets.QRadioButton) if b.text() == '选定的名称')
            wait_until(named.isEnabled)
            named.click()
            candidates = next(l for l in w.findChildren(QtWidgets.QListWidget) if l.count() and l.item(0).text() in names)
            self.assertTrue(candidates.isVisible())
            if chosen:
                candidates.item(0).setCheckState(QtCore.Qt.Checked)
            submit = next(b for b in w.findChildren(QtWidgets.QPushButton) if b.text() == '导入')
            submit.click()
            if chosen == 'Liver':
                wait_until(lambda: bool(launched))
                self.assertEqual(launched[0]['mask_selection'], 'Liver')
            else:
                self.assertFalse(launched)
                self.assertTrue(warnings)
            for timer in w.findChildren(QtCore.QTimer):
                timer.stop()
            w.close()
            return 0
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(ui, 'state_file_path', return_value=str(self.root / 'drop.json')))
            stack.enter_context(mock.patch.object(ui, 'register_window_process', return_value=''))
            stack.enter_context(mock.patch.object(ui, 'collect_recent_drops', return_value=[]))
            stack.enter_context(mock.patch.object(ui, 'classify_payload', return_value=(kind, str(source), [])))
            stack.enter_context(mock.patch.object(ui.io_ui, 'discover_single_source', return_value=case))
            stack.enter_context(mock.patch.object(ui.io_ui, 'summarize_dataset', return_value=dict(mask_names=names, summary='2例', warnings=[])))
            stack.enter_context(mock.patch.object(ui, 'submit_import', side_effect=lambda s, c: (launched.append(s) or [])))
            stack.enter_context(mock.patch.object(QtWidgets.QMessageBox, 'warning', side_effect=lambda *args: warnings.append(args)))
            stack.enter_context(mock.patch.object(QtWidgets.QMessageBox, 'information', return_value=QtWidgets.QMessageBox.Ok))
            stack.enter_context(mock.patch.object(QtWidgets.QApplication, 'exec', side_effect=loop))
            ui.run(context={})

    def test_drop_named_single_selection_reaches_submit(self):
        self._drop_check('single', 'Liver')

    def test_drop_named_batch_selection_reaches_submit(self):
        self._drop_check('batch', 'Liver')

    def test_drop_empty_named_selection_does_not_import_everything(self):
        self._drop_check('single', '')

    def test_drop_reserved_name_rejected_before_dispatch(self):
        self._drop_check('single', 'all')


class TestFrontDoorRoutes(unittest.TestCase):
    def test_all_28_entry_scripts_call_their_actual_runtime_handler(self):
        # This module supplies the repository's existing Mimics stub. It does
        # not emulate a Mimics workstation or execute actual API operations.
        import test_all  # noqa: F401
        import _mimics_entrypoint
        paths = sorted((ROOT / 'scripting_library').rglob('*.py'))
        self.assertEqual(len(paths), 28)
        for p in paths:
            with self.subTest(entry=str(p.relative_to(ROOT))):
                source = p.read_text(encoding='utf-8-sig')
                calls = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'run_runtime_entry']
                self.assertEqual(len(calls), 1)
                call = calls[0]
                module_name = ast.literal_eval(call.args[2])
                kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
                module = importlib.import_module(module_name)
                fn = kwargs.get('function_name', 'main')
                self.assertTrue(callable(getattr(module, fn)))
                expected_args = ()
                if kwargs.get('action_attr'):
                    expected_args = (getattr(module, kwargs['action_attr']),)
                elif 'action_value' in kwargs:
                    expected_args = (kwargs['action_value'],)
                with mock.patch.object(module, fn, return_value=0) as handler:
                    exec(compile(source, str(p), 'exec'), {'__file__': str(p), '__name__': '__main__'})
                    handler.assert_called_once_with(*expected_args)

    def test_family_actions_dispatch_beyond_main(self):
        import test_all  # noqa: F401
        for module_name, routes in [
            ('nnunet_mimics', {'BUTTON_TRAIN': 'start_training', 'BUTTON_STATUS': 'show_status', 'BUTTON_BATCH_PREDICT': 'show_batch_predict'}),
            ('flexict_mimics', {'BUTTON_TRAIN': 'start_training', 'BUTTON_STATUS': 'show_status', 'BUTTON_ACTIVE_LEARNING': '_start_active_learning'})
        ]:
            m = importlib.import_module(module_name)
            with mock.patch.object(m, '_resume_pending_inference_monitors'), contextlib.ExitStack() as stack:
                if hasattr(m, '_start_al_request_monitor'):
                    stack.enter_context(mock.patch.object(m, '_start_al_request_monitor'))
                for action, fn in routes.items():
                    with mock.patch.object(m, fn, return_value=7) as handler:
                        self.assertEqual(m.main(getattr(m, action)), 7)
                        handler.assert_called_once_with()
                with mock.patch.object(m, '_prediction_context', return_value={'case': 'synthetic'}), mock.patch.object(m, '_start_prediction_with_context', return_value=8) as predict:
                    self.assertEqual(m.main(m.BUTTON_PREDICT), 8)
                    predict.assert_called_once_with({'case': 'synthetic'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
