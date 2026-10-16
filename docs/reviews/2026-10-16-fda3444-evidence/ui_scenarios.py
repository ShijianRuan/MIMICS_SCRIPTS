"""Render populated/error workflows using real Qt UI; all state is synthetic."""
import argparse,json,os,sys,tempfile,time
from pathlib import Path
from unittest import mock
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'runtime_py35'),str(ROOT/'tests'),str(ROOT)]
from PySide6 import QtCore,QtGui,QtWidgets
import ui_theme,nninteractive_task_model_chooser as chooser
import nninteractive_task_model_center as center,nnunet_training_setup_ui as train
import remote_compute_ui as remote,import_drop_window as drop
parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([]);ui_theme.configure_application(app)
qt=(QtCore,QtGui,QtWidgets);instances=[];observations={}
class Screen:
 def availableGeometry(self):return QtCore.QRect(0,0,853,440)
def capture(name,w):
 w.resize(853,440);w.show();app.processEvents();w.grab().save(str(args.out/(name+'.png')))
 return {'actual':[w.width(),w.height()],'minimum_hint':[w.minimumSizeHint().width(),w.minimumSizeHint().height()]}
def close(w):
 for timer in w.findChildren(QtCore.QTimer):timer.stop()
 w.close();app.processEvents()
def wait(predicate):
 end=time.monotonic()+4
 while time.monotonic()<end:
  app.processEvents()
  if predicate():return
  time.sleep(.01)
 raise RuntimeError('synthetic scan did not settle')
with tempfile.TemporaryDirectory(prefix='mimics_ui_review_20261016_') as temp,mock.patch.object(Path,'home',return_value=Path(temp)),mock.patch.object(QtWidgets.QWidget,'screen',return_value=Screen()),mock.patch.object(remote,'load_profiles',return_value=[]):
 t=Path(temp)
 row={'model_id':'same_prefix_model_'*10+'version_A','state':'unverified','strategy':'baseline','train_case_count':2,'validation_case_count':1,'created_at_epoch':1,'quality':{}}
 task={'task_id':'kidneys','task_name':'双肾','recommended_model_id':row['model_id']}
 with mock.patch.object(chooser,'task_rows',return_value=[task]),mock.patch.object(chooser,'find_task',return_value=task),mock.patch.object(chooser,'model_rows',return_value=[row]),mock.patch.object(chooser,'audit_model_dir',return_value={'compatible':True}):
  c=chooser.Chooser(QtWidgets.QDialog(),{'workspace':str(t/'int'),'status_path':str(t/'chooser.json')},qt);instances.append(c)
  observations['long_model_chooser']=dict(capture('chooser-long-model',c.window),summary=c.summary.toPlainText(),model_id_width=c.summary.fontMetrics().horizontalAdvance(row['model_id']),summary_width=c.summary.width(),selectable=bool(c.summary.textInteractionFlags()&QtCore.Qt.TextSelectableByKeyboard))
  close(c.window)
 with mock.patch.object(center,'find_environment_python',return_value=sys.executable):m=center.ModelCenter(QtWidgets.QDialog(),{'workspace':str(t/'int')},qt)
 instances.append(m)
 for timer in m.window.findChildren(QtCore.QTimer):timer.stop()
 job=t/'retained-job';job.mkdir();(job/'status.json').write_text(json.dumps({'status':'completed','selection_outcome':'current_model_retained','model':{'model_id':row['model_id']}}))
 with mock.patch.object(center,'find_task',return_value={}),mock.patch.object(center,'selected_model',return_value=None),mock.patch.object(center,'model_rows',return_value=[dict(row,state='not_improved')]),mock.patch.object(center,'model_is_usable',return_value=True),mock.patch.object(m,'_apply_pending_scan_result',return_value=None),mock.patch.object(m,'refresh_tasks',return_value=None):
  m.current_job_dir=job;m.refresh_status();m.tabs.setCurrentWidget(m.progress_tab)
  observations['retained_completion_navigation']=dict(capture('retained-model-completion',m.window),detail=m.progress_detail.text(),default_model_rows=m.model_table.rowCount(),show_failed=m.show_failed.isChecked())
  observations['completion_layout']={'word_wrap':m.progress_detail.wordWrap(),'text_width':m.progress_detail.fontMetrics().horizontalAdvance(m.progress_detail.text()),'label_width':m.progress_detail.width(),'visible_width':m.progress_detail.visibleRegion().boundingRect().width(),'horizontal_scrolls':[{'maximum':scroll.horizontalScrollBar().maximum(),'viewport_width':scroll.viewport().width()} for scroll in m.progress_tab.findChildren(QtWidgets.QScrollArea)]}
  m.view_completed_model_button.click()
  observations['completed_model_navigation_after_click']={'models_tab':m.tabs.currentWidget()==m.models_tab,'rows':m.model_table.rowCount(),'show_failed':m.show_failed.isChecked(),'selected_row':m.model_table.currentRow()}
  capture('retained-model-selected',m.window)
 close(m.window)
 w=train.TrainingSetupWindow({'workspace':str(t/'nn'),'selected_mask_names':['肝脏','脾脏']},t/'context.json',qt);instances.append(w)
 w.task_edit.setText('全身器官标注');w.dataset_edit.setText(str(t/'data'))
 request=w._request()
 observations['chinese_label_submission']=dict(capture('chinese-label-submission',w.window),status=w.status_label.text(),request_labels=request['labels'],submission_not_launched=True)
 close(w.window)
 profile={'profile_id':'synthetic','name':'合成GPU服务器','host':'example.invalid','username':'demo','auth_method':'password','port':22,'runtime_image':'private/synthetic:2','remote_root':'other-storage','gpu_device':'auto'}
 with mock.patch.object(remote,'load_profiles',return_value=[profile]):
  server=remote.ServerProfilesDialog(None,qt,'synthetic');instances.append(server)
  observations['remote_profile_identity']=dict(capture('remote-hidden-identity',server.dialog),toggle=server.advanced_toggle.text(),summary=remote._non_default_summary(profile))
  scroll=server.dialog.findChild(QtWidgets.QScrollArea);scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum());app.processEvents();capture('remote-hidden-identity-bottom',server.dialog)
  close(server.dialog)
 # Use the actual scanner and the actual UI submit gate, with the launching boundary mocked.
 data=t/'batch';data.mkdir()
 for case in ['CaseA','CaseB','CaseC']:
  f=data/case;f.mkdir();(f/'ct.nii.gz').write_bytes(b'synthetic filename-only fixture')
 for case,mask in [('CaseA','Liver'),('CaseC','Spleen')]:
  f=data/case/'segmentations';f.mkdir();(f/(mask+'.nii.gz')).write_bytes(b'synthetic filename-only fixture')
 questions=[];launches=[]
 def loop():
  w=next(w for w in app.topLevelWidgets() if w.isVisible() and w.windowTitle()=='导入数据');instances.append(w)
  app.clipboard().setText(str(data));next(b for b in w.findChildren(QtWidgets.QPushButton) if b.text()=='粘贴路径').click()
  named=next(r for r in w.findChildren(QtWidgets.QRadioButton) if r.text()=='选定的名称');wait(named.isEnabled);named.click()
  candidates=next(l for l in w.findChildren(QtWidgets.QListWidget) if l.count() and l.item(0).text() in ['Liver','Spleen'])
  submit=next(b for b in w.findChildren(QtWidgets.QPushButton) if b.text()=='导入')
  candidates.item(0).setCheckState(QtCore.Qt.Checked);submit.click()
  observations['actual_batch_liver_warning']={'questions':list(questions),'CaseB_named_in_warning':any('CaseB' in str(q) for q in questions)}
  for i in range(candidates.count()):candidates.item(i).setCheckState(QtCore.Qt.Checked)
  questions.clear();submit.click();wait(lambda:bool(launches))
  observations['actual_batch_two_mask_submit']={'questions':list(questions),'launches':launches}
  capture('batch-two-mask-confirmed',w);close(w);return 0
 with mock.patch.object(drop,'state_file_path',return_value=str(t/'drop-state.json')),mock.patch.object(drop,'register_window_process',return_value=''),mock.patch.object(drop,'collect_recent_drops',return_value=[]),mock.patch.object(drop,'submit_import',side_effect=lambda sel,context:launches.append(sel) or []),mock.patch.object(QtWidgets.QMessageBox,'question',side_effect=lambda *a:questions.append(a[1:3]) or (QtWidgets.QMessageBox.Yes if '部分所选' in str(a[2]) else QtWidgets.QMessageBox.No)),mock.patch.object(QtWidgets.QMessageBox,'information',return_value=QtWidgets.QMessageBox.Ok),mock.patch.object(QtWidgets.QApplication,'exec',side_effect=loop):drop.run(context={})
(args.out/'ui-observations.json').write_text(json.dumps(observations,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(observations,ensure_ascii=False,indent=2))
