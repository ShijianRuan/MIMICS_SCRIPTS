"""Render and measure actual Qt controls using isolated synthetic state.

Does not execute import/prediction/training/SSH/repair actions. Offscreen
renders are not a substitute for Windows/Mimics visual acceptance.
"""
import argparse,contextlib,json,os,sys,tempfile,time,traceback
from pathlib import Path
from unittest import mock
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[3])
p.add_argument('--font-scale',type=float,default=1.0);a=p.parse_args()
ROOT=a.repo.resolve()
sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'runtime_py35')]
from PySide6 import QtCore,QtGui,QtWidgets
import ui_theme
app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
if a.font_scale != 1.0:
    original_stylesheet=ui_theme.stylesheet
    ui_theme.stylesheet=lambda extra='': original_stylesheet(extra).replace('10pt','%.2fpt'%(10*a.font_scale))
ui_theme.configure_application(app)
qt=(QtCore,QtGui,QtWidgets);a.out.mkdir(parents=True,exist_ok=True)
records=[];errors=[];instances=[]
class Screen:
    def availableGeometry(self):return QtCore.QRect(0,0,853,440)

def capture(name,w,requested):
    w.resize(*requested);w.show();app.processEvents()
    buttons=[];combos=[]
    for b in w.findChildren(QtWidgets.QAbstractButton):
        if not b.isVisible() or not b.text():continue
        text=b.text().replace('&','');tw=b.fontMetrics().horizontalAdvance(text)
        if isinstance(b,QtWidgets.QPushButton):
            opt=QtWidgets.QStyleOptionButton();b.initStyleOption(opt)
            rect=b.style().subElementRect(QtWidgets.QStyle.SE_PushButtonContents,opt,b)
            available=rect.width();available_height=rect.height()
        elif isinstance(b,QtWidgets.QToolButton):
            opt=QtWidgets.QStyleOptionToolButton();b.initStyleOption(opt)
            available=b.style().subControlRect(QtWidgets.QStyle.CC_ToolButton,opt,QtWidgets.QStyle.SC_ToolButton,b).width()
            available_height=b.contentsRect().height()
            if b.toolButtonStyle()==QtCore.Qt.ToolButtonIconOnly:available=0
        else:available=b.contentsRect().width();available_height=b.contentsRect().height()
        pos=b.mapTo(w,QtCore.QPoint(0,0))
        buttons.append({'text':b.text(),'type':type(b).__name__,'x':pos.x(),'y':pos.y(),'width':b.width(),'height':b.height(),
                        'text_width':tw,'content_width':available,'text_height':b.fontMetrics().height(),
                        'content_height':available_height,'text_clipped':tw>available+1 or b.fontMetrics().height()>available_height+1,
                        'enabled':b.isEnabled(),'visible_pixels':b.visibleRegion().boundingRect().width(),
                        'visible_height':b.visibleRegion().boundingRect().height()})
    for c in w.findChildren(QtWidgets.QComboBox):
        if not c.isVisible():continue
        opt=QtWidgets.QStyleOptionComboBox();c.initStyleOption(opt)
        rect=c.style().subControlRect(QtWidgets.QStyle.CC_ComboBox,opt,QtWidgets.QStyle.SC_ComboBoxEditField,c)
        width=c.fontMetrics().horizontalAdvance(c.currentText())
        combos.append({'text':c.currentText(),'text_width':width,'content_width':rect.width(),
                       'text_clipped':width>rect.width()+1,'tooltip':c.toolTip()})
    record={'name':name,'requested':list(requested),'actual':[w.width(),w.height()],
            'minimum_hint':[w.minimumSizeHint().width(),w.minimumSizeHint().height()],
            'buttons':buttons,'combos':combos,'image':name+'.png'}
    w.grab().save(str(a.out/(name+'.png')))
    records.append(record)

def window_cases(name,w):
    # Drain async metadata loaders before measuring the empty/ready state.
    for _ in range(12):app.processEvents();time.sleep(.015)
    tabs=w.findChildren(QtWidgets.QTabWidget)
    for size,suffix in [((1000,760),'normal'),((853,440),'small')]:
        if suffix=='small':ui_theme.clamp_window_to_screen(w,(min(w.minimumWidth(),853),min(w.minimumHeight(),440)))
        capture(name+'-'+suffix,w,size)
        for j,t in enumerate(tabs):
            for index in range(t.count()):
                t.setCurrentIndex(index);capture(name+'-'+suffix+'-tab%d-%d'%(j,index),w,size)
        if tabs:tabs[0].setCurrentIndex(min(1,tabs[0].count()-1))
    for b in w.findChildren(QtWidgets.QToolButton):
        if b.isCheckable() and ('高级' in b.text() or '参数' in b.text()):b.setChecked(True)
    if w.findChildren(QtWidgets.QToolButton):capture(name+'-expanded',w,(1000,760))
    for t in tabs:
        for index in range(t.count()):
            t.setCurrentIndex(index);capture(name+'-expanded-small-tab%d'%index,w,(853,440))
    # Scroll each long form to its end, proving secondary fields remain reachable.
    for scroll in w.findChildren(QtWidgets.QScrollArea):
        scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum())
    capture(name+'-bottom-small',w,(853,440))
    for timer in w.findChildren(QtCore.QTimer):timer.stop()
    w.close();app.processEvents()

with tempfile.TemporaryDirectory(prefix='mimics_ui_audit_') as tmp:
    t=Path(tmp);home=t/'home';home.mkdir()
    import remote_compute_ui as remote
    with mock.patch.object(Path,'home',return_value=home),mock.patch.object(remote,'load_profiles',return_value=[]),mock.patch.object(QtWidgets.QWidget,'screen',return_value=Screen()):
        import nnunet_training_setup_ui as nt,flexict_training_setup_ui as ft
        import nnunet_prediction_setup_ui as np,flexict_prediction_setup_ui as fp
        import nnunet_batch_predict_ui as bp,flexict_active_learning_ui as al
        import nnunet_status_viewer as ns,flexict_status_viewer as fs,batch_status_viewer as bs
        import model_manager_ui as mm,config_editor_ui as config,window_level_editor_ui as wl
        import nninteractive_task_model_center as center,nninteractive_task_model_chooser as chooser
        factories=[
            ('nnunet-training',lambda:nt.TrainingSetupWindow({'workspace':str(t/'nnunet')},t/'nt.json',qt)),
            ('flexict-training',lambda:ft.TrainingSetupWindow({'workspace':str(t/'flexict')},t/'ft.json',qt)),
            ('nnunet-predict',lambda:np.PredictionWindow({'workspace':str(t/'nnunet'),'case_id':'合成病例_静脉期','selected_mask_name':'肝脏'},qt)),
            ('flexict-predict',lambda:fp.PredictionWindow({'workspace':str(t/'flexict'),'case_id':'合成病例_静脉期'},qt)),
            ('batch-predict',lambda:bp.BatchPredictionWindow({'workspace':str(t/'nnunet')},qt)),
            ('active-learning',lambda:al.ActiveLearningWindow({'workspace':str(t/'flexict')},qt)),
            ('nnunet-status',lambda:ns.StatusWindow({'workspace':str(t/'nnunet')},qt)),
            ('flexict-status',lambda:fs.StatusWindow({'workspace':str(t/'flexict')},qt)),
            ('data-status',lambda:bs.BatchStatusWindow(t,qt)),
            ('model-manager',lambda:mm.ModelManagerWindow({'workspaces':{'nnunet':str(t/'nnunet')}},qt)),
            ('config',lambda:config.ConfigEditor(qt)),
            ('window-level',lambda:wl.PresetEditor(qt)),
            ('remote-server',lambda:remote.ServerProfilesDialog(None,qt)),
            ('interactive-center',lambda:center.ModelCenter(QtWidgets.QDialog(),{'workspace':str(t/'nnint')},qt)),
            ('interactive-chooser',lambda:chooser.Chooser(QtWidgets.QDialog(),{'workspace':str(t/'nnint'),'status_path':str(t/'choose.json')},qt)),
        ]
        with mock.patch.object(config,'ROOT',t),mock.patch.object(center,'find_environment_python',return_value=sys.executable):
            for name,factory in factories:
                try:
                    instance=factory();instances.append(instance);w=getattr(instance,'window',getattr(instance,'dialog',None))
                    window_cases(name,w)
                except Exception as exc:errors.append({'name':name,'error':str(exc),'traceback':traceback.format_exc()})
        # Populated states: table text, row selection, failure and remote uncertainty.
        try:
            rows=[{'family':'nnunet','workspace':str(t/'nnunet'),'task_id':'wholebody',
                   'model_id':'wholebody_multi_organ_very_long_model_version_20261012_fold_all',
                   'target':'全身多器官','configuration':'3d_fullres','created_at_epoch':1,
                   'imported':True,'current':False,'usable':True}]
            with mock.patch.object(mm,'collect_rows',return_value=rows):
                manager=mm.ModelManagerWindow({'workspaces':{'nnunet':str(t/'nnunet')}},qt)
                for _ in range(25):app.processEvents();manager._poll();time.sleep(.01)
                instances.append(manager)
                manager.table.selectRow(0)
                window_cases('model-manager-selected',manager.window)
            with mock.patch.object(center,'find_environment_python',return_value=sys.executable):
                interactive=center.ModelCenter(QtWidgets.QDialog(),{'workspace':str(t/'nnint')},qt)
            instances.append(interactive)
            interactive.tabs.setCurrentWidget(interactive.progress_tab)
            for state,kwargs in [('running',{}),('failed',{}),('running',{'execution_backend':'remote','remote_state_unknown':True})]:
                interactive._update_progress_actions(dict(status=state,**kwargs))
                capture('interactive-actions-'+('remote-unknown' if kwargs else state),interactive.window,(853,440))
            model=dict(model_id='ui_synthetic_model_long_version_20261012',state='unverified',strategy='baseline',created_at_epoch=1,train_case_count=2,validation_case_count=1)
            with mock.patch.object(center,'find_task',return_value={}),mock.patch.object(center,'selected_model',return_value=None),mock.patch.object(center,'model_rows',return_value=[model]),mock.patch.object(center,'model_is_usable',return_value=True):
                interactive.tabs.setCurrentWidget(interactive.models_tab)
                interactive.refresh_models()
            capture('interactive-models-populated',interactive.window,(853,440))
            if hasattr(interactive,'models_scroll'):
                interactive.models_scroll.ensureWidgetVisible(interactive.model_table.cellWidget(0,5))
            capture('interactive-models-populated-bottom',interactive.window,(853,440))
            interactive.window.close()
            job=t/'flexict'/'jobs'/'ui-synthetic-ranking';(job/'uncertainty').mkdir(parents=True)
            (job/'uncertainty'/'ranking.csv').write_text('case,integrated,uncertain_vol,max\n合成病例_A,0.5,10.0,1.0\n合成病例_B,0.3,8.0,1.0\n',encoding='utf-8')
            (job/'status.json').write_text(json.dumps({'kind':'active_learning','status':'completed','created_at_epoch':1,'label_name':'Liver','job_dir':str(job)}))
            ranking=al.ActiveLearningWindow({'workspace':str(t/'flexict')},qt)
            instances.append(ranking)
            for _ in range(25):app.processEvents();ranking._poll_background_results();time.sleep(.01)
            ranking.table.selectRow(0)
            capture('active-learning-selected',ranking.window,(853,440))
            with mock.patch.object(QtWidgets.QDialog,'exec',return_value=QtWidgets.QDialog.Rejected):
                ranking.new_button.click()
            capture('active-learning-create',ranking.window,(853,440))
            for timer in ranking.window.findChildren(QtCore.QTimer):timer.stop()
            ranking.window.close()
            prediction=np.PredictionWindow({'workspace':str(t/'nnunet'),'case_id':'合成病例_静脉期'},qt)
            instances.append(prediction)
            prediction._loading_models=False
            prediction._apply_models([dict(rows[0],task_name='全身多器官',labels=[{'name':'Liver'},{'name':'Spleen'}],execution_backend='local'),
                                      dict(rows[0],model_id='missing_model',_unusable_reason='模型文件已移动，请重新导入')],'')
            window_cases('nnunet-predict-selected',prediction.window)
            flex_prediction=fp.PredictionWindow({'workspace':str(t/'flexict'),'case_id':'合成病例_静脉期'},qt)
            instances.append(flex_prediction);flex_prediction._loading_models=False
            with mock.patch.object(fp,'model_usability',return_value=(True,'')):
                flex_prediction._apply_models([dict(rows[0],task_name='肝脏',label_names=['Liver'],configuration='2d')],'')
            window_cases('flexict-predict-selected',flex_prediction.window)

        except Exception as exc:errors.append({'name':'populated-states','error':str(exc),'traceback':traceback.format_exc()})
        try:
            with mock.patch.object(remote,'load_profiles',return_value=[]):cache=remote.ServerProfilesDialog(None,qt)
            instances.append(cache);cache.retention_spin.setValue(7)
            def cache_modal(d):window_cases('remote-cached-datasets',d);return QtWidgets.QDialog.Accepted
            with mock.patch.object(QtWidgets.QDialog,'exec',cache_modal):
                cache._show_cache_rows([{'fingerprint':'a'*64,'size_bytes':1024,'mtime_epoch':1,'bucket':'own'}])
            cache.dialog.close()
        except Exception as exc:errors.append({'name':'remote-cached-datasets','error':str(exc),'traceback':traceback.format_exc()})
        def capture_loop(name):
            def run(*args):
                for w in app.topLevelWidgets():
                    if w.isVisible():window_cases(name,w)
                return 0
            return run
        import io_path_setup_ui as io,import_drop_window as drop,mask_file_picker_ui as picker,env_guidance as env,system_health_panel as health
        for name,run,patches in [
            ('export',lambda:io.run_ui({'state_path':str(t/'io-state.json'),'status_path':str(t/'io-status.json')}),[]),
            ('import',lambda:drop.run(context={}),[mock.patch.object(drop,'state_file_path',return_value=str(t/'drop-state.json')),mock.patch.object(drop,'register_window_process',return_value=''),mock.patch.object(drop,'collect_recent_drops',return_value=[])]),
            ('health',lambda:health.run(),[mock.patch.object(health,'collect_health',return_value={'processes':[],'locks':[],'queues':[],'server':{},'environment':{}})]),
        ]:
            try:
                with contextlib.ExitStack() as stack:
                    for patch in patches:stack.enter_context(patch)
                    stack.enter_context(mock.patch.object(QtWidgets.QApplication,'exec',side_effect=capture_loop(name)))
                    run()
            except Exception as exc:errors.append({'name':name,'error':str(exc),'traceback':traceback.format_exc()})
        try:
            context=t/'mask_context.json';context.write_text(json.dumps({'status_path':str(t/'mask-status.json'),'case_id':'合成病例'}))
            with mock.patch.object(sys,'argv',['picker','--context',str(context)]),mock.patch.object(QtWidgets.QApplication,'exec',side_effect=capture_loop('mask-import')):picker.main()
        except Exception as exc:errors.append({'name':'mask-import','error':str(exc),'traceback':traceback.format_exc()})
        try:
            def modal(dialog):
                instances.append(dialog)
                window_cases('environment-guidance',dialog);return QtWidgets.QDialog.Accepted
            with mock.patch.object(env,'collect_issues',return_value=[{'title':'外部环境未安装','detail':'请完成环境安装后使用 AI 功能。','fix_action':'install'},{'title':'模型路径已移动','detail':'修复路径后重新检查。','fix_action':'migrate:old'}]),mock.patch.object(QtWidgets.QDialog,'exec',modal):env.show_dialog(t)
        except Exception as exc:errors.append({'name':'environment-guidance','error':str(exc),'traceback':traceback.format_exc()})
(a.out/'metrics.json').write_text(json.dumps({'screens':[1000,760,853,440],'synthetic_only':True,'font_scale':a.font_scale,'records':records,'errors':errors},ensure_ascii=False,indent=2))
print(json.dumps({'windows':len(records),'errors':errors,'clipped':[{'screen':r['name'],'actual':r['actual'],'buttons':[b['text'] for b in r['buttons'] if b['text_clipped']]} for r in records if any(b['text_clipped'] for b in r['buttons'])]},ensure_ascii=False,indent=2))
