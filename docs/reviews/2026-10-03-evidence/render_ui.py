"""Render real Qt windows with synthetic cases and isolated settings.
Offscreen macOS/Fusion output is not Windows/Mimics UI acceptance.
"""
import argparse,json,os,sys,tempfile
from pathlib import Path
from unittest import mock
os.environ['QT_QPA_PLATFORM']='offscreen'
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
sys.path[:0]=[str(a.repo.resolve()),str(a.repo.resolve()/'tools')]
from PySide6 import QtCore,QtGui,QtWidgets
from ui_theme import configure_application,stylesheet
import flexict_active_learning_ui as al
import model_manager_ui as mm
import nnunet_training_setup_ui as nt
import flexict_training_setup_ui as ft
app=QtWidgets.QApplication([]);configure_application(app);app.setStyleSheet(stylesheet());qt=(QtCore,QtGui,QtWidgets)
metrics=[];a.out.mkdir(parents=True,exist_ok=True)
def capture(name,w,requested=(980,640)):
    w.resize(*requested);w.show();app.processEvents();w.grab().save(str(a.out/(name+'.png')))
    buttons=[]
    for b in w.findChildren(QtWidgets.QPushButton):
        if b.isVisible():
            pos=b.mapTo(w,QtCore.QPoint(0,0));buttons.append({'text':b.text(),'enabled':b.isEnabled(),'x':pos.x(),'y':pos.y(),'w':b.width(),'h':b.height()})
    metrics.append({'name':name,'requested':requested,'actual':[w.width(),w.height()],'minimum_hint':[w.minimumSizeHint().width(),w.minimumSizeHint().height()],'visible_buttons':buttons})

with tempfile.TemporaryDirectory() as tmp:
    t=Path(tmp);home=t/'home';home.mkdir();job=t/'jobs'/'ranking';(job/'uncertainty').mkdir(parents=True)
    (job/'status.json').write_text(json.dumps({'kind':'active_learning','status':'completed','label_name':'肝脏 liver','job_dir':str(job),'created_at_epoch':1.0}))
    (job/'uncertainty'/'ranking.csv').write_text('case,integrated,uncertain_vol,max\n合成病例_动脉期_薄层重建,0.92,23456,1\n合成病例_静脉期_薄层重建,0.85,789,1\n')
    win=al.ActiveLearningWindow({'workspace':str(t)},qt)
    capture('active-learning',win.window,(980,640));capture('active-learning-small',win.window,(820,520));win.window.close()
    with mock.patch.object(mm.ModelManagerWindow,'_reload'):
        win=mm.ModelManagerWindow({'workspaces':{'nnunet':str(t/'nnunet')}},qt)
        win._loading=False
        win.rows=[{'family':'nnunet','workspace':str(t/'nnunet'),'task_id':'liver','model_id':'nnunet_20261003T120000_12345678','target':'肝脏与肝内病灶','configuration':'3d_fullres','created_at_epoch':1790996400,'imported':True,'current':False,'usable':True}]
        win._fill_table();win.table.selectRow(0);win.status_label.setText('1 个合成模型（仅用于界面评审）')
        capture('model-manager',win.window);win.window.close()
    with mock.patch.object(Path,'home',return_value=home):
        for module,name in [(nt,'nnunet-training'),(ft,'flexict-training')]:
            win=module.TrainingSetupWindow({'workspace':str(t/name)},t/(name+'.json'),qt)
            capture(name,win.window,(920,700))
            win.tabs.setCurrentIndex(1);capture(name+'-settings',win.window,(920,700))
            capture(name+'-small',win.window,(800,500));win.window.close()
(a.out/'ui-metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2))
