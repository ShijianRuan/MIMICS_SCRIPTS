"""Capture current Qt widgets with isolated synthetic state, never launch jobs."""
import os,sys,tempfile,time
from pathlib import Path
from unittest import mock
os.environ['QT_QPA_PLATFORM']='offscreen'
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
OUT=Path(__file__).resolve().parent
from PySide6 import QtCore,QtGui,QtWidgets,QtSvg
from ui_theme import configure_application,stylesheet
import import_drop_window as drop
import model_manager_ui as mm
app=QtWidgets.QApplication([]);configure_application(app);app.setStyleSheet(stylesheet())
with tempfile.TemporaryDirectory() as tmp:
    state=Path(tmp)/'state.json'
    with mock.patch.object(drop,'state_file_path',return_value=str(state)),mock.patch.object(drop,'register_window_process',return_value=''),mock.patch.object(drop,'collect_recent_drops',return_value=[]):
        drop.run(preview_path=str(OUT/'import-window.png'))
    with mock.patch.object(mm.ModelManagerWindow,'_reload'):
        win=mm.ModelManagerWindow({'workspaces':{'nnunet':tmp}},(QtCore,QtGui,QtWidgets));win._loading=False
        win.rows=[{'family':'nnunet','workspace':tmp,'task_id':'演示任务','model_id':'示例 多器官草稿','target':'肝脏、脾脏、双肾','configuration':'3d_fullres','created_at_epoch':1791676800,'imported':True,'current':True,'usable':True}]
        win._fill_table();win.table.selectRow(0);win.status_label.setText('演示条目：模型可用时可切换使用，缺失模型会显示原因。')
        win.window.show();app.processEvents();win.window.grab().save(str(OUT/'model-manager.png'));win._timer.stop();win.window.close()
    for name in ('workflow','architecture'):
        renderer=QtSvg.QSvgRenderer(str(OUT/(name+'.svg')));size=renderer.defaultSize()
        img=QtGui.QImage(size,QtGui.QImage.Format_ARGB32);img.fill(QtCore.Qt.white)
        painter=QtGui.QPainter(img);renderer.render(painter);painter.end();img.save(str(Path(tmp)/(name+'.png')))
        # Preview PNGs kept in a temp directory, not repository deliverables.
        import shutil
        shutil.copyfile(Path(tmp)/(name+'.png'),Path('/tmp')/('mimics-brief-'+name+'.png'))
