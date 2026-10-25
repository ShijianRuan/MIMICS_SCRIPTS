import os,sys,json,time,importlib.util
from pathlib import Path
ROOT=Path(os.environ.get('MIMICS_REVIEW_ROOT', str(Path(__file__).resolve().parents[3])))
OUT=Path(__file__).resolve().parent
SUFFIX=os.environ.get('AL_CAPTURE_SUFFIX','')
sys.path[:0]=[str(ROOT),str(ROOT/'tests'),str(ROOT/'tools')]
from PySide6 import QtCore,QtGui,QtWidgets
import test_active_learning_review_regressions as support
from ui_theme import configure_application,stylesheet
support.QtContracts.setUpClass()
app=support.QtContracts.app
configure_application(app,'主动学习界面检查');app.setStyleSheet(stylesheet())
test=support.QtContracts();out={}
import tempfile
for backend in ['nnunet','flexict']:
 with tempfile.TemporaryDirectory(prefix='al_ui_synthetic_') as folder:
  w=test.window(folder,backend)
  w.window.setWindowTitle(w.window.windowTitle()+'（合成演示数据）')
  w.window.resize(820,650)
  job=Path(w._job_dir)
  data=json.loads((job/'status.json').read_text());data['created_at_epoch']=1792900800
  (job/'status.json').write_text(json.dumps(data))
  geo=json.loads((job/'input_geometries.json').read_text());geo['cases']['caseA']['case_name']='患者甲_动脉期_长病例名称用于实际可读性检查'
  (job/'input_geometries.json').write_text(json.dumps(geo))
  import flexict_active_learning_ui as ui
  ui.set_case_state(job,'caseB','annotated');ui.set_case_state(job,'caseC','viewed')
  w.refresh();test.pump(lambda:w._jobs[0].get('created_at_epoch',0)>1)
  w._job_selected();w.table.selectRow(0)
  w.window.findChild(QtWidgets.QLabel,'title').setText(w._backend_name+' 主动学习（合成数据）')
  app.processEvents()
  p=OUT/(backend+'-review'+SUFFIX+'.png')
  w.window.grab().save(str(p))
  out[backend]={'window_size':[w.window.width(),w.window.height()],
     'selected_case':w._selected_case(),'case_name_tooltip':w.table.item(0,1).toolTip(),
     'buttons':{name:{'visible':getattr(w,name).isVisible(),'enabled':getattr(w,name).isEnabled(),
                 'width':getattr(w,name).width(),'text_width':getattr(w,name).fontMetrics().horizontalAdvance(getattr(w,name).text())}
                 for name in ['open_button','overlay_button','consensus_button','next_button','annotated_button','skip_button','export_button']},
     'horizontal_scroll_available':w.table.horizontalScrollBar().maximum()>0}
  w.window.close();app.processEvents()
(OUT/('ui-observations'+SUFFIX+'.json')).write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2))
