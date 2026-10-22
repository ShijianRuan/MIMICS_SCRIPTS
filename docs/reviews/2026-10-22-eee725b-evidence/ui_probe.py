"""Actual Qt setup progress UI, synthetic old/new job states and slow IO.
SLIP's Mimics-native dialogs are NOT rendered by this test.
"""
import json,os,sys,tempfile,time
from pathlib import Path
import subprocess
from unittest import mock
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
R=Path(__file__).resolve().parents[3];E=Path(__file__).resolve().parent
SOURCE='eee725b45ec42b7c045def95e4682089a3ea13a2'
assert not subprocess.check_output(['git','diff','--name-only',SOURCE,'--','tools/setup_progress_window.py'],cwd=R,text=True).strip(), 'Source changed; preserve old evidence'
sys.path.insert(0,str(R/'tools'))
from PySide6 import QtWidgets,QtCore
import setup_progress_window as ui
app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
out={}
with tempfile.TemporaryDirectory(prefix='slip_review_ui_') as folder:
    state=Path(folder)/'state.json';now=time.time()
    def write(value):state.write_text(json.dumps(value))
    with mock.patch.object(ui,'STATE_FILE',state),mock.patch.object(ui,'LOG_FILE',Path(folder)/'synthetic.log'):
        write({'status':'ok','action':'check','message':'上一次检查完成','started_at_epoch':now-3600,'updated_at_epoch':now-3500})
        w=ui.ProgressWindow();w._timer.stop();w.show();app.processEvents();w._tick()
        write({'status':'checking','action':'check','message':'本次正在检查 GPU','started_at_epoch':now,'updated_at_epoch':now,'network_drive':True})
        w._tick();app.processEvents();w.grab().save(str(E/'setup-progress-new-check.png'))
        out['stale_previous_state']={'progress_range':[w.bar.minimum(),w.bar.maximum()],
            'progress_value':w.bar.value(),'message':w.step_label.text(),'elapsed':w.elapsed_label.text(),
            'network_warning_visible':w.network_warning.isVisible(),'baseline_age_seconds':round(now-w._baseline_epoch)}
        real=w._read_state
        def slow():time.sleep(.15);return real()
        hits=[];timer=QtCore.QTimer();timer.setInterval(10);timer.timeout.connect(lambda:hits.append(1));timer.start()
        with mock.patch.object(w,'_read_state',side_effect=slow):
            start=time.monotonic();w._tick();elapsed=time.monotonic()-start
        before=len(hits);app.processEvents();timer.stop()
        out['slow_state_read']={'injected_seconds':.15,'callback_seconds':elapsed,
            'timer_callbacks_during_read':before,'timer_callbacks_after_events':len(hits)}
        w.close();w.deleteLater();app.processEvents()
(E/'ui-observations.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2))
