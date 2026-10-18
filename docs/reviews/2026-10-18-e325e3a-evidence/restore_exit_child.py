"""Actual Qt close/normal process exit at a controlled restore swap boundary.

Only the pause after the first real rename is injected. All data is synthetic.
The parent checks files after this child exits; no process kill is used.
"""
import importlib,json,os,sys,threading,time
from pathlib import Path
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
ROOT=Path(__file__).resolve().parents[3]
import subprocess as _audit_subprocess
_AUDIT_SOURCE='e325e3a8a91c7c64e2c68e1b8b117f6cf08fc31f'
assert not _audit_subprocess.check_output(['git','diff','--name-only',_AUDIT_SOURCE,'--','tools','runtime_py35','tests','mimics_bridge.py'],cwd=ROOT,text=True).strip(), 'Source changed: copy this evidence into a new source-pinned batch before running'

sys.path[:0]=[str(ROOT/'tools'),str(ROOT)]
from PySide6 import QtCore,QtGui,QtWidgets
import nnunet_jobs as jobs

folder=Path(sys.argv[2]);module=importlib.import_module(sys.argv[1])
app=QtWidgets.QApplication([])
w=module.StatusWindow({'workspace':str(folder/'workspace')},(QtCore,QtGui,QtWidgets))
w.refresher.stop();w.window.show()
current=folder/'current.nii.gz';entered=threading.Event();real_replace=jobs.os.replace
def pause_after_first_rename(src,dst):
    result=real_replace(src,dst)
    if Path(src)==current:
        entered.set()
        threading.Event().wait(20)
    return result
jobs.os.replace=pause_after_first_rename
timer=QtCore.QTimer();timer.setInterval(10)
def close_when_between_renames():
    if entered.is_set():
        timer.stop();w.window.close()
timer.timeout.connect(close_when_between_renames);timer.start()
QtCore.QTimer.singleShot(0,lambda:w._restore_previous({'status_path':str(folder/'status.json')}))
QtCore.QTimer.singleShot(8000,app.quit)
code=app.exec()
(folder/'child-result.json').write_text(json.dumps({'app_exit':code,'entered_swap':entered.is_set(),
    'output_exists_at_exit':current.exists(),'restore_busy_at_exit':w._restore_busy,
    'worker_is_daemon':any(t.daemon and t is not threading.current_thread() for t in threading.enumerate())})+'\n')
