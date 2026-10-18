"""Actual Qt recovery controls and slow-storage boundary, synthetic files only."""
import argparse,json,os,sys,tempfile,threading,time
from pathlib import Path
from unittest import mock
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
ROOT=Path(__file__).resolve().parents[3]
import subprocess as _audit_subprocess
_AUDIT_SOURCE='e325e3a8a91c7c64e2c68e1b8b117f6cf08fc31f'
assert not _audit_subprocess.check_output(['git','diff','--name-only',_AUDIT_SOURCE,'--','tools','runtime_py35','tests','mimics_bridge.py'],cwd=ROOT,text=True).strip(), 'Source changed: copy this evidence into a new source-pinned batch before running'

sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'runtime_py35'),str(ROOT/'tests'),str(ROOT)]
from PySide6 import QtCore,QtGui,QtWidgets
import numpy as np,nibabel as nib
import nnunet_common as nc,nnunet_jobs as jobs,nnunet_status_viewer as nv,flexict_status_viewer as fv
import batch_status_viewer as batch,ui_theme
parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([]);ui_theme.configure_application(app)
qt=(QtCore,QtGui,QtWidgets);observations={};instances=[]
class Screen:
    def availableGeometry(self):return QtCore.QRect(0,0,853,440)
def capture(name,window):
    window.resize(853,440);window.show();app.processEvents();window.grab().save(str(args.out/(name+'.png')))
def stop(window):
    for timer in window.findChildren(QtCore.QTimer):timer.stop()
    window.close();app.processEvents()
with tempfile.TemporaryDirectory(prefix='mimics_ui_e325e3a_') as tmp,mock.patch.object(Path,'home',return_value=Path(tmp)),mock.patch.object(QtWidgets.QWidget,'screen',return_value=Screen()):
    t=Path(tmp)
    for module,name in [(nv,'nnunet'),(fv,'flexict')]:
        window=module.StatusWindow({'workspace':str(t/name)},qt);instances.append(window)
        for timer in window.window.findChildren(QtCore.QTimer):timer.stop()
        image=t/(name+'_current.nii.gz');old=t/(name+'_old.nii.gz')
        for path,value in [(image,2),(old,1)]:nib.save(nib.Nifti1Image(np.full((4,4,4),value,dtype=np.uint8),np.eye(4)),path)
        status_path=t/(name+'_status.json')
        status={'task_id':'liver','task_name':'肝脏','status_path':str(status_path),'output_path':str(image),
                'status':'completed','created_at_epoch':2,'kind':'infer','message':'预测完成，空间校验通过。',
                'replaced_previous_output':str(old),'published_output_fingerprint':nc.file_fingerprint(image),
                '_replaced_mtime_epoch':1760000000,'progress_percent':100}
        nc.write_json_atomic(status_path,status)
        if module is nv:
            with mock.patch.object(window,'_selected_task',return_value='liver'):window.refresh({'jobs':[status],'models':[]})
        else:window.refresh({'jobs':[status],'models':[],'pair':(None,None),'recommended':None})
        capture(name+'-previous-output',window.window)
        b=window.previous_button
        observations[name+'_recovery_entry']={'enabled':b.isEnabled(),'detail':window.detail_label.text(),
            'button_size':[b.width(),b.height()],'button_visible_size':[b.visibleRegion().boundingRect().width(),b.visibleRegion().boundingRect().height()],
            'window_size':[window.window.width(),window.window.height()]}
        # Actual restore worker with synthetic 200ms storage reads; run
        # the real event loop while it works and wait for the GUI receipt.
        hits=[];threads=[];real_fingerprint=jobs.file_fingerprint
        def slow_fingerprint(path):
            threads.append(threading.get_ident());time.sleep(.2);return real_fingerprint(path)
        timer=QtCore.QTimer();timer.setInterval(10);timer.timeout.connect(lambda:hits.append(time.monotonic()));timer.start()
        main_thread=threading.get_ident();start=time.monotonic()
        dialogs=[]
        with mock.patch.object(jobs,'file_fingerprint',side_effect=slow_fingerprint),mock.patch.object(QtWidgets.QMessageBox,'information',side_effect=lambda *a:dialogs.append(a[1])),mock.patch.object(QtWidgets.QMessageBox,'critical',side_effect=lambda *a:dialogs.append(a[1])),mock.patch.object(window.refresher,'request',return_value=None):
            window._restore_previous(status);returned=time.monotonic()-start
            busy_snapshot={'busy':window._restore_busy,'button_enabled':window.previous_button.isEnabled(),'detail':window.detail_label.text()}
            capture(name+'-restore-running',window.window)
            deadline=time.monotonic()+5
            while window._restore_busy and time.monotonic()<deadline:
                app.processEvents();time.sleep(.005)
        elapsed=time.monotonic()-start;timer.stop()
        observations[name+'_restore_callback']={'return_seconds':returned,'total_seconds':elapsed,
            'fingerprint_reads_off_ui_thread':bool(threads) and all(x!=main_thread for x in threads),
            'timer_callbacks_during_restore':len(hits),'restored_value':int(np.asanyarray(nib.load(image).dataobj)[0,0,0]),
            'dialogs':dialogs,'busy_snapshot':busy_snapshot,'synthetic_delay_per_storage_read_seconds':.2}
        stop(window.window)
    window=batch.BatchStatusWindow(str(t),qt);instances.append(window)
    for timer in window.window.findChildren(QtCore.QTimer):timer.stop()
    row=batch._row('Import','合成批量任务',{'status':'closed','completed':1,'failed':0,'receipt_failed':1,'receipt_failed_cases':['caseA'],'total':1,'updated_at_epoch':1760000000},t/'import-status.json')
    # Use the viewer's actual snapshot application contract.
    window.refresh([row])
    capture('batch-receipt-warning',window.window)
    observations['receipt_warning']={'note':batch.receipt_failure_note(row),'rows':window.table.rowCount()}
    stop(window.window)
# A separate process reaches a real rename, closes its last Qt window,
# and exits normally. Inspect persistent files after the daemon is gone.
import subprocess
with tempfile.TemporaryDirectory(prefix='mimics_restore_exit_') as name:
    base=Path(name)
    for module in ['nnunet_status_viewer','flexict_status_viewer']:
        folder=base/module;folder.mkdir()
        for filename,value in [('current.nii.gz',2),('old.nii.gz',1)]:
            nib.save(nib.Nifti1Image(np.full((4,4,4),value,np.uint8),np.eye(4)),folder/filename)
        nc.write_json_atomic(folder/'status.json',{'status':'completed','output_path':str(folder/'current.nii.gz'),
            'replaced_previous_output':str(folder/'old.nii.gz'),'published_output_fingerprint':nc.file_fingerprint(folder/'current.nii.gz')})
        p=subprocess.run([sys.executable,str(Path(__file__).with_name('restore_exit_child.py')),module,str(folder)],capture_output=True,text=True,timeout=12)
        result=nc.read_json(folder/'child-result.json',{})
        observations[module+'_close_during_swap']={**result,'process_exit':p.returncode,
            'output_exists_after_exit':(folder/'current.nii.gz').exists(),'old_exists':(folder/'old.nii.gz').exists(),
            'kept_versions':[int(np.asanyarray(nib.load(x).dataobj)[0,0,0]) for x in folder.glob('*.replaced-*.nii.gz')],
            'status_still_completed':nc.read_json(folder/'status.json',{}).get('status')=='completed',
            'stderr':p.stderr[-1000:]}
(args.out/'ui-observations.json').write_text(json.dumps(observations,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(observations,ensure_ascii=False,indent=2))
