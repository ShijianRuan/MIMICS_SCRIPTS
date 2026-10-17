"""Actual Qt recovery controls and slow-storage boundary, synthetic files only."""
import argparse,json,os,sys,tempfile,threading,time
from pathlib import Path
from unittest import mock
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
ROOT=Path(__file__).resolve().parents[3]
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
with tempfile.TemporaryDirectory(prefix='mimics_ui_6d73583_') as tmp,mock.patch.object(Path,'home',return_value=Path(tmp)),mock.patch.object(QtWidgets.QWidget,'screen',return_value=Screen()):
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
        # Run the ACTUAL restore callback and actual file swap. Only the
        # fingerprint storage read gets a 200ms synthetic delay; it records
        # which Qt thread calls it. A due timer cannot fire during the callback.
        hits=[];threads=[];real_fingerprint=jobs.file_fingerprint
        def slow_fingerprint(path):
            threads.append(threading.get_ident());time.sleep(.2);return real_fingerprint(path)
        timer=QtCore.QTimer();timer.setInterval(10);timer.timeout.connect(lambda:hits.append(time.monotonic()));timer.start()
        main_thread=threading.get_ident();start=time.monotonic()
        with mock.patch.object(jobs,'file_fingerprint',side_effect=slow_fingerprint),mock.patch.object(QtWidgets.QMessageBox,'information',return_value=QtWidgets.QMessageBox.Ok),mock.patch.object(QtWidgets.QMessageBox,'critical',return_value=QtWidgets.QMessageBox.Ok),mock.patch.object(window.refresher,'request',return_value=None):
            window._restore_previous(status)
        elapsed=time.monotonic()-start;before_events=len(hits);app.processEvents();after_events=len(hits);timer.stop()
        observations[name+'_restore_callback']={'elapsed_seconds':elapsed,'fingerprint_reads_on_ui_thread':bool(threads) and all(x==main_thread for x in threads),
            'timer_callbacks_during_restore':before_events,'timer_callbacks_after_process_events':after_events,
            'restored_value':int(np.asanyarray(nib.load(image).dataobj)[0,0,0]),'synthetic_delay_per_storage_read_seconds':.2}
        stop(window.window)
    window=batch.BatchStatusWindow(str(t),qt);instances.append(window)
    for timer in window.window.findChildren(QtCore.QTimer):timer.stop()
    row=batch._row('Import','合成批量任务',{'status':'closed','completed':1,'failed':0,'receipt_failed':1,'receipt_failed_cases':['caseA'],'total':1,'updated_at_epoch':1760000000},t/'import-status.json')
    # Use the viewer's actual snapshot application contract.
    window.refresh([row])
    capture('batch-receipt-warning',window.window)
    observations['receipt_warning']={'note':batch.receipt_failure_note(row),'rows':window.table.rowCount()}
    stop(window.window)
(args.out/'ui-observations.json').write_text(json.dumps(observations,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(observations,ensure_ascii=False,indent=2))
