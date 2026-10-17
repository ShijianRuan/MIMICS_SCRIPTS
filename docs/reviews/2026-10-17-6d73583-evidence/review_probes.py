"""Synthetic, read-only audit of 6d73583. Native Mimics/model/GPU boundaries
are replaced; receipt replay, publication, restore, geometry, splitting,
file locks, JSON writes and the AL calculation pipeline run as production code.
"""
import argparse, gzip, hashlib, json, sys, tempfile, threading, time
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from unittest import mock
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'tests'),str(ROOT/'tools'),str(ROOT/'runtime_py35'),str(ROOT)]
import test_all
import numpy as np,nibabel as nib,pydicom
import create_mcs_batch as create,nnunet_common as nc,nnunet_pipeline as nn
import nnunet_jobs as jobs,flexict_pipeline as flex,flexict_active_learning_ui as al
import mimics_bridge as bridge,batch_status_viewer as batch

def run():
    out={}
    with tempfile.TemporaryDirectory(prefix='mimics_review_6d73583_') as tmp:
        root=Path(tmp);geo={'source_shape':[4,4,4],'source_voxel_to_ras_matrix':np.eye(4).tolist()}
        def nifti(path,value=1):
            path.parent.mkdir(parents=True,exist_ok=True)
            nib.save(nib.Nifti1Image(np.full((4,4,4),value,dtype=np.uint8),np.eye(4)),path)
        # Receipt replay: a correct historical receipt is not evidence that
        # the NEW pending import's receipt has already been written.
        for existing in (False,True):
            folder=root/('existing_receipt' if existing else 'no_receipt');folder.mkdir()
            project=folder/'caseA.mcs';project.write_bytes(b'new project generation')
            payload=create._import_receipt_payload('caseA',str(project),['Liver'],{'source_fingerprint':'new-source'},
                mask_objects=[create._MaskIdentity('Liver','new-mask-guid',1,'sha256:new')])
            receipt=Path(create._receipt_path_for(str(project),'caseA'))
            old={**payload,'source_fingerprint':'old-source','created_masks_v2':[{'name':'Liver','guid':'old-mask-guid','sha256':'sha256:old','pixel_count':1}]}
            if existing:nc.write_json_atomic(receipt,old)
            runtime=folder/'runtime';runtime.mkdir()
            create._write_pending_receipts(str(runtime),{'caseA':{'receipt_path':str(receipt),'payload':payload}})
            healed=create.flush_pending_receipts(str(runtime))
            result=nc.read_json(receipt,{})
            out['receipt_replay_existing' if existing else 'receipt_replay_missing']={
                'healed':healed,'pending':create._read_pending_receipts(str(runtime)),
                'written_guid':result['created_masks_v2'][0]['guid'],'expected_guid':'new-mask-guid',
                'source_fingerprint':result['source_fingerprint'],'project_unchanged':project.read_bytes()==b'new project generation'}
        row=batch._row('Import','synthetic',{'status':'closed','completed':1,'failed':0,'receipt_failed':1,'receipt_failed_cases':['caseA']},root/'synthetic-status.json')
        out['receipt_visible_note']={'row':row,'note':batch.receipt_failure_note(row)}

        # Output conflict guards distinguish changed-existing vs absent-at-start.
        for initially_present in (True,False):
            folder=root/('changed_existing' if initially_present else 'appeared_later');folder.mkdir()
            dst=folder/'prediction.nii.gz';candidate=folder/'candidate.nii.gz';nifti(candidate,2)
            if initially_present:nifti(dst,1)
            expected=nc.file_fingerprint(dst)
            nifti(dst,3);newer=dst.read_bytes()
            try:
                published,backup=nn.publish_prediction_output(candidate,dst,geo,expected_previous_fingerprint=expected)
                accepted=True;error='';candidate_parked=None
            except nn.PredictionOutputConflict as exc:
                accepted=False;error=str(exc);candidate_parked=exc.candidate_path.is_file()
            out['publish_changed_existing' if initially_present else 'publish_appeared_later']={
                'initial_fingerprint':expected,'accepted':accepted,'newer_target_preserved':dst.read_bytes()==newer,
                'candidate_parked':candidate_parked,'error':error,'final_value':int(np.asanyarray(nib.load(dst).dataobj)[0,0,0])}
        # Restore is positive with a recorded digest and conflict-protected after edit.
        folder=root/'restore';folder.mkdir();dst=folder/'current.nii.gz';old=folder/'old.nii.gz';nifti(dst,2);nifti(old,1)
        status=folder/'status.json';nc.write_json_atomic(status,{'output_path':str(dst),'replaced_previous_output':str(old),'published_output_fingerprint':nc.file_fingerprint(dst),'status':'completed'})
        jobs.restore_replaced_output(status);restored=nc.read_json(status,{})
        out['restore_normal']={'current_value':int(np.asanyarray(nib.load(dst).dataobj)[0,0,0]),'kept_previous_exists':Path(restored['replaced_previous_output']).is_file()}
        nifti(dst,3);before=dst.read_bytes()
        try:jobs.restore_replaced_output(status);blocked=False
        except RuntimeError:blocked=True
        out['restore_after_manual_edit']={'blocked':blocked,'current_preserved':before==dst.read_bytes()}

        # Actual decompression bytes verify the optimization without timing claims.
        path=root/'volume.nii.gz';data=np.random.default_rng(0).integers(0,4,(64,64,96),dtype=np.uint8)
        nib.save(nib.Nifti1Image(data,np.eye(4)),path);original=gzip._GzipReader.read;count=[0]
        def counted(self,*args,**kw):
            result=original(self,*args,**kw)
            if result:count[0]+=len(result)
            return result
        with mock.patch.object(gzip._GzipReader,'read',counted):nn.validate_prediction_nifti(path)
        out['gzip_cost']={'payload_bytes':data.nbytes,'decompressed_bytes':count[0],'amplification':count[0]/data.nbytes}
        rows=[{'case_id':'c0','patient_group':'p0'},{'case_id':'c1','patient_group':'p1'}]
        request={'split_seed':1,'val_cases':1,'validation_fraction':.25}
        out['frozen_grouped']=nn._split_folds(rows,request,['c0'])
        out['frozen_four']=nn._split_folds([{'case_id':'c'+str(i)} for i in range(4)],request,['c0'])
        out['frozen_default_expansion']=nn._split_folds([{'case_id':'c'+str(i)} for i in range(8)],{**request,'val_cases':0},['c0'])

        # All geometry calls use real public parser/affine code and synthetic headers.
        dicom=[]
        for name,spacings,iop,positions in [
            ('regular',[[1,1]]*3,[1,0,0,0,1,0],[[0,0,z] for z in range(3)]),
            ('partly_missing_spacing',[[2,2],None,[3,3]],[1,0,0,0,1,0],[[0,0,z] for z in range(3)]),
            ('all_missing_spacing',[None]*3,[1,0,0,0,1,0],[[0,0,z] for z in range(3)]),
            ('all_missing_position',[[1,1]]*3,[1,0,0,0,1,0],[None]*3),
            ('non_unit_direction',[[1,1]]*3,[2,0,0,0,1,0],[[0,0,z] for z in range(3)]),
            ('non_orthogonal_direction',[[1,1]]*3,[1,0,0,1,1,0],[[0,0,z] for z in range(3)])]:
            folder=root/name;folder.mkdir();headers={}
            for i,position in enumerate(positions):
                p=folder/(str(i)+'.dcm');p.write_bytes(b'synthetic-header');ds=pydicom.Dataset();ds.SeriesInstanceUID='1.2.3';ds.Rows=ds.Columns=4;ds.SliceThickness=1;ds.ImageOrientationPatient=iop
                if position is not None:ds.ImagePositionPatient=position
                if spacings[i] is not None:ds.PixelSpacing=spacings[i]
                headers[str(p)]=ds
            try:
                with mock.patch.object(pydicom,'dcmread',side_effect=lambda path,**kw:headers[str(path)]):affine=bridge.get_image_affine_from_dicom(str(folder))
                dicom.append({'case':name,'accepted':True,'affine':affine.tolist()})
            except Exception as exc:dicom.append({'case':name,'accepted':False,'error':str(exc)})
        out['dicom']=dicom
        for sep in ('_','__'):
            folder=root/('labels'+sep);folder.mkdir()
            for label in ('Liver-Left','Liver_Left'):(folder/('caseA'+sep+label+'.nii.gz')).write_bytes(b'name-only fixture')
            result={}
            for label in ('Liver-Left','Liver_Left'):
                try:result[label]={'files':[p.name for p in nn._find_label_files(folder,'caseA',[label])]}
                except Exception as exc:result[label]={'error':str(exc)}
            out['flat_labels_double' if sep=='__' else 'flat_labels_single']=result

        # One FIFO worker fixes click order. Busy tracking must also cover
        # the second dequeued write before the exit drain declares success.
        state=root/'states';state.mkdir();results=Queue();messages=[]
        window=SimpleNamespace(_job_dir=str(state),_state_write_results=results,_state_write_queue=Queue(),
            _state_write_thread=None,_state_write_busy=False,status_label=SimpleNamespace(setText=messages.append))
        window._run_state_writes=lambda:al.ActiveLearningWindow._run_state_writes(window)
        real_set=al.set_case_state;first_enter=threading.Event();second_enter=threading.Event();release_first=threading.Event();release_second=threading.Event()
        def held_write(job,case,value):
            if value=='annotated':first_enter.set();assert release_first.wait(4)
            else:second_enter.set();assert release_second.wait(4)
            real_set(job,case,value)
        with mock.patch.object(al,'set_case_state',side_effect=held_write):
            al.ActiveLearningWindow._set_case_state_async(window,'A','annotated','标记已标注');assert first_enter.wait(4)
            al.ActiveLearningWindow._set_case_state_async(window,'A','skipped','标记跳过');release_first.set();assert second_enter.wait(4)
            pending=al.ActiveLearningWindow.wait_for_state_writes(window,timeout=.05)
            at_exit=al.load_annotation_state(state)['cases']['A']['state']
            out['fifo_exit_drain']={'pending_reported':pending,'queue_size':window._state_write_queue.qsize(),'busy_flag':window._state_write_busy,'second_write_inflight':True,'persisted_before_second_release':at_exit}
            release_second.set();results.get(timeout=4);results.get(timeout=4)
        out['fifo_final_state']=al.load_annotation_state(state)['cases']['A']['state']

        # Full AL chain: normal retry failure handling and prefix-colliding IDs.
        import test_flexict_integration as tf
        helper=tf.TestActiveLearningLifecycle()
        def run_al(folder,*,rename=False,fail_save=False,partial=False,stamp=None):
            job,workspace,_=helper._setup_al_job(str(folder),cases_count=3)
            if rename:(folder/'pool'/'case02').rename(folder/'pool'/'case01_phase2')
            real_save=nib.save;real_time=flex.time.strftime
            def worker(stage,params,*args,**kw):
                value=helper._fake_predict_worker([])(stage,params,*args,**kw)
                if partial and '2d' in Path(params['model_folder']).name:(Path(params['output_path'])/'case01.nii.gz').unlink()
                return value
            def save(image,path,*args,**kw):
                if fail_save and Path(path).name=='case01_consensus.nii.gz':raise OSError('synthetic consensus write failure')
                return real_save(image,path,*args,**kw)
            def clock(fmt,*args):return stamp if stamp and fmt=='%Y%m%dT%H%M%S' else real_time(fmt,*args)
            with mock.patch.object(flex,'_spawn_worker',side_effect=worker),mock.patch.object(flex,'_acquire_local_gpu',return_value=None),mock.patch.object(nib,'save',side_effect=save),mock.patch.object(flex.time,'strftime',side_effect=clock):code=flex.run_active_learning(job)
            status=nc.read_json(job/'status.json',{})
            return {'code':code,'status':status.get('status'),'ranked':[r['case'] for r in status.get('ranking',[])],
                'excluded_case_count':status.get('excluded_case_count'),'error':status.get('error'),
                'uncertainty_dir':status.get('uncertainty_dir'),'failed_fragments':[p.name for p in job.glob('uncertainty_run_*/failed/*.nii.gz')]}
        out['al_partial_write']=run_al(root/'al_normal',fail_save=True)
        out['al_prefix_collision']=run_al(root/'al_prefix',rename=True,fail_save=True)
        # Same-second separate invocations really produce distinct output dirs.
        first=run_al(root/'al_clock1',stamp='20000101T000000')
        second=run_al(root/'al_clock2',stamp='20000101T000000')
        out['al_uuid_names']={'names':[Path(first['uncertainty_dir']).name,Path(second['uncertainty_dir']).name],
            'both_completed':first['code']==second['code']==0,'different':Path(first['uncertainty_dir']).name!=Path(second['uncertainty_dir']).name}
    return out

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    result=run();args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
