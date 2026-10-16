"""Read-only review of 336e99d. Only synthetic temporary files are used.
Model execution/native Mimics create-delete are boundary stubs; all geometry,
publication, split, receipt, uncertainty, persistence and UI callbacks are real.
"""
import argparse, contextlib, gzip, hashlib, json, sys, tempfile, threading, time
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'tests'), str(ROOT/'tools'), str(ROOT/'runtime_py35'), str(ROOT)]
import test_all  # installs the explicitly stubbed native Mimics boundary
import numpy as np, nibabel as nib, pydicom
import nnunet_common as nc, nnunet_pipeline as nn, flexict_pipeline as flex
import create_mcs_batch as create, import_undo_mimics as undo
import mimics_bridge as bridge, flexict_active_learning_ui as al
import batch_status_viewer as batch_viewer


def run():
    out = {}
    with tempfile.TemporaryDirectory(prefix='mimics_review_20261017_') as tmp:
        root = Path(tmp)
        geometry = {'source_shape':[4,4,4], 'source_voxel_to_ras_matrix':np.eye(4).tolist()}
        class Mask:
            name='Liver'; guid='synthetic-guid'; number_of_pixels=1
            def get_voxel_buffer(self): return b'\x00\x01\x00'
        snapshot = create._capture_mask_snapshot(Mask())
        for failing in (False, True):
            case = 'receipt_failure' if failing else 'receipt_success'
            output=root/case/'out'; output.mkdir(parents=True)
            runtime=root/case/'runtime'; queue=runtime/'prepared_queue'; queue.mkdir(parents=True)
            work=root/case/'work'; work.mkdir(); final=output/(case+'.mcs')
            nc.write_json_atomic(work/'prepare_manifest.json', {'output_mcs':str(final)})
            nc.write_json_atomic(queue/(case+'.json'), {'case_id':case,'work_dir':str(work),'output_mcs':str(final)})
            nc.write_json_atomic(runtime/create.QUEUE_DONE_FILE, {'status':'done'})
            def native_create(_work, path):
                Path(path).write_bytes(b'synthetic-mcs')
                result={'output_mcs':str(final),'source_fingerprint':'synthetic-fingerprint'}
                result['receipt']=create._receipt_payload(case,result,[{'name':'Liver'}],[Mask()])
                return result
            logs=[]; previous_runtime=create._ACTIVE_RUNTIME_DIR
            try:
                with contextlib.ExitStack() as stack:
                    stack.enter_context(mock.patch.object(create,'create_mcs_from_manifest',side_effect=native_create))
                    stack.enter_context(mock.patch.object(create,'log_message',side_effect=lambda _p,m:logs.append(m)))
                    if failing: stack.enter_context(mock.patch.object(create,'write_import_receipt',side_effect=OSError('synthetic disk permission failure')))
                    code=create.main(str(output),runtime_dir=str(runtime))
            finally: create._ACTIVE_RUNTIME_DIR=previous_runtime
            receipt=output/('.'+case+'.import_receipt.json')
            out[case]={'code':code,'mcs_exists':final.is_file(),'receipt_exists':receipt.is_file(),
                       'status':nc.read_json(runtime/create.STATUS_FILE,{}), 'log':logs,
                       'receipt':nc.read_json(receipt,{}), 'work_dir_retained':work.exists()}
            if failing:
                out[case]['viewer_row']=batch_viewer._row('import','synthetic',out[case]['status'],runtime/create.STATUS_FILE)

        # Successful replacement and invalid candidates exercise the actual publisher.
        for name, data, truncated in [('valid',np.ones((4,4,4),dtype=np.uint8),False),
                                     ('truncated',np.ones((4,4,4),dtype=np.uint8),True),
                                     ('four_d',np.ones((4,4,4,2),dtype=np.uint8),False),
                                     ('fractional',np.full((4,4,4),.25,dtype=np.float32),False)]:
            d=root/name; d.mkdir(); dst=d/'prediction.nii.gz'; candidate=d/'candidate.nii.gz'
            nib.save(nib.Nifti1Image(np.zeros((4,4,4),dtype=np.uint8),np.eye(4)),dst)
            old=dst.read_bytes(); nib.save(nib.Nifti1Image(data,np.eye(4)),candidate)
            if truncated: candidate.write_bytes(gzip.compress(gzip.decompress(candidate.read_bytes())[:352]))
            try:
                result, backup=nn.publish_prediction_output(candidate,dst,geometry)
                out['publish_'+name]={'accepted':True,'backup_exists':bool(backup and backup.is_file()),
                                     'backup_matches_old':bool(backup and backup.read_bytes()==old),'backup_name':backup.name if backup else None}
            except Exception as exc:
                out['publish_'+name]={'accepted':False,'old_preserved':dst.read_bytes()==old,'error':str(exc)}

        # Count actual decompressed bytes across the reader, independent of wall-clock timing.
        path=root/'volume.nii.gz'; size=(64,64,96)
        data=np.random.default_rng(0).integers(0,4,size=size,dtype=np.uint8)
        nib.save(nib.Nifti1Image(data,np.eye(4)),path)
        original_read=gzip._GzipReader.read; counts={'bytes':0,'calls':0}
        def counted_read(self,*args,**kw):
            result=original_read(self,*args,**kw)
            if result: counts['bytes']+=len(result); counts['calls']+=1
            return result
        with mock.patch.object(gzip._GzipReader,'read',counted_read):
            start=time.monotonic(); nn.validate_prediction_nifti(path); elapsed=time.monotonic()-start
        out['gzip_validation_cost']={'shape':list(size),'payload_bytes':data.nbytes,'decompressed_bytes':counts['bytes'],
                                     'decompression_amplification':counts['bytes']/data.nbytes,'read_calls':counts['calls'],'seconds':elapsed,
                                     'indexed_gzip_available':__import__('importlib.util').util.find_spec('indexed_gzip') is not None}

        class Masks(list):
            def delete(self,mask): self.remove(mask)
        for name,sha,buf in [('legacy','',b'edited'),('failed_snapshot','',b'edited'),
                             ('edited',snapshot.sha256,b'edited'),('unchanged',snapshot.sha256,Mask().get_voxel_buffer())]:
            mask=SimpleNamespace(name='Liver',guid='synthetic-guid',number_of_pixels=1,get_voxel_buffer=lambda:buf)
            masks=Masks([mask]); receipt={'created_masks_v2':[{'name':'Liver','guid':'synthetic-guid','pixel_count':1,'sha256':sha}],'provenance_token':'synthetic'}
            with mock.patch.object(undo.mimics.data,'masks',masks),mock.patch.object(undo.runtime_common,'execute_mimics_transaction',side_effect=lambda _m,fn,_label:fn()):
                deleted,ambiguous,modified=undo._delete_owned_masks(receipt)
            out['undo_'+name]={'deleted':deleted,'modified':modified,'remaining':len(masks)}

        rows=[{'case_id':'c'+str(i),'patient_group':''} for i in range(4)]
        roots={'raw':root/'raw','preprocessed':root/'pre'}
        out['flexict_frozen_two']=flex._write_flexict_splits(rows[:2],{'val_cases':1},roots,'Dataset001_synthetic',frozen_val=['c0'])
        out['flexict_frozen_four']=flex._write_flexict_splits(rows,{'val_cases':1},roots,'Dataset002_synthetic',frozen_val=['c0'])
        req={'split_seed':1,'validation_fraction':.25,'val_cases':1}
        out['nnunet_frozen_four']=nn._split_folds(rows,req,['c0'])
        grouped=[{'case_id':'c0','patient_group':'p0'},{'case_id':'c1','patient_group':'p1'}]
        out['nnunet_frozen_two_patient_groups']=nn._split_folds(grouped,req,['c0'])
        dataset_request=nn.normalize_request({'task_name':'synthetic','task_id':'synthetic','dataset_id':123,
                                             'workspace':str(root/'nn-workspace'),'dataset_root':str(root/'pool'),
                                             'labels':[{'name':'Liver','id':1}],**req})
        name=nn._dataset_name(dataset_request); raw=root/'nn-raw'; pre=root/'nn-pre'
        nc.write_json_atomic(pre/name/'splits_final.json',[{'train':['c1'],'val':['c0']}])
        materialized=[]
        for row in grouped:
            image=root/(row['case_id']+'_image.nii.gz');label=root/(row['case_id']+'_label.nii.gz')
            nib.save(nib.Nifti1Image(np.zeros((4,4,4),dtype=np.uint8),np.eye(4)),image)
            nib.save(nib.Nifti1Image(np.ones((4,4,4),dtype=np.uint8),np.eye(4)),label)
            materialized.append({**row,'image':str(image),'label':str(label),'fingerprint':'synthetic-'+row['case_id']})
        nn._materialize_nnunet_raw(materialized,dataset_request,raw,pre)
        out['nnunet_writer_empty_train']=nc.read_json(pre/name/'splits_final.json',[])

        # Real public DICOM geometry functions, synthetic pydicom headers; no pixel decode.
        cases=[('regular',[[0,0,0],[0,0,1],[0,0,2]],[[1,1]]*3),
               ('linear_shear',[[0,0,0],[1,0,1],[2,0,2]],[[1,1]]*3),
               ('zigzag',[[0,0,0],[8,0,1],[0,0,2]],[[1,1]]*3),
               ('missing_ipp',[[0,0,0],None,[0,0,2]],[[1,1]]*3),
               ('varying_spacing',[[0,0,0],[0,0,1],[0,0,2]],[[1,1],[2,2],[1,1]]),
               ('partly_missing_spacing',[[0,0,0],[0,0,1],[0,0,2]],[[2,2],None,[3,3]]),
               ('all_missing_ipp',[None]*3,[[2,2]]*3)]
        observations=[]
        for name,positions,spacings in cases:
            folder=root/('dicom_'+name);folder.mkdir();headers={}
            for i,pos in enumerate(positions):
                path=folder/(str(i)+'.dcm');path.write_bytes(b'synthetic-header-fixture')
                ds=pydicom.Dataset();ds.Rows=ds.Columns=4;ds.SeriesInstanceUID='1.2.3';ds.ImageOrientationPatient=[1,0,0,0,1,0];ds.SliceThickness=1
                if spacings[i] is not None:ds.PixelSpacing=spacings[i]
                if pos is not None:ds.ImagePositionPatient=pos
                headers[str(path)]=ds
            try:
                with mock.patch.object(pydicom,'dcmread',side_effect=lambda path,**kw:headers[str(path)]):
                    shape=bridge.get_image_shape_from_dicom(str(folder));affine=bridge.get_image_affine_from_dicom(str(folder))
                observations.append({'case':name,'accepted':True,'shape':list(shape),'affine':affine.tolist(),'source_spacings':spacings})
            except Exception as exc:observations.append({'case':name,'accepted':False,'error':str(exc)})
        out['dicom_geometry']=observations

        labels=nc.normalize_labels([{'name':'Liver-Left','id':1},{'name':'Liver_Left','id':2}])
        seg=root/'labels'/'caseA'/'segmentations';seg.mkdir(parents=True)
        for label in labels:(seg/(label['name']+'.nii.gz')).write_bytes(b'name-only fixture')
        out['label_subfolder_exact']={label['name']:[p.name for p in nn._find_label_files(root/'labels','caseA',label['aliases'])] for label in labels}
        flat=root/'flat-labels';flat.mkdir()
        for label in labels:(flat/('caseA_'+label['name']+'.nii.gz')).write_bytes(b'name-only fixture')
        flat_result={}
        for label in labels:
            try: flat_result[label['name']]={'paths':[p.name for p in nn._find_label_files(flat,'caseA',label['aliases'])]}
            except Exception as exc: flat_result[label['name']]={'error':str(exc)}
        out['label_flat_exact']=flat_result

        state_dir=root/'state';state_dir.mkdir();messages=[];results=Queue()
        window=SimpleNamespace(_job_dir=str(state_dir),_state_write_results=results,status_label=SimpleNamespace(setText=messages.append),_selected_case=lambda:'A')
        window._set_case_state_async=lambda *args:al.ActiveLearningWindow._set_case_state_async(window,*args)
        held=al._annotation_state_guard(state_dir/'annotation_state.json',wait_seconds=0)
        start=time.monotonic();al.ActiveLearningWindow._mark_annotated(window);duration=time.monotonic()-start
        al._release_annotation_state_guard(held);first=results.get(timeout=4)
        out['async_state_responsive']={'callback_seconds':duration,'completion':first,'persisted':al.load_annotation_state(state_dir)}
        first_entered=threading.Event();let_first_finish=threading.Event();real_set=al.set_case_state;persisted_order=[]
        def reordered_set(job,case,value):
            if value=='annotated':first_entered.set();assert let_first_finish.wait(4)
            real_set(job,case,value)
            persisted_order.append(al.load_annotation_state(job)['cases'][case]['state'])
        with mock.patch.object(al,'set_case_state',side_effect=reordered_set):
            window._set_case_state_async('A','annotated','标记已标注');assert first_entered.wait(4)
            window._set_case_state_async('A','skipped','标记跳过');skipped=results.get(timeout=4)
            let_first_finish.set();annotated=results.get(timeout=4)
        out['async_state_click_order']={'click_order':['annotated','skipped'],'completion_order':[skipped,annotated],
                                       'observed_persistence_order':persisted_order,
                                       'final_state':al.load_annotation_state(state_dir)['cases']['A']['state'],
                                       'controlled_thread_schedule_only':True}

        # Exercise the entire AL pipeline using actual materialization + uncertainty.
        # Only model execution and GPU reservation are replaced; no torch or GPU needed.
        import test_flexict_integration as tf
        helper=tf.TestActiveLearningLifecycle()
        job,workspace,case_ids=helper._setup_al_job(str(root/'al'),cases_count=3)
        with mock.patch.object(flex,'_spawn_worker',side_effect=helper._fake_predict_worker([])),mock.patch.object(flex,'_acquire_local_gpu',return_value=None):
            code=flex.run_active_learning(job)
        first=nc.read_json(job/'status.json',{});old_dir=first.get('uncertainty_dir')
        out['al_initial']={'code':code,'ranked':[r['case'] for r in first.get('ranking',[])],'directory':old_dir}
        def partial(stage,params,*args,**kw):
            helper._fake_predict_worker([])(stage,params,*args,**kw)
            if '2d' in Path(params['model_folder']).name:(Path(params['output_path'])/'case01.nii.gz').unlink()
            return {}
        # The new directory contract uses second-resolution names. Freeze only
        # that naming clock, not status timestamps, to test uniqueness deterministically.
        old_stamp=Path(old_dir).name.removeprefix('uncertainty_run_');real_strftime=flex.time.strftime
        def same_stamp(fmt,*args):return old_stamp if fmt=='%Y%m%dT%H%M%S' else real_strftime(fmt,*args)
        with mock.patch.object(flex,'_spawn_worker',side_effect=partial),mock.patch.object(flex,'_acquire_local_gpu',return_value=None),mock.patch.object(flex.time,'strftime',side_effect=same_stamp):
            code=flex.run_active_learning(job)
        second=nc.read_json(job/'status.json',{})
        out['al_same_second_retry']={'code':code,'directory_reused':second.get('uncertainty_dir')==old_dir,
                                      'ranked':[r['case'] for r in second.get('ranking',[])],
                                      'excluded_case_count':second.get('excluded_case_count'),'naming_clock_only_controlled':True}
        # Wrong input grid shared by both models must now fail before uncertainty.
        def wrong_grid(stage,params,*args,**kw):
            helper._fake_predict_worker([])(stage,params,*args,**kw)
            for path in Path(params['output_path']).glob('*.nii.gz'):
                image=nib.load(path);data=np.asanyarray(image.dataobj);affine=image.affine.copy();affine[0,3]+=100
                nib.save(nib.Nifti1Image(data,affine),path)
            return {}
        with mock.patch.object(flex,'_spawn_worker',side_effect=wrong_grid),mock.patch.object(flex,'_acquire_local_gpu',return_value=None):
            code=flex.run_active_learning(job)
        failed=nc.read_json(job/'status.json',{})
        out['al_wrong_grid_rejected']={'code':code,'status':failed.get('status'),'error':failed.get('error'),
                                      'raw_retained_ranking':[r['case'] for r in al.load_ranking(job)],
                                      'gui_failed_job_actions_disabled_by_source':True}
        # Even a truly fresh run directory is insufficient: uncertainty is
        # saved before consensus, so a later per-case write failure leaves a
        # partial map that the directory-scanning ranker still consumes.
        real_save=nib.save; fresh_stamp=old_stamp+'_fresh_failure_probe'
        def fail_consensus(image,path,*args,**kw):
            if Path(path).name=='case01_consensus.nii.gz':
                raise OSError('synthetic consensus write failure after uncertainty write')
            return real_save(image,path,*args,**kw)
        def fresh_time(fmt,*args):return fresh_stamp if fmt=='%Y%m%dT%H%M%S' else real_strftime(fmt,*args)
        with mock.patch.object(flex,'_spawn_worker',side_effect=helper._fake_predict_worker([])),mock.patch.object(flex,'_acquire_local_gpu',return_value=None),mock.patch.object(flex.time,'strftime',side_effect=fresh_time),mock.patch.object(nib,'save',side_effect=fail_consensus):
            code=flex.run_active_learning(job)
        partial_status=nc.read_json(job/'status.json',{})
        out['al_fresh_partial_write_failure']={'code':code,'ranked':[r['case'] for r in partial_status.get('ranking',[])],
                                               'excluded_case_count':partial_status.get('excluded_case_count'),
                                               'fresh_directory':partial_status.get('uncertainty_dir')!=old_dir,
                                               'case01_consensus_exists':(Path(partial_status['uncertainty_dir'])/'case01_consensus.nii.gz').exists(),
                                               'failure_injected_at_save_boundary_only':True}
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    result=run();args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
