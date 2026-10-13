"""Read-only audit probes for 96448ae; synthetic temporary state only.

No real Mimics, GPU, model execution, SSH or repair operation. Production
functions are exercised up to filesystem/dispatch boundaries. This records
observations rather than disguising expected current defects as passing tests.
"""
import argparse,contextlib,importlib.util,json,os,sys,tempfile,time
from pathlib import Path
from unittest import mock
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'tests'),str(ROOT/'tools'),str(ROOT/'runtime_py35'),str(ROOT)]
import test_all  # repository's existing Mimics stub
import nnunet_common as nc,nnunet_jobs as nj
import io_path_setup_ui as io,import_drop_window as drop
import single_case_import_worker as worker,import_undo_mimics as undo
import flexict_active_learning_ui as al,flexict_mimics as fm,remote_compute_ui as remote
from PySide6 import QtCore,QtGui,QtWidgets

def run():
 observations={}
 with tempfile.TemporaryDirectory(prefix='mimics_review_20261013_') as temp:
  t=Path(temp)
  # UI19: the GUI normalizes, create_job normalizes again.
  names=['a'*96+'肝脏','a'*96+'肾脏']
  first=[];second=[];jobs=[]
  for i,name in enumerate(names):
   req={'operation':'train','task_name':name,'task_id':name,'workspace':str(t/'nnunet'),
        'dataset_root':str(t/'data'),'labels':[{'name':'Liver','id':1}], 'job_id':'synthetic_job_'+str(i)}
   x=nc.normalize_request(req);y=nc.normalize_request(x)
   first.append(x['task_id']);second.append(y['task_id'])
   with mock.patch.object(nj,'_launch',return_value=mock.Mock(pid=0)),mock.patch.object(nj,'process_start_marker',return_value='synthetic'):
    job=nj.create_job(x)
   jobs.append(nc.read_json(Path(job['job_dir'])/'request.json',{})['task_id'])
  observations['task_id_roundtrip']={'names':names,'gui_normalized':first,'normalized_twice':second,'job_persisted':jobs,
      'first_distinct':first[0]!=first[1],'job_ids_collide':jobs[0]==jobs[1],'idempotent':first==second}
  observations['ascii_task_collision']={'names':['a'*96+'x','a'*96+'y'],
    'ids':[nc.stable_task_identifier('a'*96+'x'),nc.stable_task_identifier('a'*96+'y')]}
  try:
   labels=nc.normalize_labels([{'name':'肝脏','id':1},{'name':'脾脏','id':2}])
   observations['chinese_multi_organ_labels']={'result':labels}
  except Exception as exc:observations['chinese_multi_organ_labels']={'error':str(exc)}
  # UI24: real summary -> real missing-case predicate, not a hand-authored map.
  data=t/'batch';data.mkdir()
  for name in ['CaseA','CaseB','CaseC']:
   case=data/name;case.mkdir();(case/'ct.nii.gz').write_bytes(b'synthetic filename-only prescan fixture')
  for case,mask in [('CaseA','Liver'),('CaseC','Spleen')]:
   seg=data/case/'segmentations';seg.mkdir();(seg/(mask+'.nii.gz')).write_bytes(b'synthetic filename-only fixture')
  summary=io.summarize_dataset(str(data))
  observations['maskless_case_warning']={'case_count':summary['case_count'],'case_mask_names':summary['case_mask_names'],
    'selected':['Liver'],'warned':drop.cases_missing_selected_masks(summary['case_mask_names'],['Liver']),
    'expected_images_only':['CaseB','CaseC'],
    'selected_multiple':['Liver','Spleen'],
    'warned_multiple':drop.cases_missing_selected_masks(summary['case_mask_names'],['Liver','Spleen'])}
  observations['mask_case_sensitive_ui']={'ui_warned':drop.cases_missing_selected_masks({'CaseA':['liver']},['Liver']),
    'worker_selected':worker._filter_masks([{'name':'liver','path':'synthetic.nii.gz'}],'Liver')}
  # UI28: the real worker update replaces the bootstrap record.
  project=t/'project';status_dir=project/'.mimics_runtime'/'drop_import';status_dir.mkdir(parents=True)
  out=t/'custom_output';out.mkdir();status=status_dir/'synthetic_0_status.json'
  status.write_text(json.dumps({'output_path':str(out),'updated_at_epoch':1,'status':'running'}))
  receipt=out/'.CaseA.import_receipt.json';receipt.write_text(json.dumps({'schema_version':'mimics_import_receipt.v1','created_at_epoch':1,'case_id':'CaseA'}))
  with mock.patch.object(undo,'_project_root',return_value=str(project)):
   before=undo._recent_drop_output_dirs()
   worker._write_status(status,'completed','completed',case_id='CaseA',output_mcs=str(out/'CaseA.mcs'),completed=1,failed=0,total=1)
   after=undo._recent_drop_output_dirs();found=undo.find_latest_receipt()[0]
  observations['receipt_after_worker_update']={'before_found':str(out) in before,'final_status_keys':sorted(json.loads(status.read_text())),
    'after_found':str(out) in after,'found_receipt':found}
  # Editing an owned Mask does not change its GUID/inventory. Exercise the
  # divergence check and deletion dispatch; native delete is a boundary stub.
  from types import SimpleNamespace
  owned=SimpleNamespace(name='Liver',guid='synthetic-mask-guid',voxels=[0,1,0])
  owned.voxels=[0,0,1]  # same foreground volume, manually corrected location
  class Masks(list):
   def delete(self,mask):self.remove(mask)
  masks=Masks([owned])
  receipt_v2={'schema_version':undo.RECEIPT_SCHEMA_VERSION,'created_masks':['Liver'],
    'created_masks_v2':[{'guid':owned.guid,'name':owned.name}],'provenance_token':'synthetic-token'}
  with mock.patch.object(undo.mimics.data,'masks',masks),mock.patch.object(undo.runtime_common,'execute_mimics_transaction',side_effect=lambda _m,operation,_label:operation()):
   diverged,reason=undo._session_diverged(receipt_v2)
   deleted,ambiguous=undo._delete_owned_masks(receipt_v2)
  observations['edited_owned_mask_undo']={'manual_voxel_edit':True,'diverged_before_delete':diverged,
   'reason':reason,'deleted_names':deleted,'ambiguous':ambiguous,'remaining_masks':len(masks),
   'native_mimics_delete_stubbed':True}
  # UI31: measure the production Mimics callback while a real OS lock is held.
  job=t/'ranking';job.mkdir();p=job/'annotation_state.json'
  held=al._annotation_state_guard(p,wait_seconds=0)
  try:
   start=time.monotonic();fm._al_mark_viewed(str(job),'CaseA');elapsed=time.monotonic()-start
  finally:al._release_annotation_state_guard(held)
  observations['mimics_guard_contention']={'callback_seconds':round(elapsed,4),'viewed_file_written':p.exists(),
    'call_chain':'Mimics conversion monitor -> _al_finish_conversion -> _al_mark_viewed -> lock retry/sleep'}
  # Revalidate the previous data-safety findings on the current checkout.
  import numpy as np,nibabel as nib,pydicom,mimics_bridge as bridge
  import flexict_pipeline as flex,remote_training_controller as controller
  dicom=[]
  for name,zs in [('nonuniform',[0,1,3]),('duplicate',[0,1,1])]:
   folder=t/name;folder.mkdir();headers={}
   for i,z in enumerate(zs):
    f=folder/('slice%d.dcm'%i);f.write_bytes(b'synthetic header fixture')
    h=pydicom.Dataset();h.Rows=h.Columns=4;h.SeriesInstanceUID='1.2.3.4'
    h.ImagePositionPatient=[0,0,z];h.ImageOrientationPatient=[1,0,0,0,1,0]
    h.PixelSpacing=[1,1];h.SliceThickness=1;headers[str(f)]=h
   try:
    with mock.patch.object(pydicom,'dcmread',side_effect=lambda f,**kw:headers[str(f)]):
     shape=bridge.get_image_shape_from_dicom(str(folder));affine=bridge.get_image_affine_from_dicom(str(folder))
    dicom.append({'input_z':zs,'inferred_z':[float((affine@[0,0,i,1])[2]) for i in range(shape[2])],'rejected':False})
   except Exception as exc:dicom.append({'input_z':zs,'rejected':True,'error':str(exc)})
  observations['dicom_irregular_geometry']=dicom
  rows=[]
  for i in range(4):
   image=t/('c%d-image.nii.gz'%i);label=t/('c%d-label.nii.gz'%i)
   nib.save(nib.Nifti1Image(np.zeros((2,2,2)),np.eye(4)),image)
   nib.save(nib.Nifti1Image(np.ones((2,2,2),dtype=np.uint8),np.eye(4)),label)
   rows.append({'case_id':'c%d'%i,'image':image,'label':label,'fingerprint':'synthetic-%d'%i})
  req=nc.normalize_request({'operation':'train','workspace':str(t/'split-workspace'),'dataset_root':str(data),
     'task_name':'synthetic_task','dataset_id':41,'labels':[{'name':'Liver','id':1}],
     'cases':[r['case_id'] for r in rows],'val_cases':1,'validation_fraction':0.25})
  roots={'raw':t/'raw','preprocessed':t/'preprocessed'};dataset_name=flex._dataset_name(req)
  split=roots['preprocessed']/dataset_name/'splits_final.json'
  nc.write_json_atomic(split,[{'train':['c1','c2','c3'],'val':['c0']}])
  dataset_dir,_=flex._materialize_flexict_raw(rows,req,roots)
  shared=nc.read_json(split,[])
  flex._write_flexict_splits(rows,req,roots,dataset_dir.name)
  final=nc.read_json(split,[])
  observations['frozen_validation_after_retrain']={'previous_val':['c0'],'after_shared':shared,'after_flexict':final,
    'previous_validation_entered_train':'c0' in final[0]['train']}
  grouped=[dict(r,patient_group='A' if i<2 else 'B') for i,r in enumerate(rows)]
  try:
   train,val=flex._split_train_val(grouped,dict(req,val_case_ids=['c0','c2']))
   observations['empty_group_split']={'rejected':False,'train':train,'val':val}
  except ValueError as exc:observations['empty_group_split']={'rejected':True,'error':str(exc)}
  # Serialize with the actual remote preparation function; no SSH is involved.
  import tools.flexict_pipeline as remote_flex
  request_path=t/'remote-flex-request.json';status_path=t/'remote-flex-status.json';job_dir=t/'remote-flex-job';job_dir.mkdir()
  remote_request=dict(req,configuration='pair',patient_groups={'c0':'A','c1':'A','c2':'B','c3':'B'},val_case_ids=['c0'])
  nc.write_json_atomic(request_path,remote_request)
  with mock.patch.object(remote_flex,'prepare_source_grid_cases',return_value=grouped):
   controller._prepare_flexict({'job_dir':str(job_dir),'status_path':str(status_path),'request_path':str(request_path)},t/'remote-flex-bundle')
  packed=nc.read_json(t/'remote-flex-bundle'/'remote_request.json',{})['pipeline_request']
  local_train,local_val=flex._split_train_val(grouped,remote_request)
  remote_train,remote_val=flex._split_train_val(packed['prepared_cases'],packed)
  observations['remote_patient_group_handoff']={'local_split':{'train':local_train,'val':local_val},
     'remote_prepared_keys':sorted(packed['prepared_cases'][0]),'remote_request_retains_groups':packed.get('patient_groups'),
     'remote_split':{'train':remote_train,'val':remote_val},'patient_A_leaks':('c0' in remote_val and 'c1' in remote_train)}
  # Remote output publication has no validation/overwrite protection here.
  download=t/'download';download.mkdir();(download/'prediction.nii.gz').write_bytes(b'new unvalidated synthetic result')
  destination=t/'prediction.nii.gz';destination.write_bytes(b'previous reviewed result')
  output_request=t/'output-request.json';nc.write_json_atomic(output_request,{'output_path':str(destination)})
  controller._apply_remote_prediction_output({'request_path':str(output_request)},'nnunet_infer',download,{})
  observations['remote_output_overwrite']={'previous_preserved':destination.read_bytes()==b'previous reviewed result',
     'replacement_bytes':destination.read_text(),'replacement_is_valid_nifti':False}
  # The reserved background key survives normalize_labels but erases class 0.
  import nnunet_pipeline as nnpipe
  background_req=nc.normalize_request(dict(req,task_id='background_probe',task_name='background_probe',dataset_id=42,labels=[{'name':'background','id':1}]))
  background_dir,_=nnpipe._materialize_nnunet_raw(rows,background_req,t/'background_raw',t/'background_preprocessed')
  observations['reserved_background_label']={'accepted':True,'dataset_labels':nc.read_json(background_dir/'dataset.json',{})['labels']}
  # UI18: the label reads global memory; application reads the current state.
  import nninteractive_mimics as nnm
  target=mock.Mock(name='synthetic-target');target.name='肝脏'
  state={'write_mode':'in_place','source_name':'肝脏','target_name':'肝脏','interactions':[]}
  menu=[]
  def question(**kw):menu.append(kw);return nnm.BUTTON_FINISH
  with mock.patch.object(nnm,'_SESSION_WRITE_MODE','derived_copy'),mock.patch.object(nnm,'_prompt_buttons_for_profile',return_value=[]),mock.patch.object(nnm.mimics.dialogs,'question_box',side_effect=question):
   nnm._async_prompt_menu(target,state)
   applied_target=nnm._choose_completed_result_target(None,target,state,{})
  observations['nninteractive_write_mode_caption']={'session_memory':'derived_copy','current_state_mode':'in_place',
    'menu_message':menu[0]['message'],'target_reused':applied_target is target}
  # Sources are not always write-once: mutate a DICOM-like child in place.
  source_dir=t/'mutable_source';source_dir.mkdir();child=source_dir/'slice.dcm';child.write_bytes(b'old')
  with mock.patch.dict(os.environ,{'MIMICS_PATH_SIGNATURE_CACHE_DIR':str(t/'signature_cache')}):
   sig_before=nc.path_signature(source_dir);dir_mtime=source_dir.stat().st_mtime_ns
   child.write_bytes(b'changed image bytes, different size')
   sig_after=nc.path_signature(source_dir)
  with mock.patch.dict(os.environ,{'MIMICS_PATH_SIGNATURE_CACHE_DIR':'off'}):sig_fresh=nc.path_signature(source_dir)
  observations['in_place_directory_mutation']={'directory_mtime_unchanged':dir_mtime==source_dir.stat().st_mtime_ns,
      'cached_signature_unchanged':sig_before==sig_after,'fresh_signature_differs':sig_before!=sig_fresh,
      'cache_max_age_seconds':nc.PATH_SIGNATURE_CACHE_SECONDS}
  # Same voxel shape does not imply two model outputs share physical coordinates.
  uncertain_path=ROOT/'integrations'/'flexict-finetune'/'uncertainty'/'uncertainty.py'
  spec=importlib.util.spec_from_file_location('audit_uncertainty',uncertain_path)
  uncertain=importlib.util.module_from_spec(spec);spec.loader.exec_module(uncertain)
  aff=np.eye(4);shifted=aff.copy();shifted[0,3]=100
  mask=np.ones((2,2,2),dtype=np.uint8)
  p1=t/'same-grid-mask.nii.gz';p2=t/'shifted-grid-mask.nii.gz'
  nib.save(nib.Nifti1Image(mask,aff),p1);nib.save(nib.Nifti1Image(mask,shifted),p2)
  try:
   arrays,reference_affine,_=uncertain.load_masks([str(p1),str(p2)])
   observations['uncertainty_affine_mismatch']={'accepted':True,'array_shape':list(arrays.shape),
      'physical_shift_mm':100,'reference_affine':reference_affine.tolist()}
  except Exception as exc:observations['uncertainty_affine_mismatch']={'accepted':False,'error':str(exc)}
  # UI29: two advanced fields never appear in the non-default summary.
  observations['hidden_remote_identity']={'profile':{'runtime_image':'private/synthetic-runtime:2','remote_root':'different-storage'},
      'summary':remote._non_default_summary({'runtime_image':'private/synthetic-runtime:2','remote_root':'different-storage'})}
 return observations

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
 result=run();args.out.parent.mkdir(parents=True,exist_ok=True)
 args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(result,ensure_ascii=False,indent=2))
