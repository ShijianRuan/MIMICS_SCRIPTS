"""Read-only audit of fda3444. All files/cases are synthetic temporary data.
Native Mimics creation/deletion and job launch are boundary stubs; production
queue/receipt, publication, parsing, locking and splitting are exercised.
"""
import argparse,ast,hashlib,importlib.util,json,os,sys,tempfile,time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'tests'),str(ROOT/'tools'),str(ROOT/'runtime_py35'),str(ROOT)]
import test_all
import numpy as np,nibabel as nib,pydicom
import nnunet_common as nc,nnunet_pipeline as nn,flexict_pipeline as flex
import create_mcs_batch as create,import_undo_mimics as undo
import io_path_setup_ui as io,import_drop_window as drop,single_case_import_worker as sw
import flexict_active_learning_ui as al,flexict_mimics as fm,remote_compute_ui as remote
import nninteractive_mimics as ni,mimics_bridge as bridge

def run():
 out={}
 with tempfile.TemporaryDirectory(prefix='mimics_review_20261016_') as temp:
  t=Path(temp)
  names=['a'*96+'肝脏','a'*96+'肾脏','a'*96+'x','a'*96+'y']
  ids=[nc.stable_task_identifier(x) for x in names]
  out['R01_identifier']={'ids':ids,'lengths':[len(x) for x in ids],'all_distinct':len(set(ids))==4,'idempotent':all(nc.stable_task_identifier(x)==x for x in ids)}
  out['R06_chinese_labels']={'normalized':nc.normalize_labels([{'name':'肝脏','id':1},{'name':'脾脏','id':2}])}
  try:nc.normalize_labels([{'name':'background','id':1}]);out['R16_background']={'rejected':False}
  except ValueError as e:out['R16_background']={'rejected':True,'error':str(e)}
  data=t/'batch';data.mkdir()
  for case in ['A','B','C']:
   folder=data/case;folder.mkdir();(folder/'ct.nii.gz').write_bytes(b'filename-only fixture')
  for case,name in [('A','Liver'),('C','Spleen')]:
   folder=data/case/'segmentations';folder.mkdir();(folder/(name+'.nii.gz')).write_bytes(b'filename-only fixture')
  summary=io.summarize_dataset(str(data))
  out['R02_real_scanner']={'case_count':summary['case_count'],'mapping':summary['case_mask_names'],'liver':drop.cases_partial_mask_report(summary['case_mask_names'],['Liver']),'two_masks':drop.cases_partial_mask_report(summary['case_mask_names'],['Liver','Spleen'])}
  status=t/'import-status.json';nc.write_json_atomic(status,{'output_path':str(t/'custom-output'),'source_path':str(data)})
  sw._write_status(status,'completed','completed',output_mcs=str(t/'custom-output'/'A.mcs'))
  out['R03_status_context']=nc.read_json(status,{})
  rank=t/'ranking';rank.mkdir();state=rank/'annotation_state.json';held=al._annotation_state_guard(state,wait_seconds=0)
  oldpending=fm._AL_PENDING_VIEWED[:];fm._AL_PENDING_VIEWED[:]=[]
  try:
   start=time.monotonic();fm._al_mark_viewed(str(rank),'A');elapsed=time.monotonic()-start;queued=list(fm._AL_PENDING_VIEWED)
  finally:al._release_annotation_state_guard(held)
  fm._al_mark_viewed(str(rank),'A');fm._AL_PENDING_VIEWED[:]=oldpending
  out['R04_nonblocking_marker']={'seconds':elapsed,'queued':queued,'after_unlock':nc.read_json(state,{})}
  held=al._annotation_state_guard(state,wait_seconds=0)
  messages=[]
  window=SimpleNamespace(_selected_case=lambda:'A',_job_dir=str(rank),status_label=SimpleNamespace(setText=lambda x:messages.append(x)),_job_selected=lambda:messages.append('refreshed'))
  try:
   start=time.monotonic();al.ActiveLearningWindow._mark_annotated(window);elapsed=time.monotonic()-start
  finally:al._release_annotation_state_guard(held)
  out['external_ui_lock_wait']={'seconds':elapsed,'messages':messages}

  out['R05_remote_summary']=remote._non_default_summary({'runtime_image':'synthetic/runtime:2','remote_root':'other-storage'})
  target=SimpleNamespace(name='Liver');msgs=[]
  with mock.patch.object(ni,'_SESSION_WRITE_MODE','derived_copy'),mock.patch.object(ni,'_prompt_buttons_for_profile',return_value=[]),mock.patch.object(ni.mimics.dialogs,'question_box',side_effect=lambda **kw:msgs.append(kw) or ni.BUTTON_FINISH):
   ni._async_prompt_menu(target,{'write_mode':'in_place','interactions':[]})
  out['R11_effective_mode']={'message':msgs[0]['message']}
  source=t/'mutable';source.mkdir();child=source/'slice.dcm';child.write_bytes(b'old');before=nc.path_signature(source);mtime=source.stat().st_mtime_ns;child.write_bytes(b'new bytes larger');after=nc.path_signature(source)
  out['R14_mutated_source']={'parent_mtime_unchanged':mtime==source.stat().st_mtime_ns,'signature_changed':before!=after}
  geometry={'source_shape':[4,4,4],'source_voxel_to_ras_matrix':np.eye(4).tolist()}
  def publish_case(name,array=None,truncated=False,expected=geometry):
   folder=t/name;folder.mkdir();candidate=folder/'candidate.nii.gz';destination=folder/'prediction.nii.gz'
   nib.save(nib.Nifti1Image(np.zeros((4,4,4),dtype=np.uint8),np.eye(4)),destination);old=destination.read_bytes()
   if array is None:candidate.write_bytes(b'not nifti')
   else:
    nib.save(nib.Nifti1Image(array,np.eye(4)),candidate)
    if truncated:
     import gzip
     candidate.write_bytes(gzip.compress(gzip.decompress(candidate.read_bytes())[:352]))
   try:nn.publish_prediction_output(candidate,destination,expected);accepted=True;err=''
   except Exception as e:accepted=False;err=str(e)
   try:np.asarray(nib.load(destination).dataobj);readable=True;read_error=''
   except Exception as e:readable=False;read_error=str(e)
   return {'accepted':accepted,'error':err,'old_preserved':old==destination.read_bytes(),'remaining_files':[p.name for p in folder.iterdir()],'published_voxels_readable':readable,'voxel_error':read_error}
  out['R10_corrupt_header']=publish_case('bad_header')
  out['publication_valid_overwrite']=publish_case('valid_overwrite',np.ones((4,4,4),dtype=np.uint8))
  out['publication_truncated_voxels']=publish_case('truncated',np.ones((4,4,4),dtype=np.uint8),True)
  out['publication_four_dimensional']=publish_case('four_dim',np.ones((4,4,4,2),dtype=np.uint8))
  out['publication_fractional_labels']=publish_case('probabilities',np.full((4,4,4),.25,dtype=np.float32))
  # A new v2 snapshot producer feeds the real queue, whose consumer still
  # assumes dict entries. The native create/save step is the only stub.
  class Mask:
   name='Liver';guid='synthetic-guid';number_of_pixels=1
   def get_voxel_buffer(self):return b'\x00\x01\x00'
  snapshot=create._capture_mask_snapshot(Mask())
  case='snapshot_case';output=t/'receipt-output';output.mkdir();runtime=t/'queue-runtime';queue=runtime/'prepared_queue';queue.mkdir(parents=True);work=t/'work'/'case';work.mkdir(parents=True);final=output/(case+'.mcs')
  nc.write_json_atomic(work/'prepare_manifest.json',{'output_mcs':str(final),'source_fingerprint':'synthetic-fp'})
  nc.write_json_atomic(queue/(case+'.json'),{'case_id':case,'work_dir':str(work),'output_mcs':str(final)})
  nc.write_json_atomic(runtime/create.QUEUE_DONE_FILE,{'status':'done'})
  def native_create_stub(_work,path):
   Path(path).write_bytes(b'synthetic-mcs')
   return {'output_mcs':str(final),'source_fingerprint':'synthetic-fp','receipt':{'case_id':case,'mask_names':['Liver'],'manifest_data':{'source_fingerprint':'synthetic-fp'},'mask_objects':[snapshot]}}
  logs=[];previous_runtime=create._ACTIVE_RUNTIME_DIR
  try:
   with mock.patch.object(create,'create_mcs_from_manifest',side_effect=native_create_stub),mock.patch.object(create,'log_message',side_effect=lambda _root,message:logs.append(message)):
    code=create.main(str(output),runtime_dir=str(runtime))
  finally:create._ACTIVE_RUNTIME_DIR=previous_runtime
  out['queue_snapshot_handoff']={'snapshot_type':type(snapshot).__name__,'queue_exit':code,'mcs_exists':final.is_file(),'receipt_exists':(output/('.'+case+'.import_receipt.json')).is_file(),'messages':[m for m in logs if 'receipt' in m.lower()],'final_status':nc.read_json(runtime/create.STATUS_FILE,{})}
  # Current v2 without SHA (old receipt or failed import-time buffer read).
  class Masks(list):
   def delete(self,mask):self.remove(mask)
  edited=SimpleNamespace(name='Liver',guid='g-owned',number_of_pixels=1,get_voxel_buffer=lambda:b'\x00\x00\x01')
  for key,entry in [('legacy_v2',{'name':'Liver','guid':'g-owned'}),('failed_snapshot',{'name':'Liver','guid':'g-owned','pixel_count':1,'sha256':''}),('valid_snapshot',{'name':'Liver','guid':'g-owned','pixel_count':1,'sha256':snapshot.sha256})]:
   masks=Masks([edited]);receipt={'created_masks_v2':[entry],'provenance_token':'synthetic'}
   with mock.patch.object(undo.mimics.data,'masks',masks),mock.patch.object(undo.runtime_common,'execute_mimics_transaction',side_effect=lambda _m,fn,_label:fn()):deleted,ambiguous,modified=undo._delete_owned_masks(receipt)
   out['undo_'+key]={'deleted':deleted,'edited':modified,'remaining':len(masks)}
  rows=[{'case_id':'c'+str(i),'patient_group':''} for i in range(4)];roots={'raw':t/'raw','preprocessed':t/'pre'};req={'val_cases':1}
  out['R07_frozen_four_cases']=flex._write_flexict_splits(rows,req,roots,'Dataset001_synthetic',frozen_val=['c0'])
  try:split=flex._write_flexict_splits(rows[:2],req,roots,'Dataset002_synthetic',frozen_val=['c0']);out['frozen_two_case_retrain']={'accepted':True,'split':split}
  except ValueError as e:out['frozen_two_case_retrain']={'accepted':False,'error':str(e)}
  # Real prepared-row serializer roundtrip is covered separately by tests;
  # exercise top-level fallback and disagreement validation here.
  uncertain_path=ROOT/'integrations/flexict-finetune/uncertainty/uncertainty.py';spec=importlib.util.spec_from_file_location('audit_uncertainty',uncertain_path);uncertain=importlib.util.module_from_spec(spec);spec.loader.exec_module(uncertain)
  a=t/'a.nii.gz';b=t/'b.nii.gz';shift=np.eye(4);shift[0,3]=100
  nib.save(nib.Nifti1Image(np.ones((4,4,4),dtype=np.uint8),np.eye(4)),a);nib.save(nib.Nifti1Image(np.ones((4,4,4),dtype=np.uint8),shift),b)
  try:uncertain.load_masks([str(a),str(b)]);out['R15_pair_affine']={'rejected':False}
  except ValueError as e:out['R15_pair_affine']={'rejected':True,'error':str(e)}
  nib.save(nib.Nifti1Image(np.ones((4,4,4),dtype=np.uint8),shift),a)
  arrays,aff,_=uncertain.load_masks([str(a),str(b)]);out['both_predictions_wrong_grid']={'accepted':True,'output_translation_mm':aff[:3,3].tolist(),'source_translation_mm':[0,0,0]}
  # DICOM headers with regular normal projection but bad full 3D geometry.
  dicom=[]
  cases=[('regular',[[0,0,0],[0,0,1],[0,0,2]],None),('nonuniform',[[0,0,0],[0,0,1],[0,0,3]],None),('duplicate',[[0,0,0],[0,0,1],[0,0,1]],None),('zigzag',[[0,0,0],[8,0,1],[0,0,2]],None),('missing_middle_ipp',[[0,0,0],None,[0,0,2]],None),('varying_pixel_spacing',[[0,0,0],[0,0,1],[0,0,2]],[[1,1],[2,2],[1,1]])]
  for name,positions,spacings in cases:
   folder=t/name;folder.mkdir();headers={}
   for i,pos in enumerate(positions):
    path=folder/(str(i)+'.dcm');path.write_bytes(b'header-only-fixture');ds=pydicom.Dataset();ds.Rows=ds.Columns=4;ds.SeriesInstanceUID='1.2.3';ds.ImageOrientationPatient=[1,0,0,0,1,0];ds.PixelSpacing=spacings[i] if spacings else [1,1];ds.SliceThickness=1
    if pos is not None:ds.ImagePositionPatient=pos
    headers[str(path)]=ds
   try:
    with mock.patch.object(pydicom,'dcmread',side_effect=lambda path,**kw:headers[str(path)]):shape=bridge.get_image_shape_from_dicom(str(folder));aff=bridge.get_image_affine_from_dicom(str(folder))
    dicom.append({'case':name,'accepted':True,'source_positions':positions,'source_pixel_spacings':spacings,'inferred_lps_positions':[(np.diag([-1, -1, 1, 1])@aff@np.array([0,0,z,1]))[:3].tolist() for z in range(shape[2])]})
   except Exception as e:dicom.append({'case':name,'accepted':False,'error':str(e)})
  out['dicom_geometry']=dicom
  # Label semantic identity and filename lookup use different equivalence.
  labels=nc.normalize_labels([{'name':'Liver-Left','id':1},{'name':'Liver_Left','id':2}]);folder=t/'labels'/'case'/'segmentations';folder.mkdir(parents=True);(folder/'Liver_Left.nii.gz').write_bytes(b'filename-only-fixture')
  matched={row['name']:[p.name for p in nn._find_label_files(t/'labels','case',row['aliases'])] for row in labels}
  out['label_key_disagreement']={'accepted_labels':labels,'matched_files':matched}
  image=t/'label-source.nii.gz';nib.save(nib.Nifti1Image(np.zeros((4,4,4),dtype=np.float32),np.eye(4)),image)
  mask_file=folder/'Liver_Left.nii.gz';nib.save(nib.Nifti1Image(np.ones((4,4,4),dtype=np.uint8),np.eye(4)),mask_file)
  merged=nn._materialize_case('case',image,[[mask_file],[mask_file]],{'labels':labels,'missing_label_policy':'require_all','overlap_policy':'replace'},t/'merge-cache')
  out['label_key_disagreement']['materialized_values']=np.unique(np.asarray(nib.load(merged['label']).dataobj)).tolist()
  # Actual folder-prediction -> uncertainty chain; only model execution is a stub.
  job=t/'al-source-check';(job/'input').mkdir(parents=True);model=t/'synthetic-model';model.mkdir()
  nib.save(nib.Nifti1Image(np.zeros((4,4,4),dtype=np.float32),np.eye(4)),job/'input'/'caseA.nii.gz')
  nc.write_json_atomic(job/'input_geometries.json',{'schema_version':'flexict_al_geometries.v1','cases':{'caseA':geometry}})
  def predict_stub(_stage,params,*args,**kwargs):
   dst=Path(params['output_path']);dst.mkdir(parents=True,exist_ok=True)
   nib.save(nib.Nifti1Image(np.ones((4,4,4),dtype=np.uint8),shift),dst/'caseA.nii.gz')
   return 0
  prediction_dirs=[]
  with mock.patch.object(flex,'_spawn_worker',side_effect=predict_stub):
   for configuration in ['2d','3d_fullres']:
    prediction_dirs.append(flex._al_predict_configuration(job,{},roots,{'model_dir':str(model)},configuration,job/'status.json',job/'control.json',job/'job.log',None))
  metrics=uncertain.run_batch([str(p) for p in prediction_dirs],str(job/'uncertainty'),case_names=['caseA'],save_plots=False,save_components=False,log_path=str(job/'uncertainty.log'),n_workers=1)
  out['al_pair_wrong_source_chain']={'source_translation_mm':[0,0,0],'prediction_translation_mm':[100,0,0],'successful_cases':[m['case'] for m in metrics],'source_geometry_file_exists':(job/'input_geometries.json').exists(),'only_model_execution_stubbed':True}
  # Recovery with an existing uncertainty directory must not rank old output
  # from a case the current invocation just rejected.
  d1=t/'stale-a';d2=t/'stale-b';d1.mkdir();d2.mkdir();stale_out=t/'stale-uncertainty'
  for d in [d1,d2]:nib.save(nib.Nifti1Image(np.ones((4,4,4),dtype=np.uint8),np.eye(4)),d/'stale_case.nii.gz')
  plot_metrics=uncertain.run_batch([str(d1),str(d2)],str(t/'plot-diagnostic'),case_names=['stale_case'],save_plots=True,save_components=False,n_workers=1,log_path=str(t/'plot-diagnostic.log'))
  out['plot_dependency_diagnostic']={'successful_cases':[m['case'] for m in plot_metrics],'log':(t/'plot-diagnostic.log').read_text()}
  first=uncertain.run_batch([str(d1),str(d2)],str(stale_out),case_names=['stale_case'],save_plots=False,save_components=False,n_workers=1)
  nib.save(nib.Nifti1Image(np.ones((4,4,4),dtype=np.uint8),shift),d2/'stale_case.nii.gz')
  second=uncertain.run_batch([str(d1),str(d2)],str(stale_out),case_names=['stale_case'],save_plots=False,save_components=False,n_workers=1,log_path=str(t/'stale-retry.log'))
  sys.path.insert(0,str(ROOT/'integrations/flexict-finetune'))
  from uncertainty import analyze_uncertainty_dir
  ranked=analyze_uncertainty_dir(str(stale_out),method='entropy',sort_key='integrated',report_csv=str(stale_out/'ranking.csv'),compute_boundary=False,compute_cc=False)
  out['al_failed_retry_uses_old_output']={'first_successes':[m['case'] for m in first],'retry_successes':[m['case'] for m in second],'ranked_after_failed_retry':[m['case'] for m in ranked],'retry_log':(t/'stale-retry.log').read_text()}


 return out
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();result=run();args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
