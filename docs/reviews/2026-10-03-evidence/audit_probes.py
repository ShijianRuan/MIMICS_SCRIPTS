"""Read-only product audit probes. All mutations use synthetic temporary data.

Run: python audit_probes.py --repo /path/to/repo --out /path/to/evidence
These record current behavior; they are not acceptance tests asserting bugs are OK.
Mimics-specific calls are boundary stubs, never a real Mimics validation.
"""
import argparse
import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import types
from pathlib import Path
from unittest import mock

p = argparse.ArgumentParser()
p.add_argument('--repo', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
args = p.parse_args()
ROOT = args.repo.resolve()
args.out.mkdir(parents=True, exist_ok=True)
sys.path[:0] = [str(ROOT), str(ROOT/'tools'), str(ROOT/'runtime_py35')]
spec = importlib.util.spec_from_file_location('audit_test_support', ROOT/'tests/test_all.py')
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)
import runtime_common as common
import create_mcs_batch as batch
import import_undo_mimics as undo
import mimics_export as export
import flexict_mimics as flex

results = []
def record(name, result):
    results.append(dict(probe=name, **result))

with tempfile.TemporaryDirectory(prefix='mimics_audit_') as tmp:
    t = Path(tmp)
    target=t/'target.mcs'; stage=t/'stage.mcs'
    identity=common.capture_output_identity(str(target))
    target.write_bytes(b'newer human work'); stage.write_bytes(b'old worker result')
    msg=common.publish_conflict(str(stage),str(target),identity)
    record('F28_absent_then_created',dict(initially_absent=True, warning=msg, final_bytes=target.read_text()))

    target.write_bytes(b'initial'); identity=common.capture_output_identity(str(target))
    stage.write_bytes(b'old worker result'); actual_replace=os.replace
    def intervening_save(src,dst):
        Path(dst).write_bytes(b'newer human work after check')
        actual_replace(src,dst)
    with mock.patch.object(common.os,'replace',intervening_save):
        msg=common.publish_conflict(str(stage),str(target),identity)
    record('F28_save_between_check_and_replace',dict(warning=msg, final_bytes=target.read_text()))

    final=t/'case.mcs'; staging=Path(batch._staging_mcs_path(str(final)))
    staging.write_bytes(b'synthetic project')
    receipt_path=batch.write_import_receipt(str(t),'case',str(staging),[],{},mask_objects=[])
    batch._publish_mcs(str(staging),str(final))
    receipt=json.loads(Path(receipt_path).read_text())
    with mock.patch.object(undo,'find_latest_receipt',return_value=(receipt_path,receipt)):
        code=undo.undo_last_import(confirm=False)
    record('R30_staging_receipt',dict(final_exists=final.exists(),receipt_target_exists=Path(receipt['mcs_path']).exists(),undo_code=code))

    alive=[]
    class Mask:
        def __init__(self,name,guid,image=None,data=b''):
            self.name=name; self.guid=guid; self.image=image; self.data=data
        def delete(self): alive.remove(self)
    image=types.SimpleNamespace(masks=alive)
    m=Mask('liver','guidA'); alive.append(m)
    rec={'created_masks_v2':[{'name':'liver','guid':'guidA'}],'provenance_token':'token'}
    with mock.patch.object(undo.mimics.data,'images',[image]):
        before=undo._session_diverged(rec)
        with mock.patch.object(common,'execute_mimics_transaction',side_effect=lambda api,op,title:op()):
            deleted,ambiguous=undo._delete_owned_masks(rec)
        after=undo._session_diverged(rec)
    record('R31_undo_checks_after_delete',dict(before=before,after=after,deleted=deleted,remaining=len(alive)))

    def fail_delete(): raise RuntimeError('synthetic host deletion failure')
    m=types.SimpleNamespace(name='liver',guid='guidA',delete=fail_delete)
    alive[:]=[m]
    with mock.patch.object(undo.mimics.data,'images',[image]),mock.patch.object(common,'execute_mimics_transaction',side_effect=lambda api,op,title:op()):
        result=undo._delete_owned_masks(rec)
    record('R31_deletion_failure_swallowed',dict(returned=result,remaining=len(alive),exception_propagated=False))
    failed_project=t/'delete_failure.mcs';failed_project.write_bytes(b'project')
    failed_receipt=t/'delete_failure.import_receipt.json'
    full_rec=dict(rec,schema_version='mimics_import_receipt.v2',created_masks=['liver'],mcs_path=str(failed_project),mcs_fingerprint='deliberately changed')
    failed_receipt.write_text(json.dumps(full_rec))
    with mock.patch.object(undo,'find_latest_receipt',return_value=(str(failed_receipt),full_rec)),mock.patch.object(undo,'_open_project',return_value=True),mock.patch.object(undo.mimics.data,'images',[image]),mock.patch.object(common,'execute_mimics_transaction',side_effect=lambda api,op,title:op()):
        code=undo.undo_last_import(confirm=False)
    record('R31_failed_delete_consumes_receipt',dict(undo_code=code,receipt_exists=failed_receipt.exists(),mask_remaining=len(alive)))

    # v1 is deliberately still supported; verify migration cannot name-delete
    # a manual mask attached to a second image.
    deleted_names=[]
    class LegacyMask:
        def __init__(self,guid): self.name='liver';self.guid=guid
        def delete(self):deleted_names.append(self.guid)
    with mock.patch.object(undo.mimics.data,'images',[types.SimpleNamespace(masks=[LegacyMask('imported')]),types.SimpleNamespace(masks=[LegacyMask('manual-other-image')])]),mock.patch.object(common,'execute_mimics_transaction',side_effect=lambda api,op,title:op()):
        undo._delete_masks({'liver'})
    record('F19_v1_still_deletes_same_name',dict(deleted_objects=deleted_names))

    # Documented ImageData surface has linked_objects, not image.masks.
    with mock.patch.object(undo.mimics.data,'images',[types.SimpleNamespace(linked_objects=[])]):
        try: undo._resolve_owned_masks(rec);api_error=''
        except Exception as exc:api_error=type(exc).__name__+': '+str(exc)
    record('F19_documented_image_contract',dict(error=api_error,boundary='API-shaped stub, not actual Mimics'))

    imageA=types.SimpleNamespace(guid='imageA',logical_dimensions=[2,2,2])
    imageB=types.SimpleNamespace(guid='imageB',logical_dimensions=[2,2,2])
    masks=[Mask('liver manual','m1',imageA,b'\1'*8),Mask('liver_manual','m2',imageA,b'\0'*8)]
    matrix=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    def write_mask(mask,path): Path(path).write_bytes(mask.data); return len(mask.data)
    with mock.patch.object(export.mimics.data,'masks',masks),mock.patch.object(export,'_derive_mimics_voxel_to_ras_matrix',return_value=matrix),mock.patch.object(export,'_stream_mask_to_file',side_effect=write_mask),contextlib.redirect_stdout(io.StringIO()):
        manifest=export.export_masks_to_buffers(str(t/'buffers'))
    record('R32_export_name_collision',dict(names=[m.name for m in masks],entries=manifest['masks'],actual_buffer_files=[x.name for x in (t/'buffers').glob('*.u8')],first_mask_content_preserved=(t/'buffers'/'liver_manual.u8').read_bytes()==masks[0].data))

    masks=[Mask('liver arterial','mA',imageA,b'\1'*8),Mask('liver venous','mB',imageB,b'\0'*8)]
    with mock.patch.object(export.mimics.data,'masks',masks),mock.patch.object(export,'_derive_mimics_voxel_to_ras_matrix',return_value=matrix):
        selected=export._selected_masks(); manifest=export._new_export_manifest(selected)
    record('R33_export_multiple_images_one_grid',dict(selected_images=[m.image.guid for m in selected],manifest_grid_count=1,manifest_shape=manifest['mimics_shape'],rejected=False))

    # Case A overlay requested while same-grid case B is already active.
    # No mid-conversion project switch is needed. Stub only host/process boundaries.
    job=t/'al'; job.mkdir(); inp=job/'input_geometries.json'
    geometry={'source_shape':[2,2,2],'source_voxel_to_ras_matrix':matrix,'source_image_path':str(t/'patientA.nii.gz'),'mcs_path':str(t/'patientA.mcs')}
    inp.write_text(json.dumps({'cases':{'patientA':geometry}}))
    request={'_request_path':str(job/'request.json'),'_job_dir':str(job),'case_id':'patientA','what':'bands'}
    (job/'request.json').write_text(json.dumps(request))
    proc=types.SimpleNamespace(returncode=0,communicate=lambda **kw:(b'{"status":"error","error":"synthetic process"}',b''))
    class NoThread:
        def __init__(self,**kw): pass
        def start(self): pass
    apply=flex.mimics_mask_apply
    with contextlib.ExitStack() as stack:
        for obj,name,value in [(flex,'_al_mask_paths',lambda *a:[('band',str(t/'band.nii.gz'))]),(apply,'_active_source_geometry_payload',lambda:dict(geometry,source_image_path=str(t/'patientB.nii.gz'))),(apply,'_active_live_grid_payload',lambda:{'target_shape':[2,2,2],'target_voxel_to_ras_matrix':matrix}),(apply,'_current_project_path',lambda:str(t/'patientB.mcs')),(apply,'_config',lambda:{}),(apply,'_buffer_mapping_from_config',lambda c:([0,1,2],[False]*3)),(flex,'_external_python',lambda:sys.executable),(flex.subprocess,'Popen',lambda *a,**k:proc),(flex.threading,'Thread',NoThread)]:
            stack.enter_context(mock.patch.object(obj,name,value))
        stack.enter_context(mock.patch.object(flex.mimics.data.images,'get_active',lambda:imageB))
        flex._al_apply_request(request)
        transaction=flex._AL_APPLY_TRANSACTIONS.get('conversion')
        finish_guard=flex._al_target_still_matches(transaction) if transaction else 'not started'
    record('F20_wrong_patient_before_launch',dict(requested_case='patientA',captured_project=Path(transaction['target_identity']['project_path']).name if transaction else None,request_state=json.loads((job/'request.json').read_text()).get('state'),finish_guard_error=finish_guard))
    flex._AL_APPLY_TRANSACTIONS.clear()

    # Real Qt signals, no file I/O outside temporary project.
    os.environ['QT_QPA_PLATFORM']='offscreen'
    from PySide6 import QtCore,QtGui,QtWidgets
    import batch_status_viewer as viewer
    from ui_theme import configure_application, stylesheet
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    configure_application(app); app.setStyleSheet(stylesheet())
    with mock.patch.object(viewer,'collect_batch_rows',return_value=[]):
        window=viewer.BatchStatusWindow(t,(QtCore,QtGui,QtWidgets))
        errors=[]
        with mock.patch.object(sys,'excepthook',lambda typ,value,tb:errors.append(str(value))):
            next(b for b in window.window.findChildren(QtWidgets.QPushButton) if b.text()=='刷新').click()
            app.processEvents()
        record('R34_actual_refresh_click',dict(signal_errors=errors))
        window.window.close()

    # Submission identifiers currently have second resolution, shared by windows.
    import import_drop_window as drop
    commands=[]
    with mock.patch.object(drop,'_ROOT',str(t/'drop_project')),mock.patch.object(drop,'external_python',return_value=sys.executable),mock.patch.object(drop.io_ui,'discover_single_source',side_effect=lambda source:{'case_id':Path(source).name,'masks':[]}),mock.patch.object(drop.time,'strftime',return_value='20261003T120000'),mock.patch.object(drop.subprocess,'Popen',side_effect=lambda command,**kw:commands.append(command)):
        for case in ['patientA','patientB']:
            drop.submit_import({'kind':'single','source_path':str(t/case),'output_path':str(t/'out')},{})
    selections=[c[c.index('--selection-json')+1] for c in commands]
    record('R35_same_second_import_ids',dict(worker_count=len(commands),selection_paths_distinct=len(set(selections)),last_source=Path(json.loads(Path(selections[0]).read_text())['source_path']).name))

    # Source bytes are tiny placeholders: raw dataset publication copies files,
    # it does not decode NIfTI. Exercise real mapping / split / write / reload.
    import nnunet_pipeline as pipeline
    raw=t/'raw'; prep=t/'prep'; rows=[]
    for index in range(6):
        img=t/('image%d.nii.gz'%index); lab=t/('label%d.nii.gz'%index)
        img.write_bytes(b'image');lab.write_bytes(b'label')
        rows.append({'case_id':'病例%d'%index,'image':img,'label':lab,'fingerprint':str(index)})
    import nnunet_common
    req=nnunet_common.normalize_request({'task_name':'audit','task_id':'audit','dataset_id':701,'modality':'CT','labels':[{'name':'liver','id':1}], 'split_seed':2026,'validation_fraction':0.2,'workspace':str(t/'ws')})
    dataset,_=pipeline._materialize_nnunet_raw(rows,req,raw,prep)
    split_path=prep/dataset.name/'splits_final.json'
    first=json.loads(split_path.read_text())[0]
    mapping={r['nnunet_case_id']:r['case_id'] for r in json.loads((dataset/'mimics_dataset_manifest.json').read_text())['cases']}
    old_val_names=[mapping[key] for key in first['val']]
    observed=[]; original_split=pipeline._split_folds
    def spy_split(r,q,frozen_validation=None):
        observed.append(frozen_validation)
        return original_split(r,q,frozen_validation)
    for index in range(6,8):
        img=t/('image%d.nii.gz'%index);lab=t/('label%d.nii.gz'%index)
        img.write_bytes(b'image');lab.write_bytes(b'label')
        rows.append({'case_id':'病例%d'%index,'image':img,'label':lab,'fingerprint':str(index)})
    with mock.patch.object(pipeline,'_split_folds',side_effect=spy_split):
        pipeline._materialize_nnunet_raw(rows,req,raw,prep)
    second=json.loads(split_path.read_text())[0]
    record('F16_non_ascii_frozen_split',dict(previous_case_count=6,new_case_count=8,split_seed_unchanged=2026,previous_validation_names=old_val_names,frozen_argument=observed[0],previous_val_in_new_train=sorted(set(first['val']) & set(second['train']))))

    callbacks=[]; original_stream=common.stream_buffer
    def spy_stream(*a,**kw): callbacks.append(kw.get('progress_callback') is not None);return original_stream(*a,**kw)
    mask=types.SimpleNamespace(get_voxel_buffer=lambda:memoryview(bytearray(33*1024*1024)))
    with mock.patch.object(common,'stream_buffer',side_effect=spy_stream):
        written=export._stream_mask_to_file(mask,str(t/'large.u8'))
    record('F13_export_stream_no_gui_pump',dict(bytes_written=written,progress_callback_supplied=callbacks[0]))

    import numpy as np
    import nibabel as nib
    import flexict_pipeline as fp
    ds_spec=importlib.util.spec_from_file_location('audit_build_dataset',ROOT/'integrations/flexict-finetune/scripts/build_dataset.py')
    ds=importlib.util.module_from_spec(ds_spec);ds_spec.loader.exec_module(ds)
    si_results={}
    for kind,affine,position in [
        ('RAS',np.eye(4),(1,2,8)),
        ('flip_z',np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,9],[0,0,0,1]]),(1,2,1)),
        ('cycle_xyz',np.array([[0,1,0,0],[0,0,1,0],[1,0,0,0],[0,0,0,1]]),(8,1,2))]:
        arr=np.zeros((10,10,10),dtype=np.uint8);arr[position]=1
        img=nib.Nifti1Image(arr,affine);f=t/(kind+'.nii.gz');nib.save(img,f)
        si_results[kind]={'physical_point':(affine@np.array(list(position)+[1]))[:3].tolist(),'pipeline':fp.physical_si_start(img,arr),'standalone':ds._zstart(f,f)}
    record('F29_same_physical_target_different_orientation',dict(results=si_results))
    split_rows=[{'case_id':'patientA_phase1','patient_group':'patientA'},{'case_id':'patientA_phase2','patient_group':'patientA'}]
    train,val=fp._split_train_val(split_rows,{'val_cases':1,'patient_groups':{'patientA_phase1':'patientA','patientA_phase2':'patientA'}})
    record('F16_flexict_final_split_ignores_patient',dict(train=train,val=val,same_patient_on_both_sides=bool(train and val)))

    # Header probes only: multiframe fixture is not a full Enhanced CT.
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian,generate_uid,CTImageStorage
    import mimics_bridge
    def header(path,series,z,frames=None):
        meta=FileMetaDataset();meta.TransferSyntaxUID=ExplicitVRLittleEndian;meta.MediaStorageSOPClassUID=CTImageStorage;meta.MediaStorageSOPInstanceUID=generate_uid()
        ds=FileDataset(str(path),{},file_meta=meta,preamble=b'\0'*128)
        ds.SOPClassUID=meta.MediaStorageSOPClassUID;ds.SOPInstanceUID=meta.MediaStorageSOPInstanceUID;ds.SeriesInstanceUID=series
        ds.Rows=ds.Columns=4;ds.PixelSpacing=[1,1];ds.ImageOrientationPatient=[1,0,0,0,1,0];ds.ImagePositionPatient=[0,0,z];ds.SliceThickness=10;ds.Modality='CT'
        if frames:ds.NumberOfFrames=frames
        ds.save_as(str(path),enforce_file_format=True)
    for kind in ['mixed_series','multi_frame']:
        folder=t/kind;folder.mkdir()
        if kind=='mixed_series':
            for series in range(2):
                uid=generate_uid()
                for z in range(3):header(folder/('%s_%s.dcm'%(series,z)),uid,z*10)
        else:header(folder/'frames.dcm',generate_uid(),0,60)
        with contextlib.redirect_stdout(io.StringIO()):
            r=mimics_bridge.do_prepare({'image_path':str(folder),'buffers_out':str(t/(kind+'_out'))})
        record('F01_'+kind,dict(status=r.get('status'),source_image_shape=r.get('source_image_shape')))

(args.out/'scenario-results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
print(json.dumps(results,ensure_ascii=False,indent=2))
