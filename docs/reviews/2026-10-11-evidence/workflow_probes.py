"""Record current unresolved behavior, not tests that approve that behavior.

Synthetic temporary files, real production functions, documented host stubs.
Run with --repo and --out. No patient data, Mimics, GPU or SSH is used.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from unittest import mock

p = argparse.ArgumentParser()
p.add_argument('--repo', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
root = a.repo.resolve()
sys.path[:0] = [str(root), str(root/'tools'), str(root/'runtime_py35')]
import numpy as np
import nibabel as nib
import pydicom
import mimics_bridge as bridge
import flexict_pipeline as flex
import nnunet_common as common
import remote_training_controller as remote

results = []
with tempfile.TemporaryDirectory(prefix='mimics_workflow_review_') as tmp:
    t = Path(tmp)
    for label, z_positions in [('nonuniform', [0, 1, 3]), ('duplicate', [0, 1, 1])]:
        folder = t/label
        folder.mkdir()
        datasets = {}
        for i, z in enumerate(z_positions):
            path = folder/('slice%d.dcm'%i)
            path.write_bytes(b'synthetic header placeholder')
            ds = pydicom.Dataset()
            ds.Rows = ds.Columns = 4
            ds.SeriesInstanceUID = '1.2.3.4'
            ds.ImagePositionPatient = [0, 0, z]
            ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            ds.PixelSpacing = [1, 1]
            ds.SliceThickness = 1
            datasets[str(path)] = ds
        with mock.patch.object(pydicom, 'dcmread', side_effect=lambda path,**kw:datasets[str(path)]):
            shape = bridge.get_image_shape_from_dicom(str(folder))
            affine = bridge.get_image_affine_from_dicom(str(folder))
        results.append({'probe':'DICOM_'+label, 'source_z':z_positions,
                        'reported_shape':list(shape), 'inferred_z':[float((affine @ [0,0,i,1])[2]) for i in range(shape[2])],
                        'rejected':False})

    # Mimic the actual FlexiCT training order, including shared raw preparation.
    rows = []
    for i in range(4):
        image = t/('c%d-image.nii.gz'%i)
        mask = t/('c%d-label.nii.gz'%i)
        nib.save(nib.Nifti1Image(np.zeros((2,2,2)), np.eye(4)), image)
        nib.save(nib.Nifti1Image(np.ones((2,2,2),dtype=np.uint8), np.eye(4)), mask)
        rows.append({'case_id':'c%d'%i, 'image':image, 'label':mask, 'fingerprint':'synthetic-%d'%i})
    req = common.normalize_request({'operation':'train', 'workspace':str(t/'ws'),
        'task_name':'review_task', 'dataset_id':41, 'labels':[{'id':1, 'name':'Liver'}],
        'cases':[r['case_id'] for r in rows], 'val_cases':1, 'validation_fraction':0.25})
    roots = {'raw':t/'raw', 'preprocessed':t/'preprocessed'}
    name = flex._dataset_name(req)
    previous = roots['preprocessed']/name/'splits_final.json'
    common.write_json_atomic(previous, [{'train':['c1','c2','c3'],'val':['c0']}])
    dataset_dir, _ = flex._materialize_flexict_raw(rows, req, roots)
    after_shared = common.read_json(previous, [])
    flex._write_flexict_splits(rows, req, roots, dataset_dir.name)
    after_flex = common.read_json(previous, [])
    results.append({'probe':'FlexiCT_frozen_validation', 'previous_val':['c0'],
        'after_shared_materializer':after_shared[0], 'after_flexict_rewrite':after_flex[0],
        'previous_val_entered_train':'c0' in after_flex[0]['train']})
    grouped = [dict(r, patient_group='patient_A' if i<2 else 'patient_B') for i,r in enumerate(rows)]
    tr, va = flex._split_train_val(grouped, dict(req, val_case_ids=['c0','c2']))
    results.append({'probe':'FlexiCT_all_groups_validation','train':tr,'val':va,'rejected':False})

    result_dir=t/'download';result_dir.mkdir()
    (result_dir/'prediction.nii.gz').write_bytes(b'new AI result')
    final=t/'prediction.nii.gz';final.write_bytes(b'previous human-reviewed result')
    request_path=t/'remote-request.json'
    common.write_json_atomic(request_path, {'output_path':str(final)})
    status={}
    remote._apply_remote_prediction_output({'request_path':str(request_path)},'nnunet_infer',result_dir,status)
    results.append({'probe':'remote_prediction_existing_output','previous_result_preserved':final.read_bytes()==b'previous human-reviewed result',
                    'final_contents':final.read_text(),'overwrite_confirmation':False})

    # Receipt discovery has no access to per-window output choices.
    spec=importlib.util.spec_from_file_location('audit_support',root/'tests/test_all.py')
    support=importlib.util.module_from_spec(spec);spec.loader.exec_module(support)
    import import_undo_mimics as undo
    arbitrary=t/'case-source'/'mask_exports';arbitrary.mkdir(parents=True)
    (arbitrary/'case.import_receipt.json').write_text(json.dumps({'schema_version':undo.RECEIPT_SCHEMA_VERSION,'created_at_epoch':10}))
    config=t/'mimics_io_config.json';config.write_text('{}')
    with mock.patch.object(undo,'_project_root',return_value=str(t)):
        path,payload=undo.find_latest_receipt()
    results.append({'probe':'undo_default_source_output_discovery','receipt_exists':True,'receipt_found':path is not None})

a.out.parent.mkdir(parents=True,exist_ok=True)
a.out.write_text(json.dumps({'schema':'mimics_workflow_probes.v1','synthetic_only':True,'results':results},indent=2),encoding='utf-8')
print(json.dumps(results,indent=2))
