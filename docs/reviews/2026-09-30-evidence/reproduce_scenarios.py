"""Second-round review evidence; never operates on real Mimics/patient data.

Run with the review Python (numpy/nibabel/pydicom/SimpleITK installed).
All destructive filesystem paths are inside TemporaryDirectory. Real production
functions run unchanged; Mimics, process launch and GPU boundaries use stubs.
These probes characterize current defects, not repaired behavior or Windows E2E.
"""
import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
os.environ["MIMICS_PATH_SIGNATURE_CACHE_DIR"] = "off"
for path in (ROOT, ROOT / "tools", ROOT / "runtime_py35"):
    sys.path.insert(0, str(path))

import nibabel as nib
import numpy as np


def extract(relative, name, namespace):
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / relative), "exec"), namespace)
    return namespace[name]


def nifti(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(data), np.eye(4)), str(path))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


results = {}
with tempfile.TemporaryDirectory(prefix="mimics-synthetic-review-") as temp:
    tmp = Path(temp)

    # F18: a matching disk fingerprint does not represent dirty in-memory edits.
    project = tmp / "undo" / "case.mcs"
    project.parent.mkdir()
    project.write_bytes(b"imported baseline")
    receipt_path = project.with_suffix(".json")
    receipt_path.write_text("{}")
    receipt = {"mcs_path": str(project), "mcs_fingerprint": sha(project), "created_masks": ["liver"]}
    saves = []
    def save_project(**kwargs):
        saves.append("saved manual edits plus undo result")
        Path(kwargs["filename"]).write_bytes(b"contains new manually drawn kidney")
    fake_mimics = types.SimpleNamespace(
        file=types.SimpleNamespace(save_project=save_project, close_project=lambda: None,
                                   is_project_modified=lambda: True),
        logging=types.SimpleNamespace(log_user_message=lambda **kwargs: None),
        dialogs=types.SimpleNamespace(message_box=lambda *a, **kw: True),
    )
    env = {"os": os, "logging": logging, "mimics": fake_mimics,
           "find_latest_receipt": lambda: (str(receipt_path), receipt),
           "_open_project": lambda p: True, "_delete_masks": lambda names: list(names),
           "_mcs_fingerprint": lambda p: sha(Path(p))}
    code = extract("runtime_py35/import_undo_mimics.py", "undo_last_import", env)(confirm=False)
    results["F18_dirty_project_undo"] = {"return_code": code, "dirty_at_start": True,
        "save_calls": len(saves), "project_file_exists_after": project.exists()}

    # F19: name ownership does not distinguish another image's independent mask.
    deleted = []
    def fake_mask(identity):
        return types.SimpleNamespace(name="liver", guid=identity, delete=lambda: deleted.append(identity))
    env = {"mimics": types.SimpleNamespace(data=types.SimpleNamespace(images=[
        types.SimpleNamespace(masks=[fake_mask("imported-A")]),
        types.SimpleNamespace(masks=[fake_mask("manual-B")])])),
        "runtime_common": types.SimpleNamespace(execute_mimics_transaction=lambda m, op, name: op())}
    extract("runtime_py35/import_undo_mimics.py", "_delete_masks", env)({"liver"})
    results["F19_name_only_undo"] = {"deleted_mask_ids": deleted}

    # F20: same grid in another patient is accepted before conversion. No process starts.
    al_dir = tmp / "al"
    al_dir.mkdir()
    launches = []
    geom_a = {"source_shape": [2, 2, 2], "source_voxel_to_ras_matrix": np.eye(4).tolist(),
              "source_image_path": "synthetic-case-A/image.nii.gz"}
    geom_b = dict(geom_a, source_image_path="synthetic-case-B/image.nii.gz")
    class NoThread:
        def __init__(self, **kwargs): pass
        def start(self): pass
    env = {"os": os, "uuid": __import__("uuid"), "threading": types.SimpleNamespace(Thread=NoThread),
        "subprocess": types.SimpleNamespace(PIPE=-1, Popen=lambda *a, **kw: launches.append("conversion_requested") or object()),
        "runtime_common": types.SimpleNamespace(background_process_kwargs=lambda: {}),
        "_AL_APPLY_TRANSACTIONS": {}, "_external_python": lambda: "never-executed",
        "_project_root": lambda: str(ROOT), "_al_mask_paths": lambda *a: [("band", "fake-mask")],
        "_al_case_source_geometry": lambda *a: geom_a, "_al_mark_request": lambda *a: None,
        "_matrix_close_payload": lambda a,b: np.allclose(a,b),
        "_write_json": lambda p,v: Path(p).write_text(json.dumps(v)),
        "mimics_mask_apply": types.SimpleNamespace(
            _active_source_geometry_payload=lambda: geom_b,
            _active_live_grid_payload=lambda: {"target_shape": [2,2,2], "target_voxel_to_ras_matrix": np.eye(4).tolist()},
            _buffer_mapping_from_config=lambda c: ([0,1,2],[False]*3), _config=lambda: {}),
    }
    extract("runtime_py35/flexict_mimics.py", "_al_apply_request", env)(
        {"_request_path": str(al_dir / "request.json"), "_job_dir": str(al_dir), "case_id": "A", "what": "bands"})
    results["F20_al_other_patient_same_grid"] = {"different_source_paths": True,
        "conversion_requests": len(launches),
        "transaction_keys": sorted(env["_AL_APPLY_TRANSACTIONS"]["conversion"])}

    # F21: actual different binary content with equal cardinality.
    before = np.array([1,0], dtype=np.uint8)
    after = np.array([0,1], dtype=np.uint8)
    target = types.SimpleNamespace(name="liver", guid="same-guid", number_of_pixels=int(after.sum()))
    monitor = {"label_name": "liver", "matching_masks": [{"guid":"same-guid", "pixel_count":int(before.sum())}]}
    update_results = {}
    for module in ("flexict_mimics", "nnunet_mimics"):
        env = {"_active_image_masks":lambda:[target], "mimics_mask_apply":types.SimpleNamespace(_mask_identity=lambda m:m.guid),
               "logging":logging, "_log":lambda *a:None}
        fn = extract("runtime_py35/"+module+".py", "_matching_update_mask", env)
        found = fn(monitor) if module.startswith("flexict") else fn(monitor,{"name":"liver"})
        update_results[module] = found is target
    results["F21_equal_count_changed_mask"] = {"content_changed":not np.array_equal(before,after),
        "before_count":int(before.sum()),"after_count":int(after.sum()),"accepted_for_overwrite":update_results}

    # F22: execute real data preparation with only synthetic NIfTI files.
    import nnunet_pipeline as pipeline
    source = tmp / "collision" / "source"
    originals = []
    for cid, value in (("病例甲",11),("病例乙",22)):
        image = source / cid / "ct.nii.gz"
        nifti(image, np.full((2,2,2), value, np.int16))
        nifti(source/cid/"segmentations"/"liver.nii.gz", np.ones((2,2,2), np.uint8))
        originals.append(image)
    hashes_before = [sha(p) for p in originals]
    request = {"dataset_root":str(source),"workspace":str(tmp/"collision"/"workspace"),
        "task_name":"audit", "labels":[{"name":"liver","id":1}],"label_source":"dataset_masks"}
    out = tmp / "collision" / "output"
    out.mkdir()
    rows = pipeline.prepare_source_grid_cases(request,out,out/"status.json",out/"control.json")
    results["F22_case_slug_collision"] = {
        "case_ids":[r["case_id"] for r in rows],
        "normalized_ids":[pipeline.safe_identifier(r["case_id"]) for r in rows],
        "returned_cases":len(rows), "distinct_output_images":len({str(r["image"]) for r in rows}),
        "source_files_changed":[p.parent.name for p,h in zip(originals,hashes_before) if sha(p)!=h],
        "original_image_values_after":[int(np.asanyarray(nib.load(str(p)).dataobj)[0,0,0]) for p in originals],
    }

    # F23: probability semantics and nonfinite handling differ by entrypoint.
    import mimics_bridge as bridge
    probability = tmp / "probability.nii.gz"
    nifti(probability,np.array([0.01,0.49,0.51,0.99],np.float32).reshape(2,2,1))
    values,_,_ = bridge.read_mask_labels_with_affine(str(probability))
    invalid = tmp / "nonfinite.nii.gz"
    nifti(invalid,np.array([0,np.nan,np.inf,1],np.float32).reshape(2,2,1))
    invalid_binary,_ = bridge.read_nifti_mask_with_affine(str(invalid))
    results["F23_mask_semantics"]={"probability_input":[0.01,0.49,0.51,0.99],
        "imported_values":values.flatten().tolist(),"binary_reader_nonfinite_output":invalid_binary.flatten().tolist()}

    # F24: flat source layout discovers the image itself as an organ mask.
    import sync_missing_masks as sync_launcher
    flat = tmp / "flat" / "case1"
    flat.mkdir(parents=True)
    (flat / "ct.nii.gz").write_bytes(b"fixture")
    (flat / "liver.nii.gz").write_bytes(b"fixture")
    specs = sync_launcher._source_masks(flat.parent, "case1")
    results["F24_sync_source_discovery"] = {"discovered_mask_names":[s["name"] for s in specs]}

    # F25: an unrelated image's mask name makes the target look complete.
    env={"mimics":types.SimpleNamespace(data=types.SimpleNamespace(masks=[types.SimpleNamespace(name="liver",image="B")])),
         "_log":lambda *a:None,"os":os}
    extract("runtime_py35/sync_missing_masks_batch.py","_mask_names",env)
    fn=extract("runtime_py35/sync_missing_masks_batch.py","_sync_opened_case",env)
    result=fn({},str(tmp),"A",[{"name":"liver","mask_path":"synthetic"}],"out","staging","work")
    results["F25_sync_other_image_mask"]={"target_image":"A","existing_mask_image":"B","result":result}

    # F26: previous completed report survives a new process failure.
    import inspect_mcs_projects as inspect_launcher
    mcs_dir=tmp/"inspect"/"mcs";mcs_dir.mkdir(parents=True)
    (mcs_dir/"case.mcs").write_bytes(b"fixture")
    report=tmp/"inspect"/"report.json"
    report.write_text(json.dumps({"status":"completed","completed":1,"failed":0,"total":1}))
    process=types.SimpleNamespace(pid=987654321,poll=lambda:1)
    with mock.patch.object(inspect_launcher,"find_mimics_exe",return_value="never-executed"), \
         mock.patch.object(inspect_launcher,"_acquire_background_mimics_locks",return_value=[]), \
         mock.patch.object(inspect_launcher.subprocess,"Popen",return_value=process), \
         redirect_stdout(io.StringIO()):
        code=inspect_launcher.main(["--mcs-dir",str(mcs_dir),"--report",str(report),"--job-dir",str(tmp/"inspect"/"job")])
    results["F26_stale_inspection_report"]={"new_process_exit_code":1,"launcher_exit_code":code}

    # F27: cancellation between cases drops the already-completed case list.
    cancel_dir=tmp/"cancel";cancel_dir.mkdir()
    stop=cancel_dir/"stop.json"
    states=[]
    def sync_case(case,job_dir):
        stop.write_text("{}")
        return {"case_id":case["case_id"],"status":"completed"}
    config={"job_dir":str(cancel_dir),"stop_path":str(stop),"cases":[{"case_id":"A"},{"case_id":"B"}]}
    env={"os":os,"time":__import__("time"),"traceback":__import__("traceback"),
         "_read_json":lambda *a:config,"_write_json":lambda p,v:Path(p).write_text(json.dumps(v)),
         "_status":lambda *a,**k:states.append({"state":a[1],"completed":a[3]}),
         "_log":lambda *a:None,"_sync_one":sync_case,"_safe_close_project":lambda:None}
    extract("runtime_py35/sync_missing_masks_batch.py","run_job",env)("synthetic-config")
    results["F27_cancelled_sync_receipt"]={"last_status":states[-1],
        "results_file_exists":(cancel_dir/"results.json").exists(),"failed_cases_file_exists":(cancel_dir/"failed_cases.json").exists()}

    # F28: current publisher does not check whether destination advanced.
    destination=tmp/"publish.mcs";destination.write_bytes(b"newer foreground save")
    staging=tmp/"staging.mcs";staging.write_bytes(b"older background snapshot plus masks")
    env={"os":os,"time":__import__("time")}
    extract("runtime_py35/sync_missing_masks_batch.py","_publish",env)(str(staging),str(destination))
    results["F28_concurrent_publish"]={"destination_after":destination.read_text(),
        "newer_foreground_save_preserved":destination.read_bytes()==b"newer foreground save"}

    # F29: isotropic RAS image chooses X as Z-start axis; physical SI is Z.
    label=tmp/"zlabel.nii.gz";ct=tmp/"zimage.nii.gz"
    arr=np.zeros((10,10,10),np.uint8);arr[1,2,8]=1
    nifti(label,arr);nifti(ct,np.zeros(arr.shape,np.int16))
    env={"nib":nib,"np":np}
    val=extract("integrations/flexict-finetune/scripts/build_dataset.py","_zstart",env)(ct,label)
    results["F29_axis_stratification"]={"ras_isotropic_image":True,"computed_z_start":val,"physical_si_voxel_axis_start":0.8}

assert results["F18_dirty_project_undo"]["project_file_exists_after"] is False
assert results["F19_name_only_undo"]["deleted_mask_ids"]==["imported-A","manual-B"]
assert all(results["F21_equal_count_changed_mask"]["accepted_for_overwrite"].values())
assert results["F22_case_slug_collision"]["source_files_changed"]
assert results["F26_stale_inspection_report"]["launcher_exit_code"]==0
print(json.dumps({"schema":"review_evidence.v1","baseline":"36eac04","findings":results},ensure_ascii=False,indent=2))
