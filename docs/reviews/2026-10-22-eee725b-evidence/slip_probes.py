"""Source-pinned synthetic contract probes. Mimics/model boundaries are fake;
runtime result application, raw files, locks, bridge command transitions and
repair directory operations are actual production code in temporary folders.
No model/checkpoint/training/environment repair is run on the user's files.
"""
import ast,contextlib,importlib.util,io,json,os,subprocess,sys,tempfile,types
from pathlib import Path
from unittest import mock
import numpy as np
R=Path(__file__).resolve().parents[3];E=Path(__file__).resolve().parent
SOURCE='eee725b45ec42b7c045def95e4682089a3ea13a2'
assert not subprocess.check_output(['git','diff','--name-only',SOURCE,'--','tools','tests','runtime_py35','slip_bridge.py','repair','integrations','mimics_bridge.py'],cwd=R,text=True).strip()
sys.path[:0]=[str(R/'tests'),str(R/'tools'),str(R/'runtime_py35'),str(R)]
import test_slip_mimics as t
from resource_locks import FileResourceLock,default_resource_lock_dir
out={}

@contextlib.contextmanager
def fixture():
    f=t.SlipHandleResultTests();f.setUp()
    try:yield f
    finally:f.doCleanups()

with fixture() as f:
    base=t._u8_pattern_buffer(t.SHAPE,lambda x,y,z:(x+y+z)%2==0)
    (Path(f.tmp)/'base.u8').write_bytes(base.tobytes())
    f.target.set_voxel_buffer(base)
    state=f.make_state(f.job_dir,f.image,f.target,base_path=str(Path(f.tmp)/'base.u8'))
    manual=t._u8_pattern_buffer(t.SHAPE,lambda x,y,z:x==0)
    f.target.set_voxel_buffer(manual);before=f.target.get_voxel_buffer().tobytes()
    f.write_result(f.job_dir,1,{'status':'reset','action':'reset','action_history_len':0})
    result=f.runtime._handle_result(f.image,f.target,state)
    out['reset_after_manual_edit']={'outcome':result,'manual_preserved':before==f.target.get_voxel_buffer().tobytes(),
        'before_voxels':before.count(1),'after_voxels':f.target.number_of_pixels,'questions':len(f.fake.dialogs.questions)}

with fixture() as f:
    state=f.make_state(f.job_dir,f.image,f.target,pid=os.getpid(),status='ready',pending_sequence=None)
    f.runtime._save_job(state);f.nnm._metadata_set(f.target,f.runtime.ASYNC_JOB_METADATA,f.job_dir)
    f.target.selected=True
    lock=FileResourceLock(default_resource_lock_dir(R)/'gpu.lock','gpu','SLIP worker (pid {0})'.format(os.getpid()))
    lock.acquire(wait_seconds=0)
    try:
        with mock.patch.object(f.runtime,'_run_session',return_value=0) as resume:
            code=f.runtime.run()
        out['resume_own_gpu_lock']={'code':code,'resume_called':resume.called,'own_session_metadata':bool(f.nnm._metadata_get(f.target,f.runtime.ASYNC_JOB_METADATA,'')),
            'question_count':len(f.fake.dialogs.questions),'messages':f.fake.dialogs.messages}
    finally:lock.release()

with fixture() as f:
    base=t._u8_pattern_buffer(t.SHAPE,lambda x,y,z:(x+y+z)%2==0)
    (Path(f.tmp)/'base.u8').write_bytes(base.tobytes())
    predicted=t._u8_pattern_buffer(t.SHAPE,lambda x,y,z:x==1)
    f.target.set_voxel_buffer(predicted)
    state=f.make_state(f.job_dir,f.image,f.target,base_path=str(Path(f.tmp)/'base.u8'),interactions=[{'click':[0,0,0],'label':1}],action_history_len=1)
    empty=t._u8_pattern_buffer(t.SHAPE,lambda x,y,z:False)
    f.write_result(f.job_dir,1,{'status':'refined','action':'undo','voxels':0,'action_history_len':0,'output_path':f.write_output(empty)})
    outcome=f.runtime._handle_result(f.image,f.target,state)
    out['undo_first_prompt_existing_base']={'outcome':outcome,'base_voxels':base.tobytes().count(1),
        'actual_voxels':f.target.number_of_pixels,'restored_base':f.target.get_voxel_buffer().tobytes()==base.tobytes(),
        'questions':len(f.fake.dialogs.questions)}

# The module boundary is a tiny deterministic predictor. A write failure
# happens AFTER the real bridge asks it to mutate interactive history.
import slip_bridge as bridge
class Tensor:
    def squeeze(self):return self
    def cpu(self):return self
    def numpy(self):return np.ones((2,3,4),np.uint8)
class Predictor:
    def __init__(self):self.action_history=[]
    def click_inference(self,points,labels):
        self.action_history.append((points,labels));return Tensor()
with tempfile.TemporaryDirectory(prefix='slip_review_bridge_') as folder:
    session=bridge._SlipSession({'device':'cpu'});session.module=Predictor();session.image_shape=(2,3,4)
    path=str(Path(folder)/'result.u8');real_write=bridge._write_mask_u8
    with mock.patch.object(bridge,'_write_mask_u8',side_effect=OSError('synthetic disk denial')):
        try:session.predict([0,0,0],1,path)
        except OSError:pass
    after_failed=len(session.module.action_history)
    result=session.predict([0,0,0],1,path)
    out['predict_retry_after_output_failure']={'history_after_failed_write':after_failed,
        'history_after_retry':len(session.module.action_history),'result_history':result['action_history_len'],
        'published_bytes':Path(path).stat().st_size,'same_click_submitted_twice':True}

# Exact archive traversal rules; no huge portable archive is built.
import package_portable as pack
out['portable_slip_membership']={'bridge_listed':'slip_bridge.py' in pack.INCLUDE_FILES,
    'model_assets_listed':any(p=='slip_models' or p.startswith('slip_models/') for p in pack.INCLUDE_DIRS),
    'runtime_covered':'runtime_py35' in pack.INCLUDE_DIRS,'entry_covered':'scripting_library' in pack.INCLUDE_DIRS,
    'vendor_covered':'integrations' in pack.INCLUDE_DIRS}

# Build actual tiny archives from a synthetic distribution tree.
import zipfile
with tempfile.TemporaryDirectory(prefix='slip_review_pack_') as folder:
    root=Path(folder)/'dist';root.mkdir()
    for name in ['python_env/models','nninteractive_env/models','runtime_py35','scripting_library/02_AI/SLIP','integrations/slip','slip_models/deps']:
        (root/name).mkdir(parents=True,exist_ok=True)
    for name in ['slip_bridge.py','slip_models/SLIP_ckpt.pth','slip_models/deps/dependency.txt','runtime_py35/slip_mimics.py','scripting_library/02_AI/SLIP/01_Annotate.py','integrations/slip/README.md']:
        (root/name).write_text('synthetic fixture')
    result=[]
    with mock.patch.object(pack,'PROJECT_ROOT',root),mock.patch.object(pack,'_find_python',return_value='synthetic-python'):
        for flag in [False,True]:
            dest=Path(folder)/str(flag);dest.mkdir()
            with contextlib.redirect_stdout(io.StringIO()):code=pack.pack(str(dest),with_env=flag)
            archives=list(dest.glob('*.zip'))
            with zipfile.ZipFile(archives[0]) as archive:names=archive.namelist()
            result.append({'with_env':flag,'exit':code,'members':names,'has_bridge':'slip_bridge.py' in names,'has_checkpoint':'slip_models/SLIP_ckpt.pth' in names})
    out['actual_synthetic_archives']=result

# A metadata-only backup passes has_pending_backup but cannot restore torch.
# ENV_SITE and cuda_probe are redirected; no actual Python installation touched.
spec=importlib.util.spec_from_file_location('review_gpu_switch',R/'repair/cu118/switch_gpu_torch.py')
repair=importlib.util.module_from_spec(spec);spec.loader.exec_module(repair)
with tempfile.TemporaryDirectory(prefix='slip_review_repair_') as folder:
    p=Path(folder);(p/'torch').mkdir();(p/'torch/working.txt').write_text('working binary placeholder')
    (p/'torch-2.6.0+cu118.dist-info').mkdir();(p/'torch-2.6.0+cu124.dist-info.bak').mkdir()
    with mock.patch.object(repair,'ENV_SITE',folder),mock.patch.object(repair,'cuda_probe',return_value={'avail':False}),mock.patch.object(repair,'check_no_env_users',return_value=([('999','synthetic holder')],None)) as holder_check,mock.patch.object(sys,'argv',['switch_gpu_torch.py','restore']):
        before=repair.has_pending_backup()
        with contextlib.redirect_stdout(io.StringIO()):code=repair.main()
    out['restore_partial_backup']={'has_any_backup':before,'exit':code,'torch_survived':(p/'torch/working.txt').exists(),
        'process_holder_check_called':holder_check.called,'remaining':sorted(x.name for x in p.iterdir())}

# Coordinate round-trip uses the ACTUAL vendored methods, extracted by AST
# only to avoid importing unavailable TorchIO/CuPy/model dependencies.
# This is a method test, never complete SLIP inference or Mimics validation.
tree=ast.parse((R/'integrations/slip/SLIP_model/inference_module.py').read_text())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Inference_module')
selected=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ['transform_coordinates_to_processed_space','transform_mask_to_original_space']]
namespace={'np':np}
import torch,torch.nn.functional as F
namespace.update(torch=torch,F=F)
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name='Methods',bases=[],keywords=[],body=selected,decorator_list=[])],type_ignores=[])),'<vendored-methods>','exec'),namespace)
m=namespace['Methods']();m.axis_permutation=np.array([1,0,2]);m.inverse_permutation=np.argsort(m.axis_permutation)
m.original_shape=(4,2,3);m.transformed_shape=(2,4,3);m.scale_factors=np.ones(3)
coords=m.transform_coordinates_to_processed_space([[3,1,2]])
mask=torch.zeros((1,1,2,4,3));mask[0,0,1,3,2]=1
restored=m.transform_mask_to_original_space(mask)
out['vendored_axis_roundtrip']={'mapped_click':np.asarray(coords).tolist(),'shape':list(restored.shape),
    'original_point_true':bool(restored[0,0,3,1,2]),'voxel_count':int(restored.sum())}

out['cpu_path_source_risk']={'auto_resolved_device':bridge._resolve_device('auto'),'vendored_forced_cuda_line':next(i+1 for i,line in enumerate((R/'integrations/slip/SLIP_model/inference_module.py').read_text().splitlines()) if '.to("cuda")' in line),'cupy_import_required':True,'real_model_run':False}

(E/'observations.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,default=str)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2,default=str))
