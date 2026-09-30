import ast, json, pathlib, tempfile, shutil, types, time, logging
ROOT = pathlib.Path(__file__).resolve().parents[3]

def extract(path, function_name):
    module = ast.parse((ROOT / path).read_text())
    node = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == function_name)
    return compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / path), 'exec')

with tempfile.TemporaryDirectory() as folder:
    bridge = pathlib.Path(folder) / 'bridge'
    buffer_path = bridge / 'buffers' / 'al_0.u8'
    buffer_path.parent.mkdir(parents=True)
    buffer_path.write_bytes(b'\x01')
    result_path = bridge / 'result.json'
    result_path.write_text(json.dumps({'status':'ok', 'masks':[{'name':'al_0','output_path':str(buffer_path),'mimics_shape':[1,1,1]}]}))
    reads = []
    def read_buffer(mask, path, shape, label):
        reads.append((path, pathlib.Path(path).exists()))
        return pathlib.Path(path).read_bytes()
    env = {'_read_json':lambda p,d:json.loads(pathlib.Path(p).read_text()), 'shutil':shutil, 'mimics_mask_apply':types.SimpleNamespace(_new_prediction_mask=lambda title:types.SimpleNamespace(name=title),_set_mask_from_u8=read_buffer), '_al_mark_request':lambda *a:None, '_al_mark_annotated':lambda *a:None, '_log':lambda *a:None,'logging':logging,'mimics':types.SimpleNamespace(),'TITLE':'Review'}
    exec(extract('runtime_py35/flexict_mimics.py','_al_finish_conversion'), env)
    transaction={'request':{'_request_path':'unused','_job_dir':folder,'case_id':'case1'},'masks':[('uncertainty','unused')],'result_path':str(result_path),'bridge_root':str(bridge)}
    try:
        env['_al_finish_conversion'](transaction)
    except Exception as exc:
        print('AL_CONVERSION:', type(exc).__name__, str(exc))
    print('AL_BUFFER_EXISTED_WHEN_READ:', reads)

module = ast.parse((ROOT/'tools/flexict_active_learning_ui.py').read_text())
cls = next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='ActiveLearningWindow')
method = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_job_selected')
env={'load_ranking':lambda p:[{'case':'case1','integrated':1,'uncertain_vol':1}], 'load_annotation_state':lambda p:{'cases':{}},'bands_degenerate':lambda p:False}
exec(compile(ast.Module(body=[method],type_ignores=[]),'flexict_active_learning_ui.py','exec'),env)
obj=types.SimpleNamespace(job_combo=types.SimpleNamespace(currentIndex=lambda:0),_jobs=[{'job_dir':'dummy'}],table=types.SimpleNamespace(setRowCount=lambda n:None),_case_image_hint=lambda c:'dummy',_set_actions_enabled=lambda b:None,status_label=types.SimpleNamespace(setText=print))
try: env['_job_selected'](obj)
except Exception as exc: print('AL_NONEMPTY_TABLE:',type(exc).__name__,str(exc))

method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_poll_requests')
messages=[]
env={'request_updates':lambda p,e:[{'updated_at_epoch':1,'state':'failed','case':'case1','what':'bands','detail':'source missing'}], 'load_annotation_state':lambda p:{}}
exec(compile(ast.Module(body=[method],type_ignores=[]),'flexict_active_learning_ui.py','exec'),env)
obj=types.SimpleNamespace(_job_dir='dummy',_request_poll_epoch=0,status_label=types.SimpleNamespace(setText=messages.append),_job_selected=lambda:messages.append('1 case(s) ranked · 0 annotated. Double-click a row to open its case in Mimics with the uncertainty bands applied.'))
env['_poll_requests'](obj)
print('AL_STATUS_WRITES:',messages)
