"""Separate post-generation fact check; run from any working directory."""
import ast, hashlib, json, re, subprocess
from pathlib import Path

EV=Path(__file__).resolve().parent;ROOT=EV.parents[2]
record=json.loads((EV.parent/'2026-10-17-336e99d-revalidation.json').read_text())
o=json.loads((EV/'observations.json').read_text())
ui=json.loads((EV/'ui-observations.json').read_text())
reg=json.loads((EV/'regression-results.json').read_text())
checks=[]
if not (EV/'fact-check.json').exists():
    (EV/'fact-check.json').write_text('{"result":"pending"}\n')
def check(claim,condition,evidence):
    checks.append({'claim':claim,'verified':bool(condition),'evidence':evidence})
    assert condition,claim

check('Normal receipt producer→queue handoff writes actual snapshot data',
      o['receipt_success']['receipt_exists'] and bool(o['receipt_success']['receipt']['created_masks_v2'][0]['sha256']),
      'observations.json:receipt_success; native project creation boundary stub')
f=o['receipt_failure']
check('Receipt failure remains invisible to normal task-row data and cleanup removed work',
      f['status']['receipt_failed']==1 and f['code']==0 and not f['receipt_exists']
      and f['viewer_row']['failed']==0 and not f['viewer_row']['error'] and not f['work_dir_retained'],
      'observations.json:receipt_failure; write boundary raises OSError')
check('Valid publication preserves exact old backup bytes',o['publish_valid']['backup_matches_old'], 'observations.json:publish_valid')
check('Truncated/4D/fractional outputs are rejected before replacing target',
      all(not o[k]['accepted'] and o[k]['old_preserved'] for k in ['publish_truncated','publish_four_d','publish_fractional']),
      'observations.json:publish_*')
c=o['gzip_validation_cost']
check('Without indexed_gzip, 96-slice validation amplifies decompression ~49.62x',
      not c['indexed_gzip_available'] and c['payload_bytes']==393216 and c['decompressed_bytes']==19510272,
      'observations.json:gzip_validation_cost; actual gzip reader bytes')
check('Missing-proof and edited masks are retained, proven-unchanged mask can be deleted',
      all(o[k]['remaining']==1 for k in ['undo_legacy','undo_failed_snapshot','undo_edited']) and o['undo_unchanged']['remaining']==0,
      'observations.json:undo_*; native deletion is a list boundary stub')
check('FlexiCT frozen original cases no longer grow validation',
      o['flexict_frozen_two'][0]=={'train':['c1'],'val':['c0']} and o['flexict_frozen_four'][0]['val']==['c0'],
      'observations.json:flexict_frozen_*')
check('nnU-Net still grows frozen validation and final writer publishes empty grouped training folds',
      o['nnunet_frozen_four'][0]['val']==['c0','c3'] and all(not r['train'] and len(r['val'])==2 for r in o['nnunet_writer_empty_train']),
      'observations.json:nnunet_*; actual dataset/splits writer')
d={x['case']:x for x in o['dicom_geometry']}
check('DICOM original counterexamples reject; regular and linear shear remain supported',
      all(not d[k]['accepted'] for k in ['zigzag','missing_ipp','varying_spacing']) and d['regular']['accepted'] and d['linear_shear']['accepted'],
      'observations.json:dicom_geometry; synthetic pydicom headers through public geometry functions')
check('Partly missing spacing and entirely missing position still accept fabricated geometry',
      d['partly_missing_spacing']['accepted'] and d['partly_missing_spacing']['affine'][0][0]==-2 and d['all_missing_ipp']['accepted'],
      'observations.json:dicom_geometry')
check('Subfolder exact labels resolve uniquely while equivalent flat labels reject as ambiguous',
      o['label_subfolder_exact']=={'Liver-Left':['Liver-Left.nii.gz'],'Liver_Left':['Liver_Left.nii.gz']} and all('more than one' in r['error'] for r in o['label_flat_exact'].values()),
      'observations.json:label_*')
check('Async lock-held callback returns promptly and persists after unlock',
      o['async_state_responsive']['callback_seconds']<.5 and o['async_state_responsive']['persisted']['cases']['A']['state']=='annotated',
      'observations.json:async_state_responsive; native Windows Qt lock test is skipped')
a=o['async_state_click_order']
check('Controlled thread schedule reverses last-click intent in actual persisted state',
      a['click_order']==['annotated','skipped'] and a['observed_persistence_order']==['skipped','annotated'] and a['final_state']=='annotated',
      'observations.json:async_state_click_order; actual JSON and OS file lock')
check('AL same-second retry reuses directory and ranks excluded case',
      o['al_same_second_retry']['directory_reused'] and len(o['al_same_second_retry']['ranked'])==3 and o['al_same_second_retry']['excluded_case_count']==1,
      'observations.json:al_same_second_retry; controlled directory clock, actual AL chain')
p=o['al_fresh_partial_write_failure']
check('Fresh AL directory still ranks current-run partial write failure',
      p['fresh_directory'] and len(p['ranked'])==3 and p['excluded_case_count']==1 and not p['case01_consensus_exists'],
      'observations.json:al_fresh_partial_write_failure; only consensus write fails')
check('Joint wrong input grid is rejected by full AL chain',
      o['al_wrong_grid_rejected']['code']==1 and o['al_wrong_grid_rejected']['status']=='failed' and '没有病例成功' in o['al_wrong_grid_rejected']['error'],
      'observations.json:al_wrong_grid_rejected')
l=ui['completion_layout']
check('Fixed small-screen real Qt scene wraps without horizontal clipping and version navigation works',
      l['word_wrap'] and l['visible_width']==l['label_width'] and all(x['maximum']==0 for x in l['horizontal_scrolls'])
      and ui['completed_model_navigation_after_click']['selected_row']==0,
      'ui-observations.json and 7 manually inspected actual Qt PNGs')
check('Primary tests totals retain all failures/errors/skips',
      reg['totals']=={'executed':714,'passed':709,'failures':3,'errors':1,'skipped_methods':1},
      '10 raw primary logs; no diagnostic substitutions counted')
check('Independent status assessments preserve scoped fixes and pending cards',
      len(record['previous_issue_assessment'])==11 and sum(x['status']=='verified_offline' for x in record['previous_issue_assessment'])==7
      and len(record['findings'])==8 and sum(x['classification']=='remaining_partial' for x in record['findings'])==4,
      '2026-10-17-336e99d-revalidation.json')
check('New audit range is exactly 12 commits and 20 changed files',
      int(subprocess.check_output(['git','rev-list','--count',record['baseline']+'..'+record['source_commit']],cwd=ROOT,text=True))==12
      and len(subprocess.check_output(['git','diff','--name-only',record['baseline'],record['source_commit']],cwd=ROOT,text=True).splitlines())==20,
      'Git topology and fixed source snapshot')
for file in [EV.parent/'2026-10-17-336e99d-revalidation.md',EV/'README.md',EV.parent/'README.md']:
    text=file.read_text()
    if file!=EV.parent/'README.md':assert len(re.findall(r'^# ',text,re.M))==1
    for link in re.findall(r'\]\(([^)]+)\)',text):
        if link.startswith(('https:','http:','#')):continue
        assert (file.parent/link.split('#')[0]).is_file(),str(file)+': '+link
for f in record['findings']:
    for key in f['observations']:assert key in o,key
    for l in f['locations']:
        assert (ROOT/l['file']).read_text().splitlines()[l['line']-1].lstrip().startswith(('def '+l['symbol'],'class '+l['symbol'])),l
for file in EV.glob('*.py'):ast.parse(file.read_text())
for file in EV.glob('*.json'):json.loads(file.read_text())
m=json.loads((EV/'manifest.json').read_text())
for path,digest in m['tracked_source_sha256'].items():
    contents=subprocess.check_output(['git','show',record['source_commit']+':'+path],cwd=ROOT)
    assert hashlib.sha256(contents).hexdigest()==digest,path
    assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
for path,digest in m['artifacts_sha256'].items():
    assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
check('Links, executable evidence, source identity and artifact hashes are valid',True,'explicit link/AST/JSON/line checks plus manifest hashes')
(EV/'fact-check.json').write_text(json.dumps({'source_commit':record['source_commit'],'method':'Separate post-generation verification against actual observations, test logs, screenshots and Git source content; no native medical validation claim.','checks':checks,'result':'pass'},ensure_ascii=False,indent=2)+'\n')
for path in EV.iterdir():
    if path.is_file() and path.name!='manifest.json':m['artifacts_sha256'][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
(EV/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
print('PASS:',len(checks),'claims plus source/artifact hashes, links, JSON, AST and source line checks')
