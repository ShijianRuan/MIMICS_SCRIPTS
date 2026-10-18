"""Separate post-generation fact check: real observations, fixed Git source,
links, counts, source locations and evidence hashes. No production execution.
"""
import ast,hashlib,json,re,subprocess
from pathlib import Path
E=Path(__file__).resolve().parent;R=E.parents[2]
record=json.loads((E.parent/'2026-10-18-e325e3a-revalidation.json').read_text())
o=json.loads((E/'observations.json').read_text());u=json.loads((E/'ui-observations.json').read_text())
reg=json.loads((E/'regression-results.json').read_text());checks=[]
def check(claim,condition,evidence):
    checks.append({'claim':claim,'verified':bool(condition),'evidence':evidence})
    assert condition,claim
check('Normal old receipt replay now writes the queued generation',
      o['receipt_replay_existing']['written_guid']=='new-mask-guid' and not o['receipt_replay_existing']['pending'],
      'observations.json:receipt_replay_existing')
x=o['receipt_clock_rollback']
check('Clock rollback keeps old guid despite current project matching queued payload, then clears pending',
      x['disk_guid']=='old-guid' and x['project_matches_queued'] and x['disk_epoch']>x['queued_epoch'] and x['healed']==['caseA'] and not x['pending'],
      'observations.json:receipt_clock_rollback')
check('A target appearing after capture is preserved and candidate parked',
      o['publish_appeared_later']['initial_fingerprint']=='__absent__' and not o['publish_appeared_later']['accepted']
      and o['publish_appeared_later']['newer_target_preserved'] and o['publish_appeared_later']['candidate_parked'],
      'observations.json:publish_appeared_later')
x=o['publish_concurrent']
check('Both concurrent protected publishers report success; second retains first as backup',
      x['expected']=='__absent__' and not any(x['checks'].values()) and all(v['success'] for v in x['results'].values())
      and x['results']['1']['backup_value']==1 and x['final_value']==2,
      'observations.json:publish_concurrent; only scheduling gate injected')
d={x['case']:x for x in o['dicom']}
check('Missing spacing/multiple-slice bad direction refuse; valid single/multiple input pass',
      not d['all_missing_spacing']['accepted'] and not d['non_unit_direction']['accepted']
      and not d['non_orthogonal_direction']['accepted'] and d['single_regular']['accepted'] and d['regular']['accepted'],
      'observations.json:dicom')
check('Single-plane unit but nonorthogonal IOP remains accepted with sheared affine',
      d['single_nonorthogonal']['accepted'] and .7<abs(d['single_nonorthogonal']['affine'][0][1])<.71,
      'observations.json:dicom')
check('Automatic frozen validation remains c0 across all five folds as dataset grows',
      len(o['frozen_default_expansion'])==5 and all(x['val']==['c0'] and len(x['train'])==7 for x in o['frozen_default_expansion']),
      'observations.json:frozen_default_expansion')
check('Double-underscore exact labels select one distinct file apiece',
      o['flat_labels_double']['Liver-Left']['files']==['caseA__Liver-Left.nii.gz']
      and o['flat_labels_double']['Liver_Left']['files']==['caseA__Liver_Left.nii.gz'],
      'observations.json:flat_labels_double')
check('In-flight FIFO count now remains one before last write lands',
      o['fifo_exit_drain']['pending_reported']==1 and o['fifo_exit_drain']['queue_size']==0 and o['fifo_final_state']=='skipped',
      'observations.json:fifo_exit_drain/fifo_final_state')
check('Failed write still drains to zero despite empty persisted state',
      o['failed_write_exit_drain']['pending_reported']==0 and not o['failed_write_exit_drain']['disk_cases']
      and 'synthetic storage denial' in o['failed_write_exit_drain']['failure_receipt'][2],
      'observations.json:failed_write_exit_drain')
check('Prefix collision AL chain completes while excluding only failed case',
      o['al_prefix_collision']['code']==0 and o['al_prefix_collision']['ranked']==['case00','case01_phase2']
      and o['al_prefix_collision']['excluded_case_count']==1,
      'observations.json:al_prefix_collision')
for name in ['nnunet','flexict']:
    x=u[name+'_restore_callback'];b=u[name+'_recovery_entry']
    check(name+' real restore worker preserves GUI responsiveness and restores old voxels',
          x['return_seconds']<.1 and x['fingerprint_reads_off_ui_thread'] and x['timer_callbacks_during_restore']>=10
          and x['restored_value']==1 and x['dialogs']==['已恢复旧结果'] and b['enabled'] and b['button_size']==b['button_visible_size'],
          'ui-observations.json and actual 853x440 Qt PNGs; delay is synthetic, not a performance benchmark')
    x=u[name+'_status_viewer_close_during_swap']
    check(name+' normal Qt process exit interrupts restore between real swaps; versions preserved, status stale',
          x['process_exit']==0 and x['app_exit']==0 and x['entered_swap'] and x['worker_is_daemon']
          and not x['output_exists_after_exit'] and x['old_exists'] and x['kept_versions']==[2] and x['status_still_completed'],
          'ui-observations.json; child process normal exit, no kill')
check('Closed import task feedback no longer claims ongoing replay',
      '任务已结束' in u['receipt_warning']['note'] and '正在自动补写' not in u['receipt_warning']['note'],
      'ui-observations.json:receipt_warning and batch-receipt-warning.png')
check('Ten suite/class selections preserve primary results, failures and skips',
      len(reg['suites'])==10 and reg['totals']=={'executed':698,'passed':693,'failures':2,'errors':0,'skipped':3}
      and reg['totals']=={k:sum(r[k] for r in reg['suites']) for k in reg['totals']},
      'regression-results.json and each raw primary log')
for result in reg['suites']:
    log=(E/(result['suite']+'.log')).read_text()
    assert 'Ran '+str(result['executed'])+' tests' in log,(result['suite'],result['executed'])
check('Eight previous cards assessed independently; four partial plus one new yields five open',
      len(record['previous_issue_assessment'])==8 and sum(x['status']=='verified_offline' for x in record['previous_issue_assessment'])==4
      and len(record['findings'])==5 and sum(x['classification']=='remaining_partial' for x in record['findings'])==4
      and sum(x['priority']=='P1' for x in record['findings'])==4,
      '../2026-10-18-e325e3a-revalidation.json')
check('Exact increment is ten commits and nineteen files',
      int(subprocess.check_output(['git','rev-list','--count',record['baseline']+'..'+record['source_commit']],cwd=R,text=True))==10
      and len(subprocess.check_output(['git','diff','--name-only',record['baseline'],record['source_commit']],cwd=R,text=True).splitlines())==19,
      'Git topology; source-scope.json')
for f in record['findings']:
    obs=u if f['evidence_file']=='ui-observations.json' else o
    for key in f['observations']:assert key in obs,key
    for loc in f['locations']:
        assert (R/loc['file']).read_text().splitlines()[loc['line']-1].lstrip().startswith('def '+loc['symbol']),loc
for p in [E.parent/'2026-10-18-e325e3a-revalidation.md',E.parent/'README.md',E/'README.md']:
    value=p.read_text()
    if p!=E.parent/'README.md':assert len(re.findall(r'^# ',value,re.M))==1
    for link in re.findall(r'\]\(([^)]+)\)',value):
        if not link.startswith(('https:','http:','#')):assert (p.parent/link.split('#')[0]).is_file(),link
for p in E.glob('*.py'):ast.parse(p.read_text())
for p in E.glob('*.json'):json.loads(p.read_text())
m=json.loads((E/'manifest.json').read_text())
for path,digest in m['tracked_source_sha256'].items():
    assert hashlib.sha256(subprocess.check_output(['git','show',record['source_commit']+':'+path],cwd=R)).hexdigest()==digest,path
    assert hashlib.sha256((R/path).read_bytes()).hexdigest()==digest,path
for path,digest in m['artifact_sha256'].items():
    assert hashlib.sha256((E.parent/path).read_bytes()).hexdigest()==digest,path
check('Source symbols, links, AST, JSON and pinned Git/source/evidence hashes agree',True,'manifest.json; symbol and relative-link checks above')
(E/'fact-check.json').write_text(json.dumps(dict(source_commit=record['source_commit'],result='PASS',checks=checks,
    limits=['No claim of all bugs found or native Windows/Mimics/GPU acceptance','Schedule gates are injected; actual file/Qt effects are observed']),ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'result':'PASS','claims':len(checks),'source':record['source_commit']}))
