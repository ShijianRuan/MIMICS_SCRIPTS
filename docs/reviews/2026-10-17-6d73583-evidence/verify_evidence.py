"""Post-generation independent fact check against actual outputs and Git."""
import ast,hashlib,json,re,subprocess
from pathlib import Path
E=Path(__file__).resolve().parent;R=E.parents[2]
record=json.loads((E.parent/'2026-10-17-6d73583-revalidation.json').read_text())
o=json.loads((E/'observations.json').read_text());u=json.loads((E/'ui-observations.json').read_text())
reg=json.loads((E/'regression-results.json').read_text());checks=[]
if not (E/'fact-check.json').exists():(E/'fact-check.json').write_text('{"result":"pending"}\n')
def check(claim,condition,evidence):
    checks.append({'claim':claim,'verified':bool(condition),'evidence':evidence});assert condition,claim
check('Receipt absent replay writes correct snapshotted generation without changing project',
      o['receipt_replay_missing']['written_guid']=='new-mask-guid' and o['receipt_replay_missing']['project_unchanged'],
      'observations.json:receipt_replay_missing; real replay/JSON files')
x=o['receipt_replay_existing']
check('Existing old receipt is falsely healed and pending new-generation payload discarded',
      x['written_guid']=='old-mask-guid' and x['expected_guid']=='new-mask-guid' and x['healed']==['caseA'] and not x['pending'],
      'observations.json:receipt_replay_existing')
check('Visible receipt failure text is now passed through actual status-row contract',
      'caseA' in o['receipt_visible_note']['note'] and o['receipt_visible_note']['row']['receipt_failed']==1,
      'observations.json:receipt_visible_note and actual batch-receipt-warning.png')
check('Changed existing target is protected and validated candidate retained',
      not o['publish_changed_existing']['accepted'] and o['publish_changed_existing']['newer_target_preserved'] and o['publish_changed_existing']['candidate_parked'],
      'observations.json:publish_changed_existing')
x=o['publish_appeared_later']
check('Absent-at-start guard is disabled and later target replaced',
      x['initial_fingerprint'] is None and x['accepted'] and not x['newer_target_preserved'] and x['final_value']==2,
      'observations.json:publish_appeared_later; exact production fingerprint/publisher chain')
check('Normal restore succeeds and later manual edit blocks another restore',
      o['restore_normal']['current_value']==1 and o['restore_normal']['kept_previous_exists'] and o['restore_after_manual_edit']['blocked'] and o['restore_after_manual_edit']['current_preserved'],
      'observations.json:restore_*')
c=o['gzip_cost']
check('96-slice actual gzip decode is near linear ~1.04x',
      c['payload_bytes']==393216 and c['decompressed_bytes']==409952 and c['amplification']<1.1,
      'observations.json:gzip_cost; actual reader return-byte count')
check('Fixed explicit-count grouped and four-case splits preserve c0-only validation',
      all(x['val']==['c0'] and x['train']==['c1'] for x in o['frozen_grouped']) and all(x['val']==['c0'] for x in o['frozen_four']),
      'observations.json:frozen_grouped/frozen_four')
check('Automatic proportion still expands frozen validation after data growth',
      len(o['frozen_default_expansion'][0]['val'])==2 and 'c0' in o['frozen_default_expansion'][0]['val'],
      'observations.json:frozen_default_expansion')
d={x['case']:x for x in o['dicom']}
check('Partly missing spacing/all-missing position reject; regular input passes',
      not d['partly_missing_spacing']['accepted'] and not d['all_missing_position']['accepted'] and d['regular']['accepted'],
      'observations.json:dicom; real public geometry functions with synthetic headers')
check('All-missing spacing and nonorthogonal directions remain accepted',
      d['all_missing_spacing']['accepted'] and d['all_missing_spacing']['affine'][0][0]==-1
      and d['non_orthogonal_direction']['accepted'] and abs(d['non_orthogonal_direction']['affine'][0][1])>.7,
      'observations.json:dicom; DICOM official plane-module semantic reference linked in report')
check('Single underscore exact labels resolve while double underscore aliases remain ambiguous',
      all('files' in r and len(r['files'])==1 for r in o['flat_labels_single'].values())
      and all('more than one' in r['error'] for r in o['flat_labels_double'].values()),
      'observations.json:flat_labels_*')
x=o['fifo_exit_drain']
check('Exit drain reports zero while second write is active and first value still persisted',
      x['pending_reported']==0 and x['second_write_inflight'] and not x['busy_flag'] and x['queue_size']==0
      and x['persisted_before_second_release']=='annotated' and o['fifo_final_state']=='skipped',
      'observations.json:fifo_exit_drain/fifo_final_state; controlled schedule and actual OS lock/JSON')
check('Original AL partial-write result now excludes failed case',
      o['al_partial_write']['code']==0 and o['al_partial_write']['ranked']==['case00','case02'] and o['al_partial_write']['excluded_case_count']==1,
      'observations.json:al_partial_write; only model/GPU and consensus write boundaries replaced')
x=o['al_prefix_collision']
check('Shared-prefix successful case files get quarantined and whole AL run fails safely',
      x['code']==1 and '成功清单不一致' in x['error'] and 'case01_phase2_consensus.nii.gz' in x['failed_fragments']
      and 'case01_phase2_uncertainty_disagreement.nii.gz' in x['failed_fragments'],
      'observations.json:al_prefix_collision; actual full AL chain with failed consensus save')
check('Same-second generated run basenames differ by UUID',
      o['al_uuid_names']['different'] and o['al_uuid_names']['both_completed'],
      'observations.json:al_uuid_names; two invocations, naming clock controlled')
for name in ['nnunet','flexict']:
    e=u[name+'_recovery_entry'];c=u[name+'_restore_callback']
    check(name+' recovery button is visible and actual restore callback blocks UI under slow-storage simulation',
          e['enabled'] and e['button_visible_size']==e['button_size'] and c['fingerprint_reads_on_ui_thread']
          and c['elapsed_seconds']>.39 and c['timer_callbacks_during_restore']==0 and c['timer_callbacks_after_process_events']>=1 and c['restored_value']==1,
          'ui-observations.json and manually inspected actual Qt PNG; 200ms injected per fingerprint read, not native performance claim')
check('Primary regression totals preserve failures, missing dependency and platform skips',
      reg['totals']=={'executed':734,'passed':728,'failures':2,'errors':1,'skipped':3},
      '10 final primary raw logs; no diagnostic replacements counted')
check('Previous-card and open-card counts align with independent findings',
      len(record['previous_issue_assessment'])==8 and sum(x['status']=='verified_offline' for x in record['previous_issue_assessment'])==2
      and len(record['findings'])==8 and sum(x['classification']=='remaining_partial' for x in record['findings'])==6,
      '2026-10-17-6d73583-revalidation.json')
check('Git audit range has 9 commits and 20 changed files',
      int(subprocess.check_output(['git','rev-list','--count',record['baseline']+'..'+record['source_commit']],cwd=R,text=True))==9
      and len(subprocess.check_output(['git','diff','--name-only',record['baseline'],record['source_commit']],cwd=R,text=True).splitlines())==20,
      'exact Git topology and fixed source SHA')
for f in record['findings']:
    evidence=u if f['evidence_file']=='ui-observations.json' else o
    for k in f['observations']:assert k in evidence,k
    for l in f['locations']:assert (R/l['file']).read_text().splitlines()[l['line']-1].lstrip().startswith(('def '+l['symbol'],'class '+l['symbol'])),l
for p in [E.parent/'2026-10-17-6d73583-revalidation.md',E/'README.md',E.parent/'README.md']:
    text=p.read_text()
    if p!=E.parent/'README.md':assert len(re.findall(r'^# ',text,re.M))==1
    for link in re.findall(r'\]\(([^)]+)\)',text):
        if not link.startswith(('https:','http:','#')):assert (p.parent/link.split('#')[0]).is_file(),link
for p in E.glob('*.json'):json.loads(p.read_text())
for p in E.glob('*.py'):ast.parse(p.read_text())
m=json.loads((E/'manifest.json').read_text())
for path,digest in m['tracked_source_sha256'].items():
    assert hashlib.sha256(subprocess.check_output(['git','show',record['source_commit']+':'+path],cwd=R)).hexdigest()==digest,path
    assert hashlib.sha256((R/path).read_bytes()).hexdigest()==digest,path
for path,digest in m['artifacts_sha256'].items():assert hashlib.sha256((R/path).read_bytes()).hexdigest()==digest,path
check('Source snapshot, artifact hashes, document links, JSON, executable syntax and source locations validate',True,'manifest and explicit checks')
(E/'fact-check.json').write_text(json.dumps({'source_commit':record['source_commit'],'result':'pass','method':'Independent post-generation check against observed outputs and source identity; no Windows/Mimics medical validation claim','checks':checks},ensure_ascii=False,indent=2)+'\n')
for p in E.iterdir():
    if p.is_file() and p.name!='manifest.json':m['artifacts_sha256'][str(p.relative_to(R))]=hashlib.sha256(p.read_bytes()).hexdigest()
(E/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
print('PASS:',len(checks),'claims plus links, AST, JSON, source/artifact hashes and source locations')
