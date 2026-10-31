"""Verify pinned source and review evidence, without running ML or Mimics."""
import ast
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
from urllib.parse import unquote, urlsplit

E = Path(__file__).resolve().parent
ROOT = E.parents[2]
REVIEW = E.parent / '2026-10-31-09d5948-revalidation.json'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def main():
    r = json.loads(REVIEW.read_text())
    s, b = r['source_commit'], r['baseline']
    commits = git('rev-list', b+'..'+s).decode().splitlines()
    changed = git('diff', '--name-only', b, s).decode().splitlines()
    assert len(commits) == 39 == r['scope']['commit_count']
    assert len(changed) == 32 == r['scope']['changed_file_count']
    assert set(changed) == set(r['scope']['changed_files'])
    assert set(commits) == {c['sha'] for c in r['scope']['commits']}
    sources = json.loads((E/'source-files.json').read_text())
    for file, expected in sources['sha256'].items():
        raw = git('show', s+':'+file)
        assert hashlib.sha256(raw).hexdigest() == expected, file
        assert (ROOT/file).read_bytes() == raw, 'Working source changed: '+file
    count = 0
    for file in changed:
        if file.endswith('.py'):
            raw = git('show', s+':'+file).decode('utf-8-sig')
            ast.parse(raw, feature_version=(3, 5) if file.startswith('runtime_py35/') else None)
            count += 1
    assert count == 29
    observations = json.loads((E/'observations.json').read_text())
    assert observations['source_commit'] == s
    assert len(observations['observations']) == 6
    assert len(r['findings']) == 6
    assert sum(x['priority'] == 'P1' for x in r['findings']) == 4
    for card in r['findings']:
        assert observations['observations'][card['observation']]['outcome'] == 'reproduced'
        for location in card['locations']:
            code = git('show', s+':'+location['file']).decode()
            assert 0 < location['line'] <= len(code.splitlines())
        assert all(card[k] for k in ['scenario','observed','cause','impact','what_to_change','why','how','acceptance'])
    o = observations['observations']
    assert o['same_guid_changed_grid']['same_worker_reused']
    assert o['busy_guard_before_adoption']['prewarm_attempted_with_other_image_running']
    assert o['unowned_live_pid_retirement']['actual_kill_executed'] is False
    assert o['project_voxel_cache_omits_intensity_contract']['cached_values'] != o['project_voxel_cache_omits_intensity_contract']['fresh_values']
    assert o['ordinary_case_name_misclassified_as_archive']['returned_older_case'] == 'ordinary-case'
    matrix = json.loads((E/'regression-matrix.json').read_text())
    assert matrix['total_suites'] == 14
    assert matrix['failed_suites'] == 1
    assert sum(x['status'] == 'pass' and x['exit_code'] == 0 for x in matrix['results']) == 13
    failure = next(x for x in matrix['results'] if x['status'] != 'pass')
    assert failure['name'] == 'ui_workflow_regressions' and failure['exit_code'] == 1
    assert 'FAILED (failures=2)' in failure['output_tail']
    targeted = (E/'nninteractive-targeted.log').read_text()
    assert re.search(r'Ran 148 tests.*\n\nOK\s*$', targeted)
    ui = json.loads((E/'ui-failure-observations.json').read_text())
    completion = ui['test_completion_long_copy_wraps_within_viewport']
    assert completion['settled_detail_height'] == 14 < completion['settled_detail_size_hint']
    for file in E.glob('*.png'):
        raw = file.read_bytes()
        assert raw[:8] == b'\x89PNG\r\n\x1a\n'
        assert struct.unpack('>II', raw[16:24]) == (853, 440)
    links = 0
    for md in [REVIEW.with_suffix('.md'), E/'README.md', E.parent/'README.md']:
        for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', md.read_text()):
            link = link.strip().strip('<>')
            if urlsplit(link).scheme or link.startswith('#'):
                continue
            assert (md.parent/unquote(link.split('#',1)[0])).exists(), (md,link)
            links += 1
    facts = json.loads((E/'fact-check.json').read_text())
    assert facts['source_commit'] == s and facts['unverified_outcomes']
    for relative, expected in json.loads((E/'evidence-sha256.json').read_text())['sha256'].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() == expected, relative
    print(json.dumps(dict(status='pass', source_commit=s, remote_commits=39, changed_files=32,
        syntax_files=count, source_files=len(sources['sha256']), finding_cards=6,
        matrix='13 pass / 1 fail (2 UI assertions retained)', targeted_tests=148,
        verified_local_links=links, boundary='Evidence/source consistency, not a claim that product tests all pass'), indent=2))


if __name__ == '__main__':
    main()
