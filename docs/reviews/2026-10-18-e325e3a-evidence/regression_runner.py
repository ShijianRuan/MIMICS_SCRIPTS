"""Affected, isolated offline suite selections; no real case/model execution."""
import concurrent.futures, importlib, json, os, platform, subprocess, sys, unittest
from pathlib import Path
from importlib import metadata

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SOURCE = 'e325e3a8a91c7c64e2c68e1b8b117f6cf08fc31f'
assert not subprocess.check_output(['git','diff','--name-only',SOURCE,'--','tools','runtime_py35','tests','mimics_bridge.py'],cwd=ROOT,text=True).strip(), 'Source changed; preserve old evidence and start a new batch'
SELECTIONS = {
    'nnunet': ['test_nnunet_integration'],
    'flexict': ['test_flexict_integration'],
    'remote': ['test_remote_training'],
    'gui': ['test_gui_smoke'],
    'ui': ['test_ui_quality'],
    'ui-workflow': ['test_ui_workflow_regressions'],
    'geometry': ['test_geometry_manifest_regressions'],
    'cross': ['test_cross_workflow_transitions'],
    'review': ['test_review_workflow_regressions'],
    'runtime': ['test_all.' + name for name in [
        'TestCreateMcsBatch', 'TestEdgeCases', 'TestBridgeDicomLoading',
        'TestImportReceiptAndUndo', 'TestFlexiCTActiveLearningApply',
        'TestGuiThreadBlockingContract']],
}

def child(name):
    sys.path[:0] = [str(ROOT/'tests'), str(ROOT/'tools'), str(ROOT)]
    suite = unittest.TestLoader().loadTestsFromNames(SELECTIONS[name])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    data = dict(source_commit=SOURCE, suite=name, selections=SELECTIONS[name],
                executed=result.testsRun, failures=len(result.failures),
                errors=len(result.errors), skipped=len(result.skipped),
                passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped))
    (HERE/(name+'-result.json')).write_text(json.dumps(data,indent=2)+'\n')
    return 0 if result.wasSuccessful() else 1

def parent():
    packages = {}
    for name in ['numpy','nibabel','pydicom','SimpleITK','PySide6','pytest','torch','scipy']:
        try: packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError: packages[name] = None
    (HERE/'audit-environment.json').write_text(json.dumps(dict(
        source_commit=SOURCE,python=sys.version,platform=platform.platform(),packages=packages,
        limits=['No Windows/Mimics/CUDA/model weights validation','Existing host Python; no environment changes']),indent=2)+'\n')
    env = os.environ.copy(); env.update(QT_QPA_PLATFORM='offscreen', TMPDIR='/private/tmp', PYTHONDONTWRITEBYTECODE='1')
    def run(name):
        with (HERE/(name+'.log')).open('w') as log:
            try:
                p = subprocess.run([sys.executable,str(Path(__file__)),name],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
                return name,p.returncode
            except subprocess.TimeoutExpired: return name,'timeout180s'
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        exits = dict(pool.map(run,SELECTIONS))
    results = [json.loads((HERE/(name+'-result.json')).read_text()) for name in SELECTIONS if (HERE/(name+'-result.json')).exists()]
    data = dict(source_commit=SOURCE,suites=results,exits=exits,
                totals={key:sum(r[key] for r in results) for key in ['executed','passed','failures','errors','skipped']},
                scope='Suite/class method executions may overlap; offline synthetic/mocked boundaries, not end-to-end Mimics acceptance')
    (HERE/'regression-results.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data,indent=2))

if __name__ == '__main__':
    if len(sys.argv)>1: raise SystemExit(child(sys.argv[1]))
    parent()
