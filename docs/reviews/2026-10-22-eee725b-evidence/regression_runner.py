"""Affected, isolated offline suite selections; no real case/model execution."""
import concurrent.futures, importlib, json, os, platform, subprocess, sys, unittest
from pathlib import Path
from importlib import metadata

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SOURCE = 'eee725b45ec42b7c045def95e4682089a3ea13a2'
assert not subprocess.check_output(['git','diff','--name-only',SOURCE,'--','tools','runtime_py35','tests','mimics_bridge.py','slip_bridge.py','repair','integrations'],cwd=ROOT,text=True).strip(), 'Source changed; preserve old evidence and start a new batch'
SELECTIONS = {'slip-runtime': ['test_slip_mimics'], 'added-runtime': ['test_all.TestMimicsBridgeNiftiNormalization.test_single_case_worker_canonicalizes_project_root_to_cli_root', 'test_all.TestWindowLevel.test_lung_preset_negative_hu_not_wrapped_to_huge_gv', 'test_all.TestSourceImagePathEquivalence.test_prewarm_official_model_uses_buffer_export_when_fallback_enabled', 'test_all.TestSourceImagePathEquivalence.test_prewarm_official_model_keeps_source_path_when_fallback_disabled', 'test_all.TestSourceImagePathEquivalence.test_prewarm_task_model_strict_source_keeps_source_path', 'test_all.TestImportDropWindow.test_root_form_matches_background_launcher_root', 'test_all.TestEnvGuidance.test_cuda_unavailable_state_points_at_double_click_repair', 'test_all.TestEnvGuidance.test_cuda_unavailable_without_launcher_reports_missing_kit', 'test_all.TestEnvGuidance.test_cuda_available_state_has_no_gpu_issue', 'test_all.TestEnvGuidance.test_stale_cuda_unavailable_state_does_not_nag', 'test_all.TestGpuRepairLauncher.test_launchers_are_ascii_and_target_the_repair_script', 'test_all.TestGpuRepairLauncher.test_switch_script_matches_repo_layout', 'test_all.TestGpuRepairLauncher.test_restore_on_default_is_idempotent', 'test_all.TestGpuRepairLauncher.test_restore_without_backup_keeps_working_variant', 'test_all.TestGpuRepairLauncher.test_restore_removes_variant_and_restores_default'], 'source-fallback': ['test_all.TestSourceImagePathEquivalence'], 'image-window': ['test_all.TestWindowLevel'], 'setup-guidance': ['test_all.TestEnvGuidance', 'test_all.TestGpuRepairLauncher'], 'import': ['test_all.TestImportDropWindow'], 'gui': ['test_gui_smoke']}

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
