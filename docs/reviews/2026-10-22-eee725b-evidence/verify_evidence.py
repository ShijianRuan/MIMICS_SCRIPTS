"""Independent, read-only evidence checks; writes only fact-check.json here.

Does not rerun a model, package, repair, or clinical workflow. Hashes prove
artifact/source identity, not correctness of a real deployment.
"""
import ast
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DOCS = HERE.parent
SOURCE = "eee725b45ec42b7c045def95e4682089a3ea13a2"
BASE = "e33dc74b93cb57b83f9fc863dc6da018a9cecafd"
REPORT = DOCS / "2026-10-22-eee725b-revalidation.md"
RECORD = REPORT.with_suffix(".json")
checks = []


def read(name):
    return json.loads((HERE / name).read_text())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def check(name, passed, detail):
    checks.append(dict(name=name, passed=bool(passed), detail=detail))


def node_at(source, symbol, line):
    nodes = ast.walk(ast.parse(source))
    return any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and n.name == symbol and n.lineno <= line <= n.end_lineno
        for n in nodes
    )


def main():
    report = json.loads(RECORD.read_text())
    scope = read("source-scope.json")
    manifest = read("manifest.json")
    check("source_identity", report["source_commit"] == SOURCE
          and scope["source_commit"] == SOURCE and manifest["source_commit"] == SOURCE
          and report["baseline"] == BASE, "Report, scope and manifest fixed to one source/baseline")
    commits = git("rev-list", "--reverse", BASE + ".." + SOURCE).decode().splitlines()
    files = git("-c", "core.quotepath=false", "diff", "--name-only", BASE, SOURCE).decode().splitlines()
    check("range", commits == [c["sha"] for c in scope["commits"]]
          and len(commits) == 23 and files == scope["changed_files"] and len(files) == 42
          and report["commit_count"] == 23 and report["changed_file_count"] == 42,
          "23 actual commits and 42 actual changed paths; no author/email inference")

    bad = []
    for item in manifest["source_files"]:
        name = item["path"]
        original = git("show", SOURCE + ":" + name)
        if sha(original) != item["sha256"] or (ROOT / name).read_bytes() != original:
            bad.append(name)
    check("source_hashes", not bad, dict(checked=len(manifest["source_files"]), mismatches=bad))
    bad = [i["path"] for i in manifest["artifacts"]
           if not (ROOT / i["path"]).is_file()
           or sha((ROOT / i["path"]).read_bytes()) != i["sha256"]]
    check("artifact_hashes", not bad, dict(checked=len(manifest["artifacts"]), mismatches=bad))

    findings = report["findings"]
    counts = dict(new_in_increment=len(findings),
                  slip=sum(f["id"].startswith("SL") for f in findings),
                  other=sum(not f["id"].startswith("SL") for f in findings),
                  P1=sum(f["priority"] == "P1" for f in findings),
                  P2=sum(f["priority"] == "P2" for f in findings),
                  reproduced_synthetic=sum(f["evidence_type"] == "reproduced_synthetic" for f in findings),
                  source_risk=sum(f["evidence_type"] == "source_risk" for f in findings),
                  previous_retained=len(report["previous_issue_assessment"]))
    check("finding_counts", counts == report["counts"]
          and len({f["id"] for f in findings}) == 8, counts)
    bad = []
    for f in findings:
        data = read(f["evidence_file"])
        for key in f["observations"]:
            if key not in data:
                bad.append(f["id"] + ":" + key)
        for loc in f["locations"]:
            source = git("show", SOURCE + ":" + loc["file"]).decode("utf-8-sig")
            if not node_at(source, loc["symbol"], loc["line"]):
                bad.append(f["id"] + ":" + str(loc))
    check("finding_references", not bad, dict(bad=bad, meaning="Every observation key and source function/line exists"))

    obs = read("observations.json")
    check("synthetic_contract_observations",
          obs["reset_after_manual_edit"] == dict(outcome="applied", manual_preserved=False,
                                                before_voxels=12, after_voxels=12, questions=0)
          and obs["resume_own_gpu_lock"]["code"] == 1
          and obs["resume_own_gpu_lock"]["resume_called"] is False
          and obs["resume_own_gpu_lock"]["own_session_metadata"] is True
          and obs["undo_first_prompt_existing_base"] == dict(outcome="applied", base_voxels=12,
                                                           actual_voxels=0, restored_base=False, questions=0)
          and obs["predict_retry_after_output_failure"]["history_after_failed_write"] == 1
          and obs["predict_retry_after_output_failure"]["history_after_retry"] == 2,
          "SL01/02/03 real consumer + FakeMimics; SL04 real bridge + deterministic predictor")
    archives = obs["actual_synthetic_archives"]
    repair = obs["restore_partial_backup"]
    check("synthetic_distribution_and_restore", len(archives) == 2
          and {a["with_env"] for a in archives} == {True, False}
          and all(a["exit"] == 0 and not a["has_bridge"] and not a["has_checkpoint"] for a in archives)
          and repair["exit"] == 0 and repair["has_any_backup"]
          and not repair["torch_survived"] and not repair["process_holder_check_called"],
          "Tiny real zip builds and real restore logic in temporary directory only")
    coords = obs["vendored_axis_roundtrip"]
    check("coordinate_scope", coords["shape"] == [1, 1, 4, 2, 3]
          and coords["mapped_click"] == [[1.0, 3.0, 2.0]]
          and coords["original_point_true"] and coords["voxel_count"] == 1,
          "Two AST-extracted original methods; no physical/Mimics/model acceptance")
    cpu = obs["cpu_path_source_risk"]
    vendor = (ROOT / "integrations/slip/SLIP_model/inference_module.py").read_text()
    combiner = (ROOT / "integrations/slip/SLIP_model/seg_combiner.py").read_text()
    check("cpu_risk_source", cpu["auto_resolved_device"] == "cpu"
          and not cpu["real_model_run"] and '.to("cuda")' in vendor.splitlines()[cpu["vendored_forced_cuda_line"] - 1]
          and "import cupy" in combiner, "Device selection is observed; full CPU failure/inference not executed")

    ui = read("ui-observations.json")
    stale = ui["stale_previous_state"]
    slow = ui["slow_state_read"]
    check("qt_observations", stale["progress_range"] == [0, 1]
          and stale["progress_value"] == 1 and "1:00:00" in stale["elapsed"]
          and not stale["network_warning_visible"]
          and slow["callback_seconds"] >= slow["injected_seconds"]
          and slow["timer_callbacks_during_read"] == 0
          and slow["timer_callbacks_after_events"] > 0
          and (HERE / "setup-progress-new-check.png").read_bytes().startswith(b"\x89PNG"),
          "Actual offscreen Qt and synthetic state; no actual environment repair")

    regression = read("regression-results.json")
    sums = {k: sum(s[k] for s in regression["suites"])
            for k in ("executed", "passed", "failures", "errors", "skipped")}
    check("regression_counts", sums == regression["totals"] == report["regression_totals"]
          == dict(executed=269, passed=265, failures=1, errors=0, skipped=3)
          and all(s["source_commit"] == SOURCE for s in regression["suites"])
          and all(read(s["suite"] + "-result.json") == s for s in regression["suites"]),
          sums)
    logs = {s["suite"]: (HERE / (s["suite"] + ".log")).read_text() for s in regression["suites"]}
    check("regression_logs", all("Ran " + str(s["executed"]) + " tests" in logs[s["suite"]]
          for s in regression["suites"])
          and "FAIL: test_moved_checkout_detected_with_old_root" in logs["setup-guidance"]
          and "FAILED (failures=1)" in logs["setup-guidance"]
          and "OK (skipped=3)" in logs["gui"]
          and "Ran 50 tests" in logs["slip-runtime"],
          "Primary failure and skip records retained, not reclassified as passes")
    before = git("show", BASE + ":tools/env_guidance.py").decode()
    after = (ROOT / "tools/env_guidance.py").read_text()
    def detector(text):
        return ast.dump(next(n for n in ast.walk(ast.parse(text))
                             if isinstance(n, ast.FunctionDef) and n.name == "detect_old_root"))
    check("diagnostic_source", detector(before) == detector(after)
          and read("regression-diagnostics.json")["detector_ast_unchanged"],
          "Detector AST unchanged; full baseline suite was not rerun")

    previous = report["previous_issue_assessment"]
    old_files = sorted({l["file"] for f in previous for l in f["locations"]})
    check("old_findings_not_recounted", len(previous) == 5
          and all(not git("diff", BASE, SOURCE, "--", name) for name in old_files)
          and {f["id"] for f in previous}.isdisjoint({f["id"] for f in findings}),
          dict(previous_ids=[f["id"] for f in previous], unchanged_files=old_files))

    upstream = read("upstream-comparison.json")
    check("upstream_source_identity", len(upstream["files"]) == 15
          and upstream["commit"] == report["upstream_commit"]
          and all(sha((ROOT / "integrations/slip" / i["path"]).read_bytes()) == i["local_sha256"]
                  and (i["identical"] or i.get("normalized_text_identical")) for i in upstream["files"])
          and sum(i["identical"] for i in upstream["files"]) == 14,
          "Recorded primary-source comparison verified against local bytes; no network rerun")

    missing = []
    for md in [REPORT, HERE / "README.md", DOCS / "README.md"]:
        for target in re.findall(r"\]\(([^)]+)\)", md.read_text()):
            if target.startswith(("https://", "http://", "#")):
                continue
            path = unquote(target.split("#", 1)[0].strip("<>"))
            if path and not (md.parent / path).exists():
                missing.append(str(md.relative_to(ROOT)) + " -> " + target)
    check("document_links", not missing, dict(missing=missing))
    text = REPORT.read_text()
    check("claims_and_limits", all(f["id"] + " · " + f["priority"] in text for f in findings)
          and SOURCE in text and BASE in text and "265" in text
          and "269 个唯一" in text and "TorchIO/CuPy" in text
          and "https://arxiv.org/html/2607.22332v1" in text
          and "不是已经实现或验证的状态" in text,
          "Report explicitly separates probes, source risk, design and unverified deployment")
    changed = git("-c", "core.quotepath=false", "diff", "--name-only", SOURCE).decode().splitlines()
    check("publication_scope", all(p.startswith("docs/reviews/") for p in changed), changed)

    result = dict(source_commit=SOURCE, checked_at_utc=datetime.now(timezone.utc).isoformat(),
                  all_checks_passed=all(c["passed"] for c in checks), checks=checks,
                  limitations="Static evidence audit, not an independent model/Windows/Mimics rerun; external source bodies not re-fetched in this pass")
    (HERE / "fact-check.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
