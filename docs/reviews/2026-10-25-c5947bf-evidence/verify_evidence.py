"""Check this immutable review's identity and links; does not execute ML."""
from pathlib import Path
import ast
import hashlib
import json
import re
import struct
import subprocess
from urllib.parse import unquote, urlsplit

EVIDENCE = Path(__file__).resolve().parent
ROOT = EVIDENCE.parents[2]
REVIEW = EVIDENCE.parent / "2026-10-25-c5947bf-active-learning.json"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def main():
    review = json.loads(REVIEW.read_text())
    scope = json.loads((EVIDENCE / "source-scope.json").read_text())
    source = review["reviewed_source"]
    implementation = review["implementation_commit"]
    baseline = review["baseline"]
    commits = git("rev-list", baseline + ".." + source).decode().splitlines()
    changed = git("diff", "--name-only", baseline, source).decode().splitlines()
    assert len(commits) == 29 == scope["new_commit_count"]
    assert len(changed) == 29 == scope["changed_remote_file_count"]
    assert set(commits) == {c["sha"] for c in scope["commits"]}
    assert set(changed) == set(scope["changed_remote_files"])
    git("merge-base", "--is-ancestor", source, implementation)
    implementation_files = git("diff-tree", "--no-commit-id", "--name-only", "-r", implementation).decode().splitlines()
    assert len(implementation_files) == 15
    source_manifest = json.loads((EVIDENCE / "source-files.json").read_text())
    assert source_manifest["source_commit"] == implementation
    for file, expected in source_manifest["sha256"].items():
        committed = git("show", implementation + ":" + file)
        assert digest(committed) == expected, file
        assert (ROOT / file).read_bytes() == committed, "Working source changed: " + file
    for file in implementation_files:
        code = git("show", implementation + ":" + file).decode("utf-8-sig")
        if file.startswith("runtime_py35/"):
            ast.parse(code, feature_version=(3, 5))
        else:
            ast.parse(code)
    fixes = review["fixes"]
    assert len(fixes) == 9
    assert sum(f["priority"] == "P1" for f in fixes) == 6
    tests = ast.parse((ROOT / "tests/test_active_learning_review_regressions.py").read_text())
    available = {c.name + "." + f.name for c in tests.body if isinstance(c, ast.ClassDef)
                 for f in c.body if isinstance(f, ast.FunctionDef) and f.name.startswith("test_")}
    assert len(available) == 16
    for card in fixes:
        code = git("show", implementation + ":" + card["file"]).decode()
        assert "def " + card["symbol"] + "(" in code.splitlines()[card["line"] - 1]
        assert card["test"] in available
        assert card["why"] and card["acceptance"]
    validation = json.loads((EVIDENCE / "validation-summary.json").read_text())
    matrix = json.loads((EVIDENCE / "regression-matrix.json").read_text())
    assert validation["implementation_source"] == implementation
    assert matrix["total_suites"] == len(matrix["results"]) == 12
    assert matrix["failed_suites"] == 0
    assert all(r["status"] == "pass" and r["exit_code"] == 0 for r in matrix["results"])
    assert sum(int(re.findall(r"Ran (\d+) tests", r["output_tail"])[-1]) for r in matrix["results"]) == 556
    assert len(json.loads((EVIDENCE / "setup-progress-stability.json").read_text())) == 5
    assert all(r["exit_code"] == 0 for r in json.loads((EVIDENCE / "setup-progress-stability.json").read_text()))
    assert validation["retained_initial_failures"][1]["process_exit_code"] == 139
    assert all((EVIDENCE / r["log"]).is_file() for r in validation["retained_initial_failures"])
    for suffix, dims in [("", (820, 650)), ("-scale-1_5", (1230, 975))]:
        for backend in ["nnunet", "flexict"]:
            raw = (EVIDENCE / (backend + "-review" + suffix + ".png")).read_bytes()
            assert raw[:8] == b"\x89PNG\r\n\x1a\n"
            assert struct.unpack(">II", raw[16:24]) == dims
        obs = json.loads((EVIDENCE / ("ui-observations" + suffix + ".json")).read_text())
        for backend in obs.values():
            assert backend["selected_case"] == "caseA"
            assert all(b["visible"] and b["text_width"] < b["width"] for b in backend["buttons"].values())
    link_count = 0
    for md in [EVIDENCE / "README.md", REVIEW.with_suffix(".md"), EVIDENCE.parent / "README.md"]:
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", md.read_text()):
            target = target.strip().strip("<>")
            if urlsplit(target).scheme or target.startswith("#"):
                continue
            relative = unquote(target.split("#", 1)[0])
            assert (md.parent / relative).exists(), (md.name, target)
            link_count += 1
    artifact_manifest = json.loads((EVIDENCE / "evidence-sha256.json").read_text())
    for relative, expected in artifact_manifest["sha256"].items():
        assert digest((ROOT / relative).read_bytes()) == expected, relative
    result = dict(status="pass", source_commit=implementation, remote_source=source,
                  remote_commits=29, remote_changed_files=29, implementation_files=15,
                  regression_suites=12, reported_tests=556, platform_skips=3,
                  new_contract_tests=16, verified_local_links=link_count,
                  source_files=len(source_manifest["sha256"]), artifacts=len(artifact_manifest["sha256"]),
                  boundary="Artifact/source consistency only; not full repository or native clinical validation")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
