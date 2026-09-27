#!/usr/bin/env python3
"""Run the Mimics-Script regression matrix and write a results artifact.

One command to run every offline test suite the project ships, with per-suite
timing/pass-fail recorded to a JSON artifact. Profiles control how much runs:

  fast (default) — unit + integration suites that need no GUI and no GPU
                   (~10 min). Excludes the ~16 min test_all.py full sweep;
                   it is covered by the ``full`` profile and by CI-style
                   per-change targeted runs (see docs/TEST_STRATEGY.md).
  full           — everything in ``fast`` plus the test_all.py sweep.
  smoke          — the fake-mimics flow tests only (~1 min): the quickest
                   meaningful gate after touching runtime_py35/ entrypoints
                   or the export/import bridges.

Usage:
  python_env/python.exe tools/run_regression_matrix.py [--profile fast|full|smoke]
      [--only name ...] [--list] [--timeout SECONDS] [--output PATH]

Exit code is non-zero when any selected suite fails. The JSON artifact
(default: .mimics_runtime/regression/<timestamp>.json) records per-suite
command, duration, exit code, and stdout/stderr tail, so a red run can be
attached to a bug report verbatim.

GPU-gated suites are deliberately absent: nothing here needs a GPU. The
manual acceptance items that do (FlexiCT train smoke, pair→AL→overlay
end-to-end) are listed in docs/TEST_STRATEGY.md, not automated.
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Each entry: (name, command relative to ROOT, default profile membership).
# Commands are executed with cwd=ROOT using the same interpreter that runs
# this script, so the matrix is portable across hosts.
SUITES = [
    # -- core unit sweep (slow: ~16 min) --------------------------------
    ("test_all", [sys.executable, "tools/test_all.py"], {"full"}),
    # -- per-framework integration suites (unittest, exit-code gated) ----
    ("flexict_integration", [sys.executable, "tools/test_flexict_integration.py"], {"fast", "full"}),
    ("flexict_common", [sys.executable, "tools/test_flexict_common.py"], {"fast", "full"}),
    ("flexict_pkg", [sys.executable, "integrations/flexict-finetune/tests/test_flexict_pkg.py"], {"fast", "full"}),
    ("nnunet_integration", [sys.executable, "tools/test_nnunet_integration.py"], {"fast", "full"}),
    ("nninteractive_bridge_prompts", [sys.executable, "-m", "pytest", "-q", "tools/test_nninteractive_bridge_prompts.py"], {"fast", "full"}),
    ("nninteractive_task_integration", [sys.executable, "tools/test_nninteractive_task_integration.py"], {"fast", "full"}),
    ("nninteractive_finetune_pkg", [sys.executable, "-m", "pytest", "-q", *sorted(
        str(path.relative_to(ROOT)).replace("\\", "/")
        for path in (ROOT / "integrations" / "nninteractive-finetune" / "tests").glob("test_*.py")
    )], {"fast", "full"}),
    ("nnint_deep", [sys.executable, "tools/test_mimics_nnint_deep.py"], {"fast", "full"}),
    ("nnint_functional", [sys.executable, "tools/test_mimics_nnint_functional.py"], {"fast", "full"}),
    ("remote_training", [sys.executable, "tools/test_remote_training.py"], {"fast", "full"}),
    ("interactive_algorithms", [sys.executable, "tools/test_interactive_algorithms.py"], {"fast", "full"}),
    ("model_portability", [sys.executable, "tools/test_model_portability.py"], {"fast", "full"}),
    ("collect_diagnostics", [sys.executable, "tools/test_collect_diagnostics.py"], {"fast", "full"}),
    ("geometry_manifest_regressions", [sys.executable, "tools/test_geometry_manifest_regressions.py"], {"fast", "full"}),
    ("migrate_root", [sys.executable, "tools/test_migrate_root.py"], {"fast", "full"}),
    ("ui_preferences", [sys.executable, "tools/test_ui_preferences.py"], {"fast", "full"}),
    ("gui_smoke", [sys.executable, "tools/test_gui_smoke.py"], {"fast", "full"}),
    # -- fake-mimics end-to-end flow tests ------------------------------
    ("flow_imports", [sys.executable, "tools/fake_mimics_flow_test.py", "--only", "imports"], {"smoke", "fast", "full"}),
    ("flow_export", [sys.executable, "tools/fake_mimics_flow_test.py", "--only", "export"], {"smoke", "fast", "full"}),
    ("flow_entrypoint_window", [sys.executable, "tools/fake_mimics_flow_test.py", "--only", "entrypoint"], {"smoke", "fast", "full"}),
    ("flow_window_level", [sys.executable, "tools/fake_mimics_flow_test.py", "--only", "window"], {"smoke", "fast", "full"}),
    ("flow_nninteractive", [sys.executable, "tools/fake_mimics_flow_test.py", "--only", "nninteractive"], {"smoke", "fast", "full"}),
    ("flow_taskmodels_stop", [sys.executable, "tools/fake_mimics_flow_test.py", "--only", "taskmodels"], {"smoke", "fast", "full"}),
    ("flow_stop", [sys.executable, "tools/fake_mimics_flow_test.py", "--only", "stop"], {"smoke", "fast", "full"}),
    ("flow_append", [sys.executable, "tools/fake_mimics_flow_test.py", "--only", "append"], {"smoke", "fast", "full"}),
    # -- offline stress (concurrency) -----------------------------------
    ("offline_stress", [sys.executable, "tools/offline_stress_test.py"], {"fast", "full"}),
    # -- convergence smoke (real training, tiny scale; full only) --------
    ("training_convergence", [sys.executable, "tools/test_training_convergence.py"], {"full"}),
    # -- cross-workflow handoffs (train -> infer -> AL -> overlay) --------
    ("cross_workflow", [sys.executable, "tools/test_cross_workflow_transitions.py"], {"fast", "full"}),
]

DEFAULT_PROFILE = "fast"
OUTPUT_TAIL_LINES = 40
DEFAULT_TIMEOUT_SECONDS = 3600


def select_suites(profile: str, only: list[str]) -> list[tuple[str, list[str], str]]:
    """Resolve (name, command, membership) tuples for the requested profile."""
    if only:
        known = {name for name, _, _ in SUITES}
        unknown = [name for name in only if name not in known]
        if unknown:
            raise SystemExit("Unknown suite(s): {0}".format(", ".join(unknown)))
        return [(name, cmd, "explicit") for name, cmd, _ in SUITES if name in only]
    return [
        (name, cmd, profile)
        for name, cmd, profiles in SUITES
        if profile in profiles
    ]


def run_suite(name: str, command: list[str], timeout: int) -> dict:
    """Run one suite, return its result row for the artifact."""
    started = time.time()
    print("[run ] {0}: {1}".format(name, " ".join(command[1:])), flush=True)
    try:
        proc = subprocess.run(
            command,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        output = proc.stdout.decode("utf-8", errors="replace")
        code = proc.returncode
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or b"").decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        code = -1
        output += "\n[TIMEOUT after {0}s]".format(timeout)
        # The killed suite process may leave detached grandchildren (job
        # controllers, stage workers) holding the GPU lock. Sweep the process
        # registry so orphans whose parent died are terminated and their locks
        # released (B22) — this is the amplifier of the R41/R42 orphan-hang.
        try:
            sys.path.insert(0, str(ROOT))
            import resource_locks

            summary = resource_locks.sweep_processes(ROOT)
            output += "\n[POST-TIMEOUT SWEEP] {0}".format(
                json.dumps(summary, default=str)
            )
        except Exception as sweep_exc:  # never mask the timeout result
            output += "\n[POST-TIMEOUT SWEEP FAILED] {0}: {1}".format(
                type(sweep_exc).__name__, sweep_exc
            )
    duration = time.time() - started
    status = "pass" if code == 0 else "fail"
    print("[{0}] {1} ({2:.1f}s)".format(status.upper(), name, duration), flush=True)
    tail = "\n".join(output.splitlines()[-OUTPUT_TAIL_LINES:])
    return {
        "name": name,
        "command": command,
        "status": status,
        "exit_code": code,
        "duration_seconds": round(duration, 1),
        "output_tail": tail,
    }


def write_artifact(results: list[dict], profile: str, path: Path) -> Path:
    payload = {
        "schema": "regression_matrix.v1",
        "profile": profile,
        "started_at": results[0]["_started_at"] if results and "_started_at" in results[0] else None,
        "total_suites": len(results),
        "failed_suites": sum(1 for row in results if row["status"] != "pass"),
        "results": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "fast", "full"), default=DEFAULT_PROFILE)
    parser.add_argument("--only", nargs="+", default=None, help="Run only these suites (see --list).")
    parser.add_argument("--list", action="store_true", help="List suites and exit.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS, help="Per-suite timeout in seconds.")
    parser.add_argument("--output", default=None, help="Artifact path (default .mimics_runtime/regression/<ts>.json).")
    args = parser.parse_args(argv)

    if args.list:
        for name, _, profiles in SUITES:
            print("{0:32s} profiles: {1}".format(name, ", ".join(sorted(profiles))))
        return 0

    suites = select_suites(args.profile, args.only or [])
    if not suites:
        print("No suites selected for profile {0!r}.".format(args.profile))
        return 0

    started_at = datetime.datetime.now().isoformat(timespec="seconds")
    print("Regression matrix: {0} suite(s), profile={1}".format(len(suites), args.profile))
    print("=" * 72)
    results = []
    failed = 0
    for name, command, _membership in suites:
        row = run_suite(name, command, args.timeout)
        row["_started_at"] = started_at
        results.append(row)
        if row["status"] != "pass":
            failed += 1
            print(row["output_tail"], flush=True)
            print("-" * 72)

    if args.output:
        output_path = Path(args.output)
    else:
        stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        output_path = ROOT / ".mimics_runtime" / "regression" / "{0}_{1}.json".format(stamp, args.profile)
    write_artifact(results, args.profile, output_path)

    print("=" * 72)
    print("Matrix summary ({0})".format(args.profile))
    for row in results:
        print("  [{0}] {1:32s} {2:7.1f}s".format(row["status"], row["name"], row["duration_seconds"]))
    print("{0}/{1} suites passed. Artifact: {2}".format(len(results) - failed, len(results), output_path))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
