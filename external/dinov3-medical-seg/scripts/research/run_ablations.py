#!/usr/bin/env python3
"""Run one GPU-safe stage of the multi-organ few-shot study sequentially."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.research.fingerprint import build_training_fingerprint, derive_policy
from src.research.protocol import atomic_json
from src.research.regime import REGIME_CELL_IDS, apply_regime, regime_cells
from src.utils.config import deep_merge, load_config


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve(root: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (root / path).resolve()


def _selected_candidates(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return set(payload.get("selected_candidate_ids", []))


def _validate_epoch_budget(study: dict, phase: str) -> None:
    """Fail fast if a phase's epoch budget is non-positive.

    An epochs<1 budget makes the trainer's ``range(1, epochs+1)`` loop empty, so
    no checkpoint is ever written yet the process exits 0 — which the run loop
    would misreport as a training failure. Surfacing it here gives a clear cause.
    """
    key = "screen_epochs" if phase in ("screen", "regime") else "confirmation_epochs"
    if key not in study:
        return
    epochs = int(study[key])
    if epochs < 1:
        raise ValueError("{} must be >= 1 to train and checkpoint, got {}".format(key, epochs))


def _regime_run_ids(plan: dict) -> list:
    """Return (task, cell_id) pairs for the Tier-0 regime pre-screen."""
    tasks = []
    seen = set()
    for candidate in plan.get("candidates", []):
        task = candidate["task"]
        if task not in seen:
            seen.add(task)
            tasks.append(task)
    return [(task, cell_id) for task in tasks for cell_id in REGIME_CELL_IDS]


def _regime_override_for(selected_regime: dict, task: str, fingerprint: dict) -> dict:
    """Return the regime override for a task. Fail closed on any gap.

    A missing selection, an unknown cell id, or a task flagged degenerate must
    raise rather than silently substitute a default regime — otherwise the
    formal screen would build on an unvalidated baseline.
    """
    cells = regime_cells(fingerprint)
    by_task = (selected_regime or {}).get("selected_by_task", {})
    if task in (selected_regime or {}).get("degenerate_tasks", []):
        raise SystemExit("Task {} has a degenerate regime; refuse to screen on it".format(task))
    entry = by_task.get(task)
    if not entry or entry.get("cell_id") not in cells:
        raise SystemExit(
            "No valid selected regime for task {}; run --action select-regime and resolve gaps first".format(task)
        )
    return cells[entry["cell_id"]]


def _run_command(command: list[str], cwd: Path, log_path: Path, dry_run: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("$ {}".format(" ".join(command)))
    if dry_run:
        return 0
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=str(cwd), stdout=handle, stderr=subprocess.STDOUT)
    return int(result.returncode)


def _provenance() -> dict:
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=str(PROJECT_ROOT), text=True).strip()
        except Exception:
            return None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
    }


def _build_run_config(
    base_config: dict,
    candidate: dict,
    fold_root: Path,
    run_id: str,
    support_count: int,
    fingerprint: dict,
) -> tuple[dict, dict]:
    manifest_path = fold_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = deep_merge(base_config, candidate.get("overrides", {}))
    config.setdefault("data", {})["data_root"] = str(fold_root)
    config["data"]["task"] = candidate["task"]
    config["data"]["support_case_ids"] = manifest["selection"]["support_case_ids_ordered"][:support_count]
    config["data"]["k_shot"] = int(support_count)
    config["exp_name"] = "research_runs/{}".format(run_id)
    config.setdefault("training", {})["seed"] = int(config["training"].get("seed", 0))
    policy = derive_policy(fingerprint)
    data_policy = str(candidate.get("data_policy", "explicit"))
    if data_policy in ("fingerprint_patch", "fingerprint_patch_spacing"):
        config["data"]["patch"] = policy["patch"]
        config["data"]["img_size"] = policy["input_size"]
        config.setdefault("model", {})["slice_batch_size"] = policy["slice_batch_size"]
        if policy["use_2_5d"] and config["model"].get("channel_policy", "repeat") == "repeat":
            config["model"]["channel_policy"] = "2_5d"
            config["model"]["neighbor_distance_mm"] = 3.0
    if data_policy in ("fingerprint_spacing", "fingerprint_patch_spacing"):
        config["data"]["target_spacing"] = policy["target_spacing_xyz"]
    return config, policy


def main():
    parser = argparse.ArgumentParser(description="Run DINOv3 multi-organ ablations one GPU job at a time")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--phase", choices=["regime", "screen", "confirm"], required=True)
    parser.add_argument("--selection", help="selected_candidates.json; required for confirm")
    parser.add_argument("--regime", help="selected_regime.json to apply before candidate overrides")
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--candidate-id", action="append", help="Run only this candidate ID; may be repeated")
    parser.add_argument("--cell", action="append", help="Run only this regime cell id; may be repeated")
    parser.add_argument("--fold", type=int, action="append", help="Run only this fold; may be repeated")
    parser.add_argument("--support-count", type=int, action="append", help="Run only this K; may be repeated")
    parser.add_argument("--training-seed", type=int, action="append", help="Run only this stochastic training seed; may be repeated")
    parser.add_argument("--max-runs", type=int, default=0, help="0 means no limit")
    parser.add_argument("--model-path", help="Override model.model_path, used by remote workers")
    parser.add_argument("--benchmark-root", help="Override study.benchmark_root for a local smoke test")
    parser.add_argument("--results-root", help="Override study.results_root for a local smoke test")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-run completed evaluations")
    parser.add_argument("--keep-going", action="store_true", help="Finish independent jobs before reporting failures")
    args = parser.parse_args()

    plan_path = Path(args.plan).resolve()
    plan = _load_yaml(plan_path)
    study = plan["study"]
    _validate_epoch_budget(study, args.phase)
    base_config_path = _resolve(PROJECT_ROOT, study["base_config"])
    base_config = load_config(str(base_config_path), {})
    benchmark_root = Path(args.benchmark_root).resolve() if args.benchmark_root else _resolve(PROJECT_ROOT, study["benchmark_root"])
    results_root = Path(args.results_root).resolve() if args.results_root else _resolve(PROJECT_ROOT, study["results_root"])
    results_root.mkdir(parents=True, exist_ok=True)

    if args.phase == "regime":
        screen_fold = int(study["screening_fold"])
        screen_seed = int((study.get("screening_training_seeds") or [0])[0])
        epochs = int(study.get("screen_epochs", base_config.get("training", {}).get("epochs", 25)))
        wanted_cells = set(args.cell) if args.cell else set(REGIME_CELL_IDS)
        wanted_tasks = set(args.tasks) if args.tasks else None
        completed = 0
        failures = 0
        for task, cell_id in _regime_run_ids(plan):
            if cell_id not in wanted_cells:
                continue
            if wanted_tasks is not None and task not in wanted_tasks:
                continue
            fold_root = benchmark_root / task / "fold_{:02d}".format(screen_fold)
            if not (fold_root / "manifest.json").is_file():
                raise SystemExit("Missing materialized fold: {}".format(fold_root))
            manifest = json.loads((fold_root / "manifest.json").read_text(encoding="utf-8"))
            support_ids = manifest["selection"]["support_case_ids_ordered"][:5]
            fingerprint = build_training_fingerprint(fold_root, support_ids, task)
            run_root = (
                results_root / "regime" / task / cell_id
                / "fold_{:02d}".format(screen_fold) / "k5" / "seed_{}".format(screen_seed)
            )
            evaluation_path = run_root / "evaluation.json"
            if evaluation_path.is_file() and not args.force:
                print("Skipping completed regime: {}/{}".format(task, cell_id))
                continue
            config = deep_merge(base_config, {})
            config.setdefault("data", {})["data_root"] = str(fold_root)
            config["data"]["task"] = task
            config["data"]["support_case_ids"] = support_ids
            config["data"]["k_shot"] = 5
            config = apply_regime(config, regime_cells(fingerprint)[cell_id])
            config["exp_name"] = "research_runs/{}/regime/{}/{}/fold_{:02d}/k5/seed_{}".format(
                study["id"], task, cell_id, screen_fold, screen_seed)
            config.setdefault("training", {})["seed"] = screen_seed
            config["training"]["epochs"] = epochs
            config["training"]["early_stopping"] = dict(study.get("screen_early_stopping", {}))
            config.setdefault("evaluation", {})["surface_tolerance_mm"] = float(study.get("surface_tolerance_mm", 2.0))
            if args.model_path:
                config.setdefault("model", {})["model_path"] = str(Path(args.model_path).resolve())
            config["training"]["experiment_root"] = str(results_root / "artifacts")
            run_root.mkdir(parents=True, exist_ok=True)
            config_path = run_root / "config.yaml"
            with config_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(config, handle, sort_keys=False)
            atomic_json(run_root / "run_manifest.json", {
                "study": study["id"], "phase": "regime", "task": task, "cell_id": cell_id,
                "fold": screen_fold, "support_count": 5, "training_seed": screen_seed,
                "data_root": str(fold_root), "config_path": str(config_path),
                "training_fingerprint": fingerprint["fingerprint_sha256"], "provenance": _provenance(),
            })
            train_rc = _run_command(
                [sys.executable, "scripts/train.py", "--config", str(config_path)],
                PROJECT_ROOT, run_root / "training.log", args.dry_run)
            checkpoint_path = Path(config["training"]["experiment_root"]) / config["exp_name"] / "checkpoints" / "best_model.pth"
            if train_rc or (not checkpoint_path.is_file() and not args.dry_run):
                atomic_json(run_root / "failed.json", {"stage": "train", "returncode": train_rc})
                failures += 1
                if not args.keep_going:
                    raise SystemExit("Regime training failed: {}/{}".format(task, cell_id))
                continue
            eval_rc = _run_command(
                [sys.executable, "scripts/evaluate_model.py", "--config", str(config_path),
                 "--checkpoint", str(checkpoint_path), "--data-root", str(fold_root),
                 "--split", "Val", "--output", str(evaluation_path)],
                PROJECT_ROOT, run_root / "evaluation.log", args.dry_run)
            if eval_rc:
                atomic_json(run_root / "failed.json", {"stage": "evaluation", "returncode": eval_rc})
                failures += 1
                if not args.keep_going:
                    raise SystemExit("Regime evaluation failed: {}/{}".format(task, cell_id))
            completed += 1
            if args.max_runs and completed >= args.max_runs:
                return
        if failures:
            raise SystemExit("{} regime job(s) failed".format(failures))
        return

    selected = None
    if args.phase == "confirm":
        if not args.selection:
            raise SystemExit("--selection is required for independent confirmation runs")
        selected = _selected_candidates(Path(args.selection).resolve())
    selected_regime = None
    if args.regime:
        selected_regime = json.loads(Path(args.regime).resolve().read_text(encoding="utf-8"))
    elif args.phase in ("screen", "confirm"):
        # Fail closed: the factor screen/confirmation must build on a validated
        # Tier-0 regime, never silently fall back to the collapse-prone baseline.
        raise SystemExit(
            "--regime selected_regime.json is required for {}; run Tier-0 + select-regime first".format(args.phase)
        )
    if selected_regime is not None and selected_regime.get("degenerate_tasks"):
        raise SystemExit(
            "Refusing to {}: degenerate regime tasks present: {}".format(
                args.phase, selected_regime["degenerate_tasks"])
        )
    candidates = list(plan.get("candidates", []))
    if args.tasks:
        requested = set(args.tasks)
        candidates = [candidate for candidate in candidates if candidate["task"] in requested]
    if args.candidate_id:
        requested_ids = set(args.candidate_id)
        candidates = [candidate for candidate in candidates if candidate["id"] in requested_ids]
    if selected is not None:
        # Confirmation must include the fixed frozen reference even when a
        # different candidate wins screening. Otherwise no independent effect
        # size can be reported for the claimed improvement.
        selected_or_reference = selected | {
            candidate["id"] for candidate in candidates if bool(candidate.get("is_reference", False))
        }
        candidates = [candidate for candidate in candidates if candidate["id"] in selected_or_reference]
    if not candidates:
        raise SystemExit("No candidates match the requested stage")

    folds = [int(study["screening_fold"])] if args.phase == "screen" else [int(v) for v in study["confirmation_folds"]]
    shots = [5] if args.phase == "screen" else [int(v) for v in study["shot_counts"]]
    if args.fold:
        folds = [fold for fold in folds if fold in set(args.fold)]
    if args.support_count:
        shots = [shot for shot in shots if shot in set(args.support_count)]
    if not folds or not shots:
        raise SystemExit("Requested fold or support count is outside the selected study phase")
    seed_key = "screening_training_seeds" if args.phase == "screen" else "confirmation_training_seeds"
    training_seeds = [int(value) for value in study.get(seed_key, [int(base_config.get("training", {}).get("seed", 0))])]
    if args.training_seed:
        requested_seeds = set(args.training_seed)
        training_seeds = [seed for seed in training_seeds if seed in requested_seeds]
    if not training_seeds:
        raise SystemExit("Requested training seed is outside the selected study phase")
    completed = 0
    failures = 0
    for candidate in candidates:
        for fold in folds:
            fold_root = benchmark_root / candidate["task"] / "fold_{:02d}".format(fold)
            if not (fold_root / "manifest.json").is_file():
                raise SystemExit("Missing materialized fold: {}".format(fold_root))
            for support_count in shots:
                for training_seed in training_seeds:
                    run_id = "{}/{}/{}/fold_{:02d}/k{}/seed_{}".format(
                        study["id"], args.phase, candidate["id"], fold, support_count, training_seed
                    )
                    run_root = (
                        results_root / args.phase / candidate["id"] / "fold_{:02d}".format(fold)
                        / "k{}".format(support_count) / "seed_{}".format(training_seed)
                    )
                    evaluation_path = run_root / "evaluation.json"
                    if evaluation_path.is_file() and not args.force:
                        print("Skipping completed: {}".format(run_id))
                        continue
                    manifest = json.loads((fold_root / "manifest.json").read_text(encoding="utf-8"))
                    support_case_ids = manifest["selection"]["support_case_ids_ordered"][:support_count]
                    fingerprint_path = (
                        results_root / "_fingerprints" / candidate["task"] / "fold_{:02d}".format(fold) / "k{}.json".format(support_count)
                    )
                    if fingerprint_path.is_file():
                        fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))
                    else:
                        fingerprint = build_training_fingerprint(fold_root, support_case_ids, candidate["task"])
                        atomic_json(fingerprint_path, fingerprint)
                    # Apply the task's selected sampling+loss regime before the
                    # candidate's single-factor override, so the candidate is a
                    # controlled change from a trainable baseline.
                    regime_base = base_config
                    if selected_regime is not None:
                        regime_base = apply_regime(
                            base_config, _regime_override_for(selected_regime, candidate["task"], fingerprint)
                        )
                    config, policy = _build_run_config(
                        regime_base, candidate, fold_root, run_id, support_count, fingerprint
                    )
                    config["training"]["seed"] = int(training_seed)
                    config.setdefault("evaluation", {})["surface_tolerance_mm"] = float(study.get("surface_tolerance_mm", 2.0))
                    if args.phase == "screen":
                        config["training"]["epochs"] = int(study.get("screen_epochs", config["training"].get("epochs", 25)))
                        config["training"]["early_stopping"] = dict(study.get("screen_early_stopping", {}))
                    else:
                        config["training"]["epochs"] = int(study.get("confirmation_epochs", config["training"].get("epochs", 80)))
                        config["training"]["early_stopping"] = dict(study.get("confirmation_early_stopping", {}))
                    if args.model_path:
                        config.setdefault("model", {})["model_path"] = str(Path(args.model_path).resolve())
                    config.setdefault("training", {})["experiment_root"] = str(results_root / "artifacts")
                    run_root.mkdir(parents=True, exist_ok=True)
                    config_path = run_root / "config.yaml"
                    with config_path.open("w", encoding="utf-8") as handle:
                        yaml.safe_dump(config, handle, sort_keys=False)
                    atomic_json(run_root / "training_fingerprint.json", fingerprint)
                    atomic_json(run_root / "derived_data_policy.json", policy)
                    atomic_json(
                        run_root / "run_manifest.json",
                        {
                            "study": study["id"],
                            "phase": args.phase,
                            "candidate_id": candidate["id"],
                            "task": candidate["task"],
                            "changed_factor": candidate.get("changed_factor", "unspecified"),
                            "confirmation_role": (
                                "reference" if bool(candidate.get("is_reference", False)) else "screen_selected"
                            ) if args.phase == "confirm" else "screen_candidate",
                            "fold": fold,
                            "support_count": support_count,
                            "training_seed": int(training_seed),
                            "data_root": str(fold_root),
                            "config_path": str(config_path),
                            "training_fingerprint": fingerprint["fingerprint_sha256"],
                            "data_policy": candidate.get("data_policy", "explicit"),
                            "derived_policy": policy["policy_sha256"],
                            "provenance": _provenance(),
                        },
                    )
                    train_returncode = _run_command(
                        [sys.executable, "scripts/train.py", "--config", str(config_path)],
                        PROJECT_ROOT,
                        run_root / "training.log",
                        args.dry_run,
                    )
                    checkpoint_path = Path(config["training"]["experiment_root"]) / config["exp_name"] / "checkpoints" / "best_model.pth"
                    if train_returncode or (not checkpoint_path.is_file() and not args.dry_run):
                        atomic_json(run_root / "failed.json", {"stage": "train", "returncode": train_returncode})
                        print("Training failed: {}".format(run_id))
                        failures += 1
                        if not args.keep_going:
                            raise SystemExit("Training failed: {}".format(run_id))
                        continue
                    evaluate_returncode = _run_command(
                        [
                            sys.executable,
                            "scripts/evaluate_model.py",
                            "--config", str(config_path),
                            "--checkpoint", str(checkpoint_path),
                            "--data-root", str(fold_root),
                            "--split", "Val",
                            "--output", str(evaluation_path),
                        ],
                        PROJECT_ROOT,
                        run_root / "evaluation.log",
                        args.dry_run,
                    )
                    if evaluate_returncode:
                        atomic_json(run_root / "failed.json", {"stage": "evaluation", "returncode": evaluate_returncode})
                        failures += 1
                        if not args.keep_going:
                            raise SystemExit("Evaluation failed: {}".format(run_id))
                    completed += 1
                    if args.max_runs and completed >= args.max_runs:
                        return
    if failures:
        raise SystemExit("{} experiment job(s) failed; inspect failed.json before confirmation".format(failures))


if __name__ == "__main__":
    main()
