#!/usr/bin/env python3
"""Turn the declarative study matrix into an auditable remote-compute budget."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml


def _require_nonempty(study: dict, key: str) -> list:
    values = list(study.get(key, []))
    if not values:
        raise ValueError("study.{} must be non-empty".format(key))
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate exact DINOv3 study job counts before cloud submission")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--minutes-per-gpu-job", type=float, default=0.0, help="Measured median from a pilot; 0 means count only")
    parser.add_argument("--parallel-gpus", type=int, default=1, help="A10 jobs allowed concurrently after Volume stress validation")
    parser.add_argument("--selection", help="selected_candidates.json after screening; produces an exact post-screen count")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.minutes_per_gpu_job < 0.0 or args.parallel_gpus < 1:
        raise SystemExit("--minutes-per-gpu-job must be non-negative and --parallel-gpus at least one")
    payload = yaml.safe_load(Path(args.plan).read_text(encoding="utf-8")) or {}
    study = payload["study"]
    candidates = list(payload.get("candidates", []))
    by_task = Counter(candidate["task"] for candidate in candidates)
    reference_ids = {candidate["id"] for candidate in candidates if bool(candidate.get("is_reference", False))}
    screen_seeds = _require_nonempty(study, "screening_training_seeds")
    confirm_seeds = _require_nonempty(study, "confirmation_training_seeds")
    tasks = sorted(by_task)
    screening = len(candidates) * len(screen_seeds)
    selected_ids = None
    if args.selection:
        selected_ids = set(json.loads(Path(args.selection).read_text(encoding="utf-8")).get("selected_candidate_ids", []))
        unknown = selected_ids - {candidate["id"] for candidate in candidates}
        if unknown:
            raise ValueError("Selection references unknown candidates: {}".format(", ".join(sorted(unknown))))
    min_confirmation_candidates = len(tasks)
    max_confirmation_candidates = len(tasks) + len(reference_ids)
    # A winning candidate can itself be the reference. Before screening, the
    # exact overlap is unknown, hence the [min, max] count. After selection the
    # union gives an exact number.
    confirmation_candidate_count = len(selected_ids | reference_ids) if selected_ids is not None else None
    per_candidate_confirmation = len(study["confirmation_folds"]) * len(study["shot_counts"]) * len(confirm_seeds)
    confirmation_min = min_confirmation_candidates * per_candidate_confirmation
    confirmation_max = max_confirmation_candidates * per_candidate_confirmation
    confirmation = confirmation_candidate_count * per_candidate_confirmation if confirmation_candidate_count is not None else None
    inference_modes = len(tasks) * len(study["confirmation_folds"])
    cascade = len(study["confirmation_folds"]) * len(study["shot_counts"]) * len(confirm_seeds)
    total_min = 1 + screening + confirmation_min + inference_modes + cascade
    total_max = 1 + screening + confirmation_max + inference_modes + cascade
    total = 1 + screening + confirmation + inference_modes + cascade if confirmation is not None else None
    estimate = None
    if args.minutes_per_gpu_job:
        count_for_estimate = total if total is not None else total_max
        gpu_hours = count_for_estimate * args.minutes_per_gpu_job / 60.0
        wall_hours = gpu_hours / float(args.parallel_gpus)
        estimate = {
            "assumed_minutes_per_gpu_job": float(args.minutes_per_gpu_job),
            "parallel_gpus": int(args.parallel_gpus),
            "estimated_gpu_hours": gpu_hours,
            "estimated_wall_hours": wall_hours,
            "note": "Estimate uses {} job count. Use the median completed A10 pilot duration, not a local MPS duration.".format(
                "exact post-screen" if total is not None else "worst-case pre-screen"
            ),
        }
    result = {
        "schema_version": "dinov3_medical_compute_plan.v1",
        "study": study["id"],
        "candidates_by_task": dict(sorted(by_task.items())),
        "jobs": {
            "peft_gradient_preflight": 1,
            "screening_train_evaluate": screening,
            "confirmation_train_evaluate": confirmation if confirmation is not None else {
                "minimum": confirmation_min,
                "maximum": confirmation_max,
            },
            "fixed_checkpoint_inference_mode_evaluations": inference_modes,
            "adrenal_two_stage_train_evaluate": cascade,
            "total_gpu_functions": total if total is not None else {"minimum": total_min, "maximum": total_max},
            "reference_candidates": sorted(reference_ids),
        },
        "estimate": estimate,
        "execution_constraints": [
            "Screening must complete and persist selected_candidates.json before confirmation starts.",
            "Each GPU function owns one seed/fold/K result directory and writes failed.json on error.",
            "Do not increase GPU parallelism until a small concurrent Volume commit stress test has passed.",
            "Modal Function timeout is at most 24 hours per invocation; no single controller may be assumed to run indefinitely.",
        ],
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
