#!/usr/bin/env python3
"""Select screen winners only from a complete, valid candidate matrix."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml


_DICE_TOLERANCE = 0.02
_TASK_RULE = {
    "brain": "hd95_gate",
    "liver": "hd95_gate",
    "aorta": "boundary",
    "scapula_left": "boundary",
    "adrenal_gland_right": "detection",
}


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _load_evaluation(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not _finite(payload.get("mean_dice")):
        raise SystemExit("Screen evaluation has non-finite mean_dice: {}".format(path))
    return {
        "dice": float(payload["mean_dice"]),
        "hd95_mm": payload.get("mean_hd95_mm"),
        "surface_dice": payload.get("mean_surface_dice"),
        "recall": payload.get("mean_recall"),
        "empty_prediction_rate": payload.get("empty_prediction_rate"),
        "path": str(path),
    }


def _aggregate(rows: list[dict]) -> dict:
    result = {"evaluations": [row["path"] for row in rows]}
    for key in ("dice", "hd95_mm", "surface_dice", "recall", "empty_prediction_rate"):
        values = [float(row[key]) for row in rows if _finite(row.get(key))]
        result[key] = sum(values) / len(values) if values else None
    return result


def _secondary_key(rule: str, row: dict):
    hd95 = float(row["hd95_mm"]) if _finite(row.get("hd95_mm")) else float("inf")
    surface = float(row["surface_dice"]) if _finite(row.get("surface_dice")) else 0.0
    recall = float(row["recall"]) if _finite(row.get("recall")) else 0.0
    empty = float(row["empty_prediction_rate"]) if _finite(row.get("empty_prediction_rate")) else 1.0
    if rule == "detection":
        return (-empty, recall, row["dice"], -hd95)
    if rule == "boundary":
        return (surface, -hd95, row["dice"])
    return (-hd95, surface, row["dice"])


def select_screening(results_root: Path, output: Path, plan: dict, regime: dict | None = None) -> dict:
    """Validate the expected screen matrix and persist task-aware winners.

    Dice values within ``_DICE_TOLERANCE`` are treated as practically tied;
    organ-specific boundary/detection metrics then choose among them. A stale
    ``failed.json`` is an invalid ambiguous state even when evaluation exists.
    """
    root = Path(results_root).resolve() / "screen"
    study = plan["study"]
    degenerate = set((regime or {}).get("degenerate_tasks", []))
    expected_seeds = {int(seed) for seed in study.get("screening_training_seeds", [])}
    expected = {
        candidate["id"]: candidate["task"]
        for candidate in plan.get("candidates", [])
        if candidate["task"] not in degenerate
    }
    if not expected_seeds:
        raise SystemExit("Study must define screening_training_seeds")

    collected = {}
    problems = []
    for candidate_id, task in expected.items():
        candidate_root = root / candidate_id / "fold_{:02d}".format(int(study["screening_fold"])) / "k5"
        rows = []
        for seed in sorted(expected_seeds):
            run_root = candidate_root / "seed_{}".format(seed)
            evaluation = run_root / "evaluation.json"
            failed = run_root / "failed.json"
            manifest = run_root / "run_manifest.json"
            if failed.is_file():
                problems.append("{} seed {} has failed.json".format(candidate_id, seed))
                continue
            if not evaluation.is_file() or not manifest.is_file():
                problems.append("{} seed {} is incomplete".format(candidate_id, seed))
                continue
            run = json.loads(manifest.read_text(encoding="utf-8"))
            if run.get("candidate_id") != candidate_id or run.get("task") != task:
                problems.append("{} seed {} manifest mismatch".format(candidate_id, seed))
                continue
            rows.append(_load_evaluation(evaluation))
        if len(rows) == len(expected_seeds):
            collected[candidate_id] = _aggregate(rows)
    if problems:
        raise SystemExit("Incomplete or failed screen matrix: {}".format("; ".join(problems)))

    guardrails = dict(plan.get("screen_guardrails", {}))
    minimum_dice = dict(guardrails.get("minimum_dice_by_task", {}))
    maximum_hd95 = dict(guardrails.get("maximum_hd95_mm_by_task", {}))
    selected = {}
    weak_tasks = []
    tasks = sorted(set(expected.values()))
    for task in tasks:
        candidates = [(cid, collected[cid]) for cid, candidate_task in expected.items() if candidate_task == task]
        best_dice = max(row["dice"] for _, row in candidates)
        band = [(cid, row) for cid, row in candidates if row["dice"] >= best_dice - _DICE_TOLERANCE]
        rule = _TASK_RULE.get(task, "hd95_gate")
        candidate_id, winner = max(band, key=lambda item: (_secondary_key(rule, item[1]), item[0]))
        weak_reasons = []
        if task in minimum_dice and winner["dice"] < float(minimum_dice[task]):
            weak_reasons.append("Dice {:.4f} < {:.4f}".format(winner["dice"], float(minimum_dice[task])))
        if task in maximum_hd95:
            if not _finite(winner.get("hd95_mm")) or winner["hd95_mm"] > float(maximum_hd95[task]):
                weak_reasons.append("HD95 {} > {:.1f} mm".format(winner.get("hd95_mm"), float(maximum_hd95[task])))
        if weak_reasons:
            weak_tasks.append(task)
        selected[task] = {
            "candidate_id": candidate_id,
            "rule": rule,
            "mean_dice": winner["dice"],
            "mean_hd95_mm": winner["hd95_mm"],
            "mean_surface_dice": winner["surface_dice"],
            "weak": bool(weak_reasons),
            "weak_reasons": weak_reasons,
            "screen_evaluations": winner["evaluations"],
            "dice_tolerance_candidates": [cid for cid, _ in band],
        }
    payload = {
        "schema_version": "dinov3_medical_screen_selection.v2",
        "selection_rule": "complete matrix; Dice tolerance band then task-aware secondary metrics",
        "dice_tolerance": _DICE_TOLERANCE,
        "excluded_degenerate_tasks": sorted(degenerate),
        "weak_tasks": sorted(weak_tasks),
        "selected_by_task": selected,
        "selected_candidate_ids": [entry["candidate_id"] for _, entry in sorted(selected.items())],
    }
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description="Select complete screening winners without touching confirmation folds")
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--regime")
    args = parser.parse_args()
    plan = yaml.safe_load(Path(args.plan).read_text(encoding="utf-8")) or {}
    regime = json.loads(Path(args.regime).read_text(encoding="utf-8")) if args.regime else None
    payload = select_screening(Path(args.results_root), Path(args.output), plan, regime)
    for task, entry in sorted(payload["selected_by_task"].items()):
        suffix = " WEAK" if entry["weak"] else ""
        print("{} -> {} (Dice {:.4f}, HD95 {}){}".format(
            task, entry["candidate_id"], entry["mean_dice"], entry["mean_hd95_mm"], suffix))


if __name__ == "__main__":
    main()
