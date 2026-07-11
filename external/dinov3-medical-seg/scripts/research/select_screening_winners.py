#!/usr/bin/env python3
"""Choose one candidate per task from the screening fold before confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _score(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    dice = float(payload.get("mean_dice", -1.0))
    hd95 = payload.get("mean_hd95_mm")
    return dice, float("inf") if hd95 is None else float(hd95)


def main():
    parser = argparse.ArgumentParser(description="Select screening winners without touching confirmation folds")
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.results_root).resolve() / "screen"
    by_task = {}
    for path in sorted(root.glob("*/fold_00/k5/seed_*/evaluation.json")):
        run_manifest = path.parent / "run_manifest.json"
        if not run_manifest.is_file():
            continue
        run = json.loads(run_manifest.read_text(encoding="utf-8"))
        task = run["task"]
        by_task.setdefault(task, {}).setdefault(run["candidate_id"], []).append((_score(path), str(path)))
    selected = {}
    for task, candidates in by_task.items():
        rows = []
        for candidate_id, evaluations in candidates.items():
            mean_dice = sum(score[0] for score, _ in evaluations) / len(evaluations)
            mean_hd95 = sum(score[1] for score, _ in evaluations) / len(evaluations)
            rows.append((candidate_id, (mean_dice, mean_hd95), [path for _, path in evaluations]))
        candidate_id, score, paths = max(rows, key=lambda row: (row[1][0], -row[1][1], row[0]))
        selected[task] = {
            "candidate_id": candidate_id,
            "mean_dice": score[0],
            "mean_hd95_mm": score[1],
            "screen_evaluations": paths,
        }
    payload = {
        "schema_version": "dinov3_medical_screen_selection.v1",
        "selection_rule": "max mean Dice across screening seeds; tie-break lower mean HD95; then candidate id",
        "selected_by_task": selected,
        "selected_candidate_ids": [entry["candidate_id"] for _, entry in sorted(selected.items())],
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for task, entry in sorted(selected.items()):
        print("{} -> {} (Dice {:.4f})".format(task, entry["candidate_id"], entry["mean_dice"]))


if __name__ == "__main__":
    main()
