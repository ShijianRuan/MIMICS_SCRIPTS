#!/usr/bin/env python3
"""Choose one sampling+loss regime per task from the Tier-0 pre-screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _score(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    dice = payload.get("mean_dice")
    dice = -1.0 if dice is None else float(dice)
    hd95 = payload.get("mean_hd95_mm")
    return dice, float("inf") if hd95 is None else float(hd95)


def select_regime(results_root: Path, output: Path, degenerate_threshold: float = 0.05) -> dict:
    root = Path(results_root).resolve() / "regime"
    by_task = {}
    for path in sorted(root.glob("*/*/fold_00/k5/seed_*/evaluation.json")):
        manifest = path.parent / "run_manifest.json"
        if not manifest.is_file():
            continue
        run = json.loads(manifest.read_text(encoding="utf-8"))
        task, cell = run["task"], run["cell_id"]
        by_task.setdefault(task, {}).setdefault(cell, []).append((_score(path), str(path)))
    selected = {}
    degenerate_tasks = []
    for task, cells in by_task.items():
        rows = []
        for cell_id, evals in cells.items():
            mean_dice = sum(s[0] for s, _ in evals) / len(evals)
            mean_hd95 = sum(s[1] for s, _ in evals) / len(evals)
            rows.append((cell_id, mean_dice, mean_hd95, [p for _, p in evals]))
        cell_id, mean_dice, mean_hd95, paths = max(rows, key=lambda r: (r[1], -r[2], r[0]))
        is_degenerate = mean_dice < float(degenerate_threshold)
        if is_degenerate:
            degenerate_tasks.append(task)
        selected[task] = {"cell_id": cell_id, "mean_dice": mean_dice,
                          "mean_hd95_mm": mean_hd95, "degenerate": is_degenerate,
                          "regime_evaluations": paths}
    payload = {
        "schema_version": "dinov3_medical_regime_selection.v1",
        "selection_rule": "max mean Dice per task; tie-break lower HD95; then cell id",
        "degenerate_threshold": float(degenerate_threshold),
        "degenerate_tasks": sorted(degenerate_tasks),
        "selected_by_task": selected,
    }
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for task, entry in sorted(selected.items()):
        flag = "  !! DEGENERATE" if entry["degenerate"] else ""
        print("{} -> {} (Dice {:.4f}){}".format(task, entry["cell_id"], entry["mean_dice"], flag))
    if degenerate_tasks:
        print("WARNING: degenerate tasks (no regime cleared Dice>{}): {}".format(
            degenerate_threshold, ", ".join(sorted(degenerate_tasks))))
    return payload


def main():
    parser = argparse.ArgumentParser(description="Select per-task sampling+loss regime")
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--degenerate-threshold", type=float, default=0.05)
    args = parser.parse_args()
    select_regime(Path(args.results_root), Path(args.output), args.degenerate_threshold)


if __name__ == "__main__":
    main()
