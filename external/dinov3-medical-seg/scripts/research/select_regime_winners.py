#!/usr/bin/env python3
"""Choose one sampling+loss regime per task from the Tier-0 pre-screen.

The selector fails closed: unless the full expected task x cell matrix is
present with finite metrics, it refuses to emit a selection. Per-organ rules
combine Dice with HD95 / surface Dice / empty-prediction rate / recall so a
high-Dice-but-fragmented or empty-prone cell cannot silently win.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ALL_CELLS = ("full_dice_ce", "full_dice_focal", "patch_dice_ce", "patch_dice_focal")

# Near-tie margin on Dice: winners this close on Dice get flagged for more seeds.
_DICE_TIE_MARGIN = 0.02

# Per-organ metric emphasis. "hd95_gate" tasks weight boundary quality; "empty"
# tasks (tiny structures) weight detection over raw overlap.
_TASK_RULE = {
    "brain": "hd95_gate",
    "liver": "hd95_gate",
    "aorta": "boundary",
    "scapula_left": "boundary",
    "adrenal_gland_right": "detection",
}


def _finite(value):
    return value is not None and isinstance(value, (int, float)) and math.isfinite(float(value))


def _load_cell(cell_dir: Path) -> dict:
    """Return a cell's metrics, or raise if failed / missing / non-finite Dice."""
    if (cell_dir / "failed.json").is_file():
        raise SystemExit("Regime cell failed: {}".format(cell_dir))
    eval_path = cell_dir / "evaluation.json"
    if not eval_path.is_file():
        raise SystemExit("Regime cell missing evaluation.json: {}".format(cell_dir))
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    dice = payload.get("mean_dice")
    if not _finite(dice):
        raise SystemExit("Regime cell has non-finite mean_dice: {}".format(cell_dir))
    return {
        "dice": float(dice),
        "hd95_mm": payload.get("mean_hd95_mm"),
        "surface_dice": payload.get("mean_surface_dice"),
        "recall": payload.get("mean_recall"),
        "empty_prediction_rate": payload.get("empty_prediction_rate"),
        "path": str(eval_path),
    }


def _rank_key(rule: str):
    """Return a sort key (higher is better) for one task's rule."""
    if rule == "detection":
        # Tiny structures: penalise empty predictions, then reward recall, then Dice.
        def key(cell):
            empty = cell["empty_prediction_rate"]
            empty = 1.0 if not _finite(empty) else float(empty)
            recall = cell["recall"] if _finite(cell["recall"]) else 0.0
            return (-empty, recall, cell["dice"])
        return key
    if rule == "boundary":
        # Elongated/thin structures: Dice, then surface Dice, then lower HD95.
        def key(cell):
            sd = cell["surface_dice"] if _finite(cell["surface_dice"]) else 0.0
            hd = cell["hd95_mm"] if _finite(cell["hd95_mm"]) else float("inf")
            return (cell["dice"], sd, -hd)
        return key

    # hd95_gate (large compact): Dice, then lower HD95.
    def key(cell):
        hd = cell["hd95_mm"] if _finite(cell["hd95_mm"]) else float("inf")
        return (cell["dice"], -hd)
    return key


def select_regime(results_root: Path, output: Path, degenerate_threshold: float = 0.05,
                  expected_tasks=None, expected_cells=ALL_CELLS) -> dict:
    root = Path(results_root).resolve() / "regime"
    expected_cells = tuple(expected_cells)
    # Discover which tasks/cells exist on disk.
    present = {}
    for cell_dir in sorted(root.glob("*/*/fold_00/k5/seed_*")):
        manifest = cell_dir / "run_manifest.json"
        if not manifest.is_file():
            continue
        run = json.loads(manifest.read_text(encoding="utf-8"))
        present.setdefault(run["task"], {})[run["cell_id"]] = cell_dir

    tasks = list(expected_tasks) if expected_tasks is not None else sorted(present)
    # Completeness gate: every expected task must have every expected cell.
    missing = []
    for task in tasks:
        for cell in expected_cells:
            if cell not in present.get(task, {}):
                missing.append("{}/{}".format(task, cell))
    if missing:
        raise SystemExit(
            "Incomplete regime matrix; missing {} cell(s): {}".format(len(missing), ", ".join(missing))
        )

    selected = {}
    degenerate_tasks = []
    needs_more_seeds_tasks = []
    for task in tasks:
        rule = _TASK_RULE.get(task, "hd95_gate")
        key = _rank_key(rule)
        cells = {cell: _load_cell(present[task][cell]) for cell in expected_cells}
        ranked = sorted(cells.items(), key=lambda kv: key(kv[1]), reverse=True)
        (best_id, best), *rest = ranked
        is_degenerate = best["dice"] < float(degenerate_threshold)
        if is_degenerate:
            degenerate_tasks.append(task)
        # Near-tie on Dice among the top two -> not statistically separable yet.
        needs_more_seeds = False
        if rest:
            second = rest[0][1]
            if abs(best["dice"] - second["dice"]) < _DICE_TIE_MARGIN:
                needs_more_seeds = True
        if needs_more_seeds:
            needs_more_seeds_tasks.append(task)
        selected[task] = {
            "cell_id": best_id,
            "rule": rule,
            "mean_dice": best["dice"],
            "mean_hd95_mm": best["hd95_mm"],
            "mean_recall": best["recall"],
            "empty_prediction_rate": best["empty_prediction_rate"],
            "degenerate": is_degenerate,
            "needs_more_seeds": needs_more_seeds,
            "ranking": [{"cell_id": cid, "mean_dice": c["dice"], "mean_hd95_mm": c["hd95_mm"]} for cid, c in ranked],
        }
    payload = {
        "schema_version": "dinov3_medical_regime_selection.v2",
        "selection_rule": "per-organ: brain/liver Dice+HD95; aorta/scapula Dice+surface+HD95; adrenal empty-rate+recall+Dice",
        "dice_tie_margin": _DICE_TIE_MARGIN,
        "degenerate_threshold": float(degenerate_threshold),
        "expected_cells": list(expected_cells),
        "degenerate_tasks": sorted(degenerate_tasks),
        "needs_more_seeds_tasks": sorted(needs_more_seeds_tasks),
        "selected_by_task": selected,
    }
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for task, entry in sorted(selected.items()):
        flags = []
        if entry["degenerate"]:
            flags.append("DEGENERATE")
        if entry["needs_more_seeds"]:
            flags.append("needs_more_seeds")
        suffix = ("  !! " + ", ".join(flags)) if flags else ""
        print("{} -> {} (Dice {:.4f}, HD95 {}){}".format(
            task, entry["cell_id"], entry["mean_dice"],
            "NA" if not _finite(entry["mean_hd95_mm"]) else "{:.1f}".format(entry["mean_hd95_mm"]), suffix))
    if degenerate_tasks:
        print("WARNING: degenerate tasks (no regime cleared Dice>{}): {}".format(
            degenerate_threshold, ", ".join(sorted(degenerate_tasks))))
    return payload


def main():
    parser = argparse.ArgumentParser(description="Select per-task sampling+loss regime")
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--degenerate-threshold", type=float, default=0.05)
    parser.add_argument("--expected-task", action="append", help="Require this task in the matrix; may repeat")
    args = parser.parse_args()
    select_regime(Path(args.results_root), Path(args.output), args.degenerate_threshold,
                  expected_tasks=args.expected_task)


if __name__ == "__main__":
    main()
