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


def _secondary_key(rule: str):
    """Return a within-band tie-breaker key (higher is better) for one rule.

    Used only among Dice-equivalent candidates, so Dice itself is NOT in the key
    (that is what makes the earlier lexicographic ``(dice, -hd95)`` degenerate:
    any Dice difference dominated HD95). Here HD95 / surface Dice genuinely
    decide when Dice cannot separate the candidates.
    """
    if rule == "boundary":
        def key(cell):
            sd = cell["surface_dice"] if _finite(cell["surface_dice"]) else 0.0
            hd = cell["hd95_mm"] if _finite(cell["hd95_mm"]) else float("inf")
            return (sd, -hd)
        return key

    # hd95_gate (large compact): prefer the lower HD95 within the Dice band.
    def key(cell):
        hd = cell["hd95_mm"] if _finite(cell["hd95_mm"]) else float("inf")
        return (-hd,)
    return key


def _select_task(rule: str, cells: dict) -> tuple:
    """Pick one cell for a task and report whether the Dice band was decisive.

    Returns (best_id, ranking, needs_more_seeds). Detection tasks (tiny targets)
    are dominated by empty-rate + recall directly, since raw Dice overstates a
    cell that fires far-field or misses the organ entirely.
    """
    if rule == "detection":
        def det_key(item):
            cell = item[1]
            empty = cell["empty_prediction_rate"]
            empty = 1.0 if not _finite(empty) else float(empty)
            recall = cell["recall"] if _finite(cell["recall"]) else 0.0
            return (-empty, recall, cell["dice"])
        ranked = sorted(cells.items(), key=det_key, reverse=True)
        best_id = ranked[0][0]
        # Near-tie on the composite detection score of the top two.
        needs = False
        if len(ranked) > 1:
            a, b = ranked[0][1], ranked[1][1]
            if abs((a["recall"] or 0.0) - (b["recall"] or 0.0)) < _DICE_TIE_MARGIN:
                needs = True
        return best_id, [cid for cid, _ in ranked], needs

    # Dice-primary tasks: max Dice, then a tolerance band, then secondary metric.
    top_dice = max(c["dice"] for c in cells.values())
    band = {cid: c for cid, c in cells.items() if top_dice - c["dice"] <= _DICE_TIE_MARGIN}
    secondary = _secondary_key(rule)
    band_ranked = sorted(band.items(), key=lambda kv: secondary(kv[1]), reverse=True)
    best_id = band_ranked[0][0]
    # Full ranking for the record: band members (by secondary) then the rest (by Dice).
    outside = sorted(
        ((cid, c) for cid, c in cells.items() if cid not in band),
        key=lambda kv: kv[1]["dice"], reverse=True,
    )
    ranking = [cid for cid, _ in band_ranked] + [cid for cid, _ in outside]
    # More than one Dice-equivalent candidate => the secondary metric decided it,
    # so the result is not Dice-separable and needs more seeds to confirm.
    needs = len(band) > 1
    return best_id, ranking, needs


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
        cells = {cell: _load_cell(present[task][cell]) for cell in expected_cells}
        best_id, ranking_ids, needs_more_seeds = _select_task(rule, cells)
        best = cells[best_id]
        is_degenerate = best["dice"] < float(degenerate_threshold)
        if is_degenerate:
            degenerate_tasks.append(task)
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
            "ranking": [
                {"cell_id": cid, "mean_dice": cells[cid]["dice"], "mean_hd95_mm": cells[cid]["hd95_mm"]}
                for cid in ranking_ids
            ],
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
