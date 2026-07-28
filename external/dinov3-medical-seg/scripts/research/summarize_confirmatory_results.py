#!/usr/bin/env python3
"""Summarize confirmation folds with fold-level bootstrap intervals."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _bootstrap_mean(values, seed=20260711, samples=10000):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None, None, None
    if len(values) == 1:
        return float(values[0]), float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main():
    parser = argparse.ArgumentParser(description="Aggregate independent confirmation-fold DINOv3 results")
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.results_root).resolve() / "confirm"
    grouped = defaultdict(list)
    for evaluation_path in sorted(root.glob("*/fold_*/k*/seed_*/evaluation.json")):
        run_manifest = evaluation_path.parent / "run_manifest.json"
        if not run_manifest.is_file():
            continue
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        run = json.loads(run_manifest.read_text(encoding="utf-8"))
        key = (run["task"], run["candidate_id"], int(run["support_count"]))
        grouped[key].append((run, evaluation, evaluation_path))
    rows = []
    for (task, candidate_id, support_count), records in sorted(grouped.items()):
        by_fold = defaultdict(list)
        for run, evaluation, path in records:
            by_fold[int(run["fold"])].append((run, evaluation, path))
        # Seeds share one patient fold, so they are averaged before the
        # fold-level bootstrap. This avoids pretending three random seeds are
        # three independent patient cohorts.
        def fold_metric(key):
            return [
                float(np.mean([float(evaluation[key]) for _, evaluation, _ in entries if evaluation.get(key) is not None]))
                for _, entries in sorted(by_fold.items())
                if any(evaluation.get(key) is not None for _, evaluation, _ in entries)
            ]

        dice_by_fold = fold_metric("mean_dice")
        hd95_by_fold = fold_metric("mean_hd95_mm")
        assd_by_fold = fold_metric("mean_assd_mm")
        surface_dice_by_fold = fold_metric("mean_surface_dice")
        lesion_f1_by_fold = fold_metric("mean_lesion_f1")
        elapsed_by_fold = fold_metric("mean_elapsed_seconds")
        mean_dice, dice_low, dice_high = _bootstrap_mean(dice_by_fold)
        mean_hd95, hd95_low, hd95_high = _bootstrap_mean(hd95_by_fold)
        mean_assd, assd_low, assd_high = _bootstrap_mean(assd_by_fold)
        mean_surface_dice, surface_low, surface_high = _bootstrap_mean(surface_dice_by_fold)
        mean_lesion_f1, lesion_low, lesion_high = _bootstrap_mean(lesion_f1_by_fold)
        mean_elapsed, elapsed_low, elapsed_high = _bootstrap_mean(elapsed_by_fold)
        rows.append(
            {
                "task": task,
                "candidate_id": candidate_id,
                "support_count": support_count,
                "n_folds": len(by_fold),
                "n_training_seeds": len({int(run["training_seed"]) for run, _, _ in records}),
                "mean_dice": mean_dice,
                "dice_ci95": [dice_low, dice_high],
                "mean_hd95_mm": mean_hd95,
                "hd95_ci95_mm": [hd95_low, hd95_high],
                "mean_assd_mm": mean_assd,
                "assd_ci95_mm": [assd_low, assd_high],
                "mean_surface_dice": mean_surface_dice,
                "surface_dice_ci95": [surface_low, surface_high],
                "mean_lesion_f1": mean_lesion_f1,
                "lesion_f1_ci95": [lesion_low, lesion_high],
                "mean_elapsed_seconds": mean_elapsed,
                "elapsed_seconds_ci95": [elapsed_low, elapsed_high],
                "fold_mean_dice": dice_by_fold,
                "evaluation_paths": [str(path) for _, _, path in records],
            }
        )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "dinov3_medical_confirmatory_summary.v1",
        "unit_of_uncertainty": "mean metric across training seeds within each independent support fold",
        "bootstrap_samples": 10000,
        "rows": rows,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print("{} k{}: Dice {:.4f} [{:.4f}, {:.4f}] ({} folds)".format(
            row["task"], row["support_count"], row["mean_dice"], row["dice_ci95"][0], row["dice_ci95"][1], row["n_folds"]
        ))


if __name__ == "__main__":
    main()
