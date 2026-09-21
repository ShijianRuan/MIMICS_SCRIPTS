#!/usr/bin/env python3
"""Turn completed confirmation statistics into conservative task-specific guidance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def _status(row: dict, expected_folds: int, expected_seeds: int) -> str:
    if int(row.get("n_folds", 0)) < expected_folds or int(row.get("n_training_seeds", 0)) < expected_seeds:
        return "incomplete"
    if row.get("mean_dice") is None or not row.get("dice_ci95"):
        return "invalid"
    return "complete"


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive conservative organ-specific recommendations from confirmation results")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    args = parser.parse_args()
    plan = yaml.safe_load(Path(args.plan).read_text(encoding="utf-8")) or {}
    study = plan["study"]
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    expected_folds = len(study["confirmation_folds"])
    expected_seeds = len(study["confirmation_training_seeds"])
    references = {candidate["task"]: candidate["id"] for candidate in plan["candidates"] if candidate.get("is_reference", False)}
    rows_by_task = {}
    for row in summary.get("rows", []):
        if int(row.get("support_count", -1)) == 5:
            rows_by_task.setdefault(row["task"], []).append(row)

    recommendations = []
    for task, reference_id in sorted(references.items()):
        rows = rows_by_task.get(task, [])
        reference = next((row for row in rows if row["candidate_id"] == reference_id), None)
        complete = [row for row in rows if _status(row, expected_folds, expected_seeds) == "complete"]
        if reference is None or _status(reference, expected_folds, expected_seeds) != "complete":
            recommendations.append({
                "task": task,
                "status": "incomplete",
                "reason": "The fixed frozen reference has not completed all folds and seeds at K=5.",
            })
            continue
        best = max(complete, key=lambda row: (float(row["mean_dice"]), -float(row.get("mean_hd95_mm") or float("inf")))) if complete else reference
        ref_ci = [float(value) for value in reference["dice_ci95"]]
        best_ci = [float(value) for value in best["dice_ci95"]]
        if best["candidate_id"] == reference_id:
            evidence = "reference_retained"
            reason = "No completed candidate exceeded the frozen reference mean Dice."
        elif best_ci[0] > ref_ci[1]:
            evidence = "replicated_nonoverlapping_ci_improvement"
            reason = "The winner's fold-bootstrap Dice interval is entirely above the frozen reference interval."
        elif float(best["mean_dice"]) > float(reference["mean_dice"]):
            evidence = "point_estimate_improvement_only"
            reason = "The winner exceeds the reference point estimate, but intervals overlap; retain both pending more cases."
        else:
            evidence = "reference_retained"
            reason = "The frozen reference remains at least as strong by confirmed mean Dice."
        recommendations.append({
            "task": task,
            "status": "complete",
            "reference_candidate_id": reference_id,
            "recommended_candidate_id": best["candidate_id"],
            "evidence": evidence,
            "reason": reason,
            "reference": reference,
            "recommended": best,
        })

    payload = {
        "schema_version": "dinov3_medical_task_recommendations.v1",
        "study": study["id"],
        "rule": "K=5 only; require all confirmation folds and training seeds; never make a clinical claim.",
        "recommendations": recommendations,
    }
    json_output = Path(args.output_json).resolve()
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    markdown = ["# DINOv3 Few-Shot Task Recommendations", "", "Generated from completed confirmation results. This is research guidance, not clinical validation.", ""]
    for entry in recommendations:
        markdown.append("## {}".format(entry["task"]))
        markdown.append("")
        if entry["status"] != "complete":
            markdown.append("Status: incomplete. {}".format(entry["reason"]))
        else:
            recommended = entry["recommended"]
            markdown.append("Recommended candidate: `{}`".format(entry["recommended_candidate_id"]))
            markdown.append("")
            markdown.append("Evidence: `{}`. {}".format(entry["evidence"], entry["reason"]))
            markdown.append("")
            markdown.append(
                "K=5 Dice {:.4f} [{:.4f}, {:.4f}], HD95 {} mm.".format(
                    float(recommended["mean_dice"]),
                    float(recommended["dice_ci95"][0]),
                    float(recommended["dice_ci95"][1]),
                    "NA" if recommended.get("mean_hd95_mm") is None else "{:.2f}".format(float(recommended["mean_hd95_mm"])),
                )
            )
        markdown.append("")
    markdown_output = Path(args.output_markdown).resolve()
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print("Saved {} and {}".format(json_output, markdown_output))


if __name__ == "__main__":
    main()
