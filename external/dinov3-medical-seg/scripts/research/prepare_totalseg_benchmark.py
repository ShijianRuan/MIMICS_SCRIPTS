#!/usr/bin/env python3
"""Materialize immutable multi-organ few-shot folds from TotalSegmentator.

The script never edits the source directory. Each fold contains a five-case
support pool in ``imagesTr/labelsTr`` and all remaining valid cases in
``imagesVal/labelsVal``. K=1 and K=3 experiments use nested prefixes of the
same support pool, making shot-scaling comparisons auditable.

The build is resumable at task and fold granularity. A per-task case cache
materializes each canonicalized image/label pair once and links it into every
fold, so a compressed CT is never resampled four times. ``benchmark_manifest``
is merged rather than overwritten, so preparing one task later never erases the
summary of a task that already finished on a preempted worker.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.research import protocol
from src.research.protocol import TASKS, atomic_json, discover_records, materialize_fold


def _estimated_output_bytes(records_by_task, folds: int) -> int:
    total = 0
    for records in records_by_task.values():
        for record in records:
            total += Path(record["image_path"]).stat().st_size
            total += Path(record["label_path"]).stat().st_size
    # Resampled CT output is often somewhat larger than gzip source data.
    return int(total * max(1, folds) * 1.7)


def _fold_summary(destination: Path, fold: int, manifest: dict) -> dict:
    selection = manifest["selection"]
    return {
        "fold": fold,
        "path": str(destination),
        "support_case_ids": selection["support_case_ids_ordered"],
        "evaluation_cases": len(selection["evaluation_case_ids"]),
        "fingerprint_sha256": manifest["fingerprint_sha256"],
    }


def merge_benchmark_manifest(manifest_path: Path, source: Path, support_pool_size: int, folds: int, seed: int,
                             task_summaries: dict) -> dict:
    """Merge one build's task summaries into any pre-existing benchmark manifest.

    A per-task invocation must extend, never replace, the ``tasks`` map so a
    later run for one organ cannot silently drop organs already committed to the
    Volume by an earlier (possibly preempted) run.
    """
    existing = {}
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = dict(existing.get("tasks", {}))
    tasks.update(task_summaries)
    payload = {
        "schema_version": "dinov3_medical_multi_organ_benchmark.v2",
        "source": str(source),
        "support_pool_size": int(support_pool_size),
        "folds": int(folds),
        "seed": int(seed),
        "tasks": tasks,
    }
    atomic_json(manifest_path, payload)
    return payload


def build_benchmark(source: Path, output: Path, tasks, support_pool_size: int, folds: int, seed: int,
                    *, overwrite: bool = False, check_disk: bool = True) -> dict:
    """Build folds for ``tasks`` and merge their summaries into the benchmark manifest.

    Returns the merged manifest dict. Safe to call once per task: each call
    reuses a per-task case cache, skips folds whose ``manifest.json`` already
    exists, and never overwrites another task's manifest summary.
    """
    source = Path(source).resolve()
    output = Path(output).resolve()
    if support_pool_size < 1 or folds < 1:
        raise SystemExit("support-pool-size and folds must both be positive")

    records_by_task = {}
    for task in tasks:
        print("Scanning {}...".format(task), flush=True)
        records = discover_records(source, task)
        if len(records) <= support_pool_size:
            raise SystemExit(
                "{} has {} valid cases, not enough for {} supports plus held-out evaluation".format(
                    task, len(records), support_pool_size
                )
            )
        records_by_task[task] = records
        print("{}: {} valid cases".format(task, len(records)), flush=True)

    if check_disk:
        estimated = _estimated_output_bytes(records_by_task, folds)
        anchor = output.parent if output.parent.exists() else PROJECT_ROOT
        free = shutil.disk_usage(anchor).free
        print("Estimated materialized size: {:.2f} GiB; free space: {:.2f} GiB".format(
            estimated / (1024 ** 3), free / (1024 ** 3)
        ), flush=True)
        if free < estimated:
            raise SystemExit("Insufficient free disk space for the requested folds")

    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "benchmark_manifest.json"
    task_summaries = {}
    for task, records in records_by_task.items():
        # One cache per task: canonicalize+resample each case once, link into folds.
        case_cache = output / task / "_case_cache"
        summaries = []
        for fold in range(folds):
            destination = output / task / "fold_{:02d}".format(fold)
            fold_manifest_path = destination / "manifest.json"
            if fold_manifest_path.is_file() and not overwrite:
                manifest = json.loads(fold_manifest_path.read_text(encoding="utf-8"))
                print("{} fold {:02d}: already materialized, skipping".format(task, fold), flush=True)
            else:
                # A fold directory that exists with content but no manifest is a
                # worker that was preempted mid-write. Rebuild it cleanly rather
                # than tripping materialize_fold's overwrite guard.
                partial = destination.exists() and any(destination.iterdir()) and not fold_manifest_path.is_file()
                manifest = materialize_fold(
                    records,
                    task,
                    destination,
                    support_count=support_pool_size,
                    seed=seed + fold,
                    overwrite=overwrite or partial,
                    case_cache=case_cache,
                )
                selection = manifest["selection"]
                print("{} fold {:02d}: {} support + {} evaluation".format(
                    task, fold,
                    len(selection["support_case_ids_ordered"]),
                    len(selection["evaluation_case_ids"]),
                ), flush=True)
            summaries.append(_fold_summary(destination, fold, manifest))
        task_summaries[task] = summaries
        # Merge after each task so a preempted retry keeps finished organs.
        merge_benchmark_manifest(manifest_path, source, support_pool_size, folds, seed, {task: summaries})
        print("{}: committed {} folds to manifest".format(task, len(summaries)), flush=True)

    merged = merge_benchmark_manifest(manifest_path, source, support_pool_size, folds, seed, task_summaries)
    print("Saved benchmark manifest: {}".format(manifest_path), flush=True)
    return merged


def main():
    parser = argparse.ArgumentParser(description="Build reproducible TotalSegmentator few-shot folds")
    parser.add_argument("--source", required=True, help="Directory containing sXXXX/ct.nii.gz")
    parser.add_argument("--output", required=True, help="New benchmark output directory")
    parser.add_argument("--tasks", nargs="*", choices=sorted(TASKS), default=sorted(TASKS))
    parser.add_argument("--support-pool-size", type=int, default=5)
    parser.add_argument("--folds", type=int, default=4, help="One screen fold plus confirmation folds")
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace existing fold directories")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    if not source.is_dir():
        raise SystemExit("Source directory does not exist: {}".format(source))
    if args.support_pool_size < 1 or args.folds < 1:
        raise SystemExit("support-pool-size and folds must both be positive")

    if args.dry_run:
        records_by_task = {}
        for task in args.tasks:
            print("Scanning {}...".format(task), flush=True)
            records = discover_records(source, task)
            records_by_task[task] = records
            print("{}: {} valid cases".format(task, len(records)))
        estimated = _estimated_output_bytes(records_by_task, args.folds)
        anchor = output.parent if output.parent.exists() else PROJECT_ROOT
        free = shutil.disk_usage(anchor).free
        print("Estimated materialized size: {:.2f} GiB; free space: {:.2f} GiB".format(
            estimated / (1024 ** 3), free / (1024 ** 3)
        ))
        if free < estimated:
            raise SystemExit("Insufficient free disk space for the requested folds")
        return

    build_benchmark(
        source=source,
        output=output,
        tasks=args.tasks,
        support_pool_size=args.support_pool_size,
        folds=args.folds,
        seed=args.seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
