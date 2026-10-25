#!/usr/bin/env python3
"""Shared active-learning helpers for the FlexiCT pair and nnU-Net k-fold paths.

Both pipelines rank an unlabeled case pool by prediction disagreement and
write the same review artifacts (input/<case>.nii.gz, input_geometries.json
with mcs_path, uncertainty_run_*/ranking.csv, bands/), so the review window
and the Mimics-side overlay monitor serve both backends unchanged.

This module must not import nnunet_pipeline at module level: nnunet_pipeline
imports this module, so the pipeline helpers (_case_directories,
_find_case_image, _raise_if_cancelled) are imported lazily inside function
bodies — the same seam flexict_pipeline already uses.
"""

from __future__ import annotations

import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from nnunet_common import (  # noqa: E402
    read_json,
    unique_case_keys,
    update_status,
    write_json_atomic,
)


def al_fold_ids(model_dir: Path) -> list[str]:
    """Trained 0-4 folds in a model bundle (each with checkpoint_final.pth).

    fold_all (train-on-everything) yields a single model — no k-fold
    disagreement is possible from it, so it is not listed here.
    """
    return sorted(
        path.name[5:]
        for path in model_dir.glob("fold_*")
        if path.is_dir()
        and re.fullmatch(r"fold_[0-4]", path.name)
        and (path / "checkpoint_final.pth").is_file()
    )


def case_mcs_path(dataset_root: Path, original_case_name: str) -> str:
    """The case's .mcs project path under the dataset's mcs_output dir,
    or "" when none exists. Read-only: never creates anything.

    The .mcs file is created from the ORIGINAL case name by the 01_Import
    flow (mimics_import.py), not from the lossy internal slug — a Chinese
    name would otherwise never resolve here.
    """
    try:
        from tools.mimics_label_export import resolve_mimics_output_dir

        mcs_root = resolve_mimics_output_dir(dataset_root)
        candidate = mcs_root / "{}.mcs".format(original_case_name)
        return str(candidate) if candidate.is_file() else ""
    except Exception:
        return ""


def materialize_al_pool(job_dir: Path,
                        request: dict[str, Any],
                        status_path: Path,
                        control_path: Path,
                        schema_version: str) -> list[str]:
    """Materialize the unlabeled pool as flat <case>.nii.gz inputs.

    ``schema_version`` tags the geometry document per pipeline; readers only
    look at the ``cases`` payload. Never writes the source dataset — every
    output lands inside the job directory.
    """
    import nnunet_pipeline as np_mod
    from tools.mimics_label_export import materialize_source_image as _materialize

    dataset_root = Path(
        str(request.get("dataset_root") or "")).expanduser().resolve()
    requested = set(
        str(case).strip() for case in (request.get("cases") or [])
        if str(case).strip())
    case_dirs = np_mod._case_directories(dataset_root, requested or None)
    if not case_dirs:
        raise RuntimeError(
            "No image cases were found in {}.".format(dataset_root))
    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    case_ids = []
    geometries: dict[str, dict[str, Any]] = {}
    total = len(case_dirs)
    # Pool keys must be collision-free: two Chinese case names would share
    # one input .nii.gz and one geometries entry (F22). No collisions keeps
    # the plain slug.
    case_keys = unique_case_keys(
        [path.name for path in case_dirs], fallback="case")
    for index, case_dir in enumerate(case_dirs, start=1):
        np_mod._raise_if_cancelled(control_path)
        image = np_mod._find_case_image(case_dir)
        if image is None:
            continue
        case_id = case_keys[case_dir.name]
        destination = input_dir / "{}.nii.gz".format(case_id)
        _materialize(image, destination)
        case_ids.append(case_id)
        try:
            import nibabel as nib

            header = nib.load(str(destination))
            geometries[case_id] = {
                "source_shape": [int(v) for v in header.shape[:3]],
                "source_voxel_to_ras_matrix": header.affine.astype(
                    float).tolist(),
                # Where the case came from, so the review window can ask
                # Mimics to open it (the .mcs convention mirrors
                # mimics_label_export's mcs_output lookup).
                "source_image_path": str(image),
                "case_name": case_dir.name,
                "mcs_path": case_mcs_path(dataset_root, case_dir.name),
            }
        except Exception:
            geometries[case_id] = {
                "source_image_path": str(image),
                "case_name": case_dir.name,
                "mcs_path": case_mcs_path(dataset_root, case_dir.name),
            }
        update_status(
            status_path,
            phase="preparing_inputs",
            current_case=case_id,
            preparation_index=index,
            preparation_total=total,
            prepared_cases=len(case_ids),
            progress_percent=5 + int(10 * index / max(1, total)),
        )
    if not case_ids:
        raise RuntimeError(
            "No readable images were found in {}.".format(dataset_root))
    # Per-case source geometry: the Mimics-side overlay verification compares
    # the active image against this before applying any band mask.
    write_json_atomic(
        job_dir / "input_geometries.json",
        {"schema_version": schema_version, "cases": geometries},
    )
    return sorted(case_ids)


def validate_prediction_grids(predictions: dict[str, Path],
                              case_ids: list[str],
                              geometry_path: Path,
                              log_path: Path) -> list[str]:
    """S09: every prediction must match its CASE's input grid before it
    enters the disagreement math.

    run_batch compares the models' outputs against each other; predictions
    that share the same stale or wrong grid pass that pair check while both
    deviate from the case input. This pre-check compares each source's
    prediction against the input geometry recorded at materialization time
    (input_geometries.json) and drops any case whose prediction cannot be
    proven on-grid, appending the per-case reason to the uncertainty log
    run_batch also writes (append mode, same file).
    """
    import nibabel as nib
    import numpy as np

    geometry_doc = read_json(geometry_path, {}) or {}
    geo_by_case = {
        str(key): value
        for key, value in (geometry_doc.get("cases") or {}).items()
        if isinstance(value, dict)
    }
    valid: list[str] = []
    for case_id in case_ids:
        geometry = geo_by_case.get(case_id) or {}
        expected_shape = tuple(
            int(value) for value in geometry.get("source_shape") or [])
        expected_affine = geometry.get("source_voxel_to_ras_matrix")
        reason = ""
        if not expected_shape or expected_affine is None:
            reason = "输入几何未知，无法核对预测网格"
        else:
            for source, directory in sorted(predictions.items()):
                path = directory / "{}.nii.gz".format(case_id)
                if not path.is_file():
                    reason = "缺少 {0} 的预测文件".format(source)
                    break
                try:
                    image = nib.load(str(path))
                except Exception as exc:
                    reason = "{0} 预测无法读取（{1}）".format(source, exc)
                    break
                if tuple(image.shape[:3]) != expected_shape or not np.allclose(
                    image.affine,
                    np.asarray(expected_affine, dtype=float),
                    atol=1e-4,
                    rtol=0.0,
                ):
                    reason = "{0} 预测的网格与该病例输入不一致".format(source)
                    break
        if reason:
            try:
                with open(str(log_path), "a", encoding="utf-8") as handle:
                    handle.write("[{0}] [跳过] {1}: {2}\n".format(
                        time.strftime("%Y-%m-%d %H:%M:%S"), case_id, reason))
            except OSError:
                pass
        else:
            valid.append(case_id)
    return valid


def write_uncertainty_bands(uncertainty_dir: Path,
                            case_ids: list[str],
                            method: str,
                            moderate: int,
                            high: int) -> dict[str, Any]:
    """Derive the two overlay band masks (moderate >= t1, high >= t2) from the
    uint8 x10 uncertainty maps. The bands stay on the source grid so the
    Mimics bridge can resample them like any prediction mask.

    The thresholds are data-aware: a two-model disagreement pair can only
    produce the level 5 (1 - 1/2 scaled by 10), so a high band at the
    default 6 would never fire. Both thresholds are clamped to the levels
    actually present in the maps."""
    import nibabel as nib
    import numpy as np

    bands_dir = uncertainty_dir / "bands"
    bands_dir.mkdir(parents=True, exist_ok=True)
    images: dict[str, Any] = {}
    max_level = 0
    for case_id in case_ids:
        source = uncertainty_dir / "{}_uncertainty_{}.nii.gz".format(
            case_id, method)
        if not source.is_file():
            continue
        image = nib.load(str(source))
        values = np.asanyarray(image.dataobj)
        images[case_id] = (image, values)
        max_level = max(max_level, int(values.max()) if values.size else 0)
    if high > max_level:
        high = max_level
    if moderate >= high:
        moderate = max(1, high - 1)
    written = {"moderate": 0, "high": 0}
    coincident = bool(images)
    for case_id, (image, values) in images.items():
        case_bands = {}
        for level, threshold in (("moderate", moderate), ("high", high)):
            if threshold < 1:
                coincident = False
                continue
            band = (values >= threshold).astype(np.uint8)
            case_bands[level] = band
            if not band.any():
                continue
            nib.save(
                nib.Nifti1Image(band, image.affine, image.header),
                str(bands_dir / "{}_{}.nii.gz".format(case_id, level)))
            written[level] += 1
        if len(case_bands) != 2 or not np.array_equal(
                case_bands["moderate"], case_bands["high"]):
            coincident = False
    return {"dir": str(bands_dir), "moderate_threshold": moderate,
            "high_threshold": high, "written": written,
            "coincident": bool(coincident and written["high"])}


def rank_disagreement(job_dir: Path,
                      predictions: dict[str, Path],
                      case_ids: list[str],
                      method: str,
                      target_labels: "list[int] | None"
                      ) -> "tuple[Path, list[dict[str, Any]], list[str], int]":
    """S09/S10-hardened disagreement ranking shared by both AL pipelines.

    Grid-validates every prediction against input_geometries.json, runs the
    uncertainty batch over the valid cases, quarantines failed-case
    fragments under uncertainty_dir/failed/, and ranks THIS run's success
    list — refusing (Chinese RuntimeError pointing at the log) when nothing
    survives or the ranking does not match the success list.

    Returns (uncertainty_dir, ranking_rows, successful_cases, excluded_case_count).
    """
    # S10: a fresh, per-run directory. analyze_uncertainty_dir ranks
    # whatever *_uncertainty_*.nii.gz it finds in the folder — sharing
    # one "uncertainty" dir across retries let a previously successful
    # run's files keep failed cases in the ranking (and the bands) as
    # if they were current results. Old run dirs stay on disk as
    # history; they are never mixed into this run's ranking. The uuid
    # suffix keeps two runs started within the same second apart.
    uncertainty_dir = job_dir / "uncertainty_run_{}_{}".format(
        time.strftime("%Y%m%dT%H%M%S"), uuid.uuid4().hex[:8])
    # The one sanctioned sys.path seam into the standalone repo.
    from flexict_common import flexict_repo_dir, load_config

    repo = flexict_repo_dir(load_config())
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from uncertainty import analyze_uncertainty_dir, run_batch  # noqa: E402

    # S09: each prediction is first checked against its case's input
    # grid (the pair check inside run_batch cannot see a shared wrong
    # grid), with per-case skip reasons in the uncertainty log.
    valid_case_ids = validate_prediction_grids(
        predictions, case_ids,
        job_dir / "input_geometries.json",
        job_dir / "uncertainty_log.txt")
    if not valid_case_ids:
        raise RuntimeError(
            "本轮没有病例成功计算不确定度（{0} 例全部在预测网格核对中被"
            "剔除），排名未生成。逐例原因见任务日志：{1}".format(
                len(case_ids), job_dir / "uncertainty_log.txt"))
    # Cases the uncertainty batch skips (grid mismatch, missing mask)
    # vanish silently otherwise — the review window shows fewer rows than
    # the job was submitted with, with no reason anywhere in the UI.
    # run_batch's return value is the per-case success list; the log holds
    # the Chinese skip reasons for the annotator to read.
    metrics = run_batch(
        mask_dirs=[str(directory) for directory in predictions.values()],
        out_dir=str(uncertainty_dir),
        case_names=valid_case_ids,
        filename_template="{case}.nii.gz",
        method=method,
        min_masks=2,
        # target_labels=None（单标签）时 run_batch 自动取全部非零标签；
        # 多分类调用方传 1..N。
        target_labels=target_labels,
        save_components=False,
        save_consensus=True,
        save_plots=False,
        log_path=str(job_dir / "uncertainty_log.txt"),
    )
    if not metrics:
        raise RuntimeError(
            "本轮没有病例成功计算不确定度（{0} 例全部被剔除或处理失败），"
            "排名未生成。逐例原因见任务日志：{1}".format(
                len(case_ids), job_dir / "uncertainty_log.txt"))
    excluded_case_count = max(0, len(case_ids) - len(metrics))
    # S10 re-review: the ranking and the overlay bands must consume
    # THIS run's success list and nothing else. A case run_batch failed
    # mid-write (uncertainty map saved, then the consensus write failed)
    # leaves fragments in the directory that the directory scan would
    # rank as if they were current results — quarantine them under
    # failed/ (kept for diagnosis, invisible to the scan and the bands).
    successful_cases = {str(row.get("case") or "") for row in metrics}
    failed_cases = [
        case for case in valid_case_ids if case not in successful_cases]
    if failed_cases:
        failed_dir = uncertainty_dir / "failed"
        failed_dir.mkdir(parents=True, exist_ok=True)
        # U01: match the EXACT per-case output filenames, never a
        # textual prefix. A failed "case01" used to quarantine
        # "case01_phase2_uncertainty_*.nii.gz" too — a fully successful
        # case vanished from the ranking and the run failed its own
        # consistency check. run_batch writes exactly these two names
        # per case here (save_components/save_plots are off).
        fragment_names = set()
        for case in failed_cases:
            fragment_names.add(
                "{0}_uncertainty_{1}.nii.gz".format(case, method))
            fragment_names.add("{0}_consensus.nii.gz".format(case))
        for fragment in sorted(uncertainty_dir.glob("*.nii.gz")):
            if fragment.name in fragment_names:
                # os.replace (not shutil.replace): the bundled
                # python_env's stripped shutil lacks replace.
                os.replace(
                    str(fragment), str(failed_dir / fragment.name))
    ranking = analyze_uncertainty_dir(
        str(uncertainty_dir),
        method=method,
        sort_key="integrated",
        report_csv=str(uncertainty_dir / "ranking.csv"),
        compute_boundary=False,
        compute_cc=False,
    )
    # S10 contract: the ranked set IS this run's success list. Anything
    # else means fragments leaked past the quarantine — refuse to hand
    # the annotator a ranking they cannot trust.
    ranked_cases = {str(row.get("case") or "") for row in ranking}
    if ranked_cases != successful_cases:
        raise RuntimeError(
            "排名结果与本轮成功清单不一致（排名 {0} 例、成功 {1} 例），"
            "排名未生成。请重试本轮不确定度计算。".format(
                len(ranked_cases), len(successful_cases)))
    return uncertainty_dir, ranking, sorted(successful_cases), excluded_case_count
