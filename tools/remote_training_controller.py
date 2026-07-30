#!/usr/bin/env python3
"""Local controller for optional SSH/Docker AI training jobs.

The controller runs outside Mimics. It prepares source-grid NIfTI inputs with
the existing local code, uploads one job archive, mirrors remote status into
the existing local status file, downloads a verified model, and registers that
model through the same local registry used by local training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import traceback
import uuid
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "tools"):
    text = str(candidate)
    if text not in sys.path:
        sys.path.insert(0, text)

try:
    from tools.remote_compute import (  # noqa: E402
        RemoteCommandError,
        RemoteComputeError,
        SSHSession,
        docker_gpu_request,
        get_profile,
        read_json,
        safe_identifier,
        write_json_atomic,
    )
except ImportError:
    from remote_compute import (  # noqa: E402
        RemoteCommandError,
        RemoteComputeError,
        SSHSession,
        docker_gpu_request,
        get_profile,
        read_json,
        safe_identifier,
        write_json_atomic,
    )


TERMINAL = {"completed", "failed", "cancelled", "paused"}
REMOTE_TRAINING_LABEL = "mimics-script.remote-training"
REMOTE_OWNER_LABEL = "mimics-script.owner"
REMOTE_JOB_LABEL = "mimics-script.job"
LOCAL_STATUS_KEYS = {
    "cancel_path",
    "control_path",
    "controller_pid",
    "created_at_epoch",
    "execution_backend",
    "job_id",
    "kind",
    "launcher_pid",
    "local_log_path",
    "controller_log_path",
    "diagnostic_log_paths",
    "log_path",
    "organ",
    "profile_name",
    "remote_container_name",
    "remote_control_kind",
    "remote_control_path",
    "remote_job_dir",
    "remote_profile_id",
    "request_path",
    "retry_context",
    "schema_version",
    "task_id",
    "task_name",
    "training_options",
    "train_log",
    "ts_root",
    "workspace",
}


def _status_update(path: Path, **values: Any) -> dict[str, Any]:
    payload = read_json(path, {}) or {}
    payload.update(values)
    payload["updated_at_epoch"] = time.time()
    write_json_atomic(path, payload)
    return payload


def _append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write(
            "[{}] {}\n".format(
                time.strftime("%Y-%m-%d %H:%M:%S"), str(message).rstrip()
            )
        )


def _split_csv(value: object) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [
        item.strip()
        for item in str(value or "").replace(";", ",").split(",")
        if item.strip()
    ]


def _cancel_requested(status_path: Path) -> bool:
    status = read_json(status_path, {}) or {}
    cancel_path = str(status.get("cancel_path") or "")
    if cancel_path and Path(cancel_path).is_file():
        return True
    control_path = str(status.get("control_path") or "")
    control = read_json(control_path, {}) if control_path else {}
    return str((control or {}).get("action") or "").lower() in {
        "stop",
        "cancel",
    }


def _raise_if_cancelled(status_path: Path) -> None:
    if _cancel_requested(status_path):
        raise InterruptedError("cancel")


def _pause_requested(status_path: Path) -> bool:
    status = read_json(status_path, {}) or {}
    control_path = str(status.get("control_path") or "")
    control = read_json(control_path, {}) if control_path else {}
    return str((control or {}).get("action") or "").lower() == "pause"


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with tarfile.open(str(archive), "r") as handle:
        for member in handle.getmembers():
            if member.issym() or member.islnk():
                raise RuntimeError(
                    "Remote result archive contains a link, which is not "
                    "accepted: {}".format(member.name)
                )
            if not (member.isfile() or member.isdir()):
                raise RuntimeError(
                    "Remote result archive contains an unsupported entry: "
                    "{}".format(member.name)
                )
            target = (destination / member.name).resolve()
            try:
                target.relative_to(destination_root)
            except ValueError:
                raise RuntimeError(
                    "Remote result archive contains an unsafe path: {}".format(
                        member.name
                    )
                )
        try:
            handle.extractall(str(destination), filter="fully_trusted")
        except TypeError:
            # Python 3.10 does not expose tarfile extraction filters. Every
            # member has already been type-checked and confined above.
            handle.extractall(str(destination))


def _normalized_tar_info(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


def _build_tar(
    source: Path,
    archive: Path,
    *,
    excluded_top_level: set[str] | None = None,
) -> int:
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(archive.name + ".tmp")
    excluded = set(excluded_top_level or ())
    try:
        with tarfile.open(str(temporary), "w") as handle:
            for path in sorted(source.rglob("*")):
                relative = path.relative_to(source)
                if relative.parts and relative.parts[0] in excluded:
                    continue
                handle.add(
                    str(path),
                    arcname=str(relative),
                    recursive=False,
                    filter=_normalized_tar_info,
                )
        os.replace(str(temporary), str(archive))
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass
    return int(archive.stat().st_size)


def _build_dataset_tar(bundle: Path, archive: Path) -> tuple[str, int]:
    included = {
        name for name in ("input", "labels") if (bundle / name).exists()
    }
    if not included:
        return "", 0
    size = _build_tar(
        bundle,
        archive,
        excluded_top_level={
            path.name for path in bundle.iterdir() if path.name not in included
        },
    )
    return _sha256_file(archive), size


def _build_dataset_parts(
    bundle: Path, destination: Path
) -> list[dict[str, Any]]:
    """Build deterministic per-case archives for incremental remote reuse."""
    case_names = set()
    for top_name in ("input", "labels"):
        top = bundle / top_name
        if not top.is_dir():
            continue
        case_names.update(
            child.name for child in top.iterdir() if child.is_dir()
        )
    if not case_names:
        archive = destination / "dataset_shared.tar"
        fingerprint, size = _build_dataset_tar(bundle, archive)
        return (
            [{
                "case_id": "shared",
                "archive": archive,
                "fingerprint": fingerprint,
                "size": size,
            }]
            if fingerprint
            else []
        )
    destination.mkdir(parents=True, exist_ok=True)
    parts = []
    for case_name in sorted(case_names):
        archive = destination / (
            "dataset_{}.tar".format(safe_identifier(case_name, "case"))
        )
        temporary = archive.with_name(archive.name + ".tmp")
        try:
            with tarfile.open(str(temporary), "w") as handle:
                for top_name in ("input", "labels"):
                    source = bundle / top_name / case_name
                    if not source.exists():
                        continue
                    paths = [source]
                    if source.is_dir():
                        paths.extend(sorted(source.rglob("*")))
                    for path in paths:
                        handle.add(
                            str(path),
                            arcname=str(path.relative_to(bundle)),
                            recursive=False,
                            filter=_normalized_tar_info,
                        )
            os.replace(str(temporary), str(archive))
        finally:
            try:
                temporary.unlink()
            except OSError:
                pass
        parts.append({
            "case_id": case_name,
            "archive": archive,
            "fingerprint": _sha256_file(archive),
            "size": int(archive.stat().st_size),
        })
    return parts


def _dataset_parts_fingerprint(parts: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part["case_id"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(part["fingerprint"]).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest() if parts else ""


def _copy_model_input(
    source_value: str, bundle: Path
) -> tuple[str, str]:
    source = Path(source_value).expanduser().resolve()
    if not source.exists():
        raise RuntimeError("Selected base model does not exist: {}".format(source))
    destination = bundle / "custom_model"
    if source.is_dir():
        shutil.copytree(str(source), str(destination))
        return str(destination), "/job/custom_model"
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / source.name
    shutil.copy2(str(source), str(target))
    return str(target), "/job/custom_model/{}".format(source.name)


def _local_model_fingerprint(path: Path) -> str:
    path = path.resolve()
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if not path.is_dir():
        return ""
    candidates = []
    allowed_names = {"config.json", "plans.json"}
    allowed_suffixes = {".pth", ".safetensors", ".bin"}
    for candidate in path.rglob("*"):
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(path)
        if len(relative.parts) > 4:
            continue
        if (
            candidate.name not in allowed_names
            and candidate.suffix.lower() not in allowed_suffixes
        ):
            continue
        candidates.append((relative.as_posix(), candidate))
    if not candidates:
        return ""
    aggregate = hashlib.sha256()
    for relative, candidate in sorted(candidates):
        file_digest = hashlib.sha256()
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                file_digest.update(chunk)
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(file_digest.hexdigest().encode("ascii"))
        aggregate.update(b"\0")
    return aggregate.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_dino(spec: dict[str, Any], bundle: Path) -> dict[str, Any]:
    import tools.fewshot_pipeline as pipeline
    from tools.fewshot_training_setup_ui import append_training_args

    context = dict(spec["context"])
    options = dict(spec["options"])
    status_path = Path(spec["status_path"]).resolve()
    cancel_path = Path(spec["cancel_path"]).resolve()
    workspace = Path(context["workspace"]).resolve()
    run_id = str(spec["run_id"])
    organ = str(context["organ"])
    organ_slug = pipeline.safe_slug(organ)
    cases = set(_split_csv(options.get("cases"))) or None
    mask_names = pipeline.resolve_training_mask_names(
        context.get("config") or {}, organ, options.get("mask_names")
    )
    label_source = str(options.get("label_source") or "mcs_refresh")
    label_root: Path | None = None
    fresh_root: Path | None = None
    if label_source == "mcs_refresh":
        _status_update(
            status_path,
            status="exporting_labels",
            phase="preparing_remote_labels",
            progress_percent=2,
        )
        fresh_root = (
            workspace / "runs" / organ_slug / run_id / "remote_fresh_labels"
        )
        plan = pipeline._plan_dino_mcs_label_cache(
            Path(context["ts_root"]).resolve(),
            workspace,
            organ_slug,
            cases,
            mask_names,
            mcs_output_dir=options.get("mcs_output_dir")
            or context.get("mcs_output_dir"),
        )
        fresh_root.mkdir(parents=True, exist_ok=True)
        for case_id, cached_label in plan["reusable"].items():
            pipeline._copy_label_atomic(
                cached_label,
                fresh_root
                / case_id
                / "segmentations"
                / cached_label.name,
            )
        _status_update(
            status_path,
            label_cache_reused=len(plan["reusable"]),
            label_cache_refresh=len(plan["changed"]),
        )
        if plan["changed"]:
            changed_root = fresh_root.with_name(
                fresh_root.name + "_changed"
            )
            result = pipeline.launch_mimics_export(
                Path(context["ts_root"]).resolve(),
                set(plan["changed"]),
                context.get("mimics_exe"),
                workspace,
                float((context.get("config") or {}).get("label_export_timeout_seconds", 3600)),
                status_path=status_path,
                cancel_path=cancel_path,
                lock_timeout_seconds=float(
                    options.get("background_mimics_lock_timeout_seconds", 1800)
                ),
                label_staging_dir=changed_root,
                export_space="source_image",
                mask_names=mask_names,
                target_mask_name=organ_slug,
                mcs_output_dir=options.get("mcs_output_dir")
                or context.get("mcs_output_dir"),
                skip_projects_without_requested_mask=True,
                skip_invalid_projects=True,
            )
            batch = result.get("batch_status") or {}
            if (
                not result.get("launched")
                or result.get("timed_out")
                or int(result.get("returncode", 0) or 0) != 0
                or str(batch.get("status") or "").lower()
                in {"failed", "stopping"}
            ):
                raise RuntimeError(
                    "Local Mimics label export did not complete successfully. "
                    "Remote training was not started. Diagnostics: {}".format(
                        result.get("job_runtime")
                        or result.get("log")
                        or changed_root
                    )
                )
            available, cache_warnings = (
                pipeline._publish_dino_mcs_label_cache(
                    plan,
                    changed_root,
                    mask_names,
                    output_root=fresh_root,
                )
            )
            if cache_warnings:
                _status_update(
                    status_path,
                    label_cache_warnings=cache_warnings[:10],
                )
            shutil.rmtree(str(changed_root), ignore_errors=True)
        available = {
            case_id
            for case_id in plan["requested"]
            if (
                fresh_root / case_id / "segmentations"
            ).is_dir()
        }
        cases = available
        label_root = fresh_root
    elif label_source == "exported_masks":
        label_root = Path(str(options.get("label_root") or "")).resolve()
        if not label_root.is_dir():
            raise RuntimeError(
                "Exported masks folder does not exist: {}".format(label_root)
            )

    samples, skipped = pipeline.discover_samples(
        Path(context["ts_root"]).resolve(),
        organ,
        cases,
        label_root=label_root,
        fallback_to_case_labels=label_root is None,
        label_source=label_source,
        mask_names=mask_names,
    )
    selected = pipeline.select_samples(
        samples,
        str(options.get("sample_mode") or "all"),
        int(options.get("max_samples") or 0),
    )
    minimum = int(options.get("min_samples") or 1)
    if len(selected) < minimum:
        raise RuntimeError(
            "Remote training needs at least {} usable case(s), but {} were "
            "found. {} case(s) were skipped.".format(
                minimum, len(selected), len(skipped)
            )
        )
    train_rows, val_rows = pipeline.split_train_validation(
        selected,
        val_fraction=float(options.get("val_fraction") or 0.0),
        val_cases=_split_csv(options.get("val_cases")),
        min_train_samples=minimum,
        min_val_samples=int(options.get("min_val_samples") or 0),
    )
    input_root = bundle / "input"
    labels_root = bundle / "labels"
    all_rows = train_rows + val_rows
    _status_update(
        status_path,
        status="preparing_remote",
        phase="preparing_remote_data",
        preparation_total=len(all_rows),
        preparation_index=0,
        progress_percent=5,
    )
    for index, row in enumerate(all_rows):
        if _cancel_requested(status_path):
            raise InterruptedError("cancel")
        case_id = pipeline.safe_slug(row["case_id"])
        image_dst = input_root / case_id / "ct.nii.gz"
        label_dst = labels_root / case_id / "segmentations" / (
            organ_slug + ".nii.gz"
        )
        pipeline._materialize_source_image(row["image"], image_dst)
        pipeline._materialize_label_on_source_grid(
            row["label"], image_dst, label_dst
        )
        _status_update(
            status_path,
            status="preparing_remote",
            current_case=row["case_id"],
            preparation_index=index + 1,
            progress_percent=5
            + int(15.0 * (index + 1) / max(1, len(all_rows))),
        )

    remote_options = dict(options)
    remote_options["cases"] = ",".join(
        str(row["case_id"]) for row in all_rows
    )
    remote_options["val_cases"] = ",".join(
        str(row["case_id"]) for row in val_rows
    )
    remote_options["label_root"] = ""
    remote_options["mcs_output_dir"] = ""
    base_config = str(
        remote_options.get("base_config")
        or (context.get("config") or {}).get("base_config")
        or "config/research/ct_fewshot_fast.yaml"
    )
    base_path = Path(base_config)
    if base_path.is_absolute():
        dinov3_root = Path(context["dinov3_root"]).resolve()
        try:
            base_config = str(base_path.resolve().relative_to(dinov3_root))
        except ValueError:
            raise RuntimeError(
                "Remote training currently requires a base configuration inside "
                "the bundled DINOv3 project: {}".format(base_path)
            )
    remote_options["base_config"] = base_config.replace("\\", "/")

    custom_model = str(remote_options.get("model_path") or "").strip()
    if custom_model:
        _local, remote_model = _copy_model_input(custom_model, bundle)
        remote_options["model_path"] = remote_model
        required_model_relative = ""
        local_required_model = ""
    else:
        decoder = str(remote_options.get("decoder") or "")
        backend = str(remote_options.get("encoder_backend") or "auto")
        scale = str(remote_options.get("model_scale") or "vitb16")
        model_dir = "/models/dinov3/dinov3-{}".format(scale)
        remote_options["model_path"] = (
            model_dir + "/model.onnx"
            if decoder == "feature_unet2d" and backend in {"auto", "onnx"}
            else model_dir
        )
        required_model_relative = remote_options["model_path"].replace(
            "/models/", "", 1
        )
        local_model_dir = (
            Path(context["dinov3_root"]).resolve()
            / "models"
            / "dinov3-{}".format(scale)
        )
        local_required_model = str(
            local_model_dir / "model.onnx"
            if remote_options["model_path"].endswith("/model.onnx")
            else local_model_dir
        )

    arguments = [
        "train",
        "--ts-root",
        "/job/input",
        "--workspace",
        "/job/output",
        "--organ",
        organ,
        "--dinov3-root",
        "/app/external/dinov3-medical-seg",
        "--python",
        "__REMOTE_PYTHON__",
        "--run-id",
        run_id,
    ]
    append_training_args(
        arguments, context.get("config") or {}, remote_options
    )
    arguments.extend(
        [
            "--label-root",
            "/job/labels",
            "--mask-names",
            organ_slug,
        ]
    )
    request = {
        "schema_version": "mimics_remote_training_request.v1",
        "kind": "dinov3_train",
        "job_id": run_id,
        "pipeline_args": arguments,
        "remote_status_path": "/job/output/jobs/{}.json".format(run_id),
        "artifact_path": "/job/output/models/{}/{}".format(organ_slug, run_id),
        "created_at_epoch": time.time(),
    }
    write_json_atomic(bundle / "remote_request.json", request)
    if fresh_root is not None:
        shutil.rmtree(str(fresh_root), ignore_errors=True)
    return {
        "remote_status_relative": "output/jobs/{}.json".format(run_id),
        "remote_control_relative": "output/runs/{}/{}/cancel.request".format(
            organ_slug, run_id
        ),
        "remote_control_kind": "marker",
        "remote_artifact_relative": "output/models/{}/{}".format(
            organ_slug, run_id
        ),
        "train_count": len(train_rows),
        "validation_count": len(val_rows),
        "required_model_relative": required_model_relative,
        "local_required_model": local_required_model,
    }


def _prepare_nninteractive(
    spec: dict[str, Any], bundle: Path
) -> dict[str, Any]:
    import tools.nninteractive_finetune_pipeline as pipeline

    local_job_dir = Path(spec["job_dir"]).resolve()
    status_path = local_job_dir / "status.json"
    request = read_json(local_job_dir / "request.json", {}) or {}
    if not request:
        raise RuntimeError("Local nnInteractive training request is missing.")
    manifest_path, _validation_path = pipeline._prepare_manifest(
        request,
        local_job_dir,
        status_path,
        local_job_dir / "control.json",
        local_job_dir / "job.log",
    )
    manifest = read_json(manifest_path, {}) or {}
    rows = manifest.get("cases") or []
    if not rows:
        raise RuntimeError("No prepared nnInteractive case is available.")
    remote_rows = []
    input_root = bundle / "input"
    for index, row in enumerate(rows):
        if _cancel_requested(status_path):
            raise InterruptedError("cancel")
        case_id = safe_identifier(row.get("case_id"), "case")
        case_root = input_root / case_id
        case_root.mkdir(parents=True, exist_ok=True)
        image_dst = case_root / "image.nii.gz"
        label_dst = case_root / "label.nii.gz"
        shutil.copy2(str(row["image"]), str(image_dst))
        shutil.copy2(str(row["label"]), str(label_dst))
        remote_row = {
            "case_id": str(row.get("case_id") or case_id),
            "image": "/job/input/{}/image.nii.gz".format(case_id),
            "label": "/job/input/{}/label.nii.gz".format(case_id),
            "initial_mask": "",
            "split": str(row.get("split") or "train"),
            "state": "ready",
        }
        initial_mask = str(row.get("initial_mask") or "").strip()
        if initial_mask:
            initial_dst = case_root / "initial_mask.nii.gz"
            shutil.copy2(initial_mask, str(initial_dst))
            remote_row["initial_mask"] = (
                "/job/input/{}/initial_mask.nii.gz".format(case_id)
            )
        remote_rows.append(remote_row)
        _status_update(
            status_path,
            status="preparing_remote",
            phase="preparing_remote_data",
            preparation_index=index + 1,
            preparation_total=len(rows),
            progress_percent=10
            + int(10.0 * (index + 1) / max(1, len(rows))),
        )

    remote_request = dict(request)
    remote_request["workspace"] = "/job/remote_workspace"
    remote_request["source_mode"] = "prepared"
    remote_request["prepared_root"] = "/job/input"
    remote_request["image_root"] = "/job/input"
    remote_request["label_root"] = "/job/input"
    remote_request["mcs_dir"] = ""
    remote_request["cases"] = remote_rows
    if any(row.get("initial_mask") for row in remote_rows):
        remote_request["initial_mask_source"] = "exported_masks"
        remote_request["initial_mask_root"] = "/job/input"
        remote_request["initial_mask_names"] = ["initial_mask"]
    else:
        remote_request["initial_mask_source"] = "synthetic"
        remote_request["initial_mask_root"] = ""
        remote_request["initial_mask_names"] = []
    remote_request["output_model_dir"] = "/job/model_output"
    if str(request.get("parent_model_id") or "official") == "official":
        remote_request[
            "base_model_dir"
        ] = "/models/nninteractive/nnInteractive_v1.0"
        required_model_relative = "nninteractive/nnInteractive_v1.0"
        local_required_model = str(
            Path(request["base_model_dir"]).expanduser().resolve()
        )
    else:
        _local, remote_model = _copy_model_input(
            str(request["base_model_dir"]), bundle
        )
        remote_request["base_model_dir"] = remote_model
        required_model_relative = ""
        local_required_model = ""
    remote_request["mimics_exe"] = ""
    remote_payload = {
        "schema_version": "mimics_remote_training_request.v1",
        "kind": "nninteractive_train",
        "job_id": str(request["job_id"]),
        "pipeline_request": remote_request,
        "remote_status_path": "/job/pipeline_job/status.json",
        "artifact_path": "/job/model_output",
        "created_at_epoch": time.time(),
    }
    write_json_atomic(bundle / "remote_request.json", remote_payload)
    return {
        "remote_status_relative": "pipeline_job/status.json",
        "remote_control_relative": "pipeline_job/control.json",
        "remote_control_kind": "json",
        "remote_artifact_relative": "model_output",
        "train_count": sum(row.get("split") == "train" for row in remote_rows),
        "validation_count": sum(row.get("split") == "val" for row in remote_rows),
        "required_model_relative": required_model_relative,
        "local_required_model": local_required_model,
    }


def _profile_for_spec(spec: dict[str, Any]) -> dict[str, Any]:
    profile_id = str(spec.get("remote_profile_id") or "")
    if not profile_id:
        raise RuntimeError("Remote server profile is missing from the job request.")
    return get_profile(profile_id)


def _remote_paths(session: SSHSession, job_id: str) -> dict[str, str]:
    owner = safe_identifier(session.profile.get("username"), "user")
    jobs = (
        PurePosixPath(session.remote_root)
        / "jobs"
        / owner
    )
    job = jobs / safe_identifier(job_id, "job")
    return {
        "jobs": str(jobs),
        "job": str(job),
        "archive": str(jobs / (safe_identifier(job_id, "job") + ".tar")),
        "models": str(PurePosixPath(session.remote_root) / "models"),
        "locks": str(PurePosixPath(session.remote_root) / "locks"),
        "dataset_cache": str(
            PurePosixPath(session.remote_root)
            / "cache"
            / owner
            / "datasets"
        ),
    }


def _remote_owner(profile: dict[str, Any]) -> str:
    return safe_identifier(profile.get("username"), "user")


def _validate_remote_job_path(
    remote_root: str,
    owner: str,
    remote_job_dir: str,
) -> PurePosixPath:
    candidate = PurePosixPath(str(remote_job_dir or ""))
    expected_parent = (
        PurePosixPath(str(remote_root))
        / "jobs"
        / safe_identifier(owner, "user")
    )
    if (
        not remote_job_dir
        or any(part == ".." for part in candidate.parts)
        or candidate.parent != expected_parent
        or candidate.name in {"", ".", ".."}
    ):
        raise RuntimeError(
            "Refusing to modify a remote path outside this user's job folder: "
            "{}".format(remote_job_dir or "<empty>")
        )
    return candidate


def _validate_remote_control_path(
    remote_job_dir: str,
    remote_control_path: str,
) -> None:
    if not remote_control_path:
        return
    job = PurePosixPath(remote_job_dir)
    control = PurePosixPath(remote_control_path)
    if (
        any(part == ".." for part in control.parts)
        or control == job
        or job not in control.parents
    ):
        raise RuntimeError(
            "Refusing to write a control marker outside the remote job folder."
        )


def _container_name(profile: dict[str, Any], job_id: str) -> str:
    return safe_identifier(
        "mimics-ai-{}-{}".format(profile["username"], job_id),
        "mimics-ai-job",
    )[:120]


def _launch_container(
    session: SSHSession,
    paths: dict[str, str],
    profile: dict[str, Any],
    container_name: str,
    dataset_cache_paths: list[str] | None = None,
    remove_dataset_after_extract: bool = False,
) -> str:
    owner = _remote_owner(profile)
    job_slug = safe_identifier(PurePosixPath(paths["job"]).name, "job")
    _validate_remote_job_path(
        str(profile["remote_root"]),
        owner,
        paths["job"],
    )
    if _assert_container_owned(
        session,
        container_name,
        expected_owner=owner,
        expected_job=job_slug,
    ):
        existing_state, _existing_code = _container_state(
            session, container_name
        )
        if existing_state not in {"exited", "dead", "created"}:
            raise RuntimeError(
                "This remote training job already has a running container. "
                "Open its status instead of starting a duplicate controller."
            )
        if not _remove_container(
            session,
            container_name,
            expected_owner=owner,
            expected_job=job_slug,
            stop_if_running=False,
        ):
            raise RuntimeError(
                "The previous owned remote container could not be removed."
            )
    dataset_extract = ""
    for dataset_cache_path in dataset_cache_paths or []:
        dataset_extract += (
            "tar -xf {dataset} -C {job} && {after_extract}"
        ).format(
            dataset=shlex.quote(dataset_cache_path),
            job=shlex.quote(paths["job"]),
            after_extract=(
                "rm -f {dataset} && ".format(
                    dataset=shlex.quote(dataset_cache_path)
                )
                if remove_dataset_after_extract
                else "touch {dataset} && ".format(
                    dataset=shlex.quote(dataset_cache_path)
                )
            ),
        )
    session.execute(
        "rm -rf {job} && mkdir -p {job} && "
        "{dataset_extract}"
        "tar -xf {archive} -C {job} && rm -f {archive}".format(
            job=shlex.quote(paths["job"]),
            archive=shlex.quote(paths["archive"]),
            dataset_extract=dataset_extract,
        ),
        timeout=300,
    )
    gpu_device = str(profile.get("gpu_device") or "auto")
    gpu_scope = "all" if gpu_device == "auto" else "device"
    gpu_lock = "gpu-{}.lock".format(
        "automatic" if gpu_scope == "all" else safe_identifier(gpu_device, "device")
    )
    command = (
        "docker run -d --name {name} --gpus {gpu_request} --network none "
        "--label mimics-script.remote-training=true "
        "--label {owner_label} "
        "--label {job_label} "
        "--label {gpu_label} "
        "-v {job}:/job -v {models}:/models:ro -v {locks}:/remote-locks "
        "-e MIMICS_AI_APP_ROOT=/app "
        "-e MIMICS_REMOTE_GPU_GLOBAL_LOCK=/remote-locks/gpu-all.lock "
        "-e MIMICS_REMOTE_GPU_LOCK=/remote-locks/{gpu_lock} "
        "-e MIMICS_REMOTE_GPU_LOCK_SCOPE={gpu_scope} "
        "-e MIMICS_REMOTE_GPU_DEVICE={gpu_device} "
        "{image} python /app/tools/remote_worker.py run --job-dir /job"
    ).format(
        name=shlex.quote(container_name),
        gpu_request=shlex.quote(docker_gpu_request(profile)),
        owner_label=shlex.quote(
            "{}={}".format(REMOTE_OWNER_LABEL, owner)
        ),
        job_label=shlex.quote(
            "{}={}".format(REMOTE_JOB_LABEL, job_slug)
        ),
        gpu_label=shlex.quote(
            "mimics-script.gpu={}".format(
                safe_identifier(gpu_device, "automatic")
            )
        ),
        job=shlex.quote(paths["job"]),
        models=shlex.quote(paths["models"]),
        locks=shlex.quote(paths["locks"]),
        gpu_lock=shlex.quote(gpu_lock),
        gpu_scope=shlex.quote(gpu_scope),
        gpu_device=shlex.quote(gpu_device),
        image=shlex.quote(profile["runtime_image"]),
    )
    return session.execute(command).strip()


def _validate_remote_assets(
    session: SSHSession,
    paths: dict[str, str],
    prepared: dict[str, Any],
) -> dict[str, str]:
    relative = str(prepared.get("required_model_relative") or "").strip()
    if not relative:
        return {}
    model_path = str(PurePosixPath(paths["models"]) / PurePosixPath(relative))
    output = session.execute(
        "test -e {path} && printf ready || printf missing".format(
            path=shlex.quote(model_path)
        ),
        check=False,
    ).strip()
    if output != "ready":
        raise RuntimeError(
            "The required base model is not installed on the remote server: "
            "{}. Ask the server administrator to install it under the configured "
            "remote work folder.".format(model_path)
        )
    fingerprint = session.execute(
        "if test -f {path}; then "
        "sha256sum {path} | awk '{{print $1}}'; "
        "else cd {path} && "
        "find . -maxdepth 4 -type f "
        "\\( -name '*.pth' -o -name '*.safetensors' -o -name '*.bin' "
        "-o -name 'config.json' -o -name 'plans.json' \\) "
        "-printf '%P\\n' | LC_ALL=C sort | "
        "while IFS= read -r file; do "
        "printf '%s\\0' \"$file\"; "
        "sha256sum \"$file\" | awk '{{printf \"%s\", $1}}'; "
        "printf '\\0'; done | sha256sum | awk '{{print $1}}'; "
        "fi".format(path=shlex.quote(model_path)),
        timeout=300,
    ).strip()
    if (
        not fingerprint
        or fingerprint == hashlib.sha256(b"").hexdigest()
    ):
        raise RuntimeError(
            "The remote base-model path exists but contains no supported "
            "weight files: {}".format(model_path)
        )
    local_path_text = str(prepared.get("local_required_model") or "").strip()
    local_fingerprint = (
        _local_model_fingerprint(Path(local_path_text))
        if local_path_text and Path(local_path_text).exists()
        else ""
    )
    if local_fingerprint and local_fingerprint != fingerprint:
        raise RuntimeError(
            "The remote base model does not match the corresponding local "
            "model. Local SHA-256: {}; remote SHA-256: {}. Install the same "
            "base weights on the server before training.".format(
                local_fingerprint, fingerprint
            )
        )
    return {
        "remote_base_model_path": model_path,
        "remote_base_model_sha256": fingerprint,
        "local_base_model_sha256": local_fingerprint,
    }


def _container_state(session: SSHSession, container_name: str) -> tuple[str, int]:
    code, output = session.execute_result(
        "docker inspect --format '{{{{.State.Status}}}} "
        "{{{{.State.ExitCode}}}}' {}".format(shlex.quote(container_name))
    )
    output = output.strip()
    if code != 0:
        lowered = output.lower()
        if "no such object" in lowered or "no such container" in lowered:
            return "missing", -1
        raise RemoteCommandError(
            "Could not inspect remote container '{}': {}".format(
                container_name, output.strip()[-2000:]
            )
        )
    parts = output.split()
    if len(parts) < 2:
        return "missing", -1
    try:
        code = int(parts[-1])
    except Exception:
        code = -1
    return parts[0].lower(), code


def _container_labels(
    session: SSHSession,
    container_name: str,
) -> dict[str, str]:
    code, output = session.execute_result(
        "docker inspect --format '{{{{json .Config.Labels}}}}' {}".format(
            shlex.quote(container_name)
        )
    )
    if code != 0:
        raise RemoteCommandError(
            "Could not inspect ownership labels for remote container '{}': "
            "{}".format(container_name, output.strip()[-2000:])
        )
    try:
        labels = json.loads(output.strip())
    except Exception as exc:
        raise RemoteCommandError(
            "Remote container '{}' returned invalid ownership labels: {}".format(
                container_name, exc
            )
        )
    return {
        str(key): str(value)
        for key, value in (labels or {}).items()
    }


def _assert_container_owned(
    session: SSHSession,
    container_name: str,
    *,
    expected_owner: str,
    expected_job: str,
) -> bool:
    state, _code = _container_state(session, container_name)
    if state == "missing":
        return False
    labels = _container_labels(session, container_name)
    expected = {
        REMOTE_TRAINING_LABEL: "true",
        REMOTE_OWNER_LABEL: safe_identifier(expected_owner, "user"),
        REMOTE_JOB_LABEL: safe_identifier(expected_job, "job"),
    }
    mismatches = [
        key
        for key, value in expected.items()
        if labels.get(key) != value
    ]
    if mismatches:
        raise RuntimeError(
            "Refusing to control remote container '{}': its ownership labels "
            "do not match this training job ({}).".format(
                container_name, ", ".join(mismatches)
            )
        )
    return True


def _remove_container(
    session: SSHSession,
    container_name: str,
    *,
    expected_owner: str,
    expected_job: str,
    stop_if_running: bool = True,
) -> bool:
    if not _assert_container_owned(
        session,
        container_name,
        expected_owner=expected_owner,
        expected_job=expected_job,
    ):
        return True
    state, _code = _container_state(session, container_name)
    if stop_if_running and state not in {"exited", "dead", "created"}:
        session.execute(
            "docker stop --time 20 {} >/dev/null 2>&1 || true".format(
                shlex.quote(container_name)
            ),
            timeout=40,
            check=False,
        )
    session.execute(
        "docker rm -f {} >/dev/null 2>&1 || true".format(
            shlex.quote(container_name)
        ),
        timeout=40,
        check=False,
    )
    state, _code = _container_state(session, container_name)
    return state == "missing"


def _remove_remote_job(
    session: SSHSession,
    *,
    remote_root: str,
    expected_owner: str,
    remote_job_dir: str,
) -> bool:
    job = _validate_remote_job_path(
        remote_root,
        expected_owner,
        remote_job_dir,
    )
    session.execute(
        "rm -rf -- {}".format(shlex.quote(str(job))),
        timeout=120,
        check=False,
    )
    code, _output = session.execute_result(
        "test ! -e {}".format(shlex.quote(str(job)))
    )
    return code == 0


def _reconnect_session(
    profile: dict[str, Any],
    status_path: Path,
    local_log: Path,
    reason: BaseException,
    attempt: int,
) -> SSHSession | None:
    _raise_if_cancelled(status_path)
    delay = min(30.0, max(2.0, float(2 ** min(attempt, 5))))
    current = read_json(status_path, {}) or {}
    previous_status = str(
        current.get("remote_previous_status")
        or current.get("status")
        or "starting_remote"
    )
    if previous_status == "reconnecting_remote":
        previous_status = "starting_remote"
    message = (
        "Remote connection was interrupted ({}). The remote task state is "
        "unchanged; reconnecting in {:.0f} seconds.".format(reason, delay)
    )
    _append_log(local_log, message)
    _status_update(
        status_path,
        status="reconnecting_remote",
        phase="reconnecting_remote",
        remote_connection_error=str(reason),
        remote_reconnect_attempt=attempt,
        remote_state_unknown=True,
        remote_previous_status=previous_status,
    )
    deadline = time.time() + delay
    while time.time() < deadline:
        _raise_if_cancelled(status_path)
        time.sleep(min(0.5, max(0.0, deadline - time.time())))
    try:
        session = SSHSession(profile)
    except Exception as exc:
        _append_log(
            local_log,
            "Remote reconnect attempt {} failed: {}".format(attempt, exc),
        )
        return None
    if _cancel_requested(status_path):
        session.close()
        raise InterruptedError("cancel")
    _status_update(
        status_path,
        status=previous_status,
        phase="monitoring_remote",
        remote_connection_error="",
        remote_state_unknown=False,
        remote_reconnect_attempt=attempt,
    )
    return session


def _merge_remote_status(
    status_path: Path,
    remote: dict[str, Any],
    profile: dict[str, Any],
    container_name: str,
    remote_job_dir: str,
) -> dict[str, Any]:
    local = read_json(status_path, {}) or {}
    preserved = {key: local.get(key) for key in LOCAL_STATUS_KEYS if key in local}
    remote_copy = dict(remote)
    remote_state = str(remote_copy.get("status") or "").lower()
    try:
        remote_progress = float(remote_copy.get("progress_percent") or 0.0)
    except Exception:
        remote_progress = 0.0
    remote_progress = max(0.0, min(100.0, remote_progress))
    remote_copy["remote_training_progress_percent"] = remote_progress
    remote_copy["progress_percent"] = 35 + int(remote_progress * 0.60)
    if remote_state == "completed":
        remote_copy["remote_pipeline_status"] = "completed"
        remote_copy["status"] = "finalizing_remote"
        remote_copy["phase"] = "remote_training_completed"
        remote_copy["progress_percent"] = 95
    remote_metrics = remote_copy.get("metrics_history")
    for key in (
        "cancel_path",
        "control_path",
        "controller_pid",
        "log_path",
        "metrics_history",
        "request_path",
        "train_log",
        "workspace",
        "ts_root",
    ):
        remote_copy.pop(key, None)
    if isinstance(remote_metrics, list):
        remote_copy["metrics_history"] = remote_metrics[-500:]
    local.update(remote_copy)
    local.update(preserved)
    local.update(
        {
            "execution_backend": "remote",
            "remote_profile_id": profile["profile_id"],
            "profile_name": profile["name"],
            "remote_container_name": container_name,
            "remote_job_dir": remote_job_dir,
            "remote_status_updated_at_epoch": remote.get("updated_at_epoch"),
            "updated_at_epoch": time.time(),
        }
    )
    write_json_atomic(status_path, local)
    return local


def _upload_progress(
    status_path: Path,
    *,
    phase: str = "uploading_training_data",
    start_percent: int = 20,
    end_percent: int = 35,
    aggregate_offset: int = 0,
    aggregate_total: int = 0,
):
    state = {"last_time": 0.0, "last_percent": -1}

    def callback(transferred: int, total: int) -> None:
        _raise_if_cancelled(status_path)
        now = time.time()
        effective_transferred = int(aggregate_offset) + int(transferred)
        effective_total = int(aggregate_total) or int(total)
        percent = int(
            100.0 * effective_transferred / max(1, effective_total)
        )
        percent = min(100, max(0, percent))
        if percent == state["last_percent"] and now - state["last_time"] < 1.0:
            return
        if now - state["last_time"] < 0.25 and percent < 100:
            return
        state["last_time"] = now
        state["last_percent"] = percent
        _status_update(
            status_path,
            status="uploading",
            phase=phase,
            transfer_bytes=effective_transferred,
            transfer_total_bytes=effective_total,
            transfer_percent=percent,
            progress_percent=start_percent
            + int(percent * max(0, end_percent - start_percent) / 100.0),
        )

    return callback


def _ensure_remote_dataset_archive(
    session: SSHSession,
    paths: dict[str, str],
    profile: dict[str, Any],
    local_archive: Path,
    fingerprint: str,
    status_path: Path,
    progress_callback=None,
) -> tuple[str, bool]:
    use_cache = bool(profile.get("cache_training_data", True))
    if use_cache:
        session.ensure_directory(paths["dataset_cache"])
        remote_archive = str(
            PurePosixPath(paths["dataset_cache"]) / (fingerprint + ".tar")
        )
        code, output = session.execute_result(
            "if test -f {path}; then sha256sum {path} | awk '{{print $1}}'; fi".format(
                path=shlex.quote(remote_archive)
            )
        )
        if code == 0 and output.strip() == fingerprint:
            session.execute(
                "touch {}".format(shlex.quote(remote_archive)),
                check=False,
            )
            cache_progress = {}
            if progress_callback is not None:
                size = int(local_archive.stat().st_size)
                progress_callback(size, size)
            else:
                cache_progress = {
                    "transfer_bytes": 0,
                    "transfer_total_bytes": int(
                        local_archive.stat().st_size
                    ),
                    "transfer_percent": 100,
                    "progress_percent": 33,
                }
            _status_update(
                status_path,
                status="uploading",
                phase="remote_dataset_cache_hit",
                dataset_cache_hit=True,
                dataset_fingerprint=fingerprint,
                **cache_progress
            )
            return remote_archive, True
        if output.strip():
            session.execute(
                "rm -f {}".format(shlex.quote(remote_archive)),
                check=False,
            )
    else:
        remote_archive = (
            paths["archive"] + ".dataset." + fingerprint + ".tar"
        )

    _status_update(
        status_path,
        status="uploading",
        phase="uploading_training_dataset",
        dataset_cache_hit=False,
        dataset_fingerprint=fingerprint,
    )
    session.upload(
        local_archive,
        remote_archive,
        callback=progress_callback
        or _upload_progress(
            status_path,
            phase="uploading_training_dataset",
            start_percent=20,
            end_percent=33,
        ),
    )
    remote_digest = session.execute(
        "sha256sum {path} | awk '{{print $1}}'".format(
            path=shlex.quote(remote_archive)
        ),
        timeout=600,
    ).strip()
    if remote_digest != fingerprint:
        session.execute(
            "rm -f {}".format(shlex.quote(remote_archive)),
            check=False,
        )
        raise RuntimeError(
            "The uploaded training-data archive failed SHA-256 verification."
        )
    return remote_archive, False


def _download_progress(status_path: Path):
    state = {"last_time": 0.0, "last_percent": -1}

    def callback(transferred: int, total: int) -> None:
        _raise_if_cancelled(status_path)
        now = time.time()
        percent = int(100.0 * transferred / max(1, total))
        if percent == state["last_percent"] and now - state["last_time"] < 1.0:
            return
        if now - state["last_time"] < 0.25 and percent < 100:
            return
        state["last_time"] = now
        state["last_percent"] = percent
        _status_update(
            status_path,
            status="downloading",
            phase="downloading_model",
            transfer_bytes=int(transferred),
            transfer_total_bytes=int(total),
            transfer_percent=percent,
            progress_percent=96 + int(percent * 0.03),
        )

    return callback


def _stop_remote(
    session: SSHSession,
    container_name: str,
    status_path: Path,
    final: str = "cancelled",
    remote_control_path: str = "",
    remote_control_kind: str = "",
    remote_job_dir: str = "",
    remote_root: str = "",
    expected_owner: str = "",
    expected_job: str = "",
) -> None:
    job = _validate_remote_job_path(
        remote_root,
        expected_owner,
        remote_job_dir,
    )
    _validate_remote_control_path(str(job), remote_control_path)
    _status_update(
        status_path,
        status="stopping",
        phase="stopping_remote_container",
        remote_stop_confirmed=False,
    )
    container_exists = _assert_container_owned(
        session,
        container_name,
        expected_owner=expected_owner,
        expected_job=expected_job,
    )
    if remote_control_path and container_exists:
        parent = str(PurePosixPath(remote_control_path).parent)
        if remote_control_kind == "json":
            content = json.dumps(
                {"action": "stop", "requested_at_epoch": time.time()},
                sort_keys=True,
            )
        else:
            content = "cancel\n"
        session.execute(
            "mkdir -p {parent} && printf %s {content} > {temporary} && "
            "mv -f {temporary} {target}".format(
                parent=shlex.quote(parent),
                content=shlex.quote(content),
                temporary=shlex.quote(remote_control_path + ".tmp"),
                target=shlex.quote(remote_control_path),
            ),
            timeout=30,
        )
        graceful_deadline = time.time() + 20.0
        while time.time() < graceful_deadline:
            state, _code = _container_state(session, container_name)
            if state in {"exited", "dead", "missing"}:
                break
            time.sleep(1.0)
    if container_exists:
        session.execute(
            "docker stop --time 20 {} >/dev/null 2>&1 || true".format(
                shlex.quote(container_name)
            ),
            timeout=40,
            check=False,
        )
    state, _code = _container_state(session, container_name)
    confirmed = state in {"exited", "dead", "missing", "created"}
    _status_update(
        status_path,
        status=final if confirmed else "stopping",
        phase=final if confirmed else "remote_termination_pending",
        remote_stop_confirmed=confirmed,
        completed_at_epoch=time.time() if confirmed else None,
        error=""
        if confirmed
        else "The remote container did not confirm termination.",
    )
    if not confirmed:
        raise RuntimeError("The remote container did not confirm termination.")
    container_removed = _remove_container(
        session,
        container_name,
        expected_owner=expected_owner,
        expected_job=expected_job,
        stop_if_running=False,
    )
    job_removed = _remove_remote_job(
        session,
        remote_root=remote_root,
        expected_owner=expected_owner,
        remote_job_dir=str(job),
    )
    _status_update(
        status_path,
        remote_container_removed=bool(container_removed),
        remote_job_removed=bool(job_removed),
        remote_diagnostics_retained=not bool(job_removed),
        remote_cleanup_warning=(
            ""
            if container_removed and job_removed
            else "Training stopped, but some remote files could not be removed."
        ),
    )


def _sync_remote_log(
    session: SSHSession,
    remote_job: str,
    local_log: Path,
    state: dict[str, int],
) -> None:
    remote_log = str(PurePosixPath(remote_job) / "remote_worker.log")
    if session.path_exists(remote_log):
        state["offset"] = session.download_appended(
            remote_log,
            local_log,
            int(state.get("offset") or 0),
        )


def _download_artifact(
    session: SSHSession,
    remote_job: str,
    relative: str,
    local_target: Path,
    local_archive: Path,
    status_path: Path,
) -> None:
    artifact = str(PurePosixPath(remote_job) / PurePosixPath(relative))
    result_tar = str(PurePosixPath(remote_job) / "result.tar")
    session.execute(
        "test -d {artifact} && tar -cf {archive} -C {artifact} .".format(
            artifact=shlex.quote(artifact),
            archive=shlex.quote(result_tar),
        ),
        timeout=300,
    )
    remote_archive_sha256 = session.execute(
        "sha256sum {archive} | awk '{{print $1}}'".format(
            archive=shlex.quote(result_tar)
        ),
        timeout=300,
    ).strip()
    if len(remote_archive_sha256) != 64:
        raise RuntimeError(
            "The remote model archive checksum could not be read."
        )
    _status_update(
        status_path,
        status="downloading",
        phase="downloading_model",
        progress_percent=96,
    )
    session.download(
        result_tar,
        local_archive,
        callback=_download_progress(status_path),
    )
    local_archive_sha256 = _sha256_file(local_archive)
    if local_archive_sha256 != remote_archive_sha256:
        try:
            local_archive.unlink()
        except OSError:
            pass
        raise RuntimeError(
            "The downloaded model archive failed SHA-256 verification."
        )
    _status_update(
        status_path,
        remote_model_archive_sha256=remote_archive_sha256,
    )
    temporary = local_target.with_name(local_target.name + ".remote-part")
    backup = local_target.with_name(
        "{}.remote-backup-{}".format(local_target.name, uuid.uuid4().hex)
    )
    shutil.rmtree(str(temporary), ignore_errors=True)
    shutil.rmtree(str(backup), ignore_errors=True)
    _safe_extract(local_archive, temporary)
    had_existing = local_target.exists()
    try:
        if had_existing:
            os.replace(str(local_target), str(backup))
        os.replace(str(temporary), str(local_target))
    except Exception:
        if had_existing and backup.exists() and not local_target.exists():
            os.replace(str(backup), str(local_target))
        raise
    finally:
        shutil.rmtree(str(temporary), ignore_errors=True)
    shutil.rmtree(str(backup), ignore_errors=True)


def _register_dino(
    spec: dict[str, Any],
    local_model_dir: Path,
    remote_status: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    import tools.fewshot_pipeline as pipeline

    context = spec["context"]
    run_id = str(spec["run_id"])
    organ = str(context["organ"])
    organ_slug = pipeline.safe_slug(organ)
    manifest_path = local_model_dir / "manifest.json"
    manifest = read_json(manifest_path, {}) or {}
    if not manifest:
        raise RuntimeError("Downloaded DINOv3 model manifest is missing.")
    manifest.update(
        {
            "execution_backend": "remote",
            "remote_profile_id": profile["profile_id"],
            "remote_profile_name": profile["name"],
            "remote_runtime_image": profile["runtime_image"],
            "remote_gpu_device": remote_status.get(
                "remote_gpu_device", profile.get("gpu_device") or "auto"
            ),
            "remote_dataset_fingerprint": remote_status.get(
                "dataset_fingerprint", ""
            ),
            "remote_runtime_image_id": remote_status.get(
                "remote_runtime_image_id", ""
            ),
            "remote_base_model_path": remote_status.get(
                "remote_base_model_path", ""
            ),
            "remote_base_model_sha256": remote_status.get(
                "remote_base_model_sha256", ""
            ),
            "local_base_model_sha256": remote_status.get(
                "local_base_model_sha256", ""
            ),
            "ts_root": str(Path(context["ts_root"]).resolve()),
            "workspace": str(Path(context["workspace"]).resolve()),
            "downloaded_at_epoch": time.time(),
        }
    )
    pipeline.write_json_atomic(manifest_path, manifest)
    latest = dict(manifest)
    latest["checkpoint"] = "{}/model.pth".format(run_id)
    latest["config"] = "{}/config.yaml".format(run_id)
    latest["model_manifest"] = "{}/manifest.json".format(run_id)
    latest_path = (
        Path(context["workspace"]) / "models" / organ_slug / "latest.json"
    )
    pipeline.write_json_atomic(latest_path, latest)
    pipeline.register_global_model(manifest, manifest_path)
    return manifest


def _register_nninteractive(
    spec: dict[str, Any],
    local_model_dir: Path,
    remote_status: dict[str, Any],
    profile: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    import tools.nninteractive_finetune_pipeline as pipeline

    local_job_dir = Path(spec["job_dir"]).resolve()
    request = read_json(local_job_dir / "request.json", {}) or {}
    workspace = Path(request["workspace"]).resolve()
    quality = remote_status.get("quality") or None
    validation_count = sum(
        row.get("split") == "val" for row in request.get("cases") or []
    )
    model, selected = pipeline._register_model(
        request, workspace, local_model_dir, quality, validation_count
    )
    model.update(
        {
            "execution_backend": "remote",
            "remote_profile_id": profile["profile_id"],
            "remote_profile_name": profile["name"],
            "remote_runtime_image": profile["runtime_image"],
            "remote_gpu_device": remote_status.get(
                "remote_gpu_device", profile.get("gpu_device") or "auto"
            ),
            "remote_dataset_fingerprint": remote_status.get(
                "dataset_fingerprint", ""
            ),
            "remote_runtime_image_id": remote_status.get(
                "remote_runtime_image_id", ""
            ),
            "remote_base_model_sha256": remote_status.get(
                "remote_base_model_sha256", ""
            ),
            "local_base_model_sha256": remote_status.get(
                "local_base_model_sha256", ""
            ),
        }
    )
    pipeline.write_json_atomic(
        local_model_dir / "integration_manifest.json", model
    )
    registry = pipeline.load_registry(workspace)
    for task in registry.get("tasks") or []:
        if pipeline.safe_slug(task.get("task_id")) != pipeline.safe_slug(
            request["task_id"]
        ):
            continue
        for row in task.get("models") or []:
            if row.get("model_id") == model.get("model_id"):
                row.update(model)
    pipeline.save_registry(workspace, registry)
    return model, selected


def run(spec_path: Path) -> int:
    spec = read_json(spec_path, {}) or {}
    kind = str(spec.get("kind") or "")
    status_path = Path(spec["status_path"]).resolve()
    profile = _profile_for_spec(spec)
    job_id = str(spec["run_id"] if kind == "dinov3" else spec["job_id"])
    owner = _remote_owner(profile)
    job_slug = safe_identifier(job_id, "job")
    container_name = _container_name(profile, job_id)
    local_root = status_path.parent / (job_id + "_remote")
    bundle = local_root / "bundle"
    archive = local_root / "job_upload.tar"
    dataset_archive_dir = local_root / "dataset_parts"
    result_archive = local_root / "result.tar"
    if kind == "nninteractive":
        remote_log = status_path.parent / "job.log"
        controller_log = remote_log
    else:
        remote_log = local_root / "remote_training.log"
        controller_log = status_path.with_name(
            status_path.name + ".remote_controller.log"
        )
    shutil.rmtree(str(local_root), ignore_errors=True)
    bundle.mkdir(parents=True, exist_ok=True)
    _status_update(
        status_path,
        execution_backend="remote",
        remote_profile_id=profile["profile_id"],
        profile_name=profile["name"],
        remote_container_name=container_name,
        controller_pid=os.getpid(),
        local_log_path=str(remote_log),
        controller_log_path=str(controller_log),
        diagnostic_log_paths=list(
            dict.fromkeys((str(remote_log), str(controller_log)))
        ),
        train_log=str(remote_log) if kind == "dinov3" else "",
        log_path=str(remote_log) if kind == "nninteractive" else "",
        remote_gpu_device=profile.get("gpu_device") or "auto",
        status="preparing_remote",
        phase="preparing_remote_data",
        progress_percent=1,
    )
    session: SSHSession | None = None
    remote_status: dict[str, Any] = {}
    prepared: dict[str, Any] = {}
    remote_paths: dict[str, str] = {}
    asset_identity: dict[str, str] = {}
    runtime_image_id = ""
    container_may_exist = False
    try:
        prepared = (
            _prepare_dino(spec, bundle)
            if kind == "dinov3"
            else _prepare_nninteractive(spec, bundle)
        )
        if _cancel_requested(status_path):
            raise InterruptedError("cancel")
        dataset_parts = _build_dataset_parts(bundle, dataset_archive_dir)
        dataset_fingerprint = _dataset_parts_fingerprint(dataset_parts)
        dataset_archive_size = sum(
            int(part["size"]) for part in dataset_parts
        )
        archive_size = _build_tar(
            bundle,
            archive,
            excluded_top_level={"input", "labels"},
        )
        _status_update(
            status_path,
            status="connecting_remote",
            phase="connecting_remote",
            transfer_total_bytes=archive_size + dataset_archive_size,
            dataset_fingerprint=dataset_fingerprint,
            dataset_archive_bytes=dataset_archive_size,
            progress_percent=20,
        )
        _raise_if_cancelled(status_path)
        session = SSHSession(profile)
        _raise_if_cancelled(status_path)
        remote_paths = _remote_paths(session, job_id)
        _validate_remote_job_path(
            profile["remote_root"],
            owner,
            remote_paths["job"],
        )
        session.ensure_directory(remote_paths["jobs"])
        session.ensure_directory(remote_paths["locks"])
        session.ensure_directory(remote_paths["dataset_cache"])
        session.execute(
            "docker container prune -f "
            "--filter label=mimics-script.remote-training=true "
            "--filter {owner} >/dev/null 2>&1 || true".format(
                owner=shlex.quote(
                    "label=mimics-script.owner={}".format(
                        safe_identifier(profile.get("username"), "user")
                    )
                )
            ),
            check=False,
        )
        runtime_image_id = session.execute(
            "docker image inspect --format '{{{{.Id}}}}' {}".format(
                shlex.quote(profile["runtime_image"])
            )
        ).strip()
        asset_identity = _validate_remote_assets(session, remote_paths, prepared)
        _status_update(
            status_path,
            remote_runtime_image=profile["runtime_image"],
            remote_runtime_image_id=runtime_image_id,
            **asset_identity
        )
        # Dataset cache entries are immutable and content-addressed. Job
        # directories are never age-pruned here because a long-running job's
        # parent mtime can remain unchanged while nested status files update.
        session.execute(
            "find {cache} -maxdepth 1 -type f -name '*.tar' -mtime +30 "
            "-delete".format(cache=shlex.quote(remote_paths["dataset_cache"])),
            check=False,
        )
        remote_dataset_archives = []
        dataset_cache_hits = 0
        dataset_bytes_completed = 0
        dataset_reconnect_attempt = 0
        pending_dataset_parts = list(dataset_parts)
        while pending_dataset_parts:
            _raise_if_cancelled(status_path)
            try:
                if session is None:
                    raise RemoteComputeError(
                        "No active SSH dataset-upload session."
                    )
                part = pending_dataset_parts[0]
                remote_dataset_archive, part_cache_hit = (
                    _ensure_remote_dataset_archive(
                        session,
                        remote_paths,
                        profile,
                        Path(part["archive"]),
                        str(part["fingerprint"]),
                        status_path,
                        progress_callback=_upload_progress(
                            status_path,
                            phase="uploading_training_dataset",
                            start_percent=20,
                            end_percent=33,
                            aggregate_offset=dataset_bytes_completed,
                            aggregate_total=dataset_archive_size,
                        ),
                    )
                )
                remote_dataset_archives.append(remote_dataset_archive)
                if part_cache_hit:
                    dataset_cache_hits += 1
                dataset_bytes_completed += int(part["size"])
                pending_dataset_parts.pop(0)
                _status_update(
                    status_path,
                    dataset_fingerprint=dataset_fingerprint,
                    dataset_case_total=len(dataset_parts),
                    dataset_case_completed=len(remote_dataset_archives),
                    dataset_transfer_bytes=dataset_bytes_completed,
                    dataset_transfer_total_bytes=dataset_archive_size,
                    progress_percent=20
                    + int(
                        13
                        * dataset_bytes_completed
                        / max(1, dataset_archive_size)
                    ),
                    dataset_cache_hits=dataset_cache_hits,
                    dataset_cache_hit=(
                        bool(dataset_parts)
                        and dataset_cache_hits == len(dataset_parts)
                    ),
                )
            except RemoteCommandError:
                raise
            except (RemoteComputeError, EOFError, OSError, socket.error) as exc:
                try:
                    session.close()
                except Exception:
                    pass
                session = None
                dataset_reconnect_attempt += 1
                session = _reconnect_session(
                    profile,
                    status_path,
                    controller_log,
                    exc,
                    dataset_reconnect_attempt,
                )
        upload_reconnect_attempt = 0
        while True:
            _raise_if_cancelled(status_path)
            try:
                if session is None:
                    raise RemoteComputeError("No active SSH upload session.")
                session.upload(
                    archive,
                    remote_paths["archive"],
                    callback=_upload_progress(
                        status_path,
                        phase="uploading_job_configuration",
                        start_percent=33,
                        end_percent=35,
                    ),
                )
                break
            except RemoteCommandError:
                raise
            except (RemoteComputeError, EOFError, OSError, socket.error) as exc:
                try:
                    session.close()
                except Exception:
                    pass
                session = None
                upload_reconnect_attempt += 1
                session = _reconnect_session(
                    profile,
                    status_path,
                    controller_log,
                    exc,
                    upload_reconnect_attempt,
                )
        _status_update(
            status_path,
            status="starting_remote",
            phase="starting_remote_container",
            progress_percent=35,
            remote_job_dir=remote_paths["job"],
        )
        launch_reconnect_attempt = 0
        while True:
            _raise_if_cancelled(status_path)
            try:
                if session is None:
                    raise RemoteComputeError("No active SSH launch session.")
                # Refuse a same-name container owned by another user/job before
                # considering this launch capable of creating remote work.
                existing_container = _assert_container_owned(
                    session,
                    container_name,
                    expected_owner=owner,
                    expected_job=job_slug,
                )
                if existing_container:
                    existing_state, _existing_code = _container_state(
                        session, container_name
                    )
                    if existing_state not in {"exited", "dead", "created"}:
                        raise RuntimeError(
                            "This remote training job already has a running "
                            "container. Open its status instead of starting a "
                            "duplicate controller."
                        )
                container_may_exist = True
                container_id = _launch_container(
                    session,
                    remote_paths,
                    profile,
                    container_name,
                    dataset_cache_paths=remote_dataset_archives,
                    remove_dataset_after_extract=not bool(
                        profile.get("cache_training_data", True)
                    ),
                )
                break
            except RemoteCommandError:
                raise
            except (RemoteComputeError, EOFError, OSError, socket.error) as exc:
                try:
                    session.close()
                except Exception:
                    pass
                session = None
                launch_reconnect_attempt += 1
                session = _reconnect_session(
                    profile,
                    status_path,
                    controller_log,
                    exc,
                    launch_reconnect_attempt,
                )
                if session is None:
                    continue
                state, _code = _container_state(session, container_name)
                if state != "missing":
                    _assert_container_owned(
                        session,
                        container_name,
                        expected_owner=owner,
                        expected_job=job_slug,
                    )
                    container_id = session.execute(
                        "docker inspect --format '{{{{.Id}}}}' {}".format(
                            shlex.quote(container_name)
                        )
                    ).strip()
                    break
        _status_update(
            status_path,
            remote_container_id=container_id,
            remote_job_dir=remote_paths["job"],
            remote_gpu_device=profile.get("gpu_device") or "auto",
            dataset_cache_hit=bool(
                dataset_parts
                and dataset_cache_hits == len(dataset_parts)
            ),
            dataset_cache_hits=dataset_cache_hits,
            dataset_case_total=len(dataset_parts),
            remote_dataset_cache_paths=(
                remote_dataset_archives
                if profile.get("cache_training_data", True)
                else []
            ),
            remote_dataset_cache_path=(
                remote_dataset_archives[0]
                if (
                    len(remote_dataset_archives) == 1
                    and profile.get("cache_training_data", True)
                )
                else ""
            ),
            status="starting_remote",
        )
        remote_status_path = str(
            PurePosixPath(remote_paths["job"])
            / PurePosixPath(prepared["remote_status_relative"])
        )
        remote_control_path = (
            str(
                PurePosixPath(remote_paths["job"])
                / PurePosixPath(prepared["remote_control_relative"])
            )
            if prepared.get("remote_control_relative")
            else ""
        )
        remote_control_kind = str(
            prepared.get("remote_control_kind") or ""
        )
        _status_update(
            status_path,
            remote_control_path=remote_control_path,
            remote_control_kind=remote_control_kind,
        )
        missing_status_since = time.time()
        reconnect_attempt = 0
        log_sync_state = {"offset": 0}
        while True:
            try:
                if session is None:
                    raise RemoteComputeError("No active SSH monitoring session.")
                try:
                    _sync_remote_log(
                        session,
                        remote_paths["job"],
                        remote_log,
                        log_sync_state,
                    )
                except IOError:
                    pass
                if _cancel_requested(status_path):
                    _stop_remote(
                        session,
                        container_name,
                        status_path,
                        final="cancelled",
                        remote_control_path=remote_control_path,
                        remote_control_kind=remote_control_kind,
                        remote_job_dir=remote_paths["job"],
                        remote_root=profile["remote_root"],
                        expected_owner=owner,
                        expected_job=job_slug,
                    )
                    return 130
                if _pause_requested(status_path) and kind == "nninteractive":
                    if remote_control_path:
                        temporary = local_root / "pause-control.json"
                        write_json_atomic(
                            temporary,
                            {"action": "pause", "requested_at_epoch": time.time()},
                        )
                        session.upload(temporary, remote_control_path)
                    _status_update(
                        status_path,
                        status="pausing",
                        phase="pausing_remote_training",
                    )
                candidate = session.read_remote_json(remote_status_path, {}) or {}
                if candidate:
                    missing_status_since = time.time()
                    remote_status = candidate
                    merged = _merge_remote_status(
                        status_path,
                        candidate,
                        profile,
                        container_name,
                        remote_paths["job"],
                    )
                    state = str(candidate.get("status") or "").lower()
                    if state in TERMINAL:
                        break
                else:
                    worker = session.read_remote_json(
                        str(
                            PurePosixPath(remote_paths["job"])
                            / "worker_status.json"
                        ),
                        {},
                    ) or {}
                    if str(worker.get("status") or "") == "waiting_for_remote_gpu":
                        missing_status_since = time.time()
                        _status_update(
                            status_path,
                            status="waiting_for_remote_gpu",
                            phase="waiting_for_remote_gpu",
                            progress_percent=35,
                            remote_worker_pid=worker.get("worker_pid"),
                        )
                container_state, container_code = _container_state(
                    session, container_name
                )
                if container_state in {"exited", "dead", "missing"}:
                    worker = session.read_remote_json(
                        str(
                            PurePosixPath(remote_paths["job"])
                            / "worker_status.json"
                        ),
                        {},
                    ) or {}
                    if not remote_status:
                        remote_status = worker
                    if (
                        str(remote_status.get("status") or "").lower()
                        not in TERMINAL
                    ):
                        raise RuntimeError(
                            "Remote training container exited with code {} before "
                            "writing a terminal pipeline status.".format(
                                container_code
                            )
                        )
                    break
                if time.time() - missing_status_since > 300:
                    raise RuntimeError(
                        "Remote training started but did not create its status "
                        "file within five minutes."
                    )
                reconnect_attempt = 0
            except RemoteCommandError as exc:
                _append_log(
                    controller_log,
                    "Remote Docker control is temporarily unavailable: {}".format(
                        exc
                    ),
                )
                _status_update(
                    status_path,
                    status="remote_control_unavailable",
                    phase="remote_control_unavailable",
                    remote_state_unknown=True,
                    remote_connection_error=str(exc),
                )
                deadline = time.time() + 10.0
                while time.time() < deadline:
                    _raise_if_cancelled(status_path)
                    time.sleep(
                        min(0.5, max(0.0, deadline - time.time()))
                    )
                continue
            except (RemoteComputeError, EOFError, OSError, socket.error) as exc:
                try:
                    session.close()
                except Exception:
                    pass
                session = None
                reconnect_attempt += 1
                session = _reconnect_session(
                    profile,
                    status_path,
                    controller_log,
                    exc,
                    reconnect_attempt,
                )
                continue
            time.sleep(2.0)

        final_state = str(remote_status.get("status") or "").lower()
        if final_state != "completed":
            try:
                _sync_remote_log(
                    session,
                    remote_paths["job"],
                    remote_log,
                    log_sync_state,
                )
            except Exception as exc:
                _append_log(
                    controller_log,
                    "Could not synchronize the remote diagnostic log: {}".format(
                        exc
                    ),
                )
            _merge_remote_status(
                status_path,
                remote_status,
                profile,
                container_name,
                remote_paths["job"],
            )
            removed = _remove_container(
                session,
                container_name,
                expected_owner=owner,
                expected_job=job_slug,
            )
            _status_update(
                status_path,
                remote_container_removed=bool(removed),
                remote_job_removed=False,
                remote_diagnostics_retained=True,
            )
            return 0 if final_state in {"cancelled", "paused"} else 1

        if kind == "dinov3":
            import tools.fewshot_pipeline as dino_pipeline

            context = spec["context"]
            organ_slug = dino_pipeline.safe_slug(context["organ"])
            local_model_dir = (
                Path(context["workspace"])
                / "models"
                / organ_slug
                / str(spec["run_id"])
            )
        else:
            local_request = read_json(
                Path(spec["job_dir"]) / "request.json", {}
            ) or {}
            local_model_dir = Path(local_request["output_model_dir"]).resolve()
        _status_update(
            status_path,
            status="downloading",
            phase="downloading_model",
            progress_percent=96,
        )
        download_reconnect_attempt = 0
        while True:
            _raise_if_cancelled(status_path)
            try:
                if session is None:
                    raise RemoteComputeError("No active SSH download session.")
                _sync_remote_log(
                    session,
                    remote_paths["job"],
                    remote_log,
                    log_sync_state,
                )
                _download_artifact(
                    session,
                    remote_paths["job"],
                    prepared["remote_artifact_relative"],
                    local_model_dir,
                    result_archive,
                    status_path,
                )
                break
            except RemoteCommandError:
                raise
            except (RemoteComputeError, EOFError, OSError, socket.error) as exc:
                try:
                    session.close()
                except Exception:
                    pass
                session = None
                download_reconnect_attempt += 1
                session = _reconnect_session(
                    profile,
                    status_path,
                    controller_log,
                    exc,
                    download_reconnect_attempt,
                )
        if kind == "dinov3":
            remote_status.update(asset_identity)
            remote_status["remote_runtime_image_id"] = runtime_image_id
            remote_status["remote_gpu_device"] = profile.get("gpu_device") or "auto"
            remote_status["dataset_fingerprint"] = dataset_fingerprint
            model = _register_dino(
                spec, local_model_dir, remote_status, profile
            )
            selected = True
        else:
            remote_status.update(asset_identity)
            remote_status["remote_runtime_image_id"] = runtime_image_id
            remote_status["remote_gpu_device"] = profile.get("gpu_device") or "auto"
            remote_status["dataset_fingerprint"] = dataset_fingerprint
            model, selected = _register_nninteractive(
                spec, local_model_dir, remote_status, profile
            )
        _status_update(
            status_path,
            status="completed",
            phase="completed",
            progress_percent=100,
            model=model,
            execution_backend="remote",
            remote_model_downloaded=True,
            remote_model_selected=bool(selected),
            completed_at_epoch=time.time(),
            local_log_path=str(remote_log),
            controller_log_path=str(controller_log),
            remote_container_removed=False,
        )
        # Training data and the downloaded archive are no longer needed on the
        # server. The model now lives in the existing local registry.
        container_removed = False
        job_removed = False
        cleanup_warning = ""
        try:
            container_removed = _remove_container(
                session,
                container_name,
                expected_owner=owner,
                expected_job=job_slug,
                stop_if_running=False,
            )
            job_removed = _remove_remote_job(
                session,
                remote_root=profile["remote_root"],
                expected_owner=owner,
                remote_job_dir=remote_paths["job"],
            )
            if not (container_removed and job_removed):
                cleanup_warning = (
                    "Training completed, but some remote temporary files "
                    "could not be removed."
                )
        except Exception as cleanup_exc:
            cleanup_warning = (
                "Training and model registration completed, but remote cleanup "
                "could not be confirmed: {}".format(cleanup_exc)
            )
            _append_log(controller_log, cleanup_warning)
        _status_update(
            status_path,
            remote_container_removed=bool(container_removed),
            remote_job_removed=bool(job_removed),
            remote_cleanup_warning=cleanup_warning,
            remote_dataset_cache_retained=bool(
                profile.get("cache_training_data", True)
                and dataset_fingerprint
            ),
        )
        return 0
    except InterruptedError:
        if session is not None and remote_paths:
            _stop_remote(
                session,
                container_name,
                status_path,
                final="cancelled",
                remote_control_path=str(
                    (read_json(status_path, {}) or {}).get(
                        "remote_control_path"
                    )
                    or ""
                ),
                remote_control_kind=str(
                    (read_json(status_path, {}) or {}).get(
                        "remote_control_kind"
                    )
                    or ""
                ),
                remote_job_dir=str(
                    (read_json(status_path, {}) or {}).get("remote_job_dir")
                    or ""
                ),
                remote_root=profile["remote_root"],
                expected_owner=owner,
                expected_job=job_slug,
            )
        else:
            _status_update(
                status_path,
                status="cancelled",
                phase="cancelled",
                completed_at_epoch=time.time(),
            )
        return 130
    except Exception as exc:
        termination_confirmed = not container_may_exist
        cleanup_error = ""
        if session is not None and remote_paths:
            try:
                _sync_remote_log(
                    session,
                    remote_paths["job"],
                    remote_log,
                    locals().get("log_sync_state", {"offset": 0}),
                )
            except Exception:
                pass
            try:
                removed = _remove_container(
                    session,
                    container_name,
                    expected_owner=owner,
                    expected_job=job_slug,
                )
                termination_confirmed = bool(removed)
                if not termination_confirmed:
                    state, _code = _container_state(session, container_name)
                    termination_confirmed = state in {
                        "exited",
                        "dead",
                        "missing",
                        "created",
                    }
                _status_update(
                    status_path,
                    remote_container_removed=bool(removed),
                    remote_job_removed=False,
                    remote_diagnostics_retained=True,
                )
            except Exception as cleanup_exc:
                cleanup_error = str(cleanup_exc)
                _append_log(
                    controller_log,
                    "Could not confirm remote container cleanup after failure: "
                    "{}".format(cleanup_exc),
                )
        _append_log(
            controller_log,
            "{}: {}\n{}".format(type(exc).__name__, exc, traceback.format_exc()),
        )
        _status_update(
            status_path,
            status="failed" if termination_confirmed else "stopping",
            phase=(
                "remote_failed"
                if termination_confirmed
                else "remote_termination_pending"
            ),
            error=(
                "{}: {}".format(type(exc).__name__, exc)
                if termination_confirmed
                else "Remote training failed, but container termination could "
                "not be confirmed: {}{}".format(
                    exc,
                    " (cleanup: {})".format(cleanup_error)
                    if cleanup_error
                    else "",
                )
            ),
            traceback=traceback.format_exc(),
            execution_backend="remote",
            local_log_path=str(remote_log),
            controller_log_path=str(controller_log),
            diagnostic_log_paths=list(
                dict.fromkeys((str(remote_log), str(controller_log)))
            ),
            completed_at_epoch=time.time() if termination_confirmed else None,
            remote_stop_confirmed=bool(termination_confirmed),
            remote_state_unknown=not bool(termination_confirmed),
        )
        return 1
    finally:
        if session is not None:
            session.close()
        if (
            kind == "dinov3"
            and str((spec.get("options") or {}).get("label_source") or "mcs_refresh")
            == "mcs_refresh"
        ):
            try:
                context = spec.get("context") or {}
                import tools.fewshot_pipeline as cleanup_pipeline

                staging = (
                    Path(context["workspace"]).resolve()
                    / "runs"
                    / cleanup_pipeline.safe_slug(context["organ"])
                    / str(spec["run_id"])
                    / "remote_fresh_labels"
                )
                shutil.rmtree(str(staging), ignore_errors=True)
            except Exception:
                pass
        if kind == "nninteractive":
            try:
                import tools.nninteractive_finetune_pipeline as cleanup_pipeline

                job_dir = Path(spec["job_dir"]).resolve()
                request = read_json(job_dir / "request.json", {}) or {}
                final_status = str(
                    (read_json(status_path, {}) or {}).get("status") or ""
                ).lower()
                keep_failed = bool(
                    cleanup_pipeline.load_config().get(
                        "keep_failed_training_artifacts", False
                    )
                )
                if final_status in {"completed", "cancelled"} or (
                    final_status == "failed" and not keep_failed
                ):
                    cleanup = cleanup_pipeline._cleanup_terminal_artifacts(
                        request,
                        job_dir,
                        remove_partial_model=final_status != "completed",
                        reset_resume_state=final_status == "failed",
                    )
                    _status_update(status_path, artifact_cleanup=cleanup)
            except Exception as exc:
                _append_log(
                    controller_log,
                    "Could not clean local nnInteractive staging data: {}".format(
                        exc
                    ),
                )
        shutil.rmtree(str(bundle), ignore_errors=True)
        shutil.rmtree(str(dataset_archive_dir), ignore_errors=True)
        for path in (archive, result_archive):
            try:
                path.unlink()
            except OSError:
                pass


def cancel(status_path: Path) -> int:
    status = read_json(status_path, {}) or {}
    profile_id = str(status.get("remote_profile_id") or "")
    container_name = str(status.get("remote_container_name") or "")
    remote_control_path = str(status.get("remote_control_path") or "")
    remote_control_kind = str(status.get("remote_control_kind") or "")
    remote_job_dir = str(status.get("remote_job_dir") or "")
    job_id = str(status.get("job_id") or "")
    if not profile_id or not container_name or not job_id:
        raise RuntimeError("This status file does not identify a remote container.")
    profile = get_profile(profile_id)
    owner = _remote_owner(profile)
    expected_container = _container_name(profile, job_id)
    if container_name != expected_container:
        raise RuntimeError(
            "Refusing to stop a remote container that does not match this job."
        )
    _validate_remote_job_path(
        profile["remote_root"],
        owner,
        remote_job_dir,
    )
    _validate_remote_control_path(remote_job_dir, remote_control_path)
    try:
        with SSHSession(profile) as session:
            _stop_remote(
                session,
                container_name,
                status_path,
                final="cancelled",
                remote_control_path=remote_control_path,
                remote_control_kind=remote_control_kind,
                remote_job_dir=remote_job_dir,
                remote_root=profile["remote_root"],
                expected_owner=owner,
                expected_job=safe_identifier(job_id, "job"),
            )
        return 0
    except Exception as exc:
        _status_update(
            status_path,
            status="stopping",
            phase="remote_termination_pending",
            remote_stop_confirmed=False,
            error="Remote stop could not be confirmed: {}".format(exc),
        )
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--spec", required=True)
    cancel_parser = sub.add_parser("cancel")
    cancel_parser.add_argument("--status", required=True)
    args = parser.parse_args()
    if args.command == "cancel":
        return cancel(Path(args.status).resolve())
    return run(Path(args.spec).resolve())


if __name__ == "__main__":
    raise SystemExit(main())
