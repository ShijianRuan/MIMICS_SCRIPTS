#!/usr/bin/env python3
"""Export and import portable DINOv3 and nnInteractive task-model bundles."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from tools.fewshot_pipeline import (
    load_repo_config,
    portable_inference_config,
    register_global_model,
    resolve_path,
    resolved_manifest_payload,
    safe_slug as fewshot_slug,
    validate_model_manifest,
)
from tools.nninteractive_task_common import (
    audit_model_dir,
    find_task,
    load_registry,
    model_is_usable,
    relative_model_path,
    resolve_registered_model_dir,
    safe_slug,
    save_registry,
    task_dir,
    write_json_atomic,
)


def _require_nninteractive_runtime_identity(
    audit: dict[str, Any], model: dict[str, Any]
) -> str:
    if not model_is_usable(model):
        raise RuntimeError("The selected task model is not usable.")
    if not audit.get("runtime_verified"):
        raise RuntimeError(
            "This task model predates effective runtime verification. Re-export "
            "it through the current training pipeline before moving it between "
            "workspaces."
        )
    fingerprint = str(audit.get("effective_model_fingerprint") or "")
    registered = str(model.get("effective_model_fingerprint") or "")
    if not fingerprint or (registered and registered != fingerprint):
        raise RuntimeError(
            "The task model effective parameter fingerprint does not match its "
            "registered identity."
        )
    return fingerprint


def _write_bundle(
    output: Path,
    metadata: dict[str, Any],
    model_dir: Path,
) -> Path:
    output = output.expanduser().resolve()
    if output.suffix.lower() != ".zip":
        output = output.with_suffix(".zip")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(
        "{}.{}.tmp".format(output.name, uuid.uuid4().hex)
    )
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            archive.writestr(
                "bundle.json",
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            )
            for source in sorted(model_dir.rglob("*")):
                if source.is_file():
                    archive.write(
                        source,
                        str(Path("model") / source.relative_to(model_dir)),
                    )
        os.replace(str(temporary), str(output))
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass
    return output


def _extract_bundle(bundle: Path, destination: Path) -> dict[str, Any]:
    with zipfile.ZipFile(bundle, "r") as archive:
        for member in archive.infolist():
            relative = Path(member.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(
                    "The model package contains an unsafe path: {}".format(
                        member.filename
                    )
                )
        archive.extractall(destination)
    metadata_path = destination / "bundle.json"
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError("The model package metadata is invalid: {}".format(exc))
    if not isinstance(payload, dict) or not (destination / "model").is_dir():
        raise RuntimeError("The model package is incomplete.")
    return payload


def export_nninteractive(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).expanduser().resolve()
    task = find_task(workspace, args.task_id)
    if not task:
        raise RuntimeError("Task was not found: {}".format(args.task_id))
    model_id = str(args.model_id or task.get("recommended_model_id") or "")
    model = next(
        (
            row
            for row in task.get("models") or []
            if str(row.get("model_id") or "") == model_id
        ),
        None,
    )
    if not model:
        raise RuntimeError("Model was not found: {}".format(model_id or "(empty)"))
    model_dir = resolve_registered_model_dir(workspace, model, task)
    audit = audit_model_dir(model_dir)
    if not audit.get("compatible"):
        raise RuntimeError(
            "The selected model is incomplete: {}".format(
                ", ".join(audit.get("missing") or [])
            )
        )
    effective_fingerprint = _require_nninteractive_runtime_identity(audit, model)
    portable_model = dict(model)
    portable_model.pop("model_dir", None)
    portable_model["model_relpath"] = "."
    portable_model["runtime_verified"] = True
    portable_model["effective_model_fingerprint"] = effective_fingerprint
    metadata = {
        "schema_version": "mimics_ai_model_bundle.v1",
        "model_family": "nninteractive_task",
        "created_at_epoch": time.time(),
        "task": {
            "task_id": task.get("task_id"),
            "task_name": task.get("task_name"),
            "mask_names": list(task.get("mask_names") or []),
        },
        "model": portable_model,
        "checkpoint_sha256": audit.get("checkpoint_sha256"),
        "effective_model_fingerprint": effective_fingerprint,
    }
    with tempfile.TemporaryDirectory(prefix="mimics_nnint_export_") as raw:
        portable_dir = Path(raw) / "model"
        shutil.copytree(model_dir, portable_dir)
        finetune_path = portable_dir / "finetune_manifest.json"
        if finetune_path.is_file():
            finetune = json.loads(finetune_path.read_text(encoding="utf-8"))
            finetune["model_dir"] = "."
            finetune["checkpoint"] = "fold_{}/checkpoint_final.pth".format(
                audit.get("fold", "0")
            )
            finetune["base_model_id"] = str(
                model.get("parent_model_id") or "official"
            )
            finetune.pop("base_model_dir", None)
            finetune.pop("base_checkpoint", None)
            write_json_atomic(finetune_path, finetune)
        output = _write_bundle(Path(args.output), metadata, portable_dir)
    print(str(output))
    return 0


def import_nninteractive(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mimics_nnint_import_") as raw:
        staging = Path(raw)
        metadata = _extract_bundle(Path(args.bundle).expanduser().resolve(), staging)
        if metadata.get("model_family") != "nninteractive_task":
            raise RuntimeError("This is not an nnInteractive task-model package.")
        task_meta = dict(metadata.get("task") or {})
        model = dict(metadata.get("model") or {})
        task_id = safe_slug(task_meta.get("task_id"))
        model_id = safe_slug(model.get("model_id"), fallback="model")
        audit = audit_model_dir(staging / "model")
        if not audit.get("compatible"):
            raise RuntimeError(
                "The imported model is incomplete: {}".format(
                    ", ".join(audit.get("missing") or [])
                )
            )
        effective_fingerprint = _require_nninteractive_runtime_identity(
            audit, model
        )
        packaged_fingerprint = str(
            metadata.get("effective_model_fingerprint")
            or model.get("effective_model_fingerprint")
            or ""
        )
        if packaged_fingerprint and packaged_fingerprint != effective_fingerprint:
            raise RuntimeError(
                "The imported model runtime fingerprint does not match the package."
            )
        expected = str(
            metadata.get("checkpoint_sha256")
            or model.get("checkpoint_sha256")
            or ""
        ).lower()
        if expected and expected != str(audit.get("checkpoint_sha256") or "").lower():
            raise RuntimeError("The imported checkpoint checksum does not match.")
        destination = task_dir(workspace, task_id) / "models" / model_id
        if destination.exists():
            raise RuntimeError(
                "Model already exists in the target workspace: {}".format(
                    destination
                )
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        publishing = destination.with_name(
            "{}.importing_{}".format(destination.name, uuid.uuid4().hex)
        )
        published = False
        try:
            shutil.copytree(staging / "model", publishing)
            os.replace(str(publishing), str(destination))
            published = True
            registry = load_registry(workspace)
            tasks = registry.get("tasks") or []
            task = next(
                (
                    row
                    for row in tasks
                    if safe_slug(row.get("task_id")) == task_id
                ),
                None,
            )
            if task is None:
                task = {
                    "task_id": task_id,
                    "task_name": str(task_meta.get("task_name") or task_id),
                    "mask_names": list(task_meta.get("mask_names") or []),
                    "models": [],
                }
                tasks.append(task)
            model["model_id"] = model_id
            model["model_relpath"] = relative_model_path(workspace, destination)
            model.pop("model_dir", None)
            model["compatible"] = True
            model["checkpoint_sha256"] = audit.get("checkpoint_sha256")
            model["runtime_verified"] = True
            model["effective_model_fingerprint"] = effective_fingerprint
            model["validated_prompt_types"] = list(
                model.get("validated_prompt_types") or ["point"]
            )
            task["models"] = [
                row
                for row in task.get("models") or []
                if str(row.get("model_id") or "") != model_id
            ] + [model]
            if args.set_current or not task.get("recommended_model_id"):
                task["recommended_model_id"] = model_id
            task["updated_at_epoch"] = time.time()
            registry["tasks"] = tasks
            write_json_atomic(task_dir(workspace, task_id) / "task.json", task)
            save_registry(workspace, registry)
        except Exception:
            shutil.rmtree(
                destination if published else publishing,
                ignore_errors=True,
            )
            raise
    print(str(destination))
    return 0


def export_dinov3(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = validate_model_manifest(
        resolved_manifest_payload(manifest_path), manifest_path
    )
    model_dir = manifest_path.parent
    portable = {
        key: value
        for key, value in manifest.items()
        if not str(key).startswith("_")
    }
    portable["checkpoint"] = Path(manifest["checkpoint"]).name
    portable["config"] = Path(manifest["config"]).name
    portable.pop("ts_root", None)
    portable.pop("workspace", None)
    portable.pop("dataset_dir", None)
    portable.pop("experiment_dir", None)
    portable.pop("dinov3_root", None)
    portable.pop("base_config", None)
    portable.pop("training_source_checkpoint", None)
    portable.pop("metrics_history", None)
    portable.pop("samples", None)
    portable.pop("train_samples", None)
    portable.pop("validation_samples", None)
    portable.pop("fresh_label_cleanup", None)
    portable["source_checkpoint"] = Path(manifest["checkpoint"]).name
    training_parameters = dict(portable.get("training_parameters") or {})
    for key in ("base_config", "model_path", "label_root"):
        training_parameters.pop(key, None)
    portable["training_parameters"] = training_parameters
    metadata = {
        "schema_version": "mimics_ai_model_bundle.v1",
        "model_family": "dinov3_fewshot",
        "created_at_epoch": time.time(),
        "manifest": portable,
    }
    with tempfile.TemporaryDirectory(prefix="mimics_dinov3_export_") as raw:
        portable_dir = Path(raw) / "model"
        portable_dir.mkdir(parents=True)
        shutil.copy2(manifest["checkpoint"], portable_dir / "model.pth")
        dino_root = Path(
            manifest.get("dinov3_root")
            or os.environ.get("MIMICS_FEWSHOT_DINOV3_ROOT")
            or load_repo_config().get("dinov3_project")
            or (ROOT / "integrations" / "dinov3-medical-seg")
        ).expanduser().resolve()
        if str(dino_root) not in os.sys.path:
            os.sys.path.insert(0, str(dino_root))
        try:
            from src.utils.config import load_config as load_dinov3_config

            effective = load_dinov3_config(str(manifest["config"]))
        except Exception as exc:
            raise RuntimeError(
                "The DINOv3 inference configuration could not be made "
                "portable: {}".format(exc)
            )
        import yaml

        (portable_dir / "config.yaml").write_text(
            yaml.safe_dump(
                portable_inference_config(effective, dino_root),
                sort_keys=False,
                allow_unicode=False,
            ),
            encoding="utf-8",
        )
        portable["checkpoint"] = "model.pth"
        portable["config"] = "config.yaml"
        portable["config_sha256"] = __import__("hashlib").sha256(
            (portable_dir / "config.yaml").read_bytes()
        ).hexdigest()
        write_json_atomic(portable_dir / "manifest.json", portable)
        metadata["manifest"] = portable
        output = _write_bundle(Path(args.output), metadata, portable_dir)
    print(str(output))
    return 0


def import_dinov3(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).expanduser().resolve()
    with tempfile.TemporaryDirectory(prefix="mimics_dinov3_import_") as raw:
        staging = Path(raw)
        metadata = _extract_bundle(Path(args.bundle).expanduser().resolve(), staging)
        if metadata.get("model_family") != "dinov3_fewshot":
            raise RuntimeError("This is not a DINOv3 model package.")
        manifest = dict(metadata.get("manifest") or {})
        staged_manifest = staging / "model" / "manifest.json"
        write_json_atomic(staged_manifest, manifest)
        checked = validate_model_manifest(manifest, staged_manifest)
        organ_slug = fewshot_slug(
            manifest.get("organ_slug")
            or manifest.get("organ")
            or "target"
        )
        model_id = fewshot_slug(manifest.get("model_id") or "model")
        destination = workspace / "models" / organ_slug / model_id
        if destination.exists():
            raise RuntimeError(
                "Model already exists in the target workspace: {}".format(
                    destination
                )
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        publishing = destination.with_name(
            "{}.importing_{}".format(destination.name, uuid.uuid4().hex)
        )
        published = False
        try:
            shutil.copytree(staging / "model", publishing)
            manifest["checkpoint"] = Path(checked["checkpoint"]).name
            manifest["config"] = Path(checked["config"]).name
            write_json_atomic(publishing / "manifest.json", manifest)
            os.replace(str(publishing), str(destination))
            published = True
            manifest_path = destination / "manifest.json"
            if args.set_latest:
                latest = dict(manifest)
                latest["checkpoint"] = "{}/{}".format(
                    model_id, manifest["checkpoint"]
                )
                latest["config"] = "{}/{}".format(model_id, manifest["config"])
                latest["model_manifest"] = "{}/manifest.json".format(model_id)
                write_json_atomic(destination.parent / "latest.json", latest)
        except Exception:
            shutil.rmtree(
                destination if published else publishing,
                ignore_errors=True,
            )
            raise
        try:
            register_global_model(manifest, manifest_path)
        except Exception as exc:
            print(
                "Warning: the model was imported into this dataset, but the "
                "cross-dataset index could not be updated: {}".format(exc)
            )
    print(str(destination))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    export_nn = sub.add_parser("export-nninteractive")
    export_nn.add_argument("--workspace", required=True)
    export_nn.add_argument("--task-id", required=True)
    export_nn.add_argument("--model-id")
    export_nn.add_argument("--output", required=True)
    export_nn.set_defaults(func=export_nninteractive)

    import_nn = sub.add_parser("import-nninteractive")
    import_nn.add_argument("--workspace", required=True)
    import_nn.add_argument("--bundle", required=True)
    import_nn.add_argument("--set-current", action="store_true")
    import_nn.set_defaults(func=import_nninteractive)

    export_dino = sub.add_parser("export-dinov3")
    export_dino.add_argument("--manifest", required=True)
    export_dino.add_argument("--output", required=True)
    export_dino.set_defaults(func=export_dinov3)

    import_dino = sub.add_parser("import-dinov3")
    import_dino.add_argument("--workspace", required=True)
    import_dino.add_argument("--bundle", required=True)
    import_dino.add_argument("--set-latest", action="store_true")
    import_dino.set_defaults(func=import_dinov3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
