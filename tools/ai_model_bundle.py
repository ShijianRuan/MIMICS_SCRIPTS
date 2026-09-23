#!/usr/bin/env python3
"""Export and import portable model bundles for all three model families.

nnInteractive task models (nninteractive_task), managed nnU-Net models
(nnunet), and FlexiCT 2D/3D models (flexict, including active-learning
pairs) are packaged as self-describing zip files with a bundle.json
metadata header and the model folder under ``model/``. Import validates
the required files, relocates the folder into the target workspace, and
registers it in the existing per-family registry so the normal
prediction/model-selection UI finds it.
"""

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

try:
    from tools.nninteractive_task_common import (  # type: ignore
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
except ImportError:  # direct tools/ import (tests, smoke runs)
    from nninteractive_task_common import (  # type: ignore
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
            model["imported_from_bundle"] = True
            model["imported_at_epoch"] = time.time()
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


# ---------------------------------------------------------------------------
# Managed nnU-Net models
# ---------------------------------------------------------------------------


def _nnunet_model_manifest(workspace: Path, model_id: str) -> dict[str, Any]:
    from tools import nnunet_common

    for model in nnunet_common.load_models(workspace, include_missing=True):
        if str(model.get("model_id") or "") == model_id:
            return model
    raise RuntimeError("Model was not found: {}".format(model_id))


def export_nnunet(args: argparse.Namespace) -> int:
    from tools import nnunet_common

    workspace = Path(args.workspace).expanduser().resolve()
    model = _nnunet_model_manifest(workspace, args.model_id)
    model_dir = Path(str(model.get("model_dir") or "")).expanduser()
    usable, reason = nnunet_common.model_usability(model)
    if not usable:
        raise RuntimeError("The selected model is not usable: {}".format(reason))
    portable = dict(model)
    manifest_path = Path(
        str(model.get("manifest_path") or model_dir / "mimics_model_manifest.json")
    )
    metadata = {
        "schema_version": "mimics_ai_model_bundle.v1",
        "model_family": "nnunet",
        "created_at_epoch": time.time(),
        "model": portable,
    }
    with tempfile.TemporaryDirectory(prefix="mimics_nnunet_export_") as raw:
        portable_dir = Path(raw) / "model"
        shutil.copytree(model_dir, portable_dir)
        # Rewrite the manifest so absolute model_dir/manifest_path do not
        # leak the exporting machine's paths; import re-points them.
        manifest = dict(
            json.loads((manifest_path).read_text(encoding="utf-8"))
            if manifest_path.is_file()
            else portable
        )
        manifest.pop("model_dir", None)
        manifest.pop("manifest_path", None)
        (portable_dir / "mimics_model_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output = _write_bundle(Path(args.output), metadata, portable_dir)
    print(str(output))
    return 0


def import_nnunet(args: argparse.Namespace) -> int:
    from tools import nnunet_common

    workspace = Path(args.workspace).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mimics_nnunet_import_") as raw:
        staging = Path(raw)
        metadata = _extract_bundle(Path(args.bundle).expanduser().resolve(), staging)
        if metadata.get("model_family") != "nnunet":
            raise RuntimeError("This is not a managed nnU-Net model package.")
        model = dict(metadata.get("model") or {})
        model_id = str(model.get("model_id") or "").strip()
        if not model_id:
            raise RuntimeError("The model package has no model_id.")
        staged_model_dir = staging / "model"
        manifest_path = staged_model_dir / "mimics_model_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != nnunet_common.MODEL_SCHEMA_VERSION:
            raise RuntimeError("The packaged nnU-Net model manifest is invalid.")
        task_slug = nnunet_common.safe_identifier(
            manifest.get("task_id") or "task", "task"
        )
        destination = (
            nnunet_common.workspace_paths(workspace)["models"]
            / task_slug
            / nnunet_common.safe_identifier(model_id, "model")
        )
        if destination.exists():
            raise RuntimeError(
                "Model already exists in the target workspace: {}".format(destination)
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        publishing = destination.with_name(
            "{}.importing_{}".format(destination.name, uuid.uuid4().hex)
        )
        published = False
        try:
            shutil.copytree(staged_model_dir, publishing)
            os.replace(str(publishing), str(destination))
            published = True
            manifest["model_dir"] = str(destination)
            manifest["manifest_path"] = str(destination / "mimics_model_manifest.json")
            manifest["imported_from_bundle"] = True
            manifest["imported_at_epoch"] = time.time()
            nnunet_common.write_json_atomic(
                destination / "mimics_model_manifest.json", manifest
            )
            nnunet_common.register_model(workspace, manifest)
        except Exception:
            shutil.rmtree(
                destination if published else publishing, ignore_errors=True
            )
            raise
    print(str(destination))
    return 0


# ---------------------------------------------------------------------------
# FlexiCT models (single or 2D+3D active-learning pair)
# ---------------------------------------------------------------------------


def _flexict_model_or_pair(
    workspace: Path, model_id: str
) -> list[dict[str, Any]]:
    from tools import flexict_common

    models = flexict_common.load_models(workspace, include_missing=True)
    selected = [
        row for row in models if str(row.get("model_id") or "") == model_id
    ]
    if not selected:
        raise RuntimeError("Model was not found: {}".format(model_id))
    row = selected[0]
    pair_id = str(row.get("pair_id") or "")
    if pair_id:
        pair = [r for r in models if str(r.get("pair_id") or "") == pair_id]
        if len(pair) >= 2:
            # Keep the requested model plus its pair sibling(s).
            return sorted(pair, key=lambda r: str(r.get("configuration")))
    return [row]


def export_flexict(args: argparse.Namespace) -> int:
    from tools import flexict_common

    workspace = Path(args.workspace).expanduser().resolve()
    models = _flexict_model_or_pair(workspace, args.model_id)
    usable_models = []
    for model in models:
        usable, reason = flexict_common.model_usability(model)
        if not usable:
            raise RuntimeError(
                "FlexiCT model {} is not usable: {}".format(
                    model.get("model_id"), reason
                )
            )
        usable_models.append(dict(model))
    pair_id = str(usable_models[0].get("pair_id") or "")
    metadata = {
        "schema_version": "mimics_ai_model_bundle.v1",
        "model_family": "flexict",
        "created_at_epoch": time.time(),
        "pair_id": pair_id,
        "models": usable_models,
    }
    with tempfile.TemporaryDirectory(prefix="mimics_flexict_export_") as raw:
        portable_root = Path(raw) / "model"
        portable_root.mkdir()
        for model in usable_models:
            source_dir = Path(str(model.get("model_dir") or "")).expanduser()
            if not source_dir.is_dir():
                raise RuntimeError(
                    "FlexiCT model folder is missing: {}".format(source_dir)
                )
            slot = "{}_{}".format(
                model.get("configuration") or "model", model.get("model_id")
            )
            shutil.copytree(source_dir, portable_root / slot)
            # The per-model manifest keeps relative identity; import re-points.
            manifest = dict(model)
            manifest.pop("model_dir", None)
            manifest.pop("manifest_path", None)
            (portable_root / slot / "flexict_model_manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        output = _write_bundle(Path(args.output), metadata, portable_root)
    print(str(output))
    return 0


def import_flexict(args: argparse.Namespace) -> int:
    from tools import flexict_common
    from tools.nnunet_common import safe_identifier, write_json_atomic

    workspace = Path(args.workspace).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mimics_flexict_import_") as raw:
        staging = Path(raw)
        metadata = _extract_bundle(Path(args.bundle).expanduser().resolve(), staging)
        if metadata.get("model_family") != "flexict":
            raise RuntimeError("This is not a FlexiCT model package.")
        packaged = list(metadata.get("models") or [])
        if not packaged:
            raise RuntimeError("The FlexiCT model package contains no models.")
        models_root = flexict_common.workspace_paths(workspace)["root"] / "models"
        published: list[tuple[Path, dict[str, Any]]] = []
        try:
            for model in packaged:
                model = dict(model)
                model_id = str(model.get("model_id") or "").strip()
                configuration = str(model.get("configuration") or "model")
                if not model_id:
                    raise RuntimeError("A packaged FlexiCT model has no model_id.")
                staged = staging / "model" / "{}_{}".format(configuration, model_id)
                if not staged.is_dir():
                    raise RuntimeError(
                        "The package is missing the model folder for {}.".format(
                            model_id
                        )
                    )
                task_slug = safe_identifier(
                    model.get("task_id") or "task", "task"
                )
                destination = models_root / task_slug / model_id
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
                shutil.copytree(staged, publishing)
                os.replace(str(publishing), str(destination))
                manifest = dict(model)
                manifest["model_dir"] = str(destination)
                manifest["manifest_path"] = str(
                    destination / "flexict_model_manifest.json"
                )
                manifest["imported_from_bundle"] = True
                manifest["imported_at_epoch"] = time.time()
                manifest.pop("model_relpath", None)
                write_json_atomic(
                    destination / "flexict_model_manifest.json", manifest
                )
                published.append((destination, manifest))
            for _destination, manifest in published:
                flexict_common.register_model(manifest, workspace=workspace)
        except Exception:
            for destination, _manifest in published:
                shutil.rmtree(destination, ignore_errors=True)
            raise
    if args.set_current and published:
        flexict_common.save_registry(
            workspace,
            None,
            flexict_common.load_models(workspace),
            recommended_model_id=str(published[0][1].get("model_id")),
        )
    print(str(published[0][0]))
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

    export_unet = sub.add_parser("export-nnunet")
    export_unet.add_argument("--workspace", required=True)
    export_unet.add_argument("--model-id", required=True)
    export_unet.add_argument("--output", required=True)
    export_unet.set_defaults(func=export_nnunet)

    import_unet = sub.add_parser("import-nnunet")
    import_unet.add_argument("--workspace", required=True)
    import_unet.add_argument("--bundle", required=True)
    import_unet.set_defaults(func=import_nnunet)

    export_fx = sub.add_parser("export-flexict")
    export_fx.add_argument("--workspace", required=True)
    export_fx.add_argument("--model-id", required=True)
    export_fx.add_argument("--output", required=True)
    export_fx.set_defaults(func=export_flexict)

    import_fx = sub.add_parser("import-flexict")
    import_fx.add_argument("--workspace", required=True)
    import_fx.add_argument("--bundle", required=True)
    import_fx.add_argument("--set-current", action="store_true")
    import_fx.set_defaults(func=import_flexict)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
