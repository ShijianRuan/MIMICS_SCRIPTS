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
import hashlib
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


def _require_checkpoint_loadable(model_dir: Path, family: str) -> None:
    """Validate serialized checkpoint data without running package code.

    This reads checkpoint tensors on CPU; it is not a network or inference
    compatibility check. Never fall back to unrestricted pickle loading.
    """
    patterns = {
        "nninteractive_task": ("fold_*/checkpoint_final.pth",),
        "nnunet": ("fold_*/checkpoint_final.pth",),
        "flexict": (
            "fold_*/checkpoint_best.pth",
            "fold_*/checkpoint_final.pth",
        ),
    }
    names = patterns.get(family)
    if not names:
        return
    checkpoints: list[Path] = []
    for pattern in names:
        checkpoints.extend(sorted(model_dir.glob(pattern)))
    if not checkpoints:
        # Missing files are already reported by the family-specific audit
        # (audit_model_dir / manifest checks) with a clearer message.
        return
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch ships with the env
        raise RuntimeError("模型尚未导入：当前环境缺少 PyTorch，无法安全校验权重。") from exc
    # Standard nnU-Net logging/checkpoint metadata contains NumPy numeric
    # arrays/scalars. Allow these specific data constructors, never globals
    # discovered from the untrusted checkpoint itself or custom model code.
    import numpy as np
    multiarray = np._core.multiarray if hasattr(np, "_core") else np.core.multiarray
    numeric_dtypes = [np.dtype(code) for code in ("?", "i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f2", "f4", "f8", "c8", "c16")]
    allowed = [np.ndarray, np.dtype, multiarray._reconstruct, multiarray.scalar]
    allowed += list({type(dtype) for dtype in numeric_dtypes})
    # NumPy 1.x and 2.x use different module paths for the same data types.
    allowed += [(multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
                (multiarray.scalar, "numpy.core.multiarray.scalar")]
    for checkpoint in checkpoints:
        try:
            with torch.serialization.safe_globals(allowed):
                torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        except Exception as exc:
            raise RuntimeError(
                "The packaged checkpoint {} is not a loadable torch file under restricted deserialization "
                "({}). 模型尚未导入：文件损坏、PyTorch 版本不兼容或包含未支持的自定义对象；"
                "请由可信模型提供者重新导出纯数据权重，不要改用不受限的 Pickle 加载。".format(checkpoint.name, exc)
            ) from exc


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
    backbone: bool = False,
) -> Path:
    output = output.expanduser().resolve()
    if output.suffix.lower() != ".zip":
        output = output.with_suffix(".zip")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(
        "{}.{}.tmp".format(output.name, uuid.uuid4().hex)
    )
    if backbone:
        # Checksums first, so the header is written once with the final
        # metadata (a second writestr for bundle.json would leave two
        # entries in the archive).
        metadata = dict(metadata)
        metadata["backbone"] = _backbone_section()
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
            if backbone:
                for relative, source in _flexict_backbone_files():
                    archive.write(
                        str(source), str(Path("backbone") / relative)
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
        _require_checkpoint_loadable(staging / "model", "nninteractive_task")
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
        _require_checkpoint_loadable(staged_model_dir, "nnunet")
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


def _flexict_backbone_files() -> list[tuple[str, Path]]:
    """The FlexiCT backbone weights that a fine-tuned model still depends on.

    F06/G10: FlexiCT fine-tuned checkpoints carry the full network, but the
    trainer unconditionally loads the pretrained backbone while *building*
    the architecture, so a workstation without the backbone files cannot
    even run inference with an imported model. The two safetensors are the
    complete external dependency; everything else is in the model folder.
    """
    from tools import flexict_common

    weights_dir, _source = flexict_common.resolve_pretrained_dir()
    return [
        ("flexict_2d/model.safetensors", weights_dir / "flexict_2d" / "model.safetensors"),
        ("flexict_3d/model.safetensors", weights_dir / "flexict_3d" / "model.safetensors"),
    ]


def _target_backbone_dir() -> tuple[Path, str]:
    """Where the *importing* workstation's backbone weights live, if any."""
    from tools import flexict_common

    return flexict_common.resolve_pretrained_dir()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _backbone_section() -> dict[str, Any]:
    """Checksums for the FlexiCT backbone weights (computed before zipping)."""
    section: dict[str, Any] = {
        "schema_version": "mimics_ai_model_bundle_backbone.v1",
        "files": [],
    }
    for relative, source in _flexict_backbone_files():
        section["files"].append(
            {
                "path": relative,
                "sha256": _sha256_file(source),
                "size": int(source.stat().st_size),
            }
        )
    return section


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
    metadata: dict[str, Any] = {
        "schema_version": "mimics_ai_model_bundle.v1",
        "model_family": "flexict",
        "created_at_epoch": time.time(),
        "pair_id": pair_id,
        "models": usable_models,
    }
    # D15: the fine-tuned checkpoints are self-contained, but the trainer
    # loads the pretrained backbone while building the network (F06), so a
    # target workstation without the backbone files cannot run inference.
    # --include-backbone adds the two safetensors (2 x ~576 MB) under
    # backbone/; import places them where resolve_pretrained_dir() looks.
    include_backbone = bool(getattr(args, "include_backbone", False))
    if include_backbone:
        # Probe before any zip is written: a workstation without the backbone
        # files fails here instead of producing an unusable --include-backbone
        # bundle (resolve_pretrained_dir raises FileNotFoundError).
        _flexict_backbone_files()
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
        output = _write_bundle(
            Path(args.output),
            metadata,
            portable_root,
            backbone=True if include_backbone else False,
        )
    print(str(output))
    return 0


def _import_flexict_backbone(
    staging: Path, metadata: dict[str, Any], workspace: Path
) -> str:
    """Place packaged backbone weights where resolve_pretrained_dir() looks.

    D15: without the backbone files the imported fine-tuned model cannot run
    (the trainer loads the backbone while building the network, F06), so an
    importer on a fresh workstation would import successfully and only fail
    at prediction time. Placement policy:

    - Target already has the backbone: verify checksums match, keep the
      existing files (an admin-managed shared location is never overwritten
      by a bundle import). A checksum mismatch is a hard error — training
      with a different backbone than the model was fine-tuned from would
      silently change the recipe.
    - Target has none: install into a config override directory (workspace/
      flexict_pretrained) and point pretrained_weights_dir at it. The repo
      weights/ folder itself is never written: it may be a read-only
      distribution share and is shared by every task.
    """
    from tools import flexict_common

    section = metadata.get("backbone") or {}
    files = list(section.get("files") or [])
    staged_backbone = staging / "backbone"
    if not files or not staged_backbone.is_dir():
        return ""
    for row in files:
        staged_file = staged_backbone / str(row.get("path") or "")
        if not staged_file.is_file():
            raise RuntimeError(
                "The package is missing the backbone file {}.".format(
                    row.get("path")
                )
            )
        if _sha256_file(staged_file) != str(row.get("sha256") or ""):
            raise RuntimeError(
                "The packaged backbone file {} is corrupt (checksum "
                "mismatch).".format(row.get("path"))
            )
    try:
        existing_dir, _source = _target_backbone_dir()
        have_backbone = True
    except FileNotFoundError:
        existing_dir = None
        have_backbone = False
    if have_backbone:
        for row in files:
            target = existing_dir / str(row.get("path") or "")
            if _sha256_file(target) != str(row.get("sha256") or ""):
                raise RuntimeError(
                    "The target workstation already has FlexiCT backbone "
                    "weights at {} that differ from the packaged ones. The "
                    "imported model was fine-tuned from the packaged "
                    "backbone; keep the two together or move the existing "
                    "weights aside.".format(target)
                )
        return "verified_existing"
    config = flexict_common.load_config()
    destination = workspace / "flexict_pretrained"
    destination.mkdir(parents=True, exist_ok=True)
    for row in files:
        target = destination / str(row.get("path") or "")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(staged_backbone / str(row.get("path") or "")), str(target))
    config["pretrained_weights_dir"] = str(destination)
    flexict_common.save_config(config)
    return "installed"


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
        backbone_action = _import_flexict_backbone(staging, metadata, workspace)
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
                _require_checkpoint_loadable(staged, "flexict")
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
    if backbone_action:
        print(
            "FlexiCT backbone weights: {}".format(
                "verified against the existing installation."
                if backbone_action == "verified_existing"
                else "installed; pretrained_weights_dir was set."
            )
        )
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
    export_fx.add_argument(
        "--include-backbone",
        action="store_true",
        help="Also package the FlexiCT pretrained backbone weights "
        "(2 x ~576 MB) so a workstation without them can run the model.",
    )
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
