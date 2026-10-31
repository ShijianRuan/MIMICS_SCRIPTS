#!/usr/bin/env python3
"""Import external nnU-Net v2 training results (native directory or ZIP).

External models trained elsewhere are imported read-only first: the scanner
never executes package code or unrestricted pickles, ZIP archives are
extracted under strict limits, and a failed or cancelled import never leaves
a partial registration. A validated model is published under
``<workspace>/external_models/<task_slug>/<model_id>/`` with a
``mimics_nnunet_external_model.v1`` manifest, preserving the
``Trainer__Plans__Configuration`` folder name (inference parses the
configuration from it). ``nnunet_common.load_models`` picks these manifests
up for the prediction window; the live model registry JSON stays untouched
(it only records models trained on this machine).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from tools.nnunet_common import (  # type: ignore
        read_json,
        stable_task_identifier,
        workspace_paths,
        write_json_atomic,
    )
except ImportError:  # direct tools/ import (tests, smoke runs)
    from nnunet_common import (  # type: ignore
        read_json,
        stable_task_identifier,
        workspace_paths,
        write_json_atomic,
    )

EXTERNAL_SCHEMA_VERSION = "mimics_nnunet_external_model.v1"
EXTERNAL_MANIFEST_NAME = "external_model_manifest.json"
CHECKPOINT_NAME = "checkpoint_final.pth"
# 外部训练结果常在中断后交付，只有验证集最优权重（nnU-Net 官方推理支持
# -chk checkpoint_best.pth）；导入时按 fold 实际持有情况回退到它。
FALLBACK_CHECKPOINT_NAME = "checkpoint_best.pth"
MAX_ZIP_FILES = 50000
MAX_ZIP_TOTAL_BYTES = 32 * 1024 ** 3
MAX_ZIP_COMPRESSION_RATIO = 200
MAX_SCAN_DIRS = 5000
MAX_SCAN_DEPTH = 6
MAX_CONFIG_CANDIDATES = 32
MAX_REJECTED_SAMPLE = 20
_DIGEST_CHUNK = 1024 * 1024
_NUMERIC_DTYPE_CODES = (
    "?", "i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8",
    "f2", "f4", "f8", "c8", "c16",
)


def load_checkpoint_restricted(checkpoint: Path) -> dict[str, Any]:
    """Read a checkpoint payload under restricted deserialization.

    Only NumPy numeric data constructors are allowed — never globals
    discovered from the untrusted checkpoint itself or custom model code.
    There is no fallback to unrestricted pickle loading. This is the single
    shared implementation; ``ai_model_bundle`` delegates to it.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch ships with the env
        raise RuntimeError(
            "当前环境缺少 PyTorch，无法安全校验权重：{}。".format(checkpoint.name)
        ) from exc
    import numpy as np
    multiarray = np._core.multiarray if hasattr(np, "_core") else np.core.multiarray
    numeric_dtypes = [np.dtype(code) for code in _NUMERIC_DTYPE_CODES]
    allowed = [np.ndarray, np.dtype, multiarray._reconstruct, multiarray.scalar]
    allowed += list({type(dtype) for dtype in numeric_dtypes})
    # NumPy 1.x and 2.x use different module paths for the same data types.
    allowed += [(multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
                (multiarray.scalar, "numpy.core.multiarray.scalar")]
    try:
        with torch.serialization.safe_globals(allowed):
            payload = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(
            "Checkpoint {} 无法在受限反序列化下加载（{}）。该文件可能损坏、与当前 "
            "PyTorch 版本不兼容或包含未支持的自定义对象；请由可信模型提供者重新导出"
            "纯数据权重，不要改用不受限的 Pickle 加载。".format(checkpoint.name, exc)
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(
            "Checkpoint {} 的内容不是预期的字典结构，无法读取模型元信息。".format(
                checkpoint.name
            )
        )
    return payload


def _extract_zip_strict(bundle: Path, destination: Path) -> None:
    """Extract a model ZIP under size/count/ratio and path-safety limits."""
    with zipfile.ZipFile(bundle, "r") as archive:
        members = archive.infolist()
        if len(members) > MAX_ZIP_FILES:
            raise RuntimeError(
                "模型 ZIP 包含 {} 个文件，超过上限 {}，已拒绝解压。".format(
                    len(members), MAX_ZIP_FILES
                )
            )
        total = 0
        seen: set[str] = set()
        for member in members:
            name = member.filename
            relative = Path(name)
            unsafe = (
                name.startswith(("/", "\\"))
                or relative.is_absolute()
                or ".." in relative.parts
                or ("\\" in name)
                or (len(name) > 1 and name[1] == ":")
            )
            if unsafe:
                raise RuntimeError("模型 ZIP 包含不安全路径：{}".format(name))
            folded = name.replace("\\", "/").casefold()
            if folded in seen:
                raise RuntimeError(
                    "模型 ZIP 包含大小写冲突的重复路径：{}".format(name)
                )
            seen.add(folded)
            if member.flag_bits & 0x1:
                raise RuntimeError("模型 ZIP 包含加密条目：{}".format(name))
            if stat.S_ISLNK((member.external_attr >> 16) & 0xFFFF):
                raise RuntimeError("模型 ZIP 包含符号链接：{}".format(name))
            total += member.file_size
            if total > MAX_ZIP_TOTAL_BYTES:
                raise RuntimeError(
                    "模型 ZIP 解压后总大小超过上限 {} 字节，已拒绝解压。".format(
                        MAX_ZIP_TOTAL_BYTES
                    )
                )
            if member.compress_size and (
                member.file_size > member.compress_size * MAX_ZIP_COMPRESSION_RATIO
            ):
                raise RuntimeError(
                    "模型 ZIP 条目 {} 压缩比例异常（{} → {} 字节），疑似压缩炸弹，"
                    "已拒绝解压。".format(name, member.compress_size, member.file_size)
                )
        archive.extractall(str(destination))


def _read_json_dict(path: Path) -> dict[str, Any] | None:
    payload = read_json(path, default=None)
    return payload if isinstance(payload, dict) else None


def _iter_child_dirs(current: Path) -> list[Path]:
    try:
        return sorted(p for p in current.iterdir() if p.is_dir())
    except OSError:
        return []


def _discover_config_dirs(root: Path) -> tuple[list[Path], list[dict[str, Any]]]:
    """Bounded search for nnU-Net v2 configuration directories.

    Returns complete candidates (dataset.json + plans.json present) and
    near-miss rows (enough structure to look like a model, but incomplete)
    so rejections can name exactly what is missing.
    """
    candidates: list[Path] = []
    rejected: list[dict[str, Any]] = []
    visited = 0
    queue: list[tuple[Path, int]] = [(root, 0)]
    while queue:
        current, depth = queue.pop(0)
        visited += 1
        if visited > MAX_SCAN_DIRS:
            raise RuntimeError(
                "目录结构过大（超过 {} 个目录），已停止扫描。".format(MAX_SCAN_DIRS)
            )
        has_dataset = (current / "dataset.json").is_file()
        has_plans = (current / "plans.json").is_file()
        has_folds = any(p.name.startswith("fold_") for p in _iter_child_dirs(current))
        missing = [
            name
            for name, present in (("dataset.json", has_dataset), ("plans.json", has_plans))
            if not present
        ]
        if has_dataset and has_plans:
            candidates.append(current)
        elif has_dataset or has_plans or has_folds:
            if len(rejected) < MAX_REJECTED_SAMPLE:
                rejected.append({"config_dir": str(current), "missing": missing})
        if depth < MAX_SCAN_DEPTH:
            for child in _iter_child_dirs(current):
                queue.append((child, depth + 1))
    return candidates, rejected


def _trainer_is_builtin(trainer_name: str) -> bool | None:
    """True/False when the local nnU-Net installation can answer; None if not."""
    try:
        import nnunetv2
        from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
    except ImportError:  # pragma: no cover - nnunetv2 ships with the env
        return None
    found = recursive_find_python_class(
        str(Path(nnunetv2.__path__[0], "training", "nnUNetTrainer")),
        trainer_name,
        "nnunetv2.training.nnUNetTrainer",
    )
    return found is not None


def _fold_checkpoint(fold_row: dict[str, Any]) -> str | None:
    """The checkpoint one fold can predict from: final when training ran to
    the end, else the validation-best checkpoint. ``checkpoint_latest.pth``
    is deliberately not accepted: before the first validation it holds barely
    trained weights that would still produce a plausible-looking garbage mask.
    """
    checkpoints = fold_row["checkpoints"]
    if CHECKPOINT_NAME in checkpoints:
        return CHECKPOINT_NAME
    if FALLBACK_CHECKPOINT_NAME in checkpoints:
        return FALLBACK_CHECKPOINT_NAME
    return None


def _scan_config_dir(config_dir: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "config_dir": str(config_dir),
        "configuration": None,
        "trainer": None,
        "plans_name": None,
        "dataset_name": None,
        "channel_names": None,
        "labels": None,
        "folds": [],
        "usable_folds": [],
        "fallback_checkpoint": None,
        "requires_fold_selection": False,
        "mirroring_axes": None,
        "missing": [],
        "blockers": [],
    }
    blockers: list[str] = row["blockers"]
    for name in ("dataset.json", "plans.json"):
        if not (config_dir / name).is_file():
            row["missing"].append(name)
    fold_rows = []
    for fold_dir in sorted(config_dir.glob("fold_*")):
        if not fold_dir.is_dir():
            continue
        checkpoints = sorted(
            p.name
            for p in fold_dir.iterdir()
            if p.is_file() and p.name.startswith("checkpoint_")
        )
        fold_rows.append({"fold": fold_dir.name, "checkpoints": checkpoints})
    row["folds"] = fold_rows
    usable = [f["fold"] for f in fold_rows if _fold_checkpoint(f)]
    row["usable_folds"] = usable
    row["requires_fold_selection"] = "fold_all" in usable and any(
        f != "fold_all" for f in usable
    )
    # 所有可用 fold 都没有最终权重时，导入必然回退到 checkpoint_best.pth；
    # UI 用这个标记提前告知，而不是导入后才在结果里说。
    if usable and not any(_fold_checkpoint(f) == CHECKPOINT_NAME for f in fold_rows):
        row["fallback_checkpoint"] = FALLBACK_CHECKPOINT_NAME
    if not row["missing"] and not usable:
        partial = [f for f in fold_rows if f["checkpoints"]]
        if partial:
            blockers.append(
                "各 fold 均缺少可推理权重（需 {} 或 {}）：现有 {}。".format(
                    CHECKPOINT_NAME,
                    FALLBACK_CHECKPOINT_NAME,
                    "；".join(
                        "{}：{}".format(f["fold"], "、".join(f["checkpoints"]))
                        for f in partial
                    ),
                )
            )
        else:
            blockers.append("没有任何包含 {} 的完整 fold。".format(CHECKPOINT_NAME))
    if row["missing"]:
        blockers.append("缺少文件：" + "、".join(row["missing"]) + "。")
        return row
    dataset = _read_json_dict(config_dir / "dataset.json")
    plans = _read_json_dict(config_dir / "plans.json")
    if dataset is None:
        blockers.append("dataset.json 不是有效的 JSON 对象。")
    if plans is None:
        blockers.append("plans.json 不是有效的 JSON 对象。")
    if dataset is None or plans is None:
        return row
    row["dataset_name"] = str(dataset.get("name") or "")
    channels = dataset.get("channel_names")
    if not isinstance(channels, dict):
        channels = dataset.get("modality")
    if not isinstance(channels, dict) or not channels:
        blockers.append("dataset.json 缺少 channel_names/modality 输入通道声明。")
    else:
        row["channel_names"] = {str(k): str(v) for k, v in channels.items()}
        if len(channels) > 1:
            blockers.append(
                "该模型需要 {} 个输入通道（{}），首期导入仅支持单通道模型；"
                "多通道需要逐病例绑定各序列，尚未支持。".format(
                    len(channels), "、".join(str(v) for v in channels.values())
                )
            )
    labels = dataset.get("labels")
    if not isinstance(labels, dict) or not labels:
        blockers.append("dataset.json 缺少 labels 声明。")
    else:
        row["labels"] = labels
        region_entries = [
            str(key)
            for key, value in labels.items()
            if isinstance(value, (list, tuple))
        ]
        if region_entries:
            blockers.append(
                "该模型使用区域（region-based）标签：{}；需要按模型语义解码，"
                "首期导入暂不支持。".format("、".join(region_entries))
            )
        non_int_entries = [
            str(key)
            for key, value in labels.items()
            if key != "background" and not isinstance(value, int)
        ]
        if non_int_entries:
            blockers.append(
                "以下标签的取值不是整数编号：{}；无法映射为分割标签。".format(
                    "、".join(non_int_entries)
                )
            )
    parts = config_dir.name.split("__")
    if len(parts) != 3 or not all(parts):
        blockers.append(
            "模型目录名不是 Trainer__Plans__Configuration 三段式，"
            "无法确定推理配置；不会猜测默认配置。"
        )
    else:
        row["trainer"], row["plans_name"], row["configuration"] = parts
        plans_configs = plans.get("configurations")
        if isinstance(plans_configs, dict) and parts[2] not in plans_configs:
            blockers.append(
                "plans.json 不包含配置 {}，目录名与实际配置不一致。".format(parts[2])
            )
        if "cascade" in parts[2]:
            blockers.append(
                "级联（cascade）模型需要完整推理链（前级结果与后处理），"
                "首期导入暂不支持。"
            )
    if usable:
        first = next(f for f in fold_rows if f["fold"] == usable[0])
        try:
            payload = load_checkpoint_restricted(
                config_dir / usable[0] / _fold_checkpoint(first)
            )
        except RuntimeError as exc:
            blockers.append(str(exc))
        else:
            trainer_name = payload.get("trainer_name")
            if not isinstance(trainer_name, str) or not trainer_name:
                blockers.append("Checkpoint 内缺少 trainer_name，无法验证训练器。")
            else:
                row["trainer"] = trainer_name
                builtin = _trainer_is_builtin(trainer_name)
                if builtin is None:
                    blockers.append(
                        "当前环境缺少 nnU-Net，无法验证训练器 {}，暂不能导入。".format(
                            trainer_name
                        )
                    )
                elif not builtin:
                    blockers.append(
                        "自定义训练器 {} 不是当前 nnU-Net 的内置训练器；需要可信环境"
                        "安装对应代码后另行验证，暂不导入。".format(trainer_name)
                    )
            row["mirroring_axes"] = payload.get("inference_allowed_mirroring_axes")
    return row


def _scan_directory(root: Path, source_kind: str, source_display: str) -> dict[str, Any]:
    config_dirs, rejected = _discover_config_dirs(root)
    if len(config_dirs) > MAX_CONFIG_CANDIDATES:
        raise RuntimeError(
            "发现 {} 个候选配置目录（上限 {}），结构异常，已停止。".format(
                len(config_dirs), MAX_CONFIG_CANDIDATES
            )
        )
    candidates = [_scan_config_dir(path) for path in config_dirs]
    importable = [row for row in candidates if not row["blockers"]]
    report: dict[str, Any] = {
        "source": source_display,
        "source_kind": source_kind,
        "candidates": candidates,
        "rejected_candidates": rejected,
        "importable": [row["configuration"] for row in importable],
    }
    if not candidates:
        report["hint"] = (
            "未找到可识别的 nnU-Net v2 配置目录（需同时包含 dataset.json 与 "
            "plans.json）。若是 nnU-Net v1 模型，需要独立的 v1 适配环境，暂不支持导入。"
        )
    return report


def scan_external_model(source: str | Path) -> dict[str, Any]:
    """Read-only scan of an external model directory or official ZIP."""
    source = Path(source).expanduser()
    if source.is_file():
        if source.suffix.lower() != ".zip":
            raise RuntimeError(
                "不支持的来源：目前仅接受 nnU-Net 模型目录或官方导出的 .zip 包。"
            )
        with tempfile.TemporaryDirectory() as temp:
            extracted = Path(temp) / "model"
            extracted.mkdir()
            _extract_zip_strict(source, extracted)
            return _scan_directory(extracted, "zip", str(source))
    if not source.is_dir():
        raise RuntimeError("来源路径不存在：{}".format(source))
    return _scan_directory(source, "directory", str(source))


def _collect_files(model_dir: Path) -> list[Path]:
    files = [p for p in sorted(model_dir.rglob("*")) if p.is_file()]
    if len(files) > MAX_ZIP_FILES:
        raise RuntimeError(
            "模型目录包含 {} 个文件，超过上限 {}。".format(len(files), MAX_ZIP_FILES)
        )
    return files


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_DIGEST_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _file_inventory(model_dir: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(model_dir).as_posix(),
            "sha256": _file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in _collect_files(model_dir)
    ]


def _inventory_digest(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _environment_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    try:
        import torch
        versions["torch"] = str(getattr(torch, "__version__", ""))
    except ImportError:  # pragma: no cover - torch ships with the env
        versions["torch"] = ""
    try:
        import nnunetv2
        versions["nnunetv2"] = str(getattr(nnunetv2, "__version__", ""))
    except ImportError:  # pragma: no cover - nnunetv2 ships with the env
        versions["nnunetv2"] = ""
    return versions


def _select_importable_row(report: dict[str, Any], configuration: str | None) -> dict[str, Any]:
    rows = [row for row in report["candidates"] if not row["blockers"]]
    if not rows:
        details = []
        for row in report["candidates"]:
            details.append(
                "{}：{}".format(row["configuration"] or row["config_dir"], "；".join(row["blockers"]))
            )
        for row in report["rejected_candidates"]:
            details.append(
                "{}：缺少 {}。".format(row["config_dir"], "、".join(row["missing"]))
            )
        raise RuntimeError(
            "该模型当前不能导入：\n{}".format("\n".join(details) if details else report.get("hint", "原因未知。"))
        )
    if configuration is not None:
        rows = [row for row in rows if row["configuration"] == configuration]
        if not rows:
            raise RuntimeError(
                "没有名为 {} 的可用配置；可用配置：{}。".format(
                    configuration,
                    "、".join(str(row["configuration"]) for row in report["candidates"] if not row["blockers"]),
                )
            )
    if len(rows) > 1:
        raise RuntimeError(
            "存在多个可用配置（{}），请明确选择其一，不猜测最佳配置。".format(
                "、".join(str(row["configuration"]) for row in rows)
            )
        )
    return rows[0]


def _resolve_checkpoint_name(
    fold_map: dict[str, dict[str, Any]], chosen_folds: list[str]
) -> str:
    """One checkpoint file must serve every chosen fold (the predictor loads
    the same name from each fold directory). Mixed final/best coverage cannot
    be served by a single recipe and is rejected, not guessed.
    """
    per_fold = [_fold_checkpoint(fold_map[name]) for name in chosen_folds]
    if CHECKPOINT_NAME in per_fold and FALLBACK_CHECKPOINT_NAME in per_fold:
        raise RuntimeError(
            "所选 fold 的权重不一致：部分只有 {}，部分只有 {}，无法用同一份权重推理；"
            "请只选择其中一类。".format(CHECKPOINT_NAME, FALLBACK_CHECKPOINT_NAME)
        )
    for name in (CHECKPOINT_NAME, FALLBACK_CHECKPOINT_NAME):
        if name in per_fold:
            return name
    raise RuntimeError("所选 fold 没有可推理的权重文件。")


def _import_directory(
    root: Path,
    workspace: str | Path,
    use_folds: list[str] | None,
    configuration: str | None,
    source_kind: str,
    source_display: str,
) -> dict[str, Any]:
    report = _scan_directory(root, source_kind, source_display)
    row = _select_importable_row(report, configuration)
    usable = list(row["usable_folds"])
    fold_map = {f["fold"]: f for f in row["folds"]}
    chosen_folds = list(use_folds) if use_folds else usable
    unknown = [f for f in chosen_folds if f not in usable]
    if unknown:
        raise RuntimeError(
            "选择的 fold 不可用：{}；可用 fold：{}。".format(
                "、".join(unknown), "、".join(usable)
            )
        )
    if row["requires_fold_selection"] and not use_folds:
        raise RuntimeError(
            "fold_all 与数字 fold 并存（{}），请明确选择要使用的 fold，"
            "不能混用。".format("、".join(usable))
        )
    checkpoint_name = _resolve_checkpoint_name(fold_map, chosen_folds)
    config_dir = Path(row["config_dir"])
    inventory = _file_inventory(config_dir)
    digest = _inventory_digest(inventory)
    display_name = row["dataset_name"] or row["configuration"] or "external_model"
    task_slug = stable_task_identifier(display_name)
    model_id = "ext_{}".format(digest[:16])
    external_root = workspace_paths(workspace)["root"] / "external_models"
    destination = external_root / task_slug / model_id
    if destination.is_dir():
        existing = read_json(destination / EXTERNAL_MANIFEST_NAME, default=None)
        if isinstance(existing, dict) and existing.get("content_sha256") == digest:
            existing_folds = list((existing.get("recipe") or {}).get("use_folds") or [])
            if existing_folds != chosen_folds:
                # 同一内容换 fold 重新导入会静默沿用旧 recipe：混合权重场景下
                # 预测窗口按旧 recipe 找不到权重文件，误判为不可用。必须报错。
                raise RuntimeError(
                    "相同内容的外部模型已按 fold {} 导入过（{}）；"
                    "如需改用其他 fold，请先删除该外部模型后重新导入。".format(
                        "、".join(existing_folds), destination
                    )
                )
            return {
                "reused": True,
                "model_id": model_id,
                "destination": str(destination),
                "manifest": existing,
                "message": "相同内容的外部模型已导入过，未生成重复版本。",
            }
        raise RuntimeError(
            "目标目录 {} 已存在且内容不一致，已取消导入。".format(destination)
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(
        "{}.importing_{}".format(destination.name, uuid.uuid4().hex)
    )
    model_dir_name = config_dir.name
    try:
        staging.mkdir()
        # 保留 Trainer__Plans__Configuration 目录名：推理侧靠它解析配置三元组，
        # 展平会静默退回默认 3d_fullres（对 2d 等模型是错误配置）。
        shutil.copytree(str(config_dir), str(staging / model_dir_name))
        if _inventory_digest(_file_inventory(staging / model_dir_name)) != digest:
            raise RuntimeError(
                "复制期间模型文件内容发生变化，已取消本次导入；来源未做任何修改。"
            )
        manifest = {
            "schema_version": EXTERNAL_SCHEMA_VERSION,
            "model_id": model_id,
            "display_name": display_name,
            "task_slug": task_slug,
            "origin": "external_pretrained_unverified",
            "verification": {
                "local_status": "external_pretrained_unverified",
                "checkpoint_restricted_load": True,
                "notes": (
                    "外部预训练模型：仅完成元信息与受限加载校验，"
                    "未做本地推理效果验证。"
                ),
            },
            "source": {
                "kind": source_kind,
                "path": source_display,
                "license": "unknown",
                "publisher_notes": "",
            },
            "content_sha256": digest,
            "model_dir": str(destination / model_dir_name),
            "files": inventory,
            # 以下派生键供推理侧直接消费：run_inference 与
            # validate_model_input_compatibility 读的是 manifest 文件本体
            # （而非注册表行），缺这些键会静默跳过任务/模态兼容校验。
            # labels 与训练侧 manifest 同构（{"name","id","aliases"} 列表），
            # background 为保留标签不导出。
            "task_id": task_slug,
            "task_name": display_name,
            "modality": next(iter(row["channel_names"].values()), ""),
            "folds": [name[len("fold_"):] for name in chosen_folds],
            "labels": [
                {
                    "name": str(key),
                    "id": int(value),
                    "aliases": [str(key)],
                }
                for key, value in row["labels"].items()
                if key != "background"
            ],
            "trainer": row["trainer"],
            "plans_name": row["plans_name"],
            "configuration": row["configuration"],
            "folds_available": usable,
            "recipe": {
                "use_folds": chosen_folds,
                "checkpoint_name": checkpoint_name,
                "mirroring_axes": row["mirroring_axes"],
            },
            "channel_names": row["channel_names"],
            "environment": _environment_versions(),
            "created_at_epoch": time.time(),
        }
        write_json_atomic(staging / EXTERNAL_MANIFEST_NAME, manifest)
        os.replace(str(staging), str(destination))
    except BaseException:
        shutil.rmtree(str(staging), ignore_errors=True)
        for parent in (destination.parent, destination.parent.parent):
            try:
                parent.rmdir()  # only succeeds when empty: no residue left
            except OSError:
                pass
        raise
    return {
        "reused": False,
        "model_id": model_id,
        "destination": str(destination),
        "manifest": manifest,
    }


def import_external_model(
    source: str | Path,
    workspace: str | Path,
    use_folds: list[str] | None = None,
    configuration: str | None = None,
) -> dict[str, Any]:
    """Validate and publish an external model for the prediction window.

    The live model registry JSON is never written: it records models trained
    on this machine, while external imports are discovered by
    ``nnunet_common.load_models`` from ``external_models/`` manifests.
    """
    source = Path(source).expanduser()
    if source.is_file():
        with tempfile.TemporaryDirectory() as temp:
            extracted = Path(temp) / "model"
            extracted.mkdir()
            _extract_zip_strict(source, extracted)
            return _import_directory(
                extracted, workspace, use_folds, configuration, "zip", str(source)
            )
    if not source.is_dir():
        raise RuntimeError("来源路径不存在：{}".format(source))
    return _import_directory(
        source, workspace, use_folds, configuration, "directory", str(source)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scan and import external nnU-Net v2 training results."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    scan_parser = sub.add_parser("scan", help="read-only model inspection")
    scan_parser.add_argument("--source", required=True)
    import_parser = sub.add_parser("import", help="validate and publish a model")
    import_parser.add_argument("--source", required=True)
    import_parser.add_argument("--workspace", required=True)
    import_parser.add_argument("--configuration", default=None)
    import_parser.add_argument("--use-folds", nargs="*", default=None)
    args = parser.parse_args(argv)
    if args.command == "scan":
        payload = scan_external_model(args.source)
    else:
        payload = import_external_model(
            args.source, args.workspace, use_folds=args.use_folds,
            configuration=args.configuration,
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
