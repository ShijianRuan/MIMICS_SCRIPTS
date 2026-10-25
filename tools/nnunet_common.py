#!/usr/bin/env python3
"""Shared contracts for the local, remote, and Mimics nnU-Net integration."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_ROOT = ROOT / "integrations" / "nnunet_segmentation_workflow"
TRAINER_ROOT = WORKFLOW_ROOT / "trainers"
SCHEMA_VERSION = "mimics_nnunet_job.v1"
MODEL_SCHEMA_VERSION = "mimics_nnunet_model.v1"
TERMINAL_STATES = {"completed", "failed", "cancelled", "abandoned"}


def safe_identifier(value: object, fallback: str = "item") -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    text = text.strip("._-")
    return text[:96] or fallback


def stable_task_identifier(value: object, fallback: str = "item") -> str:
    """nnU-Net task_id that never collapses two distinct task names.

    UI19/F22: ``safe_identifier`` folds every non-ASCII character to "_", so
    two different Chinese target names both become "item" and silently merge
    in the model library. nnU-Net's on-disk rules (Dataset{:03d}_<id>) still
    require ASCII, so the identifier stays ASCII — but whenever the slug is
    lossy (any character replaced, stripped, or truncated) a short digest of
    the original name keeps distinct names distinct.

    R01: the result must survive a second normalization. Downstream code
    (normalize_request, cache and model directories) feeds the generated id
    back through ``safe_identifier``, which truncates to 96 — so the digest
    is fitted inside that budget (slug capped at 87 + 8 hex chars) and the
    output is idempotent: re-deriving the id from an already-generated id
    returns it unchanged. Pure-ASCII names that need no cleaning keep their
    existing identifier, so existing workspaces and models are unaffected.
    """
    text = str(value or "").strip()
    slug = safe_identifier(text, fallback)
    # Lossy iff the slug is not the exact input: replacement (non-ASCII or
    # forbidden chars), stripping (leading/trailing "._-"), truncation (>96
    # chars), or the empty-input fallback. Covers the R01 case of two long
    # same-prefix ASCII names that differ only past column 96.
    if not text or slug != text:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        slug = "{}_{}".format(slug[:87], digest)
    return slug


def unique_case_keys(case_ids: list[object], fallback: str = "case") -> dict[str, str]:
    """Collision-free internal keys for case ids (F22).

    ``safe_identifier`` is lossy — two Chinese case names, or a 96-char
    truncation, or ``a b`` vs ``a_b`` can all collapse to the same key, and
    every downstream bucket (cache dirs, imagesTr files, remote input paths)
    is keyed by it, so a collision merges two patients into one. Keep the
    plain slug when it is unique; disambiguate collisions by appending a
    short digest of the original id so existing no-collision workspaces keep
    their keys (and their caches).
    """
    keys: dict[str, str] = {}
    by_slug: dict[str, list[str]] = {}
    for case_id in case_ids:
        original = str(case_id)
        slug = safe_identifier(original, fallback)
        by_slug.setdefault(slug, []).append(original)
    for slug, originals in by_slug.items():
        if len(originals) == 1:
            keys[originals[0]] = slug
            continue
        for original in sorted(originals):
            digest = hashlib.sha256(original.encode("utf-8")).hexdigest()[:8]
            keys[original] = "{}_{}".format(slug, digest)
    return keys


def medical_stem(path: str | Path) -> str:
    name = Path(path).name
    lower = name.lower()
    for suffix in (".seg.nii.gz", ".seg.nii", ".nii.gz", ".nrrd.gz"):
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def replace_with_retry(source: str | Path, destination: str | Path) -> None:
    """os.replace for freshly built staging dirs.

    Windows callers hit transient WinError 5 (access denied) when an
    antivirus or the search indexer briefly holds a handle inside the
    just-written tree; one blip must not fail an hours-long pipeline.
    Same retry shape as write_json_atomic.
    """
    last_error: OSError | None = None
    for attempt in range(20):
        try:
            os.replace(str(source), str(destination))
            return
        except OSError as exc:
            last_error = exc
            time.sleep(min(0.25, 0.02 * (attempt + 1)))
    raise OSError(
        "Could not publish {}: {}".format(destination, last_error)
    )


def file_fingerprint(path: str | Path) -> str | None:
    """S02: content fingerprint of a prediction output file.

    sha256 over the raw bytes — linear in file size, no gzip decode. None
    when the file is missing or unreadable: callers that need to tell those
    apart use capture_output_fingerprint instead.
    """
    digest = hashlib.sha256()
    try:
        with open(str(Path(path)), "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


# S02: start-of-task identity markers for a publish target. A plain None in
# expected_previous_fingerprint means "caller did not request protection"
# (legacy calls); these sentinels are explicit recorded states and survive
# the JSON round-trip through job status files.
FINGERPRINT_ABSENT = "__absent__"
FINGERPRINT_UNREADABLE = "__unreadable__"


def capture_output_fingerprint(path: str | Path) -> str:
    """S02: record the publish target's identity as the task starts.

    Distinguishes "target missing" (FINGERPRINT_ABSENT) and "exists but
    unreadable" (FINGERPRINT_UNREADABLE) from a real content fingerprint,
    so publish can enforce "still missing" / "still the same" later instead
    of silently disabling the conflict check when the target was absent.
    """
    path = Path(path)
    if not path.exists():
        return FINGERPRINT_ABSENT
    fingerprint = file_fingerprint(path)
    return FINGERPRINT_UNREADABLE if fingerprint is None else fingerprint


def publish_target_conflict(
    path: str | Path, expected_previous_fingerprint: str | None
) -> bool:
    """S02: would publishing to ``path`` overwrite content that changed?

    ``expected_previous_fingerprint`` is what capture_output_fingerprint
    recorded when the task started. None means the caller did not request
    protection. Anything the task cannot prove unchanged (appeared, deleted,
    modified, unreadable) counts as a conflict — conservative, because a
    silent replace is how a newer human-reviewed result gets lost.
    """
    if expected_previous_fingerprint is None:
        return False
    path = Path(path)
    if not path.exists():
        # Recorded as present (fingerprint or unreadable) but gone at
        # publish time — it changed during the run.
        return expected_previous_fingerprint != FINGERPRINT_ABSENT
    if expected_previous_fingerprint == FINGERPRINT_ABSENT:
        # The target appeared while the task ran; never silently replace it.
        return True
    if expected_previous_fingerprint == FINGERPRINT_UNREADABLE:
        # Exists now, but its start-of-task content could not be read —
        # we cannot prove it is unchanged, so refuse to replace it.
        return True
    return file_fingerprint(path) != expected_previous_fingerprint


def read_json(path: str | Path, default: Any = None) -> Any:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return default


def write_json_atomic(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        "{}.{}.tmp".format(destination.name, uuid.uuid4().hex)
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        last_error: OSError | None = None
        for attempt in range(20):
            try:
                os.replace(str(temporary), str(destination))
                return
            except OSError as exc:
                last_error = exc
                time.sleep(min(0.25, 0.02 * (attempt + 1)))
        raise OSError(
            "Could not publish {}: {}".format(destination, last_error)
        )
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass


def append_log(path: str | Path, message: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write(
            "[{}] {}\n".format(
                time.strftime("%Y-%m-%d %H:%M:%S"), str(message).rstrip()
            )
        )


def compact_completed_log(
    path: str | Path,
    max_bytes: int = 32 * 1024 * 1024,
    head_bytes: int = 1024 * 1024,
    tail_bytes: int = 15 * 1024 * 1024,
) -> dict[str, Any]:
    """Bound a completed task log while retaining startup and recent diagnostics."""
    destination = Path(path)
    try:
        original_size = int(destination.stat().st_size)
    except OSError:
        return {"compacted": False, "original_bytes": 0, "retained_bytes": 0}
    if original_size <= int(max_bytes):
        return {
            "compacted": False,
            "original_bytes": original_size,
            "retained_bytes": original_size,
        }
    head_count = max(0, min(int(head_bytes), original_size))
    tail_count = max(0, min(int(tail_bytes), original_size - head_count))
    with destination.open("rb") as handle:
        head = handle.read(head_count)
        handle.seek(max(head_count, original_size - tail_count))
        tail = handle.read()
    marker = (
        "\n[Log compacted after task completion: {} bytes omitted.]\n".format(
            max(0, original_size - len(head) - len(tail))
        )
    ).encode("utf-8")
    temporary = destination.with_name(
        "{}.{}.compact.tmp".format(destination.name, uuid.uuid4().hex)
    )
    try:
        with temporary.open("wb") as handle:
            handle.write(head)
            handle.write(marker)
            handle.write(tail)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(destination))
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass
    return {
        "compacted": True,
        "original_bytes": original_size,
        "retained_bytes": int(destination.stat().st_size),
    }


def update_status(path: str | Path, **values: Any) -> dict[str, Any]:
    payload = read_json(path, {}) or {}
    payload.update(values)
    payload["updated_at_epoch"] = time.time()
    write_json_atomic(path, payload)
    return payload


def path_signature(path: str | Path) -> dict[str, Any]:
    """Identity of a file or directory from its own and children's metadata.

    R14: directories are re-scanned on every call. The previous cross-process
    manifest cache keyed on the directory's own mtime, so an in-place child
    edit (DICOM slice repaired/re-synced, same folder name) kept serving the
    stale signature for up to 7 days and downstream prepared caches reused
    conversions of the old source. Signed paths are case-level series
    folders (tens to a few thousand files), so an un-cached stat scan costs
    milliseconds — correctness of the source fingerprint comes first.
    """
    source = Path(path)
    stat = source.stat()
    if source.is_file():
        return {
            "kind": "file",
            "size": int(stat.st_size),
            "mtime_ns": int(getattr(stat, "st_mtime_ns", stat.st_mtime * 1e9)),
            "ctime_ns": int(getattr(stat, "st_ctime_ns", stat.st_ctime * 1e9)),
            "device": int(getattr(stat, "st_dev", 0) or 0),
            "file_id": int(getattr(stat, "st_ino", 0) or 0),
        }
    total = 0
    latest = int(getattr(stat, "st_mtime_ns", stat.st_mtime * 1e9))
    count = 0
    manifest = hashlib.sha256()
    for child in sorted(source.rglob("*")):
        if not child.is_file():
            continue
        child_stat = child.stat()
        size = int(child_stat.st_size)
        modified = int(
            getattr(child_stat, "st_mtime_ns", child_stat.st_mtime * 1e9)
        )
        changed = int(
            getattr(child_stat, "st_ctime_ns", child_stat.st_ctime * 1e9)
        )
        total += size
        latest = max(
            latest,
            modified,
        )
        count += 1
        manifest.update(
            json.dumps(
                [
                    child.relative_to(source).as_posix(),
                    size,
                    modified,
                    changed,
                    int(getattr(child_stat, "st_dev", 0) or 0),
                    int(getattr(child_stat, "st_ino", 0) or 0),
                ],
                separators=(",", ":"),
            ).encode("utf-8")
        )
        manifest.update(b"\n")
    signature = {
        "kind": "directory",
        "files": count,
        "size": total,
        "mtime_ns": latest,
        "ctime_ns": int(getattr(stat, "st_ctime_ns", stat.st_ctime * 1e9)),
        "manifest_sha256": manifest.hexdigest(),
    }
    return signature


def stable_digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _labels_from_mapping(mapping: object) -> list[dict[str, Any]] | None:
    if not isinstance(mapping, dict) or not mapping:
        return None
    flat_rows = []
    grouped_rows = []
    for name, value in mapping.items():
        if str(name).strip().lower() == "background":
            continue
        if isinstance(value, int) and not isinstance(value, bool):
            flat_rows.append(
                {
                    "name": str(name),
                    "id": int(value),
                    "aliases": [str(name)],
                    "source_mode": "alternatives",
                }
            )
            continue
        if (
            isinstance(value, dict)
            and isinstance(value.get("label"), int)
            and not isinstance(value.get("label"), bool)
            and isinstance(value.get("organs"), list)
        ):
            grouped_rows.append(
                {
                    "name": str(name),
                    "id": int(value["label"]),
                    "aliases": [str(item) for item in value["organs"]],
                    "source_mode": "union",
                }
            )
            continue
        return None
    if flat_rows and not grouped_rows:
        by_id: dict[int, dict[str, Any]] = {}
        for row in flat_rows:
            label_id = int(row["id"])
            existing = by_id.get(label_id)
            if existing is None:
                by_id[label_id] = dict(row)
                continue
            existing["aliases"].extend(row["aliases"])
            existing["source_mode"] = "union"
        flat_rows = list(by_id.values())
    rows = grouped_rows if grouped_rows and not flat_rows else flat_rows
    if not rows or (grouped_rows and flat_rows):
        return None
    return normalize_labels(rows)


def load_label_sets(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Load multi-class task definitions from ModelMap TOML or dataset JSON."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ValueError("Label map does not exist: {}".format(source))
    if source.suffix.lower() == ".json":
        payload = read_json(source, None)
    else:
        try:
            import tomllib
        except ModuleNotFoundError:
            try:
                import tomli as tomllib
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "Python tomllib or the tomli package is required to read ModelMap TOML files."
                ) from exc
        with source.open("rb") as handle:
            payload = tomllib.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Label map must contain a JSON/TOML object.")
    candidates: dict[str, object] = {}
    if isinstance(payload.get("labels"), dict):
        candidates[source.stem] = payload["labels"]
    else:
        direct = _labels_from_mapping(payload)
        if direct is not None:
            return {source.stem: direct}
        candidates.update(
            (str(name), value)
            for name, value in payload.items()
            if isinstance(value, dict)
        )
    result = {}
    errors = []
    for name, mapping in candidates.items():
        try:
            rows = _labels_from_mapping(mapping)
        except ValueError as exc:
            errors.append("{}: {}".format(name, exc))
            continue
        if rows:
            result[name] = rows
    if not result:
        detail = " " + "; ".join(errors[:5]) if errors else ""
        raise ValueError(
            "No valid multi-class label set was found in {}.{}".format(
                source, detail
            )
        )
    return result


def _semantic_label_key(value: object) -> str:
    """Semantic identity of a label/alias name (R06).

    ``safe_identifier`` folds every non-ASCII character to "_", so "肝脏" and
    "脾脏" both became "item" and normalize_labels rejected any two Chinese
    labels as duplicates. The ASCII slug is a file-name concern; label/alias
    uniqueness is a human-language concern and must use Unicode
    normalization (NFKC) + casefold, which keeps distinct organ names
    distinct while still folding case/whitespace variants.
    """
    import unicodedata

    return unicodedata.normalize("NFKC", str(value or "").strip()).casefold()


def normalize_labels(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("At least one nnU-Net label must be configured.")
    rows: list[dict[str, Any]] = []
    names: set[str] = set()
    aliases_seen: dict[str, str] = {}
    ids: set[int] = set()
    for index, raw in enumerate(value, start=1):
        if not isinstance(raw, dict):
            raise ValueError("Label row {} is not an object.".format(index))
        name = str(raw.get("name") or "").strip()
        if not name:
            raise ValueError("Label row {} has no output name.".format(index))
        # R16: "background" is nnU-Net's reserved name for label 0 (the zero
        # background added by _materialize_nnunet_raw). A user-defined label
        # with this name would silently overwrite the background entry in
        # dataset.json and break training/integrity checks downstream.
        if _semantic_label_key(name) == "background":
            raise ValueError(
                "标签名“background”是 nnU-Net 的保留背景项（ID 0），"
                "不能用作输出标签名；请改用其他名称。"
            )
        label_id = int(raw.get("id") or index)
        if label_id <= 0 or label_id > 255:
            raise ValueError("Label '{}' ID must be between 1 and 255.".format(name))
        aliases_value = raw.get("aliases") or [name]
        source_mode = str(raw.get("source_mode") or "alternatives").strip().lower()
        if source_mode not in {"alternatives", "union"}:
            raise ValueError(
                "Label '{}' source mode must be 'alternatives' or 'union'.".format(
                    name
                )
            )
        if isinstance(aliases_value, str):
            aliases = [
                item.strip()
                for item in aliases_value.replace(";", ",").split(",")
                if item.strip()
            ]
        else:
            aliases = [str(item).strip() for item in aliases_value if str(item).strip()]
        if name not in aliases:
            aliases.insert(0, name)
        unique_aliases = []
        local_aliases = set()
        for alias in aliases:
            canonical_alias = _semantic_label_key(alias)
            if canonical_alias == "background":
                raise ValueError(
                    "别名“{0}”是 nnU-Net 的保留背景项（ID 0），"
                    "不能用作标签别名；请改用其他名称。".format(alias)
                )
            if canonical_alias in local_aliases:
                continue
            owner = aliases_seen.get(canonical_alias)
            if owner is not None:
                raise ValueError(
                    "Mask 别名“{0}”同时分配给了标签“{1}”和“{2}”，"
                    "请改为其中一个标签专用的名称。".format(alias, owner, name)
                )
            local_aliases.add(canonical_alias)
            unique_aliases.append(alias)
        canonical_name = _semantic_label_key(name)
        if canonical_name in names:
            raise ValueError("标签名称重复：{}".format(name))
        if label_id in ids:
            raise ValueError("标签 ID 重复：{}".format(label_id))
        names.add(canonical_name)
        ids.add(label_id)
        for canonical_alias in local_aliases:
            aliases_seen[canonical_alias] = name
        rows.append(
            {
                "name": name,
                "id": label_id,
                "aliases": unique_aliases,
                "source_mode": source_mode,
            }
        )
    rows.sort(key=lambda row: int(row["id"]))
    expected_ids = list(range(1, len(rows) + 1))
    actual_ids = [int(row["id"]) for row in rows]
    if actual_ids != expected_ids:
        raise ValueError(
            "nnU-Net 标签 ID 必须从 1 开始连续编号；当前为 {}。".format(
                ", ".join(str(value) for value in actual_ids)
            )
        )
    return rows


def normalize_request(request: dict[str, Any]) -> dict[str, Any]:
    values = dict(request or {})
    operation = str(values.get("operation") or "train").lower()
    if operation not in {"train", "infer", "active_learning"}:
        raise ValueError("Unsupported nnU-Net operation: {}".format(operation))
    values["operation"] = operation
    values["job_id"] = safe_identifier(values.get("job_id"), operation)
    values["workspace"] = str(
        Path(values.get("workspace") or Path.home() / ".mimics_script" / "nnunet")
        .expanduser()
        .resolve()
    )
    if operation == "train":
        values["task_name"] = str(values.get("task_name") or "Segmentation task").strip()
        # UI19: the human-facing task_name is free-form (Chinese is fine and
        # is written to dataset.json / manifests); only the nnU-Net directory
        # identifier is ASCII, generated losslessly via stable_task_identifier.
        values["task_id"] = stable_task_identifier(
            values.get("task_id") or values["task_name"])
        values["labels"] = normalize_labels(values.get("labels"))
        values["dataset_id"] = int(values.get("dataset_id") or 701)
        if not 1 <= values["dataset_id"] <= 999:
            raise ValueError("Dataset ID 必须在 1 到 999 之间。")
        values["configuration"] = str(values.get("configuration") or "3d_fullres")
        if values["configuration"] not in {"2d", "3d_fullres", "3d_lowres"}:
            raise ValueError(
                "支持的配置为 2d、3d_fullres 和 3d_lowres。"
            )
        values["modality"] = str(values.get("modality") or "CT").strip() or "CT"
        values["fold"] = str(values.get("fold") if values.get("fold") is not None else "0")
        if values["fold"] not in {"0", "1", "2", "3", "4", "all"}:
            raise ValueError("Fold 必须是 0-4 或 all。")
        values["epochs"] = int(values.get("epochs") or 1000)
        if not 1 <= values["epochs"] <= 10000:
            raise ValueError("训练轮数必须在 1 到 10000 之间。")
        values["num_gpus"] = int(values.get("num_gpus") or 1)
        if not 1 <= values["num_gpus"] <= 16:
            raise ValueError("GPU count must be between 1 and 16.")
        values["preprocess_workers"] = int(values.get("preprocess_workers") or 4)
        if not 1 <= values["preprocess_workers"] <= 64:
            raise ValueError("Preprocessing workers must be between 1 and 64.")
        raw_validation_fraction = values.get("validation_fraction")
        values["validation_fraction"] = (
            0.2
            if raw_validation_fraction in (None, "")
            else float(raw_validation_fraction)
        )
        if not 0.0 <= values["validation_fraction"] < 0.9:
            raise ValueError("验证集比例必须在 0 到 0.9 之间。")
        if values["fold"] != "all" and values["validation_fraction"] <= 0:
            raise ValueError(
                "Fold 0-4 训练时验证集比例必须大于 0；"
                "如不需验证集请选择 fold 'all'。"
            )
        # G9: case-count semantics, matching the FlexiCT wizard. 0/absent =
        # fall back to the fraction (backwards compatible).
        raw_val_cases = values.get("val_cases")
        values["val_cases"] = (
            0 if raw_val_cases in (None, "") else int(raw_val_cases)
        )
        if values["val_cases"] < 0:
            raise ValueError("val_cases must be a non-negative case count.")
        values["split_seed"] = int(values.get("split_seed") or 2026)
        values["trainer"] = str(values.get("trainer") or "MimicsNNUNetTrainer").strip()
        values["plans"] = str(values.get("plans") or "nnUNetPlans").strip()
        values["gpu_id"] = str(values.get("gpu_id") or "").strip()
        if values["gpu_id"]:
            devices = [item.strip() for item in values["gpu_id"].split(",") if item.strip()]
            if not devices or any(not item.isdigit() for item in devices):
                raise ValueError("本机 GPU 设备必须是用英文逗号分隔的数字编号（如 0 或 0,1）。")
            if len(devices) != values["num_gpus"]:
                raise ValueError(
                    "GPU 数量必须与本机 GPU 设备编号的个数一致。"
                )
            values["gpu_id"] = ",".join(devices)
        values["gpu_lock_timeout_seconds"] = float(
            values.get("gpu_lock_timeout_seconds") or 86400
        )
        if values["gpu_lock_timeout_seconds"] <= 0:
            raise ValueError("GPU wait timeout must be positive.")
        values["label_source"] = str(values.get("label_source") or "dataset_masks")
        if values["label_source"] not in {
            "dataset_masks",
            "exported_masks",
            "mcs_refresh",
            "prepared",
        }:
            raise ValueError("Unsupported label source: {}".format(values["label_source"]))
        values["missing_label_policy"] = str(
            values.get("missing_label_policy") or "require_all"
        ).strip().lower()
        if values["missing_label_policy"] not in {"require_all", "background"}:
            raise ValueError(
                "缺失标签的处理策略必须是 'require_all' 或 'background'。"
            )
        for key, expected_length in (("spacing", 3), ("patch_size", 2 if values["configuration"] == "2d" else 3)):
            raw = values.get(key)
            if raw in (None, "", []):
                values[key] = None
                continue
            parsed = [float(item) if key == "spacing" else int(item) for item in raw]
            if len(parsed) != expected_length or any(item <= 0 for item in parsed):
                raise ValueError(
                    "{} 必须包含 {} 个正数（当前配置 {}）。".format(
                        key, expected_length, values["configuration"]
                    )
                )
            values[key] = parsed
        if values["configuration"] == "2d" and values["spacing"] is not None:
            raise ValueError(
                "2D 配置不支持手动设定 spacing；请保持自动，或改选 3D 配置。"
            )
        batch_size = values.get("batch_size")
        values["batch_size"] = int(batch_size) if batch_size not in (None, "", 0, "0") else None
        if values["batch_size"] is not None and values["batch_size"] < 1:
            raise ValueError("批次大小必须为正数。")
        if bool(values.get("continue_training")) and str(
            values.get("pretrained_weights") or ""
        ).strip():
            raise ValueError(
                "Continuing a fold and loading pretrained weights are mutually exclusive."
            )
    else:
        modality = str(values.get("source_modality") or "").strip().upper()
        values["source_modality"] = "MR" if modality == "MRI" else modality
    return values


def workspace_paths(workspace: str | Path) -> dict[str, Path]:
    root = Path(workspace).expanduser().resolve()
    return {
        "root": root,
        "jobs": root / "jobs",
        "models": root / "models",
        "cache": root / "cache",
        "runtime": root / "runtime",
        "registry": root / "model_registry.json",
    }


def _iter_dataset_dirs(base: Path):
    """iterdir over Dataset[0-9][0-9][0-9]_* dirs, swallowing races."""
    try:
        return list(base.glob("Dataset[0-9][0-9][0-9]_*"))
    except OSError:
        return []


def sweep_dataset_retention(
    workspace: str | Path,
    protected_names: set[str],
    retention_days: float,
) -> dict[str, Any]:
    """Prune unreferenced runtime Dataset dirs past their retention window.

    Both training families materialize nnUNet_raw/preprocessed/results under
    ``<workspace>/runtime/`` per Dataset dir and nothing else ever removed
    them, so disk grew without bound. Everything under runtime/ is
    rebuildable (raw re-materialized from the source .mcs cases, preprocessed
    re-planned, results re-trained) and delivered models live in models/ —
    but a Dataset still in use must survive: callers pass the dataset names
    referenced by registered models and non-terminal jobs as
    ``protected_names``. ``retention_days`` <= 0 disables.
    """
    report: dict[str, Any] = {
        "removed_datasets": [],
        "kept_datasets": 0,
        "retention_days": retention_days,
    }
    if retention_days <= 0:
        return report
    runtime = workspace_paths(workspace)["runtime"]
    cutoff = time.time() - retention_days * 86400
    for name in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        base = runtime / name
        if not base.is_dir():
            continue
        for dataset_dir in _iter_dataset_dirs(base):
            if not dataset_dir.is_dir() or dataset_dir.name in protected_names:
                report["kept_datasets"] += 1
                continue
            try:
                if dataset_dir.stat().st_mtime >= cutoff:
                    report["kept_datasets"] += 1
                    continue
                shutil.rmtree(str(dataset_dir), ignore_errors=True)
                report["removed_datasets"].append(
                    "{}/{}".format(name, dataset_dir.name)
                )
            except OSError:
                continue
    return report


def sweep_source_grid_cache_retention(
    workspace: str | Path,
    protected_task_ids: set[str],
    retention_days: float,
) -> dict[str, Any]:
    """Prune rebuildable cache/source_grid case dirs past their retention.

    ``cache/source_grid/<task_id>/<case_id>/`` holds materialized per-case
    training inputs, fully rebuildable from the source .mcs files. Whole task
    trees named in ``protected_task_ids`` (non-terminal jobs' task ids) are
    skipped so a running job's cache-hit validation never loses its buckets.
    ``retention_days`` <= 0 disables. Mirrors the nnInteractive prepared-cache
    sweep's mtime-window contract.
    """
    report: dict[str, Any] = {
        "removed_cases": [],
        "retention_days": retention_days,
    }
    if retention_days <= 0:
        return report
    root = workspace_paths(workspace)["cache"] / "source_grid"
    if not root.is_dir():
        return report
    cutoff = time.time() - retention_days * 86400
    for task_dir in _iter_safe_dir_children(root):
        if task_dir.name in protected_task_ids:
            continue
        for case_dir in _iter_safe_dir_children(task_dir):
            try:
                if case_dir.stat().st_mtime >= cutoff:
                    continue
                shutil.rmtree(str(case_dir), ignore_errors=True)
                report["removed_cases"].append(
                    "{}/{}".format(task_dir.name, case_dir.name)
                )
            except OSError:
                continue
        try:
            if task_dir.is_dir() and not any(task_dir.iterdir()):
                task_dir.rmdir()
        except OSError:
            pass
    return report


def _iter_safe_dir_children(path: Path):
    """list(path.iterdir()) that only yields directories, swallowing races."""
    try:
        return [child for child in path.iterdir() if child.is_dir()]
    except OSError:
        return []


def suggest_dataset_id(workspace: str | Path, preferred: int = 701) -> int:
    """Choose an unused nnU-Net dataset number in one managed model library."""
    paths = workspace_paths(workspace)
    used = set()
    for root in (
        paths["runtime"] / "nnUNet_raw",
        paths["runtime"] / "nnUNet_preprocessed",
        paths["runtime"] / "nnUNet_results",
    ):
        if not root.is_dir():
            continue
        for candidate in root.glob("Dataset[0-9][0-9][0-9]_*"):
            try:
                used.add(int(candidate.name[7:10]))
            except (TypeError, ValueError):
                pass
    for model in load_models(workspace):
        try:
            used.add(int(model.get("dataset_id")))
        except (TypeError, ValueError):
            pass
    order = list(range(max(1, int(preferred)), 1000)) + list(
        range(1, max(1, int(preferred)))
    )
    for value in order:
        if value not in used:
            return value
    raise RuntimeError("All nnU-Net Dataset IDs from 1 to 999 are in use.")


def model_registry_paths(workspace: str | Path) -> list[Path]:
    local = workspace_paths(workspace)["registry"]
    global_path = Path.home() / ".mimics_script" / "nnunet_model_registry.json"
    return [local, global_path]


def load_models(
    workspace: str | Path, include_missing: bool = False
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for registry_path in model_registry_paths(workspace):
        payload = read_json(registry_path, {}) or {}
        for row in payload.get("models") or []:
            if not isinstance(row, dict):
                continue
            model_id = str(row.get("model_id") or "")
            model_dir = Path(str(row.get("model_dir") or "")).expanduser()
            manifest_path = Path(str(row.get("manifest_path") or "")).expanduser()
            if not model_id:
                continue
            if not model_dir.is_dir() or not manifest_path.is_file():
                # Flag instead of drop: a registry row whose recorded absolute
                # path is dead (e.g. after moving the checkout to another
                # machine) stays visible so the user can repair or re-import
                # it instead of the model silently disappearing.
                if include_missing:
                    row = dict(row)
                    row["status"] = "missing_path"
                    merged[model_id] = row
                continue
            merged[model_id] = dict(row)
    models_root = workspace_paths(workspace)["models"]
    if models_root.is_dir():
        for manifest_path in models_root.glob("*/*/mimics_model_manifest.json"):
            row = read_json(manifest_path, {}) or {}
            model_id = str(row.get("model_id") or "")
            if not model_id or row.get("schema_version") != MODEL_SCHEMA_VERSION:
                continue
            relocated = dict(row)
            relocated["model_dir"] = str(manifest_path.parent.resolve())
            relocated["manifest_path"] = str(manifest_path.resolve())
            merged[model_id] = relocated
    return sorted(
        merged.values(),
        key=lambda row: float(row.get("created_at_epoch") or 0),
        reverse=True,
    )


def model_usability(model: dict[str, Any]) -> tuple[bool, str]:
    model_dir = Path(str(model.get("model_dir") or "")).expanduser()
    if not model_dir.is_dir():
        return False, "model folder is missing"
    folds = [str(value) for value in model.get("folds") or []]
    if not folds:
        folds = [
            path.name[5:]
            for path in model_dir.glob("fold_*")
            if path.is_dir()
        ]
    if not folds:
        return False, "no trained fold is recorded"
    missing = [
        value
        for value in folds
        if not (model_dir / ("fold_" + value) / "checkpoint_final.pth").is_file()
    ]
    if missing:
        return False, "checkpoint_final.pth is missing for fold(s): {}".format(
            ", ".join(missing)
        )
    if not (model_dir / "plans.json").is_file():
        return False, "plans.json is missing"
    if not (model_dir / "dataset.json").is_file():
        return False, "dataset.json is missing"
    return True, ""


def register_model(workspace: str | Path, manifest: dict[str, Any]) -> None:
    model = dict(manifest)
    model_id = str(model.get("model_id") or "")
    if not model_id:
        raise ValueError("Model manifest has no model_id.")
    for registry_path in model_registry_paths(workspace):
        payload = read_json(registry_path, {}) or {}
        rows = [
            dict(row)
            for row in payload.get("models") or []
            if isinstance(row, dict) and str(row.get("model_id") or "") != model_id
        ]
        rows.append(model)
        rows.sort(
            key=lambda row: float(row.get("created_at_epoch") or 0), reverse=True
        )
        write_json_atomic(
            registry_path,
            {
                "schema_version": "mimics_nnunet_registry.v1",
                "models": rows[:500],
                "updated_at_epoch": time.time(),
            },
        )


def selected_case_ids(value: object) -> set[str] | None:
    if value in (None, "", []):
        return None
    if isinstance(value, str):
        rows = value.replace(";", ",").split(",")
    elif isinstance(value, Iterable):
        rows = value
    else:
        rows = [value]
    selected = {str(item).strip() for item in rows if str(item).strip()}
    return selected or None
