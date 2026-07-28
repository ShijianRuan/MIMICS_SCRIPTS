# -*- coding: utf-8 -*-
"""Shared, relocatable case manifest for Mimics import/export and AI tools.

This module intentionally stays compatible with the Python 3.5 runtime
embedded in supported Mimics releases.
"""

from __future__ import print_function

import json
import os
import re
import shutil
import time
import uuid


MANIFEST_FILENAME = "dataset_manifest.json"
SCHEMA_VERSION = "mimics_dataset_manifest.v1"
_LOCK_SUFFIX = ".lockdir"
_LOCK_STALE_SECONDS = 120.0


def normalize_name(value):
    text = str(value or "").strip().lower()
    normalized = []
    for character in text:
        if character.isalnum() or character in ("_", "-", "."):
            normalized.append(character)
        else:
            normalized.append("_")
    return re.sub(r"_+", "_", "".join(normalized)).strip("._-")


def manifest_path(root_or_path):
    path = os.path.abspath(os.path.expanduser(str(root_or_path or "")))
    if os.path.basename(path).lower() == MANIFEST_FILENAME:
        return path
    return os.path.join(path, MANIFEST_FILENAME)


def _manifest_root(root_or_path):
    path = manifest_path(root_or_path)
    return os.path.dirname(path)


def path_reference(dataset_root, path):
    """Return a path record that remains useful after a directory move."""
    if not path:
        return {}
    root = os.path.abspath(os.path.expanduser(str(dataset_root)))
    absolute = os.path.abspath(os.path.expanduser(str(path)))
    result = {"absolute": absolute}
    try:
        result["relative"] = os.path.relpath(absolute, root)
    except (OSError, ValueError):
        pass
    return result


def resolve_path_reference(dataset_root, reference, require_exists=True):
    """Resolve a string or path record, preferring its relocatable path."""
    if not reference:
        return ""
    root = os.path.abspath(os.path.expanduser(str(dataset_root)))
    if isinstance(reference, dict):
        relative = str(reference.get("relative") or "").strip()
        absolute = str(reference.get("absolute") or reference.get("path") or "").strip()
    else:
        relative = ""
        absolute = str(reference).strip()
        if absolute and not os.path.isabs(absolute):
            relative = absolute
            absolute = ""
    candidates = []
    if relative:
        candidates.append(os.path.abspath(os.path.join(root, relative)))
    if absolute:
        candidates.append(os.path.abspath(os.path.expanduser(absolute)))
    for candidate in candidates:
        if not require_exists or os.path.exists(candidate):
            return candidate
    return ""


def empty_manifest():
    return {
        "schema_version": SCHEMA_VERSION,
        "updated_at_epoch": time.time(),
        "cases": {},
    }


def load_manifest(root_or_path):
    path = manifest_path(root_or_path)
    try:
        with open(path, "r") as handle:
            payload = json.load(handle)
    except Exception:
        return empty_manifest()
    if not isinstance(payload, dict):
        return empty_manifest()
    cases = payload.get("cases")
    if isinstance(cases, list):
        cases = dict(
            (str(row.get("case_id") or ""), row)
            for row in cases
            if isinstance(row, dict) and row.get("case_id")
        )
    if not isinstance(cases, dict):
        cases = {}
    payload["cases"] = cases
    payload["schema_version"] = SCHEMA_VERSION
    return payload


def iter_cases(root_or_path):
    payload = load_manifest(root_or_path)
    rows = []
    for case_id, row in payload.get("cases", {}).items():
        if not isinstance(row, dict):
            continue
        current = dict(row)
        current.setdefault("case_id", case_id)
        rows.append(current)
    rows.sort(key=lambda item: str(item.get("case_id") or "").lower())
    return rows


def find_case(payload_or_root, case_id):
    payload = (
        payload_or_root
        if isinstance(payload_or_root, dict)
        else load_manifest(payload_or_root)
    )
    cases = payload.get("cases") or {}
    wanted = str(case_id or "")
    if wanted in cases and isinstance(cases[wanted], dict):
        return cases[wanted]
    matches = [
        row
        for key, row in cases.items()
        if str(key).lower() == wanted.lower() and isinstance(row, dict)
    ]
    return matches[0] if len(matches) == 1 else None


def resolve_case_path(root_or_path, case_row, key, require_exists=True):
    return resolve_path_reference(
        _manifest_root(root_or_path),
        (case_row or {}).get(key),
        require_exists=require_exists,
    )


def resolve_case_label(root_or_path, case_row, mask_names):
    wanted = set(normalize_name(name) for name in (mask_names or []) if name)
    labels = (case_row or {}).get("labels") or {}
    matches = []
    for key, row in labels.items():
        if not isinstance(row, dict):
            continue
        names = set([
            normalize_name(key),
            normalize_name(row.get("mask_name")),
            normalize_name(row.get("output_name")),
        ])
        names.update(
            normalize_name(value)
            for value in (row.get("aliases") or [])
            if value
        )
        if wanted and not names.intersection(wanted):
            continue
        resolved = resolve_path_reference(
            _manifest_root(root_or_path), row.get("path") or row,
        )
        if resolved:
            matches.append((key, resolved, row))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            "More than one exported label matches {0} for case {1}: {2}".format(
                ", ".join(sorted(wanted)) or "(all labels)",
                (case_row or {}).get("case_id") or "(unknown)",
                ", ".join(str(item[1]) for item in matches),
            )
        )
    return None


def _acquire_lock(path, timeout_seconds=15.0):
    lock_dir = path + _LOCK_SUFFIX
    deadline = time.time() + max(0.1, float(timeout_seconds))
    token = uuid.uuid4().hex
    while time.time() < deadline:
        try:
            os.mkdir(lock_dir)
            try:
                owner_path = os.path.join(lock_dir, "owner.json")
                with open(owner_path, "w") as handle:
                    json.dump(
                        {
                            "pid": os.getpid(),
                            "token": token,
                            "created_at_epoch": time.time(),
                        },
                        handle,
                    )
                return lock_dir, token
            except Exception:
                try:
                    shutil.rmtree(lock_dir)
                except OSError:
                    pass
                raise
        except OSError:
            try:
                age = time.time() - os.path.getmtime(lock_dir)
            except OSError:
                age = 0.0
            if age > _LOCK_STALE_SECONDS:
                try:
                    shutil.rmtree(lock_dir)
                except OSError:
                    pass
            time.sleep(0.05)
    raise OSError("Timed out waiting to update dataset manifest: {0}".format(path))


def _release_lock(lock_dir, token):
    try:
        owner_path = os.path.join(lock_dir, "owner.json")
        with open(owner_path, "r") as handle:
            owner = json.load(handle)
        if str(owner.get("token") or "") != str(token or ""):
            return
    except Exception:
        return
    try:
        shutil.rmtree(lock_dir)
    except OSError:
        pass


def _write_json_atomic(path, payload):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    last_error = None
    for attempt in range(20):
        temporary = path + "." + uuid.uuid4().hex + ".tmp"
        try:
            with open(temporary, "w") as handle:
                handle.write(text)
                try:
                    handle.flush()
                    os.fsync(handle.fileno())
                except Exception:
                    pass
            os.replace(temporary, path)
            return
        except OSError as exc:
            last_error = exc
            try:
                os.remove(temporary)
            except OSError:
                pass
            time.sleep(min(0.25, 0.02 * (attempt + 1)))
    # Some SMB servers intermittently reject replace/rename with WinError 5
    # while still allowing a flushed direct write. The manifest lock prevents
    # competing writers during this bounded fallback.
    try:
        with open(path, "w") as handle:
            handle.write(text)
            try:
                handle.flush()
                os.fsync(handle.fileno())
            except Exception:
                pass
        return
    except OSError as exc:
        last_error = exc
    raise OSError("Could not update {0}: {1}".format(path, last_error))


def update_case(
    root_or_path,
    case_id,
    image_path=None,
    mcs_path=None,
    source_case_dir=None,
    source_geometry=None,
    mimics_geometry=None,
    labels=None,
    provenance=None,
):
    """Merge one case record without deleting fields written by other tools."""
    path = manifest_path(root_or_path)
    root = os.path.dirname(path)
    if root and not os.path.isdir(root):
        try:
            os.makedirs(root)
        except OSError:
            if not os.path.isdir(root):
                raise
    lock_dir, token = _acquire_lock(path)
    try:
        payload = load_manifest(path)
        case_key = str(case_id or "").strip()
        if not case_key:
            raise ValueError("case_id is required")
        current = payload.get("cases", {}).get(case_key)
        if not isinstance(current, dict):
            current = {"case_id": case_key, "labels": {}}
        current["case_id"] = case_key
        if image_path:
            current["image"] = path_reference(root, image_path)
        if mcs_path:
            current["mcs"] = path_reference(root, mcs_path)
        if source_case_dir:
            current["source_case_dir"] = path_reference(root, source_case_dir)
        if source_geometry:
            current["source_geometry"] = dict(source_geometry)
        if mimics_geometry:
            current["mimics_geometry"] = dict(mimics_geometry)
        if provenance:
            current["provenance"] = dict(provenance)
        current_labels = current.get("labels")
        if not isinstance(current_labels, dict):
            current_labels = {}
        for row in labels or []:
            if not isinstance(row, dict) or not row.get("path"):
                continue
            mask_name = str(row.get("mask_name") or row.get("name") or "").strip()
            key = normalize_name(row.get("output_name") or mask_name)
            if not key:
                continue
            label = dict(row)
            label["mask_name"] = mask_name
            label["path"] = path_reference(root, row["path"])
            label["updated_at_epoch"] = time.time()
            current_labels[key] = label
        current["labels"] = current_labels
        current["updated_at_epoch"] = time.time()
        payload["cases"][case_key] = current
        payload["schema_version"] = SCHEMA_VERSION
        payload["updated_at_epoch"] = time.time()
        _write_json_atomic(path, payload)
        return path
    finally:
        _release_lock(lock_dir, token)
