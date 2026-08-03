#!/usr/bin/env python3
"""Small persistent preferences shared by external Mimics-Script UIs."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any


def preferences_path() -> Path:
    configured = os.environ.get("MIMICS_USER_CONFIG_DIR", "").strip()
    if configured:
        root = Path(os.path.expandvars(os.path.expanduser(configured)))
    elif os.name == "nt" and (os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")):
        root = Path(os.environ.get("LOCALAPPDATA") or os.environ["APPDATA"]) / "Mimics-Script"
    else:
        root = Path.home() / ".mimics_script"
    return root / "ui_preferences.json"


def load_preferences(section: str) -> dict[str, Any]:
    try:
        payload = json.loads(preferences_path().read_text(encoding="utf-8"))
    except Exception:
        return {}
    value = payload.get(str(section)) if isinstance(payload, dict) else None
    return dict(value) if isinstance(value, dict) else {}


def save_preferences(section: str, values: dict[str, Any]) -> None:
    path = preferences_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            payload = {}
    except Exception:
        payload = {}
    cleaned = {
        str(key): str(value)
        for key, value in (values or {}).items()
        if str(value or "").strip()
    }
    existing = payload.get(str(section))
    merged = dict(existing) if isinstance(existing, dict) else {}
    merged.update(cleaned)
    merged["updated_at_epoch"] = time.time()
    payload[str(section)] = merged
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    last_error = None
    for attempt in range(12):
        temporary = path.with_name(
            path.name + ".{}.{}.tmp".format(os.getpid(), uuid.uuid4().hex)
        )
        try:
            temporary.write_text(text, encoding="utf-8")
            os.replace(str(temporary), str(path))
            return
        except OSError as exc:
            last_error = exc
            try:
                temporary.unlink()
            except OSError:
                pass
            time.sleep(min(0.2, 0.02 * (attempt + 1)))
    if last_error is not None:
        raise last_error
