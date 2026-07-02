# -*- coding: utf-8 -*-
"""Shared helpers for Mimics-side Python 3.5 runtime scripts."""

from __future__ import print_function

import json
import os
import subprocess
import uuid


def write_json_atomic(path, value):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    temporary = path + "." + uuid.uuid4().hex + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def read_json(path, default=None):
    try:
        with open(path, "r") as handle:
            return json.load(handle)
    except Exception:
        return default


def safe_filename(value):
    text = str(value or "unknown")
    safe = []
    for char in text:
        if char.isalnum() or char in ("-", "_", "."):
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe) or "unknown"


def safe_slug(value):
    return safe_filename(value).strip("._") or "unknown"


def find_root(start_dir, sentinel_files, max_depth=6):
    current = os.path.abspath(start_dir)
    for _ in range(max_depth):
        for sentinel in sentinel_files:
            if os.path.isfile(os.path.join(current, sentinel)):
                return current
            if os.path.isdir(os.path.join(current, sentinel)):
                return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def hidden_process_kwargs():
    if os.name != "nt":
        return {}

    STARTF_USESHOWWINDOW = 0x00000001
    SW_HIDE = 0
    CREATE_NEW_PROCESS_GROUP = 0x00000200
    CREATE_NO_WINDOW = 0x08000000

    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = SW_HIDE

    return {
        "startupinfo": startupinfo,
        "creationflags": CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP,
    }


def background_process_kwargs(low_priority=True):
    kwargs = hidden_process_kwargs()
    if os.name == "nt" and low_priority:
        BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
        kwargs["creationflags"] = kwargs.get("creationflags", 0) | BELOW_NORMAL_PRIORITY_CLASS
    return kwargs


def background_env(extra=None, include_itk=False):
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    if include_itk:
        env.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "1")
    if extra:
        env.update(extra)
    return env
