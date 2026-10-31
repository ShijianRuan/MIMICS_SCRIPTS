# -*- coding: utf-8 -*-
"""Ownership of image workers across entry-module reloads (Python 3.5)."""
import os
import sys
import uuid

import runtime_common

SCHEMA = "nninteractive_image_worker.v2"


def _session():
    # sys outlives reloads of the scripting entry. A second Mimics process
    # cannot inherit this token or the Popen handles in this interpreter.
    value = getattr(sys, "_mimics_nninteractive_worker_owner", None)
    if not isinstance(value, dict) or value.get("pid") != os.getpid():
        value = {"pid": os.getpid(), "token": uuid.uuid4().hex, "handles": {}}
        sys._mimics_nninteractive_worker_owner = value
    return value


def new_identity():
    return {
        "schema_version": SCHEMA,
        "parent_pid": os.getpid(),
        "parent_start_marker": runtime_common.process_start_marker(os.getpid()),
        "owner_token": _session()["token"],
        "worker_token": uuid.uuid4().hex,
    }


def attach_process(worker_dir, process, identity):
    result = dict(identity)
    result["worker_start_marker"] = runtime_common.process_start_marker(process.pid)
    _session()["handles"][os.path.realpath(worker_dir)] = process
    return result


def owned(worker):
    """Existence of an arbitrary PID is never ownership evidence."""
    try:
        if (worker.get("schema_version") != SCHEMA
                or worker.get("owner_token") != _session()["token"]
                or int(worker.get("parent_pid") or 0) != os.getpid()
                or not worker.get("worker_token")):
            return False
        folder = os.path.realpath(worker["worker_dir"])
        initialize = runtime_common.read_json(os.path.join(folder, "initialize.json"), {}) or {}
        identity = initialize.get("worker_identity") or {}
        if any(identity.get(key) != worker.get(key) for key in
               ("owner_token", "worker_token", "parent_pid", "parent_start_marker")):
            return False
        pid = int(worker.get("pid") or 0)
        if pid <= 0 or pid == os.getpid() or not runtime_common.process_exists(pid):
            return False
        parent_marker = str(worker.get("parent_start_marker") or "")
        if parent_marker and parent_marker != runtime_common.process_start_marker(os.getpid()):
            return False
        handle = _session()["handles"].get(folder)
        handle_live = bool(handle is not None and handle.pid == pid and handle.poll() is None)
        marker = str(worker.get("worker_start_marker") or "")
        actual = runtime_common.process_start_marker(pid) if marker else ""
        # Popen retains the actual child handle on platforms without markers.
        if marker and actual:
            return marker == actual
        return handle_live
    except Exception:
        return False


def process_handle(worker):
    if not owned(worker):
        return None
    return _session()["handles"].get(os.path.realpath(worker["worker_dir"]))


def forget(worker):
    _session()["handles"].pop(os.path.realpath(str(worker.get("worker_dir") or "")), None)
