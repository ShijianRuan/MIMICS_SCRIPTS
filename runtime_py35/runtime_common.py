# -*- coding: utf-8 -*-
"""Shared helpers for Mimics-side Python 3.5 runtime scripts."""

from __future__ import print_function

import json
import errno
import hashlib
import logging
import os
import subprocess
import sys
import threading
import time
import uuid


INVALID_LOCK_GRACE_SECONDS = 5.0

_MIMICS_EXE_CACHE = {"value": None, "checked_at": 0.0}
_MIMICS_EXE_CACHE_LOCK = threading.Lock()
_MIMICS_EXE_SCAN_IN_PROGRESS = False
_ATOMIC_WRITE_LOCKS = {}
_ATOMIC_WRITE_LOCKS_GUARD = threading.Lock()
_LOCAL_OPERATION_GUARD = threading.RLock()
_LOCAL_OPERATIONS = {}


def buffer_byte_view(view):
    """One-dimensional byte view of a voxel buffer, without copying when possible.

    Mimics hands get_voxel_buffer() results to the GUI thread, where a
    full-volume .tobytes() copy (hundreds of MB for a CT) freezes Mimics.
    The memoryview cast keeps the common path zero-copy; the tobytes()
    fallback only fires for buffer objects memoryview cannot wrap.
    """
    try:
        raw = memoryview(view)
        if raw.ndim != 1 or raw.format not in ("B", "b", "c"):
            raw = raw.cast("B")
        elif raw.format != "B":
            raw = raw.cast("B")
        return raw
    except Exception:
        return memoryview(view.tobytes())


def stream_buffer(raw, handle=None, compute_sha=True, progress_callback=None):
    """Digest/write a voxel buffer in chunks, never copying the whole volume.

    Each chunk is written to ``handle`` (if given) and folded into the SHA-256
    digest (if requested); ``progress_callback`` (typically a GUI pump) runs
    between chunks so long exports cannot freeze the Mimics UI.
    """
    digest = hashlib.sha256() if compute_sha else None
    chunk_bytes = 16 * 1024 * 1024
    byte_count = len(raw)
    for offset in range(0, byte_count, chunk_bytes):
        chunk = raw[offset:min(byte_count, offset + chunk_bytes)]
        if handle is not None:
            handle.write(chunk)
        if digest is not None:
            digest.update(chunk)
        if progress_callback is not None and offset + len(chunk) < byte_count:
            try:
                progress_callback()
            except Exception:
                pass
    if digest is None:
        return ""
    return "sha256:" + digest.hexdigest()


def progress_notice_due(state, key, detail="", interval_seconds=60.0,
                        initial_delay_seconds=0.0, now=None):
    """Return ``(due, elapsed)`` for low-noise progress/wait reporting.

    Mimics timers may poll several times per second.  Callers use this helper
    to report a new wait reason immediately, then repeat it at a restrained
    interval instead of showing modal dialogs or flooding the Mimics log.
    ``state`` is the owning monitor dictionary, so notices disappear with the
    task and cannot leak across workflows.
    """
    current = time.time() if now is None else float(now)
    notices = state.setdefault("_progress_notices", {})
    identity = str(detail or "")
    entry = notices.get(str(key))
    if not entry or entry.get("detail") != identity:
        entry = {
            "detail": identity,
            "started_at_epoch": current,
            "last_notice_at_epoch": None,
        }
        notices[str(key)] = entry
    started = float(entry.get("started_at_epoch") or current)
    elapsed = max(0.0, current - started)
    last_notice = entry.get("last_notice_at_epoch")
    if last_notice is None:
        due = elapsed >= max(0.0, float(initial_delay_seconds))
    else:
        due = current - float(last_notice) >= max(1.0, float(interval_seconds))
    if due:
        entry["last_notice_at_epoch"] = current
    return due, elapsed


def clear_progress_notice(state, key):
    notices = state.get("_progress_notices") or {}
    notices.pop(str(key), None)
    if not notices:
        state.pop("_progress_notices", None)


def execute_mimics_transaction(mimics_module, operation, transaction_name=None):
    """Run one foreground mutation in a Mimics transaction when available.

    The external computations remain outside Mimics.  Only the final, already
    validated write is grouped here so an exception cannot leave a partially
    committed API operation and Mimics can expose it as one undoable action.
    Older Mimics versions without ``Transaction`` keep the existing behavior.
    """
    transaction_class = getattr(mimics_module, "Transaction", None)
    if not callable(transaction_class):
        return operation()
    name = str(transaction_name or "Mimics-Script Mask Update")
    constructor_errors = []
    transaction = None
    try:
        transaction = transaction_class(name)
    except Exception as named_error:
        constructor_errors.append("named: {0}".format(named_error))
        # A few older/fake Mimics runtimes expose a no-argument Transaction.
        # Current Mimics requires a transaction name, so always try the
        # documented named form first.
        try:
            transaction = transaction_class()
        except Exception as fallback_error:
            constructor_errors.append(
                "no-argument: {0}".format(fallback_error)
            )
            # Some Mimics compatibility bindings expose object.__new__ (which
            # rejects constructor arguments) together with an __init__ that
            # requires transaction_name. Construct and initialize explicitly.
            try:
                transaction = transaction_class.__new__(transaction_class)
                transaction_class.__init__(transaction, name)
            except Exception as explicit_error:
                constructor_errors.append(
                    "explicit initialization: {0}".format(explicit_error)
                )
                transaction = None

    if transaction is None:
        message = (
            "Mimics Transaction is unavailable for '{0}'. The validated Mask "
            "write will continue without transaction grouping; use Mimics Undo "
            "if the result is not wanted. Details: {1}"
        ).format(name, "; ".join(constructor_errors))
        try:
            mimics_module.logging.log_user_message(
                level=logging.WARNING, message=message
            )
        except Exception:
            pass
        return operation()

    error = None
    rollback_error = None
    result = None
    with transaction:
        try:
            result = operation()
            transaction.commit()
        except BaseException:
            error = sys.exc_info()
            try:
                transaction.rollback()
            except Exception as exc:
                rollback_error = exc
    if error is not None:
        _error_type, error_value, traceback_value = error
        if rollback_error is not None:
            raise RuntimeError(
                "The Mimics operation failed and its transaction could not "
                "confirm rollback. Original error: {0}; rollback error: {1}".format(
                    error_value, rollback_error
                )
            )
        raise error_value.with_traceback(traceback_value)
    return result


def try_acquire_local_operation(resource, owner):
    """Claim a Mimics-host operation that must not be re-entered by timers."""
    token = uuid.uuid4().hex
    with _LOCAL_OPERATION_GUARD:
        if resource in _LOCAL_OPERATIONS:
            return None
        _LOCAL_OPERATIONS[resource] = {
            "resource": str(resource),
            "owner": str(owner),
            "token": token,
            "started_at_epoch": time.time(),
        }
    return token


def release_local_operation(resource, token):
    with _LOCAL_OPERATION_GUARD:
        current = _LOCAL_OPERATIONS.get(resource)
        if not current:
            return True
        if token and current.get("token") != token:
            return False
        _LOCAL_OPERATIONS.pop(resource, None)
    return True


def active_local_operation(resource):
    with _LOCAL_OPERATION_GUARD:
        current = _LOCAL_OPERATIONS.get(resource)
        return dict(current) if current else None


def active_local_operations():
    """Return a stable snapshot of Mimics-thread operations currently owned."""
    with _LOCAL_OPERATION_GUARD:
        return [dict(value) for value in _LOCAL_OPERATIONS.values()]


def clear_local_operations():
    """Release in-process leases after all owning monitors were detached.

    This is intentionally reserved for the explicit global-stop path. Normal
    tasks must release their token so one task cannot accidentally unlock
    another task's Mimics buffer operation.
    """
    with _LOCAL_OPERATION_GUARD:
        count = len(_LOCAL_OPERATIONS)
        _LOCAL_OPERATIONS.clear()
    return count


def active_runtime_blockers(project_root, exclude_modules=None):
    """Describe live integration work that makes environment mutation unsafe.

    This census is deliberately lightweight and bounded: it reads in-memory
    monitors plus the small resource-lock directory. It does not enumerate
    Windows processes or recursively scan job histories on Mimics' GUI thread.
    """
    excluded = set(str(value) for value in (exclude_modules or ()))
    blockers = []

    for operation in active_local_operations():
        blockers.append(
            "Mimics operation: {0}".format(
                operation.get("owner") or operation.get("resource") or "unknown"
            )
        )

    collections = (
        ("io_setup_mimics", "_IO_SETUP_MONITORS", "data path window"),
        ("mimics_import", "_IMPORT_MONITORS", "data import"),
        ("mimics_export", "_EXPORT_MONITORS", "mask export"),
        ("mask_import", "_MASK_IMPORT_MONITORS", "mask import"),
        ("nninteractive_mimics", "_ASYNC_MONITORS", "nnInteractive prediction"),
        (
            "nninteractive_finetune_mimics",
            "_CHOOSER_MONITORS",
            "nnInteractive model window",
        ),
        ("interactive_algorithms_mimics", "_MONITORS", "interactive algorithm"),
        ("nnunet_mimics", "_MONITORS", "nnU-Net task"),
        ("setup_environment", "_MONITORS", "environment setup"),
    )
    for module_name, collection_name, label in collections:
        if module_name in excluded:
            continue
        module = sys.modules.get(module_name)
        collection = getattr(module, collection_name, {}) if module is not None else {}
        active = False
        for item in list((collection or {}).values()):
            if item and not item.get("done"):
                active = True
                break
        if active:
            blockers.append(label)

    # A reusable image worker owns imported modules and may retain a CUDA
    # context even when no prompt result monitor is currently active.
    if "nninteractive_mimics" not in excluded:
        module = sys.modules.get("nninteractive_mimics")
        workers = getattr(module, "_ASYNC_IMAGE_WORKERS", {}) if module is not None else {}
        for worker in list((workers or {}).values()):
            pid = worker.get("pid") if isinstance(worker, dict) else None
            if pid and process_exists(pid):
                blockers.append("nnInteractive image worker")
                break

    # Training/status/setup windows also execute from nninteractive_env. Do
    # not repair packages underneath a still-running external GUI process.
    if "nninteractive_finetune_mimics" not in excluded:
        module = sys.modules.get("nninteractive_finetune_mimics")
        processes = getattr(module, "_GUI_PROCESSES", {}) if module is not None else {}
        for process in list((processes or {}).values()):
            try:
                if process.poll() is None:
                    blockers.append("nnInteractive custom-model window")
                    break
            except Exception:
                continue

    lock_dir = resource_lock_dir(project_root)
    try:
        names = sorted(os.listdir(lock_dir)) if os.path.isdir(lock_dir) else []
    except Exception:
        names = []
    for name in names:
        if not name.endswith(".lock"):
            continue
        path = os.path.join(lock_dir, name)
        payload = read_json(path, {}) or {}
        if payload and _lock_payload_is_live(payload):
            blockers.append(
                "{0}: {1}".format(name, resource_lock_summary(payload))
            )

    unique = []
    seen = set()
    for blocker in blockers:
        text = str(blocker)
        if text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def _install_subprocess_cleanup_guard():
    """Guard Python 3.5 Windows subprocess cleanup against stale bad handles.

    In long-lived Mimics sessions, subprocess._cleanup() can crash with
    WinError 5/6 when subprocess._active contains broken process handles.
    If that happens, every later Popen() may fail before process creation.
    """
    if os.name != "nt":
        return
    original = getattr(subprocess, "_cleanup", None)
    if original is None:
        return
    if getattr(subprocess, "_mimics_cleanup_guard_installed", False):
        return

    def _safe_cleanup():
        try:
            return original()
        except Exception as exc:
            winerror = getattr(exc, "winerror", None)
            if winerror not in (5, 6):
                raise
            active = getattr(subprocess, "_active", None)
            if isinstance(active, list):
                survivors = []
                for proc in list(active):
                    try:
                        if proc.poll() is None:
                            survivors.append(proc)
                    except Exception:
                        # Drop stale/broken entries that trigger WinError 5/6.
                        continue
                active[:] = survivors
            return None

    subprocess._cleanup = _safe_cleanup
    subprocess._mimics_cleanup_guard_installed = True


_install_subprocess_cleanup_guard()


def write_json_atomic(path, value):
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    write_text_atomic(path, text)


def write_text_atomic(path, text):
    parent = os.path.dirname(path)
    normalized = os.path.abspath(path)
    with _ATOMIC_WRITE_LOCKS_GUARD:
        entry = _ATOMIC_WRITE_LOCKS.get(normalized)
        if entry is None:
            entry = {"lock": threading.Lock(), "users": 0}
            _ATOMIC_WRITE_LOCKS[normalized] = entry
        entry["users"] += 1
    try:
        with entry["lock"]:
            return _write_text_atomic_locked(path, text, parent)
    finally:
        with _ATOMIC_WRITE_LOCKS_GUARD:
            entry["users"] -= 1
            if entry["users"] <= 0 and _ATOMIC_WRITE_LOCKS.get(normalized) is entry:
                _ATOMIC_WRITE_LOCKS.pop(normalized, None)


def _write_text_atomic_locked(path, text, parent):
    """Write text reliably on local disks and SMB shares.

    Some Windows SMB servers allow create/write but intermittently reject an
    atomic replace with WinError 5. Readers in this project already tolerate a
    temporarily incomplete JSON document, so after bounded replace retries a
    direct, flushed write is safer than failing the entire import.
    """
    last_error = None
    for attempt in range(12):
        # Re-check the parent each attempt: a concurrent cleanup
        # (_cleanup_job_dir on another monitor tick) can remove the job
        # directory between attempts, and the original code only created it
        # once before the loop, leaving the remaining retries to fail with
        # FileNotFoundError on open().
        if parent and not os.path.isdir(parent):
            try:
                os.makedirs(parent)
            except OSError:
                # Race with another process creating/removing it; if it now
                # exists, continue. If not, open() below will fail and retry.
                if not os.path.isdir(parent):
                    time.sleep(min(0.15, 0.02 * (attempt + 1)))
                    continue
        temporary = path + "." + uuid.uuid4().hex + ".tmp"
        try:
            with open(temporary, "w", encoding="utf-8") as handle:
                handle.write(str(text))
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
                if os.path.isfile(temporary):
                    os.remove(temporary)
            except Exception:
                pass
            time.sleep(min(0.15, 0.02 * (attempt + 1)))
    try:
        if parent and not os.path.isdir(parent):
            try:
                os.makedirs(parent)
            except OSError:
                if not os.path.isdir(parent):
                    raise
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(str(text))
            try:
                handle.flush()
                os.fsync(handle.fileno())
            except Exception:
                pass
        return
    except OSError as exc:
        last_error = exc
    if last_error is not None:
        raise last_error


def read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return default


def rotate_log_file(path, max_bytes=5 * 1024 * 1024, backups=3):
    """Rotate ``path`` to ``path.1`` .. ``path.N`` once it exceeds ``max_bytes``.

    Single home for the rotation logic previously duplicated (verbatim) in
    create_mcs_batch, mimics_export, mimics_import, and nninteractive_mimics.
    Best effort: any failure is swallowed - logging must never crash the
    Mimics-side scripts that call it.
    """
    try:
        if not os.path.isfile(path) or os.path.getsize(path) < max_bytes:
            return
        backups = int(backups)
        if backups <= 0:
            os.remove(path)
            return
        oldest = "{0}.{1}".format(path, backups)
        if os.path.isfile(oldest):
            os.remove(oldest)
        for index in range(backups - 1, 0, -1):
            src = "{0}.{1}".format(path, index)
            dst = "{0}.{1}".format(path, index + 1)
            if os.path.isfile(src):
                os.rename(src, dst)
        os.rename(path, path + ".1")
    except Exception:
        pass


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


def stable_digest_hex(value):
    """Return a deterministic, pure-Python digest suitable for local names.

    Mimics embeds an older Python runtime.  On some installations importing
    ``_hashlib.pyd`` during an import-start callback can terminate the host.
    These identifiers only need to be stable and collision-resistant for local
    queue and lock file names, not cryptographic, so avoid the native module.
    """
    if isinstance(value, bytes):
        data = value
    else:
        data = str(value).encode("utf-8", "replace")
    mask = (1 << 64) - 1
    states = (0xCBF29CE484222325, 0x9E3779B185EBCA87)
    results = []
    for seed in states:
        state = seed
        for byte in bytearray(data):
            state ^= byte
            state = (state * 0x100000001B3) & mask
        results.append("{0:016x}".format(state))
    return "".join(results)


def import_queue_runtime_dir(project_root, output_dir):
    """Return a stable local control directory for one .mcs output folder."""
    normalized = os.path.normcase(os.path.abspath(output_dir))
    digest = stable_digest_hex(normalized)[:16]
    base = safe_slug(os.path.basename(os.path.abspath(output_dir))) or "mcs_output"
    return os.path.join(import_runtime_base(project_root), "import_queues", base + "_" + digest)


def import_runtime_base(project_root):
    """Choose a local import runtime root, with an explicit override."""
    configured = os.environ.get("MIMICS_IMPORT_RUNTIME_DIR", "").strip()
    if configured:
        return os.path.abspath(os.path.expandvars(os.path.expanduser(configured)))
    project = os.path.abspath(project_root)
    if os.name == "nt" and project.startswith("\\\\"):
        local_base = (
            os.environ.get("LOCALAPPDATA")
            or os.environ.get("TEMP")
            or os.environ.get("USERPROFILE")
            or os.path.expanduser("~")
        )
        if local_base:
            digest = stable_digest_hex(os.path.normcase(project))[:16]
            return os.path.join(local_base, "Mimics-Script", digest)
        return os.path.join(os.getcwd(), ".mimics_runtime")
    return os.path.join(project, ".mimics_runtime")


def user_config_dir():
    """Return persistent per-user configuration outside disposable runtimes."""
    configured = os.environ.get("MIMICS_USER_CONFIG_DIR", "").strip()
    if configured:
        return os.path.abspath(os.path.expandvars(os.path.expanduser(configured)))
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if base:
            return os.path.join(base, "Mimics-Script")
    return os.path.join(os.path.expanduser("~"), ".mimics_script")


def find_root(start_dir, sentinel_files=None, max_depth=6):
    """Find the repository root from a file or directory path.

    ``sentinel_files`` is optional because Mimics entry modules historically
    called this helper with only ``__file__``.  Keeping the default here avoids
    signature drift when another entry point adopts the shared helper.
    """
    if sentinel_files is None:
        sentinel_files = (
            "mimics_io_config.json",
            "nninteractive_config.json",
            "runtime_py35",
        )
    elif isinstance(sentinel_files, str):
        sentinel_files = (sentinel_files,)
    current = os.path.abspath(start_dir)
    if os.path.isfile(current) or not os.path.isdir(current):
        current = os.path.dirname(current)
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


def current_project_state(mimics_module):
    """Project-loaded state via the documented API (F17).

    Returns (state, path):
    - ("none", "")     - no project is open (or no path known but the
                         loaded flag is definitively off);
    - ("path", p)      - a project is open and its file path is p;
    - ("unnamed", "")  - a project is open but has no file path yet
                         (never saved);
    - ("unknown", "")  - the query itself failed (API missing/raised):
                         callers must NOT treat this as "no project" and
                         must not open another project over it.

    Uses get_project_information()/is_project_loaded(), the calls the
    Mimics 21.0 scripting docs actually document; get_active_project()
    has no documented support and silently fails on builds without it.
    """
    loaded = None
    if hasattr(mimics_module.file, "is_project_loaded"):
        try:
            loaded = bool(mimics_module.file.is_project_loaded())
        except Exception:
            loaded = None
    info = None
    if hasattr(mimics_module.file, "get_project_information"):
        try:
            info = mimics_module.file.get_project_information()
        except Exception:
            info = None
    if info is None:
        if loaded is False:
            return "none", ""
        return "unknown", ""
    for attr in ("filename", "file_name", "path", "project_path", "project_file"):
        try:
            value = getattr(info, attr, None)
        except Exception:
            value = None
        if value:
            return "path", os.path.abspath(str(value))
    if loaded is False:
        return "none", ""
    # get_project_information returned but carries no path: either an
    # unnamed project or an info object without a path attribute on this
    # build. is_project_loaded (when available) breaks the tie.
    if loaded is True:
        return "unnamed", ""
    try:
        for attr in dir(info):
            if attr.startswith("_"):
                continue
            value = getattr(info, attr, None)
            if value and str(value).lower().endswith(".mcs"):
                return "path", os.path.abspath(str(value))
    except Exception:
        pass
    return "unknown", ""


def external_python_candidates(project_root_dir=None):
    """Canonical candidate list for the external (py3.13) interpreter.

    Single source of truth for every runtime_py35 module that needs to
    spawn a process outside Mimics.  Order matters: the bundled
    python_env wins over nninteractive_env, and env-relative layouts win
    over the root itself (portable bundles place python.exe beside the
    scripts).
    """
    root = os.path.abspath(str(project_root_dir or project_root()))
    candidates = []
    for env_name in ("python_env", "nninteractive_env"):
        for base in (os.path.join(root, env_name), os.path.join(os.path.dirname(root), env_name)):
            candidates.extend((
                os.path.join(base, "python.exe"),
                os.path.join(base, "Scripts", "python.exe"),
                os.path.join(base, "python", "python.exe"),
                os.path.join(base, "bin", "python3"),
                os.path.join(base, "bin", "python"),
            ))
    candidates.append(os.path.join(root, "python.exe"))
    return candidates


def find_external_python(project_root_dir=None, allow_system_python=False):
    """Return the external Python executable path, or '' when not found.

    Environment overrides are honored first so a deployment can point at an
    arbitrary interpreter (MIMICS_BRIDGE_PYTHON / NNINTERACTIVE_PYTHON).
    Inside Mimics, sys.executable may be MimicsResearch.exe itself, so it
    is deliberately NOT used as a fallback unless it lives inside one of
    the candidate environment directories.
    """
    for env_key in ("MIMICS_BRIDGE_PYTHON", "NNINTERACTIVE_PYTHON"):
        value = os.environ.get(env_key, "").strip()
        if value and os.path.isfile(value):
            return os.path.abspath(value)
    for candidate in external_python_candidates(project_root_dir):
        if os.path.isfile(candidate):
            return candidate
    if allow_system_python:
        try:
            import shutil
            for cmd in ("python3", "python"):
                found = shutil.which(cmd)
                if found:
                    return found
        except Exception:
            pass
    return ""


def _current_process_executable():
    """Return the host executable without relying on embedded sys.executable."""
    if os.name == "nt":
        try:
            import ctypes
            buffer_size = 32768
            buffer_value = ctypes.create_unicode_buffer(buffer_size)
            length = ctypes.windll.kernel32.GetModuleFileNameW(None, buffer_value, buffer_size)
            if length and length < buffer_size:
                return os.path.abspath(buffer_value.value)
        except Exception:
            pass
    try:
        return os.path.abspath(sys.executable)
    except Exception:
        return ""


def _configured_mimics_executable():
    """Return an explicit repository-level background Mimics path."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for filename in ("mimics_io_config.json", "nninteractive_config.json"):
        path = os.path.join(project_root, filename)
        try:
            with open(path, "r") as handle:
                payload = json.load(handle)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        raw = str(payload.get("mimics_background_exe") or "").strip()
        if not raw:
            continue
        candidate = os.path.abspath(
            os.path.expandvars(os.path.expanduser(raw))
        )
        if os.path.isfile(candidate):
            return candidate
    return ""


def _running_mimics_research_executable():
    """Find a running MimicsResearch process without PowerShell or WMI."""
    if os.name != "nt":
        return ""
    try:
        import ctypes
        from ctypes import wintypes

        TH32CS_SNAPPROCESS = 0x00000002
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        MAX_PATH = 260
        INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

        class PROCESSENTRY32W(ctypes.Structure):
            _fields_ = [
                ("dwSize", wintypes.DWORD),
                ("cntUsage", wintypes.DWORD),
                ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.c_void_p),
                ("th32ModuleID", wintypes.DWORD),
                ("cntThreads", wintypes.DWORD),
                ("th32ParentProcessID", wintypes.DWORD),
                ("pcPriClassBase", wintypes.LONG),
                ("dwFlags", wintypes.DWORD),
                ("szExeFile", wintypes.WCHAR * MAX_PATH),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateToolhelp32Snapshot.argtypes = [
            wintypes.DWORD,
            wintypes.DWORD,
        ]
        kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        kernel32.Process32FirstW.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESSENTRY32W),
        ]
        kernel32.Process32FirstW.restype = wintypes.BOOL
        kernel32.Process32NextW.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESSENTRY32W),
        ]
        kernel32.Process32NextW.restype = wintypes.BOOL
        kernel32.OpenProcess.argtypes = [
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        ]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.QueryFullProcessImageNameW.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.LPWSTR,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL

        snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
        if not snapshot or snapshot == INVALID_HANDLE_VALUE:
            return ""
        try:
            entry = PROCESSENTRY32W()
            entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
            has_entry = bool(kernel32.Process32FirstW(snapshot, ctypes.byref(entry)))
            while has_entry:
                if str(entry.szExeFile or "").lower() == "mimicsresearch.exe":
                    process = kernel32.OpenProcess(
                        PROCESS_QUERY_LIMITED_INFORMATION,
                        False,
                        int(entry.th32ProcessID),
                    )
                    if process:
                        try:
                            size = wintypes.DWORD(32768)
                            buffer_value = ctypes.create_unicode_buffer(size.value)
                            if kernel32.QueryFullProcessImageNameW(
                                process,
                                0,
                                buffer_value,
                                ctypes.byref(size),
                            ):
                                candidate = os.path.abspath(buffer_value.value)
                                if os.path.isfile(candidate):
                                    return candidate
                        finally:
                            kernel32.CloseHandle(process)
                has_entry = bool(
                    kernel32.Process32NextW(snapshot, ctypes.byref(entry))
                )
        finally:
            kernel32.CloseHandle(snapshot)
    except Exception:
        pass
    return ""


def find_mimics_exe(force_refresh=False):
    """Find a separate Mimics executable for background automation.

    Search order:
      1. MIMICS_BACKGROUND_EXE / MIMICS_EXE explicit override
      2. mimics_background_exe in repository configuration
      3. The current installation directory, preferring MimicsResearch.exe
      4. A running MimicsResearch process, queried through the Win32 API
      5. Windows registry (Uninstall keys for Materialise Mimics)
      6. Program Files sub-directories (any Mimics version)
      7. Drive-root scan on C/D/E/F for common folder names
    """
    global _MIMICS_EXE_SCAN_IN_PROGRESS
    now = time.time()
    with _MIMICS_EXE_CACHE_LOCK:
        cached = _MIMICS_EXE_CACHE.get("value")
    if not force_refresh and cached and os.path.isfile(cached):
        return cached

    # 1. Explicit override
    env_exe = (
        os.environ.get("MIMICS_BACKGROUND_EXE", "").strip()
        or os.environ.get("MIMICS_EXE", "").strip()
    )
    if env_exe and os.path.isfile(env_exe):
        value = os.path.abspath(env_exe)
        with _MIMICS_EXE_CACHE_LOCK:
            _MIMICS_EXE_CACHE.update({"value": value, "checked_at": now})
        return value

    # 2. Explicit repository configuration for managed/offline installations.
    configured_exe = _configured_mimics_executable()
    if configured_exe:
        value = os.path.abspath(configured_exe)
        with _MIMICS_EXE_CACHE_LOCK:
            _MIMICS_EXE_CACHE.update({"value": value, "checked_at": now})
        return value

    # 3. sys.executable — inside Mimics this is the Mimics process itself
    #    or a Python DLL host beside MimicsResearch.exe.
    current_exe = _current_process_executable()
    current_medical_exe = ""
    exe_dir = os.path.dirname(current_exe) if current_exe else ""
    if exe_dir:
        # The executable itself might be MimicsResearch.exe
        try:
            exe_name = os.path.basename(current_exe).lower()
        except Exception:
            exe_name = ""
        if exe_name == "mimicsresearch.exe":
            with _MIMICS_EXE_CACHE_LOCK:
                _MIMICS_EXE_CACHE.update({"value": current_exe, "checked_at": now})
            return current_exe
        current_medical_exe = current_exe if exe_name == "mimicsmedical.exe" else ""
        # Walk up from exe_dir looking for the dedicated background-capable
        # Research executable. The foreground Medical executable is excluded
        # unless the user explicitly selects it through an environment override.
        walk = exe_dir
        for _ in range(5):
            candidate = os.path.join(walk, "MimicsResearch.exe")
            if os.path.isfile(candidate):
                value = os.path.abspath(candidate)
                with _MIMICS_EXE_CACHE_LOCK:
                    _MIMICS_EXE_CACHE.update({"value": value, "checked_at": now})
                return value
            parent = os.path.dirname(walk)
            if parent == walk:
                break
            walk = parent

    def usable_background_candidate(candidate):
        if not candidate or not os.path.isfile(candidate):
            return False
        if current_medical_exe:
            try:
                if os.path.normcase(os.path.abspath(candidate)) == os.path.normcase(
                    os.path.abspath(current_medical_exe)
                ):
                    return False
            except Exception:
                pass
        return True

    # 4. A non-standard installation is often still visible through the
    # running process path even when its installer wrote no InstallLocation.
    running_exe = _running_mimics_research_executable()
    if usable_background_candidate(running_exe):
        value = os.path.abspath(running_exe)
        with _MIMICS_EXE_CACHE_LOCK:
            _MIMICS_EXE_CACHE.update({"value": value, "checked_at": now})
        return value

    # Cache only the expensive registry/Program Files/drive scan. Explicit
    # overrides and the current Mimics executable above must take effect
    # immediately even after an earlier failed lookup.
    with _MIMICS_EXE_CACHE_LOCK:
        checked_at = float(_MIMICS_EXE_CACHE.get("checked_at") or 0.0)
        if not force_refresh and now - checked_at < 30.0:
            return None
        if _MIMICS_EXE_SCAN_IN_PROGRESS:
            return None
        _MIMICS_EXE_SCAN_IN_PROGRESS = True

    found = None
    try:
        # 5. Windows registry — check Uninstall keys for any Mimics version.
        if os.name == "nt":
            try:
                import _winreg as winreg
            except ImportError:
                try:
                    import winreg
                except ImportError:
                    winreg = None
            if winreg is not None:
                for hive_key, _hive_name in [(winreg.HKEY_LOCAL_MACHINE, "HKLM"), (winreg.HKEY_CURRENT_USER, "HKCU")]:
                    for wow64 in (0, winreg.KEY_WOW64_64KEY):
                        try:
                            base = winreg.OpenKey(hive_key, r"Software\Microsoft\Windows\CurrentVersion\Uninstall", 0, winreg.KEY_READ | wow64)
                        except Exception:
                            continue
                        try:
                            idx = 0
                            while True:
                                try:
                                    sub_name = winreg.EnumKey(base, idx)
                                    idx += 1
                                except Exception:
                                    break
                                if "mimics" not in sub_name.lower():
                                    continue
                                sub_key = None
                                try:
                                    sub_key = winreg.OpenKey(base, sub_name, 0, winreg.KEY_READ)
                                    loc, _ = winreg.QueryValueEx(sub_key, "InstallLocation")
                                except Exception:
                                    continue
                                finally:
                                    if sub_key is not None:
                                        try:
                                            winreg.CloseKey(sub_key)
                                        except Exception:
                                            pass
                                if loc and os.path.isdir(loc):
                                    for executable_name in ("MimicsResearch.exe", "MimicsMedical.exe"):
                                        candidate = os.path.join(loc, executable_name)
                                        if usable_background_candidate(candidate):
                                            found = os.path.abspath(candidate)
                                            break
                                if found:
                                    break
                        finally:
                            winreg.CloseKey(base)
                        if found:
                            break
                    if found:
                        break

        # 6. Program Files sub-directories — scan for any Mimics* folder.
        if not found:
            for env_var in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)"):
                pf = os.environ.get(env_var, "")
                if not pf or not os.path.isdir(pf):
                    continue
                try:
                    entries = os.listdir(pf)
                except Exception:
                    continue
                for name in sorted(entries, reverse=True):
                    if "mimics" not in name.lower():
                        continue
                    for executable_name in ("MimicsResearch.exe", "MimicsMedical.exe"):
                        candidate = os.path.join(pf, name, executable_name)
                        if usable_background_candidate(candidate):
                            found = os.path.abspath(candidate)
                            break
                    if found:
                        break
                if found:
                    break

                materialise = os.path.join(pf, "Materialise")
                if os.path.isdir(materialise):
                    try:
                        materialise_entries = os.listdir(materialise)
                    except Exception:
                        materialise_entries = []
                    for name in sorted(materialise_entries, reverse=True):
                        if "mimics" not in name.lower():
                            continue
                        for executable_name in ("MimicsResearch.exe", "MimicsMedical.exe"):
                            candidate = os.path.join(materialise, name, executable_name)
                            if usable_background_candidate(candidate):
                                found = os.path.abspath(candidate)
                                break
                        if found:
                            break
                if found:
                    break

        # 7. Drive-root scan for common folder patterns.
        if not found and os.name == "nt":
            for drive in ("C:", "D:", "E:", "F:"):
                try:
                    entries = os.listdir(drive + "\\")
                except Exception:
                    continue
                for name in sorted(entries, reverse=True):
                    if "mimics" not in name.lower():
                        continue
                    for executable_name in ("MimicsResearch.exe", "MimicsMedical.exe"):
                        candidate = os.path.join(drive + "\\", name, executable_name)
                        if usable_background_candidate(candidate):
                            found = os.path.abspath(candidate)
                            break
                    if found:
                        break
                if found:
                    break
        return found
    finally:
        with _MIMICS_EXE_CACHE_LOCK:
            _MIMICS_EXE_CACHE.update({"value": found, "checked_at": time.time()})
            _MIMICS_EXE_SCAN_IN_PROGRESS = False


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


def background_mimics_command(mimics_exe, runner_path, mimics_log_path=None, script_args=None):
    """Build the documented background-script command consistently."""
    if not str(mimics_exe or "").strip():
        raise ValueError("A Mimics executable is required for background automation.")
    if not str(runner_path or "").strip():
        raise ValueError("A Mimics runner script is required for background automation.")
    command = [str(mimics_exe), "-background_mode"]
    if mimics_log_path:
        command.extend(["-save_log", str(mimics_log_path)])
    command.extend(["-run_script", str(runner_path)])
    for value in script_args or []:
        command.append(str(value))
    return command


def mcs_creation_supervisor_command(python_exe, supervisor_path, mimics_exe,
                                    runner_path, runtime_dir, output_dir,
                                    handshake_path=None, mimics_log_path=None):
    """Build the external supervisor command used by every batch-import entry."""
    required = {
        "Python executable": python_exe,
        "MCS creation supervisor": supervisor_path,
        "Mimics executable": mimics_exe,
        "Mimics runner": runner_path,
        "queue runtime directory": runtime_dir,
        "MCS output directory": output_dir,
    }
    for label, value in required.items():
        if not str(value or "").strip():
            raise ValueError("{} is required.".format(label))
    command = [
        str(python_exe),
        str(supervisor_path),
        "--mimics-exe", str(mimics_exe),
        "--runner", str(runner_path),
        "--runtime-dir", str(runtime_dir),
        "--output-dir", str(output_dir),
    ]
    if handshake_path:
        command.extend(["--handshake", str(handshake_path)])
    if mimics_log_path:
        command.extend(["--mimics-log", str(mimics_log_path)])
    return command


def read_text_tail(path, max_bytes=16384):
    """Read a bounded UTF-8 diagnostic tail from a possibly locked log."""
    if not path or not os.path.isfile(path):
        return ""
    try:
        max_bytes = max(0, int(max_bytes))
    except Exception:
        return ""
    if max_bytes <= 0:
        return ""
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes), os.SEEK_SET)
            data = handle.read(max_bytes)
        return data.decode("utf-8", "replace").strip()
    except Exception:
        return ""


def background_env(extra=None):
    env = os.environ.copy()
    # Remove variables that can hijack the child Python's import system.
    # The bridge runs under nninteractive_env/python.exe (3.13) whose
    # python313._pth controls sys.path.  If PYTHONPATH / PYTHONHOME from
    # the Mimics host (Python 3.5) leak through, the child tries to import
    # from the wrong stdlib and crashes immediately.
    for _var in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(_var, None)
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    if extra:
        env.update(extra)
    return env


def auto_cleanup_enabled():
    value = os.environ.get("MIMICS_AUTO_CLEANUP_ON_START", "1").strip().lower()
    return value not in ("0", "false", "no", "off")


def launch_external_gui_process(cmd, cwd=None, stderr_log=None, extra_pythonpath=None):
    """Start a visible external GUI process (pythonw preferred on Windows).

    The inverse of the hidden background launchers: no CREATE_NO_WINDOW, so
    Tk/PySide windows show up, and pythonw.exe is preferred when the
    command is a python.exe from the same environment (no console flash).
    ``extra_pythonpath`` prepends directories to PYTHONPATH so external
    scripts can import project-local modules.
    """
    launch_cmd = list(cmd)
    if os.name == "nt" and launch_cmd:
        exe = os.path.abspath(str(launch_cmd[0]))
        if os.path.basename(exe).lower() == "python.exe":
            pythonw = os.path.join(os.path.dirname(exe), "pythonw.exe")
            if os.path.isfile(pythonw):
                launch_cmd[0] = pythonw
    env = background_env()
    if extra_pythonpath:
        existing = env.get("PYTHONPATH", "")
        paths = [p for p in existing.split(os.pathsep) if p] if existing else []
        for path in extra_pythonpath:
            path = os.path.abspath(str(path))
            if path not in paths:
                paths.insert(0, path)
        env["PYTHONPATH"] = os.pathsep.join(paths)
    stderr_dest = subprocess.DEVNULL
    stderr_file_handle = None
    if stderr_log:
        try:
            parent = os.path.dirname(os.path.abspath(stderr_log))
            if parent and not os.path.isdir(parent):
                os.makedirs(parent)
            stderr_file_handle = open(stderr_log, "w", encoding="utf-8")
            stderr_dest = stderr_file_handle
        except Exception:
            stderr_dest = subprocess.DEVNULL
    try:
        return subprocess.Popen(
            launch_cmd,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=stderr_dest,
            env=env,
        )
    finally:
        if stderr_file_handle is not None:
            try:
                stderr_file_handle.close()
            except Exception:
                pass


def resource_lock_dir(project_root):
    configured = os.environ.get("MIMICS_RESOURCE_LOCK_DIR", "").strip()
    if configured:
        return os.path.abspath(os.path.expandvars(os.path.expanduser(configured)))
    project = os.path.abspath(project_root)
    if os.name == "nt" and project.startswith("\\\\"):
        local_base = (
            os.environ.get("LOCALAPPDATA")
            or os.environ.get("TEMP")
            or os.environ.get("USERPROFILE")
            or os.path.expanduser("~")
        )
        if local_base:
            digest = stable_digest_hex(os.path.normcase(project))[:16]
            return os.path.join(local_base, "Mimics-Script", digest, "locks")
    return os.path.join(project, ".mimics_runtime", "locks")


def resource_lock_path(project_root, name):
    return os.path.join(resource_lock_dir(project_root), name)


def background_mimics_lock_name(scope):
    """Return a lock name scoped to one independent background Mimics job.

    Set MIMICS_SERIALIZE_BACKGROUND_MIMICS=1 only on installations whose
    license or Mimics build genuinely permits a single background instance.
    """
    serialize = os.environ.get(
        "MIMICS_SERIALIZE_BACKGROUND_MIMICS", ""
    ).strip().lower()
    if serialize in ("1", "true", "yes", "on"):
        return "background_mimics.lock"
    normalized = os.path.normcase(os.path.abspath(str(scope or "default")))
    digest = stable_digest_hex(normalized)[:20]
    return "background_mimics_{0}.lock".format(digest)


def background_mimics_lock_path(project_root, scope):
    return resource_lock_path(project_root, background_mimics_lock_name(scope))


def import_producer_lock_name(output_dir):
    """Return the preparation-writer lock for one shared .mcs queue."""
    normalized = os.path.normcase(os.path.abspath(str(output_dir or "default")))
    digest = stable_digest_hex(normalized)[:20]
    return "import_producer_{0}.lock".format(digest)


def import_producer_lock_path(project_root, output_dir):
    return resource_lock_path(project_root, import_producer_lock_name(output_dir))


def process_exists(pid):
    try:
        value = int(pid)
    except Exception:
        return False
    if value <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(value, 0)
            return True
        except OSError as exc:
            if getattr(exc, "errno", None) == errno.ESRCH:
                return False
            if getattr(exc, "errno", None) == errno.EPERM:
                return True
            return False
        except Exception:
            return False
    try:
        import ctypes
        SYNCHRONIZE = 0x00100000
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        WAIT_OBJECT_0 = 0x00000000
        WAIT_TIMEOUT = 0x00000102
        ERROR_ACCESS_DENIED = 5
        ERROR_INVALID_PARAMETER = 87
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # Declare pointer-sized HANDLE signatures. Without restype = c_void_p,
        # ctypes defaults OpenProcess to c_int and truncates the 64-bit handle,
        # which intermittently returns a zero handle for a live process and
        # makes stale-lock / pid-alive checks wrongly report the process dead.
        kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel32.WaitForSingleObject.restype = ctypes.c_uint32
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.restype = ctypes.c_int
        handle = kernel32.OpenProcess(
            SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, 0, value
        )
        if not handle:
            error = ctypes.get_last_error()
            if error == ERROR_INVALID_PARAMETER:
                return False
            if error == ERROR_ACCESS_DENIED:
                return True
            return True
        try:
            result = int(kernel32.WaitForSingleObject(handle, 0))
            if result == WAIT_TIMEOUT:
                return True
            if result == WAIT_OBJECT_0:
                return False
            return True
        finally:
            kernel32.CloseHandle(handle)
    except Exception:
        # Unknown process state must not invalidate a live cross-process lock.
        return True


def process_start_marker(pid):
    """Return a stable process creation marker when the platform exposes one."""
    try:
        value = int(pid)
    except Exception:
        return ""
    if value <= 0:
        return ""
    if os.name == "nt":
        try:
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

            class FILETIME(ctypes.Structure):
                _fields_ = [
                    ("dwLowDateTime", ctypes.c_uint32),
                    ("dwHighDateTime", ctypes.c_uint32),
                ]

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
            kernel32.OpenProcess.restype = ctypes.c_void_p
            kernel32.GetProcessTimes.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(FILETIME),
                ctypes.POINTER(FILETIME),
                ctypes.POINTER(FILETIME),
                ctypes.POINTER(FILETIME),
            ]
            kernel32.GetProcessTimes.restype = ctypes.c_int
            kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
            kernel32.CloseHandle.restype = ctypes.c_int
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, value)
            if not handle:
                return ""
            try:
                created = FILETIME()
                exited = FILETIME()
                kernel = FILETIME()
                user = FILETIME()
                if kernel32.GetProcessTimes(
                    handle,
                    ctypes.byref(created),
                    ctypes.byref(exited),
                    ctypes.byref(kernel),
                    ctypes.byref(user),
                ) == 0:
                    return ""
                marker = (int(created.dwHighDateTime) << 32) | int(created.dwLowDateTime)
                return str(marker)
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return ""
    proc_stat = "/proc/{0}/stat".format(value)
    try:
        with open(proc_stat, "r", encoding="ascii") as handle:
            fields = handle.read().split()
        if len(fields) > 21:
            return str(fields[21])
    except Exception:
        pass
    return ""


def process_matches(pid, expected_start_marker=None):
    if not process_exists(pid):
        return False
    expected = str(expected_start_marker or "").strip()
    if not expected:
        return True
    current = process_start_marker(pid)
    if not current:
        return True
    return current == expected


def _write_all(fd, data):
    offset = 0
    while offset < len(data):
        written = os.write(fd, data[offset:])
        if not written:
            raise OSError("Could not finish writing the resource lock file.")
        offset += int(written)


def _open_resource_guard(path, wait_seconds=2.0):
    """Acquire a short-lived OS file lock protecting one JSON lock mutation."""
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        try:
            os.makedirs(parent)
        except OSError:
            if not os.path.isdir(parent):
                raise
    # Keep this one-byte file persistent. Deleting a guard while another
    # process can hold/open it may create two independent lock identities.
    guard_path = path + ".guard"
    handle = open(guard_path, "a+b")
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        deadline = time.time() + max(0.0, float(wait_seconds))
        while True:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                # Stamp activity so the stale-guard sweep can trust mtime:
                # without this the mtime stays at creation time forever and
                # a hot resource's guard would look "idle" while in daily use.
                try:
                    os.utime(guard_path, None)
                except OSError:
                    pass
                return handle
            except (IOError, OSError):
                if time.time() >= deadline:
                    handle.close()
                    return None
                time.sleep(0.02)
    except Exception:
        handle.close()
        raise


def _close_resource_guard(handle):
    if handle is None:
        return
    try:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except Exception:
        pass
    try:
        handle.close()
    except Exception:
        pass


def _lock_payload_is_live(payload):
    if not isinstance(payload, dict) or not payload:
        return False
    return process_matches(payload.get("pid"), payload.get("process_start_marker"))


def active_resource_lock(project_root, name):
    """Return a live resource lock, removing a stale lock when safe."""
    path = resource_lock_path(project_root, name)
    guard = _open_resource_guard(path, wait_seconds=0.25)
    if guard is None:
        payload = read_json(path, {}) or {}
        return payload if payload else {"owner": "another task", "pid": "unknown"}
    try:
        payload = read_json(path, {}) or {}
        if not payload:
            if os.path.isfile(path) and _invalid_lock_file_is_old(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
            return None
        if _lock_payload_is_live(payload):
            return payload
        current = read_json(path, {}) or {}
        if current.get("token") != payload.get("token"):
            return current or None
        try:
            os.remove(path)
        except OSError:
            pass
        return None
    finally:
        _close_resource_guard(guard)


def resource_lock_summary(payload):
    if not isinstance(payload, dict) or not payload:
        return "available"
    owner = str(payload.get("owner") or payload.get("resource") or "another task")
    kind = str(payload.get("kind") or "").strip()
    pid = payload.get("pid", "?")
    return "{0}{1} (PID {2})".format(owner, " / " + kind if kind else "", pid)


def terminate_process_async(process=None, pid=None, graceful_seconds=2.0, on_complete=None):
    """Terminate an owned child without waiting on the Mimics GUI thread."""
    process_pid = getattr(process, "pid", None) if process is not None else None
    if pid is not None and process_pid is not None:
        try:
            if int(pid) != int(process_pid):
                return False
        except Exception:
            return False
    try:
        target_pid = int(pid or process_pid or 0)
    except Exception:
        target_pid = 0
    if target_pid <= 0:
        return False

    def _reap():
        stopped = False
        try:
            if process is not None and process.poll() is None:
                try:
                    process.terminate()
                except Exception:
                    pass
            deadline = time.time() + max(0.0, float(graceful_seconds))
            while process_exists(target_pid) and time.time() < deadline:
                time.sleep(0.1)
            if process_exists(target_pid):
                if os.name == "nt":
                    killer = subprocess.Popen(
                        ["taskkill", "/PID", str(target_pid), "/T", "/F"],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        **hidden_process_kwargs()
                    )
                    try:
                        killer.wait(timeout=10.0)
                    except Exception:
                        try:
                            killer.terminate()
                        except Exception:
                            pass
                else:
                    try:
                        os.kill(target_pid, 9)
                    except Exception:
                        pass
            if process is not None:
                try:
                    process.wait(timeout=10.0)
                except Exception:
                    pass
            stopped = not process_exists(target_pid)
        finally:
            if stopped and on_complete is not None:
                try:
                    on_complete()
                except Exception:
                    pass

    thread = threading.Thread(target=_reap, name="MimicsOwnedProcessReaper")
    thread.daemon = True
    thread.start()
    return True


def _resource_lock_payload(resource, owner, pid, token):
    payload = {
        "schema_version": "mimics_script_resource_lock.v1",
        "resource": resource,
        "owner": owner,
        "pid": int(pid),
        "token": token,
        "created_at_epoch": time.time(),
    }
    marker = process_start_marker(pid)
    if marker:
        payload["process_start_marker"] = marker
    return payload


def acquire_resource_lock(path, resource, owner, wait_seconds=0.0, poll_seconds=2.0):
    token = uuid.uuid4().hex
    deadline = time.time() + max(0.0, float(wait_seconds))
    while True:
        parent = os.path.dirname(path)
        if parent and not os.path.isdir(parent):
            try:
                os.makedirs(parent)
            except OSError:
                if not os.path.isdir(parent):
                    if time.time() >= deadline:
                        return None
                    time.sleep(max(0.05, min(0.25, float(poll_seconds))))
                    continue
        payload = _resource_lock_payload(resource, owner, os.getpid(), token)
        payload["path"] = path
        guard = _open_resource_guard(
            path,
            wait_seconds=min(0.5, max(0.05, deadline - time.time()))
            if wait_seconds else 0.25,
        )
        if guard is None:
            if time.time() >= deadline:
                return None
            time.sleep(max(0.05, min(0.25, float(poll_seconds))))
            continue
        try:
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
                    _write_all(fd, data)
                    try:
                        os.fsync(fd)
                    except Exception:
                        pass
                finally:
                    os.close(fd)
                return token
            except OSError:
                current = read_json(path, {}) or {}
                stale = (
                    (not current and _invalid_lock_file_is_old(path))
                    or (current and not _lock_payload_is_live(current))
                )
                if stale:
                    verify = read_json(path, {}) or {}
                    if verify.get("token") == current.get("token"):
                        try:
                            os.remove(path)
                        except Exception:
                            pass
                    continue
        finally:
            _close_resource_guard(guard)
        if time.time() >= deadline:
            return None
        time.sleep(max(0.25, float(poll_seconds)))


def update_resource_lock_pid(path, token, pid, extra=None):
    guard = _open_resource_guard(path, wait_seconds=2.0)
    if guard is None:
        return False
    try:
        current = read_json(path, {}) or {}
        if not token or current.get("token") != token:
            return False
        current["pid"] = int(pid)
        marker = process_start_marker(pid)
        if marker:
            current["process_start_marker"] = marker
        else:
            current.pop("process_start_marker", None)
        current["updated_at_epoch"] = time.time()
        if extra:
            current.update(extra)
        write_json_atomic(path, current)
        return True
    finally:
        _close_resource_guard(guard)


def release_resource_lock(path, token):
    if not token:
        return not os.path.exists(path)
    guard = _open_resource_guard(path, wait_seconds=2.0)
    if guard is None:
        return False
    try:
        for attempt in range(8):
            current = read_json(path, {}) or {}
            if not current:
                return not os.path.exists(path)
            if current.get("token") != token:
                return False
            try:
                os.remove(path)
                return True
            except OSError:
                if not os.path.exists(path):
                    return True
                time.sleep(min(0.15, 0.03 * (attempt + 1)))
        return False
    finally:
        _close_resource_guard(guard)


def cleanup_stale_resource_locks(lock_dir):
    removed = 0
    if not lock_dir or not os.path.isdir(lock_dir):
        return removed
    try:
        names = os.listdir(lock_dir)
    except Exception:
        return removed
    for name in names:
        if not name.endswith(".lock"):
            continue
        path = os.path.join(lock_dir, name)
        guard = _open_resource_guard(path, wait_seconds=0.1)
        if guard is None:
            continue
        try:
            payload = read_json(path, {}) or {}
            if not payload and not _invalid_lock_file_is_old(path):
                continue
            if payload and _lock_payload_is_live(payload):
                continue
            verify = read_json(path, {}) or {}
            if verify.get("token") != payload.get("token"):
                continue
            try:
                os.remove(path)
                removed += 1
            except OSError:
                pass
        finally:
            _close_resource_guard(guard)
    return removed


def _invalid_lock_file_is_old(path):
    try:
        age = time.time() - os.path.getmtime(path)
    except Exception:
        return False
    return age >= INVALID_LOCK_GRACE_SECONDS


# ---------------------------------------------------------------------------
# Process registry (Phase B).
#
# The registry itself lives in resource_locks.py at the project root (the
# only module shared by both the py3.5 Mimics runtime and the external
# py3.13 tooling).  The helpers below give the py3.5 side a stable import
# surface: they resolve resource_locks.py relative to this file's parent
# directory, import it once, and delegate.  The registry mirrors the same
# MIMICS_RESOURCE_LOCK_DIR mapping as resource_lock_dir() above.
# ---------------------------------------------------------------------------

_PROCESS_REGISTRY_MODULE = {"module": None, "error": None}
_PROJECT_ROOT_CACHE = {"value": None}


def project_root():
    """The deploy root that holds resource_locks.py and runtime_py35/."""
    if _PROJECT_ROOT_CACHE["value"] is None:
        _PROJECT_ROOT_CACHE["value"] = os.path.dirname(
            os.path.abspath(os.path.dirname(__file__))
        )
    return _PROJECT_ROOT_CACHE["value"]


def _process_registry_module():
    if _PROCESS_REGISTRY_MODULE["module"] is not None:
        return _PROCESS_REGISTRY_MODULE["module"]
    if _PROCESS_REGISTRY_MODULE["error"] is not None:
        return None
    try:
        import importlib

        root = project_root()
        if root not in sys.path:
            sys.path.insert(0, root)
        _PROCESS_REGISTRY_MODULE["module"] = importlib.import_module(
            "resource_locks"
        )
    except Exception as exc:  # pragma: no cover - defensive
        _PROCESS_REGISTRY_MODULE["error"] = str(exc)
        return None
    return _PROCESS_REGISTRY_MODULE["module"]


def register_process(project_root_dir, role, pid, **kwargs):
    """Delegate to resource_locks.register_process; None when unavailable.

    Registration is best-effort from the py3.5 side: a failure never
    blocks the caller's own spawn logic (the cmdline fallback in the
    sweep still finds unregistered legacy processes).
    """
    module = _process_registry_module()
    if module is None:
        return None
    try:
        return module.register_process(project_root_dir, role, pid, **kwargs)
    except Exception:
        return None


def unregister_process(project_root_dir, role, pid, ownership_token=""):
    module = _process_registry_module()
    if module is None:
        return False
    try:
        return module.unregister_process(
            project_root_dir, role, pid, ownership_token=ownership_token
        )
    except Exception:
        return False


def snapshot_processes(project_root_dir, include_dead=False):
    module = _process_registry_module()
    if module is None:
        return []
    try:
        return module.snapshot_processes(project_root_dir, include_dead=include_dead)
    except Exception:
        return []


def sweep_processes(project_root_dir, protect_roles=()):
    """Delegate to resource_locks.sweep_processes; {} when unavailable."""
    module = _process_registry_module()
    if module is None:
        return {}
    try:
        return module.sweep_processes(project_root_dir, protect_roles=protect_roles)
    except Exception:
        return {}


def terminate_registered_process(project_root_dir, role, pid, graceful_seconds=5.0):
    """The single kill ladder via the registry; False when unavailable."""
    module = _process_registry_module()
    if module is None:
        return False
    try:
        return module.terminate_process(
            project_root_dir, role, pid, graceful_seconds=graceful_seconds
        )
    except Exception:
        return False
