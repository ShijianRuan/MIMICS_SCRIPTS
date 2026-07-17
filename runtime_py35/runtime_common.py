# -*- coding: utf-8 -*-
"""Shared helpers for Mimics-side Python 3.5 runtime scripts."""

from __future__ import print_function

import json
import hashlib
import errno
import os
import subprocess
import sys
import threading
import time
import uuid


INVALID_LOCK_GRACE_SECONDS = 5.0

# Hardcoded MimicsResearch.exe path for machines where find_mimics_exe()'s
# auto-detection (sys.executable, registry, Program Files, drive scan) fails.
# Set this to the full path on the target machine, e.g.
#   r"C:\Program Files\Materialise\Mimics Research 25\MimicsResearch.exe"
# Leave empty to rely on auto-detection.
HARDCODED_MIMICS_EXE = r""
_MIMICS_EXE_CACHE = {"value": None, "checked_at": 0.0}
_MIMICS_EXE_CACHE_LOCK = threading.Lock()
_MIMICS_EXE_SCAN_IN_PROGRESS = False
_ATOMIC_WRITE_LOCKS = {}
_ATOMIC_WRITE_LOCKS_GUARD = threading.Lock()
_LOCAL_OPERATION_GUARD = threading.RLock()
_LOCAL_OPERATIONS = {}


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


def import_queue_runtime_dir(project_root, output_dir):
    """Return a stable local control directory for one .mcs output folder."""
    normalized = os.path.normcase(os.path.abspath(output_dir))
    digest = hashlib.sha1(normalized.encode("utf-8", "replace")).hexdigest()[:16]
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
            digest = hashlib.sha1(os.path.normcase(project).encode("utf-8", "replace")).hexdigest()[:16]
            return os.path.join(local_base, "Mimics-Script", digest)
        return os.path.join(os.getcwd(), ".mimics_runtime")
    return os.path.join(project, ".mimics_runtime")


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


def find_mimics_exe(force_refresh=False):
    """Find a separate Mimics executable for background automation.

    Search order:
      1. MIMICS_BACKGROUND_EXE / MIMICS_EXE explicit override
      2. Hardcoded path constant (HARDCODED_MIMICS_EXE) — for machines where
         the auto-detection below fails; set it to the full path of
         MimicsResearch.exe on that machine.
      3. The current installation directory, preferring MimicsResearch.exe.
      4. Windows registry (Uninstall keys for Materialise Mimics)
      5. Program Files sub-directories (any Mimics version)
      6. Drive-root scan on C/D/E/F for common folder names
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

    # 2. Hardcoded fallback for machines where auto-detection fails.
    if HARDCODED_MIMICS_EXE and os.path.isfile(HARDCODED_MIMICS_EXE):
        value = os.path.abspath(HARDCODED_MIMICS_EXE)
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
        # 4. Windows registry — check Uninstall keys for any Mimics version.
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

        # 5. Program Files sub-directories — scan for any Mimics* folder.
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

        # 6. Drive-root scan for common folder patterns.
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


def background_env(extra=None, include_itk=False):
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
    if include_itk:
        env.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "1")
    if extra:
        env.update(extra)
    return env


def auto_cleanup_enabled():
    value = os.environ.get("MIMICS_AUTO_CLEANUP_ON_START", "1").strip().lower()
    return value not in ("0", "false", "no", "off")


def aggressive_auto_cleanup_enabled():
    value = os.environ.get("MIMICS_AGGRESSIVE_AUTO_CLEANUP_ON_START", "").strip().lower()
    return value in ("1", "true", "yes", "on")


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
            digest = hashlib.sha1(
                os.path.normcase(project).encode("utf-8", "replace")
            ).hexdigest()[:16]
            return os.path.join(local_base, "Mimics-Script", digest, "locks")
    return os.path.join(project, ".mimics_runtime", "locks")


def resource_lock_path(project_root, name):
    return os.path.join(resource_lock_dir(project_root), name)


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
