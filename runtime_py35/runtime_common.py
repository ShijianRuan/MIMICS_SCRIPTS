# -*- coding: utf-8 -*-
"""Shared helpers for Mimics-side Python 3.5 runtime scripts."""

from __future__ import print_function

import json
import os
import subprocess
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
                    pass
        temporary = path + "." + uuid.uuid4().hex + ".tmp"
        try:
            with open(temporary, "w") as handle:
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
    if last_error is not None:
        raise last_error


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


def find_mimics_exe():
    """Find MimicsResearch.exe using the most reliable methods first.

    Search order:
      1. MIMICS_EXE environment variable (explicit override)
      2. Hardcoded path constant (HARDCODED_MIMICS_EXE) — for machines where
         the auto-detection below fails; set it to the full path of
         MimicsResearch.exe on that machine.
      3. sys.executable — when running inside Mimics, this IS
         MimicsResearch.exe (or its Python wrapper).  Walk up from
         the executable's directory to find MimicsResearch.exe.
      4. Windows registry (Uninstall keys for Materialise Mimics)
      5. Program Files sub-directories (any Mimics version)
      6. Drive-root scan on C/D/E/F for common folder names
    """
    # 1. Explicit override
    env_exe = os.environ.get("MIMICS_EXE", "").strip()
    if env_exe and os.path.isfile(env_exe):
        return os.path.abspath(env_exe)

    # 2. Hardcoded fallback for machines where auto-detection fails.
    if HARDCODED_MIMICS_EXE and os.path.isfile(HARDCODED_MIMICS_EXE):
        return os.path.abspath(HARDCODED_MIMICS_EXE)

    # 2. sys.executable — inside Mimics this is the Mimics process itself
    #    or a Python DLL host beside MimicsResearch.exe.
    try:
        exe_dir = os.path.dirname(os.path.abspath(sys.executable))
    except Exception:
        exe_dir = ""
    if exe_dir:
        # The executable itself might be MimicsResearch.exe
        try:
            exe_name = os.path.basename(os.path.abspath(sys.executable)).lower()
        except Exception:
            exe_name = ""
        if exe_name == "mimicsresearch.exe":
            return os.path.abspath(sys.executable)
        # Walk up from exe_dir looking for MimicsResearch.exe
        walk = exe_dir
        for _ in range(5):
            candidate = os.path.join(walk, "MimicsResearch.exe")
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
            parent = os.path.dirname(walk)
            if parent == walk:
                break
            walk = parent

    # 3. Windows registry — check Uninstall keys for any Mimics version
    if os.name == "nt":
        try:
            import _winreg as winreg
        except ImportError:
            try:
                import winreg
            except ImportError:
                winreg = None
        if winreg is not None:
            for hive_key, hive_name in [(winreg.HKEY_LOCAL_MACHINE, "HKLM"), (winreg.HKEY_CURRENT_USER, "HKCU")]:
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
                            try:
                                sub_key = winreg.OpenKey(base, sub_name, 0, winreg.KEY_READ)
                                loc, _ = winreg.QueryValueEx(sub_key, "InstallLocation")
                                winreg.CloseKey(sub_key)
                            except Exception:
                                continue
                            if loc and os.path.isdir(loc):
                                candidate = os.path.join(loc, "MimicsResearch.exe")
                                if os.path.isfile(candidate):
                                    winreg.CloseKey(base)
                                    return os.path.abspath(candidate)
                    finally:
                        winreg.CloseKey(base)

    # 4. Program Files sub-directories — scan for any Mimics* folder
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
            candidate = os.path.join(pf, name, "MimicsResearch.exe")
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)

    # 5. Drive-root scan for common folder patterns
    for drive in ("C:", "D:", "E:", "F:"):
        try:
            entries = os.listdir(drive + "\\")
        except Exception:
            continue
        for name in sorted(entries, reverse=True):
            lower = name.lower()
            if "mimics" not in lower:
                continue
            candidate = os.path.join(drive + "\\", name, "MimicsResearch.exe")
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)

    return None


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
    # Remove variables that can hijack the child Python's import system.
    # The bridge runs under nninteractive_env/python.exe (3.13) whose
    # python313._pth controls sys.path.  If PYTHONPATH / PYTHONHOME from
    # the Mimics host (Python 3.5) leak through, the child tries to import
    # from the wrong stdlib and crashes immediately.
    for _var in ("PYTHONPATH", "PYTHONHOME", "PYTHONDONTWRITEBYTECODE"):
        env.pop(_var, None)
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
    return os.path.join(project_root, ".mimics_runtime", "locks")


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
        except Exception:
            return False
    try:
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        # Declare pointer-sized HANDLE signatures. Without restype = c_void_p,
        # ctypes defaults OpenProcess to c_int and truncates the 64-bit handle,
        # which intermittently returns a zero handle for a live process and
        # makes stale-lock / pid-alive checks wrongly report the process dead.
        kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        kernel32.GetExitCodeProcess.restype = ctypes.c_int
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.restype = ctypes.c_int
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, value)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_uint32(0)
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)) == 0:
                return False
            return int(exit_code.value) == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    except Exception:
        return False


def active_resource_lock(project_root, name):
    """Return a live resource lock, removing a stale lock when safe."""
    path = resource_lock_path(project_root, name)
    payload = read_json(path, {}) or {}
    if not payload:
        if os.path.isfile(path) and _invalid_lock_file_is_old(path):
            try:
                os.remove(path)
            except OSError:
                pass
        return None
    if process_exists(payload.get("pid")):
        return payload
    try:
        os.remove(path)
    except OSError:
        pass
    return None


def resource_lock_summary(payload):
    if not isinstance(payload, dict) or not payload:
        return "available"
    owner = str(payload.get("owner") or payload.get("resource") or "another task")
    kind = str(payload.get("kind") or "").strip()
    pid = payload.get("pid", "?")
    return "{0}{1} (PID {2})".format(owner, " / " + kind if kind else "", pid)


def terminate_process_async(process=None, pid=None, graceful_seconds=2.0, on_complete=None):
    """Terminate an owned child without waiting on the Mimics GUI thread."""
    try:
        target_pid = int(pid or getattr(process, "pid", 0) or 0)
    except Exception:
        target_pid = 0
    if target_pid <= 0:
        return False

    def _reap():
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
                    subprocess.Popen(
                        ["taskkill", "/PID", str(target_pid), "/T", "/F"],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        **hidden_process_kwargs()
                    ).wait()
                else:
                    try:
                        os.kill(target_pid, 9)
                    except Exception:
                        pass
            if process is not None:
                try:
                    process.wait()
                except Exception:
                    pass
        finally:
            if on_complete is not None:
                try:
                    on_complete()
                except Exception:
                    pass

    thread = threading.Thread(target=_reap, name="MimicsOwnedProcessReaper")
    thread.daemon = True
    thread.start()
    return True


def _resource_lock_payload(resource, owner, pid, token):
    return {
        "schema_version": "mimics_script_resource_lock.v1",
        "resource": resource,
        "owner": owner,
        "pid": int(pid),
        "token": token,
        "created_at_epoch": time.time(),
    }


def acquire_resource_lock(path, resource, owner, wait_seconds=0.0, poll_seconds=2.0):
    token = uuid.uuid4().hex
    deadline = time.time() + max(0.0, float(wait_seconds))
    while True:
        parent = os.path.dirname(path)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent)
        payload = _resource_lock_payload(resource, owner, os.getpid(), token)
        payload["path"] = path
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))
            finally:
                os.close(fd)
            return token
        except OSError:
            current = read_json(path, {}) or {}
            if (not current and _invalid_lock_file_is_old(path)) or (current and not process_exists(current.get("pid"))):
                try:
                    os.remove(path)
                except Exception:
                    pass
                continue
            if time.time() >= deadline:
                return None
            time.sleep(max(0.25, float(poll_seconds)))


def update_resource_lock_pid(path, token, pid, extra=None):
    current = read_json(path, {}) or {}
    if current.get("token") != token:
        return False
    current["pid"] = int(pid)
    current["updated_at_epoch"] = time.time()
    if extra:
        current.update(extra)
    write_json_atomic(path, current)
    return True


def release_resource_lock(path, token):
    current = read_json(path, {}) or {}
    if token and current.get("token") != token:
        return False
    for attempt in range(8):
        try:
            os.remove(path)
            return True
        except OSError:
            if not os.path.exists(path):
                return True
            time.sleep(min(0.15, 0.03 * (attempt + 1)))
    return False


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
        payload = read_json(path, {}) or {}
        if not payload and not _invalid_lock_file_is_old(path):
            continue
        if payload and process_exists(payload.get("pid")):
            continue
        try:
            os.remove(path)
            removed += 1
        except OSError:
            pass
    return removed


def _invalid_lock_file_is_old(path):
    try:
        age = time.time() - os.path.getmtime(path)
    except Exception:
        return False
    return age >= INVALID_LOCK_GRACE_SECONDS
