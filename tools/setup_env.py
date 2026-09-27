#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""External environment setup worker for Mimics-Script.

Runs outside Mimics (Python 3.10+) to check and repair the nninteractive_env.

Usage:
    python tools/setup_env.py check                # Validate env, write JSON report
    python tools/setup_env.py install              # Check + pip install missing pkgs
    python tools/setup_env.py install-remote       # Add optional SSH transport
    python tools/setup_env.py extract <archive>    # Extract portable .zip archive
    python tools/setup_env.py setup-from-scratch   # Create full env from scratch

All output is written to a state JSON file so Mimics can poll it without blocking.
"""

from __future__ import print_function

import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATE_FILE = PROJECT_ROOT / ".mimics_runtime" / "setup_env_state.json"
LOG_FILE = PROJECT_ROOT / ".mimics_runtime" / "setup_env.log"

# Import names (used by find_spec / __import__ for checking)
REQUIRED_IMPORTS = [
    "torch",
    "numpy",
    "nibabel",
    "pydicom",
    "SimpleITK",
    "scipy",
    "nnInteractive",
    "torchvision",
    "transformers",
    "yaml",         # pip package is pyyaml
    "tqdm",
    "tensorboard",
    "tomli",
    "onnxruntime",
    "nnunetv2",
    "acvl_utils",
]

# Pip package names (may differ from import names)
REQUIRED_PACKAGES = [
    "torch",
    "numpy",
    "nibabel",
    "pydicom",
    "SimpleITK",
    "scipy",
    "nnInteractive",
    "torchvision",
    "transformers",
    "pyyaml",       # import name is yaml
    "tqdm",
    "tensorboard",
    "tomli>=2.0",
    "onnxruntime-gpu",
    "nnunetv2>=2.8.1,<2.9",
]

# Preferred external GUI backend for advanced setup/status windows.
# PySide6 is intentionally kept separate from the core AI packages so checks can
# report UI readiness clearly, while the runtime can still fall back if needed.
GUI_IMPORTS = [
    "PySide6",
    "shiboken6",
]

GUI_PACKAGES = [
    "PySide6",
    "shiboken6",
]

# Optional client transport for SSH/Docker training. It is deliberately not
# part of REQUIRED_PACKAGES so a local-only workstation has no new dependency.
REMOTE_PACKAGES = [
    "paramiko>=3.5,<5",
]

# Mapping for __import__: pip name → import name
_PIP_TO_IMPORT = {
    "pyyaml": "yaml",
    "onnxruntime-gpu": "onnxruntime",
}

# Extra index URL for PyTorch CUDA builds
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu124"

# PyPI mirrors — Chinese mirrors first (much faster from mainland China)
PYPI_MIRRORS = [
    "https://pypi.tuna.tsinghua.edu.cn/simple",
    "https://mirrors.aliyun.com/pypi/simple",
    "",  # default PyPI (fallback, slow from China)
]


ENV_DIR_CANDIDATES = ("python_env", "nninteractive_env")


def env_dir_name():
    """Return the environment directory name to use.

    Prefers an existing directory (python_env first, so renamed installs win);
    defaults to python_env for fresh installs. Legacy nninteractive_env is
    still accepted so old installs keep working untouched.
    """
    for name in ENV_DIR_CANDIDATES:
        if (PROJECT_ROOT / name).is_dir():
            return name
    return ENV_DIR_CANDIDATES[0]


def _find_python():
    """Find the external Python.

    Supports three layouts:
    1. venv-style:     python_env/Scripts/python.exe
    2. embeddable:     python_env/python.exe  (offline bundle copies python/* here)
    3. standalone dir: python_env/python/python.exe
    """
    rels = []
    for name in ENV_DIR_CANDIDATES:
        rels.extend((
            "{0}/python.exe".format(name),
            "{0}/Scripts/python.exe".format(name),
            "{0}/python/python.exe".format(name),
            "{0}/bin/python3".format(name),
            "{0}/bin/python".format(name),
        ))
    for rel in rels:
        p = PROJECT_ROOT / rel
        if p.is_file():
            return str(p)
    # Check for bundled Python (from offline bundle, before env is set up)
    bundled = _find_bundled_python()
    if bundled:
        return bundled
    # Fallback: system python
    for cmd in ("python3", "python"):
        import shutil
        found = shutil.which(cmd)
        if found:
            return found
    return sys.executable


def _find_bundled_python():
    """Find the bundled Python embeddable (from offline bundle)."""
    for rel in (
        "python/python.exe",
        "python/bin/python3",
    ):
        p = PROJECT_ROOT / rel
        if p.is_file():
            return str(p)
    return None


def _write_state(status, **kwargs):
    """Update the JSON state file for Mimics to poll."""
    state = {
        "status": status,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "updated_at_epoch": time.time(),
    }
    state.update(kwargs)
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(str(STATE_FILE), "w") as f:
            json.dump(state, f, indent=2, sort_keys=True)
    except Exception:
        pass  # State file is diagnostic only — never crash the worker


LOG_ROTATE_MAX_BYTES = 5 * 1024 * 1024
LOG_ROTATE_BACKUPS = 3


def _rotate_log(max_bytes=LOG_ROTATE_MAX_BYTES, backups=LOG_ROTATE_BACKUPS):
    """Rotate setup_env.log once it exceeds ``max_bytes`` (.1/.2/.3 suffixes)."""
    try:
        if not LOG_FILE.exists() or LOG_FILE.stat().st_size < max_bytes:
            return
        for index in range(backups - 1, 0, -1):
            src = LOG_FILE.with_suffix(".log.{0}".format(index))
            if src.exists():
                src.replace(LOG_FILE.with_suffix(".log.{0}".format(index + 1)))
        LOG_FILE.replace(LOG_FILE.with_suffix(".log.1"))
    except Exception:
        pass  # Rotation is best effort — never crash the worker


def _log(message):
    """Append a line to the setup log."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = "[{0}] {1}".format(timestamp, message)
    print(line)
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        _rotate_log()
        with open(str(LOG_FILE), "a") as f:
            f.write(line + "\n")
    except Exception:
        pass  # Log file is diagnostic only — never crash the worker


def _run_python(args, timeout=600):
    """Run a command under the nninteractive_env Python, capture output."""
    python = _find_python()
    cmd = [python] + args
    _log("Running: {0} ...".format(" ".join(cmd[:3])))
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        stdout, _ = proc.communicate(timeout=timeout)
        output = stdout.decode("utf-8", "replace") if stdout else ""
        return proc.returncode, output
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        return -1, "Command timed out after {0}s".format(timeout)
    except Exception as exc:
        return -1, str(exc)


def _run_python_script(script_lines, timeout=600):
    """Run a multi-line Python snippet via a temp file to avoid shell length limits."""
    import tempfile
    python = _find_python()
    fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix="mimics_setup_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write("\n".join(script_lines))
        _log("Running script: {0}".format(
            " ".join(script_lines[0][:80].split()[:3]) if script_lines else "empty"))
        proc = subprocess.Popen(
            [python, tmp_path],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        stdout, _ = proc.communicate(timeout=timeout)
        output = stdout.decode("utf-8", "replace") if stdout else ""
        return proc.returncode, output
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        return -1, "Script timed out after {0}s".format(timeout)
    except Exception as exc:
        return -1, str(exc)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def _nnunet_version_supported():
    ret, output = _run_python_script(
        [
            "import re",
            "from importlib.metadata import version",
            "value = version('nnunetv2')",
            "parts = tuple(int(x) for x in re.findall(r'\\d+', value)[:3])",
            "parts = parts + (0,) * (3 - len(parts))",
            "print(value)",
            "raise SystemExit(0 if (2, 8, 1) <= parts < (2, 9, 0) else 2)",
        ],
        timeout=60,
    )
    version_text = output.strip().split("\n")[-1] if output else "unknown"
    return ret == 0, version_text


def _probe_gui_backends():
    """Return GUI backend availability inside nninteractive_env."""
    ret, output = _run_python_script([
        "import json, traceback",
        "r = {'pyside6': False, 'errors': {}}",
        "try:",
        "    import PySide6, shiboken6",
        "    from PySide6 import QtCore, QtWidgets",
        "    r['pyside6'] = True",
        "    r['pyside6_version'] = str(getattr(QtCore, '__version__', 'unknown'))",
        "except Exception as e:",
        "    r['errors']['pyside6'] = repr(e)",
        "print(json.dumps(r))",
    ], timeout=60)
    if ret != 0:
        return {
            "pyside6": False,
            "errors": {"probe": output[-500:] if output else "GUI probe failed"},
        }
    try:
        return json.loads(output.strip().split("\n")[-1])
    except Exception:
        return {
            "pyside6": False,
            "errors": {"probe": output[-500:] if output else "Could not parse GUI probe"},
        }


# ---------------------------------------------------------------------------
# Check
# ---------------------------------------------------------------------------

def check():
    """Validate the environment and write results to the state file."""
    _write_state("checking", step="starting", message="Environment check started.")
    _log("=== Environment Check ===")
    result = {
        "python_version": None,
        "python_exe": _find_python(),
        "packages": {},
        "package_versions": {},
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_version": None,
        "model_files": [],
        "gui_backends": {},
        "preferred_gui_backend": None,
    }

    # 1. Python version
    ret, output = _run_python(["-c", "import sys; print(sys.version)"])
    if ret == 0:
        result["python_version"] = output.strip().split("\n")[0]
        _log("Python: {0}".format(result["python_version"]))
    else:
        result["python_error"] = output[:500]
        _log("Python check failed: {0}".format(output[:200]))
        _write_state("error", message="Python check failed.", detail=result)
        return 1

    # 2. Packages
    ret, output = _run_python_script([
        "import json, importlib.util as U",
        "pkgs = {0}".format(REQUIRED_IMPORTS),
        "r = {}",
        "for p in pkgs:",
        "    s = U.find_spec(p)",
        "    r[p] = s is not None",
        "print(json.dumps(r))",
    ])
    if ret == 0:
        try:
            result["packages"] = json.loads(output.strip().split("\n")[-1])
        except Exception:
            pass
        for pkg, found in result.get("packages", {}).items():
            _log("Package {0}: {1}".format(pkg, "OK" if found else "MISSING"))
    else:
        _log("Package probe failed: {0}".format(output[:200]))

    # 3. Preferred external GUI backend
    gui_backends = _probe_gui_backends()
    result["gui_backends"] = gui_backends
    if gui_backends.get("pyside6"):
        result["preferred_gui_backend"] = "PySide6"
        _log("External GUI: PySide6 OK ({0})".format(gui_backends.get("pyside6_version", "unknown")))
    else:
        result["preferred_gui_backend"] = "none"
        _log("External GUI: no usable backend found.")
        for name, error in sorted((gui_backends.get("errors") or {}).items()):
            _log("GUI {0} error: {1}".format(name, error))

    # 4. Package versions
    ret, output = _run_python_script([
        "import json",
        "pkgs = {0}".format(REQUIRED_IMPORTS),
        "r = {}",
        "for p in pkgs:",
        "    try:",
        "        m = __import__(p.replace('-', '_'))",
        "        r[p] = str(getattr(m, '__version__', '?'))",
        "    except Exception:",
        "        r[p] = 'error'",
        "print(json.dumps(r))",
    ])
    if ret == 0:
        try:
            result["package_versions"] = json.loads(output.strip().split("\n")[-1])
        except Exception:
            pass
    nnunet_version_ok, nnunet_version = _nnunet_version_supported()
    result["nnunet_version_compatible"] = nnunet_version_ok
    result["nnunet_version"] = nnunet_version
    _log(
        "nnU-Net version {0}: {1}".format(
            nnunet_version, "OK" if nnunet_version_ok else "UNSUPPORTED"
        )
    )

    # 5. CUDA
    ret, output = _run_python_script([
        "import json, os, traceback",
        "r = {}",
        "try:",
        "    import torch",
        "    r['torch_version'] = str(getattr(torch, '__version__', '?'))",
        "    r['cuda_compiled_version'] = str(getattr(torch.version, 'cuda', 'unknown'))",
        "    r['cuda_visible_devices'] = os.environ.get('CUDA_VISIBLE_DEVICES')",
        "    r['nvidia_visible_devices'] = os.environ.get('NVIDIA_VISIBLE_DEVICES')",
        "    r['is_available'] = bool(torch.cuda.is_available())",
        "    try:",
        "        r['device_count'] = int(torch.cuda.device_count())",
        "    except Exception as e:",
        "        r['device_count'] = 0",
        "        r['device_count_error'] = repr(e)",
        "    if r.get('device_count', 0) > 0:",
        "        try:",
        "            r['device0_name'] = str(torch.cuda.get_device_name(0))",
        "        except Exception as e:",
        "            r['device0_name_error'] = repr(e)",
        "    try:",
        "        torch.cuda.init()",
        "        r['init_ok'] = True",
        "    except Exception as e:",
        "        r['init_ok'] = False",
        "        r['init_error'] = repr(e)",
        "except Exception as e:",
        "    r['probe_error'] = repr(e)",
        "    r['traceback'] = traceback.format_exc()",
        "print(json.dumps(r))",
    ])
    if ret == 0:
        try:
            cuda_probe = json.loads(output.strip().split("\n")[-1])
            result["cuda_probe"] = cuda_probe
            result["cuda_available"] = bool(cuda_probe.get("is_available", False))
            result["cuda_device_count"] = int(cuda_probe.get("device_count", 0) or 0)
            result["cuda_version"] = str(cuda_probe.get("cuda_compiled_version", "unknown"))
            _log("CUDA: available={0}, devices={1}, version={2}".format(
                result["cuda_available"], result["cuda_device_count"], result["cuda_version"]))
            if cuda_probe.get("device0_name"):
                _log("CUDA device0: {0}".format(cuda_probe.get("device0_name")))
            if not result["cuda_available"]:
                reason = cuda_probe.get("init_error") or cuda_probe.get("probe_error") or cuda_probe.get("device0_name_error") or "unknown"
                _log("CUDA unavailable reason: {0}".format(reason))
                _log("CUDA env: CUDA_VISIBLE_DEVICES={0}, NVIDIA_VISIBLE_DEVICES={1}".format(
                    cuda_probe.get("cuda_visible_devices"), cuda_probe.get("nvidia_visible_devices")))
        except Exception:
            _log("CUDA probe parse failed: {0}".format(output[:200]))
    else:
        _log("CUDA probe failed: {0}".format(output[:200]))

    # 6. Model weights
    model_roots = [
        PROJECT_ROOT / name / "models" for name in ENV_DIR_CANDIDATES
    ]
    for root in model_roots:
        if root.is_dir():
            for f in root.rglob("*"):
                if f.suffix in (".pth", ".safetensors", ".bin", ".pt"):
                    result["model_files"].append(
                        str(f.relative_to(root)).replace("\\", "/"))

    _log("Model files found: {0}".format(len(result["model_files"])))

    # Summary
    missing_pkgs = [p for p, ok in result["packages"].items() if not ok]
    gui_ready = bool(result["gui_backends"].get("pyside6"))
    all_ok = (
        result["python_version"] is not None
        and len(missing_pkgs) == 0
        and gui_ready
        and nnunet_version_ok
    )

    _write_state(
        "ok" if all_ok else "incomplete",
        message=(
            "All checks passed."
            if all_ok else
            "{0} package(s) missing; PySide6 GUI ready: {1}; nnU-Net version ready: {2}.".format(
                len(missing_pkgs), gui_ready, nnunet_version_ok
            )
        ),
        detail=result,
        missing_packages=missing_pkgs,
        gui_ready=gui_ready,
        preferred_gui_backend=result["preferred_gui_backend"],
        all_ok=all_ok,
    )
    _log("=== Check complete: {0} ===".format("OK" if all_ok else "INCOMPLETE"))
    return 0 if all_ok else 1


# ---------------------------------------------------------------------------
# Install / repair
# ---------------------------------------------------------------------------

def _offline_wheels_dir():
    """Return the path to the offline wheels directory if it exists."""
    path = PROJECT_ROOT / "wheels"
    return str(path) if path.is_dir() else None


def _pip_install(packages):
    """Install packages via pip. Uses local wheels if available (offline mode)."""
    if not packages:
        return True, ""
    python = _find_python()
    # Ensure pip is available
    ret, _ = _run_python(["-m", "pip", "--version"], timeout=60)
    if ret != 0:
        return False, "pip is not available"

    wheels_dir = _offline_wheels_dir()
    if wheels_dir:
        _log("[OFFLINE] Installing from local wheels: {0}".format(wheels_dir))
        # --no-index guarantees ZERO network access
        ret, output = _run_python(
            ["-m", "pip", "install", "--no-index", "--find-links", wheels_dir] + packages,
            timeout=1800,
        )
        if ret == 0:
            _log("[OFFLINE] All packages installed from local wheels.")
            return True, ""
        # Offline install must succeed — do NOT fall through to online.
        # The offline bundle should be complete.
        return False, "Offline install failed: {0}".format(output[-500:])

    # No offline wheels — must use network. Try mirrors.
    if "torch" in packages:
        for mirror in [TORCH_INDEX_URL]:
            _log("Trying PyTorch from {0}...".format(mirror or "default index"))
            ret, output = _run_python(
                ["-m", "pip", "install", "torch", "--extra-index-url", mirror],
                timeout=1800,
            )
            if ret == 0:
                break
        else:
            _log("PyTorch install failed from all sources.")
            return False, "Failed to install torch: {0}".format(output[-500:] if 'output' in dir() else "all mirrors failed")

    # Install remaining packages with mirror fallback
    remaining = [p for p in packages if p != "torch"]
    if remaining:
        for mirror in PYPI_MIRRORS:
            args = ["-m", "pip", "install", "--upgrade"]
            if mirror:
                args += ["-i", mirror]
            args += remaining
            _log("Trying pip install from {0}...".format(mirror or "default PyPI"))
            ret, output = _run_python(args, timeout=900)
            if ret == 0:
                return True, ""
        return False, "pip install failed from all mirrors: {0}".format(output[-500:] if 'output' in dir() else "unknown error")
    return True, ""


def install():
    """Check and install missing packages."""
    _write_state("installing", step="checking", message="Checking environment...")
    _log("=== Environment Install/Repair ===")

    # First check
    check_result = check_result_dict()
    missing = [p for p, ok in check_result.get("packages", {}).items() if not ok]
    if not check_result.get("nnunet_version_compatible", False):
        missing = [
            "nnunetv2>=2.8.1,<2.9" if value == "nnunetv2" else value
            for value in missing
        ]
        if not any(value.startswith("nnunetv2") for value in missing):
            missing.append("nnunetv2>=2.8.1,<2.9")
    gui_backends = check_result.get("gui_backends") or {}
    missing_gui = []
    if not gui_backends.get("pyside6"):
        missing_gui = list(GUI_PACKAGES)
        missing.extend(missing_gui)
    seen = set()
    missing = [p for p in missing if not (p in seen or seen.add(p))]

    if not missing:
        _write_state("ok", message="All packages and the PySide6 GUI backend are already installed.", all_ok=True)
        _log("Nothing to install.")
        return 0

    _log("Missing packages: {0}".format(", ".join(missing)))
    _write_state("installing", step="installing",
                 message="Installing {0} package(s)...".format(len(missing)),
                 packages_installing=missing)

    ok, error = _pip_install(missing)
    if not ok:
        _write_state("error", message="Installation failed.", error=error)
        _log("Install failed: {0}".format(error))
        return 1

    # Re-check after install
    _write_state("installing", step="verifying", message="Verifying installation...")
    ret, output = _run_python_script([
        "import json, importlib.util as U",
        "pkgs = {0}".format(REQUIRED_IMPORTS + GUI_IMPORTS),
        "r = {}",
        "for p in pkgs:",
        "    r[p] = U.find_spec(p) is not None",
        "print(json.dumps(r))",
    ])
    if ret == 0:
        try:
            pkg_status = json.loads(output.strip().split("\n")[-1])
            still_missing = [p for p, ok in pkg_status.items() if not ok]
        except Exception:
            still_missing = missing
    else:
        still_missing = missing

    nnunet_version_ok, _nnunet_version = _nnunet_version_supported()
    if not nnunet_version_ok:
        still_missing.append("nnunetv2>=2.8.1,<2.9")
    if still_missing:
        _write_state("incomplete",
                     message="{0} package(s) still missing.".format(len(still_missing)),
                     missing_packages=still_missing)
        _log("Still missing: {0}".format(", ".join(still_missing)))
        return 1

    _write_state("ok", message="All packages installed successfully.", all_ok=True)
    _log("=== Install complete: OK ===")
    return 0


def install_remote():
    """Install only the optional SSH client without changing local AI packages."""
    _write_state(
        "installing",
        step="installing_remote_transport",
        message="Installing optional remote training transport...",
    )
    ret, _output = _run_python(["-c", "import paramiko"], timeout=60)
    if ret == 0:
        _write_state(
            "ok",
            message="Remote training transport is already installed.",
            remote_training_ready=True,
        )
        return 0
    ok, error = _pip_install(REMOTE_PACKAGES)
    if not ok:
        _write_state(
            "error",
            message="Remote training transport installation failed.",
            error=error,
            local_training_unaffected=True,
        )
        return 1
    ret, output = _run_python(
        [
            "-c",
            "import paramiko; print(paramiko.__version__)",
        ],
        timeout=60,
    )
    if ret != 0:
        _write_state(
            "error",
            message="Paramiko was installed but could not be imported.",
            error=output[-500:],
            local_training_unaffected=True,
        )
        return 1
    _write_state(
        "ok",
        message="Remote training transport is ready.",
        remote_training_ready=True,
        paramiko_version=output.strip(),
    )
    return 0


def check_result_dict():
    """Return the last check result as a dict (without writing state)."""
    result = {
        "python_version": None,
        "packages": {},
        "gui_backends": {},
    }
    ret, output = _run_python(["-c", "import sys; print(sys.version)"])
    if ret == 0:
        result["python_version"] = output.strip().split("\n")[0]
    ret, output = _run_python_script([
        "import json, importlib.util as U",
        "pkgs = {0}".format(REQUIRED_IMPORTS),
        "r = {}",
        "for p in pkgs:",
        "    r[p] = U.find_spec(p) is not None",
        "print(json.dumps(r))",
    ])
    if ret == 0:
        try:
            result["packages"] = json.loads(output.strip().split("\n")[-1])
        except Exception:
            pass
    result["gui_backends"] = _probe_gui_backends()
    result["nnunet_version_compatible"], result["nnunet_version"] = (
        _nnunet_version_supported()
    )
    return result


# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------

def extract_archive(archive_path):
    """Extract a portable .zip archive into the project root.

    Uses Python's built-in zipfile — no external tools required.
    """
    import zipfile

    _write_state("extracting", step="extracting",
                 message="Extracting portable archive...")
    _log("=== Extract Archive ===")
    _log("Archive: {0}".format(archive_path))
    _log("Target:  {0}".format(str(PROJECT_ROOT)))

    if not os.path.isfile(archive_path):
        _write_state("error", message="Archive not found.", error=archive_path)
        _log("ERROR: archive not found")
        return 1

    archive_size_mb = os.path.getsize(archive_path) / (1024 * 1024)
    _log("Archive size: {:.0f} MB".format(archive_size_mb))

    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            names = zf.namelist()
            _log("Files in archive: {0}".format(len(names)))
            total = len(names)
            for i, name in enumerate(names):
                zf.extract(name, str(PROJECT_ROOT))
                if i % 200 == 0:
                    pct = int(100.0 * i / max(total, 1))
                    _write_state("extracting", step="extracting",
                                 message="Extracting... {0}% ({1}/{2})".format(pct, i, total))
    except zipfile.BadZipFile as exc:
        _write_state("error", message="Invalid or corrupt archive.", error=str(exc))
        _log("ERROR: bad zip file: {0}".format(exc))
        return 1
    except Exception as exc:
        _write_state("error", message="Extraction failed.", error=str(exc))
        _log("ERROR: {0}".format(exc))
        return 1

    _log("Extraction complete.")
    _write_state("extracted", step="verifying",
                 message="Archive extracted. Verifying environment...")

    # Auto-fix the venv if it's broken (different machine, different paths, etc.)
    if _venv_is_broken():
        _log("Python environment needs to be rebuilt for this machine.")
        _write_state("extracted", step="rebuilding",
                     message="This is a different machine. Rebuilding Python environment automatically...")
        ret = setup_from_scratch()
        if ret != 0:
            _log("Environment rebuild failed. Run Setup Environment > Setup From Scratch manually.")
            _write_state("incomplete",
                         message="Extraction OK, but environment rebuild failed. Run Setup From Scratch manually.",
                         all_ok=False)
            return 1
        _log("Environment rebuilt successfully.")
        _write_state("ok",
                     message="Archive extracted and environment rebuilt for this machine.",
                     all_ok=True)
    else:
        # Check packages and repair if needed
        _write_state("extracted", step="verifying_packages",
                     message="Verifying Python packages...")
        missing = _get_missing_packages()
        if not _probe_gui_backends().get("pyside6"):
            missing.extend(GUI_PACKAGES)
            missing = list(dict.fromkeys(missing))
        if missing:
            _log("Missing packages: {0}. Repairing...".format(", ".join(missing)))
            _write_state("extracted", step="repairing",
                         message="Installing {0} missing package(s)...".format(len(missing)))
            ok, error = _pip_install(missing)
            if not ok:
                _log("Package repair failed: {0}".format(error))
                _write_state("incomplete",
                             message="Extraction OK. {0} package(s) still missing.".format(len(missing)),
                             missing_packages=missing)
                return 1

        _write_state("ok",
                     message="Archive extracted and environment is ready.",
                     all_ok=True)

    return 0


def _venv_is_broken():
    """Return True if the nninteractive_env Python is not usable."""
    python = _find_python()
    if not python:
        return True
    ret, output = _run_python(["-c", "import sys; print(sys.executable)"], timeout=10)
    if ret != 0:
        return True
    # Also check that the Python can actually import a core package
    ret, _ = _run_python(["-c", "import importlib"], timeout=10)
    return ret != 0


def _get_missing_packages():
    """Return list of import names that are missing."""
    ret, output = _run_python_script([
        "import json, importlib.util as U",
        "pkgs = {0}".format(REQUIRED_IMPORTS),
        "r = []",
        "for p in pkgs:",
        "    if U.find_spec(p) is None:",
        "        r.append(p)",
        "print(json.dumps(r))",
    ])
    if ret == 0:
        try:
            missing_imports = json.loads(output.strip().split("\n")[-1])
            # Map import names back to pip names
            result = []
            for imp_name in missing_imports:
                pip_name = imp_name
                for p, i in _PIP_TO_IMPORT.items():
                    if i == imp_name:
                        pip_name = p
                        break
                result.append(pip_name)
            result = [
                "nnunetv2>=2.8.1,<2.9" if value == "nnunetv2" else value
                for value in result
            ]
            nnunet_version_ok, _nnunet_version = _nnunet_version_supported()
            if not nnunet_version_ok and not any(
                value.startswith("nnunetv2") for value in result
            ):
                result.append("nnunetv2>=2.8.1,<2.9")
            return result
        except Exception:
            pass
    return []


# ---------------------------------------------------------------------------
# Offline install (from offline bundle)
# ---------------------------------------------------------------------------

def _is_offline_bundle():
    """Return True if this looks like an offline bundle directory."""
    return (PROJECT_ROOT / "wheels").is_dir() and (PROJECT_ROOT / "python").is_dir()


def offline_install():
    """Install everything from the offline bundle (no internet needed).

    This does the same thing as setup_offline.bat, but can be triggered
    from within Mimics via Setup_Environment > Offline Install.

    Steps:
    1. Copy python/ embeddable into nninteractive_env/ (if not already done)
    2. Configure python313._pth
    3. Install pip (ensurepip → get-pip.py fallback)
    4. Install all wheels from wheels/ directory (--no-index)
    5. Verify key imports
    """
    import shutil

    _log("=== Offline Install ===")
    _write_state("setting_up", step="starting",
                 message="Starting offline installation...")

    env_dir = PROJECT_ROOT / env_dir_name()
    python_src = PROJECT_ROOT / "python"
    wheels_dir = PROJECT_ROOT / "wheels"
    get_pip = PROJECT_ROOT / "get-pip.py"

    # 1. Check offline bundle structure
    if not python_src.is_dir() or not wheels_dir.is_dir():
        _write_state("error",
                     message="Offline bundle not found. Need python/ and wheels/ directories.",
                     error="Missing offline bundle components")
        _log("ERROR: python/ or wheels/ not found — not an offline bundle?")
        return 1

    # 2. Copy Python embeddable into nninteractive_env/ (if not already done)
    env_python = env_dir / "python.exe"
    if not env_python.is_file():
        _write_state("setting_up", step="copying_python",
                     message="Copying Python embeddable...")
        _log("Copying python/ → nninteractive_env/")
        if env_dir.exists():
            shutil.rmtree(str(env_dir), ignore_errors=True)
        shutil.copytree(str(python_src), str(env_dir))
        _log("Python embeddable copied.")
    else:
        _log("nninteractive_env/python.exe already exists — skipping copy.")

    # 3. Configure python313._pth (enable site-packages + import site)
    _write_state("setting_up", step="configuring_python",
                 message="Configuring Python path...")
    pth_file = env_dir / "python313._pth"
    if not pth_file.exists():
        # Find the actual _pth file (version may differ)
        pth_files = list(env_dir.glob("python*._pth"))
        if pth_files:
            pth_file = pth_files[0]
    pth_content = (
        "python313.zip\n"
        ".\n"
        "Lib\n"
        "Lib\\site-packages\n"
        "import site\n"
    )
    pth_file.write_text(pth_content, encoding="ascii")
    _log("Configured {0}".format(pth_file.name))

    # 4. Install pip
    _write_state("setting_up", step="installing_pip",
                 message="Installing pip...")
    pip_exe = env_dir / "Scripts" / "pip.exe"
    if not pip_exe.is_file():
        _log("Trying ensurepip...")
        ret, output = _run_python(["-m", "ensurepip", "--upgrade"], timeout=120)
        if ret != 0 or not pip_exe.is_file():
            _log("ensurepip failed. Trying get-pip.py...")
            if get_pip.is_file():
                ret, output = _run_python([str(get_pip), "--no-wheel", "--no-index"], timeout=120)
                if ret != 0 and not pip_exe.is_file():
                    # Last resort: try with network (get-pip.py may need it)
                    _log("get-pip.py offline failed. Trying with network...")
                    ret, output = _run_python([str(get_pip)], timeout=120)
                if not pip_exe.is_file():
                    _write_state("error",
                                 message="pip installation failed.",
                                 error=output[-500:] if output else "unknown")
                    _log("ERROR: pip install failed: {0}".format(output[-200:] if output else ""))
                    return 1
            else:
                _write_state("error",
                             message="get-pip.py not found and ensurepip failed.",
                             error="No pip installation method available")
                _log("ERROR: no pip installation method available")
                return 1
    _log("pip is ready.")

    # 5. Install all wheels offline
    _write_state("setting_up", step="installing_packages",
                 message="Installing packages from local wheels (no internet)...")
    wheel_files = sorted(wheels_dir.glob("*.whl"))
    total = len(wheel_files)
    _log("Found {0} wheel files".format(total))

    fail_count = 0
    failed_names = []
    for i, whl in enumerate(wheel_files):
        if i % 10 == 0:
            pct = int(100.0 * i / max(total, 1))
            _write_state("setting_up", step="installing_packages",
                         message="Installing packages... {0}% ({1}/{2})".format(pct, i, total))
        ret, output = _run_python(
            ["-m", "pip", "install", str(whl), "--no-deps", "--no-index", "--quiet"],
            timeout=300,
        )
        if ret != 0:
            fail_count += 1
            failed_names.append(whl.name)
            _log("WARNING: Failed to install {0}".format(whl.name))

    _log("Installation complete. {0}/{1} packages installed, {2} failed.".format(
        total - fail_count, total, fail_count))

    if fail_count > 0:
        _log("Failed packages: {0}".format(", ".join(failed_names[:20])))

    _write_state("setting_up", step="installing_gui",
                 message="Ensuring PySide6 advanced UI wheels are installed consistently...")
    ret, output = _run_python(
        [
            "-m", "pip", "install",
            "PySide6", "shiboken6",
            "--no-index", "--find-links", str(wheels_dir),
            "--upgrade", "--quiet",
        ],
        timeout=900,
    )
    if ret != 0:
        _write_state("error",
                     message="PySide6 advanced UI installation failed.",
                     error=output[-1000:] if output else "unknown")
        _log("ERROR: PySide6 install failed: {0}".format(output[-300:] if output else ""))
        return 1
    _log("PySide6 advanced UI wheels installed.")

    # 6. Verify key imports and external GUI
    _write_state("setting_up", step="verifying",
                 message="Verifying installation...")
    ret, output = _run_python_script([
        "import json",
        "pkgs = {0}".format(REQUIRED_IMPORTS + GUI_IMPORTS),
        "r = {}",
        "for p in pkgs:",
        "    try:",
        "        __import__(p)",
        "        r[p] = True",
        "    except Exception:",
        "        r[p] = False",
        "print(json.dumps(r))",
    ], timeout=120)

    all_ok = False
    if ret == 0:
        try:
            pkg_status = json.loads(output.strip().split("\n")[-1])
            missing = [p for p, ok in pkg_status.items() if not ok]
            if not missing:
                all_ok = True
                _log("All key packages verified.")
            else:
                _log("Missing after install: {0}".format(", ".join(missing)))
        except Exception:
            _log("Could not parse verification output.")
    else:
        _log("Verification script failed: {0}".format(output[-200:] if output else ""))

    # Check CUDA
    cuda_ok = False
    ret, output = _run_python([
        "-c",
        "import torch; print(torch.cuda.is_available())",
    ], timeout=60)
    if ret == 0:
        cuda_ok = "True" in output
        _log("CUDA available: {0}".format(cuda_ok))

    if all_ok:
        _write_state("ok",
                     message="Offline installation complete. {0} packages installed, CUDA: {1}.".format(
                         total - fail_count, "available" if cuda_ok else "not available"),
                     all_ok=True,
                     cuda_available=cuda_ok,
                     failed_packages=failed_names)
        _log("=== Offline install complete: OK ===")
        return 0
    else:
        _write_state("incomplete",
                     message="Installation finished but some packages failed. {0} failed.".format(fail_count),
                     all_ok=False,
                     failed_packages=failed_names)
        _log("=== Offline install: INCOMPLETE ===")
        return 1


# ---------------------------------------------------------------------------
# Setup from scratch
# ---------------------------------------------------------------------------

def setup_from_scratch():
    """Create a fresh nninteractive_env from scratch.

    Uses the bundled Python embeddable if available, otherwise searches for
    a system Python 3.10+. Supports fully offline installation.

    If an offline bundle is detected (python/ + wheels/ directories present),
    delegates to offline_install() which is faster and needs no internet.
    """
    import shutil

    _log("=== Full Environment Setup ===")
    _write_state("setting_up", step="starting",
                 message="Setting up environment from scratch...")

    env_dir = PROJECT_ROOT / env_dir_name()

    # 0. If offline bundle is present, use it (much simpler, no internet needed)
    if _is_offline_bundle():
        _log("Offline bundle detected (python/ + wheels/). Using offline install.")
        return offline_install()

    # 1. Check for bundled Python (from offline bundle)
    bundled_python = _find_bundled_python()
    if bundled_python:
        _log("Using bundled Python: {0}".format(bundled_python))
        system_python = bundled_python
    else:
        # 2. Find a usable system Python (3.10+ preferred)
        system_python = None
        for cmd in ("python3", "python"):
            found = shutil.which(cmd)
            if found:
                try:
                    result = subprocess.run(
                        [found, "-c", "import sys; print(sys.version_info[:2])"],
                        capture_output=True, text=True, timeout=30,
                    )
                    version_str = result.stdout.strip()
                    if version_str:
                        major, minor = version_str.strip("()").split(",")
                        if int(major) >= 3 and int(minor) >= 10:
                            system_python = found
                            _log("Found Python {0}.{1} at {2}".format(major, minor, found))
                            break
                        else:
                            _log("Python at {0} is {1}.{2} — need 3.10+".format(found, major, minor))
                except Exception:
                    pass

    if not system_python:
        # Give a clear diagnostic
        found_versions = []
        for cmd in ("python3", "python"):
            found = shutil.which(cmd)
            if found:
                try:
                    result = subprocess.run(
                        [found, "-c", "import sys; print('.'.join(map(str, sys.version_info[:2])))"],
                        capture_output=True, text=True, timeout=10,
                    )
                    found_versions.append((cmd, found, result.stdout.strip()))
                except Exception:
                    pass

        if found_versions:
            lines = ["Found Python(s) but none meet the 3.10+ requirement:"]
            for cmd, path, ver in found_versions:
                lines.append("  {0} at {1} (version {2})".format(cmd, path, ver))
            lines.append("")
            lines.append("Solutions:")
            lines.append("  1. Use the offline bundle which includes Python 3.10")
            lines.append("     (ask whoever set up this workstation to prepare it:")
            lines.append("     'offline-bundle' in the packaging tool)")
            lines.append("  2. Or install Python 3.10+ from https://python.org")
            msg = "\n".join(lines)
        else:
            msg = (
                "No Python interpreter found on this system.\n\n"
                "Solutions:\n"
                "  1. Use the offline bundle which includes Python 3.10\n"
                "     (ask whoever set up this workstation to prepare it:\n"
                "     'offline-bundle' in the packaging tool)\n"
                "  2. Or install Python 3.10+ from https://python.org"
            )
        _write_state("error", message=msg, error="No usable Python found")
        _log("ERROR: no usable Python found")
        return 1

    # Create venv
    _write_state("setting_up", step="creating_venv",
                 message="Creating virtual environment (this may take a minute)...")

    ret, output = _run_shell(
        '"{0}" -m venv --clear "{1}"'.format(system_python, str(env_dir)),
        timeout=180,
    )
    if ret != 0:
        # Try without --clear for older Python versions
        ret, output = _run_shell(
            '"{0}" -m venv "{1}"'.format(system_python, str(env_dir)),
            timeout=180,
        )
    if ret != 0:
        _write_state("error",
                     message="Failed to create virtual environment.",
                     error=output[-500:])
        _log("Venv creation failed: {0}".format(output[-500:]))
        return 1

    _log("Virtual environment created.")

    # Install pip + packages
    _write_state("setting_up", step="installing_packages",
                 message="Installing required packages (this will take several minutes)...")

    ok, error = _pip_install(REQUIRED_PACKAGES + GUI_PACKAGES)
    if not ok:
        _write_state("error", message="Package installation failed.", error=error)
        _log("Install failed: {0}".format(error))
        return 1

    _write_state("ok", message="Environment setup complete. Run Check to verify.",
                 all_ok=True)
    _log("=== Setup complete ===")
    return 0


def _run_shell(command, timeout=600):
    """Run a shell command."""
    _log("Shell: {0}".format(command[:200]))
    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        stdout, _ = proc.communicate(timeout=timeout)
        output = stdout.decode("utf-8", "replace") if stdout else ""
        return proc.returncode, output
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        return -1, "Command timed out"
    except Exception as exc:
        return -1, "Command failed: {0}".format(str(exc))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/setup_env.py <check|install|install-remote|extract|setup-from-scratch|offline-install> [archive]")
        return 1

    cmd = sys.argv[1].lower()
    if cmd == "check":
        return check()
    elif cmd == "install":
        return install()
    elif cmd == "install-remote":
        return install_remote()
    elif cmd == "extract":
        archive = sys.argv[2] if len(sys.argv) > 2 else None
        if not archive:
            print("Usage: python tools/setup_env.py extract <archive.zip>")
            return 1
        return extract_archive(archive)
    elif cmd == "setup-from-scratch":
        return setup_from_scratch()
    elif cmd == "offline-install":
        return offline_install()
    else:
        print("Unknown command: {0}".format(cmd))
        return 1


if __name__ == "__main__":
    sys.exit(main())
