#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""External environment setup worker for Mimics-Script.

Runs outside Mimics (Python 3.10+) to check and repair the nninteractive_env.

Usage:
    python tools/setup_env.py check                # Validate env, write JSON report
    python tools/setup_env.py install              # Check + pip install missing pkgs
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
    "monai",
    "nnInteractive",
    "torchvision",
    "transformers",
    "yaml",         # pip package is pyyaml
    "tqdm",
    "tensorboard",
]

# Pip package names (may differ from import names)
REQUIRED_PACKAGES = [
    "torch",
    "numpy",
    "nibabel",
    "pydicom",
    "SimpleITK",
    "scipy",
    "monai",
    "nnInteractive",
    "torchvision",
    "transformers",
    "pyyaml",       # import name is yaml
    "tqdm",
    "tensorboard",
]

# Mapping for __import__: pip name → import name
_PIP_TO_IMPORT = {
    "pyyaml": "yaml",
}

# Extra index URL for PyTorch CUDA builds
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu121"

# PyPI mirrors (tried in order; useful for Chinese mainland users)
PYPI_MIRRORS = [
    "",  # default PyPI
    "https://pypi.tuna.tsinghua.edu.cn/simple",
    "https://mirrors.aliyun.com/pypi/simple",
]


def _find_python():
    """Find the nninteractive_env Python."""
    for rel in (
        "nninteractive_env/Scripts/python.exe",
        "nninteractive_env/python/python.exe",
        "nninteractive_env/bin/python3",
        "nninteractive_env/bin/python",
    ):
        p = PROJECT_ROOT / rel
        if p.is_file():
            return str(p)
    # Check for bundled Python (from offline bundle)
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


def _log(message):
    """Append a line to the setup log."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = "[{0}] {1}".format(timestamp, message)
    print(line)
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
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

    # 3. Package versions
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

    # 4. CUDA
    ret, output = _run_python([
        "-c",
        "import torch;"
        "print('CUDA=' + str(torch.cuda.is_available()));"
        "print('Devices=' + str(torch.cuda.device_count()));"
        "print('Version=' + str(getattr(torch.version, 'cuda', 'unknown')))",
    ])
    if ret == 0:
        for line in output.strip().split("\n"):
            if line.startswith("CUDA="):
                result["cuda_available"] = line.split("=")[1] == "True"
            elif line.startswith("Devices="):
                result["cuda_device_count"] = int(line.split("=")[1])
            elif line.startswith("Version="):
                result["cuda_version"] = line.split("=")[1]
        _log("CUDA: available={0}, devices={1}".format(
            result["cuda_available"], result["cuda_device_count"]))

    # 5. Model weights
    model_roots = [
        PROJECT_ROOT / "nninteractive_env" / "models",
        PROJECT_ROOT / "external" / "dinov3-medical-seg" / "models",
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
    all_ok = (
        result["python_version"] is not None
        and len(missing_pkgs) == 0
    )

    _write_state(
        "ok" if all_ok else "incomplete",
        message="All checks passed." if all_ok else "{0} package(s) missing.".format(len(missing_pkgs)),
        detail=result,
        missing_packages=missing_pkgs,
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

    if not missing:
        _write_state("ok", message="All packages are already installed.", all_ok=True)
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
        "pkgs = {0}".format(REQUIRED_IMPORTS),
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

    if still_missing:
        _write_state("incomplete",
                     message="{0} package(s) still missing.".format(len(still_missing)),
                     missing_packages=still_missing)
        _log("Still missing: {0}".format(", ".join(still_missing)))
        return 1

    _write_state("ok", message="All packages installed successfully.", all_ok=True)
    _log("=== Install complete: OK ===")
    return 0


def check_result_dict():
    """Return the last check result as a dict (without writing state)."""
    result = {
        "python_version": None,
        "packages": {},
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
            return result
        except Exception:
            pass
    return []


# ---------------------------------------------------------------------------
# Setup from scratch
# ---------------------------------------------------------------------------

def setup_from_scratch():
    """Create a fresh nninteractive_env from scratch.

    Uses the bundled Python embeddable if available, otherwise searches for
    a system Python 3.10+. Supports fully offline installation.
    """
    import shutil

    _log("=== Full Environment Setup ===")
    _write_state("setting_up", step="starting",
                 message="Setting up environment from scratch...")

    env_dir = PROJECT_ROOT / "nninteractive_env"

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
            lines.append("     Run: python tools/package_portable.py offline-bundle")
            lines.append("  2. Or install Python 3.10+ from https://python.org")
            msg = "\n".join(lines)
        else:
            msg = (
                "No Python interpreter found on this system.\n\n"
                "Solutions:\n"
                "  1. Use the offline bundle which includes Python 3.10\n"
                "     Run: python tools/package_portable.py offline-bundle\n"
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

    ok, error = _pip_install(REQUIRED_PACKAGES)
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
        print("Usage: python tools/setup_env.py <check|install|extract|setup-from-scratch> [archive]")
        return 1

    cmd = sys.argv[1].lower()
    if cmd == "check":
        return check()
    elif cmd == "install":
        return install()
    elif cmd == "extract":
        archive = sys.argv[2] if len(sys.argv) > 2 else None
        if not archive:
            print("Usage: python tools/setup_env.py extract <archive.zip>")
            return 1
        return extract_archive(archive)
    elif cmd == "setup-from-scratch":
        return setup_from_scratch()
    else:
        print("Unknown command: {0}".format(cmd))
        return 1


if __name__ == "__main__":
    sys.exit(main())
