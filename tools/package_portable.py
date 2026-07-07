#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Package Mimics-Script for offline deployment to another Windows machine.

Usage:
    python tools/package_portable.py check       # Validate everything is ready
    python tools/package_portable.py pack [out]  # Create the portable archive

A lightweight tool that validates the project is ready to deploy and
creates a portable archive. No complex manifest or multi-command workflow.
"""

from __future__ import print_function

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INCLUDE_DIRS = [
    "scripting_library",
    "runtime_py35",
    "external",
    "tools",
    # Model weights are portable (neural network parameters, not compiled code).
    # The Python environment itself (Lib/, Scripts/, etc.) is NOT included
    # because venvs are not portable across machines. Rebuild on the target
    # with: Setup Environment > Setup From Scratch.
    "nninteractive_env/models",
]
INCLUDE_FILES = [
    "mimics_bridge.py",
    "nninteractive_bridge.py",
    "resource_locks.py",
    "nninteractive_config.json",
    "fewshot_config.json",
    "window_level_presets.json",
    ".gitignore",
]

ARCHIVE_NAME = "mimics_script_portable"

# Use --with-env to include the full nninteractive_env/ (only when both
# machines are identical: same OS, architecture, CUDA, and Python version).
FULL_ENV_DIR = "nninteractive_env"


def _green(t): return "\033[32m{}\033[0m".format(t)
def _red(t):   return "\033[31m{}\033[0m".format(t)
def _yellow(t): return "\033[33m{}\033[0m".format(t)


def _find_python():
    """Find the nninteractive_env Python executable."""
    for rel in ("Scripts/python.exe", "python/python.exe", "bin/python3", "bin/python"):
        p = PROJECT_ROOT / "nninteractive_env" / rel
        if p.is_file():
            return str(p)
    return None


def check():
    """Validate the project is ready to package and deploy."""
    ok = True

    print("Mimics-Script Environment Check")
    print("  Project: {}".format(PROJECT_ROOT))
    print()

    # 1. Project structure
    for name in INCLUDE_DIRS:
        if (PROJECT_ROOT / name).is_dir():
            print("  {} {}".format(_green("[OK]"), name))
        else:
            print("  {} {}  -- MISSING".format(_red("[!!]"), name))
            ok = False
    for name in INCLUDE_FILES:
        if (PROJECT_ROOT / name).is_file():
            print("  {} {}".format(_green("[OK]"), name))
        else:
            print("  {} {}  -- MISSING".format(_red("[!!]"), name))
            ok = False

    # 2. Python environment
    python_exe = _find_python()
    if not python_exe:
        print("\n  {} Python not found in nninteractive_env/".format(_red("[!!]")))
        print("    Run: python tools/package_portable.py setup-env")
        return False
    print("\n  {} Python: {}".format(_green("[OK]"), python_exe))

    try:
        result = subprocess.run(
            [python_exe, "-c", "import sys; print(sys.version.split()[0])"],
            capture_output=True, text=True, timeout=30,
        )
        print("  {} Version: {}".format(_green("[OK]"), result.stdout.strip()))
    except Exception as exc:
        print("  {} Cannot run: {}".format(_red("[!!]"), exc))
        ok = False

    # 3. Required packages
    try:
        result = subprocess.run(
            [python_exe, "-c",
             "import json;"
             "mods=['torch','numpy','nibabel','pydicom','SimpleITK','scipy'];"
             "import importlib.util as U;"
             "print(json.dumps({m:U.find_spec(m)is not None for m in mods}))"],
            capture_output=True, text=True, timeout=60,
        )
        pkg_status = json.loads(result.stdout.strip())
        for pkg, found in sorted(pkg_status.items()):
            if found:
                print("  {} {}".format(_green("[OK]"), pkg))
            else:
                print("  {} {}  -- MISSING".format(_red("[!!]"), pkg))
                ok = False
    except Exception as exc:
        print("  {} Package probe failed: {}".format(_red("[!!]"), exc))
        ok = False

    # 4. CUDA
    try:
        result = subprocess.run(
            [python_exe, "-c",
             "import torch; print('CUDA='+str(torch.cuda.is_available())+' devices='+str(torch.cuda.device_count()))"],
            capture_output=True, text=True, timeout=60,
        )
        print("  {} {}".format(_green("[OK]"), result.stdout.strip()))
    except Exception:
        print("  {} CUDA check skipped".format(_yellow("[--]")))

    # 5. Model weights
    model_dirs = [
        "nninteractive_env/models/nnInteractive_v1.0",
        "external/dinov3-medical-seg/models/dinov3-vitb16",
        "external/dinov3-medical-seg/models/dinov3-vitl16",
    ]
    for rel in model_dirs:
        p = PROJECT_ROOT / rel
        if p.is_dir():
            ckpts = list(p.rglob("*.pth")) + list(p.rglob("*.safetensors"))
            print("  {} {} ({} files)".format(_green("[OK]"), rel, len(ckpts)))
        else:
            print("  {} {}  -- not found".format(_yellow("[--]"), rel))

    print("\n  {}".format(_green("All checks passed.") if ok else _red("Some checks failed.")))
    return 0 if ok else 1


def pack(output_dir=None, with_env=False):
    """Create a portable .zip archive for offline deployment.

    By default, only source code + configs + model weights are packaged.
    Python venvs are NOT portable across machines — the target machine
    must rebuild the environment via Setup Environment > Setup From Scratch.

    Use --with-env ONLY when both machines are identical (same OS, same
    CPU architecture, same CUDA version, same Python version).

    Uses Python's built-in zipfile — no external tools needed.
    """
    import zipfile

    if output_dir is None:
        output_dir = str(PROJECT_ROOT.parent)

    # Validate: models must exist. Full env is optional.
    models_dir = PROJECT_ROOT / "nninteractive_env" / "models"
    if not models_dir.is_dir():
        print(_red("Models directory not found: {}".format(models_dir)))
        print("Model weights are required. Download them before packaging.")
        return 1

    archive = os.path.join(output_dir, ARCHIVE_NAME + ".zip")
    print("Creating: {}".format(archive))

    if with_env:
        if not _find_python():
            print(_red("Python not found in nninteractive_env/."))
            return 1
        print(_yellow("Including full nninteractive_env/ — only works on identical machines!"))
    else:
        print("Packaging source + configs + models (no venv).")
        print("Target machine: run Setup Environment > Setup From Scratch to rebuild.")

    # Paths and patterns to exclude from packaging (auto-generated, OS junk, dev-only)
    EXCLUDE_PARTS = [
        "__pycache__",
        ".DS_Store",
        ".git",
        ".pytest_cache",
        "pyqt5_wheels",
    ]
    EXCLUDE_SUFFIXES = (".pyc", ".pyo")
    # DINOv3 external: ship src + scripts + config + models, skip docs + tests + caches
    EXCLUDE_DINOV3_DIRS = {"docs", "tests", ".cache", "__pycache__", ".pytest_cache"}

    def _should_include(fpath, arcname):
        parts = arcname.replace("\\", "/").split("/")
        for part in parts:
            if part in EXCLUDE_PARTS:
                return False
        if arcname.endswith(EXCLUDE_SUFFIXES):
            return False
        # DINOv3 external: only ship what's needed at runtime
        if arcname.startswith("external/dinov3-medical-seg/"):
            dinov3_rel = arcname[len("external/dinov3-medical-seg/"):]
            top = dinov3_rel.split("/")[0] if "/" in dinov3_rel else dinov3_rel
            if top in EXCLUDE_DINOV3_DIRS:
                return False
        return True

    # Build the list of directories to include
    dirs_to_pack = list(INCLUDE_DIRS)
    if with_env:
        dirs_to_pack.append(FULL_ENV_DIR)
        # Exclude the Python interpreter's own cache and compiled bytecode
        EXCLUDE_PARTS = EXCLUDE_PARTS + ["Include", "Lib/site-packages/pip", "share"]

    try:
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
            seen = set()
            for name in dirs_to_pack:
                p = PROJECT_ROOT / name
                if not p.exists():
                    continue
                for fpath in p.rglob("*"):
                    if fpath.is_file():
                        arcname = str(fpath.relative_to(PROJECT_ROOT))
                        if arcname in seen:
                            continue
                        if not _should_include(fpath, arcname):
                            continue
                        # When packaging with env, skip venv symlinks + caches
                        if with_env and arcname.startswith("nninteractive_env/"):
                            parts = arcname.split("/")
                            if any(p in ("__pycache__", "Include", "share") for p in parts):
                                continue
                        seen.add(arcname)
                        zf.write(str(fpath), arcname)
            for name in INCLUDE_FILES:
                p = PROJECT_ROOT / name
                if p.is_file():
                    zf.write(str(p), name)

        size_mb = os.path.getsize(archive) / (1024 * 1024)
        print(_green("Done: {} ({:.0f} MB)".format(archive, size_mb)))
        print()
        print("Deployment on target machine:")
        print("  1. Copy {} to the target".format(os.path.basename(archive)))
        print("  2. In Mimics: 99_Admin > Setup Environment > Extract Archive")
        if not with_env:
            print("  3. Then: Setup Environment > Setup From Scratch")
            print("     (rebuilds the Python environment from scratch)")
        return 0
    except Exception as exc:
        print(_red("Failed: {}".format(exc)))
        return 1


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/package_portable.py <command>")
        print("  check              Validate project structure and models")
        print("  pack [out_dir]      Create portable .zip (source + configs + models only)")
        print("  pack --with-env     Create .zip including full nninteractive_env/")
        print("                      (only use when source and target machines are identical)")
        return 1

    cmd = sys.argv[1].lower()
    args = sys.argv[2:]
    with_env = "--with-env" in args
    out_args = [a for a in args if a != "--with-env"]

    if cmd == "check":
        return check()
    elif cmd == "pack":
        out = out_args[0] if out_args else None
        return pack(out, with_env=with_env)
    elif cmd == "offline-bundle":
        return offline_bundle()
    else:
        print("Unknown command: {}".format(cmd))
        return 1


# ---------------------------------------------------------------------------
# Offline bundle: Python + wheels + source for fully offline deployment
# ---------------------------------------------------------------------------

PYTHON_VERSION = "3.10.11"
PYTHON_EMBED_URL = (
    "https://www.python.org/ftp/python/{0}/python-{0}-embed-amd64.zip"
).format(PYTHON_VERSION)
CUDA_INDEX = "https://download.pytorch.org/whl/cu121"

# PyPI mirrors for Chinese mainland users
PYPI_MIRRORS = [
    "",  # default PyPI
    "https://pypi.tuna.tsinghua.edu.cn/simple",
    "https://mirrors.aliyun.com/pypi/simple",
]

OFFLINE_WHEEL_PACKAGES = [
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
    "pyyaml",
    "tqdm",
    "tensorboard",
]


def offline_bundle():
    """Download everything needed for a fully offline Windows deployment.

    Creates a self-contained directory with:
      - Python 3.10 embeddable (no install, no admin rights)
      - All .whl files for Windows x86_64 + CUDA 12.1
      - Source code, configs, and model weights
      - setup_offline.bat (double-click to install)

    Copy the resulting directory to a USB drive and run
    setup_offline.bat on the target Windows machine.
    """
    import urllib.request
    import zipfile
    import tempfile

    bundle_dir = PROJECT_ROOT.parent / "mimics_script_offline"
    if bundle_dir.exists():
        print(_yellow("Bundle directory already exists. Overwriting..."))

    wheels_dir = bundle_dir / "wheels"
    python_dir = bundle_dir / "python"
    os.makedirs(str(wheels_dir), exist_ok=True)
    os.makedirs(str(python_dir), exist_ok=True)

    # 1. Download Python embeddable
    print(_header("1. Python embeddable"))
    python_zip = bundle_dir / "python-embed-amd64.zip"
    if python_zip.exists():
        print("  {} Already downloaded".format(_green("[OK]")))
    else:
        print("  Downloading Python {0} embeddable...".format(PYTHON_VERSION))
        print("  URL: {0}".format(PYTHON_EMBED_URL))
        try:
            urllib.request.urlretrieve(PYTHON_EMBED_URL, str(python_zip))
            print("  {} Downloaded".format(_green("[OK]")))
        except Exception as exc:
            print("  {} Download failed: {0}".format(_red("[!!]"), exc))
            print("    Manual: download from {0}".format(PYTHON_EMBED_URL))
            print("    Save as: {0}".format(python_zip))

    if python_zip.exists():
        with zipfile.ZipFile(str(python_zip), "r") as zf:
            zf.extractall(str(python_dir))
        print("  {} Python embeddable extracted to python/".format(_green("[OK]")))

    # 2. Download all wheels AND their transitive dependencies in one pass.
    #    pip download with all packages listed together resolves the FULL
    #    transitive closure — nothing is missed. The resulting wheels/
    #    directory is 100% self-contained. Target machine NEVER needs network.
    print(_header("2. Wheels (win_amd64, Python 3.10, CUDA 12.1)"))
    print("  Downloading ALL packages + transitive dependencies...")
    print("  This is a single pip download call that resolves everything.")
    print("  Size: ~2.4 GB. Time: 15-30 minutes.")
    print()

    downloaded = False
    for mirror in PYPI_MIRRORS:
        label = mirror or "default PyPI"
        print("  Trying {0}...".format(label))
        cmd = [
            sys.executable, "-m", "pip", "download",
            "--dest", str(wheels_dir),
            "--platform", "win_amd64",
            "--python-version", "310",
            "--only-binary", ":all:",
            "--extra-index-url", CUDA_INDEX,
        ]
        if mirror:
            cmd += ["-i", mirror]
        cmd += OFFLINE_WHEEL_PACKAGES

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if result.returncode == 0:
                total_wheels = len(list(wheels_dir.glob("*.whl")))
                total_size = sum(f.stat().st_size for f in wheels_dir.glob("*.whl"))
                print("  {} Downloaded {} wheels ({:.0f} MB) via {}".format(
                    _green("[OK]"), total_wheels, total_size / 1e6, label))
                downloaded = True
                break
            else:
                err = result.stderr[-150:] if result.stderr else "unknown"
                print("  {} {}: {}".format(_yellow("[--]"), label, err.strip()))
        except subprocess.TimeoutExpired:
            print("  {} {}: timed out".format(_yellow("[--]"), label))

    if not downloaded:
        print("  {} All mirrors failed.".format(_red("[!!]")))
        print("    Manual: install packages on any machine, then copy")
        print("    site-packages to wheels/ or use a different network.")
        return 1

    # Verify key packages
    print()
    print("  Verifying key packages in wheels/:")
    for pkg in ["torch", "numpy", "nibabel"]:
        found = list(wheels_dir.glob(pkg.replace("-", "_") + "*"))
        if found:
            print("    {} {}: {} file(s)".format(_green("[OK]"), pkg, len(found)))
        else:
            print("    {} {}: MISSING".format(_red("[!!]"), pkg))

    # 3. Copy source files
    print(_header("3. Source code + configs + models"))
    for name in INCLUDE_DIRS:
        src = PROJECT_ROOT / name
        dst = bundle_dir / name
        if not src.exists():
            continue
        if dst.exists():
            shutil.rmtree(str(dst), ignore_errors=True)
        shutil.copytree(str(src), str(dst),
                         ignore=shutil.ignore_patterns("__pycache__", ".DS_Store", ".pyc", ".pyo",
                                                       ".git", "docs", "tests", ".cache"))
        print("  {} {}/".format(_green("[+]"), name))
    for name in INCLUDE_FILES:
        src = PROJECT_ROOT / name
        if src.is_file():
            shutil.copy2(str(src), str(bundle_dir / name))
            print("  {} {}".format(_green("[+]"), name))

    # 4. Generate setup_offline.bat
    print(_header("4. setup_offline.bat"))
    bat_path = bundle_dir / "setup_offline.bat"
    bat_content = _generate_offline_bat()
    with open(str(bat_path), "w") as f:
        f.write(bat_content)
    print("  {} setup_offline.bat written".format(_green("[OK]")))

    # 5. Summary
    total_size = 0
    for root, _dirs, files in os.walk(str(bundle_dir)):
        for f in files:
            try:
                total_size += os.path.getsize(os.path.join(root, f))
            except Exception:
                pass
    print(_header("Done"))
    print("  Bundle: {}".format(bundle_dir))
    print("  Size:   {:.0f} MB".format(total_size / 1e6))
    print()
    print("Deployment on target Windows machine:")
    print("  1. Copy mimics_script_offline/ to the target")
    print("  2. Double-click setup_offline.bat")
    print("  3. Wait for installation to complete")
    print("  4. In Mimics: Scripting → Add Scripting Library →")
    print("     select scripting_library/ folder")
    return 0


def _generate_offline_bat():
    """Generate the Windows batch script for offline setup."""
    lines = []
    lines.append("@echo off")
    lines.append("setlocal enabledelayedexpansion")
    lines.append("echo ========================================")
    lines.append("echo   Mimics-Script Offline Setup")
    lines.append("echo   Target: Windows + CUDA 12.1")
    lines.append("echo ========================================")
    lines.append("echo.")
    lines.append("")
    lines.append("cd /d %~dp0")
    lines.append("")
    lines.append(":: 1. Setup Python embeddable")
    lines.append('echo [1/4] Setting up Python %s...' % PYTHON_VERSION)
    lines.append('if not exist "nninteractive_env\\python.exe" (')
    lines.append('    echo   Extracting Python embeddable...')
    lines.append('    if not exist "python\\python.exe" (')
    lines.append('        echo   ERROR: python\\python.exe not found.')
    lines.append('        echo   Make sure the python/ directory is present.')
    lines.append("        pause")
    lines.append("        exit /b 1")
    lines.append("    )")
    lines.append('    mkdir nninteractive_env')
    lines.append('    xcopy /E /Q /Y python\\* nninteractive_env\\ >nul')
    lines.append('    echo   Copying python310._pth...')
    lines.append('    echo python310.zip > nninteractive_env\\python310._pth')
    lines.append('    echo . >> nninteractive_env\\python310._pth')
    lines.append('    echo Lib\\site-packages >> nninteractive_env\\python310._pth')
    lines.append('    echo import site >> nninteractive_env\\python310._pth')
    lines.append(")")
    lines.append("")
    lines.append(":: 2. Install pip")
    lines.append('echo [2/4] Installing pip...')
    lines.append('if not exist "nninteractive_env\\Scripts\\pip.exe" (')
    lines.append('    nninteractive_env\\python.exe -m ensurepip --upgrade')
    lines.append(")")
    lines.append("")
    lines.append(":: 3. Install all wheels offline")
    lines.append('echo [3/4] Installing packages from local wheels (no internet)...')
    lines.append('for %%f in (wheels\\*.whl) do (')
    lines.append('    echo   Installing %%~nxf...')
    lines.append('    nninteractive_env\\python.exe -m pip install "%%f" --no-deps --no-index --quiet')
    lines.append(")")
    lines.append("")
    lines.append(":: 4. Verify")
    lines.append('echo [4/4] Verifying installation...')
    lines.append('nninteractive_env\\python.exe -c "import torch; print(\"  torch\", torch.__version__); print(\"  CUDA available:\", torch.cuda.is_available())"')
    lines.append('nninteractive_env\\python.exe -c "import numpy, nibabel, pydicom, SimpleITK, scipy, monai, nnInteractive, torchvision, transformers, yaml, tqdm; print(\"  All packages OK\")"')
    lines.append('if %errorlevel% neq 0 (')
    lines.append("    echo   Some packages failed to import.")
    lines.append('    echo   Try running: nninteractive_env\\python.exe -m pip install wheels\\*.whl')
    lines.append("    pause")
    lines.append("    exit /b 1")
    lines.append(")")
    lines.append("")
    lines.append("echo.")
    lines.append("echo ========================================")
    lines.append("echo   Setup complete!")
    lines.append("echo   In Mimics: Scripting -^> Add Scripting Library")
    lines.append("echo   Select: scripting_library\\ folder")
    lines.append("echo ========================================")
    lines.append("pause")
    return "\r\n".join(lines)


def _header(text):
    return "\n" + "=" * 64 + "\n  " + text + "\n" + "=" * 64


if __name__ == "__main__":
    sys.exit(main())
