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
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INCLUDE_DIRS = [
    "scripting_library",
    "runtime_py35",
    "integrations",
    "tools",
    "remote",
    # Model weights are portable (neural network parameters, not compiled code).
    # The Python environment itself (Lib/, Scripts/, etc.) is NOT included
    # because venvs are not portable across machines. Rebuild on the target
    # with: Setup Environment > Setup From Scratch.
    "nninteractive_env/models",
    "python_env/models",
]
INCLUDE_FILES = [
    "mimics_bridge.py",
    "nninteractive_bridge.py",
    "resource_locks.py",
    "nninteractive_config.json",
    "nninteractive_finetune_config.json",
    "interactive_algorithms_config.json",
    "fewshot_config.json",
    "window_level_presets.json",
    ".dockerignore",
    ".gitignore",
]
REQUIRED_EXTERNAL_UI_FILES = [
    "tools/io_path_setup_ui.py",
    "tools/path_dialog_helper.py",
    "tools/single_case_import_worker.py",
    "tools/ui_theme.py",
    "tools/ui_preferences.py",
    "tools/training_data_ui.py",
    "tools/mask_file_picker_ui.py",
    "tools/fewshot_training_setup_ui.py",
    "tools/fewshot_status_viewer.py",
    "tools/fewshot_model_chooser.py",
    "tools/nninteractive_task_common.py",
    "tools/nninteractive_finetune_pipeline.py",
    "tools/nninteractive_task_model_center.py",
    "tools/nninteractive_task_model_chooser.py",
    "tools/ai_model_bundle.py",
    "tools/remote_compute.py",
    "tools/remote_compute_ui.py",
    "tools/remote_training_controller.py",
    "tools/remote_worker.py",
    "tools/nnunet_common.py",
    "tools/nnunet_jobs.py",
    "tools/nnunet_pipeline.py",
    "tools/nnunet_stage_worker.py",
    "tools/nnunet_training_setup_ui.py",
    "tools/nnunet_prediction_setup_ui.py",
    "tools/nnunet_status_viewer.py",
    "tools/interactive_algorithms_worker.py",
    "tools/verify_medical_geometry.py",
    "runtime_py35/interactive_algorithms_mimics.py",
    "runtime_py35/nnunet_mimics.py",
    "integrations/nnunet_segmentation_workflow/trainers/MimicsNNUNetTrainer.py",
    "integrations/nnunet_segmentation_workflow/trainers/MimicsNNUNetTrainerNoMirroring.py",
]
DEFAULT_FROZEN_ENCODER = (
    "integrations/dinov3-medical-seg/models/dinov3-vits16/model.onnx"
)

ARCHIVE_NAME = "mimics_script_portable"

# Environment directory names, in preference order. python_env is the new
# generic name; nninteractive_env is the legacy name kept so existing
# installs keep working. Resolution: first existing directory wins, and
# fresh installs/packages default to python_env.
ENV_DIR_CANDIDATES = ("python_env", "nninteractive_env")


def env_dir_name():
    """Return the environment directory name to use for this project root."""
    for name in ENV_DIR_CANDIDATES:
        if (PROJECT_ROOT / name).is_dir():
            return name
    return ENV_DIR_CANDIDATES[0]


# Use --with-env to include the full environment (only when both machines
# are identical: same OS, architecture, CUDA, and Python version).
FULL_ENV_DIR = "python_env"


def _green(t): return "\033[32m{}\033[0m".format(t)
def _red(t):   return "\033[31m{}\033[0m".format(t)
def _yellow(t): return "\033[33m{}\033[0m".format(t)


def _find_python():
    """Find the external Python executable (python_env, legacy nninteractive_env)."""
    for name in ENV_DIR_CANDIDATES:
        for rel in ("python.exe", "Scripts/python.exe", "python/python.exe", "bin/python3", "bin/python"):
            p = PROJECT_ROOT / name / rel
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
    for name in REQUIRED_EXTERNAL_UI_FILES:
        if not (PROJECT_ROOT / name).is_file():
            print("  {} {}  -- REQUIRED EXTERNAL UI FILE MISSING".format(_red("[!!]"), name))
            ok = False

    # 2. Python environment
    python_exe = _find_python()
    if not python_exe:
        print("\n  {} Python not found in python_env/ or nninteractive_env/".format(_red("[!!]")))
        print("    Offline bundle: run setup_offline.bat on the target Windows machine.")
        print("    Online repair: use 99_Admin/01_Setup_Repair_Environment.py in Mimics.")
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

    # 3. Required packages (must match setup_env.py REQUIRED_IMPORTS)
    required_imports = [
        "torch", "numpy", "nibabel", "pydicom", "SimpleITK", "scipy",
        "nnInteractive", "torchvision", "transformers",
        "yaml", "tqdm", "tensorboard", "tomli", "onnxruntime", "nnunetv2",
        "acvl_utils",
        "PySide6", "shiboken6",
    ]
    try:
        result = subprocess.run(
            [python_exe, "-c",
             "import json;"
             "mods={0};"
             "import importlib.util as U;"
             "print(json.dumps({{m:U.find_spec(m)is not None for m in mods}}))".format(repr(required_imports))],
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
    # The configured default training method cannot run without its ViT-S
    # encoder, so packaging must fail early instead of producing a broken
    # offline bundle.
    required_model_dirs = [
        "python_env/models/nnInteractive_v1.0",
        "nninteractive_env/models/nnInteractive_v1.0",
    ]
    optional_model_dirs = [
        "integrations/dinov3-medical-seg/models/dinov3-vitb16",
        "integrations/dinov3-medical-seg/models/dinov3-vitl16",
    ]
    for rel in required_model_dirs:
        p = PROJECT_ROOT / rel
        if p.is_dir():
            ckpts = list(p.rglob("*.pth")) + list(p.rglob("*.safetensors"))
            print("  {} {} ({} files)".format(_green("[OK]"), rel, len(ckpts)))
        else:
            print("  {} {}  -- MISSING".format(_red("[!!]"), rel))
            ok = False
    for rel in optional_model_dirs:
        p = PROJECT_ROOT / rel
        if p.is_dir():
            ckpts = list(p.rglob("*.pth")) + list(p.rglob("*.safetensors"))
            print("  {} {} ({} files)".format(_green("[OK]"), rel, len(ckpts)))
        else:
            print("  {} {}  -- optional, not found".format(_yellow("[--]"), rel))
    scribbleprompt_checkpoint = (
        PROJECT_ROOT
        / "integrations"
        / "ScribblePrompt"
        / "checkpoints"
        / "ScribblePrompt_unet_v1_nf192_res128.pt"
    )
    if scribbleprompt_checkpoint.is_file():
        print("  {} ScribblePrompt UNet checkpoint".format(_green("[OK]")))
    else:
        print(
            "  {} ScribblePrompt UNet checkpoint -- optional entry unavailable".format(
                _yellow("[--]")
            )
        )
    frozen_encoder = PROJECT_ROOT / DEFAULT_FROZEN_ENCODER
    if frozen_encoder.is_file() and frozen_encoder.stat().st_size > 1024 * 1024:
        print(
            "  {} {} ({:.1f} MB)".format(
                _green("[OK]"),
                DEFAULT_FROZEN_ENCODER,
                frozen_encoder.stat().st_size / (1024.0 * 1024.0),
            )
        )
    else:
        print(
            "  {} {}  -- REQUIRED DEFAULT ENCODER MISSING".format(
                _red("[!!]"),
                DEFAULT_FROZEN_ENCODER,
            )
        )
        ok = False

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
    models_dir = PROJECT_ROOT / env_dir_name() / "models"
    if not models_dir.is_dir():
        print(_red("Models directory not found: {}".format(models_dir)))
        print("Model weights are required. Download them before packaging.")
        return 1
    frozen_encoder = PROJECT_ROOT / DEFAULT_FROZEN_ENCODER
    if not frozen_encoder.is_file() or frozen_encoder.stat().st_size <= 1024 * 1024:
        print(_red("Default frozen encoder was not found: {}".format(frozen_encoder)))
        print(
            "Place the verified ViT-S/16 model.onnx at the configured path "
            "before packaging."
        )
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
    # The fine-tuning package contains large experiment workspaces and
    # validation datasets. Runtime deployment only needs its source/config.
    EXCLUDE_NNINTERACTIVE_FINETUNE_DIRS = {
        "data",
        "docs",
        "tests",
        "validation",
        "work",
        ".cache",
        ".pytest_cache",
    }

    def _should_include(fpath, arcname):
        parts = arcname.replace("\\", "/").split("/")
        for part in parts:
            if part in EXCLUDE_PARTS:
                return False
        if arcname.endswith(EXCLUDE_SUFFIXES):
            return False
        # DINOv3 external: only ship what's needed at runtime
        if arcname.startswith("integrations/dinov3-medical-seg/"):
            dinov3_rel = arcname[len("integrations/dinov3-medical-seg/"):]
            top = dinov3_rel.split("/")[0] if "/" in dinov3_rel else dinov3_rel
            if top in EXCLUDE_DINOV3_DIRS:
                return False
        if arcname.startswith("integrations/nninteractive-finetune/"):
            package_rel = arcname[len("integrations/nninteractive-finetune/"):]
            top = package_rel.split("/")[0] if "/" in package_rel else package_rel
            if top in EXCLUDE_NNINTERACTIVE_FINETUNE_DIRS:
                return False
        return True

    # Build the list of directories to include
    dirs_to_pack = list(INCLUDE_DIRS)
    if with_env:
        dirs_to_pack.append(env_dir_name())
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
                        if with_env and arcname.startswith((FULL_ENV_DIR + "/", "nninteractive_env/")):
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

# These are detected at runtime from the actual nninteractive_env.
# Fallbacks are used only if detection fails.
_PYTHON_VERSION_FALLBACK = "3.13.7"
_CUDA_INDEX_FALLBACK = "https://download.pytorch.org/whl/cu124"

# PyPI mirrors — Chinese mirrors first (much faster from mainland China)
PYPI_MIRRORS = [
    "https://pypi.tuna.tsinghua.edu.cn/simple",
    "https://mirrors.aliyun.com/pypi/simple",
    "",  # default PyPI (fallback, slow from China)
]

# Packages to exclude from pip freeze (not needed on target machine, or
# not available on PyPI as wheels and would cause pip download to fail)
_FREEZE_EXCLUDE = {
    "pkg-resources",
    # Local-only packages that are not on PyPI
    "medshot", "segmentation-platform",
}


def _detect_python_version():
    """Detect Python version from nninteractive_env."""
    python_exe = _find_python()
    if not python_exe:
        return _PYTHON_VERSION_FALLBACK
    try:
        result = subprocess.run(
            [python_exe, "-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))"],
            capture_output=True, text=True, timeout=15,
        )
        ver = result.stdout.strip()
        if ver and ver.count(".") >= 2:
            return ver
    except Exception:
        pass
    return _PYTHON_VERSION_FALLBACK


# The correct CUDA torch version for this project.
# Used to override pip freeze when local torch is broken (e.g. CPU version
# installed by mistake). Must match the version on download.pytorch.org/whl/cu124
_TORCH_CUDA_VERSION = "2.6.0+cu124"
_TORCH_CUDA_PIN = "torch==2.6.0+cu124"
_ONNXRUNTIME_GPU_PIN = "onnxruntime-gpu==1.22.0"
_OPTIONAL_REMOTE_WHEEL_PACKAGES = [
    "paramiko>=3.5,<5",
    "bcrypt>=4.2",
    "cryptography>=44",
    "PyNaCl>=1.5",
    "cffi>=1.17",
    "pycparser>=2.22",
]


def _detect_cuda_index():
    """Detect CUDA index URL from the installed torch version.

    If local torch is a CPU build (e.g. installed by mistake), fall back to
    the configured CUDA index instead of returning the CPU index.
    """
    python_exe = _find_python()
    if not python_exe:
        return _CUDA_INDEX_FALLBACK
    try:
        result = subprocess.run(
            [python_exe, "-c",
             "import torch; v=torch.__version__; idx=v.find('+cu'); print(v[idx+1:] if idx>=0 else 'cpu')"],
            capture_output=True, text=True, timeout=30,
        )
        cu_tag = result.stdout.strip()
        if cu_tag.startswith("cu"):
            return "https://download.pytorch.org/whl/{}".format(cu_tag)
        # CPU build detected — use fallback CUDA index
        print("  {} Local torch is CPU build ({}), using CUDA fallback: {}".format(
            _yellow("[--]"), result.stdout.strip(), _CUDA_INDEX_FALLBACK))
    except Exception:
        pass
    return _CUDA_INDEX_FALLBACK


def _get_installed_packages():
    """Get pip freeze output from nninteractive_env as a list of 'pkg==ver' strings.

    This captures ALL installed packages including transitive dependencies,
    ensuring the offline bundle is 100% self-contained.
    """
    python_exe = _find_python()
    if not python_exe:
        # Fallback to the static list (may miss transitive deps)
        return list(OFFLINE_WHEEL_PACKAGES_FALLBACK)
    try:
        result = subprocess.run(
            [python_exe, "-m", "pip", "freeze"],
            capture_output=True, text=True, timeout=60,
        )
        pkgs = []
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if not line or "==" not in line:
                continue
            pkg_name = line.split("==")[0].lower()
            if pkg_name in _FREEZE_EXCLUDE:
                continue
            # Skip local installs (e.g., package names with paths)
            if " @ " in line or line.startswith("-"):
                continue
            # Fix torch: if pip freeze reports a CPU build, replace with
            # the correct CUDA version so pip download fetches CUDA wheels.
            if pkg_name == "torch" and "+cpu" in line.lower():
                print("  {} Fixing torch: {} -> {}".format(
                    _yellow("[--]"), line, _TORCH_CUDA_PIN))
                line = _TORCH_CUDA_PIN
            pkgs.append(line)
        return pkgs
    except Exception:
        return list(OFFLINE_WHEEL_PACKAGES_FALLBACK)


# Fallback package list used only when pip freeze fails
OFFLINE_WHEEL_PACKAGES_FALLBACK = [
    "torch",
    "numpy",
    "nibabel",
    "pydicom",
    "SimpleITK",
    "scipy",
    "nnInteractive",
    "torchvision",
    "transformers",
    "pyyaml",
    "tqdm",
    "tensorboard",
    _ONNXRUNTIME_GPU_PIN,
    "PySide6",
    "PySide6_Essentials",
    "PySide6_Addons",
    "shiboken6",
]


def _download_torch_wheels(python_exe, wheels_dir, cuda_index):
    """Download torch + torchvision from the PyTorch CUDA index.

    These packages MUST come from download.pytorch.org, NOT from PyPI mirrors,
    because mirrors only host CPU builds. We use --index-url (not --extra)
    so pip searches exclusively on the PyTorch index.

    Returns True if successful.
    """
    # Determine torch version from the pin constant
    torch_ver = _TORCH_CUDA_VERSION  # e.g. "2.6.0+cu124"
    torch_spec = "torch=={}".format(torch_ver)

    # Also pin torchvision to a compatible version from the same CUDA index
    # torchvision wheels on the PyTorch index are already matched to torch
    torchvision_spec = "torchvision"

    print("  Downloading torch (CUDA) from PyTorch index...")
    print("    {}".format(torch_spec))
    cmd = [
        python_exe, "-m", "pip", "download",
        "--dest", str(wheels_dir),
        "--no-deps",
        "--index-url", cuda_index,
        torch_spec,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode == 0:
            print("  {} torch CUDA wheel downloaded".format(_green("[OK]")))
        else:
            err = result.stderr[-300:] if result.stderr else "unknown"
            print("  {} torch download failed: {}".format(_red("[!!]"), err.strip()))
            return False
    except subprocess.TimeoutExpired:
        print("  {} torch download timed out".format(_red("[!!]")))
        return False

    # torchvision
    print("    {}".format(torchvision_spec))
    cmd_tv = [
        python_exe, "-m", "pip", "download",
        "--dest", str(wheels_dir),
        "--no-deps",
        "--index-url", cuda_index,
        torchvision_spec,
    ]
    try:
        result = subprocess.run(cmd_tv, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            print("  {} torchvision wheel downloaded".format(_green("[OK]")))
        else:
            err = result.stderr[-300:] if result.stderr else "unknown"
            print("  {} torchvision download failed: {}".format(_yellow("[--]"), err.strip()))
            # torchvision is optional — don't fail the whole process
    except subprocess.TimeoutExpired:
        print("  {} torchvision download timed out".format(_yellow("[--]")))

    return True


def _wheel_files_for_package(wheels_dir, pkg_name):
    """Return wheel files matching *pkg_name* (case-insensitive, exact name match).

    Uses ``pkg_name + "-"`` as the prefix so ``torch`` matches ``torch-2.5.1…``
    but not ``torchvision-0.20.1…``.
    """
    distribution = re.split(r"[<>=!~\[]", str(pkg_name), maxsplit=1)[0]
    canonical = re.sub(r"[-_.]+", "-", distribution.strip().lower())
    return [
        path for path in wheels_dir.glob("*.whl")
        if re.sub(r"[-_.]+", "-", path.name.lower()).startswith(canonical + "-")
    ]


def _download_one_wheel(python_exe, wheels_dir, pkg_spec, quiet=False):
    """Download a single package as a wheel, trying each mirror in order.

    Returns True if downloaded successfully (or already exists).
    """
    pkg_name = pkg_spec.split("==")[0].split(">=")[0].split("<=")[0].strip()
    # Check if already downloaded (glob with normalized name)
    existing = _wheel_files_for_package(wheels_dir, pkg_name)
    if existing:
        return True

    for mirror in PYPI_MIRRORS:
        label = mirror or "default PyPI"
        cmd = [
            python_exe, "-m", "pip", "download",
            "--dest", str(wheels_dir),
            "--no-deps",
            "--no-cache-dir",
        ]
        if mirror:
            cmd += ["-i", mirror]
        cmd += [pkg_spec]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                return True
            # If package not found on this mirror, try next mirror
        except subprocess.TimeoutExpired:
            if not quiet:
                print("    {} {} timed out on {}".format(_yellow("[--]"), pkg_spec, label))
            continue
    return False


def _batch_download_wheels(python_exe, wheels_dir, packages, cuda_index, quiet=False):
    """Download packages as wheels, per-package with mirror fallback.

    Packages containing '+cu' (CUDA-specific builds) are skipped here —
    they are handled separately by _download_torch_wheels() which uses
    the PyTorch index directly.

    Returns (success_count, failed_list).
    """
    # Filter out CUDA-specific packages (handled by _download_torch_wheels)
    normal_pkgs = [p for p in packages if "+cu" not in p.lower()]

    if not normal_pkgs:
        return 0, []

    success_count = 0
    failed_pkgs = []
    total = len(normal_pkgs)

    for i, pkg_spec in enumerate(normal_pkgs):
        pkg_name = pkg_spec.split("==")[0].strip()
        # Skip if already downloaded
        existing = _wheel_files_for_package(wheels_dir, pkg_name)
        if existing:
            success_count += 1
            if not quiet:
                print("  [{}/{}] {} -- already downloaded".format(
                    i + 1, total, pkg_spec))
            continue

        if not quiet:
            print("  [{}/{}] {}... ".format(i + 1, total, pkg_spec), end="", flush=True)

        ok = _download_one_wheel(python_exe, wheels_dir, pkg_spec, quiet=quiet)
        if ok:
            success_count += 1
            if not quiet:
                print(_green("OK"))
        else:
            failed_pkgs.append(pkg_spec)
            if not quiet:
                print(_red("FAILED"))

    return success_count, failed_pkgs


def _ensure_bootstrap_wheels(python_exe, wheels_dir):
    """Ensure bootstrap wheels exist for offline pip bootstrap.

    Python embeddable may fail `ensurepip` on target machines, so offline
    `get-pip.py --no-index` needs local wheels for pip/setuptools/wheel.
    """
    bootstrap_specs = ["pip", "setuptools", "wheel"]
    present = []
    missing = []
    for spec in bootstrap_specs:
        if _wheel_files_for_package(wheels_dir, spec):
            present.append(spec)
        else:
            missing.append(spec)

    for spec in missing:
        ok = _download_one_wheel(python_exe, wheels_dir, spec)
        if ok:
            present.append(spec)

    return present


def offline_bundle():
    """Download everything needed for a fully offline Windows deployment.

    Creates a self-contained directory with:
      - Python embeddable matching nninteractive_env (no install, no admin)
      - All .whl files matching the EXACT installed versions (from pip freeze)
      - Source code, configs, and model weights
      - setup_offline.bat (double-click to install)

    Copy the resulting directory to a USB drive and run
    setup_offline.bat on the target Windows machine.
    """
    import urllib.request
    import zipfile
    import tempfile

    # Detect versions from the actual environment
    python_version = _detect_python_version()
    cuda_index = _detect_cuda_index()
    python_exe = _find_python()
    python_short = "python" + python_version.replace(".", "")[:3]  # e.g. python313
    python_embed_url = (
        "https://www.python.org/ftp/python/{0}/python-{0}-embed-amd64.zip"
    ).format(python_version)

    print("Detected environment:")
    print("  Python: {}".format(python_version))
    print("  CUDA index: {}".format(cuda_index))
    print("  Python exe: {}".format(python_exe or "NOT FOUND"))
    print()
    if not python_exe:
        print("  {} nninteractive_env Python was not found.".format(_red("[!!]")))
        print("    Build/check nninteractive_env first; offline-bundle must download wheels from that environment.")
        return 1

    # Output beside the project (E:\mimics_script_bundle), NOT inside it and
    # never the same name as an existing install — a bundle write must not
    # be able to clobber a live checkout or backup.
    bundle_dir = PROJECT_ROOT.parent / "mimics_script_bundle"
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
        print("  Downloading Python {} embeddable...".format(python_version))
        print("  URL: {}".format(python_embed_url))
        try:
            urllib.request.urlretrieve(python_embed_url, str(python_zip))
            print("  {} Downloaded".format(_green("[OK]")))
        except Exception as exc:
            print("  {} Download failed: {}".format(_red("[!!]"), exc))
            print("    Manual: download from {}".format(python_embed_url))
            print("    Save as: {}".format(python_zip))

    if python_zip.exists():
        with zipfile.ZipFile(str(python_zip), "r") as zf:
            zf.extractall(str(python_dir))
        print("  {} Python embeddable extracted to python/".format(_green("[OK]")))

    # Download get-pip.py as fallback (Python embeddable doesn't include pip)
    get_pip_path = bundle_dir / "get-pip.py"
    if get_pip_path.exists():
        print("  {} get-pip.py already present".format(_green("[OK]")))
    else:
        print("  Downloading get-pip.py (fallback for pip)...")
        try:
            urllib.request.urlretrieve(
                "https://bootstrap.pypa.io/get-pip.py", str(get_pip_path))
            print("  {} get-pip.py downloaded".format(_green("[OK]")))
        except Exception as exc:
            print("  {} get-pip.py download failed: {}".format(_yellow("[--]"), exc))
            print("    ensurepip should still work as primary method")

    # Use the nninteractive_env Python (not sys.executable) so pip downloads
    # wheels matching the correct Python version and platform.
    download_python = python_exe

    # 2a. Download torch + torchvision from PyTorch CUDA index FIRST.
    #     This is critical: PyPI mirrors do NOT host CUDA torch wheels.
    #     Must use --index-url (exclusive) to download.pytorch.org.
    print(_header("2a. torch + torchvision (CUDA, from PyTorch index)"))
    torch_ok = _download_torch_wheels(download_python, wheels_dir, cuda_index)
    if not torch_ok:
        print("  {} Failed to download CUDA torch. Aborting.".format(_red("[!!]")))
        return 1

    # 2b. Download all other wheels using pip freeze from the actual environment.
    #     pip freeze captures EVERY installed package (including all transitive
    #     dependencies), so the resulting wheels/ directory is 100% self-contained.
    #     We use --no-deps because pip freeze already lists all dependencies.
    print(_header("2b. Other wheels (from pip freeze — exact installed versions)"))
    freeze_packages = _get_installed_packages()
    freeze_packages = [
        spec
        for spec in freeze_packages
        if re.sub(r"[-_.]+", "-", spec.split("==", 1)[0].lower())
        not in ("onnxruntime", "onnxruntime-gpu")
    ]
    freeze_packages.append(_ONNXRUNTIME_GPU_PIN)
    installed_names = {
        re.sub(r"[-_.]+", "-", spec.split("==", 1)[0].split(">=", 1)[0].lower())
        for spec in freeze_packages
    }
    for spec in _OPTIONAL_REMOTE_WHEEL_PACKAGES:
        name = re.sub(
            r"[-_.]+", "-", spec.split("==", 1)[0].split(">=", 1)[0].lower()
        )
        if name not in installed_names:
            freeze_packages.append(spec)
            installed_names.add(name)
    print("  Found {} packages in pip freeze".format(len(freeze_packages)))
    print("  Downloading exact versions... this may take 15-30 minutes.")
    print()

    # Download each package individually with mirror fallback.
    # This is more resilient than batch download: one bad package won't
    # block the rest, and we get per-package progress output.
    success_count, failed_pkgs = _batch_download_wheels(
        download_python, wheels_dir, freeze_packages, cuda_index)

    if failed_pkgs:
        print()
        print("  {} {} package(s) could not be downloaded:".format(
            _yellow("[--]"), len(failed_pkgs)))
        for p in failed_pkgs:
            print("    - {}".format(p))
        print("  These may be local-only packages. Check if they are needed.")

    total_wheels = len(list(wheels_dir.glob("*.whl")))
    if total_wheels == 0:
        print("  {} No wheels downloaded at all.".format(_red("[!!]")))
        return 1
    total_size = sum(f.stat().st_size for f in wheels_dir.glob("*.whl"))
    print()
    print("  {} Downloaded {} wheels ({:.0f} MB), {} failed".format(
        _green("[OK]"), total_wheels, total_size / 1e6, len(failed_pkgs)))

    # Verify key packages
    print()
    print("  Verifying key packages in wheels/:")
    for pkg in [
        "torch", "numpy", "nibabel", "nnInteractive", "scipy",
        "onnxruntime-gpu", "PySide6", "shiboken6", "paramiko",
    ]:
        found = _wheel_files_for_package(wheels_dir, pkg)
        if found:
            print("    {} {}: {} file(s)".format(_green("[OK]"), pkg, len(found)))
        else:
            print("    {} {}: MISSING".format(_red("[!!]"), pkg))

    # Ensure pip bootstrap wheels exist for get-pip.py offline fallback
    print()
    print("  Ensuring pip bootstrap wheels for offline get-pip.py...")
    bootstrap_ok = _ensure_bootstrap_wheels(download_python, wheels_dir)
    for pkg in ["pip", "setuptools", "wheel"]:
        state = _green("[OK]") if pkg in bootstrap_ok else _red("[!!]")
        print("    {} {}".format(state, pkg))

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
                                                       ".git", "docs", "tests", "validation",
                                                       "work", ".cache", ".pytest_cache"))
        print("  {} {}/".format(_green("[+]"), name))
    for name in INCLUDE_FILES:
        src = PROJECT_ROOT / name
        if src.is_file():
            shutil.copy2(str(src), str(bundle_dir / name))
            print("  {} {}".format(_green("[+]"), name))

    missing_ui_files = [
        name for name in REQUIRED_EXTERNAL_UI_FILES
        if not (bundle_dir / name).is_file()
    ]
    if missing_ui_files:
        print("  {} Offline bundle is missing required UI files:".format(_red("[!!]")))
        for name in missing_ui_files:
            print("    - {}".format(name))
        return 1

    # 4. Generate setup_offline.bat
    print(_header("4. setup_offline.bat"))
    bat_path = bundle_dir / "setup_offline.bat"
    bat_content = _generate_offline_bat(python_version, python_short)
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


def _generate_offline_bat(python_version, python_short):
    """Generate the Windows batch script for offline setup.

    Args:
        python_version: Full version string, e.g. "3.13.7"
        python_short: Short name for .pth file, e.g. "python313"
    """
    pth_name = python_short + "._pth"  # e.g. python313._pth
    zip_name = python_short + ".zip"   # e.g. python313.zip
    env = env_dir_name()
    lines = []
    lines.append("@echo off")
    lines.append("setlocal enabledelayedexpansion")
    lines.append("echo ========================================")
    lines.append("echo   Mimics-Script Offline Setup")
    lines.append("echo   Target: Windows + Python {}".format(python_version))
    lines.append("echo ========================================")
    lines.append("echo.")
    lines.append("")
    lines.append("cd /d %~dp0")
    lines.append("")
    lines.append(":: 1. Setup Python embeddable (self-contained, no system Python needed)")
    lines.append('echo [1/6] Setting up Python {}...'.format(python_version))
    lines.append('if not exist "@@ENV@@\\python.exe" (')
    lines.append('    echo   Extracting Python embeddable...')
    lines.append('    if not exist "python\\python.exe" (')
    lines.append('        echo   ERROR: python\\python.exe not found.')
    lines.append('        echo   Make sure the python/ directory is present.')
    lines.append("        pause")
    lines.append("        exit /b 1")
    lines.append("    )")
    lines.append('    mkdir @@ENV@@ 2>nul')
    lines.append('    xcopy /E /Q /Y python\\* @@ENV@@\\ >nul')
    lines.append('    echo   Configuring {}...'.format(pth_name))
    lines.append('    echo {} > @@ENV@@\\{}'.format(zip_name, pth_name))
    lines.append('    echo . >> @@ENV@@\\{}'.format(pth_name))
    lines.append('    echo Lib >> @@ENV@@\\{}'.format(pth_name))
    lines.append('    echo Lib\\site-packages >> @@ENV@@\\{}'.format(pth_name))
    lines.append('    echo import site >> @@ENV@@\\{}'.format(pth_name))
    lines.append(")")
    lines.append(':: Ensure Lib is on the path (idempotent; also covers upgraded installs)')
    lines.append('echo {} > @@ENV@@\\{}'.format(zip_name, pth_name))
    lines.append('echo . >> @@ENV@@\\{}'.format(pth_name))
    lines.append('echo Lib >> @@ENV@@\\{}'.format(pth_name))
    lines.append('echo Lib\\site-packages >> @@ENV@@\\{}'.format(pth_name))
    lines.append('echo import site >> @@ENV@@\\{}'.format(pth_name))
    lines.append('if not exist "@@ENV@@\\Lib\\site-packages" mkdir @@ENV@@\\Lib\\site-packages')
    lines.append("")
    lines.append(":: 2. Install pip (try ensurepip, fallback to get-pip.py)")
    lines.append('echo [2/6] Installing pip...')
    lines.append('@@ENV@@\\python.exe -m pip --version >nul 2>&1')
    lines.append('if !errorlevel! neq 0 (')
    lines.append('    echo   Trying ensurepip...')
    lines.append('    @@ENV@@\\python.exe -m ensurepip --upgrade 2>nul')
    lines.append('    @@ENV@@\\python.exe -m pip --version >nul 2>&1')
    lines.append('    if !errorlevel! neq 0 (')
    lines.append('        echo   ensurepip failed. Trying get-pip.py...')
    lines.append('        if exist "get-pip.py" (')
    lines.append('            @@ENV@@\\python.exe get-pip.py --no-index --find-links="wheels" pip setuptools wheel 2>nul')
    lines.append('            @@ENV@@\\python.exe -m pip --version >nul 2>&1')
    lines.append('            if !errorlevel! neq 0 (')
    lines.append('                echo   ERROR: pip installation failed.')
    lines.append('                echo   Manual: @@ENV@@\\python.exe get-pip.py --no-index --find-links="wheels" pip setuptools wheel')
    lines.append("                pause")
    lines.append("                exit /b 1")
    lines.append("            )")
    lines.append('        ) else (')
    lines.append('            echo   ERROR: get-pip.py not found and ensurepip failed.')
    lines.append('            echo   pip is required to install packages.')
    lines.append("            pause")
    lines.append("            exit /b 1")
    lines.append("        )")
    lines.append("    )")
    lines.append(")")
    lines.append('echo   pip is ready.')
    lines.append("")
    lines.append(":: 3. Check wheels directory")
    lines.append('echo [3/6] Checking wheels...')
    lines.append('if not exist "wheels\\*.whl" (')
    lines.append('    echo   ERROR: No .whl files found in wheels\\ directory.')
    lines.append('    echo   The wheels/ directory must contain all required packages.')
    lines.append("    pause")
    lines.append("    exit /b 1")
    lines.append(")")
    lines.append('dir /b wheels\\*.whl ^| find /c /v "" >nul 2>&1')
    lines.append('echo   Wheels directory OK.')
    lines.append("")
    lines.append(":: 4. Install all wheels offline (no internet, no system Python)")
    lines.append('echo [4/6] Installing packages from local wheels (no internet)...')
    lines.append('set FAIL_COUNT=0')
    lines.append('for %%f in (wheels\\*.whl) do (')
    lines.append('    echo   Installing %%~nxf...')
    lines.append('    @@ENV@@\\python.exe -m pip install "%%f" --no-deps --no-index --quiet 2>nul')
    lines.append('    if !errorlevel! neq 0 (')
    lines.append('        echo     WARNING: Failed to install %%~nxf')
    lines.append('        set /a FAIL_COUNT+=1')
    lines.append("    )")
    lines.append(")")
    lines.append('echo   Installation complete. !FAIL_COUNT! package(s) failed.')
    lines.append('echo   Ensuring PySide6 advanced UI wheels are installed consistently...')
    lines.append('@@ENV@@\\python.exe -m pip install PySide6 shiboken6 --no-index --find-links="wheels" --upgrade --quiet')
    lines.append('if !errorlevel! neq 0 (')
    lines.append('    echo   ERROR: PySide6 installation failed.')
    lines.append('    echo   Make sure wheels\\ contains matching PySide6, PySide6_Essentials, PySide6_Addons, and shiboken6 Windows wheels.')
    lines.append("    pause")
    lines.append("    exit /b 1")
    lines.append(")")
    lines.append("")
    lines.append(":: 5. Verify external GUI backend")
    lines.append('echo [5/6] Verifying PySide6 external UI backend...')
    lines.append("@@ENV@@\\python.exe -c \"import PySide6, shiboken6; from PySide6 import QtCore, QtWidgets; print('  PySide6', QtCore.__version__)\"")
    lines.append('if !errorlevel! neq 0 (')
    lines.append("    echo   ERROR: PySide6 import failed.")
    lines.append("    echo   Advanced DINOv3 Setup and Status windows require PySide6 in @@ENV@@.")
    lines.append("    pause")
    lines.append("    exit /b 1")
    lines.append(")")
    lines.append("")
    lines.append(":: 6. Verify")
    lines.append('echo [6/6] Verifying installation...')
    lines.append("@@ENV@@\\python.exe -c \"import torch; print('  torch', torch.__version__); print('  CUDA available:', torch.cuda.is_available())\"")
    lines.append('if !errorlevel! neq 0 (')
    lines.append("    echo   ERROR: torch import failed.")
    lines.append("    pause")
    lines.append("    exit /b 1")
    lines.append(")")
    lines.append("@@ENV@@\\python.exe -c \"import numpy, nibabel, pydicom, SimpleITK, scipy, nnInteractive, nnunetv2, torchvision, transformers, yaml, tqdm, tensorboard, tomli, acvl_utils, onnxruntime, PySide6, shiboken6; from importlib.metadata import version as package_version; from packaging.version import Version; nnv=Version(package_version('nnunetv2')); assert Version('2.8.1') ^<= nnv ^< Version('2.9'), 'nnunetv2 2.8.1 through 2.8.x is required'; print('  All packages OK'); print('  nnU-Net', nnv); print('  ONNX providers:', ', '.join(onnxruntime.get_available_providers()))\"")
    lines.append('if !errorlevel! neq 0 (')
    lines.append("    echo   Some packages failed to import. The default frozen-feature method requires onnxruntime-gpu.")
    lines.append('    echo   Try: @@ENV@@\\python.exe -m pip install wheels\\*.whl --no-deps --no-index')
    lines.append("    pause")
    lines.append("    exit /b 1")
    lines.append(")")
    lines.append("@@ENV@@\\python.exe -c \"import paramiko; print('  Optional remote training transport ready:', paramiko.__version__)\"")
    lines.append("if !errorlevel! neq 0 (")
    lines.append("    echo   WARNING: Paramiko is unavailable. Local training is unaffected; remote training is disabled.")
    lines.append(")")
    lines.append(
        'if not exist "integrations\\dinov3-medical-seg\\models\\dinov3-vits16\\model.onnx" ('
    )
    lines.append("    echo   ERROR: The default ViT-S/16 ONNX encoder is missing.")
    lines.append(
        "    echo   Expected: integrations\\dinov3-medical-seg\\models\\dinov3-vits16\\model.onnx"
    )
    lines.append("    pause")
    lines.append("    exit /b 1")
    lines.append(")")
    lines.append(
        'if not exist "integrations\\ScribblePrompt\\checkpoints\\ScribblePrompt_unet_v1_nf192_res128.pt" ('
    )
    lines.append("    echo   WARNING: The official ScribblePrompt UNet checkpoint is missing.")
    lines.append(
        "    echo   Expected: integrations\\ScribblePrompt\\checkpoints\\ScribblePrompt_unet_v1_nf192_res128.pt"
    )
    lines.append(")")
    lines.append("")
    lines.append("echo.")
    lines.append("echo ========================================")
    lines.append("echo   Setup complete!")
    lines.append("echo   In Mimics: Scripting -^> Add Scripting Library")
    lines.append("echo   Select: scripting_library\\ folder")
    lines.append("echo ========================================")
    lines.append("pause")
    # Fill in the environment directory name (@@ENV@@ placeholder). Fresh
    # packages use python_env; a legacy nninteractive_env project keeps its
    # existing directory so the bundle matches its source.
    return "\r\n".join(lines).replace("@@ENV@@", env)


def _header(text):
    return "\n" + "=" * 64 + "\n  " + text + "\n" + "=" * 64


if __name__ == "__main__":
    sys.exit(main())
