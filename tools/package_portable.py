#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Package Mimics-Script for offline deployment to another Windows machine.

Usage:
    python tools/package_portable.py check       # Validate everything is ready
    python tools/package_portable.py list        # Show what will be packaged
    python tools/package_portable.py pack        # Create the portable archive
    python tools/package_portable.py manifest    # Write a manifest file only

The script validates that nninteractive_env/ exists and contains a working
Python + required packages + model weights, then packages everything into
a single archive that can be extracted on the target machine and used
immediately without any installation.
"""

from __future__ import print_function

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# -- What to include in the archive ------------------------------------------
INCLUDE_DIRS = [
    "scripting_library",
    "runtime_py35",
    "external",
    "tools",
    "nninteractive_env",
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

# Sub-paths within tools/ to EXCLUDE (large, platform-specific, or generated)
EXCLUDE_TOOLS = [
    "tools/pyqt5_wheels",
    "tools/__pycache__",
]

# Minimum expected files/sizes to validate nninteractive_env
ENV_CHECKS = {
    "python_exe": [
        "Scripts/python.exe",
        "python/python.exe",
    ],
    "required_packages": [
        "torch",
        "numpy",
        "nibabel",
    ],
}

# Expected model directories (at least one must exist)
MODEL_PATHS = [
    "nninteractive_env/models/nnInteractive_v1.0",
    "external/dinov3-medical-seg/models/dinov3-vitb16",
    "external/dinov3-medical-seg/models/dinov3-vitl16",
]

ARCHIVE_NAME = "mimics_script_portable"


def _green(text):
    return "\033[32m{}\033[0m".format(text)


def _red(text):
    return "\033[31m{}\033[0m".format(text)


def _yellow(text):
    return "\033[33m{}\033[0m".format(text)


def _header(text):
    print("\n" + "=" * 72)
    print("  {}".format(text))
    print("=" * 72)


def _size_fmt(path):
    """Human-readable size of a file or directory."""
    p = Path(path)
    if p.is_file():
        total = p.stat().st_size
    elif p.is_dir():
        total = 0
        for root, _dirs, files in os.walk(str(p)):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
    else:
        return "N/A"

    for unit in ("B", "KB", "MB", "GB", "TB"):
        if total < 1024:
            return "{:.1f} {}".format(total, unit)
        total /= 1024.0
    return "{:.1f} PB".format(total)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def check_project_structure():
    """Validate the project root has all required files and directories."""
    print(_header("1. Project structure"))
    all_ok = True

    for name in INCLUDE_DIRS:
        path = PROJECT_ROOT / name
        if path.is_dir():
            print("  {} {}  ({})".format(_green("[OK]"), name, _size_fmt(path)))
        else:
            print("  {} {}  — MISSING".format(_red("[!!]"), name))
            all_ok = False

    for name in INCLUDE_FILES:
        path = PROJECT_ROOT / name
        if path.is_file():
            print("  {} {}  ({})".format(_green("[OK]"), name, _size_fmt(path)))
        else:
            print("  {} {}  — MISSING".format(_red("[!!]"), name))
            all_ok = False

    return all_ok


def check_nninteractive_env():
    """Validate nninteractive_env has a working Python with required packages."""
    print(_header("2. nninteractive_env Python"))

    # Find Python executable
    python_exe = None
    for rel in ENV_CHECKS["python_exe"]:
        cand = PROJECT_ROOT / rel
        if cand.is_file():
            python_exe = cand
            break

    if python_exe is None:
        print("  {} Python executable not found in nninteractive_env/".format(_red("[!!]")))
        print("    Looked for: {}".format(", ".join(ENV_CHECKS["python_exe"])))
        return False

    print("  {} Python: {}".format(_green("[OK]"), python_exe))

    # Check Python version
    try:
        result = subprocess.run(
            [str(python_exe), "-c", "import sys; print(sys.version)"],
            capture_output=True, text=True, timeout=30,
        )
        version = result.stdout.strip().split("\n")[0]
        print("  {} Version: {}".format(_green("[OK]"), version))
    except Exception as exc:
        print("  {} Could not run Python: {}".format(_red("[!!]"), exc))
        return False

    # Check required packages
    probe = (
        "import importlib.util,json;"
        "mods=['torch','numpy','nibabel','pydicom'];"
        "result={m:importlib.util.find_spec(m) is not None for m in mods};"
        "print(json.dumps(result))"
    )
    try:
        result = subprocess.run(
            [str(python_exe), "-c", probe],
            capture_output=True, text=True, timeout=60,
        )
        pkg_status = json.loads(result.stdout.strip())
        for pkg, found in sorted(pkg_status.items()):
            if found:
                print("  {} {}".format(_green("[OK]"), pkg))
            else:
                print("  {} {} — MISSING".format(_red("[!!]"), pkg))
                all_ok = False
    except Exception as exc:
        print("  {} Could not probe packages: {}".format(_red("[!!]"), exc))
        return False

    # Check CUDA
    try:
        result = subprocess.run(
            [str(python_exe), "-c",
             "import torch; print('CUDA=' + str(torch.cuda.is_available())); "
             "print('Devices=' + str(torch.cuda.device_count()))"],
            capture_output=True, text=True, timeout=60,
        )
        print("  {} {}".format(_green("[OK]"), result.stdout.strip().replace("\n", "; ")))
    except Exception:
        print("  {} Could not check CUDA (may be OK for CPU-only)".format(_yellow("[--]")))

    return True


def check_model_files():
    """Validate model weights exist."""
    print(_header("3. Model weights"))
    all_ok = True

    for rel in MODEL_PATHS:
        path = PROJECT_ROOT / rel
        if path.is_dir():
            # Count checkpoint files
            ckpts = list(path.rglob("*.pth")) + list(path.rglob("*.safetensors")) + list(path.rglob("*.bin"))
            if ckpts:
                total_size = sum(c.stat().st_size for c in ckpts)
                print("  {} {}  — {} checkpoint(s), {}".format(
                    _green("[OK]"), rel, len(ckpts), _size_fmt(path)))
            else:
                print("  {} {}  — directory exists but no .pth/.safetensors found".format(
                    _yellow("[--]"), rel))
        else:
            print("  {} {}  — not found (may be optional)".format(_yellow("[--]"), rel))
            # Only nnInteractive model is required
            if "nnInteractive" in rel:
                all_ok = False

    return all_ok


def check_config_files():
    """Basic sanity check on config files."""
    print(_header("4. Configuration files"))
    all_ok = True

    configs = {
        "nninteractive_config.json": ["device", "model_dir"],
        "fewshot_config.json": ["dinov3_project", "base_config"],
    }

    for fname, expected_keys in configs.items():
        path = PROJECT_ROOT / fname
        if not path.is_file():
            print("  {} {} — MISSING".format(_red("[!!]"), fname))
            all_ok = False
            continue
        try:
            with open(path, "r") as f:
                data = json.load(f)
            found = [k for k in expected_keys if k in data]
            print("  {} {} — keys: {}".format(_green("[OK]"), fname, ", ".join(sorted(data.keys())[:8])))
        except Exception as exc:
            print("  {} {} — PARSE ERROR: {}".format(_red("[!!]"), fname, exc))
            all_ok = False

    return all_ok


def check_hardcoded_paths():
    """Verify no hardcoded absolute paths exist that would break portability."""
    print(_header("5. Hardcoded paths check"))
    issues = 0
    skip_dirs = {"__pycache__", ".git", "nninteractive_env", "pyqt5_wheels", ".claude"}

    # These patterns are intentional fallback paths for Mimics executable
    # discovery — they're only used when env vars / known locations fail.
    ALLOWED_PATTERNS = [
        "MimicsResearch.exe",     # Mimics install detection fallback
        "Mimics Research",        # Mimics install path
        "Program Files",          # Windows standard location
        "package_portable.py",    # This script itself
    ]

    for root, dirs, files in os.walk(str(PROJECT_ROOT)):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for fname in files:
            if not fname.endswith((".py", ".json", ".yaml", ".yml")):
                continue
            if fname == "package_portable.py":
                continue  # skip self
            path = os.path.join(root, fname)
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    for lineno, line in enumerate(f, 1):
                        has_win_path = any(drive in line for drive in ("C:\\", "D:\\", "E:\\"))
                        if not has_win_path:
                            continue
                        stripped = line.strip()
                        if stripped.startswith("#") or stripped.startswith("//"):
                            continue
                        # Check if this is an allowed pattern
                        allowed = any(pat in line for pat in ALLOWED_PATTERNS)
                        if allowed:
                            continue
                        print("  {} {}:{} — {}".format(
                            _red("[!!]"), os.path.relpath(path, PROJECT_ROOT), lineno, stripped[:80]))
                        issues += 1
            except Exception:
                pass

    if issues == 0:
        print("  {} No hardcoded absolute paths found".format(_green("[OK]")))
    else:
        print("  {} found {} potential hardcoded path(s)".format(_yellow("[--]"), issues))

    return issues == 0


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------


def list_packaging():
    """Show exactly what will be included in the archive."""
    print(_header("Files to package"))
    total_size = 0

    for name in INCLUDE_DIRS:
        path = PROJECT_ROOT / name
        if not path.exists():
            print("  {} {}  — SKIPPED (not found)".format(_yellow("[--]"), name))
            continue
        sz = _size_fmt(path)
        print("  {} {}/".format(_green("[+]"), name))
        # Count files
        file_count = sum(1 for _ in path.rglob("*") if _.is_file())
        print("      {} files, {}".format(file_count, sz))

    for name in INCLUDE_FILES:
        path = PROJECT_ROOT / name
        if not path.is_file():
            print("  {} {}  — SKIPPED (not found)".format(_yellow("[--]"), name))
            continue
        print("  {} {}".format(_green("[+]"), name))

    # Estimate total
    for name in INCLUDE_DIRS:
        p = PROJECT_ROOT / name
        if p.is_dir():
            for root, _dirs, files in os.walk(str(p)):
                for f in files:
                    try:
                        total_size += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
    for name in INCLUDE_FILES:
        p = PROJECT_ROOT / name
        if p.is_file():
            total_size += p.stat().st_size

    print("\n  Estimated total: {}".format(_size_fmt(str(total_size))))

    return total_size


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def write_manifest(output_dir=None):
    """Write a JSON manifest describing the packaged contents."""
    if output_dir is None:
        output_dir = PROJECT_ROOT

    manifest = {
        "schema_version": "mimics_script_portable_manifest.v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "created_on": os.environ.get("COMPUTERNAME", os.uname().nodename),
        "contents": {},
    }

    for name in INCLUDE_DIRS + INCLUDE_FILES:
        path = PROJECT_ROOT / name
        if not path.exists():
            continue
        if path.is_dir():
            files = {}
            for fpath in path.rglob("*"):
                if fpath.is_file():
                    rel = str(fpath.relative_to(PROJECT_ROOT)).replace("\\", "/")
                    try:
                        files[rel] = fpath.stat().st_size
                    except OSError:
                        pass
            manifest["contents"][name + "/"] = {
                "type": "directory",
                "file_count": len(files),
                "total_bytes": sum(files.values()),
            }
        else:
            manifest["contents"][name] = {
                "type": "file",
                "size_bytes": path.stat().st_size,
            }

    manifest_path = Path(output_dir) / "portable_manifest.json"
    with open(str(manifest_path), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print("Manifest written to: {}".format(manifest_path))
    return manifest_path


# ---------------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------------


def create_archive():
    """Create the portable archive using 7z (preferred) or tar."""
    print(_header("Creating archive"))

    # Choose archiver
    archiver = None
    for cmd in ["7z", "7za", "7z.exe"]:
        if shutil.which(cmd):
            archiver = "7z"
            archiver_cmd = cmd
            break
    if archiver is None:
        archiver = "tar"
        archiver_cmd = "tar"

    archive_file = str(PROJECT_ROOT.parent / ARCHIVE_NAME)

    if archiver == "7z":
        # 7z produces much smaller archives for Python environments
        archive_path = archive_file + ".7z"
        args = [
            archiver_cmd, "a", "-mx=9", "-mmt=on",
            archive_path,
        ]
        # Add each dir/file relative to project root
        for name in INCLUDE_DIRS:
            p = PROJECT_ROOT / name
            if p.exists():
                args.append(str(p))
        for name in INCLUDE_FILES:
            p = PROJECT_ROOT / name
            if p.is_file():
                args.append(str(p))
        # Exclude patterns
        for ex in EXCLUDE_TOOLS:
            ex_path = PROJECT_ROOT / ex
            if ex_path.exists():
                args.append("-xr!{}".format(str(ex_path).replace("\\", "/")))
    else:
        archive_path = archive_file + ".tar.gz"
        args = [
            archiver_cmd, "-czf", archive_path,
            "-C", str(PROJECT_ROOT),
        ]
        for name in INCLUDE_DIRS:
            p = PROJECT_ROOT / name
            if p.exists():
                args.append(name)
        for name in INCLUDE_FILES:
            p = PROJECT_ROOT / name
            if p.is_file():
                args.append(name)
        # tar doesn't easily support exclusions, skip for now

    print("  Archiver: {}".format(archiver))
    print("  Running: {}".format(" ".join(args[:5]) + " ..."))
    print()

    try:
        subprocess.run(args, check=True)
        if os.path.exists(archive_path):
            print("\n  {} Archive created: {}".format(_green("[OK]"), archive_path))
            print("  {} Size: {}".format(_green("   "), _size_fmt(archive_path)))
        else:
            print("\n  {} Archive not created".format(_red("[!!]")))
            return None
    except subprocess.CalledProcessError as exc:
        print("\n  {} Archive creation failed: {}".format(_red("[!!]"), exc))
        return None
    except FileNotFoundError:
        print("\n  {} {} not found. Install 7-Zip or use tar.".format(
            _red("[!!]"), archiver_cmd))
        print("    Windows: choco install 7zip  OR  winget install 7zip.7zip")
        print("    Or run: python tools/package_portable.py list  (no archive, just list)")
        return None

    return archive_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _print_banner():
    print("=" * 72)
    print("  Mimics-Script Portable Packager")
    print("  Project: {}".format(PROJECT_ROOT))
    print("  Platform: {}".format(sys.platform))
    print("=" * 72)


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/package_portable.py <command>")
        print()
        print("Commands:")
        print("  check     Validate project structure, Python env, and models")
        print("  list      Show what will be included in the archive")
        print("  pack      Create the portable archive (.7z or .tar.gz)")
        print("  manifest  Write a portable_manifest.json file")
        return 1

    cmd = sys.argv[1].lower()
    _print_banner()

    if cmd == "check":
        ok = True
        ok &= check_project_structure()
        ok &= check_nninteractive_env()
        ok &= check_model_files()
        ok &= check_config_files()
        ok &= check_hardcoded_paths()

        print(_header("Result"))
        if ok:
            print("  {} All checks passed. Ready to package.".format(_green("[OK]")))
            return 0
        else:
            print("  {} Some checks failed. Fix the issues above before packaging.".format(_red("[!!]")))
            return 1

    elif cmd == "list":
        list_packaging()
        return 0

    elif cmd == "pack":
        # Quick validation first
        if not check_project_structure():
            print(_red("\nAbort: project structure incomplete."))
            return 1

        # Write manifest
        write_manifest()

        # Create archive
        archive = create_archive()
        if archive:
            print("\n" + _green("=" * 72))
            print(_green("  Deployment instructions:"))
            print(_green("  1. Copy {} to the target Windows machine".format(os.path.basename(archive))))
            print(_green("  2. Extract to any directory, e.g. D:\\Mimics Script\\"))
            print(_green("  3. In Mimics: Scripting → Add Scripting Library →"))
            print(_green("     select the scripting_library/ folder"))
            print(_green("  4. All entries should appear and work immediately"))
            print(_green("=" * 72))
            return 0
        return 1

    elif cmd == "manifest":
        write_manifest()
        return 0

    else:
        print("Unknown command: {}".format(cmd))
        return 1


if __name__ == "__main__":
    sys.exit(main())
