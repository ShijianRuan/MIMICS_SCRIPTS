#!/usr/bin/env python3
"""Collect a redacted diagnostics bundle for support requests.

Aggregates the tail of every operational log, a census of .mimics_runtime
state (processes, locks, queues, servers), config snapshots, environment
versions, and GPU status into a single zip the annotator can hand over.

Privacy: absolute paths are rewritten to their last two components; case IDs
are preserved (model diagnosis needs them) but source paths around them are
redacted. Tokens/api keys are masked.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

LOG_TAIL_LINES = 200
BUNDLE_MAX_BYTES = 10 * 1024 * 1024

# (label, runtime-relative log path) — tails are read as text; missing
# files are skipped. Resolved lazily so tests can point ROOT at a scratch
# tree.
LOG_TARGETS = [
    ("logs/import_tail.log", "import.log"),
    ("logs/export_tail.log", "export.log"),
    ("logs/setup_env_tail.log", "setup_env.log"),
    ("logs/bridge_tail.log", "bridge.log"),
    ("logs/server_tail.log", "server.log"),
]

CONFIG_SNAPSHOTS = [
    ("configs/fewshot_config.json", "fewshot_config.json"),
    ("configs/nninteractive_config.json", "nninteractive_config.json"),
    ("configs/nninteractive_finetune_config.json", "nninteractive_finetune_config.json"),
    ("configs/mimics_io_config.json", "mimics_io_config.json"),
]

_RUNTIME_DIRS = ["processes", "locks", "import_queues", "drop_import", "export_jobs", "append_jobs"]


LOG_TAIL_MAX_BYTES = 2 * 1024 * 1024


def _tail_text(path: Path, max_lines: int = LOG_TAIL_LINES) -> str:
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > LOG_TAIL_MAX_BYTES:
                handle.seek(max(0, size - LOG_TAIL_MAX_BYTES))
                handle.readline()  # drop the partial first line
            text = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:]) + "\n" if lines else ""


# Matches Windows and POSIX absolute paths (drive letters, UNC, leading /).
_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Za-z]:\\[^\s\"',;:()\[\]]+|\\\\[^\s\"',;:()\[\]]+|/[^\s\"',;:()\[\]]+)"
)
_DRIVE_SUFFIX_RE = re.compile(r"\.(?:exe|dll|pyd|bat|py|json|mcs|nii|nii\.gz|dcm|pth|onnx)$", re.IGNORECASE)


def redact_path(match: re.Match) -> str:
    """Rewrite one absolute path to its last two components."""
    path = match.group(0)
    parts = [p for p in re.split(r"[\\/]+", path.rstrip("\\/")) if p]
    if len(parts) <= 2:
        return path
    return ".../" + "/".join(parts[-2:])


def redact_text(text: str) -> str:
    text = _PATH_RE.sub(redact_path, text)
    # Mask anything that looks like a bearer token or api key value.
    text = re.sub(r"(Bearer\s+)[A-Za-z0-9._\-]+", r"\1***", text)
    text = re.sub(
        r"([\"']?api[-_]?key[\"']?\s*[:=]\s*[\"']?)[A-Za-z0-9._\-]+",
        r"\1***",
        text,
        flags=re.IGNORECASE,
    )
    return text


def _safe_name(path: Path, base: Path) -> str:
    try:
        rel = path.relative_to(base)
    except ValueError:
        return path.name
    return rel.as_posix()


def _runtime_census() -> dict:
    census: dict = {"collected_at_epoch": time.time()}
    runtime = ROOT / ".mimics_runtime"
    for name in _RUNTIME_DIRS:
        target = runtime / name
        entries = {}
        if target.is_dir():
            try:
                for child in sorted(target.iterdir()):
                    if child.is_dir():
                        count = sum(1 for _ in child.iterdir())
                        entries[child.name + "/"] = count
                    else:
                        try:
                            entries[child.name] = child.stat().st_size
                        except OSError:
                            entries[child.name] = -1
            except OSError:
                pass
        census[name] = entries
    return census


def _environment_info() -> dict:
    info = {"python_version": sys.version}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        info["torch"] = "unavailable"
    for module_name in ("nnunetv2", "nnInteractive"):
        try:
            module = __import__(module_name)
            info[module_name] = getattr(module, "__version__", "unknown")
        except Exception:
            info[module_name] = "unavailable"
    return info


def _collect_registry_rows() -> dict:
    rows = {}
    home_registry = Path.home() / ".mimics_script"
    for name in ("fewshot_model_index.json", "nnunet_model_registry.json"):
        path = home_registry / name
        if path.is_file():
            try:
                rows[name] = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                rows[name] = {"error": "unreadable"}
    task_registry = ROOT / "nninteractive_task_models" / "registry.json"
    if task_registry.is_file():
        try:
            rows["nninteractive_task_registry.json"] = json.loads(
                task_registry.read_text(encoding="utf-8")
            )
        except Exception:
            rows["nninteractive_task_registry.json"] = {"error": "unreadable"}
    return rows


def collect_bundle(output_path: Path) -> Path:
    """Build the redacted diagnostics zip at ``output_path``."""
    files: dict[str, str] = {}
    for label, name in LOG_TARGETS:
        tail = _tail_text(ROOT / ".mimics_runtime" / name)
        if tail:
            files[label] = redact_text(tail)
    # Health panel + drop window logs live in per-launch files; take the
    # newest few.
    for subdir, prefix in (
        ("health_panel", "health_panel_"),
        ("drop_import", "drop_window_"),
    ):
        directory = ROOT / ".mimics_runtime" / subdir
        if not directory.is_dir():
            continue
        logs = sorted(
            (p for p in directory.glob(prefix + "*.log")),
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
            reverse=True,
        )[:2]
        for index, path in enumerate(logs):
            tail = _tail_text(path)
            if tail:
                files["logs/{}_{}_tail.log".format(subdir, index)] = redact_text(tail)

    for label, name in CONFIG_SNAPSHOTS:
        path = ROOT / name
        if path.is_file():
            files[label] = redact_text(
                path.read_text(encoding="utf-8", errors="replace")
            )

    files["diagnostics/runtime_census.json"] = redact_text(
        json.dumps(_runtime_census(), indent=2, ensure_ascii=False)
    )
    files["diagnostics/environment.json"] = redact_text(
        json.dumps(_environment_info(), indent=2, ensure_ascii=False)
    )
    registry_rows = _collect_registry_rows()
    if registry_rows:
        files["diagnostics/model_registries.json"] = redact_text(
            json.dumps(registry_rows, indent=2, ensure_ascii=False)
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        str(output_path), "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for name, content in sorted(files.items()):
            archive.writestr(name, content)
    # Hard cap: refuse to ship oversized bundles.
    if output_path.stat().st_size > BUNDLE_MAX_BYTES:
        output_path.unlink()
        raise RuntimeError(
            "Diagnostics bundle exceeded {} MB; trim logs and retry.".format(
                BUNDLE_MAX_BYTES // (1024 * 1024)
            )
        )
    return output_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=None,
        help="Output zip path (default: <root>/diagnostics_YYYYmmddTHHMMSS.zip)",
    )
    args = parser.parse_args(argv)
    output = (
        Path(args.output)
        if args.output
        else ROOT / "diagnostics_{}.zip".format(time.strftime("%Y%m%dT%H%M%S"))
    )
    path = collect_bundle(output)
    print("Diagnostics bundle written: {}".format(path))
    print("Size: {:.1f} KB".format(path.stat().st_size / 1024))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
