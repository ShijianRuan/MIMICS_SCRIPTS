#!/usr/bin/env python3
"""Collect a redacted diagnostics bundle for support requests.

Aggregates the tail of every operational log, the newest job directories'
status and logs, a census of .mimics_runtime state (processes, locks,
queues, servers), config snapshots, environment versions, and GPU status
into a single zip the annotator can hand over.

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
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

LOG_TAIL_LINES = 200
JOB_TAIL_LINES = 200
BUNDLE_MAX_BYTES = 10 * 1024 * 1024
MAX_JOBS_PER_KIND = 5

# Fixed-name operational logs under the project runtime root. All of these
# are written by production code; missing files are skipped.
RUNTIME_LOG_TARGETS = [
    ("logs/setup_env_tail.log", ".mimics_runtime/setup_env.log"),
    ("logs/mimics_export_tail.log", ".mimics_runtime/mimics_export.log"),
    ("logs/nninteractive_mimics_tail.log",
     ".mimics_runtime/nninteractive/logs/nninteractive_mimics.log"),
    ("logs/nninteractive_bridge_tail.log",
     ".mimics_runtime/nninteractive/logs/nninteractive_bridge.jsonl"),
]

# Logs of external windows and drop-import batches that live in per-launch
# files: (archive label prefix, runtime subdir, filename glob, newest count).
SCAN_LOG_TARGETS = [
    ("logs/health_panel", ".mimics_runtime/health_panel", "health_panel_*.log", 2),
    ("logs/drop_window", ".mimics_runtime/drop_import", "drop_window_*.log", 2),
    ("logs/drop_batch", ".mimics_runtime/drop_import", "*_batch.log", 2),
    ("logs/batch_status", ".mimics_runtime/batch_status", "batch_status_*.log", 2),
    # nnU-Net / FlexiCT external-window stderr logs live under the
    # respective workspace setup folder.
    ("logs/flexict_stderr", "flexict_models/setup", "*_stderr.log", 2),
    ("logs/nnunet_stderr", "~/.mimics_script/nnunet/setup", "*_stderr.log", 2),
]

# Job roots: (label, directory pattern). Each contains <job>/status.json.
# "~" and config-derived paths are resolved at collect time.
JOB_ROOTS = [
    ("flexict", "flexict_models/jobs"),
    ("nnunet", "~/.mimics_script/nnunet/jobs"),
    ("nninteractive_tasks", "nninteractive_task_models/tasks/*/jobs"),
    ("export_jobs", ".mimics_runtime/export_jobs"),
    ("import_runs", ".mimics_runtime/import_runs"),
]

CONFIG_SNAPSHOTS = [
    ("configs/nninteractive_config.json", "nninteractive_config.json"),
    ("configs/nninteractive_finetune_config.json", "nninteractive_finetune_config.json"),
    ("configs/mimics_io_config.json", "mimics_io_config.json"),
    ("configs/flexict_config.json", "flexict_config.json"),
    ("configs/interactive_algorithms_config.json", "interactive_algorithms_config.json"),
]

_RUNTIME_DIRS = ["processes", "locks", "import_queues", "drop_import", "export_jobs", "append_jobs"]


LOG_TAIL_MAX_BYTES = 2 * 1024 * 1024

TERMINAL_JOB_STATES = {"completed", "failed", "cancelled", "abandoned"}
FAILED_JOB_STATES = {"failed", "cancelled", "abandoned"}


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


def redact_path(match: "re.Match[str]") -> str:
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


def _runtime_census() -> dict[str, Any]:
    census: dict[str, Any] = {"collected_at_epoch": time.time()}
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


def _environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {"python_version": sys.version}
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


def _collect_registry_rows() -> dict[str, Any]:
    rows = {}
    home_registry = Path.home() / ".mimics_script"
    for name in ("nnunet_model_registry.json",):
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
    # FlexiCT model registry lives in its own workspace folder.
    try:
        from tools.flexict_common import workspace_root

        flexict_registry = workspace_root() / "registry.json"
        if flexict_registry.is_file():
            rows["flexict_registry.json"] = json.loads(
                flexict_registry.read_text(encoding="utf-8")
            )
    except Exception:
        pass
    return rows


def _resolve_runtime_pattern(pattern: str) -> Path:
    """Resolve a JOB_ROOTS / SCAN_LOG_TARGETS path pattern against the
    project ROOT (~ expands to the user home)."""
    expanded = Path(os.path.expandvars(pattern)).expanduser()
    if not expanded.is_absolute():
        expanded = ROOT / expanded
    return expanded


def _discover_job_dirs() -> list[dict[str, Any]]:
    """Find job directories with a status.json across all job roots.

    Returns rows sorted by status mtime (newest first):
    {label, dir, status_path, status, phase, updated_at}.
    """
    rows: list[dict[str, Any]] = []
    for label, pattern in JOB_ROOTS:
        base = _resolve_runtime_pattern(pattern)
        if not base.is_dir():
            continue
        # A pattern with a glob (nninteractive tasks/<task>/jobs) resolves to
        # the parent of the job roots; plain patterns resolve to the job
        # root itself.
        job_dirs = []
        if "*" in pattern:
            for child in sorted(base.glob("*/jobs")):
                job_dirs.extend(
                    grand for grand in child.iterdir()
                    if (grand / "status.json").is_file()
                )
        else:
            job_dirs = [
                child for child in base.iterdir() if (child / "status.json").is_file()
            ]
        for job_dir in job_dirs:
            status_path = job_dir / "status.json"
            try:
                payload = json.loads(status_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
            rows.append(
                {
                    "label": label,
                    "dir": job_dir,
                    "status_path": status_path,
                    "status": str(payload.get("status") or ""),
                    "phase": str(payload.get("phase") or ""),
                    "updated_at": float(
                        payload.get("updated_at_epoch")
                        or status_path.stat().st_mtime
                    ),
                }
            )
    rows.sort(key=lambda row: row["updated_at"], reverse=True)
    return rows


def _job_log_candidates(job_dir: Path) -> list[Path]:
    """Logs that commonly sit next to a job status.json, newest-first."""
    candidates = [
        job_dir / "job.log",
        job_dir / "trainer.log",
        job_dir / "process.log",
        job_dir / "remote_worker.log",
    ]
    return [path for path in candidates if path.is_file()]


def _collect_job_snapshots(
    files: dict[str, str],
    focus_latest_failure: bool = False,
) -> None:
    """Add status.json + log tails for recent jobs.

    Always includes the newest MAX_JOBS_PER_KIND non-terminal jobs per kind
    (annotators call support about a stuck task) and the newest
    MAX_JOBS_PER_KIND failed jobs per kind. With focus_latest_failure the
    selection narrows to the single most recent failed job plus its logs.
    """
    discovered = _discover_job_dirs()
    if focus_latest_failure:
        failed = [row for row in discovered if row["status"] in FAILED_JOB_STATES]
        selected = failed[:1]
    else:
        selected = []
        for label in {row["label"] for row in discovered}:
            per_kind = [row for row in discovered if row["label"] == label]
            active = [
                row for row in per_kind
                if row["status"] and row["status"] not in TERMINAL_JOB_STATES
            ]
            failed = [
                row for row in per_kind if row["status"] in FAILED_JOB_STATES
            ]
            selected.extend(active[:MAX_JOBS_PER_KIND])
            selected.extend(failed[:MAX_JOBS_PER_KIND])
    for row in selected:
        prefix = "jobs/{0}/{1}".format(row["label"], row["dir"].name)
        try:
            status_text = row["status_path"].read_text(encoding="utf-8", errors="replace")
        except OSError:
            status_text = ""
        if status_text:
            files[prefix + "/status.json"] = redact_text(status_text)
        for log_path in _job_log_candidates(row["dir"]):
            tail = _tail_text(log_path, JOB_TAIL_LINES)
            if tail:
                files[prefix + "/" + log_path.name] = redact_text(tail)


def _collect_import_logs(files: dict[str, str]) -> None:
    """Import logs live next to each .mcs output folder, not in the runtime
    root. Discover the output folders through the per-queue runner scripts
    and take the newest few mimics_import.log tails."""
    queues = ROOT / ".mimics_runtime" / "import_queues"
    if not queues.is_dir():
        return
    pattern = re.compile(r"create_mcs_batch\.main\(\"(.*?)\"")
    output_dirs: list[tuple[float, Path]] = []
    for control in queues.iterdir():
        runner = control / "_run_create_mcs.py"
        if not runner.is_file():
            continue
        try:
            match = pattern.search(runner.read_text(encoding="utf-8", errors="replace"))
            mtime = runner.stat().st_mtime
        except OSError:
            continue
        if match:
            output_dirs.append((mtime, Path(match.group(1).replace("\\\\", "\\"))))
        # The per-queue background Mimics log is useful even when the output
        # folder is gone.
        bg_log = control / "_background_mimics.log"
        if bg_log.is_file():
            tail = _tail_text(bg_log)
            if tail:
                files["logs/import_queues/{}/bg_tail.log".format(control.name)] = (
                    redact_text(tail)
                )
    output_dirs.sort(reverse=True)
    for index, (_mtime, output_dir) in enumerate(output_dirs[:3]):
        tail = _tail_text(output_dir / "logs" / "mimics_import.log")
        if tail:
            files["logs/mimics_import_{}_tail.log".format(index)] = redact_text(tail)


def collect_bundle(output_path: Path, focus_latest_failure: bool = False) -> Path:
    """Build the redacted diagnostics zip at ``output_path``."""
    files: dict[str, str] = {}
    for label, relative in RUNTIME_LOG_TARGETS:
        tail = _tail_text(ROOT / relative)
        if tail:
            files[label] = redact_text(tail)
    # Health panel / drop window / external-window stderr logs live in
    # per-launch files; take the newest few of each.
    for label_prefix, subdir_pattern, pattern, count in SCAN_LOG_TARGETS:
        directory = _resolve_runtime_pattern(subdir_pattern)
        if not directory.is_dir():
            continue
        try:
            logs = sorted(
                (p for p in directory.glob(pattern) if p.is_file()),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )[:count]
        except OSError:
            continue
        for index, path in enumerate(logs):
            tail = _tail_text(path)
            if tail:
                files["{}_{}_tail.log".format(label_prefix, index)] = redact_text(tail)

    _collect_import_logs(files)
    _collect_job_snapshots(files, focus_latest_failure=focus_latest_failure)

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
    parser.add_argument(
        "--focus-latest-failure",
        action="store_true",
        help="Narrow job collection to the single most recent failed job "
        "(plus its logs) instead of the newest active/failed jobs per kind.",
    )
    args = parser.parse_args(argv)
    output = (
        Path(args.output)
        if args.output
        else ROOT / "diagnostics_{}.zip".format(time.strftime("%Y%m%dT%H%M%S"))
    )
    path = collect_bundle(output, focus_latest_failure=args.focus_latest_failure)
    print("Diagnostics bundle written: {}".format(path))
    print("Size: {:.1f} KB".format(path.stat().st_size / 1024))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
