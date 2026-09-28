#!/usr/bin/env python3
"""Local controller for optional SSH/Docker AI training jobs.

The controller runs outside Mimics. It prepares source-grid NIfTI inputs with
the existing local code, uploads one job archive, mirrors remote status into
the existing local status file, downloads a verified model, and registers that
model through the same local registry used by local training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import traceback
import uuid
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "tools"):
    text = str(candidate)
    if text not in sys.path:
        sys.path.insert(0, text)

try:
    from tools.remote_compute import (  # noqa: E402
        RemoteCommandError,
        RemoteComputeError,
        SSHSession,
        container_runtime_command,
        docker_gpu_request,
        get_profile,
        read_json,
        safe_identifier,
        write_json_atomic,
    )
except ImportError:
    from remote_compute import (  # noqa: E402
        RemoteCommandError,
        RemoteComputeError,
        SSHSession,
        assert_gpus_not_busy,
        container_runtime_command,
        docker_gpu_request,
        get_profile,
        read_json,
        safe_identifier,
        write_json_atomic,
    )


TERMINAL = {"completed", "failed", "cancelled", "paused", "abandoned"}
LOCAL_ARCHIVE_REVERIFY_SECONDS = 7 * 24 * 60 * 60
LOCAL_ARCHIVE_RETENTION_SECONDS = 30 * 24 * 60 * 60
# Bounded reconnect budget: with the exponential backoff capped at 30 s this
# is roughly fifteen minutes of unreachable server before the task moves to
# remote_unreachable and the user picks Re-attach (retry) or Abandon Locally.
RECONNECT_MAX_ATTEMPTS = 30
REMOTE_STALL_SECONDS = 24 * 60 * 60
REMOTE_STALL_HINT = (
    "The remote container is still running but has produced no status or log "
    "updates for more than 24 hours. The training may be stuck; inspect the "
    "log tab or stop the task if this looks wrong."
)
REMOTE_TRAINING_LABEL = "mimics-script.remote-training"
REMOTE_OWNER_LABEL = "mimics-script.owner"
REMOTE_JOB_LABEL = "mimics-script.job"
LOCAL_STATUS_KEYS = {
    "abandon_path",
    "cancel_path",
    "control_path",
    "controller_pid",
    "created_at_epoch",
    "execution_backend",
    "job_id",
    "kind",
    "launcher_pid",
    "local_log_path",
    "local_dataset_archive_cache_reused",
    "local_dataset_archive_cache_total",
    "local_materialization_cache_reused",
    "local_materialization_cache_total",
    "controller_log_path",
    "diagnostic_log_paths",
    "log_path",
    "organ",
    "profile_name",
    "remote_container_name",
    "remote_control_kind",
    "remote_control_path",
    "remote_job_dir",
    "remote_profile_id",
    "request_path",
    "retry_context",
    "schema_version",
    "task_id",
    "task_name",
    "training_options",
    "train_log",
    "training_curve_path",
    "source_grid_cache_reused",
    "source_grid_cache_total",
    "ts_root",
    "workspace",
}


class RemoteTaskAbandoned(RuntimeError):
    """Local monitoring ended without claiming that the remote job stopped."""


class RemoteServerUnreachable(RuntimeError):
    """The reconnect budget ran out; the remote task state is unknown.

    Raised instead of reconnecting forever so the task lands in a visible
    non-terminal state (remote_unreachable) with an explicit escape path
    (retry re-attach or Abandon Locally) rather than silently waiting.
    The remote container itself is untouched: it may still be training.
    """


def _status_update(path: Path, **values: Any) -> dict[str, Any]:
    payload = read_json(path, {}) or {}
    payload.update(values)
    payload["updated_at_epoch"] = time.time()
    write_json_atomic(path, payload)
    return payload


def _append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write(
            "[{}] {}\n".format(
                time.strftime("%Y-%m-%d %H:%M:%S"), str(message).rstrip()
            )
        )


def _split_csv(value: object) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [
        item.strip()
        for item in str(value or "").replace(";", ",").split(",")
        if item.strip()
    ]


def _cancel_requested(status_path: Path) -> bool:
    status = read_json(status_path, {}) or {}
    cancel_path = str(status.get("cancel_path") or "")
    if cancel_path and Path(cancel_path).is_file():
        return True
    control_path = str(status.get("control_path") or "")
    control = read_json(control_path, {}) if control_path else {}
    return str((control or {}).get("action") or "").lower() in {
        "stop",
        "cancel",
    }


def _abandon_path(status_path: Path) -> Path:
    status = read_json(status_path, {}) or {}
    configured = str(status.get("abandon_path") or "").strip()
    return (
        Path(configured).expanduser().resolve()
        if configured
        else status_path.with_name(status_path.name + ".abandon.request")
    )


def _abandon_requested(status_path: Path) -> bool:
    if _abandon_path(status_path).is_file():
        return True
    status = read_json(status_path, {}) or {}
    return bool(status.get("local_abandon_requested"))


def _raise_if_abandoned(status_path: Path) -> None:
    if _abandon_requested(status_path):
        raise RemoteTaskAbandoned("local abandon requested")


def _raise_if_cancelled(status_path: Path) -> None:
    _raise_if_abandoned(status_path)
    if _cancel_requested(status_path):
        raise InterruptedError("cancel")


def _pause_requested(status_path: Path) -> bool:
    status = read_json(status_path, {}) or {}
    control_path = str(status.get("control_path") or "")
    control = read_json(control_path, {}) if control_path else {}
    return str((control or {}).get("action") or "").lower() == "pause"


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with tarfile.open(str(archive), "r") as handle:
        for member in handle.getmembers():
            if member.issym() or member.islnk():
                raise RuntimeError(
                    "Remote result archive contains a link, which is not "
                    "accepted: {}".format(member.name)
                )
            if not (member.isfile() or member.isdir()):
                raise RuntimeError(
                    "Remote result archive contains an unsupported entry: "
                    "{}".format(member.name)
                )
            target = (destination / member.name).resolve()
            try:
                target.relative_to(destination_root)
            except ValueError:
                raise RuntimeError(
                    "Remote result archive contains an unsafe path: {}".format(
                        member.name
                    )
                )
        try:
            handle.extractall(str(destination), filter="fully_trusted")
        except TypeError:
            # Python 3.10 does not expose tarfile extraction filters. Every
            # member has already been type-checked and confined above.
            handle.extractall(str(destination))


def _normalized_tar_info(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    # The archive digest is the data identity used for both transfer and
    # prepared-data cache namespaces. Normalise every timestamp so a metadata
    # touch does not force an upload or a full remote preparation. Actual file
    # content changes alter the archive digest and get a new namespace below.
    info.mtime = 0
    return info


def _build_tar(
    source: Path,
    archive: Path,
    *,
    excluded_top_level: set[str] | None = None,
) -> int:
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(archive.name + ".tmp")
    excluded = set(excluded_top_level or ())
    try:
        with tarfile.open(str(temporary), "w") as handle:
            for path in sorted(source.rglob("*")):
                relative = path.relative_to(source)
                if relative.parts and relative.parts[0] in excluded:
                    continue
                handle.add(
                    str(path),
                    arcname=str(relative),
                    recursive=False,
                    filter=_normalized_tar_info,
                )
        os.replace(str(temporary), str(archive))
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass
    return int(archive.stat().st_size)


def _build_dataset_tar(bundle: Path, archive: Path) -> tuple[str, int]:
    included = {
        name for name in ("input", "labels") if (bundle / name).exists()
    }
    if not included:
        return "", 0
    size = _build_tar(
        bundle,
        archive,
        excluded_top_level={
            path.name for path in bundle.iterdir() if path.name not in included
        },
    )
    return _sha256_file(archive), size


def _build_dataset_parts(
    bundle: Path,
    destination: Path,
    *,
    cache_dir: Path | None = None,
    case_cache_keys: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Build or reuse deterministic per-case archives.

    ``case_cache_keys`` must describe the already materialized image, label,
    and optional Initial Mask. A matching local archive is immutable and its
    recorded SHA-256 can be reused without rereading the volume bytes.
    """
    if cache_dir is not None:
        cache_root = Path(cache_dir)
        cutoff = time.time() - LOCAL_ARCHIVE_RETENTION_SECONDS
        if cache_root.is_dir():
            for stale_archive in cache_root.rglob("*.tar"):
                try:
                    stale_metadata = stale_archive.with_suffix(".json")
                    metadata = read_json(stale_metadata, {}) or {}
                    last_used = float(
                        metadata.get("last_used_at_epoch")
                        or metadata.get("created_at_epoch")
                        or stale_archive.stat().st_mtime
                    )
                    if last_used >= cutoff:
                        continue
                    stale_archive.unlink()
                    try:
                        stale_metadata.unlink()
                    except OSError:
                        pass
                except OSError:
                    pass
    case_names = set()
    for top_name in ("input", "labels"):
        top = bundle / top_name
        if not top.is_dir():
            continue
        case_names.update(
            child.name for child in top.iterdir() if child.is_dir()
        )
    if not case_names:
        archive = destination / "dataset_shared.tar"
        fingerprint, size = _build_dataset_tar(bundle, archive)
        return (
            [{
                "case_id": "shared",
                "archive": archive,
                "fingerprint": fingerprint,
                "size": size,
            }]
            if fingerprint
            else []
        )
    destination.mkdir(parents=True, exist_ok=True)
    parts = []
    cache_keys = case_cache_keys or {}
    for case_name in sorted(case_names):
        source_key = str(cache_keys.get(case_name) or "").strip()
        cache_identity = (
            hashlib.sha256(
                (
                    "remote_dataset_case_archive.v1\0" + source_key
                ).encode("utf-8")
            ).hexdigest()
            if source_key
            else ""
        )
        metadata_path = None
        if cache_dir is not None and cache_identity:
            case_cache = (
                Path(cache_dir)
                / safe_identifier(case_name, "case")
            )
            archive = case_cache / (cache_identity + ".tar")
            metadata_path = case_cache / (cache_identity + ".json")
            metadata = read_json(metadata_path, {}) or {}
            cache_valid = (
                archive.is_file()
                and metadata.get("cache_identity") == cache_identity
                and int(metadata.get("size") or -1) == archive.stat().st_size
                and str(metadata.get("fingerprint") or "")
            )
            if cache_valid:
                archive_stat = archive.stat()
                last_verified = float(metadata.get("verified_at_epoch") or 0)
                recorded_mtime = int(metadata.get("archive_mtime_ns") or -1)
                needs_reverify = (
                    recorded_mtime
                    != int(
                        getattr(
                            archive_stat,
                            "st_mtime_ns",
                            archive_stat.st_mtime * 1e9,
                        )
                    )
                    or time.time() - last_verified
                    >= LOCAL_ARCHIVE_REVERIFY_SECONDS
                )
                if needs_reverify:
                    cache_valid = (
                        _sha256_file(archive)
                        == str(metadata.get("fingerprint") or "")
                    )
                    if cache_valid:
                        metadata["verified_at_epoch"] = time.time()
                        metadata["archive_mtime_ns"] = int(
                            getattr(
                                archive_stat,
                                "st_mtime_ns",
                                archive_stat.st_mtime * 1e9,
                            )
                        )
            if cache_valid:
                metadata["last_used_at_epoch"] = time.time()
                write_json_atomic(metadata_path, metadata)
                parts.append(
                    {
                        "case_id": case_name,
                        "archive": archive,
                        "fingerprint": str(metadata["fingerprint"]),
                        "size": int(metadata["size"]),
                        "local_cache_hit": True,
                    }
                )
                continue
            case_cache.mkdir(parents=True, exist_ok=True)
        else:
            archive = destination / (
                "dataset_{}.tar".format(safe_identifier(case_name, "case"))
            )
        temporary = archive.with_name(
            "{}.{}.tmp".format(archive.name, uuid.uuid4().hex)
        )
        try:
            with tarfile.open(str(temporary), "w") as handle:
                for top_name in ("input", "labels"):
                    source = bundle / top_name / case_name
                    if not source.exists():
                        continue
                    paths = [source]
                    if source.is_dir():
                        paths.extend(sorted(source.rglob("*")))
                    for path in paths:
                        handle.add(
                            str(path),
                            arcname=str(path.relative_to(bundle)),
                            recursive=False,
                            filter=_normalized_tar_info,
                        )
            os.replace(str(temporary), str(archive))
        finally:
            try:
                temporary.unlink()
            except OSError:
                pass
        fingerprint = _sha256_file(archive)
        size = int(archive.stat().st_size)
        if metadata_path is not None:
            archive_stat = archive.stat()
            write_json_atomic(
                metadata_path,
                {
                    "schema_version": "remote_dataset_case_archive.v1",
                    "case_id": case_name,
                    "cache_identity": cache_identity,
                    "source_key": source_key,
                    "fingerprint": fingerprint,
                    "size": size,
                    "created_at_epoch": time.time(),
                    "verified_at_epoch": time.time(),
                    "last_used_at_epoch": time.time(),
                    "archive_mtime_ns": int(
                        getattr(
                            archive_stat,
                            "st_mtime_ns",
                            archive_stat.st_mtime * 1e9,
                        )
                    ),
                },
            )
        parts.append({
            "case_id": case_name,
            "archive": archive,
            "fingerprint": fingerprint,
            "size": size,
            "local_cache_hit": False,
        })
    return parts


def _dataset_parts_fingerprint(parts: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part["case_id"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(part["fingerprint"]).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest() if parts else ""


_REMOTE_DATASET_CACHE_TOKEN = "__MIMICS_REMOTE_DATASET_FINGERPRINT__"


def _bind_remote_prepared_cache(
    bundle: Path, dataset_fingerprint: str
) -> None:
    """Bind persistent prepared caches to the exact archived dataset bytes.

    The request is built before the per-case archives are hashed. Replacing
    this token afterwards reuses prepared tensors only when the selected
    image/label data is byte-identical, including same-size edits.
    """
    request_path = bundle / "remote_request.json"
    request = read_json(request_path, {}) or {}
    if not dataset_fingerprint:
        raise RuntimeError("Remote training data has no content fingerprint.")

    def replace(value: Any) -> Any:
        if isinstance(value, str):
            return value.replace(_REMOTE_DATASET_CACHE_TOKEN, dataset_fingerprint)
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        return value

    bound = replace(request)
    if bound == request:
        raise RuntimeError(
            "Remote training request has no prepared-cache identity placeholder."
        )
    write_json_atomic(request_path, bound)


def _copy_model_input(
    source_value: str, bundle: Path
) -> tuple[str, str, str]:
    """Stage a user-selected custom base model into the upload bundle.

    Returns (local path, in-container path, content fingerprint). The
    fingerprint lets the run() upload path consult the remote content cache
    first and skip the multi-gigabyte re-upload on a cache hit.
    """
    source = Path(source_value).expanduser().resolve()
    if not source.exists():
        raise RuntimeError("Selected base model does not exist: {}".format(source))
    destination = bundle / "custom_model"
    if source.is_dir():
        shutil.copytree(str(source), str(destination))
        fingerprint = _local_model_fingerprint(destination)
        if not fingerprint:
            raise RuntimeError(
                "Selected base model contains no supported weight files: "
                "{}".format(source)
            )
        return str(destination), "/job/custom_model", fingerprint
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / source.name
    shutil.copy2(str(source), str(target))
    return (
        str(target),
        "/job/custom_model/{}".format(source.name),
        _sha256_file(target),
    )


def _remote_custom_model_cache_path(
    remote_paths: dict[str, str], fingerprint: str
) -> tuple[str, str]:
    """Content-addressed remote cache slot for a custom base model.

    Custom base weights ride inside every job archive by default, so the same
    multi-gigabyte selection is re-uploaded for each training run. This cache
    follows the dataset cache pattern: one tar per content fingerprint with a
    `.verified` marker, under the per-user cache root, so a repeated
    selection uploads zero model bytes. The marker content matches the
    dataset cache protocol (digest, size, mtime) so recovery logic is shared.
    """
    root = PurePosixPath(remote_paths["models_cache"])
    return (
        str(root / (fingerprint + ".tar")),
        str(root / (fingerprint + ".verified")),
    )


def _ensure_remote_custom_model(
    session: SSHSession,
    remote_paths: dict[str, str],
    profile: dict[str, Any],
    local_source: Path,
    fingerprint: str,
    status_path: Path,
    controller_log: Path,
) -> bool:
    """Return True when the custom model is content-verified on the server.

    On a miss the model tar is uploaded into the cache slot; the caller then
    still ships it inside the job archive for this run, and every later run
    with the same weights hits the cache instead. A hit means the job tar can
    drop the model entirely: the container launch extracts the cached tar
    into the job folder.
    """
    remote_archive, remote_verified = _remote_custom_model_cache_path(
        remote_paths, fingerprint
    )
    session.ensure_directory(remote_paths["models_cache"])
    # Both shapes are packed into a deterministic tar: directory selections
    # keep their inner layout, a single checkpoint file keeps its basename —
    # extraction into /job/custom_model then restores exactly what the
    # pipeline request references. Normalized metadata keeps the tar bytes
    # (and therefore the digest) stable across machines and rebuilds.
    packed_archive = local_source.with_name("custom_model_cache.tar")
    if local_source.is_dir():
        _build_tar(local_source, packed_archive)
    else:
        with tarfile.open(str(packed_archive), "w") as handle:
            handle.add(
                str(local_source),
                arcname=local_source.name,
                recursive=False,
                filter=_normalized_tar_info,
            )
    local_size = int(packed_archive.stat().st_size)
    local_digest = _sha256_file(packed_archive)
    try:
        code, output = session.execute_result(
            "if test -f {archive} && test -f {verified} && "
            "read saved_digest saved_size saved_mtime < {verified} && "
            'test "$saved_digest" = {digest} && '
            'test "$saved_size" = {size} && '
            'test "$(stat -c %s {archive} 2>/dev/null)" = {size} && '
            'test "$saved_mtime" = "$(stat -c %Y {archive} 2>/dev/null)"; '
            'then printf hit; else printf miss; fi'.format(
                archive=shlex.quote(remote_archive),
                verified=shlex.quote(remote_verified),
                digest=shlex.quote(local_digest),
                size=local_size,
            ),
            timeout=120,
        )
        if code == 0 and output.strip() == "hit":
            _status_update(
                status_path,
                custom_model_cache_hit=True,
                custom_model_fingerprint=fingerprint,
            )
            return True
        if code == 0 and output.strip() == "miss":
            session.execute(
                "rm -f {archive} {verified}".format(
                    archive=shlex.quote(remote_archive),
                    verified=shlex.quote(remote_verified),
                ),
                check=False,
            )
        _status_update(
            status_path,
            status="uploading",
            phase="uploading_custom_base_model",
            custom_model_cache_hit=False,
            custom_model_fingerprint=fingerprint,
        )
        session.upload(
            packed_archive,
            remote_archive,
            callback=_upload_progress(
                status_path,
                phase="uploading_custom_base_model",
                start_percent=33,
                end_percent=33,
            ),
        )
        remote_digest = session.execute(
            "sha256sum {archive} | awk '{{print $1}}'".format(
                archive=shlex.quote(remote_archive)
            ),
            timeout=600,
        ).strip()
        if remote_digest != local_digest:
            session.execute(
                "rm -f {archive} {verified}".format(
                    archive=shlex.quote(remote_archive),
                    verified=shlex.quote(remote_verified),
                ),
                check=False,
            )
            raise RuntimeError(
                "The uploaded custom base-model archive failed SHA-256 "
                "verification."
            )
        session.execute(
            "tmp={verified}.$$.tmp; "
            "archive_mtime=$(stat -c %Y {archive}); "
            "printf '%s %s %s\n' {digest} {size} \"$archive_mtime\" > \"$tmp\" && "
            "mv -f \"$tmp\" {verified}".format(
                verified=shlex.quote(remote_verified),
                archive=shlex.quote(remote_archive),
                digest=shlex.quote(local_digest),
                size=local_size,
            )
        )
        _append_log(
            controller_log,
            "Custom base model cached remotely ({} bytes); future runs with "
            "the same weights will skip this upload.".format(local_size),
        )
        # This run still needs the model in the job archive: the cache slot
        # is only populated now, and the container launch does not yet know
        # to extract it. Only subsequent runs take the zero-upload path.
        return False
    finally:
        try:
            packed_archive.unlink()
        except OSError:
            pass


def _local_model_fingerprint(path: Path) -> str:
    """Content-only fingerprint of a model file or directory.

    Directory fingerprints aggregate the sorted content hashes of supported
    weight and model-configuration files. Relative paths are intentionally excluded:
    the remote server may install the same weights under a different
    directory layout (custom ``remote_root`` or mount point), and an
    unrelated extra file (backup, README) must not invalidate the identity
    check.  Symbolic links are skipped so the set matches GNU ``find``
    without ``-L`` on the remote side.
    """
    path = path.resolve()
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if not path.is_dir():
        return ""
    candidates = []
    allowed_names = {"config.json", "plans.json"}
    allowed_suffixes = {".pth", ".safetensors", ".bin"}
    for candidate in path.rglob("*"):
        if candidate.is_symlink():
            # GNU find without -L never treats a symlink as a file.
            continue
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(path)
        if len(relative.parts) > 4:
            continue
        if (
            candidate.name not in allowed_names
            and candidate.suffix.lower() not in allowed_suffixes
        ):
            continue
        candidates.append(candidate)
    if not candidates:
        return ""
    digests = []
    for candidate in candidates:
        file_digest = hashlib.sha256()
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                file_digest.update(chunk)
        digests.append(file_digest.hexdigest())
    aggregate = hashlib.sha256()
    for digest in sorted(digests):
        aggregate.update(digest.encode("ascii"))
        # GNU ``sort | sha256sum`` emits one newline-terminated digest per
        # file.  Use the identical byte stream locally; the previous NUL
        # separator made equal local/remote model directories fail strict
        # verification every time.
        aggregate.update(b"\n")
    return aggregate.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Code subset that participates in remote training/inference parity. The
# runtime image is built with ``COPY . /app``, so the container runs whatever
# tools/runtime/integration code it was built with; drift between the local
# checkout and the image silently trains with stale code.
CODE_FINGERPRINT_ROOTS = ("tools", "runtime_py35", "integrations")
CODE_FINGERPRINT_SUFFIXES = (".py", ".sh", ".json")
CODE_FINGERPRINT_EXCLUDED_DIRS = {
    "__pycache__", ".git", "node_modules", ".pytest_cache",
    "weights", "jobs", "models", "tests",
    # machine-local state that never ships in the image:
    ".claude", ".idea", ".vscode", "docs",
}


def _local_code_fingerprint(root: Path | None = None) -> str:
    """Content digest of the pipeline code that a remote image must match.

    Hashes every .py/.sh/.json file under the training-relevant roots,
    keyed by relative path so added/removed files change the digest. The
    same algorithm runs inside the container (see _remote_code_fingerprint),
    against /app, so both sides see identical relative paths.
    """
    base = Path(root) if root else ROOT
    lines = []
    for name in CODE_FINGERPRINT_ROOTS:
        directory = base / name
        if not directory.is_dir():
            continue
        for candidate in directory.rglob("*"):
            if not candidate.is_file() or candidate.is_symlink():
                continue
            relative = candidate.relative_to(base)
            if any(
                part in CODE_FINGERPRINT_EXCLUDED_DIRS
                for part in relative.parts[:-1]
            ):
                continue
            if candidate.suffix.lower() not in CODE_FINGERPRINT_SUFFIXES:
                continue
            # Sort by the POSIX relative path so the ordering matches the
            # container-side walk regardless of the host OS: Windows paths
            # sort by "\" (0x5C) while Linux sorts by "/" (0x2F), which
            # flips the order of names like "foo0.py" vs "food/x.py".
            lines.append(
                (relative.as_posix(), candidate)
            )
    lines.sort(key=lambda item: item[0])
    lines = [
        "{}  {}".format(_sha256_file(candidate), relative)
        for relative, candidate in lines
    ]
    aggregate = hashlib.sha256()
    aggregate.update("\n".join(lines).encode("utf-8"))
    return aggregate.hexdigest()


# Runs inside the container; mirrors _local_code_fingerprint against /app.
# The base is overridable via env var so the regression suite can execute
# this exact script against a fixture tree with a real python subprocess —
# the container itself never sets the variable.
_REMOTE_CODE_FINGERPRINT_SCRIPT = (
    "import hashlib,os,sys\n"
    "from pathlib import Path\n"
    "base=Path(os.environ.get('MIMICS_CODE_FINGERPRINT_BASE','/app'))\n"
    "roots=('tools','runtime_py35','integrations')\n"
    "suffixes=('.py','.sh','.json')\n"
    "excluded={'__pycache__','.git','node_modules','.pytest_cache',"
    "'weights','jobs','models','tests','.claude','.idea','.vscode','docs'}\n"
    "lines=[]\n"
    "for name in roots:\n"
    "    d=base/name\n"
    "    if not d.is_dir():\n"
    "        continue\n"
    "    for c in d.rglob('*'):\n"
    "        if not c.is_file() or c.is_symlink():\n"
    "            continue\n"
    "        rel=c.relative_to(base)\n"
    "        if any(p in excluded for p in rel.parts[:-1]):\n"
    "            continue\n"
    "        if c.suffix.lower() not in suffixes:\n"
    "            continue\n"
    "        h=hashlib.sha256()\n"
    "        with c.open('rb') as f:\n"
    "            for chunk in iter(lambda:f.read(4194304),b''):\n"
    "                h.update(chunk)\n"
    "        lines.append((rel.as_posix(),h.hexdigest()))\n"
    "lines.sort()\n"
    "lines=[d+'  '+r for r,d in lines]\n"
    "sys.stdout.write(hashlib.sha256('\\n'.join(lines).encode()).hexdigest())\n"
)


def _remote_code_fingerprint(session: SSHSession, image: str) -> str:
    """Digest of the pipeline code baked into the runtime image.

    Executes the same fingerprint algorithm inside a throwaway container
    (``--network none``, no volumes) so local and remote digests are directly
    comparable regardless of the server's checkout layout.
    """
    # NOTE: pass the script text itself (shlex-quoted). Wrapping it in
    # json.dumps first hands the container a quoted one-line string
    # literal, which evaluates to a string and prints nothing — the
    # command "succeeds" with empty output, so strict mode refused every
    # run and warn mode flagged drift on every run (a false alarm, never
    # a silent pass: an empty digest never equals the local digest).
    output = session.execute(
        "{runtime} run --rm --network none --entrypoint python "
        "{image} -c {script}".format(
            runtime=container_runtime_command(getattr(session, "profile", None)),
            image=shlex.quote(image),
            script=shlex.quote(_REMOTE_CODE_FINGERPRINT_SCRIPT),
        ),
        timeout=300,
    )
    return output.strip()


def _check_code_drift(
    session: SSHSession,
    profile: dict[str, Any],
    status_path: Path,
    runtime_image_id: str,
) -> dict[str, Any]:
    """Compare local pipeline code with the image before training.

    strict: refuse to run on drift (the image would silently train with
    stale code). warn: record the drift in status. off: skip.
    """
    mode = str(profile.get("remote_code_verify") or "warn").strip().lower()
    if mode == "off":
        return {"remote_code_verify": "off"}
    local_digest = _local_code_fingerprint()
    remote_digest = _remote_code_fingerprint(
        session, str(profile["runtime_image"])
    )
    identity = {
        "remote_code_verify": mode,
        "local_code_sha256": local_digest,
        "remote_code_sha256": remote_digest,
        "remote_runtime_image_id": runtime_image_id,
    }
    if local_digest == remote_digest:
        identity["remote_code_match"] = True
        return identity
    detail = (
        "The remote runtime image does not contain the current pipeline "
        "code. Local code SHA-256: {local}; image code SHA-256: {remote}. "
        "Ask the server administrator to rebuild the image "
        "(`remote/setup_remote_server.sh --build`) before training.".format(
            local=local_digest, remote=remote_digest
        )
    )
    if mode == "strict":
        raise RuntimeError(detail)
    identity.update(
        {
            "remote_code_match": False,
            "remote_code_drift": True,
            "remote_code_drift_detail": detail,
        }
    )
    _status_update(status_path, **identity)
    return identity


def _link_or_copy(source: str | Path, destination: str | Path) -> str:
    """Stage an immutable cached file without rereading it when possible."""
    source_path = Path(source).resolve()
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(str(source_path), str(destination_path))
        return "hardlink"
    except OSError:
        shutil.copy2(str(source_path), str(destination_path))
        return "copy"




def _prepare_nninteractive(
    spec: dict[str, Any], bundle: Path
) -> dict[str, Any]:
    import tools.nninteractive_finetune_pipeline as pipeline

    local_job_dir = Path(spec["job_dir"]).resolve()
    status_path = local_job_dir / "status.json"
    request = read_json(local_job_dir / "request.json", {}) or {}
    if not request:
        raise RuntimeError("Local nnInteractive training request is missing.")
    manifest_path, _validation_path = pipeline._prepare_manifest(
        request,
        local_job_dir,
        status_path,
        local_job_dir / "control.json",
        local_job_dir / "job.log",
    )
    manifest = read_json(manifest_path, {}) or {}
    rows = manifest.get("cases") or []
    if not rows:
        raise RuntimeError("No prepared nnInteractive case is available.")
    remote_rows = []
    dataset_case_cache_keys: dict[str, str] = {}
    input_root = bundle / "input"
    for index, row in enumerate(rows):
        if _cancel_requested(status_path):
            raise InterruptedError("cancel")
        case_id = safe_identifier(row.get("case_id"), "case")
        case_root = input_root / case_id
        case_root.mkdir(parents=True, exist_ok=True)
        image_dst = case_root / "image.nii.gz"
        label_dst = case_root / "label.nii.gz"
        _link_or_copy(row["image"], image_dst)
        _link_or_copy(row["label"], label_dst)
        dataset_case_cache_keys[case_id] = str(
            row.get("source_grid_cache_fingerprint") or ""
        )
        remote_row = {
            "case_id": str(row.get("case_id") or case_id),
            "image": "/job/input/{}/image.nii.gz".format(case_id),
            "label": "/job/input/{}/label.nii.gz".format(case_id),
            "initial_mask": "",
            "initial_mask_source_type": str(
                row.get("initial_mask_source_type") or "none"
            ),
            "initial_mask_source_model": str(
                row.get("initial_mask_source_model") or ""
            ),
            "initial_mask_source_name": str(
                row.get("initial_mask_source_name") or ""
            ),
            "split": str(row.get("split") or "train"),
            "state": "ready",
        }
        initial_mask = str(row.get("initial_mask") or "").strip()
        if initial_mask:
            initial_dst = case_root / "initial_mask.nii.gz"
            _link_or_copy(initial_mask, initial_dst)
            remote_row["initial_mask"] = (
                "/job/input/{}/initial_mask.nii.gz".format(case_id)
            )
        remote_rows.append(remote_row)
        _status_update(
            status_path,
            status="preparing_remote",
            phase="preparing_remote_data",
            preparation_index=index + 1,
            preparation_total=len(rows),
            progress_percent=10
            + int(10.0 * (index + 1) / max(1, len(rows))),
        )

    remote_request = dict(request)
    remote_request["workspace"] = (
        "/remote-cache/nninteractive/datasets/{}".format(
            _REMOTE_DATASET_CACHE_TOKEN
        )
    )
    remote_request["prepared_cache_namespace"] = "dataset"
    remote_request["source_mode"] = "prepared"
    remote_request["prepared_root"] = "/job/input"
    remote_request["image_root"] = "/job/input"
    remote_request["label_root"] = "/job/input"
    remote_request["mcs_dir"] = ""
    remote_request["cases"] = remote_rows
    if any(row.get("initial_mask") for row in remote_rows):
        remote_request["initial_mask_source"] = "exported_masks"
        remote_request["initial_mask_root"] = "/job/input"
        remote_request["initial_mask_names"] = ["initial_mask"]
    else:
        remote_request["initial_mask_source"] = "none"
        remote_request["initial_mask_root"] = ""
        remote_request["initial_mask_names"] = []
    remote_request["output_model_dir"] = "/job/model_output"
    if str(request.get("parent_model_id") or "official") == "official":
        remote_request[
            "base_model_dir"
        ] = "/models/nninteractive/nnInteractive_v1.0"
        required_model_relative = "nninteractive/nnInteractive_v1.0"
        local_required_model = str(
            Path(request["base_model_dir"]).expanduser().resolve()
        )
        custom_model_fingerprint = ""
    else:
        _local, remote_model, custom_model_fingerprint = _copy_model_input(
            str(request["base_model_dir"]), bundle
        )
        remote_request["base_model_dir"] = remote_model
        required_model_relative = ""
        local_required_model = ""
    remote_request["mimics_exe"] = ""
    remote_payload = {
        "schema_version": "mimics_remote_training_request.v1",
        "kind": "nninteractive_train",
        "job_id": str(request["job_id"]),
        "pipeline_request": remote_request,
        "remote_status_path": "/job/pipeline_job/status.json",
        "artifact_path": "/job/model_output",
        "created_at_epoch": time.time(),
    }
    write_json_atomic(bundle / "remote_request.json", remote_payload)
    return {
        "remote_status_relative": "pipeline_job/status.json",
        "remote_control_relative": "pipeline_job/control.json",
        "remote_control_kind": "json",
        "remote_artifact_relative": "model_output",
        "train_count": sum(row.get("split") == "train" for row in remote_rows),
        "validation_count": sum(row.get("split") == "val" for row in remote_rows),
        "required_model_relative": required_model_relative,
        "local_required_model": local_required_model,
        "custom_model_fingerprint": custom_model_fingerprint,
        "local_dataset_archive_cache": str(
            Path(
                request.get("workspace") or local_job_dir.parent
            ).expanduser().resolve()
            / "cache"
            / "remote_dataset_archives"
            / "nninteractive"
        ),
        "dataset_case_cache_keys": dataset_case_cache_keys,
    }


def _prepare_nnunet(
    spec: dict[str, Any], bundle: Path
) -> dict[str, Any]:
    """Prepare source-grid nnU-Net cases and a portable remote request."""
    import tools.nnunet_pipeline as pipeline
    from tools.nnunet_common import (
        normalize_request,
        read_json as read_nnunet_json,
        safe_identifier as nnunet_safe_identifier,
        write_json_atomic as write_nnunet_json,
    )

    local_job_dir = Path(spec["job_dir"]).resolve()
    status_path = Path(spec["status_path"]).resolve()
    control_path = local_job_dir / "control.json"
    request = normalize_request(
        read_nnunet_json(Path(spec["request_path"]).resolve(), {}) or {}
    )
    if request["operation"] != "train":
        raise RuntimeError("Remote nnU-Net training received a non-training request.")
    prepared_root = bundle / "prepared_local"
    rows = pipeline.prepare_source_grid_cases(
        request,
        prepared_root,
        status_path,
        control_path,
    )
    remote_rows = []
    dataset_case_cache_keys: dict[str, str] = {}
    for index, row in enumerate(rows, start=1):
        _raise_if_cancelled(status_path)
        case_id = nnunet_safe_identifier(row["case_id"], "case")
        image_dst = bundle / "input" / case_id / "image.nii.gz"
        label_dst = bundle / "labels" / case_id / "label.nii.gz"
        _link_or_copy(row["image"], image_dst)
        _link_or_copy(row["label"], label_dst)
        dataset_case_cache_keys[case_id] = str(row.get("fingerprint") or "")
        remote_rows.append(
            {
                "case_id": str(row["case_id"]),
                "image": "/job/input/{}/image.nii.gz".format(case_id),
                "label": "/job/labels/{}/label.nii.gz".format(case_id),
                "fingerprint": str(row.get("fingerprint") or ""),
            }
        )
        _status_update(
            status_path,
            status="preparing_remote",
            phase="preparing_remote_data",
            message="Preparing remote nnU-Net case {} of {}.".format(
                index, len(rows)
            ),
            preparation_index=index,
            preparation_total=len(rows),
            progress_percent=10 + int(10 * index / max(1, len(rows))),
        )

    remote_request = dict(request)
    remote_request["execution_backend"] = "remote"
    remote_request["label_source"] = "prepared"
    remote_request["prepared_cases"] = remote_rows
    remote_request["dataset_root"] = "/job/input"
    remote_request["label_root"] = "/job/labels"
    remote_request["mcs_dir"] = ""
    remote_request["workspace"] = "/job/output"
    remote_request["gpu_lock_managed_externally"] = True
    remote_request["gpu_id"] = ""
    remote_request["runtime_roots"] = {
        "raw": "/remote-cache/nnunet/{}/raw".format(
            _REMOTE_DATASET_CACHE_TOKEN
        ),
        "preprocessed": "/remote-cache/nnunet/{}/preprocessed".format(
            _REMOTE_DATASET_CACHE_TOKEN
        ),
        "results": "/job/output/runtime/nnUNet_results",
    }
    pretrained = str(request.get("pretrained_weights") or "").strip()
    custom_model_fingerprint = ""
    if pretrained:
        source = Path(pretrained).expanduser().resolve()
        if not source.is_file():
            raise RuntimeError(
                "Selected nnU-Net pretrained checkpoint does not exist: {}".format(
                    source
                )
            )
        target = bundle / "custom_model" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(source), str(target))
        custom_model_fingerprint = _sha256_file(target)
        remote_request["pretrained_weights"] = "/job/custom_model/{}".format(
            source.name
        )
    model_id = str(
        remote_request.get("model_id")
        or "nnunet_{}_{}".format(
            time.strftime("%Y%m%dT%H%M%S"), uuid.uuid4().hex[:8]
        )
    )
    remote_request["model_id"] = model_id
    task_slug = nnunet_safe_identifier(request["task_id"], "task")
    artifact = "output/models/{}/{}".format(task_slug, model_id)
    payload = {
        "schema_version": "mimics_remote_training_request.v1",
        "kind": "nnunet_train",
        "job_id": request["job_id"],
        "pipeline_request": remote_request,
        "remote_status_path": "/job/pipeline_job/status.json",
        "artifact_path": "/job/{}".format(artifact),
        "created_at_epoch": time.time(),
    }
    write_nnunet_json(bundle / "remote_request.json", payload)
    shutil.rmtree(str(prepared_root), ignore_errors=True)
    return {
        "remote_status_relative": "pipeline_job/status.json",
        "remote_control_relative": "pipeline_job/control.json",
        "remote_control_kind": "json",
        "remote_artifact_relative": artifact,
        "train_count": len(rows),
        "validation_count": max(
            0,
            int(round(len(rows) * float(request.get("validation_fraction") or 0))),
        ),
        "required_model_relative": "",
        "local_required_model": "",
        "custom_model_fingerprint": custom_model_fingerprint,
        "local_model_id": model_id,
        "local_dataset_archive_cache": str(
            Path(request["workspace"])
            / "cache"
            / "remote_dataset_archives"
            / "nnunet"
            / task_slug
        ),
        "dataset_case_cache_keys": dataset_case_cache_keys,
    }


def _prepare_nnunet_infer(
    spec: dict[str, Any], bundle: Path
) -> dict[str, Any]:
    from tools.nnunet_common import (
        normalize_request,
        path_signature,
        read_json as read_nnunet_json,
        safe_identifier as nnunet_safe_identifier,
        stable_digest,
        write_json_atomic as write_nnunet_json,
    )
    from tools.mimics_label_export import materialize_source_image as _materialize_source_image
    from tools.nnunet_pipeline import (
        validate_materialized_source_geometry,
        validate_model_input_compatibility,
    )

    request = normalize_request(
        read_nnunet_json(Path(spec["request_path"]).resolve(), {}) or {}
    )
    if request["operation"] != "infer":
        raise RuntimeError("Remote nnU-Net inference received a training request.")
    source_image = Path(str(request.get("image_path") or "")).expanduser().resolve()
    if not source_image.exists():
        raise RuntimeError("Prediction image does not exist: {}".format(source_image))
    case_id = nnunet_safe_identifier(request.get("case_id") or source_image.stem, "case")
    image_dst = bundle / "input" / case_id / "image.nii.gz"
    image_dst.parent.mkdir(parents=True, exist_ok=True)
    _materialize_source_image(source_image, image_dst)
    validate_materialized_source_geometry(
        image_dst, request.get("source_geometry_expected")
    )

    manifest_path = Path(str(request.get("model_manifest") or "")).expanduser().resolve()
    manifest = read_nnunet_json(manifest_path, {}) or {}
    model_dir = Path(str(manifest.get("model_dir") or manifest_path.parent)).expanduser().resolve()
    if not manifest_path.is_file() or not model_dir.is_dir():
        raise RuntimeError("Selected nnU-Net model is missing or invalid.")
    compatibility_warnings = validate_model_input_compatibility(
        image_dst, manifest, request
    )
    if compatibility_warnings:
        _status_update(
            Path(spec["status_path"]).resolve(),
            compatibility_warnings=compatibility_warnings,
        )
    model_bundle = bundle / "labels" / "model" / "model_dir"
    model_bundle.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(str(model_dir), str(model_bundle), copy_function=os.link)
    except OSError:
        shutil.rmtree(str(model_bundle), ignore_errors=True)
        shutil.copytree(str(model_dir), str(model_bundle))
    remote_manifest = model_bundle / manifest_path.name
    if not remote_manifest.is_file():
        shutil.copy2(str(manifest_path), str(remote_manifest))

    model_fingerprint = stable_digest(
        {
            "weights": _local_model_fingerprint(model_dir),
            "manifest": path_signature(manifest_path),
            "model_id": manifest.get("model_id"),
        }
    )
    image_fingerprint = stable_digest(
        {"source": path_signature(source_image), "contract": "nnunet_remote_infer_image.v1"}
    )
    remote_request = dict(request)
    remote_request["execution_backend"] = "remote"
    remote_request["gpu_lock_managed_externally"] = True
    remote_request["gpu_id"] = ""
    remote_request["workspace"] = "/job/output"
    remote_request["image_path"] = "/job/input/{}/image.nii.gz".format(case_id)
    remote_request["model_manifest"] = "/job/labels/model/model_dir/{}".format(
        remote_manifest.name
    )
    remote_request["model_dir_override"] = "/job/labels/model/model_dir"
    remote_request["output_path"] = "/job/output/prediction_bundle/prediction.nii.gz"
    model_dataset_id = int(
        manifest.get("dataset_id") or request.get("dataset_id") or 0
    )
    inference_cache_namespace = "Dataset{:03d}_{}".format(
        model_dataset_id, _REMOTE_DATASET_CACHE_TOKEN
    )
    remote_request["runtime_roots"] = {
        "raw": "/remote-cache/nnunet/inference/{}/raw".format(
            inference_cache_namespace
        ),
        "preprocessed": "/remote-cache/nnunet/inference/{}/preprocessed".format(
            inference_cache_namespace
        ),
        "results": "/remote-cache/nnunet/inference/{}/results".format(
            inference_cache_namespace
        ),
    }
    payload = {
        "schema_version": "mimics_remote_training_request.v1",
        "kind": "nnunet_infer",
        "job_id": request["job_id"],
        "pipeline_request": remote_request,
        "remote_status_path": "/job/pipeline_job/status.json",
        "artifact_path": "/job/output/prediction_bundle",
        "created_at_epoch": time.time(),
    }
    write_nnunet_json(bundle / "remote_request.json", payload)
    return {
        "remote_status_relative": "pipeline_job/status.json",
        "remote_control_relative": "pipeline_job/control.json",
        "remote_control_kind": "json",
        "remote_artifact_relative": "output/prediction_bundle",
        "train_count": 0,
        "validation_count": 0,
        "required_model_relative": "",
        "local_required_model": "",
        "local_dataset_archive_cache": str(
            Path(request["workspace"])
            / "cache"
            / "remote_inference_archives"
            / nnunet_safe_identifier(manifest.get("model_id"), "model")
        ),
        "dataset_case_cache_keys": {
            case_id: image_fingerprint,
            "model": model_fingerprint,
        },
    }


def _prepare_flexict(
    spec: dict[str, Any], bundle: Path
) -> dict[str, Any]:
    """Prepare source-grid FlexiCT cases and a portable remote request.

    Mirrors _prepare_nnunet: cases are materialized locally on the source
    grid, then shipped as per-case content-addressed archives. The FlexiCT
    differences are the dataset band (750-799), the model artifact layout
    (output/models/<task>/<model_id>, one folder per trained configuration
    with a shared pair_id), and the pretrained-backbone requirement under
    /models/flexict (fingerprint-verified by _validate_remote_assets).
    """
    import tools.flexict_pipeline as pipeline
    from tools.flexict_pipeline import normalize_flexict_request
    from tools.nnunet_common import (
        read_json as read_flexict_json,
        safe_identifier as flexict_safe_identifier,
        write_json_atomic as write_flexict_json,
    )

    local_job_dir = Path(spec["job_dir"]).resolve()
    status_path = Path(spec["status_path"]).resolve()
    control_path = local_job_dir / "control.json"
    request = normalize_flexict_request(
        read_flexict_json(Path(spec["request_path"]).resolve(), {}) or {}
    )
    if request["operation"] != "train":
        raise RuntimeError("Remote FlexiCT training received a non-training request.")
    prepared_root = bundle / "prepared_local"
    rows = pipeline.prepare_source_grid_cases(
        request,
        prepared_root,
        status_path,
        control_path,
    )
    remote_rows = []
    dataset_case_cache_keys: dict[str, str] = {}
    for index, row in enumerate(rows, start=1):
        _raise_if_cancelled(status_path)
        case_id = flexict_safe_identifier(row["case_id"], "case")
        image_dst = bundle / "input" / case_id / "image.nii.gz"
        label_dst = bundle / "labels" / case_id / "label.nii.gz"
        _link_or_copy(row["image"], image_dst)
        _link_or_copy(row["label"], label_dst)
        dataset_case_cache_keys[case_id] = str(row.get("fingerprint") or "")
        remote_rows.append(
            {
                "case_id": str(row["case_id"]),
                "image": "/job/input/{}/image.nii.gz".format(case_id),
                "label": "/job/labels/{}/label.nii.gz".format(case_id),
                "fingerprint": str(row.get("fingerprint") or ""),
            }
        )
        _status_update(
            status_path,
            status="preparing_remote",
            phase="preparing_remote_data",
            message="Preparing remote FlexiCT case {} of {}.".format(
                index, len(rows)
            ),
            preparation_index=index,
            preparation_total=len(rows),
            progress_percent=10 + int(10 * index / max(1, len(rows))),
        )

    remote_request = dict(request)
    remote_request["execution_backend"] = "remote"
    remote_request["label_source"] = "prepared"
    remote_request["prepared_cases"] = remote_rows
    remote_request["dataset_root"] = "/job/input"
    remote_request["label_root"] = "/job/labels"
    remote_request["mcs_dir"] = ""
    remote_request["workspace"] = "/job/output"
    remote_request["gpu_lock_managed_externally"] = True
    remote_request["gpu_id"] = ""
    # The container resolves the FlexiCT backbone weights from the read-only
    # /models mount (admin-installed, fingerprint-verified below).
    remote_request["flexict_pretrained_dir"] = "/models/flexict"
    remote_request["runtime_roots"] = {
        "raw": "/remote-cache/flexict/{}/raw".format(
            _REMOTE_DATASET_CACHE_TOKEN
        ),
        "preprocessed": "/remote-cache/flexict/{}/preprocessed".format(
            _REMOTE_DATASET_CACHE_TOKEN
        ),
        "results": "/job/output/runtime/nnUNet_results",
    }
    model_id = str(
        remote_request.get("model_id")
        or "flexict_{}_{}".format(
            time.strftime("%Y%m%dT%H%M%S"), uuid.uuid4().hex[:8]
        )
    )
    remote_request["model_id"] = model_id
    task_slug = flexict_safe_identifier(request["task_id"], "task")
    artifact = "output/models/{}/{}".format(task_slug, model_id)
    payload = {
        "schema_version": "mimics_remote_training_request.v1",
        "kind": "flexict",
        "job_id": request["job_id"],
        "pipeline_request": remote_request,
        "remote_status_path": "/job/pipeline_job/status.json",
        "artifact_path": "/job/{}".format(artifact),
        "created_at_epoch": time.time(),
    }
    write_flexict_json(bundle / "remote_request.json", payload)
    shutil.rmtree(str(prepared_root), ignore_errors=True)
    return {
        "remote_status_relative": "pipeline_job/status.json",
        "remote_control_relative": "pipeline_job/control.json",
        "remote_control_kind": "json",
        "remote_artifact_relative": artifact,
        "train_count": len(rows),
        "validation_count": 0,
        # The FlexiCT backbone pair under /models/flexict must match the
        # local weights (they initialise the fine-tune). Verified with the
        # same strict/warn fingerprint check as the nnInteractive base model.
        "required_model_relative": "flexict",
        "local_required_model": "",
        "local_model_id": model_id,
        "local_dataset_archive_cache": str(
            Path(request["workspace"])
            / "cache"
            / "remote_dataset_archives"
            / "flexict"
            / task_slug
        ),
        "dataset_case_cache_keys": dataset_case_cache_keys,
    }


def _prepare_flexict_infer(
    spec: dict[str, Any], bundle: Path
) -> dict[str, Any]:
    from tools.flexict_pipeline import normalize_flexict_request
    from tools.nnunet_common import (
        path_signature,
        read_json as read_flexict_json,
        safe_identifier as flexict_safe_identifier,
        stable_digest,
        write_json_atomic as write_flexict_json,
    )
    from tools.mimics_label_export import materialize_source_image as _materialize_source_image
    from tools.nnunet_pipeline import validate_materialized_source_geometry

    request = normalize_flexict_request(
        read_flexict_json(Path(spec["request_path"]).resolve(), {}) or {}
    )
    if request["operation"] != "infer":
        raise RuntimeError("Remote FlexiCT inference received a training request.")
    source_image = Path(str(request.get("image_path") or "")).expanduser().resolve()
    if not source_image.exists():
        raise RuntimeError("Prediction image does not exist: {}".format(source_image))
    case_id = flexict_safe_identifier(request.get("case_id") or source_image.stem, "case")
    image_dst = bundle / "input" / case_id / "image.nii.gz"
    image_dst.parent.mkdir(parents=True, exist_ok=True)
    _materialize_source_image(source_image, image_dst)
    validate_materialized_source_geometry(
        image_dst, request.get("source_geometry_expected")
    )

    manifest_path = Path(str(request.get("model_manifest") or "")).expanduser().resolve()
    manifest = read_flexict_json(manifest_path, {}) or {}
    model_dir = Path(str(manifest.get("model_dir") or manifest_path.parent)).expanduser().resolve()
    if not manifest_path.is_file() or not model_dir.is_dir():
        raise RuntimeError("Selected FlexiCT model is missing or invalid.")
    model_bundle = bundle / "labels" / "model" / "model_dir"
    model_bundle.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(str(model_dir), str(model_bundle), copy_function=os.link)
    except OSError:
        shutil.rmtree(str(model_bundle), ignore_errors=True)
        shutil.copytree(str(model_dir), str(model_bundle))
    remote_manifest = model_bundle / manifest_path.name
    if not remote_manifest.is_file():
        shutil.copy2(str(manifest_path), str(remote_manifest))

    model_fingerprint = stable_digest(
        {
            "weights": _local_model_fingerprint(model_dir),
            "manifest": path_signature(manifest_path),
            "model_id": manifest.get("model_id"),
        }
    )
    image_fingerprint = stable_digest(
        {"source": path_signature(source_image), "contract": "flexict_remote_infer_image.v1"}
    )
    remote_request = dict(request)
    remote_request["execution_backend"] = "remote"
    remote_request["gpu_lock_managed_externally"] = True
    remote_request["gpu_id"] = ""
    remote_request["workspace"] = "/job/output"
    remote_request["image_path"] = "/job/input/{}/image.nii.gz".format(case_id)
    remote_request["model_manifest"] = "/job/labels/model/model_dir/{}".format(
        remote_manifest.name
    )
    remote_request["model_dir_override"] = "/job/labels/model/model_dir"
    remote_request["output_path"] = "/job/output/prediction_bundle/prediction.nii.gz"
    remote_request["runtime_roots"] = {
        "raw": "/remote-cache/flexict/inference/{}/raw".format(
            _REMOTE_DATASET_CACHE_TOKEN
        ),
        "preprocessed": "/remote-cache/flexict/inference/{}/preprocessed".format(
            _REMOTE_DATASET_CACHE_TOKEN
        ),
        "results": "/remote-cache/flexict/inference/{}/results".format(
            _REMOTE_DATASET_CACHE_TOKEN
        ),
    }
    payload = {
        "schema_version": "mimics_remote_training_request.v1",
        "kind": "flexict_infer",
        "job_id": request["job_id"],
        "pipeline_request": remote_request,
        "remote_status_path": "/job/pipeline_job/status.json",
        "artifact_path": "/job/output/prediction_bundle",
        "created_at_epoch": time.time(),
    }
    write_flexict_json(bundle / "remote_request.json", payload)
    return {
        "remote_status_relative": "pipeline_job/status.json",
        "remote_control_relative": "pipeline_job/control.json",
        "remote_control_kind": "json",
        "remote_artifact_relative": "output/prediction_bundle",
        "train_count": 0,
        "validation_count": 0,
        "required_model_relative": "",
        "local_required_model": "",
        "local_dataset_archive_cache": str(
            Path(request["workspace"])
            / "cache"
            / "remote_inference_archives"
            / flexict_safe_identifier(manifest.get("model_id"), "model")
        ),
        "dataset_case_cache_keys": {
            case_id: image_fingerprint,
            "model": model_fingerprint,
        },
    }


def _register_flexict(
    spec: dict[str, Any],
    local_model_dir: Path,
    remote_status: dict[str, Any],
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    """Register the downloaded FlexiCT model pair (or single model).

    The remote artifact is one folder per trained configuration, all sharing
    a pair_id. Each downloaded folder already carries its
    flexict_model_manifest.json written by the remote pipeline; only the
    model_dir needs to point at the local destination.
    """
    import tools.flexict_common as flexict_common
    from tools.nnunet_common import (
        read_json as read_nnunet_json,
        write_json_atomic as write_nnunet_json,
    )

    request = read_nnunet_json(
        Path(spec["request_path"]).resolve(), {}
    ) or {}
    workspace = Path(request["workspace"]).expanduser().resolve()
    manifests = []
    for manifest_path in sorted(local_model_dir.rglob("flexict_model_manifest.json")):
        manifest = read_nnunet_json(manifest_path, {}) or {}
        if not manifest.get("model_id"):
            continue
        manifest["model_dir"] = str(manifest_path.parent)
        manifest["manifest_path"] = str(manifest_path)
        manifest.update(
            {
                "execution_backend": "remote",
                "remote_profile_id": profile["profile_id"],
                "remote_profile_name": profile["name"],
                "remote_runtime_image": profile["runtime_image"],
                "remote_gpu_device": remote_status.get(
                    "remote_gpu_device", profile.get("gpu_device") or "auto"
                ),
                "remote_dataset_fingerprint": remote_status.get(
                    "dataset_fingerprint", ""
                ),
                "remote_runtime_image_id": remote_status.get(
                    "remote_runtime_image_id", ""
                ),
                "downloaded_at_epoch": time.time(),
            }
        )
        write_nnunet_json(manifest_path, manifest)
        flexict_common.register_model(manifest, workspace=workspace)
        manifests.append(manifest)
    if not manifests:
        raise RuntimeError(
            "The downloaded FlexiCT artifact contains no model manifest."
        )
    return manifests


def _profile_for_spec(spec: dict[str, Any]) -> dict[str, Any]:
    profile_id = str(spec.get("remote_profile_id") or "")
    if not profile_id:
        raise RuntimeError("Remote server profile is missing from the job request.")
    return get_profile(profile_id)


def _remote_paths(session: SSHSession, job_id: str) -> dict[str, str]:
    owner = safe_identifier(session.profile.get("username"), "user")
    jobs = (
        PurePosixPath(session.remote_root)
        / "jobs"
        / owner
    )
    job = jobs / safe_identifier(job_id, "job")
    return {
        "jobs": str(jobs),
        "job": str(job),
        "archive": str(jobs / (safe_identifier(job_id, "job") + ".tar")),
        "models": str(PurePosixPath(session.remote_root) / "models"),
        "locks": str(PurePosixPath(session.remote_root) / "locks"),
        "dataset_cache": str(
            PurePosixPath(session.remote_root)
            / "cache"
            / owner
            / "datasets"
        ),
        "prepared_cache": str(
            PurePosixPath(session.remote_root)
            / "cache"
            / owner
            / "prepared"
        ),
        "models_cache": str(
            PurePosixPath(session.remote_root)
            / "cache"
            / owner
            / "models"
        ),
    }


def _prepared_cache_namespace(bundle: Path, remote_paths: dict[str, str]) -> str:
    payload = read_json(bundle / "remote_request.json", {}) or {}
    raw = str(
        (((payload.get("pipeline_request") or {}).get("runtime_roots") or {}).get("raw"))
        or ""
    )
    raw_path = PurePosixPath(raw)
    container_root = PurePosixPath("/remote-cache")
    try:
        relative = raw_path.relative_to(container_root).parent
    except ValueError:
        return ""
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        return ""
    prepared_root = PurePosixPath(remote_paths["prepared_cache"])
    namespace = prepared_root / relative
    try:
        namespace.relative_to(prepared_root)
    except ValueError:
        return ""
    return str(namespace)


def _maintain_remote_nnunet_cache(
    session: SSHSession,
    profile: dict[str, Any],
    remote_paths: dict[str, str],
    namespace: str,
    status_path: Path,
    log_path: Path,
) -> None:
    if not namespace:
        return
    prepared_root = PurePosixPath(remote_paths["prepared_cache"])
    namespace_path = PurePosixPath(namespace)
    try:
        namespace_path.relative_to(prepared_root)
    except ValueError:
        raise RuntimeError("Refusing to maintain a cache outside the remote cache root.")
    base = namespace_path.parent
    owner_filter = "label={}={}".format(
        REMOTE_OWNER_LABEL, _remote_owner(profile)
    )
    active = str(
        session.execute(
            "{runtime} ps -q --filter {owner} --filter {training}".format(
                runtime=container_runtime_command(getattr(session, "profile", None)),
                owner=shlex.quote(owner_filter),
                training=shlex.quote("label={}={}".format(REMOTE_TRAINING_LABEL, "true")),
            ),
            check=False,
        )
        or ""
    ).strip()
    removed_old = False
    if not active:
        retention = int(profile.get("remote_cache_retention_days") or 30)
        exclude_inference = (
            " ! -name inference" if base.name == "nnunet" else ""
        )
        session.execute(
            "mkdir -p {base} && find {base} -mindepth 1 -maxdepth 1 "
            "-type d -mtime +{days} ! -path {current}{exclude} "
            "-exec rm -rf -- {{}} +".format(
                base=shlex.quote(str(base)),
                days=max(1, retention),
                current=shlex.quote(str(namespace_path)),
                exclude=exclude_inference,
            ),
            timeout=600,
        )
        removed_old = True
    session.execute(
        "mkdir -p {path} && touch {path}".format(
            path=shlex.quote(str(namespace_path))
        ),
        timeout=300,
    )
    _status_update(
        status_path,
        remote_prepared_cache_namespace=str(namespace_path),
        remote_cache_cleanup_performed=removed_old,
        remote_cache_cleanup_skipped_for_active_jobs=bool(active),
    )
    _append_log(
        log_path,
        (
            "Remote nnU-Net cache retention was checked."
            if removed_old
            else "Remote nnU-Net cache cleanup was deferred because another owned container is active."
        ),
    )


def _finalize_completed_remote_job(
    session: SSHSession,
    spec: dict[str, Any],
    profile: dict[str, Any],
    status_path: Path,
    remote_paths: dict[str, str],
    prepared: dict[str, Any],
    remote_status: dict[str, Any],
    asset_identity: dict[str, Any],
    runtime_image_id: str,
    container_name: str,
    container_may_exist: bool,
    prepared_cache_namespace: str,
    local_root: Path,
    result_archive: Path,
    controller_log: Path,
    remote_log: Path,
) -> int:
    """Download the finished artifact and register it locally.

    Shared by the fresh-run path (run) and the re-attach path (reattach):
    both end with a remote 'completed' status and must perform the same
    download → verify → publish → register sequence.
    """
    kind = str(spec.get("kind") or "")
    owner = _remote_owner(profile)
    job_slug = safe_identifier(str(spec["job_id"]), "job")
    if kind == "nninteractive":
        local_request = read_json(
            Path(spec["job_dir"]) / "request.json", {}
        ) or {}
        local_model_dir = Path(local_request["output_model_dir"]).resolve()
    elif kind == "nnunet":
        from tools.nnunet_common import read_json as read_nnunet_json
        from tools.nnunet_common import safe_identifier as nnunet_safe_identifier

        local_request = read_nnunet_json(Path(spec["request_path"]), {}) or {}
        local_model_dir = (
            Path(local_request["workspace"]).expanduser().resolve()
            / "models"
            / nnunet_safe_identifier(local_request["task_id"])
            / nnunet_safe_identifier(prepared["local_model_id"])
        )
    elif kind == "flexict":
        from tools.nnunet_common import read_json as read_nnunet_json
        from tools.nnunet_common import safe_identifier as nnunet_safe_identifier

        local_request = read_nnunet_json(Path(spec["request_path"]), {}) or {}
        local_model_dir = (
            Path(local_request["workspace"]).expanduser().resolve()
            / "models"
            / nnunet_safe_identifier(local_request["task_id"])
            / nnunet_safe_identifier(prepared["local_model_id"])
        )
    else:
        local_model_dir = Path(spec["job_dir"]).resolve() / "remote_prediction_bundle"
    _status_update(
        status_path,
        status="downloading",
        phase="downloading_model",
        progress_percent=96,
    )
    download_reconnect_attempt = 0
    while True:
        _raise_if_cancelled(status_path)
        try:
            if session is None:
                raise RemoteComputeError("No active SSH download session.")
            _sync_remote_log(
                session,
                remote_paths["job"],
                remote_log,
                {"offset": 0, "remote_relative": _remote_log_relative(kind)},
            )
            _download_artifact(
                session,
                remote_paths["job"],
                prepared["remote_artifact_relative"],
                local_model_dir,
                result_archive,
                status_path,
            )
            break
        except RemoteCommandError:
            raise
        except (RemoteComputeError, EOFError, OSError, socket.error) as exc:
            try:
                session.close()
            except Exception:
                pass
            session = None
            download_reconnect_attempt += 1
            session = _reconnect_session(
                profile,
                status_path,
                controller_log,
                exc,
                download_reconnect_attempt,
            )
    if kind == "nninteractive":
        remote_status.update(asset_identity)
        remote_status["remote_runtime_image_id"] = runtime_image_id
        remote_status["remote_gpu_device"] = profile.get("gpu_device") or "auto"
        model, selected = _register_nninteractive(
            spec, local_model_dir, remote_status, profile
        )
    elif kind == "nnunet":
        remote_status.update(asset_identity)
        remote_status["remote_runtime_image_id"] = runtime_image_id
        remote_status["remote_gpu_device"] = profile.get("gpu_device") or "auto"
        model = _register_nnunet(
            spec, local_model_dir, remote_status, profile
        )
        selected = True
    elif kind == "flexict":
        remote_status.update(asset_identity)
        remote_status["remote_runtime_image_id"] = runtime_image_id
        remote_status["remote_gpu_device"] = profile.get("gpu_device") or "auto"
        models = _register_flexict(
            spec, local_model_dir, remote_status, profile
        )
        model = models[0] if len(models) == 1 else None
        selected = True
        # The remote pipeline's status (merged earlier) carries container-side
        # /job/... paths in its models array. Overwrite it with the locally
        # registered manifests so downstream consumers (acceptance checklist,
        # active learning) resolve real local model dirs.
        local_model_rows = [
            {
                key: manifest.get(key)
                for key in ("configuration", "model_id", "model_dir")
            }
            for manifest in models
        ]
        _status_update(status_path, models=local_model_rows)
    else:
        from tools.nnunet_common import read_json as read_nnunet_json

        local_request = read_nnunet_json(Path(spec["request_path"]), {}) or {}
        downloaded = local_model_dir / "prediction.nii.gz"
        output_path = Path(str(local_request.get("output_path") or downloaded)).resolve()
        if not downloaded.is_file():
            raise RuntimeError(
                "Remote {} inference produced no prediction file.".format(
                    "nnU-Net" if kind == "nnunet_infer" else "FlexiCT"
                )
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if downloaded.resolve() != output_path.resolve():
            temporary_output = output_path.with_name(output_path.name + ".remote-part")
            shutil.copy2(str(downloaded), str(temporary_output))
            os.replace(str(temporary_output), str(output_path))
        model = {
            "model_id": str(remote_status.get("model_id") or ""),
            "output_path": str(output_path),
        }
        selected = True
        remote_status["output_path"] = str(output_path)
    _status_update(
        status_path,
        status="completed",
        phase="completed",
        progress_percent=100,
        model=model,
        execution_backend="remote",
        remote_model_downloaded=True,
        remote_model_selected=bool(selected),
        completed_at_epoch=time.time(),
        local_log_path=str(remote_log),
        controller_log_path=str(controller_log),
        remote_container_removed=False,
        output_path=(
            str(remote_status.get("output_path") or "")
            if kind in {"nnunet_infer", "flexict_infer"}
            else None
        ),
    )
    # Training data and the downloaded archive are no longer needed on the
    # server. The model now lives in the existing local registry.
    container_removed = False
    job_removed = False
    cleanup_warning = ""
    try:
        container_removed = _remove_container(
            session,
            container_name,
            expected_owner=owner,
            expected_job=job_slug,
            stop_if_running=False,
        )
        job_removed = _remove_remote_job(
            session,
            remote_root=profile["remote_root"],
            expected_owner=owner,
            remote_job_dir=remote_paths["job"],
        )
        if (
            kind in {"nnunet", "nnunet_infer", "flexict", "flexict_infer"}
            and not bool(profile.get("cache_training_data", True))
        ):
            _remove_remote_prepared_namespace(
                session, remote_paths, prepared_cache_namespace
            )
        if not (container_removed and job_removed):
            cleanup_warning = (
                "Training completed, but some remote temporary files "
                "could not be removed."
            )
    except Exception as cleanup_exc:
        cleanup_warning = (
            "Training and model registration completed, but remote cleanup "
            "could not be confirmed: {}".format(cleanup_exc)
        )
        _append_log(controller_log, cleanup_warning)
    _status_update(
        status_path,
        remote_container_removed=bool(container_removed),
        remote_job_removed=bool(job_removed),
        remote_cleanup_warning=cleanup_warning,
        remote_dataset_cache_retained=True,
    )
    return 0


def _remote_log_relative(kind: str) -> str:
    if kind in {"nnunet", "nnunet_infer", "flexict", "flexict_infer"}:
        return "pipeline_job/job.log"
    return "remote_worker.log"


def _remote_status_relative(kind: str) -> str:
    return "pipeline_job/status.json"


def _remote_artifact_relative(
    kind: str,
    spec: dict[str, Any],
    remote_status: dict[str, Any] | None = None,
) -> str:
    """Recover the artifact path when the original prepared bundle is gone.

    The re-attach path no longer has the local bundle that _prepare_*
    built, so derive the artifact location from the job kind and the
    deterministic naming the prepares use (mirrors _prepare_nnunet,
    _prepare_flexict, and the infer prepares). Training model ids are
    taken from the remote pipeline status first (the remote registration
    writes the trained model manifest there) and fall back to the local
    request.
    """
    if kind == "nninteractive":
        return "model_output"
    request = {}
    try:
        from tools.nnunet_common import read_json as read_nnunet_json
        from tools.nnunet_common import safe_identifier as nnunet_safe_identifier

        request = (
            read_nnunet_json(Path(str(spec.get("request_path") or "")), {}) or {}
        )
    except Exception:
        request = {}
    task_slug = nnunet_safe_identifier(request.get("task_id") or "task", "task")
    if kind in {"nnunet", "flexict"}:
        status = dict(remote_status or {})
        model = (
            status.get("model")
            if isinstance(status.get("model"), dict)
            else None
        )
        model_ids = [
            str(source)
            for source in (
                (model or {}).get("model_id"),
                request.get("model_id"),
            )
            if str(source or "").strip()
        ]
        if not model_ids:
            return ""
        # FlexiCT trains a 2D+3D pair into one artifact folder; use the
        # first model id, they all share the artifact directory.
        return "output/models/{}/{}".format(task_slug, model_ids[0])
    return "output/prediction_bundle"


def _reattach_finalize_from_status(
    session: SSHSession,
    spec: dict[str, Any],
    profile: dict[str, Any],
    status_path: Path,
    remote_paths: dict[str, str],
    remote_status: dict[str, Any],
    container_name: str,
    local_root: Path,
    result_archive: Path,
    controller_log: Path,
    remote_log: Path,
) -> int:
    """Download + register a remotely-completed job (controller was dead).

    The remote container already finished; only the local tail (download,
    verify, register) is missing. Used by the re-attach path when the
    remote pipeline status is terminal 'completed' but no model was ever
    downloaded.
    """
    kind = str(spec.get("kind") or "")
    prepared = {
        "remote_artifact_relative": _remote_artifact_relative(
            kind, spec, remote_status
        ),
    }
    if not prepared["remote_artifact_relative"]:
        raise RuntimeError(
            "The remote job completed, but its model id could not be "
            "recovered from the local request file. Stop the task and retry "
            "training."
        )
    local_model_dir = _resolve_local_model_dir(spec, kind, remote_status)
    _status_update(
        status_path,
        status="downloading",
        phase="downloading_model",
        progress_percent=96,
        message=(
            "Re-attached to a completed remote task; downloading the model."
        ),
    )
    download_reconnect_attempt = 0
    while True:
        _raise_if_cancelled(status_path)
        try:
            if session is None:
                raise RemoteComputeError("No active SSH download session.")
            _sync_remote_log(
                session,
                remote_paths["job"],
                remote_log,
                {"offset": 0, "remote_relative": _remote_log_relative(kind)},
            )
            _download_artifact(
                session,
                remote_paths["job"],
                prepared["remote_artifact_relative"],
                local_model_dir,
                result_archive,
                status_path,
            )
            break
        except RemoteCommandError:
            raise
        except (RemoteComputeError, EOFError, OSError, socket.error) as exc:
            try:
                session.close()
            except Exception:
                pass
            session = None
            download_reconnect_attempt += 1
            session = _reconnect_session(
                profile,
                status_path,
                controller_log,
                exc,
                download_reconnect_attempt,
            )
    asset_identity = {
        key: value
        for key, value in (
            remote_status.get("identity") or {}
        ).items()
    }
    runtime_image_id = str(
        (read_json(status_path, {}) or {}).get("remote_runtime_image_id") or ""
    )
    if kind == "nnunet":
        model = _register_nnunet(
            spec, local_model_dir, remote_status, profile
        )
    elif kind == "flexict":
        models = _register_flexict(
            spec, local_model_dir, remote_status, profile
        )
        model = models[0] if len(models) == 1 else None
    elif kind == "nninteractive":
        model, _selected = _register_nninteractive(
            spec, local_model_dir, remote_status, profile
        )
    else:
        _apply_remote_prediction_output(spec, kind, local_model_dir, remote_status)
        model = {
            "model_id": str(remote_status.get("model_id") or ""),
            "output_path": str(remote_status.get("output_path") or ""),
        }
    _status_update(
        status_path,
        status="completed",
        phase="completed",
        progress_percent=100,
        model=model,
        execution_backend="remote",
        remote_model_downloaded=True,
        remote_model_selected=True,
        remote_reattached=True,
        completed_at_epoch=time.time(),
        local_log_path=str(remote_log),
        controller_log_path=str(controller_log),
        remote_container_removed=False,
        output_path=(
            str(remote_status.get("output_path") or "")
            if kind in {"nnunet_infer", "flexict_infer"}
            else None
        ),
    )
    owner = _remote_owner(profile)
    job_slug = safe_identifier(str(spec["job_id"]), "job")
    container_removed = False
    job_removed = False
    cleanup_warning = ""
    try:
        container_removed = _remove_container(
            session,
            container_name,
            expected_owner=owner,
            expected_job=job_slug,
            stop_if_running=False,
        )
        job_removed = _remove_remote_job(
            session,
            remote_root=profile["remote_root"],
            expected_owner=owner,
            remote_job_dir=remote_paths["job"],
        )
        if not (container_removed and job_removed):
            cleanup_warning = (
                "Training completed, but some remote temporary files "
                "could not be removed."
            )
    except Exception as cleanup_exc:
        cleanup_warning = (
            "Training and model registration completed, but remote cleanup "
            "could not be confirmed: {}".format(cleanup_exc)
        )
        _append_log(controller_log, cleanup_warning)
    _status_update(
        status_path,
        remote_container_removed=bool(container_removed),
        remote_job_removed=bool(job_removed),
        remote_cleanup_warning=cleanup_warning,
        remote_dataset_cache_retained=True,
    )
    return 0


def _resolve_local_model_dir(
    spec: dict[str, Any], kind: str, remote_status: dict[str, Any]
) -> Path:
    if kind == "nninteractive":
        local_request = read_json(
            Path(spec["job_dir"]) / "request.json", {}
        ) or {}
        return Path(local_request["output_model_dir"]).resolve()
    if kind in {"nnunet", "flexict"}:
        from tools.nnunet_common import read_json as read_nnunet_json
        from tools.nnunet_common import safe_identifier as nnunet_safe_identifier

        local_request = read_nnunet_json(Path(spec["request_path"]), {}) or {}
        workspace = Path(local_request["workspace"]).expanduser().resolve()
        task = nnunet_safe_identifier(local_request["task_id"])
        model = (
            remote_status.get("model")
            if isinstance(remote_status.get("model"), dict)
            else None
        )
        model_id = str(
            (model or {}).get("model_id")
            or local_request.get("model_id")
            or ""
        )
        if not model_id:
            raise RuntimeError(
                "Neither the remote pipeline status nor the local request "
                "records a model id for this remote task, so the download "
                "target cannot be derived. Stop the task and retry training."
            )
        return workspace / "models" / task / nnunet_safe_identifier(model_id)
    return Path(spec["job_dir"]).resolve() / "remote_prediction_bundle"


def _apply_remote_prediction_output(
    spec: dict[str, Any],
    kind: str,
    local_model_dir: Path,
    remote_status: dict[str, Any],
) -> None:
    from tools.nnunet_common import read_json as read_nnunet_json

    local_request = read_nnunet_json(Path(spec["request_path"]), {}) or {}
    downloaded = local_model_dir / "prediction.nii.gz"
    output_path = Path(str(local_request.get("output_path") or downloaded)).resolve()
    if not downloaded.is_file():
        raise RuntimeError(
            "Remote {} inference produced no prediction file.".format(
                "nnU-Net" if kind == "nnunet_infer" else "FlexiCT"
            )
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if downloaded.resolve() != output_path.resolve():
        temporary_output = output_path.with_name(output_path.name + ".remote-part")
        shutil.copy2(str(downloaded), str(temporary_output))
        os.replace(str(temporary_output), str(output_path))
    remote_status["output_path"] = str(output_path)


def _remove_remote_prepared_namespace(
    session: SSHSession, remote_paths: dict[str, str], namespace: str
) -> bool:
    if not namespace:
        return True
    root = PurePosixPath(remote_paths["prepared_cache"])
    candidate = PurePosixPath(namespace)
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        raise RuntimeError("Refusing to remove a cache outside the remote cache root.")
    if not relative.parts:
        raise RuntimeError("Refusing to remove the remote prepared-cache root.")
    session.execute(
        "rm -rf -- {path}".format(path=shlex.quote(str(candidate))),
        timeout=600,
    )
    return True


def _remote_owner(profile: dict[str, Any]) -> str:
    return safe_identifier(profile.get("username"), "user")


def _validate_remote_job_path(
    remote_root: str,
    owner: str,
    remote_job_dir: str,
) -> PurePosixPath:
    candidate = PurePosixPath(str(remote_job_dir or ""))
    expected_parent = (
        PurePosixPath(str(remote_root))
        / "jobs"
        / safe_identifier(owner, "user")
    )
    if (
        not remote_job_dir
        or any(part == ".." for part in candidate.parts)
        or candidate.parent != expected_parent
        or candidate.name in {"", ".", ".."}
    ):
        raise RuntimeError(
            "Refusing to modify a remote path outside this user's job folder: "
            "{}".format(remote_job_dir or "<empty>")
        )
    return candidate


def _validate_remote_control_path(
    remote_job_dir: str,
    remote_control_path: str,
) -> None:
    if not remote_control_path:
        return
    job = PurePosixPath(remote_job_dir)
    control = PurePosixPath(remote_control_path)
    if (
        any(part == ".." for part in control.parts)
        or control == job
        or job not in control.parents
    ):
        raise RuntimeError(
            "Refusing to write a control marker outside the remote job folder."
        )


def _container_name(profile: dict[str, Any], job_id: str) -> str:
    return safe_identifier(
        "mimics-ai-{}-{}".format(profile["username"], job_id),
        "mimics-ai-job",
    )[:120]


def _launch_container(
    session: SSHSession,
    paths: dict[str, str],
    profile: dict[str, Any],
    container_name: str,
    dataset_cache_paths: list[str] | None = None,
    remove_dataset_after_extract: bool = False,
    status_path: Path | None = None,
    log_path: Path | None = None,
    custom_model_cache_path: str = "",
) -> str:
    owner = _remote_owner(profile)
    job_slug = safe_identifier(PurePosixPath(paths["job"]).name, "job")
    _validate_remote_job_path(
        str(profile["remote_root"]),
        owner,
        paths["job"],
    )
    if _assert_container_owned(
        session,
        container_name,
        expected_owner=owner,
        expected_job=job_slug,
    ):
        existing_state, _existing_code = _container_state(
            session, container_name
        )
        if existing_state not in {"exited", "dead", "created"}:
            raise RuntimeError(
                "This remote training job already has a running container. "
                "Open its status instead of starting a duplicate controller."
            )
        if not _remove_container(
            session,
            container_name,
            expected_owner=owner,
            expected_job=job_slug,
            stop_if_running=False,
        ):
            raise RuntimeError(
                "The previous owned remote container could not be removed."
            )
    # Each SSH exec passes the command through the remote login shell as a
    # single argument, which the kernel caps near MAX_ARG_STRLEN (~128 KB on
    # Linux). Concatenating one "tar -xf ... && ..." clause per dataset case
    # produced a string that exceeded the cap and failed with
    # "/bin/bash: Argument list too long". Execute each step in its own short
    # command so the per-exec size is bounded by path length, not case count.
    session.execute(
        "rm -rf {job} && mkdir -p {job}".format(
            job=shlex.quote(paths["job"]),
        ),
        timeout=300,
    )
    dataset_paths = list(dataset_cache_paths or [])
    for index, dataset_cache_path in enumerate(dataset_paths, start=1):
        if status_path is not None:
            _status_update(
                status_path,
                status="starting_remote",
                phase="extracting_remote_dataset",
                remote_dataset_extract_index=index,
                remote_dataset_extract_total=len(dataset_paths),
                progress_percent=35
                + int(3.0 * index / max(1, len(dataset_paths))),
            )
        if log_path is not None and (
            index == 1
            or index == len(dataset_paths)
            or index % 10 == 0
        ):
            _append_log(
                log_path,
                "Remote dataset extraction: {} of {} case archives.".format(
                    index, len(dataset_paths)
                ),
            )
        after_extract = (
            "rm -f {dataset}".format(
                dataset=shlex.quote(dataset_cache_path)
            )
            if remove_dataset_after_extract
            else "touch {dataset}".format(
                dataset=shlex.quote(dataset_cache_path)
            )
        )
        session.execute(
            "tar -xf {dataset} -C {job} && {after_extract}".format(
                dataset=shlex.quote(dataset_cache_path),
                job=shlex.quote(paths["job"]),
                after_extract=after_extract,
            ),
            timeout=300,
        )
    session.execute(
        "tar -xf {archive} -C {job} && rm -f {archive}".format(
            archive=shlex.quote(paths["archive"]),
            job=shlex.quote(paths["job"]),
        ),
        timeout=300,
    )
    if custom_model_cache_path:
        # The cached custom base model was excluded from the job archive
        # (content-addressed cache hit). Extract it into the same in-container
        # location the pipeline request references: the tar was built with
        # that layout (directory contents at top level / file by basename).
        session.execute(
            "mkdir -p {target} && "
            "tar -xf {model} -C {target} && touch {model}".format(
                target=shlex.quote(paths["job"] + "/custom_model"),
                model=shlex.quote(custom_model_cache_path),
            ),
            timeout=300,
        )
    gpu_device = str(profile.get("gpu_device") or "auto")
    gpu_scope = "all" if gpu_device == "auto" else "device"
    gpu_lock = "gpu-{}.lock".format(
        "automatic" if gpu_scope == "all" else safe_identifier(gpu_device, "device")
    )
    command = (
        "{runtime} run -d --name {name} --gpus {gpu_request} --network none "
        "--label mimics-script.remote-training=true "
        "--label {owner_label} "
        "--label {job_label} "
        "--label {gpu_label} "
        "-v {job}:/job -v {models}:/models:ro -v {locks}:/remote-locks "
        "-v {prepared_cache}:/remote-cache "
        "-e MIMICS_AI_APP_ROOT=/app "
        "-e MIMICS_REMOTE_GPU_GLOBAL_LOCK=/remote-locks/gpu-all.lock "
        "-e MIMICS_REMOTE_GPU_LOCK=/remote-locks/{gpu_lock} "
        "-e MIMICS_REMOTE_GPU_LOCK_SCOPE={gpu_scope} "
        "-e MIMICS_REMOTE_GPU_DEVICE={gpu_device} "
        "-e MIMICS_REMOTE_MODELS_ROOT=/models "
        "-e HF_HOME=/tmp/mimics-ai-cache/huggingface "
        "-e TORCH_HOME=/tmp/mimics-ai-cache/torch "
        "-e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 "
        "-e HF_DATASETS_OFFLINE=1 -e WANDB_MODE=offline "
        "{image} python /app/tools/remote_worker.py run --job-dir /job"
    ).format(
        runtime=container_runtime_command(profile),
        name=shlex.quote(container_name),
        gpu_request=shlex.quote(docker_gpu_request(profile)),
        owner_label=shlex.quote(
            "{}={}".format(REMOTE_OWNER_LABEL, owner)
        ),
        job_label=shlex.quote(
            "{}={}".format(REMOTE_JOB_LABEL, job_slug)
        ),
        gpu_label=shlex.quote(
            "mimics-script.gpu={}".format(
                safe_identifier(gpu_device, "automatic")
            )
        ),
        job=shlex.quote(paths["job"]),
        models=shlex.quote(paths["models"]),
        locks=shlex.quote(paths["locks"]),
        prepared_cache=shlex.quote(
            paths.get(
                "prepared_cache",
                str(PurePosixPath(paths["locks"]).parent / "prepared"),
            )
        ),
        gpu_lock=shlex.quote(gpu_lock),
        gpu_scope=shlex.quote(gpu_scope),
        gpu_device=shlex.quote(gpu_device),
        image=shlex.quote(profile["runtime_image"]),
    )
    return session.execute(command).strip()


def _validate_remote_assets(
    session: SSHSession,
    paths: dict[str, str],
    prepared: dict[str, Any],
    verify_mode: str = "strict",
) -> dict[str, str]:
    relative = str(prepared.get("required_model_relative") or "").strip()
    if not relative:
        return {}
    verify_mode = str(verify_mode or "strict").strip().lower()
    if verify_mode == "off":
        return {
            "remote_base_model_path": str(
                PurePosixPath(paths["models"]) / PurePosixPath(relative)
            ),
            "remote_base_model_verify": "off",
        }
    model_path = str(PurePosixPath(paths["models"]) / PurePosixPath(relative))
    output = session.execute(
        "test -e {path} && printf ready || printf missing".format(
            path=shlex.quote(model_path)
        ),
        check=False,
    ).strip()
    if output != "ready":
        raise RuntimeError(
            "The required base model is not installed on the remote server: "
            "{}. Ask the server administrator to install it under the configured "
            "remote work folder.".format(model_path)
        )
    # Content-only directory fingerprint: sorted SHA-256 of each supported
    # weight file, no path names. The remote layout may differ from the local
    # one (custom remote_root or mount point), and unrelated extra files must
    # not invalidate the identity check. `find -type f` without -L skips
    # symlinks, matching _local_model_fingerprint.
    #
    # The full hash of a multi-gigabyte base-model directory costs minutes of
    # remote I/O on every job. A `.mimics_digest_cache` marker next to the
    # model records the last computed digest with the newest source mtime;
    # when no file has changed since, the cached digest is trusted and the
    # rehash is skipped. A marker mismatch (stale, corrupt, missing) falls
    # back to the full hash below, and a read-only model mount makes the
    # best-effort marker write a silent no-op.
    digest_marker = model_path.rstrip("/") + ".mimics_digest_cache"
    cache_output = session.execute(
        "if test -f {marker} && "
        'read saved_digest saved_mtime < {marker} && '
        'test "$saved_mtime" = "$(cd {path} 2>/dev/null && '
        "find . -maxdepth 4 -type f "
        "\\( -name '*.pth' -o -name '*.safetensors' -o -name '*.bin' "
        "-o -name 'config.json' -o -name 'plans.json' \\) "
        "-printf '%T@\\n' 2>/dev/null | sort -n | tail -1 | "
        'cut -d. -f1)"; '
        'then printf "%s" "$saved_digest"; fi'.format(
            marker=shlex.quote(digest_marker),
            path=shlex.quote(model_path),
        ),
        timeout=60,
        check=False,
    ).strip()
    if cache_output:
        fingerprint = cache_output
    else:
        fingerprint = session.execute(
            "if test -f {path}; then "
            "sha256sum {path} | awk '{{print $1}}'; "
            "else cd {path} && "
            "find . -maxdepth 4 -type f "
            "\\( -name '*.pth' -o -name '*.safetensors' -o -name '*.bin' "
            "-o -name 'config.json' -o -name 'plans.json' \\) "
            "-exec sha256sum {{}} + 2>/dev/null | "
            "awk '{{print $1}}' | LC_ALL=C sort | "
            "sha256sum | awk '{{print $1}}'; "
            "fi".format(path=shlex.quote(model_path)),
            timeout=300,
        ).strip()
        if len(fingerprint) == 64:
            session.execute(
                "if test -d {path}; then cd {path} && "
                "newest_mtime=$(find . -maxdepth 4 -type f "
                "\\( -name '*.pth' -o -name '*.safetensors' -o -name '*.bin' "
                "-o -name 'config.json' -o -name 'plans.json' \\) "
                "-printf '%T@\\n' 2>/dev/null | sort -n | tail -1 | "
                "cut -d. -f1) && "
                "printf '%s %s\\n' {digest} \"$newest_mtime\" "
                "> {marker}.tmp && mv -f {marker}.tmp {marker}; fi || true".format(
                    path=shlex.quote(model_path),
                    digest=shlex.quote(fingerprint),
                    marker=shlex.quote(digest_marker),
                ),
                check=False,
            )
    if (
        not fingerprint
        or fingerprint == hashlib.sha256(b"").hexdigest()
    ):
        raise RuntimeError(
            "The remote base-model path exists but contains no supported "
            "weight files: {}".format(model_path)
        )
    local_path_text = str(prepared.get("local_required_model") or "").strip()
    local_fingerprint = (
        _local_model_fingerprint(Path(local_path_text))
        if local_path_text and Path(local_path_text).exists()
        else ""
    )
    identity = {
        "remote_base_model_path": model_path,
        "remote_base_model_sha256": fingerprint,
        "local_base_model_sha256": local_fingerprint,
        "remote_base_model_verify": "strict",
    }
    if local_fingerprint and local_fingerprint != fingerprint:
        detail = (
            "The remote base model does not match the corresponding local "
            "model. Local SHA-256: {}; remote SHA-256: {}. Note: supported weight "
            "and model-configuration contents are hashed; directory layout and unrelated files are "
            "ignored. Install the same base weights on the server before "
            "training.".format(local_fingerprint, fingerprint)
        )
        if verify_mode == "warn":
            identity.update(
                {
                    "remote_base_model_verify": "warn",
                    "remote_base_model_mismatch": True,
                    "remote_base_model_mismatch_detail": detail,
                }
            )
        else:
            raise RuntimeError(detail)
    return identity


def _container_state(session: SSHSession, container_name: str) -> tuple[str, int]:
    code, output = session.execute_result(
        "{runtime} inspect --format '{{{{.State.Status}}}} "
        "{{{{.State.ExitCode}}}}' {container}".format(
            runtime=container_runtime_command(getattr(session, "profile", None)),
            container=shlex.quote(container_name),
        )
    )
    output = output.strip()
    if code != 0:
        lowered = output.lower()
        if "no such object" in lowered or "no such container" in lowered:
            return "missing", -1
        raise RemoteCommandError(
            "Could not inspect remote container '{}': {}".format(
                container_name, output.strip()[-2000:]
            )
        )
    parts = output.split()
    if len(parts) < 2:
        return "missing", -1
    try:
        code = int(parts[-1])
    except Exception:
        code = -1
    return parts[0].lower(), code


def _container_labels(
    session: SSHSession,
    container_name: str,
) -> dict[str, str]:
    code, output = session.execute_result(
        "{runtime} inspect --format '{{{{json .Config.Labels}}}}' {container}".format(
            runtime=container_runtime_command(getattr(session, "profile", None)),
            container=shlex.quote(container_name),
        )
    )
    if code != 0:
        raise RemoteCommandError(
            "Could not inspect ownership labels for remote container '{}': "
            "{}".format(container_name, output.strip()[-2000:])
        )
    # nerdctl can print warning lines (e.g. "failed to inspect NetNS" for a
    # stopped container's stale network namespace) to stdout ahead of the
    # template output; parse the last JSON object on the stream.
    payload = ""
    for line in reversed(output.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            payload = line
            break
    try:
        labels = json.loads(payload) if payload else {}
    except Exception as exc:
        raise RemoteCommandError(
            "Remote container '{}' returned invalid ownership labels: {}".format(
                container_name, exc
            )
        )
    return {
        str(key): str(value)
        for key, value in (labels or {}).items()
    }


def _assert_container_owned(
    session: SSHSession,
    container_name: str,
    *,
    expected_owner: str,
    expected_job: str,
) -> bool:
    state, _code = _container_state(session, container_name)
    if state == "missing":
        return False
    labels = _container_labels(session, container_name)
    expected = {
        REMOTE_TRAINING_LABEL: "true",
        REMOTE_OWNER_LABEL: safe_identifier(expected_owner, "user"),
        REMOTE_JOB_LABEL: safe_identifier(expected_job, "job"),
    }
    mismatches = [
        key
        for key, value in expected.items()
        if labels.get(key) != value
    ]
    if mismatches:
        raise RuntimeError(
            "Refusing to control remote container '{}': its ownership labels "
            "do not match this training job ({}).".format(
                container_name, ", ".join(mismatches)
            )
        )
    return True


def _remove_container(
    session: SSHSession,
    container_name: str,
    *,
    expected_owner: str,
    expected_job: str,
    stop_if_running: bool = True,
) -> bool:
    if not _assert_container_owned(
        session,
        container_name,
        expected_owner=expected_owner,
        expected_job=expected_job,
    ):
        return True
    state, _code = _container_state(session, container_name)
    if stop_if_running and state not in {"exited", "dead", "created"}:
        session.execute(
            "{runtime} stop --time 20 {container} >/dev/null 2>&1 || true".format(
                runtime=container_runtime_command(getattr(session, "profile", None)),
                container=shlex.quote(container_name),
            ),
            timeout=40,
            check=False,
        )
    session.execute(
        "{runtime} rm -f {container} >/dev/null 2>&1 || true".format(
            runtime=container_runtime_command(getattr(session, "profile", None)),
            container=shlex.quote(container_name),
        ),
        timeout=40,
        check=False,
    )
    state, _code = _container_state(session, container_name)
    return state == "missing"


def _remove_remote_job(
    session: SSHSession,
    *,
    remote_root: str,
    expected_owner: str,
    remote_job_dir: str,
) -> bool:
    job = _validate_remote_job_path(
        remote_root,
        expected_owner,
        remote_job_dir,
    )
    session.execute(
        "rm -rf -- {job} {job}.tar {job}.tar.part".format(
            job=shlex.quote(str(job))
        ),
        timeout=120,
        check=False,
    )
    # jobs/<id>.tar is a sibling of the job dir (see _remote_paths): the
    # uploaded job archive that, before this fix, survived cancellation
    # because only the directory was removed. Its .part upload fragment
    # goes the same way.
    code, _output = session.execute_result(
        "test ! -e {}".format(shlex.quote(str(job)))
    )
    return code == 0


def _reconnect_session(
    profile: dict[str, Any],
    status_path: Path,
    local_log: Path,
    reason: BaseException,
    attempt: int,
) -> SSHSession | None:
    _raise_if_cancelled(status_path)
    if attempt >= RECONNECT_MAX_ATTEMPTS:
        raise RemoteServerUnreachable(
            "The remote server could not be reached after {} reconnect "
            "attempts ({}). The remote container may still be running; use "
            "Re-attach to retry monitoring once the server is reachable "
            "again, or Stop/Abandon Locally to end this task.".format(
                attempt, reason
            )
        )
    delay = min(30.0, max(2.0, float(2 ** min(attempt, 5))))
    current = read_json(status_path, {}) or {}
    previous_status = str(
        current.get("remote_previous_status")
        or current.get("status")
        or "starting_remote"
    )
    if previous_status == "reconnecting_remote":
        previous_status = "starting_remote"
    message = (
        "Remote connection was interrupted ({}). The remote task state is "
        "unchanged; reconnecting in {:.0f} seconds.".format(reason, delay)
    )
    _append_log(local_log, message)
    _status_update(
        status_path,
        status="reconnecting_remote",
        phase="reconnecting_remote",
        remote_connection_error=str(reason),
        remote_reconnect_attempt=attempt,
        remote_state_unknown=True,
        remote_previous_status=previous_status,
    )
    deadline = time.time() + delay
    while time.time() < deadline:
        _raise_if_cancelled(status_path)
        time.sleep(min(0.5, max(0.0, deadline - time.time())))
    try:
        session = SSHSession(profile)
    except Exception as exc:
        _append_log(
            local_log,
            "Remote reconnect attempt {} failed: {}".format(attempt, exc),
        )
        return None
    try:
        _raise_if_cancelled(status_path)
    except BaseException:
        session.close()
        raise
    _status_update(
        status_path,
        status=previous_status,
        phase="monitoring_remote",
        remote_connection_error="",
        remote_state_unknown=False,
        remote_reconnect_attempt=attempt,
    )
    return session


def _merge_remote_status(
    status_path: Path,
    remote: dict[str, Any],
    profile: dict[str, Any],
    container_name: str,
    remote_job_dir: str,
) -> dict[str, Any]:
    _raise_if_abandoned(status_path)
    local = read_json(status_path, {}) or {}
    preserved = {key: local.get(key) for key in LOCAL_STATUS_KEYS if key in local}
    remote_copy = dict(remote)
    remote_state = str(remote_copy.get("status") or "").lower()
    try:
        remote_progress = float(remote_copy.get("progress_percent") or 0.0)
    except Exception:
        remote_progress = 0.0
    remote_progress = max(0.0, min(100.0, remote_progress))
    remote_copy["remote_training_progress_percent"] = remote_progress
    remote_copy["progress_percent"] = 35 + int(remote_progress * 0.60)
    if remote_state == "completed":
        remote_copy["remote_pipeline_status"] = "completed"
        remote_copy["status"] = "finalizing_remote"
        remote_copy["phase"] = "remote_training_completed"
        remote_copy["progress_percent"] = 95
    remote_metrics = remote_copy.get("metrics_history")
    for key in (
        "cancel_path",
        "control_path",
        "controller_pid",
        "log_path",
        "metrics_history",
        "request_path",
        "train_log",
        "workspace",
        "ts_root",
    ):
        remote_copy.pop(key, None)
    if isinstance(remote_metrics, list):
        remote_copy["metrics_history"] = remote_metrics[-500:]
    local.update(remote_copy)
    local.update(preserved)
    local.update(
        {
            "execution_backend": "remote",
            "remote_profile_id": profile["profile_id"],
            "profile_name": profile["name"],
            "remote_container_name": container_name,
            "remote_job_dir": remote_job_dir,
            "remote_status_updated_at_epoch": remote.get("updated_at_epoch"),
            "updated_at_epoch": time.time(),
        }
    )
    write_json_atomic(status_path, local)
    return local


def _upload_progress(
    status_path: Path,
    *,
    phase: str = "uploading_training_data",
    start_percent: int = 20,
    end_percent: int = 35,
    aggregate_offset: int = 0,
    aggregate_total: int = 0,
):
    state = {
        "last_time": 0.0,
        "last_percent": -1,
        "started_at": 0.0,
        "started_bytes": 0,
    }

    def callback(transferred: int, total: int) -> None:
        _raise_if_cancelled(status_path)
        now = time.time()
        effective_transferred = int(aggregate_offset) + int(transferred)
        effective_total = int(aggregate_total) or int(total)
        if not state["started_at"]:
            state["started_at"] = now
            state["started_bytes"] = effective_transferred
        elapsed = max(0.001, now - float(state["started_at"]))
        measured_bytes = max(
            0, effective_transferred - int(state["started_bytes"])
        )
        bytes_per_second = (
            float(measured_bytes) / elapsed if measured_bytes else 0.0
        )
        eta_seconds = (
            max(0.0, effective_total - effective_transferred)
            / bytes_per_second
            if bytes_per_second > 0
            else None
        )
        percent = int(
            100.0 * effective_transferred / max(1, effective_total)
        )
        percent = min(100, max(0, percent))
        if percent == state["last_percent"] and now - state["last_time"] < 1.0:
            return
        if now - state["last_time"] < 0.25 and percent < 100:
            return
        state["last_time"] = now
        state["last_percent"] = percent
        _status_update(
            status_path,
            status="uploading",
            phase=phase,
            transfer_bytes=effective_transferred,
            transfer_total_bytes=effective_total,
            transfer_percent=percent,
            transfer_bytes_per_second=bytes_per_second,
            transfer_eta_seconds=eta_seconds,
            progress_percent=start_percent
            + int(percent * max(0, end_percent - start_percent) / 100.0),
        )

    return callback


def _batch_verify_remote_dataset_cache(
    session: SSHSession,
    paths: dict[str, str],
    parts: list[dict[str, Any]],
    chunk_size: int = 50,
) -> dict[str, bool]:
    """Classify every case cache slot in a bounded number of SSH round trips.

    Without this, the per-case upload loop costs one verify round trip per
    case before it even decides whether to upload. The command only trusts a
    `.verified` marker whose recorded digest, size, and archive mtime all
    match — the same acceptance rule as `_ensure_remote_dataset_archive`, so
    a "hit" here is exactly a hit there. Anything else is reported as a miss
    and the per-case path re-runs its full (repairing) verification.
    Fingerprints are chunked so one exec stays far below the remote shell's
    argument-length cap regardless of case count.
    """
    verdicts: dict[str, bool] = {}
    entries = [
        (str(part["fingerprint"]), int(part["size"]))
        for part in parts
        if str(part.get("fingerprint") or "")
    ]
    for start in range(0, len(entries), chunk_size):
        chunk = entries[start : start + chunk_size]
        clauses = []
        for fingerprint, size in chunk:
            archive = PurePosixPath(paths["dataset_cache"]) / (
                fingerprint + ".tar"
            )
            verified = PurePosixPath(paths["dataset_cache"]) / (
                fingerprint + ".verified"
            )
            clauses.append(
                "if test -f {archive} && test -f {verified} && "
                "read saved_digest saved_size saved_mtime < {verified} && "
                'test "$saved_digest" = {fingerprint} && '
                'test "$saved_size" = {size} && '
                'test "$(stat -c %s {archive} 2>/dev/null)" = {size} && '
                'test "$saved_mtime" = "$(stat -c %Y {archive} 2>/dev/null)"; '
                'then printf "{fingerprint} hit\\n"; '
                'else printf "{fingerprint} miss\\n"; fi'.format(
                    archive=shlex.quote(str(archive)),
                    verified=shlex.quote(str(verified)),
                    fingerprint=shlex.quote(fingerprint),
                    size=size,
                )
            )
        code, output = session.execute_result("\n".join(clauses), timeout=120)
        if code != 0:
            continue
        for line in output.splitlines():
            pieces = line.split()
            if len(pieces) == 2 and pieces[1] in {"hit", "miss"}:
                verdicts[pieces[0]] = pieces[1] == "hit"
    return verdicts


def _ensure_remote_dataset_archive(
    session: SSHSession,
    paths: dict[str, str],
    profile: dict[str, Any],
    local_archive: Path,
    fingerprint: str,
    status_path: Path,
    progress_callback=None,
    assumed_cache_state: str | None = None,
) -> tuple[str, bool]:
    use_cache = bool(profile.get("cache_training_data", True))
    if use_cache:
        session.ensure_directory(paths["dataset_cache"])
        remote_archive = str(
            PurePosixPath(paths["dataset_cache"]) / (fingerprint + ".tar")
        )
        remote_verified = remote_archive[:-4] + ".verified"
        local_size = int(local_archive.stat().st_size)
        if assumed_cache_state == "hit":
            # Pre-classified by _batch_verify_remote_dataset_cache with the
            # same marker rule; skip the per-case round trip entirely.
            session.execute(
                "touch {verified}".format(
                    verified=shlex.quote(remote_verified),
                ),
                check=False,
            )
            if progress_callback is not None:
                progress_callback(local_size, local_size)
            _status_update(
                status_path,
                status="uploading",
                phase="remote_dataset_cache_hit",
                dataset_cache_hit=True,
                dataset_fingerprint=fingerprint,
            )
            return remote_archive, True
        if assumed_cache_state == "miss":
            # Stale or absent marker: drop both files so the fresh upload
            # below cannot collide with a half-written predecessor.
            session.execute(
                "rm -f {archive} {verified}".format(
                    archive=shlex.quote(remote_archive),
                    verified=shlex.quote(remote_verified),
                ),
                check=False,
            )
        code, output = session.execute_result(
            "if test -f {path}; then "
            "actual_size=$(stat -c %s {path} 2>/dev/null || printf 0); "
            "actual_mtime=$(stat -c %Y {path} 2>/dev/null || printf 0); "
            "if test -f {verified} && "
            "read saved_digest saved_size saved_mtime < {verified} && "
            "test \"$saved_digest\" = {fingerprint} && "
            "test \"$saved_size\" = {size} && "
            "test \"$actual_size\" = {size} && "
            "test \"$saved_mtime\" = \"$actual_mtime\"; then "
            "printf '%s' {fingerprint}; "
            "elif test \"$actual_size\" = {size}; then "
            "actual_digest=$(sha256sum {path} | awk '{{print $1}}'); "
            "if test \"$actual_digest\" = {fingerprint}; then "
            "tmp={verified}.$$.tmp; "
            "printf '%s %s %s\n' {fingerprint} {size} \"$actual_mtime\" > \"$tmp\" && "
            "mv -f \"$tmp\" {verified}; "
            "fi; printf '%s' \"$actual_digest\"; "
            "fi; fi".format(
                path=shlex.quote(remote_archive),
                verified=shlex.quote(remote_verified),
                fingerprint=shlex.quote(fingerprint),
                size=local_size,
            )
        )
        if code == 0 and output.strip() == fingerprint:
            session.execute(
                "touch {verified}".format(
                    verified=shlex.quote(remote_verified),
                ),
                check=False,
            )
            cache_progress = {}
            if progress_callback is not None:
                size = int(local_archive.stat().st_size)
                progress_callback(size, size)
            else:
                cache_progress = {
                    "transfer_bytes": 0,
                    "transfer_total_bytes": int(
                        local_archive.stat().st_size
                    ),
                    "transfer_percent": 100,
                    "progress_percent": 33,
                }
            _status_update(
                status_path,
                status="uploading",
                phase="remote_dataset_cache_hit",
                dataset_cache_hit=True,
                dataset_fingerprint=fingerprint,
                **cache_progress
            )
            return remote_archive, True
        if output.strip():
            session.execute(
                "rm -f {archive} {verified}".format(
                    archive=shlex.quote(remote_archive),
                    verified=shlex.quote(remote_verified),
                ),
                check=False,
            )
    else:
        remote_archive = (
            paths["archive"] + ".dataset." + fingerprint + ".tar"
        )

    _status_update(
        status_path,
        status="uploading",
        phase="uploading_training_dataset",
        dataset_cache_hit=False,
        dataset_fingerprint=fingerprint,
    )
    session.upload(
        local_archive,
        remote_archive,
        callback=progress_callback
        or _upload_progress(
            status_path,
            phase="uploading_training_dataset",
            start_percent=20,
            end_percent=33,
        ),
    )
    remote_digest = session.execute(
        "sha256sum {path} | awk '{{print $1}}'".format(
            path=shlex.quote(remote_archive)
        ),
        timeout=600,
    ).strip()
    if remote_digest != fingerprint:
        session.execute(
            "rm -f {archive}{verified}".format(
                archive=shlex.quote(remote_archive),
                verified=(
                    " " + shlex.quote(remote_verified)
                    if use_cache
                    else ""
                ),
            ),
            check=False,
        )
        raise RuntimeError(
            "The uploaded training-data archive failed SHA-256 verification."
        )
    if use_cache:
        session.execute(
            "tmp={verified}.$$.tmp; "
            "archive_mtime=$(stat -c %Y {archive}); "
            "printf '%s %s %s\n' {fingerprint} {size} \"$archive_mtime\" > \"$tmp\" && "
            "mv -f \"$tmp\" {verified}".format(
                verified=shlex.quote(remote_verified),
                archive=shlex.quote(remote_archive),
                fingerprint=shlex.quote(fingerprint),
                size=int(local_archive.stat().st_size),
            )
        )
    return remote_archive, False


def _download_progress(status_path: Path):
    state = {
        "last_time": 0.0,
        "last_percent": -1,
        "started_at": 0.0,
        "started_bytes": 0,
    }

    def callback(transferred: int, total: int) -> None:
        _raise_if_cancelled(status_path)
        now = time.time()
        if not state["started_at"]:
            state["started_at"] = now
            state["started_bytes"] = int(transferred)
        elapsed = max(0.001, now - float(state["started_at"]))
        measured_bytes = max(
            0, int(transferred) - int(state["started_bytes"])
        )
        bytes_per_second = (
            float(measured_bytes) / elapsed if measured_bytes else 0.0
        )
        eta_seconds = (
            max(0.0, int(total) - int(transferred)) / bytes_per_second
            if bytes_per_second > 0
            else None
        )
        percent = int(100.0 * transferred / max(1, total))
        if percent == state["last_percent"] and now - state["last_time"] < 1.0:
            return
        if now - state["last_time"] < 0.25 and percent < 100:
            return
        state["last_time"] = now
        state["last_percent"] = percent
        _status_update(
            status_path,
            status="downloading",
            phase="downloading_model",
            transfer_bytes=int(transferred),
            transfer_total_bytes=int(total),
            transfer_percent=percent,
            transfer_bytes_per_second=bytes_per_second,
            transfer_eta_seconds=eta_seconds,
            progress_percent=96 + int(percent * 0.03),
        )

    return callback


def _existing_parent(path: Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_dir():
        candidate = candidate.parent
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _ensure_local_download_capacity(
    session: SSHSession,
    remote_archive: str,
    local_archive: Path,
    local_target: Path,
    status_path: Path,
) -> int:
    size_text = session.execute(
        "stat -c %s {}".format(shlex.quote(remote_archive)),
        timeout=30,
    ).strip()
    try:
        archive_size = int(size_text)
    except Exception as exc:
        raise RuntimeError(
            "The remote model archive size could not be read before download."
        ) from exc
    if archive_size <= 0:
        raise RuntimeError("The remote model archive is empty.")

    local_archive.parent.mkdir(parents=True, exist_ok=True)
    local_target.parent.mkdir(parents=True, exist_ok=True)
    archive_parent = _existing_parent(local_archive.parent)
    target_parent = _existing_parent(local_target.parent)
    part_path = local_archive.with_name(local_archive.name + ".part")
    try:
        partial_size = min(archive_size, int(part_path.stat().st_size))
    except OSError:
        partial_size = 0
    remaining_download = max(0, archive_size - partial_size)
    margin = max(256 * 1024 * 1024, int(archive_size * 0.05))
    same_volume = os.stat(str(archive_parent)).st_dev == os.stat(
        str(target_parent)
    ).st_dev
    archive_free = int(shutil.disk_usage(str(archive_parent)).free)
    target_free = int(shutil.disk_usage(str(target_parent)).free)
    if same_volume:
        required = remaining_download + archive_size + margin
        sufficient = archive_free >= required
        free_bytes = archive_free
    else:
        archive_required = remaining_download + margin
        target_required = archive_size + margin
        sufficient = (
            archive_free >= archive_required and target_free >= target_required
        )
        required = max(archive_required, target_required)
        free_bytes = min(archive_free, target_free)
    _status_update(
        status_path,
        remote_artifact_bytes=archive_size,
        local_download_required_bytes=required,
        local_download_free_bytes=free_bytes,
        local_download_space_checked=True,
    )
    if not sufficient:
        raise RuntimeError(
            "Not enough local disk space to download and unpack the trained "
            "model safely. Need approximately {:.2f} GiB; only {:.2f} GiB is "
            "free. Free space on the model workspace drive, then retry the "
            "download.".format(
                required / float(1024 ** 3), free_bytes / float(1024 ** 3)
            )
        )
    return archive_size


def _stop_remote(
    session: SSHSession,
    container_name: str,
    status_path: Path,
    final: str = "cancelled",
    remote_control_path: str = "",
    remote_control_kind: str = "",
    remote_job_dir: str = "",
    remote_root: str = "",
    expected_owner: str = "",
    expected_job: str = "",
) -> None:
    job = _validate_remote_job_path(
        remote_root,
        expected_owner,
        remote_job_dir,
    )
    _validate_remote_control_path(str(job), remote_control_path)
    _status_update(
        status_path,
        status="stopping",
        phase="stopping_remote_container",
        remote_stop_confirmed=False,
    )
    container_exists = _assert_container_owned(
        session,
        container_name,
        expected_owner=expected_owner,
        expected_job=expected_job,
    )
    if remote_control_path and container_exists:
        parent = str(PurePosixPath(remote_control_path).parent)
        if remote_control_kind == "json":
            content = json.dumps(
                {"action": "stop", "requested_at_epoch": time.time()},
                sort_keys=True,
            )
        else:
            content = "cancel\n"
        session.execute(
            "mkdir -p {parent} && printf %s {content} > {temporary} && "
            "mv -f {temporary} {target}".format(
                parent=shlex.quote(parent),
                content=shlex.quote(content),
                temporary=shlex.quote(remote_control_path + ".tmp"),
                target=shlex.quote(remote_control_path),
            ),
            timeout=30,
        )
        graceful_deadline = time.time() + 20.0
        while time.time() < graceful_deadline:
            state, _code = _container_state(session, container_name)
            if state in {"exited", "dead", "missing"}:
                break
            time.sleep(1.0)
    if container_exists:
        session.execute(
            "{runtime} stop --time 20 {container} >/dev/null 2>&1 || true".format(
                runtime=container_runtime_command(getattr(session, "profile", None)),
                container=shlex.quote(container_name),
            ),
            timeout=40,
            check=False,
        )
    state, _code = _container_state(session, container_name)
    confirmed = state in {"exited", "dead", "missing", "created"}
    _status_update(
        status_path,
        status=final if confirmed else "stopping",
        phase=final if confirmed else "remote_termination_pending",
        remote_stop_confirmed=confirmed,
        completed_at_epoch=time.time() if confirmed else None,
        error=""
        if confirmed
        else "The remote container did not confirm termination.",
    )
    if not confirmed:
        raise RuntimeError("The remote container did not confirm termination.")
    container_removed = _remove_container(
        session,
        container_name,
        expected_owner=expected_owner,
        expected_job=expected_job,
        stop_if_running=False,
    )
    job_removed = _remove_remote_job(
        session,
        remote_root=remote_root,
        expected_owner=expected_owner,
        remote_job_dir=str(job),
    )
    _status_update(
        status_path,
        remote_container_removed=bool(container_removed),
        remote_job_removed=bool(job_removed),
        remote_diagnostics_retained=not bool(job_removed),
        remote_cleanup_warning=(
            ""
            if container_removed and job_removed
            else "Training stopped, but some remote files could not be removed."
        ),
    )


def _sync_remote_log(
    session: SSHSession,
    remote_job: str,
    local_log: Path,
    state: dict[str, Any],
) -> None:
    remote_relative = str(state.get("remote_relative") or "remote_worker.log")
    remote_log = str(PurePosixPath(remote_job) / PurePosixPath(remote_relative))
    if session.path_exists(remote_log):
        try:
            identity = session.execute(
                "stat -c '%i' {}".format(shlex.quote(remote_log)),
                timeout=30,
            ).strip()
        except Exception:
            identity = ""
        previous_identity = str(state.get("remote_log_identity") or "")
        if identity and previous_identity and identity != previous_identity:
            state["offset"] = 0
            _append_log(
                local_log,
                "Remote worker log rotated; continuing with the new log file.",
            )
        if identity:
            state["remote_log_identity"] = identity
        state["offset"] = session.download_appended(
            remote_log,
            local_log,
            int(state.get("offset") or 0),
        )


def _download_artifact(
    session: SSHSession,
    remote_job: str,
    relative: str,
    local_target: Path,
    local_archive: Path,
    status_path: Path,
) -> None:
    artifact = str(PurePosixPath(remote_job) / PurePosixPath(relative))
    result_tar = str(PurePosixPath(remote_job) / "result.tar")
    session.execute(
        "test -d {artifact} && tar -cf {archive} -C {artifact} .".format(
            artifact=shlex.quote(artifact),
            archive=shlex.quote(result_tar),
        ),
        timeout=300,
    )
    remote_archive_sha256 = session.execute(
        "sha256sum {archive} | awk '{{print $1}}'".format(
            archive=shlex.quote(result_tar)
        ),
        timeout=300,
    ).strip()
    if len(remote_archive_sha256) != 64:
        raise RuntimeError(
            "The remote model archive checksum could not be read."
        )
    archive_size = _ensure_local_download_capacity(
        session,
        result_tar,
        local_archive,
        local_target,
        status_path,
    )
    _status_update(
        status_path,
        status="downloading",
        phase="downloading_model",
        progress_percent=96,
        transfer_total_bytes=archive_size,
    )
    session.download(
        result_tar,
        local_archive,
        callback=_download_progress(status_path),
    )
    local_archive_sha256 = _sha256_file(local_archive)
    if local_archive_sha256 != remote_archive_sha256:
        try:
            local_archive.unlink()
        except OSError:
            pass
        raise RuntimeError(
            "The downloaded model archive failed SHA-256 verification."
        )
    _status_update(
        status_path,
        remote_model_archive_sha256=remote_archive_sha256,
    )
    temporary = local_target.with_name(local_target.name + ".remote-part")
    backup = local_target.with_name(
        "{}.remote-backup-{}".format(local_target.name, uuid.uuid4().hex)
    )
    shutil.rmtree(str(temporary), ignore_errors=True)
    shutil.rmtree(str(backup), ignore_errors=True)
    _safe_extract(local_archive, temporary)
    had_existing = local_target.exists()
    published = False
    try:
        if had_existing:
            os.replace(str(local_target), str(backup))
        os.replace(str(temporary), str(local_target))
        published = True
    except Exception as publish_exc:
        restore_error = None
        if had_existing and backup.exists() and not local_target.exists():
            try:
                os.replace(str(backup), str(local_target))
            except Exception as exc:
                restore_error = exc
        if restore_error is not None:
            raise RuntimeError(
                "Could not publish the downloaded artifact and automatic rollback "
                "also failed. The previous artifact is preserved at '{}'. "
                "Publish error: {}; rollback error: {}".format(
                    backup, publish_exc, restore_error
                )
            )
        raise
    finally:
        shutil.rmtree(str(temporary), ignore_errors=True)
    if published:
        shutil.rmtree(str(backup), ignore_errors=True)




def _register_nninteractive(
    spec: dict[str, Any],
    local_model_dir: Path,
    remote_status: dict[str, Any],
    profile: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    import tools.nninteractive_finetune_pipeline as pipeline

    local_job_dir = Path(spec["job_dir"]).resolve()
    request = read_json(local_job_dir / "request.json", {}) or {}
    workspace = Path(request["workspace"]).resolve()
    quality = remote_status.get("quality") or None
    validation_count = sum(
        row.get("split") == "val" for row in request.get("cases") or []
    )
    model, selected = pipeline._register_model(
        request, workspace, local_model_dir, quality, validation_count
    )
    model.update(
        {
            "execution_backend": "remote",
            "remote_profile_id": profile["profile_id"],
            "remote_profile_name": profile["name"],
            "remote_runtime_image": profile["runtime_image"],
            "remote_gpu_device": remote_status.get(
                "remote_gpu_device", profile.get("gpu_device") or "auto"
            ),
            "remote_dataset_fingerprint": remote_status.get(
                "dataset_fingerprint", ""
            ),
            "remote_runtime_image_id": remote_status.get(
                "remote_runtime_image_id", ""
            ),
            "remote_base_model_sha256": remote_status.get(
                "remote_base_model_sha256", ""
            ),
            "local_base_model_sha256": remote_status.get(
                "local_base_model_sha256", ""
            ),
        }
    )
    pipeline.write_json_atomic(
        local_model_dir / "integration_manifest.json", model
    )
    registry = pipeline.load_registry(workspace)
    for task in registry.get("tasks") or []:
        if pipeline.safe_slug(task.get("task_id")) != pipeline.safe_slug(
            request["task_id"]
        ):
            continue
        for row in task.get("models") or []:
            if row.get("model_id") == model.get("model_id"):
                row.update(model)
    pipeline.save_registry(workspace, registry)
    return model, selected


def _register_nnunet(
    spec: dict[str, Any],
    local_model_dir: Path,
    remote_status: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    import tools.nnunet_pipeline as pipeline
    from tools.nnunet_common import read_json as read_nnunet_json

    request = read_nnunet_json(Path(spec["request_path"]).resolve(), {}) or {}
    return pipeline.register_downloaded_model(
        request,
        local_model_dir,
        {
            "remote_profile_id": profile["profile_id"],
            "remote_profile_name": profile["name"],
            "remote_runtime_image": profile["runtime_image"],
            "remote_gpu_device": remote_status.get(
                "remote_gpu_device", profile.get("gpu_device") or "auto"
            ),
            "remote_dataset_fingerprint": remote_status.get(
                "dataset_fingerprint", ""
            ),
            "remote_runtime_image_id": remote_status.get(
                "remote_runtime_image_id", ""
            ),
            "downloaded_at_epoch": time.time(),
        },
    )


def _sync_remote_nnunet_curve(
    session: SSHSession,
    remote_job: str,
    remote_status: dict[str, Any],
    local_curve: Path,
    state: dict[str, Any],
) -> bool:
    container_path = str(remote_status.get("training_curve_path") or "")
    if not container_path.startswith("/job/"):
        return False
    relative = PurePosixPath(container_path[len("/job/") :])
    if any(part == ".." for part in relative.parts):
        return False
    remote_curve = str(PurePosixPath(remote_job) / relative)
    try:
        stat = session.sftp.stat(remote_curve)
    except IOError:
        return False
    identity = (
        int(getattr(stat, "st_size", 0)),
        int(getattr(stat, "st_mtime", 0)),
    )
    if identity == state.get("curve_identity") and local_curve.is_file():
        return False
    session.download(remote_curve, local_curve)
    state["curve_identity"] = identity
    return True


def _monitor_remote(
    session: SSHSession | None,
    spec: dict[str, Any],
    profile: dict[str, Any],
    status_path: Path,
    remote_paths: dict[str, str],
    container_name: str,
    owner: str,
    controller_log: Path,
    remote_log: Path,
) -> tuple[SSHSession | None, dict[str, Any]]:
    """Poll the remote job until it reaches a terminal pipeline status.

    Shared by the fresh-run path and the re-attach path. Returns the (live
    or reconnected) session and the last remote status payload so the caller
    can download and register the artifact (or report the failure).
    """
    kind = str(spec.get("kind") or "")
    job_slug = safe_identifier(str(spec["job_id"]), "job")
    remote_status_path = str(
        PurePosixPath(remote_paths["job"]) / _remote_status_relative(kind)
    )
    status = read_json(status_path, {}) or {}
    remote_control_path = str(status.get("remote_control_path") or "")
    remote_control_kind = str(status.get("remote_control_kind") or "")
    if not remote_control_path:
        remote_control_path = str(
            PurePosixPath(remote_paths["job"]) / "pipeline_job/control.json"
        )
        remote_control_kind = "json"
    missing_status_since = time.time()
    reconnect_attempt = 0
    log_sync_state = {
        "offset": 0,
        "remote_relative": _remote_log_relative(kind),
        "last_curve_sync_epoch": 0.0,
    }
    remote_status: dict[str, Any] = {}
    # Stall detection: any observable progress (new status content or worker
    # log growth) refreshes this timestamp. Training can legitimately run for
    # days, so silence alone never fails the task — it only surfaces a hint
    # in status after a full day without any sign of life.
    last_activity_epoch = time.time()
    stall_hint_active = False
    last_worker_pid = ""
    # Budget for docker-control failures, mirroring the reconnect budget:
    # without it a persistently failing docker socket keeps this controller
    # process alive forever with the job status stuck in
    # remote_control_unavailable.
    docker_error_attempt = 0
    while True:
        try:
            _raise_if_abandoned(status_path)
            if session is None:
                raise RemoteComputeError("No active SSH monitoring session.")
            log_offset_before = int(log_sync_state.get("offset") or 0)
            try:
                _sync_remote_log(
                    session,
                    remote_paths["job"],
                    remote_log,
                    log_sync_state,
                )
            except IOError:
                pass
            if int(log_sync_state.get("offset") or 0) > log_offset_before:
                last_activity_epoch = time.time()
                if stall_hint_active:
                    stall_hint_active = False
                    _status_update(status_path, remote_stall_suspected=False)
            # A remote read can outlive the UI action that requested a
            # local abandon. Recheck before any active state is written.
            _raise_if_abandoned(status_path)
            if _cancel_requested(status_path):
                _stop_remote(
                    session,
                    container_name,
                    status_path,
                    final="cancelled",
                    remote_control_path=remote_control_path,
                    remote_control_kind=remote_control_kind,
                    remote_job_dir=remote_paths["job"],
                    remote_root=profile["remote_root"],
                    expected_owner=owner,
                    expected_job=job_slug,
                )
                raise InterruptedError("cancel")
            if _pause_requested(status_path) and kind == "nninteractive":
                if remote_control_path:
                    temporary = status_path.parent / "pause-control.json"
                    write_json_atomic(
                        temporary,
                        {"action": "pause", "requested_at_epoch": time.time()},
                    )
                    session.upload(temporary, remote_control_path)
                _status_update(
                    status_path,
                    status="pausing",
                    phase="pausing_remote_training",
                )
            candidate = session.read_remote_json(remote_status_path, {}) or {}
            if candidate:
                missing_status_since = time.time()
                if candidate != remote_status:
                    last_activity_epoch = time.time()
                    if stall_hint_active:
                        stall_hint_active = False
                        _status_update(
                            status_path, remote_stall_suspected=False
                        )
                remote_status = candidate
                _merge_remote_status(
                    status_path,
                    candidate,
                    profile,
                    container_name,
                    remote_paths["job"],
                )
                if kind in {"nnunet", "flexict"} and (
                    time.time()
                    - float(log_sync_state.get("last_curve_sync_epoch") or 0)
                    >= 5.0
                ):
                    log_sync_state["last_curve_sync_epoch"] = time.time()
                    try:
                        local_curve = status_path.parent / "remote_progress.png"
                        if _sync_remote_nnunet_curve(
                            session,
                            remote_paths["job"],
                            candidate,
                            local_curve,
                            log_sync_state,
                        ) or local_curve.is_file():
                            _status_update(
                                status_path,
                                training_curve_path=str(local_curve),
                            )
                    except (IOError, OSError, RemoteComputeError):
                        pass
                state = str(candidate.get("status") or "").lower()
                if state in TERMINAL:
                    break
            else:
                worker = session.read_remote_json(
                    str(
                        PurePosixPath(remote_paths["job"])
                        / "worker_status.json"
                    ),
                    {},
                ) or {}
                if str(worker.get("status") or "") == "waiting_for_remote_gpu":
                    missing_status_since = time.time()
                    worker_pid = str(worker.get("worker_pid") or "")
                    if worker_pid != last_worker_pid:
                        last_worker_pid = worker_pid
                        last_activity_epoch = time.time()
                        if stall_hint_active:
                            stall_hint_active = False
                            _status_update(
                                status_path, remote_stall_suspected=False
                            )
                    _status_update(
                        status_path,
                        status="waiting_for_remote_gpu",
                        phase="waiting_for_remote_gpu",
                        progress_percent=35,
                        remote_worker_pid=worker.get("worker_pid"),
                    )
            container_state, container_code = _container_state(
                session, container_name
            )
            if container_state in {"exited", "dead", "missing"}:
                worker = session.read_remote_json(
                    str(
                        PurePosixPath(remote_paths["job"])
                        / "worker_status.json"
                    ),
                    {},
                ) or {}
                if not remote_status:
                    remote_status = worker
                if (
                    str(remote_status.get("status") or "").lower()
                    not in TERMINAL
                ):
                    raise RuntimeError(
                        "Remote training container exited with code {} before "
                        "writing a terminal pipeline status.".format(
                            container_code
                        )
                    )
                break
            if time.time() - missing_status_since > 300:
                raise RuntimeError(
                    "Remote training started but did not create its status "
                    "file within five minutes."
                )
            if (
                not stall_hint_active
                and time.time() - last_activity_epoch > REMOTE_STALL_SECONDS
            ):
                stall_hint_active = True
                _append_log(controller_log, REMOTE_STALL_HINT)
                _status_update(
                    status_path,
                    remote_stall_suspected=True,
                    remote_stall_detail=REMOTE_STALL_HINT,
                )
            reconnect_attempt = 0
            docker_error_attempt = 0
        except RemoteCommandError as exc:
            _append_log(
                controller_log,
                "Remote Docker control is temporarily unavailable: {}".format(
                    exc
                ),
            )
            docker_error_attempt += 1
            _status_update(
                status_path,
                status="remote_control_unavailable",
                phase="remote_control_unavailable",
                remote_state_unknown=True,
                remote_connection_error=str(exc),
                remote_docker_error_attempt=docker_error_attempt,
            )
            if docker_error_attempt >= RECONNECT_MAX_ATTEMPTS:
                raise RemoteServerUnreachable(
                    "Remote Docker control failed {} times in a row ({}). "
                    "The remote container may still be running; use "
                    "Re-attach to retry monitoring once Docker is reachable "
                    "again, or Stop/Abandon Locally to end this task.".format(
                        docker_error_attempt, exc
                    )
                )
            deadline = time.time() + 10.0
            while time.time() < deadline:
                _raise_if_cancelled(status_path)
                time.sleep(
                    min(0.5, max(0.0, deadline - time.time()))
                )
            continue
        except (RemoteComputeError, EOFError, OSError, socket.error) as exc:
            try:
                session.close()
            except Exception:
                pass
            session = None
            reconnect_attempt += 1
            session = _reconnect_session(
                profile,
                status_path,
                controller_log,
                exc,
                reconnect_attempt,
            )
            continue
        time.sleep(2.0)
    return session, remote_status


def run(spec_path: Path) -> int:
    spec = read_json(spec_path, {}) or {}
    kind = str(spec.get("kind") or "")
    status_path = Path(spec["status_path"]).resolve()
    profile = _profile_for_spec(spec)
    job_id = str(spec["job_id"])
    owner = _remote_owner(profile)
    job_slug = safe_identifier(job_id, "job")
    container_name = _container_name(profile, job_id)
    local_root = status_path.parent / (job_id + "_remote")
    bundle = local_root / "bundle"
    archive = local_root / "job_upload.tar"
    dataset_archive_dir = local_root / "dataset_parts"
    result_archive = local_root / "result.tar"
    abandon_path = status_path.with_name(
        status_path.name + ".abandon.request"
    )
    try:
        abandon_path.unlink()
    except FileNotFoundError:
        pass
    if kind == "nninteractive":
        remote_log = status_path.parent / "job.log"
        controller_log = remote_log
    else:
        remote_log = local_root / "remote_training.log"
        controller_log = status_path.with_name(
            status_path.name + ".remote_controller.log"
        )
    shutil.rmtree(str(local_root), ignore_errors=True)
    bundle.mkdir(parents=True, exist_ok=True)
    _status_update(
        status_path,
        execution_backend="remote",
        abandon_path=str(abandon_path),
        local_abandon_requested=False,
        remote_profile_id=profile["profile_id"],
        profile_name=profile["name"],
        remote_container_name=container_name,
        controller_pid=os.getpid(),
        local_log_path=str(remote_log),
        controller_log_path=str(controller_log),
        diagnostic_log_paths=list(
            dict.fromkeys((str(remote_log), str(controller_log)))
        ),
        train_log=str(remote_log) if kind in {"nnunet", "flexict"} else "",
        log_path=str(
            remote_log
        ) if kind in {"nninteractive", "nnunet", "nnunet_infer", "flexict", "flexict_infer"} else "",
        remote_gpu_device=profile.get("gpu_device") or "auto",
        status="preparing_remote",
        phase="preparing_remote_data",
        progress_percent=1,
    )
    session: SSHSession | None = None
    remote_status: dict[str, Any] = {}
    prepared: dict[str, Any] = {}
    remote_paths: dict[str, str] = {}
    asset_identity: dict[str, str] = {}
    runtime_image_id = ""
    container_may_exist = False
    prepared_cache_namespace = ""
    prepared_cache_removed = False
    try:
        if kind == "nninteractive":
            prepared = _prepare_nninteractive(spec, bundle)
        elif kind == "nnunet":
            prepared = _prepare_nnunet(spec, bundle)
        elif kind == "nnunet_infer":
            prepared = _prepare_nnunet_infer(spec, bundle)
        elif kind == "flexict":
            prepared = _prepare_flexict(spec, bundle)
        elif kind == "flexict_infer":
            prepared = _prepare_flexict_infer(spec, bundle)
        else:
            raise RuntimeError("Unsupported remote controller kind: {}".format(kind))
        if _cancel_requested(status_path):
            raise InterruptedError("cancel")
        dataset_parts = _build_dataset_parts(
            bundle,
            dataset_archive_dir,
            cache_dir=(
                Path(prepared["local_dataset_archive_cache"])
                if prepared.get("local_dataset_archive_cache")
                else None
            ),
            case_cache_keys=dict(
                prepared.get("dataset_case_cache_keys") or {}
            ),
        )
        local_archive_cache_hits = sum(
            bool(part.get("local_cache_hit")) for part in dataset_parts
        )
        _append_log(
            controller_log,
            "Local remote-data archives: reused {} of {} case(s).".format(
                local_archive_cache_hits, len(dataset_parts)
            ),
        )
        dataset_fingerprint = _dataset_parts_fingerprint(dataset_parts)
        prepared_cache_identity = dataset_fingerprint
        if not bool(profile.get("cache_training_data", True)):
            prepared_cache_identity = "{}_{}".format(
                dataset_fingerprint, job_slug
            )
        _bind_remote_prepared_cache(bundle, prepared_cache_identity)
        dataset_archive_size = sum(
            int(part["size"]) for part in dataset_parts
        )
        archive_size = _build_tar(
            bundle,
            archive,
            excluded_top_level={"input", "labels"},
        )
        _status_update(
            status_path,
            status="connecting_remote",
            phase="connecting_remote",
            transfer_total_bytes=archive_size + dataset_archive_size,
            dataset_fingerprint=dataset_fingerprint,
            dataset_archive_bytes=dataset_archive_size,
            local_dataset_archive_cache_reused=local_archive_cache_hits,
            local_dataset_archive_cache_total=len(dataset_parts),
            progress_percent=20,
        )
        _raise_if_cancelled(status_path)
        session = SSHSession(profile)
        _raise_if_cancelled(status_path)
        remote_paths = _remote_paths(session, job_id)
        _validate_remote_job_path(
            profile["remote_root"],
            owner,
            remote_paths["job"],
        )
        session.ensure_directory(remote_paths["jobs"])
        session.ensure_directory(remote_paths["locks"])
        session.ensure_directory(remote_paths["dataset_cache"])
        session.ensure_directory(remote_paths["prepared_cache"])
        # Shared-server "check before use": the launch-time GPU busy gate
        # runs before any upload so a blocked job costs one SSH round trip,
        # not hours of transfer. Warn mode surfaces in the controller log
        # and the status file (the log alone is invisible in the UI).
        gpu_busy_note = assert_gpus_not_busy(session, profile)
        if gpu_busy_note:
            _append_log(controller_log, gpu_busy_note)
            _status_update(status_path, gpu_busy_warning=gpu_busy_note)
        if kind in {"nnunet", "nnunet_infer", "flexict", "flexict_infer"}:
            prepared_cache_namespace = _prepared_cache_namespace(
                bundle, remote_paths
            )
            _maintain_remote_nnunet_cache(
                session,
                profile,
                remote_paths,
                prepared_cache_namespace,
                status_path,
                controller_log,
            )
        session.execute(
            "{runtime} container prune -f "
            "--filter label=mimics-script.remote-training=true "
            "--filter {owner} >/dev/null 2>&1 || true".format(
                runtime=container_runtime_command(profile),
                owner=shlex.quote(
                    "label=mimics-script.owner={}".format(
                        safe_identifier(profile.get("username"), "user")
                    )
                )
            ),
            check=False,
        )
        runtime_image_id = session.execute(
            "{runtime} image inspect --format '{{{{.Id}}}}' {image}".format(
                runtime=container_runtime_command(profile),
                image=shlex.quote(profile["runtime_image"]),
            )
        ).strip()
        asset_identity = _validate_remote_assets(
            session,
            remote_paths,
            prepared,
            verify_mode=profile.get("remote_weights_verify", "strict"),
        )
        # Refuse (strict) or record (warn) when the image's baked-in pipeline
        # code differs from the local checkout — a stale image silently
        # trains with old code otherwise.
        code_identity = _check_code_drift(
            session,
            profile,
            status_path,
            runtime_image_id,
        )
        identity = dict(asset_identity)
        identity.update(code_identity)
        # _check_code_drift already reports remote_runtime_image_id when the
        # check runs; keep the inspected id for the "off" mode too without
        # ever passing the same keyword twice.
        identity.setdefault("remote_runtime_image_id", runtime_image_id)
        _status_update(
            status_path,
            remote_runtime_image=profile["runtime_image"],
            **identity
        )
        # Dataset cache entries are immutable and content-addressed. Job
        # directories are never age-pruned here because a long-running job's
        # parent mtime can remain unchanged while nested status files update.
        # .part/.tmp files are interrupted-upload fragments: never valid
        # cache entries, so any that are old enough not to be an in-flight
        # transfer are removed.
        session.execute(
            "find {cache} -maxdepth 1 -type f -name '*.verified' -mtime +30 "
            "-print | while IFS= read -r marker; do "
            "rm -f \"$marker\" \"${{marker%.verified}}.tar\"; done; "
            "find {cache} -maxdepth 1 -type f -name '*.tar' -mtime +30 "
            "-print | while IFS= read -r archive; do "
            "test -f \"${{archive%.tar}}.verified\" || rm -f \"$archive\"; "
            "done; "
            "find {cache} -maxdepth 1 -type f "
            "\\( -name '*.part' -o -name '*.tmp' \\) -mtime +7 "
            "-print -delete".format(
                cache=shlex.quote(remote_paths["dataset_cache"])
            ),
            check=False,
        )
        remote_dataset_archives = []
        dataset_cache_hits = 0
        dataset_bytes_completed = 0
        dataset_reconnect_attempt = 0
        pending_dataset_parts = list(dataset_parts)
        # One batched round trip classifies all case cache slots up front
        # (fallback documented in the design notes instead of concurrent
        # SFTP streams): previously each case cost a verify round trip even
        # on a hit, and a cache-warm 100-case dataset paid 100 sequential
        # SSH execs before any upload decision.
        cache_verdicts: dict[str, bool] = {}
        if pending_dataset_parts and bool(
            profile.get("cache_training_data", True)
        ):
            try:
                cache_verdicts = _batch_verify_remote_dataset_cache(
                    session, remote_paths, pending_dataset_parts
                )
            except Exception as batch_exc:
                _append_log(
                    controller_log,
                    "Batched dataset-cache verification failed ({}); "
                    "falling back to per-case verification.".format(batch_exc),
                )
        while pending_dataset_parts:
            _raise_if_cancelled(status_path)
            try:
                if session is None:
                    raise RemoteComputeError(
                        "No active SSH dataset-upload session."
                    )
                part = pending_dataset_parts[0]
                verification_index = len(remote_dataset_archives) + 1
                _status_update(
                    status_path,
                    status="uploading",
                    phase="verifying_remote_dataset_cache",
                    dataset_case_completed=len(remote_dataset_archives),
                    dataset_case_total=len(dataset_parts),
                    current_case=part.get("case_id"),
                )
                if (
                    verification_index == 1
                    or verification_index == len(dataset_parts)
                    or verification_index % 10 == 0
                ):
                    _append_log(
                        controller_log,
                        "Remote data cache verification: {} of {} cases.".format(
                            verification_index, len(dataset_parts)
                        ),
                    )
                remote_dataset_archive, part_cache_hit = (
                    _ensure_remote_dataset_archive(
                        session,
                        remote_paths,
                        profile,
                        Path(part["archive"]),
                        str(part["fingerprint"]),
                        status_path,
                        progress_callback=_upload_progress(
                            status_path,
                            phase="uploading_training_dataset",
                            start_percent=20,
                            end_percent=33,
                            aggregate_offset=dataset_bytes_completed,
                            aggregate_total=dataset_archive_size,
                        ),
                        assumed_cache_state=(
                            "hit"
                            if cache_verdicts.get(str(part["fingerprint"]))
                            else (
                                "miss"
                                if str(part["fingerprint"]) in cache_verdicts
                                else None
                            )
                        ),
                    )
                )
                remote_dataset_archives.append(remote_dataset_archive)
                if part_cache_hit:
                    dataset_cache_hits += 1
                dataset_bytes_completed += int(part["size"])
                pending_dataset_parts.pop(0)
                _status_update(
                    status_path,
                    dataset_fingerprint=dataset_fingerprint,
                    dataset_case_total=len(dataset_parts),
                    dataset_case_completed=len(remote_dataset_archives),
                    dataset_transfer_bytes=dataset_bytes_completed,
                    dataset_transfer_total_bytes=dataset_archive_size,
                    progress_percent=20
                    + int(
                        13
                        * dataset_bytes_completed
                        / max(1, dataset_archive_size)
                    ),
                    dataset_cache_hits=dataset_cache_hits,
                    dataset_cache_hit=(
                        bool(dataset_parts)
                        and dataset_cache_hits == len(dataset_parts)
                    ),
                )
            except RemoteCommandError:
                raise
            except (RemoteComputeError, EOFError, OSError, socket.error) as exc:
                try:
                    session.close()
                except Exception:
                    pass
                session = None
                dataset_reconnect_attempt += 1
                session = _reconnect_session(
                    profile,
                    status_path,
                    controller_log,
                    exc,
                    dataset_reconnect_attempt,
                )
        # Custom base-model upload cache: a repeated multi-gigabyte custom
        # model selection normally rides inside every job archive. Consult
        # the content-addressed remote cache first; on a hit exclude it from
        # the job tar (near-zero upload), on a miss upload it into the cache
        # once so every later run skips those bytes.
        custom_model_fingerprint = str(
            prepared.get("custom_model_fingerprint") or ""
        )
        custom_model_source = bundle / "custom_model"
        custom_model_cached = False
        if custom_model_fingerprint and custom_model_source.exists():
            try:
                custom_model_cached = _ensure_remote_custom_model(
                    session,
                    remote_paths,
                    profile,
                    custom_model_source,
                    custom_model_fingerprint,
                    status_path,
                    controller_log,
                )
            except (RemoteCommandError, RemoteComputeError) as exc:
                # Cache trouble must not block a job that would otherwise
                # work: fall back to shipping the model in the job archive.
                _append_log(
                    controller_log,
                    "Custom base-model cache check failed ({}); shipping the "
                    "model inside the job archive instead.".format(exc),
                )
        if custom_model_cached:
            archive_size = _build_tar(
                bundle,
                archive,
                excluded_top_level={"input", "labels", "custom_model"},
            )
        upload_reconnect_attempt = 0
        while True:
            _raise_if_cancelled(status_path)
            try:
                if session is None:
                    raise RemoteComputeError("No active SSH upload session.")
                session.upload(
                    archive,
                    remote_paths["archive"],
                    callback=_upload_progress(
                        status_path,
                        phase="uploading_job_configuration",
                        start_percent=33,
                        end_percent=35,
                    ),
                )
                break
            except RemoteCommandError:
                raise
            except (RemoteComputeError, EOFError, OSError, socket.error) as exc:
                try:
                    session.close()
                except Exception:
                    pass
                session = None
                upload_reconnect_attempt += 1
                session = _reconnect_session(
                    profile,
                    status_path,
                    controller_log,
                    exc,
                    upload_reconnect_attempt,
                )
        _status_update(
            status_path,
            status="starting_remote",
            phase="starting_remote_container",
            progress_percent=35,
            remote_job_dir=remote_paths["job"],
        )
        # Re-check right before launch: the dataset upload above can take
        # hours, and the moment this job starts crowding the shared GPU is
        # the container launch. Before the reconnect loop, not inside it,
        # so a reconnect never re-triggers the gate. Only block mode raises;
        # warn mode already surfaced at the pre-upload check.
        assert_gpus_not_busy(session, profile)
        launch_reconnect_attempt = 0
        while True:
            _raise_if_cancelled(status_path)
            try:
                if session is None:
                    raise RemoteComputeError("No active SSH launch session.")
                # Refuse a same-name container owned by another user/job before
                # considering this launch capable of creating remote work.
                existing_container = _assert_container_owned(
                    session,
                    container_name,
                    expected_owner=owner,
                    expected_job=job_slug,
                )
                if existing_container:
                    existing_state, _existing_code = _container_state(
                        session, container_name
                    )
                    if existing_state not in {"exited", "dead", "created"}:
                        raise RuntimeError(
                            "This remote training job already has a running "
                            "container. Open its status instead of starting a "
                            "duplicate controller."
                        )
                container_may_exist = True
                container_id = _launch_container(
                    session,
                    remote_paths,
                    profile,
                    container_name,
                    dataset_cache_paths=remote_dataset_archives,
                    custom_model_cache_path=(
                        _remote_custom_model_cache_path(
                            remote_paths, custom_model_fingerprint
                        )[0]
                        if custom_model_cached
                        else ""
                    ),
                    remove_dataset_after_extract=not bool(
                        profile.get("cache_training_data", True)
                    ),
                    status_path=status_path,
                    log_path=controller_log,
                )
                break
            except RemoteCommandError:
                raise
            except (RemoteComputeError, EOFError, OSError, socket.error) as exc:
                try:
                    session.close()
                except Exception:
                    pass
                session = None
                launch_reconnect_attempt += 1
                session = _reconnect_session(
                    profile,
                    status_path,
                    controller_log,
                    exc,
                    launch_reconnect_attempt,
                )
                if session is None:
                    continue
                state, _code = _container_state(session, container_name)
                if state != "missing":
                    _assert_container_owned(
                        session,
                        container_name,
                        expected_owner=owner,
                        expected_job=job_slug,
                    )
                    container_id = session.execute(
                        "{runtime} inspect --format '{{{{.Id}}}}' {container}".format(
                            runtime=container_runtime_command(
                                getattr(session, "profile", None)
                            ),
                            container=shlex.quote(container_name),
                        )
                    ).strip()
                    break
        _status_update(
            status_path,
            remote_container_id=container_id,
            remote_job_dir=remote_paths["job"],
            remote_gpu_device=profile.get("gpu_device") or "auto",
            dataset_cache_hit=bool(
                dataset_parts
                and dataset_cache_hits == len(dataset_parts)
            ),
            dataset_cache_hits=dataset_cache_hits,
            dataset_case_total=len(dataset_parts),
            remote_dataset_cache_paths=(
                remote_dataset_archives
                if profile.get("cache_training_data", True)
                else []
            ),
            remote_dataset_cache_path=(
                remote_dataset_archives[0]
                if (
                    len(remote_dataset_archives) == 1
                    and profile.get("cache_training_data", True)
                )
                else ""
            ),
            status="starting_remote",
        )
        remote_status_path = str(
            PurePosixPath(remote_paths["job"])
            / PurePosixPath(prepared["remote_status_relative"])
        )
        remote_control_path = (
            str(
                PurePosixPath(remote_paths["job"])
                / PurePosixPath(prepared["remote_control_relative"])
            )
            if prepared.get("remote_control_relative")
            else ""
        )
        remote_control_kind = str(
            prepared.get("remote_control_kind") or ""
        )
        _status_update(
            status_path,
            remote_control_path=remote_control_path,
            remote_control_kind=remote_control_kind,
        )
        # Failure-path fallback: if _monitor_remote raises before returning,
        # the post-mortem log sync below must not hit an unbound name.
        log_sync_state = {
            "offset": 0,
            "remote_relative": _remote_log_relative(kind),
            "last_curve_sync_epoch": 0.0,
        }
        session, remote_status = _monitor_remote(
            session,
            spec,
            profile,
            status_path,
            remote_paths,
            container_name,
            owner,
            controller_log,
            remote_log,
        )

        final_state = str(remote_status.get("status") or "").lower()
        if final_state != "completed":
            try:
                _sync_remote_log(
                    session,
                    remote_paths["job"],
                    remote_log,
                    log_sync_state,
                )
            except Exception as exc:
                _append_log(
                    controller_log,
                    "Could not synchronize the remote diagnostic log: {}".format(
                        exc
                    ),
                )
            _merge_remote_status(
                status_path,
                remote_status,
                profile,
                container_name,
                remote_paths["job"],
            )
            removed = _remove_container(
                session,
                container_name,
                expected_owner=owner,
                expected_job=job_slug,
            )
            _status_update(
                status_path,
                remote_container_removed=bool(removed),
                remote_job_removed=False,
                remote_diagnostics_retained=True,
            )
            return 0 if final_state in {"cancelled", "paused"} else 1

        # Download → verify → register → cleanup (shared with re-attach).
        dataset_fingerprint = remote_status.get("dataset_fingerprint") or dataset_fingerprint
        remote_status["dataset_fingerprint"] = dataset_fingerprint
        return _finalize_completed_remote_job(
            session,
            spec,
            profile,
            status_path,
            remote_paths,
            prepared,
            remote_status,
            asset_identity,
            runtime_image_id,
            container_name,
            container_may_exist,
            prepared_cache_namespace,
            local_root,
            result_archive,
            controller_log,
            remote_log,
        )
    except RemoteTaskAbandoned:
        warning = (
            "Local monitoring was abandoned because the remote state could not "
            "be confirmed. The remote container may still be running and may "
            "still use GPU or disk resources. Ask the server administrator to "
            "inspect the recorded container name before deleting this task."
        )
        _append_log(controller_log, warning)
        _status_update(
            status_path,
            status="abandoned",
            phase="abandoned_locally",
            local_abandon_requested=True,
            remote_state_unknown=bool(container_may_exist),
            remote_stop_confirmed=False if container_may_exist else None,
            remote_abandon_warning=warning,
            message=warning,
            error=warning,
            completed_at_epoch=time.time(),
        )
        return 125
    except InterruptedError:
        if session is not None and remote_paths:
            try:
                _stop_remote(
                    session,
                    container_name,
                    status_path,
                    final="cancelled",
                    remote_control_path=str(
                        (read_json(status_path, {}) or {}).get(
                            "remote_control_path"
                        )
                        or ""
                    ),
                    remote_control_kind=str(
                        (read_json(status_path, {}) or {}).get(
                            "remote_control_kind"
                        )
                        or ""
                    ),
                    remote_job_dir=str(
                        (read_json(status_path, {}) or {}).get("remote_job_dir")
                        or ""
                    ),
                    remote_root=profile["remote_root"],
                    expected_owner=owner,
                    expected_job=job_slug,
                )
            except Exception as stop_exc:
                warning = (
                    "Stop was requested, but the remote container stop could "
                    "not be confirmed: {}. Retry Stop, or use Abandon Locally "
                    "if the server remains unreachable.".format(stop_exc)
                )
                _append_log(controller_log, warning)
                _status_update(
                    status_path,
                    status="stopping",
                    phase="remote_termination_pending",
                    remote_state_unknown=True,
                    remote_stop_confirmed=False,
                    error=warning,
                    message=warning,
                )
                return 1
        elif container_may_exist:
            warning = (
                "Stop was requested, but the server is unreachable and the "
                "remote container stop cannot be confirmed. Retry Stop after "
                "the connection returns, or use Abandon Locally to release "
                "this workstation without claiming that the server GPU is free."
            )
            _append_log(controller_log, warning)
            _status_update(
                status_path,
                status="stopping",
                phase="remote_termination_pending",
                remote_state_unknown=True,
                remote_stop_confirmed=False,
                error=warning,
                message=warning,
            )
            return 1
        else:
            _status_update(
                status_path,
                status="cancelled",
                phase="cancelled",
                completed_at_epoch=time.time(),
            )
        return 130
    except RemoteServerUnreachable as exc:
        # The reconnect budget ran out mid-run: the remote container may
        # still be training, so this stays non-terminal and recoverable via
        # re-attach rather than being recorded as a failed or stopping task.
        _append_log(controller_log, "{}: {}".format(type(exc).__name__, exc))
        _status_update(
            status_path,
            status="remote_unreachable",
            phase="remote_unreachable",
            message=str(exc),
            error="{}: {}".format(type(exc).__name__, exc),
            remote_state_unknown=True,
            remote_reconnect_exhausted=True,
            remote_previous_status=str(
                (read_json(status_path, {}) or {}).get("status") or ""
            ),
            completed_at_epoch=None,
        )
        return 1
    except Exception as exc:
        termination_confirmed = not container_may_exist
        cleanup_error = ""
        if session is not None and remote_paths:
            try:
                _sync_remote_log(
                    session,
                    remote_paths["job"],
                    remote_log,
                    locals().get("log_sync_state", {"offset": 0}),
                )
            except Exception:
                pass
            try:
                removed = _remove_container(
                    session,
                    container_name,
                    expected_owner=owner,
                    expected_job=job_slug,
                )
                termination_confirmed = bool(removed)
                if not termination_confirmed:
                    state, _code = _container_state(session, container_name)
                    termination_confirmed = state in {
                        "exited",
                        "dead",
                        "missing",
                        "created",
                    }
                _status_update(
                    status_path,
                    remote_container_removed=bool(removed),
                    remote_job_removed=False,
                    remote_diagnostics_retained=True,
                )
            except Exception as cleanup_exc:
                cleanup_error = str(cleanup_exc)
                _append_log(
                    controller_log,
                    "Could not confirm remote container cleanup after failure: "
                    "{}".format(cleanup_exc),
                )
        _append_log(
            controller_log,
            "{}: {}\n{}".format(type(exc).__name__, exc, traceback.format_exc()),
        )
        _status_update(
            status_path,
            status="failed" if termination_confirmed else "stopping",
            phase=(
                "remote_failed"
                if termination_confirmed
                else "remote_termination_pending"
            ),
            error=(
                "{}: {}".format(type(exc).__name__, exc)
                if termination_confirmed
                else "Remote training failed, but container termination could "
                "not be confirmed: {}{}".format(
                    exc,
                    " (cleanup: {})".format(cleanup_error)
                    if cleanup_error
                    else "",
                )
            ),
            traceback=traceback.format_exc(),
            execution_backend="remote",
            local_log_path=str(remote_log),
            controller_log_path=str(controller_log),
            diagnostic_log_paths=list(
                dict.fromkeys((str(remote_log), str(controller_log)))
            ),
            completed_at_epoch=time.time() if termination_confirmed else None,
            remote_stop_confirmed=bool(termination_confirmed),
            remote_state_unknown=not bool(termination_confirmed),
        )
        return 1
    finally:
        if (
            session is not None
            and kind in {"nnunet", "nnunet_infer", "flexict", "flexict_infer"}
            and not bool(profile.get("cache_training_data", True))
            and prepared_cache_namespace
            and not prepared_cache_removed
        ):
            final_status = read_json(status_path, {}) or {}
            safe_to_remove = str(final_status.get("status") or "") in {
                "completed",
                "cancelled",
                "failed",
            } and not bool(final_status.get("remote_state_unknown"))
            if safe_to_remove:
                try:
                    prepared_cache_removed = _remove_remote_prepared_namespace(
                        session, remote_paths, prepared_cache_namespace
                    )
                    _status_update(
                        status_path,
                        remote_prepared_cache_removed=bool(
                            prepared_cache_removed
                        ),
                    )
                except Exception as cache_cleanup_exc:
                    _append_log(
                        controller_log,
                        "Could not remove the job-specific remote nnU-Net cache: {}"
                        .format(cache_cleanup_exc),
                    )
        if session is not None:
            session.close()
        if kind == "nninteractive":
            try:
                import tools.nninteractive_finetune_pipeline as cleanup_pipeline

                job_dir = Path(spec["job_dir"]).resolve()
                request = read_json(job_dir / "request.json", {}) or {}
                final_status = str(
                    (read_json(status_path, {}) or {}).get("status") or ""
                ).lower()
                keep_failed = bool(
                    cleanup_pipeline.load_config().get(
                        "keep_failed_training_artifacts", False
                    )
                )
                if final_status in {"completed", "cancelled"} or (
                    final_status == "failed" and not keep_failed
                ):
                    cleanup = cleanup_pipeline._cleanup_terminal_artifacts(
                        request,
                        job_dir,
                        remove_partial_model=final_status != "completed",
                        reset_resume_state=final_status == "failed",
                    )
                    _status_update(status_path, artifact_cleanup=cleanup)
            except Exception as exc:
                _append_log(
                    controller_log,
                    "Could not clean local nnInteractive staging data: {}".format(
                        exc
                    ),
                )
        if kind in {"nnunet", "nnunet_infer", "flexict", "flexict_infer"}:
            try:
                from tools.nnunet_common import compact_completed_log

                for completed_log in (remote_log, controller_log):
                    compact_completed_log(completed_log)
            except Exception:
                pass
        shutil.rmtree(str(bundle), ignore_errors=True)
        shutil.rmtree(str(dataset_archive_dir), ignore_errors=True)
        for path in (archive, result_archive):
            try:
                path.unlink()
            except OSError:
                pass


def reattach(status_path: Path) -> int:
    """Re-attach local monitoring to an orphaned remote container.

    After the local controller dies (crash, reboot, forced logout) the
    remote container keeps running, but until now the only local action
    was Stop. This entry point validates ownership of the still-existing
    remote job, then either resumes monitoring or — when training already
    finished remotely — downloads and registers the artifact.
    """
    path = Path(status_path).expanduser().resolve()
    status = read_json(path, {}) or {}
    if str(status.get("execution_backend") or "") != "remote":
        raise RuntimeError("Only a remote task can be re-attached.")
    state = str(status.get("status") or "").lower()
    if state in TERMINAL:
        raise RuntimeError(
            "This task already finished ({}); nothing to re-attach.".format(
                state
            )
        )
    job_id = str(status.get("job_id") or "")
    remote_job_dir = str(status.get("remote_job_dir") or "")
    remote_profile_id = str(status.get("remote_profile_id") or "")
    if not job_id or not remote_job_dir or not remote_profile_id:
        raise RuntimeError(
            "This task does not record enough remote state to re-attach. "
            "Use Stop to clean it up and retry training."
        )
    # The launch spec (kind, request_path, job_dir) is preserved next to the
    # status file by every remote launcher; fall back to the status fields
    # if it was removed.
    spec: dict[str, Any] = {}
    for name in ("remote_spec.json", "remote_launch.json"):
        candidate = path.parent / name
        if candidate.is_file():
            spec = read_json(candidate, {}) or {}
            break
    kind = str(spec.get("kind") or status.get("kind") or "")
    if kind not in {"nninteractive", "nnunet", "nnunet_infer", "flexict", "flexict_infer"}:
        raise RuntimeError(
            "The remote job kind '{}' is not recognized; cannot re-attach."
            .format(kind or "<missing>")
        )
    spec.setdefault("kind", kind)
    spec.setdefault("job_id", job_id)
    spec.setdefault("job_dir", str(status.get("job_dir") or path.parent))
    spec.setdefault("status_path", str(path))
    request_path = str(status.get("request_path") or spec.get("request_path") or "")
    if request_path:
        spec.setdefault("request_path", request_path)
    profile = get_profile(remote_profile_id)
    owner = _remote_owner(profile)
    job_slug = safe_identifier(job_id, "job")
    container_name = _container_name(profile, job_id)
    recorded_container = str(status.get("remote_container_name") or "")
    if recorded_container and recorded_container != container_name:
        raise RuntimeError(
            "Refusing to re-attach: the recorded remote container does not "
            "match this job."
        )
    _validate_remote_job_path(
        profile["remote_root"],
        owner,
        remote_job_dir,
    )
    remote_paths = {
        "job": remote_job_dir,
        "jobs": str(PurePosixPath(remote_job_dir).parent),
        "archive": str(
            PurePosixPath(remote_job_dir).with_name(
                PurePosixPath(remote_job_dir).name + ".tar"
            )
        ),
        "models": str(PurePosixPath(profile["remote_root"]) / "models"),
        "locks": str(PurePosixPath(profile["remote_root"]) / "locks"),
        "dataset_cache": "",
        "prepared_cache": "",
    }
    local_root = path.parent / (job_id + "_remote")
    local_root.mkdir(parents=True, exist_ok=True)
    result_archive = local_root / "result.tar"
    if kind == "nninteractive":
        remote_log = path.parent / "job.log"
        controller_log = remote_log
    else:
        remote_log = local_root / "remote_training.log"
        controller_log = path.with_name(
            path.name + ".remote_controller.log"
        )
    abandon_path = _abandon_path(path)
    try:
        abandon_path.unlink()
    except FileNotFoundError:
        pass
    _append_log(
        controller_log,
        "Re-attaching to the remote job at {}.".format(remote_job_dir),
    )
    _status_update(
        path,
        status="reattaching",
        phase="reattaching_remote",
        message=(
            "Re-attaching to the remote container and resuming monitoring."
        ),
        controller_pid=os.getpid(),
        local_abandon_requested=False,
        remote_state_unknown=False,
        error="",
        abandon_path=str(abandon_path),
        local_log_path=str(remote_log),
        controller_log_path=str(controller_log),
        progress_percent=max(
            30, int(status.get("progress_percent") or 30)
        ),
    )
    session: SSHSession | None = None
    try:
        session = SSHSession(profile)
        container_owned = _assert_container_owned(
            session,
            container_name,
            expected_owner=owner,
            expected_job=job_slug,
        )
        remote_status = session.read_remote_json(
            str(
                PurePosixPath(remote_job_dir)
                / _remote_status_relative(kind)
            ),
            {},
        ) or {}
        remote_state = str(remote_status.get("status") or "").lower()
        if remote_state in TERMINAL and remote_state != "paused":
            if remote_state == "completed":
                # The controller died after remote completion but before the
                # download; finish that tail now.
                return _reattach_finalize_from_status(
                    session,
                    spec,
                    profile,
                    path,
                    remote_paths,
                    remote_status,
                    container_name,
                    local_root,
                    result_archive,
                    controller_log,
                    remote_log,
                )
            # Remote failed/cancelled: mirror and clean up like run() does.
            _merge_remote_status(
                path,
                remote_status,
                profile,
                container_name,
                remote_job_dir,
            )
            try:
                _sync_remote_log(
                    session,
                    remote_job_dir,
                    remote_log,
                    {"offset": 0, "remote_relative": _remote_log_relative(kind)},
                )
            except Exception as exc:
                _append_log(
                    controller_log,
                    "Could not synchronize the remote diagnostic log: {}".format(
                        exc
                    ),
                )
            removed = _remove_container(
                session,
                container_name,
                expected_owner=owner,
                expected_job=job_slug,
            )
            _status_update(
                path,
                remote_container_removed=bool(removed),
                remote_job_removed=False,
                remote_diagnostics_retained=True,
                status=(
                    "cancelled"
                    if remote_state == "cancelled"
                    else "failed"
                ),
                phase="remote_{}_reattached".format(remote_state),
                completed_at_epoch=time.time(),
            )
            return 0 if remote_state == "cancelled" else 1
        if not container_owned:
            # Container gone and no terminal pipeline status: nothing to
            # monitor and nothing to download.
            guidance = (
                "The remote container for this task no longer exists and no "
                "result was recorded. Use Stop to finish cleanup locally, or "
                "contact the server administrator if the container name "
                "below looks wrong ({}).".format(container_name)
            )
            _status_update(
                path,
                status="orphaned_remote",
                phase="remote_container_missing",
                message=guidance,
                error=guidance,
                remote_state_unknown=True,
            )
            _append_log(controller_log, guidance)
            return 2
        # Container alive (or status non-terminal): resume monitoring.
        session, remote_status = _monitor_remote(
            session,
            spec,
            profile,
            path,
            remote_paths,
            container_name,
            owner,
            controller_log,
            remote_log,
        )
        final_state = str(remote_status.get("status") or "").lower()
        if final_state != "completed":
            try:
                _sync_remote_log(
                    session,
                    remote_job_dir,
                    remote_log,
                    {"offset": 0, "remote_relative": _remote_log_relative(kind)},
                )
            except Exception:
                pass
            _merge_remote_status(
                path,
                remote_status,
                profile,
                container_name,
                remote_job_dir,
            )
            removed = _remove_container(
                session,
                container_name,
                expected_owner=owner,
                expected_job=job_slug,
            )
            _status_update(
                path,
                remote_container_removed=bool(removed),
                remote_job_removed=False,
                remote_diagnostics_retained=True,
            )
            return 0 if final_state in {"cancelled", "paused"} else 1
        remote_status["dataset_fingerprint"] = str(
            remote_status.get("dataset_fingerprint")
            or status.get("dataset_fingerprint")
            or ""
        )
        return _reattach_finalize_from_status(
            session,
            spec,
            profile,
            path,
            remote_paths,
            remote_status,
            container_name,
            local_root,
            result_archive,
            controller_log,
            remote_log,
        )
    except RemoteTaskAbandoned:
        warning = (
            "Local monitoring was abandoned because the remote state could not "
            "be confirmed. The remote container may still be running and may "
            "still use GPU or disk resources. Ask the server administrator to "
            "inspect the recorded container name before deleting this task."
        )
        _append_log(controller_log, warning)
        _status_update(
            path,
            status="abandoned",
            phase="abandoned_locally",
            local_abandon_requested=True,
            remote_state_unknown=True,
            remote_stop_confirmed=False,
            remote_abandon_warning=warning,
            message=warning,
            error=warning,
            completed_at_epoch=time.time(),
        )
        return 125
    except Exception as exc:
        if isinstance(exc, RemoteServerUnreachable):
            # The reconnect budget ran out: the remote container may still
            # be training, so this stays non-terminal and recoverable via
            # another re-attach rather than being recorded as failed.
            _append_log(controller_log, "{}: {}".format(type(exc).__name__, exc))
            _status_update(
                path,
                status="remote_unreachable",
                phase="remote_unreachable",
                message=str(exc),
                error="{}: {}".format(type(exc).__name__, exc),
                remote_state_unknown=True,
                remote_reconnect_exhausted=True,
                remote_previous_status=str(
                    (read_json(path, {}) or {}).get("status") or ""
                ),
                completed_at_epoch=None,
            )
            return 1
        _append_log(
            controller_log,
            "{}: {}\n{}".format(type(exc).__name__, exc, traceback.format_exc()),
        )
        _status_update(
            path,
            status="orphaned_remote",
            phase="reattach_failed",
            message=(
                "Re-attaching to the remote container failed: {}. The remote "
                "container may still be running; use Stop to clean up, or "
                "retry re-attach after the connection returns.".format(exc)
            ),
            error="{}: {}".format(type(exc).__name__, exc),
            remote_state_unknown=True,
        )
        return 1
    finally:
        if session is not None:
            session.close()


def cancel(status_path: Path) -> int:
    status = read_json(status_path, {}) or {}
    profile_id = str(status.get("remote_profile_id") or "")
    container_name = str(status.get("remote_container_name") or "")
    remote_control_path = str(status.get("remote_control_path") or "")
    remote_control_kind = str(status.get("remote_control_kind") or "")
    remote_job_dir = str(status.get("remote_job_dir") or "")
    job_id = str(status.get("job_id") or "")
    if not profile_id or not container_name or not job_id:
        raise RuntimeError("This status file does not identify a remote container.")
    profile = get_profile(profile_id)
    owner = _remote_owner(profile)
    expected_container = _container_name(profile, job_id)
    if container_name != expected_container:
        raise RuntimeError(
            "Refusing to stop a remote container that does not match this job."
        )
    _validate_remote_job_path(
        profile["remote_root"],
        owner,
        remote_job_dir,
    )
    _validate_remote_control_path(remote_job_dir, remote_control_path)
    try:
        with SSHSession(profile) as session:
            _stop_remote(
                session,
                container_name,
                status_path,
                final="cancelled",
                remote_control_path=remote_control_path,
                remote_control_kind=remote_control_kind,
                remote_job_dir=remote_job_dir,
                remote_root=profile["remote_root"],
                expected_owner=owner,
                expected_job=safe_identifier(job_id, "job"),
            )
        return 0
    except Exception as exc:
        _status_update(
            status_path,
            status="stopping",
            phase="remote_termination_pending",
            remote_stop_confirmed=False,
            remote_state_unknown=True,
            error=(
                "Remote stop could not be confirmed: {}. Retry Stop after the "
                "connection returns, or use Abandon Locally if the server "
                "remains unreachable."
            ).format(exc),
        )
        return 1


def abandon(status_path: Path) -> int:
    """End local monitoring without claiming that remote work was stopped."""
    path = Path(status_path).expanduser().resolve()
    status = read_json(path, {}) or {}
    if str(status.get("execution_backend") or "") != "remote":
        raise RuntimeError("Only a remote task can be abandoned locally.")
    if str(status.get("status") or "").lower() in TERMINAL:
        return 0
    marker = _abandon_path(path)
    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_name(
        "{}.{}.tmp".format(marker.name, uuid.uuid4().hex)
    )
    try:
        temporary.write_text(
            "local abandon requested at {}\n".format(
                time.strftime("%Y-%m-%d %H:%M:%S")
            ),
            encoding="utf-8",
        )
        os.replace(str(temporary), str(marker))
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass
    warning = (
        "Local monitoring was abandoned. The remote server could not confirm "
        "the container state, so it may still be running and consuming GPU or "
        "disk resources. Give the recorded container name to the server "
        "administrator for inspection."
    )
    _status_update(
        path,
        status="abandoned",
        phase="abandoned_locally",
        abandon_path=str(marker),
        local_abandon_requested=True,
        remote_state_unknown=True,
        remote_stop_confirmed=False,
        remote_abandon_warning=warning,
        message=warning,
        error=warning,
        completed_at_epoch=time.time(),
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--spec", required=True)
    cancel_parser = sub.add_parser("cancel")
    cancel_parser.add_argument("--status", required=True)
    reattach_parser = sub.add_parser("reattach")
    reattach_parser.add_argument("--status", required=True)
    abandon_parser = sub.add_parser("abandon")
    abandon_parser.add_argument("--status", required=True)
    args = parser.parse_args()
    if args.command == "abandon":
        return abandon(Path(args.status).resolve())
    if args.command == "cancel":
        return cancel(Path(args.status).resolve())
    if args.command == "reattach":
        return reattach(Path(args.status).resolve())
    return run(Path(args.spec).resolve())


if __name__ == "__main__":
    raise SystemExit(main())
