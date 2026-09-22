#!/usr/bin/env python3
"""Optional SSH transport and server-profile storage for remote AI training.

This module is deliberately dependency-light at import time. Paramiko is only
imported after a user explicitly selects a remote server, so local training
continues to work when the optional remote package is not installed.
"""

from __future__ import annotations

import base64
import ctypes
import hashlib
import json
import os
import re
import shlex
import socket
import time
import uuid
from ctypes import wintypes
from pathlib import Path, PurePosixPath
from typing import Any, Callable


SCHEMA_VERSION = "mimics_remote_compute.v1"
DEFAULT_IMAGE = "mimics-ai-runtime:1.0"
DEFAULT_REMOTE_ROOT = "mimics-ai"
DEFAULT_GPU_DEVICE = "auto"
PROFILE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]+$")
IMAGE_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.:/@-]*$")
GPU_DEVICE_PATTERN = re.compile(
    r"^(?:auto|[0-9]+|GPU-[a-zA-Z0-9-]+|MIG-[a-zA-Z0-9./-]+)$"
)


class RemoteComputeError(RuntimeError):
    pass


class RemoteDependencyError(RemoteComputeError):
    pass


class RemoteCommandError(RemoteComputeError):
    pass


class UnknownHostKeyError(RemoteComputeError):
    def __init__(self, host: str, port: int, fingerprint: str) -> None:
        self.host = host
        self.port = int(port)
        self.fingerprint = fingerprint
        super().__init__(
            "The SSH host key for {}:{} is not trusted ({}).".format(
                host, port, fingerprint
            )
        )


class HostKeyChangedError(RemoteComputeError):
    pass


def _configuration_root(*, create: bool = False) -> Path:
    configured = os.environ.get("MIMICS_REMOTE_CONFIG_DIR", "").strip()
    if configured:
        root = Path(os.path.expandvars(os.path.expanduser(configured)))
    elif os.name == "nt":
        root = Path(
            os.environ.get("LOCALAPPDATA")
            or os.environ.get("APPDATA")
            or str(Path.home())
        ) / "MimicsScript" / "remote_compute"
    else:
        root = Path(
            os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")
        ) / "mimics-script" / "remote_compute"
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root.expanduser().absolute()


def profiles_path() -> Path:
    return _configuration_root() / "servers.json"


def known_hosts_path() -> Path:
    return _configuration_root() / "known_hosts"


def read_json(path: str | Path, default: Any = None) -> Any:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return default


def write_json_atomic(
    path: str | Path,
    payload: dict[str, Any],
    *,
    retries: int = 12,
    max_sleep: float = 0.15,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    last_error: OSError | None = None
    for attempt in range(max(1, int(retries))):
        temporary = target.with_name(
            "{}.{}.{}.tmp".format(target.name, os.getpid(), uuid.uuid4().hex)
        )
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with temporary.open("w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(str(temporary), str(target))
            return
        except OSError as exc:
            last_error = exc
            try:
                temporary.unlink()
            except OSError:
                pass
            time.sleep(min(float(max_sleep), 0.02 * (attempt + 1)))

    # Antivirus and SMB servers can temporarily reject replace() even though a
    # normal write is permitted. Status readers tolerate an incomplete JSON
    # document and retry, so a flushed direct write is a safer final fallback.
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        return
    except OSError as exc:
        last_error = exc
    if last_error is not None:
        raise last_error


def safe_identifier(value: object, fallback: str = "server") -> str:
    text = str(value or "").strip()
    output = []
    for character in text:
        if character.isalnum() or character in "_.-":
            output.append(character)
        else:
            output.append("_")
    result = re.sub(r"_+", "_", "".join(output)).strip("_.-")
    return result or fallback


def normalize_gpu_device(value: object) -> str:
    text = str(value or DEFAULT_GPU_DEVICE).strip()
    if text.lower() in {"auto", "automatic"}:
        return DEFAULT_GPU_DEVICE
    if not GPU_DEVICE_PATTERN.match(text):
        raise ValueError(
            "GPU device must be Automatic, one numeric GPU index, or one "
            "NVIDIA GPU/MIG UUID."
        )
    return text


def docker_gpu_request(profile: dict[str, Any]) -> str:
    device = normalize_gpu_device(profile.get("gpu_device"))
    return "all" if device == DEFAULT_GPU_DEVICE else "device={}".format(device)


def normalize_profile(profile: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(profile, dict):
        raise ValueError("Remote server profile must be an object.")
    name = str(profile.get("name") or "").strip()
    host = str(profile.get("host") or "").strip()
    username = str(profile.get("username") or "").strip()
    if not name:
        raise ValueError("Enter a server profile name.")
    if not host or any(character.isspace() for character in host):
        raise ValueError("Enter a valid SSH host name or IP address.")
    if not username or any(character in "\r\n" for character in username):
        raise ValueError("Enter the SSH username.")
    try:
        port = int(profile.get("port") or 22)
    except Exception:
        raise ValueError("SSH port must be an integer.")
    if port < 1 or port > 65535:
        raise ValueError("SSH port must be between 1 and 65535.")
    profile_id = safe_identifier(
        profile.get("profile_id") or "{}_{}".format(name, username)
    )
    if not PROFILE_ID_PATTERN.match(profile_id):
        raise ValueError("Remote server profile ID contains unsupported characters.")
    auth_method = str(profile.get("auth_method") or "password").strip().lower()
    if auth_method not in {"password", "key"}:
        raise ValueError("Authentication must use a password or SSH private key.")
    key_path = str(profile.get("key_path") or "").strip()
    if auth_method == "key" and not key_path:
        raise ValueError("Choose an SSH private key.")
    image = str(profile.get("runtime_image") or DEFAULT_IMAGE).strip()
    if not IMAGE_PATTERN.match(image):
        raise ValueError("Runtime image name contains unsupported characters.")
    remote_root_text = str(
        profile.get("remote_root") or DEFAULT_REMOTE_ROOT
    ).strip().replace("\\", "/")
    if not remote_root_text or "\x00" in remote_root_text:
        raise ValueError("Remote work folder is invalid.")
    remote_root_path = PurePosixPath(remote_root_text)
    if remote_root_path in {PurePosixPath("/"), PurePosixPath(".")}:
        raise ValueError(
            "Remote work folder must be a dedicated subfolder, not a filesystem root."
        )
    if any(part == ".." for part in remote_root_path.parts):
        raise ValueError("Remote work folder cannot contain '..'.")
    remote_root = str(remote_root_path)
    gpu_device = normalize_gpu_device(profile.get("gpu_device"))
    cache_training_data = bool(profile.get("cache_training_data", True))
    try:
        remote_cache_retention_days = int(
            profile.get("remote_cache_retention_days") or 30
        )
    except Exception:
        remote_cache_retention_days = 30
    remote_cache_retention_days = min(
        3650, max(1, remote_cache_retention_days)
    )
    remote_weights_verify = str(
        profile.get("remote_weights_verify") or "strict"
    ).strip().lower()
    if remote_weights_verify not in {"strict", "warn", "off"}:
        remote_weights_verify = "strict"
    return {
        "profile_id": profile_id,
        "name": name,
        "host": host,
        "port": port,
        "username": username,
        "auth_method": auth_method,
        "key_path": key_path,
        "remote_root": remote_root,
        "runtime_image": image,
        "gpu_device": gpu_device,
        "cache_training_data": cache_training_data,
        "remote_cache_retention_days": remote_cache_retention_days,
        "remote_weights_verify": remote_weights_verify,
        "updated_at_epoch": time.time(),
    }


def load_profiles() -> list[dict[str, Any]]:
    payload = read_json(profiles_path(), {}) or {}
    result = []
    for row in payload.get("profiles") or []:
        try:
            result.append(normalize_profile(row))
        except Exception:
            continue
    result.sort(key=lambda row: str(row.get("name") or "").lower())
    return result


def get_profile(profile_id: str) -> dict[str, Any]:
    wanted = str(profile_id or "")
    for profile in load_profiles():
        if profile.get("profile_id") == wanted:
            return profile
    raise RemoteComputeError(
        "Remote server profile '{}' was not found. Open Manage Servers and save it again.".format(
            wanted
        )
    )


def save_profile(profile: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_profile(profile)
    rows = [
        row
        for row in load_profiles()
        if row.get("profile_id") != normalized["profile_id"]
    ]
    rows.append(normalized)
    rows.sort(key=lambda row: str(row.get("name") or "").lower())
    write_json_atomic(
        profiles_path(),
        {
            "schema_version": SCHEMA_VERSION,
            "profiles": rows,
            "updated_at_epoch": time.time(),
        },
    )
    return normalized


def delete_profile(profile_id: str) -> None:
    rows = [
        row for row in load_profiles() if row.get("profile_id") != profile_id
    ]
    write_json_atomic(
        profiles_path(),
        {
            "schema_version": SCHEMA_VERSION,
            "profiles": rows,
            "updated_at_epoch": time.time(),
        },
    )
    delete_password(profile_id)


def _credential_target(profile_id: str) -> str:
    return "MimicsScript/RemoteCompute/{}".format(safe_identifier(profile_id))


class _FILETIME(ctypes.Structure):
    _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]


class _CREDENTIALW(ctypes.Structure):
    _fields_ = [
        ("Flags", wintypes.DWORD),
        ("Type", wintypes.DWORD),
        ("TargetName", wintypes.LPWSTR),
        ("Comment", wintypes.LPWSTR),
        ("LastWritten", _FILETIME),
        ("CredentialBlobSize", wintypes.DWORD),
        ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
        ("Persist", wintypes.DWORD),
        ("AttributeCount", wintypes.DWORD),
        ("Attributes", ctypes.c_void_p),
        ("TargetAlias", wintypes.LPWSTR),
        ("UserName", wintypes.LPWSTR),
    ]


def store_password(
    profile_id: str,
    username: str,
    password: str,
    *,
    remember: bool,
) -> None:
    if os.name != "nt":
        raise RemoteComputeError(
            "Password storage is available on the target Windows workstation only. "
            "Use an SSH key on this operating system."
        )
    blob = str(password or "").encode("utf-16-le")
    if not blob:
        raise ValueError("Enter the SSH password.")
    if len(blob) > 5120:
        raise ValueError("SSH password is too long for Windows Credential Manager.")
    buffer = (ctypes.c_ubyte * len(blob)).from_buffer_copy(blob)
    credential = _CREDENTIALW()
    credential.Flags = 0
    credential.Type = 1  # CRED_TYPE_GENERIC
    credential.TargetName = _credential_target(profile_id)
    credential.Comment = "Mimics-Script remote training"
    credential.CredentialBlobSize = len(blob)
    credential.CredentialBlob = ctypes.cast(
        buffer, ctypes.POINTER(ctypes.c_ubyte)
    )
    credential.Persist = 2 if remember else 1  # LOCAL_MACHINE or SESSION
    credential.UserName = str(username or "")
    advapi32 = ctypes.WinDLL("Advapi32.dll")
    write = advapi32.CredWriteW
    write.argtypes = [ctypes.POINTER(_CREDENTIALW), wintypes.DWORD]
    write.restype = wintypes.BOOL
    if not write(ctypes.byref(credential), 0):
        raise ctypes.WinError()


def load_password(profile_id: str) -> str:
    env_name = "MIMICS_REMOTE_PASSWORD_{}".format(
        safe_identifier(profile_id).upper()
    )
    if os.environ.get(env_name):
        return os.environ[env_name]
    if os.name != "nt":
        raise RemoteComputeError(
            "No SSH password is available. On Windows, save it from Manage "
            "Servers; on other systems, use an SSH key or {} for testing.".format(
                env_name
            )
        )
    advapi32 = ctypes.WinDLL("Advapi32.dll")
    read = advapi32.CredReadW
    read.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.POINTER(_CREDENTIALW)),
    ]
    read.restype = wintypes.BOOL
    free = advapi32.CredFree
    free.argtypes = [ctypes.c_void_p]
    pointer = ctypes.POINTER(_CREDENTIALW)()
    if not read(_credential_target(profile_id), 1, 0, ctypes.byref(pointer)):
        raise RemoteComputeError(
            "The SSH password is not stored for this server profile. Open "
            "Manage Servers and enter it again."
        )
    try:
        credential = pointer.contents
        size = int(credential.CredentialBlobSize)
        raw = ctypes.string_at(credential.CredentialBlob, size)
        return raw.decode("utf-16-le")
    finally:
        free(pointer)


def delete_password(profile_id: str) -> None:
    if os.name != "nt":
        return
    advapi32 = ctypes.WinDLL("Advapi32.dll")
    delete = advapi32.CredDeleteW
    delete.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD]
    delete.restype = wintypes.BOOL
    delete(_credential_target(profile_id), 1, 0)


def require_paramiko():
    try:
        import paramiko
    except Exception as exc:
        raise RemoteDependencyError(
            "Remote training requires the optional Paramiko package in "
            "nninteractive_env. Run 'python tools/setup_env.py install-remote' "
            "or rebuild the offline bundle. Local training is still available."
        ) from exc
    return paramiko


def _host_key_name(host: str, port: int) -> str:
    return host if int(port) == 22 else "[{}]:{}".format(host, int(port))


def _fingerprint(key: Any) -> str:
    digest = hashlib.sha256(key.asbytes()).digest()
    return "SHA256:" + base64.b64encode(digest).decode("ascii").rstrip("=")


def _server_key(profile: dict[str, Any], timeout: float = 10.0):
    paramiko = require_paramiko()
    sock = socket.create_connection(
        (profile["host"], int(profile["port"])), timeout=timeout
    )
    transport = paramiko.Transport(sock)
    try:
        transport.start_client(timeout=timeout)
        key = transport.get_remote_server_key()
        if key is None:
            raise RemoteComputeError("The SSH server did not provide a host key.")
        return key
    finally:
        transport.close()


def verify_host_key(
    profile: dict[str, Any],
    *,
    trust_unknown: bool = False,
    timeout: float = 10.0,
) -> str:
    paramiko = require_paramiko()
    profile = normalize_profile(profile)
    key = _server_key(profile, timeout=timeout)
    fingerprint = _fingerprint(key)
    host_name = _host_key_name(profile["host"], profile["port"])
    keys = paramiko.HostKeys()
    path = known_hosts_path()
    if path.is_file():
        try:
            keys.load(str(path))
        except Exception as exc:
            raise RemoteComputeError(
                "Could not read the trusted SSH host keys: {}".format(exc)
            )
    existing = keys.lookup(host_name)
    if existing:
        expected = existing.get(key.get_name())
        if expected is None or expected.asbytes() != key.asbytes():
            raise HostKeyChangedError(
                "The SSH host key for {} changed. Expected a previously trusted "
                "key, received {}. Connection was stopped.".format(
                    host_name, fingerprint
                )
            )
        return fingerprint
    if not trust_unknown:
        raise UnknownHostKeyError(
            profile["host"], profile["port"], fingerprint
        )
    keys.add(host_name, key.get_name(), key)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys.save(str(path))
    return fingerprint


class SSHSession:
    def __init__(
        self,
        profile: dict[str, Any],
        *,
        trust_unknown: bool = False,
        timeout: float = 15.0,
    ) -> None:
        self.profile = normalize_profile(profile)
        self.paramiko = require_paramiko()
        self.fingerprint = verify_host_key(
            self.profile, trust_unknown=trust_unknown, timeout=timeout
        )
        self.client = self.paramiko.SSHClient()
        self.client.load_host_keys(str(known_hosts_path()))
        self.client.set_missing_host_key_policy(self.paramiko.RejectPolicy())
        connect_args: dict[str, Any] = {
            "hostname": self.profile["host"],
            "port": int(self.profile["port"]),
            "username": self.profile["username"],
            "timeout": timeout,
            "banner_timeout": timeout,
            "auth_timeout": timeout,
            "look_for_keys": False,
            "allow_agent": False,
        }
        if self.profile["auth_method"] == "key":
            connect_args["key_filename"] = os.path.expandvars(
                os.path.expanduser(self.profile["key_path"])
            )
        else:
            connect_args["password"] = load_password(
                self.profile["profile_id"]
            )
        try:
            self.client.connect(**connect_args)
        except Exception as exc:
            self.client.close()
            raise RemoteComputeError(
                "SSH authentication or connection failed for {}@{}:{}. Check "
                "the server address and port, then verify the username and "
                "saved password or SSH key. Also confirm that a firewall or "
                "VPN is not blocking SSH. Details: {}".format(
                    self.profile["username"],
                    self.profile["host"],
                    self.profile["port"],
                    exc,
                )
            ) from exc
        transport = self.client.get_transport()
        if transport is not None:
            transport.set_keepalive(30)
        self.sftp = self.client.open_sftp()
        self.home = self._remote_home()
        self.remote_root = self._resolve_remote_root(
            self.profile["remote_root"]
        )

    def __enter__(self) -> "SSHSession":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()

    def close(self) -> None:
        try:
            self.sftp.close()
        except Exception:
            pass
        self.client.close()

    def _remote_home(self) -> str:
        output = self.execute("printf %s \"$HOME\"").strip()
        if not output.startswith("/"):
            raise RemoteComputeError(
                "The remote SSH account did not report an absolute home directory."
            )
        return output

    def _resolve_remote_root(self, value: str) -> str:
        if value == "~":
            return self.home
        if value.startswith("~/"):
            value = value[2:]
        path = PurePosixPath(value)
        if not path.is_absolute():
            path = PurePosixPath(self.home) / path
        return str(path)

    def execute(
        self,
        command: str,
        *,
        timeout: float | None = 60.0,
        check: bool = True,
    ) -> str:
        code, output = self.execute_result(command, timeout=timeout)
        if check and code != 0:
            detail = output.strip()
            raise RemoteCommandError(
                "Remote command failed with exit code {}: {}".format(
                    code, detail[-2000:] or command
                )
            )
        return output

    def execute_result(
        self,
        command: str,
        *,
        timeout: float | None = 60.0,
    ) -> tuple[int, str]:
        _stdin, stdout, _stderr = self.client.exec_command(
            command, timeout=timeout
        )
        # Read one combined stream so a verbose stderr cannot fill its SSH
        # window while stdout is being drained.
        stdout.channel.set_combine_stderr(True)
        output = stdout.read().decode("utf-8", errors="replace")
        code = int(stdout.channel.recv_exit_status())
        return code, output

    def ensure_directory(self, path: str) -> None:
        self.execute("mkdir -p {}".format(shlex.quote(path)))

    def upload(
        self,
        local_path: str | Path,
        remote_path: str,
        callback: Callable[[int, int], None] | None = None,
    ) -> None:
        self.ensure_directory(str(PurePosixPath(remote_path).parent))
        temporary = remote_path + ".part"
        source = Path(local_path)
        total = int(source.stat().st_size)
        try:
            existing = int(self.sftp.stat(temporary).st_size)
        except IOError:
            existing = 0
        if existing < 0 or existing > total:
            try:
                self.sftp.remove(temporary)
            except Exception:
                pass
            existing = 0
        if callback:
            callback(existing, total)
        with source.open("rb") as local:
            local.seek(existing)
            mode = "ab" if existing else "wb"
            with self.sftp.open(temporary, mode) as remote:
                # Pipelining lets many 32 KB SFTP_WRITE requests stay in
                # flight before collecting ACKs, instead of stalling on a
                # round-trip after every write. This is the same mechanism
                # paramiko's own SFTPClient.putfo uses and is roughly 2.4x
                # faster than the default serial writes (~50 vs ~21 MB/s on
                # a gigabit LAN). Resume semantics are preserved because we
                # still open in append mode from the existing .part offset.
                # Guarded so non-paramiko fakes used in tests still work.
                if hasattr(remote, "set_pipelined"):
                    remote.set_pipelined(True)
                transferred = existing
                while transferred < total:
                    chunk = local.read(min(4 * 1024 * 1024, total - transferred))
                    if not chunk:
                        break
                    remote.write(chunk)
                    transferred += len(chunk)
                    if callback:
                        callback(transferred, total)
                remote.flush()
        actual = int(self.sftp.stat(temporary).st_size)
        if actual != total:
            raise RemoteComputeError(
                "Remote upload is incomplete: {} of {} bytes.".format(
                    actual, total
                )
            )
        try:
            self.sftp.remove(remote_path)
        except IOError:
            pass
        self.sftp.rename(temporary, remote_path)

    def download(
        self,
        remote_path: str,
        local_path: str | Path,
        callback: Callable[[int, int], None] | None = None,
    ) -> None:
        destination = Path(local_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".part")
        total = int(self.sftp.stat(remote_path).st_size)
        existing = int(temporary.stat().st_size) if temporary.is_file() else 0
        if existing < 0 or existing > total:
            temporary.unlink(missing_ok=True)
            existing = 0
        if callback:
            callback(existing, total)
        with self.sftp.open(remote_path, "rb") as remote:
            remote.seek(existing)
            with temporary.open("ab" if existing else "wb") as local:
                transferred = existing
                while transferred < total:
                    chunk = remote.read(min(4 * 1024 * 1024, total - transferred))
                    if not chunk:
                        break
                    local.write(chunk)
                    transferred += len(chunk)
                    if callback:
                        callback(transferred, total)
                local.flush()
                os.fsync(local.fileno())
        actual = int(temporary.stat().st_size)
        if actual != total:
            raise RemoteComputeError(
                "Remote download is incomplete: {} of {} bytes.".format(
                    actual, total
                )
            )
        os.replace(str(temporary), str(destination))

    def download_appended(
        self,
        remote_path: str,
        local_path: str | Path,
        remote_offset: int = 0,
    ) -> int:
        """Append bytes added to a growing remote log and return its new offset."""
        destination = Path(local_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        total = int(self.sftp.stat(remote_path).st_size)
        offset = max(0, int(remote_offset))
        if total < offset:
            offset = 0
            with destination.open("a", encoding="utf-8", errors="replace") as local:
                local.write(
                    "\n[Remote log restarted; continuing from its new beginning.]\n"
                )
        if total == offset:
            return total
        with self.sftp.open(remote_path, "rb") as remote:
            remote.seek(offset)
            with destination.open("ab") as local:
                remaining = total - offset
                while remaining > 0:
                    chunk = remote.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    local.write(chunk)
                    remaining -= len(chunk)
                local.flush()
        return total - max(0, remaining)

    def read_remote_json(
        self, remote_path: str, default: Any = None
    ) -> Any:
        try:
            with self.sftp.open(remote_path, "r") as handle:
                raw = handle.read()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            return json.loads(raw)
        except Exception:
            return default

    def path_exists(self, remote_path: str) -> bool:
        try:
            self.sftp.stat(remote_path)
            return True
        except IOError:
            return False


def test_connection(
    profile: dict[str, Any],
    *,
    trust_unknown: bool = False,
) -> dict[str, Any]:
    with SSHSession(profile, trust_unknown=trust_unknown) as session:
        root = session.remote_root
        try:
            session.ensure_directory(str(PurePosixPath(root) / "jobs"))
            session.ensure_directory(str(PurePosixPath(root) / "models"))
            session.ensure_directory(str(PurePosixPath(root) / "outputs"))
            probe = str(
                PurePosixPath(root)
                / (".mimics_write_test_" + uuid.uuid4().hex)
            )
            session.execute(
                "printf ready > {probe} && rm -f {probe}".format(
                    probe=shlex.quote(probe)
                )
            )
        except Exception as exc:
            raise RemoteComputeError(
                "The SSH account cannot write the remote work folder '{}'. "
                "Create it and grant this account read/write permission, then "
                "run Test Connection again. Details: {}".format(root, exc)
            ) from exc
        image = session.profile["runtime_image"]
        try:
            image_id = session.execute(
                "docker image inspect --format '{{{{.Id}}}}' {}".format(
                    shlex.quote(image)
                )
            ).strip()
        except Exception as exc:
            raise RemoteComputeError(
                "Docker cannot use runtime image '{}'. Confirm Docker access "
                "for this SSH account and run remote/setup_remote_server.sh "
                "to build or load the image. Details: {}".format(image, exc)
            ) from exc
        models = str(PurePosixPath(root) / "models")
        gpu_request = docker_gpu_request(session.profile)
        try:
            preflight_output = session.execute(
                "docker run --rm --gpus {gpu} --network none "
                "-v {models}:/models:ro {image} "
                "python /app/tools/remote_worker.py preflight "
                "--models-dir /models".format(
                    gpu=shlex.quote(gpu_request),
                    models=shlex.quote(models),
                    image=shlex.quote(image),
                ),
                timeout=180,
            ).strip()
        except Exception as exc:
            raise RemoteComputeError(
                "The isolated GPU container preflight failed. Verify the NVIDIA "
                "driver and Container Toolkit, install the required "
                "nnInteractive weights under '{}/models', then rerun "
                "remote/setup_remote_server.sh. Details: {}".format(root, exc)
            ) from exc
        try:
            start = preflight_output.index("{")
            end = preflight_output.rindex("}") + 1
            preflight = json.loads(preflight_output[start:end])
        except Exception as exc:
            raise RemoteComputeError(
                "The runtime image preflight returned unreadable output: {}".format(
                    preflight_output[-1000:]
                )
            ) from exc
        if not bool(preflight.get("ok")):
            missing = []
            if not preflight.get("cuda_available"):
                missing.append("CUDA is unavailable inside the container")
            if not preflight.get("offline_mode"):
                missing.append("offline runtime variables are not active")
            if not preflight.get("nninteractive_weights"):
                missing.append("nnInteractive base weights are missing")
            if not preflight.get("nninteractive_import"):
                missing.append("nnInteractive runtime imports failed")
            if not preflight.get("nnunet_import"):
                missing.append("nnU-Net runtime import failed")
            if not preflight.get("nnunet_custom_trainer"):
                missing.append("Mimics nnU-Net trainer is missing")
            raise RemoteComputeError(
                "The runtime image is not ready: {}. Run "
                "remote/setup_remote_server.sh as the SSH user after correcting "
                "these items. Preflight: {}".format(
                    "; ".join(missing) or "an unknown preflight check failed",
                    json.dumps(preflight, sort_keys=True)
                )
            )
        try:
            gpu_lines = session.execute(
                "nvidia-smi --query-gpu=index,uuid,name,memory.total "
                "--format=csv,noheader,nounits"
            ).strip().splitlines()
        except Exception as exc:
            raise RemoteComputeError(
                "The SSH account cannot query NVIDIA GPUs. Verify the server "
                "driver and account permissions. Details: {}".format(exc)
            ) from exc
        gpus = []
        for line in gpu_lines:
            parts = [part.strip() for part in line.split(",", 3)]
            if len(parts) != 4:
                continue
            try:
                memory_mb = int(parts[3])
            except Exception:
                memory_mb = 0
            gpus.append(
                {
                    "index": parts[0],
                    "uuid": parts[1],
                    "name": parts[2],
                    "memory_mb": memory_mb,
                }
            )
        disk = session.execute(
            "df -Pk {} | tail -1".format(shlex.quote(root))
        ).strip().split()
        free_kb = int(disk[3]) if len(disk) >= 4 and disk[3].isdigit() else 0
        return {
            "ok": True,
            "fingerprint": session.fingerprint,
            "remote_root": root,
            "runtime_image": image,
            "image_id": image_id,
            "preflight": preflight,
            "gpu_device": session.profile["gpu_device"],
            "gpus": gpus,
            "free_bytes": free_kb * 1024,
        }
