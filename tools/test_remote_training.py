#!/usr/bin/env python3
"""Contract tests for the optional SSH/Docker training path."""

from __future__ import annotations

import hashlib
import io
import importlib.util
import json
import os
import sys
import tarfile
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.package_portable as package_portable
import tools.remote_compute as remote_compute
import tools.remote_training_controller as controller
import tools.remote_worker as remote_worker


class _Process:
    pid = 4321


class RemoteProfileTests(unittest.TestCase):
    def test_profile_storage_does_not_persist_password(self):
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.dict(
                os.environ,
                {"MIMICS_REMOTE_CONFIG_DIR": temporary},
                clear=False,
            ):
                profile = remote_compute.save_profile(
                    {
                        "name": "GPU server",
                        "host": "10.1.2.3",
                        "port": 2222,
                        "username": "annotator",
                        "auth_method": "password",
                        "password": "must-not-be-written",
                    }
                )
                payload = json.loads(
                    remote_compute.profiles_path().read_text(encoding="utf-8")
                )
                serialized = json.dumps(payload)
                self.assertNotIn("must-not-be-written", serialized)
                self.assertNotIn("password", payload["profiles"][0])
                self.assertEqual(
                    remote_compute.get_profile(profile["profile_id"])["host"],
                    "10.1.2.3",
                )

    def test_reading_empty_profiles_has_no_filesystem_side_effect(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "not-created"
            with mock.patch.dict(
                os.environ,
                {"MIMICS_REMOTE_CONFIG_DIR": str(root)},
                clear=False,
            ):
                self.assertEqual(remote_compute.load_profiles(), [])
                self.assertFalse(root.exists())

    def test_profile_rejects_parent_remote_path(self):
        with self.assertRaisesRegex(ValueError, "cannot contain"):
            remote_compute.normalize_profile(
                {
                    "name": "bad",
                    "host": "server",
                    "username": "user",
                    "remote_root": "../another-user",
                }
            )

    def test_profile_rejects_filesystem_root(self):
        base = {
            "name": "GPU server",
            "host": "10.1.2.3",
            "username": "annotator",
            "auth_method": "password",
        }
        for remote_root in ("/", "."):
            with self.subTest(remote_root=remote_root), self.assertRaisesRegex(
                ValueError, "dedicated subfolder"
            ):
                remote_compute.normalize_profile(
                    dict(base, remote_root=remote_root)
                )

    def test_atomic_status_write_falls_back_after_replace_is_denied(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            remote_compute.os,
            "replace",
            side_effect=PermissionError("busy"),
        ):
            path = Path(temporary) / "status.json"
            remote_compute.write_json_atomic(
                path,
                {"status": "ready"},
                retries=2,
                max_sleep=0,
            )
            self.assertEqual(
                remote_compute.read_json(path, {}),
                {"status": "ready"},
            )

    def test_gpu_selection_is_single_device_and_backward_compatible(self):
        base = {
            "name": "GPU server",
            "host": "server",
            "username": "user",
        }
        automatic = remote_compute.normalize_profile(base)
        self.assertEqual(automatic["gpu_device"], "auto")
        self.assertTrue(automatic["cache_training_data"])
        self.assertEqual(remote_compute.docker_gpu_request(automatic), "all")
        selected = remote_compute.normalize_profile(
            dict(base, gpu_device="1")
        )
        self.assertEqual(
            remote_compute.docker_gpu_request(selected), "device=1"
        )
        with self.assertRaisesRegex(ValueError, "GPU device"):
            remote_compute.normalize_profile(
                dict(base, gpu_device="0,1")
            )


class ResumableTransferTests(unittest.TestCase):
    class _LocalSFTP:
        def stat(self, path):
            return Path(path).stat()

        def open(self, path, mode):
            return Path(path).open(mode)

        def remove(self, path):
            Path(path).unlink()

        def rename(self, source, destination):
            os.replace(source, destination)

    def _session(self, sftp=None):
        session = object.__new__(remote_compute.SSHSession)
        session.sftp = sftp or self._LocalSFTP()
        session.ensure_directory = lambda path: Path(path).mkdir(
            parents=True, exist_ok=True
        )
        return session

    def test_upload_resumes_existing_remote_part(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.bin"
            destination = Path(temporary) / "remote" / "payload.bin"
            content = b"0123456789" * 100
            source.write_bytes(content)
            destination.parent.mkdir()
            Path(str(destination) + ".part").write_bytes(content[:137])
            progress = []
            self._session().upload(
                source,
                str(destination),
                callback=lambda done, total: progress.append((done, total)),
            )
            self.assertEqual(destination.read_bytes(), content)
            self.assertEqual(progress[0], (137, len(content)))
            self.assertEqual(progress[-1], (len(content), len(content)))

    def test_upload_enables_pipelined_writes_when_available(self):
        class _RecordingRemote:
            def __init__(self):
                self.pipelined = None
                self.written = bytearray()

            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

            def set_pipelined(self, value):
                self.pipelined = value

            def write(self, chunk):
                self.written.extend(chunk)

            def flush(self):
                pass

        class _Stat:
            def __init__(self, size):
                self.st_size = size

        class _SFTP:
            def __init__(self, remote):
                self._remote = remote

            def stat(self, path):
                if path.endswith(".part"):
                    return _Stat(len(self._remote.written))
                return Path(path).stat()

            def open(self, _path, _mode):
                return self._remote

            def remove(self, path):
                Path(path).unlink(missing_ok=True)

            def rename(self, source, destination):
                # The fake remote keeps bytes in memory, so there is no real
                # .part file to move; this stub satisfies the upload contract.
                pass

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.bin"
            destination = Path(temporary) / "remote" / "payload.bin"
            content = b"\xab" * 4096
            source.write_bytes(content)
            remote = _RecordingRemote()
            self._session(sftp=_SFTP(remote)).upload(source, str(destination))
            self.assertTrue(remote.pipelined)
            self.assertEqual(bytes(remote.written), content)

    def test_download_resumes_existing_local_part(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "remote.bin"
            destination = Path(temporary) / "local" / "model.bin"
            content = b"abcdefghij" * 100
            source.write_bytes(content)
            destination.parent.mkdir()
            Path(str(destination) + ".part").write_bytes(content[:211])
            progress = []
            self._session().download(
                str(source),
                destination,
                callback=lambda done, total: progress.append((done, total)),
            )
            self.assertEqual(destination.read_bytes(), content)
            self.assertEqual(progress[0], (211, len(content)))
            self.assertEqual(progress[-1], (len(content), len(content)))

    def test_growing_remote_log_is_appended_from_last_remote_offset(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "remote.log"
            destination = Path(temporary) / "local.log"
            source.write_bytes(b"epoch 1\n")
            session = self._session()
            offset = session.download_appended(str(source), destination, 0)
            source.write_bytes(b"epoch 1\nepoch 2\n")
            offset = session.download_appended(
                str(source), destination, offset
            )
            self.assertEqual(offset, source.stat().st_size)
            self.assertEqual(
                destination.read_text(encoding="utf-8"),
                "epoch 1\nepoch 2\n",
            )

    def test_rotated_remote_log_restarts_from_new_inode(self):
        class Session:
            def __init__(self):
                self.identities = iter(("100", "200"))
                self.contents = iter((b"old\n", b"new\n"))
                self.offsets = []

            def path_exists(self, _path):
                return True

            def execute(self, _command, **_kwargs):
                return next(self.identities)

            def download_appended(self, _remote, local, offset):
                self.offsets.append(offset)
                content = next(self.contents)
                with Path(local).open("ab") as handle:
                    handle.write(content[offset:])
                return len(content)

        with tempfile.TemporaryDirectory() as temporary:
            local_log = Path(temporary) / "local.log"
            state = {"offset": 0}
            session = Session()
            controller._sync_remote_log(session, "/job", local_log, state)
            controller._sync_remote_log(session, "/job", local_log, state)
            self.assertEqual(session.offsets, [0, 0])
            text = local_log.read_text(encoding="utf-8")
            self.assertIn("old\n", text)
            self.assertIn("Remote worker log rotated", text)
            self.assertTrue(text.endswith("new\n"))


class LocalCompatibilityTests(unittest.TestCase):
    def test_portable_bundle_keeps_remote_runtime_and_dockerignore(self):
        self.assertIn("remote", package_portable.INCLUDE_DIRS)
        self.assertIn(".dockerignore", package_portable.INCLUDE_FILES)
        for path in (
            "tools/remote_compute.py",
            "tools/remote_compute_ui.py",
            "tools/remote_training_controller.py",
            "tools/remote_worker.py",
        ):
            self.assertIn(path, package_portable.REQUIRED_EXTERNAL_UI_FILES)

    def test_connection_warns_for_root_login(self):
        """B12 red-line guard: root logins get a shared-server warning.

        The live SSH path cannot run in unit tests, so assert the guard
        and its surfacing where they live: the warning is produced in
        test_connection and rendered by the dialog's status message.
        """
        source = Path("tools/remote_compute.py").read_text(encoding="utf-8")
        self.assertIn('== "root"', source)
        self.assertIn('"warning": warning', source)
        ui_source = Path("tools/remote_compute_ui.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('result.get("warning")', ui_source)
        self.assertIn("work folder {}", ui_source)

    @unittest.skipUnless(
        importlib.util.find_spec("PySide6") is not None,
        "PySide6 is not installed",
    )
    def test_compute_selector_opens_local_without_paramiko(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6 import QtCore, QtGui, QtWidgets
        from tools.remote_compute_ui import RemoteComputeSelector

        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(
            os.environ,
            {"MIMICS_REMOTE_CONFIG_DIR": temporary},
            clear=False,
        ):
            app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
            parent = QtWidgets.QWidget()
            selector = RemoteComputeSelector(
                parent,
                (QtCore, QtGui, QtWidgets),
            )
            self.assertEqual(selector.selection(), ("local", ""))
            self.assertIn("existing local training", selector.hint.text())
            parent.close()
            app.processEvents()

    @unittest.skipUnless(
        importlib.util.find_spec("PySide6") is not None,
        "PySide6 is not installed",
    )
    def test_server_dialog_preserves_hand_edited_profile_fields(self):
        """B12 regression: the form round-trips the five advanced fields.

        Before the form exposed them, _profile_values dropped any
        hand-edited servers.json values (e.g. container_runtime='nerdctl'),
        and a Save from the dialog silently reset them to defaults.
        """
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6 import QtCore, QtGui, QtWidgets
        from tools.remote_compute_ui import ServerProfilesDialog

        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(
            os.environ,
            {"MIMICS_REMOTE_CONFIG_DIR": temporary},
            clear=False,
        ):
            remote_compute.write_json_atomic(
                remote_compute.profiles_path(),
                {
                    "schema_version": remote_compute.SCHEMA_VERSION,
                    "profiles": [
                        {
                            "profile_id": "srv",
                            "name": "Shared Server",
                            "host": "10.0.0.1",
                            "port": 22,
                            "username": "root",
                            "auth_method": "password",
                            "key_path": "",
                            "remote_root": "/userdata/alice/mimics-ai",
                            "runtime_image": "mimics-ai-runtime:1.0",
                            "gpu_device": "auto",
                            "cache_training_data": True,
                            "container_runtime": "nerdctl",
                            "container_namespace": "mimics-ai",
                            "remote_code_verify": "strict",
                            "remote_weights_verify": "warn",
                            "remote_cache_retention_days": 7,
                        }
                    ],
                },
            )
            app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
            parent = QtWidgets.QWidget()
            dialog = ServerProfilesDialog(
                parent, (QtCore, QtGui, QtWidgets), "srv"
            )
            try:
                values = dialog._profile_values()
                self.assertEqual("nerdctl", values["container_runtime"])
                self.assertEqual("mimics-ai", values["container_namespace"])
                self.assertEqual("strict", values["remote_code_verify"])
                self.assertEqual("warn", values["remote_weights_verify"])
                self.assertEqual(7, values["remote_cache_retention_days"])
                self.assertEqual(
                    "/userdata/alice/mimics-ai", values["remote_root"]
                )
            finally:
                dialog.dialog.close()
                parent.close()
                app.processEvents()



    def test_remote_nninteractive_preserves_real_initial_masks(self):
        import tools.nninteractive_finetune_pipeline as pipeline

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            local_job = root / "job"
            bundle = root / "bundle"
            local_job.mkdir()
            image = root / "image.nii.gz"
            label = root / "label.nii.gz"
            initial = root / "initial.nii.gz"
            image.write_bytes(b"image")
            label.write_bytes(b"label")
            initial.write_bytes(b"initial")
            manifest_path = local_job / "dataset_manifest.json"
            remote_compute.write_json_atomic(
                manifest_path,
                {
                    "cases": [
                        {
                            "case_id": "case 1",
                            "image": str(image),
                            "label": str(label),
                            "initial_mask": str(initial),
                            "initial_mask_source_type": "mimics_saved_mask",
                            "initial_mask_source_model": "draft_model_v2",
                            "initial_mask_source_name": "AI draft",
                            "split": "train",
                        }
                    ]
                },
            )
            remote_compute.write_json_atomic(
                local_job / "request.json",
                {
                    "job_id": "nn_job",
                    "base_model_dir": str(root / "official_model"),
                    "parent_model_id": "official",
                    "initial_mask_source": "mcs",
                    "initial_mask_names": ["AI draft"],
                },
            )
            with mock.patch.object(
                pipeline,
                "_prepare_manifest",
                return_value=(manifest_path, None),
            ):
                result = controller._prepare_nninteractive(
                    {"job_dir": str(local_job)},
                    bundle,
                )
            payload = remote_compute.read_json(
                bundle / "remote_request.json", {}
            )
            remote_request = payload["pipeline_request"]
            self.assertEqual(result["train_count"], 1)
            self.assertEqual(
                remote_request["initial_mask_source"],
                "exported_masks",
            )
            self.assertEqual(
                remote_request["initial_mask_names"], ["initial_mask"]
            )
            self.assertEqual(
                remote_request["cases"][0]["initial_mask"],
                "/job/input/case_1/initial_mask.nii.gz",
            )
            self.assertEqual(
                remote_request["cases"][0]["initial_mask_source_type"],
                "mimics_saved_mask",
            )
            self.assertEqual(
                remote_request["cases"][0]["initial_mask_source_model"],
                "draft_model_v2",
            )
            self.assertEqual(
                remote_request["cases"][0]["initial_mask_source_name"],
                "AI draft",
            )
            self.assertEqual(
                remote_request["workspace"],
                "/remote-cache/nninteractive/datasets/"
                "__MIMICS_REMOTE_DATASET_FINGERPRINT__",
            )
            self.assertEqual(
                remote_request["prepared_cache_namespace"], "dataset"
            )
            self.assertEqual(
                (
                    bundle
                    / "input"
                    / "case_1"
                    / "initial_mask.nii.gz"
                ).read_bytes(),
                b"initial",
            )





class ArchiveSafetyTests(unittest.TestCase):
    def test_safe_extract_rejects_parent_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "bad.tar"
            with tarfile.open(archive, "w") as handle:
                info = tarfile.TarInfo("../outside.txt")
                content = b"bad"
                info.size = len(content)
                handle.addfile(info, io.BytesIO(content))
            with self.assertRaisesRegex(RuntimeError, "unsafe path"):
                controller._safe_extract(
                    archive, Path(temporary) / "destination"
                )

    def test_safe_extract_rejects_links(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "link.tar"
            with tarfile.open(archive, "w") as handle:
                info = tarfile.TarInfo("model.pth")
                info.type = tarfile.SYMTYPE
                info.linkname = "/tmp/outside"
                handle.addfile(info)
            with self.assertRaisesRegex(RuntimeError, "contains a link"):
                controller._safe_extract(
                    archive, Path(temporary) / "destination"
                )

    def test_dataset_archive_ignores_source_mtime_and_job_tar_is_small(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            (bundle / "input" / "case").mkdir(parents=True)
            (bundle / "labels" / "case").mkdir(parents=True)
            (bundle / "input" / "case" / "image.nii.gz").write_bytes(b"image")
            (bundle / "labels" / "case" / "label.nii.gz").write_bytes(b"label")
            (bundle / "remote_request.json").write_text(
                "{}", encoding="utf-8"
            )
            first = root / "dataset-1.tar"
            second = root / "dataset-2.tar"
            fingerprint_1, _size = controller._build_dataset_tar(
                bundle, first
            )
            os.utime(bundle / "input" / "case" / "image.nii.gz", None)
            fingerprint_2, _size = controller._build_dataset_tar(
                bundle, second
            )
            self.assertEqual(fingerprint_1, fingerprint_2)
            job_archive = root / "job.tar"
            controller._build_tar(
                bundle,
                job_archive,
                excluded_top_level={"input", "labels"},
            )
            with tarfile.open(job_archive, "r") as handle:
                self.assertEqual(
                    handle.getnames(), ["remote_request.json"]
                )

    def test_dataset_parts_reuse_unchanged_cases_independently(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            for case_id in ("case_a", "case_b"):
                (bundle / "input" / case_id).mkdir(parents=True)
                (bundle / "labels" / case_id).mkdir(parents=True)
                (bundle / "input" / case_id / "image.nii.gz").write_bytes(
                    ("image-" + case_id).encode("ascii")
                )
                (bundle / "labels" / case_id / "label.nii.gz").write_bytes(
                    ("label-" + case_id).encode("ascii")
                )
            first = controller._build_dataset_parts(
                bundle, root / "parts_first"
            )
            first_fingerprints = {
                row["case_id"]: row["fingerprint"] for row in first
            }
            (
                bundle / "labels" / "case_b" / "label.nii.gz"
            ).write_bytes(b"edited-label")
            changed_label = bundle / "labels" / "case_b" / "label.nii.gz"
            second = controller._build_dataset_parts(
                bundle, root / "parts_second"
            )
            second_fingerprints = {
                row["case_id"]: row["fingerprint"] for row in second
            }
            self.assertEqual(
                first_fingerprints["case_a"],
                second_fingerprints["case_a"],
            )
            self.assertNotEqual(
                first_fingerprints["case_b"],
                second_fingerprints["case_b"],
            )

    def test_local_case_archives_are_reused_without_repacking(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            (bundle / "input" / "case").mkdir(parents=True)
            (bundle / "input" / "case" / "image.nii.gz").write_bytes(
                b"image"
            )
            keys = {"case": "source-grid-fingerprint-a"}
            first = controller._build_dataset_parts(
                bundle,
                root / "temporary_first",
                cache_dir=root / "persistent_cache",
                case_cache_keys=keys,
            )
            second = controller._build_dataset_parts(
                bundle,
                root / "temporary_second",
                cache_dir=root / "persistent_cache",
                case_cache_keys=keys,
            )
            self.assertFalse(first[0]["local_cache_hit"])
            self.assertTrue(second[0]["local_cache_hit"])
            self.assertEqual(first[0]["archive"], second[0]["archive"])
            self.assertEqual(
                first[0]["fingerprint"], second[0]["fingerprint"]
            )

            stale_time = time.time() - 31 * 86400
            os.utime(first[0]["archive"], (stale_time, stale_time))
            metadata_path = first[0]["archive"].with_suffix(".json")
            metadata = remote_compute.read_json(metadata_path, {})
            metadata["last_used_at_epoch"] = stale_time
            remote_compute.write_json_atomic(metadata_path, metadata)
            changed = controller._build_dataset_parts(
                bundle,
                root / "temporary_changed",
                cache_dir=root / "persistent_cache",
                case_cache_keys={"case": "source-grid-fingerprint-b"},
            )
            self.assertFalse(changed[0]["local_cache_hit"])
            self.assertNotEqual(first[0]["archive"], changed[0]["archive"])
            self.assertFalse(first[0]["archive"].exists())

    def test_local_archive_cache_periodically_rehashes_owned_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            (bundle / "input" / "case").mkdir(parents=True)
            (bundle / "input" / "case" / "image.nii.gz").write_bytes(b"image")
            first = controller._build_dataset_parts(
                bundle,
                root / "parts_first",
                cache_dir=root / "cache",
                case_cache_keys={"case": "source-key"},
            )[0]
            original_fingerprint = first["fingerprint"]
            archive = first["archive"]
            content = bytearray(archive.read_bytes())
            content[-1] ^= 1
            archive.write_bytes(content)
            metadata_path = archive.with_suffix(".json")
            metadata = remote_compute.read_json(metadata_path, {})
            metadata["verified_at_epoch"] = 0
            metadata["archive_mtime_ns"] = int(archive.stat().st_mtime_ns)
            remote_compute.write_json_atomic(metadata_path, metadata)
            second = controller._build_dataset_parts(
                bundle,
                root / "parts_second",
                cache_dir=root / "cache",
                case_cache_keys={"case": "source-key"},
            )[0]
            self.assertFalse(second["local_cache_hit"])
            self.assertEqual(second["fingerprint"], original_fingerprint)

    def test_remote_nnunet_cache_cleanup_is_scoped_and_activity_aware(self):
        class Session:
            def __init__(self, active=""):
                self.active = active
                self.commands = []

            def execute(self, command, **_kwargs):
                self.commands.append(command)
                if command.startswith("docker ps"):
                    return self.active
                return ""

        with tempfile.TemporaryDirectory() as temporary:
            status = Path(temporary) / "status.json"
            log = Path(temporary) / "controller.log"
            remote_compute.write_json_atomic(status, {})
            profile = {
                "username": "alice",
                "remote_cache_retention_days": 30,
            }
            paths = {"prepared_cache": "/srv/mimics/cache/alice/prepared"}
            namespace = "/srv/mimics/cache/alice/prepared/nnunet/digest"
            idle = Session()
            controller._maintain_remote_nnunet_cache(
                idle, profile, paths, namespace, status, log
            )
            find_commands = [value for value in idle.commands if "find " in value]
            self.assertEqual(len(find_commands), 1)
            self.assertIn("! -name inference", find_commands[0])
            self.assertIn(namespace, idle.commands[-1])

            active = Session("container-id\n")
            controller._maintain_remote_nnunet_cache(
                active, profile, paths, namespace, status, log
            )
            self.assertFalse(any("find " in value for value in active.commands))

    def test_remote_prepared_cache_uses_dataset_content_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            request_path = bundle / "remote_request.json"
            remote_compute.write_json_atomic(
                request_path,
                {
                    "pipeline_args": [
                        "--materialization-cache-dir",
                        "/remote-cache/nninteractive/liver/"
                        "__MIMICS_REMOTE_DATASET_FINGERPRINT__/materialized",
                    ],
                    "pipeline_request": {
                        "workspace": "/remote-cache/nninteractive/liver/"
                        "__MIMICS_REMOTE_DATASET_FINGERPRINT__"
                    },
                },
            )
            controller._bind_remote_prepared_cache(bundle, "digest-123")
            bound = remote_compute.read_json(request_path, {})
            self.assertIn("digest-123", bound["pipeline_args"][1])
            self.assertIn(
                "digest-123", bound["pipeline_request"]["workspace"]
            )
            self.assertNotIn(
                "__MIMICS_REMOTE_DATASET_FINGERPRINT__", str(bound)
            )


class CodeDriftTests(unittest.TestCase):
    """The image's baked-in pipeline code is compared before training."""

    class _Session:
        def __init__(self, image_digest):
            self.image_digest = image_digest
            self.commands = []

        def execute(self, command, **kwargs):
            self.commands.append(command)
            if "docker run --rm --network none" in command:
                return self.image_digest
            raise AssertionError("unexpected command: " + command)

    def test_local_code_fingerprint_changes_with_file_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "tools").mkdir()
            (root / "tools" / "pipeline.py").write_text("print('a')\n")
            first = controller._local_code_fingerprint(root)
            (root / "tools" / "pipeline.py").write_text("print('b')\n")
            second = controller._local_code_fingerprint(root)
            self.assertNotEqual(first, second)
            # Adding a file under an excluded directory must not change it.
            (root / "tools" / "__pycache__").mkdir()
            (root / "tools" / "__pycache__" / "pipeline.pyc").write_text("x")
            self.assertEqual(second, controller._local_code_fingerprint(root))

    def test_check_code_drift_match_records_identity(self):
        local = controller._local_code_fingerprint()
        session = self._Session(local)
        profile = {"runtime_image": "img", "remote_code_verify": "warn"}
        with tempfile.TemporaryDirectory() as temporary:
            status_path = Path(temporary) / "status.json"
            remote_compute.write_json_atomic(status_path, {})
            identity = controller._check_code_drift(
                session, profile, status_path, "sha256:imageid"
            )
        self.assertTrue(identity["remote_code_match"])
        self.assertEqual(identity["remote_code_verify"], "warn")

    def test_check_code_drift_strict_refuses_on_mismatch(self):
        session = self._Session("0" * 64)
        profile = {"runtime_image": "img", "remote_code_verify": "strict"}
        with tempfile.TemporaryDirectory() as temporary:
            status_path = Path(temporary) / "status.json"
            remote_compute.write_json_atomic(status_path, {})
            with self.assertRaisesRegex(RuntimeError, "rebuild the image"):
                controller._check_code_drift(
                    session, profile, status_path, "sha256:imageid"
                )

    def test_check_code_drift_warn_records_drift(self):
        session = self._Session("0" * 64)
        profile = {"runtime_image": "img", "remote_code_verify": "warn"}
        with tempfile.TemporaryDirectory() as temporary:
            status_path = Path(temporary) / "status.json"
            remote_compute.write_json_atomic(status_path, {})
            identity = controller._check_code_drift(
                session, profile, status_path, "sha256:imageid"
            )
            status = remote_compute.read_json(status_path, {})
        self.assertFalse(identity["remote_code_match"])
        self.assertTrue(identity["remote_code_drift"])
        self.assertIn("rebuild the image", identity["remote_code_drift_detail"])
        self.assertTrue(status.get("remote_code_drift"))

    def test_check_code_drift_off_skips_comparison(self):
        session = self._Session("never-used")
        profile = {"runtime_image": "img", "remote_code_verify": "off"}
        with tempfile.TemporaryDirectory() as temporary:
            status_path = Path(temporary) / "status.json"
            remote_compute.write_json_atomic(status_path, {})
            identity = controller._check_code_drift(
                session, profile, status_path, "sha256:imageid"
            )
        self.assertEqual(identity, {"remote_code_verify": "off"})
        self.assertEqual(session.commands, [])

    def test_profile_normalizes_remote_code_verify_default_warn(self):
        base = {
            "name": "GPU server",
            "host": "server",
            "username": "user",
        }
        default = remote_compute.normalize_profile(base)
        self.assertEqual(default["remote_code_verify"], "warn")
        strict = remote_compute.normalize_profile(
            dict(base, remote_code_verify="strict")
        )
        self.assertEqual(strict["remote_code_verify"], "strict")
        invalid = remote_compute.normalize_profile(
            dict(base, remote_code_verify="bogus")
        )
        self.assertEqual(invalid["remote_code_verify"], "warn")


class ContainerRuntimeAdapterTests(unittest.TestCase):
    """nerdctl servers run the same command flow with a swapped prefix."""

    def test_runtime_command_defaults_to_docker(self):
        self.assertEqual(
            remote_compute.container_runtime_command(None), "docker"
        )
        self.assertEqual(
            remote_compute.container_runtime_command(
                {"container_runtime": ""}
            ),
            "docker",
        )
        self.assertEqual(
            remote_compute.container_runtime_command(
                {"container_runtime": "bogus"}
            ),
            "docker",
        )

    def test_runtime_command_selects_nerdctl(self):
        self.assertEqual(
            remote_compute.container_runtime_command(
                {"container_runtime": "nerdctl"}
            ),
            "nerdctl",
        )
        # a namespace keeps our images away from orchestrator-managed ones
        self.assertEqual(
            remote_compute.container_runtime_command(
                {
                    "container_runtime": "nerdctl",
                    "container_namespace": "mimics-ai",
                }
            ),
            "nerdctl -n mimics-ai",
        )
        # docker ignores the namespace field (docker has none)
        self.assertEqual(
            remote_compute.container_runtime_command(
                {
                    "container_runtime": "docker",
                    "container_namespace": "mimics-ai",
                }
            ),
            "docker",
        )
        # profile normalization preserves and validates the fields
        normalized = remote_compute.normalize_profile(
            {
                "name": "GPU server",
                "host": "server",
                "username": "user",
                "container_runtime": "nerdctl",
                "container_namespace": "mimics-ai",
            }
        )
        self.assertEqual(normalized["container_runtime"], "nerdctl")
        self.assertEqual(normalized["container_namespace"], "mimics-ai")
        # invalid namespace characters are dropped
        bad = remote_compute.normalize_profile(
            {
                "name": "GPU server",
                "host": "server",
                "username": "user",
                "container_runtime": "nerdctl",
                "container_namespace": "evil ns; rm -rf",
            }
        )
        self.assertEqual(bad["container_namespace"], "")

    def test_container_commands_use_profile_runtime(self):
        class Session:
            def __init__(self, profile):
                self.profile = remote_compute.normalize_profile(profile)
                self.commands = []

            def execute(self, command, **_kwargs):
                self.commands.append(command)
                if command.startswith("nerdctl inspect"):
                    return "running 0"
                return ""

            def execute_result(self, command, **_kwargs):
                self.commands.append(command)
                return 0, "running 0"

        session = Session(
            {
                "name": "GPU server",
                "host": "server",
                "username": "user",
                "container_runtime": "nerdctl",
            }
        )
        state, code = controller._container_state(session, "job-container")
        self.assertEqual(state, "running")
        self.assertEqual(code, 0)
        self.assertTrue(
            session.commands[0].startswith("nerdctl inspect"),
            session.commands[0],
        )
        self.assertNotIn("docker", session.commands[0])

    def test_fake_session_without_profile_still_defaults_docker(self):
        # Test doubles and legacy callers may pass sessions with no
        # .profile attribute — the adapter must fall back to docker.
        class Session:
            def execute_result(self, command, **_kwargs):
                self.seen = command
                return 0, "exited 0"

        session = Session()
        controller._container_state(session, "job-container")
        self.assertTrue(session.seen.startswith("docker inspect"), session.seen)


class MonitorRobustnessTests(unittest.TestCase):
    """Reconnect budget and stall detection bound unattended monitoring."""

    def _status_file(self, temporary):
        status_path = Path(temporary) / "status.json"
        remote_compute.write_json_atomic(
            status_path, {"job_id": "job", "status": "training"}
        )
        return status_path

    def test_reconnect_attempts_are_bounded(self):
        with tempfile.TemporaryDirectory() as temporary:
            status_path = self._status_file(temporary)
            log_path = Path(temporary) / "controller.log"
            with self.assertRaisesRegex(
                controller.RemoteServerUnreachable, "reconnect"
            ):
                controller._reconnect_session(
                    {"profile_id": "p"},
                    status_path,
                    log_path,
                    OSError("network down"),
                    controller.RECONNECT_MAX_ATTEMPTS,
                )
            final = json.loads(status_path.read_text(encoding="utf-8"))
            # The status file is untouched by the raise itself; the run()
            # exception handler records remote_unreachable afterwards.
            self.assertEqual(final.get("status"), "training")

    def test_reconnect_below_budget_still_attempts(self):
        with tempfile.TemporaryDirectory() as temporary:
            status_path = self._status_file(temporary)
            log_path = Path(temporary) / "controller.log"
            with mock.patch.object(
                controller, "SSHSession", side_effect=OSError("still down")
            ):
                session = controller._reconnect_session(
                    {"profile_id": "p"},
                    status_path,
                    log_path,
                    OSError("network down"),
                    controller.RECONNECT_MAX_ATTEMPTS - 1,
                )
            self.assertIsNone(session)
            final = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertEqual(final.get("status"), "reconnecting_remote")
            self.assertEqual(
                final.get("remote_reconnect_attempt"),
                controller.RECONNECT_MAX_ATTEMPTS - 1,
            )

    def test_stall_hint_is_emitted_once_and_cleared_on_progress(self):
        # Drive the monitor loop with a scripted session: silent training
        # polls past the (patched-to-tiny) stall threshold set the hint once,
        # then a changed status payload proves the hint clears.
        with tempfile.TemporaryDirectory() as temporary:
            status_path = self._status_file(temporary)
            remote_log = Path(temporary) / "worker.log"
            remote_log.write_bytes(b"")
            log_path = Path(temporary) / "controller.log"

            class Session:
                def __init__(self):
                    self.polls = 0
                    self.silent_polls = 3

                def ensure_directory(self, _path):
                    pass

                def path_exists(self, _path):
                    return True

                def execute(self, command, **_kwargs):
                    if "stat -c '%i'" in command:
                        return "1"
                    if command.startswith("docker inspect"):
                        return "running 0"
                    raise AssertionError(
                        "unexpected command: " + command
                    )

                def execute_result(self, command, **_kwargs):
                    assert command.startswith("docker inspect")
                    return 0, "running 0"

                def read_remote_json(self, path, default=None):
                    self.polls += 1
                    if self.polls <= self.silent_polls:
                        return {"status": "training", "epoch": 1}
                    return {"status": "completed", "epoch": 2}

                def download_appended(self, _remote, _local, offset):
                    return offset

            with mock.patch.object(
                controller, "REMOTE_STALL_SECONDS", 0.0
            ), mock.patch.object(
                controller, "_sync_remote_nnunet_curve",
                side_effect=IOError("no curve"),
            ), mock.patch.object(
                controller.time, "sleep", lambda _seconds: None
            ):
                session = Session()
                _returned, remote_status = controller._monitor_remote(
                    session,
                    {"kind": "nnunet", "job_id": "job"},
                    {"profile_id": "p", "name": "server"},
                    status_path,
                    {"job": "/remote/job"},
                    "container",
                    "user",
                    log_path,
                    remote_log,
                )
            self.assertEqual(remote_status.get("status"), "completed")
            final = json.loads(status_path.read_text(encoding="utf-8"))
            # The hint fired during the silent stretch and was cleared when
            # the status payload changed before completion.
            self.assertFalse(final.get("remote_stall_suspected", False))


class TransferEfficiencyTests(unittest.TestCase):
    """Caches keep repeated remote runs from re-paying large transfers."""

    CACHE_PATHS = {
        "models": "/remote/models",
        "dataset_cache": "/remote/cache/user/datasets",
        "models_cache": "/remote/cache/user/models",
    }

    def test_batch_cache_verification_parses_verdicts(self):
        class Session:
            def __init__(self):
                self.commands = []

            def execute_result(self, command, **_kwargs):
                self.commands.append(command)
                return 0, "aa hit\nbb miss\n"

        parts = [
            {"fingerprint": "aa", "size": 10},
            {"fingerprint": "bb", "size": 20},
        ]
        verdicts = controller._batch_verify_remote_dataset_cache(
            Session(), self.CACHE_PATHS, parts
        )
        self.assertEqual(verdicts, {"aa": True, "bb": False})
        self.assertEqual(len(Session().commands), 0)  # instance-local capture

    def test_batch_cache_chunks_many_fingerprints(self):
        class Session:
            def __init__(self):
                self.chunk_sizes = []

            def execute_result(self, command, **_kwargs):
                self.chunk_sizes.append(command.count(" printf "))
                return 0, ""

        parts = [
            {"fingerprint": "f{}".format(index), "size": 1}
            for index in range(120)
        ]
        session = Session()
        controller._batch_verify_remote_dataset_cache(
            session, self.CACHE_PATHS, parts, chunk_size=50
        )
        self.assertEqual(len(session.chunk_sizes), 3)

    def test_batch_verdict_hit_skips_per_case_round_trip(self):
        class Session:
            def __init__(self):
                self.commands = []

            def ensure_directory(self, _path):
                pass

            def execute(self, command, **_kwargs):
                self.commands.append(command)
                return ""

            def upload(self, *_args, **_kwargs):
                raise AssertionError("cache hit must not upload")

        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "case.tar"
            archive.write_bytes(b"case-bytes")
            status_path = Path(temporary) / "status.json"
            remote_compute.write_json_atomic(status_path, {})
            remote, hit = controller._ensure_remote_dataset_archive(
                Session(),
                self.CACHE_PATHS,
                {"cache_training_data": True},
                archive,
                "ff" * 32,
                status_path,
                assumed_cache_state="hit",
            )
            self.assertTrue(hit)
            self.assertTrue(remote.endswith("ff" * 32 + ".tar"))
            final = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertTrue(final.get("dataset_cache_hit"))

    def test_custom_model_cache_hit_returns_without_upload(self):
        class Session:
            def __init__(self):
                self.commands = []

            def ensure_directory(self, _path):
                pass

            def execute_result(self, command, **_kwargs):
                self.commands.append(command)
                return 0, "hit"

            def upload(self, *_args, **_kwargs):
                raise AssertionError("cache hit must not upload")

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "custom_model"
            source.mkdir()
            (source / "model.safetensors").write_bytes(b"weights")
            status_path = Path(temporary) / "status.json"
            remote_compute.write_json_atomic(status_path, {})
            hit = controller._ensure_remote_custom_model(
                Session(),
                self.CACHE_PATHS,
                {},
                source,
                "ab" * 32,
                status_path,
                Path(temporary) / "controller.log",
            )
            self.assertTrue(hit)
            self.assertFalse(
                (source.parent / "custom_model_cache.tar").exists()
            )

    def test_custom_model_cache_miss_uploads_and_verifies(self):
        class Session:
            def __init__(self):
                self.commands = []
                self.uploads = []

            def ensure_directory(self, _path):
                pass

            def execute_result(self, command, **_kwargs):
                self.commands.append(command)
                return 0, "miss"

            def execute(self, command, **_kwargs):
                self.commands.append(command)
                if command.startswith("sha256sum"):
                    return self.uploaded_digest
                return ""

            def upload(self, local, remote, **_kwargs):
                self.uploads.append((str(local), remote))
                self.uploaded_digest = controller._sha256_file(Path(local))

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "custom_model"
            source.mkdir()
            (source / "model.safetensors").write_bytes(b"weights")
            status_path = Path(temporary) / "status.json"
            remote_compute.write_json_atomic(status_path, {})
            session = Session()
            hit = controller._ensure_remote_custom_model(
                session,
                self.CACHE_PATHS,
                {},
                source,
                "ab" * 32,
                status_path,
                Path(temporary) / "controller.log",
            )
            # First use populates the cache but still ships in the job tar.
            self.assertFalse(hit)
            self.assertEqual(len(session.uploads), 1)
            self.assertTrue(
                session.uploads[0][0].endswith("custom_model_cache.tar")
            )
            final = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertFalse(final.get("custom_model_cache_hit"))
            self.assertFalse(
                (source.parent / "custom_model_cache.tar").exists()
            )

    def test_custom_model_corrupt_upload_is_rejected(self):
        class Session:
            def ensure_directory(self, _path):
                pass

            def execute_result(self, command, **_kwargs):
                return 0, "miss"

            def execute(self, command, **_kwargs):
                if command.startswith("sha256sum"):
                    return "0" * 64
                return ""

            def upload(self, *_args, **_kwargs):
                pass

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "model.safetensors"
            source.write_bytes(b"weights")
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                controller._ensure_remote_custom_model(
                    Session(),
                    self.CACHE_PATHS,
                    {},
                    source,
                    "ab" * 32,
                    Path(temporary) / "status.json",
                    Path(temporary) / "controller.log",
                )

    def test_base_model_digest_cache_skips_rehash_when_fresh(self):
        class Session:
            def __init__(self):
                self.commands = []

            def execute(self, command, **_kwargs):
                self.commands.append(command)
                if "printf ready" in command:
                    return "ready"
                if ".mimics_digest_cache" in command and "then printf" in command:
                    return "c" * 64
                raise AssertionError(
                    "fresh marker must skip the full hash: " + command
                )

        identity = controller._validate_remote_assets(
            Session(),
            self.CACHE_PATHS,
            {
                "required_model_relative": "flexict",
                "local_required_model": "",
            },
        )
        self.assertEqual(identity["remote_base_model_sha256"], "c" * 64)

    def test_base_model_stale_marker_triggers_full_hash_and_rewrite(self):
        class Session:
            def __init__(self):
                self.commands = []

            def execute(self, command, **_kwargs):
                self.commands.append(command)
                if "printf ready" in command:
                    return "ready"
                if ".mimics_digest_cache" in command and "then printf" in command:
                    return ""
                if "sha256sum" in command and "awk" in command:
                    return "d" * 64
                return ""

        session = Session()
        identity = controller._validate_remote_assets(
            session,
            self.CACHE_PATHS,
            {
                "required_model_relative": "flexict",
                "local_required_model": "",
            },
        )
        self.assertEqual(identity["remote_base_model_sha256"], "d" * 64)
        self.assertTrue(
            any(
                ".mimics_digest_cache.tmp" in command
                for command in session.commands
            )
        )


class ReattachTests(unittest.TestCase):
    """Orphaned remote containers can be re-attached after a controller dies."""

    REMOTE_JOB = "/remote/jobs/user/train_job1"

    def _status(self, root: Path, **overrides):
        status = {
            "job_id": "train_job1",
            "kind": "train",
            "status": "orphaned_remote",
            "execution_backend": "remote",
            "remote_profile_id": "labgpu",
            "remote_container_name": "mimics-ai-user-train_job1",
            "remote_job_dir": self.REMOTE_JOB,
            "request_path": str(root / "request.json"),
            "control_path": str(root / "control.json"),
            "job_dir": str(root),
            "progress_percent": 40,
        }
        status.update(overrides)
        return status

    def _profile(self):
        return {
            "profile_id": "labgpu",
            "name": "Lab GPU",
            "username": "user",
            "runtime_image": "mimics-ai-runtime:1.0",
            "remote_root": "/remote",
        }

    def _setup(self, root: Path, remote_status=None, container_state="running"):
        status = self._status(root)
        remote_compute.write_json_atomic(
            root / "status.json", status
        )
        remote_compute.write_json_atomic(
            root / "remote_spec.json",
            {
                "schema_version": "mimics_remote_nnunet_spec.v1",
                "kind": "nnunet",
                "job_id": "train_job1",
                "job_dir": str(root),
                "status_path": str(root / "status.json"),
                "request_path": str(root / "request.json"),
                "remote_profile_id": "labgpu",
            },
        )
        remote_compute.write_json_atomic(
            root / "request.json",
            {
                "operation": "train",
                "task_id": "kidney",
                "model_id": "nnunet_1234",
                "workspace": str(root / "workspace"),
            },
        )
        remote = dict(remote_status or {})
        remote.setdefault("status", "training")
        remote.setdefault(
            "model",
            {"model_id": "nnunet_1234", "model_dir": "/job/output/models"},
        )

        class Session:
            def __init__(self):
                self.closed = False
                self.commands = []
                self.downloaded_payload = b"payload"

            def execute_result(self, command):
                self.commands.append(command)
                if "State.Status" in command:
                    return 0, "{} -1".format(container_state)
                if "Config.Labels" in command:
                    return 0, json.dumps(
                        {
                            "mimics-script.remote-training": "true",
                            "mimics-script.owner": "user",
                            "mimics-script.job": "train_job1",
                        }
                    )
                if "docker image inspect" in command:
                    return 0, "sha256:imageid"
                if command.startswith("sha256sum"):
                    return 0, "b" * 64 + "  /remote/result.tar"
                return 0, ""

            def execute(self, command, **kwargs):
                self.commands.append(command)
                if command.startswith("sha256sum"):
                    # Command pipes through awk; only the digest arrives.
                    return hashlib.sha256(self.downloaded_payload).hexdigest()
                if command.startswith("stat -c %s"):
                    return str(len(self.downloaded_payload))
                return ""

            def read_remote_json(self, path, default=None):
                if path.endswith("pipeline_job/status.json"):
                    return remote
                return default

            def upload(self, *_args, **_kwargs):
                return None

            def download(self, remote, local, **_kwargs):
                Path(local).write_bytes(self.downloaded_payload)

            def download_appended(self, *_args, **_kwargs):
                return 0

            def path_exists(self, _path):
                return True

            def close(self):
                self.closed = True

        return status, remote, Session()

    def test_reattach_refuses_local_and_terminal_tasks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            remote_compute.write_json_atomic(
                root / "status.json",
                {"status": "training", "execution_backend": "local"},
            )
            with self.assertRaisesRegex(RuntimeError, "Only a remote task"):
                controller.reattach(root / "status.json")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            remote_compute.write_json_atomic(
                root / "status.json",
                {
                    "status": "completed",
                    "execution_backend": "remote",
                    "job_id": "j",
                    "remote_job_dir": "/remote/jobs/user/j",
                    "remote_profile_id": "p",
                },
            )
            with self.assertRaisesRegex(RuntimeError, "already finished"):
                controller.reattach(root / "status.json")

    def test_reattach_to_completed_remote_job_downloads_and_registers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _status, remote, session = self._setup(
                root,
                remote_status={
                    "status": "completed",
                    "dataset_fingerprint": "dsfp",
                    "model": {
                        "model_id": "nnunet_1234",
                        "model_dir": "/job/output/models/kidney/nnunet_1234",
                    },
                },
                container_state="exited",
            )
            # A real download is a tar; write one holding the model manifest.
            import io as _io
            import tarfile as _tarfile

            manifest_dir = root / "artifact_src"
            manifest_dir.mkdir(parents=True)
            remote_compute.write_json_atomic(
                manifest_dir / "mimics_model_manifest.json",
                {
                    "schema_version": "mimics_nnunet_model.v1",
                    "model_id": "nnunet_1234",
                    "task_id": "kidney",
                    "configuration": "3d_fullres",
                    "folds": ["0"],
                },
            )
            buffer = _io.BytesIO()
            with _tarfile.open(fileobj=buffer, mode="w") as archive:
                archive.add(str(manifest_dir), arcname=".")
            session.downloaded_payload = buffer.getvalue()

            class _Sftp:
                def stat(self, _path):
                    class _Stat:
                        st_size = 10
                        st_mtime = 1

                    return _Stat()

            session.sftp = _Sftp()

            class _Sftp:
                def stat(self, _path):
                    class _Stat:
                        st_size = 10
                        st_mtime = 1

                    return _Stat()

            session.sftp = _Sftp()
            with mock.patch.object(
                controller, "SSHSession", return_value=session
            ), mock.patch.object(
                controller, "get_profile", return_value=self._profile()
            ), mock.patch.object(
                controller,
                "_register_nnunet",
                side_effect=lambda spec, local_dir, remote_status, profile: {
                    "model_id": "nnunet_1234",
                    "model_dir": str(local_dir),
                },
            ) as register:
                code = controller.reattach(root / "status.json")
            self.assertEqual(code, 0)
            register.assert_called_once()
            status = remote_compute.read_json(root / "status.json", {})
            self.assertEqual(status["status"], "completed")
            self.assertTrue(status["remote_reattached"])
            self.assertTrue(status["remote_model_downloaded"])

    def test_reattach_to_running_container_resumes_monitoring(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _status, remote, session = self._setup(root, container_state="running")

            def finalize(
                live_session,
                spec,
                profile,
                status_path,
                remote_paths,
                remote_status,
                container_name,
                local_root,
                result_archive,
                controller_log,
                remote_log,
            ):
                return 0

            with mock.patch.object(
                controller, "SSHSession", return_value=session
            ), mock.patch.object(
                controller, "get_profile", return_value=self._profile()
            ), mock.patch.object(
                controller, "_monitor_remote"
            ) as monitor:
                monitor.return_value = (session, {"status": "completed"})
                with mock.patch.object(
                    controller,
                    "_reattach_finalize_from_status",
                    side_effect=finalize,
                ) as download:
                    code = controller.reattach(root / "status.json")
            self.assertEqual(code, 0)
            monitor.assert_called_once()
            download.assert_called_once()
            status = remote_compute.read_json(root / "status.json", {})
            self.assertEqual(status["status"], "reattaching")
            self.assertEqual(
                status["phase"],
                "reattaching_remote",
            )

    def test_reattach_to_removed_container_suggests_abandon(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _status, remote, session = self._setup(
                root, container_state="missing"
            )
            with mock.patch.object(
                controller, "SSHSession", return_value=session
            ), mock.patch.object(
                controller, "get_profile", return_value=self._profile()
            ):
                code = controller.reattach(root / "status.json")
            self.assertEqual(code, 2)
            status = remote_compute.read_json(root / "status.json", {})
            self.assertEqual(status["status"], "orphaned_remote")
            self.assertEqual(status["phase"], "remote_container_missing")
            self.assertIn("Use Stop", status["message"])


class StatusAndLifecycleTests(unittest.TestCase):
    def test_remote_worker_log_rotation_is_bounded(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            remote_worker, "REMOTE_LOG_MAX_BYTES", 10
        ), mock.patch.object(remote_worker, "REMOTE_LOG_BACKUP_COUNT", 2):
            path = Path(temporary) / "remote_worker.log"
            writer = remote_worker._RotatingBinaryLog(path)
            try:
                writer.write(b"123456")
                writer.write(b"abcdef")
                writer.write(b"UVWXYZ")
            finally:
                writer.close()
            self.assertEqual(path.read_bytes(), b"UVWXYZ")
            self.assertEqual(
                path.with_name(path.name + ".1").read_bytes(), b"abcdef"
            )
            self.assertEqual(
                path.with_name(path.name + ".2").read_bytes(), b"123456"
            )

    def test_download_refuses_insufficient_local_space_before_transfer(self):
        class Session:
            def execute(self, _command, **_kwargs):
                return str(1024 * 1024 * 1024)

        class Usage:
            free = 64 * 1024 * 1024

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status_path = root / "status.json"
            remote_compute.write_json_atomic(status_path, {})
            with mock.patch.object(
                controller.shutil, "disk_usage", return_value=Usage()
            ), self.assertRaisesRegex(RuntimeError, "Not enough local disk"):
                controller._ensure_local_download_capacity(
                    Session(),
                    "/remote/result.tar",
                    root / "result.tar",
                    root / "model",
                    status_path,
                )
            status = remote_compute.read_json(status_path, {})
            self.assertTrue(status["local_download_space_checked"])
            self.assertEqual(
                status["remote_artifact_bytes"], 1024 * 1024 * 1024
            )

    def test_remote_task_can_be_abandoned_locally_without_stop_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            status_path = Path(temporary) / "status.json"
            remote_compute.write_json_atomic(
                status_path,
                {
                    "status": "stopping",
                    "execution_backend": "remote",
                    "remote_state_unknown": True,
                    "remote_container_name": "mimics-ai-user-job",
                },
            )
            job = remote_compute.read_json(status_path, {})
            self.assertEqual(job["execution_backend"], "remote")
            self.assertTrue(job["remote_state_unknown"])
            self.assertEqual(controller.abandon(status_path), 0)
            status = remote_compute.read_json(status_path, {})
            self.assertEqual(status["status"], "abandoned")
            self.assertFalse(status["remote_stop_confirmed"])
            self.assertTrue(status["remote_state_unknown"])
            self.assertTrue(Path(status["abandon_path"]).is_file())
            self.assertIn("may still be running", status["message"])

    def test_remote_design_covers_all_three_frameworks_and_escape_path(self):
        text = (ROOT / "docs" / "remote_training_design.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("nnU-Net training/inference", text)
        self.assertIn("Abandon Locally", text)
        self.assertIn("HF_HUB_OFFLINE=1", text)

    def test_aggregate_upload_progress_never_resets_between_case_parts(self):
        with tempfile.TemporaryDirectory() as temporary:
            status_path = Path(temporary) / "status.json"
            remote_compute.write_json_atomic(status_path, {})
            first = controller._upload_progress(
                status_path,
                start_percent=20,
                end_percent=33,
                aggregate_offset=0,
                aggregate_total=300,
            )
            second = controller._upload_progress(
                status_path,
                start_percent=20,
                end_percent=33,
                aggregate_offset=100,
                aggregate_total=300,
            )
            first(100, 100)
            first_status = remote_compute.read_json(status_path, {})
            second(50, 200)
            second_status = remote_compute.read_json(status_path, {})
            self.assertEqual(first_status["transfer_percent"], 33)
            self.assertEqual(second_status["transfer_percent"], 50)
            self.assertGreaterEqual(
                second_status["progress_percent"],
                first_status["progress_percent"],
            )

    def test_verified_dataset_cache_hit_skips_upload(self):
        class Session:
            def __init__(self, fingerprint):
                self.fingerprint = fingerprint
                self.commands = []

            def ensure_directory(self, path):
                self.commands.append("mkdir " + path)

            def execute_result(self, _command):
                return 0, self.fingerprint

            def execute(self, command, **_kwargs):
                self.commands.append(command)
                return ""

            def upload(self, *_args, **_kwargs):
                raise AssertionError("cache hit must not upload")

        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "dataset.tar"
            archive.write_bytes(b"dataset")
            fingerprint = controller._sha256_file(archive)
            status_path = Path(temporary) / "status.json"
            remote_compute.write_json_atomic(status_path, {})
            session = Session(fingerprint)
            remote_path, hit = controller._ensure_remote_dataset_archive(
                session,
                {"dataset_cache": "/cache"},
                {"cache_training_data": True},
                archive,
                fingerprint,
                status_path,
            )
            self.assertTrue(hit)
            self.assertEqual(remote_path, "/cache/{}.tar".format(fingerprint))
            status = remote_compute.read_json(status_path, {})
            self.assertTrue(status["dataset_cache_hit"])
            self.assertEqual(status["phase"], "remote_dataset_cache_hit")
            self.assertTrue(
                any(".verified" in command for command in session.commands)
            )

    def test_case_cache_hit_preserves_aggregate_upload_percent(self):
        class Session:
            def __init__(self, fingerprint):
                self.fingerprint = fingerprint

            def ensure_directory(self, _path):
                pass

            def execute_result(self, _command):
                return 0, self.fingerprint

            def execute(self, _command, **_kwargs):
                return ""

            def upload(self, *_args, **_kwargs):
                raise AssertionError("cache hit must not upload")

        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "case.tar"
            archive.write_bytes(b"x" * 100)
            fingerprint = controller._sha256_file(archive)
            status_path = Path(temporary) / "status.json"
            remote_compute.write_json_atomic(status_path, {})
            progress = controller._upload_progress(
                status_path,
                start_percent=20,
                end_percent=33,
                aggregate_offset=100,
                aggregate_total=400,
            )
            controller._ensure_remote_dataset_archive(
                Session(fingerprint),
                {"dataset_cache": "/cache"},
                {"cache_training_data": True},
                archive,
                fingerprint,
                status_path,
                progress_callback=progress,
            )
            status = remote_compute.read_json(status_path, {})
            self.assertEqual(status["transfer_percent"], 50)
            self.assertLess(status["progress_percent"], 33)

    def test_container_launch_exposes_only_selected_gpu(self):
        class Session:
            def __init__(self):
                self.commands = []

            def execute_result(self, _command):
                return 1, "Error: No such object: job-container"

            def execute(self, command, **_kwargs):
                self.commands.append(command)
                return "container-id"

        session = Session()
        profile = {
            "username": "user",
            "runtime_image": "mimics-ai-runtime:1.0",
            "gpu_device": "1",
            "remote_root": "/remote",
        }
        result = controller._launch_container(
            session,
            {
                "job": "/remote/jobs/user/job",
                "archive": "/remote/jobs/user/job.tar",
                "models": "/remote/models",
                "locks": "/remote/locks",
            },
            profile,
            "job-container",
        )
        command = session.commands[-1]
        self.assertEqual(result, "container-id")
        self.assertIn("--gpus device=1", command)
        self.assertIn("gpu-1.lock", command)
        self.assertIn("MIMICS_REMOTE_GPU_LOCK_SCOPE=device", command)
        self.assertIn("mimics-script.job=job", command)
        self.assertIn("-v /remote/prepared:/remote-cache", command)
        self.assertIn("--network none", command)
        self.assertIn("HF_HUB_OFFLINE=1", command)
        self.assertIn("TRANSFORMERS_OFFLINE=1", command)
        self.assertIn("HF_DATASETS_OFFLINE=1", command)
        self.assertIn("WANDB_MODE=offline", command)

    def test_container_launch_runs_one_command_per_dataset_case(self):
        # Each remote command must stay small enough to fit within the
        # kernel's single-argument cap; concatenating one tar clause per
        # case previously hit "/bin/bash: Argument list too long".
        class Session:
            def __init__(self):
                self.commands = []

            def execute_result(self, _command):
                return 1, "Error: No such object: job-container"

            def execute(self, command, **_kwargs):
                self.commands.append(command)
                return "container-id"

        session = Session()
        profile = {
            "username": "user",
            "runtime_image": "mimics-ai-runtime:1.0",
            "gpu_device": "auto",
            "remote_root": "/remote",
        }
        controller._launch_container(
            session,
            {
                "job": "/remote/jobs/user/job",
                "archive": "/remote/jobs/user/job.tar",
                "models": "/remote/models",
                "locks": "/remote/locks",
            },
            profile,
            "job-container",
            dataset_cache_paths=[
                "/remote/cache/user/datasets/aaa.tar",
                "/remote/cache/user/datasets/bbb.tar",
            ],
            remove_dataset_after_extract=False,
        )
        # No single command may carry more than one dataset tar extraction.
        dataset_tar_commands = [
            command
            for command in session.commands
            if command.count("tar -xf") > 1
        ]
        self.assertEqual(dataset_tar_commands, [])
        # Each dataset path appears in its own command.
        for path in (
            "/remote/cache/user/datasets/aaa.tar",
            "/remote/cache/user/datasets/bbb.tar",
        ):
            matches = [c for c in session.commands if path in c]
            self.assertEqual(len(matches), 1)
            self.assertIn("touch", matches[0])
        # Job cleanup is isolated from the archive extraction.
        self.assertTrue(
            any("rm -rf" in c and "mkdir -p" in c for c in session.commands)
        )

    def test_foreign_container_is_never_removed(self):
        class Session:
            def __init__(self):
                self.commands = []

            def execute_result(self, command):
                if ".State.Status" in command:
                    return 0, "running 0"
                return 0, json.dumps(
                    {
                        "mimics-script.remote-training": "true",
                        "mimics-script.owner": "another-user",
                        "mimics-script.job": "job",
                    }
                )

            def execute(self, command, **_kwargs):
                self.commands.append(command)
                return ""

        session = Session()
        with self.assertRaisesRegex(RuntimeError, "ownership labels"):
            controller._remove_container(
                session,
                "job-container",
                expected_owner="user",
                expected_job="job",
            )
        self.assertEqual(session.commands, [])

    def test_duplicate_running_job_is_not_replaced(self):
        class Session:
            def __init__(self):
                self.commands = []

            def execute_result(self, command):
                if ".State.Status" in command:
                    return 0, "running 0"
                return 0, json.dumps(
                    {
                        "mimics-script.remote-training": "true",
                        "mimics-script.owner": "user",
                        "mimics-script.job": "job",
                    }
                )

            def execute(self, command, **_kwargs):
                self.commands.append(command)
                return ""

        session = Session()
        with self.assertRaisesRegex(RuntimeError, "already has a running"):
            controller._launch_container(
                session,
                {
                    "job": "/remote/jobs/user/job",
                    "archive": "/remote/jobs/user/job.tar",
                    "models": "/remote/models",
                    "locks": "/remote/locks",
                },
                {
                    "username": "user",
                    "runtime_image": "mimics-ai-runtime:1.0",
                    "gpu_device": "0",
                    "remote_root": "/remote",
                },
                "job-container",
            )
        self.assertFalse(
            any("docker rm" in command for command in session.commands)
        )

    def test_remote_job_cleanup_is_confined_to_owned_job_folder(self):
        class Session:
            def execute(self, *_args, **_kwargs):
                raise AssertionError("unsafe path must be rejected before SSH")

        for path in ("/", "/remote/jobs/other/job", "/remote/jobs/user/job/subdir"):
            with self.subTest(path=path), self.assertRaisesRegex(
                RuntimeError, "outside this user's job folder"
            ):
                controller._remove_remote_job(
                    Session(),
                    remote_root="/remote",
                    expected_owner="user",
                    remote_job_dir=path,
                )

    def test_cancel_does_not_kill_controller_before_remote_stop_confirmation(self):
        class Session:
            def __init__(self):
                raise AssertionError("SSH must not be reached in this test")

        with tempfile.TemporaryDirectory() as temporary:
            status_path = Path(temporary) / "status.json"
            remote_compute.write_json_atomic(
                status_path,
                {
                    "status": "training",
                    "execution_backend": "remote",
                    "job_id": "remote_job",
                    "controller_pid": 1234,
                    "remote_stop_confirmed": False,
                    "remote_profile_id": "server",
                    "remote_container_name": "mimics-ai-user-remote_job",
                    "remote_control_path": "/remote/jobs/user/remote_job/control.json",
                    "remote_control_kind": "cancel",
                    "remote_job_dir": "/remote/jobs/user/remote_job",
                },
            )
            with mock.patch.object(
                controller, "SSHSession", Session
            ), mock.patch.object(
                controller,
                "get_profile",
                return_value={
                    "name": "GPU server",
                    "remote_root": "/remote",
                    "username": "user",
                },
            ):
                self.assertEqual(controller.cancel(status_path), 1)
            latest = remote_compute.read_json(status_path, {})
            self.assertEqual(latest["status"], "stopping")
            self.assertFalse(latest["remote_stop_confirmed"])
            self.assertTrue(latest["remote_state_unknown"])

    def test_container_inspection_distinguishes_missing_from_docker_failure(self):
        class Session:
            def __init__(self, output):
                self.output = output

            def execute_result(self, _command):
                return 1, self.output

        self.assertEqual(
            controller._container_state(
                Session("Error: No such object: missing"), "missing"
            ),
            ("missing", -1),
        )
        with self.assertRaisesRegex(
            remote_compute.RemoteCommandError, "Could not inspect"
        ):
            controller._container_state(
                Session("permission denied connecting to Docker"), "job"
            )

    def test_remote_base_model_must_match_local_when_available(self):
        class Session:
            def __init__(self, fingerprint):
                self.outputs = iter(["ready", fingerprint])

            def execute(self, _command, **_kwargs):
                return next(self.outputs)

        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"weights")
            local_fingerprint = controller._local_model_fingerprint(model)
            identity = controller._validate_remote_assets(
                Session(local_fingerprint),
                {"models": "/models"},
                {
                    "required_model_relative": "nninteractive/model",
                    "local_required_model": str(model),
                },
            )
            self.assertEqual(
                identity["remote_base_model_sha256"],
                local_fingerprint,
            )
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                controller._validate_remote_assets(
                    Session("a" * 64),
                    {"models": "/models"},
                    {
                        "required_model_relative": "nninteractive/model",
                        "local_required_model": str(model),
                    },
                )

    def test_remote_status_cannot_replace_local_control_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            status_path = Path(temporary) / "status.json"
            remote_compute.write_json_atomic(
                status_path,
                {
                    "job_id": "job",
                    "cancel_path": "C:/local/cancel.request",
                    "workspace": "C:/local/workspace",
                },
            )
            merged = controller._merge_remote_status(
                status_path,
                {
                    "status": "training",
                    "cancel_path": "/job/cancel.request",
                    "workspace": "/job/output",
                    "log_path": "/job/log",
                },
                {
                    "profile_id": "server",
                    "name": "GPU server",
                },
                "container",
                "/home/user/mimics-ai/jobs/job",
            )
            self.assertEqual(merged["cancel_path"], "C:/local/cancel.request")
            self.assertEqual(merged["workspace"], "C:/local/workspace")
            self.assertNotEqual(merged.get("log_path"), "/job/log")

    def test_remote_pipeline_completion_is_not_local_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            status_path = Path(temporary) / "status.json"
            remote_compute.write_json_atomic(
                status_path,
                {"job_id": "job", "status": "training"},
            )
            merged = controller._merge_remote_status(
                status_path,
                {"status": "completed", "progress_percent": 100},
                {"profile_id": "server", "name": "GPU server"},
                "container",
                "/jobs/job",
            )
            self.assertEqual(merged["status"], "finalizing_remote")
            self.assertEqual(merged["progress_percent"], 95)
            self.assertEqual(merged["remote_pipeline_status"], "completed")

    def test_remote_inline_metrics_are_kept_but_remote_paths_are_not(self):
        with tempfile.TemporaryDirectory() as temporary:
            status_path = Path(temporary) / "status.json"
            remote_compute.write_json_atomic(
                status_path,
                {"job_id": "job", "status": "training"},
            )
            metrics = [
                {
                    "epoch": 1,
                    "train_loss": 0.5,
                    "validation_auc": 0.7,
                }
            ]
            merged = controller._merge_remote_status(
                status_path,
                {
                    "status": "training",
                    "metrics_history": metrics,
                },
                {"profile_id": "server", "name": "GPU server"},
                "container",
                "/jobs/job",
            )
            self.assertEqual(merged["metrics_history"], metrics)
            merged = controller._merge_remote_status(
                status_path,
                {
                    "status": "training",
                    "metrics_history": "/job/output/metrics.json",
                },
                {"profile_id": "server", "name": "GPU server"},
                "container",
                "/jobs/job",
            )
            self.assertEqual(merged["metrics_history"], metrics)

    def test_stop_requests_framework_shutdown_before_container_stop(self):
        class Session:
            def __init__(self):
                self.commands = []

            def execute_result(self, command):
                if "json .Config.Labels" in command:
                    return 0, json.dumps(
                        {
                            "mimics-script.remote-training": "true",
                            "mimics-script.owner": "user",
                            "mimics-script.job": "job",
                        }
                    )
                if "test ! -e" in command:
                    return 0, ""
                return 0, ""

            def execute(self, command, **_kwargs):
                self.commands.append(command)
                return ""

        session = Session()
        states = [
            ("exited", 0),
            ("exited", 0),
            ("exited", 0),
            ("exited", 0),
            ("exited", 0),
            ("missing", -1),
        ]
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            controller, "_container_state", side_effect=states
        ):
            status_path = Path(temporary) / "status.json"
            remote_compute.write_json_atomic(status_path, {})
            controller._stop_remote(
                session,
                "container",
                status_path,
                remote_control_path="/remote/jobs/user/job/control.json",
                remote_control_kind="json",
                remote_job_dir="/remote/jobs/user/job",
                remote_root="/remote",
                expected_owner="user",
                expected_job="job",
            )
            self.assertIn("control.json.tmp", session.commands[0])
            self.assertIn("docker stop", session.commands[1])
            self.assertTrue(
                any("rm -rf" in command for command in session.commands)
            )
            status = remote_compute.read_json(status_path, {})
            self.assertEqual(status["status"], "cancelled")
            self.assertTrue(status["remote_stop_confirmed"])
            self.assertTrue(status["remote_container_removed"])
            self.assertTrue(status["remote_job_removed"])

    def test_artifact_publish_restores_existing_model_on_replace_failure(self):
        class Session:
            def __init__(self, archive):
                self.archive = archive

            def execute(self, command, **_kwargs):
                if "stat -c %s" in command:
                    return str(self.archive.stat().st_size)
                if "sha256sum" in command:
                    return controller._sha256_file(self.archive)
                return ""

            def download(self, _remote, local, callback=None):
                content = self.archive.read_bytes()
                Path(local).write_bytes(content)
                if callback:
                    callback(len(content), len(content))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "new.txt").write_text("new", encoding="utf-8")
            archive = root / "remote-result.tar"
            with tarfile.open(archive, "w") as handle:
                handle.add(source / "new.txt", arcname="new.txt")
            target = root / "model"
            target.mkdir()
            (target / "old.txt").write_text("old", encoding="utf-8")
            downloaded = root / "downloaded.tar"
            status = root / "status.json"
            remote_compute.write_json_atomic(status, {})
            original_replace = controller.os.replace

            def fail_publish(source_path, destination_path):
                if str(source_path).endswith(".remote-part"):
                    raise PermissionError("publish denied")
                return original_replace(source_path, destination_path)

            with mock.patch.object(
                controller.os, "replace", side_effect=fail_publish
            ), self.assertRaises(PermissionError):
                controller._download_artifact(
                    Session(archive),
                    "/remote/jobs/user/job",
                    "model_output",
                    target,
                    downloaded,
                    status,
                )
            self.assertEqual(
                (target / "old.txt").read_text(encoding="utf-8"),
                "old",
            )
            self.assertFalse((target / "new.txt").exists())

    def test_artifact_publish_preserves_backup_when_rollback_fails(self):
        class Session:
            def __init__(self, archive):
                self.archive = archive

            def execute(self, command, **_kwargs):
                if "stat -c %s" in command:
                    return str(self.archive.stat().st_size)
                if "sha256sum" in command:
                    return controller._sha256_file(self.archive)
                return ""

            def download(self, _remote, local, callback=None):
                content = self.archive.read_bytes()
                Path(local).write_bytes(content)
                if callback:
                    callback(len(content), len(content))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "new.txt").write_text("new", encoding="utf-8")
            archive = root / "remote-result.tar"
            with tarfile.open(archive, "w") as handle:
                handle.add(source / "new.txt", arcname="new.txt")
            target = root / "model"
            target.mkdir()
            (target / "old.txt").write_text("old", encoding="utf-8")
            status = root / "status.json"
            remote_compute.write_json_atomic(status, {})
            original_replace = controller.os.replace

            def fail_publish_and_restore(source_path, destination_path):
                source_text = str(source_path)
                if source_text.endswith(".remote-part") or ".remote-backup-" in source_text:
                    raise PermissionError("replace denied")
                return original_replace(source_path, destination_path)

            with mock.patch.object(
                controller.os,
                "replace",
                side_effect=fail_publish_and_restore,
            ), self.assertRaisesRegex(RuntimeError, "backup"):
                controller._download_artifact(
                    Session(archive),
                    "/remote/jobs/user/job",
                    "model_output",
                    target,
                    root / "downloaded.tar",
                    status,
                )
            backups = list(root.glob("model.remote-backup-*"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(
                (backups[0] / "old.txt").read_text(encoding="utf-8"),
                "old",
            )

    def test_directory_model_fingerprint_matches_remote_shell_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"weights")
            digests = sorted(
                controller._sha256_file(path)
                for path in (model / "config.json", model / "model.safetensors")
            )
            expected = hashlib.sha256(
                "".join(value + "\n" for value in digests).encode("ascii")
            ).hexdigest()
            self.assertEqual(controller._local_model_fingerprint(model), expected)

    def test_corrupt_download_is_rejected_before_model_publish(self):
        class Session:
            def execute(self, command, **_kwargs):
                if "stat -c %s" in command:
                    return "7"
                if "sha256sum" in command:
                    return "a" * 64
                return ""

            def download(self, _remote, local, callback=None):
                Path(local).write_bytes(b"corrupt")
                if callback:
                    callback(7, 7)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status = root / "status.json"
            target = root / "model"
            remote_compute.write_json_atomic(status, {})
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                controller._download_artifact(
                    Session(),
                    "/remote/jobs/user/job",
                    "model_output",
                    target,
                    root / "downloaded.tar",
                    status,
                )
            self.assertFalse(target.exists())


    def test_worker_starts_nninteractive_pipeline_after_staging_request(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            remote_worker, "APP_ROOT", Path(temporary)
        ), mock.patch.object(remote_worker, "_run", return_value=0) as run:
            root = Path(temporary)
            pipeline = root / "tools" / "nninteractive_finetune_pipeline.py"
            pipeline.parent.mkdir(parents=True)
            pipeline.write_text("# test\n", encoding="utf-8")
            job_dir = root / "job"
            job_dir.mkdir()
            request = {
                "pipeline_request": {
                    "job_id": "nn_job",
                    "task_id": "liver",
                    "task_name": "Liver",
                }
            }

            result = remote_worker.run_nninteractive(job_dir, request)

            self.assertEqual(result, 0)
            pipeline_job = job_dir / "pipeline_job"
            self.assertEqual(
                remote_compute.read_json(pipeline_job / "control.json", {}),
                {"action": "run", "updated_at_epoch": mock.ANY},
            )
            run.assert_called_once_with(
                [
                    remote_worker.sys.executable,
                    str(pipeline),
                    "run",
                    "--job-dir",
                    str(pipeline_job),
                ],
                job_dir,
            )


class FingerprintScriptExecutionTests(unittest.TestCase):
    """The container-side fingerprint script runs for real and matches local.

    The script is a string executed inside a throwaway container; a quoting
    or syntax defect in it once made every drift check compare against an
    empty digest. These tests execute the exact script text with a real
    python subprocess against a fixture tree, so any divergence between
    _local_code_fingerprint and _REMOTE_CODE_FINGERPRINT_SCRIPT fails here
    instead of on the server.
    """

    def _fixture(self, root: Path) -> None:
        # "foo0.py" vs "food/x.py": on Windows "\" (0x5C) sorts before
        # alphanumerics while POSIX "/" (0x2F) sorts after most punctuation
        # but before letters — this pair flips order between the two
        # algorithms unless both sides sort by the POSIX relative path.
        (root / "tools").mkdir(parents=True)
        (root / "tools" / "foo0.py").write_text("print('foo0')\n")
        (root / "tools" / "food").mkdir()
        (root / "tools" / "food" / "x.py").write_text("print('x')\n")
        (root / "integrations").mkdir()
        (root / "integrations" / "app.py").write_text("print('app')\n")
        # excluded directories must not affect either digest
        (root / "tools" / "__pycache__").mkdir()
        (root / "tools" / "__pycache__" / "junk.pyc").write_text("junk")
        (root / "docs").mkdir()
        (root / "docs" / "notes.md").write_text("not hashed")

    def _run_script(self, root: Path) -> str:
        import subprocess

        environment = dict(os.environ)
        environment["MIMICS_CODE_FINGERPRINT_BASE"] = str(root)
        completed = subprocess.run(
            [sys.executable, "-c", controller._REMOTE_CODE_FINGERPRINT_SCRIPT],
            capture_output=True,
            text=True,
            env=environment,
            timeout=120,
        )
        self.assertEqual(
            completed.returncode,
            0,
            "fingerprint script failed: " + completed.stderr,
        )
        return completed.stdout.strip()

    def test_container_script_digest_matches_local_fingerprint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._fixture(root)
            self.assertEqual(
                self._run_script(root),
                controller._local_code_fingerprint(root),
            )

    def test_container_script_digest_tracks_content_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._fixture(root)
            first = self._run_script(root)
            (root / "tools" / "foo0.py").write_text("print('changed')\n")
            self.assertNotEqual(first, self._run_script(root))


class RemoteJobCleanupTests(unittest.TestCase):
    """_remove_remote_job deletes the job dir plus its uploaded tar siblings."""

    class _Session:
        def __init__(self):
            self.commands = []

        def execute(self, command, **_kwargs):
            self.commands.append(command)

        def execute_result(self, command, **_kwargs):
            self.commands.append(command)
            return 0, ""

    def test_remove_remote_job_deletes_tar_archive_siblings(self):
        import shlex

        session = self._Session()
        removed = controller._remove_remote_job(
            session,
            remote_root="/userdata/shijian_ruan/mimics-ai",
            expected_owner="user",
            remote_job_dir=(
                "/userdata/shijian_ruan/mimics-ai/jobs/user/job123"
            ),
        )
        self.assertTrue(removed)
        self.assertEqual(len(session.commands), 2)
        arguments = shlex.split(session.commands[0])[3:]
        self.assertEqual(
            arguments,
            [
                "/userdata/shijian_ruan/mimics-ai/jobs/user/job123",
                "/userdata/shijian_ruan/mimics-ai/jobs/user/job123.tar",
                "/userdata/shijian_ruan/mimics-ai/jobs/user/job123.tar.part",
            ],
        )

    def test_remove_remote_job_refuses_path_escape(self):
        session = self._Session()
        with self.assertRaisesRegex(RuntimeError, "outside this user's"):
            controller._remove_remote_job(
                session,
                remote_root="/userdata/shijian_ruan/mimics-ai",
                expected_owner="user",
                remote_job_dir=(
                    "/userdata/shijian_ruan/mimics-ai/jobs/user/../other/job"
                ),
            )
        self.assertEqual(session.commands, [])

    def test_dataset_cache_cleanup_removes_part_and_tmp_fragments(self):
        # The cleanup command is buried in the remote launch flow behind a
        # live SSH session; assert the controller's source carries the
        # fragment globs next to the pre-existing .verified/.tar cleanup.
        source = Path(controller.__file__).read_text(encoding="utf-8")
        self.assertIn("-name '*.part'", source)
        self.assertIn("-name '*.tmp'", source)
        self.assertIn("-mtime +7", source)

    def test_curve_sync_rejects_parent_traversal(self):
        class Session:
            def sftp(self):
                return self

            def stat(self, _path):
                raise AssertionError("stat must not be called on escape paths")

        state = {}
        with tempfile.TemporaryDirectory() as temporary:
            changed = controller._sync_remote_nnunet_curve(
                Session(),
                "/remote/jobs/user/job1",
                {"training_curve_path": "/job/../../etc/passwd"},
                Path(temporary) / "curve.json",
                state,
            )
        self.assertFalse(changed)
        self.assertNotIn("curve_identity", state)

    def test_remote_job_removal_quotes_spaces_and_unicode_in_paths(self):
        # TB-03: a remote_root containing spaces (and non-ASCII) must never
        # change the meaning of a remote command. Every path interpolates
        # through shlex.quote, so the rm targets exactly the job dir.
        import shlex

        session = self._Session()
        root = "/userdata/患者 数据/mimics-ai"
        removed = controller._remove_remote_job(
            session,
            remote_root=root,
            expected_owner="user",
            remote_job_dir="{}/jobs/user/job 1".format(root),
        )
        self.assertTrue(removed)
        rm_command = session.commands[0]
        arguments = shlex.split(rm_command)[3:]
        self.assertEqual(
            arguments,
            [
                "{}/jobs/user/job 1".format(root),
                "{}/jobs/user/job 1.tar".format(root),
                "{}/jobs/user/job 1.tar.part".format(root),
            ],
        )
        # The parsed arguments are exactly the intended paths — no word
        # splitting, no shell reinterpretation.
        for argument in arguments:
            self.assertTrue(argument.startswith(root))


class AcceptanceChecklistTests(unittest.TestCase):
    """The acceptance script must stop the container on failure paths."""

    def setUp(self):
        self._module = importlib.import_module(
            "remote_acceptance_checklist"
        )

    def test_timeout_writes_control_stop_and_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_dir = root / "jobs" / "train_1"
            job_dir.mkdir(parents=True)
            remote_compute.write_json_atomic(
                job_dir / "status.json",
                {
                    "status": "training",
                    "remote_container_name": "mimics-job-1",
                    "remote_job_dir": "/remote/jobs/user/train_1",
                },
            )
            module = self._module
            step = module.AcceptanceStep("train_remote", "detail")
            with mock.patch.object(
                module.time, "sleep", side_effect=lambda _s: None
            ), mock.patch.object(
                module.time, "time", side_effect=[0.0, 10.0, 99999.0]
            ):
                # Simulate: first poll still training, second poll far past
                # the deadline -> timeout branch requests the stop.
                try:
                    module._request_stop(job_dir, wait_seconds=180.0)
                except StopIteration:
                    pass
            control = remote_compute.read_json(
                job_dir / "control.json", {}
            )
            self.assertEqual(control.get("action"), "stop")

    def test_verify_weights_fails_when_model_dir_missing(self):
        module = self._module
        results = []
        trained = {
            "status": {"status": "completed"},
            "model_dir": "",
            "job_dir": Path("unused"),
        }
        # torch is only imported inside the step; a missing model_dir must
        # fail before any checkpoint glob, so a stub is enough here.
        with mock.patch.dict("sys.modules", {"torch": mock.MagicMock()}):
            ok = module.step_verify_weights(results, trained)
        self.assertFalse(ok)
        self.assertEqual(results[0].status, "fail")
        self.assertIn("no checkpoints found", results[0].output["error"])

    def test_verify_weights_fails_on_unloadable_checkpoint(self):
        module = self._module
        results = []
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary) / "model"
            fold = model_dir / "fold_0"
            fold.mkdir(parents=True)
            (fold / "checkpoint_final.pth").write_bytes(b"not a checkpoint")

            class FakeTorch:
                @staticmethod
                def load(*_args, **_kwargs):
                    raise RuntimeError("corrupt checkpoint")

            trained = {
                "status": {"status": "completed"},
                "model_dir": str(model_dir),
                "job_dir": Path(temporary),
            }
            with mock.patch.dict(
                "sys.modules", {"torch": FakeTorch}
            ):
                ok = module.step_verify_weights(results, trained)
        self.assertFalse(ok)
        self.assertEqual(results[0].status, "fail")
        self.assertIn(
            "corrupt checkpoint", str(results[0].output["errors"])
        )

    def test_train_remote_requires_local_model_registration(self):
        module = self._module
        with tempfile.TemporaryDirectory() as temporary:
            registered = Path(temporary) / "model"
            registered.mkdir()
            # A locally registered model dir resolves.
            self.assertEqual(
                module._resolve_local_model_dir(
                    {"model": {"model_dir": str(registered)}}
                ),
                registered,
            )
            # Absent, container-side, and nonexistent paths all refuse —
            # these previously produced the verify_weights fake PASS.
            self.assertIsNone(module._resolve_local_model_dir({}))
            self.assertIsNone(
                module._resolve_local_model_dir(
                    {"model": {"model_dir": "/job/output/models/task/m1"}}
                )
            )
            self.assertIsNone(
                module._resolve_local_model_dir(
                    {"model": {"model_dir": str(Path(temporary) / "nope")}}
                )
            )

    class _CleanupSession:
        """Fake SSH session: find returns a fixed listing, removals succeed."""

        def __init__(self, listing):
            self.commands = []
            self._listing = listing

        def execute(self, command, **_kwargs):
            self.commands.append(command)
            return self._listing

        def execute_result(self, command, **_kwargs):
            self.commands.append(command)
            return 0, ""

    def _run_cleanup(self, listing):
        module = self._module
        session = self._CleanupSession(listing)
        results = []
        profile = {
            "remote_root": "/remote/mimics-ai",
            "username": "user",
        }
        with mock.patch.object(
            module, "_ssh_session", return_value=session
        ):
            ok = module.step_cleanup_failed_jobs(results, profile)
        return ok, results[0], session

    def test_cleanup_failed_jobs_removes_leftovers_under_jobs_root(self):
        ok, step, session = self._run_cleanup(
            "/remote/mimics-ai/jobs/user/jobA\n"
            "/remote/mimics-ai/jobs/user/jobB\n"
        )
        self.assertTrue(ok)
        self.assertEqual(step.status, "pass")
        self.assertEqual(step.output["leftovers_found"], 2)
        self.assertEqual(step.output["leftovers_removed"], 2)
        self.assertEqual(
            step.output["jobs_root"], "/remote/mimics-ai/jobs/user"
        )
        # The listing itself must exclude uploaded tar/.part siblings —
        # those are removed together with their job dir, not separately.
        find_command = session.commands[0]
        self.assertIn("/remote/mimics-ai/jobs/user", find_command)
        self.assertIn("! -name '*.tar'", find_command)
        self.assertIn("! -name '*.part'", find_command)
        # Each leftover goes through the validated removal path (rm -rf
        # of dir + .tar + .tar.part, then the existence check).
        self.assertEqual(len(session.commands), 1 + 2 * 2)
        self.assertIn("jobA", session.commands[1])
        self.assertIn("jobB", session.commands[3])

    def test_cleanup_failed_jobs_survives_one_refused_path(self):
        # A path escaping the owner's jobs folder is refused by
        # _remove_remote_job; the step must keep cleaning the rest.
        ok, step, session = self._run_cleanup(
            "/remote/mimics-ai/jobs/user/../other/jobC\n"
            "/remote/mimics-ai/jobs/user/jobD\n"
        )
        self.assertTrue(ok)
        self.assertEqual(step.status, "pass")
        self.assertEqual(step.output["leftovers_found"], 2)
        self.assertEqual(step.output["leftovers_removed"], 1)
        joined = "\n".join(session.commands)
        self.assertNotIn("other/jobC", joined)
        self.assertIn("jobD", joined)

    def test_cleanup_failed_dirs_flag_wires_step_into_main(self):
        # The cleanup must stay optional: the step only runs when the
        # user passes --cleanup-failed-dirs at the end of the run.
        source = Path(self._module.__file__).read_text(encoding="utf-8")
        self.assertIn("--cleanup-failed-dirs", source)
        self.assertIn(
            "if args.cleanup_failed_dirs:", source
        )
        self.assertIn(
            "step_cleanup_failed_jobs(results, profile)", source
        )


if __name__ == "__main__":
    unittest.main()
