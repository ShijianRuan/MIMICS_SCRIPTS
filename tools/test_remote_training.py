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

import tools.fewshot_training_setup_ui as dino_ui
import tools.fewshot_status_viewer as status_viewer
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

    def test_default_launch_uses_unchanged_local_command(self):
        with tempfile.TemporaryDirectory() as temporary:
            status_path = str(Path(temporary) / "job.json")
            launch = {
                "cmd": ["python", "fewshot_pipeline.py", "train"],
                "run_id": "local_job",
                "status_path": status_path,
                "cancel_path": str(Path(temporary) / "cancel.request"),
                "job_payload": {
                    "job_id": "local_job",
                    "status": "launching",
                },
                "options": {},
            }
            with mock.patch.object(
                dino_ui, "prepare_training_launch", return_value=launch
            ), mock.patch.object(
                dino_ui, "launch_remote_training"
            ) as remote_launch, mock.patch.object(
                dino_ui.subprocess, "Popen", return_value=_Process()
            ) as popen:
                result = dino_ui.launch_training(
                    {"project_root": temporary},
                    {},
                )
            remote_launch.assert_not_called()
            popen.assert_called_once()
            self.assertEqual(
                popen.call_args.args[0],
                ["python", "fewshot_pipeline.py", "train"],
            )
            self.assertEqual(result, ("local_job", status_path, 4321))

    def test_remote_launch_is_opt_in(self):
        expected = ("remote_job", "status.json", 44)
        with mock.patch.object(
            dino_ui, "launch_remote_training", return_value=expected
        ) as remote_launch, mock.patch.object(
            dino_ui, "prepare_training_launch"
        ) as local_prepare:
            result = dino_ui.launch_training(
                {},
                {
                    "execution_backend": "remote",
                    "remote_profile_id": "server",
                },
            )
        self.assertEqual(result, expected)
        remote_launch.assert_called_once()
        local_prepare.assert_not_called()

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

    def test_remote_dino_reuses_local_source_grid_cache(self):
        import nibabel as nib
        import numpy as np
        import tools.fewshot_pipeline as pipeline

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            image = root / "image.nii.gz"
            label = root / "label.nii.gz"
            model = root / "model.onnx"
            model.write_bytes(b"model")
            values = np.arange(64, dtype=np.float32).reshape((4, 4, 4)) + 1
            target = np.zeros((4, 4, 4), dtype=np.uint8)
            target[1:3, 1:3, 1:3] = 1
            nib.save(nib.Nifti1Image(values, np.eye(4)), str(image))
            nib.save(nib.Nifti1Image(target, np.eye(4)), str(label))
            sample = {
                "case_id": "case_1",
                "image": str(image),
                "label": str(label),
            }
            spec = {
                "context": {
                    "workspace": str(workspace),
                    "ts_root": str(root),
                    "dinov3_root": str(root),
                    "organ": "liver",
                    "config": {},
                },
                "options": {
                    "label_source": "exported_masks",
                    "label_root": str(root),
                    "min_samples": 1,
                    "val_fraction": 0.0,
                    "model_path": str(model),
                },
                "status_path": str(root / "status.json"),
                "cancel_path": str(root / "cancel.request"),
                "run_id": "remote_job",
            }
            with mock.patch.object(
                pipeline,
                "resolve_training_mask_names",
                return_value=["liver"],
            ), mock.patch.object(
                pipeline,
                "discover_samples",
                return_value=([sample], []),
            ), mock.patch.object(
                pipeline,
                "select_samples",
                side_effect=lambda rows, *_args: rows,
            ), mock.patch.object(
                pipeline,
                "split_train_validation",
                return_value=([sample], []),
            ), mock.patch.object(
                dino_ui,
                "append_training_args",
                return_value=None,
            ):
                controller._prepare_dino(spec, root / "bundle_1")
                controller._prepare_dino(spec, root / "bundle_2")
            status = remote_compute.read_json(root / "status.json", {})
            self.assertEqual(status["local_materialization_cache_reused"], 1)
            self.assertEqual(status["local_materialization_cache_total"], 1)
            self.assertTrue(
                (root / "bundle_2" / "input" / "case_1" / "ct.nii.gz").is_file()
            )

    def test_dino_remote_launch_does_not_overwrite_fast_controller_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            status_path = Path(temporary) / "status.json"
            cancel_path = Path(temporary) / "cancel.request"
            launch = {
                "cmd": ["python", "fewshot_pipeline.py", "train"],
                "run_id": "remote_job",
                "status_path": str(status_path),
                "cancel_path": str(cancel_path),
                "job_payload": {
                    "job_id": "remote_job",
                    "status": "launching",
                },
                "options": {},
            }

            def controller_started(*_args, **_kwargs):
                current = remote_compute.read_json(status_path, {}) or {}
                current.update(
                    {
                        "status": "uploading",
                        "phase": "uploading_training_data",
                        "progress_percent": 27,
                    }
                )
                remote_compute.write_json_atomic(status_path, current)
                return _Process()

            with mock.patch.object(
                dino_ui, "prepare_training_launch", return_value=launch
            ), mock.patch.object(
                dino_ui.subprocess, "Popen", side_effect=controller_started
            ):
                dino_ui.launch_remote_training(
                    {
                        "project_root": temporary,
                        "python_exe": sys.executable,
                    },
                    {
                        "execution_backend": "remote",
                        "remote_profile_id": "server",
                    },
                )
            status = remote_compute.read_json(status_path, {})
            self.assertEqual(status["status"], "uploading")
            self.assertEqual(status["progress_percent"], 27)
            self.assertEqual(status["controller_pid"], 4321)

    def test_remote_dino_label_refresh_uses_current_run_before_cache(self):
        import tools.fewshot_pipeline as pipeline

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ts_root = root / "dataset"
            workspace = root / "workspace"
            dinov3_root = root / "dinov3"
            bundle = root / "bundle"
            status_path = root / "status.json"
            cancel_path = root / "cancel.request"
            source_image = root / "source.nii.gz"
            source_label = root / "source_label.nii.gz"
            source_image.write_bytes(b"image")
            source_label.write_bytes(b"label")
            remote_compute.write_json_atomic(status_path, {})
            plan = {
                "cache_root": root / "cache",
                "reusable": {},
                "changed": ["case_1"],
                "fingerprints": {"case_1": "fingerprint"},
                "requested": ["case_1"],
            }
            publish_calls = []

            def launch_export(*_args, **kwargs):
                label = (
                    Path(kwargs["label_staging_dir"])
                    / "case_1"
                    / "segmentations"
                    / "liver.nii.gz"
                )
                label.parent.mkdir(parents=True, exist_ok=True)
                label.write_bytes(b"fresh-label")
                return {
                    "launched": True,
                    "timed_out": False,
                    "returncode": 0,
                    "batch_status": {"status": "completed"},
                }

            def publish_cache(
                received_plan,
                changed_root,
                mask_names,
                output_root=None,
            ):
                publish_calls.append(
                    (received_plan, Path(changed_root), mask_names, output_root)
                )
                destination = (
                    Path(output_root)
                    / "case_1"
                    / "segmentations"
                    / "liver.nii.gz"
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"fresh-label")
                return {"case_1"}, [
                    {
                        "case_id": "case_1",
                        "stage": "cache_publish",
                        "error": "simulated cache failure",
                    }
                ]

            samples = [
                {
                    "case_id": "case_1",
                    "image": str(source_image),
                    "label": str(source_label),
                }
            ]

            def materialize_image(_source, destination):
                destination = Path(destination)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"image")

            def materialize_label(_source, _image, destination):
                destination = Path(destination)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"label")
                return "test", True, 1, 1

            spec = {
                "context": {
                    "workspace": str(workspace),
                    "ts_root": str(ts_root),
                    "dinov3_root": str(dinov3_root),
                    "organ": "liver",
                    "config": {},
                },
                "options": {
                    "label_source": "mcs_refresh",
                    "min_samples": 1,
                    "val_fraction": 0.0,
                },
                "status_path": str(status_path),
                "cancel_path": str(cancel_path),
                "run_id": "remote_job",
            }

            with mock.patch.object(
                pipeline,
                "resolve_training_mask_names",
                return_value=["liver"],
            ), mock.patch.object(
                pipeline,
                "_plan_dino_mcs_label_cache",
                return_value=plan,
            ), mock.patch.object(
                pipeline,
                "launch_mimics_export",
                side_effect=launch_export,
            ), mock.patch.object(
                pipeline,
                "_publish_dino_mcs_label_cache",
                side_effect=publish_cache,
            ), mock.patch.object(
                pipeline,
                "discover_samples",
                return_value=(samples, []),
            ), mock.patch.object(
                pipeline,
                "select_samples",
                side_effect=lambda rows, *_args: rows,
            ), mock.patch.object(
                pipeline,
                "split_train_validation",
                return_value=(samples, []),
            ), mock.patch.object(
                pipeline,
                "_materialize_source_image",
                side_effect=materialize_image,
            ), mock.patch.object(
                pipeline,
                "_materialize_label_on_source_grid",
                side_effect=materialize_label,
            ), mock.patch.object(
                dino_ui,
                "append_training_args",
                return_value=None,
            ):
                prepared = controller._prepare_dino(spec, bundle)

            self.assertEqual(prepared["train_count"], 1)
            self.assertEqual(len(publish_calls), 1)
            self.assertEqual(Path(publish_calls[0][3]).name, "remote_fresh_labels")
            status = remote_compute.read_json(status_path, {})
            self.assertEqual(
                status["label_cache_warnings"][0]["stage"],
                "cache_publish",
            )
            self.assertTrue(
                (bundle / "labels" / "case_1" / "segmentations" / "liver.nii.gz").is_file()
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
                        "/remote-cache/dinov3/liver/"
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
            self.assertTrue(
                status_viewer.can_abandon_remote_job(
                    remote_compute.read_json(status_path, {})
                )
            )
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
        class ImmediateThread:
            def __init__(self, target, **_kwargs):
                self.target = target
                self.daemon = False

            def start(self):
                self.target()

        with tempfile.TemporaryDirectory() as temporary:
            status_path = Path(temporary) / "status.json"
            cancel_path = Path(temporary) / "cancel.request"
            remote_compute.write_json_atomic(
                status_path,
                {
                    "status": "training",
                    "execution_backend": "remote",
                    "controller_pid": 1234,
                    "remote_stop_confirmed": False,
                },
            )
            job = remote_compute.read_json(status_path, {})
            job["cancel_path"] = str(cancel_path)
            with mock.patch.object(
                status_viewer.subprocess, "Popen", return_value=_Process()
            ), mock.patch.object(
                status_viewer, "process_exists", return_value=False
            ), mock.patch.object(
                status_viewer, "terminate_process_tree"
            ) as terminate, mock.patch.object(
                status_viewer.threading, "Thread", ImmediateThread
            ):
                status_viewer.request_job_cancel_async(
                    job, status_path, grace_seconds=0
                )
            terminate.assert_not_called()
            latest = remote_compute.read_json(status_path, {})
            self.assertEqual(latest["status"], "stopping")
            self.assertFalse(latest["remote_stop_confirmed"])

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
                    "required_model_relative": "dinov3/model",
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
                        "required_model_relative": "dinov3/model",
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

    def test_worker_replaces_only_remote_python_placeholder(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            remote_worker, "APP_ROOT", Path(temporary)
        ), mock.patch.object(remote_worker, "_run", return_value=0) as run:
            pipeline = Path(temporary) / "tools" / "fewshot_pipeline.py"
            pipeline.parent.mkdir(parents=True)
            pipeline.write_text("# test\n", encoding="utf-8")
            remote_worker.run_dino(
                Path(temporary),
                {
                    "pipeline_args": [
                        "train",
                        "--python",
                        "__REMOTE_PYTHON__",
                        "--organ",
                        "liver",
                    ]
                },
            )
            command = run.call_args.args[0]
            self.assertEqual(command[2], "train")
            self.assertEqual(command[4], remote_worker.sys.executable)
            self.assertEqual(command[-1], "liver")


if __name__ == "__main__":
    unittest.main()
