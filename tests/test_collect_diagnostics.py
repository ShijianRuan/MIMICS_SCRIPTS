#!/usr/bin/env python3
"""Regression tests for the redacted diagnostics bundle collector."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import collect_diagnostics


class TestRedactText(unittest.TestCase):
    def test_absolute_paths_keep_last_two_components(self):
        text = (
            "Imported E:\\data\\patients\\case_042\\image.nii.gz from "
            "D:/datasets/TotalSegmentator/s0623/ct.nii.gz"
        )
        redacted = collect_diagnostics.redact_text(text)
        self.assertNotIn("patients", redacted)
        self.assertNotIn("TotalSegmentator", redacted)
        self.assertIn("case_042/image.nii.gz", redacted)
        self.assertIn("s0623/ct.nii.gz", redacted)

    def test_short_paths_left_alone(self):
        text = "relative\\path\\file.py and C:\\f.py"
        redacted = collect_diagnostics.redact_text(text)
        self.assertIn("relative\\path\\file.py", redacted)

    def test_bearer_tokens_masked(self):
        text = 'Authorization: Bearer abc123def456 XYZ'
        redacted = collect_diagnostics.redact_text(text)
        self.assertNotIn("abc123def456", redacted)
        self.assertIn("Bearer ***", redacted)

    def test_api_key_values_masked(self):
        text = '"api_key": "secretpassword123"'
        redacted = collect_diagnostics.redact_text(text)
        self.assertNotIn("secretpassword123", redacted)

    def test_case_ids_preserved(self):
        # Model diagnosis needs case IDs; the surrounding paths go, IDs stay.
        text = "case qin_t1_02 AUC 0.86 at E:\\deep\\path\\qin_t1_02\\seg.nii"
        redacted = collect_diagnostics.redact_text(text)
        self.assertIn("qin_t1_02", redacted)


class TestCollectBundle(unittest.TestCase):
    def _prepare_root(self, base: Path):
        # The "~"-relative JOB_ROOTS / registry lookups must resolve against
        # the temp base, never the real user profile.
        fake_home = base / "home"
        fake_home.mkdir()
        self._home_patch = mock.patch.dict(
            "os.environ",
            {"USERPROFILE": str(fake_home), "HOME": str(fake_home)},
        )
        self._home_patch.start()
        self.addCleanup(self._home_patch.stop)
        runtime = base / ".mimics_runtime"
        runtime.mkdir(parents=True)
        (runtime / "mimics_export.log").write_text(
            "\n".join(
                "line {0} E:\\secret\\location\\case{0}\\file.nii.gz".format(i)
                for i in range(300)
            ),
            encoding="utf-8",
        )
        (runtime / "setup_env.log").write_text(
            "setup ok Bearer topsecrettoken", encoding="utf-8"
        )
        (runtime / "health_panel").mkdir()
        (runtime / "health_panel" / "health_panel_20260101T000000.log").write_text(
            "panel error at C:\\Users\\doctor\\private\\path.py",
            encoding="utf-8",
        )
        # A failed export job with status + logs.
        job_dir = runtime / "export_jobs" / "job_failed"
        job_dir.mkdir(parents=True)
        (job_dir / "status.json").write_text(
            json.dumps({"status": "failed", "phase": "exporting_labels"}),
            encoding="utf-8",
        )
        (job_dir / "process.log").write_text(
            "boom at E:\\private\\deep\\spot\\x.nii", encoding="utf-8"
        )
        # An active job (stuck-task scenario).
        active_dir = runtime / "export_jobs" / "job_active"
        active_dir.mkdir(parents=True)
        (active_dir / "status.json").write_text(
            json.dumps({"status": "running", "phase": "training"}),
            encoding="utf-8",
        )
        (base / "mimics_io_config.json").write_text(
            json.dumps({"mcs_output_dir": "E:/exports"}),
            encoding="utf-8",
        )
        return runtime

    def test_bundle_contains_redacted_tails_and_configs(self):
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            self._prepare_root(base)
            out = base / "bundle.zip"
            with mock.patch.object(collect_diagnostics, "ROOT", base):
                collect_diagnostics.collect_bundle(out)
            self.assertTrue(out.is_file())
            with zipfile.ZipFile(str(out)) as archive:
                names = set(archive.namelist())
                self.assertIn("logs/mimics_export_tail.log", names)
                self.assertIn("logs/setup_env_tail.log", names)
                self.assertIn("configs/mimics_io_config.json", names)
                self.assertIn("diagnostics/runtime_census.json", names)
                self.assertIn("diagnostics/environment.json", names)
                export_tail = archive.read("logs/mimics_export_tail.log").decode("utf-8")
                # Only the tail (200 lines) is kept.
                self.assertEqual(len(export_tail.strip().splitlines()), 200)
                self.assertNotIn("secret", export_tail)
                setup_tail = archive.read("logs/setup_env_tail.log").decode("utf-8")
                self.assertNotIn("topsecrettoken", setup_tail)

    def test_bundle_contains_active_and_failed_job_snapshots(self):
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            self._prepare_root(base)
            out = base / "bundle.zip"
            with mock.patch.object(collect_diagnostics, "ROOT", base):
                collect_diagnostics.collect_bundle(out)
            with zipfile.ZipFile(str(out)) as archive:
                names = set(archive.namelist())
                self.assertIn("jobs/export_jobs/job_failed/status.json", names)
                self.assertIn("jobs/export_jobs/job_failed/process.log", names)
                self.assertIn("jobs/export_jobs/job_active/status.json", names)
                log = archive.read("jobs/export_jobs/job_failed/process.log").decode("utf-8")
                self.assertNotIn("private", log)

    def test_focus_latest_failure_narrows_to_failed_job(self):
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            self._prepare_root(base)
            out = base / "bundle.zip"
            with mock.patch.object(collect_diagnostics, "ROOT", base):
                collect_diagnostics.collect_bundle(out, focus_latest_failure=True)
            with zipfile.ZipFile(str(out)) as archive:
                names = set(archive.namelist())
                self.assertIn("jobs/export_jobs/job_failed/status.json", names)
                self.assertNotIn("jobs/export_jobs/job_active/status.json", names)

    def test_health_panel_newest_logs_included(self):
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            self._prepare_root(base)
            out = base / "bundle.zip"
            with mock.patch.object(collect_diagnostics, "ROOT", base):
                collect_diagnostics.collect_bundle(out)
            with zipfile.ZipFile(str(out)) as archive:
                panel = [
                    n for n in archive.namelist()
                    if n.startswith("logs/health_panel")
                ]
                self.assertTrue(panel)
                content = archive.read(panel[0]).decode("utf-8")
                self.assertNotIn("doctor", content)

    def test_import_queue_discovery_includes_output_import_log(self):
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            runtime = self._prepare_root(base)
            output_dir = base / "mcs_out"
            (output_dir / "logs").mkdir(parents=True)
            (output_dir / "logs" / "mimics_import.log").write_text(
                "imported 3 masks from E:\\deep\\private\\dir\\case01.nii",
                encoding="utf-8",
            )
            control = runtime / "import_queues" / "out_abc123def456abc"
            control.mkdir(parents=True)
            (control / "_run_create_mcs.py").write_text(
                'create_mcs_batch.main("{}")'.format(
                    str(output_dir).replace("\\", "\\\\")
                ),
                encoding="utf-8",
            )
            (control / "_background_mimics.log").write_text(
                "bg error E:\\private\\deep\\bg.log", encoding="utf-8"
            )
            out = base / "bundle.zip"
            with mock.patch.object(collect_diagnostics, "ROOT", base):
                collect_diagnostics.collect_bundle(out)
            with zipfile.ZipFile(str(out)) as archive:
                names = set(archive.namelist())
                self.assertIn("logs/mimics_import_0_tail.log", names)
                self.assertIn(
                    "logs/import_queues/out_abc123def456abc/bg_tail.log", names
                )
                tail = archive.read("logs/mimics_import_0_tail.log").decode("utf-8")
                self.assertNotIn("private", tail)

    def test_oversized_bundle_refused(self):
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            self._prepare_root(base)
            # A tiny cap makes the size check deterministic: real incompressible
            # blobs are not portable across zlib versions, so patch the cap
            # instead of fighting compression ratios.
            out = base / "bundle.zip"
            with mock.patch.object(collect_diagnostics, "ROOT", base):
                with mock.patch.object(collect_diagnostics, "BUNDLE_MAX_BYTES", 1):
                    with self.assertRaises(RuntimeError):
                        collect_diagnostics.collect_bundle(out)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
