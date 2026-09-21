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
        runtime = base / ".mimics_runtime"
        runtime.mkdir(parents=True)
        (runtime / "import.log").write_text(
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
        (base / "fewshot_config.json").write_text(
            json.dumps({"dinov3_project": "external/dinov3-medical-seg"}),
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
                self.assertIn("logs/import_tail.log", names)
                self.assertIn("logs/setup_env_tail.log", names)
                self.assertIn("configs/fewshot_config.json", names)
                self.assertIn("diagnostics/runtime_census.json", names)
                self.assertIn("diagnostics/environment.json", names)
                import_tail = archive.read("logs/import_tail.log").decode("utf-8")
                # Only the tail (200 lines) is kept.
                self.assertEqual(len(import_tail.strip().splitlines()), 200)
                self.assertNotIn("secret", import_tail)
                setup_tail = archive.read("logs/setup_env_tail.log").decode("utf-8")
                self.assertNotIn("topsecrettoken", setup_tail)

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

    def test_oversized_bundle_refused(self):
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            runtime = self._prepare_root(base)
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
