#!/usr/bin/env python3
"""Tests for persistent external-UI folder preferences."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# runnable both as `python tools/test_ui_preferences.py` and from a test
# runner that already has the project root on sys.path.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(_ROOT))

from tools import ui_preferences


class UIPreferencesTests(unittest.TestCase):
    def test_sections_are_isolated_and_partial_updates_preserve_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.dict(
                os.environ, {"MIMICS_USER_CONFIG_DIR": temporary}, clear=False
            ):
                ui_preferences.save_preferences(
                    "training", {"dataset_root": "D:/dataset", "mcs_dir": "D:/mcs"}
                )
                ui_preferences.save_preferences(
                    "training", {"dataset_root": "E:/new-dataset", "mcs_dir": ""}
                )
                ui_preferences.save_preferences("import", {"source": "F:/cases"})

                training = ui_preferences.load_preferences("training")
                self.assertEqual(training["dataset_root"], "E:/new-dataset")
                self.assertEqual(training["mcs_dir"], "D:/mcs")
                self.assertEqual(
                    ui_preferences.load_preferences("import")["source"], "F:/cases"
                )
                self.assertEqual(
                    ui_preferences.preferences_path(),
                    Path(temporary) / "ui_preferences.json",
                )


if __name__ == "__main__":
    unittest.main()
