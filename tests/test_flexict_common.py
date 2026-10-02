#!/usr/bin/env python3
"""Offline tests for flexict_common.py (no GPU, no Mimics, no nnU-Net)."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import flexict_common as fc


class TestConfig(unittest.TestCase):
    def test_load_config_defaults_are_complete(self):
        config = fc.load_config(Path(tempfile.gettempdir()) / "definitely_missing.json")
        self.assertEqual(config, fc.DEFAULT_CONFIG)
        # Internal mechanism keys (lock/poll/timeout/dataset-id band) come
        # from DEFAULT_CONFIG, not from the annotator-facing JSON file.
        self.assertEqual(config["gpu_lock_timeout_seconds"], 86400)
        self.assertEqual(config["dataset_id_first"], 750)

    def test_config_file_matches_defaults_shape(self):
        # The checked-in flexict_config.json must be loadable and every key
        # it carries must be a known key (unknown keys are a config drift).
        config = fc.load_config()
        self.assertTrue(set(config.keys()) <= set(fc.DEFAULT_CONFIG.keys()))


class TestWorkspacePaths(unittest.TestCase):
    def test_workspace_paths_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = fc.workspace_paths(tmp)
            self.assertEqual(paths["root"], Path(tmp).resolve())
            self.assertEqual(paths["registry"].name, "registry.json")
            self.assertEqual(paths["raw"].name, "nnUNet_raw")
            self.assertTrue("runtime" in str(paths["raw"]))

    def test_workspace_default_is_rooted_at_repo(self):
        root = fc.workspace_root(fc.load_config())
        self.assertTrue(str(root).endswith("flexict_models"))


class TestDatasetIdBand(unittest.TestCase):
    def test_suggest_starts_at_750(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(fc.flexict_suggest_dataset_id(tmp), 750)

    def test_suggest_skips_used_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = fc.workspace_paths(tmp)
            raw = paths["raw"]
            raw.mkdir(parents=True)
            (raw / "Dataset750_Kidney").mkdir()
            (raw / "Dataset751_Liver").mkdir()
            self.assertEqual(fc.flexict_suggest_dataset_id(tmp), 752)

    def test_suggest_respects_band_ceiling(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = fc.workspace_paths(tmp)
            raw = paths["raw"]
            raw.mkdir(parents=True)
            for i in range(750, 800):
                (raw / "Dataset{:03d}_X".format(i)).mkdir()
            with self.assertRaises(RuntimeError):
                fc.flexict_suggest_dataset_id(tmp)

    def test_band_isolated_from_nnunet_ids(self):
        # nnU-Net uses 701+; a Dataset701_x folder must not collide with the
        # FlexiCT band start (750 still suggested).
        with tempfile.TemporaryDirectory() as tmp:
            paths = fc.workspace_paths(tmp)
            raw = paths["raw"]
            raw.mkdir(parents=True)
            (raw / "Dataset701_NnunetTask").mkdir()
            self.assertEqual(fc.flexict_suggest_dataset_id(tmp), 750)


class TestRegistry(unittest.TestCase):
    def _make_model(self, tmp, model_id="m1", pair_id="", configuration="2d",
                    with_checkpoint=True):
        paths = fc.workspace_paths(tmp)
        model_dir = paths["results"] / model_id
        (model_dir / "fold_0").mkdir(parents=True, exist_ok=True)
        (model_dir / "plans.json").write_text("{}", encoding="utf-8")
        (model_dir / "dataset.json").write_text("{}", encoding="utf-8")
        if with_checkpoint:
            (model_dir / "fold_0" / "checkpoint_best.pth").write_bytes(b"x")
        return {
            "model_id": model_id,
            "model_dir": str(model_dir),
            "configuration": configuration,
            "pair_id": pair_id,
            "dataset_id": 750,
            "label_name": "kidney_left",
        }

    def test_register_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self._make_model(tmp, "m1")
            fc.register_model(manifest, workspace=tmp)
            models = fc.load_models(tmp)
            self.assertEqual(len(models), 1)
            self.assertEqual(models[0]["model_id"], "m1")
            self.assertEqual(models[0]["schema_version"], fc.MODEL_SCHEMA_VERSION)

    def test_register_replaces_same_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            fc.register_model(self._make_model(tmp, "m1"), workspace=tmp)
            fc.register_model(self._make_model(tmp, "m1"), workspace=tmp)
            self.assertEqual(len(fc.load_models(tmp)), 1)

    def test_model_usability(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self._make_model(tmp, "m1")
            usable, why = fc.model_usability(manifest)
            self.assertTrue(usable)
            broken = self._make_model(tmp, "m2", with_checkpoint=False)
            usable, why = fc.model_usability(broken)
            self.assertFalse(usable)
            self.assertIn("checkpoint_best", why)

    def test_recommended_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(fc.recommended_model(workspace=tmp))
            fc.register_model(self._make_model(tmp, "m1"), workspace=tmp)
            rec = fc.recommended_model(workspace=tmp)
            self.assertIsNotNone(rec)
            self.assertEqual(rec["model_id"], "m1")

    def test_load_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(fc.load_pair(workspace=tmp), (None, None))
            fc.register_model(
                self._make_model(tmp, "m2d", pair_id="p1", configuration="2d"),
                workspace=tmp)
            self.assertEqual(fc.load_pair(workspace=tmp), (None, None))
            fc.register_model(
                self._make_model(tmp, "m3d", pair_id="p1", configuration="3d_fullres"),
                workspace=tmp)
            m2d, m3d = fc.load_pair(workspace=tmp)
            self.assertIsNotNone(m2d)
            self.assertIsNotNone(m3d)
            self.assertEqual(m2d["configuration"], "2d")
            self.assertEqual(m3d["configuration"], "3d_fullres")

    def test_load_models_flags_missing_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            fc.register_model(self._make_model(tmp, "m1"), workspace=tmp)
            # Break the recorded path.
            import shutil
            shutil.rmtree(fc.workspace_paths(tmp)["results"] / "m1")
            self.assertEqual(fc.load_models(tmp), [])
            flagged = fc.load_models(tmp, include_missing=True)
            self.assertEqual(len(flagged), 1)
            self.assertEqual(flagged[0]["status"], "missing_path")


class TestNormalizeRequest(unittest.TestCase):
    def test_minimal_request_gets_defaults(self):
        req = fc.normalize_request(
            {"label_name": "liver", "cases": ["c1", "c2", "c3", "c4"]})
        self.assertEqual(req["configuration"], "auto")
        self.assertEqual(req["epochs"], 150)
        self.assertEqual(req["val_cases"], 3)
        self.assertEqual(req["mirror_disable_axes"], "")

    def test_rejects_empty_label(self):
        with self.assertRaises(ValueError):
            fc.normalize_request({"label_name": "", "cases": ["c1"]})

    def test_rejects_no_cases(self):
        with self.assertRaises(ValueError):
            fc.normalize_request({"label_name": "liver", "cases": []})

    def test_rejects_bad_configuration(self):
        with self.assertRaises(ValueError):
            fc.normalize_request(
                {"label_name": "liver", "cases": ["c1"], "configuration": "4d"})

    def test_val_cases_clamped_to_case_count(self):
        req = fc.normalize_request(
            {"label_name": "liver", "cases": ["c1", "c2"], "val_cases": 5})
        self.assertEqual(req["val_cases"], 1)

    def test_single_case_rejected(self):
        with self.assertRaises(ValueError):
            fc.normalize_request({"label_name": "liver", "cases": ["c1"]})

    def test_pair_configuration_accepted(self):
        for value in ("2d", "3d_fullres", "pair", "auto"):
            req = fc.normalize_request(
                {"label_name": "liver", "cases": ["c1", "c2"], "configuration": value})
            self.assertEqual(req["configuration"], value)


class TestPretrainedWeights(unittest.TestCase):
    def test_resolve_pretrained_dir_finds_repo_weights(self):
        # Resolve against a synthetic repo layout so the test does not depend
        # on the ~1.1 GB gitignored weights being present on this machine.
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "flexict-repo"
            for sub in ("flexict_2d", "flexict_3d"):
                (repo / "weights" / sub).mkdir(parents=True)
                (repo / "weights" / sub / "model.safetensors").write_bytes(b"x")
            config = dict(fc.DEFAULT_CONFIG)
            config["flexict_dir"] = str(repo)
            config["pretrained_weights_dir"] = ""
            weights, source = fc.resolve_pretrained_dir(config)
            self.assertEqual(source, "repo")
            self.assertTrue(
                (weights / "flexict_2d" / "model.safetensors").is_file())
            self.assertTrue(
                (weights / "flexict_3d" / "model.safetensors").is_file())

    def test_resolve_pretrained_dir_rejects_bad_override(self):
        config = fc.load_config()
        config["pretrained_weights_dir"] = str(Path(tempfile.gettempdir()))
        with self.assertRaises(FileNotFoundError):
            fc.resolve_pretrained_dir(config)


if __name__ == "__main__":
    unittest.main(verbosity=2)
