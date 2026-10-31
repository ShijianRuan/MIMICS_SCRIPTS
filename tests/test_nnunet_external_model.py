"""External nnU-Net model import: read-only scan, strict ZIP, atomic publish.

Failure paths first: malicious checkpoints must be rejected without executing
payloads, ZIP bombs and traversal entries must be refused, blocked models
must never leave partial registrations, and the live prediction registry
must stay untouched until the manager UI round wires the recipe.
"""
import json
import stat
import sys
import tempfile
import unittest
import unittest.mock
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT)]

import torch

import nnunet_external_model as external


def build_model(
    root,
    name="Dataset123_Example",
    trainer="nnUNetTrainer",
    plans="nnUNetPlans",
    configuration="3d_fullres",
    channels=("CT",),
    labels=None,
    folds=("fold_0",),
    checkpoint="checkpoint_final.pth",
    dataset_extra=None,
):
    """Materialize a minimal nnU-Net v2 training-result directory."""
    config_dir = root / name / "__".join((trainer, plans, configuration))
    config_dir.mkdir(parents=True)
    dataset = {
        "name": name,
        "channel_names": {str(index): channel for index, channel in enumerate(channels)},
        "labels": labels if labels is not None else {"tumor": 1, "kidney": 2},
        "numTraining": 0,
    }
    if dataset_extra:
        dataset.update(dataset_extra)
    (config_dir / "dataset.json").write_text(
        json.dumps(dataset), encoding="utf-8"
    )
    (config_dir / "plans.json").write_text(
        json.dumps({"configurations": {configuration: {}}}), encoding="utf-8"
    )
    for fold in folds:
        fold_dir = config_dir / fold
        fold_dir.mkdir()
        torch.save(
            {
                "trainer_name": trainer,
                "network_weights": {},
                "current_epoch": 100,
                "inference_allowed_mirroring_axes": [0, 1],
            },
            str(fold_dir / checkpoint),
        )
    return config_dir


def build_zip(source_dir, bundle_path):
    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                archive.write(str(path), path.relative_to(source_dir).as_posix())
    return bundle_path


class ExternalModelScan(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_scan_complete_single_channel_model_is_importable(self):
        build_model(self.root)
        report = external.scan_external_model(self.root / "Dataset123_Example")
        self.assertEqual([], report["candidates"][0]["blockers"])
        row = report["candidates"][0]
        self.assertEqual("nnUNetTrainer", row["trainer"])
        self.assertEqual(["fold_0"], row["complete_folds"])
        self.assertEqual({"0": "CT"}, row["channel_names"])
        self.assertEqual([0, 1], row["mirroring_axes"])
        self.assertFalse(report["candidates"][0]["requires_fold_selection"])
        self.assertEqual(["3d_fullres"], report["importable"])

    def test_scan_missing_plans_reports_missing_files(self):
        config_dir = build_model(self.root)
        (config_dir / "plans.json").unlink()
        report = external.scan_external_model(self.root / "Dataset123_Example")
        # Partial structure degrades to a near-miss row that names exactly
        # what is missing, instead of a candidate with guessed defaults.
        self.assertEqual([], report["candidates"])
        rejected = report["rejected_candidates"][0]
        self.assertIn("plans.json", rejected["missing"])
        self.assertEqual([], report["importable"])

    def test_scan_lone_checkpoint_reports_missing_metadata(self):
        # A checkpoint without dataset.json/plans.json must be rejected with
        # the exact missing-file list, never a guessed default network.
        fold_dir = self.root / "Dataset123_Example" / "only_checkpoint" / "fold_0"
        fold_dir.mkdir(parents=True)
        torch.save({"trainer_name": "nnUNetTrainer"}, str(fold_dir / "checkpoint_final.pth"))
        report = external.scan_external_model(self.root / "Dataset123_Example")
        self.assertEqual([], report["candidates"])
        self.assertEqual([], report["importable"])
        rejected = report["rejected_candidates"][0]
        self.assertIn("dataset.json", rejected["missing"])
        self.assertIn("plans.json", rejected["missing"])
        self.assertIn("v1", report["hint"])

    def test_scan_safetensors_only_fold_is_blocked(self):
        config_dir = build_model(self.root, folds=())
        fold_dir = config_dir / "fold_0"
        fold_dir.mkdir()
        (fold_dir / "checkpoint_final.safetensors").write_bytes(b"fixture")
        report = external.scan_external_model(self.root / "Dataset123_Example")
        blockers = "；".join(report["candidates"][0]["blockers"])
        self.assertIn("safetensors", blockers)
        self.assertEqual([], report["importable"])

    def test_scan_multichannel_model_is_blocked(self):
        build_model(self.root, channels=("CT", "MR"))
        report = external.scan_external_model(self.root / "Dataset123_Example")
        blockers = "；".join(report["candidates"][0]["blockers"])
        self.assertIn("2 个输入通道", blockers)
        self.assertEqual([], report["importable"])

    def test_scan_legacy_modality_key_is_read_as_channels(self):
        config_dir = build_model(self.root, channels=("CT",))
        dataset = json.loads((config_dir / "dataset.json").read_text(encoding="utf-8"))
        dataset["modality"] = dataset.pop("channel_names")
        (config_dir / "dataset.json").write_text(json.dumps(dataset), encoding="utf-8")
        report = external.scan_external_model(self.root / "Dataset123_Example")
        self.assertEqual({"0": "CT"}, report["candidates"][0]["channel_names"])
        self.assertEqual([], report["candidates"][0]["blockers"])

    def test_scan_region_based_labels_are_blocked(self):
        build_model(self.root, labels={"tumor": 1, "kidney": (2, 3)})
        report = external.scan_external_model(self.root / "Dataset123_Example")
        blockers = "；".join(report["candidates"][0]["blockers"])
        self.assertIn("region-based", blockers)
        self.assertEqual([], report["importable"])

    def test_scan_cascade_configuration_is_blocked(self):
        build_model(self.root, configuration="3d_cascade_fullres")
        report = external.scan_external_model(self.root / "Dataset123_Example")
        blockers = "；".join(report["candidates"][0]["blockers"])
        self.assertIn("级联", blockers)
        self.assertEqual([], report["importable"])

    def test_scan_custom_trainer_is_blocked(self):
        build_model(self.root, trainer="MyCustomTrainer")
        report = external.scan_external_model(self.root / "Dataset123_Example")
        blockers = "；".join(report["candidates"][0]["blockers"])
        self.assertIn("MyCustomTrainer", blockers)
        self.assertIn("自定义训练器", blockers)
        self.assertEqual([], report["importable"])

    def test_scan_non_three_part_dir_name_is_blocked(self):
        config_dir = build_model(self.root)
        renamed = config_dir.with_name("mymodel")
        config_dir.rename(renamed)
        report = external.scan_external_model(self.root / "Dataset123_Example")
        blockers = "；".join(report["candidates"][0]["blockers"])
        self.assertIn("三段式", blockers)
        self.assertEqual([], report["importable"])

    def test_scan_fold_all_and_numeric_folds_require_selection(self):
        build_model(self.root, folds=("fold_0", "fold_all"))
        report = external.scan_external_model(self.root / "Dataset123_Example")
        row = report["candidates"][0]
        self.assertTrue(row["requires_fold_selection"])
        self.assertEqual([], row["blockers"])

    def test_malicious_checkpoint_is_rejected_without_executing_payload(self):
        marker = self.root / "executed.txt"
        config_dir = build_model(self.root)

        # The payload must run at load time, not save time: __reduce__ only
        # returns the (forbidden) callable here. torch.save executing it
        # during pickling would be a test bug, not a loader bypass.
        class Evil(object):
            def __reduce__(self):
                return (open, (str(marker), "w"))

        torch.save({"trainer_name": Evil()}, str(config_dir / "fold_0" / "checkpoint_final.pth"))
        self.assertFalse(marker.exists())
        # A rejected checkpoint is reported as a scan blocker, never a crash.
        report = external.scan_external_model(self.root / "Dataset123_Example")
        blockers = "；".join(report["candidates"][0]["blockers"])
        self.assertIn("受限反序列化", blockers)
        self.assertFalse(marker.exists())


class ExternalZipSafety(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bundle = self.root / "model.zip"

    def make_zip(self, entries):
        with zipfile.ZipFile(self.bundle, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, data in entries:
                archive.writestr(name, data)
        return self.bundle

    def extract(self):
        destination = Path(tempfile.mkdtemp(dir=self.root))
        external._extract_zip_strict(self.bundle, destination)
        return destination

    def test_normal_official_zip_extracts(self):
        # Zip the dataset-level directory so the official
        # Dataset/Trainer__Plans__Configuration layout survives extraction.
        source = build_model(self.root / "source")
        build_zip(source.parent, self.bundle)
        destination = self.extract()
        report = external.scan_external_model(destination)
        self.assertEqual([], report["candidates"][0]["blockers"])

    def test_traversal_entry_is_rejected(self):
        self.make_zip([("../evil.txt", b"x"), ("dataset.json", b"{}")])
        with self.assertRaisesRegex(RuntimeError, "不安全路径"):
            self.extract()

    def test_absolute_and_drive_letter_entries_are_rejected(self):
        # Literal-backslash names cannot be produced on Windows: zipfile's
        # ZipInfo normalizes os.sep to "/", yielding a safe nested path.
        for name in ("/etc/evil", "C:/evil"):
            with self.subTest(name=name):
                self.make_zip([(name, b"x")])
                with self.assertRaisesRegex(RuntimeError, "不安全路径"):
                    self.extract()

    def test_symlink_entry_is_rejected(self):
        info = zipfile.ZipInfo("link.pth")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        with zipfile.ZipFile(self.bundle, "w") as archive:
            archive.writestr(info, "target.pth")
        with self.assertRaisesRegex(RuntimeError, "符号链接"):
            self.extract()

    def test_ratio_bomb_is_rejected(self):
        self.make_zip([("weights.bin", b"\x00" * (6 * 1024 * 1024))])
        with self.assertRaisesRegex(RuntimeError, "压缩炸弹"):
            self.extract()

    def test_duplicate_casefold_names_are_rejected(self):
        self.make_zip(
            [("fold_0/checkpoint_final.pth", b"a"), ("FOLD_0/CHECKPOINT_FINAL.PTH", b"b")]
        )
        with self.assertRaisesRegex(RuntimeError, "重复路径"):
            self.extract()

    def test_file_count_limit_is_enforced(self):
        self.make_zip([("a.txt", b"x"), ("b.txt", b"x"), ("c.txt", b"x")])
        with unittest.mock.patch.object(external, "MAX_ZIP_FILES", 2):
            with self.assertRaisesRegex(RuntimeError, "超过上限"):
                self.extract()

    def test_total_size_limit_is_enforced(self):
        self.make_zip([("a.bin", b"x" * 4096)])
        with unittest.mock.patch.object(external, "MAX_ZIP_TOTAL_BYTES", 1024):
            with self.assertRaisesRegex(RuntimeError, "总大小"):
                self.extract()


class ExternalModelImport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = build_model(self.root / "source")
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()

    def external_root(self):
        return self.workspace / "external_models"

    def test_import_publishes_manifest_outside_live_registry(self):
        result = external.import_external_model(self.source, self.workspace)
        self.assertFalse(result["reused"])
        manifest = result["manifest"]
        self.assertEqual("mimics_nnunet_external_model.v1", manifest["schema_version"])
        self.assertEqual("external_pretrained_unverified", manifest["origin"])
        self.assertEqual("unknown", manifest["source"]["license"])
        self.assertEqual(["fold_0"], manifest["recipe"]["use_folds"])
        self.assertEqual(
            self.external_root().resolve(), Path(result["destination"]).parent.parent
        )
        self.assertTrue(
            (Path(result["destination"]) / "external_model_manifest.json").is_file()
        )
        # Deliberately NOT in the live prediction registry: the recipe must
        # not be inferred from a foreign folder name by Action4_Predict.
        import nnunet_common
        self.assertEqual([], [m for m in nnunet_common.load_models(self.workspace)
                              if m.get("model_id") == manifest["model_id"]])

    def test_import_same_content_twice_reuses_single_version(self):
        first = external.import_external_model(self.source, self.workspace)
        second = external.import_external_model(self.source, self.workspace)
        self.assertTrue(second["reused"])
        self.assertEqual(first["model_id"], second["model_id"])
        model_dirs = list(self.external_root().rglob("external_model_manifest.json"))
        self.assertEqual(1, len(model_dirs))

    def test_import_zip_source_publishes_same_model(self):
        # Official-style ZIP keeps the Dataset/Trainer__Plans__Config layout,
        # so the three-part configuration name survives the round trip.
        bundle = build_zip(self.source.parent, self.root / "model.zip")
        result = external.import_external_model(bundle, self.workspace)
        self.assertFalse(result["reused"])
        self.assertEqual("zip", result["manifest"]["source"]["kind"])
        reused = external.import_external_model(self.source, self.workspace)
        self.assertTrue(reused["reused"])

    def test_import_fold_all_requires_explicit_choice(self):
        self.source = build_model(self.root / "source_all", folds=("fold_0", "fold_all"))
        with self.assertRaisesRegex(RuntimeError, "明确选择要使用的 fold"):
            external.import_external_model(self.source, self.workspace)
        result = external.import_external_model(
            self.source, self.workspace, use_folds=["fold_all"]
        )
        self.assertEqual(["fold_all"], result["manifest"]["recipe"]["use_folds"])

    def test_import_multiple_configurations_requires_selection(self):
        build_model(self.root / "source", configuration="2d")
        with self.assertRaisesRegex(RuntimeError, "多个可用配置"):
            external.import_external_model(self.source.parent, self.workspace)
        result = external.import_external_model(
            self.source.parent, self.workspace, configuration="3d_fullres"
        )
        self.assertEqual("3d_fullres", result["manifest"]["configuration"])

    def test_import_unknown_fold_choice_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "fold 不可用"):
            external.import_external_model(self.source, self.workspace, use_folds=["fold_3"])
        self.assertFalse(self.external_root().exists())

    def test_import_blocked_model_leaves_nothing_and_source_untouched(self):
        self.source = build_model(self.root / "source", trainer="MyCustomTrainer")
        before = sorted(p.name for p in self.source.rglob("*"))
        with self.assertRaisesRegex(RuntimeError, "自定义训练器"):
            external.import_external_model(self.source, self.workspace)
        self.assertFalse(self.external_root().exists())
        self.assertEqual(before, sorted(p.name for p in self.source.rglob("*")))

    def test_import_detects_source_changed_during_copy(self):
        inventory = external._file_inventory(self.source)
        digest = external._inventory_digest(inventory)
        with unittest.mock.patch.object(
            external, "_inventory_digest", side_effect=[digest, "mismatch"]
        ):
            with self.assertRaisesRegex(RuntimeError, "复制期间"):
                external.import_external_model(self.source, self.workspace)
        # No staging residue and no half-published model.
        if self.external_root().exists():
            self.assertEqual([], list(self.external_root().rglob("*")))

    def test_import_dedup_keeps_existing_model_on_content_collision(self):
        first = external.import_external_model(self.source, self.workspace)
        # Same content digest resolving to an existing dir with a foreign
        # manifest must abort rather than overwrite in place.
        destination = Path(first["destination"])
        (destination / "external_model_manifest.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "内容不一致"):
            external.import_external_model(self.source, self.workspace)


if __name__ == "__main__":
    unittest.main(verbosity=2)
