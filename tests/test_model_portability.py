#!/usr/bin/env python3
"""Regression tests for cross-machine AI model relocation."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
for value in (str(ROOT), str(ROOT / "tools")):
    if value not in sys.path:
        sys.path.insert(0, value)

import ai_model_bundle
import nninteractive_task_common as task_common
import flexict_common
import nnunet_common


def _torch_checkpoint_bytes(payload=None) -> bytes:
    """A real torch.save'd checkpoint, so torch.load-based import
    validation exercises the true serialization path (Phase 3b)."""
    import io

    import torch

    buffer = io.BytesIO()
    torch.save(payload or {"epoch": 2, "state_dict": {}}, buffer)
    return buffer.getvalue()


def _nninteractive_model(root: Path) -> Path:
    model = root / "tasks" / "brain" / "models" / "v1"
    for name in task_common.MODEL_METADATA_FILES:
        path = model / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    checkpoint = model / "fold_0" / "checkpoint_final.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(_torch_checkpoint_bytes())
    (model / "finetune_manifest.json").write_text(
        json.dumps(
            {
                "input_contract": task_common.NNINTERACTIVE_INPUT_CONTRACT,
                "runtime_verification": {
                    "verified": True,
                    "expected_parameter_fingerprint": "portable-runtime-v1",
                    "loaded_parameter_fingerprint": "portable-runtime-v1",
                },
                "validated_prompt_types": ["point"],
            }
        ),
        encoding="utf-8",
    )
    return model


class ModelPortabilityTests(unittest.TestCase):
    def test_nninteractive_registry_survives_workspace_relocation(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source_workspace"
            model = _nninteractive_model(source)
            registry = {
                "tasks": [
                    {
                        "task_id": "brain",
                        "task_name": "Brain",
                        "models": [
                            {
                                "model_id": "v1",
                                "model_dir": str(model),
                            }
                        ],
                        "recommended_model_id": "v1",
                    }
                ]
            }
            task_common.save_registry(source, registry)
            saved = task_common.load_registry(source)
            record = saved["tasks"][0]["models"][0]
            self.assertNotIn("model_dir", record)
            self.assertEqual(
                "tasks/brain/models/v1", record.get("model_relpath")
            )

            relocated = root / "windows_copy"
            shutil.copytree(source, relocated)
            task = task_common.find_task(relocated, "brain")
            selected = task_common.selected_model(relocated, "brain")
            resolved = task_common.resolve_registered_model_dir(
                relocated, selected, task
            )
            self.assertEqual(
                (relocated / "tasks" / "brain" / "models" / "v1").resolve(),
                resolved,
            )
            self.assertTrue(task_common.audit_model_dir(resolved)["compatible"])

    def test_nninteractive_bundle_import_rebuilds_local_registry(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source"
            model = _nninteractive_model(source)
            task_common.save_registry(
                source,
                {
                    "tasks": [
                        {
                            "task_id": "brain",
                            "task_name": "Brain",
                            "mask_names": ["Brain"],
                            "models": [
                                {
                                    "model_id": "v1",
                                    "model_dir": str(model),
                                    "checkpoint_sha256": task_common.sha256_file(
                                        model
                                        / "fold_0"
                                        / "checkpoint_final.pth"
                                    ),
                                }
                            ],
                            "recommended_model_id": "v1",
                        }
                    ]
                },
            )
            bundle = root / "brain.zip"
            ai_model_bundle.main(
                [
                    "export-nninteractive",
                    "--workspace",
                    str(source),
                    "--task-id",
                    "brain",
                    "--output",
                    str(bundle),
                ]
            )
            target = root / "target"
            ai_model_bundle.main(
                [
                    "import-nninteractive",
                    "--workspace",
                    str(target),
                    "--bundle",
                    str(bundle),
                    "--set-current",
                ]
            )
            task = task_common.find_task(target, "brain")
            model_record = task_common.selected_model(target, "brain")
            self.assertEqual("v1", task.get("recommended_model_id"))
            self.assertTrue(
                task_common.resolve_registered_model_dir(
                    target, model_record, task
                ).is_dir()
            )





def _nnunet_model_dir(workspace: Path, task_id: str, model_id: str) -> Path:
    model_dir = (
        nnunet_common.workspace_paths(workspace)["models"] / task_id / model_id
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    for name in ("plans.json", "dataset.json"):
        (model_dir / name).write_text("{}", encoding="utf-8")
    fold = model_dir / "fold_0"
    fold.mkdir()
    (fold / "checkpoint_final.pth").write_bytes(_torch_checkpoint_bytes())
    manifest = {
        "schema_version": nnunet_common.MODEL_SCHEMA_VERSION,
        "model_id": model_id,
        "task_id": "liver_task",
        "task_name": "Liver",
        "labels": [{"name": "liver", "id": 1, "aliases": []}],
        "modality": "CT",
        "dataset_id": 701,
        "dataset_name": "Dataset701_Liver",
        "configuration": "3d_fullres",
        "trainer": "MimicsNNUNetTrainer",
        "plans": "nnUNetPlans",
        "folds": ["0"],
        "epochs": 2,
        "dataset_fingerprint": "abc",
        "training_data_profile": {},
        "model_dir": str(model_dir),
        "manifest_path": str(model_dir / "mimics_model_manifest.json"),
        "execution_backend": "local",
        "created_at_epoch": 1700000000.0,
    }
    nnunet_common.write_json_atomic(
        model_dir / "mimics_model_manifest.json", manifest
    )
    nnunet_common.register_model(workspace, manifest)
    return model_dir


class NnunetBundleTests(unittest.TestCase):
    def test_nninteractive_import_rejects_corrupt_checkpoint(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source"
            model = _nninteractive_model(source)
            (model / "fold_0" / "checkpoint_final.pth").write_bytes(
                b"\x80truncated garbage"
            )
            task_common.save_registry(
                source,
                {
                    "tasks": [
                        {
                            "task_id": "brain",
                            "task_name": "Brain",
                            "models": [
                                {
                                    "model_id": "v1",
                                    "model_dir": str(model),
                                    "checkpoint_sha256": task_common.sha256_file(
                                        model / "fold_0" / "checkpoint_final.pth"
                                    ),
                                }
                            ],
                            "recommended_model_id": "v1",
                        }
                    ]
                },
            )
            bundle = root / "brain.zip"
            ai_model_bundle.main(
                [
                    "export-nninteractive",
                    "--workspace",
                    str(source),
                    "--task-id",
                    "brain",
                    "--output",
                    str(bundle),
                ]
            )
            target = root / "target"
            with self.assertRaises(RuntimeError) as ctx:
                ai_model_bundle.main(
                    [
                        "import-nninteractive",
                        "--workspace",
                        str(target),
                        "--bundle",
                        str(bundle),
                        "--set-current",
                    ]
                )
            self.assertIn("not a loadable torch file", str(ctx.exception))
            # Nothing was published into the workspace.
            self.assertFalse((target / "tasks" / "brain" / "models" / "v1").exists())

    def test_nnunet_bundle_roundtrip(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source_workspace"
            _nnunet_model_dir(source, "liver_task", "nnunet_v1")
            bundle = root / "liver.zip"
            ai_model_bundle.main(
                [
                    "export-nnunet",
                    "--workspace",
                    str(source),
                    "--model-id",
                    "nnunet_v1",
                    "--output",
                    str(bundle),
                ]
            )
            self.assertTrue(bundle.is_file())
            target = root / "target"
            ai_model_bundle.main(
                [
                    "import-nnunet",
                    "--workspace",
                    str(target),
                    "--bundle",
                    str(bundle),
                ]
            )
            models = {
                str(row.get("model_id")): row
                for row in nnunet_common.load_models(target)
            }
            self.assertIn("nnunet_v1", models)
            imported = Path(models["nnunet_v1"]["model_dir"])
            self.assertTrue((imported / "fold_0" / "checkpoint_final.pth").is_file())
            usable, reason = nnunet_common.model_usability(models["nnunet_v1"])
            self.assertTrue(usable, reason)
            # Import must not leak the exporting machine's absolute paths.
            self.assertEqual(
                imported, models["nnunet_v1"]["model_dir"].strip()
                and Path(models["nnunet_v1"]["model_dir"])
            )
            manifest = json.loads(
                (imported / "mimics_model_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(str(imported), manifest["model_dir"])
            self.assertNotIn(str(source), manifest["model_dir"])

    def test_nnunet_import_rejects_wrong_family(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source_workspace"
            _nnunet_model_dir(source, "liver_task", "nnunet_v1")
            bundle = root / "liver.zip"
            ai_model_bundle.main(
                [
                    "export-nnunet",
                    "--workspace",
                    str(source),
                    "--model-id",
                    "nnunet_v1",
                    "--output",
                    str(bundle),
                ]
            )
            with self.assertRaises(RuntimeError):
                ai_model_bundle.main(
                    [
                        "import-nninteractive",
                        "--workspace",
                        str(root / "target"),
                        "--bundle",
                        str(bundle),
                    ]
                )

    def test_nnunet_import_rejects_corrupt_checkpoint(self):
        """A truncated checkpoint must fail at import, not at inference.

        Byte-presence and checksums alone accept garbage files; the Phase
        3b torch.load validation must reject them with a clear message.
        """
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source_workspace"
            _nnunet_model_dir(source, "liver_task", "nnunet_v1")
            # Corrupt the checkpoint in place, then export the bundle.
            checkpoint = (
                nnunet_common.workspace_paths(source)["models"]
                / "liver_task"
                / "nnunet_v1"
                / "fold_0"
                / "checkpoint_final.pth"
            )
            checkpoint.write_bytes(checkpoint.read_bytes()[:20])
            bundle = root / "corrupt.zip"
            ai_model_bundle.main(
                [
                    "export-nnunet",
                    "--workspace",
                    str(source),
                    "--model-id",
                    "nnunet_v1",
                    "--output",
                    str(bundle),
                ]
            )
            target = root / "target"
            with self.assertRaises(RuntimeError) as ctx:
                ai_model_bundle.main(
                    [
                        "import-nnunet",
                        "--workspace",
                        str(target),
                        "--bundle",
                        str(bundle),
                    ]
                )
            self.assertIn("not a loadable torch file", str(ctx.exception))
            # Nothing was published into the workspace.
            self.assertFalse(
                (
                    nnunet_common.workspace_paths(target)["models"]
                    / "liver_task"
                    / "nnunet_v1"
                ).exists()
            )


def _flexict_model_dir(
    workspace: Path,
    model_id: str,
    configuration: str,
    pair_id: str = "",
    created: float = 1700000000.0,
    label_names: list[str] | None = None,
) -> dict:
    model_dir = (
        flexict_common.workspace_paths(workspace)["root"]
        / "models"
        / "kidney"
        / model_id
    )
    fold = model_dir / "fold_0"
    fold.mkdir(parents=True, exist_ok=True)
    for name in ("plans.json", "dataset.json"):
        (model_dir / name).write_text("{}", encoding="utf-8")
    for name in ("checkpoint_best.pth", "checkpoint_final.pth"):
        (fold / name).write_bytes(
            _torch_checkpoint_bytes(
                {"config": "portable-flexict-{}".format(configuration)}
            )
        )
    manifest = {
        "schema_version": flexict_common.MODEL_SCHEMA_VERSION,
        "model_id": model_id,
        "task_id": "kidney",
        "task_name": "Kidney",
        "label_name": label_names[0] if label_names else "kidney_left",
        "label_names": label_names or ["kidney_left"],
        "configuration": configuration,
        "trainer": "flexict2d_Trainer"
        if configuration == "2d"
        else "flexict3d_Trainer",
        "dataset_id": 750,
        "dataset_name": "Dataset750_kidney",
        "dataset_fingerprint": "abc",
        "pair_id": pair_id,
        "epochs": 150,
        "mirror_disable_axes": "",
        "training_data_profile": {},
        "model_dir": str(model_dir),
        "manifest_path": str(model_dir / "flexict_model_manifest.json"),
        "execution_backend": "local",
        "created_at_epoch": created,
    }
    flexict_common.workspace_paths(workspace)["root"].mkdir(
        parents=True, exist_ok=True
    )
    # Register via the public helper (host write_json_atomic import inside
    # save_registry requires tools/ on sys.path, which setUp in the class
    # below guarantees for tests; here we call register_model directly).
    flexict_common.register_model(manifest, workspace=workspace)
    return manifest


class FlexictBundleTests(unittest.TestCase):
    def setUp(self):
        for value in (str(ROOT), str(ROOT / "tools")):
            if value not in sys.path:
                sys.path.insert(0, value)

    def test_flexict_pair_bundle_roundtrip(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source_workspace"
            _flexict_model_dir(source, "flexict_2d_v1", "2d", "pair_a", 1700000000.0)
            _flexict_model_dir(source, "flexict_3d_v1", "3d_fullres", "pair_a", 1700000100.0)
            bundle = root / "kidney_pair.zip"
            # Exporting either member of the pair must package both.
            ai_model_bundle.main(
                [
                    "export-flexict",
                    "--workspace",
                    str(source),
                    "--model-id",
                    "flexict_2d_v1",
                    "--output",
                    str(bundle),
                ]
            )
            self.assertTrue(bundle.is_file())
            target = root / "target"
            ai_model_bundle.main(
                [
                    "import-flexict",
                    "--workspace",
                    str(target),
                    "--bundle",
                    str(bundle),
                    "--set-current",
                ]
            )
            models = {
                str(row.get("model_id")): row
                for row in flexict_common.load_models(target)
            }
            self.assertEqual({"flexict_2d_v1", "flexict_3d_v1"}, set(models))
            pair = flexict_common.load_pair(target)
            self.assertIsNotNone(pair[0])
            self.assertIsNotNone(pair[1])
            # The recommended model is the imported 2D member.
            recommended = flexict_common.recommended_model(target)
            self.assertIsNotNone(recommended)
            self.assertEqual("flexict_2d_v1", recommended["model_id"])
            # Usability after relocation.
            for row in models.values():
                usable, reason = flexict_common.model_usability(row)
                self.assertTrue(usable, reason)
            # No absolute path of the exporting machine leaks.
            for row in models.values():
                self.assertNotIn(str(source), str(row.get("model_dir")))

    def test_flexict_import_rejects_corrupt_checkpoint(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source_workspace"
            _flexict_model_dir(source, "flexict_bad", "2d", "")
            fold = (
                flexict_common.workspace_paths(source)["root"]
                / "models"
                / "kidney"
                / "flexict_bad"
                / "fold_0"
            )
            (fold / "checkpoint_best.pth").write_bytes(b"\x80truncated garbage")
            bundle = root / "bad.zip"
            ai_model_bundle.main(
                [
                    "export-flexict",
                    "--workspace",
                    str(source),
                    "--model-id",
                    "flexict_bad",
                    "--output",
                    str(bundle),
                ]
            )
            target = root / "target"
            with self.assertRaises(RuntimeError) as ctx:
                ai_model_bundle.main(
                    [
                        "import-flexict",
                        "--workspace",
                        str(target),
                        "--bundle",
                        str(bundle),
                    ]
                )
            self.assertIn("not a loadable torch file", str(ctx.exception))

    def test_flexict_single_model_bundle_roundtrip(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source_workspace"
            _flexict_model_dir(source, "flexict_solo", "2d", "")
            bundle = root / "solo.zip"
            ai_model_bundle.main(
                [
                    "export-flexict",
                    "--workspace",
                    str(source),
                    "--model-id",
                    "flexict_solo",
                    "--output",
                    str(bundle),
                ]
            )
            target = root / "target"
            ai_model_bundle.main(
                [
                    "import-flexict",
                    "--workspace",
                    str(target),
                    "--bundle",
                    str(bundle),
                ]
            )
            models = flexict_common.load_models(target)
            self.assertEqual(1, len(models))
            self.assertEqual("flexict_solo", models[0]["model_id"])
            # A pair-less model must not fabricate a pair on the target.
            pair = flexict_common.load_pair(target)
            self.assertIsNone(pair[0])

    def test_flexict_multiclass_bundle_roundtrip(self):
        # G5: the multiclass feature (liver+spleen in one model) shipped
        # without bundle coverage - a multiclass model exported, imported,
        # and predicted across machines had no test proving label_names
        # survives the roundtrip and the model stays usable.
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source_workspace"
            _flexict_model_dir(
                source, "flexict_multi", "2d", "",
                label_names=["liver", "spleen"],
            )
            bundle = root / "multi.zip"
            ai_model_bundle.main(
                [
                    "export-flexict",
                    "--workspace",
                    str(source),
                    "--model-id",
                    "flexict_multi",
                    "--output",
                    str(bundle),
                ]
            )
            target = root / "target"
            ai_model_bundle.main(
                [
                    "import-flexict",
                    "--workspace",
                    str(target),
                    "--bundle",
                    str(bundle),
                ]
            )
            models = flexict_common.load_models(target)
            self.assertEqual(1, len(models))
            self.assertEqual(
                ["liver", "spleen"], models[0].get("label_names")
            )
            usable, _ = flexict_common.model_usability(models[0])
            self.assertTrue(usable)


def _fake_backbone(root: Path, salt: bytes = b"") -> Path:
    """A stand-in backbone dir that resolve_pretrained_dir() would accept."""
    for config in ("flexict_2d", "flexict_3d"):
        target = root / config
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.safetensors").write_bytes(b"backbone" + config.encode() + salt)
    return root


class FlexictBackboneBundleTests(unittest.TestCase):
    """D15: --include-backbone packages the pretrained backbone weights so
    an imported fine-tuned model is actually runnable on a workstation that
    never had them (the trainer loads the backbone while building the
    network - F06)."""

    def setUp(self):
        for value in (str(ROOT), str(ROOT / "tools")):
            if value not in sys.path:
                sys.path.insert(0, value)
        # ai_model_bundle's functions import "tools.flexict_common" while
        # this file imports "flexict_common" - two module objects for the
        # same file. Patch the one the code under test actually uses.
        import importlib

        self.tools_flexict_common = importlib.import_module(
            "tools.flexict_common"
        )

    def _patched_backbone(self, source_weights: Path | None, target_weights: Path | None = ...):
        """Redirect _flexict_backbone_files()/_target_backbone_dir() away
        from the real repo weights and the global flexict_config.json.

        source_weights: what the *exporting* machine has (None = export
        without --include-backbone is still fine; the flag itself probes).
        target_weights: what the *importing* machine has (None = bare
        workstation). Default (ellipsis) = same as source.
        """
        if target_weights is ...:
            target_weights = source_weights

        def fake_files():
            if source_weights is None:
                raise FileNotFoundError("no backbone on this workstation")
            return [
                ("flexict_2d/model.safetensors",
                 source_weights / "flexict_2d" / "model.safetensors"),
                ("flexict_3d/model.safetensors",
                 source_weights / "flexict_3d" / "model.safetensors"),
            ]

        def fake_target():
            if target_weights is None:
                raise FileNotFoundError("no backbone on this workstation")
            return target_weights, "config"

        return mock.patch.multiple(
            ai_model_bundle,
            _flexict_backbone_files=fake_files,
            _target_backbone_dir=fake_target,
        )

    def test_backbone_roundtrip_installs_on_bare_workstation(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source_weights = _fake_backbone(root / "source_weights")
            source = root / "source_workspace"
            _flexict_model_dir(source, "flexict_bb", "2d", "")
            bundle = root / "bb.zip"
            with self._patched_backbone(source_weights):
                ai_model_bundle.main(
                    [
                        "export-flexict",
                        "--workspace",
                        str(source),
                        "--model-id",
                        "flexict_bb",
                        "--output",
                        str(bundle),
                        "--include-backbone",
                    ]
                )
            with zipfile.ZipFile(bundle) as archive:
                names = archive.namelist()
            self.assertIn("backbone/flexict_2d/model.safetensors", names)
            self.assertIn("backbone/flexict_3d/model.safetensors", names)

            target = root / "target"
            saved_configs: list[dict] = []

            def fake_save_config(config, explicit_path=None):
                saved_configs.append(dict(config))
                return Path(explicit_path) if explicit_path else Path(root) / "flexict_config.json"

            # Target workstation has NO backbone anywhere.
            with self._patched_backbone(source_weights, None), mock.patch.object(
                self.tools_flexict_common, "save_config", fake_save_config
            ):
                ai_model_bundle.main(
                    [
                        "import-flexict",
                        "--workspace",
                        str(target),
                        "--bundle",
                        str(bundle),
                    ]
                )
            # The weights landed inside the import workspace and the config
            # override points at them.
            installed = target / "flexict_pretrained"
            for config in ("flexict_2d", "flexict_3d"):
                self.assertEqual(
                    (installed / config / "model.safetensors").read_bytes(),
                    (source_weights / config / "model.safetensors").read_bytes(),
                )
            self.assertEqual(len(saved_configs), 1)
            self.assertEqual(
                saved_configs[0]["pretrained_weights_dir"], str(installed)
            )
            # The model itself still registered normally.
            models = flexict_common.load_models(target)
            self.assertEqual(1, len(models))

    def test_existing_backbone_is_verified_not_overwritten(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source_weights = _fake_backbone(root / "source_weights")
            source = root / "source_workspace"
            _flexict_model_dir(source, "flexict_bb2", "2d", "")
            bundle = root / "bb2.zip"
            with self._patched_backbone(source_weights):
                ai_model_bundle.main(
                    [
                        "export-flexict",
                        "--workspace",
                        str(source),
                        "--model-id",
                        "flexict_bb2",
                        "--output",
                        str(bundle),
                        "--include-backbone",
                    ]
                )
            # Target already has identical weights in an admin location.
            target_weights = _fake_backbone(root / "target_weights")
            existing_bytes = (
                target_weights / "flexict_2d" / "model.safetensors"
            ).read_bytes()
            target = root / "target"
            with self._patched_backbone(source_weights, target_weights), mock.patch.object(
                self.tools_flexict_common, "save_config"
            ) as save_config:
                ai_model_bundle.main(
                    [
                        "import-flexict",
                        "--workspace",
                        str(target),
                        "--bundle",
                        str(bundle),
                    ]
                )
                save_config.assert_not_called()
            # Untouched, byte for byte.
            self.assertEqual(
                (target_weights / "flexict_2d" / "model.safetensors").read_bytes(),
                existing_bytes,
            )
            self.assertFalse((target / "flexict_pretrained").exists())

    def test_mismatched_existing_backbone_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source_weights = _fake_backbone(root / "source_weights")
            source = root / "source_workspace"
            _flexict_model_dir(source, "flexict_bb3", "2d", "")
            bundle = root / "bb3.zip"
            with self._patched_backbone(source_weights):
                ai_model_bundle.main(
                    [
                        "export-flexict",
                        "--workspace",
                        str(source),
                        "--model-id",
                        "flexict_bb3",
                        "--output",
                        str(bundle),
                        "--include-backbone",
                    ]
                )
            # Target backbone differs: importing would silently pair the
            # model with a backbone it was never fine-tuned from.
            target_weights = _fake_backbone(
                root / "target_weights", salt=b"-different"
            )
            target = root / "target"
            with self._patched_backbone(source_weights, target_weights), mock.patch.object(
                self.tools_flexict_common, "save_config"
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    ai_model_bundle.main(
                        [
                            "import-flexict",
                            "--workspace",
                            str(target),
                            "--bundle",
                            str(bundle),
                        ]
                    )
            self.assertIn("differ from the packaged", str(ctx.exception))
            # Nothing was published.
            self.assertFalse(
                (flexict_common.workspace_paths(target)["root"] / "models").exists()
            )

    def test_bundle_without_backbone_still_imports(self):
        # Backwards compatibility: pre-D15 bundles (no --include-backbone)
        # carry no backbone section and must import unchanged on a
        # workstation with or without backbone weights.
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source_weights = _fake_backbone(root / "source_weights")
            source = root / "source_workspace"
            _flexict_model_dir(source, "flexict_plain", "2d", "")
            bundle = root / "plain.zip"
            with self._patched_backbone(source_weights):
                ai_model_bundle.main(
                    [
                        "export-flexict",
                        "--workspace",
                        str(source),
                        "--model-id",
                        "flexict_plain",
                        "--output",
                        str(bundle),
                    ]
                )
            target = root / "target"
            with self._patched_backbone(source_weights, None), mock.patch.object(
                self.tools_flexict_common, "save_config"
            ) as save_config:
                ai_model_bundle.main(
                    [
                        "import-flexict",
                        "--workspace",
                        str(target),
                        "--bundle",
                        str(bundle),
                    ]
                )
                save_config.assert_not_called()
            self.assertFalse((target / "flexict_pretrained").exists())
            self.assertEqual(1, len(flexict_common.load_models(target)))


class ModelManagerTests(unittest.TestCase):
    """The unified model manager: cross-family listing + import dispatch."""

    def setUp(self):
        for value in (str(ROOT), str(ROOT / "tools")):
            if value not in sys.path:
                sys.path.insert(0, value)
        import model_manager_ui

        self.manager = model_manager_ui

    def _isolated_home(self):
        """Point Path.home() at a temp dir so the global nnU-Net model
        registry (~/.mimics_script/nnunet_model_registry.json) is not
        polluted by test registrations."""
        home = tempfile.mkdtemp(prefix="mgr_home_")
        self.addCleanup(shutil.rmtree, home, ignore_errors=True)
        return mock.patch.dict(
            os.environ,
            {"USERPROFILE": home, "HOME": home},
            clear=False,
        )

    def test_import_dispatches_by_bundle_family(self):
        with tempfile.TemporaryDirectory() as raw, self._isolated_home():
            root = Path(raw)
            source = root / "source"
            _flexict_model_dir(source, "flexict_mgr", "2d", "")
            bundle = root / "mgr.zip"
            ai_model_bundle.main(
                [
                    "export-flexict",
                    "--workspace",
                    str(source),
                    "--model-id",
                    "flexict_mgr",
                    "--output",
                    str(bundle),
                ]
            )
            self.assertEqual(
                "flexict", self.manager.bundle_model_family(bundle)
            )
            target = root / "target"
            workspaces = {"flexict": str(target)}
            self.manager.import_bundle(bundle, workspaces, set_current=True)
            models = flexict_common.load_models(target)
            self.assertEqual(1, len(models))
            self.assertTrue(models[0].get("imported_from_bundle"))
            recommended = flexict_common.recommended_model(target)
            self.assertEqual("flexict_mgr", recommended["model_id"])

    def test_import_rejects_non_bundle_zip(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fake = root / "not_a_bundle.zip"
            with zipfile.ZipFile(fake, "w") as archive:
                archive.writestr("random.txt", "hello")
            with self.assertRaises(RuntimeError):
                self.manager.import_bundle(
                    fake, {"flexict": str(root / "target")}
                )

    def test_collect_rows_covers_all_three_families(self):
        with tempfile.TemporaryDirectory() as raw, self._isolated_home():
            root = Path(raw)
            nnint_ws = root / "nnint"
            _nninteractive_model(nnint_ws)
            task_common.save_registry(
                nnint_ws,
                {
                    "tasks": [
                        {
                            "task_id": "brain",
                            "task_name": "Brain",
                            "models": [
                                {
                                    "model_id": "v1",
                                    "model_dir": str(
                                        nnint_ws
                                        / "tasks"
                                        / "brain"
                                        / "models"
                                        / "v1"
                                    ),
                                }
                            ],
                            "recommended_model_id": "v1",
                        }
                    ]
                },
            )
            nnunet_ws = root / "nnunet"
            _nnunet_model_dir(nnunet_ws, "liver_task", "nnunet_mgr")
            flexict_ws = root / "flexict"
            _flexict_model_dir(flexict_ws, "flexict_mgr", "2d", "")
            rows = self.manager.collect_rows(
                {
                    "nninteractive": str(nnint_ws),
                    "nnunet": str(nnunet_ws),
                    "flexict": str(flexict_ws),
                }
            )
            families = {row["family"] for row in rows}
            self.assertEqual(
                {"nninteractive", "nnunet", "flexict"}, families
            )
            self.assertTrue(all(row["usable"] for row in rows))
            # The nnInteractive recommended model is flagged as current.
            current = [
                row for row in rows
                if row["family"] == "nninteractive"
                and row["model_id"] == "v1"
            ]
            self.assertTrue(current and current[0]["current"])

    def test_set_recommended_and_remove_for_flexict(self):
        with tempfile.TemporaryDirectory() as raw, self._isolated_home():
            root = Path(raw)
            workspace = root / "ws"
            _flexict_model_dir(workspace, "flexict_a", "2d", "")
            _flexict_model_dir(workspace, "flexict_b", "2d", "")
            rows = self.manager.collect_rows(
                {"flexict": str(workspace)}
            )
            by_id = {row["model_id"]: row for row in rows}
            self.manager.set_recommended_model(by_id["flexict_a"])
            recommended = flexict_common.recommended_model(workspace)
            self.assertEqual("flexict_a", recommended["model_id"])
            # Removing a usable model is refused by the UI; the helper
            # drops the registry row regardless (UI gates it).
            self.manager.remove_model_row(by_id["flexict_b"])
            remaining = {
                row["model_id"]
                for row in self.manager.collect_rows(
                    {"flexict": str(workspace)}
                )
            }
            self.assertEqual({"flexict_a"}, remaining)

    def test_manager_shows_multiclass_flexict_target(self):
        # G4: the manager's target column showed only label_name, so a
        # multiclass (liver+spleen) model displayed as just "liver" - the
        # annotator could not tell single- from multiclass models apart in
        # the manager. Mirror the prediction window's joined display.
        with tempfile.TemporaryDirectory() as raw, self._isolated_home():
            root = Path(raw)
            workspace = root / "ws"
            _flexict_model_dir(
                workspace, "flexict_multi", "2d", "",
                label_names=["liver", "spleen"],
            )
            rows = self.manager.collect_rows({"flexict": str(workspace)})
            multi = [
                row for row in rows if row["model_id"] == "flexict_multi"
            ]
            self.assertEqual(1, len(multi))
            self.assertEqual("liver + spleen", multi[0]["target"])

    def test_removing_recommended_flexict_model_clears_pointer(self):
        # G6: removing the recommended FlexiCT model left the registry's
        # recommended_model_id pointing at a model that no longer exists.
        # The next prediction would try to load a deleted checkpoint.
        with tempfile.TemporaryDirectory() as raw, self._isolated_home():
            root = Path(raw)
            workspace = root / "ws"
            _flexict_model_dir(workspace, "flexict_a", "2d", "")
            _flexict_model_dir(workspace, "flexict_b", "2d", "")
            self.manager.set_recommended_model(
                {"family": "flexict", "workspace": str(workspace),
                 "model_id": "flexict_a"}
            )
            self.manager.remove_model_row(
                {"family": "flexict", "workspace": str(workspace),
                 "model_id": "flexict_a"}
            )
            self.assertEqual(
                "", flexict_common.load_registry_recommended_id(workspace)
            )
            remaining = {
                row["model_id"]
                for row in self.manager.collect_rows(
                    {"flexict": str(workspace)}
                )
            }
            self.assertEqual({"flexict_b"}, remaining)

    def test_manager_files_and_entry_exist(self):
        self.assertTrue(
            (ROOT / "tools" / "model_manager_ui.py").is_file()
        )
        self.assertTrue(
            (ROOT / "runtime_py35" / "model_manager_mimics.py").is_file()
        )
        entry = (
            ROOT / "scripting_library" / "02_AI" / "00_Manage_AI_Models.py"
        )
        self.assertTrue(entry.is_file())
        self.assertIn(
            "model_manager_mimics",
            entry.read_text(encoding="utf-8"),
        )

    def test_manager_window_renders_offscreen(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        import PySide6
        from PySide6 import QtWidgets

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
            ["test"]
        )
        # Empty workspaces: window must still render with zero rows.
        window = self.manager.ModelManagerWindow(
            {
                "workspaces": {
                    "nninteractive": os.path.join(self.manager.ROOT.name, "nope"),
                    "nnunet": "nope",
                    "flexict": "nope",
                }
            },
            (PySide6.QtCore, PySide6.QtGui, QtWidgets),
        )
        self.assertGreaterEqual(window.table.columnCount(), 5)
        window.window.close()


if __name__ == "__main__":
    unittest.main()
