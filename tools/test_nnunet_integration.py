#!/usr/bin/env python3
"""Unit and flow tests for the managed nnU-Net integration."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
RUNTIME = ROOT / "runtime_py35"
WORKFLOW = ROOT / "integrations" / "nnunet_segmentation_workflow"
for path in (ROOT, TOOLS, RUNTIME, WORKFLOW):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

import nnunet_common as common
import nnunet_jobs as jobs
import nnunet_pipeline as pipeline
import nnunet_prediction_setup_ui as prediction_ui
import nnunet_stage_worker as stage_worker
import remote_training_controller as remote
import AutoSegmentationFramework as standalone
import SetEnvionmentVariables as standalone_environment


class ContractTests(unittest.TestCase):
    def test_existing_model_map_loads_multiclass_and_grouped_tasks(self):
        sets = common.load_label_sets(
            ROOT
            / "integrations"
            / "nnunet_segmentation_workflow"
            / "ModelMap.toml"
        )
        self.assertGreater(len(sets["CT1_Head"]), 1)
        coarse = sets["CT_All_Coarse"]
        head = next(row for row in coarse if row["name"] == "CT1_Head")
        self.assertIn("brain", head["aliases"])
        self.assertIn("skull", head["aliases"])
        self.assertEqual(head["source_mode"], "union")
        self.assertEqual(sets["CT1_Head"][0]["source_mode"], "alternatives")
        kidney_task = sets["CT6_KidneySpleen"]
        right_kidney = next(row for row in kidney_task if row["id"] == 2)
        self.assertEqual(right_kidney["source_mode"], "union")
        self.assertIn("kidney_cyst_left", right_kidney["aliases"])
        self.assertNotIn("CT_Combine", sets)

    def test_selected_mask_ranks_but_does_not_filter_multiclass_models(self):
        models = [
            {
                "model_id": "brain",
                "created_at_epoch": 2,
                "labels": [{"name": "brain", "aliases": ["brain"]}],
            },
            {
                "model_id": "abdomen",
                "created_at_epoch": 1,
                "labels": [
                    {"name": "liver", "aliases": ["liver"]},
                    {"name": "spleen", "aliases": ["spleen"]},
                ],
            },
        ]
        ranked = prediction_ui._rank_models(
            models, {"selected_mask_name": "liver"}
        )
        self.assertEqual([row["model_id"] for row in ranked], ["abdomen", "brain"])

    def test_directory_signature_tracks_each_file_timestamp(self):
        with tempfile.TemporaryDirectory() as temporary:
            # The cache is disabled so a fresh directory scan runs even
            # though a previous call cached this exact directory state.
            with mock.patch.dict(
                os.environ, {"MIMICS_PATH_SIGNATURE_CACHE_DIR": "off"}
            ):
                root = Path(temporary) / "series"
                root.mkdir()
                first = root / "slice_001.dcm"
                second = root / "slice_002.dcm"
                first.write_bytes(b"aa")
                second.write_bytes(b"bb")
                os.utime(first, (100, 100))
                os.utime(second, (200, 200))
                before = common.path_signature(root)
                os.utime(first, (150, 150))
                after = common.path_signature(root)
                self.assertEqual(before["size"], after["size"])
                self.assertEqual(before["mtime_ns"], after["mtime_ns"])
                self.assertNotEqual(
                    before["manifest_sha256"], after["manifest_sha256"]
                )

    def test_directory_signature_cache_hit_and_invalidation(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary) / "sigcache"
            root = Path(temporary) / "series"
            root.mkdir()
            first = root / "slice_001.dcm"
            first.write_bytes(b"aa")
            os.utime(first, (100, 100))
            with mock.patch.dict(
                os.environ, {"MIMICS_PATH_SIGNATURE_CACHE_DIR": str(cache_dir)}
            ):
                before = common.path_signature(root)
                self.assertTrue(list(cache_dir.glob("*.json")))
                # Changing a file's timestamp without touching the directory
                # mtime would change a fresh scan's manifest; an identical
                # result proves the cached signature was reused.
                os.utime(first, (150, 150))
                cached = common.path_signature(root)
                self.assertEqual(before, cached)
                # Adding a file changes the directory mtime and must
                # invalidate the cached entry immediately.
                (root / "slice_002.dcm").write_bytes(b"bb")
                grown = common.path_signature(root)
                self.assertEqual(grown["files"], 2)
                self.assertNotEqual(grown["manifest_sha256"], before["manifest_sha256"])
                # An expired entry is rescanned even when nothing changed.
                for entry in cache_dir.glob("*.json"):
                    payload = json.loads(entry.read_text(encoding="utf-8"))
                    payload["verified_at_epoch"] = 0
                    entry.write_text(json.dumps(payload), encoding="utf-8")
                revalidated = common.path_signature(root)
                self.assertEqual(revalidated, grown)

    def test_directory_signature_cache_off_switch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "series"
            root.mkdir()
            (root / "slice_001.dcm").write_bytes(b"aa")
            with mock.patch.dict(
                os.environ, {"MIMICS_PATH_SIGNATURE_CACHE_DIR": "off"}
            ):
                first = common.path_signature(root)
                second = common.path_signature(root)
                self.assertEqual(first, second)


        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "image.nii.gz"
            source.write_bytes(b"first")
            os.utime(source, (100, 100))
            before = common.path_signature(source)
            replacement = root / "replacement.nii.gz"
            replacement.write_bytes(b"other")
            os.utime(replacement, (100, 100))
            os.replace(replacement, source)
            os.utime(source, (100, 100))
            after = common.path_signature(source)
            self.assertEqual(before["size"], after["size"])
            self.assertEqual(before["mtime_ns"], after["mtime_ns"])
            self.assertNotEqual(before, after)

    def test_normalize_multiclass_request(self):
        request = common.normalize_request(
            {
                "operation": "train",
                "task_name": "Abdomen",
                "workspace": "/tmp/nnunet-test",
                "labels": [
                    {"name": "Liver", "id": 1, "aliases": "liver;Liver Seg"},
                    {"name": "Spleen", "id": 2, "aliases": ["spleen"]},
                ],
                "configuration": "3d_fullres",
                "patch_size": [96, 128, 128],
            }
        )
        self.assertEqual(request["labels"][0]["aliases"], ["Liver", "Liver Seg"])
        self.assertEqual(request["labels"][0]["source_mode"], "alternatives")
        self.assertEqual(request["patch_size"], [96, 128, 128])

    def test_grouped_label_preserves_union_source_mode(self):
        labels = common.normalize_labels(
            [
                {
                    "name": "head_region",
                    "id": 1,
                    "aliases": ["brain", "skull"],
                    "source_mode": "union",
                }
            ]
        )
        self.assertEqual(labels[0]["source_mode"], "union")

    def test_missing_labels_are_required_by_default(self):
        request = common.normalize_request(
            {
                "operation": "train",
                "task_name": "Multi organ",
                "workspace": "/tmp/nnunet-test",
                "labels": [{"name": "Liver", "id": 1}],
            }
        )
        self.assertEqual(request["missing_label_policy"], "require_all")

    def test_duplicate_label_id_fails(self):
        with self.assertRaisesRegex(ValueError, "Duplicate label ID"):
            common.normalize_labels(
                [
                    {"name": "A", "id": 1},
                    {"name": "B", "id": 1},
                ]
            )

    def test_label_ids_must_be_consecutive(self):
        with self.assertRaisesRegex(ValueError, "consecutive"):
            common.normalize_labels(
                [
                    {"name": "A", "id": 1},
                    {"name": "B", "id": 3},
                ]
            )

    def test_alias_cannot_map_to_two_output_labels(self):
        with self.assertRaisesRegex(ValueError, "assigned to both"):
            common.normalize_labels(
                [
                    {"name": "Left kidney", "id": 1, "aliases": ["kidney"]},
                    {"name": "Right kidney", "id": 2, "aliases": ["Kidney"]},
                ]
            )

    def test_2d_patch_has_two_dimensions(self):
        with self.assertRaisesRegex(ValueError, "2 positive values"):
            common.normalize_request(
                {
                    "operation": "train",
                    "task_name": "A",
                    "workspace": "/tmp/a",
                    "labels": [{"name": "A", "id": 1}],
                    "configuration": "2d",
                    "patch_size": [64, 64, 32],
                }
            )

    def test_2d_manual_spacing_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not supported"):
            common.normalize_request(
                {
                    "operation": "train",
                    "task_name": "A",
                    "workspace": "/tmp/a",
                    "labels": [{"name": "A", "id": 1}],
                    "configuration": "2d",
                    "spacing": [1.0, 1.0, 1.0],
                }
            )

    def test_continue_and_pretrained_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            common.normalize_request(
                {
                    "operation": "train",
                    "task_name": "A",
                    "workspace": "/tmp/a",
                    "labels": [{"name": "A", "id": 1}],
                    "continue_training": True,
                    "pretrained_weights": "/tmp/model.pth",
                }
            )

    def test_dataset_id_suggestion_skips_existing_model(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            model_dir = workspace / "models" / "liver" / "model"
            model_dir.mkdir(parents=True)
            common.write_json_atomic(
                model_dir / "mimics_model_manifest.json",
                {
                    "schema_version": common.MODEL_SCHEMA_VERSION,
                    "model_id": "model",
                    "dataset_id": 701,
                    "model_dir": str(model_dir),
                    "manifest_path": str(
                        model_dir / "mimics_model_manifest.json"
                    ),
                },
            )
            self.assertEqual(common.suggest_dataset_id(workspace), 702)

    def test_official_trainer_does_not_claim_custom_epochs(self):
        request = common.normalize_request(
            {
                "operation": "train",
                "task_name": "A",
                "workspace": "/tmp/a",
                "labels": [{"name": "A", "id": 1}],
                "trainer": "nnUNetTrainer",
                "epochs": 1000,
            }
        )
        self.assertEqual(request["trainer"], "nnUNetTrainer")

    def test_numeric_fold_requires_validation_data(self):
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            common.normalize_request(
                {
                    "operation": "train",
                    "task_name": "A",
                    "workspace": "/tmp/a",
                    "labels": [{"name": "A", "id": 1}],
                    "fold": "0",
                    "validation_fraction": 0.0,
                }
            )

    def test_local_gpu_ids_match_gpu_count(self):
        with self.assertRaisesRegex(ValueError, "GPU count"):
            common.normalize_request(
                {
                    "operation": "train",
                    "task_name": "A",
                    "workspace": "/tmp/a",
                    "labels": [{"name": "A", "id": 1}],
                    "num_gpus": 2,
                    "gpu_id": "0",
                }
            )


class StandaloneWorkflowTests(unittest.TestCase):
    def _write_config(self, root):
        model_map = root / "ModelMap.toml"
        model_map.write_text("[Multi]\nliver = 1\nspleen = 2\n", encoding="utf-8")
        config = root / "Config.toml"
        config.write_text(
            """
[COMMON]
modality = "CT"

[PATHS]
labeled_path = "labeled"
labeled_dataset = ["source"]
train_path = "train"
train_project = "project"
nnUNet_raw = "nnUNet_raw"
nnUNet_preprocessed = "nnUNet_preprocessed"
nnUNet_results = "nnUNet_results"

[MODEL]
train_dataset = ["Dataset701_Multi"]
segment_model_file = "ModelMap.toml"
segment_list_name = ["Multi"]

[GPU]
gpu_id = 0

[PREPROCESS]
configuration = "3d_fullres"
num_processes = 2
spacing = []
patch_size = []
batch_size = 0
orientation = ""
reorientaion = "NibabelIOWithReorient"

[TRAIN]
epoch = 7
fold = 0
trainer = "nnUNetTrainerNoMirroring"
plans = "nnUNetPlans"

[PREDICT]
input_path = ""
output_path = ""
disable_tta = true
enable_stats = false

[EVALUATION]
run_aggregation = true
""".strip()
            + "\n",
            encoding="utf-8",
        )
        return config

    def test_epoch_setting_resolves_to_configurable_equivalent(self):
        fake_torch = types.ModuleType("torch")
        fake_torch.device = type("device", (), {})
        previous_torch = sys.modules.get("torch")
        previous_action = sys.modules.pop("Action3_Train", None)
        sys.modules["torch"] = fake_torch
        try:
            action = importlib.import_module("Action3_Train")
            self.assertEqual(
                action.resolve_epoch_trainer("nnUNetTrainerNoMirroring", 500),
                ("MimicsNNUNetTrainerNoMirroring", 500),
            )
            with self.assertRaisesRegex(ValueError, "silently ignored"):
                action.resolve_epoch_trainer("UnverifiedCustomTrainer", 500)
        finally:
            sys.modules.pop("Action3_Train", None)
            if previous_action is not None:
                sys.modules["Action3_Train"] = previous_action
            if previous_torch is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = previous_torch

    def test_default_workflow_includes_conversion(self):
        self.assertEqual(standalone.DEFAULT_STAGES[0], "convert")

    def test_pre_cancel_is_terminal_without_starting_a_worker(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self._write_config(root)
            status = root / "status.json"
            control = root / "control.json"
            standalone._write_json_atomic(control, {"action": "cancel"})
            with mock.patch.object(standalone.subprocess, "Popen") as popen:
                returncode = standalone.run_workflow_controller(
                    config,
                    stages=["train"],
                    status_file=status,
                    control_file=control,
                )
            self.assertEqual(returncode, 130)
            popen.assert_not_called()
            payload = standalone._read_json(status)
            self.assertEqual(payload["status"], "cancelled")
            self.assertIsNone(payload["worker_pid"])

    def test_cancel_command_publishes_control_request(self):
        with tempfile.TemporaryDirectory() as temp:
            control = Path(temp) / "control.json"
            returncode = standalone.main(
                ["cancel", "--control-file", str(control)]
            )
            self.assertEqual(returncode, 0)
            payload = standalone._read_json(control)
            self.assertEqual(payload["action"], "cancel")
            self.assertGreater(payload["requested_at_epoch"], 0)
            self.assertEqual(payload["requested_by_pid"], os.getpid())

    def test_invalid_epoch_trainer_fails_before_worker_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self._write_config(root)
            content = config.read_text(encoding="utf-8").replace(
                'trainer = "nnUNetTrainerNoMirroring"',
                'trainer = "UnverifiedCustomTrainer"',
            )
            config.write_text(content, encoding="utf-8")
            status = root / "status.json"
            with mock.patch.object(standalone.subprocess, "Popen") as popen:
                with self.assertRaisesRegex(ValueError, "silently ignored"):
                    standalone.run_workflow_controller(
                        config,
                        stages=["train"],
                        status_file=status,
                    )
            popen.assert_not_called()
            payload = standalone._read_json(status)
            self.assertEqual(payload["status"], "failed")
            self.assertEqual(payload["phase"], "configuration")
            self.assertEqual(payload["error_category"], "configuration_invalid")

    def test_legacy_environment_helper_does_not_touch_shell_files(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.dict(
            os.environ, {"HOME": temp, "SHELL": "/bin/bash"}, clear=False
        ):
            standalone_environment.add_to_user_shell_config("nnUNet_raw", "/tmp/raw")
            self.assertEqual(os.environ["nnUNet_raw"], "/tmp/raw")
            self.assertFalse((Path(temp) / ".bashrc").exists())


class DiscoveryTests(unittest.TestCase):
    def test_label_lookup_is_case_insensitive_and_suffix_aware(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            seg = root / "case_1" / "segmentations"
            seg.mkdir(parents=True)
            expected = seg / "LIVER.SEG.NII.GZ"
            expected.write_bytes(b"label")
            found = pipeline._find_label_file(root, "case_1", ["liver"])
            self.assertEqual(found, expected.resolve())

    def test_ambiguous_alias_match_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            seg = root / "case_1" / "segmentations"
            seg.mkdir(parents=True)
            (seg / "liver.nii.gz").write_bytes(b"a")
            (seg / "hepatic.nii.gz").write_bytes(b"b")
            with self.assertRaisesRegex(RuntimeError, "more than one"):
                pipeline._find_label_file(root, "case_1", ["liver", "hepatic"])

    def test_raw_dataset_and_split_are_deterministic(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = []
            for index in range(5):
                image = root / "prepared" / str(index) / "image.nii.gz"
                label = root / "prepared" / str(index) / "label.nii.gz"
                image.parent.mkdir(parents=True)
                image.write_bytes(("image{}".format(index)).encode())
                label.write_bytes(("label{}".format(index)).encode())
                rows.append(
                    {
                        "case_id": "case{}".format(index),
                        "image": image,
                        "label": label,
                        "fingerprint": str(index),
                    }
                )
            request = common.normalize_request(
                {
                    "operation": "train",
                    "task_name": "Liver",
                    "task_id": "liver",
                    "workspace": root / "workspace",
                    "labels": [{"name": "liver", "id": 1}],
                    "validation_fraction": 0.2,
                    "split_seed": 42,
                }
            )
            dataset, fingerprint = pipeline._materialize_nnunet_raw(
                rows, request, root / "raw", root / "preprocessed"
            )
            self.assertTrue((dataset / "dataset.json").is_file())
            split = common.read_json(root / "preprocessed" / dataset.name / "splits_final.json")
            self.assertEqual(len(split), 5)
            self.assertEqual(len(split[0]["train"]), 4)
            self.assertEqual(len(split[0]["val"]), 1)
            self.assertEqual(len(fingerprint), 64)

    def test_dataset_id_conflict_is_rejected_before_nnunet_resolution(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            roots = {
                "raw": root / "raw",
                "preprocessed": root / "preprocessed",
                "results": root / "results",
            }
            for value in roots.values():
                value.mkdir()
            (roots["preprocessed"] / "Dataset701_other_task").mkdir()
            request = common.normalize_request(
                {
                    "operation": "train",
                    "task_name": "Liver",
                    "task_id": "liver",
                    "workspace": root / "workspace",
                    "labels": [{"name": "Liver", "id": 1}],
                    "dataset_id": 701,
                }
            )
            with self.assertRaisesRegex(RuntimeError, "already used"):
                pipeline._assert_dataset_id_available(roots, request)


@unittest.skipUnless(
    importlib.util.find_spec("numpy") and importlib.util.find_spec("nibabel"),
    "numpy and nibabel are required for spatial tests",
)
class SpatialContractTests(unittest.TestCase):
    def test_model_compatibility_allows_different_patient_grid(self):
        import nibabel as nib
        import numpy as np

        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / "image.nii.gz"
            affine = np.diag([1.2, 1.2, 2.5, 1.0])
            affine[:3, 3] = [120.0, -80.0, 40.0]
            nib.save(
                nib.Nifti1Image(
                    np.zeros((48, 56, 32), dtype=np.float32), affine
                ),
                str(image_path),
            )
            manifest = {
                "task_id": "liver",
                "modality": "CT",
                "training_data_profile": {
                    "channel_count": 1,
                    "spatial_dimensions": 3,
                    "spacing_mm_sorted": {
                        "min": [0.8, 0.8, 1.0],
                        "max": [2.0, 2.0, 4.0],
                    },
                    "field_of_view_mm_sorted": {
                        "min": [40.0, 40.0, 40.0],
                        "max": [500.0, 500.0, 500.0],
                    },
                },
            }
            warnings = pipeline.validate_model_input_compatibility(
                image_path,
                manifest,
                {"task_id": "liver", "source_modality": "CT"},
            )
            self.assertEqual(warnings, [])

    def test_model_compatibility_rejects_modality_mismatch(self):
        import nibabel as nib
        import numpy as np

        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / "image.nii.gz"
            nib.save(
                nib.Nifti1Image(
                    np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)
                ),
                str(image_path),
            )
            with self.assertRaisesRegex(RuntimeError, "trained for CT"):
                pipeline.validate_model_input_compatibility(
                    image_path,
                    {"task_id": "liver", "modality": "CT"},
                    {"task_id": "liver", "source_modality": "MR"},
                )

    def test_bridge_preserves_only_present_multiclass_value(self):
        import nibabel as nib
        import numpy as np
        from mimics_bridge import do_prepare_masks_for_grid

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prediction = root / "prediction.nii.gz"
            data = np.zeros((4, 4, 4), dtype=np.uint8)
            data[1, 1, 1] = 2
            nib.save(nib.Nifti1Image(data, np.eye(4)), str(prediction))
            result = do_prepare_masks_for_grid(
                {
                    "masks": [{"name": "prediction", "mask_path": str(prediction)}],
                    "buffers_out": str(root / "buffers"),
                    "target_shape": [4, 4, 4],
                    "target_voxel_to_ras_matrix": np.eye(4).tolist(),
                    "axes": [0, 1, 2],
                    "flips": [False, False, False],
                }
            )
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["masks"][0]["label_value"], 2)

    def test_discrete_orientation_change_preserves_label_voxel(self):
        import nibabel as nib
        import numpy as np

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image_data = np.zeros((4, 5, 6), dtype=np.float32)
            image_affine = np.diag([1.0, 1.0, 2.0, 1.0])
            image_path = root / "image.nii.gz"
            nib.save(nib.Nifti1Image(image_data, image_affine), str(image_path))
            source_mask = np.zeros(image_data.shape, dtype=np.uint8)
            source_mask[1, 2, 3] = 1
            label_data = np.flip(source_mask, axis=0).copy()
            label_affine = image_affine @ np.asarray(
                [
                    [-1.0, 0.0, 0.0, 3.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            label_path = root / "label.nii.gz"
            nib.save(nib.Nifti1Image(label_data, label_affine), str(label_path))
            request = common.normalize_request(
                {
                    "operation": "train",
                    "task_name": "Liver",
                    "workspace": root / "workspace",
                    "labels": [{"name": "Liver", "id": 1}],
                }
            )
            row = pipeline._materialize_case(
                "case", image_path, [[label_path]], request, root / "cache"
            )
            output = np.asanyarray(nib.load(str(row["label"])).dataobj)
            self.assertEqual(int(output[1, 2, 3]), 1)
            self.assertEqual(int(np.count_nonzero(output)), 1)

    def test_case_cache_revalidates_cached_image_label_grid(self):
        import nibabel as nib
        import numpy as np

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image_path = root / "image.nii.gz"
            label_path = root / "label.nii.gz"
            image = np.zeros((4, 4, 4), dtype=np.float32)
            label = np.zeros((4, 4, 4), dtype=np.uint8)
            label[1, 1, 1] = 1
            nib.save(nib.Nifti1Image(image, np.eye(4)), str(image_path))
            nib.save(nib.Nifti1Image(label, np.eye(4)), str(label_path))
            request = common.normalize_request(
                {
                    "operation": "train",
                    "task_name": "Liver",
                    "workspace": root / "workspace",
                    "labels": [{"name": "Liver", "id": 1}],
                }
            )
            first = pipeline._materialize_case(
                "case", image_path, [[label_path]], request, root / "cache"
            )
            self.assertFalse(first["cache_hit"])
            cached_label = Path(first["label"])
            wrong_affine = np.eye(4)
            wrong_affine[0, 3] = 25.0
            nib.save(nib.Nifti1Image(label, wrong_affine), str(cached_label))
            second = pipeline._materialize_case(
                "case", image_path, [[label_path]], request, root / "cache"
            )
            self.assertFalse(second["cache_hit"])
            repaired = nib.load(str(second["label"]))
            self.assertTrue(np.allclose(repaired.affine, np.eye(4)))

    def test_multiclass_overlap_fails_by_default(self):
        import nibabel as nib
        import numpy as np

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            affine = np.eye(4)
            image = root / "image.nii.gz"
            nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), affine), str(image))
            labels = []
            for name in ("a", "b"):
                path = root / (name + ".nii.gz")
                data = np.zeros((4, 4, 4), dtype=np.uint8)
                data[1, 1, 1] = 1
                nib.save(nib.Nifti1Image(data, affine), str(path))
                labels.append(path)
            request = common.normalize_request(
                {
                    "operation": "train",
                    "task_name": "AB",
                    "workspace": root / "workspace",
                    "labels": [
                        {"name": "A", "id": 1},
                        {"name": "B", "id": 2},
                    ],
                    "overlap_policy": "fail",
                }
            )
            with self.assertRaisesRegex(RuntimeError, "overlapping voxels"):
                pipeline._materialize_case(
                    "case", image, [[labels[0]], [labels[1]]], request, root / "cache"
                )

    def test_grouped_source_masks_are_unioned_into_one_output_class(self):
        import nibabel as nib
        import numpy as np

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            affine = np.eye(4)
            image = root / "image.nii.gz"
            nib.save(
                nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), affine),
                str(image),
            )
            brain = root / "brain.nii.gz"
            brain_data = np.zeros((4, 4, 4), dtype=np.uint8)
            brain_data[1, 1, 1] = 1
            nib.save(nib.Nifti1Image(brain_data, affine), str(brain))
            skull = root / "skull.nii.gz"
            skull_data = np.zeros((4, 4, 4), dtype=np.uint8)
            skull_data[1, 2, 2] = 1
            flipped_affine = affine @ np.asarray(
                [
                    [-1.0, 0.0, 0.0, 3.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            nib.save(nib.Nifti1Image(skull_data, flipped_affine), str(skull))
            request = common.normalize_request(
                {
                    "operation": "train",
                    "task_name": "Head coarse",
                    "workspace": root / "workspace",
                    "labels": [
                        {
                            "name": "head",
                            "id": 1,
                            "aliases": ["brain", "skull"],
                            "source_mode": "union",
                        }
                    ],
                }
            )
            row = pipeline._materialize_case(
                "case", image, [[brain, skull]], request, root / "cache"
            )
            output = np.asanyarray(nib.load(str(row["label"])).dataobj)
            self.assertEqual(int(output[1, 1, 1]), 1)
            self.assertEqual(int(output[2, 2, 2]), 1)
            self.assertEqual(int(np.count_nonzero(output)), 2)

    def test_inference_rejects_wrong_output_affine(self):
        import nibabel as nib
        import numpy as np

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = root / "workspace"
            job = root / "job"
            job.mkdir()
            image = root / "image.nii.gz"
            nib.save(
                nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)),
                str(image),
            )
            model = root / "model"
            model.mkdir()
            manifest = model / "mimics_model_manifest.json"
            common.write_json_atomic(
                manifest,
                {
                    "schema_version": common.MODEL_SCHEMA_VERSION,
                    "model_id": "model",
                    "model_dir": str(model),
                    "labels": [{"name": "A", "id": 1}],
                },
            )
            common.write_json_atomic(
                job / "request.json",
                {
                    "operation": "infer",
                    "job_id": "infer",
                    "workspace": str(workspace),
                    "image_path": str(image),
                    "model_manifest": str(manifest),
                    "output_path": str(job / "prediction.nii.gz"),
                },
            )
            common.write_json_atomic(job / "control.json", {"action": "run"})

            def fake_worker(*args, **kwargs):
                wrong = np.eye(4)
                wrong[0, 3] = 10.0
                nib.save(
                    nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.uint8), wrong),
                    str(job / "prediction.nii.gz"),
                )
                return {"predicted": True}

            with mock.patch.object(pipeline, "_spawn_worker", side_effect=fake_worker), mock.patch.object(
                pipeline, "_acquire_local_gpu", return_value=None
            ):
                returncode = pipeline.run_inference(job)
            self.assertEqual(returncode, 1)
            status = common.read_json(job / "status.json")
            self.assertEqual(status["status"], "failed")
            self.assertIn("grid does not match", status["error"])

    def test_inference_rejects_source_changed_since_mimics_import(self):
        import nibabel as nib
        import numpy as np

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            job = root / "job"
            job.mkdir()
            image = root / "image.nii.gz"
            nib.save(
                nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)),
                str(image),
            )
            model = root / "model"
            model.mkdir()
            manifest = model / "mimics_model_manifest.json"
            common.write_json_atomic(
                manifest,
                {
                    "schema_version": common.MODEL_SCHEMA_VERSION,
                    "model_id": "model",
                    "model_dir": str(model),
                    "labels": [{"name": "A", "id": 1}],
                },
            )
            expected = np.eye(4)
            expected[0, 3] = 15.0
            common.write_json_atomic(
                job / "request.json",
                {
                    "operation": "infer",
                    "job_id": "infer",
                    "workspace": str(root / "workspace"),
                    "image_path": str(image),
                    "model_manifest": str(manifest),
                    "output_path": str(job / "prediction.nii.gz"),
                    "source_geometry_expected": {
                        "source_shape": [4, 4, 4],
                        "source_voxel_to_ras_matrix": expected.tolist(),
                    },
                },
            )
            common.write_json_atomic(job / "control.json", {"action": "run"})
            with mock.patch.object(pipeline, "_spawn_worker") as worker:
                returncode = pipeline.run_inference(job)
            self.assertEqual(returncode, 1)
            worker.assert_not_called()
            status = common.read_json(job / "status.json")
            self.assertIn("no longer matches", status["error"])

    def test_remote_inference_rejects_changed_source_before_transfer(self):
        import nibabel as nib
        import numpy as np

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            job = root / "job"
            bundle = root / "bundle"
            job.mkdir()
            bundle.mkdir()
            source = root / "image.nii.gz"
            nib.save(
                nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)),
                str(source),
            )
            model = root / "model"
            (model / "fold_0").mkdir(parents=True)
            for name in ("plans.json", "dataset.json"):
                (model / name).write_text("{}", encoding="utf-8")
            (model / "fold_0" / "checkpoint_final.pth").write_bytes(b"weights")
            manifest = model / "mimics_model_manifest.json"
            common.write_json_atomic(
                manifest,
                {
                    "schema_version": common.MODEL_SCHEMA_VERSION,
                    "model_id": "model",
                    "model_dir": str(model),
                    "manifest_path": str(manifest),
                    "folds": ["0"],
                },
            )
            wrong_affine = np.eye(4)
            wrong_affine[1, 3] = 9.0
            request_path = job / "request.json"
            common.write_json_atomic(
                request_path,
                {
                    "operation": "infer",
                    "job_id": "infer",
                    "workspace": str(root / "workspace"),
                    "image_path": str(source),
                    "model_manifest": str(manifest),
                    "source_geometry_expected": {
                        "source_shape": [4, 4, 4],
                        "source_voxel_to_ras_matrix": wrong_affine.tolist(),
                    },
                },
            )
            with self.assertRaisesRegex(RuntimeError, "no longer matches"):
                remote._prepare_nnunet_infer(
                    {
                        "job_dir": str(job),
                        "request_path": str(request_path),
                        "status_path": str(job / "status.json"),
                    },
                    bundle,
                )

    def test_remote_inference_cache_namespace_contains_dataset_id(self):
        import nibabel as nib
        import numpy as np

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            job = root / "job"
            bundle = root / "bundle"
            job.mkdir()
            bundle.mkdir()
            source = root / "image.nii.gz"
            nib.save(
                nib.Nifti1Image(
                    np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)
                ),
                str(source),
            )
            model = root / "model"
            model.mkdir()
            (model / "checkpoint_final.pth").write_bytes(b"weights")
            manifest_path = model / "mimics_model_manifest.json"
            common.write_json_atomic(
                manifest_path,
                {
                    "schema_version": common.MODEL_SCHEMA_VERSION,
                    "model_id": "model",
                    "task_id": "liver",
                    "dataset_id": 712,
                    "model_dir": str(model),
                    "manifest_path": str(manifest_path),
                    "modality": "CT",
                    "training_data_profile": {
                        "channel_count": 1,
                        "spatial_dimensions": 3,
                    },
                },
            )
            request_path = job / "request.json"
            common.write_json_atomic(
                request_path,
                {
                    "operation": "infer",
                    "job_id": "infer",
                    "workspace": str(root / "workspace"),
                    "task_id": "liver",
                    "source_modality": "CT",
                    "image_path": str(source),
                    "model_manifest": str(manifest_path),
                },
            )
            remote._prepare_nnunet_infer(
                {
                    "job_dir": str(job),
                    "request_path": str(request_path),
                    "status_path": str(job / "status.json"),
                },
                bundle,
            )
            payload = common.read_json(bundle / "remote_request.json")
            raw = payload["pipeline_request"]["runtime_roots"]["raw"]
            self.assertIn("/nnunet/inference/Dataset712_", raw)


class TestSweepExpiredJobs(unittest.TestCase):
    def _make_job(self, tmp, name, status, completed_at=None, request=None):
        jobs = Path(tmp) / "jobs" / name
        jobs.mkdir(parents=True)
        from nnunet_common import write_json_atomic

        payload = {"status": status, "job_id": name}
        if completed_at is not None:
            payload["completed_at_epoch"] = completed_at
        write_json_atomic(jobs / "status.json", payload)
        if request is not None:
            write_json_atomic(jobs / "request.json", request)
        (jobs / "payload.bin").write_bytes(b"x" * 100)
        return jobs

    def test_old_terminal_jobs_are_pruned_keeping_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._make_job(tmp, "train_old", "completed", completed_at=time.time() - 40 * 86400)
            result = pipeline._sweep_expired_jobs(tmp, {"job_retention_days": 30})
            self.assertEqual(result["removed_jobs"], ["train_old"])
            self.assertTrue((Path(tmp) / "jobs" / "train_old" / "status.json").is_file())
            self.assertFalse((Path(tmp) / "jobs" / "train_old" / "payload.bin").exists())

    def test_recent_jobs_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._make_job(tmp, "train_new", "completed", completed_at=time.time())
            result = pipeline._sweep_expired_jobs(tmp, {"job_retention_days": 30})
            self.assertEqual(result["removed_jobs"], [])
            self.assertTrue((Path(tmp) / "jobs" / "train_new" / "payload.bin").exists())

    def test_live_jobs_never_pruned(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._make_job(tmp, "train_live", "training")
            old = time.time() - 40 * 86400
            os.utime(str(Path(tmp) / "jobs" / "train_live" / "status.json"), (old, old))
            result = pipeline._sweep_expired_jobs(tmp, {"job_retention_days": 30})
            self.assertEqual(result["removed_jobs"], [])
            self.assertTrue((Path(tmp) / "jobs" / "train_live" / "payload.bin").exists())

    def test_reattaching_remote_jobs_never_pruned(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = time.time() - 40 * 86400
            self._make_job(tmp, "remote_job", "reattaching", completed_at=old)
            result = pipeline._sweep_expired_jobs(tmp, {"job_retention_days": 30})
            self.assertEqual(result["removed_jobs"], [])
            self.assertTrue((Path(tmp) / "jobs" / "remote_job" / "payload.bin").exists())

    def test_zero_retention_disables(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._make_job(tmp, "train_old", "completed", completed_at=0.0)
            old = time.time() - 40 * 86400
            os.utime(str(Path(tmp) / "jobs" / "train_old" / "status.json"), (old, old))
            result = pipeline._sweep_expired_jobs(tmp, {"job_retention_days": 0})
            self.assertEqual(result["removed_jobs"], [])
            self.assertTrue((Path(tmp) / "jobs" / "train_old" / "payload.bin").exists())

    def test_models_and_cache_untouched(self):
        """Registered models and caches must never be swept.

        models/ holds the user's trained-model assets and cache/ the
        rebuildable source-grid cache; the sweep's contract is jobs/ only.
        """
        with tempfile.TemporaryDirectory() as tmp:
            self._make_job(tmp, "train_old", "completed", completed_at=time.time() - 40 * 86400)
            models = Path(tmp) / "models" / "task701" / "model_x"
            models.mkdir(parents=True)
            (models / "model.safetensors").write_bytes(b"weights")
            cache = Path(tmp) / "cache" / "source_grid" / "task701"
            cache.mkdir(parents=True)
            (cache / "case.bin").write_bytes(b"cache")
            old = time.time() - 40 * 86400
            os.utime(str(models / "model.safetensors"), (old, old))
            os.utime(str(cache / "case.bin"), (old, old))
            result = pipeline._sweep_expired_jobs(tmp, {"job_retention_days": 30})
            self.assertEqual(result["removed_jobs"], ["train_old"])
            self.assertTrue((models / "model.safetensors").is_file())
            self.assertTrue((cache / "case.bin").is_file())


class TestSweepRuntimeAndCache(unittest.TestCase):
    """Retention for runtime Dataset dirs and cache/source_grid (A11)."""

    CONFIG = {"runtime_retention_days": 30, "source_grid_cache_retention_days": 30}
    OLD = time.time() - 40 * 86400

    def _make_dataset(self, tmp, folder, name):
        dataset = Path(tmp) / "runtime" / folder / name
        dataset.mkdir(parents=True)
        (dataset / "data.bin").write_bytes(b"x" * 100)
        os.utime(str(dataset), (self.OLD, self.OLD))
        return dataset

    def _make_cache_case(self, tmp, task, case):
        case_dir = Path(tmp) / "cache" / "source_grid" / task / case
        case_dir.mkdir(parents=True)
        (case_dir / "image.nii.gz").write_bytes(b"img")
        os.utime(str(case_dir), (self.OLD, self.OLD))
        return case_dir

    def _make_registry_model(self, tmp, dataset_name, model_id="m1"):
        from nnunet_common import write_json_atomic

        model_dir = Path(tmp) / "models" / "task" / model_id
        model_dir.mkdir(parents=True)
        manifest = {
            "model_id": model_id,
            "dataset_name": dataset_name,
            "dataset_id": 701,
            "task_id": "task",
            "model_dir": str(model_dir),
            "manifest_path": str(model_dir / "mimics_model_manifest.json"),
        }
        write_json_atomic(
            model_dir / "mimics_model_manifest.json", manifest
        )
        write_json_atomic(
            Path(tmp) / "model_registry.json", {"models": [manifest]}
        )
        return model_dir

    def _make_job(self, tmp, name, status, request=None):
        jobs = Path(tmp) / "jobs" / name
        jobs.mkdir(parents=True)
        from nnunet_common import write_json_atomic

        write_json_atomic(jobs / "status.json", {"status": status, "job_id": name})
        if request is not None:
            write_json_atomic(jobs / "request.json", request)
        (jobs / "payload.bin").write_bytes(b"x")
        return jobs

    def test_expired_unreferenced_dataset_removed_in_all_runtime_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            for folder in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
                self._make_dataset(tmp, folder, "Dataset701_oldtask")
            self._make_job(tmp, "train_done", "completed")
            result = pipeline._sweep_runtime_and_cache(tmp, self.CONFIG)
            removed = result["runtime"]["removed_datasets"]
            self.assertEqual(len(removed), 3, removed)
            for folder in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
                self.assertFalse(
                    (Path(tmp) / "runtime" / folder / "Dataset701_oldtask").exists()
                )

    def test_dataset_referenced_by_registered_model_is_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._make_dataset(tmp, "nnUNet_results", "Dataset701_task")
            self._make_registry_model(tmp, "Dataset701_task")
            self._make_job(tmp, "train_done", "completed")
            result = pipeline._sweep_runtime_and_cache(tmp, self.CONFIG)
            self.assertEqual(result["runtime"]["removed_datasets"], [])
            self.assertTrue(
                (Path(tmp) / "runtime" / "nnUNet_results" / "Dataset701_task").is_dir()
            )

    def test_dataset_of_running_job_is_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._make_dataset(tmp, "nnUNet_raw", "Dataset702_live")
            self._make_job(
                tmp,
                "train_live",
                "training",
                request={"dataset_id": 702, "task_id": "live"},
            )
            result = pipeline._sweep_runtime_and_cache(tmp, self.CONFIG)
            self.assertEqual(result["runtime"]["removed_datasets"], [])
            self.assertTrue(
                (Path(tmp) / "runtime" / "nnUNet_raw" / "Dataset702_live").is_dir()
            )

    def test_recent_dataset_is_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "runtime" / "nnUNet_raw" / "Dataset703_fresh"
            dataset.mkdir(parents=True)
            (dataset / "data.bin").write_bytes(b"x")
            self._make_job(tmp, "train_done", "completed")
            result = pipeline._sweep_runtime_and_cache(tmp, self.CONFIG)
            self.assertEqual(result["runtime"]["removed_datasets"], [])
            self.assertTrue(dataset.is_dir())

    def test_zero_retention_disables_both_sweeps(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._make_dataset(tmp, "nnUNet_raw", "Dataset701_oldtask")
            case = self._make_cache_case(tmp, "oldtask", "case1")
            self._make_job(tmp, "train_done", "completed")
            result = pipeline._sweep_runtime_and_cache(
                tmp, {"runtime_retention_days": 0, "source_grid_cache_retention_days": 0}
            )
            self.assertEqual(result["runtime"]["removed_datasets"], [])
            self.assertEqual(result["source_grid_cache"]["removed_cases"], [])
            self.assertTrue(dataset.is_dir())
            self.assertTrue(case.is_dir())

    def test_expired_cache_case_removed_and_empty_task_dir_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = self._make_cache_case(tmp, "oldtask", "case1")
            fresh = self._make_cache_case(tmp, "oldtask", "case2")
            os.utime(str(fresh), (time.time(), time.time()))
            self._make_job(tmp, "train_done", "completed")
            result = pipeline._sweep_runtime_and_cache(tmp, self.CONFIG)
            self.assertEqual(
                result["source_grid_cache"]["removed_cases"], ["oldtask/case1"]
            )
            self.assertFalse(case.exists())
            self.assertTrue(fresh.is_dir())
            self.assertTrue(
                (Path(tmp) / "cache" / "source_grid" / "oldtask").is_dir()
            )

    def test_cache_task_tree_of_running_job_is_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = self._make_cache_case(tmp, "livetask", "case1")
            self._make_job(
                tmp,
                "train_live",
                "training",
                request={"dataset_id": 702, "task_id": "livetask"},
            )
            result = pipeline._sweep_runtime_and_cache(tmp, self.CONFIG)
            self.assertEqual(result["source_grid_cache"]["removed_cases"], [])
            self.assertTrue(case.is_dir())

    def test_models_dir_never_touched(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "models" / "task" / "model_x"
            model_dir.mkdir(parents=True)
            (model_dir / "checkpoint.bin").write_bytes(b"w")
            os.utime(str(model_dir), (self.OLD, self.OLD))
            self._make_job(tmp, "train_done", "completed")
            pipeline._sweep_runtime_and_cache(tmp, self.CONFIG)
            self.assertTrue((model_dir / "checkpoint.bin").is_file())

    def test_registry_row_without_dataset_name_is_rebuilt(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._make_dataset(tmp, "nnUNet_results", "Dataset704_renamed")
            from nnunet_common import write_json_atomic

            model_dir = Path(tmp) / "models" / "renamed" / "m2"
            model_dir.mkdir(parents=True)
            manifest = {
                "model_id": "m2",
                "dataset_id": 704,
                "task_id": "renamed",
                "model_dir": str(model_dir),
                "manifest_path": str(model_dir / "mimics_model_manifest.json"),
            }
            write_json_atomic(model_dir / "mimics_model_manifest.json", manifest)
            write_json_atomic(
                Path(tmp) / "model_registry.json", {"models": [manifest]}
            )
            self._make_job(tmp, "train_done", "completed")
            result = pipeline._sweep_runtime_and_cache(tmp, self.CONFIG)
            self.assertEqual(result["runtime"]["removed_datasets"], [])
            self.assertTrue(dataset.is_dir())


class JobLifecycleTests(unittest.TestCase):
    def test_worker_environment_caps_blas_threads_every_stage(self):
        """Every stage worker must cap BLAS threads to one.

        The preprocess stage's spawn.Pool fingerprint workers each import
        torch/OpenBLAS and (uncapped) start one BLAS thread per core; under
        virtual-memory pressure OpenBLAS dies mid-derivation and nnU-Net's
        Pool silently respawns the corpse, hanging the stage at 0 CPU forever
        (A10). nnU-Net's own run_training.py entrypoint applies the same caps.
        """
        roots = {
            "raw": Path("W:/raw"),
            "preprocessed": Path("W:/preprocessed"),
            "results": Path("W:/results"),
        }
        for stage in ("preprocess", "train", "infer"):
            environment = pipeline._worker_environment(
                {"epochs": 2}, roots, stage)
            for blas_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                             "OPENBLAS_NUM_THREADS"):
                self.assertEqual(
                    environment.get(blas_var), "1",
                    "{} stage must cap {}".format(stage, blas_var))

    def test_gpu_lock_is_transferred_to_worker_pid(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            process = mock.Mock(pid=4321, returncode=0)
            resource_lock = mock.Mock()
            events = []

            def transfer(*_args, **_kwargs):
                spec = common.read_json(root / "train_spec.json")
                self.assertFalse(Path(spec["start_gate"]).exists())
                events.append("lock")
                return True

            def poll():
                common.write_json_atomic(
                    root / "train_result.json",
                    {"status": "ok", "result": {"trained": True}},
                )
                return 0

            original_write = pipeline.write_json_atomic

            def record_write(path, values):
                if str(path).endswith(".ready"):
                    events.append("gate")
                return original_write(path, values)

            process.poll.side_effect = poll
            resource_lock.update_pid.side_effect = transfer
            with mock.patch.object(
                pipeline.subprocess, "Popen", return_value=process
            ), mock.patch.object(
                pipeline, "write_json_atomic", side_effect=record_write
            ):
                result = pipeline._spawn_worker(
                    "train",
                    {},
                    {"job_id": "train_1"},
                    {
                        "raw": root / "raw",
                        "preprocessed": root / "preprocessed",
                        "results": root / "results",
                    },
                    root,
                    root / "status.json",
                    root / "control.json",
                    root / "job.log",
                    resource_lock=resource_lock,
                )
            self.assertTrue(result["trained"])
            self.assertEqual(events, ["lock", "gate"])
            resource_lock.update_pid.assert_called_once_with(
                4321,
                kind="nnunet_train",
                job_id="train_1",
                stop_path=str(root / "control.json"),
            )

    def test_stage_worker_never_starts_without_controller_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            missing_gate = Path(temp) / "missing.ready"
            with mock.patch.object(
                stage_worker.time, "time", side_effect=[0.0, 2.0]
            ), mock.patch.object(stage_worker.time, "sleep"):
                with self.assertRaisesRegex(RuntimeError, "before authorizing"):
                    stage_worker._wait_for_start_gate(
                        {
                            "start_gate": str(missing_gate),
                            "start_gate_timeout_seconds": 1,
                        }
                    )

    def test_gpu_lock_transfer_failure_terminates_worker(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            process = mock.Mock(pid=4321, returncode=None)
            process.wait.return_value = 0
            resource_lock = mock.Mock()
            resource_lock.update_pid.return_value = False
            with mock.patch.object(
                pipeline.subprocess, "Popen", return_value=process
            ), mock.patch(
                "tools.mimics_label_export.terminate_process_tree"
            ) as terminate, self.assertRaisesRegex(
                RuntimeError, "could not be transferred"
            ):
                pipeline._spawn_worker(
                    "train",
                    {},
                    {"job_id": "train_1"},
                    {
                        "raw": root / "raw",
                        "preprocessed": root / "preprocessed",
                        "results": root / "results",
                    },
                    root,
                    root / "status.json",
                    root / "control.json",
                    root / "job.log",
                    resource_lock=resource_lock,
                )
            terminate.assert_called_once_with(4321)

    def test_copied_model_is_discovered_without_old_registry(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            model_dir = workspace / "models" / "liver" / "copied_model"
            model_dir.mkdir(parents=True)
            common.write_json_atomic(
                model_dir / "mimics_model_manifest.json",
                {
                    "schema_version": common.MODEL_SCHEMA_VERSION,
                    "model_id": "copied_model",
                    "task_id": "liver",
                    "model_dir": "D:/old-machine/model",
                    "manifest_path": "D:/old-machine/model/mimics_model_manifest.json",
                },
            )
            models = common.load_models(workspace)
            self.assertEqual(len(models), 1)
            self.assertEqual(Path(models[0]["model_dir"]), model_dir.resolve())

    def test_dead_registry_rows_are_flagged_not_dropped(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            registry_path = workspace / "model_registry.json"
            common.write_json_atomic(
                registry_path,
                {
                    "models": [
                        {
                            "model_id": "dead_model",
                            "task_id": "liver",
                            "model_dir": str(
                                workspace / "missing" / "dead_model"
                            ),
                            "manifest_path": str(
                                workspace / "missing" / "dead_model"
                                / "mimics_model_manifest.json"
                            ),
                        }
                    ]
                },
            )
            # Default mode: dead rows are dropped (prediction surfaces keep
            # only usable models). Isolate from the workstation's global
            # registry, which may legitimately carry its own dead rows.
            with mock.patch.object(
                common,
                "model_registry_paths",
                return_value=[registry_path],
            ):
                self.assertEqual(common.load_models(workspace), [])
                # Migration mode: dead rows stay visible flagged missing_path
                # so the status viewer can tell the user what to repair.
                flagged = common.load_models(
                    workspace, include_missing=True
                )
            self.assertEqual(len(flagged), 1)
            self.assertEqual(flagged[0]["status"], "missing_path")
            self.assertEqual(flagged[0]["model_id"], "dead_model")

    def test_model_usability_rejects_incomplete_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            model_dir = Path(temp)
            (model_dir / "fold_0").mkdir()
            (model_dir / "plans.json").write_text("{}", encoding="utf-8")
            (model_dir / "dataset.json").write_text("{}", encoding="utf-8")
            usable, reason = common.model_usability(
                {"model_dir": str(model_dir), "folds": ["0"]}
            )
            self.assertFalse(usable)
            self.assertIn("checkpoint_final", reason)

    def test_prediction_backend_accepts_fold_all(self):
        source = (
            ROOT
            / "integrations"
            / "nnunet_segmentation_workflow"
            / "Action4_Predict.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"all" if fold_id_str == "all"', source)

    def test_inference_output_is_bound_before_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp) / "workspace"
            request = {
                "operation": "infer",
                "workspace": str(workspace),
                "image_path": str(Path(temp) / "image.nii.gz"),
                "model_manifest": str(Path(temp) / "manifest.json"),
            }
            with mock.patch.object(jobs, "_launch") as launch:
                launch.return_value = SimpleNamespace(pid=123)
                created = jobs.create_job(request)
            saved = common.read_json(Path(created["job_dir"]) / "request.json")
            self.assertEqual(
                Path(saved["output_path"]), Path(created["job_dir"]) / "prediction.nii.gz"
            )

    def test_stop_writes_control_before_remote_cancel(self):
        with tempfile.TemporaryDirectory() as temp:
            status_path = Path(temp) / "status.json"
            control_path = Path(temp) / "control.json"
            common.write_json_atomic(
                status_path,
                {
                    "status": "training",
                    "execution_backend": "local",
                    "control_path": str(control_path),
                },
            )
            self.assertTrue(jobs.stop_job(status_path))
            self.assertEqual(common.read_json(control_path)["action"], "cancel")
            self.assertEqual(common.read_json(status_path)["status"], "cancelling")

    def test_stop_terminates_verified_orphan_worker(self):
        with tempfile.TemporaryDirectory() as temp:
            status_path = Path(temp) / "status.json"
            control_path = Path(temp) / "control.json"
            common.write_json_atomic(
                status_path,
                {
                    "status": "training",
                    "execution_backend": "local",
                    "control_path": str(control_path),
                    "controller_pid": 1001,
                    "controller_start_marker": "controller-start",
                    "worker_pid": 2002,
                    "worker_start_marker": "worker-start",
                },
            )
            stopped = {"value": False}

            def matches(pid, marker=None):
                if pid == 1001:
                    return False
                return pid == 2002 and marker == "worker-start" and not stopped["value"]

            def terminate(pid):
                self.assertEqual(pid, 2002)
                stopped["value"] = True
                return True

            with mock.patch.object(jobs, "process_matches", side_effect=matches), mock.patch(
                "tools.mimics_label_export.terminate_process_tree", side_effect=terminate
            ):
                self.assertTrue(jobs.stop_job(status_path))
            status = common.read_json(status_path)
            self.assertEqual(status["status"], "cancelled")
            self.assertIsNone(status["worker_pid"])
            self.assertFalse(status["termination_pending"])

    def test_stop_does_not_kill_worker_when_controller_is_alive(self):
        with tempfile.TemporaryDirectory() as temp:
            status_path = Path(temp) / "status.json"
            common.write_json_atomic(
                status_path,
                {
                    "status": "training",
                    "execution_backend": "local",
                    "controller_pid": 1001,
                    "controller_start_marker": "controller-start",
                    "worker_pid": 2002,
                    "worker_start_marker": "worker-start",
                },
            )
            with mock.patch.object(jobs, "process_matches", return_value=True), mock.patch(
                "tools.mimics_label_export.terminate_process_tree"
            ) as terminate:
                self.assertTrue(jobs.stop_job(status_path))
            terminate.assert_not_called()
            self.assertEqual(common.read_json(status_path)["status"], "cancelling")

    def test_dead_local_controller_becomes_failed(self):
        with tempfile.TemporaryDirectory() as temp:
            status_path = Path(temp) / "status.json"
            common.write_json_atomic(
                status_path,
                {
                    "status": "training",
                    "execution_backend": "local",
                    "launcher_pid": 99999999,
                    "updated_at_epoch": time.time() - 30,
                },
            )
            with mock.patch.object(jobs, "process_exists", return_value=False):
                status = jobs.reconcile_job_status(status_path)
            self.assertEqual(status["status"], "failed")

    def test_dead_remote_controller_remains_stoppable(self):
        with tempfile.TemporaryDirectory() as temp:
            status_path = Path(temp) / "status.json"
            common.write_json_atomic(
                status_path,
                {
                    "status": "training",
                    "execution_backend": "remote",
                    "launcher_pid": 99999999,
                    "remote_job_dir": "/srv/mimics/jobs/user/train_1",
                    "updated_at_epoch": time.time() - 30,
                },
            )
            with mock.patch.object(jobs, "process_exists", return_value=False):
                status = jobs.reconcile_job_status(status_path)
            self.assertEqual(status["status"], "orphaned_remote")
            self.assertNotIn(status["status"], common.TERMINAL_STATES)

    def test_dead_remote_controller_before_launch_becomes_failed(self):
        with tempfile.TemporaryDirectory() as temp:
            status_path = Path(temp) / "status.json"
            common.write_json_atomic(
                status_path,
                {
                    "status": "preparing_remote",
                    "execution_backend": "remote",
                    "launcher_pid": 99999999,
                    "updated_at_epoch": time.time() - 30,
                },
            )
            with mock.patch.object(jobs, "process_exists", return_value=False):
                status = jobs.reconcile_job_status(status_path)
            self.assertEqual(status["status"], "failed")

    def test_remote_nnunet_log_sync_uses_pipeline_log(self):
        with tempfile.TemporaryDirectory() as temp:
            local_log = Path(temp) / "job.log"
            session = mock.Mock()
            session.path_exists.return_value = True
            session.download_appended.return_value = 42
            state = {"offset": 0, "remote_relative": "pipeline_job/job.log"}
            remote._sync_remote_log(
                session, "/srv/jobs/run", local_log, state
            )
            session.path_exists.assert_called_once_with(
                "/srv/jobs/run/pipeline_job/job.log"
            )
            self.assertEqual(state["offset"], 42)

    def test_remote_nnunet_curve_is_downloaded_atomically(self):
        with tempfile.TemporaryDirectory() as temp:
            local_curve = Path(temp) / "progress.png"
            session = mock.Mock()
            session.sftp.stat.return_value = SimpleNamespace(
                st_size=123, st_mtime=456
            )

            def download(_remote, local):
                Path(local).write_bytes(b"png")

            session.download.side_effect = download
            state = {}
            changed = remote._sync_remote_nnunet_curve(
                session,
                "/srv/jobs/run",
                {"training_curve_path": "/job/output/runtime/progress.png"},
                local_curve,
                state,
            )
            self.assertTrue(changed)
            session.download.assert_called_once_with(
                "/srv/jobs/run/output/runtime/progress.png", local_curve
            )
            self.assertEqual(state["curve_identity"], (123, 456))

    def test_completed_log_compaction_is_bounded(self):
        with tempfile.TemporaryDirectory() as temp:
            log_path = Path(temp) / "job.log"
            log_path.write_bytes(b"head\n" + b"x" * 4096 + b"\ntail\n")
            result = common.compact_completed_log(
                log_path, max_bytes=1024, head_bytes=64, tail_bytes=128
            )
            self.assertTrue(result["compacted"])
            text = log_path.read_text(encoding="utf-8")
            self.assertIn("Log compacted", text)
            self.assertIn("tail", text)

    def test_remote_request_binds_dataset_cache_token(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp)
            common.write_json_atomic(
                bundle / "remote_request.json",
                {"path": "/remote-cache/{}/raw".format(remote._REMOTE_DATASET_CACHE_TOKEN)},
            )
            remote._bind_remote_prepared_cache(bundle, "a" * 64)
            self.assertIn("a" * 64, common.read_json(bundle / "remote_request.json")["path"])

    def test_remote_training_prepare_returns_complete_request(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            job = root / "job"
            bundle = root / "bundle"
            job.mkdir()
            bundle.mkdir()
            image = root / "image.nii.gz"
            label = root / "label.nii.gz"
            image.write_bytes(b"image")
            label.write_bytes(b"label")
            request_path = job / "request.json"
            status_path = job / "status.json"
            common.write_json_atomic(
                request_path,
                {
                    "operation": "train",
                    "job_id": "remote_train",
                    "task_name": "Liver",
                    "task_id": "liver",
                    "workspace": str(root / "workspace"),
                    "dataset_root": str(root),
                    "labels": [{"name": "Liver", "id": 1}],
                },
            )
            common.write_json_atomic(status_path, {"status": "preparing_remote"})
            common.write_json_atomic(job / "control.json", {"action": "run"})
            spec = {
                "job_dir": str(job),
                "status_path": str(status_path),
                "request_path": str(request_path),
            }
            rows = [
                {
                    "case_id": "case_1",
                    "image": image,
                    "label": label,
                    "fingerprint": "fingerprint",
                }
            ]
            with mock.patch(
                "tools.nnunet_pipeline.prepare_source_grid_cases", return_value=rows
            ):
                prepared = remote._prepare_nnunet(spec, bundle)
            payload = common.read_json(bundle / "remote_request.json")
            self.assertEqual(payload["kind"], "nnunet_train")
            self.assertTrue(payload["pipeline_request"]["gpu_lock_managed_externally"])
            self.assertEqual(prepared["train_count"], 1)
            self.assertTrue(prepared["remote_artifact_relative"].startswith("output/models/"))


class FakeMask:
    def __init__(self, name, pixels=1):
        self.name = name
        self.guid = name + "-guid"
        self.number_of_pixels = pixels
        self.selected = False
        self.image = None


class MimicsRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_modules = {
            name: sys.modules.get(name)
            for name in ("mimics", "mimics_mask_apply", "runtime_common", "nnunet_mimics")
        }
        cls.dialog_answer = "Create Editable Copies"
        cls.masks = [FakeMask("Liver", 10)]
        mimics = types.ModuleType("mimics")
        mimics.data = SimpleNamespace(masks=cls.masks, images=SimpleNamespace(get_active=lambda: None))
        mimics.logging = SimpleNamespace(log_user_message=lambda **kwargs: None)
        mimics.dialogs = SimpleNamespace(
            question_box=lambda **kwargs: cls.dialog_answer,
            message_box=lambda *args, **kwargs: None,
        )
        mimics.segment = SimpleNamespace(create_mask=lambda: FakeMask("New", 0))
        mimics.update_gui = lambda: None
        sys.modules["mimics"] = mimics

        runtime = types.ModuleType("runtime_common")
        runtime.find_root = lambda value, sentinel_files=None, max_depth=6: str(ROOT)
        runtime.read_json = common.read_json
        runtime.write_json_atomic = common.write_json_atomic
        runtime.safe_slug = common.safe_identifier
        runtime.try_acquire_local_operation = lambda *args: "token"
        runtime.release_local_operation = lambda *args: None
        runtime.process_exists = lambda pid: True
        runtime.background_process_kwargs = lambda: {}
        runtime.find_mimics_exe = lambda: ""
        sys.modules["runtime_common"] = runtime

        mask_apply = types.ModuleType("mimics_mask_apply")
        mask_apply._mask_identity = lambda mask: mask.guid
        mask_apply._new_prediction_mask = lambda name: FakeMask(name, 0)
        mask_apply._set_mask_from_u8 = lambda mask, path, shape, transaction_name=None: setattr(
            mask, "applied", True
        )
        mask_apply._monitor_target_is_open = lambda monitor: (True, "")
        mask_apply._current_project_path = lambda: ""
        mask_apply._active_live_grid_payload = lambda: {}
        mask_apply._active_source_geometry_payload = lambda: {}
        mask_apply._resolve_prediction_context = lambda: ("", "", "")
        mask_apply._config = lambda: {}
        mask_apply._integration_root = lambda config: str(ROOT)
        mask_apply._integration_python = lambda config, root: sys.executable
        mask_apply._buffer_mapping_from_config = lambda config: ([0, 1, 2], [False, False, False])
        sys.modules["mimics_mask_apply"] = mask_apply
        cls.module = importlib.import_module("nnunet_mimics")

    @classmethod
    def tearDownClass(cls):
        for name, module in cls.original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def test_single_class_bridge_row_uses_manifest_label(self):
        self.dialog_answer = "Create Editable Copies"
        monitor = {
            "labels": [{"id": 1, "name": "Liver", "aliases": ["liver"]}],
            "matching_masks": [],
        }
        self.module._prepare_apply_queue(
            monitor,
            {
                "masks": [
                    {
                        "name": "nnunet_prediction",
                        "output_path": "a.u8",
                        "mimics_shape": [2, 2, 2],
                        "foreground_voxels": 1,
                        "label_value": 1,
                    }
                ]
            },
        )
        self.assertEqual(monitor["apply_queue"][0]["label"]["name"], "Liver")

    def test_single_present_class_keeps_multiclass_label_id(self):
        self.dialog_answer = "Create Editable Copies"
        monitor = {
            "labels": [
                {"id": 1, "name": "Liver", "aliases": []},
                {"id": 2, "name": "Spleen", "aliases": []},
            ],
            "matching_masks": [],
        }
        self.module._prepare_apply_queue(
            monitor,
            {
                "masks": [
                    {
                        "name": "nnunet_prediction",
                        "output_path": "a.u8",
                        "mimics_shape": [2, 2, 2],
                        "foreground_voxels": 1,
                        "label_value": 2,
                    }
                ]
            },
        )
        self.assertEqual(monitor["apply_queue"][0]["label"]["name"], "Spleen")

    def test_grouped_prediction_never_overwrites_a_constituent_mask(self):
        brain = FakeMask("brain", 10)
        self.masks[:] = [brain]
        monitor = {
            "matching_masks": [
                {"guid": brain.guid, "name": brain.name, "pixel_count": 10}
            ]
        }
        grouped = {
            "id": 1,
            "name": "CT1_Head",
            "aliases": ["brain", "skull"],
            "source_mode": "union",
        }
        destination = self.module._mask_for_label(monitor, grouped, "update")
        self.assertIsNot(destination, brain)
        self.assertEqual(destination.name, "AI_CT1_Head")

    def test_empty_prediction_is_not_offered_for_overwrite(self):
        monitor = {
            "labels": [{"id": 1, "name": "Liver", "aliases": []}],
            "matching_masks": [],
        }
        with self.assertRaisesRegex(RuntimeError, "no non-background"):
            self.module._prepare_apply_queue(
                monitor,
                {
                    "masks": [
                        {
                            "name": "nnunet_prediction",
                            "output_path": "empty.u8",
                            "mimics_shape": [2, 2, 2],
                            "foreground_voxels": 0,
                            "label_value": 0,
                        }
                    ]
                },
            )

    def test_changed_matching_mask_creates_copy(self):
        monitor = {
            "matching_masks": [
                {"guid": "Liver-guid", "name": "Liver", "pixel_count": 5}
            ]
        }
        result = self.module._mask_for_label(
            monitor, {"name": "Liver", "aliases": []}, "update"
        )
        self.assertEqual(result.name, "AI_Liver")

    def test_new_matching_mask_is_not_overwritten_without_launch_snapshot(self):
        monitor = {"matching_masks": []}
        result = self.module._mask_for_label(
            monitor, {"name": "Liver", "aliases": []}, "update"
        )
        self.assertEqual(result.name, "AI_Liver")

    def test_mask_application_resume_reuses_planned_copy_after_crash(self):
        with tempfile.TemporaryDirectory() as temp:
            status_path = Path(temp) / "status.json"
            common.write_json_atomic(status_path, {"status": "completed"})
            self.masks[:] = []
            created = []
            original_new = self.module.mimics_mask_apply._new_prediction_mask
            original_set = self.module.mimics_mask_apply._set_mask_from_u8

            def create_mask(name):
                mask = FakeMask(name, 0)
                self.masks.append(mask)
                created.append(mask)
                return mask

            calls = [0]

            def fail_once(mask, path, shape, transaction_name=None):
                calls[0] += 1
                if calls[0] == 1:
                    raise RuntimeError("simulated Mimics crash")
                mask.applied = True

            self.module.mimics_mask_apply._new_prediction_mask = create_mask
            self.module.mimics_mask_apply._set_mask_from_u8 = fail_once
            bridge = {
                "masks": [
                    {
                        "output_path": "liver.u8",
                        "mimics_shape": [2, 2, 2],
                        "foreground_voxels": 1,
                        "label_value": 1,
                    }
                ]
            }
            try:
                first = {
                    "status_path": str(status_path),
                    "labels": [{"id": 1, "name": "Liver", "aliases": []}],
                    "matching_masks": [],
                }
                self.module._prepare_apply_queue(first, bridge)
                with self.assertRaisesRegex(RuntimeError, "simulated"):
                    self.module._apply_one(first)
                self.assertEqual(len(created), 1)

                resumed = {
                    "status_path": str(status_path),
                    "labels": [{"id": 1, "name": "Liver", "aliases": []}],
                    "matching_masks": [],
                }
                self.module._prepare_apply_queue(resumed, bridge)
                self.assertFalse(self.module._apply_one(resumed))
                self.assertEqual(len(created), 1)
                application = common.read_json(status_path)["mimics_application"]
                self.assertEqual(application["records"]["1"]["state"], "applied")
                self.assertEqual(application["records"]["1"]["target_name"], "AI_Liver")
            finally:
                self.module.mimics_mask_apply._new_prediction_mask = original_new
                self.module.mimics_mask_apply._set_mask_from_u8 = original_set

    def test_stopping_monitor_terminates_owned_bridge(self):
        process = mock.Mock()
        process.poll.return_value = None
        terminator = mock.Mock(return_value=True)
        previous = getattr(
            self.module.runtime_common, "terminate_process_async", None
        )
        self.module.runtime_common.terminate_process_async = terminator
        try:
            self.module._MONITORS["bridge"] = {
                "monitor_key": "bridge",
                "bridge_pid": 123,
                "bridge_process": process,
            }
            self.module._stop_monitor("bridge")
        finally:
            if previous is None:
                delattr(self.module.runtime_common, "terminate_process_async")
            else:
                self.module.runtime_common.terminate_process_async = previous
        terminator.assert_called_once_with(
            process=process, pid=123, graceful_seconds=0.5
        )


if __name__ == "__main__":
    # Isolate real lock acquisitions (dataset lock, GPU lock) to a temp
    # dir: per-run temp workspace paths hash to new lock names and would
    # leak one-byte guard anchors into the production
    # .mimics_runtime/locks directory forever. This suite runs as its own
    # process (regression matrix / direct invocation), so a process-wide
    # env var covers every test class.
    with tempfile.TemporaryDirectory(prefix="nnunet_test_locks_") as _lock_tmp:
        os.environ["MIMICS_RESOURCE_LOCK_DIR"] = _lock_tmp
        unittest.main(verbosity=2)
