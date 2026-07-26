#!/usr/bin/env python3
"""Regression tests for relocatable paths and source-grid AI workflows."""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
import itertools
from pathlib import Path

import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for value in (str(ROOT), str(ROOT / "tools"), str(ROOT / "runtime_py35")):
    if value not in sys.path:
        sys.path.insert(0, value)

import dataset_manifest
import nninteractive_bridge
from mimics_bridge import _validate_resampled_mask_foreground
from tools import fewshot_pipeline
from tools import nninteractive_task_common


class GeometryManifestRegressionTests(unittest.TestCase):
    def test_manifest_paths_survive_moving_the_dataset_bundle(self):
        with tempfile.TemporaryDirectory() as value:
            parent = Path(value)
            bundle = parent / "bundle"
            image = bundle / "images" / "case_a" / "ct.nii.gz"
            label = bundle / "exports" / "case_a" / "segmentations" / "Liver.nii.gz"
            image.parent.mkdir(parents=True)
            label.parent.mkdir(parents=True)
            image.write_bytes(b"image")
            label.write_bytes(b"label")
            catalog = bundle / "catalog"
            dataset_manifest.update_case(
                catalog,
                "case_a",
                image_path=image,
                labels=[{"mask_name": "Liver", "path": label}],
            )
            moved = parent / "moved_bundle"
            shutil.move(str(bundle), str(moved))
            manifest = moved / "catalog" / dataset_manifest.MANIFEST_FILENAME
            row = dataset_manifest.load_manifest(manifest)["cases"]["case_a"]
            self.assertEqual(
                moved / "images" / "case_a" / "ct.nii.gz",
                Path(dataset_manifest.resolve_case_path(manifest, row, "image")),
            )
            match = dataset_manifest.resolve_case_label(
                manifest, row, ["liver"]
            )
            self.assertEqual(
                moved / "exports" / "case_a" / "segmentations" / "Liver.nii.gz",
                Path(match[1]),
            )

    def test_dinov3_uses_target_mask_alias_and_manifest_path(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            case = root / "images" / "case_a"
            case.mkdir(parents=True)
            image = case / "ct.nii.gz"
            nib.save(nib.Nifti1Image(np.ones((3, 3, 3)), np.eye(4)), image)
            export_root = root / "exports"
            label = root / "labels_elsewhere" / "liver_seg.nii.gz"
            label.parent.mkdir(parents=True)
            mask = np.zeros((3, 3, 3), dtype=np.uint8)
            mask[1, 1, 1] = 1
            nib.save(nib.Nifti1Image(mask, np.eye(4)), label)
            dataset_manifest.update_case(
                export_root,
                "case_a",
                image_path=image,
                labels=[{"mask_name": "Liver Seg", "path": label}],
            )
            samples, skipped = fewshot_pipeline.discover_samples(
                root / "images",
                "liver",
                label_root=export_root,
                fallback_to_case_labels=False,
                mask_names=["Liver Seg"],
            )
            self.assertEqual([], skipped)
            self.assertTrue(label.samefile(samples[0]["label"]))

    def test_nonempty_mask_cannot_silently_become_empty(self):
        source = np.zeros((3, 3, 3), dtype=np.uint8)
        source[1, 1, 1] = 1
        with self.assertRaisesRegex(RuntimeError, "silently misregistered"):
            _validate_resampled_mask_foreground(
                source, np.zeros_like(source), "test mapping"
            )

    def test_task_model_canonical_mapping_round_trip(self):
        affine = np.array(
            [
                [0.0, -2.0, 0.0, 10.0],
                [-1.0, 0.0, 0.0, 20.0],
                [0.0, 0.0, 3.0, 30.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        mapping = nninteractive_bridge.canonical_ras_buffer_mapping(affine)
        source = np.arange(2 * 3 * 4).reshape((2, 3, 4))
        canonical = nninteractive_bridge.mimics_to_platform(source, mapping)
        expected = nib.orientations.apply_orientation(
            source, nib.orientations.io_orientation(affine)
        )
        np.testing.assert_array_equal(expected, canonical)
        np.testing.assert_array_equal(
            source,
            nninteractive_bridge.platform_to_mimics(canonical, mapping),
        )

    def test_task_model_mapping_covers_every_axis_flip_orientation(self):
        source = np.arange(2 * 3 * 4).reshape((2, 3, 4))
        for axes in itertools.permutations((0, 1, 2)):
            for signs in itertools.product((-1.0, 1.0), repeat=3):
                affine = np.eye(4)
                affine[:3, :3] = 0.0
                for input_axis, world_axis in enumerate(axes):
                    affine[world_axis, input_axis] = signs[input_axis]
                mapping = nninteractive_bridge.canonical_ras_buffer_mapping(
                    affine
                )
                canonical = nninteractive_bridge.mimics_to_platform(
                    source, mapping
                )
                expected = nib.orientations.apply_orientation(
                    source, nib.orientations.io_orientation(affine)
                )
                np.testing.assert_array_equal(expected, canonical)
                np.testing.assert_array_equal(
                    source,
                    nninteractive_bridge.platform_to_mimics(
                        canonical, mapping
                    ),
                )

    def test_task_model_buffer_intensity_is_restored_to_source_values(self):
        gv = np.asarray([[[0.0, 1024.0, 2024.0]]], dtype=np.float32)
        hu = nninteractive_bridge._apply_buffer_model_intensity_transform(
            gv,
            {
                "image_buffer_to_model_slope": 1.0,
                "image_buffer_to_model_intercept": -1024.0,
            },
        )
        np.testing.assert_array_equal(
            np.asarray([[[-1024.0, 0.0, 1000.0]]], dtype=np.float32),
            hu,
        )

    def test_nninteractive_prepared_data_accepts_separate_label_root(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            case = root / "images" / "case_a"
            case.mkdir(parents=True)
            nib.save(
                nib.Nifti1Image(np.ones((2, 2, 2)), np.eye(4)),
                case / "ct.nii.gz",
            )
            labels = root / "exports" / "case_a" / "segmentations"
            labels.mkdir(parents=True)
            nib.save(
                nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4)),
                labels / "Liver.nii.gz",
            )
            rows = nninteractive_task_common.discover_prepared_cases(
                root / "images", ["Liver"], label_root=root / "exports"
            )
            self.assertEqual("ready", rows[0]["state"])
            self.assertEqual("separate_label_root", rows[0]["label_source"])

    def test_dinov3_materializes_mhd_image_and_label_on_source_grid(self):
        import SimpleITK as sitk

        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            case = root / "case_a"
            segmentations = case / "segmentations"
            segmentations.mkdir(parents=True)
            image_array = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
            label_array = np.zeros((2, 3, 4), dtype=np.uint8)
            label_array[:, 1:, 1:3] = 1
            image = sitk.GetImageFromArray(image_array)
            label = sitk.GetImageFromArray(label_array)
            for current in (image, label):
                current.SetSpacing((0.8, 0.9, 2.5))
                current.SetOrigin((12.0, -8.0, 30.0))
            sitk.WriteImage(image, str(case / "ct.mhd"))
            sitk.WriteImage(label, str(segmentations / "Liver.mhd"))
            samples, skipped = fewshot_pipeline.discover_samples(
                root, "liver", mask_names=["Liver"]
            )
            self.assertEqual([], skipped)
            rows = fewshot_pipeline._materialize_split(
                samples,
                root / "prepared" / "imagesTr",
                root / "prepared" / "labelsTr",
                "train",
            )
            prepared_image = nib.load(rows[0]["dataset_image"])
            prepared_label = nib.load(rows[0]["dataset_label"])
            self.assertEqual(prepared_image.shape, prepared_label.shape)
            np.testing.assert_allclose(
                prepared_image.affine, prepared_label.affine, atol=1.0e-5
            )
            self.assertGreater(
                int(np.count_nonzero(np.asanyarray(prepared_label.dataobj))), 0
            )


if __name__ == "__main__":
    unittest.main()
