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


if __name__ == "__main__":
    unittest.main()
