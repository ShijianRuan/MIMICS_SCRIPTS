#!/usr/bin/env python3
"""Focused tests for ITK Snake, IGAC, and ScribblePrompt integration."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "tools", ROOT / "runtime_py35"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import interactive_algorithms_worker as worker
import igac_engine
from fake_mimics_flow_test import FakeImage, FakeMask, FakeMimics, _u8_pattern_buffer


class WorkerGeometryTests(unittest.TestCase):
    def test_prompt_plane_accepts_axial_scribbles(self):
        specs = [
            {"bbox": [[2, 6], [3, 8], [4, 5]]},
            {"bbox": [[1, 5], [1, 4], [4, 5]]},
        ]
        self.assertEqual(worker._prompt_plane(specs, (12, 13, 14)), (2, 4))

    def test_prompt_plane_rejects_different_slices(self):
        specs = [
            {"bbox": [[2, 6], [3, 8], [4, 5]]},
            {"bbox": [[1, 5], [1, 4], [5, 6]]},
        ]
        with self.assertRaisesRegex(RuntimeError, "same slice"):
            worker._prompt_plane(specs, (12, 13, 14))

    def test_scribble_channels_and_plane_write_are_local(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            foreground = np.ones((2, 3, 1), dtype=np.uint8)
            background = np.ones((1, 2, 1), dtype=np.uint8)
            fg_path = root / "fg.u8"
            bg_path = root / "bg.u8"
            foreground.tofile(fg_path)
            background.tofile(bg_path)
            specs = [
                {"path": str(fg_path), "shape": [2, 3, 1], "bbox": [[1, 3], [2, 5], [4, 5]], "include": True},
                {"path": str(bg_path), "shape": [1, 2, 1], "bbox": [[4, 5], [6, 8], [4, 5]], "include": False},
            ]
            result = worker._scribble_planes(specs, (8, 10, 6), 2, 4)
            self.assertEqual(result.shape, (2, 8, 10))
            self.assertEqual(int(result[0].sum()), 6)
            self.assertEqual(int(result[1].sum()), 2)

            volume = np.zeros((8, 10, 6), dtype=np.uint8)
            plane = np.ones((8, 10), dtype=np.uint8)
            worker._write_plane(volume, plane, 2, 4)
            self.assertEqual(int(volume[:, :, 4].sum()), 80)
            self.assertEqual(int(volume[:, :, :4].sum()), 0)
            self.assertEqual(int(volume[:, :, 5:].sum()), 0)

    def test_normalisation_is_bounded(self):
        image = np.arange(1000, dtype=np.float32).reshape(10, 10, 10)
        output = worker._normalise_image(image)
        self.assertGreaterEqual(float(output.min()), 0.0)
        self.assertLessEqual(float(output.max()), 1.0)
        self.assertEqual(output.dtype, np.float32)

    def test_igac_display_mapping_preserves_xyz_axes(self):
        values = np.arange(4 * 5 * 6).reshape((4, 5, 6))
        self.assertTrue(np.array_equal(igac_engine.plane_from_xyz(values, "axial", 2), values[:, :, 2].T))
        self.assertTrue(np.array_equal(igac_engine.plane_from_xyz(values, "coronal", 3), values[:, 3, :].T))
        self.assertTrue(np.array_equal(igac_engine.plane_from_xyz(values, "sagittal", 1), values[1, :, :].T))
        self.assertEqual(igac_engine.display_to_xyz("coronal", 3, 2, 4), (2.0, 3.0, 4.0))

    def test_igac_empty_large_volume_requires_a_seed(self):
        mask = np.zeros((100, 100, 100), dtype=np.uint8)
        with self.assertRaisesRegex(RuntimeError, "rough seed Mask"):
            igac_engine.roi_slices(mask, (1.0, 1.0, 1.0), 20.0, 1000)


@unittest.skipUnless(importlib.util.find_spec("PySide6") is not None, "PySide6 is unavailable")
class IGACCanvasTests(unittest.TestCase):
    def test_zoomed_canvas_maps_brush_back_to_full_image_coordinates(self):
        from PySide6 import QtCore, QtWidgets
        from tools.igac_gui import ImageCanvas

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        canvas = ImageCanvas()
        canvas.resize(800, 600)
        image = np.zeros((120, 200), dtype=np.float32)
        mask = np.zeros_like(image, dtype=bool)
        canvas.set_frame({"image": image, "mask": mask, "add": mask, "barrier": mask})
        canvas.fit_source_rect(QtCore.QRectF(50.0, 30.0, 40.0, 20.0))
        target = canvas._compute_target()
        mapped = canvas._voxel_at(target.center())
        self.assertIsNotNone(mapped)
        self.assertAlmostEqual(mapped[0], 70.0, places=4)
        self.assertAlmostEqual(mapped[1], 40.0, places=4)
        canvas.zoom(2.0, mapped)
        self.assertAlmostEqual(canvas._view_rect.width(), 20.0, places=4)
        self.assertAlmostEqual(canvas._view_rect.height(), 10.0, places=4)
        app.processEvents()

    def test_correct_gesture_infers_error_type_and_locks_it_for_the_stroke(self):
        from PySide6 import QtWidgets
        from tools.igac_gui import ImageCanvas

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        canvas = ImageCanvas()
        image = np.zeros((24, 24), dtype=np.float32)
        mask = np.zeros_like(image, dtype=bool)
        mask[:, :12] = True
        canvas.set_frame({"image": image, "mask": mask})
        self.assertEqual(canvas._automatic_mode((5.0, 12.0)), "barrier")
        self.assertEqual(canvas._automatic_mode((18.0, 12.0)), "add")
        self.assertTrue(canvas._near_mask_boundary(mask, (11.5, 12.0)))

        emitted = []
        canvas.brush_requested.connect(
            lambda mode, start, end: emitted.append((str(mode), tuple(start), tuple(end)))
        )
        canvas._drawing = True
        canvas._pending_auto = True
        canvas._press_voxel = (11.5, 12.0)
        canvas._press_mask = mask.copy()
        canvas._stroke_points = [canvas._press_voxel]
        canvas._resolve_pending_auto((15.0, 12.0))
        self.assertTrue(emitted)
        self.assertTrue(all(value[0] == "add" for value in emitted))

        # A live frame may move the visible boundary while the pointer is down;
        # the user's correction meaning must remain stable for the whole stroke.
        canvas._mask_plane[:] = True
        canvas._emit_stroke_segment((18.0, 12.0), force=True)
        self.assertEqual(emitted[-1][0], "add")
        app.processEvents()

    def test_mask_only_frame_reuses_cached_grayscale_background(self):
        from PySide6 import QtWidgets
        from tools.igac_gui import ImageCanvas

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        canvas = ImageCanvas()
        image = np.linspace(0.0, 1.0, 20 * 30, dtype=np.float32).reshape(20, 30)
        mask = np.zeros_like(image, dtype=bool)
        canvas.set_frame({"image": image, "mask": mask})
        background = canvas._image
        mask[4:12, 6:18] = True
        canvas.set_frame({"mask": mask})
        self.assertIs(canvas._image, background)
        self.assertTrue(np.array_equal(canvas._mask_plane, mask))
        app.processEvents()

    def test_transposed_noncontiguous_image_plane_is_accepted(self):
        from PySide6 import QtWidgets
        from tools.igac_gui import ImageCanvas

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        canvas = ImageCanvas()
        image = np.arange(20 * 30, dtype=np.float32).reshape(20, 30).T
        self.assertFalse(image.flags.c_contiguous)
        mask = np.zeros_like(image, dtype=bool)
        canvas.set_frame({"image": image, "mask": mask})
        self.assertIsNotNone(canvas._image)
        self.assertEqual(canvas._image.width(), image.shape[1])
        self.assertEqual(canvas._image.height(), image.shape[0])
        app.processEvents()

    def test_correction_stroke_translates_to_add_or_remove_guidance(self):
        from PySide6 import QtCore, QtTest, QtWidgets
        from tools.igac_gui import ImageCanvas

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        canvas = ImageCanvas()
        canvas.interaction_mode = "auto"
        canvas.resize(480, 420)
        canvas.show()
        image = np.zeros((40, 40), dtype=np.float32)
        mask = np.zeros_like(image, dtype=bool)
        mask[:, :20] = True
        canvas.set_frame({"image": image, "mask": mask})
        app.processEvents()
        emitted = []
        started = []
        finished = []
        canvas.brush_requested.connect(
            lambda mode, start, end: emitted.append((str(mode), tuple(start), tuple(end)))
        )
        canvas.stroke_started.connect(lambda: started.append(True))
        canvas.stroke_finished.connect(lambda: finished.append(True))
        start = canvas._screen_at_voxel((19.5, 20.0))
        end = canvas._screen_at_voxel((27.0, 20.0))
        self.assertIsNotNone(start)
        self.assertIsNotNone(end)
        QtTest.QTest.mousePress(canvas, QtCore.Qt.LeftButton, pos=start.toPoint())
        QtTest.QTest.mouseMove(canvas, end.toPoint())
        QtTest.QTest.mouseRelease(canvas, QtCore.Qt.LeftButton, pos=end.toPoint())
        app.processEvents()
        self.assertEqual(len(started), 1)
        self.assertEqual(len(finished), 1)
        self.assertGreaterEqual(len(emitted), 2)
        self.assertTrue(all(value[0] == "add" for value in emitted))

        emitted[:] = []
        inward = canvas._screen_at_voxel((12.0, 20.0))
        self.assertIsNotNone(inward)
        QtTest.QTest.mousePress(canvas, QtCore.Qt.LeftButton, pos=start.toPoint())
        QtTest.QTest.mouseMove(canvas, inward.toPoint())
        QtTest.QTest.mouseRelease(canvas, QtCore.Qt.LeftButton, pos=inward.toPoint())
        app.processEvents()
        self.assertGreaterEqual(len(emitted), 2)
        self.assertTrue(all(value[0] == "barrier" for value in emitted))
        canvas.close()

    def test_ffd_drag_captures_a_boundary_point_without_emitting_a_brush(self):
        from PySide6 import QtCore, QtTest, QtWidgets
        from tools.igac_gui import ImageCanvas

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        canvas = ImageCanvas()
        canvas.interaction_mode = "ffd"
        canvas.display_spacing_uv = (0.7, 1.4)
        canvas.boundary_capture_mm = 3.0
        canvas.resize(480, 420)
        canvas.show()
        image = np.zeros((40, 40), dtype=np.float32)
        mask = np.zeros_like(image, dtype=bool)
        mask[8:32, 6:20] = True
        canvas.set_frame({"image": image, "mask": mask})
        app.processEvents()

        starts = []
        updates = []
        finishes = []
        brushes = []
        modes = []
        canvas.boundary_drag_started.connect(lambda point: starts.append(tuple(point)))
        canvas.boundary_drag_requested.connect(
            lambda start, end: updates.append((tuple(start), tuple(end)))
        )
        canvas.boundary_drag_finished.connect(lambda: finishes.append(True))
        canvas.brush_requested.connect(lambda *values: brushes.append(values))
        canvas.gesture_mode_changed.connect(lambda value: modes.append(str(value)))

        press = canvas._screen_at_voxel((20.8, 20.0))
        end = canvas._screen_at_voxel((28.0, 20.0))
        self.assertIsNotNone(press)
        self.assertIsNotNone(end)
        QtTest.QTest.mousePress(canvas, QtCore.Qt.LeftButton, pos=press.toPoint())
        QtTest.QTest.mouseMove(canvas, end.toPoint())
        QtTest.QTest.mouseRelease(canvas, QtCore.Qt.LeftButton, pos=end.toPoint())
        app.processEvents()
        self.assertEqual(len(starts), 1)
        self.assertAlmostEqual(starts[0][0], 19.5, places=3)
        self.assertGreaterEqual(len(updates), 1)
        self.assertEqual(len(finishes), 1)
        self.assertFalse(brushes)
        self.assertIn("boundary_captured", modes)

        far = canvas._screen_at_voxel((35.0, 3.0))
        self.assertIsNotNone(far)
        QtTest.QTest.mousePress(canvas, QtCore.Qt.LeftButton, pos=far.toPoint())
        QtTest.QTest.mouseRelease(canvas, QtCore.Qt.LeftButton, pos=far.toPoint())
        app.processEvents()
        self.assertEqual(len(starts), 1)
        self.assertIn("boundary_miss", modes)
        canvas.close()

    def test_crossing_from_deep_inside_reclassifies_the_whole_stroke(self):
        from PySide6 import QtCore, QtTest, QtWidgets
        from tools.igac_gui import ImageCanvas

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        canvas = ImageCanvas()
        canvas.interaction_mode = "auto"
        canvas.resize(480, 420)
        canvas.show()
        image = np.zeros((40, 40), dtype=np.float32)
        mask = np.zeros_like(image, dtype=bool)
        mask[:, :20] = True
        canvas.set_frame({"image": image, "mask": mask})
        app.processEvents()
        emitted = []
        reclassified = []
        canvas.brush_requested.connect(
            lambda mode, start, end: emitted.append((str(mode), tuple(start), tuple(end)))
        )
        canvas.stroke_reclassified.connect(
            lambda mode, segments: reclassified.append((str(mode), list(segments)))
        )
        start = canvas._screen_at_voxel((6.0, 20.0))
        outside = canvas._screen_at_voxel((28.0, 20.0))
        self.assertIsNotNone(start)
        self.assertIsNotNone(outside)
        QtTest.QTest.mousePress(canvas, QtCore.Qt.LeftButton, pos=start.toPoint())
        self.assertTrue(emitted)
        self.assertEqual(emitted[0][0], "barrier")
        QtTest.QTest.mouseMove(canvas, outside.toPoint())
        reclassification_count = len(reclassified)
        back_inside = canvas._screen_at_voxel((10.0, 20.0))
        self.assertIsNotNone(back_inside)
        QtTest.QTest.mouseMove(canvas, back_inside.toPoint())
        QtTest.QTest.mouseRelease(canvas, QtCore.Qt.LeftButton, pos=back_inside.toPoint())
        app.processEvents()
        self.assertTrue(reclassified)
        self.assertEqual(reclassified[-1][0], "add")
        self.assertGreaterEqual(len(reclassified[-1][1]), 2)
        self.assertEqual(len(reclassified), reclassification_count)
        self.assertEqual(canvas._active_brush_mode, "add")
        canvas.close()

    def test_brush_queue_coalesces_within_but_not_across_strokes(self):
        from tools.igac_gui import EngineThread

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            thread = EngineThread(
                {
                    "shape": [10, 10, 10],
                    "igac": {},
                    "status_path": str(root / "status.json"),
                    "log_path": str(root / "igac.log"),
                }
            )
            first = {"mode": "add", "start_u": 1.0}
            second = {"mode": "add", "start_u": 2.0}
            third = {"mode": "barrier", "start_u": 3.0}
            thread.submit("stroke_start")
            thread.submit("brush_segment", first)
            thread.submit("brush_segment", second)
            thread.submit("stroke_end")
            thread.submit("stroke_start")
            thread.submit("brush_segment", third)

            commands = []
            while not thread.commands.empty():
                commands.append(thread.commands.get_nowait())
            self.assertEqual(
                [command for command, _payload in commands],
                ["stroke_start", "brush_flush", "stroke_end", "stroke_start", "brush_flush"],
            )
            self.assertEqual([item["start_u"] for item in commands[1][1]], [1.0, 2.0])
            self.assertEqual([item["start_u"] for item in commands[4][1]], [3.0])

    def test_boundary_pull_queue_keeps_only_the_latest_pointer_position(self):
        from tools.igac_gui import EngineThread

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            thread = EngineThread(
                {
                    "shape": [10, 10, 10],
                    "igac": {},
                    "status_path": str(root / "status.json"),
                    "log_path": str(root / "igac.log"),
                }
            )
            thread.submit("boundary_pull_start")
            thread.submit("boundary_pull_update", {"end_u": 2.0})
            thread.submit("boundary_pull_update", {"end_u": 8.0})
            thread.submit("boundary_pull_end")
            commands = []
            while not thread.commands.empty():
                commands.append(thread.commands.get_nowait())
            self.assertEqual(
                [command for command, _payload in commands],
                ["boundary_pull_start", "boundary_pull_flush", "boundary_pull_end"],
            )
            self.assertEqual(commands[1][1]["payload"]["end_u"], 8.0)


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("scipy") is not None,
    "PyTorch/scipy are not installed in this test interpreter",
)
class IGACEngineTests(unittest.TestCase):
    def test_mask_bounds_and_roi_do_not_require_foreground_coordinate_table(self):
        mask = np.zeros((64, 48, 32), dtype=bool)
        mask[7:51, 9:39, 3:27] = True
        bounds = igac_engine._mask_bounds(mask)
        self.assertIsNotNone(bounds)
        minimum, maximum = bounds
        self.assertEqual(minimum.tolist(), [7, 9, 3])
        self.assertEqual(maximum.tolist(), [51, 39, 27])
        roi = igac_engine.roi_slices(
            mask, (0.5, 1.0, 2.0), 2.0, 500_000, bounds=bounds
        )
        self.assertEqual([(item.start, item.stop) for item in roi], [(3, 55), (7, 41), (1, 29)])

    @staticmethod
    def _dice(first, second):
        first = np.asarray(first, dtype=bool)
        second = np.asarray(second, dtype=bool)
        return 2.0 * float(np.logical_and(first, second).sum()) / float(first.sum() + second.sum())

    def test_anisotropic_evolution_and_brushes_remain_on_mimics_grid(self):
        shape = (28, 30, 24)
        x, y, z = np.indices(shape)
        image = np.exp(
            -(((x - 14) / 8.0) ** 2 + ((y - 15) / 9.0) ** 2 + ((z - 12) / 7.0) ** 2)
        ).astype(np.float32)
        initial = (
            ((x - 14) / 5.0) ** 2 + ((y - 15) / 6.0) ** 2 + ((z - 12) / 4.0) ** 2 < 1.0
        )
        engine = igac_engine.IGACEngine(
            image,
            initial,
            (0.7, 0.7, 1.5),
            device="cpu",
            workspace_margin_mm=5.0,
            max_roi_voxels=100000,
            parameters={"range_mm": 1.4, "smooth": 8, "data_weight": 3, "time_step": 0.03},
        )
        engine.step(2)
        self.assertTrue(bool(np.isfinite(engine.phi.cpu().numpy()).all()))
        self.assertEqual(engine.result_xyz().shape, shape)
        self.assertTrue(engine.apply_brush("add", "axial", 12, 14, 15, 2.0))
        self.assertTrue(engine.apply_brush("barrier", "coronal", 15, 14, 12, 1.5))
        for orientation, index, expected in (
            ("axial", 12, (30, 28)),
            ("coronal", 15, (24, 28)),
            ("sagittal", 14, (24, 30)),
        ):
            frame = engine.frame(orientation, index)
            self.assertTrue(all(value.shape == expected for value in frame.values()))

    def test_display_window_changes_only_rendering_not_3d_result(self):
        shape = (18, 20, 12)
        image = np.linspace(-1000.0, 1400.0, num=int(np.prod(shape)), dtype=np.float32).reshape(shape)
        initial = np.zeros(shape, dtype=bool)
        initial[6:12, 7:14, 4:9] = True
        engine = igac_engine.IGACEngine(
            image,
            initial,
            (0.8, 0.8, 1.5),
            device="cpu",
            workspace_margin_mm=3.0,
            max_roi_voxels=100000,
            display_window=(-200.0, 300.0),
        )
        before = engine.result_xyz().copy()
        first = engine.frame("axial", 6)["image"]
        engine.set_display_window(-1000.0, 1400.0)
        second = engine.frame("axial", 6)["image"]
        self.assertFalse(np.allclose(first, second))
        self.assertTrue(np.array_equal(before, engine.result_xyz()))
        overlay_only = engine.frame("axial", 6, include_image=False)
        self.assertNotIn("image", overlay_only)
        self.assertEqual(set(overlay_only), {"mask", "add", "barrier"})

    def test_hard_boundary_sphere_converges_without_leakage(self):
        shape = (28, 28, 28)
        x, y, z = np.indices(shape)
        radius = np.sqrt((x - 14) ** 2 + (y - 14) ** 2 + (z - 14) ** 2)
        target = radius <= 8
        initial = radius <= 5
        image = np.where(target, 1.0, 0.0).astype(np.float32)
        engine = igac_engine.IGACEngine(
            image,
            initial,
            (1.0, 1.0, 1.0),
            device="cpu",
            workspace_margin_mm=7.0,
            max_roi_voxels=100000,
            parameters={
                "range_mm": 1.5,
                "smooth": 5.0,
                "data_weight": 20.0,
                "time_step": 0.05,
                "grow": 0.0,
                "max_displacement_mm": 5.0,
            },
        )
        initial_dice = self._dice(initial, target)
        engine.step(15)
        result = engine.result_xyz()
        self.assertGreater(self._dice(result, target), 0.98)
        self.assertGreater(self._dice(result, target), initial_dice + 0.5)
        self.assertEqual(int(np.logical_and(result, ~target).sum()), 0)

    def test_movement_shell_and_manual_guidance_have_explicit_precedence(self):
        shape = (32, 32, 24)
        x, y, z = np.indices(shape)
        radius = np.sqrt((x - 16) ** 2 + (y - 16) ** 2 + (z - 12) ** 2)
        initial = radius <= 5
        image = np.clip(1.0 - radius / 20.0, 0.0, 1.0).astype(np.float32)
        engine = igac_engine.IGACEngine(
            image,
            initial,
            (1.0, 1.0, 1.0),
            device="cpu",
            workspace_margin_mm=12.0,
            max_roi_voxels=100000,
            parameters={"max_displacement_mm": 3.0},
        )
        with engine.torch.inference_mode():
            engine.phi.fill_(-1.0)
            engine._apply_constraints()
        self.assertFalse(bool(engine.result_xyz()[27, 16, 12]))

        with engine.torch.inference_mode():
            engine.phi.fill_(1.0)
            engine._apply_constraints()
        self.assertTrue(bool(engine.result_xyz()[16, 16, 12]))

        self.assertTrue(engine.apply_brush("add", "axial", 12, 27, 16, 1.0))
        with engine.torch.inference_mode():
            engine._apply_constraints()
        self.assertTrue(bool(engine.result_xyz()[27, 16, 12]))

        self.assertTrue(engine.apply_brush("barrier", "axial", 12, 16, 16, 1.0))
        with engine.torch.inference_mode():
            engine._apply_constraints()
        self.assertFalse(bool(engine.result_xyz()[16, 16, 12]))

    def test_brush_segment_is_a_continuous_3d_capsule(self):
        shape = (28, 24, 16)
        x, y, z = np.indices(shape)
        image = (x + y + z).astype(np.float32)
        initial = ((x - 6) ** 2 + (y - 12) ** 2 + (z - 8) ** 2) <= 2 ** 2
        engine = igac_engine.IGACEngine(
            image,
            initial,
            (0.8, 0.8, 1.6),
            device="cpu",
            workspace_margin_mm=20.0,
            max_roi_voxels=100000,
        )
        self.assertTrue(
            engine.apply_brush_segment("add", "axial", 8, 4, 12, 23, 12, 0.6)
        )
        result = engine.result_xyz()
        self.assertTrue(bool(np.all(result[4:24, 12, 8])))

    def test_single_action_undo_restores_phi_constraints_and_iteration(self):
        shape = (24, 24, 12)
        x, y, z = np.indices(shape)
        initial = ((x - 12) ** 2 + (y - 12) ** 2 + (z - 6) ** 2) <= 4 ** 2
        engine = igac_engine.IGACEngine(
            initial.astype(np.float32),
            initial,
            (1.0, 1.0, 1.0),
            device="cpu",
            workspace_margin_mm=8.0,
            max_roi_voxels=100000,
        )
        state = engine.capture_state()
        before = engine.result_xyz().copy()
        self.assertTrue(engine.apply_brush("add", "axial", 6, 19, 12, 1.5))
        engine.step(1)
        self.assertFalse(np.array_equal(before, engine.result_xyz()))
        self.assertGreater(engine.iteration, 0)
        engine.restore_state(state)
        self.assertEqual(engine.iteration, 0)
        self.assertTrue(np.array_equal(before, engine.result_xyz()))
        self.assertEqual(int(engine.add_constraint.sum().item()), 0)
        self.assertEqual(int(engine.barrier_constraint.sum().item()), 0)

    def test_boundary_pull_is_local_anisotropic_and_recomputed_from_mouse_down_state(self):
        shape = (40, 40, 32)
        x, y, z = np.indices(shape)
        radius = np.sqrt((x - 20.0) ** 2 + (y - 20.0) ** 2 + (z - 16.0) ** 2)
        initial = radius <= 7.0
        image = np.where(radius <= 10.0, 1.0, 0.0).astype(np.float32)
        engine = igac_engine.IGACEngine(
            image,
            initial,
            (0.7, 0.7, 1.5),
            device="cpu",
            workspace_margin_mm=16.0,
            max_roi_voxels=500000,
            parameters={"max_displacement_mm": 20.0},
        )
        base = engine.capture_state()
        common = {
            "base_state": base,
            "orientation": "axial",
            "slice_index": 16,
            "start_u": 27.5,
            "start_v": 20.5,
            "influence_radius_mm": 12.0,
            "maximum_drag_ratio": 0.65,
            "anchor_radius_mm": 0.7,
        }
        first = engine.apply_boundary_pull(end_u=31.5, end_v=20.5, **common)
        first_mask = engine.result_xyz().copy()
        self.assertIsNotNone(first)
        self.assertAlmostEqual(first["drag_distance_mm"], 2.8, places=4)
        self.assertTrue(bool(first_mask[31, 20, 16]))
        self.assertTrue(np.array_equal(first_mask[:8], initial[:8]))

        engine.apply_boundary_pull(end_u=34.5, end_v=20.5, **common)
        self.assertTrue(bool(engine.result_xyz()[34, 20, 16]))
        engine.apply_boundary_pull(end_u=31.5, end_v=20.5, **common)
        self.assertTrue(np.array_equal(engine.result_xyz(), first_mask))

        engine.restore_state(base)
        self.assertTrue(np.array_equal(engine.result_xyz(), initial))

    def test_boundary_drag_returning_to_start_preserves_the_previous_undo(self):
        if importlib.util.find_spec("PySide6") is None:
            self.skipTest("PySide6 is unavailable")
        from tools.igac_gui import EngineThread

        shape = (28, 28, 20)
        x, y, z = np.indices(shape)
        initial = ((x - 14) ** 2 + (y - 14) ** 2 + (z - 10) ** 2) <= 5 ** 2
        engine = igac_engine.IGACEngine(
            initial.astype(np.float32),
            initial,
            (1.0, 1.0, 1.0),
            device="cpu",
            workspace_margin_mm=10.0,
            max_roi_voxels=100000,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            thread = EngineThread(
                {
                    "shape": list(shape),
                    "igac": {},
                    "status_path": str(root / "status.json"),
                    "log_path": str(root / "igac.log"),
                }
            )
            thread.engine = engine
            previous_undo = engine.capture_state()
            thread._undo_state = previous_undo
            thread.submit("boundary_pull_start")
            self.assertTrue(thread._process_commands())
            common = {
                "orientation": "axial",
                "slice_index": 10,
                "start_u": 19.5,
                "start_v": 14.5,
                "end_v": 14.5,
                "influence_radius_mm": 10.0,
                "maximum_drag_ratio": 0.65,
                "anchor_radius_mm": 0.8,
            }
            thread.submit("boundary_pull_update", dict(common, end_u=23.5))
            self.assertTrue(thread._process_commands())
            self.assertTrue(thread._boundary_changed)
            self.assertFalse(thread.running)
            thread.submit("boundary_pull_update", dict(common, end_u=19.5))
            self.assertTrue(thread._process_commands())
            self.assertFalse(thread._boundary_changed)
            self.assertTrue(np.array_equal(engine.result_xyz(), initial))
            thread.submit("boundary_pull_end")
            self.assertTrue(thread._process_commands())
            self.assertIs(thread._undo_state, previous_undo)
            self.assertFalse(thread.running)

    def test_reclassified_stroke_restores_then_replays_only_the_new_constraint(self):
        if importlib.util.find_spec("PySide6") is None:
            self.skipTest("PySide6 is unavailable")
        from tools.igac_gui import EngineThread

        shape = (24, 24, 12)
        x, y, z = np.indices(shape)
        initial = ((x - 12) ** 2 + (y - 12) ** 2 + (z - 6) ** 2) <= 4 ** 2
        engine = igac_engine.IGACEngine(
            initial.astype(np.float32),
            initial,
            (1.0, 1.0, 1.0),
            device="cpu",
            workspace_margin_mm=8.0,
            max_roi_voxels=100000,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            thread = EngineThread(
                {
                    "shape": list(shape),
                    "igac": {},
                    "status_path": str(root / "status.json"),
                    "log_path": str(root / "igac.log"),
                }
            )
            thread.engine = engine
            thread._undo_state = engine.capture_state()
            self.assertTrue(engine.apply_brush("barrier", "axial", 6, 12, 12, 1.5))
            self.assertGreater(int(engine.barrier_constraint.sum().item()), 0)
            thread.submit(
                "reclassify_stroke",
                [
                    {
                        "mode": "add",
                        "orientation": "axial",
                        "slice_index": 6,
                        "start_u": 12.0,
                        "start_v": 12.0,
                        "end_u": 18.0,
                        "end_v": 12.0,
                        "radius_mm": 1.5,
                    }
                ],
            )
            self.assertTrue(thread._process_commands())
            self.assertEqual(int(engine.barrier_constraint.sum().item()), 0)
            self.assertGreater(int(engine.add_constraint.sum().item()), 0)
            self.assertTrue(thread.running)

    def test_convergence_monitor_requires_stability_after_minimum_iterations(self):
        monitor = igac_engine.IGACConvergenceMonitor(
            minimum_iterations=3,
            maximum_iterations=20,
            stable_cycles_required=2,
            changed_voxel_ratio=0.001,
            maximum_seconds=10.0,
        )
        self.assertIsNone(
            monitor.observe(iterations=1, elapsed_seconds=0.1, changed_voxel_ratio=0.0)
        )
        self.assertIsNone(
            monitor.observe(iterations=2, elapsed_seconds=0.2, changed_voxel_ratio=0.01)
        )
        self.assertIsNone(
            monitor.observe(iterations=3, elapsed_seconds=0.3, changed_voxel_ratio=0.0)
        )
        self.assertEqual(
            monitor.observe(iterations=4, elapsed_seconds=0.4, changed_voxel_ratio=0.0),
            "converged",
        )

    def test_convergence_monitor_reports_safety_limit(self):
        monitor = igac_engine.IGACConvergenceMonitor(
            minimum_iterations=2,
            maximum_iterations=3,
            stable_cycles_required=3,
            changed_voxel_ratio=0.0,
            maximum_seconds=10.0,
        )
        self.assertIsNone(
            monitor.observe(iterations=1, elapsed_seconds=0.1, changed_voxel_ratio=0.2)
        )
        self.assertEqual(
            monitor.observe(iterations=3, elapsed_seconds=0.3, changed_voxel_ratio=0.2),
            "iteration_limit",
        )


@unittest.skipUnless(
    importlib.util.find_spec("SimpleITK") is not None
    and importlib.util.find_spec("scipy") is not None,
    "SimpleITK/scipy are not installed in this test interpreter",
)
class ITKSnakeIntegrationTests(unittest.TestCase):
    def test_recursive_gaussian_sigma_uses_physical_spacing(self):
        import SimpleITK as sitk

        step = np.zeros((25, 25, 25), dtype=np.float32)
        step[:, :, 12:] = 1.0

        def gradient(spacing):
            image = sitk.GetImageFromArray(step)
            image.SetSpacing(spacing)
            result = sitk.GradientMagnitudeRecursiveGaussian(image, sigma=1.0)
            return sitk.GetArrayFromImage(result)

        isotropic = gradient((1.0, 1.0, 1.0))
        anisotropic = gradient((2.0, 1.0, 1.0))
        self.assertFalse(np.allclose(isotropic, anisotropic))

    def test_synthetic_sphere_expands_to_edge_without_remote_leakage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shape = (48, 48, 48)
            grid = np.indices(shape)
            radius = np.sqrt(sum((grid[axis] - 24.0) ** 2 for axis in range(3)))
            image = np.where(radius <= 12, 1000.0, 0.0).astype(np.float32)
            initial = (radius <= 10).astype(np.uint8)
            image.tofile(root / "image.raw")
            initial.tofile(root / "mask.u8")
            request = {
                "tool": "itk_snake",
                "display_name": "ITK Snake",
                "shape": list(shape),
                "spacing_mm": [1.0, 1.0, 1.0],
                "image_path": str(root / "image.raw"),
                "image_dtype": "float32",
                "mask_path": str(root / "mask.u8"),
                "status_path": str(root / "status.json"),
                "cancel_path": str(root / "cancel.requested"),
                "log_path": str(root / "worker.log"),
                "result_path": str(root / "result.u8"),
                "parameters": {
                    "maximum_displacement_mm": 5.0,
                    "gradient_sigma_mm": 1.0,
                    "maximum_iterations": 80,
                    "propagation_scaling": 0.7,
                    "curvature_scaling": 0.8,
                    "advection_scaling": 1.0,
                    "maximum_volume_change_ratio": 1.5,
                },
            }
            request_path = root / "request.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            self.assertEqual(worker.run(request_path), 0)
            status = json.loads((root / "status.json").read_text(encoding="utf-8"))
            result = np.fromfile(root / "result.u8", dtype=np.uint8).reshape(shape)
            self.assertEqual(status["status"], "completed")
            self.assertGreater(int(result.sum()), int(initial.sum()))
            self.assertTrue(np.all(result[radius > 14] == 0))


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None
    and (
        ROOT
        / "external"
        / "ScribblePrompt"
        / "checkpoints"
        / "ScribblePrompt_unet_v1_nf192_res128.pt"
    ).is_file(),
    "PyTorch or the official ScribblePrompt checkpoint is unavailable",
)
class ScribblePromptIntegrationTests(unittest.TestCase):
    def test_official_checkpoint_runs_and_only_changes_prompted_slice(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shape = (32, 40, 8)
            x, y, _z = np.indices(shape)
            image = np.exp(-((x - 16) ** 2 + (y - 20) ** 2) / 80.0).astype(np.float32)
            base = np.zeros(shape, dtype=np.uint8)
            foreground = np.ones((3, 3, 1), dtype=np.uint8)
            image.tofile(root / "image.raw")
            base.tofile(root / "mask.u8")
            foreground.tofile(root / "foreground.u8")
            request = {
                "tool": "scribbleprompt",
                "display_name": "ScribblePrompt",
                "shape": list(shape),
                "spacing_mm": [1.0, 1.0, 2.0],
                "image_path": str(root / "image.raw"),
                "image_dtype": "float32",
                "mask_path": str(root / "mask.u8"),
                "status_path": str(root / "status.json"),
                "cancel_path": str(root / "cancel.requested"),
                "log_path": str(root / "worker.log"),
                "result_path": str(root / "result.u8"),
                "logits_result_path": str(root / "logits.f32"),
                "checkpoint_path": str(
                    ROOT
                    / "external"
                    / "ScribblePrompt"
                    / "checkpoints"
                    / "ScribblePrompt_unet_v1_nf192_res128.pt"
                ),
                "device": "cpu",
                "input_size": 128,
                "prior_logit_magnitude": 6.0,
                "scribbles": [
                    {
                        "path": str(root / "foreground.u8"),
                        "shape": [3, 3, 1],
                        "bbox": [[15, 18], [19, 22], [3, 4]],
                        "include": True,
                    }
                ],
            }
            request_path = root / "request.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            self.assertEqual(worker.run(request_path), 0)
            status = json.loads((root / "status.json").read_text(encoding="utf-8"))
            result = np.fromfile(root / "result.u8", dtype=np.uint8).reshape(shape)
            self.assertEqual(status["status"], "completed")
            self.assertEqual(int(result[:, :, :3].sum()), 0)
            self.assertEqual(int(result[:, :, 4:].sum()), 0)
            self.assertEqual((root / "logits.f32").stat().st_size, 32 * 40 * 4)


class MimicsRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.original_mimics = sys.modules.get("mimics")
        self.fake = FakeMimics()
        sys.modules["mimics"] = self.fake
        for name in ("nninteractive_mimics", "interactive_algorithms_mimics"):
            sys.modules.pop(name, None)
        self.runtime = importlib.import_module("interactive_algorithms_mimics")

    def tearDown(self):
        for name in ("nninteractive_mimics", "interactive_algorithms_mimics"):
            sys.modules.pop(name, None)
        if self.original_mimics is None:
            sys.modules.pop("mimics", None)
        else:
            sys.modules["mimics"] = self.original_mimics

    def test_selected_mask_contract(self):
        image = FakeImage(_u8_pattern_buffer((4, 5, 6), lambda x, y, z: x > 1))
        mask = FakeMask(
            "Target",
            image=image,
            array=_u8_pattern_buffer((4, 5, 6), lambda x, y, z: x == 2),
            selected=True,
        )
        self.fake.data.images.append(image)
        self.fake.data.images.set_active(image)
        self.fake.data.masks.append(mask)
        self.assertIs(self.runtime._selected_mask(image, True), mask)

    def test_buffer_streaming_preserves_bytes_and_hash(self):
        image = FakeImage(_u8_pattern_buffer((7, 9, 5), lambda x, y, z: (x + y + z) % 2 == 0))
        expected = image.get_voxel_buffer().tobytes()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "image.raw"
            result = self.runtime._write_raw_buffer(image, str(path), compute_sha=True)
            self.assertEqual(path.read_bytes(), expected)
            self.assertEqual(result["sha256"], __import__("hashlib").sha256(expected).hexdigest())

    def test_igac_captures_current_mimics_contrast_in_gv(self):
        self.fake.view.contrast = ((125.0, 0.0), (875.0, 1.0))
        self.assertEqual(self.runtime._current_display_contrast_gv(), [125.0, 875.0])

    def test_stale_scribble_logits_are_not_reused(self):
        image = FakeImage(_u8_pattern_buffer((2, 3, 4), lambda x, y, z: False))
        mask = FakeMask("Target", image=image, selected=True)
        with tempfile.TemporaryDirectory() as temporary:
            logits = Path(temporary) / "logits.f32"
            np.zeros((2, 3), dtype=np.float32).tofile(logits)
            self.runtime.nnm._metadata_set(mask, self.runtime.SESSION_PATH_METADATA, str(logits))
            self.runtime.nnm._metadata_set(mask, self.runtime.SESSION_MASK_SHA_METADATA, "old")
            self.assertEqual(self.runtime._session_values(mask, "new"), {})

    def test_result_copy_name_keeps_algorithm(self):
        image = FakeImage(_u8_pattern_buffer((2, 3, 4), lambda x, y, z: False))
        self.fake.data.images.append(image)
        self.fake.data.images.set_active(image)
        result = self.runtime._create_copy(image, "Liver - ScribblePrompt")
        self.assertEqual(result.name, "Liver - ScribblePrompt")

    def test_completed_result_updates_unchanged_target(self):
        image = FakeImage(_u8_pattern_buffer((2, 3, 4), lambda x, y, z: False))
        mask = FakeMask("Target", image=image, selected=True)
        self.fake.data.images.append(image)
        self.fake.data.images.set_active(image)
        self.fake.data.masks.append(mask)
        with tempfile.TemporaryDirectory() as temporary:
            result_path = Path(temporary) / "result.u8"
            expected = _u8_pattern_buffer((2, 3, 4), lambda x, y, z: x == 1)
            result_path.write_bytes(expected.tobytes())
            monitor = {
                "action": self.runtime.ACTION_ITK_SNAKE,
                "display_name": "ITK Snake",
                "image": image,
                "target_guid": self.runtime.nnm._object_id(mask),
                "target_name": "Target",
                "base_sha256": self.runtime._current_mask_sha(mask),
                "request": {"shape": [2, 3, 4]},
            }
            self.runtime._apply_completed(
                monitor,
                {
                    "elapsed_seconds": 1.2,
                    "result": {
                        "result_path": str(result_path),
                        "shape": [2, 3, 4],
                        "foreground_voxels": 12,
                    },
                },
            )
        self.assertEqual(mask.get_voxel_buffer().tobytes(), expected.tobytes())
        self.assertEqual(self.fake.dialogs.questions[-1]["title"], "ITK Snake Ready")

    def test_changed_target_is_preserved_in_editable_copy(self):
        image = FakeImage(_u8_pattern_buffer((2, 3, 4), lambda x, y, z: False))
        mask = FakeMask("Target", image=image, selected=True)
        self.fake.data.images.append(image)
        self.fake.data.images.set_active(image)
        self.fake.data.masks.append(mask)
        original = mask.get_voxel_buffer().tobytes()
        with tempfile.TemporaryDirectory() as temporary:
            result_path = Path(temporary) / "result.u8"
            expected = _u8_pattern_buffer((2, 3, 4), lambda x, y, z: z == 2)
            result_path.write_bytes(expected.tobytes())
            monitor = {
                "action": self.runtime.ACTION_SCRIBBLEPROMPT,
                "display_name": "ScribblePrompt",
                "image": image,
                "target_guid": self.runtime.nnm._object_id(mask),
                "target_name": "Target",
                "base_sha256": "stale-sha",
                "request": {"shape": [2, 3, 4]},
            }
            self.runtime._apply_completed(
                monitor,
                {
                    "result": {
                        "result_path": str(result_path),
                        "shape": [2, 3, 4],
                        "foreground_voxels": 6,
                    },
                },
            )
        self.assertEqual(mask.get_voxel_buffer().tobytes(), original)
        copies = [value for value in self.fake.data.masks if value is not mask]
        self.assertEqual(len(copies), 1)
        self.assertEqual(copies[0].name, "Target - ScribblePrompt")
        self.assertEqual(copies[0].get_voxel_buffer().tobytes(), expected.tobytes())

    def test_igac_uses_external_destination_without_another_mimics_dialog(self):
        image = FakeImage(_u8_pattern_buffer((2, 3, 4), lambda x, y, z: False))
        mask = FakeMask("Target", image=image, selected=True)
        self.fake.data.images.append(image)
        self.fake.data.images.set_active(image)
        self.fake.data.masks.append(mask)
        monitor = {
            "action": self.runtime.ACTION_IGAC,
            "display_name": "IGAC",
            "image": image,
            "target_guid": self.runtime.nnm._object_id(mask),
            "target_name": "Target",
            "base_sha256": self.runtime._current_mask_sha(mask),
        }
        before = len(self.fake.dialogs.questions)
        target = self.runtime._choose_result_target(monitor, {"apply_mode": "update"})
        self.assertIs(target, mask)
        self.assertEqual(len(self.fake.dialogs.questions), before)
        copy = self.runtime._choose_result_target(monitor, {"apply_mode": "copy"})
        self.assertIsNot(copy, mask)
        self.assertEqual(copy.name, "Target - IGAC")

    def test_igac_never_overwrites_a_mask_changed_during_external_editing(self):
        image = FakeImage(_u8_pattern_buffer((2, 3, 4), lambda x, y, z: False))
        mask = FakeMask("Target", image=image, selected=True)
        self.fake.data.images.append(image)
        self.fake.data.images.set_active(image)
        self.fake.data.masks.append(mask)
        monitor = {
            "action": self.runtime.ACTION_IGAC,
            "display_name": "IGAC",
            "image": image,
            "target_guid": self.runtime.nnm._object_id(mask),
            "target_name": "Target",
            "base_sha256": "stale",
        }
        target = self.runtime._choose_result_target(monitor, {"apply_mode": "update"})
        self.assertIsNot(target, mask)
        self.assertEqual(target.name, "Target - IGAC")


if __name__ == "__main__":
    unittest.main()
