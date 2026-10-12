#!/usr/bin/env python3
"""Focused tests for ScribblePrompt integration."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
for value in (ROOT, TESTS, ROOT / "tools", ROOT / "runtime_py35"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import interactive_algorithms_worker as worker
from fake_mimics_flow_test import (
    FakeImage,
    FakeMask,
    FakeMimics,
    FakeVoxelBuffer,
    _u8_pattern_buffer,
)


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

    def test_requested_prompt_plane_accepts_thin_in_plane_stroke(self):
        specs = [{"bbox": [[2, 6], [3, 4], [4, 5]]}]
        self.assertEqual(
            worker._prompt_plane(
                specs,
                (12, 13, 14),
                requested_axis=2,
                requested_index=4,
            ),
            (2, 4),
        )

    def test_click_channels_preserve_sign_and_xyz_to_xy_mapping(self):
        points = [
            {"point": [4, 6, 3], "include": True},
            {"point": [8, 2, 3], "include": False},
        ]
        channels = worker._point_channels(
            points,
            (10, 12, 7),
            axis=2,
            index=3,
            output_shape=(20, 24),
        )
        self.assertEqual(channels.shape, (2, 20, 24))
        self.assertEqual(float(channels[0, 8, 12]), 1.0)
        self.assertEqual(float(channels[1, 16, 4]), 1.0)
        self.assertEqual(float(channels.sum()), 2.0)

    def test_box_channel_uses_foreground_shaded_rectangle(self):
        boxes = [{"bbox": [[2, 6], [3, 8], [4, 5]]}]
        channel = worker._box_channel(
            boxes,
            (10, 12, 7),
            axis=2,
            index=4,
            output_shape=(20, 24),
        )
        self.assertEqual(channel.shape, (1, 20, 24))
        self.assertEqual(float(channel.sum()), 80.0)
        self.assertTrue(np.all(channel[0, 4:12, 6:16] == 1.0))

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


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None
    and (
        ROOT
        / "integrations"
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
                    / "integrations"
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

    def test_official_checkpoint_accepts_clicks_box_and_explicit_plane(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shape = (32, 40, 8)
            x, y, _z = np.indices(shape)
            image = np.exp(-((x - 16) ** 2 + (y - 20) ** 2) / 80.0).astype(np.float32)
            base = np.zeros(shape, dtype=np.uint8)
            image.tofile(root / "image.raw")
            base.tofile(root / "mask.u8")
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
                    / "integrations"
                    / "ScribblePrompt"
                    / "checkpoints"
                    / "ScribblePrompt_unet_v1_nf192_res128.pt"
                ),
                "device": "cpu",
                "input_size": 128,
                "prior_logit_magnitude": 6.0,
                "prompt_plane_axis": 2,
                "prompt_plane_index": 3,
                "scribbles": [],
                "points": [
                    {"point": [16, 20, 3], "include": True},
                    {"point": [4, 4, 3], "include": False},
                ],
                "boxes": [
                    {"bbox": [[10, 23], [12, 29], [3, 4]]}
                ],
            }
            request_path = root / "request.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            self.assertEqual(worker.run(request_path), 0)
            status = json.loads((root / "status.json").read_text(encoding="utf-8"))
            result = np.fromfile(root / "result.u8", dtype=np.uint8).reshape(shape)
            self.assertEqual(status["status"], "completed")
            self.assertEqual(status["result"]["prompt_counts"], {"points": 2, "scribbles": 0, "boxes": 1})
            self.assertEqual(int(result[:, :, :3].sum()), 0)
            self.assertEqual(int(result[:, :, 4:].sum()), 0)


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
        self.assertIs(self.runtime._selected_mask(image), mask)

    def test_buffer_streaming_preserves_bytes_and_hash(self):
        image = FakeImage(_u8_pattern_buffer((7, 9, 5), lambda x, y, z: (x + y + z) % 2 == 0))
        expected = image.get_voxel_buffer().tobytes()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "image.raw"
            result = self.runtime._write_raw_buffer(image, str(path), compute_sha=True)
            self.assertEqual(path.read_bytes(), expected)
            self.assertEqual(
                result["sha256"],
                "sha256:" + __import__("hashlib").sha256(expected).hexdigest(),
            )

    def test_session_survives_prefixed_base_sha(self):
        """R61-12 regression: stream_buffer prefixes 'sha256:' but the worker
        stores a bare hex digest; the session comparison must normalize or
        previous-logits continuity is silently dropped every prompt."""
        image = FakeImage(_u8_pattern_buffer((2, 3, 4), lambda x, y, z: False))
        mask = FakeMask("Target", image=image, selected=True)
        with tempfile.TemporaryDirectory() as temporary:
            logits = Path(temporary) / "logits.f32"
            np.zeros((2, 3), dtype=np.float32).tofile(logits)
            shape = json.dumps([2, 3])
            self.runtime.nnm._metadata_set(mask, self.runtime.SESSION_PATH_METADATA, str(logits))
            self.runtime.nnm._metadata_set(mask, self.runtime.SESSION_MASK_SHA_METADATA, "abc123")
            self.runtime.nnm._metadata_set(mask, self.runtime.SESSION_SHAPE_METADATA, shape)
            self.runtime.nnm._metadata_set(mask, self.runtime.SESSION_AXIS_METADATA, "2")
            self.runtime.nnm._metadata_set(mask, self.runtime.SESSION_INDEX_METADATA, "5")
            values = self.runtime._session_values(mask, "sha256:abc123")
            self.assertEqual(values.get("previous_logits_path"), str(logits))
            self.assertEqual(values.get("previous_plane_index"), 5)

    def test_click_plane_labels_are_derived_from_voxel_to_ras(self):
        image = FakeImage(_u8_pattern_buffer((4, 5, 6), lambda x, y, z: False))
        self.assertEqual(
            self.runtime._click_plane_axes(image),
            {"轴位视图": 2, "冠状位视图": 1, "矢状位视图": 0},
        )

    def test_stale_scribble_logits_are_not_reused(self):
        image = FakeImage(_u8_pattern_buffer((2, 3, 4), lambda x, y, z: False))
        mask = FakeMask("Target", image=image, selected=True)
        with tempfile.TemporaryDirectory() as temporary:
            logits = Path(temporary) / "logits.f32"
            np.zeros((2, 3), dtype=np.float32).tofile(logits)
            self.runtime.nnm._metadata_set(mask, self.runtime.SESSION_PATH_METADATA, str(logits))
            self.runtime.nnm._metadata_set(mask, self.runtime.SESSION_MASK_SHA_METADATA, "old")
            self.assertEqual(self.runtime._session_values(mask, "new"), {})

    def test_scribble_capture_avoids_full_volume_copy(self):
        # R62-1: every drawn stroke used to copy the whole image volume
        # via .tobytes() before finding its bbox. The numpy path must view
        # the buffer zero-copy (np.asarray) and copy only the bbox crop.
        class BufferProtocolVoxelBuffer(FakeVoxelBuffer):
            """FakeVoxelBuffer plus the buffer protocol, which real Mimics
            voxel buffers support and np.asarray needs for a zero-copy view."""

            def __buffer__(self, flags):
                return self._data.__buffer__(flags)

        image = FakeImage(_u8_pattern_buffer((4, 5, 6), lambda x, y, z: False))
        self.fake.data.images.append(image)
        self.fake.data.images.set_active(image)
        original_tobytes = FakeVoxelBuffer.tobytes
        volume_bytes = bytearray(_u8_pattern_buffer(
            (4, 5, 6), lambda x, y, z: False).tobytes())
        volume_bytes[2 * 5 * 6 + 3 * 6 + 4] = 1  # stroke voxel at (2, 3, 4)
        copied_sizes = []

        def counting_tobytes(inner):
            copied_sizes.append(len(inner._data))
            return original_tobytes(inner)

        with tempfile.TemporaryDirectory() as job_dir:
            os.makedirs(os.path.join(job_dir, "inputs"))

            def create_mask():
                mask = FakeMask(
                    "Scribble Foreground Scribble", image=image,
                    array=BufferProtocolVoxelBuffer(
                        (4, 5, 6), bytes(volume_bytes), "B"))
                self.fake.data.masks.append(mask)
                return mask

            self.fake.segment.create_mask = create_mask
            with mock.patch.object(FakeVoxelBuffer, "tobytes", counting_tobytes):
                value = self.runtime._capture_scribble(
                    image, True, job_dir, [])
            self.assertIsNotNone(value)
            self.assertEqual(value["shape"], [1, 1, 1])
            self.assertEqual(value["bbox"], [[2, 3], [3, 4], [4, 5]])
            with open(value["path"], "rb") as handle:
                self.assertEqual(handle.read(), b"\x01")
        # The single full-volume read is the fake's own bookkeeping
        # (_refresh_pixels inside FakeMask.__init__); the production code
        # must contribute only crop-sized copies. Pre-fix, the numpy path
        # added a second full-volume copy (4*5*6) per drawn stroke.
        self.assertEqual(copied_sizes, [4 * 5 * 6], copied_sizes)

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
                "action": self.runtime.ACTION_SCRIBBLEPROMPT,
                "display_name": "ScribblePrompt",
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
        self.assertEqual(self.fake.dialogs.questions[-1]["title"], "ScribblePrompt 就绪")


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

    def _monitor_for(self, image, mask, base_sha, temporary_shape=(2, 3, 4)):
        return {
            "action": self.runtime.ACTION_SCRIBBLEPROMPT,
            "display_name": "ScribblePrompt",
            "image": image,
            "target_guid": self.runtime.nnm._object_id(mask),
            "target_name": "Target",
            "base_sha256": base_sha,
            "request": {"shape": list(temporary_shape)},
        }

    def test_result_destination_is_asked_once_per_session(self):
        # R62-3: back-to-back rounds used to pop the ready dialog after every
        # completion. The first answer must stick for the whole Mimics
        # session (module reload = new session, as in Mimics restart).
        image = FakeImage(_u8_pattern_buffer((2, 3, 4), lambda x, y, z: False))
        mask = FakeMask("Target", image=image, selected=True)
        self.fake.data.images.append(image)
        self.fake.data.images.set_active(image)
        self.fake.data.masks.append(mask)
        base_sha = self.runtime._current_mask_sha(mask)
        with tempfile.TemporaryDirectory() as temporary:
            result_path = Path(temporary) / "result.u8"
            expected = _u8_pattern_buffer((2, 3, 4), lambda x, y, z: x == 1)
            result_path.write_bytes(expected.tobytes())
            monitor = self._monitor_for(image, mask, base_sha)
            state = {"elapsed_seconds": 1.0, "result": {
                "result_path": str(result_path),
                "shape": [2, 3, 4],
                "foreground_voxels": 12,
            }}
            # First completion: dialog shown, answered Update Selected Mask.
            self.fake.dialogs.question_answers = ["更新匹配掩膜"]
            self.fake.dialogs.questions = []
            self.runtime._apply_completed(monitor, dict(state))
            self.assertEqual(len(self.fake.dialogs.questions), 1)
            self.assertIn(
                "该选择将在本次 Mimics 会话中记住",
                self.fake.dialogs.questions[0]["message"],
            )
            # Second completion: no dialog, same destination applied.
            self.fake.dialogs.questions = []
            mask.set_voxel_buffer(_u8_pattern_buffer(
                (2, 3, 4), lambda x, y, z: False))
            self.runtime._apply_completed(monitor, dict(state))
            self.assertEqual(self.fake.dialogs.questions, [])
            self.assertEqual(
                mask.get_voxel_buffer().tobytes(), expected.tobytes())

    def test_result_destination_changed_mask_still_asks(self):
        # The session memory must not swallow the changed-mask safety
        # question: applying onto a mask the user edited mid-run is a
        # data-risk path and always confirms.
        image = FakeImage(_u8_pattern_buffer((2, 3, 4), lambda x, y, z: False))
        mask = FakeMask("Target", image=image, selected=True)
        self.fake.data.images.append(image)
        self.fake.data.images.set_active(image)
        self.fake.data.masks.append(mask)
        with tempfile.TemporaryDirectory() as temporary:
            result_path = Path(temporary) / "result.u8"
            result_path.write_bytes(_u8_pattern_buffer(
                (2, 3, 4), lambda x, y, z: x == 1).tobytes())
            # First completion establishes session memory.
            self.fake.dialogs.question_answers = ["更新匹配掩膜"]
            self.runtime._apply_completed(
                self._monitor_for(
                    image, mask, self.runtime._current_mask_sha(mask)),
                {"elapsed_seconds": 1.0, "result": {
                    "result_path": str(result_path),
                    "shape": [2, 3, 4], "foreground_voxels": 12,
                }},
            )
            # Second completion with a stale base_sha (user edited the mask
            # while the round ran): the safety dialog must appear.
            self.fake.dialogs.questions = []
            self.runtime._apply_completed(
                self._monitor_for(image, mask, "stale-sha"),
                {"elapsed_seconds": 1.0, "result": {
                    "result_path": str(result_path),
                    "shape": [2, 3, 4], "foreground_voxels": 12,
                }},
            )
            self.assertEqual(len(self.fake.dialogs.questions), 1)
            self.assertIn(
                "在运行期间发生了改动",
                self.fake.dialogs.questions[0]["message"],
            )

    def test_prompt_sign_asked_once_per_session(self):
        # R62-2: Foreground/Background was asked before every click and
        # scribble, stacking a second modal per prompt. The first answer
        # must stick for the session.
        self.fake.dialogs.questions = []
        self.fake.dialogs.question_answers = ["背景"]
        self.assertFalse(self.runtime._choose_prompt_sign())
        asked = len(self.fake.dialogs.questions)
        self.assertEqual(asked, 1)
        # Subsequent prompts: no dialog, same sign.
        self.fake.dialogs.questions = []
        self.assertFalse(self.runtime._choose_prompt_sign())
        self.assertEqual(self.fake.dialogs.questions, [])
        # The main prompt dialog exposes the switch button.
        self.runtime._SESSION_PROMPT_SIGN = True
        self.assertTrue(self.runtime._current_prompt_sign())
        self.runtime._SESSION_PROMPT_SIGN = False
        self.assertFalse(self.runtime._current_prompt_sign())

    def test_result_mode_label_reflects_session_memory(self):
        # UI18: a remembered result destination must be visible — the label
        # distinguishes unset / in_place / copy, and the switch dialog can
        # change the session memory without a Mimics restart.
        try:
            self.assertEqual(self.runtime._result_mode_label(), "每轮询问")
            self.runtime._SESSION_RESULT_MODE = "in_place"
            self.assertEqual(self.runtime._result_mode_label(), "更新匹配掩膜")
            self.runtime._SESSION_RESULT_MODE = "copy"
            self.assertEqual(self.runtime._result_mode_label(), "创建可编辑副本")
        finally:
            self.runtime._SESSION_RESULT_MODE = None

    def test_prompt_dialog_shows_and_switches_result_mode(self):
        # UI18 验收：第二轮能看清当前方式并就地切换。进入提示收集循环，
        # 选择「结果写入方式…」→ 切到可编辑副本 → 取消退出。
        self.runtime._SESSION_RESULT_MODE = "in_place"
        self.fake.dialogs.questions = []
        # 循环顺序：主对话框（结果写入方式…）→ 切换对话框（创建可编辑副本）
        # → 主对话框（取消）
        self.fake.dialogs.question_answers = [
            "结果写入方式…", "创建可编辑副本", "取消"]
        image = FakeImage(_u8_pattern_buffer((2, 3, 4), lambda x, y, z: False))
        self.fake.data.images.append(image)
        self.fake.data.images.set_active(image)
        mask = FakeMask("Target", image=image, selected=True)
        self.fake.data.masks.append(mask)
        try:
            result = self.runtime._collect_scribbleprompt_prompts(
                image, mask, tempfile.mkdtemp(), [])
            self.assertIsNone(result)
            # 主对话框第一轮必须显示当前方式
            self.assertIn(
                "结果写入方式：更新匹配掩膜",
                self.fake.dialogs.questions[0]["message"])
            self.assertIn(
                "结果写入方式…", self.fake.dialogs.questions[0]["buttons"])
            # 切换对话框显示旧方式并记录新方式
            self.assertIn(
                "当前方式：更新匹配掩膜",
                self.fake.dialogs.questions[1]["message"])
            self.assertEqual(self.runtime._SESSION_RESULT_MODE, "copy")
        finally:
            self.runtime._SESSION_RESULT_MODE = None


if __name__ == "__main__":
    unittest.main()
