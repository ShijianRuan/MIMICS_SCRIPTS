#!/usr/bin/env python3
"""Fake-Mimics flow tests for Mimics-Script runtime modules.

This suite injects a small in-memory ``mimics`` module so the Mimics-side
runtime code can be exercised on machines without Mimics installed.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
import types
import uuid
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = ROOT / "runtime_py35"
LIBRARY_DIR = ROOT / "scripting_library"
for path in (str(RUNTIME_DIR), str(LIBRARY_DIR), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


class TestFailure(RuntimeError):
    pass


def assert_true(condition, message):
    if not condition:
        raise TestFailure(message)


def assert_equal(left, right, message):
    if left != right:
        raise TestFailure("{}: {!r} != {!r}".format(message, left, right))


class TestRunner:
    def __init__(self):
        self.results = []

    def run(self, name, func):
        started = time.time()
        try:
            detail = func()
        except Exception:
            self.results.append((name, "FAIL", traceback.format_exc()))
            return
        self.results.append((name, "PASS", "{} ({:.2f}s)".format(detail or "ok", time.time() - started)))

    def report(self):
        failed = 0
        print("\nFake Mimics flow test summary")
        print("=" * 72)
        for name, status, detail in self.results:
            print("[{}] {}".format(status, name))
            if status == "FAIL":
                failed += 1
                print(detail.rstrip())
            elif detail:
                print("  " + detail)
        print("=" * 72)
        print("{} passed, {} failed".format(len(self.results) - failed, failed))
        return 1 if failed else 0


class FakeMetadataItem:
    def __init__(self, name, value):
        self.name = name
        self.value = value


class FakeMetadata:
    def __init__(self):
        self._items = {}

    def find(self, name):
        return self._items.get(name)

    def create(self, name, value):
        item = FakeMetadataItem(name, value)
        self._items[name] = item
        return item

    def delete(self, name):
        self._items.pop(name, None)

    def __getitem__(self, name):
        return self._items[name]

    def set(self, name, value):
        item = self.find(name)
        if item is None:
            self.create(name, value)
        else:
            item.value = value


class FakeCollection(list):
    def __init__(self, values=None):
        super().__init__(values or [])
        self._active = self[0] if self else None
        self.deleted = []

    def get_active(self):
        return self._active

    def set_active(self, value):
        if value not in self:
            self.append(value)
        self._active = value

    def delete(self, value):
        self.deleted.append(value)
        try:
            self.remove(value)
        except ValueError:
            pass
        if self._active is value:
            self._active = self[0] if self else None


def _voxel_count(shape):
    count = 1
    for value in shape:
        count *= int(value)
    return count


class FakeVoxelBuffer:
    def __init__(self, shape, data=None, fmt="B"):
        self.shape = tuple(int(value) for value in shape)
        self.format = fmt
        item_size = 2 if fmt in ("h", "H") else 1
        expected = _voxel_count(self.shape) * item_size
        if data is None:
            data = bytes(expected)
        raw = bytes(data)
        if len(raw) != expected:
            raise ValueError("fake voxel buffer size mismatch: {} != {}".format(len(raw), expected))
        self._data = bytearray(raw)

    def tobytes(self):
        return bytes(self._data)

    def tostring(self):
        return self.tobytes()

    def __getitem__(self, index):
        x, y, z = [int(value) for value in index]
        offset = (x * self.shape[1] * self.shape[2]) + (y * self.shape[2]) + z
        return self._data[offset]

    def zero(self):
        self._data[:] = bytes(len(self._data))


def _u8_buffer(shape, fill=0):
    return FakeVoxelBuffer(shape, bytes([int(fill) & 0xFF]) * _voxel_count(shape), "B")


def _u8_pattern_buffer(shape, predicate):
    data = bytearray()
    x_dim, y_dim, z_dim = [int(value) for value in shape]
    for x in range(x_dim):
        for y in range(y_dim):
            for z in range(z_dim):
                data.append(1 if predicate(x, y, z) else 0)
    return FakeVoxelBuffer(shape, data, "B")


class FakeImage:
    def __init__(self, buffer, minimum_value=0, maximum_value=4095, name="Image"):
        self._buffer = buffer
        self.logical_dimensions = list(self._buffer.shape)
        self.minimum_value = minimum_value
        self.maximum_value = maximum_value
        self._contrast_minimum_value = minimum_value
        self._contrast_maximum_value = maximum_value
        self.name = name
        self.guid = uuid.uuid4().hex
        self.metadata = FakeMetadata()

    def get_voxel_buffer(self):
        return self._buffer

    def get_image_information(self):
        return SimpleNamespace(
            minimum_value=getattr(self, "minimum_value", None),
            maximum_value=getattr(self, "maximum_value", None),
        )

    def get_voxel_center(self, *args):
        if len(args) == 1:
            index = args[0]
        else:
            index = args
        x, y, z = [float(value) for value in index]
        # Mimics/DICOM patient coordinates are LPS. The default fake image is
        # equivalent to an identity RAS voxel grid, so LPS negates x and y.
        return [-x, -y, z]

    def get_voxel_indexes(self, point):
        x, y, z = [float(value) for value in point]
        idx = [int(round(-x)), int(round(-y)), int(round(z))]
        for axis, value in enumerate(idx):
            if value < 0 or value >= int(self.logical_dimensions[axis]):
                raise ValueError("point outside image")
        return idx


class FakeMask:
    def __init__(self, name, image=None, array=None, selected=False):
        if array is None and image is not None:
            array = _u8_buffer(tuple(image.logical_dimensions), 0)
        elif array is None:
            array = _u8_buffer((1, 1, 1), 0)
        self.name = name
        self.image = image
        if isinstance(array, FakeVoxelBuffer):
            self._buffer = array
        elif isinstance(array, (bytes, bytearray)):
            shape = tuple(getattr(image, "logical_dimensions", (len(array), 1, 1)))
            self._buffer = FakeVoxelBuffer(shape, array, "B")
        else:
            try:
                import numpy as np
                arr = np.asarray(array).astype(np.uint8)
                self._buffer = FakeVoxelBuffer(arr.shape, arr.tobytes(), "B")
            except Exception:
                raise TypeError("FakeMask array must be FakeVoxelBuffer, bytes, or array-like")
        self.selected = selected
        self.visible = True
        self.guid = uuid.uuid4().hex
        self.metadata = FakeMetadata()
        self._refresh_pixels()

    def _refresh_pixels(self):
        self.number_of_pixels = sum(1 for value in self._buffer.tobytes() if value)

    def get_voxel_buffer(self):
        return self._buffer

    def set_voxel_buffer(self, pixels):
        if hasattr(pixels, "tobytes"):
            raw = pixels.tobytes()
            shape = tuple(getattr(pixels, "shape", self._buffer.shape))
        else:
            raw = bytes(pixels)
            shape = self._buffer.shape
        if self.image is not None:
            expected_shape = tuple(int(value) for value in getattr(self.image, "logical_dimensions", ()))
            if expected_shape and tuple(int(value) for value in shape) != expected_shape:
                raise ValueError("Dimensions of input pixels do not coincide with mask region dimensions")
        raw = bytes(1 if value else 0 for value in raw)
        self._buffer = FakeVoxelBuffer(shape, raw, "B")
        self._refresh_pixels()

    def clear(self):
        self._buffer.zero()
        self._refresh_pixels()


class FakeDialogs:
    def __init__(self):
        self.messages = []
        self.questions = []
        self.question_answers = []

    def message_box(self, message=None, title="", ui_blocking=None, **kwargs):
        if message is None:
            message = kwargs.get("message", "")
        self.messages.append({
            "title": title or kwargs.get("title", ""),
            "message": message,
            "ui_blocking": ui_blocking,
        })
        return None

    def question_box(self, message="", buttons="", title="", ui_blocking=None, **kwargs):
        record = {
            "title": title or kwargs.get("title", ""),
            "message": message or kwargs.get("message", ""),
            "buttons": buttons or kwargs.get("buttons", ""),
            "ui_blocking": ui_blocking,
        }
        self.questions.append(record)
        if self.question_answers:
            return self.question_answers.pop(0)
        choices = [item for item in record["buttons"].split(";") if item]
        return choices[0] if choices else "OK"


class FakeLogging:
    def __init__(self):
        self.messages = []

    def log_user_message(self, level, message):
        self.messages.append((level, message))


class FakeView:
    def __init__(self, fake):
        self.fake = fake
        self.contrast = ((0.0, 0.0), (4095.0, 1.0))
        self.log_panel_shown = 0

    def set_contrast(self, low, high):
        image = self.fake.data.images.get_active()
        if image is not None:
            minimum = float(getattr(image, "_contrast_minimum_value", getattr(image, "minimum_value", -1e9)))
            maximum = float(getattr(image, "_contrast_maximum_value", getattr(image, "maximum_value", 1e9)))
            if float(low[0]) < minimum or float(high[0]) > maximum:
                if float(high[0]) > maximum:
                    raise ValueError("Upper contrast point should be in range from {0} to {1}".format(int(minimum), int(maximum)))
                raise ValueError("Lower contrast point should be in range from {0} to {1}".format(int(minimum), int(maximum)))
            if float(high[0]) <= float(low[0]):
                raise ValueError("upper contrast point must be greater than lower")
        self.contrast = (tuple(low), tuple(high))

    def get_contrast(self):
        return self.contrast

    def show_log_panel(self):
        self.log_panel_shown += 1


class FakeSegment:
    def __init__(self, fake):
        self.fake = fake

    def HU2GV(self, value):
        return float(value)

    def create_mask(self):
        image = self.fake.data.images.get_active()
        mask = FakeMask("New Mask", image=image)
        self.fake.data.masks.append(mask)
        return mask

    def activate_edit_mask(self, *args, **kwargs):
        self.fake.segment_edit_calls.append((args, kwargs))


class FakeFile:
    def __init__(self, fake):
        self.fake = fake
        self.project_path = ""
        self.opened = []
        self.saved = []
        self.closed = 0

    def get_project_information(self):
        return SimpleNamespace(filename=self.project_path)

    def open_project(self, filename=None, **kwargs):
        value = filename or kwargs.get("filename")
        self.opened.append(value)
        self.project_path = value or self.project_path

    def close_project(self):
        self.closed += 1

    def save_project(self, filename=None, save_as_type=None, **kwargs):
        self.saved.append((filename or kwargs.get("filename"), save_as_type))

    def import_dicom_images(self, source_folder=None, **kwargs):
        self.fake.imported_dicom.append(source_folder or kwargs.get("source_folder"))


class FakeMimics(types.ModuleType):
    def __init__(self):
        super().__init__("mimics")
        self.UserInterrupted = type("UserInterrupted", (Exception,), {})
        self.logging = FakeLogging()
        self.dialogs = FakeDialogs()
        self.data = SimpleNamespace(
            images=FakeCollection(),
            masks=FakeCollection(),
            points=FakeCollection(),
            distance_measurements=FakeCollection(),
            measurements=FakeCollection(),
            splines=FakeCollection(),
        )
        self.view = FakeView(self)
        self.segment = FakeSegment(self)
        self.file = FakeFile(self)
        self.segment_edit_calls = []
        self.imported_dicom = []
        self.indicated_coordinates = []
        self.gui_enabled = True
        self.update_gui_calls = 0
        self.disable_gui_calls = 0
        self.enable_gui_calls = 0
        self.analyze = SimpleNamespace(
            create_point=lambda *args, **kwargs: SimpleNamespace(name="point"),
            indicate_spline=lambda *args, **kwargs: SimpleNamespace(points=[]),
        )
        self.measure = SimpleNamespace(
            indicate_distance_measurement=lambda *args, **kwargs: None,
            get_bounding_box=self._get_bounding_box,
        )

    def _get_bounding_box(self, obj, *args, **kwargs):
        masks = list(obj) if isinstance(obj, (list, tuple, FakeCollection)) else [obj]
        points = []
        for mask in masks:
            buffer = getattr(mask, "_buffer", None)
            if buffer is None:
                continue
            sx, sy, sz = [int(value) for value in buffer.shape]
            raw = buffer.tobytes()
            for x in range(sx):
                for y in range(sy):
                    for z in range(sz):
                        offset = (x * sy * sz) + (y * sz) + z
                        if raw[offset]:
                            points.append((x, y, z))
        if not points:
            raise ValueError("empty object")
        xs = [item[0] for item in points]
        ys = [item[1] for item in points]
        zs = [item[2] for item in points]
        min_x = -max(xs) - 0.5
        max_x = -min(xs) + 0.5
        min_y = -max(ys) - 0.5
        max_y = -min(ys) + 0.5
        min_z = min(zs) - 0.5
        max_z = max(zs) + 0.5
        return SimpleNamespace(
            origin=(min_x, min_y, min_z),
            first_vector=(max_x - min_x, 0.0, 0.0),
            second_vector=(0.0, max_y - min_y, 0.0),
            third_vector=(0.0, 0.0, max_z - min_z),
        )

    def update_gui(self):
        self.update_gui_calls += 1

    def indicate_coordinate(self, *args, **kwargs):
        if self.indicated_coordinates:
            return self.indicated_coordinates.pop(0)
        raise self.UserInterrupted()

    def is_update_gui_enabled(self):
        return self.gui_enabled

    def disable_update_gui(self):
        self.disable_gui_calls += 1
        self.gui_enabled = False

    def enable_update_gui(self):
        self.enable_gui_calls += 1
        self.gui_enabled = True

    def reset_scene(self, image_shape=(2, 3, 4), minimum_value=0, maximum_value=4095):
        image = FakeImage(
            FakeVoxelBuffer(tuple(image_shape), bytes(_voxel_count(image_shape) * 2), "h"),
            minimum_value=minimum_value,
            maximum_value=maximum_value,
        )
        self.data.images = FakeCollection([image])
        self.data.images.set_active(image)
        self.data.masks = FakeCollection()
        self.view.contrast = ((float(minimum_value), 0.0), (float(maximum_value), 1.0))
        return image


def install_fake_mimics():
    fake = FakeMimics()
    sys.modules["mimics"] = fake
    return fake


def import_runtime_module(name):
    if name in sys.modules:
        return importlib.reload(sys.modules[name])
    return importlib.import_module(name)


def test_runtime_imports(fake, tmp):
    modules = [
        "window_level_mimics",
        "mimics_export",
        "mimics_import",
        "mimics_stop_background",
        "fewshot_mimics",
        "nninteractive_mimics",
        "nninteractive_finetune_mimics",
        "create_mcs_batch",
        "mask_identifier",
    ]
    loaded = []
    for name in modules:
        import_runtime_module(name)
        loaded.append(name)
    return "imported {}".format(", ".join(loaded))


def test_scripting_entrypoint(fake, tmp):
    fake.reset_scene(image_shape=(2, 2, 2), minimum_value=0, maximum_value=100)
    entry = import_runtime_module("_mimics_entrypoint")
    result = entry.run_runtime_entry(
        {"__name__": "scripting_library.03_Review.05_Window_Reset_Full_Range"},
        str(LIBRARY_DIR / "03_Review" / "05_Window_Reset_Full_Range.py"),
        "window_level_mimics",
        action_value="reset",
    )
    assert_equal(result, 0, "entrypoint result")
    assert_equal(fake.view.get_contrast(), ((0, 0.0), (100, 1.0)), "reset contrast via entrypoint")
    assert_true(fake.update_gui_calls >= 1, "entrypoint did not trigger GUI update")
    return "shared Scripting Library entrypoint executed inside fake Mimics"


def test_nninteractive_task_model_routing(fake, tmp):
    image = fake.reset_scene(
        image_shape=(2, 2, 2),
        minimum_value=0,
        maximum_value=100,
    )
    mask = FakeMask(
        "Liver",
        image=image,
        array=_u8_buffer((2, 2, 2), 1),
        selected=True,
    )
    mask.metadata.set("nninteractive.task_id", "liver")
    fake.data.masks = FakeCollection([mask])
    fake.file.project_path = str(tmp / "s0001.mcs")

    model_dir = tmp / "models" / "liver_v1"
    for name in (
        "dataset.json",
        "plans.json",
        "inference_info.json",
        "inference_session_class.json",
    ):
        path = model_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    checkpoint = model_dir / "fold_0" / "checkpoint_final.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")

    workspace = tmp / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "registry.json").write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_id": "liver",
                        "task_name": "Liver",
                        "mask_names": ["Liver"],
                        "recommended_model_id": "liver_v1",
                        "models": [
                            {
                                "model_id": "liver_v1",
                                "model_dir": str(model_dir),
                                "checkpoint_sha256": "registered-checksum",
                                "strategy": "clopa_in",
                                "state": "validated",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    module = import_runtime_module("nninteractive_finetune_mimics")
    inference = import_runtime_module("nninteractive_mimics")
    module._workspace = lambda: str(workspace)
    captured = []
    original_run = inference.run_with_model_profile
    try:
        inference.run_with_model_profile = lambda profile: captured.append(profile) or 0
        result = module.annotate_with_task_model()
    finally:
        inference.run_with_model_profile = original_run
    assert_equal(result, 0, "task-model annotation result")
    assert_equal(len(captured), 1, "task model should launch once")
    assert_equal(captured[0].get("task_id"), "liver", "resolved task ID")
    assert_equal(captured[0].get("model_id"), "liver_v1", "resolved model ID")
    assert_equal(
        mask.metadata.find("nninteractive.task_id").value,
        "liver",
        "task metadata should remain intact",
    )

    official_key = inference._worker_cache_key({}, image)
    task_config = {"_model_profile": captured[0]}
    task_key = inference._worker_cache_key(task_config, image)
    assert_true(
        official_key != task_key,
        "official and task models must not share an image worker cache",
    )
    second_profile = dict(captured[0])
    second_profile["model_id"] = "liver_v2"
    second_profile["profile_id"] = "liver:liver_v2"
    second_key = inference._worker_cache_key(
        {"_model_profile": second_profile},
        image,
    )
    assert_true(
        task_key != second_key,
        "different task-model versions must not share an image worker cache",
    )

    switch_root = tmp / "switch_workers"
    official_worker_dir = switch_root / "image_worker_official"
    official_worker_dir.mkdir(parents=True)
    (official_worker_dir / "worker_status.json").write_text(
        json.dumps({"status": "ready", "stage": "waiting_for_prompt"}),
        encoding="utf-8",
    )
    official_worker = {
        "worker_dir": str(official_worker_dir),
        "pid": 12345,
        "model_identity": "official",
        "model_id": "official",
    }
    inference._ASYNC_MONITORS.clear()
    inference._ASYNC_IMAGE_WORKERS.clear()
    inference._ASYNC_IMAGE_WORKERS[official_key] = official_worker
    terminated = []
    original_terminate = inference.runtime_common.terminate_process_async
    try:
        inference.runtime_common.terminate_process_async = (
            lambda **kwargs: terminated.append(kwargs) or True
        )
        task_config = {"_model_profile": captured[0], "gpu_lock_timeout_seconds": 30}
        assert_equal(
            inference._different_model_busy_workers(task_config),
            [],
            "idle official worker should not block task-model switch",
        )
        retired = inference._retire_different_model_workers(task_config)
    finally:
        inference.runtime_common.terminate_process_async = original_terminate
    assert_equal(retired, 1, "idle official worker should be retired")
    assert_equal(len(terminated), 1, "idle worker should be reaped asynchronously")
    assert_true(
        (official_worker_dir / "close.json").is_file(),
        "model switch should write a close request",
    )
    assert_true(
        float(task_config["gpu_lock_timeout_seconds"]) >= 120,
        "model switch should tolerate asynchronous GPU lock release",
    )

    (official_worker_dir / "worker_status.json").write_text(
        json.dumps({"status": "result_ready", "stage": "waiting_for_mimics"}),
        encoding="utf-8",
    )
    inference._ASYNC_IMAGE_WORKERS[official_key] = official_worker
    assert_equal(
        inference._different_model_busy_workers(task_config),
        [],
        "an already-consumed result_ready status must not block model switch",
    )

    (official_worker_dir / "worker_status.json").write_text(
        json.dumps({"status": "running", "stage": "prediction"}),
        encoding="utf-8",
    )
    inference._ASYNC_IMAGE_WORKERS[official_key] = official_worker
    busy = inference._different_model_busy_workers(task_config)
    assert_equal(len(busy), 1, "active official prediction must block model switch")
    assert_equal(
        inference._retire_different_model_workers(task_config),
        0,
        "active prediction worker must never be retired",
    )
    inference._ASYNC_IMAGE_WORKERS.clear()
    return (
        "task resolution, model profile routing, worker isolation, and safe "
        "cross-model switching passed"
    )


def test_window_level_from_selected_mask(fake, tmp):
    image = fake.reset_scene(image_shape=(3, 4, 5), minimum_value=0, maximum_value=2026)
    mask = FakeMask("liver_portal_region", image=image, array=_u8_buffer((3, 4, 5), 1), selected=True)
    fake.data.masks.append(mask)
    module = import_runtime_module("window_level_mimics")
    state_path = tmp / "window_state.json"
    module._state_path = lambda: str(state_path)
    result = module.apply_from_selected_mask()
    assert_equal(result, 0, "apply_from_selected_mask result")
    assert_equal(fake.view.get_contrast(), ((0, 0.0), (250, 1.0)), "liver preset clamped contrast")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert_equal(state.get("last_preset"), "Abdomen / Soft Tissue", "selected mask preset")
    module.undo_last()
    assert_equal(fake.view.get_contrast(), ((0.0, 0.0), (2026.0, 1.0)), "undo restored previous contrast")

    image = fake.reset_scene(image_shape=(3, 4, 5), minimum_value=0, maximum_value=4095)
    image._contrast_maximum_value = 3625
    module.reset_full_range()
    assert_equal(fake.view.get_contrast(), ((0, 0.0), (3625, 1.0)), "reset should clamp to Mimics reported range")
    return "mask-name preset, image-range clamp, reset retry, state save, and undo passed"


def test_mask_identifier_all_mask_bbox_scan(fake, tmp):
    image = fake.reset_scene(image_shape=(3, 3, 3), minimum_value=0, maximum_value=100)
    fake.dialogs.messages = []
    fake.logging.messages = []
    fake.update_gui_calls = 0

    class CountingMask(FakeMask):
        def __init__(self, *args, **kwargs):
            super(CountingMask, self).__init__(*args, **kwargs)
            self.reads = 0

        def get_voxel_buffer(self):
            self.reads += 1
            return super(CountingMask, self).get_voxel_buffer()

    miss = CountingMask("kidney", image=image, array=_u8_pattern_buffer((3, 3, 3), lambda x, y, z: (x, y, z) == (2, 2, 2)))
    hit = CountingMask("liver", image=image, array=_u8_pattern_buffer((3, 3, 3), lambda x, y, z: (x, y, z) == (1, 1, 1)))
    hidden = CountingMask("hidden_liver", image=image, array=_u8_buffer((3, 3, 3), 1))
    hidden.visible = False
    fake.data.masks = FakeCollection([miss, hit, hidden])
    fake.indicated_coordinates = [image.get_voxel_center(1, 1, 1)]

    module = import_runtime_module("mask_identifier")
    module.main()

    assert_equal(miss.reads, 0, "far mask should be skipped by bounding-box filtering")
    assert_equal(hit.reads, 1, "visible hit mask should be read on demand")
    assert_equal(hidden.reads, 1, "hidden mask should be scanned by default")
    assert_true(fake.update_gui_calls >= 2, "mask identifier should yield GUI updates between buffer reads")
    # Results are shown via a non-blocking question_box (so tools can be switched
    # safely between clicks). Find the result record by its signature content:
    # the concise result body lists hit mask names.
    result_message = ""
    for record in fake.dialogs.questions:
        text = record.get("message", "")
        if "liver" in text or "No mask at this point" in text:
            result_message = text
            break
    assert_true("liver" in result_message and "hidden_liver" in result_message, "identifier result should include visible and hidden hits")
    return "mask identifier scans hidden masks and skips distant masks before reading voxel buffers"


def test_export_masks_to_buffers(fake, tmp):
    image = fake.reset_scene(image_shape=(2, 3, 4), minimum_value=0, maximum_value=100)
    liver = FakeMask("liver mask", image=image, array=_u8_buffer((2, 3, 4), 1), selected=True)
    kidney = FakeMask("kidney/right", image=image, array=_u8_pattern_buffer((2, 3, 4), lambda x, y, _z: x == y))
    fake.data.masks = FakeCollection([liver, kidney])
    module = import_runtime_module("mimics_export")
    buffers_dir = tmp / "buffers"
    with contextlib.redirect_stdout(io.StringIO()):
        manifest = module.export_masks_to_buffers(str(buffers_dir))
    assert_equal(manifest.get("mimics_shape"), [2, 3, 4], "manifest shape")
    assert_equal(len(manifest.get("masks", [])), 2, "manifest mask count")
    for row in manifest["masks"]:
        path = buffers_dir / row["u8_filename"]
        assert_true(path.is_file(), "missing exported mask buffer: {}".format(path))
        assert_equal(path.stat().st_size, 24, "mask buffer byte count")
    assert_true((buffers_dir / "manifest.json").is_file(), "manifest file missing")
    return "exported 2 mask buffers with manifest and expected byte counts"


def test_external_io_setup_routing(fake, tmp):
    image = fake.reset_scene(image_shape=(2, 2, 2), minimum_value=0, maximum_value=100)
    import_module = import_runtime_module("mimics_import")
    export_module = import_runtime_module("mimics_export")
    setup_module = import_runtime_module("io_setup_mimics")
    captured = []
    launched = []
    old_launch = setup_module.launch
    old_import_run = import_module._run_main_with_args
    old_export_run = export_module._run_main_with_args
    old_import_python = import_module._python_exe
    old_export_python = export_module._python_exe
    try:
        def launch(mode, python_exe, context, callback, timeout_seconds=3600):
            captured.append((mode, context))
            if mode == "import_batch":
                callback({"source_path": str(tmp / "dataset"), "output_path": str(tmp / "chosen_mcs")})
            elif mode == "export_masks":
                callback({
                    "source_path": str(tmp / "dataset" / "s0001" / "ct.nii.gz"),
                    "output_path": str(tmp / "chosen_labels"),
                    "case_info": {
                        "case_id": "s0001",
                        "image": str(tmp / "dataset" / "s0001" / "ct.nii.gz"),
                        "case_dir": str(tmp / "dataset" / "s0001"),
                        "image_type": "medical_image",
                        "masks": [],
                    },
                    "conflict_policy": "skip",
                })
            return 0

        setup_module.launch = launch
        import_module._python_exe = lambda: "external-python"
        export_module._python_exe = lambda: "external-python"
        def run_import(args, import_mode=None, case_info_override=None):
            launched.append(("import", list(args)))
            import_module._LAST_TASK_DESCRIPTOR = {
                "kind": "import",
                "title": "Import dataset",
                "status_path": str(tmp / "import_status.json"),
                "stop_path": str(tmp / "import_stop.json"),
            }
            return 0

        import_module._run_main_with_args = run_import
        export_module._run_main_with_args = lambda args, source_info_override=None: launched.append(("export", list(args), source_info_override)) or 0
        import_module._launch_external_import_setup(None)

        project = tmp / "saved_projects" / "s0001.mcs"
        fake.file.project_path = str(project)
        image.metadata.create(export_module.SOURCE_IMAGE_PATH_METADATA, str(tmp / "dataset" / "s0001" / "ct.nii.gz"))
        # _start_current_project_export validates that at least one mask exists
        # and needs a foreground export monitor timer (Win32 / PyQt5).
        liver = FakeMask("liver", image=image, array=_u8_buffer((2, 2, 2), 1))
        fake.data.masks.append(liver)
        export_calls = []
        old_foreground_monitor = export_module._start_foreground_export_monitor
        old_start_export = export_module._start_current_project_export
        export_module._start_foreground_export_monitor = lambda monitor: True
        def _recorded_export(*args, **kwargs):
            export_calls.append((args, kwargs))
            return {"kind": "export", "title": "test"}
        export_module._start_current_project_export = _recorded_export
        try:
            result = export_module._launch_external_export_setup()
        finally:
            export_module._start_foreground_export_monitor = old_foreground_monitor
            export_module._start_current_project_export = old_start_export
        assert_equal(result, 0, "foreground export launch must not raise")
        assert_equal(len(export_calls), 1, "_start_current_project_export call count")
        eargs, ekwargs = export_calls[0]
        assert_equal(eargs[2] if len(eargs) > 2 else None, str(tmp / "chosen_labels"), "output root")
        assert_equal(eargs[1] if len(eargs) > 1 else None, str(tmp / "dataset" / "s0001" / "ct.nii.gz"), "source image")
        assert_equal(ekwargs.get("overwrite_existing"), False, "skip policy became overwrite")
    finally:
        setup_module.launch = old_launch
        import_module._run_main_with_args = old_import_run
        export_module._run_main_with_args = old_export_run
        import_module._python_exe = old_import_python
        export_module._python_exe = old_export_python

    assert_equal(captured[0][0], "import_batch", "batch import setup mode")
    assert_true("--output-dir" in launched[0][1], "chosen import output was not routed")
    assert_true(str(tmp / "chosen_mcs") in launched[0][1], "chosen import output path missing")
    assert_equal(captured[1][0], "export_masks", "mask export setup mode")
    return "external PySide6 selections route explicit import/export paths"


def test_async_mask_import_apply(fake, tmp):
    image = fake.reset_scene(image_shape=(2, 3, 4), minimum_value=0, maximum_value=100)
    fake.disable_gui_calls = 0
    fake.update_gui_calls = 0
    module = import_runtime_module("mask_import")
    tmp.mkdir(parents=True, exist_ok=True)
    buffer_path = tmp / "liver.u8"
    buffer_path.write_bytes(_u8_pattern_buffer((2, 3, 4), lambda x, y, z: x == 0 and y == z % 3).tobytes())
    result_path = tmp / "bridge_result.json"
    result_path.write_text(json.dumps({
        "status": "ok",
        "masks": [{
            "name": "liver",
            "u8_path": str(buffer_path),
            "mimics_shape": [2, 3, 4],
            "foreground_voxels": 4,
        }],
    }), encoding="utf-8")
    monitor = {
        "monitor_key": str(tmp),
        "result_path": str(result_path),
        "work_dir": str(tmp),
        "active_image": image,
        "image_shape": [2, 3, 4],
        "pending": None,
        "created_names": [],
        "errors": [],
        "deadline": time.time() + 10,
    }
    module._MASK_IMPORT_MONITORS[str(tmp)] = monitor
    module._mask_import_monitor_tick(monitor)
    imported = [mask for mask in fake.data.masks if mask.name == "liver"]
    assert_equal(len(imported), 1, "prepared mask should be created once")
    assert_true(imported[0].visible, "imported mask should be visible")
    assert_equal(imported[0].number_of_pixels, 4, "imported foreground voxel count")
    assert_equal(fake.disable_gui_calls, 0, "mask import must not disable Mimics GUI updates")
    assert_true(fake.update_gui_calls >= 2, "mask import should repaint around buffer apply")
    return "prepared buffer applied visibly without disabling GUI updates"


def test_nninteractive_fast_path_and_mask_buffer(fake, tmp):
    tmp.mkdir(parents=True, exist_ok=True)
    image = fake.reset_scene(image_shape=(2, 3, 4), minimum_value=0, maximum_value=100)
    source_path = tmp / "source.nii.gz"
    source_path.write_bytes(b"fake nifti")
    module = import_runtime_module("nninteractive_mimics")
    ras_to_lps = [[-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    identity = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    image.metadata.set(module.SOURCE_IMAGE_PATH_METADATA, str(source_path))
    image.metadata.set(module.SOURCE_IMAGE_KIND_METADATA, "nifti")
    image.metadata.set(module.SOURCE_IMAGE_SHAPE_METADATA, json.dumps([2, 3, 4]))
    image.metadata.set(module.SOURCE_IMAGE_INDEX_SPACE_METADATA, "nifti_ijk_matches_derived_dicom_columns_rows_slices_v1")
    image.metadata.set(module.SOURCE_IMAGE_MODALITY_METADATA, "CT")
    image.metadata.set(module.SOURCE_WORLD_COORDINATE_SYSTEM_METADATA, "ras")
    image.metadata.set(module.MIMICS_WORLD_COORDINATE_SYSTEM_METADATA, "lps")
    image.metadata.set(module.SOURCE_TO_MIMICS_WORLD_MATRIX_METADATA, json.dumps(ras_to_lps))
    image.metadata.set(module.SOURCE_VOXEL_TO_RAS_MATRIX_METADATA, json.dumps(identity))
    image.metadata.set(module.MIMICS_VOXEL_TO_RAS_MATRIX_METADATA, json.dumps(identity))
    image.metadata.set(module.MIMICS_TO_SOURCE_INDEX_MATRIX_METADATA, json.dumps(identity))
    exported = module._source_image_export(image, {"image_input_mode": "source_image"})
    assert_true(exported is not None, "source-image fast path was not selected")
    assert_equal(exported["kind"], "source_image", "source export kind")
    assert_equal(exported["path"], "", "source fast path should not export a Mimics buffer")
    assert_equal(exported["image_path"], str(source_path), "source image path")
    assert_equal(exported["source_intensity_space"], "hu_to_mimics_gv", "source intensity conversion")
    assert_equal(exported["source_to_mimics_gv_slope"], 1.0, "source intensity slope")
    assert_equal(exported["source_to_mimics_gv_intercept"], 0.0, "source intensity intercept")

    dicom_dir = tmp / "dicom_source"
    dicom_dir.mkdir()
    image.metadata.set(module.SOURCE_IMAGE_PATH_METADATA, str(dicom_dir))
    image.metadata.set(module.SOURCE_IMAGE_KIND_METADATA, "dicom_folder")
    image.metadata.set(module.SOURCE_IMAGE_INDEX_SPACE_METADATA, "dicom_columns_rows_slices_sorted_by_position_v1")
    image.metadata.set(module.SOURCE_IMAGE_MODALITY_METADATA, "MR")
    image.metadata.set(module.SOURCE_WORLD_COORDINATE_SYSTEM_METADATA, "lps")
    image.metadata.set(module.MIMICS_WORLD_COORDINATE_SYSTEM_METADATA, "lps")
    image.metadata.set(module.SOURCE_TO_MIMICS_WORLD_MATRIX_METADATA, json.dumps(identity))
    image.metadata.set(module.SOURCE_VOXEL_TO_RAS_MATRIX_METADATA, json.dumps(identity))
    image.metadata.set(module.MIMICS_VOXEL_TO_RAS_MATRIX_METADATA, json.dumps(identity))
    exported_mr = module._source_image_export(image, {"image_input_mode": "source_image"})
    assert_equal(exported_mr["source_modality"], "MR", "source modality metadata")
    assert_equal(exported_mr["source_intensity_space"], "source_values", "MR source should not use HU-to-GV")

    empty = FakeMask("empty", image=image, array=_u8_buffer((2, 3, 4), 0))
    empty_export = module._export_mask(empty, str(tmp / "empty.u8"), shape_hint=[2, 3, 4])
    assert_equal(empty_export["path"], "", "empty mask export path")
    assert_true(not (tmp / "empty.u8").exists(), "empty mask should skip full buffer export")

    raw_path = tmp / "prediction.u8"
    raw_path.write_bytes(bytes([0, 1] * 12))
    target = FakeMask("target", image=image, array=_u8_buffer((2, 3, 4), 0))
    module._set_mask_from_u8(target, str(raw_path), [2, 3, 4])
    assert_equal(target.number_of_pixels, 12, "applied nnInteractive foreground count")
    assert_true(fake.disable_gui_calls >= 1 and fake.enable_gui_calls >= 1, "mask apply did not bracket GUI updates")

    class CountingMask(FakeMask):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.buffer_reads = 0

        def get_voxel_buffer(self):
            self.buffer_reads += 1
            return super().get_voxel_buffer()

    job_dir = tmp / "async_job"
    (job_dir / "commands").mkdir(parents=True, exist_ok=True)
    counting = CountingMask("counting", image=image, array=_u8_pattern_buffer((2, 3, 4), lambda x, y, z: x == 0))
    state = {
        "_job_dir": str(job_dir),
        "shape": [2, 3, 4],
        "base_path": "",
        "interactions": [],
        "target_guid": counting.guid,
        "target_name": counting.name,
    }
    module._enqueue_async_prediction(state, counting, expected_hash="already-validated")
    assert_equal(counting.buffer_reads, 0, "enqueue should reuse already validated target hash")
    command_path = job_dir / "commands" / "command_000001.json"
    command = json.loads(command_path.read_text(encoding="utf-8"))
    assert_equal(command["expected_target_sha256"], "already-validated", "queued expected hash")

    module._enqueue_async_prediction(state, counting)
    assert_equal(counting.buffer_reads, 1, "enqueue without supplied hash should read target once")
    return "source fast path, empty-mask optimization, u8 apply, and hash reuse passed"


def test_nninteractive_derived_draft_session(fake, tmp):
    tmp.mkdir(parents=True, exist_ok=True)
    image = fake.reset_scene(image_shape=(2, 3, 4), minimum_value=0, maximum_value=100)
    source = FakeMask(
        "liver",
        image=image,
        array=_u8_pattern_buffer((2, 3, 4), lambda x, _y, _z: x == 0),
        selected=True,
    )
    fake.data.masks.append(source)
    module = import_runtime_module("nninteractive_mimics")

    baseline_questions = len(fake.dialogs.questions)
    deferred = module._select_session_masks(image, {})
    assert_true(deferred["target"] is source, "default policy should not create an early Draft")
    assert_equal(deferred["write_mode"], "choose_on_first_result", "default result destination should be deferred")
    assert_equal(len(fake.dialogs.questions), baseline_questions, "starting nnInteractive should not ask for a result destination")
    fake.dialogs.question_answers.append("Create Editable Copy")
    deferred_state = {"_job_dir": str(tmp / "deferred-job"), "write_mode": "choose_on_first_result"}
    deferred_target = module._choose_completed_result_target(image, source, deferred_state, {"elapsed_seconds": 1.25})
    assert_true(deferred_target is not source, "completion choice should create a new Draft")
    assert_equal(source.number_of_pixels, 12, "completion copy choice changed source Mask")
    fake.data.masks.delete(deferred_target)
    source.selected = True

    session = module._select_session_masks(
        image, {"existing_mask_result_mode": "derived_copy"}
    )
    target = session["target"]
    assert_true(session["source"] is source, "selected manual mask should be the source")
    assert_true(target is not source, "non-empty manual mask should create a separate Draft")
    assert_equal(target.name, "liver - AI Draft", "derived Draft name")
    assert_equal(target.number_of_pixels, 0, "Draft must stay empty until AI produces a result")
    assert_equal(source.number_of_pixels, 12, "source Mask changed while creating Draft")
    assert_equal(
        target.metadata.find(module.DRAFT_ROLE_METADATA).value,
        module.DRAFT_ROLE_VALUE,
        "Draft metadata role",
    )
    assert_equal(
        target.metadata.find(module.DRAFT_SOURCE_GUID_METADATA).value,
        source.guid,
        "Draft source GUID",
    )

    worker_dir = tmp / "shared_worker"
    worker_dir.mkdir()
    worker_log = tmp / "worker.log"
    runtime_log = tmp / "runtime.log"
    cached = {
        "worker_dir": str(worker_dir),
        "pid": 12345,
        "python": sys.executable,
        "python_version": "3",
        "folds": ["0"],
        "worker_log": str(worker_log),
        "runtime_log": str(runtime_log),
        "shape": [2, 3, 4],
        "image_source": "fake",
    }
    old_alive = module._shared_image_worker_alive
    old_cache = dict(module._ASYNC_IMAGE_WORKERS)
    try:
        module._shared_image_worker_alive = lambda _worker: True
        module._ASYNC_IMAGE_WORKERS.clear()
        module._ASYNC_IMAGE_WORKERS[module._object_id(image)] = cached
        state = module._start_async_job(
            {}, image, target, source=source, write_mode="derived_copy"
        )
    finally:
        module._shared_image_worker_alive = old_alive
        module._ASYNC_IMAGE_WORKERS.clear()
        module._ASYNC_IMAGE_WORKERS.update(old_cache)

    assert_equal(state["source_guid"], source.guid, "async source GUID")
    assert_equal(state["target_guid"], target.guid, "async target GUID")
    assert_equal(state["write_mode"], "derived_copy", "async write mode")
    assert_true(state["base_path"], "non-empty source snapshot was not exported")
    assert_true(state["base_sha256"].startswith("sha256:"), "source snapshot hash")
    assert_true(
        state["expected_target_sha256"].startswith("empty-mask:"),
        "target stale guard must describe the empty Draft, not the source",
    )
    assert_equal(source.number_of_pixels, 12, "source changed while starting async job")
    assert_equal(target.number_of_pixels, 0, "starting async job copied source into Draft synchronously")

    source.selected = False
    target.selected = True
    continued = module._select_session_masks(image, {})
    assert_true(continued["source"] is target, "existing AI Draft should be its own session source")
    assert_true(continued["target"] is target, "continuing an AI Draft created another Draft")
    assert_equal(continued["write_mode"], "in_place", "existing AI Draft continuation mode")
    assert_equal(
        len([mask for mask in fake.data.masks if "AI Draft" in mask.name]),
        1,
        "continuing an AI Draft should not create a Draft of a Draft",
    )

    unused = module._create_result_mask(image, "Unused Draft")
    module._mark_ai_draft(unused)
    module._delete_unused_auto_draft(unused, True)
    assert_true(unused not in fake.data.masks, "unused empty auto-Draft was not removed")
    module._delete_unused_auto_draft(target, True)
    assert_true(target in fake.data.masks, "active Draft with async metadata was deleted")

    old_check = module._check_async_result_nonblocking
    old_stop = module._stop_async_monitor
    baseline_messages = len(fake.dialogs.messages)
    try:
        module._check_async_result_nonblocking = lambda _image, _target, _state: "applied"
        module._stop_async_monitor = lambda _job_dir: None
        monitor = {
            "done": False,
            "deadline": time.time() + 60,
            "timeout_seconds": 60,
            "image": image,
            "target": target,
            "state": {"_job_dir": state["_job_dir"]},
        }
        module._async_monitor_tick(monitor)
    finally:
        module._check_async_result_nonblocking = old_check
        module._stop_async_monitor = old_stop
    assert_equal(len(fake.dialogs.messages), baseline_messages, "result application should not add a second completion notice")
    return "source snapshot and target Draft remain separate across async startup"


def test_fewshot_apply_prediction_and_stop(fake, tmp):
    tmp.mkdir(parents=True, exist_ok=True)
    image = fake.reset_scene(image_shape=(2, 2, 2), minimum_value=0, maximum_value=100)
    module = import_runtime_module("fewshot_mimics")
    ts_root = tmp / "dataset"
    (ts_root / "mcs_output").mkdir(parents=True, exist_ok=True)
    source_project = ts_root / "mcs_output" / "s0001.mcs"
    fake.file.project_path = str(source_project)

    status_path = tmp / "infer_status.json"
    status_path.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    output_path = tmp / "prediction_source.nii.gz"
    output_path.write_bytes(b"fake")
    bridge_output_path = tmp / "prediction.u8"
    bridge_output_path.write_bytes(bytes([1, 0, 0, 1, 0, 0, 0, 0]))
    bridge_result_path = tmp / "bridge_result.json"
    bridge_result_path.write_text(json.dumps({
        "status": "ok",
        "output_path": str(bridge_output_path),
        "mimics_shape": [2, 2, 2],
        "foreground_voxels": 2,
    }), encoding="utf-8")
    source_mask = FakeMask("liver", image=image, array=_u8_buffer((2, 2, 2), 1), selected=True)
    fake.data.masks.append(source_mask)
    fake.dialogs.question_answers.append(module.BUTTON_UPDATE_SELECTED)
    monitor = {
        "monitor_key": "fake-monitor",
        "deadline": time.time() + 60,
        "status_path": str(status_path),
        "bridge_started": True,
        "bridge_result_path": str(bridge_result_path),
        "mask_name": "liver",
        "ts_root": str(ts_root),
        "case_id": "s0001",
        "launch_project_path": str(source_project),
        "target_grid": module._active_live_grid_payload(),
        "prediction_target": module._deferred_prediction_target(source_mask),
    }
    module._MONITORS["fake-monitor"] = monitor
    fake.file.project_path = str(ts_root / "mcs_output" / "different_case.mcs")
    module._monitor_tick(monitor)
    assert_equal(source_mask.number_of_pixels, 8,
                 "prediction must not alter the target in a different open project")
    waiting_status = json.loads(status_path.read_text(encoding="utf-8"))
    assert_equal(waiting_status.get("application_state"), "waiting_for_source_case",
                 "switched-case prediction state")
    fake.file.project_path = str(source_project)
    module._monitor_tick(monitor)
    applied = [mask for mask in fake.data.masks if mask is source_mask]
    assert_equal(len(applied), 1, "few-shot result mask count")
    assert_equal(applied[0].number_of_pixels, 2, "few-shot result foreground count")
    assert_true("fake-monitor" not in module._MONITORS, "few-shot monitor was not stopped")
    completion_choices = [item for item in fake.dialogs.questions if item.get("title") == "DINOv3 Prediction Ready"]
    assert_equal(len(completion_choices), 1, "few-shot completion choice count")

    fake.file.project_path = str(source_project)
    jobs_dir = ts_root / "fewshot_models" / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    cancel_path = tmp / "cancel.request"
    job_path = jobs_dir / "job1.json"
    job_path.write_text(json.dumps({
        "job_id": "job1",
        "kind": "train",
        "organ": "liver",
        "status": "waiting_for_gpu",
        "cancel_path": str(cancel_path),
    }), encoding="utf-8")
    fake.dialogs.question_answers.append(module.BUTTON_STOP)
    result = module._stop_latest_job()
    assert_equal(result, 0, "stop latest job result")
    stopped = json.loads(job_path.read_text(encoding="utf-8"))
    assert_equal(stopped.get("status"), "cancelled", "stopped job status")
    assert_true(cancel_path.is_file(), "cancel request file missing")
    assert_true(fake.dialogs.messages[-1]["ui_blocking"] is False, "stop confirmation should be non-blocking")
    return "prediction apply and stop-latest-job flow passed"


def test_fewshot_profile_selector(fake, tmp):
    fake.reset_scene(image_shape=(2, 2, 2), minimum_value=0, maximum_value=100)
    module = import_runtime_module("fewshot_mimics")
    config = {
        "default_training_profile": "balanced",
        "training_profiles": {
            "balanced": {
                "epochs": 10,
                "finetune_method": "lora",
                "batch_size": 1,
                "img_size": "224,224",
                "val_fraction": 0.2,
            },
            "quality_lora": {
                "epochs": 20,
                "finetune_method": "lora",
                "batch_size": 1,
                "img_size": "256,256",
                "val_fraction": 0.25,
            },
        },
    }
    fake.dialogs.question_answers.append("quality_lora")
    options = module._profile_training_options(config)
    assert_equal(options.get("epochs"), 20, "selected profile epochs")
    assert_equal(options.get("img_size"), "256,256", "selected profile image size")
    assert_true(fake.dialogs.questions, "profile selector did not ask the user")
    assert_true("quality_lora" in fake.dialogs.questions[-1]["buttons"], "profile button missing")
    return "DINOv3 profile selector returned the chosen training profile without PyQt5"


def test_fewshot_external_advanced_setup(fake, tmp):
    fake.reset_scene(image_shape=(2, 2, 2), minimum_value=0, maximum_value=100)
    module = import_runtime_module("fewshot_mimics")
    ts_root = tmp / "dataset"
    (ts_root / "mcs_output").mkdir(parents=True, exist_ok=True)
    (ts_root / "mcs_output" / "s0001.mcs").write_text("", encoding="utf-8")
    fake.file.project_path = str(ts_root / "mcs_output" / "s0001.mcs")
    launched = []
    baseline_messages = len(fake.dialogs.messages)

    class Proc:
        pid = 76543
        def poll(self):
            return None

    old_launch = module._launch_gui_process
    old_monitor = module._start_monitor
    old_project = module._project_root
    old_script = module._training_setup_ui_script
    try:
        module._launch_gui_process = lambda cmd, cwd=None, stderr_log=None: launched.append((cmd, cwd)) or Proc()
        module._start_monitor = lambda monitor, poll_seconds=1.0: True
        module._project_root = lambda: str(ROOT)
        module._training_setup_ui_script = lambda: str(ROOT / "tools" / "fewshot_training_setup_ui.py")
        result = module._launch_external_advanced_training(
            {"python": sys.executable, "dinov3_project": "external/dinov3-medical-seg"},
            "liver",
            str(ts_root),
        )
    finally:
        module._launch_gui_process = old_launch
        module._start_monitor = old_monitor
        module._project_root = old_project
        module._training_setup_ui_script = old_script
    assert_equal(result, 0, "external setup launch result")
    assert_equal(len(launched), 1, "external setup Popen call count")
    cmd, _cwd = launched[0]
    assert_true("fewshot_training_setup_ui.py" in cmd[1], "external setup script not launched")
    assert_true("--context" in cmd, "external setup context argument missing")
    context_path = Path(cmd[cmd.index("--context") + 1])
    assert_true(context_path.is_file(), "external setup context file missing")
    context = json.loads(context_path.read_text(encoding="utf-8"))
    assert_equal(context["organ"], "liver", "external setup organ")
    status = json.loads(Path(context["setup_status_path"]).read_text(encoding="utf-8"))
    assert_equal(status["status"], "configuring", "external setup status")
    assert_equal(status["controller_pid"], 76543, "external setup controller pid")
    assert_equal(len(fake.dialogs.messages), baseline_messages, "external setup should not show a Mimics popup")
    return "DINOv3 Advanced setup launches externally and returns immediately"


def test_stop_background_locks(fake, tmp):
    module = import_runtime_module("mimics_stop_background")
    module._project_root = lambda: str(tmp)
    lock_dir = tmp / ".mimics_runtime" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    for name in ("gpu.lock", "background_mimics.lock"):
        path = lock_dir / name
        path.write_text(
            json.dumps({
                "pid": 987654321,
                "token": name + "-token",
                "owner": "stale test owner",
            }),
            encoding="utf-8",
        )
    output_dir = tmp / "mcs_output"
    output_dir.mkdir()
    output_runtime = Path(module.runtime_common.import_queue_runtime_dir(str(tmp), str(output_dir)))
    output_runtime.mkdir(parents=True)
    (output_runtime / "_mcs_queue_active.json").write_text(
        json.dumps({"output_dir": str(output_dir)}), encoding="utf-8"
    )
    registry = tmp / ".mimics_runtime" / "mcs_queues"
    registry.mkdir(parents=True, exist_ok=True)
    (registry / "queue.json").write_text(json.dumps({"output_dir": str(output_dir)}), encoding="utf-8")
    stopped = module._request_queue_stop()
    assert_true(str(output_dir) in stopped, "queue stop marker did not report output dir")
    assert_true((output_runtime / "_mcs_queue_stop.json").is_file(), "queue stop marker missing")
    assert_true(not (output_runtime / "_mcs_queue_active.json").exists(), "queue active marker was not cleared")
    module._clear_resource_locks()
    assert_true(not (lock_dir / "gpu.lock").exists(), "gpu lock was not cleared")
    assert_true(not (lock_dir / "background_mimics.lock").exists(), "background Mimics lock was not cleared")
    result = module.main()
    assert_equal(result, 0, "stop background main result")
    assert_true(fake.dialogs.messages[-1]["ui_blocking"] is False, "stop background message should be non-blocking")
    return "owned resource locks cleared and user notification stayed non-blocking"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep-temp", action="store_true", help="Keep the temporary test directory.")
    parser.add_argument(
        "--only",
        choices=("imports", "entrypoint", "window", "export", "nninteractive", "taskmodels", "fewshot", "stop", "all"),
        default="all",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    fake = install_fake_mimics()
    temp = tempfile.TemporaryDirectory(prefix="mimics_script_fake_mimics_")
    tmp = Path(temp.name)
    runner = TestRunner()
    tests = []
    if args.only in ("imports", "all"):
        tests.append(("runtime modules import with fake mimics", lambda: test_runtime_imports(fake, tmp / "imports")))
    if args.only in ("entrypoint", "all"):
        tests.append(("Scripting Library shared entrypoint", lambda: test_scripting_entrypoint(fake, tmp / "entrypoint")))
    if args.only in ("window", "all"):
        tests.append(("window/level selected-mask flow", lambda: test_window_level_from_selected_mask(fake, tmp / "window")))
        tests.append(("mask identifier all-mask bounding-box scan", lambda: test_mask_identifier_all_mask_bbox_scan(fake, tmp / "mask_identifier")))
    if args.only in ("export", "all"):
        tests.append(("mask export buffer flow", lambda: test_export_masks_to_buffers(fake, tmp / "export")))
        tests.append(("external I/O setup routing", lambda: test_external_io_setup_routing(fake, tmp / "io_setup")))
        tests.append(("asynchronous mask import apply", lambda: test_async_mask_import_apply(fake, tmp / "mask_import")))
    if args.only in ("nninteractive", "all"):
        tests.append(("nnInteractive Mimics-side buffer flow", lambda: test_nninteractive_fast_path_and_mask_buffer(fake, tmp / "nninteractive")))
        tests.append(("nnInteractive derived Draft flow", lambda: test_nninteractive_derived_draft_session(fake, tmp / "nninteractive_draft")))
    if args.only in ("taskmodels", "all"):
        tests.append(("nnInteractive task-model routing", lambda: test_nninteractive_task_model_routing(fake, tmp / "nninteractive_task_models")))
    if args.only in ("fewshot", "all"):
        tests.append(("DINOv3 few-shot Mimics-side flow", lambda: test_fewshot_apply_prediction_and_stop(fake, tmp / "fewshot")))
        tests.append(("DINOv3 few-shot profile selector", lambda: test_fewshot_profile_selector(fake, tmp / "fewshot_profile")))
        tests.append(("DINOv3 external Advanced setup", lambda: test_fewshot_external_advanced_setup(fake, tmp / "fewshot_external_setup")))
    if args.only in ("stop", "all"):
        tests.append(("Stop Background Services lock cleanup", lambda: test_stop_background_locks(fake, tmp / "stop")))
    for name, func in tests:
        runner.run(name, func)
    code = runner.report()
    if args.keep_temp:
        print("Temporary directory kept: {}".format(tmp))
    else:
        temp.cleanup()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
