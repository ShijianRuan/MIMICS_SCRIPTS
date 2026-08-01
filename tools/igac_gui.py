#!/usr/bin/env python3
"""Non-modal PySide6 workspace for live IGAC contour interaction."""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resource_locks import (  # noqa: E402
    FileResourceLock,
    ResourceLockCancelled,
    default_resource_lock_dir,
    process_exists,
)
from tools.igac_engine import (  # noqa: E402
    IGACEngine,
    ORIENTATIONS,
    load_raw_inputs,
    orientation_extent,
)
from tools.ui_theme import configure_application, stylesheet as shared_stylesheet  # noqa: E402


TERMINAL_STATES = {"completed", "failed", "cancelled"}


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    last_error: OSError | None = None
    for attempt in range(20):
        temporary = path.with_name(path.name + ".{}.{}.tmp".format(os.getpid(), uuid.uuid4().hex))
        try:
            temporary.write_text(text, encoding="utf-8")
            os.replace(str(temporary), str(path))
            return
        except OSError as exc:
            last_error = exc
            try:
                temporary.unlink()
            except OSError:
                pass
            time.sleep(min(0.25, 0.04 * (attempt + 1)))
    try:
        path.write_text(text, encoding="utf-8")
    except OSError:
        if last_error is not None:
            raise last_error
        raise


class StatusWriter:
    def __init__(self, request: dict[str, Any]) -> None:
        self.request = request
        self.path = Path(str(request["status_path"]))
        self.log_path = Path(str(request["log_path"]))
        self.started = time.time()
        self._last_log_key = ""
        self._last_log_epoch = 0.0

    def log(self, message: str) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write("[{}] {}\n".format(time.strftime("%Y-%m-%d %H:%M:%S"), message))

    def update(self, phase: str, progress: int, message: str, **extra: Any) -> None:
        payload: dict[str, Any] = {
            "schema_version": "mimics_interactive_algorithm_job.v1",
            "status": "running",
            "tool": "igac",
            "phase": str(phase),
            "progress_percent": max(0, min(100, int(progress))),
            "message": str(message),
            "pid": os.getpid(),
            "started_at_epoch": self.started,
            "updated_at_epoch": time.time(),
            "log_path": str(self.log_path),
        }
        payload.update(extra)
        write_json_atomic(self.path, payload)
        key = "{}:{}".format(phase, message)
        now = time.time()
        if key != self._last_log_key or now - self._last_log_epoch >= 60.0:
            suffix = " Iteration: {}.".format(extra.get("iteration")) if extra.get("iteration") is not None else ""
            self.log(str(message) + suffix)
            self._last_log_key = key
            self._last_log_epoch = now

    def finish(self, status: str, message: str, **extra: Any) -> None:
        payload: dict[str, Any] = {
            "schema_version": "mimics_interactive_algorithm_job.v1",
            "status": str(status),
            "tool": "igac",
            "phase": str(status),
            "progress_percent": 100 if status == "completed" else 0,
            "message": str(message),
            "pid": os.getpid(),
            "started_at_epoch": self.started,
            "updated_at_epoch": time.time(),
            "elapsed_seconds": round(time.time() - self.started, 3),
            "log_path": str(self.log_path),
        }
        payload.update(extra)
        write_json_atomic(self.path, payload)
        self.log(message)


class EngineThread(QtCore.QThread):
    ready = QtCore.Signal(object)
    frame_ready = QtCore.Signal(object, int, float)
    metrics_ready = QtCore.Signal(object)
    phase_changed = QtCore.Signal(str, str)
    failed = QtCore.Signal(str)
    terminal = QtCore.Signal(str)

    def __init__(self, request: dict[str, Any], parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.request = request
        self.commands: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.cancelled = False
        self.writer = StatusWriter(request)
        self.orientation = "axial"
        shape = tuple(int(value) for value in request["shape"])
        self.slice_index = max(0, shape[2] // 2)
        self.running = bool((request.get("igac") or {}).get("auto_run", True))
        self.engine: IGACEngine | None = None
        self.lock: FileResourceLock | None = None
        self._last_status_write = 0.0
        self._last_metrics_emit = 0.0

    def submit(self, command: str, payload: Any = None) -> None:
        self.commands.put((str(command), payload))

    def should_cancel(self) -> bool:
        if self.cancelled or Path(str(self.request["cancel_path"])).exists():
            return True
        parent_pid = int(self.request.get("parent_pid") or 0)
        return bool(parent_pid > 0 and not process_exists(parent_pid))

    def _choose_device_and_lock(self) -> str:
        import torch

        values = self.request.get("igac") or {}
        requested = str(values.get("device") or "auto").lower()
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        if not requested.startswith("cuda"):
            return requested
        if not torch.cuda.is_available():
            if bool(values.get("allow_cpu_fallback", True)):
                self.writer.log("CUDA is unavailable; IGAC is using the CPU fallback.")
                return "cpu"
            raise RuntimeError("CUDA is unavailable and IGAC CPU fallback is disabled.")
        self.phase_changed.emit("waiting", "Waiting for GPU")
        self.writer.update("waiting_for_gpu", 8, "Waiting for the shared GPU resource.")
        self.lock = FileResourceLock(
            default_resource_lock_dir(ROOT) / "gpu.lock",
            resource="gpu",
            owner="IGAC interactive session",
        )
        last_notice = [0.0]

        def on_wait(holder: dict[str, Any]) -> None:
            try:
                from tools.fewshot_pipeline import (
                    request_nninteractive_server_release_on_contention,
                )

                request_nninteractive_server_release_on_contention(holder)
            except Exception:
                pass
            now = time.time()
            if now - last_notice[0] >= 10.0:
                self.writer.update(
                    "waiting_for_gpu",
                    8,
                    "Waiting for the GPU held by {}. Mimics remains available.".format(
                        str((holder or {}).get("owner") or "another AI task")
                    ),
                )
                last_notice[0] = now
        self.lock.acquire(
            wait_seconds=float(values.get("gpu_lock_timeout_seconds", 300.0)),
            poll_seconds=0.25,
            on_wait=on_wait,
            should_cancel=self.should_cancel,
        )
        self.lock.update_pid(
            os.getpid(),
            command="tools/igac_gui.py --request",
            cancel_path=str(self.request["cancel_path"]),
        )
        return requested

    def _write_result(self, mode: str) -> None:
        if self.engine is None:
            raise RuntimeError("IGAC has not finished initializing.")
        self.phase_changed.emit("applying", "Preparing result")
        result = np.asarray(self.engine.result_xyz(), dtype=np.uint8)
        result_path = Path(str(self.request["result_path"]))
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result.tofile(str(result_path))
        foreground = int(np.count_nonzero(result))
        self.writer.finish(
            "completed",
            "IGAC result is ready for Mimics.",
            result={
                "result_path": str(result_path),
                "shape": list(result.shape),
                "foreground_voxels": foreground,
                "iteration": int(self.engine.iteration),
                "apply_mode": str(mode),
                "device": str(self.engine.device),
                "roi_shape_xyz": list(self.engine.roi_shape_xyz),
            },
        )
        self._release_compute_resources()
        self.terminal.emit("completed")

    def _cancel(self, message: str) -> None:
        self.cancelled = True
        self.writer.finish("cancelled", str(message))
        self._release_compute_resources()
        self.terminal.emit("cancelled")

    def _release_compute_resources(self) -> None:
        if self.lock is not None:
            try:
                self.lock.release()
            except Exception:
                pass
            self.lock = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _process_commands(self) -> bool:
        refresh = False
        while True:
            try:
                command, payload = self.commands.get_nowait()
            except queue.Empty:
                break
            if command == "cancel":
                self._cancel(str(payload or "IGAC was cancelled; Mimics was not changed."))
                return False
            if command == "apply":
                self.running = False
                self._write_result(str(payload or "copy"))
                return False
            if command == "running":
                self.running = bool(payload)
                self.phase_changed.emit("running" if self.running else "paused", "Evolving" if self.running else "Paused")
            elif command == "view":
                orientation, index = payload
                self.orientation = str(orientation)
                self.slice_index = int(index)
                refresh = True
            elif command == "brush_segment" and self.engine is not None:
                applied = self.engine.apply_brush_segment(**dict(payload))
                if not applied:
                    self.phase_changed.emit("warning", "Brush is outside workspace")
                refresh = True
            elif command == "parameters" and self.engine is not None:
                self.engine.set_parameters(dict(payload or {}))
                refresh = True
            elif command == "display_window" and self.engine is not None:
                low, high = payload
                self.engine.set_display_window(float(low), float(high))
                refresh = True
            elif command == "reset" and self.engine is not None:
                self.engine.reset()
                refresh = True
        if refresh and self.engine is not None:
            self._emit_frame(0.0)
        return True

    def _emit_frame(self, fps: float) -> None:
        if self.engine is None:
            return
        frame = self.engine.frame(self.orientation, self.slice_index)
        self.frame_ready.emit(frame, int(self.engine.iteration), float(fps))
        now = time.monotonic()
        if now - self._last_metrics_emit >= 0.75:
            self._last_metrics_emit = now
            self.metrics_ready.emit(
                {
                    "foreground_voxels": self.engine.foreground_voxels(),
                    "initial_foreground_voxels": self.engine.initial_foreground_voxels,
                }
            )

    def run(self) -> None:
        try:
            self.writer.update("opening", 3, "Opening the IGAC workspace.")
            self.phase_changed.emit("loading", "Loading 3D image")
            device = self._choose_device_and_lock()
            if self.should_cancel():
                raise ResourceLockCancelled("IGAC was stopped before initialization completed.")
            image, mask = load_raw_inputs(self.request)
            values = self.request.get("igac") or {}
            self.writer.update("initializing", 20, "Initializing the spacing-aware 3D contour.")
            self.engine = IGACEngine(
                image,
                mask,
                tuple(float(value) for value in self.request.get("spacing_mm") or (1.0, 1.0, 1.0)),
                device=device,
                workspace_margin_mm=float(values.get("workspace_margin_mm", 40.0)),
                max_roi_voxels=int(values.get("max_roi_voxels", 10_000_000)),
                parameters=dict(values.get("parameters") or {}),
                display_window=tuple(self.request.get("display_contrast_gv") or ()) or None,
            )
            if str(self.engine.device) == "cpu" and self.engine.roi_voxels > int(values.get("max_cpu_roi_voxels", 1_500_000)):
                raise RuntimeError(
                    "The IGAC workspace contains {:,} voxels, which is too large for responsive CPU use. "
                    "Use a CUDA workstation or a tighter initial Mask."
                    .format(self.engine.roi_voxels)
                )
            metadata = self.engine.metadata()
            self.ready.emit(metadata)
            self.writer.update(
                "interactive",
                25,
                "IGAC is ready in the external interactive workspace.",
                progress_indeterminate=True,
                iteration=0,
                device=str(self.engine.device),
                roi_shape_xyz=list(self.engine.roi_shape_xyz),
            )
            self.phase_changed.emit("running" if self.running else "paused", "Evolving" if self.running else "Paused")
            self._emit_frame(0.0)
            last_frame = time.perf_counter()
            last_iteration = 0
            frame_interval = max(0.08, float(values.get("display_interval_seconds", 0.12)))
            while not self.should_cancel():
                if not self._process_commands():
                    return
                if not self.running:
                    self.msleep(35)
                    continue
                started = time.perf_counter()
                self.engine.step(max(1, int(values.get("iterations_per_cycle", 1))))
                now = time.perf_counter()
                if now - last_frame >= frame_interval:
                    delta_iterations = self.engine.iteration - last_iteration
                    fps = float(delta_iterations) / max(1.0e-6, now - last_frame)
                    self._emit_frame(fps)
                    last_frame = now
                    last_iteration = self.engine.iteration
                if now - self._last_status_write >= 1.0:
                    self._last_status_write = now
                    self.writer.update(
                        "interactive",
                        25,
                        "IGAC interactive evolution is running.",
                        progress_indeterminate=True,
                        iteration=int(self.engine.iteration),
                        device=str(self.engine.device),
                    )
                elapsed = time.perf_counter() - started
                if elapsed < 0.01:
                    self.msleep(max(1, int((0.01 - elapsed) * 1000)))
            self._cancel("IGAC stopped because Mimics closed or a stop request was received.")
        except ResourceLockCancelled as exc:
            self._cancel(str(exc))
        except Exception as exc:
            detail = "{}\n\n{}".format(exc, traceback.format_exc())
            try:
                self.writer.finish("failed", str(exc), error=str(exc), traceback=traceback.format_exc())
            except Exception:
                pass
            self._release_compute_resources()
            self.failed.emit(detail)
            self.terminal.emit("failed")
        finally:
            self._release_compute_resources()


class ImageCanvas(QtWidgets.QWidget):
    brush_requested = QtCore.Signal(object, object)
    stroke_started = QtCore.Signal()
    stroke_finished = QtCore.Signal()
    slice_wheel = QtCore.Signal(int)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(540, 500)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self._image: QtGui.QImage | None = None
        self._array_shape = (1, 1)
        self._target_rect = QtCore.QRectF()
        self._view_rect = QtCore.QRectF()
        self._pending_fit_rect: QtCore.QRectF | None = None
        self._cursor_pos: QtCore.QPointF | None = None
        self._drawing = False
        self._last_voxel: tuple[float, float] | None = None
        self._panning = False
        self._pan_start: QtCore.QPointF | None = None
        self._pan_view = QtCore.QRectF()
        self.brush_radius_voxels = (8.0, 8.0)

    def set_frame(self, frame: dict[str, np.ndarray]) -> None:
        image = np.asarray(frame["image"], dtype=np.float32)
        shape_changed = tuple(image.shape) != tuple(self._array_shape)
        mask = np.asarray(frame["mask"], dtype=bool)
        add = np.asarray(frame.get("add", np.zeros_like(mask)), dtype=bool)
        barrier = np.asarray(frame.get("barrier", np.zeros_like(mask)), dtype=bool)
        gray = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        rgb = np.repeat(gray[:, :, None], 3, axis=2).astype(np.float32)
        rgb[mask] = rgb[mask] * 0.58 + np.array([19.0, 190.0, 174.0]) * 0.42
        if mask.size:
            inner = mask.copy()
            inner[1:, :] &= mask[:-1, :]
            inner[:-1, :] &= mask[1:, :]
            inner[:, 1:] &= mask[:, :-1]
            inner[:, :-1] &= mask[:, 1:]
            boundary = mask & ~inner
            rgb[boundary] = np.array([69.0, 230.0, 210.0])
        rgb[add] = np.array([80.0, 220.0, 132.0])
        rgb[barrier] = np.array([255.0, 105.0, 97.0])
        rgba = np.empty((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
        rgba[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
        rgba[:, :, 3] = 255
        self._array_shape = image.shape
        self._image = QtGui.QImage(
            rgba.data,
            rgba.shape[1],
            rgba.shape[0],
            rgba.strides[0],
            QtGui.QImage.Format_RGBA8888,
        ).copy()
        if shape_changed or self._view_rect.isEmpty():
            self.fit_image()
        if self._pending_fit_rect is not None:
            pending = QtCore.QRectF(self._pending_fit_rect)
            self._pending_fit_rect = None
            self.fit_source_rect(pending)
        self.update()

    def _full_source_rect(self) -> QtCore.QRectF:
        return QtCore.QRectF(0.0, 0.0, float(self._array_shape[1]), float(self._array_shape[0]))

    def _bounded_view_rect(self, value: QtCore.QRectF) -> QtCore.QRectF:
        full = self._full_source_rect()
        width = min(full.width(), max(4.0, float(value.width())))
        height = min(full.height(), max(4.0, float(value.height())))
        left = min(max(full.left(), float(value.left())), full.right() - width)
        top = min(max(full.top(), float(value.top())), full.bottom() - height)
        return QtCore.QRectF(left, top, width, height)

    def fit_source_rect(self, value: QtCore.QRectF) -> None:
        if self._image is None:
            self._pending_fit_rect = QtCore.QRectF(value)
            return
        self._view_rect = self._bounded_view_rect(value)
        self.update()

    def fit_image(self) -> None:
        if self._image is None:
            return
        self._view_rect = self._full_source_rect()
        self.update()

    def zoom(self, factor: float, anchor: tuple[float, float] | None = None) -> None:
        if self._image is None or self._view_rect.isEmpty() or factor <= 0:
            return
        current = QtCore.QRectF(self._view_rect)
        anchor_x, anchor_y = anchor or (current.center().x(), current.center().y())
        relative_x = (float(anchor_x) - current.left()) / max(1.0e-6, current.width())
        relative_y = (float(anchor_y) - current.top()) / max(1.0e-6, current.height())
        width = current.width() / float(factor)
        height = current.height() / float(factor)
        candidate = QtCore.QRectF(
            float(anchor_x) - relative_x * width,
            float(anchor_y) - relative_y * height,
            width,
            height,
        )
        self._view_rect = self._bounded_view_rect(candidate)
        self.update()

    def _compute_target(self) -> QtCore.QRectF:
        if self._image is None:
            return QtCore.QRectF()
        available = QtCore.QRectF(self.rect()).adjusted(18, 18, -18, -18)
        source = self._view_rect if not self._view_rect.isEmpty() else self._full_source_rect()
        scale = min(available.width() / source.width(), available.height() / source.height())
        width = source.width() * scale
        height = source.height() * scale
        return QtCore.QRectF(
            available.center().x() - width / 2.0,
            available.center().y() - height / 2.0,
            width,
            height,
        )

    def _voxel_at(self, position: QtCore.QPointF) -> tuple[float, float] | None:
        target = self._compute_target()
        if not target.contains(position) or self._image is None:
            return None
        source = self._view_rect if not self._view_rect.isEmpty() else self._full_source_rect()
        u = source.left() + (position.x() - target.left()) * source.width() / target.width()
        v = source.top() + (position.y() - target.top()) * source.height() / target.height()
        return float(u), float(v)

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
        painter.fillRect(self.rect(), QtGui.QColor("#101820"))
        target = self._compute_target()
        self._target_rect = target
        if self._image is None:
            painter.setPen(QtGui.QColor("#9eabb8"))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, "Preparing the 3D workspace…")
            return
        source = self._view_rect if not self._view_rect.isEmpty() else self._full_source_rect()
        painter.drawImage(target, self._image, source)
        if self._cursor_pos is not None and target.contains(self._cursor_pos):
            scale_x = target.width() / max(1.0, float(source.width()))
            scale_y = target.height() / max(1.0, float(source.height()))
            radius_x = max(3.0, self.brush_radius_voxels[0] * scale_x)
            radius_y = max(3.0, self.brush_radius_voxels[1] * scale_y)
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 225), 1.25))
            painter.setBrush(QtGui.QColor(255, 255, 255, 25))
            painter.drawEllipse(self._cursor_pos, radius_x, radius_y)

    def enterEvent(self, _event: QtCore.QEvent) -> None:
        self.setCursor(QtCore.Qt.CrossCursor)

    def leaveEvent(self, _event: QtCore.QEvent) -> None:
        self._cursor_pos = None
        self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._panning and self._pan_start is not None:
            target = self._compute_target()
            if target.width() > 0 and target.height() > 0:
                delta = event.position() - self._pan_start
                source_dx = -delta.x() * self._pan_view.width() / target.width()
                source_dy = -delta.y() * self._pan_view.height() / target.height()
                candidate = QtCore.QRectF(self._pan_view)
                candidate.translate(source_dx, source_dy)
                self._view_rect = self._bounded_view_rect(candidate)
                self.update()
            return
        self._cursor_pos = event.position()
        voxel = self._voxel_at(event.position())
        if self._drawing and voxel is not None:
            if self._last_voxel is None:
                self._last_voxel = voxel
            elif math.hypot(voxel[0] - self._last_voxel[0], voxel[1] - self._last_voxel[1]) >= 0.25:
                start = self._last_voxel
                self._last_voxel = voxel
                self.brush_requested.emit(start, voxel)
        self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.position()
            self._pan_view = QtCore.QRectF(self._view_rect)
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            event.accept()
            return
        if event.button() != QtCore.Qt.LeftButton:
            return
        voxel = self._voxel_at(event.position())
        if voxel is None:
            return
        self._drawing = True
        self._last_voxel = voxel
        self.stroke_started.emit()
        self.brush_requested.emit(voxel, voxel)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MiddleButton and self._panning:
            self._panning = False
            self._pan_start = None
            self.setCursor(QtCore.Qt.CrossCursor)
            event.accept()
            return
        if event.button() == QtCore.Qt.LeftButton and self._drawing:
            self._drawing = False
            self._last_voxel = None
            self.stroke_finished.emit()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if event.modifiers() & QtCore.Qt.ControlModifier:
            anchor = self._voxel_at(event.position())
            self.zoom(1.25 if event.angleDelta().y() > 0 else 0.8, anchor)
            event.accept()
            return
        delta = 1 if event.angleDelta().y() > 0 else -1
        self.slice_wheel.emit(delta)
        event.accept()


def make_segmented(buttons: list[tuple[str, str]], checked: str) -> tuple[QtWidgets.QWidget, QtWidgets.QButtonGroup]:
    container = QtWidgets.QWidget()
    layout = QtWidgets.QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    group = QtWidgets.QButtonGroup(container)
    group.setExclusive(True)
    for label, value in buttons:
        button = QtWidgets.QPushButton(label)
        button.setCheckable(True)
        button.setProperty("segmentValue", value)
        button.setChecked(value == checked)
        button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        group.addButton(button)
        layout.addWidget(button)
    return container, group


class ParameterRow(QtWidgets.QWidget):
    value_changed = QtCore.Signal(float)

    def __init__(self, label: str, minimum: float, maximum: float, step: float, value: float, decimals: int = 2) -> None:
        super().__init__()
        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(3)
        title = QtWidgets.QLabel(label)
        title.setObjectName("section")
        self.spin = QtWidgets.QDoubleSpinBox()
        self.spin.setRange(minimum, maximum)
        self.spin.setSingleStep(step)
        self.spin.setDecimals(decimals)
        self.spin.setValue(value)
        self.spin.setFixedWidth(82)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.scale = 1.0 / step
        self.slider.setRange(int(round(minimum * self.scale)), int(round(maximum * self.scale)))
        self.slider.setValue(int(round(value * self.scale)))
        layout.addWidget(title, 0, 0)
        layout.addWidget(self.spin, 0, 1)
        layout.addWidget(self.slider, 1, 0, 1, 2)
        self.slider.valueChanged.connect(lambda raw: self.spin.setValue(raw / self.scale))
        self.spin.valueChanged.connect(self._from_spin)

    def _from_spin(self, value: float) -> None:
        target = int(round(value * self.scale))
        if self.slider.value() != target:
            blocker = QtCore.QSignalBlocker(self.slider)
            self.slider.setValue(target)
            del blocker
        self.value_changed.emit(float(value))

    def value(self) -> float:
        return float(self.spin.value())


class IGACWindow(QtWidgets.QMainWindow):
    def __init__(self, request: dict[str, Any] | None, preview: bool = False) -> None:
        super().__init__()
        self.request = request or {}
        self.preview = bool(preview)
        self.worker: EngineThread | None = None
        self.ready = False
        self.terminal = False
        self.terminal_status = ""
        self.was_running_before_stroke = False
        self.orientation = "axial"
        self.shape_xyz = tuple(int(value) for value in self.request.get("shape") or (512, 512, 180))
        self.spacing_xyz = tuple(float(value) for value in self.request.get("spacing_mm") or (0.7, 0.7, 1.2))
        self.focus_bbox_xyz: list[list[int]] | None = None
        contrast = self.request.get("display_contrast_gv") or []
        self.mimics_display_window = (
            (float(contrast[0]), float(contrast[1])) if len(contrast) == 2 else None
        )
        self.auto_display_window: tuple[float, float] | None = None
        self._updating_display_controls = False
        self.parameter_timer = QtCore.QTimer(self)
        self.parameter_timer.setSingleShot(True)
        self.parameter_timer.setInterval(180)
        self.parameter_timer.timeout.connect(self._submit_parameters)
        self.display_timer = QtCore.QTimer(self)
        self.display_timer.setSingleShot(True)
        self.display_timer.setInterval(90)
        self.display_timer.timeout.connect(self._submit_display_window)
        self.setWindowTitle("IGAC · Interactive 3D Contour")
        self.resize(1280, 950)
        self.setMinimumSize(1040, 720)
        self._build_ui()
        self._connect_ui()
        self._set_orientation("axial")
        if self.preview:
            self._install_preview()
        else:
            self.worker = EngineThread(self.request, self)
            self.worker.ready.connect(self._on_ready)
            self.worker.frame_ready.connect(self._on_frame)
            self.worker.metrics_ready.connect(self._on_metrics)
            self.worker.phase_changed.connect(self._set_phase)
            self.worker.failed.connect(self._on_failed)
            self.worker.terminal.connect(self._on_terminal)
            self.worker.finished.connect(self._on_worker_finished)
            self.worker.start()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(22, 18, 22, 18)
        outer.setSpacing(14)

        header = QtWidgets.QHBoxLayout()
        title_column = QtWidgets.QVBoxLayout()
        title_column.setSpacing(2)
        title = QtWidgets.QLabel("IGAC")
        title.setObjectName("title")
        subtitle = QtWidgets.QLabel("Interactive 3D contour workspace")
        subtitle.setObjectName("subtitle")
        title_column.addWidget(title)
        title_column.addWidget(subtitle)
        header.addLayout(title_column)
        header.addStretch(1)
        self.device_label = QtWidgets.QLabel("Preparing")
        self.device_label.setObjectName("devicePill")
        self.status_label = QtWidgets.QLabel("Loading")
        self.status_label.setObjectName("statusPill")
        header.addWidget(self.device_label)
        header.addWidget(self.status_label)
        outer.addLayout(header)

        body = QtWidgets.QHBoxLayout()
        body.setSpacing(16)
        left = QtWidgets.QVBoxLayout()
        left.setSpacing(9)
        view_row = QtWidgets.QHBoxLayout()
        orientation_widget, self.orientation_group = make_segmented(
            [("Axial", "axial"), ("Coronal", "coronal"), ("Sagittal", "sagittal")],
            "axial",
        )
        orientation_widget.setMaximumWidth(420)
        view_row.addWidget(orientation_widget)
        view_row.addStretch(1)
        self.iteration_label = QtWidgets.QLabel("Iteration 0")
        self.iteration_label.setObjectName("hint")
        self.speed_label = QtWidgets.QLabel("— it/s")
        self.speed_label.setObjectName("hint")
        view_row.addWidget(self.iteration_label)
        view_row.addSpacing(10)
        view_row.addWidget(self.speed_label)
        view_row.addSpacing(10)
        self.volume_label = QtWidgets.QLabel("Mask volume —")
        self.volume_label.setObjectName("hint")
        view_row.addWidget(self.volume_label)
        left.addLayout(view_row)
        self.canvas = ImageCanvas()
        left.addWidget(self.canvas, 1)
        slice_row = QtWidgets.QHBoxLayout()
        slice_title = QtWidgets.QLabel("Slice")
        slice_title.setObjectName("section")
        self.slice_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slice_spin = QtWidgets.QSpinBox()
        self.slice_spin.setFixedWidth(76)
        self.fit_mask_button = QtWidgets.QPushButton("Fit Mask")
        self.fit_mask_button.setToolTip("Center and enlarge the selected Mask with local context")
        self.fit_mask_button.setEnabled(False)
        self.fit_image_button = QtWidgets.QPushButton("Fit Image")
        self.fit_image_button.setToolTip("Show the complete image slice")
        self.zoom_out_button = QtWidgets.QToolButton()
        self.zoom_out_button.setText("−")
        self.zoom_out_button.setToolTip("Zoom out")
        self.zoom_out_button.setFixedSize(32, 30)
        self.zoom_in_button = QtWidgets.QToolButton()
        self.zoom_in_button.setText("+")
        self.zoom_in_button.setToolTip("Zoom in")
        self.zoom_in_button.setFixedSize(32, 30)
        slice_row.addWidget(slice_title)
        slice_row.addWidget(self.slice_slider, 1)
        slice_row.addWidget(self.slice_spin)
        slice_row.addWidget(self.fit_mask_button)
        slice_row.addWidget(self.fit_image_button)
        slice_row.addWidget(self.zoom_out_button)
        slice_row.addWidget(self.zoom_in_button)
        left.addLayout(slice_row)
        body.addLayout(left, 1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setFixedWidth(350)
        sidebar = QtWidgets.QWidget()
        side = QtWidgets.QVBoxLayout(sidebar)
        side.setContentsMargins(2, 2, 8, 2)
        side.setSpacing(12)

        display = QtWidgets.QGroupBox("Display")
        display_layout = QtWidgets.QVBoxLayout(display)
        display_layout.setSpacing(8)
        preset_row = QtWidgets.QHBoxLayout()
        self.display_mimics_button = QtWidgets.QPushButton("Mimics")
        self.display_mimics_button.setToolTip(
            "Restore the contrast captured from Mimics when IGAC opened"
        )
        self.display_mimics_button.setEnabled(self.mimics_display_window is not None)
        self.display_auto_button = QtWidgets.QPushButton("Auto")
        self.display_auto_button.setToolTip("Use the robust intensity range of this image")
        preset_row.addWidget(self.display_mimics_button)
        preset_row.addWidget(self.display_auto_button)
        display_layout.addLayout(preset_row)
        values_row = QtWidgets.QHBoxLayout()
        values_row.setSpacing(7)
        values_row.addWidget(QtWidgets.QLabel("Width"))
        self.display_width = QtWidgets.QDoubleSpinBox()
        self.display_width.setRange(0.1, 20000000.0)
        self.display_width.setDecimals(1)
        self.display_width.setFixedWidth(96)
        values_row.addWidget(self.display_width)
        values_row.addWidget(QtWidgets.QLabel("Level"))
        self.display_level = QtWidgets.QDoubleSpinBox()
        self.display_level.setRange(-10000000.0, 10000000.0)
        self.display_level.setDecimals(1)
        self.display_level.setFixedWidth(96)
        values_row.addWidget(self.display_level)
        display_layout.addLayout(values_row)
        display.setToolTip("Display contrast only; changing it does not alter IGAC evolution")
        side.addWidget(display)

        tools = QtWidgets.QGroupBox("Guidance")
        tools_layout = QtWidgets.QVBoxLayout(tools)
        tools_layout.setSpacing(9)
        tool_widget, self.tool_group = make_segmented(
            [("Add", "add"), ("Barrier", "barrier"), ("Neutral", "neutral")],
            "add",
        )
        tools_layout.addWidget(tool_widget)
        brush_row = QtWidgets.QHBoxLayout()
        brush_row.addWidget(QtWidgets.QLabel("Brush size"))
        brush_row.addStretch(1)
        self.brush_value = QtWidgets.QLabel("6.0 mm")
        self.brush_value.setObjectName("preview")
        brush_row.addWidget(self.brush_value)
        tools_layout.addLayout(brush_row)
        self.brush_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.brush_slider.setRange(2, 40)
        self.brush_slider.setValue(12)
        self.brush_slider.setToolTip("Physical brush diameter in millimetres")
        tools_layout.addWidget(self.brush_slider)
        side.addWidget(tools)

        evolution = QtWidgets.QGroupBox("Evolution")
        evolution_layout = QtWidgets.QVBoxLayout(evolution)
        button_row = QtWidgets.QHBoxLayout()
        self.run_button = QtWidgets.QPushButton("Pause")
        self.run_button.setObjectName("primary")
        self.run_button.setCheckable(True)
        auto_run = bool((self.request.get("igac") or {}).get("auto_run", True))
        self.run_button.setChecked(auto_run)
        self.run_button.setText("Pause" if auto_run else "Run")
        self.reset_button = QtWidgets.QPushButton("Reset")
        self.reset_button.setToolTip("Restore the Mask exactly as it was when IGAC opened")
        button_row.addWidget(self.run_button, 1)
        button_row.addWidget(self.reset_button)
        evolution_layout.addLayout(button_row)
        default = dict((self.request.get("igac") or {}).get("parameters") or {})
        limit_row = QtWidgets.QHBoxLayout()
        self.limit_movement = QtWidgets.QCheckBox("Limit auto movement")
        maximum_default = float(default.get("max_displacement_mm", 10.0))
        self.limit_movement.setChecked(maximum_default > 0.0)
        self.limit_movement.setToolTip(
            "Keep automatic evolution near the initial Mask; explicit Add and Barrier guidance can override it"
        )
        self.max_displacement = QtWidgets.QDoubleSpinBox()
        self.max_displacement.setRange(1.0, 40.0)
        self.max_displacement.setSingleStep(1.0)
        self.max_displacement.setDecimals(1)
        self.max_displacement.setSuffix(" mm")
        self.max_displacement.setValue(max(1.0, maximum_default or 10.0))
        self.max_displacement.setEnabled(self.limit_movement.isChecked())
        self.max_displacement.setFixedWidth(96)
        limit_row.addWidget(self.limit_movement)
        limit_row.addStretch(1)
        limit_row.addWidget(self.max_displacement)
        evolution_layout.addLayout(limit_row)
        self.range_row = ParameterRow("Range", 0.5, 8.0, 0.1, float(default.get("range_mm", 3.0)), 1)
        self.range_row.setToolTip("Physical Gaussian scale used by the local intensity model")
        self.smooth_row = ParameterRow("Smooth", 0.0, 100.0, 1.0, float(default.get("smooth", 50.0)), 0)
        self.smooth_row.setToolTip("Strength of contour regularity")
        self.grow_row = ParameterRow("Grow", -0.50, 0.50, 0.01, float(default.get("grow", 0.0)), 2)
        self.grow_row.setToolTip("Positive values favour a slightly larger foreground region")
        evolution_layout.addWidget(self.range_row)
        evolution_layout.addWidget(self.smooth_row)
        evolution_layout.addWidget(self.grow_row)
        side.addWidget(evolution)

        result_group = QtWidgets.QGroupBox("Result")
        result_layout = QtWidgets.QVBoxLayout(result_group)
        destination_widget, self.destination_group = make_segmented(
            [("Update selected", "update"), ("Editable copy", "copy")],
            "copy",
        )
        result_layout.addWidget(destination_widget)
        self.apply_button = QtWidgets.QPushButton("Apply to Mimics")
        self.apply_button.setObjectName("primary")
        self.apply_button.setEnabled(False)
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setObjectName("dangerButton")
        result_layout.addWidget(self.apply_button)
        result_layout.addWidget(self.cancel_button)
        side.addWidget(result_group)
        side.addStretch(1)

        self.roi_label = QtWidgets.QLabel("Preparing 3D workspace")
        self.roi_label.setObjectName("hint")
        self.roi_label.setWordWrap(True)
        side.addWidget(self.roi_label)
        scroll.setWidget(sidebar)
        body.addWidget(scroll)
        outer.addLayout(body, 1)

    def _connect_ui(self) -> None:
        self.orientation_group.buttonClicked.connect(
            lambda button: self._set_orientation(str(button.property("segmentValue")))
        )
        self.tool_group.buttonClicked.connect(lambda _button: None)
        self.slice_slider.valueChanged.connect(self._slice_changed)
        self.slice_spin.valueChanged.connect(self.slice_slider.setValue)
        self.fit_mask_button.clicked.connect(self._fit_mask_view)
        self.fit_image_button.clicked.connect(self.canvas.fit_image)
        self.zoom_out_button.clicked.connect(lambda: self.canvas.zoom(0.8))
        self.zoom_in_button.clicked.connect(lambda: self.canvas.zoom(1.25))
        self.canvas.slice_wheel.connect(lambda delta: self.slice_slider.setValue(self.slice_slider.value() + delta))
        self.canvas.brush_requested.connect(self._brush)
        self.canvas.stroke_started.connect(self._stroke_started)
        self.canvas.stroke_finished.connect(self._stroke_finished)
        self.brush_slider.valueChanged.connect(self._brush_size_changed)
        self.display_mimics_button.clicked.connect(self._use_mimics_display)
        self.display_auto_button.clicked.connect(self._use_auto_display)
        self.display_width.valueChanged.connect(self._display_controls_changed)
        self.display_level.valueChanged.connect(self._display_controls_changed)
        self.run_button.clicked.connect(self._toggle_running)
        self.reset_button.clicked.connect(self._reset)
        self.limit_movement.toggled.connect(self.max_displacement.setEnabled)
        self.limit_movement.toggled.connect(lambda _checked: self._parameters_changed())
        self.max_displacement.valueChanged.connect(lambda _value: self._parameters_changed())
        self.range_row.value_changed.connect(lambda _value: self._parameters_changed())
        self.smooth_row.value_changed.connect(lambda _value: self._parameters_changed())
        self.grow_row.value_changed.connect(lambda _value: self._parameters_changed())
        self.apply_button.clicked.connect(self._apply)
        self.cancel_button.clicked.connect(self._cancel)

    def _selected_value(self, group: QtWidgets.QButtonGroup, default: str) -> str:
        button = group.checkedButton()
        return str(button.property("segmentValue")) if button is not None else default

    def _set_orientation(self, orientation: str) -> None:
        self.orientation = orientation if orientation in ORIENTATIONS else "axial"
        _width, _height, depth = orientation_extent(self.shape_xyz, self.orientation)
        if self.focus_bbox_xyz is not None:
            depth_axis = {"axial": 2, "coronal": 1, "sagittal": 0}[self.orientation]
            bounds = self.focus_bbox_xyz[depth_axis]
            current = max(0, min(depth - 1, int((bounds[0] + bounds[1] - 1) // 2)))
        else:
            current = min(max(0, self.slice_slider.value()), depth - 1)
        if self.focus_bbox_xyz is None and (self.slice_slider.maximum() <= 0 or current == 0):
            current = depth // 2
        with QtCore.QSignalBlocker(self.slice_slider), QtCore.QSignalBlocker(self.slice_spin):
            self.slice_slider.setRange(0, max(0, depth - 1))
            self.slice_spin.setRange(0, max(0, depth - 1))
            self.slice_slider.setValue(current)
            self.slice_spin.setValue(current)
        self._update_canvas_brush_radius()
        if self.focus_bbox_xyz is not None:
            self._fit_mask_view()
        if self.worker is not None:
            self.worker.submit("view", (self.orientation, current))

    def _fit_mask_view(self) -> None:
        if self.focus_bbox_xyz is None:
            self.canvas.fit_image()
            return
        if self.orientation == "axial":
            horizontal_axis, vertical_axis = 0, 1
        elif self.orientation == "coronal":
            horizontal_axis, vertical_axis = 0, 2
        else:
            horizontal_axis, vertical_axis = 1, 2
        margin_mm = 12.0
        horizontal_margin = max(2, int(math.ceil(margin_mm / self.spacing_xyz[horizontal_axis])))
        vertical_margin = max(2, int(math.ceil(margin_mm / self.spacing_xyz[vertical_axis])))
        horizontal = self.focus_bbox_xyz[horizontal_axis]
        vertical = self.focus_bbox_xyz[vertical_axis]
        left = max(0, int(horizontal[0]) - horizontal_margin)
        right = min(self.shape_xyz[horizontal_axis], int(horizontal[1]) + horizontal_margin)
        top = max(0, int(vertical[0]) - vertical_margin)
        bottom = min(self.shape_xyz[vertical_axis], int(vertical[1]) + vertical_margin)
        self.canvas.fit_source_rect(
            QtCore.QRectF(float(left), float(top), float(max(1, right - left)), float(max(1, bottom - top)))
        )

    def _slice_changed(self, value: int) -> None:
        with QtCore.QSignalBlocker(self.slice_spin):
            self.slice_spin.setValue(value)
        if self.worker is not None:
            self.worker.submit("view", (self.orientation, int(value)))

    def _set_display_controls(self, values: tuple[float, float], submit: bool) -> None:
        low, high = float(values[0]), float(values[1])
        if not math.isfinite(low) or not math.isfinite(high) or high <= low:
            return
        self._updating_display_controls = True
        try:
            with QtCore.QSignalBlocker(self.display_width), QtCore.QSignalBlocker(self.display_level):
                self.display_width.setValue(high - low)
                self.display_level.setValue((high + low) / 2.0)
        finally:
            self._updating_display_controls = False
        if submit:
            self._submit_display_window()

    def _display_controls_changed(self, _value: float) -> None:
        if not self._updating_display_controls and self.ready:
            self.display_timer.start()

    def _submit_display_window(self) -> None:
        if self.worker is None or not self.ready:
            return
        width = max(0.1, float(self.display_width.value()))
        level = float(self.display_level.value())
        self.worker.submit("display_window", (level - width / 2.0, level + width / 2.0))

    def _use_mimics_display(self) -> None:
        if self.mimics_display_window is not None:
            self._set_display_controls(self.mimics_display_window, submit=True)

    def _use_auto_display(self) -> None:
        if self.auto_display_window is not None:
            self._set_display_controls(self.auto_display_window, submit=True)

    def _brush_radius_mm(self) -> float:
        return float(self.brush_slider.value()) / 2.0

    def _brush_size_changed(self, _value: int) -> None:
        self.brush_value.setText("{:.1f} mm".format(self._brush_radius_mm() * 2.0))
        self._update_canvas_brush_radius()

    def _update_canvas_brush_radius(self) -> None:
        if self.orientation == "axial":
            horizontal_spacing, vertical_spacing = self.spacing_xyz[0], self.spacing_xyz[1]
        elif self.orientation == "coronal":
            horizontal_spacing, vertical_spacing = self.spacing_xyz[0], self.spacing_xyz[2]
        else:
            horizontal_spacing, vertical_spacing = self.spacing_xyz[1], self.spacing_xyz[2]
        self.canvas.brush_radius_voxels = (
            self._brush_radius_mm() / max(0.01, horizontal_spacing),
            self._brush_radius_mm() / max(0.01, vertical_spacing),
        )
        self.canvas.update()

    def _brush(self, start: tuple[float, float], end: tuple[float, float]) -> None:
        if self.worker is None or not self.ready:
            return
        self.worker.submit(
            "brush_segment",
            {
                "mode": self._selected_value(self.tool_group, "add"),
                "orientation": self.orientation,
                "slice_index": self.slice_slider.value(),
                "start_u": float(start[0]),
                "start_v": float(start[1]),
                "end_u": float(end[0]),
                "end_v": float(end[1]),
                "radius_mm": self._brush_radius_mm(),
            },
        )

    def _stroke_started(self) -> None:
        self.was_running_before_stroke = bool(self.run_button.isChecked())
        if self.worker is not None and self.was_running_before_stroke:
            self.worker.submit("running", False)

    def _stroke_finished(self) -> None:
        if self.worker is not None and self.was_running_before_stroke:
            self.worker.submit("running", True)

    def _toggle_running(self, checked: bool) -> None:
        self.run_button.setText("Pause" if checked else "Run")
        if self.worker is not None:
            self.worker.submit("running", bool(checked))

    def _reset(self) -> None:
        if self.worker is None:
            return
        answer = QtWidgets.QMessageBox.question(
            self,
            "Reset IGAC",
            "Restore the Mask and remove all IGAC guidance?",
            QtWidgets.QMessageBox.Reset | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Cancel,
        )
        if answer == QtWidgets.QMessageBox.Reset:
            self.worker.submit("reset")

    def _parameters_changed(self) -> None:
        if self.worker is not None and self.ready:
            self.parameter_timer.start()

    def _submit_parameters(self) -> None:
        if self.worker is not None and self.ready:
            self.worker.submit(
                "parameters",
                {
                    "range_mm": self.range_row.value(),
                    "smooth": self.smooth_row.value(),
                    "grow": self.grow_row.value(),
                    "max_displacement_mm": (
                        float(self.max_displacement.value()) if self.limit_movement.isChecked() else 0.0
                    ),
                },
            )

    def _apply(self) -> None:
        if self.worker is None or not self.ready:
            return
        self.apply_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self.worker.submit("apply", self._selected_value(self.destination_group, "copy"))

    def _cancel(self) -> None:
        if self.terminal:
            self.close()
            return
        answer = QtWidgets.QMessageBox.question(
            self,
            "Cancel IGAC",
            "Discard this IGAC session? Mimics will not be changed.",
            QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Cancel,
        )
        if answer == QtWidgets.QMessageBox.Discard:
            self.apply_button.setEnabled(False)
            self.cancel_button.setEnabled(False)
            self._set_phase("stopping", "Stopping")
            if self.worker is not None:
                self.worker.submit("cancel", "IGAC was cancelled; Mimics was not changed.")

    def _on_ready(self, metadata: dict[str, Any]) -> None:
        self.ready = True
        self.apply_button.setEnabled(True)
        self.device_label.setText(str(metadata.get("device", "CPU")).upper())
        bbox = metadata.get("initial_bbox_xyz")
        if (
            isinstance(bbox, list)
            and len(bbox) == 3
            and all(isinstance(value, list) and len(value) == 2 for value in bbox)
        ):
            self.focus_bbox_xyz = [[int(value[0]), int(value[1])] for value in bbox]
            self.fit_mask_button.setEnabled(True)
        self.auto_display_window = (
            float(metadata.get("intensity_low", 0.0)),
            float(metadata.get("intensity_high", 1.0)),
        )
        self._set_display_controls(
            (
                float(metadata.get("display_low", self.auto_display_window[0])),
                float(metadata.get("display_high", self.auto_display_window[1])),
            ),
            submit=False,
        )
        roi = metadata.get("roi_shape_xyz") or []
        memory = metadata.get("estimated_memory_mb", 0)
        self.roi_label.setText(
            "Workspace {} · about {:.0f} MB".format(" × ".join(str(value) for value in roi), float(memory))
        )
        if self.focus_bbox_xyz is not None:
            self._set_orientation(self.orientation)

    def _on_frame(self, frame: dict[str, np.ndarray], iteration: int, fps: float) -> None:
        self.canvas.set_frame(frame)
        self.iteration_label.setText("Iteration {:,}".format(int(iteration)))
        self.speed_label.setText("{:.1f} it/s".format(float(fps)) if fps > 0 else "— it/s")

    def _on_metrics(self, values: dict[str, Any]) -> None:
        current = max(0, int(values.get("foreground_voxels", 0)))
        initial = max(0, int(values.get("initial_foreground_voxels", 0)))
        voxel_volume_mm3 = float(self.spacing_xyz[0] * self.spacing_xyz[1] * self.spacing_xyz[2])
        current_cm3 = current * voxel_volume_mm3 / 1000.0
        if initial > 0:
            change = 100.0 * float(current - initial) / float(initial)
            self.volume_label.setText("Mask {:.1f} cm³ · {:+.1f}%".format(current_cm3, change))
        else:
            self.volume_label.setText("Mask {:.1f} cm³".format(current_cm3))

    def _set_phase(self, phase: str, text: str) -> None:
        self.status_label.setText(str(text))
        self.status_label.setProperty("phase", str(phase))
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def _on_failed(self, detail: str) -> None:
        self.ready = False
        self.apply_button.setEnabled(False)
        self.cancel_button.setText("Close")
        self._set_phase("failed", "Failed")
        message = str(detail).split("\n\n", 1)[0]
        QtWidgets.QMessageBox.critical(self, "IGAC", message)

    def _on_terminal(self, status: str) -> None:
        self.terminal = True
        self.terminal_status = str(status)

    def _on_worker_finished(self) -> None:
        if self.terminal_status in ("completed", "cancelled"):
            self.close()

    def _install_preview(self) -> None:
        self.ready = True
        self.apply_button.setEnabled(True)
        self.mimics_display_window = (350.0, 750.0)
        self.auto_display_window = (220.0, 920.0)
        self.display_mimics_button.setEnabled(True)
        self._set_display_controls(self.mimics_display_window, submit=False)
        self.device_label.setText("CUDA")
        self.status_label.setText("Evolving")
        self.roi_label.setText("Workspace 196 × 182 × 144 · about 544 MB")
        self.volume_label.setText("Mask 38.4 cm³ · +6.2%")
        self.iteration_label.setText("Iteration 128")
        self.speed_label.setText("7.8 it/s")
        height, width = 640, 640
        self.shape_xyz = (width, height, 180)
        self.focus_bbox_xyz = [[270, 461], [200, 451], [70, 111]]
        self.fit_mask_button.setEnabled(True)
        self._set_orientation("axial")
        yy, xx = np.mgrid[:height, :width]
        image = 0.16 + 0.44 * np.exp(-(((xx - 320) / 210) ** 2 + ((yy - 318) / 235) ** 2))
        image += 0.18 * np.exp(-(((xx - 365) / 82) ** 2 + ((yy - 325) / 105) ** 2))
        image += 0.025 * np.sin(xx / 11.0) * np.cos(yy / 14.0)
        mask = ((xx - 365) / 95) ** 2 + ((yy - 325) / 125) ** 2 < 1.0
        add = (xx - 315) ** 2 + (yy - 295) ** 2 < 8 ** 2
        barrier = ((xx - 440) ** 2 + (yy - 355) ** 2 < 8 ** 2)
        self.canvas.set_frame({"image": np.clip(image, 0, 1), "mask": mask, "add": add, "barrier": barrier})

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.preview or self.terminal:
            event.accept()
            return
        answer = QtWidgets.QMessageBox.question(
            self,
            "Close IGAC",
            "Discard this IGAC session? Mimics will not be changed.",
            QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Cancel,
        )
        if answer != QtWidgets.QMessageBox.Discard:
            event.ignore()
            return
        event.ignore()
        self.hide()
        if self.worker is not None:
            self.worker.submit("cancel", "IGAC window was closed; Mimics was not changed.")


IGAC_STYLES = """
QLabel#statusPill, QLabel#devicePill {
    border-radius: 6px;
    padding: 5px 10px;
    font-size: 9pt;
    font-weight: 650;
}
QLabel#statusPill {
    color: #0f766e;
    background: #e8f8f5;
    border: 1px solid #a7ddd5;
}
QLabel#statusPill[phase="failed"] {
    color: #b42318;
    background: #fff1f0;
    border-color: #f2b8b2;
}
QLabel#statusPill[phase="warning"] {
    color: #8a5700;
    background: #fff8e7;
    border-color: #f1d28a;
}
QLabel#devicePill {
    color: #344054;
    background: #eef2f6;
    border: 1px solid #d3dae3;
}
QButtonGroup { background: transparent; }
QPushButton:checked {
    color: #0f5e60;
    background: #dff4f1;
    border-color: #5bb8ae;
    font-weight: 650;
}
QGroupBox { padding: 14px 12px 12px 12px; }
QScrollArea { background: #f5f7fa; }
QSlider::groove:horizontal {
    height: 5px;
    border-radius: 2px;
    background: #d9e0e8;
}
QSlider::sub-page:horizontal {
    border-radius: 2px;
    background: #50aaa3;
}
QSlider::handle:horizontal {
    width: 15px;
    height: 15px;
    margin: -6px 0;
    border-radius: 7px;
    background: #ffffff;
    border: 2px solid #16857e;
}
QSlider::handle:horizontal:hover {
    background: #e8f8f5;
}
"""


def read_request(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = ("status_path", "cancel_path", "log_path", "image_path", "mask_path", "result_path", "shape")
    missing = [name for name in required if not value.get(name)]
    if missing:
        raise RuntimeError("IGAC request is missing: {}".format(", ".join(missing)))
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request")
    parser.add_argument("--preview-output")
    args = parser.parse_args(argv)
    if not args.request and not args.preview_output:
        parser.error("--request or --preview-output is required")
    request = read_request(Path(args.request).resolve()) if args.request else None
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    configure_application(app, "IGAC")
    app.setStyleSheet(shared_stylesheet(IGAC_STYLES))
    window = IGACWindow(request, preview=bool(args.preview_output))
    window.show()
    if args.preview_output:
        output = Path(args.preview_output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)

        def save_preview() -> None:
            window.grab().save(str(output))
            window.terminal = True
            window.close()
            app.quit()

        QtCore.QTimer.singleShot(350, save_preview)
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
