#!/usr/bin/env python3
"""Non-modal PySide6 workspace for live IGAC contour interaction."""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import sys
import threading
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
    IGACConvergenceMonitor,
    IGACEngine,
    ORIENTATIONS,
    load_raw_inputs,
    display_to_xyz,
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
    undo_available = QtCore.Signal(bool)

    def __init__(self, request: dict[str, Any], parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.request = request
        self.commands: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.cancelled = False
        self.writer = StatusWriter(request)
        self.orientation = "axial"
        shape = tuple(int(value) for value in request["shape"])
        self.slice_index = max(0, shape[2] // 2)
        # Opening the editor must never mutate the Mask. Evolution is bounded
        # and starts only after an explicit user action.
        self.running = False
        self.stroke_active = False
        self.engine: IGACEngine | None = None
        self.lock: FileResourceLock | None = None
        self._last_status_write = 0.0
        self._last_metrics_emit = 0.0
        self._last_image_frame_key: tuple[Any, ...] | None = None
        self._undo_state: dict[str, Any] | None = None
        self._evolution_start_iteration = 0
        self._evolution_started_at = 0.0
        self._evolution_context = ""
        self._convergence = IGACConvergenceMonitor.from_mapping(
            (request.get("igac") or {}).get("convergence") or {}
        )
        self._pending_brush_lock = threading.Lock()
        self._pending_brush_batch: list[dict[str, Any]] | None = None
        self._pending_boundary_lock = threading.Lock()
        self._pending_boundary_holder: dict[str, Any] | None = None
        self._boundary_base_state: dict[str, Any] | None = None
        self._undo_state_before_boundary: dict[str, Any] | None = None
        self._boundary_changed = False

    def submit(self, command: str, payload: Any = None) -> None:
        if str(command) == "boundary_pull_update":
            queue_update = False
            with self._pending_boundary_lock:
                if self._pending_boundary_holder is None:
                    self._pending_boundary_holder = {"payload": dict(payload or {})}
                    queue_update = True
                else:
                    self._pending_boundary_holder["payload"] = dict(payload or {})
                holder = self._pending_boundary_holder
            if queue_update:
                self.commands.put(("boundary_pull_flush", holder))
            return
        if str(command) == "brush_segment":
            queue_batch = False
            with self._pending_brush_lock:
                if self._pending_brush_batch is None:
                    self._pending_brush_batch = []
                    queue_batch = True
                batch = self._pending_brush_batch
                batch.append(dict(payload or {}))
            if queue_batch:
                self.commands.put(("brush_flush", batch))
            return
        if str(command) in ("stroke_start", "stroke_end", "reclassify_stroke"):
            # A queued brush batch may still be waiting for the worker. Detach
            # it at stroke boundaries so two user actions can never share one
            # undo snapshot or be reordered around stroke_start/stroke_end.
            with self._pending_brush_lock:
                self._pending_brush_batch = None
        self.commands.put((str(command), payload))

    def _take_pending_boundary_pull(self, holder: dict[str, Any]) -> dict[str, Any]:
        with self._pending_boundary_lock:
            payload = dict(holder.get("payload") or {})
            if self._pending_boundary_holder is holder:
                self._pending_boundary_holder = None
        return payload

    def _take_pending_brush_segments(
        self, batch: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        with self._pending_brush_lock:
            if self._pending_brush_batch is batch:
                self._pending_brush_batch = None
            segments = list(batch)
            batch[:] = []
        return segments

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
            release_requested = False
            try:
                from tools.fewshot_pipeline import (
                    request_nninteractive_server_release_on_contention,
                )

                release_requested = request_nninteractive_server_release_on_contention(holder)
            except Exception:
                pass
            now = time.time()
            if now - last_notice[0] >= 10.0:
                if release_requested:
                    message = (
                        "Releasing the idle nnInteractive model before IGAC starts. "
                        "Mimics remains available."
                    )
                else:
                    message = (
                        "Waiting for the GPU held by {}. Mimics remains available."
                        .format(str((holder or {}).get("owner") or "another AI task"))
                    )
                self.writer.update(
                    "waiting_for_gpu",
                    8,
                    message,
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
        self._undo_state = None
        self._boundary_base_state = None
        self._undo_state_before_boundary = None
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

    def _capture_undo_state(self) -> None:
        if self.engine is None:
            return
        self._undo_state = self.engine.capture_state()
        self.undo_available.emit(True)

    def _stop_evolution(self, text: str, *, warning: bool = False) -> None:
        self.running = False
        self._convergence.reset()
        phase = "warning" if warning else "paused"
        self.phase_changed.emit(phase, str(text))

    def _start_evolution(self, context: str, text: str) -> None:
        if self.engine is None:
            return
        self.running = True
        self._evolution_context = str(context)
        self._evolution_start_iteration = int(self.engine.iteration)
        self._evolution_started_at = time.perf_counter()
        self._convergence.reset()
        self.phase_changed.emit("running", str(text))

    def _finish_evolution_cycle(self) -> bool:
        if self.engine is None:
            return False
        iterations = int(self.engine.iteration) - int(self._evolution_start_iteration)
        elapsed = time.perf_counter() - float(self._evolution_started_at)
        reason = self._convergence.observe(
            iterations=iterations,
            elapsed_seconds=elapsed,
            changed_voxel_ratio=float(self.engine.last_step_changed_ratio),
        )
        if reason is None:
            return False
        if reason == "converged":
            text = "Live preview stable" if self.stroke_active else "Boundary fit complete"
            self._stop_evolution(text)
            message = "IGAC boundary fitting converged and paused."
        else:
            label = "iteration" if reason == "iteration_limit" else "time"
            self._stop_evolution(
                "Stopped at the {} safety limit - inspect the preview".format(label),
                warning=True,
            )
            message = (
                "IGAC boundary fitting reached the {} safety limit and paused. "
                "The current preview was kept."
            ).format(label)
        self.writer.update(
            "interactive",
            25,
            message,
            progress_indeterminate=False,
            iteration=int(self.engine.iteration),
            changed_voxels=int(self.engine.last_step_changed_voxels),
            changed_voxel_ratio=float(self.engine.last_step_changed_ratio),
            stop_reason=reason,
            device=str(self.engine.device),
        )
        self._emit_frame(0.0)
        return True

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
            if command == "stroke_start" and self.engine is not None:
                self.running = False
                self.stroke_active = True
                self._capture_undo_state()
                self.phase_changed.emit("editing", "Applying guidance")
            elif command == "boundary_pull_start" and self.engine is not None:
                self.running = False
                self.stroke_active = True
                self._undo_state_before_boundary = self._undo_state
                self._capture_undo_state()
                self._boundary_base_state = self._undo_state
                self._boundary_changed = False
                self.phase_changed.emit("editing", "Boundary point captured")
            elif command == "boundary_pull_flush" and self.engine is not None:
                values = self._take_pending_boundary_pull(payload)
                if self._boundary_base_state is None:
                    self.phase_changed.emit(
                        "warning", "Boundary drag was not initialized - release and try again"
                    )
                    continue
                result = self.engine.apply_boundary_pull(
                    base_state=self._boundary_base_state,
                    **values
                )
                if result is not None:
                    self._boundary_changed = True
                    # Keep pointer tracking lightweight. The FFD preview is
                    # updated while dragging; LGDF edge fitting begins only on
                    # release so an expensive iteration cannot stall the mouse.
                    self.running = False
                    self.phase_changed.emit("editing", "Boundary follows pointer")
                    self.writer.update(
                        "interactive",
                        25,
                        "IGAC boundary drag preview updated.",
                        progress_indeterminate=True,
                        drag_distance_mm=round(float(result["drag_distance_mm"]), 2),
                        influence_radius_mm=round(float(result["influence_radius_mm"]), 2),
                    )
                else:
                    self._boundary_changed = False
                    self.running = False
                    self.phase_changed.emit("editing", "Boundary returned to start")
                refresh = True
            elif command == "boundary_pull_end":
                self.stroke_active = False
                self._boundary_base_state = None
                if self._boundary_changed:
                    self._start_evolution("boundary_pull", "Snapping dragged boundary")
                else:
                    self._undo_state = self._undo_state_before_boundary
                    self.undo_available.emit(self._undo_state is not None)
                    self.phase_changed.emit("paused", "Boundary drag cancelled")
                self._undo_state_before_boundary = None
                self._boundary_changed = False
                refresh = True
            elif command == "stroke_end":
                self.stroke_active = False
                if self.running:
                    self.phase_changed.emit("running", "Settling boundary")
                refresh = True
            elif command == "brush_flush" and self.engine is not None:
                applied = False
                for segment in self._take_pending_brush_segments(payload):
                    applied = self.engine.apply_brush_segment(**segment) or applied
                if applied:
                    self._start_evolution(
                        "guidance",
                        "Live boundary fitting" if self.stroke_active else "Settling boundary",
                    )
                else:
                    self.phase_changed.emit("warning", "Guidance is outside the workspace")
                refresh = True
            elif command == "reclassify_stroke" and self.engine is not None:
                if self._undo_state is None:
                    self.phase_changed.emit(
                        "warning",
                        "Could not reinterpret this gesture - use Undo and try again",
                    )
                    continue
                self.running = False
                self.engine.restore_state(self._undo_state)
                applied = False
                for segment in list(payload or []):
                    applied = self.engine.apply_brush_segment(**segment) or applied
                if applied:
                    self._start_evolution("guidance", "Live boundary fitting")
                else:
                    self.phase_changed.emit("warning", "Guidance is outside the workspace")
                refresh = True
            elif command in ("fit", "refine") and self.engine is not None:
                self._capture_undo_state()
                self.stroke_active = False
                self._start_evolution("manual", "Fitting boundary")
            elif command == "stop" or (command == "running" and not bool(payload)):
                self.stroke_active = False
                self._stop_evolution("Stopped - current preview kept")
                if self.engine is not None:
                    self.writer.update(
                        "interactive",
                        25,
                        "IGAC boundary fitting was stopped by the user; the current preview was kept.",
                        progress_indeterminate=False,
                        iteration=int(self.engine.iteration),
                    )
            elif command == "running" and bool(payload):
                self._start_evolution("manual", "Fitting boundary")
            elif command == "undo" and self.engine is not None:
                self.running = False
                self.stroke_active = False
                if self._undo_state is not None:
                    self.engine.restore_state(self._undo_state)
                    self._undo_state = None
                    self.undo_available.emit(False)
                    self.phase_changed.emit("paused", "Last action undone")
                    refresh = True
            elif command == "view":
                orientation, index = payload
                self.orientation = str(orientation)
                self.slice_index = int(index)
                refresh = True
            elif command == "parameters" and self.engine is not None:
                self.engine.set_parameters(dict(payload or {}))
                if self.running:
                    self._start_evolution(self._evolution_context or "manual", "Fitting boundary")
                refresh = True
            elif command == "display_window" and self.engine is not None:
                low, high = payload
                self.engine.set_display_window(float(low), float(high))
                refresh = True
            elif command == "reset" and self.engine is not None:
                self.running = False
                self.stroke_active = False
                self._capture_undo_state()
                self.engine.reset()
                self.phase_changed.emit("paused", "Original Mask restored")
                refresh = True
        if refresh and self.engine is not None:
            self._emit_frame(0.0)
        return True

    def _emit_frame(self, fps: float) -> None:
        if self.engine is None:
            return
        image_key = (
            str(self.orientation),
            int(self.slice_index),
            float(self.engine.display_low),
            float(self.engine.display_high),
        )
        include_image = image_key != self._last_image_frame_key
        frame = self.engine.frame(
            self.orientation,
            self.slice_index,
            include_image=include_image,
        )
        if include_image:
            self._last_image_frame_key = image_key
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
            self.phase_changed.emit("paused", "Ready - correct an error")
            self._emit_frame(0.0)
            last_frame = time.perf_counter()
            last_iteration = 0
            frame_interval = max(0.04, float(values.get("display_interval_seconds", 0.06)))
            while not self.should_cancel():
                if not self._process_commands():
                    return
                if not self.running:
                    self.msleep(35)
                    continue
                started = time.perf_counter()
                cycle = max(1, int(values.get("iterations_per_cycle", 1)))
                self.engine.step(cycle)
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
                        changed_voxels=int(self.engine.last_step_changed_voxels),
                        changed_voxel_ratio=float(self.engine.last_step_changed_ratio),
                    )
                self._finish_evolution_cycle()
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
    brush_requested = QtCore.Signal(str, object, object)
    stroke_started = QtCore.Signal()
    stroke_finished = QtCore.Signal()
    boundary_drag_started = QtCore.Signal(object)
    boundary_drag_requested = QtCore.Signal(object, object)
    boundary_drag_finished = QtCore.Signal()
    gesture_mode_changed = QtCore.Signal(str)
    stroke_reclassified = QtCore.Signal(str, object)
    slice_wheel = QtCore.Signal(int)
    pointer_changed = QtCore.Signal(object)

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
        self._press_voxel: tuple[float, float] | None = None
        self._press_mask: np.ndarray | None = None
        self._mask_plane = np.zeros((1, 1), dtype=bool)
        self._boundary_plane = np.zeros((1, 1), dtype=bool)
        self._raw_plane: np.ndarray | None = None
        self._stroke_points: list[tuple[float, float]] = []
        self._overlay: QtGui.QImage | None = None
        self._active_brush_mode: str | None = None
        self._gesture_segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
        self._auto_stroke = False
        self._press_inside = False
        self._mode_locked_by_crossing = False
        self._stroke_open = False
        self._pending_auto = False
        self._boundary_dragging = False
        self._boundary_start: tuple[float, float] | None = None
        self._boundary_current: tuple[float, float] | None = None
        self.interaction_mode = "ffd"
        self._panning = False
        self._pan_start: QtCore.QPointF | None = None
        self._pan_view = QtCore.QRectF()
        self.brush_radius_voxels = (8.0, 8.0)
        self.display_spacing_uv = (1.0, 1.0)
        self.boundary_capture_mm = 3.0

    def set_frame(self, frame: dict[str, np.ndarray]) -> None:
        mask = np.asarray(frame["mask"], dtype=bool)
        image_value = frame.get("image")
        raw_value = frame.get("raw_image")
        shape_changed = tuple(mask.shape) != tuple(self._array_shape)
        if image_value is not None:
            image = np.asarray(image_value, dtype=np.float32)
            if tuple(image.shape) != tuple(mask.shape):
                raise ValueError("The IGAC image and Mask display planes do not match.")
            gray = np.ascontiguousarray(
                np.clip(image * 255.0, 0, 255).astype(np.uint8)
            )
            self._image = QtGui.QImage(
                gray.data,
                gray.shape[1],
                gray.shape[0],
                gray.strides[0],
                QtGui.QImage.Format_Grayscale8,
            ).copy()
            if raw_value is not None:
                raw = np.asarray(raw_value)
                if tuple(raw.shape) != tuple(mask.shape):
                    raise ValueError("The IGAC value and Mask display planes do not match.")
                self._raw_plane = raw
        elif self._image is None:
            raise ValueError("The first IGAC display frame must contain the image plane.")
        add_value = frame.get("add")
        barrier_value = frame.get("barrier")
        add = np.zeros_like(mask) if add_value is None else np.asarray(add_value, dtype=bool)
        barrier = (
            np.zeros_like(mask)
            if barrier_value is None
            else np.asarray(barrier_value, dtype=bool)
        )
        overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        overlay[mask] = np.array([19, 190, 174, 104], dtype=np.uint8)
        boundary = self._boundary_mask(mask)
        if mask.size:
            overlay[boundary] = np.array([69, 230, 210, 245], dtype=np.uint8)
        overlay[add] = np.array([80, 220, 132, 220], dtype=np.uint8)
        overlay[barrier] = np.array([255, 105, 97, 220], dtype=np.uint8)
        self._array_shape = mask.shape
        self._mask_plane = mask.copy()
        self._boundary_plane = boundary
        self._overlay = QtGui.QImage(
            overlay.data,
            overlay.shape[1],
            overlay.shape[0],
            overlay.strides[0],
            QtGui.QImage.Format_RGBA8888,
        ).copy()
        if not self._drawing:
            self._stroke_points = []
            self._active_brush_mode = None
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

    @staticmethod
    def _mask_value(mask: np.ndarray, voxel: tuple[float, float]) -> bool:
        if mask.ndim != 2 or mask.size == 0:
            return False
        column = max(0, min(mask.shape[1] - 1, int(math.floor(float(voxel[0])))))
        row = max(0, min(mask.shape[0] - 1, int(math.floor(float(voxel[1])))))
        return bool(mask[row, column])

    @staticmethod
    def _boundary_mask(mask: np.ndarray) -> np.ndarray:
        source = np.asarray(mask, dtype=bool)
        if source.ndim != 2 or source.size == 0:
            return np.zeros_like(source, dtype=bool)
        inner = source.copy()
        inner[1:, :] &= source[:-1, :]
        inner[:-1, :] &= source[1:, :]
        inner[:, 1:] &= source[:, :-1]
        inner[:, :-1] &= source[:, 1:]
        return source & ~inner

    def _nearest_mask_boundary(
        self,
        voxel: tuple[float, float],
        mask: np.ndarray | None = None,
    ) -> tuple[float, float] | None:
        boundary = self._boundary_plane if mask is None else self._boundary_mask(mask)
        if boundary.ndim != 2 or boundary.size == 0 or not np.any(boundary):
            return None
        spacing_u = max(1.0e-4, float(self.display_spacing_uv[0]))
        spacing_v = max(1.0e-4, float(self.display_spacing_uv[1]))
        radius = max(min(spacing_u, spacing_v), float(self.boundary_capture_mm))
        column_extent = max(1, int(math.ceil(radius / spacing_u)))
        row_extent = max(1, int(math.ceil(radius / spacing_v)))
        column = int(math.floor(float(voxel[0])))
        row = int(math.floor(float(voxel[1])))
        left = max(0, column - column_extent)
        right = min(boundary.shape[1], column + column_extent + 1)
        top = max(0, row - row_extent)
        bottom = min(boundary.shape[0], row + row_extent + 1)
        candidates = np.argwhere(boundary[top:bottom, left:right])
        if candidates.size == 0:
            return None
        candidate_v = candidates[:, 0].astype(np.float64) + float(top) + 0.5
        candidate_u = candidates[:, 1].astype(np.float64) + float(left) + 0.5
        distance_sq = (
            ((candidate_u - float(voxel[0])) * spacing_u) ** 2
            + ((candidate_v - float(voxel[1])) * spacing_v) ** 2
        )
        index = int(np.argmin(distance_sq))
        if float(distance_sq[index]) > radius * radius:
            return None
        return float(candidate_u[index]), float(candidate_v[index])

    def _near_mask_boundary(self, mask: np.ndarray, voxel: tuple[float, float]) -> bool:
        return self._nearest_mask_boundary(voxel, mask) is not None

    def _automatic_mode(
        self,
        voxel: tuple[float, float],
        mask: np.ndarray | None = None,
    ) -> str:
        # A correction click labels the error under the pointer: missing Mask
        # is foreground guidance; excess Mask is protected background.
        source = self._mask_plane if mask is None else mask
        return "barrier" if self._mask_value(source, voxel) else "add"

    @staticmethod
    def _mode_color(mode: str, alpha: int) -> QtGui.QColor:
        if mode == "add":
            return QtGui.QColor(80, 220, 132, alpha)
        if mode == "barrier":
            return QtGui.QColor(255, 105, 97, alpha)
        if mode == "neutral":
            return QtGui.QColor(226, 232, 240, alpha)
        return QtGui.QColor(245, 183, 66, alpha)

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
        if self._overlay is not None:
            painter.drawImage(target, self._overlay, source)
        scale_x = target.width() / max(1.0, float(source.width()))
        scale_y = target.height() / max(1.0, float(source.height()))
        radius_x = max(3.0, self.brush_radius_voxels[0] * scale_x)
        radius_y = max(3.0, self.brush_radius_voxels[1] * scale_y)
        if self._boundary_dragging and self._boundary_start is not None:
            start = self._screen_at_voxel(self._boundary_start)
            current = self._screen_at_voxel(self._boundary_current or self._boundary_start)
            if start is not None and current is not None:
                pull_color = QtGui.QColor(255, 196, 74, 245)
                painter.setPen(QtGui.QPen(pull_color, 2.4, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
                painter.setBrush(QtCore.Qt.NoBrush)
                painter.drawLine(start, current)
                painter.setBrush(QtGui.QColor(16, 24, 32, 225))
                painter.drawEllipse(start, 5.5, 5.5)
                painter.setBrush(pull_color)
                painter.drawEllipse(current, 5.5, 5.5)
        elif self._stroke_points:
            color = self._mode_color(self._active_brush_mode or "pending", 110)
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(color)
            for point in self._stroke_points:
                screen = self._screen_at_voxel(point)
                if screen is not None:
                    painter.drawEllipse(screen, radius_x, radius_y)
        if self._cursor_pos is not None and target.contains(self._cursor_pos):
            voxel = self._voxel_at(self._cursor_pos)
            cursor_mode = self.interaction_mode
            if cursor_mode == "ffd":
                boundary_point = self._nearest_mask_boundary(voxel) if voxel is not None else None
                if boundary_point is not None:
                    screen = self._screen_at_voxel(boundary_point)
                    if screen is not None:
                        painter.setPen(QtGui.QPen(QtGui.QColor(255, 196, 74, 245), 2.0))
                        painter.setBrush(QtGui.QColor(255, 196, 74, 48))
                        painter.drawEllipse(screen, 6.0, 6.0)
                else:
                    painter.setPen(QtGui.QPen(QtGui.QColor(158, 171, 184, 190), 1.2))
                    painter.drawLine(
                        self._cursor_pos + QtCore.QPointF(-4.0, 0.0),
                        self._cursor_pos + QtCore.QPointF(4.0, 0.0),
                    )
                    painter.drawLine(
                        self._cursor_pos + QtCore.QPointF(0.0, -4.0),
                        self._cursor_pos + QtCore.QPointF(0.0, 4.0),
                    )
                return
            if cursor_mode == "auto" and voxel is not None:
                cursor_mode = self._automatic_mode(voxel)
            color = self._mode_color(cursor_mode, 225)
            painter.setPen(QtGui.QPen(color, 1.5))
            painter.setBrush(self._mode_color(cursor_mode, 28))
            painter.drawEllipse(self._cursor_pos, radius_x, radius_y)

    def enterEvent(self, _event: QtCore.QEvent) -> None:
        self.setCursor(QtCore.Qt.CrossCursor)

    def leaveEvent(self, _event: QtCore.QEvent) -> None:
        self._cursor_pos = None
        self.pointer_changed.emit(None)
        self.update()

    def _emit_pointer(self, voxel: tuple[float, float] | None) -> None:
        if voxel is None:
            self.pointer_changed.emit(None)
            return
        value = None
        if self._raw_plane is not None and self._raw_plane.size:
            column = max(0, min(self._raw_plane.shape[1] - 1, int(math.floor(voxel[0]))))
            row = max(0, min(self._raw_plane.shape[0] - 1, int(math.floor(voxel[1]))))
            candidate = float(self._raw_plane[row, column])
            if math.isfinite(candidate):
                value = candidate
        self.pointer_changed.emit({"u": float(voxel[0]), "v": float(voxel[1]), "value": value})

    def _gesture_spacing(self) -> float:
        return max(
            0.75,
            min(float(value) for value in self.brush_radius_voxels) * 0.2,
        )

    def _begin_stroke(self, mode: str, voxel: tuple[float, float]) -> None:
        self._pending_auto = False
        self._active_brush_mode = str(mode)
        self._last_voxel = voxel
        if not self._stroke_open:
            self._stroke_open = True
            self.stroke_started.emit()
        self.gesture_mode_changed.emit(str(mode))
        self._gesture_segments.append((voxel, voxel))
        self.brush_requested.emit(str(mode), voxel, voxel)

    def _emit_stroke_segment(
        self,
        voxel: tuple[float, float],
        *,
        force: bool = False,
    ) -> None:
        if self._active_brush_mode is None or self._last_voxel is None:
            return
        distance = math.hypot(
            voxel[0] - self._last_voxel[0],
            voxel[1] - self._last_voxel[1],
        )
        if distance <= 1.0e-6 or (not force and distance < self._gesture_spacing()):
            return
        start = self._last_voxel
        self._last_voxel = voxel
        self._stroke_points.append(voxel)
        self._gesture_segments.append((start, voxel))
        desired_mode = self._active_brush_mode
        crossed_boundary = False
        if (
            self._auto_stroke
            and not self._mode_locked_by_crossing
            and self._press_mask is not None
        ):
            endpoint_inside = self._mask_value(self._press_mask, voxel)
            crossed_boundary = endpoint_inside != self._press_inside
            if crossed_boundary:
                desired_mode = "barrier" if endpoint_inside else "add"
        if desired_mode != self._active_brush_mode:
            self._active_brush_mode = desired_mode
            self.gesture_mode_changed.emit(str(desired_mode))
            self.stroke_reclassified.emit(str(desired_mode), list(self._gesture_segments))
        else:
            self.brush_requested.emit(self._active_brush_mode, start, voxel)
        if crossed_boundary:
            self._mode_locked_by_crossing = True

    def _resolve_pending_auto(self, voxel: tuple[float, float]) -> None:
        if not self._pending_auto or self._press_voxel is None:
            return
        source = self._press_mask if self._press_mask is not None else self._mask_plane
        mode = self._automatic_mode(voxel, source)
        press = self._press_voxel
        self._begin_stroke(mode, press)
        self._emit_stroke_segment(voxel, force=True)

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
        self._emit_pointer(voxel)
        if self._boundary_dragging:
            if voxel is not None and self._boundary_start is not None:
                self._boundary_current = voxel
                self.boundary_drag_requested.emit(self._boundary_start, voxel)
            self.update()
            return
        if self._drawing and voxel is not None:
            if self._pending_auto and self._press_voxel is not None:
                distance = math.hypot(
                    voxel[0] - self._press_voxel[0],
                    voxel[1] - self._press_voxel[1],
                )
                if distance >= self._gesture_spacing():
                    # Near a boundary, the side reached by the drag determines
                    # whether the swept correction is missing or excessive.
                    self._resolve_pending_auto(voxel)
            else:
                self._emit_stroke_segment(voxel)
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
        requested_mode = str(self.interaction_mode or "ffd")
        if requested_mode == "ffd":
            boundary_point = self._nearest_mask_boundary(voxel)
            if boundary_point is None:
                self.gesture_mode_changed.emit("boundary_miss")
                self.update()
                return
            self._boundary_dragging = True
            self._boundary_start = boundary_point
            self._boundary_current = boundary_point
            self.boundary_drag_started.emit(boundary_point)
            self.gesture_mode_changed.emit("boundary_captured")
            self.update()
            return
        self._drawing = True
        self._stroke_open = True
        self._pending_auto = False
        self._gesture_segments = []
        self._mode_locked_by_crossing = False
        self._last_voxel = voxel
        self._press_voxel = voxel
        self._press_mask = self._mask_plane.copy()
        self._press_inside = self._mask_value(self._press_mask, voxel)
        self._active_brush_mode = None
        self._stroke_points = [voxel]
        # Stop any previous settling cycle and capture Undo at mouse-down. For
        # an ambiguous boundary press, only the Add/Barrier decision is delayed.
        self.stroke_started.emit()
        self._auto_stroke = requested_mode == "auto"
        if requested_mode == "auto" and self._near_mask_boundary(self._press_mask, voxel):
            # Delay only ambiguous boundary presses. The first meaningful drag
            # direction selects Add or Barrier and then remains locked.
            self._pending_auto = True
            self.gesture_mode_changed.emit("pending")
        else:
            mode = self._automatic_mode(voxel, self._press_mask) if requested_mode == "auto" else requested_mode
            self._begin_stroke(mode, voxel)
        self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MiddleButton and self._panning:
            self._panning = False
            self._pan_start = None
            self.setCursor(QtCore.Qt.CrossCursor)
            event.accept()
            return
        if event.button() == QtCore.Qt.LeftButton and self._boundary_dragging:
            voxel = self._voxel_at(event.position()) or self._boundary_current
            if voxel is not None and self._boundary_start is not None:
                self._boundary_current = voxel
                self.boundary_drag_requested.emit(self._boundary_start, voxel)
            self._boundary_dragging = False
            self.boundary_drag_finished.emit()
            self._boundary_start = None
            self._boundary_current = None
            self.update()
            return
        if event.button() == QtCore.Qt.LeftButton and self._drawing:
            voxel = self._voxel_at(event.position())
            if self._pending_auto:
                self._resolve_pending_auto(voxel or self._press_voxel)
            elif voxel is not None:
                self._emit_stroke_segment(voxel, force=True)
            self._drawing = False
            self._last_voxel = None
            self._press_voxel = None
            self._press_mask = None
            self._pending_auto = False
            self._auto_stroke = False
            self._press_inside = False
            self._mode_locked_by_crossing = False
            if self._stroke_open:
                self._stroke_open = False
                self.stroke_finished.emit()
            self.update()

    def _screen_at_voxel(self, voxel: tuple[float, float]) -> QtCore.QPointF | None:
        target = self._compute_target()
        source = self._view_rect if not self._view_rect.isEmpty() else self._full_source_rect()
        if target.isEmpty() or source.isEmpty():
            return None
        return QtCore.QPointF(
            target.left() + (float(voxel[0]) - source.left()) * target.width() / source.width(),
            target.top() + (float(voxel[1]) - source.top()) * target.height() / source.height(),
        )

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
        self.orientation = "axial"
        self._stroke_context: dict[str, Any] | None = None
        self._boundary_context: dict[str, Any] | None = None
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
        self._tool_changed()
        self._set_orientation("axial")
        if self.preview:
            self._install_preview()
        else:
            self.worker = EngineThread(self.request, self)
            self.worker.ready.connect(self._on_ready)
            self.worker.frame_ready.connect(self._on_frame)
            self.worker.metrics_ready.connect(self._on_metrics)
            self.worker.phase_changed.connect(self._set_phase)
            self.worker.undo_available.connect(self._set_undo_available)
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
        self.pointer_label = QtWidgets.QLabel("Move over the image to inspect a voxel")
        self.pointer_label.setObjectName("hint")
        self.statusBar().setSizeGripEnabled(False)
        self.statusBar().addWidget(self.pointer_label, 1)

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

        tools = QtWidgets.QGroupBox("Correction")
        tools_layout = QtWidgets.QVBoxLayout(tools)
        tools_layout.setSpacing(9)
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel("Mode"))
        self.correction_mode = QtWidgets.QComboBox()
        self.correction_mode.addItem("Boundary drag (FFD)", "ffd")
        self.correction_mode.addItem("Correct by stroke", "auto")
        self.correction_mode.addItem("Always add", "add")
        self.correction_mode.addItem("Always remove", "barrier")
        self.correction_mode.addItem("Clear local guidance", "neutral")
        mode_row.addWidget(self.correction_mode, 1)
        tools_layout.addLayout(mode_row)
        self.tool_hint = QtWidgets.QLabel(
            "Grab a highlighted Mask boundary point and drag it to the intended edge."
        )
        self.tool_hint.setObjectName("hint")
        self.tool_hint.setWordWrap(True)
        tools_layout.addWidget(self.tool_hint)
        self.brush_controls = QtWidgets.QWidget()
        brush_controls_layout = QtWidgets.QVBoxLayout(self.brush_controls)
        brush_controls_layout.setContentsMargins(0, 0, 0, 0)
        brush_controls_layout.setSpacing(5)
        brush_row = QtWidgets.QHBoxLayout()
        brush_row.addWidget(QtWidgets.QLabel("Brush size"))
        brush_row.addStretch(1)
        self.brush_value = QtWidgets.QLabel("6.0 mm")
        self.brush_value.setObjectName("preview")
        brush_row.addWidget(self.brush_value)
        brush_controls_layout.addLayout(brush_row)
        self.brush_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.brush_slider.setRange(2, 40)
        self.brush_slider.setValue(12)
        self.brush_slider.setToolTip("Physical brush diameter in millimetres")
        brush_controls_layout.addWidget(self.brush_slider)
        tools_layout.addWidget(self.brush_controls)

        ffd_defaults = dict((self.request.get("igac") or {}).get("ffd") or {})
        self.pull_controls = QtWidgets.QWidget()
        pull_controls_layout = QtWidgets.QVBoxLayout(self.pull_controls)
        pull_controls_layout.setContentsMargins(0, 0, 0, 0)
        pull_controls_layout.setSpacing(5)
        pull_row = QtWidgets.QHBoxLayout()
        pull_row.addWidget(QtWidgets.QLabel("Influence radius"))
        pull_row.addStretch(1)
        pull_default = max(
            4,
            min(80, int(round(float(ffd_defaults.get("influence_radius_mm", 18.0))))),
        )
        self.pull_value = QtWidgets.QLabel("{:.1f} mm".format(float(pull_default)))
        self.pull_value.setObjectName("preview")
        pull_row.addWidget(self.pull_value)
        pull_controls_layout.addLayout(pull_row)
        self.pull_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.pull_slider.setRange(4, 80)
        self.pull_slider.setValue(pull_default)
        self.pull_slider.setToolTip(
            "Physical radius of the local 3D B-spline deformation around the grabbed point"
        )
        pull_controls_layout.addWidget(self.pull_slider)
        tools_layout.addWidget(self.pull_controls)
        side.addWidget(tools)

        evolution = QtWidgets.QGroupBox("Evolution")
        evolution_layout = QtWidgets.QVBoxLayout(evolution)
        self.run_button = QtWidgets.QPushButton("Fit boundary")
        self.run_button.setObjectName("primary")
        self.run_button.setToolTip(
            "Run LGDF until the visible Mask stabilizes or a safety limit is reached"
        )
        recovery_row = QtWidgets.QHBoxLayout()
        self.undo_button = QtWidgets.QPushButton("Undo last")
        self.undo_button.setEnabled(False)
        self.undo_button.setToolTip("Restore the Mask and guidance from before the last action")
        self.reset_button = QtWidgets.QPushButton("Reset")
        self.reset_button.setToolTip("Restore the Mask exactly as it was when IGAC opened")
        recovery_row.addWidget(self.undo_button)
        recovery_row.addWidget(self.reset_button)
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

        result_group = QtWidgets.QGroupBox("Session")
        result_layout = QtWidgets.QVBoxLayout(result_group)
        destination_widget, self.destination_group = make_segmented(
            [("Update selected", "update"), ("Editable copy", "copy")],
            "copy",
        )
        result_layout.addWidget(self.run_button)
        result_layout.addLayout(recovery_row)
        result_layout.addWidget(destination_widget)
        self.apply_button = QtWidgets.QPushButton("Apply to Mimics")
        self.apply_button.setObjectName("primary")
        self.apply_button.setEnabled(False)
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setObjectName("dangerButton")
        result_layout.addWidget(self.apply_button)
        result_layout.addWidget(self.cancel_button)
        side.addStretch(1)

        self.roi_label = QtWidgets.QLabel("Preparing 3D workspace")
        self.roi_label.setObjectName("hint")
        self.roi_label.setWordWrap(True)
        side.addWidget(self.roi_label)
        scroll.setWidget(sidebar)
        right = QtWidgets.QWidget()
        right.setFixedWidth(350)
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)
        right_layout.addWidget(scroll, 1)
        # Applying or discarding is always reachable; only tuning controls
        # scroll on short or high-DPI Windows displays.
        right_layout.addWidget(result_group, 0)
        body.addWidget(right)
        outer.addLayout(body, 1)

    def _connect_ui(self) -> None:
        self.orientation_group.buttonClicked.connect(
            lambda button: self._set_orientation(str(button.property("segmentValue")))
        )
        self.correction_mode.currentIndexChanged.connect(self._tool_changed)
        self.slice_slider.valueChanged.connect(self._slice_changed)
        self.slice_spin.valueChanged.connect(self.slice_slider.setValue)
        self.fit_mask_button.clicked.connect(self._fit_mask_view)
        self.fit_image_button.clicked.connect(self.canvas.fit_image)
        self.zoom_out_button.clicked.connect(lambda: self.canvas.zoom(0.8))
        self.zoom_in_button.clicked.connect(lambda: self.canvas.zoom(1.25))
        self.canvas.slice_wheel.connect(lambda delta: self.slice_slider.setValue(self.slice_slider.value() + delta))
        self.canvas.pointer_changed.connect(self._pointer_changed)
        self.canvas.brush_requested.connect(self._brush)
        self.canvas.stroke_started.connect(self._stroke_started)
        self.canvas.stroke_finished.connect(self._stroke_finished)
        self.canvas.boundary_drag_started.connect(self._boundary_drag_started)
        self.canvas.boundary_drag_requested.connect(self._boundary_drag_requested)
        self.canvas.boundary_drag_finished.connect(self._boundary_drag_finished)
        self.canvas.gesture_mode_changed.connect(self._gesture_mode_changed)
        self.canvas.stroke_reclassified.connect(self._stroke_reclassified)
        self.brush_slider.valueChanged.connect(self._brush_size_changed)
        self.pull_slider.valueChanged.connect(self._pull_size_changed)
        self.display_mimics_button.clicked.connect(self._use_mimics_display)
        self.display_auto_button.clicked.connect(self._use_auto_display)
        self.display_width.valueChanged.connect(self._display_controls_changed)
        self.display_level.valueChanged.connect(self._display_controls_changed)
        self.run_button.clicked.connect(self._toggle_running)
        self.undo_button.clicked.connect(self._undo_last)
        self.reset_button.clicked.connect(self._reset)
        self.limit_movement.toggled.connect(self.max_displacement.setEnabled)
        self.limit_movement.toggled.connect(lambda _checked: self._parameters_changed())
        self.max_displacement.valueChanged.connect(lambda _value: self._parameters_changed())
        self.range_row.value_changed.connect(lambda _value: self._parameters_changed())
        self.smooth_row.value_changed.connect(lambda _value: self._parameters_changed())
        self.grow_row.value_changed.connect(lambda _value: self._parameters_changed())
        self.apply_button.clicked.connect(self._apply)
        self.cancel_button.clicked.connect(self._cancel)

    def _pointer_changed(self, values: dict[str, Any] | None) -> None:
        if not values:
            self.pointer_label.setText("Move over the image to inspect a voxel")
            return
        x, y, z = display_to_xyz(
            self.orientation,
            self.slice_slider.value(),
            float(values["u"]),
            float(values["v"]),
        )
        value = values.get("value")
        value_text = "—" if value is None else "{:.1f}".format(float(value))
        self.pointer_label.setText(
            "Voxel x {:.1f}  y {:.1f}  z {:.1f}    Image value {}".format(x, y, z, value_text)
        )

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

    def _pull_radius_mm(self) -> float:
        return float(self.pull_slider.value())

    def _pull_size_changed(self, _value: int) -> None:
        self.pull_value.setText("{:.1f} mm".format(self._pull_radius_mm()))

    def _update_canvas_brush_radius(self) -> None:
        if self.orientation == "axial":
            horizontal_spacing, vertical_spacing = self.spacing_xyz[0], self.spacing_xyz[1]
        elif self.orientation == "coronal":
            horizontal_spacing, vertical_spacing = self.spacing_xyz[0], self.spacing_xyz[2]
        else:
            horizontal_spacing, vertical_spacing = self.spacing_xyz[1], self.spacing_xyz[2]
        self.canvas.display_spacing_uv = (horizontal_spacing, vertical_spacing)
        ffd_values = dict((self.request.get("igac") or {}).get("ffd") or {})
        self.canvas.boundary_capture_mm = max(
            min(horizontal_spacing, vertical_spacing),
            float(ffd_values.get("boundary_capture_mm", 3.0)),
        )
        self.canvas.brush_radius_voxels = (
            self._brush_radius_mm() / max(0.01, horizontal_spacing),
            self._brush_radius_mm() / max(0.01, vertical_spacing),
        )
        self.canvas.update()

    def _boundary_drag_started(self, start: tuple[float, float]) -> None:
        if self.worker is None or not self.ready:
            return
        values = dict((self.request.get("igac") or {}).get("ffd") or {})
        self._boundary_context = {
            "orientation": self.orientation,
            "slice_index": self.slice_slider.value(),
            "start": tuple(start),
            "influence_radius_mm": self._pull_radius_mm(),
            "maximum_drag_ratio": float(values.get("maximum_drag_ratio", 0.65)),
            "anchor_radius_mm": float(values.get("anchor_radius_mm", 0.8)),
        }
        self.worker.submit("boundary_pull_start")

    def _boundary_drag_requested(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> None:
        if self.worker is None or not self.ready or self._boundary_context is None:
            return
        context = self._boundary_context
        self.worker.submit(
            "boundary_pull_update",
            {
                "orientation": str(context["orientation"]),
                "slice_index": int(context["slice_index"]),
                "start_u": float(start[0]),
                "start_v": float(start[1]),
                "end_u": float(end[0]),
                "end_v": float(end[1]),
                "influence_radius_mm": float(context["influence_radius_mm"]),
                "maximum_drag_ratio": float(context["maximum_drag_ratio"]),
                "anchor_radius_mm": float(context["anchor_radius_mm"]),
            },
        )

    def _boundary_drag_finished(self) -> None:
        if self.worker is not None and self.ready and self._boundary_context is not None:
            self.worker.submit("boundary_pull_end")
        self._boundary_context = None

    def _brush(
        self,
        mode: str,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> None:
        if self.worker is None or not self.ready:
            return
        context = self._stroke_context or {
            "orientation": self.orientation,
            "slice_index": self.slice_slider.value(),
            "radius_mm": self._brush_radius_mm(),
        }
        self.worker.submit(
            "brush_segment",
            {
                "mode": str(mode),
                "orientation": str(context["orientation"]),
                "slice_index": int(context["slice_index"]),
                "start_u": float(start[0]),
                "start_v": float(start[1]),
                "end_u": float(end[0]),
                "end_v": float(end[1]),
                "radius_mm": float(context["radius_mm"]),
            },
        )

    def _stroke_reclassified(
        self,
        mode: str,
        segments: list[tuple[tuple[float, float], tuple[float, float]]],
    ) -> None:
        if self.worker is None or not self.ready or not segments:
            return
        context = self._stroke_context or {
            "orientation": self.orientation,
            "slice_index": self.slice_slider.value(),
            "radius_mm": self._brush_radius_mm(),
        }
        payload = []
        for start, end in segments:
            payload.append(
                {
                    "mode": str(mode),
                    "orientation": str(context["orientation"]),
                    "slice_index": int(context["slice_index"]),
                    "start_u": float(start[0]),
                    "start_v": float(start[1]),
                    "end_u": float(end[0]),
                    "end_v": float(end[1]),
                    "radius_mm": float(context["radius_mm"]),
                }
            )
        self.worker.submit("reclassify_stroke", payload)

    def _tool_changed(self, _index: int = -1) -> None:
        mode = str(self.correction_mode.currentData() or "ffd")
        self.canvas.interaction_mode = mode
        self.pull_controls.setVisible(mode == "ffd")
        self.brush_controls.setVisible(mode != "ffd")
        if mode == "ffd":
            self.tool_hint.setText(
                "Grab a highlighted Mask boundary point and drag it to the intended edge."
            )
        elif mode == "auto":
            self.tool_hint.setText(
                "Paint across an error. Start outside to add missing Mask; start inside to remove excess Mask."
            )
        elif mode == "add":
            self.tool_hint.setText("Force the clicked or painted region to remain inside the Mask.")
        elif mode == "barrier":
            self.tool_hint.setText("Force the clicked or painted region to remain outside the Mask.")
        else:
            self.tool_hint.setText("Clear local guidance and remove Mask from the painted region.")
        self.canvas.update()

    def _gesture_mode_changed(self, mode: str) -> None:
        labels = {
            "add": "Adding missing region",
            "barrier": "Removing excess region",
            "neutral": "Clearing local correction",
            "pending": "Drag inward or outward",
            "boundary_captured": "Boundary point captured - drag to the intended edge",
            "boundary_miss": "Start directly on the highlighted Mask boundary",
        }
        self.tool_hint.setText(labels.get(str(mode), "Correcting boundary"))

    def _stroke_started(self) -> None:
        if self.worker is not None and self.ready:
            self._stroke_context = {
                "orientation": self.orientation,
                "slice_index": self.slice_slider.value(),
                "radius_mm": self._brush_radius_mm(),
            }
            self.worker.submit("stroke_start")

    def _stroke_finished(self) -> None:
        if self.worker is not None and self.ready:
            self.worker.submit("stroke_end")
        self._stroke_context = None

    def _toggle_running(self, _checked: bool = False) -> None:
        if self.worker is None or not self.ready:
            return
        if bool(self.run_button.property("running")):
            self.run_button.setProperty("running", False)
            self.run_button.setText("Fit boundary")
            self.worker.submit("stop")
            return
        self.run_button.setProperty("running", True)
        self.run_button.setText("Stop")
        self.worker.submit("fit")

    def _undo_last(self) -> None:
        if self.worker is not None and self.ready:
            self.worker.submit("undo")

    def _set_undo_available(self, available: bool) -> None:
        self.undo_button.setEnabled(bool(available) and self.ready)

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
        if hasattr(self, "run_button"):
            running = str(phase) == "running"
            self.run_button.setProperty("running", running)
            self.run_button.setText("Stop" if running else "Fit boundary")

    def _on_failed(self, detail: str) -> None:
        self.ready = False
        self.apply_button.setEnabled(False)
        self.undo_button.setEnabled(False)
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
        self._set_phase("paused", "Ready - correct an error")
        self.roi_label.setText("Workspace 196 × 182 × 144 · about 544 MB")
        self.volume_label.setText("Mask 38.4 cm³")
        self.iteration_label.setText("Iteration 0")
        self.speed_label.setText("Paused")
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
        self.canvas.set_frame(
            {
                "image": np.clip(image, 0, 1),
                "raw_image": np.asarray(image * 1000.0, dtype=np.float32),
                "mask": mask,
            }
        )
        self.canvas._boundary_dragging = True
        self.canvas._boundary_start = (459.5, 325.5)
        self.canvas._boundary_current = (486.0, 310.0)
        self.canvas.update()

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
