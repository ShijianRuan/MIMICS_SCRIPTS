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
    orthogonal_views_ready = QtCore.Signal(object)
    metrics_ready = QtCore.Signal(object)
    phase_changed = QtCore.Signal(str, str)
    failed = QtCore.Signal(str)
    terminal = QtCore.Signal(str)
    undo_available = QtCore.Signal(bool)
    boundary_preview = QtCore.Signal(object)
    contour_trace_preview = QtCore.Signal(object)
    local_refinement_available = QtCore.Signal(bool)

    def __init__(self, request: dict[str, Any], parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.request = request
        self.commands: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.cancelled = False
        self.writer = StatusWriter(request)
        self.orientation = "axial"
        shape = tuple(int(value) for value in request["shape"])
        self.slice_index = max(0, shape[2] // 2)
        self.crosshair_xyz = [
            max(0.0, (float(shape[0]) - 1.0) / 2.0),
            max(0.0, (float(shape[1]) - 1.0) / 2.0),
            max(0.0, (float(shape[2]) - 1.0) / 2.0),
        ]
        # Opening the editor must never mutate the Mask. Evolution is bounded
        # and starts only after an explicit user action.
        self.running = False
        self.stroke_active = False
        self.engine: IGACEngine | None = None
        self.lock: FileResourceLock | None = None
        self._last_status_write = 0.0
        self._last_metrics_emit = 0.0
        self._last_image_frame_key: tuple[Any, ...] | None = None
        self._last_context_image_keys: dict[str, tuple[Any, ...]] = {}
        self._last_context_emit = 0.0
        self._force_context_refresh = True
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
        self._pending_trace_lock = threading.Lock()
        self._pending_trace_batch: list[dict[str, Any]] | None = None
        self._pending_crosshair_lock = threading.Lock()
        self._pending_crosshair_holder: dict[str, Any] | None = None
        self._boundary_base_state: dict[str, Any] | None = None
        self._undo_state_before_boundary: dict[str, Any] | None = None
        self._boundary_changed = False
        self._boundary_preview_slices: tuple[slice, slice, slice] | None = None
        self._undo_state_before_trace: dict[str, Any] | None = None
        self._trace_changed = False
        self._trace_last_target: tuple[float, float] | None = None
        self._stroke_changed = False

    def submit(self, command: str, payload: Any = None) -> None:
        if str(command) == "crosshair":
            queue_update = False
            with self._pending_crosshair_lock:
                if self._pending_crosshair_holder is None:
                    self._pending_crosshair_holder = {"payload": tuple(payload or ())}
                    queue_update = True
                else:
                    self._pending_crosshair_holder["payload"] = tuple(payload or ())
                holder = self._pending_crosshair_holder
            if queue_update:
                self.commands.put(("crosshair_flush", holder))
            return
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
        if str(command) == "contour_trace_update":
            queue_batch = False
            with self._pending_trace_lock:
                if self._pending_trace_batch is None:
                    self._pending_trace_batch = []
                    queue_batch = True
                batch = self._pending_trace_batch
                batch.append(dict(payload or {}))
                # Bound input memory if a slow device produces mouse events
                # faster than the worker can consume them. Endpoints and path
                # order are preserved; worker-side physical resampling handles
                # the reduced polyline.
                if len(batch) > 64:
                    batch[:] = batch[:1] + batch[2:-1:2] + batch[-1:]
            if queue_batch:
                self.commands.put(("contour_trace_flush", batch))
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

    def _take_pending_crosshair(self, holder: dict[str, Any]) -> tuple[float, ...]:
        with self._pending_crosshair_lock:
            payload = tuple(float(value) for value in holder.get("payload") or ())
            if self._pending_crosshair_holder is holder:
                self._pending_crosshair_holder = None
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

    def _take_pending_trace_targets(
        self, batch: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        with self._pending_trace_lock:
            if self._pending_trace_batch is batch:
                self._pending_trace_batch = None
            targets = list(batch)
            batch[:] = []
        return targets

    def _resample_trace_targets(
        self, targets: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Resample a coalesced pointer polyline in physical display space."""
        if not targets:
            return []
        first = dict(targets[0])
        orientation = str(first.get("orientation") or self.orientation).lower()
        if self.engine is not None:
            spacing_xyz = self.engine.spacing_xyz
        else:
            spacing_xyz = tuple(
                float(value) for value in self.request.get("spacing_mm") or (1.0, 1.0, 1.0)
            )
        if orientation == "axial":
            spacing_u, spacing_v = spacing_xyz[0], spacing_xyz[1]
        elif orientation == "coronal":
            spacing_u, spacing_v = spacing_xyz[0], spacing_xyz[2]
        else:
            spacing_u, spacing_v = spacing_xyz[1], spacing_xyz[2]
        trace_values = dict((self.request.get("igac") or {}).get("trace") or {})
        sample_spacing = max(
            0.5,
            float(trace_values.get("sample_spacing_mm", 2.0)),
        )
        maximum_samples = max(
            1,
            int(trace_values.get("maximum_samples_per_flush", 8)),
        )
        previous = self._trace_last_target
        if previous is None:
            previous = (
                float(first.get("target_u", 0.0)),
                float(first.get("target_v", 0.0)),
            )
        points = [previous]
        for target in targets:
            point = (float(target["target_u"]), float(target["target_v"]))
            if point != points[-1]:
                points.append(point)
        if len(points) == 1:
            self._trace_last_target = points[0]
            return []
        lengths = [0.0]
        for start, end in zip(points[:-1], points[1:]):
            distance = math.hypot(
                (end[0] - start[0]) * spacing_u,
                (end[1] - start[1]) * spacing_v,
            )
            lengths.append(lengths[-1] + distance)
        total = lengths[-1]
        count = min(maximum_samples, max(1, int(math.ceil(total / sample_spacing))))
        distances = np.linspace(total / count, total, count)
        output: list[dict[str, Any]] = []
        segment = 0
        template = dict(targets[-1])
        for distance in distances:
            while segment + 1 < len(lengths) - 1 and distance > lengths[segment + 1]:
                segment += 1
            span = max(1.0e-8, lengths[segment + 1] - lengths[segment])
            amount = float((distance - lengths[segment]) / span)
            start, end = points[segment], points[segment + 1]
            value = dict(template)
            value["target_u"] = start[0] + (end[0] - start[0]) * amount
            value["target_v"] = start[1] + (end[1] - start[1]) * amount
            output.append(value)
        self._trace_last_target = points[-1]
        return output

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
        self._boundary_preview_slices = None
        self._undo_state_before_trace = None
        self._trace_last_target = None
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
            changed_voxels=int(self.engine.last_step_changed_voxels),
            observed_voxels=int(self.engine.last_step_observed_voxels),
        )
        if reason is None:
            return False
        if reason == "converged":
            text = "Local boundary stable"
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
            observed_voxels=int(self.engine.last_step_observed_voxels),
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
                self.engine.begin_local_edit()
                self._stroke_changed = False
                self.phase_changed.emit("editing", "Applying guidance")
            elif command == "boundary_pull_start" and self.engine is not None:
                self.running = False
                self.stroke_active = True
                self._undo_state_before_boundary = self._undo_state
                self._capture_undo_state()
                self._boundary_base_state = self._undo_state
                self._boundary_changed = False
                self._boundary_preview_slices = None
                self.engine.begin_local_edit()
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
                    incremental_preview=True,
                    previous_slices_zyx=self._boundary_preview_slices,
                    **values
                )
                if result is not None:
                    self._boundary_changed = True
                    self._boundary_preview_slices = result.get("slices_zyx")
                    # Keep pointer tracking lightweight. The FFD preview is
                    # updated while dragging; LGDF edge fitting begins only on
                    # release so an expensive iteration cannot stall the mouse.
                    self.running = False
                    if bool(result.get("drag_was_clamped")):
                        self.phase_changed.emit(
                            "warning", "Local pull limit reached - use another short pull"
                        )
                    else:
                        self.phase_changed.emit("editing", "Boundary follows pointer")
                    self.boundary_preview.emit(dict(result))
                    self.writer.update(
                        "interactive",
                        25,
                        "IGAC boundary drag preview updated.",
                        progress_indeterminate=True,
                        drag_distance_mm=round(float(result["drag_distance_mm"]), 2),
                        applied_drag_distance_mm=round(
                            float(result["applied_drag_distance_mm"]), 2
                        ),
                        influence_radius_mm=round(float(result["influence_radius_mm"]), 2),
                        drag_was_clamped=bool(result.get("drag_was_clamped")),
                    )
                else:
                    self._boundary_changed = False
                    self._boundary_preview_slices = None
                    self.running = False
                    self.boundary_preview.emit(None)
                    self.phase_changed.emit("editing", "Boundary returned to start")
                refresh = True
            elif command == "boundary_pull_cancel" and self.engine is not None:
                self.running = False
                self.stroke_active = False
                if self._boundary_base_state is not None:
                    self.engine.restore_state(self._boundary_base_state)
                self._undo_state = self._undo_state_before_boundary
                self.undo_available.emit(self._undo_state is not None)
                self.local_refinement_available.emit(
                    self.engine.evolution_slices_zyx is not None
                )
                self._boundary_base_state = None
                self._undo_state_before_boundary = None
                self._boundary_changed = False
                self._boundary_preview_slices = None
                self.boundary_preview.emit(None)
                self.phase_changed.emit("paused", "Contour drag cancelled")
                refresh = True
            elif command == "boundary_pull_end":
                self.stroke_active = False
                self._boundary_base_state = None
                if self._boundary_changed:
                    values = self.request.get("igac") or {}
                    ffd_values = values.get("ffd") or {}
                    self.engine.set_local_evolution_region(
                        self._boundary_preview_slices,
                        float(ffd_values.get("refine_margin_mm", 3.0)),
                    )
                    self.local_refinement_available.emit(True)
                    self._start_evolution("boundary_pull", "Refining the local boundary")
                else:
                    self._undo_state = self._undo_state_before_boundary
                    self.undo_available.emit(self._undo_state is not None)
                    self.phase_changed.emit("paused", "Boundary drag cancelled")
                self._undo_state_before_boundary = None
                self._boundary_changed = False
                self._boundary_preview_slices = None
                self.boundary_preview.emit(None)
                refresh = True
            elif command == "contour_trace_start" and self.engine is not None:
                self.running = False
                self.stroke_active = True
                self._undo_state_before_trace = self._undo_state
                self._capture_undo_state()
                self.engine.begin_local_edit()
                self._trace_changed = False
                start = dict(payload or {})
                self._trace_last_target = (
                    float(start.get("target_u", 0.0)),
                    float(start.get("target_v", 0.0)),
                )
                self.phase_changed.emit("editing", "Contour ready to follow the cursor")
            elif command == "contour_trace_flush" and self.engine is not None:
                targets = self._resample_trace_targets(
                    self._take_pending_trace_targets(payload)
                )
                applied = False
                clamped = False
                last_result = None
                for values in targets:
                    result = self.engine.apply_contour_attraction(**values)
                    if result is not None:
                        applied = True
                        last_result = result
                        clamped = clamped or bool(result.get("drag_was_clamped"))
                if applied:
                    self._trace_changed = True
                    self.running = False
                    self.contour_trace_preview.emit(dict(last_result))
                    self.phase_changed.emit(
                        "editing",
                        "Contour follows cursor"
                        if not clamped
                        else "Pointer moved beyond the local pull range",
                    )
                    self.writer.update(
                        "interactive",
                        25,
                        "IGAC contour guidance preview updated.",
                        progress_indeterminate=True,
                    )
                elif targets:
                    self.phase_changed.emit(
                        "warning", "Move closer to the cyan contour and continue"
                    )
                refresh = refresh or bool(targets)
            elif command == "contour_trace_cancel" and self.engine is not None:
                self.running = False
                self.stroke_active = False
                if self._undo_state is not None:
                    self.engine.restore_state(self._undo_state)
                self._undo_state = self._undo_state_before_trace
                self.undo_available.emit(self._undo_state is not None)
                self.local_refinement_available.emit(
                    self.engine.evolution_slices_zyx is not None
                )
                self._undo_state_before_trace = None
                self._trace_changed = False
                self._trace_last_target = None
                self.contour_trace_preview.emit(None)
                self.phase_changed.emit("paused", "Edge guidance cancelled")
                refresh = True
            elif command == "contour_trace_end" and self.engine is not None:
                self.stroke_active = False
                self._trace_last_target = None
                if self._trace_changed:
                    trace_values = dict((self.request.get("igac") or {}).get("trace") or {})
                    self.engine.set_local_evolution_region(
                        self.engine.last_edit_slices_zyx,
                        float(trace_values.get("refine_margin_mm", 3.0)),
                    )
                    self.local_refinement_available.emit(True)
                    self._start_evolution("contour_trace", "Refining the guided contour")
                else:
                    self._undo_state = self._undo_state_before_trace
                    self.undo_available.emit(self._undo_state is not None)
                    self.phase_changed.emit("paused", "Contour guidance cancelled")
                self._undo_state_before_trace = None
                self._trace_changed = False
                self.contour_trace_preview.emit(None)
                refresh = True
            elif command == "stroke_end":
                self.stroke_active = False
                if self._stroke_changed:
                    values = self.request.get("igac") or {}
                    self.engine.set_local_evolution_region(
                        self.engine.last_edit_slices_zyx,
                        float(values.get("local_refine_margin_mm", 3.0)),
                    )
                    self.local_refinement_available.emit(True)
                    self._start_evolution("guidance", "Refining the local boundary")
                else:
                    self.phase_changed.emit("paused", "No local guidance was applied")
                self._stroke_changed = False
                refresh = True
            elif command == "brush_flush" and self.engine is not None:
                applied = False
                for segment in self._take_pending_brush_segments(payload):
                    applied = self.engine.apply_brush_segment(**segment) or applied
                if applied:
                    self._stroke_changed = True
                    self.running = False
                    self.phase_changed.emit("editing", "Guidance follows pointer")
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
                self.engine.begin_local_edit()
                applied = False
                for segment in list(payload or []):
                    applied = self.engine.apply_brush_segment(**segment) or applied
                if applied:
                    self._stroke_changed = True
                    self.running = False
                    self.phase_changed.emit("editing", "Guidance follows pointer")
                else:
                    self.phase_changed.emit("warning", "Guidance is outside the workspace")
                refresh = True
            elif command in ("fit", "refine") and self.engine is not None:
                if self.engine.evolution_slices_zyx is None:
                    self.phase_changed.emit(
                        "warning", "Make a local boundary correction before refining"
                    )
                    continue
                self._capture_undo_state()
                self.stroke_active = False
                self._start_evolution("manual", "Refining the local boundary")
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
                    self.local_refinement_available.emit(
                        self.engine.evolution_slices_zyx is not None
                    )
                    self.phase_changed.emit("paused", "Last action undone")
                    refresh = True
            elif command == "view":
                orientation, index = payload
                self.orientation = str(orientation)
                self.slice_index = int(index)
                depth_axis = {"axial": 2, "coronal": 1, "sagittal": 0}[self.orientation]
                self.crosshair_xyz[depth_axis] = float(self.slice_index)
                self._force_context_refresh = True
                refresh = True
            elif command == "crosshair_flush":
                values = self._take_pending_crosshair(payload)
                if len(values) != 3:
                    continue
                shape = tuple(int(value) for value in self.request["shape"])
                self.crosshair_xyz = [
                    max(0.0, min(float(shape[axis] - 1), float(values[axis])))
                    for axis in range(3)
                ]
                depth_axis = {"axial": 2, "coronal": 1, "sagittal": 0}[self.orientation]
                self.slice_index = int(round(self.crosshair_xyz[depth_axis]))
                self._force_context_refresh = True
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
                self.local_refinement_available.emit(False)
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
        values = self.request.get("igac") or {}
        context_interval = max(
            0.08,
            float(values.get("context_display_interval_seconds", 0.12)),
        )
        if self._force_context_refresh or now - self._last_context_emit >= context_interval:
            views = {}
            depth_axes = {"axial": 2, "coronal": 1, "sagittal": 0}
            for orientation, depth_axis in depth_axes.items():
                if orientation == self.orientation:
                    continue
                index = int(round(self.crosshair_xyz[depth_axis]))
                context_key = (
                    str(orientation),
                    index,
                    float(self.engine.display_low),
                    float(self.engine.display_high),
                )
                include_context_image = (
                    self._last_context_image_keys.get(orientation) != context_key
                )
                views[orientation] = {
                    "index": index,
                    "frame": self.engine.frame(
                        orientation,
                        index,
                        include_image=include_context_image,
                    ),
                }
                if include_context_image:
                    self._last_context_image_keys[orientation] = context_key
            self.orthogonal_views_ready.emit(
                {"crosshair_xyz": list(self.crosshair_xyz), "views": views}
            )
            self._last_context_emit = now
            self._force_context_refresh = False
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
            bbox = metadata.get("initial_bbox_xyz") or []
            if len(bbox) == 3:
                self.crosshair_xyz = [
                    (float(bounds[0]) + float(bounds[1]) - 1.0) / 2.0
                    for bounds in bbox
                ]
                self.slice_index = int(round(self.crosshair_xyz[2]))
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
    contour_trace_started = QtCore.Signal(object)
    contour_trace_requested = QtCore.Signal(object)
    contour_trace_finished = QtCore.Signal()
    interaction_active_changed = QtCore.Signal(bool)
    interaction_cancel_requested = QtCore.Signal(str)
    navigation_requested = QtCore.Signal(object)
    promote_requested = QtCore.Signal()
    gesture_mode_changed = QtCore.Signal(str)
    stroke_reclassified = QtCore.Signal(str, object)
    slice_wheel = QtCore.Signal(int)
    pointer_changed = QtCore.Signal(object)

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        compact: bool = False,
        read_only: bool = False,
    ) -> None:
        super().__init__(parent)
        self.compact = bool(compact)
        self.read_only = bool(read_only)
        if self.compact:
            self.setMinimumSize(230, 190)
        else:
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
        self._boundary_segments: list[QtCore.QLineF] = []
        self._boundary_path_exact = QtGui.QPainterPath()
        self._boundary_path_smooth = QtGui.QPainterPath()
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
        self._boundary_applied_current: tuple[float, float] | None = None
        self._boundary_drag_clamped = False
        self._contour_tracing = False
        self._trace_current: tuple[float, float] | None = None
        self._trace_applied_current: tuple[float, float] | None = None
        self._trace_drag_clamped = False
        self._trace_path: list[tuple[float, float]] = []
        self._interaction_origin_segments: list[QtCore.QLineF] = []
        self._interaction_origin_path_exact = QtGui.QPainterPath()
        self._interaction_origin_path_smooth = QtGui.QPainterPath()
        self._hover_boundary_point: tuple[float, float] | None = None
        self._boundary_miss_pos: QtCore.QPointF | None = None
        self._boundary_miss_until = 0.0
        self.interaction_mode = "ffd"
        self._panning = False
        self._pan_start: QtCore.QPointF | None = None
        self._pan_view = QtCore.QRectF()
        self.brush_radius_voxels = (8.0, 8.0)
        self.display_spacing_uv = (1.0, 1.0)
        self.boundary_capture_mm = 8.0
        self.boundary_capture_pixels = 16.0
        self.trace_capture_mm = 12.0
        self.trace_capture_pixels = 28.0
        self.influence_radius_mm = 8.0
        self.phase = "loading"
        self.phase_text = "Loading"
        self.view_orientation = "axial"
        self.view_slice_index = 0
        self.linked_crosshair_uv: tuple[float, float] | None = None
        self._loading_bar = QtWidgets.QProgressBar(self)
        self._loading_bar.setRange(0, 0)
        self._loading_bar.setTextVisible(False)
        self._loading_bar.setFixedSize(230, 5)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._loading_bar.move(
            max(0, (self.width() - self._loading_bar.width()) // 2),
            max(0, self.height() // 2 + 30),
        )

    def set_phase(self, phase: str, text: str) -> None:
        self.phase = str(phase)
        self.phase_text = str(text)
        self.update()

    def set_view_context(
        self,
        orientation: str,
        slice_index: int,
        crosshair_xyz: tuple[float, float, float] | list[float] | None,
    ) -> None:
        self.view_orientation = str(orientation)
        self.view_slice_index = int(slice_index)
        if crosshair_xyz is None or len(crosshair_xyz) != 3:
            self.linked_crosshair_uv = None
        elif self.view_orientation == "axial":
            self.linked_crosshair_uv = (float(crosshair_xyz[0]), float(crosshair_xyz[1]))
        elif self.view_orientation == "coronal":
            self.linked_crosshair_uv = (float(crosshair_xyz[0]), float(crosshair_xyz[2]))
        else:
            self.linked_crosshair_uv = (float(crosshair_xyz[1]), float(crosshair_xyz[2]))
        self.update()

    def set_boundary_preview(self, values: dict[str, Any] | None) -> None:
        if values:
            self._boundary_applied_current = (
                float(values.get("applied_end_u", 0.0)),
                float(values.get("applied_end_v", 0.0)),
            )
            self._boundary_drag_clamped = bool(values.get("drag_was_clamped"))
        else:
            self._boundary_applied_current = None
            self._boundary_drag_clamped = False
        self.update()

    def set_contour_trace_preview(self, values: dict[str, Any] | None) -> None:
        if values:
            self._trace_applied_current = (
                float(values.get("applied_end_u", 0.0)),
                float(values.get("applied_end_v", 0.0)),
            )
            self._trace_drag_clamped = bool(values.get("drag_was_clamped"))
        else:
            self._trace_applied_current = None
            self._trace_drag_clamped = False
        self.update()

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
        overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        overlay[mask] = np.array(
            [20, 190, 174, 34 if self.compact else 48],
            dtype=np.uint8,
        )
        boundary = self._boundary_mask(mask)
        self._array_shape = mask.shape
        self._mask_plane = mask.copy()
        self._boundary_plane = boundary
        self._boundary_segments = self._contour_segments(mask)
        polylines = self._contour_polylines(self._boundary_segments)
        self._boundary_path_exact = self._contour_path(polylines, smoothing_passes=0)
        self._boundary_path_smooth = self._contour_path(polylines, smoothing_passes=1)
        self._overlay = QtGui.QImage(
            overlay.data,
            overlay.shape[1],
            overlay.shape[0],
            overlay.strides[0],
            QtGui.QImage.Format_RGBA8888,
        ).copy()
        self._loading_bar.hide()
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
        spacing_u = max(1.0e-4, float(self.display_spacing_uv[0]))
        spacing_v = max(1.0e-4, float(self.display_spacing_uv[1]))
        physical_width = source.width() * spacing_u
        physical_height = source.height() * spacing_v
        scale = min(
            available.width() / physical_width,
            available.height() / physical_height,
        )
        width = physical_width * scale
        height = physical_height * scale
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

    @staticmethod
    def _contour_segments(mask: np.ndarray) -> list[QtCore.QLineF]:
        """Return continuous pixel-edge contour segments for a binary plane."""
        source = np.asarray(mask, dtype=bool)
        if source.ndim != 2 or source.size == 0 or not np.any(source):
            return []
        height, width = source.shape
        segments: list[QtCore.QLineF] = []

        rows, columns = np.nonzero(source[:, :-1] != source[:, 1:])
        segments.extend(
            QtCore.QLineF(float(column + 1), float(row), float(column + 1), float(row + 1))
            for row, column in zip(rows, columns)
        )
        rows, columns = np.nonzero(source[:-1, :] != source[1:, :])
        segments.extend(
            QtCore.QLineF(float(column), float(row + 1), float(column + 1), float(row + 1))
            for row, column in zip(rows, columns)
        )
        rows = np.nonzero(source[:, 0])[0]
        segments.extend(
            QtCore.QLineF(0.0, float(row), 0.0, float(row + 1)) for row in rows
        )
        rows = np.nonzero(source[:, -1])[0]
        segments.extend(
            QtCore.QLineF(float(width), float(row), float(width), float(row + 1))
            for row in rows
        )
        columns = np.nonzero(source[0, :])[0]
        segments.extend(
            QtCore.QLineF(float(column), 0.0, float(column + 1), 0.0)
            for column in columns
        )
        columns = np.nonzero(source[-1, :])[0]
        segments.extend(
            QtCore.QLineF(float(column), float(height), float(column + 1), float(height))
            for column in columns
        )
        return segments

    @staticmethod
    def _contour_polylines(
        segments: list[QtCore.QLineF],
    ) -> list[tuple[list[tuple[float, float]], bool]]:
        """Join pixel edges into ordered contours for antialiased display."""
        if not segments:
            return []
        endpoints: list[tuple[tuple[float, float], tuple[float, float]]] = []
        adjacency: dict[tuple[float, float], list[int]] = {}
        for index, segment in enumerate(segments):
            start = (float(segment.x1()), float(segment.y1()))
            end = (float(segment.x2()), float(segment.y2()))
            endpoints.append((start, end))
            adjacency.setdefault(start, []).append(index)
            adjacency.setdefault(end, []).append(index)

        unused = set(range(len(endpoints)))
        output: list[tuple[list[tuple[float, float]], bool]] = []
        while unused:
            first_edge = next(iter(unused))
            first_start, first_end = endpoints[first_edge]
            start = (
                first_start
                if len(adjacency.get(first_start, ())) == 1
                else first_end
                if len(adjacency.get(first_end, ())) == 1
                else first_start
            )
            points = [start]
            current = start
            previous: tuple[float, float] | None = None
            closed = False
            while True:
                candidates = [edge for edge in adjacency.get(current, ()) if edge in unused]
                if not candidates:
                    break
                if previous is not None and len(candidates) > 1:
                    incoming = (current[0] - previous[0], current[1] - previous[1])

                    def continuation_score(edge: int) -> float:
                        edge_start, edge_end = endpoints[edge]
                        other = edge_end if edge_start == current else edge_start
                        outgoing = (other[0] - current[0], other[1] - current[1])
                        return incoming[0] * outgoing[0] + incoming[1] * outgoing[1]

                    edge = max(candidates, key=continuation_score)
                else:
                    edge = candidates[0]
                unused.remove(edge)
                edge_start, edge_end = endpoints[edge]
                other = edge_end if edge_start == current else edge_start
                previous, current = current, other
                points.append(current)
                if current == start:
                    closed = True
                    break
            if len(points) >= 2:
                output.append((points, closed))
        return output

    @staticmethod
    def _remove_collinear_points(
        points: list[tuple[float, float]],
        closed: bool,
    ) -> list[tuple[float, float]]:
        values = list(points[:-1] if closed and points[-1] == points[0] else points)
        if len(values) < 3:
            return values
        output: list[tuple[float, float]] = []
        count = len(values)
        for index, point in enumerate(values):
            if not closed and index in (0, count - 1):
                output.append(point)
                continue
            previous = values[(index - 1) % count]
            following = values[(index + 1) % count]
            first = (point[0] - previous[0], point[1] - previous[1])
            second = (following[0] - point[0], following[1] - point[1])
            cross = first[0] * second[1] - first[1] * second[0]
            dot = first[0] * second[0] + first[1] * second[1]
            if abs(cross) <= 1.0e-8 and dot >= 0.0:
                continue
            output.append(point)
        return output

    @staticmethod
    def _chaikin_points(
        points: list[tuple[float, float]],
        closed: bool,
        passes: int,
    ) -> list[tuple[float, float]]:
        values = list(points)
        for _ in range(max(0, int(passes))):
            if len(values) < 3:
                break
            refined: list[tuple[float, float]] = []
            if not closed:
                refined.append(values[0])
            pair_count = len(values) if closed else len(values) - 1
            for index in range(pair_count):
                start = values[index]
                end = values[(index + 1) % len(values)]
                refined.extend(
                    [
                        (
                            0.75 * start[0] + 0.25 * end[0],
                            0.75 * start[1] + 0.25 * end[1],
                        ),
                        (
                            0.25 * start[0] + 0.75 * end[0],
                            0.25 * start[1] + 0.75 * end[1],
                        ),
                    ]
                )
            if not closed:
                refined.append(values[-1])
            values = refined
        return values

    @classmethod
    def _contour_path(
        cls,
        polylines: list[tuple[list[tuple[float, float]], bool]],
        *,
        smoothing_passes: int,
    ) -> QtGui.QPainterPath:
        path = QtGui.QPainterPath()
        for points, closed in polylines:
            values = cls._remove_collinear_points(points, closed)
            if smoothing_passes > 0:
                values = cls._chaikin_points(values, closed, smoothing_passes)
            if len(values) < 2:
                continue
            path.moveTo(*values[0])
            for point in values[1:]:
                path.lineTo(*point)
            if closed:
                path.closeSubpath()
        return path

    def _nearest_mask_boundary(
        self,
        voxel: tuple[float, float],
        mask: np.ndarray | None = None,
        *,
        maximum_distance_mm: float | None = None,
        maximum_distance_pixels: float | None = None,
    ) -> tuple[float, float] | None:
        boundary = self._boundary_plane if mask is None else self._boundary_mask(mask)
        if boundary.ndim != 2 or boundary.size == 0 or not np.any(boundary):
            return None
        spacing_u = max(1.0e-4, float(self.display_spacing_uv[0]))
        spacing_v = max(1.0e-4, float(self.display_spacing_uv[1]))
        source = self._view_rect if not self._view_rect.isEmpty() else self._full_source_rect()
        target = self._compute_target()
        scale_u = target.width() / max(1.0, source.width())
        scale_v = target.height() / max(1.0, source.height())
        physical_limit = max(
            min(spacing_u, spacing_v),
            float(
                self.boundary_capture_mm
                if maximum_distance_mm is None
                else maximum_distance_mm
            ),
        )
        pixel_limit = max(
            4.0,
            float(
                self.boundary_capture_pixels
                if maximum_distance_pixels is None
                else maximum_distance_pixels
            ),
        )
        column_extent = max(
            1,
            int(
                math.ceil(
                    min(
                        physical_limit / spacing_u,
                        pixel_limit / max(1.0e-4, scale_u),
                    )
                )
            ),
        )
        row_extent = max(
            1,
            int(
                math.ceil(
                    min(
                        physical_limit / spacing_v,
                        pixel_limit / max(1.0e-4, scale_v),
                    )
                )
            ),
        )
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
        physical_distance_sq = (
            ((candidate_u - float(voxel[0])) * spacing_u) ** 2
            + ((candidate_v - float(voxel[1])) * spacing_v) ** 2
        )
        screen_distance_sq = (
            ((candidate_u - float(voxel[0])) * scale_u) ** 2
            + ((candidate_v - float(voxel[1])) * scale_v) ** 2
        )
        score = screen_distance_sq / max(1.0, pixel_limit * pixel_limit)
        index = int(np.argmin(score))
        if (
            float(physical_distance_sq[index]) > physical_limit * physical_limit
            or float(screen_distance_sq[index]) > pixel_limit * pixel_limit
        ):
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

    def _paint_phase_badge(self, painter: QtGui.QPainter) -> None:
        if self.compact or self.phase not in ("running", "warning"):
            return
        text = "REFINING LOCAL BOUNDARY" if self.phase == "running" else self.phase_text.upper()
        metrics = painter.fontMetrics()
        text = metrics.elidedText(text, QtCore.Qt.ElideRight, max(100, self.width() - 64))
        width = min(self.width() - 36, metrics.horizontalAdvance(text) + 28)
        badge = QtCore.QRectF(24.0, 24.0, float(max(120, width)), 30.0)
        color = QtGui.QColor(23, 107, 117, 232)
        if self.phase == "warning":
            color = QtGui.QColor(158, 91, 26, 232)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(color)
        painter.drawRoundedRect(badge, 5.0, 5.0)
        painter.setPen(QtGui.QColor("#ffffff"))
        painter.drawText(badge, QtCore.Qt.AlignCenter, text)

    def _paint_linked_crosshair(
        self,
        painter: QtGui.QPainter,
        target: QtCore.QRectF,
    ) -> None:
        if self.linked_crosshair_uv is None or self._image is None:
            return
        point = self._screen_at_voxel(self.linked_crosshair_uv)
        if point is None or not target.contains(point):
            return
        color = QtGui.QColor(255, 196, 74, 185 if self.compact else 135)
        pen = QtGui.QPen(color, 1.0, QtCore.Qt.DashLine)
        pen.setCosmetic(True)
        painter.setPen(pen)
        gap = 7.0
        painter.drawLine(QtCore.QPointF(target.left(), point.y()), QtCore.QPointF(point.x() - gap, point.y()))
        painter.drawLine(QtCore.QPointF(point.x() + gap, point.y()), QtCore.QPointF(target.right(), point.y()))
        painter.drawLine(QtCore.QPointF(point.x(), target.top()), QtCore.QPointF(point.x(), point.y() - gap))
        painter.drawLine(QtCore.QPointF(point.x(), point.y() + gap), QtCore.QPointF(point.x(), target.bottom()))
        painter.setBrush(QtGui.QColor(255, 196, 74, 55))
        painter.drawEllipse(point, 3.5, 3.5)

    def _paint_view_badge(self, painter: QtGui.QPainter, target: QtCore.QRectF) -> None:
        if not self.compact:
            return
        title = "{}  {}".format(self.view_orientation.capitalize(), self.view_slice_index + 1)
        metrics = painter.fontMetrics()
        badge = QtCore.QRectF(
            target.left() + 12.0,
            target.top() + 12.0,
            float(metrics.horizontalAdvance(title) + 22),
            28.0,
        )
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(12, 23, 32, 215))
        painter.drawRoundedRect(badge, 5.0, 5.0)
        painter.setPen(QtGui.QColor(229, 238, 244, 235))
        painter.drawText(badge, QtCore.Qt.AlignCenter, title)

    @staticmethod
    def _paint_active_control_point(
        painter: QtGui.QPainter,
        point: QtCore.QPointF,
    ) -> None:
        """Draw one stable handle for the contour point actually being moved."""
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(255, 196, 74, 52))
        painter.drawEllipse(point, 9.0, 9.0)
        painter.setPen(QtGui.QPen(QtGui.QColor(4, 16, 21, 235), 2.6))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawEllipse(point, 6.2, 6.2)
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 238), 1.1))
        painter.setBrush(QtGui.QColor(255, 196, 74, 245))
        painter.drawEllipse(point, 4.4, 4.4)

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
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
        show_exact_contour = max(scale_x, scale_y) >= 4.0
        if (
            (self._boundary_dragging or self._contour_tracing)
            and (
                not self._interaction_origin_path_exact.isEmpty()
                or self._interaction_origin_segments
            )
        ):
            painter.save()
            painter.setClipRect(target)
            painter.translate(target.left(), target.top())
            painter.scale(scale_x, scale_y)
            painter.translate(-source.left(), -source.top())
            origin_pen = QtGui.QPen(
                QtGui.QColor(225, 232, 238, 82),
                1.15,
                QtCore.Qt.DashLine,
                QtCore.Qt.RoundCap,
                QtCore.Qt.RoundJoin,
            )
            origin_pen.setCosmetic(True)
            painter.setPen(origin_pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            origin_path = (
                self._interaction_origin_path_exact
                if show_exact_contour
                else self._interaction_origin_path_smooth
            )
            if not origin_path.isEmpty():
                painter.drawPath(origin_path)
            else:
                painter.drawLines(self._interaction_origin_segments)
            painter.restore()
        if self._boundary_segments or not self._boundary_path_exact.isEmpty():
            painter.save()
            painter.setClipRect(target)
            painter.translate(target.left(), target.top())
            painter.scale(scale_x, scale_y)
            painter.translate(-source.left(), -source.top())
            contour_shadow = QtGui.QPen(
                QtGui.QColor(4, 15, 20, 205),
                3.0 if not self.compact else 2.5,
                QtCore.Qt.SolidLine,
                QtCore.Qt.RoundCap,
                QtCore.Qt.RoundJoin,
            )
            contour_shadow.setCosmetic(True)
            painter.setPen(contour_shadow)
            painter.setBrush(QtCore.Qt.NoBrush)
            contour_path = (
                self._boundary_path_exact
                if show_exact_contour
                else self._boundary_path_smooth
            )
            if not contour_path.isEmpty():
                painter.drawPath(contour_path)
            else:
                painter.drawLines(self._boundary_segments)
            contour_pen = QtGui.QPen(
                QtGui.QColor(93, 239, 220, 248 if not self.compact else 225),
                1.4 if not self.compact else 1.2,
                QtCore.Qt.SolidLine,
                QtCore.Qt.RoundCap,
                QtCore.Qt.RoundJoin,
            )
            contour_pen.setCosmetic(True)
            painter.setPen(contour_pen)
            if not contour_path.isEmpty():
                painter.drawPath(contour_path)
            else:
                painter.drawLines(self._boundary_segments)
            painter.restore()
        if (
            not self._boundary_dragging
            and not self._contour_tracing
            and (self.compact or self._cursor_pos is None)
        ):
            self._paint_linked_crosshair(painter, target)
        if self._boundary_dragging and self._boundary_start is not None:
            requested = self._screen_at_voxel(self._boundary_current or self._boundary_start)
            applied = self._screen_at_voxel(
                self._boundary_applied_current or self._boundary_current or self._boundary_start
            )
            if requested is not None and applied is not None:
                self._paint_active_control_point(painter, applied)
                if self._boundary_drag_clamped and (requested - applied).manhattanLength() > 1.0:
                    limited_pen = QtGui.QPen(QtGui.QColor(255, 137, 72, 210), 1.2, QtCore.Qt.DotLine)
                    limited_pen.setCosmetic(True)
                    painter.setPen(limited_pen)
                    painter.drawLine(applied, requested)
                    painter.setBrush(QtCore.Qt.NoBrush)
                    painter.drawEllipse(requested, 6.5, 6.5)
        elif self._contour_tracing and self._trace_current is not None:
            requested = self._screen_at_voxel(self._trace_current)
            applied = self._screen_at_voxel(
                self._trace_applied_current or self._trace_current
            )
            if requested is not None and applied is not None:
                self._paint_active_control_point(painter, applied)
                if self._trace_drag_clamped and (requested - applied).manhattanLength() > 1.0:
                    limited_pen = QtGui.QPen(
                        QtGui.QColor(255, 137, 72, 190),
                        1.2,
                        QtCore.Qt.DotLine,
                    )
                    limited_pen.setCosmetic(True)
                    painter.setPen(limited_pen)
                    painter.setBrush(QtCore.Qt.NoBrush)
                    painter.drawLine(applied, requested)
                    painter.drawEllipse(requested, 6.5, 6.5)
        elif self._stroke_points:
            color = self._mode_color(self._active_brush_mode or "pending", 110)
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(color)
            for point in self._stroke_points:
                screen = self._screen_at_voxel(point)
                if screen is not None:
                    painter.drawEllipse(screen, radius_x, radius_y)
        if (
            not self.read_only
            and self._cursor_pos is not None
            and target.contains(self._cursor_pos)
        ):
            voxel = self._voxel_at(self._cursor_pos)
            cursor_mode = self.interaction_mode
            if cursor_mode in ("ffd", "trace"):
                boundary_point = self._hover_boundary_point
                if boundary_point is not None:
                    screen = self._screen_at_voxel(boundary_point)
                    if screen is not None:
                        self._paint_active_control_point(painter, screen)
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
                if (
                    self._boundary_miss_pos is not None
                    and time.monotonic() < self._boundary_miss_until
                ):
                    painter.setPen(QtGui.QPen(QtGui.QColor(255, 105, 97, 235), 2.0))
                    painter.setBrush(QtCore.Qt.NoBrush)
                    painter.drawEllipse(self._boundary_miss_pos, 10.0, 10.0)
                self._paint_phase_badge(painter)
                return
            if cursor_mode == "auto" and voxel is not None:
                cursor_mode = self._automatic_mode(voxel)
            color = self._mode_color(cursor_mode, 225)
            painter.setPen(QtGui.QPen(color, 1.5))
            painter.setBrush(self._mode_color(cursor_mode, 28))
            painter.drawEllipse(self._cursor_pos, radius_x, radius_y)
        self._paint_view_badge(painter, target)
        self._paint_phase_badge(painter)

    def enterEvent(self, _event: QtCore.QEvent) -> None:
        self.setCursor(
            QtCore.Qt.PointingHandCursor if self.read_only else QtCore.Qt.CrossCursor
        )

    def leaveEvent(self, _event: QtCore.QEvent) -> None:
        self._cursor_pos = None
        self._hover_boundary_point = None
        self.pointer_changed.emit(None)
        self.update()

    def _update_hover_boundary(self, voxel: tuple[float, float] | None) -> None:
        if self.read_only or self._boundary_dragging or self._contour_tracing:
            self._hover_boundary_point = None
            return
        if voxel is None or self.interaction_mode not in ("ffd", "trace"):
            self._hover_boundary_point = None
            self.setCursor(QtCore.Qt.CrossCursor)
            return
        if self.interaction_mode == "trace":
            self._hover_boundary_point = self._nearest_mask_boundary(
                voxel,
                maximum_distance_mm=self.trace_capture_mm,
                maximum_distance_pixels=self.trace_capture_pixels,
            )
        else:
            self._hover_boundary_point = self._nearest_mask_boundary(voxel)
        self.setCursor(
            QtCore.Qt.OpenHandCursor
            if self._hover_boundary_point is not None
            else QtCore.Qt.CrossCursor
        )

    def _capture_interaction_origin(self) -> None:
        self._interaction_origin_segments = [
            QtCore.QLineF(segment) for segment in self._boundary_segments
        ]
        self._interaction_origin_path_exact = QtGui.QPainterPath(
            self._boundary_path_exact
        )
        self._interaction_origin_path_smooth = QtGui.QPainterPath(
            self._boundary_path_smooth
        )

    def _clear_interaction_origin(self) -> None:
        self._interaction_origin_segments = []
        self._interaction_origin_path_exact = QtGui.QPainterPath()
        self._interaction_origin_path_smooth = QtGui.QPainterPath()

    def cancel_active_interaction(self) -> bool:
        if self._boundary_dragging:
            self._boundary_dragging = False
            self._boundary_start = None
            self._boundary_current = None
            self._boundary_applied_current = None
            self._boundary_drag_clamped = False
            self._trace_path = []
            self._clear_interaction_origin()
            self.interaction_active_changed.emit(False)
            self.interaction_cancel_requested.emit("ffd")
            self.setCursor(QtCore.Qt.CrossCursor)
            self.update()
            return True
        if self._contour_tracing:
            self._contour_tracing = False
            self._trace_current = None
            self._trace_applied_current = None
            self._trace_drag_clamped = False
            self._trace_path = []
            self._clear_interaction_origin()
            self.interaction_active_changed.emit(False)
            self.interaction_cancel_requested.emit("trace")
            self.setCursor(QtCore.Qt.CrossCursor)
            self.update()
            return True
        return False

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key_Escape and self.cancel_active_interaction():
            event.accept()
            return
        super().keyPressEvent(event)

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
        if self._contour_tracing:
            if voxel is not None:
                self._trace_current = voxel
                if (
                    not self._trace_path
                    or math.hypot(
                        voxel[0] - self._trace_path[-1][0],
                        voxel[1] - self._trace_path[-1][1],
                    )
                    >= 0.5
                ):
                    self._trace_path.append(voxel)
                    if len(self._trace_path) > 256:
                        self._trace_path = self._trace_path[:1] + self._trace_path[2::2]
                self.contour_trace_requested.emit(voxel)
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
        self._update_hover_boundary(voxel)
        self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.RightButton and self.cancel_active_interaction():
            event.accept()
            return
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
        if self.read_only:
            self.navigation_requested.emit(voxel)
            self.update()
            return
        requested_mode = str(self.interaction_mode or "ffd")
        if requested_mode == "ffd":
            boundary_point = self._nearest_mask_boundary(voxel)
            if boundary_point is None:
                self._boundary_miss_pos = QtCore.QPointF(event.position())
                self._boundary_miss_until = time.monotonic() + 0.8
                QtCore.QTimer.singleShot(850, self.update)
                self.gesture_mode_changed.emit("boundary_miss")
                self.update()
                return
            self._boundary_miss_pos = None
            self._boundary_dragging = True
            self._boundary_start = boundary_point
            self._boundary_current = boundary_point
            self._boundary_applied_current = boundary_point
            self._boundary_drag_clamped = False
            self._capture_interaction_origin()
            self.boundary_drag_started.emit(boundary_point)
            self.interaction_active_changed.emit(True)
            self.gesture_mode_changed.emit("boundary_captured")
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            self.update()
            return
        if requested_mode == "trace":
            boundary_point = self._nearest_mask_boundary(
                voxel,
                maximum_distance_mm=self.trace_capture_mm,
                maximum_distance_pixels=self.trace_capture_pixels,
            )
            if boundary_point is None:
                self._boundary_miss_pos = QtCore.QPointF(event.position())
                self._boundary_miss_until = time.monotonic() + 0.8
                QtCore.QTimer.singleShot(850, self.update)
                self.gesture_mode_changed.emit("trace_miss")
                self.update()
                return
            self._boundary_miss_pos = None
            self._contour_tracing = True
            self._trace_current = voxel
            self._trace_applied_current = boundary_point
            self._trace_drag_clamped = False
            self._trace_path = [boundary_point, voxel]
            self._capture_interaction_origin()
            self.contour_trace_started.emit(voxel)
            self.interaction_active_changed.emit(True)
            self.gesture_mode_changed.emit("trace_started")
            self.setCursor(QtCore.Qt.ClosedHandCursor)
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

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.read_only and event.button() == QtCore.Qt.LeftButton:
            voxel = self._voxel_at(event.position())
            if voxel is not None:
                self.navigation_requested.emit(voxel)
                self.promote_requested.emit()
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

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
            self.interaction_active_changed.emit(False)
            self._boundary_start = None
            self._boundary_current = None
            self._boundary_applied_current = None
            self._boundary_drag_clamped = False
            self._clear_interaction_origin()
            self.setCursor(QtCore.Qt.CrossCursor)
            self.update()
            return
        if event.button() == QtCore.Qt.LeftButton and self._contour_tracing:
            voxel = self._voxel_at(event.position()) or self._trace_current
            if voxel is not None:
                self._trace_current = voxel
                self.contour_trace_requested.emit(voxel)
            self._contour_tracing = False
            self.contour_trace_finished.emit()
            self.interaction_active_changed.emit(False)
            self._trace_current = None
            self._trace_applied_current = None
            self._trace_drag_clamped = False
            self._trace_path = []
            self._clear_interaction_origin()
            self.setCursor(QtCore.Qt.CrossCursor)
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
        if self._boundary_dragging or self._contour_tracing:
            event.accept()
            return
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
        self._boundary_context: dict[str, Any] | None = None
        self._trace_context: dict[str, Any] | None = None
        self._interaction_active = False
        self.crosshair_xyz: tuple[float, float, float] | None = None
        self._shortcuts: list[QtGui.QShortcut] = []
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
        self.crosshair_timer = QtCore.QTimer(self)
        self.crosshair_timer.setSingleShot(True)
        self.crosshair_timer.setInterval(100)
        self.crosshair_timer.timeout.connect(self._submit_crosshair)
        self.setWindowTitle("IGAC · Local Contour Refinement")
        self.resize(1460, 920)
        self.setMinimumSize(1180, 720)
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
            self.worker.orthogonal_views_ready.connect(self._on_orthogonal_views)
            self.worker.metrics_ready.connect(self._on_metrics)
            self.worker.phase_changed.connect(self._set_phase)
            self.worker.undo_available.connect(self._set_undo_available)
            self.worker.boundary_preview.connect(self.canvas.set_boundary_preview)
            self.worker.contour_trace_preview.connect(
                self.canvas.set_contour_trace_preview
            )
            self.worker.local_refinement_available.connect(
                self._set_local_refinement_available
            )
            self.worker.failed.connect(self._on_failed)
            self.worker.terminal.connect(self._on_terminal)
            self.worker.finished.connect(self._on_worker_finished)
            self.worker.start()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        central.setObjectName("igacRoot")
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(18, 14, 18, 16)
        outer.setSpacing(12)
        self.pointer_label = QtWidgets.QLabel("Move over the image to inspect a voxel")
        self.pointer_label.setObjectName("hint")
        self.statusBar().setSizeGripEnabled(False)
        self.statusBar().addWidget(self.pointer_label, 1)

        header = QtWidgets.QHBoxLayout()
        title_column = QtWidgets.QVBoxLayout()
        title_column.setSpacing(2)
        title = QtWidgets.QLabel("IGAC contour refinement")
        title.setObjectName("title")
        mask_name = str(self.request.get("mask_name") or "Selected Mask")
        subtitle = QtWidgets.QLabel("{} · local 3D workspace".format(mask_name))
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
        body.setSpacing(12)

        workspace = QtWidgets.QFrame()
        workspace.setObjectName("workspace")
        workspace_layout = QtWidgets.QVBoxLayout(workspace)
        workspace_layout.setContentsMargins(10, 10, 10, 10)
        workspace_layout.setSpacing(8)
        view_row = QtWidgets.QHBoxLayout()
        view_row.setSpacing(8)
        orientation_widget, self.orientation_group = make_segmented(
            [("Axial", "axial"), ("Coronal", "coronal"), ("Sagittal", "sagittal")],
            "axial",
        )
        orientation_widget.setObjectName("darkSegments")
        orientation_widget.setMaximumWidth(390)
        view_row.addWidget(orientation_widget)
        view_row.addStretch(1)
        self.iteration_label = QtWidgets.QLabel("Iteration 0")
        self.iteration_label.setObjectName("viewportMeta")
        self.speed_label = QtWidgets.QLabel("— it/s")
        self.speed_label.setObjectName("viewportMeta")
        view_row.addWidget(self.iteration_label)
        view_row.addSpacing(6)
        view_row.addWidget(self.speed_label)
        view_row.addSpacing(6)
        self.volume_label = QtWidgets.QLabel("Mask volume —")
        self.volume_label.setObjectName("viewportMeta")
        view_row.addWidget(self.volume_label)
        workspace_layout.addLayout(view_row)

        view_grid = QtWidgets.QHBoxLayout()
        view_grid.setSpacing(8)
        self.canvas = ImageCanvas()
        self.canvas.setObjectName("primaryCanvas")
        view_grid.addWidget(self.canvas, 1)
        context_column = QtWidgets.QVBoxLayout()
        context_column.setSpacing(8)
        self.context_canvases: list[ImageCanvas] = []
        for orientation in ("coronal", "sagittal"):
            context = ImageCanvas(compact=True, read_only=True)
            context.setObjectName("contextCanvas")
            context.setMaximumWidth(290)
            context.set_view_context(orientation, 0, None)
            context.setToolTip("Click to position the linked cursor; double-click to edit this plane")
            self.context_canvases.append(context)
            context_column.addWidget(context, 1)
        view_grid.addLayout(context_column, 0)
        workspace_layout.addLayout(view_grid, 1)

        slice_row = QtWidgets.QHBoxLayout()
        slice_row.setSpacing(8)
        slice_title = QtWidgets.QLabel("Slice")
        slice_title.setObjectName("viewportLabel")
        self.slice_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slice_spin = QtWidgets.QSpinBox()
        self.slice_spin.setFixedWidth(76)
        self.fit_mask_button = QtWidgets.QPushButton("Fit Mask")
        self.fit_mask_button.setObjectName("viewportButton")
        self.fit_mask_button.setToolTip("Center and enlarge the selected Mask with local context")
        self.fit_mask_button.setEnabled(False)
        self.fit_image_button = QtWidgets.QPushButton("Fit Image")
        self.fit_image_button.setObjectName("viewportButton")
        self.fit_image_button.setToolTip("Show the complete image slice")
        self.zoom_out_button = QtWidgets.QToolButton()
        self.zoom_out_button.setObjectName("viewportToolButton")
        self.zoom_out_button.setText("−")
        self.zoom_out_button.setToolTip("Zoom out")
        self.zoom_out_button.setFixedSize(32, 30)
        self.zoom_in_button = QtWidgets.QToolButton()
        self.zoom_in_button.setObjectName("viewportToolButton")
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
        workspace_layout.addLayout(slice_row)
        body.addWidget(workspace, 1)

        scroll = QtWidgets.QScrollArea()
        scroll.setObjectName("inspectorScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        sidebar = QtWidgets.QWidget()
        sidebar.setObjectName("inspectorBody")
        side = QtWidgets.QVBoxLayout(sidebar)
        side.setContentsMargins(18, 18, 14, 14)
        side.setSpacing(10)

        edit_heading = QtWidgets.QLabel("CORRECTION")
        edit_heading.setObjectName("inspectorHeading")
        side.addWidget(edit_heading)
        tool_widget, self.tool_group = make_segmented(
            [("Drag contour", "ffd"), ("Guide edge", "trace")],
            "ffd",
        )
        tool_widget.setObjectName("lightSegments")
        side.addWidget(tool_widget)
        self.tool_hint = QtWidgets.QLabel("Local contour deformation")
        self.tool_hint.setObjectName("toolSummary")
        self.tool_hint.setWordWrap(True)
        side.addWidget(self.tool_hint)

        ffd_defaults = dict((self.request.get("igac") or {}).get("ffd") or {})
        self.pull_controls = QtWidgets.QWidget()
        pull_controls_layout = QtWidgets.QVBoxLayout(self.pull_controls)
        pull_controls_layout.setContentsMargins(0, 2, 0, 0)
        pull_controls_layout.setSpacing(5)
        pull_row = QtWidgets.QHBoxLayout()
        pull_title = QtWidgets.QLabel("Influence radius")
        pull_title.setObjectName("fieldLabel")
        pull_row.addWidget(pull_title)
        pull_row.addStretch(1)
        pull_default = max(
            4,
            min(40, int(round(float(ffd_defaults.get("influence_radius_mm", 8.0))))),
        )
        self.pull_value = QtWidgets.QLabel("{:.1f} mm".format(float(pull_default)))
        self.pull_value.setObjectName("fieldValue")
        pull_row.addWidget(self.pull_value)
        pull_controls_layout.addLayout(pull_row)
        self.pull_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.pull_slider.setRange(4, 40)
        self.pull_slider.setValue(pull_default)
        self.pull_slider.setToolTip(
            "Physical radius of the local 3D B-spline deformation around the selected contour point"
        )
        pull_controls_layout.addWidget(self.pull_slider)
        side.addWidget(self.pull_controls)

        correction_actions = QtWidgets.QHBoxLayout()
        self.undo_button = QtWidgets.QPushButton("Undo")
        self.undo_button.setObjectName("quietAction")
        self.undo_button.setEnabled(False)
        self.undo_button.setToolTip("Restore the Mask from before the last action")
        self.reset_button = QtWidgets.QPushButton("Reset Mask")
        self.reset_button.setObjectName("quietAction")
        self.reset_button.setToolTip("Restore the Mask exactly as it was when IGAC opened")
        correction_actions.addWidget(self.undo_button)
        correction_actions.addWidget(self.reset_button)
        side.addLayout(correction_actions)
        side.addWidget(self._separator())

        display_heading = QtWidgets.QLabel("DISPLAY")
        display_heading.setObjectName("inspectorHeading")
        side.addWidget(display_heading)
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
        side.addLayout(preset_row)
        display_grid = QtWidgets.QGridLayout()
        display_grid.setHorizontalSpacing(8)
        display_grid.setVerticalSpacing(4)
        width_label = QtWidgets.QLabel("Width")
        width_label.setObjectName("fieldLabel")
        display_grid.addWidget(width_label, 0, 0)
        self.display_width = QtWidgets.QDoubleSpinBox()
        self.display_width.setRange(0.1, 20000000.0)
        self.display_width.setDecimals(1)
        display_grid.addWidget(self.display_width, 1, 0)
        level_label = QtWidgets.QLabel("Level")
        level_label.setObjectName("fieldLabel")
        display_grid.addWidget(level_label, 0, 1)
        self.display_level = QtWidgets.QDoubleSpinBox()
        self.display_level.setRange(-10000000.0, 10000000.0)
        self.display_level.setDecimals(1)
        display_grid.addWidget(self.display_level, 1, 1)
        side.addLayout(display_grid)
        side.addWidget(self._separator())

        refine_heading = QtWidgets.QLabel("LOCAL FITTING")
        refine_heading.setObjectName("inspectorHeading")
        side.addWidget(refine_heading)
        self.run_button = QtWidgets.QPushButton("Refine local edit")
        self.run_button.setObjectName("primary")
        self.run_button.setEnabled(False)
        self.run_button.setToolTip(
            "Repeat the bounded LGDF refinement around the most recent edit"
        )
        side.addWidget(self.run_button)
        default = dict((self.request.get("igac") or {}).get("parameters") or {})
        limit_row = QtWidgets.QHBoxLayout()
        self.limit_movement = QtWidgets.QCheckBox("Limit auto movement")
        maximum_default = float(default.get("max_displacement_mm", 10.0))
        self.limit_movement.setChecked(maximum_default > 0.0)
        self.limit_movement.setToolTip(
            "Keep automatic evolution near the Mask captured when IGAC opened"
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
        side.addLayout(limit_row)

        self.refinement_toggle = QtWidgets.QToolButton()
        self.refinement_toggle.setObjectName("disclosure")
        self.refinement_toggle.setText("Refinement settings")
        self.refinement_toggle.setCheckable(True)
        self.refinement_toggle.setChecked(False)
        self.refinement_toggle.setArrowType(QtCore.Qt.RightArrow)
        self.refinement_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        side.addWidget(self.refinement_toggle)
        self.refinement_panel = QtWidgets.QWidget()
        refinement_layout = QtWidgets.QVBoxLayout(self.refinement_panel)
        refinement_layout.setContentsMargins(8, 2, 2, 2)
        refinement_layout.setSpacing(8)
        self.range_row = ParameterRow("Range", 0.5, 8.0, 0.1, float(default.get("range_mm", 3.0)), 1)
        self.range_row.setToolTip("Physical Gaussian scale used by the local intensity model")
        self.smooth_row = ParameterRow("Smooth", 0.0, 100.0, 1.0, float(default.get("smooth", 50.0)), 0)
        self.smooth_row.setToolTip("Strength of contour regularity")
        self.grow_row = ParameterRow("Grow", -0.50, 0.50, 0.01, float(default.get("grow", 0.0)), 2)
        self.grow_row.setToolTip("Positive values favour a slightly larger foreground region")
        refinement_layout.addWidget(self.range_row)
        refinement_layout.addWidget(self.smooth_row)
        refinement_layout.addWidget(self.grow_row)
        self.refinement_panel.setVisible(False)
        side.addWidget(self.refinement_panel)
        side.addStretch(1)
        self.roi_label = QtWidgets.QLabel("Preparing 3D workspace")
        self.roi_label.setObjectName("inspectorFootnote")
        self.roi_label.setWordWrap(True)
        side.addWidget(self.roi_label)
        scroll.setWidget(sidebar)

        result_group = QtWidgets.QFrame()
        result_group.setObjectName("inspectorFooter")
        result_layout = QtWidgets.QVBoxLayout(result_group)
        result_layout.setContentsMargins(16, 12, 16, 14)
        result_layout.setSpacing(8)
        destination_label = QtWidgets.QLabel("RESULT IN MIMICS")
        destination_label.setObjectName("inspectorHeading")
        result_layout.addWidget(destination_label)
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
        right = QtWidgets.QWidget()
        right.setObjectName("inspector")
        right.setFixedWidth(330)
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        right_layout.addWidget(scroll, 1)
        right_layout.addWidget(result_group, 0)
        body.addWidget(right)
        outer.addLayout(body, 1)

    def _separator(self) -> QtWidgets.QFrame:
        line = QtWidgets.QFrame()
        line.setObjectName("inspectorSeparator")
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFixedHeight(1)
        return line

    def _connect_ui(self) -> None:
        self.orientation_group.buttonClicked.connect(
            lambda button: self._set_orientation(str(button.property("segmentValue")))
        )
        self.tool_group.buttonClicked.connect(self._tool_changed)
        self.slice_slider.valueChanged.connect(self._slice_changed)
        self.slice_spin.valueChanged.connect(self.slice_slider.setValue)
        self.fit_mask_button.clicked.connect(self._fit_mask_view)
        self.fit_image_button.clicked.connect(self.canvas.fit_image)
        self.zoom_out_button.clicked.connect(lambda: self.canvas.zoom(0.8))
        self.zoom_in_button.clicked.connect(lambda: self.canvas.zoom(1.25))
        self.canvas.slice_wheel.connect(lambda delta: self.slice_slider.setValue(self.slice_slider.value() + delta))
        self.canvas.pointer_changed.connect(self._pointer_changed)
        self.canvas.boundary_drag_started.connect(self._boundary_drag_started)
        self.canvas.boundary_drag_requested.connect(self._boundary_drag_requested)
        self.canvas.boundary_drag_finished.connect(self._boundary_drag_finished)
        self.canvas.contour_trace_started.connect(self._contour_trace_started)
        self.canvas.contour_trace_requested.connect(self._contour_trace_requested)
        self.canvas.contour_trace_finished.connect(self._contour_trace_finished)
        self.canvas.interaction_active_changed.connect(self._set_interaction_active)
        self.canvas.interaction_cancel_requested.connect(
            self._cancel_worker_interaction
        )
        self.canvas.gesture_mode_changed.connect(self._gesture_mode_changed)
        for context in self.context_canvases:
            context.navigation_requested.connect(
                lambda voxel, canvas=context: self._navigate_from_context(canvas, voxel)
            )
            context.promote_requested.connect(
                lambda canvas=context: self._promote_context(canvas)
            )
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
        self.refinement_toggle.toggled.connect(self._toggle_refinement_panel)
        self.apply_button.clicked.connect(self._apply)
        self.cancel_button.clicked.connect(self._cancel)
        self._install_shortcuts()

    def _add_shortcut(self, sequence: str, callback: Any) -> None:
        shortcut = QtGui.QShortcut(QtGui.QKeySequence(sequence), self)
        shortcut.activated.connect(lambda callback=callback: self._run_shortcut(callback))
        self._shortcuts.append(shortcut)

    def _run_shortcut(self, callback: Any) -> None:
        if self._interaction_active:
            return
        focus = QtWidgets.QApplication.focusWidget()
        if isinstance(
            focus,
            (QtWidgets.QLineEdit, QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox),
        ):
            return
        callback()

    def _set_interaction_active(self, active: bool) -> None:
        self._interaction_active = bool(active)
        enabled = not self._interaction_active
        for button in self.orientation_group.buttons():
            button.setEnabled(enabled)
        for widget in (
            self.slice_slider,
            self.slice_spin,
            self.fit_mask_button,
            self.fit_image_button,
            self.zoom_out_button,
            self.zoom_in_button,
        ):
            widget.setEnabled(enabled)
        for context in self.context_canvases:
            context.setEnabled(enabled)

    def _toggle_refinement_panel(self, checked: bool) -> None:
        self.refinement_toggle.setArrowType(
            QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow
        )
        self.refinement_panel.setVisible(bool(checked))

    def _context_orientations(self) -> list[str]:
        return [value for value in ORIENTATIONS if value != self.orientation]

    def _update_canvas_contexts(self) -> None:
        if self.crosshair_xyz is None:
            crosshair = None
        else:
            crosshair = self.crosshair_xyz
        self.canvas.set_view_context(
            self.orientation,
            int(self.slice_slider.value()),
            crosshair,
        )
        for canvas, orientation in zip(self.context_canvases, self._context_orientations()):
            canvas.display_spacing_uv = self._display_spacing_for_orientation(orientation)
            depth_axis = {"axial": 2, "coronal": 1, "sagittal": 0}[orientation]
            index = (
                int(round(crosshair[depth_axis]))
                if crosshair is not None
                else orientation_extent(self.shape_xyz, orientation)[2] // 2
            )
            canvas.set_view_context(orientation, index, crosshair)
            focus_rect = self._focus_rect_for_orientation(orientation, margin_mm=18.0)
            if focus_rect is not None:
                canvas.fit_source_rect(focus_rect)

    def _set_crosshair_xyz(
        self,
        values: tuple[float, float, float] | list[float],
        *,
        submit: bool,
        immediate: bool = False,
    ) -> None:
        if len(values) != 3:
            return
        self.crosshair_xyz = tuple(
            max(0.0, min(float(self.shape_xyz[axis] - 1), float(values[axis])))
            for axis in range(3)
        )
        depth_axis = {"axial": 2, "coronal": 1, "sagittal": 0}[self.orientation]
        active_index = int(round(self.crosshair_xyz[depth_axis]))
        with QtCore.QSignalBlocker(self.slice_slider), QtCore.QSignalBlocker(self.slice_spin):
            self.slice_slider.setValue(active_index)
            self.slice_spin.setValue(active_index)
        self._update_canvas_contexts()
        if submit and self.worker is not None:
            if immediate:
                self.crosshair_timer.stop()
                self.worker.submit("crosshair", self.crosshair_xyz)
            else:
                self.crosshair_timer.start()

    def _submit_crosshair(self) -> None:
        if self.worker is not None and self.crosshair_xyz is not None:
            self.worker.submit("crosshair", self.crosshair_xyz)

    def _navigate_from_context(
        self,
        canvas: ImageCanvas,
        voxel: tuple[float, float],
    ) -> None:
        if self._interaction_active:
            return
        xyz = display_to_xyz(
            canvas.view_orientation,
            canvas.view_slice_index,
            float(voxel[0]),
            float(voxel[1]),
        )
        self._set_crosshair_xyz(xyz, submit=True, immediate=True)

    def _promote_context(self, canvas: ImageCanvas) -> None:
        if self._interaction_active:
            return
        self._set_orientation(canvas.view_orientation)

    def _select_tool(self, value: str) -> None:
        for button in self.tool_group.buttons():
            if str(button.property("segmentValue")) == str(value):
                button.setChecked(True)
                self._tool_changed(button)
                return

    def _install_shortcuts(self) -> None:
        for key, mode in (("1", "ffd"), ("2", "trace")):
            self._add_shortcut(key, lambda mode=mode: self._select_tool(mode))
        self._add_shortcut("Ctrl+Z", self._undo_last)
        self._add_shortcut("Up", lambda: self.slice_slider.setValue(self.slice_slider.value() + 1))
        self._add_shortcut("Down", lambda: self.slice_slider.setValue(self.slice_slider.value() - 1))
        self._add_shortcut("[", lambda: self.pull_slider.setValue(self.pull_slider.value() - 1))
        self._add_shortcut("]", lambda: self.pull_slider.setValue(self.pull_slider.value() + 1))
        self._add_shortcut("Space", self._toggle_running)
        self._add_shortcut("F", self._fit_mask_view)
        self._add_shortcut("I", self.canvas.fit_image)
        escape = QtGui.QShortcut(QtGui.QKeySequence("Escape"), self)
        escape.activated.connect(self.canvas.cancel_active_interaction)
        self._shortcuts.append(escape)

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
        self._set_crosshair_xyz((float(x), float(y), float(z)), submit=True)
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
        for button in self.orientation_group.buttons():
            if str(button.property("segmentValue")) == self.orientation:
                button.setChecked(True)
                break
        _width, _height, depth = orientation_extent(self.shape_xyz, self.orientation)
        if self.crosshair_xyz is not None:
            depth_axis = {"axial": 2, "coronal": 1, "sagittal": 0}[self.orientation]
            current = max(0, min(depth - 1, int(round(self.crosshair_xyz[depth_axis]))))
        elif self.focus_bbox_xyz is not None:
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
        self._update_canvas_contexts()
        if self.focus_bbox_xyz is not None:
            self._fit_mask_view()
        if self.worker is not None:
            self.worker.submit("view", (self.orientation, current))

    def _fit_mask_view(self) -> None:
        focus_rect = self._focus_rect_for_orientation(self.orientation, margin_mm=12.0)
        if focus_rect is None:
            self.canvas.fit_image()
            return
        self.canvas.fit_source_rect(focus_rect)

    def _focus_rect_for_orientation(
        self,
        orientation: str,
        *,
        margin_mm: float,
    ) -> QtCore.QRectF | None:
        if self.focus_bbox_xyz is None:
            return None
        if orientation == "axial":
            horizontal_axis, vertical_axis = 0, 1
        elif orientation == "coronal":
            horizontal_axis, vertical_axis = 0, 2
        else:
            horizontal_axis, vertical_axis = 1, 2
        horizontal_margin = max(2, int(math.ceil(margin_mm / self.spacing_xyz[horizontal_axis])))
        vertical_margin = max(2, int(math.ceil(margin_mm / self.spacing_xyz[vertical_axis])))
        horizontal = self.focus_bbox_xyz[horizontal_axis]
        vertical = self.focus_bbox_xyz[vertical_axis]
        left = max(0, int(horizontal[0]) - horizontal_margin)
        right = min(self.shape_xyz[horizontal_axis], int(horizontal[1]) + horizontal_margin)
        top = max(0, int(vertical[0]) - vertical_margin)
        bottom = min(self.shape_xyz[vertical_axis], int(vertical[1]) + vertical_margin)
        return QtCore.QRectF(
            float(left),
            float(top),
            float(max(1, right - left)),
            float(max(1, bottom - top)),
        )

    def _slice_changed(self, value: int) -> None:
        with QtCore.QSignalBlocker(self.slice_spin):
            self.slice_spin.setValue(value)
        if self.crosshair_xyz is not None:
            updated = list(self.crosshair_xyz)
            depth_axis = {"axial": 2, "coronal": 1, "sagittal": 0}[self.orientation]
            updated[depth_axis] = float(value)
            self.crosshair_xyz = tuple(updated)
        self._update_canvas_contexts()
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

    def _pull_radius_mm(self) -> float:
        return float(self.pull_slider.value())

    def _pull_size_changed(self, _value: int) -> None:
        self.pull_value.setText("{:.1f} mm".format(self._pull_radius_mm()))
        self.canvas.influence_radius_mm = self._pull_radius_mm()
        self.canvas.update()

    def _display_spacing_for_orientation(self, orientation: str) -> tuple[float, float]:
        if orientation == "axial":
            return self.spacing_xyz[0], self.spacing_xyz[1]
        if orientation == "coronal":
            return self.spacing_xyz[0], self.spacing_xyz[2]
        return self.spacing_xyz[1], self.spacing_xyz[2]

    def _update_canvas_brush_radius(self) -> None:
        horizontal_spacing, vertical_spacing = self._display_spacing_for_orientation(
            self.orientation
        )
        self.canvas.display_spacing_uv = (horizontal_spacing, vertical_spacing)
        ffd_values = dict((self.request.get("igac") or {}).get("ffd") or {})
        self.canvas.boundary_capture_mm = max(
            min(horizontal_spacing, vertical_spacing),
            float(ffd_values.get("boundary_capture_max_mm", ffd_values.get("boundary_capture_mm", 8.0))),
        )
        self.canvas.boundary_capture_pixels = max(
            4.0, float(ffd_values.get("boundary_capture_pixels", 16.0))
        )
        trace_values = dict((self.request.get("igac") or {}).get("trace") or {})
        self.canvas.trace_capture_mm = max(
            min(horizontal_spacing, vertical_spacing),
            float(trace_values.get("capture_max_mm", 12.0)),
        )
        self.canvas.trace_capture_pixels = max(
            4.0, float(trace_values.get("capture_pixels", 28.0))
        )
        self.canvas.influence_radius_mm = self._pull_radius_mm()
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

    def _contour_trace_started(self, target: tuple[float, float]) -> None:
        if self.worker is None or not self.ready:
            return
        trace_values = dict((self.request.get("igac") or {}).get("trace") or {})
        ffd_values = dict((self.request.get("igac") or {}).get("ffd") or {})
        self._trace_context = {
            "orientation": self.orientation,
            "slice_index": self.slice_slider.value(),
            "influence_radius_mm": self._pull_radius_mm(),
            "maximum_drag_ratio": float(ffd_values.get("maximum_drag_ratio", 0.65)),
            "capture_distance_mm": float(trace_values.get("capture_max_mm", 12.0)),
        }
        self.worker.submit(
            "contour_trace_start",
            {"target_u": float(target[0]), "target_v": float(target[1])},
        )

    def _contour_trace_requested(self, target: tuple[float, float]) -> None:
        if self.worker is None or not self.ready or self._trace_context is None:
            return
        context = self._trace_context
        self.worker.submit(
            "contour_trace_update",
            {
                "orientation": str(context["orientation"]),
                "slice_index": int(context["slice_index"]),
                "target_u": float(target[0]),
                "target_v": float(target[1]),
                "influence_radius_mm": float(context["influence_radius_mm"]),
                "maximum_drag_ratio": float(context["maximum_drag_ratio"]),
                "anchor_radius_mm": 0.0,
                "capture_distance_mm": float(context["capture_distance_mm"]),
            },
        )

    def _contour_trace_finished(self) -> None:
        if self.worker is not None and self.ready and self._trace_context is not None:
            self.worker.submit("contour_trace_end")
        self._trace_context = None

    def _cancel_worker_interaction(self, mode: str) -> None:
        if self.worker is None or not self.ready:
            return
        if str(mode) == "ffd":
            self._boundary_context = None
            self.worker.submit("boundary_pull_cancel")
        elif str(mode) == "trace":
            self._trace_context = None
            self.worker.submit("contour_trace_cancel")

    def _tool_changed(self, _button: Any = None) -> None:
        mode = self._selected_value(self.tool_group, "ffd")
        self.canvas.interaction_mode = mode
        self.pull_controls.setVisible(True)
        if mode == "ffd":
            self.tool_hint.setText("Direct contour adjustment")
        else:
            self.tool_hint.setText("Edge-guided contour adjustment")
        self.canvas.update()

    def _gesture_mode_changed(self, mode: str) -> None:
        labels = {
            "boundary_captured": "Contour captured",
            "boundary_miss": "Move closer to the cyan contour",
            "trace_started": "Following pointer path",
            "trace_miss": "Move closer to the cyan contour",
        }
        self.tool_hint.setText(labels.get(str(mode), "Correcting boundary"))

    def _toggle_running(self, _checked: bool = False) -> None:
        if self.worker is None or not self.ready:
            return
        if bool(self.run_button.property("running")):
            self.run_button.setProperty("running", False)
            self.run_button.setText("Refine local edit")
            self.worker.submit("stop")
            return
        self.run_button.setProperty("running", True)
        self.run_button.setText("Stop")
        self.worker.submit("refine")

    def _set_local_refinement_available(self, available: bool) -> None:
        if not bool(self.run_button.property("running")):
            self.run_button.setEnabled(bool(available) and self.ready)

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
            self.crosshair_xyz = tuple(
                (float(value[0]) + float(value[1]) - 1.0) / 2.0
                for value in self.focus_bbox_xyz
            )
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
        self.canvas.set_view_context(
            self.orientation,
            int(self.slice_slider.value()),
            self.crosshair_xyz,
        )
        self.iteration_label.setText("Iteration {:,}".format(int(iteration)))
        self.speed_label.setText("{:.1f} it/s".format(float(fps)) if fps > 0 else "— it/s")

    def _on_orthogonal_views(self, payload: dict[str, Any]) -> None:
        values = payload.get("crosshair_xyz") or []
        if len(values) == 3 and not self.crosshair_timer.isActive():
            self._set_crosshair_xyz(
                (float(values[0]), float(values[1]), float(values[2])),
                submit=False,
            )
        views = payload.get("views") or {}
        for canvas in self.context_canvases:
            view = views.get(canvas.view_orientation) or {}
            frame = view.get("frame")
            if not isinstance(frame, dict):
                continue
            index = int(view.get("index", canvas.view_slice_index))
            canvas.set_frame(frame)
            canvas.set_view_context(
                canvas.view_orientation,
                index,
                self.crosshair_xyz,
            )

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
        self.canvas.set_phase(str(phase), str(text))
        for context in self.context_canvases:
            context.set_phase(str(phase), str(text))
        if hasattr(self, "run_button"):
            running = str(phase) == "running"
            self.run_button.setProperty("running", running)
            self.run_button.setText("Stop" if running else "Refine local edit")
            if running:
                self.run_button.setEnabled(True)

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
        self._set_local_refinement_available(True)
        self.roi_label.setText("Workspace 196 × 182 × 144 · about 544 MB")
        self.volume_label.setText("Mask 38.4 cm³")
        self.iteration_label.setText("Iteration 0")
        self.speed_label.setText("Paused")
        height, width = 640, 640
        self.shape_xyz = (width, height, 180)
        self.focus_bbox_xyz = [[270, 461], [200, 451], [70, 111]]
        self.crosshair_xyz = (365.0, 325.0, 90.0)
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
        origin_segments = [
            QtCore.QLineF(segment) for segment in self.canvas._boundary_segments
        ]
        origin_path_exact = QtGui.QPainterPath(self.canvas._boundary_path_exact)
        origin_path_smooth = QtGui.QPainterPath(self.canvas._boundary_path_smooth)
        context_views: dict[str, dict[str, Any]] = {}
        for orientation, plane_width, plane_height, center_u in (
            ("coronal", width, self.shape_xyz[2], 365.0),
            ("sagittal", height, self.shape_xyz[2], 325.0),
        ):
            plane_y, plane_x = np.mgrid[:plane_height, :plane_width]
            context_image = 0.12 + 0.52 * np.exp(
                -(
                    ((plane_x - plane_width * 0.5) / (plane_width * 0.36)) ** 2
                    + ((plane_y - plane_height * 0.52) / (plane_height * 0.46)) ** 2
                )
            )
            context_image += 0.16 * np.exp(
                -(
                    ((plane_x - center_u) / 92.0) ** 2
                    + ((plane_y - 92.0) / 42.0) ** 2
                )
            )
            context_mask = (
                ((plane_x - center_u) / 96.0) ** 2
                + ((plane_y - 92.0) / 44.0) ** 2
                < 1.0
            )
            depth_axis = {"coronal": 1, "sagittal": 0}[orientation]
            context_views[orientation] = {
                "index": int(round(self.crosshair_xyz[depth_axis])),
                "frame": {
                    "image": np.clip(context_image, 0.0, 1.0),
                    "raw_image": np.asarray(context_image * 1000.0, dtype=np.float32),
                    "mask": context_mask,
                },
            }
        self._on_orthogonal_views(
            {"crosshair_xyz": list(self.crosshair_xyz), "views": context_views}
        )
        vertical = ((yy - 325.0) / 125.0) ** 2
        half_width = 95.0 * np.sqrt(np.clip(1.0 - vertical, 0.0, 1.0))
        local_pull = 13.0 * np.exp(-((yy - 310.0) / 32.0) ** 2)
        preview_mask = (
            (vertical < 1.0)
            & (xx >= 365.0 - half_width)
            & (xx <= 365.0 + half_width + local_pull)
        )
        self.canvas.set_frame(
            {
                "image": np.clip(image, 0, 1),
                "raw_image": np.asarray(image * 1000.0, dtype=np.float32),
                "mask": preview_mask,
            }
        )
        self.canvas._boundary_dragging = True
        self.canvas._boundary_start = (459.5, 325.5)
        self.canvas._boundary_current = (472.5, 310.0)
        self.canvas._boundary_applied_current = (472.5, 310.0)
        self.canvas._interaction_origin_segments = origin_segments
        self.canvas._interaction_origin_path_exact = origin_path_exact
        self.canvas._interaction_origin_path_smooth = origin_path_smooth
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
QMainWindow, QWidget#igacRoot {
    background: #edf1f4;
}
QMainWindow::separator {
    background: #d7dde4;
}
QStatusBar {
    background: #f8fafb;
    border-top: 1px solid #d7dde4;
    color: #5f6b78;
}
QLabel#title {
    color: #14212b;
    font-size: 17pt;
    font-weight: 650;
}
QLabel#subtitle {
    color: #6b7682;
    font-size: 9.5pt;
}
QLabel#statusPill, QLabel#devicePill {
    border-radius: 6px;
    padding: 4px 9px;
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
    color: #40505d;
    background: #e6eaee;
    border: 1px solid #ccd4dc;
}
QFrame#workspace {
    background: #0e161d;
    border: 1px solid #202d37;
    border-radius: 7px;
}
QFrame#workspace QWidget {
    background: transparent;
}
QFrame#workspace QLabel#viewportMeta,
QFrame#workspace QLabel#viewportLabel {
    color: #aebbc5;
    font-size: 8.8pt;
}
QFrame#workspace QSpinBox {
    color: #e9eff3;
    background: #17232c;
    border: 1px solid #35444f;
    border-radius: 5px;
    padding: 3px 7px;
}
QWidget#darkSegments QPushButton,
QPushButton#viewportButton,
QToolButton#viewportToolButton {
    min-height: 28px;
    color: #b9c5cd;
    background: #17232c;
    border: 1px solid #35444f;
    border-radius: 5px;
    padding: 1px 10px;
}
QWidget#darkSegments QPushButton:hover,
QPushButton#viewportButton:hover,
QToolButton#viewportToolButton:hover {
    color: #f5f8fa;
    background: #20303a;
    border-color: #50616d;
}
QWidget#darkSegments QPushButton:checked {
    color: #ecfffc;
    background: #147c75;
    border-color: #2ca79d;
    font-weight: 650;
}
QFrame#workspace QSlider::groove:horizontal {
    height: 4px;
    border-radius: 2px;
    background: #35434e;
}
QFrame#workspace QSlider::sub-page:horizontal {
    border-radius: 2px;
    background: #3fb3a8;
}
QFrame#workspace QSlider::handle:horizontal {
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
    background: #f4fbfa;
    border: 2px solid #2c9b92;
}
QWidget#inspector {
    background: #ffffff;
    border: 1px solid #d7dde4;
    border-radius: 7px;
}
QScrollArea#inspectorScroll,
QWidget#inspectorBody {
    background: #ffffff;
    border: none;
}
QFrame#inspectorFooter {
    background: #f7f9fa;
    border: none;
    border-top: 1px solid #dce2e8;
    border-bottom-left-radius: 7px;
    border-bottom-right-radius: 7px;
}
QLabel#inspectorHeading {
    color: #7a8590;
    font-size: 8pt;
    font-weight: 700;
}
QLabel#fieldLabel {
    color: #52606d;
    font-size: 9pt;
}
QLabel#fieldValue {
    color: #147c75;
    font-size: 9pt;
    font-weight: 650;
}
QLabel#toolSummary {
    color: #40505d;
    font-size: 9pt;
    padding: 1px 0 2px 1px;
}
QLabel#inspectorFootnote {
    color: #7a8590;
    font-size: 8.5pt;
}
QFrame#inspectorSeparator {
    color: #e2e7ec;
    background: #e2e7ec;
    border: none;
}
QWidget#lightSegments QPushButton {
    min-height: 32px;
    background: #f1f4f6;
    border: 1px solid #d5dde4;
    color: #4a5865;
}
QWidget#lightSegments QPushButton:hover {
    background: #e9eef1;
    border-color: #b8c3cc;
}
QWidget#lightSegments QPushButton:checked,
QFrame#inspectorFooter QPushButton:checked {
    color: #0d645f;
    background: #e1f3f0;
    border-color: #62b9b0;
    font-weight: 650;
}
QPushButton#quietAction {
    color: #40505d;
    background: transparent;
    border-color: #d2dae1;
}
QPushButton#quietAction:hover {
    background: #f2f5f7;
    border-color: #aebbc5;
}
QToolButton#disclosure {
    min-height: 26px;
    color: #53616d;
    background: transparent;
    border: none;
    padding: 2px 0;
    text-align: left;
    font-weight: 600;
}
QToolButton#disclosure:hover {
    color: #147c75;
}
QPushButton#primary {
    min-height: 34px;
    color: #ffffff;
    background: #147c75;
    border: 1px solid #147c75;
    border-radius: 6px;
    font-weight: 650;
}
QPushButton#primary:hover {
    background: #0f6c66;
    border-color: #0f6c66;
}
QPushButton#primary:pressed {
    background: #0b5c57;
}
QPushButton#primary:disabled {
    color: #8d9ba4;
    background: #e2e7ea;
    border-color: #e2e7ea;
}
QSlider::groove:horizontal {
    height: 4px;
    border-radius: 2px;
    background: #dbe2e7;
}
QSlider::sub-page:horizontal {
    border-radius: 2px;
    background: #4baaa2;
}
QSlider::handle:horizontal {
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
    background: #ffffff;
    border: 2px solid #278f87;
}
QSlider::handle:horizontal:hover {
    background: #e5f7f4;
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
