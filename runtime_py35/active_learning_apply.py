# -*- coding: utf-8 -*-
"""Shared active-learning overlay monitor/apply family for the Mimics hosts.

Extracted from flexict_mimics (R93 step 6) so the FlexiCT and nnU-Net entry
points run one implementation: each host module calls bind() at import time
and receives the whole _al_* family as attributes. The review window
(tools/flexict_active_learning_ui.py) and the apply-request / annotation-state
file schemas are shared by both backends, so the monitor only differs in
workspace, monitor key, and the backend label shown to the annotator.

Design contract:
- Closures over ``host`` instead of a module-level _HOST global: both hosts
  can be imported into the SAME Mimics Python 3.5 process (the annotator
  clicks both menu entries); a single shared global flipped by the second
  import would run one host's monitor against the other's workspace.
- Every cross-call inside the family routes through ``host.<name>`` so tests
  that patch a host-module attribute (flexict_mimics._al_mark_request = ...)
  intercept the internal callers too.
- Host attributes used: _workspace(), _read_json, _write_json, _log,
  _external_python(), _project_root(), _MONITORS, _start_monitor. mimics,
  mimics_mask_apply and runtime_common are also resolved through the host at
  call time (host.mimics, ...): the test suites pop and re-import a host
  against stub modules, and a module-level import here would keep this module
  (cached in sys.modules forever) bound to the first stub it saw.

Py3.5 constraints: no f-strings, no pathlib, .format() with positional
indexes only. Never imports from the tools package.
"""

from __future__ import print_function

import json
import logging
import math
import os
import shutil
import subprocess
import threading
import time
import traceback
import uuid


def _matrix_close_payload(a, b, tolerance=1e-4):
    if not a or not b:
        return False
    try:
        for row_index in range(4):
            for column_index in range(4):
                if abs(
                    float(a[row_index][column_index])
                    - float(b[row_index][column_index])
                ) > tolerance or not all(math.isfinite(float(value)) for value in (
                        a[row_index][column_index], b[row_index][column_index])):
                    return False
    except Exception:
        return False
    return True


def bind(host, monitor_key, backend_label):
    """Inject the active-learning family onto a Mimics host module.

    ``monitor_key`` namespaces the host's _MONITORS entry ("flexict_al_
    requests" / "nnunet_al_requests"); ``backend_label`` ("FlexiCT" /
    "nnU-Net") names the backend in mask titles, log lines, dialogs and the
    undo label. The apply-request and annotation-state file schemas stay the
    shared "flexict_al_apply_request.v1" / "flexict_annotation_state.v1" —
    one review UI reads and writes both backends.
    """
    host._AL_MONITOR_KEY = monitor_key
    host._AL_APPLY_TRANSACTIONS = {}
    # R04: 'viewed' writes that lost the guard race (the external review window
    # held the lock) wait here and retry on a later monitor tick instead of
    # blocking the Mimics GUI thread. Same process-local lifetime as
    # _AL_APPLY_TRANSACTIONS: a Mimics restart drops a pending best-effort
    # 'viewed' entry — acceptable, the external window re-writes state on any
    # explicit action.
    host._AL_PENDING_VIEWED = []

    def _al_request_dir(job_dir):
        return os.path.join(job_dir, "apply_requests")

    def _al_pending_requests():
        """Newest-first overlay requests across every active-learning job."""
        jobs = os.path.join(host._workspace(), "jobs")
        requests = []
        if not os.path.isdir(jobs):
            return requests
        try:
            names = os.listdir(jobs)
        except OSError:
            return requests
        for name in names:
            status = host._read_json(os.path.join(jobs, name, "status.json"), {}) or {}
            if str(status.get("kind") or "") != "active_learning":
                continue
            if str(status.get("status") or "") != "completed":
                continue
            request_dir = _al_request_dir(os.path.join(jobs, name))
            if not os.path.isdir(request_dir):
                continue
            try:
                files = os.listdir(request_dir)
            except OSError:
                continue
            for file_name in files:
                if not file_name.endswith(".json"):
                    continue
                path = os.path.join(request_dir, file_name)
                payload = host._read_json(path, {}) or {}
                if not payload:
                    continue
                if str(payload.get("state") or "pending") not in (
                    "pending", "converting"):
                    continue
                payload["_request_path"] = path
                payload["_job_dir"] = os.path.join(jobs, name)
                requests.append(payload)
        requests.sort(
            key=lambda row: float(row.get("requested_at_epoch") or 0),
            reverse=True,
        )
        return requests

    def _al_tick():
        """Poll for review-UI overlay requests and apply the newest one.

        Conversion runs in a background thread (communicate can take minutes);
        this tick only starts it and, once result.json exists, applies the
        finished buffers on the GUI thread. Both stages are guarded by the
        "busy" flag so ticks never overlap on one request.
        """
        if host._AL_APPLY_TRANSACTIONS.get("busy"):
            return
        host._AL_APPLY_TRANSACTIONS["busy"] = True
        try:
            # R04: retry 'viewed' entries that lost the guard race (each retry
            # is the same non-blocking single attempt; entries that keep losing
            # stay pending for the next tick).
            if host._AL_PENDING_VIEWED:
                retries = host._AL_PENDING_VIEWED[:]
                del host._AL_PENDING_VIEWED[:]
                for job_dir, case_id in retries:
                    host._al_mark_viewed(job_dir, case_id)
            transaction = host._AL_APPLY_TRANSACTIONS.get("conversion")
            if transaction is not None:
                # Stage two: the background thread wrote result.json when the
                # conversion finished. Until then, keep the GUI responsive and
                # wait for a later tick.
                if os.path.isfile(transaction["result_path"]):
                    host._AL_APPLY_TRANSACTIONS.pop("conversion", None)
                    host._al_finish_conversion(transaction)
                return
            requests = _al_pending_requests()
            if not requests:
                return
            # R62-6: a "converting" request with no live in-memory transaction
            # means Mimics was restarted mid-conversion (the wait thread and
            # _AL_APPLY_TRANSACTIONS are process-local). Nothing will ever
            # finish it — fail it explicitly instead of leaving it hanging
            # forever in a state the review UI never reports.
            pending = []
            for request in requests:
                if str(request.get("state") or "") == "converting":
                    host._al_mark_request(
                        request["_request_path"], "failed",
                        "Mimics was restarted while this overlay was "
                        "converting. Request it again from the review window.",
                    )
                else:
                    pending.append(request)
            if not pending:
                return
            host._al_apply_request(pending[0])
        except Exception as exc:
            host._log(
                logging.ERROR,
                "{0} active-learning overlay error: {1}\n{2}".format(
                    backend_label, exc, traceback.format_exc()
                ),
            )
        finally:
            host._AL_APPLY_TRANSACTIONS["busy"] = False

    def _al_mark_request(request_path, state, detail=""):
        payload = host._read_json(request_path, {}) or {}
        payload["state"] = state
        payload["detail"] = str(detail)
        payload["updated_at_epoch"] = time.time()
        host._write_json(request_path, payload)

    def _al_case_source_geometry(job_dir, case_id):
        geometries = host._read_json(
            os.path.join(job_dir, "input_geometries.json"), {}
        ) or {}
        cases = geometries.get("cases") or {}
        return cases.get(case_id) or {}

    def _al_mask_paths(job_dir, case_id, what):
        """The NIfTI(s) an overlay request maps to."""
        # S10: each uncertainty run writes a fresh uncertainty_run_<ts> directory,
        # recorded in the job status. Jobs created before S10 keep the legacy
        # fixed directory, so the status key is preferred with a fallback.
        status = host._read_json(os.path.join(job_dir, "status.json"), {}) or {}
        uncertainty_dir = str(
            status.get("uncertainty_dir") or "") or os.path.join(
                job_dir, "uncertainty")
        if what == "consensus":
            path = os.path.join(uncertainty_dir, "{0}_consensus.nii.gz".format(case_id))
            if os.path.isfile(path):
                return [("{0} Consensus".format(backend_label), path)]
            return []
        bands_dir = os.path.join(uncertainty_dir, "bands")
        bands = status.get("uncertainty_bands") or {}
        if bands.get("coincident"):
            path = os.path.join(bands_dir, "{0}_high.nii.gz".format(case_id))
            if os.path.isfile(path):
                return [("{0} Disagreement".format(backend_label), path)]
        rows = []
        for level, title in (
            ("high", "{0} Uncertainty (High)".format(backend_label)),
            ("moderate", "{0} Uncertainty (Moderate)".format(backend_label)),
        ):
            path = os.path.join(bands_dir, "{0}_{1}.nii.gz".format(case_id, level))
            if os.path.isfile(path):
                rows.append((title, path))
        return rows

    def _al_open_case(job_dir, case_id):
        """Open the case's .mcs project in this Mimics session.

        Returns (ok, detail). The path comes from input_geometries.json, where
        the AL materializer recorded it when the pool was built. Never stomps
        on an open unsaved session: if another project is active, the user is
        told to save and close it first (same protection import_undo_mimics
        applies). Only projects created through the 01_Import flow exist for
        a dataset, so a missing .mcs is a guidance case, not an error state.
        """
        geometry = host._al_case_source_geometry(job_dir, case_id)
        mcs_path = str(geometry.get("mcs_path") or "")
        source_path = str(geometry.get("source_image_path") or "")
        if not mcs_path or not os.path.isfile(mcs_path):
            return False, (
                "未找到病例 {0} 的 .mcs 工程。请先生成一个"
                "（01 Import，病例文件夹 {1}），然后重试。".format(
                    case_id, source_path or "（未记录源路径）")
            )
        # If the requested project is already active, opening it again would
        # close/reload it needlessly — skip straight to success. F17: the state
        # comes from the documented project API; "cannot tell" is never treated
        # as "nothing is open" and blocks switching to another project.
        state, active = host.runtime_common.current_project_state(host.mimics)
        if state == "path" and active and os.path.abspath(str(active)) == os.path.abspath(mcs_path):
            return True, "project already open"
        if state == "unknown":
            host.mimics.dialogs.message_box(
                "无法确认 Mimics 中是否打开了其他工程"
                "（工程查询失败，或当前版本不支持该查询）。\n\n"
                "打开病例 {0} 需要它自己的工程（{1}）；当前会话状态未知时，"
                "窗口不会切换工程，因此未做任何改动。\n\n"
                "第 1 步：关闭所有已打开的工程（必要时先保存）。\n"
                "第 2 步：回到 {2} 主动学习窗口，再次打开该病例。".format(
                    case_id, mcs_path, backend_label
                ),
                title=backend_label,
                ui_blocking=False,
            )
            return False, "current project state unknown; not switching projects"
        if state in ("path", "unnamed"):
            host.mimics.dialogs.message_box(
                "Mimics 中打开了其他工程。\n\n"
                "打开病例 {0} 需要它自己的工程（{1}）。\n\n"
                "第 1 步：关闭当前打开的工程（必要时先保存）。\n"
                "第 2 步：回到 {2} 主动学习窗口，再次打开该病例。".format(
                    case_id, mcs_path, backend_label
                ),
                title=backend_label,
                ui_blocking=False,
            )
            return False, "another project is open"
        try:
            host.mimics.file.open_project(filename=mcs_path)
        except Exception as exc:
            return False, "could not open {0}: {1}".format(mcs_path, exc)
        return True, "opened {0}".format(mcs_path)

    def _annotation_state_guard(path, wait_seconds=2.0):
        """Cross-process single-writer guard for annotation_state.json.

        UI31/F45: the external review UI (Py3.13, flexict_active_learning_ui)
        and this monitor both read-modify-write that JSON; without a lock one
        writer's read-modify-write can clobber the other's case entries. The
        one-byte msvcrt/fcntl lock on <path>.guard follows the same protocol as
        runtime_common._open_resource_guard (kept local so Py3.5 never imports
        from the tools package). Returns None when contended past the wait —
        callers then skip the write (viewed is best-effort, queued for a later
        tick retry) rather than block the Mimics GUI (iron rule 1). The Mimics
        caller passes wait_seconds=0 (R04): one non-blocking attempt; the
        bounded wait exists for callers off the GUI thread."""
        parent = os.path.dirname(path)
        if parent and not os.path.isdir(parent):
            try:
                os.makedirs(parent)
            except OSError:
                if not os.path.isdir(parent):
                    return None
        try:
            handle = open(path + ".guard", "a+b")
        except OSError:
            return None
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            deadline = time.time() + max(0.0, float(wait_seconds))
            while True:
                try:
                    handle.seek(0)
                    if os.name == "nt":
                        import msvcrt
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return handle
                except (IOError, OSError):
                    if time.time() >= deadline:
                        handle.close()
                        return None
                    time.sleep(0.02)
        except Exception:
            try:
                handle.close()
            except Exception:
                pass
            return None

    def _release_annotation_state_guard(handle):
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            handle.close()
        except Exception:
            pass

    def _al_mark_viewed(job_dir, case_id):
        """Record that the case's overlay was viewed in the state file.

        UI14/F05: applying an overlay only means the annotator is *looking* at
        the case, not that the organ was corrected and confirmed. The entry is
        therefore "viewed", never "annotated" — only the review window's
        explicit 标记为已标注 action writes "annotated". Same schema the external
        UI writes (flexict_annotation_state.v1), and never downgrades an
        explicit annotated/skipped entry.

        UI31/F45: the read-modify-write runs under the cross-process guard so a
        concurrent UI write (e.g. the annotator marking another case annotated
        in the review window) is never clobbered."""
        path = os.path.join(job_dir, "annotation_state.json")
        # R04: this runs on the Mimics GUI thread (from _al_finish_conversion
        # via the monitor tick), so the guard is tried once without waiting —
        # the old 2 s bounded wait froze slicing/tooltips whenever the external
        # review window held the lock. On contention the entry goes to the
        # pending list and the next tick retries it.
        guard = host._annotation_state_guard(path, wait_seconds=0.0)
        if guard is None:
            pending = (job_dir, case_id)
            if pending not in host._AL_PENDING_VIEWED:
                host._AL_PENDING_VIEWED.append(pending)
            return
        try:
            payload = host._read_json(path, {}) or {}
            if str(payload.get("schema_version") or "") != "flexict_annotation_state.v1":
                payload = {"schema_version": "flexict_annotation_state.v1",
                           "cases": {}}
            cases = payload.setdefault("cases", {})
            entry = cases.get(case_id) or {}
            if str(entry.get("state") or "new") != "new":
                return
            cases[case_id] = {
                "state": "viewed",
                "updated_at_epoch": time.time(),
            }
            payload["updated_at_epoch"] = time.time()
            host._write_json(path, payload)
        finally:
            host._release_annotation_state_guard(guard)

    def _al_apply_request(request):
        """Verify the active image matches the request's case, then apply the
        overlay masks through the same bridge path the prediction flow uses.

        `what` values: "bands"/"consensus" apply an overlay to the already
        active case; "open" opens the case's .mcs project; "open_bands" opens
        the project and then applies the uncertainty bands in one action."""
        request_path = request["_request_path"]
        job_dir = request["_job_dir"]
        case_id = str(request.get("case_id") or "")
        what = str(request.get("what") or "bands")
        if what == "open":
            opened, detail = host._al_open_case(job_dir, case_id)
            host._al_mark_request(
                request_path, "applied" if opened else "failed", detail)
            if opened:
                host._log(
                    logging.INFO,
                    "{0} active-learning opened case {1}: {2}.".format(
                        backend_label, case_id, detail),
                )
            return
        if what == "open_bands":
            opened, detail = host._al_open_case(job_dir, case_id)
            if not opened:
                host._al_mark_request(request_path, "failed", detail)
                return
            # The just-opened project's active image is this case in the
            # normal flow; the verified-grid check below confirms it and
            # fails with guidance if the project's image is not the case.
            what = "bands"
        masks = host._al_mask_paths(job_dir, case_id, what)
        status = host._read_json(os.path.join(job_dir, "status.json"), {}) or {}
        zero_disagreement = what == "bands" and any(
            str(row.get("case") or "") == case_id and row.get("max") == 0
            for row in status.get("ranking") or [])
        if not masks and not zero_disagreement:
            host._al_mark_request(request_path, "failed", "overlay files not found")
            return
        # Verified-grid contract: the active image must still be this case on
        # its original source grid (the one the pool was materialized from).
        source_geometry = host.mimics_mask_apply._active_source_geometry_payload()
        if not source_geometry:
            host._al_mark_request(
                request_path, "failed",
                "The active image source geometry could not be verified.",
            )
            return
        expected = host._al_case_source_geometry(job_dir, case_id)
        if not expected:
            host._al_mark_request(
                request_path, "failed",
                "No recorded source geometry for case {0}.".format(case_id),
            )
            return
        if list(source_geometry.get("source_shape") or []) != list(
                expected.get("source_shape") or []) or not _matrix_close_payload(
                    source_geometry.get("source_voxel_to_ras_matrix"),
                    expected.get("source_voxel_to_ras_matrix"),
                ):
            host._al_mark_request(
                request_path, "failed",
                "The active image does not match case {0}. Open that case "
                "(its original source image) and retry.".format(case_id),
            )
            return
        # F20: geometry equality is not case identity. Two normalized cases
        # (e.g. the same patient's phases, or a shared resampling grid) can
        # carry identical shape/affine; the active image's recorded source
        # file must be the case's own source, not merely a same-grid case.
        active_source_path = str(source_geometry.get("source_image_path") or "")
        expected_source_path = str(expected.get("source_image_path") or "")
        if not (active_source_path and expected_source_path and
                host.mimics_mask_apply._same_path(
                    active_source_path, expected_source_path)):
            host._al_mark_request(
                request_path, "failed",
                "The active image is a different case: its source image is\n"
                "{0}\nbut case {1} was recorded with\n{2}.\n"
                "Open the case's own project/image and retry.".format(
                    active_source_path or "(unknown)", case_id,
                    expected_source_path or "(unknown)",
                ),
            )
            return
        target_grid = host.mimics_mask_apply._active_live_grid_payload()
        if not target_grid:
            host._al_mark_request(
                request_path, "failed",
                "The active Mimics image grid could not be measured.",
            )
            return
        # F20: geometry equality is not case identity. Two normalized cases can
        # share shape/affine, and the conversion runs asynchronously — the user
        # may switch projects or images before it finishes. Record who the
        # result belongs to so the finish step re-verifies before writing.
        active_image = None
        try:
            active_image = host.mimics.data.images.get_active()
        except Exception:
            active_image = None
        target_identity = {
            "project_path": host.mimics_mask_apply._current_project_path() or "",
            "image_guid": (
                host.mimics_mask_apply._mask_identity(active_image)
                if active_image is not None else ""
            ),
        }
        if not target_identity["image_guid"]:
            host._al_mark_request(
                request_path, "failed",
                "The active Mimics image could not be identified; the overlay "
                "was not started.",
            )
            return
        if not masks:
            host._al_mark_request(request_path, "applied",
                                  "模型预测一致，本例没有分歧带；这不代表标注正确。")
            host._al_mark_viewed(job_dir, case_id)
            return
        bridge_root = os.path.join(
            job_dir, "al_apply_" + uuid.uuid4().hex[:8]
        )
        buffers = os.path.join(bridge_root, "buffers")
        os.makedirs(buffers)
        axes, flips = host.mimics_mask_apply._buffer_mapping_from_config(
            host.mimics_mask_apply._config()
        )
        params = {
            "action": "prepare_masks_for_grid",
            "masks": [
                {"name": "al_{0}".format(index), "mask_path": path}
                for index, (_title, path) in enumerate(masks)
            ],
            "buffers_out": buffers,
            "target_shape": target_grid["target_shape"],
            "target_voxel_to_ras_matrix": target_grid[
                "target_voxel_to_ras_matrix"],
            "source_voxel_to_ras_matrix": expected.get(
                "source_voxel_to_ras_matrix"),
            "axes": axes,
            "flips": flips,
        }
        input_path = os.path.join(bridge_root, "input.json")
        result_path = os.path.join(bridge_root, "result.json")
        host._write_json(input_path, params)
        with open(input_path, "rb") as stdin_handle:
            process = subprocess.Popen(
                [host._external_python(), os.path.join(host._project_root(), "mimics_bridge.py")],
                stdin=stdin_handle,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                **host.runtime_common.background_process_kwargs()
            )
        # The conversion can run for minutes — never on the GUI timer thread.
        # Same pattern as the prediction flow's wait_bridge: the thread only
        # communicates and writes result.json; the Mimics-side application
        # happens in _al_finish_conversion, back on the monitor tick.
        host._al_mark_request(request_path, "converting", "conversion started")

        def wait_bridge():
            try:
                stdout, stderr = process.communicate(timeout=600)
                if process.returncode == 0:
                    result = json.loads(stdout.decode("utf-8"))
                else:
                    result = {
                        "status": "error",
                        "error": stderr.decode("utf-8", "replace")[-2000:],
                    }
            except Exception:
                try:
                    host.runtime_common.terminate_process_async(
                        process=process, graceful_seconds=0.0
                    )
                except Exception:
                    pass
                result = {"status": "error", "error": "conversion timed out"}
            try:
                host._write_json(result_path, result)
            except Exception:
                pass

        thread = threading.Thread(target=wait_bridge)
        thread.daemon = True
        thread.start()
        host._AL_APPLY_TRANSACTIONS["conversion"] = {
            "request": request,
            "result_path": result_path,
            "bridge_root": bridge_root,
            "masks": masks,
            "target_identity": target_identity,
            "target_grid": target_grid,
            "source_geometry": expected,
        }

    def _al_verify_conversion_buffers(result, masks):
        """F02: check every buffer exists with the right byte count up front.

        Returns (rows, error). rows is the ordered list of (title, buffer row)
        ready to apply; error is "" on success. All-or-nothing: a missing or
        truncated second band must not leave a half-applied overlay.
        """
        rows = []
        candidates = result.get("masks") or []
        names = [str(candidate.get("name") or "") for candidate in candidates]
        if len(set(names)) != len(names):
            return [], "conversion result contains duplicate buffer names"
        consumed = set()
        for index, (title, _path) in enumerate(masks):
            original_name = "al_{0}".format(index)
            matches = [candidate for candidate in candidates
                       if candidate.get("name") == original_name or (
                           candidate.get("original_name") == original_name
                           and candidate.get("name") == original_name + "_label{0}".format(
                               candidate.get("label_value")))]
            if not matches:
                return [], "conversion result is missing buffer {0}".format(index)
            if len(matches) > 1 and any(row.get("name") == original_name for row in matches):
                return [], "conversion mixes binary and split-label buffers"
            for row in matches:
                consumed.add(row["name"])
                output_path = str(row.get("output_path") or "")
                if not output_path or not os.path.isfile(output_path):
                    return [], "conversion buffer {0} was not written".format(index)
                shape = row.get("mimics_shape") or []
                try:
                    if len(shape) != 3 or any(int(v) != v or int(v) <= 0 for v in shape):
                        raise ValueError("invalid shape")
                    expected_bytes = int(shape[0]) * int(shape[1]) * int(shape[2])
                    actual_bytes = os.path.getsize(output_path)
                    label_value = int(row.get("label_value") or 0)
                    if row.get("name") != original_name and label_value <= 0:
                        raise ValueError("invalid label")
                except Exception:
                    return [], "conversion buffer {0} has no readable shape or label".format(index)
                if actual_bytes != expected_bytes:
                    return [], (
                        "conversion buffer {0} is truncated ({1} of {2} bytes)"
                        .format(index, actual_bytes, expected_bytes)
                    )
                rows.append((title, row))
        if consumed != set(names):
            return [], "conversion result contains unexpected buffers"
        return rows, ""

    def _al_target_still_matches(transaction):
        """F20: the async finish must land on the object it was launched on.

        Returns "" when the active project and image are still the recorded
        launch-time identity and the live source geometry still matches the
        case; otherwise a guidance message for the request detail.
        """
        identity = transaction.get("target_identity") or {}
        expected_guid = str(identity.get("image_guid") or "")
        launch_project = str(identity.get("project_path") or "")
        current_project = host.mimics_mask_apply._current_project_path() or ""
        if launch_project and not host.mimics_mask_apply._same_path(
                current_project, launch_project):
            return (
                "The project changed while the overlay was converting. Reopen "
                "case {0} and request the overlay again.".format(
                    transaction["request"].get("case_id") or "")
            )
        try:
            active_image = host.mimics.data.images.get_active()
        except Exception:
            active_image = None
        if active_image is None or host.mimics_mask_apply._mask_identity(
                active_image) != expected_guid:
            return (
                "The active image changed while the overlay was converting. "
                "Reactivate the case image and request the overlay again."
            )
        # Registration/resampling can change the physical grid without
        # replacing the Image object. Conversion belongs to the launch grid.
        expected_grid = transaction.get("target_grid")
        if expected_grid:
            current_grid = host.mimics_mask_apply._active_live_grid_payload() or {}
            if list(current_grid.get("target_shape") or []) != list(
                    expected_grid.get("target_shape") or []) or not _matrix_close_payload(
                        current_grid.get("target_voxel_to_ras_matrix"),
                        expected_grid.get("target_voxel_to_ras_matrix")):
                return "The active image grid changed while the overlay was converting. Request it again."
        return ""

    def _al_finish_conversion(transaction):
        """Apply a finished AL conversion result (called from the monitor tick).

        All Mimics API access (mask creation, buffer writes, dialogs) stays on
        the GUI thread; only the subprocess wait ran in the background.

        F02: the bridge buffers are validated up front and the bridge directory
        is deleted only after the whole transaction (apply or rollback) ends —
        it used to be rmtree'd before the buffers were read, so every real
        apply died with FileNotFoundError. F20: the finish step re-verifies
        the project/image identity captured at launch before creating masks.
        """
        request = transaction["request"]
        request_path = request["_request_path"]
        job_dir = request["_job_dir"]
        case_id = str(request.get("case_id") or "")
        masks = transaction["masks"]
        try:
            result = host._read_json(transaction["result_path"], {}) or {}
            if result.get("status") == "error" or not result:
                host._al_mark_request(
                    request_path, "failed",
                    str(result.get("error") or "conversion produced no result"),
                )
                return
            rows, error = host._al_verify_conversion_buffers(result, masks)
            if error:
                host._al_mark_request(request_path, "failed", error)
                return
            if not rows:
                host._al_mark_request(request_path, "failed", "no foreground voxels")
                return
            if request.get("what") == "consensus":
                status = host._read_json(os.path.join(job_dir, "status.json"), {}) or {}
                labels = status.get("labels") or [status.get("label_name") or ""]
                titled_rows = []
                for title, row in rows:
                    label_value = int(row.get("label_value") or 0)
                    if 0 < label_value <= len(labels):
                        value = labels[label_value - 1]
                        label_name = value.get("name", "") if isinstance(value, dict) else str(value)
                    else:
                        label_name = "label {0}".format(label_value)
                    titled_rows.append((title + " · " + label_name, row))
                rows = titled_rows
            error = host._al_target_still_matches(transaction)
            if error:
                host._al_mark_request(request_path, "failed", error)
                return
            applied = []
            created = []
            try:
                for title, row in rows:
                    mask = host.mimics_mask_apply._new_prediction_mask(title)
                    created.append(mask)
                    host.mimics_mask_apply._set_mask_from_u8(
                        mask,
                        row["output_path"],
                        row["mimics_shape"],
                        "Apply {0} Active-Learning Overlay".format(backend_label),
                    )
                    applied.append(str(getattr(mask, "name", "") or title))
            except Exception as exc:
                # F02 rollback: no half-applied overlay, no leftover empty
                # masks from this transaction.
                rollback_errors = []
                for mask in created:
                    try:
                        host.mimics.data.masks.delete(mask)
                    except Exception as rollback_exc:
                        rollback_errors.append("{0}: {1}".format(
                            str(getattr(mask, "name", "") or "(unnamed)"),
                            rollback_exc))
                detail = "overlay apply failed and was rolled back: {0}".format(exc)
                if rollback_errors:
                    detail = (
                        "Overlay apply failed: {0}. Rollback incomplete; "
                        "remove the remaining overlay masks manually: {1}"
                    ).format(exc, "; ".join(rollback_errors))
                host._al_mark_request(
                    request_path, "failed",
                    detail,
                )
                host._log(
                    logging.ERROR,
                    "{0} active-learning overlay apply failed for {1}: "
                    "{2}\n{3}".format(
                        backend_label, case_id, exc, traceback.format_exc()),
                )
                return
            host._al_mark_request(
                request_path, "applied",
                "applied: {0}".format(", ".join(applied)))
            # UI14/F05: viewing the overlay is not annotation. Record "viewed"
            # so the external window's Status column updates without a manual
            # click, but the case only enters the annotated set when the
            # annotator explicitly confirms it there.
            host._al_mark_viewed(job_dir, case_id)
            host._log(
                logging.INFO,
                "{0} active-learning overlay applied for {1}: {2}.".format(
                    backend_label, case_id, ", ".join(applied)),
            )
            try:
                host.mimics.dialogs.message_box(
                    "已为病例 {1} 应用 {0}。".format(
                        ", ".join(applied), case_id),
                    title=backend_label,
                    ui_blocking=False,
                )
            except TypeError:
                host.mimics.dialogs.message_box(
                    title=backend_label,
                    message="已为病例 {1} 应用 {0}。".format(
                        ", ".join(applied), case_id),
                )
        finally:
            # F02: cleanup is the last step of the transaction, not the first.
            # The bridge directory holds the only copy of the converted
            # buffers, so it may only go away once they are either applied or
            # the request has a terminal failed state.
            shutil.rmtree(transaction["bridge_root"], ignore_errors=True)

    def _start_al_request_monitor():
        if host._AL_MONITOR_KEY in host._MONITORS:
            return
        monitor = {
            "monitor_key": host._AL_MONITOR_KEY,
            "kind": "al_requests",
            "deadline": time.time() + 30 * 24 * 60 * 60,
            "last_line": "",
        }
        if host._start_monitor(monitor, 2.0):
            host._MONITORS[host._AL_MONITOR_KEY] = monitor

    host._al_request_dir = _al_request_dir
    host._al_pending_requests = _al_pending_requests
    host._al_tick = _al_tick
    host._al_mark_request = _al_mark_request
    host._al_case_source_geometry = _al_case_source_geometry
    host._al_mask_paths = _al_mask_paths
    host._al_open_case = _al_open_case
    host._annotation_state_guard = _annotation_state_guard
    host._release_annotation_state_guard = _release_annotation_state_guard
    host._al_mark_viewed = _al_mark_viewed
    host._al_apply_request = _al_apply_request
    host._al_verify_conversion_buffers = _al_verify_conversion_buffers
    host._al_target_still_matches = _al_target_still_matches
    host._al_finish_conversion = _al_finish_conversion
    host._start_al_request_monitor = _start_al_request_monitor
    return host
