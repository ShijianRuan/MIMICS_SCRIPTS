#!/usr/bin/env python3
"""Active-learning review window: ranked uncertainty over the unlabeled pool.

Reads a completed flexict active_learning job (ranking.csv / status.ranking),
tracks annotation progress per case (annotation_state.json next to the job),
and hands the selected case to the Mimics side: opening the case's project
and overlaying the uncertainty bands / consensus mask run in the foreground
Mimics via the runtime module's apply-request monitor, whose outcomes are
polled back here so the table and status line stay current.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from queue import Empty, Queue

from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "tools"):
    value = str(candidate)
    if value not in sys.path:
        sys.path.insert(0, value)

from flexict_common import TERMINAL_STATES  # noqa: E402
from nnunet_common import read_json, write_json_atomic  # noqa: E402
from ui_theme import (  # noqa: E402
    PALETTE,
    choose_existing_directory_async,
    choose_save_file_async,
    clamp_window_to_screen,
    configure_application,
    friendly_error,
    scroll_layout_body,
    stylesheet,
)


STATE_SCHEMA = "flexict_annotation_state.v1"

# P3-5: 病例状态列对标注者显示为中文（此前是 new/annotated/skipped）。
# UI14/F05: 叠加只把状态推进到"已查看"（viewed）——查看辅助带不等于完成
# 标注；进入"已标注"只能通过本窗口的显式确认按钮。
_CASE_STATE_LABELS = {
    "new": "待标注",
    "viewed": "已查看",
    "annotated": "已标注",
    "skipped": "已跳过",
}


def case_state_label(value: str) -> str:
    return _CASE_STATE_LABELS.get(str(value), str(value))


def find_al_jobs(workspace: str | Path | None) -> list[dict[str, Any]]:
    """Terminal active_learning jobs, newest first — completed ones first,
    then failed/cancelled ones (R62-10: a failed ranking run must stay
    visible with its error, not silently disappear from the list)."""
    from flexict_pipeline import list_flexict_jobs

    rows, failed_rows = [], []
    for status in list_flexict_jobs(workspace):
        if str(status.get("kind") or "") != "active_learning":
            continue
        state = str(status.get("status") or "")
        if state not in TERMINAL_STATES:
            continue
        (failed_rows if state in {"failed", "cancelled", "abandoned"}
         else rows).append(status)
    key = lambda row: float(row.get("created_at_epoch") or 0)  # noqa: E731
    rows.sort(key=key, reverse=True)
    failed_rows.sort(key=key, reverse=True)
    return rows + failed_rows


def find_running_al_job(workspace: str | None) -> dict[str, Any] | None:
    """The newest non-terminal active_learning job, or None (F04).

    Lets the review window show live progress of a ranking the user just
    started instead of an empty-state dead end.
    """
    from flexict_pipeline import list_flexict_jobs

    for status in list_flexict_jobs(workspace):
        if str(status.get("kind") or "") != "active_learning":
            continue
        if str(status.get("status") or "") in TERMINAL_STATES:
            continue
        return status
    return None


def load_ranking(job_dir: str | Path) -> list[dict[str, Any]]:
    """Ranking rows from the job's status.json (or ranking.csv fallback)."""
    status = read_json(Path(job_dir) / "status.json", {}) or {}
    ranking = status.get("ranking")
    if isinstance(ranking, list) and ranking:
        return [dict(row) for row in ranking]
    # Prefer the per-run ranking CSV the pipeline records in status (S10:
    # each run writes its own uncertainty_run_* directory); the fixed
    # "uncertainty" path only serves jobs written before that change.
    csv_path = (
        Path(str(status.get("ranking_csv") or ""))
        if status.get("ranking_csv")
        else Path(job_dir) / "uncertainty" / "ranking.csv"
    )
    if not csv_path.is_file():
        return []
    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "case": str(row.get("case") or ""),
                    "integrated": float(row.get("integrated") or 0.0),
                    "uncertain_vol": float(row.get("uncertain_vol") or 0.0),
                    "max": float(row.get("max") or 0.0),
                }
            )
    return rows


def load_annotation_state(job_dir: str | Path) -> dict[str, Any]:
    path = Path(job_dir) / "annotation_state.json"
    payload = read_json(path, {}) or {}
    if not isinstance(payload, dict) or str(
            payload.get("schema_version") or "") != STATE_SCHEMA:
        payload = {"schema_version": STATE_SCHEMA, "cases": {}}
    payload.setdefault("cases", {})
    return payload


def save_annotation_state(job_dir: str | Path, state: dict[str, Any]) -> None:
    state = dict(state)
    state["schema_version"] = STATE_SCHEMA
    state["updated_at_epoch"] = time.time()
    write_json_atomic(Path(job_dir) / "annotation_state.json", state)


def _annotation_state_guard(path: Path, wait_seconds: float = 2.0):
    """Cross-process single-writer guard for annotation_state.json.

    UI31/F45: the UI (this process) and the Mimics-side monitor
    (runtime_py35) both read-modify-write the same JSON; the atomic replace
    only prevents half files, not one writer clobbering the other's case
    entries. The one-byte msvcrt/fcntl lock on <path>.guard follows the same
    protocol as runtime_common._open_resource_guard (its Py3.13 twin lives
    in flexict_mimics.py), so both writers serialize their read-modify-write
    under it. Bounded wait; None means contended — the caller must fail
    closed rather than write unguarded.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(str(path) + ".guard", "a+b")
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
        handle.close()
        raise


def _release_annotation_state_guard(handle) -> None:
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


def set_case_state(job_dir: str | Path, case_id: str, value: str) -> None:
    """Merge one case's state under the cross-process guard (UI31).

    Re-reads the file inside the lock so a concurrent Mimics-side 'viewed'
    entry written between our load and save survives. Raises on lock
    contention or write failure — the caller shows the failure instead of
    reporting success for a state that was never persisted.
    """
    path = Path(job_dir) / "annotation_state.json"
    guard = _annotation_state_guard(path)
    if guard is None:
        raise TimeoutError(
            "标注状态文件正被 Mimics 写入，请稍候再试。"
        )
    try:
        state = load_annotation_state(job_dir)
        state["cases"][str(case_id)] = {
            "state": str(value),
            "updated_at_epoch": time.time(),
        }
        save_annotation_state(job_dir, state)
    finally:
        _release_annotation_state_guard(guard)


def request_updates(job_dir: str | Path,
                    since_epoch: float) -> list[dict[str, Any]]:
    """Applied/failed apply-requests newer than `since_epoch`, oldest first.

    The Mimics-side monitor rewrites each request file with a state and an
    updated_at_epoch when it finishes; this reads that outcome back so the
    window can refresh its table and status line without the annotator
    switching windows. Pure function over the job folder — the UI's poll
    timer simply calls it with its last-seen timestamp.
    """
    rows: list[dict[str, Any]] = []
    request_dir = Path(job_dir) / "apply_requests"
    if not request_dir.is_dir():
        return rows
    for path in request_dir.glob("*.json"):
        payload = read_json(path, {}) or {}
        state = str(payload.get("state") or "")
        if state not in ("applied", "failed"):
            continue
        updated = float(payload.get("updated_at_epoch") or 0.0)
        if updated <= since_epoch:
            continue
        rows.append({
            "case": str(payload.get("case_id") or ""),
            "what": str(payload.get("what") or ""),
            "state": state,
            "detail": str(payload.get("detail") or ""),
            "updated_at_epoch": updated,
        })
    rows.sort(key=lambda row: row["updated_at_epoch"])
    return rows


def pending_request_age(job_dir: str | Path,
                        now_epoch: float) -> float:
    """Seconds the oldest still-pending apply-request has been waiting.

    A request is pending until the Mimics-side monitor rewrites it with
    applied/failed. The UI uses this to hint that the monitor may not be
    running instead of leaving "本窗口会自动更新" on screen forever.
    Pure function over the job folder.
    """
    request_dir = Path(job_dir) / "apply_requests"
    if not request_dir.is_dir():
        return 0.0
    oldest = 0.0
    for path in request_dir.glob("*.json"):
        payload = read_json(path, {}) or {}
        if str(payload.get("state") or "") in ("applied", "failed"):
            continue
        requested = float(payload.get("requested_at_epoch") or 0.0)
        if requested > 0:
            oldest = max(oldest, now_epoch - requested)
    return oldest


def bands_degenerate(job: dict[str, Any]) -> bool:
    """True when the moderate and high uncertainty bands coincide.

    A two-model disagreement map has a single non-zero level, so the
    data-aware threshold clamp in the pipeline collapses the two bands
    onto the same voxels. The annotator should know the pair of overlay
    masks is one signal, not two.
    """
    bands = job.get("uncertainty_bands")
    if not isinstance(bands, dict):
        return False
    written = bands.get("written") or {}
    moderate = int(written.get("moderate") or 0)
    high = int(written.get("high") or 0)
    if moderate and moderate == high:
        return True
    thresholds = bands.get("moderate_threshold"), bands.get("high_threshold")
    if None in thresholds:
        return False
    return float(thresholds[0]) == float(thresholds[1]) and moderate > 0


class ActiveLearningWindow:
    def __init__(self, context, qt_modules):
        self.context = context
        self.QtCore, self.QtGui, self.QtWidgets = qt_modules
        QtCore, QtWidgets = self.QtCore, self.QtWidgets
        self.workspace = str(context.get("workspace") or "")
        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle('FlexiCT 主动学习')
        self.window.resize(980, 640)
        self.window.setMinimumSize(820, 520)
        # F37: at 150% DPI the available screen can be smaller than the
        # layout minimum; clamp so the window never exceeds the screen and
        # labels are reachable by resizing, not clipped.
        clamp_window_to_screen(self.window, (820, 520))
        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(22, 18, 22, 18)
        root.setSpacing(12)
        title = QtWidgets.QLabel('FlexiCT 主动学习')
        title.setObjectName("title")
        subtitle = QtWidgets.QLabel(
            '按两种模型的分歧提供复核顺序建议。双击病例可在 Mimics 中打开并叠加复核提示；分歧分数不代表准确率。'
        )
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(subtitle)

        # F04: the review half used to be reachable only after a run existed,
        # and the empty state pointed back at this very menu. A "new run"
        # form on the same window closes the loop: pick the unlabeled pool
        # once, submit, and the run below refreshes as it completes.
        self._new_run_group = self._build_new_run_form()
        self._new_run_group.hide()
        picker = QtWidgets.QHBoxLayout()
        picker.addWidget(QtWidgets.QLabel('排序任务'))
        self.job_combo = QtWidgets.QComboBox()
        self.job_combo.currentIndexChanged.connect(self._job_selected)
        picker.addWidget(self.job_combo, 1)
        refresh = QtWidgets.QPushButton('刷新')
        refresh.clicked.connect(self.refresh)
        picker.addWidget(refresh)
        self.new_button = QtWidgets.QPushButton('新建排序…')
        self.new_button.setObjectName("primary")
        self.new_button.setCheckable(True)
        self.new_button.clicked.connect(self._new_ranking_clicked)
        picker.addWidget(self.new_button)
        root.addLayout(picker)
        root.addWidget(self._new_run_group)

        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ['排序', '病例', '分数', '不确定体积 (mm³)', '状态', '路径']
        )
        self.table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(
            5, QtWidgets.QHeaderView.Stretch)
        # F15: the numeric columns must never be squeezed below their unit
        # headers by the two stretch columns.
        self.table.setColumnWidth(3, 130)
        self.table.horizontalHeader().setMinimumSectionSize(120)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.itemDoubleClicked.connect(self._open_and_overlay_selected)
        root.addWidget(self.table, 1)

        self.status_label = QtWidgets.QLabel('请选择一个已完成的主动学习运行。')
        self.status_label.setObjectName("hint")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        # Two rows: 8 single-line buttons at the window's 820px minimum clip
        # (each 6-char CJK button is ~105-120px with theme padding), so the
        # run actions and the marking actions split like-for-like.
        actions = QtWidgets.QHBoxLayout()
        # F04: the first ranking run had no reachable graphical entry — the
        # empty state told the annotator to use this very window, which only
        # lists completed runs. A "new ranking" action lives here now, so
        # first-run is possible from the same menu entry.
        self.open_button = QtWidgets.QPushButton('打开并叠加')
        self.open_button.clicked.connect(self._open_and_overlay_selected)
        actions.addWidget(self.open_button)
        self.overlay_button = QtWidgets.QPushButton('叠加不确定度')
        self.overlay_button.clicked.connect(self._overlay_selected)
        actions.addWidget(self.overlay_button)
        self.consensus_button = QtWidgets.QPushButton('应用共识 Mask')
        self.consensus_button.clicked.connect(self._apply_consensus)
        actions.addWidget(self.consensus_button)
        actions.addStretch(1)
        root.addLayout(actions)
        marking = QtWidgets.QHBoxLayout()
        self.annotated_button = QtWidgets.QPushButton('标记为已标注')
        self.annotated_button.clicked.connect(self._mark_annotated)
        marking.addWidget(self.annotated_button)
        self.skip_button = QtWidgets.QPushButton('标记为跳过')
        self.skip_button.clicked.connect(self._mark_skipped)
        marking.addWidget(self.skip_button)
        marking.addStretch(1)
        self.export_button = QtWidgets.QPushButton('导出 CSV')
        self.export_button.clicked.connect(self._export_csv)
        marking.addWidget(self.export_button)
        close = QtWidgets.QPushButton('关闭')
        close.clicked.connect(self.window.close)
        marking.addWidget(close)
        root.addLayout(marking)
        self.body_scroll = scroll_layout_body(root, first=3, footer_items=3)
        self.window.setCentralWidget(central)
        self._set_actions_enabled(False)
        self._jobs: list[dict[str, Any]] = []
        self._rows: list[dict[str, Any]] = []
        self.table.itemSelectionChanged.connect(
            lambda: self._set_actions_enabled(bool(self._rows)))
        self._state: dict[str, Any] = {}
        self._job_dir = ""
        self._request_poll_epoch = time.time()
        # F04: background submit results for a new ranking (Queue because
        # create_flexict_job must never run on the GUI thread).
        self._submit_results: Queue = Queue()
        self._seen_running_job_id = ""
        # Pair discovery runs off the GUI thread too (R62-9): loading state +
        # 120 s deadline + its own result queue.
        self._loading_pairs = True
        self._pair_deadline = time.time() + 120.0
        self._pair_results: Queue = Queue()
        self._submit_deadline = 0.0
        # R75 P2: the running-job scan stats every job directory — seconds
        # on a network workspace. It runs on a worker thread; the poll
        # timer only drains the result (铁律 1).
        self._running_scan_results: Queue = Queue()
        self._running_scan_pending = False
        # find_al_jobs stats every job directory — the same freeze the
        # running-job scan already avoids. refresh() is now async: it
        # scans on a worker thread and drains via the poll timer.
        self._refresh_results: Queue = Queue()
        self._refresh_pending = False
        # S08: annotating/skipping writes annotation_state.json under a
        # cross-process guard; when Mimics holds the lock the bounded wait
        # (up to 2 s) must not run on the GUI thread. Same worker + poll
        # pattern as the scans above.
        self._state_write_results: Queue = Queue()
        self._refresh_async()
        # The Mimics-side monitor reports outcomes by rewriting request
        # files; poll them so the table and status line follow along
        # without the annotator switching windows.
        self.poll_timer = QtCore.QTimer(self.window)
        self.poll_timer.setInterval(2000)
        self.poll_timer.timeout.connect(self._poll_requests)
        self.poll_timer.start()
        # Pair discovery + job submission also complete off the GUI thread.
        self.fast_timer = QtCore.QTimer(self.window)
        self.fast_timer.setInterval(80)
        self.fast_timer.timeout.connect(self._poll_background_results)
        self.fast_timer.start()
        self._start_pair_loading()

    # -- new run ---------------------------------------------------------------

    def _build_new_run_form(self):
        """The minimal "start a ranking run" form (F04).

        It reuses the pipeline's own pair resolution: the run uses the newest
        usable 2D+3D pair, and a missing end is explained here instead of
        after the GPU has been reserved.
        """
        QtWidgets = self.QtWidgets
        group = QtWidgets.QGroupBox('新建病例排序')
        layout = QtWidgets.QGridLayout(group)
        layout.setContentsMargins(16, 14, 16, 14)
        remembered = read_json(
            Path.home() / ".mimics_script" / "flexict_settings.json", {}) or {}
        self.pool_edit = QtWidgets.QLineEdit()
        self.pool_edit.setText(
            str(self.context.get("dataset_root")
                or remembered.get("dataset_root") or "")
        )
        self.pool_edit.setPlaceholderText('未标注数据池（数据集文件夹，每例一个子目录）')
        layout.addWidget(QtWidgets.QLabel('数据池'), 0, 0)
        layout.addWidget(self.pool_edit, 0, 1)
        self.pool_browse_button = QtWidgets.QPushButton('浏览...')
        self.pool_browse_button.clicked.connect(self._browse_pool)
        layout.addWidget(self.pool_browse_button, 0, 2)
        self.pair_label = QtWidgets.QLabel('正在后台检查 2D+3D 模型对...')
        self.pair_label.setWordWrap(True)
        layout.addWidget(QtWidgets.QLabel('模型对'), 1, 0)
        layout.addWidget(self.pair_label, 1, 1, 1, 2)
        self.start_run_button = QtWidgets.QPushButton('开始排序')
        self.start_run_button.setObjectName('primary')
        self.start_run_button.setEnabled(False)
        self.start_run_button.clicked.connect(self._submit_new_run)
        layout.addWidget(self.start_run_button, 2, 2)
        layout.setColumnStretch(1, 1)
        return group

    def _browse_pool(self):
        choose_existing_directory_async(
            self.QtCore,
            self.QtWidgets,
            self.window,
            '选择未标注数据池',
            self.pool_edit.text().strip(),
            lambda value: self.pool_edit.setText(str(value)) if value else None,
            button=self.pool_browse_button,
        )

    def _start_pair_loading(self):
        from flexict_common import load_pair

        def load():
            try:
                model_2d, model_3d = load_pair(self.workspace or None)
                self._pair_results.put(((model_2d, model_3d), ""))
            except Exception as exc:
                self._pair_results.put((None, str(exc)))

        worker = threading.Thread(target=load, name="flexict-al-pair-check")
        worker.daemon = True
        worker.start()

    def _apply_pair(self, pair, error):
        self._loading_pairs = False
        if error:
            self.pair_label.setStyleSheet("color: #b42318; font-weight: 600;")
            self.pair_label.setText(
                '模型库检查失败: {0}'.format(error))
            return
        model_2d, model_3d = pair or (None, None)
        if model_2d is None or model_3d is None:
            self.pair_label.setStyleSheet("color: #b42318; font-weight: 600;")
            self.pair_label.setText(
                '未找到可用的 2D+3D 模型对。请先在 FlexiCT 训练窗口选择'
                '“2D + 3D 成对模型（主动学习用）”完成训练，再启动排序。'
            )
            return
        label = " + ".join(
            str(row.get("label_name") or "") for row in (model_2d, model_3d)
        ) or str(model_2d.get("task_name") or "")
        self.pair_label.setStyleSheet("")
        self.pair_label.setText(
            '{0}（{1} · {2}）'.format(
                label,
                str(model_2d.get("model_id") or ""),
                str(model_3d.get("model_id") or ""),
            )
        )
        self.start_run_button.setEnabled(True)

    def _poll_background_results(self):
        if self._refresh_pending:
            try:
                jobs = self._refresh_results.get_nowait()
            except Empty:
                pass
            else:
                self._refresh_pending = False
                self._apply_jobs(jobs)
        if self._loading_pairs:
            try:
                pair, error = self._pair_results.get_nowait()
            except Empty:
                pass
            else:
                self._apply_pair(pair, error)
            if self._loading_pairs and time.time() >= self._pair_deadline:
                self._loading_pairs = False
                self.pair_label.setStyleSheet(
                    "color: #b42318; font-weight: 600;")
                self.pair_label.setText(
                    '模型对检查超时。FlexiCT 工作区所在磁盘可能离线或过慢；'
                    '请关闭本窗口、重新连接磁盘后重试。'
                )
        if self._submit_deadline and time.time() >= self._submit_deadline:
            self._submit_deadline = 0.0
            self.start_run_button.setEnabled(True)
            self.pair_label.setStyleSheet("color: #b42318; font-weight: 600;")
            self.pair_label.setText('任务创建超时。请重试。')
        # S08: drain async annotate/skip writes. Success refreshes the row
        # (only then does it show as annotated — UI31); a failure keeps the
        # old row and says so, leaving the click retryable.
        applied_any = False
        first_error = ""
        while True:
            try:
                _case_id, error = self._state_write_results.get_nowait()
            except Empty:
                break
            if error:
                first_error = first_error or error
            else:
                applied_any = True
        if applied_any:
            self._job_selected()
        if first_error:
            self.status_label.setText(friendly_error(first_error))
        try:
            payload = self._submit_results.get_nowait()
        except Empty:
            return
        self._submit_deadline = 0.0
        if isinstance(payload, str):
            self.start_run_button.setEnabled(True)
            self.pair_label.setStyleSheet("color: #b42318; font-weight: 600;")
            self.pair_label.setText(friendly_error(payload))
            return
        job = payload
        self.start_run_button.setEnabled(True)
        # Refresh first, outcome message after — refresh() rewrites the
        # status line for the (still empty) run list (same trap as F14).
        self.refresh()
        self.status_label.setText(
            '主动学习任务 {0} 已启动。进度见「FlexiCT Show Status and Stop」；'
            '完成后本窗口刷新即可看到排序。'.format(job.get("job_id"))
        )

    def _submit_new_run(self):
        try:
            dataset_root = self.pool_edit.text().strip()
            if not dataset_root:
                raise RuntimeError('请先选择未标注数据池文件夹。')
            workspace = self.workspace or None
            self.start_run_button.setEnabled(False)
            self.pair_label.setStyleSheet("")
            self.pair_label.setText('正在后台创建主动学习任务...')
            self._submit_deadline = time.time() + 60.0

            def create():
                try:
                    from flexict_common import load_pair
                    from flexict_pipeline import create_flexict_job

                    model_2d, model_3d = load_pair(workspace)
                    if model_2d is None or model_3d is None:
                        raise RuntimeError(
                            'No usable FlexiCT model pair was found.')
                    request = {
                        "operation": "active_learning",
                        "workspace": workspace or "",
                        "dataset_root": dataset_root,
                        "label_name": str(model_2d.get("label_name") or ""),
                        "task_id": str(model_2d.get("task_id") or ""),
                        "task_name": str(
                            model_2d.get("task_name")
                            or model_2d.get("task_id") or ""),
                        "model_manifest_2d": str(
                            model_2d.get("manifest_path") or ""),
                        "model_manifest_3d": str(
                            model_3d.get("manifest_path") or ""),
                    }
                    self._submit_results.put(create_flexict_job(request))
                except Exception as exc:
                    self._submit_results.put(str(exc))

            worker = threading.Thread(
                target=create, name="flexict-al-job-create")
            worker.daemon = True
            worker.start()
        except Exception as exc:
            self.start_run_button.setEnabled(True)
            self.pair_label.setStyleSheet("color: #b42318; font-weight: 600;")
            self.pair_label.setText(friendly_error(exc))

    # -- new ranking (F04) ----------------------------------------------------

    def _new_ranking_clicked(self, checked=True):
        """Use the same inline form for first and subsequent ranking runs."""
        self.new_button.setChecked(bool(checked))
        self.new_button.setText('收起新建表单' if checked else '新建排序…')
        self._new_run_group.setVisible(bool(checked))
        if checked:
            self.pool_edit.setFocus()
            if self.body_scroll is not None:
                self.body_scroll.ensureWidgetVisible(self._new_run_group)

    def _start_ranking(self, dataset_root: str, label_name: str):
        """Validate + submit the ranking job off the GUI thread (F04).

        Puts the SAME payload shape on _submit_results as _submit_new_run
        (dict on success, str on error): the 80ms fast_timer consumer
        (_poll_background_results) expects exactly that protocol, and a
        mismatched 3-tuple used to crash it with AttributeError while the
        2s consumer swallowed the dict — the ranking result/error was lost
        both ways and the status line stalled forever.
        """
        request = {
            "operation": "active_learning",
            "workspace": self.workspace,
            "dataset_root": dataset_root,
            "label_name": label_name,
        }

        def _submit():
            from flexict_pipeline import create_flexict_job

            try:
                if not Path(dataset_root).is_dir():
                    raise ValueError(
                        "数据集根目录不存在：{}".format(dataset_root))
                self._submit_results.put(create_flexict_job(request))
            except Exception as exc:  # surfaced by the fast poll timer
                self._submit_results.put(
                    "启动排序失败：{}".format(exc)[:400])

        worker = threading.Thread(
            target=_submit, name="flexict-al-submit", daemon=True)
        worker.start()
        self.status_label.setText(
            '正在后台启动主动学习排序（检查路径与模型对）…')


    def _poll_running_job(self):
        """Show live progress of a running ranking and auto-refresh at
        completion. Submit results are consumed by _poll_background_results
        (single protocol, single consumer); this poller only tracks the
        running job itself. The scan itself runs on a worker thread —
        stat-ing every job directory on the GUI thread froze the window
        on network workspaces (R75 P2)."""
        if not self._running_scan_pending:
            self._running_scan_pending = True

            def scan():
                try:
                    result = find_running_al_job(self.workspace or None)
                except Exception:
                    result = None
                self._running_scan_results.put(result)

            worker = threading.Thread(
                target=scan, name="al-running-scan", daemon=True)
            worker.start()
        try:
            running = self._running_scan_results.get_nowait()
        except Empty:
            return
        self._running_scan_pending = False
        if running:
            self._seen_running_job_id = str(running.get("job_id") or "")
            phase = str(running.get("phase") or running.get("status") or "running")
            percent = int(running.get("progress_percent") or 0)
            self.status_label.setText(
                '排序进行中（{}，{}%）——完成后列表会自动刷新。'.format(
                    phase, percent))
            return
        # A job we announced disappeared from the running list: it reached a
        # terminal state — refresh so it appears in the picker.
        if self._seen_running_job_id:
            self._seen_running_job_id = ""
            self.refresh()
            if self._jobs:
                excluded = int(
                    (self._jobs[0] or {}).get("excluded_case_count") or 0)
                text = '排序完成：{0} 例。双击行即可在 Mimics 中打开并叠加。'.format(
                    len(self._rows))
                if excluded:
                    text += '（另有 {0} 例因数据问题被剔除，原因见任务日志）'.format(
                        excluded)
                self.status_label.setText(text)

    # -- data ---------------------------------------------------------------

    def refresh(self):
        self._refresh_async()

    def _refresh_async(self):
        """Scan for completed runs off the GUI thread (铁律 1) and apply
        the result via the poll timer — the same pattern the running-job
        scan already uses."""
        if self._refresh_pending:
            return
        self._refresh_pending = True

        def scan():
            try:
                jobs = find_al_jobs(self.workspace or None)
            except Exception:
                jobs = []
            self._refresh_results.put(jobs)

        worker = threading.Thread(
            target=scan, name="al-jobs-scan", daemon=True)
        worker.start()

    def _apply_jobs(self, jobs):
        self._jobs = list(jobs or [])
        self.job_combo.blockSignals(True)
        self.job_combo.clear()
        for job in self._jobs:
            state = str(job.get("status") or "")
            if state in {"failed", "cancelled", "abandoned"}:
                # R62-10: failed runs stay listed, clearly marked, so the
                # annotator knows a ranking was attempted and why it died.
                self.job_combo.addItem(
                    "{} · {} · {}".format(
                        time.strftime(
                            "%Y-%m-%d %H:%M",
                            time.localtime(
                                float(job.get("created_at_epoch") or 0)),
                        ),
                        {"failed": "失败", "cancelled": "已取消",
                         "abandoned": "已放弃"}.get(state, state),
                        str(job.get("label_name") or ""),
                    )
                )
                continue
            self.job_combo.addItem(
                "{} · {} 例 · {}".format(
                    time.strftime(
                        "%Y-%m-%d %H:%M",
                        time.localtime(float(job.get("created_at_epoch") or 0)),
                    ),
                    len(job.get("ranking") or []) or "-",
                    str(job.get("label_name") or ""),
                )
            )
        self.job_combo.blockSignals(False)
        if self._jobs:
            self.job_combo.setCurrentIndex(0)
            self._job_selected()
        else:
            self._job_dir = ""
            self._rows = []
            self._state = {}
            self.table.setRowCount(0)
            self._set_actions_enabled(False)
            self.status_label.setText(
                '还没有已完成的主动学习运行。点击「新建排序…」，用已训练的 2D+3D 模型对未标注病例池启动第一次排序。'
            )

    def _job_selected(self):
        QtCore, QtWidgets = self.QtCore, self.QtWidgets
        index = self.job_combo.currentIndex()
        if not 0 <= index < len(self._jobs):
            return
        job = self._jobs[index]
        state = str(job.get("status") or "")
        if state in {"failed", "cancelled", "abandoned"}:
            # R62-10: no ranking exists to review — surface the error and
            # disable the row actions instead of an empty table.
            self._job_dir = ""
            self._rows = []
            self._state = {}
            self.table.setRowCount(0)
            self._set_actions_enabled(False)
            self.status_label.setText(
                "该运行{0}，无排序结果。原因：{1}".format(
                    {"failed": "失败",
                     "cancelled": "已取消",
                     "abandoned": "已放弃"}.get(state, state),
                    str(job.get("error") or job.get("message") or "未知"),
                )
            )
            return
        job_dir = str(job.get("job_dir") or "")
        if not job_dir:
            return
        self._job_dir = job_dir
        self._rows = load_ranking(job_dir)
        self._state = load_annotation_state(job_dir)
        self.table.setRowCount(len(self._rows))
        cases = self._state.get("cases") or {}
        for row_index, row in enumerate(self._rows):
            case_id = str(row.get("case") or "")
            entry = cases.get(case_id) or {}
            values = [
                row_index + 1,
                case_id,
                "{:.2f}".format(float(row.get("integrated") or 0.0)),
                "{:.1f}".format(float(row.get("uncertain_vol") or 0.0)),
                case_state_label(entry.get("state") or "new"),
                self._case_image_hint(case_id),
            ]
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(value))
                if column in (0, 2, 3):
                    item.setTextAlignment(
                        QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                if str(entry.get("state") or "new") == "annotated":
                    item.setForeground(self.QtGui.QBrush(
                        self.QtGui.QColor(PALETTE["success"])))
                self.table.setItem(row_index, column, item)
        self._set_actions_enabled(bool(self._rows))
        self._refresh_status_text()

    def _refresh_status_text(self):
        """Summary line for the current job (F14: callers that have a more
        important outcome message set it AFTER this, so it is not wiped)."""
        cases = self._state.get("cases") or {}
        annotated = sum(
            1 for entry in cases.values()
            if str(entry.get("state")) == "annotated")
        viewed = sum(
            1 for entry in cases.values()
            if str(entry.get("state")) == "viewed")
        # UI14: 已查看与已标注分开统计——查看辅助带不计入完成数，只有
        # 显式确认（标记为已标注）才算。
        text = (
            "已排序 {0} 例 · 已查看 {1} 例 · 已标注 {2} 例。双击一行可在 Mimics "
            "中打开该病例并叠加不确定度带。".format(
                len(self._rows), viewed, annotated)
        )
        if self._jobs and bands_degenerate(
                self._jobs[self.job_combo.currentIndex()]):
            text += (" 注意：双模型对的分歧图只有一级，中带与高带覆盖"
                     "相同的体素。")
        # R81 复审: cases the uncertainty batch excluded (grid mismatch,
        # missing mask) silently shortened the ranking — say so and point at
        # the log holding the per-case Chinese reasons.
        job = (self._jobs[self.job_combo.currentIndex()]
               if self._jobs else {})
        excluded = int(job.get("excluded_case_count") or 0)
        if excluded > 0:
            log_path = str(job.get("uncertainty_log") or "")
            text += " 注意：有 {0} 例因数据问题被剔除出排序（如网格不一致），原因见任务日志{1}。".format(
                excluded, "：{0}".format(log_path) if log_path else "")
        self.status_label.setText(text)

    def _case_image_hint(self, case_id: str) -> str:
        if not self._job_dir:
            return ""
        path = Path(self._job_dir) / "input" / "{}.nii.gz".format(case_id)
        return str(path) if path.is_file() else ""

    def _selected_case(self) -> str:
        row = self.table.currentRow()
        if not 0 <= row < len(self._rows):
            return ""
        return str(self._rows[row].get("case") or "")

    def _set_actions_enabled(self, enabled: bool):
        self.export_button.setEnabled(enabled)
        selected = bool(enabled and self.table.selectionModel().hasSelection() and self._selected_case())
        for button in (self.open_button, self.overlay_button,
                       self.consensus_button, self.annotated_button,
                       self.skip_button):
            button.setEnabled(selected)

    # -- actions --------------------------------------------------------------

    def _open_and_overlay_selected(self):
        self._apply_to_mimics("open_bands")

    def _overlay_selected(self):
        self._apply_to_mimics("bands")

    def _apply_consensus(self):
        self._apply_to_mimics("consensus")

    def _poll_requests(self):
        """Read back apply-request outcomes the Mimics monitor finished."""
        # F04: live progress for a just-started ranking and submit-result
        # draining run on every tick, selected job or not.
        self._poll_running_job()
        if not self._job_dir:
            return
        updates = request_updates(self._job_dir, self._request_poll_epoch)
        if not updates:
            # P2-2: "本窗口会自动更新" is only true while the Mimics-side
            # monitor runs. If the oldest pending request has been waiting a
            # long time, say so instead of letting that promise sit forever.
            age = pending_request_age(self._job_dir, time.time())
            if age > 30.0 and "可能尚未运行" not in (
                self.status_label.text()
            ):
                self.status_label.setText(
                    "请求已等待 {0:.0f} 秒仍无结果——Mimics 中的主动学习"
                    "监控（03 Active Learning Review）可能尚未运行。".format(
                        age)
                )
            return
        self._request_poll_epoch = max(
            row["updated_at_epoch"] for row in updates)
        self._state = load_annotation_state(self._job_dir)
        latest = updates[-1]
        # F14: refresh the table first, then set the outcome text — the
        # old order let the job summary overwrite the failure reason the
        # annotator needed to see.
        self._job_selected()
        if latest["state"] == "applied":
            self.status_label.setText(
                "已在 Mimics 中应用：{0}（{1}）。表格已是最新。".format(
                    latest["case"], latest["detail"][:160]))
        else:
            self.status_label.setText(
                "请求失败（{0}，{1}）：{2}".format(
                    latest["case"], latest["what"],
                    latest["detail"][:200]))

    def _apply_to_mimics(self, what: str):
        """Request an in-Mimics application through the runtime monitor.

        Writes a pending request file the Mimics-side module polls, then
        reports how to finish in Mimics. This keeps every Mimics API call in
        the foreground Mimics process (Py3.5 runtime), never here.
        """
        case_id = self._selected_case()
        if not case_id:
            return
        request = {
            "schema_version": "flexict_al_apply_request.v1",
            "job_dir": self._job_dir,
            "case_id": case_id,
            "what": what,
            "requested_at_epoch": time.time(),
        }
        request_dir = Path(self._job_dir) / "apply_requests"
        request_dir.mkdir(parents=True, exist_ok=True)
        name = "{}_{}_{}.json".format(
            case_id, what, int(time.time() * 1000))
        write_json_atomic(request_dir / name, request)
        labels = {
            "bands": "在当前病例上叠加不确定度带",
            "consensus": "在当前病例上应用共识 Mask",
            "open": "该病例的工程",
            "open_bands": "该病例的工程及其不确定度带",
        }
        self.status_label.setText(
            "已为 {1} 请求：{0}。它通过 FlexiCT 菜单的主动学习监控"
            "（03 Active Learning Review）在 Mimics 中执行；本窗口会自动更新。".format(
                labels.get(what, what), case_id,
            )
        )

    def _set_case_state_async(self, case_id: str, value: str, verb: str):
        """S08: the state write can block on the cross-process guard for up
        to its timeout while Mimics holds the lock — running it directly in
        the click handler froze the window for those seconds. Write on a
        worker thread (the guard serializes concurrent writers); the poll
        timer applies the outcome and refreshes the row."""
        job_dir = self._job_dir

        def write():
            try:
                set_case_state(job_dir, case_id, value)
                self._state_write_results.put((case_id, None))
            except Exception as exc:
                self._state_write_results.put((case_id, str(exc)))

        worker = threading.Thread(
            target=write, name="al-state-write", daemon=True)
        worker.start()
        self.status_label.setText('正在{0} {1}…'.format(verb, case_id))

    def _mark_annotated(self):
        case_id = self._selected_case()
        if case_id and self._job_dir:
            self._set_case_state_async(case_id, "annotated", "标记已标注")

    def _mark_skipped(self):
        case_id = self._selected_case()
        if case_id and self._job_dir:
            self._set_case_state_async(case_id, "skipped", "标记跳过")

    def _export_csv(self):
        if not self._rows:
            return
        default = str(Path(self._job_dir) / "annotation_progress.csv")
        choose_save_file_async(
            self.QtCore,
            self.QtWidgets,
            self.window,
            "导出标注进度",
            default,
            "CSV 文件 (*.csv)",
            self._write_csv,
        )

    def _write_csv(self, path):
        if not path:
            return
        cases = self._state.get("cases") or {}
        with open(path, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["rank", "case", "integrated", "uncertain_vol", "max", "state"])
            for index, row in enumerate(self._rows, start=1):
                case_id = str(row.get("case") or "")
                writer.writerow([
                    index,
                    case_id,
                    "{:.4f}".format(float(row.get("integrated") or 0.0)),
                    "{:.4f}".format(float(row.get("uncertain_vol") or 0.0)),
                    "{:.4f}".format(float(row.get("max") or 0.0)),
                    str((cases.get(case_id) or {}).get("state") or "new"),
                ])
        self.status_label.setText("已导出到 {0}".format(path))

    def show(self):
        self.window.show()
        self.window.raise_()
        self.window.activateWindow()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", required=True)
    args = parser.parse_args()
    context = read_json(Path(args.context).expanduser().resolve(), {}) or {}
    try:
        from PySide6 import QtCore, QtGui, QtWidgets

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        configure_application(app, 'FlexiCT 主动学习')
        app.setStyleSheet(stylesheet())
        window = ActiveLearningWindow(context, (QtCore, QtGui, QtWidgets))
        window.show()
        return int(app.exec())
    except Exception as exc:
        if context.get("setup_status_path"):
            from nnunet_common import update_status

            update_status(
                context["setup_status_path"],
                status="failed",
                phase="ui_failed",
                error="{}: {}".format(type(exc).__name__, exc),
                traceback=traceback.format_exc(),
            )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
