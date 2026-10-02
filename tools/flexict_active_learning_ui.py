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
import sys
import threading
import time
import traceback
from pathlib import Path
from queue import Queue

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
    configure_application,
    stylesheet,
)


STATE_SCHEMA = "flexict_annotation_state.v1"


def find_al_jobs(workspace: str | None) -> list[dict[str, Any]]:
    """Completed active_learning jobs, newest first."""
    from flexict_pipeline import list_flexict_jobs

    rows = []
    for status in list_flexict_jobs(workspace):
        if str(status.get("kind") or "") != "active_learning":
            continue
        if str(status.get("status") or "") not in TERMINAL_STATES:
            continue
        if str(status.get("status") or "") in {"failed", "cancelled", "abandoned"}:
            continue
        rows.append(status)
    rows.sort(key=lambda row: float(row.get("created_at_epoch") or 0), reverse=True)
    return rows


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
    csv_path = Path(job_dir) / "uncertainty" / "ranking.csv"
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


def set_case_state(job_dir: str | Path, case_id: str, value: str) -> None:
    state = load_annotation_state(job_dir)
    state["cases"][str(case_id)] = {
        "state": str(value),
        "updated_at_epoch": time.time(),
    }
    save_annotation_state(job_dir, state)


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
        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(22, 18, 22, 18)
        root.setSpacing(12)
        title = QtWidgets.QLabel('FlexiCT 主动学习')
        title.setObjectName("title")
        subtitle = QtWidgets.QLabel(
            '病例按 2D 与 3D 模型的分歧排序：优先标注列表最前面的病例——那是两个模型分歧最大的地方。双击在 Mimics 中打开病例并叠加不确定度条带；标注状态会实时更新到这里。'
        )
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(subtitle)

        picker = QtWidgets.QHBoxLayout()
        picker.addWidget(QtWidgets.QLabel('运行'))
        self.job_combo = QtWidgets.QComboBox()
        self.job_combo.currentIndexChanged.connect(self._job_selected)
        picker.addWidget(self.job_combo, 1)
        refresh = QtWidgets.QPushButton('刷新')
        refresh.clicked.connect(self.refresh)
        picker.addWidget(refresh)
        root.addLayout(picker)

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

        actions = QtWidgets.QHBoxLayout()
        # F04: the first ranking run had no reachable graphical entry — the
        # empty state told the annotator to use this very window, which only
        # lists completed runs. A "new ranking" action lives here now, so
        # first-run is possible from the same menu entry.
        self.new_button = QtWidgets.QPushButton('新建排序…')
        self.new_button.setObjectName("primary")
        self.new_button.clicked.connect(self._new_ranking_clicked)
        actions.addWidget(self.new_button)
        self.open_button = QtWidgets.QPushButton('打开并叠加')
        self.open_button.setObjectName("primary")
        self.open_button.clicked.connect(self._open_and_overlay_selected)
        actions.addWidget(self.open_button)
        self.overlay_button = QtWidgets.QPushButton('叠加不确定度')
        self.overlay_button.clicked.connect(self._overlay_selected)
        actions.addWidget(self.overlay_button)
        self.consensus_button = QtWidgets.QPushButton('应用共识 Mask')
        self.consensus_button.clicked.connect(self._apply_consensus)
        actions.addWidget(self.consensus_button)
        self.annotated_button = QtWidgets.QPushButton('标记为已标注')
        self.annotated_button.clicked.connect(self._mark_annotated)
        actions.addWidget(self.annotated_button)
        self.skip_button = QtWidgets.QPushButton('标记为跳过')
        self.skip_button.clicked.connect(self._mark_skipped)
        actions.addWidget(self.skip_button)
        actions.addStretch(1)
        self.export_button = QtWidgets.QPushButton('导出 CSV')
        self.export_button.clicked.connect(self._export_csv)
        actions.addWidget(self.export_button)
        close = QtWidgets.QPushButton('关闭')
        close.clicked.connect(self.window.close)
        actions.addWidget(close)
        root.addLayout(actions)
        self.window.setCentralWidget(central)
        self._jobs: list[dict[str, Any]] = []
        self._rows: list[dict[str, Any]] = []
        self._state: dict[str, Any] = {}
        self._job_dir = ""
        self._request_poll_epoch = time.time()
        # F04: background submit results for a new ranking (Queue because
        # create_flexict_job must never run on the GUI thread).
        self._submit_results: Queue = Queue()
        self._seen_running_job_id = ""
        self.refresh()
        # The Mimics-side monitor reports outcomes by rewriting request
        # files; poll them so the table and status line follow along
        # without the annotator switching windows.
        self.poll_timer = QtCore.QTimer(self.window)
        self.poll_timer.setInterval(2000)
        self.poll_timer.timeout.connect(self._poll_requests)
        self.poll_timer.start()

    # -- new ranking (F04) ----------------------------------------------------

    def _new_ranking_clicked(self):
        """Minimal first-run form: dataset pool + target, submit via the
        existing flexict pipeline in a background thread."""
        QtWidgets = self.QtWidgets
        remembered = read_json(
            Path.home() / ".mimics_script" / "flexict_settings.json", {}) or {}
        dialog = QtWidgets.QDialog(self.window)
        dialog.setWindowTitle('新建主动学习排序')
        form = QtWidgets.QFormLayout(dialog)
        dataset_edit = QtWidgets.QLineEdit(
            str(remembered.get("dataset_root") or ""))
        dataset_edit.setPlaceholderText('未标注病例池的根目录')
        dataset_row = QtWidgets.QHBoxLayout()
        dataset_row.addWidget(dataset_edit, 1)
        browse = QtWidgets.QPushButton('浏览…')
        dataset_row.addWidget(browse)
        wrap = QtWidgets.QWidget()
        wrap.setLayout(dataset_row)
        form.addRow('数据集根目录', wrap)
        label_edit = QtWidgets.QLineEdit(str(remembered.get("label_name") or ""))
        label_edit.setPlaceholderText('例如 kidney_left')
        form.addRow('目标 (label)', label_edit)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.button(QtWidgets.QDialogButtonBox.Ok).setText('启动排序')
        buttons.button(QtWidgets.QDialogButtonBox.Cancel).setText('取消')
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        def _browse():
            # Native dialogs can hang on disconnected drives / SMB shares;
            # the async helper isolates the pick in its own process so this
            # window (and Mimics) stays responsive (铁律 1).
            choose_existing_directory_async(
                self.QtCore, QtWidgets, dialog, '选择数据集根目录',
                dataset_edit.text().strip() or str(Path.home()),
                lambda value: value and dataset_edit.setText(value),
                button=browse)

        browse.clicked.connect(_browse)
        dialog.resize(520, dialog.sizeHint().height())
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        dataset_root = dataset_edit.text().strip()
        label_name = label_edit.text().strip()
        if not dataset_root or not label_name:
            self.status_label.setText(
                '启动排序需要数据集根目录和目标名称。')
            return
        self._start_ranking(dataset_root, label_name)

    def _start_ranking(self, dataset_root: str, label_name: str):
        """Validate + submit the ranking job off the GUI thread (F04)."""
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
                job = create_flexict_job(request)
                self._submit_results.put(
                    (job.get("job_id") or "", job.get("job_dir") or "", ""))
            except Exception as exc:  # surfaced by the poll timer
                self._submit_results.put(("", "", "{}: {}".format(
                    type(exc).__name__, exc)))

        worker = threading.Thread(
            target=_submit, name="flexict-al-submit", daemon=True)
        worker.start()
        self.status_label.setText(
            '正在后台启动主动学习排序（检查路径与模型对）…')

    def _poll_running_job(self):
        """Show live progress of a running ranking; auto-refresh at completion
        and surface submit errors (F04)."""
        # Drain submit results first: an error message must survive.
        try:
            while True:
                job_id, job_dir, error = self._submit_results.get_nowait()
                if error:
                    self.status_label.setText(
                        '启动排序失败：{}'.format(error[:300]))
                    return
                self._seen_running_job_id = job_id
        except Exception:
            pass
        running = find_running_al_job(self.workspace or None)
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
                self.status_label.setText(
                    '排序完成：{} case(s)。双击行即可在 Mimics 中打开并叠加。'.format(
                        len(self._rows)))

    # -- data ---------------------------------------------------------------

    def refresh(self):
        self._jobs = find_al_jobs(self.workspace or None)
        self.job_combo.blockSignals(True)
        self.job_combo.clear()
        for job in self._jobs:
            self.job_combo.addItem(
                "{} · {} case(s) · {}".format(
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
                '还没有已完成的主动学习运行。点击「新建排序…」，用已训练的 2D+3D 模型对对未标注病例池启动第一次排序。'
            )

    def _job_selected(self):
        QtCore, QtWidgets = self.QtCore, self.QtWidgets
        index = self.job_combo.currentIndex()
        if not 0 <= index < len(self._jobs):
            return
        job_dir = str(self._jobs[index].get("job_dir") or "")
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
                str(entry.get("state") or "new"),
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
        text = (
            "{} case(s) ranked · {} annotated. Double-click a row to open "
            "its case in Mimics with the uncertainty bands applied.".format(
                len(self._rows), annotated)
        )
        if self._jobs and bands_degenerate(
                self._jobs[self.job_combo.currentIndex()]):
            text += (" Note: with a two-model pair the disagreement map has "
                     "one level, so the moderate and high bands cover the "
                     "same voxels.")
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
        for button in (self.open_button, self.overlay_button,
                       self.consensus_button, self.annotated_button,
                       self.skip_button, self.export_button):
            button.setEnabled(enabled)

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
            return
        updates = request_updates(self._job_dir, self._request_poll_epoch)
        if not updates:
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
                "Applied in Mimics: {} ({}). The table is up to date.".format(
                    latest["case"], latest["detail"][:160]))
        else:
            self.status_label.setText(
                "Request failed for {} ({}): {}".format(
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
            "bands": "uncertainty bands on the open case",
            "consensus": "consensus mask on the open case",
            "open": "the case's project",
            "open_bands": "the case's project and its uncertainty bands",
        }
        self.status_label.setText(
            "Requested {} for {}. It runs in Mimics via the FlexiCT menu's "
            "active-learning monitor (03 Active Learning Review); this "
            "window updates automatically.".format(
                labels.get(what, what), case_id,
            )
        )

    def _mark_annotated(self):
        case_id = self._selected_case()
        if case_id and self._job_dir:
            set_case_state(self._job_dir, case_id, "annotated")
            self._job_selected()

    def _mark_skipped(self):
        case_id = self._selected_case()
        if case_id and self._job_dir:
            set_case_state(self._job_dir, case_id, "skipped")
            self._job_selected()

    def _export_csv(self):
        if not self._rows:
            return
        QtWidgets = self.QtWidgets
        default = str(Path(self._job_dir) / "annotation_progress.csv")
        path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self.window, "Export Annotation Progress", default,
            "CSV files (*.csv)")
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
        self.status_label.setText("Exported to {}".format(path))

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
