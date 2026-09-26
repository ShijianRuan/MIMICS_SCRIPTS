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
import time
import traceback
from pathlib import Path

from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "tools"):
    value = str(candidate)
    if value not in sys.path:
        sys.path.insert(0, value)

from flexict_common import TERMINAL_STATES  # noqa: E402
from nnunet_common import read_json, write_json_atomic  # noqa: E402
from ui_theme import configure_application, stylesheet  # noqa: E402


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
        self.window.setWindowTitle("FlexiCT Active Learning")
        self.window.resize(980, 640)
        self.window.setMinimumSize(820, 520)
        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(22, 18, 22, 18)
        root.setSpacing(12)
        title = QtWidgets.QLabel("FlexiCT Active Learning")
        title.setObjectName("title")
        subtitle = QtWidgets.QLabel(
            "Cases ranked by 2D-vs-3D disagreement: annotate the top of the "
            "list first — that is where the two models disagree most. "
            "Double-click opens the case in Mimics with its uncertainty "
            "bands applied; the state updates here as you go."
        )
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(subtitle)

        picker = QtWidgets.QHBoxLayout()
        picker.addWidget(QtWidgets.QLabel("Run"))
        self.job_combo = QtWidgets.QComboBox()
        self.job_combo.currentIndexChanged.connect(self._job_selected)
        picker.addWidget(self.job_combo, 1)
        refresh = QtWidgets.QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        picker.addWidget(refresh)
        root.addLayout(picker)

        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Rank", "Case", "Score", "Uncertain volume (mm³)", "Status", "Path"]
        )
        self.table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(
            5, QtWidgets.QHeaderView.Stretch)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.itemDoubleClicked.connect(self._open_and_overlay_selected)
        root.addWidget(self.table, 1)

        self.status_label = QtWidgets.QLabel("Select a completed active-learning run.")
        self.status_label.setObjectName("hint")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        actions = QtWidgets.QHBoxLayout()
        self.open_button = QtWidgets.QPushButton("Open + Overlay")
        self.open_button.setObjectName("primary")
        self.open_button.clicked.connect(self._open_and_overlay_selected)
        actions.addWidget(self.open_button)
        self.overlay_button = QtWidgets.QPushButton("Overlay Uncertainty")
        self.overlay_button.clicked.connect(self._overlay_selected)
        actions.addWidget(self.overlay_button)
        self.consensus_button = QtWidgets.QPushButton("Apply Consensus Mask")
        self.consensus_button.clicked.connect(self._apply_consensus)
        actions.addWidget(self.consensus_button)
        self.annotated_button = QtWidgets.QPushButton("Mark Annotated")
        self.annotated_button.clicked.connect(self._mark_annotated)
        actions.addWidget(self.annotated_button)
        self.skip_button = QtWidgets.QPushButton("Mark Skipped")
        self.skip_button.clicked.connect(self._mark_skipped)
        actions.addWidget(self.skip_button)
        actions.addStretch(1)
        self.export_button = QtWidgets.QPushButton("Export CSV")
        self.export_button.clicked.connect(self._export_csv)
        actions.addWidget(self.export_button)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.window.close)
        actions.addWidget(close)
        root.addLayout(actions)
        self.window.setCentralWidget(central)
        self._jobs: list[dict[str, Any]] = []
        self._rows: list[dict[str, Any]] = []
        self._state: dict[str, Any] = {}
        self._job_dir = ""
        self._request_poll_epoch = time.time()
        self.refresh()
        # The Mimics-side monitor reports outcomes by rewriting request
        # files; poll them so the table and status line follow along
        # without the annotator switching windows.
        self.poll_timer = QtCore.QTimer(self.window)
        self.poll_timer.setInterval(2000)
        self.poll_timer.timeout.connect(self._poll_requests)
        self.poll_timer.start()

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
                "No completed active-learning run found. Start one from "
                "02_AI > FlexiCT > 03 Active Learning Review with a trained "
                "2D+3D pair."
            )

    def _job_selected(self):
        index = self.job_combo.currentIndex()
        if not 0 <= index < len(self._jobs):
            return
        status = self._jobs[index]
        job_dir = str(status.get("job_dir") or "")
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
                    from PySide6 import QtGui

                    item.setForeground(QtGui.QBrush(QtGui.QColor("#49aa55")))
                self.table.setItem(row_index, column, item)
        self._set_actions_enabled(bool(self._rows))
        annotated = sum(
            1 for entry in cases.values()
            if str(entry.get("state")) == "annotated")
        text = (
            "{} case(s) ranked · {} annotated. Double-click a row to open "
            "its case in Mimics with the uncertainty bands applied.".format(
                len(self._rows), annotated)
        )
        if bands_degenerate(status):
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
        if not self._job_dir:
            return
        updates = request_updates(self._job_dir, self._request_poll_epoch)
        if not updates:
            return
        self._request_poll_epoch = max(
            row["updated_at_epoch"] for row in updates)
        self._state = load_annotation_state(self._job_dir)
        latest = updates[-1]
        if latest["state"] == "applied":
            self.status_label.setText(
                "Applied in Mimics: {} ({}). The table is up to date.".format(
                    latest["case"], latest["detail"][:160]))
        else:
            self.status_label.setText(
                "Request failed for {} ({}): {}".format(
                    latest["case"], latest["what"],
                    latest["detail"][:200]))
        self._job_selected()

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
        configure_application(app, "FlexiCT Active Learning")
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
