#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""System health overview panel (external PySide6 window).

Aggregates, in one page, everything an annotator needs to answer "is
anything stuck, and what can I do about it":

1. Owned processes - from the Phase B process registry
   (resource_locks.snapshot_processes), with liveness and the locks held;
2. Resource locks - live vs stale (owner PID dead or absent);
3. Import/export queues - per output-folder queue state
   (_mcs_queue_active.json / _mcs_queue_stop.json) and recent job runs;
4. nnInteractive server - .nninteractive_server.json state when present;
5. Environment guidance - setup/migration/weights issues detected by
   env_guidance.collect_issues, with a jump into the guidance dialog.

Every problem row carries an action: Stop All Owned Services (the existing
mimics_batch_cli kill-background, which publishes stop markers first and
only then kills), Sweep Stale State (resource_locks.sweep_processes), or
opening the relevant folder. Nothing here mutates state without the user
clicking a clearly-labeled button.

Run inside Mimics via 99_Admin/06_System_Health.py, or standalone:
    python tools/system_health_panel.py [--preview out.png]
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _candidate in (_HERE, _ROOT):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from ui_theme import configure_application, stylesheet as shared_stylesheet  # noqa: E402

import resource_locks  # noqa: E402

import env_guidance  # noqa: E402

REFRESH_SECONDS = 10


# ---------------------------------------------------------------------------
# Read-only aggregation (no Qt imports - unit-testable headless)
# ---------------------------------------------------------------------------


def read_json(path, default=None):
    import json
    try:
        with open(str(path), "r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value
    except Exception:
        return default


def collect_health(project_root):
    """Build the full health snapshot dict. Pure reads; safe to call often."""
    root = Path(project_root)
    snapshot = {
        "generated_at_epoch": time.time(),
        "processes": collect_processes(root),
        "locks": collect_locks(root),
        "queues": collect_queues(root),
        "server": collect_server(root),
        "environment": env_guidance.collect_issues(root),
    }
    return snapshot


def collect_processes(project_root):
    """Registry snapshot plus the locks each live process holds."""
    records = resource_locks.snapshot_processes(project_root, include_dead=True)
    lock_dir = resource_locks.default_resource_lock_dir(project_root)
    lock_owners = {}
    for name, payload in _iter_locks(lock_dir):
        pid = payload.get("pid")
        if pid:
            lock_owners.setdefault(int(pid), []).append(name)
    entries = []
    for record in records:
        pid = record.get("pid")
        entries.append({
            "role": record.get("role") or "?",
            "pid": pid,
            "live": bool(record.get("_live")),
            "cleanup_policy": record.get("cleanup_policy") or "",
            "state_path": record.get("state_path") or "",
            "locks": lock_owners.get(int(pid), []) if pid else [],
        })
    return entries


def _iter_locks(lock_dir):
    """Yield (name, payload) for each held lock.

    Only real ``.lock`` JSON files count: the persistent one-byte
    ``.lock.guard`` files are OS-lock anchors, not locks (a released lock
    unlinks its JSON; the guard deliberately stays).
    """
    try:
        names = sorted(os.listdir(str(lock_dir)))
    except OSError:
        return
    for name in names:
        if not name.endswith(".lock"):
            continue
        payload = read_json(Path(lock_dir) / name, None)
        if isinstance(payload, dict):
            yield name, payload


def collect_locks(project_root):
    """Lock files classified live/stale by owner PID liveness."""
    lock_dir = resource_locks.default_resource_lock_dir(project_root)
    entries = []
    for name, payload in _iter_locks(lock_dir):
        pid = payload.get("pid")
        live = bool(pid) and resource_locks.process_exists(pid)
        entries.append({
            "name": name,
            "owner": payload.get("owner") or payload.get("kind") or "?",
            "pid": pid,
            "live": live,
            "stale": not live,
        })
    return entries


def collect_queues(project_root):
    """Import queues under the runtime base: active flag, stop marker, depth."""
    root = Path(project_root)
    base = root / ".mimics_runtime" / "import_queues"
    runtime_base = read_json(root / "mimics_io_config.json", {}) or {}
    # The queue base is per output dir; both default and configured roots are
    # covered by scanning every mcs_output_* folder under import_queues.
    entries = []
    try:
        names = sorted(os.listdir(str(base)))
    except OSError:
        return entries
    for name in names:
        queue_dir = base / name
        if not queue_dir.is_dir():
            continue
        active = (queue_dir / "_mcs_queue_active.json").is_file()
        stopping = (queue_dir / "_mcs_queue_stop.json").is_file()
        prepared = 0
        prepared_dir = queue_dir / "prepared_queue"
        try:
            if prepared_dir.is_dir():
                prepared = sum(
                    1 for item in os.listdir(str(prepared_dir))
                    if item.endswith(".json")
                )
        except OSError:
            pass
        state = "idle"
        if active and not stopping:
            state = "running"
        elif active and stopping:
            state = "stopping"
        elif stopping:
            state = "stop requested"
        entries.append({
            "name": name,
            "state": state,
            "prepared_cases": prepared,
            "path": str(queue_dir),
        })
    return entries


def collect_server(project_root):
    """Newest .nninteractive_server.json under the project, if any."""
    root = Path(project_root)
    candidates = []
    for pattern in (".nninteractive_server.json", "*/.nninteractive_server.json"):
        candidates.extend(root.glob(pattern))
    newest = None
    newest_mtime = -1.0
    for path in candidates:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime > newest_mtime:
            newest_mtime = mtime
            newest = path
    if newest is None:
        return None
    payload = read_json(newest, {}) or {}
    pid = payload.get("pid")
    return {
        "path": str(newest),
        "pid": pid,
        "live": bool(pid) and resource_locks.process_exists(pid),
        "model_dir": payload.get("model_dir") or "",
        "port": payload.get("port") or "",
        "state": payload.get("state") or "",
    }


def external_python():
    candidates = [
        os.path.join(_ROOT, "python_env", "python.exe"),
        os.path.join(_ROOT, "nninteractive_env", "python.exe"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return sys.executable


def stop_all_owned_services(project_root):
    """Delegate to the existing, battle-tested kill-background path."""
    command = [
        external_python(),
        os.path.join(_HERE, "mimics_batch_cli.py"),
        "kill-background",
    ]
    subprocess.Popen(
        command,
        cwd=str(project_root),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def sweep_stale_state(project_root):
    """One registry sweep: clear dead records, kill orphans, release locks."""
    return resource_locks.sweep_processes(project_root)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


class _Section:
    """One titled block of the health page."""

    def __init__(self, parent_layout, title):
        from PySide6 import QtWidgets
        self.frame = QtWidgets.QFrame()
        self.frame.setObjectName("healthSection")
        layout = QtWidgets.QVBoxLayout(self.frame)
        header = QtWidgets.QLabel(title)
        header.setObjectName("sectionHeader")
        layout.addWidget(header)
        self.body = QtWidgets.QVBoxLayout()
        layout.addLayout(self.body)
        parent_layout.addWidget(self.frame)

    def clear(self):
        while self.body.count():
            item = self.body.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            nested = item.layout()
            if nested is not None:
                while nested.count():
                    nested_item = nested.takeAt(0)
                    if nested_item.widget() is not None:
                        nested_item.widget().deleteLater()


def _row(parent, columns, stretch_last=True):
    from PySide6 import QtWidgets
    box = QtWidgets.QHBoxLayout()
    for index, text in enumerate(columns):
        label = QtWidgets.QLabel(str(text))
        label.setWordWrap(True)
        box.addWidget(label, 1 if (stretch_last and index == len(columns) - 1) else 0)
    parent.addLayout(box)
    return box


def _open_folder(path):
    if not path or not os.path.isdir(path):
        return
    os.startfile(path)  # noqa: S606 - Windows shell open, user-initiated


def run(preview_path=""):
    from PySide6 import QtCore, QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    configure_application(app)
    app.setStyleSheet(shared_stylesheet("""
    QFrame#healthSection {
        background: #ffffff;
        border: 1px solid #dbe2ea;
        border-radius: 8px;
    }
    QLabel#sectionHeader {
        font-weight: 600;
        color: #0f766e;
    }
    QLabel[status="ok"] { color: #15803d; }
    QLabel[status="warn"] { color: #b45309; }
    QLabel[status="bad"] { color: #b91c1c; }
    """))

    window = QtWidgets.QMainWindow()
    window.setWindowTitle("System Health")
    window.setMinimumSize(680, 560)

    central = QtWidgets.QWidget()
    window.setCentralWidget(central)
    root = QtWidgets.QVBoxLayout(central)

    title = QtWidgets.QLabel("System Health")
    title.setObjectName("title")
    root.addWidget(title)

    summary = QtWidgets.QLabel("Loading...")
    summary.setObjectName("subtitle")
    summary.setWordWrap(True)
    root.addWidget(summary)

    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    inner = QtWidgets.QWidget()
    scroll.setWidget(inner)
    page = QtWidgets.QVBoxLayout(inner)
    root.addWidget(scroll, 1)

    section_processes = _Section(page, "Owned Processes")
    section_locks = _Section(page, "Resource Locks")
    section_queues = _Section(page, "Import Queues")
    section_server = _Section(page, "nnInteractive Server")
    section_environment = _Section(page, "Environment")
    page.addStretch(1)

    actions = QtWidgets.QHBoxLayout()
    refresh_btn = QtWidgets.QPushButton("Refresh Now")
    sweep_btn = QtWidgets.QPushButton("Sweep Stale State")
    sweep_btn.setObjectName("primary")
    stop_btn = QtWidgets.QPushButton("Stop All Owned Services")
    actions.addWidget(refresh_btn)
    actions.addWidget(sweep_btn)
    actions.addWidget(stop_btn)
    actions.addStretch(1)
    root.addLayout(actions)

    def render(snapshot):
        live = [p for p in snapshot["processes"] if p["live"]]
        dead = [p for p in snapshot["processes"] if not p["live"]]
        stale_locks = [lk for lk in snapshot["locks"] if lk["stale"]]
        busy_queues = [q for q in snapshot["queues"] if q["state"] != "idle"]
        server = snapshot["server"]

        if not live and not stale_locks and not busy_queues:
            summary_text = "All clear - no live owned processes, no stale locks."
        else:
            summary_text = "{0} live process(es), {1} dead record(s), {2} stale lock(s), {3} active queue(s).".format(
                len(live), len(dead), len(stale_locks), len(busy_queues)
            )
        if server and server["live"]:
            summary_text += " nnInteractive server is running."
        env_issues = snapshot.get("environment") or []
        if env_issues:
            summary_text += " {0} environment issue(s).".format(len(env_issues))
        summary.setText(summary_text)

        _render_processes(section_processes, snapshot["processes"])
        _render_locks(section_locks, snapshot["locks"])
        _render_queues(section_queues, snapshot["queues"])
        _render_server(section_server, server)
        _render_environment(section_environment, env_issues)

    def _render_processes(section, entries):
        section.clear()
        from PySide6 import QtWidgets
        live = [p for p in entries if p["live"]]
        dead = [p for p in entries if not p["live"]]
        if not entries:
            _row(section.body, ["No registered processes."])
            return
        for proc in live:
            box = _row(section.body, [
                "● {0} (PID {1})".format(proc["role"], proc["pid"]),
                "locks: {0}".format(", ".join(proc["locks"]) or "none"),
            ])
            status_label = box.itemAt(0).widget()
            status_label.setProperty("status", "ok")
            if proc["state_path"] and os.path.isfile(proc["state_path"]):
                btn = QtWidgets.QPushButton("Open State")
                btn.clicked.connect(lambda _=False, p=str(proc["state_path"]): _open_folder(os.path.dirname(p)))
                box.addWidget(btn)
        for proc in dead:
            box = _row(section.body, [
                "○ {0} (PID {1}) - dead record, will be cleared on sweep".format(
                    proc["role"], proc["pid"]
                ),
                "",
            ])
            box.itemAt(0).widget().setProperty("status", "warn")

    def _render_locks(section, entries):
        section.clear()
        if not entries:
            _row(section.body, ["No lock files."])
            return
        for lk in entries:
            status = "ok" if lk["live"] else "bad"
            _row_status(section.body, status, {
                "name": "{0} {1} - {2} (PID {3})".format(
                    "●" if lk["live"] else "✕",
                    lk["name"],
                    lk["owner"],
                    lk["pid"] if lk["pid"] is not None else "unknown",
                ),
                "detail": "stale - safe to clear" if lk["stale"] else "held",
            })
        stale = [lk for lk in entries if lk["stale"]]
        if stale:
            _row(section.body, [
                "{0} stale lock(s) - use Sweep Stale State, or they will be "
                "cleared the next time the owner workflow starts.".format(len(stale)),
                "",
            ])

    def _render_queues(section, entries):
        section.clear()
        from PySide6 import QtWidgets
        if not entries:
            _row(section.body, ["No import queue directories."])
            return
        busy = [q for q in entries if q["state"] != "idle"]
        shown = busy or entries
        for q in shown[:20]:
            box = _row(section.body, [
                "{0}: {1}, {2} prepared case(s)".format(
                    q["name"], q["state"], q["prepared_cases"]
                ),
                "",
            ])
            box.itemAt(0).widget().setProperty(
                "status", "warn" if q["state"] != "idle" else "ok"
            )
            btn = QtWidgets.QPushButton("Open Folder")
            btn.clicked.connect(lambda _=False, p=q["path"]: _open_folder(p))
            box.addWidget(btn)
        if len(entries) > 20:
            _row(section.body, ["... and {0} more queue folder(s).".format(len(entries) - 20), ""])

    def _render_server(section, server):
        section.clear()
        if not server:
            _row(section.body, ["No nnInteractive server state file found - the server is not running from here."])
            return
        status = "ok" if server["live"] else "warn"
        _row_status(section.body, status, {
            "name": "Server (PID {0}){1}".format(
                server["pid"],
                " - {0}".format(server["state"]) if server["state"] else "",
            ),
            "detail": "running" if server["live"] else "state file exists but the process is not running",
        })

    def _row_status(parent, status, entry):
        box = _row(parent, [entry["name"], entry["detail"]])
        box.itemAt(0).widget().setProperty("status", status)
        return box

    def _render_environment(section, issues):
        section.clear()
        if not issues:
            box = _row(section.body, [
                "● Environment OK - Python, setup state, paths, weights and the nnInteractive model check out.",
                "",
            ])
            box.itemAt(0).widget().setProperty("status", "ok")
            return
        for issue in issues:
            severity = issue.get("severity") or "warn"
            box = _row_status(section.body, severity, {
                "name": "{0} {1}".format(
                    "✕" if severity == "bad" else "○", issue.get("title") or issue.get("kind")
                ),
                "detail": (issue.get("detail") or "").split("\n")[0],
            })
            btn = QtWidgets.QPushButton("Guidance")
            btn.clicked.connect(lambda _=False: env_guidance.show_dialog(Path(_ROOT)))
            box.addWidget(btn)

    # ---- Non-blocking background work (daemon thread + queue + QTimer) ----
    # collect_health and sweep_stale_state do filesystem/process probing that
    # can take seconds; running them on the GUI thread would freeze the panel.
    import queue as _queue
    import threading as _threading

    _results = _queue.Queue()

    def _poll_results():
        # Drain every completed result without blocking; each entry is
        # (kind, payload). Rendering happens here on the GUI thread.
        while True:
            try:
                kind, payload = _results.get_nowait()
            except _queue.Empty:
                return
            if kind == "snapshot":
                if payload is not None:
                    render(payload)
                summary.setText(summary.text().replace(" (refreshing...)", ""))
            elif kind == "sweep":
                summary.setText(
                    "Swept: removed {0} dead record(s), released {1} lock(s).".format(
                        payload.get("removed_dead_records") or 0,
                        len(payload.get("released_locks") or []),
                    )
                )
                start_refresh()

    _result_timer = QtCore.QTimer(window)
    _result_timer.setInterval(250)
    _result_timer.timeout.connect(_poll_results)
    _result_timer.start()

    def start_refresh():
        text = summary.text()
        if not text.endswith("(refreshing...)"):
            summary.setText(text + " (refreshing...)")

        def worker():
            try:
                _results.put(("snapshot", collect_health(_ROOT)))
            except Exception as exc:  # never leave the panel stuck
                _results.put(
                    ("snapshot", {
                        "processes": [], "locks": [],
                        "queues": [], "server": None,
                        "error": "collect failed: {0}".format(exc),
                    })
                )

        thread = _threading.Thread(target=worker)
        thread.daemon = True
        thread.start()

    def refresh():
        start_refresh()

    refresh_btn.clicked.connect(refresh)

    def do_sweep():
        summary.setText("Sweeping stale state... (the panel stays usable)")
        sweep_btn.setEnabled(False)

        def worker():
            try:
                _results.put(("sweep", sweep_stale_state(_ROOT)))
            except Exception as exc:
                _results.put(
                    ("sweep", {
                        "removed_dead_records": 0,
                        "released_locks": [],
                        "error": str(exc),
                    })
                )

        thread = _threading.Thread(target=worker)
        thread.daemon = True
        thread.start()
        sweep_btn.setEnabled(True)

    sweep_btn.clicked.connect(do_sweep)

    def do_stop():
        answer = QtWidgets.QMessageBox.question(
            window, "Stop All Owned Services",
            "Stop every background process this project started?\n\n"
            "Running imports, exports, and training jobs will be asked to "
            "stop gracefully first, then terminated.",
        )
        if answer == QtWidgets.QMessageBox.Yes:
            summary.setText("Stop requested - processes are shutting down...")

            def worker():
                try:
                    stop_all_owned_services(_ROOT)
                except Exception:
                    pass
                _results.put(("snapshot", None))

            thread = _threading.Thread(target=worker)
            thread.daemon = True
            thread.start()
            QtCore.QTimer.singleShot(4000, refresh)

    stop_btn.clicked.connect(do_stop)

    timer = QtCore.QTimer(window)
    timer.setInterval(int(REFRESH_SECONDS * 1000))
    timer.timeout.connect(refresh)
    timer.start()

    if preview_path:
        window.show()
        app.processEvents()
        window.grab().save(preview_path)
        return 0
    window.show()
    refresh()
    return int(app.exec())


def main(argv):
    preview_path = ""
    index = 1
    while index < len(argv):
        arg = argv[index]
        if arg == "--preview" and index + 1 < len(argv):
            preview_path = argv[index + 1]
            index += 2
            continue
        index += 1
    return run(preview_path)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
