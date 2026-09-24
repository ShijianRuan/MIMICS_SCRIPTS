"""Non-blocking periodic refresh helper for status viewer windows.

list_jobs, model scans, log tails and directory mtime sorts can take
seconds on a network workspace; running them on the GUI thread makes the
window stutter on every refresh tick. This helper runs ``collect()`` on a
daemon worker thread and applies the result on the GUI thread via a queue
drained by a QTimer — the same pattern as tools/system_health_panel.py.

Usage::

    refresher = BackgroundRefresh(
        QtCore, parent=window, interval_ms=2000,
        collect=self._collect, apply=self._apply,
    )
    refresher.request()          # schedule one background refresh
    refresher.refresh_now()      # collect + apply synchronously (tests)
"""

from __future__ import annotations

import queue
import threading


class BackgroundRefresh:
    def __init__(self, qt_core, parent, collect, apply, interval_ms=2000):
        self._collect = collect
        self._apply = apply
        self._results = queue.Queue()
        self._lock = threading.Lock()
        self._pending = False
        self._stop = False
        # Drain timer: runs on the GUI thread, applies finished results.
        self._drain_timer = qt_core.QTimer(parent)
        self._drain_timer.setInterval(250)
        self._drain_timer.timeout.connect(self._drain)
        self._drain_timer.start()
        # Period timer: kicks off background collections.
        self._timer = qt_core.QTimer(parent)
        self._timer.setInterval(int(interval_ms))
        self._timer.timeout.connect(self.request)
        self._timer.start()

    def request(self):
        """Schedule one collection on the worker thread (no-op if busy)."""
        with self._lock:
            if self._pending:
                return
            self._pending = True

        def worker():
            result = None
            try:
                result = self._collect()
            except Exception:  # never leave the viewer stuck on a scan error
                result = None
            finally:
                with self._lock:
                    self._pending = False
                self._results.put(result)

        threading.Thread(target=worker, daemon=True, name="viewer-refresh").start()

    def _drain(self):
        while True:
            try:
                result = self._results.get_nowait()
            except queue.Empty:
                return
            if result is None:
                continue
            try:
                self._apply(result)
            except Exception:
                pass

    def refresh_now(self):
        """Collect and apply synchronously on the calling thread."""
        try:
            result = self._collect()
        except Exception:
            return
        if result is not None:
            self._apply(result)
