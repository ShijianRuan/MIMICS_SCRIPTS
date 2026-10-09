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
from pathlib import Path


class BackgroundRefresh:
    # While the viewer window is hidden or minimized the periodic scan
    # still stats every job directory; on a network workspace that is
    # seconds of I/O per tick for a table nobody is looking at. Keep a
    # slow heartbeat (10s) so the window is fresh when re-shown.
    HIDDEN_INTERVAL_MS = 10000

    def __init__(self, qt_core, parent, collect, apply, interval_ms=2000,
                 on_error=None):
        self._collect = collect
        self._apply = apply
        self._on_error = on_error
        self._parent = parent
        self._results = queue.Queue()
        self._lock = threading.Lock()
        self._pending = False
        self._stop = False
        self._interval_ms = int(interval_ms)
        # Drain timer: runs on the GUI thread, applies finished results.
        self._drain_timer = qt_core.QTimer(parent)
        self._drain_timer.setInterval(250)
        self._drain_timer.timeout.connect(self._drain)
        self._drain_timer.start()
        # Period timer: kicks off background collections.
        self._timer = qt_core.QTimer(parent)
        self._timer.setInterval(self._interval_ms)
        self._timer.timeout.connect(self._on_period_tick)
        self._timer.start()

    def stop(self):
        """Stop both timers. A closed window must not keep collecting on
        its hidden heartbeat forever, and timers firing while the wrapped
        window is being torn down corrupt the heap (0xc0000374 in the GUI
        smoke suite, ~70% of runs)."""
        self._stop = True
        try:
            self._timer.stop()
        except RuntimeError:  # C++ parent already deleted
            pass
        try:
            self._drain_timer.stop()
        except RuntimeError:
            pass

    @staticmethod
    def _is_shown(parent):
        """True unless parent is a widget that is hidden or minimized.

        A never-shown widget counts as hidden — nobody is looking at it.
        Non-widget parents (tests) count as shown. Visibility only slows
        periodic ticks; explicit ``request()`` and ``refresh_now()``
        always collect immediately.
        """
        try:
            if hasattr(parent, "isMinimized") and parent.isMinimized():
                return False
            if hasattr(parent, "isVisible"):
                return bool(parent.isVisible())
        except Exception:
            pass
        return True

    def _on_period_tick(self):
        # Hidden/minimized windows still get a slow heartbeat (10s) so the
        # table is fresh when re-shown; visible windows tick at full speed.
        # Checking visibility here rather than in hideEvent/showEvent keeps
        # this deterministic: Qt virtual dispatch does not reliably reach
        # monkey-patched instance attributes.
        if self._stop:
            return
        parent = self._parent
        if not self._is_shown(parent):
            if self._timer.interval() != self.HIDDEN_INTERVAL_MS:
                self._timer.setInterval(self.HIDDEN_INTERVAL_MS)
        else:
            if self._timer.interval() != self._interval_ms:
                # Window just became visible again — refresh immediately
                # instead of waiting out the leftover hidden interval.
                self._timer.setInterval(self._interval_ms)
                self.request()
                return
        self.request()

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
                # A collect that raised (F34: keep the last good frame, but
                # let the viewer say so instead of showing a stale "live"
                # timestamp as if nothing happened).
                if self._on_error is not None:
                    try:
                        self._on_error()
                    except Exception:
                        pass
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


def read_log_tail(status, cache):
    """Last 512 KiB of a job log, cached by (path, size, mtime).

    Called from the collect worker thread: reading a training log on a
    network workspace can stall, so the GUI thread must never do it. The
    cache means finished jobs (whose logs never change) are read once,
    not on every refresh tick.
    """
    path = Path(str(status.get("log_path") or ""))
    try:
        st = path.stat()
    except OSError:
        return ""
    if not path.is_file():
        return ""
    key = (str(path), st.st_size, st.st_mtime)
    if key in cache:
        return cache[key]
    try:
        with path.open("rb") as handle:
            handle.seek(max(0, st.st_size - 512 * 1024))
            text = handle.read().decode("utf-8", "replace")
    except OSError as exc:
        return "无法读取日志：{}".format(exc)
    if len(cache) >= 64:
        cache.clear()
    cache[key] = text
    return text
