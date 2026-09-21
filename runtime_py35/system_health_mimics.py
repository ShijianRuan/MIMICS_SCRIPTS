# -*- coding: utf-8 -*-
"""Open the system health overview panel (inside Mimics).

The panel is an external PySide6 window (tools/system_health_panel.py): it
aggregates the process registry, resource locks, import queues, and the
nnInteractive server state into one read-only page with explicit action
buttons. This launcher starts it as a registered external_ui process and
returns immediately.
"""

from __future__ import print_function

import external_window_launcher


def open_health_panel():
    """Start the health panel window. Returns 0 on success."""
    return external_window_launcher.open_external_window(
        "system_health_panel.py",
        "System Health",
        "health_panel",
        "health_panel_",
        log_start_message=(
            "System health panel started in an external process (PID {pid})."
        ),
    )


def main():
    return open_health_panel()


if __name__ == "__main__":
    main()
