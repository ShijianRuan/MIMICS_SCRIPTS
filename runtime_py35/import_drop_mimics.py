# -*- coding: utf-8 -*-
"""Open the floating drop-to-import window (inside Mimics).

Mimics' scripting API cannot receive native drag events, so the drop zone
lives in an external always-on-top PySide6 window (tools/import_drop_window.py).
This launcher starts that window as a registered external_ui process and
returns immediately; the window is fully self-contained: it recognizes the
dropped payload against the dataset profile, confirms with the user, and
submits to the same import workers the path-setup UI uses. It exits on its
own after an idle timeout, so there is no resident service to manage.
"""

from __future__ import print_function

import external_window_launcher


def open_drop_window():
    """Start the drop-to-import window. Returns 0 on success."""
    return external_window_launcher.open_external_window(
        "import_drop_window.py",
        "Drop Import",
        "drop_import",
        "drop_window_",
        cleanup_policy="idle_timeout_s:1800",
        log_start_message=(
            "Drop-to-import window started in an external process (PID {pid}). "
            "It closes automatically when idle."
        ),
    )


def main():
    return open_drop_window()


if __name__ == "__main__":
    main()
