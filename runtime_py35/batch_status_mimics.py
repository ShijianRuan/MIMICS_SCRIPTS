# -*- coding: utf-8 -*-
"""Open the batch status window (inside Mimics).

Aggregates every on-disk import/export record (import runs, .mcs queues,
mask-export jobs, foreground export tasks, mask-append jobs, drop imports)
into one external read-only window so an annotator can see what is running,
what failed, and where the logs are, without hunting through folders.

Py3.5 constraints: no f-strings, no pathlib, .format() with positional
indexes only.
"""

from __future__ import print_function

import external_window_launcher


def open_batch_status():
    """Start the batch status window. Returns 0 on success."""
    return external_window_launcher.open_external_window(
        "batch_status_viewer.py",
        "Batch Status",
        "batch_status",
        "batch_status_",
        log_start_message=(
            "Batch status window opened in an external process (PID {pid})."
        ),
    )


def main():
    return open_batch_status()


if __name__ == "__main__":
    main()
