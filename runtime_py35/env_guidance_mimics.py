# -*- coding: utf-8 -*-
"""Open the environment guidance window (inside Mimics).

The guidance window is an external PySide6 dialog (tools/env_guidance.py):
it detects the common non-zero-config situations -- missing Python
environment, failed or incomplete setup, a checkout moved from another
machine, missing FlexiCT weights -- and explains each one with a repair
button. This launcher starts it as a registered external_ui process and
returns immediately.
"""

from __future__ import print_function

import external_window_launcher


def open_env_guidance():
    """Start the environment guidance window. Returns 0 on success."""
    return external_window_launcher.open_external_window(
        "env_guidance.py",
        "Environment Guidance",
        "env_guidance",
        "env_guidance_",
        log_start_message=(
            "Environment guidance window started in an external process (PID {pid})."
        ),
    )


def main():
    return open_env_guidance()


if __name__ == "__main__":
    main()
