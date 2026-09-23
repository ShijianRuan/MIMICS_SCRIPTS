# -*- coding: utf-8 -*-
"""Open the external configuration editor (inside Mimics).

The editor (tools/config_editor_ui.py) exposes the commonly tuned keys of
the user-facing JSON configs (IO, nnInteractive, nnInteractive fine-tune,
FlexiCT, interactive algorithms) with type validation and one-step
rollback — no hand-editing JSON in a text editor.
"""

from __future__ import print_function

import external_window_launcher


def open_config_editor():
    """Start the configuration editor window. Returns 0 on success."""
    return external_window_launcher.open_external_window(
        "config_editor_ui.py",
        "Configuration Editor",
        "config_editor",
        "config_editor_",
        log_keep=5,
    )


def main():
    return open_config_editor()


if __name__ == "__main__":
    main()
