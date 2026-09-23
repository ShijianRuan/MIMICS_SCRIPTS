# -*- coding: utf-8 -*-
"""Open the external window/level preset editor (inside Mimics).

The editor (tools/window_level_editor_ui.py) lets an annotator hand-enter
window width/level values, add or remove presets, and edit the mask-name
keywords — everything that previously required editing
window_level_presets.json by hand. This launcher starts it as a registered
external_ui process and returns immediately.
"""

from __future__ import print_function

import external_window_launcher


def open_window_level_editor():
    """Start the preset editor window. Returns 0 on success."""
    return external_window_launcher.open_external_window(
        "window_level_editor_ui.py",
        "Window/Level Presets",
        "window_level_editor",
        "window_level_editor_",
        log_keep=5,
    )


def main():
    return open_window_level_editor()


if __name__ == "__main__":
    main()
