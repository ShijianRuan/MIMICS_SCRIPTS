# -*- coding: utf-8 -*-
"""Open the external AI Model Manager window (inside Mimics).

The manager (tools/model_manager_ui.py) lists every registered model of
the three families (nnInteractive task models, managed nnU-Net models,
FlexiCT models), imports model packages (zip) received from a developer
with one click, switches the default model per family, and removes
broken registry entries.
"""

from __future__ import print_function

import external_window_launcher


def open_model_manager():
    """Start the AI Model Manager window. Returns 0 on success."""
    return external_window_launcher.open_external_window(
        "model_manager_ui.py",
        "AI 模型管理器",
        "model_manager",
        "model_manager_",
        log_keep=5,
    )


def main():
    return open_model_manager()


if __name__ == "__main__":
    main()
