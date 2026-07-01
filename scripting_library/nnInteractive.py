# -*- coding: utf-8 -*-
"""Standalone Mimics Scripting Library entry for nnInteractive."""

from __future__ import print_function

import os
import sys
import importlib


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# In the repo layout nnInteractive.py lives in scripting_library/ and runtime
# scripts sit in ../runtime_py35.  In an exported worklist both this entry and
# runtime_py35/ are siblings under the same root.
for candidate in (
    os.path.join(SCRIPT_DIR, "runtime_py35"),
    os.path.join(SCRIPT_DIR, "..", "runtime_py35"),
):
    if os.path.isdir(candidate):
        RUNTIME_DIR = os.path.abspath(candidate)
        break
else:
    RUNTIME_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "runtime_py35"))

if RUNTIME_DIR not in sys.path:
    sys.path.insert(0, RUNTIME_DIR)

import nninteractive_mimics


def _launch():
    # Some Mimics Scripting Library hosts execute scripts with a module name
    # other than "__main__". Keep this entry robust across both modes. Avoid
    # reload while an async monitor is alive because reload resets module-level
    # monitor/visual-object registries that are needed for auto-apply cleanup.
    active_async_state = (
        getattr(nninteractive_mimics, "_ASYNC_MONITORS", None)
        or getattr(nninteractive_mimics, "_ASYNC_VISUAL_OBJECTS", None)
        or getattr(nninteractive_mimics, "_ASYNC_IMAGE_WORKERS", None)
    )
    if not active_async_state:
        importlib.reload(nninteractive_mimics)
    nninteractive_mimics.main()


if __name__ == "__main__":
    _launch()
elif "mimics" in sys.modules:
    _launch()
