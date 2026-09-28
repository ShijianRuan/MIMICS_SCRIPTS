# -*- coding: utf-8 -*-
"""Open the floating drop-to-import window.

Starts a small always-on-top external window. Drag image files, case
folders, or a whole dataset folder onto it (or paste paths with Ctrl+V);
it recognizes what was dropped against the dataset profile and submits it
to the same background import workers as Import Dataset / Import Single
Case. The window closes itself after an idle timeout.
"""

from __future__ import print_function

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = _here
for _ in range(5):
    _rt = os.path.join(_root, "runtime_py35")
    if os.path.isdir(_rt):
        if _rt not in sys.path:
            sys.path.insert(0, _rt)
        break
    _root = os.path.dirname(_root)

from _mimics_entrypoint import run_runtime_entry

run_runtime_entry(globals(), __file__, "import_drop_mimics")
