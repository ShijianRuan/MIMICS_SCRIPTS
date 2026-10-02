# -*- coding: utf-8 -*-
"""Import data into a Mimics project.

Opens the import window: drag image files, case folders, or a whole
dataset folder onto it, paste paths with Ctrl+V, or pick them with the
choose buttons. It recognizes what you gave it against the dataset
profile and submits it to the background import workers. The window
closes itself after an idle timeout.
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
