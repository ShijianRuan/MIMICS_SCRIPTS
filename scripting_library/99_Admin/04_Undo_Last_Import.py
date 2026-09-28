# -*- coding: utf-8 -*-
"""Undo the most recent import.

Every import writes a receipt next to the created .mcs file listing exactly
what it created. This action finds the newest receipt, deletes the masks
that import created (in one transaction), and removes the .mcs file when it
has not changed since the import. You are asked to confirm first.
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

run_runtime_entry(globals(), __file__, "import_undo_mimics")
