# -*- coding: utf-8 -*-
"""Show every import/export batch task in one window.

Lists the recent import runs, background .mcs queues, mask-export jobs,
foreground export tasks, mask-append jobs, and drop imports with live
status, progress, and one-click access to each task's folder and status
file. Read-only: stopping a task stays in the matching Stop entry.
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


run_runtime_entry(globals(), __file__, "batch_status_mimics")
