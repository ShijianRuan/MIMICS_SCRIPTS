# -*- coding: utf-8 -*-
"""Collect a redacted diagnostics bundle for support.

Aggregates log tails, runtime state (processes/locks/queues), config
snapshots, and environment versions into one zip next to the project root.
Absolute paths are redacted to their last two components and tokens are
masked, so the bundle is safe to share.
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

run_runtime_entry(globals(), __file__, "collect_diagnostics_mimics")
