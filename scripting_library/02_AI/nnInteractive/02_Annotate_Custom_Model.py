# -*- coding: utf-8 -*-
"""Annotate with a reusable nnInteractive model trained for this target."""

from __future__ import print_function

import os
import sys


_here = os.path.dirname(os.path.abspath(__file__))
_root = _here
for _ in range(6):
    _runtime = os.path.join(_root, "runtime_py35")
    if os.path.isdir(_runtime):
        if _runtime not in sys.path:
            sys.path.insert(0, _runtime)
        break
    _root = os.path.dirname(_root)

from _mimics_entrypoint import run_runtime_entry


run_runtime_entry(
    globals(),
    __file__,
    "nninteractive_finetune_mimics",
    action_attr="ACTION_ANNOTATE",
)
