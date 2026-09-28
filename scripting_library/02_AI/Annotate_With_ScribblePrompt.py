# -*- coding: utf-8 -*-
"""Run ScribblePrompt on a Mimics Mask in an external Python process."""

from __future__ import print_function

import os
import sys


_here = os.path.dirname(os.path.abspath(__file__))
_root = _here
for _ in range(5):
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
    "interactive_algorithms_mimics",
    action_attr="ACTION_SCRIBBLEPROMPT",
)
