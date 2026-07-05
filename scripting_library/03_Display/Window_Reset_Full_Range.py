# -*- coding: utf-8 -*-
"""Reset window/level to the full active image gray-value range."""

from __future__ import print_function

import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIBRARY_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if LIBRARY_DIR not in sys.path:
    sys.path.insert(0, LIBRARY_DIR)

from _mimics_entrypoint import run_runtime_entry


run_runtime_entry(globals(), __file__, "window_level_mimics", action_value="reset")
