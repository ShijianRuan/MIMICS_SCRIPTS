# -*- coding: utf-8 -*-
"""Start DINOv3 few-shot training for the selected organ Mask."""

from __future__ import print_function

import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIBRARY_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if LIBRARY_DIR not in sys.path:
    sys.path.insert(0, LIBRARY_DIR)

from _mimics_entrypoint import run_runtime_entry


run_runtime_entry(globals(), __file__, "fewshot_mimics", action_attr="BUTTON_TRAIN")
