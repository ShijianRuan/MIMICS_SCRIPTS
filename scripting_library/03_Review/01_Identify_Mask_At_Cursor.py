# -*- coding: utf-8 -*-
"""Identify which mask(s) contain the voxel under the cursor.

Click on any 2D/3D view to see which masks contain that voxel,
together with the mask name, visibility status, and color.
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


run_runtime_entry(globals(), __file__, "mask_identifier")
