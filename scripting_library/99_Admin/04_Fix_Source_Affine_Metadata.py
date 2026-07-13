# -*- coding: utf-8 -*-
"""Repair stale source_voxel_to_ras_matrix metadata on the active image.

Rewrites the active image's source_voxel_to_ras_matrix metadata to the true
nibabel RAS affine read from the source NIfTI on disk. Fixes the inference
error "source image affine does not match the open Mimics project (max abs
diff ~438)" on cases prepared with an older version of the bridge.

Prerequisites:
  * Open the target project (.mcs) in Mimics.
  * Activate the source image (the one imported from the NIfTI).

Run this entry, then re-run few-shot inference on the case.
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

run_runtime_entry(globals(), __file__, "fix_source_affine_metadata", function_name="main")
