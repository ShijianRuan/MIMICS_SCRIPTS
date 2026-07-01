# -*- coding: utf-8 -*-
"""Export Mimics masks back to a dataset.

Scripting Library entry point.  The annotator clicks this script in Mimics
after finishing annotation to write masks back as NIfTI files into the
original dataset's segmentations/ directory.

Two modes:
  1. Single: enter the case directory. The saved
     <dataset>/mcs_output/<case>.mcs is exported in a background Mimics process.
  2. Batch: enter the dataset root. Each saved .mcs in mcs_output/ is exported
     in a background Mimics process.
"""

from __future__ import print_function

import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for candidate in (
    os.path.join(SCRIPT_DIR, "runtime_py35"),
    os.path.join(SCRIPT_DIR, "..", "runtime_py35"),
):
    if os.path.isdir(candidate):
        sys.path.insert(0, os.path.abspath(candidate))
        break

import mimics_export


if __name__ == "__main__":
    mimics_export.main()
elif "mimics" in sys.modules:
    mimics_export.main()
