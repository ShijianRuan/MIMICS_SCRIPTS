# -*- coding: utf-8 -*-
"""Import a dataset into Mimics .mcs files.

Scripting Library entry point.  The annotator clicks this script in Mimics
to convert a dataset (or single case) into .mcs work packages.

Two modes:
  1. Batch: enter the dataset root directory → all cases become .mcs files
  2. Single: enter a single case directory → one .mcs file
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

import mimics_import


if __name__ == "__main__":
    mimics_import.main()
elif "mimics" in sys.modules:
    mimics_import.main()
