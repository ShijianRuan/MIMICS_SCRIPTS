# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LIBRARY_ROOT = os.path.dirname(os.path.dirname(HERE))
if LIBRARY_ROOT not in sys.path:
    sys.path.insert(0, LIBRARY_ROOT)

import _mimics_entrypoint

sys.exit(_mimics_entrypoint.run_runtime_entry(globals(), __file__, "fewshot_mimics"))
