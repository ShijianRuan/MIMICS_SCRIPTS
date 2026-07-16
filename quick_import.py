# -*- coding: utf-8 -*-
"""Toggle Editor entry for the supported single-case import workflow."""

from __future__ import print_function

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNTIME = os.path.join(ROOT, "runtime_py35")
if RUNTIME not in sys.path:
    sys.path.insert(0, RUNTIME)

import mimics_import

mimics_import.main(import_mode="single_case")
