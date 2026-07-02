# -*- coding: utf-8 -*-
"""Stop Mimics-Script background processes and services."""

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

import mimics_stop_background


if __name__ == "__main__":
    mimics_stop_background.main()
elif "mimics" in sys.modules:
    mimics_stop_background.main()
