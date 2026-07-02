# -*- coding: utf-8 -*-
"""Scripting Library entry for DINOv3 few-shot segmentation."""

from __future__ import print_function

import os
import sys
import importlib


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for candidate in (
    os.path.join(SCRIPT_DIR, "runtime_py35"),
    os.path.join(SCRIPT_DIR, "..", "runtime_py35"),
):
    if os.path.isdir(candidate):
        RUNTIME_DIR = os.path.abspath(candidate)
        break
else:
    RUNTIME_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "runtime_py35"))

if RUNTIME_DIR not in sys.path:
    sys.path.insert(0, RUNTIME_DIR)

import fewshot_mimics


def _launch():
    active = getattr(fewshot_mimics, "_MONITORS", None)
    if not active:
        importlib.reload(fewshot_mimics)
    fewshot_mimics.main()


if __name__ == "__main__":
    _launch()
elif "mimics" in sys.modules:
    _launch()
