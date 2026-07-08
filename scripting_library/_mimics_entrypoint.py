# -*- coding: utf-8 -*-
"""Compatibility shim for old Mimics deployments.

The shared entrypoint now lives in runtime_py35/_mimics_entrypoint.py so it is
not shown as a normal scripting-library action. Some deployed folders may still
contain or cache this legacy location inside Mimics Python, so keep this small
shim and forward calls to the runtime implementation.
"""

from __future__ import print_function

import importlib.util
import os


def _find_runtime_entrypoint():
    current = os.path.abspath(os.path.dirname(__file__))
    for _ in range(6):
        candidate = os.path.join(current, "runtime_py35", "_mimics_entrypoint.py")
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    raise ImportError("Could not find runtime_py35/_mimics_entrypoint.py")


def _load_runtime(path):
    try:
        spec = importlib.util.spec_from_file_location("_mimics_runtime_entrypoint", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:
        import imp
        return imp.load_source("_mimics_runtime_entrypoint", path)


_runtime = _load_runtime(_find_runtime_entrypoint())

run_runtime_entry = _runtime.run_runtime_entry
load_runtime_module = _runtime.load_runtime_module
