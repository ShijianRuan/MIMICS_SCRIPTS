# -*- coding: utf-8 -*-
"""Small shared launcher for Mimics Scripting Library entries.

Each visible entry should only declare the runtime module/action it wants.
This keeps path setup and module loading behavior identical across folders.
"""

from __future__ import print_function

import importlib
import os
import sys


def _library_root(caller_file):
    current = os.path.abspath(os.path.dirname(caller_file))
    for _ in range(4):
        if os.path.isfile(os.path.join(current, "_mimics_entrypoint.py")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return os.path.abspath(os.path.join(os.path.dirname(caller_file), ".."))


def _runtime_dir(caller_file):
    script_dir = os.path.abspath(os.path.dirname(caller_file))
    library_root = _library_root(caller_file)
    candidates = (
        os.path.join(script_dir, "runtime_py35"),
        os.path.join(library_root, "runtime_py35"),
        os.path.join(os.path.dirname(library_root), "runtime_py35"),
    )
    for candidate in candidates:
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)
    return os.path.abspath(os.path.join(os.path.dirname(library_root), "runtime_py35"))


def load_runtime_module(caller_file, module_name):
    runtime_dir = _runtime_dir(caller_file)
    if runtime_dir not in sys.path:
        sys.path.insert(0, runtime_dir)
    return importlib.import_module(module_name)


def _inside_mimics():
    if "mimics" in sys.modules:
        return True
    try:
        import mimics  # noqa: F401
        return True
    except Exception:
        return False


def run_runtime_entry(caller_globals, caller_file, module_name, function_name="main", action_attr=None, action_value=None):
    name = caller_globals.get("__name__", "")
    if name != "__main__" and not _inside_mimics():
        return None
    module = load_runtime_module(caller_file, module_name)
    function = getattr(module, function_name)
    if action_attr:
        return function(getattr(module, action_attr))
    if action_value is not None:
        return function(action_value)
    return function()
