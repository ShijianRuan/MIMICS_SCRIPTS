# -*- coding: utf-8 -*-
"""Shared launcher for Mimics Scripting Library entries.

Each visible entry declares the runtime module/action it wants.
This keeps path setup and module loading behavior identical across folders.

This file lives in runtime_py35/ so it does NOT appear as an entry in the
Mimics Scripting Library menu.
"""

from __future__ import print_function

import importlib
import inspect
import os
import sys


def _project_root():
    """Return the project root (parent of runtime_py35/)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _runtime_dir():
    """Return the runtime_py35/ directory."""
    return os.path.dirname(os.path.abspath(__file__))


def _find_runtime_dir(caller_file):
    """Walk up from caller_file to find the project root's runtime_py35/."""
    current = os.path.abspath(os.path.dirname(caller_file))
    for _ in range(6):
        candidate = os.path.join(current, "runtime_py35")
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    # Fallback: use our own location
    return _runtime_dir()


def load_runtime_module(caller_file, module_name):
    runtime_dir = _find_runtime_dir(caller_file)
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
    action_provided = False
    action = None
    if action_attr:
        action = getattr(module, action_attr)
        action_provided = True
    elif action_value is not None:
        action = action_value
        action_provided = True
    if action_provided:
        setattr(module, "MIMICS_ENTRY_ACTION", action)
        if _accepts_positional_arg(function):
            return function(action)
    return function()


def _accepts_positional_arg(function):
    try:
        spec = inspect.getfullargspec(function)
        return bool(spec.varargs or spec.args)
    except Exception:
        return True
