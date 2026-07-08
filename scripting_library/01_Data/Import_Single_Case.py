# -*- coding: utf-8 -*-
"""Import a single case folder into Mimics as an .mcs file.

Opens a folder picker to select one case directory containing imaging data
(ct.nii.gz, mri.nii.gz, or dicom/) and optional segmentations, then
converts it to an .mcs package.

This is the single-case equivalent of Import_Dataset — no batch scanning,
no queue, no background Mimics polling.
"""

from __future__ import print_function

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = _here
_entrypoint_path = None
for _ in range(5):
    _rt = os.path.join(_root, "runtime_py35")
    if os.path.isdir(_rt):
        if _rt not in sys.path:
            sys.path.insert(0, _rt)
        _entrypoint_path = os.path.join(_rt, "_mimics_entrypoint.py")
        break
    _root = os.path.dirname(_root)

if _entrypoint_path and os.path.isfile(_entrypoint_path):
    try:
        import importlib.util as _importlib_util
        _spec = _importlib_util.spec_from_file_location("_mimics_entrypoint_import_single_case", _entrypoint_path)
        _entrypoint = _importlib_util.module_from_spec(_spec)
        _spec.loader.exec_module(_entrypoint)
    except Exception:
        import imp as _imp
        _entrypoint = _imp.load_source("_mimics_entrypoint_import_single_case", _entrypoint_path)
    run_runtime_entry = _entrypoint.run_runtime_entry
else:
    from _mimics_entrypoint import run_runtime_entry

run_runtime_entry(globals(), __file__, "mimics_import", action_value="single_case")
