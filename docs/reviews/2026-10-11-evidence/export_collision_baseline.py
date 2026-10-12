"""Negative control: run the existing F32 test against original baseline code.

Only an isolated Python module and synthetic test files are changed. No source
file is edited. On a case-sensitive filesystem the observation may differ.
"""
import importlib.util
from pathlib import Path
import subprocess
import sys
import types
import unittest

root=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(root),str(root/'tools'),str(root/'runtime_py35')]
spec=importlib.util.spec_from_file_location('baseline_test_support',root/'tests/test_all.py')
support=importlib.util.module_from_spec(spec);spec.loader.exec_module(support)
source=subprocess.check_output(['git','show','a34805e3b6724c25353642aa40017080e9e81b7e:runtime_py35/mimics_export.py'],cwd=root,text=True)
module=types.ModuleType('mimics_export')
module.__file__=str(root/'runtime_py35/mimics_export.py')
sys.modules['mimics_export']=module
exec(compile(source,module.__file__,'exec'),module.__dict__)
suite=unittest.TestSuite([support.TestNewFeatures('test_mimics_export_case_collisions_get_distinct_buffers')])
result=unittest.TextTestRunner(verbosity=2).run(suite)
print('BASELINE_OBSERVATION:', 'collision reproduced' if result.failures else 'not reproduced on this filesystem')
