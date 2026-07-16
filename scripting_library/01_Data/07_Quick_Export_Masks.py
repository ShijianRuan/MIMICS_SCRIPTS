# -*- coding: utf-8 -*-
"""快速导出当前 Mimics 项目的所有 Mask 为 NIfTI 文件。

选择输出文件夹后，自动导出所有 Mask 到 <输出目录>/<case_id>/segmentations/。
无需额外的 UI 配置，即点即用。
"""

from __future__ import print_function

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = _here
for _ in range(5):
    _rt = os.path.join(_root, "runtime_py35")
    if os.path.isdir(_rt):
        if _rt not in sys.path:
            sys.path.insert(0, _rt)
        break
    _root = os.path.dirname(_root)

from _mimics_entrypoint import run_runtime_entry

run_runtime_entry(globals(), __file__, "mimics_export", function_name="quick_export_main")
