"""Audit-only probe of project guards when get_active_project is unavailable.

Executes unchanged production function bodies against a documented-API-shaped
stub. No Mimics instance or real project is opened or modified.
"""
import ast
import json
import os
from pathlib import Path
import tempfile
import types

ROOT = Path(__file__).resolve().parents[3]


def extract(relative, name, namespace):
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / relative), "exec"), namespace)


results = {}
with tempfile.TemporaryDirectory() as folder:
    target = Path(folder) / "target.mcs"
    target.write_bytes(b"synthetic audit fixture; not a real Mimics project")
    for key, module, function in (
        ("al", "runtime_py35/flexict_mimics.py", "_al_open_case"),
        ("undo_import", "runtime_py35/import_undo_mimics.py", "_open_project"),
    ):
        opened = []
        messages = []
        fake_file = types.SimpleNamespace(
            is_project_loaded=lambda: True,
            is_project_modified=lambda: True,
            get_project_information=lambda: types.SimpleNamespace(project_path=str(Path(folder) / "current.mcs")),
            open_project=lambda **kwargs: opened.append(kwargs),
        )
        fake_mimics = types.SimpleNamespace(
            file=fake_file,
            dialogs=types.SimpleNamespace(message_box=lambda *a, **k: messages.append((a, k))),
        )
        namespace = {
            "os": os,
            "mimics": fake_mimics,
            "TITLE": "Audit",
            "_al_case_source_geometry": lambda *a: {"mcs_path": str(target)},
        }
        extract(module, function, namespace)
        arguments = (folder, "case1") if key == "al" else (str(target),)
        returned = namespace[function](*arguments)
        results[key] = {
            "current_project_loaded": True,
            "current_project_modified": True,
            "get_active_project_available": False,
            "open_project_calls": len(opened),
            "guard_message_calls": len(messages),
            "returned_success": bool(returned[0] if isinstance(returned, tuple) else returned),
        }
print(json.dumps(results, ensure_ascii=False, indent=2))
