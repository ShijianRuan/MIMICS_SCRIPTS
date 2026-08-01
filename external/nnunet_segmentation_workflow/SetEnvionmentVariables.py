"""Process-scoped nnU-Net environment helpers.

The workflow must never modify a user's shell startup files. Callers that need
child isolation should pass the dictionary returned by ``environment_values``
to ``subprocess.Popen(env=...)``.
"""

from __future__ import annotations

import os
from typing import Mapping


def environment_values(values: Mapping[str, object]) -> dict[str, str]:
    return {
        str(name): str(value)
        for name, value in values.items()
        if value not in (None, "")
    }


def set_process_environment(values: Mapping[str, object]) -> dict[str, str]:
    normalized = environment_values(values)
    os.environ.update(normalized)
    return normalized


def add_to_user_shell_config(variable_name, variable_value):
    """Backward-compatible name with process-only behavior.

    Older workflow code called this helper expecting persistence. Persisting
    nnU-Net paths globally caused unrelated terminals and projects to resolve
    the wrong dataset roots, so the compatibility function is intentionally
    scoped to the current process now.
    """
    os.environ[str(variable_name)] = str(variable_value)
    return str(variable_value)
