"""Environment guidance: detect common setup problems and explain fixes.

Phase 2b of the product-quality program: when the deployment is not
zero-config (interpreter missing, setup failed mid-way, the checkout was
moved to a new machine, or FlexiCT weights are absent), the annotator
should get a graphical explanation with a repair path instead of a
terminal-only error.

The module is split in two:

* :func:`collect_issues` -- pure reads (no Qt, no Mimics), unit-testable
  headless. Returns a list of issue dicts with human-readable text and a
  suggested fix action.
* :func:`show_dialog` -- a small PySide6 window that renders the issues
  and, where possible, starts the repair worker via the same
  ``setup_environment`` orchestration the Admin menu uses.

Issue kinds and their fix actions:

* ``python_missing``      -> run Setup Environment (setup-from-scratch /
                             offline install depending on bundle presence)
* ``setup_failed``        -> re-run the failed setup action
* ``setup_incomplete``    -> run Repair (install missing packages)
* ``migration_pending``   -> run migrate_root with the detected old root
* ``flexict_weights``     -> informational (where to put the weights)
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# A setup state older than this is stale (from a previous session) and is
# not reported as a current problem.
SETUP_STATE_MAX_AGE_SECONDS = 24 * 60 * 60

# Absolute Windows paths in the root config JSONs that do not exist on
# this machine are the classic "moved checkout" symptom. Only values that
# look like absolute Windows/UNC paths and contain a separator are
# considered; root-relative values are already fine.
_MIGRATION_PATH_KEYS = (
    "workspace_dir",
    "official_model_dir",
    "mimics_output_dir",
    "mimics_background_exe",
    "pretrained_weights_dir",
)

_MIGRATION_CONFIG_FILES = (
    "nninteractive_config.json",
    "nninteractive_finetune_config.json",
    "mimics_io_config.json",
    "interactive_algorithms_config.json",
    "flexict_config.json",
    "dataset_profiles.json",
)


def _read_json(path: Path, default=None):
    import json

    try:
        with open(str(path), "r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value
    except Exception:
        return default


def find_external_python(project_root: Path | None = None) -> str:
    """Same candidates as runtime_common.find_external_python (py3.13)."""
    root = Path(project_root or ROOT)
    candidates = []
    for env_name in ("python_env", "nninteractive_env"):
        for base in (root / env_name, root.parent / env_name):
            candidates.extend((
                base / "python.exe",
                base / "Scripts" / "python.exe",
                base / "python" / "python.exe",
                base / "bin" / "python3",
                base / "bin" / "python",
            ))
    candidates.append(root / "python.exe")
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return ""


def setup_state(project_root: Path | None = None) -> dict:
    """Newest setup_env state file, or {} when absent/old/unreadable."""
    root = Path(project_root or ROOT)
    state = _read_json(root / ".mimics_runtime" / "setup_env_state.json", {}) or {}
    if not isinstance(state, dict):
        return {}
    age = time.time() - float(state.get("updated_at_epoch") or 0)
    if age < 0 or age > SETUP_STATE_MAX_AGE_SECONDS:
        return {}
    return state


def detect_old_root(project_root: Path | None = None) -> str:
    """Best-effort detection of a leftover old install root.

    Scans the root config JSONs for absolute path values under a common
    dead prefix. Returns the longest common dead prefix (a plausible old
    root) or "" when the configs look clean.

    Detection is deliberately conservative: a value only counts when it is
    an absolute path that does NOT exist on this machine. Relative values
    and existing absolute paths are fine and never reported.
    """
    root = Path(project_root or ROOT)
    dead_values = []
    for name in _MIGRATION_CONFIG_FILES:
        payload = _read_json(root / name, {}) or {}
        if not isinstance(payload, dict):
            continue
        for key in _MIGRATION_PATH_KEYS:
            value = payload.get(key)
            if not isinstance(value, str) or not value.strip():
                continue
            if not re.match(r"^[A-Za-z]:[\\/]", value) and not value.startswith("\\\\"):
                continue  # relative or env-like -- fine
            if Path(value).exists():
                continue
            dead_values.append(value)
    if not dead_values:
        return ""
    # The old root is the common prefix of the dead values (or its parent
    # when the values all sit one level below the root). The candidate must
    # itself be dead on this machine and every value must be strictly under
    # it so migrate_root rewrites "<old-root>/sub/path" -> "sub/path".
    absolute = [os.path.abspath(v) for v in dead_values]
    for candidate in (os.path.commonpath(absolute), os.path.dirname(os.path.commonpath(absolute))):
        if not candidate or len(candidate) <= 3:  # "C:\" or garbage
            continue
        if os.path.exists(candidate):
            continue  # a live directory is not a plausible dead old root
        if all(os.path.relpath(v, candidate) not in (".", "..") for v in absolute):
            return candidate
    return ""


def _nninteractive_model_candidates(root: Path) -> list[str]:
    """Same candidates as runtime_py35.nninteractive_mimics's official-model
    chain (the chain whose bare error this guidance replaces): env var ->
    nninteractive_config.json's model_dir -> the found external environment's
    models folder -> the two repo env folders."""
    candidates = [os.environ.get("NNINTERACTIVE_MODEL_DIR", "")]
    config = _read_json(root / "nninteractive_config.json", {}) or {}
    configured = str(config.get("model_dir") or "").strip()
    if configured:
        resolved = Path(configured)
        if not resolved.is_absolute():
            resolved = root / configured
        candidates.append(str(resolved))
    environment_root = ""
    found_python = find_external_python(root)
    if found_python:
        environment_root = str(Path(found_python).parent)
    if environment_root:
        candidates.append(os.path.join(environment_root, "models", "nnInteractive_v1.0"))
    candidates.extend((
        str(root / "python_env" / "models" / "nnInteractive_v1.0"),
        str(root / "nninteractive_env" / "models" / "nnInteractive_v1.0"),
    ))
    return candidates


def _nninteractive_official_model(root: Path) -> str:
    """First candidate that is a directory with a fold_*/checkpoint_final.pth
    (the same usability test as nninteractive_mimics._model_folds)."""
    for candidate in _nninteractive_model_candidates(root):
        if not candidate:
            continue
        model_dir = Path(candidate)
        if not model_dir.is_dir():
            continue
        try:
            entries = sorted(model_dir.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.name.startswith("fold_") and (entry / "checkpoint_final.pth").is_file():
                return str(model_dir)
    return ""


def collect_issues(project_root: Path | None = None) -> list[dict]:
    """Collect current environment issues. Pure reads; safe anywhere."""
    root = Path(project_root or ROOT)
    issues: list[dict] = []

    python = find_external_python(root)
    if not python:
        bundle = (root / "python").is_dir() and (root / "wheels").is_dir()
        issues.append({
            "kind": "python_missing",
            "title": "External Python environment is missing",
            "detail": (
                "The bundled Python environment (python_env/) was not found, so "
                "no AI features can run. "
                + ("An offline bundle is present, so setup needs no internet."
                   if bundle else
                   "Run Setup Environment, or re-extract the deployment package "
                   "if this machine never had it.")
            ),
            "fix_action": "offline-install" if bundle else "setup-from-scratch",
            "severity": "bad",
        })
        return issues  # everything else depends on the interpreter

    state = setup_state(root)
    if state:
        status = str(state.get("status") or "")
        if status == "error":
            issues.append({
                "kind": "setup_failed",
                "title": "Last environment setup failed",
                "detail": "{0}\nTechnical detail: {1}".format(
                    state.get("message") or "The setup worker reported an error.",
                    str(state.get("error") or "")[:300],
                ),
                "fix_action": "install",
                "severity": "bad",
            })
        elif status == "incomplete":
            missing = state.get("failed_packages") or state.get("missing_packages") or []
            issues.append({
                "kind": "setup_incomplete",
                "title": "Environment setup is incomplete",
                "detail": "{0}\nAffected packages: {1}".format(
                    state.get("message") or "Some packages did not install.",
                    ", ".join(str(m) for m in missing[:10]) or "unknown",
                ),
                "fix_action": "install",
                "severity": "warn",
            })

    old_root = detect_old_root(root)
    if old_root:
        issues.append({
            "kind": "migration_pending",
            "title": "This checkout appears to be moved from another location",
            "detail": (
                "Some configured paths still point under {0}, which does not "
                "exist on this machine. Run the migration to rewrite them to "
                "this location ({1}).".format(old_root, root)
            ),
            "fix_action": "migrate:" + old_root,
            "severity": "warn",
        })

    flexict_repo = root / "integrations" / "flexict-finetune"
    flexict_config = _read_json(root / "flexict_config.json", {}) or {}
    weights_dir = str(flexict_config.get("pretrained_weights_dir") or "").strip()
    if flexict_repo.is_dir() and not weights_dir:
        has_weights = (
            (flexict_repo / "weights" / "flexict_2d" / "model.safetensors").is_file()
            and (flexict_repo / "weights" / "flexict_3d" / "model.safetensors").is_file()
        )
        if not has_weights:
            issues.append({
                "kind": "flexict_weights",
                "title": "FlexiCT pretrained weights are not installed",
                "detail": (
                    "FlexiCT training needs the pretrained backbone files "
                    "flexict_2d/model.safetensors and flexict_3d/model.safetensors "
                    "under {0}, or a pretrained_weights_dir entry in "
                    "flexict_config.json. They ship outside the deployment "
                    "package (2 x ~576 MB); copy them from the distribution "
                    "source when FlexiCT training is needed.".format(
                        flexict_repo / "weights"
                    )
                ),
                "fix_action": "",
                "severity": "warn",
            })

    nn_model = _nninteractive_official_model(root)
    if not nn_model:
        issues.append({
            "kind": "nninteractive_model",
            "title": "nnInteractive official model is not installed",
            "detail": (
                "The official nnInteractive model (nnInteractive_v1.0 with "
                "fold_*/checkpoint_final.pth) was not found in any of the "
                "searched locations:\n{0}\n"
                "Without it the official-model annotate entry cannot run "
                "(custom trained models are unaffected). Copy the model "
                "folder from your distribution source into one of the "
                "locations above, or set NNINTERACTIVE_MODEL_DIR / the "
                "model_dir entry in nninteractive_config.json.".format(
                    "\n".join(c for c in _nninteractive_model_candidates(root) if c)
                )
            ),
            "fix_action": "",
            "severity": "bad",
        })

    return issues


def _run_setup_worker(project_root: Path, action: str) -> int:
    """Start tools/setup_env.py <action> detached; return its PID (0=fail)."""
    import subprocess

    python = find_external_python(project_root)
    if not python:
        return 0
    cmd = [python, str(Path(project_root) / "tools" / "setup_env.py"), action]
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return proc.pid
    except Exception:
        return 0


def _run_migrate(project_root: Path, old_root: str) -> int:
    """Start tools/migrate_root.py --old-root <old_root> detached."""
    import subprocess

    python = find_external_python(project_root)
    if not python:
        return 0
    cmd = [
        python, str(Path(project_root) / "tools" / "migrate_root.py"),
        "--old-root", old_root,
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return proc.pid
    except Exception:
        return 0


def run_fix(project_root: Path, fix_action: str) -> str:
    """Run the repair action for an issue. Returns human feedback text."""
    root = Path(project_root or ROOT)
    if fix_action.startswith("migrate:"):
        old_root = fix_action.split(":", 1)[1]
        pid = _run_migrate(root, old_root)
        if pid:
            return "Migration started (PID {0}). Re-run Setup > Check afterwards.".format(pid)
        return "Could not start the migration worker."
    if fix_action in ("check", "install", "offline-install", "setup-from-scratch"):
        pid = _run_setup_worker(root, fix_action)
        if pid:
            return "Setup worker started (PID {0}). Progress is shown by Admin > Setup/Repair Environment.".format(pid)
    return ""


def show_dialog(project_root: Path | None = None, parent=None) -> int:
    """Open the environment guidance window. Blocking modal."""
    from PySide6 import QtWidgets

    import ui_theme

    root = Path(project_root or ROOT)
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])  # pragma: no cover - dialog used inside Mimics
    if app is not None and hasattr(ui_theme, "configure_application"):
        try:
            ui_theme.configure_application(app)
        except Exception:
            pass

    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle("Environment Guidance")
    dialog.setMinimumSize(560, 320)
    layout = QtWidgets.QVBoxLayout(dialog)

    header = QtWidgets.QLabel("Environment Guidance")
    header.setObjectName("title")
    layout.addWidget(header)

    issues = collect_issues(root)
    if not issues:
        note = QtWidgets.QLabel(
            "No environment problems detected.\n"
            "Python, packages, configured paths, FlexiCT weights and the "
            "nnInteractive official model all check out."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
    else:
        intro = QtWidgets.QLabel(
            "{0} issue(s) need attention before all features work:".format(len(issues))
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QtWidgets.QWidget()
        page = QtWidgets.QVBoxLayout(inner)
        for issue in issues:
            box = QtWidgets.QGroupBox(issue["title"])
            box_layout = QtWidgets.QVBoxLayout(box)
            detail = QtWidgets.QLabel(issue["detail"])
            detail.setWordWrap(True)
            box_layout.addWidget(detail)
            if issue.get("fix_action"):
                fix_btn = QtWidgets.QPushButton("Fix: {0}".format(
                    "Run migration" if issue["fix_action"].startswith("migrate:")
                    else "Run Setup / Repair Environment"
                ))
                fix_btn.setObjectName("primary")
                feedback = QtWidgets.QLabel("")
                feedback.setWordWrap(True)
                feedback.hide()

                def _clicked(_=False, action=str(issue["fix_action"])):
                    text = run_fix(root, action)
                    feedback.setText(text or "Nothing to run for this issue.")
                    feedback.show()

                fix_btn.clicked.connect(_clicked)
                box_layout.addWidget(fix_btn)
                box_layout.addWidget(feedback)
            page.addWidget(box)
        page.addStretch(1)
        scroll.setWidget(inner)
        layout.addWidget(scroll, 1)

    close = QtWidgets.QPushButton("Close")
    close.clicked.connect(dialog.accept)
    layout.addWidget(close)
    dialog.exec()
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    project_root = Path(argv[0]) if argv and Path(argv[0]).is_dir() else None
    return show_dialog(project_root)


if __name__ == "__main__":
    sys.exit(main())
