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
* ``scribbleprompt_checkpoint`` -> informational (where to put the .pt file)
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
            "title": "缺少外部 Python 环境",
            "detail": (
                "未找到内置 Python 环境（python_env/），所有 AI 功能都无法运行。"
                + ("检测到离线安装包，修复不需要联网。"
                   if bundle else
                   "请运行“安装/修复环境”，或如果本机从未安装过，重新解压部署包。")
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
                "title": "上次环境安装失败",
                "detail": "{0}\n技术细节：{1}".format(
                    state.get("message") or "安装 worker 报告了错误。",
                    str(state.get("error") or "")[:300],
                ),
                "fix_action": "install",
                "severity": "bad",
            })
        elif status == "incomplete":
            missing = state.get("failed_packages") or state.get("missing_packages") or []
            issues.append({
                "kind": "setup_incomplete",
                "title": "环境安装不完整",
                "detail": "{0}\n受影响的包：{1}".format(
                    state.get("message") or "部分依赖包未安装成功。",
                    ", ".join(str(m) for m in missing[:10]) or "未知",
                ),
                "fix_action": "install",
                "severity": "warn",
            })

    old_root = detect_old_root(root)
    if old_root:
        issues.append({
            "kind": "migration_pending",
            "title": "本目录似乎是从其他位置移动过来的",
            "detail": (
                "部分配置路径仍指向 {0}，该路径在本机不存在。请运行迁移，"
                "把路径改写为当前位置（{1}）。".format(old_root, root)
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
                "title": "FlexiCT 预训练权重未安装",
                "detail": (
                    "FlexiCT 训练需要预训练骨干文件 "
                    "flexict_2d/model.safetensors 和 flexict_3d/model.safetensors "
                    "（位于 {0}），或在 flexict_config.json 中配置 "
                    "pretrained_weights_dir。它们不随部署包分发"
                    "（2 个，各约 576 MB）；需要 FlexiCT 训练时，请从分发源复制。".format(
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
            "title": "nnInteractive 官方模型未安装",
            "detail": (
                "在以下位置均未找到 nnInteractive 官方模型"
                "（nnInteractive_v1.0，含 fold_*/checkpoint_final.pth）：\n{0}\n"
                "缺少它时“官方模型标注”入口无法运行（自定义训练的模型不受影响）。"
                "请把模型文件夹从分发源复制到上述任一位置，或设置 "
                "NNINTERACTIVE_MODEL_DIR / nninteractive_config.json 中的 "
                "model_dir。".format(
                    "\n".join(c for c in _nninteractive_model_candidates(root) if c)
                )
            ),
            "fix_action": "",
            "severity": "bad",
        })

    scribble = _scribbleprompt_checkpoint(root)
    if scribble is not None and not scribble:
        issues.append({
            "kind": "scribbleprompt_checkpoint",
            "title": "ScribblePrompt checkpoint 未安装",
            "detail": (
                "未找到涂鸦标注入口所配置的 ScribblePrompt checkpoint：\n{0}\n"
                "请从分发源复制该 .pt 文件，或设置 SCRIBBLEPROMPT_CHECKPOINT / "
                "interactive_algorithms_config.json 中 \"scribbleprompt\" 下的 "
                "checkpoint 项。".format(
                    _scribbleprompt_configured_path(root)
                )
            ),
            "fix_action": "",
            "severity": "warn",
        })

    return issues


def _scribbleprompt_configured_path(root: Path) -> str:
    """The checkpoint path as the runtime resolves it (env var > config >
    code default, relative paths anchored at the project root)."""
    value = os.environ.get("SCRIBBLEPROMPT_CHECKPOINT", "").strip()
    if not value:
        section = (
            (_read_json(root / "interactive_algorithms_config.json", {}) or {})
            .get("scribbleprompt") or {}
        )
        value = str(section.get("checkpoint") or "").strip()
    # Mirror the runtime's code default (interactive_algorithms_mimics.
    # _checkpoint_path): the key was removed from the shipped JSON, so
    # without this default the health check would silently pass on a
    # broken install.
    if not value:
        value = "integrations/ScribblePrompt/checkpoints/ScribblePrompt_unet_v1_nf192_res128.pt"
    close = QtWidgets.QPushButton("关闭")
    close.clicked.connect(dialog.accept)
    buttons.addWidget(close)
    layout.addLayout(buttons)
    result = dialog.exec()
    if result == QtWidgets.QDialog.Rejected and recheck.underMouse():
        return show_dialog(root, parent)  # re-run detection and re-render
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    project_root = Path(argv[0]) if argv and Path(argv[0]).is_dir() else None
    return show_dialog(project_root)


if __name__ == "__main__":
    sys.exit(main())
