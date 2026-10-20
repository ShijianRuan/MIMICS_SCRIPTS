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
* ``gpu_unavailable``     -> informational (double-click 修复GPU加速.bat
                             after closing Mimics; the torch switch cannot
                             run from inside this environment)
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


def _gpu_unavailable_issue(root: Path) -> dict:
    """The old-driver CUDA issue with the double-click repair path.

    The torch switch itself cannot be a fix_action started from here: it
    replaces DLLs of the very interpreter this dialog runs in. The repair
    is a launcher in the package root the user double-clicks after fully
    closing Mimics (修复GPU加速.bat wraps repair/cu118/switch_gpu_torch.py).
    """
    bat = root / "修复GPU加速.bat"
    if bat.is_file():
        repair = (
            "完全退出 Mimics 和右下角所有小窗口后，双击部署包根目录的"
            "「修复GPU加速.bat」（{0}），约 5 分钟、不需要联网，完成后重新"
            "打开 Mimics。".format(bat)
        )
    else:
        repair = "部署包根目录缺少「修复GPU加速.bat」，请重新完整拷贝部署包。"
    return {
        "kind": "gpu_unavailable",
        "title": "GPU 加速不可用（AI 功能会明显变慢）",
        "detail": (
            "上次环境检查显示 CUDA 不可用，最常见的原因是显卡驱动过旧。\n"
            "首选：请 IT 把 NVIDIA 驱动更新到最新版（nvidia.cn/drivers）并"
            "重启电脑，不用改包内任何东西。\n"
            "驱动实在无法更新时：{0}\n"
            "若这台电脑没有独立 NVIDIA 显卡，可忽略本条（AI 功能用 CPU 也能"
            "运行，只是慢）。".format(repair)
        ),
        "fix_action": "",
        "severity": "warn",
    }


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

    # GPU: only from the last check's recorded result. No live torch probe
    # here on purpose -- collect_issues also feeds the system health panel's
    # 10-second refresh, and importing torch on a network drive takes tens
    # of seconds (R84).
    if state:
        detail = state.get("detail")
        if isinstance(detail, dict) and detail.get("cuda_available") is False:
            issues.append(_gpu_unavailable_issue(root))

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
    if value and not os.path.isabs(value):
        value = str(root / value)
    return value


def _scribbleprompt_checkpoint(root: Path) -> str | None:
    """Resolved checkpoint path, or '' when missing.

    Returns None when no checkpoint is configured at all: an empty config
    is a different problem (not reported here) than a broken path."""
    value = _scribbleprompt_configured_path(root)
    if not value:
        return None
    return value if Path(value).is_file() else ""


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
            return "迁移已启动（PID {0}）。完成后请点击“重新检查”。".format(pid)
        return "无法启动迁移 worker。"
    if fix_action in ("check", "install", "offline-install", "setup-from-scratch"):
        pid = _run_setup_worker(root, fix_action)
        if pid:
            return "安装 worker 已启动（PID {0}）。完成后可点击“重新检查”确认。".format(pid)
    return ""


def _show_dialog_once(project_root: Path | None = None, parent=None) -> int:
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
    dialog.setWindowTitle("环境指引")
    dialog.setMinimumSize(560, 320)
    layout = QtWidgets.QVBoxLayout(dialog)

    header = QtWidgets.QLabel("环境指引")
    header.setObjectName("title")
    layout.addWidget(header)

    issues = collect_issues(root)
    if not issues:
        note = QtWidgets.QLabel(
            "未检测到环境问题。\nPython、依赖包、已配置路径、FlexiCT 权重和 nnInteractive 官方模型均正常。"
        )
        note.setWordWrap(True)
        layout.addWidget(note)
    else:
        intro = QtWidgets.QLabel(
            "有 {0} 个问题需要处理，处理后功能才能全部可用：".format(len(issues))
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
                fix_btn = QtWidgets.QPushButton("修复：{0}".format(
                    "运行迁移" if issue["fix_action"].startswith("migrate:")
                    else "运行安装/修复环境"
                ))
                fix_btn.setObjectName("primary")
                feedback = QtWidgets.QLabel("")
                feedback.setWordWrap(True)
                feedback.hide()

                def _clicked(_=False, action=str(issue["fix_action"]), feedback=feedback):
                    text = run_fix(root, action)
                    feedback.setText(text or "该问题没有可运行的操作。")
                    feedback.show()

                fix_btn.clicked.connect(_clicked)
                box_layout.addWidget(fix_btn)
                box_layout.addWidget(feedback)
            page.addWidget(box)
        page.addStretch(1)
        scroll.setWidget(inner)
        layout.addWidget(scroll, 1)

    # Re-check: repair workers run detached, so the user needs an in-place way
    # to see whether the environment actually got fixed (the alternative is
    # closing and reopening the whole dialog from the menu).
    buttons = QtWidgets.QHBoxLayout()
    recheck = QtWidgets.QPushButton("重新检查")
    recheck.clicked.connect(lambda: dialog.done(2))
    buttons.addWidget(recheck)
    buttons.addStretch(1)
    close = QtWidgets.QPushButton("关闭")
    close.clicked.connect(dialog.accept)
    buttons.addWidget(close)
    layout.addLayout(buttons)
    ui_theme.clamp_window_to_screen(dialog, (560, 320))
    result = dialog.exec()
    dialog.deleteLater()
    return 1 if result == 2 else 0


def show_dialog(project_root: Path | None = None, parent=None) -> int:
    """Re-check explicitly, including keyboard activation, without recursion."""
    while _show_dialog_once(project_root, parent):
        pass
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    project_root = Path(argv[0]) if argv and Path(argv[0]).is_dir() else None
    return show_dialog(project_root)


if __name__ == "__main__":
    sys.exit(main())
