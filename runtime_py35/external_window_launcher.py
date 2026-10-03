# -*- coding: utf-8 -*-
"""Shared launcher for external PySide6 windows started from inside Mimics.

Every "open window X" Mimics action does the same thing: locate the external
tools Python, locate a window script under ``tools/``, start it as a
registered ``external_ui`` process with a stderr log, and return immediately
so the Mimics GUI never waits. The per-window launchers (drop-to-import,
system health, ...) are thin wrappers around :func:`open_external_window`,
which also maps startup failures to plain-language messages (environment
broken vs. installation incomplete) instead of raw tracebacks.
"""

from __future__ import print_function

import logging
import os
import time

import mimics

import runtime_common


def _project_root():
    return runtime_common.find_root(
        os.path.dirname(os.path.abspath(__file__)),
        ("nninteractive_config.json", "python_env", "nninteractive_env", ".git"),
    )


def error_guidance(exc):
    """Map a raw startup exception to annotator-facing guidance text.

    Returns a short plain-language sentence plus the original error for the
    log; callers embed it in their message boxes so an annotator never sees
    a bare traceback.
    """
    text = str(exc).lower()
    if "no such file" in text or "not found" in text or (
        "winerror 2" in text
    ):
        return (
            "The installation looks incomplete (a required file or Python is "
            "missing). Re-extract the deployment package, then retry. "
            "Technical detail: {0}".format(exc)
        )
    if "no module named" in text or "import" in text:
        return (
            "The external Python environment is damaged or outdated. Run "
            "Admin > Setup/Repair Environment, then retry. "
            "Technical detail: {0}".format(exc)
        )
    return "Technical detail: {0}".format(exc)


def find_window_script(root, script_name):
    """Return the tools/<script_name> path inside root (or its parent)."""
    candidates = [
        os.path.join(root, "tools", script_name),
        os.path.join(os.path.dirname(root), "tools", script_name),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return os.path.abspath(path)
    return os.path.abspath(candidates[0])


def prune_logs(runtime_dir, prefix, keep):
    """Keep only the newest ``keep`` ``<prefix>*.log`` files (best effort)."""
    try:
        logs = [
            os.path.join(runtime_dir, name)
            for name in os.listdir(runtime_dir)
            if name.startswith(prefix) and name.endswith(".log")
        ]
        if len(logs) <= keep:
            return
        logs.sort(key=lambda path: os.path.getmtime(path), reverse=True)
        for path in logs[keep:]:
            try:
                os.remove(path)
            except OSError:
                continue
    except Exception:
        pass


def open_external_window(
    script_name,
    window_title,
    runtime_subdir,
    log_prefix,
    log_keep=10,
    cleanup_policy=None,
    log_start_message=None,
):
    """Start an external window and return 0 on success, 1 on failure.

    ``script_name`` is a file under ``tools/``; ``runtime_subdir`` and
    ``log_prefix`` decide where the stderr log goes (``.mimics_runtime/
    <runtime_subdir>/<log_prefix><timestamp>.log``).
    """
    root = _project_root()
    python_exe = runtime_common.find_external_python(root, allow_system_python=False)
    if not python_exe or not os.path.isfile(python_exe):
        mimics.dialogs.message_box(
            "未找到外部工具 Python。\n\n"
            "请打开 管理菜单 > 环境指引 查看分步修复窗口，"
            "或运行 管理菜单 > 环境设置/修复。",
            title=window_title,
            ui_blocking=False,
        )
        return 1
    script = find_window_script(root, script_name)
    if not os.path.isfile(script):
        mimics.dialogs.message_box(
            "未找到窗口脚本：{0}\n"
            "安装似乎不完整；请重新解压部署包，或运行 管理菜单 > 环境设置/修复。".format(script),
            title=window_title,
            ui_blocking=False,
        )
        return 1
    runtime_dir = os.path.join(root, ".mimics_runtime", runtime_subdir)
    if not os.path.isdir(runtime_dir):
        os.makedirs(runtime_dir)
    prune_logs(runtime_dir, log_prefix, log_keep)
    stderr_log = os.path.join(
        runtime_dir,
        "{0}{1}.log".format(log_prefix, time.strftime("%Y%m%dT%H%M%S")),
    )
    try:
        process = runtime_common.launch_external_gui_process(
            [python_exe, script], cwd=root, stderr_log=stderr_log
        )
    except Exception as exc:
        mimics.dialogs.message_box(
            "无法启动窗口。外部 Python 环境可能已损坏（详情见下）；"
            "请运行 管理菜单 > 环境设置/修复 来修复。\n\n{0}".format(exc),
            title=window_title,
            ui_blocking=False,
        )
        return 1
    register_kwargs = {
        "parent_pid": os.getpid(),
        "state_path": stderr_log,
    }
    if cleanup_policy:
        register_kwargs["cleanup_policy"] = cleanup_policy
    runtime_common.register_process(
        root,
        "external_ui",
        process.pid,
        **register_kwargs
    )
    if log_start_message:
        mimics.logging.log_user_message(
            level=logging.INFO,
            message=log_start_message.format(pid=process.pid),
        )
    return 0
