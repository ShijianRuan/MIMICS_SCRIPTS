# -*- coding: utf-8 -*-
"""一键在默认 cu124、cu118（旧驱动兼容）与 cu128（RTX 50 系新显卡）torch 之间切换。

交付背景：部署包默认装 torch 2.6.0+cu124，需要 2024-03 之后的 NVIDIA
驱动，且最多支持到 RTX 40 系（sm_90）。接收方机器分两种常见情况：

1. 驱动较旧（但仍 ≥ 2019 年驱动，即 R45x 以后）：换成 cu118 版本即可
   恢复 GPU 加速（使用「修复GPU加速.bat」）。
2. 显卡太新（RTX 50 系 / Blackwell，sm_120）：cu124 版本没有 50 系的
   内核，报"CUDA 可用但跑不起来/没有内核映像"。换成 cu128 版本
   （torch 2.7.1+cu128，支持到 sm_120）即可（使用「RTX50系列GPU修复.bat」）。

本脚本负责替换与回滚，替换时把当前 torch/torchvision 目录改名为 .bak
保留（不复制，不占双倍空间），随时可恢复默认状态。

用法：
  python switch_gpu_torch.py           检查并切换到 cu118（旧驱动）
  python switch_gpu_torch.py switch128 检查并切换到 cu128（RTX 50 系）
  python switch_gpu_torch.py restore   切回默认 cu124

运行前必须关闭 Mimics 和所有相关小工具（拖拽导入窗口等），否则进程
锁住的 DLL 会让替换中途卡死并损坏环境——脚本启动时会自动检查。
"""

from __future__ import print_function

import argparse
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# HERE = <部署包>/repair/cu118
ENV_SITE = os.path.normpath(os.path.join(HERE, "..", "..", "python_env", "Lib", "site-packages"))
PYTHON = os.path.normpath(os.path.join(HERE, "..", "..", "python_env", "python.exe"))
C128_DIR = os.path.normpath(os.path.join(HERE, "..", "cu128"))

# (wheel dir, pip pins) 每个变体一套
VARIANTS = {
    "cu118": (
        HERE,
        {"torch": "torch-2.6.0+cu118-cp313-cp313-win_amd64.whl",
         "torchvision": "torchvision-0.21.0+cu118-cp313-cp313-win_amd64.whl"},
        {"torch": "torch==2.6.0+cu118", "torchvision": "torchvision==0.21.0+cu118"},
    ),
    "cu128": (
        C128_DIR,
        {"torch": "torch-2.7.1+cu128-cp313-cp313-win_amd64.whl",
         "torchvision": "torchvision-0.22.1+cu128-cp313-cp313-win_amd64.whl"},
        {"torch": "torch==2.7.1+cu128", "torchvision": "torchvision==0.22.1+cu128"},
    ),
}


def _log(msg):
    print(msg)


def _run(args, **kw):
    proc = subprocess.Popen(args, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, **kw)
    out, _ = proc.communicate()
    return proc.returncode, out.decode("utf-8", "replace")


def check_no_env_users():
    """返回正在使用本部署包 python_env 的其他进程列表。

    运行中的进程会锁住 torch DLL（网络盘上表现为删除挂起，替换中途
    失败并留下残缺目录），切换前必须全部退出。
    """
    root = os.path.normpath(os.path.join(HERE, "..", "..")).lower().replace("/", "\\")
    ps = (
        "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
        "ForEach-Object { \"{0}|{1}\" -f $_.ProcessId, $_.CommandLine }"
    )
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, timeout=60
                             ).stdout.decode("utf-8", "replace")
    except Exception as e:
        return None, "无法检查运行中的进程（{0}），请手动确认后重试".format(e)
    holders = []
    my_pid = str(os.getpid())
    for line in out.splitlines():
        line = line.strip()
        if "|" not in line:
            continue
        pid, cmd = line.split("|", 1)
        if pid == my_pid:
            continue
        if root in cmd.lower().replace("/", "\\"):
            holders.append((pid, cmd[:120]))
    return holders, None


def cuda_probe():
    code = (
        "import json, traceback\n"
        "r={}\n"
        "try:\n"
        "    import torch\n"
        "    r['torch']=str(torch.__version__)\n"
        "    r['cuda_build']=str(torch.version.cuda)\n"
        "    r['avail']=bool(torch.cuda.is_available())\n"
        "    r['arch']=list(torch.cuda.get_arch_list())\n"
        "    try:\n"
        "        r['name']=str(torch.cuda.get_device_name(0))\n"
        "        r['cap']=[int(x) for x in torch.cuda.get_device_capability(0)]\n"
        "    except Exception as e:\n"
        "        r['dev_error']=repr(e)\n"
        "except Exception as e:\n"
        "    r['error']='{0}: {1}'.format(type(e).__name__, e)\n"
        "print(json.dumps(r))\n"
    )
    rc, out = _run([PYTHON, "-c", code])
    if rc != 0:
        return {"probe_failed": out.strip()[:300]}
    try:
        return json.loads(out.strip().splitlines()[-1])
    except Exception:
        return {"parse_failed": out.strip()[:300]}


def is_blackwell(probe):
    """RTX 50 系 / Blackwell：计算能力 12.0 以上，或型号名带 50 系字样。"""
    cap = probe.get("cap")
    if cap and len(cap) >= 1:
        return cap[0] >= 12
    name = (probe.get("name") or "").lower()
    return any(k in name for k in (
        "rtx 50", " 5090", " 5080", " 5070", " 5060", "blackwell"))


def installed_local_version(dist):
    """从 dist-info 目录读取 local 版本后缀（cu124 / cu118 / cu128）。"""
    display_name = dist.replace(".dist-info", "")
    # torch-2.6.0+cu124 -> +cu124
    if "+" in display_name:
        return display_name.split("+", 1)[1]
    return display_name.rsplit("-", 1)[-1]


def find_installed():
    if not os.path.isdir(ENV_SITE):
        _log("!! 找不到 site-packages: {0}".format(ENV_SITE))
        return None
    infos = sorted(
        d for d in os.listdir(ENV_SITE)
        if d.endswith(".dist-info") and (
            d.startswith("torch-") or d.startswith("torchvision-"))
    )
    versions = {}
    for d in infos:
        base = "torch" if d.startswith("torch-") and not d.startswith("torchvision") else "torchvision"
        versions[base] = installed_local_version(d)
    return versions


def _remove(path):
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    else:
        os.remove(path)


def backup_current(name):
    """把 site-packages 下 torch/torchvision 相关目录改名 .bak，不复制。"""
    targets = []
    for entry in os.listdir(ENV_SITE):
        if name == "torch" and entry == name:
            targets.append(entry)
        elif name == "torch" and entry.startswith("torch-") and not entry.startswith("torchvision"):
            targets.append(entry)
        elif name == "torchvision" and (entry == name or entry.startswith("torchvision-")):
            targets.append(entry)
    renamed = []
    for entry in targets:
        src = os.path.join(ENV_SITE, entry)
        dst = os.path.join(ENV_SITE, entry + ".bak")
        if os.path.exists(dst):
            _remove(dst)
        os.rename(src, dst)
        renamed.append((src, dst))
    return renamed


def restore_backups():
    """把本脚本创建的 .bak 备份改回原名（其他来源的 .bak 不动）。"""
    restored = []
    for entry in os.listdir(ENV_SITE):
        if not entry.endswith(".bak"):
            continue
        base = entry[:-4]  # 去掉 .bak
        if not (base == "torch" or base == "torchvision"
                or (base.startswith("torch-") and not base.startswith("torchvision"))
                or base.startswith("torchvision-")):
            continue
        src = os.path.join(ENV_SITE, entry)
        dst = os.path.join(ENV_SITE, base)
        if os.path.exists(dst):
            _remove(dst)
        os.rename(src, dst)
        restored.append(dst)
    return restored


def has_pending_backup():
    """有尚未恢复的 .bak（上次切换后没 restore），此时不能再次切换。"""
    if not os.path.isdir(ENV_SITE):
        return False
    for entry in os.listdir(ENV_SITE):
        if not entry.endswith(".bak"):
            continue
        base = entry[:-4]  # 去掉 .bak
        if (base == "torch" or base == "torchvision"
                or (base.startswith("torch-") and not base.startswith("torchvision"))
                or base.startswith("torchvision-")):
            return True
    return False


def remove_variant_files():
    """切回默认时清掉已装的 cu118/cu128 torch/torchvision（.bak 备份保留）。"""
    removed = []
    for entry in list(os.listdir(ENV_SITE)):
        if entry.endswith(".bak"):
            continue  # 备份由 restore_backups() 恢复，这里绝不能删
        if entry in ("torch", "torchvision"):
            _remove(os.path.join(ENV_SITE, entry))
            removed.append(entry)
        elif (entry.startswith("torch-") and not entry.startswith("torchvision")) \
                or entry.startswith("torchvision-"):
            _remove(os.path.join(ENV_SITE, entry))
            removed.append(entry)
    return removed


def pip_install_torch(wheels_dir, pins):
    """离线从 wheels_dir 安装指定 torch/torchvision 版本。"""
    args = [PYTHON, "-m", "pip", "install", "--no-index", "--no-deps",
            "--find-links", wheels_dir,
            pins["torch"], pins["torchvision"]]
    _log("运行: " + " ".join(args))
    rc, out = _run(args)
    _log(out)
    return rc == 0


def _guard_ready(variant):
    """切换前的通用检查：无占用进程、离线 wheel 齐全。

    备份/当前版本状态不在这里挡——由 _switch() 统一处理（可安全地先恢复
    默认再切换），避免把"当前已是目标版本"或"可先恢复默认"误当错误拒绝。
    """
    holders, err = check_no_env_users()
    if err:
        _log("!! " + err)
        return False
    if holders:
        _log("!! 以下进程正在使用本环境，请先全部关闭（Mimics、拖拽导入小窗口等）再运行：")
        for pid, cmd in holders:
            _log("   PID {0}: {1}".format(pid, cmd))
        return False
    wheels_dir, wheels, pins = VARIANTS[variant]
    missing = [name for name, wheel in wheels.items()
               if not os.path.isfile(os.path.join(wheels_dir, wheel))]
    if missing:
        _log("!! 缺少离线 wheel: {0}（应位于 {1}）".format(missing, wheels_dir))
        return False
    return True


def _switch(variant):
    wheels_dir, _wheels, pins = VARIANTS[variant]

    # 先看当前是否已是目标变体：50 系机器重复执行不重装、不产生嵌套备份。
    current = find_installed() or {}
    current_torch = current.get("torch")
    if current_torch == variant:
        _log("当前已在使用 {0}，无需重复切换。".format(variant))
        return 0

    probe = cuda_probe()
    _log("当前 GPU 状态: {0}".format(json.dumps(probe, ensure_ascii=False)))

    if variant == "cu118":
        if probe.get("avail"):
            _log("CUDA 已可用，无需切换到 cu118。")
            return 0
    else:  # cu128 —— 只针对 RTX 50 系
        if not is_blackwell(probe):
            if probe.get("avail"):
                _log("CUDA 可用，且显卡不是 RTX 50 系（无需 cu128）。")
            else:
                _log("当前 CUDA 不可用。cu128 修复针对 RTX 50 系新显卡；"
                     "若显卡驱动较旧请改用「修复GPU加速.bat」（cu118），"
                     "或先更新 NVIDIA 驱动后再试。")
            return 0

    if current_torch in ("cu118", "cu128"):
        # 另一个变体处于活动状态：先恢复默认 cu124，再执行本次切换，
        # 保证 .bak 里备份的始终是默认版本，restore 后回到 cu124。
        if has_pending_backup():
            _log("检测到当前使用 {0}，先恢复默认 cu124 再切换…".format(current_torch))
            remove_variant_files()
            restore_backups()
        else:
            _log("!! 当前是非默认版本（{0}）但没有可恢复的备份，环境状态异常；"
                 "请先运行「切回默认GPU.bat」，若仍异常请重新解压部署包。".format(current_torch))
            return 1

    _log("=> 切换中...")
    backup_current("torch")
    backup_current("torchvision")
    if not pip_install_torch(wheels_dir, pins):
        _log("!! 安装失败，尝试回滚...")
        remove_variant_files()
        restore_backups()
        return 1
    probe2 = cuda_probe()
    _log("切换后 GPU 状态: {0}".format(json.dumps(probe2, ensure_ascii=False)))
    if probe2.get("avail"):
        _log("成功：现在使用 {0}，GPU 可用。".format(probe2.get("torch", variant)))
        _log("若想切回默认 cu124：python \"{0}\" restore".format(os.path.abspath(__file__)))
    else:
        _log("警告：切换完成但 CUDA 仍不可用。"
             "若驱动早于 2019 年（R45x 之前），此机器将只能用 CPU 运行。")
    return 0


def main():
    parser = argparse.ArgumentParser(description="切换 torch 的 CUDA 版本")
    parser.add_argument("action", nargs="?", default="switch",
                        choices=["switch", "switch128", "restore"],
                        help="switch=切到 cu118（旧驱动）；switch128=切到 cu128（RTX 50 系）；restore=切回默认 cu124")
    args = parser.parse_args()

    if not os.path.isdir(ENV_SITE):
        _log("!! 找不到 site-packages: {0}".format(ENV_SITE))
        return 1

    versions = find_installed()
    _log("site-packages: {0}".format(ENV_SITE))
    _log("当前 torch 版本: {0}".format(versions or "未找到"))

    if args.action == "restore":
        current_torch = (versions or {}).get("torch")
        if current_torch not in ("cu118", "cu128") and not has_pending_backup():
            _log("当前已是默认版本（cu124），无需恢复。")
            probe = cuda_probe()
            _log("当前 GPU 状态: " + json.dumps(probe, ensure_ascii=False))
            return 0
        if current_torch in ("cu118", "cu128") and not has_pending_backup():
            # .bak 丢失（如部署目录被外部工具回退/覆盖）时绝不能把还能用的
            # torch 删掉——删了环境就彻底坏了。保留现状并指路重装。
            _log("!! 当前使用 {0}，但默认版本（cu124）的备份 .bak 丢失，"
                 "无法恢复默认。为避免删坏环境，本次不做任何改动。".format(current_torch))
            _log("   请重新拷贝/解压一份完整部署包覆盖本目录，"
                 "或先用对应修复 bat 重装后再运行本脚本 restore。")
            return 1
        removed = remove_variant_files()
        restored = restore_backups()
        _log("已移除 cu118/cu128（{0} 项）并恢复默认: {1}".format(
            len(removed), restored or "无备份可恢复"))
        probe = cuda_probe()
        _log("恢复后检查: " + json.dumps(probe, ensure_ascii=False))
        return 0

    if not os.path.isdir(os.path.join(ENV_SITE, "torch")):
        _log("!! 未找到已安装的 torch，终止。")
        return 1

    variant = "cu118" if args.action == "switch" else "cu128"
    if not _guard_ready(variant):
        return 1
    return _switch(variant)


if __name__ == "__main__":
    sys.exit(main())
