# -*- coding: utf-8 -*-
"""SLIP 交互标注 —— Mimics (Python 3.5) 侧的会话管理.

将 integrations/slip 中的 SLIP 交互分割模型通过 slip_bridge.py 接入
Mimics。会话协议沿用 nninteractive_mimics.py 的异步模式并复用其助手
函数；与 nnInteractive 的主要差异：

- undo/reset 是 worker 原生命令，客户端无需重放提示；
- job 目录即 worker 目录（initialize.json / commands/ / results/ /
  worker_status.json / close.json 同处一目录）；
- 结果 Mask 的写入方式在会话开始时就确定（in_place / derived_copy /
  new_empty），不存在 choose_on_first_result。

worker 由外部 Python 运行 slip_bridge.py --async-worker <job_dir> 启动，
通过共享 gpu.lock 与 nnInteractive 互斥（任一方持有锁时另一方拒绝
启动）。协议细节见 slip_bridge.py（slip_async_job.v1）。
"""

from __future__ import print_function

import logging
import os
import shutil
import time
import traceback
import uuid

import mimics
import runtime_common
import nninteractive_mimics as nnm


TITLE = "SLIP 交互标注"
BUTTON_INCLUDE_POINT = "包含点"
BUTTON_EXCLUDE_POINT = "排除点"
BUTTON_UNDO = "撤销"
BUTTON_RESET = "重置"
BUTTON_FINISH = "完成"
BUTTON_START_CURRENT = "从当前 Mask 开始"
BUTTON_APPLY_EMPTY = "应用空结果"
BUTTON_KEEP = "不应用"
BUTTON_RETRY = "重试"
BUTTON_DISCARD_SESSION = "放弃 AI 会话"
BUTTON_CANCEL = "取消"

ASYNC_JOB_METADATA = "slip.async_job_path"
DEFAULT_RESULT_NAME = "SLIP Result"

# 与 slip_bridge.py 的 SUPPORTED_IMAGE_DTYPES 保持一致（Mimics int8/int64
# 缓冲区不被 SLIP 支持，在导出前就拒绝，避免 worker 启动后才失败）。
SUPPORTED_DTYPES = ("uint8", "int16", "uint16", "int32", "uint32", "float32")

# worker 启动（模型加载 + patch 预计算）在参考 GPU 上约 45 秒；此后每次
# 交互数秒内完成。600 秒的超时覆盖首条 predict 命令的完整等待。
RESULT_TIMEOUT_SECONDS = 600.0
POLL_SECONDS = 0.25
IDLE_TIMEOUT_SECONDS = 600
RETENTION_DAYS = 3

_ASYNC_MONITORS = {}
_ASYNC_VISUAL_OBJECTS = {}
_SESSION_WRITE_MODE = None

_log = nnm._mimics_log
_project_root = nnm._project_root


# ---------------------------------------------------------------------------
# 影像导出（GV 缓冲区 -> SLIP 强度空间）
# ---------------------------------------------------------------------------

def _buffer_intensity_transform(image):
    """决定 GV 缓冲区 -> SLIP 强度空间的线性变换。

    SLIP 论文 Table 4：所有模态（CT/MRI/PET/US/Microscopy）共用同一强度
    管线（Clamp [-1000,1000] + z-score），模型本身不限模态。因此按模态
    导出与 nnInteractive 相同依据的强度：CT 恢复 HU（GV = slope*HU +
    intercept 的逆变换）；MR 尽量恢复 DICOM 存储值（复用
    nnm._source_intensity_mapping）；其余模态 GV 直通——z-score 对 clamp
    边界内的线性缩放不变，直通不改变推理结果。

    返回 (slope, intercept, basis, modality)；CT 无法恢复 HU 时 slope 为
    None（调用方负责拒绝）。
    """
    modality = nnm._image_modality(image)
    kind = nnm._metadata_get(image, nnm.SOURCE_IMAGE_KIND_METADATA, "")
    if nnm._source_uses_hu_to_gv(kind, modality):
        transform = nnm._hu_to_mimics_gv_transform()
        if transform is None:
            return (None, None, "mimics_hu_gv", modality)
        slope, intercept = transform
        return (1.0 / float(slope), -float(intercept) / float(slope),
                "mimics_hu_gv", modality)
    if modality == "MR":
        mapping = nnm._source_intensity_mapping(image, modality=modality)
        if mapping is not None:
            return (float(mapping["slope"]), float(mapping["intercept"]),
                    mapping["basis"], modality)
    return (1.0, 0.0, "gv_passthrough", modality)


def _validate_image(image):
    """Fail fast on inputs SLIP cannot process, before any state is created."""
    view = image.get_voxel_buffer()
    dtype = nnm._buffer_dtype(view)
    if dtype not in SUPPORTED_DTYPES:
        raise RuntimeError(
            "当前影像缓冲区的数据类型（{0}）不被 SLIP 支持（支持：{1}）。".format(
                dtype, ", ".join(SUPPORTED_DTYPES)))
    slope, _intercept, _basis, _modality = _buffer_intensity_transform(image)
    if slope is None:
        raise RuntimeError(
            "无法从 Mimics 获取 HU 灰度映射（HU2GV）。SLIP 处理 CT 影像需要"
            " HU 值进行推理；请检查影像是否为 CT。")


def _export_image(image, path):
    """Write the GV voxel buffer and return the buffer/intensity export dict.

    to_hu_slope/to_hu_intercept 是 slip_async_job.v1 协议的历史键名，
    语义为"缓冲区 -> 模型原生强度"（CT 即 HU 逆变换，其余模态见
    _buffer_intensity_transform；桥接层对直通值 1.0/0.0 零成本直通）。
    """
    started = time.time()
    view = image.get_voxel_buffer()
    with open(path, "wb") as handle:
        nnm._stream_buffer(
            nnm._buffer_byte_view(view), compute_sha=False, handle=handle)
    to_model_slope, to_model_intercept, basis, modality = (
        _buffer_intensity_transform(image))
    result = {
        "path": path,
        "shape": [int(value) for value in view.shape],
        "dtype": nnm._buffer_dtype(view),
        "to_hu_slope": to_model_slope,
        "to_hu_intercept": to_model_intercept,
    }
    _log(
        logging.INFO,
        "SLIP image buffer exported in {0:.1f}s ({1} {2}, modality {3}, "
        "intensity basis {4}, buffer->model slope {5:.6g}, intercept "
        "{6:.6g}).".format(
            time.time() - started,
            "x".join(str(v) for v in result["shape"]),
            result["dtype"],
            modality or "unknown",
            basis,
            result["to_hu_slope"],
            result["to_hu_intercept"],
        ),
    )
    return result


# ---------------------------------------------------------------------------
# job 状态（job_dir 即 worker_dir，见 slip_bridge.py）
# ---------------------------------------------------------------------------

def _jobs_root(root=None):
    return os.path.join(
        root or _project_root(), ".mimics_runtime", "slip", "async_jobs")


def _job_state_path(job_dir):
    return os.path.join(job_dir, "job.json")


def _load_job(target):
    job_dir = nnm._metadata_get(target, ASYNC_JOB_METADATA, "")
    if not job_dir:
        return None
    state = runtime_common.read_json(_job_state_path(job_dir))
    if not state:
        nnm._metadata_delete(target, ASYNC_JOB_METADATA)
        return None
    if state.get("status") in (
        "closing", "closed", "discarded", "failed", "expired"
    ) or os.path.isfile(os.path.join(job_dir, "close.json")):
        nnm._metadata_delete(target, ASYNC_JOB_METADATA)
        return None
    state["_job_dir"] = job_dir
    return state


def _save_job(state):
    runtime_common.write_json_atomic(
        _job_state_path(state["_job_dir"]),
        dict((key, value) for key, value in state.items()
             if not key.startswith("_")),
    )


def _result_path(state, sequence):
    return os.path.join(
        state["_job_dir"], "results", "result_{0:06d}.json".format(int(sequence)))


def _worker_status(state):
    return runtime_common.read_json(
        os.path.join(state["_job_dir"], "worker_status.json"), {}) or {}


def _state_worker_alive(state):
    if not state:
        return False
    pid = state.get("pid")
    if not pid or not nnm._process_exists(pid):
        return False
    return _worker_status(state).get("status") not in (
        "closing", "closed", "failed", "expired")


def _find_own_slip_job(gpu_holder):
    """gpu.lock 持有者是本项目的 SLIP worker 时返回其 job 状态。

    监控定时器失效（脚本错误停止、Mimics 窗口关闭重开）后，用户重新
    运行 SLIP 入口是取回会话的唯一途径。这里用 gpu.lock 载荷里的 pid
    与 job.json 里记录的 worker pid 匹配来识别“自己的”会话。
    """
    if not gpu_holder:
        return None
    try:
        holder_pid = int(gpu_holder.get("pid"))
    except (TypeError, ValueError):
        return None
    if holder_pid <= 0:
        return None
    jobs_root = _jobs_root()
    if not os.path.isdir(jobs_root):
        return None
    for child in sorted(os.listdir(jobs_root)):
        job_dir = os.path.join(jobs_root, child)
        if os.path.isfile(os.path.join(job_dir, "close.json")):
            continue
        state = runtime_common.read_json(_job_state_path(job_dir))
        if not state:
            continue
        if state.get("status") in (
                "closing", "closed", "discarded", "failed", "expired"):
            continue
        try:
            if int(state.get("pid") or 0) != holder_pid:
                continue
        except (TypeError, ValueError):
            continue
        state["_job_dir"] = job_dir
        if _state_worker_alive(state):
            return state
    return None


def _session_target_mask(image, state):
    """按 target_guid + 会话元数据找到当前影像上该会话的结果 Mask。"""
    for mask in nnm._masks_for_image(image):
        if nnm._object_id(mask) != state.get("target_guid"):
            continue
        if nnm._metadata_get(
                mask, ASYNC_JOB_METADATA, "") != state.get("_job_dir"):
            continue
        return mask
    return None


def _enqueue_command(state, payload):
    sequence = nnm._next_async_sequence(state["_job_dir"])
    command = dict(payload)
    command["sequence"] = sequence
    command["command_id"] = "cmd{0:06d}".format(sequence)
    runtime_common.write_json_atomic(
        os.path.join(
            state["_job_dir"], "commands", "command_{0:06d}.json".format(sequence)),
        command,
    )
    if payload.get("action") == "predict":
        state["last_command"] = dict(payload)
    state["pending_sequence"] = sequence
    state["status"] = "running"
    state["updated_at_epoch"] = time.time()
    _save_job(state)
    return sequence


def _delete_visual_objects(state):
    job_dir = state.get("_job_dir") if state else None
    if not job_dir:
        return
    for obj in _ASYNC_VISUAL_OBJECTS.pop(job_dir, []):
        nnm._delete_mimics_object(obj)


def _start_job(image, target, source=None, write_mode="in_place",
               auto_created=False):
    """Create the job directory, snapshot inputs, and spawn the worker.

    与 nnInteractive 不同，worker 在会话开始时就启动（而不是第一条提示
    之后），让模型加载与用户的读图时间重叠；用户提交第一条提示时推理
    已就绪。
    """
    root = _project_root()
    python_exe = runtime_common.find_external_python(root)
    if not python_exe:
        raise RuntimeError(
            "未找到外部 Python 环境（python_env）。SLIP 依赖 GPU 推理，"
            "需要外部环境运行。")
    bridge_script = os.path.join(root, "slip_bridge.py")
    if not os.path.isfile(bridge_script):
        raise RuntimeError("未找到 slip_bridge.py：{0}".format(bridge_script))
    checkpoint = os.path.join(root, "slip_models", "SLIP_ckpt.pth")
    if not os.path.isfile(checkpoint):
        raise RuntimeError(
            "未找到 SLIP 模型文件：{0}\n"
            "请先按部署说明放置 SLIP_ckpt.pth。".format(checkpoint))

    jobs_root = _jobs_root(root)
    if not os.path.isdir(jobs_root):
        os.makedirs(jobs_root)
    nnm._cleanup_async_jobs(jobs_root, RETENTION_DAYS)

    job_id = "job_{0}_{1}".format(
        time.strftime("%Y%m%dT%H%M%S"), uuid.uuid4().hex[:10])
    job_dir = os.path.join(jobs_root, job_id)
    for sub in ("inputs", "commands", "results"):
        os.makedirs(os.path.join(job_dir, sub))
    token = runtime_common.try_acquire_local_operation(
        "mask_buffer_access", "SLIP input snapshot")
    if not token:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise RuntimeError(
            "另一个 Mimics-Script 任务正在使用 Mask 缓冲区，无法获取一致"
            "的快照。请稍后重试。")
    try:
        try:
            image_export = _export_image(
                image, os.path.join(job_dir, "inputs", "image.raw"))
            shape = image_export["shape"]
            base_export = nnm._export_mask(
                source, os.path.join(job_dir, "inputs", "target_at_start.u8"),
                shape)
            if list(base_export["shape"]) != list(shape):
                raise RuntimeError(
                    "源 Mask 与影像的形状不一致（{0} vs {1}）。".format(
                        "x".join(str(v) for v in base_export["shape"]),
                        "x".join(str(v) for v in shape)))
            target_sha256 = nnm._mask_sha256(target, shape)
        finally:
            runtime_common.release_local_operation("mask_buffer_access", token)
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise

    runtime_common.write_json_atomic(
        os.path.join(job_dir, "initialize.json"),
        {
            "device": "auto",
            "checkpoint_path": checkpoint,
            "image": {
                "buffer_path": image_export["path"],
                "shape": shape,
                "dtype": image_export["dtype"],
                "to_hu_slope": image_export["to_hu_slope"],
                "to_hu_intercept": image_export["to_hu_intercept"],
            },
            "parent_pid": os.getpid(),
            "idle_timeout_seconds": IDLE_TIMEOUT_SECONDS,
            "poll_seconds": 0.5,
        },
    )
    process, worker_log_path = nnm._start_async_worker(
        python_exe, bridge_script, job_dir)
    now = time.time()
    state = {
        "_job_dir": job_dir,
        "schema_version": "slip_async_job.v1",
        "job_id": job_id,
        "status": "initializing",
        "worker_dir": job_dir,
        "image_guid": nnm._object_id(image),
        "image_name": str(getattr(image, "name", "") or ""),
        "launch_project_path": nnm._current_project_path(),
        "target_guid": nnm._object_id(target),
        "target_name": str(getattr(target, "name", "") or ""),
        "source_guid": nnm._object_id(source),
        "source_name": str(getattr(source, "name", "") or ""),
        "write_mode": write_mode,
        "auto_created": bool(auto_created),
        "shape": shape,
        "base_path": base_export["path"],
        "base_pixel_count": base_export["pixel_count"],
        "expected_target_sha256": target_sha256,
        "interactions": [],
        "action_history_len": 0,
        "pending_sequence": None,
        "applied_sequence": 0,
        "created_at_epoch": now,
        "updated_at_epoch": now,
        "pid": process.pid,
        "worker_log": worker_log_path,
    }
    _save_job(state)
    nnm._metadata_set(target, ASYNC_JOB_METADATA, job_dir)
    _log(
        logging.INFO,
        "SLIP worker started (pid {0}); the model loads in the background "
        "while you place prompts.".format(process.pid),
    )
    return state


def _close_job(target, state, reason):
    auto_created = False
    empty_session = False
    if state:
        job_dir = state["_job_dir"]
        _delete_visual_objects(state)
        try:
            runtime_common.write_json_atomic(
                os.path.join(job_dir, "close.json"),
                {"reason": reason, "requested_at_epoch": time.time()})
        except Exception:
            pass
        state["status"] = "closing"
        state["closed_reason"] = reason
        state["updated_at_epoch"] = time.time()
        _save_job(state)
        _stop_monitor(job_dir)
        auto_created = bool(state.get("auto_created"))
        empty_session = not state.get("interactions")
    if target is not None:
        nnm._metadata_delete(target, ASYNC_JOB_METADATA)
        if (auto_created and empty_session and reason == "user_finished"
                and nnm._mask_is_empty(target)):
            # 自动创建但从未收到任何提示的空草稿不留垃圾（铁律 4）。
            nnm._delete_mimics_object(target)


# ---------------------------------------------------------------------------
# 提示采集与目标 Mask 选择
# ---------------------------------------------------------------------------

def _capture_point(image, include):
    """采集一个体素点击点；用户取消时返回 None。"""
    try:
        coordinates = mimics.indicate_coordinate(
            message="在目标结构内部点击一个点。"
            if include else "点击一个要从结果中排除的点。",
            show_message_box=True,
            confirm=False,
            title=TITLE,
        )
    except mimics.UserInterrupted:
        return None
    indexes = image.get_voxel_indexes(coordinates)
    marker = None
    try:
        marker = mimics.analyze.create_point(
            point=nnm._point_coordinates(coordinates),
            name="SLIP Prompt {0} Point".format(
                "Include" if include else "Exclude"),
            color=(0.1, 1.0, 0.2) if include else (1.0, 0.2, 0.1),
        )
    except Exception:
        marker = None
    return {
        "point": [int(value) for value in indexes],
        "label": 1 if include else 0,
        "_marker": marker,
    }


def _ask_write_mode():
    """非空手动 Mask 的写入方式：每次 Mimics 会话只问一次（R61-10）。"""
    global _SESSION_WRITE_MODE
    if _SESSION_WRITE_MODE is None:
        decision = mimics.dialogs.question_box(
            title="SLIP 写入方式",
            message=(
                "所选 Mask 已有内容。SLIP 结果如何写入？\n\n"
                "更新所选 Mask 直接修改它。\n"
                "创建可编辑副本 保持其不变，并新建一个可编辑的 AI 草稿。\n\n"
                "本次选择将在本次 Mimics 会话中一直记住。"
            ),
            buttons="更新所选 Mask;创建可编辑副本",
            ui_blocking=True,
        )
        # 对话框被直接关闭时选择保守的副本方式，保护原 Mask。
        _SESSION_WRITE_MODE = (
            "in_place" if decision == "更新所选 Mask" else "derived_copy")
    return _SESSION_WRITE_MODE


def _select_session_target(image):
    """解析会话的 source/target（见 nnm._select_session_masks）。

    与 nnInteractive 不同，写入方式在会话开始时就确定，因此非空手动
    Mask 立即询问，而不是等第一个结果。
    """
    selected = [
        mask for mask in nnm._masks_for_image(image)
        if bool(getattr(mask, "selected", False))
    ]
    if len(selected) > 1:
        raise RuntimeError("请只选择一个源 Mask 或 AI 草稿 Mask，然后重新运行 SLIP。")
    if not selected:
        target = nnm._create_result_mask(image, DEFAULT_RESULT_NAME)
        nnm._mark_ai_draft(target)
        return {
            "source": target,
            "target": target,
            "auto_created": True,
            "write_mode": "new_empty",
        }

    source = selected[0]
    try:
        source_image = getattr(source, "image", None)
    except Exception:
        source_image = None
    if source_image is None:
        try:
            source.image = image
        except Exception as error:
            try:
                bound_image = getattr(source, "image", None)
            except Exception:
                bound_image = None
            if bound_image is None or not nnm._same_object(bound_image, image):
                raise RuntimeError(
                    "所选 Mask 未绑定到当前激活的影像序列：{0}".format(error))
    elif not nnm._same_object(source_image, image):
        raise RuntimeError(
            "所选 Mask 属于另一个影像序列。请激活其影像序列或选择其他 Mask。")

    if nnm._metadata_get(source, nnm.ASYNC_JOB_METADATA, ""):
        raise RuntimeError(
            "所选 Mask 有一个进行中的 nnInteractive AI 会话。"
            "请先完成或放弃该会话，再运行 SLIP。")
    if nnm._metadata_get(source, ASYNC_JOB_METADATA, ""):
        return {
            "source": source,
            "target": source,
            "auto_created": False,
            "write_mode": "in_place",
        }
    if nnm._mask_is_empty(source) or nnm._is_ai_draft(source):
        return {
            "source": source,
            "target": source,
            "auto_created": False,
            "write_mode": "in_place",
        }
    if _ask_write_mode() == "in_place":
        return {
            "source": source,
            "target": source,
            "auto_created": False,
            "write_mode": "in_place",
        }
    draft = nnm._create_result_mask(
        image, "{0} - AI Draft".format(
            str(getattr(source, "name", "") or DEFAULT_RESULT_NAME)))
    nnm._mark_ai_draft(draft, source)
    try:
        source.selected = False
    except Exception:
        pass
    return {
        "source": source,
        "target": draft,
        "auto_created": True,
        "write_mode": "derived_copy",
    }


# ---------------------------------------------------------------------------
# 提示菜单
# ---------------------------------------------------------------------------

def _prompt_menu(target, state):
    count = int(state.get("action_history_len") or 0)
    buttons = [BUTTON_INCLUDE_POINT, BUTTON_EXCLUDE_POINT]
    if count:
        buttons.extend([BUTTON_UNDO, BUTTON_RESET])
    buttons.append(BUTTON_FINISH)
    apply_note = "结果就绪后将自动应用（当前方式：{0}）。".format(
        "更新所选 Mask"
        if state.get("write_mode") == "in_place"
        else "写入 AI 草稿 {0}".format(str(getattr(target, "name", "") or "")))
    load_note = ""
    if not state.get("applied_sequence") and not state.get("interactions"):
        load_note = "\n首次推理前模型在后台加载（约 30-60 秒）。"
    return mimics.dialogs.question_box(
        message=(
            "AI 结果 Mask：{0}\n"
            "本 AI 会话中的提示数：{1}\n\n"
            "提交提示后会启动后台推理，并立即将控制权交还给 Mimics。{2}\n{3}"
        ).format(
            str(getattr(target, "name", "") or ""),
            count,
            apply_note,
            load_note,
        ),
        buttons=";".join(buttons),
        title=TITLE,
        ui_blocking=True,
    )


def _continue_prompt(image, target, state):
    """显示提示菜单；返回 True 表示后台监控已接手（控制权交还 Mimics）。"""
    while True:
        action = _prompt_menu(target, state)
        if not action or action == BUTTON_FINISH:
            _close_job(target, state, "user_finished")
            _log(logging.INFO, "SLIP session finished by the user.")
            return False
        if action == BUTTON_UNDO:
            if not state.get("action_history_len"):
                mimics.dialogs.message_box(
                    "当前没有可撤销的提示。", title=TITLE, ui_blocking=False)
                continue
            sequence = _enqueue_command(state, {"action": "undo"})
            mimics.dialogs.message_box(
                "正在撤销上一次提示。撤销需在后台重新组合 Mask（约需数秒），"
                "完成后结果会自动应用；期间 Mimics 可正常操作。",
                title=TITLE, ui_blocking=False)
            _log(logging.INFO,
                 "SLIP undo enqueued (sequence {0}).".format(sequence))
            break
        if action == BUTTON_RESET:
            answer = mimics.dialogs.question_box(
                message=(
                    "重置将丢弃本 AI 会话中的全部 {0} 个提示，并把 Mask 恢复"
                    "到会话开始时的状态。该操作无法撤销，需在后台重新组合 Mask"
                    "（约需数秒）。是否重置？"
                ).format(int(state.get("action_history_len") or 0)),
                buttons=BUTTON_RESET + ";" + BUTTON_CANCEL,
                title=TITLE,
                ui_blocking=True,
            )
            if answer != BUTTON_RESET:
                continue
            sequence = _enqueue_command(state, {"action": "reset"})
            _log(logging.INFO,
                 "SLIP reset enqueued (sequence {0}).".format(sequence))
            break
        if action not in (BUTTON_INCLUDE_POINT, BUTTON_EXCLUDE_POINT):
            continue
        point = _capture_point(image, action == BUTTON_INCLUDE_POINT)
        if point is None:
            mimics.dialogs.message_box(
                "未提交任何点提示。\n\nAI 预测未启动。",
                title="SLIP 提示为空", ui_blocking=False)
            continue
        marker = point.pop("_marker", None)
        if marker is not None:
            _ASYNC_VISUAL_OBJECTS.setdefault(state["_job_dir"], []).append(marker)
        state.setdefault("interactions", []).append(
            {"click": point["point"], "label": point["label"]})
        sequence = _enqueue_command(
            state,
            {"action": "predict", "click": point["point"], "label": point["label"]},
        )
        _log(logging.INFO,
             "SLIP predict enqueued (sequence {0}, {1} point {2}).".format(
                 sequence, "include" if point["label"] else "exclude",
                 point["point"]))
        break
    _start_monitor(image, target, state)
    return True


# ---------------------------------------------------------------------------
# 结果应用
# ---------------------------------------------------------------------------

def _handle_error_result(target, state, result):
    error_text = str(result.get("error") or "未知错误")
    state["pending_sequence"] = None
    state["status"] = "ready"
    state["updated_at_epoch"] = time.time()
    _save_job(state)
    if "Nothing to undo" in error_text:
        mimics.dialogs.message_box(
            "撤销失败：{0}".format(error_text), title=TITLE, ui_blocking=False)
        return "ready"
    answer = mimics.dialogs.question_box(
        message=(
            "SLIP 后台预测失败。\n\n"
            "阶段：{0}\n错误：{1}\n\n"
            "重试会重新提交上一次提示；放弃将关闭本次 AI 会话。"
        ).format(str(result.get("stage") or "prediction"), error_text),
        buttons=BUTTON_RETRY + ";" + BUTTON_DISCARD_SESSION + ";" + BUTTON_CANCEL,
        title=TITLE,
        ui_blocking=True,
    )
    if answer == BUTTON_RETRY:
        payload = state.get("last_command")
        if payload and payload.get("action") == "predict":
            _enqueue_command(state, payload)
            return "waiting"
        mimics.dialogs.message_box(
            "没有可重试的提示。", title=TITLE, ui_blocking=False)
        return "ready"
    if answer == BUTTON_DISCARD_SESSION:
        _close_job(target, state, "worker_error_discarded")
        return "ended"
    return "ready"


def _manual_mask_change_dialog(target, state):
    """Mask 在会话外被修改：询问是否从当前 Mask 重启会话。

    返回 restart/ended；两种选择下会话都已关闭、当前 Mask 内容保留。
    """
    answer = mimics.dialogs.question_box(
        message=(
            "所选 Mask 在 AI 会话之外被修改过（手动编辑或被其他操作"
            "改变）。\n\n继续应用 AI 结果会覆盖这些修改。\n\n"
            "是否从当前 Mask 启动新的 AI 会话？"
        ),
        buttons=BUTTON_START_CURRENT + ";" + BUTTON_CANCEL,
        title=TITLE,
        ui_blocking=True,
    )
    _close_job(target, state, "manual_mask_change")
    if answer == BUTTON_START_CURRENT:
        return "restart"
    return "ended"


def _handle_result(image, target, state, interactive=True):
    """消费 pending 结果。返回 waiting/applied/ready/restart/ended。

    interactive=True（工具启动/续接路径）可在 worker 停止时弹
    “从当前 Mask 开始”对话框；监控定时器路径传 False，改为抛出异常
    （由 _monitor_tick 统一弹窗）。
    """
    sequence = state.get("pending_sequence")
    if sequence is None:
        return "ready"
    result_path = _result_path(state, sequence)
    if not os.path.isfile(result_path):
        worker = _worker_status(state)
        worker_status = worker.get("status")
        if worker_status in ("failed", "expired", "closed") or (
            state.get("pid") and not nnm._process_exists(state.get("pid"))
        ):
            stage, error = nnm._describe_async_worker_failure(
                worker, worker_status)
            if not interactive:
                raise RuntimeError(
                    "SLIP 后台工作进程在生成结果前停止了。\n\n"
                    "阶段：{0}\n错误：{1}".format(stage, error))
            answer = mimics.dialogs.question_box(
                message=(
                    "SLIP 后台工作进程在生成结果前停止了。\n\n"
                    "阶段：{0}\n错误：{1}\n\n"
                    "是否从当前 Mask 启动新的 AI 会话？"
                ).format(stage, error),
                buttons=BUTTON_START_CURRENT + ";" + BUTTON_CANCEL,
                title=TITLE,
                ui_blocking=True,
            )
            if answer == BUTTON_START_CURRENT:
                _close_job(target, state, "worker_stopped")
                return "restart"
            return "waiting"
        return "waiting"

    result = runtime_common.read_json(result_path, {}) or {}
    status = result.get("status")
    if not status:
        # 结果文件为原子写入；读到空/半截内容按仍在传输处理。
        return "waiting"

    if status == "error":
        return _handle_error_result(target, state, result)

    if not nnm._same_project_path(
            nnm._current_project_path(), state.get("launch_project_path")):
        _log(logging.INFO,
             "SLIP session was started in another project; closing it.")
        mimics.dialogs.message_box(
            "本 SLIP 会话属于另一个项目，已被关闭。请在当前项目中重新启动。",
            title=TITLE, ui_blocking=False)
        _close_job(target, state, "project_changed")
        return "ended"

    if nnm._object_id(image) != state.get("image_guid") \
            or nnm._object_id(target) != state.get("target_guid"):
        raise RuntimeError(
            "SLIP 会话与当前影像/Mask 不匹配；请重新运行 SLIP。")

    if status == "reset":
        # 与 refined 分支共用同一道内容校验关口：用户同意重置的是提交时
        # 的版本，reset 回执到达前产生的人工编辑同样不能被静默覆盖。
        current_hash = nnm._mask_sha256(target, state.get("shape"))
        if current_hash != state.get("expected_target_sha256"):
            return _manual_mask_change_dialog(target, state)
        nnm._restore_base(target, state.get("base_path"), state.get("shape"))
        _delete_visual_objects(state)
        state["interactions"] = []
        state["action_history_len"] = 0
        state["pending_sequence"] = None
        state["applied_sequence"] = int(sequence)
        state["expected_target_sha256"] = nnm._mask_sha256(
            target, state.get("shape"))
        state["status"] = "ready"
        state["updated_at_epoch"] = time.time()
        _save_job(state)
        _log(logging.INFO, "SLIP session reset; base Mask restored.")
        return "applied"

    if status != "refined":
        raise RuntimeError("SLIP 返回了意外的结果状态：{0}".format(status))

    current_hash = nnm._mask_sha256(target, state.get("shape"))
    if current_hash != state.get("expected_target_sha256"):
        return _manual_mask_change_dialog(target, state)

    output_path = result.get("output_path")
    if not output_path or not os.path.isfile(output_path):
        raise RuntimeError("SLIP 结果文件缺失：{0}".format(output_path))
    voxels = int(result.get("voxels", -1) or 0)
    action = str(result.get("action") or "predict")
    if voxels <= 0 and action == "predict":
        answer = mimics.dialogs.question_box(
            message=(
                "本次预测返回了空结果（0 个体素）。\n\n"
                "是否仍将所选 Mask 清空为该空结果？\n\n"
                "提示：选择“不应用”不会移除该次点击——它仍计入撤销历史，"
                "下次撤销将先移除它。"
            ),
            buttons=BUTTON_APPLY_EMPTY + ";" + BUTTON_KEEP,
            title=TITLE,
            ui_blocking=True,
        )
        if answer != BUTTON_APPLY_EMPTY:
            state["pending_sequence"] = None
            # 该点击仍留在 worker 的撤销历史中，本地计数与其保持一致，
            # 否则菜单不会给出撤销入口。
            state["action_history_len"] = int(
                result.get("action_history_len") or 0)
            state["status"] = "ready"
            state["updated_at_epoch"] = time.time()
            _save_job(state)
            return "ready"

    if action == "undo" and int(result.get("action_history_len") or 0) <= 0:
        # 上游把“撤销第一个提示”实现为回到空模型状态，其输出是空 Mask；
        # 本会话的基线是会话开始时的 Mask 内容——非空基线不能被一次撤销
        # 清空，退到零条提示时恢复基线而不是应用空输出。
        nnm._restore_base(target, state.get("base_path"), state.get("shape"))
        _delete_visual_objects(state)
        state["pending_sequence"] = None
        state["applied_sequence"] = int(sequence)
        state["interactions"] = []
        state["action_history_len"] = 0
        state["expected_target_sha256"] = nnm._mask_sha256(
            target, state.get("shape"))
        state["status"] = "ready"
        state["updated_at_epoch"] = time.time()
        _save_job(state)
        _log(logging.INFO,
             "SLIP undo to zero prompts; session base Mask restored.")
        return "applied"

    nnm._set_mask_from_u8(target, output_path, state.get("shape"))
    nnm._make_mask_visible(target)
    _delete_visual_objects(state)
    state["pending_sequence"] = None
    state["applied_sequence"] = int(sequence)
    state["action_history_len"] = int(result.get("action_history_len") or 0)
    if action == "undo" and state.get("interactions"):
        state["interactions"].pop()
    # 重新锚定哈希：对输出文件本身做哈希曾造成误报“手动修改”。
    state["expected_target_sha256"] = nnm._mask_sha256(
        target, state.get("shape"))
    state["status"] = "ready"
    state["updated_at_epoch"] = time.time()
    _save_job(state)
    _log(logging.INFO,
         "SLIP {0} applied ({1} voxels, {2:.2f}s, history {3}).".format(
             action, voxels,
             float(result.get("interaction_seconds") or 0.0),
             state["action_history_len"]))
    return "applied"


# ---------------------------------------------------------------------------
# 非阻塞结果监控（PyQt5 QTimer 优先，Win32 SetTimer 兜底）
# ---------------------------------------------------------------------------

def _monitor_tick(monitor):
    if monitor.get("done"):
        return
    # _handle_result 可能弹出 ui_blocking 对话框；对话框会泵 Mimics 消息
    # 循环，导致本定时器在对话框未关闭时重入。整个 tick 期间用 busy 防
    # 重入。
    if monitor.get("busy"):
        return
    operation_token = runtime_common.try_acquire_local_operation(
        "mask_buffer_access", "SLIP result monitor")
    if not operation_token:
        owner = runtime_common.active_local_operation(
            "mask_buffer_access") or {}
        owner_text = str(owner.get("owner") or "另一个 Mask 操作")
        due, elapsed = runtime_common.progress_notice_due(
            monitor, "slip_buffer_wait", detail=owner_text,
            interval_seconds=60.0, initial_delay_seconds=5.0)
        if due:
            _log(logging.INFO,
                 "SLIP result handling is waiting for {0} ({1}s). "
                 "Run SLIP again to inspect or discard the pending "
                 "session.".format(owner_text, int(elapsed)))
        return
    runtime_common.clear_progress_notice(monitor, "slip_buffer_wait")
    monitor["busy"] = True
    job_dir = monitor["state"]["_job_dir"]
    try:
        if time.time() > monitor["deadline"]:
            raise RuntimeError(
                "SLIP 后台预测在 {0} 秒内未产生结果。".format(
                    int(monitor["timeout_seconds"])))
        outcome = _handle_result(
            monitor["image"], monitor["target"], monitor["state"],
            interactive=False)
        if outcome == "waiting":
            worker = _worker_status(monitor["state"])
            stage = str(
                worker.get("stage") or worker.get("status")
                or monitor["state"].get("status") or "running")
            sequence = monitor["state"].get("pending_sequence")
            due, elapsed = runtime_common.progress_notice_due(
                monitor, "slip_inference",
                detail="{0}|{1}".format(stage, sequence),
                interval_seconds=60.0, initial_delay_seconds=30.0)
            if due:
                _log(logging.INFO,
                     "SLIP prediction is still running ({0}s). Stage: {1}; "
                     "sequence: {2}. Mimics remains available. Run SLIP "
                     "again to inspect or discard it.".format(
                         int(elapsed), stage, sequence))
        if outcome != "waiting":
            runtime_common.clear_progress_notice(monitor, "slip_inference")
            if outcome == "applied":
                # 连续提示：结果应用后立即给出下一个提示菜单，而不是强迫
                # 用户重新运行入口脚本。
                monitor["deadline"] = time.time() + monitor["timeout_seconds"]
                # 提示菜单是 ui_blocking 且可能长时间停留；显示菜单前先
                # 释放缓冲区锁，避免其他 Mask 操作被菜单阻塞。busy 保持
                # True 作为重入守卫；finally 中重复释放同一 token 是安全
                # 的（已空则无操作，他人持有则不误删）。
                runtime_common.release_local_operation(
                    "mask_buffer_access", operation_token)
                continued = _continue_prompt(
                    monitor["image"], monitor["target"], monitor["state"])
                if continued:
                    return
            monitor["done"] = True
            _stop_monitor(job_dir)
    except Exception as error:
        monitor["done"] = True
        _stop_monitor(job_dir)
        _log(logging.ERROR,
             "SLIP monitor failed: {0}".format(traceback.format_exc()))
        mimics.dialogs.message_box(
            "SLIP 后台预测失败。\n\n{0}".format(error),
            title="SLIP 失败",
            ui_blocking=True,
        )
    finally:
        monitor["busy"] = False
        runtime_common.release_local_operation(
            "mask_buffer_access", operation_token)


def _stop_monitor(job_dir):
    monitor = _ASYNC_MONITORS.pop(job_dir, None)
    if not monitor:
        return
    timer = monitor.get("timer")
    try:
        if timer is not None and timer.isActive():
            timer.stop()
    except Exception:
        pass
    win32_timer = monitor.get("win32_timer")
    if win32_timer:
        user32, timer_id = win32_timer
        try:
            user32.KillTimer(None, timer_id)
        except Exception:
            pass


def _start_win32_monitor(image, target, state, poll_seconds, timeout_seconds):
    if os.name != "nt":
        return False
    try:
        import ctypes
    except Exception:
        return False

    job_dir = state["_job_dir"]
    _stop_monitor(job_dir)
    user32 = ctypes.windll.user32
    timer_interval_ms = max(100, int(max(0.1, poll_seconds) * 1000))
    TIMERPROC = ctypes.WINFUNCTYPE(
        None,
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_size_t,
        ctypes.c_uint,
    )
    monitor = {
        "timer": None,
        "image": image,
        "target": target,
        "state": state,
        "done": False,
        "busy": False,
        "deadline": time.time() + timeout_seconds,
        "timeout_seconds": timeout_seconds,
    }

    def _timer_proc(hwnd, message, timer_id, tick_count):
        _monitor_tick(monitor)

    callback = TIMERPROC(_timer_proc)
    # 其他 Mimics 模块可能用不同的 WINFUNCTYPE 类配置过 SetTimer；传原始
    # 指针避免 ctypes 类型标识不匹配。
    user32.SetTimer.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_uint,
        ctypes.c_void_p,
    ]
    user32.SetTimer.restype = ctypes.c_size_t
    user32.KillTimer.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    user32.KillTimer.restype = ctypes.c_int
    callback_void = ctypes.cast(callback, ctypes.c_void_p)
    timer_id = user32.SetTimer(None, 0, timer_interval_ms, callback_void)
    if not timer_id:
        return False
    monitor["callback"] = callback
    monitor["callback_void"] = callback_void
    monitor["win32_timer"] = (user32, timer_id)
    _ASYNC_MONITORS[job_dir] = monitor
    return True


def _start_monitor(image, target, state):
    """启动非阻塞定时器监控并在结果就绪时自动应用。"""
    try:
        from PyQt5.QtCore import QTimer
        from PyQt5.QtWidgets import QApplication
    except Exception:
        if _start_win32_monitor(
                image, target, state, POLL_SECONDS, RESULT_TIMEOUT_SECONDS):
            return True
        mimics.dialogs.message_box(
            "后台推理已启动。\n\n"
            "本次会话中 Mimics 未提供可用的定时器 API，因此无法自动应用"
            "结果。稍后再次运行 SLIP 以查看结果。",
            title="SLIP 运行中",
            ui_blocking=False,
        )
        return False

    qapp = QApplication.instance()
    if qapp is None:
        if _start_win32_monitor(
                image, target, state, POLL_SECONDS, RESULT_TIMEOUT_SECONDS):
            return True
        mimics.dialogs.message_box(
            "后台推理已启动。\n\n"
            "未找到活动的应用程序定时器，因此无法自动应用结果。"
            "稍后再次运行 SLIP 以查看结果。",
            title="SLIP 运行中",
            ui_blocking=False,
        )
        return False

    timer = QTimer()
    job_dir = state["_job_dir"]
    _stop_monitor(job_dir)
    monitor = {
        "timer": timer,
        "image": image,
        "target": target,
        "state": state,
        "done": False,
        "busy": False,
        "deadline": time.time() + RESULT_TIMEOUT_SECONDS,
        "timeout_seconds": RESULT_TIMEOUT_SECONDS,
    }
    _ASYNC_MONITORS[job_dir] = monitor

    def _tick():
        _monitor_tick(monitor)

    timer.timeout.connect(_tick)
    timer.start(max(100, int(POLL_SECONDS * 1000)))
    return True


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def _run_session(image, target, source, auto_created, write_mode):
    state = _load_job(target)
    if state is not None:
        if nnm._object_id(image) != state.get("image_guid") \
                or nnm._object_id(target) != state.get("target_guid"):
            _close_job(target, state, "object_identity_changed")
            state = None
        else:
            outcome = _handle_result(image, target, state, interactive=True)
            if outcome in ("waiting", "ended"):
                return 0
            if outcome == "restart":
                state = None
    if state is not None:
        if state.get("pending_sequence") is None \
                and not _state_worker_alive(state):
            _log(logging.INFO,
                 "SLIP previous session worker is no longer running; "
                 "a new session will start from the current Mask.")
            _close_job(target, state, "ready_worker_not_alive")
            state = None
        else:
            current_hash = nnm._mask_sha256(target, state.get("shape"))
            if current_hash != state.get("expected_target_sha256"):
                _close_job(target, state, "manual_mask_change")
                state = None
    if state is None:
        state = _start_job(
            image, target, source=source, write_mode=write_mode,
            auto_created=auto_created)
    _continue_prompt(image, target, state)
    return 0


def run():
    image = mimics.data.images.get_active()
    if image is None:
        raise RuntimeError("请先打开项目并激活一个影像序列，然后再运行 SLIP。")
    _validate_image(image)
    buffer_owner = runtime_common.active_local_operation("mask_buffer_access")
    if buffer_owner:
        mimics.dialogs.message_box(
            message=(
                "在 {0} 正在使用 Mimics 缓冲区时，SLIP 无法获取一致的图像"
                "和 Mask 快照。\n\n请等待该操作完成或停止它，然后重试。"
            ).format(buffer_owner.get("owner") or "另一个 Mimics-Script 任务"),
            title=TITLE,
            ui_blocking=False,
        )
        return 1
    gpu_holder = runtime_common.active_resource_lock(
        _project_root(), "gpu.lock")
    own_state = _find_own_slip_job(gpu_holder)
    if gpu_holder and own_state is None:
        mimics.dialogs.message_box(
            message=(
                "GPU 当前正被 {0} 占用。\n\n"
                "SLIP 未启动，因此未更改任何提示或 Mask 状态。请等待该任务"
                "完成、在其任务窗口中停止它，或使用管理菜单 > 停止所有自有"
                "服务。\n\n"
                "若是遗留的交互会话（SLIP/nnInteractive），也可等待其无操作"
                "超时后自动退出。"
            ).format(runtime_common.resource_lock_summary(gpu_holder)),
            title=TITLE,
            ui_blocking=False,
        )
        return 1
    if own_state is not None:
        # 自己的 SLIP worker 持有 gpu.lock：监控定时器失效后的恢复入口。
        # 只允许取回该会话（继续提示或结束并释放资源），不新建会话。
        target = _session_target_mask(image, own_state)
        if target is None:
            mimics.dialogs.message_box(
                message=(
                    "SLIP 已有一个进行中的会话（结果 Mask：{0}），GPU 正由"
                    "其后台进程占用。\n\n请切换到该会话所在的影像序列后重新"
                    "运行 SLIP 以继续或结束它；或在管理菜单 > 停止所有自有"
                    "服务中停止它。"
                ).format(str(own_state.get("target_name") or "")),
                title=TITLE,
                ui_blocking=False,
            )
            return 1
        return _run_session(
            image, target, target,
            auto_created=bool(own_state.get("auto_created")),
            write_mode=str(own_state.get("write_mode") or "in_place"))
    session = _select_session_target(image)
    source = session["source"]
    target = session["target"]
    if source is not target:
        _log(logging.INFO,
             "SLIP created AI Draft {0} from source Mask {1}; the source "
             "will not be modified.".format(
                 getattr(target, "name", ""), getattr(source, "name", "")))
    return _run_session(
        image, target, source,
        auto_created=session.get("auto_created", False),
        write_mode=session.get("write_mode", "in_place"),
    )


def main():
    try:
        return run()
    except Exception as error:
        _log(logging.ERROR,
             "SLIP failed: {0}".format(traceback.format_exc()))
        mimics.dialogs.message_box(
            "SLIP 交互标注失败：\n\n{0}".format(error),
            title="SLIP 失败",
            ui_blocking=True,
        )
        return 2
