# -*- coding: utf-8 -*-
"""Per-voxel segmentation uncertainty from multiple algorithm masks.

给定同一例数据、由 **不同算法** 得到的多个分割 mask（数量 >= 2，通常 3 个以上），
本脚本在每个像素/体素上统计这些 mask 的分割分歧，计算逐体素的不确定度，
并把不确定度图谱保存为 NIfTI，同时绘制热力图（单层 + 多层拼图）。

支持三种不确定度度量（``method`` 可选）：

* ``"entropy"``      —— 归一化投票熵（vote entropy）。对每个体素统计各标签
                        的得票比例 ``p_k``，计算 Shannon 熵
                        ``H = -sum p_k log p_k``，再除以 ``log(K)`` 归一到
                        ``[0, 1]``。天然支持多类分割。**默认**。
* ``"disagreement"`` —— 分歧比例。``1 - max_k(count_k) / N``，即与多数票不一致
                        的 mask 占比，直观易解释，范围 ``[0, 1-1/N]``。
* ``"variance"``     —— 二值方差。把非零视为前景，``p`` 为前景占比，
                        不确定度 ``= p(1-p)``（并额外给出归一化的 ``4p(1-p)``）。
* ``"compare"``      —— 以 **第一个文件夹** 的 mask 为基准，其余文件夹逐一与其比较，
                        最终只输出 *基准* mask 的不确定度。逐体素定义
                        ``U = d * a``，其中 ``d`` 为“其他 mask 与基准不一致”的比例，
                        ``a`` 为“其他 mask 彼此之间的一致程度”（众数占比）。于是：
                        其他都一致却与基准不同 -> 高；其他彼此不一致且与基准不同
                        -> 中；其他与基准一致 -> 低。范围 ``[0, 1]``。

批量处理：不同算法的 mask 分别放在不同的文件夹（每个文件夹一个算法），同一例数据
在各文件夹下用相同的文件名（由 ``filename_template`` 定义）。需要处理的 case 名称
可以直接在 ``main()`` 中列出，或从外部 txt 文本读入（每行一个 case 名）。每例计算
完成后，其不确定度 NIfTI 与热力图写入指定输出目录。

分析排序：``analyze_uncertainty_dir`` 读入输出目录里已算好的逐体素不确定度 NIfTI，
为每例计算总体不确定度指标（积分不确定度、不确定体积、均值、最大值等），找出
“不确定度最大且区域最大”（即各算法分割差异最大）的病例，并把所有文件按总体指标
从高到低排序输出（打印 + CSV）。

用法：在文件末尾的 ``main()`` 中手动填写各 mask 文件夹、case 列表与输出目录，
直接运行本文件即可。
"""

import csv
import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import nibabel as nib
from scipy import ndimage

# matplotlib 是绘图专用依赖，仅在 plot_uncertainty 里用到。延迟到绘图时才
# import，这样不绘图（save_plots=False）的批量计算/排序流程无需安装 matplotlib
# 也能运行（numpy/scipy/nibabel 即可）。
#
# 不确定度保存时的放大倍数：原始不确定度为 [0,1] 的小数，乘以该倍数
# 并取整后以 unsigned char（uint8）保存，便于当作 mask 叠加到图像上查看。
UNCERTAINTY_SCALE = 10


# ---------------------------------------------------------------------------
# 轻量日志：把跳过/警告/错误等提示写到 log 文件，控制台只走进度条，不刷屏。
# ---------------------------------------------------------------------------
def _log_message(log_path, level, message):
    """把一条日志写入 ``log_path``（追加），带时间戳与级别前缀。

    ``log_path`` 为 None 时不写文件。日志同时不打印到控制台（由调用方决定是否
    额外打印），避免破坏进度条的单行刷新。
    """
    if not log_path:
        return
    log_dir = os.path.dirname(log_path)
    if log_dir and not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = "[{0}] {1} {2}\n".format(stamp, level, message)
    try:
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(line)
    except OSError:
        pass  # 日志写失败不应影响主流程


# ---------------------------------------------------------------------------
# 读取与校验
# ---------------------------------------------------------------------------
def load_masks(mask_paths):
    """读取多个 mask，返回 (堆叠数组, 参考 affine, 参考 header)。

    - ``mask_paths``：mask 文件路径列表（NIfTI），长度必须 >= 2。
    - 所有 mask 必须具有相同的空间尺寸（shape）。
    - 返回的堆叠数组形状为 ``(N, X, Y, Z)``，整型标签。
    """
    if mask_paths is None or len(mask_paths) < 2:
        raise ValueError("至少需要 2 个 mask 才能计算不确定度，当前数量: "
                         "{0}".format(0 if mask_paths is None else len(mask_paths)))

    arrays = []
    ref_affine = None
    ref_header = None
    ref_shape = None
    for idx, path in enumerate(mask_paths):
        if not os.path.isfile(path):
            raise FileNotFoundError("找不到 mask 文件: {0}".format(path))
        img = nib.load(path)
        data = np.asarray(img.dataobj)
        # 分割标签取整，避免浮点误差导致的伪标签。标签值很小，用 int16 就够（
        # 相比 int32 减半内存与内存带宽，==/sum 等逐体素运算更快），整型比较
        # 结果与 int32 完全一致。
        data = np.rint(data).astype(np.int16)
        if ref_shape is None:
            ref_shape = data.shape
            ref_affine = img.affine
            ref_header = img.header
        elif data.shape != ref_shape:
            raise ValueError(
                "mask 尺寸不一致: {0} 的 shape 为 {1}，但参考 shape 为 {2}".format(
                    path, data.shape, ref_shape))
        arrays.append(data)

    stack = np.stack(arrays, axis=0)  # (N, X, Y, Z)
    return stack, ref_affine, ref_header


# ---------------------------------------------------------------------------
# 不确定度计算
# ---------------------------------------------------------------------------
def _label_counts(stack, labels):
    """返回 (计数字典, 多数票标签体素)。

    ``counts[l]`` 为形状 ``(X, Y, Z)`` 的数组，表示标签 ``l`` 在每个体素上的得票数。
    """
    n_masks = stack.shape[0]
    counts = {}
    # 逐标签统计得票（标签数量一般很少，循环开销可忽略）
    count_stack = np.zeros((len(labels),) + stack.shape[1:], dtype=np.int32)
    for i, l in enumerate(labels):
        c = np.sum(stack == l, axis=0).astype(np.int32)
        counts[l] = c
        count_stack[i] = c

    # 多数票标签：得票最多的标签（并列时取标签值较小者，argmax 行为）
    winner_idx = np.argmax(count_stack, axis=0)
    labels_arr = np.asarray(labels)
    majority = labels_arr[winner_idx].astype(np.int32)
    max_count = np.max(count_stack, axis=0)
    return counts, majority, max_count, n_masks


def compute_uncertainty(stack, method="entropy"):
    """计算逐体素不确定度。

    返回 dict：
        ``uncertainty``  —— 不确定度体数据 (float32)，范围见各 method 说明。
        ``majority``     —— 多数票（共识）标签体数据 (int32)。
        ``labels``       —— 出现过的标签列表（含背景 0）。
        ``method``       —— 实际使用的度量名称。
    """
    labels = sorted(int(v) for v in np.unique(stack))
    counts, majority, max_count, n_masks = _label_counts(stack, labels)

    if method == "entropy":
        # 归一化投票熵
        entropy = np.zeros(stack.shape[1:], dtype=np.float64)
        for l in labels:
            p = counts[l].astype(np.float64) / n_masks
            nz = p > 0
            entropy[nz] -= p[nz] * np.log(p[nz])
        k = len(labels)
        if k > 1:
            entropy /= np.log(k)  # 归一到 [0, 1]
        uncertainty = entropy.astype(np.float32)

    elif method == "disagreement":
        uncertainty = (1.0 - max_count.astype(np.float64) / n_masks).astype(np.float32)

    elif method == "variance":
        # 二值：非零视为前景
        fg = np.sum(stack != 0, axis=0).astype(np.float64)
        p = fg / n_masks
        uncertainty = (p * (1.0 - p)).astype(np.float32)  # 原始方差，最大 0.25

    elif method == "compare":
        # 以第一个 mask 为基准，其余 mask 与其比较，输出基准 mask 的不确定度。
        if n_masks < 2:
            raise ValueError("compare 方法至少需要 1 个基准 + 1 个对比 mask。")
        ref = stack[0]                      # 基准（第一个文件夹）
        others = stack[1:]                  # 其余算法
        n_others = others.shape[0]

        # d：其他 mask 与基准不一致的比例（0~1），越大表示越偏离基准。
        disagree = np.sum(others != ref[np.newaxis], axis=0).astype(np.float64) / n_others

        # a：其他 mask 彼此之间的一致程度（众数得票占比，1/n_others ~ 1），
        #    越大表示“其他算法越抱团一致”。
        max_other = np.zeros(stack.shape[1:], dtype=np.int32)
        for l in labels:
            cnt = np.sum(others == l, axis=0).astype(np.int32)
            max_other = np.maximum(max_other, cnt)
        agree_others = max_other.astype(np.float64) / n_others

        # U = d * a：
        #   其他一致(a高)且偏离基准(d高) -> 高；
        #   其他不一致(a低)且偏离基准(d高) -> 中；
        #   其他与基准一致(d低)          -> 低。
        uncertainty = (disagree * agree_others).astype(np.float32)

    else:
        raise ValueError(
            "未知的 method: {0}（可选 entropy/disagreement/variance/compare）".format(method))

    return {
        "uncertainty": uncertainty,
        "majority": majority,
        "labels": labels,
        "method": method,
        "n_masks": n_masks,
    }


# ---------------------------------------------------------------------------
# 多 label 不确定度
# ---------------------------------------------------------------------------
# 多 label 统一按 independent 处理：每个 label 单独当作 one-vs-rest 的二值分割，
# 各自计算不确定度，再逐体素取最大合成一张总图（``compute_uncertainty_independent``）。
# 这样既适用于互不相邻的标签（如多个彼此分离的器官——各算各的），也适用于
# 同一器官的相邻子区域（如某器官各叶——共享边界，在公共边界处两个标签的图都会亮起，
# 边界自然被纳入总图）。
#
# 对相邻标签的“label 交界处”（叶裂缝附近）的不确定度，另外单独提取为
# ``partition_boundary`` 分量（见 ``compute_partition_uncertainty`` 与 ``_interface_band``），
# 便于单独查看与统计，而不必再把标签按器官分组。用户可手动指定参与计算的
# 标签，未列入的标签不参与。


def _foreground_bbox(stack, margin=0):
    """返回所有 mask 非零体素联合包围盒的切片元组 ``(sx, sy, sz)``；全背景返回 None。

    分割 mask 绝大多数体素是背景(0)，各算法在背景处完全一致，不确定度恒为 0。
    先求“任一 mask 非零”的前景包围盒，只在该 ROI 上计算即可，ROI 之外结果全 0。
    ``margin`` 在各方向向外留边（体素），给交界带膨胀等操作预留空间。
    """
    fg = np.any(stack != 0, axis=0)
    if not fg.any():
        return None
    slices = []
    for ax in range(fg.ndim):
        # 沿其余各轴归约，得到该轴“是否含前景”的一维布尔向量，取首尾即边界。
        axis_any = np.any(fg, axis=tuple(a for a in range(fg.ndim) if a != ax))
        nz = np.where(axis_any)[0]
        lo = max(int(nz[0]) - margin, 0)
        hi = min(int(nz[-1]) + 1 + margin, fg.shape[ax])
        slices.append(slice(lo, hi))
    return tuple(slices)


def compute_uncertainty_independent(stack, method="entropy", target_labels=None,
                                    boundary_dilation=1, compute_boundary=True,
                                    need_majority=True):
    """所有 label 统一按 independent 计算：逐标签 one-vs-rest 后取最大合成。

    - 对每个目标标签 ``l``，把体数据二值化为 ``stack == l``（前景/背景），复用
      ``compute_uncertainty`` 计算其不确定度；再逐体素取最大值合成总图。相邻标签
      （如肺各叶）在共享边界处，两个标签的图都会亮起，边界自然被纳入。
    - ``target_labels``：用户手动指定参与计算的标签列表；``None`` 时自动取所有
      非零标签。只有列入的标签参与，其余标签不参与不确定度计算。
    - 额外计算 ``partition_boundary`` 分量：把“label 交界处”（相邻标签的公共边界，
      即叶裂缝附近）的不确定度单独提取出来，便于单独统计交界处体积。
      ``boundary_dilation`` 控制交界带向外膨胀的宽度（体素）。互不相邻的标签
      （如多个彼此分离的器官）彼此不接触，交界带为空，不会产生交界统计；相邻标签
      （如肺各叶）则给出其公共边界的不确定度。

    返回 dict，键与 ``compute_uncertainty`` 兼容，另含：
        ``mode``       —— ``"independent"``。
        ``extra_maps`` —— dict，``partition_boundary`` -> label 交界处不确定度图。
                          （各标签分图从不被保存/使用，故不再单独存储以省内存。）
        ``need_majority`` 为 False 时不计算共识（多数票）标签，``majority`` 返回全零，
        仅在既不保存 consensus 也不出图时使用，可省去一整轮全 ROI 扫描。
    """
    full_shape = stack.shape[1:]
    # 先裁剪到所有 mask 非零区域的联合包围盒(ROI)，只在前景上计算，最后放回原尺寸。
    # 背景处各算法完全一致，不确定度恒为 0，无需参与运算。计算交界带时按膨胀
    # 宽度向外留边，避免裁剪边缘把膨胀截断。
    margin = int(boundary_dilation) if compute_boundary else 0
    bbox = _foreground_bbox(stack, margin=margin)
    if bbox is None:
        # 全背景：无任何前景，各处一致，直接返回全零结果。
        return {
            "uncertainty": np.zeros(full_shape, dtype=np.float32),
            "majority": np.zeros(full_shape, dtype=np.int32),
            "labels": [0],
            "method": method,
            "n_masks": stack.shape[0],
            "mode": "independent",
            "extra_maps": {},
        }
    stack = stack[(slice(None),) + bbox]  # 各 mask 同步裁剪到 ROI

    labels = sorted(int(v) for v in np.unique(stack))
    # 共识（多数票）仅在需要保存 consensus 或绘图时才用得上，否则跳过整轮
    # _label_counts（会为所有标签构造 count_stack 并 argmax），n_masks 直接取堆叠数。
    if need_majority:
        _, majority, _, n_masks = _label_counts(stack, labels)
    else:
        majority = None
        n_masks = stack.shape[0]
    if target_labels is None:
        target_labels = [l for l in labels if l != 0]
    else:
        target_labels = sorted(int(l) for l in target_labels)

    shape = stack.shape[1:]
    combined = np.zeros(shape, dtype=np.float32)
    extra = {}
    if method == "compare":
        # compare 专用快速路径：结果与逐 label one-vs-rest 完全一致，但避免每个
        # label 都构造整型 bstack 并在 compute_uncertainty 里重复计算 majority。
        # 二值化后 max_other = max(前景票, 背景票)，等价于原通用实现。
        n_others = n_masks - 1
        ref = stack[0]
        others = stack[1:]
        inv = 1.0 / n_others
        for l in target_labels:
            ref_l = ref == l
            others_l = others == l                       # (N-1, X, Y, Z) bool
            fg = others_l.sum(axis=0)
            disagree = np.count_nonzero(
                others_l != ref_l[np.newaxis], axis=0).astype(np.float64)
            max_other = np.maximum(fg, n_others - fg).astype(np.float64)
            u_l = ((disagree * inv) * (max_other * inv)).astype(np.float32)
            combined = np.maximum(combined, u_l)
    else:
        for l in target_labels:
            bstack = (stack == l).astype(np.int32)
            res = compute_uncertainty(bstack, method=method)
            u_l = res["uncertainty"]
            combined = np.maximum(combined, u_l)

    # label 交界处（相邻标签公共边界）的不确定度，单独作为一个分量供统计。
    if compute_boundary and len(target_labels) >= 2:
        # valid（各 mask 是否属于 target_labels）只算一次，供下面两处复用。
        valid = _valid_mask(stack, target_labels)
        u_part = compute_partition_uncertainty(
            stack, target_labels, method=method, valid=valid)
        band = _interface_band(
            stack, target_labels, dilation=boundary_dilation, valid=valid)
        extra["partition_boundary"] = (u_part * band).astype(np.float32)

    # 把 ROI 上的结果放回原始尺寸（ROI 之外全部为 0）。
    combined_full = np.zeros(full_shape, dtype=np.float32)
    combined_full[bbox] = combined
    majority_full = np.zeros(full_shape, dtype=np.int32)
    if majority is not None:
        majority_full[bbox] = majority
    extra_full = {}
    for key, vol in extra.items():
        full_vol = np.zeros(full_shape, dtype=np.float32)
        full_vol[bbox] = vol
        extra_full[key] = full_vol

    return {
        "uncertainty": combined_full,
        "majority": majority_full,
        "labels": labels,
        "method": method,
        "n_masks": n_masks,
        "mode": "independent",
        "extra_maps": extra_full,
    }


def _valid_mask(stack, group_labels):
    """返回 (N, X, Y, Z) 布尔数组：各 mask 是否属于 ``group_labels``。

    与 ``np.isin(stack, group_labels)`` 逐位等价，但用布尔查找表（LUT）实现，
    对“少量标签、体数据大”的场景更快。
    """
    group_labels = [int(l) for l in group_labels]
    hi = int(stack.max()) if stack.size else 0
    if group_labels:
        hi = max(hi, max(group_labels))
    lut = np.zeros(hi + 1, dtype=bool)
    for l in group_labels:
        if 0 <= l <= hi:
            lut[l] = True
    return lut[stack]


def _interface_band(stack, group_labels, dilation=1, valid=None):
    """找出器官内部“子标签交界”的体素带（多个 mask 的 label 交界处）。

    对每个 mask，在器官内部（标签落在 ``group_labels``）找出相邻但子标签不同
    的体素（即叶裂缝），标记交界两侧；再对所有 mask 的交界取并集。因不同 mask
    的叶裂位置不同，并集形成一条“交界带”。可用 ``dilation`` 再向外膨胀若干体素，
    以包住交界附近的分歧。返回布尔数组。``valid`` 为可选的预计算“是否属于器官”
    布尔数组（(N,X,Y,Z)），省去重复的 ``np.isin``。
    """
    group_labels = sorted(int(l) for l in group_labels)
    shape = stack.shape[1:]
    if valid is None:
        valid = _valid_mask(stack, group_labels)
    seam = np.zeros(shape, dtype=bool)
    for m in range(stack.shape[0]):
        sub = np.where(valid[m], stack[m], 0)
        for ax in range(sub.ndim):
            a = np.swapaxes(sub, 0, ax)
            # 相邻体素都是前景子标签但不同 -> 交界，标记两侧。
            diff = (a[:-1] != a[1:]) & (a[:-1] > 0) & (a[1:] > 0)
            marked = np.zeros_like(a, dtype=bool)
            marked[:-1] |= diff
            marked[1:] |= diff
            seam |= np.swapaxes(marked, 0, ax)
    if dilation and int(dilation) > 0:
        # 6 邻域膨胀 iterations 次，逐体素等价于“到 seam 的城区距离(L1) <= iterations”。
        # 用一次城区距离变换替代多遍形态学膨胀，结果完全一致但更快。
        dist = ndimage.distance_transform_cdt(~seam, metric="taxicab")
        seam = dist <= int(dilation)
    return seam


def compute_partition_uncertainty(stack, group_labels, method="entropy", valid=None):
    """“label 交界处”子区域划分不确定度（供提取交界分量用）。

    只在**认为该体素属于该器官**（标签落在 ``group_labels`` 内）的 mask 之间统计
    分歧，衡量它们对“该体素属于哪个子区域”是否一致。凡是把该体素判为背景（或
    其他器官）的 mask 不参与本层投票——那部分分歧由“器官范围”一层负责。

    仅在至少有 2 个 mask 认为该体素属于该器官处才有意义，其余位置为 0。
    ``method`` 语义与单 label 版一致；``variance`` 在多子标签下退化为 ``disagreement``。
    ``valid`` 为可选的预计算“是否属于器官”布尔数组（(N,X,Y,Z)），省去重复的 ``np.isin``。
    """
    group_labels = sorted(int(l) for l in group_labels)
    if valid is None:
        valid = _valid_mask(stack, group_labels)   # (N, X, Y, Z) 是否投“属于器官”
    n_vote = valid.sum(axis=0).astype(np.int32)   # 每体素投“属于器官”的 mask 数
    shape = stack.shape[1:]
    out = np.zeros(shape, dtype=np.float32)

    if method == "compare":
        ref = stack[0]
        ref_valid = valid[0]                      # 基准是否认为该体素属于器官
        others = stack[1:]
        others_valid = valid[1:]
        n_ov = others_valid.sum(axis=0).astype(np.float64)  # 其他里投器官的数量
        with np.errstate(divide="ignore", invalid="ignore"):
            # d：投器官的“其他 mask”里，子标签与基准不一致的比例。
            dis = np.sum(others_valid & (others != ref[np.newaxis]),
                         axis=0).astype(np.float64)
            d = np.where(n_ov > 0, dis / n_ov, 0.0)
            # a：这些“其他 mask”对子标签的一致程度（众数占比）。
            max_other = np.zeros(shape, dtype=np.int32)
            for l in group_labels:
                cnt = np.sum(others_valid & (others == l), axis=0).astype(np.int32)
                max_other = np.maximum(max_other, cnt)
            a = np.where(n_ov > 0, max_other.astype(np.float64) / n_ov, 0.0)
        part = d * a
        # 仅当基准认为属于器官、且至少有 1 个其他 mask 也认为属于器官时才计入；
        # 基准判为背景时，子区域划分无从谈起，交由“器官范围”层处理。
        out = np.where(ref_valid & (n_ov > 0), part, 0.0).astype(np.float32)
        return out

    active = n_vote >= 2
    if not np.any(active):
        return out
    nv = n_vote.astype(np.float64)
    count_stack = np.zeros((len(group_labels),) + shape, dtype=np.int32)
    for i, l in enumerate(group_labels):
        count_stack[i] = np.sum(valid & (stack == l), axis=0)

    if method == "entropy":
        ent = np.zeros(shape, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            for i in range(len(group_labels)):
                p = np.where(active, count_stack[i] / nv, 0.0)
                nz = p > 0
                ent[nz] -= p[nz] * np.log(p[nz])
        k = len(group_labels)
        if k > 1:
            ent /= np.log(k)
        out = np.where(active, ent, 0.0).astype(np.float32)
    elif method in ("disagreement", "variance"):
        max_count = count_stack.max(axis=0)
        val = np.where(active, 1.0 - max_count.astype(np.float64) / nv, 0.0)
        out = val.astype(np.float32)
    else:
        raise ValueError(
            "未知的 method: {0}（可选 entropy/disagreement/variance/compare）".format(method))
    return out


# ---------------------------------------------------------------------------
# 保存与绘图
# ---------------------------------------------------------------------------
def save_nifti(data, affine, header, out_path, dtype=np.float32):
    """把体数据保存为 NIfTI。"""
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    img = nib.Nifti1Image(np.asarray(data).astype(dtype), affine, header)
    img.header.set_data_dtype(dtype)
    nib.save(img, out_path)


def _pick_slices(uncertainty, axis, n_slices):
    """按“该层不确定度总和”从大到小挑选若干层的索引（升序返回）。"""
    axes_sum = tuple(a for a in range(uncertainty.ndim) if a != axis)
    per_slice = uncertainty.sum(axis=axes_sum)
    order = np.argsort(per_slice)[::-1]
    chosen = [i for i in order[:n_slices] if per_slice[i] > 0]
    if not chosen:  # 完全一致（无不确定度）时退化为取中间层
        chosen = [uncertainty.shape[axis] // 2]
    return sorted(chosen)


def _take_slice(volume, axis, index):
    """沿 ``axis`` 取第 ``index`` 层，返回 2D 数组。"""
    slicer = [slice(None)] * volume.ndim
    slicer[axis] = index
    return volume[tuple(slicer)]


def plot_uncertainty(result, out_prefix, axis=2, n_slices=9, cmap="inferno"):
    """绘制不确定度图谱。

    生成两张图：
        ``<out_prefix>_slice.png``   —— 不确定度最高层的单层热力图（含共识标签轮廓）。
        ``<out_prefix>_montage.png`` —— 多层不确定度热力图拼图。
    """
    # matplotlib 仅在此处需要，延迟导入（见模块顶部说明）。
    import matplotlib
    matplotlib.use("Agg")  # 无界面后端，直接存图
    from matplotlib.figure import Figure  # 线程安全的 OO 绘图接口（不依赖 pyplot 全局态）

    uncertainty = result["uncertainty"]
    majority = result["majority"]
    method = result["method"]
    vmax = float(uncertainty.max())
    if vmax <= 0:
        vmax = 1.0

    out_dir = os.path.dirname(out_prefix)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    slices = _pick_slices(uncertainty, axis, n_slices)
    best = slices[int(np.argmax(
        [_take_slice(uncertainty, axis, s).sum() for s in slices]))]

    # ---- 单层图：共识标签 + 不确定度热力图 ----
    # 用 Figure() 直接绘图（不经 pyplot 全局态），线程安全，可在线程池中并发出图。
    unc2d = np.rot90(_take_slice(uncertainty, axis, best))
    maj2d = np.rot90(_take_slice(majority, axis, best))
    fig = Figure(figsize=(11, 5.5))
    axes = fig.subplots(1, 2)
    axes[0].imshow(maj2d, cmap="tab20", interpolation="nearest")
    axes[0].set_title("Consensus (majority vote)  z={0}".format(best))
    axes[0].axis("off")
    im = axes[1].imshow(unc2d, cmap=cmap, vmin=0.0, vmax=vmax, interpolation="nearest")
    axes[1].set_title("Uncertainty ({0})".format(method))
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_prefix + "_slice.png", dpi=150)

    # ---- 多层拼图 ----
    cols = int(np.ceil(np.sqrt(len(slices))))
    rows = int(np.ceil(len(slices) / float(cols)))
    fig = Figure(figsize=(cols * 3.0, rows * 3.0))
    axes = np.atleast_1d(fig.subplots(rows, cols)).ravel()
    last_im = None
    for ax, s in zip(axes, slices):
        img2d = np.rot90(_take_slice(uncertainty, axis, s))
        last_im = ax.imshow(img2d, cmap=cmap, vmin=0.0, vmax=vmax, interpolation="nearest")
        ax.set_title("z={0}".format(s), fontsize=9)
        ax.axis("off")
    for ax in axes[len(slices):]:
        ax.axis("off")
    if last_im is not None:
        fig.colorbar(last_im, ax=axes.tolist(), fraction=0.025, pad=0.02)
    fig.suptitle("Per-voxel uncertainty map ({0}, N={1})".format(
        method, result["n_masks"]))
    fig.savefig(out_prefix + "_montage.png", dpi=150)

    return out_prefix + "_slice.png", out_prefix + "_montage.png"


# ---------------------------------------------------------------------------
# 单例流程
# ---------------------------------------------------------------------------
def _voxel_volume_mm3(affine):
    """由 affine 计算单个体素的物理体积（mm^3）。"""
    # 体素物理体积 = |det(方向-尺度 3x3 子矩阵)|
    return float(abs(np.linalg.det(np.asarray(affine)[:3, :3])))


def compute_case_metrics(uncertainty, affine, threshold=1e-6, compute_cc=True):
    """由不确定度体数据计算一例的总体指标。

    返回 dict：
        ``integrated``   —— 积分不确定度 = sum(unc) * 体素体积(mm^3)。
                            同时反映“不确定度高”和“区域大”，作为默认排序键。
        ``uncertain_vol``—— 不确定体积(mm^3) = 体素数(unc>threshold) * 体素体积。
        ``mean``         —— 全体素不确定度均值。
        ``mean_nonzero`` —— 仅不确定体素上的均值。
        ``max``          —— 最大不确定度。
        ``n_uncertain``  —— 不确定体素数量。
        ``voxel_mm3``    —— 单体素体积(mm^3)。
        ``cc_vol``       —— 最大连通域体积(mm^3)。
        ``cc_mean``      —— 最大连通域内的不确定度均值。
        ``cc_integrated``—— 最大连通域的积分不确定度 = sum(unc_cc) * 体素体积。
        ``cc_nvox``      —— 最大连通域的体素数。
        ``level_vol``    —— dict，键为离散级别 k(1..UNCERTAINTY_SCALE)，
                            值为“离散不确定度 >= k”的累计体积(mm^3)。
    """
    unc = np.asarray(uncertainty, dtype=np.float32)
    vox = _voxel_volume_mm3(affine)
    mask = unc > threshold
    n_unc = int(mask.sum())
    total = float(unc.sum())

    # 最大连通域：在“不确定体素”上做 3D 连通标记（默认 6-连通），
    # 取体素数最多的那个连通域，统计其体积/均值/积分不确定度。
    # compute_cc=False 时跳过连通域分析（排序阶段若不需要 cc_* 指标可省时）。
    cc_nvox = 0
    cc_sum = 0.0
    if compute_cc and n_unc > 0:
        labeled, n_cc = ndimage.label(mask)
        if n_cc > 0:
            comp_sizes = np.bincount(labeled.ravel())
            comp_sizes[0] = 0  # 背景不计
            largest = int(np.argmax(comp_sizes))
            cc_mask = labeled == largest
            cc_nvox = int(cc_mask.sum())
            cc_sum = float(unc[cc_mask].sum())

    # 逐级累计体积：把不确定度离散化到 1..UNCERTAINTY_SCALE（与保存口径一致），
    # 分别统计 >=1, >=2, ..., >=UNCERTAINTY_SCALE 的体积。
    scaled = np.clip(np.rint(unc * UNCERTAINTY_SCALE), 0, UNCERTAINTY_SCALE).astype(np.int32)
    # 一次直方图 + 后缀和替代 UNCERTAINTY_SCALE 次全量 >=k 比较，整数结果完全一致。
    counts = np.bincount(scaled.ravel(), minlength=UNCERTAINTY_SCALE + 1)
    suffix = counts[::-1].cumsum()[::-1]  # suffix[k] = #(scaled >= k)
    level_vol = {k: int(suffix[k]) * vox for k in range(1, UNCERTAINTY_SCALE + 1)}

    return {
        "integrated": total * vox,
        "uncertain_vol": n_unc * vox,
        "mean": float(unc.mean()),
        "mean_nonzero": float(unc[mask].mean()) if n_unc > 0 else 0.0,
        "max": float(unc.max()),
        "n_uncertain": n_unc,
        "voxel_mm3": vox,
        "cc_vol": cc_nvox * vox,
        "cc_mean": (cc_sum / cc_nvox) if cc_nvox > 0 else 0.0,
        "cc_integrated": cc_sum * vox,
        "cc_nvox": cc_nvox,
        "level_vol": level_vol,
    }


def run_case(mask_paths, out_dir, case_name="case", method="entropy",
             plot_axis=2, n_slices=9, target_labels=None,
             boundary_dilation=1, save_components=True, save_consensus=True,
             save_plots=True, verbose=True):
    """处理一例：读取多个 mask -> 计算不确定度 -> 保存 NIfTI 与图谱。

    不确定度统一按 independent 计算：每个 label 单独做 one-vs-rest 二值不确定度，
    再逐体素取最大合成总图（相邻标签的公共边界会自然纳入）。``target_labels`` 为
    用户手动指定参与计算的标签列表，只有列入的标签参与，其余不参与；``None`` 时
    自动取所有非零标签。另额外输出 ``partition_boundary`` 分量（label 交界处不确定度），
    ``boundary_dilation`` 控制交界带宽度。

    ``save_consensus``：是否保存共识（多数票）标签 ``*_consensus.nii.gz``，
    有时并不需要，可置 False 关闭。``save_plots``：是否绘制并保存热力图
    （``*_slice.png`` 与 ``*_montage.png``），不需要出图时可置 False 关闭。

    返回该例的总体指标 dict（见 ``compute_case_metrics``），并附带输出文件路径。
    """
    # save_components=False 时既不计算也不保存 label 交界分量，省去 partition 计算。
    compute_boundary = save_components
    # 共识（多数票）标签仅在保存 consensus 或绘图时才用到，否则跳过其计算。
    need_majority = save_consensus or save_plots
    stack, affine, header = load_masks(mask_paths)
    result = compute_uncertainty_independent(
        stack, method=method, target_labels=target_labels,
        boundary_dilation=boundary_dilation, compute_boundary=compute_boundary,
        need_majority=need_majority)

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    unc_nii = os.path.join(out_dir, "{0}_uncertainty_{1}.nii.gz".format(case_name, method))
    maj_nii = os.path.join(out_dir, "{0}_consensus.nii.gz".format(case_name))
    # 不确定度乘以 UNCERTAINTY_SCALE 并四舍五入，以 uint8 保存，可当作 mask 查看。
    unc_scaled = np.clip(
        np.rint(result["uncertainty"].astype(np.float64) * UNCERTAINTY_SCALE),
        0, 255).astype(np.uint8)
    save_nifti(unc_scaled, affine, header, unc_nii, dtype=np.uint8)
    # 共识（多数票）标签有时并不需要，save_consensus=False 时不保存。
    if save_consensus:
        save_nifti(result["majority"], affine, header, maj_nii, dtype=np.int16)
    else:
        maj_nii = None

    # 只额外保存 label 交界分量（partition_boundary，供 CSV 交界统计）；
    # 不保存各标签分图，避免占用空间与耗时。
    component_niis = []
    if save_components and result.get("extra_maps"):
        for comp_name, comp_vol in sorted(result["extra_maps"].items()):
            if comp_name.startswith("label"):
                continue
            comp_path = os.path.join(
                out_dir, "{0}_uncertainty_{1}_{2}.nii.gz".format(
                    case_name, method, comp_name))
            comp_scaled = np.clip(
                np.rint(np.asarray(comp_vol, dtype=np.float64) * UNCERTAINTY_SCALE),
                0, 255).astype(np.uint8)
            save_nifti(comp_scaled, affine, header, comp_path, dtype=np.uint8)
            component_niis.append(comp_path)

    prefix = os.path.join(out_dir, "{0}_uncertainty_{1}".format(case_name, method))
    # 绘图会截出图片（单层 + 多层拼图），有时不需要，save_plots=False 时跳过。
    if save_plots:
        slice_png, montage_png = plot_uncertainty(
            result, prefix, axis=plot_axis, n_slices=n_slices)
    else:
        slice_png, montage_png = None, None

    metrics = compute_case_metrics(result["uncertainty"], affine)
    metrics.update({
        "case": case_name,
        "method": method,
        "label_mode": result.get("mode", "independent"),
        "n_masks": result["n_masks"],
        "labels": result["labels"],
        "uncertainty_nii": unc_nii,
        "consensus_nii": maj_nii,
        "component_niis": component_niis,
        "slice_png": slice_png,
        "montage_png": montage_png,
    })

    if verbose:
        print("[{0}] 使用 {1} 个 mask，标签={2}".format(case_name, result["n_masks"], result["labels"]))
        print("    不确定度: 均值={0:.4f} 最大={1:.4f} 非零体素={2}".format(
            metrics["mean"], metrics["max"], metrics["n_uncertain"]))
        print("    积分不确定度={0:.1f} 不确定体积={1:.1f} mm^3".format(
            metrics["integrated"], metrics["uncertain_vol"]))
        print("    最大连通域: 体积={0:.1f} mm^3 均值={1:.4f} 积分={2:.1f}".format(
            metrics["cc_vol"], metrics["cc_mean"], metrics["cc_integrated"]))
        lv = metrics["level_vol"]
        print("    逐级体积(mm^3): " + " ".join(
            ">={0}:{1:.1f}".format(k, lv[k]) for k in sorted(lv)))
        print("    已保存: {0}".format(unc_nii))
        if maj_nii:
            print("             {0}".format(maj_nii))
        if slice_png:
            print("             {0}".format(slice_png))
        if montage_png:
            print("             {0}".format(montage_png))
    return metrics


# ---------------------------------------------------------------------------
# 批量处理
# ---------------------------------------------------------------------------
def read_case_list(case_names=None, case_list_txt=None):
    """得到需要处理的 case 名称列表。

    - ``case_names``：直接给出的列表（优先）。
    - ``case_list_txt``：txt 文本路径，每行一个 case 名（支持 ``#`` 注释、空行）。
    至少要提供其中之一。
    """
    cases = []
    if case_names:
        cases = [str(c).strip() for c in case_names if str(c).strip()]
    elif case_list_txt:
        if not os.path.isfile(case_list_txt):
            raise FileNotFoundError("找不到 case 列表文件: {0}".format(case_list_txt))
        with open(case_list_txt, "r", encoding="utf-8-sig") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                cases.append(line)
    if not cases:
        raise ValueError("未得到任何 case 名称，请填写 case_names 或 case_list_txt。")
    return cases


def resolve_mask_paths(case_name, mask_dirs, filename_template="{case}.nii.gz"):
    """在各算法文件夹中为某个 case 定位 mask 文件。

    - ``mask_dirs``：每个算法一个文件夹的路径列表（长度 >= 2）。
    - ``filename_template``：文件名模板，``{case}`` 会被替换为 case 名，可含子路径。

    某例数据不一定在每个文件夹中都有对应 mask（例如 5 个文件夹中只有 4 个有该
    数据，且不一定是哪 4 个）。本函数只返回 **实际存在** 的 mask 路径（按
    ``mask_dirs`` 的顺序），缺失的记入 ``missing``。调用方据此用已有 mask 计算
    不确定度即可。

    返回 ``(paths, missing)``：
        ``paths``   —— 存在的 mask 路径列表（每个可用文件夹一个，保持输入顺序）。
        ``missing`` —— 缺失的 mask 路径列表。
    """
    if not mask_dirs or len(mask_dirs) < 2:
        raise ValueError("至少需要 2 个 mask 文件夹（不同算法各一个）。")
    paths = []
    missing = []
    for d in mask_dirs:
        p = os.path.join(d, filename_template.format(case=case_name))
        if os.path.isfile(p):
            paths.append(p)
        else:
            missing.append(p)
    return paths, missing


def _run_case_job(job):
    """线程 worker：处理一例并返回 ``(case_name, metrics, error_str)``。

    在批量计算中被调度到线程池执行（各例相互独立）。单例异常在此捕获并以字符串
    返回，不中断整批（与原串行版“单例失败不影响其余批次”的行为一致）。用线程而非
    进程：本框架入口模块顶层会 import torch/nnUNet，进程池（Windows spawn）会在
    每个子进程重新导入这些重模块而卡死；本步骤热点运算（numpy/scipy/nibabel）均
    释放 GIL，线程池既能真并行又不必重导入、不必 pickle。
    """
    case_name = job["case_name"]
    try:
        metrics = run_case(
            job["paths"], job["out_dir"], case_name=case_name,
            method=job["method"], plot_axis=job["plot_axis"],
            n_slices=job["n_slices"], target_labels=job["target_labels"],
            boundary_dilation=job["boundary_dilation"],
            save_components=job["save_components"],
            save_consensus=job["save_consensus"],
            save_plots=job["save_plots"], verbose=False)
        return case_name, metrics, None
    except Exception as exc:  # 单例失败不影响其余批次
        return case_name, None, "{0}".format(exc)


def run_batch(mask_dirs, out_dir, case_names=None, case_list_txt=None,
              filename_template="{case}.nii.gz", method="entropy",
              plot_axis=2, n_slices=9, min_masks=2, skip_incomplete=True,
              target_labels=None, boundary_dilation=1, save_components=True,
              save_consensus=True, save_plots=True, log_path=None,
              n_workers=None):
    """批量处理多例数据。

    对每个 case，从各算法文件夹中收集 mask，计算不确定度并写入 ``out_dir``。
    返回所有成功处理病例的指标列表。

    ``log_path``：可选日志文件路径。跳过/警告/错误等提示写入该文件（追加，带
    时间戳），控制台只显示进度条与最终完成信息，不刷屏。``None`` 时不写日志，
    这些提示也不打印到控制台。

    ``n_workers``：并行线程数。各例相互独立，写入磁盘的结果与串行完全一致，
    仅完成顺序不同。``None`` 时自动取 CPU 核数；``1`` 则退化为串行（逐例，与原
    实现等价）；``>1`` 则使用固定的并行线程数。采用线程池而非进程池：本框架
    入口模块顶层会 import torch/nnUNet，进程池（Windows spawn）会在每个子进程
    重新导入这些重模块而卡死；本步骤热点运算均释放 GIL，线程池既能真并行又
    避开上述问题。
    """
    cases = read_case_list(case_names=case_names, case_list_txt=case_list_txt)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # compare 方法以第一个文件夹为基准，该基准 mask 必须存在；缺失则跳过此例。
    ref_dir = mask_dirs[0] if mask_dirs else None

    n_cases = len(cases)
    t0 = time.time()
    print("开始批量计算：共 {0} 例...".format(n_cases))

    # 决定并行线程数：None -> 自动取 CPU 核数；<=1 -> 串行（与原逐例行为一致）。
    if n_workers is None:
        n_workers = os.cpu_count() or 1
    n_workers = max(1, int(n_workers))

    # 阶段一（串行、仅轻量 I/O 检查）：为每例收集可用 mask，处理缺失/跳过与日志，
    # 得到待计算作业列表 jobs。跳过判定与结果口径和原串行版完全一致。
    done = [0]  # 已处理计数（跳过 + 完成），用列表以便在闭包内累加。

    def _tick(suffix):
        done[0] += 1
        _print_progress_bar(done[0], n_cases, prefix="批量进度", suffix=suffix)

    jobs = []
    for case_name in cases:
        paths, missing = resolve_mask_paths(case_name, mask_dirs, filename_template)

        # compare 方法需要基准 mask（第一个文件夹）；基准缺失则跳过此例，不另找基准。
        if method == "compare":
            ref_path = os.path.join(ref_dir, filename_template.format(case=case_name))
            if not os.path.isfile(ref_path):
                _log_message(log_path, "跳过",
                             "{0}: compare 需要基准文件夹的 mask，但缺失: {1}".format(
                                 case_name, ref_path))
                _tick("跳过 {0}".format(case_name))
                continue

        if len(paths) < min_masks:
            msg = "{0}: 仅找到 {1} 个 mask（需要 >= {2}）。缺失: {3}".format(
                case_name, len(paths), min_masks, missing)
            if skip_incomplete:
                _log_message(log_path, "跳过", msg)
                _tick("跳过 {0}".format(case_name))
                continue
            raise FileNotFoundError(msg)
        if missing:
            _log_message(log_path, "警告",
                         "{0}: 缺失 {1} 个算法的 mask，将用剩余 {2} 个计算。".format(
                             case_name, len(missing), len(paths)))
        jobs.append({
            "case_name": case_name, "paths": paths, "out_dir": out_dir,
            "method": method, "plot_axis": plot_axis, "n_slices": n_slices,
            "target_labels": target_labels, "boundary_dilation": boundary_dilation,
            "save_components": save_components, "save_consensus": save_consensus,
            "save_plots": save_plots,
        })

    all_metrics = []

    def _consume(case_name, metrics, err):
        if err is not None:
            _log_message(log_path, "错误", "{0}: 处理失败 -> {1}".format(case_name, err))
        elif metrics is not None:
            all_metrics.append(metrics)
        # 进度条：原地刷新单行，显示百分比/计数/已用时间/预估剩余，不刷屏。
        elapsed = time.time() - t0
        avg = elapsed / max(done[0] + 1, 1)
        remain = avg * (n_cases - done[0] - 1)
        _tick("已用 {0:.1f}s  剩余 {1:.1f}s".format(elapsed, remain))

    # 阶段二：计算不确定度。多线程并行（各例独立，热点运算释放 GIL），或退化为串行。
    # _consume 只在主线程的 as_completed 循环里调用，故 all_metrics/进度条/日志无并发问题。
    workers = min(n_workers, len(jobs)) if jobs else 1
    if workers <= 1:
        for job in jobs:
            case_name, metrics, err = _run_case_job(job)
            _consume(case_name, metrics, err)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_run_case_job, job) for job in jobs]
            for fut in as_completed(futures):
                case_name, metrics, err = fut.result()
                _consume(case_name, metrics, err)

    print("批量完成：成功 {0}/{1} 例，耗时 {2:.1f}s。".format(
        len(all_metrics), n_cases, time.time() - t0))
    if log_path:
        print("日志已写入: {0}".format(log_path))
    return all_metrics


# ---------------------------------------------------------------------------
# 分析与排序：找出差异最大的病例
# ---------------------------------------------------------------------------
def _print_progress_bar(iteration, total, prefix="", suffix="", bar_len=30):
    """在终端原地刷新一行进度条（用 \\r 回到行首覆盖），避免逐例刷屏。

    类似 pip 安装库时的进度条：同一行不断更新，显示百分比、进度条、计数与
    自定义后缀（如已用时间/预估剩余）。``iteration == total`` 时换行收尾。
    """
    if total <= 0:
        return
    frac = iteration / total
    filled = int(round(bar_len * frac))
    bar = "#" * filled + "-" * (bar_len - filled)
    line = "\r{0} |{1}| {2:5.1f}%  {3}/{4}  {5}".format(
        prefix, bar, frac * 100.0, iteration, total, suffix)
    # 截断到 100 字符，防止超宽终端换行破坏单行刷新效果。
    print(line[:100], end="", flush=True)
    if iteration >= total:
        print()  # 完成后换到下一行


def analyze_uncertainty_dir(out_dir, method="entropy", sort_key="integrated",
                            report_csv=None, threshold=1e-6,
                            compute_boundary=True, compute_cc=True):
    """读入输出目录里已算好的逐体素不确定度 NIfTI，逐例计算总体指标并排序。

    - 匹配文件名形如 ``*_uncertainty_<method>.nii.gz``。
    - 为每例计算积分不确定度、不确定体积等指标（见 ``compute_case_metrics``）。
    - 若同目录存在对应的 label 交界分量 ``*_uncertainty_<method>_partition_boundary*.nii.gz``
      （由 ``run_case``/``run_batch`` 保存），则读入它们（多分量逐体素取最大合成）
      并单独计算“label 交界处（叶裂缝）不确定度”指标，作为额外
      列并入 CSV。因此本函数完全依据磁盘上已有结果自足运行，无需重新执行 ``run_batch``。
    - 按 ``sort_key``（默认 ``"integrated"`` 积分不确定度，兼顾“不确定度高”与
      “区域大”）从高到低排序，打印并可写出 CSV。
    - ``compute_boundary``：是否读取并统计 label 交界分量（partition_boundary）。
      与 ``run_batch`` 的 ``save_components`` 对应——若计算阶段未保存交界分量，
      此处置 False 可跳过读取与统计，避免无谓的文件查找。
    - ``compute_cc``：是否计算最大连通域指标（cc_*）。连通域分析是逐例最重的一步，
      若 ``sort_key`` 不依赖 cc_*（如用 ``integrated``/``uncertain_vol``），可置 False
      跳过以加速；需要时再置 True。默认 True（保持原行为）。
    返回排序后的指标列表（第 0 个即差异最大的病例）。
    """
    pattern = os.path.join(out_dir, "*_uncertainty_{0}.nii.gz".format(method))
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError("在 {0} 未找到不确定度文件（模式: {1}）。".format(
            out_dir, os.path.basename(pattern)))
    # 排除各分量文件（*_uncertainty_<method>_<comp>.nii.gz），只保留总图。
    suffix = "_uncertainty_{0}".format(method)
    files = [f for f in files
             if os.path.basename(f).endswith("{0}.nii.gz".format(suffix))]

    n_files = len(files)
    print("\n开始统计分析：共 {0} 例，逐例读取并计算指标...".format(n_files))
    rows = []
    t0 = time.time()
    for idx, path in enumerate(files, 1):
        img = nib.load(path)
        # 保存时不确定度已乘以 UNCERTAINTY_SCALE 并存为整数，这里还原回 [0,1] 小数。
        # 直接以 float32 读入并就地除，避免再转一次 float64（compute_case_metrics
        # 内部会按需取统计量，float32 已足够精确）。
        unc = np.asarray(img.dataobj, dtype=np.float32)
        unc *= (1.0 / float(UNCERTAINTY_SCALE))
        m = compute_case_metrics(unc, img.affine, threshold=threshold,
                                 compute_cc=compute_cc)
        base = os.path.basename(path)
        name = base[:-len(".nii.gz")] if base.endswith(".nii.gz") else base
        if name.endswith(suffix):
            name = name[:-len(suffix)]
        m["case"] = name
        m["file"] = path

        # 内部交界处（叶裂缝）不确定度：读入该例的 partition_boundary 分量文件。
        # 命名为 <case>_uncertainty_<method>_partition_boundary[_g<k>].nii.gz。
        # compute_boundary=False 时跳过（与计算阶段 save_components 对应）。
        if compute_boundary:
            boundary_prefix = os.path.join(
                out_dir, "{0}{1}_partition_boundary".format(name, suffix))
            boundary_files = sorted(glob.glob(boundary_prefix + "*.nii.gz"))
            if boundary_files:
                band = None
                for bpath in boundary_files:
                    bimg = nib.load(bpath)
                    bvol = np.asarray(bimg.dataobj, dtype=np.float32)
                    bvol *= (1.0 / float(UNCERTAINTY_SCALE))
                    band = bvol if band is None else np.maximum(band, bvol)
                bm = compute_case_metrics(band, img.affine, threshold=threshold,
                                          compute_cc=compute_cc)
                m["boundary_integrated"] = bm["integrated"]
                m["boundary_vol"] = bm["uncertain_vol"]
                m["boundary_mean_nonzero"] = bm["mean_nonzero"]
                m["boundary_max"] = bm["max"]
        rows.append(m)

        # 进度条：原地刷新单行，显示百分比/计数/已用时间/预估剩余，不刷屏。
        elapsed = time.time() - t0
        avg = elapsed / idx
        remain = avg * (n_files - idx)
        _print_progress_bar(
            idx, n_files, prefix="分析进度",
            suffix="已用 {0:.1f}s  剩余 {1:.1f}s".format(elapsed, remain))
    print("全部 {0} 例读取+计算完成，耗时 {1:.1f}s。".format(n_files, time.time() - t0))

    # 只要有任一例读到了内部交界分量，就在 CSV 里补齐这些列（缺失者填 0）。
    has_boundary = any("boundary_integrated" in r for r in rows)
    if has_boundary:
        for r in rows:
            r.setdefault("boundary_integrated", 0.0)
            r.setdefault("boundary_vol", 0.0)
            r.setdefault("boundary_mean_nonzero", 0.0)
            r.setdefault("boundary_max", 0.0)

    if sort_key not in rows[0]:
        raise ValueError("未知的 sort_key: {0}（可选 {1}）".format(
            sort_key, [k for k in rows[0] if isinstance(rows[0][k], (int, float))]))
    rows.sort(key=lambda r: r[sort_key], reverse=True)

    print("\n==== 不确定度排序（按 {0} 从高到低）====".format(sort_key))
    header = "{0:>4}  {1:<24} {2:>14} {3:>16} {4:>10} {5:>8} {6:>14} {7:>12}".format(
        "rank", "case", "integrated", "uncertain_vol", "mean_nz", "max",
        "cc_integrated", "cc_vol")
    print(header)
    print("-" * len(header))
    for i, r in enumerate(rows, 1):
        print("{0:>4}  {1:<24} {2:>14.1f} {3:>16.1f} {4:>10.4f} {5:>8.4f} {6:>14.1f} {7:>12.1f}".format(
            i, r["case"], r["integrated"], r["uncertain_vol"],
            r["mean_nonzero"], r["max"], r["cc_integrated"], r["cc_vol"]))

    top = rows[0]
    print("\n差异最大（不确定度最大且区域最大）的病例: {0}".format(top["case"]))
    print("  积分不确定度={0:.1f}  不确定体积={1:.1f} mm^3  最大不确定度={2:.4f}".format(
        top["integrated"], top["uncertain_vol"], top["max"]))
    print("  最大连通域: 体积={0:.1f} mm^3  均值={1:.4f}  积分={2:.1f}".format(
        top["cc_vol"], top["cc_mean"], top["cc_integrated"]))
    print("  文件: {0}".format(top["file"]))

    if report_csv:
        csv_dir = os.path.dirname(report_csv)
        if csv_dir and not os.path.isdir(csv_dir):
            os.makedirs(csv_dir)
        fields = ["rank", "case", "integrated", "uncertain_vol", "mean",
                  "mean_nonzero", "max", "n_uncertain", "cc_integrated",
                  "cc_vol", "cc_mean", "cc_nvox", "voxel_mm3"]
        if has_boundary:
            # 内部交界处（叶裂缝）不确定度的统计列，紧跟在总体指标之后。
            insert_at = fields.index("cc_integrated")
            fields = (fields[:insert_at]
                      + ["boundary_integrated", "boundary_vol",
                         "boundary_mean_nonzero", "boundary_max"]
                      + fields[insert_at:])
        level_keys = sorted(rows[0]["level_vol"])
        level_fields = ["vol_ge_{0}".format(k) for k in level_keys]
        fields = fields + level_fields + ["file"]
        with open(report_csv, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for i, r in enumerate(rows, 1):
                row = {
                    "rank": i,
                    "case": r["case"],
                    "integrated": "{0:.4f}".format(r["integrated"]),
                    "uncertain_vol": "{0:.4f}".format(r["uncertain_vol"]),
                    "mean": "{0:.6f}".format(r["mean"]),
                    "mean_nonzero": "{0:.6f}".format(r["mean_nonzero"]),
                    "max": "{0:.6f}".format(r["max"]),
                    "n_uncertain": r["n_uncertain"],
                    "cc_integrated": "{0:.4f}".format(r["cc_integrated"]),
                    "cc_vol": "{0:.4f}".format(r["cc_vol"]),
                    "cc_mean": "{0:.6f}".format(r["cc_mean"]),
                    "cc_nvox": r["cc_nvox"],
                    "voxel_mm3": "{0:.6f}".format(r["voxel_mm3"]),
                    "file": r["file"],
                }
                if has_boundary:
                    row["boundary_integrated"] = "{0:.4f}".format(
                        r.get("boundary_integrated", 0.0))
                    row["boundary_vol"] = "{0:.4f}".format(
                        r.get("boundary_vol", 0.0))
                    row["boundary_mean_nonzero"] = "{0:.6f}".format(
                        r.get("boundary_mean_nonzero", 0.0))
                    row["boundary_max"] = "{0:.6f}".format(
                        r.get("boundary_max", 0.0))
                for k in level_keys:
                    row["vol_ge_{0}".format(k)] = "{0:.4f}".format(r["level_vol"][k])
                writer.writerow(row)
        print("\n排序报告已写入: {0}".format(report_csv))

    return rows


# ---------------------------------------------------------------------------
# 集成到 nnUNet 训练验证框架的接口
# ---------------------------------------------------------------------------
def _extract_target_labels(class_map):
    """从 class_map（扁平或分组格式）提取参与计算的非零标签列表。

    - 扁平格式 ``{organ_name: label_idx}`` -> 取所有 label_idx。
    - 分组格式 ``{group: {"label": l, "organs": [...]}}`` -> 取每组的 label。
    背景标签 0 不参与计算。返回升序标签列表，无非零标签时返回 ``None``。
    """
    labels = set()
    if isinstance(class_map, dict):
        for v in class_map.values():
            if isinstance(v, dict) and "label" in v:
                labels.add(int(v["label"]))
            else:
                try:
                    labels.add(int(v))
                except (TypeError, ValueError):
                    pass
    labels.discard(0)
    return sorted(labels) if labels else None


def _collect_fold_validation_dirs(model_folder, total_folds):
    """收集某模型各折的验证输出目录 ``fold_<f>/validation``（仅返回实际存在者）。

    nnUNet 训练完一折后，会把该折验证集的预测写入
    ``<model_folder>/fold_<f>/validation``。五折交叉验证下同一 case 可能出现在
    多个折的验证集中（如反转划分或大验证比例），据此得到同一 case 的多个 mask。
    """
    dirs = []
    for f in range(int(total_folds)):
        vdir = os.path.join(model_folder, "fold_{0}".format(f), "validation")
        if os.path.isdir(vdir):
            dirs.append(vdir)
    return dirs


def _collect_case_names(mask_dirs, filename_suffix=".nii.gz"):
    """取各文件夹中所有 mask 文件名（去后缀）的并集，作为待处理 case 列表。"""
    cases = set()
    for d in mask_dirs:
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            if name.endswith(filename_suffix):
                cases.add(name[:-len(filename_suffix)])
    return sorted(cases)


def run_from_config(config):
    """nnUNet 训练验证框架中「不确定度」步骤的入口函数。

    与 evaluation 步骤类似，读入已解析的 ``.toml`` 配置（dict），自动推导五折
    交叉验证各折的验证预测目录、参与计算的标签等参数（这些在 nnUNet 训练中已有
    默认路径与约定），再结合配置中 ``[uncertainty]`` 段里与不确定度计算相关的参数，
    对每个 ``train_dataset`` 逐一计算并分析逐体素分割不确定度。

    自动推导（无需人工设置）：
      * mask 目录：``<nnUNet_results>/<dataset>/<trainer>__<plans>__<configuration>/
        fold_<f>/validation``（各折验证预测），可选把金标准 ``labelsTr`` 作为基准。
      * 目标标签：由该数据集的 ``segment_list`` 自动得到非零标签。
      * case 列表：各折验证目录内 mask 文件名的并集。
      * 输出目录：``<nnUNet_raw>/<dataset>/uncertainty``（可在配置中覆盖）。

    仅与不确定度相关、需手动设置的参数（method / boundary_dilation 等）从
    ``[uncertainty]`` 段读取，缺省时使用合理默认值。
    """
    paths = config["PATHS"]
    model = config["MODEL"]
    nnunet_results = paths["nnUNet_results"]
    nnunet_raw = paths["nnUNet_raw"]
    train_datasets = model["train_dataset"]
    class_maps = model["segment_list"]
    trainer = config["TRAIN"]["trainer"]
    plans = config["TRAIN"]["plans"]
    configuration = config["PREPROCESS"]["configuration"]
    total_folds = config.get("DATA", {}).get("total_folds", 5)

    # 与不确定度计算相关、需手动设置的参数放在 [uncertainty] 段。
    unc_cfg = config.get("uncertainty") or config.get("UNCERTAINTY") or {}
    method = unc_cfg.get("method", "compare")
    boundary_dilation = unc_cfg.get("boundary_dilation", 1)
    plot_axis = unc_cfg.get("plot_axis", 2)
    n_slices = unc_cfg.get("n_slices", 9)
    save_components = unc_cfg.get("save_components", True)
    save_consensus = unc_cfg.get("save_consensus", True)
    save_plots = unc_cfg.get("save_plots", True)
    min_masks = unc_cfg.get("min_masks", 2)
    do_analyze = unc_cfg.get("do_analyze", True)
    sort_key = unc_cfg.get("sort_key", "integrated")
    compute_cc = unc_cfg.get("compute_cc", True)
    # n_workers：批量并行线程数。TOML 无 None，用 0 表示“自动取 CPU 核数”，
    # 在此翻译为 None 交给 run_batch；1 为串行，>1 为固定线程数。
    n_workers = unc_cfg.get("n_workers", None)
    if n_workers is not None and int(n_workers) <= 0:
        n_workers = None
    cfg_target_labels = unc_cfg.get("target_labels") or None
    base_out = unc_cfg.get("out_dir") or ""
    # compare 方法默认以金标准 labelsTr 作为基准（第一个文件夹）；
    # 其余方法默认只比较各折预测彼此之间的分歧，不引入金标准。
    use_gt_as_base = unc_cfg.get("use_gt_as_base", method == "compare")

    for dataset_name, class_map in zip(train_datasets, class_maps):
        model_folder = os.path.join(
            nnunet_results, dataset_name,
            "{0}__{1}__{2}".format(trainer, plans, configuration))
        fold_dirs = _collect_fold_validation_dirs(model_folder, total_folds)
        if not fold_dirs:
            print("[uncertainty] 跳过 {0}: 未找到任一折的验证输出目录 "
                  "{1}/fold_*/validation".format(dataset_name, model_folder))
            continue

        mask_dirs = list(fold_dirs)
        if use_gt_as_base:
            gt_dir = os.path.join(nnunet_raw, dataset_name, "labelsTr")
            if os.path.isdir(gt_dir):
                mask_dirs = [gt_dir] + mask_dirs  # 基准放在第一个
            elif method == "compare":
                print("[uncertainty] 跳过 {0}: compare 方法需要基准 labelsTr，"
                      "但未找到 {1}。".format(dataset_name, gt_dir))
                continue
            else:
                print("[uncertainty] {0}: 未找到金标准目录 {1}，将只比较各折预测。".format(
                    dataset_name, gt_dir))

        if len(mask_dirs) < min_masks:
            print("[uncertainty] 跳过 {0}: 可用 mask 目录 {1} 个，少于 "
                  "min_masks={2}。".format(dataset_name, len(mask_dirs), min_masks))
            continue

        case_names = _collect_case_names(mask_dirs)
        if not case_names:
            print("[uncertainty] 跳过 {0}: 各验证目录中未找到任何 mask 文件。".format(
                dataset_name))
            continue

        target_labels = (sorted(int(l) for l in cfg_target_labels)
                         if cfg_target_labels else _extract_target_labels(class_map))

        out_dir = (os.path.join(base_out, dataset_name) if base_out
                   else os.path.join(nnunet_raw, dataset_name, "uncertainty"))
        report_csv = os.path.join(out_dir, "uncertainty_ranking.csv")
        log_path = os.path.join(out_dir, "uncertainty_batch.log")

        print("\n{0}".format("=" * 60))
        print("[uncertainty] 数据集 {0}".format(dataset_name))
        print("{0}".format("=" * 60))
        print("  方法: {0}  基准(labelsTr): {1}".format(method, use_gt_as_base))
        print("  mask 目录 ({0} 个):".format(len(mask_dirs)))
        for d in mask_dirs:
            print("    {0}".format(d))
        print("  case 数: {0}  目标标签: {1}".format(len(case_names), target_labels))
        print("  输出目录: {0}".format(out_dir))

        run_batch(
            mask_dirs, out_dir, case_names=case_names,
            filename_template="{case}.nii.gz", method=method,
            plot_axis=plot_axis, n_slices=n_slices, min_masks=min_masks,
            target_labels=target_labels, boundary_dilation=boundary_dilation,
            save_components=save_components, save_consensus=save_consensus,
            save_plots=save_plots, log_path=log_path, n_workers=n_workers)

        if do_analyze:
            analyze_uncertainty_dir(
                out_dir, method=method, sort_key=sort_key,
                report_csv=report_csv, compute_boundary=save_components,
                compute_cc=compute_cc)


def main():
    # ==== 用户手动填写以下参数后直接运行本文件 ====
    # 不同算法的 mask 各放一个文件夹（长度 >= 2，通常 3 个以上）。
    # 同一例数据在各文件夹下使用相同的文件名。
    mask_dirs = [
        r"/path/to/algorithm_A_predictions",
        r"/path/to/algorithm_B_predictions",
        r"/path/to/algorithm_C_predictions",
    ]
    # 各文件夹内的文件名模板，{case} 会被替换为 case 名（可含子路径）。
    filename_template = "{case}.nii.gz"

    # 需要处理的 case 名称：两种方式二选一。
    # 方式一：直接列出（优先生效）。不用时置为 None。
    case_names = None
    # 方式二：从 txt 文本读入（每行一个 case 名，支持 # 注释）。不用时置为 None。
    case_list_txt = r"/path/to/case_list.txt"

    # 输出目录（每例的不确定度 NIfTI、共识 mask 与热力图都写到这里）
    out_dir = r"/path/to/uncertainty_output"

    # 不确定度度量: "entropy"（默认，归一化投票熵）/ "disagreement" / "variance"
    #             / "compare"（以 mask_dirs[0] 为基准，输出基准 mask 的不确定度）
    method = "compare"
    # 绘图切片方向: 0=矢状, 1=冠状, 2=轴向
    plot_axis = 2
    # 拼图中展示的切片数量（自动挑选不确定度最高的若干层）
    n_slices = 9

    # ---- 参与计算的 label ----
    # 不确定度统一按 independent 计算：每个 label 单独做 one-vs-rest 二值不确定度，
    # 再逐体素取最大合成总图。相邻标签（如肺各叶）的公共边界会自然纳入总图。
    # target_labels：用户手动指定参与计算的标签列表；只有列入的标签参与，其余不参与。
    # 置为 None 时自动取所有非零标签。例如只算标签 1、2、3 -> [1, 2, 3]。
    target_labels = None
    # “label 交界带”的膨胀宽度（体素）。额外输出 partition_boundary 分量图与 CSV 统计：
    # 把相邻标签公共边界（叶裂缝附近）的不确定度单独提取，便于单独查看/统计交界处。
    # 取 0 则仅保留严格交界体素，调大则向外拓宽交界带。互不相邻的标签不产生交界统计。
    boundary_dilation = 10
    # 是否额外保存 label 交界分量图（partition_boundary，供 CSV 交界统计）。
    # 只输出总不确定度与该交界分量，不输出各标签分图，节省空间与时间。
    save_components = True
    # 是否保存共识（多数票）标签 *_consensus.nii.gz。有时并不需要，置 False 关闭。
    save_consensus = True
    # 是否绘制并保存热力图（*_slice.png 与 *_montage.png）。有时不需要出图，
    # 置 False 关闭以省时并避免生成图片。
    save_plots = True

    # 是否在批量结束后进行排序分析
    do_analyze = True
    # 排序键: "integrated"（积分不确定度，默认，兼顾高低与区域大小）
    #        / "uncertain_vol"（不确定体积）/ "max" / "mean_nonzero"
    #        / "cc_integrated"（最大连通域积分）/ "cc_vol"（最大连通域体积）
    #        / "cc_mean"（最大连通域均值）
    #        / "boundary_integrated"（label 交界积分）/ "boundary_vol"（label 交界体积）
    #          / "boundary_max"（label 交界最大值）
    sort_key = "integrated"
    # 排序报告 CSV 输出路径
    report_csv = os.path.join(out_dir, "uncertainty_ranking.csv")

    # ---- 分析阶段加速开关 ----
    # compute_cc：是否在 analyze 阶段计算最大连通域指标（cc_*）。连通域分析
    #   (ndimage.label) 是逐例最重的一步。若 sort_key 不依赖 cc_*（如用
    #   "integrated"/"uncertain_vol"），可置 False 跳过以加速；需要 cc_* 时再置 True。
    #   默认 True（保持原行为，结果完整）。
    compute_cc = True

    # n_workers：批量计算的并行线程数。各例相互独立，写入磁盘的结果与串行完全
    #   一致，仅完成顺序不同。None 表示自动取 CPU 核数；1 表示串行（逐例）；
    #   >1 表示固定并行线程数。（若改用 Config_windows.toml + run_from_config
    #   运行，TOML 无 None，对应配置项改用 0 表示自动，语义相同。）
    n_workers = None

    # 批量计算的日志文件路径：跳过/警告/错误等提示写入该文件（追加，带时间戳），
    # 控制台只显示进度条与最终完成信息，不刷屏。置为 None 则不写日志。
    log_path = os.path.join(out_dir, "uncertainty_batch.log")
    # ============================================

    run_batch(mask_dirs, out_dir, case_names=case_names, case_list_txt=case_list_txt,
              filename_template=filename_template, method=method,
              plot_axis=plot_axis, n_slices=n_slices,
              target_labels=target_labels, boundary_dilation=boundary_dilation,
              save_components=save_components, save_consensus=save_consensus,
              save_plots=save_plots, log_path=log_path, n_workers=n_workers)

    if do_analyze:
        analyze_uncertainty_dir(out_dir, method=method, sort_key=sort_key,
                                report_csv=report_csv,
                                compute_boundary=save_components,
                                compute_cc=compute_cc)


if __name__ == "__main__":
    main()
