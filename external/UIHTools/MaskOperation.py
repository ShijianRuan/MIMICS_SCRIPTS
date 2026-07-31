"""
MaskOperation.py
提供对分割 mask 文件的标签值修改功能，支持 .mhd 和 .nii.gz 格式。
"""

import os
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# 内部辅助函数
# ---------------------------------------------------------------------------

def _load_mhd(file_path: str):
    """使用 SimpleITK 读取 .mhd 文件，返回 (numpy_array, sitk_image)。"""
    try:
        import SimpleITK as sitk
    except ImportError:
        raise ImportError("读取 .mhd 文件需要安装 SimpleITK：pip install SimpleITK")

    sitk_image = sitk.ReadImage(file_path)
    data = sitk.GetArrayFromImage(sitk_image)  # shape: (z, y, x)
    return data, sitk_image


def _save_mhd(data: np.ndarray, reference_sitk_image, output_path: str):
    """将 numpy 数组保存为 .mhd 文件，保留原始图像的空间信息。"""
    import SimpleITK as sitk

    out_image = sitk.GetImageFromArray(data)
    out_image.CopyInformation(reference_sitk_image)
    sitk.WriteImage(out_image, output_path)


def _load_nifti(file_path: str):
    """使用 nibabel 读取 .nii 或 .nii.gz 文件，返回 (numpy_array, nib_img)。"""
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("读取 .nii.gz 文件需要安装 nibabel：pip install nibabel")

    nib_img = nib.load(file_path)
    data = np.asarray(nib_img.dataobj)
    return data, nib_img


def _save_nifti(data: np.ndarray, reference_nib_img, output_path: str):
    """将 numpy 数组保存为 .nii.gz 文件，保留原始图像的 header 和 affine。"""
    import nibabel as nib

    new_img = nib.Nifti1Image(data, affine=reference_nib_img.affine, header=reference_nib_img.header)
    nib.save(new_img, output_path)


def _get_spacing_by_array_axes(fmt: str, reference_image) -> tuple:
    """获取与数组轴顺序一致的体素间距（单位 mm）。

    返回顺序与 numpy 数组轴一致：
    - mhd/mha: 数组为 (z, y, x)，返回 (sz, sy, sx)
    - nii/nii.gz: 数组通常为 (x, y, z)，返回 (sx, sy, sz)
    """
    if fmt == "mhd":
        spacing_xyz = tuple(float(v) for v in reference_image.GetSpacing())
        return spacing_xyz[::-1]

    spacing = reference_image.header.get_zooms()[:3]
    return tuple(float(v) for v in spacing)


def _build_mm_ball_structure(radius_mm: float, spacing_by_axis: tuple) -> np.ndarray:
    """按物理半径（mm）构造 3D 结构元。"""
    if radius_mm <= 0:
        return np.ones((1, 1, 1), dtype=bool)

    voxel_radii = [int(np.ceil(radius_mm / s)) for s in spacing_by_axis]
    ranges = [np.arange(-r, r + 1, dtype=float) for r in voxel_radii]
    grid = np.meshgrid(*ranges, indexing="ij")

    dist2 = np.zeros_like(grid[0], dtype=float)
    for axis_grid, axis_spacing in zip(grid, spacing_by_axis):
        dist2 += (axis_grid * axis_spacing) ** 2

    return dist2 <= (radius_mm ** 2 + 1e-12)


# ---------------------------------------------------------------------------
# 公开函数
# ---------------------------------------------------------------------------

def relabel_mask(
    input_path: str,
    output_path: str,
    label_map: dict,
    dtype=None,
):
    """读取 mask 文件，将指定的标签值替换为新值，然后保存结果。

    支持 .mhd、.nii 和 .nii.gz 格式。

    Args:
        input_path  (str): 输入 mask 文件路径，支持 .mhd / .nii / .nii.gz。
        output_path (str): 输出文件路径，格式由扩展名决定，与输入格式可以不同。
        label_map   (dict): 标签替换映射，键为原标签值，值为目标标签值。
                            例如 {1: 5, 3: 0} 表示把标签 1 改为 5，标签 3 改为 0。
        dtype       (numpy dtype, optional): 输出数组的数据类型。
                            默认与输入一致；若替换后的值超出原类型范围，
                            建议手动指定，如 np.uint16。

    Returns:
        np.ndarray: 修改后的 mask 数组。

    Raises:
        ValueError: 不支持的文件格式。
        FileNotFoundError: 输入文件不存在。
    """
    input_path = str(input_path)
    output_path = str(output_path)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在：{input_path}")

    # 判断格式
    def _get_format(path: str) -> str:
        p = path.lower()
        if p.endswith(".mhd") or p.endswith(".mha"):
            return "mhd"
        if p.endswith(".nii.gz") or p.endswith(".nii"):
            return "nii"
        raise ValueError(f"不支持的文件格式，仅支持 .mhd / .mha / .nii / .nii.gz：{path}")

    in_fmt = _get_format(input_path)
    out_fmt = _get_format(output_path)

    # ---------- 读取 ----------
    if in_fmt == "mhd":
        data, ref = _load_mhd(input_path)
    else:
        data, ref = _load_nifti(input_path)

    # ---------- 替换标签 ----------
    original_dtype = data.dtype
    out_dtype = dtype if dtype is not None else original_dtype

    # 使用副本，避免原地操作时映射冲突（例如 {1: 2, 2: 3} 连锁替换问题）
    result = data.copy()
    for old_val, new_val in label_map.items():
        mask = data == old_val
        if mask.any():
            result[mask] = new_val

    result = result.astype(out_dtype)

    # 确保输出目录存在
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # ---------- 保存 ----------
    if out_fmt == "mhd":
        if in_fmt == "mhd":
            _save_mhd(result, ref, output_path)
        else:
            # 输入为 nifti，输出为 mhd：需要新建 SimpleITK 图像
            import SimpleITK as sitk
            out_image = sitk.GetImageFromArray(result)
            # 从 nibabel header 提取间距和原点
            zooms = ref.header.get_zooms()[:3]
            out_image.SetSpacing([float(z) for z in zooms])
            origin = ref.affine[:3, 3].tolist()
            out_image.SetOrigin(origin)
            sitk.WriteImage(out_image, output_path)
    else:
        if in_fmt == "nii":
            _save_nifti(result, ref, output_path)
        else:
            # 输入为 mhd，输出为 nifti：从 SimpleITK 图像构造 affine
            import SimpleITK as sitk
            import nibabel as nib
            spacing = np.array(ref.GetSpacing())       # (x, y, x) → (col, row, slice)
            origin = np.array(ref.GetOrigin())
            direction = np.array(ref.GetDirection()).reshape(3, 3)
            # 构造 affine：将 SimpleITK LPS 方向矩阵转为 nibabel RAS
            affine = np.eye(4)
            affine[:3, :3] = direction * spacing
            affine[:3, 3] = origin
            new_img = nib.Nifti1Image(result, affine=affine)
            nib.save(new_img, output_path)

    print(f"[relabel_mask] 标签替换完成：{label_map}")
    print(f"[relabel_mask] 结果已保存至：{output_path}")
    return result


def batch_relabel_masks(
    input_dir: str,
    output_dir: str,
    label_map: dict,
    dtype=None,
    glob_pattern: str = "**/*.nii.gz",
):
    """批量对目录下所有 mask 文件执行标签替换。

    Args:
        input_dir    (str): 输入目录。
        output_dir   (str): 输出目录（保持原有相对目录结构）。
        label_map    (dict): 同 relabel_mask。
        dtype        (numpy dtype, optional): 同 relabel_mask。
        glob_pattern (str): 文件匹配模式，默认 "**/*.nii.gz"。
                            也可用 "**/*.mhd" 匹配 .mhd 文件。
    Returns:
        list[str]: 成功处理的输出文件路径列表。
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    files = list(input_dir.glob(glob_pattern))

    if not files:
        print(f"[batch_relabel_masks] 未找到匹配文件：{input_dir / glob_pattern}")
        return []

    results = []
    for src in files:
        rel = src.relative_to(input_dir)
        dst = output_dir / rel
        try:
            relabel_mask(str(src), str(dst), label_map, dtype=dtype)
            results.append(str(dst))
        except Exception as e:
            print(f"[batch_relabel_masks] 处理失败 {src}：{e}")

    print(f"[batch_relabel_masks] 共处理 {len(results)}/{len(files)} 个文件。")
    return results


def mask_dilate(
    input_path: str,
    output_path: str,
    target_labels,
    dilate_mm: float,
    dilated_label: int,
    dtype=None,
):
    """读取 mask 文件，对指定标签做按 mm 尺度膨胀，并保存结果。

    膨胀策略：
    1) 先将 target_labels 组成一个联合二值区域；
    2) 按 dilate_mm（毫米）和体素分辨率构造结构元执行膨胀；
    3) 仅对“新膨胀出来”的体素（原本不在 target_labels 内）赋值为 dilated_label；
       原始体素值保持不变。

    Args:
        input_path (str): 输入 mask 文件路径，支持 .mhd/.mha/.nii/.nii.gz。
        output_path (str): 输出文件路径，格式由扩展名决定。
        target_labels (Iterable[int] | int): 需要参与膨胀的标签值（可单个或多个）。
        dilate_mm (float): 膨胀半径（单位 mm），必须 >= 0。
        dilated_label (int): 新增膨胀体素写入的标签值。
        dtype (numpy dtype, optional): 输出数组类型，默认与输入一致。

    Returns:
        np.ndarray: 膨胀后的 mask 数组。
    """
    try:
        from scipy.ndimage import binary_dilation
    except ImportError:
        raise ImportError("mask_dilate 需要 scipy：pip install scipy")

    input_path = str(input_path)
    output_path = str(output_path)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在：{input_path}")
    if dilate_mm < 0:
        raise ValueError(f"dilate_mm 必须 >= 0，当前为：{dilate_mm}")

    if np.isscalar(target_labels):
        target_labels = [int(target_labels)]
    else:
        target_labels = [int(v) for v in target_labels]

    if len(target_labels) == 0:
        raise ValueError("target_labels 不能为空")

    def _get_format(path: str) -> str:
        p = path.lower()
        if p.endswith(".mhd") or p.endswith(".mha"):
            return "mhd"
        if p.endswith(".nii.gz") or p.endswith(".nii"):
            return "nii"
        raise ValueError(f"不支持的文件格式，仅支持 .mhd / .mha / .nii / .nii.gz：{path}")

    in_fmt = _get_format(input_path)
    out_fmt = _get_format(output_path)

    if in_fmt == "mhd":
        data, ref = _load_mhd(input_path)
    else:
        data, ref = _load_nifti(input_path)

    original_dtype = data.dtype
    out_dtype = dtype if dtype is not None else original_dtype

    if data.ndim < 3:
        raise ValueError(f"mask_dilate 仅支持至少 3 维数组，当前维度为：{data.ndim}")

    if data.ndim > 3:
        raise ValueError(
            f"当前输入维度为 {data.ndim}，mask_dilate 仅对 3D mask 生效。"
            "请先选择单个通道/单个时间点再调用。"
        )

    spacing = _get_spacing_by_array_axes(in_fmt, ref)
    structure = _build_mm_ball_structure(float(dilate_mm), spacing)

    source_mask = np.isin(data, target_labels)
    dilated_mask = binary_dilation(source_mask, structure=structure)
    new_voxels = dilated_mask & (~source_mask)

    result = data.copy()
    result[new_voxels] = dilated_label
    result = result.astype(out_dtype)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if out_fmt == "mhd":
        if in_fmt == "mhd":
            _save_mhd(result, ref, output_path)
        else:
            import SimpleITK as sitk
            out_image = sitk.GetImageFromArray(result)
            zooms = ref.header.get_zooms()[:3]
            out_image.SetSpacing([float(z) for z in zooms])
            origin = ref.affine[:3, 3].tolist()
            out_image.SetOrigin(origin)
            sitk.WriteImage(out_image, output_path)
    else:
        if in_fmt == "nii":
            _save_nifti(result, ref, output_path)
        else:
            import nibabel as nib
            spacing_xyz = np.array(ref.GetSpacing())
            origin = np.array(ref.GetOrigin())
            direction = np.array(ref.GetDirection()).reshape(3, 3)
            affine = np.eye(4)
            affine[:3, :3] = direction * spacing_xyz
            affine[:3, 3] = origin
            new_img = nib.Nifti1Image(result, affine=affine)
            nib.save(new_img, output_path)

    print(
        f"[mask_dilate] 膨胀完成：labels={target_labels}, "
        f"dilate_mm={dilate_mm}, dilated_label={dilated_label}"
    )
    print(f"[mask_dilate] 结果已保存至：{output_path}")
    return result


def mask_outer_region(
    input_path: str,
    output_path: str,
    target_labels,
    outer_label: int,
    margin_mm: float = 3.0,
    radius_scale: float = 1.20,
    smooth_mm: float = 6.0,
    axis="auto",
    dtype=None,
):
    """生成包裹指定标签的凸且平滑外区域（弯曲、半径可变），并写入指定标签。

    方法说明：
    - 将 target_labels 合并为二值区域；
    - 沿主轴逐层（2D）计算包络椭圆，椭圆天然凸且边界平滑；
    - 椭圆中心与半径沿主轴做平滑，形成“弯曲且半径可变”的近似圆柱外区域；
    - 仅在原 mask 为 0 的位置写入 outer_label，不覆盖已有标签。

    Args:
        input_path (str): 输入 mask 路径，支持 .mhd/.mha/.nii/.nii.gz。
        output_path (str): 输出路径。
        target_labels (Iterable[int] | int): 需要被包裹的标签值（可单个或多个）。
        outer_label (int): 外包区域写入的标签值。
        margin_mm (float): 在每层包络基础上额外外扩距离（mm）。
        radius_scale (float): 每层包络半径缩放系数（>1 更宽）。
        smooth_mm (float): 沿主轴对中心/半径进行平滑的尺度（mm）。
        axis ("auto" | int): 主轴选择。"auto" 自动选最长延展轴，也可传 0/1/2。
        dtype (numpy dtype, optional): 输出数组 dtype，默认与输入一致。

    Returns:
        np.ndarray: 结果数组。
    """
    try:
        from scipy.ndimage import gaussian_filter1d
    except ImportError:
        raise ImportError("mask_outer_region 需要 scipy：pip install scipy")

    input_path = str(input_path)
    output_path = str(output_path)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在：{input_path}")
    if margin_mm < 0:
        raise ValueError(f"margin_mm 必须 >= 0，当前为：{margin_mm}")
    if radius_scale <= 0:
        raise ValueError(f"radius_scale 必须 > 0，当前为：{radius_scale}")
    if smooth_mm < 0:
        raise ValueError(f"smooth_mm 必须 >= 0，当前为：{smooth_mm}")

    if np.isscalar(target_labels):
        target_labels = [int(target_labels)]
    else:
        target_labels = [int(v) for v in target_labels]
    if len(target_labels) == 0:
        raise ValueError("target_labels 不能为空")

    def _get_format(path: str) -> str:
        p = path.lower()
        if p.endswith(".mhd") or p.endswith(".mha"):
            return "mhd"
        if p.endswith(".nii.gz") or p.endswith(".nii"):
            return "nii"
        raise ValueError(f"不支持的文件格式，仅支持 .mhd / .mha / .nii / .nii.gz：{path}")

    in_fmt = _get_format(input_path)
    out_fmt = _get_format(output_path)

    if in_fmt == "mhd":
        data, ref = _load_mhd(input_path)
    else:
        data, ref = _load_nifti(input_path)

    if data.ndim != 3:
        raise ValueError(f"mask_outer_region 仅支持 3D mask，当前维度为：{data.ndim}")

    spacing = _get_spacing_by_array_axes(in_fmt, ref)
    source_mask = np.isin(data, target_labels)
    if not source_mask.any():
        raise ValueError(f"输入中未找到 target_labels={target_labels}")

    original_dtype = data.dtype
    out_dtype = dtype if dtype is not None else original_dtype

    # 自动主轴：选择目标区域延展范围最大的轴
    if axis == "auto":
        idx = np.argwhere(source_mask)
        extents = idx.max(axis=0) - idx.min(axis=0) + 1
        main_axis = int(np.argmax(extents))
    else:
        main_axis = int(axis)
        if main_axis not in (0, 1, 2):
            raise ValueError(f"axis 仅支持 'auto' 或 0/1/2，当前为：{axis}")

    source_moved = np.moveaxis(source_mask, main_axis, 0)  # (S, H, W)
    spacing_moved = [spacing[main_axis]] + [spacing[i] for i in range(3) if i != main_axis]
    spacing_axis = float(spacing_moved[0])
    spacing_h = float(spacing_moved[1])
    spacing_w = float(spacing_moved[2])

    num_slices, h, w = source_moved.shape
    centers_h = np.full(num_slices, np.nan, dtype=float)
    centers_w = np.full(num_slices, np.nan, dtype=float)
    radii_h = np.full(num_slices, np.nan, dtype=float)
    radii_w = np.full(num_slices, np.nan, dtype=float)
    valid = np.zeros(num_slices, dtype=bool)

    margin_h_pix = margin_mm / spacing_h
    margin_w_pix = margin_mm / spacing_w

    for s in range(num_slices):
        pts = np.argwhere(source_moved[s])
        if pts.shape[0] < 3:
            continue

        valid[s] = True
        center = pts.mean(axis=0)  # (h, w)
        centers_h[s], centers_w[s] = float(center[0]), float(center[1])

        demean = pts.astype(float) - center[None, :]
        cov = (demean.T @ demean) / max(pts.shape[0] - 1, 1)
        cov += np.eye(2) * 1e-6
        eigvals, eigvecs = np.linalg.eigh(cov)

        proj = demean @ eigvecs
        q = 0.98
        base_r0 = max(float(np.quantile(np.abs(proj[:, 0]), q)), 1.0)
        base_r1 = max(float(np.quantile(np.abs(proj[:, 1]), q)), 1.0)

        axis_r = eigvecs[:, 0]
        axis_c = eigvecs[:, 1]
        radii_h[s] = radius_scale * (
            abs(axis_r[0]) * base_r0 + abs(axis_c[0]) * base_r1
        ) + margin_h_pix
        radii_w[s] = radius_scale * (
            abs(axis_r[1]) * base_r0 + abs(axis_c[1]) * base_r1
        ) + margin_w_pix

    valid_idx = np.where(valid)[0]
    if valid_idx.size == 0:
        raise ValueError("未能从目标标签中提取有效切片")

    if valid_idx.size == 1:
        only = int(valid_idx[0])
        centers_h[:] = centers_h[only]
        centers_w[:] = centers_w[only]
        radii_h[:] = max(radii_h[only], 1.0)
        radii_w[:] = max(radii_w[only], 1.0)
        fill_start, fill_end = 0, num_slices - 1
    else:
        s_all = np.arange(num_slices)
        centers_h = np.interp(s_all, valid_idx, centers_h[valid])
        centers_w = np.interp(s_all, valid_idx, centers_w[valid])
        radii_h = np.interp(s_all, valid_idx, radii_h[valid])
        radii_w = np.interp(s_all, valid_idx, radii_w[valid])
        fill_start, fill_end = int(valid_idx.min()), int(valid_idx.max())

    sigma_slices = smooth_mm / spacing_axis if spacing_axis > 0 else 0.0
    if sigma_slices > 0:
        centers_h = gaussian_filter1d(centers_h, sigma=sigma_slices, mode="nearest")
        centers_w = gaussian_filter1d(centers_w, sigma=sigma_slices, mode="nearest")
        radii_h = gaussian_filter1d(radii_h, sigma=sigma_slices, mode="nearest")
        radii_w = gaussian_filter1d(radii_w, sigma=sigma_slices, mode="nearest")

    radii_h = np.clip(radii_h, 1.0, None)
    radii_w = np.clip(radii_w, 1.0, None)

    rr = np.arange(h, dtype=float)[:, None]
    cc = np.arange(w, dtype=float)[None, :]
    outer_moved = np.zeros_like(source_moved, dtype=bool)

    for s in range(fill_start, fill_end + 1):
        dh = (rr - centers_h[s]) / radii_h[s]
        dw = (cc - centers_w[s]) / radii_w[s]
        slice_ellipse = (dh * dh + dw * dw) <= 1.0
        outer_moved[s] = slice_ellipse | source_moved[s]

    outer_mask = np.moveaxis(outer_moved, 0, main_axis)

    # 不覆盖已有标签：仅在背景(0)中写入外包区域
    write_mask = outer_mask & (data == 0)

    result = data.copy()
    result[write_mask] = outer_label
    result = result.astype(out_dtype)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if out_fmt == "mhd":
        if in_fmt == "mhd":
            _save_mhd(result, ref, output_path)
        else:
            import SimpleITK as sitk
            out_image = sitk.GetImageFromArray(result)
            zooms = ref.header.get_zooms()[:3]
            out_image.SetSpacing([float(z) for z in zooms])
            origin = ref.affine[:3, 3].tolist()
            out_image.SetOrigin(origin)
            sitk.WriteImage(out_image, output_path)
    else:
        if in_fmt == "nii":
            _save_nifti(result, ref, output_path)
        else:
            import nibabel as nib
            spacing_xyz = np.array(ref.GetSpacing())
            origin = np.array(ref.GetOrigin())
            direction = np.array(ref.GetDirection()).reshape(3, 3)
            affine = np.eye(4)
            affine[:3, :3] = direction * spacing_xyz
            affine[:3, 3] = origin
            new_img = nib.Nifti1Image(result, affine=affine)
            nib.save(new_img, output_path)

    print(
        f"[mask_outer_region] 完成：labels={target_labels}, "
        f"outer_label={outer_label}, margin_mm={margin_mm}, "
        f"radius_scale={radius_scale}, smooth_mm={smooth_mm}, axis={main_axis}"
    )
    print(f"[mask_outer_region] 结果已保存至：{output_path}")
    return result




def mask_multilabel_dilate(
    input_path: str,
    output_path: str,
    target_labels,
    dilate_mm: float,
    dtype=None,
):
    """读取目录下所有 mask 文件，对多个标签同时进行按 mm 尺度膨胀，并保存结果。

    膨胀策略（同时膨胀，避免先后顺序导致的不公平扩张）：
    1) 对每个 target_label，计算其到所有体素的欧氏距离变换（单位 mm）；
    2) 对于背景体素（值为 0），找到距离最近的 target_label；
    3) 若最近距离 <= dilate_mm，则将该体素赋值为对应的最近 label；
    4) 所有 target_labels 同时竞争填充，距离相同时取标签值较小者；
    5) 原始非零标签保持不变。

    支持输入目录下的 .mhd / .mha / .nii / .nii.gz 文件批量处理。

    Args:
        input_path  (str): 输入目录路径，或单个 mask 文件路径。
        output_path (str): 输出目录路径，或单个输出文件路径（与输入对应）。
        target_labels (Iterable[int] | int): 参与膨胀的标签值列表。
        dilate_mm   (float): 膨胀半径（单位 mm），必须 >= 0。
        dtype       (numpy dtype, optional): 输出数组类型，默认与输入一致。

    Returns:
        list[str]: 成功处理的输出文件路径列表（目录模式），或单文件时返回 [output_path]。
    """
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        raise ImportError("mask_multilabel_dilate 需要 scipy：pip install scipy")

    if np.isscalar(target_labels):
        target_labels = [int(target_labels)]
    else:
        target_labels = [int(v) for v in target_labels]
    if len(target_labels) == 0:
        raise ValueError("target_labels 不能为空")
    if dilate_mm < 0:
        raise ValueError(f"dilate_mm 必须 >= 0，当前为：{dilate_mm}")

    def _get_format(path: str) -> str:
        p = path.lower()
        if p.endswith(".mhd") or p.endswith(".mha"):
            return "mhd"
        if p.endswith(".nii.gz") or p.endswith(".nii"):
            return "nii"
        raise ValueError(f"不支持的文件格式，仅支持 .mhd / .mha / .nii / .nii.gz：{path}")

    def _process_single(in_file: str, out_file: str):
        in_fmt = _get_format(in_file)
        out_fmt = _get_format(out_file)

        if in_fmt == "mhd":
            data, ref = _load_mhd(in_file)
        else:
            data, ref = _load_nifti(in_file)

        if data.ndim != 3:
            raise ValueError(f"mask_multilabel_dilate 仅支持 3D mask，当前维度为：{data.ndim}")

        spacing = _get_spacing_by_array_axes(in_fmt, ref)   # (s0, s1, s2) 与数组轴对应

        original_dtype = data.dtype
        out_dtype = dtype if dtype is not None else original_dtype

        # 背景体素位置
        background_mask = (data == 0)

        # 用于存储每个背景体素的"最近 label"及其距离（mm）
        best_dist = np.full(data.shape, np.inf, dtype=np.float32)
        best_label = np.zeros(data.shape, dtype=np.int32)

        for lbl in target_labels:
            # 前景 = 当前 label 所在体素；计算到前景的最近距离
            foreground = (data == lbl)
            if not foreground.any():
                continue

            # distance_transform_edt 计算到最近前景体素的欧氏距离（体素单位）
            # sampling 参数指定各轴的体素尺寸（mm），使结果直接为 mm
            dist_mm = distance_transform_edt(~foreground, sampling=spacing).astype(np.float32)

            # 在背景区域内，用距离更近的 label 覆盖
            update = background_mask & (dist_mm < best_dist)
            best_dist[update] = dist_mm[update]
            best_label[update] = lbl

        # 只有距离 <= dilate_mm 的背景体素才被填充
        fill_mask = background_mask & (best_dist <= dilate_mm)

        result = data.copy()
        result[fill_mask] = best_label[fill_mask]
        result = result.astype(out_dtype)

        out_dir = os.path.dirname(out_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        if out_fmt == "mhd":
            if in_fmt == "mhd":
                _save_mhd(result, ref, out_file)
            else:
                import SimpleITK as sitk
                out_image = sitk.GetImageFromArray(result)
                zooms = ref.header.get_zooms()[:3]
                out_image.SetSpacing([float(z) for z in zooms])
                origin = ref.affine[:3, 3].tolist()
                out_image.SetOrigin(origin)
                sitk.WriteImage(out_image, out_file)
        else:
            if in_fmt == "nii":
                _save_nifti(result, ref, out_file)
            else:
                import nibabel as nib
                spacing_xyz = np.array(ref.GetSpacing())
                origin = np.array(ref.GetOrigin())
                direction = np.array(ref.GetDirection()).reshape(3, 3)
                affine = np.eye(4)
                affine[:3, :3] = direction * spacing_xyz
                affine[:3, 3] = origin
                new_img = nib.Nifti1Image(result, affine=affine)
                nib.save(new_img, out_file)

        filled_count = int(fill_mask.sum())
        print(
            f"[mask_multilabel_dilate] {os.path.basename(in_file)} → "
            f"新填充体素数={filled_count}, dilate_mm={dilate_mm}"
        )
        print(f"[mask_multilabel_dilate] 已保存至：{out_file}")
        return result

    # ---- 判断是目录模式还是单文件模式 ----
    input_path = str(input_path)
    output_path = str(output_path)

    if os.path.isdir(input_path):
        from pathlib import Path
        in_dir = Path(input_path)
        out_dir = Path(output_path)
        # 收集所有支持格式的文件（.nii.gz 需先排除 .nii 然后再合并）
        files = []
        for pat in ("**/*.mhd", "**/*.mha", "**/*.nii.gz", "**/*.nii"):
            files.extend(in_dir.glob(pat))
        # 去重并排序
        seen = set()
        unique_files = []
        for f in sorted(files):
            if str(f) not in seen:
                seen.add(str(f))
                unique_files.append(f)

        if not unique_files:
            print(f"[mask_multilabel_dilate] 目录下未找到支持格式的 mask 文件：{input_path}")
            return []

        results = []
        for src in unique_files:
            rel = src.relative_to(in_dir)
            dst = out_dir / rel
            try:
                _process_single(str(src), str(dst))
                results.append(str(dst))
            except Exception as e:
                print(f"[mask_multilabel_dilate] 处理失败 {src}：{e}")

        print(f"[mask_multilabel_dilate] 共处理 {len(results)}/{len(unique_files)} 个文件。")
        return results
    else:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"输入文件不存在：{input_path}")
        _process_single(input_path, output_path)
        return [output_path]


import nibabel as nib
import numpy as np
import os

def merge_masks_to_single_label(input_mask_paths, output_path, glob_pattern="*.nii.gz"):
    """
    读取多个nii.gz格式的二分类mask，拼接为一个多分类mask
    规则：第1个mask label=1，第2个=2，第3个=3...依次递增
    
    参数:
        input_mask_paths (list | str): 
            - 当为 list 时：输入mask文件路径列表 [path1, path2, ...]
            - 当为 str 时：输入文件夹路径，自动扫描该目录下匹配 glob_pattern 的所有文件（按文件名排序）
        output_path (str): 输出合并后mask的保存路径
        glob_pattern (str): 当 input_mask_paths 为目录时，用于匹配文件的 glob 模式，默认 "*.nii.gz"
    返回:
        nib.Nifti1Image: 合并后的mask图像对象
    异常:
        ValueError: 输入文件为空、尺寸不匹配、空间信息不匹配时抛出
    """
    # 0. 如果传入的是目录，自动扫描目录下的 mask 文件
    if isinstance(input_mask_paths, str) and os.path.isdir(input_mask_paths):
        from pathlib import Path
        mask_dir = Path(input_mask_paths)
        input_mask_paths = sorted([str(p) for p in mask_dir.glob(glob_pattern)])
        if not input_mask_paths:
            raise ValueError(f"错误：目录 {mask_dir} 下未找到匹配 '{glob_pattern}' 的文件！")
        print(f"📂 从目录扫描到 {len(input_mask_paths)} 个文件：")
        for p in input_mask_paths:
            print(f"   - {os.path.basename(p)}")
        print()

    # 1. 基础校验
    if not input_mask_paths:
        raise ValueError("错误：未传入任何输入mask文件路径！")
    
    # 2. 读取第一个mask作为基准（尺寸、仿射矩阵、头信息）
    base_img = nib.load(input_mask_paths[0])
    base_data = base_img.get_fdata()
    base_affine = base_img.affine
    base_header = base_img.header
    
    # 初始化合并后的mask数组（与基准mask尺寸一致）
    merged_data = np.zeros_like(base_data, dtype=np.int16)
    
    # 3. 遍历所有mask，按顺序赋值label
    for idx, mask_path in enumerate(input_mask_paths):
        # 当前mask的label值 = 索引+1（从1开始）
        current_label = idx + 1
        
        # 读取mask
        img = nib.load(mask_path)
        img_data = img.get_fdata()
        
        # 校验：所有mask必须尺寸完全相同
        if img_data.shape != base_data.shape:
            raise ValueError(f"错误：文件 {os.path.basename(mask_path)} 尺寸不匹配！\n"
                             f"基准尺寸: {base_data.shape}\n"
                             f"当前文件尺寸: {img_data.shape}")
        
        # 校验：所有mask必须空间信息一致（仿射矩阵）
        if not np.allclose(img.affine, base_affine):
            raise ValueError(f"错误：文件 {os.path.basename(mask_path)} 空间仿射矩阵不匹配！")
        
        # 将当前mask中为1的区域赋值为对应label
        merged_data[img_data == 1] = current_label
        print(f"✅ 已处理：{os.path.basename(mask_path)} → Label = {current_label}")
    
    # 4. 保存合并后的mask
    merged_img = nib.Nifti1Image(merged_data, base_affine, base_header)
    nib.save(merged_img, output_path)
    
    print(f"\n🎉 合并完成！合并后mask已保存至：{output_path}")
    print(f"📊 合并信息：总类别数 = {len(input_mask_paths)}")
    print(f"📊 最终mask数据类型：{merged_data.dtype}")
    print(f"📊 最终mask尺寸：{merged_data.shape}")
    
    return merged_img






# ---------------------------------------------------------------------------
# 直接运行入口：在此处手动填写参数后运行本文件即可
# ---------------------------------------------------------------------------

def do_relabel_mask():

    # ====== 在此处修改参数 ======

    input_path  = r"E:\test\spine\scoliosis\t2_mx_1mm_mask.mhd"   # 输入 mask 文件路径（.mhd / .nii / .nii.gz）
    output_path = r"E:\test\spine\scoliosis\t2_mx_1mm_process.mhd"  # 输出文件路径

    # 标签替换映射：{原标签值: 新标签值, ...}
    # 例如：把标签 1 改为 5，把标签 3 改为 0
    label_map = {
        75: 60,
        74: 60,
        73: 60,
        72: 60,
        71: 60,
        70: 60,
        69: 60,
        68: 60,
        67: 60,
        66: 60,
        65: 60,
        64: 60,
        63: 60,
        62: 60,
        61: 60,
        60: 60,
        59: 60,
        58: 60,
        57: 60,
        56: 60,
        55: 60,
        54: 60,
        53: 60,
        52: 60,
        51: 60,
        50: 0,
        22: 0,
        5: 0,
        6: 0,
        23: 0,
        24: 0,
        46: 0,
        47: 0,
        11: 0,
        34: 0,
        22: 0,
        25: 0,
        13: 0,
        16: 0,
        38: 0,
        15: 0,
        3: 0,
        48: 0,
        49: 0,
        35: 0,
        10: 0,
        12: 0,
        8: 0,
        7: 0,
        2: 0,
        14: 0,
        26: 0,
        27: 0,
        28: 0,
        29: 0,
        39: 0,
        41: 0,
        17: 0,
        1: 0,
        40: 0,
        42: 0,
        44: 0,
        43: 0,
        45: 0,
        30: 0,
        31: 0,
        32: 0,
        33: 0,
        9: 0,
    }

    # 输出数据类型，None 表示与输入一致；如需指定可写 np.uint8 / np.uint16 等
    out_dtype = None

    # ============================

    relabel_mask(input_path, output_path, label_map, dtype=out_dtype)




def do_mask_dilate():

    input_path = r"E:\test\spine\segmentation\wholespine_stir_process.mhd"
    output_path = r"E:\test\spine\segmentation\wholespine_stir_dilate.mhd"
    target_labels = [20, 21, 60]
    dilate_mm = 5.0
    dilated_label = 1

    mask_dilate(input_path, output_path, target_labels, dilate_mm, dilated_label)


def do_mask_multilabel_dilate():

    input_path = r"E:\test\CTcoase\labelsTr"
    output_path = r"E:\test\CTcoase\labelsTr_dilate"
    target_labels = [1,2,3,4,5,6,7,8,9,10,11,12]
    dilate_mm = 5.0
    mask_multilabel_dilate(input_path, output_path, target_labels, dilate_mm)



def do_mask_outer_region():

    input_path = r"E:\test\spine\scoliosis\t2_mx_1mm_process.mhd"
    output_path = r"E:\test\spine\scoliosis\t2_mx_1mm_region.mhd"
    target_labels = [20, 21, 60]
    outer_label = 1

    mask_outer_region(input_path, output_path, target_labels, outer_label)



def do_combine_multiply_masks():
    # ========== 用户自定义参数 ==========
    # 方式一：手动指定mask文件路径列表（按顺序排列，label从1开始递增）
    # INPUT_MASKS = [
    #     "D:/code/segmentation/segmentations/lung_lower_lobe_left.nii.gz",
    #     "D:/code/segmentation/segmentations/lung_lower_lobe_right.nii.gz",
    #     "D:/code/segmentation/segmentations/lung_middle_lobe_right.nii.gz",
    #     "D:/code/segmentation/segmentations/lung_upper_lobe_left.nii.gz",
    #     "D:/code/segmentation/segmentations/lung_upper_lobe_right.nii.gz",
    #     # 可继续添加更多mask
    # ]

    # 方式二：传入文件夹路径，自动扫描目录下所有mask文件（按文件名排序）
    INPUT_MASKS = r"E:\RTAI_AutoSeg\test_data\Abdomen_nii\segmentations/"
    
    # 输出合并后mask的保存路径
    OUTPUT_MASK = r"E:\RTAI_AutoSeg\test_data\Abdomen_nii\segmentation.nii.gz"
    # ====================================
    
    # 执行合并
    merge_masks_to_single_label(INPUT_MASKS, OUTPUT_MASK)





if __name__ == "__main__":

    #do_relabel_mask()
    # do_mask_dilate()
    #do_mask_outer_region()
    #do_combine_multiply_masks()

    do_mask_multilabel_dilate()

    print("end!")