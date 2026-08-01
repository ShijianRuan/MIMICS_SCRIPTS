
import os
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from tqdm import tqdm



"""
第一部分，将高分辨率图像和mask插值到低分辨率

"""


def _build_resampled_affine(orig_affine, current_spacing, target_spacing_arr):
    """根据原始 affine、原始 spacing 和目标 spacing 构建重采样后的 affine。

    对退化 affine（3×3 子矩阵全零或含 NaN）做降级处理：
    用目标 spacing 构建对角 affine，保留原始原点。
    """
    affine_3x3 = orig_affine[:3, :3]
    is_degenerate = (
        np.any(np.isnan(affine_3x3))
        or np.all(np.abs(affine_3x3) < 1e-12)
    )

    if is_degenerate:
        new_affine = np.eye(4)
        new_affine[0, 0] = target_spacing_arr[0]
        new_affine[1, 1] = target_spacing_arr[1]
        new_affine[2, 2] = target_spacing_arr[2]
        origin = orig_affine[:3, 3]
        if not np.any(np.isnan(origin)):
            new_affine[:3, 3] = origin
        return new_affine

    # 正常路径：缩放 affine 旋转/缩放列
    zoom_factors = np.where(current_spacing == 0, 1.0, current_spacing) / target_spacing_arr
    new_affine = orig_affine.copy()
    for i in range(3):
        new_affine[:3, i] = orig_affine[:3, i] / zoom_factors[i]
    return new_affine


def resample_nifti(img, target_spacing, order=1):
    """Resample a nifti image to the target spacing.
    Args:
        img: nibabel Nifti1Image
        target_spacing: list/array of target voxel spacing, e.g. [3.0, 3.0, 3.0]
        order: interpolation order, 0 for nearest (masks), 1 for linear (images)
    Returns:
        resampled nibabel Nifti1Image
    """
    data = img.get_fdata()
    current_spacing = np.array(img.header.get_zooms()[:3], dtype=np.float64)
    current_spacing = np.where(current_spacing == 0, 1.0, current_spacing)
    target_spacing = np.array(target_spacing, dtype=np.float64)
    zoom_factors = current_spacing / target_spacing

    resampled_data = zoom(data, zoom_factors, order=order)

    new_affine = _build_resampled_affine(img.affine, current_spacing, target_spacing)
    return nib.Nifti1Image(resampled_data, new_affine)


def resample_and_save_image(src_path, dst_path, target_spacing=None):
    """Load an image, optionally resample to target_spacing, and save to dst_path."""
    if target_spacing is None:
        shutil.copy(src_path, dst_path)
    else:
        img = nib.load(src_path)
        resampled = resample_nifti(img, target_spacing, order=1)
        nib.save(resampled, dst_path)





def _resample_nifti_fast(img, target_spacing, order=1, dtype=None):
    """A lighter-weight resampler used by the accelerated helper functions."""
    data = np.asarray(img.dataobj, dtype=dtype)
    current_spacing = np.array(img.header.get_zooms()[:3], dtype=np.float64)
    current_spacing = np.where(current_spacing == 0, 1.0, current_spacing)
    target_spacing = np.array(target_spacing, dtype=np.float64)
    zoom_factors = current_spacing / target_spacing

    resampled_data = zoom(
        data,
        zoom_factors,
        order=order,
        prefilter=order > 1,
    )

    new_affine = _build_resampled_affine(img.affine, current_spacing, target_spacing)
    return nib.Nifti1Image(resampled_data, new_affine)


def resample_and_save_image_fast(src_path, dst_path, target_spacing=None, image_dtype=np.float32):
    """Faster version of `resample_and_save_image`.

    Optimization points:
    1. Use `np.asarray(img.dataobj)` to avoid the extra float64 copy from `get_fdata()`.
    2. Convert image data to float32 by default to reduce memory pressure.
    3. Keep the original fast-copy path when `target_spacing` is None.
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)

    if target_spacing is None:
        shutil.copy(src_path, dst_path)
        return

    img = nib.load(src_path)
    resampled = _resample_nifti_fast(img, target_spacing, order=1, dtype=image_dtype)
    nib.save(resampled, dst_path)






"""
第二部分，将低分辨率的预测图像和多标签mask插值回原始（高）分辨率。

背景
----
训练时将原始高分辨率（如1.5mm）数据下采样到低分辨率（如3.0mm）进行训练和预测，
预测完成后需要将结果插值回原始分辨率。

核心难点：多标签mask的上采样
---------------------------
1. 不能直接对多标签mask（值为0,1,2,3,...）做线性插值 → 不同标签会混叠
2. 不能用最近邻插值 → 会产生锯齿状边界
3. 逐标签线性插值再拼接 → 可能出现重叠或缝隙

解决方案：One-Hot编码 → 线性插值 → 增量ArgMax
---------------------------------------------
1. 将多标签mask转为One-Hot编码（每个标签一个二值通道）
2. 对每个二值通道做线性插值（得到 [0,1] 之间的概率图，边界平滑）
3. 对所有通道取 ArgMax → 每个体素分配到概率最大的标签
   - 无缝隙：每个体素都会被分配（包括背景）
   - 无重叠：ArgMax 只选一个赢家
   - 光滑边界：线性插值保证了平滑过渡

并行加速
--------
- 使用 ThreadPoolExecutor 并行插值各标签通道（scipy.ndimage.zoom 的 C 内核释放 GIL）
- 使用增量 ArgMax 策略：只需保持 best_prob + best_label 两个体积
- 内存占用与标签总数无关（O(1) 额外空间），仅与并发线程数相关
"""



def _get_valid_affine(nifti_img):
    """从 NIfTI 图像中提取有效的 affine 矩阵。

    当 affine 的 3×3 旋转/缩放子矩阵全零或含 NaN（退化 affine）时，
    从 header zooms 构建对角 affine 作为降级方案。
    """
    affine = nifti_img.affine.copy()
    affine_3x3 = affine[:3, :3]
    is_degenerate = (
        np.any(np.isnan(affine_3x3))
        or np.all(np.abs(affine_3x3) < 1e-12)
    )

    if not is_degenerate:
        return affine

    # 降级：从 header zooms 构建对角 affine
    zooms = nifti_img.header.get_zooms()
    if len(zooms) >= 3:
        spacing = [float(z) if z > 0 else 1.0 for z in zooms[:3]]
    else:
        spacing = [1.0, 1.0, 1.0]

    new_affine = np.eye(4)
    new_affine[0, 0] = spacing[0]
    new_affine[1, 1] = spacing[1]
    new_affine[2, 2] = spacing[2]
    # 保留原点（若有效）
    origin = affine[:3, 3]
    if not np.any(np.isnan(origin)):
        new_affine[:3, 3] = origin

    print(f"[WARNING] NIfTI affine 退化（3×3 全零或含 NaN），"
          f"已从 header zooms 构建降级 affine: spacing={spacing}")
    return new_affine


# ---------------------------------------------------------------------------
#  图像上采样
# ---------------------------------------------------------------------------

def resample_image_to_original(lowres_path, ref_path, output_path):
    """将低分辨率图像线性插值回原始分辨率。

    Parameters
    ----------
    lowres_path : str or Path
        低分辨率图像路径（.nii.gz）
    ref_path : str or Path
        原始分辨率参考图像路径，用于获取目标 shape 和 affine
    output_path : str or Path
        输出图像路径
    """
    lowres_img = nib.load(str(lowres_path))
    ref_img = nib.load(str(ref_path))
    ref_affine = _get_valid_affine(ref_img)

    lowres_data = np.asarray(lowres_img.dataobj, dtype=np.float32)
    target_shape = ref_img.shape[:3]

    if lowres_data.shape[:3] == target_shape:
        shutil.copy(str(lowres_path), str(output_path))
        return

    zoom_factors = (
        np.array(target_shape, dtype=np.float64)
        / np.array(lowres_data.shape[:3], dtype=np.float64)
    )
    resampled = zoom(lowres_data, zoom_factors, order=1, prefilter=False)

    nib.save(nib.Nifti1Image(resampled, ref_affine), str(output_path))


# ---------------------------------------------------------------------------
#  多标签mask上采样 — 最近邻（极速版）
# ---------------------------------------------------------------------------

def resample_mask_to_original_nearest(lowres_mask_path, ref_path, output_path):
    """将低分辨率多标签mask用最近邻插值回原始分辨率。

    只做一次 scipy.ndimage.zoom(order=0)，速度极快（通常 < 1s），
    代价是边界呈锯齿状（阶梯）。对大多数分割评价场景足够。

    Parameters
    ----------
    lowres_mask_path : str or Path
        低分辨率mask路径（.nii.gz）
    ref_path : str or Path
        原始分辨率参考图像路径，用于获取目标 shape 和 affine
    output_path : str or Path
        输出mask路径
    """
    lowres_mask = nib.load(str(lowres_mask_path))
    ref_img = nib.load(str(ref_path))
    ref_affine = _get_valid_affine(ref_img)

    mask_data = np.asarray(lowres_mask.dataobj)
    if mask_data.dtype.kind == 'f':
        mask_data = np.round(mask_data).astype(np.int16)
    else:
        mask_data = mask_data.astype(np.int16)

    target_shape = ref_img.shape[:3]

    if mask_data.shape[:3] == target_shape:
        shutil.copy(str(lowres_mask_path), str(output_path))
        return

    zoom_factors = (
        np.array(target_shape, dtype=np.float64)
        / np.array(mask_data.shape[:3], dtype=np.float64)
    )
    resampled = zoom(mask_data, zoom_factors, order=0)  # nearest-neighbor

    max_label = int(resampled.max())
    out_dtype = np.uint8 if max_label <= 255 else np.uint16
    nib.save(
        nib.Nifti1Image(resampled.astype(out_dtype), ref_affine),
        str(output_path),
    )


# ---------------------------------------------------------------------------
#  多标签mask上采样 — GPU加速光滑版（推荐）
# ---------------------------------------------------------------------------

def resample_mask_to_original_torch(
    lowres_mask_path,
    ref_path,
    output_path,
    device="cuda",
    batch_size=4,
):
    """用 PyTorch GPU 三线性插值将低分辨率多标签mask上采样回原始分辨率。

    算法：One-Hot → GPU trilinear interpolate (float16) → ArgMax
    - 与 CPU 版 smooth（resample_multilabel_mask_to_original）结果质量相同：
      边界光滑、无锯齿、无缝隙
    - 速度快 10-30 倍（典型 < 1s vs 10-20s）

    内存优化要点
    ----------
    - 批内插值在 float16 下执行，显存占用减半
    - best_prob 保持 float32 保证 argmax 比较精度
    - 默认 batch_size=4，跟推理后的模型共占内存时不会爆显存

    Parameters
    ----------
    lowres_mask_path : str or Path
        低分辨率mask路径（.nii.gz），标签值为整数 0,1,2,...
    ref_path : str or Path
        原始分辨率参考图像路径，用于获取目标 shape 和 affine
    output_path : str or Path
        输出mask路径
    device : str
        PyTorch 设备，默认 "cuda"。若无 GPU 则自动回落到 CPU。
    batch_size : int
        每批并行插值的标签数。默认 4。
        内存估算：batch_size 标签 × 目标体积 × 2字节（float16）。
        例：4×512×512×400×2B ≈ 800MB。OOM 时可进一步减小。
    """
    import torch
    import torch.nn.functional as F

    lowres_mask = nib.load(str(lowres_mask_path))
    ref_img = nib.load(str(ref_path))
    ref_affine = _get_valid_affine(ref_img)

    mask_data = np.asarray(lowres_mask.dataobj)
    if mask_data.dtype.kind == 'f':
        mask_data = np.round(mask_data).astype(np.int16)
    else:
        mask_data = mask_data.astype(np.int16)

    target_shape = ref_img.shape[:3]

    if mask_data.shape[:3] == target_shape:
        shutil.copy(str(lowres_mask_path), str(output_path))
        return

    labels = np.unique(mask_data)
    labels = labels[labels > 0]  # 排除背景

    if len(labels) == 0:
        result = np.zeros(target_shape, dtype=np.uint8)
        nib.save(nib.Nifti1Image(result, ref_affine), str(output_path))
        return

    # 确定设备
    import torch.cuda
    torch_device = torch.device(device if torch.cuda.is_available() else 'cpu')
    target_shape_list = list(target_shape)

    with torch.no_grad():
        # 背景通道插值，初始化 best_prob / best_label
        # best_prob 用 float32 保证比较精度
        bg = torch.from_numpy(
            (mask_data == 0).astype(np.float32)
        ).unsqueeze(0).unsqueeze(0).to(torch_device)
        best_prob = F.interpolate(
            bg, size=target_shape_list, mode='trilinear', align_corners=False
        )[0, 0]  # float32, shape=(D',H',W')
        del bg
        best_label = torch.zeros(
            target_shape_list, dtype=torch.int16, device=torch_device
        )

        # 分批处理前景标签，批内运算用 float16 节省显存
        for i in range(0, len(labels), batch_size):
            batch_labels = labels[i : i + batch_size]
            n = len(batch_labels)

            # 构造 one-hot batch (CPU numpy，float16 节省 CPU 内存)
            binary_np = np.stack(
                [(mask_data == int(lbl)).astype(np.float16) for lbl in batch_labels],
                axis=0,
            )  # (n, D, H, W)

            # 送入 GPU，保持 float16
            batch_tensor = (
                torch.from_numpy(binary_np)
                .unsqueeze(0)  # (1, n, D, H, W)
                .to(torch_device)   # float16 在 GPU
            )
            del binary_np

            # GPU 三线性插值，float16 显存占用减半
            batch_prob = F.interpolate(
                batch_tensor,
                size=target_shape_list,
                mode='trilinear',
                align_corners=False,
            )[0]  # (n, D', H', W'), float16
            del batch_tensor

            # 增量 ArgMax：转为 float32 再与 best_prob 比较保证精度
            batch_prob_f32 = batch_prob.float()
            del batch_prob
            for j, lbl in enumerate(batch_labels):
                update = batch_prob_f32[j] > best_prob
                best_prob[update] = batch_prob_f32[j][update]
                best_label[update] = int(lbl)
            del batch_prob_f32

        # GPU → CPU → 保存
        result = best_label.cpu().numpy()
        del best_prob, best_label

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    max_label = int(result.max())
    out_dtype = np.uint8 if max_label <= 255 else np.uint16
    nib.save(
        nib.Nifti1Image(result.astype(out_dtype), ref_affine),
        str(output_path),
    )


# ---------------------------------------------------------------------------
#  多标签mask上采样（核心函数，CPU光滑版）
# ---------------------------------------------------------------------------

def resample_multilabel_mask_to_original(
    lowres_mask_path,
    ref_path,
    output_path,
    num_workers=None,
    verbose=True,
):
    """将低分辨率多标签mask插值回原始分辨率（并行加速，光滑无锯齿）。

    算法：One-Hot + 线性插值 + 增量ArgMax
    1. 背景通道和每个标签通道分别做线性插值
    2. 使用增量 ArgMax 策略依次比较，概率更大者胜出
    3. 最终每个体素只属于一个标签 → 无缝隙、无重叠

    Parameters
    ----------
    lowres_mask_path : str or Path
        低分辨率多标签mask路径（.nii.gz），标签值为整数 0,1,2,...
    ref_path : str or Path
        原始分辨率参考图像路径，用于获取目标 shape 和 affine
    output_path : str or Path
        输出mask路径
    num_workers : int, optional
        并行线程数。默认自动选择 min(8, cpu_count, label_count)。
        注意：每个线程的峰值内存 ≈ 一个 float32 目标体积，
        例如 512×512×300 × 4B ≈ 300MB，请根据内存酌情设置。
    verbose : bool
        是否打印进度信息
    """
    lowres_mask = nib.load(str(lowres_mask_path))
    ref_img = nib.load(str(ref_path))
    ref_affine = _get_valid_affine(ref_img)

    mask_data = np.asarray(lowres_mask.dataobj)
    # 转为整型以便做 == 比较
    if mask_data.dtype.kind == 'f':
        mask_data = np.round(mask_data).astype(np.int16)
    else:
        mask_data = mask_data.astype(np.int16)

    target_shape = ref_img.shape[:3]

    # shape 已一致则直接复制
    if mask_data.shape[:3] == target_shape:
        if verbose:
            print("Shape already matches, copying directly.")
        shutil.copy(str(lowres_mask_path), str(output_path))
        return

    zoom_factors = (
        np.array(target_shape, dtype=np.float64)
        / np.array(mask_data.shape[:3], dtype=np.float64)
    )

    labels = np.unique(mask_data)
    labels = labels[labels > 0]  # 排除背景 (0)

    if len(labels) == 0:
        # 全是背景
        result = np.zeros(target_shape, dtype=np.uint8)
        nib.save(nib.Nifti1Image(result, ref_affine), str(output_path))
        return

    if verbose:
        print(
            f"Resampling {len(labels)} labels from "
            f"{mask_data.shape[:3]} → {target_shape} ..."
        )

    # ---- 增量 ArgMax：用背景概率初始化 ----
    bg_prob = zoom(
        (mask_data == 0).astype(np.float32),
        zoom_factors,
        order=1,
        prefilter=False,
    )
    best_prob = bg_prob          # shape = target_shape, float32
    best_label = np.zeros(target_shape, dtype=np.int16)  # 0 = background
    del bg_prob

    # ---- 确定线程数 ----
    if num_workers is None:
        cpu_count = os.cpu_count() or 1
        num_workers = min(8, max(1, cpu_count), len(labels))

    # ---- 单标签插值函数（在工作线程中执行） ----
    def _interp_one_label(label_val):
        binary = (mask_data == label_val).astype(np.float32)
        prob = zoom(binary, zoom_factors, order=1, prefilter=False)
        return label_val, prob

    # ---- 并行插值，增量合并 ----
    finished = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_interp_one_label, int(lbl)): int(lbl)
            for lbl in labels
        }
        for future in as_completed(futures):
            label_val, prob = future.result()

            # 增量 ArgMax：概率更大则更新
            update_mask = prob > best_prob
            best_prob[update_mask] = prob[update_mask]
            best_label[update_mask] = label_val
            del prob, update_mask  # 及时释放内存

            finished += 1
            if verbose:
                print(f"  [{finished}/{len(labels)}] label {label_val} done.")

    # ---- 保存结果 ----
    max_label = int(best_label.max())
    out_dtype = np.uint8 if max_label <= 255 else np.uint16
    nib.save(
        nib.Nifti1Image(best_label.astype(out_dtype), ref_affine),
        str(output_path),
    )
    if verbose:
        print(f"Saved → {output_path}")


# ---------------------------------------------------------------------------
#  批量处理
# ---------------------------------------------------------------------------

def batch_resample_to_original(
    pred_image_dir,
    pred_mask_dir,
    ref_image_dir,
    output_image_dir,
    output_mask_dir,
    num_workers=None,
    skip_image=False,
    verbose=True,
):
    """批量将低分辨率预测（图像+mask）插值回原始分辨率。

    目录约定
    --------
    pred_image_dir  : 低分辨率预测图像目录（可选，skip_image=True 时跳过）
    pred_mask_dir   : 低分辨率预测mask目录
    ref_image_dir   : 原始分辨率参考图像目录（文件名需与预测对应）
    output_image_dir: 上采样图像输出目录
    output_mask_dir : 上采样mask输出目录

    文件名匹配规则
    --------------
    预测mask文件名:  <subject>.nii.gz
    参考图像文件名:  <subject>_0000.nii.gz   （nnUNet 通道后缀）

    Parameters
    ----------
    num_workers : int, optional
        每个mask内部并行插值的线程数
    skip_image : bool
        若为 True 则只处理mask，跳过图像
    verbose : bool
        打印进度
    """
    pred_mask_dir = Path(pred_mask_dir)
    ref_image_dir = Path(ref_image_dir)
    output_mask_dir = Path(output_mask_dir)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    if not skip_image:
        pred_image_dir = Path(pred_image_dir)
        output_image_dir = Path(output_image_dir)
        output_image_dir.mkdir(parents=True, exist_ok=True)

    mask_files = sorted(pred_mask_dir.glob("*.nii.gz"))
    if len(mask_files) == 0:
        print(f"No .nii.gz files found in {pred_mask_dir}")
        return

    for mask_file in tqdm(mask_files, desc="Resampling to original"):
        subject_name = mask_file.name  # e.g. "subj001.nii.gz"
        # nnUNet 的图像文件名比mask多一个 _0000 后缀
        subject_stem = mask_file.name.replace(".nii.gz", "")
        ref_image_name = f"{subject_stem}_0000.nii.gz"
        ref_path = ref_image_dir / ref_image_name

        if not ref_path.exists():
            # 也尝试同名匹配
            ref_path = ref_image_dir / subject_name
        if not ref_path.exists():
            print(f"  [SKIP] Reference not found for {subject_name}")
            continue

        # 上采样 mask
        if verbose:
            print(f"\n--- {subject_name} ---")
        resample_multilabel_mask_to_original(
            lowres_mask_path=mask_file,
            ref_path=ref_path,
            output_path=output_mask_dir / subject_name,
            num_workers=num_workers,
            verbose=verbose,
        )

        # 上采样图像（可选）
        if not skip_image:
            pred_image_name = f"{subject_stem}_0000.nii.gz"
            pred_image_path = pred_image_dir / pred_image_name
            if not pred_image_path.exists():
                pred_image_path = pred_image_dir / subject_name
            if pred_image_path.exists():
                resample_image_to_original(
                    lowres_path=pred_image_path,
                    ref_path=ref_path,
                    output_path=output_image_dir / pred_image_name,
                )
            else:
                if verbose:
                    print(f"  [SKIP] Predicted image not found: {pred_image_path}")


# ---------------------------------------------------------------------------
#  命令行入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    
    # =========================================================================
    # 选择运行模式："mask" | "image" | "batch"
    # =========================================================================
    MODE = "batch"

    # -------------------------------------------------------------------------
    # MODE = "mask"：上采样单个多标签 mask
    # -------------------------------------------------------------------------
    MASK_LOWRES  = r"D:\data\pred\subj001.nii.gz"   # 低分辨率 mask 路径
    MASK_REF     = r"D:\data\raw\subj001_0000.nii.gz"  # 原始分辨率参考图像
    MASK_OUTPUT  = r"D:\data\output\subj001.nii.gz"    # 输出路径
    MASK_WORKERS = None                                 # 并行线程数，None=自动

    # -------------------------------------------------------------------------
    # MODE = "image"：上采样单个图像
    # -------------------------------------------------------------------------
    IMG_LOWRES   = r"D:\data\pred\subj001_0000.nii.gz"  # 低分辨率图像路径
    IMG_REF      = r"D:\data\raw\subj001_0000.nii.gz"   # 原始分辨率参考图像
    IMG_OUTPUT   = r"D:\data\output\subj001_0000.nii.gz" # 输出路径

    # -------------------------------------------------------------------------
    # MODE = "batch"：批量上采样
    # -------------------------------------------------------------------------
    BATCH_PRED_MASK_DIR   = r"/data1/segmentationForTrain/traindata/CTWholeBodyBone/nnUNet_raw/Dataset102_Rapid_Bone_3/labelsTs_predicted"        # 预测 mask 目录
    BATCH_REF_IMAGE_DIR   = r"/data1/segmentationForTrain/traindata/CTWholeBodyBone/nnUNet_raw/Dataset101_Rapid_Bone/imagesTs"        # 参考图像目录
    BATCH_OUTPUT_MASK_DIR = r"/data1/segmentationForTrain/traindata/CTWholeBodyBone/nnUNet_raw/Dataset101_Rapid_Bone/labelsTs_predictedbylowmodel"      # 输出 mask 目录
    BATCH_PRED_IMAGE_DIR  = None   # 预测图像目录，不需要时设为 None
    BATCH_OUTPUT_IMAGE_DIR = None  # 输出图像目录，不需要时设为 None
    BATCH_WORKERS         = None   # 并行线程数，None=自动
    # =========================================================================

    if MODE == "mask":
        resample_multilabel_mask_to_original(
            MASK_LOWRES, MASK_REF, MASK_OUTPUT,
            num_workers=MASK_WORKERS,
        )
    elif MODE == "image":
        resample_image_to_original(IMG_LOWRES, IMG_REF, IMG_OUTPUT)
    elif MODE == "batch":
        batch_resample_to_original(
            pred_image_dir=BATCH_PRED_IMAGE_DIR,
            pred_mask_dir=BATCH_PRED_MASK_DIR,
            ref_image_dir=BATCH_REF_IMAGE_DIR,
            output_image_dir=BATCH_OUTPUT_IMAGE_DIR,
            output_mask_dir=BATCH_OUTPUT_MASK_DIR,
            num_workers=BATCH_WORKERS,
            skip_image=(BATCH_PRED_IMAGE_DIR is None),
        )
    else:
        raise ValueError(f"未知 MODE: {MODE!r}，请设置为 'mask'、'image' 或 'batch'")