import sys
import os
from pathlib import Path
import shutil
import json

import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor
import random
from scipy.ndimage import zoom

import ResampleImageAndMask
from nibabel.orientations import axcodes2ornt, ornt_transform, apply_orientation, inv_ornt_aff

# MHD/ITK-SNAP 约定：每个字母表示该轴 **低索引端（起始端）** 的方向。
# nibabel 约定：每个字母表示该轴 **正方向（递增方向）**。
# 因此 MHD 的 RAI 等价于 nibabel 的 LPS（每个字母取反）。
_OPPOSITE_AXIS = {
    "R": "L", "L": "R",
    "A": "P", "P": "A",
    "S": "I", "I": "S",
}


def _mhd_to_nibabel_orientation(orientation: str) -> str:
    """将 MHD/ITK-SNAP 约定的方位码转换为 nibabel 约定（每个字母取反）。"""
    return "".join(_OPPOSITE_AXIS[ch.upper()] for ch in orientation)


def _orthonormalize_affine(affine):
    """确保 4×4 affine 的旋转部分（方向余弦）严格正交归一化。

    从 3×3 子矩阵中提取 spacing（列范数），归一化得到方向余弦矩阵，
    通过 SVD 寻找最近正交矩阵，再乘回 spacing 重建旋转/缩放部分。
    保留原始平移（origin）和底行不变。

    用途：避免 nibabel 的 affine → quaternion → affine 往返转换引入微小误差，
    导致 SimpleITK / ITK-SNAP 因方向余弦不正交而拒绝读取文件。
    """
    affine = np.array(affine, dtype=np.float64)
    R = affine[:3, :3]
    # 退化 affine 不做处理
    if np.any(np.isnan(R)) or np.all(np.abs(R) < 1e-12):
        return affine

    spacing = np.linalg.norm(R, axis=0)
    spacing = np.where(spacing < 1e-12, 1.0, spacing)
    D = R / spacing

    U, _, Vt = np.linalg.svd(D)
    if np.linalg.det(U @ Vt) < 0:
        U[:, -1] *= -1

    result = affine.copy()
    result[:3, :3] = (U @ Vt) * spacing
    return result


def reorient_nifti(img, target_orientation):
    """
    将 nibabel 图像重定向到目标方位，仅通过轴置换和翻转实现，不做任何插值，
    因此原始体素值完全不变。

    Parameters
    ----------
    img : nibabel.Nifti1Image
        输入图像。
    target_orientation : str
        三位方位码，如 "RAS", "LPI", "RAI" 等，与 ITK-SNAP / MHD 约定一致。
        每个字母表示对应轴 **低索引端（起始端）** 的方向：
        R/L = Right/Left, A/P = Anterior/Posterior, I/S = Inferior/Superior.
        例如 RAI 表示 x 轴从 R→L，y 轴从 A→P，z 轴从 I→S。

    Returns
    -------
    nibabel.Nifti1Image
        重定向后的图像。若已是目标方位则直接返回原对象。
    """
    current_ornt = nib.io_orientation(img.affine)
    # 将 MHD 约定转换为 nibabel 约定（取反），再生成 ornt 数组
    nib_orientation = _mhd_to_nibabel_orientation(target_orientation)
    target_ornt = axcodes2ornt(tuple(nib_orientation))
    transform = ornt_transform(current_ornt, target_ornt)

    # 如果已经是目标方位，直接返回
    identity = np.column_stack([np.arange(len(transform)), np.ones(len(transform))])
    if np.array_equal(transform, identity):
        return img

    data = np.asarray(img.dataobj)
    reoriented_data = apply_orientation(data, transform)
    new_affine = _orthonormalize_affine(img.affine @ inv_ornt_aff(transform, img.shape))
    # 不传旧 header，避免 qform/sform 中残留的旧参数与新 affine 冲突
    new_img = nib.Nifti1Image(reoriented_data, new_affine)
    new_img.header.set_data_dtype(img.header.get_data_dtype())
    return new_img


def _is_grouped_class_map(class_map):
    """判断 class_map 是否为分组格式（TOML 子表，每个 value 是含 label+organs 的 dict）。

    分组格式示例（来自 TOML 解析结果）::

        {"head":      {"label": 1, "organs": ["brain", "skull"]},
         "chest":     {"label": 2, "organs": ["heart", "aorta", ...]},
         ...}

    扁平格式示例::

        {"brain": 1, "skull": 2, ...}
    """
    if not isinstance(class_map, dict):
        return False
    for v in class_map.values():
        return isinstance(v, dict) and "label" in v and "organs" in v
    return False


def _expand_grouped_class_map(class_map):
    """将分组格式的 class_map 展开为扁平字典和分组标签字典。

    Returns
    -------
    flat_map : dict[str, int]
        {organ_name: label_value}，用于 mask 文件查找和标签赋值。
    group_labels : dict[str, int]
        {group_name: label_value}，用于 dataset.json 的 labels 字段。
    """
    flat_map = {}
    group_labels = {}
    for group_name, group_info in class_map.items():
        label = int(group_info["label"])
        group_labels[group_name] = label
        for organ in group_info["organs"]:
            flat_map[organ] = label
    return flat_map, group_labels


def generate_json_from_dir_v2(train_dataset_name, subjects_train, subjects_val, labels,
                              modality, image_reader_writer):
    """生成 dataset.json 和 splits_final.json。

    Parameters
    ----------
    modality : str
        图像模态，如 "CT" 或 "MR"，写入 channel_names。
    image_reader_writer : str
        nnUNet 的 image reader/writer 类名，写入 overwrite_image_reader_writer。
    """
    print("Creating dataset.json...")

    out_base = Path(os.environ["nnUNet_raw"]) / train_dataset_name

    json_dict = {}
    json_dict['name'] = "TotalSegmentator"
    json_dict['description'] = "Segmentation of TotalSegmentator classes"
    json_dict['reference'] = "https://zenodo.org/record/6802614"
    json_dict['licence'] = "Apache 2.0"
    json_dict['release'] = "2.0"
    json_dict['channel_names'] = {"0": modality}
    if isinstance(labels, dict):
        # 当多个器官共享同一 label 值时（如粗分割），合并为一个 channel，
        # 仅保留每个 label 值对应的第一个器官名称，避免 nnUNet planner
        # 误以为有 100+ 个输出通道导致训练变慢。
        seen_values = {}
        for name, value in labels.items():
            int_val = int(value)
            if int_val not in seen_values:
                seen_values[int_val] = name
        json_dict['labels'] = {"background": 0, **{name: val for val, name in seen_values.items()}}
    else:
        json_dict['labels'] = {val:idx for idx,val in enumerate(["background",] + list(labels))}
    json_dict['numTraining'] = len(subjects_train + subjects_val)
    json_dict['file_ending'] = '.nii.gz'
    json_dict['overwrite_image_reader_writer'] = image_reader_writer

    json.dump(json_dict, open(out_base / "dataset.json", "w"), sort_keys=False, indent=4)

    print("Creating split_final.json...")
    output_folder_pkl = Path(os.environ['nnUNet_preprocessed']) / train_dataset_name
    output_folder_pkl.mkdir(exist_ok=True)


    #此处subjects_train是带路径的，需要改成只有文件名
    train_filelist = [f"{Path(path).parent.name}_{Path(path).name}" for path in subjects_train]
    val_filelist = [f"{Path(path).parent.name}_{Path(path).name}" for path in subjects_val]

    splits = []
    splits.append({
        "train": train_filelist,
        "val": val_filelist
    })

    print(f"nr of folds: {len(splits)}")
    print(f"nr train subjects (fold 0): {len(splits[0]['train'])}")
    print(f"nr val subjects (fold 0): {len(splits[0]['val'])}")

    json.dump(splits, open(output_folder_pkl / "splits_final.json", "w"), sort_keys=False, indent=4)


def generate_train_test_dataset(dataset_path):

    '''
    本函数来自totalsegmentator, 优先读取 meta.csv 进行数据集划分。
    若 meta.csv 不存在或读取失败，则自动扫描 dataset 路径下的所有子文件夹，
    按 80% / 10% / 10% 随机划分训练集、验证集和测试集。
    '''

    subjects_train = []
    subjects_val = []
    subjects_test = []
    for dataset in dataset_path:
        datapath = str(dataset) + "/"
        try:
            meta = pd.read_csv(Path(dataset) / "meta.csv", sep=";")
            # 训练集
            trainlist = list(meta[meta["split"] == "train"]["image_id"].values)
            subjects_train += [datapath + f for f in trainlist]
            # 验证集
            vallist = list(meta[meta["split"] == "val"]["image_id"].values)
            subjects_val += [datapath + f for f in vallist]
            # 测试集
            testlist = list(meta[meta["split"] == "test"]["image_id"].values)
            subjects_test += [datapath + f for f in testlist]
            #如果没划分验证集，则从训练集中取后10%作为验证集
            if len(subjects_val) == 0:
                n_val_from_train = max(1, int(len(subjects_train) * 0.1))
                val_from_train = subjects_train[-n_val_from_train:]
                subjects_train = subjects_train[:-n_val_from_train]
                subjects_val = val_from_train
            print(f"[meta.csv] {Path(dataset).name}: train={len(subjects_train)}, val={len(subjects_val)}, test={len(subjects_test)}")
        except Exception as e:
            print(f"[警告] 读取 {dataset}/meta.csv 失败（{e}），改为自动扫描子文件夹并随机划分。")
            all_subjects = sorted([
                datapath + d.name
                for d in Path(dataset).iterdir()
                if d.is_dir()
            ])
            if len(all_subjects) == 0:
                print(f"[警告] {dataset} 下未找到任何子文件夹，跳过该数据集。")
                continue
            random.shuffle(all_subjects)
            n = len(all_subjects)
            n_train = max(1, int(n * 0.8))
            n_val   = max(1, int(n * 0.1))
            # 测试集取剩余，保证三部分不重叠且覆盖所有数据
            trainlist = all_subjects[:n_train]
            vallist   = all_subjects[n_train:n_train + n_val]
            testlist  = all_subjects[n_train + n_val:]
            subjects_train += trainlist
            subjects_val   += vallist
            subjects_test  += testlist
            print(f"[自动划分] {Path(dataset).name}: train={len(trainlist)}, val={len(vallist)}, test={len(testlist)}")

    return subjects_train, subjects_val, subjects_test



def resample_and_combine_labels(ref_img, file_out, masks, target_spacing=None, label_values=None, target_orientation=None):
    """label_values: 与 masks 一一对应的整数列表；为 None 时退化为从 1 开始顺序编号。
    target_orientation: 三位方位码（如 'RAS'），为 None 时不做重定向。"""
    ref_img = nib.load(ref_img)
    if target_spacing is not None:
        ref_img = ResampleImageAndMask.resample_nifti(ref_img, target_spacing, order=1)
    if target_orientation is not None:
        ref_img = reorient_nifti(ref_img, target_orientation)
    combined = np.zeros(ref_img.shape).astype(np.uint8)

    for idx, arg in enumerate(masks):
        file_in = Path(arg)
        if file_in.exists():
            img = nib.load(file_in)
            if target_spacing is not None:
                img = ResampleImageAndMask.resample_nifti(img, target_spacing, order=0)
            if target_orientation is not None:
                img = reorient_nifti(img, target_orientation)
            lv = label_values[idx] if label_values is not None else idx + 1
            combined[img.get_fdata() > 0] = lv
        else:
            print(f"Missing: {file_in}")
    nib.save(nib.Nifti1Image(combined.astype(np.uint8), ref_img.affine), file_out)




def resample_and_combine_labels_fast(
    ref_img,
    file_out,
    masks,
    target_spacing=None,
    label_values=None,
    target_orientation=None,
    num_workers=None,
    verbose=True,
):
    """Faster version of `resample_and_combine_labels` using parallel mask loading/resampling.

    Notes:
    - Each single-label mask is loaded and, if needed, resampled in parallel.
    - Final label writing still follows the original mask order, so later masks keep
      the same overwrite priority as the original implementation.
    - `num_workers` defaults to a conservative thread count suitable for mixed IO/CPU work.
    """
    ref_img = nib.load(ref_img)
    if target_spacing is not None:
        ref_img = ResampleImageAndMask._resample_nifti_fast(ref_img, target_spacing, order=1, dtype=np.float32)
    if target_orientation is not None:
        ref_img = reorient_nifti(ref_img, target_orientation)

    combined = np.zeros(ref_img.shape, dtype=np.uint8)
    indexed_masks = [(idx, Path(mask_path)) for idx, mask_path in enumerate(masks)]

    if num_workers is None:
        cpu_count = os.cpu_count() or 1
        num_workers = min(16, max(1, cpu_count))

    def _load_and_resample_single_mask(index_and_path):
        idx, file_in = index_and_path
        if not file_in.exists():
            return idx, None, file_in

        img = nib.load(file_in)
        if target_spacing is not None:
            img = ResampleImageAndMask._resample_nifti_fast(img, target_spacing, order=0, dtype=np.uint8)
        if target_orientation is not None:
            img = reorient_nifti(img, target_orientation)
        mask_data = np.asarray(img.dataobj) > 0

        return idx, mask_data, file_in

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(_load_and_resample_single_mask, indexed_masks))

    for idx, mask_data, file_in in sorted(results, key=lambda item: item[0]):
        if mask_data is None:
            if verbose:
                print(f"Missing: {file_in}")
            continue
        lv = label_values[idx] if label_values is not None else idx + 1
        combined[mask_data] = lv

    nib.save(nib.Nifti1Image(combined, ref_img.affine), file_out)



def combine_labels(ref_img, file_out, masks, label_values=None, target_orientation=None):
    ref_img = nib.load(ref_img)
    if target_orientation is not None:
        ref_img = reorient_nifti(ref_img, target_orientation)
    combined = np.zeros(ref_img.shape).astype(np.uint8)
    for idx, arg in enumerate(masks):
        file_in = Path(arg)
        if file_in.exists():
            img = nib.load(file_in)
            if target_orientation is not None:
                img = reorient_nifti(img, target_orientation)
            lv = label_values[idx] if label_values is not None else idx + 1
            combined[img.get_fdata() > 0] = lv
        else:
            print(f"Missing: {file_in}")
    nib.save(nib.Nifti1Image(combined.astype(np.uint8), ref_img.affine), file_out)




def convert(dataset_path, nnunet_path, class_map_name, target_spacing=None, target_orientation=None,
            modality="CT", image_reader_writer="NibabelIOWithReorient"):
    """
    Convert the downloaded TotalSegmentator dataset (after unzipping it) to nnUNet format and
    generate dataset.json and splits_final.json

    example usage:
    python convert_dataset_to_nnunet.py /my_downloads/TotalSegmentator_dataset /nnunet/raw/Dataset100_TotalSegmentator_part1 class_map_part_organs

    You must set nnUNet_raw and nnUNet_preprocessed environment variables before running this (see nnUNet documentation).

    Parameters
    ----------
    target_orientation : str or None
        三位方位码（如 'RAS', 'LPI', 'RAI'），用于将图像和 mask 重定向到目标方位。
        仅通过轴置换和翻转实现，不做插值，不改变原始体素值。
        为 None 或空字符串时不做重定向。
    modality : str
        图像模态，如 "CT" 或 "MR"，传递给 generate_json_from_dir_v2。
    image_reader_writer : str
        nnUNet 的 image reader/writer 类名，传递给 generate_json_from_dir_v2。
    """

    class_map = class_map_name

    # 如果 class_map 是分组格式（粗分割），展开为扁平字典用于 mask 处理，
    # 同时提取分组标签字典用于 dataset.json
    if _is_grouped_class_map(class_map):
        class_map, json_labels = _expand_grouped_class_map(class_map)
    else:
        json_labels = class_map  # 扁平格式直接用，generate_json_from_dir_v2 内部会去重

    # 空字符串视为 None
    if target_orientation is not None and len(target_orientation.strip()) == 0:
        target_orientation = None

    (nnunet_path / "imagesTr").mkdir(parents=True, exist_ok=True)
    (nnunet_path / "labelsTr").mkdir(parents=True, exist_ok=True)
    (nnunet_path / "imagesTs").mkdir(parents=True, exist_ok=True)
    (nnunet_path / "labelsTs").mkdir(parents=True, exist_ok=True)

    # if target_spacing is not None:
    #     (nnunet_path / "origin_images").mkdir(parents=True, exist_ok=True)
    #     (nnunet_path / "origin_labels").mkdir(parents=True, exist_ok=True)

    #划分数据集
    subjects_train, subjects_val, subjects_test = generate_train_test_dataset(dataset_path)

    #抽样数据集，主要为调试代码用，避免太多数据不好调试
    # subjects_train = subjects_train[0:5]
    # subjects_val = subjects_val[0:2]
    # subjects_test = subjects_test[0:3]



    #生成训练用数据集
    print("Copying train data...")
    for subject in tqdm(subjects_train + subjects_val):
        subject_path = Path(subject)

        #把原来的指定文件名，改成不指定具体文件名
        file_names =  (lambda folder: [f.name for f in subject_path.iterdir() if f.is_file()])(subject_path)
        if len(file_names) == 0:
            raise ValueError(f"{subject_path}下没有数据")
        origin_file = subject_path / file_names[0]

        #由于数据来自多个数据集，目标文件名需要增加数据集名，以避免重名
        dst_file = nnunet_path / "imagesTr" / f"{Path(subject).parent.name}_{Path(subject).name}_0000.nii.gz" #0000为nnUnet的通道数

        #mask的路径
        dstmask_file = nnunet_path / "labelsTr" / f"{Path(subject).parent.name}_{Path(subject).name}.nii.gz"
        mask_paths = [subject_path / "segmentations" / f"{roi}.nii.gz" for roi in class_map]
        lv_list = list(class_map.values()) if isinstance(class_map, dict) else None
        no_resample = (target_spacing is None or len(target_spacing) == 0)
        if no_resample and target_orientation is None:
            shutil.copy(origin_file, dst_file)
            combine_labels(origin_file, dstmask_file, mask_paths, label_values=lv_list)
        elif no_resample and target_orientation is not None:
            # 仅重定向，不重采样
            img = nib.load(origin_file)
            img = reorient_nifti(img, target_orientation)
            nib.save(img, dst_file)
            combine_labels(origin_file, dstmask_file, mask_paths,
                           label_values=lv_list, target_orientation=target_orientation)
        else:
            # 重采样 + 重定向，在内存中一次完成，避免多余的磁盘读写
            img = nib.load(origin_file)
            img = ResampleImageAndMask.resample_nifti(img, target_spacing, order=1)
            if target_orientation is not None:
                img = reorient_nifti(img, target_orientation)
            nib.save(img, dst_file)
            resample_and_combine_labels(origin_file, dstmask_file, mask_paths,
                       target_spacing, label_values=lv_list,
                       target_orientation=target_orientation)


    print("Copying test data...")
    for subject in tqdm(subjects_test):
        subject_path = Path(subject)
        file_names =  (lambda folder: [f.name for f in subject_path.iterdir() if f.is_file()])(subject_path)
        if len(file_names) == 0:
            raise ValueError(f"{subject_path}下没有数据")
        origin_file = subject_path / file_names[0]
        dst_file = nnunet_path / "imagesTs" / f"{Path(subject).parent.name}_{Path(subject).name}_0000.nii.gz" #0000为nnUnet的通道数
        dstmask_file = nnunet_path / "labelsTs" / f"{Path(subject).parent.name}_{Path(subject).name}.nii.gz"

        mask_paths = [subject_path / "segmentations" / f"{roi}.nii.gz" for roi in class_map]
        lv_list = list(class_map.values()) if isinstance(class_map, dict) else None
        no_resample = (target_spacing is None or len(target_spacing) == 0)
        if no_resample and target_orientation is None:
            shutil.copy(origin_file, dst_file)
            combine_labels(origin_file, dstmask_file, mask_paths, label_values=lv_list)
        elif no_resample and target_orientation is not None:
            img = nib.load(origin_file)
            img = reorient_nifti(img, target_orientation)
            nib.save(img, dst_file)
            combine_labels(origin_file, dstmask_file, mask_paths,
                           label_values=lv_list, target_orientation=target_orientation)
        else:
            img = nib.load(origin_file)
            img = ResampleImageAndMask.resample_nifti(img, target_spacing, order=1)
            if target_orientation is not None:
                img = reorient_nifti(img, target_orientation)
            nib.save(img, dst_file)
            resample_and_combine_labels(origin_file, dstmask_file, mask_paths,
                       target_spacing, label_values=lv_list,
                       target_orientation=target_orientation)
    

    generate_json_from_dir_v2(nnunet_path.name, subjects_train, subjects_val, json_labels,
                              modality=modality, image_reader_writer=image_reader_writer)




def convert_multilabel_to_one(inputmask_paths, outputmask_path=None, class_map=None, combine_map=None, num_workers=None):
    """
    将多个模型各自预测的 mask 拼接成一个全身 mask。

    参数
    ----
    inputmask_paths : list[str | Path]
        可传两种形式：
        1. 多个目录：每个目录对应一个模型，按文件名对齐后逐例合并。
        2. 多个文件：每个文件对应一个模型，仅合并这组给定文件。
    class_map : list[dict] | None
        每个模型的标签映射字典列表（从 ModelMap.toml 读取），与 inputmask_paths 一一对应。
        每个 dict 的 key 为组织名，value 为该模型 mask 中该组织的 label 整数值。
        例如 MR2_Chest = {heart: 1, aorta: 2, lung_left: 3, lung_right: 3}。
        当同一 dict 中多个 key 对应相同 value 时，取第一个 key 作为组织名。
    combine_map : dict | None
        拼接目标标签映射（如 ModelMap.toml 中的 MR_Combine）。
        key 为组织名，value 为拼接后 mask 中该组织的最终 label 值。
        当 class_map 和 combine_map 同时提供时，映射流程为：
          输入 mask label 值 → 通过 class_map 查到组织名 → 通过 combine_map 查到最终 label 值。
        当 combine_map 为 None 时，退化为旧逻辑（直接使用 class_map 中的 value 或自动推断）。
    outputmask_path : str | Path
        目录模式下为输出目录；文件模式下可为输出文件路径，也可为输出目录。
    num_workers : int | None
        并行处理的线程数，None 时自动设置为 min(8, cpu_count)。
    """
    import time

    if outputmask_path is None:
        raise ValueError("outputmask_path 不能为空。")

    inputmask_paths = [Path(p) for p in inputmask_paths]

    if len(inputmask_paths) == 0:
        raise ValueError("inputmask_paths 不能为空。")

    if class_map is not None and len(class_map) != len(inputmask_paths):
        raise ValueError("class_map 的长度必须与 inputmask_paths 一致。")

    def _is_nifti_file(file_path: Path) -> bool:
        return file_path.is_file() and (file_path.name.endswith(".nii.gz") or file_path.suffix.lower() == ".nii")

    has_dir = any(path.is_dir() for path in inputmask_paths)
    has_file = any(path.is_file() for path in inputmask_paths)
    if has_dir and has_file:
        raise ValueError("inputmask_paths 不能同时混用文件和文件夹路径。")

    input_mode = "directory" if has_dir else "file"

    if input_mode == "directory":
        outputmask_path = Path(outputmask_path)
        outputmask_path.mkdir(parents=True, exist_ok=True)

        case_specs: list[dict] = []
        all_files: set[str] = set()
        for mask_dir in inputmask_paths:
            if not mask_dir.exists():
                continue
            if not mask_dir.is_dir():
                raise ValueError(f"输入不是目录: {mask_dir}")
            for f in mask_dir.iterdir():
                if _is_nifti_file(f):
                    all_files.add(f.name)

        if not all_files:
            print("未找到任何 .nii / .nii.gz 文件，请检查 inputmask_paths。")
            return

        for filename in sorted(all_files):
            case_specs.append({
                "name": filename,
                "inputs": [mask_dir / filename for mask_dir in inputmask_paths],
                "output": outputmask_path / filename,
            })
    else:
        invalid_files = [path for path in inputmask_paths if not _is_nifti_file(path)]
        if invalid_files:
            raise ValueError(f"以下输入不是有效的 .nii/.nii.gz 文件: {invalid_files}")

        outputmask_path = Path(outputmask_path)
        if outputmask_path.suffix.lower() == ".nii" or outputmask_path.name.endswith(".nii.gz"):
            output_file = outputmask_path
            output_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            outputmask_path.mkdir(parents=True, exist_ok=True)
            output_file = outputmask_path / inputmask_paths[0].name

        case_specs = [{
            "name": output_file.name,
            "inputs": inputmask_paths,
            "output": output_file,
        }]

    print(f"共找到 {len(case_specs)} 个待合并项，开始合并…")

    def _infer_labels_from_paths(mask_paths: list[Path]) -> list[int]:
        positive_labels: set[int] = set()
        for fpath in mask_paths:
            if not fpath.exists():
                continue
            data = np.asarray(nib.load(fpath).dataobj)
            labels = np.unique(data)
            positive_labels.update(int(v) for v in labels if int(v) > 0)
        return sorted(positive_labels)

    # 提前计算输出 dtype 和每个模型的 LUT（查找表）
    # LUT: lut[local_label] = target_value，local_label 从 0（背景）开始
    #
    # 当 class_map + combine_map 同时提供时，映射流程：
    #   输入 mask 的 label 值 → class_map[i] 反查组织名 → combine_map 查最终 label 值
    # 当仅 class_map 提供（combine_map=None）时，直接用 class_map 中的 value 作为最终 label
    # 当 class_map 也为 None 时，自动扫描数据并按输入顺序连续编号

    provided_class_map = class_map
    provided_combine_map = combine_map

    # ---------- 构建每个输入对应的 local_label → target_value 映射 ----------
    all_local_to_target: list[dict[int, int]] = []  # 每个输入一个 dict
    next_target_value = 1  # 仅在自动推断模式下使用

    for index, input_path in enumerate(inputmask_paths):
        part_dict = provided_class_map[index] if provided_class_map is not None else None

        if part_dict is not None and provided_combine_map is not None:
            # ---- 新流程：class_map + combine_map 联合映射 ----
            # 1) 反转 part_dict：value → 第一个 key（组织名）
            value_to_name: dict[int, str] = {}
            for tissue_name, local_val in part_dict.items():
                local_val = int(local_val)
                if local_val not in value_to_name:
                    value_to_name[local_val] = tissue_name

            # 2) 通过组织名在 combine_map 中查最终 label
            local_to_target: dict[int, int] = {}
            for local_val, tissue_name in value_to_name.items():
                if tissue_name in provided_combine_map:
                    local_to_target[local_val] = int(provided_combine_map[tissue_name])
                else:
                    print(f"[警告] 组织 '{tissue_name}'（来自输入 {index}，local_label={local_val}）"
                          f"在 combine_map 中未找到，已跳过。")
            all_local_to_target.append(local_to_target)

        elif part_dict is not None:
            # ---- 旧流程：直接用 class_map 中的 value ----
            ordered_values = [int(v) for v in part_dict.values()]
            local_to_target = {i + 1: v for i, v in enumerate(ordered_values)}
            all_local_to_target.append(local_to_target)

        else:
            # ---- 自动推断模式 ----
            if input_mode == "directory":
                mask_candidates = [case_spec["inputs"][index] for case_spec in case_specs]
            else:
                mask_candidates = [input_path]

            local_labels = _infer_labels_from_paths(mask_candidates)
            local_to_target = {
                int(local_label): next_target_value + idx
                for idx, local_label in enumerate(local_labels)
            }
            all_local_to_target.append(local_to_target)
            next_target_value += len(local_labels)

    # ---------- 确定输出 dtype ----------
    max_target = 0
    for mapping in all_local_to_target:
        if mapping:
            max_target = max(max_target, max(mapping.values()))

    dtype = np.uint8 if max_target <= 255 else np.uint16

    # ---------- 构建 numpy LUT（加速体素级映射）----------
    luts = []
    for mapping in all_local_to_target:
        if not mapping:
            luts.append(np.zeros(1, dtype=dtype))
            continue
        lut_size = max(mapping.keys()) + 1
        lut = np.zeros(lut_size, dtype=dtype)
        for local_label, target_value in mapping.items():
            lut[local_label] = target_value
        luts.append(lut)

    def _process_one(case_spec: dict) -> tuple[str, float]:
        """处理单个合并项：加载各模型 mask → LUT 重映射 → 合并 → 保存。返回 (name, elapsed_s)。"""
        t0 = time.perf_counter()
        case_name = case_spec["name"]
        case_inputs = case_spec["inputs"]
        case_output = case_spec["output"]

        # 找参考 affine / shape
        ref_affine = None
        ref_shape = None
        for fpath in case_inputs:
            if fpath.exists():
                ref_nib = nib.load(fpath)
                ref_shape = ref_nib.shape
                ref_affine = ref_nib.affine
                break

        if ref_shape is None:
            return case_name, -1.0  # 标记为跳过

        combined = np.zeros(ref_shape, dtype=dtype)

        for fpath, lut in zip(case_inputs, luts):
            if not fpath.exists():
                continue

            img = nib.load(fpath)
            data = np.asarray(img.dataobj).astype(np.intp)
            remapped = np.zeros(data.shape, dtype=dtype)
            valid = (data >= 0) & (data < len(lut))
            remapped[valid] = lut[data[valid]]
            nonzero = remapped > 0
            combined[nonzero] = remapped[nonzero]

        nib.save(nib.Nifti1Image(combined, ref_affine), str(case_output))

        return case_name, time.perf_counter() - t0

    if num_workers is None:
        num_workers = min(8, max(1, (os.cpu_count() or 1)))

    sorted_cases = sorted(case_specs, key=lambda item: item["name"])
    total_t0 = time.perf_counter()
    per_case_records: list[tuple[str, float]] = []   # (filename, elapsed_s)

    if num_workers <= 1:
        # 单线程，保留进度条
        for case_spec in tqdm(sorted_cases, desc="合并 mask"):
            name, elapsed = _process_one(case_spec)
            if elapsed < 0:
                print(f"\n[跳过] {name}：所有输入项均无此文件")
            else:
                tqdm.write(f"  {name}  {elapsed:.1f}s")
                per_case_records.append((name, elapsed))
    else:
        # 多线程并行处理
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_process_one, case_spec): case_spec for case_spec in sorted_cases}
            for future in tqdm(
                futures,  # iterate in submission order for stable output
                total=len(futures),
                desc="合并 mask",
            ):
                name, elapsed = future.result()
                if elapsed < 0:
                    tqdm.write(f"  [跳过] {name}：所有输入项均无此文件")
                else:
                    tqdm.write(f"  {name}  {elapsed:.1f}s")
                    per_case_records.append((name, elapsed))

    wall_elapsed = time.perf_counter() - total_t0
    per_case_times = [e for _, e in per_case_records]
    sum_elapsed = sum(per_case_times)
    n = len(per_case_times)
    speedup = sum_elapsed / wall_elapsed if wall_elapsed > 0 else 1.0

    summary = (
        f"\n合并完成，共 {n} 例（{num_workers} 线程并行）\n"
        f"  各例耗时之和：{sum_elapsed:.1f}s，平均每例 {sum_elapsed / max(n, 1):.1f}s\n"
        f"  实际挂钟耗时：{wall_elapsed:.1f}s（并行加速比 {speedup:.1f}x）\n"
        f"  结果保存至：{outputmask_path}"
    )
    print(summary)

    # 将每例计时结果写入 CSV，末尾附汇总行
    timing_dir = outputmask_path if outputmask_path.is_dir() else outputmask_path.parent
    timing_file = timing_dir / "merge_timing.csv"
    with open(timing_file, "w", encoding="utf-8") as f:
        f.write("filename,elapsed_s\n")
        for fname, elapsed in sorted(per_case_records, key=lambda x: x[0]):
            f.write(f"{fname},{elapsed:.3f}\n")
        f.write("\n")
        f.write(f"# 例数,{n}\n")
        f.write(f"# 各例耗时之和(s),{sum_elapsed:.3f}\n")
        f.write(f"# 平均每例(s),{sum_elapsed / max(n, 1):.3f}\n")
        f.write(f"# 实际挂钟耗时(s),{wall_elapsed:.3f}\n")
        f.write(f"# 并行加速比,{speedup:.2f}x\n")
        f.write(f"# 并行线程数,{num_workers}\n")
    print(f"  计时结果已保存至：{timing_file}")



# if __name__ == "__main__":
#     convert_multilabel_to_one()