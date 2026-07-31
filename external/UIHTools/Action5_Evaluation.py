import sys
import os
from pathlib import Path
import shutil
from functools import partial
import json
from datetime import datetime

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm
from p_tqdm import p_map
# package from: https://github.com/deepmind/surface-distance
#from surface_distance import compute_surface_distances, compute_surface_dice_at_tolerance


DEFAULT_GT_SUBDIR = "labelsTs"
DEFAULT_PRED_SUBDIR = "labelsTs_predicted"
DEFAULT_EVAL_SUBDIR = "evaluation"


def _build_eval_class_map(raw_class_map: dict) -> dict:
    """
    将 TOML 中的 class_map（organ_name -> label_idx）转换为评估用格式。
    相同 label_idx 的器官合并为一个 ROI 名称（用 '+' 连接）。

    例如::

        输入: {"lung_upper_lobe_left": 2, "lung_lower_lobe_left": 2, "brain": 1}
        输出: OrderedDict({"brain": 1, "lung_upper_lobe_left+lung_lower_lobe_left": 2})

    返回按 label_idx 排序的 OrderedDict: {合并名称: label_idx}
    """
    from collections import OrderedDict

    label_to_names: dict[int, list[str]] = {}
    for name, idx in raw_class_map.items():
        label_to_names.setdefault(idx, []).append(name)

    result = OrderedDict()
    for idx in sorted(label_to_names.keys()):
        combined_name = "+".join(label_to_names[idx])
        result[combined_name] = idx
    return result


def dice_score(y_true, y_pred):
    """
    Binary Dice score. Same results as sklearn f1 binary.
    """
    intersect = np.sum(y_true * y_pred)
    denominator = np.sum(y_true) + np.sum(y_pred)
    f1 = (2 * intersect) / (denominator + 1e-6)
    return f1


def calc_metrics(subject, gt_dir=None, pred_dir=None, class_map=None):
    """
    计算单个病例的评估指标
    返回包含详细结果的结构化字典
    """
    gt_img = nib.load(gt_dir / f"{subject}.nii.gz")
    gt_all = gt_img.get_fdata()
    pred_all = nib.load(pred_dir / f"{subject}.nii.gz").get_fdata()

    # 从金标准图像 header 读取体素间距（单位 mm），用于 surface distance 计算
    # zooms 返回 (x, y, z) 三个方向的分辨率，与 compute_surface_distances 的 spacing_mm 参数对应
    zooms = gt_img.header.get_zooms()
    spacing_mm = [float(zooms[0]), float(zooms[1]), float(zooms[2])]

    # 基础信息
    r = {
        "subject": subject,
        "voxel_counts": {},
        "metrics": {}
    }

    # 记录总体体素信息
    r["voxel_counts"]["total_gt_voxels"] = int(np.sum(gt_all > 0))
    r["voxel_counts"]["total_pred_voxels"] = int(np.sum(pred_all > 0))

    for roi_name, label_idx in class_map.items():
        gt = gt_all == label_idx
        pred = pred_all == label_idx
        
        # 每类体素计数
        gt_voxels = int(np.sum(gt))
        pred_voxels = int(np.sum(pred))
        
        roi_metrics = {
            "gt_voxels": gt_voxels,
            "pred_voxels": pred_voxels,
            "dice": None,
            "surface_dice_3": None,
            "evaluation_status": None
        }

        if gt_voxels > 0 and pred_voxels == 0:
            # 金标准中有该类，但预测完全缺失
            roi_metrics["dice"] = 0.0
            roi_metrics["surface_dice_3"] = 0.0
            roi_metrics["evaluation_status"] = "FN_only"  # 假阴性
        elif gt_voxels > 0:
            # 正常计算情况
            roi_metrics["dice"] = float(dice_score(gt, pred))
            try:
                sd = compute_surface_distances(gt, pred, spacing_mm)
                roi_metrics["surface_dice_3"] = float(compute_surface_dice_at_tolerance(sd, 3.0))
            except Exception as e:
                roi_metrics["surface_dice_3"] = None
                roi_metrics["evaluation_status"] = f"SurfaceDice_error: {str(e)}"
            else:
                roi_metrics["evaluation_status"] = "success"
        else:
            # 金标准中不存在该类
            roi_metrics["dice"] = None
            roi_metrics["surface_dice_3"] = None
            roi_metrics["evaluation_status"] = "not_present"

        r["metrics"][roi_name] = roi_metrics

    return r


def save_detailed_results(results_df, class_map, save_dir, eval_name="evaluation"):
    """
    保存详细的评估结果到多种格式的文件
    """
    save_dir = Path(save_dir)
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)    #清空该文件夹下的之前的旧的文件
    save_dir.mkdir(parents=True, exist_ok=True)

    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{eval_name}_{timestamp}"
    
    # 1. 保存完整的详细结果到CSV（扁平化结构）
    detailed_records = []
    for _, row in results_df.iterrows():
        subject = row['subject']
        for roi_name in class_map:
            metrics = row['metrics'][roi_name]
            detailed_records.append({
                'subject': subject,
                'roi_name': roi_name,
                'dice': metrics['dice'],
                'surface_dice_3': metrics['surface_dice_3'],
                'gt_voxels': metrics['gt_voxels'],
                'pred_voxels': metrics['pred_voxels'],
                'evaluation_status': metrics['evaluation_status']
            })
    
    detailed_df = pd.DataFrame(detailed_records)
    detailed_csv_path = save_dir / f"{base_filename}_detailed.csv"
    detailed_df.to_csv(detailed_csv_path, index=False, float_format='%.4f')
    
    # 2. 保存汇总统计结果到CSV
    summary_data = []
    for roi_name in class_map:
        # 只统计成功评估的病例（金标准中存在该ROI）
        valid_cases = detailed_df[
            (detailed_df['roi_name'] == roi_name) & 
            (detailed_df['evaluation_status'].isin(['success', 'FN_only']))
        ]
        
        dice_scores = valid_cases['dice'].dropna()
        surface_dice_scores = valid_cases['surface_dice_3'].dropna()
        
        summary_data.append({
            'roi_name': roi_name,
            'num_valid_cases': len(valid_cases),
            'dice_mean': dice_scores.mean() if len(dice_scores) > 0 else None,
            'dice_std': dice_scores.std() if len(dice_scores) > 0 else None,
            'dice_median': dice_scores.median() if len(dice_scores) > 0 else None,
            'surface_dice_mean': surface_dice_scores.mean() if len(surface_dice_scores) > 0 else None,
            'surface_dice_std': surface_dice_scores.std() if len(surface_dice_scores) > 0 else None,
            'surface_dice_median': surface_dice_scores.median() if len(surface_dice_scores) > 0 else None
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = save_dir / f"{base_filename}_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False, float_format='%.4f')
    
    # 3. 保存完整的结构化结果到JSON（保留原始嵌套结构）
    json_results = {
        "evaluation_info": {
            "timestamp": timestamp,
            "eval_name": eval_name,
            "total_subjects": len(results_df),
            "class_names": list(class_map)
        },
        "detailed_results": results_df.to_dict('records')
    }
    
    json_path = save_dir / f"{base_filename}_full.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False, default=str)
    
    # 4. 生成易读的文本报告
    report_path = save_dir / f"{base_filename}_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Segmentation Evaluation Report\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total subjects evaluated: {len(results_df)}\n")
        f.write(f"Classes: {', '.join(class_map)}\n")
        f.write("="*60 + "\n\n")
        
        f.write("SUMMARY STATISTICS:\n")
        f.write("-" * 40 + "\n")
        for _, summary in summary_df.iterrows():
            f.write(f"\n{summary['roi_name']}:\n")
            f.write(f"  Valid cases: {summary['num_valid_cases']}\n")
            if (summary['dice_mean'] is None or summary['dice_std'] is None):
                continue
            f.write(f"  Dice - Mean: {summary['dice_mean']:.4f} ± {summary['dice_std']:.4f}\n")
            f.write(f"  Surface Dice (3mm) - Mean: {summary['surface_dice_mean']:.4f} ± {summary['surface_dice_std']:.4f}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("FILE PATHS:\n")
        f.write(f"Detailed CSV: {detailed_csv_path}\n")
        f.write(f"Summary CSV: {summary_csv_path}\n")
        f.write(f"Full JSON: {json_path}\n")
        f.write(f"This report: {report_path}\n")
    
    return {
        "detailed_csv": detailed_csv_path,
        "summary_csv": summary_csv_path,
        "json_results": json_path,
        "report": report_path
    }


def evaluate(gt_dir, pred_dir, class_map, save_dir=None, eval_name="evaluation"):
    """
    增强版的评估函数，支持详细结果记录和文件保存
    
    Args:
        gt_dir: 金标准mask目录
        pred_dir: 预测mask目录  
        class_map: 类别映射字典
        save_dir: 结果保存目录（如为None则不保存）
        eval_name: 评估任务名称
    
    Returns:
        results_df: 包含详细结果的DataFrame
        file_paths: 保存的文件路径字典（如保存了文件）
    """
    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)
    
    # 获取病例列表
    subjects = [x.stem.split(".")[0] for x in gt_dir.glob("*.nii.gz")]
    print(f"Found {len(subjects)} subjects for evaluation")
    
    # 计算指标
    res = p_map(partial(calc_metrics, gt_dir=gt_dir, pred_dir=pred_dir,
                        class_map=class_map), subjects, num_cpus=8, disable=False)
    res_df = pd.DataFrame(res)
    
    # 打印汇总结果
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    for metric in ["dice", "surface_dice_3"]:
        print(f"\n{metric.upper()} Results:")
        print("-" * 30)
        res_all_rois = []
        for roi_name in class_map:
            # 只统计有效病例（金标准中存在该ROI的病例）
            valid_scores = []
            for _, row in res_df.iterrows():
                roi_metrics = row['metrics'][roi_name]
                if roi_metrics['evaluation_status'] in ['success', 'FN_only']:
                    score = roi_metrics[metric]
                    if score is not None:
                        valid_scores.append(score)
            
            if valid_scores:
                mean_score = np.mean(valid_scores)
                std_score = np.std(valid_scores)
                res_all_rois.append(mean_score)
                print(f"{roi_name}: {mean_score:.3f} ± {std_score:.3f} (n={len(valid_scores)})")
            else:
                print(f"{roi_name}: No valid cases")
        
        if res_all_rois:
            print(f"Overall {metric}: {np.mean(res_all_rois):.3f}")
    
    # 保存结果到文件
    file_paths = None
    if save_dir is not None:
        file_paths = save_detailed_results(res_df, class_map, save_dir, eval_name)
        print(f"\nResults saved to: {save_dir}")
    
    #return res_df, file_paths
    return file_paths["detailed_csv"]


# 使用示例
# if __name__ == "__main__":
#     # 示例配置
#     class_map_example = ["Liver", "Spleen", "Kidney", "Pancreas"]  # 根据实际情况修改
    
#     # 使用示例
#     gt_directory = Path("/path/to/ground_truth")
#     pred_directory = Path("/path/to/predictions") 
#     output_directory = Path("/path/to/evaluation_results")
    
#     # 执行评估
#     results, saved_files = evaluate(
#         gt_dir=gt_directory,
#         pred_dir=pred_directory,
#         class_map=class_map_example,
#         save_dir=output_directory,
#         eval_name="organ_segmentation"
#     )


import pandas as pd
import numpy as np
from pathlib import Path
import os

def aggregate_model_evaluations(csv_files, output_dir=None, output_basename="aggregated"):
    """
    汇总多个模型的评估结果
    
    参数:
    csv_files: list，包含所有模型评估结果CSV文件路径的列表
    output_dir: str，可选，输出目录路径
    output_basename: str，输出文件的基本名称
    
    返回:
    tuple: (汇总后的DataFrame, 每个标签总体Dice的DataFrame)
    """
    
    # 读取所有CSV文件并合并
    all_data = []
    for file_path in csv_files:
        # 从文件名提取模型名称
        model_name = Path(file_path).stem
        
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 添加模型名称列
        df['model_name'] = model_name
        # 添加原文件名列
        df['source_file'] = Path(file_path).name
        
        all_data.append(df)
    
    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 问题1: 按照subject排序
    combined_df = combined_df.sort_values(['subject', 'model_name', 'roi_name']).reset_index(drop=True)
    
    # 计算每个标签的总体Dice系数
    # 只考虑评估成功的样本且dice不为空且大于0
    success_data = combined_df[
        (combined_df['evaluation_status'] == 'success') & 
        (combined_df['dice'].notna()) &
        (combined_df['dice'] > 0)
    ].copy()
    
    # 计算总体Dice（考虑voxels权重）
    roi_dice_stats = []
    
    for roi_name in success_data['roi_name'].unique():
        roi_data = success_data[success_data['roi_name'] == roi_name]
        
        if len(roi_data) == 0:
            continue
            
        # 计算加权平均Dice（使用pred_voxels + gt_voxels作为权重）
        total_voxels = roi_data['pred_voxels'] + roi_data['gt_voxels']
        if total_voxels.sum() > 0:
            weighted_dice = np.average(roi_data['dice'], weights=total_voxels)
        else:
            weighted_dice = roi_data['dice'].mean()
        
        # 简单平均Dice
        mean_dice = roi_data['dice'].mean()
        
        # 统计样本数量
        num_subjects = roi_data['subject'].nunique()
        num_evaluations = len(roi_data)
        
        # 获取来源文件信息
        source_files = roi_data['source_file'].unique()
        
        roi_dice_stats.append({
            'roi_name': roi_name,
            'source_files': ', '.join(source_files),
            'weighted_mean_dice': weighted_dice,
            'mean_dice': mean_dice,
            'num_subjects': num_subjects,
            'num_evaluations': num_evaluations,
            'min_dice': roi_data['dice'].min(),
            'max_dice': roi_data['dice'].max(),
            'std_dice': roi_data['dice'].std()
        })
    
    # 创建总体Dice统计DataFrame
    overall_dice_df = pd.DataFrame(roi_dice_stats)
    
    # 问题2: 按照输入文件的顺序保持roi_name的顺序
    # 首先从原始数据中提取roi_name的顺序
    original_roi_order = []
    for file_path in csv_files:
        df_temp = pd.read_csv(file_path)
        # 只添加新的roi_name，保持顺序
        for roi in df_temp['roi_name'].unique():
            if roi not in original_roi_order:
                original_roi_order.append(roi)
    
    # 如果overall_dice_df中有不在original_roi_order中的roi，添加到末尾
    # 注意：当没有任何成功评估记录时 overall_dice_df 为空且无列，需先判断再访问 'roi_name'
    if not overall_dice_df.empty and 'roi_name' in overall_dice_df.columns:
        for roi in overall_dice_df['roi_name']:
            if roi not in original_roi_order:
                original_roi_order.append(roi)

        # 创建排序映射，按 original_roi_order 对 overall_dice_df 排序
        roi_order_map = {roi: i for i, roi in enumerate(original_roi_order)}
        overall_dice_df['order'] = overall_dice_df['roi_name'].map(roi_order_map)
        overall_dice_df = overall_dice_df.sort_values('order').drop('order', axis=1).reset_index(drop=True)
    
    # 如果有输出目录，保存结果
    if output_dir:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存合并的详细结果
        combined_output = os.path.join(output_dir, f"{output_basename}_detailed.csv")
        combined_df.to_csv(combined_output, index=False)
        print(f"详细汇总结果已保存至: {combined_output}")
        
        # 保存总体Dice统计
        dice_output = os.path.join(output_dir, f"{output_basename}_overall_dice.csv")
        overall_dice_df.to_csv(dice_output, index=False)
        print(f"总体Dice统计已保存至: {dice_output}")
    
    return combined_df, overall_dice_df

def aggregate_by_subject(combined_df, output_dir=None, output_basename="aggregated"):
    """
    按subject汇总每个模型的表现
    
    参数:
    combined_df: 合并后的DataFrame
    output_dir: 可选，输出目录路径
    output_basename: str，输出文件的基本名称
    
    返回:
    DataFrame: 按subject汇总的结果
    """
    # 问题3: 过滤掉空白或无效数据
    valid_data = combined_df[
        (combined_df['evaluation_status'] == 'success') & 
        (combined_df['dice'].notna()) &
        (combined_df['dice'] > 0)
    ].copy()
    
    if len(valid_data) == 0:
        print("警告: 没有找到有效的评估数据")
        return pd.DataFrame()
    
    # 按subject和模型名称分组，统计成功的评估数量
    subject_stats = valid_data.groupby(['subject', 'model_name']).agg({
        'dice': ['mean', 'count'],
        'roi_name': 'nunique'
    }).reset_index()
    
    # 简化列名
    subject_stats.columns = ['subject', 'model_name', 'mean_dice', 'successful_evaluations', 'unique_rois']
    
    # 计算每个subject的总评估数（包括失败的）
    total_evaluations = combined_df.groupby(['subject', 'model_name']).size().reset_index(name='total_evaluations')
    
    # 合并数据
    subject_stats = subject_stats.merge(total_evaluations, on=['subject', 'model_name'], how='left')
    
    # 计算成功率
    subject_stats['success_rate'] = subject_stats['successful_evaluations'] / subject_stats['total_evaluations']
    
    # 按照subject和model_name排序
    subject_stats = subject_stats.sort_values(['subject', 'model_name']).reset_index(drop=True)
    
    if output_dir:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        subject_output = os.path.join(output_dir, f"{output_basename}_case_dice.csv")
        subject_stats.to_csv(subject_output, index=False)
        print(f"按subject汇总结果已保存至: {subject_output}")
    
    return subject_stats

def run_evaluation_aggregation(input_files, output_dir, output_basename="aggregated_results", clear_output=False):
    """
    运行评估结果汇总的主要函数
    
    参数:
    input_files: list，包含所有模型评估结果CSV文件路径的列表
    output_dir: str，汇总结果的保存目录
    output_basename: str，输出文件的基本名称
    clear_output: bool，是否在开始前清空输出文件夹
    
    返回:
    tuple: (combined_df, overall_dice_df, subject_stats)
    """
    
    # 确保输入是列表格式
    if isinstance(input_files, str):
        input_files = [input_files]
    
    # 检查文件是否存在
    for file_path in input_files:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"找不到输入文件: {file_path}")
    
    # 如果需要清空输出文件夹
    if clear_output and os.path.exists(output_dir):
        print(f"清空输出目录: {output_dir}")
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    import shutil
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"删除 {file_path} 时出错: {e}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    print("开始汇总评估结果...")
    print(f"输入文件: {input_files}")
    print(f"输出目录: {output_dir}")
    print(f"输出文件前缀: {output_basename}")
    print("-" * 50)
    
    # 汇总评估结果
    combined_df, overall_dice_df = aggregate_model_evaluations(
        input_files, 
        output_dir=output_dir,
        output_basename=output_basename
    )
    
    # 按subject汇总
    subject_stats = aggregate_by_subject(
        combined_df,
        output_dir=output_dir,
        output_basename=output_basename
    )
    
    # 打印总体统计
    print("\n" + "="*50)
    print("汇总完成！统计信息:")
    print("="*50)
    print(f"总评估数量: {len(combined_df)}")
    print(f"唯一subject数量: {combined_df['subject'].nunique()}")
    print(f"唯一ROI标签数量: {combined_df['roi_name'].nunique()}")
    print(f"模型数量: {combined_df['model_name'].nunique()}")
    
    # 成功率统计
    success_count = len(combined_df[combined_df['evaluation_status'] == 'success'])
    success_rate = success_count / len(combined_df) * 100 if len(combined_df) > 0 else 0
    print(f"成功评估数量: {success_count}")
    print(f"总体成功率: {success_rate:.2f}%")
    
    # 显示前10个ROI的Dice统计
    if len(overall_dice_df) > 0:
        print(f"\nROI Dice统计 (按原始文件顺序):")
        print("-" * 60)
        for idx, row in overall_dice_df.head(15).iterrows():
            print(f"{row['roi_name']:25} | {row['weighted_mean_dice']:.4f} | 来源: {row['source_files']}")
    else:
        print("\n没有可用的Dice统计结果")
    
    return combined_df, overall_dice_df, subject_stats


def _resolve_class_maps_from_config(config_file: Path):
    """从 Config + ModelMap 解析每个 train_dataset 对应的 class_map。"""
    with config_file.open("rb") as f:
        config = tomllib.load(f)

    model_file = config["MODEL"]["segment_model_file"]
    model_file_path = (config_file.parent / model_file).resolve()
    with model_file_path.open("rb") as f:
        model_map = tomllib.load(f)

    segment_list_name = config["MODEL"].get("segment_list_name", [])
    class_maps = [_build_eval_class_map(model_map[name]) for name in segment_list_name]
    train_datasets = config["MODEL"].get("train_dataset", [])
    return config, train_datasets, class_maps


def stage_evaluation(
    nnunet_paths,
    class_maps,
    gt_subdir: str = DEFAULT_GT_SUBDIR,
    pred_subdir: str = DEFAULT_PRED_SUBDIR,
    eval_subdir: str = DEFAULT_EVAL_SUBDIR,
    aggregate_output_dir=None,
    aggregate_output_basename: str = "evaluation",
    run_aggregation: bool = True,
    clear_aggregate_output: bool = False,
):
    """
    按数据集执行评估，并可选聚合多个模型的结果。

    Returns:
        list[str]: 每个数据集详细评估 CSV 的路径列表
    """
    evaluate_files = []
    if len(class_maps) != len(nnunet_paths):
        raise ValueError(f"class_maps 与 nnunet_paths 长度不一致: {len(class_maps)} vs {len(nnunet_paths)}")

    for nnunet_path, class_map in zip(nnunet_paths, class_maps):
        dataset_root = Path(nnunet_path)
        eval_file = evaluate(
            gt_dir=dataset_root / gt_subdir,
            pred_dir=dataset_root / pred_subdir,
            class_map=class_map,
            save_dir=dataset_root / eval_subdir,
            eval_name=aggregate_output_basename,
        )
        evaluate_files.append(eval_file)

    if run_aggregation and evaluate_files:
        if aggregate_output_dir is None:
            aggregate_output_dir = str(Path(nnunet_paths[0]).parent)
        run_evaluation_aggregation(
            input_files=evaluate_files,
            output_dir=aggregate_output_dir,
            output_basename=aggregate_output_basename,
            clear_output=clear_aggregate_output,
        )

    return evaluate_files


def stage_evaluation_from_config(config_file, run_aggregation=True):
    """
    从配置文件直接执行评估。
    需要 Config 中包含 PATHS/MODEL，EVALUATION 段为可选。
    """
    config_path = Path(config_file).resolve()
    config, train_datasets, class_maps = _resolve_class_maps_from_config(config_path)

    train_root = Path(config["PATHS"]["train_path"]) / config["PATHS"]["train_project"]
    nnunet_raw_name = config["PATHS"].get("nnUNet_raw", "nnUNet_raw")
    nnunet_raw_root = train_root / nnunet_raw_name
    nnunet_paths = [nnunet_raw_root / name for name in train_datasets]

    eval_cfg = config.get("EVALUATION", {})
    gt_subdir = eval_cfg.get("gt_subdir", DEFAULT_GT_SUBDIR)
    pred_subdir = eval_cfg.get("pred_subdir", DEFAULT_PRED_SUBDIR)
    eval_subdir = eval_cfg.get("eval_subdir", DEFAULT_EVAL_SUBDIR)
    aggregate_output_dir = eval_cfg.get("aggregate_output_dir")
    aggregate_output_basename = eval_cfg.get("aggregate_output_basename", "evaluation")
    clear_aggregate_output = bool(eval_cfg.get("clear_aggregate_output", False))
    run_aggregation_cfg = bool(eval_cfg.get("run_aggregation", True))

    return stage_evaluation(
        nnunet_paths=nnunet_paths,
        class_maps=class_maps,
        gt_subdir=gt_subdir,
        pred_subdir=pred_subdir,
        eval_subdir=eval_subdir,
        aggregate_output_dir=aggregate_output_dir,
        aggregate_output_basename=aggregate_output_basename,
        run_aggregation=run_aggregation and run_aggregation_cfg,
        clear_aggregate_output=clear_aggregate_output,
    )


def main() -> None:
    # # 独立运行时仅需修改这里，不依赖 argparse。
    # run_cfg = {
    #     "config_file": "/data1/User/shijian_ruan/UIH_Seg/PythonFiles/auto-segmentation/workflow/Config_MROrganMIv500.toml",
    #     "run_aggregation": True,
    # }

    # stage_evaluation_from_config(
    #     config_file=run_cfg["config_file"],
    #     run_aggregation=bool(run_cfg.get("run_aggregation", True)),
    # )

       # -------------------------------------------------------------------------
    # 手动配置区域：修改以下变量以适配实际路径和模型
    # -------------------------------------------------------------------------

    # ModelMap 文件路径（包含各模型的 class_map 定义）
    model_map_file = Path(__file__).parent / "ModelMap.toml"

    # 需要评估的模型名称列表（对应 ModelMap_Bone.toml 中 [MODELS] 下的 key）
    segment_list_names = [
        "CT_Coarse",   # 根据实际情况修改，可添加多个
    ]

    # 每个模型对应的路径配置，顺序须与 segment_list_names 一一对应
    # gt_dir:   金标准 mask 目录（直接指定完整路径）
    # pred_dir: 预测结果 mask 目录（直接指定完整路径）
    # eval_dir: 单模型评估结果输出目录（直接指定完整路径）
    path_configs = [
        {
            "gt_dir":   r"D:\data\CoarseSegmentation\Dataset101_CTCoarseSegmentation\labelsTs",
            "pred_dir": r"D:\data\CoarseSegmentation\Dataset101_CTCoarseSegmentation\labelsTs_predicted",
            "eval_dir": r"D:\data\CoarseSegmentation\Dataset101_CTCoarseSegmentation\evaluation",
        },
        # 如有多个模型，继续添加...
    ]

    # 聚合结果配置
    aggregate_output_dir      = r"D:\data\CoarseSegmentation\Dataset101_CTCoarseSegmentation\aggregated"  # 聚合结果保存目录
    aggregate_output_basename = "evaluation"   # 输出文件前缀
    run_aggregation           = False           # 是否进行多模型结果聚合
    clear_aggregate_output    = False          # 是否在聚合前清空输出目录
    eval_name                 = "evaluation"   # 评估任务名称

    # -------------------------------------------------------------------------
    # 从 ModelMap 文件中读取 class_maps
    # -------------------------------------------------------------------------
    with model_map_file.open("rb") as f:
        model_map = tomllib.load(f)

    class_maps = [_build_eval_class_map(model_map[name]) for name in segment_list_names]

    # -------------------------------------------------------------------------
    # 执行评估（直接调用 evaluate()，不依赖 nnUNet 目录结构或环境变量）
    # -------------------------------------------------------------------------
    if len(class_maps) != len(path_configs):
        raise ValueError(
            f"segment_list_names 与 path_configs 长度不一致: "
            f"{len(class_maps)} vs {len(path_configs)}"
        )

    evaluate_files = []
    for class_map, paths in zip(class_maps, path_configs):
        eval_file = evaluate(
            gt_dir=paths["gt_dir"],
            pred_dir=paths["pred_dir"],
            class_map=class_map,
            save_dir=paths["eval_dir"],
            eval_name=eval_name,
        )
        evaluate_files.append(eval_file)

    # 聚合多个模型的评估结果
    if run_aggregation and evaluate_files:
        run_evaluation_aggregation(
            input_files=evaluate_files,
            output_dir=aggregate_output_dir,
            output_basename=aggregate_output_basename,
            clear_output=clear_aggregate_output,
        )


if __name__ == "__main__":
    main()

    

# 使用示例
# if __name__ == "__main__":
#     # 示例调用
#     input_files = [
#         'model1_evaluation.csv',
#         'model2_evaluation.csv',
#         # 添加更多模型文件...
#     ]
    
#     output_dir = './aggregated_results'  # 现在这是一个目录路径
#     output_basename = 'final_summary'    # 输出文件的基本名称
    
#     # 运行汇总
#     combined_df, overall_dice_df, subject_stats = run_evaluation_aggregation(
#         input_files, 
#         output_dir=output_dir,
#         output_basename=output_basename,
#         clear_output=True  # 可选：是否清空输出目录
#     )