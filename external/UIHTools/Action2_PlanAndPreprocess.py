

def _override_plans(dataset_id, plans_identifier, configuration,
                    target_patch_size=None, target_batch_size=None):
    """
    在 nnUNet 自动规划完成后，修改 plans JSON 中指定 configuration 的参数。

    支持覆盖：
      - patch_size: 同时自动重算网络拓扑参数（pool/conv kernel、n_stages 等）
      - batch_size: 直接覆盖（优先级最高，跳过 VRAM 自动估算）

    不需要重新执行预处理（预处理只依赖 spacing，与 patch_size/batch_size 无关）。
    """
    import json
    from pathlib import Path
    import os

    from nnunetv2.paths import nnUNet_preprocessed
    from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name
    from nnunetv2.experiment_planning.experiment_planners.network_topology import get_pool_and_conv_props

    dataset_name = convert_id_to_dataset_name(dataset_id)
    plans_path = Path(nnUNet_preprocessed) / dataset_name / f"{plans_identifier}.json"

    with open(plans_path, "r", encoding="utf-8") as f:
        plans = json.load(f)

    if configuration not in plans["configurations"]:
        print(f"  警告：configuration '{configuration}' 不存在于 plans 中，跳过 patch_size 覆盖")
        return

    cfg = plans["configurations"][configuration]
    spacing = cfg["spacing"]

    # ── 仅覆盖 batch_size（不改 patch_size / 网络拓扑）──
    if target_patch_size is None and target_batch_size is not None:
        old_bs = cfg["batch_size"]
        cfg["batch_size"] = target_batch_size
        print(f"  batch_size 覆盖: {old_bs} → {target_batch_size}")
        from nnunetv2.utilities.json_export import recursive_fix_for_json_export
        recursive_fix_for_json_export(plans)
        with open(plans_path, "w", encoding="utf-8") as f:
            json.dump(plans, f, indent=2, ensure_ascii=False)
        return

    # ── 覆盖 patch_size（同时重算网络拓扑）──
    old_patch_size = cfg["patch_size"]
    print(f"  原始 patch_size: {old_patch_size}")
    print(f"  目标 patch_size: {target_patch_size}")
    print(f"  spacing:         {spacing}")

    # ── 使用 get_pool_and_conv_props 重新计算网络拓扑 ──
    # 该函数会自动将 patch_size 向上 pad 为 2^num_pool 的整数倍
    num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, new_patch_size, \
        shape_must_be_divisible_by = get_pool_and_conv_props(
            spacing, target_patch_size,
            min_feature_map_size=4, max_numpool=999999
        )

    num_stages = len(pool_op_kernel_sizes)

    if list(new_patch_size) != list(target_patch_size):
        print(f"  patch_size 已自动调整为可整除值: {list(target_patch_size)} → {list(new_patch_size)}")

    # ── 更新 architecture 参数 ──
    arch = cfg["architecture"]["arch_kwargs"]
    # 保持 UNet_base_num_features=32, max_features 根据维度判断
    dim = len(spacing)
    max_features = 320 if dim == 3 else 512
    base_features = 32
    features_per_stage = [min(max_features, base_features * 2 ** i) for i in range(num_stages)]

    # encoder 每层 conv 数（nnUNet 默认全部为 2）
    blocks_per_stage_encoder = [2] * num_stages
    blocks_per_stage_decoder = [2] * (num_stages - 1)

    arch["n_stages"] = num_stages
    arch["features_per_stage"] = features_per_stage
    arch["kernel_sizes"] = [list(k) for k in conv_kernel_sizes]
    arch["strides"] = [list(s) for s in pool_op_kernel_sizes]
    arch["n_conv_per_stage"] = blocks_per_stage_encoder
    arch["n_conv_per_stage_decoder"] = blocks_per_stage_decoder

    cfg["patch_size"] = list(new_patch_size)

    # ── 重新估算 batch_size ──
    try:
        import numpy as np
        from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner

        # 读取必要信息
        dataset_json_path = Path(nnUNet_preprocessed) / dataset_name / "dataset.json"
        with open(dataset_json_path, "r", encoding="utf-8") as f:
            dataset_json = json.load(f)

        num_input_channels = len(dataset_json.get('channel_names', dataset_json.get('modality', {})))
        num_output_channels = len(dataset_json.get('labels', {}))

        estimate = ExperimentPlanner.static_estimate_VRAM_usage(
            tuple(new_patch_size),
            num_input_channels,
            num_output_channels,
            cfg["architecture"]["network_class_name"],
            arch,
            cfg["architecture"]["_kw_requires_import"],
        )

        # 使用 nnUNet 默认参考值估算 batch_size
        UNet_vram_target_GB = 8
        UNet_reference_val_corresp_GB = 8
        if dim == 3:
            reference = 560000000 * (UNet_vram_target_GB / UNet_reference_val_corresp_GB)
            ref_bs = 2
        else:
            reference = 85000000 * (UNet_vram_target_GB / UNet_reference_val_corresp_GB)
            ref_bs = 12

        batch_size = max(2, round((reference / estimate) * ref_bs))
        cfg["batch_size"] = batch_size
        print(f"  重新估算 batch_size: {batch_size}")
    except Exception as e:
        print(f"  警告：无法自动估算 batch_size，保持原值 {cfg['batch_size']}。原因：{e}")

    # ── 用户指定的 batch_size 优先级最高，覆盖自动估算值 ──
    if target_batch_size is not None:
        cfg["batch_size"] = target_batch_size
        print(f"  用户指定 batch_size 覆盖: {target_batch_size}")

    # ── 写回 plans ──
    # 将 numpy 类型（np.int64 等）转换为 Python 原生类型，否则 json.dump 会写出无效 JSON
    from nnunetv2.utilities.json_export import recursive_fix_for_json_export
    recursive_fix_for_json_export(plans)

    with open(plans_path, "w", encoding="utf-8") as f:
        json.dump(plans, f, indent=2, ensure_ascii=False)

    print(f"  patch_size 覆盖完成: {list(new_patch_size)}")
    print(f"  网络层数: {num_stages}, 池化核: {[list(s) for s in pool_op_kernel_sizes]}")


def stage_preprocess(
    dataset_id: int,
    configuration: str,
    num_processes: int,
    target_spacing = None,
    target_patch_size = None,
    target_batch_size = None,
    verify_integrity: bool = True,
) -> str:
    """
    数据预处理：指纹提取 → 实验规划 → (可选)覆盖 patch_size/batch_size → 预处理。
    对应命令：
        nnUNetv2_plan_and_preprocess -d <dataset_id> -c <config>
            --verify_dataset_integrity -np <num_processes>

    参数:
        target_patch_size: 可选，手动指定的 patch_size（如 [128, 128, 128]）。
            为 None 时由 nnUNet 自动规划。
            设置后会自动向上取整为满足网络拓扑约束的值，
            并重新计算 pool_op_kernel_sizes / conv_kernel_sizes / batch_size 等参数。
        target_batch_size: 可选，手动指定的 batch_size（如 4、8）。
            为 None 时由 nnUNet 自动规划（默认按 8GB 显存估算）。

    返回生成的 plans_identifier 字符串。
    """
    # 延迟导入：避免 worker 进程重新执行脚本时触发重量级 DLL 加载
    from nnunetv2.experiment_planning.plan_and_preprocess_api import (
        extract_fingerprints, plan_experiments, preprocess,
    )

    print(f"\n{'='*60}")
    print(f"阶段 1 / 预处理  dataset={dataset_id}  config={configuration}")
    print(f"{'='*60}")

    print("  步骤 1/3：指纹提取 (extract_fingerprints)...")
    extract_fingerprints(
        dataset_ids=[dataset_id],
        num_processes=num_processes,
        check_dataset_integrity=verify_integrity,
    )

    print("  步骤 2/3：实验规划 (plan_experiments)...")
    if target_spacing is not None and len(target_spacing) > 0:
        print(f"  目标间距: {target_spacing} mm")
        plans_id = plan_experiments(
            dataset_ids=[dataset_id],
            overwrite_target_spacing=target_spacing,
        )
    else:
        print("  目标间距: 由 nnUNet 自动规划")
        plans_id = plan_experiments(dataset_ids=[dataset_id])

    # 可选：覆盖 patch_size / batch_size
    if target_patch_size is not None or target_batch_size is not None:
        overrides = []
        if target_patch_size is not None:
            overrides.append(f"patch_size={target_patch_size}")
        if target_batch_size is not None:
            overrides.append(f"batch_size={target_batch_size}")
        print(f"  步骤 2.5/3：覆盖 {', '.join(overrides)}...")
        _override_plans(dataset_id, plans_id, configuration,
                        target_patch_size=target_patch_size,
                        target_batch_size=target_batch_size)

    print("  步骤 3/3：数据预处理 (preprocess)...")
    preprocess(
        dataset_ids=[dataset_id],
        plans_identifier=plans_id,
        configurations=[configuration],
        num_processes=[num_processes],
    )

    print(f"  预处理完成，plans_identifier = {plans_id}")
    return plans_id