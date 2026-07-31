import multiprocessing
import os
from typing import Union, Optional

import torch


def stage_train(
    dataset_id: Union[int, str],
    configuration: str,
    fold: Union[int, str],
    trainer: str,
    plans: str,
    pretrained_weights: Optional[str] = None,
    num_gpus: int = 1,
    continue_training: bool = False,
    only_run_validation: bool = False,
    export_validation_probabilities: bool = False,
    disable_checkpointing: bool = False,
    val_with_best: bool = False,
    device: torch.device = None,
    gpu_id: Optional[int] = None,
) -> None:
    """
    模型训练。
    对应命令：
        nnUNetv2_train <dataset_id> <configuration> <fold>
            -tr <trainer> -p <plans>
            [-pretrained_weights <path>]
            [--c] [--val] [--npz] [--val_best] [--disable_checkpointing]
            [-num_gpus <n>] [-device <dev>]

    参数说明
    ----------
    dataset_id : int 或 str
        数据集 ID（如 101）或名称（如 "Dataset101_brain_mr"）。
    configuration : str
        训练配置，一般为 "3d_fullres"、"2d" 等。
    fold : int 或 str
        交叉验证折数（0-4）或 "all"。
    trainer : str
        nnUNet Trainer 类名，默认 "nnUNetTrainerNoMirroring"。
    plans : str
        计划标识符，默认 "nnUNetPlans"。
    pretrained_weights : str, optional
        预训练权重文件路径（.pth）。仅在首次训练时使用，不能与 continue_training 同时使用。
    num_gpus : int
        使用的 GPU 数量，>1 时启用 DDP 分布式训练。
    continue_training : bool
        是否从上次中断处继续训练（等价于 --c）。
    only_run_validation : bool
        仅运行验证，不训练（等价于 --val）。需要训练已完成。
    export_validation_probabilities : bool
        是否以 .npz 格式保存验证阶段的 softmax 概率（等价于 --npz）。
    disable_checkpointing : bool
        是否禁用 checkpoint 保存（等价于 --disable_checkpointing）。
    val_with_best : bool
        验证时使用 checkpoint_best 而非 checkpoint_final（等价于 --val_best）。
    device : torch.device, optional
        训练设备。默认自动检测 cuda/cpu。
    gpu_id : int, optional
        指定使用的 GPU 编号（对应 CUDA_VISIBLE_DEVICES）。
        为 None 时不修改当前环境的 GPU 设定。
    """

    # ── 指定 GPU ────────────────────────────────────────────
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # ── 设置多线程参数（与 nnUNet 官方入口保持一致） ──────────
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '1')

    # ── 设备选择 ────────────────────────────────────────────
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    if device.type == 'cpu':
        torch.set_num_threads(multiprocessing.cpu_count())
    elif device.type == 'cuda':
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    print(f"\n{'='*60}")
    print(f"阶段 2 / 训练  dataset={dataset_id}  config={configuration}  fold={fold}")
    print(f"  trainer : {trainer}")
    print(f"  plans   : {plans}")
    print(f"  device  : {device}")
    print(f"  num_gpus: {num_gpus}")
    if pretrained_weights:
        print(f"  pretrained_weights: {pretrained_weights}")
    if continue_training:
        print(f"  继续训练模式 (--c)")
    if only_run_validation:
        print(f"  仅运行验证 (--val)")
    print(f"{'='*60}")

    # ── 延迟导入：避免 worker 进程重新执行脚本时触发重量级 DLL 加载 ──
    from nnunetv2.run.run_training import run_training

    run_training(
        dataset_name_or_id=str(dataset_id),
        configuration=configuration,
        fold=fold,
        trainer_class_name=trainer,
        plans_identifier=plans,
        pretrained_weights=pretrained_weights,
        num_gpus=num_gpus,
        export_validation_probabilities=export_validation_probabilities,
        continue_training=continue_training,
        only_run_validation=only_run_validation,
        disable_checkpointing=disable_checkpointing,
        val_with_best=val_with_best,
        device=device,
    )

    print("  训练完成。")


# ── 支持直接运行本文件 ──────────────────────────────────────
if __name__ == "__main__":
    stage_train()
