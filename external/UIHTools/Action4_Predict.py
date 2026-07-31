from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import csv
import os
import platform
import shutil
import tempfile
import threading
import time
import multiprocessing as mp
from datetime import datetime

import torch



SUPPORTED_IMAGE_SUFFIXES: Tuple[str, ...] = (
    ".nii.gz",
    ".nii",
    ".mhd",
    ".mha",
    ".nrrd",
)


def _to_path(value: Union[str, Path]) -> Path:
    return value if isinstance(value, Path) else Path(str(value))


def _normalize_single_dataset_id(dataset_id: Union[int, str, Sequence[Union[int, str]]]) -> Union[int, str]:
    if isinstance(dataset_id, (list, tuple)):
        if len(dataset_id) != 1:
            raise ValueError(f"dataset_id 当前仅支持单模型预测，收到: {dataset_id}")
        return dataset_id[0]
    return dataset_id


def _suffix_lower(path: Path) -> str:
    name = path.name.lower()
    for suffix in SUPPORTED_IMAGE_SUFFIXES:
        if name.endswith(suffix):
            return suffix
    return path.suffix.lower()


def _is_supported_image_file(path: Path) -> bool:
    return path.is_file() and _suffix_lower(path) in SUPPORTED_IMAGE_SUFFIXES


def parse_input_paths(input_path: Union[str, Path]) -> List[Path]:
    """解析输入路径（文件或文件夹），返回受支持的医学影像文件列表。"""
    src = _to_path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"输入路径不存在: {src}")

    if src.is_file():
        if not _is_supported_image_file(src):
            raise ValueError(f"不支持的输入格式: {src}")
        return [src.resolve()]

    files = [p.resolve() for p in src.rglob("*") if _is_supported_image_file(p)]
    if not files:
        raise ValueError(f"目录中未找到受支持的影像文件: {src}")
    files.sort()
    return files


def _safe_relpath(path: Path, root: Optional[Path]) -> Path:
    if root is None:
        return Path(path.name)
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return Path(path.name)


def build_io_mapping(
    input_files: Sequence[Path],
    output_path: Union[str, Path],
    input_root: Optional[Union[str, Path]] = None,
) -> Dict[Path, Path]:
    """
    建立输入输出映射。
    - 单文件输入可映射到单文件输出
    - 多文件输入要求输出为目录
    - 默认保留输入相对目录结构与扩展名
    """
    if not input_files:
        raise ValueError("input_files 不能为空")

    out = _to_path(output_path)
    root = _to_path(input_root) if input_root is not None else None

    mapping: Dict[Path, Path] = {}
    if len(input_files) == 1 and (out.suffix or out.name.lower().endswith(".nii.gz")):
        src = input_files[0]
        mapping[src] = out
        return mapping

    if len(input_files) > 1 and (out.suffix or out.name.lower().endswith(".nii.gz")):
        raise ValueError("多文件输入时 output_path 必须是目录，不能是单文件路径")

    for src in input_files:
        rel = _safe_relpath(src, root)
        # nnUNet 多通道输入约定：第一通道文件名以 _0000 结尾（如 patient001_0000.nii.gz）。
        # 预测输出的 mask 不需要通道后缀，因此仅精确去除 _0000，
        # 其他数值后缀（_0001/_0002 等）不处理，避免误改正常文件名。
        # 例如: patient001_0000.nii.gz → patient001.nii.gz
        rel_name = rel.name
        for img_suffix in SUPPORTED_IMAGE_SUFFIXES:
            if rel_name.lower().endswith(img_suffix):
                stem = rel_name[: len(rel_name) - len(img_suffix)]
                if stem.endswith("_0000"):
                    rel = rel.parent / (stem[:-5] + img_suffix)
                break
        mapping[src] = out / rel
    return mapping


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_image_with_sitk(image_path: Path):
    import SimpleITK as sitk

    return sitk.ReadImage(str(image_path))


def _is_nifti_path(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def _repair_nifti_header_with_nibabel(src: Path, dst: Path) -> None:
    """使用 nibabel 重写 NIfTI header，尽量修复 qform/sform 不一致问题。"""
    try:
        import nibabel as nib
    except Exception as exc:
        raise RuntimeError("未安装 nibabel，无法执行 NIfTI header 修复") from exc

    img = nib.load(str(src))
    affine = img.affine
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)
    nib.save(img, str(dst))


def convert_to_nnunet_format(
    image_path: Union[str, Path],
    temp_cases_dir: Union[str, Path],
    case_id: str,
) -> Path:
    """将任意支持格式转为 nnUNet 预测输入格式 `<case>_0000.nii.gz`。"""
    src = _to_path(image_path)
    case_dir = _to_path(temp_cases_dir) / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    dst = case_dir / f"{case_id}_0000.nii.gz"

    if src.name.lower().endswith("_0000.nii.gz") and src.suffix.lower() == ".gz":
        shutil.copy2(src, dst)
        return dst

    # NIfTI 仅需改名/重写到 `_0000.nii.gz`，尽量避免 SITK 读取触发方向余弦约束报错。
    if _is_nifti_path(src):
        if src.name.lower().endswith(".nii.gz"):
            shutil.copy2(src, dst)
            return dst

        try:
            _repair_nifti_header_with_nibabel(src, dst)
            return dst
        except Exception as exc:
            print(f"[警告] NIfTI header 修复失败，尝试使用 SITK 转换: {src} ({exc})")

    import SimpleITK as sitk

    img = sitk.ReadImage(str(src))
    sitk.WriteImage(img, str(dst))
    return dst


@dataclass
class CaseStats:
    dataset: str
    case_file: str
    prediction_time_sec: float      # 本例总耗时（含预插值+推理+后插值）
    gpu_mem_peak_mb: float
    gpu_mem_valley_mb: float
    gpu_mem_diff_mb: float
    image_size_x: int
    image_size_y: int
    image_size_z: int
    spacing_x: float
    spacing_y: float
    spacing_z: float
    timestamp: str
    # 分阶段耗时（easy_predict_with_preresample 填写，其他接口默认 0）
    preresample_time_sec: float = field(default=0.0)
    inference_time_sec: float = field(default=0.0)
    postresample_time_sec: float = field(default=0.0)


class TimeStats:
    def __init__(self) -> None:
        self._records: List[CaseStats] = []

    @property
    def records(self) -> List[CaseStats]:
        return self._records

    def add(self, record: CaseStats) -> None:
        self._records.append(record)

    def print_summary(self) -> None:
        if not self._records:
            print("[统计] 没有可汇总的病例。")
            return

        n = len(self._records)
        times  = [r.prediction_time_sec for r in self._records]
        pres   = [r.preresample_time_sec for r in self._records]
        infers = [r.inference_time_sec for r in self._records]
        posts  = [r.postresample_time_sec for r in self._records]
        peaks   = [r.gpu_mem_peak_mb for r in self._records]
        valleys = [r.gpu_mem_valley_mb for r in self._records]
        diffs   = [r.gpu_mem_diff_mb for r in self._records]

        print("\n" + "=" * 60)
        print("全局统计")
        print(f"  病例数              : {n}")
        print(f"  {'阶段':<16}  {'总计(s)':>10}  {'均值(s)':>10}  {'最大(s)':>10}  {'最小(s)':>10}")
        print(f"  {'-'*16}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
        for label, vals in [
            ("总耗时",   times),
            ("  预插值", pres),
            ("  推理",   infers),
            ("  后插值", posts),
        ]:
            print(f"  {label:<16}  {sum(vals):>10.2f}  {sum(vals)/n:>10.2f}  {max(vals):>10.2f}  {min(vals):>10.2f}")
        print(f"  {'显存峰值均值(MB)':<16}  {sum(peaks)/n:>10.2f}")
        print(f"  {'显存低谷均值(MB)':<16}  {sum(valleys)/n:>10.2f}")
        print(f"  {'显存波动均值(MB)':<16}  {sum(diffs)/n:>10.2f}")
        print("=" * 60)

    def export_csv(self, file_path: Union[str, Path]) -> Path:
        dst = _to_path(file_path)
        _ensure_parent_dir(dst)
        with dst.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "dataset",
                    "case_file",
                    "total_time_sec",
                    "preresample_time_sec",
                    "inference_time_sec",
                    "postresample_time_sec",
                    "gpu_mem_peak_mb",
                    "gpu_mem_valley_mb",
                    "gpu_mem_diff_mb",
                    "image_size_x",
                    "image_size_y",
                    "image_size_z",
                    "spacing_x",
                    "spacing_y",
                    "spacing_z",
                    "timestamp",
                ]
            )
            for r in self._records:
                writer.writerow(
                    [
                        r.dataset,
                        r.case_file,
                        f"{r.prediction_time_sec:.4f}",
                        f"{r.preresample_time_sec:.4f}",
                        f"{r.inference_time_sec:.4f}",
                        f"{r.postresample_time_sec:.4f}",
                        f"{r.gpu_mem_peak_mb:.2f}",
                        f"{r.gpu_mem_valley_mb:.2f}",
                        f"{r.gpu_mem_diff_mb:.2f}",
                        r.image_size_x,
                        r.image_size_y,
                        r.image_size_z,
                        f"{r.spacing_x:.6f}",
                        f"{r.spacing_y:.6f}",
                        f"{r.spacing_z:.6f}",
                        r.timestamp,
                    ]
                )
        return dst


class GPUMemoryMonitor:
    """基于 pynvml + psutil 的进程级显存监控，非 NVIDIA 环境自动降级。"""

    def __init__(self, gpu_device_id: int = 0, interval_sec: float = 0.1) -> None:
        self.gpu_device_id = gpu_device_id
        self.interval_sec = interval_sec
        self._available = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._peak_mb = 0.0
        self._valley_mb = float("inf")
        self._pid = os.getpid()
        self._nvml = None
        self._psutil = None
        self._handle = None

        try:
            import pynvml
            import psutil

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_device_id)
            self._nvml = pynvml
            self._psutil = psutil
            self._available = True
        except Exception:
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def _sample(self) -> float:
        assert self._nvml is not None
        assert self._psutil is not None

        try:
            parent = self._psutil.Process(self._pid)
            pids = {self._pid, *[p.pid for p in parent.children(recursive=True)]}
        except Exception:
            pids = {self._pid}

        total_bytes = 0
        try:
            procs = self._nvml.nvmlDeviceGetComputeRunningProcesses(self._handle)
            for proc in procs:
                if proc.pid in pids:
                    total_bytes += int(proc.usedGpuMemory)
        except Exception:
            return 0.0
        return total_bytes / (1024.0 * 1024.0)

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            mem = self._sample()
            if mem > self._peak_mb:
                self._peak_mb = mem
            if mem < self._valley_mb:
                self._valley_mb = mem
            time.sleep(self.interval_sec)

    def start(self) -> None:
        self._peak_mb = 0.0
        self._valley_mb = float("inf")
        self._stop_event.clear()
        if not self._available:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> Tuple[float, float, float]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        peak = float(self._peak_mb)
        valley = 0.0 if self._valley_mb == float("inf") else float(self._valley_mb)
        diff = peak - valley
        return peak, valley, diff

    def close(self) -> None:
        if self._available and self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass


def collect_image_metadata(image_path: Union[str, Path]) -> Tuple[Tuple[int, int, int], Tuple[float, float, float]]:
    """读取影像 size 与 spacing，二维数据会自动补 z=1。"""
    src = _to_path(image_path)

    if _is_nifti_path(src):
        try:
            import nibabel as nib

            img = nib.load(str(src))
            shape = tuple(int(v) for v in img.shape)
            zooms = tuple(float(v) for v in img.header.get_zooms())

            if len(shape) == 2:
                return (shape[0], shape[1], 1), (zooms[0], zooms[1], 1.0)
            if len(shape) >= 3:
                spacing_z = zooms[2] if len(zooms) >= 3 else 1.0
                return (shape[0], shape[1], shape[2]), (zooms[0], zooms[1], float(spacing_z))
        except Exception:
            pass

    img = _read_image_with_sitk(src)
    size_raw = tuple(int(v) for v in img.GetSize())
    spacing_raw = tuple(float(v) for v in img.GetSpacing())

    if len(size_raw) == 2:
        size = (size_raw[0], size_raw[1], 1)
        spacing = (spacing_raw[0], spacing_raw[1], 1.0)
    else:
        size = (size_raw[0], size_raw[1], size_raw[2])
        spacing = (spacing_raw[0], spacing_raw[1], spacing_raw[2])
    return size, spacing


def _release_predictor(predictor) -> None:
    """彻底释放 nnUNetPredictor 占用的所有 CPU 和 GPU 内存。

    nnUNet 的 PlansManager / ConfigurationManager 在实例方法上使用了 @lru_cache，
    cache 的 key 包含 self，形成循环引用，导致 Python GC 无法及时回收；
    predict_single_npy_array 也不会清理 compute_gaussian 的模块级 GPU 缓存。
    此函数按顺序执行所有必要的清理步骤。
    """
    import gc

    # 1. 清理 compute_gaussian 模块级 GPU 张量缓存
    try:
        from nnunetv2.inference.sliding_window_prediction import compute_gaussian
        compute_gaussian.cache_clear()
    except Exception:
        pass

    # 2. 清理 ConfigurationManager 上的 @lru_cache（打破循环引用）
    cm = getattr(predictor, "configuration_manager", None)
    if cm is not None:
        for attr_name in ("preprocessor_class", "resampling_fn_data",
                          "resampling_fn_probabilities", "resampling_fn_seg"):
            prop = getattr(type(cm), attr_name, None)
            if prop is not None:
                fget = getattr(prop, "fget", None)
                if fget is not None and hasattr(fget, "cache_clear"):
                    fget.cache_clear()

    # 3. 清理 PlansManager 上的 @lru_cache（打破循环引用）
    pm = getattr(predictor, "plans_manager", None)
    if pm is not None:
        if hasattr(pm, "get_configuration") and hasattr(pm.get_configuration, "cache_clear"):
            pm.get_configuration.cache_clear()
        for attr_name in ("image_reader_writer_class", "experiment_planner_class",
                          "label_manager_class"):
            prop = getattr(type(pm), attr_name, None)
            if prop is not None:
                fget = getattr(prop, "fget", None)
                if fget is not None and hasattr(fget, "cache_clear"):
                    fget.cache_clear()

    # 4. 将网络权重从 GPU 移到 CPU，再删除，避免 GPU 内存残留
    net = getattr(predictor, "network", None)
    if net is not None:
        try:
            net.cpu()
        except Exception:
            pass
        predictor.network = None

    # 5. 释放 list_of_parameters（每个 fold 的完整权重副本，占大量 CPU 内存）
    if getattr(predictor, "list_of_parameters", None) is not None:
        predictor.list_of_parameters.clear()
        predictor.list_of_parameters = None

    # 6. 清空其余引用
    predictor.plans_manager = None
    predictor.configuration_manager = None
    predictor.dataset_json = None
    predictor.label_manager = None

    # 7. 强制 GC + 清理 CUDA 缓存
    del predictor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _light_inference_cleanup() -> None:
    """轻量清理：用于每个 case 推理后，抑制缓存累积导致的内存基线抬升。"""
    try:
        from nnunetv2.inference.sliding_window_prediction import compute_gaussian
        compute_gaussian.cache_clear()
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _log_memory_snapshot(stage: str, device: Optional[torch.device] = None) -> None:
    """打印当前进程内存与 CUDA 显存快照，便于观察是否存在持续增长。"""
    rss_mb: Optional[float] = None
    try:
        import psutil

        rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        pass

    use_cuda = (device is not None and getattr(device, "type", "") == "cuda" and torch.cuda.is_available())
    if use_cuda:
        try:
            allocated_mb = torch.cuda.memory_allocated(device) / (1024.0 * 1024.0)
            reserved_mb = torch.cuda.memory_reserved(device) / (1024.0 * 1024.0)
            if rss_mb is not None:
                print(
                    f"[内存快照] {stage}: RSS={rss_mb:.1f} MB, "
                    f"CUDA allocated={allocated_mb:.1f} MB, reserved={reserved_mb:.1f} MB"
                )
            else:
                print(
                    f"[内存快照] {stage}: "
                    f"CUDA allocated={allocated_mb:.1f} MB, reserved={reserved_mb:.1f} MB"
                )
        except Exception:
            if rss_mb is not None:
                print(f"[内存快照] {stage}: RSS={rss_mb:.1f} MB")
    else:
        if rss_mb is not None:
            print(f"[内存快照] {stage}: RSS={rss_mb:.1f} MB")


def _get_available_folds_without_loading_weights(
    model_folder: Union[str, Path],
    checkpoint_name: str = "checkpoint_final.pth",
) -> List[int]:
    """通过扫描 fold 目录获取可用 folds，不加载任何 checkpoint 权重。"""
    mf = _to_path(model_folder)
    fold_dirs = sorted(
        [d for d in mf.iterdir() if d.is_dir() and d.name.startswith("fold_") and d.name != "fold_all"],
        key=lambda d: d.name,
    )
    available_folds: List[int] = []
    for fd in fold_dirs:
        ckpt_path = fd / checkpoint_name
        if ckpt_path.exists():
            fold_id_str = fd.name.split("_")[-1]
            available_folds.append(int(fold_id_str))
    if not available_folds:
        raise FileNotFoundError(f"模型目录中未找到可用的 fold checkpoint: {mf}")
    return available_folds


def _parse_model_folder_triplet(model_folder: Union[str, Path]) -> Tuple[str, str, str]:
    """从标准目录名 Trainer__Plans__Configuration 解析三元组。"""
    folder_name = _to_path(model_folder).name
    parts = folder_name.split("__")
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    return "nnUNetTrainer", "nnUNetPlans", "3d_fullres"


def _run_single_model_prediction_worker(
    model_folder: str,
    converted_input_paths: List[str],
    output_dir: str,
    disable_tta: bool,
    use_cpu: bool,
    timing_json_path: str,
) -> None:
    """子进程：单模型批量推理。

    目的：通过“模型级进程隔离”强制释放 CPU 内存，避免主进程 RSS 随模型数量持续抬升。
    """
    import json
    import traceback

    try:
        device = torch.device("cpu" if use_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))

        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json

        t_load0 = time.time()
        folds = tuple(_get_available_folds_without_loading_weights(model_folder))

        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=not disable_tta,
            perform_everything_on_device=True,
            device=device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True,
        )
        predictor.initialize_from_trained_model_folder(
            model_training_output_dir=str(model_folder),
            use_folds=folds,
            checkpoint_name="checkpoint_final.pth",
        )

        rw_cls = determine_reader_writer_from_dataset_json(
            predictor.dataset_json, verbose=False,
        )
        reader_writer = rw_cls()
        file_ending = predictor.dataset_json.get("file_ending", ".nii.gz")
        t_load = time.time() - t_load0

        infer_times: List[float] = []
        out_dir = _to_path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for case_idx, converted_input in enumerate(converted_input_paths, start=1):
            case_id = f"case_{case_idx:05d}"
            truncated_output = str(out_dir / case_id)

            image_data, image_properties = reader_writer.read_images([str(converted_input)])
            t0 = time.time()
            predictor.predict_single_npy_array(
                input_image=image_data,
                image_properties=image_properties,
                output_file_truncated=truncated_output,
                save_or_return_probabilities=False,
            )
            infer_times.append(time.time() - t0)
            del image_data, image_properties
            _light_inference_cleanup()

        _release_predictor(predictor)
        del reader_writer, folds

        payload = {
            "file_ending": file_ending,
            "model_load_sec": float(t_load),
            "infer_times_sec": infer_times,
        }
        with _to_path(timing_json_path).open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    except Exception as exc:
        err_payload = {
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        try:
            with _to_path(timing_json_path).open("w", encoding="utf-8") as f:
                json.dump(err_payload, f, ensure_ascii=False)
        except Exception:
            pass
        raise


def _resolve_model_folder(
    dataset_id: Union[int, str],
    trainer: str,
    plans: str,
    configuration: str,
    model_folder: Optional[Union[str, Path]] = None,
) -> str:
    if model_folder is not None:
        folder = str(_to_path(model_folder))
        if not Path(folder).exists():
            raise FileNotFoundError(f"模型目录不存在: {folder}")
        return folder

    from nnunetv2.utilities.file_path_utilities import get_output_folder

    return get_output_folder(str(dataset_id), trainer, plans, configuration)


def _prepare_paths(
    dataset_folder_name: str,
    input_subdir: str,
    output_subdir: str,
    input_path: Optional[Union[str, Path]],
    output_path: Optional[Union[str, Path]],
) -> Tuple[Path, Path, Path]:
    nnunet_raw = os.environ.get("nnUNet_raw")
    if nnunet_raw:
        dataset_dir = Path(nnunet_raw) / dataset_folder_name
    else:
        dataset_dir = Path.cwd() / dataset_folder_name
    default_input = dataset_dir / input_subdir
    default_output = dataset_dir / output_subdir

    real_input = _to_path(input_path) if input_path is not None else default_input
    real_output = _to_path(output_path) if output_path is not None else default_output
    return dataset_dir, real_input, real_output


def _export_prediction_to_target(
    pred_nifti_path: Path,
    target_path: Path,
    reference_image_path: Path,
) -> None:
    """将 nnUNet 输出 NIfTI 转换/保存为目标格式，并复制输入图像空间信息。"""
    _ensure_parent_dir(target_path)
    suffix = _suffix_lower(target_path)

    # 若目标也是 NIfTI，优先避免使用 SITK 读取预测结果，绕开非正交方向余弦导致的读取失败。
    if suffix in {".nii.gz", ".nii"}:
        try:
            _repair_nifti_header_with_nibabel(pred_nifti_path, target_path)
        except Exception as exc:
            print(f"[警告] NIfTI header 重写失败，改为直接复制预测结果: {pred_nifti_path} ({exc})")
            shutil.copy2(pred_nifti_path, target_path)
        return

    import SimpleITK as sitk

    pred_img = sitk.ReadImage(str(pred_nifti_path))
    try:
        ref_img = sitk.ReadImage(str(reference_image_path))
        pred_img.CopyInformation(ref_img)
    except Exception as exc:
        print(f"[警告] 读取参考图像失败，跳过 CopyInformation: {reference_image_path} ({exc})")

    if suffix in {".mha", ".mhd", ".nrrd"}:
        sitk.WriteImage(pred_img, str(target_path))
        return

    raise ValueError(f"不支持的输出格式: {target_path}")


def stage_predict(
    dataset_id: Union[int, str, Sequence[Union[int, str]]],
    configuration: str,
    fold: Union[int, Sequence[int]],
    trainer: str,
    plans: str,
    dataset_folder_name: str,
    input_subdir: str = "imagesTs",
    output_subdir: str = "labelsTs_predicted",
    input_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    model_folder: Optional[Union[str, Path]] = None,
    disable_tta: bool = True,
    device: Optional[torch.device] = None,
    enable_stats: bool = False,
    stats_output_file: Optional[Union[str, Path]] = None,
    gpu_device_id: int = 0,
    monitor_interval_sec: float = 0.1,
    num_processes_preprocessing: int = 3,
    num_processes_segmentation_export: int = 3,
    gpu_id: Optional[int] = None,
) -> None:
    """
    模型推理预测（支持文件/文件夹输入、格式转换、显式模型目录、可选统计）。
    
    使用已训练好的 nnUNet 模型对医学影像进行分割预测。
    主要流程：
      1. 参数标准化（设备、fold、路径）
      2. 构建 nnUNetPredictor 并加载模型权重
      3. 逐个病例：格式转换 → 调用 nnUNet 推理 → 输出写回目标路径
      4. 可选：记录每个病例的耗时和显存消耗，并汇总导出 CSV

    关于不同分辨率的处理（见下方正文注释）：
      nnUNet 的 preprocessing pipeline 会根据训练时保存的 plans.json，
      自动将每张输入图像重采样到训练时使用的 target_spacing，
      推理完成后再将预测结果反向重采样回原始分辨率/尺寸，
      因此待预测图像的分辨率各不相同完全不影响结果正确性。

    兼容旧接口：
    - 未传 input_path/output_path 时，仍使用 nnUNet_raw/<dataset_folder_name>/<input_subdir|output_subdir>
    """
    # ------------------------------------------------------------------ #
    # 步骤 1：基础参数标准化
    # ------------------------------------------------------------------ #

    # ── 指定 GPU ────────────────────────────────────────────
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 若未指定设备，优先使用 GPU（CUDA），否则回落到 CPU
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset_id 可能传入列表（兼容多模型集成接口），此处仅支持单模型，强制取单值
    ds_id = _normalize_single_dataset_id(dataset_id)

    # fold 统一转为 tuple，nnUNetPredictor 支持同时使用多个 fold 的权重做集成预测
    if isinstance(fold, (list, tuple)):
        folds = tuple(int(f) for f in fold)
    else:
        folds = (int(fold),)

    # 延迟导入：避免 multiprocessing worker 进程重新执行脚本时触发重量级 DLL 加载，
    # 将 import 放在函数内部可将其限制在主进程中执行
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    # ------------------------------------------------------------------ #
    # 步骤 2：解析输入/输出路径 & 模型目录
    # ------------------------------------------------------------------ #

    # _prepare_paths 根据环境变量 nnUNet_raw 和参数确定
    # dataset_dir（数据集根目录）、real_input（实际输入目录/文件）、real_output（实际输出目录/文件）
    dataset_dir, real_input, real_output = _prepare_paths(
        dataset_folder_name=dataset_folder_name,
        input_subdir=input_subdir,
        output_subdir=output_subdir,
        input_path=input_path,
        output_path=output_path,
    )

    # _resolve_model_folder 优先使用显式传入的 model_folder，
    # 否则根据 dataset_id/trainer/plans/configuration 自动拼接 nnUNet_results 下的标准路径
    model_dir = _resolve_model_folder(
        dataset_id=ds_id,
        trainer=trainer,
        plans=plans,
        configuration=configuration,
        model_folder=model_folder,
    )

    # 递归扫描 real_input，收集所有支持格式（.nii.gz/.nii/.mhd/.mha/.nrrd）的文件列表
    input_files = parse_input_paths(real_input)
    # 若输入是目录，则保留相对路径结构映射到输出目录；若输入是单文件则直接映射
    input_root = real_input if real_input.is_dir() else None
    io_mapping = build_io_mapping(input_files, real_output, input_root=input_root)

    print(f"\n{'=' * 72}")
    print("阶段 3 / 预测")
    print(f"  dataset_id          : {ds_id}")
    print(f"  configuration       : {configuration}")
    print(f"  fold(s)             : {folds}")
    print(f"  model_folder        : {model_dir}")
    print(f"  dataset_dir         : {dataset_dir}")
    print(f"  input_path          : {real_input}")
    print(f"  output_path         : {real_output}")
    print(f"  case_count          : {len(input_files)}")
    print(f"  enable_stats        : {enable_stats}")
    print(f"{'=' * 72}")

    # ------------------------------------------------------------------ #
    # 步骤 3：构建 nnUNetPredictor 并加载模型
    # ------------------------------------------------------------------ #

    predictor = nnUNetPredictor(
        tile_step_size=0.5,       # 滑窗推理时相邻 patch 之间的步长比例（0.5 = 50% 重叠），重叠越大精度越高但越慢
        use_gaussian=True,        # 对重叠区域的预测结果使用高斯权重加权融合，减少拼接边界伪影
        use_mirroring=not disable_tta,  # TTA（测试时增强）：对输入做镜像翻转后预测再平均，提升精度但耗时翻倍
        perform_everything_on_device=True,  # 将预处理/后处理也放到 GPU 上执行，加速整体流程
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,          # 显示进度条
    )

    # 从模型目录加载 plans.json、dataset.json 及各 fold 的 checkpoint_final.pth
    # plans.json 中保存了训练时的 target_spacing、patch_size、归一化方式等关键配置，
    # 推理时的预处理和后处理步骤均依赖这份配置，這也是不同分辨率图像能被正确处理的关键所在
    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=model_dir,
        use_folds=folds,
        checkpoint_name="checkpoint_final.pth",
    )

    # ------------------------------------------------------------------ #
    # 步骤 4：初始化统计工具（可选）
    # ------------------------------------------------------------------ #

    # TimeStats 用于汇总所有病例的预测时间和显存消耗
    time_stats = TimeStats()
    # GPUMemoryMonitor 启动后台线程，以 monitor_interval_sec 的频率采样当前进程的显存用量，
    # 记录峰值（peak）和低谷（valley），差值（diff）反映单次推理的显存占用波动；
    # 若未检测到 NVIDIA GPU / pynvml，自动降级为全零记录
    gpu_monitor = GPUMemoryMonitor(gpu_device_id=gpu_device_id, interval_sec=monitor_interval_sec) if enable_stats else None
    if enable_stats and gpu_monitor is not None and not gpu_monitor.available:
        print("[提示] 未检测到可用 NVML/GPU，显存统计将自动降级为 0，仅记录时间。")

    # ------------------------------------------------------------------ #
    # 步骤 5：逐病例推理循环
    # ------------------------------------------------------------------ #

    # 使用系统临时目录隔离每次运行的中间文件，函数结束后自动清理，避免残留脏数据
    with tempfile.TemporaryDirectory(prefix="nnunet_predict_cases_") as temp_root:
        temp_root_path = Path(temp_root)
        temp_inputs = temp_root_path / "inputs"   # 存放转换为 nnUNet 格式后的输入文件
        temp_outputs = temp_root_path / "outputs" # 存放 nnUNet 推理生成的原始预测 NIfTI
        temp_inputs.mkdir(parents=True, exist_ok=True)
        temp_outputs.mkdir(parents=True, exist_ok=True)

        for idx, src in enumerate(input_files, start=1):
            # --- 5.1 格式转换 ---
            # nnUNet 要求输入为 <case_id>_0000.nii.gz（通道索引后缀 _0000 表示第一通道）
            # convert_to_nnunet_format 负责：
            #   - .nii.gz 直接改名/复制
            #   - .nii 用 nibabel 重写 header 后转为 .nii.gz
            #   - .mhd/.mha/.nrrd 等用 SimpleITK 读取后保存为 .nii.gz
            case_id = f"case_{idx:05d}"
            converted_input = convert_to_nnunet_format(src, temp_inputs, case_id)

            # truncated_output 是 nnUNet 的"截断路径"约定：
            # predict_from_files 会在该路径后自动追加 .nii.gz 生成最终预测文件
            truncated_output = temp_outputs / case_id
            pred_nifti = temp_outputs / f"{case_id}.nii.gz"  # 推理后实际生成的预测文件路径
            target_output = io_mapping[src]                   # 最终要写入的目标路径（用户指定的输出目录下）

            # --- 5.2 读取图像元信息（仅用于日志和统计，不影响推理） ---
            size, spacing = collect_image_metadata(src)
            start_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

            print(f"\n[{idx}/{len(input_files)}] 预测: {src.name}")
            print(f"  输入尺寸(size)       : {list(size)}")
            print(f"  输入间距(spacing)    : {[round(s, 6) for s in spacing]}")
            print(f"  目标输出             : {target_output}")

            # --- 5.3 启动显存监控 & 计时 ---
            if gpu_monitor is not None:
                gpu_monitor.start()
            t0 = time.time()

            # --- 5.4 核心推理调用 ---
            # predict_from_files 内部完整流程：
            #   a) 读取 converted_input（已是标准 .nii.gz）
            #   b) 预处理：按 plans.json 中的 target_spacing 将图像重采样到训练时的固定间距，
            #              再做强度归一化（CT 一般用 clip + z-score）
            #              ★ 这一步使得不同分辨率的输入图像都被统一到一致的特征空间
            #   c) 滑窗推理：将重采样后的图像分成若干 patch（大小由 patch_size 决定），
            #              逐 patch 送入网络前向传播，重叠区域用高斯权重融合
            #   d) 后处理：将概率图 argmax 得到 label mask，
            #              再反向重采样回原始图像的分辨率和尺寸
            #              ★ 最终输出与原始输入图像的 size/spacing 完全一致
            #   e) 保存预测结果为 <truncated_output>.nii.gz（即 pred_nifti）
            predictor.predict_from_files(
                list_of_lists_or_source_folder=[[str(converted_input)]],
                output_folder_or_list_of_truncated_output_files=[str(truncated_output)],
                save_probabilities=False,   # 不保存各类别的概率图，节省磁盘空间
                overwrite=True,             # 若已存在预测结果则覆盖
                num_processes_preprocessing=num_processes_preprocessing,           # 预处理并发进程数
                num_processes_segmentation_export=num_processes_segmentation_export,  # 后处理/保存并发进程数
            )

            pred_time = time.time() - t0
            # 停止显存监控并获取本次推理的峰值/低谷/波动（MB）
            if gpu_monitor is not None:
                peak, valley, diff = gpu_monitor.stop()
            else:
                peak, valley, diff = 0.0, 0.0, 0.0

            # --- 5.5 校验推理输出是否生成 ---
            if not pred_nifti.exists():
                raise FileNotFoundError(f"预测输出未生成: {pred_nifti}")

            # --- 5.6 将临时目录中的预测结果写回目标路径 ---
            # _export_prediction_to_target 负责：
            #   - 目标为 NIfTI：用 nibabel 重写 header（同步 qform/sform），避免部分软件读取报错
            #   - 目标为 .mha/.mhd/.nrrd：用 SITK 读取并 CopyInformation（复制原始图像的空间信息）后保存
            _export_prediction_to_target(
                pred_nifti_path=pred_nifti,
                target_path=target_output,
                reference_image_path=src,  # 参考原始输入，用于在非 NIfTI 输出时对齐空间信息
            )

            print(f"  预测耗时(s)          : {pred_time:.3f}")
            if enable_stats:
                print(f"  显存统计(MB)         : peak={peak:.2f}, valley={valley:.2f}, diff={diff:.2f}")

            if enable_stats:
                record = CaseStats(
                    dataset=str(dataset_folder_name),
                    case_file=str(src),
                    prediction_time_sec=float(pred_time),
                    gpu_mem_peak_mb=float(peak),
                    gpu_mem_valley_mb=float(valley),
                    gpu_mem_diff_mb=float(diff),
                    image_size_x=size[0],
                    image_size_y=size[1],
                    image_size_z=size[2],
                    spacing_x=spacing[0],
                    spacing_y=spacing[1],
                    spacing_z=spacing[2],
                    timestamp=start_ts,
                )
                time_stats.add(record)

    if enable_stats:
        time_stats.print_summary()
        if stats_output_file is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = real_output if real_output.is_dir() else real_output.parent
            stats_output_file = base / f"prediction_stats_{ts}.csv"
        csv_path = time_stats.export_csv(stats_output_file)
        print(f"[统计] CSV 已保存: {csv_path}")

        if gpu_monitor is not None:
            gpu_monitor.close()

    print("\n预测完成。")


def _read_model_meta(model_folder: Union[str, Path], checkpoint_name: str = "checkpoint_final.pth") -> dict:
    """
    从已训练好的 nnUNet 模型目录中读取与模型绑定的元信息。

    返回 dict 包含以下字段：
      - dataset_name        : str   数据集目录名（如 "Dataset304_appendicular_bones_ext_1559subj"）
      - dataset_id          : int   数据集 ID（如 304）
      - trainer             : str   训练器类名（如 "nnUNetTrainerNoMirroring"）
      - plans               : str   plans 标识名（如 "nnUNetPlans"）
      - configuration       : str   配置名（如 "3d_fullres"）
      - available_folds     : list  模型目录中可用的 fold 列表（如 [0]）
      - dataset_folder_name : str   等同于 dataset_name，用于兼容 stage_predict 的参数
    """
    import json

    mf = _to_path(model_folder)
    if not mf.exists():
        raise FileNotFoundError(f"模型目录不存在: {mf}")

    # --- 从 plans.json 读取 plans 标识名 ---
    plans_json_path = mf / "plans.json"
    if not plans_json_path.exists():
        raise FileNotFoundError(f"模型目录中缺少 plans.json: {plans_json_path}")
    with plans_json_path.open("r", encoding="utf-8") as f:
        plans_data = json.load(f)
    plans_identifier = plans_data.get("plans_name", "nnUNetPlans")

    # --- 轻量解析 trainer/plans/configuration，不加载 checkpoint 权重 ---
    trainer_name, plans_from_folder, configuration_name = _parse_model_folder_triplet(mf)
    if plans_identifier == "nnUNetPlans" and plans_from_folder:
        plans_identifier = plans_from_folder

    # --- 扫描 fold 目录，仅检查 checkpoint 文件存在性 ---
    available_folds = _get_available_folds_without_loading_weights(
        mf, checkpoint_name=checkpoint_name
    )

    # --- 从路径结构推断 dataset_name 和 dataset_id ---
    # nnUNet 标准目录结构: .../nnUNet_results/DatasetXXX_Name/Trainer__Plans__Config/
    # mf 通常指向 Trainer__Plans__Config 这一层
    dataset_dir_name = mf.parent.name  # 如 "Dataset304_appendicular_bones_ext_1559subj"
    try:
        # 格式: DatasetNNN_xxx
        dataset_id = int(dataset_dir_name.split("_")[0].replace("Dataset", ""))
    except (ValueError, IndexError):
        dataset_id = -1  # 无法解析时用 -1 标记

    return {
        "dataset_name": dataset_dir_name,
        "dataset_id": dataset_id,
        "trainer": trainer_name,
        "plans": plans_identifier,
        "configuration": configuration_name,
        "available_folds": available_folds,
        "dataset_folder_name": dataset_dir_name,
    }


def easy_predict(
    model_folder: Union[str, Path],
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    disable_tta: bool = True,
    use_cpu: bool = False,
    enable_stats: bool = False,
    stats_output_file: Optional[Union[str, Path]] = None,
    gpu_device_id: int = 0,
    monitor_interval_sec: float = 0.1,
    num_processes_preprocessing: int = 3,
    num_processes_segmentation_export: int = 3,
) -> None:
    """
    简化版预测接口：用户只需提供 model_folder、input_path、output_path。

    与模型绑定的参数（dataset_id、configuration、fold、trainer、plans、
    dataset_folder_name）将自动从模型目录中的 plans.json、dataset.json
    以及 checkpoint 文件中读取，无需用户手动指定，避免因参数不一致导致
    预测结果错误或报错。

    跨平台性能优化
    ----------
    - **Linux**: 使用 nnUNet 原生的 predict_from_files 批量推理路径，
      利用 fork 模式的多进程并行加速预处理和后处理导出。
    - **Windows**: 自动切换为单线程推理路径（predict_single_npy_array），
      避免 spawn 模式创建子进程时重复导入 torch/nnunet 带来的巨大开销
      （每病例可节省 10-15 秒）。

    参数
    ----------
    model_folder : str | Path
        已训练好的 nnUNet 模型目录路径，例如:
        "/data1/.../nnUNet_results/Dataset304_xxx/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres"
    input_path : str | Path
        待预测的影像文件或目录，支持 .nii.gz / .nii / .mhd / .mha / .nrrd
    output_path : str | Path
        预测结果输出路径（文件或目录）
    disable_tta : bool
        是否禁用测试时增强（TTA），默认 True（禁用，速度更快）
    use_cpu : bool
        是否强制使用 CPU 推理，默认 False（自动使用 GPU）
    enable_stats : bool
        是否记录每个病例的预测耗时和显存统计，默认 False
    stats_output_file : str | Path | None
        统计 CSV 输出路径，None 则自动在 output_path 下生成
    gpu_device_id : int
        用于显存监控的 GPU 设备 ID，默认 0
    monitor_interval_sec : float
        显存采样间隔（秒），默认 0.1
    num_processes_preprocessing : int
        预处理并发进程数，默认 3（仅 Linux 生效；Windows 自动忽略）
    num_processes_segmentation_export : int
        后处理/保存并发进程数，默认 3（仅 Linux 生效；Windows 自动忽略）

    示例
    ----------
    >>> easy_predict(
    ...     model_folder="/data1/.../nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres",
    ...     input_path="/data1/User/patient_images",
    ...     output_path="/data1/User/patient_predictions",
    ...     enable_stats=True,
    ... )
    """
    meta = _read_model_meta(model_folder)

    # ------------------------------------------------------------------ #
    # 步骤 1：打印模型元信息
    # ------------------------------------------------------------------ #
    print(f"\n{'=' * 72}")
    print("模型元信息（自动从模型目录读取）")
    print(f"  model_folder        : {model_folder}")
    print(f"  dataset_name        : {meta['dataset_name']}")
    print(f"  dataset_id          : {meta['dataset_id']}")
    print(f"  trainer             : {meta['trainer']}")
    print(f"  plans               : {meta['plans']}")
    print(f"  configuration       : {meta['configuration']}")
    print(f"  available_folds     : {meta['available_folds']}")
    print(f"{'=' * 72}")

    # ------------------------------------------------------------------ #
    # 步骤 2：确定设备
    # ------------------------------------------------------------------ #
    if use_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------ #
    # 步骤 3：构建 nnUNetPredictor 并加载模型
    # ------------------------------------------------------------------ #
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=not disable_tta,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )

    mf = str(_to_path(model_folder))
    folds = tuple(meta["available_folds"])
    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=mf,
        use_folds=folds,
        checkpoint_name="checkpoint_final.pth",
    )

    # ------------------------------------------------------------------ #
    # 步骤 4：解析输入输出路径
    # ------------------------------------------------------------------ #
    real_input = _to_path(input_path)
    real_output = _to_path(output_path)
    input_files = parse_input_paths(real_input)
    input_root = real_input if real_input.is_dir() else None
    io_mapping = build_io_mapping(input_files, real_output, input_root=input_root)

    # ------------------------------------------------------------------ #
    # 步骤 5：根据操作系统选择推理策略
    # ------------------------------------------------------------------ #
    is_windows = platform.system() == "Windows"
    strategy = "单线程（Windows 优化）" if is_windows else f"多进程（Linux, prep={num_processes_preprocessing}, export={num_processes_segmentation_export}）"

    print(f"\n{'=' * 72}")
    print("预测")
    print(f"  推理策略            : {strategy}")
    print(f"  device              : {device}")
    print(f"  input_path          : {real_input}")
    print(f"  output_path         : {real_output}")
    print(f"  case_count          : {len(input_files)}")
    print(f"  enable_stats        : {enable_stats}")
    print(f"  TTA (mirroring)     : {not disable_tta}")
    print(f"{'=' * 72}")

    # ------------------------------------------------------------------ #
    # 步骤 6：初始化统计工具（可选）
    # ------------------------------------------------------------------ #
    time_stats = TimeStats()
    gpu_monitor = (
        GPUMemoryMonitor(gpu_device_id=gpu_device_id, interval_sec=monitor_interval_sec)
        if enable_stats
        else None
    )
    if enable_stats and gpu_monitor is not None and not gpu_monitor.available:
        print("[提示] 未检测到可用 NVML/GPU，显存统计将自动降级为 0，仅记录时间。")

    # ------------------------------------------------------------------ #
    # 步骤 7：推理
    # ------------------------------------------------------------------ #
    if is_windows:
        _easy_predict_singlethread(
            predictor=predictor,
            input_files=input_files,
            io_mapping=io_mapping,
            meta=meta,
            enable_stats=enable_stats,
            time_stats=time_stats,
            gpu_monitor=gpu_monitor,
        )
    else:
        _easy_predict_multiprocess(
            predictor=predictor,
            input_files=input_files,
            io_mapping=io_mapping,
            meta=meta,
            enable_stats=enable_stats,
            time_stats=time_stats,
            gpu_monitor=gpu_monitor,
            num_processes_preprocessing=num_processes_preprocessing,
            num_processes_segmentation_export=num_processes_segmentation_export,
        )

    # ------------------------------------------------------------------ #
    # 步骤 8：统计汇总
    # ------------------------------------------------------------------ #
    if enable_stats:
        time_stats.print_summary()
        if stats_output_file is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = real_output if real_output.is_dir() else real_output.parent
            stats_output_file = base / f"prediction_stats_{ts}.csv"
        csv_path = time_stats.export_csv(stats_output_file)
        print(f"[统计] CSV 已保存: {csv_path}")

        if gpu_monitor is not None:
            gpu_monitor.close()

    print("\n预测完成。")


def _easy_predict_singlethread(
    predictor,
    input_files: List[Path],
    io_mapping: Dict[Path, Path],
    meta: dict,
    enable_stats: bool,
    time_stats: TimeStats,
    gpu_monitor: Optional[GPUMemoryMonitor],
) -> None:
    """
    Windows 优化路径：使用 predict_single_npy_array 在主进程中完成全部
    预处理→GPU推理→后处理→保存，零子进程创建开销。
    """
    from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json

    reader_writer_cls = determine_reader_writer_from_dataset_json(
        predictor.dataset_json, verbose=False,
    )
    reader_writer = reader_writer_cls()

    with tempfile.TemporaryDirectory(prefix="nnunet_easy_predict_") as temp_root:
        temp_root_path = Path(temp_root)
        temp_outputs = temp_root_path / "outputs"
        temp_outputs.mkdir(parents=True, exist_ok=True)

        for idx, src in enumerate(input_files, start=1):
            target_output = io_mapping[src]

            size, spacing = collect_image_metadata(src)
            start_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

            print(f"\n[{idx}/{len(input_files)}] 预测: {src.name}")
            print(f"  输入尺寸(size)       : {list(size)}")
            print(f"  输入间距(spacing)    : {[round(s, 6) for s in spacing]}")
            print(f"  目标输出             : {target_output}")

            case_id = f"case_{idx:05d}"
            converted_input = convert_to_nnunet_format(src, temp_root_path / "inputs", case_id)

            # 使用 nnUNet 的 IO 读取器，保证轴序与训练时一致
            image_data, image_properties = reader_writer.read_images([str(converted_input)])

            truncated_output = str(temp_outputs / case_id)
            pred_nifti = temp_outputs / f"{case_id}{predictor.dataset_json.get('file_ending', '.nii.gz')}"

            if gpu_monitor is not None:
                gpu_monitor.start()
            t0 = time.time()

            # predict_single_npy_array: 全在主进程中执行，不创建任何子进程
            predictor.predict_single_npy_array(
                input_image=image_data,
                image_properties=image_properties,
                output_file_truncated=truncated_output,
                save_or_return_probabilities=False,
            )

            pred_time = time.time() - t0

            if gpu_monitor is not None:
                peak, valley, diff = gpu_monitor.stop()
            else:
                peak, valley, diff = 0.0, 0.0, 0.0

            if not pred_nifti.exists():
                raise FileNotFoundError(f"预测输出未生成: {pred_nifti}")

            _export_prediction_to_target(
                pred_nifti_path=pred_nifti,
                target_path=target_output,
                reference_image_path=src,
            )

            print(f"  预测耗时(s)          : {pred_time:.3f}")
            if enable_stats:
                print(f"  显存统计(MB)         : peak={peak:.2f}, valley={valley:.2f}, diff={diff:.2f}")

            if enable_stats:
                record = CaseStats(
                    dataset=str(meta["dataset_folder_name"]),
                    case_file=str(src),
                    prediction_time_sec=float(pred_time),
                    gpu_mem_peak_mb=float(peak),
                    gpu_mem_valley_mb=float(valley),
                    gpu_mem_diff_mb=float(diff),
                    image_size_x=size[0],
                    image_size_y=size[1],
                    image_size_z=size[2],
                    spacing_x=spacing[0],
                    spacing_y=spacing[1],
                    spacing_z=spacing[2],
                    timestamp=start_ts,
                )
                time_stats.add(record)


def _easy_predict_multiprocess(
    predictor,
    input_files: List[Path],
    io_mapping: Dict[Path, Path],
    meta: dict,
    enable_stats: bool,
    time_stats: TimeStats,
    gpu_monitor: Optional[GPUMemoryMonitor],
    num_processes_preprocessing: int = 3,
    num_processes_segmentation_export: int = 3,
) -> None:
    """
    Linux 路径：使用 nnUNet 原生的 predict_from_files 批量推理，
    利用 fork 模式多进程并行预处理和后处理导出，吞吐量更高。
    """
    with tempfile.TemporaryDirectory(prefix="nnunet_easy_predict_") as temp_root:
        temp_root_path = Path(temp_root)
        temp_inputs = temp_root_path / "inputs"
        temp_outputs = temp_root_path / "outputs"
        temp_inputs.mkdir(parents=True, exist_ok=True)
        temp_outputs.mkdir(parents=True, exist_ok=True)

        # 先批量转换所有输入为 nnUNet 格式
        converted_list: List[List[str]] = []
        truncated_list: List[str] = []
        src_list: List[Path] = []

        for idx, src in enumerate(input_files, start=1):
            case_id = f"case_{idx:05d}"
            converted_input = convert_to_nnunet_format(src, temp_inputs, case_id)
            converted_list.append([str(converted_input)])
            truncated_list.append(str(temp_outputs / case_id))
            src_list.append(src)

            size, spacing = collect_image_metadata(src)
            print(f"\n[{idx}/{len(input_files)}] 准备: {src.name}")
            print(f"  输入尺寸(size)       : {list(size)}")
            print(f"  输入间距(spacing)    : {[round(s, 6) for s in spacing]}")

        # 整批推理（nnUNet 内部自行管理多进程预处理和导出）
        if gpu_monitor is not None:
            gpu_monitor.start()
        t0 = time.time()

        predictor.predict_from_files(
            list_of_lists_or_source_folder=converted_list,
            output_folder_or_list_of_truncated_output_files=truncated_list,
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=num_processes_preprocessing,
            num_processes_segmentation_export=num_processes_segmentation_export,
        )

        total_pred_time = time.time() - t0

        if gpu_monitor is not None:
            peak, valley, diff = gpu_monitor.stop()
        else:
            peak, valley, diff = 0.0, 0.0, 0.0

        avg_time = total_pred_time / len(input_files) if input_files else 0.0
        print(f"\n  批量推理总耗时(s)    : {total_pred_time:.3f}")
        print(f"  平均每病例(s)        : {avg_time:.3f}")

        # 将预测结果导出到目标路径，并收集统计信息
        file_ending = predictor.dataset_json.get("file_ending", ".nii.gz")
        for idx, src in enumerate(src_list):
            target_output = io_mapping[src]
            case_id = f"case_{idx + 1:05d}"
            pred_nifti = temp_outputs / f"{case_id}{file_ending}"

            if not pred_nifti.exists():
                raise FileNotFoundError(f"预测输出未生成: {pred_nifti}")

            _export_prediction_to_target(
                pred_nifti_path=pred_nifti,
                target_path=target_output,
                reference_image_path=src,
            )
            print(f"  已导出: {target_output}")

            if enable_stats:
                size, spacing = collect_image_metadata(src)
                record = CaseStats(
                    dataset=str(meta["dataset_folder_name"]),
                    case_file=str(src),
                    prediction_time_sec=float(avg_time),
                    gpu_mem_peak_mb=float(peak),
                    gpu_mem_valley_mb=float(valley),
                    gpu_mem_diff_mb=float(diff),
                    image_size_x=size[0],
                    image_size_y=size[1],
                    image_size_z=size[2],
                    spacing_x=spacing[0],
                    spacing_y=spacing[1],
                    spacing_z=spacing[2],
                    timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
                )
                time_stats.add(record)


def read_target_spacing_from_model(
    model_folder: Union[str, Path],
    configuration: Optional[str] = None,
) -> List[float]:
    """从模型目录的 plans.json 中读取训练时使用的 target_spacing。

    Parameters
    ----------
    model_folder : str | Path
        已训练好的 nnUNet 模型目录路径
    configuration : str | None
        配置名（如 "3d_fullres"）。如果为 None 则自动从模型元信息中读取。

    Returns
    -------
    List[float]
        训练时使用的 target spacing，例如 [3.0, 3.0, 3.0]
    """
    import json

    mf = _to_path(model_folder)
    plans_json_path = mf / "plans.json"
    if not plans_json_path.exists():
        raise FileNotFoundError(f"模型目录中缺少 plans.json: {plans_json_path}")

    with plans_json_path.open("r", encoding="utf-8") as f:
        plans_data = json.load(f)

    if configuration is None:
        meta = _read_model_meta(model_folder)
        configuration = meta["configuration"]

    configs = plans_data.get("configurations", {})
    if configuration not in configs:
        raise KeyError(f"plans.json 中未找到配置 '{configuration}'，可用: {list(configs.keys())}")

    spacing = configs[configuration].get("spacing")
    if spacing is None:
        raise KeyError(f"配置 '{configuration}' 中未找到 spacing 字段")

    return list(spacing)


def easy_predict_with_preresample(
    model_folder: Union[str, Path],
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    target_spacing: Optional[List[float]] = None,
    resample_threshold: float = 0.3,
    mask_resample_mode: str = "torch_gpu",
    disable_tta: bool = True,
    use_cpu: bool = False,
    enable_stats: bool = False,
    stats_output_file: Optional[Union[str, Path]] = None,
    gpu_device_id: int = 0,
    monitor_interval_sec: float = 0.1,
) -> None:
    """
    带预插值的快速预测接口 — 逐病例流水线，含每例完整计时。

    对每个病例依次执行：
      1. 预插值（image → target_spacing）
      2. nnUNet 推理
      3. 后插值（mask → 原始分辨率）
    并统计每例的总耗时（含预插值 + 推理 + 后插值）。

    Parameters
    ----------
    model_folder : str | Path
        已训练好的 nnUNet 模型目录路径
    input_path : str | Path
        待预测的影像文件或目录
    output_path : str | Path
        预测结果输出路径（文件或目录）
    target_spacing : list[float] | None
        目标分辨率。None 则自动从 plans.json 读取。
    resample_threshold : float
        spacing 差异容差（mm），默认 0.3
    mask_resample_mode : str
        mask 上采样模式：
        - "torch_gpu"（默认）: PyTorch GPU 三线性插值，又快又光滑（< 1s）
        - "nearest" : 最近邻，极快但边界有锯齿
        - "smooth"  : CPU One-Hot + 线性插值 + ArgMax，光滑但较慢
    disable_tta : bool
        是否禁用 TTA，默认 True
    use_cpu : bool
        是否强制 CPU，默认 False
    enable_stats : bool
        是否记录统计，默认 False
    stats_output_file : str | Path | None
        统计 CSV 路径
    gpu_device_id : int
        GPU 设备 ID
    monitor_interval_sec : float
        显存采样间隔（秒）
    """
    import numpy as np
    import ResampleImageAndMask

    # ------------------------------------------------------------------ #
    # 1. 读取模型元信息 & target spacing
    # ------------------------------------------------------------------ #
    meta = _read_model_meta(model_folder)
    if target_spacing is None:
        target_spacing = read_target_spacing_from_model(model_folder, meta["configuration"])
    target_spacing_arr = np.array(target_spacing, dtype=np.float64)

    # ------------------------------------------------------------------ #
    # 2. 确定设备 & 构建 predictor（只初始化一次）
    # ------------------------------------------------------------------ #
    if use_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=not disable_tta,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    mf = str(_to_path(model_folder))
    folds = tuple(meta["available_folds"])
    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=mf,
        use_folds=folds,
        checkpoint_name="checkpoint_final.pth",
    )

    reader_writer_cls = determine_reader_writer_from_dataset_json(
        predictor.dataset_json, verbose=False,
    )
    reader_writer = reader_writer_cls()
    file_ending = predictor.dataset_json.get("file_ending", ".nii.gz")

    # ------------------------------------------------------------------ #
    # 3. 扫描输入文件 & 构建 IO 映射
    # ------------------------------------------------------------------ #
    real_input = _to_path(input_path)
    real_output = _to_path(output_path)
    input_files = parse_input_paths(real_input)
    input_root = real_input if real_input.is_dir() else None
    io_mapping = build_io_mapping(input_files, real_output, input_root=input_root)

    print(f"\n{'=' * 72}")
    print("预插值加速预测（逐病例）")
    print(f"  model_folder        : {model_folder}")
    print(f"  target_spacing      : {target_spacing}")
    print(f"  mask_resample_mode  : {mask_resample_mode}")
    print(f"  device              : {device}")
    print(f"  case_count          : {len(input_files)}")
    print(f"{'=' * 72}")

    # ------------------------------------------------------------------ #
    # 4. 统计工具
    # ------------------------------------------------------------------ #
    time_stats = TimeStats()
    gpu_monitor = (
        GPUMemoryMonitor(gpu_device_id=gpu_device_id, interval_sec=monitor_interval_sec)
        if enable_stats else None
    )
    if enable_stats and gpu_monitor is not None and not gpu_monitor.available:
        print("[提示] 未检测到可用 NVML/GPU，显存统计将自动降级为 0。")

    # ------------------------------------------------------------------ #
    # 5. 逐病例流水线
    # ------------------------------------------------------------------ #
    with tempfile.TemporaryDirectory(prefix="nnunet_preresample_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        tmp_inputs = tmp_root / "inputs"
        tmp_outputs = tmp_root / "outputs"
        tmp_inputs.mkdir(parents=True, exist_ok=True)
        tmp_outputs.mkdir(parents=True, exist_ok=True)

        for idx, src in enumerate(input_files, start=1):
            target_output = io_mapping[src]
            _ensure_parent_dir(target_output)
            orig_size, orig_spacing = collect_image_metadata(src)

            # 判断是否需要预插值
            sp_sorted = np.sort(np.array(orig_spacing[:3], dtype=np.float64))
            ts_sorted = np.sort(target_spacing_arr)
            needs_resample = not np.all(np.abs(sp_sorted - ts_sorted) < resample_threshold)

            print(f"\n[{idx}/{len(input_files)}] {src.name}")
            print(f"  原始 spacing        : {[round(s, 4) for s in orig_spacing]}")
            print(f"  需要预插值          : {'是' if needs_resample else '否'}")

            t_case_start = time.time()
            start_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

            if gpu_monitor is not None:
                gpu_monitor.start()

            # --- 5a. 预插值 ---
            case_id = f"case_{idx:05d}"
            if needs_resample:
                resampled_path = tmp_inputs / src.name
                t0 = time.time()
                ResampleImageAndMask.resample_and_save_image_fast(
                    src_path=str(src),
                    dst_path=str(resampled_path),
                    target_spacing=target_spacing,
                    image_dtype=np.float32,
                )
                t_pre = time.time() - t0
                # 用下采样后的图像作为输入
                converted_input = convert_to_nnunet_format(
                    resampled_path, tmp_inputs / "converted", case_id,
                )
            else:
                t_pre = 0.0
                converted_input = convert_to_nnunet_format(
                    src, tmp_inputs / "converted", case_id,
                )

            # --- 5b. 推理 ---
            image_data, image_properties = reader_writer.read_images([str(converted_input)])
            truncated_output = str(tmp_outputs / case_id)
            pred_nifti = tmp_outputs / f"{case_id}{file_ending}"

            t0 = time.time()
            predictor.predict_single_npy_array(
                input_image=image_data,
                image_properties=image_properties,
                output_file_truncated=truncated_output,
                save_or_return_probabilities=False,
            )
            t_pred = time.time() - t0

            if not pred_nifti.exists():
                raise FileNotFoundError(f"预测输出未生成: {pred_nifti}")

            # --- 5c. 后插值 ---
            if needs_resample:
                # 推理完成后释放 nnUNet 留在显存中的未使用缓存，
                # 防止模型占用显存 + mask插值占用显存同时达到峰値导致 OOM
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                t0 = time.time()
                if mask_resample_mode == "torch_gpu":
                    ResampleImageAndMask.resample_mask_to_original_torch(
                        lowres_mask_path=str(pred_nifti),
                        ref_path=str(src),
                        output_path=str(target_output),
                        device=str(device),
                    )
                elif mask_resample_mode == "nearest":
                    ResampleImageAndMask.resample_mask_to_original_nearest(
                        lowres_mask_path=str(pred_nifti),
                        ref_path=str(src),
                        output_path=str(target_output),
                    )
                else:  # smooth
                    ResampleImageAndMask.resample_multilabel_mask_to_original(
                        lowres_mask_path=str(pred_nifti),
                        ref_path=str(src),
                        output_path=str(target_output),
                        verbose=False,
                    )
                t_post = time.time() - t0
            else:
                t_post = 0.0
                _export_prediction_to_target(
                    pred_nifti_path=pred_nifti,
                    target_path=target_output,
                    reference_image_path=src,
                )

            t_case_total = time.time() - t_case_start

            if gpu_monitor is not None:
                peak, valley, diff = gpu_monitor.stop()
            else:
                peak, valley, diff = 0.0, 0.0, 0.0

            # --- 5d. 打印本例统计 ---
            print(f"  预插值(s)           : {t_pre:.3f}")
            print(f"  推理(s)             : {t_pred:.3f}")
            print(f"  后插值(s)           : {t_post:.3f}")
            print(f"  本例总耗时(s)       : {t_case_total:.3f}")
            if enable_stats:
                print(f"  显存(MB)            : peak={peak:.2f}, valley={valley:.2f}, diff={diff:.2f}")

            if enable_stats:
                record = CaseStats(
                    dataset=str(meta["dataset_folder_name"]),
                    case_file=str(src),
                    prediction_time_sec=float(t_case_total),
                    gpu_mem_peak_mb=float(peak),
                    gpu_mem_valley_mb=float(valley),
                    gpu_mem_diff_mb=float(diff),
                    image_size_x=orig_size[0],
                    image_size_y=orig_size[1],
                    image_size_z=orig_size[2],
                    spacing_x=orig_spacing[0],
                    spacing_y=orig_spacing[1],
                    spacing_z=orig_spacing[2],
                    timestamp=start_ts,
                    preresample_time_sec=float(t_pre),
                    inference_time_sec=float(t_pred),
                    postresample_time_sec=float(t_post),
                )
                time_stats.add(record)

            # 清理本例临时文件，避免磁盘占用累积
            if pred_nifti.exists():
                pred_nifti.unlink()
            if needs_resample:
                resampled_path = tmp_inputs / src.name
                if resampled_path.exists():
                    resampled_path.unlink()

    # ------------------------------------------------------------------ #
    # 6. 汇总
    # ------------------------------------------------------------------ #
    if enable_stats:
        time_stats.print_summary()
        if stats_output_file is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = real_output if real_output.is_dir() else real_output.parent
            stats_output_file = base / f"prediction_stats_{ts}.csv"
        csv_path = time_stats.export_csv(stats_output_file)
        print(f"[统计] CSV 已保存: {csv_path}")
        if gpu_monitor is not None:
            gpu_monitor.close()

    print("\n预测完成。")


def multimodel_predict_and_merge(
    model_folders: List[Union[str, Path]],
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    class_maps: List[dict],
    *,
    combine_map: Optional[dict] = None,
    model_names: Optional[List[str]] = None,
    target_spacing: Optional[List[float]] = None,
    resample_threshold: float = 0.3,
    mask_resample_mode: str = "torch_gpu",
    disable_tta: bool = True,
    use_cpu: bool = False,
    stats_output_file: Optional[Union[str, Path]] = None,
) -> None:
    """
    多模型共享分辨率的优化预测接口 — 逐例处理，内存友好版本。

    处理流程（逐例）：
      对每例图像 → 预插值 1 次 → 逐模型加载/推理/释放 →
      低分辨率上拼接 → 后插值 1 次 → 保存 → 释放内存 → 下一例。

    同一时刻仅 1 例图像 + 1 个模型驻留内存，适合数百例图像批量处理。

    Parameters
    ----------
    model_folders : list[str | Path]
        各部位模型的目录路径列表。
    input_path : str | Path
        待预测的影像目录（或单文件）。
    output_path : str | Path
        输出合并 mask 的目录（或单文件）。
    class_maps : list[dict]
        每个模型的标签映射字典（从 ModelMap.toml 读取），
        key 为组织名，value 为该模型 mask 中该组织的 label 整数值。
        列表长度必须与 model_folders 一一对应。
    combine_map : dict | None
        拼接目标标签映射（如 ModelMap.toml 中的 MR_Combine / CT_Combine）。
        key 为组织名，value 为拼接后 mask 中该组织的最终 label 值。
        当与 class_maps 同时提供时，映射流程为：
          模型 mask label 值 → class_maps[i] 反查组织名 → combine_map 查最终 label。
        当 combine_map 为 None 时，退化为旧逻辑（直接用 class_maps 中的 value）。
    model_names : list[str] | None
        可选的模型名称标签，用于日志打印。None 则自动从路径推断。
    target_spacing : list[float] | None
        共享的目标分辨率。None 时自动从第一个模型的 plans.json 读取。
    resample_threshold : float
        spacing 差异容差（mm），低于此值认为无需预插值，默认 0.3。
    mask_resample_mode : str
        合并 mask 上采样模式：
        - "torch_gpu"（默认）: PyTorch GPU 三线性 One-Hot 插值
        - "nearest" : 最近邻，极快但边界锯齿
        - "smooth"  : CPU One-Hot + 线性插值 + ArgMax，光滑但较慢
    disable_tta : bool
        是否禁用 TTA，默认 True。
    use_cpu : bool
        是否强制 CPU 推理，默认 False。
    stats_output_file : str | Path | None
        统计 CSV 输出路径。None 则自动在 output_path 目录下生成带时间戳的文件。
    """
    import gc
    import json
    import numpy as np
    import nibabel as nib
    import ResampleImageAndMask

    if len(model_folders) != len(class_maps):
        raise ValueError(
            f"model_folders ({len(model_folders)}) 与 class_maps ({len(class_maps)}) 长度不一致"
        )
    if model_names is None:
        model_names = [Path(p).parent.name for p in model_folders]

    # ------------------------------------------------------------------ #
    # 1. 读取共享 target_spacing
    # ------------------------------------------------------------------ #
    _, _, configuration0 = _parse_model_folder_triplet(model_folders[0])
    if target_spacing is None:
        target_spacing = read_target_spacing_from_model(
            model_folders[0], configuration0
        )
    target_spacing_arr = np.array(target_spacing, dtype=np.float64)

    # ------------------------------------------------------------------ #
    # 2. 确定设备
    # ------------------------------------------------------------------ #
    if use_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    del configuration0

    # ------------------------------------------------------------------ #
    # 3. 构建每个模型的 LUT（label 重映射查找表）
    #    当 combine_map 提供时：local_label → class_map 反查组织名 → combine_map 查最终值
    #    当 combine_map 为 None 时：直接用 class_map 中的 value（旧逻辑）
    # ------------------------------------------------------------------ #
    all_local_to_target: list[dict[int, int]] = []

    for label_dict in class_maps:
        if combine_map is not None:
            # 反转 label_dict：value → 第一个 key（同值取第一个）
            value_to_name: dict[int, str] = {}
            for tissue_name, local_val in label_dict.items():
                local_val = int(local_val)
                if local_val not in value_to_name:
                    value_to_name[local_val] = tissue_name
            # 通过组织名在 combine_map 中查最终 label
            local_to_target: dict[int, int] = {}
            for local_val, tissue_name in value_to_name.items():
                if tissue_name in combine_map:
                    local_to_target[local_val] = int(combine_map[tissue_name])
                else:
                    print(f"[警告] 组织 '{tissue_name}'（local_label={local_val}）"
                          f"在 combine_map 中未找到，已跳过。")
            all_local_to_target.append(local_to_target)
        else:
            # 旧逻辑：label=i 直接映射到 class_map 中第 i 个 value
            ordered_values = [int(v) for v in label_dict.values()]
            local_to_target = {i + 1: v for i, v in enumerate(ordered_values)}
            all_local_to_target.append(local_to_target)

    max_target = 0
    for mapping in all_local_to_target:
        if mapping:
            max_target = max(max_target, max(mapping.values()))
    merge_dtype = np.uint8 if max_target <= 255 else np.uint16

    luts = []
    for mapping in all_local_to_target:
        if not mapping:
            luts.append(np.zeros(1, dtype=merge_dtype))
            continue
        lut_size = max(mapping.keys()) + 1
        lut = np.zeros(lut_size, dtype=merge_dtype)
        for local_label, target_value in mapping.items():
            lut[local_label] = target_value
        luts.append(lut)

    # ------------------------------------------------------------------ #
    # 4. 扫描输入 & 构建 IO 映射
    # ------------------------------------------------------------------ #
    real_input = _to_path(input_path)
    real_output = _to_path(output_path)
    input_files = parse_input_paths(real_input)
    input_root = real_input if real_input.is_dir() else None
    io_mapping = build_io_mapping(input_files, real_output, input_root=input_root)

    print(f"\n{'=' * 72}")
    print("多模型共享分辨率优化预测（逐例处理模式）")
    print(f"  target_spacing      : {target_spacing}")
    print(f"  model_count         : {len(model_folders)}")
    print(f"  mask_resample_mode  : {mask_resample_mode}")
    print(f"  device              : {device}")
    print(f"  case_count          : {len(input_files)}")
    print(f"{'=' * 72}")

    # ------------------------------------------------------------------ #
    # 5. 逐例处理：预插值 → 逐模型推理 → 拼接 → 后插值 → 释放
    #    同一时刻仅 1 例图像 + 1 个模型驻留内存
    # ------------------------------------------------------------------ #
    case_timings: List[dict] = []
    case_metas: List[dict] = []
    needs_resample_flags: List[bool] = []
    # 累计每个模型的加载时间（取各 case 中该模型加载时间的最大值作为代表）
    model_load_times: List[float] = [0.0] * len(model_folders)

    total_t0 = time.time()

    for case_idx, src in enumerate(input_files):
        case_timing = {
            "preresample_sec": 0.0,
            "postresample_sec": 0.0,
            "infer_per_model": [0.0] * len(model_folders),
            "load_per_model": [0.0] * len(model_folders),
        }
        case_t0 = time.time()

        print(f"\n{'=' * 72}")
        print(f"[病例 {case_idx + 1}/{len(input_files)}] {src.name}")
        print(f"{'=' * 72}")

        # -------------------------------------------------------------- #
        # A. 收集元数据 & 预插值（本例仅 1 次）
        # -------------------------------------------------------------- #
        orig_size, orig_spacing = collect_image_metadata(src)
        case_metas.append({"orig_size": orig_size, "orig_spacing": orig_spacing})

        sp_sorted = np.sort(np.array(orig_spacing[:3], dtype=np.float64))
        ts_sorted = np.sort(target_spacing_arr)
        needs_resample = not np.all(
            np.abs(sp_sorted - ts_sorted) < resample_threshold
        )
        needs_resample_flags.append(needs_resample)

        print(f"  原始 spacing        : {[round(s, 4) for s in orig_spacing]}")
        print(f"  需要预插值          : {'是' if needs_resample else '否'}")

        with tempfile.TemporaryDirectory(prefix=f"nnunet_case{case_idx + 1:05d}_") as case_tmp_dir:
            case_tmp_root = Path(case_tmp_dir)
            case_id = f"case_{case_idx + 1:05d}"

            if needs_resample:
                resampled_path = case_tmp_root / f"{case_id}_resampled.nii.gz"
                t0 = time.time()
                ResampleImageAndMask.resample_and_save_image_fast(
                    src_path=str(src),
                    dst_path=str(resampled_path),
                    target_spacing=target_spacing,
                    image_dtype=np.float32,
                )
                t_pre = time.time() - t0
                case_timing["preresample_sec"] = t_pre
                predict_input_path = resampled_path
                print(f"  预插值耗时(s)       : {t_pre:.3f}")
            else:
                predict_input_path = src

            converted_input = convert_to_nnunet_format(
                predict_input_path, case_tmp_root / "converted", case_id,
            )

            # -------------------------------------------------------------- #
            # B. 逐模型加载 → 推理本例 → 释放
            # -------------------------------------------------------------- #
            model_pred_dir = case_tmp_root / "predictions"
            model_pred_dir.mkdir(parents=True, exist_ok=True)
            model_file_endings: List[str] = []

            converted_input_strs = [str(converted_input)]

            for mi, mpath in enumerate(model_folders):
                print(f"  ── 模型 [{mi + 1}/{len(model_folders)}]: {model_names[mi]}")
                _log_memory_snapshot(f"  加载前-{model_names[mi]}", device)

                per_model_dir = model_pred_dir / f"model_{mi}"
                per_model_dir.mkdir(parents=True, exist_ok=True)
                timing_json_path = per_model_dir / "timing.json"

                worker = mp.Process(
                    target=_run_single_model_prediction_worker,
                    args=(
                        str(mpath),
                        converted_input_strs,
                        str(per_model_dir),
                        disable_tta,
                        use_cpu,
                        str(timing_json_path),
                    ),
                )
                worker.start()
                worker.join()

                if not timing_json_path.exists():
                    raise RuntimeError(
                        f"模型 {mi} ({model_names[mi]}) 子进程未生成 timing 文件: {timing_json_path}"
                    )

                with timing_json_path.open("r", encoding="utf-8") as f:
                    timing_payload = json.load(f)

                if worker.exitcode != 0 or "error" in timing_payload:
                    tb = timing_payload.get("traceback", "")
                    err = timing_payload.get("error", f"worker exit code={worker.exitcode}")
                    raise RuntimeError(
                        f"模型 {mi} ({model_names[mi]}) 子进程推理失败: {err}\n{tb}"
                    )

                file_ending = timing_payload.get("file_ending", ".nii.gz")
                model_file_endings.append(file_ending)

                t_load = float(timing_payload.get("model_load_sec", 0.0))
                case_timing["load_per_model"][mi] = t_load
                # 记录每个模型在所有 case 中的最大加载时间作为代表值
                model_load_times[mi] = max(model_load_times[mi], t_load)

                infer_times = timing_payload.get("infer_times_sec", [])
                if len(infer_times) != 1:
                    raise RuntimeError(
                        f"模型 {mi} ({model_names[mi]}) 推理计时数量异常: "
                        f"expect=1, got={len(infer_times)}"
                    )
                t_infer = float(infer_times[0])
                case_timing["infer_per_model"][mi] = t_infer

                # 验证预测输出文件存在
                # 注意 _run_single_model_prediction_worker 内部 case 编号从 1 开始
                pred_nifti_check = per_model_dir / f"case_00001{file_ending}"
                if not pred_nifti_check.exists():
                    raise FileNotFoundError(
                        f"模型 {mi} ({model_names[mi]}) 预测输出未生成: {pred_nifti_check}"
                    )

                print(
                    f"    加载(s): {t_load:.3f}, 推理(s): {t_infer:.3f}"
                )

                if timing_json_path.exists():
                    timing_json_path.unlink()

                _log_memory_snapshot(f"  卸载后-{model_names[mi]}", device)

            # 预插值的临时图像不再需要，释放磁盘空间
            if needs_resample:
                resampled_path_to_del = case_tmp_root / f"{case_id}_resampled.nii.gz"
                if resampled_path_to_del.exists():
                    resampled_path_to_del.unlink()
            converted_dir = case_tmp_root / "converted"
            if converted_dir.exists():
                shutil.rmtree(converted_dir, ignore_errors=True)

            # -------------------------------------------------------------- #
            # C. 合并各模型预测结果 & 后插值回原始分辨率
            # -------------------------------------------------------------- #
            target_output = io_mapping[src]
            _ensure_parent_dir(target_output)

            combined_mask = None
            combined_affine = None

            for mi in range(len(model_folders)):
                # worker 内部对单个 case 编号始终为 case_00001
                pred_nifti = (
                    model_pred_dir / f"model_{mi}" / f"case_00001{model_file_endings[mi]}"
                )
                pred_img = nib.load(str(pred_nifti))
                pred_data = np.asarray(pred_img.dataobj)
                if combined_mask is None:
                    combined_mask = np.zeros(pred_data.shape[:3], dtype=merge_dtype)
                    combined_affine = pred_img.affine.copy()

                lut = luts[mi]
                pred_clipped = np.clip(pred_data, 0, len(lut) - 1).astype(np.intp)
                remapped = lut[pred_clipped]
                nonzero = remapped > 0
                combined_mask[nonzero] = remapped[nonzero]

                del pred_img, pred_data, pred_clipped, remapped, nonzero
                if pred_nifti.exists():
                    pred_nifti.unlink()

            # 后插值（仅 1 次）
            if needs_resample:
                combined_nifti_path = case_tmp_root / f"combined_{case_id}.nii.gz"
                nib.save(
                    nib.Nifti1Image(combined_mask, combined_affine),
                    str(combined_nifti_path),
                )
                del combined_mask, combined_affine

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                t0 = time.time()
                if mask_resample_mode == "torch_gpu":
                    ResampleImageAndMask.resample_mask_to_original_torch(
                        lowres_mask_path=str(combined_nifti_path),
                        ref_path=str(src),
                        output_path=str(target_output),
                        device=str(device),
                    )
                elif mask_resample_mode == "nearest":
                    ResampleImageAndMask.resample_mask_to_original_nearest(
                        lowres_mask_path=str(combined_nifti_path),
                        ref_path=str(src),
                        output_path=str(target_output),
                    )
                else:   # smooth
                    ResampleImageAndMask.resample_multilabel_mask_to_original(
                        lowres_mask_path=str(combined_nifti_path),
                        ref_path=str(src),
                        output_path=str(target_output),
                        verbose=False,
                    )
                t_post = time.time() - t0
                case_timing["postresample_sec"] = t_post
            else:
                t_post = 0.0
                nib.save(
                    nib.Nifti1Image(combined_mask, combined_affine),
                    str(target_output),
                )
                del combined_mask, combined_affine

        # case_tmp_dir 上下文退出后，该例所有临时文件自动清理

        t_pre = case_timing["preresample_sec"]
        t_infer_total = sum(case_timing["infer_per_model"])
        t_case_total = time.time() - case_t0

        case_timings.append(case_timing)

        # 强制垃圾回收，确保本例内存完全释放后再处理下一例
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n  本例汇总: 预插值(s): {t_pre:.3f},"
              f" 推理(s): {t_infer_total:.3f},"
              f" 后插值(s): {t_post:.3f},"
              f" 总(s): {t_case_total:.3f}")
        _log_memory_snapshot(f"病例 {case_idx + 1} 完成后", device)

    # ------------------------------------------------------------------ #
    # 6. 构建统计记录 & 导出 CSV
    # ------------------------------------------------------------------ #
    stats_records: List[dict] = []
    model_load_total_sec = sum(model_load_times)
    for case_idx, src in enumerate(input_files):
        cm = case_metas[case_idx]
        ct = case_timings[case_idx]
        t_infer_total = sum(ct["infer_per_model"])
        t_case_total = ct["preresample_sec"] + t_infer_total + ct["postresample_sec"]
        record = {
            "case_file": str(src),
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "image_size_x": cm["orig_size"][0],
            "image_size_y": cm["orig_size"][1],
            "image_size_z": cm["orig_size"][2],
            "spacing_x": cm["orig_spacing"][0],
            "spacing_y": cm["orig_spacing"][1],
            "spacing_z": cm["orig_spacing"][2],
            "needs_resample": needs_resample_flags[case_idx],
            "preresample_sec": ct["preresample_sec"],
            "postresample_sec": ct["postresample_sec"],
            "model_load_total_sec": model_load_total_sec,
            "inference_total_sec": t_infer_total,
            "case_total_sec": t_case_total,
        }
        for mi_idx, mname in enumerate(model_names):
            record[f"infer_{mname}_sec"] = ct["infer_per_model"][mi_idx]
            record[f"load_{mname}_sec"] = ct["load_per_model"][mi_idx]
        stats_records.append(record)

    total_elapsed = time.time() - total_t0
    print(f"\n{'=' * 72}")
    print(f"多模型共享分辨率预测完成，共 {len(input_files)} 例，总耗时 {total_elapsed:.1f}s")
    print(f"{'=' * 72}")

    if stats_output_file is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = real_output if real_output.is_dir() else real_output.parent
        stats_output_file = base / f"multimodel_stats_{ts}.csv"
    csv_path = _to_path(stats_output_file)
    _ensure_parent_dir(csv_path)

    if stats_records:
        fieldnames = list(stats_records[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in stats_records:
                row = {}
                for k, v in rec.items():
                    if isinstance(v, float):
                        row[k] = f"{v:.4f}"
                    else:
                        row[k] = v
                writer.writerow(row)
        print(f"[统计] CSV 已保存: {csv_path}")

    # 打印汇总
    if stats_records:
        n = len(stats_records)
        total_pre  = sum(r["preresample_sec"] for r in stats_records)
        total_post = sum(r["postresample_sec"] for r in stats_records)
        total_inf  = sum(r["inference_total_sec"] for r in stats_records)
        total_all  = sum(r["case_total_sec"] for r in stats_records)
        total_load = model_load_total_sec
        print(f"\n{'=' * 60}")
        print("全局统计")
        print(f"  病例数              : {n}")
        print(f"  模型加载总计(s)     : {total_load:.2f}")
        if model_folders:
            print(f"  模型加载均值(s/模型): {total_load / len(model_folders):.2f}")
        print(f"  {'阶段':<16}  {'总计(s)':>10}  {'均值(s)':>10}")
        print(f"  {'-'*16}  {'-'*10}  {'-'*10}")
        for label, val in [
            ("预插值", total_pre),
            ("推理(全模型)", total_inf),
            ("后插值", total_post),
            ("本例总计", total_all),
        ]:
            print(f"  {label:<16}  {val:>10.2f}  {val/n:>10.2f}")
        # 每个模型的推理时间汇总
        for mname in model_names:
            key = f"infer_{mname}_sec"
            vals = [r.get(key, 0.0) for r in stats_records]
            print(f"  {'  ' + mname:<16}  {sum(vals):>10.2f}  {sum(vals)/n:>10.2f}")
        print(f"{'=' * 60}")


def main() -> None:
    # 独立运行时在这里硬编码参数，不使用 argparse。

    # ============================================================
    # 推荐方式：使用 easy_predict，只需 3 个必填参数
    # dataset_id / configuration / fold / trainer / plans 等模型绑定参数
    # 会自动从 model_folder 中的 JSON/checkpoint 文件读取，无需手动指定。
    # ============================================================
    easy_predict(
        model_folder="D:/data/totalsegmemtormrca/Dataset301_heart_highres_1559subj/nnUNetTrainer__nnUNetPlans__3d_fullres",
        input_path="D:/data/totalsegmemtormrca/test",
        output_path="D:/data/totalsegmemtormrca/test_predicted",
        enable_stats=True,
    )

    # ============================================================
    # 旧方式（保留兼容）：手动指定所有参数调用 stage_predict
    # ============================================================
    # run_cfg = {
    #     "dataset_id": DATASET_ID,
    #     "configuration": CONFIGURATION,
    #     "fold": FOLD,
    #     "trainer": TRAINER,
    #     "plans": PLANS,
    #     "dataset_folder_name": DATASET_FOLDER_NAME,
    #     "input_path": "/data1/User/shijian_ruan/extremities",
    #     "output_path": "/data1/User/shijian_ruan/extremities_totalseg",
    #     "model_folder": "/data1/.../nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres",
    #     "disable_tta": True,
    #     "use_cpu": False,
    #     "enable_stats": True,
    #     "stats_output_file": None,
    #     "gpu_device_id": 0,
    #     "monitor_interval_sec": 0.1,
    #     "num_processes_preprocessing": 3,
    #     "num_processes_segmentation_export": 3,
    # }
    # infer_device = torch.device("cpu") if bool(run_cfg.get("use_cpu", False)) else None
    # stage_predict(
    #     dataset_id=run_cfg["dataset_id"],
    #     configuration=run_cfg["configuration"],
    #     fold=run_cfg["fold"],
    #     trainer=run_cfg["trainer"],
    #     plans=run_cfg["plans"],
    #     dataset_folder_name=run_cfg["dataset_folder_name"],
    #     input_path=run_cfg.get("input_path"),
    #     output_path=run_cfg.get("output_path"),
    #     model_folder=run_cfg.get("model_folder"),
    #     disable_tta=bool(run_cfg.get("disable_tta", True)),
    #     device=infer_device,
    #     enable_stats=bool(run_cfg.get("enable_stats", False)),
    #     stats_output_file=run_cfg.get("stats_output_file"),
    #     gpu_device_id=int(run_cfg.get("gpu_device_id", 0)),
    #     monitor_interval_sec=float(run_cfg.get("monitor_interval_sec", 0.1)),
    #     num_processes_preprocessing=int(run_cfg.get("num_processes_preprocessing", 3)),
    #     num_processes_segmentation_export=int(run_cfg.get("num_processes_segmentation_export", 3)),
    # )


if __name__ == "__main__":
    main()

