'''
本文件负责图像数据的各种非标准转换
'''

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import SimpleITK as sitk


SUPPORTED_IMAGE_SUFFIXES = {
	".nii",
	".gz",
	".mha",
	".mhd",
	".nrrd",
	".img",
	".hdr",
}


def _normalize_vector(values: Sequence[float], expected_len: int, name: str) -> tuple[float, ...]:
	if len(values) != expected_len:
		raise ValueError(f"{name} 必须包含 {expected_len} 个数值。")

	normalized = tuple(float(v) for v in values)
	if any(not math.isfinite(v) for v in normalized):
		raise ValueError(f"{name} 中包含非有限值。")

	return normalized


def _normalize_voi_size(voi_size: Sequence[int]) -> list[int]:
	if len(voi_size) != 3:
		raise ValueError("voi_size 必须包含 3 个整数，分别对应 x、y、z 方向大小。")

	normalized = [int(v) for v in voi_size]
	if any(v <= 0 for v in normalized):
		raise ValueError("voi_size 中的所有尺寸都必须大于 0。")

	return normalized


def _normalize_voi_center(voi_center: Sequence[float] | None, dimension: int) -> tuple[float, ...] | None:
	if voi_center is None:
		return None

	if len(voi_center) != dimension:
		raise ValueError(f"voi_center 必须包含 {dimension} 个数值，分别对应各维中心坐标。")

	normalized = tuple(float(v) for v in voi_center)
	if any(not math.isfinite(v) for v in normalized):
		raise ValueError("voi_center 中包含非有限值。")

	return normalized


def _is_supported_image_file(file_path: Path) -> bool:
	suffixes = {suffix.lower() for suffix in file_path.suffixes}
	return bool(suffixes & SUPPORTED_IMAGE_SUFFIXES)


def _iter_image_files(input_dir: Path, recursive: bool = False) -> Iterable[Path]:
	pattern = "**/*" if recursive else "*"
	for file_path in sorted(input_dir.glob(pattern)):
		if file_path.is_file() and _is_supported_image_file(file_path):
			yield file_path


def _identity_direction(dimension: int) -> tuple[float, ...]:
	return tuple(1.0 if row == col else 0.0 for row in range(dimension) for col in range(dimension))


def _determinant(matrix_values: Sequence[float], dimension: int) -> float:
	if dimension == 2:
		a, b, c, d = matrix_values
		return a * d - b * c

	if dimension == 3:
		a, b, c, d, e, f, g, h, i = matrix_values
		return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

	raise ValueError(f"暂不支持 {dimension} 维方向矩阵行列式计算。")


def _sanitize_spacing(spacing: Sequence[float], dimension: int, fallback_spacing: Sequence[float]) -> tuple[float, ...]:
	fallback = _normalize_vector(fallback_spacing, dimension, "fallback_spacing")
	if len(spacing) != dimension:
		return fallback

	result = []
	for current, default in zip(spacing, fallback):
		value = float(current)
		if math.isfinite(value) and value > 0:
			result.append(value)
		else:
			result.append(default)
	return tuple(result)


def _sanitize_origin(origin: Sequence[float], dimension: int, fallback_origin: Sequence[float] | None = None) -> tuple[float, ...]:
	if fallback_origin is None:
		fallback = tuple(0.0 for _ in range(dimension))
	else:
		fallback = _normalize_vector(fallback_origin, dimension, "fallback_origin")

	if len(origin) != dimension:
		return fallback

	result = []
	for current, default in zip(origin, fallback):
		value = float(current)
		if math.isfinite(value):
			result.append(value)
		else:
			result.append(default)
	return tuple(result)


def _sanitize_direction(direction: Sequence[float], dimension: int) -> tuple[float, ...]:
	identity = _identity_direction(dimension)
	if len(direction) != dimension * dimension:
		return identity

	normalized = tuple(float(v) for v in direction)
	if any(not math.isfinite(v) for v in normalized):
		return identity

	try:
		det = _determinant(normalized, dimension)
	except ValueError:
		return identity

	if abs(det) < 1e-8:
		return identity

	return normalized


def _orthonormalize_nifti_affine(affine):
	"""确保 4×4 NIfTI affine 的方向余弦严格正交归一化。

	从 3×3 子矩阵中提取 spacing（列范数），归一化得到方向余弦矩阵，
	通过 SVD 寻找最近正交矩阵，再乘回 spacing 重建旋转/缩放部分。
	保留原始平移（origin）和底行不变。

	若 3×3 子矩阵退化（全零或含 NaN），则返回单位矩阵 + 原始原点。
	"""
	affine = np.array(affine, dtype=np.float64)
	R = affine[:3, :3]

	if np.any(np.isnan(R)) or np.all(np.abs(R) < 1e-12):
		# 退化情况：用单位方向、spacing=1
		result = np.eye(4)
		origin = affine[:3, 3]
		if not np.any(np.isnan(origin)):
			result[:3, 3] = origin
		return result

	spacing = np.linalg.norm(R, axis=0)
	spacing = np.where(spacing < 1e-12, 1.0, spacing)
	D = R / spacing

	U, _, Vt = np.linalg.svd(D)
	if np.linalg.det(U @ Vt) < 0:
		U[:, -1] *= -1

	result = affine.copy()
	result[:3, :3] = (U @ Vt) * spacing
	return result


def _read_nifti_with_fixed_affine(input_path: Path) -> sitk.Image:
	"""当 SimpleITK 因方向余弦不正交无法读取 NIfTI 时，
	用 nibabel 加载、正交化 affine 后写入临时文件再交给 SimpleITK 读取。"""
	import nibabel as nib

	print(f"  [修复] 用 nibabel 加载并正交化方向余弦…")
	nib_img = nib.load(str(input_path))
	fixed_affine = _orthonormalize_nifti_affine(np.array(nib_img.affine, dtype=np.float64))
	data = np.asarray(nib_img.dataobj)

	fixed_img = nib.Nifti1Image(data, fixed_affine)
	fixed_img.header.set_data_dtype(nib_img.header.get_data_dtype())

	import os
	tmp_fd, tmp_path = tempfile.mkstemp(suffix='.nii.gz')
	try:
		os.close(tmp_fd)
		nib.save(fixed_img, tmp_path)
		return sitk.ReadImage(tmp_path)
	finally:
		Path(tmp_path).unlink(missing_ok=True)


def repair_image_for_totalsegmentator(
	input_path: str | Path,
	output_path: str | Path,
	reference_path: str | Path | None = None,
	fallback_spacing: Sequence[float] = (1.0, 1.0, 1.0),
	fallback_origin: Sequence[float] = (0.0, 0.0, 0.0),
	use_compression: bool = True,
) -> Path:
	"""
	修复图像空间元数据并重新保存，使 TotalSegmentator 可以正常读取。

	重点处理：
	1. NIfTI affine 全 0 或 qform/sform 无法分解；
	2. spacing 为 0、NaN 或缺失；
	3. origin 或 direction 非法；
	4. 已有参考图像时，拷贝参考图像的空间信息。

	当原始 `.nii.gz` 的 affine 已经全部损坏时，真实空间信息无法仅靠该文件恢复。
	因此本函数支持传入 `reference_path`，将同病例原始 `.mhd`、`.nrrd`、`.nii.gz`
	等图像中的 spacing、origin、direction 复制到输出文件；如果没有参考图像，
	则补齐为合法默认值，至少保证输出文件头可被 nibabel/TotalSegmentator 正常解析。
	"""
	input_path = Path(input_path)
	output_path = _ensure_supported_output_path(Path(output_path), input_path)

	if not input_path.exists():
		raise FileNotFoundError(f"输入图像不存在: {input_path}")

	try:
		image = sitk.ReadImage(str(input_path))
	except RuntimeError as _e:
		print(f"[警告] SimpleITK 无法直接读取 {input_path}（{_e}），尝试修复方向余弦…")
		image = _read_nifti_with_fixed_affine(input_path)

	dimension = image.GetDimension()
	fallback_spacing = fallback_spacing[:dimension]
	fallback_origin = fallback_origin[:dimension]

	if reference_path is not None:
		reference_image = sitk.ReadImage(str(reference_path))
		if reference_image.GetDimension() != dimension:
			raise ValueError("参考图像与待修复图像维度不一致。")

		spacing = _sanitize_spacing(reference_image.GetSpacing(), dimension, fallback_spacing)
		origin = _sanitize_origin(reference_image.GetOrigin(), dimension, fallback_origin)
		direction = _sanitize_direction(reference_image.GetDirection(), dimension)
	else:
		spacing = _sanitize_spacing(image.GetSpacing(), dimension, fallback_spacing)
		origin = _sanitize_origin(image.GetOrigin(), dimension, fallback_origin)
		direction = _sanitize_direction(image.GetDirection(), dimension)

	fixed_image = sitk.Image(image)
	fixed_image.SetSpacing(spacing)
	fixed_image.SetOrigin(origin)
	fixed_image.SetDirection(direction)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	sitk.WriteImage(fixed_image, str(output_path), use_compression)
	return output_path


def repair_image_for_totalsegmentator_from_path(
	input_path: str | Path,
	output_dir: str | Path,
	reference_dir: str | Path | None = None,
	recursive: bool = False,
	fallback_spacing: Sequence[float] = (1.0, 1.0, 1.0),
	fallback_origin: Sequence[float] = (0.0, 0.0, 0.0),
) -> list[Path]:
	"""
	批量修复单个图像或目录中的图像空间元数据。

	如果提供 `reference_dir`：
	- 当 `input_path` 是文件时，`reference_dir` 可以是单个参考文件；
	- 当 `input_path` 是目录时，会按相对路径匹配参考文件。
	"""
	input_path = Path(input_path)
	output_dir = Path(output_dir)
	reference_dir_path = Path(reference_dir) if reference_dir is not None else None

	if not input_path.exists():
		raise FileNotFoundError(f"输入路径不存在: {input_path}")

	output_dir.mkdir(parents=True, exist_ok=True)

	if input_path.is_file():
		if not _is_supported_image_file(input_path):
			raise ValueError(f"不支持的图像文件格式: {input_path}")

		reference_file = reference_dir_path if reference_dir_path is not None else None
		target_path = output_dir / input_path.name
		return [
			repair_image_for_totalsegmentator(
				input_path=input_path,
				output_path=target_path,
				reference_path=reference_file,
				fallback_spacing=fallback_spacing,
				fallback_origin=fallback_origin,
			)
		]

	if not input_path.is_dir():
		raise ValueError(f"无法识别的输入路径类型: {input_path}")

	image_files = list(_iter_image_files(input_path, recursive=recursive))
	if not image_files:
		raise FileNotFoundError(f"在目录中未找到支持的图像文件: {input_path}")

	saved_files: list[Path] = []
	for image_path in image_files:
		relative_path = image_path.relative_to(input_path)
		target_path = output_dir / relative_path
		target_path.parent.mkdir(parents=True, exist_ok=True)

		reference_file = None
		if reference_dir_path is not None:
			candidate = reference_dir_path / relative_path
			if candidate.exists():
				reference_file = candidate

		saved_files.append(
			repair_image_for_totalsegmentator(
				input_path=image_path,
				output_path=target_path,
				reference_path=reference_file,
				fallback_spacing=fallback_spacing,
				fallback_origin=fallback_origin,
			)
		)

	return saved_files


def crop_center_voi(
	image: sitk.Image,
	voi_size: Sequence[int],
	pad_value: float = 0,
	voi_center: Sequence[float] | None = None,
) -> sitk.Image:
	"""
	从图像中心或指定中心坐标截取指定大小的 VOI。

	当输入图像尺寸小于目标 VOI 尺寸时，会先进行常数填充，再截取中心区域，
	以保证输出尺寸始终与 `voi_size` 一致。

	参数
	----
	image:
		输入图像。
	voi_size:
		目标 VOI 大小，格式为 `(x, y, z)`。
	pad_value:
		超出原图像范围时的填充值。
	voi_center:
		VOI 在原图像中的中心点坐标，按像素索引顺序填写，例如 `(300, 300, 50)`。
		若为 `None`，则默认使用原图像中心。
	"""
	target_size = _normalize_voi_size(voi_size)
	image_size = list(image.GetSize())
	dimension = image.GetDimension()
	center = _normalize_voi_center(voi_center, dimension)
	if center is None:
		center = tuple(current / 2.0 for current in image_size)

	start_index = [int(round(c - target / 2.0)) for c, target in zip(center, target_size)]
	end_index = [start + target for start, target in zip(start_index, target_size)]

	lower_pad = [max(-start, 0) for start in start_index]
	upper_pad = [max(end - current, 0) for end, current in zip(end_index, image_size)]

	if any(pad > 0 for pad in lower_pad + upper_pad):
		image = sitk.ConstantPad(image, lower_pad, upper_pad, pad_value)
		start_index = [start + pad for start, pad in zip(start_index, lower_pad)]

	return sitk.RegionOfInterest(image, size=target_size, index=start_index)


def crop_center_voi_in_directory(
	input_dir: str | Path,
	output_dir: str | Path,
	voi_size: Sequence[int],
	voi_center: Sequence[float] | None = None,
	recursive: bool = False,
	pad_value: float = 0,
) -> list[Path]:
	"""
	对目录中的图像批量截取中心 VOI，并保存到输出目录。

	参数
	----
	input_dir:
		输入图像目录。
	output_dir:
		输出目录。不存在时会自动创建。
	voi_size:
		目标 VOI 大小，格式为 `(x, y, z)`。
	voi_center:
		目标 VOI 在原图像中的中心点坐标，格式为 `(x, y, z)`。
		为 `None` 时默认使用每幅图像自身中心。
	recursive:
		是否递归处理子目录，默认 `False`。
	pad_value:
		当图像尺寸不足时的填充值，默认 `0`。

	返回
	----
	list[Path]
		成功保存的输出文件路径列表。
	"""
	input_dir = Path(input_dir)
	output_dir = Path(output_dir)

	if not input_dir.exists():
		raise FileNotFoundError(f"输入目录不存在: {input_dir}")
	if not input_dir.is_dir():
		raise NotADirectoryError(f"输入路径不是目录: {input_dir}")

	_normalize_voi_size(voi_size)
	output_dir.mkdir(parents=True, exist_ok=True)

	saved_files: list[Path] = []
	image_files = list(_iter_image_files(input_dir, recursive=recursive))
	if not image_files:
		raise FileNotFoundError(f"在目录中未找到支持的图像文件: {input_dir}")

	for image_path in image_files:
		relative_path = image_path.relative_to(input_dir)
		target_path = output_dir / relative_path
		target_path.parent.mkdir(parents=True, exist_ok=True)

		image = sitk.ReadImage(str(image_path))
		cropped_image = crop_center_voi(
			image,
			voi_size=voi_size,
			pad_value=pad_value,
			voi_center=voi_center,
		)
		sitk.WriteImage(cropped_image, str(target_path))
		saved_files.append(target_path)

	return saved_files


def crop_center_voi_from_path(
	input_path: str | Path,
	output_dir: str | Path,
	voi_size: Sequence[int],
	voi_center: Sequence[float] | None = None,
	recursive: bool = False,
	pad_value: float = 0,
) -> list[Path]:
	"""
	根据输入路径截取中心 VOI。

	- 如果 `input_path` 是单个文件，则只处理这一个文件。
	- 如果 `input_path` 是目录，则处理该目录下所有支持的图像文件。
	- `voi_center` 用于指定截取中心在原图像中的像素坐标。
	"""
	input_path = Path(input_path)
	output_dir = Path(output_dir)

	if not input_path.exists():
		raise FileNotFoundError(f"输入路径不存在: {input_path}")

	if input_path.is_file():
		if not _is_supported_image_file(input_path):
			raise ValueError(f"不支持的图像文件格式: {input_path}")

		_normalize_voi_size(voi_size)
		output_dir.mkdir(parents=True, exist_ok=True)

		target_path = output_dir / input_path.name
		image = sitk.ReadImage(str(input_path))
		cropped_image = crop_center_voi(
			image,
			voi_size=voi_size,
			pad_value=pad_value,
			voi_center=voi_center,
		)
		sitk.WriteImage(cropped_image, str(target_path))
		return [target_path]

	if input_path.is_dir():
		return crop_center_voi_in_directory(
			input_dir=input_path,
			output_dir=output_dir,
			voi_size=voi_size,
			voi_center=voi_center,
			recursive=recursive,
			pad_value=pad_value,
		)

	raise ValueError(f"无法识别的输入路径类型: {input_path}")


def repair_image_example() -> None:
	# 直接在这里手动填写参数
	input_path = Path(r"E:\Linux\Totalsegmentator_dataset_v201_s0295.nii.gz")
	output_path = Path(r"E:\Linux\Totalsegmentator_dataset_v201_s0295_mask.nii.gz")
	reference_path = None  # 如果有原始正确头信息的图像，可填写其路径

	fixed_path = repair_image_for_totalsegmentator(
		input_path=input_path,
		output_path=output_path,
		reference_path=reference_path,
		fallback_spacing=(1.0, 1.0, 1.0),
		fallback_origin=(0.0, 0.0, 0.0),
	)

	print(f"修复完成，输出文件: {fixed_path}")


def crop_image() -> None:
	# 直接在这里手动填写参数
	input_path = Path(r"D:/data/liversegments/CTimage/image_fix.nii.gz")  # 可以是单个图像文件，也可以是图像目录
	output_dir = Path(r"D:/data/liversegments/CTimage/image_fix_crop")
	voi_size = (256, 256, 100)  # 按 (x, y, z) 顺序填写
	voi_center = (235, 235, 128)  # VOI 中心在原图像中的像素坐标；填 None 表示使用图像中心
	recursive = False
	pad_value = 0

	saved_files = crop_center_voi_from_path(
		input_path=input_path,
		output_dir=output_dir,
		voi_size=voi_size,
		voi_center=voi_center,
		recursive=recursive,
		pad_value=pad_value,
	)

	print(f"处理完成，共保存 {len(saved_files)} 个文件到: {output_dir}")
	for file_path in saved_files:
		print(file_path)







def _ensure_supported_output_path(output_path: Path, source_path: Path) -> Path:
	if output_path.suffixes:
		return output_path

	if source_path.name.endswith(".nii.gz"):
		return output_path.with_name(f"{output_path.name}.nii.gz")

	return output_path.with_suffix(source_path.suffix or ".nii.gz")


def _normalize_orientation_code(orientation: str, name: str = "orientation") -> str:
	orientation = orientation.strip().upper()
	if len(orientation) != 3:
		raise ValueError(f"{name} 必须是 3 位方向码，例如 'RAI' 或 'ASL'。")

	axis_to_group = {
		"R": "RL",
		"L": "RL",
		"A": "AP",
		"P": "AP",
		"S": "SI",
		"I": "SI",
	}
	groups: list[str] = []
	for index, axis_code in enumerate(orientation):
		if axis_code not in axis_to_group:
			raise ValueError(
				f"{name} 的第 {index + 1} 位非法: '{axis_code}'。"
				"方向码只能使用 R/L/A/P/S/I。"
			)
		groups.append(axis_to_group[axis_code])

	if len(set(groups)) != 3:
		raise ValueError(
			f"{name} 非法: '{orientation}'。"
			"必须且只能各包含一个 [R/L]、[A/P]、[S/I] 轴向字符。"
		)

	return orientation


_OPPOSITE_AXIS = {
	"R": "L", "L": "R",
	"A": "P", "P": "A",
	"S": "I", "I": "S",
}


def _anatomical_to_sitk_orientation(orientation: str) -> str:
	"""
	将解剖方向码（MHD AnatomicalOrientation 约定）转换为 SimpleITK DICOMOrient 约定。

	两种约定的区别：
	- MHD 约定：每个字母表示该轴 **低索引端（起始端）** 的方向。
	  例如 RAI 表示 x 轴从 R→L，y 轴从 A→P，z 轴从 I→S。
	- SimpleITK DICOMOrient 约定：每个字母表示该轴 **正方向（递增方向）**。
	  例如 LPS 表示 x 正方向为 L，y 正方向为 P，z 正方向为 S。

	因此 MHD 的 RAI 等价于 SimpleITK 的 LPS（每个字母取反）。
	"""
	return "".join(_OPPOSITE_AXIS[ch] for ch in orientation)


def reorient_image_or_mask(
	input_path: str | Path,
	output_path: str | Path,
	target_orientation: str = "ASL",
	source_orientation: str | None = None,
	use_compression: bool = True,
) -> Path:
	"""
	将 mhd/nii 图像（包括 image 或 mask）按方向码重排到目标方向。

	方向码采用 MHD AnatomicalOrientation 约定（每个字母表示该轴低索引端方向）。
	例如 RAI 表示 x 从 R→L、y 从 A→P、z 从 I→S。

	示例：把 RAI 方向重排到 ASL。

	参数
	----
	input_path:
		输入图像路径，仅支持 `.mhd`、`.nii`、`.nii.gz`。
	output_path:
		输出图像路径。若不带后缀，会自动沿用输入后缀。
	target_orientation:
		目标方向码（MHD 约定），默认 `ASL`。
	source_orientation:
		可选。若提供，会先把图像 direction 头信息改为该方向码，
		适用于已知原图真实方向但头信息不正确的情况。
	use_compression:
		写出时是否启用压缩。
	"""
	input_path = Path(input_path)
	output_path = _ensure_supported_output_path(Path(output_path), input_path)

	if not input_path.exists():
		raise FileNotFoundError(f"输入图像不存在: {input_path}")

	allowed_suffix_pairs = {
		(".mhd",),
		(".nii",),
		(".nii", ".gz"),
	}
	input_suffixes = tuple(s.lower() for s in input_path.suffixes)
	if input_suffixes not in allowed_suffix_pairs:
		raise ValueError(f"仅支持 .mhd/.nii/.nii.gz，当前文件为: {input_path}")

	target_orientation = _normalize_orientation_code(target_orientation, "target_orientation")
	if source_orientation is not None:
		source_orientation = _normalize_orientation_code(source_orientation, "source_orientation")

	# 将 MHD 约定的方向码转换为 SimpleITK DICOMOrient 约定（每个字母取反）
	sitk_target = _anatomical_to_sitk_orientation(target_orientation)

	image = sitk.ReadImage(str(input_path))
	if image.GetDimension() != 3:
		raise ValueError("当前函数仅支持 3D 图像方向重排。")

	if source_orientation is not None:
		sitk_source = _anatomical_to_sitk_orientation(source_orientation)
		image = sitk.DICOMOrient(image, sitk_source)

	reoriented = sitk.DICOMOrient(image, sitk_target)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	sitk.WriteImage(reoriented, str(output_path), use_compression)
	return output_path



def rotate_image_and_mask():
	input_path = r"E:\test\image.mhd"
	output_path = r"E:\test\image_invert.mhd"
	target_orientation = "LPI"
	reorient_image_or_mask(input_path, output_path, target_orientation, None, False)



def _direction_to_orientation_code(direction: tuple, dim: int = 3) -> str:
	"""从 SimpleITK LPS 方向余弦矩阵推导每个体素轴的解剖方位码。

	SimpleITK/DICOM 使用 LPS 坐标系：
	  +X = Left（左）、+Y = Posterior（后）、+Z = Superior（上）
	方向矩阵每一列是对应图像轴（i/j/k）在 LPS 空间中的单位方向向量。
	取每列中绝对值最大的分量确定主方向，正值对应 L/P/S，负值对应 R/A/I。
	"""
	import numpy as np
	LPS_POS = ["L", "P", "S"]  # LPS 各轴正方向
	LPS_NEG = ["R", "A", "I"]  # LPS 各轴负方向
	dcm = np.array(direction).reshape(dim, dim)
	codes = []
	for col in range(dim):
		vec = dcm[:, col]
		dom = int(np.argmax(np.abs(vec)))
		codes.append(LPS_POS[dom] if vec[dom] > 0 else LPS_NEG[dom])
	return "".join(codes)

def read_nii_header(input_path: str | Path) -> dict:
	"""读取 .nii / .nii.gz 文件的头部信息。

	通过 SimpleITK 读取几何属性，并尝试通过 nibabel 读取完整的
	NIfTI 头部字段（包含从 DICOM 转换时可能保留的各项参数）。
	若未安装 nibabel，则仅返回 SimpleITK 所能提供的信息。

	Args:
		input_path: .nii 或 .nii.gz 文件路径。

	Returns:
		包含所有可获取头部信息的字典，同时将内容打印到控制台。
	"""
	input_path = Path(input_path)
	if not input_path.is_file():
		raise FileNotFoundError(f"文件不存在: {input_path}")

	suffixes = tuple(s.lower() for s in input_path.suffixes)
	if suffixes not in ((".nii",), (".nii", ".gz")):
		raise ValueError(f"仅支持 .nii / .nii.gz 格式，当前文件: {input_path}")

	result: dict = {}

	# ── 1. SimpleITK 几何信息 ──────────────────────────────────────────────
	sitk_image = None
	sitk_info: dict = {}
	meta_dict: dict = {}
	dim = 3  # NIfTI 默认 3D

	try:
		sitk_image = sitk.ReadImage(str(input_path))
		dim = sitk_image.GetDimension()

		sitk_info = {
			"维度 (Dimension)":        dim,
			"尺寸 (Size)":             sitk_image.GetSize(),
			"体素间距 (Spacing)":      sitk_image.GetSpacing(),
			"原点 (Origin)":           sitk_image.GetOrigin(),
			"方向余弦矩阵 (Direction)": sitk_image.GetDirection(),
			"像素类型 (PixelType)":    sitk.GetPixelIDValueAsString(sitk_image.GetPixelIDValue()),
			"组件数 (NumberOfComponents)": sitk_image.GetNumberOfComponentsPerPixel(),
		}
		# 从方向余弦矩阵推导解剖方位码（LPS 坐标系）
		if dim == 3 and len(sitk_image.GetDirection()) == 9:
			try:
				_orient_lps = _direction_to_orientation_code(sitk_image.GetDirection(), 3)
				# 转成起始端约定（同 ITK-SNAP 显示）
				_orient_origin = "".join(_OPPOSITE_AXIS[c] for c in _orient_lps)
				sitk_info["方位码（起始端约定, 同 ITK-SNAP）"] = _orient_origin
			except Exception as _e:
				sitk_info["方位码"] = f"<推导失败: {_e}>"

		for key in sitk_image.GetMetaDataKeys():
			meta_dict[key] = sitk_image.GetMetaData(key)

	except RuntimeError as _e:
		sitk_info = {"错误": f"SimpleITK 无法读取此文件（方向余弦可能不正交等）: {_e}"}
		print(f"\n[警告] SimpleITK 读取失败: {_e}")
		print("  将仅使用 nibabel 显示头部信息。\n")

	result["SimpleITK_几何信息"] = sitk_info
	result["SimpleITK_元数据"] = meta_dict

	# ── 2. nibabel NIfTI 原始头部 ──────────────────────────────────────────
	try:
		import nibabel as nib  # type: ignore

		nib_img = nib.load(str(input_path))
		hdr = nib_img.header

		nib_info: dict = {}
		for field_name in hdr.keys():
			try:
				value = hdr[field_name]
				# 将 numpy 数组转为 Python 原生类型以便序列化
				nib_info[field_name] = value.tolist() if hasattr(value, "tolist") else value
			except Exception:
				nib_info[field_name] = "<无法读取>"

		# 附加 nibabel 衍生信息
		nib_info["__zooms__"]        = list(hdr.get_zooms())
		nib_info["__data_dtype__"]   = str(hdr.get_data_dtype())
		nib_info["__qform_code__"]   = int(hdr["qform_code"])
		nib_info["__sform_code__"]   = int(hdr["sform_code"])
		nib_info["__qform_matrix__"] = hdr.get_qform().tolist()
		nib_info["__sform_matrix__"] = hdr.get_sform().tolist()

		result["nibabel_NIfTI头部"] = nib_info

		# ── 3. 从 nibabel affine 推导方位码 ──────────────────────────────────
		import nibabel.orientations as _nib_ornt
		_ornt  = _nib_ornt.io_orientation(nib_img.affine)
		# nibabel ornt2axcodes 返回各轴递增方向的解剖标签 (RAS 正方向约定)
		_ras_codes = _nib_ornt.ornt2axcodes(_ornt)
		_ras_str = "".join(_ras_codes)              # 递增方向约定，如 RAS
		_origin_str = "".join(                      # 起始端约定 (ITK-SNAP 同款)，如 LPI
			_OPPOSITE_AXIS[c] for c in _ras_codes
		)

		# 解剖全称映射
		_full_name = {
			"R": "Right（右）", "L": "Left（左）",
			"A": "Anterior（前）", "P": "Posterior（后）",
			"S": "Superior（上）", "I": "Inferior（下）",
		}
		_axis_labels = ["x", "y", "z"]

		orient_info: dict = {
			"方位码（起始端约定, 同 ITK-SNAP）": _origin_str,
			"方位码（递增方向约定, nibabel/NIfTI）": _ras_str,
		}
		# 逐轴输出 "from → to" 格式
		for idx, ras_code in enumerate(_ras_codes):
			origin_code = _OPPOSITE_AXIS[ras_code]
			ax = _axis_labels[idx] if idx < 3 else f"axis{idx}"
			orient_info[f"{ax} 轴方向"] = (
				f"{_full_name[origin_code]} → {_full_name[ras_code]}"
			)

		# 同时附上 SimpleITK Direction 推导的 LPS 方位码
		if sitk_image is not None and dim == 3 and len(sitk_image.GetDirection()) == 9:
			try:
				_lps_code = _direction_to_orientation_code(sitk_image.GetDirection(), 3)
				orient_info["方位码_LPS坐标系 (SimpleITK/DICOM)"] = _lps_code
			except Exception:
				pass

		result["方位信息"] = orient_info

	except ImportError:
		result["nibabel_NIfTI头部"] = "nibabel 未安装，跳过原始头部解析（pip install nibabel）"
		# nibabel 不可用时仅保留 LPS 方位码（来源于 SimpleITK）
		if sitk_image is not None and dim == 3 and len(sitk_image.GetDirection()) == 9:
			try:
				_lps_code = _direction_to_orientation_code(sitk_image.GetDirection(), 3)
				# LPS 方位码也转成起始端约定
				_origin_from_lps = "".join(_OPPOSITE_AXIS[c] for c in _lps_code)
				_full_name = {
					"R": "Right（右）", "L": "Left（左）",
					"A": "Anterior（前）", "P": "Posterior（后）",
					"S": "Superior（上）", "I": "Inferior（下）",
				}
				_axis_labels = ["x", "y", "z"]
				orient_fallback: dict = {
					"方位码（起始端约定, 同 ITK-SNAP）": _origin_from_lps,
					"方位码_LPS坐标系 (SimpleITK/DICOM)": _lps_code,
				}
				for idx, lps_c in enumerate(_lps_code):
					opp = _OPPOSITE_AXIS[lps_c]
					ax = _axis_labels[idx] if idx < 3 else f"axis{idx}"
					orient_fallback[f"{ax} 轴方向"] = (
						f"{_full_name[opp]} → {_full_name[lps_c]}"
					)
				orient_fallback["备注"] = "nibabel 未安装，方位码仅由 SimpleITK Direction 推导"
				result["方位信息"] = orient_fallback
			except Exception:
				pass

	# ── 3. 打印输出 ────────────────────────────────────────────────────────
	separator = "─" * 60
	print(f"\n{'═' * 60}")
	print(f"  NIfTI 文件头部信息: {input_path.name}")
	print(f"{'═' * 60}")

	print(f"\n{separator}")
	print("  【SimpleITK 几何信息】")
	print(separator)
	for k, v in sitk_info.items():
		print(f"  {k}: {v}")

	if meta_dict:
		print(f"\n{separator}")
		print("  【SimpleITK 元数据 / NIfTI 扩展字段】")
		print(separator)
		for k, v in meta_dict.items():
			print(f"  {k}: {v}")
	else:
		print("\n  （SimpleITK 未读取到额外元数据字段）")

	if isinstance(result.get("方位信息"), dict):
		print(f"\n{separator}")
		print("  【解剖方位信息】")
		print(separator)
		for k, v in result["方位信息"].items():
			print(f"  {k}: {v}")

	if isinstance(result.get("nibabel_NIfTI头部"), dict):
		print(f"\n{separator}")
		print("  【nibabel NIfTI 原始头部字段】")
		print(separator)
		for k, v in result["nibabel_NIfTI头部"].items():
			print(f"  {k}: {v}")
	else:
		print(f"\n  {result.get('nibabel_NIfTI头部', '')}")

	print(f"\n{'═' * 60}\n")
	return result


if __name__ == "__main__":

	#repair_image_example()
    
	#crop_image()

	rotate_image_and_mask()
	
	#read_nii_header(r"E:\Linux\ct.nii.gz")
