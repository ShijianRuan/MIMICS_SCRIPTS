import os
import re
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import pydicom
import warnings
warnings.filterwarnings('ignore')

class DICOMToMHDConverter:
    def __init__(self, output_lps=True):
        """
        初始化转换器

        Args:
            output_lps: 是否转换为LPS方位（默认为True）
        """
        self.output_lps = output_lps

    def get_dicom_series_info(self, dicom_dir):
        """
        获取DICOM序列信息

        Args:
            dicom_dir: DICOM文件夹路径

        Returns:
            tuple: (sitk.Image对象, 序列信息字典)
        """
        reader = sitk.ImageSeriesReader()

        # 获取DICOM序列文件
        dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_dir))

        if len(dicom_names) == 0:
            print(f"警告: 在 {dicom_dir} 中未找到DICOM文件")
            return None, None

        # 读取第一个文件获取序列信息
        try:
            first_dcm = pydicom.dcmread(dicom_names[0], stop_before_pixels=True)
        except:
            first_dcm = None

        # 提取序列信息
        series_info = {
            'series_number': '000',
            'series_description': 'Unknown',
            'modality': 'OT',
            'patient_id': 'Unknown',
            'patient_name': 'Unknown',
            'study_date': 'Unknown'
        }

        if first_dcm:
            # 序列号
            if hasattr(first_dcm, 'SeriesNumber'):
                series_info['series_number'] = f"{first_dcm.SeriesNumber:03d}"

            # 序列描述
            if hasattr(first_dcm, 'SeriesDescription'):
                # 清理序列描述中的非法字符
                series_desc = str(first_dcm.SeriesDescription).strip()
                # 替换可能引起问题的字符
                series_desc = self.clean_filename(series_desc)
                if series_desc:  # 确保不为空
                    series_info['series_description'] = series_desc

            # 模态
            if hasattr(first_dcm, 'Modality'):
                series_info['modality'] = str(first_dcm.Modality)

            # 患者信息
            if hasattr(first_dcm, 'PatientID'):
                series_info['patient_id'] = str(first_dcm.PatientID)

            if hasattr(first_dcm, 'PatientName'):
                if hasattr(first_dcm.PatientName, 'given_name') and hasattr(first_dcm.PatientName, 'family_name'):
                    patient_name = f"{first_dcm.PatientName.family_name}_{first_dcm.PatientName.given_name}"
                else:
                    patient_name = str(first_dcm.PatientName)
                series_info['patient_name'] = self.clean_filename(patient_name)

            if hasattr(first_dcm, 'StudyDate'):
                series_info['study_date'] = str(first_dcm.StudyDate)

        reader.SetFileNames(dicom_names)

        # 读取图像
        image = reader.Execute()

        return image, series_info

    def clean_filename(self, filename):
        """
        清理文件名，移除或替换非法字符

        Args:
            filename: 原始文件名

        Returns:
            清理后的文件名
        """
        if not filename:
            return "Unknown"

        # 替换各种空格字符为下划线
        filename = re.sub(r'\s+', '_', filename)  # 所有空白字符

        # 移除或替换其他非法字符
        # Windows非法字符: <>:"/\|?*
        # Unix/Mac非法字符: / (斜杠)
        illegal_chars = r'[<>:"/\\|?*]'
        filename = re.sub(illegal_chars, '_', filename)

        # 移除首尾的点、空格、下划线
        filename = filename.strip(' ._')

        # 如果文件名变得太短或为空，使用默认值
        if len(filename) < 1:
            return "Unknown"

        # 限制文件名长度
        if len(filename) > 100:
            filename = filename[:100]

        return filename

    def generate_series_name(self, series_info, folder_name):
        """
        生成序列名称：序列描述_序列号

        Args:
            series_info: 序列信息字典
            folder_name: 文件夹名称（备用）

        Returns:
            生成的序列名称
        """
        series_desc = series_info['series_description']
        series_num = series_info['series_number']
        modality = series_info['modality']

        # 如果序列描述是默认值，尝试使用其他信息
        if series_desc == 'Unknown' or series_desc == '':
            # 尝试使用模态和序列号
            name_parts = [modality]
            if series_num != '000':
                name_parts.append(f"S{series_num}")

            # 如果没有有效信息，使用文件夹名
            if len(name_parts) <= 1:
                name_parts.append(self.clean_filename(folder_name))

            series_name = '_'.join(name_parts)
        else:
            # 使用序列描述+序列号
            if series_num != '000':
                series_name = f"{series_desc}_S{series_num}"
            else:
                series_name = series_desc

        # 再次清理以确保安全
        series_name = self.clean_filename(series_name)

        return series_name

    def convert_to_lps(self, image):
        """
        将图像转换为LPS方位

        Args:
            image: sitk.Image对象

        Returns:
            LPS方位的sitk.Image对象
        """
        # 获取当前方向
        current_direction = image.GetDirection()

        # 定义LPS方向矩阵 (3x3)
        lps_direction = (
            1, 0, 0,  # x方向
            0, 1, 0,  # y方向
            0, 0, 1   # z方向
        )

        # 如果当前方向不是LPS，重新定向
        if current_direction != lps_direction:
            print(f"当前方向: {current_direction}")
            print("转换为LPS方向...")

            # 使用SimpleITK的重新定向功能
            image = sitk.DICOMOrient(image, "LPS")

        return image

    def save_as_mhd_raw(self, image, output_path):
        """
        保存为MHD/RAW格式，像素类型为MET_SHORT (int16)
        Args:
            image: sitk.Image对象
            output_path: 输出路径（不含扩展名）
        """
        # 确保输出目录存在
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        # 转换像素类型为int16（MET_SHORT）
        if image.GetPixelID() != sitk.sitkInt16:
            image = sitk.Cast(image, sitk.sitkInt16)
        # 写入MHD文件
        mhd_path = f"{output_path}.mhd"
        sitk.WriteImage(image, mhd_path)
        # 检查文件是否成功创建
        if Path(mhd_path).exists():
            print(f"MHD文件已保存: {mhd_path}")
        else:
            print(f"警告: MHD文件可能未成功创建: {mhd_path}")

    def save_as_nifti(self, image, output_path):
        """
        保存为NIfTI格式（.nii.gz）

        Args:
            image: sitk.Image对象
            output_path: 输出路径（不含扩展名）
        """
        # 确保输出目录存在
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # 写入NIfTI文件
        nifti_path = f"{output_path}.nii.gz"
        sitk.WriteImage(image, nifti_path)

        # 检查文件是否成功创建
        if Path(nifti_path).exists():
            print(f"NIfTI文件已保存: {nifti_path}")
        else:
            print(f"警告: NIfTI文件可能未成功创建: {nifti_path}")

    def convert_single_series(self, dicom_dir, output_base_dir, series_name=None):
        """
        转换单个DICOM序列

        Args:
            dicom_dir: DICOM文件夹路径
            output_base_dir: 输出基础目录
            series_name: 序列名称（如果为None，则自动生成）

        Returns:
            是否成功转换
        """
        dicom_path = Path(dicom_dir)

        if not dicom_path.exists():
            print(f"错误: DICOM路径不存在 - {dicom_dir}")
            return False

        # 读取DICOM序列并获取信息
        print(f"正在读取DICOM序列: {dicom_dir}")
        image, series_info = self.get_dicom_series_info(dicom_dir)

        if image is None:
            return False

        # 确定序列名称
        if series_name is None:
            series_name = self.generate_series_name(series_info, dicom_path.name)
        else:
            # 清理用户提供的序列名称
            series_name = self.clean_filename(series_name)

        print(f"序列信息:")
        print(f"  序列描述: {series_info['series_description']}")
        print(f"  序列号: {series_info['series_number']}")
        print(f"  模态: {series_info['modality']}")
        print(f"  患者ID: {series_info['patient_id']}")
        print(f"  生成名称: {series_name}")

        # 转换为LPS方位（如果启用）
        if self.output_lps:
            image = self.convert_to_lps(image)

        # 创建患者子目录（可选）
        patient_id = self.clean_filename(series_info['patient_id'])
        if patient_id and patient_id != 'Unknown':
            output_dir = Path(output_base_dir) / patient_id / series_name
        else:
            output_dir = Path(output_base_dir) / series_name

        output_base_path = output_dir / series_name

        # 保存为MHD/RAW格式
        self.save_as_mhd_raw(image, str(output_base_path))

        # 保存为NIfTI格式
        self.save_as_nifti(image, str(output_base_path))

        # 打印图像信息
        print(f"图像信息:")
        print(f"  尺寸: {image.GetSize()}")
        print(f"  间距: {image.GetSpacing()}")
        print(f"  原点: {image.GetOrigin()}")
        print(f"  方向: {image.GetDirection()}")
        print(f"  像素类型: {image.GetPixelIDTypeAsString()}")

        return True, series_name

    def convert_multiple_paths(self, dicom_paths, output_base_dir, 
                               series_names=None, recursive=True, 
                               organize_by_patient=True):
        """
        转换多个DICOM路径

        Args:
            dicom_paths: DICOM路径列表
            output_base_dir: 输出基础目录
            series_names: 序列名称列表（如果为None，则自动生成）
            recursive: 是否递归查找DICOM序列
            organize_by_patient: 是否按患者ID组织输出目录

        Returns:
            tuple: (成功转换的序列数量, 转换详情列表)
        """
        success_count = 0
        total_count = 0
        conversion_details = []

        # 确保输出目录存在
        output_base_path = Path(output_base_dir)
        output_base_path.mkdir(parents=True, exist_ok=True)

        for i, dicom_path_str in enumerate(dicom_paths):
            dicom_path = Path(dicom_path_str)

            if not dicom_path.exists():
                print(f"警告: 路径不存在 - {dicom_path}")
                continue

            if dicom_path.is_file():
                print(f"警告: 跳过文件（期望文件夹）- {dicom_path}")
                continue

            if recursive:
                # 递归查找包含DICOM文件的文件夹
                dicom_folders = self.find_dicom_folders(dicom_path)
            else:
                # 只检查当前文件夹
                dicom_folders = [dicom_path] if self.has_dicom_files(dicom_path) else []

            for j, folder in enumerate(dicom_folders):
                total_count += 1

                # 确定序列名称
                custom_name = None
                if series_names is not None and i < len(series_names):
                    custom_name = series_names[i]

                print(f"\n{'='*60}")
                print(f"处理序列 {total_count}")
                print(f"DICOM文件夹: {folder}")
                print(f"{'='*60}")

                try:
                    success, final_series_name = self.convert_single_series(
                        str(folder), 
                        output_base_dir, 
                        custom_name
                    )

                    if success:
                        success_count += 1
                        conversion_details.append({
                            'input_path': str(folder),
                            'series_name': final_series_name,
                            'status': 'success'
                        })
                    else:
                        conversion_details.append({
                            'input_path': str(folder),
                            'series_name': custom_name or 'Unknown',
                            'status': 'failed'
                        })

                except Exception as e:
                    print(f"转换失败: {str(e)}")
                    conversion_details.append({
                        'input_path': str(folder),
                        'series_name': custom_name or 'Unknown',
                        'status': 'error',
                        'error': str(e)
                    })

        # 打印汇总信息
        print(f"\n{'='*60}")
        print(f"转换完成: {success_count}/{total_count} 个序列成功转换")
        print(f"输出目录: {output_base_dir}")

        # 保存转换日志
        self.save_conversion_log(output_base_dir, conversion_details)

        return success_count, conversion_details

    def save_conversion_log(self, output_base_dir, conversion_details):
        """
        保存转换日志

        Args:
            output_base_dir: 输出基础目录
            conversion_details: 转换详情列表
        """
        log_file = Path(output_base_dir) / "conversion_log.csv"

        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("序号,输入路径,序列名称,状态,错误信息\n")

            for i, detail in enumerate(conversion_details, 1):
                error_msg = detail.get('error', '')
                f.write(f'{i},"{detail["input_path"]}","{detail["series_name"]}",{detail["status"]},"{error_msg}"\n')

        print(f"转换日志已保存: {log_file}")

    def find_dicom_folders(self, base_path):
        """
        递归查找包含DICOM文件的文件夹

        Args:
            base_path: 基础路径

        Returns:
            包含DICOM文件的文件夹列表
        """
        dicom_folders = []
        base_path = Path(base_path)

        # 检查当前文件夹是否有DICOM文件
        if self.has_dicom_files(base_path):
            dicom_folders.append(base_path)

        # 递归检查子文件夹
        for item in base_path.iterdir():
            if item.is_dir():
                # 递归查找
                sub_folders = self.find_dicom_folders(item)
                dicom_folders.extend(sub_folders)

        return dicom_folders

    def has_dicom_files(self, folder_path):
        """
        检查文件夹是否包含DICOM文件

        Args:
            folder_path: 文件夹路径

        Returns:
            是否包含DICOM文件
        """
        folder = Path(folder_path)

        # 检查常见的DICOM文件扩展名
        dicom_extensions = ['.dcm', '.dicom', '.DCM', '.DICOM', '']

        dicom_count = 0
        for file_path in folder.iterdir():
            if file_path.is_file():
                # 检查扩展名
                if file_path.suffix in dicom_extensions:
                    # 尝试读取DICOM头信息确认
                    try:
                        pydicom.dcmread(str(file_path), stop_before_pixels=True)
                        dicom_count += 1
                        if dicom_count >= 2:  # 至少找到2个DICOM文件才认为是序列
                            return True
                    except:
                        continue

        return False


def do_dicom_to_mhd():
    """
    主函数示例
    """
    # ============ 配置参数 ============

    # 输入DICOM路径列表
    dicom_paths = [
        r"E:\test\spine\scoliosis\T2 mx 1mm_8004",
    ]

    # 可选的序列名称（如果不提供，将自动生成）
    series_names = [
        "t2_mx_1mm",  # 自动生成名称
    ]

    # 输出目录
    output_base_dir = "E:/test/spine/scoliosis/"

    # ============ 执行转换 ============

    # 创建转换器实例（默认输出LPS方位）
    converter = DICOMToMHDConverter(output_lps=True)

    # 转换多个路径
    success_count, details = converter.convert_multiple_paths(
        dicom_paths=dicom_paths,
        output_base_dir=output_base_dir,
        series_names=series_names,
        recursive=True,
        organize_by_patient=True
    )

    if success_count > 0:
        print(f"\n所有转换已完成！")
        print(f"输出目录: {output_base_dir}")
        print("\n输出文件命名规则:")
        print("  1. 默认: [序列描述]_S[序列号]")
        print("  2. 如果序列描述为空: [模态]_S[序列号]")
        print("  3. 所有空格和非法字符已替换为下划线")
        print("  4. 文件组织: 按患者ID/序列名称")
    else:
        print("没有成功转换任何序列")


def batch_convert_folder(folder_path, output_base_dir, recursive=True):
    """
    批量转换文件夹中的所有DICOM序列

    Args:
        folder_path: 包含DICOM序列的根文件夹
        output_base_dir: 输出目录
        recursive: 是否递归查找
    """
    converter = DICOMToMHDConverter(output_lps=True)

    # 查找所有DICOM文件夹
    dicom_folders = converter.find_dicom_folders(folder_path)

    if not dicom_folders:
        print(f"在 {folder_path} 中未找到DICOM序列")
        return

    print(f"找到 {len(dicom_folders)} 个DICOM序列")

    # 转换所有找到的序列
    success_count, details = converter.convert_multiple_paths(
        dicom_paths=[str(f) for f in dicom_folders],
        output_base_dir=output_base_dir,
        series_names=None,  # 全部自动命名
        recursive=False,  # 已经递归查找过了
        organize_by_patient=True
    )

    return success_count, details


def nii_to_mhd(input_path: str = None, output_path: str = None):
    """
    将 .nii 或 .nii.gz 格式的图像转换为 .mhd 格式。

    Args:
        input_path: 输入的 .nii/.nii.gz 文件路径，为空时交互输入
        output_path: 输出的 .mhd 文件路径，为空时交互输入
    """
    if input_path is None:
        input_path = input("请输入 .nii/.nii.gz 文件路径: ").strip().strip('"').strip("'")
    if output_path is None:
        output_path = input("请输入输出的 .mhd 文件路径（含文件名，如 output.mhd）: ").strip().strip('"').strip("'")

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        print(f"错误: 输入文件不存在: {input_path}")
        return False

    if input_path.suffix not in ('.gz', '.nii') and not str(input_path).endswith('.nii.gz'):
        print(f"警告: 输入文件可能不是 .nii/.nii.gz 格式: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() != '.mhd':
        output_path = output_path.with_suffix('.mhd')

    print(f"正在读取: {input_path}")
    image = sitk.ReadImage(str(input_path))

    print(f"正在写入: {output_path}")
    sitk.WriteImage(image, str(output_path))

    # 检查 .raw 伴随文件是否生成
    raw_file = output_path.with_suffix('.raw')
    if raw_file.exists():
        print(f"转换成功: {output_path}  +  {raw_file.name}")
    else:
        print(f"转换成功: {output_path}")
    return True


def do_nii_to_mhd():

    input_file = r"E:\test\brain\t1_mx3d\t1_mx3d_mask.nii.gz"
    output_file = r"E:\test\brain\t1_mx3d\t1_mx3d_mask.mhd"
    nii_to_mhd(input_file, output_file)




import os
import SimpleITK as sitk
import pydicom
import numpy as np
from typing import List, Optional


def dicom_series_to_mhd_nii_single_series(
    input_root: str,
    output_root: str,
    output_formats: List[str] = ["mhd", "nii.gz"],
    target_folders: Optional[List[str]] = None
) -> None:
    """
    批量将DICOM序列文件夹转换为 MHD 和/或 nii.gz 格式，强制转换为RAI方位
    RAI方位：Right→Left, Anterior→Posterior, Inferior→Superior (脚→头)
    
    参数:
        input_root: 输入根目录，下含多个DICOM序列子文件夹
        output_root: 输出根目录，自动创建
        output_formats: 输出格式，可选 ["mhd"], ["nii.gz"], ["mhd", "nii.gz"]
        target_folders: 指定要转换的子文件夹名称列表，为空则转换所有子文件夹
    """
    # 1. 校验输入目录
    if not os.path.isdir(input_root):
        raise ValueError(f"输入目录不存在: {input_root}")
    
    # 创建输出目录
    os.makedirs(output_root, exist_ok=True)
    
    # 2. 获取需要转换的文件夹列表
    all_subfolders = [f for f in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, f))]
    
    if target_folders is not None and len(target_folders) > 0:
        # 只转换指定的文件夹
        convert_folders = [f for f in target_folders if f in all_subfolders]
        if not convert_folders:
            print("未找到任何指定的转换文件夹！")
            return
    else:
        # 转换所有子文件夹
        convert_folders = all_subfolders
    
    print(f"即将转换的序列文件夹: {convert_folders}")
    print(f"输出格式: {output_formats}")
    print("=" * 60)

    # 3. 遍历转换每个DICOM序列
    for folder_name in convert_folders:
        dicom_dir = os.path.join(input_root, folder_name)
        print(f"\n正在处理: {folder_name}")

        try:
            # ===================== 核心：读取DICOM序列 =====================
            reader = sitk.ImageSeriesReader()
            # 获取排序后的DICOM文件路径
            dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
            if not dicom_names:
                print(f"警告：{folder_name} 中无DICOM文件，跳过")
                continue
            
            reader.SetFileNames(dicom_names)
            image = reader.Execute()

            # ===================== 核心：强制转换为RAI方位 =====================
            # RAI: Right (X), Anterior (Y), Inferior (Z)
            target_direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputDirection(target_direction)
            resampler.SetOutputOrigin(image.GetOrigin())
            resampler.SetOutputSpacing(image.GetSpacing())
            resampler.SetSize(image.GetSize())
            # 线性插值，保证图像质量
            resampler.SetInterpolator(sitk.sitkLinear)
            # 执行重采样，得到RAI方位图像
            rai_image = resampler.Execute(image)

            # ===================== 保存输出文件 =====================
            output_basename = os.path.join(output_root, folder_name)

            # 转换为 int16 (MET_SHORT)
            if rai_image.GetPixelID() != sitk.sitkInt16:
                rai_image = sitk.Cast(rai_image, sitk.sitkInt16)

            if "mhd" in output_formats:
                mhd_path = output_basename + ".mhd"
                sitk.WriteImage(rai_image, mhd_path)
                print(f"  ✅ 保存MHD: {mhd_path}")
            
            if "nii.gz" in output_formats:
                nii_path = output_basename + ".nii.gz"
                sitk.WriteImage(rai_image, nii_path)
                print(f"  ✅ 保存NII.GZ: {nii_path}")

        except Exception as e:
            print(f"❌ 处理 {folder_name} 失败: {str(e)}")

    print("\n" + "=" * 60)
    print("所有任务处理完成！")



def do_dicom_to_mhd_nii_single_series():
    """
    主函数：直接在这里手动配置所有参数，无需命令行
    """
    # ===================== 【手动配置参数】 =====================
    # 输入根目录：包含多个DICOM序列子文件夹
    INPUT_ROOT = r"E:\test\MIseg\CT"
    
    # 输出根目录：转换后的文件保存在这里
    OUTPUT_ROOT = r"E:\test\MIseg\CT"
    
    # 输出格式选择：
    # ["mhd"] → 只转MHD
    # ["nii.gz"] → 只转nii.gz
    # ["mhd", "nii.gz"] → 两者都转
    OUTPUT_FORMATS = ["mhd"]
    
    # 指定要转换的子文件夹名称（为空列表则转换所有）
    TARGET_FOLDERS = []  # 示例
    # TARGET_FOLDERS = []  # 取消注释则转换所有子文件夹
    # ==========================================================

    # 调用转换函数
    dicom_series_to_mhd_nii_single_series(
        input_root=INPUT_ROOT,
        output_root=OUTPUT_ROOT,
        output_formats=OUTPUT_FORMATS,
        target_folders=TARGET_FOLDERS
    )





if __name__ == "__main__":
    # 示例1: 转换特定路径列表
    #do_dicom_to_mhd()

    # 示例2: 批量转换整个文件夹
    # batch_convert_folder(
    #     folder_path="/data/dicom_studies",
    #     output_base_dir="/output/converted",
    #     recursive=True
    # )

    # 实例3：dicom序列已经整理好，一个序列一个文件夹的情况下，批量转换
    do_dicom_to_mhd_nii_single_series()

    #do_nii_to_mhd()