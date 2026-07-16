import os

# 硬编码输出目录
out = r"E:\total_test\mask_exports"

case_id = os.path.basename(os.path.dirname(
    getattr(mimics.file.get_project_information(), "filename", "") or ""
)) or "case"

seg_dir = os.path.join(out, case_id, "segmentations")
os.makedirs(seg_dir, exist_ok=True)

try:
    spacing = mimics.data.spacing  # 假设返回 (dx, dy, dz)
except Exception:
    spacing = (1.0, 1.0, 1.0)

print("原始 Spacing:", spacing)

for mask in mimics.data.masks:
    name = str(mask.name).replace("/", "_").replace(" ", "_")
    buf = mask.get_voxel_buffer()
    
    if buf is None:
        continue

    # 直接获取原始字节，绝对不做转置搬运
    raw = bytes(buf)
    
    # 原始尺寸 (X=259, Y=259, Z=283)
    # Mimics 的 buf 在内存中是 (Z, Y, X) 连续排列的
    X, Y, Z = 259, 259, 283 
    # 注意：写入 MHD 时，把顺序调整为 Z, Y, X，与内存数据完全对齐！
    mhd_dims = (Z, Y, X) 
    # 对应的 Spacing 也按 Z, Y, X 的顺序写进头文件
    mhd_spacing = (spacing[2], spacing[1], spacing[0])

    # 写入 .raw 二进制数据
    raw_path = os.path.join(seg_dir, name + ".raw")
    with open(raw_path, "wb") as f:
        f.write(raw)

    # 写入 .mhd 头文件 (注意 DimSize 和 ElementSpacing 的顺序)
    mhd_path = os.path.join(seg_dir, name + ".mhd")
    mhd_content = """ObjectType = Image
NDims = 3
DimSize = %d %d %d
ElementType = MET_UCHAR
ElementSpacing = %.4f %.4f %.4f
ElementByteOrderMSB = False
TransformMatrix = 0 0 1 0 1 0 1 0 0
Offset = 0 0 0
ElementDataFile = %s
AnatomicalOrientation = RAI
""" % (mhd_dims[0], mhd_dims[1], mhd_dims[2],
       mhd_spacing[0], mhd_spacing[1], mhd_spacing[2],
       os.path.basename(raw_path))
       
    with open(mhd_path, "w") as f:
        f.write(mhd_content)

print("🎉 导出完成！")
