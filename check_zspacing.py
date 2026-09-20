import pydicom, glob, os, numpy as np
base = r"\\isi-sh\HSW\ImageAnalysisData\12-SPECT\SIRT\Spectral CT\Yuan Xiao Yan_611892931_153135"
# 检查所有文件的 SliceThickness 是否一致，以及实际层间距（用 ImagePositionPatient 计算）
for name, d in [("VMI40", "VMI_40KeV_Spectral_Series_130027_8002"), ("VMI70", "VMI_70KeV_Spectral_Series_130014_8001")]:
    files = sorted(glob.glob(os.path.join(base, d, "*.dcm")))
    print(f"===== {name} =====")
    # 收集 z 位置
    zpos = []
    for f in files:
        ds = pydicom.dcmread(f, force=True)
        zpos.append(float(ds.ImagePositionPatient[2]))
    zpos = np.array(zpos)
    zpos.sort()
    diffs = np.diff(zpos)
    print("  n slices:", len(zpos))
    print("  z range: %.3f ~ %.3f" % (zpos.min(), zpos.max()))
    print("  unique z-diffs:", np.unique(np.round(diffs, 4)))
    print("  median z-diff (真实层间距):", np.median(diffs))
    # 检查是否有重复 z（多帧/重复切片）
    print("  duplicate z count:", len(zpos) - len(np.unique(np.round(zpos,4))))
