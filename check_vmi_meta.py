import pydicom, glob, os, numpy as np
base = r"\\isi-sh\HSW\ImageAnalysisData\12-SPECT\SIRT\Spectral CT\Yuan Xiao Yan_611892931_153135"
for name, d in [("VMI40", "VMI_40KeV_Spectral_Series_130027_8002"), ("VMI70", "VMI_70KeV_Spectral_Series_130014_8001")]:
    files = sorted(glob.glob(os.path.join(base, d, "*.dcm")))
    print(f"===== {name} ({len(files)} files) =====")
    ds = pydicom.dcmread(files[0], force=True)
    print("  PixelSpacing:", ds.PixelSpacing)
    print("  SpacingBetweenSlices:", getattr(ds, 'SpacingBetweenSlices', 'N/A'))
    print("  SliceThickness:", getattr(ds, 'SliceThickness', 'N/A'))
    print("  RescaleSlope:", getattr(ds, 'RescaleSlope', 'N/A'))
    print("  RescaleIntercept:", getattr(ds, 'RescaleIntercept', 'N/A'))
    print("  BitsAllocated:", getattr(ds, 'BitsAllocated', 'N/A'), "BitsStored:", getattr(ds, 'BitsStored', 'N/A'), "PixelRepresentation:", getattr(ds, 'PixelRepresentation', 'N/A'))
    print("  Rows/Cols:", getattr(ds,'Rows',None), getattr(ds,'Columns',None))
    # 检查像素 dtype 和范围
    arr = ds.pixel_array
    print("  pixel_array dtype:", arr.dtype, "min:", arr.min(), "max:", arr.max())
    # 应用 rescale 后的 HU 范围
    slope = float(getattr(ds, 'RescaleSlope', 1)); inter = float(getattr(ds, 'RescaleIntercept', 0))
    hu = arr.astype(np.float64)*slope + inter
    print("  HU range after rescale: min=%.1f max=%.1f" % (hu.min(), hu.max()))
    # 检查最后一个文件的层厚是否一致
    ds2 = pydicom.dcmread(files[-1], force=True)
    print("  last SliceThickness:", getattr(ds2, 'SliceThickness', 'N/A'), "SpacingBetweenSlices:", getattr(ds2, 'SpacingBetweenSlices', 'N/A'))
