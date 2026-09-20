import pydicom, glob, os, numpy as np
base = r"\\isi-sh\HSW\ImageAnalysisData\12-SPECT\SIRT\Spectral CT\Anonymous-liver cancer_ZS26089293_144809"
for name, d in [("VMI40", "VMI_40KeV_Spectral_Series_151859_8001"), ("VMI70", "VMI_70KeV_Spectral_Series_151904_8002")]:
    files = sorted(glob.glob(os.path.join(base, d, "*.dcm")))
    print(f"===== {name} ({len(files)} files) =====")
    ds = pydicom.dcmread(files[0], force=True)
    print("  PixelSpacing:", ds.PixelSpacing)
    print("  SliceThickness:", getattr(ds, 'SliceThickness', 'N/A'))
    print("  RescaleSlope:", getattr(ds, 'RescaleSlope', 'N/A'), "RescaleIntercept:", getattr(ds, 'RescaleIntercept', 'N/A'))
    print("  PixelRepresentation:", getattr(ds, 'PixelRepresentation', 'N/A'), "BitsStored:", getattr(ds, 'BitsStored', 'N/A'))
    arr = ds.pixel_array
    slope = float(getattr(ds, 'RescaleSlope', 1)); inter = float(getattr(ds, 'RescaleIntercept', 0))
    hu = arr.astype(np.float64)*slope + inter
    print("  pixel dtype:", arr.dtype, "HU range: %.1f ~ %.1f" % (hu.min(), hu.max()))
    # 真实层间距
    zpos = []
    for f in files:
        d2 = pydicom.dcmread(f, force=True)
        zpos.append(float(d2.ImagePositionPatient[2]))
    zpos = np.array(zpos); zpos.sort()
    diffs = np.diff(zpos)
    print("  median z-diff:", np.median(diffs), "unique:", np.unique(np.round(diffs,4)))
