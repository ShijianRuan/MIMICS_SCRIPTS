import pydicom, glob, os
base = r"\\isi-sh\HSW\ImageAnalysisData\12-SPECT\SIRT\Spectral CT\Yuan Xiao Yan_611892931_153135"
rtss40 = glob.glob(os.path.join(base, "RTSS_VMI_40keV_Abdomen_CT_HHMMSS_93070324", "*.dcm"))[0]
rtss70 = glob.glob(os.path.join(base, "RTSS_VMI_70keV_CT_HHMMSS_93070326", "*.dcm"))[0]
for name, rtss in [("RTSS40", rtss40), ("RTSS70", rtss70)]:
    ds = pydicom.dcmread(rtss, force=True)
    print(f"=== {name} ===")
    for ref in ds.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence:
        print("  SeriesInstanceUID:", ref.SeriesInstanceUID)
        for c in ref.ContourImageSequence:
            print("    RefSOPClassUID:", c.ReferencedSOPClassUID, "RefSOPInstanceUID:", c.ReferencedSOPInstanceUID)
            break
        break
for name, d in [("VMI40", "VMI_40KeV_Spectral_Series_130027_8002"), ("VMI70", "VMI_70KeV_Spectral_Series_130014_8001")]:
    f = glob.glob(os.path.join(base, d, "*.dcm"))[0]
    ds = pydicom.dcmread(f, force=True)
    print(f"{name}: SeriesInstanceUID={ds.SeriesInstanceUID} SOPClassUID={ds.SOPClassUID} Modality={getattr(ds,'Modality',None)}")
