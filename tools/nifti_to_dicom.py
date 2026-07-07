#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone script: convert NIfTI file to derived DICOM series.

Usage:
    python tools/nifti_to_dicom.py <input.nii.gz> [output_dir]

If output_dir is omitted, it defaults to <input_basename>_dicom/ alongside the input.
"""

import os
import sys
import argparse

# Add project root so mimics_bridge can be imported
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Convert NIfTI to derived DICOM")
    parser.add_argument("input", help="Path to input NIfTI file (.nii or .nii.gz)")
    parser.add_argument("output", nargs="?", default=None,
                        help="Output DICOM directory (default: <input_stem>_dicom/)")
    parser.add_argument("--case-id", default="case", help="Patient ID / case identifier")
    args = parser.parse_args()

    nifti_path = os.path.abspath(args.input)
    if not os.path.isfile(nifti_path):
        print("ERROR: input file not found: {}".format(nifti_path))
        sys.exit(1)

    if args.output:
        dicom_out = os.path.abspath(args.output)
    else:
        base = os.path.splitext(nifti_path)[0]
        if base.endswith(".nii"):
            base = base[:-4]
        dicom_out = base + "_dicom"

    print("Converting: {}".format(nifti_path))
    print("Output:     {}".format(dicom_out))

    from mimics_bridge import nifti_to_derived_dicom

    result = nifti_to_derived_dicom(nifti_path, dicom_out, case_id=args.case_id)

    print()
    print("Conversion successful!")
    print("  Shape:     {}".format(result["shape"]))
    print("  Spacing:   {}".format(result["spacing"]))
    print("  Origin:    {}".format(result["origin"]))
    print("  SeriesUID: {}".format(result["series_uid"]))
    print("  Slices:    {} files in {}".format(result["shape"][2], result["dicom_folder"]))


if __name__ == "__main__":
    main()
