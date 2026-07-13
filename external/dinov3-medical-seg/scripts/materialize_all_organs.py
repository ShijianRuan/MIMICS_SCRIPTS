#!/usr/bin/env python3
"""
Multi-organ materialization from TotalSegmentator source data.

Downloads (if needed) and materializes datasets for all target organs:
brain, liver, aorta, scapula_left, adrenal_gland_right

Each organ gets its own dataset directory:
    data/totalseg/<organ>/{imagesTr,labelsTr,imagesVal,labelsVal}/

Usage:
    # Materialize all organs
    python scripts/materialize_all_organs.py --src data/totalseg/source

    # Materialize specific organs
    python scripts/materialize_all_organs.py --src data/totalseg/source --organs liver brain

    # Check what's available without materializing
    python scripts/materialize_all_organs.py --src data/totalseg/source --check
"""

import argparse
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates


# ──────────────────────────────────────────────
# Target organs with TotalSegmentator label names
# ──────────────────────────────────────────────

TARGET_ORGANS = {
    "brain": {
        "ts_label": "brain",
        "description": "Brain (neurocranium)",
        "min_cases": 10,  # Minimum cases needed for meaningful few-shot
    },
    "liver": {
        "ts_label": "liver",
        "description": "Liver",
        "min_cases": 10,
    },
    "aorta": {
        "ts_label": "aorta",
        "description": "Aorta",
        "min_cases": 5,
    },
    "scapula_left": {
        "ts_label": "scapula_left",
        "description": "Scapula (left)",
        "min_cases": 5,
    },
    "adrenal_gland_right": {
        "ts_label": "adrenal_gland_right",
        "description": "Adrenal Gland (right)",
        "min_cases": 5,
    },
}

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC = PROJECT_ROOT / "data" / "totalseg" / "source"
DEFAULT_OUT_BASE = PROJECT_ROOT / "data" / "totalseg"


def resample_image_to_label_grid(ct_path, lb_path, out_path):
    """Resample CT onto the label's affine/shape for affine match."""
    ct_img = nib.as_closest_canonical(nib.load(str(ct_path)))
    lb_img = nib.as_closest_canonical(nib.load(str(lb_path)))
    ct = ct_img.get_fdata(dtype=np.float32)
    lb_shape = lb_img.shape
    lb_aff = lb_img.affine
    ct_aff = ct_img.affine

    zz, yy, xx = np.meshgrid(
        np.arange(lb_shape[0]), np.arange(lb_shape[1]), np.arange(lb_shape[2]),
        indexing="ij")
    grid = np.stack([zz.ravel(), yy.ravel(), xx.ravel(), np.ones(zz.size)], axis=0)
    world = lb_aff @ grid
    inv = np.linalg.inv(ct_aff)
    vox = inv @ np.vstack([world[:3], np.ones((1, world.shape[1]))])
    coords = vox[:3].reshape(3, *lb_shape)
    resampled = map_coordinates(ct, coords, order=1, mode="nearest").astype(np.float32)

    out = nib.Nifti1Image(resampled, lb_aff, header=lb_img.header)
    nib.save(out, str(out_path))


def check_source(src: Path) -> dict:
    """Check source data availability for all target organs."""
    results = {}
    if not src.is_dir():
        return results

    cases = sorted([d for d in src.iterdir() if d.is_dir() and d.name.startswith("s")])
    n_cases = len(cases)
    if n_cases == 0:
        return results

    for organ_name, info in TARGET_ORGANS.items():
        ts_label = info["ts_label"]
        usable = 0
        for c in cases:
            ct = c / "ct.nii.gz"
            lb = c / "segmentations" / (ts_label + ".nii.gz")
            if ct.is_file() and lb.is_file():
                try:
                    d = np.asanyarray(nib.load(str(lb)).dataobj)
                    if (d > 0).sum() > 0:
                        usable += 1
                except Exception:
                    pass

        results[organ_name] = {
            "usable": usable,
            "total_cases": n_cases,
            "ts_label": ts_label,
            "description": info["description"],
            "sufficient": usable >= info["min_cases"],
        }

    return results


def materialize_organ(src: Path, out_base: Path, organ_name: str,
                      val_frac: float = 0.2, seed: int = 42) -> dict:
    """Materialize a single organ's dataset."""
    info = TARGET_ORGANS[organ_name]
    ts_label = info["ts_label"]
    out = out_base / organ_name

    cases = sorted([d for d in src.iterdir() if d.is_dir() and d.name.startswith("s")])
    usable = []
    for c in cases:
        ct = c / "ct.nii.gz"
        lb = c / "segmentations" / (ts_label + ".nii.gz")
        if not ct.is_file() or not lb.is_file():
            continue
        try:
            d = np.asanyarray(nib.load(str(lb)).dataobj)
            if (d > 0).sum() == 0:
                continue
        except Exception:
            continue
        usable.append((c.name, ct, lb))

    if len(usable) < info["min_cases"]:
        return {
            "organ": organ_name,
            "status": "insufficient_data",
            "usable": len(usable),
            "required": info["min_cases"],
        }

    rng = random.Random(seed)
    rng.shuffle(usable)
    n_val = max(1, int(round(len(usable) * val_frac))) if len(usable) > 3 else 1
    val = usable[:n_val]
    train = usable[n_val:]

    # Clean existing output
    if out.exists():
        shutil.rmtree(str(out))

    for split, rows in (("Tr", train), ("Val", val)):
        if not rows:
            continue
        idir = out / f"images{split}"
        ldir = out / f"labels{split}"
        idir.mkdir(parents=True, exist_ok=True)
        ldir.mkdir(parents=True, exist_ok=True)
        for name, ct, lb in rows:
            idst = idir / f"{name}.nii.gz"
            ldst = ldir / f"{name}.nii.gz"
            if not ldst.exists():
                shutil.copy2(str(lb), str(ldst))
            if not idst.exists():
                resample_image_to_label_grid(ct, lb, idst)

    return {
        "organ": organ_name,
        "status": "ok",
        "train": len(train),
        "val": len(val),
        "total_usable": len(usable),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Multi-organ TotalSegmentator data materialization"
    )
    parser.add_argument("--src", type=str, default=str(DEFAULT_SRC),
                        help="Source directory with sXXXX/ cases")
    parser.add_argument("--out-base", type=str, default=str(DEFAULT_OUT_BASE),
                        help="Output base directory")
    parser.add_argument("--organs", nargs="*",
                        choices=list(TARGET_ORGANS.keys()),
                        help="Organs to materialize (default: all)")
    parser.add_argument("--val-frac", type=float, default=0.2,
                        help="Validation fraction")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check", action="store_true",
                        help="Only check availability, don't materialize")
    args = parser.parse_args()

    src = Path(args.src).resolve()
    out_base = Path(args.out_base).resolve()
    organs = args.organs or list(TARGET_ORGANS.keys())

    print(f"Source: {src}")
    print(f"Output: {out_base}")
    print()

    # Check
    results = check_source(src)
    if not results:
        print("No source data found.")
        print(f"\nTo prepare data:")
        print(f"  1. Download TotalSegmentator dataset")
        print(f"  2. Extract to: {src}")
        print(f"  3. Structure: {src}/sXXXX/ct.nii.gz + segmentations/<organ>.nii.gz")
        print(f"\nOr run TotalSegmentator on your own CT data:")
        print(f"  TotalSegmentator -i ct.nii.gz -o sXXXX/segmentations/ --roi_subset <organs>")
        return

    print("Data availability check:")
    print(f"{'Organ':<25s} {'Usable':<8s} {'Total':<8s} {'Status':<12s} {'Description'}")
    print("-" * 80)
    for organ, info in sorted(results.items()):
        status = "✅ Ready" if info["sufficient"] else "❌ Insufficient"
        print(f"{organ:<25s} {info['usable']:<8d} {info['total_cases']:<8d} "
              f"{status:<12s} {info['description']}")

    if args.check:
        return

    # Materialize
    print(f"\nMaterializing {len(organs)} organs...")
    for organ in organs:
        result = materialize_organ(src, out_base, organ,
                                   val_frac=args.val_frac, seed=args.seed)
        if result["status"] == "ok":
            print(f"  ✅ {organ}: {result['train']} train + {result['val']} val "
                  f"({result['total_usable']} total)")
        else:
            print(f"  ❌ {organ}: {result['status']} "
                  f"(usable={result.get('usable', 0)}, "
                  f"need={result.get('required', '?')})")

    print(f"\nDone. Data in: {out_base}")


if __name__ == "__main__":
    main()
