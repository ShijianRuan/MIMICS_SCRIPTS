"""Prepare nnU-Net inference inputs from a Totalsegmentator-style source tree.

The source tree has per-case directories like ``<source>/<case>/ct.nii.gz``.
nnU-Net v2 predict expects one file per case named ``<case>_0000.nii.gz`` (the
``_0000`` is the channel-0 suffix). This script links each CT into an output
directory under the nnU-Net name, and writes a ``case_list.txt`` listing every
case id (one per line) for downstream steps.

Linking (not copying) is used so a 1228-case tree costs ~no extra disk. On Linux
this is a plain symlink. On Windows, symlinks need privileges/developer mode, so
the script falls back to a hard link, then to a copy.

This is target-agnostic — it works for any Totalsegmentator-style source tree,
not just liver. Reorientation is left to the nnU-Net predictor (the dataset uses
``NibabelIOWithReorient``), so only the filename suffix is normalized here.

Usage:
    python scripts/prepare_inputs.py \
        --source Z:/.../Totalsegmentator_dataset_v201 \
        --out ./infer_input --cases-file ./infer_input/case_list.txt

    # smoke (first N cases only):
    python scripts/prepare_inputs.py --source ... --out ... --max-cases 5
"""
import argparse
import os
from pathlib import Path


def _link(src: Path, dst: Path) -> str:
    """Link src -> dst. Try symlink, then hard link, then copy. Return which."""
    if dst.exists() or dst.is_symlink():
        return "exists"
    try:
        os.symlink(src, dst)
        return "symlink"
    except (OSError, NotImplementedError):
        pass
    try:
        os.link(src, dst)
        return "hardlink"
    except (OSError, NotImplementedError):
        pass
    import shutil
    shutil.copy2(src, dst)
    return "copy"


def _iter_cases(source: Path):
    """Yield (case_id, ct_path) for every <source>/<case>/ct.nii.gz found."""
    for cdir in sorted(p for p in source.iterdir() if p.is_dir()):
        ct = cdir / "ct.nii.gz"
        if ct.exists():
            yield cdir.name, ct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    help="source root: <source>/<case>/ct.nii.gz")
    ap.add_argument("--out", required=True, help="output dir for <case>_0000.nii.gz")
    ap.add_argument("--cases-file", default="",
                    help="write case list here (default: <out>/case_list.txt)")
    ap.add_argument("--max-cases", type=int, default=0,
                    help="cap number of cases (0 = all)")
    ap.add_argument("--start", type=int, default=0,
                    help="skip the first N cases (0 = start from beginning)")
    args = ap.parse_args()

    source = Path(args.source)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases_file = Path(args.cases_file) if args.cases_file else out_dir / "case_list.txt"

    cases = list(_iter_cases(source))
    if not cases:
        raise SystemExit(f"no <case>/ct.nii.gz found under {source}")
    if args.start:
        cases = cases[args.start:]
    if args.max_cases:
        cases = cases[:args.max_cases]

    print(f"preparing {len(cases)} cases -> {out_dir}")
    counts = {"symlink": 0, "hardlink": 0, "copy": 0, "exists": 0}
    case_ids = []
    for i, (cid, ct) in enumerate(cases, 1):
        dst = out_dir / f"{cid}_0000.nii.gz"
        kind = _link(ct, dst)
        counts[kind] = counts.get(kind, 0) + 1
        case_ids.append(cid)
        if i % 100 == 0 or i == len(cases):
            print(f"  {i}/{len(cases)}")

    cases_file.parent.mkdir(parents=True, exist_ok=True)
    cases_file.write_text("\n".join(case_ids) + "\n", encoding="utf-8")
    print(f"done: {len(case_ids)} cases linked ({counts})")
    print(f"case list -> {cases_file}")


if __name__ == "__main__":
    main()
