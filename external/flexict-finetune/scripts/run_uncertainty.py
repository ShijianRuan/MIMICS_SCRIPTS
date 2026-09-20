"""Rank cases by per-voxel segmentation uncertainty across multiple models.

Wraps the generic Action6 uncertainty method (``uncertainty/uncertainty.py``)
for the FlexiCT few-shot setting: each trained model (2D, 3D, ...) is one
"algorithm". Run every model on the same inputs, point this script at the
prediction directories, and it computes per-voxel disagreement, saves per-case
uncertainty NIfTI maps, and ranks all cases by an aggregate metric.

With only 2 masks (2D + 3D), ``disagreement`` is the recommended method: per
voxel it is the fraction of masks that disagree with the majority (0 when both
agree, 0.5 when they split), so the default sort key ``integrated`` (= Σ
voxel-uncertainty × voxel volume) reduces to "total volume where the two models
disagree" — intuitive and well-scaled for ranking. ``entropy``/``variance``/
``compare`` are also available; see ``docs/uncertainty.md``.

Usage (2D + 3D predictions already produced):
    python scripts/run_uncertainty.py \
        --mask-dirs preds_2d preds_3d_fullres \
        --cases-file infer_input/case_list.txt \
        --out uncertainty_out --method disagreement \
        --filename-template "{case}_0000.nii.gz" --target-labels 1

The ``--filename-template`` must match the prediction filenames. nnU-Net predict
names outputs after the input, so inputs ``<case>_0000.nii.gz`` produce
``<case>_0000.nii.gz`` predictions — hence the ``_0000`` template above.
"""
import argparse
import os
import sys
from pathlib import Path

# make the repo-root `uncertainty` package importable when run from scripts/
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from uncertainty import run_batch, analyze_uncertainty_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description="Rank cases by segmentation uncertainty across models.")
    ap.add_argument("--mask-dirs", nargs="+", required=True,
                    help="one dir per model/algorithm, each holding <case><suffix>.nii.gz (>= 2 dirs)")
    ap.add_argument("--out", required=True, help="output dir for uncertainty maps + ranking CSV")
    ap.add_argument("--cases-file", default="",
                    help="txt of case ids, one per line (default: union of all mask dirs)")
    ap.add_argument("--method", default="disagreement",
                    choices=["entropy", "disagreement", "variance", "compare"],
                    help="uncertainty metric (default: disagreement — best for 2 masks)")
    ap.add_argument("--filename-template", default="{case}_0000.nii.gz",
                    help="mask filename template, {case} substituted (default matches nnU-Net output)")
    ap.add_argument("--sort-key", default="integrated",
                    help="ranking key (integrated/uncertain_vol/max/mean_nonzero/cc_*/boundary_*)")
    ap.add_argument("--target-labels", nargs="*", type=int, default=None,
                    help="labels to include (default: all non-zero). e.g. --target-labels 1")
    ap.add_argument("--min-masks", type=int, default=2,
                    help="skip cases with fewer than this many masks")
    ap.add_argument("--boundary-dilation", type=int, default=1,
                    help="dilation (voxels) of the label-interface band")
    ap.add_argument("--no-components", action="store_true",
                    help="skip label-interface (partition_boundary) component")
    ap.add_argument("--no-consensus", action="store_true",
                    help="do not save the majority-vote consensus mask")
    ap.add_argument("--plots", action="store_true",
                    help="draw per-case heatmaps (slow for many cases; off by default)")
    ap.add_argument("--no-cc", action="store_true",
                    help="skip largest-connected-component metrics in the ranking pass")
    ap.add_argument("--n-workers", type=int, default=0,
                    help="parallel threads (0 = auto = cpu count, 1 = serial)")
    ap.add_argument("--report-csv", default="",
                    help="ranking CSV path (default: <out>/uncertainty_ranking.csv)")
    args = ap.parse_args()

    if len(args.mask_dirs) < 2:
        raise SystemExit("need >= 2 mask dirs (one per model)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_csv = args.report_csv or str(out_dir / "uncertainty_ranking.csv")
    log_path = str(out_dir / "uncertainty_batch.log")
    n_workers = None if args.n_workers <= 0 else args.n_workers

    save_components = not args.no_components
    save_consensus = not args.no_consensus
    save_plots = args.plots
    # multi-label interface stats only matter when >1 target label; for a single
    # binary target there is no interface, so skip the component pass entirely.
    if args.target_labels is not None and len(args.target_labels) < 2:
        save_components = False

    print("=" * 60)
    print(f"uncertainty ranking")
    print(f"  mask dirs ({len(args.mask_dirs)}): {args.mask_dirs}")
    print(f"  method: {args.method}  sort_key: {args.sort_key}")
    print(f"  template: {args.filename_template}  target_labels: {args.target_labels}")
    print(f"  out: {out_dir}")
    print("=" * 60)

    run_batch(
        args.mask_dirs, str(out_dir),
        case_list_txt=args.cases_file or None,
        filename_template=args.filename_template,
        method=args.method,
        target_labels=args.target_labels,
        min_masks=args.min_masks,
        boundary_dilation=args.boundary_dilation,
        save_components=save_components,
        save_consensus=save_consensus,
        save_plots=save_plots,
        log_path=log_path,
        n_workers=n_workers,
    )

    analyze_uncertainty_dir(
        str(out_dir), method=args.method, sort_key=args.sort_key,
        report_csv=report_csv, compute_boundary=save_components,
        compute_cc=not args.no_cc,
    )
    print(f"\nranking -> {report_csv}")


if __name__ == "__main__":
    main()
