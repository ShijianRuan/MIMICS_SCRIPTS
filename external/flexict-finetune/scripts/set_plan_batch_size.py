"""Lower the batch_size in a dataset's nnUNetPlans.json.

FlexiCT backbones (175M 2D / 355M 3D) train in fp32 and are far more
memory-heavy than the small PlainConvUNet nnU-Net sizes its batch_size for.
After preprocessing, lower the batch_size to fit the GPU. (This is exactly how
the validated runs handled it — the plans' batch_size was edited down rather
than overriding it in the trainer, which desyncs the oversample schedule.)

Usage:
    python scripts/set_plan_batch_size.py --preprocessed <nnUNet_preprocessed> \\
        --dataset 907 --config 2d --batch-size 8
    python scripts/set_plan_batch_size.py --preprocessed <nnUNet_preprocessed> \\
        --dataset 907 --config 3d_fullres --batch-size 2
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprocessed", required=True, help="nnUNet_preprocessed dir")
    ap.add_argument("--dataset", required=True, help="dataset id or name (e.g. 907 or LiverFS)")
    ap.add_argument("--config", required=True, help="configuration name (2d / 3d_fullres)")
    ap.add_argument("--batch-size", type=int, required=True)
    args = ap.parse_args()

    # find the dataset dir
    candidates = sorted(Path(args.preprocessed).glob(f"Dataset*{args.dataset}*"))
    if not candidates:
        candidates = sorted(Path(args.preprocessed).glob(f"Dataset*"))
        candidates = [c for c in candidates if args.dataset in c.name] or candidates
    if not candidates:
        raise SystemExit(f"no dataset dir matching {args.dataset} under {args.preprocessed}")
    ds_dir = candidates[0]
    plans_path = ds_dir / "nnUNetPlans.json"
    if not plans_path.exists():
        raise SystemExit(f"no nnUNetPlans.json in {ds_dir}")

    plans = json.loads(plans_path.read_text())
    cfg = plans["configurations"].get(args.config)
    if cfg is None:
        raise SystemExit(f"config '{args.config}' not in {plans_path}")
    old = cfg["batch_size"]
    cfg["batch_size"] = args.batch_size
    plans_path.write_text(json.dumps(plans, indent=2) + "\n")
    print(f"{ds_dir.name} [{args.config}] batch_size: {old} -> {args.batch_size}")


if __name__ == "__main__":
    main()
