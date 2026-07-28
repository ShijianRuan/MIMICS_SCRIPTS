#!/usr/bin/env python3
"""Compare no-TTA, mirror-TTA and multi-scale inference on one checkpoint."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


MODES = {
    "single": {"tta_axes": [], "scales": []},
    # Model-space axis 1 is anterior/posterior after canonical ZYX conversion.
    # Deliberately omit axis 2 so right/left anatomy is never mirrored.
    "ap_tta": {"tta_axes": [[1]], "scales": []},
    "multiscale": {"tta_axes": [], "scales": [0.875, 1.125]},
    "ap_tta_multiscale": {"tta_axes": [[1]], "scales": [0.875, 1.125]},
}


def main():
    parser = argparse.ArgumentParser(description="Evaluate fixed checkpoint inference modes")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="Val", choices=["Tr", "Val"])
    parser.add_argument("--mode", action="append", choices=sorted(MODES), help="Repeat to restrict modes")
    args = parser.parse_args()
    base = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    selected_modes = args.mode or list(MODES)
    results = {}
    with tempfile.TemporaryDirectory(prefix="dinov3_inference_modes_") as temporary:
        for mode in selected_modes:
            config = json.loads(json.dumps(base))
            config["inference"] = MODES[mode]
            config_path = Path(temporary) / (mode + ".yaml")
            mode_output = Path(temporary) / (mode + ".json")
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            command = [
                sys.executable,
                "scripts/evaluate_model.py",
                "--config", str(config_path),
                "--checkpoint", args.checkpoint,
                "--data-root", args.data_root,
                "--split", args.split,
                "--output", str(mode_output),
            ]
            print("$ {}".format(" ".join(command)))
            subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)
            results[mode] = json.loads(mode_output.read_text(encoding="utf-8"))
    output.write_text(json.dumps({"modes": results}, indent=2) + "\n", encoding="utf-8")
    for mode, result in results.items():
        print("{}: Dice={:.4f}, HD95={}".format(
            mode,
            result["mean_dice"],
            "NA" if result["mean_hd95_mm"] is None else "{:.2f} mm".format(result["mean_hd95_mm"]),
        ))


if __name__ == "__main__":
    main()
