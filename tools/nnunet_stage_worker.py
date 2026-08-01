#!/usr/bin/env python3
"""Execute one heavyweight nnU-Net stage in an isolated child process."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "external" / "nnunet_segmentation_workflow"
TRAINERS = WORKFLOW / "trainers"
for candidate in (ROOT, ROOT / "tools", WORKFLOW):
    value = str(candidate)
    if value not in sys.path:
        sys.path.insert(0, value)

from nnunet_common import read_json, write_json_atomic  # noqa: E402


def _wait_for_start_gate(spec: dict) -> None:
    gate_text = str(spec.get("start_gate") or "").strip()
    if not gate_text:
        return
    gate = Path(gate_text)
    control_text = str(spec.get("control_path") or "").strip()
    control = Path(control_text) if control_text else None
    deadline = time.time() + max(
        1.0, float(spec.get("start_gate_timeout_seconds") or 120.0)
    )
    while not gate.is_file():
        if control is not None:
            values = read_json(control, {}) or {}
            if str(values.get("action") or "").lower() in {"cancel", "stop"}:
                raise InterruptedError("cancel")
        if time.time() >= deadline:
            raise RuntimeError(
                "The nnU-Net controller stopped before authorizing the worker to start."
            )
        time.sleep(0.1)


def _apply_environment(values: dict) -> None:
    for key, value in (values or {}).items():
        if value not in (None, ""):
            os.environ[str(key)] = str(value)
    existing = str(os.environ.get("nnUNet_extTrainer") or "").strip()
    roots = [item for item in existing.split(os.pathsep) if item]
    if str(TRAINERS) not in roots:
        roots.insert(0, str(TRAINERS))
    os.environ["nnUNet_extTrainer"] = os.pathsep.join(roots)


def run_stage(spec: dict) -> dict:
    stage = str(spec.get("stage") or "")
    params = dict(spec.get("params") or {})
    _apply_environment(dict(spec.get("environment") or {}))
    if stage == "preprocess":
        from Action2_PlanAndPreprocess import stage_preprocess

        plans = stage_preprocess(**params)
        if isinstance(plans, (list, tuple)):
            plans = plans[0] if plans else "nnUNetPlans"
        if not isinstance(plans, str) or not plans:
            plans = "nnUNetPlans"
        return {"plans": plans}
    if stage == "train":
        from Action3_Train import stage_train

        stage_train(**params)
        return {"trained": True}
    if stage == "infer":
        from Action4_Predict import easy_predict

        easy_predict(**params)
        return {"predicted": True, "output_path": str(params["output_path"])}
    raise ValueError("Unsupported nnU-Net worker stage: {}".format(stage))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    result_path = Path(args.result).resolve()
    try:
        spec = read_json(Path(args.spec).resolve(), {}) or {}
        _wait_for_start_gate(spec)
        result = run_stage(spec)
        write_json_atomic(result_path, {"status": "ok", "result": result})
        return 0
    except Exception as exc:
        write_json_atomic(
            result_path,
            {
                "status": "error",
                "error": "{}: {}".format(type(exc).__name__, exc),
                "traceback": traceback.format_exc(),
            },
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
