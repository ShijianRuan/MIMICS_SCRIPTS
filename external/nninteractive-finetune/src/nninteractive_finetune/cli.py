"""Command line interface for independent fine-tuning and verification."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

from .config import load_config
from .evaluate import evaluate_model
from .checkpoint import verify_exported_model
from .model import audit_model_dir
from .runtime import CancelledError, write_json_atomic
from .trainer import train


def _print(payload: dict) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def cmd_audit(args: argparse.Namespace) -> None:
    _print(audit_model_dir(args.model_dir, args.fold, args.checkpoint))


def cmd_train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    manifest = train(config)
    # train() has returned, so its network/optimizer can be reclaimed before a
    # fresh full network is constructed for runtime-identity verification.
    gc.collect()
    try:
        manifest = verify_exported_model(
            manifest["model_dir"],
            fold=config["model"].get("fold", 0),
        )
    except Exception as exc:
        _finish_training_status(
            config,
            status="failed",
            phase="runtime_verification_failed",
            error="{}: {}".format(type(exc).__name__, exc),
        )
        raise
    _finish_training_status(
        config,
        status="completed",
        phase="completed",
        model_dir=str(manifest.get("model_dir") or ""),
        manifest=str(
            Path(str(manifest.get("model_dir") or ""))
            / "finetune_manifest.json"
        ),
    )
    _print(manifest)


def _finish_training_status(
    config: dict, status: str, phase: str, **values: object
) -> None:
    path_value = config.get("training", {}).get("status_path")
    if not path_value:
        return
    path = Path(path_value)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = {
            "schema_version": "nninteractive_finetune_status.v1",
        }
    payload.update(values)
    payload["status"] = status
    payload["phase"] = phase
    payload["updated_at_epoch"] = time.time()
    write_json_atomic(path, payload)


def cmd_verify_runtime(args: argparse.Namespace) -> None:
    _print(
        verify_exported_model(
            args.model_dir,
            fold=args.fold,
            checkpoint_name=args.checkpoint,
        )
    )


def cmd_evaluate(args: argparse.Namespace) -> None:
    _print(
        evaluate_model(
            args.model_dir,
            args.manifest,
            args.output,
            [int(value) for value in args.label_values.split(",")],
            fold=args.fold,
            checkpoint_name=args.checkpoint,
            clicks=args.clicks,
            max_cases=args.max_cases,
            device=args.device,
            training_goal=args.training_goal,
            initial_mask_probability=args.initial_mask_probability,
            provided_initial_mask_probability=(
                args.provided_initial_mask_probability
            ),
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Audit an nnInteractive checkpoint")
    audit.add_argument("--model-dir", required=True)
    audit.add_argument("--fold", default="0")
    audit.add_argument("--checkpoint", default="checkpoint_final.pth")
    audit.set_defaults(func=cmd_audit)

    train_parser = subparsers.add_parser("train", help="Run task adaptation")
    train_parser.add_argument("--config", required=True)
    train_parser.set_defaults(func=cmd_train)

    verify = subparsers.add_parser(
        "verify-runtime",
        help="Reload an exported checkpoint and verify effective parameters",
    )
    verify.add_argument("--model-dir", required=True)
    verify.add_argument("--fold", default="0")
    verify.add_argument("--checkpoint", default="checkpoint_final.pth")
    verify.set_defaults(func=cmd_verify_runtime)

    evaluate = subparsers.add_parser(
        "evaluate", help="Evaluate a model through the real nnInteractive session"
    )
    evaluate.add_argument("--model-dir", required=True)
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--label-values", default="1")
    evaluate.add_argument("--fold", default="0")
    evaluate.add_argument("--checkpoint", default="checkpoint_final.pth")
    evaluate.add_argument("--clicks", type=int, default=5)
    evaluate.add_argument("--max-cases", type=int, default=0)
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument(
        "--training-goal",
        choices=("general", "start_empty", "refine_existing"),
        default="general",
    )
    evaluate.add_argument(
        "--initial-mask-probability", type=float, default=0.5
    )
    evaluate.add_argument(
        "--provided-initial-mask-probability", type=float, default=0.7
    )
    evaluate.set_defaults(func=cmd_evaluate)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
        return 0
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        return 130
    except CancelledError as exc:
        print(str(exc), file=sys.stderr)
        return 130
    except Exception as exc:
        print("{}: {}".format(type(exc).__name__, exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
