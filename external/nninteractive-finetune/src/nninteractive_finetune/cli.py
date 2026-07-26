"""Command line interface for independent fine-tuning and verification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .evaluate import evaluate_model
from .model import audit_model_dir
from .runtime import CancelledError
from .trainer import train


def _print(payload: dict) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def cmd_audit(args: argparse.Namespace) -> None:
    _print(audit_model_dir(args.model_dir, args.fold, args.checkpoint))


def cmd_train(args: argparse.Namespace) -> None:
    _print(train(load_config(args.config)))


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
