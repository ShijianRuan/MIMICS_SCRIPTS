#!/usr/bin/env python3
"""Run nnInteractive fine-tuning from a JSON/YAML configuration."""

import sys

from nninteractive_finetune.cli import main

if __name__ == "__main__":
    sys.argv.insert(1, "train")
    raise SystemExit(main())
