#!/usr/bin/env python3
"""Backward-compatible alias for the maintained evaluation entry point.

Older notes referred to ``scripts/eval.py``.  The implementation now lives in
``scripts/evaluate_model.py`` so that the filename explains that it evaluates a
checkpoint on a materialized dataset.  Keep this thin alias to avoid broken
automation or copied commands; do not add evaluation logic here.
"""

from evaluate_model import main


if __name__ == "__main__":
    main()
