#!/usr/bin/env python3
"""Isolated validation adapter for a user-supplied FFD reference DLL.

This developer tool is intentionally not used by the Mimics/IGAC runtime.  It
allows an authorized Windows user to compare the clean-room PyTorch FFD with a
compatible native ``FreeFormDeformation(int, void**)`` export.  The native call
runs in a child process because an access violation cannot be caught safely by
Python in the IGAC GUI process.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np


EXPORT_NAME = "FreeFormDeformation"


def _vector(values: Sequence[float], name: str) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError("{} requires exactly three coordinates.".format(name))
    return tuple(float(value) for value in values)


def _native_call(
    dll_path: Path,
    input_path: Path,
    output_path: Path,
    start_xyz: Sequence[float],
    end_xyz: Sequence[float],
    radius: float,
    control_points: int,
) -> None:
    if os.name != "nt":
        raise RuntimeError("The reference DLL can only be loaded on Windows.")
    points = np.ascontiguousarray(np.load(str(input_path)), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 1:
        raise ValueError("The input must be an N x 3 NumPy point array.")
    start = _vector(start_xyz, "start")
    end = _vector(end_xyz, "end")
    influence = max(1.0e-3, float(radius))
    controls = max(4, int(control_points))

    library = ctypes.CDLL(str(dll_path))
    try:
        function = getattr(library, EXPORT_NAME)
    except AttributeError as exc:
        raise RuntimeError("The DLL does not export {}.".format(EXPORT_NAME)) from exc
    function.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
    function.restype = None

    point_count = ctypes.c_int(int(points.shape[0]))
    positions = (ctypes.c_double * 6)(*(start + end))
    move_count = ctypes.c_int(1)
    bounds = (ctypes.c_double * 6)(
        start[0] - 2.0 * influence,
        start[0] + 2.0 * influence,
        start[1] - 2.0 * influence,
        start[1] + 2.0 * influence,
        start[2] - 2.0 * influence,
        start[2] + 2.0 * influence,
    )
    control_grid = (ctypes.c_int * 3)(controls, controls, controls)
    arguments = (ctypes.c_void_p * 6)(
        ctypes.c_void_p(points.ctypes.data),
        ctypes.cast(ctypes.byref(point_count), ctypes.c_void_p),
        ctypes.cast(positions, ctypes.c_void_p),
        ctypes.cast(ctypes.byref(move_count), ctypes.c_void_p),
        ctypes.cast(bounds, ctypes.c_void_p),
        ctypes.cast(control_grid, ctypes.c_void_p),
    )
    function(len(arguments), arguments)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=output_path.name + ".",
        suffix=".tmp.npy",
        dir=str(output_path.parent),
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.save(str(temporary), points)
        os.replace(str(temporary), str(output_path))
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass


def _isolated_call(args: argparse.Namespace) -> int:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--dll",
        str(Path(args.dll).resolve()),
        "--input",
        str(Path(args.input).resolve()),
        "--output",
        str(Path(args.output).resolve()),
        "--start",
        *(str(value) for value in args.start),
        "--end",
        *(str(value) for value in args.end),
        "--radius",
        str(args.radius),
        "--control-points",
        str(args.control_points),
    ]
    creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) if os.name == "nt" else 0
    completed = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=creationflags,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "no diagnostic output").strip()
        raise RuntimeError(
            "The isolated reference FFD process failed with code {}: {}".format(
                completed.returncode, detail
            )
        )
    print(
        json.dumps(
            {
                "status": "completed",
                "output": str(Path(args.output).resolve()),
                "note": "Validation-only third-party DLL call; not used by IGAC runtime.",
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", required=True, help="Authorized x64 Release FFD DLL")
    parser.add_argument("--input", required=True, help="Input N x 3 float point array (.npy)")
    parser.add_argument("--output", required=True, help="Output deformed point array (.npy)")
    parser.add_argument("--start", nargs=3, type=float, required=True)
    parser.add_argument("--end", nargs=3, type=float, required=True)
    parser.add_argument("--radius", type=float, required=True)
    parser.add_argument("--control-points", type=int, default=10)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.worker:
        _native_call(
            Path(args.dll),
            Path(args.input),
            Path(args.output),
            args.start,
            args.end,
            args.radius,
            args.control_points,
        )
        return 0
    return _isolated_call(args)


if __name__ == "__main__":
    raise SystemExit(main())
