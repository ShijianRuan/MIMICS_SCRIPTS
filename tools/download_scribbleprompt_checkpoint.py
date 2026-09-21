#!/usr/bin/env python3
"""Download the official ScribblePrompt-UNet checkpoint for offline packaging."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "integrations"
    / "ScribblePrompt"
    / "checkpoints"
    / "ScribblePrompt_unet_v1_nf192_res128.pt"
)
URL = (
    "https://www.dropbox.com/scl/fi/pnw88n05irnv5z1snlklr/"
    "ScribblePrompt_unet_v1_nf192_res128.pt"
    "?rlkey=dr8xvkf0wj2r082h1zzpcmz5o&dl=1"
)


def download(output: Path) -> Path:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=output.name + ".", suffix=".download", dir=str(output.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        request = urllib.request.Request(URL, headers={"User-Agent": "Mimics-Script/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        if temporary.stat().st_size < 1024 * 1024:
            raise RuntimeError("Downloaded file is too small to be the ScribblePrompt checkpoint.")
        os.replace(str(temporary), str(output))
        return output
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    path = download(args.output)
    print("ScribblePrompt checkpoint ready: {} ({:.1f} MB)".format(path, path.stat().st_size / 1e6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
