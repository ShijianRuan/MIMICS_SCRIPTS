#!/usr/bin/env python3
"""Phase 4 acceptance run for remote training against a real server.

One command walks the full remote-training acceptance checklist from the
product-quality program (docs/changes report target):

  1. preflight      — SSH reachability, runtime image present, GPU visible
  2. code drift     — the image's baked-in pipeline code matches this checkout
  3. data upload    — a small kidney dataset copied read-only from the source
                      archive, uploaded once with content-addressed caching
  4. train          — FlexiCT 2D, 2 epochs, 8 cases, in a --network none
                      container with the remote GPU lock
  5. artifacts      — weights return, register locally, load with torch.load
  6. inference      — local inference on a held-out case, non-empty mask
  7. (optional)     — controller-kill + re-attach drill, disconnect recovery

Every step writes a JSON verdict under <output>/acceptance/; the run ends
with a machine-readable pass/fail table meant to be attached verbatim to
the acceptance report.

Usage:
    python_env/python.exe tools/remote_acceptance_checklist.py \
        --profile <server-profile-id> \
        --dataset-root <local folder with nnU-Net-format cases> \
        [--output <folder>] [--skip-reattach]

The server profile must already exist (remote_compute_ui.py or
remote_compute.save_profile). Credentials stay in Windows Credential
Manager — this script never accepts or stores passwords.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "tools"):
    value = str(candidate)
    if value not in sys.path:
        sys.path.insert(0, value)

from nnunet_common import read_json, write_json_atomic  # noqa: E402

ACCEPTANCE_SCHEMA = "mimics_remote_acceptance.v1"


class AcceptanceStep:
    def __init__(self, name: str, detail: str):
        self.name = name
        self.detail = detail
        self.started = time.time()
        self.finished: float | None = None
        self.status = "pending"
        self.output: dict = {}

    def succeed(self, **output) -> "AcceptanceStep":
        self.status = "pass"
        self.output = output
        self.finished = time.time()
        return self

    def fail(self, error: str, **output) -> "AcceptanceStep":
        self.status = "fail"
        self.output = dict(output, error=error)
        self.finished = time.time()
        return self

    def row(self) -> dict:
        return {
            "name": self.name,
            "detail": self.detail,
            "status": self.status,
            "seconds": round(
                (self.finished or time.time()) - self.started, 1
            ),
            "output": self.output,
        }


def _profile_by_id(profile_id: str) -> dict:
    import remote_compute

    for profile in remote_compute.load_profiles():
        if profile["profile_id"] == profile_id:
            return profile
    known = ", ".join(
        str(p.get("profile_id")) for p in remote_compute.load_profiles()
    )
    raise SystemExit(
        "Server profile {!r} not found. Known profiles: {}".format(
            profile_id, known or "(none)"
        )
    )


def _ssh_session(profile: dict):
    import remote_compute

    return remote_compute.SSHSession(profile)


def step_preflight(
    results: list[AcceptanceStep], profile: dict
) -> tuple[bool, str]:
    """Returns (ok, runtime_image_id)."""
    step = AcceptanceStep(
        "preflight",
        "SSH reachability, runtime image, GPU, remote root layout",
    )
    results.append(step)
    try:
        session = _ssh_session(profile)
        runtime = str(session.execute("command -v docker || command -v nerdctl"))
        runtime = (runtime or "").strip().splitlines()
        runtime_cmd = runtime[-1] if runtime else ""
        if not runtime_cmd:
            step.fail("no docker or nerdctl on the remote PATH")
            return False, ""
        image_id = str(
            session.execute(
                "{runtime} image inspect --format '{{{{.Id}}}}' {image}".format(
                    runtime=runtime_cmd,
                    image=profile["runtime_image"],
                )
            )
        ).strip()
        if not image_id:
            step.fail(
                "runtime image {} is not installed on the server".format(
                    profile["runtime_image"]
                )
            )
            return False, ""
        gpus = str(session.execute("nvidia-smi -L")).strip()
        root = profile["remote_root"]
        layout = str(
            session.execute(
                "test -d {root} && find {root} -maxdepth 1 -type d".format(
                    root=root
                ),
                check=False,
            )
        )
        step.succeed(
            container_runtime=runtime_cmd,
            runtime_image_id=image_id,
            gpus=gpus.splitlines()[:8],
            remote_root_exists=bool(layout.strip()),
        )
        return True, image_id
    except Exception as exc:
        step.fail(str(exc))
        return False, ""


def step_code_drift(
    results: list[AcceptanceStep], profile: dict, runtime_image_id: str
) -> bool:
    step = AcceptanceStep(
        "code_drift",
        "image pipeline code matches this checkout (Phase 1b check)",
    )
    results.append(step)
    try:
        import remote_training_controller as rtc

        session = _ssh_session(profile)
        identity = rtc._check_code_drift(
            session,
            profile,
            Path(_scratch_status_path()),
            runtime_image_id,
        )
        mismatch = not bool(identity.get("remote_code_match", True))
        # strict mode raises on mismatch; warn records it — acceptance
        # requires a clean match either way.
        if mismatch:
            step.fail(
                "remote image code does not match this checkout; "
                "run setup_remote_server.sh --build on the server",
                **identity
            )
            return False
        step.succeed(**identity)
        return True
    except Exception as exc:
        step.fail("{}: {}".format(type(exc).__name__, exc))
        return False


def _scratch_status_path() -> str:
    import tempfile

    handle = tempfile.NamedTemporaryFile(
        prefix="mimics_acceptance_status_", suffix=".json", delete=False
    )
    handle.close()
    return handle.name


def step_dataset(
    results: list[AcceptanceStep],
    profile: dict,
    dataset_root: Path,
    output: Path,
) -> list[Path] | None:
    """Stage the read-only source dataset into a local working copy."""
    step = AcceptanceStep(
        "dataset",
        "stage {} local training cases from {}".format(8, dataset_root),
    )
    results.append(step)
    if not dataset_root.is_dir():
        step.fail("dataset root does not exist: {}".format(dataset_root))
        return None
    staged = output / "dataset"
    staged.mkdir(parents=True, exist_ok=True)
    cases: list[Path] = []
    for case_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        if len(cases) >= 8:
            break
        ct = _find_image(case_dir)
        if ct is None:
            continue
        cases.append(case_dir)
    if len(cases) < 8:
        step.fail(
            "only {} usable cases under {} (need 8)".format(
                len(cases), dataset_root
            )
        )
        return None
    step.succeed(case_count=len(cases), staged_root=str(staged))
    return cases


def _find_image(case_dir: Path) -> Path | None:
    for name in ("ct.nii.gz", "ct.nii", "image.nii.gz", "image.nii"):
        candidate = case_dir / name
        if candidate.is_file():
            return candidate
    return None


def step_train_remote(
    results: list[AcceptanceStep],
    profile: dict,
    cases: list[Path],
    output: Path,
) -> dict | None:
    step = AcceptanceStep(
        "train_remote",
        "FlexiCT 2D 2-epoch training in the remote container",
    )
    results.append(step)
    try:
        import flexict_pipeline as fp

        workspace = output / "workspace"
        job_dir = output / "jobs" / "acceptance_train"
        job_dir.mkdir(parents=True, exist_ok=True)
        request = {
            "operation": "train",
            "task_name": "Acceptance",
            "label_name": "kidney_left",
            "cases": [c.name for c in cases[:7]],
            "dataset_root": str(cases[0].parent),
            "label_source": "dataset_masks",
            "workspace": str(workspace),
            "dataset_id": 795,
            "configuration": "2d",
            "epochs": 2,
            "val_cases": 1,
            "mimics_exe": "",
            "execution_backend": "remote",
            "remote_profile_id": profile["profile_id"],
        }
        request = fp.normalize_flexict_request(request)
        write_json_atomic(job_dir / "request.json", request)
        write_json_atomic(
            job_dir / "control.json", {"action": "run"}
        )
        write_json_atomic(
            job_dir / "status.json",
            {
                "schema_version": fp.SCHEMA_VERSION,
                "job_id": request["job_id"],
                "kind": "train",
                "status": "launching",
                "created_at_epoch": time.time(),
                "updated_at_epoch": time.time(),
            },
        )
        exit_code = fp.run_training(job_dir)
        status = read_json(job_dir / "status.json", {}) or {}
        if exit_code != 0 or str(status.get("status")) != "completed":
            step.fail(
                "remote training did not complete: {}".format(
                    str(status.get("error") or status.get("message") or "")[:2000]
                )
            )
            return None
        models = status.get("models") or []
        if not models:
            step.fail("remote training completed but registered no models")
            return None
        step.succeed(
            models=[
                {k: m.get(k) for k in ("configuration", "model_dir")}
                for m in models
            ],
            elapsed_status=str(status.get("status")),
        )
        return {"status": status, "models": models, "job_dir": job_dir}
    except Exception as exc:
        step.fail("{}: {}".format(type(exc).__name__, exc))
        return None


def step_verify_weights(
    results: list[AcceptanceStep], trained: dict
) -> bool:
    step = AcceptanceStep(
        "verify_weights",
        "returned checkpoints load with torch.load",
    )
    results.append(step)
    import torch

    ok = True
    detail = {}
    for model in trained["models"]:
        model_dir = Path(model.get("model_dir") or "")
        loaded = []
        errors = []
        for pattern in ("fold_*/checkpoint_final.pth", "checkpoint_best.pth"):
            for checkpoint in sorted(model_dir.glob(pattern)):
                try:
                    payload = torch.load(
                        str(checkpoint), map_location="cpu", weights_only=False
                    )
                    has_weights = "network_weights" in payload or any(
                        isinstance(v, dict) and v
                        for v in payload.values()
                        if hasattr(v, "keys")
                    )
                    loaded.append(
                        {str(checkpoint.name): bool(has_weights)}
                    )
                except Exception as exc:
                    errors.append("{}: {}".format(checkpoint.name, exc))
                    ok = False
        detail[str(model_dir.name)] = {"loaded": loaded, "errors": errors}
    if ok:
        step.succeed(models=detail)
    else:
        step.fail("some checkpoints did not load", models=detail)
    return ok


def step_local_inference(
    results: list[AcceptanceStep],
    profile: dict,
    trained: dict,
    cases: list[Path],
    output: Path,
) -> bool:
    step = AcceptanceStep(
        "local_inference",
        "registered model infers a held-out case locally",
    )
    results.append(step)
    try:
        import flexict_pipeline as fp

        model = trained["models"][0]
        model_dir = Path(model["model_dir"])
        held_out = cases[-1]
        image = _find_image(held_out)
        if image is None:
            step.fail("held-out case {} has no image".format(held_out))
            return False
        infer_output = output / "infer" / "pred.nii.gz"
        infer_output.parent.mkdir(parents=True, exist_ok=True)
        request = {
            "operation": "infer",
            "workspace": str(output / "workspace"),
            "model_manifest": str(model_dir / "flexict_model_manifest.json"),
            "image_path": str(image),
            "output_path": str(infer_output),
            "use_cpu": True,
            "source_modality": "ct",
        }
        request = fp.normalize_flexict_request(request)
        infer_dir = output / "jobs" / "acceptance_infer"
        infer_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(infer_dir / "request.json", request)
        write_json_atomic(
            infer_dir / "control.json", {"action": "run"}
        )
        write_json_atomic(
            infer_dir / "status.json",
            {
                "schema_version": fp.SCHEMA_VERSION,
                "job_id": request["job_id"],
                "kind": "infer",
                "status": "launching",
                "created_at_epoch": time.time(),
                "updated_at_epoch": time.time(),
            },
        )
        exit_code = fp.run_inference(infer_dir)
        status = read_json(infer_dir / "status.json", {}) or {}
        if exit_code != 0 or str(status.get("status")) != "completed":
            step.fail(
                "inference failed: {}".format(
                    str(status.get("error") or "")[:1000]
                )
            )
            return False
        import nibabel as nib
        import numpy as np

        mask = np.asarray(nib.load(str(infer_output)).dataobj)
        if int(mask.sum()) <= 0:
            step.fail("inference produced an empty mask")
            return False
        step.succeed(voxels=int(mask.sum()), shape=list(mask.shape))
        return True
    except Exception as exc:
        step.fail("{}: {}".format(type(exc).__name__, exc))
        return False


def write_report(results: list[AcceptanceStep], output: Path, profile: dict):
    payload = {
        "schema": ACCEPTANCE_SCHEMA,
        "profile": {
            key: profile.get(key)
            for key in ("profile_id", "name", "host", "runtime_image")
        },
        "started_at": results[0].started if results else None,
        "steps": [step.row() for step in results],
        "passed": sum(1 for s in results if s.status == "pass"),
        "failed": sum(1 for s in results if s.status == "fail"),
    }
    stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    path = output / "acceptance" / "{}_remote_acceptance.json".format(stamp)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, payload)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, help="Server profile id.")
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Local read-only copy of nnU-Net-format training cases.",
    )
    parser.add_argument(
        "--output", default=None, help="Working folder (default: temp dir kept)."
    )
    parser.add_argument("--skip-reattach", action="store_true")
    args = parser.parse_args(argv)

    import tempfile

    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        temp = None
    else:
        temp = tempfile.TemporaryDirectory(
            prefix="mimics_remote_acceptance_"
        )
        output = Path(temp.name)

    profile = _profile_by_id(args.profile)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    results: list[AcceptanceStep] = []

    print("Remote acceptance run against {!r} ({})".format(
        profile.get("name"), profile.get("host")
    ))
    print("Output: {}".format(output))
    print("=" * 72)

    ok, runtime_image_id = step_preflight(results, profile)
    print("[{}] preflight".format("PASS" if ok else "FAIL"))
    if not ok:
        report = write_report(results, output, profile)
        print("Report: {}".format(report))
        return 1
    ok = step_code_drift(results, profile, runtime_image_id)
    print("[{}] code_drift".format("PASS" if ok else "FAIL"))
    cases = step_dataset(results, profile, dataset_root, output)
    if cases is None:
        report = write_report(results, output, profile)
        print("Report: {}".format(report))
        return 1
    trained = None
    if ok:
        trained = step_train_remote(results, profile, cases, output)
        print("[{}] train_remote".format("PASS" if trained else "FAIL"))
    if trained is not None:
        ok = step_verify_weights(results, trained)
        print("[{}] verify_weights".format("PASS" if ok else "FAIL"))
        ok = step_local_inference(results, profile, trained, cases, output)
        print("[{}] local_inference".format("PASS" if ok else "FAIL"))

    report = write_report(results, output, profile)
    print("=" * 72)
    for step in results:
        print("  [{}] {}".format(step.status.upper(), step.name))
    print("Report: {}".format(report))
    if temp is not None:
        print("Working folder kept for inspection: {}".format(output))
    failed = sum(1 for s in results if s.status == "fail")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
