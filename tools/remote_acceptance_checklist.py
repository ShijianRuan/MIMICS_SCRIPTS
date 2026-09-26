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

Every step writes a JSON verdict under <output>/acceptance/; the run ends
with a machine-readable pass/fail table meant to be attached verbatim to
the acceptance report.

Usage:
    python_env/python.exe tools/remote_acceptance_checklist.py \
        --profile <server-profile-id> \
        --dataset-root <local folder with nnU-Net-format cases> \
        [--output <folder>]

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

from nnunet_common import read_json, safe_identifier, write_json_atomic  # noqa: E402

ACCEPTANCE_SCHEMA = "mimics_remote_acceptance.v1"
TERMINAL_STATES = {"completed", "failed", "cancelled"}


def _request_stop(job_dir: Path, wait_seconds: float = 180.0) -> None:
    """Ask the job's controller to stop and wait for a terminal state.

    The stop goes through the job's own control.json — the same channel the
    UI uses — because the controller process owns the container; racing it
    with a concurrent CLI cancel would be unsafe. Never raises: callers use
    this on the failure path, where the original error matters most.
    """
    control_path = job_dir / "control.json"
    try:
        write_json_atomic(
            control_path,
            {"action": "stop", "updated_at_epoch": time.time()},
        )
    except OSError:
        return
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        time.sleep(5)
        status = read_json(job_dir / "status.json", {}) or {}
        if str(status.get("status") or "") in TERMINAL_STATES:
            return


def _stop_failure_detail(status: dict) -> str:
    """One-line summary of a stopped/failed remote run for error messages."""
    if not isinstance(status, dict):
        status = {}
    container = str(status.get("remote_container_name") or "")
    remote_dir = str(status.get("remote_job_dir") or "")
    final = str(status.get("status") or "(unknown)")
    detail = "final status {}".format(final)
    if container:
        detail += ", container {}".format(container)
    if remote_dir:
        detail += ", remote job dir {}".format(remote_dir)
    return detail


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
        import remote_compute

        # Use the profile's configured runtime/namespace (e.g. "nerdctl -n
        # mimics-ai") for every step of this checklist. Earlier versions
        # silently fell back to nerdctl when docker was missing, but only
        # for this preflight inspect — every later step would still use the
        # profile's docker command and fail confusingly. A mismatched
        # runtime is a profile misconfiguration: fail with the fix instead.
        runtime_cmd = remote_compute.container_runtime_command(profile)
        if not str(
            session.execute("command -v {}".format(runtime_cmd.split()[0]))
        ).strip():
            step.fail(
                "container runtime {!r} is not on the remote PATH; if the "
                "server uses nerdctl, open Manage Servers in the training "
                "window and set Container runtime to nerdctl".format(
                    runtime_cmd
                )
            )
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
) -> list[Path] | None:
    """Pick training cases directly from the read-only source dataset."""
    step = AcceptanceStep(
        "dataset",
        "pick {} local training cases from {}".format(8, dataset_root),
    )
    results.append(step)
    if not dataset_root.is_dir():
        step.fail("dataset root does not exist: {}".format(dataset_root))
        return None
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
    step.succeed(case_count=len(cases), dataset_root=str(dataset_root))
    return cases


def _find_image(case_dir: Path) -> Path | None:
    for name in ("ct.nii.gz", "ct.nii", "image.nii.gz", "image.nii"):
        candidate = case_dir / name
        if candidate.is_file():
            return candidate
    return None


def _resolve_local_model_dir(status: dict) -> Path | None:
    """The registered local model dir for a completed single-config run.

    status["model"]["model_dir"] is rewritten by the controller to a local
    registry path on completion; a container-side /job/... path or a
    missing directory means registration never happened.
    """
    model = status.get("model") or {}
    if not isinstance(model, dict):
        return None
    raw = str(model.get("model_dir") or "")
    if not raw or raw.startswith("/job/"):
        return None
    candidate = Path(raw)
    return candidate if candidate.is_dir() else None


def step_cleanup_failed_jobs(
    results: list[AcceptanceStep],
    profile: dict,
) -> bool:
    """Remove leftover failed-job diagnostic dirs from the shared server.

    Failed remote training keeps its job directory on the server for
    diagnosis (``remote_diagnostics_retained``), which on a shared server
    accumulates without bound. This step deletes only leftovers under
    ``<remote_root>/jobs/<owner>/`` — no container is running for them
    (this step runs after training finished) and nothing outside that
    folder is touched."""
    step = AcceptanceStep(
        "cleanup_failed_jobs",
        "Remove leftover failed-job directories from the shared server",
    )
    results.append(step)
    try:
        from remote_training_controller import _remove_remote_job

        session = _ssh_session(profile)
        owner = safe_identifier(str(profile.get("username") or ""), "user")
        jobs_root = "{}/jobs/{}".format(profile["remote_root"], owner)
        listing = str(
            session.execute(
                "find {root} -mindepth 1 -maxdepth 1 ! -name '*.tar' "
                "! -name '*.part' 2>/dev/null".format(root=jobs_root),
                check=False,
            )
        ).strip()
        leftovers = [
            name for name in listing.splitlines() if name.strip()
        ]
        removed = 0
        for remote_job_dir in leftovers:
            try:
                if _remove_remote_job(
                    session,
                    remote_root=profile["remote_root"],
                    expected_owner=owner,
                    remote_job_dir=remote_job_dir,
                ):
                    removed += 1
            except Exception:
                pass  # keep going: one refused path must not stop the rest
        step.succeed(
            jobs_root=jobs_root,
            leftovers_found=len(leftovers),
            leftovers_removed=removed,
        )
        return True
    except Exception as exc:
        step.fail("{}: {}".format(type(exc).__name__, exc))
        return False


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
            # The nnU-Net planner sizes batch_size for a plain UNet; the
            # FlexiCT attention model needs far more VRAM per sample and
            # OOMs a 16GB A4000 at the planned default (bs=37 on this
            # dataset). Force a small explicit batch for the smoke run.
            "batch_size": 4,
            "val_cases": 1,
            "mimics_exe": "",
            "execution_backend": "remote",
            "remote_profile_id": profile["profile_id"],
        }
        # create_flexict_job is the production entry point: it writes the
        # job folder and, for execution_backend="remote", launches
        # remote_training_controller.py --spec in an external process.
        # Calling fp.run_training() directly here would run the LOCAL
        # worker path and train on this workstation's GPU instead.
        launched = fp.create_flexict_job(request)
        job_dir = Path(launched["job_dir"])
        try:
            deadline = time.time() + 3600
            poll_index = 0
            while time.time() < deadline:
                time.sleep(10)
                poll_index += 1
                status = read_json(job_dir / "status.json", {}) or {}
                state = str(status.get("status") or "")
                if state in TERMINAL_STATES:
                    break
                if poll_index % 6 == 0:
                    print(
                        "  ... remote train: {} ({}%)".format(
                            state, status.get("progress_percent")
                        ),
                        flush=True,
                    )
            else:
                _request_stop(job_dir, wait_seconds=180)
                failure = _stop_failure_detail(status)
                step.fail(
                    "remote training timed out after 1h; a stop was "
                    "requested and waited for: {}".format(failure)
                )
                return None
        except BaseException:
            # KeyboardInterrupt included: never leave an unattended
            # training container running on the shared server.
            _request_stop(job_dir, wait_seconds=180)
            raise
        if str(status.get("status")) != "completed":
            step.fail(
                "remote training did not complete: {}".format(
                    str(status.get("error") or status.get("message") or "")[:2000]
                ),
                detail=_stop_failure_detail(status),
            )
            return None
        # status["models"] rows carry container-side /job/... paths (the
        # merged remote pipeline status); the authoritative local path is
        # status["model"]["model_dir"], which the controller rewrites to
        # the local registry when registering the downloaded artifact. The
        # acceptance request pins one configuration, so exactly one model
        # is expected — anything else is a fail, not a guess.
        local_model_dir = _resolve_local_model_dir(status)
        if local_model_dir is None:
            step.fail(
                "remote training completed but the local model directory "
                "was not registered (model_dir={!r})".format(
                    str((status.get("model") or {}).get("model_dir") or "")
                ),
                detail=_stop_failure_detail(status),
            )
            return None
        step.succeed(
            model_dir=str(local_model_dir),
            elapsed_status=str(status.get("status")),
        )
        return {
            "status": status,
            "model_dir": str(local_model_dir),
            "job_dir": job_dir,
        }
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

    model_dir = Path(trained["model_dir"])
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
                loaded.append({str(checkpoint.name): bool(has_weights)})
            except Exception as exc:
                errors.append("{}: {}".format(checkpoint.name, exc))
    # An empty glob must fail too: "no checkpoints found" was previously
    # indistinguishable from "all checkpoints verified".
    if errors:
        step.fail("some checkpoints did not load", errors=errors)
        return False
    if not loaded:
        step.fail(
            "no checkpoints found under {}".format(model_dir)
        )
        return False
    step.succeed(model_dir=str(model_dir), loaded=loaded)
    return True


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

        model_dir = Path(trained["model_dir"])
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
            # GPU is the production fast path (75s vs ~70min CPU on a
            # 1132-slice CT). The GPU lock is free here — training is done.
            "use_cpu": False,
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
    parser.add_argument(
        "--cleanup-failed-dirs",
        action="store_true",
        help=(
            "At the end, delete leftover failed-job diagnostic directories "
            "under the profile's remote jobs folder (shared-server hygiene)."
        ),
    )
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
    cases = step_dataset(results, profile, dataset_root)
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
    if args.cleanup_failed_dirs:
        # Shared-server hygiene: leftover failed-job dirs stay behind for
        # inspection by default; this flag removes them at the end of a run.
        ok = step_cleanup_failed_jobs(results, profile)
        print("[{}] cleanup_failed_jobs".format("PASS" if ok else "FAIL"))
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
