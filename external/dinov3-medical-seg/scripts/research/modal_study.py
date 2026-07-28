"""Modal execution entry points for the reproducible multi-organ study.

This file is intentionally separate from the Mimics integration. It runs
offline training/evaluation workloads only; no clinical image is sent to a web
endpoint and no interactive GUI session is involved.

Before invoking it, upload de-identified source data and the locally approved
DINOv3 ViT-B weights to the named Modal Volumes documented in the study guide.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_CONTAINER_PROJECT_ROOT = Path("/opt/dinov3-medical-seg")
# Modal mounts the entry script at /root in the container, so its location does
# not retain the local scripts/research hierarchy used while building the image.
PROJECT_ROOT = (
    _CONTAINER_PROJECT_ROOT
    if _CONTAINER_PROJECT_ROOT.is_dir()
    else Path(__file__).resolve().parents[2]
)
# The entry script is mounted at /root inside the container, so the project tree
# is not on sys.path by default. Add it so in-process helpers (build_benchmark,
# protocol) import the same code the subprocess runners execute.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import modal
except ImportError:  # Allows local static checks without cloud credentials.
    modal = None


def _prepare_tasks_with_checkpoint(tasks, run_task, commit) -> None:
    """Materialize each task, committing after every one.

    Modal CPU workers are preemptible and Volume writes are only visible after a
    ``commit``. Committing per task (instead of once after all five) means a
    preempted retry keeps every organ that already finished, rather than losing
    the whole batch. ``run_task`` and ``commit`` are injected so the ordering
    contract can be tested without cloud credentials.
    """
    for task in tasks:
        run_task(task)
        commit()


def _poll_handle(handle) -> bool:
    """Return True when a spawned job has settled (succeeded OR failed).

    ``FunctionCall.get(timeout=0)`` polls immediately: it raises ``TimeoutError``
    while the job is still running, returns the value on success, and re-raises
    the remote exception on failure. A failed job must free its concurrency slot
    rather than crash the local driver — the job already wrote ``failed.json``
    remotely and the selector's completeness gate will refuse to proceed on it.
    """
    try:
        handle.get(timeout=0)
        return True
    except TimeoutError:
        return False
    except Exception as exc:  # remote job failed; slot is done, keep the batch alive
        print("WARNING: spawned job failed (continuing batch): {}".format(exc))
        return True


def _bounded_spawn(specs, launch, poll, max_concurrency=4, poll_interval=5.0) -> int:
    """Spawn jobs for each spec, keeping at most ``max_concurrency`` in flight.

    ``.remote()`` is a BLOCKING call, so a naive loop over specs runs them
    serially and pins the local driver for the whole matrix. This scheduler uses
    a non-blocking ``launch`` (e.g. Modal ``.spawn()`` returning a handle) plus a
    ``poll(handle) -> done`` check, bounding concurrency so Volume v1 never sees
    more than a few concurrent committers (recommended < ~5). Returns the number
    of completed jobs. ``launch``/``poll`` are injected for testability.
    """
    if int(max_concurrency) < 1:
        raise ValueError("max_concurrency must be >= 1, got {}".format(max_concurrency))
    import time

    pending = list(specs)
    in_flight = []
    completed = 0
    while pending or in_flight:
        while pending and len(in_flight) < int(max_concurrency):
            in_flight.append(launch(pending.pop(0)))
        still = []
        for handle in in_flight:
            if poll(handle):
                completed += 1
            else:
                still.append(handle)
        in_flight = still
        if in_flight and float(poll_interval) > 0:
            time.sleep(float(poll_interval))
    return completed


def _phase_job_specs(plan, phase: str, confirmation_ids=None) -> list:
    """Return the per-job spec list for a study phase, one dict per GPU job.

    Each spec drives one ``run_single`` call — the resumable, per-job-commit
    path. Screening spans every candidate on the screening fold at K=5; a
    measurement run just slices the returned list. Confirmation expands the
    given winner/reference ids over confirmation folds, shot counts and seeds.
    Pure and side-effect free so the fan-out can be tested without Modal.
    """
    study = plan["study"]
    candidates = plan["candidates"]
    if phase == "regime":
        from src.research.regime import REGIME_CELL_IDS
        screen_fold = int(study["screening_fold"])
        screen_seed = int((study.get("screening_training_seeds") or [0])[0])
        tasks, seen = [], set()
        for candidate in candidates:
            if candidate["task"] not in seen:
                seen.add(candidate["task"])
                tasks.append(candidate["task"])
        return [
            {"phase": "regime", "task": task, "cell_id": cell_id,
             "fold": screen_fold, "support_count": 5, "training_seed": screen_seed}
            for task in tasks for cell_id in REGIME_CELL_IDS
        ]
    if phase == "screen":
        screen_fold = int(study["screening_fold"])
        screen_seeds = [int(value) for value in study.get("screening_training_seeds", [])]
        if not screen_seeds:
            raise ValueError("Study must define non-empty screening_training_seeds")
        specs = []
        for candidate in candidates:
            for training_seed in screen_seeds:
                specs.append({
                    "phase": "screen",
                    "candidate_id": candidate["id"],
                    "fold": screen_fold,
                    "support_count": 5,
                    "training_seed": training_seed,
                })
        return specs
    if phase == "confirm":
        if not confirmation_ids:
            raise ValueError("confirm phase requires a non-empty confirmation_ids list")
        confirm_seeds = [int(value) for value in study.get("confirmation_training_seeds", [])]
        if not confirm_seeds:
            raise ValueError("Study must define non-empty confirmation_training_seeds")
        folds = [int(value) for value in study["confirmation_folds"]]
        shots = [int(value) for value in study["shot_counts"]]
        specs = []
        for candidate_id in sorted(set(confirmation_ids)):
            for fold in folds:
                for support_count in shots:
                    for training_seed in confirm_seeds:
                        specs.append({
                            "phase": "confirm",
                            "candidate_id": candidate_id,
                            "fold": fold,
                            "support_count": support_count,
                            "training_seed": training_seed,
                        })
        return specs
    raise ValueError("phase must be 'screen' or 'confirm', got {}".format(phase))


def _confirmation_ids(plan, selected_ids, weak_tasks=None, degenerate_tasks=None) -> list:
    """Confirmation candidate ids: selected winners plus references, minus reserved tasks.

    A reserved task is either degenerate (Tier-0 collapse) or weak (screen winner
    below a guardrail). Both are excluded per-task rather than blocking the whole
    confirm run, so a clearly passing task (e.g. aorta) still advances to the
    confirmation folds even when a sibling task (e.g. scapula) is reserved for a
    rescue study. Pure and side-effect free so the exclusion is unit-testable.
    """
    reserved = set(weak_tasks or []) | set(degenerate_tasks or [])
    task_by_candidate = {candidate["id"]: candidate["task"] for candidate in plan["candidates"]}
    reference_ids = [candidate["id"] for candidate in plan["candidates"] if candidate.get("is_reference", False)]
    return sorted(
        candidate_id
        for candidate_id in (set(selected_ids) | set(reference_ids))
        if task_by_candidate.get(candidate_id) not in reserved
    )


# Pinned to the versions verified locally against the uploaded ViT-B weights.
# The weights are ``model_type: dinov3_vit`` / ``DINOv3ViTModel``; the
# ``DINOv3ViTBackbone`` class that ``src/models/backbone.py`` imports at module
# load only exists in transformers 5.x. An earlier ``transformers>=4.56,<5`` pin
# installed a 4.x without that class and crashed every GPU function on import.
# transformers 5.x requires torch>=2.4, so the widened torch pin stays valid.
IMAGE_PIP_PACKAGES = [
    "torch>=2.5,<2.13",
    "torchvision>=0.20,<0.28",
    "transformers>=5.12,<6",
    "numpy>=1.24",
    "scipy>=1.10",
    "nibabel>=5.2",
    "pyyaml>=6.0",
    "tqdm>=4.65",
    "tensorboard>=2.14",
]


if modal is not None:
    app = modal.App("dinov3-medical-fewshot-study")
    source_volume = modal.Volume.from_name("dinov3-medical-source", create_if_missing=True)
    benchmark_volume = modal.Volume.from_name("dinov3-medical-benchmark", create_if_missing=True)
    model_volume = modal.Volume.from_name("dinov3-medical-models", create_if_missing=True)
    # Focused studies write to a v2 Volume so distinct jobs can commit safely
    # at higher concurrency. The original v1 result Volume is left untouched.
    result_volume = modal.Volume.from_name("dinov3-medical-results-v2", create_if_missing=True, version=2)
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git")
        .uv_pip_install(*IMAGE_PIP_PACKAGES)
        # Reduce CUDA fragmentation for trainable-backbone (LoRA/adapter/full)
        # full-volume forwards on the 24GB A10; pairs with slice_batch_size=1.
        .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
        .add_local_dir(str(PROJECT_ROOT / "src"), remote_path="/opt/dinov3-medical-seg/src", copy=True)
        .add_local_dir(str(PROJECT_ROOT / "scripts"), remote_path="/opt/dinov3-medical-seg/scripts", copy=True)
        .add_local_dir(str(PROJECT_ROOT / "config"), remote_path="/opt/dinov3-medical-seg/config", copy=True)
    )

    def _run(command: list[str]) -> None:
        subprocess.run(command, cwd="/opt/dinov3-medical-seg", check=True)

    def _runtime_plan(benchmark_subdir: str, result_subdir: str, plan_name: str) -> Path:
        import yaml

        plan_path = Path("/opt/dinov3-medical-seg/config/research") / plan_name
        plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
        study = plan["study"]
        base_config = Path(str(study["base_config"]))
        if not base_config.is_absolute():
            base_config = Path("/opt/dinov3-medical-seg") / base_config
        study["base_config"] = str(base_config)
        study["benchmark_root"] = "/vol/benchmark/{}".format(benchmark_subdir)
        study["results_root"] = "/vol/results/{}".format(result_subdir)
        runtime_plan = Path("/tmp/runtime_study.yaml")
        runtime_plan.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
        return runtime_plan

    @app.function(
        image=image,
        cpu=4,
        memory=16384,
        timeout=4 * 60 * 60,
        volumes={"/vol/source": source_volume, "/vol/benchmark": benchmark_volume},
    )
    def prepare_benchmark(source_subdir: str, benchmark_subdir: str, folds: int = 4, seed: int = 20260711):
        """Materialize folds in the benchmark Volume, committing after each task.

        The five tasks run in one process so the module-level CT intensity cache
        dedupes decompression of a CT shared across organs (the previous run's
        pathology was rescanning the same compressed volume for every task). The
        benchmark Volume is still committed per task, so a preempted CPU worker
        keeps finished organs; ``build_benchmark`` skips folds whose
        ``manifest.json`` already exists, so a rerun resumes instead of
        recomputing.
        """
        from src.research.protocol import TASKS
        from scripts.research.prepare_totalseg_benchmark import build_benchmark

        tasks = sorted(TASKS)
        source = Path("/vol/source") / source_subdir
        output = Path("/vol/benchmark") / benchmark_subdir

        def _run_task(task: str) -> None:
            print("=== preparing task: {} ===".format(task), flush=True)
            build_benchmark(
                source=source,
                output=output,
                tasks=[task],
                support_pool_size=5,
                folds=folds,
                seed=seed,
            )

        _prepare_tasks_with_checkpoint(
            tasks,
            run_task=_run_task,
            commit=benchmark_volume.commit,
        )

    @app.function(
        image=image,
        cpu=2,
        memory=16384,
        timeout=30 * 60,
        volumes={"/vol/models": model_volume},
    )
    def verify_image():
        """Cheap CPU check that the image can import and load the DINOv3 model.

        This isolates the transformers/torch version contract from GPU cost: it
        catches an image-pin regression (e.g. a transformers build without
        ``DINOv3ViTBackbone``) in a couple of CPU minutes instead of on a paid
        A10 during preflight.
        """
        import transformers

        print("transformers version: {}".format(transformers.__version__), flush=True)
        from transformers import DINOv3ViTBackbone  # noqa: F401  fails fast if the pin regressed
        from src.models.backbone import DINOv3Backbone

        backbone = DINOv3Backbone("/vol/models/dinov3-vitb16", out_indices=[2, 5, 8, 11], freeze=True)
        info = {
            "transformers": transformers.__version__,
            "patch_size": int(backbone.patch_size),
            "embed_dim": int(backbone.embed_dim),
            "num_layers": int(backbone.num_layers),
        }
        print("loaded DINOv3 backbone: {}".format(json.dumps(info)), flush=True)
        return info

    @app.function(
        image=image,
        gpu="A10",
        cpu=4,
        memory=24576,
        timeout=2 * 60 * 60,
        volumes={"/vol/models": model_volume, "/vol/results": result_volume},
    )
    def preflight_trainability(result_subdir: str):
        """Fail before the matrix if any PEFT path does not receive gradients."""
        base = Path("/opt/dinov3-medical-seg/config/research/ct_fewshot_base.yaml")
        output = Path("/vol/results") / result_subdir / "preflight" / "trainability.json"
        _run([
            sys.executable,
            "scripts/research/verify_trainability.py",
            "--config", str(base),
            "--model-path", "/vol/models/dinov3-vitb16",
            "--method", "frozen",
            "--method", "lora",
            "--method", "adapter",
            "--method", "full",
            "--device", "cuda",
            "--output", str(output),
        ])
        result_volume.commit()

    @app.function(
        image=image,
        gpu="A10",
        cpu=8,
        memory=32768,
        timeout=4 * 60 * 60,
        volumes={
            "/vol/benchmark": benchmark_volume,
            "/vol/models": model_volume,
            "/vol/results": result_volume,
        },
    )
    def run_single(
        phase: str,
        candidate_id: str,
        fold: int,
        support_count: int,
        training_seed: int,
        benchmark_subdir: str,
        result_subdir: str,
        selection_subpath: str = "selected_candidates.json",
        plan_name: str = "multi_organ_study.yaml",
    ):
        """Run one resumable GPU experiment, including train and evaluation."""
        runtime_plan = _runtime_plan(benchmark_subdir, result_subdir, plan_name)
        command = [
            sys.executable,
            "scripts/research/run_ablations.py",
            "--plan", str(runtime_plan),
            "--phase", phase,
            "--candidate-id", candidate_id,
            "--fold", str(fold),
            "--support-count", str(support_count),
            "--training-seed", str(training_seed),
            "--model-path", "/vol/models/dinov3-vitb16",
        ]
        if phase == "confirm":
            command.extend(["--selection", "/vol/results/{}/{}".format(result_subdir, selection_subpath)])
        # Screen/confirm build on the Tier-0 selected regime when present.
        if phase in ("screen", "confirm"):
            regime_path = Path("/vol/results") / result_subdir / "selected_regime.json"
            if regime_path.is_file():
                command.extend(["--regime", str(regime_path)])
        _run(command)
        result_volume.commit()

    @app.function(
        image=image,
        gpu="A10",
        cpu=8,
        memory=32768,
        timeout=8 * 60 * 60,
        volumes={
            "/vol/benchmark": benchmark_volume,
            "/vol/models": model_volume,
            "/vol/results": result_volume,
        },
    )
    def regime_run(task: str, cell_id: str, benchmark_subdir: str, result_subdir: str,
                   plan_name: str = "multi_organ_study.yaml"):
        """Run one Tier-0 regime cell (train+eval) with a per-job commit."""
        runtime_plan = _runtime_plan(benchmark_subdir, result_subdir, plan_name)
        _run([
            sys.executable,
            "scripts/research/run_ablations.py",
            "--plan", str(runtime_plan),
            "--phase", "regime",
            "--tasks", task,
            "--cell", cell_id,
            "--model-path", "/vol/models/dinov3-vitb16",
        ])
        result_volume.commit()

    @app.function(
        image=image,
        gpu="A10",
        cpu=8,
        memory=32768,
        timeout=8 * 60 * 60,
        volumes={
            "/vol/benchmark": benchmark_volume,
            "/vol/models": model_volume,
            "/vol/results": result_volume,
        },
    )
    def run_two_stage_single(
        task: str,
        fold: int,
        support_count: int,
        training_seed: int,
        benchmark_subdir: str,
        result_subdir: str,
        plan_name: str = "multi_organ_study.yaml",
    ):
        """Run the explicit coarse-to-fine ablation for a tiny-structure task."""
        import yaml

        plan = yaml.safe_load((Path("/opt/dinov3-medical-seg/config/research") / plan_name).read_text(encoding="utf-8"))
        base_path_config = Path("/opt/dinov3-medical-seg") / plan["study"]["base_config"]
        from src.utils.config import load_config
        base = load_config(str(base_path_config), {})
        base["model"]["model_path"] = "/vol/models/dinov3-vitb16"
        base.setdefault("training", {})["seed"] = int(training_seed)
        base_path = Path("/tmp/two_stage_base.yaml")
        base_path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
        _run([
            sys.executable,
            "scripts/research/run_two_stage.py",
            "--base-config", str(base_path),
            "--data-root", "/vol/benchmark/{}/{}/fold_{:02d}".format(benchmark_subdir, task, fold),
            "--task", task,
            "--support-count", str(support_count),
            "--training-seed", str(training_seed),
            "--output-root", "/vol/results/{}/cascade/{}/fold_{:02d}/k{}/seed_{}".format(
                result_subdir, task, fold, support_count, training_seed
            ),
        ])
        result_volume.commit()

    @app.function(
        image=image,
        gpu="A10",
        cpu=4,
        memory=24576,
        timeout=3 * 60 * 60,
        volumes={
            "/vol/benchmark": benchmark_volume,
            "/vol/models": model_volume,
            "/vol/results": result_volume,
        },
    )
    def evaluate_inference_modes_single(
        candidate_id: str,
        task: str,
        fold: int,
        support_count: int,
        training_seed: int,
        study_id: str,
        benchmark_subdir: str,
        result_subdir: str,
    ):
        """Run post-training inference ablations without retraining a model."""
        run_root = (
            Path("/vol/results") / result_subdir / "confirm" / candidate_id
            / "fold_{:02d}".format(fold) / "k{}".format(support_count) / "seed_{}".format(training_seed)
        )
        checkpoint = (
            Path("/vol/results") / result_subdir / "artifacts" / "research_runs" /
            study_id / "confirm" / candidate_id /
            "fold_{:02d}".format(fold) / "k{}".format(support_count) / "seed_{}".format(training_seed)
            / "checkpoints" / "best_model.pth"
        )
        output_path = run_root / "inference_modes.json"
        if output_path.is_file():
            print("Skipping completed inference modes: {}".format(output_path))
            return
        _run([
            sys.executable,
            "scripts/research/evaluate_inference_modes.py",
            "--config", str(run_root / "config.yaml"),
            "--checkpoint", str(checkpoint),
            "--data-root", "/vol/benchmark/{}/{}/fold_{:02d}".format(benchmark_subdir, task, fold),
            "--output", str(output_path),
            "--mode", "single",
            "--mode", "ap_tta",
            "--mode", "multiscale",
            "--mode", "ap_tta_multiscale",
        ])
        result_volume.commit()

    @app.function(
        image=image,
        gpu="A10",
        cpu=8,
        memory=32768,
        timeout=3 * 60 * 60,
        volumes={
            "/vol/benchmark": benchmark_volume,
            "/vol/models": model_volume,
            "/vol/results": result_volume,
        },
    )
    def diagnose_inference_gap_single(
        candidate_id: str,
        task: str,
        fold: int,
        study_id: str,
        result_subdir: str,
        benchmark_subdir: str,
        thresholds: str = "0.3,0.4,0.5,0.6,0.7,0.8,0.9",
    ):
        """Isolate the screen train-val vs deploy inference gap on one checkpoint.

        Reuses the retained screen checkpoint (no retraining) to compare whole-ROI
        inference against a sliding-window threshold sweep, answering whether the
        deployable path can recover the train-val Dice by tuning the decision rule.
        """
        run_root = (
            Path("/vol/results") / result_subdir / "screen" / candidate_id
            / "fold_{:02d}".format(fold) / "k5" / "seed_{}".format(20260711)
        )
        checkpoint = (
            Path("/vol/results") / result_subdir / "artifacts" / "research_runs" /
            study_id / "screen" / candidate_id /
            "fold_{:02d}".format(fold) / "k5" / "seed_{}".format(20260711)
            / "checkpoints" / "best_model.pth"
        )
        output_path = run_root / "inference_gap.json"
        if output_path.is_file():
            print("Skipping completed inference-gap diagnosis: {}".format(output_path))
            return
        if not checkpoint.is_file():
            raise SystemExit("Screen checkpoint missing: {}".format(checkpoint))
        _run([
            sys.executable,
            "scripts/research/diagnose_inference_gap.py",
            "--config", str(run_root / "config.yaml"),
            "--checkpoint", str(checkpoint),
            "--data-root", "/vol/benchmark/{}/{}/fold_{:02d}".format(benchmark_subdir, task, fold),
            "--output", str(output_path),
            "--split", "Val",
            "--thresholds", thresholds,
        ])
        result_volume.commit()

    @app.function(image=image, cpu=1, volumes={"/vol/results": result_volume})
    def select_screening(result_subdir: str, selection_subpath: str = "selected_candidates.json",
                         plan_name: str = "multi_organ_study.yaml"):
        """Persist screen-fold choices before any confirmation function starts."""
        command = [
            sys.executable,
            "scripts/research/select_screening_winners.py",
            "--results-root", "/vol/results/{}".format(result_subdir),
            "--output", "/vol/results/{}/{}".format(result_subdir, selection_subpath),
            "--plan", "/opt/dinov3-medical-seg/config/research/{}".format(plan_name),
        ]
        regime_path = Path("/vol/results") / result_subdir / "selected_regime.json"
        if regime_path.is_file():
            command.extend(["--regime", str(regime_path)])
        _run(command)
        result_volume.commit()
        payload = json.loads((Path("/vol/results") / result_subdir / selection_subpath).read_text(encoding="utf-8"))
        # Weak tasks are reserved per-task (excluded from confirmation like a
        # degenerate task), NOT a whole-run block: a passing task must still
        # confirm even when a sibling task's screen winner is below guardrail.
        if payload.get("weak_tasks"):
            print("Weak screen task(s) reserved for a rescue study (excluded from confirm): {}".format(
                payload["weak_tasks"]))
        return {
            "selected_candidate_ids": payload["selected_candidate_ids"],
            "weak_tasks": payload.get("weak_tasks", []),
        }

    @app.function(image=image, cpu=1, volumes={"/vol/results": result_volume})
    def read_selected_ids(result_subdir: str, selection_subpath: str = "selected_candidates.json"):
        """Read the screen selection from the Volume for confirm fan-out.

        Returns both the selected candidate ids and the weak-task list so the
        driver can reserve weak tasks per-task rather than blocking the whole
        confirmation run.
        """
        path = Path("/vol/results") / result_subdir / selection_subpath
        if not path.is_file():
            raise SystemExit(
                "Selection file missing: {}. Run --action select after screening.".format(path)
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "dinov3_medical_screen_selection.v2":
            raise SystemExit("Selection is stale or predates complete-matrix validation: {}".format(path))
        return {
            "selected_candidate_ids": payload["selected_candidate_ids"],
            "weak_tasks": payload.get("weak_tasks", []),
        }

    @app.function(image=image, cpu=1, volumes={"/vol/results": result_volume})
    def select_regime(result_subdir: str, regime_subpath: str = "selected_regime.json"):
        """Pick one sampling+loss regime per task from the Tier-0 pre-screen.

        Passes the full expected task set so the selector fails closed unless the
        complete task x cell matrix is present with finite metrics.
        """
        import yaml

        plan = yaml.safe_load(
            Path("/opt/dinov3-medical-seg/config/research/multi_organ_study.yaml").read_text(encoding="utf-8")
        )
        tasks, seen = [], set()
        for candidate in plan["candidates"]:
            if candidate["task"] not in seen:
                seen.add(candidate["task"])
                tasks.append(candidate["task"])
        command = [
            sys.executable,
            "scripts/research/select_regime_winners.py",
            "--results-root", "/vol/results/{}".format(result_subdir),
            "--output", "/vol/results/{}/{}".format(result_subdir, regime_subpath),
        ]
        for task in tasks:
            command.extend(["--expected-task", task])
        _run(command)
        result_volume.commit()
        return json.loads((Path("/vol/results") / result_subdir / regime_subpath).read_text(encoding="utf-8"))

    @app.function(image=image, cpu=1, volumes={"/vol/results": result_volume})
    def read_selected_regime(result_subdir: str, regime_subpath: str = "selected_regime.json"):
        """Read the selected regime map from the results Volume."""
        path = Path("/vol/results") / result_subdir / regime_subpath
        if not path.is_file():
            raise SystemExit("Missing {}. Run --action select-regime first.".format(path))
        return json.loads(path.read_text(encoding="utf-8"))

    @app.local_entrypoint()
    def main(
        action: str = "screen",
        source_subdir: str = "totalsegmentator/source",
        # Reuse validated benchmark folds; focused results use a new v2 Volume.
        benchmark_subdir: str = "totalseg_multi_organ_v3",
        result_subdir: str = "focused_5shot_v1",
        selection_subpath: str = "selected_candidates.json",
        max_runs: int = 0,
        max_concurrency: int = 8,
        cascade_task: str = "adrenal_gland_right",
        plan_name: str = "focused_5shot_study.yaml",
        candidate_id: str = "",
    ):
        # Bounded async fan-out: .spawn() is non-blocking; _bounded_spawn keeps at
        # most max_concurrency jobs in flight. Results live on Volume v2 and
        # every job owns a disjoint directory, so eight writers are safe.
        def _spawn_regime(spec):
            return regime_run.spawn(spec["task"], spec["cell_id"], benchmark_subdir, result_subdir, plan_name)

        def _spawn_run_single(phase):
            def launch(spec):
                return run_single.spawn(
                    phase, spec["candidate_id"], spec["fold"], spec["support_count"],
                    spec["training_seed"], benchmark_subdir, result_subdir, selection_subpath, plan_name,
                )
            return launch

        def _poll(handle):
            return _poll_handle(handle)

        if action == "prepare":
            prepare_benchmark.remote(source_subdir, benchmark_subdir)
        elif action == "regime":
            import yaml

            plan = yaml.safe_load((PROJECT_ROOT / "config/research" / plan_name).read_text(encoding="utf-8"))
            specs = _phase_job_specs(plan, "regime")
            if max_runs:
                specs = specs[:max_runs]
            print("Dispatching {} regime job(s) at concurrency {}".format(len(specs), max_concurrency))
            _bounded_spawn(specs, _spawn_regime, _poll, max_concurrency=max_concurrency)
        elif action == "select-regime":
            payload = select_regime.remote(result_subdir)
            print("selected regime: {}".format(json.dumps(payload.get("selected_by_task", {}))))
            if payload.get("degenerate_tasks"):
                print("WARNING degenerate tasks (do NOT proceed to screen): {}".format(payload["degenerate_tasks"]))
        elif action == "cascade":
            # Diagnostic / ablation: run the coarse-to-fine cascade for one tiny
            # task on the screening fold. Reports coarse_detection_rate so a
            # degenerate single-stage target (e.g. adrenal) can be assessed
            # before designing a dedicated localizer.
            import yaml

            plan = yaml.safe_load((PROJECT_ROOT / "config/research" / plan_name).read_text(encoding="utf-8"))
            study = plan["study"]
            task = cascade_task
            fold = int(study["screening_fold"])
            seed = int((study.get("screening_training_seeds") or [0])[0])
            print("Running cascade diagnostic: {} fold_{:02d} k5 seed_{}".format(task, fold, seed))
            run_two_stage_single.remote(task, fold, 5, seed, benchmark_subdir, result_subdir, plan_name)
        elif action == "diagnose-inference":
            # Isolate the screen train-val vs deploy inference gap for one
            # candidate (whole-ROI vs sliding-window threshold sweep) on the
            # retained checkpoint. Requires --candidate-id.
            import yaml

            if not candidate_id:
                raise ValueError("diagnose-inference requires --candidate-id")
            plan = yaml.safe_load((PROJECT_ROOT / "config/research" / plan_name).read_text(encoding="utf-8"))
            study = plan["study"]
            task_by_candidate = {c["id"]: c["task"] for c in plan["candidates"]}
            if candidate_id not in task_by_candidate:
                raise ValueError("Unknown candidate: {}".format(candidate_id))
            fold = int(study["screening_fold"])
            print("Diagnosing inference gap: {} fold_{:02d}".format(candidate_id, fold))
            diagnose_inference_gap_single.remote(
                candidate_id, task_by_candidate[candidate_id], fold,
                study["id"], result_subdir, benchmark_subdir,
            )
        elif action == "verify-image":
            info = verify_image.remote()
            print("image verification: {}".format(json.dumps(info)))
        elif action == "preflight":
            preflight_trainability.remote(result_subdir)
        elif action == "screen":
            # Per-job fan-out: each job is its own Modal function with a per-job
            # commit and evaluation.json resume, so a preempted worker never
            # loses finished jobs (the single-commit run_phase path did). Cap
            # with --max-runs / max_runs to measure one job before the matrix.
            import yaml

            plan = yaml.safe_load((PROJECT_ROOT / "config/research" / plan_name).read_text(encoding="utf-8"))
            # Drop candidates whose task collapsed in Tier-0 (reserved, not blocked).
            regime = read_selected_regime.remote(result_subdir) if plan["study"].get("require_regime", True) else {}
            degenerate = set(regime.get("degenerate_tasks", []))
            task_by_candidate = {c["id"]: c["task"] for c in plan["candidates"]}
            specs = [s for s in _phase_job_specs(plan, "screen")
                     if task_by_candidate.get(s["candidate_id"]) not in degenerate]
            if candidate_id:
                specs = [spec for spec in specs if spec["candidate_id"] == candidate_id]
                if not specs:
                    raise ValueError("Unknown or excluded screen candidate: {}".format(candidate_id))
            if degenerate:
                print("Excluding degenerate task(s) from screen (reserved): {}".format(sorted(degenerate)))
            if max_runs:
                specs = specs[:max_runs]
            print("Dispatching {} screen job(s) at concurrency {}".format(len(specs), max_concurrency))
            _bounded_spawn(specs, _spawn_run_single("screen"), _poll, max_concurrency=max_concurrency)
        elif action == "confirm":
            import yaml

            plan = yaml.safe_load((PROJECT_ROOT / "config/research" / plan_name).read_text(encoding="utf-8"))
            if not selection_subpath:
                raise ValueError("confirm requires selection_subpath pointing at selected_candidates.json")
            # The selection file lives on the results Volume; read it remotely.
            selection = read_selected_ids.remote(result_subdir, selection_subpath)
            selected_ids = selection["selected_candidate_ids"]
            weak_tasks = selection.get("weak_tasks", [])
            regime = read_selected_regime.remote(result_subdir) if plan["study"].get("require_regime", True) else {}
            degenerate = sorted(set(regime.get("degenerate_tasks", [])))
            confirmation_ids = _confirmation_ids(
                plan, selected_ids, weak_tasks=weak_tasks, degenerate_tasks=degenerate)
            reserved = sorted(set(weak_tasks) | set(degenerate))
            if reserved:
                print("Excluding reserved task(s) from confirm (degenerate {} / weak {}): {}".format(
                    degenerate, sorted(weak_tasks), reserved))
            specs = _phase_job_specs(plan, "confirm", confirmation_ids)
            if candidate_id:
                specs = [spec for spec in specs if spec["candidate_id"] == candidate_id]
                if not specs:
                    raise ValueError("Unknown or excluded confirm candidate: {}".format(candidate_id))
            if max_runs:
                specs = specs[:max_runs]
            print("Dispatching {} confirm job(s) for ids {} at concurrency {}".format(
                len(specs), confirmation_ids, max_concurrency))
            _bounded_spawn(specs, _spawn_run_single("confirm"), _poll, max_concurrency=max_concurrency)
        elif action == "select":
            select_screening.remote(result_subdir, selection_subpath, plan_name)
        elif action == "full":
            # A single command drives the dependency graph. Jobs are separate
            # Modal functions so each checkpoint is committed and the command
            # can be rerun safely after a laptop/network interruption.
            import yaml

            prepare_benchmark.remote(source_subdir, benchmark_subdir)
            preflight_trainability.remote(result_subdir)
            plan = yaml.safe_load((PROJECT_ROOT / "config/research" / plan_name).read_text(encoding="utf-8"))
            study = plan["study"]
            # Tier-0: choose each organ's sampling+loss regime before the factor
            # screen, so every screen candidate builds on a trainable baseline.
            _bounded_spawn(_phase_job_specs(plan, "regime"), _spawn_regime, _poll, max_concurrency=max_concurrency)
            regime_payload = select_regime.remote(result_subdir)
            degenerate = set(regime_payload.get("degenerate_tasks", []))
            if degenerate:
                print("Excluding degenerate task(s) from screen/confirm; rescue study required: {}".format(
                    sorted(degenerate)))
            if regime_payload.get("needs_more_seeds_tasks"):
                print("NOTE: near-tie regimes need more seeds before final claims: {}".format(
                    regime_payload["needs_more_seeds_tasks"]))
            # Screen: one run_single per (candidate, screen seed) via the shared
            # spec builder, so the full path and the standalone screen action
            # cannot drift apart.
            task_by_candidate = {candidate["id"]: candidate["task"] for candidate in plan["candidates"]}
            screen_specs = [
                spec for spec in _phase_job_specs(plan, "screen")
                if task_by_candidate.get(spec["candidate_id"]) not in degenerate
            ]
            _bounded_spawn(screen_specs, _spawn_run_single("screen"), _poll, max_concurrency=max_concurrency)
            selection = select_screening.remote(result_subdir, selection_subpath, plan_name)
            selected_ids = selection["selected_candidate_ids"]
            weak_tasks = selection.get("weak_tasks", [])
            confirmation_ids = _confirmation_ids(
                plan, selected_ids, weak_tasks=weak_tasks, degenerate_tasks=sorted(degenerate))
            _bounded_spawn(
                _phase_job_specs(plan, "confirm", confirmation_ids),
                _spawn_run_single("confirm"), _poll, max_concurrency=max_concurrency,
            )
            # Inference-mode selection is intentionally evaluated only on the
            # screen winner; the frozen reference is already confirmed for
            # effect-size comparison and does not multiply this post-hoc grid.
            # Weak/degenerate winners are reserved, so post-hoc inference tuning
            # only runs for confirmed winners.
            reserved_tasks = set(weak_tasks) | set(degenerate)
            task_by_candidate = {candidate["id"]: candidate["task"] for candidate in plan["candidates"]}
            inference_specs = [
                {"candidate_id": candidate_id, "fold": int(fold)}
                for candidate_id in selected_ids
                if task_by_candidate.get(candidate_id) not in reserved_tasks
                for fold in study["confirmation_folds"]
            ]
            _bounded_spawn(
                inference_specs,
                lambda s: evaluate_inference_modes_single.spawn(
                    s["candidate_id"], task_by_candidate[s["candidate_id"]], s["fold"], 5,
                    int(study["inference_seed"]), study["id"], benchmark_subdir, result_subdir),
                _poll, max_concurrency=max_concurrency,
            )
            # Two-stage inference is tested independently for the tiny adrenal
            # task. It never receives label-derived inference crops.
            confirm_seeds = [int(value) for value in study.get("confirmation_training_seeds", [])]
            two_stage_specs = [
                {"fold": int(fold), "support_count": int(support_count), "training_seed": training_seed}
                for fold in study["confirmation_folds"]
                for support_count in study["shot_counts"]
                for training_seed in confirm_seeds
            ]
            _bounded_spawn(
                two_stage_specs,
                lambda s: run_two_stage_single.spawn(
                    "adrenal_gland_right", s["fold"], s["support_count"], s["training_seed"],
                    benchmark_subdir, result_subdir, plan_name),
                _poll, max_concurrency=max_concurrency,
            )
        else:
            raise ValueError("action must be prepare, regime, select-regime, cascade, verify-image, preflight, screen, select, confirm, or full")
else:
    app = None


if __name__ == "__main__" and modal is None:
    raise SystemExit(
        "Modal is not installed in this interpreter. Install it in a separate runner environment: pip install modal"
    )
