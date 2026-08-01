#!/usr/bin/env python3
"""Portable command-line workflow for the bundled nnU-Net framework.

The controller runs every heavyweight unit in an owned child process. This
keeps environment changes process-scoped and gives external callers stable
status, log, cancellation, and error contracts.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Iterable, Sequence, Union

import SetEnvionmentVariables


SCRIPT_DIR = Path(__file__).resolve().parent
TERMINAL_STATES = {"completed", "failed", "cancelled"}
DEFAULT_STAGES = ("convert", "preprocess", "train", "predict")


def _toml_module():
    try:
        import tomllib

        return tomllib
    except ModuleNotFoundError:
        try:
            import tomli

            return tomli
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Python below 3.11 requires the 'tomli' package to read workflow TOML files."
            ) from exc


def _module(name: str):
    return importlib.import_module(name)


def _get_preprocess_configuration(config):
    return config["PREPROCESS"]["configuration"]


def _dataset_id_from_name(dataset_name: str) -> int:
    token = Path(dataset_name).name.split("_")[0]
    digits = "".join(ch for ch in token if ch.isdigit())
    if not digits:
        raise ValueError("Cannot parse a dataset ID from: {}".format(dataset_name))
    return int(digits)


def _resolve_file(value: Union[str, Path], base: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (base / path).resolve()


def _resolve_directory_value(value: Union[str, Path], base: Path) -> str:
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else base / path).resolve())


def _write_json_atomic(path: Union[str, Path], payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        "{}.{}.tmp".format(destination.name, uuid.uuid4().hex)
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        last_error = None
        for attempt in range(8):
            try:
                os.replace(str(temporary), str(destination))
                return
            except PermissionError as exc:
                last_error = exc
                time.sleep(0.05 * (attempt + 1))
        raise last_error or RuntimeError("Atomic JSON publication failed")
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_json(path: Union[str, Path], default=None):
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return default


def _update_status(path: Union[str, Path], **values) -> dict[str, Any]:
    status = _read_json(path, {}) or {}
    status.update(values)
    status["updated_at_epoch"] = time.time()
    _write_json_atomic(path, status)
    return status


def _classify_error(exc: BaseException, stage: str) -> str:
    message = str(exc).lower()
    if isinstance(exc, FileNotFoundError):
        return "input_missing"
    if isinstance(exc, PermissionError):
        return "permission_denied"
    if isinstance(exc, (ValueError, KeyError, TypeError)):
        return "configuration_invalid"
    if isinstance(exc, MemoryError) or "out of memory" in message or "cuda oom" in message:
        return "out_of_memory"
    if "cuda" in message or "gpu" in message:
        return "gpu_failure"
    return "{}_failed".format(stage)


def ReadConfigFile(file_path, write_snapshot=True, create_directories=True):
    """Resolve a TOML workflow without changing user-level environment files."""
    toml = _toml_module()
    config_file = _resolve_file(file_path, SCRIPT_DIR)
    if not config_file.is_file():
        raise FileNotFoundError("Configuration file does not exist: {}".format(config_file))
    with config_file.open("rb") as handle:
        config = toml.load(handle)
    config["_config_file"] = str(config_file)
    config_dir = config_file.parent

    model_file = _resolve_file(config["MODEL"]["segment_model_file"], config_dir)
    with model_file.open("rb") as handle:
        model_map = toml.load(handle)
    names = list(config["MODEL"]["segment_list_name"])
    datasets = list(config["MODEL"]["train_dataset"])
    if len(names) != len(datasets):
        raise ValueError(
            "MODEL.segment_list_name and MODEL.train_dataset must have equal lengths."
        )
    missing_names = [name for name in names if name not in model_map]
    if missing_names:
        raise KeyError("Unknown ModelMap sections: {}".format(", ".join(missing_names)))
    config["MODEL"]["segment_list"] = [model_map[name] for name in names]
    config["MODEL"]["segment_model_file"] = str(model_file)

    paths = config["PATHS"]
    labeled_root = Path(_resolve_directory_value(paths["labeled_path"], config_dir))
    paths["labeled_path"] = str(labeled_root)
    paths["dataset_path"] = [
        str((labeled_root / dataset).resolve())
        for dataset in paths["labeled_dataset"]
    ]
    train_root = Path(_resolve_directory_value(paths["train_path"], config_dir))
    train_path = (train_root / paths["train_project"]).resolve()
    if create_directories:
        train_path.mkdir(parents=True, exist_ok=True)
    paths["train_path"] = str(train_path)
    for key in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        root = train_path / paths[key]
        if create_directories:
            root.mkdir(parents=True, exist_ok=True)
        paths[key] = str(root.resolve())
    paths["nnUNet_path"] = [
        str((Path(paths["nnUNet_raw"]) / dataset).resolve())
        for dataset in datasets
    ]

    _normalize_gpu_ids(config, len(datasets))
    _resolved_training_values(config)
    if write_snapshot:
        snapshot = train_path / "config_resolved_{}.json".format(
            "{}_{}".format(time.strftime("%Y%m%dT%H%M%S"), uuid.uuid4().hex[:8])
        )
        _write_json_atomic(snapshot, config)
        config["_resolved_snapshot"] = str(snapshot)
    return config


def environment_variables(config):
    return SetEnvionmentVariables.environment_values(
        {
        "nnUNet_raw": config["PATHS"]["nnUNet_raw"],
        "nnUNet_preprocessed": config["PATHS"]["nnUNet_preprocessed"],
        "nnUNet_results": config["PATHS"]["nnUNet_results"],
        }
    )


def set_environment_variables(config):
    return SetEnvionmentVariables.set_process_environment(
        environment_variables(config)
    )


def _indices(config, dataset_index=None) -> Iterable[int]:
    count = len(config["MODEL"]["train_dataset"])
    if dataset_index is None:
        return range(count)
    index = int(dataset_index)
    if not 0 <= index < count:
        raise IndexError("Dataset index {} is outside 0..{}.".format(index, count - 1))
    return (index,)


def convertdata(config, dataset_index=None):
    action = _module("Action1_ConvertLabeledToTrainData")
    preprocess_config = config["PREPROCESS"]
    orientation = preprocess_config.get("orientation")
    if isinstance(orientation, str) and not orientation.strip():
        orientation = None
    datasets = config["PATHS"]["dataset_path"]
    for index in _indices(config, dataset_index):
        action.convert(
            datasets,
            Path(config["PATHS"]["nnUNet_path"][index]),
            config["MODEL"]["segment_list"][index],
            preprocess_config.get("spacing") or None,
            target_orientation=orientation,
            modality=config["COMMON"]["modality"],
            image_reader_writer=preprocess_config["reorientaion"],
        )


def preprocess(config, dataset_index=None):
    action = _module("Action2_PlanAndPreprocess")
    values = config["PREPROCESS"]
    for index in _indices(config, dataset_index):
        action.stage_preprocess(
            dataset_id=_dataset_id_from_name(config["MODEL"]["train_dataset"][index]),
            configuration=values["configuration"],
            num_processes=int(values["num_processes"]),
            target_spacing=values.get("spacing") or None,
            target_patch_size=values.get("patch_size") or None,
            target_batch_size=int(values.get("batch_size") or 0) or None,
        )


def _resolved_training_values(config):
    action = _module("Action3_Train")
    train_config = config["TRAIN"]
    trainer, epochs = action.resolve_epoch_trainer(
        str(train_config["trainer"]), int(train_config["epoch"])
    )
    return trainer, epochs


def _normalize_gpu_ids(config, count):
    value = config["GPU"]["gpu_id"]
    rows = list(value) if isinstance(value, list) else [value] * count
    if len(rows) != count:
        raise ValueError(
            "GPU.gpu_id count ({}) does not match dataset count ({}).".format(
                len(rows), count
            )
        )
    return [int(item) for item in rows]


def _get_single_gpu_id(config):
    rows = _normalize_gpu_ids(config, len(config["MODEL"]["train_dataset"]))
    return rows[0] if len(set(rows)) == 1 else None


def _is_multigpu(configs):
    rows = set()
    for config in configs:
        rows.update(_normalize_gpu_ids(config, len(config["MODEL"]["train_dataset"])))
    return len(rows) > 1


def train(config, dataset_index=None):
    action = _module("Action3_Train")
    train_config = config["TRAIN"]
    gpu_ids = _normalize_gpu_ids(config, len(config["MODEL"]["train_dataset"]))
    for index in _indices(config, dataset_index):
        action.stage_train(
            dataset_id=_dataset_id_from_name(config["MODEL"]["train_dataset"][index]),
            configuration=config["PREPROCESS"]["configuration"],
            fold=train_config["fold"],
            trainer=train_config["trainer"],
            plans=train_config["plans"],
            gpu_id=gpu_ids[index],
            epochs=int(train_config["epoch"]),
        )


def predict(config, dataset_index=None):
    action = _module("Action4_Predict")
    trainer, _ = _resolved_training_values(config)
    train_config = config["TRAIN"]
    prediction = config.get("PREDICT") or {}
    gpu_ids = _normalize_gpu_ids(config, len(config["MODEL"]["train_dataset"]))
    for index in _indices(config, dataset_index):
        dataset_name = config["MODEL"]["train_dataset"][index]
        action.stage_predict(
            dataset_id=_dataset_id_from_name(dataset_name),
            configuration=config["PREPROCESS"]["configuration"],
            fold=train_config["fold"],
            trainer=trainer,
            plans=train_config["plans"],
            dataset_folder_name=dataset_name,
            input_path=prediction.get("input_path") or None,
            output_path=prediction.get("output_path") or None,
            disable_tta=bool(prediction.get("disable_tta", True)),
            enable_stats=bool(prediction.get("enable_stats", False)),
            gpu_id=gpu_ids[index],
        )


def evaluation(config, dataset_index=None, aggregate=True):
    action1 = _module("Action1_ConvertLabeledToTrainData")
    action5 = _module("Action5_Evaluation")
    evaluation_config = config.get("EVALUATION") or {}
    result_files = []
    for index in _indices(config, dataset_index):
        root = Path(config["PATHS"]["nnUNet_path"][index])
        class_map = config["MODEL"]["segment_list"][index]
        if action1._is_grouped_class_map(class_map):
            _, class_map = action1._expand_grouped_class_map(class_map)
        result_files.append(
            action5.evaluate(
                root / (evaluation_config.get("gt_subdir") or "labelsTs"),
                root / (evaluation_config.get("pred_subdir") or "labelsTs_predicted"),
                class_map,
                root / (evaluation_config.get("eval_subdir") or "evaluation"),
            )
        )
    if aggregate and result_files:
        output = evaluation_config.get("aggregate_output_dir") or str(
            Path(config["PATHS"]["nnUNet_path"][0]).parent
        )
        action5.run_evaluation_aggregation(
            result_files,
            Path(output),
            evaluation_config.get("aggregate_output_basename") or "evaluation",
        )


def _execute_unit(stage: str, config: dict[str, Any], dataset_index=None):
    set_environment_variables(config)
    functions = {
        "convert": convertdata,
        "preprocess": preprocess,
        "train": train,
        "predict": predict,
        "evaluate": evaluation,
    }
    if stage not in functions:
        raise ValueError("Unsupported workflow stage: {}".format(stage))
    functions[stage](config, dataset_index=dataset_index)


def _stage_units(config: dict[str, Any], stages: Sequence[str]):
    count = len(config["MODEL"]["train_dataset"])
    units = []
    for stage in stages:
        if stage == "evaluate":
            units.append((stage, None, "all datasets"))
            continue
        for index, dataset in enumerate(config["MODEL"]["train_dataset"]):
            units.append((stage, index, dataset))
    return units


def _cancel_requested(control_path: Path) -> bool:
    control = _read_json(control_path, {}) or {}
    return str(control.get("action") or "").lower() in {"cancel", "stop"}


def request_cancel(control_file: Union[str, Path]) -> Path:
    """Publish a cancellation request without requiring manual JSON edits."""
    control_path = Path(control_file).expanduser().resolve()
    control = _read_json(control_path, {}) or {}
    control.update(
        {
            "action": "cancel",
            "requested_at_epoch": time.time(),
            "requested_by_pid": os.getpid(),
        }
    )
    _write_json_atomic(control_path, control)
    return control_path


def show_status(status_file: Union[str, Path]) -> dict[str, Any]:
    """Print and return the latest structured workflow status."""
    status_path = Path(status_file).expanduser().resolve()
    payload = _read_json(status_path, None)
    if not isinstance(payload, dict):
        raise FileNotFoundError(
            "Workflow status does not exist or is not valid JSON: {}".format(
                status_path
            )
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def _terminate_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        process.wait()


def _child_command(config_path, stage, dataset_index, result_path):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_stage",
        "--resolved-config",
        str(config_path),
        "--stage",
        stage,
        "--result-file",
        str(result_path),
    ]
    if dataset_index is not None:
        command.extend(["--dataset-index", str(dataset_index)])
    return command


def run_workflow_controller(
    config_file,
    stages=DEFAULT_STAGES,
    status_file=None,
    control_file=None,
    dry_run=False,
) -> int:
    try:
        config = ReadConfigFile(
            config_file,
            write_snapshot=not dry_run,
            create_directories=not dry_run,
        )
    except Exception as exc:
        if status_file:
            _update_status(
                Path(status_file).expanduser().resolve(),
                schema_version="nnunet_standalone_workflow.v1",
                status="failed",
                phase="configuration",
                message="The nnU-Net workflow configuration is invalid.",
                error="{}: {}".format(type(exc).__name__, exc),
                error_category=_classify_error(exc, "configuration"),
                traceback=traceback.format_exc(),
                completed_at_epoch=time.time(),
            )
        raise
    stages = tuple(str(value).lower() for value in stages)
    invalid = [value for value in stages if value not in {*DEFAULT_STAGES, "evaluate"}]
    if invalid:
        raise ValueError("Unsupported stages: {}".format(", ".join(invalid)))
    units = _stage_units(config, stages)
    if dry_run:
        print(json.dumps({"config": config["_config_file"], "units": units}, indent=2))
        return 0

    runtime_root = Path(config["PATHS"]["train_path"]) / ".nnunet_runtime"
    run_id = "workflow_{}_{}".format(time.strftime("%Y%m%dT%H%M%S"), uuid.uuid4().hex[:8])
    run_root = runtime_root / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    status_path = Path(status_file).expanduser().resolve() if status_file else run_root / "status.json"
    control_path = Path(control_file).expanduser().resolve() if control_file else run_root / "control.json"
    log_path = run_root / "workflow.log"
    resolved_path = run_root / "resolved_config.json"
    _write_json_atomic(resolved_path, config)
    if not control_path.exists():
        _write_json_atomic(control_path, {"action": "run", "created_at_epoch": time.time()})
    _update_status(
        status_path,
        schema_version="nnunet_standalone_workflow.v1",
        run_id=run_id,
        status="running",
        phase="starting",
        message="Starting the nnU-Net workflow.",
        config_path=str(config["_config_file"]),
        resolved_config_path=str(resolved_path),
        control_path=str(control_path),
        log_path=str(log_path),
        stages=list(stages),
        total_units=len(units),
        completed_units=0,
        progress_percent=0,
        controller_pid=os.getpid(),
        created_at_epoch=time.time(),
    )
    print("Status: {}".format(status_path))
    print("Control: {}".format(control_path))
    print("Log: {}".format(log_path))

    environment = os.environ.copy()
    environment.update(environment_variables(config))
    environment["PYTHONUNBUFFERED"] = "1"
    try:
        for unit_index, (stage, dataset_index, dataset_name) in enumerate(units, start=1):
            if _cancel_requested(control_path):
                raise InterruptedError("cancel")
            result_path = run_root / "result_{:03d}_{}.json".format(unit_index, stage)
            _update_status(
                status_path,
                status="running",
                phase=stage,
                message="Running {} for {}.".format(stage, dataset_name),
                current_stage=stage,
                current_dataset=dataset_name,
                unit_index=unit_index,
                progress_percent=int(100 * (unit_index - 1) / max(1, len(units))),
            )
            flags = 0
            popen_values = {}
            if os.name == "nt":
                flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            else:
                popen_values["start_new_session"] = True
            with log_path.open("ab") as log_handle:
                process = subprocess.Popen(
                    _child_command(resolved_path, stage, dataset_index, result_path),
                    cwd=str(SCRIPT_DIR),
                    stdin=subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    env=environment,
                    creationflags=flags,
                    **popen_values,
                )
                _update_status(status_path, worker_pid=process.pid)
                last_epoch = -1
                while process.poll() is None:
                    if _cancel_requested(control_path):
                        _terminate_process_tree(process)
                        raise InterruptedError("cancel")
                    if stage == "train":
                        try:
                            with log_path.open("rb") as reader:
                                reader.seek(0, os.SEEK_END)
                                reader.seek(max(0, reader.tell() - 131072), os.SEEK_SET)
                                tail = reader.read().decode("utf-8", "replace")
                            matches = re.findall(r"(?i)\bepoch\s*[:#]?\s*(\d+)\b", tail)
                            if matches:
                                epoch = max(int(value) for value in matches)
                                if epoch != last_epoch:
                                    total_epochs = int(config["TRAIN"]["epoch"])
                                    completed_epoch = min(total_epochs, epoch + 1)
                                    unit_progress = completed_epoch / max(1, total_epochs)
                                    _update_status(
                                        status_path,
                                        message="Training {}: epoch {} of {}.".format(
                                            dataset_name, completed_epoch, total_epochs
                                        ),
                                        current_epoch=completed_epoch,
                                        total_epochs=total_epochs,
                                        progress_percent=min(
                                            99,
                                            int(
                                                100
                                                * ((unit_index - 1) + unit_progress)
                                                / max(1, len(units))
                                            ),
                                        ),
                                    )
                                    last_epoch = epoch
                        except OSError:
                            pass
                    time.sleep(0.5)
            if _cancel_requested(control_path):
                raise InterruptedError("cancel")
            result = _read_json(result_path, {}) or {}
            if process.returncode != 0 or result.get("status") != "ok":
                error = result.get("error") or "Worker exited with code {}".format(
                    process.returncode
                )
                category = result.get("error_category") or "{}_failed".format(stage)
                _update_status(
                    status_path,
                    status="failed",
                    phase=stage,
                    message="The {} stage failed for {}.".format(stage, dataset_name),
                    error=error,
                    error_category=category,
                    traceback=result.get("traceback") or "",
                    worker_pid=None,
                    completed_at_epoch=time.time(),
                )
                return 1
            _update_status(
                status_path,
                completed_units=unit_index,
                progress_percent=int(100 * unit_index / max(1, len(units))),
                worker_pid=None,
            )
        _update_status(
            status_path,
            status="completed",
            phase="completed",
            message="The nnU-Net workflow completed.",
            progress_percent=100,
            worker_pid=None,
            completed_at_epoch=time.time(),
        )
        return 0
    except InterruptedError:
        _update_status(
            status_path,
            status="cancelled",
            phase="cancelled",
            message="The nnU-Net workflow was cancelled.",
            worker_pid=None,
            completed_at_epoch=time.time(),
        )
        return 130
    except Exception as exc:
        _update_status(
            status_path,
            status="failed",
            phase="controller_failed",
            message="The nnU-Net workflow controller failed.",
            error="{}: {}".format(type(exc).__name__, exc),
            error_category=_classify_error(exc, "controller"),
            traceback=traceback.format_exc(),
            worker_pid=None,
            completed_at_epoch=time.time(),
        )
        return 1


def workflow1_nnUnet_train_and_predict(config_file, **kwargs):
    """Run one complete, user-selected TOML workflow."""
    return run_workflow_controller(config_file, stages=DEFAULT_STAGES, **kwargs)


def workflow2_nnUnet_train_and_predict_batch(config_files, **kwargs):
    """Run several explicit TOML workflows without embedded path assumptions."""
    for config_file in config_files:
        returncode = run_workflow_controller(config_file, stages=DEFAULT_STAGES, **kwargs)
        if returncode:
            return returncode
    return 0


def workflow3_sample_and_predict(model_paths, input_path, output_paths):
    """Run several independent models; all paths are supplied by the caller."""
    if len(model_paths) != len(output_paths):
        raise ValueError("model_paths and output_paths must have equal lengths")
    action = _module("Action4_Predict")
    for model_path, output_path in zip(model_paths, output_paths):
        action.easy_predict_with_preresample(
            model_folder=model_path,
            input_path=input_path,
            output_path=output_path,
            enable_stats=True,
        )


def _model_maps(model_map_file, names):
    toml = _toml_module()
    source = _resolve_file(model_map_file, SCRIPT_DIR)
    with source.open("rb") as handle:
        values = toml.load(handle)
    missing = [name for name in names if name not in values]
    if missing:
        raise KeyError("Unknown ModelMap sections: {}".format(", ".join(missing)))
    return [values[name] for name in names]


def combine_multimask_to_one(
    inputmask_paths,
    outputmask_path,
    model_parts,
    model_combine,
    model_map_file="ModelMap.toml",
):
    action = _module("Action1_ConvertLabeledToTrainData")
    class_maps = _model_maps(model_map_file, model_parts)
    combine_maps = _model_maps(model_map_file, model_combine)
    combine_map = {}
    for row in combine_maps:
        combine_map.update(row)
    action.convert_multilabel_to_one(
        inputmask_paths,
        outputmask_path,
        class_map=class_maps,
        combine_map=combine_map,
    )


def workflow4_shared_spacing_predict_and_merge(
    model_paths,
    model_parts,
    model_combine,
    input_path,
    outputmask_path,
    model_map_file="ModelMap.toml",
):
    action = _module("Action4_Predict")
    class_maps = _model_maps(model_map_file, model_parts)
    combine_maps = _model_maps(model_map_file, model_combine)
    combine_map = {}
    for row in combine_maps:
        combine_map.update(row)
    action.multimodel_predict_and_merge(
        model_folders=model_paths,
        input_path=input_path,
        output_path=outputmask_path,
        class_maps=class_maps,
        combine_map=combine_map,
        model_names=model_parts,
    )


def _internal_stage(args) -> int:
    result_path = Path(args.result_file).resolve()
    stage = str(args.stage)
    try:
        config = _read_json(Path(args.resolved_config).resolve(), None)
        if not isinstance(config, dict):
            raise ValueError("Resolved workflow configuration is missing or invalid")
        _execute_unit(stage, config, args.dataset_index)
        _write_json_atomic(result_path, {"status": "ok", "stage": stage})
        return 0
    except Exception as exc:
        _write_json_atomic(
            result_path,
            {
                "status": "error",
                "stage": stage,
                "error": "{}: {}".format(type(exc).__name__, exc),
                "error_category": _classify_error(exc, stage),
                "traceback": traceback.format_exc(),
            },
        )
        traceback.print_exc()
        return 1


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "convert", "preprocess", "train", "predict", "evaluate"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--status-file")
        command.add_argument("--control-file")
        command.add_argument("--dry-run", action="store_true")
    run = subparsers.choices["run"]
    run.add_argument(
        "--stages",
        nargs="+",
        choices=(*DEFAULT_STAGES, "evaluate"),
        default=list(DEFAULT_STAGES),
    )

    cancel = subparsers.add_parser(
        "cancel", description="Request a bounded stop for one running workflow."
    )
    cancel.add_argument("--control-file", required=True)
    status = subparsers.add_parser(
        "status", description="Print one workflow's latest structured status."
    )
    status.add_argument("--status-file", required=True)

    return parser


def _build_internal_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("command")
    stage = parser
    stage.add_argument("--resolved-config", required=True)
    stage.add_argument("--stage", required=True)
    stage.add_argument("--dataset-index", type=int)
    stage.add_argument("--result-file", required=True)
    return parser


def main(argv=None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] == "_stage":
        args = _build_internal_parser().parse_args(values)
        return _internal_stage(args)
    args = _build_parser().parse_args(values)
    if args.command == "cancel":
        try:
            path = request_cancel(args.control_file)
            print("Cancellation requested: {}".format(path))
            return 0
        except Exception as exc:
            print("{}: {}".format(type(exc).__name__, exc), file=sys.stderr)
            return 2
    if args.command == "status":
        try:
            show_status(args.status_file)
            return 0
        except Exception as exc:
            print("{}: {}".format(type(exc).__name__, exc), file=sys.stderr)
            return 2
    stages = args.stages if args.command == "run" else [args.command]
    try:
        return run_workflow_controller(
            args.config,
            stages=stages,
            status_file=args.status_file,
            control_file=args.control_file,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        print("{}: {}".format(type(exc).__name__, exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
