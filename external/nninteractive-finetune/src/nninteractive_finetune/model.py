"""Official nnInteractive network reconstruction and adaptation policies."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from batchgenerators.utilities.file_and_folder_operations import load_json
from nnunetv2.utilities.label_handling.label_handling import (
    determine_num_input_channels,
)
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from nnInteractive.trainer.nnInteractiveTrainer import nnInteractiveTrainer_stub


@dataclass
class ModelBundle:
    network: torch.nn.Module
    model_dir: Path
    checkpoint_path: Path
    checkpoint: dict[str, Any]
    plans: dict[str, Any]
    dataset_json: dict[str, Any]
    plans_manager: PlansManager
    configuration_manager: Any
    interaction_channels: int


def _fold_folder(model_dir: Path, fold: int | str) -> Path:
    requested = "fold_{}".format(fold)
    path = model_dir / requested
    if path.is_dir():
        return path
    candidates = sorted(value for value in model_dir.glob("fold_*") if value.is_dir())
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(
        "Could not resolve {} under {}. Available folds: {}".format(
            requested,
            model_dir,
            ", ".join(value.name for value in candidates) or "none",
        )
    )


def _interaction_channel_count(model_dir: Path) -> int:
    capability_path = model_dir / "inference_info.json"
    if capability_path.is_file():
        with capability_path.open("r", encoding="utf-8") as handle:
            capability = json.load(handle)
        if "interaction_channels" in capability:
            # Metadata excludes previous segmentation from this count.
            return int(capability["interaction_channels"]) + 1
    return 7


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def network_parameter_fingerprint(network: torch.nn.Module) -> str:
    """Hash the effective, de-duplicated parameters owned by a loaded network."""
    return _named_parameter_fingerprint(network.named_parameters())


def _named_parameter_fingerprint(parameters: Any) -> str:
    digest = hashlib.sha256()
    digest.update(b"nninteractive_effective_parameters.v1\0")
    for name, parameter in parameters:
        tensor = parameter.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tuple(int(value) for value in tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def state_dict_loaded_parameter_fingerprint(
    network: torch.nn.Module, state_dict: dict[str, torch.Tensor]
) -> str:
    """Predict the effective parameters after loading a possibly aliased state.

    PyTorch state dictionaries can expose the same Parameter under several
    module paths. Loading assigns each path in traversal order, so the last
    alias wins. This reproduces that ownership rule without allocating a
    second full nnInteractive network.
    """
    references = network.state_dict(keep_vars=True)
    final_values: dict[int, torch.Tensor] = {}
    for name, reference in references.items():
        if isinstance(reference, torch.nn.Parameter):
            if name not in state_dict:
                raise RuntimeError(
                    "Exported state is missing parameter alias: {}".format(name)
                )
            final_values[id(reference)] = state_dict[name]
    effective = []
    for name, parameter in network.named_parameters():
        value = final_values.get(id(parameter))
        if value is None:
            raise RuntimeError(
                "Could not resolve the exported value for parameter: {}".format(
                    name
                )
            )
        effective.append((name, value))
    return _named_parameter_fingerprint(effective)


def audit_model_dir(
    model_dir: str | Path,
    fold: int | str = 0,
    checkpoint_name: str = "checkpoint_final.pth",
) -> dict[str, Any]:
    root = Path(model_dir).expanduser().resolve()
    required = ["plans.json", "dataset.json"]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Model directory is missing: {}".format(", ".join(missing))
        )
    fold_dir = _fold_folder(root, fold)
    checkpoint_path = fold_dir / checkpoint_name
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Checkpoint is missing: {}".format(checkpoint_path))

    checkpoint = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=False
    )
    return _audit_loaded_checkpoint(root, checkpoint_path, checkpoint)


def _audit_loaded_checkpoint(
    root: Path,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    state = checkpoint.get("network_weights")
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint has no network_weights state dictionary.")
    first_weight = state.get("encoder.stem.convs.0.conv.weight")
    if first_weight is None:
        raise ValueError("Checkpoint does not contain the expected ResEnc stem.")
    output_weights = [
        tensor
        for name, tensor in state.items()
        if "decoder.seg_layers" in name and name.endswith(".weight")
    ]
    input_channels = int(first_weight.shape[1])
    output_channels = sorted({int(tensor.shape[0]) for tensor in output_weights})
    configuration = checkpoint.get("init_args", {}).get("configuration")
    plans = load_json(str(root / "plans.json"))
    if not configuration:
        raise ValueError("Checkpoint does not identify its nnUNet configuration.")
    configuration_manager = PlansManager(plans).get_configuration(configuration)
    expected_interactions = _interaction_channel_count(root)
    report = {
        "schema_version": "nninteractive_checkpoint_audit.v1",
        "model_dir": str(root),
        "checkpoint_path": str(checkpoint_path),
        "trainer_name": checkpoint.get("trainer_name"),
        "configuration": configuration,
        "patch_size": list(configuration_manager.patch_size),
        "input_channels": input_channels,
        "interaction_channels": expected_interactions,
        "output_channels": output_channels,
        "state_keys": len(state),
        "current_epoch": checkpoint.get("current_epoch"),
        "compatible": (
            input_channels == 1 + expected_interactions and output_channels == [2]
        ),
    }
    if not report["compatible"]:
        raise ValueError(
            "Checkpoint failed nnInteractive compatibility audit: {}".format(report)
        )
    return report


def load_model(
    model_dir: str | Path,
    device: torch.device,
    fold: int | str = 0,
    checkpoint_name: str = "checkpoint_final.pth",
) -> ModelBundle:
    root = Path(model_dir).expanduser().resolve()
    plans = load_json(str(root / "plans.json"))
    dataset_json = load_json(str(root / "dataset.json"))
    plans_manager = PlansManager(plans)
    fold_dir = _fold_folder(root, fold)
    checkpoint_path = fold_dir / checkpoint_name
    checkpoint = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=False
    )
    _audit_loaded_checkpoint(root, checkpoint_path, checkpoint)
    configuration_name = checkpoint["init_args"]["configuration"]
    configuration_manager = plans_manager.get_configuration(configuration_name)
    interaction_channels = _interaction_channel_count(root)
    image_channels = determine_num_input_channels(
        plans_manager, configuration_manager, dataset_json
    )
    network = nnInteractiveTrainer_stub.build_network_architecture(
        plans_manager,
        configuration_manager,
        image_channels + interaction_channels,
        2,
        enable_deep_supervision=False,
    )
    incompatible = network.load_state_dict(checkpoint["network_weights"], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Official checkpoint does not match the reconstructed network. "
            "Missing: {}; unexpected: {}.".format(
                incompatible.missing_keys, incompatible.unexpected_keys
            )
        )
    # The reconstructed module now owns the weights. Keeping the original
    # 392 MB state dictionary in ModelBundle would double host RAM throughout
    # training without adding information needed for export.
    checkpoint_metadata = dict(checkpoint)
    checkpoint_metadata.pop("network_weights", None)
    del checkpoint
    network.to(device)
    return ModelBundle(
        network=network,
        model_dir=root,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint_metadata,
        plans=plans,
        dataset_json=dataset_json,
        plans_manager=plans_manager,
        configuration_manager=configuration_manager,
        interaction_channels=interaction_channels,
    )


def _highest_index(module_names: list[str], prefix: str) -> int:
    indices = []
    for name in module_names:
        if name.startswith(prefix):
            remainder = name[len(prefix) :].split(".", 1)[0]
            if remainder.isdigit():
                indices.append(int(remainder))
    if not indices:
        raise RuntimeError("Could not find network modules under {}.".format(prefix))
    return max(indices)


def configure_trainable_parameters(
    network: torch.nn.Module, strategy: str
) -> dict[str, Any]:
    """Apply CLoPA-I.N, CLoPA-C.N, or full-network trainability."""
    for parameter in network.parameters():
        parameter.requires_grad = False

    module_names = [name for name, _ in network.named_modules()]
    last_decoder_stage = _highest_index(module_names, "decoder.stages.")
    last_seg_layer = _highest_index(module_names, "decoder.seg_layers.")
    selected: set[int] = set()
    selected_names: list[str] = []

    def select(name: str, parameter: torch.nn.Parameter) -> None:
        if id(parameter) in selected:
            return
        parameter.requires_grad = True
        selected.add(id(parameter))
        selected_names.append(name)

    if strategy == "full":
        for name, parameter in network.named_parameters():
            select(name, parameter)
    else:
        for module_name, module in network.named_modules():
            if isinstance(module, torch.nn.InstanceNorm3d) and module.affine:
                for parameter_name, parameter in module.named_parameters(recurse=False):
                    select("{}.{}".format(module_name, parameter_name), parameter)

        if strategy == "clopa_conv":
            convolution_roots = (
                "encoder.stem",
                "encoder.stages.0",
                "decoder.stages.{}".format(last_decoder_stage),
                "decoder.seg_layers.{}".format(last_seg_layer),
            )
            for module_name, module in network.named_modules():
                selected_module = any(
                    module_name == root or module_name.startswith(root + ".")
                    for root in convolution_roots
                )
                if isinstance(module, torch.nn.Conv3d) and selected_module:
                    # CLoPA-C.N specifies convolution kernels. Bias remains frozen.
                    select("{}.weight".format(module_name), module.weight)
        elif strategy != "clopa_in":
            raise ValueError("Unsupported fine-tuning strategy: {}".format(strategy))

    trainable = [
        parameter for parameter in network.parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("The selected strategy produced no trainable parameters.")
    total_count = sum(parameter.numel() for parameter in network.parameters())
    trainable_count = sum(parameter.numel() for parameter in trainable)
    return {
        "strategy": strategy,
        "parameters": trainable,
        "names": sorted(selected_names),
        "trainable_count": int(trainable_count),
        "total_count": int(total_count),
        "trainable_fraction": float(trainable_count / total_count),
    }
