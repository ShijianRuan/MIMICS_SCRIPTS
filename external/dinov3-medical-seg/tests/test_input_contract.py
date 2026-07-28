import copy

import pytest

from src.data.input_contract import (
    input_contract_for_config,
    validate_input_contract,
)


def _config(pipeline="volume"):
    config = {
        "data": {
            "modality": "ct",
            "img_size": [256, 256],
            "slice_normalization": "timeslice_casewise",
        },
        "model": {
            "input_normalization": "imagenet" if pipeline == "volume" else "none",
            "channel_policy": "repeat",
            "slice_axis": "axial",
        },
        "training": {"pipeline": pipeline},
        "runtime": {},
    }
    config["runtime"]["input_contract"] = input_contract_for_config(config)
    return config


@pytest.mark.parametrize("pipeline", ["volume", "cached_slices"])
def test_saved_input_contract_accepts_unchanged_config(pipeline):
    config = _config(pipeline)
    assert validate_input_contract(config, required=True) == config["runtime"][
        "input_contract"
    ]


def test_volume_contract_rejects_modality_drift():
    config = _config("volume")
    changed = copy.deepcopy(config)
    changed["data"]["modality"] = "mr"
    with pytest.raises(RuntimeError, match="differs"):
        validate_input_contract(changed, required=True)


def test_cached_slice_contract_rejects_normalization_drift():
    config = _config("cached_slices")
    changed = copy.deepcopy(config)
    changed["data"]["slice_normalization"] = "percentile_minmax"
    with pytest.raises(RuntimeError, match="differs"):
        validate_input_contract(changed, required=True)


def test_cached_slice_contract_rejects_spacing_resampling():
    config = _config("cached_slices")
    config["data"]["target_spacing"] = [1.0, 1.0, 1.0]
    config["runtime"]["input_contract"] = input_contract_for_config(config)
    with pytest.raises(RuntimeError, match="preserve the source NIfTI grid"):
        validate_input_contract(config, required=True)
