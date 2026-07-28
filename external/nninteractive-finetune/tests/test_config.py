from pathlib import Path

import pytest

from nninteractive_finetune.config import DEFAULT_CONFIG, validate_config


def _valid_config(tmp_path: Path):
    config = {section: dict(values) for section, values in DEFAULT_CONFIG.items()}
    config["data"]["augmentation"] = dict(DEFAULT_CONFIG["data"]["augmentation"])
    config["model"]["base_model_dir"] = str(tmp_path / "model")
    config["data"]["manifest"] = str(tmp_path / "manifest.json")
    config["training"]["output_dir"] = str(tmp_path / "output")
    return config


def test_rejects_unsafe_geometry_mode(tmp_path):
    config = _valid_config(tmp_path)
    config["data"]["strict_geometry"] = False
    with pytest.raises(ValueError, match="unsafe"):
        validate_config(config)


def test_rejects_unvalidated_prompt_mode(tmp_path):
    config = _valid_config(tmp_path)
    config["prompts"]["mode"] = "mixed"
    with pytest.raises(ValueError, match="clicks"):
        validate_config(config)
