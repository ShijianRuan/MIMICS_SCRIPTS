from pathlib import Path

import pytest

from nninteractive_finetune.config import (
    DEFAULT_CONFIG,
    load_config,
    validate_config,
)


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


def test_legacy_interaction_steps_preserves_exact_historical_budget(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  base_model_dir: model
data:
  manifest: manifest.json
training:
  output_dir: output
prompts:
  interaction_steps: 5
""".strip(),
        encoding="utf-8",
    )
    config = load_config(config_path)
    prompts = config["prompts"]
    assert prompts["min_interaction_steps"] == 5
    assert prompts["max_interaction_steps"] == 5
    assert prompts["validation_interaction_steps"] == [1, 3, 5]
    assert prompts["training_goal"] == "legacy"
    assert prompts["correction_policy"] == "clopa_paired"
    assert prompts["initial_mask_probability"] == 0.0


def test_default_prompt_policy_is_bounded_sequential_adaptation(tmp_path):
    config = _valid_config(tmp_path)
    validate_config(config)
    prompts = config["prompts"]
    assert prompts["training_goal"] == "general"
    assert prompts["correction_policy"] == "official_single"
    assert prompts["min_interaction_steps"] == 1
    assert prompts["max_interaction_steps"] == 5
    assert prompts["validation_interaction_steps"] == [1, 3, 5]
    assert sum(prompts["interaction_step_weights"]) == pytest.approx(1.0)
    assert prompts["provided_initial_mask_probability"] == pytest.approx(0.7)


def test_rejects_invalid_real_initial_mask_mix_probability(tmp_path):
    config = _valid_config(tmp_path)
    config["prompts"]["provided_initial_mask_probability"] = 1.1
    with pytest.raises(
        ValueError, match="provided_initial_mask_probability"
    ):
        validate_config(config)
