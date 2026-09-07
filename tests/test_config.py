"""Tests for experiment configuration loading and validation."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from fdl_project.config.loader import (
    apply_overrides,
    build_experiment_config,
    deep_merge,
    dump_experiment_config,
    load_experiment_config,
    parse_override,
)
from fdl_project.config.registry import (
    available_models,
    available_optimizers,
    available_schedulers,
    build_model,
)
from fdl_project.config.schema import ExperimentConfig, ParameterGroupConfig

CONFIG_DIRECTORY = Path(__file__).resolve().parents[1] / "configs" / "train"


def _write(directory: Path, name: str, payload: str) -> Path:
    path = directory / name
    path.write_text(textwrap.dedent(payload), encoding="utf-8")
    return path


def test_defaults_are_merged_underneath_the_experiment(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "defaults.yaml",
        """
        seed: 86
        trainer:
          max_epochs: 40
          batch_size: 512
        """,
    )
    path = _write(
        tmp_path,
        "experiment.yaml",
        """
        name: partial-override
        trainer:
          batch_size: 64
        """,
    )

    config = load_experiment_config(path)

    assert config.name == "partial-override"
    assert config.trainer.batch_size == 64
    # untouched keys keep the shared default rather than the dataclass default
    assert config.trainer.max_epochs == 40
    assert config.seed == 86


def test_a_misspelled_key_stops_the_run(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "typo.yaml",
        """
        name: typo
        trainer:
          max_epoch: 10
        """,
    )
    with pytest.raises(ValueError, match="Unknown key"):
        load_experiment_config(path)


def test_an_unknown_top_level_section_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown key"):
        build_experiment_config({"name": "x", "trainner": {}})


def test_a_config_must_be_named() -> None:
    with pytest.raises(ValueError, match="name"):
        build_experiment_config({"seed": 86})


@pytest.mark.parametrize("name", ["has space", "-leading", "slash/name", ""])
def test_names_that_cannot_be_directories_are_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="name"):
        build_experiment_config({"name": name})


def test_imbalance_accepts_a_preset_or_an_explicit_strategy() -> None:
    preset = build_experiment_config(
        {"name": "p", "imbalance": {"preset": "focal_loss"}}
    )
    assert preset.imbalance.name == "focal_loss"
    assert preset.imbalance.loss == "focal"

    explicit = build_experiment_config(
        {
            "name": "e",
            "imbalance": {"name": "custom_ce", "loss_weighting": "inverse_sqrt"},
        }
    )
    assert explicit.imbalance.loss_weighting == "inverse_sqrt"


def test_an_unknown_imbalance_preset_lists_the_real_ones() -> None:
    with pytest.raises(ValueError, match="inverse_sqrt_sampler"):
        build_experiment_config({"name": "p", "imbalance": {"preset": "nope"}})


def test_a_preset_cannot_be_half_overridden() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        build_experiment_config(
            {
                "name": "p",
                "imbalance": {"preset": "focal_loss", "focal_gamma": 3.0},
            }
        )


def test_target_size_accepts_a_scalar_or_a_pair() -> None:
    scalar = build_experiment_config(
        {"name": "s", "data": {"preprocessing": {"target_size": 224}}}
    )
    assert scalar.data.preprocessing.target_size == (224, 224)

    pair = build_experiment_config(
        {"name": "p", "data": {"preprocessing": {"target_size": [96, 128]}}}
    )
    assert pair.data.preprocessing.target_size == (96, 128)


def test_parameter_groups_are_built_and_validated() -> None:
    config = build_experiment_config(
        {
            "name": "groups",
            "optimizer": {
                "name": "adamw",
                "kwargs": {"lr": 1e-3},
                "param_groups": [
                    {"name": "encoder", "pattern": "^encoder\\.", "kwargs": {"lr": 1e-5}}
                ],
            },
        }
    )
    group = config.optimizer.param_groups[0]
    assert group.group_name == "encoder"
    assert group.kwargs["lr"] == 1e-5


def test_a_parameter_group_must_actually_override_something() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ParameterGroupConfig(pattern="^encoder\\.")


def test_an_invalid_regular_expression_is_rejected() -> None:
    with pytest.raises(ValueError, match="regular expression"):
        ParameterGroupConfig(pattern="^encoder[", kwargs={"lr": 1e-5})


def test_early_stopping_floor_is_clamped_to_the_budget() -> None:
    """`--override trainer.max_epochs=2` must not also require restating min_epochs."""

    config = build_experiment_config(
        {
            "name": "short",
            "trainer": {"max_epochs": 2, "early_stopping": {"min_epochs": 5}},
        }
    )
    assert config.trainer.early_stopping.min_epochs == 2


@pytest.mark.parametrize(
    ("text", "expected_path", "expected_value"),
    [
        ("trainer.max_epochs=2", ["trainer", "max_epochs"], 2),
        ("trainer.amp=false", ["trainer", "amp"], False),
        ("logging.wandb.tags=[a, b]", ["logging", "wandb", "tags"], ["a", "b"]),
        ("seed=7", ["seed"], 7),
    ],
)
def test_overrides_are_parsed_as_yaml(
    text: str, expected_path: list[str], expected_value: object
) -> None:
    assert parse_override(text) == (expected_path, expected_value)


def test_a_malformed_override_is_rejected() -> None:
    with pytest.raises(ValueError, match="section.key=value"):
        parse_override("trainer.max_epochs")


def test_overrides_reach_the_built_config() -> None:
    payload = apply_overrides(
        {"name": "o", "trainer": {"max_epochs": 40}},
        ["trainer.max_epochs=3", "trainer.device=cpu"],
    )
    config = build_experiment_config(payload)
    assert config.trainer.max_epochs == 3
    assert config.trainer.device == "cpu"


def test_deep_merge_does_not_mutate_its_inputs() -> None:
    base = {"trainer": {"max_epochs": 40, "batch_size": 512}}
    override = {"trainer": {"batch_size": 64}}
    merged = deep_merge(base, override)

    assert merged["trainer"] == {"max_epochs": 40, "batch_size": 64}
    assert base["trainer"]["batch_size"] == 512


def test_run_and_checkpoint_directories_are_named_from_the_config() -> None:
    config = build_experiment_config({"name": "resnet18-224-finetune"})
    assert config.run_directory.name == "resnet18-224-finetune"
    assert config.checkpoint_directory.name == "resnet18-224-finetune"


def test_a_config_round_trips_through_yaml(tmp_path: Path) -> None:
    original = build_experiment_config(
        {
            "name": "round-trip",
            "data": {"preprocessing": {"target_size": [224, 224]}},
            "optimizer": {
                "param_groups": [{"pattern": "^encoder\\.", "kwargs": {"lr": 1e-5}}]
            },
            "scheduler": {"name": "cosine", "interval": "step"},
        }
    )
    path = dump_experiment_config(original, tmp_path / "config.yaml")
    reloaded = build_experiment_config(yaml.safe_load(path.read_text()))

    assert reloaded.to_dict() == original.to_dict()


def test_the_repository_configs_all_load() -> None:
    """Every checked-in experiment config must be runnable as written."""

    configs = sorted(
        path for path in CONFIG_DIRECTORY.glob("*.yaml") if path.name != "defaults.yaml"
    )
    assert configs, "no experiment configs found"
    for path in configs:
        config = load_experiment_config(path)
        assert isinstance(config, ExperimentConfig)
        assert config.model.name in available_models()


def test_registries_report_what_they_contain() -> None:
    assert "baseline_cnn" in available_models()
    assert "resnet18" in available_models()
    assert {"sgd", "adam", "adamw"} <= set(available_optimizers())
    assert {"cosine", "cosine_with_warmup", "plateau"} <= set(available_schedulers())


def test_an_unknown_model_name_lists_the_available_ones() -> None:
    with pytest.raises(KeyError, match="baseline_cnn"):
        build_model("resnet99")
