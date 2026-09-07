"""Load an experiment YAML into the validated config objects.

Deliberately not Hydra: the composition and override machinery costs more to
learn than it saves for a handful of configs, and the failure modes are opaque.
This is a deep merge over one defaults file plus strict key checking, so a
misspelled key stops the run instead of silently training the default.
"""

from __future__ import annotations

import copy
from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml

from fdl_project.config.schema import (
    AugmentationConfig,
    CheckpointConfig,
    DataConfig,
    EarlyStoppingConfig,
    ExperimentConfig,
    LoggingConfig,
    ModelConfig,
    OptimizerConfig,
    ParameterGroupConfig,
    SchedulerConfig,
    TrainerConfig,
    WandbConfig,
)
from fdl_project.data.imbalance import PRESET_IMBALANCE_CONFIGS, ImbalanceConfig
from fdl_project.data.preprocessing import PreprocessingConfig

DEFAULTS_FILENAME = "defaults.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}.")
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a mapping at the top level: {path}.")
    return payload


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge ``override`` into ``base`` recursively, without mutating either."""

    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def parse_override(text: str) -> tuple[list[str], Any]:
    """Parse one ``section.key=value`` CLI override into a path and a value.

    The value is read as YAML, so ``trainer.max_epochs=2`` gives an int,
    ``trainer.amp=false`` a bool, and ``logging.wandb.tags=[a, b]`` a list.
    """

    if "=" not in text:
        raise ValueError(
            f"Override {text!r} must have the form section.key=value."
        )
    dotted, _, raw_value = text.partition("=")
    path = [part for part in dotted.strip().split(".") if part]
    if not path:
        raise ValueError(f"Override {text!r} is missing a key path.")
    return path, yaml.safe_load(raw_value)


def apply_overrides(payload: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``section.key=value`` overrides to a raw config mapping."""

    result = copy.deepcopy(payload)
    for text in overrides:
        path, value = parse_override(text)
        cursor = result
        for key in path[:-1]:
            existing = cursor.get(key)
            if not isinstance(existing, dict):
                existing = {}
                cursor[key] = existing
            cursor = existing
        cursor[path[-1]] = value
    return result


def _check_keys(payload: dict[str, Any], allowed: set[str], *, section: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        options = ", ".join(sorted(allowed))
        raise ValueError(
            f"Unknown key(s) in section {section!r}: {', '.join(unknown)}. "
            f"Allowed: {options}."
        )


def _field_names(dataclass_type: type) -> set[str]:
    return {item.name for item in fields(dataclass_type)}


def _build_section(
    payload: Any, dataclass_type: type, *, section: str, **replacements: Any
) -> Any:
    """Validate the keys of one mapping and construct its dataclass."""

    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError(f"Section {section!r} must be a mapping.")
    _check_keys(payload, _field_names(dataclass_type), section=section)
    arguments = {key: value for key, value in payload.items() if key not in replacements}
    arguments.update(replacements)
    return dataclass_type(**arguments)


def _build_preprocessing(payload: Any) -> PreprocessingConfig:
    if payload is None:
        return PreprocessingConfig()
    if not isinstance(payload, dict):
        raise ValueError("Section 'data.preprocessing' must be a mapping.")
    _check_keys(
        payload, _field_names(PreprocessingConfig), section="data.preprocessing"
    )
    arguments = dict(payload)
    if "target_size" in arguments:
        size = arguments["target_size"]
        if isinstance(size, int):
            size = [size, size]
        if not isinstance(size, (list, tuple)) or len(size) != 2:
            raise ValueError(
                "data.preprocessing.target_size must be an integer or [height, width]."
            )
        arguments["target_size"] = (int(size[0]), int(size[1]))
    return PreprocessingConfig(**arguments)


def _build_imbalance(payload: Any) -> ImbalanceConfig:
    """Accept either ``preset: <name>`` or a fully spelled-out strategy."""

    if payload is None:
        return PRESET_IMBALANCE_CONFIGS["inverse_sqrt_sampler"]
    if isinstance(payload, str):
        payload = {"preset": payload}
    if not isinstance(payload, dict):
        raise ValueError("Section 'imbalance' must be a mapping or a preset name.")
    if "preset" in payload:
        if len(payload) > 1:
            extra = ", ".join(sorted(set(payload) - {"preset"}))
            raise ValueError(
                f"imbalance.preset cannot be combined with other keys: {extra}. "
                "Spell the strategy out instead."
            )
        preset = payload["preset"]
        if preset not in PRESET_IMBALANCE_CONFIGS:
            available = ", ".join(sorted(PRESET_IMBALANCE_CONFIGS))
            raise ValueError(
                f"Unknown imbalance preset {preset!r}. Available: {available}."
            )
        return PRESET_IMBALANCE_CONFIGS[preset]
    _check_keys(payload, _field_names(ImbalanceConfig), section="imbalance")
    return ImbalanceConfig(**payload)


def _build_optimizer(payload: Any) -> OptimizerConfig:
    if payload is None:
        return OptimizerConfig()
    if not isinstance(payload, dict):
        raise ValueError("Section 'optimizer' must be a mapping.")
    _check_keys(payload, _field_names(OptimizerConfig), section="optimizer")
    arguments = dict(payload)
    raw_groups = arguments.pop("param_groups", ()) or ()
    if not isinstance(raw_groups, (list, tuple)):
        raise ValueError("optimizer.param_groups must be a list.")
    groups = []
    for position, entry in enumerate(raw_groups):
        if not isinstance(entry, dict):
            raise ValueError(
                f"optimizer.param_groups[{position}] must be a mapping."
            )
        _check_keys(
            entry,
            _field_names(ParameterGroupConfig),
            section=f"optimizer.param_groups[{position}]",
        )
        groups.append(ParameterGroupConfig(**entry))
    return OptimizerConfig(param_groups=tuple(groups), **arguments)


def _build_trainer(payload: Any) -> TrainerConfig:
    if payload is None:
        return TrainerConfig()
    if not isinstance(payload, dict):
        raise ValueError("Section 'trainer' must be a mapping.")
    _check_keys(payload, _field_names(TrainerConfig), section="trainer")
    early_stopping = _build_section(
        payload.get("early_stopping"),
        EarlyStoppingConfig,
        section="trainer.early_stopping",
    )
    arguments = {key: value for key, value in payload.items() if key != "early_stopping"}
    return TrainerConfig(early_stopping=early_stopping, **arguments)


def _build_logging(payload: Any) -> LoggingConfig:
    if payload is None:
        return LoggingConfig()
    if not isinstance(payload, dict):
        raise ValueError("Section 'logging' must be a mapping.")
    _check_keys(payload, _field_names(LoggingConfig), section="logging")
    wandb = _build_section(
        payload.get("wandb"), WandbConfig, section="logging.wandb"
    )
    arguments = {key: value for key, value in payload.items() if key != "wandb"}
    return LoggingConfig(wandb=wandb, **arguments)


def build_experiment_config(payload: dict[str, Any]) -> ExperimentConfig:
    """Turn a merged raw mapping into a fully validated ``ExperimentConfig``."""

    if not isinstance(payload, dict):
        raise ValueError("An experiment config must be a mapping.")
    _check_keys(payload, _field_names(ExperimentConfig), section="<root>")
    if "name" not in payload:
        raise ValueError("An experiment config must set a 'name'.")

    data_payload = payload.get("data") or {}
    if not isinstance(data_payload, dict):
        raise ValueError("Section 'data' must be a mapping.")
    data = _build_section(
        data_payload,
        DataConfig,
        section="data",
        preprocessing=_build_preprocessing(data_payload.get("preprocessing")),
        augmentation=_build_section(
            data_payload.get("augmentation"),
            AugmentationConfig,
            section="data.augmentation",
        ),
    )
    return ExperimentConfig(
        name=payload["name"],
        seed=payload.get("seed", 86),
        data=data,
        imbalance=_build_imbalance(payload.get("imbalance")),
        model=_build_section(payload.get("model"), ModelConfig, section="model"),
        optimizer=_build_optimizer(payload.get("optimizer")),
        scheduler=_build_section(
            payload.get("scheduler"), SchedulerConfig, section="scheduler"
        ),
        trainer=_build_trainer(payload.get("trainer")),
        checkpoint=_build_section(
            payload.get("checkpoint"), CheckpointConfig, section="checkpoint"
        ),
        logging=_build_logging(payload.get("logging")),
    )


def load_experiment_config(
    config_path: str | Path,
    *,
    overrides: list[str] | None = None,
    defaults_path: str | Path | None = None,
) -> ExperimentConfig:
    """Load ``config_path`` on top of the shared defaults and validate it.

    The defaults file is ``defaults.yaml`` next to the experiment config unless
    ``defaults_path`` says otherwise; it is optional.
    """

    path = Path(config_path)
    experiment_payload = _load_yaml(path)
    if defaults_path is None:
        candidate = path.parent / DEFAULTS_FILENAME
        defaults_payload = (
            {} if candidate == path or not candidate.is_file() else _load_yaml(candidate)
        )
    else:
        defaults_payload = _load_yaml(Path(defaults_path))

    merged = deep_merge(defaults_payload, experiment_payload)
    merged = apply_overrides(merged, list(overrides or []))
    merged.setdefault("name", path.stem)
    return build_experiment_config(merged)


def dump_experiment_config(config: ExperimentConfig, destination: str | Path) -> Path:
    """Write the fully resolved config next to a run's artifacts."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
    return path
