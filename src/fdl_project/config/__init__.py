"""Declarative experiment configuration: schema, YAML loading, and registries."""

from fdl_project.config.loader import (
    apply_overrides,
    build_experiment_config,
    deep_merge,
    dump_experiment_config,
    load_experiment_config,
    parse_override,
)
from fdl_project.config.registry import (
    ScheduleContext,
    available_models,
    available_optimizers,
    available_schedulers,
    build_model,
    build_scheduler,
)
from fdl_project.config.schema import (
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

__all__ = [
    "CheckpointConfig",
    "DataConfig",
    "EarlyStoppingConfig",
    "ExperimentConfig",
    "LoggingConfig",
    "ModelConfig",
    "OptimizerConfig",
    "ParameterGroupConfig",
    "ScheduleContext",
    "SchedulerConfig",
    "TrainerConfig",
    "WandbConfig",
    "apply_overrides",
    "available_models",
    "available_optimizers",
    "available_schedulers",
    "build_experiment_config",
    "build_model",
    "build_scheduler",
    "deep_merge",
    "dump_experiment_config",
    "load_experiment_config",
    "parse_override",
]
