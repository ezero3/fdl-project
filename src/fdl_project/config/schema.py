"""Typed, validated description of one experiment.

Every field an experiment YAML can set has a dataclass here, validated on
construction in the same style as ``PreprocessingConfig`` and
``ImbalanceConfig``. A malformed config must fail before a GPU is touched, not
forty epochs into a Colab session.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from fdl_project.data.imbalance import ImbalanceConfig
from fdl_project.data.preprocessing import PreprocessingConfig

# Run names become directory names under output/ and on Drive, and are passed
# to save_evaluation_results, which enforces the same pattern.
RUN_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

DeviceName = Literal["auto", "cpu", "cuda", "mps"]
SchedulerInterval = Literal["epoch", "step"]
MonitorMode = Literal["max", "min"]
ResumeMode = Literal["auto", "never"]
WandbMode = Literal["online", "offline", "disabled"]


def _freeze_kwargs(kwargs: Any, *, field_name: str) -> MappingProxyType:
    if kwargs is None:
        return MappingProxyType({})
    if not isinstance(kwargs, Mapping):
        raise ValueError(f"{field_name} must be a mapping.")
    for key in kwargs:
        if not isinstance(key, str):
            raise ValueError(f"{field_name} keys must be strings; got {key!r}.")
    return MappingProxyType(dict(kwargs))


def _require_positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _require_non_negative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return value


@dataclass(frozen=True)
class AugmentationConfig:
    """Train-only augmentation. ``name: null`` disables it."""

    name: str | None = None
    probability: float = 1.0

    def __post_init__(self) -> None:
        name = self.name
        if isinstance(name, str) and name.lower() in {"none", "null", ""}:
            name = None
        if name is not None and not isinstance(name, str):
            raise ValueError("data.augmentation.name must be a string or null.")
        object.__setattr__(self, "name", name)
        if (
            isinstance(self.probability, bool)
            or not isinstance(self.probability, (int, float))
            or not 0.0 <= self.probability <= 1.0
        ):
            raise ValueError("data.augmentation.probability must lie in [0, 1].")

    @property
    def enabled(self) -> bool:
        return self.name is not None

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "probability": float(self.probability)}


@dataclass(frozen=True)
class DataConfig:
    """Where the data lives and how it is turned into batches."""

    dataset_path: Path = Path("data/MIR-WM811K/WM811K.pkl")
    split_directory: Path = Path("data/splits")
    num_workers: int = 0
    pin_memory: bool = False
    cache: bool = True
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    augmentation: AugmentationConfig = AugmentationConfig()
    subset: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_path", Path(self.dataset_path))
        object.__setattr__(self, "split_directory", Path(self.split_directory))
        _require_non_negative_integer(self.num_workers, "data.num_workers")
        if not isinstance(self.pin_memory, bool):
            raise ValueError("data.pin_memory must be a boolean.")
        if not isinstance(self.cache, bool):
            raise ValueError("data.cache must be a boolean.")
        if not isinstance(self.preprocessing, PreprocessingConfig):
            raise ValueError("data.preprocessing must be a PreprocessingConfig.")
        if not isinstance(self.augmentation, AugmentationConfig):
            raise ValueError("data.augmentation must be an AugmentationConfig.")
        if self.subset is not None:
            _require_positive_integer(self.subset, "data.subset")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_path": str(self.dataset_path),
            "split_directory": str(self.split_directory),
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "cache": self.cache,
            "preprocessing": self.preprocessing.to_dict(),
            "augmentation": self.augmentation.to_dict(),
            "subset": self.subset,
        }


@dataclass(frozen=True)
class ModelConfig:
    """A registry name plus whatever keyword arguments that builder accepts."""

    name: str = "baseline_cnn"
    kwargs: MappingProxyType = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("model.name must be a non-empty string.")
        object.__setattr__(
            self, "kwargs", _freeze_kwargs(self.kwargs, field_name="model.kwargs")
        )

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kwargs": dict(self.kwargs)}


@dataclass(frozen=True)
class ParameterGroupConfig:
    """One regular expression over parameter names and its optimizer overrides.

    This is what makes discriminative fine-tuning expressible from YAML: the
    pretrained encoder gets a much smaller learning rate than the new head.
    """

    pattern: str
    kwargs: MappingProxyType = field(default_factory=lambda: MappingProxyType({}))
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.pattern, str) or not self.pattern:
            raise ValueError("param_groups[].pattern must be a non-empty string.")
        try:
            re.compile(self.pattern)
        except re.error as error:
            raise ValueError(
                f"param_groups[].pattern is not a valid regular expression: {error}."
            ) from error
        object.__setattr__(
            self,
            "kwargs",
            _freeze_kwargs(self.kwargs, field_name="param_groups[].kwargs"),
        )
        if not self.kwargs:
            raise ValueError(
                "param_groups[].kwargs must override at least one optimizer setting; "
                "a group identical to the default is not a group."
            )
        if self.name is not None and not isinstance(self.name, str):
            raise ValueError("param_groups[].name must be a string.")

    @property
    def group_name(self) -> str:
        return self.name or self.pattern

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern,
            "kwargs": dict(self.kwargs),
            "name": self.name,
        }


@dataclass(frozen=True)
class OptimizerConfig:
    """Optimizer choice, its defaults, and optional per-parameter-group overrides."""

    name: str = "adamw"
    kwargs: MappingProxyType = field(
        default_factory=lambda: MappingProxyType({"lr": 1e-3, "weight_decay": 1e-4})
    )
    param_groups: tuple[ParameterGroupConfig, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("optimizer.name must be a non-empty string.")
        object.__setattr__(
            self, "kwargs", _freeze_kwargs(self.kwargs, field_name="optimizer.kwargs")
        )
        groups = tuple(self.param_groups)
        if not all(isinstance(group, ParameterGroupConfig) for group in groups):
            raise ValueError(
                "optimizer.param_groups must contain parameter-group definitions."
            )
        patterns = [group.pattern for group in groups]
        if len(set(patterns)) != len(patterns):
            raise ValueError("optimizer.param_groups patterns must be unique.")
        object.__setattr__(self, "param_groups", groups)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kwargs": dict(self.kwargs),
            "param_groups": [group.to_dict() for group in self.param_groups],
        }


@dataclass(frozen=True)
class SchedulerConfig:
    """Learning-rate schedule; ``name: none`` keeps the rate constant."""

    name: str | None = None
    interval: SchedulerInterval = "epoch"
    kwargs: MappingProxyType = field(default_factory=lambda: MappingProxyType({}))
    monitor: str = "validation_macro_f1"

    def __post_init__(self) -> None:
        name = self.name
        if isinstance(name, str) and name.lower() in {"none", "null", ""}:
            name = None
        if name is not None and not isinstance(name, str):
            raise ValueError("scheduler.name must be a string or null.")
        object.__setattr__(self, "name", name)
        if self.interval not in {"epoch", "step"}:
            raise ValueError("scheduler.interval must be 'epoch' or 'step'.")
        object.__setattr__(
            self, "kwargs", _freeze_kwargs(self.kwargs, field_name="scheduler.kwargs")
        )
        if not isinstance(self.monitor, str) or not self.monitor:
            raise ValueError("scheduler.monitor must be a non-empty metric name.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "interval": self.interval,
            "kwargs": dict(self.kwargs),
            "monitor": self.monitor,
        }


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Which validation metric decides the best epoch and when to stop."""

    monitor: str = "validation_macro_f1"
    mode: MonitorMode = "max"
    patience: int = 5
    min_epochs: int = 5

    def __post_init__(self) -> None:
        if not isinstance(self.monitor, str) or not self.monitor:
            raise ValueError("early_stopping.monitor must be a non-empty metric name.")
        if self.mode not in {"max", "min"}:
            raise ValueError("early_stopping.mode must be 'max' or 'min'.")
        _require_positive_integer(self.patience, "early_stopping.patience")
        _require_positive_integer(self.min_epochs, "early_stopping.min_epochs")

    def to_dict(self) -> dict[str, Any]:
        return {
            "monitor": self.monitor,
            "mode": self.mode,
            "patience": self.patience,
            "min_epochs": self.min_epochs,
        }


@dataclass(frozen=True)
class TrainerConfig:
    """The optimization budget.

    The default is a real training budget. The four-epoch budget used by the
    class-imbalance screening in ``fdl_project.analysis`` is a deliberate
    exception passed explicitly there, not a project-wide default.
    """

    max_epochs: int = 40
    batch_size: int = 512
    max_gradient_norm: float = 5.0
    amp: bool = True
    device: DeviceName = "auto"
    early_stopping: EarlyStoppingConfig = EarlyStoppingConfig()

    def __post_init__(self) -> None:
        _require_positive_integer(self.max_epochs, "trainer.max_epochs")
        _require_positive_integer(self.batch_size, "trainer.batch_size")
        if (
            isinstance(self.max_gradient_norm, bool)
            or not isinstance(self.max_gradient_norm, (int, float))
            or self.max_gradient_norm <= 0
        ):
            raise ValueError("trainer.max_gradient_norm must be a positive number.")
        if not isinstance(self.amp, bool):
            raise ValueError("trainer.amp must be a boolean.")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("trainer.device must be auto, cpu, cuda, or mps.")
        if not isinstance(self.early_stopping, EarlyStoppingConfig):
            raise ValueError("trainer.early_stopping must be an EarlyStoppingConfig.")
        if self.early_stopping.min_epochs > self.max_epochs:
            # A floor longer than the budget is simply unreachable, not an
            # error: '--override trainer.max_epochs=2' for a smoke test must
            # not have to restate early_stopping.min_epochs as well.
            object.__setattr__(
                self,
                "early_stopping",
                replace(self.early_stopping, min_epochs=self.max_epochs),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "max_gradient_norm": float(self.max_gradient_norm),
            "amp": self.amp,
            "device": self.device,
            "early_stopping": self.early_stopping.to_dict(),
        }


@dataclass(frozen=True)
class CheckpointConfig:
    """Where per-epoch state is written so a dropped Colab session is recoverable."""

    directory: Path | None = Path("trained-models/checkpoints")
    keep_last: int = 2
    save_best: bool = True
    resume: ResumeMode = "auto"

    def __post_init__(self) -> None:
        if self.directory is not None:
            object.__setattr__(self, "directory", Path(self.directory))
        _require_non_negative_integer(self.keep_last, "checkpoint.keep_last")
        if not isinstance(self.save_best, bool):
            raise ValueError("checkpoint.save_best must be a boolean.")
        if self.resume not in {"auto", "never"}:
            raise ValueError("checkpoint.resume must be 'auto' or 'never'.")

    @property
    def enabled(self) -> bool:
        return self.directory is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "directory": None if self.directory is None else str(self.directory),
            "keep_last": self.keep_last,
            "save_best": self.save_best,
            "resume": self.resume,
        }


@dataclass(frozen=True)
class WandbConfig:
    """Weights & Biases destination. Keep the project private."""

    enabled: bool = False
    entity: str | None = None
    project: str = "wm811k-wafer-defects"
    mode: WandbMode = "online"
    tags: tuple[str, ...] = ()
    notes: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("logging.wandb.enabled must be a boolean.")
        if self.entity is not None and not isinstance(self.entity, str):
            raise ValueError("logging.wandb.entity must be a string or null.")
        if not isinstance(self.project, str) or not self.project:
            raise ValueError("logging.wandb.project must be a non-empty string.")
        if self.mode not in {"online", "offline", "disabled"}:
            raise ValueError(
                "logging.wandb.mode must be online, offline, or disabled."
            )
        tags = tuple(self.tags)
        if not all(isinstance(tag, str) and tag for tag in tags):
            raise ValueError("logging.wandb.tags must be non-empty strings.")
        object.__setattr__(self, "tags", tags)
        if self.notes is not None and not isinstance(self.notes, str):
            raise ValueError("logging.wandb.notes must be a string or null.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "entity": self.entity,
            "project": self.project,
            "mode": self.mode,
            "tags": list(self.tags),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class LoggingConfig:
    """Where run artifacts are written and whether they are mirrored to W&B."""

    output_root: Path = Path("output/runs")
    wandb: WandbConfig = WandbConfig()

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        if not isinstance(self.wandb, WandbConfig):
            raise ValueError("logging.wandb must be a WandbConfig.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_root": str(self.output_root),
            "wandb": self.wandb.to_dict(),
        }


@dataclass(frozen=True)
class ExperimentConfig:
    """One complete, runnable experiment."""

    name: str
    seed: int = 86
    data: DataConfig = DataConfig()
    imbalance: ImbalanceConfig = ImbalanceConfig(
        name="inverse_sqrt_sampler",
        sampling="weighted",
        sampling_weighting="inverse_sqrt",
    )
    model: ModelConfig = ModelConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    trainer: TrainerConfig = TrainerConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    logging: LoggingConfig = LoggingConfig()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not RUN_NAME_PATTERN.fullmatch(self.name):
            raise ValueError(
                "name must start with an alphanumeric character and contain only "
                "letters, numbers, '.', '_' or '-'; it becomes a directory name."
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer.")
        sections = {
            "data": (self.data, DataConfig),
            "imbalance": (self.imbalance, ImbalanceConfig),
            "model": (self.model, ModelConfig),
            "optimizer": (self.optimizer, OptimizerConfig),
            "scheduler": (self.scheduler, SchedulerConfig),
            "trainer": (self.trainer, TrainerConfig),
            "checkpoint": (self.checkpoint, CheckpointConfig),
            "logging": (self.logging, LoggingConfig),
        }
        for section_name, (value, expected) in sections.items():
            if not isinstance(value, expected):
                raise ValueError(f"{section_name} must be a {expected.__name__}.")

    @property
    def run_directory(self) -> Path:
        return self.logging.output_root / self.name

    @property
    def checkpoint_directory(self) -> Path | None:
        if self.checkpoint.directory is None:
            return None
        return self.checkpoint.directory / self.name

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-Python view suitable for YAML, JSON, and W&B."""

        return {
            "name": self.name,
            "seed": self.seed,
            "data": self.data.to_dict(),
            "imbalance": self.imbalance.to_dict(),
            "model": self.model.to_dict(),
            "optimizer": self.optimizer.to_dict(),
            "scheduler": self.scheduler.to_dict(),
            "trainer": self.trainer.to_dict(),
            "checkpoint": self.checkpoint.to_dict(),
            "logging": self.logging.to_dict(),
        }
