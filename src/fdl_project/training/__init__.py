"""Training loop, optimization, seeding, checkpointing, and callbacks."""

from fdl_project.training.callbacks import (
    BaseCallback,
    Callback,
    CheckpointCallback,
    ConsoleLogger,
    WandbLogger,
    build_callbacks,
)
from fdl_project.training.checkpoint import (
    CheckpointManager,
    ResumeState,
    load_checkpoint,
)
from fdl_project.training.loop import (
    FitResult,
    fit_model,
    resolve_device,
    train_one_epoch,
)
from fdl_project.training.optim import (
    build_learning_rate_scheduler,
    build_optimizer,
    build_parameter_groups,
    describe_parameter_groups,
)
from fdl_project.training.seed import seed_everything, seed_worker

__all__ = [
    "BaseCallback",
    "Callback",
    "CheckpointCallback",
    "CheckpointManager",
    "ConsoleLogger",
    "FitResult",
    "ResumeState",
    "WandbLogger",
    "build_callbacks",
    "build_learning_rate_scheduler",
    "build_optimizer",
    "build_parameter_groups",
    "describe_parameter_groups",
    "fit_model",
    "load_checkpoint",
    "resolve_device",
    "seed_everything",
    "seed_worker",
    "train_one_epoch",
]
