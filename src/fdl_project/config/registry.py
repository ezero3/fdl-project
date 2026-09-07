"""Name-to-builder tables that turn config strings into PyTorch objects.

An explicit registry rather than an import path in the YAML: a typo names a
key that does not exist and the error lists what does, instead of importing
something arbitrary. Builders forward ``kwargs`` straight through to the
underlying class, so nothing PyTorch can express is unreachable from a config.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

ModelBuilder = Callable[..., nn.Module]
OptimizerBuilder = Callable[..., Optimizer]


def _lookup(table: dict[str, Any], name: str, *, kind: str) -> Any:
    try:
        return table[name]
    except KeyError:
        available = ", ".join(sorted(table))
        raise KeyError(
            f"Unknown {kind} {name!r}. Available: {available}."
        ) from None


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


def _baseline_cnn(**kwargs: Any) -> nn.Module:
    from fdl_project.models.baseline_cnn import BaselineCNN

    return BaselineCNN(**kwargs)


def _pretrained(architecture: str) -> ModelBuilder:
    def build(**kwargs: Any) -> nn.Module:
        from fdl_project.models.pretrained import PretrainedClassifier

        return PretrainedClassifier(architecture=architecture, **kwargs)

    return build


MODEL_REGISTRY: dict[str, ModelBuilder] = {
    "baseline_cnn": _baseline_cnn,
    "resnet18": _pretrained("resnet18"),
    "resnet34": _pretrained("resnet34"),
    "mobilenet_v3_small": _pretrained("mobilenet_v3_small"),
    "mobilenet_v3_large": _pretrained("mobilenet_v3_large"),
    "efficientnet_b0": _pretrained("efficientnet_b0"),
}


def build_model(name: str, **kwargs: Any) -> nn.Module:
    """Instantiate a registered model."""

    return _lookup(MODEL_REGISTRY, name, kind="model")(**kwargs)


def available_models() -> tuple[str, ...]:
    return tuple(sorted(MODEL_REGISTRY))


# --------------------------------------------------------------------------
# Optimizers
# --------------------------------------------------------------------------

OPTIMIZER_REGISTRY: dict[str, OptimizerBuilder] = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "rmsprop": torch.optim.RMSprop,
    "adadelta": torch.optim.Adadelta,
    "nadam": torch.optim.NAdam,
}


def build_optimizer_from_groups(
    name: str, parameter_groups: list[dict[str, Any]], **defaults: Any
) -> Optimizer:
    """Instantiate a registered optimizer over pre-built parameter groups."""

    return _lookup(OPTIMIZER_REGISTRY, name, kind="optimizer")(
        parameter_groups, **defaults
    )


def available_optimizers() -> tuple[str, ...]:
    return tuple(sorted(OPTIMIZER_REGISTRY))


# --------------------------------------------------------------------------
# Learning-rate schedules
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ScheduleContext:
    """Run-shape facts a schedule needs but the YAML should not have to repeat."""

    max_epochs: int
    steps_per_epoch: int
    interval: str

    @property
    def total_steps(self) -> int:
        """How many times ``scheduler.step()`` will be called over the run."""

        if self.interval == "step":
            return max(1, self.max_epochs * self.steps_per_epoch)
        return max(1, self.max_epochs)


def _cosine(
    optimizer: Optimizer, context: ScheduleContext, **kwargs: Any
) -> LRScheduler:
    kwargs.setdefault("T_max", context.total_steps)
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)


def _cosine_warm_restarts(
    optimizer: Optimizer, context: ScheduleContext, **kwargs: Any
) -> LRScheduler:
    kwargs.setdefault("T_0", max(1, context.total_steps // 4))
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **kwargs)


def _step(optimizer: Optimizer, context: ScheduleContext, **kwargs: Any) -> LRScheduler:
    kwargs.setdefault("step_size", max(1, context.total_steps // 3))
    return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)


def _multistep(
    optimizer: Optimizer, context: ScheduleContext, **kwargs: Any
) -> LRScheduler:
    if "milestones" not in kwargs:
        raise ValueError("The multistep schedule requires explicit 'milestones'.")
    return torch.optim.lr_scheduler.MultiStepLR(optimizer, **kwargs)


def _exponential(
    optimizer: Optimizer, context: ScheduleContext, **kwargs: Any
) -> LRScheduler:
    kwargs.setdefault("gamma", 0.95)
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)


def _one_cycle(
    optimizer: Optimizer, context: ScheduleContext, **kwargs: Any
) -> LRScheduler:
    if context.interval != "step":
        raise ValueError("The onecycle schedule must use scheduler.interval='step'.")
    kwargs.setdefault("total_steps", context.total_steps)
    kwargs.setdefault(
        "max_lr", [group["lr"] for group in optimizer.param_groups]
    )
    return torch.optim.lr_scheduler.OneCycleLR(optimizer, **kwargs)


def _plateau(
    optimizer: Optimizer, context: ScheduleContext, **kwargs: Any
) -> ReduceLROnPlateau:
    if context.interval != "epoch":
        raise ValueError(
            "The plateau schedule reads a validation metric, so it must use "
            "scheduler.interval='epoch'."
        )
    kwargs.setdefault("mode", "max")
    kwargs.setdefault("patience", 2)
    return ReduceLROnPlateau(optimizer, **kwargs)


def _cosine_with_warmup(
    optimizer: Optimizer, context: ScheduleContext, **kwargs: Any
) -> LRScheduler:
    """Linear warmup into cosine decay, as a multiplier on each group's own LR.

    PyTorch has no built-in warmup. Expressing the floor as a *ratio* rather
    than an absolute learning rate is what keeps this correct for
    discriminative fine-tuning: the encoder and the head keep their 100x
    separation for the whole run instead of converging to a shared floor.
    """

    warmup_epochs = int(kwargs.pop("warmup_epochs", 0))
    warmup_steps = int(kwargs.pop("warmup_steps", 0))
    minimum_ratio = float(kwargs.pop("min_lr_ratio", 0.0))
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise ValueError(
            f"cosine_with_warmup received unsupported options: {unexpected}. "
            "It accepts warmup_epochs, warmup_steps, and min_lr_ratio."
        )
    if warmup_epochs and warmup_steps:
        raise ValueError("Set warmup_epochs or warmup_steps, not both.")
    if context.interval == "epoch":
        if warmup_steps:
            raise ValueError(
                "warmup_steps requires scheduler.interval='step'; use warmup_epochs."
            )
        warmup = warmup_epochs
    else:
        warmup = warmup_steps or warmup_epochs * context.steps_per_epoch
    if not 0.0 <= minimum_ratio < 1.0:
        raise ValueError("min_lr_ratio must lie in [0, 1).")
    total = context.total_steps
    if not 0 <= warmup < total:
        raise ValueError(
            f"warmup must be non-negative and shorter than the run ({total} steps)."
        )

    def factor(step: int) -> float:
        if warmup and step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)


SCHEDULER_REGISTRY: dict[str, Callable[..., LRScheduler | ReduceLROnPlateau]] = {
    "cosine": _cosine,
    "cosine_warm_restarts": _cosine_warm_restarts,
    "cosine_with_warmup": _cosine_with_warmup,
    "step": _step,
    "multistep": _multistep,
    "exponential": _exponential,
    "onecycle": _one_cycle,
    "plateau": _plateau,
}


def build_scheduler(
    name: str,
    optimizer: Optimizer,
    context: ScheduleContext,
    **kwargs: Any,
) -> LRScheduler | ReduceLROnPlateau:
    """Instantiate a registered learning-rate schedule."""

    builder = _lookup(SCHEDULER_REGISTRY, name, kind="scheduler")
    return builder(optimizer, context, **kwargs)


def available_schedulers() -> tuple[str, ...]:
    return tuple(sorted(SCHEDULER_REGISTRY))
