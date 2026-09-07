"""Optimizer and learning-rate-schedule construction from a validated config."""

from __future__ import annotations

import re
from typing import Any

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from fdl_project.config.registry import (
    ScheduleContext,
    build_optimizer_from_groups,
    build_scheduler,
)
from fdl_project.config.schema import OptimizerConfig, SchedulerConfig

DEFAULT_GROUP_NAME = "default"


def build_parameter_groups(
    model: nn.Module, config: OptimizerConfig
) -> list[dict[str, Any]]:
    """Assign every trainable parameter to its first matching group.

    Patterns are tried in the order they appear in the config, so a specific
    pattern must come before a general one. A pattern that matches nothing
    raises: an accidentally empty encoder group is the classic way to believe
    you are fine-tuning at 1e-5 while actually training everything at 1e-3.
    """

    patterns = [
        (group, re.compile(group.pattern)) for group in config.param_groups
    ]
    buckets: list[list[nn.Parameter]] = [[] for _ in patterns]
    names: list[list[str]] = [[] for _ in patterns]
    default_bucket: list[nn.Parameter] = []
    default_names: list[str] = []

    for parameter_name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        for position, (_, compiled) in enumerate(patterns):
            if compiled.search(parameter_name):
                buckets[position].append(parameter)
                names[position].append(parameter_name)
                break
        else:
            default_bucket.append(parameter)
            default_names.append(parameter_name)

    groups: list[dict[str, Any]] = []
    for position, (group, _) in enumerate(patterns):
        if not buckets[position]:
            raise ValueError(
                f"optimizer.param_groups pattern {group.pattern!r} matched no "
                "trainable parameter. Check the module names on the model."
            )
        groups.append(
            {
                "params": buckets[position],
                "name": group.group_name,
                **dict(group.kwargs),
            }
        )
    if default_bucket:
        groups.append({"params": default_bucket, "name": DEFAULT_GROUP_NAME})
    if not groups:
        raise ValueError("The model exposes no trainable parameters to optimize.")
    return groups


def describe_parameter_groups(optimizer: Optimizer) -> list[dict[str, Any]]:
    """Summarise the resolved groups for the run metadata and the console."""

    summary = []
    for position, group in enumerate(optimizer.param_groups):
        summary.append(
            {
                "name": group.get("name", f"group_{position}"),
                "learning_rate": float(group["lr"]),
                "weight_decay": float(group.get("weight_decay", 0.0)),
                "num_tensors": len(group["params"]),
                "num_parameters": sum(
                    parameter.numel() for parameter in group["params"]
                ),
            }
        )
    return summary


def build_optimizer(model: nn.Module, config: OptimizerConfig) -> Optimizer:
    """Build the configured optimizer over the configured parameter groups."""

    groups = build_parameter_groups(model, config)
    return build_optimizer_from_groups(config.name, groups, **dict(config.kwargs))


def build_learning_rate_scheduler(
    optimizer: Optimizer,
    config: SchedulerConfig,
    *,
    max_epochs: int,
    steps_per_epoch: int,
) -> LRScheduler | ReduceLROnPlateau | None:
    """Build the configured schedule, or ``None`` for a constant rate."""

    if config.name is None:
        return None
    context = ScheduleContext(
        max_epochs=max_epochs,
        steps_per_epoch=max(1, steps_per_epoch),
        interval=config.interval,
    )
    return build_scheduler(config.name, optimizer, context, **dict(config.kwargs))
