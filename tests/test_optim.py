"""Tests for parameter-group assignment and learning-rate schedules."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from fdl_project.config.registry import ScheduleContext, build_scheduler
from fdl_project.config.schema import OptimizerConfig, ParameterGroupConfig, SchedulerConfig
from fdl_project.training.optim import (
    build_learning_rate_scheduler,
    build_optimizer,
    build_parameter_groups,
    describe_parameter_groups,
)


class TwoPartModel(nn.Module):
    """Stands in for a pretrained backbone plus a freshly initialised head."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 16))
        self.head = nn.Linear(16, 9)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(inputs))


def _finetune_config(**kwargs: object) -> OptimizerConfig:
    return OptimizerConfig(
        name="adamw",
        kwargs={"lr": 1e-3, "weight_decay": 1e-4},
        param_groups=(
            ParameterGroupConfig(
                pattern="^encoder\\.", kwargs={"lr": 1e-5}, name="encoder"
            ),
        ),
        **kwargs,
    )


def test_the_encoder_gets_its_own_learning_rate() -> None:
    optimizer = build_optimizer(TwoPartModel(), _finetune_config())
    groups = {group["name"]: group for group in optimizer.param_groups}

    assert groups["encoder"]["lr"] == 1e-5
    assert groups["default"]["lr"] == 1e-3
    # the defaults still apply to keys the group did not override
    assert groups["encoder"]["weight_decay"] == 1e-4


def test_every_trainable_parameter_lands_in_exactly_one_group() -> None:
    model = TwoPartModel()
    groups = build_parameter_groups(model, _finetune_config())

    assigned = [id(parameter) for group in groups for parameter in group["params"]]
    expected = [id(parameter) for parameter in model.parameters()]
    assert sorted(assigned) == sorted(expected)
    assert len(assigned) == len(set(assigned))


def test_a_pattern_that_matches_nothing_is_an_error() -> None:
    """Silently training everything at the head's rate is the failure to prevent."""

    config = OptimizerConfig(
        param_groups=(
            ParameterGroupConfig(pattern="^backbone\\.", kwargs={"lr": 1e-5}),
        )
    )
    with pytest.raises(ValueError, match="matched no"):
        build_optimizer(TwoPartModel(), config)


def test_the_first_matching_pattern_wins() -> None:
    config = OptimizerConfig(
        param_groups=(
            ParameterGroupConfig(pattern="^encoder\\.0\\.", kwargs={"lr": 1e-6}),
            ParameterGroupConfig(pattern="^encoder\\.", kwargs={"lr": 1e-5}),
        )
    )
    optimizer = build_optimizer(TwoPartModel(), config)
    by_rate = {
        group["learning_rate"]: group["num_tensors"]
        for group in describe_parameter_groups(optimizer)
    }

    assert by_rate[1e-6] == 2  # encoder.0 weight and bias
    assert by_rate[1e-5] == 2  # encoder.1 weight and bias


def test_frozen_parameters_are_left_out_of_the_optimizer() -> None:
    model = TwoPartModel()
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False

    optimizer = build_optimizer(model, OptimizerConfig())
    trainable = sum(group["num_parameters"] for group in describe_parameter_groups(optimizer))
    assert trainable == sum(p.numel() for p in model.head.parameters())


def test_a_model_with_nothing_to_train_is_rejected() -> None:
    model = TwoPartModel()
    for parameter in model.parameters():
        parameter.requires_grad = False
    with pytest.raises(ValueError, match="no trainable parameters"):
        build_optimizer(model, OptimizerConfig())


def test_no_scheduler_name_means_a_constant_rate() -> None:
    optimizer = build_optimizer(TwoPartModel(), OptimizerConfig())
    assert (
        build_learning_rate_scheduler(
            optimizer, SchedulerConfig(), max_epochs=10, steps_per_epoch=4
        )
        is None
    )


@pytest.mark.parametrize("spelling", ["none", "null", ""])
def test_the_scheduler_can_be_switched_off_from_yaml(spelling: str) -> None:
    assert SchedulerConfig(name=spelling).name is None


def test_warmup_then_cosine_keeps_the_ratio_between_groups() -> None:
    """Discriminative fine-tuning must survive the whole schedule."""

    optimizer = build_optimizer(TwoPartModel(), _finetune_config())
    scheduler = build_learning_rate_scheduler(
        optimizer,
        SchedulerConfig(
            name="cosine_with_warmup",
            kwargs={"warmup_epochs": 2, "min_lr_ratio": 0.01},
        ),
        max_epochs=10,
        steps_per_epoch=4,
    )

    observed = []
    for _ in range(10):
        observed.append([group["lr"] for group in optimizer.param_groups])
        scheduler.step()

    encoder_rates = [rates[0] for rates in observed]
    head_rates = [rates[1] for rates in observed]
    # warmup climbs, then cosine decays
    assert head_rates[0] < head_rates[1] <= head_rates[2]
    assert head_rates[-1] < head_rates[2]
    # the 100x separation holds at every step
    for encoder_rate, head_rate in zip(encoder_rates, head_rates, strict=True):
        assert head_rate == pytest.approx(encoder_rate * 100, rel=1e-6)


def test_the_cosine_floor_is_a_ratio_of_each_group_rate() -> None:
    optimizer = build_optimizer(TwoPartModel(), _finetune_config())
    scheduler = build_learning_rate_scheduler(
        optimizer,
        SchedulerConfig(name="cosine_with_warmup", kwargs={"min_lr_ratio": 0.1}),
        max_epochs=4,
        steps_per_epoch=1,
    )
    for _ in range(4):
        scheduler.step()

    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-5 * 0.1)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(1e-3 * 0.1)


def test_cosine_defaults_its_horizon_to_the_run_length() -> None:
    optimizer = build_optimizer(TwoPartModel(), OptimizerConfig())
    scheduler = build_learning_rate_scheduler(
        optimizer, SchedulerConfig(name="cosine"), max_epochs=6, steps_per_epoch=3
    )
    assert scheduler.T_max == 6


def test_a_step_interval_schedule_counts_optimizer_steps() -> None:
    optimizer = build_optimizer(TwoPartModel(), OptimizerConfig())
    scheduler = build_learning_rate_scheduler(
        optimizer,
        SchedulerConfig(name="cosine", interval="step"),
        max_epochs=6,
        steps_per_epoch=3,
    )
    assert scheduler.T_max == 18


def test_onecycle_refuses_an_epoch_interval() -> None:
    optimizer = build_optimizer(TwoPartModel(), OptimizerConfig())
    with pytest.raises(ValueError, match="interval='step'"):
        build_learning_rate_scheduler(
            optimizer,
            SchedulerConfig(name="onecycle"),
            max_epochs=4,
            steps_per_epoch=2,
        )


def test_plateau_refuses_a_step_interval() -> None:
    optimizer = build_optimizer(TwoPartModel(), OptimizerConfig())
    with pytest.raises(ValueError, match="interval='epoch'"):
        build_learning_rate_scheduler(
            optimizer,
            SchedulerConfig(name="plateau", interval="step"),
            max_epochs=4,
            steps_per_epoch=2,
        )


def test_an_unknown_schedule_lists_the_available_ones() -> None:
    optimizer = build_optimizer(TwoPartModel(), OptimizerConfig())
    with pytest.raises(KeyError, match="cosine"):
        build_scheduler(
            "magic",
            optimizer,
            ScheduleContext(max_epochs=4, steps_per_epoch=2, interval="epoch"),
        )


def test_unsupported_warmup_options_are_named() -> None:
    optimizer = build_optimizer(TwoPartModel(), OptimizerConfig())
    with pytest.raises(ValueError, match="min_lr"):
        build_learning_rate_scheduler(
            optimizer,
            SchedulerConfig(name="cosine_with_warmup", kwargs={"min_lr": 1e-6}),
            max_epochs=4,
            steps_per_epoch=2,
        )
