"""Deterministic baseline training used by controlled WM-811K experiments."""

from __future__ import annotations

import copy
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from fdl_project.evaluation import EvaluationResult, evaluate_model


@dataclass(frozen=True)
class TrainingConfig:
    """Shared optimization budget for every imbalance strategy."""

    batch_size: int = 512
    max_epochs: int = 4
    minimum_epochs: int = 3
    early_stopping_patience: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_gradient_norm: float = 5.0

    def __post_init__(self) -> None:
        integer_fields = {
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "minimum_epochs": self.minimum_epochs,
            "early_stopping_patience": self.early_stopping_patience,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.minimum_epochs > self.max_epochs:
            raise ValueError("minimum_epochs cannot exceed max_epochs.")
        for name, value in {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_gradient_norm": self.max_gradient_norm,
        }.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite.")
        if (
            self.learning_rate <= 0
            or self.weight_decay < 0
            or self.max_gradient_norm <= 0
        ):
            raise ValueError(
                "learning_rate and max_gradient_norm must be positive; "
                "weight_decay must be non-negative."
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FitResult:
    """Best validation state and complete epoch history for one run."""

    best_epoch: int
    best_validation: EvaluationResult
    best_state_dict: dict[str, Tensor]
    history: pd.DataFrame
    stopped_early: bool


def set_reproducible_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible CPU experiments."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _unpack_training_batch(batch: Any) -> tuple[Tensor, Tensor]:
    if not isinstance(batch, (tuple, list)) or len(batch) != 3:
        raise TypeError(
            "Training batches must be (inputs, targets, source_row_indices)."
        )
    inputs, targets, _ = batch
    if not isinstance(inputs, Tensor) or not isinstance(targets, Tensor):
        raise TypeError("Training inputs and targets must be tensors.")
    return inputs, targets


def train_one_epoch(
    model: nn.Module,
    dataloader: Any,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    *,
    device: str | torch.device,
    max_gradient_norm: float,
) -> dict[str, float | int]:
    """Train one complete epoch and report sample-weighted diagnostics."""

    device_object = torch.device(device)
    model.to(device_object)
    criterion.to(device_object)
    model.train()
    sample_count = 0
    correct_count = 0
    accumulated_loss = 0.0

    for batch in dataloader:
        inputs, targets = _unpack_training_batch(batch)
        inputs = inputs.to(device_object)
        targets = targets.to(device_object, dtype=torch.long)
        if targets.ndim != 1 or len(targets) != len(inputs):
            raise ValueError("Training targets must contain one label per input.")

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise ValueError("Training criterion must return one finite scalar loss.")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_gradient_norm
        )
        if not torch.isfinite(gradient_norm):
            raise ValueError("Training produced non-finite gradients.")
        optimizer.step()

        batch_size = len(targets)
        sample_count += batch_size
        accumulated_loss += float(loss.detach().item()) * batch_size
        correct_count += int((logits.detach().argmax(dim=1) == targets).sum().item())

    if sample_count == 0:
        raise ValueError("Training DataLoader did not yield any samples.")
    return {
        "train_loss": accumulated_loss / sample_count,
        "train_accuracy": correct_count / sample_count,
        "train_samples": sample_count,
    }


def fit_model(
    model: nn.Module,
    train_dataloader: Any,
    validation_dataloader: Any,
    training_criterion: nn.Module,
    config: TrainingConfig,
    *,
    device: str | torch.device = "cpu",
    epoch_callback: Callable[[dict[str, float | int]], None] | None = None,
) -> FitResult:
    """Fit with validation macro-F1 selection and deterministic early stopping."""

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    validation_criterion = nn.CrossEntropyLoss()
    history_rows: list[dict[str, float | int]] = []
    best_macro_f1 = float("-inf")
    best_epoch = 0
    best_state_dict: dict[str, Tensor] | None = None
    best_validation: EvaluationResult | None = None
    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(1, config.max_epochs + 1):
        training_metrics = train_one_epoch(
            model,
            train_dataloader,
            optimizer,
            training_criterion,
            device=device,
            max_gradient_norm=config.max_gradient_norm,
        )
        validation = evaluate_model(
            model,
            validation_dataloader,
            device=device,
            criterion=validation_criterion,
            split_name="validation",
        )
        macro_f1 = float(validation.metrics["macro_f1"])
        epoch_metrics: dict[str, float | int] = {
            "epoch": epoch,
            **training_metrics,
            "validation_loss": float(validation.metrics["mean_loss"]),
            "validation_accuracy": float(validation.metrics["accuracy"]),
            "validation_balanced_accuracy": float(
                validation.metrics["balanced_accuracy"]
            ),
            "validation_macro_f1": macro_f1,
            "validation_weighted_f1": float(validation.metrics["weighted_f1"]),
        }
        history_rows.append(epoch_metrics)
        if epoch_callback is not None:
            epoch_callback(dict(epoch_metrics))

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_epoch = epoch
            best_state_dict = copy.deepcopy(
                {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                }
            )
            best_validation = validation
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (
            epoch >= config.minimum_epochs
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            stopped_early = True
            break

    assert best_state_dict is not None and best_validation is not None
    model.load_state_dict(best_state_dict)
    return FitResult(
        best_epoch=best_epoch,
        best_validation=best_validation,
        best_state_dict=best_state_dict,
        history=pd.DataFrame(history_rows),
        stopped_early=stopped_early,
    )
