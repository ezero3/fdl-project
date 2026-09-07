"""The training loop: one epoch at a time, with validation-driven selection.

The loop owns optimization and nothing else. The optimizer, the schedule and
the criterion are built outside and handed in, so an experiment can change any
of them from YAML without this file knowing about it; everything that reacts to
an epoch finishing is a callback.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from fdl_project.config.schema import TrainerConfig
from fdl_project.evaluation import EvaluationResult, evaluate_model
from fdl_project.training.checkpoint import ResumeState

EpochCallback = Callable[[dict[str, float | int]], None]


@dataclass(frozen=True)
class FitResult:
    """Best validation state and complete epoch history for one run."""

    best_epoch: int
    best_metric: float
    best_validation: EvaluationResult
    best_state_dict: dict[str, Tensor]
    history: pd.DataFrame
    stopped_early: bool
    last_epoch: int


def resolve_device(requested: str | torch.device = "auto") -> torch.device:
    """Turn ``auto`` into the best device actually available here."""

    if isinstance(requested, torch.device):
        return requested
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _unpack_training_batch(batch: Any) -> tuple[Tensor, Tensor]:
    if not isinstance(batch, (tuple, list)) or len(batch) != 3:
        raise TypeError(
            "Training batches must be (inputs, targets, source_row_indices)."
        )
    inputs, targets, _ = batch
    if not isinstance(inputs, Tensor) or not isinstance(targets, Tensor):
        raise TypeError("Training inputs and targets must be tensors.")
    return inputs, targets


def _is_improvement(candidate: float, incumbent: float | None, mode: str) -> bool:
    if incumbent is None:
        return True
    return candidate > incumbent if mode == "max" else candidate < incumbent


def _current_learning_rate(optimizer: Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _reseed_sampler(dataloader: Any, seed: int | None, epoch: int) -> None:
    """Give each epoch its own sampler draw, reproducibly.

    ``WeightedRandomSampler`` holds one generator for the whole run, so a run
    resumed at epoch 12 would otherwise replay epoch 1's draw. Deriving the
    seed from the epoch makes the resumed half identical to an uninterrupted
    run, which is the whole point of checkpointing.
    """

    if seed is None:
        return
    sampler = getattr(dataloader, "sampler", None)
    generator = getattr(sampler, "generator", None)
    if generator is not None:
        generator.manual_seed(seed + epoch)


def train_one_epoch(
    model: nn.Module,
    dataloader: Any,
    optimizer: Optimizer,
    criterion: nn.Module,
    *,
    device: str | torch.device,
    max_gradient_norm: float,
    scaler: torch.amp.GradScaler | None = None,
    scheduler: Any = None,
    scheduler_interval: str = "epoch",
    batch_transform: Any = None,
) -> dict[str, float | int]:
    """Train one complete epoch and report sample-weighted diagnostics."""

    device_object = torch.device(device)
    model.to(device_object)
    criterion.to(device_object)
    model.train()
    sample_count = 0
    correct_count = 0
    accumulated_loss = 0.0
    # The scaler is created disabled off CUDA, so it alone decides whether this
    # epoch runs in mixed precision.
    use_amp = scaler is not None and scaler.is_enabled()

    for batch in dataloader:
        inputs, targets = _unpack_training_batch(batch)
        # pin_memory is on, so a non-blocking copy lets the transfer overlap
        # the previous batch's compute. CUDA ordering makes the following
        # forward wait for it, so no explicit synchronise is needed.
        inputs = inputs.to(device_object, non_blocking=True)
        targets = targets.to(device_object, dtype=torch.long, non_blocking=True)
        if batch_transform is not None:
            # Categorical uint8 arrived; augment and encode on the device.
            inputs = batch_transform(inputs, training=True)
        if targets.ndim != 1 or len(targets) != len(inputs):
            raise ValueError("Training targets must contain one label per input.")

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device_object.type, enabled=use_amp):
            logits = model(inputs)
            loss = criterion(logits, targets)
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise ValueError("Training criterion must return one finite scalar loss.")

        if use_amp:
            scaler.scale(loss).backward()
            # Gradients must be unscaled before the norm is meaningful.
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_gradient_norm
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_gradient_norm
            )
            if not torch.isfinite(gradient_norm):
                raise ValueError("Training produced non-finite gradients.")
            optimizer.step()

        if scheduler is not None and scheduler_interval == "step":
            scheduler.step()

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
    config: TrainerConfig,
    *,
    optimizer: Optimizer,
    scheduler: Any = None,
    scheduler_interval: str = "epoch",
    device: str | torch.device = "cpu",
    callbacks: Sequence[Any] = (),
    epoch_callback: EpochCallback | None = None,
    sampler_seed: int | None = None,
    resume: ResumeState | None = None,
    batch_transform: Any = None,
) -> FitResult:
    """Fit until the monitored validation metric stops improving.

    Pass ``resume`` -- the state ``CheckpointManager.resume`` returns -- to
    continue a checkpointed run as if it had never stopped. The four values it
    carries only make sense together, which is why they travel as one object.
    """

    start_epoch = 1 if resume is None else resume.next_epoch
    history = None if resume is None else resume.history
    best_metric = None if resume is None else resume.best_metric
    best_epoch = 0 if resume is None else resume.best_epoch

    device_object = resolve_device(device)
    early_stopping = config.early_stopping
    scaler = torch.amp.GradScaler(
        device_object.type, enabled=config.amp and device_object.type == "cuda"
    )
    validation_criterion = nn.CrossEntropyLoss()

    history_rows: list[dict[str, Any]] = list(history or [])
    best_state_dict: dict[str, Tensor] | None = None
    best_validation: EvaluationResult | None = None
    stopped_early = False
    last_epoch = start_epoch - 1
    # Derived, not stored: how many epochs have passed since the best one. A
    # resumed run must inherit this, or it silently gets `patience` extra
    # epochs and outlives an uninterrupted run with the same config.
    epochs_without_improvement = max(0, last_epoch - best_epoch)

    context: dict[str, Any] = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "scaler": scaler,
        "device": str(device_object),
        "max_epochs": config.max_epochs,
        "history": history_rows,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
    }
    for callback in callbacks:
        callback.on_train_start(context)

    for epoch in range(start_epoch, config.max_epochs + 1):
        started = time.monotonic()
        _reseed_sampler(train_dataloader, sampler_seed, epoch)
        learning_rate = _current_learning_rate(optimizer)

        training_metrics = train_one_epoch(
            model,
            train_dataloader,
            optimizer,
            training_criterion,
            device=device_object,
            max_gradient_norm=config.max_gradient_norm,
            scaler=scaler,
            scheduler=scheduler,
            scheduler_interval=scheduler_interval,
            batch_transform=batch_transform,
        )
        validation = evaluate_model(
            model,
            validation_dataloader,
            device=device_object,
            criterion=validation_criterion,
            split_name="validation",
            batch_transform=batch_transform,
        )
        epoch_metrics: dict[str, Any] = {
            "epoch": epoch,
            **training_metrics,
            "validation_loss": float(validation.metrics["mean_loss"]),
            "validation_accuracy": float(validation.metrics["accuracy"]),
            "validation_balanced_accuracy": float(
                validation.metrics["balanced_accuracy"]
            ),
            "validation_macro_f1": float(validation.metrics["macro_f1"]),
            "validation_weighted_f1": float(validation.metrics["weighted_f1"]),
            "learning_rate": learning_rate,
            "epoch_seconds": time.monotonic() - started,
        }
        if early_stopping.monitor not in epoch_metrics:
            available = ", ".join(sorted(epoch_metrics))
            raise ValueError(
                f"Monitored metric {early_stopping.monitor!r} is not produced by "
                f"the loop. Available: {available}."
            )
        monitored = float(epoch_metrics[early_stopping.monitor])

        if scheduler is not None and scheduler_interval == "epoch":
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(monitored)
            else:
                scheduler.step()

        history_rows.append(epoch_metrics)
        last_epoch = epoch

        is_best = _is_improvement(monitored, best_metric, early_stopping.mode)
        if is_best:
            best_metric = monitored
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

        context.update(
            {
                "epoch": epoch,
                "validation": validation,
                "is_best": is_best,
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "learning_rate": learning_rate,
                "history": history_rows,
            }
        )
        if epoch_callback is not None:
            epoch_callback(dict(epoch_metrics))
        for callback in callbacks:
            callback.on_epoch_end(dict(epoch_metrics), context)

        if (
            epoch >= early_stopping.min_epochs
            and epochs_without_improvement >= early_stopping.patience
        ):
            stopped_early = True
            break

    if not history_rows or last_epoch < start_epoch:
        raise ValueError(
            "Training completed without evaluating a single epoch; check "
            "start_epoch against trainer.max_epochs."
        )
    if best_state_dict is None or best_validation is None:
        # A resumed run whose remaining epochs never beat the incoming best.
        # Report the resumed baseline together with the final weights; the
        # historical best weights are in best.pt, which this loop never wrote.
        best_state_dict = {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        }
        best_validation = validation

    model.load_state_dict(best_state_dict)
    context.update({"stopped_early": stopped_early, "last_epoch": last_epoch})
    for callback in callbacks:
        callback.on_train_end(context)

    return FitResult(
        best_epoch=best_epoch,
        best_metric=float(best_metric),
        best_validation=best_validation,
        best_state_dict=best_state_dict,
        history=pd.DataFrame(history_rows),
        stopped_early=stopped_early,
        last_epoch=last_epoch,
    )
