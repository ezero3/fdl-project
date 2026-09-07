"""Architecture-independent metric computation for one evaluated split."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch import Tensor, nn

from fdl_project.constants import CLASS_NAMES, NUM_CLASSES, validate_class_names

_SUPPORTED_SPLITS = frozenset({"validation", "test"})


@dataclass(frozen=True)
class CollectedPredictions:
    """CPU-side outputs collected from one complete evaluation DataLoader."""

    row_indices: np.ndarray
    true_indices: np.ndarray
    predicted_indices: np.ndarray
    probabilities: np.ndarray
    mean_loss: float | None
    model_class: str
    device: str


@dataclass(frozen=True)
class EvaluationResult:
    """Metrics, tables, and matrices for one evaluated model and split."""

    metrics: dict[str, float | int | None]
    per_class_metrics: pd.DataFrame
    predictions: pd.DataFrame
    confusion_matrix: np.ndarray
    normalized_confusion_matrix: np.ndarray
    class_names: tuple[str, ...]
    split_name: str
    model_class: str
    device: str


def _validate_split_name(split_name: str) -> str:
    if split_name not in _SUPPORTED_SPLITS:
        raise ValueError(
            f"split_name must be one of {sorted(_SUPPORTED_SPLITS)!r}; "
            f"received {split_name!r}."
        )
    return split_name


def _first_present(mapping: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    raise KeyError(f"Expected one of the batch keys {tuple(names)!r}.")


def _unpack_batch(batch: Any) -> tuple[Tensor, Any, Any]:
    if isinstance(batch, Mapping):
        inputs = _first_present(batch, ("inputs", "images"))
        targets = _first_present(batch, ("targets", "labels"))
        row_indices = _first_present(batch, ("row_indices", "indices"))
        return inputs, targets, row_indices

    if isinstance(batch, (tuple, list)) and len(batch) == 3:
        inputs, targets, row_indices = batch
        return inputs, targets, row_indices

    raise TypeError(
        "Each evaluation batch must be a three-item tuple/list "
        "(inputs, targets, row_indices), or a mapping containing equivalent keys."
    )


def _extract_logits(model_output: Any) -> Tensor:
    if isinstance(model_output, Tensor):
        return model_output
    if isinstance(model_output, Mapping) and isinstance(
        model_output.get("logits"), Tensor
    ):
        return model_output["logits"]
    if isinstance(getattr(model_output, "logits", None), Tensor):
        return model_output.logits
    if (
        isinstance(model_output, (tuple, list))
        and model_output
        and isinstance(model_output[0], Tensor)
    ):
        return model_output[0]
    raise TypeError(
        "The model must return a logits Tensor, an object/mapping with a logits Tensor, "
        "or a tuple/list whose first item is a logits Tensor."
    )


def _as_one_dimensional_indices(
    values: Any, *, name: str, batch_size: int
) -> np.ndarray:
    if isinstance(values, Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)

    if array.ndim != 1 or len(array) != batch_size:
        raise ValueError(
            f"{name} must be one-dimensional with one value per sample; "
            f"received shape {array.shape!r} for batch size {batch_size}."
        )
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must contain integers; received dtype {array.dtype}.")
    return array.astype(np.int64, copy=False)


def _loss_sum_for_batch(loss: Tensor, criterion: Any, batch_size: int) -> float:
    if not isinstance(loss, Tensor):
        raise TypeError("criterion must return a torch.Tensor.")
    detached = loss.detach()
    if detached.numel() == 1:
        reduction = getattr(criterion, "reduction", "mean")
        value = float(detached.item())
        return value if reduction == "sum" else value * batch_size
    if detached.numel() != batch_size:
        raise ValueError(
            "A non-scalar criterion output must contain exactly one loss per sample."
        )
    return float(detached.sum().item())


def collect_predictions(
    model: nn.Module,
    dataloader: Iterable[Any],
    device: str | torch.device,
    criterion: Any | None = None,
    *,
    class_names: Iterable[str] = CLASS_NAMES,
) -> CollectedPredictions:
    """Run a complete DataLoader in inference mode and collect CPU-side outputs.

    Batches must include the original dataset row index so saved predictions remain
    traceable to ``WM811K.pkl``. The model is moved to ``device`` and left in eval mode.
    """

    names = validate_class_names(class_names)
    device_object = torch.device(device)
    model.to(device_object)
    model.eval()

    all_row_indices: list[np.ndarray] = []
    all_true_indices: list[np.ndarray] = []
    all_predicted_indices: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    total_loss = 0.0
    total_samples = 0

    with torch.inference_mode():
        for batch in dataloader:
            inputs, targets, row_indices = _unpack_batch(batch)
            if not isinstance(inputs, Tensor):
                raise TypeError("Batch inputs must be a torch.Tensor.")

            target_tensor = torch.as_tensor(
                targets, dtype=torch.long, device=device_object
            )
            if target_tensor.ndim != 1:
                raise ValueError(
                    f"Batch targets must be one-dimensional; received {tuple(target_tensor.shape)!r}."
                )
            batch_size = int(target_tensor.shape[0])
            if batch_size == 0:
                raise ValueError("Evaluation batches must not be empty.")
            if torch.any((target_tensor < 0) | (target_tensor >= len(names))):
                raise ValueError(
                    f"Targets must contain class indices in [0, {len(names) - 1}]."
                )

            logits = _extract_logits(model(inputs.to(device_object)))
            if logits.ndim != 2 or tuple(logits.shape) != (batch_size, len(names)):
                raise ValueError(
                    "Model logits must have shape "
                    f"(batch_size, {len(names)}); received {tuple(logits.shape)!r}."
                )
            if not torch.isfinite(logits).all():
                raise ValueError("Model logits contain NaN or infinite values.")

            probabilities = torch.softmax(logits, dim=1)
            predicted_indices = logits.argmax(dim=1)
            row_index_array = _as_one_dimensional_indices(
                row_indices, name="row_indices", batch_size=batch_size
            )

            if criterion is not None:
                batch_loss = criterion(logits, target_tensor)
                total_loss += _loss_sum_for_batch(batch_loss, criterion, batch_size)

            all_row_indices.append(row_index_array)
            all_true_indices.append(target_tensor.detach().cpu().numpy())
            all_predicted_indices.append(predicted_indices.detach().cpu().numpy())
            all_probabilities.append(probabilities.detach().cpu().numpy())
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("The evaluation DataLoader did not yield any samples.")

    collected_row_indices = np.concatenate(all_row_indices).astype(np.int64, copy=False)
    if len(np.unique(collected_row_indices)) != total_samples:
        raise ValueError("Evaluation row_indices must be unique across the DataLoader.")

    return CollectedPredictions(
        row_indices=collected_row_indices,
        true_indices=np.concatenate(all_true_indices).astype(np.int64, copy=False),
        predicted_indices=np.concatenate(all_predicted_indices).astype(
            np.int64, copy=False
        ),
        probabilities=np.concatenate(all_probabilities),
        mean_loss=(total_loss / total_samples) if criterion is not None else None,
        model_class=model.__class__.__name__,
        device=str(device_object),
    )


def evaluate_predictions(
    true_indices: Any,
    predicted_indices: Any,
    *,
    row_indices: Any | None = None,
    probabilities: Any | None = None,
    mean_loss: float | None = None,
    class_names: Iterable[str] = CLASS_NAMES,
    split_name: str = "validation",
    model_class: str = "unknown",
    device: str = "unknown",
) -> EvaluationResult:
    """Compute the fixed WM-811K classification metrics from prediction arrays."""

    names = validate_class_names(class_names)
    split = _validate_split_name(split_name)
    true_array = np.asarray(true_indices)
    predicted_array = np.asarray(predicted_indices)

    if true_array.ndim != 1 or predicted_array.ndim != 1:
        raise ValueError("true_indices and predicted_indices must be one-dimensional.")
    if len(true_array) == 0:
        raise ValueError("At least one prediction is required.")
    if len(true_array) != len(predicted_array):
        raise ValueError("true_indices and predicted_indices must have equal lengths.")
    if not np.issubdtype(true_array.dtype, np.integer) or not np.issubdtype(
        predicted_array.dtype, np.integer
    ):
        raise ValueError("true_indices and predicted_indices must contain integers.")

    true_array = true_array.astype(np.int64, copy=False)
    predicted_array = predicted_array.astype(np.int64, copy=False)
    if np.any((true_array < 0) | (true_array >= NUM_CLASSES)):
        raise ValueError(f"true_indices must contain values in [0, {NUM_CLASSES - 1}].")
    if np.any((predicted_array < 0) | (predicted_array >= NUM_CLASSES)):
        raise ValueError(
            f"predicted_indices must contain values in [0, {NUM_CLASSES - 1}]."
        )

    sample_count = len(true_array)
    if row_indices is None:
        row_index_array = np.arange(sample_count, dtype=np.int64)
    else:
        row_index_array = _as_one_dimensional_indices(
            row_indices, name="row_indices", batch_size=sample_count
        )
    if len(np.unique(row_index_array)) != sample_count:
        raise ValueError("row_indices must be unique.")

    probability_array: np.ndarray | None = None
    if probabilities is not None:
        probability_array = np.asarray(probabilities, dtype=np.float64)
        if probability_array.shape != (sample_count, NUM_CLASSES):
            raise ValueError(
                f"probabilities must have shape ({sample_count}, {NUM_CLASSES}); "
                f"received {probability_array.shape!r}."
            )
        if not np.isfinite(probability_array).all() or np.any(probability_array < 0):
            raise ValueError("probabilities must contain finite, non-negative values.")
        if not np.allclose(probability_array.sum(axis=1), 1.0, rtol=1e-5, atol=1e-7):
            raise ValueError("Each probability row must sum to one.")

    if mean_loss is not None and (not np.isfinite(mean_loss) or mean_loss < 0):
        raise ValueError("mean_loss must be a finite, non-negative value.")

    label_indices = np.arange(NUM_CLASSES)
    precision, recall, class_f1, support = precision_recall_fscore_support(
        true_array,
        predicted_array,
        labels=label_indices,
        zero_division=0,
    )
    raw_confusion_matrix = confusion_matrix(
        true_array, predicted_array, labels=label_indices
    ).astype(np.int64, copy=False)
    row_totals = raw_confusion_matrix.sum(axis=1, keepdims=True)
    normalized_confusion_matrix = np.divide(
        raw_confusion_matrix,
        row_totals,
        out=np.zeros_like(raw_confusion_matrix, dtype=np.float64),
        where=row_totals != 0,
    )

    metrics: dict[str, float | int | None] = {
        "num_samples": sample_count,
        "mean_loss": float(mean_loss) if mean_loss is not None else None,
        "accuracy": float(accuracy_score(true_array, predicted_array)),
        "balanced_accuracy": float(
            balanced_accuracy_score(true_array, predicted_array)
        ),
        "macro_f1": float(
            f1_score(
                true_array,
                predicted_array,
                labels=label_indices,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                true_array,
                predicted_array,
                labels=label_indices,
                average="weighted",
                zero_division=0,
            )
        ),
    }

    per_class_metrics = pd.DataFrame(
        {
            "class_index": label_indices,
            "class_name": names,
            "precision": precision,
            "recall": recall,
            "f1": class_f1,
            "support": support.astype(np.int64),
        }
    )

    predictions = pd.DataFrame(
        {
            "row_index": row_index_array,
            "true_index": true_array,
            "true_label": [names[index] for index in true_array],
            "predicted_index": predicted_array,
            "predicted_label": [names[index] for index in predicted_array],
        }
    )
    if probability_array is not None:
        predictions["confidence"] = probability_array.max(axis=1)
        for class_index, class_name in enumerate(names):
            predictions[f"probability_{class_name}"] = probability_array[:, class_index]

    return EvaluationResult(
        metrics=metrics,
        per_class_metrics=per_class_metrics,
        predictions=predictions,
        confusion_matrix=raw_confusion_matrix,
        normalized_confusion_matrix=normalized_confusion_matrix,
        class_names=names,
        split_name=split,
        model_class=model_class,
        device=device,
    )


def evaluate_model(
    model: nn.Module,
    dataloader: Iterable[Any],
    device: str | torch.device,
    criterion: Any | None = None,
    *,
    class_names: Iterable[str] = CLASS_NAMES,
    split_name: str = "validation",
) -> EvaluationResult:
    """Collect PyTorch model outputs and compute the common project metrics."""

    names = validate_class_names(class_names)
    collected = collect_predictions(
        model=model,
        dataloader=dataloader,
        device=device,
        criterion=criterion,
        class_names=names,
    )
    return evaluate_predictions(
        true_indices=collected.true_indices,
        predicted_indices=collected.predicted_indices,
        row_indices=collected.row_indices,
        probabilities=collected.probabilities,
        mean_loss=collected.mean_loss,
        class_names=names,
        split_name=split_name,
        model_class=collected.model_class,
        device=collected.device,
    )

