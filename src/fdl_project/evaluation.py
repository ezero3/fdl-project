"""Reusable PyTorch evaluation and artifact-generation utilities."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch import Tensor, nn

from fdl_project.constants import CLASS_NAMES, NUM_CLASSES, validate_class_names

_RUN_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
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


def _plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: tuple[str, ...],
    output_path: Path,
    *,
    normalized: bool,
) -> None:
    figure = Figure(figsize=(10, 8), constrained_layout=True)
    axes = figure.subplots()
    image = axes.imshow(matrix, interpolation="nearest", cmap="Blues", vmin=0)
    figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04)
    axes.set(
        title="Normalized confusion matrix" if normalized else "Confusion matrix",
        xlabel="Predicted class",
        ylabel="True class",
        xticks=np.arange(NUM_CLASSES),
        yticks=np.arange(NUM_CLASSES),
        xticklabels=class_names,
        yticklabels=class_names,
    )
    axes.tick_params(axis="x", rotation=45)

    threshold = float(matrix.max()) / 2 if matrix.size and matrix.max() > 0 else 0
    for row_index in range(NUM_CLASSES):
        for column_index in range(NUM_CLASSES):
            value = matrix[row_index, column_index]
            text = f"{value:.2f}" if normalized else f"{int(value)}"
            axes.text(
                column_index,
                row_index,
                text,
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > threshold else "black",
            )
    figure.savefig(output_path, dpi=160)
    figure.clear()


@dataclass(frozen=True)
class BootstrapResult:
    """Resampling-based uncertainty for one evaluated split."""

    aggregate: pd.DataFrame
    per_class: pd.DataFrame
    num_resamples: int
    confidence_level: float
    seed: int


def _metrics_from_confusion(matrix: np.ndarray) -> tuple[float, float, float, float]:
    """Return accuracy, balanced accuracy, macro-F1, and weighted-F1."""

    true_positives = np.diag(matrix).astype(np.float64)
    support = matrix.sum(axis=1).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    total = support.sum()

    recall = np.divide(
        true_positives, support, out=np.zeros(NUM_CLASSES), where=support != 0
    )
    precision = np.divide(
        true_positives, predicted, out=np.zeros(NUM_CLASSES), where=predicted != 0
    )
    denominator = precision + recall
    class_f1 = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros(NUM_CLASSES),
        where=denominator != 0,
    )

    accuracy = float(true_positives.sum() / total) if total else 0.0
    # balanced_accuracy_score averages recall over the classes actually present,
    # so an absent class must not contribute a zero here.
    present = support != 0
    balanced_accuracy = float(recall[present].mean()) if present.any() else 0.0
    # macro-F1 keeps the full nine-class shape, matching evaluate_predictions.
    macro_f1 = float(class_f1.mean())
    weighted_f1 = float((class_f1 * support).sum() / total) if total else 0.0
    return accuracy, balanced_accuracy, macro_f1, weighted_f1


def _class_f1_from_confusion(matrix: np.ndarray) -> np.ndarray:
    true_positives = np.diag(matrix).astype(np.float64)
    support = matrix.sum(axis=1).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    recall = np.divide(
        true_positives, support, out=np.zeros(NUM_CLASSES), where=support != 0
    )
    precision = np.divide(
        true_positives, predicted, out=np.zeros(NUM_CLASSES), where=predicted != 0
    )
    denominator = precision + recall
    return np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros(NUM_CLASSES),
        where=denominator != 0,
    )


def bootstrap_evaluation(
    result: EvaluationResult,
    *,
    num_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 86,
) -> BootstrapResult:
    """Estimate percentile confidence intervals by resampling the predictions.

    Macro-F1 has no closed-form sampling distribution, and the rare WM-811K
    classes have very small support in the fixed folds, so a point estimate
    alone cannot show whether two models are actually separable. This draws
    ``num_resamples`` samples of the evaluated rows with replacement and
    reports the percentile interval of each metric.
    """

    if num_resamples < 2:
        raise ValueError("num_resamples must be at least 2.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one.")

    true_array = result.predictions["true_index"].to_numpy(dtype=np.int64)
    predicted_array = result.predictions["predicted_index"].to_numpy(dtype=np.int64)
    sample_count = len(true_array)

    generator = np.random.default_rng(seed)
    aggregate_draws = np.empty((num_resamples, 4), dtype=np.float64)
    class_f1_draws = np.empty((num_resamples, NUM_CLASSES), dtype=np.float64)

    for draw in range(num_resamples):
        positions = generator.integers(0, sample_count, sample_count)
        pairs = true_array[positions] * NUM_CLASSES + predicted_array[positions]
        matrix = np.bincount(pairs, minlength=NUM_CLASSES * NUM_CLASSES).reshape(
            NUM_CLASSES, NUM_CLASSES
        )
        aggregate_draws[draw] = _metrics_from_confusion(matrix)
        class_f1_draws[draw] = _class_f1_from_confusion(matrix)

    tail = (1.0 - confidence_level) / 2.0 * 100.0
    percentiles = (tail, 100.0 - tail)

    aggregate_names = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
    lower, upper = np.percentile(aggregate_draws, percentiles, axis=0)
    aggregate = pd.DataFrame(
        {
            "metric": aggregate_names,
            "point_estimate": [float(result.metrics[name]) for name in aggregate_names],
            "ci_lower": lower,
            "ci_upper": upper,
            "standard_error": aggregate_draws.std(axis=0, ddof=1),
        }
    )

    class_lower, class_upper = np.percentile(class_f1_draws, percentiles, axis=0)
    per_class = pd.DataFrame(
        {
            "class_index": np.arange(NUM_CLASSES),
            "class_name": result.class_names,
            "f1": result.per_class_metrics["f1"].to_numpy(dtype=np.float64),
            "ci_lower": class_lower,
            "ci_upper": class_upper,
            "standard_error": class_f1_draws.std(axis=0, ddof=1),
            "support": result.per_class_metrics["support"].to_numpy(dtype=np.int64),
        }
    )

    return BootstrapResult(
        aggregate=aggregate,
        per_class=per_class,
        num_resamples=num_resamples,
        confidence_level=confidence_level,
        seed=seed,
    )


def save_evaluation_results(
    result: EvaluationResult,
    output_root: str | Path,
    run_name: str,
    *,
    metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
    bootstrap: BootstrapResult | None = None,
) -> dict[str, Path]:
    """Persist the standard comparison artifacts for one experiment.

    Passing ``bootstrap`` additionally writes the resampled confidence
    intervals and records the aggregate ones inside ``metrics.json``.
    """

    if not _RUN_NAME_PATTERN.fullmatch(run_name):
        raise ValueError(
            "run_name must start with an alphanumeric character and contain only "
            "letters, numbers, '.', '_' or '-'."
        )

    experiment_directory = Path(output_root) / run_name
    artifact_paths = {
        "metrics": experiment_directory / "metrics.json",
        "per_class_metrics": experiment_directory / "per_class_metrics.csv",
        "predictions": experiment_directory / "predictions.csv",
        "confusion_matrix": experiment_directory / "confusion_matrix.png",
        "normalized_confusion_matrix": experiment_directory
        / "confusion_matrix_normalized.png",
    }
    if bootstrap is not None:
        artifact_paths["bootstrap_aggregate"] = (
            experiment_directory / "bootstrap_aggregate.csv"
        )
        artifact_paths["bootstrap_per_class"] = (
            experiment_directory / "bootstrap_per_class.csv"
        )
    existing_artifacts = [path for path in artifact_paths.values() if path.exists()]
    if existing_artifacts and not overwrite:
        raise FileExistsError(
            f"Evaluation artifacts already exist for run {run_name!r}; "
            "choose a new run name or pass overwrite=True."
        )
    experiment_directory.mkdir(parents=True, exist_ok=True)

    metadata_payload = dict(metadata or {})
    metrics_payload = {
        "schema_version": 1,
        "run_name": run_name,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "split_name": result.split_name,
        "model_class": result.model_class,
        "device": result.device,
        "class_names": list(result.class_names),
        "metrics": result.metrics,
        "metadata": metadata_payload,
    }
    if bootstrap is not None:
        metrics_payload["bootstrap"] = {
            "num_resamples": bootstrap.num_resamples,
            "confidence_level": bootstrap.confidence_level,
            "seed": bootstrap.seed,
            "intervals": {
                row.metric: {
                    "point_estimate": row.point_estimate,
                    "ci_lower": row.ci_lower,
                    "ci_upper": row.ci_upper,
                }
                for row in bootstrap.aggregate.itertuples()
            },
        }
    with artifact_paths["metrics"].open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    result.per_class_metrics.to_csv(artifact_paths["per_class_metrics"], index=False)
    result.predictions.to_csv(artifact_paths["predictions"], index=False)
    _plot_confusion_matrix(
        result.confusion_matrix,
        result.class_names,
        artifact_paths["confusion_matrix"],
        normalized=False,
    )
    _plot_confusion_matrix(
        result.normalized_confusion_matrix,
        result.class_names,
        artifact_paths["normalized_confusion_matrix"],
        normalized=True,
    )
    if bootstrap is not None:
        bootstrap.aggregate.to_csv(artifact_paths["bootstrap_aggregate"], index=False)
        bootstrap.per_class.to_csv(artifact_paths["bootstrap_per_class"], index=False)
    return artifact_paths
