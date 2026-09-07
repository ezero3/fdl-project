"""Resampling-based confidence intervals for the project metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fdl_project.constants import NUM_CLASSES
from fdl_project.evaluation.metrics import EvaluationResult


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

