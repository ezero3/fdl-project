"""Tests for the per-epoch callbacks."""

from __future__ import annotations

import numpy as np

from fdl_project import CLASS_NAMES, NUM_CLASSES, evaluate_predictions
from fdl_project.training.callbacks import per_class_metrics


def _validation_result():
    true_indices = np.repeat(np.arange(NUM_CLASSES), 4)
    predicted_indices = true_indices.copy()
    predicted_indices[0] = (predicted_indices[0] + 1) % NUM_CLASSES
    return evaluate_predictions(true_indices, predicted_indices)


def test_every_class_gets_its_own_series() -> None:
    """Macro-F1 alone hides which classes a change actually moved."""

    flattened = per_class_metrics(_validation_result())

    assert set(flattened) == {f"validation_f1_{name}" for name in CLASS_NAMES}
    assert all(0.0 <= value <= 1.0 for value in flattened.values())


def test_the_values_match_the_evaluation_table() -> None:
    result = _validation_result()
    flattened = per_class_metrics(result)

    for row in result.per_class_metrics.itertuples():
        assert flattened[f"validation_f1_{row.class_name}"] == float(row.f1)


def test_a_missing_validation_result_logs_nothing() -> None:
    """The context has no validation entry before the first epoch ends."""

    assert per_class_metrics(None) == {}
