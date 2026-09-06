"""Tests for the resampling-based confidence intervals."""

import json

import numpy as np
import pytest

from fdl_project import (
    CLASS_NAMES,
    NUM_CLASSES,
    bootstrap_evaluation,
    evaluate_predictions,
    save_evaluation_results,
)


def _balanced_result(samples_per_class: int = 20, *, seed: int = 86):
    """Build a result whose predictions are mostly, but not perfectly, correct."""

    generator = np.random.default_rng(seed)
    true_indices = np.repeat(np.arange(NUM_CLASSES), samples_per_class)
    predicted_indices = true_indices.copy()
    wrong = generator.random(len(true_indices)) < 0.2
    predicted_indices[wrong] = (predicted_indices[wrong] + 1) % NUM_CLASSES
    return evaluate_predictions(true_indices, predicted_indices)


def test_intervals_bracket_the_point_estimate():
    result = _balanced_result()
    bootstrap = bootstrap_evaluation(result, num_resamples=200, seed=86)

    for row in bootstrap.aggregate.itertuples():
        assert row.ci_lower <= row.point_estimate <= row.ci_upper
        assert 0.0 <= row.ci_lower <= 1.0
        assert 0.0 <= row.ci_upper <= 1.0
        assert row.standard_error >= 0.0


def test_reports_every_aggregate_metric_and_every_class():
    result = _balanced_result()
    bootstrap = bootstrap_evaluation(result, num_resamples=100)

    assert list(bootstrap.aggregate["metric"]) == [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
    ]
    assert list(bootstrap.per_class["class_name"]) == list(CLASS_NAMES)
    assert len(bootstrap.per_class) == NUM_CLASSES
    assert bootstrap.per_class["support"].sum() == len(result.predictions)


def test_is_deterministic_for_one_seed():
    result = _balanced_result()
    first = bootstrap_evaluation(result, num_resamples=150, seed=86)
    second = bootstrap_evaluation(result, num_resamples=150, seed=86)
    other = bootstrap_evaluation(result, num_resamples=150, seed=87)

    np.testing.assert_array_equal(
        first.aggregate["ci_lower"].to_numpy(), second.aggregate["ci_lower"].to_numpy()
    )
    assert not np.array_equal(
        first.aggregate["ci_lower"].to_numpy(), other.aggregate["ci_lower"].to_numpy()
    )


def test_rare_classes_get_wider_intervals():
    """A class with 5 observations must be visibly less certain than one with 500."""

    true_indices = np.concatenate(
        [np.zeros(500, dtype=int), np.ones(5, dtype=int)]
    )
    predicted_indices = true_indices.copy()
    predicted_indices[0] = 1
    predicted_indices[-1] = 0
    result = evaluate_predictions(true_indices, predicted_indices)
    bootstrap = bootstrap_evaluation(result, num_resamples=400, seed=86)

    common = bootstrap.per_class.loc[0]
    rare = bootstrap.per_class.loc[1]
    assert rare.ci_upper - rare.ci_lower > common.ci_upper - common.ci_lower


def test_perfect_predictions_give_a_degenerate_interval():
    true_indices = np.repeat(np.arange(NUM_CLASSES), 10)
    result = evaluate_predictions(true_indices, true_indices.copy())
    bootstrap = bootstrap_evaluation(result, num_resamples=100)

    macro = bootstrap.aggregate.set_index("metric").loc["macro_f1"]
    assert macro.point_estimate == pytest.approx(1.0)
    assert macro.ci_lower == pytest.approx(1.0)
    assert macro.ci_upper == pytest.approx(1.0)


def test_matches_the_pipeline_metrics_on_the_full_sample():
    """A resample of the original rows must reproduce evaluate_predictions."""

    result = _balanced_result()
    bootstrap = bootstrap_evaluation(result, num_resamples=2)
    aggregate = bootstrap.aggregate.set_index("metric")

    for name in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"):
        assert aggregate.loc[name, "point_estimate"] == pytest.approx(
            result.metrics[name]
        )
    np.testing.assert_allclose(
        bootstrap.per_class["f1"].to_numpy(),
        result.per_class_metrics["f1"].to_numpy(),
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_resamples": 1}, "num_resamples"),
        ({"confidence_level": 0.0}, "confidence_level"),
        ({"confidence_level": 1.0}, "confidence_level"),
    ],
)
def test_rejects_invalid_configuration(kwargs, message):
    result = _balanced_result()
    with pytest.raises(ValueError, match=message):
        bootstrap_evaluation(result, **kwargs)


def test_confidence_level_widens_the_interval():
    result = _balanced_result()
    narrow = bootstrap_evaluation(result, num_resamples=400, confidence_level=0.5)
    wide = bootstrap_evaluation(result, num_resamples=400, confidence_level=0.99)

    narrow_row = narrow.aggregate.set_index("metric").loc["macro_f1"]
    wide_row = wide.aggregate.set_index("metric").loc["macro_f1"]
    assert (wide_row.ci_upper - wide_row.ci_lower) > (
        narrow_row.ci_upper - narrow_row.ci_lower
    )


def test_saved_artifacts_include_the_intervals(tmp_path):
    result = _balanced_result()
    bootstrap = bootstrap_evaluation(result, num_resamples=100, seed=86)
    paths = save_evaluation_results(
        result, output_root=tmp_path, run_name="with-bootstrap", bootstrap=bootstrap
    )

    assert paths["bootstrap_aggregate"].exists()
    assert paths["bootstrap_per_class"].exists()

    payload = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    assert payload["bootstrap"]["num_resamples"] == 100
    assert payload["bootstrap"]["seed"] == 86
    assert set(payload["bootstrap"]["intervals"]) == {
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
    }


def test_saving_without_bootstrap_is_unchanged(tmp_path):
    result = _balanced_result()
    paths = save_evaluation_results(
        result, output_root=tmp_path, run_name="no-bootstrap"
    )

    assert "bootstrap_aggregate" not in paths
    payload = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    assert "bootstrap" not in payload
