"""Tests for test-time augmentation and per-class decision thresholds."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fdl_project.constants import NUM_CLASSES
from fdl_project.evaluation.postprocessing import (
    _macro_f1,
    apply_class_weights,
    collect_predictions_with_tta,
    load_class_weights,
    save_class_weights,
    tune_class_weights,
)


class OrientationSensitiveModel(nn.Module):
    """Reads one corner, so its answer depends on the wafer's orientation.

    That dependence is exactly what test-time augmentation removes: a wafer map
    has no canonical orientation, so a prediction should not have one either.
    """

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        corner = inputs[:, 0, 0, 0]
        logits = torch.zeros((len(inputs), NUM_CLASSES))
        logits[:, 0] = corner * 4.0
        logits[:, 1] = (1.0 - corner) * 4.0
        return logits


def _loader() -> DataLoader:
    generator = torch.Generator().manual_seed(86)
    inputs = torch.rand((12, 3, 6, 6), generator=generator)
    targets = torch.arange(12) % NUM_CLASSES
    return DataLoader(
        TensorDataset(inputs, targets, torch.arange(12)), batch_size=4
    )


def test_tta_returns_the_full_prediction_record() -> None:
    collected = collect_predictions_with_tta(
        model=OrientationSensitiveModel(), dataloader=_loader(), device="cpu"
    )

    assert collected.probabilities.shape == (12, NUM_CLASSES)
    assert len(collected.row_indices) == len(collected.true_indices) == 12
    np.testing.assert_allclose(collected.probabilities.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(
        collected.predicted_indices, collected.probabilities.argmax(axis=1)
    )


def test_tta_averages_away_orientation_dependence() -> None:
    single = collect_predictions_with_tta(
        model=OrientationSensitiveModel(),
        dataloader=_loader(),
        device="cpu",
        num_transforms=1,
    )
    averaged = collect_predictions_with_tta(
        model=OrientationSensitiveModel(), dataloader=_loader(), device="cpu"
    )

    assert not np.allclose(single.probabilities, averaged.probabilities)
    # averaging pulls predictions toward the middle
    assert averaged.probabilities.max(axis=1).mean() < single.probabilities.max(
        axis=1
    ).mean()


@pytest.mark.parametrize("num_transforms", [0, 9, -1])
def test_tta_rejects_an_impossible_transform_count(num_transforms: int) -> None:
    with pytest.raises(ValueError, match="num_transforms"):
        collect_predictions_with_tta(
            model=OrientationSensitiveModel(),
            dataloader=_loader(),
            device="cpu",
            num_transforms=num_transforms,
        )


def _imbalanced_probabilities(seed: int = 86):
    """Majority class plus eight rare ones, the shape of this dataset."""

    generator = np.random.default_rng(seed)
    true_indices = np.concatenate(
        [np.full(600, NUM_CLASSES - 1), np.repeat(np.arange(NUM_CLASSES - 1), 10)]
    )
    probabilities = generator.dirichlet(np.ones(NUM_CLASSES) * 0.4, len(true_indices))
    probabilities[np.arange(len(true_indices)), true_indices] += 0.5
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities, true_indices


def test_tuning_improves_macro_f1_over_plain_argmax() -> None:
    """Arg-max favours the majority class; macro-F1 weights all nine equally."""

    probabilities, true_indices = _imbalanced_probabilities()
    baseline = _macro_f1(true_indices, probabilities.argmax(axis=1))
    weights, tuned = tune_class_weights(probabilities, true_indices)

    assert weights.shape == (NUM_CLASSES,)
    assert np.all(weights > 0)
    assert tuned >= baseline
    assert tuned == pytest.approx(
        _macro_f1(true_indices, apply_class_weights(probabilities, weights))
    )


def test_all_ones_reproduces_argmax() -> None:
    probabilities, _ = _imbalanced_probabilities()
    np.testing.assert_array_equal(
        apply_class_weights(probabilities, np.ones(NUM_CLASSES)),
        probabilities.argmax(axis=1),
    )


def test_tuning_is_deterministic() -> None:
    probabilities, true_indices = _imbalanced_probabilities()
    first, _ = tune_class_weights(probabilities, true_indices)
    second, _ = tune_class_weights(probabilities, true_indices)

    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"rounds": 0}, "rounds"), ({"candidates": [0.0, 1.0]}, "positive")],
)
def test_invalid_tuning_settings_are_rejected(kwargs, message) -> None:
    probabilities, true_indices = _imbalanced_probabilities()
    with pytest.raises(ValueError, match=message):
        tune_class_weights(probabilities, true_indices, **kwargs)


def test_weights_round_trip_through_disk(tmp_path) -> None:
    weights = np.linspace(0.5, 2.0, NUM_CLASSES)
    path = save_class_weights(weights, tmp_path / "class_weights.json")

    np.testing.assert_allclose(load_class_weights(path), weights)


def test_weights_from_another_class_encoding_are_refused(tmp_path) -> None:
    path = tmp_path / "foreign.json"
    path.write_text('{"class_names": ["a", "b"], "weights": [1.0, 1.0]}')

    with pytest.raises(ValueError, match="class encoding"):
        load_class_weights(path)
