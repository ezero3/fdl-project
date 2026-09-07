"""Post-hoc gains that need no retraining: TTA and per-class thresholds.

Both are tuned or applied on **validation** and carried unchanged to the single
test evaluation. Neither touches labels, and neither changes the model.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from fdl_project.constants import CLASS_NAMES, NUM_CLASSES
from fdl_project.data.augmentation import NUM_DIHEDRAL_TRANSFORMS, apply_dihedral
from fdl_project.evaluation.metrics import CollectedPredictions, _extract_logits, _unpack_batch


def collect_predictions_with_tta(
    *,
    model: nn.Module,
    dataloader: Iterable[Any],
    device: str | torch.device,
    num_transforms: int = NUM_DIHEDRAL_TRANSFORMS,
) -> CollectedPredictions:
    """Average class probabilities over the eight square symmetries.

    A wafer map carries no canonical orientation, so a model's prediction
    should not depend on one. Averaging over the group the model was trained on
    removes that variance for the cost of an extra forward pass per symmetry --
    seconds, and no labels are involved.
    """

    if not 1 <= num_transforms <= NUM_DIHEDRAL_TRANSFORMS:
        raise ValueError(
            f"num_transforms must lie in [1, {NUM_DIHEDRAL_TRANSFORMS}]."
        )

    device_object = torch.device(device)
    model.to(device_object)
    model.eval()

    row_index_batches: list[np.ndarray] = []
    true_batches: list[np.ndarray] = []
    probability_batches: list[np.ndarray] = []

    with torch.inference_mode():
        for batch in dataloader:
            inputs, targets, row_indices = _unpack_batch(batch)
            inputs = inputs.to(device_object)

            accumulated: Tensor | None = None
            for index in range(num_transforms):
                logits = _extract_logits(model(apply_dihedral(inputs, index)))
                probabilities = torch.softmax(logits.float(), dim=1)
                accumulated = (
                    probabilities if accumulated is None else accumulated + probabilities
                )
            averaged = (accumulated / num_transforms).cpu().numpy()

            probability_batches.append(averaged)
            true_batches.append(np.asarray(targets, dtype=np.int64).reshape(-1))
            row_index_batches.append(np.asarray(row_indices, dtype=np.int64).reshape(-1))

    probabilities = np.concatenate(probability_batches)
    return CollectedPredictions(
        row_indices=np.concatenate(row_index_batches),
        true_indices=np.concatenate(true_batches),
        predicted_indices=probabilities.argmax(axis=1),
        probabilities=probabilities,
        mean_loss=None,
        model_class=type(model).__name__,
        device=str(device_object),
    )


def apply_class_weights(probabilities: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Predict with per-class multipliers instead of a plain arg-max."""

    scaled = np.asarray(probabilities, dtype=np.float64) * np.asarray(
        weights, dtype=np.float64
    )
    return scaled.argmax(axis=1)


def _macro_f1(true_indices: np.ndarray, predicted_indices: np.ndarray) -> float:
    pairs = true_indices * NUM_CLASSES + predicted_indices
    matrix = np.bincount(pairs, minlength=NUM_CLASSES**2).reshape(
        NUM_CLASSES, NUM_CLASSES
    )
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
    class_f1 = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros(NUM_CLASSES),
        where=denominator != 0,
    )
    return float(class_f1.mean())


def tune_class_weights(
    probabilities: np.ndarray,
    true_indices: np.ndarray,
    *,
    candidates: Iterable[float] | None = None,
    rounds: int = 3,
) -> tuple[np.ndarray, float]:
    """Fit one multiplier per class to maximise macro-F1 on this split.

    Coordinate ascent over a small grid: each class in turn, keeping the best
    value, repeated ``rounds`` times. Macro-F1 weights all nine classes
    equally while arg-max implicitly favours the majority class, so the
    optimum is generally not "all ones".

    Must be fitted on **validation** only. Fitting on test would be selecting
    on the test split, which is exactly what the evaluation protocol forbids.
    """

    if rounds < 1:
        raise ValueError("rounds must be at least 1.")
    grid = np.asarray(
        list(candidates)
        if candidates is not None
        else [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0],
        dtype=np.float64,
    )
    if grid.size == 0 or np.any(grid <= 0):
        raise ValueError("candidates must be positive and non-empty.")

    probabilities = np.asarray(probabilities, dtype=np.float64)
    true_indices = np.asarray(true_indices, dtype=np.int64)
    weights = np.ones(NUM_CLASSES, dtype=np.float64)
    best_score = _macro_f1(true_indices, apply_class_weights(probabilities, weights))

    for _ in range(rounds):
        improved = False
        for class_index in range(NUM_CLASSES):
            incumbent = weights[class_index]
            for candidate in grid:
                weights[class_index] = candidate
                score = _macro_f1(
                    true_indices, apply_class_weights(probabilities, weights)
                )
                if score > best_score:
                    best_score, incumbent, improved = score, candidate, True
            weights[class_index] = incumbent
        if not improved:
            break
    return weights, best_score


def save_class_weights(weights: np.ndarray, path: str | Path) -> Path:
    """Persist tuned weights so the test evaluation reuses them verbatim."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "class_names": list(CLASS_NAMES),
        "weights": [float(value) for value in weights],
    }
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination


def load_class_weights(path: str | Path) -> np.ndarray:
    """Load tuned weights, refusing any file from a different class encoding."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if tuple(payload.get("class_names", ())) != CLASS_NAMES:
        raise ValueError(
            "Class weights were tuned under a different class encoding."
        )
    weights = np.asarray(payload["weights"], dtype=np.float64)
    if weights.shape != (NUM_CLASSES,):
        raise ValueError(f"Expected {NUM_CLASSES} class weights.")
    return weights
