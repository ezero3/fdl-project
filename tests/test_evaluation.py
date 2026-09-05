import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from fdl_project.constants import CLASS_NAMES, NUM_CLASSES
from fdl_project.evaluation import (
    evaluate_model,
    evaluate_predictions,
    save_evaluation_results,
)


class RecordingIdentityModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.saw_eval_mode = False
        self.saw_inference_mode = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.saw_eval_mode = not self.training
        self.saw_inference_mode = not torch.is_grad_enabled()
        return inputs


class MappingDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        self.logits = logits
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "images": self.logits[index],
            "labels": self.targets[index],
            "row_indices": torch.tensor(50_000 + index),
        }


def _perfect_probability_matrix(labels: np.ndarray) -> np.ndarray:
    probabilities = np.zeros((len(labels), NUM_CLASSES), dtype=np.float64)
    probabilities[np.arange(len(labels)), labels] = 1.0
    return probabilities


def test_perfect_predictions_produce_perfect_metrics() -> None:
    true_indices = np.tile(np.arange(NUM_CLASSES), 2)
    result = evaluate_predictions(
        true_indices,
        true_indices.copy(),
        probabilities=_perfect_probability_matrix(true_indices),
    )

    assert result.metrics["accuracy"] == pytest.approx(1.0)
    assert result.metrics["balanced_accuracy"] == pytest.approx(1.0)
    assert result.metrics["macro_f1"] == pytest.approx(1.0)
    assert result.metrics["weighted_f1"] == pytest.approx(1.0)
    assert np.array_equal(
        result.confusion_matrix, np.eye(NUM_CLASSES, dtype=np.int64) * 2
    )
    assert np.array_equal(result.normalized_confusion_matrix, np.eye(NUM_CLASSES))
    assert list(result.per_class_metrics["class_name"]) == list(CLASS_NAMES)


def test_always_none_model_exposes_misleading_accuracy() -> None:
    true_indices = np.concatenate(
        [np.full(90, 8, dtype=np.int64), np.arange(8, dtype=np.int64)]
    )
    predicted_indices = np.full(len(true_indices), 8, dtype=np.int64)
    result = evaluate_predictions(true_indices, predicted_indices)

    assert result.metrics["accuracy"] > 0.90
    assert result.metrics["macro_f1"] < 0.12
    assert result.per_class_metrics.loc[0:7, "recall"].eq(0).all()
    assert result.per_class_metrics.loc[8, "recall"] == pytest.approx(1.0)


def test_missing_predicted_classes_keep_fixed_nine_class_shape() -> None:
    true_indices = np.arange(NUM_CLASSES, dtype=np.int64)
    predicted_indices = np.zeros(NUM_CLASSES, dtype=np.int64)
    result = evaluate_predictions(true_indices, predicted_indices)

    assert result.confusion_matrix.shape == (NUM_CLASSES, NUM_CLASSES)
    assert result.normalized_confusion_matrix.shape == (NUM_CLASSES, NUM_CLASSES)
    assert len(result.per_class_metrics) == NUM_CLASSES
    assert (
        np.isfinite(result.per_class_metrics[["precision", "recall", "f1"]]).all().all()
    )


def test_evaluate_model_uses_eval_inference_mode_and_weights_final_batch_loss() -> None:
    targets = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 0], dtype=torch.long)
    logits = torch.full((len(targets), NUM_CLASSES), -2.0)
    logits[torch.arange(len(targets)), targets] = 3.0
    logits[-1, 0] = -1.0
    logits[-1, 1] = 2.0
    row_indices = torch.arange(10_000, 10_000 + len(targets))
    dataloader = DataLoader(
        TensorDataset(logits, targets, row_indices), batch_size=4, shuffle=False
    )
    model = RecordingIdentityModel()

    result = evaluate_model(
        model,
        dataloader,
        device="cpu",
        criterion=nn.CrossEntropyLoss(),
        split_name="validation",
    )

    expected_loss = float(F.cross_entropy(logits, targets).item())
    assert model.saw_eval_mode
    assert model.saw_inference_mode
    assert not model.training
    assert result.metrics["num_samples"] == 10
    assert result.metrics["mean_loss"] == pytest.approx(expected_loss)
    assert result.predictions["row_index"].tolist() == row_indices.tolist()
    assert result.predictions.columns.tolist() == [
        "row_index",
        "true_index",
        "true_label",
        "predicted_index",
        "predicted_label",
        "confidence",
        *(f"probability_{name}" for name in CLASS_NAMES),
    ]


def test_evaluate_model_accepts_mapping_batches() -> None:
    targets = torch.arange(NUM_CLASSES, dtype=torch.long)
    logits = torch.full((NUM_CLASSES, NUM_CLASSES), -1.0)
    logits[torch.arange(NUM_CLASSES), targets] = 2.0
    dataloader = DataLoader(
        MappingDataset(logits, targets), batch_size=5, shuffle=False
    )

    result = evaluate_model(
        RecordingIdentityModel(), dataloader, device="cpu", split_name="test"
    )

    assert result.metrics["accuracy"] == pytest.approx(1.0)
    assert result.split_name == "test"
    assert result.predictions["row_index"].tolist() == list(range(50_000, 50_009))


@pytest.mark.parametrize(
    ("true_indices", "predicted_indices", "message"),
    [
        ([0, 1], [0], "equal lengths"),
        ([0, 9], [0, 1], "true_indices"),
        ([0, 1], [0, -1], "predicted_indices"),
        (["Center"], [0], "contain integers"),
    ],
)
def test_invalid_prediction_inputs_are_rejected(
    true_indices: object, predicted_indices: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_predictions(true_indices, predicted_indices)


def test_duplicate_row_indices_are_rejected() -> None:
    with pytest.raises(ValueError, match="row_indices must be unique"):
        evaluate_predictions([0, 1], [0, 1], row_indices=[42, 42])


def test_invalid_probability_matrix_is_rejected() -> None:
    invalid_probabilities = np.full((2, NUM_CLASSES), 0.5)
    with pytest.raises(ValueError, match="sum to one"):
        evaluate_predictions([0, 1], [0, 1], probabilities=invalid_probabilities)


def test_batch_without_source_indices_is_rejected() -> None:
    logits = torch.zeros((2, NUM_CLASSES))
    targets = torch.tensor([0, 1])
    dataloader = DataLoader(TensorDataset(logits, targets), batch_size=2)

    with pytest.raises(TypeError, match="three-item"):
        evaluate_model(RecordingIdentityModel(), dataloader, device="cpu")


def test_model_with_wrong_logit_shape_is_rejected() -> None:
    logits = torch.zeros((2, NUM_CLASSES - 1))
    targets = torch.tensor([0, 1])
    row_indices = torch.tensor([10, 11])
    dataloader = DataLoader(TensorDataset(logits, targets, row_indices), batch_size=2)

    with pytest.raises(ValueError, match="Model logits must have shape"):
        evaluate_model(RecordingIdentityModel(), dataloader, device="cpu")


def test_incompatible_class_order_and_split_are_rejected() -> None:
    with pytest.raises(ValueError, match="canonical WM-811K encoding"):
        evaluate_predictions([0], [0], class_names=reversed(CLASS_NAMES))
    with pytest.raises(ValueError, match="split_name"):
        evaluate_predictions([0], [0], split_name="train")


def test_save_evaluation_results_writes_the_complete_artifact_set(
    tmp_path: Path,
) -> None:
    true_indices = np.arange(NUM_CLASSES, dtype=np.int64)
    result = evaluate_predictions(
        true_indices,
        true_indices,
        row_indices=np.arange(1_000, 1_000 + NUM_CLASSES),
        probabilities=_perfect_probability_matrix(true_indices),
        mean_loss=0.01,
        model_class="DummyClassifier",
        device="cpu",
    )

    artifacts = save_evaluation_results(
        result,
        tmp_path,
        "dummy-validation",
        metadata={"seed": 86, "git_commit": "abc123"},
    )

    assert set(artifacts) == {
        "metrics",
        "per_class_metrics",
        "predictions",
        "confusion_matrix",
        "normalized_confusion_matrix",
    }
    assert all(
        path.is_file() and path.stat().st_size > 0 for path in artifacts.values()
    )

    payload = json.loads(artifacts["metrics"].read_text(encoding="utf-8"))
    assert payload["class_names"] == list(CLASS_NAMES)
    assert payload["metrics"]["macro_f1"] == pytest.approx(1.0)
    assert payload["metadata"] == {"git_commit": "abc123", "seed": 86}

    per_class = pd.read_csv(artifacts["per_class_metrics"])
    predictions = pd.read_csv(artifacts["predictions"])
    assert per_class["class_name"].tolist() == list(CLASS_NAMES)
    assert predictions["row_index"].tolist() == list(range(1_000, 1_009))

    with pytest.raises(FileExistsError, match="already exist"):
        save_evaluation_results(result, tmp_path, "dummy-validation")


def test_unsafe_run_name_is_rejected(tmp_path: Path) -> None:
    class_indices = np.arange(NUM_CLASSES, dtype=np.int64)
    result = evaluate_predictions(class_indices, class_indices)
    with pytest.raises(ValueError, match="run_name"):
        save_evaluation_results(result, tmp_path, "../outside")
