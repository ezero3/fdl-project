from types import MappingProxyType

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, TensorDataset

from fdl_project.constants import NUM_CLASSES
from fdl_project.imbalance import (
    DEFAULT_IMBALANCE_CONFIG,
    PRESET_IMBALANCE_CONFIGS,
    FocalLoss,
    ImbalanceConfig,
    build_training_loss,
    build_weighted_sampler,
    compute_class_counts,
    compute_class_weights,
    create_imbalance_training_dataloader,
    imbalance_metadata,
)
from fdl_project.imbalance_experiment import (
    ImbalanceExperimentConfig,
    categorical_one_hot_collate,
    create_experiment_dataloader,
)
from fdl_project.models import BaselineCNN, count_trainable_parameters
from fdl_project.training import TrainingConfig, fit_model, set_reproducible_seed

TRAIN_COUNTS = torch.tensor(
    [3006, 389, 3632, 6776, 2516, 104, 606, 835, 103199],
    dtype=torch.long,
)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "Invalid name"},
        {"name": "invalid_loss", "loss": "dice"},
        {
            "name": "invalid_sampler",
            "sampling": "weighted",
            "sampling_weighting": "none",
        },
        {"name": "invalid_beta", "effective_number_beta": 1.0},
        {"name": "invalid_gamma", "focal_gamma": -1.0},
        {
            "name": "double_intervention",
            "loss_weighting": "inverse_sqrt",
            "sampling": "weighted",
            "sampling_weighting": "inverse_sqrt",
        },
    ],
)
def test_invalid_imbalance_configurations_are_rejected(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        ImbalanceConfig(**kwargs)  # type: ignore[arg-type]


def test_presets_are_immutable_and_cover_isolated_interventions() -> None:
    assert isinstance(PRESET_IMBALANCE_CONFIGS, MappingProxyType)
    assert set(PRESET_IMBALANCE_CONFIGS) == {
        "unweighted_ce",
        "inverse_sqrt_ce",
        "effective_number_ce",
        "inverse_sqrt_sampler",
        "focal_loss",
    }
    with pytest.raises(TypeError):
        PRESET_IMBALANCE_CONFIGS["new"] = ImbalanceConfig(name="new")  # type: ignore[index]
    assert DEFAULT_IMBALANCE_CONFIG is PRESET_IMBALANCE_CONFIGS["inverse_sqrt_sampler"]


def test_class_counts_require_all_canonical_classes() -> None:
    targets = torch.repeat_interleave(torch.arange(NUM_CLASSES), torch.arange(1, 10))
    assert torch.equal(compute_class_counts(targets), torch.arange(1, 10))

    with pytest.raises(ValueError, match="missing canonical classes"):
        compute_class_counts([0, 1, 2])
    with pytest.raises(ValueError, match="integer"):
        compute_class_counts([0.0, 1.0])


def test_class_weights_are_positive_normalized_and_favor_rare_classes() -> None:
    inverse_sqrt = compute_class_weights(TRAIN_COUNTS, "inverse_sqrt")
    effective = compute_class_weights(TRAIN_COUNTS, "effective_number")

    assert inverse_sqrt.mean().item() == pytest.approx(1.0)
    assert effective.mean().item() == pytest.approx(1.0)
    assert inverse_sqrt[5] > inverse_sqrt[8]
    assert effective[5] > effective[8]
    assert torch.equal(
        compute_class_weights(TRAIN_COUNTS, "none"), torch.ones(NUM_CLASSES)
    )
    assert torch.equal(
        compute_class_weights(
            TRAIN_COUNTS, "effective_number", effective_number_beta=0
        ),
        torch.ones(NUM_CLASSES),
    )


def test_focal_loss_gamma_zero_matches_cross_entropy() -> None:
    logits = torch.randn(12, NUM_CLASSES, generator=torch.Generator().manual_seed(86))
    targets = torch.arange(12) % NUM_CLASSES
    focal = FocalLoss(gamma=0)(logits, targets)
    expected = F.cross_entropy(logits, targets)
    assert focal.item() == pytest.approx(expected.item())


def test_focal_loss_is_finite_differentiable_and_validated() -> None:
    logits = torch.randn(5, NUM_CLASSES, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 3, 8])
    loss = FocalLoss(gamma=2)(logits, targets)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    with pytest.raises(ValueError, match="logits"):
        FocalLoss()(torch.zeros(2, 8), torch.tensor([0, 1]))


def test_training_loss_builds_the_requested_criterion() -> None:
    weighted = build_training_loss(
        PRESET_IMBALANCE_CONFIGS["inverse_sqrt_ce"], TRAIN_COUNTS
    )
    focal = build_training_loss(PRESET_IMBALANCE_CONFIGS["focal_loss"], TRAIN_COUNTS)

    assert isinstance(weighted, nn.CrossEntropyLoss)
    assert weighted.weight is not None and weighted.weight[5] > weighted.weight[8]
    assert isinstance(focal, FocalLoss)


def test_imbalance_metadata_is_checkpoint_safe() -> None:
    metadata = imbalance_metadata(DEFAULT_IMBALANCE_CONFIG, TRAIN_COUNTS)

    assert metadata["imbalance_config"]["name"] == "inverse_sqrt_sampler"
    assert metadata["class_counts"] == TRAIN_COUNTS.tolist()
    assert len(metadata["class_weights"]) == NUM_CLASSES


def test_weighted_sampler_is_deterministic_and_increases_rare_share() -> None:
    targets = torch.repeat_interleave(torch.arange(NUM_CLASSES), TRAIN_COUNTS)
    config = PRESET_IMBALANCE_CONFIGS["inverse_sqrt_sampler"]
    first = list(build_weighted_sampler(config, targets, TRAIN_COUNTS, seed=86))
    second = list(build_weighted_sampler(config, targets, TRAIN_COUNTS, seed=86))
    sampled_targets = targets[first]

    assert first == second
    original_rare_share = float(TRAIN_COUNTS[5] / TRAIN_COUNTS.sum())
    sampled_rare_share = float((sampled_targets == 5).float().mean())
    assert sampled_rare_share > original_rare_share


def _categorical_dataset(samples_per_class: int = 2) -> TensorDataset:
    sample_count = NUM_CLASSES * samples_per_class
    categorical = torch.zeros((sample_count, 9, 9), dtype=torch.uint8)
    targets = torch.arange(sample_count, dtype=torch.long) % NUM_CLASSES
    for position, target in enumerate(targets):
        categorical[position, int(target), int(target)] = 2
        categorical[position, int(target), (int(target) + 1) % 9] = 1
    row_indices = torch.arange(10_000, 10_000 + sample_count)
    return TensorDataset(categorical, targets, row_indices)


class _TrainingDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    split_name = "train"

    def __init__(self) -> None:
        self.target_indices = np.repeat(np.arange(NUM_CLASSES), 2)

    def __len__(self) -> int:
        return len(self.target_indices)

    def __getitem__(
        self, position: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.zeros((3, 8, 8)),
            torch.tensor(self.target_indices[position]),
            torch.tensor(position),
        )


def test_selected_training_loader_is_train_only_and_deterministic() -> None:
    dataset = _TrainingDataset()
    first = create_imbalance_training_dataloader(dataset, batch_size=5, seed=86)
    second = create_imbalance_training_dataloader(dataset, batch_size=5, seed=86)

    first_indices = torch.cat([batch[2] for batch in first])
    second_indices = torch.cat([batch[2] for batch in second])
    assert torch.equal(first_indices, second_indices)
    assert len(first_indices) == len(dataset)

    dataset.split_name = "validation"  # type: ignore[misc]
    with pytest.raises(ValueError, match="only on the train split"):
        create_imbalance_training_dataloader(dataset, batch_size=5)


def test_vectorized_collate_preserves_task04_contract() -> None:
    dataset = _categorical_dataset()
    images, targets, row_indices = categorical_one_hot_collate([dataset[0], dataset[1]])

    assert images.shape == (2, 3, 9, 9)
    assert images.dtype == torch.float32
    assert torch.all(images.sum(dim=1) == 1)
    assert targets.tolist() == [0, 1]
    assert row_indices.tolist() == [10_000, 10_001]


def test_experiment_loaders_keep_validation_order_and_train_determinism() -> None:
    dataset = _categorical_dataset(samples_per_class=3)
    counts = compute_class_counts(dataset.tensors[1])
    validation_loader = create_experiment_dataloader(
        dataset,
        batch_size=5,
        seed=86,
        imbalance_config=None,
    )
    first_train = create_experiment_dataloader(
        dataset,
        batch_size=5,
        seed=86,
        imbalance_config=PRESET_IMBALANCE_CONFIGS["unweighted_ce"],
        class_counts=counts,
    )
    second_train = create_experiment_dataloader(
        dataset,
        batch_size=5,
        seed=86,
        imbalance_config=PRESET_IMBALANCE_CONFIGS["unweighted_ce"],
        class_counts=counts,
    )

    validation_order = torch.cat([batch[2] for batch in validation_loader])
    first_order = torch.cat([batch[2] for batch in first_train])
    second_order = torch.cat([batch[2] for batch in second_train])
    assert torch.equal(validation_order, dataset.tensors[2])
    assert torch.equal(first_order, second_order)
    assert not torch.equal(first_order, dataset.tensors[2])


def test_baseline_cnn_has_fixed_nine_class_output() -> None:
    model = BaselineCNN(dropout=0)
    logits = model(torch.zeros((4, 3, 64, 64), dtype=torch.float32))

    assert logits.shape == (4, NUM_CLASSES)
    assert count_trainable_parameters(model) > 0
    with pytest.raises(ValueError, match="shape"):
        model(torch.zeros((4, 1, 64, 64)))


def test_tiny_fit_returns_best_state_and_complete_history() -> None:
    set_reproducible_seed(86)
    dataset = _categorical_dataset(samples_per_class=2)
    train_loader = create_experiment_dataloader(
        dataset,
        batch_size=9,
        seed=86,
        imbalance_config=PRESET_IMBALANCE_CONFIGS["unweighted_ce"],
        class_counts=compute_class_counts(dataset.tensors[1]),
    )
    validation_loader = create_experiment_dataloader(
        dataset,
        batch_size=9,
        seed=86,
        imbalance_config=None,
    )
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 9 * 9, NUM_CLASSES))
    result = fit_model(
        model,
        train_loader,
        validation_loader,
        nn.CrossEntropyLoss(),
        TrainingConfig(
            batch_size=9,
            max_epochs=2,
            minimum_epochs=1,
            early_stopping_patience=1,
            learning_rate=0.05,
        ),
    )

    assert 1 <= result.best_epoch <= 2
    assert len(result.history) in {1, 2}
    assert set(result.best_state_dict) == set(model.state_dict())
    assert result.best_validation.metrics["num_samples"] == len(dataset)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"confirmation_seeds": ()},
        {"screening_seed": 86, "confirmation_seeds": (86,)},
        {"top_nonbaseline_strategies": 0},
        {"torch_threads": 0},
        {"device": "cuda"},
    ],
)
def test_invalid_experiment_configurations_are_rejected(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        ImbalanceExperimentConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"max_epochs": 2, "minimum_epochs": 3},
        {"learning_rate": 0},
        {"weight_decay": -1},
    ],
)
def test_invalid_training_configurations_are_rejected(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)  # type: ignore[arg-type]
