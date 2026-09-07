"""Tests for process-wide seeding and per-worker dataloader seeding."""

from __future__ import annotations

import os
import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from fdl_project.training.seed import (
    CUBLAS_WORKSPACE_CONFIG,
    build_dataloader_generator,
    seed_everything,
    seed_worker,
    validate_seed,
)


class RandomDrawDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Each item draws fresh randomness, the way an augmentation would."""

    def __len__(self) -> int:
        return 16

    def __getitem__(self, index: int):
        draw = torch.tensor(
            [random.random(), float(np.random.rand()), float(torch.rand(1))]
        )
        return draw, torch.tensor(0), torch.tensor(index)


def _batches(num_workers: int, seed: int) -> list[torch.Tensor]:
    seed_everything(seed)
    loader = DataLoader(
        RandomDrawDataset(),
        batch_size=4,
        num_workers=num_workers,
        worker_init_fn=seed_worker if num_workers else None,
        generator=build_dataloader_generator(seed),
    )
    return [batch[0] for batch in loader]


def _draws(num_workers: int, seed: int) -> torch.Tensor:
    return torch.cat(_batches(num_workers, seed))


def test_seeding_makes_every_library_reproducible() -> None:
    seed_everything(86)
    first = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    seed_everything(86)
    second = (random.random(), float(np.random.rand()), float(torch.rand(1)))

    assert first == second


def test_a_different_seed_gives_different_numbers() -> None:
    seed_everything(86)
    first = torch.rand(8)
    seed_everything(87)
    second = torch.rand(8)

    assert not torch.equal(first, second)


def test_the_default_leaves_the_cudnn_autotuner_on() -> None:
    """Seeding is what reproducibility needs; deterministic kernels cost
    throughput without delivering bit-exact results, so they stay off."""

    seed_everything(86)

    assert os.environ["PYTHONHASHSEED"] == "86"
    assert not torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.benchmark is True


def test_determinism_can_be_switched_on_for_a_strict_check() -> None:
    seed_everything(86, deterministic=True)

    assert torch.are_deterministic_algorithms_enabled()
    # Strict mode raises mid-training on CUDA, where several conv-backward
    # kernels have no deterministic implementation, so it must warn instead.
    assert torch.is_deterministic_algorithms_warn_only_enabled()
    assert torch.backends.cudnn.benchmark is False
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == CUBLAS_WORKSPACE_CONFIG

    seed_everything(86)  # restore the default for the rest of the session


@pytest.mark.parametrize("seed", [-1, 2**32, True, 1.5, "86"])
def test_invalid_seeds_are_rejected(seed: object) -> None:
    with pytest.raises(ValueError, match="seed"):
        validate_seed(seed)  # type: ignore[arg-type]


def test_single_process_loading_is_reproducible() -> None:
    assert torch.equal(_draws(0, 86), _draws(0, 86))
    assert not torch.equal(_draws(0, 86), _draws(0, 87))


def test_workers_are_seeded_reproducibly_and_differently() -> None:
    """Unseeded workers repeat one identical view; that is the bug being fixed."""

    first = _batches(2, 86)
    second = _batches(2, 86)
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))

    # Batches alternate between the two workers, so consecutive batches come
    # from different processes. Sharing one seed would make them identical.
    assert not torch.equal(first[0], first[1])
