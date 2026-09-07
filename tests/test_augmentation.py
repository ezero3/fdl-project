"""Tests for dihedral augmentation and the attention block."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from fdl_project.data.augmentation import (
    NUM_DIHEDRAL_TRANSFORMS,
    DihedralAugmentation,
    apply_dihedral,
    available_augmentations,
    build_augmentation,
    dihedral_views,
)
from fdl_project.models.attention import CBAM, build_attention
from fdl_project.models.baseline_cnn import BaselineCNN
from fdl_project.training.seed import seed_everything


def _wafer() -> torch.Tensor:
    """A one-hot map with an asymmetric mark, so every symmetry is distinct."""

    tensor = torch.zeros((3, 8, 8))
    tensor[1] = 1.0
    tensor[2, 1, 5] = 1.0
    tensor[2, 2, 5] = 1.0
    tensor[2, 1, 6] = 1.0
    return tensor


def test_there_are_exactly_eight_distinct_symmetries() -> None:
    views = dihedral_views(_wafer())

    assert len(views) == NUM_DIHEDRAL_TRANSFORMS
    flattened = {view.numpy().tobytes() for view in views}
    assert len(flattened) == NUM_DIHEDRAL_TRANSFORMS


def test_the_identity_comes_first() -> None:
    wafer = _wafer()
    assert torch.equal(apply_dihedral(wafer, 0), wafer)


def test_transforms_only_permute_cells() -> None:
    """No interpolation means no new values and no lost defective dies."""

    wafer = _wafer()
    for index in range(NUM_DIHEDRAL_TRANSFORMS):
        view = apply_dihedral(wafer, index)
        assert view.shape == wafer.shape
        assert torch.equal(view.sum(dim=(1, 2)), wafer.sum(dim=(1, 2)))
        assert set(view.unique().tolist()) <= set(wafer.unique().tolist())


@pytest.mark.parametrize("index", [-1, 8, 100])
def test_an_out_of_range_transform_is_rejected(index: int) -> None:
    with pytest.raises(ValueError, match="index"):
        apply_dihedral(_wafer(), index)


def test_augmentation_is_reproducible_for_a_seed() -> None:
    augmentation = DihedralAugmentation()
    wafer = _wafer()

    seed_everything(86)
    first = [augmentation(wafer) for _ in range(12)]
    seed_everything(86)
    second = [augmentation(wafer) for _ in range(12)]

    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))


def test_augmentation_actually_varies_the_view() -> None:
    seed_everything(86)
    augmentation = DihedralAugmentation()
    wafer = _wafer()
    views = {augmentation(wafer).numpy().tobytes() for _ in range(200)}

    assert len(views) == NUM_DIHEDRAL_TRANSFORMS


def test_probability_zero_is_the_identity() -> None:
    seed_everything(86)
    augmentation = DihedralAugmentation(probability=0.0)
    wafer = _wafer()

    assert all(torch.equal(augmentation(wafer), wafer) for _ in range(20))


@pytest.mark.parametrize("probability", [-0.1, 1.5, True])
def test_invalid_probabilities_are_rejected(probability: object) -> None:
    with pytest.raises(ValueError, match="probability"):
        DihedralAugmentation(probability=probability)  # type: ignore[arg-type]


def test_the_registry_reports_and_validates_names() -> None:
    assert "dihedral8" in available_augmentations()
    assert build_augmentation(None) is None
    assert isinstance(build_augmentation("dihedral8"), DihedralAugmentation)
    with pytest.raises(KeyError, match="dihedral8"):
        build_augmentation("rotate_freely")


def test_attention_preserves_shape_and_is_learnable() -> None:
    block = CBAM(16)
    inputs = torch.rand(2, 16, 12, 12)
    outputs = block(inputs)

    assert outputs.shape == inputs.shape
    assert not torch.equal(outputs, inputs)
    outputs.sum().backward()
    assert any(p.grad is not None for p in block.parameters())


def test_attention_rejects_a_pooled_feature_vector() -> None:
    """Spatial attention needs spatial extent; catching this early matters."""

    with pytest.raises(ValueError, match="batch, channels, height, width"):
        CBAM(16)(torch.rand(2, 16))


def test_the_attention_registry_validates_names() -> None:
    assert build_attention(None, 8) is None
    assert build_attention("none", 8) is None
    assert isinstance(build_attention("cbam", 8), CBAM)
    with pytest.raises(KeyError, match="cbam"):
        build_attention("squeeze_excite", 8)


def test_a_backbone_takes_attention_without_changing_its_interface() -> None:
    inputs = torch.rand(2, 3, 64, 64)
    plain = BaselineCNN()
    attended = BaselineCNN(attention="cbam")

    assert plain(inputs).shape == attended(inputs).shape == (2, 9)
    assert isinstance(plain.attention, nn.Identity)
    assert isinstance(attended.attention, CBAM)
    extra = sum(p.numel() for p in attended.parameters()) - sum(
        p.numel() for p in plain.parameters()
    )
    assert 0 < extra < 5_000  # cheap, as the roadmap claims


def test_augmentation_changes_training_views_only(tmp_path) -> None:
    """Validation and test must be provably untouched."""

    from fdl_project.constants import CLASS_NAMES
    import pandas as pd
    from fdl_project.data.datasets import WM811KDataset

    frame = pd.DataFrame(
        {
            "waferMap": [np.array([[0, 1], [2, 1]], dtype=np.uint8)] * 3,
            "failureType": [np.array([[name]]) for name in CLASS_NAMES[:3]],
        },
        index=pd.Index([1, 2, 3]),
    )
    indices = frame.index.to_numpy()

    train = WM811KDataset(
        frame, indices, split_name="train", augmentation=DihedralAugmentation()
    )
    assert train.augmentation is not None

    for split in ("validation", "test"):
        with pytest.raises(ValueError, match="train-only"):
            WM811KDataset(
                frame,
                indices,
                split_name=split,
                augmentation=DihedralAugmentation(),
            )
