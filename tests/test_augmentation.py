"""Tests for dihedral augmentation and the attention block."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from fdl_project.data.augmentation import (
    NUM_DIHEDRAL_TRANSFORMS,
    TRANSFORM_SUBSETS,
    RotationAugmentation,
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
    """A genuinely one-hot map with an asymmetric mark.

    Exactly one channel is set per cell -- channel 1 (functional die)
    everywhere except the three defect cells, which are channel 2.
    """

    states = torch.ones((8, 8), dtype=torch.long)
    for row, column in ((1, 5), (2, 5), (1, 6)):
        states[row, column] = 2
    tensor = torch.zeros((3, 8, 8))
    tensor.scatter_(0, states.unsqueeze(0), 1.0)
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


@pytest.mark.parametrize(
    ("name", "size"), [("dihedral8", 8), ("rotations", 4), ("flips", 4)]
)
def test_subgroups_draw_only_from_their_own_transforms(name: str, size: int) -> None:
    """Each named subset is a closed subgroup of D4, so ablating between them
    compares structure rather than three different amounts of noise."""

    seed_everything(86)
    augmentation = build_augmentation(name)
    wafer = _wafer()
    views = {augmentation(wafer).numpy().tobytes() for _ in range(300)}
    allowed = {
        apply_dihedral(wafer, index).numpy().tobytes()
        for index in TRANSFORM_SUBSETS[name]
    }

    assert len(augmentation.transforms) == size
    assert views == allowed


def test_flips_are_the_actual_mirrors() -> None:
    wafer = _wafer()
    horizontal, vertical = TRANSFORM_SUBSETS["flips"][2], TRANSFORM_SUBSETS["flips"][3]

    assert torch.equal(apply_dihedral(wafer, horizontal), torch.flip(wafer, dims=(-1,)))
    assert torch.equal(apply_dihedral(wafer, vertical), torch.flip(wafer, dims=(-2,)))


@pytest.mark.parametrize("transforms", [(), (0, 0), (9,), (-1,)])
def test_invalid_transform_sets_are_rejected(transforms: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        DihedralAugmentation(transforms=transforms)


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
    assert set(available_augmentations()) == {
        "dihedral8",
        "rotations",
        "flips",
        "rotation",
    }
    # off unless a config asks for it
    assert build_augmentation(None) is None
    assert isinstance(build_augmentation("dihedral8"), DihedralAugmentation)
    with pytest.raises(KeyError, match="dihedral8"):
        build_augmentation("rotate_freely")


def test_rotation_keeps_the_encoding_exactly_one_hot() -> None:
    """Nearest neighbour copies whole cells, so no fractional state appears.

    The corners a rotation exposes are filled with the background channel --
    filling every channel with zero would produce a cell in no state at all.
    """

    seed_everything(86)
    rotated = RotationAugmentation(degrees=45.0)(_wafer())

    assert torch.equal(rotated.sum(dim=0), torch.ones(8, 8))
    assert set(rotated.unique().tolist()) <= {0.0, 1.0}
    # exposed corners become background, never a die
    assert rotated[0, 0, 0] == 1.0


def test_rotation_preserves_the_defect_count() -> None:
    """Measured on the real split: median change is under 1% and no wafer
    loses its pattern. See docs/ROADMAP.md item 10."""

    seed_everything(86)
    wafer = _wafer()
    augmentation = RotationAugmentation(degrees=180.0)
    before = int(wafer[2].sum())
    counts = [int(augmentation(wafer)[2].sum()) for _ in range(50)]

    assert all(count > 0 for count in counts)
    assert abs(np.median(counts) - before) <= 1


def test_rotation_is_reproducible_and_varies() -> None:
    augmentation = RotationAugmentation()
    wafer = _wafer()

    seed_everything(86)
    first = [augmentation(wafer) for _ in range(8)]
    seed_everything(86)
    second = [augmentation(wafer) for _ in range(8)]

    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
    assert len({view.numpy().tobytes() for view in first}) > 1


@pytest.mark.parametrize("degrees", [0, -5, 181, True])
def test_invalid_rotation_ranges_are_rejected(degrees: object) -> None:
    with pytest.raises(ValueError, match="degrees"):
        RotationAugmentation(degrees=degrees)  # type: ignore[arg-type]


def test_rotation_is_reachable_from_the_registry() -> None:
    augmentation = build_augmentation("rotation", degrees=90.0)

    assert isinstance(augmentation, RotationAugmentation)
    assert augmentation.degrees == 90.0
    assert "rotation" in available_augmentations()


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
