"""Tests for dihedral augmentation and the attention block."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from fdl_project.data.augmentation import (
    NUM_DIHEDRAL_TRANSFORMS,
    ComposedAugmentation,
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
        "dihedral8_rotation",
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


def test_per_class_probabilities_target_the_rare_classes() -> None:
    """Augment the rare classes and leave the majority class alone."""

    from fdl_project.constants import CLASS_TO_INDEX

    augmentation = build_augmentation(
        "dihedral8",
        probability=0.0,
        class_probabilities={"Scratch": 1.0, "Near-full": 1.0},
    )
    wafer = _wafer()
    seed_everything(86)

    majority = sum(
        not torch.equal(augmentation(wafer, CLASS_TO_INDEX["none"]), wafer)
        for _ in range(60)
    )
    rare = sum(
        not torch.equal(augmentation(wafer, CLASS_TO_INDEX["Scratch"]), wafer)
        for _ in range(60)
    )

    assert majority == 0
    # dihedral8 includes the identity, so a few draws are unchanged by chance
    assert rare > 40


def test_an_unknown_class_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown class"):
        build_augmentation("dihedral8", class_probabilities={"Scratches": 1.0})


def test_rotation_refuses_per_class_probabilities() -> None:
    """Uneven resampling makes 'looks resampled' predict the rare classes."""

    with pytest.raises(ValueError, match="leaks the label"):
        build_augmentation("rotation", class_probabilities={"Scratch": 1.0})


def test_uniform_augmentation_ignores_the_class_index() -> None:
    augmentation = build_augmentation("dihedral8")
    wafer = _wafer()
    seed_everything(86)
    with_class = [augmentation(wafer, 3) for _ in range(10)]
    seed_everything(86)
    without_class = [augmentation(wafer) for _ in range(10)]

    assert all(
        torch.equal(a, b) for a, b in zip(with_class, without_class, strict=True)
    )


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


def test_wafer_resnet_matches_the_project_interface() -> None:
    """The second from-scratch model must be a drop-in for the pipeline."""

    from fdl_project.models import WaferResNet, count_trainable_parameters

    model = WaferResNet()
    assert model(torch.rand(2, 3, 64, 64)).shape == (2, 9)
    # global pooling, so it accepts the pretrained geometry too
    assert model(torch.rand(2, 3, 224, 224)).shape == (2, 9)
    # encoder/head naming is what the param_groups regexes address
    assert hasattr(model, "encoder") and hasattr(model, "head")
    assert count_trainable_parameters(model) > 1_000_000

    with pytest.raises(ValueError, match="shape"):
        model(torch.zeros(2, 1, 64, 64))


def test_wafer_resnet_keeps_resolution_through_the_stem() -> None:
    """A 7x7 stride-2 stem plus pooling would drop 64x64 to 16x16 before the
    first block, which is where thin Scratch patterns are lost."""

    from fdl_project.models import WaferResNet

    model = WaferResNet()
    stem_output = model.encoder[0](torch.rand(1, 3, 64, 64))

    assert stem_output.shape[-2:] == (64, 64)


def test_wafer_resnet_depth_and_attention_are_configurable() -> None:
    from fdl_project.models import WaferResNet, count_trainable_parameters

    small = WaferResNet(widths=(16, 32), blocks_per_stage=1)
    attended = WaferResNet(attention="cbam")

    assert small(torch.rand(2, 3, 64, 64)).shape == (2, 9)
    assert isinstance(attended.attention, CBAM)
    assert count_trainable_parameters(small) < count_trainable_parameters(attended)


@pytest.mark.parametrize(
    "kwargs",
    [{"widths": ()}, {"blocks_per_stage": 0}, {"dropout": 1.0}, {"hidden_features": 0}],
)
def test_invalid_wafer_resnet_settings_are_rejected(kwargs: dict) -> None:
    from fdl_project.models import WaferResNet

    with pytest.raises(ValueError):
        WaferResNet(**kwargs)


def test_residual_blocks_actually_add_the_skip() -> None:
    """Without the skip a deep stack is what we already have in BaselineCNN."""

    from fdl_project.models.wafer_resnet import ResidualBlock

    block = ResidualBlock(8, 8)
    # zero the residual branch: output must then be the input, via the skip
    with torch.no_grad():
        block.normalization2.weight.zero_()
        block.normalization2.bias.zero_()
    inputs = torch.rand(2, 8, 6, 6)

    assert torch.allclose(block(inputs), torch.relu(inputs), atol=1e-5)


def test_residual_block_projects_when_shapes_differ() -> None:
    from fdl_project.models.wafer_resnet import ResidualBlock

    same = ResidualBlock(16, 16)
    projected = ResidualBlock(16, 32, stride=2)

    assert isinstance(same.skip, nn.Identity)
    assert not isinstance(projected.skip, nn.Identity)
    assert projected(torch.rand(2, 16, 8, 8)).shape == (2, 32, 4, 4)


# -- composition -----------------------------------------------------------


def test_dihedral_and_rotation_compose_into_two_stages() -> None:
    """The group alone reaches 8 views; composing a free angle onto it makes
    the reachable set continuous, which is the point of the arm."""

    composed = build_augmentation("dihedral8_rotation")

    assert len(composed.stages) == 2
    assert isinstance(composed.stages[0], DihedralAugmentation)
    assert isinstance(composed.stages[1], RotationAugmentation)


def test_composition_preserves_the_one_hot_encoding() -> None:
    wafer = _wafer()
    composed = build_augmentation("dihedral8_rotation", degrees=180.0)

    augmented = composed(wafer)

    assert augmented.shape == wafer.shape
    assert torch.all(augmented.sum(dim=0) == 1)


def test_composition_refuses_per_class_probabilities() -> None:
    """Inherited from the rotation stage: resampling unevenly leaks the label."""

    with pytest.raises(ValueError, match="leaks the label"):
        build_augmentation("dihedral8_rotation", class_probabilities={"Scratch": 1.0})


def test_composition_is_reachable_from_the_registry() -> None:
    assert "dihedral8_rotation" in available_augmentations()


def test_an_empty_composition_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one stage"):
        ComposedAugmentation(stages=())


def test_the_rotation_stage_applies_less_often_than_the_group() -> None:
    """The group transform is exact and free, so it always applies; the
    rotation resamples, so clean examples are kept in the distribution."""

    composed = build_augmentation("dihedral8_rotation", rotation_probability=0.5)

    assert composed.stages[0].probability == 1.0
    assert composed.stages[1].probability == 0.5


def test_rotation_fill_follows_the_encoding() -> None:
    """A one-hot fill in a grayscale tensor would decode to a nonexistent state."""

    from fdl_project.data.preprocessing import PreprocessingConfig, background_fill

    one_hot = background_fill(PreprocessingConfig())
    grayscale = background_fill(PreprocessingConfig(encoding="grayscale_rgb"))
    normalized = background_fill(
        PreprocessingConfig(encoding="grayscale_rgb", normalization="imagenet")
    )

    assert one_hot == [1.0, 0.0, 0.0]
    assert grayscale == [0.0, 0.0, 0.0]
    assert all(value < 0 for value in normalized)  # 0.0 pushed through the norm


def test_rotation_rejects_a_fill_of_the_wrong_width() -> None:
    augmentation = RotationAugmentation(fill=(1.0, 0.0))

    with pytest.raises(ValueError, match="channels"):
        augmentation(_wafer())


def test_rotation_imports_nothing_at_call_time() -> None:
    """This runs inside forked DataLoader workers. An import there trips W&B's
    import hook with ForkedError and kills the run -- so torchvision must be
    imported at module scope, not lazily in __call__."""

    import inspect

    from fdl_project.data import augmentation

    source = inspect.getsource(augmentation.RotationAugmentation.__call__)

    assert "import" not in source
    assert hasattr(augmentation, "InterpolationMode")
