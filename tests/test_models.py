"""Tests for the from-scratch architectures.

Every ``*_style`` model is our own scaled-down design in the spirit of a known
family. They all have to be drop-in replacements for each other, or the
architecture comparison is not controlled.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from fdl_project.config.registry import available_models, build_model
from fdl_project.constants import NUM_CLASSES
from fdl_project.models import CBAM, WaferDilatedCNN, count_trainable_parameters

OWN_MODELS = ("baseline_cnn", "resnet_style", "inception_style", "dilated_style", "densenet_style", "convnext_style")
CONVOLUTIONAL = OWN_MODELS  # everything except the transformer


def test_our_models_are_named_apart_from_the_pretrained_ones() -> None:
    """`*_style` is ours from scratch; a plain name is torchvision weights."""

    names = available_models()
    assert {"resnet_style", "inception_style", "dilated_style", "densenet_style", "vit_style", "convnext_style"} <= set(names)
    assert {"resnet18", "mobilenet_v3_small", "vit_b_16"} <= set(names)
    assert "resnet_style" != "resnet18"


@pytest.mark.parametrize("name", OWN_MODELS)
def test_every_model_shares_one_interface(name: str) -> None:
    model = build_model(name)
    inputs = torch.rand(2, 3, 64, 64)

    assert model(inputs).shape == (2, NUM_CLASSES)
    assert hasattr(model, "encoder") or hasattr(model, "features")
    assert hasattr(model, "head") or hasattr(model, "classifier")
    with pytest.raises(ValueError, match="shape"):
        model(torch.zeros(2, 1, 64, 64))
    with pytest.raises(TypeError):
        model(torch.zeros(2, 3, 64, 64, dtype=torch.long))


@pytest.mark.parametrize("name", ("resnet_style", "inception_style", "dilated_style", "densenet_style", "convnext_style"))
def test_convolutional_models_are_resolution_agnostic(name: str) -> None:
    """Global pooling, so the same model runs at the pretrained geometry."""

    model = build_model(name)
    assert model(torch.rand(2, 3, 224, 224)).shape == (2, NUM_CLASSES)


@pytest.mark.parametrize("name", ("resnet_style", "inception_style", "dilated_style", "densenet_style", "convnext_style"))
def test_convolutional_models_accept_attention(name: str) -> None:
    attended = build_model(name, attention="cbam")

    assert isinstance(attended.attention, CBAM)
    assert attended(torch.rand(2, 3, 64, 64)).shape == (2, NUM_CLASSES)
    assert count_trainable_parameters(attended) > count_trainable_parameters(
        build_model(name)
    )


@pytest.mark.parametrize("name", OWN_MODELS)
def test_every_model_produces_gradients(name: str) -> None:
    model = build_model(name)
    loss = model(torch.rand(2, 3, 64, 64)).sum()
    loss.backward()

    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# -- dilated ---------------------------------------------------------------


def test_dilation_widens_the_receptive_field_without_downsampling() -> None:
    """The whole argument for this architecture on this data."""

    model = WaferDilatedCNN()
    features = model.encoder(torch.rand(1, 3, 64, 64))

    assert features.shape[-2:] == (64, 64)  # nothing lost
    assert model.receptive_field >= 60  # yet it sees essentially the whole map


def test_dilation_rates_cycle_rather_than_climb() -> None:
    """Consecutive layers at one rate sample disjoint lattices and leave
    checkerboard holes; cycling the rates is the standard fix."""

    rates = WaferDilatedCNN().dilation_rates

    assert rates[0] == 1
    assert len(set(rates)) < len(rates)  # rates repeat


def test_downsampling_can_be_traded_back_in() -> None:
    small = WaferDilatedCNN(downsample_every=2)
    assert small.encoder(torch.rand(1, 3, 64, 64)).shape[-2:] < (64, 64)


# -- inception -------------------------------------------------------------


def test_inception_branches_all_contribute_channels() -> None:
    from fdl_project.models.inception import InceptionBlock

    block = InceptionBlock(16, 8)
    outputs = block(torch.rand(2, 16, 12, 12))

    assert block.out_channels == 8 * 4
    assert outputs.shape == (2, 32, 12, 12)  # resolution preserved in-block


# -- densenet --------------------------------------------------------------


def test_dense_layers_concatenate_rather_than_replace() -> None:
    from fdl_project.models.densenet import DenseLayer

    layer = DenseLayer(16, growth_rate=8)
    outputs = layer(torch.rand(2, 16, 10, 10))

    assert outputs.shape == (2, 24, 10, 10)


# -- vit -------------------------------------------------------------------


def test_the_transformer_runs_and_reports_its_geometry() -> None:
    model = build_model("vit_style")

    assert model(torch.rand(2, 3, 64, 64)).shape == (2, NUM_CLASSES)
    assert model.image_size == 64


def test_the_transformer_rejects_a_geometry_it_was_not_built_for() -> None:
    """The positional table is sized at construction, so a silent mismatch
    would be a wrong model rather than an error."""

    model = build_model("vit_style", image_size=64)
    with pytest.raises(ValueError, match="target_size"):
        model(torch.rand(2, 3, 224, 224))


def test_the_transformer_refuses_convolutional_attention() -> None:
    with pytest.raises(ValueError, match="already an attention model"):
        build_model("vit_style", attention="cbam")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"image_size": 64, "patch_size": 7},
        {"embedding_dimension": 100, "heads": 3},
        {"depth": 0},
    ],
)
def test_invalid_transformer_geometry_is_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        build_model("vit_style", **kwargs)


# -- pretrained ------------------------------------------------------------


def test_pretrained_transformers_are_available_but_take_no_cbam() -> None:
    from fdl_project.models.pretrained import TOKEN_ARCHITECTURES

    assert {"vit_b_16", "vit_b_32", "swin_t"} == set(TOKEN_ARCHITECTURES)
    model = build_model("vit_b_16", pretrained=False)
    assert model(torch.rand(2, 3, 224, 224)).shape == (2, NUM_CLASSES)
    with pytest.raises(ValueError, match="tokens, not a spatial feature map"):
        build_model("vit_b_16", pretrained=False, attention="cbam")


def test_pretrained_models_expose_encoder_and_head_for_param_groups() -> None:
    """`^encoder\\.` has to address the backbone on every pretrained model."""

    for name in ("resnet18", "mobilenet_v3_small", "vit_b_16"):
        model = build_model(name, pretrained=False)
        prefixes = {parameter.split(".")[0] for parameter, _ in model.named_parameters()}
        assert prefixes <= {"encoder", "head", "attention"}
        assert isinstance(model.head, nn.Module)


# -- convnext --------------------------------------------------------------


def test_convnext_block_uses_a_depthwise_kernel() -> None:
    """A 7x7 depthwise kernel costs about what a 3x3 dense one does, which is
    what makes the large receptive field affordable."""

    from fdl_project.models.convnext import ConvNeXtBlock

    block = ConvNeXtBlock(32)

    assert block.depthwise.groups == 32
    assert block.depthwise.kernel_size == (7, 7)
    assert block.expand.out_channels == 32 * 4  # inverted bottleneck


def test_convnext_refuses_the_reference_patchify_stem() -> None:
    """A 4x4 stride-4 stem drops 94% of a 64x64 map before the first block,
    taking a one-die-wide Scratch with it."""

    with pytest.raises(ValueError, match="stem_stride"):
        build_model("convnext_style", stem_stride=4)


def test_stochastic_depth_is_a_training_only_identity() -> None:
    from fdl_project.models.convnext import StochasticDepth

    layer = StochasticDepth(0.5)
    inputs = torch.ones(64, 4, 2, 2)

    layer.eval()
    assert torch.equal(layer(inputs), inputs)


def test_layer_scale_starts_each_block_near_the_identity() -> None:
    """A deep stack begins close to the identity and learns how much to add."""

    from fdl_project.models.convnext import ConvNeXtBlock

    block = ConvNeXtBlock(16).eval()
    inputs = torch.rand(2, 16, 8, 8)

    assert torch.allclose(block(inputs), inputs, atol=1e-3)


# -- baseline v2 -----------------------------------------------------------


def test_v2_global_pooling_removes_the_layer_that_overfits() -> None:
    """84% of the original's parameters sit in one 1024->128 linear. Global
    average pooling deletes it; pooled_size=4 recovers the original head, so
    the two changes can be attributed separately."""

    original = build_model("baseline_cnn")
    pooled = build_model("baseline_v2", pooled_size=4)
    global_average = build_model("baseline_v2")

    assert count_trainable_parameters(pooled) == count_trainable_parameters(original)
    assert count_trainable_parameters(global_average) < count_trainable_parameters(original) / 4


def test_v2_keeps_the_stem_identical_to_v1() -> None:
    """A comparison between them has to be about the head, nothing else."""

    original = build_model("baseline_cnn")
    v2 = build_model("baseline_v2", block_dropout=0.0)

    convolutions = lambda model: [
        (m.in_channels, m.out_channels, m.kernel_size, m.stride)
        for m in model.features.modules()
        if isinstance(m, nn.Conv2d)
    ]
    assert convolutions(v2) == convolutions(original)


def test_v2_uses_channel_dropout_not_cell_dropout() -> None:
    """Ordinary dropout on a feature map is largely undone by spatial
    correlation; Dropout2d drops whole channels."""

    v2 = build_model("baseline_v2", block_dropout=0.1)

    assert any(isinstance(m, nn.Dropout2d) for m in v2.features.modules())


@pytest.mark.parametrize("kwargs", [{"dropout": 1.0}, {"block_dropout": -0.1}, {"pooled_size": 0}])
def test_v2_rejects_invalid_settings(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        build_model("baseline_v2", **kwargs)
