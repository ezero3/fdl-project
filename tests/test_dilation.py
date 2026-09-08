"""Dilation must change the receptive field and nothing else.

If it also changed the parameter count or the spatial dimensions, a dilated arm
would differ from its control in more than one way and the comparison would be
worthless -- which is the entire point of the v31 series.
"""

import pytest
import torch

from fdl_project.config.registry import build_model
from fdl_project.models.baseline_cnn import count_trainable_parameters

CONVNEXT = {"widths": (32, 64, 128), "blocks_per_stage": (2, 2, 2), "stem_stride": 2}
RESNET = {"widths": (32, 64, 128, 256), "blocks_per_stage": 2}


@pytest.mark.parametrize(
    ("name", "base", "rates"),
    [
        ("convnext_style", CONVNEXT, (1, 2, 4)),
        ("resnet_style", RESNET, (1, 1, 2, 4)),
    ],
)
def test_dilation_changes_neither_parameters_nor_shape(
    name: str, base: dict, rates: tuple[int, ...]
) -> None:
    plain = build_model(name, **base)
    dilated = build_model(name, **base, dilation=rates)
    inputs = torch.rand(2, 3, 64, 64)

    assert count_trainable_parameters(plain) == count_trainable_parameters(dilated)
    plain.eval(), dilated.eval()
    with torch.no_grad():
        assert plain(inputs).shape == dilated(inputs).shape == (2, 9)
    assert dilated.dilation_rates == rates


@pytest.mark.parametrize(("name", "base"), [("convnext_style", CONVNEXT), ("resnet_style", RESNET)])
def test_scalar_dilation_applies_to_every_stage(name: str, base: dict) -> None:
    model = build_model(name, **base, dilation=2)
    assert model.dilation_rates == (2,) * len(base["widths"])


@pytest.mark.parametrize(("name", "base"), [("convnext_style", CONVNEXT), ("resnet_style", RESNET)])
def test_default_is_undilated(name: str, base: dict) -> None:
    assert build_model(name, **base).dilation_rates == (1,) * len(base["widths"])


@pytest.mark.parametrize(("name", "base"), [("convnext_style", CONVNEXT), ("resnet_style", RESNET)])
def test_wrong_rate_count_and_bad_rates_raise(name: str, base: dict) -> None:
    with pytest.raises(ValueError, match="one rate per stage"):
        build_model(name, **base, dilation=(1, 2))
    with pytest.raises(ValueError, match="positive"):
        build_model(name, **base, dilation=(1,) * (len(base["widths"]) - 1) + (0,))


def test_dilation_survives_a_non_square_input() -> None:
    """128x128 arms exist, and letterboxing can hand through odd sizes."""

    model = build_model("convnext_style", **CONVNEXT, dilation=(1, 2, 4))
    model.eval()
    with torch.no_grad():
        assert model(torch.rand(2, 3, 128, 128)).shape == (2, 9)
