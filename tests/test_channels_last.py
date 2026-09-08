"""channels_last must change speed and nothing else.

It is a memory-layout choice, not an arithmetic one. If it altered outputs, a
run using it would not be comparable to the arms that do not -- which defeats
the point of turning it on for one expensive arm.
"""

import pytest
import torch

from fdl_project.config.registry import build_model
from fdl_project.config.schema import TrainerConfig


def test_off_by_default() -> None:
    assert TrainerConfig().channels_last is False


def test_rejects_a_non_boolean() -> None:
    with pytest.raises(ValueError, match="channels_last must be a boolean"):
        TrainerConfig(channels_last="yes")


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("convnext_style", {"widths": (32, 64), "blocks_per_stage": (1, 1),
                            "stem_stride": 2, "dilation": (1, 2)}),
        ("resnet_style", {"widths": (16, 32), "blocks_per_stage": 1}),
        ("dilated_style", {"channels": 16, "dilation_rates": (1, 2)}),
    ],
)
def test_outputs_are_unchanged(name: str, kwargs: dict) -> None:
    """Same weights, same input, two memory formats, same numbers."""

    torch.manual_seed(0)
    model = build_model(name, **kwargs).eval()
    inputs = torch.rand(2, 3, 64, 64)

    with torch.no_grad():
        expected = model(inputs)

    model_nhwc = model.to(memory_format=torch.channels_last)
    inputs_nhwc = inputs.contiguous(memory_format=torch.channels_last)
    with torch.no_grad():
        actual = model_nhwc(inputs_nhwc)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_gradients_match() -> None:
    """The backward pass matters as much as the forward one: a layout that
    changed gradients would change training, not just throughput."""

    torch.manual_seed(0)
    kwargs = {"widths": (16, 32), "blocks_per_stage": 1}
    inputs = torch.rand(2, 3, 64, 64)
    targets = torch.randint(0, 9, (2,))

    grads = []
    for channels_last in (False, True):
        torch.manual_seed(0)
        model = build_model("resnet_style", **kwargs).train()
        batch = inputs
        if channels_last:
            model = model.to(memory_format=torch.channels_last)
            batch = inputs.contiguous(memory_format=torch.channels_last)
        torch.nn.functional.cross_entropy(model(batch), targets).backward()
        grads.append(
            torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        )

    torch.testing.assert_close(grads[0], grads[1], rtol=1e-4, atol=1e-5)


def test_it_is_recorded_in_the_run_config() -> None:
    """A speed flag that is not logged is a flag nobody can attribute a result
    to later. W&B and the checkpoint metadata both read to_dict()."""

    assert TrainerConfig(channels_last=True).to_dict()["channels_last"] is True
    assert TrainerConfig().to_dict()["channels_last"] is False
