"""A frozen backbone that quietly unfreezes would invalidate the whole
transfer-learning comparison, and it fails silently -- the loss still falls,
the metrics still look plausible. These tests are the only thing that catches
it."""

import pytest
import torch

from fdl_project.config.registry import available_models, build_model
from fdl_project.constants import NUM_CLASSES
from fdl_project.models.frozen_backbone import FrozenBackboneMLP


def _model(**kwargs) -> FrozenBackboneMLP:
    return FrozenBackboneMLP(pretrained=False, **kwargs)


def test_registered_under_the_name_configs_use() -> None:
    assert "frozen_backbone_mlp" in available_models()
    assert isinstance(
        build_model("frozen_backbone_mlp", pretrained=False), FrozenBackboneMLP
    )


def test_encoder_stays_in_eval_through_a_train_call() -> None:
    """requires_grad=False does not stop BatchNorm. `fit_model` calls
    `model.train()` every epoch, which would put the backbone's BN layers back
    into training mode and let their running statistics drift towards wafer
    maps -- the encoder's outputs change while its weights do not."""

    model = _model()
    model.train()

    assert not any(module.training for module in model.encoder.modules())
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert model.head.training, "the head must still train"


def test_running_statistics_do_not_move() -> None:
    """The behavioural version of the test above: the same input twice, with a
    training pass in between, must give the same features."""

    model = _model()
    model.train()
    inputs = torch.rand(4, 3, 64, 64)

    with torch.no_grad():
        before = model.encoder(inputs)
    model(inputs).sum().backward()
    with torch.no_grad():
        after = model.encoder(inputs)

    assert torch.equal(before, after)


def test_only_the_head_is_trainable() -> None:
    model = _model(architecture="resnet18")
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}

    assert trainable
    assert all(name.startswith("head.") for name in trainable)


def test_unfrozen_encoder_trains() -> None:
    model = _model(freeze_encoder=False)
    model.train()

    assert any(module.training for module in model.encoder.modules())
    assert all(p.requires_grad for p in model.encoder.parameters())


@pytest.mark.parametrize("architecture", ["resnet18", "mobilenet_v3_small"])
def test_output_shape_is_the_nine_classes(architecture: str) -> None:
    model = _model(architecture=architecture)
    model.eval()

    with torch.no_grad():
        logits = model(torch.rand(2, 3, 128, 128))

    assert logits.shape == (2, NUM_CLASSES)


def test_submodules_are_named_encoder_and_head() -> None:
    """Configs address parameter groups with regexes like '^encoder\\.'."""

    model = _model()
    assert hasattr(model, "encoder") and hasattr(model, "head")


def test_rejects_unknown_architecture_and_bad_dropout() -> None:
    with pytest.raises(ValueError, match="Unsupported architecture"):
        _model(architecture="not_a_backbone")
    with pytest.raises(ValueError, match="dropout"):
        _model(dropout=1.0)


def test_rejects_wrong_channel_count() -> None:
    model = _model()
    model.eval()
    with pytest.raises(ValueError, match="shape"):
        model(torch.rand(2, 1, 128, 128))
