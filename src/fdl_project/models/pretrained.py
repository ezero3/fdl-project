"""Torchvision backbones adapted to the nine wafer-map classes.

The module deliberately exposes exactly two top-level submodules, ``encoder``
and ``head``, so that an experiment config can address them with stable
regular expressions:

```yaml
param_groups:
  - {pattern: "^encoder\\.", kwargs: {lr: 1.0e-5}}
```

That is what "fine-tuned, not frozen" means in practice — the pretrained
weights keep training, just far more slowly than the randomly initialised head.

No stem surgery is needed. The project's one-hot encoding already produces
three channels (background, functional die, defective die), which maps onto an
RGB stem directly. ImageNet mean/std normalisation is *not* applied: these are
indicator channels, not photographs. Whether matching the pretraining
distribution more closely helps anyway is an open experiment recorded in
`docs/design-notes.md`.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from fdl_project.constants import NUM_CLASSES
from fdl_project.data.preprocessing import WAFER_STATE_COUNT

#: Torchvision factory name and default pretrained weights enum per architecture.
SUPPORTED_ARCHITECTURES: dict[str, str] = {
    "resnet18": "ResNet18_Weights",
    "resnet34": "ResNet34_Weights",
    "mobilenet_v3_small": "MobileNet_V3_Small_Weights",
    "mobilenet_v3_large": "MobileNet_V3_Large_Weights",
    "efficientnet_b0": "EfficientNet_B0_Weights",
}


def _split_backbone(model: nn.Module, architecture: str) -> tuple[nn.Module, int]:
    """Return the feature extractor and the width of its output."""

    if architecture.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = nn.Identity()
        return model, in_features
    # mobilenet_v3_* and efficientnet_b0 both end in a Sequential classifier
    # whose first Linear carries the feature width.
    classifier = model.classifier
    linear_layers = [layer for layer in classifier if isinstance(layer, nn.Linear)]
    if not linear_layers:
        raise ValueError(f"Cannot locate the classifier head of {architecture!r}.")
    in_features = linear_layers[0].in_features
    model.classifier = nn.Identity()
    return model, in_features


class PretrainedClassifier(nn.Module):
    """A torchvision backbone plus a fresh nine-class head.

    ``freeze_encoder`` is available as the cheap comparison baseline the
    roadmap asks for, not as the default: it trains in minutes and shows how
    much full fine-tuning actually buys.
    """

    def __init__(
        self,
        *,
        architecture: str = "resnet18",
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_encoder: bool = False,
        **backbone_kwargs: Any,
    ) -> None:
        super().__init__()
        if architecture not in SUPPORTED_ARCHITECTURES:
            available = ", ".join(sorted(SUPPORTED_ARCHITECTURES))
            raise ValueError(
                f"Unsupported architecture {architecture!r}. Available: {available}."
            )
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")

        try:
            import torchvision.models as torchvision_models
        except ImportError as error:  # pragma: no cover - environment guard
            raise ImportError(
                "Pretrained backbones require torchvision; run 'uv sync'."
            ) from error

        weights = None
        if pretrained:
            weights_enum = getattr(torchvision_models, SUPPORTED_ARCHITECTURES[architecture])
            weights = weights_enum.DEFAULT
        backbone = getattr(torchvision_models, architecture)(
            weights=weights, **backbone_kwargs
        )
        encoder, in_features = _split_backbone(backbone, architecture)

        self.architecture = architecture
        self.pretrained = pretrained
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, NUM_CLASSES),
        )
        if freeze_encoder:
            self.freeze_encoder()

    def freeze_encoder(self) -> None:
        """Turn the backbone into a fixed feature extractor."""

        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.encoder.eval()

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != WAFER_STATE_COUNT:
            raise ValueError(
                "PretrainedClassifier inputs must have shape "
                f"(batch_size, {WAFER_STATE_COUNT}, height, width)."
            )
        if not inputs.is_floating_point():
            raise TypeError("PretrainedClassifier inputs must be floating-point.")
        return self.head(self.encoder(inputs))
