r"""Frozen pretrained backbone + a small MLP head, the transfer-learning arm of the project.

The pretrained half contributes fixed representations learned from photographs;
every trained parameter in the model is ours. Registered as
``frozen_backbone_mlp``. `notebooks/experiments/29_pretrained_backbones.ipynb`
carries an identical copy inline so it can run without this package version.

Two details that are easy to get wrong and that both matter here:

1. `requires_grad = False` alone does NOT freeze a backbone that contains
   BatchNorm. `fit_model` calls `model.train()` every epoch, which puts BN back
   into training mode, and BN then keeps updating `running_mean`/`running_var`
   from wafer maps. The encoder's *outputs* drift even though its weights do
   not. Overriding `train()` to force the encoder back to `eval()` is what
   actually makes "frozen" mean frozen.

2. The submodules must be named `encoder` and `head`. Every config in this
   project addresses parameter groups with regexes like `^encoder\.`, and the
   checkpoint metadata assumes that layout.
"""

from __future__ import annotations

from torch import Tensor, nn

from fdl_project.constants import NUM_CLASSES
from fdl_project.data.preprocessing import WAFER_STATE_COUNT

BACKBONE_WIDTH = {          # features the backbone emits before its own classifier
    "resnet18": 512,
    "resnet34": 512,
    "mobilenet_v3_small": 576,
    "mobilenet_v3_large": 960,
    "efficientnet_b0": 1280,
}


class FrozenBackboneMLP(nn.Module):
    """A torchvision backbone as a fixed feature extractor, plus our own MLP.

    This is the "extract pretrained features and train original layers on top"
    strategy: the pretrained half contributes representations it learned from
    photographs, and every trained parameter in the model is ours.
    """

    def __init__(
        self,
        *,
        architecture: str = "resnet18",
        hidden_features: int = 256,
        dropout: float = 0.3,
        freeze_encoder: bool = True,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if architecture not in BACKBONE_WIDTH:
            raise ValueError(
                f"Unsupported architecture {architecture!r}. "
                f"Available: {', '.join(sorted(BACKBONE_WIDTH))}."
            )
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")

        import torchvision.models as tv

        weights = None
        if pretrained:
            enum_name = {
                "resnet18": "ResNet18_Weights",
                "resnet34": "ResNet34_Weights",
                "mobilenet_v3_small": "MobileNet_V3_Small_Weights",
                "mobilenet_v3_large": "MobileNet_V3_Large_Weights",
                "efficientnet_b0": "EfficientNet_B0_Weights",
            }[architecture]
            weights = getattr(tv, enum_name).DEFAULT
        backbone = getattr(tv, architecture)(weights=weights)

        width = BACKBONE_WIDTH[architecture]
        if architecture.startswith("resnet"):
            backbone.fc = nn.Identity()
        else:
            # mobilenet/efficientnet end in a Sequential classifier; dropping it
            # leaves features + avgpool + flatten, which emits `width` values.
            backbone.classifier = nn.Identity()

        self.architecture = architecture
        self.frozen = freeze_encoder
        self.encoder = backbone
        # The "original neural layers" the project brief asks for: one hidden
        # layer, not a bare linear probe, so the head can recombine ImageNet
        # features rather than only reweight them.
        self.head = nn.Sequential(
            nn.Linear(width, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, NUM_CLASSES),
        )
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
            self.encoder.eval()

    def train(self, mode: bool = True) -> "FrozenBackboneMLP":
        """Keep a frozen encoder in eval mode however the loop calls train()."""

        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != WAFER_STATE_COUNT:
            raise ValueError(
                "FrozenBackboneMLP inputs must have shape "
                f"(batch, {WAFER_STATE_COUNT}, height, width); "
                f"got {tuple(inputs.shape)}."
            )
        if not inputs.is_floating_point():
            raise TypeError("FrozenBackboneMLP inputs must be floating-point.")
        return self.head(self.encoder(inputs))
