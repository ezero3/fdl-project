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
    "resnet50": 2048,       # bottleneck blocks, so 4x the feature width
    "mobilenet_v3_small": 576,
    "mobilenet_v3_large": 960,
    "efficientnet_b0": 1280,
    "vit_b_16": 768,
    "vit_b_32": 768,
    "convnext_tiny": 768,
    "convnext_small": 768,
    "swin_t": 768,
    "swin_v2_t": 768,
    "swin_s": 768,
    "maxvit_t": 512,        # its head projects down to 512 before classifying
}

#: Torchvision weight enums, by architecture.
WEIGHT_ENUM = {
    "resnet18": "ResNet18_Weights",
    "resnet34": "ResNet34_Weights",
    "resnet50": "ResNet50_Weights",
    "mobilenet_v3_small": "MobileNet_V3_Small_Weights",
    "mobilenet_v3_large": "MobileNet_V3_Large_Weights",
    "efficientnet_b0": "EfficientNet_B0_Weights",
    "vit_b_16": "ViT_B_16_Weights",
    "vit_b_32": "ViT_B_32_Weights",
    "convnext_tiny": "ConvNeXt_Tiny_Weights",
    "convnext_small": "ConvNeXt_Small_Weights",
    "swin_t": "Swin_T_Weights",
    "swin_v2_t": "Swin_V2_T_Weights",
    "swin_s": "Swin_S_Weights",
    "maxvit_t": "MaxVit_T_Weights",
}

#: These keep their classifier in `heads`, not `fc` or `classifier`, and their
#: positional embeddings are fitted to one input size -- 224x224 for every
#: torchvision checkpoint. A config that feeds them anything else must say so.
TOKEN_ARCHITECTURES = frozenset({"vit_b_16", "vit_b_32"})

#: Fixed 224x224 for a different reason: MaxViT's grid attention partitions the
#: map into a 7x7 lattice, so a 128px input fails on a reshape rather than an
#: assertion. Swin is not here -- its windowed attention adapts to any size
#: divisible by 32, which is why it can be compared at 128 like everything else.
FIXED_224_ARCHITECTURES = frozenset({"maxvit_t"}) | TOKEN_ARCHITECTURES


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
            weights = getattr(tv, WEIGHT_ENUM[architecture]).DEFAULT
        backbone = getattr(tv, architecture)(weights=weights)

        width = BACKBONE_WIDTH[architecture]
        if architecture in TOKEN_ARCHITECTURES:
            # A ViT emits the class token through `heads`; replacing it leaves
            # the pooled 768-wide embedding.
            backbone.heads = nn.Identity()
        elif architecture.startswith("swin"):
            backbone.head = nn.Identity()
        elif architecture == "maxvit_t":
            # MaxViT's classifier is pool -> flatten -> LayerNorm -> Linear ->
            # Tanh -> Linear. Everything but the last Linear is the pretrained
            # projection into its 512-wide embedding, so only that goes.
            backbone.classifier[-1] = nn.Identity()
        elif architecture.startswith("resnet"):
            backbone.fc = nn.Identity()
        elif architecture.startswith("convnext"):
            # ConvNeXt's classifier is LayerNorm2d -> Flatten -> Linear, so the
            # Flatten that produces a 2-D feature vector lives INSIDE it.
            # Replacing the whole Sequential would leave (batch, 768, 1, 1) and
            # the head's Linear would silently consume the wrong axis. Only the
            # final Linear goes.
            backbone.classifier[2] = nn.Identity()
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

    @property
    def required_input_size(self) -> int | None:
        """The one input size this backbone accepts, if it is fixed.

        Torchvision ViT checkpoints carry positional embeddings fitted to
        224x224 and raise on anything else, so a config pairing one with
        target_size 128 fails at the first batch rather than at import.
        """

        return 224 if self.architecture in FIXED_224_ARCHITECTURES else None

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
