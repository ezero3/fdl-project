"""A residual CNN built for this dataset, and the second model of our own.

Deeper than ``BaselineCNN`` and residual, so it can actually be trained at
depth, while staying small enough to run many configurations on one Colab GPU.

Fitted to this data rather than borrowed wholesale:

* the stem uses a **3x3 stride-1 convolution and no max-pool**, unlike a
  torchvision ResNet's 7x7 stride-2 stem. At 64x64 that stem would throw away
  three quarters of the spatial resolution before the first residual block,
  and the defects that are hardest here -- a one-die-wide `Scratch`, a small
  `Loc` -- are exactly what disappears first;
* attention is optional and sits after the last stage, where the map still has
  spatial extent, matching ``BaselineCNN`` and ``PretrainedClassifier``;
* the encoder is exposed as ``encoder`` and the classifier as ``head``, so the
  same ``^encoder\\.`` parameter-group regexes work on it as on the pretrained
  backbones.
"""

from __future__ import annotations

from torch import Tensor, nn

from fdl_project.constants import NUM_CLASSES
from fdl_project.data.preprocessing import WAFER_STATE_COUNT
from fdl_project.models.attention import build_attention


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions plus a skip connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        if dilation < 1:
            raise ValueError("dilation must be at least 1.")
        # padding = dilation for a 3x3 kernel keeps the output shape identical
        # at any rate, including alongside a stride, so dilation changes the
        # receptive field and nothing else. The 1x1 skip is left undilated --
        # a 1x1 kernel has no spatial extent for a rate to act on.
        self.convolution1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.normalization1 = nn.BatchNorm2d(out_channels)
        self.convolution2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.normalization2 = nn.BatchNorm2d(out_channels)
        # A projection is needed only when the skip cannot line up with the
        # residual branch in shape.
        self.skip: nn.Module = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        identity = self.skip(inputs)
        residual = self.activation(self.normalization1(self.convolution1(inputs)))
        residual = self.normalization2(self.convolution2(residual))
        # Out-of-place: `+=` on the residual would break autograd for the
        # inplace ReLU that follows.
        return self.activation(residual + identity)


def _stage(
    in_channels: int, out_channels: int, *, stride: int, blocks: int, dilation: int = 1
) -> nn.Sequential:
    layers = [ResidualBlock(in_channels, out_channels, stride=stride, dilation=dilation)]
    layers += [
        ResidualBlock(out_channels, out_channels, dilation=dilation)
        for _ in range(blocks - 1)
    ]
    return nn.Sequential(*layers)


class WaferResNet(nn.Module):
    """Residual CNN over the one-hot wafer encoding.

    ``widths`` and ``blocks_per_stage`` are exposed so the depth/width can be
    tuned from a config without editing the class.
    """

    def __init__(
        self,
        *,
        widths: tuple[int, ...] = (32, 64, 128, 256),
        blocks_per_stage: int = 2,
        dilation: int | tuple[int, ...] = 1,
        dropout: float = 0.4,
        head_dropout: float = 0.3,
        hidden_features: int = 128,
        attention: str | None = None,
    ) -> None:
        super().__init__()
        if len(widths) < 1:
            raise ValueError("widths must name at least one stage.")
        if blocks_per_stage < 1:
            raise ValueError("blocks_per_stage must be at least 1.")
        for name, value in {"dropout": dropout, "head_dropout": head_dropout}.items():
            if not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1).")
        if hidden_features < 1:
            raise ValueError("hidden_features must be positive.")

        # One rate per stage. Dilating the late stages is the useful case: by
        # then the map has been downsampled and context is what is missing,
        # while the early stages still hold the resolution a thin Scratch needs.
        rates_per_stage = (
            (dilation,) * len(widths) if isinstance(dilation, int) else tuple(dilation)
        )
        if len(rates_per_stage) != len(widths):
            raise ValueError(
                "dilation must be an int or one rate per stage; got "
                f"{len(rates_per_stage)} rates for {len(widths)} stages."
            )
        if any(rate < 1 for rate in rates_per_stage):
            raise ValueError("dilation rates must be positive.")

        # 3x3 stride-1, no max-pool: at 64x64 a torchvision-style 7x7 stride-2
        # stem plus pooling would drop to 16x16 before the first block, which
        # is where thin Scratch patterns are lost.
        stages: list[nn.Module] = [
            nn.Sequential(
                nn.Conv2d(
                    WAFER_STATE_COUNT, widths[0], kernel_size=3, padding=1, bias=False
                ),
                nn.BatchNorm2d(widths[0]),
                nn.ReLU(inplace=True),
            )
        ]
        in_channels = widths[0]
        for position, width in enumerate(widths):
            stages.append(
                _stage(
                    in_channels,
                    width,
                    stride=1 if position == 0 else 2,
                    blocks=blocks_per_stage,
                    dilation=rates_per_stage[position],
                )
            )
            in_channels = width

        self.dilation_rates = rates_per_stage
        self.encoder = nn.Sequential(*stages)
        self.attention = build_attention(attention, in_channels) or nn.Identity()
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_channels, hidden_features),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(hidden_features, NUM_CLASSES),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != WAFER_STATE_COUNT:
            raise ValueError(
                "WaferResNet inputs must have shape "
                f"(batch_size, {WAFER_STATE_COUNT}, height, width)."
            )
        if not inputs.is_floating_point():
            raise TypeError("WaferResNet inputs must be floating-point tensors.")
        return self.head(self.pool(self.attention(self.encoder(inputs))))
