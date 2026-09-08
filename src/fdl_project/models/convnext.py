"""ConvNeXt-style network: a CNN rebuilt with transformer design choices.

This is the most interesting architecture in the set for the report, because it
isolates a specific question. `vit_style` asks whether attention beats
convolution on this data. ConvNeXt asks something sharper: **how much of a
transformer's advantage is attention at all, and how much is the surrounding
design?** It keeps convolution and adopts everything else -- large depthwise
kernels, an inverted bottleneck, LayerNorm, GELU, and far fewer activations and
normalizations than a ResNet.

Against a `resnet_style` of comparable size, the difference is exactly that
modernization, with the operator held fixed.

Scaled to this project: the reference ConvNeXt-T is 28M parameters for 224x224
ImageNet. Ours is a few hundred thousand for 64x64 wafer maps.

One deliberate deviation. The reference stem is a 4x4 stride-4 patchify, which
on a 64x64 map would discard 94% of the grid before the first block and take a
one-die-wide `Scratch` with it. The stem here is stride 2 -- 64x64 to 32x32,
the same first step `densenet_style` and `inception_style` take.

Stride 1 is allowed but is not the default, and the reason is cost rather than
principle: holding the full 64x64 grid through a 7x7 depthwise and a 4x
inverted bottleneck measured **3.8x** the time of stride 2 for identical
parameters. Parameter count does not predict cost in this family -- the
resolution the blocks run at does.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from fdl_project.constants import NUM_CLASSES
from fdl_project.data.preprocessing import WAFER_STATE_COUNT
from fdl_project.models.attention import build_attention


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel axis of an NCHW tensor.

    `nn.LayerNorm` normalizes trailing dimensions, so it cannot be applied to
    NCHW directly. ConvNeXt normalizes per position across channels, which is
    what this does.
    """

    def __init__(self, channels: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.epsilon = epsilon

    def forward(self, inputs: Tensor) -> Tensor:
        mean = inputs.mean(dim=1, keepdim=True)
        variance = inputs.var(dim=1, keepdim=True, unbiased=False)
        normalized = (inputs - mean) / torch.sqrt(variance + self.epsilon)
        return normalized * self.weight[:, None, None] + self.bias[:, None, None]


class StochasticDepth(nn.Module):
    """Drop the residual branch for whole samples during training.

    ConvNeXt relies on this rather than dropout inside the block. At the depths
    used here the rate is small; it is exposed because it is the regularizer
    the architecture is designed around.
    """

    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError("stochastic depth probability must be in [0, 1).")
        self.probability = probability

    def forward(self, inputs: Tensor) -> Tensor:
        if self.probability == 0.0 or not self.training:
            return inputs
        keep = 1.0 - self.probability
        # One Bernoulli draw per sample, broadcast over C/H/W.
        mask = inputs.new_empty(inputs.shape[0], 1, 1, 1).bernoulli_(keep)
        return inputs * mask / keep


class ConvNeXtBlock(nn.Module):
    """Depthwise 7x7, LayerNorm, inverted bottleneck with a single GELU.

    The ordering is the whole point, and it is what a ResNet block does not do:
    one normalization instead of three, one activation instead of three, and
    the expansion *after* the spatial mixing rather than before it.

    A 7x7 depthwise kernel costs about what a 3x3 dense one does at these
    widths, which is what makes the large receptive field affordable -- the
    same trade the dilated model makes by another route.
    """

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int = 7,
        expansion: int = 4,
        dilation: int = 1,
        stochastic_depth: float = 0.0,
        layer_scale: float = 1e-6,
    ) -> None:
        super().__init__()
        if dilation < 1:
            raise ValueError("dilation must be at least 1.")
        # padding = dilation * (kernel // 2) keeps the output shape identical
        # for any rate, so dilation changes the receptive field and nothing
        # else -- not the spatial dimensions, not the parameter count.
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=dilation * (kernel_size // 2),
            dilation=dilation,
            groups=channels,  # depthwise: this is what keeps 7x7 cheap
        )
        self.dilation = dilation
        self.normalization = LayerNorm2d(channels)
        self.expand = nn.Conv2d(channels, channels * expansion, kernel_size=1)
        self.activation = nn.GELU()
        self.project = nn.Conv2d(channels * expansion, channels, kernel_size=1)
        # LayerScale starts each residual branch near zero, so a deep stack
        # begins as close to the identity and learns how much to add.
        self.scale = (
            nn.Parameter(layer_scale * torch.ones(channels))
            if layer_scale > 0
            else None
        )
        self.stochastic_depth = StochasticDepth(stochastic_depth)

    def forward(self, inputs: Tensor) -> Tensor:
        residual = self.project(self.activation(self.expand(
            self.normalization(self.depthwise(inputs))
        )))
        if self.scale is not None:
            residual = residual * self.scale[:, None, None]
        return inputs + self.stochastic_depth(residual)


class WaferConvNeXt(nn.Module):
    """Stages of ConvNeXt blocks with normalized downsampling between them."""

    def __init__(
        self,
        *,
        widths: tuple[int, ...] = (32, 64, 128),
        blocks_per_stage: tuple[int, ...] = (2, 2, 2),
        kernel_size: int = 7,
        expansion: int = 4,
        dilation: int | tuple[int, ...] = 1,
        stem_stride: int = 2,
        stochastic_depth: float = 0.05,
        dropout: float = 0.3,
        attention: str | None = None,
    ) -> None:
        super().__init__()
        if len(widths) != len(blocks_per_stage):
            raise ValueError("widths and blocks_per_stage must have equal length.")
        if not widths:
            raise ValueError("widths must name at least one stage.")
        if stem_stride not in (1, 2):
            raise ValueError(
                "stem_stride must be 1 or 2. The reference 4x4 stride-4 patchify "
                "discards 94% of a 64x64 wafer map before the first block."
            )
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")

        # One rate per stage, so a config can dilate only where it is useful --
        # typically the last stage, once the map is small and context is what
        # is missing. An int applies the same rate everywhere.
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

        layers: list[nn.Module] = [
            nn.Conv2d(
                WAFER_STATE_COUNT,
                widths[0],
                kernel_size=3,
                stride=stem_stride,
                padding=1,
            ),
            LayerNorm2d(widths[0]),
        ]
        # Linearly increasing drop rate with depth, as in the paper.
        total_blocks = sum(blocks_per_stage)
        rates = [
            stochastic_depth * index / max(1, total_blocks - 1)
            for index in range(total_blocks)
        ]
        position = 0
        for stage, (width, count) in enumerate(zip(widths, blocks_per_stage)):
            if stage > 0:
                # Downsampling is its own normalized layer, not folded into a
                # block -- the separation is part of the design.
                layers += [
                    LayerNorm2d(widths[stage - 1]),
                    nn.Conv2d(widths[stage - 1], width, kernel_size=2, stride=2),
                ]
            for _ in range(count):
                layers.append(
                    ConvNeXtBlock(
                        width,
                        kernel_size=kernel_size,
                        expansion=expansion,
                        dilation=rates_per_stage[stage],
                        stochastic_depth=rates[position],
                    )
                )
                position += 1

        self.dilation_rates = rates_per_stage
        self.encoder = nn.Sequential(*layers)
        self.attention = build_attention(attention, widths[-1]) or nn.Identity()
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(
            nn.LayerNorm(widths[-1]),   # applied after pooling, so 1-D
            nn.Dropout(dropout),
            nn.Linear(widths[-1], NUM_CLASSES),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != WAFER_STATE_COUNT:
            raise ValueError(
                "WaferConvNeXt inputs must have shape "
                f"(batch_size, {WAFER_STATE_COUNT}, height, width)."
            )
        if not inputs.is_floating_point():
            raise TypeError("WaferConvNeXt inputs must be floating-point tensors.")
        return self.head(self.pool(self.attention(self.encoder(inputs))))
