"""Multi-scale CNN with parallel kernel sizes, at this project's scale.

The motivating observation: defect size varies by more than an order of
magnitude between classes. A `Scratch` is one die wide, a `Loc` cluster spans a
handful, `Center` a sizeable disc, and `Near-full` most of the wafer. A single
kernel size has to compromise; parallel branches do not.

This is the Inception *idea* -- concatenated branches of different receptive
field -- at a few million parameters rather than GoogLeNet's scale.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from fdl_project.constants import NUM_CLASSES
from fdl_project.data.preprocessing import WAFER_STATE_COUNT
from fdl_project.models.attention import build_attention


def _conv_block(in_channels: int, out_channels: int, kernel_size: int, **kwargs) -> nn.Sequential:
    """Convolution, batch norm, ReLU -- the unit every branch is built from."""

    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
            **kwargs,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class InceptionBlock(nn.Module):
    """Four parallel branches, concatenated on the channel axis.

    The 5x5 receptive field is built from two stacked 3x3 convolutions: same
    coverage, fewer parameters, one more non-linearity. The 1x1 reductions in
    front of the wider branches are what keep the block affordable.
    """

    def __init__(self, in_channels: int, branch_channels: int) -> None:
        super().__init__()
        reduced = max(1, branch_channels // 2)

        self.branch_pointwise = _conv_block(in_channels, branch_channels, 1)
        self.branch_medium = nn.Sequential(
            _conv_block(in_channels, reduced, 1),
            _conv_block(reduced, branch_channels, 3),
        )
        self.branch_wide = nn.Sequential(
            _conv_block(in_channels, reduced, 1),
            _conv_block(reduced, branch_channels, 3),
            _conv_block(branch_channels, branch_channels, 3),
        )
        self.branch_pooled = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            _conv_block(in_channels, branch_channels, 1),
        )
        self.out_channels = branch_channels * 4

    def forward(self, inputs: Tensor) -> Tensor:
        return torch.cat(
            [
                self.branch_pointwise(inputs),
                self.branch_medium(inputs),
                self.branch_wide(inputs),
                self.branch_pooled(inputs),
            ],
            dim=1,
        )


class WaferInception(nn.Module):
    """Stacked multi-scale blocks with strided transitions between them."""

    def __init__(
        self,
        *,
        branch_channels: tuple[int, ...] = (32, 64, 96),
        blocks_per_stage: int = 2,
        dropout: float = 0.3,
        attention: str | None = None,
    ) -> None:
        super().__init__()
        if not branch_channels:
            raise ValueError("branch_channels must name at least one stage.")
        if blocks_per_stage < 1:
            raise ValueError("blocks_per_stage must be at least 1.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")

        layers: list[nn.Module] = [_conv_block(WAFER_STATE_COUNT, 32, 3)]
        channels = 32
        for position, width in enumerate(branch_channels):
            if position > 0:
                # Halve the resolution between stages, not inside a block, so
                # every branch in a block sees the same grid.
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            for _ in range(blocks_per_stage):
                block = InceptionBlock(channels, width)
                layers.append(block)
                channels = block.out_channels

        self.encoder = nn.Sequential(*layers)
        self.attention = build_attention(attention, channels) or nn.Identity()
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels, NUM_CLASSES),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != WAFER_STATE_COUNT:
            raise ValueError(
                "WaferInception inputs must have shape "
                f"(batch_size, {WAFER_STATE_COUNT}, height, width)."
            )
        if not inputs.is_floating_point():
            raise TypeError("WaferInception inputs must be floating-point tensors.")
        return self.head(self.pool(self.attention(self.encoder(inputs))))
