"""DenseNet-style network: every layer sees every earlier layer's features.

The argument for it here is parameter efficiency on a small dataset. Dense
connectivity reuses features instead of relearning them, so a given accuracy
costs fewer parameters -- and with ~121k training images dominated by one
class, fewer parameters is the safer side to err on.

It also preserves early, high-resolution feature maps all the way to the
classifier by concatenation rather than replacement, which is the same concern
that shapes the dilated model: fine detail should not have to survive being
repeatedly overwritten.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from fdl_project.constants import NUM_CLASSES
from fdl_project.data.preprocessing import WAFER_STATE_COUNT
from fdl_project.models.attention import build_attention


class DenseLayer(nn.Module):
    """Bottleneck 1x1 then 3x3, producing ``growth_rate`` new channels."""

    def __init__(
        self, in_channels: int, growth_rate: int, *, bottleneck: int = 4, dropout: float = 0.0
    ) -> None:
        super().__init__()
        hidden = bottleneck * growth_rate
        # Pre-activation ordering (norm, relu, conv), as in the paper.
        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, growth_rate, kernel_size=3, padding=1, bias=False),
        )
        self.dropout = nn.Dropout2d(dropout) if dropout else nn.Identity()

    def forward(self, inputs: Tensor) -> Tensor:
        return torch.cat([inputs, self.dropout(self.layers(inputs))], dim=1)


class Transition(nn.Module):
    """Compress channels and halve the resolution between dense blocks."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


class WaferDenseNet(nn.Module):
    """Dense blocks with compressing transitions."""

    def __init__(
        self,
        *,
        growth_rate: int = 16,
        block_layers: tuple[int, ...] = (4, 6, 8),
        initial_channels: int = 32,
        compression: float = 0.5,
        block_dropout: float = 0.0,
        dropout: float = 0.3,
        attention: str | None = None,
    ) -> None:
        super().__init__()
        if growth_rate < 1:
            raise ValueError("growth_rate must be positive.")
        if not block_layers or any(count < 1 for count in block_layers):
            raise ValueError("block_layers must be positive counts.")
        if not 0 < compression <= 1:
            raise ValueError("compression must lie in (0, 1].")
        for name, value in {"dropout": dropout, "block_dropout": block_dropout}.items():
            if not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1).")

        layers: list[nn.Module] = [
            nn.Conv2d(
                WAFER_STATE_COUNT,
                initial_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            )
        ]
        channels = initial_channels
        for position, count in enumerate(block_layers):
            for _ in range(count):
                layers.append(DenseLayer(channels, growth_rate, dropout=block_dropout))
                channels += growth_rate
            if position < len(block_layers) - 1:
                compressed = max(1, int(channels * compression))
                layers.append(Transition(channels, compressed))
                channels = compressed
        layers += [nn.BatchNorm2d(channels), nn.ReLU(inplace=True)]

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
                "WaferDenseNet inputs must have shape "
                f"(batch_size, {WAFER_STATE_COUNT}, height, width)."
            )
        if not inputs.is_floating_point():
            raise TypeError("WaferDenseNet inputs must be floating-point tensors.")
        return self.head(self.pool(self.attention(self.encoder(inputs))))
