"""Dilated CNN: a wide receptive field without throwing away resolution.

This is the architecture with the most specific argument for this dataset.
Every other design widens its receptive field by downsampling, and downsampling
is precisely what destroys the patterns we already know are hardest here -- a
one-die-wide `Scratch`, a small `Loc` cluster. Dilation buys the same context
by spacing the kernel out instead, at full resolution.

Concretely, a stack with dilation rates 1, 2, 4, 8 reaches a receptive field of
31x31 on a 64x64 map having downsampled **not at all**. A stride-2 stack
reaching the same context has already thrown away 87.5% of the grid.

The known failure mode of dilation is gridding: consecutive layers at the same
rate sample disjoint lattices and leave checkerboard holes. Cycling the rates
(1, 2, 4, 8, 1, 2, 4, 8) rather than climbing monotonically is the standard fix
and is what this does.
"""

from __future__ import annotations

from torch import Tensor, nn

from fdl_project.constants import NUM_CLASSES
from fdl_project.data.preprocessing import WAFER_STATE_COUNT
from fdl_project.models.attention import build_attention

#: Rate cycle. Repeating rather than climbing avoids gridding artifacts.
DEFAULT_DILATION_RATES = (1, 2, 4, 8, 1, 2, 4, 8)


class DilatedBlock(nn.Module):
    """One dilated convolution with a residual connection, at fixed resolution."""

    def __init__(self, channels: int, dilation: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,  # keeps the spatial size for any rate
            dilation=dilation,
            bias=False,
        )
        self.normalization = nn.BatchNorm2d(channels)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout else nn.Identity()
        self.dilation = dilation

    def forward(self, inputs: Tensor) -> Tensor:
        residual = self.dropout(self.normalization(self.convolution(inputs)))
        return self.activation(residual + inputs)


class WaferDilatedCNN(nn.Module):
    """Dilated residual stack; ``downsample_every`` controls how much is lost.

    Leave ``downsample_every`` at 0 to keep full resolution throughout, which
    is the point of the architecture. Set it to 3 or 4 for a cheaper run.
    """

    def __init__(
        self,
        *,
        channels: int = 64,
        dilation_rates: tuple[int, ...] = DEFAULT_DILATION_RATES,
        downsample_every: int = 0,
        block_dropout: float = 0.05,
        dropout: float = 0.3,
        attention: str | None = None,
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive.")
        if not dilation_rates:
            raise ValueError("dilation_rates must not be empty.")
        if any(rate < 1 for rate in dilation_rates):
            raise ValueError("dilation rates must be positive.")
        if downsample_every < 0:
            raise ValueError("downsample_every must be non-negative.")
        for name, value in {"dropout": dropout, "block_dropout": block_dropout}.items():
            if not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1).")

        layers: list[nn.Module] = [
            nn.Conv2d(WAFER_STATE_COUNT, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        ]
        for position, rate in enumerate(dilation_rates):
            layers.append(DilatedBlock(channels, rate, dropout=block_dropout))
            if downsample_every and (position + 1) % downsample_every == 0:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.encoder = nn.Sequential(*layers)
        self.attention = build_attention(attention, channels) or nn.Identity()
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels, NUM_CLASSES),
        )
        self.dilation_rates = tuple(dilation_rates)

    @property
    def receptive_field(self) -> int:
        """Side length in cells, ignoring any downsampling.

        Each 3x3 convolution at rate d adds 2*d, and the stem adds 2.
        """

        return 1 + 2 + sum(2 * rate for rate in self.dilation_rates)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != WAFER_STATE_COUNT:
            raise ValueError(
                "WaferDilatedCNN inputs must have shape "
                f"(batch_size, {WAFER_STATE_COUNT}, height, width)."
            )
        if not inputs.is_floating_point():
            raise TypeError("WaferDilatedCNN inputs must be floating-point tensors.")
        return self.head(self.pool(self.attention(self.encoder(inputs))))
