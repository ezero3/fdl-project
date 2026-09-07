"""CBAM: convolutional block attention, channel then spatial.

Woo et al., "CBAM: Convolutional Block Attention Module" (ECCV 2018).

The reason it is worth a run on this dataset: a defect occupies a small
fraction of a wafer map -- often a few dozen dies out of a few thousand -- and
the rest of the image is a large, almost identical disc shared by every class.
Spatial attention is the standard answer to "the signal is small and the
background dominates".

The module is shape-preserving, so it drops in between a feature extractor and
its pooling without touching either.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ChannelAttention(nn.Module):
    """Reweight channels from their global average and maximum responses.

    Average pooling says how much of a feature is present overall; max pooling
    says whether it is present strongly anywhere. A small localized defect
    barely moves the average, which is why both are used.
    """

    def __init__(self, channels: int, *, reduction: int = 16) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive.")
        if reduction <= 0:
            raise ValueError("reduction must be positive.")
        hidden = max(1, channels // reduction)
        self.shared = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        average = self.shared(inputs.mean(dim=(2, 3), keepdim=True))
        maximum = self.shared(inputs.amax(dim=(2, 3), keepdim=True))
        return inputs * (average + maximum).sigmoid()


class SpatialAttention(nn.Module):
    """Reweight positions from the channel-wise average and maximum."""

    def __init__(self, *, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size % 2 == 0 or kernel_size <= 0:
            raise ValueError("kernel_size must be a positive odd number.")
        self.convolution = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )

    def forward(self, inputs: Tensor) -> Tensor:
        average = inputs.mean(dim=1, keepdim=True)
        maximum = inputs.amax(dim=1, keepdim=True)
        weights = self.convolution(torch.cat([average, maximum], dim=1))
        return inputs * weights.sigmoid()


class CBAM(nn.Module):
    """Channel attention followed by spatial attention, as in the paper."""

    def __init__(
        self, channels: int, *, reduction: int = 16, kernel_size: int = 7
    ) -> None:
        super().__init__()
        self.channel = ChannelAttention(channels, reduction=reduction)
        self.spatial = SpatialAttention(kernel_size=kernel_size)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4:
            raise ValueError(
                "CBAM expects a feature map of shape (batch, channels, height, width)."
            )
        return self.spatial(self.channel(inputs))


#: Name -> factory, so a config can say `attention: cbam`.
ATTENTION_REGISTRY = {"cbam": CBAM}


def build_attention(name: str | None, channels: int, **kwargs: int) -> nn.Module | None:
    """Build a shape-preserving attention block, or ``None`` when disabled."""

    if name is None or (isinstance(name, str) and name.lower() in {"none", ""}):
        return None
    try:
        factory = ATTENTION_REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(ATTENTION_REGISTRY))
        raise KeyError(
            f"Unknown attention block {name!r}. Available: {available}."
        ) from None
    return factory(channels, **kwargs)


def available_attention_blocks() -> tuple[str, ...]:
    return tuple(sorted(ATTENTION_REGISTRY))
