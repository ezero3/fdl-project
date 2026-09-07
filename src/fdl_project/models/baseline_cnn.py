"""Small controlled model used to compare training interventions."""

from __future__ import annotations

from torch import Tensor, nn

from fdl_project.constants import NUM_CLASSES
from fdl_project.data.preprocessing import WAFER_STATE_COUNT
from fdl_project.models.attention import build_attention


class BaselineCNN(nn.Module):
    """Compact spatial CNN used as a task-05 experimental instrument."""

    def __init__(
        self, *, dropout: float = 0.2, attention: str | None = None
    ) -> None:
        super().__init__()
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        self.features = nn.Sequential(
            nn.Conv2d(WAFER_STATE_COUNT, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # Shape-preserving, and placed while the map still has spatial extent:
        # after pooling there is nothing left for spatial attention to weigh.
        self.attention = build_attention(attention, 64) or nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != WAFER_STATE_COUNT:
            raise ValueError(
                "BaselineCNN inputs must have shape (batch_size, 3, height, width)."
            )
        if not inputs.is_floating_point():
            raise TypeError("BaselineCNN inputs must be floating-point tensors.")
        return self.classifier(self.pool(self.attention(self.features(inputs))))


def count_trainable_parameters(model: nn.Module) -> int:
    """Count trainable scalar parameters for experiment metadata."""

    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


class BaselineCNNv2(nn.Module):
    """The baseline with the overfitting fixed, holding the stem identical.

    The phase-2 sweep diagnosed the problem: the original reaches 98.6% train
    accuracy by epoch 12 while validation stalls and validation loss climbs.
    **84% of its parameters (132k of 157k) are in a single 1024->128 linear**
    fed by a 4x4 pool -- the convolutions are not what memorises the training
    set, that layer is.

    Two levers, both configurable so they can be attributed separately:

    * ``pooled_size`` -- 1 is global average pooling, which deletes that layer
      outright. Set it to 4 to keep the original head shape and isolate the
      dropout change instead.
    * ``block_dropout`` -- ``Dropout2d`` between convolution stages. It drops
      whole channels rather than individual cells, which is the form that
      actually regularises convolutional features; ordinary dropout on a
      feature map is largely undone by spatial correlation.

    The convolutional stem is byte-identical to ``BaselineCNN`` so a comparison
    between them is about the head and the regularisation, nothing else.
    """

    def __init__(
        self,
        *,
        pooled_size: int = 1,
        block_dropout: float = 0.1,
        dropout: float = 0.4,
        hidden_features: int = 128,
        attention: str | None = None,
    ) -> None:
        super().__init__()
        for name, value in {"dropout": dropout, "block_dropout": block_dropout}.items():
            if not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1).")
        if pooled_size < 1:
            raise ValueError("pooled_size must be at least 1.")

        def block(in_channels: int, out_channels: int, kernel_size: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=kernel_size // 2,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(block_dropout) if block_dropout else nn.Identity(),
            )

        self.features = nn.Sequential(
            block(WAFER_STATE_COUNT, 16, 5), block(16, 32, 3), block(32, 64, 3)
        )
        self.attention = build_attention(attention, 64) or nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d((pooled_size, pooled_size))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * pooled_size * pooled_size, hidden_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, NUM_CLASSES),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != WAFER_STATE_COUNT:
            raise ValueError(
                "BaselineCNNv2 inputs must have shape (batch_size, 3, height, width)."
            )
        if not inputs.is_floating_point():
            raise TypeError("BaselineCNNv2 inputs must be floating-point tensors.")
        return self.classifier(self.pool(self.attention(self.features(inputs))))
