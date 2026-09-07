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
