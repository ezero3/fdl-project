"""Preprocessing that runs on the accelerator instead of the dataloader.

Phase 0 established that these runs are dataloader-bound, not GPU-bound: the
T4 sat at ~22% utilisation with every core busy encoding wafer maps. The
phase-2 sweep then made it worse, because the augmentation that wins --
free-angle rotation -- is also the most expensive CPU step, at 19.4s per epoch
against 11.8s with none.

Both of those costs move to the GPU cleanly:

* **Encoding.** One-hot is a comparison against three constants. Doing it on
  device also means shipping ``uint8`` categorical maps rather than
  ``float32`` three-channel ones -- exactly 12x less data over PCIe.
* **Rotation.** ``affine_grid`` + ``grid_sample`` rotates a whole batch at
  per-sample angles in one call, with no Python loop.

Semantics are unchanged. The CPU path rotates the one-hot tensor with nearest
-neighbour interpolation and a background fill; rotating the categorical map
with nearest and padding 0, then encoding, produces the same result, because
nearest-neighbour copies whole cells either way. That is the property that
keeps the output exactly one-hot.

What does change is where the randomness comes from: CUDA RNG rather than
per-worker CPU RNG. Both are seeded and checkpointed, but a ``cuda`` run is
not bit-identical to a ``cpu`` run at the same seed. It is a different
pipeline drawing a different stream of views, not a loss of reproducibility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from fdl_project.data.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    WAFER_STATE_COUNT,
    PreprocessingConfig,
)


def encode_batch(states: Tensor, config: PreprocessingConfig) -> Tensor:
    """Encode a batch of categorical maps ``[B, H, W]`` into model input.

    Comparisons against the three state constants rather than ``F.one_hot``:
    it avoids materialising an int64 intermediate the size of the output.
    """

    if states.ndim != 3:
        raise ValueError(
            f"states must have shape (batch, height, width); got {tuple(states.shape)}."
        )

    if config.encoding == "one_hot":
        return torch.stack(
            [states == state for state in range(WAFER_STATE_COUNT)], dim=1
        ).to(torch.float32)

    grayscale = states.to(torch.float32) / (WAFER_STATE_COUNT - 1)
    if config.encoding == "single_channel":
        if config.normalization == "divide_by_two":
            return grayscale.unsqueeze(1)
        return states.to(torch.float32).unsqueeze(1)

    image = grayscale.unsqueeze(1).expand(-1, WAFER_STATE_COUNT, -1, -1)
    if config.normalization == "imagenet":
        mean = torch.tensor(IMAGENET_MEAN, device=states.device)[None, :, None, None]
        std = torch.tensor(IMAGENET_STD, device=states.device)[None, :, None, None]
        return (image - mean) / std
    return image.contiguous()


def rotate_batch(states: Tensor, angles: Tensor) -> Tensor:
    """Rotate each map by its own angle, nearest neighbour, background fill.

    ``grid_sample`` samples in normalised coordinates, so the rotation is
    about the centre of each map without any explicit translation. Padding
    zeros is what fills the corners a rotation exposes, and 0 is the
    background state -- the categorical equivalent of the one-hot background
    channel the CPU path fills.
    """

    if len(states) != len(angles):
        raise ValueError("one angle per map is required.")

    radians = angles * (math.pi / 180.0)
    cosine, sine = torch.cos(radians), torch.sin(radians)
    matrices = torch.zeros(len(states), 2, 3, device=states.device, dtype=torch.float32)
    matrices[:, 0, 0], matrices[:, 0, 1] = cosine, -sine
    matrices[:, 1, 0], matrices[:, 1, 1] = sine, cosine

    working = states.unsqueeze(1).to(torch.float32)
    grid = F.affine_grid(matrices, list(working.shape), align_corners=False)
    rotated = F.grid_sample(
        working, grid, mode="nearest", padding_mode="zeros", align_corners=False
    )
    return rotated.squeeze(1).to(states.dtype)


@dataclass
class BatchTransform:
    """Turn a batch of categorical maps into model input, on the device.

    ``augmentation`` is applied only when ``training`` is true, which is what
    keeps validation and test evaluated at their natural distribution.
    """

    preprocessing: PreprocessingConfig
    augmentation_name: str | None = None
    probability: float = 1.0
    degrees: float = 180.0

    def __post_init__(self) -> None:
        supported = {None, "rotation"}
        if self.augmentation_name not in supported:
            raise ValueError(
                f"transform_device='cuda' supports {sorted(str(n) for n in supported)} "
                f"augmentation, not {self.augmentation_name!r}. The dihedral "
                "subsets are exact index permutations and cost almost nothing "
                "on CPU, so they stay there; run them with transform_device='cpu'."
            )
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must lie in [0, 1].")

    def __call__(self, states: Tensor, *, training: bool) -> Tensor:
        if states.dtype != torch.uint8:
            raise TypeError(
                f"BatchTransform expects uint8 categorical maps, got {states.dtype}. "
                "The dataset returns those only under transform_device='cuda'."
            )
        if training and self.augmentation_name == "rotation":
            states = self._augment(states)
        return encode_batch(states, self.preprocessing)

    def _augment(self, states: Tensor) -> Tensor:
        angles = torch.empty(len(states), device=states.device).uniform_(
            -self.degrees, self.degrees
        )
        if self.probability < 1.0:
            # Zero the angle for the samples left untouched, so the whole
            # batch still goes through one grid_sample call.
            keep = torch.rand(len(states), device=states.device) >= self.probability
            angles = angles.masked_fill(keep, 0.0)
        return rotate_batch(states, angles)
