"""Label-preserving augmentation for wafer maps.

Only the **dihedral group of order 8** is used: the four rotations by multiples
of 90 degrees, and each of those mirrored. These are index permutations, so
they introduce no interpolation, produce no fractional cell states, and cannot
turn one class into another.

What is deliberately excluded, and why:

* **Free-angle rotation** resamples the grid. A one-die-wide `Scratch` breaks
  into a dotted line or disappears, while the label still says `Scratch`.
* **Translation** moves a localized defect toward the wafer edge, which is
  exactly the difference between `Loc` and `Edge-Loc`. The image becomes a
  different class while keeping the old label.
* **Intensity jitter, blur, noise** are meaningless here: cells are categorical
  (0 background, 1 functional die, 2 defective die), not intensities.

The same eight transforms are reused for test-time augmentation, so a model
trained on them is evaluated over the group it already knows.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

#: Rotations by 90 degrees, each optionally mirrored: |D4| = 8.
NUM_DIHEDRAL_TRANSFORMS = 8


def apply_dihedral(tensor: Tensor, index: int) -> Tensor:
    """Apply one of the eight square symmetries to the last two dimensions.

    ``index`` 0-3 are rotations by 0, 90, 180, 270 degrees; 4-7 are those
    rotations followed by a horizontal mirror. Index 0 is the identity.
    """

    if not 0 <= index < NUM_DIHEDRAL_TRANSFORMS:
        raise ValueError(
            f"index must lie in [0, {NUM_DIHEDRAL_TRANSFORMS}); received {index}."
        )
    if tensor.ndim < 2:
        raise ValueError("Dihedral transforms need at least two dimensions.")

    transformed = torch.rot90(tensor, index % 4, dims=(-2, -1))
    if index >= 4:
        transformed = torch.flip(transformed, dims=(-1,))
    return transformed


def dihedral_views(tensor: Tensor) -> list[Tensor]:
    """Return all eight symmetries of ``tensor``, identity first."""

    return [apply_dihedral(tensor, index) for index in range(NUM_DIHEDRAL_TRANSFORMS)]


@dataclass(frozen=True)
class DihedralAugmentation:
    """Pick one of the eight symmetries per sample.

    ``probability`` is the chance of applying a *non-identity* transform; at
    1.0 every sample is drawn uniformly from all eight, which includes the
    identity one time in eight.
    """

    probability: float = 1.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.probability, bool)
            or not isinstance(self.probability, (int, float))
            or not 0.0 <= self.probability <= 1.0
        ):
            raise ValueError("augmentation probability must lie in [0, 1].")

    def __call__(self, tensor: Tensor) -> Tensor:
        """Augment one sample, drawing from the ambient (seeded) torch RNG.

        The draw uses torch's global generator, which ``seed_everything`` seeds
        and ``seed_worker`` re-seeds per dataloader worker -- so views are
        reproducible for a given seed and differ between workers.
        """

        if self.probability < 1.0:
            if float(torch.rand(())) >= self.probability:
                return tensor
        index = int(torch.randint(NUM_DIHEDRAL_TRANSFORMS, ()))
        return apply_dihedral(tensor, index)


#: Name -> factory. Augmentation is train-only; see WM811KDataset.
AUGMENTATION_REGISTRY = {
    "dihedral8": DihedralAugmentation,
}


def build_augmentation(name: str | None, **kwargs: float) -> DihedralAugmentation | None:
    """Build a configured augmentation, or ``None`` when disabled."""

    if name is None:
        return None
    try:
        factory = AUGMENTATION_REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(AUGMENTATION_REGISTRY))
        raise KeyError(
            f"Unknown augmentation {name!r}. Available: {available}."
        ) from None
    return factory(**kwargs)


def available_augmentations() -> tuple[str, ...]:
    return tuple(sorted(AUGMENTATION_REGISTRY))
