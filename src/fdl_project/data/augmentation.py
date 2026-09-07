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

#: Named subgroups of D4. Every one is closed and label-preserving, so an
#: ablation between them is a fair comparison rather than three different
#: amounts of noise.
#:
#: index 0-3 rotate by 0/90/180/270; 4-7 are those rotations then mirrored.
#: So 4 is the horizontal mirror, 6 the vertical mirror, and 2 is both.
TRANSFORM_SUBSETS: dict[str, tuple[int, ...]] = {
    "dihedral8": (0, 1, 2, 3, 4, 5, 6, 7),  # full D4
    "rotations": (0, 1, 2, 3),              # C4: rotations only
    "flips": (0, 2, 4, 6),                  # Klein group: identity, h, v, both
}


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
    """Pick one symmetry per sample from a chosen subgroup of D4.

    ``probability`` is the chance of augmenting at all; at 1.0 every sample is
    drawn uniformly from ``transforms``, which includes the identity.
    """

    transforms: tuple[int, ...] = TRANSFORM_SUBSETS["dihedral8"]
    probability: float = 1.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.probability, bool)
            or not isinstance(self.probability, (int, float))
            or not 0.0 <= self.probability <= 1.0
        ):
            raise ValueError("augmentation probability must lie in [0, 1].")
        transforms = tuple(int(index) for index in self.transforms)
        if not transforms:
            raise ValueError("augmentation transforms must not be empty.")
        if any(not 0 <= index < NUM_DIHEDRAL_TRANSFORMS for index in transforms):
            raise ValueError(
                f"transform indices must lie in [0, {NUM_DIHEDRAL_TRANSFORMS})."
            )
        if len(set(transforms)) != len(transforms):
            raise ValueError("transform indices must be unique.")
        object.__setattr__(self, "transforms", transforms)

    def __call__(self, tensor: Tensor) -> Tensor:
        """Augment one sample, drawing from the ambient (seeded) torch RNG.

        The draw uses torch's global generator, which ``seed_everything`` seeds
        and ``seed_worker`` re-seeds per dataloader worker -- so views are
        reproducible for a given seed and differ between workers.
        """

        if self.probability < 1.0 and float(torch.rand(())) >= self.probability:
            return tensor
        choice = int(torch.randint(len(self.transforms), ()))
        return apply_dihedral(tensor, self.transforms[choice])


def build_augmentation(
    name: str | None, **kwargs: float
) -> DihedralAugmentation | None:
    """Build a configured augmentation, or ``None`` when disabled.

    Augmentation is off unless a config names one, and is train-only -- see
    ``WM811KDataset``, which refuses it on validation and test.
    """

    if name is None:
        return None
    try:
        transforms = TRANSFORM_SUBSETS[name]
    except KeyError:
        available = ", ".join(sorted(TRANSFORM_SUBSETS))
        raise KeyError(
            f"Unknown augmentation {name!r}. Available: {available}."
        ) from None
    return DihedralAugmentation(transforms=transforms, **kwargs)


def available_augmentations() -> tuple[str, ...]:
    return tuple(sorted(TRANSFORM_SUBSETS))
