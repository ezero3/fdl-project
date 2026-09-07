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

from fdl_project.constants import CLASS_NAMES, NUM_CLASSES

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


def build_class_probabilities(
    mapping: dict[str, float] | None, *, default: float
) -> tuple[float, ...] | None:
    """Turn a {class name: probability} mapping into a per-index vector.

    Classes the mapping omits keep ``default``. Returns ``None`` when there is
    nothing class-specific to apply, so the uniform path stays free.
    """

    if not mapping:
        return None
    probabilities = [float(default)] * NUM_CLASSES
    for class_name, value in mapping.items():
        if class_name not in CLASS_NAMES:
            raise ValueError(
                f"Unknown class {class_name!r} in per-class augmentation. "
                f"Expected one of {CLASS_NAMES!r}."
            )
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("per-class augmentation probabilities must be numbers.")
        if not 0.0 <= value <= 1.0:
            raise ValueError("per-class augmentation probabilities must lie in [0, 1].")
        probabilities[CLASS_NAMES.index(class_name)] = float(value)
    return tuple(probabilities)


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
    class_probabilities: tuple[float, ...] | None = None

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
        _validate_class_probabilities(self.class_probabilities)

    def __call__(self, tensor: Tensor, class_index: int | None = None) -> Tensor:
        """Augment one sample, drawing from the ambient (seeded) torch RNG.

        The draw uses torch's global generator, which ``seed_everything`` seeds
        and ``seed_worker`` re-seeds per dataloader worker -- so views are
        reproducible for a given seed and differ between workers.
        """

        if not _should_augment(
            self.probability, self.class_probabilities, class_index
        ):
            return tensor
        choice = int(torch.randint(len(self.transforms), ()))
        return apply_dihedral(tensor, self.transforms[choice])


@dataclass(frozen=True)
class RotationAugmentation:
    """Rotate by a free angle, nearest-neighbour, on the one-hot tensor.

    Unlike the dihedral group this is *not* an index permutation: the output
    grid does not line up with the input grid, so cells are resampled. Nearest
    neighbour keeps every output cell a copy of some input cell, so the one-hot
    encoding stays exactly one-hot and no fractional state appears -- but a
    thin structure can still be thinned, thickened or broken by the resampling.

    Whether that matters is a measured question, not an assumed one; see
    `docs/reports/` for what it does to `Scratch`.

    ``fill`` sets the corners the rotation exposes. It defaults to the
    background channel, which is the only valid one-hot for "no die here".
    """

    degrees: float = 180.0
    probability: float = 1.0
    class_probabilities: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.class_probabilities is not None:
            # Resampling leaves faint artifacts. Applying it to some classes
            # and not others makes "looks resampled" predict "is a rare class",
            # and the network will happily learn that instead of the defect.
            # The exact-permutation subsets have no such artifact, so they are
            # the ones that may vary by class.
            raise ValueError(
                "Per-class probabilities are not allowed for 'rotation': it "
                "resamples, so applying it unevenly across classes leaks the "
                "label. Use dihedral8, rotations, or flips instead."
            )
        if (
            isinstance(self.degrees, bool)
            or not isinstance(self.degrees, (int, float))
            or not 0.0 < self.degrees <= 180.0
        ):
            raise ValueError("rotation degrees must lie in (0, 180].")
        if (
            isinstance(self.probability, bool)
            or not isinstance(self.probability, (int, float))
            or not 0.0 <= self.probability <= 1.0
        ):
            raise ValueError("augmentation probability must lie in [0, 1].")

    def __call__(self, tensor: Tensor, class_index: int | None = None) -> Tensor:
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.v2 import functional as transforms_functional

        if not _should_augment(self.probability, None, class_index):
            return tensor
        angle = float(torch.empty(()).uniform_(-self.degrees, self.degrees))
        # One-hot channel 0 is "no die", so filling it is the only encoding of
        # the empty corners a rotation exposes. Filling every channel with 0
        # would produce a cell belonging to no state at all.
        fill = [1.0] + [0.0] * (tensor.shape[0] - 1)
        return transforms_functional.rotate(
            tensor,
            angle,
            interpolation=InterpolationMode.NEAREST,
            fill=fill,
        )


def _validate_class_probabilities(probabilities: tuple[float, ...] | None) -> None:
    if probabilities is None:
        return
    if len(probabilities) != NUM_CLASSES:
        raise ValueError(
            f"class_probabilities must have one entry per class ({NUM_CLASSES})."
        )
    if any(not 0.0 <= value <= 1.0 for value in probabilities):
        raise ValueError("class_probabilities must lie in [0, 1].")


def _should_augment(
    probability: float,
    class_probabilities: tuple[float, ...] | None,
    class_index: int | None,
) -> bool:
    """Decide whether this sample is augmented at all."""

    chance = probability
    if class_probabilities is not None and class_index is not None:
        chance = class_probabilities[int(class_index)]
    if chance >= 1.0:
        return True
    if chance <= 0.0:
        return False
    return float(torch.rand(())) < chance


def _dihedral_factory(name: str):
    def build(**kwargs) -> DihedralAugmentation:
        return DihedralAugmentation(transforms=TRANSFORM_SUBSETS[name], **kwargs)

    return build


#: Name -> factory. Augmentation is train-only; see WM811KDataset.
AUGMENTATION_REGISTRY = {
    name: _dihedral_factory(name) for name in TRANSFORM_SUBSETS
} | {"rotation": RotationAugmentation}


def build_augmentation(
    name: str | None,
    *,
    class_probabilities: dict[str, float] | None = None,
    probability: float = 1.0,
    **kwargs,
):
    """Build a configured augmentation, or ``None`` when disabled.

    Augmentation is off unless a config names one, and is train-only -- see
    ``WM811KDataset``, which refuses it on validation and test.
    """

    if name is None:
        return None
    try:
        factory = AUGMENTATION_REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(AUGMENTATION_REGISTRY))
        raise KeyError(
            f"Unknown augmentation {name!r}. Available: {available}."
        ) from None
    resolved = build_class_probabilities(class_probabilities, default=probability)
    if resolved is not None:
        kwargs["class_probabilities"] = resolved
    return factory(probability=probability, **kwargs)


def available_augmentations() -> tuple[str, ...]:
    return tuple(sorted(AUGMENTATION_REGISTRY))
