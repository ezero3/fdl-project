"""Canonical class encoding shared by every WM-811K pipeline."""

from collections.abc import Iterable
from numbers import Integral
from types import MappingProxyType
from typing import Final

CLASS_NAMES: Final[tuple[str, ...]] = (
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
    "none",
)
"""Class name at each model-output index. This order must never change."""

NUM_CLASSES: Final[int] = len(CLASS_NAMES)

CLASS_TO_INDEX = MappingProxyType(
    {class_name: class_index for class_index, class_name in enumerate(CLASS_NAMES)}
)
INDEX_TO_CLASS = MappingProxyType(
    {class_index: class_name for class_index, class_name in enumerate(CLASS_NAMES)}
)


def validate_class_names(class_names: Iterable[str]) -> tuple[str, ...]:
    """Return the canonical tuple or reject an incompatible class ordering."""

    names = tuple(class_names)
    if names != CLASS_NAMES:
        raise ValueError(
            "Class names must match the canonical WM-811K encoding exactly: "
            f"{CLASS_NAMES!r}; received {names!r}."
        )
    return names


def encode_label(label: str) -> int:
    """Encode one canonical WM-811K label as its model-output index."""

    try:
        return CLASS_TO_INDEX[label]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Unknown WM-811K label {label!r}. Expected one of {CLASS_NAMES!r}."
        ) from exc


def decode_index(index: int) -> str:
    """Decode one model-output index as its canonical WM-811K label."""

    if isinstance(index, bool) or not isinstance(index, Integral):
        raise TypeError(f"Class index must be an integer in [0, {NUM_CLASSES - 1}].")
    try:
        return INDEX_TO_CLASS[int(index)]
    except KeyError as exc:
        raise ValueError(
            f"Class index must be an integer in [0, {NUM_CLASSES - 1}]; received {index}."
        ) from exc


def class_encoding_metadata() -> dict[str, object]:
    """Return JSON-serializable class metadata for checkpoints and reports."""

    return {
        "class_names": list(CLASS_NAMES),
        "class_to_index": dict(CLASS_TO_INDEX),
        "num_classes": NUM_CLASSES,
    }


def validate_checkpoint_class_names(class_names: Iterable[str]) -> None:
    """Reject a checkpoint whose logits use a different class ordering."""

    try:
        validate_class_names(class_names)
    except ValueError as exc:
        raise ValueError(
            "Checkpoint class encoding is incompatible with this project."
        ) from exc
