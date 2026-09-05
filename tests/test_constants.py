from types import MappingProxyType

import pytest

from fdl_project.constants import (
    CLASS_NAMES,
    CLASS_TO_INDEX,
    INDEX_TO_CLASS,
    NUM_CLASSES,
    class_encoding_metadata,
    decode_index,
    encode_label,
    validate_checkpoint_class_names,
)

EXPECTED_CLASS_NAMES = (
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


def test_class_encoding_is_canonical_and_round_trips() -> None:
    assert CLASS_NAMES == EXPECTED_CLASS_NAMES
    assert NUM_CLASSES == 9
    assert len(set(CLASS_NAMES)) == NUM_CLASSES

    for expected_index, class_name in enumerate(CLASS_NAMES):
        assert encode_label(class_name) == expected_index
        assert decode_index(expected_index) == class_name


def test_class_mappings_are_runtime_read_only() -> None:
    assert isinstance(CLASS_TO_INDEX, MappingProxyType)
    assert isinstance(INDEX_TO_CLASS, MappingProxyType)

    with pytest.raises(TypeError):
        CLASS_TO_INDEX["Center"] = 8  # type: ignore[index]
    with pytest.raises(TypeError):
        INDEX_TO_CLASS[0] = "none"  # type: ignore[index]


@pytest.mark.parametrize("label", ["Near-Full", "None", "unknown", 0, None])
def test_unknown_or_noncanonical_labels_are_rejected(label: object) -> None:
    with pytest.raises(ValueError, match="Unknown WM-811K label"):
        encode_label(label)  # type: ignore[arg-type]


@pytest.mark.parametrize("index", [-1, 9])
def test_out_of_range_class_indices_are_rejected(index: int) -> None:
    with pytest.raises(ValueError, match="Class index"):
        decode_index(index)


@pytest.mark.parametrize("index", [1.5, True, "0", None])
def test_noninteger_class_indices_are_rejected(index: object) -> None:
    with pytest.raises(TypeError, match="Class index"):
        decode_index(index)  # type: ignore[arg-type]


def test_checkpoint_metadata_is_serializable_and_validated() -> None:
    metadata = class_encoding_metadata()
    assert metadata == {
        "class_names": list(CLASS_NAMES),
        "class_to_index": dict(CLASS_TO_INDEX),
        "num_classes": NUM_CLASSES,
    }
    validate_checkpoint_class_names(metadata["class_names"])

    with pytest.raises(ValueError, match="Checkpoint class encoding"):
        validate_checkpoint_class_names(reversed(CLASS_NAMES))
