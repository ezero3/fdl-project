"""Shared utilities for the WM-811K deep-learning project."""

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
from fdl_project.evaluation import (
    CollectedPredictions,
    EvaluationResult,
    collect_predictions,
    evaluate_model,
    evaluate_predictions,
    save_evaluation_results,
)

__all__ = [
    "CLASS_NAMES",
    "CLASS_TO_INDEX",
    "INDEX_TO_CLASS",
    "NUM_CLASSES",
    "CollectedPredictions",
    "EvaluationResult",
    "class_encoding_metadata",
    "collect_predictions",
    "decode_index",
    "encode_label",
    "evaluate_model",
    "evaluate_predictions",
    "save_evaluation_results",
    "validate_checkpoint_class_names",
]
