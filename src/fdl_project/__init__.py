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
from fdl_project.datasets import (
    WM811KDataset,
    create_dataloader,
    create_split_dataset,
    load_split_indices,
    load_wm811k_dataframe,
)
from fdl_project.evaluation import (
    CollectedPredictions,
    EvaluationResult,
    collect_predictions,
    evaluate_model,
    evaluate_predictions,
    save_evaluation_results,
)
from fdl_project.preprocessing import (
    DEFAULT_PREPROCESSING_CONFIG,
    WAFER_STATE_COUNT,
    WAFER_STATE_NAMES,
    PreprocessingConfig,
    WaferMapPreprocessor,
    decode_preprocessed_map,
    transform_categorical_map,
    validate_wafer_map,
)

__all__ = [
    "CLASS_NAMES",
    "CLASS_TO_INDEX",
    "DEFAULT_PREPROCESSING_CONFIG",
    "INDEX_TO_CLASS",
    "NUM_CLASSES",
    "WAFER_STATE_COUNT",
    "WAFER_STATE_NAMES",
    "CollectedPredictions",
    "EvaluationResult",
    "PreprocessingConfig",
    "WM811KDataset",
    "WaferMapPreprocessor",
    "class_encoding_metadata",
    "collect_predictions",
    "create_dataloader",
    "create_split_dataset",
    "decode_index",
    "decode_preprocessed_map",
    "encode_label",
    "evaluate_model",
    "evaluate_predictions",
    "load_split_indices",
    "load_wm811k_dataframe",
    "save_evaluation_results",
    "transform_categorical_map",
    "validate_checkpoint_class_names",
    "validate_wafer_map",
]
