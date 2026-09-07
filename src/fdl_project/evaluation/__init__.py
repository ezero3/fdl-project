"""Shared evaluation pipeline: metrics, uncertainty, and saved artifacts."""

from fdl_project.evaluation.artifacts import save_evaluation_results
from fdl_project.evaluation.bootstrap import BootstrapResult, bootstrap_evaluation
from fdl_project.evaluation.postprocessing import (
    apply_class_weights,
    collect_predictions_with_tta,
    load_class_weights,
    save_class_weights,
    tune_class_weights,
)
from fdl_project.evaluation.metrics import (
    CollectedPredictions,
    EvaluationResult,
    collect_predictions,
    evaluate_model,
    evaluate_predictions,
)

__all__ = [
    "BootstrapResult",
    "apply_class_weights",
    "collect_predictions_with_tta",
    "load_class_weights",
    "save_class_weights",
    "tune_class_weights",
    "CollectedPredictions",
    "EvaluationResult",
    "bootstrap_evaluation",
    "collect_predictions",
    "evaluate_model",
    "evaluate_predictions",
    "save_evaluation_results",
]
