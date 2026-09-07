"""Shared evaluation pipeline: metrics, uncertainty, and saved artifacts."""

from fdl_project.evaluation.artifacts import save_evaluation_results
from fdl_project.evaluation.bootstrap import BootstrapResult, bootstrap_evaluation
from fdl_project.evaluation.metrics import (
    CollectedPredictions,
    EvaluationResult,
    collect_predictions,
    evaluate_model,
    evaluate_predictions,
)

__all__ = [
    "BootstrapResult",
    "CollectedPredictions",
    "EvaluationResult",
    "bootstrap_evaluation",
    "collect_predictions",
    "evaluate_model",
    "evaluate_predictions",
    "save_evaluation_results",
]
