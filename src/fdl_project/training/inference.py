"""Evaluate a saved checkpoint on one split, with its own training geometry.

Preprocessing is rebuilt from the config stored *inside* the checkpoint rather
than from whatever YAML happens to be at hand, so a model trained at 224x224
can never be silently evaluated at 64x64.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fdl_project.config.loader import build_experiment_config
from fdl_project.config.registry import build_model
from fdl_project.config.schema import ExperimentConfig
from fdl_project.constants import CLASS_NAMES
from fdl_project.data.datasets import (
    create_dataloader,
    create_split_dataset,
    load_wm811k_dataframe,
)
from fdl_project.evaluation import (
    BootstrapResult,
    EvaluationResult,
    apply_class_weights,
    bootstrap_evaluation,
    collect_predictions,
    collect_predictions_with_tta,
    evaluate_predictions,
    load_class_weights,
    save_class_weights,
    save_evaluation_results,
    tune_class_weights,
)
from fdl_project.training.checkpoint import load_checkpoint
from fdl_project.training.loop import resolve_device
from fdl_project.training.seed import seed_everything


@dataclass(frozen=True)
class InferenceResult:
    """One checkpoint evaluated on one split."""

    config: ExperimentConfig
    split_name: str
    evaluation: EvaluationResult
    bootstrap: BootstrapResult
    probabilities: pd.DataFrame
    artifact_paths: dict[str, Path]
    class_weights: np.ndarray | None = None


def probability_table(
    row_indices: np.ndarray,
    true_indices: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """Full class-probability matrix, keyed by source row index.

    Saved so that post-hoc work -- test-time augmentation, per-class decision
    thresholds, error analysis -- can run later without retraining anything.
    """

    table = pd.DataFrame(
        {"row_index": row_indices, "true_index": true_indices}
    )
    for class_index, class_name in enumerate(CLASS_NAMES):
        table[f"probability_{class_name}"] = probabilities[:, class_index]
    return table


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    *,
    split_name: str = "validation",
    output_root: str | Path | None = None,
    run_name: str | None = None,
    device: str | None = None,
    overwrite: bool = False,
    bootstrap_resamples: int = 1000,
    batch_size: int | None = None,
    tta: bool = False,
    tune_thresholds: bool = False,
    class_weights_path: str | Path | None = None,
) -> InferenceResult:
    """Rebuild the model from a checkpoint and score it on one split."""

    if split_name not in {"validation", "test"}:
        raise ValueError("split_name must be 'validation' or 'test'.")

    payload = load_checkpoint(checkpoint_path)
    stored_config = payload.get("config")
    if not stored_config:
        raise ValueError(
            f"{checkpoint_path} carries no experiment config, so the "
            "preprocessing it was trained with is unknown."
        )
    config = build_experiment_config(stored_config)

    seed_everything(config.seed)
    device_object = resolve_device(device or config.trainer.device)

    model = build_model(config.model.name, **dict(config.model.kwargs))
    model.load_state_dict(payload["model_state"])
    model.to(device_object)

    dataframe = load_wm811k_dataframe(config.data.dataset_path)
    dataset = create_split_dataset(
        dataframe,
        config.data.split_directory,
        split_name,
        preprocessing_config=config.data.preprocessing,
        cache_maps=config.data.cache,
    )
    dataloader = create_dataloader(
        dataset,
        batch_size=batch_size or config.trainer.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        seed=config.seed,
    )

    if tta:
        collected = collect_predictions_with_tta(
            model=model, dataloader=dataloader, device=device_object
        )
    else:
        collected = collect_predictions(
            model=model, dataloader=dataloader, device=device_object
        )

    # Per-class weights are *fitted* on validation and *reused* on test; fitting
    # them on test would be selecting on the frozen split.
    class_weights = None
    if tune_thresholds and class_weights_path is not None:
        raise ValueError(
            "Pass either tune_thresholds (fit new weights) or class_weights_path "
            "(reuse fitted ones), not both."
        )
    if tune_thresholds:
        if split_name != "validation":
            raise ValueError("Per-class thresholds must be tuned on validation.")
        class_weights, _ = tune_class_weights(
            collected.probabilities, collected.true_indices
        )
    elif class_weights_path is not None:
        class_weights = load_class_weights(class_weights_path)

    predicted_indices = collected.predicted_indices
    if class_weights is not None:
        predicted_indices = apply_class_weights(collected.probabilities, class_weights)

    evaluation = evaluate_predictions(
        true_indices=collected.true_indices,
        predicted_indices=predicted_indices,
        row_indices=collected.row_indices,
        probabilities=collected.probabilities,
        mean_loss=collected.mean_loss,
        split_name=split_name,
        model_class=collected.model_class,
        device=collected.device,
    )
    bootstrap = bootstrap_evaluation(
        evaluation, num_resamples=bootstrap_resamples, seed=config.seed
    )
    probabilities = probability_table(
        collected.row_indices, collected.true_indices, collected.probabilities
    )

    resolved_run_name = run_name or f"{config.name}-{split_name}"
    resolved_root = Path(output_root or config.logging.output_root)
    metadata: dict[str, Any] = {
        "config": config.to_dict(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": payload.get("epoch"),
        "split_name": split_name,
        "device": str(device_object),
        "tta": tta,
        "class_weights": (
            None if class_weights is None else [float(w) for w in class_weights]
        ),
    }
    artifact_paths = save_evaluation_results(
        evaluation,
        resolved_root,
        resolved_run_name,
        metadata=metadata,
        overwrite=overwrite,
        bootstrap=bootstrap,
    )
    probability_path = artifact_paths["metrics"].parent / "probabilities.csv"
    probabilities.to_csv(probability_path, index=False)
    artifact_paths["probabilities"] = probability_path
    if class_weights is not None:
        artifact_paths["class_weights"] = save_class_weights(
            class_weights, artifact_paths["metrics"].parent / "class_weights.json"
        )

    return InferenceResult(
        config=config,
        split_name=split_name,
        evaluation=evaluation,
        bootstrap=bootstrap,
        probabilities=probabilities,
        artifact_paths=artifact_paths,
        class_weights=class_weights,
    )
