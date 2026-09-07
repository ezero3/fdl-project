"""Persisted comparison artifacts for one evaluated run."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from fdl_project.constants import NUM_CLASSES
from fdl_project.evaluation.bootstrap import BootstrapResult
from fdl_project.evaluation.metrics import EvaluationResult

_RUN_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: tuple[str, ...],
    output_path: Path,
    *,
    normalized: bool,
) -> None:
    figure = Figure(figsize=(10, 8), constrained_layout=True)
    axes = figure.subplots()
    image = axes.imshow(matrix, interpolation="nearest", cmap="Blues", vmin=0)
    figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04)
    axes.set(
        title="Normalized confusion matrix" if normalized else "Confusion matrix",
        xlabel="Predicted class",
        ylabel="True class",
        xticks=np.arange(NUM_CLASSES),
        yticks=np.arange(NUM_CLASSES),
        xticklabels=class_names,
        yticklabels=class_names,
    )
    axes.tick_params(axis="x", rotation=45)

    threshold = float(matrix.max()) / 2 if matrix.size and matrix.max() > 0 else 0
    for row_index in range(NUM_CLASSES):
        for column_index in range(NUM_CLASSES):
            value = matrix[row_index, column_index]
            text = f"{value:.2f}" if normalized else f"{int(value)}"
            axes.text(
                column_index,
                row_index,
                text,
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > threshold else "black",
            )
    figure.savefig(output_path, dpi=160)
    figure.clear()


def save_evaluation_results(
    result: EvaluationResult,
    output_root: str | Path,
    run_name: str,
    *,
    metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
    bootstrap: BootstrapResult | None = None,
) -> dict[str, Path]:
    """Persist the standard comparison artifacts for one experiment.

    Passing ``bootstrap`` additionally writes the resampled confidence
    intervals and records the aggregate ones inside ``metrics.json``.
    """

    if not _RUN_NAME_PATTERN.fullmatch(run_name):
        raise ValueError(
            "run_name must start with an alphanumeric character and contain only "
            "letters, numbers, '.', '_' or '-'."
        )

    experiment_directory = Path(output_root) / run_name
    artifact_paths = {
        "metrics": experiment_directory / "metrics.json",
        "per_class_metrics": experiment_directory / "per_class_metrics.csv",
        "predictions": experiment_directory / "predictions.csv",
        "confusion_matrix": experiment_directory / "confusion_matrix.png",
        "normalized_confusion_matrix": experiment_directory
        / "confusion_matrix_normalized.png",
    }
    if bootstrap is not None:
        artifact_paths["bootstrap_aggregate"] = (
            experiment_directory / "bootstrap_aggregate.csv"
        )
        artifact_paths["bootstrap_per_class"] = (
            experiment_directory / "bootstrap_per_class.csv"
        )
    existing_artifacts = [path for path in artifact_paths.values() if path.exists()]
    if existing_artifacts and not overwrite:
        raise FileExistsError(
            f"Evaluation artifacts already exist for run {run_name!r}; "
            "choose a new run name or pass overwrite=True."
        )
    experiment_directory.mkdir(parents=True, exist_ok=True)

    metadata_payload = dict(metadata or {})
    metrics_payload = {
        "schema_version": 1,
        "run_name": run_name,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "split_name": result.split_name,
        "model_class": result.model_class,
        "device": result.device,
        "class_names": list(result.class_names),
        "metrics": result.metrics,
        "metadata": metadata_payload,
    }
    if bootstrap is not None:
        metrics_payload["bootstrap"] = {
            "num_resamples": bootstrap.num_resamples,
            "confidence_level": bootstrap.confidence_level,
            "seed": bootstrap.seed,
            "intervals": {
                row.metric: {
                    "point_estimate": row.point_estimate,
                    "ci_lower": row.ci_lower,
                    "ci_upper": row.ci_upper,
                }
                for row in bootstrap.aggregate.itertuples()
            },
        }
    with artifact_paths["metrics"].open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    result.per_class_metrics.to_csv(artifact_paths["per_class_metrics"], index=False)
    result.predictions.to_csv(artifact_paths["predictions"], index=False)
    _plot_confusion_matrix(
        result.confusion_matrix,
        result.class_names,
        artifact_paths["confusion_matrix"],
        normalized=False,
    )
    _plot_confusion_matrix(
        result.normalized_confusion_matrix,
        result.class_names,
        artifact_paths["normalized_confusion_matrix"],
        normalized=True,
    )
    if bootstrap is not None:
        bootstrap.aggregate.to_csv(artifact_paths["bootstrap_aggregate"], index=False)
        bootstrap.per_class.to_csv(artifact_paths["bootstrap_per_class"], index=False)
    return artifact_paths
