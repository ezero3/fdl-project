"""End-to-end train/validation experiment for task 05."""

from __future__ import annotations

import gc
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from matplotlib.figure import Figure
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, default_collate

from fdl_project.constants import CLASS_NAMES
from fdl_project.data.datasets import create_split_dataset, load_wm811k_dataframe
from fdl_project.evaluation import evaluate_model, save_evaluation_results
from fdl_project.data.imbalance import (
    PRESET_IMBALANCE_CONFIGS,
    ImbalanceConfig,
    build_training_loss,
    build_weighted_sampler,
    compute_class_counts,
    compute_class_weights,
)
from fdl_project.models.baseline_cnn import BaselineCNN, count_trainable_parameters
from fdl_project.data.preprocessing import DEFAULT_PREPROCESSING_CONFIG
from fdl_project.training.loop import TrainingConfig, fit_model, set_reproducible_seed


@dataclass(frozen=True)
class ImbalanceExperimentConfig:
    """Fixed screening and multi-seed confirmation protocol."""

    screening_seed: int = 86
    confirmation_seeds: tuple[int, ...] = (87, 88)
    top_nonbaseline_strategies: int = 2
    torch_threads: int = 6
    device: str = "cpu"
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self) -> None:
        all_seeds = (self.screening_seed, *self.confirmation_seeds)
        if any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in all_seeds
        ):
            raise ValueError("All experiment seeds must be non-negative integers.")
        if len(set(all_seeds)) != len(all_seeds):
            raise ValueError("Screening and confirmation seeds must be unique.")
        if not self.confirmation_seeds:
            raise ValueError("At least one confirmation seed is required.")
        if (
            isinstance(self.top_nonbaseline_strategies, bool)
            or not isinstance(self.top_nonbaseline_strategies, int)
            or self.top_nonbaseline_strategies <= 0
        ):
            raise ValueError("top_nonbaseline_strategies must be a positive integer.")
        if (
            isinstance(self.torch_threads, bool)
            or not isinstance(self.torch_threads, int)
            or self.torch_threads <= 0
        ):
            raise ValueError("torch_threads must be a positive integer.")
        if self.device != "cpu":
            raise ValueError(
                "The committed task-05 protocol is fixed to CPU for reproducibility."
            )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["confirmation_seeds"] = list(self.confirmation_seeds)
        return payload


@dataclass(frozen=True)
class ImbalanceExperimentResult:
    """Selected strategy and paths produced by one complete experiment."""

    selected_strategy: str
    representative_seed: int
    screened_strategies: tuple[str, ...]
    confirmed_strategies: tuple[str, ...]
    artifact_paths: Mapping[str, Path]


def categorical_one_hot_collate(
    batch: Sequence[tuple[Tensor, Tensor, Tensor]],
) -> tuple[Tensor, Tensor, Tensor]:
    """Vectorize cached categorical maps into the task-04 one-hot contract."""

    categorical, targets, row_indices = default_collate(batch)
    if categorical.ndim != 3 or categorical.dtype != torch.uint8:
        raise ValueError(
            "Cached wafer maps must be uint8 tensors shaped (batch, H, W)."
        )
    if torch.any(categorical >= 3):
        raise ValueError("Cached wafer maps must contain only states 0, 1, and 2.")
    images = (
        F.one_hot(categorical.to(dtype=torch.long), num_classes=3)
        .permute(0, 3, 1, 2)
        .to(dtype=torch.float32)
    )
    return images, targets, row_indices


def materialize_categorical_split(dataset: Any) -> TensorDataset:
    """Preprocess one split once in RAM without changing samples or labels."""

    height, width = dataset.preprocessing_config.target_size
    maps = torch.empty((len(dataset), height, width), dtype=torch.uint8)
    for position, row_index in enumerate(dataset.row_indices):
        wafer_map = dataset.dataframe.at[int(row_index), "waferMap"]
        maps[position] = dataset.preprocessor.transform_categories(wafer_map).to(
            dtype=torch.uint8
        )
    targets = torch.tensor(dataset.target_indices.copy(), dtype=torch.long)
    row_indices = torch.tensor(dataset.row_indices.copy(), dtype=torch.long)
    return TensorDataset(maps, targets, row_indices)


def _dataset_targets(dataset: TensorDataset) -> Tensor:
    targets = dataset.tensors[1]
    if targets.ndim != 1 or targets.dtype != torch.long:
        raise ValueError("Materialized targets must be a one-dimensional long tensor.")
    return targets


def create_experiment_dataloader(
    dataset: TensorDataset,
    *,
    batch_size: int,
    seed: int,
    imbalance_config: ImbalanceConfig | None,
    class_counts: Tensor | None = None,
) -> DataLoader:
    """Create a natural validation loader or one configured training loader."""

    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer.")
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    shuffle = False
    if imbalance_config is not None:
        if class_counts is None:
            raise ValueError("Training loaders require class_counts.")
        sampler = build_weighted_sampler(
            imbalance_config,
            _dataset_targets(dataset),
            class_counts,
            seed=seed,
        )
        shuffle = sampler is None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=0,
        drop_last=False,
        generator=generator if sampler is None else None,
        collate_fn=categorical_one_hot_collate,
    )


def _row_index_hash(dataset: TensorDataset) -> str:
    row_indices = dataset.tensors[2].numpy().astype("<i8", copy=False)
    return hashlib.sha256(row_indices.tobytes()).hexdigest()


def _strategy_weight_rows(
    strategies: Mapping[str, ImbalanceConfig], class_counts: Tensor
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for strategy_name, strategy in strategies.items():
        method = (
            strategy.loss_weighting
            if strategy.loss_weighting != "none"
            else strategy.sampling_weighting
        )
        weights = compute_class_weights(
            class_counts,
            method,
            effective_number_beta=strategy.effective_number_beta,
        )
        if strategy.sampling_weighting != "none":
            sampling_mass = class_counts.to(dtype=torch.float64) * weights
            expected_shares = sampling_mass / sampling_mass.sum()
        else:
            expected_shares = class_counts / class_counts.sum()
        for class_index, class_name in enumerate(CLASS_NAMES):
            rows.append(
                {
                    "strategy": strategy_name,
                    "class_index": class_index,
                    "class_name": class_name,
                    "train_count": int(class_counts[class_index]),
                    "class_weight": float(weights[class_index]),
                    "applied_to_loss": strategy.loss_weighting != "none",
                    "applied_to_sampler": strategy.sampling_weighting != "none",
                    "expected_sampling_share": float(expected_shares[class_index]),
                    "expected_epoch_samples": float(
                        expected_shares[class_index] * class_counts.sum()
                    ),
                }
            )
    return rows


def _save_figures(
    output_directory: Path,
    class_weights: pd.DataFrame,
    run_summary: pd.DataFrame,
    strategy_summary: pd.DataFrame,
    training_history: pd.DataFrame,
) -> dict[str, Path]:
    class_path = output_directory / "class_distribution_and_weights.png"
    comparison_path = output_directory / "strategy_comparison.png"
    history_path = output_directory / "training_curves.png"

    figure = Figure(figsize=(13, 6), constrained_layout=True)
    count_axes, weight_axes = figure.subplots(1, 2)
    baseline_rows = class_weights.loc[class_weights["strategy"] == "unweighted_ce"]
    count_axes.bar(baseline_rows["class_name"], baseline_rows["train_count"])
    count_axes.set_yscale("log")
    count_axes.set(title="Training class counts", ylabel="Count (log scale)")
    count_axes.tick_params(axis="x", rotation=45)
    weighted_rows = class_weights.loc[
        class_weights["applied_to_loss"] | class_weights["applied_to_sampler"]
    ]
    for strategy_name, rows in weighted_rows.groupby("strategy", sort=False):
        if strategy_name != "unweighted_ce":
            weight_axes.plot(
                rows["class_name"],
                rows["class_weight"],
                marker="o",
                label=strategy_name,
            )
    weight_axes.set_yscale("log")
    weight_axes.set(title="Normalized class weights", ylabel="Weight (log scale)")
    weight_axes.tick_params(axis="x", rotation=45)
    weight_axes.legend(fontsize=8)
    figure.savefig(class_path, dpi=160)
    figure.clear()

    figure = Figure(figsize=(11, 6), constrained_layout=True)
    axes = figure.subplots()
    screening = run_summary.loc[run_summary["phase"] == "screening"]
    axes.scatter(
        screening["strategy"],
        screening["macro_f1"],
        marker="x",
        s=80,
        label="screening seed 86",
    )
    axes.errorbar(
        strategy_summary["strategy"],
        strategy_summary["macro_f1_mean"],
        yerr=strategy_summary["macro_f1_std"],
        fmt="o",
        capsize=5,
        label="confirmed mean ± std",
    )
    axes.set(
        title="Validation macro-F1 by imbalance strategy",
        ylabel="Macro-F1",
        xlabel="Strategy",
        ylim=(0, 1),
    )
    axes.tick_params(axis="x", rotation=30)
    axes.legend()
    figure.savefig(comparison_path, dpi=160)
    figure.clear()

    figure = Figure(figsize=(11, 6), constrained_layout=True)
    axes = figure.subplots()
    for (strategy, seed), rows in training_history.groupby(["strategy", "seed"]):
        axes.plot(
            rows["epoch"],
            rows["validation_macro_f1"],
            marker="o",
            alpha=0.75,
            label=f"{strategy} / {seed}",
        )
    axes.set(
        title="Validation macro-F1 during training",
        xlabel="Epoch",
        ylabel="Macro-F1",
        ylim=(0, 1),
    )
    axes.legend(fontsize=7, ncols=2)
    figure.savefig(history_path, dpi=160)
    figure.clear()
    return {
        "class_distribution_figure": class_path,
        "strategy_comparison_figure": comparison_path,
        "training_curves_figure": history_path,
    }


def run_class_imbalance_experiment(
    dataset_path: str | Path,
    split_directory: str | Path,
    output_directory: str | Path,
    *,
    config: ImbalanceExperimentConfig | None = None,
    strategies: Mapping[str, ImbalanceConfig] = PRESET_IMBALANCE_CONFIGS,
    overwrite: bool = False,
    progress: Callable[[str], None] | None = print,
) -> ImbalanceExperimentResult:
    """Screen all candidates, confirm the best two, and select by mean macro-F1."""

    config = config or ImbalanceExperimentConfig()
    if "unweighted_ce" not in strategies:
        raise ValueError("Strategies must include the unweighted_ce baseline.")
    if len(strategies) <= config.top_nonbaseline_strategies:
        raise ValueError("Not enough non-baseline candidates for the requested top-k.")
    if any(name != strategy.name for name, strategy in strategies.items()):
        raise ValueError("Strategy mapping keys must match ImbalanceConfig.name.")

    output_path = Path(output_directory)
    known_outputs = [
        output_path / "run_summary.csv",
        output_path / "strategy_summary.csv",
        output_path / "per_class_results.csv",
        output_path / "training_history.csv",
        output_path / "class_weights.csv",
        output_path / "experiment_metadata.json",
    ]
    if not overwrite and any(path.exists() for path in known_outputs):
        raise FileExistsError(
            f"Task-05 artifacts already exist under {output_path}; pass overwrite=True."
        )
    output_path.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(config.torch_threads)
    if progress is not None:
        progress(
            "Loading WM-811K and materializing train/validation categories once..."
        )
    dataframe = load_wm811k_dataframe(dataset_path)
    train_source = create_split_dataset(dataframe, split_directory, "train")
    validation_source = create_split_dataset(dataframe, split_directory, "validation")
    if train_source.preprocessing_config != DEFAULT_PREPROCESSING_CONFIG:
        raise ValueError(
            "Experiment must use the shared task-04 preprocessing default."
        )
    train_dataset = materialize_categorical_split(train_source)
    validation_dataset = materialize_categorical_split(validation_source)
    del train_source, validation_source, dataframe
    gc.collect()

    train_targets = _dataset_targets(train_dataset)
    class_counts = compute_class_counts(train_targets)
    validation_loader = create_experiment_dataloader(
        validation_dataset,
        batch_size=config.training.batch_size,
        seed=config.screening_seed,
        imbalance_config=None,
    )

    run_rows: list[dict[str, Any]] = []
    per_class_frames: list[pd.DataFrame] = []
    history_frames: list[pd.DataFrame] = []
    state_dicts: dict[tuple[str, int], dict[str, Tensor]] = {}

    def run_one(strategy_name: str, seed: int, phase: str) -> None:
        strategy = strategies[strategy_name]
        if progress is not None:
            progress(f"Starting {phase}: {strategy_name}, seed {seed}.")
        set_reproducible_seed(seed)
        model = BaselineCNN()
        criterion = build_training_loss(strategy, class_counts)
        train_loader = create_experiment_dataloader(
            train_dataset,
            batch_size=config.training.batch_size,
            seed=seed,
            imbalance_config=strategy,
            class_counts=class_counts,
        )

        def report_epoch(metrics: dict[str, float | int]) -> None:
            if progress is not None:
                progress(
                    f"  epoch {metrics['epoch']}: "
                    f"train_loss={metrics['train_loss']:.4f}, "
                    f"val_macro_f1={metrics['validation_macro_f1']:.4f}"
                )

        fitted = fit_model(
            model,
            train_loader,
            validation_loader,
            criterion,
            config.training,
            device=config.device,
            epoch_callback=report_epoch,
        )
        validation = fitted.best_validation
        state_dicts[(strategy_name, seed)] = fitted.best_state_dict
        run_rows.append(
            {
                "phase": phase,
                "strategy": strategy_name,
                "seed": seed,
                "epochs_ran": len(fitted.history),
                "best_epoch": fitted.best_epoch,
                "stopped_early": fitted.stopped_early,
                "macro_f1": float(validation.metrics["macro_f1"]),
                "balanced_accuracy": float(validation.metrics["balanced_accuracy"]),
                "accuracy": float(validation.metrics["accuracy"]),
                "weighted_f1": float(validation.metrics["weighted_f1"]),
                "validation_loss": float(validation.metrics["mean_loss"]),
            }
        )
        per_class = validation.per_class_metrics.copy()
        per_class.insert(0, "seed", seed)
        per_class.insert(0, "strategy", strategy_name)
        per_class.insert(0, "phase", phase)
        per_class_frames.append(per_class)
        history = fitted.history.copy()
        history.insert(0, "seed", seed)
        history.insert(0, "strategy", strategy_name)
        history.insert(0, "phase", phase)
        history_frames.append(history)

    for strategy_name in strategies:
        run_one(strategy_name, config.screening_seed, "screening")

    screening_frame = pd.DataFrame(run_rows)
    ranked_nonbaseline = screening_frame.loc[
        screening_frame["strategy"] != "unweighted_ce"
    ].sort_values(["macro_f1", "balanced_accuracy"], ascending=False)
    top_strategies = tuple(
        ranked_nonbaseline.head(config.top_nonbaseline_strategies)["strategy"]
    )
    confirmed_strategies = ("unweighted_ce", *top_strategies)
    if progress is not None:
        progress(f"Confirming across seeds: {confirmed_strategies!r}.")
    for seed in config.confirmation_seeds:
        for strategy_name in confirmed_strategies:
            run_one(strategy_name, seed, "confirmation")

    run_summary = pd.DataFrame(run_rows)
    confirmation_runs = run_summary.loc[
        run_summary["strategy"].isin(confirmed_strategies)
    ]
    strategy_summary = (
        confirmation_runs.groupby("strategy", sort=False)
        .agg(
            runs=("seed", "count"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            accuracy_mean=("accuracy", "mean"),
            weighted_f1_mean=("weighted_f1", "mean"),
        )
        .reset_index()
        .sort_values(
            ["macro_f1_mean", "balanced_accuracy_mean"],
            ascending=False,
            ignore_index=True,
        )
    )
    selected_strategy = str(strategy_summary.iloc[0]["strategy"])
    selected_mean = float(strategy_summary.iloc[0]["macro_f1_mean"])
    selected_runs = confirmation_runs.loc[
        confirmation_runs["strategy"] == selected_strategy
    ].copy()
    selected_runs["distance_from_mean"] = (
        selected_runs["macro_f1"] - selected_mean
    ).abs()
    representative_seed = int(
        selected_runs.sort_values(["distance_from_mean", "seed"]).iloc[0]["seed"]
    )

    representative_model = BaselineCNN()
    representative_model.load_state_dict(
        state_dicts[(selected_strategy, representative_seed)]
    )
    representative_validation = evaluate_model(
        representative_model,
        validation_loader,
        device=config.device,
        criterion=nn.CrossEntropyLoss(),
        split_name="validation",
    )
    selected_evaluation_paths = save_evaluation_results(
        representative_validation,
        output_path,
        "selected_validation",
        metadata={
            "strategy": selected_strategy,
            "representative_seed": representative_seed,
            "selection_metric": "mean_validation_macro_f1",
            "test_set_used": False,
        },
        overwrite=overwrite,
    )

    per_class_results = pd.concat(per_class_frames, ignore_index=True)
    training_history = pd.concat(history_frames, ignore_index=True)
    class_weights = pd.DataFrame(_strategy_weight_rows(strategies, class_counts))
    run_summary.to_csv(output_path / "run_summary.csv", index=False)
    strategy_summary.to_csv(output_path / "strategy_summary.csv", index=False)
    per_class_results.to_csv(output_path / "per_class_results.csv", index=False)
    training_history.to_csv(output_path / "training_history.csv", index=False)
    class_weights.to_csv(output_path / "class_weights.csv", index=False)
    figure_paths = _save_figures(
        output_path,
        class_weights,
        run_summary,
        strategy_summary,
        training_history,
    )

    model_for_count = BaselineCNN()
    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "selection_split": "validation",
        "test_set_used": False,
        "selection_metric": "mean_validation_macro_f1",
        "screened_strategies": list(strategies),
        "confirmed_strategies": list(confirmed_strategies),
        "selected_strategy": selected_strategy,
        "representative_seed": representative_seed,
        "experiment_config": config.to_dict(),
        "preprocessing_config": DEFAULT_PREPROCESSING_CONFIG.to_dict(),
        "class_names": list(CLASS_NAMES),
        "train_class_counts": {
            class_name: int(class_counts[index])
            for index, class_name in enumerate(CLASS_NAMES)
        },
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "train_row_indices_sha256": _row_index_hash(train_dataset),
        "validation_row_indices_sha256": _row_index_hash(validation_dataset),
        "baseline_model": "BaselineCNN",
        "trainable_parameters": count_trainable_parameters(model_for_count),
        "strategies": {
            name: strategy.to_dict() for name, strategy in strategies.items()
        },
    }
    metadata_path = output_path / "experiment_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    artifact_paths: dict[str, Path] = {
        "run_summary": output_path / "run_summary.csv",
        "strategy_summary": output_path / "strategy_summary.csv",
        "per_class_results": output_path / "per_class_results.csv",
        "training_history": output_path / "training_history.csv",
        "class_weights": output_path / "class_weights.csv",
        "metadata": metadata_path,
        **figure_paths,
        **{
            f"selected_{name}": path for name, path in selected_evaluation_paths.items()
        },
    }
    if progress is not None:
        progress(
            f"Selected {selected_strategy} with representative seed "
            f"{representative_seed}."
        )
    return ImbalanceExperimentResult(
        selected_strategy=selected_strategy,
        representative_seed=representative_seed,
        screened_strategies=tuple(strategies),
        confirmed_strategies=confirmed_strategies,
        artifact_paths=artifact_paths,
    )
