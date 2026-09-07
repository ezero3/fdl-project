"""Wire one validated experiment config into a complete training run.

``scripts/train.py`` is a thin argument parser over this module, so the same
run can be launched from a Colab notebook cell without shelling out.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from fdl_project.config.loader import dump_experiment_config
from fdl_project.config.registry import build_model
from fdl_project.config.schema import ExperimentConfig
from fdl_project.constants import class_encoding_metadata
from fdl_project.data.augmentation import build_augmentation
from fdl_project.data.datasets import (
    WM811KDataset,
    create_dataloader,
    create_split_dataset,
    load_wm811k_dataframe,
)
from fdl_project.data.imbalance import (
    build_training_loss,
    compute_class_counts,
    create_imbalance_training_dataloader,
    imbalance_metadata,
)
from fdl_project.evaluation import (
    BootstrapResult,
    EvaluationResult,
    bootstrap_evaluation,
    evaluate_model,
    save_evaluation_results,
)
from fdl_project.models.baseline_cnn import count_trainable_parameters
from fdl_project.training.callbacks import build_callbacks
from fdl_project.training.checkpoint import CheckpointManager, load_checkpoint
from fdl_project.training.loop import FitResult, fit_model, resolve_device
from fdl_project.training.optim import (
    build_learning_rate_scheduler,
    build_optimizer,
    describe_parameter_groups,
)
from fdl_project.training.seed import seed_everything

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunResult:
    """Everything one finished training run produced."""

    config: ExperimentConfig
    fit: FitResult
    validation: EvaluationResult
    bootstrap: BootstrapResult
    artifact_paths: dict[str, Path]
    parameter_groups: list[dict[str, Any]]


MINIMUM_SAMPLES_PER_CLASS = 4


def _subset_dataset(dataset: WM811KDataset, limit: int, seed: int) -> WM811KDataset:
    """Take a stratified random slice, for smoke tests only.

    Stratified rather than uniform because a uniform draw of a few thousand
    rows loses ``Near-full`` entirely -- it is 0.1% of the data -- and the
    imbalance policy rejects a training split with a missing class.
    """

    generator = np.random.default_rng(seed)
    targets = np.asarray(dataset.target_indices)
    total = min(limit, len(dataset))

    by_class = {
        class_index: generator.permutation(np.flatnonzero(targets == class_index))
        for class_index in np.unique(targets)
    }
    quotas = {
        class_index: min(len(positions), MINIMUM_SAMPLES_PER_CLASS)
        for class_index, positions in by_class.items()
    }
    remaining = max(0, total - sum(quotas.values()))
    if remaining:
        shares = np.array([len(by_class[key]) for key in by_class], dtype=np.float64)
        shares /= shares.sum()
        for position, class_index in enumerate(by_class):
            available = len(by_class[class_index]) - quotas[class_index]
            quotas[class_index] += min(available, int(remaining * shares[position]))

    selected = np.concatenate(
        [by_class[class_index][: quotas[class_index]] for class_index in by_class]
    )
    return dataset.select(selected)


def build_datasets(
    config: ExperimentConfig, dataframe: pd.DataFrame | None = None
) -> tuple[WM811KDataset, WM811KDataset]:
    """Load the frozen train and validation splits under this config's geometry."""

    if dataframe is None:
        dataframe = load_wm811k_dataframe(config.data.dataset_path)
    datasets = []
    for split_name in ("train", "validation"):
        dataset = create_split_dataset(
            dataframe,
            config.data.split_directory,
            split_name,
            preprocessing_config=config.data.preprocessing,
            cache_maps=config.data.cache,
            augmentation=(
                build_augmentation(
                    config.data.augmentation.name,
                    probability=config.data.augmentation.probability,
                    class_probabilities=dict(
                        config.data.augmentation.class_probabilities
                    ),
                    **dict(config.data.augmentation.kwargs),
                )
                if split_name == "train"
                else None
            ),
        )
        if config.data.subset is not None:
            dataset = _subset_dataset(dataset, config.data.subset, config.seed)
        datasets.append(dataset)
    return datasets[0], datasets[1]


def build_dataloaders(
    config: ExperimentConfig,
    train_dataset: WM811KDataset,
    validation_dataset: WM811KDataset,
) -> tuple[DataLoader, DataLoader]:
    """Training loader with the configured imbalance policy; validation untouched."""

    train_loader = create_imbalance_training_dataloader(
        train_dataset,
        batch_size=config.trainer.batch_size,
        config=config.imbalance,
        seed=config.seed,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    validation_loader = create_dataloader(
        validation_dataset,
        batch_size=config.trainer.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        seed=config.seed,
    )
    return train_loader, validation_loader


def run_experiment(
    config: ExperimentConfig,
    *,
    overwrite: bool = False,
    resume: str | None = None,
    device: str | None = None,
    bootstrap_resamples: int = 1000,
) -> RunResult:
    """Train, select on validation, and write the standard artifacts.

    The test split is never opened here. It is evaluated exactly once, at the
    very end of the project, through ``scripts/inference.py``.
    """

    seed_everything(config.seed)
    device_object = resolve_device(device or config.trainer.device)

    train_dataset, validation_dataset = build_datasets(config)
    train_loader, validation_loader = build_dataloaders(
        config, train_dataset, validation_dataset
    )
    class_counts = compute_class_counts(train_dataset.target_indices)
    criterion = build_training_loss(config.imbalance, class_counts)

    model = build_model(config.model.name, **dict(config.model.kwargs))
    optimizer = build_optimizer(model, config.optimizer)
    scheduler = build_learning_rate_scheduler(
        optimizer,
        config.scheduler,
        max_epochs=config.trainer.max_epochs,
        steps_per_epoch=len(train_loader),
    )
    parameter_groups = describe_parameter_groups(optimizer)

    manager: CheckpointManager | None = None
    if config.checkpoint.enabled:
        manager = CheckpointManager(
            config.checkpoint_directory,
            keep_last=config.checkpoint.keep_last,
            save_best=config.checkpoint.save_best,
        )

    resume_mode = resume or config.checkpoint.resume
    resume_state = None
    if manager is not None and resume_mode == "auto":
        model.to(device_object)
        resume_state = manager.resume(
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device_object,
        )
        if resume_state is not None:
            logger.info(
                "Resuming %s from %s at epoch %d (best so far %s at epoch %d).",
                config.name,
                resume_state.path,
                resume_state.next_epoch,
                resume_state.best_metric,
                resume_state.best_epoch,
            )

    callbacks = build_callbacks(config, checkpoint_manager=manager)
    fit = fit_model(
        model,
        train_loader,
        validation_loader,
        criterion,
        config.trainer,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_interval=config.scheduler.interval,
        device=device_object,
        callbacks=callbacks,
        sampler_seed=config.seed,
        resume=resume_state,
    )

    # A run resumed past its best epoch never sees that epoch, so fit reports
    # the resumed baseline with the *final* weights. The real best weights are
    # in best.pt; recover them and re-score, or the run would be written up
    # with the wrong model and the wrong metrics.
    best_state_dict = fit.best_state_dict
    validation = fit.best_validation
    start_epoch = 1 if resume_state is None else resume_state.next_epoch
    if fit.best_epoch < start_epoch and manager is not None and manager.best_path.is_file():
        payload = load_checkpoint(manager.best_path, map_location=device_object)
        best_state_dict = payload["model_state"]
        model.load_state_dict(best_state_dict)
        validation = evaluate_model(
            model,
            validation_loader,
            device=device_object,
            criterion=nn.CrossEntropyLoss(),
            split_name="validation",
        )
        logger.info(
            "No epoch after %d improved; reporting epoch %d recovered from %s.",
            start_epoch,
            fit.best_epoch,
            manager.best_path,
        )

    bootstrap = bootstrap_evaluation(
        validation, num_resamples=bootstrap_resamples, seed=config.seed
    )
    metadata = {
        "config": config.to_dict(),
        "best_epoch": fit.best_epoch,
        "last_epoch": fit.last_epoch,
        "stopped_early": fit.stopped_early,
        "num_trainable_parameters": count_trainable_parameters(model),
        "parameter_groups": parameter_groups,
        "device": str(device_object),
        **imbalance_metadata(config.imbalance, class_counts),
    }
    artifact_paths = save_evaluation_results(
        validation,
        config.logging.output_root,
        config.name,
        metadata=metadata,
        overwrite=overwrite,
        bootstrap=bootstrap,
    )
    run_directory = artifact_paths["metrics"].parent
    fit.history.to_csv(run_directory / "history.csv", index=False)
    dump_experiment_config(config, run_directory / "config.yaml")
    # Written in the checkpoint payload shape so scripts/inference.py can load
    # it exactly like any epoch checkpoint, class encoding included.
    torch.save(
        {
            "schema_version": 1,
            "epoch": fit.best_epoch,
            "model_state": best_state_dict,
            "model_class": type(model).__name__,
            "best_metric": fit.best_metric,
            "best_epoch": fit.best_epoch,
            "config": config.to_dict(),
            **class_encoding_metadata(),
        },
        run_directory / "best_model.pt",
    )

    return RunResult(
        config=config,
        fit=fit,
        validation=validation,
        bootstrap=bootstrap,
        artifact_paths=artifact_paths,
        parameter_groups=parameter_groups,
    )
