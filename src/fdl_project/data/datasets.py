"""WM-811K supervised Dataset and reproducible DataLoader construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from fdl_project.constants import encode_label
from fdl_project.data.preprocessing import (
    DEFAULT_PREPROCESSING_CONFIG,
    PreprocessingConfig,
    WaferMapPreprocessor,
)
from fdl_project.training.seed import build_dataloader_generator, seed_worker

SplitName = Literal["train", "validation", "test"]
_SPLIT_NAMES = frozenset({"train", "validation", "test"})


def normalize_scalar_label(value: Any) -> str | None:
    """Normalize the scalar-like objects used by the original pickle."""

    values = np.asarray(value, dtype=object).reshape(-1)
    if values.size == 0:
        return None
    scalar = values[0]
    if scalar is None:
        return None
    missing = pd.isna(scalar)
    if isinstance(missing, (bool, np.bool_)) and missing:
        return None
    if scalar == 0:
        return None
    text = str(scalar).strip()
    return text if text and text != "0" else None


def load_wm811k_dataframe(dataset_path: str | Path) -> pd.DataFrame:
    """Load and validate the original WM811K.pkl table once per process."""

    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"WM-811K dataset not found: {path}.")
    dataframe = pd.read_pickle(path)
    required_columns = {"waferMap", "failureType"}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)!r}."
        )
    if not dataframe.index.is_unique:
        raise ValueError("WM-811K DataFrame index must be unique.")
    return dataframe


def load_split_indices(
    split_directory: str | Path, split_name: SplitName
) -> np.ndarray:
    """Load one immutable supervised split produced by task 02."""

    if split_name not in _SPLIT_NAMES:
        raise ValueError(
            f"Unknown split {split_name!r}; expected train, validation, or test."
        )
    path = Path(split_directory) / f"{split_name}_indices.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Split indices not found: {path}.")
    indices = np.load(path, allow_pickle=False)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError(
            f"Split indices must be a one-dimensional integer array: {path}."
        )
    indices = indices.astype(np.int64, copy=False)
    if len(indices) == 0 or len(np.unique(indices)) != len(indices):
        raise ValueError(f"Split indices must be non-empty and unique: {path}.")
    indices.setflags(write=False)
    return indices


class WM811KDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Supervised wafer maps addressed by the persisted source-row indices."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        row_indices: np.ndarray,
        *,
        split_name: SplitName,
        preprocessing_config: PreprocessingConfig = DEFAULT_PREPROCESSING_CONFIG,
    ) -> None:
        if split_name not in _SPLIT_NAMES:
            raise ValueError(f"Unknown split {split_name!r}.")
        if not dataframe.index.is_unique:
            raise ValueError("WM-811K DataFrame index must be unique.")
        missing_columns = {"waferMap", "failureType"} - set(dataframe.columns)
        if missing_columns:
            raise ValueError(
                f"Dataset is missing required columns: {sorted(missing_columns)!r}."
            )

        indices = np.asarray(row_indices)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("row_indices must be a one-dimensional integer array.")
        indices = indices.astype(np.int64, copy=False)
        if len(indices) == 0 or len(np.unique(indices)) != len(indices):
            raise ValueError("row_indices must be non-empty and unique.")
        missing_indices = indices[~pd.Index(indices).isin(dataframe.index)]
        if len(missing_indices):
            raise ValueError(
                f"row_indices are absent from the DataFrame: {missing_indices[:5]!r}."
            )

        labels = dataframe.loc[indices, "failureType"]
        normalized_labels = [normalize_scalar_label(value) for value in labels]
        unlabeled_positions = [
            int(indices[position])
            for position, label in enumerate(normalized_labels)
            if label is None
        ]
        if unlabeled_positions:
            raise ValueError(
                "Supervised splits must not contain unlabeled rows; found source indices "
                f"{unlabeled_positions[:5]!r}."
            )

        self.dataframe = dataframe
        self.row_indices = indices.copy()
        self.row_indices.setflags(write=False)
        self.target_indices = np.asarray(
            [encode_label(label) for label in normalized_labels], dtype=np.int64
        )
        self.target_indices.setflags(write=False)
        self.split_name = split_name
        self.preprocessing_config = preprocessing_config
        self.preprocessor = WaferMapPreprocessor(preprocessing_config)

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, position: int) -> tuple[Tensor, Tensor, Tensor]:
        row_index = int(self.row_indices[position])
        wafer_map = self.dataframe.at[row_index, "waferMap"]
        image = self.preprocessor(wafer_map)
        target = torch.tensor(int(self.target_indices[position]), dtype=torch.long)
        source_index = torch.tensor(row_index, dtype=torch.long)
        return image, target, source_index


def create_split_dataset(
    dataframe: pd.DataFrame,
    split_directory: str | Path,
    split_name: SplitName,
    *,
    preprocessing_config: PreprocessingConfig = DEFAULT_PREPROCESSING_CONFIG,
) -> WM811KDataset:
    """Build a supervised Dataset from one persisted split."""

    return WM811KDataset(
        dataframe,
        load_split_indices(split_directory, split_name),
        split_name=split_name,
        preprocessing_config=preprocessing_config,
    )


def create_dataloader(
    dataset: WM811KDataset,
    *,
    batch_size: int,
    shuffle: bool | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 86,
) -> DataLoader[tuple[Tensor, Tensor, Tensor]]:
    """Create a split-safe DataLoader with deterministic train shuffling."""

    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer.")
    if (
        isinstance(num_workers, bool)
        or not isinstance(num_workers, int)
        or num_workers < 0
    ):
        raise ValueError("num_workers must be a non-negative integer.")

    should_shuffle = dataset.split_name == "train" if shuffle is None else shuffle
    if dataset.split_name in {"validation", "test"} and should_shuffle:
        raise ValueError("Validation and test DataLoaders must use shuffle=False.")

    # The generator drives shuffling and, together with seed_worker, the
    # per-worker random streams; workers that share one seed would repeat the
    # same augmented views every epoch.
    generator = build_dataloader_generator(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=should_shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
        generator=generator,
        worker_init_fn=seed_worker if num_workers > 0 else None,
    )
