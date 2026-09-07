from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from fdl_project.constants import CLASS_NAMES, CLASS_TO_INDEX
from fdl_project.data.datasets import (
    WM811KDataset,
    create_dataloader,
    create_split_dataset,
    load_split_indices,
    load_wm811k_dataframe,
    normalize_scalar_label,
)
from fdl_project.data.preprocessing import PreprocessingConfig


def _toy_dataframe() -> pd.DataFrame:
    maps = [
        np.array([[0, 1], [1, 2]], dtype=np.uint8),
        np.array([[0, 1, 1], [2, 1, 0]], dtype=np.uint8),
        np.array([[1], [2], [1]], dtype=np.uint8),
        np.array([[0, 2], [1, 0]], dtype=np.uint8),
        np.array([[1, 1], [1, 2]], dtype=np.uint8),
    ]
    labels = [np.array([[name]]) for name in CLASS_NAMES[:5]]
    return pd.DataFrame(
        {"waferMap": maps, "failureType": labels},
        index=pd.Index([10, 20, 30, 40, 50], name="source_index"),
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (np.array([["Center"]]), "Center"),
        ("none", "none"),
        (np.array([]), None),
        (np.array([[0]]), None),
        (None, None),
    ],
)
def test_scalar_label_normalization(raw: object, expected: str | None) -> None:
    assert normalize_scalar_label(raw) == expected


def test_load_dataframe_validates_required_schema(tmp_path: Path) -> None:
    path = tmp_path / "WM811K.pkl"
    dataframe = _toy_dataframe()
    dataframe.to_pickle(path)

    loaded = load_wm811k_dataframe(path)
    assert loaded.index.tolist() == dataframe.index.tolist()

    pd.DataFrame({"waferMap": []}).to_pickle(path)
    with pytest.raises(ValueError, match="missing required columns"):
        load_wm811k_dataframe(path)


def test_split_indices_are_validated_and_read_only(tmp_path: Path) -> None:
    indices = np.array([10, 20, 30], dtype=np.int64)
    np.save(tmp_path / "train_indices.npy", indices)

    loaded = load_split_indices(tmp_path, "train")
    assert np.array_equal(loaded, indices)
    assert not loaded.flags.writeable
    with pytest.raises(ValueError):
        loaded[0] = 99


def test_dataset_returns_preprocessed_image_target_and_source_index() -> None:
    dataframe = _toy_dataframe()
    config = PreprocessingConfig(target_size=(8, 8), geometry="letterbox")
    dataset = WM811KDataset(
        dataframe,
        np.array([10, 30, 50]),
        split_name="validation",
        preprocessing_config=config,
    )

    image, target, row_index = dataset[1]
    assert image.shape == (3, 8, 8)
    assert image.dtype == torch.float32
    assert target.dtype == torch.long
    assert target.item() == CLASS_TO_INDEX[CLASS_NAMES[2]]
    assert row_index.item() == 30


def test_supervised_dataset_rejects_unlabeled_rows() -> None:
    dataframe = _toy_dataframe()
    dataframe.at[20, "failureType"] = np.array([[0]])

    with pytest.raises(ValueError, match="must not contain unlabeled"):
        WM811KDataset(dataframe, np.array([10, 20]), split_name="train")


def test_create_split_dataset_uses_persisted_indices(tmp_path: Path) -> None:
    dataframe = _toy_dataframe()
    np.save(tmp_path / "test_indices.npy", np.array([20, 40], dtype=np.int64))
    dataset = create_split_dataset(dataframe, tmp_path, "test")
    assert dataset.row_indices.tolist() == [20, 40]


def test_validation_loader_is_ordered_and_keeps_smaller_final_batch() -> None:
    dataset = WM811KDataset(
        _toy_dataframe(),
        np.array([10, 20, 30, 40, 50]),
        split_name="validation",
        preprocessing_config=PreprocessingConfig(target_size=(8, 8)),
    )
    dataloader = create_dataloader(dataset, batch_size=2)
    batches = list(dataloader)

    assert [len(batch[1]) for batch in batches] == [2, 2, 1]
    assert torch.cat([batch[2] for batch in batches]).tolist() == [10, 20, 30, 40, 50]

    with pytest.raises(ValueError, match="shuffle=False"):
        create_dataloader(dataset, batch_size=2, shuffle=True)


def test_training_loader_shuffle_is_reproducible() -> None:
    dataframe = _toy_dataframe()
    indices = np.array([10, 20, 30, 40, 50])
    dataset = WM811KDataset(dataframe, indices, split_name="train")

    first_loader = create_dataloader(dataset, batch_size=2, seed=86)
    second_loader = create_dataloader(dataset, batch_size=2, seed=86)
    first_order = torch.cat([batch[2] for batch in first_loader])
    second_order = torch.cat([batch[2] for batch in second_loader])

    assert torch.equal(first_order, second_order)
    assert sorted(first_order.tolist()) == sorted(indices.tolist())


def test_caching_yields_identical_tensors(tmp_path) -> None:
    """The cache must change speed and memory, never the data."""

    frame = _toy_dataframe()
    np.save(tmp_path / "train_indices.npy", frame.index.to_numpy())
    plain = create_split_dataset(frame, tmp_path, "train")
    cached = create_split_dataset(frame, tmp_path, "train", cache_maps=True)

    assert cached.cached_maps is not None
    assert len(cached.cached_maps) == len(cached)
    assert all(item.dtype == np.uint8 for item in cached.cached_maps)
    for position in range(len(plain)):
        expected = plain[position]
        actual = cached[position]
        assert torch.equal(expected[0], actual[0])
        assert torch.equal(expected[1], actual[1])
        assert torch.equal(expected[2], actual[2])


def test_caching_releases_the_source_table(tmp_path) -> None:
    """Holding the 2 GB pickle would make the copy pure overhead."""

    frame = _toy_dataframe()
    np.save(tmp_path / "train_indices.npy", frame.index.to_numpy())
    cached = create_split_dataset(frame, tmp_path, "train", cache_maps=True)

    assert cached.dataframe is None


def test_select_returns_a_subset_view(tmp_path) -> None:
    frame = _toy_dataframe()
    np.save(tmp_path / "train_indices.npy", frame.index.to_numpy())
    for cache_maps in (False, True):
        dataset = create_split_dataset(
            frame, tmp_path, "train", cache_maps=cache_maps
        )
        subset = dataset.select(np.array([2, 0]))

        assert len(subset) == 2
        # positions are sorted, so the subset keeps dataset order
        assert list(subset.row_indices) == [
            int(dataset.row_indices[0]),
            int(dataset.row_indices[2]),
        ]
        assert torch.equal(subset[1][0], dataset[2][0])
        assert subset.split_name == dataset.split_name


@pytest.mark.parametrize("positions", [[], [0, 0], [-1], [999]])
def test_select_rejects_invalid_positions(
    tmp_path, positions: list[int]
) -> None:
    frame = _toy_dataframe()
    np.save(tmp_path / "train_indices.npy", frame.index.to_numpy())
    dataset = create_split_dataset(frame, tmp_path, "train")
    with pytest.raises(ValueError):
        dataset.select(np.array(positions, dtype=np.int64))
