"""Tests for resumable checkpointing.

The scenario these protect is a Colab session dropping mid-run: the second
half of a resumed run must be indistinguishable from an uninterrupted one.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fdl_project.config.schema import EarlyStoppingConfig, TrainerConfig
from fdl_project.training.checkpoint import (
    BEST_CHECKPOINT_NAME,
    LAST_CHECKPOINT_NAME,
    CheckpointManager,
    ResumeState,
    load_checkpoint,
)
from fdl_project.training.loop import fit_model
from fdl_project.training.seed import seed_everything

NUM_CLASSES = 9


def _model(seed: int = 86) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Flatten(), nn.Linear(12, NUM_CLASSES))


def _loader(seed: int = 86) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.rand((36, 12), generator=generator)
    targets = torch.arange(36) % NUM_CLASSES
    row_indices = torch.arange(36)
    return DataLoader(
        TensorDataset(inputs, targets, row_indices), batch_size=12, shuffle=False
    )


def _trainer_config(max_epochs: int, patience: int = 99) -> TrainerConfig:
    return TrainerConfig(
        max_epochs=max_epochs,
        batch_size=12,
        amp=False,
        device="cpu",
        early_stopping=EarlyStoppingConfig(patience=patience, min_epochs=1),
    )


def _fit(
    max_epochs: int,
    manager: CheckpointManager | None,
    patience: int = 99,
    **resume_state,
):
    from fdl_project.training.callbacks import CheckpointCallback

    seed_everything(86)
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    manager_to_resume_from = resume_state.pop("resume_from", None)
    if manager_to_resume_from is not None:
        resume_state["resume"] = manager_to_resume_from.resume(
            model, optimizer=optimizer
        )
    callbacks = [CheckpointCallback(manager)] if manager is not None else []
    return fit_model(
        model,
        _loader(),
        _loader(),
        nn.CrossEntropyLoss(),
        _trainer_config(max_epochs, patience),
        optimizer=optimizer,
        callbacks=callbacks,
        **resume_state,
    )


def test_a_saved_checkpoint_records_the_whole_run_state(tmp_path) -> None:
    manager = CheckpointManager(tmp_path)
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    manager.save(
        epoch=3,
        model=model,
        optimizer=optimizer,
        best_metric=0.5,
        best_epoch=2,
        history=[{"epoch": 1}, {"epoch": 2}, {"epoch": 3}],
        config={"name": "demo"},
        is_best=True,
    )
    payload = load_checkpoint(manager.last_path)

    assert payload["epoch"] == 3
    assert payload["best_epoch"] == 2
    assert payload["optimizer_state"] is not None
    assert payload["config"]["name"] == "demo"
    assert set(payload["random_state"]) >= {"python", "numpy", "torch"}
    assert payload["class_names"][0] == "Center"


def test_the_newest_epoch_is_the_latest_state_and_best_is_separate(tmp_path) -> None:
    manager = CheckpointManager(tmp_path)
    model = _model()
    manager.save(epoch=1, model=model, best_metric=0.1, is_best=True)
    manager.save(epoch=2, model=model, best_metric=0.1, is_best=False)

    assert load_checkpoint(manager.last_path)["epoch"] == 2
    assert load_checkpoint(manager.best_path)["epoch"] == 1
    # last.pt would be a byte-for-byte copy of the newest epoch file, which
    # costs a whole extra payload per epoch on Drive.
    assert not (tmp_path / LAST_CHECKPOINT_NAME).is_file()
    assert (tmp_path / BEST_CHECKPOINT_NAME).is_file()


def test_keeping_no_history_falls_back_to_last(tmp_path) -> None:
    manager = CheckpointManager(tmp_path, keep_last=0)
    manager.save(epoch=4, model=_model())

    assert manager.epoch_checkpoints() == []
    assert manager.last_path == tmp_path / LAST_CHECKPOINT_NAME
    assert load_checkpoint(manager.last_path)["epoch"] == 4


def test_the_rolling_window_keeps_only_the_most_recent_epochs(tmp_path) -> None:
    manager = CheckpointManager(tmp_path, keep_last=3)
    model = _model()
    for epoch in range(1, 8):
        manager.save(epoch=epoch, model=model)

    kept = [path.name for path in manager.epoch_checkpoints()]
    assert kept == ["epoch_0005.pt", "epoch_0006.pt", "epoch_0007.pt"]
    # resume reads the newest of them, so no duplicate copy is needed
    assert manager.last_path.name == "epoch_0007.pt"


def test_keep_last_zero_keeps_only_last_and_best(tmp_path) -> None:
    manager = CheckpointManager(tmp_path, keep_last=0)
    for epoch in range(1, 4):
        manager.save(epoch=epoch, model=_model(), is_best=True)

    assert manager.epoch_checkpoints() == []
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        BEST_CHECKPOINT_NAME,
        LAST_CHECKPOINT_NAME,
    ]


def test_saving_leaves_no_temporary_files_behind(tmp_path) -> None:
    """Writes are atomic so a disconnect cannot truncate last.pt."""

    manager = CheckpointManager(tmp_path)
    manager.save(epoch=1, model=_model(), is_best=True)

    assert not list(tmp_path.glob("*.tmp"))


def test_resuming_restores_the_optimizer_and_random_state(tmp_path) -> None:
    manager = CheckpointManager(tmp_path)
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    # give the optimizer real momentum buffers to restore
    model(torch.rand(4, 12)).sum().backward()
    optimizer.step()
    seed_everything(123)
    expected_draw = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    seed_everything(123)
    manager.save(epoch=5, model=model, optimizer=optimizer, best_metric=0.4, best_epoch=4)

    fresh_model = _model(seed=1)
    fresh_optimizer = torch.optim.SGD(fresh_model.parameters(), lr=0.1, momentum=0.9)
    state = manager.resume(fresh_model, optimizer=fresh_optimizer)

    assert state is not None
    assert state.next_epoch == 6
    assert state.best_epoch == 4
    assert state.best_metric == pytest.approx(0.4)
    for original, restored in zip(
        model.parameters(), fresh_model.parameters(), strict=True
    ):
        assert torch.equal(original, restored)
    assert fresh_optimizer.state_dict()["param_groups"][0]["momentum"] == 0.9
    assert (random.random(), float(np.random.rand()), float(torch.rand(1))) == (
        expected_draw
    )


def test_resuming_an_empty_directory_returns_nothing(tmp_path) -> None:
    assert CheckpointManager(tmp_path).resume(_model()) is None


def test_an_interrupted_run_resumes_to_the_same_result(tmp_path) -> None:
    """The acceptance test for item 4: a killed run must finish identically."""

    uninterrupted = _fit(6, None)

    manager = CheckpointManager(tmp_path / "interrupted")
    _fit(3, manager)  # the session "drops" after epoch 3
    resumed = _fit(6, manager, resume_from=manager)

    assert len(resumed.history) == len(uninterrupted.history) == 6
    assert resumed.best_epoch == uninterrupted.best_epoch
    assert resumed.best_metric == pytest.approx(uninterrupted.best_metric)
    np.testing.assert_allclose(
        resumed.history["validation_macro_f1"].to_numpy(),
        uninterrupted.history["validation_macro_f1"].to_numpy(),
        atol=1e-6,
    )


def test_resuming_inherits_the_early_stopping_counter() -> None:
    """A resumed run must not win back `patience` fresh epochs.

    Resume at epoch 6 from a run whose best epoch was 1 and whose best score
    is unbeatable: four epochs have already passed without improvement, so
    with patience=2 the run must stop after a single further epoch. If the
    counter restarted at zero it would run two more instead.
    """

    seed_everything(86)
    model = _model()
    result = fit_model(
        model,
        _loader(),
        _loader(),
        nn.CrossEntropyLoss(),
        _trainer_config(max_epochs=20, patience=2),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        resume=ResumeState(
            next_epoch=6,
            best_metric=1.0,  # perfect macro-F1: nothing can improve on it
            best_epoch=1,
            history=[{"epoch": index} for index in range(1, 6)],
        ),
    )

    assert result.stopped_early
    assert result.last_epoch == 6
    assert result.best_epoch == 1
    # Nothing beat the resumed baseline, so the loop reports it rather than
    # failing; the historical best weights live in best.pt.
    assert result.best_metric == pytest.approx(1.0)


def test_a_checkpoint_from_another_class_encoding_is_refused(tmp_path) -> None:
    path = tmp_path / "foreign.pt"
    torch.save(
        {"model_state": {}, "class_names": ["a", "b", "c"], "epoch": 1}, path
    )
    with pytest.raises(ValueError, match="class encoding"):
        load_checkpoint(path)


def test_a_file_that_is_not_a_checkpoint_is_refused(tmp_path) -> None:
    path = tmp_path / "not_a_checkpoint.pt"
    torch.save({"weights": 1}, path)
    with pytest.raises(ValueError, match="not a project checkpoint"):
        load_checkpoint(path)


def test_a_checkpoint_without_a_class_encoding_is_refused(tmp_path) -> None:
    path = tmp_path / "unlabelled.pt"
    torch.save({"model_state": {}, "epoch": 1}, path)
    with pytest.raises(ValueError, match="no class encoding"):
        load_checkpoint(path)


def test_a_missing_checkpoint_reports_its_path(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="missing.pt"):
        load_checkpoint(tmp_path / "missing.pt")


@pytest.mark.parametrize("keep_last", [-1, 1.5, True])
def test_invalid_retention_settings_are_rejected(tmp_path, keep_last: object) -> None:
    with pytest.raises(ValueError, match="keep_last"):
        CheckpointManager(tmp_path, keep_last=keep_last)  # type: ignore[arg-type]
