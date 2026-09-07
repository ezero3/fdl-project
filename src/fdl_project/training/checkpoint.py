"""Resumable per-epoch checkpointing, written to survive a Colab disconnect.

Colab drops sessions without warning and wipes local disk, so the checkpoint
directory normally points at mounted Google Drive. Two consequences shape this
module:

* every write is atomic — a temporary file plus ``os.replace`` — because a
  disconnect halfway through writing ``last.pt`` must not leave a truncated
  file that cannot be resumed from;
* resume restores the *whole* run, not just the weights: optimizer, schedule,
  gradient scaler, epoch counter, best-so-far, history, and the random-number
  state of Python, NumPy and PyTorch. Anything less and the resumed half of the
  run is a different experiment from the first half.
"""

from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from fdl_project.constants import class_encoding_metadata, validate_checkpoint_class_names

LAST_CHECKPOINT_NAME = "last.pt"
BEST_CHECKPOINT_NAME = "best.pt"
EPOCH_CHECKPOINT_PATTERN = re.compile(r"^epoch_(\d{4,})\.pt$")
CHECKPOINT_SCHEMA_VERSION = 1


def _epoch_checkpoint_name(epoch: int) -> str:
    return f"epoch_{epoch:04d}.pt"


def capture_random_state() -> dict[str, Any]:
    """Snapshot every random source that influences the next epoch."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_random_state(state: dict[str, Any] | None) -> None:
    """Restore a snapshot taken by :func:`capture_random_state`."""

    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(_as_byte_tensor(state["torch"]))
    cuda_state = state.get("cuda")
    if cuda_state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([_as_byte_tensor(item) for item in cuda_state])


def _as_byte_tensor(value: Any) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return tensor.cpu().to(torch.uint8)


def _state_dict(module: Any) -> dict[str, Any] | None:
    return None if module is None else module.state_dict()


@dataclass(frozen=True)
class ResumeState:
    """Everything needed to continue a run from where it stopped.

    Passed straight to ``fit_model(resume=...)``. ``path`` records where it was
    loaded from and is absent when the state was built by hand.
    """

    next_epoch: int
    best_metric: float | None
    best_epoch: int
    history: list[dict[str, Any]]
    path: Path | None = None


class CheckpointManager:
    """Write and reload run state under ``<directory>/`` for one experiment.

    Files, all in the same directory so a run is one self-contained folder:

    ``epoch_0007.pt``    a rolling window of the most recent ``keep_last``
                         epochs; the newest one is what resume reads
    ``best.pt``          the best monitored epoch so far
    ``last.pt``          written *only* when ``keep_last=0``, since otherwise
                         it would duplicate the newest epoch file

    The window is small by default: a ResNet18 payload is ~140 MB once the
    optimizer moments are included, so every retained epoch is expensive on a
    15 GB Drive.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        keep_last: int = 2,
        save_best: bool = True,
    ) -> None:
        if isinstance(keep_last, bool) or not isinstance(keep_last, int) or keep_last < 0:
            raise ValueError("keep_last must be a non-negative integer.")
        self.directory = Path(directory)
        self.keep_last = keep_last
        self.save_best = save_best
        self.directory.mkdir(parents=True, exist_ok=True)

    # -- writing ---------------------------------------------------------

    def save(
        self,
        *,
        epoch: int,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        scheduler: Any = None,
        scaler: Any = None,
        best_metric: float | None = None,
        best_epoch: int = 0,
        history: list[dict[str, Any]] | None = None,
        config: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
        is_best: bool = False,
    ) -> Path:
        """Persist one epoch and return the path of the rolling checkpoint."""

        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "epoch": int(epoch),
            "model_state": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
            "model_class": type(model).__name__,
            "optimizer_state": _state_dict(optimizer),
            "scheduler_state": _state_dict(scheduler),
            "scaler_state": _state_dict(scaler),
            "best_metric": None if best_metric is None else float(best_metric),
            "best_epoch": int(best_epoch),
            "history": list(history or []),
            "config": dict(config or {}),
            "random_state": capture_random_state(),
            **class_encoding_metadata(),
            **(extra or {}),
        }

        # The newest epoch file *is* the latest state, so writing last.pt as
        # well would duplicate it -- 140 MB per epoch for a ResNet18. last.pt
        # is therefore written only when no rolling history is kept at all.
        epoch_path = self.directory / _epoch_checkpoint_name(epoch)
        if self.keep_last > 0:
            self._atomic_save(payload, epoch_path)
        else:
            epoch_path = self.directory / LAST_CHECKPOINT_NAME
            self._atomic_save(payload, epoch_path)
        if is_best and self.save_best:
            self._atomic_save(payload, self.directory / BEST_CHECKPOINT_NAME)
        self.prune()
        return epoch_path

    def _atomic_save(self, payload: dict[str, Any], path: Path) -> None:
        """Write via a temporary file so an interrupted save cannot corrupt ``path``."""

        temporary = path.with_name(path.name + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)

    def prune(self) -> list[Path]:
        """Delete rolling checkpoints beyond ``keep_last`` and return what went."""

        checkpoints = self.epoch_checkpoints()
        if len(checkpoints) <= self.keep_last:
            return []
        removed = checkpoints[: len(checkpoints) - self.keep_last]
        for path in removed:
            path.unlink(missing_ok=True)
        return removed

    # -- reading ---------------------------------------------------------

    def epoch_checkpoints(self) -> list[Path]:
        """Rolling checkpoints, oldest first."""

        found = []
        for path in self.directory.iterdir() if self.directory.is_dir() else []:
            match = EPOCH_CHECKPOINT_PATTERN.match(path.name)
            if match:
                found.append((int(match.group(1)), path))
        return [path for _, path in sorted(found)]

    @property
    def last_path(self) -> Path:
        """The most recent checkpoint: the highest-numbered epoch, else last.pt."""

        epochs = self.epoch_checkpoints()
        if epochs:
            return epochs[-1]
        return self.directory / LAST_CHECKPOINT_NAME

    @property
    def best_path(self) -> Path:
        return self.directory / BEST_CHECKPOINT_NAME

    def has_checkpoint(self) -> bool:
        return self.last_path.is_file()

    def prune_orphaned_last(self) -> None:
        """Remove a last.pt left behind by a run that used keep_last=0."""

        if self.keep_last > 0:
            (self.directory / LAST_CHECKPOINT_NAME).unlink(missing_ok=True)

    def load(self, path: str | Path | None = None, *, map_location: Any = "cpu") -> dict[str, Any]:
        """Load a checkpoint and verify it was trained on our class encoding."""

        return load_checkpoint(self.last_path if path is None else path, map_location=map_location)

    def resume(
        self,
        model: nn.Module,
        *,
        optimizer: Optimizer | None = None,
        scheduler: Any = None,
        scaler: Any = None,
        map_location: Any = "cpu",
    ) -> ResumeState | None:
        """Restore a run in place from ``last.pt``; return ``None`` if absent."""

        if not self.has_checkpoint():
            return None
        payload = self.load(map_location=map_location)
        model.load_state_dict(payload["model_state"])
        if optimizer is not None and payload.get("optimizer_state"):
            optimizer.load_state_dict(payload["optimizer_state"])
        if scheduler is not None and payload.get("scheduler_state"):
            scheduler.load_state_dict(payload["scheduler_state"])
        if scaler is not None and payload.get("scaler_state"):
            scaler.load_state_dict(payload["scaler_state"])
        restore_random_state(payload.get("random_state"))
        return ResumeState(
            next_epoch=int(payload["epoch"]) + 1,
            best_metric=payload.get("best_metric"),
            best_epoch=int(payload.get("best_epoch", 0)),
            history=list(payload.get("history", [])),
            path=self.last_path,
        )


def load_checkpoint(path: str | Path, *, map_location: Any = "cpu") -> dict[str, Any]:
    """Load one checkpoint file, validating its recorded class encoding."""

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}.")
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError(f"{checkpoint_path} is not a project checkpoint.")
    if "class_names" not in payload:
        raise ValueError(
            f"{checkpoint_path} records no class encoding, so its logits cannot "
            "be interpreted."
        )
    # A checkpoint trained under a different class order would silently produce
    # relabelled predictions, so this is a hard failure rather than a warning.
    validate_checkpoint_class_names(payload["class_names"])
    return payload
