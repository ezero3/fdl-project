"""Per-epoch side effects: console output, checkpointing, and W&B logging.

The training loop stays about optimization. Everything that reacts to an epoch
finishing — printing it, saving it, shipping it to Weights & Biases — is a
callback, so adding a destination never means editing the loop.
"""

from __future__ import annotations

import time
from typing import Any, Protocol

from fdl_project.config.schema import ExperimentConfig, WandbConfig
from fdl_project.training.checkpoint import CheckpointManager


class Callback(Protocol):
    """Anything the loop notifies as a run progresses."""

    def on_train_start(self, context: dict[str, Any]) -> None: ...

    def on_epoch_end(self, metrics: dict[str, Any], context: dict[str, Any]) -> None: ...

    def on_train_end(self, context: dict[str, Any]) -> None: ...


class BaseCallback:
    """No-op defaults so a callback only implements what it cares about."""

    def on_train_start(self, context: dict[str, Any]) -> None:
        return None

    def on_epoch_end(self, metrics: dict[str, Any], context: dict[str, Any]) -> None:
        return None

    def on_train_end(self, context: dict[str, Any]) -> None:
        return None


class ConsoleLogger(BaseCallback):
    """One readable line per epoch, plus wall-clock timing."""

    def __init__(
        self, *, monitor: str = "validation_macro_f1", run_name: str = "run"
    ) -> None:
        self.monitor = monitor
        self.run_name = run_name
        self._started = 0.0

    def on_train_start(self, context: dict[str, Any]) -> None:
        self._started = time.monotonic()
        print(
            f"Training {self.run_name} on {context.get('device')} "
            f"for up to {context.get('max_epochs')} epochs."
        )

    def on_epoch_end(self, metrics: dict[str, Any], context: dict[str, Any]) -> None:
        marker = " *" if context.get("is_best") else ""
        learning_rate = context.get("learning_rate")
        rate = f" lr {learning_rate:.2e}" if learning_rate is not None else ""
        print(
            f"epoch {metrics['epoch']:>3}"
            f"  train_loss {metrics.get('train_loss', float('nan')):.4f}"
            f"  val_loss {metrics.get('validation_loss', float('nan')):.4f}"
            f"  {self.monitor} {metrics.get(self.monitor, float('nan')):.4f}"
            f"{rate}"
            f"  {metrics.get('epoch_seconds', 0.0):.1f}s{marker}"
        )

    def on_train_end(self, context: dict[str, Any]) -> None:
        elapsed = time.monotonic() - self._started
        print(
            f"Finished at epoch {context.get('last_epoch')} "
            f"(best epoch {context.get('best_epoch')}, "
            f"{self.monitor} {context.get('best_metric')}) in {elapsed / 60:.1f} min."
        )


class CheckpointCallback(BaseCallback):
    """Write a full resumable checkpoint at the end of every epoch."""

    def __init__(
        self,
        manager: CheckpointManager,
        *,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.manager = manager
        self.config = dict(config or {})

    def on_epoch_end(self, metrics: dict[str, Any], context: dict[str, Any]) -> None:
        self.manager.save(
            epoch=int(metrics["epoch"]),
            model=context["model"],
            optimizer=context.get("optimizer"),
            scheduler=context.get("scheduler"),
            scaler=context.get("scaler"),
            best_metric=context.get("best_metric"),
            best_epoch=context.get("best_epoch", 0),
            history=context.get("history"),
            config=self.config,
            is_best=bool(context.get("is_best")),
        )


def per_class_metrics(validation: Any) -> dict[str, float]:
    """Flatten the per-class table into loggable ``validation_f1_<class>`` keys.

    The aggregate macro-F1 hides which classes moved. On this dataset the
    interesting signal is almost entirely in the rare classes -- a run that
    lifts `Scratch` from 0.00 to 0.20 barely shifts macro-F1 -- so the
    per-class series is what makes two runs comparable.
    """

    table = getattr(validation, "per_class_metrics", None)
    if table is None:
        return {}
    return {
        f"validation_f1_{row.class_name}": float(row.f1)
        for row in table.itertuples()
    }


class WandbLogger(BaseCallback):
    """Mirror every epoch into one shared Weights & Biases project.

    ``wandb`` is imported lazily and is an optional dependency, so the package
    and its tests work without it installed. Use a *private* project.

    On Colab it needs its own ``!pip install wandb``: the documented install
    there is ``pip install -e . --no-deps``, which deliberately skips
    dependencies so Colab's driver-matched torch survives -- and therefore
    skips extras as well.
    """

    def __init__(
        self,
        config: WandbConfig,
        *,
        run_name: str,
        resolved_config: dict[str, Any] | None = None,
        resume_id: str | None = None,
    ) -> None:
        try:
            import wandb
        except ImportError as error:  # pragma: no cover - environment guard
            raise ImportError(
                "Weights & Biases logging is enabled but wandb is not installed. "
                "Locally: uv sync --extra logging. On Colab: !pip install wandb "
                "(the documented '-e . --no-deps' install skips extras)."
            ) from error

        self._wandb = wandb
        self.run = wandb.init(
            entity=config.entity,
            project=config.project,
            name=run_name,
            mode=config.mode,
            tags=list(config.tags) or None,
            notes=config.notes,
            config=dict(resolved_config or {}),
            id=resume_id,
            resume="allow" if resume_id else None,
        )

    @property
    def run_id(self) -> str | None:
        return None if self.run is None else self.run.id

    def on_epoch_end(self, metrics: dict[str, Any], context: dict[str, Any]) -> None:
        payload = dict(metrics)
        learning_rate = context.get("learning_rate")
        if learning_rate is not None:
            payload["learning_rate"] = learning_rate
        payload.update(per_class_metrics(context.get("validation")))
        self._wandb.log(payload, step=int(metrics["epoch"]))

    def on_train_end(self, context: dict[str, Any]) -> None:
        if self.run is not None:
            self.run.summary["best_epoch"] = context.get("best_epoch")
            self.run.summary["best_metric"] = context.get("best_metric")
            self.run.summary["stopped_early"] = context.get("stopped_early")
            self.run.finish()


def build_callbacks(
    config: ExperimentConfig,
    *,
    checkpoint_manager: CheckpointManager | None = None,
    console: bool = True,
) -> list[Callback]:
    """Assemble the callbacks an experiment config asks for."""

    callbacks: list[Callback] = []
    if console:
        callbacks.append(
            ConsoleLogger(
                monitor=config.trainer.early_stopping.monitor, run_name=config.name
            )
        )
    if checkpoint_manager is not None:
        callbacks.append(
            CheckpointCallback(checkpoint_manager, config=config.to_dict())
        )
    if config.logging.wandb.enabled:
        callbacks.append(
            WandbLogger(
                config.logging.wandb,
                run_name=config.name,
                resolved_config=config.to_dict(),
            )
        )
    return callbacks
