"""Train-only class-imbalance strategies for WM-811K classification."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any, Final, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from fdl_project.constants import NUM_CLASSES
from fdl_project.training.seed import build_dataloader_generator, seed_worker

WeightingMethod = Literal["none", "inverse_sqrt", "effective_number"]
LossName = Literal["cross_entropy", "focal"]
SamplingMethod = Literal["shuffle", "weighted"]

_STRATEGY_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class ImbalanceConfig:
    """Immutable description of one isolated imbalance intervention."""

    name: str
    loss: LossName = "cross_entropy"
    loss_weighting: WeightingMethod = "none"
    sampling: SamplingMethod = "shuffle"
    sampling_weighting: WeightingMethod = "none"
    focal_gamma: float = 2.0
    effective_number_beta: float = 0.9999

    def __post_init__(self) -> None:
        if not _STRATEGY_NAME_PATTERN.fullmatch(self.name):
            raise ValueError(
                "name must start with a lowercase letter and contain only "
                "lowercase letters, numbers, and underscores."
            )
        if self.loss not in {"cross_entropy", "focal"}:
            raise ValueError(f"Unsupported loss: {self.loss!r}.")
        supported_weighting = {"none", "inverse_sqrt", "effective_number"}
        if self.loss_weighting not in supported_weighting:
            raise ValueError(f"Unsupported loss weighting: {self.loss_weighting!r}.")
        if self.sampling not in {"shuffle", "weighted"}:
            raise ValueError(f"Unsupported sampling method: {self.sampling!r}.")
        if self.sampling_weighting not in supported_weighting:
            raise ValueError(
                f"Unsupported sampling weighting: {self.sampling_weighting!r}."
            )
        if self.sampling == "shuffle" and self.sampling_weighting != "none":
            raise ValueError("Shuffle sampling cannot use class sampling weights.")
        if self.sampling == "weighted" and self.sampling_weighting == "none":
            raise ValueError("Weighted sampling requires a class weighting method.")
        if self.loss_weighting != "none" and self.sampling_weighting != "none":
            raise ValueError(
                "Loss reweighting and weighted sampling must be tested separately."
            )
        if self.loss == "focal" and self.loss_weighting != "none":
            raise ValueError(
                "The focal-loss candidate must remain unweighted to isolate its effect."
            )
        if (
            isinstance(self.focal_gamma, bool)
            or not isinstance(self.focal_gamma, (int, float))
            or self.focal_gamma < 0
        ):
            raise ValueError("focal_gamma must be a finite non-negative number.")
        if not torch.isfinite(torch.tensor(float(self.focal_gamma))):
            raise ValueError("focal_gamma must be a finite non-negative number.")
        if (
            isinstance(self.effective_number_beta, bool)
            or not isinstance(self.effective_number_beta, (int, float))
            or not 0 <= self.effective_number_beta < 1
        ):
            raise ValueError("effective_number_beta must be in [0, 1).")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PRESET_IMBALANCE_CONFIGS: Final = MappingProxyType(
    {
        "unweighted_ce": ImbalanceConfig(name="unweighted_ce"),
        "inverse_sqrt_ce": ImbalanceConfig(
            name="inverse_sqrt_ce",
            loss_weighting="inverse_sqrt",
        ),
        "effective_number_ce": ImbalanceConfig(
            name="effective_number_ce",
            loss_weighting="effective_number",
        ),
        "inverse_sqrt_sampler": ImbalanceConfig(
            name="inverse_sqrt_sampler",
            sampling="weighted",
            sampling_weighting="inverse_sqrt",
        ),
        "focal_loss": ImbalanceConfig(
            name="focal_loss",
            loss="focal",
            focal_gamma=2.0,
        ),
    }
)

DEFAULT_IMBALANCE_CONFIG: Final[ImbalanceConfig] = PRESET_IMBALANCE_CONFIGS[
    "inverse_sqrt_sampler"
]


def _validate_targets(targets: Any) -> Tensor:
    tensor = targets.detach() if isinstance(targets, Tensor) else torch.tensor(targets)
    if tensor.ndim != 1 or tensor.numel() == 0:
        raise ValueError("targets must be a non-empty one-dimensional array.")
    if tensor.dtype == torch.bool or tensor.is_floating_point() or tensor.is_complex():
        raise ValueError("targets must contain integer class indices.")
    tensor = tensor.to(dtype=torch.long, device="cpu")
    if torch.any((tensor < 0) | (tensor >= NUM_CLASSES)):
        raise ValueError(
            f"targets must contain class indices in [0, {NUM_CLASSES - 1}]."
        )
    return tensor


def compute_class_counts(targets: Any) -> Tensor:
    """Count canonical training targets and require coverage of all nine classes."""

    target_tensor = _validate_targets(targets)
    counts = torch.bincount(target_tensor, minlength=NUM_CLASSES)
    if torch.any(counts == 0):
        missing = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
        raise ValueError(
            f"Training targets are missing canonical classes: {missing!r}."
        )
    return counts


def _validate_class_counts(class_counts: Any) -> Tensor:
    counts = (
        class_counts.detach()
        if isinstance(class_counts, Tensor)
        else torch.tensor(class_counts)
    )
    if counts.ndim != 1 or len(counts) != NUM_CLASSES:
        raise ValueError(f"class_counts must contain exactly {NUM_CLASSES} values.")
    if counts.dtype == torch.bool or counts.is_floating_point() or counts.is_complex():
        raise ValueError("class_counts must contain positive integers.")
    counts = counts.to(dtype=torch.long, device="cpu")
    if torch.any(counts <= 0):
        raise ValueError("class_counts must contain positive integers.")
    return counts


def compute_class_weights(
    class_counts: Any,
    method: WeightingMethod,
    *,
    effective_number_beta: float = 0.9999,
) -> Tensor:
    """Return float32 class weights normalized to arithmetic mean one."""

    counts = _validate_class_counts(class_counts).to(dtype=torch.float64)
    if method == "none":
        raw_weights = torch.ones_like(counts)
    elif method == "inverse_sqrt":
        raw_weights = counts.rsqrt()
    elif method == "effective_number":
        if not 0 <= effective_number_beta < 1:
            raise ValueError("effective_number_beta must be in [0, 1).")
        if effective_number_beta == 0:
            raw_weights = torch.ones_like(counts)
        else:
            beta = torch.tensor(effective_number_beta, dtype=torch.float64)
            effective_numbers = -torch.expm1(counts * torch.log(beta)) / (1 - beta)
            raw_weights = effective_numbers.reciprocal()
    else:
        raise ValueError(f"Unsupported weighting method: {method!r}.")

    weights = raw_weights / raw_weights.mean()
    if not torch.isfinite(weights).all() or torch.any(weights <= 0):
        raise ValueError("Calculated class weights must be finite and positive.")
    return weights.to(dtype=torch.float32)


class FocalLoss(nn.Module):
    """Multiclass focal loss with optional canonical class weights."""

    def __init__(
        self,
        *,
        gamma: float = 2.0,
        class_weights: Tensor | None = None,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ) -> None:
        super().__init__()
        if isinstance(gamma, bool) or not isinstance(gamma, (int, float)) or gamma < 0:
            raise ValueError("gamma must be a finite non-negative number.")
        if not torch.isfinite(torch.tensor(float(gamma))):
            raise ValueError("gamma must be a finite non-negative number.")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be 'none', 'mean', or 'sum'.")
        if class_weights is not None:
            weights = torch.as_tensor(class_weights, dtype=torch.float32)
            if weights.shape != (NUM_CLASSES,) or not torch.isfinite(weights).all():
                raise ValueError(
                    f"class_weights must be finite with shape ({NUM_CLASSES},)."
                )
            if torch.any(weights <= 0):
                raise ValueError("class_weights must be positive.")
        else:
            weights = None

        self.gamma = float(gamma)
        self.reduction = reduction
        self.register_buffer("class_weights", weights)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        if logits.ndim != 2 or logits.shape[1] != NUM_CLASSES:
            raise ValueError(
                f"logits must have shape (batch_size, {NUM_CLASSES}); "
                f"received {tuple(logits.shape)!r}."
            )
        if targets.ndim != 1 or len(targets) != len(logits):
            raise ValueError("targets must contain one class index per logit row.")
        if targets.dtype != torch.long:
            raise ValueError("targets must contain torch.long class indices.")
        if torch.any((targets < 0) | (targets >= NUM_CLASSES)):
            raise ValueError(f"targets must contain indices in [0, {NUM_CLASSES - 1}].")

        log_probabilities = F.log_softmax(logits, dim=1)
        target_log_probabilities = log_probabilities.gather(
            1, targets[:, None]
        ).squeeze(1)
        target_probabilities = target_log_probabilities.exp()
        losses = -((1 - target_probabilities) ** self.gamma) * target_log_probabilities
        if self.class_weights is not None:
            losses = losses * self.class_weights[targets]

        if self.reduction == "none":
            return losses
        if self.reduction == "sum":
            return losses.sum()
        return losses.mean()


def build_training_loss(
    config: ImbalanceConfig,
    class_counts: Any,
) -> nn.Module:
    """Construct the training criterion for one isolated strategy."""

    class_weights = None
    if config.loss_weighting != "none":
        class_weights = compute_class_weights(
            class_counts,
            config.loss_weighting,
            effective_number_beta=config.effective_number_beta,
        )
    if config.loss == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights)
    return FocalLoss(gamma=config.focal_gamma, class_weights=class_weights)


def build_weighted_sampler(
    config: ImbalanceConfig,
    targets: Any,
    class_counts: Any,
    *,
    seed: int,
) -> WeightedRandomSampler | None:
    """Build a deterministic train-only sampler, or return None for shuffling."""

    target_tensor = _validate_targets(targets)
    counts = _validate_class_counts(class_counts)
    if not torch.equal(torch.bincount(target_tensor, minlength=NUM_CLASSES), counts):
        raise ValueError("class_counts do not match the supplied training targets.")
    if config.sampling == "shuffle":
        return None

    class_weights = compute_class_weights(
        counts,
        config.sampling_weighting,
        effective_number_beta=config.effective_number_beta,
    )
    sample_weights = class_weights[target_tensor].to(dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        sample_weights,
        num_samples=len(target_tensor),
        replacement=True,
        generator=generator,
    )


def imbalance_metadata(
    config: ImbalanceConfig,
    class_counts: Any,
) -> dict[str, Any]:
    """Return checkpoint-safe metadata for one configured intervention."""

    counts = _validate_class_counts(class_counts)
    method = (
        config.loss_weighting
        if config.loss_weighting != "none"
        else config.sampling_weighting
    )
    weights = compute_class_weights(
        counts,
        method,
        effective_number_beta=config.effective_number_beta,
    )
    return {
        "imbalance_config": config.to_dict(),
        "class_counts": counts.tolist(),
        "class_weights": weights.tolist(),
    }


def create_imbalance_training_dataloader(
    dataset: Any,
    *,
    batch_size: int,
    config: ImbalanceConfig = DEFAULT_IMBALANCE_CONFIG,
    prefetch_factor: int = 2,
    seed: int = 86,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Create a deterministic training loader with the selected intervention."""

    if getattr(dataset, "split_name", None) != "train":
        raise ValueError("Imbalance interventions are allowed only on the train split.")
    if not hasattr(dataset, "target_indices"):
        raise TypeError("Training dataset must expose canonical target_indices.")
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

    targets = _validate_targets(dataset.target_indices)
    if len(targets) != len(dataset):
        raise ValueError("target_indices must contain one target per dataset item.")
    counts = compute_class_counts(targets)
    sampler = build_weighted_sampler(config, targets, counts, seed=seed)
    generator = build_dataloader_generator(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
        # Batches held ready per worker; see create_dataloader.
        **({} if num_workers == 0 else {"prefetch_factor": prefetch_factor}),
        generator=generator if sampler is None else None,
        worker_init_fn=seed_worker if num_workers > 0 else None,
    )
