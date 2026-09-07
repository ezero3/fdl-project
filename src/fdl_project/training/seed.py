"""Process-wide seeding for reproducible GPU training with dataloader workers.

Modelled on Lightning's ``seed_everything``. Two things the previous
``set_reproducible_seed`` got wrong on the hardware we actually train on:

* it enabled deterministic algorithms in strict mode, which raises a
  ``RuntimeError`` mid-training on CUDA because several convolution backward
  kernels have no deterministic implementation;
* it left dataloader workers unseeded, so every worker inherits the same base
  seed and augmentation silently collapses to one repeated view per epoch.

GPU random state itself was already fine: ``torch.manual_seed`` seeds all CUDA
devices internally.

**Deterministic kernels are off by default.** What reproducibility actually
needs here is identical initialization, batch order, sampler draws and
augmentation views, and seeding alone gives all of that. Forcing deterministic
algorithms would additionally disable the cuDNN autotuner -- a real throughput
cost, since our input shapes are fixed and autotuning wins on exactly that
case -- while still not delivering bit-exact results, because kernels without
a deterministic implementation fall back silently. Remaining kernel-level
jitter is far below the bootstrap confidence intervals we report, so it cannot
change a conclusion. Pass ``deterministic=True`` for a one-off strict check.
"""

from __future__ import annotations

import os
import random
import warnings

import numpy as np
import torch

MAXIMUM_SEED = 2**32 - 1
CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def validate_seed(seed: int) -> int:
    """Return ``seed`` if it is a plain non-negative integer in NumPy's range."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be a non-negative integer.")
    if not 0 <= seed <= MAXIMUM_SEED:
        raise ValueError(f"seed must lie between 0 and {MAXIMUM_SEED}.")
    return seed


def seed_everything(
    seed: int,
    *,
    deterministic: bool = False,
    warn_only: bool = True,
) -> int:
    """Seed every random source this project uses and return the seed.

    ``deterministic`` additionally asks PyTorch for deterministic kernels and
    turns off the cuDNN autotuner. It is off by default -- see the module
    docstring for why the cost is not worth it here. ``warn_only`` keeps it
    non-fatal when enabled: operations without a deterministic kernel warn and
    fall back instead of raising, which is the only way a convolutional model
    trains to completion on CUDA.
    """

    validate_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds every CUDA device

    if deterministic:
        # cuBLAS needs this before CUDA initializes for reproducible matmuls.
        configure_cublas_workspace()
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    torch.backends.cudnn.deterministic = deterministic
    # Autotuning picks a different algorithm per run, so it goes with the rest.
    torch.backends.cudnn.benchmark = not deterministic
    return seed


def configure_cublas_workspace() -> None:
    """Request a reproducible cuBLAS workspace, warning if CUDA already started."""

    current = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if current == CUBLAS_WORKSPACE_CONFIG:
        return
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        warnings.warn(
            "CUDA is already initialized, so CUBLAS_WORKSPACE_CONFIG"
            f"={CUBLAS_WORKSPACE_CONFIG!r} will not take effect in this process. "
            "Call seed_everything before creating any CUDA tensor.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG


def seed_worker(worker_id: int) -> None:
    """Give one dataloader worker its own reproducible random streams.

    PyTorch derives ``torch.initial_seed()`` per worker from the loader's
    generator, so it already differs between workers and repeats across runs.
    Python's ``random`` and NumPy do not follow it, and this closes that gap.
    """

    worker_seed = torch.initial_seed() % (MAXIMUM_SEED + 1)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_dataloader_generator(seed: int) -> torch.Generator:
    """Create the CPU generator that drives shuffling and worker seeding."""

    return torch.Generator().manual_seed(validate_seed(seed))
