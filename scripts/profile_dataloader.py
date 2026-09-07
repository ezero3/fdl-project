"""Find out whether the dataloader or the GPU is the limit, and what to fix.

    python scripts/profile_dataloader.py --config configs/train/baseline_cnn_64.yaml

Reports three numbers per setting:

* **loader only** -- iterate batches, touch nothing else. This is the floor.
* **full epoch**  -- the real training step on top.
* **implied GPU** -- the difference. If it is small, the GPU is idle waiting.

If "loader only" is close to "full epoch", more GPU buys nothing and the fix
is upstream: fewer per-item CPU operations, or more workers.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from fdl_project.config.loader import load_experiment_config
from fdl_project.config.registry import build_model
from fdl_project.data.datasets import load_wm811k_dataframe
from fdl_project.data.imbalance import build_training_loss, compute_class_counts
from fdl_project.training.loop import resolve_device, train_one_epoch
from fdl_project.training.optim import build_optimizer
from fdl_project.training.runner import build_dataloaders, build_datasets
from fdl_project.training.seed import seed_everything

SUBSET = 20_000


def time_setting(config_path, overrides, dataframe) -> dict:
    config = load_experiment_config(
        config_path, overrides=[f"data.subset={SUBSET}", *overrides]
    )
    seed_everything(config.seed)
    device = resolve_device(config.trainer.device)
    train_dataset, validation_dataset = build_datasets(config, dataframe)
    loader, _ = build_dataloaders(config, train_dataset, validation_dataset)

    # Warm the workers so their startup is not counted.
    for _ in loader:
        break

    started = time.monotonic()
    items = 0
    for inputs, targets, _ in loader:
        items += len(targets)      # touch the batch, do nothing with it
    loader_only = time.monotonic() - started

    criterion = build_training_loss(
        config.imbalance, compute_class_counts(train_dataset.target_indices)
    )
    model = build_model(config.model.name, **dict(config.model.kwargs)).to(device)
    optimizer = build_optimizer(model, config.optimizer)
    scaler = torch.amp.GradScaler(
        device.type, enabled=config.trainer.amp and device.type == "cuda"
    )
    train_one_epoch(model, loader, optimizer, criterion, device=device,
                    max_gradient_norm=config.trainer.max_gradient_norm, scaler=scaler)
    if device.type == "cuda":
        torch.cuda.synchronize()

    started = time.monotonic()
    train_one_epoch(model, loader, optimizer, criterion, device=device,
                    max_gradient_norm=config.trainer.max_gradient_norm, scaler=scaler)
    if device.type == "cuda":
        torch.cuda.synchronize()
    full = time.monotonic() - started

    return {"loader_only": loader_only, "full": full,
            "implied_gpu": full - loader_only, "items": items}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/train/baseline_cnn_64.yaml"))
    args = parser.parse_args()

    dataframe = load_wm811k_dataframe(
        load_experiment_config(args.config).data.dataset_path
    )

    settings = [
        ("no aug,  8 workers", ["data.num_workers=8"]),
        ("dihedral8, 8 workers", ["data.num_workers=8",
                                  "data.augmentation.name=dihedral8"]),
        ("dihedral8, 16 workers", ["data.num_workers=16",
                                   "data.augmentation.name=dihedral8"]),
        ("rotation,  8 workers", ["data.num_workers=8",
                                  "data.augmentation.name=rotation"]),
        ("rotation, 16 workers", ["data.num_workers=16",
                                  "data.augmentation.name=rotation"]),
    ]

    print(f"{'setting':24} {'loader':>9} {'full':>9} {'implied GPU':>12} {'loader %':>9}")
    for label, overrides in settings:
        result = time_setting(args.config, overrides, dataframe)
        share = result["loader_only"] / result["full"] * 100
        print(f"{label:24} {result['loader_only']:8.1f}s {result['full']:8.1f}s "
              f"{result['implied_gpu']:11.1f}s {share:8.0f}%")

    print("\nA high 'loader %' means the GPU is waiting. Fixes, in order of size:")
    print("  1. move the one-hot encoding to the GPU (~50% of per-item CPU cost,")
    print("     and 12x less data over PCIe: uint8 categorical vs float32 3-channel)")
    print("  2. cache the letterboxed uint8 map (~40% more; 496 MB at 64x64)")
    print("  3. more workers, if the numbers above show it still helps")


if __name__ == "__main__":
    main()
