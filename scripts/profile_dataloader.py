"""Find out whether the dataloader or the GPU is the limit, and what to fix.

    python scripts/profile_dataloader.py --config configs/train/baseline_cnn_64.yaml

Reports three numbers per setting:

* **loader only** -- iterate batches, touch nothing else. This is the floor
  the training loop cannot beat.
* **full epoch**  -- the real training step on top.
* **ratio**       -- loader / full.

The difference between them is **not** "GPU time": in a real epoch the loader
and the GPU overlap, so subtracting one from the other decomposes nothing.
Read the ratio instead. Near 1.0 means the loop is spending essentially all
its time waiting for data, and no GPU change will help; well under 1.0 means
loading is already hidden behind compute.
"""

from __future__ import annotations

import argparse
import os
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
            "ratio": loader_only / full, "items": items}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/train/baseline_cnn_64.yaml"))
    args = parser.parse_args()

    dataframe = load_wm811k_dataframe(
        load_experiment_config(args.config).data.dataset_path
    )

    # The VM has 8 cores, and `num_workers=8` leaves none for the main
    # process -- which still has to run the training loop, dispatch to the
    # GPU and collate. Under-subscribing may beat the core count; heavy
    # over-subscription usually just adds context switching. 12 is included
    # only because workers spend some time blocked on IPC rather than
    # computing, which is the one case where it can help.
    cores = os.cpu_count() or 8
    counts = sorted({4, cores - 2, cores, cores + 4})
    settings = [(f"no aug,    {n:2} workers", [f"data.num_workers={n}"])
                for n in counts]
    settings += [(f"rotation,  {n:2} workers",
                  [f"data.num_workers={n}", "data.augmentation.name=rotation"])
                 for n in counts]

    print(f"{'setting':24} {'loader':>9} {'full':>9} {'ratio':>8} {'items/s':>10}")
    for label, overrides in settings:
        result = time_setting(args.config, overrides, dataframe)
        print(f"{label:24} {result['loader_only']:8.1f}s {result['full']:8.1f}s "
              f"{result['ratio']:8.2f} {result['items'] / result['full']:10.0f}")

    print("\nA ratio near 1.0 means the loop is waiting on data. Fixes, by size:")
    print("  1. move the one-hot encoding to the GPU (~50% of per-item CPU cost,")
    print("     and 12x less data over PCIe: uint8 categorical vs float32 3-channel)")
    print("  2. cache the letterboxed uint8 map (~40% more; 496 MB at 64x64)")
    print(f"  3. worker count -- this machine has {os.cpu_count()} cores, and the main")
    print("     process needs one of them, so the best setting may be below that")


if __name__ == "__main__":
    main()
