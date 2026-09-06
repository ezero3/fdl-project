"""Run the complete task-05 class-imbalance experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from fdl_project.imbalance_experiment import run_class_imbalance_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/MIR-WM811K/WM811K.pkl"),
    )
    parser.add_argument("--splits", type=Path, default=Path("data/splits"))
    parser.add_argument("--output", type=Path, default=Path("output/class_imbalance"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_class_imbalance_experiment(
        args.dataset,
        args.splits,
        args.output,
        overwrite=args.overwrite,
    )
    print(f"Selected strategy: {result.selected_strategy}")
    print(f"Representative seed: {result.representative_seed}")
    print(f"Artifacts: {args.output.resolve()}")


if __name__ == "__main__":
    main()
