"""Train one experiment from its YAML config.

    uv run python scripts/train.py configs/train/baseline_cnn_64.yaml

Everything the run needs is in the config; the flags here only cover things
that change per machine or per attempt, not per experiment.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from fdl_project.config.loader import load_experiment_config
from fdl_project.training.runner import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to the experiment YAML.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override one config value, e.g. --override trainer.max_epochs=2. "
            "Repeatable. Values are parsed as YAML."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override trainer.device (auto, cpu, cuda, mps).",
    )
    parser.add_argument(
        "--resume",
        choices=("auto", "never"),
        default=None,
        help="Override checkpoint.resume for this launch.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing artifacts for this run name.",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=1000,
        help="Resamples used for the confidence intervals (default: 1000).",
    )
    return parser.parse_args()


def main() -> None:
    # The package logs through `logging`; this renders it as plain lines, and
    # lets a notebook caller configure it differently.
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    config = load_experiment_config(args.config, overrides=args.override)

    result = run_experiment(
        config,
        overwrite=args.overwrite,
        resume=args.resume,
        device=args.device,
        bootstrap_resamples=args.bootstrap_resamples,
    )

    print()
    for group in result.parameter_groups:
        print(
            f"  group {group['name']}: lr {group['learning_rate']:.2e}, "
            f"{group['num_parameters']:,} parameters"
        )
    macro = result.bootstrap.aggregate.set_index("metric").loc["macro_f1"]
    print(
        f"\nBest epoch {result.fit.best_epoch} | validation macro-F1 "
        f"{macro.point_estimate:.4f} [{macro.ci_lower:.4f}, {macro.ci_upper:.4f}]"
    )
    print(f"Artifacts: {result.artifact_paths['metrics'].parent.resolve()}")


if __name__ == "__main__":
    main()
