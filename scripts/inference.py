"""Evaluate a saved checkpoint on the validation or test split.

    uv run python scripts/inference.py --checkpoint output/runs/<name>/best_model.pt

The test split is frozen until the final comparison, so evaluating it needs an
explicit acknowledgement flag. Every selection decision -- architecture,
hyperparameters, augmentation, thresholds -- must be made on validation.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from fdl_project.training.inference import evaluate_checkpoint

FINAL_EVALUATION_FLAG = "--final-test-evaluation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint to evaluate (best_model.pt, best.pt, or any epoch file).",
    )
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
        help="Split to evaluate (default: validation).",
    )
    parser.add_argument(
        FINAL_EVALUATION_FLAG,
        action="store_true",
        help="Required to touch the test split. Use once, at the very end.",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    # The package logs through `logging`; this renders it as plain lines, and
    # lets a notebook caller configure it differently.
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    if args.split == "test" and not args.final_test_evaluation:
        raise SystemExit(
            "Refusing to evaluate the test split without "
            f"{FINAL_EVALUATION_FLAG}. The protocol allows exactly one test "
            "evaluation, at the end of the project; use --split validation for "
            "everything else."
        )

    result = evaluate_checkpoint(
        args.checkpoint,
        split_name=args.split,
        output_root=args.output_root,
        run_name=args.run_name,
        device=args.device,
        overwrite=args.overwrite,
        bootstrap_resamples=args.bootstrap_resamples,
        batch_size=args.batch_size,
    )

    aggregate = result.bootstrap.aggregate.set_index("metric")
    print(f"\n{result.split_name} results for {result.config.name}:")
    for metric in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"):
        row = aggregate.loc[metric]
        print(
            f"  {metric:<20} {row.point_estimate:.4f} "
            f"[{row.ci_lower:.4f}, {row.ci_upper:.4f}]"
        )
    print("\nPer-class F1 with intervals:")
    for row in result.bootstrap.per_class.itertuples():
        print(
            f"  {row.class_name:<10} {row.f1:.4f} "
            f"[{row.ci_lower:.4f}, {row.ci_upper:.4f}]  n={row.support}"
        )
    print(f"\nArtifacts: {result.artifact_paths['metrics'].parent.resolve()}")


if __name__ == "__main__":
    main()
