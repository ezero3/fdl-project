"""Score a set of trained checkpoints without training anything.

    uv run python scripts/posthoc.py --inspect  --checkpoints-dir <drive>/checkpoints
    uv run python scripts/posthoc.py --checkpoints-dir <drive>/checkpoints --split validation
    uv run python scripts/posthoc.py --checkpoint a/best.pt --checkpoint b/best.pt \
        --split both --tta --tune-thresholds --final-test-evaluation

Every model is rebuilt from the config stored inside its own checkpoint, so the
preprocessing, encoding and resolution each one was trained with are reproduced
exactly -- there is no way to score a 128px model through a 64px pipeline by
accident.

Three things this enforces that a loop over `scripts/inference.py` would not:

* **The pickle loads once.** Reading it costs ~2 minutes; a dozen checkpoints
  across two splits would otherwise spend an hour on the same file.
* **The test split needs `--final-test-evaluation`.** The protocol allows one
  test evaluation, at the end. Validation drives every decision before that.
* **Per-class weights are fitted on validation and reused on test.** Fitting
  them on test is selecting on the frozen split, which is the whole thing the
  split exists to prevent. With `--split both` this notebook wires the reuse
  automatically; it is not left to the caller to remember.
"""

from __future__ import annotations

import argparse
import json
import logging
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from fdl_project.config.loader import build_experiment_config
from fdl_project.data.datasets import load_wm811k_dataframe
from fdl_project.training.checkpoint import load_checkpoint
from fdl_project.training.inference import evaluate_checkpoint

logger = logging.getLogger("posthoc")

FINAL_EVALUATION_FLAG = "--final-test-evaluation"
#: Preference order when a directory is given rather than an explicit file.
CHECKPOINT_PREFERENCE = ("best.pt", "best_model.pt", "last.pt")


def discover_checkpoints(directory: Path) -> list[Path]:
    """One checkpoint per run directory, preferring the best-epoch file.

    Drive folders acquire duplicates -- a second copy of a run lands beside the
    first as `<name> (1)` -- so every candidate is returned and `--inspect`
    is the way to tell them apart before spending GPU time on the wrong one.
    """

    found: list[Path] = []
    for run_directory in sorted(p for p in directory.iterdir() if p.is_dir()):
        for name in CHECKPOINT_PREFERENCE:
            candidate = run_directory / name
            if candidate.is_file():
                found.append(candidate)
                break
        else:
            epochs = sorted(run_directory.glob("epoch_*.pt"))
            if epochs:
                found.append(epochs[-1])
    return found


def describe(path: Path) -> dict[str, Any]:
    """Read a checkpoint's identity without building the model."""

    row: dict[str, Any] = {"checkpoint": str(path)}
    try:
        payload = load_checkpoint(path)
    except Exception as error:  # noqa: BLE001 - a corrupt file must not stop the sweep
        return {**row, "error": f"{type(error).__name__}: {error}"}

    stored = payload.get("config") or {}
    try:
        config = build_experiment_config(stored)
        row.update(
            run=config.name,
            model=config.model.name,
            px=config.data.preprocessing.target_size[0],
            encoding=config.data.preprocessing.encoding,
            augmentation=config.data.augmentation.name,
        )
    except Exception:  # noqa: BLE001
        row.update(run=stored.get("name"), model=(stored.get("model") or {}).get("name"))
    row.update(
        epoch=payload.get("epoch"),
        best_epoch=payload.get("best_epoch"),
        best_metric=payload.get("best_metric"),
        size_mb=round(path.stat().st_size / 1024**2, 1),
    )
    return row


def score_one(
    path: Path,
    split: str,
    *,
    dataframe: pd.DataFrame,
    output_root: Path,
    device: str | None,
    bootstrap_resamples: int,
    batch_size: int | None,
    tta: bool,
    tune_thresholds: bool,
    class_weights_path: Path | None,
) -> tuple[dict[str, Any], Path | None]:
    """Evaluate one checkpoint on one split; return a summary row."""

    result = evaluate_checkpoint(
        path,
        split_name=split,
        output_root=output_root,
        device=device,
        overwrite=True,
        bootstrap_resamples=bootstrap_resamples,
        batch_size=batch_size,
        tta=tta,
        tune_thresholds=tune_thresholds,
        class_weights_path=class_weights_path,
        dataframe=dataframe,
    )
    macro = result.bootstrap.aggregate.set_index("metric").loc["macro_f1"]
    per_class = result.evaluation.per_class_metrics.set_index("class_name")["f1"]

    def number(key: str, digits: int = 4) -> float | None:
        # TTA averages probabilities over the 8 symmetries and never computes a
        # loss, so mean_loss is legitimately absent there rather than an error.
        value = result.evaluation.metrics.get(key)
        return None if value is None else round(float(value), digits)

    row: dict[str, Any] = {
        "run": result.config.name,
        "split": split,
        "model": result.config.model.name,
        "px": result.config.data.preprocessing.target_size[0],
        "tta": tta,
        "thresholds": bool(tune_thresholds or class_weights_path),
        "macro_f1": round(float(macro.point_estimate), 4),
        "ci_lower": round(float(macro.ci_lower), 4),
        "ci_upper": round(float(macro.ci_upper), 4),
        "accuracy": number("accuracy"),
        "balanced_accuracy": number("balanced_accuracy"),
        "weighted_f1": number("weighted_f1"),
        "mean_loss": number("mean_loss"),
        **{f"f1_{name}": round(float(value), 3) for name, value in per_class.items()},
        "checkpoint": str(path),
        "artifacts": str(result.artifact_paths["metrics"].parent),
    }
    return row, result.artifact_paths.get("class_weights")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, action="append", default=[],
                        help="A checkpoint file. Repeatable.")
    parser.add_argument("--checkpoints-dir", type=Path, default=None,
                        help="Directory of run folders; one checkpoint is taken from each.")
    parser.add_argument("--inspect", action="store_true",
                        help="List what each checkpoint is and exit. No GPU work.")
    parser.add_argument("--split", choices=("validation", "test", "both"),
                        default="validation")
    parser.add_argument(FINAL_EVALUATION_FLAG, action="store_true",
                        help="Required to touch the test split. Use once, at the end.")
    parser.add_argument("--output-root", type=Path, default=Path("output/posthoc"))
    parser.add_argument("--results", type=Path, default=None,
                        help="Where to write the comparison table (default: <root>/comparison.csv).")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--tta", action="store_true",
                        help="Average over the 8 square symmetries (8x inference).")
    parser.add_argument("--tune-thresholds", action="store_true",
                        help="Fit per-class decision weights on validation, reuse on test.")
    parser.add_argument("--only", action="append", default=[],
                        help="Substring filter on the run name. Repeatable.")
    return parser.parse_args()


def collect_paths(args: argparse.Namespace) -> list[Path]:
    paths = list(args.checkpoint)
    if args.checkpoints_dir is not None:
        if not args.checkpoints_dir.is_dir():
            raise SystemExit(f"{args.checkpoints_dir} is not a directory.")
        paths += discover_checkpoints(args.checkpoints_dir)
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise SystemExit("Missing checkpoint(s): " + ", ".join(str(p) for p in missing))
    if not paths:
        raise SystemExit("Nothing to do: pass --checkpoint or --checkpoints-dir.")
    return paths


def run(args: argparse.Namespace) -> pd.DataFrame:
    paths = collect_paths(args)

    described = [describe(path) for path in paths]
    if args.only:
        keep = {
            row["checkpoint"]
            for row in described
            if any(f in str(row.get("run") or row["checkpoint"]) for f in args.only)
        }
        paths = [p for p in paths if str(p) in keep]
        described = [r for r in described if r["checkpoint"] in keep]
        if not paths:
            raise SystemExit(f"No checkpoint matched --only {args.only}.")

    table = pd.DataFrame(described)
    pd.set_option("display.width", 220)
    print(f"\n{len(paths)} checkpoint(s):\n")
    print(table.to_string(index=False))
    if args.inspect:
        return table

    splits = ["validation", "test"] if args.split == "both" else [args.split]
    if "test" in splits and not args.final_test_evaluation:
        raise SystemExit(
            f"Refusing to evaluate the test split without {FINAL_EVALUATION_FLAG}. "
            "The protocol allows exactly one test evaluation, at the very end."
        )

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results_path = Path(args.results or output_root / "comparison.csv")

    # One read of the 2 GB pickle for every checkpoint and both splits. Each
    # config points at the same file; the first one resolves it.
    first = build_experiment_config(load_checkpoint(paths[0])["config"])
    logger.info("Loading %s once for every checkpoint...", first.data.dataset_path)
    dataframe = load_wm811k_dataframe(first.data.dataset_path)

    rows: list[dict[str, Any]] = []
    for path in paths:
        # Weights fitted on validation, reused on test. Fitting on test would be
        # selecting on the frozen split.
        fitted_weights: Path | None = None
        for split in splits:
            label = f"{path.parent.name}/{path.name} [{split}]"
            try:
                row, weights = score_one(
                    path,
                    split,
                    dataframe=dataframe,
                    output_root=output_root,
                    device=args.device,
                    bootstrap_resamples=args.bootstrap_resamples,
                    batch_size=args.batch_size,
                    tta=args.tta,
                    tune_thresholds=args.tune_thresholds and split == "validation",
                    class_weights_path=fitted_weights if split == "test" else None,
                )
            except Exception as error:  # noqa: BLE001 - one bad checkpoint must not stop the rest
                logger.error("%s failed: %s", label, error)
                logger.debug(traceback.format_exc())
                rows.append({"run": path.parent.name, "split": split,
                             "error": f"{type(error).__name__}: {error}",
                             "checkpoint": str(path)})
                continue
            if weights is not None:
                fitted_weights = Path(weights)
            rows.append(row)
            print(f"  {label:58} macro-F1 {row['macro_f1']:.4f} "
                  f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]")
            # Written after every model, so an interrupted sweep keeps its work.
            pd.DataFrame(rows).to_csv(results_path, index=False)

    frame = pd.DataFrame(rows)
    frame.to_csv(results_path, index=False)
    (output_root / "comparison.json").write_text(json.dumps(rows, indent=2, default=str))
    print(f"\n{len(frame)} row(s) -> {results_path}")
    return frame


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    frame = run(args)
    if not args.inspect and not frame.empty and "macro_f1" in frame:
        done = frame[frame["macro_f1"].notna()].sort_values("macro_f1", ascending=False)
        print("\nRanked:")
        columns = [c for c in ("run", "split", "macro_f1", "ci_lower", "ci_upper",
                               "balanced_accuracy", "f1_Scratch", "f1_Loc") if c in done]
        print(done[columns].to_string(index=False))


if __name__ == "__main__":
    main()
