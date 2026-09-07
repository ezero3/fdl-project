"""Run the phase-2 pipeline sweep on one fixed model.

    uv run python scripts/sweep.py --stage 0        # control, run this alone first
    uv run python scripts/sweep.py --stage 1 2 3
    uv run python scripts/sweep.py --all --seeds 86 87 88

Everything that is not the model -- augmentation, imbalance, attention,
encoding, resolution, schedule -- is compared here on `baseline_cnn`, so each
comparison is controlled and cheap. Only a setting that wins is carried to the
real architectures. See docs/training-plan.md.

Designed to survive a dropped Colab session: every arm writes its own artifacts
and results are appended after each one, so re-running skips whatever already
finished. Launch it from a terminal rather than a notebook cell, so a kernel
disconnect does not take the sweep with it:

    nohup uv run python scripts/sweep.py --all > sweep.log 2>&1 &
    tail -f sweep.log

Per-machine settings go through --override, not by editing the configs:

    python scripts/sweep.py --stage 0 \
        --override checkpoint.directory=/content/drive/MyDrive/BICOCCA/FDL/checkpoints \
        --override logging.wandb.enabled=true
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from fdl_project.config.loader import load_experiment_config
from fdl_project.data.datasets import load_wm811k_dataframe
from fdl_project.training.runner import run_experiment

LOGGER = logging.getLogger("sweep")

#: The config every arm varies from. The sweep is only controlled if this is
#: the one thing that never changes.
BASE_CONFIG = Path("configs/train/baseline_cnn_64.yaml")
RESULTS = Path("output/phase2/results.csv")


@dataclass(frozen=True)
class Arm:
    stage: int
    name: str
    overrides: tuple[str, ...]
    note: str


#: Stage 0 is the control every other arm is measured against. Stages run in
#: order and each is meant to start from the previous stage's winner -- that is
#: a manual step, because deciding the winner is a judgement about overlapping
#: intervals, not an argmax.
ARMS: tuple[Arm, ...] = (
    Arm(0, "baseline", (), "defaults untouched"),
    # -- stage 1: augmentation ------------------------------------------
    Arm(1, "dihedral8", ("data.augmentation.name=dihedral8",),
        "all 8 square symmetries, exact permutations"),
    Arm(1, "rotation", ("data.augmentation.name=rotation",
                            "data.augmentation.probability=0.5"),
        "free angle, resamples, so half the samples"),
    Arm(1, "dihedral8-rotation", ("data.augmentation.name=dihedral8_rotation",),
        "group always, rotation on half"),
    # -- stage 2: imbalance ---------------------------------------------
    Arm(2, "unweighted-ce", ("imbalance.preset=unweighted_ce",),
        "the baseline the sampler's gain is measured against"),
    Arm(2, "focal-loss", ("imbalance.preset=focal_loss",),
        "compare against unweighted, not stacked on the sampler"),
    # -- stage 3: attention ---------------------------------------------
    Arm(3, "cbam", ("model.kwargs.attention=cbam",), "+610 parameters"),
    # -- stage 4: encoding ----------------------------------------------
    Arm(4, "grayscale", ("data.preprocessing.encoding=grayscale_rgb",),
        "does the false ordering hurt?"),
    # -- stage 5: resolution --------------------------------------------
    Arm(5, "224px", ("data.preprocessing.target_size=[224,224]",),
        "224 is near-native (max 212x187); 64 is the compression"),
    # -- stage 6: schedule ----------------------------------------------
    Arm(6, "cosine", ("scheduler.name=cosine",),
        "only meaningful if the control runs near max_epochs"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stage", type=int, nargs="+", help="Stages to run.")
    group.add_argument("--all", action="store_true", help="Every stage in order.")
    group.add_argument("--list", action="store_true", help="Show the arms and exit.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[86],
                        help="One run per seed per arm (default: 86).")
    parser.add_argument("--device", default=None)
    parser.add_argument("--config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--rerun", action="store_true",
                        help="Repeat arms that already have results.")
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument(
        "--override", action="append", default=[], metavar="KEY=VALUE",
        help=(
            "Applied to every arm, after the arm's own overrides. For things "
            "that vary by machine rather than by experiment: checkpoint "
            "directory, W&B, device. Repeatable."
        ),
    )
    return parser.parse_args()


def verify_checkpoint_directory(directory: Path) -> None:
    """Fail now rather than after two hours of training.

    The dangerous failure is not a missing folder -- `CheckpointManager`
    creates those. It is an *unmounted* Drive: `/content/drive/MyDrive/...` is
    then just a local path, `mkdir` happily creates it, every write succeeds,
    and the whole run vanishes when the session ends. It looks completely
    healthy until the moment the progress is gone.
    """

    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise SystemExit(f"Cannot create checkpoint directory {directory}: {error}")

    probe = directory / ".write_probe"
    try:
        probe.write_text("ok")
        assert probe.read_text() == "ok"
        probe.unlink()
    except Exception as error:
        raise SystemExit(f"Checkpoint directory is not writable: {directory} ({error})")

    if "/drive/" in str(directory):
        mount = Path("/content/drive/MyDrive")
        if not mount.is_dir() or not any(mount.iterdir()):
            raise SystemExit(
                f"{directory} looks like a Drive path, but Drive is not "
                "mounted -- these writes would go to local disk and be lost "
                "when the session ends. Mount Drive first, or pass a local "
                "--override checkpoint.directory=... if that is what you want."
            )

    free = shutil.disk_usage(directory).free / 1024**3
    print(f"checkpoints -> {directory}  ({free:.1f} GiB free)")
    if free < 5:
        print("  WARNING: under 5 GiB free; a long sweep may run out mid-run.")


def completed_runs() -> set[str]:
    """Arm names already recorded, so a resumed sweep does not repeat work."""

    if not RESULTS.is_file():
        return set()
    with RESULTS.open() as handle:
        return {row["run"] for row in csv.DictReader(handle)}


def record(row: dict) -> None:
    """Append one result immediately -- a crash must not lose earlier arms."""

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    exists = RESULTS.is_file()
    with RESULTS.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    selected = [
        arm for arm in ARMS
        if args.all or args.list or arm.stage in set(args.stage or ())
    ]
    if args.list:
        model = load_experiment_config(args.config).model.name
        for arm in ARMS:
            print(f"  stage {arm.stage}  {model}-{arm.name:22} {arm.note}")
        return

    # The model is part of every run name: `baseline_cnn-dihedral8-s86` says
    # what it is from the directory listing alone, without opening the CSV.
    model = load_experiment_config(args.config).model.name
    done = set() if args.rerun else completed_runs()
    planned = [
        (arm, seed) for arm in selected for seed in args.seeds
        if args.rerun or f"{model}-{arm.name}-s{seed}" not in done
    ]
    if not planned:
        print("Nothing to do: every selected arm already has a result.")
        return

    verify_checkpoint_directory(
        Path(load_experiment_config(args.config, overrides=args.override)
             .checkpoint.directory)
    )
    print(f"{len(planned)} run(s) to go; {len(done)} already recorded.\n")
    dataframe = load_wm811k_dataframe(
        load_experiment_config(args.config).data.dataset_path
    )

    for index, (arm, seed) in enumerate(planned, start=1):
        run = f"{model}-{arm.name}-s{seed}"
        print(f"[{index}/{len(planned)}] stage {arm.stage}  {run}  -- {arm.note}")
        started = time.monotonic()
        try:
            config = load_experiment_config(
                args.config,
                # Arm overrides define the experiment; --override comes last
                # so per-machine settings can be forced on top.
                overrides=[f"name={run}", f"seed={seed}", *arm.overrides,
                           *args.override],
            )
            result = run_experiment(
                config,
                overwrite=True,
                device=args.device,
                bootstrap_resamples=args.bootstrap_resamples,
                dataframe=dataframe,
            )
            macro = result.bootstrap.aggregate.set_index("metric").loc["macro_f1"]
            record({
                "run": run, "stage": arm.stage, "arm": arm.name, "seed": seed,
                "macro_f1": round(float(macro.point_estimate), 4),
                "ci_lower": round(float(macro.ci_lower), 4),
                "ci_upper": round(float(macro.ci_upper), 4),
                "best_epoch": result.fit.best_epoch,
                "epochs_run": len(result.fit.history),
                "minutes": round((time.monotonic() - started) / 60, 1),
                "overrides": " ".join(arm.overrides),
                "note": arm.note,
            })
            print(f"    macro-F1 {macro.point_estimate:.4f} "
                  f"[{macro.ci_lower:.4f}, {macro.ci_upper:.4f}]  "
                  f"best epoch {result.fit.best_epoch}/{len(result.fit.history)}  "
                  f"{(time.monotonic() - started) / 60:.1f} min\n")
        except Exception as error:  # keep going; one bad arm must not end the sweep
            LOGGER.exception("arm %s failed: %s", run, error)
            record({
                "run": run, "stage": arm.stage, "arm": arm.name, "seed": seed,
                "macro_f1": "", "ci_lower": "", "ci_upper": "", "best_epoch": "",
                "epochs_run": "", "minutes": round((time.monotonic() - started) / 60, 1),
                "overrides": " ".join(arm.overrides), "note": f"FAILED: {error}",
            })

    print(f"Results: {RESULTS.resolve()}")


if __name__ == "__main__":
    main()
