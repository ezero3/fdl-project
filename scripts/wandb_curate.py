"""Tag (and optionally delete) Weights & Biases runs so the useful ones are findable.

    python scripts/wandb_curate.py                 # dry run, shows the plan
    python scripts/wandb_curate.py --apply         # write the tags
    python scripts/wandb_curate.py --apply --delete-invalid

Tagging beats renaming: a rename breaks every existing link and the W&B UI
filters on tags anyway. Deletion is separate, opt-in, and irreversible --
default to tagging `invalid` and leaving the run in place, since a run that
was trained on the wrong settings is still evidence of what those settings do.
"""

from __future__ import annotations

import argparse
import os
import re

#: Name pattern -> tag, first match wins. The reasoning behind each is what
#: makes this script reviewable rather than a pile of string matches.
RULES: tuple[tuple[str, str, str], ...] = (
    (r"-s(8[78])$", "seeds",
     "repeat seeds, for the spread rather than a new comparison"),
    (r"^baseline_v2-", "invalid-settings",
     "trained before the defaults-inheritance fix: batch 512, lr 1e-3, "
     "40 epochs, patience 5 instead of 256 / 7e-4 / 50 / 7"),
    (r"^baseline_cnn-rotation-(sampler|focal)", "v26-focal",
     "focal series on the model that wins"),
    (r"-p10-s86$", "phase2", "the clean patience-10 pipeline sweep"),
    (r"-s86$", "superseded",
     "patience-5 pass; some runs also carry the wrong data, because a crashed "
     "arm left its W&B run open and the next arm logged into it"),
)

#: Named individually because their *contents* are wrong, not just their
#: settings -- no tag rule can express that.
KNOWN_BAD: dict[str, str] = {
    "baseline_cnn-rotation-s86":
        "mislabeled: this run holds the unweighted-ce arm's data. The rotation "
        "arm crashed with ForkedError and W&B returned the still-open run to "
        "the next arm.",
    "baseline_cnn-224px-p10-s86":
        "crashed when the Colab VM died mid-run.",
}


def classify(name: str) -> str | None:
    for pattern, tag, _ in RULES:
        if re.search(pattern, name):
            return tag
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project",
                        default="vlad-yelisieiev-bicocca-milano-bicocca/wm811k-wafer-defects")
    parser.add_argument("--apply", action="store_true", help="Write the tags.")
    parser.add_argument("--delete-invalid", action="store_true",
                        help="Also delete runs tagged 'invalid-contents'. Irreversible.")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    os.environ.setdefault("WANDB_API_KEY", os.environ.get("WANDB_KEY", ""))

    import wandb

    runs = list(wandb.Api().runs(args.project))
    print(f"{len(runs)} runs in {args.project}\n")

    planned: list[tuple[object, list[str]]] = []
    for run in sorted(runs, key=lambda r: r.name):
        tags = set(run.tags)
        if run.name in KNOWN_BAD:
            tags.add("invalid-contents")
        tag = classify(run.name)
        if tag:
            tags.add(tag)
        if run.state != "finished":
            tags.add("incomplete")
        new = sorted(tags - set(run.tags))
        marker = "+" if new else " "
        print(f" {marker} {run.name:46} {run.state:9} {sorted(tags)}")
        if new:
            planned.append((run, sorted(tags)))

    print(f"\n{len(planned)} runs would gain tags.")
    for name, reason in KNOWN_BAD.items():
        print(f"\n  {name}\n    {reason}")

    if not args.apply:
        print("\nDry run. Pass --apply to write these tags.")
        return

    for run, tags in planned:
        run.tags = tags
        run.update()
    print(f"\nTagged {len(planned)} runs.")

    if args.delete_invalid:
        doomed = [r for r in runs if r.name in KNOWN_BAD]
        for run in doomed:
            run.delete()
        print(f"Deleted {len(doomed)} runs with wrong contents.")
    elif KNOWN_BAD:
        print("Runs with wrong contents are tagged, not deleted. Use "
              "--delete-invalid if you want them gone.")


if __name__ == "__main__":
    main()
