#!/usr/bin/env bash
# Three seeds on the two arms that matter, to replace single-run numbers with
# a mean and a spread.
#
# Why: the identical config scored 0.8800 and 0.8696 on two runs, so the
# 0.010 gap between them is nondeterminism, not signal -- and the best focal
# arm beat its control by only 0.0042. Nothing in that comparison is
# measurable without repeats.
#
#   bash scripts/seed_runs.sh                        # both arms, 6 runs
#   ARMS=control:sampler bash scripts/seed_runs.sh   # control only, 3 runs
set -euo pipefail

DRIVE=/content/drive/MyDrive/BICOCCA/FDL
COMMON=(--override "checkpoint.directory=${DRIVE}/checkpoints"
        --override data.transform_device=cuda
        --override logging.wandb.enabled=true
        --override logging.wandb.project=wm811k-wafer-defects)

for seed in 86 87 88; do
  # Set ARMS=control:sampler to skip the focal comparison and only measure
  # the settled pipeline's spread -- which is the part the report needs.
  for arm in ${ARMS:-control:sampler focal_g1_sampler:focal_g1-sampler}; do
    config="${arm%%:*}"
    label="${arm##*:}"
    run="baseline_cnn-rotation-${label}-s${seed}"
    if [ -f "output/runs/${run}/metrics.json" ]; then
      echo "skip ${run} (already done)"
      continue
    fi
    echo "=== ${run}"
    python scripts/train.py "configs/train/v26_focal_cnn/${config}.yaml" \
      --override "seed=${seed}" --override "name=${run}" "${COMMON[@]}" --overwrite
  done
done

echo
echo "Summary:"
python - <<'PY'
import json, pathlib
import statistics

rows = {}
for path in sorted(pathlib.Path("output/runs").glob("baseline_cnn-rotation-*-s8*/metrics.json")):
    arm = path.parent.name.rsplit("-s", 1)[0]
    rows.setdefault(arm, []).append(json.loads(path.read_text())["macro_f1"])

for arm, scores in sorted(rows.items()):
    spread = statistics.stdev(scores) if len(scores) > 1 else float("nan")
    print(f"  {arm:44} {statistics.mean(scores):.4f} +- {spread:.4f}  (n={len(scores)})")
print("\nA gap between arms smaller than these spreads is not a result.")
PY
