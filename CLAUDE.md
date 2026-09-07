# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

University group project (Foundations of Deep Learning, Milano-Bicocca 2025/2026): multiclass classification of failure patterns in the **MIR WM-811K** wafer-map dataset (811,457 maps, ~172,950 labeled, severely imbalanced). Deliverables, deadlines, and presentation rules are in `docs/requirements/project-requirements.md`.

## Environment and commands

`uv` manages the environment (Python pinned in `.python-version`); always run Python through it.

```bash
uv sync                                   # create/refresh .venv from uv.lock
uv run pytest                             # all tests (testpaths = tests/, addopts = -ra)
uv run pytest tests/test_evaluation.py    # one file
uv run pytest tests/test_evaluation.py::test_name   # one test
uv run pytest -k "confusion"              # by keyword
uv run python examples/evaluate_dummy_model.py --overwrite   # end-to-end pipeline smoke test
uv add <pkg> / uv add --dev <pkg>         # commit both pyproject.toml and uv.lock
```

The dataset file `data/MIR-WM811K/WM811K.pkl` is gitignored and must be downloaded separately; tests must not depend on it.

## Non-negotiable invariants

These are enforced in code and by specs; violating them silently corrupts cross-model comparability.

**Class encoding** (`src/fdl_project/constants.py`): the nine-class output order is frozen — `Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Random, Scratch, none`. Note the raw-dataset spellings `Near-full` and `none` (lowercase) — these are literal label strings, not display names. Alternative spellings or orderings must raise, never be normalized. Checkpoints carry `class_encoding_metadata()` and are validated with `validate_checkpoint_class_names()`.

**Data splits** (`data/splits/`): a group-aware (by `lotName`) 70/20/10 partition generated once by `notebooks/02_split_data.ipynb` with seed 86. The `.npy` files hold *row indices* into the pickle, not copies of the maps. Never regenerate, re-shuffle, or derive a different split per model — load the persisted indices. Unlabeled maps belonging to validation/test lots are in `unlabeled_excluded_indices.npy` and are ineligible for training. Rationale in `split_strategy.md` / `docs/reports/split_strategy.md`.

**Evaluation protocol** (`src/fdl_project/evaluation/`): validation drives every selection decision (architecture, hyperparameters, early stopping, loss, sampling, augmentation); the test split stays frozen until the final comparison. Macro-F1 is the primary metric because `none` dominates. Validation/test are evaluated at their natural distribution — no augmentation, resampling, or reordering. Every prediction retains its source dataset row index for error traceability.

**Wafer maps are categorical, not images**: cells are `0` = background/no die, `1` = functional die, `2` = defective die. Transforms (resize, interpolation, normalization) must not blend these into meaningless fractional states. Maps have 632 distinct shapes, so padding/resizing choices matter.

## Architecture

`src/fdl_project/` is an installed package (hatchling, `packages = ["src/fdl_project"]`) whose public API is re-exported from `__init__.py` — import as `from fdl_project import evaluate_model`, never by relative path from a notebook. It is split by component:

- `constants.py` — the frozen class encoding plus encode/decode/validate helpers. Everything else depends on it.
- `config/` — `schema.py` (frozen dataclasses, one per config section, validated in `__post_init__`), `loader.py` (YAML deep-merged over `configs/train/defaults.yaml`; **unknown keys raise**), `registry.py` (name → builder tables for models, optimizers, schedules). No Hydra — considered and rejected in `docs/design-notes.md` §12.
- `data/` — `datasets.py`, `preprocessing.py`, `imbalance.py`, `augmentation.py` (dihedral subsets + free-angle rotation; train splits only — passing it to validation/test raises).
- `models/` — ours from scratch (`baseline_cnn.py`, `wafer_resnet.py`, `inception.py`, `dilated.py`, `densenet.py`, `vit.py`) plus `pretrained.py` and `attention.py` (CBAM). Every model exposes exactly `encoder` and `head` so config regexes like `^encoder\.` stay stable across architectures. Registry names ending `_style` are ours; a plain torchvision name (`resnet18`) means ImageNet weights. Token models (`vit_style`, `vit_b_16`, `vit_b_32`, `swin_t`) refuse `attention: cbam` — they are already attention models.
- `training/` — `seed.py`, `optim.py`, `checkpoint.py`, `callbacks.py`, `loop.py`, plus `runner.py`/`inference.py` which wire a config into a complete run.
- `evaluation/` — `metrics.py` (collect → evaluate), `bootstrap.py` (percentile CIs), `artifacts.py` (`save_evaluation_results`), `postprocessing.py` (TTA, per-class decision weights tuned on validation only).
- `analysis/` — finished offline studies (`preprocessing_analysis.py`, `imbalance_experiment.py`).

The evaluation flow: `collect_predictions` (model.eval + `torch.inference_mode`, sample-weighted mean loss, CPU collection) → `evaluate_predictions` (metrics, per-class table, absolute and row-normalized confusion matrices, always full nine-class shape even for unpredicted classes) → `evaluate_model` (both) → `save_evaluation_results` (writes `metrics.json`, `per_class_metrics.csv`, `predictions.csv`, `confusion_matrix.png`, `confusion_matrix_normalized.png` under `output/<root>/<run_name>/`, refusing to overwrite without `overwrite=True`).

## Running experiments

```bash
uv run python scripts/train.py configs/train/baseline_cnn_64.yaml
uv run python scripts/train.py configs/train/smoke_test.yaml --overwrite   # 2 epochs, CPU, small slice
uv run python scripts/train.py <config> --override trainer.max_epochs=2 --device cpu
uv run python scripts/inference.py --checkpoint output/runs/<name>/best_model.pt --split validation
```

Artifacts go to `output/runs/<config name>/`, checkpoints to `<checkpoint.directory>/<config name>/`. `trainer.max_epochs` defaults to **40**; the 4-epoch budget lives only inside `analysis/imbalance_experiment.py`, where it is a deliberate screening budget. `batch_size` (256), `num_workers` (8) and the AdamW learning rate (7e-4) come from measurements in `docs/phase0-results.md` on a Colab T4 — re-measure on different hardware rather than assuming them. Training runs fp16 AMP; **validation runs fp32** (no autocast in `collect_predictions`). Call `seed_everything` before creating any CUDA tensor — it sets `CUBLAS_WORKSPACE_CONFIG`, which has no effect once CUDA is initialized.

Two invariants worth restating because they are easy to break from a config: a `param_groups` pattern that matches no parameter **raises** rather than silently folding into the default group, and `scripts/inference.py` refuses `--split test` without `--final-test-evaluation`.

## Planning documents

Read these before proposing experiments; they exist so decisions are not re-derived.

- `docs/ROADMAP.md` — the task list, P0–P3, with what is done marked inline.
- `docs/design-notes.md` — why each choice was made. Read before changing one.
- `docs/training-plan.md` — the four phases: measure cost, pick a test-bed, sweep the pipeline on one model, then compare architectures on the settled pipeline. The rule everything follows: **anything that is not the model is compared on one fixed cheap model.**
- `docs/experiment-grid.md` — every model × every option the pipeline can express, as fillable grids. The single page to open when choosing what to run next.
- `docs/phase0-results.md` — measured epoch costs, batch/worker curves, GPU guidance.

## Repository conventions

Each work item flows through four artifacts sharing a numeric prefix: `specs/NN_*.md` (the contract), `docs/tasks/NN_*.md` (checklist), `notebooks/NN_*.ipynb` (executed analysis, committed with outputs), `docs/reports/*.md` (the written result). Generated figures/tables go to `output/`; trained weights to `trained-models/`. Read the matching spec before changing a module — the specs, not the code, are the source of truth for these invariants.

Work happens on `feature/<slug>` branches merged via PR, one per GitHub issue, with Conventional Commit + emoji subjects (see the `commit-message` and `spec` skills). Project skills live in `.agents/skills/` and are symlinked as `.claude/skills`; `wafer-map-data-analyst` carries the domain-analysis conventions for anything touching WM-811K structure or EDA.

A `PreToolUse` hook (`.claude/hooks/block_dangerous_commands.sh`) hard-blocks destructive shell commands — file removal, hard resets, forced cleans, force-pushes to main, piping remote content to a shell. It matches on the command text, so even a heredoc or a script that merely *contains* those strings is rejected; write such content with the Write tool instead of via the shell.
