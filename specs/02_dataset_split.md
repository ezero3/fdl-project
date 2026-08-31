# Specification 02: Dataset Split and Reusable Validation Strategy

## Goal

Define and implement one leakage-safe, deterministic, and reusable WM-811K partition for supervised training, model validation, final testing, and the controlled retention of unlabeled observations.

## Required partitioning strategy

- Use every observation with a valid `failureType` in the supervised partitioning procedure.
- Create one fixed hold-out split targeting 70% train, 20% validation, and 10% test.
- Use random seed 86 and make repeated executions return identical indices.
- Preserve the nine-class distribution approximately across all supervised subsets.
- Treat `lotName` as an indivisible group: every labeled wafer from one lot must belong to exactly one supervised subset.
- Give priority to group isolation and class coverage when exact sample proportions conflict with the lot constraint.
- Ensure every supervised class is represented in train, validation, and test.
- Retain `trainTestLabel` only for comparison and traceability; it must not determine the new partition.

## Intermediate-fold requirements

- Construct ten approximately stratified, group-disjoint intermediate folds from labeled lots.
- Base fold balance on per-lot class counts and total labeled observations.
- Use a deterministic ordering and tie-breaking procedure controlled by the fixed seed.
- Select seven complete folds for train, two for validation, and one for test.
- Select the final fold combination by minimizing normalized deviations from the desired sample and per-class proportions.
- Do not use the ten folds as a cross-validation protocol.

## Unlabeled-data requirements

- Keep all observations without a valid `failureType` separate from supervised targets, losses, stratification, and evaluation metrics.
- Retain unlabeled observations instead of converting them to `none` or deleting them.
- Mark unlabeled observations from training lots as eligible for future training-only methods.
- Mark unlabeled observations from completely unlabeled lots as a separate training-eligible pool.
- Exclude unlabeled observations associated with validation or test lots from every training procedure.
- Preserve enough metadata to distinguish training-eligible and excluded unlabeled roles.
- Allow future deterministic unlabeled subsets for SSL or pseudo-labeling without changing the supervised split.

## Validation requirements

The implementation must assert:

- pairwise disjunction of train, validation, and test indices;
- complete and unique assignment of all labeled observations;
- zero `lotName` overlap between supervised subsets;
- presence of all nine classes in every supervised subset;
- approximate agreement with the target sample and class proportions;
- deterministic regeneration with seed 86;
- complete assignment of unlabeled observations to eligibility roles;
- absence of validation/test-lot observations from the eligible unlabeled training pool.

## Persisted artifacts

The implementation must save:

- deterministic train, validation, and test row indices;
- indices for all unlabeled observations;
- separate indices for training-eligible and excluded unlabeled observations;
- a labeled split manifest containing stable row identifiers, `lotName`, `failureType`, native split metadata, and new subset membership;
- supervised class-count and unlabeled-role summaries;
- a machine-readable configuration containing the seed, target proportions, group column, target column, and selected folds;
- concise instructions for reconstructing subsets from the original dataset.

The original wafer-map arrays should not be duplicated in the split artifacts.

## Evaluation protocol

- Use train for model fitting and for deriving any future loss weights, samplers, or augmentation policies.
- Use validation for model selection, hyperparameter tuning, and early stopping.
- Freeze test after split validation and use it only for final evaluation.
- Do not oversample, undersample, or augment validation and test.
- Require every modeling pipeline to reuse the same persisted split artifacts.

## Deliverables

- An executable, documented dataset-splitting notebook under `notebooks/`.
- Reusable split artifacts under `data/splits/`.
- Automated validation checks that fail on leakage, incomplete assignment, missing class coverage, or non-determinism.
- A report describing the implemented method, final distributions, unlabeled-data roles, and validation outcomes.

Sampling within training batches and data augmentation experiments are outside this specification; they may be selected later without changing the persisted split.
