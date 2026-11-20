# Task 02: Dataset Split and Reusable Validation Strategy

Define a leakage-safe, reproducible, and reusable data-partitioning protocol for all subsequent preprocessing and modeling experiments.

## Goal

Decide which WM-811K observations are used for supervised training, validation, final testing, and controlled unlabeled retention, while preserving class coverage and strictly preventing information leakage.

## Partitioning Strategy & Requirements

- **Supervised partition**: Target 70% train, 20% validation, and 10% test across all observations with a valid `failureType`.
- **Determinism**: Fixed random seed **86** ensuring bit-for-bit identical regeneration.
- **Lot-level grouping constraint**: Treat `lotName` as indivisible; all labeled wafers from one lot belong strictly to exactly one supervised subset (zero lot overlap).
- **Class coverage**: Preserve the nine-class distribution approximately across all subsets, ensuring every class is represented in train, validation, and test.
- **Native split policy**: Retain `trainTestLabel` as metadata only; it does not determine the new partition.
- **Ten intermediate folds**: Group labeled lots into 10 stratified, group-disjoint folds, then select combinations (7 train, 2 val, 1 test) by minimizing normalized deviations from target ratios.
- **Unlabeled data retention**:
  - Keep observations without valid labels separate from supervised targets and metrics.
  - Mark unlabeled observations from training lots as eligible for future training-only methods (SSL/pseudo-labeling).
  - Exclude unlabeled observations from validation/test lots from training pools.

## Tasks & Checklist

- [x] Verify original `trainTestLabel` split and class distributions.
- [x] Confirm `lotName` groups and prevent cross-partition lot overlap.
- [x] Define eligibility roles for unlabeled samples (train-eligible vs excluded).
- [x] Choose split proportions (70/20/10) and implement group-aware stratification with seed 86.
- [x] Ensure all 9 supervised classes are present in train, validation, and test.
- [x] Freeze the test set and keep validation/test distributions unweighted.
- [x] Generate deterministic split indices and manifest tables.
- [x] Validate disjunction, completeness, class coverage, group separation, and determinism.
- [x] Document final split methodology and distribution statistics.

## Deliverables

- Executable splitting notebook: [`notebooks/preparation/split_data.ipynb`](../../notebooks/preparation/split_data.ipynb).
- Reusable split artifacts: row indices and manifests under [`data/splits/`](../../data/splits/).
- Automated validation checks asserting disjunction and zero lot leakage.
- Written summary report: [`docs/reports/split_strategy.md`](../reports/split_strategy.md).
