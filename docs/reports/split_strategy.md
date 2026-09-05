# WM-811K Dataset Split Report

This report summarizes the methodology and results produced by [`notebooks/02_split_data.ipynb`](../../notebooks/02_split_data.ipynb). The notebook and the generated files under [`data/splits/`](../../data/splits/) are the source of truth for the reported split.

## Objective

The split was designed to provide one reproducible train/validation/test partition for all subsequent modeling pipelines. It had to satisfy three competing requirements:

- reserve approximately 70% of labeled observations for training, 20% for validation, and 10% for final testing;
- preserve the highly imbalanced nine-class distribution as closely as possible;
- keep every `lotName` entirely within one subset to prevent group leakage.

The split uses fixed random seed **86**. It is a single hold-out protocol, not ten-fold cross-validation. The intermediate folds are only a mechanism for constructing the final 70/20/10 partition.

## Why the native split was replaced

The original `trainTestLabel` field assigns **54,355 labeled wafers (31.4%)** to training and **118,595 (68.6%)** to test. It provides no validation subset, and its test portion is unusually larger than its training portion.

Its class distributions also differ substantially: `none` represents approximately **67.6%** of native training observations and **93.3%** of native test observations. Although the native training and test lots do not overlap, this allocation is unsuitable for the project's model-selection and final-evaluation protocol. The native label was therefore retained as metadata but not used to determine the new split.

## Role of `lotName`

`lotName` is a grouping constraint, not a model feature. Wafer maps from the same manufacturing lot frequently share similar process conditions and a dominant failure type. A wafer-level random split could consequently place highly related maps in training and evaluation data, leading to overly optimistic metrics.

The notebook first aggregates the labeled observations into a lot-by-class matrix:

```text
one row    = one complete lotName
one column = one supervised failureType
one value  = labeled wafers of that class in that lot
```

The algorithm assigns rows of this matrix, rather than individual wafers. Once a lot is placed in a fold, all its labeled observations remain together.

## Construction of the ten intermediate folds

The 10,762 labeled lots were distributed across ten approximately stratified intermediate folds. Lots with comparatively difficult class compositions, rare-class content, or larger labeled sample counts were considered first. Seed 86 provides deterministic tie-breaking.

For every unassigned lot, the algorithm evaluates its temporary placement in each fold. The placement score combines:

1. dispersion of the nine relative class distributions across folds;
2. dispersion of the total labeled sample counts across folds.

Class dispersion receives the primary weight. This prevents the dominant `none` class from concealing poor allocation of rare classes such as `Near-full` or `Donut`.

After constructing the ten group-disjoint folds, the notebook evaluates every possible choice of:

- one fold for test;
- two folds for validation;
- the remaining seven folds for train.

Each candidate is compared with the desired per-class and total-sample proportions. The combination with the smallest normalized squared deviation is selected.

The final fold assignment was:

```text
Train:      folds 0, 2, 3, 4, 5, 7, 8
Validation: folds 1, 9
Test:       fold 6
```

The final objective score was **0.00000799**, indicating a very small normalized deviation from the targets.

## Final supervised split

| Subset | Labeled wafers | Share | Distinct lots |
|---|---:|---:|---:|
| Train | 121,063 | 69.999% | 7,537 |
| Validation | 34,591 | 20.001% | 2,149 |
| Test | 17,296 | 10.001% | 1,076 |
| **Total** | **172,950** | **100%** | **10,762** |

The class counts are:

| Class | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| `Center` | 3,006 | 859 | 429 | 4,294 |
| `Donut` | 389 | 111 | 55 | 555 |
| `Edge-Loc` | 3,632 | 1,038 | 519 | 5,189 |
| `Edge-Ring` | 6,776 | 1,936 | 968 | 9,680 |
| `Loc` | 2,516 | 718 | 359 | 3,593 |
| `Near-full` | 104 | 30 | 15 | 149 |
| `Random` | 606 | 173 | 87 | 866 |
| `Scratch` | 835 | 239 | 119 | 1,193 |
| `none` | 103,199 | 29,487 | 14,745 | 147,431 |

Every class is present in every subset. For all classes, the observed allocation is very close to 70/20/10 despite the indivisible-lot constraint.

## Unlabeled data

All **638,507 unlabeled observations** were retained separately. They did not influence supervised stratification and are not valid inputs to supervised losses or evaluation metrics.

Their relationship with the labeled-lot assignment determines their future role:

| Unlabeled role | Observations | Policy |
|---|---:|---|
| Completely unlabeled lots | 566,611 | Eligible for future training-only SSL or pseudo-labeling |
| Labeled lots assigned to train | 50,635 | Eligible for future training-only SSL or pseudo-labeling |
| Labeled lots assigned to validation | 14,061 | Retained but excluded from training |
| Labeled lots assigned to test | 7,200 | Retained but excluded from training |

In total, **617,246 unlabeled wafers are training-eligible**, while **21,261 are excluded** because their lots belong to validation or test. This preserves lot-level isolation even if a later pipeline uses self-supervised learning or pseudo-labels.

No unlabeled wafer inherits the known class or dominant class of its lot. Such information is contextual evidence, not ground truth.

## Validation performed

The notebook verifies that:

- train, validation, and test row indices are pairwise disjoint;
- all 172,950 labeled observations are assigned exactly once;
- train, validation, and test `lotName` sets are pairwise disjoint;
- all nine supervised classes occur in every subset;
- rerunning the fold construction with seed 86 returns identical assignments;
- all 638,507 unlabeled indices are assigned to exactly one eligibility role;
- no training-eligible unlabeled wafer belongs to a validation or test lot.

All checks passed.

## Reusable artifacts

The generated files store indices and compact metadata rather than duplicating the large wafer-map arrays:

- `train_indices.npy`, `validation_indices.npy`, and `test_indices.npy` define the supervised subsets;
- `unlabeled_indices.npy` retains every unlabeled observation;
- `unlabeled_train_eligible_indices.npy` defines the allowed future unlabeled training pool;
- `unlabeled_excluded_indices.npy` protects validation/test lots;
- `labeled_split_manifest.csv` records row, lot, class, native split, and new membership;
- `class_counts.csv`, `unlabeled_role_counts.csv`, and `split_config.json` document the result.

All model pipelines must reuse these artifacts. The test set is frozen and must not be used for architecture selection, hyperparameter tuning, loss selection, sampling decisions, or early stopping. Sampling and augmentation choices remain train-only modeling decisions and do not alter this split.
