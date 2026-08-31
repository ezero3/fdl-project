# Dataset Split Strategy

## Objective

This document defines the leakage-safe, reproducible data-partitioning strategy shared by all WM-811K modeling pipelines. It describes how labeled and unlabeled wafer maps must be assigned before preprocessing, augmentation, model selection, or training.

WM-811K contains:

- **811,457 total wafer maps**;
- **172,950 labeled wafer maps**;
- **638,507 unlabeled wafer maps**.

Among the labeled observations, **147,431** have the supervised class `none`, while **25,519** belong to one of the eight failure-pattern classes. `none` is a valid label meaning that no recognized failure pattern was assigned. An unlabeled wafer has an unknown target and must never be converted automatically to `none`.

## Validation protocol

The project uses one fixed **train/validation/test hold-out split** instead of full k-fold cross-validation. Deep learning experiments are computationally expensive, and a fixed split makes comparisons between preprocessing methods, architectures, and training strategies practical and consistent.

The target proportions for the labeled data are:

```text
172,950 labeled wafers
          |
          +-- 70% train
          +-- 20% validation
          +-- 10% test
```

These proportions are targets rather than exact guarantees because complete lots, rather than individual wafers, are assigned to subsets.

The dataset split uses the fixed seed:

```python
RANDOM_SEED = 86
```

The split must be generated once, validated, saved, and reused unchanged by every modeling pipeline. Neural-network initialization seeds may vary in later repeated experiments, but they must not alter the dataset split.

### Why the native split is not used

WM-811K provides an original `trainTestLabel` field, but it is not adopted as the project's validation protocol. Among the 172,950 labeled observations, it assigns only **54,355 wafers (31.4%)** to `Training` and **118,595 wafers (68.6%)** to `Test`, with no separate validation subset. A test partition larger than the training partition is unusual for this project and would leave substantially less labeled data for fitting the models.

The native partitions also have markedly different class distributions. In particular, `none` represents approximately **67.6%** of the native training data and **93.3%** of the native test data. The native training and test partitions have zero overlapping labeled lots, so group overlap is not the reason for rejecting them. They are rejected because they do not provide the approximately stratified 70/20/10 comparison and dedicated validation subset required for model selection, early stopping, and final evaluation. The project therefore constructs and freezes a new split that explicitly preserves `lotName` group separation while adding class coverage, reproducibility, and validation data. The original `trainTestLabel` is retained as metadata and documented for comparison, but it does not determine membership in the new subsets.

## Stratification and class imbalance

The labeled split must preserve the nine-class distribution as closely as possible. This is particularly important for rare classes such as `Near-full` and `Donut` and for the dominant `none` class.

Stratification does not balance the classes. It aims to give train, validation, and test comparable class proportions and to ensure that every class is represented adequately in each subset. Class imbalance will be handled only during model training through train-only techniques such as weighted losses, samplers, or augmentation experiments.

Validation and test must retain their assigned natural distributions. They must not be oversampled, undersampled, or augmented to create additional observations.

## `lotName` as a group constraint

`lotName` is used to constrain the split. It is not a model input and is not itself a target for stratification.

All wafer maps belonging to the same `lotName` must be assigned to the same subset. A wafer-level random split could otherwise produce leakage:

```text
Incorrect:
LOT A
├── wafer 1 -> train
├── wafer 2 -> train
├── wafer 3 -> validation
└── wafer 4 -> test

Correct:
LOT A -> train
LOT B -> validation
LOT C -> test
```

Wafer maps from the same manufacturing lot are often similar and frequently share a dominant `failureType`. If the same lot appeared in training and evaluation data, the model could exploit lot-specific characteristics and produce overly optimistic validation or test results.

The required procedure is therefore an **approximately stratified, group-aware split**:

1. separate labeled and unlabeled observations;
2. summarize labeled class counts for every `lotName`;
3. assign complete lots to `train_lots`, `validation_lots`, or `test_lots`;
4. optimize the assignment toward the 70/20/10 sample targets and similar class proportions;
5. explicitly protect the coverage of rare classes;
6. verify that no `lotName` occurs in more than one subset.

Because lots cannot be divided, exact class and sample proportions are neither expected nor preferred over group isolation.

## Unlabeled data policy

All **638,507 unlabeled wafer maps are retained** as a separate pool. They are not included in supervised stratification, supervised losses, validation metrics, or test metrics. Their final modeling use is intentionally deferred.

They may later support:

1. self-supervised pretraining without invented targets;
2. controlled pseudo-labeling with an already trained supervised model;
3. consistency learning or a teacher-student method.

Depending on computational resources, a deterministic subset of approximately **150,000-200,000** observations may be selected. Any such subset must use a fixed seed and persistent indices. Smaller fixed subsets may be used for debugging and pilot experiments.

### Relationship between labeled and unlabeled lots

The EDA found substantial overlap between labeled and unlabeled `lotName` values:

| Information | Count |
|---|---:|
| Lots with at least one labeled wafer | 10,762 |
| Lots with at least one unlabeled wafer | 41,608 |
| Lots containing both | 6,077 |
| Labeled wafers in mixed lots | 61,182 |
| Unlabeled wafers in mixed lots | 71,896 |
| Unlabeled wafers in unlabeled-only lots | 566,611 |

Approximately **11.3% of all unlabeled wafers** belong to a mixed lot. A mixed lot contains, on average, approximately 10 labeled wafers, 12 unlabeled wafers, and 22 wafers in total.

The known labels in a mixed lot may provide useful contextual information because lots often have a dominant failure type. However, this is probabilistic evidence rather than ground truth. An unlabeled wafer must not automatically inherit the label or dominant class of its lot.

### Leakage rule for unlabeled wafers

The group assignment controls whether an unlabeled wafer is eligible for future training:

```text
Lot assigned to train
├── labeled wafers   -> supervised training
└── unlabeled wafers -> eligible for SSL or pseudo-labeling

Lot assigned to validation or test
├── labeled wafers   -> supervised evaluation
└── unlabeled wafers -> excluded from every training procedure
```

Unlabeled wafers from lots containing no labeled observations may be retained in a separate training-only pool. The split artifact should distinguish these from unlabeled wafers belonging to labeled training lots.

## Persisted split artifacts

The reusable schema should save stable row identifiers or indices for at least:

```text
train
validation
test
unlabeled
```

The `unlabeled` metadata should make it possible to distinguish:

- unlabeled wafers from labeled lots assigned to train;
- unlabeled wafers from completely unlabeled lots;
- unlabeled wafers excluded because their lot belongs to validation or test.

The storage format will be selected during implementation, but it must preserve indices exactly and be practical to load without reopening or duplicating the full image dataset.

## Required validation checks

The implementation must verify and report:

- deterministic reproduction with seed 86;
- disjoint train, validation, and test row indices;
- complete assignment of every labeled observation;
- zero `lotName` overlap between supervised subsets;
- presence and count of all nine classes in every supervised subset;
- sample and lot proportions for each subset;
- class-distribution differences from the full labeled dataset;
- correct isolation of unlabeled wafers associated with validation/test lots;
- absence of unlabeled observations from supervised losses and metrics.

The test set is frozen after these checks and must remain untouched until final evaluation. The validation set may guide architecture selection, hyperparameters, and early stopping. All samplers, loss weights, and augmentations must be derived from or applied to the training set only.
