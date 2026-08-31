# Candidate Modeling Pipelines for WM-811K

This document describes three candidate modeling pipelines. All of them must reuse the group-aware 70/20/10 partition and seed 86 defined in [`split_strategy.md`](split_strategy.md).

---

# Pipeline 1 — Fully Supervised CNN Baseline


## Goal

Build a strong and interpretable Deep Learning baseline using only reliable labeled observations.

This should be implemented first because every more advanced pipeline must prove that it improves over this baseline.

## Data used

Use all **172,950 labeled wafers**.

This includes:

```text
147,431 None
25,519 failure-pattern wafers
```

### `None`

**Use all real labeled `None` samples.**

Do not remove the class.

The model should solve the complete 9-class supervised problem.

Because `None` dominates the data, class imbalance must be handled during training.

### Unlabeled

The **638,507 unlabeled wafers are not used in Pipeline 1**.

They are intentionally excluded only to establish a clean supervised benchmark.

## Split

Use:

```text
70% train
20% validation
10% test
```

with:

```text
approximate stratification by class, grouped by lotName
random seed = 86
```

No cross-validation.

## Architecture

A suitable architecture is:

```text
Wafer map
    |
    v
Size normalization
    |
    v
Tensor representation
    |
    v
CNN encoder
    |
    v
Global Average Pooling
    |
    v
Classification head
    |
    v
9 output logits
    |
    v
Center / Donut / Edge-Loc / Edge-Ring /
Loc / Random / Scratch / Near-full / None
```

Possible encoders:

- small custom CNN
- ResNet18
- EfficientNet-B0

A custom CNN gives a simple baseline.

A pretrained ResNet18 or EfficientNet can then be used to test **transfer learning**.

## Class imbalance

Recommended first loss:

```text
Weighted Cross-Entropy
```

Class weights must be computed from the training set only.

A second experiment can compare:

```text
Focal Loss
```

Do not combine too many imbalance techniques immediately, otherwise it becomes difficult to understand which change produced the improvement.

## Training

Use:

- mini-batch training
- AdamW
- learning-rate scheduler
- early stopping based on validation Macro-F1
- checkpoint of the best validation model

## Evaluation

Report:

- Accuracy
- Macro-F1
- Weighted-F1
- Precision per class
- Recall per class
- F1 per class
- Confusion matrix

### Why Macro-F1?

Macro-F1 calculates F1 separately for every class and gives every class the same importance.

Therefore:

```text
Near-full
```

has the same weight as:

```text
None
```

in the final average.

This makes Macro-F1 much more informative than accuracy alone.

## Strengths

- Simple
- Reproducible
- Uses all reliable labeled information
- Includes `None`
- Strong benchmark for later pipelines

## Main limitation

The unlabeled portion is unused.

---

# Pipeline 2 — Self-Supervised Pretraining + Supervised Fine-Tuning

## Goal

Exploit the huge unlabeled part of WM-811K without inventing labels.

This is the recommended advanced pipeline if the project wants to make serious use of the 638,507 unlabeled wafer maps.

## Data used

### SSL pretraining

Use:

```text
638,507 unlabeled wafers
```

plus optionally the **training portion of the labeled wafers with their labels hidden**.

Recommended configuration:

```text
638,507 unlabeled
+
approximately 121,065 labeled-training wafers
(labels temporarily ignored)
```

Do **not** include validation or test wafers in self-supervised pretraining.

This avoids leakage.

### `None`

All `None` wafers belonging to the labeled **training split** can participate in self-supervised pretraining with their labels ignored.

During fine-tuning, all labeled `None` training samples are used normally as the `None` class.

### Unlabeled

In Pipeline 2 the unlabeled data are a central resource.

They teach the encoder the structure of wafer maps before the classifier sees supervised labels.

## Phase A — Self-supervised pretraining

Recommended approach:

```text
Contrastive Learning
```

For example, a SimCLR-style framework adapted to wafer maps.

```text
               wafer map
                   |
            two valid augmentations
              /             \
             v               v
          view A           view B
             |               |
             +---- CNN -------+
                   encoder
                 /       \
                v         v
           embedding A embedding B
                 \       /
                  \     /
             contrastive loss
```

The model does not know whether the wafer is `Center`, `Scratch`, `None`, etc.

Its task is to learn a useful wafer representation.

## Augmentations

Wafer maps are not normal photographs.

Only transformations that preserve failure-pattern semantics should be considered.

Candidates:

- rotations
- flips
- small translations
- controlled masking
- mild spatial perturbations

Natural-image transformations such as arbitrary color jitter are not automatically meaningful here.

## Phase B — Fine-tuning

After SSL pretraining:

```text
Self-supervised encoder
           |
           + remove SSL projection head
           |
           + add 9-class classification head
           |
           v
Supervised fine-tuning
```

Fine-tuning uses only the labeled training split.

Weighted Cross-Entropy or Focal Loss can again be used for class imbalance.

## Split

The labeled split remains identical to Pipeline 1:

```text
70% train
20% validation
10% test
```

Only the unlabeled pool declared training-eligible by `split_strategy.md` may be used.

Validation and test metrics are calculated only on labeled wafers.

## Architecture

```text
                    WM-811K
                       |
        +--------------+---------------+
        |                              |
        v                              v
638,507 unlabeled              labeled TRAIN
                               labels hidden
        |                              |
        +--------------+---------------+
                       |
                       v
            Self-supervised pretraining
                       |
                       v
               pretrained encoder
                       |
                       v
              classification head
                       |
                       v
           supervised fine-tuning
                       |
                       v
                 9-class model
                       |
             +---------+---------+
             |                   |
             v                   v
        validation             test
```

## Critical comparison

The clean scientific comparison is:

```text
Pipeline 1:
same CNN from random initialization

VS

Pipeline 2:
same CNN encoder pretrained with SSL
```

If Pipeline 2 performs better, the improvement can reasonably be attributed to the unlabeled-data pretraining.

## Strengths

- Uses hundreds of thousands of unlabeled wafers
- Does not invent labels
- Strongly aligned with Deep Learning
- Produces a clear experimental research question

## Limitations

- More computationally expensive
- Augmentation design must be justified
- More complex than a supervised baseline

---

# Pipeline 3 — Semi-Supervised Teacher-Student

## Goal

Use unlabeled wafers directly during classification training by generating reliable pseudo-labels and enforcing prediction consistency.

This is more aggressive than self-supervised pretraining because the unlabeled data directly influence the classifier decision boundaries.

## Data used

### Labeled data

Use all **172,950 labeled wafers** through the same:

```text
70 / 20 / 10
```

stratified split.

### `None`

Keep **all real labeled `None` samples** in train, validation, and test.

For training, start by retaining all real `None` observations and handle imbalance with:

- weighted loss
- balanced batch sampling if needed

Do not silently remove `None`.

### Unlabeled

Use training-eligible unlabeled samples directly during semi-supervised learning. Unlabeled wafers from validation or test lots remain excluded.

Two practical options:

### Option A — full unlabeled pool

```text
638,507 unlabeled wafers
```

Use this if GPU time and storage permit.

### Option B — reproducible unlabeled subset

Start with:

```text
150,000–300,000 unlabeled wafers
```

sampled once with a fixed random seed.

Because labels are unknown, this sample cannot be stratified by target class.

The selected IDs should be saved so every experiment uses the same subset.

## Phase A — Train the Teacher

Start with a reliable supervised classifier.

The Teacher can be:

- the best Pipeline 1 model
- or the fine-tuned SSL model from Pipeline 2

```text
labeled TRAIN
     |
     v
Teacher CNN
```

## Phase B — Pseudo-labeling

Use the Teacher to predict labels for unlabeled wafers.

Example:

```text
wafer A -> Edge-Ring  confidence 0.99
wafer B -> Scratch    confidence 0.97
wafer C -> None       confidence 0.54
```

Only sufficiently confident predictions are accepted.

For example:

```text
confidence >= 0.95
```

The exact threshold must be selected using validation behavior, not the test set.

A **pseudo-label** is a model-generated estimated label, not human ground truth.

## Important problem: pseudo-labeled `None`

Because `None` dominates the real labeled dataset, the Teacher may become biased toward predicting `None`.

If every high-confidence pseudo-`None` is accepted, the unlabeled training pool could become even more imbalanced.

Therefore:

- monitor pseudo-label class distribution;
- consider class-specific confidence thresholds;
- cap the number of pseudo-`None` samples per epoch or batch;
- prevent pseudo-`None` from dominating the unlabeled loss.

This does **not** mean deleting real labeled `None` observations.

Real `None` labels are trusted.

Pseudo-`None` labels are uncertain predictions and should be controlled.

## Phase C — Student training

The Student learns from:

```text
real labeled data
+
high-confidence pseudo-labeled data
```

Conceptually:

```text
TOTAL LOSS
    =
supervised classification loss
    +
lambda * consistency loss
```

## Consistency learning

For one unlabeled wafer:

```text
                  unlabeled wafer
                    /        \
                   /          \
          weak augmentation  strong augmentation
                 |                 |
                 v                 v
              Teacher           Student
                 |                 |
                 v                 v
           pseudo-label       prediction
                 \                 /
                  \               /
                   consistency loss
```

The Student should produce a prediction compatible with the Teacher's high-confidence pseudo-label even under a stronger valid augmentation.

This is similar to methods such as **FixMatch**.

## Split

Only labeled samples define supervised evaluation:

```text
172,950 labeled
    |
    +-- 70% train
    +-- 20% validation
    +-- 10% test
```

Unlabeled samples are used only during training.

They cannot be used to compute supervised validation or test accuracy because their true labels are unknown.

## Architecture

```text
                    LABELED TRAIN
                         |
                         v
                    Teacher CNN
                         |
                         | predictions
                         v
                  UNLABELED POOL
                         |
                  confidence filter
                         |
                         v
                    pseudo-labels
                         |
              +----------+-----------+
              |                      |
              v                      v
       real labeled data      pseudo-labeled data
              |                      |
              +----------+-----------+
                         |
                         v
                    Student CNN
                         |
             supervised + consistency loss
                         |
                         v
                 final classifier
                         |
                 validation / test
```

## Strengths

- Uses unlabeled data directly for classification
- Can exploit a very large unlabeled pool
- Teacher confidence provides a quality filter
- Strong research value

## Limitations

- More complex
- Wrong pseudo-labels can reinforce model errors
- `None` dominance must be explicitly monitored
- Confidence threshold and consistency weighting add hyperparameters

---

# Comparison

| Property | Pipeline 1 | Pipeline 2 | Pipeline 3 |
|---|---|---|---|
| Main idea | Supervised CNN | Self-supervised pretraining + fine-tuning | Semi-supervised Teacher-Student |
| Labeled data | All labeled | All labeled for fine-tuning | All labeled |
| Real `None` | All retained | All retained | All retained |
| Unlabeled | Not used | Used for SSL pretraining | Used directly in classification training |
| Final classes | 9 | 9 | 9 |
| Split | 70/20/10 group-aware and approximately stratified | Same split | Same split |
| Cross-validation | No | No | No |
| Random seed | Fixed | Fixed | Fixed |
| Imbalance handling | Weighted CE / Focal Loss | Weighted CE / Focal Loss in fine-tuning | Weighted loss + pseudo-label control |
| Main model | CNN | SSL-pretrained CNN | Teacher + Student CNN |
| Complexity | Low | Medium | High |

---

# Recommended experimental progression

The three pipelines are best treated as a progression rather than unrelated alternatives.

```text
PIPELINE 1
Fully supervised CNN
        |
        | establishes baseline
        v

PIPELINE 2
Self-supervised pretraining
+ supervised fine-tuning
        |
        | tests the value of unlabeled representation learning
        v

PIPELINE 3
Teacher-Student semi-supervised learning
        |
        | tests whether unlabeled samples can directly
        | improve classification
        v

FINAL COMPARISON
```

## Development order

1. Implement and save the reusable stratified split.
2. Implement Pipeline 1 and verify preprocessing, batching, training, and metrics.
3. Implement Pipeline 2 using the same encoder architecture where possible.
4. Compare random initialization against SSL pretraining.
5. Only after the first two are stable, implement Pipeline 3.

---

# Final recommendation

If time and compute allow only two substantial approaches, prioritize:

```text
Pipeline 1:
Supervised imbalance-aware CNN

              VS

Pipeline 2:
Self-supervised pretrained CNN
+ supervised fine-tuning
```

This gives the project a very clear scientific question:

> **Can the large unlabeled portion of WM-811K improve Deep Learning classification when used for representation learning instead of being discarded?**

Pipeline 3 is the most ambitious extension once the first two pipelines are reliable.

