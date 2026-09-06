# Specification 05: Class-Imbalance Handling

## Goal

Select and implement one reproducible train-only intervention that improves recognition of minority WM-811K failure patterns without changing the persisted split, contaminating model selection with test data, or hiding trade-offs behind overall accuracy.

## Invariants

- Use all 121,063 labeled training observations and the fixed task-02 row indices.
- Use all 34,591 validation observations in their natural distribution for model selection.
- Never load or inspect the frozen test split during this task.
- Reuse the task-04 `3 x 64 x 64` preprocessing and task-03 canonical class order and evaluation metrics.
- Derive every class count, loss weight, and sample weight from training targets only.
- Never apply weighted sampling, oversampling, undersampling, or class weights to validation or test.
- Keep individual experiment seeds deterministic and preserve original source-row indices.

## Controlled model

Use one compact spatial CNN as an experimental instrument rather than a final architecture. Every intervention must use the same 156,937-parameter model, initialization seed, optimizer, preprocessing, batch size, maximum epoch budget, validation procedure, and checkpoint-selection metric.

Select the best epoch within each run by validation macro-F1. The unweighted validation cross-entropy must remain comparable across every strategy.

## Candidate interventions

Screen these isolated strategies with seed 86:

1. natural shuffling with unweighted cross-entropy;
2. inverse-square-root class-weighted cross-entropy;
3. effective-number class-weighted cross-entropy with beta 0.9999;
4. inverse-square-root weighted sampling with replacement and unweighted cross-entropy;
5. unweighted focal loss with gamma 2.

Do not combine loss reweighting and weighted sampling in the comparison. Preserve the natural training epoch length for weighted sampling.

Select the best two non-baseline candidates by validation macro-F1, using balanced accuracy as the deterministic secondary key. Confirm those candidates and the unweighted baseline over seeds 86, 87, and 88. Select the default by mean validation macro-F1 and report its sample standard deviation.

## Selected default

- Use weighted random sampling on training only.
- Give each observation from class `c` weight `1 / sqrt(n_c)`, where `n_c` is its training support.
- Draw exactly 121,063 observations per epoch with replacement.
- Use deterministic generator seed 86 unless an experiment specifies another recorded seed.
- Pair the sampler with ordinary unweighted cross-entropy; do not apply class correction twice.
- Keep validation and test ordered, unweighted, unmodified, and complete.

## Reusable implementation

- Represent each intervention with an immutable validated configuration.
- Provide validated train-count and normalized class-weight calculations.
- Provide a numerically stable multiclass focal loss.
- Provide deterministic sampler and train-only DataLoader builders.
- Reject missing classes, invalid targets/counts, non-finite parameters, mismatched counts, simultaneous sampling and loss weighting, and use on non-training splits.
- Provide JSON-safe imbalance metadata for checkpoints and experiment reports.

## Validation and artifacts

Automated tests must cover configurations, counts, weights, focal-loss equivalence at gamma zero, gradients, deterministic sampling, task-04 one-hot collation, train-only enforcement, baseline output shape, deterministic loaders, training history, and invalid inputs.

The complete experiment must persist run-level, strategy-level, epoch-level, class-level, and weight-level CSV files; machine-readable metadata; comparison figures; and the full standard validation artifacts for a representative selected run. The representative seed must be closest to the selected strategy's mean, not cherry-picked as its maximum.

An executed analysis notebook and written report must document the decision and explicitly state that the test set was not used.
