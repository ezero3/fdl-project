# Investigate Class-Imbalance Handling

Select a train-only class-imbalance policy through controlled PyTorch experiments on the fixed WM-811K splits.

## Goal

Improve minority-pattern recognition without changing the natural validation/test distributions or mistaking high majority-class accuracy for useful multiclass performance.

## Tasks

- [x] Quantify the complete training imbalance in canonical class order
- [x] Define immutable and mutually isolated imbalance configurations
- [x] Implement inverse-square-root and effective-number class weights
- [x] Implement deterministic weighted sampling with replacement
- [x] Implement and validate multiclass focal loss
- [x] Build a compact spatial CNN as a controlled experimental instrument
- [x] Keep preprocessing, initialization, optimizer, budget, and validation fixed across interventions
- [x] Screen unweighted CE, two weighted losses, weighted sampling, and focal loss with seed 86
- [x] Confirm the baseline and top two interventions over seeds 86, 87, and 88
- [x] Select by mean validation macro-F1 and report variability and per-class effects
- [x] Process the full 121,063-row train and 34,591-row validation splits without test access
- [x] Preserve natural, ordered, unweighted validation and test DataLoaders
- [x] Implement the selected inverse-square-root sampler as the reusable training default
- [x] Store checkpoint-safe configuration, counts, weights, class order, and row-index hashes
- [x] Save complete run, strategy, epoch, class, sampling, prediction, and confusion-matrix artifacts
- [x] Add automated tests for loss, weights, sampler, model, loaders, and training
- [x] Execute and verify the analysis notebook
- [x] Document protocol, result, limitations, and future-model usage
