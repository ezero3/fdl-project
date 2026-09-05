# Define Dataset Split

Define a leakage-safe, reproducible, and reusable data-partitioning protocol for all subsequent preprocessing and modeling experiments.

## Goal

Decide which WM-811K observations are used for supervised training, self-supervised pretraining, validation, and final testing, while preserving class coverage and preventing information leakage.

## Tasks (You Can Decide What More to Do)

- [ ] Verify the original trainTestLabel split and its class distributions
- [ ] Check whether lotName groups overlap between the original or candidate splits
- [ ] Decide whether the final experiments use the entire labeled dataset or a subsample
- [ ] Define the role of unlabeled samples and prevent them from entering supervised losses
- [ ] Decide between cross-validation and a fixed train/validation/test split and document the rationale
- [ ] Choose the split proportions and implement stratification with fixed random seeds
- [ ] Decide whether the split should be group-aware for lotName
- [ ] Ensure each supervised class is adequately represented in train, validation, and test
- [ ] Freeze the test set and keep the validation/test distributions unchanged from the training samplers
- [ ] Implement a reusable validation scheme that returns Deterministic split indices
- [ ] Generate split summaries with sample counts, class distributions, and batch counts
- [ ] Validate disjunction, completeness, class coverage, group separation, and determinism
- [ ] Document final splitting
