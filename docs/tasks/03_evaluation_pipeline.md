# Implement the Shared Evaluation Pipeline

Implement one reusable PyTorch evaluation protocol for every WM-811K model.

## Goal

Ensure that all researchers compare models on the same fixed class encoding, persisted data folds, and metric implementations.

## Tasks

- [x] Define and validate the immutable nine-class output order
- [x] Collect logits, predictions, probabilities, targets, losses, and row indices in PyTorch inference mode
- [x] Calculate the shared aggregate and per-class metrics
- [x] Generate absolute and normalized confusion matrices
- [x] Persist JSON, CSV, and PNG comparison artifacts
- [x] Reject malformed batches, outputs, labels, probabilities, and duplicate row indices
- [x] Test perfect, majority-only, and missing-class predictions
- [x] Test a smaller final batch and sample-weighted loss
- [x] Run a deterministic dummy model end to end
- [x] Document the API, DataLoader contract, checkpoint encoding, and split policy
