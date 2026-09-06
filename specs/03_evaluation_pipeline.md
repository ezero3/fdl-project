# Specification 03: Shared PyTorch Evaluation Pipeline

## Goal

Provide one architecture-independent evaluation implementation so every researcher compares WM-811K models on the same persisted folds, class encoding, and metric definitions.

## Evaluation protocol

- Use validation for architecture selection, hyperparameter tuning, early stopping, loss selection, sampling decisions, and augmentation decisions.
- Keep test frozen until the final model comparison.
- Evaluate validation and test using their natural distributions, without random augmentation, oversampling, undersampling, or shuffled result ordering.
- Preserve the original dataset row index for every prediction so errors can be traced back to the source wafer map.
- Use macro-F1 as the primary model-selection metric because each of the nine classes must contribute equally despite the dominant `none` class.

## Canonical class encoding

Every model, target tensor, metric table, confusion matrix, prediction file, and checkpoint must use this exact output order:

| Index | Class |
|---:|---|
| 0 | `Center` |
| 1 | `Donut` |
| 2 | `Edge-Loc` |
| 3 | `Edge-Ring` |
| 4 | `Loc` |
| 5 | `Near-full` |
| 6 | `Random` |
| 7 | `Scratch` |
| 8 | `none` |

Unknown spellings and alternative orders must fail explicitly instead of being silently normalized.

## PyTorch interface

- Put the model in evaluation mode with `model.eval()`.
- Disable gradient recording with `torch.inference_mode()`.
- Accept a DataLoader yielding `(inputs, targets, row_indices)` or an equivalent mapping.
- Require integer targets in `[0, 8]` and logits shaped `(batch_size, 9)`.
- Support a final batch smaller than the configured batch size.
- Calculate mean loss over samples rather than averaging batch means.
- Collect targets, predictions, softmax probabilities, confidence, and source row indices on CPU.

## Required metrics

- accuracy;
- balanced accuracy;
- macro-F1 as the primary metric;
- weighted-F1;
- per-class precision, recall, F1, and support with zero-division handling;
- absolute confusion matrix;
- row-normalized confusion matrix.

All per-class outputs and matrices must retain the complete canonical nine-class shape even if a model never predicts one of the classes.

## Persisted artifacts

Each named experiment must write:

- `metrics.json`, including class order, split, device, model class, metadata, and aggregate metrics;
- `per_class_metrics.csv`;
- `predictions.csv`, including row index, true/predicted indices and labels, confidence, and nine probabilities;
- `confusion_matrix.png`;
- `confusion_matrix_normalized.png`.

Existing run artifacts must not be overwritten unless the caller opts in explicitly.

## Validation requirements

Automated tests must cover perfect predictions, an always-`none` classifier, missing predicted classes, the smaller final batch, fixed class order, invalid inputs, unique row indices, sample-weighted loss, mapping batches, and the complete artifact set. A deterministic dummy PyTorch model must exercise the pipeline end to end without requiring WM811K.pkl.
