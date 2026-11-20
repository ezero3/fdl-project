# Task 03: Implement Shared PyTorch Evaluation Pipeline

Implement one architecture-independent evaluation protocol and metric harness for every WM-811K model.

## Goal

Ensure that all models are evaluated and compared on the same fixed class encoding, persisted data folds, and metric definitions.

## Evaluation Protocol & Specifications

- **Protocol discipline**: Validation drives model selection, early stopping, and hyperparameter tuning; the test split remains strictly frozen until final benchmark evaluation.
- **Natural distributions**: Validation and test are evaluated on their natural class distributions without sampling reweighting or synthetic augmentation.
- **Source tracking**: Preserve original DataFrame row index for every prediction to enable error analysis and visual inspection.
- **Primary metric**: **Macro-averaged F1** ($\frac{1}{C}\sum F1_c$) to give equal weight to all nine classes regardless of frequency.
- **Canonical 9-class encoding**: Strict output mapping:
  `0: Center`, `1: Donut`, `2: Edge-Loc`, `3: Edge-Ring`, `4: Loc`, `5: Near-full`, `6: Random`, `7: Scratch`, `8: none`.

## Tasks & Checklist

- [x] Define and validate immutable nine-class output order.
- [x] Collect logits, predictions, probabilities, targets, losses, and row indices in PyTorch inference mode.
- [x] Calculate shared aggregate metrics (Accuracy, Balanced Accuracy, Macro-F1, Weighted-F1).
- [x] Calculate per-class precision, recall, F1, and support with zero-division handling.
- [x] Generate absolute and row-normalized confusion matrices.
- [x] Persist run artifacts: `metrics.json`, `per_class_metrics.csv`, `predictions.csv`, and confusion matrix PNGs.
- [x] Reject malformed batches, invalid probabilities, out-of-range labels, and duplicate row indices.
- [x] Add automated unit tests (perfect predictions, majority classifier, missing classes, non-standard batch sizes).
- [x] Implement deterministic dummy model to verify end-to-end evaluation flow.
- [x] Document DataLoader contracts, artifact schemas, and API usage.

## Deliverables

- Reusable evaluation module in [`src/fdl_project/evaluation/`](../../src/fdl_project/evaluation/).
- Automated test coverage in [`tests/`](../../tests/).
- Minimal runnable example in `examples/evaluate_dummy_model.py`.
- Written technical report: [`docs/reports/evaluation_pipeline.md`](../reports/evaluation_pipeline.md).
