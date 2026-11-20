# Task 05: Investigate Class-Imbalance Handling

Select and implement a reproducible train-only class-imbalance policy through controlled PyTorch experiments on fixed WM-811K splits.

## Goal

Improve recognition of minority defect patterns (e.g. `Near-full`, `Donut`) without altering natural validation/test distributions or relying on overall accuracy.

## Experimental Invariants & Controlled Protocol

- **Split discipline**: Training derives counts and weights using all 121,063 labeled training samples; validation (34,591 samples) and test (17,296 samples) remain unmodified, ordered, and unweighted.
- **Controlled baseline model**: Use a compact 156,937-parameter CNN (`BaselineCNN`) as an identical experimental instrument across all interventions (same optimizer, batch size, epochs, and initialization).
- **Candidate strategies screened (seed 86)**:
  1. Natural shuffling + unweighted Cross-Entropy (baseline)
  2. Inverse-square-root class-weighted Cross-Entropy
  3. Effective-number class-weighted Cross-Entropy ($\beta=0.9999$)
  4. Inverse-square-root weighted sampling with replacement + unweighted Cross-Entropy
  5. Unweighted Focal Loss ($\gamma=2$)
- **Confirmation across seeds**: Validate baseline and top two candidates across seeds 86, 87, and 88.
- **Selected default**: Inverse-square-root weighted sampling ($w_c \propto 1 / \sqrt{N_c}$) with replacement, keeping epoch length fixed at 121,063 samples paired with unweighted Cross-Entropy.

## Tasks & Checklist

- [x] Quantify training imbalance in canonical class order (992:1 ratio).
- [x] Implement validated inverse-square-root and effective-number loss weights.
- [x] Implement deterministic weighted random sampler with replacement.
- [x] Implement multiclass focal loss.
- [x] Build and fix `BaselineCNN` as the controlled experimental benchmark.
- [x] Screen all candidate interventions with seed 86.
- [x] Confirm baseline and top strategies over seeds 86, 87, and 88.
- [x] Select winner based on mean validation Macro-F1 (with Balanced Accuracy secondary key).
- [x] Enforce train-only sampling, leaving validation and test natural.
- [x] Persist run, strategy, epoch, class, sampling, and confusion-matrix artifacts.
- [x] Add automated unit tests covering loss functions, weights, samplers, and loaders.
- [x] Document protocol, per-class metrics, limitations, and future modeling guidance.

## Deliverables

- Imbalance utilities and sampler in [`src/fdl_project/data/imbalance.py`](../../src/fdl_project/data/imbalance.py).
- Analysis notebook: [`notebooks/preparation/class_imbalance_analysis.ipynb`](../../notebooks/preparation/class_imbalance_analysis.ipynb).
- Comparison artifacts under `output/class_imbalance/`.
- Written summary report: [`docs/reports/class_imbalance_report.md`](../reports/class_imbalance_report.md).
