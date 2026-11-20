# Task 04: Explore Image Preprocessing Pipelines

Compare categorical-safe transformations for variable-size WM-811K maps and implement the selected configuration as a reusable PyTorch input pipeline.

## Goal

Provide every model with a deterministic, fixed-shape input while preserving wafer geometry, die-state semantics, the shared class encoding, and the frozen-test policy.

## Data Semantics & Preprocessing Rules

- **Discrete categorical states**: Raw maps contain discrete identifiers (`0`: outside wafer, `1`: functional die, `2`: defective die). They are categories, not continuous intensities.
- **Categorical-safe geometry**: Resizing must use `nearest-exact` interpolation to prevent creating fictitious intermediate values.
- **Candidate screening on training split**: Candidate dimensions, aspect policies, and encodings must be chosen exclusively using a class-balanced training sample without inspecting the test set.
- **Selected default**:
  - Aspect-ratio-preserving **letterbox resize** to `64 x 64` (padded with state 0).
  - **One-hot encoding** into 3 binary channels (`outside_wafer`, `functional_die`, `defective_die`), producing a `float32` tensor of shape `(3, 64, 64)`.

## Tasks & Checklist

- [x] Document 3 categorical die states and validate all raw wafer map inputs.
- [x] Compare pure padding, direct resizing, and aspect-ratio-preserving letterbox resizing.
- [x] Compare one-hot encoding vs normalized single-channel representations.
- [x] Benchmark geometric distortion, defective-die retention, CPU processing time, and memory footprint.
- [x] Select and freeze `64 x 64` letterbox + 3-channel one-hot as default.
- [x] Implement immutable preprocessing configuration and categorical transforms.
- [x] Implement supervised Dataset (`WM811KDataset`) returning `(image, target, row_index)`.
- [x] Implement deterministic training loaders and ordered validation/test DataLoaders.
- [x] Reject malformed maps, non-2D arrays, invalid splits, and unlabeled supervised rows.
- [x] Add automated unit tests for transformations, encodings, datasets, and loaders.
- [x] Execute candidate comparison on real training data and save benchmark artifacts.
- [x] Validate selected pipeline on the complete validation split.
- [x] Document decision rationale, trade-offs, and limitations.

## Deliverables

- Preprocessing and dataset modules in [`src/fdl_project/data/`](../../src/fdl_project/data/).
- Executed comparison notebook: [`notebooks/preparation/preprocessing_comparison.ipynb`](../../notebooks/preparation/preprocessing_comparison.ipynb).
- Benchmark artifacts under `output/preprocessing/`.
- Written summary report: [`docs/reports/preprocessing_report.md`](../reports/preprocessing_report.md).
