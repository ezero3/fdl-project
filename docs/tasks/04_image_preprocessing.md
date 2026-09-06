# Explore Image Preprocessing Pipelines

Compare categorical-safe transformations for variable-size WM-811K maps and implement the selected configuration as a reusable PyTorch input pipeline.

## Goal

Give every model a deterministic, fixed-shape input while preserving wafer geometry, die-state semantics, the shared class encoding, and the frozen-test policy.

## Tasks

- [x] Document the three categorical wafer-map states and validate all raw inputs
- [x] Compare pure padding, direct resizing, and aspect-ratio-preserving letterbox resizing
- [x] Compare one-hot and normalized single-channel encodings
- [x] Use only a deterministic, class-balanced sample from the training split for selection
- [x] Measure geometry distortion, defective-die retention, processing time, occupancy, and input memory
- [x] Select and freeze the default `64 x 64` letterbox plus three-channel one-hot configuration
- [x] Implement immutable preprocessing configuration and categorical-safe PyTorch transforms
- [x] Implement a supervised Dataset using the canonical nine-class encoding and persisted splits
- [x] Implement deterministic training and ordered validation/test DataLoaders
- [x] Preserve original row indices for compatibility with the shared evaluation pipeline
- [x] Reject malformed maps, invalid split artifacts, and unlabeled supervised rows
- [x] Add automated tests for transforms, encodings, datasets, loaders, and benchmark results
- [x] Execute the comparison on the real dataset and save reproducible artifacts
- [x] Validate the selected pipeline on the complete validation split
- [x] Document the decision, API, trade-offs, and frozen-test policy
