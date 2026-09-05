# Specification 04: Image Preprocessing Pipelines

## Goal

Define, compare, and implement one deterministic PyTorch preprocessing pipeline that converts variable-size categorical WM-811K wafer maps into fixed-shape model inputs without leaking information from the frozen test split.

## Data semantics

Each raw wafer map is a non-empty two-dimensional grid with exactly three categorical states:

| Value | Meaning |
|---:|---|
| 0 | Outside the wafer |
| 1 | Functional die |
| 2 | Defective die |

These identifiers are categories, not continuous grayscale intensities. Every geometry operation must therefore use categorical-safe nearest-neighbor interpolation and must never create fractional states.

## Candidate comparison

- Determine candidate dimensions and select the final preprocessing configuration from the persisted training split only.
- Do not inspect the frozen test split while choosing geometry, dimensions, encoding, or normalization.
- Use a fixed, class-balanced training sample and record its seed and size.
- Compare at least pure padding, direct resizing, aspect-ratio-preserving letterbox resizing, one-hot encoding, and normalized single-channel encoding.
- Measure spatial fidelity, defective-die retention, CPU transformation time, active-map occupancy, and theoretical input memory.
- Save a machine-readable benchmark, comparison metadata, and visual examples.

## Selected default

- Resize each map within a `64 x 64` canvas while preserving its aspect ratio.
- Center the resized map and fill unused letterbox cells with state 0.
- Allow upscaling so small maps use the available spatial resolution.
- Use `nearest-exact` interpolation so output cells remain in `{0, 1, 2}`.
- One-hot encode the three states in the fixed order `outside_wafer`, `functional_die`, `defective_die`.
- Return a `torch.float32` tensor shaped `(3, 64, 64)`.
- Do not apply scalar normalization to one-hot channels.

## Reusable PyTorch interface

- Represent preprocessing choices with an immutable, validated configuration object.
- Reject malformed dimensions, unsupported strategies, empty maps, non-two-dimensional inputs, non-finite values, fractional values, and states outside `{0, 1, 2}`.
- Provide an exact decoder for testing and visual diagnostics.
- Load the original DataFrame and the persisted train/validation/test row indices without duplicating wafer arrays.
- Return `(image, target, row_index)` from the supervised Dataset, using the canonical nine-class target encoding.
- Shuffle training deterministically with seed 86 by default.
- Never shuffle validation or test, and keep the smaller final batch.
- Reject unlabeled rows from supervised datasets.

## Validation requirements

Automated tests must cover configuration immutability and validation, malformed raw maps, categorical-safe geometry, aspect-ratio preservation, encoding round trips, deterministic output, benchmark summaries, persisted split loading, canonical target encoding, source-row retention, split-safe DataLoader behavior, and deterministic train shuffling.

An executed notebook must use the real WM-811K training data to compare candidates and then verify the selected Dataset on the validation split without using test observations.

## Deliverables

- Reusable preprocessing, Dataset, DataLoader, and benchmark modules under `src/fdl_project/`.
- An executed candidate-comparison notebook under `notebooks/`.
- Benchmark CSV, metadata JSON, and visual comparison under `output/preprocessing/`.
- Automated tests under `tests/`.
- A report explaining the experiment, selected default, usage, and known limitations.
