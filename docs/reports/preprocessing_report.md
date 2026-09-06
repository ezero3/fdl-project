# WM-811K Image Preprocessing Report

This report documents the experiment in [`notebooks/04_preprocessing_comparison.ipynb`](../../notebooks/04_preprocessing_comparison.ipynb) and the reusable implementation in `src/fdl_project/`. The generated files under [`output/preprocessing/`](../../output/preprocessing/) record the measured results.

## Objective

WM-811K wafer maps do not share a common height and width. A PyTorch image model, however, needs batches with a fixed tensor shape. Preprocessing must resolve that mismatch without confusing the categorical die states or distorting failure patterns more than necessary.

The raw values have these meanings:

| Value | Meaning |
|---:|---|
| 0 | Outside the wafer |
| 1 | Functional die |
| 2 | Defective die |

They are identifiers rather than grayscale measurements. Bilinear or bicubic interpolation would invent fractional states with no physical meaning, so every resizing candidate uses PyTorch `nearest-exact` interpolation.

## Leakage-safe experiment

Candidate selection used only the persisted **training split** from task 02. With seed **86**, the notebook sampled 100 maps from each of the nine classes, for a class-balanced total of **900** maps. The rarest training class still contains 104 observations, so no class required replacement sampling.

The maximum native training dimensions, **212 x 187**, were also derived from training data. The frozen test split was not loaded or inspected. After the decision, the selected transform was instantiated over all **34,591 validation rows** only as an integration check; the first 32-item batch had shape `(32, 3, 64, 64)`.

## Candidates

Four pipelines were evaluated:

1. **Native padding, one-hot:** center every map on the largest training canvas (`212 x 187`) without resizing.
2. **Direct resize, one-hot:** independently resize height and width to `64 x 64`.
3. **Letterbox, one-hot:** resize uniformly to fit inside `64 x 64`, then center-pad the unused dimension.
4. **Letterbox, single channel:** use the same geometry but encode the states as `0.0`, `0.5`, and `1.0` in one channel.

The one-hot candidates use the fixed channel order `outside_wafer`, `functional_die`, `defective_die` and require no scalar normalization. This retains categorical semantics and avoids implying that state 1 is numerically halfway between states 0 and 2.

## Results

The following results were measured on the 900-map training sample. Timing is a local CPU measurement and should be treated as comparative rather than a hardware-independent benchmark. Input memory is the theoretical `float32` tensor memory for a batch of 64 before activations.

| Candidate | Output | ms/map | Batch input | Median defect-ratio error | P95 defect-ratio error | Median aspect error | P95 aspect error | Lost-defect maps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Native padding, one-hot | `3 x 212 x 187` | 3.864 | 29.04 MiB | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 |
| Direct resize, one-hot | `3 x 64 x 64` | 1.455 | 3.00 MiB | 0.0023 | 0.0078 | 0.0770 | 0.2301 | 0 |
| Letterbox, one-hot | `3 x 64 x 64` | 1.476 | 3.00 MiB | 0.0025 | 0.0094 | 0.0044 | 0.0084 | 0 |
| Letterbox, single channel | `1 x 64 x 64` | 1.121 | 1.00 MiB | 0.0025 | 0.0094 | 0.0044 | 0.0084 | 0 |

The aspect error is the absolute log-ratio between the active-die bounding-box aspect ratios before and after transformation; zero is ideal. Defect-ratio error measures the absolute change in defective dies divided by active dies. No sampled map that originally contained defects lost all of them under any candidate.

Pure padding is lossless but inefficient: the average active wafer occupies only **3.54%** of its canvas, and a batch requires about 9.7 times the input memory of a `64 x 64` one-hot batch. Direct resize is compact, but its median shape distortion is about 17.6 times the letterbox value. Letterbox keeps the compact tensor while its small residual aspect error comes only from rounding resized integer dimensions.

The visual artifact [`preprocessing_examples.png`](../../output/preprocessing/preprocessing_examples.png) confirms these trade-offs across all nine classes: native padding makes typical wafers very small, direct resize stretches non-square maps, and letterbox keeps their recognizable proportions.

## Selected default

The shared default is:

```python
PreprocessingConfig(
    target_size=(64, 64),
    geometry="letterbox",
    encoding="one_hot",
    normalization="none",
    allow_upscale=True,
)
```

It was selected because it:

- preserves aspect ratio substantially better than direct resizing;
- reduces input memory from 29.04 MiB to 3.00 MiB per reference batch compared with native padding;
- preserves exact categorical states and lost no defect-bearing maps in the sample;
- gives the CNN an explicit channel for each state rather than imposing an ordinal relationship;
- produces one stable `(3, 64, 64)` shape for every model.

The single-channel variant is cheaper, but it encodes an artificial numerical ordering. It remains available for controlled experiments, not as the shared default.

## PyTorch API

Load the original DataFrame once and reconstruct a split from the persisted row indices:

```python
from fdl_project import (
    DEFAULT_PREPROCESSING_CONFIG,
    create_dataloader,
    create_split_dataset,
    load_wm811k_dataframe,
)

dataframe = load_wm811k_dataframe("data/MIR-WM811K/WM811K.pkl")
train_dataset = create_split_dataset(
    dataframe,
    "data/splits",
    "train",
    preprocessing_config=DEFAULT_PREPROCESSING_CONFIG,
)
train_loader = create_dataloader(
    train_dataset,
    batch_size=64,
    num_workers=0,
)

images, targets, row_indices = next(iter(train_loader))
```

`images` is `float32` with shape `(batch, 3, 64, 64)`, `targets` is `long` using the immutable nine-class order, and `row_indices` retains the source DataFrame index required by the shared evaluation pipeline. Training shuffles deterministically with seed 86. Validation and test loaders reject shuffling and never discard their final smaller batch.

On Windows, `num_workers=0` is the safest default for the two-gigabyte in-memory DataFrame because spawned workers may otherwise duplicate it. Worker tuning can be revisited with the actual training hardware.

## Reproducible artifacts

- [`preprocessing_benchmark.csv`](../../output/preprocessing/preprocessing_benchmark.csv) contains the full-precision measurements.
- [`comparison_metadata.json`](../../output/preprocessing/comparison_metadata.json) records the seed, selection split, sample size, candidates, chosen configuration, and explicit `test_set_used: false` flag.
- [`preprocessing_examples.png`](../../output/preprocessing/preprocessing_examples.png) compares representative maps from all nine classes.

## Scope and limitations

This task fixes deterministic base preprocessing, not stochastic training augmentation. Rotations, flips, sampling policies, class-weighted losses, and learned resampling remain separate model-training decisions and must be selected using train and validation only.

The comparison measures preservation of map-level geometry and state ratios, not downstream classification accuracy. The `64 x 64` choice is therefore a well-supported baseline, but a later controlled modeling experiment may compare other resolutions without changing the frozen test policy. Validation and test preprocessing must always remain deterministic and identical.
