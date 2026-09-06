# Shared PyTorch Evaluation Pipeline

The project uses one evaluation implementation for every model architecture. Its primary selection metric is **macro-F1**, which gives each WM-811K class equal influence despite the strong dominance of `none`.

## Protocol

- Use `validation` for model selection, tuning, and early stopping.
- Use the frozen `test` split only for the final comparison.
- Set evaluation DataLoaders to `shuffle=False`.
- Do not apply random augmentation, oversampling, or undersampling to validation or test.
- Reuse the indices under `data/splits/`; every sample returned by the Dataset must include its original `WM811K.pkl` row index.

The implementation lives in `src/fdl_project/evaluation.py`. The immutable class encoding lives in `src/fdl_project/constants.py`.

## Dataset and DataLoader contract

A Dataset item may be a three-item tuple:

```python
image, target, row_index
```

or a mapping with `inputs`/`images`, `targets`/`labels`, and `row_indices`/`indices` keys. Targets are `torch.long` values from 0 through 8. The final classification layer must produce logits with shape `(batch_size, 9)`.

```python
from fdl_project import CLASS_TO_INDEX

target = torch.tensor(CLASS_TO_INDEX[failure_type], dtype=torch.long)
return image, target, row_index
```

## Evaluate a model

```python
from torch import nn

from fdl_project import CLASS_NAMES, evaluate_model

validation_result = evaluate_model(
    model=model,
    dataloader=validation_loader,
    device="cuda",
    criterion=nn.CrossEntropyLoss(),
    class_names=CLASS_NAMES,
    split_name="validation",
)
```

`evaluate_model` moves the model to the selected device, calls `model.eval()`, and uses `torch.inference_mode()`. The returned result contains:

- sample-weighted mean loss;
- accuracy and balanced accuracy;
- macro-F1 and weighted-F1;
- per-class precision, recall, F1, and support;
- absolute and row-normalized confusion matrices;
- one prediction row per original wafer, including all nine class probabilities.

PyTorch models may return a logits tensor directly, an object or mapping with a `logits` tensor, or a tuple whose first item is the logits tensor.

## Save a comparable experiment

```python
from fdl_project import save_evaluation_results

artifact_paths = save_evaluation_results(
    validation_result,
    output_root="output/evaluation",
    run_name="baseline-cnn-seed-86",
    metadata={
        "seed": 86,
        "git_commit": "<commit used for the experiment>",
        "checkpoint": "trained-models/baseline-cnn-seed-86.pt",
    },
)
```

This creates:

```text
output/evaluation/baseline-cnn-seed-86/
├── metrics.json
├── per_class_metrics.csv
├── predictions.csv
├── confusion_matrix.png
└── confusion_matrix_normalized.png
```

Run names cannot contain path separators. Existing artifacts are protected by default; pass `overwrite=True` only when replacement is intentional.

## Checkpoint compatibility

Save the canonical encoding alongside every model state:

```python
from fdl_project import class_encoding_metadata

torch.save(
    {
        "model_state_dict": model.state_dict(),
        **class_encoding_metadata(),
    },
    checkpoint_path,
)
```

Validate it before loading the weights:

```python
from fdl_project import validate_checkpoint_class_names

checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
validate_checkpoint_class_names(checkpoint["class_names"])
model.load_state_dict(checkpoint["model_state_dict"])
```

This prevents a checkpoint trained with a different logit order from producing plausible-looking but incorrectly named predictions.

## Confidence intervals

Macro-F1 has no closed-form sampling distribution, and the rare classes have small support in the fixed folds, so a point estimate alone cannot show whether two models are separable. `bootstrap_evaluation` resamples the evaluated rows with replacement and reports percentile intervals.

```python
from fdl_project import bootstrap_evaluation, evaluate_model, save_evaluation_results

result = evaluate_model(model=model, dataloader=loader, device=device, split_name="test")
bootstrap = bootstrap_evaluation(result, num_resamples=1000, seed=86)
save_evaluation_results(result, output_root="output/evaluation", run_name="model-a",
                        bootstrap=bootstrap)
```

`bootstrap.aggregate` holds accuracy, balanced accuracy, macro-F1, and weighted-F1 with `ci_lower`, `ci_upper`, and `standard_error`; `bootstrap.per_class` holds the same for per-class F1 alongside its support. Passing `bootstrap` to `save_evaluation_results` additionally writes `bootstrap_aggregate.csv` and `bootstrap_per_class.csv` and records the aggregate intervals under a `bootstrap` key in `metrics.json`. Omitting it leaves every existing artifact byte-identical.

Measured on the 34,591 validation predictions of the selected class-imbalance run, the 95% interval for macro-F1 spans 0.7800 to 0.8091, while per-class F1 for `Near-full` (support 30) spans 0.8302 to 0.9818. The frozen test split holds half as many observations and only 15 `Near-full` wafers, so its intervals are wider still. Two models whose test macro-F1 differ by less than roughly 0.04 should therefore be reported as not separable rather than ranked.

## Reproducible smoke test

The example below uses a deterministic pass-through PyTorch model, includes all nine classes, and uses a smaller final batch:

```bash
uv run python examples/evaluate_dummy_model.py --output-root <temporary-directory>
```

It exercises model evaluation, loss aggregation, metric computation, and all five output artifacts without requiring the full dataset.
