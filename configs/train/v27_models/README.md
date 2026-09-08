# v27 — architectures on the settled pipeline

Phase 3. **Only the model changes.** Every arm inherits `configs/train/defaults.yaml`
unmodified:

```
64x64 letterbox, one_hot (3 channels)   rotation p=0.5, +/-180 deg
inverse-sqrt weighted sampler           cross-entropy
AdamW lr 7e-4, wd 1e-4, no schedule     batch 256, fp16 AMP
50 epochs, early stop on validation macro-F1, patience 10
```

That is the exact configuration that scored **0.8800** on `baseline_cnn`, which is
rerun here as `00_baseline_cnn` so the comparison is same-machine, same-session.

## Deliberate omissions

- **No cosine schedule.** The standalone `*_style_64.yaml` configs carry one; the
  baseline was settled without it. Adding it here would confound architecture with
  schedule. If an architecture underperforms, the schedule is the first follow-up.
- **No per-model learning rate or weight decay.** `convnext_style_64.yaml` uses
  wd 0.05 (ConvNeXt's own recipe) and `resnet_style_64.yaml` uses lr 1e-3. Both are
  held at the baseline's values here for the same reason.
- **No `baseline_v2`.** Measured ~0.80 against `baseline_cnn`'s ~0.87 once rotation
  is on. Dropped.
- **No CBAM.** Added nothing in phase 2 (0.8158 vs 0.8173).

## Reading the results

The noise floor is **0.010**, measured: the identical config scored 0.8800 and 0.8696.
An architecture beats the baseline only if its bootstrap CI lower bound sits above the
baseline's point estimate. Anything closer is a tie, and the tie-breaker is cost.

| arm | params | notes |
|---|---|---|
| `00_baseline_cnn` | 157k | reference; ~13s/epoch on a T4 |
| `01_convnext_style` | 414k | |
| `02_densenet_style` | 304k | ~164s/epoch measured, 3.94 GiB |
| `03_inception_style` | 799k | |
| `04_resnet_style` | 2.83M | heaviest |

Ordered cheapest-first so an interrupted session keeps the most arms.
