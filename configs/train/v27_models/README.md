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

### Tier 1 — architecture

| arm | params | notes |
|---|---|---|
| `00_baseline_cnn` | 157k | reference; ~13s/epoch on a T4 |
| `01_convnext_style` | 414k | |
| `02_densenet_style` | 304k | ~164s/epoch measured, 3.94 GiB |
| `03_inception_style` | 799k | |
| `04_resnet_style` | 2.83M | |

### Tier 2 — depth, at fixed width

> **Superseded by `configs/train/v28_capacity/`.** These were written before v27 had
> results. v28 asks the same capacity question better: it scales ConvNeXt to 2.68M so it
> *matches* `resnet_style`'s 2.83M, which is what makes architecture and capacity separable.
> Run v28 instead; these are kept only for the record.

Every width, growth rate and branch size is held at its tier-1 value, so `05` minus
`01` is stages and nothing else. Regularization is held too (convnext keeps
`stochastic_depth: 0.05` rather than the higher value a deeper ConvNeXt would normally
get) — otherwise the arm would differ in two ways at once.

| arm | params | change |
|---|---|---|
| `05_convnext_style_deep` | 1.43M | blocks (2,2,2) → (3,3,9), ConvNeXt-T's own ratio |
| `06_densenet_style_deep` | 775k | block layers (4,6,8) → (6,12,16) |
| `07_inception_style_deep` | 1.66M | 2 → 4 blocks per stage |
| `08_resnet_style_deep` | 5.97M | 2 → 4 blocks per stage; 38x the baseline |

Ordered cheapest-first so an interrupted session keeps the most arms, and so tier 1
completes before tier 2 starts. If no tier-1 arm clears the baseline, tier 2 is answering
a question that is already closed — skip it and spend the hours on seeds.
