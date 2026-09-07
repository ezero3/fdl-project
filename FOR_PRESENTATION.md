# Wafer-map failure classification — what we did

MIR WM-811K: 811,457 wafer maps, 172,950 labeled, nine failure classes.
Foundations of Deep Learning, Milano-Bicocca 2025/2026.

---

## 1. The problem, in one number

The majority class `none` is **85.2%** of the labeled data. The rarest, `Near-full`, is
0.06% — 104 training wafers against 103,199. A model that predicts `none` for everything
scores 85% accuracy and is worthless.

So **macro-F1 is the metric**: it weights all nine classes equally, which means a class with
30 validation wafers counts as much as one with 29,487.

| class | train | validation | test |
|---|---|---|---|
| none | 103,199 | 29,487 | 14,745 |
| Edge-Ring | 6,776 | 1,936 | 968 |
| Edge-Loc | 3,632 | 1,038 | 519 |
| Center | 3,006 | 859 | 429 |
| Loc | 2,516 | 718 | 359 |
| Scratch | 835 | 239 | 119 |
| Random | 606 | 173 | 87 |
| Donut | 389 | 111 | 55 |
| Near-full | 104 | 30 | 15 |

---

## 2. What we fixed before modeling

**Wafer maps are categorical, not images.** Each cell is 0 = no die, 1 = working die,
2 = defective die. Any transform that interpolates between those states invents a value that
does not exist. Everything downstream respects that: one-hot into three channels, and
nearest-neighbour wherever resampling is unavoidable.

**The split is group-aware and frozen.** Wafers from one production lot are correlated, so a
random split leaks: the model recognises the lot rather than the defect. We split by
`lotName` 70/20/10, generated once, stored as row indices and committed. **The test split
has not been opened.**

**Geometry was decided by measurement, not taste.** Wafer maps have 632 distinct shapes.
Padding everything onto the largest canvas costs 9.7x the memory for a wafer occupying 3.5%
of it; resizing directly distorts aspect ratio 17.6x more than letterboxing. We letterbox to
64x64.

---

## 3. How we ran experiments

One rule: **anything that is not the model is compared on one fixed cheap model.** Otherwise
a difference cannot be attributed. Only settings that win get carried to the real
architectures.

Every run: config file, fixed seed, validation-driven early stopping, bootstrap confidence
intervals, artifacts and per-class metrics written to disk and mirrored to Weights & Biases.

---

## 4. What we measured

Free-angle rotation is the finding. Everything else is a negative result.

| intervention | macro-F1 | vs baseline | verdict |
|---|---|---|---|
| baseline (no augmentation) | 0.8173 | — | |
| **rotation** | **0.8800** | **+0.063** | **the only clear gain** |
| dihedral8 (the 8 square symmetries) | 0.8587 | +0.041 | real, smaller |
| dihedral8 + rotation | 0.8677 | +0.050 | no better than rotation alone |
| 224x224 instead of 64x64 | 0.8321 | +0.015 | not worth 6.4x the compute |
| focal loss | 0.8246 | +0.007 | indistinguishable |
| CBAM attention | 0.8158 | -0.002 | nothing |
| grayscale instead of one-hot | 0.8042 | -0.013 | worse |

**Rotation nearly doubles the hardest class.** `Scratch` — thin, one-die-wide lines — goes
from **0.376 to 0.732**. About half the macro-F1 gain comes from that class alone, which is
what you would expect if the mechanism is giving a rare, orientation-specific pattern more
distinct views.

Rotation also fixed overfitting as a side effect. The unaugmented baseline reaches 99.3%
training accuracy by epoch 12 while validation stalls; with rotation it sits at 96.1% and
keeps improving to epoch 36.

---

## 5. What did not work, and why that matters

**Imbalance handling was not the binding constraint.** We tested five policies — unweighted
cross-entropy, inverse-sqrt weighting, effective-number weighting, an inverse-sqrt sampler,
and focal loss at two γ values, with and without the sampler. **Every one landed inside the
noise.** An earlier 4-epoch screening study had shown a 6.7-point gap in favour of the
sampler; at full training length that gap disappeared, which is itself worth reporting —
short screening budgets can rank interventions wrongly.

**Attention added nothing.** CBAM on the baseline: -0.002.

**Higher resolution did not pay.** 224x224 is close to the native maximum (212x187) and
gained 0.015 for 6.4x the epoch time, with overlapping intervals.

**A smaller, more regularised model was worse.** We built a variant replacing the classifier
head with global average pooling (157k -> 34k parameters) to attack the overfitting. It lost
7 points. Rotation had already removed the overfitting, so the capacity cut was pure loss.
The lesson is about ordering: we diagnosed a problem, fixed it one way, then applied a second
fix for the same problem.

---

## 6. How confident we are

**We measured our own noise floor.** The identical configuration, run twice, scored
**0.8800** and **0.8696**. That 0.010 spread comes from nondeterminism alone — we leave
deterministic GPU kernels off because they are substantially slower.

That number disciplines everything above. Differences under ~0.01 are not results. Of every
intervention we tested, **only rotation clears it by a wide margin.**

Validation has also been selected on across roughly 30 runs, which adds about +0.02 of
optimism to any reported number. **The test split remains untouched**, and the final figures
in the report will come from a single evaluation on it — likely a point or two below the
validation numbers, which is expected and will be stated.

---

## 7. Engineering, briefly

Config-driven runs with validated schemas, an explicit registry, and unknown keys raising
rather than silently defaulting. Atomic checkpointing that resumes optimizer, schedule,
scaler, epoch, history and RNG state. 322 tests.

**We profiled before optimising, and it changed what we did.** The GPU sat at 15%
utilisation: these runs were dataloader-bound, not compute-bound. Batch size was nearly
irrelevant (throughput flat from 64 to 1024) while worker count gave 1.9x for one config
line. Caching maps already letterboxed cut per-item cost 60→32 µs, and moving encoding and
rotation onto the GPU removed the rest — shipping one byte per cell instead of twelve.

Parameter count predicted neither time nor memory: two ~300k models cost 7x the time and
10x the memory of a 157k one, because they hold full resolution deeper.

---

## 8. The pipeline we settled on

```
baseline_cnn · free-angle rotation (p=0.5) · inverse-sqrt sampler · cross-entropy
64x64 letterbox one-hot · AdamW 7e-4 · batch 256 · 50 epochs, patience 10 · fp16 AMP
```

Simplest setup that captures the one effect we could actually measure.

---

## 9. Still to do

- Multiple seeds on the final candidates, to replace single-run numbers with means and spreads
- The architecture comparison: five more from-scratch designs and the pretrained backbones
- Test-time augmentation and per-class thresholds, fitted on validation and applied once
- Grad-CAM and error analysis on the models we present
- One test evaluation, at the end
