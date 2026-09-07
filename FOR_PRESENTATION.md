# Wafer-map failure classification — progress

MIR WM-811K, 9 classes, 172,950 labeled maps. **~20% done**: one model explored, the
architecture comparison and the test evaluation are still ahead.

## The problem

`none` is 85.2% of the data; `Near-full` is 0.06% (104 training wafers). Predicting `none`
for everything scores 85% accuracy and is useless, so the metric is **macro-F1**.

## The one result so far

Free-angle rotation, on `baseline_cnn` at 64x64:

| | macro-F1 | vs baseline |
|---|---|---|
| baseline | 0.8173 | — |
| **rotation** | **0.8800** | **+0.063** |
| dihedral8 (90° symmetries) | 0.8587 | +0.041 |
| dihedral8 + rotation | 0.8677 | +0.050 |

**`Scratch` goes 0.376 → 0.732** — about half the gain comes from that one class. Rotation
also removed the overfitting: the unaugmented baseline hits 99.3% train accuracy by epoch 12
while validation stalls.

## What did not work

Every imbalance policy (5 tested, plus focal at two γ with and without the sampler) landed
inside the noise. CBAM: −0.002. Grayscale instead of one-hot: −0.013. 224×224: +0.015 for
6.4× the compute. A smaller, more-regularised model: −7 points, because rotation had already
fixed the overfitting it was built to fix.

An earlier 4-epoch screening study had ranked the imbalance policies 6.7 points apart. At
full training length that gap vanished — short budgets can rank interventions wrongly.

## How much to trust these

The identical config, run twice, scored **0.8800** and **0.8696**. That 0.010 spread is
nondeterminism. **Differences under ~0.01 are not results** — only rotation clears it.

Validation has also been selected on across ~30 runs, adding roughly +0.02 of optimism. The
test split is untouched and gets one evaluation at the end.

## Settled pipeline

```
baseline_cnn · rotation (p=0.5) · inverse-sqrt sampler · cross-entropy
64x64 letterbox one-hot · AdamW 7e-4 · batch 256 · 50 epochs, patience 10
```

## Next

Seeds on the final candidates · the architecture comparison (5 from-scratch designs, 4
pretrained) · TTA and per-class thresholds · Grad-CAM · one test evaluation.
