# Experiment grid

Every model × every option the pipeline can express. Cells are for results: write
`macro-F1 [ci]` plus a note. Blank = untried.

## What is settled so far

All on `baseline_cnn` at 64x64, validation macro-F1, patience 10.

| finding | evidence |
|---|---|
| **Free-angle rotation is the only intervention that cleared its interval** | 0.8800 vs baseline 0.8173; `Scratch` 0.376 -> 0.732 |
| Composing rotation with `dihedral8` does not beat rotation alone | 0.8677 vs 0.8800 |
| Imbalance handling is **not** the binding constraint | every arm inside +-0.013: sampler 0.8173, unweighted 0.8160, focal 0.8246 |
| Focal loss makes no measurable difference | best arm +0.0042 against a 0.010 noise floor (`baseline_cnn`, 4 arms) |
| Focal without the sampler *looked* catastrophic on `baseline_v2` (~0.70 vs ~0.78) but not on `baseline_cnn` (-0.007) | a `baseline_v2` artifact, not a property of focal |
| The sampler's own gain is **not** established at full budget | unweighted 0.8160 vs sampler 0.8173 in phase 2; +0.011 in v26 but the no-sampler arms were truncated at the epoch cap |
| CBAM adds nothing | 0.8158 vs 0.8173 |
| Grayscale encoding is worse than one-hot | 0.8042 vs 0.8173 |
| 224x224 does not pay for itself | 0.8321 for 6.4x the compute; interval overlaps the baseline |
| **`baseline_v2` is worse than `baseline_cnn` once rotation is on** | ~0.80 vs ~0.87 at 50 epochs; the 157k -> 34k cut removed capacity to fix overfitting that rotation had already fixed |

**The noise floor, measured directly.** The identical config -- `baseline_cnn` + rotation +
sampler -- scored **0.8800** and **0.8696** on two runs. That 0.010 spread comes from
nondeterminism alone (deterministic kernels are off for speed), and it is nearly as large as
the entire spread of the six-arm focal series. Any difference under ~0.01 is unmeasurable
with one seed.

Caveat on all of it: single seed, and validation has been selected on heavily, so roughly
+0.02 of optimism is priced in on top. **Only the rotation result is large enough to survive
both.**

`—` = not applicable. `▪` = the default pipeline.
Defaults: 64×64 letterbox one-hot · no augmentation · `inverse_sqrt_sampler` · no attention ·
AdamW 7e-4 · no schedule · batch 256 · AMP on · 50 epochs, patience 5.

**Fixed, not axes.** `geometry: letterbox` for every model and every encoding — settled
offline (`docs/reports/preprocessing_report.md`): pad costs 9.7× the memory for a wafer
filling 3.5% of its canvas, resize distorts 17.6× more. `amp: true` throughout.

---

## Models

| model | kind | params | epoch (T4) | CBAM | notes |
|---|---|---|---|---|---|
| `baseline_cnn` | ours | 157k | 12.8s | ✓ | test-bed for the sweep |
| `dilated_style` | ours | 298k | 168s | ✓ | context, no downsampling |
| `densenet_style` | ours | 304k | 164s | ✓ | feature reuse |
| `convnext_style` | ours | 414k | — | ✓ | conv operator, transformer design |
| `inception_style` | ours | 799k | — | ✓ | parallel kernel sizes |
| `vit_style` | ours | 2.72M | — | — | no conv prior |
| `resnet_style` | ours | 2.83M | — | ✓ | residual depth |
| `resnet18` | pretrained | — | — | ✓ | |
| `resnet34` | pretrained | — | — | ✓ | |
| `mobilenet_v3_small` | pretrained | — | — | ✓ | |
| `mobilenet_v3_large` | pretrained | — | — | ✓ | |
| `efficientnet_b0` | pretrained | — | — | ✓ | |
| `vit_b_16` | pretrained | 86M | 1482s ft / 766s frozen | — | |
| `vit_b_32` | pretrained | 86M | 458s | — | 49 tokens vs 196 |
| `swin_t` | pretrained | — | — | — | |

---

## Augmentation

`data.augmentation.name` · train only · `probability: 1.0` · per-class weights allowed only
where nothing resamples.

| model | none ▪ | dihedral8 | rotation | dihedral8_rotation |
|---|---|---|---|---|
| `baseline_cnn` | 0.8173 | 0.8587 | **0.8800** | 0.8677 |
| `dilated_style` | | | | |
| `densenet_style` | | |
| `convnext_style` | | | |
| `convnext_style` | | | | |
| `convnext_style` | | | | |
| `inception_style` | | | | |
| `vit_style` | | | | |
| `resnet_style` | | | | |
| `resnet18` | | ✗ | | |
| `resnet34` | | ✗ | | |
| `mobilenet_v3_small` | | ✗ | | |
| `mobilenet_v3_large` | | ✗ | | |
| `efficientnet_b0` | | ✗ | | |
| `vit_b_16` | | ✗ | | |
| `vit_b_32` | | ✗ | | |
| `swin_t` | | ✗ | | |

`dihedral8` = all 8 square symmetries, exact index permutations — always applied
(`probability: 1.0`), since it is exact and free. `rotation` = free angle, nearest-neighbour,
the only one that resamples, so it runs at **`rotation_probability: 0.5`** to keep clean
unresampled examples in the training distribution. `dihedral8_rotation` composes them: the
group always, the rotation on half the samples, making the reachable view set continuous
rather than 8 elements.

Corners exposed by a rotation are filled with the background value **for the active
encoding** — the runner derives it, so a greyscale or ImageNet-normalised arm does not get a
one-hot fill.

Also in the registry but not gridded, because both are strict subgroups of `dihedral8` and
so would answer a narrower question: `rotations` (C4) and `flips` (Klein four-group).

## Imbalance

`imbalance.preset` · majority class 85.2%, worst ratio 992:1.

| model | unweighted_ce | inverse_sqrt_sampler ▪ | focal_loss |
|---|---|---|---|
| `baseline_cnn` | 0.8160 | 0.8173 | 0.8246 |
| `baseline_cnn` + rotation | — | 0.8696 / 0.8800 | 0.8600 – 0.8624 (truncated) |
| `baseline_cnn` + rotation, focal **and** sampler | — | — | γ1 0.8738 · γ2 0.8711 |
| `baseline_v2` + rotation | — | 0.7830 | 0.7201 – 0.7250 |
| `baseline_v2` + rotation, focal **and** sampler | — | — | γ1 0.7724 · γ2 0.7828 |
| `dilated_style` | | | |
| `densenet_style` | | |
| `convnext_style` | | | |
| `convnext_style` | | | |
| `vit_style` | | | |
| `resnet18` | | | |
| `vit_b_16` | | | |

`unweighted_ce` is the baseline the gain is measured against; `inverse_sqrt_sampler` is the
pick; `focal_loss` is the if-time-allows arm.

Settled over three seeds (`docs/reports/class_imbalance_report.md`): sampler **0.7955** ±
0.0066, `inverse_sqrt_ce` 0.7718, `unweighted_ce` 0.7286, `focal_loss` 0.7194.

**Two caveats on focal loss placing last.** The ranking came from a **4-epoch** screening
budget, and focal loss suppresses gradient from well-classified examples by `(1-p)^γ` — at 4
epochs almost nothing is well classified, so that measures its slow start and none of its
payoff. And our preset is `gamma: 2.0` with **no α class weighting**, so it targets easy/hard
imbalance rather than class frequency directly. If it gets a run, compare it against
`unweighted_ce`, not stacked on the sampler: flattening the prior *and* down-weighting easy
examples double-corrects.

Also available: `inverse_sqrt_ce` (2nd place), `effective_number_ce`.
Not implemented: majority cap per epoch (roadmap 12), logit adjustment, decoupled two-stage,
hierarchical (roadmap 14).

---

## Resolution

Geometry is fixed at letterbox; only resolution varies, and it does not apply uniformly.
**Max native training dimensions are 212×187**, so 224² is essentially native and 64² is a
~3× downsample chosen for speed — not the other way round.

| model | 64² | 224² | why |
|---|---|---|---|
| `baseline_cnn` | ▪ 0.8173 | 0.8321 | global pooling — resolution-agnostic; 6.4x the compute |
| `dilated_style` | ▪ | | same; but full-resolution stack, so 224² is costly |
| `densenet_style` | ▪ | | same |
| `convnext_style` | ▪ | | same |
| `inception_style` | ▪ | | same |
| `resnet_style` | ▪ | | same |
| `vit_style` | ▪ | rebuild | positional table sized at construction; 784 tokens at patch 8 |
| `resnet18` | ✗ | ▪ | 32× downsampling leaves a 2×2 map at 64² |
| `resnet34` | ✗ | ▪ | same |
| `mobilenet_v3_small` | ✗ | ▪ | same |
| `mobilenet_v3_large` | ✗ | ▪ | same |
| `efficientnet_b0` | ✗ | ▪ | same |
| `vit_b_16` | ✗ | ▪ | fixed positional table |
| `vit_b_32` | ✗ | ▪ | fixed positional table |
| `swin_t` | ✗ | ▪ | fixed window size |

---

## Encoding

Orthogonal to geometry — letterbox applies either way. One-hot forbids normalization; the
combination raises.

| model | one_hot ▪ | single_channel | grayscale_rgb | grayscale_rgb + imagenet |
|---|---|---|---|---|
| `baseline_cnn` | ▪ 0.8173 | ✗ | 0.8042 | — |
| `dilated_style` | | ✗ | | — |
| `densenet_style` | | ✗ | | — |
| `convnext_style` | | ✗ | | — |
| `inception_style` | | ✗ | | — |
| `vit_style` | | ✗ | | — |
| `resnet_style` | | ✗ | | — |
| `resnet18` | | ✗ | | |
| `resnet34` | | ✗ | | |
| `mobilenet_v3_small` | | ✗ | | |
| `mobilenet_v3_large` | | ✗ | | |
| `efficientnet_b0` | | ✗ | | |
| `vit_b_16` | | ✗ | | |
| `vit_b_32` | | ✗ | | |
| `swin_t` | | ✗ | | |

`single_channel` maps the three states to 0.0 / 0.5 / 1.0. It was rejected on principle — it
implies a defect is "twice" a functional die — but never measured, so it is a legitimate
one-run ablation on any model.

`grayscale_rgb` maps the three states to 0.0/0.5/1.0 replicated across RGB;
`+ imagenet` then applies torchvision's ImageNet mean/std. **Implemented.**

`single_channel` is marked `✗` because **no model accepts it**: every architecture here
hard-requires 3 input channels and raises on 1. Running that ablation needs a model change,
and `grayscale_rgb` already asks the same question (does the false ordering hurt?) in a
shape the models accept — so use that instead.

The ImageNet arm is marked `—` for our models on purpose: the question it asks is whether
matching a *pretraining* distribution helps, and there is no pretraining to match when
training from scratch. Running it there would measure normalisation, not transfer.

Both greyscale encodings reintroduce the false ordering one-hot exists to remove — they imply
a defect is "twice" a functional die. That is the trade being measured, not an oversight.

## Attention

`model.kwargs.attention` · CBAM sits before global pooling · +610 params on `baseline_cnn`.

| model | none ▪ | cbam |
|---|---|---|
| `baseline_cnn` | 0.8173 | 0.8158 |
| `dilated_style` | | |
| `densenet_style` | | |
| `convnext_style` | | |
| `inception_style` | | |
| `resnet_style` | | |
| `resnet18` | | |
| `resnet34` | | |
| `mobilenet_v3_small` | | |
| `mobilenet_v3_large` | | |
| `efficientnet_b0` | | |

`vit_style`, `vit_b_16`, `vit_b_32`, `swin_t` refuse it — already attention models.

---

## Pretrained encoder handling

| model | full fine-tune ▪ | frozen encoder | discriminative LR (100×) |
|---|---|---|---|
| `resnet18` | | | |
| `resnet34` | | | |
| `mobilenet_v3_small` | | | |
| `mobilenet_v3_large` | | | |
| `efficientnet_b0` | | | |
| `vit_b_16` | | | |
| `vit_b_32` | | | |
| `swin_t` | | | |

Freezing ViT-B buys only 1.9×, not 10× — the 86M forward still runs. Budget it as a real run.

---

## Post-hoc (no retraining)

Fitted on validation, reused unchanged for the single test evaluation.

| model | argmax ▪ | + TTA | + per-class thresholds | + both |
|---|---|---|---|---|
| `baseline_cnn` | | | | |
| `dilated_style` | | | | |
| `densenet_style` | | |
| `convnext_style` | | | |
| `convnext_style` | | | | |
| `convnext_style` | | | | |
| `inception_style` | | | | |
| `vit_style` | | | | |
| `resnet_style` | | | | |
| `resnet18` | | ✗ | | |
| `vit_b_16` | | ✗ | | |

Indicative, on a 6-epoch undertrained baseline: 0.6130 → +TTA 0.6190 → +thresholds 0.6614.
Expect the threshold gain to shrink on a converged model.

---

## Optimizer and schedule

Not per-model — sweep on the test-bed only.

| axis | options |
|---|---|
| optimizer | `adamw` ▪ · `sgd` · `adam` · `nadam` · `rmsprop` · `adadelta` |
| schedule | none ▪ · `cosine` · `cosine_with_warmup` · `cosine_warm_restarts` · `onecycle` · `step` · `multistep` · `exponential` · `plateau` |
| interval | `epoch` ▪ · `step` |
| param groups | regex → per-group LR; a pattern matching nothing **raises** |

Trainer, fixed: `max_epochs` 50 · `batch_size` 256 · `amp` true (fp16; validation is fp32) ·
`max_gradient_norm` 5.0 · early stopping on `validation_macro_f1`, max, **patience 5**, min 5.
