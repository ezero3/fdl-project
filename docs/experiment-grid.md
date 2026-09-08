# Experiment grid

Every model x every option the pipeline can express. **36 runs measured**; the table below
is the complete record, and the fillable grids further down mark what remains untried.

Validation macro-F1 with bootstrap percentile CIs. The test split is still frozen.

## What is settled

**The noise floor is 0.02.** One fixed config -- `baseline_cnn` + rotation + sampler --
scored **0.8800**, **0.8696** and **0.8646** across three runs. That spread is
nondeterminism alone (deterministic kernels are off for speed). Validation has also been
selected on heavily, so more optimism is priced in on top. **Two interventions are large
enough to survive both.**

| finding | evidence |
|---|---|
| **Free-angle rotation is the largest confirmed effect** | 0.8800 vs 0.8173 on `baseline_cnn`; `Scratch` 0.376 -> 0.732 |
| **Dilation is the second -- but only where the network holds full resolution** | On `dilated_style`, which never downsamples: `v31-dilated_rotation_64` 0.8852 [0.8714, 0.8955] vs `v31-dilated_control_64` 0.8216 [0.8065, 0.8347] -- **+0.0636, non-overlapping intervals**, identical 298,377 parameters. On ConvNeXt at 128px, which downsamples 3x: `v31-convnext_dilated` 0.8957 vs `v28-convnext_big_128` 0.8988 -- **-0.0031, nothing.** Dilation is a *route to receptive field*, not a free gain: it pays when spacing the kernels is the only route available, and adds nothing to an architecture that already got there by downsampling |
| **Fine-tuning rescues a pretrained backbone; freezing does not** | ResNet18 frozen 0.7351 -> fine-tuned at encoder lr 1e-5 **0.8938**. **+0.159**, by far the largest single delta measured |
| **Architecture is not the binding constraint** | 157k to 88M parameters, ours and pretrained, 64px to 224px -- fifteen runs land between 0.8646 and 0.9041. Only `baseline_cnn` sits clearly below the pack |
| **Capacity is not either, within an architecture** | ConvNeXt 414k -> 2.68M at 64px: **-0.0021**. The one clean within-architecture capacity test |
| Resolution helps slightly, below the floor | ConvNeXt 64px 0.8862 -> 128px 0.8988 (+0.0126, `Scratch` 0.740 -> 0.787). Same direction as phase 2's 224px result (+0.015), never confirmed |
| Imbalance handling is not the binding constraint | every phase-2 arm inside +-0.013: sampler 0.8173, unweighted 0.8160, focal 0.8246 |
| Focal loss makes no measurable difference | best arm +0.0042 on `baseline_cnn`; the catastrophic-looking result was a `baseline_v2` artifact |
| The sampler's own gain is not established | unweighted 0.8160 vs sampler 0.8173 in phase 2 |
| CBAM adds nothing | 0.8158 vs 0.8173 |
| **`one_hot` beats `grayscale_rgb`, even on a frozen ImageNet backbone** | from scratch 0.8173 vs 0.8042; frozen ResNet18 0.7351 vs 0.6877 (**+0.047**, clears the floor). Keeping the categorical states beats matching ImageNet's input distribution |
| A from-scratch transformer loses to CNNs of the same size | `vit_style` 2.83M at 128px: 0.8744, converged (best 70/80, early stopping fired). `convnext_style` at **414k** scored 0.8883 |
| `baseline_v2` is worse than `baseline_cnn` once rotation is on | ~0.80 vs ~0.87; the 157k -> 34k cut removed capacity to fix overfitting rotation had already fixed |

### The dilation pair, side by side

| | rates | downsamples | macro-F1 | Center | Loc | Scratch |
|---|---|---|---|---|---|---|
| `v31-dilated_rotation_64` | 1,2,4,8 x2 | **never** | 0.8852 | 0.910 | 0.782 | 0.731 |
| `v31-dilated_control_64` | all 1 | **never** | 0.8216 | 0.797 | 0.606 | 0.731 |
| `v28-convnext_big_128` | none | 3x | 0.8988 | 0.941 | 0.811 | 0.793 |
| `v31-convnext_dilated` | 1,2,4 | 3x | 0.8957 | 0.934 | 0.794 | 0.784 |

The first pair differs by +0.064, the second by -0.003. Same intervention, opposite verdicts,
and the reason is in the third column.

### The shape of the result

The best from-scratch model (298k, dilated) and the best pretrained one (21.4M, ResNet34)
are **0.0046 apart**. A 414k ConvNeXt we wrote matches an 87.7M pretrained ViT. On this
dataset the ceiling is the data -- 121k training wafers, 104 `Near-full` examples -- not the
model. What moved the needle was **what you show the network** (rotation, dilation's
receptive field, the categorical encoding), not how big it is.

---

## All results

Sorted by validation macro-F1. `⚠` = best epoch at the cap, so the number is a floor.

| run | model | kind | px | macro-F1 | 95% CI | Scratch | best/epochs | s/epoch |
|---|---|---|---|---|---|---|---|---|
| `v32-resnet34_finetune` | `frozen_backbone_mlp` | pretrained | 128 | **0.9041** | [0.8911, 0.9143] | 0.793 | 36/46 | 58 |
| `v32-vit_b_32_finetune` | `frozen_backbone_mlp` | pretrained | 224 | **0.9029** | [0.8892, 0.9141] | 0.790 | 15/25 | 62 |
| `v33-convnext_tiny_finetune` | `frozen_backbone_mlp` | pretrained | 128 | **0.9027** | [0.8912, 0.9124] | 0.816 | 15/25 | 232 |
| `dilated-style-64-dihedral8` | `dilated_style` | ours | 64 | **0.8995** | [0.8856, 0.9104] | 0.793 | 45/50 ⚠ | 172 |
| `v28-convnext_big_128` | `convnext_style` | ours | 128 | **0.8988** | [0.8852, 0.9092] | 0.787 | 30/40 | 356 |
| `v30-finetune_encoder_1e-5` | `frozen_backbone_mlp` | pretrained | 128 | **0.8938** | [0.8799, 0.9049] | 0.813 | 31/40 ⚠ | 37 |
| `v33-efficientnet_b0_finetune` | `frozen_backbone_mlp` | pretrained | 128 | **0.8923** | [0.8795, 0.9026] | 0.788 | 28/38 | 118 |
| `v27-resnet_style` | `resnet_style` | ours | 64 | **0.8900** | [0.8767, 0.9011] | 0.759 | 24/34 | 51 |
| `v27-convnext_style` | `convnext_style` | ours | 64 | **0.8883** | [0.8739, 0.8999] | 0.736 | 46/50 ⚠ | 32 |
| `v28-resnet_style_128` | `resnet_style` | ours | 128 | **0.8878** | [0.8729, 0.9001] | 0.808 | 13/24 | 218 |
| `v28-convnext_big_64` | `convnext_style` | ours | 64 | **0.8862** | [0.8716, 0.8976] | 0.740 | 25/35 | 85 |
| `v27-densenet_style` | `densenet_style` | ours | 64 | **0.8853** | [0.8708, 0.8969] | 0.769 | 27/37 | 114 |
| `v31-dilated_rotation_64` | `dilated_style` | ours | 64 | **0.8852** | [0.8714, 0.8955] | 0.705 | 23/33 | 171 |
| `v27-inception_style` | `inception_style` | ours | 64 | **0.8779** | [0.8641, 0.8892] | 0.775 | 8/18 | 138 |
| `v28-vit_style_128` | `vit_style` | ours | 128 | **0.8744** | [0.8615, 0.8844] | 0.684 | 70/80 | 13 |
| `baseline_cnn-rotation-focal_g1-sampler` | `baseline_cnn` | ours | 64 | **0.8738** | [0.8593, 0.8866] | 0.696 | 20/30 | 4 |
| `baseline_cnn-rotation-focal_g2-sampler` | `baseline_cnn` | ours | 64 | **0.8711** | [0.8554, 0.8839] | 0.702 | 30/40 | 4 |
| `baseline_cnn-rotation-sampler` | `baseline_cnn` | ours | 64 | **0.8696** | [0.8538, 0.8822] | 0.726 | 18/28 | 4 |
| `v27-baseline_cnn` | `baseline_cnn` | ours | 64 | **0.8646** | [0.8474, 0.8788] | 0.715 | 20/30 | 3 |
| `baseline_cnn-rotation-focal_g1` | `baseline_cnn` | ours | 64 | **0.8624** | [0.8462, 0.8760] | 0.647 | 48/50 ⚠ | 4 |
| `baseline_cnn-rotation-focal_g2` | `baseline_cnn` | ours | 64 | **0.8600** | [0.8450, 0.8739] | 0.613 | 47/50 ⚠ | 4 |
| `v31-dilated_control_64` | `dilated_style` | ours | 64 | **0.8216** | [0.8065, 0.8347] | 0.722 | 17/27 | 168 |
| `baseline_v2-rotation-focal_g1-sampler` | `baseline_v2` | ours | 64 | **0.8084** | [0.7912, 0.8223] | 0.517 | 49/50 ⚠ | 4 |
| `baseline_v2-rotation-sampler` | `baseline_v2` | ours | 64 | **0.8061** | [0.7908, 0.8194] | 0.492 | 48/50 ⚠ | 3 |
| `baseline_v2-rotation-focal_g2-sampler` | `baseline_v2` | ours | 64 | **0.7995** | [0.7829, 0.8133] | 0.449 | 45/50 ⚠ | 4 |
| `v25-control` | `baseline_v2` | ours | 64 | **0.7830** | [0.7677, 0.7963] | 0.358 | 38/40 ⚠ | 2 |
| `v25-focal-g2-sampler` | `baseline_v2` | ours | 64 | **0.7828** | [0.7672, 0.7965] | 0.376 | 36/40 ⚠ | 3 |
| `v25-focal-g1-sampler` | `baseline_v2` | ours | 64 | **0.7724** | [0.7561, 0.7864] | 0.337 | 36/40 ⚠ | 3 |
| `v29-resnet18_frozen_onehot` | `frozen_backbone_mlp` | pretrained | 128 | **0.7351** | [0.7178, 0.7507] | 0.401 | 37/47 | 17 |
| `v29-mobilenet_frozen_grayscale` | `frozen_backbone_mlp` | pretrained | 128 | **0.7308** | [0.7141, 0.7460] | 0.405 | 31/41 | 10 |
| `v25-focal-g2` | `baseline_v2` | ours | 64 | **0.7250** | [0.7084, 0.7386] | 0.000 | 39/40 ⚠ | 3 |
| `v25-focal-g1` | `baseline_v2` | ours | 64 | **0.7201** | [0.7036, 0.7339] | 0.000 | 39/40 ⚠ | 3 |
| `baseline_v2-rotation-focal_g2` | `baseline_v2` | ours | 64 | **0.7024** | [0.6847, 0.7168] | 0.000 | 19/26 | 4 |
| `baseline_v2-rotation-focal_g1` | `baseline_v2` | ours | 64 | **0.7005** | [0.6844, 0.7144] | 0.000 | 19/26 | 4 |
| `v29-resnet18_frozen_grayscale` | `frozen_backbone_mlp` | pretrained | 128 | **0.6877** | [0.6687, 0.7035] | 0.348 | 21/31 | 17 |
| `resnet18-224-frozen-rotation-p05` | `resnet18` | pretrained | 224 | **0.6600** | [0.6444, 0.6742] | 0.220 | 3/6 | 105 |

---

## Per-class F1, at each run's best epoch

Macro-F1 weights all nine classes equally, so the averages above hide where models actually
differ. They differ in three places and nowhere else.

| run | Center | Donut | Edge-Loc | Edge-Ring | Loc | Near-full | Random | Scratch | none |
|---|---|---|---|---|---|---|---|---|---|
| `v32-resnet34_finetune` | 0.935 | 0.855 | 0.863 | 0.983 | 0.809 | 0.966 | 0.926 | 0.809 | 0.992 |
| `v32-vit_b_32_finetune` | 0.938 | 0.872 | 0.866 | 0.985 | 0.807 | 0.915 | 0.921 | 0.829 | 0.992 |
| `v33-convnext_tiny_finetune` | 0.922 | 0.855 | 0.847 | 0.983 | 0.808 | 0.966 | 0.929 | 0.825 | 0.990 |
| `dilated-style-64-dihedral8` | 0.918 | 0.887 | 0.854 | 0.984 | 0.799 | 0.918 | 0.934 | 0.811 | 0.990 |
| `v28-convnext_big_128` | 0.941 | 0.860 | 0.860 | 0.983 | 0.811 | 0.949 | 0.901 | 0.793 | 0.991 |
| `v30-finetune_encoder_1e-5` | 0.921 | 0.843 | 0.849 | 0.979 | 0.802 | 0.947 | 0.895 | 0.817 | 0.991 |
| `v33-efficientnet_b0_finetune` | 0.923 | 0.878 | 0.835 | 0.980 | 0.772 | 0.949 | 0.928 | 0.776 | 0.990 |
| `v27-resnet_style` | 0.928 | 0.895 | 0.829 | 0.978 | 0.796 | 0.915 | 0.911 | 0.770 | 0.989 |
| `v27-convnext_style` | 0.930 | 0.869 | 0.840 | 0.977 | 0.786 | 0.931 | 0.912 | 0.760 | 0.990 |
| `v28-vit_style_128` | 0.911 | 0.841 | 0.838 | 0.979 | 0.765 | 0.966 | 0.916 | 0.668 | 0.988 |
| `v27-baseline_cnn` | 0.924 | 0.857 | 0.820 | 0.979 | 0.760 | 0.857 | 0.903 | 0.694 | 0.989 |
| `v31-dilated_rotation_64` | 0.910 | 0.872 | 0.828 | 0.978 | 0.782 | 0.949 | 0.929 | 0.731 | 0.988 |
| `v31-dilated_control_64` | 0.797 | 0.739 | 0.775 | 0.974 | 0.606 | 0.933 | 0.852 | 0.731 | 0.988 |
| `v29-resnet18_frozen_onehot` | 0.795 | 0.791 | 0.471 | 0.947 | 0.460 | 0.877 | 0.884 | 0.417 | 0.973 |

**`Edge-Ring` and `none` are solved** -- 0.97-0.99 for every model including the frozen
backbone. They carry 6,776 and 103,199 training wafers, and nothing any experiment did moved
them. They are also 78% of the validation set, which is why accuracy is useless here and
macro-F1 is the metric.

**`Loc`, `Scratch` and `Edge-Loc` are the whole spread.** Between the best and worst
non-frozen run: `Loc` 0.606 -> 0.811, `Scratch` 0.668 -> 0.829, `Edge-Loc` 0.775 -> 0.866.
Every other class moves by under 0.06. `Loc` and `Edge-Loc` are the confusable pair -- a
defect cluster that is either near the edge or not -- and `Scratch` is a one-die-wide line
that survives no downsampling. All three are about **spatial context and resolution**, which
is why rotation, dilation and 128px are the interventions that worked.

**`Near-full` is noise, not signal.** It swings 0.857-0.966 across runs with **30 validation
wafers**. A single wafer is 3 points of F1. Do not read a ranking from that column.

**`Donut` never became the problem** -- 0.84-0.90 everywhere, on 389 training wafers. Rarity
alone did not make a class hard; `Near-full` has 104 and scores 0.92+. What makes a class
hard here is spatial ambiguity, not sample count.

**The dilation control reads cleanest per-class.** `v31-dilated_control_64` loses to its
dilated twin on `Center` (0.797 vs 0.910), `Loc` (0.606 vs 0.782) and `Donut` (0.739 vs
0.872) -- all shapes defined by extent -- while `Scratch` is unchanged (0.731 both) and
`Edge-Ring`/`none` are untouched. A 19x19 receptive field cannot see a whole `Center` blob on
a 64x64 map; a 63x63 one can.

---

## Models

Best measured validation macro-F1 per architecture, on the settled pipeline
(rotation p=0.5, inverse-sqrt sampler, cross-entropy, one-hot). Pretrained rows are
fine-tuned at encoder lr 1e-5 with our own 256-unit MLP head unless marked frozen.

| model | kind | params | px | best macro-F1 | s/epoch | note |
|---|---|---|---|---|---|---|
| `baseline_cnn` | ours | 157k | 64 | 0.8800 | 3.4 | the test-bed for the whole sweep |
| `dilated_style` | ours | 298k | 64 | **0.8995** | 172 | best of ours; dilation worth +0.064 against its own control |
| `densenet_style` | ours | 304k | 64 | 0.8853 | 114 | feature reuse |
| `convnext_style` | ours | 414k | 64 | 0.8883 | 32 | best value of ours. 0.8988 at 128px/2.68M; 0.8862 at 64px/2.68M; dilation 0.8957 (no gain) |
| `inception_style` | ours | 799k | 64 | 0.8779 | 138 | parallel kernel sizes |
| `vit_style` | ours | 2.83M | 128 | 0.8744 | 13 | no conv prior; converged, still last of ours |
| `resnet_style` | ours | 2.83M | 64 | 0.8900 | 51 | 0.8878 at 128px |
| `baseline_v2` | ours | 34k | 64 | 0.8084 | 3.4 | dropped: capacity cut that rotation had made unnecessary |
| `resnet18` | pretrained | 11.3M | 128 | 0.8938 | 37 | frozen: 0.7351. **+0.159 from unfreezing** |
| `resnet34` | pretrained | 21.4M | 128 | **0.9041** | 58 | best overall |
| `resnet50` | pretrained | 24.0M | 128 | — | — | crashed at epoch 15, `Scratch` still climbing |
| `mobilenet_v3_small` | pretrained | 1.1M | 128 | 0.7308 | 10 | frozen only |
| `efficientnet_b0` | pretrained | 4.3M | 128 | 0.8923 | 118 | best parameters-per-point of the pretrained set |
| `convnext_tiny` | pretrained | 28.0M | 128 | **0.9027** | 232 | **best `Scratch` measured, 0.816** |
| `swin_t` | pretrained | 27.7M | 128 | — | — | running |
| `vit_b_32` | pretrained | 87.7M | 224 | **0.9029** | 62 | 224px forced by its positional embeddings |
| `vit_b_16` | pretrained | 86.0M | 224 | — | — | untried: ~3x b_32's cost |
| `swin_v2_t` | pretrained | 27.8M | 128 | — | — | untried |
| `maxvit_t` | pretrained | 30.5M | 224 | — | — | abandoned: >10 min/epoch even on an A100 |

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
