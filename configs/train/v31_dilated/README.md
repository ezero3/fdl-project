# v31 — is dilation actually the reason?

`dilated-style-64-dihedral8` scored **0.8995** [0.8856, 0.9104] with `Scratch` at **0.793** —
the best numbers in the project, above `v27-resnet_style` (0.8900). Two things make that
result provisional:

1. It ran with **`dihedral8`**, not the settled `rotation`. Every other result in v27–v30
   uses rotation, and rotation beat dihedral8 on `baseline_cnn` (0.8800 vs 0.8587), so this
   arm has never been measured on the pipeline everything else uses.
2. Its best epoch was **45 of 50** — it was still improving when the cap stopped it. 0.8995
   is a floor.

Three arms, no new code, 298,377 parameters in every one of them:

| arm | rates | receptive field | size | asks |
|---|---|---|---|---|
| `11_dilated_rotation_64` | (1,2,4,8)x2 | 63x63 | 64 | does the best result survive the switch to rotation, given room to run? |
| `12_dilated_control_64` | all 1 | 19x19 | 64 | **is it the dilation?** |
| `13_dilated_rotation_128` | (1,2,4,8)x2 | 63x63 | 128 | does dilation compound with resolution? |

## Why 12 is the arm that matters

Same architecture, same depth, same 298,377 parameters, same everything — the *only*
difference is that the kernels are spaced out. If 11 beats 12, dilation is the cause and
nothing else can be. If they tie, the dilated model's advantage is its full-resolution
stack (it never downsamples at all) and the dilation rates are incidental — still a real
finding, and a different sentence in the presentation.

Bolting a dilation option onto `resnet_style` or `convnext_style` would not answer this:
neither supports it, adding it is real surgery, and the resulting arm would differ from
`v27-resnet_style` in two ways at once.

## Why 13 downsamples, which is a compromise

Dilation's whole argument is context without losing resolution, so `downsample_every: 0`
(the default) is the honest version. At 128x128 that costs ~4x the 64x64 epoch, or roughly
5 hours, which is not available. `downsample_every: 3` pools after blocks 3 and 6, keeping
the first three blocks at full 128x128 where the thin-`Scratch` argument applies, and
lands near 2x the 64x64 cost. Say so if 13 underperforms — it is not a clean test of
dilation at 128.

## Context for reading the results

`v28-convnext_big_128` reached `Scratch` **0.804** by epoch 28, the best seen. Resolution is
helping the class dilation is supposed to help, so 13 is testing whether the two compound
or overlap.
