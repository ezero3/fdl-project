# v30 — how much should the pretrained encoder move?

v29 froze the backbone completely. This series unfreezes it and varies only *how fast* it
is allowed to change, holding everything else at the v29 settings so the arms form one
ladder with v29 on its bottom rung:

| arm | encoder lr | ratio to head | measured |
|---|---|---|---|
| `v29-resnet18_frozen_onehot` | 0 | frozen | **0.7351** |
| `08_finetune_encoder_1e-5` | 1e-5 | 1/70 | |
| `09_finetune_encoder_1e-4` | 1e-4 | 1/7 | |
| `10_finetune_uniform` | 7e-4 | 1/1 | |

The head is 7e-4 in every arm, the same MLP as v29, on `resnet18` + `one_hot` — the best
v29 arm. From scratch on the same splits: **0.8900**.

## What the ladder is asking

Frozen says "ImageNet features, used as-is". Uniform lr says "ImageNet as an initialisation
and nothing more". The two middle rungs are the actual transfer-learning claim: that a
pretrained encoder nudged gently beats both. If 08/09 do not beat both ends, there was no
transfer to exploit — which, given frozen lost by 0.155, is a live possibility.

## Deliberately no warmup schedule

`configs/train/resnet18_224_finetune.yaml` uses cosine warmup, and for fine-tuning that is
the textbook choice: a large gradient from the randomly initialised head can wreck
pretrained features in the first few hundred steps. It is left out here because v29 ran at
constant lr, and adding a schedule at the same time as unfreezing would confound the two.
Warmup is the first follow-up if `10_finetune_uniform` collapses.

## Cost

Fine-tuning backpropagates through all 11.2M encoder parameters instead of none, so expect
roughly 3x the v29 epoch cost.
