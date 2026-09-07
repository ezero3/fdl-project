# Phase 0 results — what an epoch costs

Measured 2026-09-07 on a Colab T4 (16 GB, compute capability 7.5), 8 vCPUs, torch
2.11.0+cu128, Python 3.13.15. Raw CSVs in `output/phase0/`; the notebook that produced them
is `notebooks/experiments/base_notebook.ipynb`.

Every figure is a property of *that machine*. Re-run before trusting them on other hardware.

**Method.** The harness calls the real `train_one_epoch` and `evaluate_model` over a
stratified 20,000-row subset (4,000 for the 224x224 arm) and extrapolates to the full
121,063-row training split. The first pass of every measurement is discarded — cuDNN
autotuning, dataloader worker startup and CUDA context creation land there and none recur.
GPU utilisation is sampled from `nvidia-smi` on a background thread rather than inferred.

**Precision.** Training ran in **fp16 mixed precision** throughout (`trainer.amp: true`;
`torch.amp.autocast` plus `GradScaler`, gradients unscaled before clipping). The T4 has no
bf16 — capability 7.5, bf16 needs 8.0 — so fp16 is the only option here. **Validation ran in
fp32**: `collect_predictions` has no autocast. That asymmetry is deliberate for numerical
safety in the reported metrics, but it means every `validation_epoch_seconds` below is an
fp32 number and is a standing optimisation opportunity (see "Open items").

---

## 1. The three phase-1 candidates

Stock pipeline as it was: 64x64, batch 512, `num_workers: 2`, `inverse_sqrt_sampler`.

| model | parameters | train epoch | validation | full epoch | peak memory | GPU util |
|---|---|---|---|---|---|---|
| `baseline_cnn` | 156,937 | 18.6s | 5.0s | **23.6s** | 0.29 GiB | 15.7% |
| `dilated_style` | 298,377 | 135.0s | 33.4s | **168.4s** | 3.07 GiB | 97.4% |
| `densenet_style` | 303,937 | 139.8s | 23.7s | **163.6s** | 3.94 GiB | 96.3% |

**Parameter count predicts neither time nor memory.** `dilated_style` and `densenet_style`
carry roughly 2x `baseline_cnn`'s parameters and cost **7x** the wall-clock and **10x** the
memory. The reason is architectural, not incidental: both hold high spatial resolution deep
into the network — `dilated_style` by design, since refusing to downsample is the entire
argument for it on thin `Scratch` defects — and activation memory scales with resolution,
not with weights.

A prediction recorded in `training-plan.md` was half right: it said `dilated_style` would be
the most expensive of the three despite its parameter count. It is expensive, but
`densenet_style` is the heavier of the two on memory (3.94 vs 3.07 GiB). The general claim
held; the specific ordering did not.

**This settles the phase-1 test-bed.** The plan said to break a tie on cost when the
confidence intervals overlap. The cost difference is a factor of seven, so `baseline_cnn` is
the test-bed regardless of how the accuracy comparison lands — it is the instrument for the
~16-arm phase-2 sweep, not a result.

---

## 2. Batch size is not the lever

`baseline_cnn`, `num_workers: 2`, sweeping batch size:

| batch | samples/s | train epoch | steps/epoch | updates in 20 epochs | peak memory | GPU util |
|---|---|---|---|---|---|---|
| 64 | 5,642 | 21.5s | 1,892 | 37,840 | 0.06 GiB | 20.2% |
| 128 | 5,710 | 21.2s | 946 | 18,920 | 0.09 GiB | 16.7% |
| 256 | 6,132 | 19.7s | 473 | 9,460 | 0.16 GiB | 16.7% |
| 512 | 6,171 | 19.6s | 237 | 4,740 | 0.13 GiB | 14.2% |
| 1024 | 6,403 | 18.9s | 119 | 2,380 | 0.56 GiB | 15.6% |

**Throughput is flat.** A 16x increase in batch size bought 13% more samples per second
while cutting optimizer steps by 16x. Utilisation never exceeded 20%, so the GPU was idle
~80% of the time at every batch size — the bottleneck is upstream.

The shipped default of 512 was therefore close to the worst available choice: no faster than
256, and half the gradient updates. At 25 epochs it yields ~5,900 updates, which is thin for
training a network from scratch.

**"Fill the VRAM" is the wrong heuristic here, and the right one for the 224x224 arm.**
Filling memory is a proxy for saturating compute, and the two only coincide when the model
is large enough. Section 4 shows a case where they do.

---

## 3. Dataloader width — the actual lever

`baseline_cnn`, batch 256:

| num_workers | samples/s | full epoch | GPU util |
|---|---|---|---|
| 2 | 6,258 | 24.5s | 16.3% |
| 4 | 10,227 | 15.3s | 25.5% |
| **8** | **12,852** | **12.8s** | 21.6% |

**A 1.9x speedup from one config line.** `defaults.yaml` shipped `num_workers: 2` on a
machine with 8 cores.

This is consistent with the design rather than a defect in it: the letterbox and one-hot
encoding run per batch on CPU deliberately, because augmentation makes the transform output
differ every epoch, and caching preprocessed float tensors was measured and rejected at
6.1 GB for 224x224 (roadmap item 7). The transform cost is real — it was profiled at 76 us
per item at 64x64 — and with only two workers it could not be hidden behind GPU compute.

Even at 8 workers utilisation is ~22%, so the small models remain CPU-bound. There is more
headroom here, but it is not on the GPU side: a faster card would idle more, not less.

---

## 4. The 224x224 pretrained arm

`vit_b_16` / `vit_b_32` at 224x224, dihedral-8 augmentation on. These run at 95-99%
utilisation — genuinely compute-bound, the opposite regime to section 2.

**Frozen encoder, classifier only:**

| batch | peak memory | full epoch |
|---|---|---|
| 64 | 0.84 GiB | 792s |
| **128** | **1.35 GiB** | **766s** |
| 256 | 2.36 GiB | 833s |
| 512 | 4.38 GiB | 877s |
| 1024 | 8.42 GiB | 1,040s |
| 2048 | — | out of memory |

**Full fine-tuning:**

| batch | peak memory | full epoch |
|---|---|---|
| 32 | 3.33 GiB | 1,592s |
| 64 | 5.49 GiB | 1,539s |
| **128** | **9.80 GiB** | **1,482s** |
| 256 | — | out of memory |

**`vit_b_32`** (the cheap transformer counterpart — same 86M parameters, but 32x32 patches
give 49 tokens instead of 196 and ~4x less attention compute; torchvision ships no
`vit_small`):

| batch | peak memory | full epoch |
|---|---|---|
| 128 | 3.47 GiB | 458s |
| 256 | 5.76 GiB | 472s |

Per-run cost at 25 epochs: **frozen ViT-B 5.3 h · fine-tuned ViT-B 10.3 h · `vit_b_32` 3.2 h.**

Three things follow.

**Freezing buys 1.9x, not the 10x it is usually assumed to buy.** Removing the encoder
backward roughly halves the cost; the 86M-parameter forward pass still runs on every sample.
`resnet18_224_frozen.yaml` describes itself as training "in minutes" — for ResNet18 that may
hold, but the ViT-B equivalent is a 5.3-hour run and must be budgeted as one.

**The batch comment in `vit_b_16_224_finetune.yaml` was wrong.** It said `batch_size: 32` was
"what fits a T4". 128 fits in 9.8 GiB and is 7% faster. Corrected.

**Even here, bigger is not better past a point.** Frozen ViT-B tolerates batch 1024, but 128
is the fastest setting and gives 8x the optimizer steps. The memory ceiling and the
throughput optimum are different numbers, and only the second one matters once both fit.

---

## 5. Budget, and cross-validation

Phases 1 and 3 are where folds change a decision; phase 2's ~16 sweep arms stay on a single
split, since K x 16 is the expensive multiplication. Bootstrap resamples the same wafers;
cross-validation retrains on different ones, so the across-fold spread is the better basis
for choosing between models whose intervals overlap.

64x64 only, at 25 epochs per run:

| test-bed | K=1 | K=3 | K=5 |
|---|---|---|---|
| `baseline_cnn` | 11.8 h | 32.6 h | 53.3 h |
| `dilated_style` | 29.1 h | 49.8 h | 70.6 h |
| `densenet_style` | 28.6 h | 49.3 h | 70.1 h |

Split three ways, `baseline_cnn` at K=3 is **~10.8 h each** — comfortable. K=5 is ~17.8 h
each: possible, but the 224x224 arm above is *not* in these totals and dominates them.

**Recommendation: `baseline_cnn` as the test-bed, K=3, K=5 as a stretch.**

Cross-validation is **not yet implemented**. It needs `StratifiedGroupKFold` over the
train+validation pool only (155,654 rows; the 17,296 test rows stay frozen and outside every
fold), grouped by `lotName`, stratified so no fold is starved of `Near-full` at 0.1%, with
the folds generated once, committed under `data/splits/cv/`, and a `--fold` argument threaded
through the runner. Folds that differ per person or per model would destroy comparability
exactly as a re-shuffled split would.

---

## 6. Changes applied

| file | change | why |
|---|---|---|
| `configs/train/defaults.yaml` | `num_workers: 2 -> 8` | measured 1.9x, free |
| | `batch_size: 512 -> 256` | same speed, 2x the optimizer steps |
| | `lr: 1.0e-3 -> 7.0e-4` | square-root scaling for the halved batch |
| `configs/train/vit_b_16_224_finetune.yaml` | `batch_size: 32 -> 128` | fits in 9.8 GiB, 7% faster |
| `pyproject.toml` | `requires-python <3.13 -> <3.14` | Colab runs 3.13; pip refused the install |

Batch size and worker count are now **pipeline settings, held fixed across every phase-2
arm** — the comparisons are only controlled if the arms share them. Phase 3 is the
exception: batch size, learning rate, input size and epoch budget are precisely the four
things that do not transfer across architectures.

### Which GPU

| workload | measured utilisation | card |
|---|---|---|
| 64x64 `baseline_cnn` | 15.7% (2 workers) / 21.6% (8) | **T4** — dataloader-bound; a faster card idles more |
| 64x64 `dilated`/`densenet` | 96-97% | **L4** — compute-bound, roughly cost-neutral |
| 224x224 pretrained | 95-99% | **A100** — the budget sink, and the one place to spend |

Colab compute units burn at roughly T4 1x, L4 ~2.5x, A100 ~6x per hour, against speedups of
~2x and ~4-5x. L4 is close to a wash; **A100 knowingly trades budget for wall-clock** — worth
doing once, on the 224x224 arm, never on 64x64 screening that was already cheap.

Only the T4 column is measured. The L4 and A100 figures are spec-ratio estimates.

The strongest argument for A100 on the 224 arm is not speed: `vit_b_16` fine-tuning OOMs at
batch 256 on 16 GB, so a T4 caps the arm at 128. 40 GB removes the ceiling, and capability
8.0 brings bf16, retiring the fp16 + GradScaler path.

---

## 7. Open items

- **Validation runs in fp32.** `collect_predictions` has no autocast, so validation is
  roughly a third of a `dilated_style` epoch (33.4s of 168.4s) at full precision. Wrapping it
  in `torch.inference_mode` + autocast would cut that, at the cost of fp16 rounding in the
  reported metrics — which is exactly why it was not done. Worth measuring the metric
  difference before deciding.
- **The 64x64 models are still CPU-bound at 8 workers** (~22% utilisation). The remaining
  headroom is in the transform, not the card.
- **Cross-validation is unimplemented** — see section 5 for the design.
- **`resnet18_224_frozen.yaml` claims it trains "in minutes"** — unverified, and the ViT-B
  equivalent is 5.3 hours. Measure it before relying on the claim.
