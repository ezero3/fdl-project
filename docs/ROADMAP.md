# Roadmap

Wafer-map failure-pattern classification on MIR WM-811K: 9 classes, ~173k labeled images, severe imbalance (the majority class is ~992× the rarest).

**Already done and on `main`:** dataset exploration, a frozen train/validation/test split, a shared evaluation pipeline with confidence intervals, a preprocessing pipeline, and a validated imbalance policy. What remains is the modeling work.

Reasoning behind every choice below is in [`design-notes.md`](design-notes.md). Read that before changing a decision; read this to pick up a task.

Priorities: **P0** blocks other people, **P1** is needed for a defensible submission, **P2** improves results, **P3** is stretch.

---

## P0 — Do first, everything else waits on these

### 1. Experiment config files and a runner script
One YAML per experiment (model, input size, augmentation, imbalance strategy, optimizer budget, seed), loaded into the existing validated config objects. A `train.py` that takes a YAML path and runs end to end.

*Why:* three people need to run different setups on Colab and compare results afterwards. Without this, everyone hand-edits notebooks and nothing is comparable.
*Done when:* `python train.py experiments/<name>.yaml` trains, evaluates, and writes results under `output/`, and the config is recorded in the run's `metrics.json`.

### 2. Fix reproducible seeding for GPU
`set_reproducible_seed` enables strict deterministic algorithms. On CUDA that raises a `RuntimeError` mid-training for several convolution backward kernels unless `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set before CUDA initializes.

*Why:* every GPU run crashes or silently varies until this is fixed. Small change, blocks everything.
*Done when:* a short GPU training run completes, and two runs with the same seed give the same validation macro-F1.

### 3. Raise the training budget
The shared `TrainingConfig` defaults to **4 epochs** — that was deliberately chosen for a fast strategy comparison, not for real training. Anything trained with the defaults is undertrained.

*Why:* the current defaults make every model look equally mediocre and hide real differences between them.
*Done when:* configs specify e.g. 40 epochs with early-stopping patience 5, and at least one model has actually stopped early rather than hitting the cap.

---

## P1 — Needed for a submission that holds up

### 4. Three models
Two built by us, one pretrained, per the course requirement.

- **From scratch:** the existing small CNN is the baseline; a second, deeper design is one of ours.
- **Pretrained:** MobileNetV3 or ResNet18. **These need a different input size** — 224×224 rather than 64×64, because pretrained filters expect that scale. The preprocessing config takes a target size, so this is a config change, not new code.

*Why:* it's the deliverable.
*Done when:* three trained models, each with saved weights and evaluation artifacts, all evaluated by the shared pipeline.

### 5. Metric logging across people
Weights & Biases free tier. The training loop already accepts a per-epoch callback that receives a flat metrics dict, so this is one argument, not a refactor. **Use a private project, not a public one.**

*Why:* three people × several configs = one comparable table instead of screenshots in a group chat.
*Done when:* all runs from all three of us appear in one project with their config attached.

### 6. In-memory caching of wafer maps
Cache the raw wafer maps as `uint8` arrays (~350 MB for all labeled data) and do the geometry and encoding on the fly. Do **not** cache preprocessed float tensors — that is ~8.5 GB and will exhaust Colab's memory.

*Why:* removes repeated work per epoch and lets the 2 GB source dataframe be dropped.
*Done when:* an epoch is measurably faster and memory use is stable across epochs.

### 7. Report results with uncertainty, and touch the test split once
The evaluation pipeline can produce bootstrap confidence intervals; use them in the final comparison. Rare classes have very little test support (15 images for the rarest), so point estimates alone are misleading.

*Why:* two models within roughly 0.04 test macro-F1 of each other are not distinguishable, and claiming a winner there is not defensible.
*Done when:* the final table reports intervals, and the test split has been evaluated exactly once, at the end.

---

## P2 — Improves the numbers

### 8. Data augmentation
Use the 8 exact symmetries — 4 rotations by 90° and their mirrors. These are index permutations, so they introduce no interpolation and no invalid pixel values, and they preserve every class label.

**Do not use free-angle rotation** (destroys thin scratch patterns) and **be careful with translation**: shifting a localized defect toward the wafer edge can genuinely turn a `Loc` into an `Edge-Loc` while keeping the old label.

*Why:* the minority classes are oversampled with replacement today, so the model sees the same handful of images repeatedly. Augmentation turns each repeat into a different view — the two techniques compound.
*Done when:* augmentation applies to training only, validation and test are provably untouched, and a fixed seed reproduces the same views.

### 9. Attention block on one backbone
Defects occupy a small fraction of each image, which is the standard case for spatial attention (e.g. CBAM).

*Why:* cheap, plausible gain, and a good comparison to show in the presentation.
*Done when:* one backbone is reported with and without it, everything else held fixed.

### 10. Cap majority-class exposure per epoch
Optional sixth imbalance strategy: limit the majority class to ~12k images **per epoch, resampled each epoch**, rather than deleting rows permanently. Compare it on validation against the current policy.

*Why:* keeps all the hard negatives available across training while reducing per-epoch dominance.
*Done when:* it has been screened on validation like the other strategies, and the better one is used.

---

## P3 — Stretch, only with spare time

### 11. Self-supervised pretraining on the unlabeled data
~617k unlabeled wafers are available and already filtered so none of them come from validation or test groups. Pretrain an autoencoder on them, then fine-tune a classifier head on the labeled set. Reconstruct pixels as a **3-way classification per pixel**, not regression — otherwise the decoder outputs meaningless in-between values.

*Rough cost:* ~30 minutes of GPU pretraining; most of the effort is in the fine-tuning protocol.

### 12. Generative augmentation — future work, do not attempt now
Synthesizing minority samples with a VAE/GAN appears in the literature but cannot create information absent from 104 examples, and needs a day of tuning plus strict guarantees that synthetic data never reaches validation. Mention it as future work.

---

## Known limitations to write up

- Thin scratch patterns lose 30–91% of their defective pixels when a large wafer is downscaled to 64×64. This affects ~15% of that class. It is documented and is *not* the main reason that class scores poorly — it is rare and undertrained — but it belongs in the limitations section.
- The imbalance strategy comparison ran on a 4-epoch budget; its ranking below the winner is within seed-to-seed noise.

## If time runs short

Items **1–7** are the minimum for a submission that stands up to questions. Items 8–10 are where the remaining accuracy is. Items 11–12 are presentation material either way.
