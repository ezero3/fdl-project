# Roadmap

Wafer-map failure-pattern classification on MIR WM-811K: 9 classes, ~173k labeled images, severe imbalance (the majority class is ~992× the rarest).

**Already done and on `main`:** dataset exploration, a frozen train/validation/test split, a shared evaluation pipeline with confidence intervals, a preprocessing pipeline, and a validated imbalance policy. What remains is the modeling work.

Reasoning behind every choice below is in [`design-notes.md`](design-notes.md). Read that before changing a decision; read this to pick up a task.

Priorities: **P0** blocks other people, **P1** is needed for a defensible submission, **P2** improves results, **P3** is stretch.

## How we run experiments

Everything that is **not** the model itself — preprocessing, augmentation, imbalance handling, optimizer settings — is compared on **one fixed model**, so each comparison is controlled and cheap. Only once a setup wins do we carry it to the real models and re-tune the parts that must change with the architecture.

What transfers between models and what does not: augmentation and imbalance results usually carry across architectures; input size, learning rate and epoch budget never do and must be re-tuned per model.

---

## P0 — Do first, everything else waits on these

### 1. Experiment config files and a runner script
One YAML per experiment (model, input size, augmentation, imbalance strategy, optimizer setup/budget, seed), loaded into the existing validated config objects. A `train.py` that takes a YAML path and runs end to end.

*Why:* three people need to run different setups on Colab and compare results afterwards. Without this, everyone hand-edits notebooks and nothing is comparable.
*Constraint:* the config must not narrow what PyTorch can express. Optimizer choice, learning-rate schedules (cosine, warmup, step), and **separate learning rates per parameter group** all have to be reachable from YAML — the last one is required for fine-tuning, see item 5.
*Done when:* `python train.py experiments/<name>.yaml` trains, evaluates, and writes results under `output/`, and the config is recorded in the run's `metrics.json`.

### 2. Replace `set_reproducible_seed` with one seed-everything function
Model it on Lightning's `seed_everything`: seed Python, NumPy and PyTorch, set `PYTHONHASHSEED`, give each dataloader worker its own derived seed, set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA initializes, and enable deterministic algorithms in **warn-only** mode.

*Why:* the current version raises a `RuntimeError` mid-training on GPU, because several convolution backward kernels have no deterministic implementation. It also leaves dataloader workers unseeded, which silently collapses augmentation to a single repeated view as soon as workers are used. (GPU random state itself is already fine — `torch.manual_seed` seeds all devices internally.)
*Done when:* a short GPU run completes, two runs with the same seed give the same validation macro-F1, and that still holds with workers enabled.

### 3. Raise the training budget
The shared `TrainingConfig` defaults to **4 epochs** — that was deliberately chosen for a fast strategy comparison, not for real training. Anything trained with the defaults is undertrained.

*Why:* the current defaults make every model look equally mediocre and hide real differences between them.
*Done when:* configs specify e.g. 40 epochs with early-stopping patience 5, and at least one model has actually stopped early rather than hitting the cap.

### 4. Checkpoint to Google Drive during training
Write model weights, optimizer state, epoch number and RNG state to Drive at the end of every epoch, and support resuming from the last one.

*Why:* Colab disconnects without warning and wipes local disk. Once runs are 40 epochs rather than 4, losing one costs hours of GPU time we may not get back the same day.
*Done when:* a run killed halfway resumes from its last checkpoint and finishes with the same result as an uninterrupted run.

---

## P1 — Needed for a submission that holds up

### 5. Three models
Two built by us, one pretrained, per the course requirement.

- **From scratch:** the existing small CNN is the baseline; a second, deeper design is one of ours.
- **Pretrained:** MobileNetV3 or ResNet18, **fine-tuned or built upon — not used as a frozen feature extractor.** Train the whole network, but give the pretrained encoder a much smaller learning rate than the newly initialised head (10–100× smaller is the usual range). Freezing the encoder is also worth one run as a cheap, fast baseline — it trains in minutes and tells you how much the fine-tuning actually buys — but it is a comparison point, not the plan.
- Use a **learning-rate schedule**; cosine annealing with a short warmup is a sensible default.
- Pretrained backbones need **224×224 input** rather than 64×64, because their filters expect that scale. This is a config change, not new code.

*Why:* it's the deliverable.
*Done when:* three trained models, each with saved weights and evaluation artifacts, all evaluated by the shared pipeline.

### 6. Metric logging across people
Weights & Biases free tier. The training loop already accepts a per-epoch callback that receives a flat metrics dict, so this is one argument, not a refactor. **Use a private project, not a public one.**

*Why:* three people × several configs = one comparable table instead of screenshots in a group chat.
*Done when:* all runs from all three of us appear in one project with their config attached.

### 7. In-memory caching of wafer maps
Cache the raw wafer maps as `uint8` arrays (~350 MB for all labeled data) and do the geometry and encoding on the fly. Do **not** cache preprocessed float tensors — that is ~8.5 GB and will exhaust Colab's memory.

*Why:* removes repeated work per epoch and lets the 2 GB source dataframe be dropped.
*Done when:* an epoch is measurably faster and memory use is stable across epochs.

### 8. Report results with uncertainty, and touch the test split once
The evaluation pipeline can produce bootstrap confidence intervals; use them in the final comparison. Rare classes have very little test support (15 images for the rarest), so point estimates alone are misleading.

*Why:* two models within roughly 0.04 test macro-F1 of each other are not distinguishable, and claiming a winner there is not defensible.
*Done when:* the final table reports intervals, and the test split has been evaluated exactly once, at the end.

---

## P2 — Improves the numbers

### 9. Data augmentation
Use the 8 exact symmetries — 4 rotations by 90° and their mirrors. These are index permutations, so they introduce no interpolation and no invalid pixel values, and they preserve every class label.

**Do not use free-angle rotation** (destroys thin scratch patterns) and **be careful with translation**: shifting a localized defect toward the wafer edge can genuinely turn a `Loc` into an `Edge-Loc` while keeping the old label.

*Why:* the minority classes are oversampled with replacement today, so the model sees the same handful of images repeatedly. Augmentation turns each repeat into a different view — the two techniques compound.
*Done when:* augmentation applies to training only, validation and test are provably untouched, and a fixed seed reproduces the same views.

### 10. Attention block on one backbone
Defects occupy a small fraction of each image, which is the standard case for spatial attention (e.g. CBAM).

*Why:* cheap, plausible gain, and a good comparison to show in the presentation.
*Done when:* one backbone is reported with and without it, everything else held fixed.

### 11. Cap majority-class exposure per epoch
Optional sixth imbalance strategy: limit the majority class to ~12k images **per epoch, resampled each epoch**, rather than deleting rows permanently. Compare it on validation against the current policy.

*Why:* keeps all the hard negatives available across training while reducing per-epoch dominance.
*Done when:* it has been screened on validation like the other strategies, and the better one is used.

### 12. Free post-hoc wins — do these near the end
Two techniques that need no retraining:

- **Test-time augmentation:** average predictions over the same 8 symmetries used for training augmentation. Costs 8× inference (seconds), typically worth 1–2 points of macro-F1, and never touches labels.
- **Per-class decision thresholds** tuned on validation to maximise macro-F1, instead of always taking the arg-max.

*Why:* the cheapest remaining accuracy in the project.
*Done when:* each is measured on validation, and only the ones that actually help are used for the single test evaluation.

*Deliberately excluded:* ensembling the three final models. It would raise the headline number, but it is competition tuning rather than a deep-learning result, and it obscures the architecture comparison that the report is actually about.

### 13. Alternative long-tail methods worth one run each
Two well-established techniques we have not tried, both cheap:

- **Logit adjustment:** shift each class's output score by the log of its training frequency, moving the decision boundary to where it would sit under balanced classes. Applied *post-hoc* it needs no retraining at all. Note it must be compared against the **unweighted baseline**, not stacked on the current sampler — the sampler already flattens the effective prior, so doing both double-corrects.
- **Two-stage (decoupled) training:** train the whole network on the natural, imbalanced distribution, then freeze it and retrain **only the final classifier layer** with balanced sampling. Learning features on the natural distribution and rebalancing only the classifier beats one-stage resampling on long-tailed data, and the second stage takes minutes.

*Why:* our current policy was chosen from five candidates under a 4-epoch budget. These two work by a different mechanism than anything in that comparison — reweighting changes what mistakes cost, resampling changes what the model sees, these change where the decision boundary sits.
*Done when:* each is screened on validation against the current sampler, on the fixed test-bed model.

---

## P3 — Stretch, only with spare time

### 14. Pseudo-labeling the unlabeled wafers
Train the best model, predict on the ~617k unlabeled wafers, keep only high-confidence predictions, and retrain with them included. Costs roughly two training runs.

**Guard rails are not optional here.** The majority class is ~89% of the labeled data, so pseudo-labels will be overwhelmingly that class, and the model is least reliable on exactly the rare classes we want more of — unguarded, this amplifies the imbalance it is meant to fix. Use a high confidence threshold, a **per-class cap** on how many pseudo-labels each class may contribute, and exclude any class whose validation precision is poor.

*Worth doing only if a full training run turns out to be fast.*

### 15. Self-supervised pretraining on the unlabeled data
~617k unlabeled wafers are available and already filtered so none of them come from validation or test groups. Pretrain an autoencoder on them, then fine-tune a classifier head on the labeled set. Reconstruct pixels as a **3-way classification per pixel**, not regression — otherwise the decoder outputs meaningless in-between values.

*Rough cost:* ~30 minutes of GPU pretraining; most of the effort is in the fine-tuning protocol.

*Alternative worth knowing:* **Mean Teacher with a supervised contrastive loss**, published on this exact dataset with a ~4.5 point F1 gain over a plain ResNet18. Mean Teacher keeps an exponential moving average of the model as a "teacher", shows it and the student two differently augmented views of the same *unlabeled* wafer, and penalises disagreement — so unlabeled data is used without ever needing its label. The contrastive part pulls same-class embeddings together on the labeled data. It is a **single training run at ~1.5–2× cost**, not a separate pretraining stage, and our 8 exact symmetries are ideal as the two views. See `design-notes.md` §12.

### 16. LoRA fine-tuning of a larger pretrained model
Adapt a backbone too large to fine-tune outright (a ViT, say) by training small low-rank adapters while the original weights stay fixed. Memory-cheap, and an unusual technique to show in a course project.

*Why:* it turns "too big to fine-tune, so we froze it" into "too big to fine-tune, so we adapted it properly".

---

## Known limitations to write up

- Thin scratch patterns lose 30–91% of their defective pixels when a large wafer is downscaled to 64×64. This affects ~15% of that class. It is documented and is *not* the main reason that class scores poorly — it is rare and undertrained — but it belongs in the limitations section.
- The imbalance strategy comparison ran on a 4-epoch budget; its ranking below the winner is within seed-to-seed noise.

## If time runs short

Items **1–8** are the minimum for a submission that stands up to questions. Items 9–13 are where the remaining accuracy is — **12 is the cheapest of all and should not be skipped**. Items 14–16 are stretch, and good presentation material either way.

External benchmarks and why most published WM-811K numbers are not comparable to ours: `design-notes.md` §12. Read it before quoting anyone's accuracy.
