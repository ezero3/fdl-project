# Roadmap

Wafer-map failure-pattern classification on MIR WM-811K: 9 classes, ~173k labeled images, severe imbalance (the majority class is ~992× the rarest).

**Already done and on `main`:** dataset exploration, a frozen train/validation/test split, a shared evaluation pipeline with confidence intervals, a preprocessing pipeline, and a validated imbalance policy. What remains is the modeling work.

Reasoning behind every choice below is in [`design-notes.md`](design-notes.md). Read that before changing a decision; read this to pick up a task.

Priorities: **P0** blocks other people, **P1** is needed for a defensible submission, **P2** improves results, **P3** is stretch.

> [!NOTE]
> **P0 items 1-4 and P1 item 6 are done** (branch `feature/training-framework`).
> What exists now:
>
> - `configs/train/*.yaml` + `uv run python scripts/train.py configs/train/<name>.yaml` — one file per experiment, deep-merged over `defaults.yaml`, with a typo in a key stopping the run rather than silently training the default. Optimizer choice, LR schedules and per-parameter-group learning rates are all reachable from YAML.
> - `uv run python scripts/inference.py --checkpoint <path> --split validation` — scores a checkpoint, saves full class probabilities for later post-hoc work, and refuses `--split test` without an explicit final-evaluation flag.
> - `seed_everything` in `src/fdl_project/training/seed.py`, wired into both dataloader factories, with per-worker seeds. Deterministic CUDA kernels are off by default — see item 2 for why.
> - Training budget now defaults to 40 epochs with early-stopping patience 5. The 4-epoch budget is now a documented, explicit choice inside the class-imbalance screening experiment only.
> - `CheckpointManager` writes `best.pt` plus a small rolling window of recent epochs (the newest is what resume reads) to any directory (point it at mounted Drive on Colab), atomically, and resumes the whole run — optimizer, schedule, scaler, epoch, best-so-far, history and RNG state.
> - W&B is a callback, off by default; turn it on per config. Logs the aggregate metrics, the learning rate and **per-class F1** each epoch, with the resolved config attached. See "Metric logging" below for what to set up on the platform.
> - `data.cache: true` copies the split's raw uint8 maps into memory and releases the 2 GB source table (item 7).
> - Dihedral-8 augmentation (item 10), CBAM attention on any backbone (item 11), and TTA plus per-class thresholds (item 13) are all implemented and configurable.
> - `src/fdl_project/` is now split into `config/ data/ models/ training/ evaluation/ analysis/`.
>
> Still open in P0/P1: three models (item 5), the final report with intervals (8), and qualitative error analysis (9).


## How we run experiments

Everything that is **not** the model itself — preprocessing, augmentation, imbalance handling, optimizer settings — is compared on **one fixed model**, so each comparison is controlled and cheap. Only once a setup wins do we carry it to the real models and re-tune the parts that must change with the architecture.

What transfers between models and what does not: augmentation and imbalance results usually carry across architectures; input size, learning rate and epoch budget never do and must be re-tuned per model.

---

## P0 — Do first, everything else waits on these

### 1. Experiment config files and a runner script
One YAML per experiment (model, input size, augmentation, imbalance strategy, optimizer setup/budget, seed), loaded into the existing validated config objects. A `train.py` that takes a YAML path and runs end to end.

*Why:* three people need to run different setups on Colab and compare results afterwards. Without this, everyone hand-edits notebooks and nothing is comparable.
*Constraint:* the config must not narrow what PyTorch can express. Optimizer choice, learning-rate schedules (cosine, warmup, step), and **separate learning rates per parameter group** all have to be reachable from YAML — the last one is required for fine-tuning, see item 5.
*Done when:* `uv run python scripts/train.py configs/train/<name>.yaml` trains, evaluates, and writes results under `output/runs/<name>/`, and the config is recorded in the run's `metrics.json`. ✅

### 2. Replace `set_reproducible_seed` with one seed-everything function
Model it on Lightning's `seed_everything`: seed Python, NumPy and PyTorch, set `PYTHONHASHSEED`, and give each dataloader worker its own derived seed.

**Deterministic CUDA kernels are deliberately *not* enabled.** They were in the first version of this item; that was wrong. Forcing them disables the cuDNN autotuner — a real throughput cost, because our input shapes are fixed and autotuning wins on exactly that case — and still does not give bit-exact results, since kernels with no deterministic implementation fall back silently under `warn_only`. What reproducibility actually needs is identical initialization, batch order, sampler draws and augmentation views, and seeding alone gives all of that. The kernel-level jitter that remains is far below the bootstrap intervals we report, so it cannot change a conclusion. `seed_everything(seed, deterministic=True)` is still there for a one-off strict check, and only that path sets `CUBLAS_WORKSPACE_CONFIG`.

*Why:* the current version raises a `RuntimeError` mid-training on GPU, because it enables deterministic algorithms in strict mode and several convolution backward kernels have no deterministic implementation. It also leaves dataloader workers unseeded, which silently collapses augmentation to a single repeated view as soon as workers are used. (GPU random state itself is already fine — `torch.manual_seed` seeds all devices internally.)
*Done when:* a short GPU run completes, two runs with the same seed give the same validation macro-F1 to within a small tolerance, and that still holds with workers enabled. ✅ *(code and CPU tests done; the GPU half still needs one Colab run to confirm)*

### 3. Raise the training budget
The shared `TrainingConfig` defaults to **4 epochs** — that was deliberately chosen for a fast strategy comparison, not for real training. Anything trained with the defaults is undertrained.

*Why:* the current defaults make every model look equally mediocre and hide real differences between them.
*Done when:* configs specify e.g. 40 epochs with early-stopping patience 5, and at least one model has actually stopped early rather than hitting the cap. ✅ *(defaults changed; the second half waits on a real run)*

### 4. Checkpoint to Google Drive during training
Write model weights, optimizer state, epoch number and RNG state to Drive at the end of every epoch, and support resuming from the last one.

*Why:* Colab disconnects without warning and wipes local disk. Once runs are 40 epochs rather than 4, losing one costs hours of GPU time we may not get back the same day.
*Done when:* a run killed halfway resumes from its last checkpoint and finishes with the same result as an uninterrupted run. ✅ *(covered by `tests/test_checkpoint.py`)*

---

## P1 — Needed for a submission that holds up

### 5. Three models
Two built by us, one pretrained, per the course requirement.

- **From scratch:** the existing small CNN is the baseline; a second, deeper design is one of ours.
- **Pretrained:** MobileNetV3 or ResNet18, **fine-tuned or built upon — not used as a frozen feature extractor.** Train the whole network, but give the pretrained encoder a much smaller learning rate than the newly initialised head (10–100× smaller is the usual range). Freezing the encoder is also worth one run as a cheap, fast baseline — it trains in minutes and tells you how much the fine-tuning actually buys — but it is a comparison point, not the plan.
- Use a **learning-rate schedule**; cosine annealing with a short warmup is a sensible default.
- Pretrained backbones need **224×224 input** rather than 64×64, because their filters expect that scale. This is a config change, not new code.
- **Run the input-representation comparison on the pretrained model.** Two arms: our default 3-channel one-hot, and the three states mapped to a grayscale-style image replicated to 3 channels with ImageNet mean/std normalization. Cheap, and it occasionally makes a large difference. The reason to bother: ImageNet filters were trained on natural photographs, and one-hot indicator channels look nothing like that distribution, so the pretrained weights may transfer poorly to them. The reason it is not the default: the grayscale arm reintroduces the false ordering that one-hot exists to remove (it implies a defect is "twice" a functional die), which is why we rejected single-channel input in the first place. Measure it, do not argue about it. Run it **only on the pretrained backbone** — a result here is about matching the pretraining distribution and will not transfer to our from-scratch models. Needs a small code change, not just config: `PreprocessingConfig` currently refuses to combine one-hot with any normalization and has no ImageNet strategy.

*Why:* it's the deliverable.
*Done when:* three trained models, each with saved weights and evaluation artifacts, all evaluated by the shared pipeline.

### 6. Metric logging across people
Weights & Biases free tier. The training loop already accepts a per-epoch callback that receives a flat metrics dict, so this is one argument, not a refactor. **Use a private project, not a public one.**

*Why:* three people × several configs = one comparable table instead of screenshots in a group chat.
*Done when:* all runs from all three of us appear in one project with their config attached. ⏳ *(code done; the account setup below is not)*

**What to set up on wandb.ai** — the plain free plan has no team access control, so three people cannot share a project on it:
1. Sign up with your institutional address (`@campus.unimib.it`).
2. Apply for the free **Academic** plan: free forever for students, unlimited teams, 200 GB, up to 100 seats. This is the step that makes a shared project possible.
3. Create a team (entity), e.g. `unimib-fdl`, and invite the other two.
4. Create the project inside that team and set its visibility to **Private**.
5. Each person generates their own API key and stores it in Colab Secrets as `WANDB_API_KEY` — never pasted into a cell.
6. Put `entity` and `project` into `configs/train/defaults.yaml` and flip `enabled: true`.

While approval is pending, `mode: offline` records runs locally and `wandb sync` uploads them later, so nobody is blocked.

### 7. In-memory caching of wafer maps
Cache the raw wafer maps as `uint8` arrays (~350 MB for all labeled data) and do the geometry and encoding on the fly. Do **not** cache preprocessed float tensors — that is ~8.5 GB and will exhaust Colab's memory.

*Why:* lets the 2 GB source dataframe be dropped.
*Done when:* the source table is released and memory use is stable across epochs. ✅

**Measured, and the original justification was wrong.** Caching does *not* make an epoch measurably faster: on the real train split it moved 6,000 item reads from 0.54 s to 0.53 s, which is noise. Pandas lookup was never the bottleneck — the letterbox and one-hot encoding dominate, and those still run per batch by design, because augmentation will make the transform output differ per epoch anyway. The real gain is memory: **2 GB of dataframe replaced by 161 MB of uint8 maps** (~200 MB including validation), which is what matters on a 12 GB Colab VM at 224x224 with workers.

There is a second, smaller win that came out of the same measurement: a cached map is validated once when the cache is built, so the per-access dtype and state checks can be skipped. That took item reads from 0.45 s to 0.36 s per 6,000 (~26%), about 2 s per epoch at 64x64.

**Precomputing more was measured and rejected.** Per item the pipeline costs 76 us at 64x64 and 371 us at 224x224, split between the letterbox and the one-hot encoding. Caching the letterboxed uint8 map would remove about half of that, but it needs 496 MB at 64x64 and **6.1 GB** at 224x224 — so it only helps in the setting that is already fast and is impossible in the one that is slow. With `num_workers: 2` the remaining cost is overlapped with GPU compute anyway, and it would constrain augmentation to transforms that commute with the letterbox.

### 8. Report results with uncertainty, and touch the test split once
The evaluation pipeline can produce bootstrap confidence intervals; use them in the final comparison. Rare classes have very little test support (15 images for the rarest), so point estimates alone are misleading.

*Why:* two models within roughly 0.04 test macro-F1 of each other are not distinguishable, and claiming a winner there is not defensible.
*Done when:* the final table reports intervals, and the test split has been evaluated exactly once, at the end.

### 9. Qualitative error analysis
Look at what the models get wrong, not only how much. Inspect the most-confused pairs in the confusion matrix, plot a grid of misclassified wafers per class, project the learned embeddings with t-SNE or UMAP, and run Grad-CAM to check the network attends to the defect rather than the wafer outline. Grad-CAM specifically is scheduled as a final step — see below — because it has to run on the models actually being presented.

*Why:* the project brief explicitly asks for **qualitative** error analysis alongside the quantitative kind, and it is what turns a results table into an explanation. Cheap — it runs on predictions we already save.
*Done when:* the report shows concrete failure cases with a stated hypothesis for each, not just metrics.

---

## P2 — Improves the numbers

### 10. Data augmentation
Use exact symmetries — the 4 rotations by 90° and their mirrors, or a subgroup of them. These are index permutations, so they introduce no interpolation and no invalid pixel values, and they preserve every class label.

**Be careful with translation**: shifting a localized defect toward the wafer edge can genuinely turn a `Loc` into an `Edge-Loc` while keeping the old label. That one is not available from config at all.

**Free-angle rotation was measured, and the original warning here was wrong.** This item used to say it "destroys thin scratch patterns". That is true if you rotate a *native-resolution* wafer before downscaling; it is not what happens, because augmentation runs on the already-preprocessed 64x64 tensor. Measured there, on the real training split with nearest-neighbour resampling:

| class | defective dies | blobs before | blobs after 30 deg | ratio |
|---|---|---|---|---|
| Center | 714 | 41 | 51 | 1.24x |
| Edge-Ring | 426 | 59 | 61 | 1.03x |
| Loc | 407 | 45 | 50 | 1.11x |
| **Scratch** | **284** | **49** | **54** | **1.10x** |
| Random | 1428 | 17 | 30 | 1.76x |

Defective-die count is preserved (median -0.6% at 30 degrees) and **no wafer in any class lost its pattern at any angle**. `Scratch` fragments no more than `Loc` or `Donut`, and far less than `Random`. So `rotation` is offered as an option — but it is the only one that resamples, so it is not the default, and the three exact-permutation subgroups remain the safe baseline.

*Why:* the minority classes are oversampled with replacement today, so the model sees the same handful of images repeatedly. Augmentation turns each repeat into a different view — the two techniques compound. That compounding is automatic: the weighted sampler draws a `Near-full` wafer ~40 times an epoch and a `none` wafer once, so the rare one already yields 40 distinct views for free.

**Per-class augmentation is available on top of that**, for spending the budget only where it helps:

```yaml
data:
  augmentation:
    name: dihedral8
    probability: 0.0                                    # default for unlisted classes
    class_probabilities: {Scratch: 1.0, Near-full: 1.0}  # and these instead
```

**Only the exact-permutation options may vary by class.** `rotation` refuses a per-class policy and raises if given one: resampling leaves faint artifacts, so augmenting rare classes and not common ones makes "looks resampled" a perfect predictor of "is a rare class", and the network learns the augmentation rather than the defect. A dihedral transform leaves no such trace — a rotated wafer is pixel-identical to one that was natively in that orientation — so varying it by class is safe.
*Done when:* augmentation applies to training only, validation and test are provably untouched, and a fixed seed reproduces the same views. ✅

**Off by default.** A config has to ask for it:

```yaml
data:
  augmentation:
    name: dihedral8      # null (default) | dihedral8 | rotations | flips
    probability: 1.0
```

Three named options, all closed subgroups of D4 — so an ablation between them compares *structure*, not three different amounts of noise:

| name | transforms | group | resamples? |
|---|---|---|---|
| `dihedral8` | all 8 | D4 | no |
| `rotations` | 4 rotations by 90° | C4 | no |
| `flips` | identity, horizontal, vertical, both | Klein four-group | no |
| `rotation` | free angle, `kwargs: {degrees: 180.0}` | — | **yes** |

Horizontal and vertical flips are not a separate mechanism — they are elements of D4, so `dihedral8` already includes them. `flips` exists to isolate them.

**Considered and rejected**, so nobody has to re-derive it: `RandomZoomOut` shrinks the wafer inside a border, an appearance that never occurs at validation or test — a train/serve mismatch with no upside. `RandomAffine` without `translate` reduces to rotation plus scale plus shear; scaling up crops the wafer edge and destroys what `Edge-Ring` and `Edge-Loc` mean, scaling down is the zoom-out problem again, and shear turns a circular wafer into an ellipse, which no physical process produces. `RandomRotation` gives the one useful component with nothing attached, which is why `rotation` is the only resampling option offered.

`WM811KDataset` **raises** if augmentation is passed with a validation or test split, so "provably untouched" is enforced rather than remembered. Views are drawn from torch's seeded RNG, which `seed_worker` re-seeds per dataloader worker. Free-angle rotation and translation are not reachable from config at all — the registry has one entry.

### 11. Attention block on one backbone
Defects occupy a small fraction of each image, which is the standard case for spatial attention (e.g. CBAM).

*Why:* cheap, plausible gain, and a good comparison to show in the presentation.
*Done when:* one backbone is reported with and without it, everything else held fixed. ⏳ *(implemented; the comparison run is still to do)*

CBAM is available on every model as a kwarg, so any config can turn it on:

```yaml
model:
  name: baseline_cnn     # or resnet18, mobilenet_v3_small, ...
  kwargs: {attention: cbam}
```

It is placed **before** global pooling — after pooling there is no spatial extent left to weigh — which for a torchvision backbone means the encoder is built without its own pooling layer when attention is requested. Cost on the baseline CNN is 610 parameters (157,547 vs 156,937).

`configs/train/baseline_cnn_64_augmented.yaml` turns on augmentation *and* attention together. That is the combined arm; run them separately to attribute a gain to either.

### 12. Cap majority-class exposure per epoch
Optional sixth imbalance strategy: limit the majority class to ~12k images **per epoch, resampled each epoch**, rather than deleting rows permanently. Compare it on validation against the current policy.

*Why:* keeps all the hard negatives available across training while reducing per-epoch dominance.
*Done when:* it has been screened on validation like the other strategies, and the better one is used.

### 13. Free post-hoc wins — do these near the end
Two techniques that need no retraining:

- **Test-time augmentation:** average predictions over the same 8 symmetries used for training augmentation. Costs 8× inference (seconds), typically worth 1–2 points of macro-F1, and never touches labels.
- **Per-class decision thresholds** tuned on validation to maximise macro-F1, instead of always taking the arg-max.

*Why:* the cheapest remaining accuracy in the project.
*Done when:* each is measured on validation, and only the ones that actually help are used for the single test evaluation. ✅ *(implemented and measured; apply to the final models)*

```bash
# measure on validation, fit the weights
scripts/inference.py --checkpoint <ckpt> --split validation --tta --tune-thresholds
# reuse those exact weights for the one test evaluation
scripts/inference.py --checkpoint <ckpt> --split test --final-test-evaluation     --tta --class-weights output/runs/<name>-validation/class_weights.json
```

Thresholds are fitted by coordinate ascent on macro-F1 over one multiplier per class, and `--tune-thresholds` **refuses any split but validation** — fitting them on test would be selecting on the frozen split. The fitted weights are saved with the class encoding attached and rejected on load if it differs.

**Early measurement**, on a deliberately undertrained 6-epoch baseline, so treat the magnitudes as indicative: plain arg-max 0.6130, +TTA 0.6190, +thresholds **0.6614**. TTA is worth a fraction of a point; thresholds were worth about four, which fits the reason — arg-max implicitly favours the majority class while macro-F1 weights all nine equally.

*Deliberately excluded:* ensembling the three final models. It would raise the headline number, but it is competition tuning rather than a deep-learning result, and it obscures the architecture comparison that the report is actually about.

### 14. Alternative long-tail methods worth one run each
Three techniques we have not tried, all cheap:

- **Logit adjustment:** shift each class's output score by the log of its training frequency, moving the decision boundary to where it would sit under balanced classes. Applied *post-hoc* it needs no retraining at all. Note it must be compared against the **unweighted baseline**, not stacked on the current sampler — the sampler already flattens the effective prior, so doing both double-corrects.
- **Two-stage (decoupled) training:** train the whole network on the natural, imbalanced distribution, then freeze it and retrain **only the final classifier layer** with balanced sampling. Learning features on the natural distribution and rebalancing only the classifier beats one-stage resampling on long-tailed data, and the second stage takes minutes.
- **Hierarchical classification:** first decide defect vs. no-defect, then send only the defects to an 8-way classifier. This removes the 85% majority class from the fine-grained decision entirely, so the second stage sees a far less skewed problem. Compare against the single 9-class model, and watch for error compounding — anything the first stage misses is unrecoverable.

*Why:* our current policy was chosen from five candidates under a 4-epoch budget. These two work by a different mechanism than anything in that comparison — reweighting changes what mistakes cost, resampling changes what the model sees, these change where the decision boundary sits.
*Done when:* each is screened on validation against the current sampler, on the fixed test-bed model.

### 15. Handcrafted geometric features alongside the CNN
Compute classical wafer-map descriptors and concatenate them with the CNN embedding before the classifier. The reference design is 59 features: **13 density** (defect density over 13 wafer regions), **40 Radon-based** (`skimage.transform.radon` projections, reduced by cubic interpolation), and **6 geometry** features of the largest connected defect region (area, perimeter, major/minor axis length, solidity, eccentricity).

*Why:* this is open issue #7 (feature engineering), and there is a specific reason to expect a gain: the Radon transform is a line detector, and `Scratch` — thin linear defects — is by far our worst class at ~0.32 F1. It also poses a real question for the report: does a CNN rediscover the spatial statistics that were hand-engineered for this task, or does explicit domain knowledge still add something?
*Done when:* three variants are compared on the fixed test-bed model — handcrafted features alone, CNN alone, and both concatenated.

---

## P3 — Stretch, only with spare time

### 16. Pseudo-labeling the unlabeled wafers
Train the best model, predict on the ~617k unlabeled wafers, keep only high-confidence predictions, and retrain with them included. Costs roughly two training runs.

**Guard rails are not optional here.** The majority class is 85.2% of the labeled data, so pseudo-labels will be overwhelmingly that class, and the model is least reliable on exactly the rare classes we want more of — unguarded, this amplifies the imbalance it is meant to fix. Use a high confidence threshold, a **per-class cap** on how many pseudo-labels each class may contribute, and exclude any class whose validation precision is poor.

*Worth doing only if a full training run turns out to be fast.*

### 17. Self-supervised pretraining on the unlabeled data
~617k unlabeled wafers are available and already filtered so none of them come from validation or test groups. Pretrain an autoencoder on them, then fine-tune a classifier head on the labeled set. Reconstruct pixels as a **3-way classification per pixel**, not regression — otherwise the decoder outputs meaningless in-between values.

*Rough cost:* ~30 minutes of GPU pretraining; most of the effort is in the fine-tuning protocol.

*Two cheaper variants of the same idea.* **Multi-task:** rather than pretraining separately, train one network with both a classification loss and a reconstruction loss on a shared encoder — a single run, where the reconstruction term forces the encoder to understand wafer structure. **SimCLR-style contrastive:** train the encoder to agree across two augmented views of the same wafer; our 8 exact symmetries are the natural views and no decoder is needed.

*The experiment that makes this worth doing:* train each variant with 5%, 10%, 25% and 100% of the labels and plot macro-F1 against label count. The question — *how much labeled data does self-supervised pretraining save?* — is a better story than any single number, and a negative result is still a result.

*Alternative worth knowing:* **Mean Teacher with a supervised contrastive loss**, published on this exact dataset with a ~4.5 point F1 gain over a plain ResNet18. Mean Teacher keeps an exponential moving average of the model as a "teacher", shows it and the student two differently augmented views of the same *unlabeled* wafer, and penalises disagreement — so unlabeled data is used without ever needing its label. The contrastive part pulls same-class embeddings together on the labeled data. It is a **single training run at ~1.5–2× cost**, not a separate pretraining stage, and our 8 exact symmetries are ideal as the two views. See `design-notes.md` §12.

### 18. LoRA fine-tuning of a larger pretrained model
Adapt a backbone too large to fine-tune outright (a ViT, say) by training small low-rank adapters while the original weights stay fixed. Memory-cheap, and an unusual technique to show in a course project.

*Why:* it turns "too big to fine-tune, so we froze it" into "too big to fine-tune, so we adapted it properly".

---

## Final steps, in order

Do these last, once the models are trained. The order matters: every selection decision happens on validation, and the test split is opened exactly once, after all of it.

1. **Pick the models to present** on validation macro-F1 with bootstrap intervals (item 8). Two models within ~0.04 are not separable — say so rather than declaring a winner.
2. **Apply the free post-hoc wins** (item 13) — TTA over the 8 symmetries, per-class thresholds — measured on validation, keeping only what actually helps.
3. **Evaluate the test split once**, for the chosen configuration only: `scripts/inference.py --checkpoint ... --split test --final-test-evaluation`.
4. **Run Grad-CAM on the presented models.** Not on every experiment — on the two or three that go in the report and the slides.
5. **Write up** the error analysis and limitations around what steps 3 and 4 show.

### Grad-CAM, concretely

Run it on each presented model, targeting the **last convolutional block** before global pooling (`encoder.layer4` on ResNet18, `features[-1]` on MobileNetV3, the final `Conv2d` on our own CNNs). Use `scripts/inference.py`'s saved `probabilities.csv` to choose examples rather than picking by hand.

Sample per class, roughly six each:

- **correct and confident** — what the model uses when it is right;
- **wrong and confident** — the most informative failures;
- **the confused pairs from the confusion matrix**, which on this dataset means `Loc` vs `Edge-Loc` above all, plus `Scratch`.

What to look for, and what to write down:

- Does the heat map sit on the **defective dies**, or on the **wafer rim**? Attention on the outline means the model is using wafer geometry as a proxy for the class — the same failure mode that makes `Edge-Ring` easy and `Loc` hard.
- For `Loc` vs `Edge-Loc`, is attention on the cluster itself or on its **distance to the edge**? That distinction is the whole difference between the two labels.
- For `Scratch`, is the thin line attended to at all, or has 64x64 downscaling already destroyed it? This connects directly to the downscaling limitation below, and is the one place a picture settles an argument that numbers cannot.

Grad-CAM needs a backward pass, so it runs on the model, not on saved predictions — but it is seconds per image, and only a few dozen images are needed. No new training.

*Done when:* the report shows a Grad-CAM panel for each presented model with a stated reading of what the model attends to, including at least one case where the attention explains a specific error.

## Known limitations to write up

- Thin scratch patterns lose 30–91% of their defective pixels when a large wafer is downscaled to 64×64. This affects ~15% of that class. It is documented and is *not* the main reason that class scores poorly — it is rare and undertrained — but it belongs in the limitations section.
- The imbalance strategy comparison ran on a 4-epoch budget; its ranking below the winner is within seed-to-seed noise.

## If time runs short

Items **1–9** are the minimum for a submission that stands up to questions — item 9 included, since qualitative analysis is explicitly asked for in the brief. The Grad-CAM step above is part of item 9 and is not optional for the same reason; it is also the cheapest slide material in the project. Items 10–15 are where the remaining accuracy is; **13 is the cheapest of all and should not be skipped**. Items 16–18 are stretch, and good presentation material either way.

External benchmarks and why most published WM-811K numbers are not comparable to ours: `design-notes.md` §12. Read it before quoting anyone's accuracy.
