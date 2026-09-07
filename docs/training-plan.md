# Training plan

What we actually run, in what order, and how each choice gets decided.

The framework is finished: every branch below is reachable from a YAML file, so nothing here needs new code except the two places that say so explicitly. The point of this document is to stop us from training twelve models in parallel on twelve different setups and then being unable to say why any of them won.

**The rule this plan is built on** (from `ROADMAP.md`, "How we run experiments"): everything that is *not* the model — preprocessing, augmentation, imbalance handling, optimizer settings — gets compared on **one fixed cheap model**. Only a setup that wins there is carried to the real models. Comparing an augmentation choice on ResNet18 and an imbalance choice on our DenseNet tells us nothing about either.

Four phases:

| phase | question | runs | fixed | varied |
|---|---|---|---|---|
| 0 | what does an epoch cost? | 3 short | — | — |
| 1 | which cheap model is the test-bed? | 3 full | pipeline defaults | model |
| 2 | what is the best pipeline? | ~16 full | test-bed model | one factor at a time |
| 3 | which architecture wins on it? | ~10 full | pipeline from phase 2 | model |
| 4 | what do we report? | 0 training | everything | — |

Validation drives every decision in phases 0–3. **The test split is opened exactly once, in phase 4.**

---

## Phase 0 — Measure the epoch cost, then decide the evaluation protocol

**This is the first experiment, and it decides the shape of everything after it.** We have never run this pipeline on a GPU for a full epoch. Every budget below is guesswork until we have that number, and two decisions hang on it: whether we can afford **grouped cross-validation**, and what **batch size** we standardise on.

```bash
uv run python scripts/train.py configs/train/baseline_cnn_64.yaml   --override trainer.max_epochs=2 --overwrite
uv run python scripts/train.py configs/train/dilated_style_64.yaml  --override trainer.max_epochs=2 --overwrite
uv run python scripts/train.py configs/train/densenet_style_64.yaml --override trainer.max_epochs=2 --overwrite
```

Record per model: seconds per training epoch, seconds per validation pass, peak GPU memory, GPU utilisation, and whether `batch_size: 512` fits.

Two things worth checking in the same runs, because they are cheap here and expensive to discover in phase 3:

- `num_workers: 2` may be the bottleneck rather than the GPU. If utilisation is low, raise it — the one-hot encoding runs per batch by design (item 7).
- `dilated_style` at `downsample_every: 0` holds 64 channels at full 64x64 resolution through 8 blocks. That is likely the most expensive of the three by *activation* memory despite being the second-smallest by parameters. Parameter count is not a cost estimate.

### Decision 1 — grouped cross-validation, if training is fast

A single 70/20 split gives one validation estimate whose bootstrap interval is wide, because the rare classes have little support. That is the reason phase 1 expects overlapping intervals and has to fall back on tie-breaks. **Cross-validation is the direct fix**: it replaces one estimate with K, and the spread across folds is a far better uncertainty estimate for *model selection* than a bootstrap over a single fixed validation set — bootstrap resamples the same wafers, CV actually retrains on different ones.

If phase 0 says an epoch is cheap, do it. Design:

- **`StratifiedGroupKFold` over the train+validation pool only** (155,654 rows). The test split's 17,296 rows stay frozen and are not part of any fold. Grouping by `lotName` is non-negotiable — wafers from one lot are correlated, and a lot spanning two folds leaks.
- Stratified as well as grouped, because plain `GroupKFold` can hand one fold almost no `Near-full` (0.1% of the data) and make that fold's macro-F1 meaningless.
- **Generate the folds once**, with a fixed seed, persist them as row indices under `data/splits/cv/`, and commit them. Same rule as the main split: everyone loads the same folds, nobody regenerates. Folds that differ per person or per model destroy comparability exactly the way a re-shuffled split would.
- Report **mean +- across-fold standard deviation**, and use that spread, not a bootstrap interval, to decide whether two setups differ.

Cost is K x everything. K=5 turns phase 2's ~16 arms into 80 runs, which is almost certainly not affordable. The sensible allocation, decided by the phase 0 numbers:

| phase | protocol | why |
|---|---|---|
| 1 (pick test-bed) | CV if affordable | this is where overlapping intervals actually block a decision |
| 2 (~16 sweep arms) | single split | K x 16 is the expensive one; relative comparisons on one fixed split are what the sweep needs |
| 3 (final architectures) | CV if affordable | this comparison goes in the report and needs real error bars |

K=3 is a reasonable compromise if K=5 does not fit. One extra thing CV changes: each fold produces its own best checkpoint, so "the model we present" is either one nominated fold or a retrain on the full pool — decide which before phase 4, and say which in the report.

This is **not yet implemented.** It needs a fold-generation notebook or script plus a `--fold` argument threaded through the runner. Only build it if phase 0 says it is affordable.

### Decision 2 — batch size

See "Choosing batch size" below. Phase 0 is where the throughput curve gets measured.

### Decision 3 — the epoch budget for phase 2

If a full 40-epoch run is affordable, sweep at 40. If not, sweep at a fixed shorter budget (15-20 epochs) and say so in the report — the class-imbalance study already did exactly this at 4 epochs, and a screening budget is legitimate as long as it is stated and identical across arms.

---

## Choosing batch size

`defaults.yaml` currently says 512. That was a guess, and phase 0 is where it gets replaced with a measurement. Four constraints, in the order they bite:

**1. Memory — the hard ceiling.** Find the largest that fits, by doubling until it OOMs. This is the only constraint that is a hard error rather than a trade-off, and it is what will force the 224x224 pretrained models down in phase 3. `resnet18` at 224x224 will not take 512 on a T4.

**2. Throughput — where the ceiling stops mattering.** Samples/sec rises with batch size and then flattens once the GPU is saturated. Past that point a bigger batch buys nothing and costs optimizer steps. Measure at 64/128/256/512/1024 and **pick the smallest batch that still sits on the flat part of the curve**, not the largest that fits.

**3. Optimizer steps — the argument against 512 here.** With 121,063 training rows, batch 512 gives 237 steps per epoch; 40 epochs is ~9,500 updates, and early stopping at ~20 epochs makes it ~4,700. That is on the low side for training a network from scratch. Batch 128 gives 946 steps/epoch for the same wall-clock cost if the GPU is not saturated at 128 — which is exactly what the throughput curve tells us. **This is the reason to suspect 512 is too big for the 64x64 models**, and it has nothing to do with memory.

**4. Class imbalance — the argument for a large batch here.** This dataset makes small batches riskier than usual. `Near-full` is ~0.1% of the training rows, so at batch 64 under the *natural* distribution a batch contains one roughly every 16 batches, and the gradient for that class is extremely noisy. `inverse_sqrt_sampler` flattens the effective distribution and largely removes the problem — but the phase 2b arms that drop the sampler (`unweighted_ce`, `inverse_sqrt_ce`, `focal_loss`) reintroduce it. All our models also lean heavily on BatchNorm, whose running statistics degrade below ~32 per batch.

Constraints 3 and 4 pull in opposite directions, which is why this is measured rather than reasoned to a number.

**"Just fill the VRAM" — right for phase 3, wrong for phase 1-2.** The heuristic is really "saturate the GPU", and filling memory is a proxy for that which only holds when the model is big enough for the two to coincide. They do for `resnet18`/`vit_b_16` at 224x224: compute scales with the activations, so the largest batch that fits is close to the fastest. **Fill it there.**

They do not coincide for a 157k-parameter CNN at 64x64. A batch of 512 one-hot 64x64 tensors is 25 MB of input and the activations are small, so a 16 GB card is nowhere near full at the point where the GPU is already compute-saturated — and quite possibly the bottleneck is not the GPU at all but the dataloader, since the letterbox and one-hot encoding run per batch on CPU with `num_workers: 2`. If phase 0 shows low GPU utilisation at batch 512, a bigger batch buys **nothing**: it will not go faster, and it will halve the optimizer steps per epoch. That is the case where filling VRAM actively costs accuracy.

So: measure utilisation, not just memory. Fill the card when the card is the bottleneck; raise `num_workers` when it is not.

**Batch size and learning rate are not independent.** Changing one without the other conflates two variables. Rescale when you change it — linear (`lr * k` for `batch * k`) is the standard rule for SGD, and square-root scaling is the better-motivated choice for Adam-family optimizers, which is what we use. Either way, a batch-size comparison at fixed LR is not a batch-size comparison.

**Then hold it fixed.** Batch size is part of the pipeline, not a per-arm choice: phase 2's comparisons are only controlled if every arm uses the same one. Phase 3 is the exception — batch size, learning rate, input size and epoch budget are the four things that do **not** transfer across architectures and must be re-tuned per model.

---

## Phase 1 — Pick the test-bed model (3 runs)

Three candidates, all cheap, all from scratch, run at the **stock pipeline** (`defaults.yaml`: 64×64 letterbox one-hot, `inverse_sqrt_sampler`, AdamW 1e-3, no augmentation, no attention, no schedule, 40 epochs, early stopping patience 5).

| config | parameters | the claim being tested |
|---|---|---|
| `baseline_cnn_64` | 157k | the floor — plain conv stack |
| `dilated_style_64` | 298k | context without downsampling; the best-motivated design for `Scratch` |
| `densenet_style_64` | 304k | feature reuse; most capacity per parameter |

```bash
uv run python scripts/train.py configs/train/baseline_cnn_64.yaml
uv run python scripts/train.py configs/train/dilated_style_64.yaml
uv run python scripts/train.py configs/train/densenet_style_64.yaml
```

`inception_style` (799k) and the two 2.7M-parameter models are deliberately not here. They are phase-3 entries; a test-bed that costs 10× more makes the sweep 10× more expensive for no gain in what it tells us.

**How the winner is chosen.** Validation macro-F1 with its bootstrap interval, from `output/runs/<name>/metrics.json`. The intervals will very likely overlap — the rare classes have little validation support, and three architectures at this scale on the same data are not going to separate cleanly. **Overlapping intervals mean we do not have a winner on accuracy**, and saying otherwise is exactly the mistake item 8 exists to prevent. In that case pick on the tie-breaks, in order:

1. **Cost** — the test-bed is run ~16 more times in phase 2; the cheapest of a statistical tie is worth real hours.
2. **`Scratch` and `Donut` F1** — the two hardest classes are where a pipeline change has room to show an effect. A model already at 0.05 on `Scratch` is a poor instrument for detecting a two-point improvement.
3. **Stability across the run** — a jagged validation curve means phase 2's differences get buried in seed noise.

Write the choice and the reason into this file when it is made. If the winner is not also the best on macro-F1, that is fine and should be stated: this model is an *instrument*, not a result.

**Sanity gate before continuing.** If all three sit near ~0.30 macro-F1 or the majority class is swallowing everything, something is wrong with the pipeline rather than with the architectures, and phase 2 would be sweeping noise. Check the per-class table and the confusion matrix first.

---

## Phase 2 — Sweep the pipeline on that one model (~16 runs)

**One factor at a time, each stage starting from the winner of the last.** This is a greedy walk, not a grid: a full grid over these branches is thousands of runs. Greedy can miss an interaction (e.g. augmentation might only pay off once the sampler is off), and the two places that plausibly interact are called out below.

### 2a. Augmentation (5 runs)

| arm | config | group |
|---|---|---|
| off | `augmentation.name: null` | — (phase-1 result, already run) |
| flips | `flips` | Klein four-group |
| rotations | `rotations` | C4 |
| all 8 | `dihedral8` | D4 |
| free angle | `rotation`, `kwargs: {degrees: 180.0}` | resamples |

```bash
uv run python scripts/train.py configs/train/<testbed>.yaml \
    --override data.augmentation.name=dihedral8 --overwrite
```

The three exact subsets are closed groups, so this compares *structure*, not three amounts of noise. `rotation` is the only arm that resamples; it was measured safe (defect count preserved, no wafer loses its pattern, `Scratch` fragments 1.10× — less than `Loc`), but it is the one to be suspicious of if it wins by a hair.

Keep `probability: 1.0` throughout. It is not a strength dial — the identity is already one of the 8 group elements, so at 1.0 a sample is unchanged 12.5% of the time and at 0.5 it is unchanged 56%. Lowering it biases training toward the orientation the fab happened to record, which has no special status.

**Not swept here:** `class_probabilities`. It is available and safe for the exact permutations, but the weighted sampler already gives a `Near-full` wafer ~40 distinct views per epoch for free, so per-class augmentation mostly duplicates work the sampler is doing. Revisit only if 2b lands on a non-sampler imbalance policy — that is the interaction where it starts to matter.

### 2b. Imbalance policy (4 runs)

`inverse_sqrt_sampler` is the current default, chosen from five candidates under a **4-epoch screening budget**. That decision deserves one confirmation at full budget, because a policy that wins in 4 epochs is not automatically the one that wins in 40 — reweighting mostly changes early-training dynamics.

Arms: `unweighted_ce`, `inverse_sqrt_ce`, `effective_number_ce`, `focal_loss`, against the incumbent `inverse_sqrt_sampler`.

```bash
uv run python scripts/train.py configs/train/<testbed>.yaml \
    --override imbalance.preset=focal_loss --overwrite
```

**Interaction to watch:** the sampler and augmentation compound (a rare wafer drawn 40× yields 40 different views). If 2a's winner was augmentation-on, re-check the best loss-based arm here with augmentation both on and off before adopting — that is the one pair in this sweep where greedy order can genuinely mislead.

**Item 12 (cap the majority class per epoch) is not implemented.** It belongs in this comparison and needs a small amount of code: resample `none` down to ~12k rows each epoch rather than deleting them. Worth adding if phase 2 shows the imbalance policy matters much; skip it if all five arms land within noise.

### 2c. Attention (1 run)

`model.kwargs: {attention: cbam}` on the test-bed, everything else at the 2b winner. 610 extra parameters on the baseline CNN, so the comparison is close to free and is a clean with/without statement for the presentation — which is exactly what item 11 asks for.

Only adopt it into the pipeline if it wins here. Note it does not transfer automatically to phase 3: `vit_style` and the pretrained transformers refuse it by design.

### 2d. Schedule and optimizer (3 runs)

Currently `scheduler.name: null` — a constant learning rate for 40 epochs, which is the one obviously improvable setting in the defaults.

- `cosine` — the plain default
- `cosine_with_warmup`, `kwargs: {warmup_epochs: 2, min_lr_ratio: 0.01}`
- `onecycle`, `interval: step`

If none beats a constant LR, that is a finding about the budget (early stopping is probably firing before the schedule matters) and the constant LR stays.

**Not swept:** the optimizer itself. Six are available, but AdamW at 1e-3 is a defensible default and optimizer-vs-optimizer is not a question this report is asking. One `sgd` + momentum run is worth it only if there is spare GPU time.

**Not swept:** `target_size`. `letterbox` + `one_hot` at 64×64 was already settled by the preprocessing study (`docs/reports/preprocessing_report.md`) — pure padding costs 9.7× the memory for a wafer occupying 3.5% of its canvas, and direct resize distorts 17.6× more than letterbox. Re-deciding it here would be re-running a finished experiment. The one open question is whether 64×64 is *enough resolution* for `Scratch`; if `Scratch` F1 stays poor through all of phase 2, one run at 96×96 is the right diagnostic, and it is a `data.preprocessing.target_size` override, not new code.

### 2e. Post-hoc, on the phase-2 winner (0 training runs)

Free — no retraining, fitted on validation only:

```bash
uv run python scripts/inference.py --checkpoint output/runs/<winner>/best_model.pt \
    --split validation --tta --tune-thresholds
```

Early indication on a deliberately undertrained 6-epoch baseline: arg-max 0.6130 → +TTA 0.6190 → +thresholds 0.6614. Expect the threshold gain to *shrink* on a properly trained model — much of those four points is arg-max favouring the majority class, which a converged model with a balanced sampler already partly corrects. Measure it; do not assume the number carries.

TTA and thresholds are applied at the end to every phase-3 model, not folded into the pipeline, since they change nothing about training.

### End of phase 2

Write the settled pipeline into a real config file (`configs/train/pipeline.yaml` or similar) so phase 3 inherits it by reference rather than by 10 copied override flags. Record in the report which arms won, by how much, and which were within the interval — "we tried five imbalance policies and they were indistinguishable at 40 epochs" is a result worth reporting, not a failure.

---

## Phase 3 — Every architecture on the settled pipeline (~10 runs)

Now the pipeline is fixed and the model varies. This is the comparison the report is actually about.

**Ours, from scratch** (all at 64×64, pipeline from phase 2):

`baseline_cnn` · `densenet_style` · `dilated_style` · `inception_style` · `resnet_style` · `vit_style`

`vit_style` is a deliberate negative control and is expected to lose — no convolutional prior, 121k training wafers, 85% one class. Run it anyway; a measured negative result beats a dismissed architecture. Note the 2-epoch/1200-sample smoke runs put it mid-pack rather than last, which is meaningless at that scale but makes it the interesting one to watch.

**Pretrained** (224×224, `param_groups` giving the encoder ~100× the head's smaller LR):

`resnet18` · `mobilenet_v3_small` · `efficientnet_b0` · `vit_b_16`

`vit_b_16` is the direct counterpart to our `vit_style`: same family, ImageNet pretraining instead of learning the prior from 121k wafers. That pairing is the single most informative comparison in the whole set, so it is not optional.

**What must be re-tuned per model, because it does not transfer:** learning rate, batch size, epoch budget, and input size. Augmentation and imbalance results generally do carry across architectures, which is the entire reason phase 2 was run on one model.

Two extra runs that are comparison points rather than contenders:

- `resnet18_224_frozen` — frozen encoder, trains in minutes, says how much fine-tuning actually buys.
- **The input-representation arm** (roadmap item 5): one-hot vs grayscale-replicated-to-3-channels with ImageNet mean/std, **on the pretrained backbone only**. ImageNet filters were trained on photographs and one-hot indicator channels look nothing like that distribution. This is the one thing in phase 3 that needs code: `PreprocessingConfig` currently refuses to combine one-hot with any normalization and has no ImageNet strategy. Run it only on the pretrained model — the result is about matching a pretraining distribution and will not transfer to our models.

**Budget note.** Ten runs at 224×224 is where the GPU time actually goes; the 64×64 models are cheap by comparison. If phase 0 says this does not fit, cut the pretrained set to `resnet18` + `vit_b_16` + one mobile model and say why. Cutting a from-scratch model is worse — the brief specifically wants our own component.

---

## Phase 4 — Report (no training)

In this order, and not before:

1. **Pick the final models on validation.** Two or three, the ones we present.
2. **Fit TTA and per-class thresholds on validation** for exactly those models; save the weights.
3. **Evaluate on test, once.** `--final-test-evaluation`, reusing the saved validation-fitted class weights. Everything reported with bootstrap intervals; two models within ~0.04 test macro-F1 are **not** distinguishable and the report says so.
4. **Qualitative error analysis** (item 9) — confusion pairs, a grid of misclassified wafers per class, embedding projection.
5. **Grad-CAM last**, on the models being presented, to check the network attends to the defect and not the wafer outline.

Nothing after step 3 may change a model, a threshold, or a hyperparameter. If step 4 reveals something we want to fix, that is future work in the report, not another test evaluation.

---

## What we are deliberately not doing

Recorded so nobody re-derives them mid-project:

- **Ensembling the final models.** It would raise the headline number and obscure the architecture comparison the report is about.
- **Translation augmentation.** Shifting a `Loc` toward the edge genuinely makes it an `Edge-Loc` while keeping the old label. Not reachable from config at all.
- **`RandomZoomOut` / `RandomAffine`.** Zoom-out creates an appearance that never occurs at validation or test; shear turns a circular wafer into an ellipse, which no physical process produces.
- **Caching preprocessed float tensors.** Measured: 496 MB at 64×64 and 6.1 GB at 224×224, and it only helps where the pipeline is already fast.
- **Re-deciding geometry or encoding.** Settled by a completed study; see phase 2d.
- **Handcrafted geometric features** (item 15) and everything in P3 — pseudo-labeling, self-supervised pretraining. Stretch only, and only after phase 4 is written.

Items 14 (logit adjustment, decoupled two-stage, hierarchical) are genuinely interesting and work by a different mechanism than anything in phase 2b. They belong in the sweep *if* there is time after phase 3 — screened on validation on the test-bed model, like everything else.
