# Design notes

Reasoning and measurements behind the items in [`ROADMAP.md`](ROADMAP.md). Each section records what was decided, what evidence supports it, and what is still unverified. Where a decision is already implemented, the module is named.

Terminology used throughout: a wafer map is a 2-D grid whose cells are one of three categorical states — `0` outside the wafer, `1` functional die, `2` defective die. They are identifiers, not intensities.

---

## 1. Input representation

**Decided.** One-hot encode the three states into 3 channels rather than feeding the raw `{0,1,2}` grid as a single channel. A single channel implies `2` is "twice" `1`, which is meaningless. Implemented in `preprocessing.py`; the default config already produces `(3, 64, 64)` float32.

**Open experiment.** Pretrained backbones were trained on natural RGB images, so one-hot channels are unlike anything in their pretraining distribution. Worth testing both: raw one-hot, and mapping the three states to a grayscale-style image with ImageNet normalization. Cheap, and occasionally makes a large difference. Now a task — see ROADMAP item 5. Note the tension: the grayscale arm knowingly reintroduces the false numeric ordering that the decision above rejects, buying distribution-match with the pretrained filters in exchange. That trade only makes sense for a pretrained encoder, so the comparison is scoped to that model and its outcome does not transfer to the from-scratch CNNs. Implementing it needs a new `normalization` strategy in `preprocessing.py` plus channel replication; the current validator rejects one-hot combined with any normalization.

## 2. Fixed input size

**Decided.** Aspect-ratio-preserving letterbox into 64×64: scale the map until its longer side reaches 64, keep proportions, pad the remainder with state 0, resample with `nearest-exact` so cells stay in `{0,1,2}`. Selected in the preprocessing comparison over pure padding, direct resize, and a single-channel variant.

Direct resize distorts shape ~17.6× more than letterbox by the measured aspect-error metric. Since the dataset contains 632 distinct wafer shapes, direct resizing would stretch every wafer by a different factor, so the same physical defect would appear at a different angle depending on wafer size.

**Caveat recorded in the preprocessing report:** the comparison measured geometric fidelity and state ratios, not downstream accuracy. 64×64 is a well-supported baseline, not an empirically optimal resolution.

**For pretrained models, use 224×224 instead.** Pretrained filters expect that spatial scale, and a ViT's position embeddings are fixed to it — at 64×64 with 16-pixel patches you get a 4×4 grid of patches, which is not a usable image. Upscaling 64→224 would add nothing, since the information was already discarded at 64; letterbox from the native map directly to 224. This is a config argument, not new code.

A side effect worth knowing: the largest wafer in the training data is 212×187, so at 224 every wafer is upscaled and nothing is ever lost.

## 3. Measured behaviour of downscaling

Measured over all 172,950 labeled wafers (not a sample):

| | |
|---|---|
| Wafers larger than 64 in either dimension | **2.0%** |
| Median longest side | **34** |
| No wafer lost *all* its defective pixels | confirmed at full scale |

So the 64×64 canvas *upscales* almost everything. The preprocessing benchmark's claim that no map loses all its defects, originally checked on 900 samples, holds across the whole dataset.

The exception is thin patterns on large wafers. For the `Scratch` class, 15.5% of wafers are downscaled, and those lose defective pixels heavily:

| Wafer longest side | count | median loss of defective pixels |
|---|---:|---:|
| ≤ 64 (upscaled) | 1008 | none — pixels multiply |
| 65–100 | 126 | 30% |
| 101–150 | 56 | 79% |
| > 150 | 3 | 91% |

A one-pixel-wide diagonal line sampled at stride 3 becomes a dotted trail, which resembles scattered noise rather than a line. `Edge-Ring` is downscaled at a comparable rate (14.6%) and is unaffected, because thick ring structures survive nearest-neighbour sampling.

**Conclusion:** real, worth documenting, but *not* the explanation for that class scoring ~0.32 F1 while others reach 0.80–0.96. It affects 15% of an already rare class, and the class is also undertrained. Treat it as a limitation, not a root cause.

## 4. Class imbalance

**Decided.** Deterministic inverse-square-root weighted sampling on the training split only: each sample's draw probability is proportional to `1/√(class count)`, drawn with replacement, keeping the epoch length unchanged. Implemented in `imbalance.py`. Validation and test distributions are never modified.

Selected from five candidates — plain cross-entropy, two loss-reweighting schemes, weighted sampling, and focal loss — screened on one seed, with the top two plus the baseline confirmed on three seeds. Mean validation macro-F1 went from 0.7286 (baseline) to 0.7955, balanced accuracy 0.7087 to 0.8106, while overall accuracy fell slightly (0.9590 to 0.9577). The selected sampler also had the lowest variation across seeds, which is a stronger argument for it than the mean alone.

**Two caveats.**

- Every run used a **4-epoch budget**, and in 7 of 11 runs the best epoch was the last one — training was cut off while still improving. The conclusions describe which strategy *learns minority classes fastest early*, not which converges best. The often-quoted "baseline never predicts `Scratch`" result is the signature of an undertrained model on a rare class. The direction is trustworthy; the effect size is inflated.
- One candidate (`effective_number_ce`) was eliminated for being 0.0049 behind, when seed-to-seed standard deviation is 0.0127 — the gap that decided the cut is smaller than the noise. This does not affect the selected policy, which won on all three seeds, but the ranking below the winner should be read as indicative only.

**Rejected: statically deleting majority-class rows.** Capping the majority class at 10–15k images is a common convention, but deleting ~90k images permanently discards the hard negatives — the ambiguous patterns the model most needs in order to learn to reject them. If tested, cap **per epoch with a fresh random subset each epoch**, so exposure is limited while access to the data is not. Expressed as sampling weights, this is another weighting scheme rather than new sampler machinery.

## 5. Evaluation

**Decided.** Macro-F1 is the primary metric: all nine classes count equally despite the majority class holding 85.2% of the labeled data. Validation drives every selection decision; the test split stays frozen until the final comparison. Implemented in `evaluation.py`.

**Added: bootstrap confidence intervals** (`bootstrap_evaluation`). Macro-F1 has no closed-form sampling distribution, so uncertainty is estimated by resampling the evaluated rows with replacement and taking percentile intervals.

This matters because rare classes have very little support in the fixed folds — validation holds 30 images of the rarest class, test only 15. Measured on the 34,591 validation predictions of the selected run:

| Metric | Point | 95% interval |
|---|---:|---|
| macro-F1 | 0.7956 | 0.7800 – 0.8091 |
| balanced accuracy | 0.8200 | 0.8021 – 0.8343 |
| accuracy | 0.9562 | 0.9541 – 0.9584 |

Per-class F1 for the rarest class spans **0.8302 – 0.9818** — the "15 images" problem made concrete. The test split is half the size, so its intervals are wider still: **two models whose test macro-F1 differ by less than roughly 0.04 should be reported as not separable**, rather than ranked.

Note this is a precision problem, not a metric problem. Macro-F1 is the right choice; no alternative metric fixes a small sample, because any metric that treats rare classes fairly inherits their variance.

## 6. Augmentation

**Safe: the 8 exact symmetries** — rotations by 90/180/270° and their mirrors. These are index permutations: no interpolation, no invalid pixel values, and no label change. None of the nine classes is defined by absolute orientation or handedness, so a rotated pattern keeps its label.

**Unsafe: free-angle rotation.** It requires nearest resampling of a categorical grid, producing a jagged lattice and breaking thin lines — the same failure mode measured in section 3.

**Unsafe: translation, for two specific classes.** Rotation moves the whole wafer, so a defect at the edge stays at the edge. Translation moves the defect *relative to* the wafer boundary, and distance from the boundary is exactly what separates a localized defect from an edge-localized one. Shifting can therefore change the true class while the old label stays attached. Restrict shifts to a couple of pixels, or exclude those two classes.

**Also rejected:** elastic/affine warps and mixup-style blending (both create fractional states between "no die" and "defective die"); brightness, contrast and blur (meaningless — these are not intensities); random erasing (changes the defect ratio, which is itself a class signal).

**Why it matters more than usual here.** The chosen imbalance policy oversamples with replacement, so a rare wafer is currently drawn several times per epoch as *byte-identical* copies. With augmentation each draw is a different view, so oversampling and augmentation compound rather than merely coexist.

**Implementation constraints.** Apply to the raw map *before* the letterbox step, so the symmetries stay exact and the existing geometry code is unchanged. Gate on the training split — the preprocessing report deliberately scopes itself to deterministic base preprocessing and leaves augmentation to training. With more than zero dataloader workers, each worker forks with identical RNG state and will produce identical augmentations unless a per-worker seed is set; draw from a generator owned by the dataset to keep runs reproducible.

## 7. Semi-supervised pretraining

~617k unlabeled wafers are available, already partitioned so that none of them belong to validation or test groups — the leakage question is settled.

Plan: pretrain an autoencoder on the unlabeled set, discard the decoder, attach a classifier head, fine-tune on the labeled set. **Reconstruct as a 3-way classification per pixel, not regression** — an MSE objective on one-hot targets will output values like 0.4, which correspond to no physical state. A denoising variant (randomly flipping a small fraction of cells before encoding) usually yields better features at no extra cost.

Rough cost at 64×64 with mixed precision: about a minute per epoch on a T4, so ~30 minutes for pretraining. The effort is in the fine-tuning protocol, not the compute.

## 8. Generative augmentation — considered and dropped

Synthesizing minority-class samples with an autoencoder or GAN appears in recent work on this dataset. **The team decided not to pursue it.** A generator trained on ~104 examples of the rarest class produces interpolations of those 104 rather than new information, the classifier can end up learning the decoder's manifold, and it needs a firm guarantee that no synthetic image reaches validation — roughly a day of work including tuning. The 8 exact symmetries deliver a large fraction of the benefit for an afternoon and no modeling risk. Recorded here so the option is not re-litigated.

## 9. How experiments are sequenced

Anything that is not the model itself — preprocessing, augmentation, imbalance handling, optimizer settings — is compared **on a single fixed model**, and only the winning setup is carried to the real models. This is the same instrument-based approach the imbalance comparison already used, where one small CNN served as a controlled test bed for five strategies.

The reason is cost and confounding: comparing an augmentation policy across three architectures at once means nine runs and no clean attribution when they disagree. One model, one variable at a time, then transfer.

What transfers and what does not:

- **Usually transfers:** augmentation policy, imbalance strategy, input encoding. These act on the data, and their effects are largely architecture-independent.
- **Never transfers:** learning rate, epoch budget, input resolution, batch size. These are properties of the specific network and must be re-tuned when the winning setup moves to a real model.

Treat a transferred setup as a strong starting point, not a finished configuration.

## 10. Training and hardware

**Budget.** The shared training config defaults to 4 epochs with patience 2 — deliberately small, chosen so eleven strategy comparisons could run quickly. It is not a training budget. Real runs should raise it substantially and confirm that early stopping actually triggers.

**Estimated cost per epoch on Colab**, from FLOPs and typical throughput (estimates, not measured — one real run replaces them):

| Model | Input | ~min/epoch on a T4 |
|---|---|---:|
| Small CNN | 64×64 | well under 1 |
| MobileNetV3-Large | 224×224 | ~0.5 |
| ResNet18 | 224×224 | ~3 |
| VGG19 | 64×64 | ~3 |
| ViT-Base/16 | 224×224 | ~30 — avoid, or freeze the backbone |

For a large transformer, freezing the backbone and precomputing embeddings once (~15 min for the labeled set) makes head experiments effectively free.

**Seeding and determinism.** The current `set_reproducible_seed` calls `torch.use_deterministic_algorithms(True)`. On CUDA this raises unless `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set before CUDA initializes, and several convolution and pooling backward kernels have no deterministic implementation at all, so `warn_only=True` is required for training to proceed.

GPU random state itself is already correct — `torch.manual_seed` seeds all devices internally, so an explicit `torch.cuda.manual_seed_all` would be redundant. What is genuinely missing is **per-worker seeding**: dataloader workers fork with identical random state and will produce identical augmentations, silently reducing an 8× augmentation policy to 1×. The fix is to reshape the function along the lines of Lightning's `seed_everything` — Python, NumPy and PyTorch seeds, `PYTHONHASHSEED`, a derived seed per worker, and the CUDA settings above — while keeping it dependency-free.

**Mixed precision.** Autocast operates inside the model and does not change what the dataloader returns; keep yielding float32 and let autocast handle the rest. T4 has no bf16, so use fp16 with a gradient scaler; A100 supports bf16 without one.

**Optimizers and schedules.** Fine-tuning a pretrained backbone should train the whole network with **different learning rates per parameter group** — a small rate for pretrained weights, a larger one for the newly initialised head, typically 10–100× apart. Freezing the encoder is a memory fallback, not the intended approach: a frozen backbone cannot adapt features to categorical wafer maps, which are far from its natural-image pretraining distribution. Pair this with a schedule; cosine annealing with a short warmup is a reasonable default. The configuration format must therefore be able to express optimizer choice, schedule, and per-group learning rates — anything less restricts what can be tried.

**LoRA** is worth one experiment if a large backbone is attempted. Training small low-rank adapters while the original weights stay frozen keeps memory low enough for a free GPU, and unlike plain freezing it still adapts the representation.

**Checkpointing.** Colab disconnects without warning and wipes local disk, so checkpoints belong on Drive, written every epoch, and must include optimizer and RNG state as well as weights — otherwise a resumed run is not the same experiment. This only becomes important once budgets rise from 4 epochs to realistic ones, which is exactly what is planned.

**Memory.** Cache raw wafer maps as `uint8` (~350 MB for all labeled data) and transform on the fly. Caching preprocessed float tensors would be ~8.5 GB at 64×64 and over 6 GB at 224×224, exceeding Colab's memory. The `uint8` cache also works for any target size, and once built the ~2 GB source dataframe can be released. Keep worker count at 0–2 for the same reason.

## 11. Experiment configuration

**Decided: YAML files loaded into the existing config objects.** The three config objects already validate their fields and serialize themselves for run metadata, so a thin loader is all that is missing.

**Hydra was considered and rejected** for this project: it would duplicate the validation already present in those objects, it changes the working directory on every run (breaking the relative paths used for data, splits, and outputs), and its notebook workflow differs from its documented CLI workflow. It earns its complexity at hundreds of sweep runs; this project has on the order of ten.

**Colab.** Install the package with `pip install -e . --no-deps` rather than syncing the lock file — Colab's PyTorch is matched to its driver, and a lock-driven install can replace it with a build that does not match. Keep the dataset on Drive rather than re-downloading each session. Record the actual torch version in run metadata so the divergence from the local lock is visible.

**Logging.** The training loop already accepts a per-epoch callback receiving a flat metrics dict, which matches what experiment trackers expect, so no changes to the training code are needed. Use a private project — the repository is private, and a public tracker project would publish metrics, configs and confusion matrices.

`wandb` is an optional dependency, imported lazily, so the package and its tests run without it. Locally that means `uv sync --extra logging`; on Colab it needs a separate `!pip install wandb`, because the `--no-deps` install above skips extras along with everything else. The API key goes in Colab Secrets as `WANDB_API_KEY`, never in a cell.

## 12. External results, and why most of them are not comparable

Published WM-811K numbers vary from ~79% to ~99% accuracy, and almost none of them measure the same thing we do. Before quoting any of them, check three things: whether the **test** distribution is natural or rebalanced, whether all nine classes are used, and whether the metric is accuracy or macro-F1.

The headline figures in the ~98–99% range are typically obtained on a **balanced subset** of the nine classes. On the natural distribution the majority class alone is 85.2% of the data, so accuracy above 95% is nearly free and says almost nothing — our own selected run reaches 0.9562 accuracy with a macro-F1 of 0.7956. Those numbers are not evidence that we are far behind; they answer a different question.

One comparable reference point, Wei et al., *Utilizing the Mean Teacher with Supcontrast Loss for Wafer Pattern Recognition* (arXiv:2411.18533, 2024), uses all nine classes on WM-811K with a ResNet18, trains on 10% labeled data with the rest as unlabeled, and rebalances training with SMOTE plus undersampling:

| Method | Accuracy | F1 |
|---|---:|---:|
| ResNet18 baseline | 79.17% | 78.87% |
| + Mean Teacher | 81.14% | 81.29% |
| + SupCon loss | 84.13% | 82.98% |
| + Mean Teacher & SupCon | 84.63% | 83.40% |

Their per-class F1 for the baseline is worth reading even though the protocol differs: `Loc` 50.26, `Edge-Loc` 59.63, `Scratch` 64.72, against `Near-full` 96.03 and `Edge-Ring` 94.27. **The hard classes are the same ones that are hard for us** — the localized and linear defects — which is a useful independent confirmation that our weak spots are properties of the problem rather than bugs in our pipeline.

Note their `None` F1 is only 74.62, far below ours (0.9815), which is the clearest sign their evaluation distribution is rebalanced rather than natural. Read their numbers as a ranking of methods, not as a target to beat.

## 13. What the public Kaggle notebooks actually contain

Reviewed directly (46 notebooks listed on the dataset's Code tab), the ecosystem clusters into a handful of distinct ideas; most entries are forks of two or three originals.

**The canonical notebook** (*WM-811k Wafermap*, ~715 upvotes) is not a deep-learning solution at all. It engineers **59 handcrafted features** — 13 density features over wafer regions, 40 Radon-transform projection features reduced by cubic interpolation, and 6 geometry features of the largest connected defect region (area, perimeter, major and minor axis length, solidity, eccentricity) — and classifies them with a One-vs-One SVM. That is the source of the feature design in roadmap item 15, and it is verified rather than assumed.

**Important caveat:** its training target counts cover only **8 classes and about 19k samples** — it drops `none` entirely. So the most-copied notebook on this dataset solves an easier problem than ours, and its results are not comparable to nine-class macro-F1 on the natural distribution.

Other distinct approaches present: plain Keras and PyTorch CNNs; a convolutional autoencoder used for minority-class augmentation; CNN-WDI; a ViT variant; MobileNetV2 and V3 lightweight models; ResNet-50 with ASVD; class-activation-map visualisation; attention fusion; image retrieval; wafer segmentation; and several generative attempts (DCGAN, WGAN-GP, conditional diffusion).

Two things this tells us. The generative direction has been tried repeatedly here without producing a dominant result, which supports the decision to drop it. And a class-activation-map notebook exists, which is worth borrowing from for the qualitative error analysis in roadmap item 9.

