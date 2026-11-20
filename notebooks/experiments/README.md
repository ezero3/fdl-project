# Experiment Driver Notebooks

This directory contains the driver Jupyter notebooks used to launch, monitor, and reproduce model training runs and ablation sweeps, primarily executed on **Google Colab** to leverage cloud GPUs.

Each notebook clones the repository, syncs dependencies, loads deterministic dataset split indices, and trains models using configurations defined under [`configs/train/`](../../configs/train/).

---

## Experiment Map & Structure

The notebooks correspond to specific experimental phases and architecture sweeps:

| Notebook | Focus & Architecture | Role / Question Investigated |
|---|---|---|
| `base_notebook.ipynb` | Environment & Phase 0 | GPU profiling, memory measurement, and baseline training harness. |
| `phase2_sweep.ipynb` | Phase 2 Screening | Multi-arm ablation over loss formulations, normalizations, and initial sampling. |
| `25_focal_with_rotation.ipynb` | Augmentation & Losses | Isolating the impact of continuous rotation vs. Focal Loss variations on `BaselineCNN`. |
| `27_architectures.ipynb` | Initial Custom Models | Screening custom architectures (ResNet-style, ConvNeXt, DenseNet, Inception) at 64×64. |
| `28_capacity_and_resolution.ipynb` | Scaling & Resolution | Testing capacity scaling and resolution increase (64×64 to 128×128) across ConvNeXt and ResNet. |
| `28b_vit_style_128.ipynb` | Vision Transformer (Scratch) | Evaluating patch-based ViT built from scratch at 128×128. |
| `29_pretrained_backbones.ipynb` | Transfer Learning (Frozen) | Evaluating frozen ImageNet backbones (ResNet-18, MobileNet) with adapted one-hot inputs. |
| `30_finetune_ladder.ipynb` | Fine-Tuning Strategy | Comparing discriminative learning rates and end-to-end fine-tuning vs. frozen feature extractors. |
| `31_dilation.ipynb` | Dilated Convolutions | Testing receptive field expansion without downsampling (`DilatedCNN`). |
| `31b_convnext_dilated_128.ipynb` | Dilated ConvNeXt | Investigating whether dilation adds benefit to architectures that already downsample. |
| `32_bigger_pretrained.ipynb` | Scaled Pre-trained Models | Fine-tuning larger vision backbones (e.g., ResNet-34). |
| `32b_vit_b_32.ipynb` | Pre-trained ViT | Discriminative fine-tuning of pre-trained ViT-B/32 at 224×224. |
| `33a_efficientnet_b0.ipynb` | EfficientNet | Evaluating fine-tuned EfficientNet-B0. |
| `33b_convnext_tiny.ipynb` | Pre-trained ConvNeXt | Evaluating fine-tuned ConvNeXt-Tiny from torchvision. |
| `33c_swin_t.ipynb` | Hierarchical Transformer | Testing Swin Transformer (Swin-T). |
| `RESUME_RESNET_V28.ipynb` | Training Resumption | Utility notebook demonstrating checkpoint recovery and training continuation. |

---

## Execution Guide (Google Colab)

To run any of these experiments on Google Colab:
1. Open the notebook in Colab and select a GPU runtime (**Runtime $\rightarrow$ Change runtime type $\rightarrow$ T4 / V100 / A100**).
2. Set up GitHub credentials or mount Google Drive if loading weights.
3. Weights & Biases tracking is automatically configured via `WandbLogger` (ensure `WANDB_API_KEY` is provided in Colab Secrets or terminal login).
