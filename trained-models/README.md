# Trained Models and Checkpoints

This directory is designated for locally storing trained model checkpoints (`*.pt`) and model weights.

> Large model weight binaries are intentionally excluded from version control (enforced via `.gitignore`). Save or download checkpoints generated from Google Colab or local training runs into this folder.

To evaluate checkpoints saved in this directory using post-hoc scoring and Test-Time Augmentation (TTA):

```bash
uv run python scripts/posthoc.py \
    --checkpoint trained-models/checkpoints/best.pt \
    --split validation --tta --tune-thresholds
```
