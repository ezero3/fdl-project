"""Grad-CAM over trained checkpoints: what part of the wafer decided the class.

    uv run python scripts/gradcam.py --checkpoint a/best.pt --checkpoint b/best.pt \
        --classes Scratch Loc Edge-Loc Center --output output/gradcam

Produces one figure per class: the wafer map, then one heat map per model, so a
row reads as "here is the defect, and here is where each model looked".

Method: hook the last convolutional layer, take the gradient of the true class
logit with respect to its activations, average those gradients per channel to
get channel weights, and sum the activations under them (Selvaraju et al.,
2017). No extra dependency -- two hooks and one backward pass.

Two honest limitations, both worth a sentence on the slide rather than hiding:

* **Token models are skipped.** A ViT or Swin has no final convolution whose
  spatial layout matches the image, so the same recipe would need attention
  rollout instead. The script says so and moves on rather than producing a
  plausible-looking map that means something else.
* **Resolution is the model's, not the wafer's.** The heat map comes from a
  feature map that may be 4x4 or 8x8 and is upsampled to the input size, so it
  localises coarsely by construction. A blob covering half the wafer is the
  method's resolution, not necessarily the model's uncertainty.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn

from fdl_project.config.loader import build_experiment_config
from fdl_project.config.registry import build_model
from fdl_project.constants import CLASS_NAMES, CLASS_TO_INDEX
from fdl_project.data.datasets import create_split_dataset, load_wm811k_dataframe
from fdl_project.training.checkpoint import load_checkpoint
from fdl_project.training.loop import resolve_device

logger = logging.getLogger("gradcam")


def last_convolution(model: nn.Module) -> nn.Conv2d | None:
    """The deepest Conv2d, which is where Grad-CAM is normally taken.

    Deepest rather than named: every architecture here spells its stages
    differently, and the last convolution is the last layer whose spatial axes
    still correspond to the image.
    """

    found = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            found = module
    return found


class GradCAM:
    """Activations and gradients from one layer, via two hooks."""

    def __init__(self, model: nn.Module, layer: nn.Module) -> None:
        self.model = model
        self.activations: Tensor | None = None
        self.gradients: Tensor | None = None
        self._handles = [
            layer.register_forward_hook(self._save_activations),
            layer.register_full_backward_hook(self._save_gradients),
        ]

    def _save_activations(self, _module, _inputs, output: Tensor) -> None:
        self.activations = output.detach()

    def _save_gradients(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()

    def __call__(self, inputs: Tensor, class_index: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(inputs)
        logits[0, class_index].backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("The hooked layer did not fire; wrong target layer.")
        # One weight per channel: how much raising that channel raises the logit.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(
            cam, size=inputs.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam[0, 0].cpu().numpy()
        # Per-map normalisation: Grad-CAM is only ever read as a relative map.
        span = cam.max() - cam.min()
        return (cam - cam.min()) / span if span > 1e-12 else np.zeros_like(cam)


def wafer_background(inputs: Tensor) -> np.ndarray:
    """The categorical map, for display, whatever encoding the model uses.

    one_hot is three indicator planes, so an argmax recovers the states exactly;
    grayscale_rgb repeated one channel, so any channel does.
    """

    array = inputs[0].detach().cpu().numpy()
    if array.shape[0] == 3 and np.allclose(array.sum(axis=0), 1.0, atol=1e-3):
        return array.argmax(axis=0).astype(float) / 2.0
    channel = array[0]
    span = channel.max() - channel.min()
    return (channel - channel.min()) / span if span > 1e-12 else channel


def load_one(path: Path, device: torch.device) -> tuple[nn.Module, Any, str]:
    payload = load_checkpoint(path)
    stored = payload.get("config")
    if not stored:
        raise ValueError(f"{path} carries no config.")
    config = build_experiment_config(stored)
    model = build_model(config.model.name, **dict(config.model.kwargs))
    model.load_state_dict(payload["model_state"])
    return model.to(device).eval(), config, config.name


def pick_examples(dataset, class_names: list[str], per_class: int, seed: int) -> dict:
    """One or more validation wafers per requested class, chosen reproducibly."""

    targets = np.asarray(dataset.target_indices)
    generator = np.random.default_rng(seed)
    chosen: dict[str, list[int]] = {}
    for name in class_names:
        positions = np.flatnonzero(targets == CLASS_TO_INDEX[name])
        if positions.size == 0:
            logger.warning("No %s wafers in the split; skipping.", name)
            continue
        take = min(per_class, positions.size)
        chosen[name] = generator.choice(positions, size=take, replace=False).tolist()
    return chosen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True,
                        help="A checkpoint. Repeatable; 2-4 reads well on a slide.")
    parser.add_argument("--classes", nargs="+", default=["Scratch", "Loc", "Edge-Loc", "Center"],
                        help=f"Any of: {', '.join(CLASS_NAMES)}")
    parser.add_argument("--per-class", type=int, default=1)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output", type=Path, default=Path("output/gradcam"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=86)
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    for name in args.classes:
        if name not in CLASS_TO_INDEX:
            raise SystemExit(f"Unknown class {name!r}. Available: {', '.join(CLASS_NAMES)}")

    device = resolve_device(args.device or "auto")
    models = []
    for path in args.checkpoint:
        model, config, name = load_one(Path(path), device)
        layer = last_convolution(model)
        if layer is None:
            logger.warning(
                "%s has no Conv2d (a token model); Grad-CAM needs attention "
                "rollout instead. Skipping.", name
            )
            continue
        models.append({"name": name, "model": model, "config": config, "layer": layer})
    if not models:
        raise SystemExit("No convolutional checkpoints to explain.")

    print(f"{len(models)} model(s): " + ", ".join(m["name"] for m in models))

    dataframe = load_wm811k_dataframe(models[0]["config"].data.dataset_path)
    args.output.mkdir(parents=True, exist_ok=True)

    # Each model has its own preprocessing, so each gets its own dataset view of
    # the same wafer rows. The row indices are what tie the columns together.
    datasets = {}
    for entry in models:
        config = entry["config"]
        datasets[entry["name"]] = create_split_dataset(
            dataframe, config.data.split_directory, args.split,
            preprocessing_config=config.data.preprocessing, cache_maps=False,
        )
    reference = datasets[models[0]["name"]]
    chosen = pick_examples(reference, args.classes, args.per_class, args.seed)
    row_index_to_position = {
        int(row): position for position, row in enumerate(reference.row_indices)
    }

    written = []
    for class_name, positions in chosen.items():
        for order, position in enumerate(positions):
            row_index = int(reference.row_indices[position])
            columns = 1 + len(models)
            figure, axes = plt.subplots(1, columns, figsize=(2.6 * columns, 3.0))
            axes = np.atleast_1d(axes)

            base_inputs, target, _ = reference[position]
            axes[0].imshow(wafer_background(base_inputs.unsqueeze(0)), cmap="viridis")
            axes[0].set_title(f"{class_name}\nwafer {row_index}", fontsize=10)
            axes[0].axis("off")

            for column, entry in enumerate(models, start=1):
                dataset = datasets[entry["name"]]
                # Same wafer, this model's own geometry.
                local = row_index_to_position.get(row_index, position)
                inputs, _, _ = dataset[local]
                inputs = inputs.unsqueeze(0).to(device).requires_grad_(False)

                cam_tool = GradCAM(entry["model"], entry["layer"])
                try:
                    with torch.enable_grad():
                        cam = cam_tool(inputs, CLASS_TO_INDEX[class_name])
                    with torch.no_grad():
                        probabilities = torch.softmax(entry["model"](inputs), dim=1)[0]
                finally:
                    cam_tool.close()

                predicted = CLASS_NAMES[int(probabilities.argmax())]
                confidence = float(probabilities.max())
                axes[column].imshow(wafer_background(inputs.cpu()), cmap="gray")
                axes[column].imshow(cam, cmap="jet", alpha=0.5)
                mark = "" if predicted == class_name else "  MISS"
                axes[column].set_title(
                    f"{entry['name']}\n{predicted} {confidence:.2f}{mark}",
                    fontsize=8,
                    color="black" if predicted == class_name else "firebrick",
                )
                axes[column].axis("off")

            figure.tight_layout()
            suffix = "" if args.per_class == 1 else f"_{order}"
            path = args.output / f"gradcam_{class_name.replace('-', '_')}{suffix}.png"
            figure.savefig(path, dpi=args.dpi, bbox_inches="tight")
            plt.close(figure)
            written.append(path)
            print(f"  {path}")

    print(f"\n{len(written)} figure(s) -> {args.output}")


if __name__ == "__main__":
    main()
