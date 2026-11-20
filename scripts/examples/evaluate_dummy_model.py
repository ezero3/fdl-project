"""Run the evaluation pipeline end to end with a deterministic dummy model."""

from argparse import ArgumentParser
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fdl_project import (
    CLASS_NAMES,
    NUM_CLASSES,
    evaluate_model,
    save_evaluation_results,
)


class PassThroughClassifier(nn.Module):
    """Treat the input tensor as already-computed class logits."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("output/evaluation"))
    parser.add_argument("--run-name", default="dummy-validation")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(86)
    targets = torch.tensor(
        [*range(NUM_CLASSES), *range(NUM_CLASSES), 0], dtype=torch.long
    )
    logits = torch.randn(len(targets), NUM_CLASSES)
    logits[torch.arange(len(targets)), targets] += 4.0
    logits[-1, 0] = -2.0
    logits[-1, 1] = 3.0
    row_indices = torch.arange(100_000, 100_000 + len(targets))

    dataloader = DataLoader(
        TensorDataset(logits, targets, row_indices),
        batch_size=7,
        shuffle=False,
    )
    result = evaluate_model(
        model=PassThroughClassifier(),
        dataloader=dataloader,
        device="cpu",
        criterion=nn.CrossEntropyLoss(),
        class_names=CLASS_NAMES,
        split_name="validation",
    )
    artifacts = save_evaluation_results(
        result,
        output_root=args.output_root,
        run_name=args.run_name,
        metadata={"seed": 86, "purpose": "evaluation-pipeline smoke test"},
        overwrite=args.overwrite,
    )

    print(f"macro_f1={result.metrics['macro_f1']:.6f}")
    print(f"accuracy={result.metrics['accuracy']:.6f}")
    print(f"artifacts={next(iter(artifacts.values())).parent.resolve()}")


if __name__ == "__main__":
    main()
