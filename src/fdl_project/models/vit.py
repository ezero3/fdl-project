"""A small Vision Transformer, included as a controlled negative result.

Expect this to lose, and run it anyway.

Why it should lose here: a ViT replaces the convolutional prior -- locality and
translation equivariance -- with learned attention, and pays for that with
data. The usual figures are tens of millions of images, or heavy augmentation
and distillation to compensate. We have ~121k training wafers at 64x64, 85% of
them one class, on a rigid grid of dies where locality is not an assumption but
a fact of the physics.

Why it is still worth running: "we tried a transformer and it underperformed,
here is the parameter count and the learning curve" is a real finding about
this dataset, and it is the kind of comparison the report is asking for. A
negative result that is measured beats an architecture that is dismissed.

CBAM is deliberately not accepted here: this model is already attention, and
bolting a convolutional attention block onto token embeddings would confuse
what the comparison is testing.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from fdl_project.constants import NUM_CLASSES
from fdl_project.data.preprocessing import WAFER_STATE_COUNT


class WaferViT(nn.Module):
    """Patch embedding, a class token, learned positions, transformer encoder.

    ``image_size`` must be given because the positional table is sized at
    construction; a config that changes ``target_size`` must change it too.
    """

    def __init__(
        self,
        *,
        image_size: int = 64,
        patch_size: int = 8,
        embedding_dimension: int = 192,
        depth: int = 6,
        heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention: str | None = None,
    ) -> None:
        super().__init__()
        if attention is not None:
            raise ValueError(
                "WaferViT does not accept a convolutional attention block: it is "
                "already an attention model. Compare it against a CNN with CBAM "
                "instead of adding one here."
            )
        if image_size % patch_size:
            raise ValueError(
                f"image_size {image_size} must be divisible by patch_size {patch_size}."
            )
        if embedding_dimension % heads:
            raise ValueError(
                f"embedding_dimension {embedding_dimension} must be divisible by "
                f"heads {heads}."
            )
        if depth < 1:
            raise ValueError("depth must be at least 1.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")

        num_patches = (image_size // patch_size) ** 2
        # A strided convolution is exactly non-overlapping patch embedding.
        self.patch_embedding = nn.Conv2d(
            WAFER_STATE_COUNT,
            embedding_dimension,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.class_token = nn.Parameter(torch.zeros(1, 1, embedding_dimension))
        self.positions = nn.Parameter(
            torch.zeros(1, num_patches + 1, embedding_dimension)
        )
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dimension,
                nhead=heads,
                dim_feedforward=int(embedding_dimension * mlp_ratio),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,  # pre-norm trains far more stably at this size
            ),
            num_layers=depth,
            # Pre-norm layers cannot use the nested-tensor fast path anyway;
            # saying so explicitly keeps the warning out of every run log.
            enable_nested_tensor=False,
        )
        self.normalization = nn.LayerNorm(embedding_dimension)
        self.head = nn.Linear(embedding_dimension, NUM_CLASSES)

        self.image_size = image_size
        self.patch_size = patch_size
        nn.init.trunc_normal_(self.positions, std=0.02)
        nn.init.trunc_normal_(self.class_token, std=0.02)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != WAFER_STATE_COUNT:
            raise ValueError(
                "WaferViT inputs must have shape "
                f"(batch_size, {WAFER_STATE_COUNT}, height, width)."
            )
        if not inputs.is_floating_point():
            raise TypeError("WaferViT inputs must be floating-point tensors.")
        if inputs.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                f"WaferViT was built for {self.image_size}x{self.image_size} inputs "
                f"but received {tuple(inputs.shape[-2:])}. The positional table is "
                "sized at construction, so image_size must match "
                "data.preprocessing.target_size."
            )

        tokens = self.patch_embedding(inputs).flatten(2).transpose(1, 2)
        class_tokens = self.class_token.expand(len(tokens), -1, -1)
        tokens = torch.cat([class_tokens, tokens], dim=1) + self.positions
        tokens = self.encoder(self.dropout(tokens))
        return self.head(self.normalization(tokens[:, 0]))
