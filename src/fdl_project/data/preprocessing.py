"""Deterministic, categorical-preserving preprocessing for WM-811K maps."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Final, Literal

import torch
from torch import Tensor
from torch.nn import functional as F

GeometryStrategy = Literal["pad", "resize", "letterbox"]
EncodingStrategy = Literal["one_hot", "single_channel"]
NormalizationStrategy = Literal["none", "divide_by_two"]

WAFER_STATE_COUNT: Final[int] = 3
WAFER_STATE_NAMES: Final[tuple[str, ...]] = (
    "outside_wafer",
    "functional_die",
    "defective_die",
)


@dataclass(frozen=True)
class PreprocessingConfig:
    """Immutable geometry and encoding decisions for one model pipeline."""

    target_size: tuple[int, int] = (64, 64)
    geometry: GeometryStrategy = "letterbox"
    encoding: EncodingStrategy = "one_hot"
    normalization: NormalizationStrategy = "none"
    allow_upscale: bool = True

    def __post_init__(self) -> None:
        if (
            not isinstance(self.target_size, tuple)
            or len(self.target_size) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in self.target_size
            )
            or any(value <= 0 for value in self.target_size)
        ):
            raise ValueError("target_size must be a tuple of two positive integers.")
        if self.geometry not in {"pad", "resize", "letterbox"}:
            raise ValueError(f"Unsupported geometry strategy: {self.geometry!r}.")
        if self.encoding not in {"one_hot", "single_channel"}:
            raise ValueError(f"Unsupported encoding strategy: {self.encoding!r}.")
        if self.normalization not in {"none", "divide_by_two"}:
            raise ValueError(
                f"Unsupported normalization strategy: {self.normalization!r}."
            )
        if self.encoding == "one_hot" and self.normalization != "none":
            raise ValueError("One-hot inputs must use normalization='none'.")

    @property
    def output_shape(self) -> tuple[int, int, int]:
        channels = WAFER_STATE_COUNT if self.encoding == "one_hot" else 1
        return channels, *self.target_size

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["target_size"] = list(self.target_size)
        return payload


DEFAULT_PREPROCESSING_CONFIG: Final[PreprocessingConfig] = PreprocessingConfig()


def validate_wafer_map(wafer_map: Any) -> Tensor:
    """Return a two-dimensional int64 tensor containing only states 0, 1, 2."""

    try:
        tensor = torch.as_tensor(wafer_map)
    except (TypeError, ValueError) as exc:
        raise ValueError("wafer_map must be a numeric two-dimensional array.") from exc

    if tensor.ndim != 2 or tensor.numel() == 0:
        raise ValueError(
            f"wafer_map must be a non-empty two-dimensional array; received {tuple(tensor.shape)!r}."
        )
    if tensor.dtype == torch.bool or tensor.is_complex():
        raise ValueError(
            f"wafer_map must contain numeric states 0, 1, 2; received {tensor.dtype}."
        )
    if tensor.is_floating_point():
        if not torch.isfinite(tensor).all():
            raise ValueError("wafer_map contains NaN or infinite values.")
        if not torch.equal(tensor, tensor.round()):
            raise ValueError(
                "wafer_map contains fractional values instead of states 0, 1, 2."
            )

    categorical = tensor.to(dtype=torch.long)
    if not torch.all((categorical >= 0) & (categorical < WAFER_STATE_COUNT)):
        unique_values = torch.unique(categorical).tolist()
        raise ValueError(
            f"wafer_map contains states outside 0, 1, 2: {unique_values!r}."
        )
    return categorical


def _center_on_canvas(categorical: Tensor, target_size: tuple[int, int]) -> Tensor:
    source_height, source_width = categorical.shape
    target_height, target_width = target_size
    if source_height > target_height or source_width > target_width:
        raise ValueError(
            f"Cannot pad source shape {(source_height, source_width)!r} to smaller "
            f"target {target_size!r}. Use resize or letterbox instead."
        )

    top = (target_height - source_height) // 2
    left = (target_width - source_width) // 2
    canvas = torch.zeros(target_size, dtype=torch.long, device=categorical.device)
    canvas[top : top + source_height, left : left + source_width] = categorical
    return canvas


def _resize_nearest(categorical: Tensor, target_size: tuple[int, int]) -> Tensor:
    resized = F.interpolate(
        categorical[None, None].to(dtype=torch.float32),
        size=target_size,
        mode="nearest-exact",
    )
    return resized[0, 0].to(dtype=torch.long)


def _letterbox(
    categorical: Tensor,
    target_size: tuple[int, int],
    *,
    allow_upscale: bool,
) -> Tensor:
    source_height, source_width = categorical.shape
    target_height, target_width = target_size
    scale = min(target_height / source_height, target_width / source_width)
    if not allow_upscale:
        scale = min(scale, 1.0)

    resized_height = min(target_height, max(1, round(source_height * scale)))
    resized_width = min(target_width, max(1, round(source_width * scale)))
    resized = _resize_nearest(categorical, (resized_height, resized_width))
    return _center_on_canvas(resized, target_size)


def transform_categorical_map(
    wafer_map: Any,
    config: PreprocessingConfig = DEFAULT_PREPROCESSING_CONFIG,
    *,
    validate: bool = True,
) -> Tensor:
    """Apply deterministic geometry while keeping categorical states exact.

    ``validate=False`` skips the state and dtype checks. Only pass it for maps
    already validated once -- the dataset cache does this -- since the checks
    are what keep an out-of-range value from silently becoming a class.
    """

    categorical = (
        validate_wafer_map(wafer_map)
        if validate
        else torch.as_tensor(wafer_map).to(dtype=torch.long)
    )
    if config.geometry == "pad":
        return _center_on_canvas(categorical, config.target_size)
    if config.geometry == "resize":
        return _resize_nearest(categorical, config.target_size)
    return _letterbox(
        categorical,
        config.target_size,
        allow_upscale=config.allow_upscale,
    )


def encode_categorical_map(categorical: Tensor, config: PreprocessingConfig) -> Tensor:
    """Encode a transformed categorical map for a PyTorch image model."""

    if config.encoding == "one_hot":
        return (
            F.one_hot(categorical, num_classes=WAFER_STATE_COUNT)
            .permute(2, 0, 1)
            .to(dtype=torch.float32)
        )

    single_channel = categorical[None].to(dtype=torch.float32)
    if config.normalization == "divide_by_two":
        single_channel = single_channel / (WAFER_STATE_COUNT - 1)
    return single_channel


def decode_preprocessed_map(tensor: Tensor, config: PreprocessingConfig) -> Tensor:
    """Recover categorical states for validation and visual diagnostics."""

    if tuple(tensor.shape) != config.output_shape:
        raise ValueError(
            f"Preprocessed tensor must have shape {config.output_shape!r}; "
            f"received {tuple(tensor.shape)!r}."
        )
    if not torch.isfinite(tensor).all():
        raise ValueError("Preprocessed tensor contains NaN or infinite values.")

    if config.encoding == "one_hot":
        if torch.any((tensor != 0) & (tensor != 1)) or not torch.allclose(
            tensor.sum(dim=0),
            torch.ones(
                config.target_size,
                dtype=tensor.dtype,
                device=tensor.device,
            ),
        ):
            raise ValueError(
                "One-hot tensor must contain exactly one active state per pixel."
            )
        return tensor.argmax(dim=0).to(dtype=torch.long)

    decoded = tensor[0]
    if config.normalization == "divide_by_two":
        decoded = decoded * (WAFER_STATE_COUNT - 1)
    rounded = decoded.round()
    if not torch.allclose(decoded, rounded, rtol=0, atol=1e-6):
        raise ValueError(
            "Single-channel tensor does not encode exact categorical states."
        )
    return validate_wafer_map(rounded)


class WaferMapPreprocessor:
    """Callable deterministic preprocessing transform."""

    def __init__(
        self, config: PreprocessingConfig = DEFAULT_PREPROCESSING_CONFIG
    ) -> None:
        self.config = config

    def transform_categories(self, wafer_map: Any, *, validate: bool = True) -> Tensor:
        return transform_categorical_map(wafer_map, self.config, validate=validate)

    def __call__(self, wafer_map: Any, *, validate: bool = True) -> Tensor:
        categorical = self.transform_categories(wafer_map, validate=validate)
        return encode_categorical_map(categorical, self.config)
