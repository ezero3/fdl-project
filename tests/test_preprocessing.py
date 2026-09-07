import numpy as np
import pytest
import torch

from fdl_project.data.preprocessing import (
    DEFAULT_PREPROCESSING_CONFIG,
    PreprocessingConfig,
    WaferMapPreprocessor,
    decode_preprocessed_map,
    transform_categorical_map,
    validate_wafer_map,
)
from fdl_project.analysis.preprocessing_analysis import benchmark_preprocessors

SAMPLE_MAP = np.array(
    [
        [0, 1, 0],
        [1, 2, 1],
    ],
    dtype=np.uint8,
)


def test_default_configuration_is_immutable_and_model_ready() -> None:
    assert DEFAULT_PREPROCESSING_CONFIG == PreprocessingConfig()
    assert DEFAULT_PREPROCESSING_CONFIG.output_shape == (3, 64, 64)
    with pytest.raises(AttributeError):
        DEFAULT_PREPROCESSING_CONFIG.geometry = "resize"  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_size": (0, 64)},
        {"target_size": [64, 64]},
        {"geometry": "crop"},
        {"encoding": "rgb"},
        {"normalization": "standardize"},
        {"encoding": "one_hot", "normalization": "divide_by_two"},
    ],
)
def test_invalid_preprocessing_configurations_are_rejected(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        PreprocessingConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "wafer_map",
    [
        np.array([]),
        np.zeros((1, 2, 3)),
        np.array([[0, 3]]),
        np.array([[0.0, 1.5]]),
        np.array([[0.0, np.nan]]),
        np.array([[True, False]]),
        np.array([["0", "1"]]),
    ],
)
def test_malformed_wafer_maps_are_rejected(wafer_map: np.ndarray) -> None:
    with pytest.raises(ValueError, match="wafer_map"):
        validate_wafer_map(wafer_map)


def test_padding_centers_map_without_changing_categories() -> None:
    config = PreprocessingConfig(target_size=(4, 5), geometry="pad")
    transformed = transform_categorical_map(SAMPLE_MAP, config)
    expected = torch.zeros((4, 5), dtype=torch.long)
    expected[1:3, 1:4] = torch.from_numpy(SAMPLE_MAP).long()

    assert torch.equal(transformed, expected)
    assert set(torch.unique(transformed).tolist()) == {0, 1, 2}


def test_padding_rejects_a_smaller_target_instead_of_cropping() -> None:
    config = PreprocessingConfig(target_size=(1, 2), geometry="pad")
    with pytest.raises(ValueError, match="Cannot pad"):
        transform_categorical_map(SAMPLE_MAP, config)


def test_direct_resize_uses_only_nearest_categorical_values() -> None:
    config = PreprocessingConfig(target_size=(7, 8), geometry="resize")
    transformed = transform_categorical_map(SAMPLE_MAP, config)

    assert transformed.shape == (7, 8)
    assert set(torch.unique(transformed).tolist()) <= {0, 1, 2}


def test_letterbox_preserves_aspect_ratio_and_centers_result() -> None:
    wide_map = np.array([[1, 1, 2, 2], [1, 1, 2, 2]], dtype=np.uint8)
    config = PreprocessingConfig(target_size=(8, 8), geometry="letterbox")
    transformed = transform_categorical_map(wide_map, config)

    assert transformed.shape == (8, 8)
    assert torch.count_nonzero(transformed[:2]) == 0
    assert torch.count_nonzero(transformed[6:]) == 0
    assert torch.count_nonzero(transformed[2:6]) > 0


def test_letterbox_can_disable_upscaling() -> None:
    config = PreprocessingConfig(
        target_size=(8, 8), geometry="letterbox", allow_upscale=False
    )
    transformed = transform_categorical_map(SAMPLE_MAP, config)

    assert torch.count_nonzero(transformed) == np.count_nonzero(SAMPLE_MAP)


def test_one_hot_encoding_is_exact_and_round_trips() -> None:
    preprocessor = WaferMapPreprocessor(
        PreprocessingConfig(target_size=(4, 5), geometry="pad", encoding="one_hot")
    )
    output = preprocessor(SAMPLE_MAP)
    decoded = decode_preprocessed_map(output, preprocessor.config)

    assert output.dtype == torch.float32
    assert output.shape == (3, 4, 5)
    assert torch.all(output.sum(dim=0) == 1)
    assert torch.equal(decoded, preprocessor.transform_categories(SAMPLE_MAP))


@pytest.mark.parametrize(
    ("normalization", "expected_values"),
    [
        ("none", {0.0, 1.0, 2.0}),
        ("divide_by_two", {0.0, 0.5, 1.0}),
    ],
)
def test_single_channel_encoding_round_trips(
    normalization: str, expected_values: set[float]
) -> None:
    config = PreprocessingConfig(
        target_size=(2, 3),
        geometry="pad",
        encoding="single_channel",
        normalization=normalization,  # type: ignore[arg-type]
    )
    preprocessor = WaferMapPreprocessor(config)
    output = preprocessor(SAMPLE_MAP)

    assert output.shape == (1, 2, 3)
    assert set(torch.unique(output).tolist()) == expected_values
    assert torch.equal(
        decode_preprocessed_map(output, config), torch.from_numpy(SAMPLE_MAP)
    )


def test_preprocessing_is_deterministic() -> None:
    preprocessor = WaferMapPreprocessor(DEFAULT_PREPROCESSING_CONFIG)
    first = preprocessor(SAMPLE_MAP)
    second = preprocessor(SAMPLE_MAP.copy())
    assert torch.equal(first, second)


def test_benchmark_reports_fidelity_and_memory_tradeoffs() -> None:
    maps = [SAMPLE_MAP, np.rot90(SAMPLE_MAP).copy()]
    preprocessors = {
        "pad": WaferMapPreprocessor(
            PreprocessingConfig(target_size=(4, 4), geometry="pad")
        ),
        "resize": WaferMapPreprocessor(
            PreprocessingConfig(target_size=(4, 4), geometry="resize")
        ),
    }
    benchmark = benchmark_preprocessors(maps, preprocessors, reference_batch_size=8)

    assert benchmark["candidate"].tolist() == ["pad", "resize"]
    assert (benchmark["milliseconds_per_map"] > 0).all()
    assert (benchmark["reference_batch_mib"] > 0).all()
    assert (
        benchmark.loc[benchmark["candidate"] == "pad", "lost_defect_maps"].item() == 0
    )
    assert benchmark.loc[
        benchmark["candidate"] == "pad", "median_abs_defective_ratio_error"
    ].item() == pytest.approx(0.0)
