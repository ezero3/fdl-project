"""Quantitative comparison helpers for deterministic preprocessing candidates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from time import perf_counter

import numpy as np
import pandas as pd

from fdl_project.preprocessing import (
    WaferMapPreprocessor,
    decode_preprocessed_map,
    validate_wafer_map,
)


def _defective_ratio(categorical: np.ndarray) -> float:
    active_count = int(np.count_nonzero(categorical > 0))
    if active_count == 0:
        return 0.0
    return float(np.count_nonzero(categorical == 2) / active_count)


def _active_bbox_aspect_ratio(categorical: np.ndarray) -> float:
    coordinates = np.argwhere(categorical > 0)
    if len(coordinates) == 0:
        return 1.0
    height = int(coordinates[:, 0].max() - coordinates[:, 0].min() + 1)
    width = int(coordinates[:, 1].max() - coordinates[:, 1].min() + 1)
    return width / height


def benchmark_preprocessors(
    wafer_maps: Sequence[object],
    preprocessors: Mapping[str, WaferMapPreprocessor],
    *,
    reference_batch_size: int = 64,
) -> pd.DataFrame:
    """Compare spatial fidelity, input memory, and CPU transformation time."""

    if not wafer_maps:
        raise ValueError("At least one wafer map is required for benchmarking.")
    if not preprocessors:
        raise ValueError("At least one preprocessor is required for benchmarking.")
    if reference_batch_size <= 0:
        raise ValueError("reference_batch_size must be positive.")

    originals = [
        validate_wafer_map(wafer_map).cpu().numpy() for wafer_map in wafer_maps
    ]
    rows: list[dict[str, object]] = []
    for candidate_name, preprocessor in preprocessors.items():
        defect_ratio_errors: list[float] = []
        bbox_log_aspect_errors: list[float] = []
        output_active_fractions: list[float] = []
        lost_defect_maps = 0
        example_output = None
        started = perf_counter()

        for original in originals:
            output = preprocessor(original)
            transformed = (
                decode_preprocessed_map(output, preprocessor.config).cpu().numpy()
            )
            example_output = output

            original_defect_count = int(np.count_nonzero(original == 2))
            transformed_defect_count = int(np.count_nonzero(transformed == 2))
            if original_defect_count > 0 and transformed_defect_count == 0:
                lost_defect_maps += 1

            defect_ratio_errors.append(
                abs(_defective_ratio(transformed) - _defective_ratio(original))
            )
            original_aspect = _active_bbox_aspect_ratio(original)
            transformed_aspect = _active_bbox_aspect_ratio(transformed)
            bbox_log_aspect_errors.append(
                abs(np.log(transformed_aspect / original_aspect))
            )
            output_active_fractions.append(
                float(np.count_nonzero(transformed > 0) / transformed.size)
            )

        elapsed = perf_counter() - started
        assert example_output is not None
        bytes_per_map = example_output.numel() * example_output.element_size()
        rows.append(
            {
                "candidate": candidate_name,
                "geometry": preprocessor.config.geometry,
                "encoding": preprocessor.config.encoding,
                "normalization": preprocessor.config.normalization,
                "target_height": preprocessor.config.target_size[0],
                "target_width": preprocessor.config.target_size[1],
                "channels": preprocessor.config.output_shape[0],
                "milliseconds_per_map": elapsed / len(originals) * 1_000,
                "reference_batch_mib": bytes_per_map * reference_batch_size / (1024**2),
                "median_abs_defective_ratio_error": float(
                    np.median(defect_ratio_errors)
                ),
                "p95_abs_defective_ratio_error": float(
                    np.quantile(defect_ratio_errors, 0.95)
                ),
                "median_bbox_log_aspect_error": float(
                    np.median(bbox_log_aspect_errors)
                ),
                "p95_bbox_log_aspect_error": float(
                    np.quantile(bbox_log_aspect_errors, 0.95)
                ),
                "lost_defect_maps": lost_defect_maps,
                "mean_output_active_fraction": float(np.mean(output_active_fractions)),
            }
        )
    return pd.DataFrame(rows)
