"""Numerical validation shared by all predictors."""

from typing import Iterable, Tuple

import numpy as np


def compute_interval_bounds(
    values: Iterable[float], lower_quantile: float, upper_quantile: float,
) -> Tuple[float, float]:
    """Compute positive finite clipping bounds from observed intervals only."""
    values_array = np.asarray(list(values), dtype=float)
    if (values_array.ndim != 1 or values_array.size == 0
            or not np.all(np.isfinite(values_array)) or np.any(values_array <= 0)):
        raise ValueError("Intervals must be a nonempty vector of finite positive values")
    if not 0 <= lower_quantile <= upper_quantile <= 1:
        raise ValueError("Quantiles must satisfy 0 <= lower <= upper <= 1")
    if values_array.size == 1:
        floor, cap = values_array[0] * 0.8, values_array[0] * 1.2
    else:
        floor, cap = np.quantile(values_array, [lower_quantile, upper_quantile])
    floor = max(1.0, float(floor))
    return floor, max(floor, float(cap))


def validate_unit_parameter(value, name):
    if isinstance(value, (bool, str)):
        raise ValueError(f"{name} must be finite and in (0, 1]")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and in (0, 1]") from exc
    if not np.isfinite(value) or not 0 < value <= 1:
        raise ValueError(f"{name} must be finite and in (0, 1]")
    return value
