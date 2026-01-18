"""
Shared utilities for predictor modules.
"""

from typing import Iterable, Tuple

import numpy as np


def compute_interval_bounds(
    values: Iterable[float],
    lower_quantile: float,
    upper_quantile: float,
) -> Tuple[float, float]:
    """Return (floor, cap) bounds for interval values with sane defaults."""
    values_array = np.asarray(values, dtype=float)

    if values_array.size == 0:
        raise ValueError("intervals is empty; cannot compute bounds")

    if values_array.size > 1:
        floor = max(1.0, float(np.quantile(values_array, lower_quantile)))
        cap = float(np.quantile(values_array, upper_quantile))
    else:
        value = float(values_array[0])
        floor = max(1.0, value * 0.8)
        cap = value * 1.2

    return floor, cap
