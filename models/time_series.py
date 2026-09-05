"""Exponential smoothing and starting-month interval seasonality."""

from datetime import datetime
from typing import List

import numpy as np
import pandas as pd

from .base import BasePredictor
from .utils import compute_interval_bounds, validate_unit_parameter


class ExponentialSmoothingPredictor(BasePredictor):
    def __init__(self, alpha=0.3):
        super().__init__('Exponential Smoothing')
        self.alpha = validate_unit_parameter(alpha, 'alpha')

    def fit(self, df: pd.DataFrame) -> None:
        df = self._fit_data(df)
        values = df['interval_days'].iloc[1:].to_numpy()
        self.interval_floor, self.interval_cap = compute_interval_bounds(values, 0.05, 0.95)
        level = float(values[0])
        fitted = []
        for value in values[1:]:
            fitted.append(level)
            level = self.alpha * float(value) + (1 - self.alpha) * level
        self.smoothed_interval = float(np.clip(level, self.interval_floor, self.interval_cap))
        # Score predictions before incorporating their target observation.
        if fitted:
            self.evaluate(values[1:], np.asarray(fitted))
        else:
            self.evaluate(values, values)
        self.is_fitted = True

    def predict(self, df: pd.DataFrame, n_predictions: int = 5,
                today: datetime = None) -> List[datetime]:
        df = self._predict_data(df, n_predictions, today)
        return self.roll_future_dates(df['date'].iloc[-1], today,
                                      lambda step, date: self.smoothed_interval, n_predictions)


class SeasonalPredictor(BasePredictor):
    """Mean interval by its known start month; unseen months use the mean."""

    def __init__(self):
        super().__init__('Seasonal Pattern')

    def fit(self, df: pd.DataFrame) -> None:
        df = self._fit_data(df)
        values = df['interval_days'].iloc[1:].to_numpy()
        start_months = df['date'].iloc[:-1].dt.month.to_numpy()
        self.monthly_patterns = {int(month): float(values[start_months == month].mean())
                                 for month in np.unique(start_months)}
        self.default_interval = float(values.mean())
        self.interval_floor, self.interval_cap = compute_interval_bounds(values, 0.05, 0.95)
        fitted = [self.monthly_patterns[month] for month in start_months]
        self.evaluate(values, np.clip(fitted, self.interval_floor, self.interval_cap))
        self.is_fitted = True

    def predict(self, df: pd.DataFrame, n_predictions: int = 5,
                today: datetime = None) -> List[datetime]:
        df = self._predict_data(df, n_predictions, today)

        def next_interval(step, current_date):
            interval = self.monthly_patterns.get(current_date.month, self.default_interval)
            return float(np.clip(interval, self.interval_floor, self.interval_cap))

        return self.roll_future_dates(df['date'].iloc[-1], today, next_interval, n_predictions)
