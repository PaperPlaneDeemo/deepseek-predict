"""Mean, robust, recent, weighted and trend-adjusted release intervals."""

from datetime import datetime
from typing import List

import numpy as np
import pandas as pd

from .base import BasePredictor
from .utils import compute_interval_bounds, validate_unit_parameter


class IntervalPredictor(BasePredictor):
    def __init__(self, strategy='mean'):
        names = {'mean': 'Mean Interval', 'median': 'Median Interval', 'recent': 'Recent 3 Mean'}
        if strategy not in names:
            raise ValueError(f"Unsupported interval strategy: {strategy}")
        super().__init__(names[strategy])
        self.strategy = strategy

    def fit(self, df: pd.DataFrame) -> None:
        df = self._fit_data(df)
        values = df['interval_days'].iloc[1:].to_numpy()
        self.interval_floor, self.interval_cap = compute_interval_bounds(values, 0.1, 0.9)
        if self.strategy == 'median':
            self.interval_value = float(np.median(values))
        else:
            self.interval_value = float(np.mean(values[-3:] if self.strategy == 'recent' else values))
        interval = float(np.clip(self.interval_value, self.interval_floor, self.interval_cap))
        self.evaluate(values, np.full_like(values, interval))
        self.is_fitted = True

    def predict(self, df: pd.DataFrame, n_predictions: int = 5,
                today: datetime = None) -> List[datetime]:
        df = self._predict_data(df, n_predictions, today)
        interval = float(np.clip(self.interval_value, self.interval_floor, self.interval_cap))
        return self.roll_future_dates(df['date'].iloc[-1], today, lambda step, date: interval, n_predictions)


class AdaptiveIntervalPredictor(BasePredictor):
    """A centered interval trend, bounded by observed interval quantiles."""

    def __init__(self):
        super().__init__('Adaptive Interval')

    def fit(self, df: pd.DataFrame) -> None:
        df = self._fit_data(df)
        values = df['interval_days'].iloc[1:].to_numpy()
        self.history_length = len(values)
        self.trend_origin = (len(values) - 1) / 2
        self.base_interval = float(values.mean())
        self.trend_slope = 0.0
        if len(values) >= 3:
            centered_index = np.arange(len(values)) - self.trend_origin
            self.trend_slope = float(np.dot(centered_index, values) / np.dot(centered_index, centered_index))
            self.trend_slope = float(np.clip(self.trend_slope, -self.base_interval / 2, self.base_interval / 2))
        self.interval_floor, self.interval_cap = compute_interval_bounds(values, 0.1, 0.9)
        fitted = self.base_interval + self.trend_slope * (np.arange(len(values)) - self.trend_origin)
        self.evaluate(values, np.clip(fitted, self.interval_floor, self.interval_cap))
        self.is_fitted = True

    def predict(self, df: pd.DataFrame, n_predictions: int = 5,
                today: datetime = None) -> List[datetime]:
        df = self._predict_data(df, n_predictions, today)

        def next_interval(step, current_date):
            interval = self.base_interval + self.trend_slope * (self.history_length + step - self.trend_origin)
            return float(np.clip(interval, self.interval_floor, self.interval_cap))

        return self.roll_future_dates(df['date'].iloc[-1], today, next_interval, n_predictions)


class WeightedIntervalPredictor(BasePredictor):
    def __init__(self, decay_rate=0.8):
        super().__init__('Weighted Interval')
        self.decay_rate = validate_unit_parameter(decay_rate, 'decay_rate')

    def fit(self, df: pd.DataFrame) -> None:
        df = self._fit_data(df)
        values = df['interval_days'].iloc[1:].to_numpy()
        self.interval_floor, self.interval_cap = compute_interval_bounds(values, 0.1, 0.9)
        weights = self.decay_rate ** np.arange(len(values) - 1, -1, -1)
        self.weighted_interval = float(np.average(values, weights=weights))
        interval = float(np.clip(self.weighted_interval, self.interval_floor, self.interval_cap))
        self.evaluate(values, np.full_like(values, interval))
        self.is_fitted = True

    def predict(self, df: pd.DataFrame, n_predictions: int = 5,
                today: datetime = None) -> List[datetime]:
        df = self._predict_data(df, n_predictions, today)
        interval = float(np.clip(self.weighted_interval, self.interval_floor, self.interval_cap))
        return self.roll_future_dates(df['date'].iloc[-1], today, lambda step, date: interval, n_predictions)


def create_mean_interval_predictor():
    return IntervalPredictor('mean')


def create_median_interval_predictor():
    return IntervalPredictor('median')


def create_recent_interval_predictor():
    return IntervalPredictor('recent')


def create_adaptive_interval_predictor():
    return AdaptiveIntervalPredictor()


def create_weighted_interval_predictor():
    return WeightedIntervalPredictor()
