"""Bounded trends, interval cycles and a transparent statistical ensemble."""

from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd

from .base import BasePredictor
from .utils import compute_interval_bounds


class TrendAnalysisPredictor(BasePredictor):
    """Use the fitted cadence slope, anchored at the last observed release.

    A regression intercept describes historical dates; it must never move the
    forecast origin backwards or imply unobserved release events already exist.
    """

    def __init__(self):
        super().__init__('Trend Analysis')

    def fit(self, df: pd.DataFrame) -> None:
        df = self._fit_data(df)
        values = df['interval_days'].iloc[1:].to_numpy()
        index = np.arange(len(df), dtype=float)
        elapsed = df['days_since_start'].to_numpy(dtype=float)
        self.slope, self.intercept = map(float, np.polyfit(index, elapsed, 1, w=np.linspace(0.5, 1, len(df))))
        self.interval_floor, self.interval_cap = compute_interval_bounds(values, 0.05, 0.95)
        self.slope = float(np.clip(self.slope, self.interval_floor, self.interval_cap))
        self.evaluate(values, np.full_like(values, self.slope))
        self.is_fitted = True

    def predict(self, df: pd.DataFrame, n_predictions: int = 5,
                today: datetime = None) -> List[datetime]:
        df = self._predict_data(df, n_predictions, today)
        return self.roll_future_dates(df['date'].iloc[-1], today, lambda step, date: self.slope, n_predictions)


class SeasonalDecomposePredictor(BasePredictor):
    """Centered linear trend plus starting-month means of detrended residuals."""

    def __init__(self):
        super().__init__('Seasonal Decompose')

    def fit(self, df: pd.DataFrame) -> None:
        df = self._fit_data(df)
        values = df['interval_days'].iloc[1:].to_numpy()
        self.history_length = len(values)
        self.trend_origin = (len(values) - 1) / 2
        index = np.arange(len(values)) - self.trend_origin
        self.base_level = float(values.mean())
        self.trend_slope = float(np.dot(index, values) / np.dot(index, index)) if len(values) > 1 else 0.0
        trend = self.base_level + self.trend_slope * index
        residuals = values - trend
        start_months = df['date'].iloc[:-1].dt.month.to_numpy()
        self.monthly_effects = {month: float(residuals[start_months == month].mean())
                                if np.any(start_months == month) else 0.0 for month in range(1, 13)}
        self.interval_floor, self.interval_cap = compute_interval_bounds(values, 0.05, 0.95)
        fitted = trend + np.asarray([self.monthly_effects[month] for month in start_months])
        self.evaluate(values, np.clip(fitted, self.interval_floor, self.interval_cap))
        self.is_fitted = True

    def predict(self, df: pd.DataFrame, n_predictions: int = 5,
                today: datetime = None) -> List[datetime]:
        df = self._predict_data(df, n_predictions, today)

        def next_interval(step, current_date):
            trend = self.base_level + self.trend_slope * (self.history_length + step - self.trend_origin)
            interval = trend + self.monthly_effects[current_date.month]
            return float(np.clip(interval, self.interval_floor, self.interval_cap))

        return self.roll_future_dates(df['date'].iloc[-1], today, next_interval, n_predictions)


class CyclicalAnalysisPredictor(BasePredictor):
    """Learn short repeating interval patterns and continue the observed phase."""

    def __init__(self):
        super().__init__('Cyclical Analysis')

    def fit(self, df: pd.DataFrame) -> None:
        df = self._fit_data(df)
        values = df['interval_days'].iloc[1:].to_numpy()
        self.history_length = len(values)
        self.cycle_length = 1
        best_score = 0.0
        # Include n // 2: four observations are enough to test a two-step cycle.
        for length in range(2, min(6, len(values) // 2) + 1):
            score = self._evaluate_cycle(values, length)
            if score > best_score:
                self.cycle_length, best_score = length, score
        self.cycle_pattern = self._extract_cycle_pattern(values, self.cycle_length)
        self.interval_floor, self.interval_cap = compute_interval_bounds(values, 0.05, 0.95)
        fitted = [self.cycle_pattern[index % self.cycle_length] for index in range(len(values))]
        self.evaluate(values, np.clip(fitted, self.interval_floor, self.interval_cap))
        self.is_fitted = True

    @staticmethod
    def _evaluate_cycle(intervals, cycle_length):
        values = np.asarray(intervals, dtype=float)
        if len(values) < 2 * cycle_length or np.var(values) == 0:
            return 0.0
        # Score each cycle using earlier phase observations, not its own target.
        errors = [values[index] - values[index % cycle_length:index:cycle_length].mean()
                  for index in range(cycle_length, len(values))]
        return float(1 - np.mean(np.square(errors)) / np.var(values))

    @staticmethod
    def _extract_cycle_pattern(intervals, cycle_length):
        values = np.asarray(intervals, dtype=float)
        return [float(values[phase::cycle_length].mean()) for phase in range(cycle_length)]

    def predict(self, df: pd.DataFrame, n_predictions: int = 5,
                today: datetime = None) -> List[datetime]:
        df = self._predict_data(df, n_predictions, today)

        def next_interval(step, current_date):
            phase = (self.history_length + step) % self.cycle_length
            return float(np.clip(self.cycle_pattern[phase], self.interval_floor, self.interval_cap))

        return self.roll_future_dates(df['date'].iloc[-1], today, next_interval, n_predictions)


class StatisticalPredictor(BasePredictor):
    """Combine event-aligned forecasts using finite training-error weights.

    Weights are a fitting heuristic, not a claim of validation accuracy. Overall
    accuracy must be measured separately by chronological holdout evaluation.
    Ensemble training metrics are calculated from the combined fitted intervals.
    """

    def __init__(self):
        super().__init__('Statistical Ensemble')
        self.predictors = [TrendAnalysisPredictor(), SeasonalDecomposePredictor(), CyclicalAnalysisPredictor()]
        self.weights = None
        self.fit_errors = {}

    @staticmethod
    def _weights_from_errors(errors):
        errors = np.asarray(errors, dtype=float)
        valid = np.isfinite(errors) & (errors >= 0)
        if not np.any(valid):
            raise ValueError("No finite nonnegative model errors available for ensemble")
        weights = np.zeros_like(errors)
        perfect = valid & (errors == 0)
        if np.any(perfect):
            weights[perfect] = 1 / np.count_nonzero(perfect)
        else:
            # Relative reciprocals avoid overflow for a very small positive MAE.
            weights[valid] = errors[valid].min() / errors[valid]
            weights /= weights.sum()
        return weights

    def fit(self, df: pd.DataFrame) -> None:
        df = self._fit_data(df)
        self.weights = None
        self.fit_errors = {}
        values = df['interval_days'].iloc[1:].to_numpy()
        errors = []
        for predictor in self.predictors:
            predictor._reset_fit()
            try:
                predictor.fit(df)
                fitted = np.asarray(predictor.fitted_values, dtype=float)
                if fitted.shape != values.shape or not np.all(np.isfinite(fitted)):
                    raise ValueError("Submodel returned invalid fitted intervals")
                errors.append(predictor.performance_metrics.get('MAE', float('nan')))
            except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
                predictor._reset_fit()
                self.fit_errors[predictor.name] = str(exc)
                errors.append(float('nan'))
        self.weights = self._weights_from_errors(errors).tolist()
        fitted = sum(weight * predictor.fitted_values for weight, predictor in zip(self.weights, self.predictors) if weight > 0)
        self.evaluate(values, fitted)
        self.is_fitted = True

    def predict(self, df: pd.DataFrame, n_predictions: int = 5,
                today: datetime = None) -> List[datetime]:
        df = self._predict_data(df, n_predictions, today)
        origin = df['date'].iloc[-1].to_pydatetime()
        offsets = np.zeros(n_predictions)
        for predictor, weight in zip(self.predictors, self.weights):
            if weight <= 0:
                continue
            predictions = predictor.predict(df, n_predictions, today)
            if len(predictions) != n_predictions:
                raise ValueError(f"{predictor.name} returned an incomplete forecast")
            offsets += weight * np.asarray([(date - origin).days for date in predictions])
        dates = []
        previous_offset = 0
        for value in offsets:
            offset = max(previous_offset + 1, int(round(value)))
            dates.append(origin + timedelta(days=offset))
            previous_offset = offset
        return dates
