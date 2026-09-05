"""Shared validation and event-indexed forecasting contract.

Forecasts are the next unobserved release events after the training history. An
as-of date never causes an overdue event to be discarded or silently replaced.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from numbers import Integral
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd


class BasePredictor(ABC):
    """Predict successive release dates from a strictly ordered daily history."""

    MAX_PREDICTIONS = 100

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self._reset_fit()

    def _reset_fit(self):
        self.is_fitted = False
        self.performance_metrics = {}
        self.fitted_values = None
        self._training_dates = None

    @staticmethod
    def _validate_frame(df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame) or len(df) < 2:
            raise ValueError("At least two release dates are required")
        if not {'date', 'interval_days'}.issubset(df.columns):
            raise ValueError("History requires date and interval_days columns")
        try:
            dates = pd.to_datetime(df['date'], errors='raise')
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid release dates") from exc
        if dates.isna().any() or dates.dt.tz is not None:
            raise ValueError("Release dates must be valid timezone-naive calendar dates")
        if not dates.equals(dates.dt.normalize()):
            raise ValueError("Release dates must be normalized to calendar days")
        if not dates.is_monotonic_increasing or dates.duplicated().any():
            raise ValueError("Release dates must be unique and strictly increasing")
        try:
            intervals = np.asarray(df['interval_days'], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Release intervals must be numeric") from exc
        expected = dates.diff().dt.days.to_numpy(dtype=float)
        if not np.isnan(intervals[0]) or not np.all(np.isfinite(intervals[1:])):
            raise ValueError("Only the first interval may be missing; others must be finite")
        if not np.array_equal(intervals[1:], expected[1:]):
            raise ValueError("Release intervals must match consecutive calendar dates")
        result = df.copy().reset_index(drop=True)
        result['date'] = dates.to_numpy()
        result['interval_days'] = expected
        result['month'] = result['date'].dt.month
        result['days_since_start'] = (result['date'] - result['date'].iloc[0]).dt.days
        return result

    def _fit_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # A failed refit must never leave an older model usable by accident.
        self._reset_fit()
        result = self._validate_frame(df)
        self._training_dates = tuple(result['date'])
        return result

    def _predict_data(self, df: pd.DataFrame, n_predictions: int, today) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model is not fitted; call fit() first")
        self._validate_count(n_predictions)
        self._validate_today(today)
        result = self._validate_frame(df)
        if tuple(result['date']) != self._training_dates:
            raise ValueError("Prediction history differs from fitted history; refit the model")
        return result

    @staticmethod
    def _validate_count(n_predictions):
        if isinstance(n_predictions, bool) or not isinstance(n_predictions, Integral):
            raise ValueError("n_predictions must be an integer from 1 to 100")
        if not 1 <= n_predictions <= BasePredictor.MAX_PREDICTIONS:
            raise ValueError("n_predictions must be an integer from 1 to 100")

    @staticmethod
    def _validate_today(today):
        if today is not None:
            try:
                value = pd.Timestamp(today)
            except (TypeError, ValueError) as exc:
                raise ValueError("today must be a valid calendar date") from exc
            if pd.isna(value) or value.tzinfo is not None:
                raise ValueError("today must be a valid timezone-naive calendar date")

    @staticmethod
    def roll_future_dates(
        last_date: datetime,
        today: datetime,
        next_interval: Callable[[int, datetime], float],
        n_predictions: int,
    ) -> List[datetime]:
        """Generate exactly n successive events strictly after last_date.

        The legacy method name is retained for callers. ``today`` is validated
        for API compatibility, but does not filter forecasts: overdue dates are
        evidence of a missed forecast, not permission to invent unseen releases.
        Intervals must be finite and positive, and are rounded to whole days
        with a minimum of one day. Callback step zero is the first unseen event.
        """
        BasePredictor._validate_count(n_predictions)
        BasePredictor._validate_today(today)
        current = pd.Timestamp(last_date)
        if pd.isna(current) or current.tzinfo is not None or current != current.normalize():
            raise ValueError("last_date must be a timezone-naive calendar date")
        current = current.to_pydatetime()
        dates = []
        for step in range(n_predictions):
            interval = float(next_interval(step, current))
            if not np.isfinite(interval) or interval <= 0:
                raise ValueError("Predicted intervals must be finite and positive")
            try:
                current += timedelta(days=max(1, int(round(interval))))
            except (OverflowError, ValueError) as exc:
                raise ValueError("Predicted date exceeds supported calendar range") from exc
            dates.append(current)
        return dates

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
        """Fit the model to observed releases."""

    @abstractmethod
    def predict(self, df: pd.DataFrame, n_predictions: int = 5,
                today: datetime = None) -> List[datetime]:
        """Return the next n events, including forecasts overdue as of today."""

    def evaluate(self, y_true, y_pred) -> Dict[str, float]:
        """Record training diagnostics; these are not out-of-sample accuracy.

        R2 is undefined for fewer than two observations or a constant target.
        Invalid predictions are errors, never silently converted to zero loss.
        """
        actual = np.asarray(y_true, dtype=float)
        predicted = np.asarray(y_pred, dtype=float)
        if (actual.ndim != 1 or actual.size == 0 or actual.shape != predicted.shape
                or not np.all(np.isfinite(actual)) or not np.all(np.isfinite(predicted))):
            raise ValueError("Metrics require equally sized finite, nonempty vectors")
        residual = actual - predicted
        squared_error = float(np.dot(residual, residual))
        total = float(np.sum((actual - actual.mean()) ** 2))
        self.performance_metrics = {
            'MAE': float(np.mean(np.abs(residual))),
            'RMSE': float(np.sqrt(squared_error / actual.size)),
            'R2': 1.0 - squared_error / total if actual.size > 1 and total > 0 else float('nan'),
        }
        self.fitted_values = predicted.copy()
        return self.performance_metrics

    def get_info(self) -> Dict[str, Any]:
        return {'name': self.name, 'is_fitted': self.is_fitted,
                'performance': self.performance_metrics, 'metric_scope': 'training_fit'}
