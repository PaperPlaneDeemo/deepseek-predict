"""Autoregressive interval models with features available at forecast time."""

from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from .base import BasePredictor
from .utils import compute_interval_bounds


class LinearPredictor(BasePredictor):
    """Predict an interval using prior intervals and its known starting month.

    The first interval has no observed lag and is excluded from regression.
    With only one observed interval, forecast that interval as a cold start.
    """

    def __init__(self, model_type='linear'):
        if model_type not in ('linear', 'ridge', 'lasso'):
            raise ValueError(f"Unsupported linear model: {model_type}")
        super().__init__(f'Linear {model_type.title()}')
        self.model_type = model_type
        self.scaler = StandardScaler()

    @staticmethod
    def _feature_row(history, month):
        recent = np.asarray(history[-3:], dtype=float)
        return [history[-1], float(recent.mean()), float(recent.std()),
                np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)]

    def _create_features(self, df: pd.DataFrame):
        """Feature row i uses only releases strictly before its target event.

        Appending future observations cannot change existing feature rows.
        Starting months come from date, never the unknown target release month.
        """
        df = self._validate_frame(df)
        values = df['interval_days'].iloc[1:].to_numpy()
        features = [self._feature_row(values[:index], df['date'].iloc[index].month)
                    for index in range(1, len(values))]
        return np.asarray(features, dtype=float).reshape(-1, 5), values[1:].copy()

    def fit(self, df: pd.DataFrame) -> None:
        df = self._fit_data(df)
        values = df['interval_days'].iloc[1:].to_numpy()
        self.min_interval, self.max_interval = compute_interval_bounds(values, 0.05, 0.95)
        self.base_interval = float(values.mean())
        self.scaler = StandardScaler()
        self.model = {'linear': LinearRegression, 'ridge': Ridge, 'lasso': Lasso}[self.model_type]()
        features, targets = self._create_features(df)
        self._constant_fallback = len(targets) < 2 or np.ptp(targets) == 0
        if self._constant_fallback:
            self.base_interval = float(targets.mean()) if len(targets) else float(values[-1])
            observed = targets if len(targets) else values
            self.evaluate(observed, np.full_like(observed, self.base_interval))
        else:
            if self.model_type in ('ridge', 'lasso'):
                features = self.scaler.fit_transform(features)
            self.model.fit(features, targets)
            predictions = np.clip(self.model.predict(features), self.min_interval, self.max_interval)
            self.evaluate(targets, predictions)
        self.is_fitted = True

    def predict(self, df: pd.DataFrame, n_predictions: int = 5,
                today: datetime = None) -> List[datetime]:
        df = self._predict_data(df, n_predictions, today)
        history = df['interval_days'].iloc[1:].tolist()

        def next_interval(step, current_date):
            if self._constant_fallback:
                interval = self.base_interval
            else:
                features = np.asarray([self._feature_row(history, current_date.month)])
                if self.model_type in ('ridge', 'lasso'):
                    features = self.scaler.transform(features)
                interval = float(self.model.predict(features)[0])
            interval = float(np.clip(interval, self.min_interval, self.max_interval))
            # Recursion must use the actual day-rounded interval being emitted.
            history.append(max(1, round(interval)))
            return interval

        return self.roll_future_dates(df['date'].iloc[-1], today, next_interval, n_predictions)


def create_linear_predictor():
    return LinearPredictor('linear')


def create_ridge_predictor():
    return LinearPredictor('ridge')


def create_lasso_predictor():
    return LinearPredictor('lasso')
