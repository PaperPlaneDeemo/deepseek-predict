"""The single registry of prediction methods used by analysis and backtesting."""

from dataclasses import dataclass
from typing import Callable

from models.base import BasePredictor
from models.interval_based import (
    create_adaptive_interval_predictor,
    create_mean_interval_predictor,
    create_median_interval_predictor,
    create_recent_interval_predictor,
    create_weighted_interval_predictor,
)
from models.linear_models import (
    create_lasso_predictor,
    create_linear_predictor,
    create_ridge_predictor,
)
from models.statistical import StatisticalPredictor, TrendAnalysisPredictor
from models.time_series import ExponentialSmoothingPredictor, SeasonalPredictor


@dataclass(frozen=True)
class PredictorSpec:
    name: str
    group: str
    factory: Callable[[], BasePredictor]


PREDICTORS = (
    PredictorSpec("Linear Regression", "Linear Models", create_linear_predictor),
    PredictorSpec("Ridge Regression", "Linear Models", create_ridge_predictor),
    PredictorSpec("Lasso Regression", "Linear Models", create_lasso_predictor),
    PredictorSpec("Exponential Smoothing", "Time Series", ExponentialSmoothingPredictor),
    PredictorSpec("Seasonal Pattern", "Time Series", SeasonalPredictor),
    PredictorSpec("Mean Interval", "Interval Based", create_mean_interval_predictor),
    PredictorSpec("Median Interval", "Interval Based", create_median_interval_predictor),
    PredictorSpec("Recent 3 Mean", "Interval Based", create_recent_interval_predictor),
    PredictorSpec("Adaptive Interval", "Interval Based", create_adaptive_interval_predictor),
    PredictorSpec("Weighted Interval", "Interval Based", create_weighted_interval_predictor),
    PredictorSpec("Trend Analysis", "Statistical", TrendAnalysisPredictor),
    PredictorSpec("Statistical Ensemble", "Statistical", StatisticalPredictor),
)
