"""Regression coverage for forecast identity, causal features and model state."""

from datetime import datetime, timedelta
import unittest
import warnings

import numpy as np
import pandas as pd

from models.base import BasePredictor
from models.interval_based import AdaptiveIntervalPredictor, IntervalPredictor, WeightedIntervalPredictor
from models.linear_models import LinearPredictor
from models.statistical import (CyclicalAnalysisPredictor, SeasonalDecomposePredictor,
                                StatisticalPredictor, TrendAnalysisPredictor)
from models.time_series import ExponentialSmoothingPredictor, SeasonalPredictor
from models.utils import compute_interval_bounds


def history(intervals, start='2024-01-01'):
    dates = [pd.Timestamp(start)]
    for interval in intervals:
        dates.append(dates[-1] + pd.Timedelta(days=interval))
    df = pd.DataFrame({'version': [f'v{i}' for i in range(len(dates))], 'date': dates})
    df['days_since_start'] = (df['date'] - df['date'].iloc[0]).dt.days
    df['interval_days'] = df['days_since_start'].diff()
    df['month'] = df['date'].dt.month
    return df


def predictors():
    return [LinearPredictor(kind) for kind in ('linear', 'ridge', 'lasso')] + [
        IntervalPredictor(kind) for kind in ('mean', 'median', 'recent')
    ] + [AdaptiveIntervalPredictor(), WeightedIntervalPredictor(),
         ExponentialSmoothingPredictor(), SeasonalPredictor(),
         TrendAnalysisPredictor(), SeasonalDecomposePredictor(),
         CyclicalAnalysisPredictor(), StatisticalPredictor()]


class ForecastContractTests(unittest.TestCase):
    def test_every_model_preserves_overdue_event_identity(self):
        df = history([12, 18, 8, 22, 12, 18])
        for model in predictors():
            with self.subTest(model=model.name):
                model.fit(df)
                before = model.predict(df, 8, datetime(2024, 1, 1))
                overdue = model.predict(df, 8, datetime(2040, 1, 1))
                self.assertEqual(before, overdue)
                self.assertEqual(len(overdue), 8)
                self.assertLess(overdue[-1], datetime(2040, 1, 1))
                self.assertTrue(all(right > left for left, right in zip([df['date'].iloc[-1]] + overdue, overdue)))

    def test_minimal_and_constant_history_all_models(self):
        for intervals in ([7], [7, 7, 7, 7, 7]):
            df = history(intervals)
            expected = [df['date'].iloc[-1] + timedelta(days=7 * i) for i in range(1, 4)]
            for model in predictors():
                with self.subTest(model=model.name, intervals=intervals), warnings.catch_warnings():
                    warnings.simplefilter('error')
                    model.fit(df)
                    self.assertEqual(model.predict(df, 3), expected)
                    self.assertTrue(np.isfinite(model.performance_metrics['MAE']))
                    self.assertTrue(np.isnan(model.performance_metrics['R2']))

    def test_all_models_reject_invalid_forecast_counts(self):
        df = history([10, 20, 15])
        for model in predictors():
            model.fit(df)
            for count in (0, -1, 101, 2.5, True, '3', None):
                with self.subTest(model=model.name, count=count), self.assertRaises(ValueError):
                    model.predict(df, count)

    def test_invalid_refit_cannot_reuse_old_fitted_state(self):
        valid = history([10, 20, 15])
        invalid = valid.copy()
        invalid.loc[2, 'interval_days'] = np.inf
        for model in predictors():
            with self.subTest(model=model.name):
                model.fit(valid)
                with self.assertRaises(ValueError):
                    model.fit(invalid)
                self.assertFalse(model.is_fitted)
                self.assertEqual(model.performance_metrics, {})
                with self.assertRaises(ValueError):
                    model.predict(valid)

    def test_refit_matches_fresh_model_after_shorter_history(self):
        long = history([10, 40, 20, 80, 50])
        short = history([9])
        for model, fresh in zip(predictors(), predictors()):
            with self.subTest(model=model.name):
                model.fit(long)
                model.fit(short)
                fresh.fit(short)
                self.assertEqual(model.predict(short), fresh.predict(short))
                self.assertAlmostEqual(model.performance_metrics['MAE'], fresh.performance_metrics['MAE'])

    def test_prediction_requires_same_observed_history(self):
        model = IntervalPredictor()
        model.fit(history([10, 20]))
        with self.assertRaisesRegex(ValueError, 'refit'):
            model.predict(history([10, 20, 30]))

    def test_invalid_history_is_rejected(self):
        df = history([10, 20, 15])
        invalid_frames = [df.iloc[:1], df.iloc[::-1]]
        for column, value in [('interval_days', 0), ('interval_days', np.nan), ('date', pd.NaT)]:
            bad = df.copy()
            bad.loc[2, column] = value
            invalid_frames.append(bad)
        duplicate = df.copy()
        duplicate.loc[2, 'date'] = duplicate.loc[1, 'date']
        invalid_frames.append(duplicate)
        invalid_frames.append(df.assign(date=df['date'] + pd.Timedelta(hours=1)))
        invalid_frames.append(df.assign(date=df['date'].dt.tz_localize('UTC')))
        for invalid in invalid_frames:
            with self.subTest(invalid=str(invalid)), self.assertRaises(ValueError):
                IntervalPredictor().fit(invalid)

    def test_roll_rejects_invalid_intervals_and_retains_fraction_rounding(self):
        for value in (np.nan, np.inf, -1, 0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                BasePredictor.roll_future_dates(datetime(2024, 1, 1), None, lambda step, date: value, 1)
        dates = BasePredictor.roll_future_dates(datetime(2024, 1, 1), None, lambda step, date: 0.1, 3)
        self.assertEqual(dates[-1], datetime(2024, 1, 4))
        with self.assertRaises(ValueError):
            BasePredictor.roll_future_dates(datetime(2024, 1, 1), pd.NaT, lambda step, date: 1, 1)


class NumericalRegressionTests(unittest.TestCase):
    def test_linear_features_are_prefix_invariant_and_use_start_month(self):
        prefix = history([30, 8, 40], '2024-01-25')
        extended = history([30, 8, 40, 200, 5], '2024-01-25')
        model = LinearPredictor()
        x_prefix, targets = model._create_features(prefix)
        x_full, _ = model._create_features(extended)
        np.testing.assert_array_equal(x_prefix, x_full[:len(x_prefix)])
        np.testing.assert_array_equal(targets, [8, 40])
        self.assertEqual(x_prefix[0, 0], 30)
        self.assertEqual(x_prefix[0, 1], 30)
        self.assertAlmostEqual(x_prefix[0, 3], np.sin(2 * np.pi * 2 / 12))
        empty, targets = model._create_features(history([9]))
        self.assertEqual(empty.shape, (0, 5))
        self.assertEqual(len(targets), 0)

    def test_adaptive_trend_is_centered_on_historical_mean(self):
        model = AdaptiveIntervalPredictor()
        df = history([10, 20, 30])
        model.fit(df)
        self.assertEqual(model.base_interval, 20)
        self.assertEqual(model.trend_slope, 10)
        np.testing.assert_allclose(model.fitted_values, [12, 20, 28])
        model.interval_cap = 1000
        self.assertEqual((model.predict(df, 1)[0] - df['date'].iloc[-1]).days, 40)

    def test_decomposition_detrends_before_month_effects(self):
        df = history([10, 20, 30])
        model = SeasonalDecomposePredictor()
        model.fit(df)
        self.assertEqual(model.base_level, 20)
        self.assertEqual(model.trend_slope, 10)
        self.assertTrue(all(abs(effect) < 1e-10 for effect in model.monthly_effects.values()))
        model.interval_cap = 1000
        self.assertEqual((model.predict(df, 1)[0] - df['date'].iloc[-1]).days, 40)

    def test_seasonal_interval_uses_known_start_month(self):
        df = history([40, 5], '2024-01-25')
        model = SeasonalPredictor()
        model.fit(df)
        self.assertEqual(model.monthly_patterns, {1: 40.0, 3: 5.0})
        model.interval_floor = 1
        self.assertEqual((model.predict(df, 1)[0] - df['date'].iloc[-1]).days, 5)

    def test_cycle_continues_phase_including_incomplete_last_cycle(self):
        model = CyclicalAnalysisPredictor()
        df = history([10, 20, 10, 20, 10])
        model.fit(df)
        self.assertEqual(model.cycle_length, 2)
        self.assertEqual(model.cycle_pattern, [10, 20])
        dates = model.predict(df, 3)
        self.assertEqual([(right - left).days for left, right in zip([df['date'].iloc[-1]] + dates, dates)], [20, 10, 20])
        model.fit(history([10, 20, 10, 20]))
        self.assertEqual(model.cycle_length, 2)

    def test_trend_forecast_is_anchored_after_last_observed_event(self):
        df = history([1, 1, 1, 200])
        model = TrendAnalysisPredictor()
        model.fit(df)
        self.assertGreater(model.predict(df, 1)[0], df['date'].iloc[-1])

    def test_smoothing_scores_pre_update_predictions(self):
        model = ExponentialSmoothingPredictor(alpha=1)
        model.fit(history([10, 20, 30]))
        self.assertEqual(model.performance_metrics['MAE'], 10)
        np.testing.assert_array_equal(model.fitted_values, [10, 20])

    def test_ensemble_weights_handle_zero_and_invalid_errors(self):
        weights = StatisticalPredictor._weights_from_errors([0, 4, np.nan, 0, np.inf, -2])
        np.testing.assert_array_equal(weights, [0.5, 0, 0, 0.5, 0, 0])
        np.testing.assert_allclose(StatisticalPredictor._weights_from_errors([2, 4]), [2 / 3, 1 / 3])
        self.assertTrue(np.all(np.isfinite(StatisticalPredictor._weights_from_errors([1e-320, 1e200]))))
        with self.assertRaises(ValueError):
            StatisticalPredictor._weights_from_errors([np.nan, np.inf, -1])

    def test_ensemble_metrics_evaluate_actual_combined_predictions(self):
        df = history([10, 70, 20, 60, 30, 50])
        model = StatisticalPredictor()
        model.fit(df)
        fitted = sum(weight * child.fitted_values for child, weight in zip(model.predictors, model.weights))
        residual = df['interval_days'].iloc[1:].to_numpy() - fitted
        self.assertAlmostEqual(model.performance_metrics['MAE'], np.abs(residual).mean())
        self.assertAlmostEqual(model.performance_metrics['RMSE'], np.sqrt(np.square(residual).mean()))
        weighted_rmse = sum(weight * child.performance_metrics['RMSE'] for child, weight in zip(model.predictors, model.weights))
        self.assertGreater(abs(model.performance_metrics['RMSE'] - weighted_rmse), 0.01)

    def test_constructor_and_bounds_parameters_are_validated(self):
        for constructor in (WeightedIntervalPredictor, ExponentialSmoothingPredictor):
            for value in (0, -0.1, 1.1, np.nan, np.inf, True, '0.5', None):
                with self.subTest(constructor=constructor.__name__, value=value), self.assertRaises(ValueError):
                    constructor(value)
        with self.assertRaises(ValueError):
            IntervalPredictor('unsupported')
        with self.assertRaises(ValueError):
            LinearPredictor('unsupported')
        for values, low, high in [([], 0, 1), ([np.inf], 0, 1), ([-1], 0, 1), ([1], 0.9, 0.1), ([1], np.nan, 1)]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                compute_interval_bounds(values, low, high)
        self.assertEqual(compute_interval_bounds([0.1], 0, 1), (1, 1))


if __name__ == '__main__':
    unittest.main()
