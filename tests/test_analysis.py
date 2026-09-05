"""Behavioral tests for the analysis pipeline using independently known dates."""

import json
import unittest
from datetime import date, datetime, timedelta

import pandas as pd

from deepseek_predict.analysis import run_analysis, run_backtest
from deepseek_predict.data import Release, releases_to_frame
from deepseek_predict.registry import PredictorSpec
from deepseek_predict.report import report_json


def catalog(count=5, interval=10):
    return [
        Release(f"model-{index}", f"Model {index}", date(2024, 1, 1) + timedelta(days=interval * index), None, "")
        for index in range(count)
    ]


class ProbePredictor:
    """A fixed-interval model with observable calls and configurable failures."""

    def __init__(self, interval=10, fail_at=(), result_factory=None, fit_error=False, fitted=True):
        self.interval = interval
        self.fail_at = fail_at
        self.result_factory = result_factory
        self.fit_error = fit_error
        self.fitted = fitted
        self.is_fitted = False
        self.fit_histories = []
        self.predict_histories = []
        self.as_of_values = []

    def fit(self, history):
        self.fit_histories.append(history.copy(deep=True))
        if self.fit_error:
            raise RuntimeError("intentional fit failure")
        self.is_fitted = self.fitted

    def predict(self, history, n_predictions, today):
        self.predict_histories.append(history.copy(deep=True))
        self.as_of_values.append(today)
        if len(history) in self.fail_at:
            raise RuntimeError("intentional prediction failure")
        last = history["date"].iloc[-1]
        if self.result_factory is not None:
            return self.result_factory(last, n_predictions)
        return [last + timedelta(days=self.interval * step) for step in range(1, n_predictions + 1)]


def spec(name="Known interval", **kwargs):
    return PredictorSpec(name, "Test", lambda: ProbePredictor(**kwargs))


class AnalysisTests(unittest.TestCase):
    def analyze(self, releases=None, **options):
        values = {"as_of": date(2024, 3, 1), "n_predictions": 3, "min_train_size": 2, "specs": [spec()]}
        values.update(options)
        return run_analysis(catalog() if releases is None else releases, **values)

    def test_constant_intervals_have_known_future_dates_and_zero_holdout_error(self):
        report = self.analyze()
        forecast = report["forecasts"][0]
        self.assertEqual(forecast["dates"], ["2024-02-20", "2024-03-01", "2024-03-11"])
        self.assertEqual(forecast["status"], "ok")
        self.assertEqual(report["backtest"]["total_folds"], 3)
        self.assertEqual([item["error_days"] for item in report["backtest"]["records"]], [0, 0, 0])
        summary = report["backtest"]["summaries"][0]
        for key in ["mae", "rmse", "bias"]:
            self.assertEqual(summary[key], 0.0)
        self.assertEqual(summary["hit_rate_30"], 1.0)
        self.assertEqual(summary["coverage"], 1.0)
        self.assertTrue(summary["eligible"])
        self.assertEqual(report["summary"]["best_method"], "Known interval")

    def test_date_error_sign_and_metrics_match_hand_calculation(self):
        for interval, signed_error in [(5, -5), (45, 35)]:
            with self.subTest(interval=interval):
                report = self.analyze(specs=[spec(interval=interval)])
                summary = report["backtest"]["summaries"][0]
                self.assertEqual([item["error_days"] for item in report["backtest"]["records"]], [signed_error] * 3)
                self.assertEqual(summary["mae"], abs(signed_error))
                self.assertEqual(summary["rmse"], abs(signed_error))
                self.assertEqual(summary["bias"], signed_error)
                self.assertEqual(summary["hit_rate_30"], float(abs(signed_error) <= 30))

    def test_fresh_factory_per_fit_and_every_fold_sees_only_its_prefix(self):
        models = []

        def factory():
            model = ProbePredictor()
            models.append(model)
            return model

        releases = catalog(count=6)
        as_of = releases[-2].date
        report = self.analyze(releases, as_of=as_of, specs=[PredictorSpec("Probe", "Test", factory)])
        self.assertEqual(len(models), 1 + report["backtest"]["total_folds"])
        self.assertEqual([len(model.fit_histories[0]) for model in models], [5, 2, 3, 4])
        self.assertEqual(models[0].as_of_values, [pd.Timestamp(as_of)])
        for model in models:
            self.assertEqual(len(model.fit_histories), 1)
            self.assertEqual(len(model.predict_histories), 1)
            pd.testing.assert_frame_equal(model.fit_histories[0], model.predict_histories[0])
            self.assertLessEqual(model.fit_histories[0]["date"].max().date(), as_of)
        for model, fold in zip(models[1:], report["backtest"]["records"]):
            train = model.fit_histories[0]
            self.assertEqual(train["version"].tolist(), [item.name for item in releases[:len(train)]])
            self.assertEqual(model.as_of_values, [train["date"].iloc[-1]])
            self.assertLess(train["date"].iloc[-1].date(), date.fromisoformat(fold["actual_date"]))
        self.assertEqual(report["meta"]["excluded_future_count"], 1)
        self.assertEqual(report["meta"]["release_count"], 5)
        self.assertEqual(report["meta"]["catalog_count"], 6)
        self.assertEqual(report["releases"][-1]["included"], False)
        self.assertEqual(report["meta"]["last_release"], as_of.isoformat())

    def test_catalog_and_targets_are_sorted_and_same_day_models_are_one_event(self):
        releases = catalog()
        releases.append(Release("same-day", "Same day", releases[2].date, None, ""))
        report = self.analyze(list(reversed(releases)))
        self.assertEqual(report["meta"]["release_count"], 6)
        self.assertEqual(report["meta"]["event_count"], 5)
        self.assertEqual([item["date"] for item in report["releases"]], sorted(item.date.isoformat() for item in releases))
        folds = report["backtest"]["records"]
        self.assertEqual([item["actual_date"] for item in folds], ["2024-01-21", "2024-01-31", "2024-02-10"])
        self.assertEqual(folds[0]["actual_name"], "Model 2 / Same day")

    def test_partial_and_full_failures_stay_in_coverage_and_hit_rate_denominators(self):
        report = self.analyze(specs=[
            spec("Partial", fail_at={3}),
            spec("Complete", interval=11),
            spec("Broken", fit_error=True),
        ])
        self.assertEqual(len(report["backtest"]["records"]), 9)
        summaries = {item["method"]: item for item in report["backtest"]["summaries"]}
        partial = summaries["Partial"]
        self.assertEqual(partial["successful_folds"], 2)
        self.assertEqual(partial["total_folds"], 3)
        self.assertAlmostEqual(partial["coverage"], 2 / 3)
        self.assertAlmostEqual(partial["hit_rate_30"], 2 / 3)
        self.assertEqual(partial["mae"], 0.0)
        self.assertFalse(partial["eligible"])
        broken = summaries["Broken"]
        self.assertEqual(broken["successful_folds"], 0)
        self.assertEqual(broken["coverage"], 0.0)
        self.assertEqual(broken["hit_rate_30"], 0.0)
        self.assertIsNone(broken["mae"])
        self.assertIsNone(broken["rmse"])
        self.assertIsNone(broken["bias"])
        self.assertFalse(broken["eligible"])
        self.assertEqual(report["summary"]["best_method"], "Complete")
        for record in report["backtest"]["records"]:
            if record["status"] == "error":
                self.assertIsNone(record["predicted_date"])
                self.assertIsNone(record["error_days"])
                self.assertIn("RuntimeError", record["error"])

    def test_no_predictions_and_all_method_failures_produce_inspectable_report(self):
        for options in [
            {"fit_error": True}, {"fitted": False},
            {"result_factory": lambda last, count: None},
            {"result_factory": lambda last, count: []},
        ]:
            with self.subTest(options=options):
                report = self.analyze(specs=[spec(**options)])
                self.assertEqual(report["forecasts"][0]["status"], "error")
                self.assertEqual(report["forecasts"][0]["dates"], [])
                self.assertIsNotNone(report["forecasts"][0]["error"])
                self.assertEqual(len(report["backtest"]["records"]), 3)
                self.assertTrue(all(item["status"] == "error" for item in report["backtest"]["records"]))
                for key in ["median_next_date", "earliest_next_date", "latest_next_date", "best_method", "best_mae"]:
                    self.assertIsNone(report["summary"][key])
                self.assertEqual(json.loads(report_json(report))["summary"], report["summary"])

    def test_forecast_failure_excludes_an_otherwise_complete_method_from_best_method(self):
        report = self.analyze(specs=[spec("Fails latest fit", fail_at={5}), spec("Available", interval=11)])
        summaries = {item["method"]: item for item in report["backtest"]["summaries"]}
        self.assertTrue(summaries["Fails latest fit"]["eligible"])
        self.assertEqual(summaries["Fails latest fit"]["mae"], 0.0)
        self.assertEqual(report["summary"]["best_method"], "Available")

    def test_no_holdout_folds_have_null_metrics_and_no_best_method(self):
        report = self.analyze(catalog(count=2), min_train_size=3)
        self.assertEqual(report["backtest"]["total_folds"], 0)
        self.assertEqual(report["backtest"]["records"], [])
        summary = report["backtest"]["summaries"][0]
        for key in ["mae", "rmse", "bias", "coverage", "hit_rate_30"]:
            self.assertIsNone(summary[key])
        self.assertFalse(summary["eligible"])
        self.assertIsNone(report["summary"]["best_method"])
        self.assertIsNone(report["summary"]["best_mae"])
        self.assertNotIn("NaN", report_json(report))
        self.assertEqual(json.loads(report_json(report))["backtest"]["summaries"][0], summary)

    def test_overdue_forecasts_stay_anchored_to_observations_when_as_of_advances(self):
        early = self.analyze(as_of=date(2024, 2, 11))
        late = self.analyze(as_of=date(2026, 1, 1))
        self.assertEqual(early["forecasts"][0]["dates"], late["forecasts"][0]["dates"])
        self.assertGreater(early["forecasts"][0]["days_from_as_of"], 0)
        self.assertLess(late["forecasts"][0]["days_from_as_of"], 0)
        self.assertEqual(early["backtest"], late["backtest"])

    def test_incomplete_non_daily_or_non_increasing_forecasts_are_rejected(self):
        bad_outputs = {
            "too many": lambda last, count: [last + timedelta(days=10 * step) for step in range(1, count + 2)],
            "last observation": lambda last, count: [last] * count,
            "before observation": lambda last, count: [last - timedelta(days=10)] * count,
            "duplicate": lambda last, count: [last + timedelta(days=10)] * count,
            "descending": lambda last, count: [last + timedelta(days=10 * step) for step in range(count, 0, -1)],
            "NaT": lambda last, count: [pd.NaT] * count,
            "with time": lambda last, count: [last + timedelta(days=10 * step, hours=1) for step in range(1, count + 1)],
            "timezone": lambda last, count: [(last + timedelta(days=10 * step)).tz_localize("UTC") for step in range(1, count + 1)],
            "invalid text": lambda last, count: ["no date"] * count,
        }
        for label, result_factory in bad_outputs.items():
            with self.subTest(output=label):
                report = self.analyze(n_predictions=2, specs=[spec(result_factory=result_factory)])
                forecast = report["forecasts"][0]
                self.assertEqual(forecast["status"], "error")
                self.assertEqual(forecast["dates"], [])
                self.assertIsNone(forecast["next_date"])
                self.assertTrue(forecast["error"])

    def test_invalid_analysis_parameters_fail_before_prediction(self):
        invalid = (
            [{"n_predictions": value} for value in [0, -1, 101, True, 2.5, "3", None]]
            + [{"min_train_size": value} for value in [0, 1, -1, True, 2.5, "3", None]]
            + [{"as_of": value} for value in ["2024-03-01", datetime(2024, 3, 1), None, True]]
            + [{"specs": []}, {"specs": [spec(), spec()]}]
        )
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.analyze(**options)

    def test_backtest_validates_its_own_minimum_window(self):
        frame = releases_to_frame(catalog(), date(2024, 3, 1))
        for value in [1, 0, -1, True, 2.5, "3"]:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "min_train_size"):
                run_backtest(frame, [spec()], value)


if __name__ == "__main__":
    unittest.main()
