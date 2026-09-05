"""Forecast observed release events and evaluate expanding-window holdouts.

Forecasts always begin at the last *observed* release. An overdue estimate stays
overdue; an absence of new observations does not create synthetic releases.
All report metrics are held-out, one-step release-date errors in calendar days.
"""

from datetime import date, datetime
from statistics import mean, median
from typing import Sequence

import numpy as np
import pandas as pd

from .data import Release, releases_to_frame
from .registry import PREDICTORS, PredictorSpec


def _validate_integer(value: int, name: str, minimum: int, maximum: int | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")


def _predict(spec: PredictorSpec, frame: pd.DataFrame, horizon: int, as_of: date) -> list[str]:
    """Fit a fresh model; reject incomplete or invalid output at the boundary."""
    predictor = spec.factory()
    history = frame.copy(deep=True)
    predictor.fit(history)
    if not predictor.is_fitted:
        raise ValueError("predictor did not finish fitting")
    predictions = predictor.predict(history, n_predictions=horizon, today=pd.Timestamp(as_of))
    if predictions is None or len(predictions) != horizon:
        raise ValueError(f"expected {horizon} predictions, received {0 if predictions is None else len(predictions)}")
    previous = pd.Timestamp(frame["date"].iloc[-1])
    dates = []
    for value in predictions:
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp) or timestamp.tzinfo is not None or timestamp != timestamp.normalize():
            raise ValueError("predictions must be finite, timezone-free calendar dates")
        if timestamp <= previous:
            raise ValueError("predictions must increase strictly after the last observed release")
        dates.append(timestamp.date().isoformat())
        previous = timestamp
    return dates


def _error_message(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def run_backtest(frame: pd.DataFrame, specs: Sequence[PredictorSpec], min_train_size: int) -> dict:
    """Each fold fits only its prefix; every attempted fold has one record."""
    _validate_integer(min_train_size, "min_train_size", 2)
    total_folds = max(0, len(frame) - min_train_size)
    records = []
    summaries = []
    for spec in specs:
        method_records = []
        for split in range(min_train_size, len(frame)):
            train = frame.iloc[:split].copy()
            target = frame.iloc[split]
            record = {
                "method": spec.name,
                "group": spec.group,
                "fold": split - min_train_size + 1,
                "train_end": train["date"].iloc[-1].date().isoformat(),
                "actual_name": target["version"],
                "actual_date": target["date"].date().isoformat(),
                "predicted_date": None,
                "error_days": None,
                "status": "error",
                "error": None,
            }
            try:
                predicted = _predict(spec, train, 1, train["date"].iloc[-1].date())[0]
                record.update(
                    predicted_date=predicted,
                    error_days=(date.fromisoformat(predicted) - target["date"].date()).days,
                    status="ok",
                )
            except Exception as error:
                record["error"] = _error_message(error)
            method_records.append(record)
        errors = [item["error_days"] for item in method_records if item["status"] == "ok"]
        successful = len(errors)
        summaries.append({
            "method": spec.name,
            "group": spec.group,
            "mae": float(np.mean(np.abs(errors))) if errors else None,
            "rmse": float(np.sqrt(np.mean(np.square(np.asarray(errors, dtype=float))))) if errors else None,
            "bias": float(mean(errors)) if errors else None,
            # Failed attempts count as misses, never disappear from denominators.
            "hit_rate_30": sum(abs(error) <= 30 for error in errors) / total_folds if total_folds else None,
            "coverage": successful / total_folds if total_folds else None,
            "successful_folds": successful,
            "total_folds": total_folds,
            "eligible": total_folds > 0 and successful == total_folds,
        })
        records.extend(method_records)
    summaries.sort(key=lambda item: (not item["eligible"], item["mae"] if item["mae"] is not None else float("inf"), item["method"]))
    return {"min_train_size": min_train_size, "total_folds": total_folds, "summaries": summaries, "records": records}


def run_analysis(
    releases: Sequence[Release],
    *,
    as_of: date,
    n_predictions: int = 3,
    min_train_size: int = 3,
    dataset_path: str = "",
    specs: Sequence[PredictorSpec] = PREDICTORS,
) -> dict:
    """Return a JSON-compatible report without file I/O or terminal output."""
    if not isinstance(as_of, date) or isinstance(as_of, datetime):
        raise ValueError("as_of must be a calendar date")
    _validate_integer(n_predictions, "n_predictions", 1, 100)
    _validate_integer(min_train_size, "min_train_size", 2)
    if not specs or len({spec.name for spec in specs}) != len(specs):
        raise ValueError("predictor registry must have unique names and at least one method")
    releases = sorted(releases, key=lambda release: (release.date, release.id))
    frame = releases_to_frame(releases, as_of=as_of)
    forecasts = []
    for spec in specs:
        result = {"method": spec.name, "group": spec.group, "dates": [], "next_date": None,
                  "days_from_as_of": None, "status": "error", "error": None}
        try:
            dates = _predict(spec, frame, n_predictions, as_of)
            result.update(dates=dates, next_date=dates[0], status="ok",
                          days_from_as_of=(date.fromisoformat(dates[0]) - as_of).days)
        except Exception as error:
            result["error"] = _error_message(error)
        forecasts.append(result)

    backtest = run_backtest(frame, specs, min_train_size)
    dates = [date.fromisoformat(item["next_date"]).toordinal() for item in forecasts if item["status"] == "ok"]
    successful_methods = {item["method"] for item in forecasts if item["status"] == "ok"}
    ranked = [item for item in backtest["summaries"] if item["eligible"] and item["method"] in successful_methods]
    warnings = [
        "预测对象是下一次有模型发布的日期；同一天的多个模型合并为一次发布事件。",
        "预测从最后已观测发布日推进；已逾期的估计会原样保留，不自动跳到下一轮。",
        "方法间的日期范围表示估计分歧，不是统计置信区间；回测排名也不保证未来表现。",
    ]
    missing_sources = sum(release.source_url is None for release in releases)
    if missing_sources:
        warnings.append(f"数据目录有 {missing_sources} 条记录未提供来源链接；迁移记录尚未独立核实，日期以数据文件为准。")
    excluded = sum(release.date > as_of for release in releases)
    if excluded:
        warnings.append(f"{excluded} 条晚于分析基准日的记录已从训练和回测中排除。")
    if not backtest["total_folds"]:
        warnings.append("历史事件不足以进行所配置的回测；不显示最佳方法。")
    if any(item["status"] == "error" for item in forecasts):
        warnings.append("部分方法未能生成预测，请查看方法对比中的错误详情。")
    failed_folds = sum(item["status"] == "error" for item in backtest["records"])
    if failed_folds:
        warnings.append(f"共 {failed_folds} 次方法回测失败；未覆盖全部折次的方法不参与排名，失败计入命中率分母。")
    intervals = frame["interval_days"].dropna()
    return {
        "schema_version": 1,
        "meta": {
            "as_of": as_of.isoformat(), "dataset_name": "DeepSeek", "dataset_path": dataset_path,
            "n_predictions": n_predictions, "min_train_size": min_train_size,
            "release_count": len(releases) - excluded, "catalog_count": len(releases),
            "event_count": len(frame), "excluded_future_count": excluded,
            "first_release": frame["date"].iloc[0].date().isoformat(),
            "last_release": frame["date"].iloc[-1].date().isoformat(),
            "elapsed_days": (as_of - frame["date"].iloc[-1].date()).days,
        },
        "releases": [
            {"id": release.id, "name": release.name, "date": release.date.isoformat(),
             "source_url": release.source_url, "notes": release.notes, "included": release.date <= as_of}
            for release in releases
        ],
        "forecasts": forecasts,
        "backtest": backtest,
        "summary": {
            "median_next_date": date.fromordinal(int(median(dates))).isoformat() if dates else None,
            "earliest_next_date": date.fromordinal(min(dates)).isoformat() if dates else None,
            "latest_next_date": date.fromordinal(max(dates)).isoformat() if dates else None,
            "best_method": ranked[0]["method"] if ranked else None,
            "best_mae": ranked[0]["mae"] if ranked else None,
            "mean_interval": float(intervals.mean()), "median_interval": float(intervals.median()),
        },
        "warnings": warnings,
    }
