"""
时间序列预测器
包括指数平滑、季节性模式等时间序列分析方法
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List

from .base import BasePredictor
from .utils import compute_interval_bounds


class ExponentialSmoothingPredictor(BasePredictor):
    """指数平滑预测器"""
    
    def __init__(self, alpha=0.3):
        super().__init__("Exponential Smoothing")
        self.alpha = alpha
        self.smoothed_interval = None
        self.interval_floor = None
        self.interval_cap = None

    def fit(self, df: pd.DataFrame) -> None:
        """训练指数平滑模型"""
        intervals = df['interval_days'].dropna()

        if intervals.empty:
            raise ValueError("interval_days 为空，无法训练指数平滑模型")

        values = intervals.values.astype(float)
        self.interval_floor, self.interval_cap = compute_interval_bounds(values, 0.05, 0.95)

        # 指数平滑
        smoothed_values = [float(intervals.iloc[0])]
        for value in intervals.iloc[1:]:
            next_smoothed = self.alpha * float(value) + (1 - self.alpha) * smoothed_values[-1]
            smoothed_values.append(next_smoothed)

        smoothed_array = np.clip(np.array(smoothed_values, dtype=float), self.interval_floor, self.interval_cap)
        self.smoothed_interval = float(smoothed_array[-1])
        true_intervals = intervals.values.astype(float)
        self.evaluate(true_intervals, smoothed_array)
        self.is_fitted = True
    
    def predict(self, df: pd.DataFrame, n_predictions: int = 5, 
                today: datetime = None) -> List[datetime]:
        """生成指数平滑预测"""
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit()方法")
        
        if today is None:
            today = datetime.now()
        
        future_predictions = []
        last_date = df['date'].iloc[-1]
        
        for i in range(n_predictions):
            interval = float(np.clip(self.smoothed_interval, self.interval_floor, self.interval_cap))
            last_date = last_date + timedelta(days=int(round(interval)))
            if last_date > today:
                future_predictions.append(last_date)

        return future_predictions


class SeasonalPredictor(BasePredictor):
    """季节性模式预测器"""
    
    def __init__(self):
        super().__init__("Seasonal Pattern")
        self.monthly_patterns = {}
        self.default_interval = None
        self.interval_floor = None
        self.interval_cap = None

    def fit(self, df: pd.DataFrame) -> None:
        """学习季节性模式"""
        intervals = df['interval_days'].dropna()

        if intervals.empty:
            raise ValueError("interval_days 为空，无法训练季节性模型")

        # 分析每个月的发布模式
        df_with_interval = df.iloc[1:].copy()
        df_with_interval['interval'] = intervals.values

        self.monthly_patterns = df_with_interval.groupby('month')['interval'].mean().to_dict()
        self.default_interval = float(intervals.mean())

        values = intervals.values.astype(float)
        self.interval_floor, self.interval_cap = compute_interval_bounds(values, 0.05, 0.95)

        true_intervals = values
        predicted_intervals = []
        for idx, month in enumerate(df_with_interval['month']):
            interval = self.monthly_patterns.get(month, self.default_interval)
            predicted_intervals.append(float(np.clip(interval, self.interval_floor, self.interval_cap)))

        predicted_array = np.array(predicted_intervals, dtype=float)
        self.evaluate(true_intervals, predicted_array)
        self.is_fitted = True
    
    def predict(self, df: pd.DataFrame, n_predictions: int = 5, 
                today: datetime = None) -> List[datetime]:
        """基于季节性模式预测"""
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit()方法")
        
        if today is None:
            today = datetime.now()
        
        future_predictions = []
        last_date = df['date'].iloc[-1]
        
        for i in range(n_predictions):
            next_month = (last_date + timedelta(days=30)).month
            interval = self.monthly_patterns.get(next_month, self.default_interval)
            interval = float(np.clip(interval, self.interval_floor, self.interval_cap))

            last_date = last_date + timedelta(days=int(round(interval)))
            if last_date > today:
                future_predictions.append(last_date)

        return future_predictions
