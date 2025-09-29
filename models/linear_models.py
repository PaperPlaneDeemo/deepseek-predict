"""
线性模型预测器
包括线性回归、Ridge回归、Lasso回归等
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler

from .base import BasePredictor


class LinearPredictor(BasePredictor):
    """线性回归预测器"""
    
    def __init__(self, model_type='linear'):
        super().__init__(f"Linear {model_type.title()}")
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.min_interval = None
        self.max_interval = None
        self.base_interval = None
        
        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'ridge':
            self.model = Ridge(alpha=1.0)
        elif model_type == 'lasso':
            self.model = Lasso(alpha=1.0)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    def _create_features(self, df: pd.DataFrame):
        """创建时间序列特征"""
        intervals = df['interval_days'].dropna()

        if intervals.empty:
            raise ValueError("interval_days 为空，无法训练线性模型")

        values = intervals.values.astype(float)
        X = []
        y = []

        overall_mean = float(values.mean())

        for idx, interval in enumerate(values):
            if idx > 0:
                prev_interval = values[idx - 1]
                recent = values[max(0, idx - 3):idx]
            else:
                prev_interval = overall_mean
                recent = values[:1]

            rolling_mean = recent.mean() if len(recent) > 0 else overall_mean
            rolling_std = recent.std(ddof=0) if len(recent) > 1 else 0.0
            month = df['month'].iloc[idx]
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)

            X.append([
                idx,  # 时间索引
                prev_interval,
                rolling_mean,
                rolling_std,
                month_sin,
                month_cos
            ])
            y.append(interval)

        return np.array(X, dtype=float), np.array(y, dtype=float)

    def fit(self, df: pd.DataFrame) -> None:
        """训练线性模型"""
        X, y = self._create_features(df)

        values = df['interval_days'].dropna().values.astype(float)
        if len(values) > 1:
            self.min_interval = max(1.0, float(np.percentile(values, 5)))
            self.max_interval = float(np.percentile(values, 95))
        else:
            value = float(values[0])
            self.min_interval = max(1.0, value * 0.8)
            self.max_interval = value * 1.2

        self.base_interval = float(values.mean())

        if self.model_type in ['ridge', 'lasso']:
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            y_pred = self.model.predict(X_scaled)
        else:
            self.model.fit(X, y)
            y_pred = self.model.predict(X)

        self.evaluate(y, y_pred)
        self.is_fitted = True

    def predict(self, df: pd.DataFrame, n_predictions: int = 5, 
                today: datetime = None) -> List[datetime]:
        """生成预测"""
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit()方法")
        
        if today is None:
            today = datetime.now()
        
        future_predictions = []
        intervals = df['interval_days'].dropna()

        if intervals.empty:
            raise ValueError("interval_days 为空，无法进行预测")

        history = intervals.values.astype(float).tolist()
        last_date = df['date'].iloc[-1]

        for step in range(n_predictions):
            prev_interval = history[-1] if history else self.base_interval
            recent = history[-3:] if history else [self.base_interval]
            rolling_mean = np.mean(recent)
            rolling_std = np.std(recent, ddof=0) if len(recent) > 1 else 0.0
            month = last_date.month
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)

            X_future = np.array([[len(history), prev_interval, rolling_mean, rolling_std, month_sin, month_cos]], dtype=float)

            if self.model_type in ['ridge', 'lasso']:
                X_future = self.scaler.transform(X_future)

            pred_interval = float(self.model.predict(X_future)[0])
            pred_interval = float(np.clip(pred_interval, self.min_interval, self.max_interval))

            last_date = last_date + timedelta(days=int(round(pred_interval)))
            history.append(pred_interval)

            if last_date > today:
                future_predictions.append(last_date)

        return future_predictions


# 便捷的工厂函数
def create_linear_predictor():
    """创建线性回归预测器"""
    return LinearPredictor('linear')

def create_ridge_predictor():
    """创建Ridge回归预测器"""
    return LinearPredictor('ridge')

def create_lasso_predictor():
    """创建Lasso回归预测器"""
    return LinearPredictor('lasso') 
