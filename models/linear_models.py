"""
线性模型预测器
包括线性回归、Ridge回归、Lasso回归等
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import List
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler

from .base import BasePredictor
from .utils import compute_interval_bounds


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
                rolling_mean = recent.mean()
                rolling_std = recent.std(ddof=0) if len(recent) > 1 else 0.0
            else:
                # 冷启动样本没有历史，特征全部回退到全局统计量；
                # 不能让 recent 含当前值，否则 rolling_mean 恰等于目标值（泄漏）
                prev_interval = overall_mean
                rolling_mean = overall_mean
                rolling_std = 0.0

            month = df['month'].iloc[idx]
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)

            X.append([
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
        self.min_interval, self.max_interval = compute_interval_bounds(values, 0.05, 0.95)

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

        intervals = df['interval_days'].dropna()

        if intervals.empty:
            raise ValueError("interval_days 为空，无法进行预测")

        history = intervals.values.astype(float).tolist()

        def next_interval(step, current_date):
            prev_interval = history[-1] if history else self.base_interval
            recent = history[-3:] if history else [self.base_interval]
            rolling_mean = np.mean(recent)
            rolling_std = np.std(recent, ddof=0) if len(recent) > 1 else 0.0
            month = current_date.month
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)

            X_future = np.array([[prev_interval, rolling_mean, rolling_std, month_sin, month_cos]], dtype=float)

            if self.model_type in ['ridge', 'lasso']:
                X_future = self.scaler.transform(X_future)

            pred_interval = float(self.model.predict(X_future)[0])
            pred_interval = float(np.clip(pred_interval, self.min_interval, self.max_interval))
            history.append(pred_interval)
            return pred_interval

        return self.roll_future_dates(df['date'].iloc[-1], today, next_interval, n_predictions)


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
