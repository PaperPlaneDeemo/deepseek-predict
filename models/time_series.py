"""
时间序列预测器
包括ARIMA、指数平滑等时间序列分析方法
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List
from statsmodels.tsa.arima.model import ARIMA

from .base import BasePredictor


class ARIMAPredictor(BasePredictor):
    """ARIMA时间序列预测器"""
    
    def __init__(self, order=(1, 1, 1)):
        super().__init__("ARIMA")
        self.order = order
        self.fitted_model = None
        self.interval_floor = None
        self.interval_cap = None
    
    def fit(self, df: pd.DataFrame) -> None:
        """训练ARIMA模型"""
        # 使用发布间隔进行ARIMA预测
        intervals = df['interval_days'].dropna()

        if intervals.empty:
            raise ValueError("interval_days 为空，无法训练ARIMA模型")

        values = intervals.values.astype(float)
        if len(values) > 1:
            self.interval_floor = max(1.0, float(np.percentile(values, 5)))
            self.interval_cap = float(np.percentile(values, 95))
        else:
            value = float(values[0])
            self.interval_floor = max(1.0, value * 0.8)
            self.interval_cap = value * 1.2
        
        try:
            # 自动选择最佳参数 - 扩展搜索范围
            best_aic = float('inf')
            best_order = None
            best_bic = float('inf')
            
            # 更全面的参数搜索
            for p in range(0, 5):  # 扩展AR阶数
                for d in range(0, 3):  # 扩展差分阶数
                    for q in range(0, 5):  # 扩展MA阶数
                        try:
                            model = ARIMA(intervals, order=(p, d, q))
                            fitted = model.fit()
                            
                            # 使用AIC和BIC综合评估
                            current_aic = fitted.aic
                            current_bic = fitted.bic
                            
                            # 优先选择更简单的模型 (BIC惩罚复杂度更强)
                            score = current_aic + 0.3 * current_bic
                            
                            if score < best_aic + 0.3 * best_bic:
                                best_aic = current_aic
                                best_bic = current_bic
                                best_order = (p, d, q)
                        except Exception as e:
                            # 记录错误但不中断
                            continue
            
            if best_order:
                self.order = best_order
                self.model = ARIMA(intervals, order=self.order)
                self.fitted_model = self.model.fit()
                
                # 评估性能
                forecast_fit = self.fitted_model.fittedvalues
                actual = intervals[1:] if len(intervals) > 1 else intervals
                
                if len(forecast_fit) == len(actual):
                    self.performance_metrics = {
                        'MAE': np.mean(np.abs(actual - forecast_fit)),
                        'RMSE': np.sqrt(np.mean((actual - forecast_fit) ** 2)),
                        'AIC': self.fitted_model.aic
                    }
                
                self.is_fitted = True
            else:
                # 尝试使用默认参数 (1,1,1)
                try:
                    self.order = (1, 1, 1)
                    self.model = ARIMA(intervals, order=self.order)
                    self.fitted_model = self.model.fit()
                    print("使用默认ARIMA参数 (1,1,1)")
                except:
                    raise ValueError("无法找到合适的ARIMA参数")
                
        except Exception as e:
            print(f"ARIMA模型训练失败: {e}")
            # 使用更智能的替代方法
            self.name = "ARIMA (Fallback)"
            # 使用加权移动平均 (近期数据权重更高)
            if len(intervals) >= 3:
                weights = [0.1, 0.3, 0.6]  # 最近的数据权重最高
                recent = intervals.tail(3)
                self.recent_avg = float(np.average(recent, weights=weights[:len(recent)]))
            else:
                self.recent_avg = float(intervals.mean()) if len(intervals) > 0 else 60.0
            self.recent_avg = float(np.clip(self.recent_avg, self.interval_floor, self.interval_cap))
            self.is_fitted = True
    
    def predict(self, df: pd.DataFrame, n_predictions: int = 5, 
                today: datetime = None) -> List[datetime]:
        """生成ARIMA预测"""
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit()方法")
        
        if today is None:
            today = datetime.now()
        
        future_predictions = []
        last_date = df['date'].iloc[-1]
        
        if self.fitted_model:
            # 使用ARIMA模型预测
            forecast_intervals = self.fitted_model.forecast(steps=n_predictions)
            
            for interval in forecast_intervals:
                clipped = float(np.clip(float(interval), self.interval_floor, self.interval_cap))
                last_date = last_date + timedelta(days=int(round(clipped)))
                if last_date > today:
                    future_predictions.append(last_date)
        else:
            # 使用移动平均替代
            for i in range(n_predictions):
                last_date = last_date + timedelta(days=int(round(self.recent_avg)))
                if last_date > today:
                    future_predictions.append(last_date)
        
        return future_predictions


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
        if len(values) > 1:
            self.interval_floor = max(1.0, float(np.percentile(values, 5)))
            self.interval_cap = float(np.percentile(values, 95))
        else:
            value = float(values[0])
            self.interval_floor = max(1.0, value * 0.8)
            self.interval_cap = value * 1.2

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
        self.trend_slope = 0.0

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
        if len(values) > 1:
            self.interval_floor = max(1.0, float(np.percentile(values, 5)))
            self.interval_cap = float(np.percentile(values, 95))
        else:
            value = float(values[0])
            self.interval_floor = max(1.0, value * 0.8)
            self.interval_cap = value * 1.2

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
