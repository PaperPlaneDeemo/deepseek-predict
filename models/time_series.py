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
    
    def fit(self, df: pd.DataFrame) -> None:
        """训练ARIMA模型"""
        # 使用发布间隔进行ARIMA预测
        intervals = df['interval_days'].dropna()
        
        try:
            # 自动选择最佳参数
            best_aic = float('inf')
            best_order = None
            
            for p in range(3):
                for d in range(2):
                    for q in range(3):
                        try:
                            model = ARIMA(intervals, order=(p, d, q))
                            fitted = model.fit()
                            if fitted.aic < best_aic:
                                best_aic = fitted.aic
                                best_order = (p, d, q)
                        except:
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
                raise ValueError("无法找到合适的ARIMA参数")
                
        except Exception as e:
            print(f"ARIMA模型训练失败: {e}")
            # 使用简单的移动平均作为替代
            self.name = "ARIMA (Fallback)"
            self.recent_avg = intervals.tail(3).mean()
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
                last_date = last_date + timedelta(days=int(max(30, interval)))
                if last_date > today:
                    future_predictions.append(last_date)
        else:
            # 使用移动平均替代
            for i in range(n_predictions):
                last_date = last_date + timedelta(days=int(self.recent_avg))
                if last_date > today:
                    future_predictions.append(last_date)
        
        return future_predictions


class ExponentialSmoothingPredictor(BasePredictor):
    """指数平滑预测器"""
    
    def __init__(self, alpha=0.3):
        super().__init__("Exponential Smoothing")
        self.alpha = alpha
        self.smoothed_interval = None
    
    def fit(self, df: pd.DataFrame) -> None:
        """训练指数平滑模型"""
        intervals = df['interval_days'].dropna()
        
        # 指数平滑
        self.smoothed_interval = intervals.iloc[0]
        for value in intervals.iloc[1:]:
            self.smoothed_interval = self.alpha * value + (1 - self.alpha) * self.smoothed_interval
        
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
            last_date = last_date + timedelta(days=int(max(30, self.smoothed_interval)))
            if last_date > today:
                future_predictions.append(last_date)
        
        return future_predictions


class SeasonalPredictor(BasePredictor):
    """季节性模式预测器"""
    
    def __init__(self):
        super().__init__("Seasonal Pattern")
        self.monthly_patterns = {}
        self.default_interval = None
    
    def fit(self, df: pd.DataFrame) -> None:
        """学习季节性模式"""
        intervals = df['interval_days'].dropna()
        
        # 分析每个月的发布模式
        df_with_interval = df.iloc[1:].copy()
        df_with_interval['interval'] = intervals.values
        
        self.monthly_patterns = df_with_interval.groupby('month')['interval'].mean().to_dict()
        self.default_interval = intervals.mean()
        
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
            # 获取下个月的预期间隔
            next_month = (last_date + timedelta(days=30)).month
            interval = self.monthly_patterns.get(next_month, self.default_interval)
            
            # 根据季节调整间隔
            if next_month in [12, 1, 2]:  # 冬季，可能有更多发布
                interval *= 0.8
            elif next_month in [6, 7, 8]:  # 夏季
                interval *= 1.2
            
            last_date = last_date + timedelta(days=int(max(30, interval)))
            if last_date > today:
                future_predictions.append(last_date)
        
        return future_predictions 