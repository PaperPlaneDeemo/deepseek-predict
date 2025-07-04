"""
基于间隔的预测器
包括平均间隔、中位数间隔、移动平均等方法
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List

from .base import BasePredictor


class IntervalPredictor(BasePredictor):
    """基于间隔的预测器"""
    
    def __init__(self, strategy='mean'):
        """
        初始化间隔预测器
        
        Args:
            strategy: 预测策略，可选 'mean', 'median', 'recent', 'exponential'
        """
        self.strategy = strategy
        strategy_names = {
            'mean': 'Mean Interval',
            'median': 'Median Interval', 
            'recent': 'Recent 3 Mean',
            'exponential': 'Exponential Smoothing'
        }
        super().__init__(strategy_names.get(strategy, f"Interval {strategy}"))
        self.interval_value = None
    
    def fit(self, df: pd.DataFrame) -> None:
        """训练间隔模型"""
        intervals = df['interval_days'].dropna()
        
        if self.strategy == 'mean':
            self.interval_value = intervals.mean()
        elif self.strategy == 'median':
            self.interval_value = intervals.median()
        elif self.strategy == 'recent':
            self.interval_value = intervals.tail(3).mean()
        elif self.strategy == 'exponential':
            self.interval_value = self._exponential_smoothing(intervals)
        else:
            raise ValueError(f"不支持的策略: {self.strategy}")
        
        self.is_fitted = True
    
    def _exponential_smoothing(self, intervals, alpha=0.3):
        """指数平滑"""
        smoothed = intervals.iloc[0]
        for value in intervals.iloc[1:]:
            smoothed = alpha * value + (1 - alpha) * smoothed
        return smoothed
    
    def predict(self, df: pd.DataFrame, n_predictions: int = 5, 
                today: datetime = None) -> List[datetime]:
        """生成基于间隔的预测"""
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit()方法")
        
        if today is None:
            today = datetime.now()
        
        future_predictions = []
        last_date = df['date'].iloc[-1]
        
        for i in range(n_predictions):
            last_date = last_date + timedelta(days=int(max(30, self.interval_value)))
            if last_date > today:
                future_predictions.append(last_date)
        
        return future_predictions


class AdaptiveIntervalPredictor(BasePredictor):
    """自适应间隔预测器 - 根据历史趋势调整间隔"""
    
    def __init__(self):
        super().__init__("Adaptive Interval")
        self.trend_factor = 1.0
        self.base_interval = None
    
    def fit(self, df: pd.DataFrame) -> None:
        """训练自适应间隔模型"""
        intervals = df['interval_days'].dropna()
        
        # 基础间隔
        self.base_interval = intervals.mean()
        
        # 计算趋势因子
        if len(intervals) >= 6:
            recent_intervals = intervals.tail(3).mean()
            early_intervals = intervals.head(3).mean()
            self.trend_factor = recent_intervals / early_intervals if early_intervals > 0 else 1.0
        
        self.is_fitted = True
    
    def predict(self, df: pd.DataFrame, n_predictions: int = 5, 
                today: datetime = None) -> List[datetime]:
        """生成自适应预测"""
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit()方法")
        
        if today is None:
            today = datetime.now()
        
        future_predictions = []
        last_date = df['date'].iloc[-1]
        
        for i in range(n_predictions):
            # 随时间调整间隔
            adjusted_interval = self.base_interval * (self.trend_factor ** (i + 1))
            last_date = last_date + timedelta(days=int(max(30, adjusted_interval)))
            if last_date > today:
                future_predictions.append(last_date)
        
        return future_predictions


class WeightedIntervalPredictor(BasePredictor):
    """加权间隔预测器 - 给近期数据更高权重"""
    
    def __init__(self, decay_rate=0.8):
        super().__init__("Weighted Interval")
        self.decay_rate = decay_rate
        self.weighted_interval = None
    
    def fit(self, df: pd.DataFrame) -> None:
        """训练加权间隔模型"""
        intervals = df['interval_days'].dropna()
        
        # 计算加权平均间隔
        weights = np.array([self.decay_rate ** i for i in range(len(intervals))])
        weights = weights[::-1]  # 反转，让最新的数据权重最高
        
        self.weighted_interval = np.average(intervals, weights=weights)
        self.is_fitted = True
    
    def predict(self, df: pd.DataFrame, n_predictions: int = 5, 
                today: datetime = None) -> List[datetime]:
        """生成加权预测"""
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit()方法")
        
        if today is None:
            today = datetime.now()
        
        future_predictions = []
        last_date = df['date'].iloc[-1]
        
        for i in range(n_predictions):
            last_date = last_date + timedelta(days=int(max(30, self.weighted_interval)))
            if last_date > today:
                future_predictions.append(last_date)
        
        return future_predictions


# 便捷的工厂函数
def create_mean_interval_predictor():
    """创建平均间隔预测器"""
    return IntervalPredictor('mean')

def create_median_interval_predictor():
    """创建中位数间隔预测器"""
    return IntervalPredictor('median')

def create_recent_interval_predictor():
    """创建近期间隔预测器"""
    return IntervalPredictor('recent')

def create_exponential_interval_predictor():
    """创建指数平滑间隔预测器"""
    return IntervalPredictor('exponential')

def create_adaptive_interval_predictor():
    """创建自适应间隔预测器"""
    return AdaptiveIntervalPredictor()

def create_weighted_interval_predictor():
    """创建加权间隔预测器"""
    return WeightedIntervalPredictor() 