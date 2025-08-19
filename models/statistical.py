"""
统计学预测器
包括趋势分析、季节性分解等统计方法
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List
from scipy import stats

from .base import BasePredictor


class TrendAnalysisPredictor(BasePredictor):
    """趋势分析预测器"""
    
    def __init__(self):
        super().__init__("Trend Analysis")
        self.slope = None
        self.intercept = None
        self.r_value = None
    
    def fit(self, df: pd.DataFrame) -> None:
        """拟合趋势线"""
        x = np.arange(len(df))
        y = df['days_since_start'].values
        
        # 线性回归拟合趋势
        self.slope, self.intercept, self.r_value, p_value, std_err = stats.linregress(x, y)
        
        # 计算性能指标
        y_pred = self.slope * x + self.intercept
        self.evaluate(y, y_pred)
        
        self.is_fitted = True
    
    def predict(self, df: pd.DataFrame, n_predictions: int = 5, 
                today: datetime = None) -> List[datetime]:
        """基于趋势线预测"""
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit()方法")
        
        if today is None:
            today = datetime.now()
        
        future_predictions = []
        start_date = df['date'].iloc[0]
        last_index = len(df)
        
        for i in range(1, n_predictions + 1):
            future_index = last_index + i  # 修复：应该是last_index + i，不是last_index + i - 1
            pred_days = self.slope * future_index + self.intercept
            pred_date = start_date + timedelta(days=int(pred_days))
            
            if pred_date > today:
                future_predictions.append(pred_date)
        
        return future_predictions


class SeasonalDecomposePredictor(BasePredictor):
    """季节性分解预测器"""
    
    def __init__(self):
        super().__init__("Seasonal Decompose")
        self.monthly_effects = {}
        self.trend_slope = 0
        self.base_level = 0
    
    def fit(self, df: pd.DataFrame) -> None:
        """分析季节性模式"""
        # 计算月度效应
        intervals = df['interval_days'].dropna()
        df_with_intervals = df.iloc[1:].copy()
        df_with_intervals['interval'] = intervals.values
        
        # 计算每月的平均偏差
        overall_mean = intervals.mean()
        monthly_means = df_with_intervals.groupby('month')['interval'].mean()
        
        for month in range(1, 13):
            if month in monthly_means.index:
                self.monthly_effects[month] = monthly_means[month] - overall_mean
            else:
                self.monthly_effects[month] = 0
        
        # 计算趋势
        x = np.arange(len(intervals))
        if len(intervals) > 1:
            self.trend_slope, intercept, _, _, _ = stats.linregress(x, intervals)
        
        self.base_level = overall_mean
        self.is_fitted = True
    
    def predict(self, df: pd.DataFrame, n_predictions: int = 5, 
                today: datetime = None) -> List[datetime]:
        """基于季节性分解预测"""
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit()方法")
        
        if today is None:
            today = datetime.now()
        
        future_predictions = []
        last_date = df['date'].iloc[-1]
        
        for i in range(n_predictions):
            # 计算预期间隔
            next_month = (last_date + timedelta(days=30)).month
            seasonal_effect = self.monthly_effects.get(next_month, 0)
            trend_effect = self.trend_slope * i
            
            predicted_interval = self.base_level + seasonal_effect + trend_effect
            predicted_interval = max(30, predicted_interval)  # 最小30天
            
            last_date = last_date + timedelta(days=int(predicted_interval))
            if last_date > today:
                future_predictions.append(last_date)
        
        return future_predictions


class CyclicalAnalysisPredictor(BasePredictor):
    """周期性分析预测器"""
    
    def __init__(self):
        super().__init__("Cyclical Analysis")
        self.cycle_length = None
        self.cycle_pattern = None
    
    def fit(self, df: pd.DataFrame) -> None:
        """检测周期性模式"""
        intervals = df['interval_days'].dropna().values
        
        if len(intervals) < 4:
            # 数据太少，使用简单平均
            self.cycle_pattern = [intervals.mean()]
            self.cycle_length = 1
        else:
            # 尝试检测周期
            # 这里使用简化的方法，寻找重复模式
            best_cycle_length = 1
            best_score = 0
            
            for cycle_len in range(2, min(6, len(intervals) // 2)):
                score = self._evaluate_cycle(intervals, cycle_len)
                if score > best_score:
                    best_score = score
                    best_cycle_length = cycle_len
            
            self.cycle_length = best_cycle_length
            self.cycle_pattern = self._extract_cycle_pattern(intervals, best_cycle_length)
        
        self.is_fitted = True
    
    def _evaluate_cycle(self, intervals, cycle_length):
        """评估周期性模式的质量"""
        if len(intervals) < cycle_length * 2:
            return 0
        
        cycles = []
        for i in range(0, len(intervals) - cycle_length + 1, cycle_length):
            if i + cycle_length <= len(intervals):
                cycles.append(intervals[i:i+cycle_length])
        
        if len(cycles) < 2:
            return 0
        
        # 计算周期间的相关性
        correlations = []
        for i in range(len(cycles)-1):
            corr = np.corrcoef(cycles[i], cycles[i+1])[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0
    
    def _extract_cycle_pattern(self, intervals, cycle_length):
        """提取周期模式"""
        patterns = []
        for i in range(0, len(intervals), cycle_length):
            if i + cycle_length <= len(intervals):
                patterns.append(intervals[i:i+cycle_length])
        
        if patterns:
            return np.mean(patterns, axis=0)
        else:
            return [intervals.mean()]
    
    def predict(self, df: pd.DataFrame, n_predictions: int = 5, 
                today: datetime = None) -> List[datetime]:
        """基于周期性模式预测"""
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit()方法")
        
        if today is None:
            today = datetime.now()
        
        future_predictions = []
        last_date = df['date'].iloc[-1]
        
        for i in range(n_predictions):
            pattern_index = i % len(self.cycle_pattern)
            predicted_interval = self.cycle_pattern[pattern_index]
            predicted_interval = max(30, predicted_interval)
            
            last_date = last_date + timedelta(days=int(predicted_interval))
            if last_date > today:
                future_predictions.append(last_date)
        
        return future_predictions


class StatisticalPredictor(BasePredictor):
    """综合统计预测器 - 结合多种统计方法"""
    
    def __init__(self):
        super().__init__("Statistical Ensemble")
        self.predictors = [
            TrendAnalysisPredictor(),
            SeasonalDecomposePredictor(),
            CyclicalAnalysisPredictor()
        ]
        self.weights = None
    
    def fit(self, df: pd.DataFrame) -> None:
        """训练所有统计模型"""
        # 训练各个子预测器
        for predictor in self.predictors:
            try:
                predictor.fit(df)
            except Exception as e:
                print(f"统计模型 {predictor.name} 训练失败: {e}")
        
        # 根据性能设置权重
        self.weights = []
        for predictor in self.predictors:
            if predictor.is_fitted and predictor.performance_metrics:
                r2 = predictor.performance_metrics.get('R2', 0)
                weight = max(0, r2)  # 使用R²作为权重
            else:
                weight = 0.1  # 默认权重
            self.weights.append(weight)
        
        # 标准化权重
        total_weight = sum(self.weights)
        if total_weight > 0:
            self.weights = [w / total_weight for w in self.weights]
        else:
            self.weights = [1.0 / len(self.predictors)] * len(self.predictors)
        
        self.is_fitted = True
    
    def predict(self, df: pd.DataFrame, n_predictions: int = 5, 
                today: datetime = None) -> List[datetime]:
        """生成综合统计预测"""
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit()方法")
        
        if today is None:
            today = datetime.now()
        
        # 收集各个预测器的结果
        all_predictions = []
        for i, predictor in enumerate(self.predictors):
            if predictor.is_fitted:
                try:
                    preds = predictor.predict(df, n_predictions, today)
                    all_predictions.append((preds, self.weights[i]))
                except Exception as e:
                    print(f"预测器 {predictor.name} 预测失败: {e}")
        
        if not all_predictions:
            # 如果所有预测器都失败，使用简单方法
            intervals = df['interval_days'].dropna()
            avg_interval = intervals.mean()
            
            future_predictions = []
            last_date = df['date'].iloc[-1]
            for i in range(n_predictions):
                last_date = last_date + timedelta(days=int(avg_interval))
                if last_date > today:
                    future_predictions.append(last_date)
            return future_predictions
        
        # 加权平均预测结果
        ensemble_predictions = []
        for pred_idx in range(n_predictions):
            weighted_days = []
            total_weight = 0
            
            for preds, weight in all_predictions:
                if pred_idx < len(preds):
                    days_from_start = (preds[pred_idx] - df['date'].iloc[0]).days
                    weighted_days.append(days_from_start * weight)
                    total_weight += weight
            
            if weighted_days and total_weight > 0:
                avg_days = sum(weighted_days) / total_weight
                pred_date = df['date'].iloc[0] + timedelta(days=int(avg_days))
                if pred_date > today:
                    ensemble_predictions.append(pred_date)
        
        return ensemble_predictions 