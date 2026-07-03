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
from .utils import compute_interval_bounds


class TrendAnalysisPredictor(BasePredictor):
    """趋势分析预测器"""
    
    def __init__(self):
        super().__init__("Trend Analysis")
        self.slope = None
        self.intercept = None
    
    def fit(self, df: pd.DataFrame) -> None:
        """拟合趋势线"""
        if len(df) < 2:
            raise ValueError("需要至少2个数据点进行趋势分析")

        x = np.arange(len(df))
        y = df['days_since_start'].values

        # 线性回归拟合趋势
        # 使用对近期数据权重更高的加权线性回归，贴合加速趋势
        weights = np.linspace(0.5, 1.0, len(df))
        coeffs = np.polyfit(x, y, 1, w=weights)
        self.slope = float(coeffs[0])
        self.intercept = float(coeffs[1])
        # 在间隔天数（一阶差分）上评估，与其它预测器的指标口径保持一致，
        # 否则在累积量 days_since_start 上 R² 天然接近 1，跨模型不可比
        y_pred = self.slope * x + self.intercept
        self.evaluate(np.diff(y), np.diff(y_pred))

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

        # 训练用索引 0..len(df)-1，因此下一次发布对应索引 len(df)；
        # 早于 today 的日期只跳过、不占用 n_predictions 预算
        future_index = len(df)
        for _ in range(self.MAX_ROLL_ITERATIONS):
            if len(future_predictions) >= n_predictions:
                break
            pred_days = self.slope * future_index + self.intercept
            pred_date = start_date + timedelta(days=int(pred_days))
            if pred_date > today:
                future_predictions.append(pred_date)
            future_index += 1

        return future_predictions


class SeasonalDecomposePredictor(BasePredictor):
    """季节性分解预测器"""
    
    def __init__(self):
        super().__init__("Seasonal Decompose")
        self.monthly_effects = {}
        self.trend_slope = 0
        self.base_level = 0
        self.interval_floor = None
        self.interval_cap = None
        self.history_length = 0

    def fit(self, df: pd.DataFrame) -> None:
        """分析季节性模式"""
        # 计算月度效应
        intervals = df['interval_days'].dropna()
        if intervals.empty:
            raise ValueError("interval_days 为空，无法训练季节性分解模型")

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
        
        self.base_level = float(overall_mean)
        self.history_length = len(intervals)

        values = intervals.values.astype(float)
        self.interval_floor, self.interval_cap = compute_interval_bounds(values, 0.05, 0.95)

        # 使用估计的季节与趋势对历史数据回测
        predicted_intervals = []
        for idx, month in enumerate(df_with_intervals['month']):
            seasonal_effect = self.monthly_effects.get(month, 0)
            trend_effect = self.trend_slope * idx
            interval = self.base_level + seasonal_effect + trend_effect
            predicted_intervals.append(float(np.clip(interval, self.interval_floor, self.interval_cap)))

        self.evaluate(values, np.array(predicted_intervals, dtype=float))
        self.is_fitted = True
    
    def predict(self, df: pd.DataFrame, n_predictions: int = 5,
                today: datetime = None) -> List[datetime]:
        """基于季节性分解预测"""
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit()方法")

        if today is None:
            today = datetime.now()

        base_interval = float(np.clip(self.base_level, self.interval_floor, self.interval_cap))

        def next_interval(step, current_date):
            # 估计下一个发布时间所处的月份
            tentative_date = current_date + timedelta(days=int(round(base_interval)))
            seasonal_effect = self.monthly_effects.get(tentative_date.month, 0)
            # 趋势从历史末尾续接，而非从 index 0 重启
            trend_effect = self.trend_slope * (self.history_length + step)
            predicted_interval = self.base_level + seasonal_effect + trend_effect
            return float(np.clip(predicted_interval, self.interval_floor, self.interval_cap))

        return self.roll_future_dates(df['date'].iloc[-1], today, next_interval, n_predictions)


class CyclicalAnalysisPredictor(BasePredictor):
    """周期性分析预测器"""
    
    def __init__(self):
        super().__init__("Cyclical Analysis")
        self.cycle_length = None
        self.cycle_pattern = None
        self.interval_floor = None
        self.interval_cap = None

    def fit(self, df: pd.DataFrame) -> None:
        """检测周期性模式"""
        intervals = df['interval_days'].dropna().values
        if len(intervals) == 0:
            raise ValueError("interval_days 为空，无法训练周期性模型")

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

        values = np.array(self.cycle_pattern, dtype=float)
        floor_candidates = np.concatenate([intervals, values])
        self.interval_floor, self.interval_cap = compute_interval_bounds(floor_candidates, 0.05, 0.95)
        preds = []
        for idx in range(len(intervals)):
            cycle_val = self.cycle_pattern[idx % len(self.cycle_pattern)]
            preds.append(float(np.clip(cycle_val, self.interval_floor, self.interval_cap)))
        self.evaluate(intervals.astype(float), np.array(preds, dtype=float))
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
        
        # 计算周期间的相关性；只奖励正相关（真正重复），
        # 取绝对值会把反相位的段也误判为强周期
        correlations = []
        for i in range(len(cycles)-1):
            corr = np.corrcoef(cycles[i], cycles[i+1])[0, 1]
            if not np.isnan(corr):
                correlations.append(max(corr, 0.0))
        
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

        def next_interval(step, current_date):
            pattern_index = step % len(self.cycle_pattern)
            return float(np.clip(self.cycle_pattern[pattern_index], self.interval_floor, self.interval_cap))

        return self.roll_future_dates(df['date'].iloc[-1], today, next_interval, n_predictions)


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
                mae = predictor.performance_metrics.get('MAE')
                if mae and mae > 0:
                    weight = 1.0 / mae
                else:
                    weight = 0.1
            else:
                weight = 0.1  # 默认权重
            self.weights.append(weight)
        
        # 标准化权重
        total_weight = sum(self.weights)
        if total_weight > 0:
            self.weights = [w / total_weight for w in self.weights]
        else:
            self.weights = [1.0 / len(self.predictors)] * len(self.predictors)
        
        aggregated_mae = 0.0
        aggregated_rmse = 0.0
        weight_sum_mae = 0.0
        weight_sum_rmse = 0.0

        for weight, predictor in zip(self.weights, self.predictors):
            metrics = predictor.performance_metrics
            if not metrics:
                continue
            mae = metrics.get('MAE')
            rmse = metrics.get('RMSE')
            if mae is not None:
                aggregated_mae += weight * mae
                weight_sum_mae += weight
            if rmse is not None:
                aggregated_rmse += weight * rmse
                weight_sum_rmse += weight

        perf = {}
        if weight_sum_mae > 0:
            # 除以累计权重：部分子预测器缺失指标时重新归一化，避免加权和被低估
            perf['MAE'] = aggregated_mae / weight_sum_mae
        if weight_sum_rmse > 0:
            perf['RMSE'] = aggregated_rmse / weight_sum_rmse
        self.performance_metrics = perf

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
            avg_interval = float(intervals.mean())
            return self.roll_future_dates(
                df['date'].iloc[-1], today,
                lambda step, current_date: avg_interval,
                n_predictions,
            )

        # 加权平均预测结果（各子预测器均返回恰好 n_predictions 个未来日期，
        # 因此 pred_idx 即"第 k 个未来发布"，跨预测器按序号对齐）
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
