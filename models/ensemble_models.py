"""
集成学习预测器
包括随机森林、梯度提升、XGBoost等集成方法
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

from .base import BasePredictor

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class RandomForestPredictor(BasePredictor):
    """随机森林预测器"""
    
    def __init__(self, n_estimators=100, random_state=42):
        super().__init__("Random Forest")
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    
    def _create_features(self, df: pd.DataFrame):
        """创建特征"""
        X = []
        y = []
        
        for i in range(1, len(df)):
            X.append([
                i,  # 时间序列索引
                df.iloc[i]['month'],
                df.iloc[i]['quarter'], 
                df.iloc[i]['year'],
                df.iloc[i]['is_coder'],
                df.iloc[i]['is_v2'],
                df.iloc[i]['is_v3'],
                df.iloc[i]['is_r1']
            ])
            y.append(df.iloc[i]['days_since_start'])
            
        return np.array(X), np.array(y)
    
    def fit(self, df: pd.DataFrame) -> None:
        """训练随机森林模型"""
        X, y = self._create_features(df)
        self.model.fit(X, y)
        
        # 评估性能
        y_pred = self.model.predict(X)
        self.evaluate(y, y_pred)
        self.is_fitted = True
    
    def predict(self, df: pd.DataFrame, n_predictions: int = 5, 
                today: datetime = None) -> List[datetime]:
        """生成随机森林预测"""
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit()方法")
        
        if today is None:
            today = datetime.now()
        
        future_predictions = []
        last_index = len(df)
        current_days = (today - df['date'].iloc[0]).days
        
        # 计算历史平均发布间隔
        intervals = df['days_since_start'].diff().dropna()
        avg_interval = intervals.mean()
        
        for i in range(1, n_predictions + 1):
            # 基于历史数据创建未来特征
            estimated_future_days = current_days + avg_interval * i
            future_date_est = df['date'].iloc[0] + timedelta(days=estimated_future_days)
            
            X_future = np.array([[
                last_index + i - 1,
                future_date_est.month,
                future_date_est.quarter,
                future_date_est.year,
                0, 0, 1 if i <= 3 else 0, 1 if i > 3 else 0
            ]])
            
            pred_days = self.model.predict(X_future)[0]
            pred_date = df['date'].iloc[0] + timedelta(days=int(pred_days))
            
            # 只有当预测明显在过去时才使用约束
            if pred_date <= today:
                min_future_days = current_days + avg_interval * 0.2
                pred_date = df['date'].iloc[0] + timedelta(days=int(min_future_days))
            
            if pred_date > today:
                future_predictions.append(pred_date)
        
        return future_predictions


class GradientBoostingPredictor(BasePredictor):
    """梯度提升预测器"""
    
    def __init__(self, n_estimators=100, random_state=42):
        super().__init__("Gradient Boosting")
        self.model = GradientBoostingRegressor(n_estimators=n_estimators, random_state=random_state)
    
    def _create_features(self, df: pd.DataFrame):
        """创建特征"""
        X = []
        y = []
        
        for i in range(1, len(df)):
            X.append([
                i,  # 时间序列索引
                df.iloc[i]['month'],
                df.iloc[i]['quarter'], 
                df.iloc[i]['year'],
                df.iloc[i]['is_coder'],
                df.iloc[i]['is_v2'],
                df.iloc[i]['is_v3'],
                df.iloc[i]['is_r1']
            ])
            y.append(df.iloc[i]['days_since_start'])
            
        return np.array(X), np.array(y)
    
    def fit(self, df: pd.DataFrame) -> None:
        """训练梯度提升模型"""
        X, y = self._create_features(df)
        self.model.fit(X, y)
        
        # 评估性能
        y_pred = self.model.predict(X)
        self.evaluate(y, y_pred)
        self.is_fitted = True
    
    def predict(self, df: pd.DataFrame, n_predictions: int = 5, 
                today: datetime = None) -> List[datetime]:
        """生成梯度提升预测"""
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit()方法")
        
        if today is None:
            today = datetime.now()
        
        future_predictions = []
        last_index = len(df)
        current_days = (today - df['date'].iloc[0]).days
        
        # 计算历史平均发布间隔
        intervals = df['days_since_start'].diff().dropna()
        avg_interval = intervals.mean()
        
        for i in range(1, n_predictions + 1):
            # 基于历史数据创建未来特征
            estimated_future_days = current_days + avg_interval * i
            future_date_est = df['date'].iloc[0] + timedelta(days=estimated_future_days)
            
            X_future = np.array([[
                last_index + i - 1,
                future_date_est.month,
                future_date_est.quarter,
                future_date_est.year,
                0, 0, 1 if i <= 3 else 0, 1 if i > 3 else 0
            ]])
            
            pred_days = self.model.predict(X_future)[0]
            pred_date = df['date'].iloc[0] + timedelta(days=int(pred_days))
            
            # 只有当预测明显在过去时才使用约束
            if pred_date <= today:
                min_future_days = current_days + avg_interval * 0.3
                pred_date = df['date'].iloc[0] + timedelta(days=int(min_future_days))
            
            if pred_date > today:
                future_predictions.append(pred_date)
        
        return future_predictions


class XGBoostPredictor(BasePredictor):
    """XGBoost预测器"""
    
    def __init__(self, n_estimators=100, random_state=42):
        super().__init__("XGBoost")
        if XGBOOST_AVAILABLE:
            self.model = xgb.XGBRegressor(n_estimators=n_estimators, random_state=random_state)
        else:
            # 如果XGBoost不可用，使用梯度提升作为替代
            self.model = GradientBoostingRegressor(n_estimators=n_estimators, random_state=random_state)
            self.name = "XGBoost (Fallback to GBM)"
    
    def _create_features(self, df: pd.DataFrame):
        """创建特征"""
        X = []
        y = []
        
        for i in range(1, len(df)):
            X.append([
                i,  # 时间序列索引
                df.iloc[i]['month'],
                df.iloc[i]['quarter'], 
                df.iloc[i]['year'],
                df.iloc[i]['is_coder'],
                df.iloc[i]['is_v2'],
                df.iloc[i]['is_v3'],
                df.iloc[i]['is_r1']
            ])
            y.append(df.iloc[i]['days_since_start'])
            
        return np.array(X), np.array(y)
    
    def fit(self, df: pd.DataFrame) -> None:
        """训练XGBoost模型"""
        X, y = self._create_features(df)
        self.model.fit(X, y)
        
        # 评估性能
        y_pred = self.model.predict(X)
        self.evaluate(y, y_pred)
        self.is_fitted = True
    
    def predict(self, df: pd.DataFrame, n_predictions: int = 5, 
                today: datetime = None) -> List[datetime]:
        """生成XGBoost预测"""
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit()方法")
        
        if today is None:
            today = datetime.now()
        
        future_predictions = []
        last_index = len(df)
        current_days = (today - df['date'].iloc[0]).days
        
        # 计算历史平均发布间隔
        intervals = df['days_since_start'].diff().dropna()
        avg_interval = intervals.mean()
        
        for i in range(1, n_predictions + 1):
            # 基于历史数据创建未来特征
            estimated_future_days = current_days + avg_interval * i
            future_date_est = df['date'].iloc[0] + timedelta(days=estimated_future_days)
            
            X_future = np.array([[
                last_index + i - 1,
                future_date_est.month,
                future_date_est.quarter,
                future_date_est.year,
                0, 0, 1 if i <= 3 else 0, 1 if i > 3 else 0
            ]])
            
            pred_days = self.model.predict(X_future)[0]
            pred_date = df['date'].iloc[0] + timedelta(days=int(pred_days))
            
            # 只有当预测明显在过去时才使用约束
            if pred_date <= today:
                min_future_days = current_days + avg_interval * 0.3
                pred_date = df['date'].iloc[0] + timedelta(days=int(min_future_days))
            
            if pred_date > today:
                future_predictions.append(pred_date)
        
        return future_predictions


class SVRPredictor(BasePredictor):
    """支持向量回归预测器"""
    
    def __init__(self, kernel='rbf'):
        super().__init__("SVR")
        self.model = SVR(kernel=kernel)
        self.scaler = StandardScaler()
    
    def _create_features(self, df: pd.DataFrame):
        """创建特征"""
        X = []
        y = []
        
        for i in range(1, len(df)):
            X.append([
                i,  # 时间序列索引
                df.iloc[i]['month'],
                df.iloc[i]['quarter'], 
                df.iloc[i]['year'],
                df.iloc[i]['is_coder'],
                df.iloc[i]['is_v2'],
                df.iloc[i]['is_v3'],
                df.iloc[i]['is_r1']
            ])
            y.append(df.iloc[i]['days_since_start'])
            
        return np.array(X), np.array(y)
    
    def fit(self, df: pd.DataFrame) -> None:
        """训练SVR模型"""
        X, y = self._create_features(df)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
        # 评估性能
        y_pred = self.model.predict(X_scaled)
        self.evaluate(y, y_pred)
        self.is_fitted = True
    
    def predict(self, df: pd.DataFrame, n_predictions: int = 5, 
                today: datetime = None) -> List[datetime]:
        """生成SVR预测"""
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit()方法")
        
        if today is None:
            today = datetime.now()
        
        future_predictions = []
        last_index = len(df)
        current_days = (today - df['date'].iloc[0]).days
        
        # 计算历史平均发布间隔
        intervals = df['days_since_start'].diff().dropna()
        avg_interval = intervals.mean()
        
        for i in range(1, n_predictions + 1):
            # 基于历史数据创建未来特征
            estimated_future_days = current_days + avg_interval * i
            future_date_est = df['date'].iloc[0] + timedelta(days=estimated_future_days)
            
            X_future = np.array([[
                last_index + i - 1,
                future_date_est.month,
                future_date_est.quarter,
                future_date_est.year,
                0, 0, 1 if i <= 3 else 0, 1 if i > 3 else 0
            ]])
            
            X_future_scaled = self.scaler.transform(X_future)
            pred_days = self.model.predict(X_future_scaled)[0]
            pred_date = df['date'].iloc[0] + timedelta(days=int(pred_days))
            
            # 只有当预测明显在过去时才使用约束  
            if pred_date <= today:
                min_future_days = current_days + avg_interval * 0.5  # SVR使用稍大的约束
                pred_date = df['date'].iloc[0] + timedelta(days=int(min_future_days))
            
            if pred_date > today:
                future_predictions.append(pred_date)
        
        return future_predictions 