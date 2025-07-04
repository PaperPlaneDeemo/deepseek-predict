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
        """训练线性模型"""
        X, y = self._create_features(df)
        
        if self.model_type in ['ridge', 'lasso']:
            X = self.scaler.fit_transform(X)
        
        self.model.fit(X, y)
        
        # 评估性能
        if self.model_type in ['ridge', 'lasso']:
            y_pred = self.model.predict(X)
        else:
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
        last_index = len(df)
        
        for i in range(1, n_predictions + 1):
            # 基于历史模式预测特征
            future_month = ((today.month + i * 2 - 1) % 12) + 1
            future_quarter = ((future_month - 1) // 3) + 1
            future_year = today.year + ((today.month + i * 2 - 1) // 12)
            
            X_future = np.array([[
                last_index + i - 1,
                future_month,
                future_quarter,
                future_year,
                0,  # 假设不是Coder版本
                0,  # 假设不是V2
                1 if i <= 3 else 0,  # 前几个可能是V3
                1 if i > 3 else 0   # 后几个可能是R1或新系列
            ]])
            
            if self.model_type in ['ridge', 'lasso']:
                X_future = self.scaler.transform(X_future)
            
            pred_days = self.model.predict(X_future)[0]
            pred_date = df['date'].iloc[0] + timedelta(days=int(pred_days))
            
            if pred_date > today:
                future_predictions.append(pred_date)
        
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