"""
基础预测器抽象类
定义所有预测方法的统一接口
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd


class BasePredictor(ABC):
    """预测器基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.performance_metrics = {}
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
        """训练模型"""
        pass
    
    @abstractmethod
    def predict(self, df: pd.DataFrame, n_predictions: int = 5, 
                today: datetime = None) -> List[datetime]:
        """生成预测"""
        pass
    
    def evaluate(self, y_true, y_pred) -> Dict[str, float]:
        """评估模型性能"""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        import numpy as np
        
        try:
            self.performance_metrics = {
                'MAE': mean_absolute_error(y_true, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
                'R2': r2_score(y_true, y_pred)
            }
        except Exception as e:
            print(f"性能评估失败 {self.name}: {e}")
            self.performance_metrics = {'MAE': 0, 'RMSE': 0, 'R2': 0}
        
        return self.performance_metrics
    
    def get_info(self) -> Dict[str, Any]:
        """获取预测器信息"""
        return {
            'name': self.name,
            'is_fitted': self.is_fitted,
            'performance': self.performance_metrics
        } 