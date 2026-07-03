"""
基础预测器抽象类
定义所有预测方法的统一接口
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Callable, List, Dict, Any
import pandas as pd


class BasePredictor(ABC):
    """预测器基类"""

    # 防止间隔退化时滚动循环失控
    MAX_ROLL_ITERATIONS = 1000

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.performance_metrics = {}
        self.is_fitted = False

    @staticmethod
    def roll_future_dates(
        last_date: datetime,
        today: datetime,
        next_interval: Callable[[int, datetime], float],
        n_predictions: int,
    ) -> List[datetime]:
        """从最后一次发布日期向前滚动，收集 today 之后的 n_predictions 个日期。

        追赶 today 的步数不占用 n_predictions 预算；next_interval(step, current_date)
        返回第 step 步使用的间隔天数（可依赖当前滚动到的日期，如季节性查表）。
        """
        future_dates = []
        for step in range(BasePredictor.MAX_ROLL_ITERATIONS):
            if len(future_dates) >= n_predictions:
                break
            interval = max(1, int(round(float(next_interval(step, last_date)))))
            last_date = last_date + timedelta(days=interval)
            if last_date > today:
                future_dates.append(last_date)
        return future_dates
    
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
            # 用 NaN 表示评估失败；置 0 会让失败模型冒充"零误差"排到最前
            self.performance_metrics = {
                'MAE': float('nan'),
                'RMSE': float('nan'),
                'R2': float('nan'),
            }
        
        return self.performance_metrics
    
    def get_info(self) -> Dict[str, Any]:
        """获取预测器信息"""
        return {
            'name': self.name,
            'is_fitted': self.is_fitted,
            'performance': self.performance_metrics
        } 