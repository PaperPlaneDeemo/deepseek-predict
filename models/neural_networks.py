"""
神经网络预测器
包括多层感知机等深度学习方法
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .base import BasePredictor


class MLPPredictor(BasePredictor):
    """多层感知机预测器"""
    
    def __init__(self, hidden_layer_sizes=(100, 50), random_state=42):
        super().__init__("MLP Neural Network")
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            random_state=random_state,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=10
        )
        self.scaler = StandardScaler()
    
    def _create_features(self, df: pd.DataFrame):
        """创建神经网络特征"""
        X = []
        y = []
        
        for i in range(1, len(df)):
            # 扩展特征集
            features = [
                i,  # 时间序列索引
                df.iloc[i]['month'],
                df.iloc[i]['quarter'], 
                df.iloc[i]['year'],
                df.iloc[i]['is_coder'],
                df.iloc[i]['is_v2'],
                df.iloc[i]['is_v3'],
                df.iloc[i]['is_r1'],
                # 添加更多特征
                df.iloc[i]['day_of_year'],
                df.iloc[i]['month'] / 12.0,  # 标准化月份
                np.sin(2 * np.pi * df.iloc[i]['month'] / 12),  # 月份的正弦变换
                np.cos(2 * np.pi * df.iloc[i]['month'] / 12),  # 月份的余弦变换
            ]
            
            # 添加滞后特征
            if i >= 2:
                prev_interval = df.iloc[i-1]['interval_days'] if 'interval_days' in df.columns else 0
                features.append(prev_interval)
            else:
                features.append(0)
            
            X.append(features)
            y.append(df.iloc[i]['days_since_start'])
            
        return np.array(X), np.array(y)
    
    def fit(self, df: pd.DataFrame) -> None:
        """训练神经网络模型"""
        X, y = self._create_features(df)
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        try:
            self.model.fit(X_scaled, y)
            
            # 评估性能
            y_pred = self.model.predict(X_scaled)
            self.evaluate(y, y_pred)
            self.is_fitted = True
            
        except Exception as e:
            print(f"神经网络训练失败: {e}")
            # 降级为简单的线性模型
            from sklearn.linear_model import LinearRegression
            self.model = LinearRegression()
            self.model.fit(X_scaled, y)
            self.name = "MLP (Fallback to Linear)"
            
            y_pred = self.model.predict(X_scaled)
            self.evaluate(y, y_pred)
            self.is_fitted = True
    
    def predict(self, df: pd.DataFrame, n_predictions: int = 5, 
                today: datetime = None) -> List[datetime]:
        """生成神经网络预测"""
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit()方法")
        
        if today is None:
            today = datetime.now()
        
        future_predictions = []
        last_index = len(df)
        
        for i in range(1, n_predictions + 1):
            future_month = ((today.month + i * 2 - 1) % 12) + 1
            future_quarter = ((future_month - 1) // 3) + 1
            future_year = today.year + ((today.month + i * 2 - 1) // 12)
            future_day_of_year = (today + timedelta(days=i*60)).timetuple().tm_yday
            
            # 构建特征向量
            features = [
                last_index + i - 1,
                future_month,
                future_quarter,
                future_year,
                0, 0, 1 if i <= 3 else 0, 1 if i > 3 else 0,
                future_day_of_year,
                future_month / 12.0,
                np.sin(2 * np.pi * future_month / 12),
                np.cos(2 * np.pi * future_month / 12),
                60  # 假设的间隔
            ]
            
            X_future = np.array([features])
            X_future_scaled = self.scaler.transform(X_future)
            
            pred_days = self.model.predict(X_future_scaled)[0]
            pred_date = df['date'].iloc[0] + timedelta(days=int(pred_days))
            
            if pred_date > today:
                future_predictions.append(pred_date)
        
        return future_predictions


class LSTMPredictor(BasePredictor):
    """LSTM预测器 (需要安装tensorflow)"""
    
    def __init__(self, sequence_length=3):
        super().__init__("LSTM")
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = StandardScaler()
        
        # 检查是否可以使用LSTM
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense
            self.tf_available = True
        except ImportError:
            self.tf_available = False
            print("TensorFlow未安装，LSTM预测器将使用简化方法")
    
    def _create_sequences(self, intervals):
        """创建LSTM序列数据"""
        X, y = [], []
        for i in range(self.sequence_length, len(intervals)):
            X.append(intervals[i-self.sequence_length:i])
            y.append(intervals[i])
        return np.array(X), np.array(y)
    
    def fit(self, df: pd.DataFrame) -> None:
        """训练LSTM模型"""
        intervals = df['interval_days'].dropna().values
        
        if not self.tf_available or len(intervals) < self.sequence_length + 2:
            # 降级为简单预测
            self.name = "LSTM (Fallback)"
            self.mean_interval = np.mean(intervals)
            self.is_fitted = True
            return
        
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense
            
            # 标准化数据
            intervals_scaled = self.scaler.fit_transform(intervals.reshape(-1, 1)).flatten()
            
            # 创建序列
            X, y = self._create_sequences(intervals_scaled)
            
            if len(X) < 3:
                # 数据太少，降级
                self.name = "LSTM (Fallback)"
                self.mean_interval = np.mean(intervals)
                self.is_fitted = True
                return
            
            # 构建LSTM模型
            self.model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(self.sequence_length, 1)),
                LSTM(50),
                Dense(25),
                Dense(1)
            ])
            
            self.model.compile(optimizer='adam', loss='mse')
            
            # 训练模型
            X = X.reshape((X.shape[0], X.shape[1], 1))
            self.model.fit(X, y, epochs=50, batch_size=1, verbose=0)
            
            self.is_fitted = True
            
        except Exception as e:
            print(f"LSTM训练失败: {e}")
            self.name = "LSTM (Fallback)"
            self.mean_interval = np.mean(intervals)
            self.is_fitted = True
    
    def predict(self, df: pd.DataFrame, n_predictions: int = 5, 
                today: datetime = None) -> List[datetime]:
        """生成LSTM预测"""
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit()方法")
        
        if today is None:
            today = datetime.now()
        
        future_predictions = []
        last_date = df['date'].iloc[-1]
        
        if self.model is None:
            # 使用简化方法
            for i in range(n_predictions):
                last_date = last_date + timedelta(days=int(max(30, self.mean_interval)))
                if last_date > today:
                    future_predictions.append(last_date)
        else:
            # 使用LSTM预测
            intervals = df['interval_days'].dropna().values
            intervals_scaled = self.scaler.transform(intervals.reshape(-1, 1)).flatten()
            
            # 使用最后的序列进行预测
            last_sequence = intervals_scaled[-self.sequence_length:]
            
            for i in range(n_predictions):
                # 预测下一个间隔
                X_pred = last_sequence.reshape((1, self.sequence_length, 1))
                next_interval_scaled = self.model.predict(X_pred, verbose=0)[0, 0]
                
                # 反标准化
                next_interval = self.scaler.inverse_transform([[next_interval_scaled]])[0, 0]
                
                # 更新序列
                last_sequence = np.append(last_sequence[1:], next_interval_scaled)
                
                # 计算日期
                last_date = last_date + timedelta(days=int(max(30, next_interval)))
                if last_date > today:
                    future_predictions.append(last_date)
        
        return future_predictions 