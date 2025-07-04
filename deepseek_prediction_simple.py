import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

class DeepSeekPredictor:
    def __init__(self):
        # 原始数据
        self.data = {
            'version': [
                'DeepSeek Coder',
                'DeepSeek-LLM', 
                'DeepSeek-V2 (Apr)',
                'DeepSeek-Coder V2 (Jun)',
                'DeepSeek-V2 (Jun)',
                'DeepSeek-Coder V2 (Jul)',
                'DeepSeek-V2.5 (Sep)',
                'DeepSeek-V3',
                'DeepSeek-V2.5 (Dec)',
                'DeepSeek-R1',
                'DeepSeek-V3-0324',
                'DeepSeek-R1-0528'
            ],
            'date': [
                '2023-11-02',
                '2023-11-29',
                '2024-04-28',
                '2024-06-14',
                '2024-06-28',
                '2024-07-24',
                '2024-09-05',
                '2024-12-25',
                '2024-12-10',
                '2025-01-20',
                '2025-03-24',
                '2025-05-28'
            ]
        }
        
        self.today = datetime(2025, 7, 4)
        self.df = None
        self.predictions = {}
        self.model_performances = {}
        
    def prepare_data(self):
        """准备和预处理数据"""
        self.df = pd.DataFrame(self.data)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date').reset_index(drop=True)
        
        # 创建特征
        self.df['days_since_start'] = (self.df['date'] - self.df['date'].iloc[0]).dt.days
        self.df['month'] = self.df['date'].dt.month
        self.df['quarter'] = self.df['date'].dt.quarter
        self.df['year'] = self.df['date'].dt.year
        self.df['day_of_year'] = self.df['date'].dt.dayofyear
        
        # 计算发布间隔
        self.df['interval_days'] = self.df['days_since_start'].diff()
        
        # 创建版本类型特征
        self.df['is_coder'] = self.df['version'].str.contains('Coder').astype(int)
        self.df['is_v2'] = self.df['version'].str.contains('V2').astype(int)
        self.df['is_v3'] = self.df['version'].str.contains('V3').astype(int)
        self.df['is_r1'] = self.df['version'].str.contains('R1').astype(int)
        
        print("数据预处理完成:")
        print(self.df.head())
        
    def create_time_series_features(self):
        """为时间序列模型创建特征"""
        X = []
        y = []
        
        for i in range(1, len(self.df)):
            X.append([
                i,  # 时间序列索引
                self.df.iloc[i]['month'],
                self.df.iloc[i]['quarter'], 
                self.df.iloc[i]['year'],
                self.df.iloc[i]['is_coder'],
                self.df.iloc[i]['is_v2'],
                self.df.iloc[i]['is_v3'],
                self.df.iloc[i]['is_r1']
            ])
            y.append(self.df.iloc[i]['days_since_start'])
            
        return np.array(X), np.array(y)
    
    def linear_regression_forecast(self):
        """线性回归预测"""
        X, y = self.create_time_series_features()
        
        model = LinearRegression()
        model.fit(X, y)
        
        # 预测未来6个模型发布时间
        future_predictions = []
        last_index = len(self.df)
        
        for i in range(1, 7):
            # 基于历史模式预测特征
            future_month = ((self.today.month + i * 2 - 1) % 12) + 1
            future_quarter = ((future_month - 1) // 3) + 1
            future_year = self.today.year + ((self.today.month + i * 2 - 1) // 12)
            
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
            
            pred_days = model.predict(X_future)[0]
            pred_date = self.df['date'].iloc[0] + timedelta(days=int(pred_days))
            
            if pred_date > self.today:
                future_predictions.append(pred_date)
        
        self.predictions['Linear Regression'] = future_predictions[:5]
        
        # 计算模型性能
        y_pred = model.predict(X)
        self.model_performances['Linear Regression'] = {
            'MAE': mean_absolute_error(y, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
            'R2': r2_score(y, y_pred)
        }
        
        return future_predictions[:5]
    
    def arima_forecast(self):
        """ARIMA时间序列预测"""
        # 使用发布间隔进行ARIMA预测
        intervals = self.df['interval_days'].dropna()
        
        # 简单的ARIMA参数
        try:
            model = ARIMA(intervals, order=(1, 1, 1))
            fitted_model = model.fit()
            
            # 预测未来间隔
            forecast_intervals = fitted_model.forecast(steps=5)
            
            # 转换为实际日期
            future_predictions = []
            last_date = self.df['date'].iloc[-1]
            
            for interval in forecast_intervals:
                last_date = last_date + timedelta(days=int(max(30, interval)))  # 最少30天间隔
                if last_date > self.today:
                    future_predictions.append(last_date)
            
            self.predictions['ARIMA'] = future_predictions
            
            # 计算模型性能
            forecast_fit = fitted_model.fittedvalues
            actual = intervals[1:]  # ARIMA从第二个值开始预测
            
            self.model_performances['ARIMA'] = {
                'MAE': mean_absolute_error(actual, forecast_fit),
                'RMSE': np.sqrt(mean_squared_error(actual, forecast_fit)),
                'AIC': fitted_model.aic
            }
            
        except Exception as e:
            print(f"ARIMA模型训练失败: {e}")
            # 使用简单的移动平均作为替代
            recent_avg = intervals.tail(3).mean()
            future_predictions = []
            last_date = self.df['date'].iloc[-1]
            
            for i in range(5):
                last_date = last_date + timedelta(days=int(recent_avg))
                if last_date > self.today:
                    future_predictions.append(last_date)
            
            self.predictions['ARIMA (Fallback)'] = future_predictions
        
        return future_predictions
    
    def ensemble_forecast(self):
        """集成学习预测（简化版）"""
        X, y = self.create_time_series_features()
        
        # 简化的模型集合
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf'),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=1.0)
        }
        
        ensemble_predictions = {}
        
        for name, model in models.items():
            try:
                if name in ['SVR', 'Ridge', 'Lasso']:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    model.fit(X_scaled, y)
                else:
                    model.fit(X, y)
                
                # 预测未来
                future_predictions = []
                last_index = len(self.df)
                
                for i in range(1, 6):
                    future_month = ((self.today.month + i * 2 - 1) % 12) + 1
                    future_quarter = ((future_month - 1) // 3) + 1
                    future_year = self.today.year + ((self.today.month + i * 2 - 1) // 12)
                    
                    X_future = np.array([[
                        last_index + i - 1,
                        future_month,
                        future_quarter,
                        future_year,
                        0, 0, 1 if i <= 3 else 0, 1 if i > 3 else 0
                    ]])
                    
                    if name in ['SVR', 'Ridge', 'Lasso']:
                        X_future_scaled = scaler.transform(X_future)
                        pred_days = model.predict(X_future_scaled)[0]
                    else:
                        pred_days = model.predict(X_future)[0]
                    
                    pred_date = self.df['date'].iloc[0] + timedelta(days=int(pred_days))
                    
                    if pred_date > self.today:
                        future_predictions.append(pred_date)
                
                ensemble_predictions[name] = future_predictions
                
                # 计算性能
                if name in ['SVR', 'Ridge', 'Lasso']:
                    y_pred = model.predict(X_scaled)
                else:
                    y_pred = model.predict(X)
                
                self.model_performances[name] = {
                    'MAE': mean_absolute_error(y, y_pred),
                    'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
                    'R2': r2_score(y, y_pred)
                }
                
            except Exception as e:
                print(f"模型 {name} 训练失败: {e}")
                continue
        
        # 集成预测（取平均）
        if ensemble_predictions:
            ensemble_avg = []
            for i in range(5):
                dates = []
                for model_preds in ensemble_predictions.values():
                    if i < len(model_preds):
                        dates.append(model_preds[i])
                
                if dates:
                    # 转换为天数，计算平均，再转回日期
                    days = [(d - self.df['date'].iloc[0]).days for d in dates]
                    avg_days = np.mean(days)
                    avg_date = self.df['date'].iloc[0] + timedelta(days=int(avg_days))
                    ensemble_avg.append(avg_date)
            
            self.predictions['Ensemble Average'] = ensemble_avg
        
        # 保存各个模型的预测
        for name, preds in ensemble_predictions.items():
            self.predictions[name] = preds
        
        return ensemble_predictions
    
    def interval_based_forecast(self):
        """基于间隔的预测"""
        intervals = self.df['interval_days'].dropna()
        
        # 不同的间隔策略
        strategies = {
            'Mean Interval': intervals.mean(),
            'Median Interval': intervals.median(),
            'Recent 3 Mean': intervals.tail(3).mean(),
            'Exponential Smoothing': self._exponential_smoothing(intervals),
            'Seasonal Pattern': self._seasonal_interval(intervals)
        }
        
        for strategy_name, interval in strategies.items():
            future_predictions = []
            last_date = self.df['date'].iloc[-1]
            
            for i in range(5):
                if strategy_name == 'Seasonal Pattern':
                    # 根据季节调整间隔
                    next_month = (last_date + timedelta(days=int(interval))).month
                    if next_month in [12, 1, 2]:  # 冬季，可能有更多发布
                        interval *= 0.8
                    elif next_month in [6, 7, 8]:  # 夏季
                        interval *= 1.2
                
                last_date = last_date + timedelta(days=int(max(30, interval)))
                if last_date > self.today:
                    future_predictions.append(last_date)
            
            self.predictions[strategy_name] = future_predictions
        
        return strategies
    
    def _exponential_smoothing(self, intervals, alpha=0.3):
        """指数平滑"""
        smoothed = intervals.iloc[0]
        for value in intervals.iloc[1:]:
            smoothed = alpha * value + (1 - alpha) * smoothed
        return smoothed
    
    def _seasonal_interval(self, intervals):
        """季节性间隔模式"""
        # 简单的季节性模式：检查是否有月份相关的模式
        df_with_month = self.df.iloc[1:].copy()  # 跳过第一行（没有间隔）
        df_with_month['interval'] = intervals.values
        
        monthly_avg = df_with_month.groupby('month')['interval'].mean()
        current_month = self.today.month
        
        # 如果当前月份有历史数据，使用它；否则使用总体平均值
        if current_month in monthly_avg.index:
            return monthly_avg[current_month]
        else:
            return intervals.mean()
    
    def run_all_predictions(self):
        """运行所有预测方法"""
        print("开始数据准备...")
        self.prepare_data()
        
        print("运行线性回归预测...")
        self.linear_regression_forecast()
        
        print("运行ARIMA预测...")
        self.arima_forecast()
        
        print("运行集成学习预测...")
        self.ensemble_forecast()
        
        print("运行间隔预测...")
        self.interval_based_forecast()
        
        # 过滤掉早于今天的预测
        filtered_predictions = {}
        for method, dates in self.predictions.items():
            valid_dates = [d for d in dates if d > self.today]
            if valid_dates:
                filtered_predictions[method] = valid_dates[:5]  # 最多保留5个预测
        
        self.predictions = filtered_predictions
        
        print(f"\n预测完成！共 {len(self.predictions)} 种方法，今天是 {self.today.strftime('%Y-%m-%d')}")
        return self.predictions
    
    def create_visualizations(self):
        """创建可视化图表"""
        # 创建综合分析图表
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=['历史发布时间线', '发布间隔分析', '预测结果对比', 
                          '模型性能对比', '发布频率热力图', '预测统计分析'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"colspan": 2}, None],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 历史时间线
        fig.add_trace(
            go.Scatter(
                x=self.df['date'], 
                y=list(range(len(self.df))),
                mode='markers+lines',
                name='历史发布',
                text=self.df['version'],
                hovertemplate='%{text}<br>%{x}<extra></extra>',
                marker=dict(size=10, color='gold'),
                line=dict(color='lightblue', width=3)
            ),
            row=1, col=1
        )
        
        # 发布间隔
        intervals = self.df['interval_days'].dropna()
        fig.add_trace(
            go.Bar(
                x=self.df['version'].iloc[1:],
                y=intervals,
                name='发布间隔(天)',
                text=[f'{int(x)}天' for x in intervals],
                textposition='auto',
                marker_color='lightcoral'
            ),
            row=1, col=2
        )
        
        # 预测结果对比
        colors = px.colors.qualitative.Set3
        for i, (method, dates) in enumerate(self.predictions.items()):
            if dates:  # 确保有有效预测
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=[f'{method}'] * len(dates),
                        mode='markers',
                        marker=dict(size=12, color=colors[i % len(colors)]),
                        name=method,
                        text=[f'预测 {j+1}' for j in range(len(dates))],
                        hovertemplate='%{text}<br>%{x}<extra></extra>'
                    ),
                    row=2, col=1
                )
        
        # 添加今天的线
        fig.add_vline(
            x=self.today, 
            line_dash="dash", 
            line_color="red",
            annotation_text="今天",
            row=2, col=1
        )
        
        # 模型性能对比
        if self.model_performances:
            methods = list(self.model_performances.keys())
            r2_scores = [perf.get('R2', 0) for perf in self.model_performances.values()]
            
            fig.add_trace(
                go.Bar(
                    x=methods,
                    y=r2_scores,
                    name='R² Score',
                    text=[f'{x:.3f}' for x in r2_scores],
                    textposition='auto',
                    marker_color='lightgreen'
                ),
                row=3, col=1
            )
        
        # 预测统计
        all_first_preds = []
        for dates in self.predictions.values():
            if dates:
                all_first_preds.append(dates[0])
        
        if all_first_preds:
            pred_counts = {}
            for pred in all_first_preds:
                month_key = pred.strftime('%Y-%m')
                pred_counts[month_key] = pred_counts.get(month_key, 0) + 1
            
            fig.add_trace(
                go.Bar(
                    x=list(pred_counts.keys()),
                    y=list(pred_counts.values()),
                    name='预测集中度',
                    marker_color='lightsteelblue'
                ),
                row=3, col=2
            )
        
        fig.update_layout(
            height=1200,
            title_text="DeepSeek模型发布预测分析",
            showlegend=True
        )
        
        fig.write_html("deepseek_prediction_analysis.html")
        print("可视化结果已保存到 deepseek_prediction_analysis.html")
        
        return fig
    
    def print_results(self):
        """打印预测结果"""
        print("\n" + "="*80)
        print("🚀 DEEPSEEK 下一代模型发布时间预测结果")
        print("="*80)
        print(f"📅 预测基准日期: {self.today.strftime('%Y年%m月%d日')}")
        print(f"📊 历史数据点: {len(self.df)} 个模型发布记录")
        print(f"🔮 预测方法数: {len(self.predictions)} 种")
        
        print("\n📈 各方法预测结果:")
        print("-" * 80)
        
        all_predictions = []
        for method, dates in self.predictions.items():
            if dates:
                print(f"\n🔹 {method}:")
                for i, date in enumerate(dates[:3], 1):  # 只显示前3个预测
                    days_from_now = (date - self.today).days
                    print(f"   第{i}个模型: {date.strftime('%Y年%m月%d日')} (距今 {days_from_now} 天)")
                    all_predictions.append((date, method))
        
        # 综合分析
        if all_predictions:
            all_predictions.sort()
            print(f"\n🎯 综合预测分析:")
            print("-" * 40)
            
            next_predictions = [pred for pred in all_predictions if pred[0] <= datetime(2025, 12, 31)]
            if next_predictions:
                earliest = next_predictions[0]
                print(f"⚡ 最早预测: {earliest[0].strftime('%Y年%m月%d日')} ({earliest[1]})")
                
                # 计算预测集中度
                next_3_months = [pred for pred in all_predictions 
                               if pred[0] <= self.today + timedelta(days=90)]
                if len(next_3_months) >= 3:
                    print(f"📍 3个月内预测集中度: {len(next_3_months)} 个方法预测")
        
        # 模型性能
        if self.model_performances:
            print(f"\n📊 模型性能排名 (R² Score):")
            print("-" * 40)
            sorted_performance = sorted(
                [(name, perf.get('R2', 0)) for name, perf in self.model_performances.items()],
                key=lambda x: x[1], reverse=True
            )
            
            for i, (name, r2) in enumerate(sorted_performance[:5], 1):
                print(f"{i}. {name}: {r2:.3f}")
        
        print("\n" + "="*80)


def main():
    """主函数"""
    predictor = DeepSeekPredictor()
    
    # 运行所有预测
    predictions = predictor.run_all_predictions()
    
    # 打印结果
    predictor.print_results()
    
    # 创建可视化
    fig = predictor.create_visualizations()
    
    return predictor

if __name__ == "__main__":
    predictor = main() 