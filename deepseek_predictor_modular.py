"""
DeepSeek模型发布预测器 - 模块化版本
整合所有独立的预测方法，提供统一的预测接口
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List

# 导入所有预测器
from models.linear_models import create_linear_predictor, create_ridge_predictor, create_lasso_predictor
from models.time_series import ARIMAPredictor, ExponentialSmoothingPredictor, SeasonalPredictor
from models.interval_based import (
    create_mean_interval_predictor, create_median_interval_predictor,
    create_recent_interval_predictor, create_adaptive_interval_predictor,
    create_weighted_interval_predictor
)
from models.statistical import TrendAnalysisPredictor, SeasonalDecomposePredictor, StatisticalPredictor


class DeepSeekPredictorModular:
    """模块化DeepSeek预测器"""
    
    def __init__(self):
        # 原始数据
        self.data = {
            'version': [
            'DeepSeek Coder',
            'DeepSeek-LLM',
            'DeepSeek-MoE',
            'DeepSeek-Math',
            'DeepSeek-V2 (May)',
            'DeepSeek-Coder V2 (Jun)',
            'DeepSeek-V2 (Jun)',
            'DeepSeek-Coder V2 (Jul)',
            'DeepSeek-V2.5 (Sep)',
            'DeepSeek-R1-Lite',
            'DeepSeek-V2.5 (Dec)',
            'DeepSeek-V3',
            'DeepSeek-R1',
            'DeepSeek-V3-0324',
            'DeepSeek-R1-0528',
            'DeepSeek-V3.1',
            'DeepSeek-V3.1-Terminus',
            'DeepSeek-V3.2'
            ],
            'date': [
            '2023-11-02',
            '2023-11-29',
            '2024-01-11',
            '2024-02-06',
            '2024-05-04',
            '2024-06-14',
            '2024-06-28',
            '2024-07-24',
            '2024-09-05',
            '2024-11-20',
            '2024-12-10',
            '2024-12-25',
            '2025-01-20',
            '2025-03-24',
            '2025-05-28',
            '2025-08-19',
            '2025-09-22',
            '2025-09-29'
            ]
        }
        
        # 固定预测基准日期为 2025-10-01（按需调整）
        self.today = datetime(2025, 10, 1)
        self.df = None
        self.predictors = {}
        self.predictions = {}
        self.performance_summary = {}
        self.backtest_results = {}
        
    def _prepare_data(self):
        """数据预处理"""
        self.df = pd.DataFrame(self.data)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date').reset_index(drop=True)
        
        # 创建特征
        self.df['days_since_start'] = (self.df['date'] - self.df['date'].iloc[0]).dt.days
        self.df['month'] = self.df['date'].dt.month
        self.df['quarter'] = self.df['date'].dt.quarter
        self.df['year'] = self.df['date'].dt.year
        self.df['day_of_year'] = self.df['date'].dt.dayofyear
        self.df['interval_days'] = self.df['days_since_start'].diff()
        
        # 版本特征
        self.df['is_coder'] = self.df['version'].str.contains('Coder').astype(int)
        self.df['is_v2'] = self.df['version'].str.contains('V2').astype(int)
        self.df['is_v3'] = self.df['version'].str.contains('V3').astype(int)
        self.df['is_r1'] = self.df['version'].str.contains('R1').astype(int)
        
        print("✅ 数据预处理完成")
        print(f"📊 数据形状: {self.df.shape}")
        
    def _initialize_predictors(self):
        """初始化所有预测器（排除深度学习）"""
        print("🔧 初始化预测器（排除深度学习算法）...")
        
        # 线性模型组
        self.predictors['Linear Models'] = {
            'Linear Regression': create_linear_predictor(),
            'Ridge Regression': create_ridge_predictor(), 
            'Lasso Regression': create_lasso_predictor()
        }
        
        # 时间序列组
        self.predictors['Time Series'] = {
            # 'ARIMA': ARIMAPredictor(),
            'Exponential Smoothing': ExponentialSmoothingPredictor(),
            'Seasonal Pattern': SeasonalPredictor()
        }
        
        # 间隔分析组
        self.predictors['Interval Based'] = {
            'Mean Interval': create_mean_interval_predictor(),
            'Median Interval': create_median_interval_predictor(),
            'Recent 3 Mean': create_recent_interval_predictor(),
            'Adaptive Interval': create_adaptive_interval_predictor(),
            'Weighted Interval': create_weighted_interval_predictor()
        }
        
        # 统计学组
        self.predictors['Statistical'] = {
            'Trend Analysis': TrendAnalysisPredictor(),
            # 'Seasonal Decompose': SeasonalDecomposePredictor(),
            'Statistical Ensemble': StatisticalPredictor()
        }
        
        # 计算总数
        total_predictors = sum(len(group) for group in self.predictors.values())
        print(f"🎯 已初始化 {total_predictors} 个预测器，分为 {len(self.predictors)} 个类别")
        print("🚫 已排除深度学习算法：MLP和LSTM")
        
    def fit_all_models(self):
        """训练所有模型"""
        print("\n🚀 开始训练所有模型...")
        
        trained_count = 0
        failed_count = 0
        
        for group_name, group_predictors in self.predictors.items():
            print(f"\n📋 训练 {group_name} 组...")
            
            for name, predictor in group_predictors.items():
                try:
                    predictor.fit(self.df)
                    if predictor.is_fitted:
                        trained_count += 1
                        print(f"  ✅ {name}")
                    else:
                        failed_count += 1
                        print(f"  ❌ {name} - 训练失败")
                except Exception as e:
                    failed_count += 1
                    print(f"  ❌ {name} - 错误: {e}")
        
        print(f"\n📈 训练完成! 成功: {trained_count}, 失败: {failed_count}")
    
    def generate_all_predictions(self, n_predictions=1):
        """生成所有预测"""
        print(f"\n🔮 生成预测 (未来 {n_predictions} 个模型)...")
        
        prediction_count = 0
        
        for group_name, group_predictors in self.predictors.items():
            for name, predictor in group_predictors.items():
                if predictor.is_fitted:
                    try:
                        preds = predictor.predict(self.df, n_predictions, self.today)
                        # 过滤有效预测
                        valid_preds = [p for p in preds if p > self.today]
                        if valid_preds and n_predictions == 1:
                            valid_preds = valid_preds[:1]
                        if valid_preds:
                            self.predictions[name] = valid_preds
                            prediction_count += 1
                    except Exception as e:
                        print(f"  ❌ {name} 预测失败: {e}")
        
        print(f"✅ 生成了 {prediction_count} 个有效预测结果")
        return self.predictions
    
    def analyze_performance(self):
        """分析模型性能"""
        print("\n📊 分析模型性能...")
        
        performance_data = []
        
        for group_name, group_predictors in self.predictors.items():
            for name, predictor in group_predictors.items():
                if predictor.is_fitted and predictor.performance_metrics:
                    metrics = predictor.performance_metrics.copy()
                    metrics['Name'] = name
                    metrics['Group'] = group_name
                    performance_data.append(metrics)
        
        if performance_data:
            perf_df = pd.DataFrame(performance_data)
            
            # 按R²排序
            if 'R2' in perf_df.columns:
                perf_df = perf_df.sort_values('R2', ascending=False)
                
                print("\n🏆 Top 5 模型性能 (R² Score):")
                print("-" * 50)
                for i, row in perf_df.head().iterrows():
                    print(f"{row['Name']}: {row['R2']:.4f}")
            
            self.performance_summary = perf_df
        else:
            print("⚠️  没有可用的性能数据")
    
    def create_comprehensive_analysis(self):
        """创建综合分析"""
        print("\n📈 创建综合分析...")
        
        # 收集所有第一个预测
        first_predictions = []
        for name, dates in self.predictions.items():
            if dates:
                first_predictions.append({
                    'Method': name,
                    'Date': dates[0],
                    'Days_from_now': (dates[0] - self.today).days
                })
        
        if not first_predictions:
            print("❌ 没有有效的预测结果")
            return
        
        pred_df = pd.DataFrame(first_predictions)
        pred_df = pred_df.sort_values('Days_from_now')
        
        # 分析结果
        earliest = pred_df.iloc[0]
        latest = pred_df.iloc[-1]
        
        # 3个月内的预测
        three_months_later = self.today + timedelta(days=90)
        near_term = pred_df[pred_df['Date'] <= three_months_later]
        
        print("\n🎯 预测分析结果:")
        print("=" * 60)
        print(f"📅 预测基准日期: {self.today.strftime('%Y年%m月%d日')}")
        print(f"📊 有效预测方法: {len(self.predictions)} 种")
        print(f"⚡ 最早预测: {earliest['Date'].strftime('%Y年%m月%d日')} ({earliest['Method']})")
        print(f"⏰ 距离最早预测: {earliest['Days_from_now']} 天")
        print(f"🔄 预测时间跨度: {(latest['Date'] - earliest['Date']).days} 天")
        print(f"📍 3个月内预测数: {len(near_term)} 个方法")
        
        # 一致性评级
        consistency_ratio = len(near_term) / len(pred_df)
        if consistency_ratio >= 0.6:
            consistency = "⭐⭐⭐ 高一致性"
        elif consistency_ratio >= 0.4:
            consistency = "⭐⭐ 中等一致性"
        else:
            consistency = "⭐ 低一致性"
        
        print(f"🎖️  预测一致性: {consistency} ({consistency_ratio:.1%})")
        
        return {
            'earliest_prediction': earliest,
            'prediction_count': len(self.predictions),
            'near_term_count': len(near_term),
            'consistency': consistency
        }
    
    def create_advanced_visualizations(self):
        """创建高级可视化"""
        print("\n🎨 创建高级可视化...")

        if not self.predictions:
            print("❌ 没有预测结果，跳过可视化")
            return None

        group_lookup = {}
        for group_name, group_predictors in self.predictors.items():
            for name in group_predictors.keys():
                group_lookup[name] = group_name

        records = []
        last_release = self.df['date'].iloc[-1]
        for name, dates in self.predictions.items():
            if not dates:
                continue
            next_date = min(dates)
            records.append({
                'Model': name,
                'Next Release': next_date,
                'Days Ahead': (next_date - self.today).days,
                'Interval From Last': (next_date - last_release).days
            })

        if not records:
            print("❌ 无有效预测结果")
            return None

        pred_df = pd.DataFrame(records)
        perf_df = getattr(self, 'performance_summary', None)
        if perf_df is not None and not perf_df.empty:
            metrics_df = perf_df[['Name', 'Group', 'MAE', 'RMSE']].rename(columns={'Name': 'Model'})
            pred_df = pred_df.merge(metrics_df, how='left', on='Model')
        else:
            pred_df['Group'] = pred_df['Model'].map(group_lookup)
            pred_df['MAE'] = np.nan
            pred_df['RMSE'] = np.nan

        pred_df['Group'] = pred_df.get('Group', pd.Series(dtype=str)).fillna(pred_df['Model'].map(group_lookup))
        pred_df['Group'] = pred_df['Group'].fillna('N/A')
        pred_df['MAE'] = pred_df['MAE'].astype(float)
        pred_df['RMSE'] = pred_df['RMSE'].astype(float)
        pred_df['Days Ahead'] = pred_df['Days Ahead'].astype(int)
        pred_df['Interval From Last'] = pred_df['Interval From Last'].astype(int)

        sorted_pred_df = pred_df.sort_values(by=['MAE', 'Days Ahead'], na_position='last').reset_index(drop=True)

        timeline_y = self.df['days_since_start']
        best_predictions = sorted_pred_df.head(3)

        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[[{"type": "xy"}, {"type": "xy"}],
                   [{"type": "table"}, {"type": "xy"}]],
            subplot_titles=[
                '📈 历史发布时间线与最佳预测',
                '🏆 MAE 表现最优模型',
                '🔮 下一次发布日期预测摘要',
                '📊 发布间隔分布 & 预测间隔对比'
            ]
        )

        # 1. 时间线
        fig.add_trace(
            go.Scatter(
                x=self.df['date'],
                y=timeline_y,
                mode='lines+markers',
                name='历史发布',
                text=self.df['version'],
                hovertemplate='%{text}<br>%{x|%Y-%m-%d}<extra></extra>',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=8)
            ),
            row=1, col=1
        )

        if not best_predictions.empty:
            pred_y = (best_predictions['Next Release'] - self.df['date'].iloc[0]).dt.days
            fig.add_trace(
                go.Scatter(
                    x=best_predictions['Next Release'],
                    y=pred_y,
                    mode='markers',
                    name='最佳预测',
                    marker=dict(size=12, color='#ff7f0e', symbol='diamond'),
                    text=[f"{row.Model}<br>MAE: {row.MAE:.2f}" if not np.isnan(row.MAE) else row.Model
                          for row in best_predictions.itertuples()],
                    hovertemplate='%{text}<br>日期: %{x|%Y-%m-%d}<extra></extra>'
                ),
                row=1, col=1
            )

        fig.update_yaxes(title_text='距首发天数', row=1, col=1)
        fig.update_xaxes(title_text='日期', row=1, col=1)

        # 2. MAE 排名
        mae_df = sorted_pred_df.dropna(subset=['MAE']).head(8)
        if not mae_df.empty:
            fig.add_trace(
                go.Bar(
                    x=mae_df['Model'],
                    y=mae_df['MAE'],
                    text=[f"{v:.2f}" for v in mae_df['MAE']],
                    textposition='auto',
                    marker_color=px.colors.qualitative.Safe[:len(mae_df)],
                    name='MAE'
                ),
                row=1, col=2
            )
            fig.update_yaxes(title_text='MAE (天)', row=1, col=2)
        else:
            fig.add_annotation(
                text='暂无 MAE 信息',
                xref='x2', yref='y2',
                x=0.5, y=0.5, showarrow=False,
                row=1, col=2
            )

        # 3. 表格摘要
        table_df = sorted_pred_df.copy()
        table_df['Next Release'] = table_df['Next Release'].dt.strftime('%Y-%m-%d')
        table_df['Days Ahead'] = table_df['Days Ahead'].apply(lambda x: f"{x} 天")
        table_df['Interval From Last'] = table_df['Interval From Last'].apply(lambda x: f"{x} 天")
        table_df['MAE'] = table_df['MAE'].apply(lambda x: f"{x:.2f}" if not np.isnan(x) else '—')
        table_df['RMSE'] = table_df['RMSE'].apply(lambda x: f"{x:.2f}" if not np.isnan(x) else '—')

        fig.add_trace(
            go.Table(
                header=dict(
                    values=['模型', '类别', '下一次发布日期', '距今天数', '距上次间隔', 'MAE', 'RMSE'],
                    fill_color='#1f77b4',
                    font=dict(color='white', size=12),
                    align='left'
                ),
                cells=dict(
                    values=[
                        table_df['Model'],
                        table_df['Group'],
                        table_df['Next Release'],
                        table_df['Days Ahead'],
                        table_df['Interval From Last'],
                        table_df['MAE'],
                        table_df['RMSE'],
                    ],
                    fill_color=['#fdfefe'] * 7,
                    align='left',
                    font=dict(size=11)
                )
            ),
            row=2, col=1
        )

        # 4. 发布间隔直方图
        intervals = self.df['interval_days'].dropna()
        if not intervals.empty:
            bins = min(20, len(intervals))
            hist_counts, _ = np.histogram(intervals, bins=bins)
            max_count = hist_counts.max() if hist_counts.size else 1
            line_height = max_count * 1.05

            fig.add_trace(
                go.Histogram(
                    x=intervals,
                    nbinsx=bins,
                    marker=dict(color='#9edae5'),
                    name='历史间隔分布',
                    opacity=0.85
                ),
                row=2, col=2
            )

            def add_reference_line(value, color, name, dash='dash'):
                fig.add_trace(
                    go.Scatter(
                        x=[value, value],
                        y=[0, line_height],
                        mode='lines',
                        name=name,
                        line=dict(color=color, dash=dash, width=2)
                    ),
                    row=2, col=2
                )

            add_reference_line(intervals.mean(), '#ff7f0e', '平均间隔', dash='dash')
            add_reference_line(intervals.median(), '#2ca02c', '中位数', dash='dot')

            highlight = sorted_pred_df.dropna(subset=['MAE']).head(3)
            for idx, (_, row_data) in enumerate(highlight.iterrows()):
                interval_val = row_data['Interval From Last']
                color = px.colors.qualitative.Safe[idx % len(px.colors.qualitative.Safe)]
                add_reference_line(interval_val, color, f"{row_data['Model']} 预测间隔", dash='solid')

            fig.update_xaxes(title_text='发布间隔 (天)', row=2, col=2)
            fig.update_yaxes(title_text='次数', row=2, col=2)
        else:
            fig.add_annotation(
                text='暂无间隔数据',
                xref='x4', yref='y4',
                x=0.5, y=0.5, showarrow=False,
                row=2, col=2
            )

        fig.update_layout(
            height=1000,
            title_text="🚀 DeepSeek 发布预测概览",
            showlegend=True,
            template="plotly_white",
            font=dict(size=12),
            hovermode="closest",
            margin=dict(l=40, r=40, t=60, b=40)
        )

        fig.write_html(
            "deepseek_modular_analysis.html",
            include_plotlyjs='cdn',
            full_html=True,
            config={'displayModeBar': True, 'displaylogo': False}
        )
        print("✅ 高级可视化已保存到 deepseek_modular_analysis.html")

        return fig
    
    def print_detailed_results(self):
        """打印详细结果"""
        print("\n" + "="*80)
        print("🚀 DEEPSEEK 模块化预测结果")
        print("="*80)
        
        # 按组显示预测结果
        for group_name, group_predictors in self.predictors.items():
            group_predictions = {}
            for name, predictor in group_predictors.items():
                if name in self.predictions:
                    group_predictions[name] = self.predictions[name]
            
            if group_predictions:
                print(f"\n📋 {group_name} 组预测:")
                print("-" * 50)
                
                for name, dates in group_predictions.items():
                    print(f"\n🔹 {name}:")
                    for i, date in enumerate(dates[:3], 1):
                        days_from_now = (date - self.today).days
                        print(f"   第{i}个模型: {date.strftime('%Y年%m月%d日')} (距今 {days_from_now} 天)")
        
        print("\n" + "="*80)
    
    def run_backtest(self, start_from=3, verbose=True):
        """
        运行回测：从第start_from个数据点开始，逐步预测下一个发布时间
        
        Args:
            start_from: 从第几个数据点开始回测（默认3，即从第3次发布开始）
            verbose: 是否打印详细信息
        """
        print(f"\n🔄 开始回测分析 (从第{start_from}个数据点开始)")
        print("="*60)
        
        backtest_results = {}
        
        # 获取所有预测器名称
        all_predictors = {}
        for group_name, group_predictors in self.predictors.items():
            for name, predictor in group_predictors.items():
                all_predictors[name] = predictor
        
        # 初始化结果存储
        for name in all_predictors.keys():
            backtest_results[name] = {
                'predictions': [],
                'actual_dates': [],
                'errors_days': [],
                'success_count': 0,
                'total_attempts': 0
            }
        
        # 从第start_from个数据点开始逐步预测
        for i in range(start_from, len(self.df)):
            target_date = self.df.iloc[i]['date']
            target_version = self.df.iloc[i]['version']
            
            if verbose:
                print(f"\n📅 预测第{i+1}个发布: {target_version} ({target_date.strftime('%Y-%m-%d')})")
                print("-" * 40)
            
            # 使用前i个数据点训练和预测
            train_df = self.df.iloc[:i].copy()
            
            # 重新计算训练数据的特征
            train_df['days_since_start'] = (train_df['date'] - train_df['date'].iloc[0]).dt.days
            train_df['month'] = train_df['date'].dt.month
            train_df['quarter'] = train_df['date'].dt.quarter
            train_df['year'] = train_df['date'].dt.year
            train_df['day_of_year'] = train_df['date'].dt.dayofyear
            train_df['interval_days'] = train_df['days_since_start'].diff()
            
            # 版本特征
            train_df['is_coder'] = train_df['version'].str.contains('Coder').astype(int)
            train_df['is_v2'] = train_df['version'].str.contains('V2').astype(int)
            train_df['is_v3'] = train_df['version'].str.contains('V3').astype(int)
            train_df['is_r1'] = train_df['version'].str.contains('R1').astype(int)
            
            # 对每个预测器进行训练和预测
            for name, predictor in all_predictors.items():
                try:
                    # 重新创建预测器实例以避免状态污染
                    if name == 'Linear Regression':
                        predictor = create_linear_predictor()
                    elif name == 'Ridge Regression':
                        predictor = create_ridge_predictor()
                    elif name == 'Lasso Regression':
                        predictor = create_lasso_predictor()
                    elif name == 'ARIMA':
                        predictor = ARIMAPredictor()
                    elif name == 'Exponential Smoothing':
                        predictor = ExponentialSmoothingPredictor()
                    elif name == 'Seasonal Pattern':
                        predictor = SeasonalPredictor()
                    # 注释掉ensemble models - 小数据集不适合
                    # elif name == 'Random Forest':
                    #     predictor = RandomForestPredictor()
                    # elif name == 'Gradient Boosting':
                    #     predictor = GradientBoostingPredictor()
                    # elif name == 'XGBoost':
                    #     predictor = XGBoostPredictor()
                    # elif name == 'SVR':
                    #     predictor = SVRPredictor()
                    elif name == 'Mean Interval':
                        predictor = create_mean_interval_predictor()
                    elif name == 'Median Interval':
                        predictor = create_median_interval_predictor()
                    elif name == 'Recent 3 Mean':
                        predictor = create_recent_interval_predictor()
                    elif name == 'Adaptive Interval':
                        predictor = create_adaptive_interval_predictor()
                    elif name == 'Weighted Interval':
                        predictor = create_weighted_interval_predictor()
                    elif name == 'Trend Analysis':
                        predictor = TrendAnalysisPredictor()
                    elif name == 'Seasonal Decompose':
                        predictor = SeasonalDecomposePredictor()
                    elif name == 'Statistical Ensemble':
                        predictor = StatisticalPredictor()
                    
                    # 训练模型
                    predictor.fit(train_df)
                    
                    if predictor.is_fitted:
                        # 预测下一个发布时间
                        # 使用训练数据的最后一天作为"今天"
                        last_known_date = train_df['date'].iloc[-1]
                        predictions = predictor.predict(train_df, n_predictions=1, today=last_known_date)
                        
                        backtest_results[name]['total_attempts'] += 1
                        
                        if predictions:
                            pred_date = predictions[0]
                            error_days = (pred_date - target_date).days
                            
                            backtest_results[name]['predictions'].append(pred_date)
                            backtest_results[name]['actual_dates'].append(target_date)
                            backtest_results[name]['errors_days'].append(error_days)
                            
                            if abs(error_days) <= 30:  # 30天内算成功
                                backtest_results[name]['success_count'] += 1
                            
                            if verbose:
                                print(f"  🔹 {name}: {pred_date.strftime('%Y-%m-%d')} (误差: {error_days:+d}天)")
                        else:
                            backtest_results[name]['errors_days'].append(float('inf'))
                            if verbose:
                                print(f"  ❌ {name}: 无预测结果")
                    else:
                        backtest_results[name]['total_attempts'] += 1
                        backtest_results[name]['errors_days'].append(float('inf'))
                        if verbose:
                            print(f"  ❌ {name}: 训练失败")
                            
                except Exception as e:
                    backtest_results[name]['total_attempts'] += 1
                    backtest_results[name]['errors_days'].append(float('inf'))
                    if verbose:
                        print(f"  ❌ {name}: 错误 - {str(e)[:50]}...")
        
        # 计算回测统计
        print(f"\n📊 回测统计分析 (共{len(self.df) - start_from}次预测)")
        print("="*60)
        
        backtest_summary = []
        for name, results in backtest_results.items():
            if results['total_attempts'] > 0:
                valid_errors = [e for e in results['errors_days'] if e != float('inf')]
                
                if valid_errors:
                    mae = np.mean(np.abs(valid_errors))
                    rmse = np.sqrt(np.mean(np.square(valid_errors)))
                    success_rate = results['success_count'] / results['total_attempts']
                    
                    # 用完整数据集重新训练并预测下一次发布时间
                    next_release_prediction = "训练失败"
                    try:
                        # 创建对应的预测器
                        if name == 'Linear Regression':
                            predictor = create_linear_predictor()
                        elif name == 'Ridge Regression':
                            predictor = create_ridge_predictor()
                        elif name == 'Lasso Regression':
                            predictor = create_lasso_predictor()
                        elif name == 'ARIMA':
                            predictor = ARIMAPredictor()
                        elif name == 'Exponential Smoothing':
                            predictor = ExponentialSmoothingPredictor()
                        elif name == 'Seasonal Pattern':
                            predictor = SeasonalPredictor()
                        elif name == 'Mean Interval':
                            predictor = create_mean_interval_predictor()
                        elif name == 'Median Interval':
                            predictor = create_median_interval_predictor()
                        elif name == 'Recent 3 Mean':
                            predictor = create_recent_interval_predictor()
                        elif name == 'Adaptive Interval':
                            predictor = create_adaptive_interval_predictor()
                        elif name == 'Weighted Interval':
                            predictor = create_weighted_interval_predictor()
                        elif name == 'Trend Analysis':
                            predictor = TrendAnalysisPredictor()
                        elif name == 'Seasonal Decompose':
                            predictor = SeasonalDecomposePredictor()
                        elif name == 'Statistical Ensemble':
                            predictor = StatisticalPredictor()
                        else:
                            predictor = None
                        
                        if predictor:
                            # 用完整数据集训练
                            predictor.fit(self.df)
                            if predictor.is_fitted:
                                # 预测下一次发布时间
                                next_predictions = predictor.predict(self.df, n_predictions=1, today=self.today)
                                if next_predictions:
                                    next_release_prediction = next_predictions[0].strftime('%Y-%m-%d')
                                else:
                                    next_release_prediction = "无预测"
                            else:
                                next_release_prediction = "训练失败"
                    except Exception as e:
                        next_release_prediction = f"错误: {str(e)[:10]}"
                    
                    backtest_summary.append({
                        'Method': name,
                        'MAE (days)': mae,
                        'RMSE (days)': rmse,
                        'Success_Rate': success_rate,
                        'Valid_Predictions': len(valid_errors),
                        'Total_Attempts': results['total_attempts'],
                        'Next_Release': next_release_prediction
                    })
        
        # 排序并显示结果
        if backtest_summary:
            bt_df = pd.DataFrame(backtest_summary)
            bt_df = bt_df.sort_values('MAE (days)')
            
            print("\n🏆 回测排名 (按MAE排序):")
            print("-" * 110)
            print(f"{'方法':<20} {'MAE':<10} {'RMSE':<10} {'成功率':<10} {'有效预测':<10} {'预测下次发布':<15}")
            print("-" * 110)
            
            for _, row in bt_df.head(15).iterrows():
                print(f"{row['Method']:<20} {row['MAE (days)']:<10.1f} {row['RMSE (days)']:<10.1f} "
                      f"{row['Success_Rate']:<10.1%} {row['Valid_Predictions']:<10d} {row['Next_Release']:<15}")
            
            self.backtest_results = backtest_results
            return bt_df
        else:
            print("❌ 没有有效的回测结果")
            return None
    
    def compare_r2_vs_backtest(self):
        """比较R²和回测结果的关系"""
        print("\n🔍 比较R²和回测表现的关系")
        print("="*60)
        
        if not hasattr(self, 'backtest_results') or not self.backtest_results:
            print("❌ 需要先运行回测分析")
            return None
        
        # 收集R²和回测数据
        comparison_data = []
        
        for group_name, group_predictors in self.predictors.items():
            for name, predictor in group_predictors.items():
                # 获取R²分数
                r2_score = None
                if hasattr(predictor, 'performance_metrics') and predictor.performance_metrics:
                    r2_score = predictor.performance_metrics.get('R2', None)
                
                # 获取回测结果
                if name in self.backtest_results:
                    results = self.backtest_results[name]
                    valid_errors = [e for e in results['errors_days'] if e != float('inf')]
                    
                    if valid_errors and r2_score is not None:
                        mae = np.mean(np.abs(valid_errors))
                        rmse = np.sqrt(np.mean(np.square(valid_errors)))
                        success_rate = results['success_count'] / results['total_attempts']
                        
                        comparison_data.append({
                            'Method': name,
                            'Group': group_name,
                            'R2_Score': r2_score,
                            'MAE': mae,
                            'RMSE': rmse,
                            'Success_Rate': success_rate,
                            'Valid_Predictions': len(valid_errors)
                        })
        
        if not comparison_data:
            print("❌ 没有足够的数据进行比较")
            return None
        
        comp_df = pd.DataFrame(comparison_data)
        
        # 计算相关性
        r2_mae_corr = comp_df['R2_Score'].corr(comp_df['MAE'])
        r2_success_corr = comp_df['R2_Score'].corr(comp_df['Success_Rate'])
        
        print(f"\n📊 相关性分析:")
        print(f"R² vs MAE相关性: {r2_mae_corr:.3f}")
        print(f"R² vs 成功率相关性: {r2_success_corr:.3f}")
        
        # 按不同指标排序
        print(f"\n🏆 按R²排序 vs 按MAE排序:")
        print("-" * 60)
        print("R²排序 (前5):")
        r2_top = comp_df.nlargest(5, 'R2_Score')
        for _, row in r2_top.iterrows():
            print(f"  {row['Method']:<20} R²:{row['R2_Score']:.3f} MAE:{row['MAE']:.1f}")
        
        print("\nMAE排序 (前5):")
        mae_top = comp_df.nsmallest(5, 'MAE')
        for _, row in mae_top.iterrows():
            print(f"  {row['Method']:<20} R²:{row['R2_Score']:.3f} MAE:{row['MAE']:.1f}")
        
        # 找出R²高但MAE大的方法
        print(f"\n⚠️  R²高但回测表现差的方法:")
        print("-" * 40)
        high_r2_poor_mae = comp_df[(comp_df['R2_Score'] > 0.8) & (comp_df['MAE'] > 100)]
        if not high_r2_poor_mae.empty:
            for _, row in high_r2_poor_mae.iterrows():
                print(f"  {row['Method']:<20} R²:{row['R2_Score']:.3f} MAE:{row['MAE']:.1f}")
        else:
            print("  没有发现这类方法")
        
        # 找出R²低但MAE小的方法
        print(f"\n✅ R²低但回测表现好的方法:")
        print("-" * 40)
        low_r2_good_mae = comp_df[(comp_df['R2_Score'] < 0.5) & (comp_df['MAE'] < 50)]
        if not low_r2_good_mae.empty:
            for _, row in low_r2_good_mae.iterrows():
                print(f"  {row['Method']:<20} R²:{row['R2_Score']:.3f} MAE:{row['MAE']:.1f}")
        else:
            print("  没有发现这类方法")
        
        print(f"\n📝 结论:")
        if abs(r2_mae_corr) < 0.3:
            print("❌ R²和回测表现相关性很弱，R²不是选择模型的好指标")
        elif r2_mae_corr < -0.5:
            print("✅ R²和回测表现负相关较强，R²是选择模型的好指标")
        else:
            print("⚠️  R²和回测表现相关性中等，建议同时参考两个指标")
        
        return comp_df
    
    def create_backtest_visualization(self):
        """创建回测结果可视化"""
        print("\n🎨 创建回测结果可视化...")
        
        if not hasattr(self, 'backtest_results') or not self.backtest_results:
            print("❌ 需要先运行回测分析")
            return None
        
        # 收集回测数据
        backtest_data = []
        for name, results in self.backtest_results.items():
            for i, (pred_date, actual_date, error) in enumerate(zip(
                results['predictions'], results['actual_dates'], results['errors_days']
            )):
                if error != float('inf'):
                    backtest_data.append({
                        'Method': name,
                        'Prediction_Number': i + 1,
                        'Predicted_Date': pred_date,
                        'Actual_Date': actual_date,
                        'Error_Days': error,
                        'Absolute_Error': abs(error)
                    })
        
        if not backtest_data:
            print("❌ 没有有效的回测数据")
            return None
        
        bt_df = pd.DataFrame(backtest_data)
        
        summary_df = bt_df.groupby('Method').agg(
            MAE=('Absolute_Error', 'mean'),
            MedianAE=('Absolute_Error', 'median'),
            Std=('Absolute_Error', 'std'),
            HitRate30=('Absolute_Error', lambda x: (x <= 30).mean()),
            Samples=('Absolute_Error', 'count')
        ).reset_index()

        summary_df['Std'] = summary_df['Std'].fillna(0.0)
        summary_df['HitRate30'] = (summary_df['HitRate30'] * 100).round(1)
        summary_df = summary_df.sort_values('MAE')

        top_methods = summary_df.head(6)['Method'].tolist()
        line_methods = summary_df.head(4)['Method'].tolist()

        palette = px.colors.qualitative.Safe

        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[[{"type": "xy"}, {"type": "xy"}],
                   [{"type": "table"}, {"type": "xy"}]],
            subplot_titles=[
                '📊 回测误差分布 (Top 6)',
                '📈 误差走势 (Top 4)',
                '🧾 方法表现摘要',
                '🎯 预测 vs 实际 (Top 6)'
            ]
        )

        # 1. 箱线图
        for i, method in enumerate(top_methods):
            method_data = bt_df[bt_df['Method'] == method]
            fig.add_trace(
                go.Box(
                    y=method_data['Error_Days'],
                    name=method,
                    marker_color=palette[i % len(palette)],
                    boxmean=True,
                    boxpoints='suspectedoutliers',
                    showlegend=False
                ),
                row=1, col=1
            )
        fig.update_yaxes(title_text='误差 (天)', row=1, col=1)

        # 2. 误差时间序列
        for i, method in enumerate(line_methods):
            method_data = bt_df[bt_df['Method'] == method].sort_values('Actual_Date')
            fig.add_trace(
                go.Scatter(
                    x=method_data['Actual_Date'],
                    y=method_data['Error_Days'],
                    mode='lines+markers',
                    name=method,
                    line=dict(color=palette[i % len(palette)], width=2),
                    marker=dict(size=6),
                    hovertemplate='<b>%{fullData.name}</b><br>实际: %{x|%Y-%m-%d}<br>误差: %{y} 天<extra></extra>'
                ),
                row=1, col=2
            )

        fig.add_hline(y=0, line_dash='dot', line_color='#7f7f7f', row=1, col=2)
        fig.add_hline(y=30, line_dash='dash', line_color='#bcbd22', row=1, col=2)
        fig.add_hline(y=-30, line_dash='dash', line_color='#bcbd22', row=1, col=2)
        fig.update_yaxes(title_text='误差 (天)', row=1, col=2)
        fig.update_xaxes(title_text='实际发布日期', row=1, col=2)

        # 3. 表格
        table_summary = summary_df.copy()
        table_summary['MAE'] = table_summary['MAE'].map(lambda x: f"{x:.2f}")
        table_summary['MedianAE'] = table_summary['MedianAE'].map(lambda x: f"{x:.2f}")
        table_summary['Std'] = table_summary['Std'].map(lambda x: f"{x:.2f}")
        table_summary['HitRate30'] = table_summary['HitRate30'].map(lambda x: f"{x:.1f}%")

        fig.add_trace(
            go.Table(
                header=dict(
                    values=['方法', 'MAE', '中位误差', '标准差', '30天命中率', '样本数'],
                    fill_color='#1f77b4',
                    font=dict(color='white', size=12),
                    align='left'
                ),
                cells=dict(
                    values=[
                        table_summary['Method'],
                        table_summary['MAE'],
                        table_summary['MedianAE'],
                        table_summary['Std'],
                        table_summary['HitRate30'],
                        table_summary['Samples']
                    ],
                    fill_color=['#fdfefe'] * 6,
                    align='left',
                    font=dict(size=11)
                )
            ),
            row=2, col=1
        )

        # 4. 预测 vs 实际
        for i, method in enumerate(top_methods):
            method_data = bt_df[bt_df['Method'] == method]
            fig.add_trace(
                go.Scatter(
                    x=method_data['Actual_Date'],
                    y=method_data['Predicted_Date'],
                    mode='markers',
                    name=method,
                    marker=dict(color=palette[i % len(palette)], size=8, opacity=0.8),
                    showlegend=False,
                    hovertemplate='<b>%{fullData.name}</b><br>实际: %{x|%Y-%m-%d}<br>预测: %{y|%Y-%m-%d}<extra></extra>'
                ),
                row=2, col=2
            )

        min_date = bt_df['Actual_Date'].min()
        max_date = bt_df['Actual_Date'].max()
        fig.add_trace(
            go.Scatter(
                x=[min_date, max_date],
                y=[min_date, max_date],
                mode='lines',
                name='理想预测',
                line=dict(color='#7f7f7f', dash='dash', width=2),
                showlegend=False
            ),
            row=2, col=2
        )
        fig.update_xaxes(title_text='实际发布日期', row=2, col=2)
        fig.update_yaxes(title_text='预测发布日期', row=2, col=2)

        fig.update_layout(
            height=1000,
            title_text="🔄 DeepSeek 回测分析报告",
            showlegend=True,
            template="plotly_white",
            font=dict(size=12),
            hovermode="closest",
            margin=dict(l=40, r=40, t=60, b=40)
        )

        fig.write_html(
            "deepseek_backtest_analysis.html",
            include_plotlyjs='cdn',
            full_html=True,
            config={'displayModeBar': True, 'displaylogo': False}
        )
        print("✅ 回测可视化已保存到 deepseek_backtest_analysis.html")

        return fig
    
    def run_complete_analysis(self):
        """运行完整分析流程"""
        print("🎯 开始DeepSeek模型发布预测完整分析")
        print("="*60)
        
        # 1. 数据准备
        self._prepare_data()
        
        # 2. 初始化预测器
        self._initialize_predictors()
        
        # 3. 训练模型
        self.fit_all_models()
        
        # 4. 生成预测
        self.generate_all_predictions(n_predictions=1)
        
        # 5. 性能分析
        self.analyze_performance()
        
        # 6. 综合分析
        analysis_summary = self.create_comprehensive_analysis()
        
        # 7. 打印详细结果
        self.print_detailed_results()
        
        # 8. 创建可视化
        self.create_advanced_visualizations()
        
        # 9. 运行回测
        self.run_backtest()
        
        # 10. 比较R²和回测结果的关系
        self.compare_r2_vs_backtest()
        
        # 11. 创建回测可视化
        self.create_backtest_visualization()
        
        print("\n🎉 分析完成!")
        return {
            'predictions': self.predictions,
            'performance': self.performance_summary,
            'summary': analysis_summary,
            'backtest_results': self.backtest_results
        }


def main():
    """主函数"""
    predictor = DeepSeekPredictorModular()
    results = predictor.run_complete_analysis()
    return predictor, results


if __name__ == "__main__":
    predictor, results = main() 
