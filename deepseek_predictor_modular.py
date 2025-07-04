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
from models.ensemble_models import RandomForestPredictor, GradientBoostingPredictor, XGBoostPredictor, SVRPredictor
from models.interval_based import (
    create_mean_interval_predictor, create_median_interval_predictor,
    create_recent_interval_predictor, create_adaptive_interval_predictor,
    create_weighted_interval_predictor
)
from models.neural_networks import MLPPredictor, LSTMPredictor
from models.statistical import TrendAnalysisPredictor, SeasonalDecomposePredictor, StatisticalPredictor


class DeepSeekPredictorModular:
    """模块化DeepSeek预测器"""
    
    def __init__(self):
        # 原始数据
        self.data = {
            'version': [
                'DeepSeek Coder', 'DeepSeek-LLM', 'DeepSeek-V2 (Apr)',
                'DeepSeek-Coder V2 (Jun)', 'DeepSeek-V2 (Jun)', 'DeepSeek-Coder V2 (Jul)',
                'DeepSeek-V2.5 (Sep)', 'DeepSeek-V3', 'DeepSeek-V2.5 (Dec)',
                'DeepSeek-R1', 'DeepSeek-V3-0324', 'DeepSeek-R1-0528'
            ],
            'date': [
                '2023-11-02', '2023-11-29', '2024-04-28', '2024-06-14',
                '2024-06-28', '2024-07-24', '2024-09-05', '2024-12-25',
                '2024-12-10', '2025-01-20', '2025-03-24', '2025-05-28'
            ]
        }
        
        self.today = datetime(2025, 7, 4)
        self.df = None
        self.predictors = {}
        self.predictions = {}
        self.performance_summary = {}
        
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
        """初始化所有预测器"""
        print("🔧 初始化预测器...")
        
        # 线性模型组
        self.predictors['Linear Models'] = {
            'Linear Regression': create_linear_predictor(),
            'Ridge Regression': create_ridge_predictor(), 
            'Lasso Regression': create_lasso_predictor()
        }
        
        # 时间序列组
        self.predictors['Time Series'] = {
            'ARIMA': ARIMAPredictor(),
            'Exponential Smoothing': ExponentialSmoothingPredictor(),
            'Seasonal Pattern': SeasonalPredictor()
        }
        
        # 集成学习组
        self.predictors['Ensemble Models'] = {
            'Random Forest': RandomForestPredictor(),
            'Gradient Boosting': GradientBoostingPredictor(),
            'XGBoost': XGBoostPredictor(),
            'SVR': SVRPredictor()
        }
        
        # 间隔分析组
        self.predictors['Interval Based'] = {
            'Mean Interval': create_mean_interval_predictor(),
            'Median Interval': create_median_interval_predictor(),
            'Recent 3 Mean': create_recent_interval_predictor(),
            'Adaptive Interval': create_adaptive_interval_predictor(),
            'Weighted Interval': create_weighted_interval_predictor()
        }
        
        # 神经网络组
        self.predictors['Neural Networks'] = {
            'MLP': MLPPredictor(),
            'LSTM': LSTMPredictor()
        }
        
        # 统计学组
        self.predictors['Statistical'] = {
            'Trend Analysis': TrendAnalysisPredictor(),
            'Seasonal Decompose': SeasonalDecomposePredictor(),
            'Statistical Ensemble': StatisticalPredictor()
        }
        
        # 计算总数
        total_predictors = sum(len(group) for group in self.predictors.values())
        print(f"🎯 已初始化 {total_predictors} 个预测器，分为 {len(self.predictors)} 个类别")
        
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
    
    def generate_all_predictions(self, n_predictions=5):
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
        
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=[
                '📈 历史发布时间线', '📊 发布间隔分析',
                '🔮 预测结果对比 (按方法分组)', '⚡ 最早预测分布',
                '🏆 模型性能排名', '📅 预测日期热力图',
                '📊 预测一致性分析', '🎯 综合预测建议'
            ],
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"colspan": 2}, None],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ]
        )
        
        # 1. 历史时间线
        fig.add_trace(
            go.Scatter(
                x=self.df['date'],
                y=list(range(len(self.df))),
                mode='markers+lines',
                name='历史发布',
                text=self.df['version'],
                hovertemplate='%{text}<br>%{x}<extra></extra>',
                marker=dict(size=12, color='gold', symbol='diamond'),
                line=dict(color='lightblue', width=3)
            ),
            row=1, col=1
        )
        
        # 2. 发布间隔
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
        
        # 3. 预测结果对比 (按组分类)
        colors = px.colors.qualitative.Set3
        method_groups = {}
        
        for group_name, group_predictors in self.predictors.items():
            for name, predictor in group_predictors.items():
                if name in self.predictions:
                    method_groups[name] = group_name
        
        # 为每个组分配颜色
        unique_groups = list(set(method_groups.values()))
        group_colors = {group: colors[i % len(colors)] for i, group in enumerate(unique_groups)}
        
        for i, (method, dates) in enumerate(self.predictions.items()):
            if dates:
                group = method_groups.get(method, 'Other')
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=[f'{method}'] * len(dates),
                        mode='markers',
                        marker=dict(size=10, color=group_colors.get(group, 'gray')),
                        name=f'{group}: {method}',
                        text=[f'预测 {j+1}' for j in range(len(dates))],
                        hovertemplate='%{text}<br>%{x}<extra></extra>',
                        showlegend=(i < 10)  # 只显示前10个图例
                    ),
                    row=2, col=1
                )
        
        # 添加今天的标记线（简化版本）
        if self.predictions:
            today_line = go.Scatter(
                x=[self.today, self.today],
                y=[0, len(self.predictions)],
                mode='lines',
                line=dict(color='red', dash='dash', width=2),
                name='今天',
                showlegend=True
            )
            fig.add_trace(today_line, row=2, col=1)
        
        # 4. 最早预测分布
        first_predictions = [dates[0] for dates in self.predictions.values() if dates]
        if first_predictions:
            pred_months = [d.strftime('%Y-%m') for d in first_predictions]
            month_counts = pd.Series(pred_months).value_counts().sort_index()
            
            fig.add_trace(
                go.Bar(
                    x=month_counts.index,
                    y=month_counts.values,
                    name='预测集中度',
                    marker_color='lightsteelblue',
                    text=month_counts.values,
                    textposition='auto'
                ),
                row=3, col=2
            )
        
        # 5. 模型性能排名
        if hasattr(self, 'performance_summary') and not self.performance_summary.empty:
            top_performers = self.performance_summary.head(8)
            fig.add_trace(
                go.Bar(
                    x=top_performers['Name'],
                    y=top_performers['R2'],
                    name='R² Score',
                    text=[f'{x:.3f}' for x in top_performers['R2']],
                    textposition='auto',
                    marker_color='lightgreen'
                ),
                row=4, col=1
            )
        
        # 6. 预测统计
        if first_predictions:
            # 按周统计
            week_data = {}
            for pred in first_predictions:
                week = pred.strftime('%Y-W%U')
                week_data[week] = week_data.get(week, 0) + 1
            
            if week_data:
                fig.add_trace(
                    go.Bar(
                        x=list(week_data.keys()),
                        y=list(week_data.values()),
                        name='按周预测分布',
                        marker_color='plum'
                    ),
                    row=4, col=2
                )
        
        fig.update_layout(
            height=1600,
            title_text="🚀 DeepSeek 模型发布预测 - 全面分析报告",
            showlegend=True
        )
        
        fig.write_html("deepseek_modular_analysis.html")
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
        self.generate_all_predictions()
        
        # 5. 性能分析
        self.analyze_performance()
        
        # 6. 综合分析
        analysis_summary = self.create_comprehensive_analysis()
        
        # 7. 打印详细结果
        self.print_detailed_results()
        
        # 8. 创建可视化
        self.create_advanced_visualizations()
        
        print("\n🎉 分析完成!")
        return {
            'predictions': self.predictions,
            'performance': self.performance_summary,
            'summary': analysis_summary
        }


def main():
    """主函数"""
    predictor = DeepSeekPredictorModular()
    results = predictor.run_complete_analysis()
    return predictor, results


if __name__ == "__main__":
    predictor, results = main() 