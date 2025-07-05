"""
DeepSeek 回测分析脚本
专门用于运行回测分析，比较R²和回测表现的关系
"""

from deepseek_predictor_modular import DeepSeekPredictorModular
import pandas as pd
import numpy as np


def run_backtest_only():
    """仅运行回测分析"""
    print("🎯 DeepSeek 回测分析")
    print("=" * 60)
    print("📋 本次分析将:")
    print("   1. 排除深度学习算法 (MLP, LSTM)")
    print("   2. 从第3个数据点开始逐步回测")
    print("   3. 比较R²和回测表现的关系")
    print("   4. 生成回测可视化报告")
    print("=" * 60)
    
    # 创建预测器实例
    predictor = DeepSeekPredictorModular()
    
    # 数据准备
    predictor._prepare_data()
    
    # 初始化预测器（已排除深度学习）
    predictor._initialize_predictors()
    
    # 先训练所有模型（用于获取R²）
    print("\n📊 训练模型以获取R²指标...")
    predictor.fit_all_models()
    
    # 运行回测
    backtest_df = predictor.run_backtest(start_from=3, verbose=True)
    
    if backtest_df is not None:
        print(f"\n✅ 回测完成！共有 {len(backtest_df)} 个方法参与回测")
        
        # 比较R²和回测结果
        comparison_df = predictor.compare_r2_vs_backtest()
        
        # 创建回测可视化
        predictor.create_backtest_visualization()
        
        # 保存详细结果
        if comparison_df is not None:
            comparison_df.to_csv('r2_vs_backtest_comparison.csv', index=False)
            print("📄 详细对比结果已保存到 r2_vs_backtest_comparison.csv")
        
        backtest_df.to_csv('backtest_results.csv', index=False)
        print("📄 回测结果已保存到 backtest_results.csv")
        
        return predictor, backtest_df, comparison_df
    else:
        print("❌ 回测失败")
        return None, None, None


def analyze_best_methods(predictor, backtest_df, comparison_df):
    """分析最佳方法"""
    print("\n🔍 深度分析最佳方法")
    print("=" * 50)
    
    if backtest_df is None or comparison_df is None:
        print("❌ 没有足够的数据进行分析")
        return
    
    # 最佳回测方法
    best_mae = backtest_df.iloc[0]
    print(f"🏆 最佳回测方法 (MAE): {best_mae['Method']}")
    print(f"   MAE: {best_mae['MAE (days)']:.1f} 天")
    print(f"   RMSE: {best_mae['RMSE (days)']:.1f} 天")
    print(f"   成功率: {best_mae['Success_Rate']:.1%}")
    
    # 最佳成功率方法
    best_success = backtest_df.loc[backtest_df['Success_Rate'].idxmax()]
    print(f"\n🎯 最佳成功率方法: {best_success['Method']}")
    print(f"   成功率: {best_success['Success_Rate']:.1%}")
    print(f"   MAE: {best_success['MAE (days)']:.1f} 天")
    
    # 最稳定的方法（RMSE相对于MAE最小）
    backtest_df['Stability_Ratio'] = backtest_df['RMSE (days)'] / backtest_df['MAE (days)']
    most_stable = backtest_df.loc[backtest_df['Stability_Ratio'].idxmin()]
    print(f"\n⚖️  最稳定方法: {most_stable['Method']}")
    print(f"   稳定性比值: {most_stable['Stability_Ratio']:.2f}")
    print(f"   MAE: {most_stable['MAE (days)']:.1f} 天")
    
    # 综合评分
    backtest_df['Combined_Score'] = (
        (1 / (1 + backtest_df['MAE (days)'] / 50)) * 0.4 +  # 准确性权重40%
        backtest_df['Success_Rate'] * 0.4 +                 # 成功率权重40%
        (1 / backtest_df['Stability_Ratio']) * 0.2          # 稳定性权重20%
    )
    
    best_combined = backtest_df.loc[backtest_df['Combined_Score'].idxmax()]
    print(f"\n🏅 综合最佳方法: {best_combined['Method']}")
    print(f"   综合评分: {best_combined['Combined_Score']:.3f}")
    print(f"   MAE: {best_combined['MAE (days)']:.1f} 天")
    print(f"   成功率: {best_combined['Success_Rate']:.1%}")
    
    # 推荐使用的方法
    print(f"\n💡 推荐使用的方法组合:")
    print("-" * 30)
    
    # 选择MAE < 60天且成功率 > 50%的方法
    good_methods = backtest_df[
        (backtest_df['MAE (days)'] < 60) & 
        (backtest_df['Success_Rate'] > 0.5)
    ].head(3)
    
    if not good_methods.empty:
        for i, (_, row) in enumerate(good_methods.iterrows(), 1):
            print(f"   {i}. {row['Method']}")
            print(f"      MAE: {row['MAE (days)']:.1f}天, 成功率: {row['Success_Rate']:.1%}")
    else:
        print("   建议使用MAE最小的前3个方法")


def main():
    """主函数"""
    predictor, backtest_df, comparison_df = run_backtest_only()
    
    if predictor is not None:
        analyze_best_methods(predictor, backtest_df, comparison_df)
        
        print(f"\n📋 生成的文件:")
        print("   1. deepseek_backtest_analysis.html - 回测可视化报告")
        print("   2. backtest_results.csv - 回测结果详情")
        print("   3. r2_vs_backtest_comparison.csv - R²与回测对比")
        
        return predictor
    else:
        print("❌ 分析失败")
        return None


if __name__ == "__main__":
    main() 