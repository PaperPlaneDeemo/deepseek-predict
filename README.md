# DeepSeek 模型发布时间预测

这是一个基于机器学习的DeepSeek模型发布时间预测项目，使用多种算法分析历史发布模式，预测未来模型发布时间。

## 🚀 项目特点

- **多种预测方法**: 包含18+种预测算法（模块化实现，深度学习算法可选）
- **丰富的可视化**: Python 生成的交互式HTML可视化
- **智能分析**: 自动模型性能评估和回测分析

## 📊 数据来源

基于DeepSeek历史发布记录，示例包含15个数据点用于分析与回测。

## 🛠️ 安装依赖

```bash
pip install -r requirements.txt
```

## 📈 使用方法

```bash
python deepseek_predictor_modular.py
```

运行将：
- 进行数据预处理和特征工程
- 运行多组预测算法（排除或包含深度学习算法可配置）
- 生成详细预测结果并保存可视化HTML文件

## 🔍 运行示例输出（2025-08-15 当次运行）

- 数据预处理完成，数据形状: (15, 12)
- 初始化并训练 18 个预测器（已排除 MLP 和 LSTM）
- 成功生成 14 个有效预测结果
- 回测分析已生成，回测统计示例：Statistical Ensemble MAE: 21.8，Trend Analysis MAE: 22.2

## 🔧 生成文件

运行后会生成如下主要文件：

- deepseek_modular_analysis.html  # 高级分析与交互式可视化
- deepseek_backtest_analysis.html  # 回测结果可视化

## 📁 文件结构（简要）

```
deepseek-predict/
├── deepseek_predictor_modular.py     # 主预测脚本（模块化实现）
├── requirements.txt                  # Python依赖
├── deepseek_modular_analysis.html    # 运行生成的交互式分析
├── deepseek_backtest_analysis.html   # 回测分析结果
└── README.md                         # 项目文档
```

## 🔧 自定义配置

可以在 `deepseek_predictor_modular.py` 中修改：
- 预测基准日期
- 预测数量和步长
- 是否包含深度学习模型（MLP、LSTM）
- 模型超参数与特征工程设置

## 📚 技术栈

- pandas, numpy, scikit-learn, xgboost, statsmodels, plotly

---

*本项目仅供学习和研究使用，预测结果不构成投资建议。*