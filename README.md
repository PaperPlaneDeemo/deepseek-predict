# DeepSeek 模型发布时间预测

这是一个基于机器学习的DeepSeek模型发布时间预测项目，使用多种算法分析历史发布模式，预测未来模型发布时间。

## 🚀 项目特点

- **多种预测方法**: 当前默认包含 12 种预测器（线性模型 / 时间序列 / 间隔分析 / 统计方法）
- **丰富的可视化**: Python 生成的交互式HTML可视化
- **智能分析**: 自动模型性能评估和回测分析

## 📊 数据来源

基于DeepSeek历史发布记录，示例包含 20 个数据点用于分析与回测。

## 🛠️ 安装依赖

使用 uv（推荐）：

```bash
uv venv
uv pip install -r requirements.txt
```

或使用 pip：

```bash
pip install -r requirements.txt
```

## 📈 使用方法

```bash
python3 deepseek_predictor_modular.py
```

运行将：
- 进行数据预处理和特征工程
- 运行多组预测算法
- 生成详细预测结果并保存可视化HTML文件

## 🔧 生成文件

运行后会生成如下主要文件：

- deepseek_modular_analysis.html  # 高级分析与交互式可视化（已在 `.gitignore` 中忽略）
- deepseek_backtest_analysis.html  # 回测结果可视化（已在 `.gitignore` 中忽略）

## 📁 文件结构（简要）

```
deepseek-predict/
├── deepseek_predictor_modular.py     # 主预测脚本（模块化实现）
├── models/                           # 各类预测器实现
├── requirements.txt                  # Python依赖
└── README.md                         # 项目文档
```

## 🔧 自定义配置

可以在 `deepseek_predictor_modular.py` 中修改：
- 预测基准日期
- 预测数量和步长
- 模型超参数与特征工程设置

## 📚 技术栈

- pandas, numpy, scikit-learn, scipy, plotly

---

*本项目仅供学习和研究使用，预测结果不构成投资建议。*
