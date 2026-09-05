# DeepSeek 发布节奏观察

以可维护的发布目录为输入，比较 12 种方法对**下一次模型发布日**的估计，并用逐步扩展训练窗口的回测检查误差。输出一个可直接离线打开的交互报告和一份 JSON 结果。

## 快速开始

需要 Python 3.10+。推荐用 `uv.lock` 复现依赖：

```bash
uv sync --locked
uv run python -m deepseek_predict --as-of 2026-09-05 --open
```

也可以安装已导出的锁定依赖：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m deepseek_predict --as-of 2026-09-05
```

默认生成 `output/index.html` 和 `output/report.json`。HTML 内嵌样式、脚本与数据，不依赖服务器、CDN 或联网；双击即可打开。报告包含概览、方法对比、可搜索的发布记录和逐折回测，支持方法类别筛选、排序、方法联动和 JSON 下载。

```bash
# 默认使用本机今天作为基准日；从最后已观测发布日预测接下来 3 次事件
uv run python -m deepseek_predict

# 指定数据、预测数量、回测起点及输出位置
uv run python -m deepseek_predict \
  --data data/releases.json --as-of 2026-09-05 \
  --horizon 5 --min-train-size 5 --output output/custom

# 仅校验目录，允许不足以预测的小型目录
uv run python -m deepseek_predict --validate-data

# 原命令仍可用，转发到同一 CLI
uv run python deepseek_predictor_modular.py --as-of 2026-09-05
```

`--horizon` 为 1–100；`--min-train-size` 至少为 2。少于两个已知发布日时无法预测；没有足够回测折次时仍可生成报告，但不评选最佳方法。默认数据路径相对于仓库解析，输出目录相对于执行命令的工作目录。

## 维护发布记录

唯一的数据源是 [`data/releases.json`](data/releases.json)。每个模型的名称与发布日期存于同一条记录，不再维护平行列表：

```json
{
  "schema_version": 1,
  "notes": "数据集备注",
  "releases": [
    {
      "id": "example-model",
      "name": "Example Model",
      "date": "2026-09-01",
      "source_url": null,
      "notes": "来源待补充"
    }
  ]
}
```

- `id` 是稳定的小写 slug，修改名称或日期时保留 ID。
- `date` 必须为有效的 `YYYY-MM-DD` 日历日期；不从型号后缀推导日期。
- `source_url` 必须为 HTTP(S) 链接或 `null`。加载程序只验证格式，不验证来源事实。
- `notes` 用于记录来源说明、日期歧义或更正理由。
- 重复 ID、重复名称与日期组合、缺失字段、拼错字段和未知 schema 会报错，并指出记录索引。
- 文件内顺序不影响结果。同日的不同模型都保留在目录中，训练时合并为一个发布事件，避免人为制造零天间隔。

现有 **24 条记录是原仓库数据的原样迁移，尚未独立核实**，包括原工作区新增的 3 条记录。`DeepSeek-V4-Pro-0813` 的记录日期仍为 `2026-08-12`；名称后缀与日期不一致没有被自动更正。请按可核实的来源更新数据。

## 如何读结果

**预测语义。** 所有方法从最后已观测的发布日起预测连续事件。若第一条估计已早于基准日，页面显示“已逾期”，不会跳过该预测或凭空假设期间发生了发布。后续预测只是递推情景，不是条件于“截至基准日仍未发布”的生存模型。估计中位数向下取整到日，方法之间的日期范围表示分歧，不是置信区间。

**基准日。** `--as-of` 会从训练、统计与回测中排除更晚的记录，目录仍展示这些记录及其排除状态。这是按发布日期截断的历史模拟；目录未记录每条资料何时被知悉，因此不宣称重建当时可获得的完整信息。

**回测。** 默认用最早 3 个发布事件训练，预测第 4 个；再用前 4 个预测第 5 个，依次推进。每个方法、每折都新建模型，只访问训练前缀。比较的是下一次发布的日期，误差为 `预测日期 − 实际日期`；负数表示预测偏早。回测只评估一步预测，多步结果没有同等的验证证据。

**排名。** MAE 和 RMSE 单位为天，偏差为带符号的平均误差。±30 天命中率以全部尝试为分母，失败算未命中；覆盖率为成功预测折数除以总折数。仅覆盖全部折次且当前预测成功的方法参与方法表 MAE 排名及概览最佳方法评选。部分成功方法的误差只描述其成功折次，页面会标明未参与排名。无折次的指标为 `null`，而非零误差。网页只显示样本外回测指标，模型内部拟合诊断不用于排名。

24 条样本混合了不同模型系列，方法数量多于独立历史周期。最佳回测成绩是在同一组折次上选出的描述性结果，不是独立测试集上的未来表现保证。

## 实现结构

```text
data/releases.json           # 发布目录：模型、日期、来源
deepseek_predict/
  data.py                    # 校验、时间截断、事件与特征派生
  registry.py                # 12 种方法的单一注册表
  analysis.py                # 预测、逐折回测、严格 JSON 数据协议
  report.py                  # 安全嵌入数据、生成离线产物
  cli.py                     # 参数与终端结果
models/                      # 预测方法与统一输入/输出约束
web/report.html              # 文档结构
web/report.css               # 响应式样式
web/report.js                # 原生 SVG 与交互
tests/                       # 数据、预测器、回测、报告与 CLI 回归测试
```

新增预测方法只需实现 `BasePredictor` 并加入 `registry.py`。报告 JSON schema 独立于数据目录 schema，当前版本均为 1。计算模块不写文件，展示模块不训练模型，前端只读取生成时的快照；修改源数据或基准日后需重新运行命令。

旧的两份 Plotly HTML、Python 图表拼接实现及 Plotly 依赖已移除。旧脚本保留为命令入口，旧 `DeepSeekPredictorModular` 类接口不再提供。

## 验证与开发

```bash
uv run --locked python -m unittest discover -s tests -v
uv run --locked python -m deepseek_predict --as-of 2026-09-05
node --check web/report.js
```

测试覆盖目录迁移、同日事件、历史截断、预测逾期、特征泄漏、回测失败分母、异常输出、JSON 安全嵌入和跨目录 CLI。CI 在 Python 3.10 与 3.13 执行测试和完整报告生成。

依赖以 `pyproject.toml` / `uv.lock` 为准，变更依赖后同步 pip 导出：

```bash
uv lock
uv export --format requirements-txt --no-hashes --no-annotate --no-header --output-file requirements.txt
```
