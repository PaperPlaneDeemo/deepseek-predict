"""Command-line entry point; computation and rendering are independently usable."""

import argparse
from datetime import date
from pathlib import Path
import sys
import webbrowser

from .analysis import run_analysis
from .data import DEFAULT_DATA_PATH, load_releases, parse_date
from .report import write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="分析 DeepSeek 发布记录并生成可离线打开的 HTML 报告。")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="发布数据 JSON 文件")
    parser.add_argument("--as-of", type=parse_date, default=date.today(), metavar="YYYY-MM-DD", help="分析基准日，默认本机今天；之后的记录不参与训练")
    parser.add_argument("--horizon", type=int, default=3, metavar="N", help="从最后已知发布日起连续预测的事件数（1–100，默认 3）")
    parser.add_argument("--min-train-size", type=int, default=3, metavar="N", help="首个回测训练窗口的发布事件数（至少 2，默认 3）")
    parser.add_argument("--output", type=Path, default=Path("output"), metavar="DIR", help="报告输出目录，默认 ./output")
    parser.add_argument("--validate-data", action="store_true", help="仅校验数据文件并显示记录数")
    parser.add_argument("--open", action="store_true", help="生成后在默认浏览器打开报告")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        releases = load_releases(args.data)
        if args.validate_data:
            print(f"数据校验通过：{len(releases)} 条模型记录，{len({item.date for item in releases})} 个发布日。")
            return 0
        report = run_analysis(releases, as_of=args.as_of, n_predictions=args.horizon,
                              min_train_size=args.min_train_size, dataset_path=str(args.data.resolve()))
        html_path, json_path = write_report(report, args.output)
    except (ValueError, OSError) as error:
        parser.error(str(error))

    meta, summary = report["meta"], report["summary"]
    successful = sum(item["status"] == "ok" for item in report["forecasts"])
    print(f"基准日 {meta['as_of']} · {meta['release_count']} 条模型记录 / {meta['event_count']} 个发布日")
    print(f"预测完成 {successful}/{len(report['forecasts'])} 个方法 · 回测 {report['backtest']['total_folds']} 折")
    if summary["best_method"]:
        print(f"回测 MAE 最低：{summary['best_method']} · {summary['best_mae']:.2f} 天")
    if summary["median_next_date"]:
        print(f"下一发布日估计中位数：{summary['median_next_date']}（估计逾期时不自动顺延）")
    for warning in report["warnings"]:
        print(f"说明：{warning}")
    print(f"HTML: {html_path}\nJSON: {json_path}")
    if args.open:
        webbrowser.open(html_path.as_uri())
    if not successful:
        print("所有方法预测失败；错误详情已写入报告。", file=sys.stderr)
        return 1
    return 0
