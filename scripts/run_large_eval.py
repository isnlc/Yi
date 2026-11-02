# -*- coding: utf-8 -*-
# 实验6：大数据集对照（219 条语料 · 48 题基准）
# 数据来源：对话历史摘记（chat）+ 脱敏资料笔记（note）+ 原始日记（diary）
import sys, os
os.environ["MEMAGENT_LLM"] = "0"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memagent import config
from memagent.evaluate import run_all, print_report
from scripts.run_eval import make_chart


def main():
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dataset = str(config.DATA_DIR / "dataset_large.jsonl")
    benchmark = str(config.EVAL_DIR / "benchmark_large.jsonl")

    import memagent.evaluate as ev
    from memagent.evaluate import load_jsonl
    # 复用 run_all 但换数据集与基准
    results = {}
    bench = load_jsonl(benchmark)
    for arch in ("A", "B", "C"):
        r = ev.run_arch(arch, bench, dataset)
        results[arch] = r
    with open(config.RESULTS_DIR / "large_metrics.json", "w", encoding="utf-8") as f:
        import json
        json.dump(results, f, ensure_ascii=False, indent=1)

    print_report(results)
    chart = make_chart(results, title="三套记忆架构对照实验（48 题基准 · 219 条语料）")
    chart_new = config.RESULTS_DIR / "large_comparison.png"
    chart.replace(chart_new)
    print(f"\n对比图已保存：{chart_new}")
    print(f"明细指标：{config.RESULTS_DIR / 'large_metrics.json'}")


if __name__ == "__main__":
    main()
