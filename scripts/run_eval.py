# -*- coding: utf-8 -*-
# 步骤3：三套架构对照评测 + 生成对比图
import sys, os
os.environ["MEMAGENT_LLM"] = "0"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memagent import config
from memagent.evaluate import run_all, print_report

def make_chart(results, title="三套记忆架构对照实验（36 题基准 · 100 条语料）"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    types = ["recent", "contradiction", "summary", "time", "timeline", "trap"]
    labels = ["近期事实", "矛盾处理", "总结归纳", "时间回溯", "时间线", "诱导拒答"]
    colors = {"A": "#B9C2CE", "B": "#7A9CC6", "C": "#002060"}
    names = {"A": "A 无记忆", "B": "B 扁平记忆", "C": "C 本方案"}

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
    width = 0.26
    x = range(len(types))
    for i, arch in enumerate(("A", "B", "C")):
        vals = [results[arch]["type_acc"].get(t, 0.0) for t in types]
        bars = ax.bar([xi + (i - 1) * width for xi in x], vals, width,
                      label=names[arch], color=colors[arch])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.0%}",
                    ha="center", fontsize=8, color="#424953")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("关键词命中率")
    ax.set_title(title, fontsize=13, color="#002060", pad=12)
    ax.legend(frameon=False, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    out = config.RESULTS_DIR / "comparison.png"
    fig.tight_layout()
    fig.savefig(out)
    return out

def main():
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = run_all()
    print_report(results)
    chart = make_chart(results)
    print(f"\n对比图已保存：{chart}")
    print(f"明细指标：{config.RESULTS_DIR / 'metrics.json'}")

if __name__ == "__main__":
    main()
