# -*- coding: utf-8 -*-
# 实验2：单体编排 vs 四项任务分工
# 主口径与汇报一致：219 条记录、48 题，离线确定性运行。
import sys, os
os.environ["MEMAGENT_LLM"] = "0"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memagent import config
from scripts.run_large_evidence import BENCH, LARGE, section_multi_vs_single
import json

def make_chart(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    types = ["recent", "contradiction", "summary", "time"]
    labels = ["近期事实", "矛盾处理", "总结归纳", "时间回溯"]
    colors = {"single": "#7A9CC6", "multi": "#002060"}
    names = {"single": "单体编排", "multi": "四项任务分工"}

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    width = 0.32
    x = range(len(types))
    for i, mode in enumerate(("single", "multi")):
        vals = [results[mode]["type_acc"].get(t, 0.0) for t in types]
        bars = ax.bar([xi + (i - 0.5) * width for xi in x], vals, width,
                      label=names[mode], color=colors[mode])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.0%}",
                    ha="center", fontsize=9, color="#424953")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("关键词命中率")
    ax.set_title("任务编排对照（48 题 · 219 条记录）", fontsize=13, color="#002060", pad=12)
    ax.legend(frameon=False, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    out = config.RESULTS_DIR / "agent_compare.png"
    fig.tight_layout()
    fig.savefig(out)
    return out

def main():
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = section_multi_vs_single()
    with open(config.RESULTS_DIR / "agent_compare.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": os.path.basename(LARGE),
                "questions": len(BENCH),
                "single": results["single"],
                "multi": results["multi"],
            },
            f,
            ensure_ascii=False,
            indent=1,
        )
    for mode, label in (("single", "单体编排"), ("multi", "四项任务分工")):
        r = results[mode]
        ta = r["type_acc"]
        print(f"{label}：综合 {r['overall']:.1%} | 近期 {ta.get('recent',0):.0%} "
              f"矛盾 {ta.get('contradiction',0):.0%} 总结 {ta.get('summary',0):.0%} "
              f"时间 {ta.get('time',0):.0%} | 冲突账本 {r['conflicts']} / "
              f"已解决 {r['conflicts_resolved']} | 活跃记忆 {r['memory_size']}")
    chart = make_chart(results)
    print(f"\n对比图已保存：{chart}")

if __name__ == "__main__":
    main()
