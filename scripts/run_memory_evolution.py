# -*- coding: utf-8 -*-
# 实验5：记忆演化可视化——记忆库随时间的主题变迁 + 遗忘决策分布
# 答辩开场图：一眼看懂"记忆怎么进来、怎么沉淀、怎么遗忘"
import sys, os, json
from collections import Counter
os.environ["MEMAGENT_LLM"] = "0"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memagent import config
from memagent.evaluate import build_agent


def make_chart(store):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import date
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    topics = ["健身", "运动", "饮食", "作息", "课程", "社交", "生活"]
    colors = plt.cm.tab10(range(len(topics)))

    # ---- 上图：每月新记忆条数（按主题堆叠）----
    months = sorted({m.date[:7] for m in store.memories})
    series = {t: [0] * len(months) for t in topics}
    for m in store.memories:
        if m.topic in series:
            series[m.topic][months.index(m.date[:7])] += 1

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), dpi=150)
    bottom = [0] * len(months)
    for t, c in zip(topics, colors):
        vals = series[t]
        ax1.bar(months, vals, bottom=bottom, label=t, color=c, width=0.6)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax1.set_title("记忆入库：每月新增记忆条数（按主题）", fontsize=13, color="#002060", pad=10)
    ax1.legend(ncol=7, fontsize=8, frameon=False, loc="upper left")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.set_ylabel("条数")

    # ---- 下图：记忆现状——状态与主题分布（保留 vs 沉淀 vs 遗忘）----
    status_label = {"active": "活跃层", "superseded": "被取代", "compressed": "压缩归档", "discarded": "主动遗忘"}
    status_order = ["active", "superseded", "compressed", "discarded"]
    status_colors = {"active": "#002060", "superseded": "#7A9CC6", "compressed": "#B9C2CE", "discarded": "#D8DEE6"}
    counts = Counter(m.topic for m in store.memories if m.status == "active")
    left = 0
    by_status = {}
    for st in status_order:
        items = [m for m in store.memories if m.status == st]
        by_status[st] = len(items)
    total = sum(by_status.values())
    for st in status_order:
        n = by_status[st]
        ax2.barh([0], [n], left=left, color=status_colors[st], height=0.5, label=f"{status_label[st]} {n}")
        if n >= 3:
            ax2.text(left + n / 2, 0, f"{status_label[st]}\n{n}", ha="center", va="center",
                     fontsize=9, color="#1a1a2e" if st == "discarded" else "white")
        left += n
    ax2.set_xlim(0, total)
    ax2.set_title(f"记忆现状：共 {total} 条——活跃 {by_status['active']} 条承担日常检索，"
                  f"其余 {total - by_status['active']} 条沉淀入档案层可回溯", fontsize=12, color="#002060", pad=10)
    ax2.legend(ncol=4, fontsize=8, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.5))
    ax2.set_yticks([])
    ax2.spines[["top", "right", "left", "bottom"]].set_visible(False)

    out = config.RESULTS_DIR / "memory_evolution.png"
    fig.tight_layout()
    fig.savefig(out)
    return out


def main():
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    agent = build_agent("C", dataset=str(config.DATA_DIR / "dataset_large.jsonl"))
    out = make_chart(agent.store)
    print(f"记忆演化图已保存：{out}")


if __name__ == "__main__":
    main()
