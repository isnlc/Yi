# -*- coding: utf-8 -*-
# 实验3：三路检索消融
# 主口径与汇报一致：219 条记录、48 题，离线确定性运行。
import sys, os, json, copy
os.environ["MEMAGENT_LLM"] = "0"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memagent import config
from memagent.evaluate import load_jsonl, run_arch

LARGE = str(config.DATA_DIR / "dataset_large.jsonl")
BENCHMARK = config.EVAL_DIR / "benchmark_large.jsonl"

CONFIGS = [
    ("lex_only",  {"lex": True,  "time": False, "graph": False}, "只词法 BM25"),
    ("lex_time",  {"lex": True,  "time": True,  "graph": False}, "词法+时间衰减"),
    ("lex_graph", {"lex": True,  "time": False, "graph": True},  "词法+主题图"),
    ("full_rrf",  {"lex": True,  "time": True,  "graph": True},  "三路 RRF 全开（本方案）"),
]


def run_ablation(benchmark: list[dict]) -> list[dict]:
    saved = copy.deepcopy(config.RETRIEVE_CHANNELS)
    rows = []
    try:
        for key, channels, label in CONFIGS:
            config.RETRIEVE_CHANNELS.clear()
            config.RETRIEVE_CHANNELS.update(channels)
            r = run_arch("C", benchmark, dataset=LARGE)
            rows.append({
                "key": key, "label": label, "channels": dict(channels),
                "overall": r["overall"], "type_acc": r["type_acc"],
                "memory_size": r["memory_size"],
            })
            print(f"{label:<18} 综合 {r['overall']:.1%}")
    finally:
        config.RETRIEVE_CHANNELS.clear()
        config.RETRIEVE_CHANNELS.update(saved)
    with open(config.RESULTS_DIR / "ablation.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    return rows


def make_chart(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    types = ["recent", "contradiction", "summary", "time", "timeline", "trap"]
    labels = ["近期事实", "矛盾处理", "总结归纳", "时间回溯", "时间线", "诱导拒答"]
    colors = ["#B9C2CE", "#7A9CC6", "#5B7DA8", "#002060"]

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
    width = 0.2
    x = range(len(types))
    for i, row in enumerate(rows):
        vals = [row["type_acc"].get(t, 0.0) for t in types]
        bars = ax.bar([xi + (i - 1.5) * width for xi in x], vals, width,
                      label=row["label"], color=colors[i])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.0%}",
                    ha="center", fontsize=7, color="#424953")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("关键词命中率")
    ax.set_title("三路检索消融（48 题 · 219 条记录）", fontsize=13, color="#002060", pad=12)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    out = config.RESULTS_DIR / "ablation.png"
    fig.tight_layout()
    fig.savefig(out)
    return out


def main():
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = run_ablation(load_jsonl(BENCHMARK))
    print("\n消融结论：")
    full = next(r for r in rows if r["key"] == "full_rrf")
    for r in rows:
        if r["key"] != "full_rrf":
            delta = full["overall"] - r["overall"]
            print(f"  去掉{('时间衰减' if r['key']=='lex_time' else '主题图' if r['key']=='lex_graph' else '时间与主题图')}："
                  f"综合下降 {delta:+.1%}")
    chart = make_chart(rows)
    print(f"\n消融图已保存：{chart}")
    print(f"明细指标：{config.RESULTS_DIR / 'ablation.json'}")


if __name__ == "__main__":
    main()
