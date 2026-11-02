# -*- coding: utf-8 -*-
# 实验1：遗忘阈值扫描 —— 遗忘是损失还是收益？
# 扫描 discard_below 阈值，画"准确率-存储量"随阈值变化的曲线；
# 并做消融对照（关闭档案层），验证"安全遗忘"是档案层的功劳
import sys, os
os.environ["MEMAGENT_LLM"] = "0"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memagent import config
from memagent.evaluate import load_jsonl, run_forget_sweep

def make_chart(rows, rows_abl):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    xs = [r["threshold"] for r in rows]
    acc = [r["overall"] for r in rows]
    mem = [r["active"] for r in rows]
    xs2 = [r["threshold"] for r in rows_abl]
    acc2 = [r["overall"] for r in rows_abl]

    fig, ax1 = plt.subplots(figsize=(8.6, 4.6), dpi=150)
    ax1.plot(xs, acc, "o-", color="#002060", lw=2, label="本方案（含档案层）")
    ax1.plot(xs2, acc2, "s--", color="#C0504D", lw=1.8, label="消融：关闭档案层")
    for x, y in zip(xs, acc):
        ax1.annotate(f"{y:.0%}", (x, y), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=8, color="#002060")
    for x, y in zip(xs2, acc2):
        ax1.annotate(f"{y:.0%}", (x, y), textcoords="offset points", xytext=(0, -14),
                     ha="center", fontsize=8, color="#C0504D")
    ax1.set_xlabel("遗忘阈值 discard_below（越大忘得越多）")
    ax1.set_ylabel("20 题基准综合命中率", color="#002060")
    ax1.set_ylim(0.4, 1.08)
    ax1.spines[["top"]].set_visible(False)

    ax2 = ax1.twinx()
    ax2.bar(xs, mem, width=0.012, color="#7A9CC6", alpha=0.45, label="活跃记忆条数")
    ax2.set_ylabel("活跃记忆条数", color="#5A7BA6")
    ax2.spines[["top"]].set_visible(False)

    fig.suptitle("遗忘阈值扫描：档案层让遗忘变安全（消融对照）", fontsize=13, color="#002060")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=9, loc="lower left")
    out = config.RESULTS_DIR / "forget_curve.png"
    fig.tight_layout()
    fig.savefig(out)
    return out

def main():
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    thresholds = [0.0, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    bench = load_jsonl(config.BENCHMARK_FILE)
    rows = run_forget_sweep(thresholds, bench)
    rows_abl = run_forget_sweep(thresholds, bench, no_archive=True)

    for tag, rs in (("本方案", rows), ("消融·无档案层", rows_abl)):
        print(f"----- {tag} -----")
        print(f"{'阈值':>6}{'综合命中率':>12}{'活跃记忆':>10}{'已遗忘':>10}")
        for r in rs:
            print(f"{r['threshold']:>6.2f}{r['overall']:>12.1%}{r['active']:>10}{r['discarded']:>10}")
    full = [r for r in rows if r["threshold"] == 0.0][0]
    print(f"\n本方案在当前阈值区间的综合命中率为 {rows[0]['overall']:.1%}；"
          f"阈值 0.00 时活跃记忆 {full['active']} 条，"
          f"阈值 0.75 时活跃记忆 {rows[-1]['active']} 条。")
    chart = make_chart(rows, rows_abl)
    print(f"曲线图已保存：{chart}")

if __name__ == "__main__":
    main()
