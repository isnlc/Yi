# -*- coding: utf-8 -*-
"""主证据实验：在 dataset_large 的 219 条记录和 48 题上，
量化任务分工、检索通道和关键机制的影响。全部关闭生成模型调用运行。

产出 eval/results/large_evidence.json：
  1. abc            —— 三架构对照（A 无记忆 / B 只存不整理 / C 本方案）
  2. multi_vs_single—— 完整任务分工 vs 关闭独立冲突治理
  3. ablation_channels —— 三路检索消融（BM25 / 时间衰减+日历 / 主题图）
  4. ablation_opts  —— 逐项关闭后期优化：日历召回 / 档案层回溯 / ReAct 追问轮
  5. log_excerpt    —— 网页后端一次问答的处理日志摘录
"""
import copy
import json
import os
import sys

os.environ["MEMAGENT_LLM"] = "0"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memagent import config
from memagent import retrieve as retrieve_mod
from memagent.agents import MemoryAgent
from memagent.evaluate import load_jsonl, run_arch, score_agent, summarize
from memagent.store import MemoryStore
from tools.audit_tool import append_event

LARGE = str(config.DATA_DIR / "dataset_large.jsonl")
BENCH = load_jsonl(config.EVAL_DIR / "benchmark_large.jsonl")
OUT = config.RESULTS_DIR / "large_evidence.json"


def build_c(store_path: str) -> MemoryAgent:
    if os.path.exists(store_path):
        os.remove(store_path)   # 每轮重建，防止加载上一轮遗留库导致重复灌入
    store = MemoryStore(path=store_path)
    agent = MemoryAgent(store, with_reconciler=True)
    for e in load_jsonl(LARGE):
        agent.ingest(e["date"], e["text"], e.get("source", "diary"))
    agent.maintain()
    return agent


def eval_agent(agent, extra=None) -> dict:
    per_type, details = score_agent(agent, BENCH)
    st = agent.store.stats()
    by = st["memories_by_status"]
    ex = {"memory_size": by.get("active", 0),
          "memory_archive": by.get("superseded", 0) + by.get("compressed", 0) + by.get("discarded", 0),
          "conflicts": st["conflicts"], "conflicts_resolved": st["conflicts_resolved"]}
    if extra:
        ex.update(extra)
    return summarize(per_type, details, ex.pop("memory_size"), ex)


def section_multi_vs_single() -> dict:
    out = {}
    for mode, reconciler in (("single", False), ("multi", True)):
        store = MemoryStore(path=str(config.RESULTS_DIR / f"_ev_{mode}.json"))
        if store.path and os.path.exists(store.path):
            os.remove(store.path)
        store = MemoryStore(path=str(config.RESULTS_DIR / f"_ev_{mode}.json"))
        agent = MemoryAgent(store, with_reconciler=reconciler)
        for e in load_jsonl(LARGE):
            agent.ingest(e["date"], e["text"], e.get("source", "diary"))
        agent.maintain()
        out[mode] = eval_agent(agent)
        out[mode].pop("details", None)
        print(f"[multi_vs_single] {mode}: {out[mode]['overall']:.1%} "
              f"冲突 {out[mode]['conflicts']} 解决 {out[mode]['conflicts_resolved']}")
    return out


def section_ablation_channels() -> list:
    saved = copy.deepcopy(config.RETRIEVE_CHANNELS)
    rows = []
    try:
        for key, channels, label in [
            ("lex_only",  {"lex": True, "time": False, "graph": False}, "只词法 BM25"),
            ("lex_time",  {"lex": True, "time": True, "graph": False}, "词法+时间衰减"),
            ("lex_graph", {"lex": True, "time": False, "graph": True}, "词法+主题图"),
            ("full_rrf",  {"lex": True, "time": True, "graph": True}, "三路 RRF 全开（本方案）"),
        ]:
            config.RETRIEVE_CHANNELS.clear()
            config.RETRIEVE_CHANNELS.update(channels)
            r = run_arch("C", BENCH, dataset=LARGE)
            r.pop("details", None)
            rows.append({"key": key, "label": label, "overall": r["overall"],
                         "type_acc": r["type_acc"], "memory_size": r["memory_size"]})
            print(f"[channels] {label}: {r['overall']:.1%}")
    finally:
        config.RETRIEVE_CHANNELS.clear()
        config.RETRIEVE_CHANNELS.update(saved)
    return rows


def section_ablation_opts() -> list:
    saved_calendar_spans = retrieve_mod.calendar_spans
    saved_archive = config.ALLOW_ARCHIVE
    saved_rounds = config.MAX_QUERY_ROUNDS
    rows = []
    variants = [
        ("full", "本方案（全开）"),
        ("no_calendar", "关闭日历窗口召回"),
        ("no_archive", "关闭档案层回溯"),
        ("no_refine", "关闭 ReAct 追问轮（单轮检索）"),
    ]
    try:
        for key, label in variants:
            retrieve_mod.calendar_spans = saved_calendar_spans
            config.ALLOW_ARCHIVE = saved_archive
            config.MAX_QUERY_ROUNDS = saved_rounds
            if key == "no_calendar":
                retrieve_mod.calendar_spans = lambda *_args, **_kwargs: []
            if key == "no_archive":
                config.ALLOW_ARCHIVE = False
            if key == "no_refine":
                config.MAX_QUERY_ROUNDS = 1
            agent = build_c(str(config.RESULTS_DIR / "_ev_opt.json"))
            r = eval_agent(agent)
            r.pop("details", None)
            rows.append({"key": key, "label": label, "overall": r["overall"],
                         "type_acc": r["type_acc"]})
            print(f"[opts] {label}: {r['overall']:.1%}")
    finally:
        retrieve_mod.calendar_spans = saved_calendar_spans
        config.ALLOW_ARCHIVE = saved_archive
        config.MAX_QUERY_ROUNDS = saved_rounds
    return rows


def section_log_excerpt() -> list:
    """网页后端同款调用链（MemoryPipeline.ask）的真实留痕摘录。"""
    from pipeline import MemoryPipeline
    from pathlib import Path
    store_p = config.RESULTS_DIR / "_ev_log.json"
    log_p = config.RESULTS_DIR / "_ev_log.jsonl"
    for f in (store_p, log_p):
        if f.exists():
            f.unlink()
    p = MemoryPipeline(store_p, log_p)
    p.ingest(load_jsonl(LARGE))
    p.ask("我现在还在坚持跑步吗？")
    p.ask("我国庆做了什么？")
    events = [json.loads(line) for line in log_p.open(encoding="utf-8") if line.strip()]
    keep = {"ingest_start", "maintenance_complete", "query", "query_round",
            "reconciler_decision", "retention_decision"}
    return [e for e in events if e.get("event") in keep][-25:]


def main():
    print("== 1/5 三架构对照 ==")
    abc = {a: {k: v for k, v in run_arch(a, BENCH, dataset=LARGE).items() if k != "details"}
           for a in ("A", "B", "C")}
    for a, r in abc.items():
        print(f"[abc] {a}: {r['overall']:.1%}")
    print("== 2/5 完整任务分工 vs 单体编排 ==")
    multi = section_multi_vs_single()
    print("== 3/5 三路检索消融 ==")
    channels = section_ablation_channels()
    print("== 4/5 优化项消融 ==")
    opts = section_ablation_opts()
    print("== 5/5 决策日志摘录 ==")
    excerpt = section_log_excerpt()

    data = {"dataset": os.path.basename(LARGE), "questions": len(BENCH),
            "abc": abc, "multi_vs_single": multi,
            "ablation_channels": channels, "ablation_opts": opts,
            "log_excerpt": excerpt}
    with OUT.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print(f"\n已保存：{OUT}")


if __name__ == "__main__":
    main()
