# 评测：三套架构对照 + 遗忘阈值扫描 + 单/多智能体维护对比
import copy
import json
import os

from . import config
from .store import MemoryStore
from .agents import MemoryAgent, FlatAgent, BaselineAgent


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def _reset_store(name: str) -> MemoryStore:
    path = str(config.RESULTS_DIR / f"store_{name}.json")
    if os.path.exists(path):
        os.remove(path)   # 每轮评测重置，防止重复灌入
    return MemoryStore(path=path)


def build_agent(arch: str, dataset: str = None):
    """每种架构独立建库、独立灌入同一份数据，保证对照公平。
    dataset 缺省用 sample_diary（100 条），可传 dataset_large（219 条）。"""
    entries = load_jsonl(dataset or config.DIARY_FILE)
    store = _reset_store(arch)
    if arch == "A":
        agent = BaselineAgent(store)
    elif arch == "B":
        agent = FlatAgent(store)
    else:
        agent = MemoryAgent(store, with_reconciler=True)
    for e in entries:
        agent.ingest(e["date"], e["text"], e.get("source", "diary"))
    agent.maintain()
    return agent


def score_answer(answer_text: str, keywords: list[str]) -> float:
    hit = sum(1 for k in keywords if k in answer_text)
    return hit / max(len(keywords), 1)


def score_question(q: dict, answer_text: str) -> float:
    """矛盾类问题从严判分：必须采用最新说法，且（有冲突提醒 或 没把过时说法当现状端出来）。"""
    if q["type"] == "contradiction":
        winner = score_answer(answer_text, q["keywords"])
        if winner < 0.5:
            return 0.0
        has_reminder = "以最新的为准" in answer_text
        stale = any(k in answer_text for k in q.get("stale_keywords", []))
        change_context = any(marker in answer_text for marker in (
            "改为", "改成", "改回", "改选", "恢复", "重新开始",
            "不再", "放弃", "停止", "当前", "最近一次明确记录",
        ))
        return 1.0 if (has_reminder or not stale or change_context) else 0.5
    if q["type"] == "trap":
        # 诱导问题：记录里没有的答案，正确的表现是拒答，而不是硬编
        refusal = any(m in answer_text
                      for m in ("没有", "无法", "记不", "还不能", "不足以", "找不到"))
        return 1.0 if refusal else 0.0
    return score_answer(answer_text, q["keywords"])


def score_agent(agent, benchmark: list[dict]):
    per_type: dict[str, list[float]] = {}
    details = []
    for q in benchmark:
        out = agent.answer(q["question"])
        s = score_question(q, out["answer"])
        per_type.setdefault(q["type"], []).append(s)
        details.append({"id": q["id"], "type": q["type"], "question": q["question"],
                        "score": round(s, 2), "answer": out["answer"][:160]})
    return per_type, details


def summarize(per_type: dict, details: list, mem_size, extra=None):
    type_acc = {t: round(sum(v) / len(v), 3) for t, v in per_type.items()}
    overall = round(sum(sum(v) for v in per_type.values())
                    / sum(len(v) for v in per_type.values()), 3)
    out = {"type_acc": type_acc, "overall": overall,
           "memory_size": mem_size, "details": details}
    if extra:
        out.update(extra)
    return out


def run_arch(arch: str, benchmark: list[dict], dataset: str = None) -> dict:
    agent = build_agent(arch, dataset)
    per_type, details = score_agent(agent, benchmark)
    mem_size = {"A": 0, "B": len(agent.store.entries)}.get(arch)
    extra = {}
    if arch == "C":
        st = agent.store.stats()
        mem_size = st["memories_by_status"].get("active", 0)
        extra = {"memory_archive": st["memories_by_status"].get("superseded", 0)
                 + st["memories_by_status"].get("compressed", 0)}
    r = summarize(per_type, details, mem_size, extra)
    r["arch"] = arch
    return r


def run_all() -> dict:
    benchmark = load_jsonl(config.BENCHMARK_FILE)
    results = {arch: run_arch(arch, benchmark) for arch in ("A", "B", "C")}
    with open(config.RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    return results


# ---------- 遗忘阈值扫描：回应"遗忘是损失还是收益" ----------
def run_forget_sweep(thresholds: list[float], benchmark: list[dict], no_archive: bool = False) -> list[dict]:
    """no_archive=True 为消融模式：关掉档案层，验证'安全遗忘'是档案层的功劳。"""
    saved_r = copy.deepcopy(config.RETENTION)
    saved_a = config.ALLOW_ARCHIVE
    rows = []
    try:
        config.ALLOW_ARCHIVE = not no_archive
        for t in thresholds:
            config.RETENTION["discard_below"] = t
            config.RETENTION["compress_below"] = t + 0.11
            store = _reset_store(f"sweep")
            agent = MemoryAgent(store, with_reconciler=True)
            for e in load_jsonl(config.DIARY_FILE):
                agent.ingest(e["date"], e["text"])
            agent.maintain()
            per_type, details = score_agent(agent, benchmark)
            st = store.stats()
            by = st["memories_by_status"]
            rows.append({
                "threshold": t,
                "no_archive": no_archive,
                "overall": round(sum(sum(v) for v in per_type.values())
                                 / sum(len(v) for v in per_type.values()), 3),
                "type_acc": {k: round(sum(v) / len(v), 3) for k, v in per_type.items()},
                "active": by.get("active", 0),
                "discarded": by.get("discarded", 0) + by.get("compressed", 0),
            })
    finally:
        config.RETENTION.clear()
        config.RETENTION.update(saved_r)
        config.ALLOW_ARCHIVE = saved_a
    suffix = "_no_archive" if no_archive else ""
    with open(config.RESULTS_DIR / f"forget_sweep{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    return rows


# ---------- 关闭独立冲突治理 vs 完整任务分工 ----------
def run_agent_compare(benchmark: list[dict]) -> dict:
    out = {}
    for mode, reconciler in (("single", False), ("multi", True)):
        store = _reset_store(f"agent_{mode}")
        agent = MemoryAgent(store, with_reconciler=reconciler)
        for e in load_jsonl(config.DIARY_FILE):
            agent.ingest(e["date"], e["text"])
        agent.maintain()
        per_type, details = score_agent(agent, benchmark)
        st = store.stats()
        out[mode] = summarize(per_type, details,
                              st["memories_by_status"].get("active", 0),
                              {"conflicts_found": st["conflicts"],
                               "conflicts_resolved": st["conflicts_resolved"]})
    with open(config.RESULTS_DIR / "agent_compare.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return out


def print_report(results: dict):
    types = ["recent", "contradiction", "summary", "time", "timeline", "trap"]
    names = {"A": "A·无记忆(仅最近3条)", "B": "B·扁平向量(只存不整理)", "C": "C·本方案(分层+遗忘)"}
    header = f"{'架构':<24}" + "".join(f"{t:>10}" for t in
              ["近期事实", "矛盾处理", "总结归纳", "时间回溯", "时间线", "诱导拒答"]) + f"{'综合':>8}{'记忆条目':>10}"
    print(header)
    for arch in ("A", "B", "C"):
        r = results[arch]
        row = [r["type_acc"].get(t, 0.0) for t in types]
        print(f"{names[arch]:<24}" + "".join(f"{v:>10.0%}" for v in row)
              + f"{r['overall']:>8.1%}{r['memory_size']:>10}")
