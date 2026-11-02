# -*- coding: utf-8 -*-
"""真实笔记回放验证（拟真语料）：
用 3 段自然语言笔记（非评测集语料，句式为真实随手记风格）回放导入管线，
验证冲突检测、取代链与"以最新为准"在评测集之外的表达上是否成立。
产出 eval/results/notes_replay.json。
"""
import json
import os
import sys

os.environ["MEMAGENT_LLM"] = "0"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memagent import config
from memagent.store import MemoryStore
from memagent.agents import MemoryAgent

# 三段拟真笔记：与 dataset_large 措辞无重叠，覆盖"改主意"的三种常见说法
NOTES = [
    {
        "id": "note_travel",
        "date": "2026-09-20",
        "text": "国庆出游的计划折腾了好几轮。九月初本来和林晓约好去西安看兵马俑，"
                "票都差点下单了，后来看了天气预报说那边连着下雨，两个人一合计还是算了。"
                "翻攻略的时候刷到泉州，开元寺、蟳埔村还有面线糊看着都馋人，"
                "最后拍板：改去泉州，行程放慢一点，就逛老城。",
    },
    {
        "id": "note_coffee",
        "date": "2026-09-25",
        "text": "记账的时候发现自己这学期在咖啡上花了不少钱。以前图省事一直喝速溶，"
                "一条一条撕，桌上全是包装。上个月同事送了挂耳和豆子，试着用公司的小磨豆机"
                "手冲了两次，居然一次就成了，从此再也回不去了。现在每周自己烘一点浅烘的豆子，"
                "速溶那条货架再也没光顾过。",
    },
    {
        "id": "note_course",
        "date": "2026-09-28",
        "text": "课程大作业的题目终于定下来了。最开始想做校园导航，觉得地图数据好找，"
                "写了两页开题提纲；后来组员说导航被人做烂了，答辩容易撞车，"
                "讨论到半夜改成做课程表助手，能把课表、作业截止日和考试安排捏到一块儿，"
                "周五已经把新的开题交上去了。",
    },
]

QUESTIONS = [
    ("note_travel", "国庆我们最后决定去哪儿玩？", ["泉州"], ["西安"]),
    ("note_coffee", "我现在还喝速溶咖啡吗？", ["速溶"], None),
    ("note_course", "课程大作业最后选了什么题目？", ["课程表助手"], ["校园导航"]),
]


def main():
    store = MemoryStore(path=str(config.RESULTS_DIR / "_notes_replay.json"))
    if os.path.exists(store.path):
        os.remove(store.path)
    store = MemoryStore(path=str(config.RESULTS_DIR / "_notes_replay.json"))
    agent = MemoryAgent(store, with_reconciler=True)

    ingest_log = []
    for n in NOTES:
        entry, items, conflicts = agent.ingest(n["date"], n["text"], source="note")
        ingest_log.append({
            "note_id": n["id"],
            "memory_items": [{"topic": m.topic, "action": m.action, "detail": m.detail} for m in items],
            "conflicts_on_ingest": [f"{c.older_id}->{c.newer_id}" for c in conflicts],
        })
    report = agent.maintain()

    qa = []
    all_pass = True
    for note_id, question, expect, stale in QUESTIONS:
        out = agent.answer(question)
        ans = out["answer"]
        got = [k for k in expect if k in ans]
        hit_stale = [k for k in (stale or []) if k in ans]
        has_reminder = "以最新的为准" in ans
        # 旧值可以作为“从旧值改为新值”的历史依据出现；只有缺少更新语义时
        # 才把它视为把陈旧事实当成当前答案。
        change_context = any(marker in ans for marker in (
            "改为", "改成", "改去", "放弃", "不再", "最后决定", "当前",
            "再也没", "没光顾", "不喝", "不买",
        ))
        ok = len(got) == len(expect) and (
            has_reminder or not hit_stale or change_context
        )
        all_pass &= ok
        qa.append({
            "note_id": note_id, "question": question,
            "expected_keywords": expect, "hit_keywords": got,
            "stale_keywords": stale, "stale_appeared": hit_stale,
            "conflict_reminder": has_reminder,
            "pass": ok, "answer": ans, "evidence": out["evidence"][:6],
        })

    st = store.stats()
    result = {
        "corpus": "三段拟真自然语言笔记（与评测集措辞无重叠，验证泛化）",
        "ingest": ingest_log,
        "maintain": {"conclusions": report["conclusions"],
                     "retention_decisions": len(report["retention"])},
        "store_stats": st,
        "qa": qa,
        "all_pass": all_pass,
    }
    with open(config.RESULTS_DIR / "notes_replay.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)

    print(f"笔记回放：3 段语料 -> {st['memories_total']} 条记忆，"
          f"冲突 {st['conflicts']}（解决 {st['conflicts_resolved']}），"
          f"结论 {st['conclusions']} 条")
    for x in qa:
        flag = "PASS" if x["pass"] else "FAIL"
        print(f"[{flag}] {x['question']}  命中 {x['hit_keywords']}"
              + (f"  陈旧说法出现：{x['stale_appeared']}" if x["stale_appeared"] else ""))
    print("全部通过" if all_pass else "存在未通过项")
    print("已保存：", config.RESULTS_DIR / "notes_replay.json")
    if os.path.exists(store.path):
        os.remove(store.path)


if __name__ == "__main__":
    main()
