# -*- coding: utf-8 -*-
# 步骤1：把示例日记灌入记忆系统并执行记录抽取与冲突治理
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memagent import config
from memagent.evaluate import load_jsonl
from memagent.agents import MemoryAgent

def main():
    # 每次灌入前重置记忆库，避免重复运行时追加
    if os.path.exists(config.STORE_FILE):
        os.remove(config.STORE_FILE)
    entries = load_jsonl(config.DIARY_FILE)
    agent = MemoryAgent()
    n_items = 0
    for e in entries:
        _, items, conflicts = agent.ingest(e["date"], e["text"])
        n_items += len(items)
    report = agent.maintain()
    st = agent.store.stats()
    print("=== 灌入完成 ===")
    print(f"原始记录：{st['entries']} 条 -> 抽出记忆：{n_items} 条")
    print(f"记忆状态：{st['memories_by_status']}")
    print(f"冲突：{st['conflicts']} 条（已解决 {st['conflicts_resolved']}）")
    print(f"结论：{st['conclusions']} 条 -> {report['conclusions']}")
    print(f"遗忘决策：{st['forget_decisions']} 条")
    for d in report["retention"]:
        print(f"  [{d['decision']:>8}] {d['date']} ({d['topic']}) {d['detail'][:30]} score={d['score']}")
    print(f"\n记忆库已保存：{config.STORE_FILE}")

if __name__ == "__main__":
    main()
