# -*- coding: utf-8 -*-
# 步骤2：和记忆智能体对话（证据检索按问题选路 -> 组织带引用的回答）
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memagent.agents import MemoryAgent

QUESTIONS = [
    "我现在还在坚持跑步吗？",
    "我是不是已经放弃了跑步计划？",
    "总结一下我的作息习惯怎么样？",
    "我什么时候开始手冲咖啡的？",
    "现在大扫除是怎么安排的？",
]

def main():
    agent = MemoryAgent()
    if not agent.store.memories:
        print("记忆库为空，请先运行：python scripts/ingest_demo.py")
        return
    for q in QUESTIONS:
        out = agent.answer(q)
        print("=" * 56)
        print("问：" + q)
        print("-" * 56)
        print(out["answer"])
        if out["used_conclusion"]:
            print("[已引用长期结论]")

if __name__ == "__main__":
    main()
