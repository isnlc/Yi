# -*- coding: utf-8 -*-
# 接入真实个人数据：把你自己的笔记/日记/聊天导出灌进记忆系统
# 用法：
#   python scripts/import_data.py <文件或目录>
# 支持格式：
#   1. .md / .txt 日记或笔记 —— 文件名含日期(YYYY-MM-DD 或 YYYYMMDD)则用之，
#      否则用文件修改时间；全文作为一条记录。
#   2. 聊天记录导出（.txt）—— 识别行首 "2026-08-01" 或 "2026-08-01 12:00" 的行，
#      每行拆成一条带时间戳的记录。
import os
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memagent import config
from memagent.agents import MemoryAgent

DATE_IN_NAME = re.compile(r"(20\d{2})[-_.年]?(\d{1,2})[-_.月]?(\d{1,2})")
LINE_DATE = re.compile(r"^(20\d{2}-\d{2}-\d{2})(?:[ T]\d{2}:\d{2}(?::\d{2})?)?")


def entries_from_file(path: str):
    with open(path, encoding="utf-8", errors="ignore") as f:
        content = f.read()
    name = os.path.basename(path)

    m = DATE_IN_NAME.search(name)
    default_date = m.group(1) + "-" + m.group(2).zfill(2) + "-" + m.group(3).zfill(2) \
        if m else datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d")

    chat_lines = []
    for ln in content.splitlines():
        dm = LINE_DATE.match(ln.strip())
        if dm:
            chat_lines.append((dm.group(1), ln.strip()[dm.end():].strip(" :：")))
    if len(chat_lines) >= 5:   # 像聊天导出：按行拆
        return [(d, t or "(空)") for d, t in chat_lines if t]
    return [(default_date, content.strip()[:2000])] if content.strip() else []


def main():
    if len(sys.argv) < 2:
        print("用法：python scripts/import_data.py <笔记文件或目录>")
        return
    target = sys.argv[1]
    files = []
    if os.path.isdir(target):
        for root, _dirs, names in os.walk(target):
            files += [os.path.join(root, n) for n in names if n.lower().endswith((".md", ".txt"))]
    else:
        files = [target]

    all_entries = []
    for f in files:
        all_entries += entries_from_file(f)
    if not all_entries:
        print("没有解析到任何记录（文件需为 .md/.txt，且能确定日期）。")
        return

    agent = MemoryAgent()
    for date, text in sorted(all_entries):
        agent.ingest(date, text, source="real")
    agent.maintain()
    st = agent.store.stats()
    print(f"已灌入 {len(all_entries)} 条真实记录（来源 {len(files)} 个文件）")
    print(f"记忆库状态：{st['memories_by_status']} | 冲突 {st['conflicts_resolved']}/{st['conflicts']} "
          f"| 结论 {st['conclusions']} 条 | 遗忘决策 {st['forget_decisions']} 条")
    print(f"现在可以直接运行 python scripts/chat_demo.py 用你的真实记忆对话。")

if __name__ == "__main__":
    main()
