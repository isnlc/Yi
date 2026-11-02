# -*- coding: utf-8 -*-
# 实验4：本体通路演示——记忆库 -> RDFS/OWL 图 -> SPARQL 查询
# 展示与三路检索并列的第二条查询通路：时间线 / 当前结论 / 冲突溯源链
import sys, os, json
os.environ["MEMAGENT_LLM"] = "0"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memagent import config
from memagent.evaluate import build_agent
from tools import ontology_tool as ot


def main():
    if not ot.is_available():
        print("未安装 rdflib（pip install rdflib），本体通路不可用，主流程不受影响。")
        return
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    agent = build_agent("C")
    g = ot.build_graph(agent.store)
    print(f"RDF 图构建完成：{len(g)} 条三元组")

    ttl = ot.export_turtle(g, config.RESULTS_DIR / "ontology.ttl")
    print(f"本体已导出：{ttl}\n")

    for topic in ("健身", "课程"):
        print(f"—— SPARQL 时间线：{topic} ——")
        rows = ot.sparql_timeline(g, topic)
        for r in rows:
            mark = " [档案]" if r["status"] != "active" else ""
            print(f"  ({r['date']}) {r['action']} | {r['detail'][:30]}{mark}")
        concl = ot.sparql_current_conclusion(g, topic)
        if concl:
            print(f"  当前结论（{concl['updated']}）：{concl['text'][:60]}")
        print()

    print("—— SPARQL 冲突溯源链（supersedes） ——")
    for r in ot.sparql_supersedes(g):
        print(f"  {r['newer']} 取代 {r['older']}")

    out = config.RESULTS_DIR / "ontology_demo.json"
    demo = {
        "triples": len(g),
        "timeline_健身": ot.sparql_timeline(g, "健身"),
        "timeline_课程": ot.sparql_timeline(g, "课程"),
        "supersedes": ot.sparql_supersedes(g),
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(demo, f, ensure_ascii=False, indent=1)
    print(f"\n演示结果已保存：{out}")


if __name__ == "__main__":
    main()
