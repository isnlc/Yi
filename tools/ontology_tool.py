"""Tool 5: project the memory store into an RDFS/OWL ontology and query it with SPARQL.

本体通路（呼应课程第 9-10 讲）：把"主题—动作—结论—时间"轻量本体形式化为
RDFS 类与属性，记忆条目作为实例（ABox）灌入；时间线/当前结论等关系型问题
可走 SPARQL 查询，与三路检索并列成第二条查询通路。

依赖 rdflib（pip install rdflib）；未安装时 is_available() 返回 False，
上层跳过本体通路，不影响主流程使用本地降级路径。
"""

from __future__ import annotations

NAMESPACES = {
    "mem": "http://example.org/memagent#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
}

SCHEMA_TTL = """
@prefix mem: <http://example.org/memagent#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

# ---------- 类（TBox：先于数据的骨架） ----------
mem:Entry        a owl:Class ; rdfs:label "原始记录" .
mem:Memory       a owl:Class ; rdfs:label "结构化记忆" .
mem:Conflict     a owl:Class ; rdfs:label "冲突" .
mem:Conclusion   a owl:Class ; rdfs:label "结论" .
mem:Topic        a owl:Class ; rdfs:label "主题" .
mem:Action       a owl:Class ; rdfs:label "动作" .

# ---------- 对象属性 ----------
mem:hasTopic     a owl:ObjectProperty ; rdfs:domain mem:Memory ; rdfs:range mem:Topic ; rdfs:label "属于主题" .
mem:hasAction    a owl:ObjectProperty ; rdfs:domain mem:Memory ; rdfs:range mem:Action ; rdfs:label "动作类型" .
mem:derivedFrom  a owl:ObjectProperty ; rdfs:domain mem:Memory ; rdfs:range mem:Entry ; rdfs:label "抽取自" .
mem:supersedes   a owl:ObjectProperty ; rdfs:domain mem:Memory ; rdfs:range mem:Memory ; rdfs:label "取代" .
mem:supports     a owl:ObjectProperty ; rdfs:domain mem:Conclusion ; rdfs:range mem:Memory ; rdfs:label "支撑" .

# ---------- 数据属性 ----------
mem:hasDate      a owl:DatatypeProperty ; rdfs:domain mem:Memory ; rdfs:range xsd:date ; rdfs:label "发生日期" .
mem:hasStatus    a owl:DatatypeProperty ; rdfs:domain mem:Memory ; rdfs:range xsd:string ; rdfs:label "状态" .
mem:detail       a owl:DatatypeProperty ; rdfs:domain mem:Memory ; rdfs:range xsd:string ; rdfs:label "内容" .
mem:importance   a owl:DatatypeProperty ; rdfs:domain mem:Memory ; rdfs:range xsd:double ; rdfs:label "重要性" .
mem:conflictText a owl:DatatypeProperty ; rdfs:domain mem:Conflict ; rdfs:range xsd:string ; rdfs:label "冲突描述" .
"""


def is_available() -> bool:
    try:
        import rdflib  # noqa: F401
        return True
    except ImportError:
        return False


def build_graph(store):
    """把记忆库投影成 RDF 图：TBox（骨架）+ ABox（实例）。返回 rdflib.Graph。"""
    from rdflib import Graph, Namespace, Literal, RDF, RDFS, XSD

    MEM = Namespace(NAMESPACES["mem"])
    g = Graph()
    for prefix, uri in NAMESPACES.items():
        g.bind(prefix, Namespace(uri))
    g.bind("owl", "http://www.w3.org/2002/07/owl#")
    g.bind("xsd", str(XSD))
    g.parse(data=SCHEMA_TTL, format="turtle")

    def uri(cls: str, key: str):
        return MEM[f"{cls}_{key}"]

    # 主题与动作作为个体（受控词表）
    topics = sorted({m.topic for m in store.memories})
    actions = sorted({m.action for m in store.memories})
    for t in topics:
        g.add((uri("Topic", t), RDF.type, MEM.Topic))
        g.add((uri("Topic", t), RDFS.label, Literal(t)))
    for a in actions:
        g.add((uri("Action", a), RDF.type, MEM.Action))
        g.add((uri("Action", a), RDFS.label, Literal(a)))

    for e in store.entries:
        g.add((uri("Entry", e.id), RDF.type, MEM.Entry))
        g.add((uri("Entry", e.id), MEM.hasDate, Literal(e.date, datatype=XSD.date)))

    for m in store.memories:
        s = uri("Memory", m.id)
        g.add((s, RDF.type, MEM.Memory))
        g.add((s, MEM.hasDate, Literal(m.date, datatype=XSD.date)))
        g.add((s, MEM.hasTopic, uri("Topic", m.topic)))
        g.add((s, MEM.hasAction, uri("Action", m.action)))
        g.add((s, MEM.hasStatus, Literal(m.status)))
        g.add((s, MEM.detail, Literal(m.detail, lang="zh")))
        g.add((s, MEM.importance, Literal(m.importance, datatype=XSD.double)))
        if m.evidence:
            g.add((s, MEM.derivedFrom, uri("Entry", m.evidence)))

    for c in store.conflicts:
        s = uri("Conflict", c.id)
        g.add((s, RDF.type, MEM.Conflict))
        newer = store.memory_by_id(c.newer_id)
        older = store.memory_by_id(c.older_id)
        if newer is not None and older is not None:
            g.add((uri("Memory", c.newer_id), MEM.supersedes, uri("Memory", c.older_id)))
            g.add((s, MEM.conflictText,
                   Literal(f"({older.date}) {older.detail} 被 ({newer.date}) {newer.detail} 取代", lang="zh")))

    for con in store.conclusions:
        s = uri("Conclusion", con.id)
        g.add((s, RDF.type, MEM.Conclusion))
        g.add((s, MEM.hasTopic, uri("Topic", con.topic)))
        g.add((s, MEM.detail, Literal(con.text, lang="zh")))
        g.add((s, MEM.hasDate, Literal(con.updated, datatype=XSD.date)))
        for mid in con.support:
            g.add((s, MEM.supports, uri("Memory", mid)))

    return g


# ---------- SPARQL 查询模板 ----------

TIMELINE_QUERY = """
PREFIX mem: <http://example.org/memagent#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?date ?action ?detail ?status WHERE {{
  ?m mem:hasTopic ?t ;
     mem:hasDate ?date ;
     mem:hasAction ?a ;
     mem:detail ?detail ;
     mem:hasStatus ?status .
  ?a rdfs:label ?action .
  ?t rdfs:label "{topic}" .
}}
ORDER BY ?date
"""

CURRENT_CONCLUSION_QUERY = """
PREFIX mem: <http://example.org/memagent#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?text ?updated WHERE {{
  ?c a mem:Conclusion ;
     mem:hasTopic ?t ;
     mem:detail ?text ;
     mem:hasDate ?updated .
  ?t rdfs:label "{topic}" .
}}
ORDER BY DESC(?updated)
LIMIT 1
"""

SUPERSEDED_QUERY = """
PREFIX mem: <http://example.org/memagent#>
SELECT ?newer ?older WHERE {
  ?newer mem:supersedes ?older .
}
"""


def sparql_timeline(graph, topic: str) -> list[dict]:
    """SPARQL：某主题的完整时间线（含档案层状态）。"""
    from rdflib.plugins.sparql import prepareQuery
    rows = []
    res = graph.query(TIMELINE_QUERY.format(topic=topic))
    for row in res:
        rows.append({"date": str(row.date), "action": str(row.action),
                     "detail": str(row.detail), "status": str(row.status)})
    return rows


def sparql_current_conclusion(graph, topic: str) -> dict | None:
    """SPARQL：某主题当前结论（最新一条）。"""
    res = graph.query(CURRENT_CONCLUSION_QUERY.format(topic=topic))
    for row in res:
        return {"text": str(row.text), "updated": str(row.updated)}
    return None


def sparql_supersedes(graph) -> list[dict]:
    """SPARQL：全部取代关系（冲突的溯源链）。"""
    rows = []
    for row in graph.query(SUPERSEDED_QUERY):
        rows.append({"newer": str(row.newer).split("_")[-1],
                     "older": str(row.older).split("_")[-1]})
    return rows


def export_turtle(graph, path) -> str:
    turtle = graph.serialize(format="turtle")
    with open(path, "w", encoding="utf-8") as f:
        f.write(turtle)
    return str(path)
