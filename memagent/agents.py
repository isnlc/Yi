# 任务智能体层：记录抽取、冲突治理、证据检索、回答审核。
# 四项任务由同一编排器管理，共享记忆状态与处理日志。
import os
from datetime import date as _date

from . import config, llm
from .memory import (
    IMPORTANCE,
    apply_retention,
    consolidate,
    detect_conflicts,
    extract,
    is_state_update,
)
from .retrieve import Retriever
from .answer import compose, _evidence_lines
from .semantics import analyze_semantics, frame_for_memory, semantic_match_score
from .store import MemoryStore, Conflict


class MemoryAgent:
    """完整编排：四项任务在统一状态上依次执行。
    ingest   = 记录抽取：原始记录 -> 结构化记忆 -> 写入时冲突检测
    maintain = 冲突治理：规则补扫 + 可选语义判断 + 提炼 + 保留策略
    answer   = 证据检索与回答审核：路由 -> 检索 -> 校验 -> 必要时重查
    """

    def __init__(
        self,
        store: MemoryStore = None,
        with_reconciler: bool = True,
        log=None,
        reference_date: str | _date | None = None,
    ):
        self.store = store or MemoryStore()
        self.retriever = Retriever(self.store, reference_date=reference_date)
        self.reference_date = self.retriever.reference_date
        self.with_reconciler = with_reconciler
        self.log_fn = log   # 决策留痕回调：log(event, **payload)

    def _log(self, event: str, **payload):
        if self.log_fn:
            try:
                self.log_fn(event, **payload)
            except Exception:
                pass  # 日志失败不影响主流程

    # ---------- 回答审核（规则兜底） ----------
    @staticmethod
    def _has_overlap(query: str, hits) -> bool:
        """证据与问题的内容词是否仍有交集（用扩展词表，含同义词）。"""
        from .retrieve import expand_query, tokenize
        query_frame = analyze_semantics(query, is_query=True)
        if (query_frame.entities or query_frame.facets) and any(
            semantic_match_score(query_frame, frame_for_memory(memory))
            for memory, _score in hits
        ):
            return True
        qtoks = {t for t in expand_query(tokenize(query)) if len(t) >= 2}
        etoks = set()
        for m, _s in hits:
            etoks.update(tokenize(m.detail))
            etoks.add(m.topic)
        return bool(qtoks & etoks)

    def _rule_audit(self, query: str, hits) -> dict:
        """证据底线校验：无证据时拒答，证据跑偏时回溯档案层。"""
        if not hits:
            return {"sufficient": False, "action": "refuse", "reason": "无证据，应拒答"}
        from .retrieve import calendar_spans
        if calendar_spans(query, self.reference_date):
            return {
                "sufficient": True,
                "action": "answer",
                "reason": "证据位于问题指定的时间窗口内",
            }
        if self._has_overlap(query, hits):
            return {"sufficient": True, "action": "answer", "reason": "证据与问题内容词有交集"}
        return {"sufficient": False, "action": "refine", "reason": "证据与问题内容词零重叠，疑似跑偏"}

    # ---------- 记录抽取 ----------
    def ingest(
        self,
        date: str,
        text: str,
        source: str = "diary",
        *,
        memory_text: str | None = None,
        kind: str = "record",
        parent_id: str = "",
        memory_type: str = "long_term",
        confidence: float = 1.0,
        preference_key: str = "",
        action_override: str = "",
        topic_override: str = "",
        entity_override: str = "",
    ):
        entry = self.store.add_entry(
            date,
            text,
            source,
            kind=kind,
            parent_id=parent_id,
        )
        items = extract(self.store, memory_text or text, date, entry.id)
        for item in items:
            if topic_override:
                item.topic = topic_override
            item.memory_type = memory_type
            item.confidence = confidence
            item.parent_id = parent_id
            item.preference_key = preference_key
            if action_override:
                item.action = action_override
                item.importance = IMPORTANCE.get(action_override, item.importance)
            frame = analyze_semantics(
                f"{item.detail} {entity_override}",
                topic_hint=item.topic,
                preference_value=preference_key,
            )
            item.entities = list(frame.entities)
            item.facets = list(frame.facets)
            item.semantic_values = list(frame.values)
        conflicts = detect_conflicts(self.store, items) if self.with_reconciler else []
        for c in conflicts:
            self._log("reconciler_decision", pair=f"{c.older_id}|{c.newer_id}",
                      conflict=True, reason="同一对象的状态更新", method="rule_ingest")
        return entry, items, conflicts

    # ---------- 冲突治理 ----------
    def maintain(self):
        """完整模式执行全库冲突补扫、提炼和保留策略。
        对照模式（with_reconciler=False）关闭独立冲突治理，只保留提炼和保留策略。"""
        report = {}
        have = set()
        if self.with_reconciler:
            acts = sorted([m for m in self.store.active_memories()
                           if m.action in ("开始", "放弃", "更改", "约定")], key=lambda m: m.date)
            have = {(c.older_id, c.newer_id) for c in self.store.conflicts}
            for i, a in enumerate(acts):
                for b in acts[i + 1:]:
                    if (
                        b.date >= a.date
                        and is_state_update(a, b)
                        and (a.id, b.id) not in have
                    ):
                        a.status = "superseded"
                        c = Conflict(id="c" + a.id[1:] + b.id[1:], topic=b.topic, action=b.action,
                                     older_id=a.id, newer_id=b.id, resolved=True, winner_id=b.id)
                        self.store.conflicts.append(c)
                        have.add((a.id, b.id))
                        self._log("reconciler_decision", pair=f"{a.id}|{b.id}",
                                  conflict=True, reason="同一对象的状态更新", method="rule_sweep")

            # LLM 语义判冲突：规则词表覆盖不到的改口（如"约定->约定"换了内容、表述换了说法）
            if llm.is_enabled():
                pairs = []
                for i, a in enumerate(acts):
                    for b in acts[i + 1:]:
                        if a.topic == b.topic and b.date > a.date and (a.id, b.id) not in have:
                            pairs.append({
                                "pair_id": f"{a.id}|{b.id}",
                                "older": {"date": a.date, "detail": a.detail},
                                "newer": {"date": b.date, "detail": b.detail},
                                "_a": a, "_b": b,
                            })
                pairs = pairs[:40]   # 控制单次判断规模
                try:
                    verdicts = llm.judge_conflicts(
                        [{k: p[k] for k in ("pair_id", "older", "newer")} for p in pairs])
                    by_id = {v.get("pair_id"): v for v in verdicts if isinstance(v, dict)}
                    for p in pairs:
                        v = by_id.get(p["pair_id"])
                        if not v:
                            continue
                        if v.get("conflict"):
                            a, b = p["_a"], p["_b"]
                            a.status = "superseded"
                            c = Conflict(id="c" + a.id[1:] + b.id[1:], topic=b.topic, action=b.action,
                                         older_id=a.id, newer_id=b.id, resolved=True, winner_id=b.id)
                            self.store.conflicts.append(c)
                            have.add((a.id, b.id))
                        self._log("reconciler_decision", pair=p["pair_id"],
                                  conflict=bool(v.get("conflict")),
                                  reason=str(v.get("reason", "")), method="llm")
                except Exception:
                    self._log("reconciler_decision", method="llm_failed", fallback="rule_only")

        report["conclusions"] = [c.topic for c in consolidate(self.store)]
        report["retention"] = apply_retention(self.store, self.reference_date)
        self.store.save()
        return report

    # ---------- 证据检索 + 回答审核 ----------
    def answer(self, query: str, answer_mode: str = "auto") -> dict:
        from .retrieve import calendar_spans
        trace = []
        allow_llm = answer_mode in ("auto", "retrieval_llm")
        deep_reasoning = (
            allow_llm
            and llm.is_enabled()
            and os.environ.get("MEMAGENT_DEEP_REASONING", "0") == "1"
        )
        route = None
        planned_reason = "规则路由"
        plan_method = "rule"
        is_calendar = bool(calendar_spans(query, self.reference_date))
        topk = 16 if is_calendar else None   # 日历类问题需要更宽的证据面（窗口内全量召回）

        # 轮 1：快速模式使用确定性路由；深度模式才调用 LLM 规划器
        if deep_reasoning:
            try:
                planned = llm.plan_route(query)
                proposed = str(planned.get("route") or "")
                if proposed in ("time+lex", "lex+graph", "lex+time"):
                    route = proposed
                    planned_reason = str(planned.get("reason", ""))
                    plan_method = "llm"
            except Exception:
                pass
        if route is None:
            route = self.retriever.route(query)
            planned_reason = "本地确定性路由"
            plan_method = "rule"
        self._log("query_round", round=1, role="retriever", action="plan",
                  route=route, reason=planned_reason, method=plan_method)
        hits = self.retriever.search(
            query,
            route=route,
            topk=topk,
            use_llm=allow_llm,
        )
        trace.append({"round": 1, "action": "plan+search", "route": route,
                      "hits": len(hits), "reason": planned_reason,
                      "method": plan_method})
        # 轮 2：活跃层无结果 -> 翻档案层（遗忘≠销毁证据链）
        if not hits and config.ALLOW_ARCHIVE:
            self._log("query_round", round=2, role="retriever", action="retry_archive",
                      reason="活跃层无结果，回溯档案层", method="rule")
            hits = self.retriever.search(
                query,
                active_only="archive",
                topk=topk,
                use_llm=allow_llm,
            )
            trace.append({"round": 2, "action": "retry_archive", "hits": len(hits),
                          "reason": "活跃层无结果，回溯档案层"})

        # 检查证据是否足够、是否跑偏、是否应拒答
        if deep_reasoning:
            lines = _evidence_lines(hits)
            audit = None
            try:
                audit = llm.audit_evidence(query, lines)
            except Exception:
                pass
            if audit:
                self._log("auditor_decision", sufficient=bool(audit.get("sufficient")),
                          action=audit.get("action", ""), reason=str(audit.get("reason", "")),
                          stale=audit.get("stale", []), method="llm")
                action = audit.get("action", "answer")
                # 轮 3：证据不足时改写查询并重查
                if action == "refine" and len(trace) < config.MAX_QUERY_ROUNDS:
                    try:
                        ref = llm.refine_query(query, query, lines)
                        new_q = str(ref.get("new_query") or query)
                        self._log("query_round", round=len(trace) + 1, role="retriever",
                                  action="refine", query=new_q,
                                  reason=str(ref.get("reason", "")), method="llm")
                        new_hits = self.retriever.search(new_q, use_llm=True)
                        if len(new_hits) > len(hits):   # 改写更有效才采纳
                            hits = new_hits
                        trace.append({"round": len(trace) + 1, "action": "refine",
                                      "query": new_q, "hits": len(hits),
                                      "reason": str(ref.get("reason", ""))})
                    except Exception:
                        pass
                elif action == "refuse" and hits:
                    # 证据无法支持回答时拒答
                    trace.append({"action": "refuse", "reason": str(audit.get("reason", ""))})
                    return {"answer": f"这件事我手里还没有相关的记录，所以不敢瞎说（{audit.get('reason', '')}）。\n"
                                      f"现在告诉我也不迟——说「帮我记一下……」，我马上替你记住。",
                            "evidence": [], "used_conclusion": False, "trace": trace}
        else:
            # 规则兜底：证据内容词与问题内容词零重叠时回溯档案层，
            # 触发档案层回溯；回溯结果也答不上才认输
            audit = self._rule_audit(query, hits)
            self._log("auditor_decision", method="rule", **audit)
            if audit["action"] == "refine" and config.ALLOW_ARCHIVE \
                    and len(trace) < config.MAX_QUERY_ROUNDS:
                new_hits = self.retriever.search(
                    query,
                    active_only="archive",
                    use_llm=allow_llm,
                )
                if new_hits and self._has_overlap(query, new_hits):
                    self._log("query_round", round=len(trace) + 1, role="retriever",
                              action="refine_archive", reason="证据跑偏，回溯档案层重新检索",
                              method="rule")
                    hits = new_hits
                    trace.append({"round": len(trace) + 1, "action": "refine_archive",
                                  "hits": len(hits), "reason": "证据跑偏，回溯档案层重新检索"})

        out = compose(
            self.store,
            query,
            hits,
            self.retriever,
            display=self._display_mode(query, route),
            answer_mode=answer_mode,
        )
        out["trace"] = trace
        out["answer_mode"] = answer_mode
        self.store.save()
        return out

    def _display_mode(self, query: str, route: str) -> str:
        """证据呈现模式：日历窗口/时间回溯/总结类按时间线铺开，现状类最新在前。"""
        from .retrieve import calendar_spans
        if bool(calendar_spans(query, self.reference_date)):
            return "calendar"
        if route.startswith("time") or route == "lex+graph":
            return "timeline"
        return "recent"


class FlatAgent:
    """对照架构 B：一个智能体全包，只存不整理——直接对原始记录做检索。"""

    def __init__(self, store: MemoryStore):
        self.store = store
        self.retriever = Retriever(store)

    def ingest(self, date: str, text: str, source: str = "diary"):
        return self.store.add_entry(date, text, source)

    def maintain(self):
        return {}

    def answer(self, query: str) -> dict:
        from .retrieve import expand_query, tokenize
        docs = [{"id": e.id, "date": e.date, "text": e.text} for e in self.store.entries]
        toks = expand_query(tokenize(query))
        hits = self.retriever._bm25_search(toks, docs, config.TOP_K)
        by_id = {e.id: e for e in self.store.entries}
        lines = [f"（{by_id[h[0]].date}）{by_id[h[0]].text}" for h in hits if h[0] in by_id]
        return {"answer": "根据你的记录：\n" + "\n".join("- " + ln for ln in lines[:6]),
                "evidence": lines, "used_conclusion": False}


class BaselineAgent:
    """对照架构 A：无记忆——只能看到最近几条记录（模拟有限上下文窗口）。"""

    def __init__(self, store: MemoryStore, window: int = 3):
        self.store = store
        self.window = window

    def ingest(self, date: str, text: str, source: str = "diary"):
        return self.store.add_entry(date, text, source)

    def maintain(self):
        return {}

    def answer(self, query: str) -> dict:
        recent = sorted(self.store.entries, key=lambda e: e.date)[-self.window:]
        lines = [f"（{e.date}）{e.text}" for e in recent]
        return {"answer": "我只能看到最近的几条记录：\n" + "\n".join("- " + ln for ln in lines),
                "evidence": lines, "used_conclusion": False}
