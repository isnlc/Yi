# 推理层：把检索到的证据组织成带引用、带冲突说明的回答
import re

from . import llm
from .retrieve import Retriever, expand_query, is_temporal_activity_query, tokenize

# 归纳类问法：命中才引用结论层（避免现状类问题被主题级结论带偏）
SUMMARY_TRIGGERS = ("总结", "习惯", "怎么样", "整体", "做什么", "哪些",
                    "平时", "兴趣", "爱好", "历程", "经历")
CURRENT_TRIGGERS = (
    "现在", "目前", "当前", "如今", "最近", "已经", "还在", "是不是还",
    "最终", "最后", "最后状态",
)
LATEST_EVENT_TRIGGERS = ("最近一次", "上一次", "最后一次")
CURRENT_ACTION_PRIORITY = {
    "更改": 6,
    "偏好": 6,
    "开始": 5,
    "放弃": 5,
    "约定": 5,
    "完成": 3,
}


def _evidence_lines(hits) -> list[str]:
    lines, seen = [], set()
    for m, _score in sorted(hits, key=lambda x: x[0].date):
        ln = f"（{m.date}）{m.detail}"
        if ln not in seen:
            seen.add(ln)
            lines.append(ln)
    return lines


def _split_core_extra(hits):
    """核心证据（RRF 排名，score>=0）与扩充证据（取代链/关键事件，score<0）分离。
    扩充证据只追加在核心证据之后，保证不挤占核心证据的截断名额。"""
    core = [(m, s) for m, s in hits if s >= 0]
    extra = [(m, s) for m, s in hits if s < 0]
    return core, extra


def _conflict_note(conflicts, hit_ids: set) -> list[str]:
    seen, notes = set(), []
    for c in conflicts:
        if (c.older_id in hit_ids and c.newer_id in hit_ids) and "最新" not in seen:
            seen.add("最新")
            notes.append("曾有过前后不同的说法，以最新的为准")
    return notes


def _is_current_query(query: str) -> bool:
    return (
        not is_temporal_activity_query(query)
        and any(t in query for t in CURRENT_TRIGGERS)
    )


def _is_latest_event_query(query: str) -> bool:
    return any(t in query for t in LATEST_EVENT_TRIGGERS)


def _is_summary_query(query: str) -> bool:
    return bool(
        query.startswith(("总结", "归纳", "概括", "梳理", "复盘"))
        or any(marker in query for marker in (
            "习惯怎么样", "整体怎么样", "有哪些兴趣", "有哪些爱好",
            "平时做什么", "历程", "经历了什么", "怎么变化",
        ))
    )


def _is_attribute_value_query(query: str) -> bool:
    if re.search(r"是不是|是否|还(?:在|有)|有没有|对不对|吗[？?]?$", query):
        return False
    return bool(re.search(
        r"是什么|是多少|什么|"
        r"哪(?:个|种|里|儿)|多久|几次|多长时间|多少钱|多少(?:本|个|次|元)|"
        r"频率|时间|地点|城市|类型|设备|配置|颜色|风格|金额",
        query,
    ))


def _historical_event_hits(query: str, hits):
    """Keep records that express the event asked by a 'when' question."""
    if not re.search(r"什么时候|哪天|哪一年|哪一月|几月几日", query):
        return hits
    selectors = []
    if "开始" in query:
        selectors.append(lambda m: m.action in ("开始", "更改")
                         or bool(re.search(r"开始|换成|改成|改用|决定", m.detail)))
    if re.search(r"买|购买|入手", query):
        selectors.append(lambda m: bool(re.search(r"买|购买|入手|下单", m.detail)))
    if re.search(r"提交|交上|交了", query):
        selectors.append(lambda m: m.action == "完成"
                         or bool(re.search(r"提交|交上|交了", m.detail)))
    if re.search(r"停止|放弃|中断", query):
        selectors.append(lambda m: m.action == "放弃"
                         or bool(re.search(r"停止|放弃|中断", m.detail)))
    if re.search(r"写|总结", query):
        selectors.append(lambda m: bool(re.search(r"写|总结", m.detail)))
    if not selectors:
        return hits
    selected = [
        (memory, score) for memory, score in hits
        if any(selector(memory) for selector in selectors)
    ]
    return selected or hits


def _latest_event_hits(query: str, hits):
    """“最近一次”问的是事件，不是当前状态；按具体对象匹配后取最新记录。"""
    if not _is_latest_event_query(query):
        return hits
    generic = {
        "最近", "一次", "最近一次", "上次", "上一次", "最后", "最后一次",
        "什么", "什么时候", "哪天", "时候", "是", "的", "了", "我",
    }
    qtoks = {
        t for t in expand_query(tokenize(query))
        if len(t) >= 2 and t not in generic
    }
    candidates = []
    for m, score in hits:
        mtoks = set(tokenize(m.detail)) | {m.topic}
        overlap = len(qtoks & mtoks)
        if overlap:
            candidates.append((m, score, overlap))
    if not candidates:
        return []
    selected, selected_score, _overlap = max(
        candidates,
        key=lambda x: (x[2], x[0].date, x[1]),
    )
    return [(selected, selected_score)]


def _current_state_hits(store, query: str, hits):
    """现状类问题只保留最近一次明确状态，以及与它直接相连的取代链证据。"""
    if not _is_current_query(query):
        return hits
    qtoks = {t for t in expand_query(tokenize(query)) if len(t) >= 2}
    write_order = {memory.id: index for index, memory in enumerate(store.memories)}

    from .semantics import analyze_semantics, frame_for_memory
    query_frame = analyze_semantics(query, is_query=True)
    requested_evidence_facets = (
        set(query_frame.facets) - {"状态"}
        if _is_attribute_value_query(query)
        else set()
    )
    if requested_evidence_facets:
        attribute_hits = []
        state_hits = []
        for memory, score in hits:
            parent = store.memory_by_id(memory.parent_id) if memory.parent_id else None
            memory_frame = frame_for_memory(memory, parent.detail if parent else "")
            if requested_evidence_facets & set(memory_frame.facets):
                attribute_hits.append((memory, score))
            if memory.action in {"更改", "放弃", "开始", "约定"}:
                state_hits.append((memory, score, memory_frame))
        if attribute_hits:
            selected, selected_score = max(
                attribute_hits,
                key=lambda x: (
                    x[0].status != "superseded",
                    x[0].date,
                    write_order.get(x[0].id, -1),
                ),
            )
            if state_hits:
                latest_state, latest_state_score, latest_state_frame = max(
                    state_hits,
                    key=lambda x: (
                        x[0].status != "superseded",
                        x[0].date,
                        write_order.get(x[0].id, -1),
                        CURRENT_ACTION_PRIORITY.get(x[0].action, 0),
                    ),
                )
                if (
                    latest_state.date > selected.date
                    and not requested_evidence_facets & set(latest_state_frame.facets)
                ):
                    return [
                        (latest_state, latest_state_score),
                        (selected, -1.0),
                    ]
            concise = [(selected, selected_score)]
            for conflict in store.conflicts:
                if selected.id not in (conflict.older_id, conflict.newer_id):
                    continue
                other_id = (
                    conflict.newer_id
                    if selected.id == conflict.older_id
                    else conflict.older_id
                )
                other = store.memory_by_id(other_id)
                if other:
                    concise.append((other, -1.0))
                break
            return concise

    candidates = []
    for m, score in hits:
        if m.action not in CURRENT_ACTION_PRIORITY:
            continue
        mtoks = set(tokenize(m.detail))
        overlap = len(qtoks & mtoks)
        candidates.append((m, score, overlap))
    asks_state_skeleton = (
        bool(set(query_frame.facets) & {"状态"})
        or bool(re.search(r"计划|安排|约定|是否继续|还(?:在|有)|放弃", query))
    )
    if asks_state_skeleton:
        skeleton = [
            item for item in candidates
            if item[0].action in {"更改", "放弃", "开始", "约定"}
        ]
        if skeleton:
            candidates = skeleton
    if not candidates:
        if "最近" in query:
            usable = [(m, score) for m, score in hits if m.status != "superseded"]
            if usable:
                return [max(usable, key=lambda x: (x[0].date, write_order.get(x[0].id, -1)))]
        return hits

    selected, selected_score, _overlap = max(
        candidates,
        key=lambda x: (
            x[0].status != "superseded",
            x[0].date,
            write_order.get(x[0].id, -1),
            CURRENT_ACTION_PRIORITY.get(x[0].action, 0),
            x[2],
            x[1],
        ),
    )
    concise = [(selected, selected_score)]
    for c in store.conflicts:
        if selected.id not in (c.older_id, c.newer_id):
            continue
        other_id = c.newer_id if selected.id == c.older_id else c.older_id
        other = store.memory_by_id(other_id)
        if other:
            concise.append((other, -1.0))
        break
    return concise


def compose(
    store,
    query: str,
    hits,
    retriever: Retriever = None,
    display: str = "recent",
    answer_mode: str = "auto",
) -> dict:
    """返回 {answer, evidence, used_conclusion}。

    display 控制证据呈现顺序（本地模板与 LLM 上下文共用同一份证据）：
    - "recent"   现状类问题：最新证据排最前（体现"以最新为准"），最多 6 条；
    - "timeline" 时间回溯/总结类：按时间顺序铺开叙事，最多 8 条；
    - "calendar" 日历窗口类（"十月以来""国庆"）：按时间顺序铺开，窗口内全量呈现。

    answer_mode:
    - retrieval_only：只用确定性模板组织检索证据，不联系 LLM；
    - retrieval_llm：将检索证据交给 LLM 做分析、建议或综合表达；
    - auto：兼容旧调用，模型接口可用时使用 LLM。
    """
    hits = _historical_event_hits(query, hits)
    if _is_latest_event_query(query):
        hits = _latest_event_hits(query, hits)
    else:
        hits = _current_state_hits(store, query, hits)
    hit_ids = {m.id for m, _ in hits}
    conflicts = [] if display == "calendar" else store.conflicts_in(hit_ids)
    core, extras = _split_core_extra(hits)
    core_lines = _evidence_lines(core)
    extra_lines = _evidence_lines(extras)
    if display == "recent":
        lines = list(reversed(core_lines))[:8] + extra_lines[:3]
    elif display == "calendar":
        # 检索层已保证"窗口内记录优先且按时间排序"，这里保持该顺序不重排，
        # 避免窗口外噪声记录按日期插到窗口记录前面
        seen = set()
        ordered = []
        for m, _s in core:
            ln = f"（{m.date}）{m.detail}"
            if ln not in seen:
                seen.add(ln)
                ordered.append(ln)
        lines = ordered[:20] + extra_lines[:2]
    else:
        lines = core_lines[:8] + extra_lines[:2]
    # 结论层引用：归纳类问题按命中主题聚合结论；泛问（"兴趣爱好"）按主题组聚合
    conclusions = []
    if hits and display == "timeline" and _is_summary_query(query):
        from .memory import BROAD_TOPICS
        wanted: list[str] = []
        for keys, ts in BROAD_TOPICS.items():
            if any(k in query for k in keys):
                wanted.extend(ts)
        for m, _s in hits:
            wanted.append(m.topic)
        seen_topics = set()
        for topic in wanted:
            if topic in seen_topics:
                continue
            seen_topics.add(topic)
            c = store.conclusion_for(topic)
            if c and all(c.topic != x.topic for x in conclusions):
                conclusions.append(c)
            if len(conclusions) >= 3:
                break

    conflict_notes = _conflict_note(conflicts, hit_ids)
    window_note = ""
    if display == "calendar" and retriever is not None:
        window = retriever.calendar_window_info(query)
        if window["fallback"]:
            start, end = window["effective"][0]
            window_note = (
                f"当前最近七天没有相关记录，以下为记忆库最近有记录的时间段"
                f"（{start} 至 {end}）。"
            )
    allow_llm = answer_mode in ("auto", "retrieval_llm") and llm.is_enabled()
    if allow_llm and (lines or answer_mode == "retrieval_llm"):
        ctx = "\n".join(lines) if lines else "（未检索到相关个人记录）"
        extra = f"\n注意：{('；'.join(conflict_notes))}" if conflict_notes else ""
        if window_note:
            extra += f"\n时间范围：{window_note}"
        concl = ("\n历史总结：" + " / ".join(c.text for c in conclusions)) if conclusions else ""
        if _is_latest_event_query(query):
            current_instruction = "这是最近一次事件问题。只回答与问题对象直接匹配的最新一条事件，不要改答同主题的偏好或状态；"
        elif _is_current_query(query):
            current_instruction = "这是现状问题。先用一句话直接回答最近一次明确状态，不要罗列无关记录；"
        else:
            current_instruction = ""
        try:
            if answer_mode == "retrieval_llm":
                system_prompt = (
                    "你是个人记忆助手。先以给出的个人记忆证据为事实基础，再回答用户的分析、"
                    "比较、总结或建议问题。可以使用通用知识完成推理，但不得虚构用户经历、"
                    "偏好或状态；推断必须与已记录事实分开表达。引用个人事实时标注日期。"
                    "若没有相关证据，明确说明缺少个性化依据，再给出通用建议或只询问一个关键条件。"
                    "直接回答问题，不解释检索流程。"
                )
            else:
                system_prompt = (
                    "你是个人记忆问答系统。只依据给出的记忆证据回答，并标注日期；"
                    "证据有矛盾时采用最新说法。直接回答问题，不添加推测、建议、寒暄或反问。"
                    + current_instruction
                )
            ans = llm.chat([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"记忆证据：\n{ctx}{extra}{concl}\n\n问题：{query}"},
            ])
            return {
                "answer": ans,
                "evidence": lines,
                "used_conclusion": bool(conclusions),
                "llm_used": True,
            }
        except Exception:
            pass

    # ---------- 本地模板作答 ----------
    parts = []
    if conclusions and display == "timeline" and _is_summary_query(query):
        for c in conclusions:
            parts.append("总结：" + c.text)
    if window_note:
        parts.append(window_note)
    if _is_current_query(query) and hits:
        current = max(
            (m for m, _s in hits),
            key=lambda m: (m.status == "active", m.date),
        )
        parts.append(f"根据你最近一次明确记录：{current.detail}（{current.date}）")
        from .semantics import analyze_semantics, frame_for_memory
        query_frame = analyze_semantics(query, is_query=True)
        requested_facets = (
            set(query_frame.facets) - {"状态"}
            if _is_attribute_value_query(query)
            else set()
        )
        current_parent = store.memory_by_id(current.parent_id) if current.parent_id else None
        current_frame = frame_for_memory(
            current,
            current_parent.detail if current_parent else "",
        )
        missing_facets = requested_facets - set(current_frame.facets)
        historical_attribute = None
        if missing_facets:
            candidates = []
            for memory, _score in hits:
                if memory.id == current.id:
                    continue
                parent = store.memory_by_id(memory.parent_id) if memory.parent_id else None
                memory_frame = frame_for_memory(
                    memory,
                    parent.detail if parent else "",
                )
                if missing_facets & set(memory_frame.facets):
                    candidates.append(memory)
            if candidates:
                historical_attribute = max(candidates, key=lambda memory: memory.date)
        if historical_attribute:
            facet_label = "、".join(sorted(missing_facets))
            parts.append(
                f"这条记录没有说明当前{facet_label}。上一阶段记录为："
                f"{historical_attribute.detail}（{historical_attribute.date}），"
                "该值不能直接视为当前状态。"
            )
        elif conflict_notes:
            parts.append("这一状态有过更新，当前以上述最新记录为准。")
    elif lines:
        if conflict_notes:
            parts.append("提醒：关于这一点" + "、".join(conflict_notes) + "。")
        parts.append("根据你的记录：\n" + "\n".join("- " + ln for ln in lines))
    else:
        parts.append("记忆库中没有找到与这个问题对应的记录。")
    return {
        "answer": "\n".join(parts),
        "evidence": lines,
        "used_conclusion": bool(conclusions),
        "llm_used": False,
    }


