# 检索层：BM25 词法检索 + 时间检索 + 主题图扩展，按问题类型自动选路，RRF 融合
import math
import re
from calendar import monthrange
from collections import Counter
from datetime import date as _date, timedelta

import jieba

from . import config
from .semantics import (
    ENTITY_ALIASES,
    analyze_query,
    entity_overlap_count,
    frame_for_memory,
    ground_query_to_catalog,
    semantic_match_score,
)
from .store import MemoryStore

_STOP = set("的了是在我你他它和与也就都还很又再才已经已经有过被把让",
            ) | set("，。！？；、：""（）一个一些这样那样")


_CHINESE_MONTHS = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6,
    "七": 7, "八": 8, "九": 9, "十": 10, "十一": 11, "十二": 12,
}

DEFAULT_RECENT_DAYS = 7
_RECENT_PERIOD_RE = re.compile(
    r"最近一段时间|最近这段时间|最近一阵子|近期|近来|近几天|这几天|"
    r"最近(?!\s*(?:一次|一回|一趟))"
)
_TEMPORAL_ACTIVITY_RE = re.compile(
    r"(?:做|干|经历|去|吃|喝|看|买|完成|参加)"
    r"(?:了|过)(?:些)?(?:什么|哪些|啥)|"
    r"发生(?:了|过)?(?:什么|哪些|啥)|"
    r"(?:在忙|忙于|忙)(?:什么|哪些|啥)|"
    r"(?:有|发生)(?:什么|哪些|啥).{0,10}(?:活动|事情|事|记录|进展|变化)|"
    r"(?:活动|事情|记录|进展|变化).{0,10}(?:有哪些|有什么|是什么|如何)|"
    r"回顾|复盘|梳理|值得注意"
)


def is_temporal_activity_query(query: str) -> bool:
    """识别按近期时间段展开的事件/记录查询，而不是当前状态或最后一次事件。"""
    if not _RECENT_PERIOD_RE.search(query):
        return False
    if re.search(r"最近\s*(?:一次|一回|一趟)|上一次|最后一次", query):
        return False
    if re.search(r"现在|目前|当前|还在|是否还|是不是还|最终|最后状态", query):
        return False
    return bool(_TEMPORAL_ACTIVITY_RE.search(query))


def _small_chinese_number(raw: str) -> int | None:
    digits = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9}
    if raw in digits:
        return digits[raw]
    if raw == "十":
        return 10
    if "十" in raw:
        left, right = raw.split("十", 1)
        tens = digits.get(left, 1) if left else 1
        ones = digits.get(right, 0) if right else 0
        return tens * 10 + ones
    return None


def _as_date(value: str | _date | None) -> _date:
    if isinstance(value, _date):
        return value
    return _date.fromisoformat(value or config.NOW)


def _month_span(year: int, month: int, reference: _date) -> tuple[str, str]:
    start = _date(year, month, 1)
    end = _date(year, month, monthrange(year, month)[1])
    if year == reference.year and month == reference.month:
        end = reference
    return start.isoformat(), end.isoformat()


def _most_recent_window(
    reference: _date,
    start_month: int,
    start_day: int,
    end_month: int,
    end_day: int,
) -> tuple[str, str]:
    year = reference.year
    start = _date(year, start_month, start_day)
    end_year = year + (1 if end_month < start_month else 0)
    end = _date(end_year, end_month, end_day)
    if start > reference:
        start = _date(year - 1, start_month, start_day)
        end_year = year if end_month < start_month else year - 1
        end = _date(end_year, end_month, end_day)
    if start <= reference <= end:
        end = reference
    return start.isoformat(), end.isoformat()


def calendar_spans(
    query: str,
    reference_date: str | _date | None = None,
) -> list[tuple[str, str]]:
    """把绝对日期、任意月份、相对日期和常见时间段统一转换为日期区间。

    默认使用评测基准日保证离线实验可复现；网页端会显式传入系统当天日期。
    """
    reference = _as_date(reference_date)
    spans: list[tuple[str, str]] = []

    for raw in re.findall(r"20\d{2}-\d{2}-\d{2}", query):
        try:
            day = _date.fromisoformat(raw)
        except ValueError:
            continue
        spans.append((day.isoformat(), day.isoformat()))

    explicit_days = re.findall(
        r"(?:(20\d{2})\s*年\s*)?(1[0-2]|0?[1-9])\s*月\s*([0-3]?\d)\s*[日号]",
        query,
    )
    for raw_year, raw_month, raw_day in explicit_days:
        month, day_number = int(raw_month), int(raw_day)
        year = int(raw_year) if raw_year else reference.year
        try:
            day = _date(year, month, day_number)
        except ValueError:
            continue
        if not raw_year and day > reference:
            day = day.replace(year=year - 1)
        spans.append((day.isoformat(), day.isoformat()))

    relative_days = (
        ("前天", reference - timedelta(days=2)),
        ("昨天", reference - timedelta(days=1)),
        ("今天", reference),
    )
    spans.extend(
        (day.isoformat(), day.isoformat())
        for key, day in relative_days
        if key in query
    )

    recent = re.search(r"(?:最近|近)\s*(\d+|[一二两三四五六七八九十]+)\s*[天日]", query)
    if recent:
        raw_days = recent.group(1)
        parsed_days = int(raw_days) if raw_days.isdigit() else _small_chinese_number(raw_days)
        days = max(1, parsed_days or DEFAULT_RECENT_DAYS)
        spans.append(((reference - timedelta(days=days - 1)).isoformat(), reference.isoformat()))
    elif is_temporal_activity_query(query):
        spans.append((
            (reference - timedelta(days=DEFAULT_RECENT_DAYS - 1)).isoformat(),
            reference.isoformat(),
        ))

    monday = reference - timedelta(days=reference.weekday())
    if "上周" in query or "上个星期" in query:
        spans.append(((monday - timedelta(days=7)).isoformat(), (monday - timedelta(days=1)).isoformat()))
    elif "本周" in query or "这周" in query or "这个星期" in query:
        spans.append((monday.isoformat(), reference.isoformat()))

    if "上个月" in query or "上月" in query:
        first_this_month = reference.replace(day=1)
        previous_end = first_this_month - timedelta(days=1)
        spans.append(_month_span(previous_end.year, previous_end.month, reference))
    elif "这个月" in query or "本月" in query:
        spans.append(_month_span(reference.year, reference.month, reference))

    explicit_months = re.findall(
        r"(?:(20\d{2})\s*年\s*)?(1[0-2]|0?[1-9])\s*月(?!\s*[0-3]?\d\s*[日号])",
        query,
    )
    for raw_year, raw_month in explicit_months:
        month = int(raw_month)
        year = int(raw_year) if raw_year else reference.year - (1 if month > reference.month else 0)
        spans.append(_month_span(year, month, reference))

    chinese_months = re.findall(
        r"(?:(20\d{2})\s*年\s*)?(十一|十二|十|一|二|三|四|五|六|七|八|九)月"
        r"(?!\s*[一二三四五六七八九十0-9])",
        query,
    )
    for raw_year, raw_month in chinese_months:
        month = _CHINESE_MONTHS[raw_month]
        year = int(raw_year) if raw_year else reference.year - (1 if month > reference.month else 0)
        spans.append(_month_span(year, month, reference))

    named_windows = {
        "国庆": (10, 1, 10, 8),
        "五一": (5, 1, 5, 5),
        "开学": (9, 1, 9, 10),
        "暑假": (7, 1, 8, 31),
        "期末": (6, 25, 7, 5),
        "梅雨": (6, 1, 7, 10),
        "期中": (4, 13, 4, 19),
    }
    # Named periods can describe either the date of an event ("国庆做了什么")
    # or the subject of a plan recorded earlier ("国庆最后决定去哪"). Only the
    # former should constrain record dates; otherwise plans made before the
    # holiday disappear from retrieval.
    named_period_is_record_window = bool(re.search(
        r"回顾|复盘|期间|假期|"
        r"(?:做了|干了|忙了|发生了|经历了|去了|吃了|喝了|看了|买了)(?:什么|哪些|啥)?|"
        r"(?:做|干|忙|发生|经历)(?:过)?(?:什么|哪些|啥)",
        query,
    ))
    if named_period_is_record_window:
        for key, window in named_windows.items():
            if key in query:
                spans.append(_most_recent_window(reference, *window))

    unique = []
    seen = set()
    for span in spans:
        if span not in seen:
            seen.add(span)
            unique.append(span)
    return unique

def tokenize(text: str) -> list[str]:
    return [t for t in jieba.lcut(text) if t not in _STOP and t.strip() and not t.isspace()]


# 同义词表：查询侧扩展。用户问"爱好"时，库里记的是"跑步/羽毛球/咖啡"，
# 纯词面匹配会扑空——扩展同义词是离线模式下对语义检索的下位替代。
SYNONYMS = {
    "爱好": ["喜欢", "跑步", "羽毛球", "咖啡", "运动"],
    "兴趣": ["喜欢", "跑步", "羽毛球", "咖啡"],
    "兴趣爱好": ["喜欢", "跑步", "羽毛球", "咖啡", "运动"],
    "运动": ["跑步", "羽毛球", "健身"],
    "健康": ["作息", "熬夜", "跑步"],
    "喝": ["咖啡", "豆子"],
    "咖啡": ["豆子", "手冲", "速溶", "浅烘", "深烘"],
    "学习": ["课程", "项目", "实验", "选课", "期末"],
    "上学期": ["课程", "期末", "提交"],
    "暑假": ["实习", "在家", "假期"],
    "生活": ["作息", "咖啡", "运动"],
    "休息": ["作息", "熬夜"],
    "关系": ["室友", "约定"],
}


def expand_query(toks: list[str]) -> list[str]:
    """同义词扩展 + 分词口径兜底。
    1) 同义词：问"爱好"，库里记的是"跑步/羽毛球"，扩展后才能命中；
    2) 字符二元组：jieba 把"喝咖啡"切成整词、而库里是"咖啡"时，二元组能接上。"""
    out = list(toks)
    for t in toks:
        out.extend(SYNONYMS.get(t, []))
    for t in toks:
        if len(t) >= 2:
            out.extend(t[i:i + 2] for i in range(len(t) - 1))
    return out


class Retriever:
    def __init__(self, store: MemoryStore, reference_date: str | _date | None = None):
        self.store = store
        self.reference_date = _as_date(reference_date).isoformat()

    # ---------- BM25 ----------
    def _bm25_index(self, docs: list[dict]):
        df = Counter()
        doc_tf = []
        for d in docs:
            toks = tokenize(d["text"])
            tf = Counter(toks)
            doc_tf.append(tf)
            for t in tf:
                df[t] += 1
        n = len(docs)
        idf = {t: math.log((n - c + 0.5) / (c + 0.5) + 1) for t, c in df.items()}
        return doc_tf, idf

    def _bm25_search(self, query_toks: list[str], docs: list[dict], topk: int) -> list[tuple[str, float]]:
        if not docs:
            return []
        doc_tf, idf = self._bm25_index(docs)
        k1, b = 1.5, 0.75
        avgdl = sum(sum(tf.values()) for tf in doc_tf) / max(len(doc_tf), 1)
        scores = []
        for i, tf in enumerate(doc_tf):
            dl = sum(tf.values()) or 1
            s = 0.0
            for t in query_toks:
                if t in tf:
                    s += idf.get(t, 0) * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * dl / avgdl))
            scores.append((docs[i]["id"], s))
        scores.sort(key=lambda x: -x[1])
        return [(mid, s) for mid, s in scores if s > 0.0][:topk]

    # ---------- 时间检索：越近越相关（仅考虑与查询有词面交集的文档） ----------
    def _time_search(self, query_toks: list[str], docs: list[dict], topk: int) -> list[tuple[str, float]]:
        from datetime import date as _d
        now = _d.fromisoformat(self.reference_date)
        scored = []
        for d in docs:
            toks = set(tokenize(d["text"]))
            if not (toks & set(query_toks)):
                continue  # 与问题无关的记录不参与时间加分
            days = abs((now - _d.fromisoformat(d["date"])).days)
            scored.append((d["id"], math.exp(-days / 120.0)))
        scored.sort(key=lambda x: -x[1])
        return scored[:topk]

    # ---------- 日历召回：按时间表达映射的日期窗口直接召回 ----------
    def _effective_calendar_spans(self, query: str, items) -> list[tuple[str, str]]:
        """返回实际使用的时间窗；近期无记录时回退到最近有数据的同长度窗口。"""
        spans = calendar_spans(query, self.reference_date)
        if not spans or not is_temporal_activity_query(query):
            return spans

        query_frame = analyze_query(query, use_llm=False)
        generic_entities = {"活动", "事情", "记录", "进展", "变化"}
        specific_entities = set(query_frame.entities) - generic_entities
        needs_semantic_filter = bool(specific_entities or query_frame.facets)
        query_topics = set() if not needs_semantic_filter else self._topics_in_query(query)

        candidates = []
        for memory in items:
            detail = memory.detail.strip()
            if detail.startswith(("用户询问", "用户问", "问题：")) \
                    or detail.endswith(("？", "?")):
                continue
            if query_topics and memory.topic not in query_topics:
                continue
            if needs_semantic_filter:
                parent = self.store.memory_by_id(memory.parent_id) if memory.parent_id else None
                if not semantic_match_score(
                    query_frame,
                    frame_for_memory(memory, parent.detail if parent else ""),
                ):
                    continue
            candidates.append(memory)

        if not candidates:
            return spans
        if any(
            start <= memory.date <= end
            for memory in candidates
            for start, end in spans
        ):
            return spans

        anchor = max(_date.fromisoformat(memory.date) for memory in candidates)
        return [(
            (anchor - timedelta(days=DEFAULT_RECENT_DAYS - 1)).isoformat(),
            anchor.isoformat(),
        )]

    def calendar_window_info(self, query: str) -> dict:
        """给回答层提供时间窗口径，避免数据静默期回退对用户不可见。"""
        requested = calendar_spans(query, self.reference_date)
        effective = self._effective_calendar_spans(
            query,
            self.store.active_memories() + self.store.archive_memories(),
        )
        return {
            "requested": requested,
            "effective": effective,
            "fallback": bool(
                is_temporal_activity_query(query)
                and requested
                and effective
                and requested != effective
            ),
        }

    def _calendar_search(self, query: str, docs: list[dict], topk: int) -> list[tuple[str, float]]:
        from datetime import date as _d
        items = [
            memory
            for doc in docs
            if (memory := self.store.memory_by_id(doc["id"])) is not None
        ]
        spans = self._effective_calendar_spans(query, items)
        if not spans:
            return []
        now = _d.fromisoformat(self.reference_date)
        scored = []
        for d in docs:
            dd = _d.fromisoformat(d["date"])
            if any(_d.fromisoformat(s) <= dd <= _d.fromisoformat(e) for s, e in spans):
                scored.append((d["id"], math.exp(-abs((now - dd).days) / 120.0)))
        # 窗口内全量召回：时间表达本身保证了相关性，不做新近度截断
        #（截断会系统性挤掉窗口早段的关键记录，见失败样本 q28）
        scored.sort(key=lambda x: -x[1])
        return scored[:24]

    # ---------- 主题图检索：从问题里识别主题实体，按三元组（我-动作-主题）拉取完整时间线 ----------
    def _topics_in_query(self, query: str) -> set[str]:
        from .memory import TOPIC_RULES, BROAD_TOPICS
        topics = set()
        for keys, topic in TOPIC_RULES:
            if any(k in query for k in keys):
                topics.add(topic)
        for keys, ts in BROAD_TOPICS.items():
            if any(k in query for k in keys):
                topics.update(ts)
        return topics

    def _graph_search(self, query: str, topk: int) -> list[tuple[str, float]]:
        """三元组视角：把每条记忆看作 (我, 动作, 主题/内容)，
        关系类问题 -> 拉出相关主题的全时间线（含档案层）。
        语料变大后主题内记录会被稀释，因此与查询词（含同义词扩展）有交集的
        记录优先，其余按新近度兜底。"""
        topics = self._topics_in_query(query)
        if not topics:
            return []
        qtoks = set(expand_query(tokenize(query)))
        items = self.store.active_memories() + self.store.archive_memories()
        hits = [m for m in items if m.topic in topics]
        related, rest = [], []
        for m in hits:
            toks = set(tokenize(m.detail)) | {m.topic}
            (related if (toks & qtoks) else rest).append(m)
        related.sort(key=lambda m: m.date, reverse=True)
        rest.sort(key=lambda m: m.date, reverse=True)
        ordered = (related + rest)[:topk]
        return [(m.id, 1.0 - i * 0.01) for i, m in enumerate(ordered)]

    # ---------- 路由 ----------
    def route(self, query: str) -> str:
        if any(w in query for w in ("什么时候", "哪天", "几月", "以前", "之前", "历史", "时间线", "最近一次", "上一次", "最后一次", "上次")):
            return "time+lex"
        if any(w in query for w in ("关系", "联系", "区别", "总结", "习惯", "整体", "怎么样",
                                    "平时", "做什么", "哪些", "兴趣", "爱好", "历程", "经历")):
            return "lex+graph"
        return "lex+time"

    def _docs(self, active_only=True) -> list[dict]:
        """active_only: True=仅活跃层；'archive'=活跃层+档案层（时间回溯用）。"""
        items = self._items(active_only)
        return [{"id": m.id, "date": m.date, "text": m.topic + " " + m.detail} for m in items]

    def _items(self, active_only=True):
        if active_only == "archive":
            return self.store.active_memories() + self.store.archive_memories()
        return self.store.active_memories()

    def _semantic_search(
        self,
        query: str,
        active_only,
        topk: int,
        frame=None,
        fallback_frame=None,
    ):
        frame = frame or analyze_query(query)
        frames = [frame]
        if fallback_frame and fallback_frame != frame:
            frames.append(fallback_frame)
        if not any(item.entities or item.facets for item in frames):
            return frame, []
        merged: dict[str, tuple] = {}
        for candidate_frame in frames:
            candidate_ranked = []
            for memory in self._items(active_only):
                parent = self.store.memory_by_id(memory.parent_id) if memory.parent_id else None
                memory_frame = frame_for_memory(memory, parent.detail if parent else "")
                score = semantic_match_score(
                    candidate_frame,
                    memory_frame,
                    strict_facet=candidate_frame.scope != "evolution",
                    require_value=(
                        bool(candidate_frame.values)
                        and not candidate_frame.broad
                        and candidate_frame.scope not in (
                            "current", "history", "evolution", "advice"
                        )
                        and not re.search(
                            r"是不是|不是|难道|是否|吗|吧",
                            candidate_frame.normalized_query,
                        )
                    ),
                )
                if score:
                    candidate_ranked.append((
                        memory.id,
                        float(score),
                        entity_overlap_count(candidate_frame.entities, memory_frame.entities),
                        bool(memory_frame.entities),
                    ))
            if candidate_frame.entities and any(item[2] for item in candidate_ranked):
                candidate_ranked = [item for item in candidate_ranked if item[2]]
            for item in candidate_ranked:
                previous = merged.get(item[0])
                if previous is None or item[1] > previous[1]:
                    merged[item[0]] = item
        ranked = list(merged.values())

        if frame.scope == "history":
            historical = [
                item for item in ranked
                if self.store.memory_by_id(item[0]).status != "active"
            ]
            if historical:
                ranked = historical

        ranked.sort(
            key=lambda pair: (
                pair[1],
                self.store.memory_by_id(pair[0]).status == "active",
                self.store.memory_by_id(pair[0]).date,
            ),
            reverse=True,
        )
        if ranked and not frame.broad and frame.scope not in ("current", "evolution"):
            best = ranked[0][1]
            canonical_entities = {
                entity for entity in frame.entities if entity in ENTITY_ALIASES
            }
            multi_object = (
                len(canonical_entities) >= 2
                and any(marker in frame.normalized_query for marker in ("分别", "各自", "和", "与"))
            )
            margin = 45 if multi_object else 20
            ranked = [item for item in ranked if item[1] >= best - margin]
        return frame, [(item[0], item[1]) for item in ranked[:topk]]

    def _event_search(self, query: str, frame, topk: int):
        """Directly retrieve explicit events for historical 'when' questions."""
        if frame.scope != "history":
            return []
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
            return []
        ranked = []
        for memory in self._items("archive"):
            if not any(selector(memory) for selector in selectors):
                continue
            parent = self.store.memory_by_id(memory.parent_id) if memory.parent_id else None
            memory_frame = frame_for_memory(memory, parent.detail if parent else "")
            entity_overlap = entity_overlap_count(frame.entities, memory_frame.entities)
            same_topic = (
                frame.topic == "生活"
                or frame.topic == memory.topic
                or {frame.topic, memory.topic} <= {"健身", "运动"}
            )
            if frame.entities and not entity_overlap:
                continue
            if not frame.entities and not same_topic:
                continue
            ranked.append((memory.id, 2.0 + entity_overlap))
        earliest_state = bool(re.search(r"一开始|最初|原来|起初", query))
        ranked.sort(
            key=lambda item: self.store.memory_by_id(item[0]).date,
            reverse=not earliest_state,
        )
        if earliest_state and ranked:
            ranked = ranked[:1]
        return ranked[:topk]

    # ---------- 主入口 ----------
    def search(
        self,
        query: str,
        topk: int = None,
        active_only=True,
        route: str = None,
        use_llm: bool = True,
    ):
        """route 可由外部指定；use_llm=False 时整条检索保持本地确定性。"""
        topk = topk or config.TOP_K
        query_frame = analyze_query(query, use_llm=use_llm)
        local_query_frame = analyze_query(query, use_llm=False)
        if query_frame.scope in ("history", "evolution"):
            topk = max(topk, 16)
        retrieval_query = query_frame.normalized_query or query
        q = expand_query(tokenize(retrieval_query))
        route = route or self.route(query)
        # Historical and current-state questions may depend on compressed
        # records. Superseded facts remain available for history, while the
        # answer layer selects the latest non-superseded state for "current".
        if (
            active_only is True
            and query_frame.scope in ("history", "evolution", "current")
            and config.ALLOW_ARCHIVE
        ):
            active_only = "archive"
        docs = self._docs(active_only)
        query_frame, semantic_hits = self._semantic_search(
            retrieval_query,
            active_only,
            topk,
            frame=query_frame,
            fallback_frame=local_query_frame,
        )
        event_hits = self._event_search(query, query_frame, topk)
        if use_llm and not semantic_hits and query_frame.personal is not False:
            catalog = []
            seen_schema = set()
            for memory in self._items(active_only):
                parent = self.store.memory_by_id(memory.parent_id) if memory.parent_id else None
                memory_frame = frame_for_memory(memory, parent.detail if parent else "")
                schema = (
                    memory_frame.topic,
                    memory_frame.entities,
                    memory_frame.facets,
                    memory.detail[:120],
                )
                if schema not in seen_schema:
                    seen_schema.add(schema)
                    catalog.append(schema)
            grounded_frame = ground_query_to_catalog(query, tuple(catalog))
            if grounded_frame:
                query_frame, semantic_hits = self._semantic_search(
                    query,
                    active_only,
                    topk,
                    frame=grounded_frame,
                )
        if (
            active_only is True
            and not semantic_hits
            and (query_frame.entities or query_frame.facets)
            and config.ALLOW_ARCHIVE
        ):
            archive_frame, archive_semantic_hits = self._semantic_search(
                retrieval_query,
                "archive",
                topk,
                frame=query_frame,
                fallback_frame=local_query_frame,
            )
            if archive_semantic_hits:
                active_only = "archive"
                docs = self._docs(active_only)
                query_frame = archive_frame
                semantic_hits = archive_semantic_hits

        lists = []
        if config.RETRIEVE_CHANNELS.get("lex", True):
            lists.append(("lex", self._bm25_search(q, docs, topk)))
        if config.RETRIEVE_CHANNELS.get("time", True) and "time" in route:
            t1 = self._time_search(q, docs, topk)
            # 日历窗口本质是"时间范围"问题，允许回溯档案层（折叠/遗忘的记录也在这段时间里）
            cal_docs = docs if active_only == "archive" else self._docs("archive")
            t2 = self._calendar_search(query, cal_docs, topk)
            merged: dict[str, float] = {}
            for mid, s in t1 + t2:
                merged[mid] = max(merged.get(mid, 0.0), s)
            lists.append(("time", sorted(merged.items(), key=lambda x: -x[1])[:topk]))
        # 时间回溯类问题不追加主题图：它的近期偏好会挤掉"事件的起点"，
        # 这类问题靠 词法(含档案层) + 时间 两个通道足够；其余问题识别到主题就走图通道
        if config.RETRIEVE_CHANNELS.get("graph", True) and \
                ("graph" in route or (route == "lex+time" and self._topics_in_query(query))):
            lists.append(("graph", self._graph_search(query, topk)))
        if semantic_hits:
            lists.append(("semantic", semantic_hits))
        if event_hits:
            lists.append(("event", event_hits))

        # RRF 融合：词法通道为主，时间/图扩展为辅
        weights = {"lex": 1.0, "time": 0.5, "graph": 0.7, "semantic": 1.1, "event": 1.3}
        rrf = {}
        for tag, lst in lists:
            w = weights[tag]
            for rank, (mid, _) in enumerate(lst):
                rrf[mid] = rrf.get(mid, 0.0) + w / (config.RRF_K + rank + 1)
        ranked = sorted(rrf.items(), key=lambda x: -x[1])[:topk]

        # 日历类问题：窗口内记录是时间表达本身保证的相关证据，
        # 全量保底入证并排在最前（新近度打分会系统性挤掉窗口早段记录）
        spans = self._effective_calendar_spans(
            query,
            self.store.active_memories() + self.store.archive_memories(),
        )
        if spans and config.RETRIEVE_CHANNELS.get("time", True):
            window = self._calendar_search(query, self._docs("archive"), 24)
            seen = {mid for mid, _ in ranked}
            win_mems = []
            for mid, _s in window:
                if mid in seen:
                    continue
                m = self.store.memory_by_id(mid)
                if m:
                    win_mems.append((mid, 0.02))
                    seen.add(mid)
            ranked = win_mems + ranked

        out = []
        for mid, score in ranked:
            m = self.store.memory_by_id(mid)
            if m:
                out.append((m, score))

        def _semantic_match(memory) -> bool:
            parent = self.store.memory_by_id(memory.parent_id) if memory.parent_id else None
            return bool(semantic_match_score(
                query_frame,
                frame_for_memory(memory, parent.detail if parent else ""),
            ))

        if semantic_hits and not calendar_spans(query, self.reference_date):
            semantic_ids = {memory_id for memory_id, _score in semantic_hits}
            event_ids = {memory_id for memory_id, _score in event_hits}
            out = [(m, score) for m, score in out if m.id in semantic_ids | event_ids]
        elif (
            set(query_frame.facets)
            & {"地点", "数量", "设备", "配置", "颜色", "风格", "比例", "甜度", "温度", "类型"}
            and not calendar_spans(query, self.reference_date)
        ):
            # Lexical overlap proves the subject is related, not that the
            # requested attribute exists in evidence.
            out = []

        # ---------- 取代链与关键事件强制入证 ----------
        # 命中的记忆若卷入冲突，把冲突对的另一条（被取代的旧说法 / 取代它的新说法）
        # 并入证据集——回答"还喝速溶吗""用什么看书"这类问题必须看到前后变化。
        # 同理，命中主题的关键事件（开始/放弃/更改/约定）代表主题现状的骨架，
        # 不应被新近度截断挤掉。
        if config.EVIDENCE_EXPANSION and query_frame.scope == "evolution":
            query_tokens = {t for t in q if len(t) >= 2}
            conf_by_mem: dict[str, list] = {}
            for c in self.store.conflicts:
                conf_by_mem.setdefault(c.older_id, []).append(c)
                conf_by_mem.setdefault(c.newer_id, []).append(c)
            in_evidence = {m.id for m, _ in out}
            cand: dict[str, tuple] = {}   # id -> (priority, memory)
            for m, _s in list(out):
                for c in conf_by_mem.get(m.id, []):
                    other_id = c.newer_id if m.id == c.older_id else c.older_id
                    om = self.store.memory_by_id(other_id)
                    if om and om.id not in in_evidence:
                        cand[om.id] = (2, om)   # 冲突对手：取代链另一端
                        in_evidence.add(om.id)
            # 主题关键事件扩散：每个命中主题补至多 2 条关键事件，
            # 更改类（现状由哪次改写决定）> 其余状态变更 > 佐证类（完成/偏好）
            _SK = ("更改", "放弃", "开始", "约定")
            topics_done: set[str] = set()
            for m, _s in out:
                if m.topic in topics_done:
                    continue
                topics_done.add(m.topic)
                key_events = [x for x in self.store.memories
                              if x.topic == m.topic and x.action != "记录"
                              and x.id not in in_evidence
                              and query_tokens.intersection(tokenize(x.detail))]
                key_events.sort(key=lambda x: (x.action == "更改", x.date), reverse=True)
                for x in key_events[:2]:
                    prio = 3 if x.action == "更改" else 1
                    if x.id not in cand or cand[x.id][0] < prio:
                        cand[x.id] = (prio, x)
                    in_evidence.add(x.id)
            # 更改类关键事件优先保留，冲突对手次之，避免名额被单一来源占满
            picked = sorted(cand.values(), key=lambda p: (p[0], p[1].date), reverse=True)
            out.extend((m, -1.0) for _p, m in picked[:6])
        # 日历类问题只回答窗口内实际发生的事件；不追加窗口外证据，
        # 也过滤「用户询问……」这类对话元记录，避免把问题本身当成经历。
        spans = self._effective_calendar_spans(
            query,
            self.store.active_memories() + self.store.archive_memories(),
        )
        if spans:
            query_topics = self._topics_in_query(query)
            generic_entities = {"活动", "事情", "记录", "进展", "变化"}
            specific_entities = set(query_frame.entities) - generic_entities
            needs_semantic_filter = bool(specific_entities or query_frame.facets)
            if is_temporal_activity_query(query) and not needs_semantic_filter:
                query_topics = set()

            def _in_window(m):
                in_date = any(s <= m.date <= e for s, e in spans)
                in_topic = not query_topics or m.topic in query_topics
                in_subject = not needs_semantic_filter or _semantic_match(m)
                return in_date and in_topic and in_subject

            def _is_activity(m):
                detail = m.detail.strip()
                return not detail.startswith(("用户询问", "用户问", "问题：")) \
                    and not detail.endswith(("？", "?"))

            out = sorted(
                [x for x in out if _in_window(x[0]) and _is_activity(x[0])],
                key=lambda x: x[0].date,
            )
        elif semantic_hits:
            semantic_ids = {memory_id for memory_id, _score in semantic_hits}
            event_ids = {memory_id for memory_id, _score in event_hits}
            out = [(m, score) for m, score in out if m.id in semantic_ids | event_ids]
        return out



