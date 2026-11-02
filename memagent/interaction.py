"""Conservative conversational intent parsing; this module never writes memory."""
from __future__ import annotations

import re
from dataclasses import dataclass

from . import llm
from .memory import _ALLOWED_TOPICS
from .memory_policy import looks_like_question
from .semantics import analyze_semantics, infer_topic


_TIME = (
    r"(?:刚才|刚刚|最近|上周|上个月|今天|昨天|前天|明天|后天|今晚|昨晚|今早|明早|本周|这周|下周|"
    r"周[一二三四五六日天]|星期[一二三四五六日天]|"
    r"(?:20\d{2}年)?\d{1,2}月\d{1,2}[日号]?)"
)
_TEMPORAL_PERSONAL = re.compile(
    rf"(?:我|本人|我们).{{0,24}}{_TIME}.{{0,100}}|"
    rf"{_TIME}.{{0,24}}(?:我|本人|我们).{{0,100}}",
    re.S,
)
_PERSONAL_ACTION = re.compile(
    r"(?:想吃|想喝|想去|想看|想买|想玩|准备|打算|计划|要去|要做|去|在|有|见|参加|完成|"
    r"会去|将要|正在|去了|吃了|喝了|买了|看了|做了|完成了|跑步|运动|"
    r"复诊|答辩|开会|提交|参加|出发|预约)"
)
_STABLE_PREFERENCE = re.compile(
    r"(?:我|本人).{0,20}(?:一直|平时|通常|向来|长期|最|更)?"
    r"(?:喜欢|偏好|偏爱|钟爱|爱吃|爱喝|爱(?!好)|习惯|不喜欢|讨厌).{1,80}",
    re.S,
)
_UNCERTAIN_PREFERENCE = re.compile(
    r"(?:我|本人).{0,24}(?:可能|也许|好像|似乎|不确定|说不准|暂时觉得)"
    r".{0,24}(?:喜欢|偏好|爱吃|爱喝|习惯|不喜欢|讨厌)|"
    r"(?:我|本人).{0,24}(?:可能|也许|好像|似乎|不确定|说不准|暂时觉得)"
    r".{0,60}(?:口味|偏好|习惯|更清淡|更辣|更甜)",
    re.S,
)
_LONG_TERM_PLAN = re.compile(
    r"(?:我|本人).{0,24}(?:以后|今后|长期|每天|每周|每月).{1,80}",
    re.S,
)
_DECISION = re.compile(
    r"^\s*(?:那|那么)?\s*(?:就|还是|决定|选|定|改成|换成|要)"
    r".{1,32}(?:吧|了|就好|即可)?[。.!！]?\s*$"
)
_QUERY_SPLIT = re.compile(
    r"[，,；;。]\s*(?=(?:你说|你觉得|你看|帮我|给我|建议|推荐|该|应该|怎么|如何|选什么|吃什么|喝什么))"
)
_PREFERENCE_TARGET = re.compile(
    r"(?:想吃|想喝|爱吃|爱喝|偏爱|钟爱|爱(?!好)|喜欢|偏好|不喜欢|讨厌)"
    r"\s*([^，,。！？?；;]{1,40})"
)
_PREFERENCE_SIGNAL = re.compile(
    r"(?:喜欢|偏好|偏爱|钟爱|最爱|爱吃|爱喝|爱(?!好)|合口味|不喜欢|讨厌)"
)
_CHOICE_PREFIX = re.compile(r"^\s*(?:那|那么)?\s*(?:就|还是|决定|选|定|改成|换成|要)\s*")
_THIRD_PARTY = re.compile(
    r"(?:我(?:的)?(?:孩子|女儿|儿子|妻子|丈夫|伴侣|朋友|同事|室友|老师|父母|妈妈|爸爸|家人)|^他|^她|^他们|^她们)"
)
_UNTRUSTED_CLAIM_CONTEXT = re.compile(
    r"(?:假设|假如|比如|例如|举例|引用|原文|截图里|文档里|文件里|"
    r"这句话|这段话|开玩笑|我不是说|不是我本人|替朋友|朋友说)"
)
_SENSITIVE = re.compile(
    r"(?i)(?:\bsk-[A-Za-z0-9_-]{20,}\b|\bgh[pousr]_[A-Za-z0-9]{20,}\b|"
    r"\bAIza[A-Za-z0-9_-]{30,}\b|Bearer\s+\S{20,}|api[_ -]?key|access[_ -]?token|"
    r"password|passwd|密钥|口令|密码)"
)
_PREFERENCE_QUESTION = re.compile(
    r"(?:我|我的|本人).{0,20}(?:喜欢|偏好|口味|习惯).{0,16}(?:什么|哪种|哪个|哪些|怎么样)|"
    r"(?:我|我的|本人).{0,16}(?:是什么口味|平时吃什么|偏好是什么)|"
    r"(?:我|我的|本人).{0,12}(?:有|有哪些|是什么).{0,12}(?:偏好|口味|习惯)",
    re.S,
)


@dataclass
class InteractionIntent:
    kind: str = "none"
    record_text: str = ""
    query_text: str = ""
    topic: str = "生活"
    subject: str = ""
    preference_key: str = ""
    reply: str = ""
    needs_answer: bool = False
    parent_id: str = ""
    method: str = "rule"
    confidence: float = 0.0
    reason: str = ""


def _compact(text: str) -> str:
    return " ".join((text or "").split()).strip(" ，,。.!！")


def _topic(text: str) -> str:
    return infer_topic(text)


def _preference_target(text: str) -> str:
    match = _PREFERENCE_TARGET.search(text)
    if not match:
        return ""
    value = _compact(match.group(1))
    value = re.sub(r"(?:今天|昨天|明天|后天|今晚|吧|了)$", "", value).strip()
    return value[:40]


def _subject(text: str) -> str:
    target = _preference_target(text)
    if target:
        return target
    patterns = (
        r"(?:要去|想去|准备去|计划去)\s*([^，,。！？?；;]{1,32})",
        r"(?:要做|准备|打算|计划)\s*([^，,。！？?；;]{1,32})",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return _compact(match.group(1))
    frame = analyze_semantics(text)
    if frame.entities:
        return frame.entities[0]
    return ""


def _decision_choice(text: str) -> str:
    value = _CHOICE_PREFIX.sub("", _compact(text))
    return value.removesuffix("吧").removesuffix("了").strip(" ，,。")


def classify_interaction(
    text: str,
    previous_episode: dict | None = None,
) -> InteractionIntent:
    """Identify high-confidence memory events without granting storage authority."""
    value = _compact(text)
    if (
        not value
        or _SENSITIVE.search(value)
        or _THIRD_PARTY.search(value)
        or _UNTRUSTED_CLAIM_CONTEXT.search(value)
    ):
        return InteractionIntent()

    if _PREFERENCE_QUESTION.search(value):
        return InteractionIntent(
            kind="preference_query",
            query_text=value,
            topic=_topic(value),
            needs_answer=True,
        )

    if previous_episode and _DECISION.match(value) and not looks_like_question(value):
        choice = _decision_choice(value)
        if choice:
            subject = str(previous_episode.get("subject") or "")
            topic = str(previous_episode.get("topic") or "生活")
            detail = f"关于{subject}，本次选择{choice}" if subject else f"本次选择{choice}"
            return InteractionIntent(
                kind="decision",
                record_text=detail,
                topic=topic,
                subject=subject,
                preference_key=choice,
                reply=f"好，这次就定{choice}。",
                parent_id=str(previous_episode.get("memory_id") or ""),
            )

    if _UNCERTAIN_PREFERENCE.search(value) and not looks_like_question(value):
        return InteractionIntent(
            kind="inferred_preference_candidate",
            record_text=value,
            topic=_topic(value),
            subject=_preference_target(value),
            preference_key=_preference_target(value),
        )

    stable = _STABLE_PREFERENCE.search(value)
    if stable and not looks_like_question(value):
        target = _preference_target(value)
        if target:
            return InteractionIntent(
                kind="stable_preference",
                record_text=value,
                topic=_topic(value),
                subject=target,
                preference_key=target,
                reply=f"记下了，你明确提到偏好{target}。",
            )

    if _LONG_TERM_PLAN.search(value) and not looks_like_question(value):
        return InteractionIntent(
            kind="long_term_plan",
            record_text=value,
            topic=_topic(value),
            subject=_subject(value),
            reply=f"已记下这项长期安排：{value}。",
        )

    parts = re.split(
        r"[，,；;。]\s*(?=(?:你说|你觉得|你看|帮我|给我|建议|推荐|该|应该|怎么|如何|选什么|吃什么|喝什么))",
        value,
        maxsplit=1,
    )
    event = _compact(parts[0]) if len(parts) > 1 else value
    asks = looks_like_question(value)
    has_personal_time = bool(_TEMPORAL_PERSONAL.search(event))
    is_personal_action = bool(re.search(r"(?:我|本人|我们)", event)) and bool(
        _PERSONAL_ACTION.search(event)
    )
    immediate_advice_event = len(parts) > 1 and is_personal_action
    if (
        event
        and not looks_like_question(event)
        and is_personal_action
        and (has_personal_time or immediate_advice_event)
    ):
        subject = _subject(event)
        return InteractionIntent(
            kind="mixed_request" if asks else "event",
            record_text=event,
            query_text=value if asks else "",
            topic=_topic(event),
            subject=subject,
            preference_key=_preference_target(event),
            reply="" if asks else f"记下了：{event}。",
            needs_answer=asks,
        )

    return InteractionIntent()


def classify_interaction_online(
    text: str,
    previous_episode: dict | None = None,
) -> InteractionIntent:
    """Use the model only for ambiguous first-person events; validate its claims."""
    value = _compact(text)
    if (
        not value
        or not llm.is_enabled()
        or _SENSITIVE.search(value)
        or _THIRD_PARTY.search(value)
        or _UNTRUSTED_CLAIM_CONTEXT.search(value)
        or not re.search(r"(?:我|我的|本人|我们)", value)
        or not (
            _TEMPORAL_PERSONAL.search(value)
            or previous_episode
            or _PREFERENCE_SIGNAL.search(value)
            or _LONG_TERM_PLAN.search(value)
        )
    ):
        return InteractionIntent()

    try:
        verdict = llm.chat_json([
            {
                "role": "system",
                "content": (
                    "判断用户本轮话语是否包含可记录的本人事实、明确偏好或需确认的偏好线索，"
                    "并将事实与提问分开。只返回 JSON："
                    '{"kind":"none|event|mixed_request|stable_preference|'
                    'inferred_preference_candidate|long_term_plan_candidate",'
                    '"fact_clause":"必须是用户原话中连续、逐字出现的片段",'
                    '"topic":"健身/运动/饮食/作息/课程/社交/阅读/旅行/理财/健康/生活",'
                    '"subject":"原话明确涉及的对象",'
                    '"preference_key":"原话明确表达的偏好值",'
                    '"confidence":0到1,"reason":"简短理由"}。'
                    "问句、通用知识、第三方事实、附件内容不得成为用户记忆。"
                    "明确日期的本人事件可记为 event；同时提问则为 mixed_request。"
                    "单次决定不是稳定偏好；不确定偏好只能作为待确认候选。"
                    "明确长期偏好可为 stable_preference；反复计划只能为待确认候选。"
                    "没有把握时 kind 必须为 none。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"最近明确事件上下文：{previous_episode or '无'}\n"
                    f"用户原话：{text}"
                ),
            },
        ])
        kind = str(verdict.get("kind") or "none")
        confidence = float(verdict.get("confidence", 0))
        if kind not in {
            "event",
            "mixed_request",
            "stable_preference",
            "inferred_preference_candidate",
            "long_term_plan_candidate",
        } or confidence < 0.82:
            return InteractionIntent()

        fact = str(verdict.get("fact_clause") or "").strip()
        normalized_text = " ".join(text.split())
        normalized_fact = " ".join(fact.split())
        if not fact or normalized_fact not in normalized_text:
            return InteractionIntent()
        if _SENSITIVE.search(fact) or _THIRD_PARTY.search(fact):
            return InteractionIntent()

        topic = str(verdict.get("topic") or "生活")
        if topic not in _ALLOWED_TOPICS:
            topic = _topic(fact)
        subject = str(verdict.get("subject") or "").strip()[:48]
        preference_key = str(verdict.get("preference_key") or "").strip()[:48]
        if subject and subject not in normalized_text:
            subject = _subject(fact)
        if preference_key and preference_key not in normalized_text:
            preference_key = _preference_target(fact)

        has_question = looks_like_question(text)
        if kind in ("event", "mixed_request"):
            if not re.search(r"(?:我|本人|我们)", fact):
                return InteractionIntent()
            if not _TEMPORAL_PERSONAL.search(fact) and not _PERSONAL_ACTION.search(fact):
                return InteractionIntent()
            if has_question != (kind == "mixed_request"):
                return InteractionIntent()
            if kind == "mixed_request":
                return InteractionIntent(
                    kind=kind,
                    record_text=fact,
                    query_text=text,
                    topic=topic,
                    subject=subject,
                    preference_key=preference_key,
                    needs_answer=True,
                    method="llm",
                    confidence=confidence,
                    reason=str(verdict.get("reason") or "")[:160],
                )
            return InteractionIntent(
                kind=kind,
                record_text=fact,
                topic=topic,
                subject=subject,
                preference_key=preference_key,
                reply=f"记下了：{fact}。",
                method="llm",
                confidence=confidence,
                reason=str(verdict.get("reason") or "")[:160],
            )

        if has_question or not re.search(r"(?:我|本人|我们)", fact):
            return InteractionIntent()
        if kind == "stable_preference":
            if not _PREFERENCE_SIGNAL.search(fact):
                return InteractionIntent()
            if _UNCERTAIN_PREFERENCE.search(fact):
                kind = "inferred_preference_candidate"
        if kind in ("inferred_preference_candidate", "long_term_plan_candidate"):
            return InteractionIntent(
                kind=kind,
                record_text=fact,
                topic=topic,
                subject=subject,
                preference_key=preference_key,
                method="llm",
                confidence=confidence,
                reason=str(verdict.get("reason") or "")[:160],
            )
        target = preference_key or _preference_target(fact)
        if not target:
            return InteractionIntent()
        return InteractionIntent(
            kind="stable_preference",
            record_text=fact,
            topic=topic,
            subject=subject or target,
            preference_key=target,
            reply=f"记下了，你明确提到偏好{target}。",
            method="llm",
            confidence=confidence,
            reason=str(verdict.get("reason") or "")[:160],
        )
    except Exception:
        return InteractionIntent()
