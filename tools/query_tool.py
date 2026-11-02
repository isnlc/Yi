"""Tool 3: validate a question and read evidence from long-term memory."""

import re
from datetime import date

from memagent.agents import MemoryAgent
from memagent.retrieve import (
    calendar_spans,
    expand_query,
    is_temporal_activity_query,
    tokenize,
)


_DATE_PATTERN = re.compile(r"20\d{2}-\d{2}-\d{2}")
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z]{2,}(?:[-_]\d+|\d+)")
_TEMPORAL_ACTIVITY_TERMS = (
    "吃", "喝", "做", "干", "忙", "去", "买", "看",
    "跑步", "运动", "发生", "经历", "活动", "记录",
)

_QUERY_STOP = {
    "请", "请问", "告诉", "检索", "查找", "查询", "是否", "是不是",
    "现在", "最近", "今天", "这一天", "什么", "哪个", "怎么", "怎么样",
    "做了", "有关", "相关",
}


def _no_evidence(message: str) -> dict:
    return {"answer": message, "evidence": [], "used_conclusion": False}


def _corpus_text(agent: MemoryAgent) -> str:
    entries = (entry.text for entry in agent.store.entries)
    memories = (f"{memory.topic} {memory.detail}" for memory in agent.store.memories)
    return "\n".join([*entries, *memories]).lower()


def _validate_question(agent: MemoryAgent, question: str) -> dict | None:
    for raw_date in _DATE_PATTERN.findall(question):
        try:
            date.fromisoformat(raw_date)
        except ValueError:
            return _no_evidence("问题中的日期格式无效，无法检索记忆。")

    corpus = _corpus_text(agent)
    identifiers = _IDENTIFIER_PATTERN.findall(question)
    if identifiers and any(identifier.lower() not in corpus for identifier in identifiers):
        return _no_evidence("记忆库中没有该编号对应的记录。")

    query_tokens = {
        token
        for token in expand_query(tokenize(question))
        if len(token.strip()) >= 2
        and token not in _QUERY_STOP
        and not token.isdigit()
        and not _DATE_PATTERN.fullmatch(token)
    }
    if not query_tokens:
        if (
            is_temporal_activity_query(question)
            or (
                calendar_spans(question, agent.reference_date)
                and any(term in question for term in _TEMPORAL_ACTIVITY_TERMS)
            )
        ):
            return None
        return _no_evidence("问题中缺少可检索的主题。")
    # 日历窗口类问题（"我国庆""十月以来"）走时间窗召回，不做词面校验——
    # 分词可能把"我国庆"切成"我国/庆"，词面不命中不代表记忆里没有。
    if calendar_spans(question, agent.reference_date):
        return None
    # Do not reject solely on token substrings. Chinese tokenization can split
    # a known object differently between a question and its stored sentence
    # (for example "定投是多少" -> "定 / 投是"). The retriever performs the
    # grounded entity/facet check and is the source of truth for relevance.
    return None


def query_memory(agent: MemoryAgent, question: str, answer_mode: str = "auto") -> dict:
    """Answer a question and return the evidence selected by the retriever."""
    question = question.strip()
    if not question:
        return _no_evidence("问题为空，无法检索记忆。")
    rejected = _validate_question(agent, question)
    if rejected:
        return rejected
    return agent.answer(question, answer_mode=answer_mode)
