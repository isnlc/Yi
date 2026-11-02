"""网页端后端：FastAPI 包装 MemoryPipeline。

在"问答 + 记录"之外提供更完整的产品能力：
- 对话式主动遗忘（"帮我忘掉X" → 淡忘，遗忘≠删除，可恢复）
- 阶段回顾（"回顾一下最近一周" → LLM 生成回顾报告）
- 多轮追问（结合上一轮问答把"那第二天呢"改写成独立问题）
- 记忆矛盾提醒（录入与旧记忆冲突时，界面明确告知"以最新为准"）
- 日历视图（按月浏览记忆密度，点任意一天查看 + 随口记）
- 记忆画像（主题分布 / 记忆状态 / 记录节奏）
- 近期约定（提醒类记忆单独浮出）
- 记忆导出（JSON 下载，数据始终属于用户）

启动（项目根目录）:
    python web/app.py --port 8765              # 规则模式（离线可复现）
    MEMAGENT_LLM=1 python web/app.py --port 8765  # 在线语义增强模式
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
import threading
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from memagent import config  # noqa: E402
from memagent.llm import chat, chat_json, is_enabled, vision_chat, vision_is_enabled  # noqa: E402
from memagent.router import (  # noqa: E402
    FOLLOW_UP_MARKERS,
    classify_query,
    extract_record_content,
    is_record_command,
)
from memagent.retrieve import calendar_spans  # noqa: E402
from memagent.semantics import (  # noqa: E402
    analyze_semantics,
    enrich_memory_semantics,
    frame_for_memory,
    semantic_label,
    semantic_match_score,
    semantic_signature,
)
from memagent.memory_policy import (  # noqa: E402
    attachment_prompt_is_untrusted,
    implicit_memory_candidate,
    looks_like_question,
    sensitive_reason,
)
from memagent.interaction import (  # noqa: E402
    classify_interaction,
    classify_interaction_online,
)
from pipeline import MemoryPipeline  # noqa: E402
from tools.audit_tool import append_event  # noqa: E402
from tools.data_tool import load_records  # noqa: E402
from tools.document_tool import MAX_FILE_BYTES, DocumentParseError, ocr_path, parse_document  # noqa: E402

WEB_DIR = Path(__file__).resolve().parent
STORE_PATH = ROOT / "data" / "web_store.json"
LOG_PATH = ROOT / "logs" / "web_log.jsonl"
DATASET_PATH = ROOT / "data" / "dataset_large.jsonl"
DOCUMENT_DIR = ROOT / "data" / "documents"
DOCUMENT_CATALOG_PATH = DOCUMENT_DIR / "catalog.json"
PENDING_MEMORIES_PATH = ROOT / "data" / "pending_memories.json"
MAX_ATTACHMENT_FILES = 8
MAX_ATTACHMENT_TOTAL_BYTES = 80 * 1024 * 1024
MAX_IMAGE_BYTES = 15 * 1024 * 1024

app = FastAPI(title="长期记忆智能体")
_pipeline: MemoryPipeline | None = None
_pipeline_lock = threading.Lock()
# 单用户演示场景：保存最近几轮问答用于多轮追问
SESSION: dict = {"history": []}   # [{"q":..., "a":..., "scope":..., "episode":...}]

def get_pipeline() -> MemoryPipeline:
    global _pipeline
    current_day = date.today().isoformat()
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                first_build = not STORE_PATH.exists()
                pipeline = MemoryPipeline(
                    STORE_PATH,
                    LOG_PATH,
                    reference_date=current_day,
                )
                if first_build:
                    pipeline.ingest(load_records(DATASET_PATH))
                _normalize_preference_topics(pipeline.store)
                enrich_memory_semantics(pipeline.store)
                _pipeline = pipeline
    else:
        _pipeline.agent.reference_date = current_day
        _pipeline.agent.retriever.reference_date = current_day
    return _pipeline


# ==================== 基础 ====================

class ChatRequest(BaseModel):
    question: str


class ImportRequest(BaseModel):
    text: str
    date: str | None = None
    source: str = "随手记"


@app.get("/api/memory-candidates")
def memory_candidates():
    items = _load_pending_memories()
    items.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return {"items": items}


@app.post("/api/memory-candidates/{candidate_id}/accept")
def accept_memory_candidate(candidate_id: str):
    items = _load_pending_memories()
    candidate = next((item for item in items if item.get("id") == candidate_id), None)
    if not candidate:
        return {"ok": False, "message": "这条待确认内容不存在或已处理。"}
    reason = sensitive_reason(candidate.get("detail", ""))
    if reason:
        return {"ok": False, "message": "内容包含敏感信息，不能写入长期记忆。"}

    pipeline = get_pipeline()
    pipeline.ingest([{
        "date": candidate.get("date") or date.today().isoformat(),
        "text": candidate["detail"],
        "source": candidate.get("source") or "用户确认的候选记忆",
        "memory_type": candidate.get("memory_type", "long_term"),
        "confidence": candidate.get("confidence", 0.85),
        "preference_key": candidate.get("preference_key", ""),
        "action_override": candidate.get("action_override", ""),
        "topic_override": candidate.get("topic", ""),
        "entity_override": candidate.get("subject", ""),
    }])
    _save_json_list(
        PENDING_MEMORIES_PATH,
        [item for item in items if item.get("id") != candidate_id],
    )
    append_event(LOG_PATH, "memory_candidate_accepted", role="user", candidate_id=candidate_id)
    return {"ok": True, "message": "已确认并写入长期记忆。"}


@app.delete("/api/memory-candidates/{candidate_id}")
def dismiss_memory_candidate(candidate_id: str):
    items = _load_pending_memories()
    retained = [item for item in items if item.get("id") != candidate_id]
    if len(retained) == len(items):
        return {"ok": False, "message": "这条待确认内容不存在或已处理。"}
    _save_json_list(PENDING_MEMORIES_PATH, retained)
    append_event(LOG_PATH, "memory_candidate_dismissed", role="user", candidate_id=candidate_id)
    return {"ok": True, "message": "已忽略，不会写入长期记忆。"}


@app.get("/")
def index():
    return FileResponse(
        WEB_DIR / "index.html",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/status")
def status():
    stats = get_pipeline().store.stats()
    return {
        "llm_enabled": is_enabled(),
        "vision_enabled": vision_is_enabled(),
        "dataset": DATASET_PATH.name,
        "attachment_limits": {
            "max_files": MAX_ATTACHMENT_FILES,
            "max_total_bytes": MAX_ATTACHMENT_TOTAL_BYTES,
            "max_image_bytes": MAX_IMAGE_BYTES,
            "max_document_bytes": MAX_FILE_BYTES,
        },
        "pending_memories": len(_load_pending_memories()),
        "documents": len(_load_json_list(DOCUMENT_CATALOG_PATH)),
        **stats,
    }


def _load_json_list(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def _save_json_list(path: Path, value: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")
    temporary.replace(path)


def _load_pending_memories() -> list[dict]:
    items = _load_json_list(PENDING_MEMORIES_PATH)
    cutoff = datetime.now().astimezone() - timedelta(days=30)
    retained = []
    for item in items:
        try:
            created = datetime.fromisoformat(item["created_at"])
        except (KeyError, TypeError, ValueError):
            continue
        if created >= cutoff:
            retained.append(item)
    if len(retained) != len(items):
        _save_json_list(PENDING_MEMORIES_PATH, retained)
    return retained


def _add_memory_candidate(
    detail: str,
    *,
    day: str,
    source: str,
    reason: str,
    confidence: float = 0.65,
    memory_type: str = "long_term",
    action_override: str = "",
    preference_key: str = "",
    topic: str = "",
    subject: str = "",
) -> dict:
    candidate = {
        "id": "p" + uuid4().hex[:10],
        "date": day,
        "detail": detail.strip(),
        "source": source,
        "reason": reason,
        "confidence": round(max(0.0, min(1.0, confidence)), 2),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "memory_type": memory_type,
        "action_override": action_override,
        "preference_key": preference_key,
        "topic": topic,
        "subject": subject,
    }
    items = _load_pending_memories()
    normalized = " ".join(candidate["detail"].split()).casefold()
    if not any(" ".join(item.get("detail", "").split()).casefold() == normalized for item in items):
        items.append(candidate)
        _save_json_list(PENDING_MEMORIES_PATH, items)
    else:
        candidate = next(
            item for item in items
            if " ".join(item.get("detail", "").split()).casefold() == normalized
        )
    append_event(
        LOG_PATH,
        "memory_candidate",
        role="recorder",
        candidate_id=candidate["id"],
        source=source,
        confidence=candidate["confidence"],
        reason=reason,
    )
    return candidate


def _review_implicit_candidate(text: str) -> tuple[bool, str, float]:
    eligible, reason = implicit_memory_candidate(text)
    if not eligible:
        return False, "", 0.0
    if not is_enabled():
        return True, reason, 0.62
    try:
        verdict = chat_json([
            {
                "role": "system",
                "content": (
                    "判断这句用户直接输入是否值得作为待确认的个人记忆候选。"
                    "考虑明确属于用户本人的稳定偏好、长期计划、重要安排或带明确时间的近期个人安排。"
                    "普通对话、问题、第三方信息、敏感信息都不应成为候选。"
                    "用户明确表达但仍不确定的本人偏好，可以作为低置信待确认候选，不能当成既定事实。"
                    "你只能建议，不得决定写入。只输出 JSON："
                    '{"candidate":true/false,"confidence":0到1,"reason":"简短原因"}'
                ),
            },
            {"role": "user", "content": text},
        ])
        confidence = float(verdict.get("confidence", 0))
        if verdict.get("candidate") is True and confidence >= 0.72:
            return True, str(verdict.get("reason") or reason)[:160], confidence
        if verdict.get("candidate") is False:
            return False, "", confidence
    except Exception:
        pass
    return True, reason, 0.62


# ==================== 对话问答（含意图分发） ====================

def _is_general_query(query: str) -> bool:
    """兼容入口：统一路由判定为 general/utility 时，不进入个人记忆。"""
    return classify_query(query)["intent"] in ("general", "utility")


def _handle_utility_query(query: str) -> dict:
    now = datetime.now()
    weekday = "星期" + "一二三四五六日"[now.weekday()]
    wants_weekday = any(k in query for k in ("星期几", "周几"))
    wants_date = any(k in query for k in ("几号", "日期", "哪一天"))
    wants_time = any(k in query for k in ("几点", "时间"))
    if wants_weekday:
        answer = f"今天是 {now.year} 年 {now.month} 月 {now.day} 日，{weekday}。"
    elif wants_date:
        answer = f"今天是 {now.year} 年 {now.month} 月 {now.day} 日。"
    elif wants_time:
        answer = f"现在时间是 {now:%H:%M}。"
    else:
        answer = f"当前日期为 {now:%Y-%m-%d}，{weekday}。"
    return {"type": "utility", "answer": answer, "evidence": []}


def _handle_general_query(query: str) -> dict:
    if not is_enabled():
        return {
            "type": "general",
            "answer": _local_conversation_response(query),
            "evidence": [],
            "llm_used": False,
        }
    hist = "\n".join(
        f"用户：{h['q']}\n助手：{h['a'][:120]}"
        for h in SESSION["history"][-2:]
    )
    try:
        text = chat([
            {"role": "system", "content": (
                "你是长期记忆助手「忆」。当前问题属于通用问答，不调用个人记忆库。"
                "直接、准确、简洁地回答，不虚构用户经历。"
                "不得声称已经记住、记录、保存或更新了用户信息。")},
            {"role": "user", "content": f"近期对话：\n{hist}\n\n用户现在说：{query}"},
        ], temperature=0.6)
        return {
            "type": "general",
            "answer": text.strip(),
            "evidence": [],
            "llm_used": True,
        }
    except Exception:
        return {
            "type": "general",
            "answer": _local_conversation_response(query),
            "evidence": [],
            "llm_used": False,
        }


def _local_conversation_response(query: str) -> str:
    """Minimal local fallback for ordinary conversation without exposing runtime mode."""
    value = " ".join((query or "").split())
    if re.search(r"困|想睡|没睡醒|熬不住", value):
        return "听起来你现在很困。能休息的话先眯一会儿；还要继续做事，就先喝点水、起身活动一下。"
    if re.search(r"累|疲惫|没精神|精疲力尽", value):
        return "先停一下，做几次深呼吸，喝点水，再把手头任务缩小到下一步。"
    if re.search(r"饿|没吃饭", value):
        return "先补充一点容易消化的食物和水，再继续处理事情。"
    if re.search(r"渴|口干", value):
        return "先喝点水，稍微休息一下。"
    if re.search(r"难过|伤心|焦虑|烦躁|心情不好|不开心|压力大", value):
        return "先别急着把所有事情一次解决。可以告诉我具体发生了什么，我帮你一起理清。"
    if re.search(r"开心|高兴|顺利|太好了", value):
        return "听起来进展不错。"
    if re.fullmatch(r"(?:你好|您好|嗨|早上好|下午好|晚上好)[！!。.\s]*", value):
        return "你好。"
    if re.fullmatch(r"(?:谢谢|感谢|多谢)[你！!。.\s]*", value):
        return "不客气。"
    if looks_like_question(value):
        return "这个问题我目前无法可靠回答。可以补充更具体的对象、条件或期望结果。"
    return "我明白了。你可以继续说，或者直接告诉我希望我帮你分析、记录还是查找什么。"


def _latest_episode() -> dict | None:
    for item in reversed(SESSION["history"][-8:]):
        episode = item.get("episode")
        if episode and episode.get("memory_id"):
            return episode
    return None


def _memory_semantic_frame(store, item):
    parent = store.memory_by_id(item.parent_id) if item.parent_id else None
    return frame_for_memory(item, parent.detail if parent else "")


def _preference_query_score(store, item, query: str) -> int:
    query_frame = analyze_semantics(query, is_query=True)
    memory_frame = _memory_semantic_frame(store, item)
    return semantic_match_score(query_frame, memory_frame)


def _preference_signature(store, item) -> tuple:
    frame = _memory_semantic_frame(store, item)
    signature = semantic_signature(frame, item.preference_key or item.detail)
    # Object-specific preferences represent one current slot. Legacy stores may
    # contain several active phrasings of the same slot, so the newest one wins
    # even when an older sentence mentioned an alternative value.
    values = () if frame.entities else signature[3]
    return (
        signature[0],
        signature[1][:1],
        signature[2],
        values,
    )


def _normalize_preference_topics(store) -> None:
    """Repair legacy preference topics with the shared semantic analyser."""
    changed = False
    for item in store.memories:
        if item.action != "偏好" or item.topic != "生活":
            continue
        frame = analyze_semantics(
            f"{item.detail} {item.preference_key}",
            topic_hint=item.topic,
            preference_value=item.preference_key,
        )
        if frame.topic != "生活":
            item.topic = frame.topic
            changed = True
    if changed:
        store.save()


def _context_memory_relevance(store, item, plan) -> int:
    """Score advice evidence by the same object/attribute semantics as retrieval."""
    query = " ".join(filter(None, (plan.query_text, plan.record_text, plan.subject)))
    query_frame = analyze_semantics(
        query,
        topic_hint=plan.topic,
        is_query=True,
    )
    if not (query_frame.entities or query_frame.facets):
        return 0
    memory_frame = _memory_semantic_frame(store, item)
    score = semantic_match_score(query_frame, memory_frame)
    if not score:
        return 0

    query_entities = set(query_frame.entities)
    memory_entities = set(memory_frame.entities)
    if item.memory_type in ("decision", "preference_evidence"):
        if not query_entities or not (query_entities & memory_entities):
            return 0
        return 100
    return min(score, 90)


def _handle_preference_question(topic: str, query: str) -> dict:
    store = get_pipeline().store
    query_frame = analyze_semantics(query, topic_hint=topic, is_query=True)
    confirmed = []
    for item in store.memories:
        if (
            item.status != "active"
            or item.memory_type != "long_term"
            or item.action != "偏好"
        ):
            continue
        if topic != "生活" and item.topic not in (topic, "生活"):
            continue
        score = semantic_match_score(
            query_frame,
            _memory_semantic_frame(store, item),
        )
        if score:
            confirmed.append((score, item))
    confirmed.sort(
        key=lambda pair: (pair[0], pair[1].date, pair[1].confidence),
        reverse=True,
    )
    deduped = []
    seen = set()
    for _score, item in confirmed:
        signature = _preference_signature(store, item)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(item)
    if deduped:
        lines = [f"（{item.date}）{item.detail}" for item in deduped[:5]]
        label = semantic_label(query_frame)
        return {
            "type": "preference_answer",
            "answer": f"目前明确记录的{label}偏好有：\n" + "\n".join(f"- {line}" for line in lines),
            "evidence": lines,
            "answer_mode": "retrieval_only",
            "llm_used": False,
        }

    choices = []
    for item in store.memories:
        if (
            item.status == "active"
            and item.memory_type == "decision"
            and semantic_match_score(
                query_frame,
                _memory_semantic_frame(store, item),
            )
        ):
            if topic != "生活" and item.topic not in (topic, "生活"):
                continue
            choices.append(item)
    choices.sort(key=lambda item: item.date, reverse=True)
    if choices:
        lines = [f"（{item.date}，单次选择）{item.detail}" for item in choices[:5]]
        answer = (
            "目前还没有已确认的长期偏好。记录里只有这些具体选择，"
            "它们说明当时怎么选，不足以单独判断长期偏好：\n"
            + "\n".join(f"- {line}" for line in lines)
        )
        return {
            "type": "preference_answer",
            "answer": answer,
            "evidence": lines,
            "answer_mode": "retrieval_only",
            "llm_used": False,
        }
    return {
        "type": "preference_answer",
        "answer": "目前没有已确认的相关偏好记录，我还不能判断你长期喜欢哪一种。",
        "evidence": [],
        "answer_mode": "retrieval_only",
        "llm_used": False,
    }


def record_interaction(
    pipeline: MemoryPipeline,
    interaction,
    raw_text: str,
    day: str,
) -> dict:
    """Persist a high-confidence interaction through the normal pipeline."""
    memory_type = {
        "event": "event",
        "mixed_request": "event",
        "decision": "decision",
        "stable_preference": "long_term",
        "long_term_plan": "long_term",
    }.get(interaction.kind, "event")
    action_override = {
        "decision": "选择",
        "stable_preference": "偏好",
        "long_term_plan": "约定",
    }.get(interaction.kind, "记录")
    confidence = 0.95 if interaction.kind == "stable_preference" else 0.9
    report = pipeline.ingest([{
        "date": day,
        "text": raw_text,
        "memory_text": interaction.record_text,
        "source": "用户对话",
        "kind": interaction.kind,
        "parent_id": interaction.parent_id,
        "memory_type": memory_type,
        "confidence": confidence,
        "preference_key": interaction.preference_key,
        "action_override": action_override,
        "topic_override": interaction.topic,
        "entity_override": interaction.subject,
    }])
    memory_ids = report.get("memory_ids", [])
    episode = None
    if memory_ids and interaction.kind in (
        "event", "mixed_request", "decision", "long_term_plan"
    ):
        episode = {
            "memory_id": memory_ids[0],
            "topic": interaction.topic,
            "subject": interaction.subject,
            "date": day,
        }
    effects = []
    if interaction.kind in ("event", "mixed_request"):
        effects.append("已记录本次事实")
    elif interaction.kind == "decision":
        effects.append("已记录本次选择")
    elif interaction.kind == "stable_preference":
        effects.append("已记录明确偏好")
    elif interaction.kind == "long_term_plan":
        effects.append("已更新长期计划")
    return {
        **report,
        "effects": effects,
        "episode": episode,
        "memory_ids": memory_ids,
    }


def _handle_contextual_request(query: str, plan, excluded_ids: set[str]) -> dict:
    """Answer a request while keeping its recordable clause as a separate event."""
    store = get_pipeline().store
    relevant = []
    for item in store.memories:
        if (
            item.id in excluded_ids
            or item.status != "active"
            or item.topic != plan.topic
            or item.memory_type not in ("long_term", "preference_evidence", "decision")
        ):
            continue
        score = _context_memory_relevance(store, item, plan)
        if score:
            relevant.append((
                score,
                item.date,
                f"（{item.date}，可信度 {item.confidence:.2f}）{item.detail}"
            ))
    relevant.sort(reverse=True)
    evidence = [line for _score, _day, line in relevant[:6]]
    if is_enabled():
        try:
            context = "\n".join(evidence) if evidence else "暂无稳定的相关偏好记录。"
            answer = chat([
                {
                    "role": "system",
                    "content": (
                        "你是个人记忆助手。直接回答用户当前的建议或选择问题。"
                        "给出的个人记忆只能作为参考，不能把一次临时想法说成稳定偏好；"
                        "有可靠偏好时给出个性化建议，没有时给出稳妥建议，必要时只问一个关键问题。"
                        "不要解释后台如何记录，也不要使用“候选记忆、写入、检索”等系统术语。"
                    ),
                },
                {
                    "role": "user",
                    "content": f"相关个人记录：\n{context}\n\n当前问题：{query}",
                },
            ], temperature=0.45).strip()
            if answer:
                return {
                    "type": "contextual_answer",
                    "answer": answer,
                    "evidence": evidence,
                    "answer_mode": "retrieval_llm",
                    "llm_used": True,
                }
        except Exception:
            pass

    if evidence:
        return {
            "type": "contextual_answer",
            "answer": f"可以参考你之前的选择：{evidence[0]}。这次优先沿用相近的选项会更稳妥。",
            "evidence": evidence,
            "answer_mode": "retrieval_llm",
            "llm_used": False,
        }
    if plan.topic == "饮食":
        return {
            "type": "contextual_answer",
            "answer": "第一次选的话可以从微辣开始，既有味道也比较稳妥；如果你平时很能吃辣，再选中辣。",
            "evidence": [],
            "answer_mode": "retrieval_llm",
            "llm_used": False,
        }
    return {
        "type": "contextual_answer",
        "answer": "我还没有足够的相关偏好记录。先按你今天的实际时间和状态选一个容易执行的方案更合适。",
        "evidence": [],
        "answer_mode": "retrieval_llm",
        "llm_used": False,
    }

def _rewrite_followup(question: str) -> str:
    """多轮追问：结合上一轮把省略主语/时间的问题补全。LLM 优先，规则兜底原样返回。"""
    history = SESSION["history"]
    if not history:
        return question
    needs_rewrite = question.startswith(FOLLOW_UP_MARKERS)
    if not needs_rewrite:
        return question
    prev = history[-1]
    if is_enabled():
        try:
            out = chat_json([
                {"role": "system", "content": (
                    "把用户的追问改写成一个独立、完整的问题（补全省略的主语和时间）。"
                    '只输出 JSON：{"question": "..."}。上一轮主题之外的完整问题则原样返回。')},
                {"role": "user", "content":
                    f"上一轮问题：{prev['q']}\n"
                    f"上一轮回答：{prev['a'][:200]}\n"
                    f"本轮追问：{question}"},
            ])
            q = str(out.get("question", "")).strip()
            if q:
                return q
        except Exception:
            pass
    return question


def _handle_forget(question: str) -> dict:
    """对话式主动遗忘：优先处理当前上下文，再保守匹配记忆库。"""
    from memagent.memory import consolidate
    from memagent.retrieve import tokenize

    p = get_pipeline()
    target = re.sub(r"(帮我|请|我想)?(忘掉|忘记|忘了|别记|删掉|删除|去掉)(了|掉|吧|关于|之前)?", " ", question)
    target = re.sub(r"(这件事|这事儿|的?事|的记忆|的?记录|它|这个|那个)", " ", target).strip(" ，。？?")
    if not target:
        return {"type": "forget", "answer": "请说明需要淡忘的具体内容，例如「忘掉跑步计划」。"}

    generic = {"计划", "事情", "记忆", "记录", "内容", "相关"}
    target_tokens = {t for t in tokenize(target) if len(t) >= 2 and t not in generic}
    if not target_tokens:
        target_tokens = {t for t in tokenize(target) if len(t) >= 2}
    target_frame = analyze_semantics(target, is_query=True)
    has_semantic_target = bool(
        target_frame.entities or target_frame.facets or target_frame.values
    )

    def match_score(memory) -> tuple[int, int, int]:
        memory_tokens = set(tokenize(memory.topic + " " + memory.detail))
        overlap = target_tokens & memory_tokens
        exact = sum(1 for token in target_tokens if token in memory.detail)
        semantic = 0
        if has_semantic_target:
            semantic = semantic_match_score(
                target_frame,
                _memory_semantic_frame(p.store, memory),
                strict_facet=False,
                require_value=bool(target_frame.values),
            )
        return semantic, len(overlap), exact

    matched = []
    recent = _latest_episode()
    if recent and recent.get("memory_id"):
        memory = p.store.memory_by_id(recent["memory_id"])
        if memory and memory.status != "discarded":
            semantic, overlap, exact = match_score(memory)
            if semantic or overlap:
                matched.append((memory, semantic, overlap, exact))
                for child in p.store.memories:
                    if child.parent_id == memory.id and child.status != "discarded":
                        child_score = match_score(child)
                        if child_score[0] or child_score[1]:
                            matched.append((child, *child_score))

    if not matched:
        for memory in p.store.memories:
            if memory.status == "discarded":
                continue
            semantic, overlap, exact = match_score(memory)
            if semantic or overlap:
                matched.append((memory, semantic, overlap, exact))
    if not matched:
        already_forgotten = []
        for memory in p.store.memories:
            if memory.status != "discarded":
                continue
            semantic, overlap, exact = match_score(memory)
            if semantic or overlap:
                already_forgotten.append((memory, semantic, overlap, exact))
        if already_forgotten:
            return {
                "type": "forget",
                "count": 0,
                "answer": f"与「{target}」直接相关的记忆已经在淡忘层，无需重复处理。",
            }
        return {"type": "forget", "answer": f"没有找到与「{target}」直接相关的活跃记忆。"}

    matched.sort(key=lambda x: (x[1], x[3], x[2], x[0].date), reverse=True)
    memories = list({
        memory.id: memory
        for memory, _semantic, _overlap, _exact in matched
    }.values())
    for m in memories:
        m.status = "discarded"

    forgotten_ids = {m.id for m in memories}
    p.store.conclusions = [
        conclusion
        for conclusion in p.store.conclusions
        if not (forgotten_ids & set(conclusion.support))
    ]
    consolidate(p.store)
    p.store.save()
    append_event(LOG_PATH, "forget_manual", role="reconciler",
                 memory_ids=[m.id for m in memories], count=len(memories), query=target)
    latest = memories[0]
    ordered = sorted(memories, key=lambda item: item.date, reverse=True)
    shown = ordered[:3]
    lines = [f"- （{item.date}）{item.detail}" for item in shown]
    if len(ordered) > len(shown):
        lines.append(f"- 另有 {len(ordered) - len(shown)} 条相关记忆")
    heading = (
        "已将这条记忆移入淡忘层："
        if len(memories) == 1
        else f"已将以下 {len(memories)} 条记忆移入淡忘层："
    )
    return {
        "type": "forget",
        "count": len(memories),
        "frozen": {"date": latest.date, "detail": latest.detail},
        "answer": heading + "\n" + "\n".join(lines) + "\n可在记忆库中恢复。",
    }
def _review_window(question: str, today: date) -> tuple[date, date]:
    spans = calendar_spans(question, today)
    if spans:
        start, end = spans[0]
        return date.fromisoformat(start), date.fromisoformat(end)
    return today - timedelta(days=6), today


def _handle_review(question: str) -> dict:
    """阶段回顾：取时间窗内的记录，LLM 生成回顾报告（规则兜底分主题罗列）。"""
    p = get_pipeline()
    start, end = _review_window(question, date.today())
    start_s, end_s = start.isoformat(), end.isoformat()
    ents = [e for e in p.store.entries if start_s <= e.date <= end_s]
    mems = [x for x in p.store.memories if x.status == "active" and start_s <= x.date <= end_s]
    lines = [f"{e.date} {e.text}" for e in sorted(ents, key=lambda x: x.date)]

    if not lines:
        return {
            "type": "review",
            "answer": f"{start_s} 到 {end_s} 这段时间我们没聊过什么，我手里空空的。\n"
                      "不过正好——这几天的趣事你现在讲给我，我就替你记住，直接说「帮我记一下……」就行。",
            "llm_used": False,
        }

    if is_enabled():
        try:
            text = chat([
                {"role": "system", "content": (
                    "你是用户的长期记忆助手，也是用户的老朋友。根据这段时间的原始记录，用轻松自然的口吻写一段 150 字以内的回顾，"
                    "分点讲清：做了什么、坚持了什么、改了什么主意。不要编造记录里没有的事。"
                    "语气像老朋友聊天，结尾给一句自然的收尾或一个跟进建议（比如想细看哪天的事，告诉我日期就行）；"
                    "禁止用「就这些」「没有更多可回顾的内容」这类冷冰冰的话收尾。")},
                {"role": "user", "content":
                    f"时间范围 {start_s} 至 {end_s}，记录：\n" + "\n".join(lines[:60])},
            ], temperature=0.4)
            return {
                "type": "review",
                "window": [start_s, end_s],
                "answer": text.strip(),
                "llm_used": True,
            }
        except Exception:
            pass

    by_topic: dict[str, list] = {}
    for x in mems:
        by_topic.setdefault(x.topic, []).append(x)
    parts = [f"**{start_s} ~ {end_s} 回顾**（{len(ents)} 条记录）"]
    for t, xs in by_topic.items():
        parts.append(f"- {t}：{ '；'.join(x.detail for x in xs[:3]) }")
    parts.append("想细看哪一天的事，或者让我把这段写成周报，说一声就行。")
    return {
        "type": "review",
        "window": [start_s, end_s],
        "answer": "\n".join(parts),
        "llm_used": False,
    }


@app.post("/api/chat")
def chat_api(req: ChatRequest):
    q_raw = req.question.strip()
    if not q_raw:
        return {"type": "normal", "answer": "想问什么？", "evidence": []}

    if not is_record_command(q_raw):
        interaction = classify_interaction(q_raw, previous_episode=_latest_episode())
        if interaction.kind == "none" and is_enabled():
            online_interaction = classify_interaction_online(
                q_raw,
                previous_episode=_latest_episode(),
            )
            if online_interaction.kind != "none":
                interaction = online_interaction
        if interaction.kind != "none":
            if interaction.kind == "preference_query":
                result = _handle_preference_question(
                    interaction.topic,
                    interaction.query_text or q_raw,
                )
                result["scope"] = "memory"
                result["routing"] = {
                    "intent": interaction.kind,
                    "method": "interaction_policy",
                    "answer_mode": "retrieval_only",
                    "requires_llm": False,
                }
                SESSION["history"].append({
                    "q": q_raw,
                    "a": result["answer"][:200],
                    "scope": "memory",
                    "answer_mode": "retrieval_only",
                })
                SESSION["history"] = SESSION["history"][-8:]
                return result

            if interaction.kind in (
                "inferred_preference_candidate",
                "long_term_plan_candidate",
            ):
                if interaction.method == "llm":
                    eligible = True
                    reason = interaction.reason or "模型判断这可能是有长期价值的个人信息；尚未确认。"
                    confidence = interaction.confidence or 0.82
                else:
                    eligible, reason, confidence = _review_implicit_candidate(
                        interaction.record_text
                    )
                if eligible:
                    candidate = _add_memory_candidate(
                        interaction.record_text,
                        day=date.today().isoformat(),
                        source="用户对话（待确认）",
                        reason=reason,
                        confidence=confidence,
                        memory_type="long_term",
                        action_override="偏好" if interaction.kind == "inferred_preference_candidate" else "",
                        preference_key=interaction.preference_key,
                        topic=interaction.topic,
                        subject=interaction.subject,
                    )
                    answer = (
                        "这可能是一项长期偏好，已放入待确认。"
                        if interaction.kind == "inferred_preference_candidate"
                        else "这项长期安排已放入待确认。"
                    )
                    result = {
                        "type": "memory_candidate",
                        "answer": answer,
                        "candidate_id": candidate["id"],
                        "evidence": [],
                        "scope": "memory",
                        "routing": {"intent": interaction.kind, "method": "interaction_policy"},
                        "answer_mode": "local_action",
                        "llm_used": False,
                    }
                    SESSION["history"].append({
                        "q": q_raw, "a": answer[:200], "scope": "memory",
                        "answer_mode": "local_action",
                    })
                    SESSION["history"] = SESSION["history"][-8:]
                    return result

            if sensitive_reason(interaction.record_text):
                if not interaction.needs_answer:
                    return {
                        "type": "record_blocked",
                        "answer": "这段内容包含敏感信息，我不会把它保存到个人记忆中。",
                        "evidence": [],
                        "scope": "general",
                    }
                recorded = {"memory_ids": [], "entry_ids": []}
            else:
                recorded = record_interaction(
                    get_pipeline(),
                    interaction,
                    q_raw,
                    date.today().isoformat(),
                )
                append_event(
                    LOG_PATH,
                    "interaction_recorded",
                    role="recorder",
                    kind=interaction.kind,
                    detail=interaction.record_text,
                    memory_ids=recorded.get("memory_ids", []),
                )

            if interaction.needs_answer:
                result = _handle_contextual_request(
                    interaction.query_text or q_raw,
                    interaction,
                    set(recorded.get("memory_ids", [])),
                )
            else:
                result = {
                    "type": interaction.kind,
                    "answer": interaction.reply,
                    "evidence": [],
                    "answer_mode": "local_action",
                    "llm_used": False,
                }

            episode = None
            memory_ids = recorded.get("memory_ids", [])
            if memory_ids and interaction.kind in ("event", "mixed_request", "decision"):
                episode = {
                    "memory_id": memory_ids[0],
                    "topic": interaction.topic,
                    "subject": interaction.subject,
                    "date": date.today().isoformat(),
                }

            # Repeated choices create a proposal, not an automatic preference.
            preference_candidate_pending = False
            if interaction.kind == "decision" and interaction.subject:
                store = get_pipeline().store
                matched_episodes = set()
                for item in store.memories:
                    if (
                        item.memory_type != "decision"
                        or item.preference_key != interaction.preference_key
                        or item.topic != interaction.topic
                    ):
                        continue
                    parent = store.memory_by_id(item.parent_id)
                    if parent and interaction.subject in parent.detail:
                        matched_episodes.add(item.parent_id)
                if len(matched_episodes) >= 2:
                    candidate = _add_memory_candidate(
                        f"我可能偏好{interaction.subject}时选择{interaction.preference_key}",
                        day=date.today().isoformat(),
                        source="多次相同选择（待确认）",
                        reason="多个独立场景中选择相同；这只是偏好线索，需由你确认。",
                        confidence=0.72,
                        memory_type="long_term",
                        action_override="偏好",
                        preference_key=interaction.preference_key,
                        topic=interaction.topic,
                        subject=interaction.subject,
                    )
                    result["candidate_id"] = candidate["id"]
                    preference_candidate_pending = True

            result["memory_effects"] = recorded.get("effects") or (
                [
                    "已记录本次事实" if interaction.kind in ("event", "mixed_request") else
                    "已记录本次选择" if interaction.kind == "decision" else
                    "已记录明确表达的偏好"
                ] if memory_ids else []
            )
            if preference_candidate_pending:
                result["memory_effects"].append("偏好待确认")
            result["scope"] = "memory"
            result["routing"] = {
                "intent": interaction.kind,
                "method": "interaction_policy",
                "reason": "区分本轮事实、当前请求与待确认的长期偏好",
                "answer_mode": (
                    "retrieval_llm" if interaction.needs_answer else "local_action"
                ),
                "requires_llm": bool(interaction.needs_answer),
            }
            result.setdefault(
                "answer_mode",
                "retrieval_llm" if interaction.needs_answer else "local_action",
            )
            result.setdefault("llm_used", False)
            SESSION["history"].append({
                "q": q_raw,
                "a": result.get("answer", "")[:200],
                "scope": "memory",
                "episode": episode,
                "answer_mode": result["answer_mode"],
            })
            SESSION["history"] = SESSION["history"][-8:]
            return result

    previous_scope = SESSION["history"][-1].get("scope") if SESSION["history"] else None
    previous_answer_mode = (
        SESSION["history"][-1].get("answer_mode") if SESSION["history"] else None
    )
    decision = classify_query(
        q_raw,
        previous_scope=previous_scope,
        previous_answer_mode=previous_answer_mode,
        memory_probe=lambda: bool(
            get_pipeline().agent.retriever.search(q_raw, topk=3, use_llm=False)
        ),
    )
    intent = decision["intent"]

    candidate_match, candidate_reason, candidate_confidence = (False, "", 0.0)
    if intent not in ("record", "forget", "review", "utility") and not is_record_command(q_raw):
        candidate_match, candidate_reason, candidate_confidence = _review_implicit_candidate(q_raw)
    if candidate_match:
        candidate = _add_memory_candidate(
            q_raw,
            day=date.today().isoformat(),
            source="用户对话（待确认）",
            reason=candidate_reason,
            confidence=candidate_confidence,
        )
        result = {
            "type": "memory_candidate",
            "answer": f"已放入待确认：{candidate['detail']}。",
            "candidate_id": candidate["id"],
            "evidence": [],
            "scope": "memory",
            "routing": decision,
            "answer_mode": "local_action",
            "llm_used": False,
        }
        SESSION["history"].append({
            "q": q_raw,
            "a": result["answer"][:200],
            "scope": "memory",
            "answer_mode": "local_action",
        })
        SESSION["history"] = SESSION["history"][-8:]
        return result

    if intent == "record":
        text = extract_record_content(q_raw)
        if not text:
            result = {"type": "record", "answer": "请说明需要记录的具体内容。", "evidence": []}
        elif sensitive_reason(text):
            result = {
                "type": "record",
                "answer": "这段内容疑似包含密码、密钥、联系方式或证件/支付信息，已阻止写入长期记忆。",
                "evidence": [],
                "write_blocked": True,
            }
        else:
            p = get_pipeline()
            conflicts_before = len(p.store.conflicts)
            report = p.ingest([{"date": date.today().isoformat(),
                                "text": text, "source": "对话记录"}])
            extra = ""
            if len(p.store.conflicts) > conflicts_before:
                extra = "\n检测到与旧记录存在更新关系，当前状态按最新记录处理。"
            result = {
                "type": "record",
                "answer": f"已记录：{text}\n提取 {report['memories_extracted']} 条记忆。{extra}",
                "evidence": [],
            }
    elif intent == "utility":
        result = _handle_utility_query(q_raw)
    elif intent == "general":
        result = _handle_general_query(q_raw)
    else:
        p = get_pipeline()
        if intent == "forget":
            result = _handle_forget(q_raw)
        elif intent == "review":
            result = _handle_review(q_raw)
        else:
            q = _rewrite_followup(q_raw) if decision["method"] == "followup" else q_raw
            result = p.ask(q, answer_mode=decision["answer_mode"])
            result["type"] = "normal"
            result["rewritten"] = q if q != q_raw else None

    scope = "memory" if intent in ("record", "forget", "review", "memory") else "general"
    result["scope"] = scope
    result["routing"] = decision
    result.setdefault("answer_mode", decision["answer_mode"])
    result.setdefault("llm_used", False)
    SESSION["history"].append({
        "q": q_raw,
        "a": result.get("answer", "")[:200],
        "scope": scope,
        "answer_mode": result["answer_mode"],
    })
    SESSION["history"] = SESSION["history"][-8:]
    return result


# ==================== 记录 / 遗忘恢复 ====================
def _llm_contradiction_check(p: MemoryPipeline, new_mem, day: str):
    """偏好/习惯类反转（如浅烘→深烘）不在核心冲突配对规则内，网页层用 LLM 补审。
    判定冲突时与核心行为一致：记冲突、旧记忆降为被取代，留痕进日志。"""
    from memagent.store import Conflict
    candidates = [m for m in p.store.memories
                  if m.status == "active" and m.id != new_mem.id
                  and m.topic == new_mem.topic
                  and m.action in ("偏好", "更改", "开始", "放弃", "约定")][-6:]
    if not candidates:
        return None
    try:
        out = chat_json([
            {"role": "system", "content": (
                "判断新记的事情是否推翻或改变了某条旧记忆所述的状态（口味变了、计划变了、约定变了都算；"
                "只是补充细节或完全无关则不算）。只输出 JSON："
                '{"conflict": true/false, "index": 命中的旧记忆序号(从0开始), "reason": "一句话理由"}')},
            {"role": "user", "content":
                "新记的事：" + new_mem.detail + "\n旧记忆：\n" +
                "\n".join(f"{i}. {m.date} {m.detail}" for i, m in enumerate(candidates))},
        ])
    except Exception:
        return None
    if not out.get("conflict"):
        return None
    try:
        old = candidates[int(out.get("index", 0))]
    except (ValueError, IndexError):
        return None
    c = Conflict(id="c" + old.id[1:] + new_mem.id[1:], topic=new_mem.topic,
                 action=new_mem.action, older_id=old.id, newer_id=new_mem.id,
                 resolved=True, winner_id=new_mem.id)
    p.store.conflicts.append(c)
    old.status = "superseded"
    p.store.save()
    append_event(LOG_PATH, "reconciler_decision", role="reconciler",
                 pair=f"{old.id}|{new_mem.id}", conflict=True,
                 reason=str(out.get("reason", "")), method="llm-web")
    return {"old": {"date": old.date, "detail": old.detail},
            "new": {"date": new_mem.date, "detail": new_mem.detail}}


@app.post("/api/import")
def import_memory(req: ImportRequest):
    text = req.text.strip()
    if not text:
        return {"ok": False, "message": "内容为空。"}
    if sensitive_reason(text):
        return {"ok": False, "message": "内容疑似包含敏感信息，已阻止写入长期记忆。"}
    p = get_pipeline()
    day = req.date or date.today().isoformat()
    conflicts_before = {c.id for c in p.store.conflicts}
    report = p.ingest([{"date": day, "text": text, "source": req.source}])
    stats = p.store.stats()

    contradicted = None
    new_conflicts = [c for c in p.store.conflicts if c.id not in conflicts_before]
    if new_conflicts:
        c = new_conflicts[-1]
        older = p.store.memory_by_id(c.older_id)
        newer = p.store.memory_by_id(c.newer_id)
        if older and newer:
            contradicted = {"old": {"date": older.date, "detail": older.detail},
                            "new": {"date": newer.date, "detail": newer.detail}}
    else:
        # 核心规则没配到的（偏好反转等）交给 LLM 补审
        new_mem = next((m for m in reversed(p.store.memories)
                        if m.date == day and (m.detail in text or text.rstrip("。") in m.detail or m.detail.rstrip("。") in text)), None)
        if new_mem is None:
            new_mem = p.store.memories[-1] if p.store.memories else None
        if new_mem is not None and is_enabled():
            contradicted = _llm_contradiction_check(p, new_mem, day)

    return {"ok": True,
            "message": f"记好了（提取 {report['memories_extracted']} 条记忆）。",
            "contradicted": contradicted, "stats": stats}


def _document_chunks(filename: str, sections: list[tuple[str, str]], limit: int = 36) -> list[str]:
    """按文档结构和段落切片，保留文件名及页/表/幻灯片位置。"""
    chunks = []
    for label, section_text in sections:
        current = ""
        for paragraph in re.split(r"\n+", section_text):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if current and len(current) + len(paragraph) + 1 > 2400:
                chunks.append(f"【文档《{filename}》·{label}】\n{current}")
                current = ""
                if len(chunks) >= limit:
                    return chunks
            while len(paragraph) > 2400:
                chunks.append(f"【文档《{filename}》·{label}】\n{paragraph[:2400]}")
                paragraph = paragraph[2400:]
                if len(chunks) >= limit:
                    return chunks
            current = f"{current}\n{paragraph}".strip()
        if current:
            chunks.append(f"【文档《{filename}》·{label}】\n{current}")
        if len(chunks) >= limit:
            return chunks[:limit]
    return chunks


def _store_parsed_document(parsed, data: bytes, record_day: str) -> dict:
    DOCUMENT_DIR.mkdir(parents=True, exist_ok=True)
    stored_name = f"{uuid4().hex[:8]}_{parsed.filename}"
    stored_path = DOCUMENT_DIR / stored_name
    stored_path.write_bytes(data)
    document = {
        "id": "d" + uuid4().hex[:10],
        "filename": parsed.filename,
        "stored_file": stored_name,
        "kind": parsed.kind,
        "date": record_day,
        "char_count": parsed.char_count,
        "warning": parsed.warning,
        "sections": [{"label": label, "text": text} for label, text in parsed.sections],
        "trust": "untrusted_attachment",
    }
    try:
        catalog = _load_json_list(DOCUMENT_CATALOG_PATH)
        catalog.append(document)
        _save_json_list(DOCUMENT_CATALOG_PATH, catalog)
    except Exception:
        stored_path.unlink(missing_ok=True)
        raise
    append_event(
        LOG_PATH,
        "document_import",
        role="recorder",
        filename=parsed.filename,
        kind=parsed.kind,
        sections=len(parsed.sections),
        chunks=0,
        chars=parsed.char_count,
        destination="document_library",
    )
    return {
        "ok": True,
        "message": f"已保存到资料库并解析为 {len(parsed.sections)} 个内容区，未写入个人记忆。",
        "document_id": document["id"],
        "filename": parsed.filename,
        "kind": parsed.kind,
        "sections": len(parsed.sections),
        "chunks": 0,
        "chars": parsed.char_count,
        "warning": parsed.warning,
        "stored_file": stored_name,
        "storage_kind": "document",
    }


@app.post("/api/document")
async def import_document(file: UploadFile = File(...), day: str = Form("")):
    data = await file.read()
    try:
        parsed = parse_document(file.filename or "uploaded", data)
    except DocumentParseError as exc:
        return {"ok": False, "message": str(exc)}
    record_day = day or date.today().isoformat()
    return _store_parsed_document(parsed, data, record_day)


@app.get("/api/documents")
def list_documents():
    items = _load_json_list(DOCUMENT_CATALOG_PATH)
    summaries = [{
        key: document.get(key)
        for key in ("id", "filename", "kind", "date", "char_count", "warning")
    } for document in reversed(items)]
    return {"items": summaries}


@app.get("/api/documents/{document_id}")
def document_detail(document_id: str):
    document = next(
        (item for item in _load_json_list(DOCUMENT_CATALOG_PATH)
         if item.get("id") == document_id),
        None,
    )
    if not document:
        return {"found": False}
    return {"found": True, **document}


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: str):
    items = _load_json_list(DOCUMENT_CATALOG_PATH)
    document = next((item for item in items if item.get("id") == document_id), None)
    if not document:
        return {"ok": False, "message": "资料不存在或已删除。"}
    retained = [item for item in items if item.get("id") != document_id]
    _save_json_list(DOCUMENT_CATALOG_PATH, retained)
    stored_name = Path(document.get("stored_file", "")).name
    if stored_name and (DOCUMENT_DIR / stored_name).resolve().parent == DOCUMENT_DIR.resolve():
        (DOCUMENT_DIR / stored_name).unlink(missing_ok=True)
    append_event(LOG_PATH, "document_deleted", role="user", document_id=document_id)
    return {"ok": True, "message": "资料及其解析内容已删除。"}


def _is_attachment_memory_request(prompt: str) -> bool:
    return is_record_command(prompt)


def _attachment_memory_note(prompt: str) -> str:
    content = extract_record_content(prompt)
    content = re.sub(
        r"^(?:这些|所有|以上|下列|这份|这个|这张|这幅|这|该)?\s*"
        r"(?:附件|图片|照片|文件|资料|图)"
        r"(?:和|以及|与|连同)?\s*",
        "",
        content,
    )
    content = content.strip(" ：:，。,.")
    generic_attachment_request = re.fullmatch(
        r"(?:的|里|中|里面|里提到的)?"
        r"(?:所有|全部|主要|核心)?"
        r"(?:附件|图片|照片|文件|资料|图|内容|信息|观点|结论|重点|摘要|文字)?",
        content,
    )
    return "" if generic_attachment_request else content


def _explicit_attachment_note(prompt: str, *, endpoint_is_explicit: bool = False) -> str:
    """Return only a user's own statement, never the attachment-storage command."""
    value = (prompt or "").strip()
    if not value:
        return ""
    if is_record_command(value):
        return _attachment_memory_note(value)
    return value if endpoint_is_explicit else ""


def _cloud_attachment_privacy_reason(*texts: str) -> str | None:
    """Keep sensitive prompts and locally extracted attachment text off cloud models."""
    for text in texts:
        reason = sensitive_reason(text or "")
        if reason:
            return reason
    return None


def _rollback_saved_attachments(results: list[dict]) -> None:
    """Best-effort rollback for a mixed attachment batch."""
    document_ids = {item.get("document_id") for item in results if item.get("document_id")}
    photo_files = {item.get("file") for item in results if item.get("storage_kind") == "photo"}

    if document_ids:
        catalog = _load_json_list(DOCUMENT_CATALOG_PATH)
        removed = [item for item in catalog if item.get("id") in document_ids]
        _save_json_list(
            DOCUMENT_CATALOG_PATH,
            [item for item in catalog if item.get("id") not in document_ids],
        )
        for document in removed:
            stored_name = Path(document.get("stored_file", "")).name
            if stored_name:
                (DOCUMENT_DIR / stored_name).unlink(missing_ok=True)

    if photo_files:
        _save_photos([item for item in _load_photos() if item.get("file") not in photo_files])
        for filename in photo_files:
            (PHOTO_DIR / Path(filename).name).unlink(missing_ok=True)


def _store_attachment_batch(loaded: list[dict], record_day: str) -> tuple[list[dict], str]:
    """Store a validated mixed batch or restore the pre-call storage state."""
    results: list[dict] = []
    try:
        for item in loaded:
            if item["is_image"]:
                data_url = (
                    f"data:{item['content_type']};base64,"
                    + base64.b64encode(item["data"]).decode("ascii")
                )
                result = _save_photo(PhotoRequest(
                    date=record_day,
                    note="",
                    dataUrl=data_url,
                ))
            else:
                result = _store_parsed_document(item["parsed"], item["data"], record_day)
            if not result.get("ok"):
                raise RuntimeError(result.get("message") or "附件保存失败。")
            results.append(result)
    except Exception as exc:
        _rollback_saved_attachments(results)
        return [], str(exc) or "附件保存失败。"
    return results, ""


def _attachment_context(parsed_documents: list, limit: int = 24_000) -> str:
    if not parsed_documents:
        return ""
    parts = []
    per_document = max(1000, limit // len(parsed_documents))
    for parsed in parsed_documents:
        lines = []
        used = 0
        for label, text in parsed.sections:
            block = f"[{label}]\n{text}"
            remaining = per_document - used
            if remaining <= 0:
                break
            block = block[:remaining]
            lines.append(block)
            used += len(block)
        if parsed.char_count > used:
            lines.append("[该文件内容较长，已按多文件公平配额截取]")
        if lines:
            parts.append(f"《{parsed.filename}》\n" + "\n".join(lines))
    return "\n\n".join(parts)


def _attachment_fallback_answer(file_summaries: list[dict]) -> str:
    lines = []
    for item in file_summaries:
        detail = item["kind"]
        if item.get("sections"):
            detail += f"，{item['sections']} 个内容区"
        if item.get("status"):
            detail += f"，{item['status']}"
        lines.append(f"- 《{item['name']}》（{detail}）")
    listing = "\n".join(lines)
    return (
        f"已读取 {len(file_summaries)} 个附件：\n{listing}\n"
        "当前未完成语义分析；附件本身已保留，可重试或补充更具体的问题。"
    )


def _model_image_data_url(data: bytes, content_type: str) -> str:
    """Downscale uploads before sending them to the online vision model."""
    try:
        from PIL import Image, ImageOps

        with Image.open(io.BytesIO(data)) as source:
            image = ImageOps.exif_transpose(source)
            image.thumbnail((1600, 1600))
            if image.mode in ("RGBA", "LA"):
                background = Image.new("RGB", image.size, "white")
                alpha = image.getchannel("A")
                background.paste(image.convert("RGB"), mask=alpha)
                image = background
            elif image.mode != "RGB":
                image = image.convert("RGB")
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=85, optimize=True)
            encoded = base64.b64encode(output.getvalue()).decode("ascii")
            return "data:image/jpeg;base64," + encoded
    except Exception:
        encoded = base64.b64encode(data).decode("ascii")
        return f"data:{content_type};base64," + encoded


@app.post("/api/attachments")
async def handle_attachments(
    files: list[UploadFile] = File(...),
    prompt: str = Form(""),
    day: str = Form(""),
):
    if not files:
        return {"ok": False, "message": "没有收到附件。"}
    if len(files) > MAX_ATTACHMENT_FILES:
        return {"ok": False, "message": f"一次最多发送 {MAX_ATTACHMENT_FILES} 个附件。"}

    loaded = []
    total_bytes = 0
    image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
    for upload in files:
        data = await upload.read()
        total_bytes += len(data)
        if total_bytes > MAX_ATTACHMENT_TOTAL_BYTES:
            return {"ok": False, "message": "附件总大小超过 80MB，请分批发送。"}
        filename = Path(upload.filename or "uploaded").name
        ext = Path(filename).suffix.lower()
        is_image = (upload.content_type or "").startswith("image/") or ext in image_extensions
        if is_image and len(data) > MAX_IMAGE_BYTES:
            return {"ok": False, "message": f"图片《{filename}》超过 15MB，请压缩后发送。"}
        if not is_image and len(data) > MAX_FILE_BYTES:
            return {"ok": False, "message": f"文件《{filename}》超过 50MB，请压缩或拆分后发送。"}
        loaded.append({
            "filename": filename,
            "content_type": upload.content_type or "application/octet-stream",
            "data": data,
            "is_image": is_image,
        })

    clean_prompt = prompt.strip()
    record_day = day or date.today().isoformat()
    for item in loaded:
        if item["is_image"]:
            continue
        try:
            item["parsed"] = parse_document(item["filename"], item["data"])
        except DocumentParseError as exc:
            return {"ok": False, "message": f"《{item['filename']}》：{exc}"}

    if _is_attachment_memory_request(clean_prompt):
        memory_note = _attachment_memory_note(clean_prompt)
        if memory_note:
            reason = sensitive_reason(memory_note)
            if reason:
                return {
                    "ok": False,
                    "message": "说明文字疑似包含敏感信息，附件和长期记忆均未保存。",
                }
        saved_results, storage_error = _store_attachment_batch(loaded, record_day)
        if storage_error:
            return {"ok": False, "message": storage_error}
        stored = [item["filename"] for item in loaded]
        try:
            if memory_note:
                get_pipeline().ingest([{
                    "date": record_day,
                    "text": memory_note,
                    "source": "用户明确记录",
                }])
        except Exception:
            _rollback_saved_attachments(saved_results)
            return {"ok": False, "message": "记忆写入失败，本批附件已回滚，请重试。"}
        return {
            "ok": True,
            "mode": "saved",
            "answer": (
                f"已将 {len(stored)} 个附件保存到资料库，文件正文没有写入个人记忆。"
                + ("明确说明已写入个人记忆。" if memory_note else "")
            ),
            "files": stored,
        }

    parsed_documents = []
    image_ocr_documents = []
    image_data_urls = []
    file_summaries = []
    for item in loaded:
        if item["is_image"]:
            image_data_urls.append(_model_image_data_url(
                item["data"],
                item["content_type"],
            ))
            try:
                item["ocr_parsed"] = parse_document(item["filename"], item["data"])
                image_ocr_documents.append(item["ocr_parsed"])
                image_status = "已通过 OCR 提取文字"
                image_sections = len(item["ocr_parsed"].sections)
            except DocumentParseError:
                item["ocr_parsed"] = None
                image_status = "已接收，未识别出文字"
                image_sections = 0
            summary = {
                "name": item["filename"],
                "kind": "图片",
                "status": image_status,
            }
            if image_sections:
                summary["sections"] = image_sections
            item["summary"] = summary
            file_summaries.append(summary)
            continue
        parsed = item["parsed"]
        parsed_documents.append(parsed)
        summary = {
            "name": item["filename"],
            "kind": parsed.kind,
            "sections": len(parsed.sections),
            "status": "已提取文字",
        }
        item["summary"] = summary
        file_summaries.append(summary)

    question = clean_prompt or "请分别说明这些附件的主要内容，并概括它们之间的关系。"
    document_context = _attachment_context(parsed_documents)
    image_ocr_context = _attachment_context(image_ocr_documents, limit=12_000)
    privacy_reason = _cloud_attachment_privacy_reason(
        clean_prompt,
        document_context,
        image_ocr_context,
    )
    context_prompt = question
    if document_context:
        context_prompt += (
            "\n\n以下是同时上传的文档提取内容。回答时结合图片与文档，不要虚构未出现的信息：\n"
            + document_context
        )

    if image_data_urls and vision_is_enabled() and not privacy_reason:
        try:
            answer = vision_chat(
                context_prompt,
                image_data_urls,
                temperature=0.2,
                system_prompt=attachment_prompt_is_untrusted()
                + " 仅按用户问题分析图片，不推断图片中未显示的信息。",
            ).strip()
            if answer:
                return {
                    "ok": True,
                    "mode": "analysis",
                    "answer": answer,
                    "files": file_summaries,
                    "visual_understanding": True,
                }
        except Exception:
            pass

    if not image_data_urls and document_context and is_enabled() and not privacy_reason:
        try:
            answer = chat([
                {"role": "system", "content": (
                    attachment_prompt_is_untrusted() +
                    "根据用户上传的文档内容回答问题。需要综合多个文件时明确指出共同点和差异，"
                    "不得虚构文档中没有的信息。")},
                {"role": "user", "content": context_prompt},
            ], temperature=0.2).strip()
            if answer:
                return {
                    "ok": True,
                    "mode": "analysis",
                    "answer": answer,
                    "files": file_summaries,
                    "visual_understanding": False,
                }
        except Exception:
            pass

    fallback_documents = list(parsed_documents) + image_ocr_documents
    fallback_text = _attachment_context(fallback_documents, limit=30_000).strip()
    if fallback_text and is_enabled() and not privacy_reason:
        try:
            answer = chat([
                {"role": "system", "content": (
                    attachment_prompt_is_untrusted() +
                    "只能根据附件中提取到的文字回答，不能声称看到了未被识别的图片内容。"
                    "证据不足时明确说明。")},
                {"role": "user", "content": f"问题：{question}\n\n附件文字：\n{fallback_text[:30000]}"},
            ], temperature=0.2).strip()
            if answer:
                return {
                    "ok": True,
                    "mode": "analysis",
                    "answer": answer,
                    "files": file_summaries,
                    "visual_understanding": False,
                }
        except Exception:
            pass

    fallback_answer = _attachment_fallback_answer(file_summaries)
    if privacy_reason:
        fallback_answer = "检测到敏感信息，本次附件未发送到在线模型，仅完成本地读取。\n" + fallback_answer
    return {
        "ok": True,
        "mode": "analysis",
        "answer": fallback_answer,
        "files": file_summaries,
        "visual_understanding": False,
        "privacy_local_only": bool(privacy_reason),
    }


@app.post("/api/restore")
def restore_memory(detail: str):
    """恢复一条被淡忘的记忆（遗忘≠删除的闭环）。"""
    p = get_pipeline()
    for m in reversed(p.store.memories):
        if m.status == "discarded" and m.detail == detail:
            m.status = "active"
            p.store.save()
            return {"ok": True, "message": "想起来了，已恢复。"}
    return {"ok": False, "message": "没找到这条记忆。"}


# ==================== 日历视图 ====================

@app.get("/api/memories")
def memories(q: str = "", topic: str = "", status: str = "active", limit: int = 300):
    """记忆库列表：支持关键词、主题、状态筛选（status=all 表示全部）。"""
    store = get_pipeline().store
    q = q.strip()
    items = []
    for m in store.memories:
        if status != "all" and m.status != status:
            continue
        if topic and m.topic != topic:
            continue
        if q and q not in m.detail and q not in m.topic:
            continue
        entry = next((e for e in store.entries if e.id == m.evidence), None)
        items.append({"id": m.id, "date": m.date, "topic": m.topic, "action": m.action,
                      "detail": m.detail, "status": m.status,
                      "importance": round(m.importance, 2),
                      "memory_type": m.memory_type,
                      "confidence": round(m.confidence, 2),
                      "source": entry.source if entry else ""})
    items.sort(key=lambda x: x["date"], reverse=True)
    return {"memories": items[:limit], "total": len(items)}


@app.get("/api/memory/{mid}")
def memory_detail(mid: str):
    """单条记忆详情：来源证据、取代链（改过几次）、同主题相关记忆。"""
    store = get_pipeline().store
    m = store.memory_by_id(mid)
    if not m:
        return {"found": False}
    entry = next((e for e in store.entries if e.id == m.evidence), None)

    # 取代链：沿着冲突关系向更早追溯（你以为的现状，其实改过几次）
    chain, cur, guard = [], m, 0
    while cur is not None and guard < 10:
        conf_prev = next((c for c in store.conflicts if c.newer_id == cur.id), None)
        chain.append({"id": cur.id, "date": cur.date, "detail": cur.detail,
                      "status": cur.status})
        cur = store.memory_by_id(conf_prev.older_id) if conf_prev else None
        guard += 1

    related = [{"id": x.id, "date": x.date, "detail": x.detail, "status": x.status}
               for x in store.memories
               if x.topic == m.topic and x.id != m.id][-6:]
    related.sort(key=lambda x: x["date"], reverse=True)
    conclusion = store.conclusion_for(m.topic)
    return {
        "found": True,
        "memory": {"id": m.id, "date": m.date, "topic": m.topic, "action": m.action,
                   "detail": m.detail, "status": m.status,
                   "importance": round(m.importance, 2), "frequency": m.frequency,
                   "memory_type": m.memory_type, "confidence": round(m.confidence, 2),
                   "parent_id": m.parent_id, "preference_key": m.preference_key},
        "source": {"date": entry.date, "text": entry.text, "source": entry.source} if entry else None,
        "chain": chain,
        "related": related,
        "conclusion": conclusion.text if conclusion else "",
    }


@app.post("/api/forget/{mid}")
def forget_by_id(mid: str):
    p = get_pipeline()
    m = p.store.memory_by_id(mid)
    if not m:
        return {"ok": False, "message": "没找到这条记忆。"}
    m.status = "discarded"
    p.store.save()
    append_event(LOG_PATH, "forget_manual", role="reconciler",
                 memory_id=m.id, detail=m.detail, query="(记忆库操作)")
    return {"ok": True, "message": "已淡忘。它还在归档层，需要时可恢复。"}


@app.post("/api/restore/{mid}")
def restore_by_id(mid: str):
    p = get_pipeline()
    m = p.store.memory_by_id(mid)
    if not m:
        return {"ok": False, "message": "没找到这条记忆。"}
    m.status = "active"
    p.store.save()
    return {"ok": True, "message": "已恢复为「还记着」。"}


@app.get("/api/events")
def events(limit: int = 40):
    """决策日志（最新在前）：角色、事件、摘要——流程页的实时Feed。"""
    if not LOG_PATH.exists():
        return {"events": []}
    rows = []
    for line in LOG_PATH.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    rows = rows[-limit:][::-1]
    out = []
    for e in rows:
        desc = ""
        for k in ("reason", "detail", "question", "text", "action"):
            if e.get(k):
                desc = str(e[k])[:60]
                break
        if not desc and e.get("event") == "query_round":
            desc = f"路由 {e.get('route', '')} · 命中 {e.get('hits', '')}"
        out.append({"time": str(e.get("timestamp", ""))[11:19],
                    "event": e.get("event", ""), "role": e.get("role", ""),
                    "desc": desc})
    return {"events": out}


@app.get("/api/overview")
def overview():
    """总览：今日与近期记忆流、最近的安排、记忆健康、活跃主题、最近问过的话。"""
    p = get_pipeline()
    store = p.store
    today = date.today()
    today_s = today.isoformat()
    week_start = (today - timedelta(days=6)).isoformat()
    st = store.stats()
    by = st["memories_by_status"]

    today_entries = [{"date": e.date, "text": e.text, "source": e.source}
                     for e in store.entries if e.date == today_s]
    actives = sorted([m for m in store.memories if m.status == "active"],
                     key=lambda m: m.date, reverse=True)
    recent = [{"id": m.id, "date": m.date, "topic": m.topic, "detail": m.detail}
              for m in actives if m.date >= week_start]
    recent_fallback = False
    if len(recent) < 6:                  # 近七天不足时，用更早的还记着的事补齐
        picked = {r["id"] for r in recent}
        for m in actives:
            if m.id not in picked:
                recent.append({"id": m.id, "date": m.date, "topic": m.topic, "detail": m.detail})
                picked.add(m.id)
            if len(recent) >= 6:
                break
        recent_fallback = True

    plans = [{"date": m.date, "detail": m.detail, "action": m.action}
             for m in store.memories
             if m.status == "active" and m.action in ("约定", "更改", "开始")]
    plans.sort(key=lambda x: x["date"], reverse=True)

    topics = Counter(m.topic for m in store.memories if m.status == "active")
    photos = _load_photos()
    return {
        "today": today_s,
        "today_count": len(today_entries),
        "today_entries": today_entries[:5],
        "week_count": sum(1 for e in store.entries if e.date >= week_start),
        "recent": recent,
        "recent_fallback": recent_fallback,
        "plans": plans[:4],
        "health": {
            "entries": st["entries"], "active": by.get("active", 0),
            "archived": by.get("superseded", 0) + by.get("compressed", 0),
            "forgotten": by.get("discarded", 0),
            "conflicts": st["conflicts"], "conflicts_resolved": st["conflicts_resolved"],
            "conclusions": st["conclusions"], "photos": len(photos),
        },
        "topics": topics.most_common(5),
        "recent_questions": [h["q"] for h in SESSION["history"]][-3:],
    }


@app.get("/api/calendar")
def calendar(year: int, month: int):
    """某月每一天的记忆密度（活跃记忆 + 原始记录），供日历格子打点。"""
    p = get_pipeline()
    prefix = f"{year:04d}-{month:02d}"
    days: dict[str, dict] = {}
    for e in p.store.entries:
        if e.date.startswith(prefix):
            d = days.setdefault(e.date, {"count": 0, "topics": [], "texts": []})
            d["count"] += 1
            d["texts"].append(e.text)
    for m in p.store.memories:
        if m.status == "active" and m.date.startswith(prefix):
            d = days.setdefault(m.date, {"count": 0, "topics": [], "texts": []})
            d["topics"].append(m.topic)
            d["texts"].append(m.detail)
    return {"year": year, "month": month, "days": days}


@app.get("/api/day")
def day_detail(date: str):
    """某一天的原始记录 + 活跃记忆（点日历格子时用）。"""
    p = get_pipeline()
    entries = [{"text": e.text, "source": e.source}
               for e in p.store.entries if e.date == date]
    mems = [{"topic": m.topic, "detail": m.detail, "action": m.action}
            for m in p.store.memories if m.date == date and m.status == "active"]
    return {"date": date, "entries": entries, "memories": mems}


# ==================== 记忆画像 ====================

@app.get("/api/profile")
def profile():
    """主题分布 / 记忆状态 / 近 8 周记录节奏。"""
    p = get_pipeline()
    active = [m for m in p.store.memories if m.status == "active"]
    topic_count = Counter(m.topic for m in active)
    status_count = Counter(m.status for m in p.store.memories)
    weeks = Counter()
    for e in p.store.entries:
        try:
            d = datetime.strptime(e.date, "%Y-%m-%d").date()
        except ValueError:
            continue
        monday = d - timedelta(days=d.weekday())
        weeks[monday.isoformat()] += 1
    trend = sorted(weeks.items())[-8:]
    return {
        "topics": topic_count.most_common(),
        "statuses": status_count.most_common(),
        "trend": [{"week": w, "count": c} for w, c in trend],
        "entries": len(p.store.entries),
        "conclusions": len(p.store.conclusions),
    }


# ==================== 约定提醒 / 导出 ====================

@app.get("/api/upcoming")
def upcoming():
    """安排类记忆（约定/更改/开始，时间上最近的 5 条）——像日历提醒一样浮出。"""
    p = get_pipeline()
    plans = [m for m in p.store.memories
             if m.status == "active" and m.action in ("约定", "更改", "开始")]
    plans.sort(key=lambda m: (m.date, m.action != "约定"), reverse=True)
    return {"items": [{"date": m.date, "detail": m.detail,
                       "topic": m.topic, "action": m.action}
                      for m in plans[:5]]}


@app.get("/api/export")
def export():
    """把整本记忆库下载为 JSON——数据始终属于用户。"""
    return FileResponse(STORE_PATH, media_type="application/json",
                        filename="my_memory.json")


# ==================== 多模态：照片记忆 ====================
# 照片以 dataUrl(base64) 上传，落盘到 data/photos/，清单存 web_photos.json。
# 同时把照片说明作为一条记录灌入记忆管线（提取记忆、参与检索与遗忘）。

PHOTO_DIR = ROOT / "data" / "photos"
PHOTOS_PATH = ROOT / "data" / "web_photos.json"
PHOTO_DIR.mkdir(parents=True, exist_ok=True)


def _load_photos() -> list:
    if PHOTOS_PATH.exists():
        return json.loads(PHOTOS_PATH.read_text(encoding="utf-8"))
    return []


def _save_photos(items: list) -> None:
    PHOTOS_PATH.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")


class PhotoRequest(BaseModel):
    date: str
    note: str = ""
    dataUrl: str


def _decode_image_data_url(data_url: str) -> tuple[bytes, str, str]:
    header, _, b64 = data_url.partition(",")
    if not b64 or not header.lower().startswith("data:image/"):
        raise DocumentParseError("图片数据无效。")
    try:
        data = base64.b64decode(b64, validate=True)
    except (ValueError, TypeError) as exc:
        raise DocumentParseError("图片数据无法解码。") from exc
    if len(data) > MAX_IMAGE_BYTES:
        raise DocumentParseError("图片超过 15MB，请压缩后上传。")
    content_type = header.split(";", 1)[0].removeprefix("data:").lower()
    ext = next(
        (value for marker, value in (
            ("png", ".png"), ("webp", ".webp"), ("gif", ".gif"),
            ("bmp", ".bmp"), ("tiff", ".tiff"), ("jpeg", ".jpg"), ("jpg", ".jpg"),
        ) if marker in content_type),
        ".jpg",
    )
    return data, content_type, ext




def _ocr_data_url(data_url: str) -> tuple[str, str]:
    try:
        data, _content_type, ext = _decode_image_data_url(data_url)
    except DocumentParseError as exc:
        return "", str(exc)
    temporary = PHOTO_DIR / f".analysis_{uuid4().hex[:10]}{ext}"
    try:
        temporary.write_bytes(data)
        sections = ocr_path(temporary, max_pages=1)
        text = "\n".join(value for _label, value in sections).strip()[:6000]
        return text, ""
    except DocumentParseError as exc:
        return "", str(exc)
    finally:
        temporary.unlink(missing_ok=True)


@app.post("/api/image")
def handle_image(req: PhotoRequest):
    prompt = req.note.strip()
    if is_record_command(prompt):
        memory_note = _explicit_attachment_note(prompt)
        if memory_note and sensitive_reason(memory_note):
            return {
                "ok": False,
                "mode": "saved",
                "message": "图片未保存：说明疑似包含敏感信息，已阻止写入长期记忆。",
            }
        saved = add_photo(PhotoRequest(date=req.date, note=memory_note, dataUrl=req.dataUrl))
        if not saved.get("ok"):
            return saved
        saved["mode"] = "saved"
        saved["message"] = (
            "图片已保存到照片资料区，图片 OCR 不会自动写入个人记忆。"
            + ("明确说明已写入长期记忆。" if memory_note else "")
        )
        return saved

    question = prompt or "请客观描述这张图片的主要内容；如果含有文字，同时提取关键文字。"
    try:
        image_data, content_type, _extension = _decode_image_data_url(req.dataUrl)
    except DocumentParseError as exc:
        return {"ok": False, "mode": "analysis", "message": str(exc)}
    ocr_text, ocr_error = _ocr_data_url(req.dataUrl)
    privacy_reason = _cloud_attachment_privacy_reason(prompt, ocr_text)
    vision_error = ""
    if vision_is_enabled() and not privacy_reason:
        try:
            model_image = _model_image_data_url(image_data, content_type)
            answer = vision_chat(
                question,
                model_image,
                temperature=0.2,
                system_prompt=attachment_prompt_is_untrusted()
                + " 仅按用户问题分析图片，不推断图片中未显示的信息。",
            ).strip()
            if answer:
                return {
                    "ok": True,
                    "mode": "analysis",
                    "answer": answer,
                    "visual_understanding": True,
                }
        except Exception as exc:
            vision_error = str(exc)

    if ocr_text and is_enabled() and not privacy_reason:
        try:
            answer = chat([
                {"role": "system", "content": (
                    "你只能根据 OCR 文字回答图片问题，不能声称看到了图片中的物体、颜色或场景。"
                    "证据不足时明确说明限制。")},
                {"role": "user", "content": f"用户问题：{question}\n\nOCR 文字：\n{ocr_text}"},
            ], temperature=0.2).strip()
            if answer:
                return {
                    "ok": True,
                    "mode": "analysis",
                    "answer": answer,
                    "ocr_text": ocr_text,
                    "visual_understanding": False,
                }
        except Exception:
            pass

    if ocr_text:
        return {
            "ok": True,
            "mode": "analysis",
            "answer": (
                ("检测到敏感信息，本次图片未发送到在线模型。\n" if privacy_reason else "")
                + f"当前只能识别图片中的文字，结果如下：\n{ocr_text}"
            ),
            "ocr_text": ocr_text,
            "visual_understanding": False,
            "privacy_local_only": bool(privacy_reason),
        }
    limitation = "当前设备未识别出清晰文字，且在线图片理解暂不可用。"
    if not vision_is_enabled():
        limitation += " 离线模式只提供 OCR，不具备完整视觉理解。"
    return {
        "ok": False,
        "mode": "analysis",
        "message": limitation,
        "ocr_error": ocr_error,
        "vision_error": vision_error,
    }

def _save_photo(req: PhotoRequest) -> dict:
    note = req.note.strip()
    try:
        image_data, _content_type, extension = _decode_image_data_url(req.dataUrl)
    except DocumentParseError as exc:
        return {"ok": False, "message": str(exc)}
    PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{req.date}_{uuid4().hex[:8]}{extension}"
    file_path = PHOTO_DIR / name
    file_path.write_bytes(image_data)

    ocr_text = ""
    ocr_error = ""
    try:
        sections = ocr_path(file_path, max_pages=1)
        ocr_text = "\n".join(text for _label, text in sections).strip()[:6000]
    except DocumentParseError as exc:
        ocr_error = str(exc)

    display_note = note or "上传了一张图片。"
    try:
        items = _load_photos()
        items.append({
            "date": req.date,
            "file": name,
            "note": display_note,
            "ocr_text": ocr_text,
        })
        _save_photos(items)
    except Exception:
        file_path.unlink(missing_ok=True)
        raise
    append_event(
        LOG_PATH,
        "photo_import",
        role="recorder",
        filename=name,
        ocr_chars=len(ocr_text),
        has_note=bool(note),
    )

    message = (
        "图片已保存；识别文字仅用于资料预览，不会自动写入个人记忆。"
        if ocr_text else "图片已保存，未识别出清晰文字。"
    )
    return {
        "ok": True,
        "message": message,
        "file": name,
        "ocr_text": ocr_text,
        "ocr_error": ocr_error,
        "storage_kind": "photo",
    }


@app.post("/api/photo")
def add_photo(req: PhotoRequest):
    memory_note = _explicit_attachment_note(req.note, endpoint_is_explicit=True)
    if memory_note and sensitive_reason(memory_note):
        return {"ok": False, "message": "图片说明疑似包含敏感信息，已阻止保存和长期记忆写入。"}
    saved = _save_photo(PhotoRequest(date=req.date, note=memory_note, dataUrl=req.dataUrl))
    if not saved.get("ok"):
        return saved
    if memory_note:
        try:
            get_pipeline().ingest([{
                "date": req.date,
                "text": memory_note,
                "source": "用户图片说明",
            }])
        except Exception:
            _rollback_saved_attachments([saved])
            return {"ok": False, "message": "记忆写入失败，图片保存已回滚，请重试。"}
        saved["message"] = "图片已保存，明确填写的说明已写入长期记忆；OCR 文字未写入。"
    return saved


@app.get("/api/photos")
def list_photos(date: str | None = None):
    items = _load_photos()
    if date:
        items = [x for x in items if x["date"] == date]
    return {"photos": sorted(items, key=lambda x: x["date"], reverse=True)}


app.mount("/photos", StaticFiles(directory=str(PHOTO_DIR)), name="photos")


# ==================== 重建 ====================

@app.post("/api/rebuild")
def rebuild():
    global _pipeline
    if STORE_PATH.exists():
        STORE_PATH.unlink()
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    _pipeline = None
    SESSION["history"] = []
    p = get_pipeline()
    return {"ok": True, "message": "记忆已恢复到初始状态。", "stats": p.store.stats()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()








