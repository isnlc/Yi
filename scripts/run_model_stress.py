"""Generate diverse memory questions with the configured LLM and score them locally.

The model expands language coverage. Ground truth remains tied to explicit fact IDs,
so retrieval and routing are judged deterministically instead of by model opinion.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memagent.agents import MemoryAgent
from memagent.answer import compose
from memagent.llm import chat_json, is_enabled
from memagent.router import classify_query
from memagent.semantics import analyze_query, enrich_memory_semantics
from memagent.store import MemoryStore


FACTS_PATH = ROOT / "eval" / "stress_facts.json"
CASES_PATH = ROOT / "eval" / "generated_semantic_stress.json"
FIXED_CASES_PATH = ROOT / "eval" / "fixed_semantic_stress.json"
RESULTS_PATH = ROOT / "eval" / "results" / "generated_semantic_stress_results.json"
FIXED_RESULTS_PATH = ROOT / "eval" / "results" / "fixed_semantic_stress_results.json"

CATEGORIES = (
    ("日常直接询问与自然改写", 12, 4),
    ("省略主语但询问个人既定习惯或配置", 12, 4),
    ("否定、反问、错误预设与纠正", 12, 4),
    ("冷门器材、养护、手作与小众兴趣", 12, 4),
    ("旧状态、当前状态与前后变化", 12, 4),
    ("跨对象干扰与相似属性隔离", 12, 4),
    ("需要个人记忆的建议与不需要记忆的通用建议", 12, 4),
    ("世界知识、解释、计算、写作等纯通用问题", 6, 8),
)


@contextmanager
def offline_router():
    with patch("memagent.router.llm.is_enabled", return_value=False):
        yield


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _generation_prompt(
    facts: list[dict],
    category: str,
    memory_count: int,
    general_count: int,
    existing: list[str],
) -> list[dict]:
    catalog = [
        {
            "id": item["id"],
            "date": item["date"],
            "status": item["status"],
            "detail": item["detail"],
        }
        for item in facts
    ]
    return [
        {
            "role": "system",
            "content": (
                "你是个人记忆系统的压力测试设计者。根据给定事实目录生成真实用户可能提出的问题。"
                "问题既要自然，也要包含容易混淆的改写；不能在问题中暴露事实编号。"
                "输出 JSON 对象：{\"cases\":[...]}。每个 case 必须包含："
                "id、query、expected_intent、query_kind、target_fact_ids、"
                "forbidden_fact_ids、difficulty、reason。"
                "expected_intent 只能是 memory 或 general。"
                "query_kind 只能是 current、history、evolution、profile、advice、general。"
                "memory 表示必须依赖目录中的用户个人事实才能回答；general 表示不依赖个人事实。"
                "memory 问题的 target_fact_ids 必须引用目录中的编号，通常为一条；"
                "若询问历史变化可引用两条。forbidden_fact_ids 填最容易被错误混入的事实。"
                "general 问题的两个 fact id 数组必须为空。"
                "通用问题应覆盖与目录领域相邻但不应读取个人记忆的知识、解释、计算、写作或一般建议。"
                "不要把所有问题都写成第一人称，也不要靠机械替换生成近义重复。"
                "不要生成要求医疗诊断、投资承诺或违法操作的问题。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"本批类别：{category}\n"
                f"必须生成 memory={memory_count} 条，general={general_count} 条，数量必须完整。\n"
                f"已生成问题，禁止重复：{json.dumps(existing[-120:], ensure_ascii=False)}\n"
                f"事实目录：{json.dumps(catalog, ensure_ascii=False)}"
            ),
        },
    ]


def _normalize_cases(raw_cases: list[dict], facts: list[dict], seen: set[str]) -> list[dict]:
    valid_ids = {item["id"] for item in facts}
    valid_kinds = {"current", "history", "evolution", "profile", "advice", "general"}
    normalized = []
    local_seen = set(seen)
    for case in raw_cases:
        if not isinstance(case, dict):
            continue
        query = str(case.get("query", "")).strip()
        intent = str(case.get("expected_intent", "")).strip()
        kind = str(case.get("query_kind", "")).strip()
        targets = [value for value in case.get("target_fact_ids", []) if value in valid_ids]
        forbidden = [value for value in case.get("forbidden_fact_ids", []) if value in valid_ids]
        if (
            not query
            or query in local_seen
            or intent not in ("memory", "general")
            or kind not in valid_kinds
            or (intent == "memory" and not targets)
        ):
            continue
        if intent == "general":
            targets = []
            forbidden = []
            kind = "general"
        case["query"] = query
        case["query_kind"] = kind
        case["target_fact_ids"] = list(dict.fromkeys(targets))
        case["forbidden_fact_ids"] = list(dict.fromkeys(forbidden))
        local_seen.add(query)
        normalized.append(case)
    return normalized


def _audit_case_labels(facts: list[dict], cases: list[dict]) -> list[dict]:
    """Second-pass audit prevents malformed model-generated ground truth."""
    catalog = [
        {
            "id": item["id"],
            "date": item["date"],
            "status": item["status"],
            "detail": item["detail"],
        }
        for item in facts
    ]
    by_id = {case["id"]: case for case in cases}
    for start in range(0, len(cases), 16):
        batch = cases[start:start + 16]
        payload = chat_json([
            {
                "role": "system",
                "content": (
                    "审核个人记忆系统测试题的标准答案。根据事实目录纠正每题标签。"
                    "若问题必须依赖用户事实则为 memory，并把回答所需的全部事实编号放入"
                    "target_fact_ids；例如同时问两种交通方式，就必须列两条。"
                    "forbidden_fact_ids 只能放不应进入答案但容易混入的事实，不能与 target 重叠。"
                    "不依赖个人事实的知识、解释、一般建议、写作或计算为 general，两个编号数组为空。"
                    "保留 id 和 query。输出 JSON：{\"cases\":[...]}, 每项含 id、expected_intent、"
                    "query_kind、target_fact_ids、forbidden_fact_ids、audit_reason。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"事实目录：{json.dumps(catalog, ensure_ascii=False)}\n"
                    f"待审核题目：{json.dumps(batch, ensure_ascii=False)}"
                ),
            },
        ])
        for audited in payload.get("cases", []):
            case = by_id.get(str(audited.get("id", "")))
            if not case:
                continue
            intent = str(audited.get("expected_intent", ""))
            targets = [
                value for value in audited.get("target_fact_ids", [])
                if value in {item["id"] for item in facts}
            ]
            forbidden = [
                value for value in audited.get("forbidden_fact_ids", [])
                if value in {item["id"] for item in facts} and value not in targets
            ]
            if intent == "memory" and targets:
                case["expected_intent"] = "memory"
                case["target_fact_ids"] = list(dict.fromkeys(targets))
                case["forbidden_fact_ids"] = list(dict.fromkeys(forbidden))
            elif intent == "general":
                case["expected_intent"] = "general"
                case["query_kind"] = "general"
                case["target_fact_ids"] = []
                case["forbidden_fact_ids"] = []
            kind = str(audited.get("query_kind", ""))
            if kind in {"current", "history", "evolution", "profile", "advice", "general"}:
                case["query_kind"] = kind
            case["audit_reason"] = str(audited.get("audit_reason", ""))[:200]
    return cases


def generate_cases(facts: list[dict]) -> list[dict]:
    if not is_enabled():
        raise RuntimeError("Online LLM is not enabled; configure the project .env first.")
    cases: list[dict] = []
    seen = set()
    for batch_index, (category, wanted_memory, wanted_general) in enumerate(CATEGORIES, start=1):
        accepted: list[dict] = []
        for _attempt in range(5):
            have_memory = sum(case["expected_intent"] == "memory" for case in accepted)
            have_general = sum(case["expected_intent"] == "general" for case in accepted)
            missing_memory = max(0, wanted_memory - have_memory)
            missing_general = max(0, wanted_general - have_general)
            if not missing_memory and not missing_general:
                break
            payload = chat_json(_generation_prompt(
                facts,
                category,
                min(missing_memory, 8),
                min(missing_general, 6),
                [case["query"] for case in cases + accepted],
            ))
            batch = _normalize_cases(payload.get("cases", []), facts, seen)
            for case in batch:
                if (
                    case["expected_intent"] == "memory"
                    and sum(item["expected_intent"] == "memory" for item in accepted) >= wanted_memory
                ):
                    continue
                if (
                    case["expected_intent"] == "general"
                    and sum(item["expected_intent"] == "general" for item in accepted) >= wanted_general
                ):
                    continue
                seen.add(case["query"])
                accepted.append(case)
        for index, case in enumerate(accepted, start=1):
            case["id"] = f"b{batch_index:02d}q{index:03d}"
            case["category"] = category
            cases.append(case)
    target_memory = sum(item[1] for item in CATEGORIES)
    target_general = sum(item[2] for item in CATEGORIES)
    for attempt in range(8):
        have_memory = sum(case["expected_intent"] == "memory" for case in cases)
        have_general = sum(case["expected_intent"] == "general" for case in cases)
        missing_memory = max(0, target_memory - have_memory)
        missing_general = max(0, target_general - have_general)
        if not missing_memory and not missing_general:
            break
        payload = chat_json(_generation_prompt(
            facts,
            "综合盲测补足：优先生成此前未覆盖的表达和领域",
            min(missing_memory, 8),
            min(missing_general, 6),
            [case["query"] for case in cases],
        ))
        batch = _normalize_cases(payload.get("cases", []), facts, seen)
        for case in batch:
            have_memory = sum(item["expected_intent"] == "memory" for item in cases)
            have_general = sum(item["expected_intent"] == "general" for item in cases)
            if case["expected_intent"] == "memory" and have_memory >= target_memory:
                continue
            if case["expected_intent"] == "general" and have_general >= target_general:
                continue
            seen.add(case["query"])
            case["id"] = f"b99q{len(cases) + 1:03d}"
            case["category"] = "综合盲测补足"
            cases.append(case)
    return _audit_case_labels(facts, cases)


def extend_cases(facts: list[dict], cases: list[dict]) -> list[dict]:
    """Top up an audited blind set without regenerating earlier questions."""
    target_memory = sum(item[1] for item in CATEGORIES)
    target_general = sum(item[2] for item in CATEGORIES)
    seen = {case["query"] for case in cases}
    for attempt in range(12):
        have_memory = sum(case["expected_intent"] == "memory" for case in cases)
        have_general = sum(case["expected_intent"] == "general" for case in cases)
        missing_memory = max(0, target_memory - have_memory)
        missing_general = max(0, target_general - have_general)
        if not missing_memory and not missing_general:
            break
        payload = chat_json(_generation_prompt(
            facts,
            "补充全新盲测题，不能与已有题目重复",
            min(missing_memory, 6),
            min(missing_general, 6),
            list(seen),
        ))
        new_cases = _normalize_cases(payload.get("cases", []), facts, seen)
        accepted = []
        for case in new_cases:
            if case["query"] in seen:
                continue
            case["id"] = f"x{len(cases) + len(accepted) + 1:03d}"
            case["category"] = "补充盲测"
            seen.add(case["query"])
            accepted.append(case)
        if not accepted:
            continue
        cases.extend(_audit_case_labels(facts, accepted))
    return cases


def build_agent(facts: list[dict], path: Path):
    store = MemoryStore(str(path))
    fact_to_memory = {}
    for fact in facts:
        memory = store.add_memory(
            date=fact["date"],
            topic=fact["topic"],
            action=fact["action"],
            detail=fact["detail"],
            memory_type=fact["memory_type"],
            status=fact["status"],
        )
        fact_to_memory[fact["id"]] = memory.id
    enrich_memory_semantics(store)
    return MemoryAgent(store, reference_date="2026-11-02"), fact_to_memory


def evaluate_cases(facts: list[dict], cases: list[dict]) -> dict:
    with tempfile.TemporaryDirectory() as directory:
        agent, fact_to_memory = build_agent(facts, Path(directory) / "stress_store.json")
        memory_to_fact = {memory_id: fact_id for fact_id, memory_id in fact_to_memory.items()}
        details = []
        for case in cases:
            query = case["query"]
            memory_probe = lambda q=query: bool(agent.retriever.search(q, topk=3))
            with offline_router():
                offline = classify_query(query, memory_probe=memory_probe)
            analyze_query.cache_clear()
            online = classify_query(query, memory_probe=memory_probe)
            target_ids = set(case["target_fact_ids"])
            forbidden_ids = set(case["forbidden_fact_ids"])
            hit_fact_ids = []
            answer_fact_ids = []
            answer_text = ""
            if case["expected_intent"] == "memory":
                route = agent.retriever.route(query)
                hits = agent.retriever.search(query, topk=10, route=route, use_llm=True)
                hit_fact_ids = [
                    memory_to_fact[memory.id]
                    for memory, _score in hits
                    if memory.id in memory_to_fact
                ]
                answer_text = compose(
                    agent.store,
                    query,
                    hits,
                    agent.retriever,
                    display=agent._display_mode(query, route),
                    answer_mode="retrieval_only",
                )["answer"]
                answer_fact_ids = [
                    fact["id"]
                    for fact in facts
                    if fact["detail"].rstrip("。") in answer_text
                ]
            target_hit = (
                True
                if not target_ids
                else bool(target_ids & set(answer_fact_ids))
            )
            forbidden_hit = bool(forbidden_ids & set(answer_fact_ids))
            detail = {
                **case,
                "offline_intent": offline["intent"],
                "offline_method": offline["method"],
                "online_intent": online["intent"],
                "online_method": online["method"],
                "hit_fact_ids": hit_fact_ids,
                "answer_fact_ids": answer_fact_ids,
                "answer": answer_text,
                "offline_route_ok": offline["intent"] == case["expected_intent"],
                "online_route_ok": online["intent"] == case["expected_intent"],
                "target_hit": target_hit,
                "forbidden_hit": forbidden_hit,
            }
            detail["passed"] = (
                detail["online_route_ok"]
                and (
                    case["expected_intent"] == "general"
                    or (target_hit and not forbidden_hit)
                )
            )
            details.append(detail)

    def rate(field: str) -> float:
        return round(
            100 * sum(bool(item[field]) for item in details) / max(len(details), 1),
            1,
        )

    memory_cases = [item for item in details if item["expected_intent"] == "memory"]
    result = {
        "model": os.environ.get("MEMAGENT_MODEL", ""),
        "cases": len(details),
        "memory_cases": len(memory_cases),
        "general_cases": len(details) - len(memory_cases),
        "offline_route_accuracy": rate("offline_route_ok"),
        "online_route_accuracy": rate("online_route_ok"),
        "overall_pass_rate": rate("passed"),
        "memory_target_recall": round(
            100 * sum(item["target_hit"] for item in memory_cases) / max(len(memory_cases), 1),
            1,
        ),
        "memory_forbidden_clean_rate": round(
            100 * sum(not item["forbidden_hit"] for item in memory_cases) / max(len(memory_cases), 1),
            1,
        ),
        "failures": [item for item in details if not item["passed"]],
        "details": details,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--fixed", action="store_true")
    parser.add_argument("--extend", action="store_true")
    args = parser.parse_args()
    facts = _read_json(FACTS_PATH)
    if args.fixed:
        cases = _read_json(FIXED_CASES_PATH)
    elif args.extend:
        cases = extend_cases(facts, _read_json(CASES_PATH))
        _write_json(CASES_PATH, cases)
    elif args.generate or not CASES_PATH.exists():
        cases = generate_cases(facts)
        _write_json(CASES_PATH, cases)
    else:
        cases = _read_json(CASES_PATH)
    result = evaluate_cases(facts, cases)
    _write_json(FIXED_RESULTS_PATH if args.fixed else RESULTS_PATH, result)
    summary = {key: value for key, value in result.items() if key not in ("failures", "details")}
    summary["failure_count"] = len(result["failures"])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for failure in result["failures"][:20]:
        print(json.dumps({
            "query": failure["query"],
            "expected": failure["expected_intent"],
            "online": failure["online_intent"],
            "targets": failure["target_fact_ids"],
            "hits": failure["hit_fact_ids"],
            "forbidden": failure["forbidden_fact_ids"],
            "reason": failure.get("reason", ""),
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
