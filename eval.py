"""Reproducible course-project evaluation, including abnormal inputs."""

import json
import os
from pathlib import Path

# This benchmark evaluates the deterministic memory pipeline. Generative-model
# wording is tested separately and must not change the reproducible score.
os.environ["MEMAGENT_LLM"] = "0"

from memagent import config
from memagent.evaluate import load_jsonl, score_question
from pipeline import MemoryPipeline
from tools.data_tool import load_records


ABNORMAL_CASES = [
    {
        "id": "a01",
        "question": "",
        "expected": "empty_question_is_rejected",
    },
    {
        "id": "a02",
        "question": "请检索不存在的项目编号 XQZ-999。",
        "expected": "no_evidence",
    },
    {
        "id": "a03",
        "question": "请告诉我银河系外某颗行星今天的天气。",
        "expected": "no_evidence",
    },
    {
        "id": "a04",
        "question": "2026-99-99 这一天我做了什么？",
        "expected": "no_evidence",
    },
]


def score_abnormal(case: dict, result: dict) -> float:
    if case["expected"] == "empty_question_is_rejected":
        return 1.0 if not result.get("evidence") and "问题为空" in result["answer"] else 0.0
    return 1.0 if not result.get("evidence") else 0.0


def run_evaluation() -> dict:
    result_store = config.RESULTS_DIR / "eval_pipeline_store.json"
    result_log = Path(__file__).resolve().parent / "logs" / "run_log.jsonl"
    for path in (result_store, result_log):
        if path.exists():
            path.unlink()

    pipeline = MemoryPipeline(result_store, result_log)
    pipeline.ingest(load_records(config.DIARY_FILE))

    normal_cases = load_jsonl(config.BENCHMARK_FILE)
    normal_results = []
    for case in normal_cases:
        result = pipeline.ask(case["question"], answer_mode="retrieval_only")
        normal_results.append(
            {
                "id": case["id"],
                "type": case["type"],
                "question": case["question"],
                "score": score_question(case, result["answer"]),
                "answer": result["answer"][:200],
                "evidence_count": len(result.get("evidence", [])),
            }
        )

    abnormal_results = []
    for case in ABNORMAL_CASES:
        result = pipeline.ask(case["question"])
        abnormal_results.append(
            {
                "id": case["id"],
                "question": case["question"],
                "expected": case["expected"],
                "score": score_abnormal(case, result),
                "evidence_count": len(result.get("evidence", [])),
            }
        )

    all_results = normal_results + abnormal_results
    summary = {
        "normal_cases": len(normal_results),
        "abnormal_cases": len(abnormal_results),
        "total_cases": len(all_results),
        "normal_accuracy": round(
            sum(row["score"] for row in normal_results) / len(normal_results), 3
        ),
        "abnormal_accuracy": round(
            sum(row["score"] for row in abnormal_results) / len(abnormal_results), 3
        ),
        "overall_accuracy": round(
            sum(row["score"] for row in all_results) / len(all_results), 3
        ),
        "run_log": str(result_log),
    }

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (config.RESULTS_DIR / "eval_cases.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (config.RESULTS_DIR / "eval_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    if result_store.exists():
        result_store.unlink()
    return summary


if __name__ == "__main__":
    print(json.dumps(run_evaluation(), ensure_ascii=False, indent=2))
