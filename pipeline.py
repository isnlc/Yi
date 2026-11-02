"""Course-project pipeline: data loading -> memory maintenance -> retrieval."""

import argparse
import json
from datetime import date
from pathlib import Path

from memagent import config
from memagent.agents import MemoryAgent
from memagent.semantics import enrich_memory_semantics
from memagent.store import MemoryStore
from tools.audit_tool import append_event
from tools.data_tool import load_records
from tools.memory_tool import ingest_and_maintain
from tools.query_tool import query_memory


DEFAULT_STORE = config.RESULTS_DIR / "pipeline_store.json"
DEFAULT_LOG = Path(__file__).resolve().parent / "logs" / "run_log.jsonl"


class MemoryPipeline:
    """Deterministic project pipeline with optional model-assisted semantics."""

    def __init__(self, store_path=DEFAULT_STORE, log_path=DEFAULT_LOG, reference_date: str | date | None = None):
        self.store = MemoryStore(path=str(store_path))
        enrich_memory_semantics(self.store)
        self.log_path = Path(log_path)
        # 冲突治理、检索和审核事件统一写入 JSONL 日志。
        self.agent = MemoryAgent(
            self.store, with_reconciler=True,
            log=lambda event, **payload: append_event(self.log_path, event, **payload),
            reference_date=reference_date,
        )

    def ingest(self, records: list[dict]) -> dict:
        append_event(self.log_path, "ingest_start", record_count=len(records))
        report = ingest_and_maintain(self.agent, records)
        stats = self.store.stats()
        append_event(
            self.log_path,
            "maintenance_complete",
            records=report["records"],
            memories_extracted=report["memories_extracted"],
            conflicts=stats["conflicts"],
            active_memories=stats["memories_by_status"].get("active", 0),
            archive_memories=(
                stats["memories_by_status"].get("superseded", 0)
                + stats["memories_by_status"].get("compressed", 0)
            ),
            retention_decisions=stats["forget_decisions"],
        )
        return report

    def ask(self, question: str, answer_mode: str = "auto") -> dict:
        result = query_memory(self.agent, question, answer_mode=answer_mode)
        result.setdefault("answer_mode", answer_mode)
        result.setdefault("llm_used", False)
        append_event(
            self.log_path,
            "query",
            question=question,
            answer_mode=answer_mode,
            llm_used=result.get("llm_used", False),
            evidence_count=len(result.get("evidence", [])),
            used_conclusion=result.get("used_conclusion", False),
            rounds=len(result.get("trace", [])),
        )
        return result


def run_demo(store_path=DEFAULT_STORE, log_path=DEFAULT_LOG) -> dict:
    """Run one deterministic demo and return a compact summary."""
    store_path = Path(store_path)
    if store_path.exists():
        store_path.unlink()
    log_path = Path(log_path)
    if log_path.exists():
        log_path.unlink()

    pipeline = MemoryPipeline(store_path, log_path)
    records = load_records(config.DIARY_FILE)
    maintenance = pipeline.ingest(records)
    questions = [
        "我现在还在坚持跑步吗？",
        "我什么时候放弃过跑步？",
        "我的课程项目最后是什么？",
    ]
    answers = []
    for question in questions:
        result = pipeline.ask(question)
        answers.append(
            {
                "question": question,
                "answer": result["answer"],
                "evidence_count": len(result.get("evidence", [])),
            }
        )
    summary = {
        "maintenance": maintenance,
        "answers": answers,
        "store": str(store_path),
        "log": str(log_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run the long-term memory agent pipeline.")
    parser.add_argument("--data", default=str(config.DIARY_FILE))
    parser.add_argument("--store", default=str(DEFAULT_STORE))
    parser.add_argument("--log", default=str(DEFAULT_LOG))
    args = parser.parse_args()

    store_path = Path(args.store)
    log_path = Path(args.log)
    if store_path.exists():
        store_path.unlink()
    if log_path.exists():
        log_path.unlink()

    pipeline = MemoryPipeline(store_path, log_path)
    records = load_records(args.data)
    pipeline.ingest(records)
    print(json.dumps(pipeline.ask("我现在还在坚持跑步吗？"), ensure_ascii=False, indent=2))
    print(f"run log: {log_path}")


if __name__ == "__main__":
    main()
