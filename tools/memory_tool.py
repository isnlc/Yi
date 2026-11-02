"""Tool 2: write records and maintain the long-term memory store."""

from memagent.agents import MemoryAgent


def ingest_and_maintain(agent: MemoryAgent, records: list[dict]) -> dict:
    """Ingest records, detect conflicts, consolidate, and apply retention."""
    extracted = 0
    conflicts = 0
    entry_ids = []
    memory_ids = []
    for record in records:
        entry, items, found = agent.ingest(
            record["date"],
            record["text"],
            record.get("source", "jsonl"),
            memory_text=record.get("memory_text"),
            kind=record.get("kind", "record"),
            parent_id=record.get("parent_id", ""),
            memory_type=record.get("memory_type", "long_term"),
            confidence=float(record.get("confidence", 1.0)),
            preference_key=record.get("preference_key", ""),
            action_override=record.get("action_override", ""),
            topic_override=record.get("topic_override", ""),
            entity_override=record.get("entity_override", ""),
        )
        entry_ids.append(entry.id)
        memory_ids.extend(item.id for item in items)
        extracted += len(items)
        conflicts += len(found)
    report = agent.maintain()
    return {
        "records": len(records),
        "memories_extracted": extracted,
        "conflicts_found_during_ingest": conflicts,
        "entry_ids": entry_ids,
        "memory_ids": memory_ids,
        "conclusions": report.get("conclusions", []),
        "retention_decisions": report.get("retention", []),
    }
