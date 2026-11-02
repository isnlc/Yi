"""Tool 1: load dated records from JSONL input."""

import json
from pathlib import Path


def load_records(path: str | Path) -> list[dict]:
    """Load records with date/text fields from a JSONL file."""
    source = Path(path)
    records = []
    with source.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("date") or not row.get("text"):
                raise ValueError(f"record {line_no} needs date and text")
            records.append(
                {
                    "date": str(row["date"]),
                    "text": str(row["text"]),
                    "source": row.get("source", "jsonl"),
                }
            )
    return records
