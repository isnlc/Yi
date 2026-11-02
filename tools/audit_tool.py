"""Tool 4: append structured pipeline events to a JSONL audit log."""

import json
from datetime import datetime, timezone
from pathlib import Path


def append_event(path: str | Path, event: str, **payload) -> None:
    """Append one JSON object. The log is intentionally append-only."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **payload,
    }
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
